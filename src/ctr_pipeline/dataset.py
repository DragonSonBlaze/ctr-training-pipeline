"""Versioned dataset: a time-based train, validation and test split over the daily training sets.

The dataset is a manifest, not a copy. Each split lists its days, and the rows stay in the
per-day folders that build-training-set wrote. The manifest records the row and click counts
and the sha256 of each day's manifest, so a rebuilt day gives a new version.

Input:  data/training/v1/dt=D/ plus data/training/v1/_manifests/D.json
Output: data/datasets/<version>/dataset.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from ctr_pipeline import config
from ctr_pipeline.config import (
    EVAL_BATCH_ROWS,
    FEATURE_VERSION,
    TRAIN_SAMPLE_FRACTION,
    TRAIN_SAMPLE_SEED,
    VALIDATION_SAMPLE_FRACTION,
    get_settings,
)
from ctr_pipeline.errors import MissingInputError, PipelineError

logger = logging.getLogger(__name__)

DATASET_FILE = "dataset.json"


@dataclass(frozen=True)
class DatasetResult:
    version: str
    path: Path
    manifest: dict


def split_days() -> dict[str, list[str]]:
    """The days of each split, after checking they are disjoint and in time order.

    Read at call time so a test can replace the day constants in config.
    """
    days_by_split = {
        "train": list(config.TRAIN_DAYS),
        "validation": list(config.VALIDATION_DAYS),
        "test": list(config.TEST_DAYS),
    }
    _check_split_days(days_by_split)
    return days_by_split


def _check_split_days(days_by_split: dict[str, list[str]]) -> None:
    seen: set[str] = set()
    for split in config.SPLITS:
        days = days_by_split[split]
        if not days:
            raise PipelineError(f"the {split} split has no days")
        for day in days:
            if day in seen:
                raise PipelineError(f"{day} appears more than once in the split definition")
            seen.add(day)

    ordered = [days_by_split[split] for split in config.SPLITS]
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        # ISO dates compare in calendar order as plain strings.
        if max(earlier) >= min(later):
            raise PipelineError(
                f"splits are out of time order: {max(earlier)} must come before {min(later)}"
            )


def _read_day_manifests(days: list[str]) -> dict[str, bytes]:
    """The raw bytes of each day's manifest. Names every day whose manifest is missing."""
    manifests_dir = get_settings().manifests_dir
    missing = [day for day in days if not (manifests_dir / f"{day}.json").exists()]
    if missing:
        raise MissingInputError(f"no training set manifest for: {', '.join(missing)}")
    return {day: (manifests_dir / f"{day}.json").read_bytes() for day in days}


def dataset_version(days_by_split: dict[str, list[str]]) -> str:
    """Name the dataset by its split definition and by the content of every day's manifest."""
    all_days = [day for split in config.SPLITS for day in days_by_split[split]]
    manifests = _read_day_manifests(all_days)
    digest = hashlib.sha256()
    digest.update(json.dumps(days_by_split, sort_keys=True).encode())
    for day in all_days:
        digest.update(manifests[day])
    return f"ds-{digest.hexdigest()[:12]}"


def _count_day(day: str) -> tuple[int, int]:
    """Rows and clicks in one day's training set, read from the click column only."""
    folder = get_settings().training_dir / f"dt={day}"
    if not folder.exists():
        raise MissingInputError(f"training set folder for {day} is missing: {folder}")
    table = ds.dataset(folder).to_table(columns=["click"])
    if table.num_rows == 0:
        raise MissingInputError(f"training set for {day} has no rows")
    clicks = pc.sum(table["click"]).as_py() or 0
    return table.num_rows, int(clicks)


def _split_block(days: list[str]) -> dict:
    manifests = _read_day_manifests(days)
    rows = 0
    clicks = 0
    for day in days:
        day_rows, day_clicks = _count_day(day)
        rows += day_rows
        clicks += day_clicks
    return {
        "days": days,
        "rows": rows,
        "clicks": clicks,
        "click_rate": clicks / rows,
        "day_manifest_sha256": {day: hashlib.sha256(manifests[day]).hexdigest() for day in days},
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Write to a temporary name, then rename, so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_dataset() -> DatasetResult:
    """Write the dataset manifest for the current split definition.

    The same inputs give the same version. If that version already exists, it is returned
    unchanged rather than rewritten.
    """
    days_by_split = split_days()
    version = dataset_version(days_by_split)
    path = get_settings().datasets_dir / version / DATASET_FILE
    if path.exists():
        logger.info("dataset %s already exists at %s, returning it unchanged", version, path)
        return DatasetResult(version=version, path=path, manifest=json.loads(path.read_text()))

    manifest = {
        "version": version,
        "feature_version": FEATURE_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "splits": {split: _split_block(days_by_split[split]) for split in config.SPLITS},
    }
    _write_json_atomic(path, manifest)
    logger.info("wrote dataset %s to %s", version, path)
    return DatasetResult(version=version, path=path, manifest=manifest)


def load_dataset_manifest(version: str) -> dict:
    path = get_settings().datasets_dir / version / DATASET_FILE
    if not path.exists():
        raise MissingInputError(f"no dataset named {version} at {path}")
    return json.loads(path.read_text())


def latest_version() -> str:
    """The version whose manifest has the newest created_at."""
    datasets_dir = get_settings().datasets_dir
    candidates: list[tuple[datetime, str]] = []
    if datasets_dir.exists():
        for path in datasets_dir.glob(f"*/{DATASET_FILE}"):
            manifest = json.loads(path.read_text())
            candidates.append((datetime.fromisoformat(manifest["created_at"]), manifest["version"]))
    if not candidates:
        raise MissingInputError(f"no dataset found under {datasets_dir}")
    return max(candidates)[1]


def _split_days_of(version: str, split: str) -> list[str]:
    if split not in config.SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {list(config.SPLITS)}")
    return load_dataset_manifest(version)["splits"][split]["days"]


def _default_sample_fraction(split: str) -> float | None:
    """Training and validation are sampled; test is read in full so its metrics cover every row."""
    return {
        "train": TRAIN_SAMPLE_FRACTION,
        "validation": VALIDATION_SAMPLE_FRACTION,
        "test": None,
    }[split]


def load_split(
    version: str,
    split: str,
    columns: list[str],
    sample_fraction: float | None = None,
    seed: int = TRAIN_SAMPLE_SEED,
) -> pd.DataFrame:
    """Read a split's days one at a time, keeping a uniform sample of each day's rows.

    Sampling is uniform over rows, never by label. Each day gets its own seed
    (seed plus its position in the split), so a rerun samples the same rows.
    """
    fraction = _default_sample_fraction(split) if sample_fraction is None else sample_fraction
    training_dir = get_settings().training_dir
    samples = []
    for index, day in enumerate(_split_days_of(version, split)):
        frame = pd.read_parquet(training_dir / f"dt={day}", columns=columns)
        if fraction is not None:
            frame = frame.sample(frac=fraction, random_state=seed + index)
        samples.append(frame)
    return pd.concat(samples, ignore_index=True)


def iter_split_batches(
    version: str, split: str, columns: list[str], batch_rows: int = EVAL_BATCH_ROWS
) -> Iterator[pd.DataFrame]:
    """Every row of a split's days, in batches of at most batch_rows. Nothing is sampled."""
    training_dir = get_settings().training_dir
    for day in _split_days_of(version, split):
        dataset = ds.dataset(training_dir / f"dt={day}")
        for batch in dataset.to_batches(columns=columns, batch_size=batch_rows):
            yield batch.to_pandas()
