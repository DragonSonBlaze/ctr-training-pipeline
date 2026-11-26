import json
import os

import pandas as pd
import pytest

from ctr_pipeline import config
from ctr_pipeline.config import FEATURE_VERSION, TRAIN_SAMPLE_FRACTION
from ctr_pipeline.dataset import (
    build_dataset,
    dataset_version,
    iter_split_batches,
    load_dataset_manifest,
    load_split,
    split_days,
)
from ctr_pipeline.errors import MissingInputError, PipelineError
from helpers import training_frame, write_training_day

TRAIN = ("2014-10-22", "2014-10-23")
VALIDATION = ("2014-10-24",)
TEST = ("2014-10-25",)
# Different row counts per day, so a day's rows can be told apart by their count.
DAY_ROWS = {
    "2014-10-22": 100,
    "2014-10-23": 110,
    "2014-10-24": 120,
    "2014-10-25": 130,
}


@pytest.fixture
def small_split(monkeypatch, mlflow_sandbox):
    monkeypatch.setattr(config, "TRAIN_DAYS", TRAIN)
    monkeypatch.setattr(config, "VALIDATION_DAYS", VALIDATION)
    monkeypatch.setattr(config, "TEST_DAYS", TEST)
    for day, rows in DAY_ROWS.items():
        write_training_day(mlflow_sandbox, day, training_frame(rows))
    return mlflow_sandbox


def test_split_days_are_disjoint_and_in_time_order():
    days_by_split = split_days()
    all_days = [day for days in days_by_split.values() for day in days]
    assert len(all_days) == len(set(all_days))
    assert max(days_by_split["train"]) < min(days_by_split["validation"])
    assert max(days_by_split["validation"]) < min(days_by_split["test"])
    assert "2014-10-21" not in all_days


def test_overlapping_split_rejected(monkeypatch):
    monkeypatch.setattr(config, "TEST_DAYS", ("2014-10-23",))
    with pytest.raises(PipelineError):
        split_days()


def test_out_of_order_split_rejected(monkeypatch):
    monkeypatch.setattr(config, "TEST_DAYS", ("2014-10-21",))
    with pytest.raises(PipelineError):
        split_days()


def test_version_is_stable_for_same_inputs(small_split):
    first = dataset_version(split_days())
    second = dataset_version(split_days())
    assert first == second
    assert first.startswith("ds-")
    assert len(first) == len("ds-") + 12


def test_version_changes_when_a_day_is_rebuilt(small_split):
    before = dataset_version(split_days())
    manifest_path = small_split.manifests_dir / "2014-10-23.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["created_at"] = "2020-01-01T00:00:00+00:00"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    assert dataset_version(split_days()) != before


def test_missing_day_manifest_raises_and_names_the_day(small_split):
    (small_split.manifests_dir / "2014-10-23.json").unlink()
    with pytest.raises(MissingInputError, match="2014-10-23"):
        dataset_version(split_days())


def test_build_writes_manifest_with_counts(small_split):
    result = build_dataset()
    manifest = load_dataset_manifest(result.version)

    assert manifest["version"] == result.version
    assert manifest["feature_version"] == FEATURE_VERSION
    for split, days in {"train": TRAIN, "validation": VALIDATION, "test": TEST}.items():
        block = manifest["splits"][split]
        frames = [pd.read_parquet(small_split.training_dir / f"dt={day}") for day in days]
        rows = sum(len(frame) for frame in frames)
        clicks = sum(int(frame["click"].sum()) for frame in frames)
        assert block["days"] == list(days)
        assert block["rows"] == rows
        assert block["clicks"] == clicks
        assert block["click_rate"] == pytest.approx(clicks / rows)
        assert set(block["day_manifest_sha256"]) == set(days)


def test_build_is_idempotent(small_split):
    first = build_dataset()
    mtime = os.stat(first.path).st_mtime_ns

    second = build_dataset()

    assert second.version == first.version
    assert os.stat(second.path).st_mtime_ns == mtime


def test_load_split_samples_uniformly_and_reproducibly(small_split):
    version = build_dataset().version
    train_rows = sum(DAY_ROWS[day] for day in TRAIN)

    sample = load_split(version, "train", ["click"])
    expected = train_rows * TRAIN_SAMPLE_FRACTION
    assert 0.8 * expected <= len(sample) <= 1.2 * expected

    first = load_split(version, "train", ["click"], sample_fraction=0.5, seed=7)
    second = load_split(version, "train", ["click"], sample_fraction=0.5, seed=7)
    pd.testing.assert_frame_equal(first, second)

    test_rows = load_split(version, "test", ["click"])
    assert len(test_rows) == DAY_ROWS[TEST[0]]


def test_load_split_never_mixes_splits(small_split):
    # Day files do not carry dt (Spark writes it into the folder name), so each split is
    # checked by its row count per day: every day has a distinct count.
    version = build_dataset().version

    train = load_split(version, "train", ["click"], sample_fraction=1.0)
    validation = load_split(version, "validation", ["click"], sample_fraction=1.0)
    test = load_split(version, "test", ["click"])

    assert len(train) == sum(DAY_ROWS[day] for day in TRAIN)
    assert len(validation) == DAY_ROWS[VALIDATION[0]]
    assert len(test) == DAY_ROWS[TEST[0]]


def test_iter_split_batches_covers_every_row(small_split):
    version = build_dataset().version

    batches = list(iter_split_batches(version, "test", ["click"], batch_rows=50))

    assert all(len(batch) <= 50 for batch in batches)
    assert sum(len(batch) for batch in batches) == DAY_ROWS[TEST[0]]
