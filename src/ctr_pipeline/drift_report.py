"""Evidently data drift report: the train split (reference) against the test split (current).

It answers whether the days the model is scored on look like the days it learned from. The
report is observability only: nothing here blocks training or registration.

Input:  data/datasets/<version>/dataset.json plus the daily training sets
Output: on the parent MLflow run, the metrics drift_share and drifted_columns, the artifacts
        drift/data_drift.html and drift/columns.json
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset
from evidently.core.report import Report, Snapshot
from evidently.presets import DataDriftPreset
from mlflow.tracking import MlflowClient

from ctr_pipeline.config import DRIFT_SAMPLE_ROWS, TRAIN_SAMPLE_SEED, get_settings
from ctr_pipeline.contracts import DRIFT_CATEGORICAL, NUMERIC_FEATURES
from ctr_pipeline.dataset import load_dataset_manifest, load_split

logger = logging.getLogger(__name__)

DRIFT_COLUMNS: list[str] = NUMERIC_FEATURES + DRIFT_CATEGORICAL
ARTIFACT_DIR = "drift"
HTML_NAME = "data_drift.html"
COLUMNS_NAME = "columns.json"
VALUE_DRIFT_TYPE = "evidently:metric_v2:ValueDrift"
# Evidently reports a column's drift test as SUCCESS (no drift) or FAIL (drift detected).
NO_DRIFT_STATUS = "SUCCESS"
DRIFT_STATUS = "FAIL"


@dataclass(frozen=True)
class DriftSummary:
    drift_share: float
    drifted_columns: int
    columns: dict[str, dict]  # column -> {"test": str, "score": float, "drifted": bool}


def sample_split(version: str, split: str, rows: int, seed: int) -> pd.DataFrame:
    """About `rows` rows of a split, sampled uniformly over its days.

    The fraction comes from the row count in the dataset manifest, so the sample is close to
    the target size without reading the whole split twice.
    """
    split_rows = load_dataset_manifest(version)["splits"][split]["rows"]
    fraction = min(1.0, rows / split_rows)
    frame = load_split(version, split, DRIFT_COLUMNS, sample_fraction=fraction, seed=seed)
    return frame.head(rows)


def _to_dataset(frame: pd.DataFrame) -> Dataset:
    # Categories are cast to text so Evidently never tests them as numbers.
    typed = frame[DRIFT_COLUMNS].astype({column: str for column in DRIFT_CATEGORICAL})
    definition = DataDefinition(
        numerical_columns=NUMERIC_FEATURES,
        categorical_columns=DRIFT_CATEGORICAL,
    )
    return Dataset.from_pandas(typed, data_definition=definition)


def build_report(reference: pd.DataFrame, current: pd.DataFrame) -> Snapshot:
    # include_tests makes Evidently record a pass or fail per column, which summarize reads.
    report = Report([DataDriftPreset(columns=DRIFT_COLUMNS)], include_tests=True)
    return report.run(current_data=_to_dataset(current), reference_data=_to_dataset(reference))


def _value_drift_scores(metrics: list[dict]) -> dict[str, tuple[str, float]]:
    """Column -> (test name, drift score) from the ValueDrift metrics."""
    scores = {}
    for metric in metrics:
        config = metric["config"]
        if config.get("type") == VALUE_DRIFT_TYPE:
            scores[config["column"]] = (config["method"], float(metric["value"]))
    return scores


def _value_drift_statuses(tests: list[dict]) -> dict[str, str]:
    """Column -> status of its ValueDrift test."""
    statuses = {}
    for test in tests:
        params = test["metric_config"]["params"]
        if params.get("type") == VALUE_DRIFT_TYPE:
            statuses[params["column"]] = test["status"]
    return statuses


def summarize(snapshot: Snapshot) -> DriftSummary:
    """Count the drifted columns of a snapshot. Pure: it reads the snapshot's dict only."""
    payload = snapshot.dict()
    scores = _value_drift_scores(payload["metrics"])
    statuses = _value_drift_statuses(payload["tests"])

    columns: dict[str, dict] = {}
    for column, (test, score) in scores.items():
        status = statuses.get(column)
        if status not in (NO_DRIFT_STATUS, DRIFT_STATUS):
            raise ValueError(f"no drift result for column {column}: status {status!r}")
        columns[column] = {"test": test, "score": score, "drifted": status == DRIFT_STATUS}

    if not columns:
        raise ValueError("the drift snapshot has no ValueDrift results")
    drifted = sum(1 for column in columns.values() if column["drifted"])
    return DriftSummary(
        drift_share=drifted / len(columns),
        drifted_columns=drifted,
        columns=columns,
    )


def _write_artifacts(snapshot: Snapshot, summary: DriftSummary, directory: Path) -> list[Path]:
    html_path = directory / HTML_NAME
    snapshot.save_html(str(html_path))
    columns_path = directory / COLUMNS_NAME
    columns_path.write_text(json.dumps(summary.columns, indent=2, sort_keys=True) + "\n")
    return [html_path, columns_path]


def run_drift_report(
    version: str, parent_run_id: str, rows: int = DRIFT_SAMPLE_ROWS
) -> DriftSummary:
    """Compare train days with test days and log the report to the parent run."""
    reference = sample_split(version, "train", rows, TRAIN_SAMPLE_SEED)
    current = sample_split(version, "test", rows, TRAIN_SAMPLE_SEED)
    snapshot = build_report(reference, current)
    summary = summarize(snapshot)

    client = MlflowClient(tracking_uri=get_settings().mlflow_tracking_uri)
    with tempfile.TemporaryDirectory() as temporary:
        for path in _write_artifacts(snapshot, summary, Path(temporary)):
            client.log_artifact(parent_run_id, str(path), artifact_path=ARTIFACT_DIR)
    client.log_metric(parent_run_id, "drift_share", summary.drift_share)
    client.log_metric(parent_run_id, "drifted_columns", float(summary.drifted_columns))
    logger.info(
        "drift report for %s: %d of %d columns drifted",
        version,
        summary.drifted_columns,
        len(summary.columns),
    )
    return summary
