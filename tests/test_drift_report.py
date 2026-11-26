"""Evidently drift report: the train split against the test split, logged to a parent run.

Frames are built in memory and the pipeline runs on a small dataset in a temporary folder, so
no test reads data/.
"""

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from ctr_pipeline import config
from ctr_pipeline.contracts import DRIFT_CATEGORICAL, NUMERIC_FEATURES
from ctr_pipeline.dataset import build_dataset
from ctr_pipeline.drift_report import (
    DRIFT_COLUMNS,
    build_report,
    run_drift_report,
    summarize,
)
from helpers import training_frame, write_training_day

NUMERIC_TESTS = {"Wasserstein distance (normed)", "K-S p_value"}
TRAIN = ("2014-10-22", "2014-10-23")
VALIDATION = ("2014-10-24",)
TEST = ("2014-10-25",)
DAY_ROWS = {"2014-10-22": 100, "2014-10-23": 110, "2014-10-24": 120, "2014-10-25": 130}


def drift_frame(rows: int, seed: int, shift: float = 0.0) -> pd.DataFrame:
    """Continuous numeric features and varied categories, so every column has a real spread."""
    rng = np.random.default_rng(seed)
    frame = training_frame(rows)
    for column in NUMERIC_FEATURES:
        frame[column] = rng.random(rows) * 0.2
    frame["app_ctr_smooth"] = frame["app_ctr_smooth"] + shift
    for column in DRIFT_CATEGORICAL:
        frame[column] = rng.integers(0, 4, rows)
    return frame


@pytest.fixture
def small_dataset(monkeypatch, mlflow_sandbox):
    monkeypatch.setattr(config, "TRAIN_DAYS", TRAIN)
    monkeypatch.setattr(config, "VALIDATION_DAYS", VALIDATION)
    monkeypatch.setattr(config, "TEST_DAYS", TEST)
    for day, rows in DAY_ROWS.items():
        write_training_day(mlflow_sandbox, day, drift_frame(rows, seed=rows))
    return build_dataset().version


def test_identical_frames_have_no_drift():
    reference = training_frame(2000)
    current = training_frame(2000)

    summary = summarize(build_report(reference, current))

    assert summary.drift_share == 0
    assert summary.drifted_columns == 0


def test_shifted_numeric_column_is_flagged():
    reference = drift_frame(2000, seed=1)
    current = drift_frame(2000, seed=2, shift=3.0)

    summary = summarize(build_report(reference, current))

    assert summary.columns["app_ctr_smooth"]["test"] == "Wasserstein distance (normed)"
    assert summary.columns["app_ctr_smooth"]["drifted"] is True
    assert summary.drifted_columns >= 1


def test_categorical_columns_are_treated_as_categorical():
    reference = drift_frame(2000, seed=1)
    current = drift_frame(2000, seed=2)

    summary = summarize(build_report(reference, current))

    for column in DRIFT_CATEGORICAL:
        assert summary.columns[column]["test"] not in NUMERIC_TESTS, column
    for column in NUMERIC_FEATURES:
        assert summary.columns[column]["test"] in NUMERIC_TESTS, column


def test_summary_counts_match_columns():
    reference = drift_frame(2000, seed=1)
    current = drift_frame(2000, seed=2, shift=3.0)

    summary = summarize(build_report(reference, current))

    assert set(summary.columns) == set(DRIFT_COLUMNS)
    assert summary.drifted_columns == sum(c["drifted"] for c in summary.columns.values())
    assert summary.drift_share == summary.drifted_columns / len(summary.columns)


def test_report_logged_to_parent_run(small_dataset):
    with mlflow.start_run(run_name="drift-test") as parent:
        parent_id = parent.info.run_id

    summary = run_drift_report(small_dataset, parent_id, rows=500)

    client = MlflowClient(tracking_uri=config.get_settings().mlflow_tracking_uri)
    run = client.get_run(parent_id)
    assert run.data.metrics["drift_share"] == pytest.approx(summary.drift_share)
    assert run.data.metrics["drifted_columns"] == summary.drifted_columns
    artifact_paths = {item.path for item in client.list_artifacts(parent_id, "drift")}
    assert artifact_paths == {"drift/data_drift.html", "drift/columns.json"}
