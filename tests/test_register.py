import mlflow
import pytest
from mlflow.tracking import MlflowClient

from ctr_pipeline import config
from ctr_pipeline.config import CHAMPION_ALIAS, MLFLOW_EXPERIMENT, REGISTERED_MODEL
from ctr_pipeline.dataset import build_dataset
from ctr_pipeline.models.historical_ctr import HistoricalCtrPyfunc
from ctr_pipeline.promotion import register_if_better
from helpers import training_frame, write_training_day

TRAIN = ("2014-10-22",)
VALIDATION = ("2014-10-23",)
TEST = ("2014-10-24",)
BETA = 20.0
TRAIN_DAYS = "2014-10-22,2014-10-23"
CALIBRATION_METRICS = (
    "test_auc",
    "test_log_loss",
    "test_mean_pred",
    "test_actual_ctr",
    "test_calibration_rel_error",
)


@pytest.fixture
def dataset(monkeypatch, mlflow_sandbox) -> str:
    monkeypatch.setattr(config, "TRAIN_DAYS", TRAIN)
    monkeypatch.setattr(config, "VALIDATION_DAYS", VALIDATION)
    monkeypatch.setattr(config, "TEST_DAYS", TEST)
    for day in (*TRAIN, *VALIDATION, *TEST):
        write_training_day(mlflow_sandbox, day, training_frame(1000))
    return build_dataset().version


def log_candidate(p0: float, train_days: str | None = TRAIN_DAYS) -> str:
    """A candidate that predicts p0 for every row. Its test log loss is smallest at p0 = 0.2."""
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run() as run:
        if train_days is not None:
            mlflow.log_param("train_days", train_days)
        mlflow.pyfunc.log_model(
            name="model",
            python_model=HistoricalCtrPyfunc(p0, BETA),
            pip_requirements=["numpy", "pandas"],
        )
    return run.info.run_id


def champion_version() -> int:
    return int(MlflowClient().get_model_version_by_alias(REGISTERED_MODEL, CHAMPION_ALIAS).version)


def registry_versions() -> list[int]:
    matches = MlflowClient().search_model_versions(f"name='{REGISTERED_MODEL}'")
    return sorted(int(version.version) for version in matches)


def test_first_candidate_registered_and_becomes_champion(dataset):
    run_id = log_candidate(0.2)

    result = register_if_better(dataset, [run_id])

    assert result.promoted is True
    assert "no champion" in result.reason
    assert champion_version() == 1
    assert registry_versions() == [1]


def test_worse_candidate_not_registered(dataset):
    register_if_better(dataset, [log_candidate(0.2)])
    worse_run = log_candidate(0.5)

    result = register_if_better(dataset, [worse_run])

    assert result.promoted is False
    assert registry_versions() == [1]
    assert champion_version() == 1


def test_best_of_two_candidates_wins_and_only_it_is_registered(dataset):
    worse_run = log_candidate(0.5)
    better_run = log_candidate(0.2)

    result = register_if_better(dataset, [worse_run, better_run])

    assert result.best_run_id == better_run
    assert registry_versions() == [1]
    assert MlflowClient().get_model_version(REGISTERED_MODEL, "1").run_id == better_run
    assert MlflowClient().get_run(worse_run).data.tags["promoted"] == "false"


def test_rerun_with_identical_candidate_changes_nothing(dataset):
    first_run = log_candidate(0.2)
    register_if_better(dataset, [first_run])

    rerun_of_same = register_if_better(dataset, [first_run])
    identical_run = log_candidate(0.2)
    identical = register_if_better(dataset, [identical_run])

    assert "already champion" in rerun_of_same.reason
    assert identical.promoted is False
    assert "tie on test log loss" in identical.reason
    assert registry_versions() == [1]
    assert champion_version() == 1


def test_champion_trained_on_test_days_is_kept_without_comparison(dataset):
    # The champion's training days run on into the test day 2014-10-24, so it has seen those rows.
    champion_run = log_candidate(0.9, train_days="2014-10-22,2014-10-24")
    register_if_better(dataset, [champion_run])
    better_run = log_candidate(0.2)

    result = register_if_better(dataset, [better_run])

    assert result.promoted is False
    assert "trained on test days" in result.reason
    assert result.champion_log_loss is None
    assert champion_version() == 1


def test_candidate_runs_get_test_metrics_and_reliability_artifact(dataset):
    run_id = log_candidate(0.2)

    register_if_better(dataset, [run_id])

    metrics = MlflowClient().get_run(run_id).data.metrics
    for key in CALIBRATION_METRICS:
        assert key in metrics
    artifacts = [artifact.path for artifact in MlflowClient().list_artifacts(run_id, "calibration")]
    assert "calibration/reliability.csv" in artifacts


def test_registered_version_tagged_with_dataset_version(dataset):
    run_id = log_candidate(0.2)

    register_if_better(dataset, [run_id])

    tags = MlflowClient().get_model_version(REGISTERED_MODEL, "1").tags
    assert tags["dataset_version"] == dataset
    assert "test_log_loss" in tags
