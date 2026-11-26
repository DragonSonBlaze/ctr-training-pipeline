import mlflow
import pytest
from mlflow.tracking import MlflowClient

from ctr_pipeline import config
from ctr_pipeline.config import MLFLOW_EXPERIMENT, REGISTERED_MODEL
from ctr_pipeline.dataset import build_dataset
from ctr_pipeline.experiment import run_experiment, sweep_grid
from ctr_pipeline.models.keras_model import KerasEmbeddingModel
from ctr_pipeline.models.sklearn_baseline import SklearnLogisticModel
from helpers import training_frame, write_training_day

TRAIN = ("2014-10-22", "2014-10-23")
VALIDATION = ("2014-10-24",)
TEST = ("2014-10-25",)
DAY_ROWS = {
    "2014-10-22": 100,
    "2014-10-23": 110,
    "2014-10-24": 120,
    "2014-10-25": 130,
}
GRID = [
    {"embedding_dim": 4, "num_buckets": 2**10, "learning_rate": 1e-3},
    {"embedding_dim": 4, "num_buckets": 2**10, "learning_rate": 3e-3},
]


def small_trial(trial: dict) -> KerasEmbeddingModel:
    return KerasEmbeddingModel(hidden_units=(8,), batch_size=256, max_epochs=2, **trial)


def small_baseline() -> SklearnLogisticModel:
    return SklearnLogisticModel(num_buckets=2**10)


@pytest.fixture
def small_dataset(monkeypatch, mlflow_sandbox) -> str:
    monkeypatch.setattr(config, "TRAIN_DAYS", TRAIN)
    monkeypatch.setattr(config, "VALIDATION_DAYS", VALIDATION)
    monkeypatch.setattr(config, "TEST_DAYS", TEST)
    for day, rows in DAY_ROWS.items():
        write_training_day(mlflow_sandbox, day, training_frame(rows))
    return build_dataset().version


@pytest.fixture
def experiment(small_dataset):
    result = run_experiment(
        small_dataset,
        grid=GRID,
        model_factory=small_trial,
        baseline_factory=small_baseline,
    )
    return small_dataset, result


def _all_runs():
    experiment_id = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT).experiment_id
    return MlflowClient().search_runs([experiment_id], max_results=1000)


def _children(parent_id: str):
    return [run for run in _all_runs() if run.data.tags.get("mlflow.parentRunId") == parent_id]


def test_sweep_grid_has_eight_trials_in_order():
    grid = sweep_grid()
    assert len(grid) == 8
    assert grid[0] == {"embedding_dim": 8, "num_buckets": 2**18, "learning_rate": 1e-3}
    assert len({tuple(sorted(point.items())) for point in grid}) == 8


def test_children_are_nested_under_one_parent(experiment):
    _, result = experiment
    runs = _all_runs()
    children = [run for run in runs if run.data.tags.get("mlflow.parentRunId") is not None]

    assert {run.data.tags["mlflow.parentRunId"] for run in children} == {result.parent_run_id}
    assert len(children) == 1 + len(GRID)
    assert len(runs) == 1 + len(children)


def test_every_child_logs_params_metrics_and_model(experiment):
    _, result = experiment
    children = _children(result.parent_run_id)
    assert len(children) == 1 + len(GRID)

    trial_settings = set()
    for child in children:
        assert "val_log_loss" in child.data.metrics
        if child.data.tags["role"] == "trial":
            params = child.data.params
            trial_settings.add(
                (
                    int(params["embedding_dim"]),
                    int(params["num_buckets"]),
                    float(params["learning_rate"]),
                )
            )
        mlflow.pyfunc.load_model(f"runs:/{child.info.run_id}/model")

    expected = {
        (point["embedding_dim"], point["num_buckets"], point["learning_rate"]) for point in GRID
    }
    assert trial_settings == expected


def test_best_trial_has_lowest_val_log_loss(experiment):
    _, result = experiment
    assert result.best_trial.val_log_loss == min(t.val_log_loss for t in result.trials)
    assert result.best_trial in result.trials


def test_trials_log_per_epoch_validation_loss(experiment):
    _, result = experiment
    client = MlflowClient()
    for trial in result.trials:
        history = client.get_metric_history(trial.run_id, "val_loss")
        assert len(history) == trial.epochs_run


def test_nothing_is_registered(experiment):
    assert len(MlflowClient().search_model_versions(f"name='{REGISTERED_MODEL}'")) == 0


def test_parent_records_the_dataset_version(experiment):
    version, result = experiment
    parent = MlflowClient().get_run(result.parent_run_id)
    assert parent.data.tags["dataset_version"] == version
