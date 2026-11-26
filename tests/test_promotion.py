import mlflow
import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.tracking import MlflowClient

from ctr_pipeline import promotion
from ctr_pipeline.config import (
    CHAMPION_ALIAS,
    MIN_LOG_LOSS_IMPROVEMENT,
    MLFLOW_EXPERIMENT,
    REGISTERED_MODEL,
)
from ctr_pipeline.errors import PipelineError
from ctr_pipeline.models.embedding_nn import HashedEmbeddingModel
from ctr_pipeline.models.historical_ctr import HistoricalCtrPyfunc
from ctr_pipeline.models.logistic import HashedLogisticModel
from ctr_pipeline.promotion import _champion, promote_if_better
from helpers import training_frame, write_training_day

EVAL_DAY = "2014-10-24"
BETA = 20.0


@pytest.fixture
def eval_day(mlflow_sandbox):
    write_training_day(mlflow_sandbox, EVAL_DAY, training_frame(1000))
    return mlflow_sandbox


def log_candidate(p0: float, train_end: str | None = "2014-10-23") -> str:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run() as run:
        if train_end is not None:
            mlflow.log_param("train_end", train_end)
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


def test_first_candidate_becomes_champion(eval_day):
    run_id = log_candidate(0.5)
    result = promote_if_better(EVAL_DAY, [run_id])

    assert result.promoted is True
    assert "no champion" in result.reason
    assert champion_version() == 1


def test_worse_candidate_rejected(eval_day):
    champion_run = log_candidate(0.2)
    promote_if_better(EVAL_DAY, [champion_run])
    candidate_run = log_candidate(0.9)

    result = promote_if_better(EVAL_DAY, [candidate_run])

    assert result.promoted is False
    assert result.champion_log_loss < result.candidate_log_loss
    assert champion_version() == 1
    assert MlflowClient().get_run(candidate_run).data.tags["promoted"] == "false"


def test_candidates_that_lose_register_no_version(eval_day):
    champion_run = log_candidate(0.2)
    promote_if_better(EVAL_DAY, [champion_run])
    worse_run = log_candidate(0.9)

    result = promote_if_better(EVAL_DAY, [worse_run])

    assert result.promoted is False
    assert registry_versions() == [1]


def test_better_candidate_promoted(eval_day):
    champion_run = log_candidate(0.9)
    promote_if_better(EVAL_DAY, [champion_run])
    candidate_run = log_candidate(0.2)

    result = promote_if_better(EVAL_DAY, [candidate_run])

    assert result.promoted is True
    assert result.candidate_log_loss < result.champion_log_loss
    assert champion_version() == 2


def test_tie_keeps_champion(eval_day):
    champion_run = log_candidate(0.2)
    promote_if_better(EVAL_DAY, [champion_run])
    candidate_run = log_candidate(0.2)

    result = promote_if_better(EVAL_DAY, [candidate_run])

    assert result.promoted is False
    assert "tie" in result.reason
    assert champion_version() == 1


def test_rerun_for_same_candidate_changes_nothing(eval_day):
    run_id = log_candidate(0.5)
    promote_if_better(EVAL_DAY, [run_id])

    result = promote_if_better(EVAL_DAY, [run_id])

    assert result.promoted is True
    assert result.reason == "candidate is already champion"
    assert champion_version() == 1


def test_rerun_of_identical_candidate_keeps_registry_and_alias(eval_day):
    promote_if_better(EVAL_DAY, [log_candidate(0.2)])
    versions_before = registry_versions()
    alias_before = champion_version()

    rerun = log_candidate(0.2)
    result = promote_if_better(EVAL_DAY, [rerun])

    assert result.promoted is False
    assert "tie" in result.reason
    assert registry_versions() == versions_before
    assert champion_version() == alias_before


def test_promoting_twice_with_same_runs_changes_nothing(eval_day):
    first = log_candidate(0.2)
    second = log_candidate(0.5)
    promote_if_better(EVAL_DAY, [first, second])
    versions_before = registry_versions()
    alias_before = champion_version()

    promote_if_better(EVAL_DAY, [first, second])

    assert registry_versions() == versions_before
    assert champion_version() == alias_before


def test_champion_trained_on_eval_day_is_not_compared(eval_day, monkeypatch):
    champion_run = log_candidate(0.9, train_end=EVAL_DAY)
    better_run = log_candidate(0.2)
    # No entry for the champion's models:/ URI: the fake raises KeyError if it is scored.
    fake_log_losses(
        monkeypatch,
        {f"runs:/{champion_run}/model": 0.5, f"runs:/{better_run}/model": 0.1},
    )
    promote_if_better(EVAL_DAY, [champion_run])

    result = promote_if_better(EVAL_DAY, [better_run])

    assert result.promoted is False
    assert "trained on this day or later" in result.reason
    assert result.champion_log_loss is None
    assert champion_version() == 1


def test_champion_without_train_end_is_compared(eval_day):
    champion_run = log_candidate(0.9, train_end=None)
    promote_if_better(EVAL_DAY, [champion_run])
    better_run = log_candidate(0.2)

    result = promote_if_better(EVAL_DAY, [better_run])

    assert result.promoted is True
    assert result.candidate_log_loss < result.champion_log_loss
    assert champion_version() == 2


def test_promoted_version_is_tagged_with_date_and_log_loss(eval_day):
    run_id = log_candidate(0.2)

    result = promote_if_better(EVAL_DAY, [run_id])

    version = MlflowClient().get_model_version(REGISTERED_MODEL, str(result.candidate_version))
    assert version.tags["data_date"] == EVAL_DAY
    assert float(version.tags["eval_log_loss"]) == result.candidate_log_loss


def test_unknown_run_raises(eval_day):
    with pytest.raises(PipelineError):
        promote_if_better(EVAL_DAY, ["0" * 32])


def test_malformed_run_id_raises(eval_day):
    with pytest.raises(PipelineError):
        promote_if_better(EVAL_DAY, ["2026-11-25 INFO something"])


class _FailingClient:
    def __init__(self, message: str, error_code: int) -> None:
        self.message = message
        self.error_code = error_code

    def get_model_version_by_alias(self, name, alias):
        raise MlflowException(self.message, error_code=self.error_code)


def test_missing_alias_means_no_champion():
    client = _FailingClient("Registered model alias champion not found.", INVALID_PARAMETER_VALUE)
    assert _champion(client) is None


def test_missing_model_means_no_champion():
    client = _FailingClient("Registered Model ctr-model not found", RESOURCE_DOES_NOT_EXIST)
    assert _champion(client) is None


def test_other_registry_errors_are_not_treated_as_no_champion():
    with pytest.raises(MlflowException):
        _champion(_FailingClient("API request failed", INTERNAL_ERROR))


def log_logistic_candidate() -> str:
    model = HashedLogisticModel(num_buckets=2**12, epochs=1, batch_size=256)
    model.fit(training_frame(200))
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            name="model",
            python_model=model.to_pyfunc(),
            pip_requirements=["numpy", "pandas", "torch"],
        )
    return run.info.run_id


def fake_log_losses(monkeypatch, log_losses: dict[str, float]) -> None:
    """Replace the day evaluation with fixed values, keyed by model URI."""
    monkeypatch.setattr(
        promotion,
        "evaluate_model_uri_on_day",
        lambda model_uri, date: log_losses[model_uri],
    )


def test_improvement_below_threshold_keeps_champion(eval_day, monkeypatch):
    champion_loss = 0.4
    champion_run = log_candidate(0.5)
    challenger_run = log_candidate(0.5)
    fake_log_losses(
        monkeypatch,
        {
            f"runs:/{champion_run}/model": champion_loss,
            f"models:/{REGISTERED_MODEL}/1": champion_loss,
            f"runs:/{challenger_run}/model": champion_loss - MIN_LOG_LOSS_IMPROVEMENT / 2,
        },
    )
    promote_if_better(EVAL_DAY, [champion_run])

    result = promote_if_better(EVAL_DAY, [challenger_run])

    assert result.promoted is False
    assert "MIN_LOG_LOSS_IMPROVEMENT" in result.reason
    assert champion_version() == 1


def test_logistic_candidate_can_replace_historical_champion(eval_day, monkeypatch):
    champion_run = log_candidate(0.5)
    logistic_run = log_logistic_candidate()
    fake_log_losses(
        monkeypatch,
        {
            f"runs:/{champion_run}/model": 0.45,
            f"models:/{REGISTERED_MODEL}/1": 0.45,
            f"runs:/{logistic_run}/model": 0.40,
        },
    )
    promote_if_better(EVAL_DAY, [champion_run])

    result = promote_if_better(EVAL_DAY, [logistic_run])

    assert result.promoted is True
    assert result.champion_log_loss == 0.45
    assert result.candidate_log_loss == 0.40
    assert champion_version() == 2


def log_embedding_candidate() -> str:
    model = HashedEmbeddingModel(num_buckets=2**12, epochs=1, batch_size=256, hidden_units=(8,))
    model.fit(training_frame(200))
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            name="model",
            python_model=model.to_pyfunc(),
            pip_requirements=["numpy", "pandas", "torch"],
        )
    return run.info.run_id


def test_best_of_two_candidates_is_compared_with_champion(eval_day, monkeypatch):
    champion_run = log_candidate(0.5)
    logistic_run = log_logistic_candidate()
    embedding_run = log_embedding_candidate()
    fake_log_losses(
        monkeypatch,
        {
            f"runs:/{champion_run}/model": 0.45,
            f"models:/{REGISTERED_MODEL}/1": 0.45,
            f"runs:/{logistic_run}/model": 0.43,
            f"runs:/{embedding_run}/model": 0.41,
        },
    )
    promote_if_better(EVAL_DAY, [champion_run])

    result = promote_if_better(EVAL_DAY, [logistic_run, embedding_run])

    assert result.promoted is True
    assert result.best_run_id == embedding_run
    assert result.candidate_log_losses == {logistic_run: 0.43, embedding_run: 0.41}
    # Only the winner is registered, so the embedding model takes version 2.
    assert registry_versions() == [1, 2]
    assert champion_version() == 2
    loser = MlflowClient().get_run(logistic_run).data.tags
    assert loser["promoted"] == "false"
    assert loser["reason"] == f"lower log loss from candidate run {embedding_run}"


def test_best_candidate_must_still_beat_champion(eval_day, monkeypatch):
    champion_run = log_candidate(0.5)
    logistic_run = log_logistic_candidate()
    embedding_run = log_embedding_candidate()
    fake_log_losses(
        monkeypatch,
        {
            f"runs:/{champion_run}/model": 0.40,
            f"models:/{REGISTERED_MODEL}/1": 0.40,
            f"runs:/{logistic_run}/model": 0.42,
            f"runs:/{embedding_run}/model": 0.41,
        },
    )
    promote_if_better(EVAL_DAY, [champion_run])

    result = promote_if_better(EVAL_DAY, [logistic_run, embedding_run])

    assert result.promoted is False
    assert result.best_run_id == embedding_run
    assert result.candidate_log_loss == 0.41
    assert champion_version() == 1


def test_empty_run_list_raises():
    with pytest.raises(PipelineError):
        promote_if_better(EVAL_DAY, [])
