"""Stage 4: register the best candidate on the test split, and move the `champion` alias
only if it beats the champion on the same test rows.

Every candidate and the champion are scored by the same function on the same test days,
so the comparison measures the model rather than the day. Log loss decides, because
predicted click rates are used as probabilities and must be calibrated.

Experiment runs log each model to its run only. A candidate is registered as a new version
of the model here, and only when it wins the comparison, so the registry holds only
models that were champion at some point. Re-running register with the same candidate
therefore leaves the registry as it was.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import mlflow
import mlflow.pyfunc
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from ctr_pipeline import monitoring
from ctr_pipeline.calibration import StreamingScores
from ctr_pipeline.config import (
    CHAMPION_ALIAS,
    MIN_LOG_LOSS_IMPROVEMENT,
    REGISTERED_MODEL,
    get_settings,
)
from ctr_pipeline.contracts import PREDICT_COLUMNS
from ctr_pipeline.dataset import iter_split_batches, load_dataset_manifest
from ctr_pipeline.errors import PipelineError

logger = logging.getLogger(__name__)

# MLflow run ids are 32 lowercase hex characters. Checking the shape first turns a
# bad hand-off from Airflow into a clear error, instead of a search that matches nothing.
RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True)
class Candidate:
    run_id: str
    model_uri: str
    log_loss: float


@dataclass(frozen=True)
class PromotionResult:
    """The decision for the best candidate. candidate_version and candidate_log_loss are
    the best candidate's; candidate_log_losses holds every candidate's log loss by run id.

    candidate_version is the version registered by this call, the champion's version when
    the best candidate already is the champion, and None when it was not promoted."""

    promoted: bool
    candidate_version: int | None
    candidate_log_loss: float
    champion_version: int | None
    champion_log_loss: float | None
    reason: str
    best_run_id: str
    candidate_log_losses: dict[str, float]


def _check_run_id(run_id: str) -> None:
    if not RUN_ID_PATTERN.match(run_id):
        raise PipelineError(f"not an MLflow run id (32 lowercase hex characters): {run_id!r}")


def _require_run(client: MlflowClient, run_id: str) -> Run:
    try:
        return client.get_run(run_id)
    except MlflowException as exc:
        if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
            raise PipelineError(f"no MLflow run {run_id}") from exc
        raise


def _champion(client: MlflowClient) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(REGISTERED_MODEL, CHAMPION_ALIAS)
    except MlflowException as exc:
        # Only "no such model" or "no such alias" means there is no champion. Any other
        # error (server down, bad credentials) must fail the step, or a candidate would be
        # promoted without a comparison. MLflow reports a missing alias as
        # INVALID_PARAMETER_VALUE with "not found" in the message.
        missing_model = exc.error_code == "RESOURCE_DOES_NOT_EXIST"
        missing_alias = exc.error_code == "INVALID_PARAMETER_VALUE" and "not found" in str(exc)
        if missing_model or missing_alias:
            return None
        raise


# Test metrics are logged to each candidate run under these names. calibration_rel_error is
# the calibration check from task 20: how far the mean predicted click rate is from the actual.
TEST_METRIC_NAMES: dict[str, str] = {
    "test_auc": "auc",
    "test_log_loss": "log_loss",
    "test_mean_pred": "mean_pred",
    "test_actual_ctr": "actual_ctr",
    "test_calibration_rel_error": "calibration_rel_error",
}
RELIABILITY_ARTIFACT = "calibration/reliability.csv"


def score_on_test(model_uri: str, version: str) -> StreamingScores:
    """Score every row of the test split with the model at model_uri, one batch at a time.

    The model is loaded once. Only the test days are read, so no model is scored on rows it
    could have trained on.
    """
    model = mlflow.pyfunc.load_model(model_uri)
    scores = StreamingScores()
    for batch in iter_split_batches(version, "test", [*PREDICT_COLUMNS, "click"]):
        scores.add(batch["click"].to_numpy(), model.predict(batch[PREDICT_COLUMNS]))
    return scores


def candidate_run_ids(parent_run_id: str) -> tuple[str, str]:
    """The baseline run id and the best Keras trial run id of an experiment's parent run."""
    _check_run_id(parent_run_id)
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    tags = _require_run(client, parent_run_id).data.tags
    baseline_run_id = tags.get("baseline_run_id")
    best_trial_run_id = tags.get("best_trial_run_id")
    if not baseline_run_id or not best_trial_run_id:
        raise PipelineError(f"run {parent_run_id} has no candidate tags, is it an experiment run?")
    return baseline_run_id, best_trial_run_id


def _test_days(version: str) -> list[str]:
    days = sorted(load_dataset_manifest(version)["splits"]["test"]["days"])
    if not days:
        raise PipelineError(f"dataset {version} has no test days")
    return days


def _champion_last_train_day(client: MlflowClient, champion: ModelVersion) -> str | None:
    """The last day the champion's run trained on, from train_days or, for older runs, train_end."""
    params = client.get_run(champion.run_id).data.params
    if "train_days" in params:
        days = [day for day in params["train_days"].split(",") if day]
        return max(days) if days else None
    return params.get("train_end")


def _compare_on_test(
    candidate_log_loss: float,
    champion_log_loss: float,
) -> tuple[bool, str]:
    """Return (promote, reason) for a candidate and a champion scored on the same test split."""
    if candidate_log_loss < champion_log_loss - MIN_LOG_LOSS_IMPROVEMENT:
        return True, "candidate has lower test log loss than the champion"
    if candidate_log_loss > champion_log_loss:
        return False, "candidate has higher test log loss than the champion"
    if candidate_log_loss == champion_log_loss:
        return (
            False,
            "tie on test log loss, improvement below MIN_LOG_LOSS_IMPROVEMENT, champion kept",
        )
    return False, "improvement below MIN_LOG_LOSS_IMPROVEMENT, champion kept"


def _decide_on_test(
    client: MlflowClient,
    best: Candidate,
    champion: ModelVersion | None,
    version: str,
    first_test_day: str,
) -> tuple[bool, float | None, str]:
    """Return (promote, champion test log loss, reason).

    The champion is scored only when a comparison is needed. A champion whose training data
    reaches into the test days has seen the rows it would be judged on, so it is kept as it is.
    """
    if champion is None:
        return True, None, "no champion yet"
    if champion.run_id == best.run_id:
        return True, None, "candidate is already champion"
    last_train_day = _champion_last_train_day(client, champion)
    if last_train_day is not None and last_train_day >= first_test_day:
        return False, None, "champion was trained on test days or later, champion kept"
    champion_scores = score_on_test(f"models:/{REGISTERED_MODEL}/{champion.version}", version)
    champion_log_loss = float(champion_scores.metrics()["log_loss"])
    promote, reason = _compare_on_test(best.log_loss, champion_log_loss)
    return promote, champion_log_loss, reason


def _log_test_results(
    client: MlflowClient,
    run_id: str,
    version: str,
    scores: StreamingScores,
    metrics: dict[str, float | None],
) -> None:
    for key, name in TEST_METRIC_NAMES.items():
        value = metrics[name]
        # AUC is undefined when the test rows hold one class only, so it is left out then.
        if value is not None:
            client.log_metric(run_id, key, float(value))
    client.log_text(run_id, scores.table().to_csv(index=False), RELIABILITY_ARTIFACT)
    client.set_tag(run_id, "test_dataset_version", version)


def _model_type(client: MlflowClient, run_id: str) -> str | None:
    """Daily runs tag their model type, experiment runs log it as a param."""
    data = client.get_run(run_id).data
    return data.tags.get("model_type") or data.params.get("model_type")


def _record_run_decisions(
    client: MlflowClient,
    version: str,
    last_test_day: str,
    candidates: list[Candidate],
    metrics_by_run: dict[str, dict[str, float | None]],
    best: Candidate,
    result: PromotionResult,
    registered_version: int | None,
) -> None:
    """Write the decision to MLflow tags and to the Postgres model_runs table, per candidate."""
    for candidate in candidates:
        is_best = candidate.run_id == best.run_id
        if is_best:
            client.set_tag(candidate.run_id, "promoted", str(result.promoted).lower())
            client.set_tag(candidate.run_id, "reason", result.reason)
            client.set_tag(
                candidate.run_id,
                "champion_version_before",
                "none" if result.champion_version is None else str(result.champion_version),
            )
        else:
            client.set_tag(candidate.run_id, "promoted", "false")
            client.set_tag(
                candidate.run_id,
                "reason",
                f"lost on test log loss to candidate run {best.run_id}",
            )

        metrics = metrics_by_run[candidate.run_id]
        monitoring.record_model_run(
            {
                "mlflow_run_id": candidate.run_id,
                "data_date": last_test_day,
                "model_type": _model_type(client, candidate.run_id),
                "model_version": registered_version if is_best else None,
                "dataset_version": version,
                "eval_rows": metrics["rows"],
                "auc": metrics["auc"],
                "log_loss": metrics["log_loss"],
                "mean_pred": metrics["mean_pred"],
                "actual_ctr": metrics["actual_ctr"],
            }
        )
        if is_best:
            monitoring.update_model_run_promotion(
                candidate.run_id, result.promoted, result.champion_log_loss, registered_version
            )
        else:
            monitoring.update_model_run_promotion(candidate.run_id, False, None)


def register_if_better(version: str, run_ids: list[str]) -> PromotionResult:
    """Score each candidate on the test split, then register the best one if it beats the champion.

    The candidate with the lowest test log loss wins; on a tie the earlier run id wins. It is
    registered and takes the champion alias only when its test log loss is lower than the
    champion's by more than MIN_LOG_LOSS_IMPROVEMENT, both scored on the same test rows.
    """
    if not run_ids:
        raise PipelineError("register needs at least one MLflow run id")
    for run_id in run_ids:
        _check_run_id(run_id)
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    for run_id in run_ids:
        _require_run(client, run_id)

    test_days = _test_days(version)
    candidates: list[Candidate] = []
    metrics_by_run: dict[str, dict[str, float | None]] = {}
    for run_id in run_ids:
        model_uri = f"runs:/{run_id}/model"
        scores = score_on_test(model_uri, version)
        metrics = scores.metrics()
        _log_test_results(client, run_id, version, scores, metrics)
        metrics_by_run[run_id] = metrics
        candidates.append(
            Candidate(run_id=run_id, model_uri=model_uri, log_loss=float(metrics["log_loss"]))
        )

    # min() returns the first of equal items, so a tie goes to the earlier run id.
    best = min(candidates, key=lambda candidate: candidate.log_loss)
    champion = _champion(client)
    promote, champion_log_loss, reason = _decide_on_test(
        client, best, champion, version, test_days[0]
    )

    registered_version: int | None = None
    if promote and (champion is None or champion.run_id != best.run_id):
        registered_version = int(mlflow.register_model(best.model_uri, REGISTERED_MODEL).version)
        client.set_registered_model_alias(REGISTERED_MODEL, CHAMPION_ALIAS, registered_version)
        client.set_model_version_tag(
            REGISTERED_MODEL, str(registered_version), "dataset_version", version
        )
        client.set_model_version_tag(
            REGISTERED_MODEL, str(registered_version), "test_log_loss", repr(best.log_loss)
        )

    candidate_version: int | None = None
    if registered_version is not None:
        candidate_version = registered_version
    elif promote and champion is not None:
        candidate_version = int(champion.version)

    result = PromotionResult(
        promoted=promote,
        candidate_version=candidate_version,
        candidate_log_loss=best.log_loss,
        champion_version=None if champion is None else int(champion.version),
        champion_log_loss=champion_log_loss,
        reason=reason,
        best_run_id=best.run_id,
        candidate_log_losses={candidate.run_id: candidate.log_loss for candidate in candidates},
    )
    _record_run_decisions(
        client,
        version,
        test_days[-1],
        candidates,
        metrics_by_run,
        best,
        result,
        registered_version,
    )

    logger.info(
        "register for dataset %s: %s (%s); best candidate %s test log loss %.6f, champion %s "
        "test log loss %s, registered %s",
        version,
        "promoted" if promote else "not promoted",
        reason,
        best.run_id,
        best.log_loss,
        "none" if result.champion_version is None else f"v{result.champion_version}",
        "n/a" if champion_log_loss is None else f"{champion_log_loss:.6f}",
        "not registered" if registered_version is None else f"v{registered_version}",
    )
    return result
