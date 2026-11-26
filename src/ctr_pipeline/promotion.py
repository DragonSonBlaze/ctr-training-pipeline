"""Stage 4: move the `champion` alias only if the candidate beats the champion on the same day.

Both models are scored by the same function on the same evaluation day, so the
comparison measures the model rather than the day. Log loss decides, because
predicted click rates are used as probabilities and must be calibrated.

Training logs each model to its run only. A candidate is registered as a new version
of the model here, and only when it wins the comparison, so the registry holds only
models that were champion at some point. Re-running a day therefore leaves the
registry as it was.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import mlflow
import mlflow.pyfunc
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from ctr_pipeline import monitoring
from ctr_pipeline.config import (
    CHAMPION_ALIAS,
    MIN_LOG_LOSS_IMPROVEMENT,
    REGISTERED_MODEL,
    get_settings,
)
from ctr_pipeline.contracts import PREDICT_COLUMNS
from ctr_pipeline.errors import PipelineError
from ctr_pipeline.models.evaluate import compute_metrics
from ctr_pipeline.models.train import predict_day

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


def evaluate_model_uri_on_day(model_uri: str, date: str) -> float:
    """Log loss of the model at model_uri on every row of date.

    Every model type reads the same PREDICT_COLUMNS, so a learned candidate and a
    historical champion are scored by the same code on the same rows.
    """
    model = mlflow.pyfunc.load_model(model_uri)
    labels, predictions = predict_day(
        lambda batch: model.predict(batch[PREDICT_COLUMNS]),
        date,
        [*PREDICT_COLUMNS, "click"],
    )
    return float(compute_metrics(labels, predictions)["log_loss"])


def _check_run_id(run_id: str) -> None:
    if not RUN_ID_PATTERN.match(run_id):
        raise PipelineError(f"not an MLflow run id (32 lowercase hex characters): {run_id!r}")


def _score_candidate(client: MlflowClient, run_id: str, date: str) -> Candidate:
    try:
        client.get_run(run_id)
    except MlflowException as exc:
        if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
            raise PipelineError(f"no MLflow run {run_id}") from exc
        raise
    model_uri = f"runs:/{run_id}/model"
    log_loss = evaluate_model_uri_on_day(model_uri, date)
    return Candidate(run_id=run_id, model_uri=model_uri, log_loss=log_loss)


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


def _champion_train_end(client: MlflowClient, champion: ModelVersion) -> str | None:
    """The last training day of the champion's run, or None if the run did not log it."""
    return client.get_run(champion.run_id).data.params.get("train_end")


def _decide(
    date: str,
    best: Candidate,
    champion: ModelVersion | None,
    champion_train_end: str | None,
) -> tuple[bool, float | None, str]:
    """Return (promote, champion log loss on date, reason).

    A champion trained on date or later has seen the evaluation rows, so its log loss
    would be too good to compare with. It is kept without being evaluated.
    """
    if champion is None:
        return True, None, "no champion yet"
    if champion.run_id == best.run_id:
        return True, None, "candidate is already champion"
    if champion_train_end is not None and champion_train_end >= date:
        return False, None, "champion was trained on this day or later, champion kept"
    champion_log_loss = evaluate_model_uri_on_day(
        f"models:/{REGISTERED_MODEL}/{champion.version}", date
    )
    # The margin keeps rounding noise from deciding a promotion. Ties are the closest
    # case of "not better by the margin", so their reason says so as well.
    if best.log_loss < champion_log_loss - MIN_LOG_LOSS_IMPROVEMENT:
        return True, champion_log_loss, "candidate has lower log loss on the same day"
    if best.log_loss > champion_log_loss:
        return False, champion_log_loss, "candidate has higher log loss on the same day"
    if best.log_loss == champion_log_loss:
        return (
            False,
            champion_log_loss,
            "tie on log loss, improvement below MIN_LOG_LOSS_IMPROVEMENT, champion kept",
        )
    return (
        False,
        champion_log_loss,
        "improvement below MIN_LOG_LOSS_IMPROVEMENT, champion kept",
    )


def _record_decision(
    client: MlflowClient,
    run_id: str,
    date: str,
    result: PromotionResult,
    registered_version: int | None,
) -> None:
    """Write the decision for the best candidate to MLflow and Postgres.

    Model version tags go only on a version this call registered, so a version
    never gets a second date or loss from a later run.
    """
    client.set_tag(run_id, "promoted", str(result.promoted).lower())
    client.set_tag(run_id, "reason", result.reason)
    client.set_tag(
        run_id,
        "champion_version_before",
        "none" if result.champion_version is None else str(result.champion_version),
    )
    if result.champion_log_loss is not None:
        client.set_tag(run_id, "champion_log_loss", repr(result.champion_log_loss))
    if registered_version is not None:
        client.set_model_version_tag(REGISTERED_MODEL, str(registered_version), "data_date", date)
        client.set_model_version_tag(
            REGISTERED_MODEL,
            str(registered_version),
            "eval_log_loss",
            repr(result.candidate_log_loss),
        )
    monitoring.update_model_run_promotion(
        run_id, result.promoted, result.champion_log_loss, registered_version
    )


def _record_not_best(
    client: MlflowClient,
    candidate: Candidate,
    best_run_id: str,
) -> None:
    """Mark a candidate that lost to another one on the same day. It is never registered."""
    client.set_tag(candidate.run_id, "promoted", "false")
    client.set_tag(candidate.run_id, "reason", f"lower log loss from candidate run {best_run_id}")
    monitoring.update_model_run_promotion(candidate.run_id, False, None)


def promote_if_better(date: str, run_ids: list[str]) -> PromotionResult:
    """Score every candidate on date, then compare the best one with the champion.

    The candidate with the lowest log loss wins; on a tie the earlier run id wins. That
    candidate must beat the champion on the same day to be registered and take the alias.
    """
    if not run_ids:
        raise PipelineError("promote needs at least one MLflow run id")
    for run_id in run_ids:
        _check_run_id(run_id)
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

    candidates = [_score_candidate(client, run_id, date) for run_id in run_ids]
    # min() returns the first of equal items, so a tie goes to the earlier run id.
    best = min(candidates, key=lambda candidate: candidate.log_loss)
    champion = _champion(client)
    champion_train_end = None if champion is None else _champion_train_end(client, champion)
    promote, champion_log_loss, reason = _decide(date, best, champion, champion_train_end)

    registered_version: int | None = None
    if promote and (champion is None or champion.run_id != best.run_id):
        registered_version = int(mlflow.register_model(best.model_uri, REGISTERED_MODEL).version)
        client.set_registered_model_alias(REGISTERED_MODEL, CHAMPION_ALIAS, registered_version)

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
    _record_decision(client, best.run_id, date, result, registered_version)
    for candidate in candidates:
        if candidate is not best:
            _record_not_best(client, candidate, best.run_id)

    logger.info(
        "promotion for %s: %s (%s); best candidate %s log loss %.6f, champion %s log loss %s, "
        "registered %s",
        date,
        "promoted" if promote else "not promoted",
        reason,
        best.run_id,
        best.log_loss,
        "none" if result.champion_version is None else f"v{result.champion_version}",
        "n/a" if champion_log_loss is None else f"{champion_log_loss:.6f}",
        "not registered" if registered_version is None else f"v{registered_version}",
    )
    return result
