"""Training loop for one data date: fit on earlier days, evaluate on the date, log to MLflow.

Day D is never used for fitting: training uses the training sets of the days
before D, and the evaluation uses D alone.

Each trained model is logged to its MLflow run as the artifact `model` and is not
registered here. Only promotion registers a model, and only when it becomes champion.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from ctr_pipeline import monitoring
from ctr_pipeline.config import (
    DEFAULT_MODEL_TYPE,
    EVAL_BATCH_ROWS,
    FEATURE_VERSION,
    FIRST_TRAINING_DAY,
    MLFLOW_EXPERIMENT,
    TRAIN_SAMPLE_FRACTION,
    TRAIN_SAMPLE_SEED,
    day_range,
    get_settings,
    previous_day,
)
from ctr_pipeline.errors import MissingInputError, SkipStep
from ctr_pipeline.models.base import CtrModel
from ctr_pipeline.models.embedding_nn import HashedEmbeddingModel
from ctr_pipeline.models.evaluate import compute_metrics
from ctr_pipeline.models.historical_ctr import HistoricalCtrModel
from ctr_pipeline.models.logistic import HashedLogisticModel
from ctr_pipeline.models.sklearn_baseline import SklearnLogisticModel

logger = logging.getLogger(__name__)

MODEL_TYPES: dict[str, Callable[[], CtrModel]] = {
    HistoricalCtrModel.model_type: HistoricalCtrModel,
    HashedLogisticModel.model_type: HashedLogisticModel,
    HashedEmbeddingModel.model_type: HashedEmbeddingModel,
    SklearnLogisticModel.model_type: SklearnLogisticModel,
}
PYFUNC_PIP_REQUIREMENTS = ["numpy", "pandas", "torch"]


@dataclass(frozen=True)
class TrainResult:
    run_id: str
    dataset_version: str
    metrics: dict[str, float | None]
    train_rows: int


def training_days_for(date: str) -> list[str]:
    """Every day with a training set before date, from the first training day on."""
    return day_range(FIRST_TRAINING_DAY, previous_day(date))


def _manifest_path(day: str, manifests_dir: Path) -> Path:
    return manifests_dir / f"{day}.json"


def dataset_version(days: list[str]) -> str:
    """Name the training data by its days and by the content of their manifests.

    A rebuilt day changes its manifest (created_at), so the version changes with it.
    """
    if not days:
        raise ValueError("a dataset version needs at least one day")
    manifests_dir = get_settings().manifests_dir
    digest = hashlib.sha256()
    for day in days:
        digest.update(_manifest_path(day, manifests_dir).read_bytes())
    return f"{FEATURE_VERSION}:{days[0]}..{days[-1]}:{digest.hexdigest()[:12]}"


def load_training_frame(days: list[str], columns: list[str]) -> pd.DataFrame:
    """Read each day one at a time and keep a uniform sample of its rows.

    The sample is uniform over all rows, never only negatives: keeping only negatives
    would bias the predicted click rate and break the calibration check. Each day gets
    its own seed, so a rerun samples the same rows.
    """
    training_dir = get_settings().training_dir
    samples = []
    for index, day in enumerate(days):
        frame = pd.read_parquet(
            training_dir / f"dt={day}", columns=columns, dtype_backend="pyarrow"
        )
        samples.append(
            frame.sample(frac=TRAIN_SAMPLE_FRACTION, random_state=TRAIN_SAMPLE_SEED + index)
        )
    return pd.concat(samples, ignore_index=True)


def load_eval_frame(date: str, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(get_settings().training_dir / f"dt={date}", columns=columns)


def iter_day_batches(
    date: str, columns: list[str], batch_rows: int = EVAL_BATCH_ROWS
) -> Iterator[pd.DataFrame]:
    """One day's rows in batches of at most batch_rows, so a whole day never sits in memory."""
    dataset = ds.dataset(get_settings().training_dir / f"dt={date}")
    for batch in dataset.to_batches(columns=columns, batch_size=batch_rows):
        yield batch.to_pandas()


def predict_day(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    date: str,
    columns: list[str],
    batch_rows: int = EVAL_BATCH_ROWS,
) -> tuple[np.ndarray, np.ndarray]:
    """Run predict_fn on every batch of one day. Returns (true clicks, predictions)."""
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for batch in iter_day_batches(date, columns, batch_rows):
        labels.append(batch["click"].to_numpy())
        predictions.append(np.asarray(predict_fn(batch), dtype=np.float64))
    if not labels:
        raise MissingInputError(f"training set for {date} has no rows")
    return np.concatenate(labels), np.concatenate(predictions)


def _check_manifests(days: list[str], date: str) -> None:
    manifests_dir = get_settings().manifests_dir
    missing = [d for d in [*days, date] if not _manifest_path(d, manifests_dir).exists()]
    if missing:
        raise MissingInputError(f"no training set manifest for: {', '.join(missing)}")


def _build_model(model_type: str) -> CtrModel:
    if model_type not in MODEL_TYPES:
        raise ValueError(
            f"unknown model type {model_type!r}, expected one of {sorted(MODEL_TYPES)}"
        )
    return MODEL_TYPES[model_type]()


def _log_to_mlflow(
    model: CtrModel,
    date: str,
    days: list[str],
    dataset: str,
    metrics: dict[str, float | None],
    train_rows: int,
    eval_rows: int,
) -> str:
    """Log one run: params, metrics and the pyfunc model as a run artifact. Returns the run id.

    The model is not registered: the registry holds only models that became champion.
    """
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"{model.model_type}-{date}") as run:
        run_id = run.info.run_id
        mlflow.set_tags(
            {
                "data_date": date,
                "model_type": model.model_type,
                "dataset_version": dataset,
                "feature_version": FEATURE_VERSION,
            }
        )
        mlflow.log_params(
            {
                **model.params(),
                "train_start": days[0],
                "train_end": days[-1],
                "train_days": len(days),
                "sample_fraction": TRAIN_SAMPLE_FRACTION,
                "sample_seed": TRAIN_SAMPLE_SEED,
            }
        )
        values: dict[str, float] = {k: float(v) for k, v in metrics.items() if v is not None}
        values["train_rows"] = float(train_rows)
        values["eval_rows"] = float(eval_rows)
        values.update({f"fitted_{k}": float(v) for k, v in model.fitted_values().items()})
        mlflow.log_metrics(values)

        mlflow.pyfunc.log_model(
            name="model",
            python_model=model.to_pyfunc(),
            pip_requirements=PYFUNC_PIP_REQUIREMENTS,
        )
    return run_id


def _record_monitoring(
    run_id: str,
    date: str,
    model: CtrModel,
    dataset: str,
    metrics: dict[str, float | None],
    train_rows: int,
    eval_rows: int,
) -> None:
    monitoring.record_model_run(
        {
            "mlflow_run_id": run_id,
            "data_date": date,
            "model_type": model.model_type,
            "model_version": None,
            "dataset_version": dataset,
            "train_rows": train_rows,
            "eval_rows": eval_rows,
            "auc": metrics["auc"],
            "log_loss": metrics["log_loss"],
            "mean_pred": metrics["mean_pred"],
            "actual_ctr": metrics["actual_ctr"],
        }
    )


def train_for_date(
    date: str,
    model_type: str = DEFAULT_MODEL_TYPE,
    model: CtrModel | None = None,
) -> TrainResult:
    """Fit on the earlier days, evaluate on date, and log the run without registering a model.

    The model type defaults to DEFAULT_MODEL_TYPE, the one the daily pipeline trains.
    A ready-made model object may be passed instead of a type. The model is available
    as runs:/<run_id>/model; promotion decides whether it is registered.
    """
    if model is None:
        model = _build_model(model_type)
    days = training_days_for(date)
    if not days:
        raise SkipStep(f"no earlier training sets before {date}")
    _check_manifests(days, date)

    dataset = dataset_version(days)
    train = load_training_frame(days, model.training_columns)
    model.fit(train)

    eval_columns = [*model.input_columns, "click"]
    y_true, y_pred = predict_day(model.predict, date, eval_columns)
    metrics = compute_metrics(y_true, y_pred)
    train_rows = len(train)
    eval_rows = int(y_true.size)

    run_id = _log_to_mlflow(model, date, days, dataset, metrics, train_rows, eval_rows)
    _record_monitoring(run_id, date, model, dataset, metrics, train_rows, eval_rows)
    logger.info(
        "trained %s on %d rows from %s..%s, evaluated on %s (%d rows): auc=%s log_loss=%.6f",
        model.model_type,
        train_rows,
        days[0],
        days[-1],
        date,
        eval_rows,
        metrics["auc"],
        metrics["log_loss"],
    )
    return TrainResult(
        run_id=run_id,
        dataset_version=dataset,
        metrics=metrics,
        train_rows=train_rows,
    )
