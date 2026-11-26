"""Experiment sweep: the scikit-learn baseline and one Keras trial per grid point, all tracked in
one MLflow parent run.

Every child trains on the same train sample and is scored on the same validation sample, so
the trials differ only in their hyperparameters. The validation log loss from compute_metrics
picks the best trial. Nothing is registered here: promotion does that.

Input:  data/datasets/<version>/dataset.json plus the daily training sets
Output: MLflow runs only
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass

import keras
import mlflow
import mlflow.pyfunc
import pandas as pd

from ctr_pipeline import config
from ctr_pipeline.config import FEATURE_VERSION, MLFLOW_EXPERIMENT, get_settings
from ctr_pipeline.contracts import CATEGORICAL_FIELDS, NUMERIC_FEATURES
from ctr_pipeline.dataset import load_dataset_manifest, load_split
from ctr_pipeline.models.base import CtrModel
from ctr_pipeline.models.evaluate import compute_metrics
from ctr_pipeline.models.keras_model import KerasEmbeddingModel
from ctr_pipeline.models.sklearn_baseline import SklearnLogisticModel

logger = logging.getLogger(__name__)

TRAINING_COLUMNS = CATEGORICAL_FIELDS + NUMERIC_FEATURES + ["click"]
BASELINE_PIP_REQUIREMENTS = ["numpy", "pandas", "scikit-learn", "scipy"]
TRIAL_PIP_REQUIREMENTS = ["numpy", "pandas", "tensorflow"]
SCORE_NAMES = {
    "val_log_loss": "log_loss",
    "val_auc": "auc",
    "val_mean_pred": "mean_pred",
    "val_actual_ctr": "actual_ctr",
}


@dataclass(frozen=True)
class ChildResult:
    run_id: str
    model_type: str
    params: dict[str, float | int | str]
    val_log_loss: float
    val_auc: float | None
    epochs_run: int | None  # None for the baseline


@dataclass(frozen=True)
class ExperimentResult:
    parent_run_id: str
    dataset_version: str
    baseline: ChildResult
    trials: list[ChildResult]
    best_trial: ChildResult  # lowest val_log_loss among the trials; ties go to the earlier trial


def sweep_grid() -> list[dict[str, float | int]]:
    """Every combination of the sweep lists, with embedding size varying slowest."""
    return [
        {"embedding_dim": dim, "num_buckets": buckets, "learning_rate": rate}
        for dim in config.SWEEP_EMBEDDING_DIMS
        for buckets in config.SWEEP_NUM_BUCKETS
        for rate in config.SWEEP_LEARNING_RATES
    ]


def default_trial_model(trial: dict[str, float | int]) -> CtrModel:
    return KerasEmbeddingModel(**trial)


def _score_values(metrics: dict[str, float | None]) -> dict[str, float]:
    """Validation scores under their logged names. A None metric (AUC on one class) is left out."""
    return {
        logged: float(metrics[name])
        for logged, name in SCORE_NAMES.items()
        if metrics[name] is not None
    }


def _score(model: CtrModel, frame: pd.DataFrame) -> dict[str, float | None]:
    return compute_metrics(frame["click"].to_numpy(), model.predict(frame))


def _run_baseline(
    make_baseline: Callable[[], CtrModel],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    version: str,
    train_days: str,
) -> ChildResult:
    with mlflow.start_run(nested=True, run_name="sklearn-baseline") as run:
        model = make_baseline()
        model.fit(train)
        metrics = _score(model, validation)
        mlflow.set_tags({"role": "baseline", "dataset_version": version})
        mlflow.log_params({**model.params(), "train_days": train_days})
        fitted = {f"fitted_{name}": value for name, value in model.fitted_values().items()}
        mlflow.log_metrics({**_score_values(metrics), **fitted})
        mlflow.pyfunc.log_model(
            name="model",
            python_model=model.to_pyfunc(),
            pip_requirements=BASELINE_PIP_REQUIREMENTS,
        )
        return ChildResult(
            run_id=run.info.run_id,
            model_type=model.model_type,
            params=model.params(),
            val_log_loss=float(metrics["log_loss"]),
            val_auc=metrics["auc"],
            epochs_run=None,
        )


def _log_epoch_history(model: CtrModel) -> None:
    history = getattr(model, "history", None)
    if not history:
        return
    for epoch, (loss, val_loss) in enumerate(
        zip(history["loss"], history["val_loss"], strict=True)
    ):
        mlflow.log_metrics({"loss": loss, "val_loss": val_loss}, step=epoch)


def _run_trial(
    index: int,
    model: CtrModel,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    version: str,
    train_days: str,
) -> ChildResult:
    with mlflow.start_run(nested=True, run_name=f"trial-{index}") as run:
        model.fit(train, validation=validation)
        metrics = _score(model, validation)
        epochs_run = getattr(model, "epochs_run", None)
        _log_epoch_history(model)
        mlflow.set_tags({"role": "trial", "dataset_version": version})
        mlflow.log_params({**model.params(), "train_days": train_days})
        values = _score_values(metrics)
        if epochs_run is not None:
            values["epochs_run"] = float(epochs_run)
        mlflow.log_metrics(values)
        mlflow.pyfunc.log_model(
            name="model",
            python_model=model.to_pyfunc(),
            pip_requirements=TRIAL_PIP_REQUIREMENTS,
        )
        return ChildResult(
            run_id=run.info.run_id,
            model_type=model.model_type,
            params=model.params(),
            val_log_loss=float(metrics["log_loss"]),
            val_auc=metrics["auc"],
            epochs_run=epochs_run,
        )


def run_experiment(
    version: str,
    grid: list[dict] | None = None,
    model_factory: Callable[[dict], CtrModel] | None = None,
    baseline_factory: Callable[[], CtrModel] | None = None,
) -> ExperimentResult:
    trial_grid = sweep_grid() if grid is None else grid
    if not trial_grid:
        raise ValueError("the sweep grid is empty")
    make_trial = model_factory or default_trial_model
    make_baseline = baseline_factory or SklearnLogisticModel

    manifest = load_dataset_manifest(version)
    days = {split: manifest["splits"][split]["days"] for split in config.SPLITS}
    train_days = ",".join(days["train"])
    # Loaded once, so every child sees exactly the same rows.
    train = load_split(version, "train", TRAINING_COLUMNS)
    validation = load_split(version, "validation", TRAINING_COLUMNS)

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"experiment-{version}") as parent:
        parent_id = parent.info.run_id
        mlflow.set_tags(
            {
                "dataset_version": version,
                "feature_version": FEATURE_VERSION,
                "stage": "experiment",
            }
        )
        mlflow.log_params(
            {
                "train_days": train_days,
                "validation_days": ",".join(days["validation"]),
                "test_days": ",".join(days["test"]),
                "train_rows": len(train),
                "validation_rows": len(validation),
                "sample_fraction": config.TRAIN_SAMPLE_FRACTION,
                "validation_sample_fraction": config.VALIDATION_SAMPLE_FRACTION,
                "sweep_grid": json.dumps(trial_grid, sort_keys=True),
            }
        )

        baseline = _run_baseline(make_baseline, train, validation, version, train_days)
        trials = []
        for index, trial in enumerate(trial_grid):
            model = make_trial(trial)
            is_keras = isinstance(model, KerasEmbeddingModel)
            trials.append(_run_trial(index, model, train, validation, version, train_days))
            # Free each trial's graph before the next one is built, so memory does not grow.
            del model
            if is_keras:
                keras.backend.clear_session()

        best = min(trials, key=lambda trial: trial.val_log_loss)
        mlflow.set_tags({"best_trial_run_id": best.run_id, "baseline_run_id": baseline.run_id})
        mlflow.log_metrics(
            {
                "best_val_log_loss": best.val_log_loss,
                "baseline_val_log_loss": baseline.val_log_loss,
            }
        )

    logger.info(
        "experiment %s: baseline val_log_loss=%.6f, best of %d trials val_log_loss=%.6f",
        version,
        baseline.val_log_loss,
        len(trials),
        best.val_log_loss,
    )
    return ExperimentResult(
        parent_run_id=parent_id,
        dataset_version=version,
        baseline=baseline,
        trials=trials,
        best_trial=best,
    )
