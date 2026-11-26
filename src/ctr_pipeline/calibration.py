"""Calibration of predicted click rates: a reliability table and a streaming score accumulator.

Log loss and AUC say how well a model ranks and scores clicks. Calibration says whether the
predicted rates match the clicks that followed. It matters here because the bid is the
predicted click rate times a price, so a model that is 20 percent too high overbids by 20 percent.

The accumulator keeps one float per row for labels and one for predictions, so a test split of
about 8M rows costs about 130 MB. That is why scoring streams batches into it rather than
loading the whole split into a frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ctr_pipeline.errors import PipelineError
from ctr_pipeline.models.evaluate import EPS, compute_metrics


def calibration_rel_error(mean_pred: float, actual_ctr: float) -> float:
    """How far the mean predicted click rate is from the actual one, as a fraction of the actual.

    A value of 0.1 means the model predicts 10 percent too many clicks (or too few).
    """
    if actual_ctr <= 0:
        raise PipelineError("no clicks in the scored rows, so calibration error is undefined")
    return abs(mean_pred - actual_ctr) / actual_ctr


def reliability_table(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Group rows by predicted click rate and compare the mean prediction with the actual rate.

    Bin edges are the deciles of the predictions, so each bin holds about the same number of
    rows. Duplicate edges (many rows with the same prediction) are collapsed, which leaves fewer
    bins rather than empty ones. Columns: bin, rows, mean_pred, actual_ctr.
    """
    labels = np.asarray(y_true, dtype=np.float64)
    predictions = np.asarray(y_pred, dtype=np.float64)
    edges = np.unique(np.quantile(predictions, np.linspace(0.0, 1.0, bins + 1)))
    # The last edge is the largest prediction, so it falls in the last bin, not one past it.
    index = np.clip(
        np.searchsorted(edges, predictions, side="right") - 1,
        0,
        max(edges.size - 2, 0),
    )
    frame = pd.DataFrame({"bin": index, "label": labels, "prediction": predictions})
    table = frame.groupby("bin").agg(
        rows=("label", "size"),
        mean_pred=("prediction", "mean"),
        actual_ctr=("label", "mean"),
    )
    return table.reset_index()


class StreamingScores:
    """Labels and predictions collected batch by batch, scored once at the end."""

    def __init__(self) -> None:
        self._labels: list[np.ndarray] = []
        self._predictions: list[np.ndarray] = []

    def add(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        # Clipped as compute_metrics clips, so the table and the metrics see the same values.
        self._labels.append(np.asarray(y_true, dtype=np.float64))
        self._predictions.append(np.clip(np.asarray(y_pred, dtype=np.float64), EPS, 1 - EPS))

    def _arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._labels:
            raise PipelineError("no scored rows: the split is empty")
        return np.concatenate(self._labels), np.concatenate(self._predictions)

    def metrics(self) -> dict[str, float | None]:
        labels, predictions = self._arrays()
        metrics = compute_metrics(labels, predictions)
        metrics["calibration_rel_error"] = calibration_rel_error(
            metrics["mean_pred"], metrics["actual_ctr"]
        )
        return metrics

    def table(self) -> pd.DataFrame:
        labels, predictions = self._arrays()
        return reliability_table(labels, predictions)
