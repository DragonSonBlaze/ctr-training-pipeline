"""The v1 model: a historical click-rate estimate with no learned weights.

A row's click probability comes from the click history of its site, app and
device. Each estimate is shrunk toward the next broader one, so a key with few
impressions falls back to the broader rate instead of an unstable ratio.
"""

from __future__ import annotations

from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd

from ctr_pipeline.config import MODEL_BETA
from ctr_pipeline.contracts import MODEL_INPUT_COLUMNS


def _shrink(
    clicks: np.ndarray, impressions: np.ndarray, prior: np.ndarray, beta: float
) -> np.ndarray:
    """Click rate of a key, pulled toward its prior by beta pseudo-impressions."""
    return (clicks + beta * prior) / (impressions + beta)


def predict_click_rate(frame: pd.DataFrame, p0: float, beta: float) -> np.ndarray:
    """Chain site -> app -> device, each level shrunk toward the one above it."""
    site = _shrink(
        frame["site_prior_clicks"].to_numpy(dtype=np.float64),
        frame["site_prior_impr"].to_numpy(dtype=np.float64),
        np.full(len(frame), p0),
        beta,
    )
    app = _shrink(
        frame["app_prior_clicks"].to_numpy(dtype=np.float64),
        frame["app_prior_impr"].to_numpy(dtype=np.float64),
        site,
        beta,
    )
    return _shrink(
        frame["device_prior_clicks"].to_numpy(dtype=np.float64),
        frame["device_prior_impr"].to_numpy(dtype=np.float64),
        app,
        beta,
    )


class HistoricalCtrModel:
    model_type = "historical_ctr"
    training_columns: list[str] = ["click"]
    input_columns: list[str] = MODEL_INPUT_COLUMNS

    def __init__(self, beta: float = MODEL_BETA) -> None:
        self.beta = beta
        self.p0: float | None = None

    def fit(self, train: pd.DataFrame) -> None:
        if train.empty:
            raise ValueError("cannot fit on an empty training sample")
        self.p0 = float(train["click"].mean())

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.p0 is None:
            raise RuntimeError("predict() called before fit()")
        return predict_click_rate(frame, self.p0, self.beta)

    def params(self) -> dict[str, float | int | str]:
        return {"beta": self.beta}

    def fitted_values(self) -> dict[str, float]:
        if self.p0 is None:
            raise RuntimeError("fitted_values() called before fit()")
        return {"p0": self.p0}

    def to_pyfunc(self) -> HistoricalCtrPyfunc:
        if self.p0 is None:
            raise RuntimeError("to_pyfunc() called before fit()")
        return HistoricalCtrPyfunc(self.p0, self.beta)


class HistoricalCtrPyfunc(mlflow.pyfunc.PythonModel):
    """MLflow wrapper: the fitted values are stored, and predictions are rebuilt from them."""

    def __init__(self, p0: float, beta: float) -> None:
        self.p0 = p0
        self.beta = beta

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> np.ndarray:
        model = HistoricalCtrModel(beta=self.beta)
        model.p0 = self.p0
        return model.predict(model_input)
