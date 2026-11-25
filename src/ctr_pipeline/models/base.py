"""The interface every click model implements, so the DAG, promotion and monitoring
do not change when the model does.
"""

from __future__ import annotations

from typing import Protocol

import mlflow.pyfunc
import numpy as np
import pandas as pd


class CtrModel(Protocol):
    model_type: str
    # Columns fit() reads. Always includes "click".
    training_columns: list[str]
    # Columns predict() reads.
    input_columns: list[str]

    def fit(self, train: pd.DataFrame) -> None: ...

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Click probabilities in [0, 1], one per row of frame."""
        ...

    def params(self) -> dict[str, float | int | str]:
        """Hyperparameters, logged to MLflow as run params."""
        ...

    def fitted_values(self) -> dict[str, float]:
        """Values learned by fit(), logged to MLflow as metrics prefixed with fitted_."""
        ...

    def to_pyfunc(self) -> mlflow.pyfunc.PythonModel:
        """An MLflow pyfunc that predicts like this fitted model, for logging and loading."""
        ...
