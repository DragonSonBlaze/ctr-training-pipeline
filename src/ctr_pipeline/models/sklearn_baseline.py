"""scikit-learn logistic regression on the same inputs as the Keras model.

Each categorical field is hashed to one slot, and each row gets a 1 in its 20 slots.
The 6 numeric features sit in the last 6 columns. The matrix is sparse, so the slot
columns cost nothing when a slot is unused. The L2 penalty keeps weights of rare slots
small, and the intercept is not penalised, so the model is calibrated on its own
training rows.
"""

from __future__ import annotations

import logging

import mlflow.pyfunc
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from ctr_pipeline.config import SK_C, SK_MAX_ITER, SK_NUM_BUCKETS
from ctr_pipeline.contracts import CATEGORICAL_FIELDS, NUMERIC_FEATURES
from ctr_pipeline.models.hashing import hash_frame
from ctr_pipeline.models.numeric import Standardizer, raw_numeric

logger = logging.getLogger(__name__)

SOLVER = "lbfgs"


def design_matrix(
    frame: pd.DataFrame, standardizer: Standardizer, num_buckets: int
) -> sparse.csr_matrix:
    """One row per frame row: a 1 in each hashed categorical slot, then the scaled numerics."""
    slots = hash_frame(frame, CATEGORICAL_FIELDS, num_buckets)
    numeric = standardizer.transform(raw_numeric(frame))
    rows, fields = slots.shape
    width = fields + len(NUMERIC_FEATURES)

    indices = np.empty((rows, width), dtype=np.int64)
    indices[:, :fields] = slots
    indices[:, fields:] = num_buckets + np.arange(len(NUMERIC_FEATURES))

    data = np.ones((rows, width), dtype=np.float64)
    data[:, fields:] = numeric

    indptr = np.arange(0, rows * width + 1, width, dtype=np.int64)
    matrix = sparse.csr_matrix(
        (data.ravel(), indices.ravel(), indptr),
        shape=(rows, num_buckets + len(NUMERIC_FEATURES)),
    )
    # Two fields can hash to the same slot in one row. Summing them keeps the matrix in
    # canonical form, so each slot appears once per row.
    matrix.sum_duplicates()
    return matrix


class SklearnLogisticModel:
    model_type = "sklearn_logistic"
    training_columns: list[str] = CATEGORICAL_FIELDS + NUMERIC_FEATURES + ["click"]
    input_columns: list[str] = CATEGORICAL_FIELDS + NUMERIC_FEATURES

    def __init__(
        self,
        num_buckets: int = SK_NUM_BUCKETS,
        c: float = SK_C,
        max_iter: int = SK_MAX_ITER,
    ) -> None:
        self.num_buckets = num_buckets
        self.c = c
        self.max_iter = max_iter
        self.standardizer: Standardizer | None = None
        self.estimator: LogisticRegression | None = None
        self.train_click_rate: float | None = None
        self.n_iter: int | None = None

    def design_matrix(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        if self.standardizer is None:
            raise RuntimeError("design_matrix() called before fit()")
        return design_matrix(frame, self.standardizer, self.num_buckets)

    def fit(self, train: pd.DataFrame) -> None:
        if train.empty:
            raise ValueError("cannot fit on an empty training sample")
        labels = train["click"].to_numpy()
        if np.unique(labels).size < 2:
            raise ValueError("training sample has a single click class, cannot fit")

        self.standardizer = Standardizer().fit(raw_numeric(train))
        matrix = self.design_matrix(train)
        estimator = LogisticRegression(solver=SOLVER, C=self.c, max_iter=self.max_iter)
        estimator.fit(matrix, labels)

        self.estimator = estimator
        self.train_click_rate = float(labels.mean())
        self.n_iter = int(estimator.n_iter_[0])
        logger.info(
            "fitted sklearn logistic regression on %d rows in %d iterations",
            len(labels),
            self.n_iter,
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("predict() called before fit()")
        return _predict_proba(self.estimator, self.design_matrix(frame))

    def params(self) -> dict[str, float | int | str]:
        return {
            "model_type": self.model_type,
            "num_buckets": self.num_buckets,
            "c": self.c,
            "max_iter": self.max_iter,
            "solver": SOLVER,
            "categorical_fields": ",".join(CATEGORICAL_FIELDS),
            "numeric_features": ",".join(NUMERIC_FEATURES),
        }

    def fitted_values(self) -> dict[str, float]:
        if self.estimator is None or self.train_click_rate is None or self.n_iter is None:
            raise RuntimeError("fitted_values() called before fit()")
        return {
            "train_click_rate": self.train_click_rate,
            "n_iter": float(self.n_iter),
            "intercept": float(self.estimator.intercept_[0]),
        }

    def to_pyfunc(self) -> SklearnLogisticPyfunc:
        if self.estimator is None or self.standardizer is None:
            raise RuntimeError("to_pyfunc() called before fit()")
        return SklearnLogisticPyfunc(self.estimator, self.standardizer, self.num_buckets)


def _predict_proba(estimator: LogisticRegression, matrix: sparse.csr_matrix) -> np.ndarray:
    return estimator.predict_proba(matrix)[:, 1].astype(np.float64)


class SklearnLogisticPyfunc(mlflow.pyfunc.PythonModel):
    """MLflow wrapper: pickles the fitted estimator and scaler, and predicts with the same code."""

    def __init__(
        self, estimator: LogisticRegression, standardizer: Standardizer, num_buckets: int
    ) -> None:
        self.estimator = estimator
        self.standardizer = standardizer
        self.num_buckets = num_buckets

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        matrix = design_matrix(model_input, self.standardizer, self.num_buckets)
        return _predict_proba(self.estimator, matrix)
