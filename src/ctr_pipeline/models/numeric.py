"""Numeric click-history features, scaled the same way for every model.

Click rates become log-odds and impression counts become log1p, then each column is
standardised with the mean and standard deviation of the training rows. The scikit-learn
baseline and the Keras model share this code, so a comparison between them is about the
model, not the inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ctr_pipeline.contracts import NUMERIC_FEATURES

# Same clipping value as evaluate.EPS, so a rate of exactly 0 or 1 gives a finite log-odds.
EPS = 1e-6
# A feature with no spread gets a standard deviation of 1, so dividing by it is safe.
STD_FLOOR = 1e-6


def raw_numeric(frame: pd.DataFrame) -> np.ndarray:
    """Unscaled features, float64, one column per name in NUMERIC_FEATURES."""
    columns = []
    for name in NUMERIC_FEATURES:
        values = frame[name].to_numpy(dtype=np.float64)
        if name.endswith("_ctr_smooth"):
            clipped = np.clip(values, EPS, 1 - EPS)
            columns.append(np.log(clipped / (1 - clipped)))
        else:
            columns.append(np.log1p(values))
    return np.column_stack(columns)


@dataclass
class Standardizer:
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, raw: np.ndarray) -> Standardizer:
        self.mean = raw.mean(axis=0)
        std = raw.std(axis=0)
        std[std < STD_FLOOR] = 1.0
        self.std = std
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer.transform() called before fit()")
        return ((raw - self.mean) / self.std).astype(np.float32)
