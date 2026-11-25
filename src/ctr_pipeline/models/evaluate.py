"""Metrics for one evaluation day: AUC (ranking) and log loss (calibration).

Log loss is the metric that decides promotion, because predicted click rates
are used as probabilities. AUC only measures ranking, so it is reported but
not used to choose a model.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

# Predictions are clipped so that a 0 or 1 prediction on the wrong side costs
# a finite log loss (about 13.8 at the clip) instead of infinity.
EPS = 1e-6


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    labels = np.asarray(y_true, dtype=np.float64)
    probabilities = np.clip(np.asarray(y_pred, dtype=np.float64), EPS, 1 - EPS)
    has_both_classes = np.unique(labels).size == 2
    return {
        "auc": float(roc_auc_score(labels, probabilities)) if has_both_classes else None,
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "mean_pred": float(probabilities.mean()),
        "actual_ctr": float(labels.mean()),
        "rows": int(labels.size),
    }
