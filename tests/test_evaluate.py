import math

import numpy as np
import pytest

from ctr_pipeline.models.evaluate import compute_metrics


def test_perfect_ranking_auc_is_one():
    metrics = compute_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert metrics["auc"] == 1.0


def test_single_class_auc_is_none():
    metrics = compute_metrics(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))
    assert metrics["auc"] is None
    assert metrics["log_loss"] is not None


def test_log_loss_is_clipped():
    metrics = compute_metrics(np.array([1]), np.array([0.0]))
    assert math.isfinite(metrics["log_loss"])
    assert metrics["log_loss"] == pytest.approx(-math.log(1e-6))
