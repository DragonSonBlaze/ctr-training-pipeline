import numpy as np
import pytest

from ctr_pipeline.calibration import StreamingScores, calibration_rel_error, reliability_table
from ctr_pipeline.errors import PipelineError
from ctr_pipeline.models.evaluate import compute_metrics


def test_reliability_rows_sum_to_input():
    rng = np.random.default_rng(1)
    predictions = rng.uniform(0.05, 0.4, size=10_000)
    labels = (rng.random(10_000) < predictions).astype(np.float64)

    table = reliability_table(labels, predictions)

    assert list(table.columns) == ["bin", "rows", "mean_pred", "actual_ctr"]
    assert int(table["rows"].sum()) == 10_000


def test_reliability_of_a_calibrated_predictor():
    rng = np.random.default_rng(7)
    predictions = rng.uniform(0.0, 1.0, size=50_000)
    labels = (rng.random(50_000) < predictions).astype(np.float64)

    table = reliability_table(labels, predictions)

    gaps = (table["mean_pred"] - table["actual_ctr"]).abs()
    assert len(table) == 10
    assert (gaps < 0.02).all()


def test_streaming_scores_equal_one_shot_metrics():
    rng = np.random.default_rng(3)
    predictions = rng.uniform(0.01, 0.6, size=3_000)
    labels = (rng.random(3_000) < predictions).astype(np.float64)
    scores = StreamingScores()
    for start in range(0, 3_000, 1_000):
        scores.add(labels[start : start + 1_000], predictions[start : start + 1_000])

    streamed = scores.metrics()
    one_shot = compute_metrics(labels, predictions)

    for key, value in one_shot.items():
        assert streamed[key] == value
    assert streamed["calibration_rel_error"] == calibration_rel_error(
        one_shot["mean_pred"], one_shot["actual_ctr"]
    )


def test_rel_error_formula():
    assert calibration_rel_error(0.3, 0.2) == pytest.approx(0.5)
    assert calibration_rel_error(0.18, 0.2) == pytest.approx(0.1)
    with pytest.raises(PipelineError):
        calibration_rel_error(0.3, 0.0)
