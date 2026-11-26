"""Feature drift (Wasserstein distance) and the Grafana alert rules that read it.

The distance tests on plain numbers need no Spark. The quantile tests run on a small
Spark frame built in memory, so no test reads data/.
"""

import random

import pytest
import yaml

from ctr_pipeline.config import (
    ALERT_CALIBRATION_REL_ERROR,
    ALERT_MIN_ROWS,
    ALERT_STEP_SECONDS,
    DRIFT_ALERT_THRESHOLD,
    FIRST_TRAINING_DAY,
    REPO_ROOT,
)
from ctr_pipeline.drift import (
    drift_for_date,
    feature_distance,
    quantile_probabilities,
    wasserstein_from_quantiles,
)
from ctr_pipeline.errors import SkipStep

ALERT_RULES_PATH = (
    REPO_ROOT / "docker" / "grafana" / "provisioning" / "alerting" / "ctr-alerts.yaml"
)


def _frame(spark, values: list[float]):
    return spark.createDataFrame([(float(v),) for v in values], "x double")


def test_probabilities_are_evenly_spaced_inside_zero_one():
    probabilities = quantile_probabilities(99)
    assert len(probabilities) == 99
    assert probabilities[0] == pytest.approx(0.01)
    assert probabilities[-1] == pytest.approx(0.99)


def test_distance_of_identical_quantiles_is_zero():
    assert wasserstein_from_quantiles([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]) == 0


def test_distance_of_a_constant_shift_is_the_shift():
    reference = [0.10, 0.15, 0.20]
    shifted = [q + 0.03 for q in reference]
    assert wasserstein_from_quantiles(reference, shifted) == pytest.approx(0.03)


def test_distance_hand_computed():
    # |0.2-0.1| + |0.2-0.2| + |0.1-0.4| = 0.4, over 3 quantiles
    assert wasserstein_from_quantiles([0.1, 0.2, 0.4], [0.2, 0.2, 0.1]) == pytest.approx(0.4 / 3)


def test_distance_rejects_lists_of_different_length():
    with pytest.raises(ValueError):
        wasserstein_from_quantiles([0.1, 0.2], [0.1])


def test_distance_rejects_empty_lists():
    with pytest.raises(ValueError):
        wasserstein_from_quantiles([], [])


def test_same_distribution_small_distance(spark):
    reference_rng, current_rng = random.Random(1), random.Random(2)
    reference = _frame(spark, [0.2 * reference_rng.random() for _ in range(10_000)])
    current = _frame(spark, [0.2 * current_rng.random() for _ in range(10_000)])
    assert feature_distance(reference, current, "x") < 0.005


def test_shifted_distribution_distance_matches_shift(spark):
    reference_rng, current_rng = random.Random(3), random.Random(4)
    reference = _frame(spark, [0.2 * reference_rng.random() for _ in range(10_000)])
    current = _frame(spark, [0.05 + 0.2 * current_rng.random() for _ in range(10_000)])
    assert feature_distance(reference, current, "x") == pytest.approx(0.05, abs=0.005)


def test_tiny_move_of_a_tied_block_stays_small(spark):
    # The case that broke PSI: 80% of rows share one value, and that value moves by 0.003.
    rng = random.Random(5)
    tail = [0.3 * rng.random() for _ in range(2_000)]
    reference = _frame(spark, [0.170] * 8_000 + tail)
    current = _frame(spark, [0.173] * 8_000 + tail)
    assert feature_distance(reference, current, "x") < DRIFT_ALERT_THRESHOLD


def test_first_training_day_skips():
    with pytest.raises(SkipStep):
        drift_for_date(FIRST_TRAINING_DAY)


def _alert_rules() -> dict[str, dict]:
    document = yaml.safe_load(ALERT_RULES_PATH.read_text())
    rules = [rule for group in document["groups"] for rule in group["rules"]]
    return {rule["uid"]: rule for rule in rules}


def test_alert_thresholds_match_config():
    rules = _alert_rules()
    expected = {
        "ctr-step-run-time": (ALERT_STEP_SECONDS, "gt"),
        "ctr-step-failed": (0, "gt"),
        "ctr-row-count": (ALERT_MIN_ROWS, "lt"),
        "ctr-feature-drift": (DRIFT_ALERT_THRESHOLD, "gt"),
        "ctr-calibration": (ALERT_CALIBRATION_REL_ERROR, "gt"),
    }
    assert set(rules) == set(expected)
    for uid, (threshold, comparison) in expected.items():
        condition = next(query for query in rules[uid]["data"] if query["refId"] == "C")
        evaluator = condition["model"]["conditions"][0]["evaluator"]
        assert evaluator["type"] == comparison, uid
        assert evaluator["params"][0] == pytest.approx(threshold), uid
