"""Feature drift: how far each smoothed click-rate feature moved since the day before.

The drift score is the Wasserstein-1 distance between day D-1 and day D: the average
distance, in click-rate units, that the values had to move. It is computed from
DRIFT_QUANTILES evenly spaced quantiles of each day as the mean of |q_D - q_(D-1)|.
A score of 0.02 means the values moved 2 click-rate points on average.

Why not PSI: most rows of these features sit on a few identical values (a handful of
huge apps and sites share almost the same smoothed rate). PSI bins the reference day,
so when that block of rows shifts by 0.003 across a bin edge, PSI jumps to 1 or more.
On this data it was above 0.25 on every day. The distance moves smoothly instead.

Input:  training sets of D-1 and D (data/training/v1/dt=YYYY-MM-DD/)
Output: feature -> distance for each of DRIFT_FEATURES, stored by the CLI in feature_drift
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ctr_pipeline.config import (
    DRIFT_QUANTILES,
    FIRST_TRAINING_DAY,
    get_settings,
    previous_day,
)
from ctr_pipeline.contracts import DRIFT_FEATURES
from ctr_pipeline.errors import MissingInputError, SkipStep
from ctr_pipeline.features import load_training_day

logger = logging.getLogger(__name__)

# Relative error of Spark's approximate quantiles, far below the 0.02 alert threshold.
QUANTILE_ERROR = 0.001


def quantile_probabilities(count: int = DRIFT_QUANTILES) -> list[float]:
    """count probabilities spread evenly inside (0, 1): 0.01 .. 0.99 for 99."""
    return [i / (count + 1) for i in range(1, count + 1)]


def wasserstein_from_quantiles(reference: Sequence[float], current: Sequence[float]) -> float:
    """Mean absolute difference of matching quantiles of two distributions.

    In one dimension, Wasserstein-1 is the integral of |F1^-1(p) - F2^-1(p)| over p,
    and this is that integral sampled at evenly spaced p.
    """
    if len(reference) != len(current):
        raise ValueError(f"quantile lists differ in length: {len(reference)} and {len(current)}")
    if not reference:
        raise ValueError("no quantiles given")
    total = sum(abs(cur - ref) for ref, cur in zip(reference, current, strict=True))
    return total / len(reference)


def feature_quantiles(df: DataFrame, feature: str) -> list[float]:
    """DRIFT_QUANTILES quantiles of the non-null values of one feature."""
    values = df.select(feature).filter(F.col(feature).isNotNull())
    quantiles = values.approxQuantile(feature, quantile_probabilities(), QUANTILE_ERROR)
    if not quantiles:
        raise MissingInputError(f"no non-null values in {feature}")
    return quantiles


def feature_distance(reference_df: DataFrame, current_df: DataFrame, feature: str) -> float:
    """Wasserstein-1 distance of one feature between the reference and the current day."""
    return wasserstein_from_quantiles(
        feature_quantiles(reference_df, feature),
        feature_quantiles(current_df, feature),
    )


def _check_manifests(days: list[str]) -> None:
    manifests_dir = get_settings().manifests_dir
    missing = [d for d in days if not (manifests_dir / f"{d}.json").exists()]
    if missing:
        raise MissingInputError(f"no training set manifest for: {', '.join(missing)}")


def drift_for_date(date: str) -> dict[str, float]:
    """Distance of each DRIFT_FEATURES column, comparing date with the day before it.

    The first training day has no earlier training set, so there is nothing to compare
    with and the step is skipped.
    """
    if date == FIRST_TRAINING_DAY:
        raise SkipStep(f"{date} is the first training day, no earlier training set to compare")
    reference_day = previous_day(date)
    _check_manifests([reference_day, date])

    reference = load_training_day(reference_day).select(*DRIFT_FEATURES).cache()
    current = load_training_day(date).select(*DRIFT_FEATURES).cache()
    try:
        return {
            feature: feature_distance(reference, current, feature) for feature in DRIFT_FEATURES
        }
    finally:
        reference.unpersist()
        current.unpersist()
