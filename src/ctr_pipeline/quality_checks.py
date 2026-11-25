"""Stage 1 gate: data quality checks on one day of hourly data.

A day that fails is not used for features or training. Every check runs from
one aggregation over the day, so the cost is two Spark jobs whatever the number of checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, TimestampType

from ctr_pipeline.config import (
    CLICK_RATE_RANGE,
    MAX_NULL_FRACTION,
    MIN_ROWS_PER_DAY,
    MIN_ROWS_PER_HOUR,
    get_settings,
)
from ctr_pipeline.contracts import PARTITION_COLUMNS, RAW_COLUMNS, RAW_SCHEMA
from ctr_pipeline.spark import get_spark

logger = logging.getLogger(__name__)

EXPECTED_HOURS = 24
REQUIRED_COLUMNS = ("id", "click", "hour")
CHECK_NAMES = (
    "schema_matches_contract",
    "hours_present",
    "min_rows_per_hour",
    "min_rows_per_day",
    "required_columns_not_null",
    "max_null_fraction",
    "click_values_binary",
    "click_rate_in_range",
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    observed: float | None
    threshold: str


@dataclass(frozen=True)
class QualityThresholds:
    min_rows_per_hour: int = MIN_ROWS_PER_HOUR
    min_rows_per_day: int = MIN_ROWS_PER_DAY
    click_rate_range: tuple[float, float] = CLICK_RATE_RANGE
    max_null_fraction: float = MAX_NULL_FRACTION


def read_day(spark: SparkSession, hourly_dir_path: str, date: str) -> DataFrame:
    """Rows of one day. The filter on the partition column lets Spark skip other days' folders."""
    return spark.read.parquet(hourly_dir_path).filter(F.col("dt").cast("string") == date)


def _null_count_expr(column: str) -> Any:
    return F.count(F.when(F.col(column).isNull(), 1)).alias(f"nulls_{column}")


def _day_aggregates(day: DataFrame) -> dict[str, Any]:
    """Every per-day number the checks need, computed in one aggregation."""
    not_binary = F.col("click").isNotNull() & ~F.col("click").isin(0, 1)
    exprs = [
        F.count(F.lit(1)).alias("rows"),
        F.count(F.when(not_binary, 1)).alias("click_not_binary"),
        F.avg("click").alias("click_mean"),
        *[_null_count_expr(column) for column in RAW_COLUMNS],
    ]
    return day.agg(*exprs).collect()[0].asDict()


def _hour_row_counts(day: DataFrame) -> list[int]:
    return [row["count"] for row in day.groupBy("hr").count().collect()]


def _schema_mismatches(schema: StructType) -> int:
    """Columns missing or with a type other than the contract (hour_ts is a timestamp)."""
    actual = {field.name: field.dataType for field in schema.fields}
    expected = {field.name: field.dataType for field in RAW_SCHEMA.fields}
    expected["hour_ts"] = TimestampType()
    mismatches = sum(1 for name, dtype in expected.items() if actual.get(name) != dtype)
    # dt and hr come from the folder names, so only their presence is checked.
    mismatches += sum(1 for name in PARTITION_COLUMNS if name not in actual)
    return mismatches


def _threshold_texts(thresholds: QualityThresholds) -> dict[str, str]:
    low, high = thresholds.click_rate_range
    return {
        "schema_matches_contract": "0 mismatches",
        "hours_present": str(EXPECTED_HOURS),
        "min_rows_per_hour": f">= {thresholds.min_rows_per_hour}",
        "min_rows_per_day": f">= {thresholds.min_rows_per_day}",
        "required_columns_not_null": "0",
        "max_null_fraction": f"<= {thresholds.max_null_fraction}",
        "click_values_binary": "0",
        "click_rate_in_range": f"[{low}, {high}]",
    }


def _build_results(
    aggregates: dict[str, Any],
    hour_counts: list[int],
    schema_mismatches: int,
    thresholds: QualityThresholds,
) -> list[CheckResult]:
    """Turn the day's measurements into the eight named results, in CHECK_NAMES order."""
    texts = _threshold_texts(thresholds)
    low, high = thresholds.click_rate_range
    rows = aggregates["rows"]
    hours = len(hour_counts)
    smallest_hour = min(hour_counts)
    required_nulls = sum(aggregates[f"nulls_{c}"] for c in REQUIRED_COLUMNS)
    max_null = max(aggregates[f"nulls_{c}"] / rows for c in RAW_COLUMNS)
    click_mean = aggregates["click_mean"]
    not_binary = aggregates["click_not_binary"]
    return [
        CheckResult(
            "schema_matches_contract",
            schema_mismatches == 0,
            float(schema_mismatches),
            texts["schema_matches_contract"],
        ),
        CheckResult(
            "hours_present",
            hours == EXPECTED_HOURS,
            float(hours),
            texts["hours_present"],
        ),
        CheckResult(
            "min_rows_per_hour",
            smallest_hour >= thresholds.min_rows_per_hour,
            float(smallest_hour),
            texts["min_rows_per_hour"],
        ),
        CheckResult(
            "min_rows_per_day",
            rows >= thresholds.min_rows_per_day,
            float(rows),
            texts["min_rows_per_day"],
        ),
        CheckResult(
            "required_columns_not_null",
            required_nulls == 0,
            float(required_nulls),
            texts["required_columns_not_null"],
        ),
        CheckResult(
            "max_null_fraction",
            max_null <= thresholds.max_null_fraction,
            float(max_null),
            texts["max_null_fraction"],
        ),
        CheckResult(
            "click_values_binary",
            not_binary == 0,
            float(not_binary),
            texts["click_values_binary"],
        ),
        CheckResult(
            "click_rate_in_range",
            click_mean is not None and low <= click_mean <= high,
            None if click_mean is None else float(click_mean),
            texts["click_rate_in_range"],
        ),
    ]


def _all_failed(thresholds: QualityThresholds) -> list[CheckResult]:
    """Results for a day with no rows: nothing was measured, so every check fails."""
    texts = _threshold_texts(thresholds)
    return [CheckResult(name, False, None, texts[name]) for name in CHECK_NAMES]


def run_quality_checks(
    date: str, thresholds: QualityThresholds = QualityThresholds()
) -> list[CheckResult]:
    settings = get_settings()
    spark = get_spark("quality-check")
    day = read_day(spark, str(settings.hourly_dir), date)
    aggregates = _day_aggregates(day)
    if aggregates["rows"] == 0:
        logger.warning("no rows for %s in %s", date, settings.hourly_dir)
        return _all_failed(thresholds)
    hour_counts = _hour_row_counts(day)
    mismatches = _schema_mismatches(day.schema)
    return _build_results(aggregates, hour_counts, mismatches, thresholds)
