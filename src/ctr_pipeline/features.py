"""Stage 2: build per-day training sets with click-history features.

Features per app, site and device (device_ip + device_model) use only hours
strictly before each impression's hour. That cutoff is the leakage rule: an
impression must never see the click it is trying to predict, nor any click
from its own hour, since a live system would not have them yet.

Input:  data/hourly/events/ (hours of FIRST_DAY up to the build date)
Output: data/training/v1/dt=YYYY-MM-DD/ plus data/training/v1/_manifests/YYYY-MM-DD.json
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from ctr_pipeline.config import (
    FEATURE_ALPHA,
    FEATURE_VERSION,
    FIRST_DAY,
    FIRST_TRAINING_DAY,
    LAST_DAY,
    day_range,
    get_settings,
)
from ctr_pipeline.contracts import FEATURE_COLUMNS, RAW_COLUMNS
from ctr_pipeline.errors import MissingInputError, PipelineError
from ctr_pipeline.hourly_load import is_day_complete
from ctr_pipeline.spark import get_spark

logger = logging.getLogger(__name__)

# Column that holds each history key's value. Device history is keyed on a
# derived column, because its key is two raw columns.
KEY_COLUMNS: dict[str, str] = {"app": "app_id", "site": "site_id", "device": "device_key"}
STAT_QUANTILES = [0.5, 0.95]
STAT_RELATIVE_ERROR = 0.01
CUTOFF_RULE = "hours strictly before the row's hour"
PARTITION_COUNT = 4


@dataclass(frozen=True)
class BuildResult:
    date: str
    rows: int
    path: Path
    manifest: dict


def _check_build_date(date: str, require_complete_history: bool) -> None:
    in_range = FIRST_TRAINING_DAY <= date <= LAST_DAY
    if not in_range and require_complete_history:
        raise ValueError(f"{date} is outside {FIRST_TRAINING_DAY}..{LAST_DAY}")


def _check_history_complete(date: str, hourly_dir: Path) -> None:
    incomplete = [d for d in day_range(FIRST_DAY, date) if not is_day_complete(d, hourly_dir)]
    if incomplete:
        raise MissingInputError(f"hourly history incomplete for: {', '.join(incomplete)}")


def _read_events(spark: SparkSession, hourly_dir: Path, date: str) -> DataFrame:
    """Hourly rows from FIRST_DAY to date, with the helper columns the windows need."""
    events = (
        spark.read.parquet(str(hourly_dir))
        .withColumn("dt", F.col("dt").cast("string"))
        .filter(F.col("dt").between(FIRST_DAY, date))
    )
    # Whole hours since the epoch, so "one hour earlier" is hour_idx - 1.
    events = events.withColumn("hour_idx", (F.unix_timestamp("hour_ts") / 3600).cast("long"))
    return events.withColumn(
        "device_key", F.concat_ws("|", F.col("device_ip"), F.col("device_model"))
    )


def _prior_window(partition_col: str | None) -> Any:
    """Window over all earlier hours (strictly before the current one), per key if given."""
    base = Window.partitionBy(partition_col) if partition_col else Window.partitionBy()
    return base.orderBy("hour_idx").rangeBetween(Window.unboundedPreceding, -1)


def _key_history(events: DataFrame, name: str, key_col: str) -> DataFrame:
    """Per key and hour: impressions and clicks from all strictly earlier hours."""
    per_hour = events.groupBy(key_col, "hour_idx").agg(
        F.count(F.lit(1)).alias("impr"), F.sum("click").alias("clicks")
    )
    window = _prior_window(key_col)
    return per_hour.select(
        key_col,
        "hour_idx",
        F.coalesce(F.sum("impr").over(window), F.lit(0)).cast("long").alias(f"{name}_prior_impr"),
        F.coalesce(F.sum("clicks").over(window), F.lit(0))
        .cast("long")
        .alias(f"{name}_prior_clicks"),
    )


def _global_history(events: DataFrame) -> DataFrame:
    """Per hour: impressions and clicks from all strictly earlier hours, for every key."""
    per_hour = events.groupBy("hour_idx").agg(
        F.count(F.lit(1)).alias("impr"), F.sum("click").alias("clicks")
    )
    window = _prior_window(None)
    return per_hour.select(
        "hour_idx",
        F.coalesce(F.sum("impr").over(window), F.lit(0)).cast("long").alias("global_prior_impr"),
        F.coalesce(F.sum("clicks").over(window), F.lit(0))
        .cast("long")
        .alias("global_prior_clicks"),
    ).withColumn(
        "global_prior_ctr",
        F.when(
            F.col("global_prior_impr") > 0,
            F.col("global_prior_clicks") / F.col("global_prior_impr"),
        ),
    )


def _add_features(events: DataFrame, date: str) -> DataFrame:
    """Join each day-D row with its key histories and the global history, then smooth the rates."""
    day = events.filter(F.col("dt") == date)
    out = day
    for name, key_col in KEY_COLUMNS.items():
        out = out.join(_key_history(events, name, key_col), on=[key_col, "hour_idx"], how="left")
    out = out.join(_global_history(events), on="hour_idx", how="left")
    for name in KEY_COLUMNS:
        out = out.withColumn(
            f"{name}_ctr_smooth",
            (F.col(f"{name}_prior_clicks") + F.lit(FEATURE_ALPHA) * F.col("global_prior_ctr"))
            / (F.col(f"{name}_prior_impr") + F.lit(FEATURE_ALPHA)),
        )
    return out.select(*RAW_COLUMNS, "hour_ts", "hr", "dt", *FEATURE_COLUMNS)


def _check_output(out: DataFrame, day_rows: int) -> int:
    """Row count must match day D, and every row must have a global prior. Returns the row count."""
    checks = out.agg(
        F.count(F.lit(1)).alias("rows"),
        F.count(F.when(F.col("global_prior_ctr").isNull(), 1)).alias("no_prior"),
    ).collect()[0]
    if checks["rows"] != day_rows:
        raise PipelineError(f"output has {checks['rows']} rows, day has {day_rows}")
    if checks["no_prior"]:
        raise PipelineError(
            f"{checks['no_prior']} rows have no earlier hour to learn from, "
            "which only happens in the first hour of the data"
        )
    return int(checks["rows"])


def _write_manifest(manifests_dir: Path, date: str, manifest: dict) -> None:
    """Write to a temporary name, then rename, so a crash never leaves a half-written file."""
    manifests_dir.mkdir(parents=True, exist_ok=True)
    final = manifests_dir / f"{date}.json"
    temporary = manifests_dir / f"{date}.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, final)


def build_training_set(date: str, require_complete_history: bool = True) -> BuildResult:
    """Write the training set for one day and its manifest. Raises if the output fails a check."""
    _check_build_date(date, require_complete_history)
    settings = get_settings()
    if require_complete_history:
        _check_history_complete(date, settings.hourly_dir)

    spark = get_spark("build-training-set")
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

    events = _read_events(spark, settings.hourly_dir, date)
    day_rows = events.filter(F.col("dt") == date).count()
    out = _add_features(events, date)
    rows = _check_output(out, day_rows)

    (
        out.repartition(PARTITION_COUNT)
        .write.mode("overwrite")
        .partitionBy("dt")
        .parquet(str(settings.training_dir))
    )

    manifest = {
        "data_date": date,
        "feature_version": FEATURE_VERSION,
        "rows": rows,
        "history_start": FIRST_DAY,
        "cutoff_rule": CUTOFF_RULE,
        "feature_columns": FEATURE_COLUMNS,
        "feature_alpha": FEATURE_ALPHA,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _write_manifest(settings.manifests_dir, date, manifest)
    logger.info("wrote %d training rows for %s to %s", rows, date, settings.training_dir)
    return BuildResult(
        date=date,
        rows=rows,
        path=settings.training_dir / f"dt={date}",
        manifest=manifest,
    )


def load_training_day(date: str) -> DataFrame:
    """Read one day of the written training set back from disk."""
    settings = get_settings()
    spark = get_spark("build-training-set")
    return (
        spark.read.parquet(str(settings.training_dir))
        .withColumn("dt", F.col("dt").cast("string"))
        .filter(F.col("dt") == date)
    )


def feature_stats(df: DataFrame) -> dict[str, dict[str, float | None]]:
    """Mean, median, 95th percentile and null fraction for every feature column."""
    aggregates = [F.avg(name).alias(f"{name}__mean") for name in FEATURE_COLUMNS]
    aggregates += [
        F.count(F.when(F.col(name).isNull(), 1)).alias(f"{name}__nulls") for name in FEATURE_COLUMNS
    ]
    aggregates.append(F.count(F.lit(1)).alias("rows"))
    summary = df.agg(*aggregates).collect()[0]
    quantiles = df.approxQuantile(FEATURE_COLUMNS, STAT_QUANTILES, STAT_RELATIVE_ERROR)

    total = summary["rows"]
    stats: dict[str, dict[str, float | None]] = {}
    for name, values in zip(FEATURE_COLUMNS, quantiles, strict=True):
        # approxQuantile ignores nulls and returns no values for an all-null column.
        p50, p95 = values if len(values) == len(STAT_QUANTILES) else (None, None)
        mean = summary[f"{name}__mean"]
        nulls = summary[f"{name}__nulls"]
        stats[name] = {
            "mean": None if mean is None else float(mean),
            "p50": p50,
            "p95": p95,
            "null_fraction": nulls / total if total else None,
        }
    return stats
