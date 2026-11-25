"""Stage 1: load the raw Avazu CSV into hourly Parquet tables.

Input:  data/raw/train.csv
Output: data/hourly/events/dt=YYYY-MM-DD/hr=H/*.parquet (240 hour folders)

This job stands in for Project 2, which will later write the same layout from a stream.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ctr_pipeline.config import get_settings
from ctr_pipeline.contracts import PARTITION_COLUMNS, RAW_SCHEMA
from ctr_pipeline.errors import LoadVerificationError, MissingInputError
from ctr_pipeline.spark import get_spark

logger = logging.getLogger(__name__)

HOURS_PER_DAY = 24


@dataclass(frozen=True)
class LoadResult:
    rows_read: int
    rows_written: int
    days: list[str]
    hour_partitions: int


def _yymmdd(date: str) -> str:
    """'2014-10-21' -> '141021', the prefix of the raw `hour` field."""
    return date.replace("-", "")[2:]


def add_derived_columns(df: DataFrame) -> DataFrame:
    """Add hour_ts (timestamp), dt and hr (partition columns) from the raw `hour` string.

    The session time zone is UTC, so hour_ts is UTC and so are dt and hr.
    """
    with_ts = df.withColumn("hour_ts", F.to_timestamp("hour", "yyMMddHH"))
    return with_ts.withColumn("dt", F.date_format("hour_ts", "yyyy-MM-dd")).withColumn(
        "hr", F.hour("hour_ts")
    )


def read_raw_csv(spark: SparkSession, path: Path) -> DataFrame:
    # FAILFAST stops the load on a malformed row, so bad input is never silently dropped.
    # The schema is given, not inferred: inference is a second full pass over the file
    # and would turn `id` into a number that overflows.
    return spark.read.csv(str(path), header=True, schema=RAW_SCHEMA, mode="FAILFAST")


def write_hourly(df: DataFrame, hourly_dir: Path) -> None:
    # Repartitioning by the partition columns sends each hour to one task, so each
    # hour folder gets one file instead of one file per task.
    (
        df.repartition(*PARTITION_COLUMNS)
        .write.mode("overwrite")
        .partitionBy(*PARTITION_COLUMNS)
        .parquet(str(hourly_dir))
    )


def _count_written(spark: SparkSession, hourly_dir: Path, days: list[str]) -> tuple[int, int]:
    """Rows and (dt, hr) pairs for the given days, read back from disk."""
    written = spark.read.parquet(str(hourly_dir)).filter(F.col("dt").cast("string").isin(days))
    rows = written.count()
    partitions = written.select("dt", "hr").distinct().count()
    return rows, partitions


def load_hourly(date: str | None = None) -> LoadResult:
    """Write the hourly tables for one day, or for every day when date is None.

    Only the partitions written here are replaced, so reloading one day leaves
    the others as they are.
    """
    settings = get_settings()
    if not settings.raw_csv.exists():
        raise MissingInputError(f"raw CSV not found: {settings.raw_csv}")

    spark = get_spark("hourly-load")
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

    raw = read_raw_csv(spark, settings.raw_csv)
    if date is not None:
        raw = raw.filter(F.substring("hour", 1, 6) == _yymmdd(date))
    df = add_derived_columns(raw).cache()
    try:
        rows_read = df.count()
        if rows_read == 0:
            raise MissingInputError(f"no rows in {settings.raw_csv} for {date or 'any day'}")
        days = sorted(row["dt"] for row in df.select("dt").distinct().collect())
        hour_partitions = df.select("dt", "hr").distinct().count()

        write_hourly(df, settings.hourly_dir)

        rows_written, written_partitions = _count_written(spark, settings.hourly_dir, days)
        if rows_written != rows_read or written_partitions != hour_partitions:
            raise LoadVerificationError(
                f"read {rows_read} rows in {hour_partitions} hour partitions, "
                f"found {rows_written} rows in {written_partitions} after writing"
            )
    finally:
        df.unpersist()

    logger.info(
        "loaded %d rows into %d hour partitions for %d day(s), %s to %s",
        rows_written,
        hour_partitions,
        len(days),
        days[0],
        days[-1],
    )
    return LoadResult(
        rows_read=rows_read,
        rows_written=rows_written,
        days=days,
        hour_partitions=hour_partitions,
    )


def missing_hours(date: str, hourly_dir: Path | None = None) -> list[int]:
    """Hours of the day with no folder, or a folder with no .parquet file in it."""
    base = hourly_dir or get_settings().hourly_dir
    missing = []
    for hour in range(HOURS_PER_DAY):
        folder = base / f"dt={date}" / f"hr={hour}"
        has_file = folder.is_dir() and any(p.name.endswith(".parquet") for p in folder.iterdir())
        if not has_file:
            missing.append(hour)
    return missing


def is_day_complete(date: str, hourly_dir: Path | None = None) -> bool:
    """True when all 24 hour folders of the day hold a Parquet file.

    Plain filesystem check, no Spark: the Airflow sensor calls it every minute.
    """
    return not missing_hours(date, hourly_dir)
