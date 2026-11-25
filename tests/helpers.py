"""Builders for small hourly tables, reused by tests of later stages."""

import itertools
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

from ctr_pipeline.config import FEATURE_VERSION, Settings
from ctr_pipeline.contracts import (
    CATEGORICAL_FIELDS,
    MODEL_INPUT_COLUMNS,
    PREDICT_COLUMNS,
    RAW_COLUMNS,
    RAW_SCHEMA,
)
from ctr_pipeline.hourly_load import add_derived_columns

_ids = itertools.count(1)
_STRING_COLUMNS = {
    field.name for field in RAW_SCHEMA.fields if field.dataType.typeName() == "string"
}


def hourly_row(hour: str, click: int, **overrides) -> dict:
    """One raw impression as a dict with all 24 columns.

    Overrides replace any column, including with None.
    """
    row = {}
    for name in RAW_COLUMNS:
        if name == "id":
            row[name] = str(next(_ids))
        elif name == "hour":
            row[name] = hour
        elif name == "click":
            row[name] = click
        elif name in _STRING_COLUMNS:
            row[name] = "x"
        else:
            row[name] = 0
    row.update(overrides)
    return row


def write_hourly(spark: SparkSession, hourly_dir: Path, rows: list[dict]) -> None:
    """Write rows as hourly Parquet, partitioned the same way as load_hourly does."""
    data = [tuple(row[name] for name in RAW_COLUMNS) for row in rows]
    df = add_derived_columns(spark.createDataFrame(data, schema=RAW_SCHEMA))
    df.write.mode("append").partitionBy("dt", "hr").parquet(str(hourly_dir))


def training_frame(rows: int, click_every: int = 5) -> pd.DataFrame:
    """Training rows with all history counts at zero and a click every `click_every` rows.

    With zero history every prediction equals p0, so the model's output is easy to reason about.
    The default gives a click rate of exactly 0.2.
    """
    clicks = (np.arange(rows) % click_every == 0).astype("int64")
    frame = pd.DataFrame({"click": clicks})
    for column in MODEL_INPUT_COLUMNS:
        frame[column] = np.zeros(rows, dtype="int64")
    # Every column any model reads, so a day can be scored by any model type.
    for column in PREDICT_COLUMNS:
        if column in frame.columns:
            continue
        if column == "hr":
            frame[column] = np.zeros(rows, dtype="int64")
        elif column in CATEGORICAL_FIELDS:
            frame[column] = "x"
        else:
            frame[column] = np.zeros(rows, dtype="int64")
    return frame


def write_training_day(settings: Settings, date: str, frame: pd.DataFrame) -> None:
    """Write one training-set day and its manifest, without running the Spark job."""
    part_dir = settings.training_dir / f"dt={date}"
    part_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(part_dir / "part-0.parquet", index=False)

    settings.manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "data_date": date,
        "feature_version": FEATURE_VERSION,
        "rows": len(frame),
        "created_at": datetime.now(UTC).isoformat(),
    }
    (settings.manifests_dir / f"{date}.json").write_text(json.dumps(manifest, sort_keys=True))
