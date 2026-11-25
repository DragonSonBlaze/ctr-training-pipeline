from pathlib import Path

import pytest
from py4j.protocol import Py4JError
from pyspark.errors import PySparkException
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from ctr_pipeline.config import get_settings
from ctr_pipeline.errors import MissingInputError
from ctr_pipeline.hourly_load import is_day_complete, load_hourly

FIXTURES = Path(__file__).parent / "fixtures"
LARGE_ID = "18446744073709551615"


@pytest.fixture
def hourly_dir(monkeypatch, tmp_path) -> Path:
    monkeypatch.setenv("CTR_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CTR_RAW_CSV", str(FIXTURES / "raw_small.csv"))
    return get_settings().hourly_dir


def _read_table(spark, hourly_dir: Path):
    return spark.read.parquet(str(hourly_dir))


def test_rows_in_equal_rows_out(hourly_dir):
    result = load_hourly()
    assert result.rows_read == 8
    assert result.rows_written == 8
    assert result.hour_partitions == 3
    assert result.days == ["2014-10-21", "2014-10-22"]


def test_one_file_per_hour_folder(hourly_dir):
    load_hourly()
    folders = [
        hourly_dir / "dt=2014-10-21" / "hr=0",
        hourly_dir / "dt=2014-10-21" / "hr=1",
        hourly_dir / "dt=2014-10-22" / "hr=0",
    ]
    for folder in folders:
        assert folder.is_dir(), folder
        assert len([p for p in folder.iterdir() if p.suffix == ".parquet"]) == 1, folder


def test_id_stays_string(hourly_dir, spark):
    load_hourly()
    df = _read_table(spark, hourly_dir)
    assert isinstance(df.schema["id"].dataType, StringType)
    assert df.filter(F.col("id") == LARGE_ID).count() == 1


def test_hour_ts_and_partitions_derived(hourly_dir, spark):
    load_hourly()
    # Formatting in Spark (session time zone UTC) avoids collect(), which converts
    # timestamps to naive datetimes in the machine's local time zone.
    row = (
        _read_table(spark, hourly_dir)
        .filter(F.col("hour") == "14102101")
        .select(
            F.date_format("hour_ts", "yyyy-MM-dd HH:mm:ss").alias("hour_ts_utc"),
            "hr",
        )
        .distinct()
        .collect()
    )
    assert len(row) == 1
    assert row[0]["hour_ts_utc"] == "2014-10-21 01:00:00"
    assert row[0]["hr"] == 1


def test_rerun_does_not_duplicate(hourly_dir, spark):
    load_hourly()
    load_hourly()
    assert _read_table(spark, hourly_dir).count() == 8


def test_reload_one_day_keeps_other_days(hourly_dir, spark):
    load_hourly()
    load_hourly("2014-10-22")
    table = _read_table(spark, hourly_dir)
    assert table.filter(F.col("dt").cast("string") == "2014-10-21").count() == 6
    assert table.filter(F.col("dt").cast("string") == "2014-10-22").count() == 2


def test_malformed_row_fails_load(monkeypatch, tmp_path, hourly_dir):
    monkeypatch.setenv("CTR_RAW_CSV", str(FIXTURES / "raw_malformed.csv"))
    # FAILFAST surfaces as a Spark (JVM) error, which reaches Python as a Py4J error.
    with pytest.raises((PySparkException, Py4JError)):
        load_hourly()
    assert not hourly_dir.exists() or not any(hourly_dir.glob("dt=*"))


def test_missing_raw_csv_is_reported(monkeypatch, tmp_path, hourly_dir):
    monkeypatch.setenv("CTR_RAW_CSV", str(tmp_path / "absent.csv"))
    with pytest.raises(MissingInputError):
        load_hourly()


def test_partition_pruning_in_plan(hourly_dir, spark):
    load_hourly()
    df = _read_table(spark, hourly_dir).filter(F.col("dt") == "2014-10-22")
    plan = df._jdf.queryExecution().executedPlan().toString()
    assert "PartitionFilters" in plan
    assert "dt" in plan


def test_is_day_complete(hourly_dir):
    for hour in range(23):
        folder = hourly_dir / "dt=2014-10-21" / f"hr={hour}"
        folder.mkdir(parents=True)
        (folder / "part-0.parquet").touch()
    assert is_day_complete("2014-10-21", hourly_dir) is False

    folder = hourly_dir / "dt=2014-10-21" / "hr=23"
    folder.mkdir(parents=True)
    (folder / "part-0.parquet").touch()
    assert is_day_complete("2014-10-21", hourly_dir) is True

    (hourly_dir / "dt=2014-10-21" / "hr=5" / "part-0.parquet").unlink()
    assert is_day_complete("2014-10-21", hourly_dir) is False
