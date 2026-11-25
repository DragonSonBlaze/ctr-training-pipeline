"""Checks that Spark starts on this machine's Java and uses the shared settings."""


def test_runs_in_local_mode(spark):
    assert spark.sparkContext.master.startswith("local")


def test_session_time_zone_is_utc(spark):
    assert spark.conf.get("spark.sql.session.timeZone") == "UTC"


def test_small_job_runs(spark):
    df = spark.createDataFrame([(1, 0), (2, 1), (3, 1)], ["id", "click"])
    assert df.count() == 3
    assert df.groupBy().sum("click").collect()[0][0] == 2
