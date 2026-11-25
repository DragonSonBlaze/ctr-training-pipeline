"""Shared SparkSession setup for every job in the pipeline.

Only settings that every job needs live here. A job's own settings, such as the
partition overwrite mode for the hourly load, belong in that job.
"""

import os

from pyspark.sql import SparkSession

DEFAULT_DRIVER_MEMORY = "6g"


def get_spark(app_name: str, driver_memory: str | None = None) -> SparkSession:
    """Start a local-mode SparkSession, or return the one already running.

    Local mode runs the driver and the executors in one JVM on this machine, so
    the driver memory is the only memory setting that matters. It is read when
    the JVM starts: if a session already exists in this process, getOrCreate()
    returns it and these settings have no effect.

    The driver binds to 127.0.0.1. By default it binds to the machine's hostname,
    which on a Mac can resolve to an address no interface holds, and then the
    session fails to start. Local mode never talks to another machine.

    The session time zone is UTC so that parsing and formatting timestamps gives
    the same result on any machine, whatever its local time zone.
    """
    memory = driver_memory or os.environ.get("SPARK_DRIVER_MEMORY", DEFAULT_DRIVER_MEMORY)
    return (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.driver.memory", memory)
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
