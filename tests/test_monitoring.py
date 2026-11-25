"""Monitoring recorder. Unit tests need no database. Integration tests need `make up`
and skip cleanly when Postgres is not listening on POSTGRES_PORT.
"""

import socket
from collections import namedtuple

import pytest

from ctr_pipeline import monitoring
from ctr_pipeline.errors import SkipStep

UNREACHABLE_DSN = "postgresql://x:y@127.0.0.1:1/none"
TEST_DATE = "1999-01-01"

QualityResult = namedtuple("QualityResult", ["name", "passed", "observed", "threshold"])


def _port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture
def postgres(env):
    import psycopg
    from psycopg.conninfo import make_conninfo

    port = int(env["POSTGRES_PORT"])
    if not _port_open(port):
        pytest.skip("Postgres is not running (make up)")

    def dsn(user: str, password: str) -> str:
        return make_conninfo(
            host="127.0.0.1",
            port=port,
            user=user,
            password=password,
            dbname="monitoring",
        )

    owner_dsn = dsn(env["POSTGRES_USER"], env["POSTGRES_PASSWORD"])
    monitoring.init_schema(owner_dsn)
    yield {
        "owner_dsn": owner_dsn,
        "reader_dsn": dsn(env["GRAFANA_DB_USER"], env["GRAFANA_DB_PASSWORD"]),
    }
    with psycopg.connect(owner_dsn, autocommit=True) as conn:
        for table in (
            "pipeline_step_runs",
            "data_quality_results",
            "model_runs",
            "feature_daily_stats",
        ):
            conn.execute(f"DELETE FROM {table} WHERE data_date = %s", (TEST_DATE,))
        conn.execute("DELETE FROM pipeline_step_runs WHERE step LIKE 'test-%'")


# Unit tests: no database.


def test_record_step_success_survives_db_down():
    with monitoring.record_step("test-step", TEST_DATE, dsn=UNREACHABLE_DSN) as stats:
        stats.rows_in = 1
    # Reaching this line means no exception escaped from the monitoring write.


def test_record_step_reraises_step_error_when_db_down():
    with pytest.raises(ValueError) as excinfo:
        with monitoring.record_step("test-step", TEST_DATE, dsn=UNREACHABLE_DSN):
            raise ValueError("boom")
    assert str(excinfo.value) == "boom"


def test_record_step_reraises_skip():
    with pytest.raises(SkipStep):
        with monitoring.record_step("test-step", TEST_DATE, dsn=UNREACHABLE_DSN):
            raise SkipStep("nothing to do")


# Integration tests: need the Compose Postgres.


def test_init_schema_is_idempotent(postgres):
    monitoring.init_schema(postgres["owner_dsn"])
    monitoring.init_schema(postgres["owner_dsn"])


def test_record_step_writes_success_and_failure_rows(postgres):
    import psycopg

    dsn = postgres["owner_dsn"]
    with monitoring.record_step("test-step-ok", TEST_DATE, dsn=dsn) as stats:
        stats.rows_in = 10
        stats.rows_out = 9

    with pytest.raises(ValueError):
        with monitoring.record_step("test-step-fail", TEST_DATE, dsn=dsn):
            raise ValueError("bad input")

    with psycopg.connect(dsn) as conn:
        success = conn.execute(
            "SELECT status, rows_in, rows_out, duration_s FROM pipeline_step_runs "
            "WHERE step = 'test-step-ok' AND data_date = %s",
            (TEST_DATE,),
        ).fetchall()
        failure = conn.execute(
            "SELECT status, error FROM pipeline_step_runs "
            "WHERE step = 'test-step-fail' AND data_date = %s",
            (TEST_DATE,),
        ).fetchall()

    assert len(success) == 1
    assert success[0][0] == "success"
    assert (success[0][1], success[0][2]) == (10, 9)
    assert success[0][3] >= 0
    assert len(failure) == 1
    assert failure[0][0] == "failed"
    assert failure[0][1].startswith("ValueError")


def test_quality_results_upsert_replaces(postgres):
    import psycopg

    dsn = postgres["owner_dsn"]
    monitoring.record_quality_results(
        TEST_DATE, [QualityResult("hours_present", False, 23.0, "24")], dsn=dsn
    )
    monitoring.record_quality_results(
        TEST_DATE, [QualityResult("hours_present", True, 24.0, "24")], dsn=dsn
    )

    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            "SELECT passed FROM data_quality_results "
            "WHERE data_date = %s AND check_name = 'hours_present'",
            (TEST_DATE,),
        ).fetchall()
    assert rows == [(True,)]


def test_grafana_reader_can_read_new_tables(postgres):
    import psycopg

    with psycopg.connect(postgres["reader_dsn"]) as reader:
        count = reader.execute("SELECT count(*) FROM pipeline_step_runs").fetchone()[0]
    assert count >= 0
