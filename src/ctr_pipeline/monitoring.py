"""Per-run pipeline metrics in the Postgres `monitoring` database.

Grafana charts these tables. Writing is best effort: a database that is down
logs a warning and the pipeline step carries on. The one exception is
init_schema(), which must fail loudly so that monitoring-init reports a dead database.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import psycopg

from ctr_pipeline.config import REPO_ROOT, get_settings
from ctr_pipeline.errors import SkipStep

logger = logging.getLogger(__name__)

SCHEMA_PATH = REPO_ROOT / "sql" / "monitoring_schema.sql"
ERROR_TEXT_LIMIT = 2000
MODEL_RUN_COLUMNS = (
    "mlflow_run_id",
    "data_date",
    "model_type",
    "model_version",
    "dataset_version",
    "train_rows",
    "eval_rows",
    "auc",
    "log_loss",
    "mean_pred",
    "actual_ctr",
    "champion_log_loss",
    "promoted",
)
FEATURE_STAT_KEYS = ("mean", "p50", "p95", "null_fraction")


@dataclass
class StepStats:
    rows_in: int | None = None
    rows_out: int | None = None


def _resolve_dsn(dsn: str | None) -> str:
    return dsn or get_settings().monitoring_dsn


def _connect(dsn: str) -> psycopg.Connection:
    # Autocommit means every statement is stored at once, so a step that fails
    # later cannot roll back the row that records its own failure.
    return psycopg.connect(dsn, autocommit=True, connect_timeout=3)


def _best_effort_write(
    what: str, dsn: str | None, write: Callable[[psycopg.Connection], None]
) -> None:
    try:
        with _connect(_resolve_dsn(dsn)) as conn:
            write(conn)
    except (psycopg.Error, OSError) as exc:
        logger.warning("monitoring: could not write %s, continuing: %s", what, exc)


def init_schema(dsn: str | None = None) -> None:
    """Create the monitoring tables if they are missing. Raises if the database is unreachable."""
    sql = SCHEMA_PATH.read_text()
    with _connect(_resolve_dsn(dsn)) as conn:
        conn.execute(sql)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:ERROR_TEXT_LIMIT]


def _write_step(
    step: str,
    data_date: str | None,
    started_at: datetime,
    status: str,
    stats: StepStats,
    error: str | None,
    dsn: str | None,
) -> None:
    finished_at = _utc_now()
    duration_s = (finished_at - started_at).total_seconds()

    def write(conn: psycopg.Connection) -> None:
        conn.execute(
            """
            INSERT INTO pipeline_step_runs
                (step, data_date, started_at, finished_at, duration_s,
                 status, rows_in, rows_out, error)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                step,
                data_date,
                started_at,
                finished_at,
                duration_s,
                status,
                stats.rows_in,
                stats.rows_out,
                error,
            ),
        )

    _best_effort_write(f"step run {step} ({status})", dsn, write)


@contextmanager
def record_step(step: str, data_date: str | None, dsn: str | None = None) -> Iterator[StepStats]:
    """Time a pipeline step and store one row for it, whatever its outcome.

    The step's own exception always comes out unchanged: the monitoring write
    happens first and never replaces it.
    """
    stats = StepStats()
    started_at = _utc_now()
    try:
        yield stats
    except SkipStep:
        _write_step(step, data_date, started_at, "skipped", stats, None, dsn)
        raise
    except BaseException as exc:
        _write_step(step, data_date, started_at, "failed", stats, _error_text(exc), dsn)
        raise
    else:
        _write_step(step, data_date, started_at, "success", stats, None, dsn)


def record_quality_results(data_date: str, results: Sequence[Any], dsn: str | None = None) -> None:
    """Store one row per check. A re-run for the same day replaces the earlier rows."""
    rows = [(data_date, r.name, r.passed, r.observed, r.threshold) for r in results]

    def write(conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO data_quality_results
                    (data_date, check_name, passed, observed, threshold, checked_at)
                VALUES (%s, %s, %s, %s, %s, now())
                ON CONFLICT (data_date, check_name) DO UPDATE SET
                    passed = EXCLUDED.passed,
                    observed = EXCLUDED.observed,
                    threshold = EXCLUDED.threshold,
                    checked_at = now()
                """,
                rows,
            )

    _best_effort_write(f"quality results for {data_date}", dsn, write)


def record_model_run(row: dict[str, Any], dsn: str | None = None) -> None:
    """Store one model run, keyed by its MLflow run id. Missing keys are stored as NULL."""
    unknown = set(row) - set(MODEL_RUN_COLUMNS)
    if unknown:
        raise ValueError(f"unknown model_runs columns: {sorted(unknown)}")
    columns = ", ".join(MODEL_RUN_COLUMNS)
    placeholders = ", ".join(["%s"] * len(MODEL_RUN_COLUMNS))
    updates = ", ".join(
        f"{column} = EXCLUDED.{column}" for column in MODEL_RUN_COLUMNS if column != "mlflow_run_id"
    )
    sql = (
        f"INSERT INTO model_runs ({columns}) VALUES ({placeholders}) "
        f"ON CONFLICT (mlflow_run_id) DO UPDATE SET {updates}"
    )
    values = tuple(row.get(column) for column in MODEL_RUN_COLUMNS)

    def write(conn: psycopg.Connection) -> None:
        conn.execute(sql, values)

    _best_effort_write(f"model run {row.get('mlflow_run_id')}", dsn, write)


def update_model_run_promotion(
    mlflow_run_id: str,
    promoted: bool,
    champion_log_loss: float | None,
    model_version: int | None = None,
    dsn: str | None = None,
) -> None:
    """Record the promotion decision. A None model_version keeps the stored version."""

    def write(conn: psycopg.Connection) -> None:
        conn.execute(
            "UPDATE model_runs SET promoted = %s, champion_log_loss = %s, "
            "model_version = COALESCE(%s, model_version) WHERE mlflow_run_id = %s",
            (promoted, champion_log_loss, model_version, mlflow_run_id),
        )

    _best_effort_write(f"promotion of model run {mlflow_run_id}", dsn, write)


def record_feature_stats(
    data_date: str,
    stats: dict[str, dict[str, float | None]],
    dsn: str | None = None,
) -> None:
    """Store mean, p50, p95 and null fraction per feature. Re-runs replace the day's rows."""
    rows = [
        (
            data_date,
            feature,
            *(values.get(key) for key in FEATURE_STAT_KEYS),
        )
        for feature, values in stats.items()
    ]

    def write(conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO feature_daily_stats
                    (data_date, feature, mean, p50, p95, null_fraction)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (data_date, feature) DO UPDATE SET
                    mean = EXCLUDED.mean,
                    p50 = EXCLUDED.p50,
                    p95 = EXCLUDED.p95,
                    null_fraction = EXCLUDED.null_fraction
                """,
                rows,
            )

    _best_effort_write(f"feature stats for {data_date}", dsn, write)


def record_feature_drift(
    data_date: str,
    reference_date: str,
    values: dict[str, float],
    dsn: str | None = None,
) -> None:
    """Store the drift distance of each feature for one day. A re-run replaces that day's rows."""
    rows = [(data_date, reference_date, feature, distance) for feature, distance in values.items()]

    def write(conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO feature_drift
                    (data_date, reference_date, feature, distance, computed_at)
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT (data_date, feature) DO UPDATE SET
                    reference_date = EXCLUDED.reference_date,
                    distance = EXCLUDED.distance,
                    computed_at = now()
                """,
                rows,
            )

    _best_effort_write(f"feature drift for {data_date}", dsn, write)
