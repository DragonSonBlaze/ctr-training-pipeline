"""Daily CTR pipeline: one Airflow run per data date.

Each run handles one data date D. The sensor makes the run wait until all 24
hourly folders for D exist, so no step starts on a partial day. The DAG never
loads data: the hourly tables arrive from load-hourly, which stands in for
Project 2. Every task is one call to the ctr_pipeline command line, so each
step can also be run by hand with `python -m ctr_pipeline`.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.sensors.bash import BashSensor
from airflow.sdk import DAG

# This file is parsed by the Airflow venv, which does not have the project's
# packages, so it must not import ctr_pipeline.
REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv" / "bin" / "python"


def cli(args: str) -> str:
    return f"cd {REPO} && {PY} -m ctr_pipeline {args}"


with DAG(
    dag_id="ctr_daily",
    schedule="@daily",
    start_date=datetime(2014, 10, 22, tzinfo=UTC),
    end_date=datetime(2014, 10, 30, tzinfo=UTC),
    catchup=True,
    max_active_runs=1,
    tags=["ctr"],
    # A failed day blocks the next one, so the champion history has no gaps.
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=1),
        "depends_on_past": True,
    },
) as dag:
    wait_for_day = BashSensor(
        task_id="wait_for_day",
        bash_command=cli("check-complete --date {{ ds }}"),
        mode="reschedule",
        poke_interval=60,
        timeout=6 * 60 * 60,
    )

    quality_checks = BashOperator(
        task_id="quality_checks",
        bash_command=cli("quality-check --date {{ ds }}"),
    )

    build_training_set = BashOperator(
        task_id="build_training_set",
        bash_command=cli("build-training-set --date {{ ds }}"),
    )

    # Exit code 99 marks a task skipped. The run id is the last line of stdout,
    # which BashOperator pushes to XCom. The models train one after the other, so
    # they never compete for memory.
    train_logistic = BashOperator(
        task_id="train_logistic",
        bash_command=cli("train --date {{ ds }} --model logistic_hashed"),
    )

    train_embedding = BashOperator(
        task_id="train_embedding",
        bash_command=cli("train --date {{ ds }} --model embedding_nn"),
    )

    promote = BashOperator(
        task_id="promote",
        bash_command=cli(
            "promote --date {{ ds }} "
            "--run-id {{ ti.xcom_pull(task_ids='train_logistic') }} "
            "--run-id {{ ti.xcom_pull(task_ids='train_embedding') }}"
        ),
    )

    (
        wait_for_day
        >> quality_checks
        >> build_training_set
        >> train_logistic
        >> train_embedding
        >> promote
    )
