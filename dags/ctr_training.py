"""CTR training run: from the hourly tables of 10 days to a registered model.

Triggered by hand. The run waits until every day has its 24 hourly folders, checks
and features each day, versions the train, validation and test split, runs the
baseline and the Keras sweep, writes the Evidently drift report, and registers the
better candidate in MLflow only if its test log loss beats the champion.

Every task is one call to the ctr_pipeline command line, so each step can also be
run by hand with `python -m ctr_pipeline`.
"""

import os
import shlex
from datetime import UTC, datetime, timedelta
from pathlib import Path

from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.sensors.bash import BashSensor
from airflow.sdk import DAG

# This file is parsed by the Airflow venv, which does not have the project's
# packages, so it must never load the pipeline package. The day lists below are therefore
# literals. They must equal config.day_range(FIRST_DAY, LAST_DAY) and
# config.day_range(FIRST_TRAINING_DAY, LAST_DAY); tests/test_dag_file.py checks both.
ALL_DAYS = [
    "2014-10-21",
    "2014-10-22",
    "2014-10-23",
    "2014-10-24",
    "2014-10-25",
    "2014-10-26",
    "2014-10-27",
    "2014-10-28",
    "2014-10-29",
    "2014-10-30",
]
TRAINING_DAYS = [
    "2014-10-22",
    "2014-10-23",
    "2014-10-24",
    "2014-10-25",
    "2014-10-26",
    "2014-10-27",
    "2014-10-28",
    "2014-10-29",
    "2014-10-30",
]

# CTR_REPO and CTR_PYTHON let the container point the DAG at the mounted repo and its
# Python. Without them the DAG uses the repo this file lives in and that repo's .venv.
REPO = Path(os.environ.get("CTR_REPO", Path(__file__).resolve().parents[1]))
PY = os.environ.get("CTR_PYTHON", str(REPO / ".venv" / "bin" / "python"))


def cli(args: str) -> str:
    return f"cd {shlex.quote(str(REPO))} && {shlex.quote(PY)} -m ctr_pipeline {args}"


def xcom(task_id: str) -> str:
    """Jinja template for the last stdout line of another task."""
    return "{{ ti.xcom_pull(task_ids='" + task_id + "') }}"


def wait_for_data_command() -> str:
    # One loop over every day: the first incomplete day exits non-zero, and the
    # sensor then waits and checks again.
    days = " ".join(ALL_DAYS)
    return (
        f"cd {shlex.quote(str(REPO))} && "
        f"for day in {days}; do "
        f'{shlex.quote(PY)} -m ctr_pipeline check-complete --date "$day" || exit 1; '
        "done"
    )


def register_command() -> str:
    # experiment-candidates prints "<baseline run id> <best trial run id>". Parameter
    # expansion splits that line into the two --run-id values.
    return (
        f"cd {shlex.quote(str(REPO))} && "
        f"CANDS=$({shlex.quote(PY)} -m ctr_pipeline experiment-candidates "
        f"--parent-run-id {xcom('experiment')}) && "
        f"{shlex.quote(PY)} -m ctr_pipeline register "
        f"--dataset {xcom('build_dataset')} "
        '--run-id "${CANDS% *}" --run-id "${CANDS#* }"'
    )


with DAG(
    dag_id="ctr_training",
    schedule=None,
    start_date=datetime(2014, 10, 22, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    tags=["ctr"],
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=1),
    },
) as dag:
    wait_for_data = BashSensor(
        task_id="wait_for_data",
        bash_command=wait_for_data_command(),
        mode="reschedule",
        poke_interval=60,
        timeout=6 * 60 * 60,
    )

    quality_checks = BashOperator.partial(task_id="quality_checks").expand(
        bash_command=[cli(f"quality-check --date {day}") for day in ALL_DAYS]
    )

    # One Spark job at a time: the machine has 16 GB of RAM.
    build_training_set = BashOperator.partial(
        task_id="build_training_set",
        max_active_tis_per_dag=1,
    ).expand(bash_command=[cli(f"build-training-set --date {day}") for day in TRAINING_DAYS])

    # Its last stdout line is the dataset version, which the later tasks read from XCom.
    build_dataset = BashOperator(
        task_id="build_dataset",
        bash_command=cli("build-dataset"),
    )

    # Its last stdout line is the MLflow parent run id.
    experiment = BashOperator(
        task_id="experiment",
        bash_command=cli(f"experiment --dataset {xcom('build_dataset')}"),
    )

    drift_report = BashOperator(
        task_id="drift_report",
        bash_command=cli(
            f"drift-report --dataset {xcom('build_dataset')} --parent-run-id {xcom('experiment')}"
        ),
    )

    register = BashOperator(
        task_id="register",
        bash_command=register_command(),
    )

    (wait_for_data >> quality_checks >> build_training_set >> build_dataset >> experiment)
    experiment >> drift_report >> register
