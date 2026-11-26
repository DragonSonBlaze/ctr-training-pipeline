"""Load the ctr_training DAG the way Airflow does and check its shape.

Run in the Airflow venv: .venv-airflow/bin/python scripts/check_dag.py
"""

import sys
from pathlib import Path

from airflow.dag_processing.dagbag import DagBag
from airflow.sdk.definitions.mappedoperator import MappedOperator

REPO = Path(__file__).resolve().parents[1]
DAG_ID = "ctr_training"
TASK_IDS = {
    "wait_for_data",
    "quality_checks",
    "build_training_set",
    "build_dataset",
    "experiment",
    "drift_report",
    "register",
}
# Each task maps to the tasks that run directly after it
EDGES: dict[str, set[str]] = {
    "wait_for_data": {"quality_checks"},
    "quality_checks": {"build_training_set"},
    "build_training_set": {"build_dataset"},
    "build_dataset": {"experiment"},
    "experiment": {"drift_report"},
    "drift_report": {"register"},
    "register": set(),
}
MAPPED_TASKS = {"quality_checks", "build_training_set"}


def check() -> list[str]:
    """Return the problems found; an empty list means the DAG is as specified."""
    # Example DAGs are off through AIRFLOW__CORE__LOAD_EXAMPLES, set by the Makefile.
    bag = DagBag(dag_folder=str(REPO / "dags"))
    if bag.import_errors:
        return [f"import error in {path}: {error}" for path, error in bag.import_errors.items()]

    dag = bag.dags.get(DAG_ID)
    if dag is None:
        return [f"DAG {DAG_ID} not found"]

    problems = []
    if set(dag.task_dict) != TASK_IDS:
        problems.append(f"tasks are {sorted(dag.task_dict)}, expected {sorted(TASK_IDS)}")
    for task_id, expected in EDGES.items():
        if task_id in dag.task_dict and dag.task_dict[task_id].downstream_task_ids != expected:
            problems.append(
                f"{task_id} must run directly before {sorted(expected)}, "
                f"found {sorted(dag.task_dict[task_id].downstream_task_ids)}"
            )
    for task_id in sorted(MAPPED_TASKS):
        task = dag.task_dict.get(task_id)
        if task is not None and not isinstance(task, MappedOperator):
            problems.append(f"{task_id} must be dynamically mapped")
    build = dag.task_dict.get("build_training_set")
    if build is not None and build.max_active_tis_per_dag != 1:
        problems.append(
            f"build_training_set max_active_tis_per_dag is {build.max_active_tis_per_dag}, "
            "expected 1"
        )
    if dag.max_active_runs != 1:
        problems.append(f"max_active_runs is {dag.max_active_runs}, expected 1")
    if dag.schedule is not None:
        problems.append(f"schedule is {dag.schedule}, expected None (triggered by hand)")
    return problems


def main() -> int:
    problems = check()
    if problems:
        for problem in problems:
            sys.stderr.write(f"FAILED: {problem}\n")
        return 1
    print("DAG OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
