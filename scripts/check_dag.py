"""Load the ctr_daily DAG the way Airflow does and check its shape.

Run in the Airflow venv: .venv-airflow/bin/python scripts/check_dag.py
"""

import sys
from pathlib import Path

from airflow.dag_processing.dagbag import DagBag

REPO = Path(__file__).resolve().parents[1]
DAG_ID = "ctr_daily"
TASK_ORDER = [
    "wait_for_day",
    "quality_checks",
    "build_training_set",
    "feature_drift",
    "train_logistic",
    "train_embedding",
    "promote",
]


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
    if set(dag.task_dict) != set(TASK_ORDER):
        problems.append(f"tasks are {sorted(dag.task_dict)}, expected {sorted(TASK_ORDER)}")
    for upstream, downstream in zip(TASK_ORDER, TASK_ORDER[1:], strict=False):
        if upstream in dag.task_dict and dag.task_dict[upstream].downstream_task_ids != {
            downstream
        }:
            problems.append(f"{upstream} must run directly before {downstream}")
    train = dag.task_dict.get("train_logistic")
    # A skipped drift check (first training day) must not block training.
    if train is not None and train.trigger_rule != "none_failed":
        problems.append("train_logistic must use trigger_rule none_failed")
    if dag.max_active_runs != 1:
        problems.append(f"max_active_runs is {dag.max_active_runs}, expected 1")
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
