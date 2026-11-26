"""Checks on dags/ctr_training.py that run in the project venv, without Airflow.

The DAG itself is loaded and checked by scripts/check_dag.py in the Airflow venv.
These tests read the DAG file as text or with ast, so a day list or a CLI
subcommand that drifts from the pipeline fails here.
"""

import argparse
import ast
import re

from ctr_pipeline import cli, config

DAG_PATH = config.REPO_ROOT / "dags" / "ctr_training.py"


def _dag_source() -> str:
    return DAG_PATH.read_text()


def _literal_assignment(name: str) -> list[str]:
    tree = ast.parse(_dag_source())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, list)
            return value
    raise AssertionError(f"{name} is not a module-level list in {DAG_PATH.name}")


def test_dag_days_match_config() -> None:
    assert _literal_assignment("ALL_DAYS") == config.day_range(config.FIRST_DAY, config.LAST_DAY)
    assert _literal_assignment("TRAINING_DAYS") == config.day_range(
        config.FIRST_TRAINING_DAY, config.LAST_DAY
    )


def test_dag_does_not_import_ctr_pipeline() -> None:
    source = _dag_source()
    assert "import ctr_pipeline" not in source
    assert "from ctr_pipeline" not in source


def test_every_cli_subcommand_in_dag_exists() -> None:
    named = set(re.findall(r"-m ctr_pipeline\s+([a-z][a-z-]*)", _dag_source()))
    assert named, "the DAG should call at least one ctr_pipeline subcommand"

    parser = cli.build_parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    available = set(subparsers.choices)
    missing = named - available
    assert not missing, f"DAG calls subcommands that do not exist: {sorted(missing)}"
