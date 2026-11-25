import argparse

import pytest

from ctr_pipeline import cli
from ctr_pipeline.errors import QualityCheckFailed, SkipStep

DATE = "2014-10-22"
RUN_ID = "0123456789abcdef0123456789abcdef"
EMBEDDING_RUN_ID = "fedcba9876543210fedcba9876543210"
RUN_IDS = {"logistic_hashed": RUN_ID, "embedding_nn": EMBEDDING_RUN_ID}


@pytest.fixture
def calls(monkeypatch) -> list[str]:
    """Replace the five step functions run-day uses with fakes that record their calls."""
    recorded: list[str] = []

    def complete(date: str, hourly_dir=None) -> bool:
        recorded.append("check-complete")
        return True

    def quality(args: argparse.Namespace) -> int:
        recorded.append("quality-check")
        return cli.EXIT_OK

    def build(args: argparse.Namespace) -> int:
        recorded.append("build-training-set")
        return cli.EXIT_OK

    def train(date: str, model_type: str) -> str:
        recorded.append(f"train:{model_type}")
        return RUN_IDS[model_type]

    def promote(args: argparse.Namespace) -> int:
        recorded.append(f"promote:{','.join(args.run_id)}")
        return cli.EXIT_OK

    monkeypatch.setattr(cli, "is_day_complete", complete)
    monkeypatch.setattr(cli, "cmd_quality_check", quality)
    monkeypatch.setattr(cli, "cmd_build_training_set", build)
    monkeypatch.setattr(cli, "_run_training", train)
    monkeypatch.setattr(cli, "cmd_promote", promote)
    return recorded


def test_steps_run_in_order_and_promote_gets_every_train_run_id(calls):
    assert cli.main(["run-day", "--date", DATE]) == 0
    assert calls == [
        "check-complete",
        "quality-check",
        "build-training-set",
        "train:logistic_hashed",
        "train:embedding_nn",
        f"promote:{RUN_ID},{EMBEDDING_RUN_ID}",
    ]


def test_skip_from_train_stops_before_promote(monkeypatch, calls):
    def skip(date: str, model_type: str) -> str:
        calls.append(f"train:{model_type}")
        raise SkipStep("no earlier training sets")

    monkeypatch.setattr(cli, "_run_training", skip)
    assert cli.main(["run-day", "--date", DATE]) == 99
    assert calls == [
        "check-complete",
        "quality-check",
        "build-training-set",
        "train:logistic_hashed",
    ]


def test_failed_quality_check_stops_before_build(monkeypatch, calls):
    def failing(args: argparse.Namespace) -> int:
        calls.append("quality-check")
        raise QualityCheckFailed("failed checks: min_rows_per_day")

    monkeypatch.setattr(cli, "cmd_quality_check", failing)
    assert cli.main(["run-day", "--date", DATE]) == 1
    assert calls == ["check-complete", "quality-check"]


def test_incomplete_day_stops_before_quality_check(monkeypatch, calls):
    monkeypatch.setattr(cli, "is_day_complete", lambda date, hourly_dir=None: False)
    assert cli.main(["run-day", "--date", DATE]) == 1
    assert calls == []
