import pytest

from ctr_pipeline import cli
from ctr_pipeline.errors import SkipStep


def test_unknown_command_fails():
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["nope"])
    assert excinfo.value.code != 0


def test_skip_maps_to_99(monkeypatch):
    def skip(args):
        raise SkipStep("x")

    monkeypatch.setattr(cli, "cmd_train", skip)
    assert cli.main(["train", "--date", "2014-10-22"]) == 99


def test_error_maps_to_1(monkeypatch):
    def boom(args):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "cmd_train", boom)
    assert cli.main(["train", "--date", "2014-10-22"]) == 1


def test_bad_date_rejected():
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["quality-check", "--date", "2014-13-01"])
    assert excinfo.value.code != 0


def test_unimplemented_handler_maps_to_1(monkeypatch):
    # Stub the handler: the real one builds a training set from data/, which tests never touch.
    def not_implemented(args):
        raise NotImplementedError("stub")

    monkeypatch.setattr(cli, "cmd_build_training_set", not_implemented)
    assert cli.main(["build-training-set", "--date", "2014-10-22"]) == 1


def test_parser_lists_subcommands():
    parser = cli.build_parser()
    subparsers = next(a for a in parser._actions if a.dest == "command")
    assert set(subparsers.choices) == {
        "load-hourly",
        "check-complete",
        "quality-check",
        "build-training-set",
        "build-dataset",
        "feature-drift",
        "train",
        "promote",
        "monitoring-init",
        "run-day",
    }
