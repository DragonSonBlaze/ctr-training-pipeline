"""Command-line entry point: one subcommand per pipeline step.

Airflow only calls these subcommands, so every step can also run by hand.
Exit codes are decided once, in main(): 0 ok, 1 failed, 99 skipped.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable

from ctr_pipeline.config import DAILY_MODEL_TYPES, DEFAULT_MODEL_TYPE, validate_date
from ctr_pipeline.errors import MissingInputError, QualityCheckFailed, SkipStep
from ctr_pipeline.features import build_training_set, feature_stats, load_training_day
from ctr_pipeline.hourly_load import is_day_complete, load_hourly, missing_hours
from ctr_pipeline.models.train import MODEL_TYPES, train_for_date
from ctr_pipeline.monitoring import (
    init_schema,
    record_feature_stats,
    record_quality_results,
    record_step,
)
from ctr_pipeline.promotion import promote_if_better
from ctr_pipeline.quality_checks import CheckResult, run_quality_checks

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_SKIPPED = 99

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def cmd_load_hourly(args: argparse.Namespace) -> int:
    with record_step("load-hourly", args.date) as stats:
        result = load_hourly(args.date)
        stats.rows_in = result.rows_read
        stats.rows_out = result.rows_written
    return EXIT_OK


def cmd_check_complete(args: argparse.Namespace) -> int:
    if is_day_complete(args.date):
        return EXIT_OK
    logger.info("%s is not complete yet, missing hours: %s", args.date, missing_hours(args.date))
    return EXIT_FAILED


def cmd_quality_check(args: argparse.Namespace) -> int:
    with record_step("quality-check", args.date) as stats:
        results = run_quality_checks(args.date)
        record_quality_results(args.date, results)
        stats.rows_in = _observed_int(results, "min_rows_per_day")
        for result in results:
            status = "PASS" if result.passed else "FAIL"
            logger.info(
                "%s %s observed=%s threshold=%s",
                status,
                result.name,
                result.observed,
                result.threshold,
            )
        failed = [result.name for result in results if not result.passed]
        if failed:
            raise QualityCheckFailed(f"{args.date}: failed checks: {', '.join(failed)}")
    return EXIT_OK


def _observed_int(results: list[CheckResult], name: str) -> int | None:
    for result in results:
        if result.name == name and result.observed is not None:
            return int(result.observed)
    return None


def cmd_build_training_set(args: argparse.Namespace) -> int:
    with record_step("build-training-set", args.date) as stats:
        result = build_training_set(args.date)
        stats.rows_out = result.rows
        record_feature_stats(args.date, feature_stats(load_training_day(args.date)))
    logger.info("training set for %s: %d rows at %s", args.date, result.rows, result.path)
    return EXIT_OK


def _run_training(date: str, model_type: str = DEFAULT_MODEL_TYPE) -> str:
    """Fit, evaluate and log one day's model, and return the MLflow run id.

    Raises SkipStep when there are no earlier training sets to fit on.
    """
    with record_step("train", date) as stats:
        result = train_for_date(date, model_type)
        stats.rows_in = result.train_rows
    logger.info(
        "run %s: model version %s, metrics %s",
        result.run_id,
        result.model_version,
        result.metrics,
    )
    return result.run_id


def cmd_train(args: argparse.Namespace) -> int:
    run_id = _run_training(args.date, args.model)
    # The run id must be the last line of stdout, alone: Airflow reads it and passes it to promote.
    sys.stdout.write(run_id + "\n")
    sys.stdout.flush()
    return EXIT_OK


def cmd_promote(args: argparse.Namespace) -> int:
    with record_step("promote", args.date):
        promote_if_better(args.date, args.run_id)
    # Not promoting is a normal outcome, so the exit code stays 0 either way.
    return EXIT_OK


def cmd_monitoring_init(args: argparse.Namespace) -> int:
    init_schema()
    logger.info("monitoring tables ready")
    return EXIT_OK


def cmd_run_day(args: argparse.Namespace) -> int:
    date = args.date
    if not is_day_complete(date):
        raise MissingInputError(f"{date}: the 24 hourly folders are not all present yet")
    cmd_quality_check(argparse.Namespace(date=date))
    cmd_build_training_set(argparse.Namespace(date=date))
    # One model at a time: the models would compete for memory if they ran in parallel.
    run_ids: list[str] = []
    for model_type in DAILY_MODEL_TYPES:
        try:
            run_ids.append(_run_training(date, model_type))
        except SkipStep as skip:
            logger.info("train skipped for %s, promote not run: %s", date, skip)
            return EXIT_SKIPPED
    cmd_promote(argparse.Namespace(date=date, run_id=run_ids))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ctr_pipeline",
        description="CTR training-data pipeline: one subcommand per step.",
    )
    subcommands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    load = subcommands.add_parser("load-hourly", help="CSV to hourly Parquet, all days or one day")
    load.add_argument("--date", type=validate_date, default=None, help="YYYY-MM-DD")
    load.set_defaults(handler=cmd_load_hourly)

    complete = subcommands.add_parser(
        "check-complete", help="are all 24 hour folders for a day present"
    )
    complete.add_argument("--date", type=validate_date, required=True, help="YYYY-MM-DD")
    complete.set_defaults(handler=cmd_check_complete)

    quality = subcommands.add_parser("quality-check", help="run checks and store results")
    quality.add_argument("--date", type=validate_date, required=True, help="YYYY-MM-DD")
    quality.set_defaults(handler=cmd_quality_check)

    build = subcommands.add_parser("build-training-set", help="features for one day")
    build.add_argument("--date", type=validate_date, required=True, help="YYYY-MM-DD")
    build.set_defaults(handler=cmd_build_training_set)

    train = subcommands.add_parser(
        "train", help="fit on earlier days, evaluate on one day, log to MLflow"
    )
    train.add_argument("--date", type=validate_date, required=True, help="YYYY-MM-DD")
    train.add_argument(
        "--model",
        choices=sorted(MODEL_TYPES),
        default=DEFAULT_MODEL_TYPE,
        help="model type to fit (default: %(default)s)",
    )
    train.set_defaults(handler=cmd_train)

    promote = subcommands.add_parser(
        "promote", help="pick the best candidate of the day and move the champion alias if better"
    )
    promote.add_argument("--date", type=validate_date, required=True, help="YYYY-MM-DD")
    promote.add_argument(
        "--run-id",
        action="append",
        required=True,
        help="MLflow run id from train; repeat for each candidate",
    )
    promote.set_defaults(handler=cmd_promote)

    init = subcommands.add_parser("monitoring-init", help="create monitoring tables if missing")
    init.set_defaults(handler=cmd_monitoring_init)

    day = subcommands.add_parser("run-day", help="every step in order, for one day")
    day.add_argument("--date", type=validate_date, required=True, help="YYYY-MM-DD")
    day.set_defaults(handler=cmd_run_day)

    return parser


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = build_parser().parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    try:
        return handler(args)
    except SkipStep as skip:
        logger.info("skipped: %s", skip)
        return EXIT_SKIPPED
    except Exception:
        logger.exception("%s failed", args.command)
        return EXIT_FAILED
