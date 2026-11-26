"""Command-line entry point: one subcommand per pipeline step.

Airflow only calls these subcommands, so every step can also run by hand.
Exit codes are decided once, in main(): 0 ok, 1 failed, 99 skipped.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable

from ctr_pipeline.config import DRIFT_ALERT_THRESHOLD, previous_day, validate_date
from ctr_pipeline.dataset import build_dataset, latest_version
from ctr_pipeline.drift import drift_for_date
from ctr_pipeline.drift_report import run_drift_report
from ctr_pipeline.errors import QualityCheckFailed, SkipStep
from ctr_pipeline.experiment import run_experiment
from ctr_pipeline.features import build_training_set, feature_stats, load_training_day
from ctr_pipeline.hourly_load import is_day_complete, load_hourly, missing_hours
from ctr_pipeline.monitoring import (
    init_schema,
    record_feature_drift,
    record_feature_stats,
    record_quality_results,
    record_step,
)
from ctr_pipeline.promotion import candidate_run_ids, register_if_better
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


def cmd_build_dataset(args: argparse.Namespace) -> int:
    with record_step("build-dataset", None) as stats:
        result = build_dataset()
        stats.rows_out = sum(block["rows"] for block in result.manifest["splits"].values())
    # The version must be the last line of stdout, alone: Airflow reads it, as it reads the run id.
    sys.stdout.write(result.version + "\n")
    sys.stdout.flush()
    return EXIT_OK


def cmd_feature_drift(args: argparse.Namespace) -> int:
    """Compare the day's smoothed features with the day before and store the distance of each.

    Drift never fails the step: it is for alerting, and a large value must not stop training.
    A SkipStep (the first training day) propagates, so main() exits 99.
    """
    date = args.date
    with record_step("feature-drift", date):
        values = drift_for_date(date)
        record_feature_drift(date, previous_day(date), values)
        for feature, distance in values.items():
            logger.info("%s %.4f", feature, distance)
            if distance > DRIFT_ALERT_THRESHOLD:
                logger.warning(
                    "%s drifted on %s: distance %.4f is above %.2f",
                    feature,
                    date,
                    distance,
                    DRIFT_ALERT_THRESHOLD,
                )
    return EXIT_OK


def cmd_drift_report(args: argparse.Namespace) -> int:
    with record_step("drift-report", None):
        summary = run_drift_report(args.dataset, args.parent_run_id)
    logger.info(
        "drift report for %s: drift_share=%.3f, drifted_columns=%d",
        args.dataset,
        summary.drift_share,
        summary.drifted_columns,
    )
    return EXIT_OK


def cmd_experiment(args: argparse.Namespace) -> int:
    version = args.dataset or latest_version()
    with record_step("experiment", None):
        result = run_experiment(version)
    logger.info(
        "experiment %s: baseline val_log_loss=%.6f, best trial %s val_log_loss=%.6f",
        version,
        result.baseline.val_log_loss,
        result.best_trial.run_id,
        result.best_trial.val_log_loss,
    )
    # The parent run id must be the last line of stdout, alone: Airflow reads it.
    sys.stdout.write(result.parent_run_id + "\n")
    sys.stdout.flush()
    return EXIT_OK


def cmd_experiment_candidates(args: argparse.Namespace) -> int:
    baseline_run_id, best_trial_run_id = candidate_run_ids(args.parent_run_id)
    # The two run ids must be the last line of stdout, space-separated, for the DAG to read.
    sys.stdout.write(f"{baseline_run_id} {best_trial_run_id}\n")
    sys.stdout.flush()
    return EXIT_OK


def cmd_register(args: argparse.Namespace) -> int:
    with record_step("register", None):
        register_if_better(args.dataset, args.run_id)
    # Not registering is a normal outcome, so the exit code stays 0 either way.
    return EXIT_OK


def cmd_monitoring_init(args: argparse.Namespace) -> int:
    init_schema()
    logger.info("monitoring tables ready")
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

    dataset = subcommands.add_parser(
        "build-dataset", help="versioned train, validation and test split over the daily sets"
    )
    dataset.set_defaults(handler=cmd_build_dataset)

    experiment = subcommands.add_parser(
        "experiment", help="baseline and Keras sweep over one dataset, tracked in MLflow"
    )
    experiment.add_argument(
        "--dataset", default=None, help="dataset version (default: the newest one)"
    )
    experiment.set_defaults(handler=cmd_experiment)

    candidates = subcommands.add_parser(
        "experiment-candidates",
        help="print the baseline and best Keras trial run ids of an experiment run",
    )
    candidates.add_argument(
        "--parent-run-id", required=True, help="parent run id printed by experiment"
    )
    candidates.set_defaults(handler=cmd_experiment_candidates)

    register = subcommands.add_parser(
        "register",
        help="score candidates on the test split, register the best if it beats the champion",
    )
    register.add_argument("--dataset", required=True, help="dataset version to test on")
    register.add_argument(
        "--run-id",
        action="append",
        required=True,
        help="MLflow run id of a candidate; repeat for each candidate",
    )
    register.set_defaults(handler=cmd_register)

    drift = subcommands.add_parser(
        "feature-drift", help="Distance of each smoothed feature from the day before"
    )
    drift.add_argument("--date", type=validate_date, required=True, help="YYYY-MM-DD")
    drift.set_defaults(handler=cmd_feature_drift)

    drift_report = subcommands.add_parser(
        "drift-report",
        help="Evidently drift of the test days against the train days, logged to a parent run",
    )
    drift_report.add_argument("--dataset", required=True, help="dataset version")
    drift_report.add_argument(
        "--parent-run-id", required=True, help="MLflow parent run id printed by experiment"
    )
    drift_report.set_defaults(handler=cmd_drift_report)

    init = subcommands.add_parser("monitoring-init", help="create monitoring tables if missing")
    init.set_defaults(handler=cmd_monitoring_init)

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
