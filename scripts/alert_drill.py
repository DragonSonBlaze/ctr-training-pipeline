"""Break each monitored metric on purpose and check that every Grafana alert fires.

Run with make alert-drill (needs make up). The drill writes one broken monitoring row per
alert rule, dated DRILL_DATE, waits until Grafana reports each rule firing, deletes the rows,
waits until each rule is back to inactive, and prints a Markdown table of the timings.

Grafana is only read. The drill never changes an alert rule or a dashboard.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import psycopg
import yaml

from ctr_pipeline.config import REPO_ROOT, get_settings

logger = logging.getLogger(__name__)

# A date after the Avazu data (ends 2014-10-30). The marker keeps the broken rows apart from
# real rows, and makes the clean-up a DELETE on one value. Every alert reads the latest date
# or the latest finished row, so these rows become the latest and trip the rules.
DRILL_DATE = "2099-01-01"
DRILL_REFERENCE_DATE = "2098-12-31"
ALERT_RULES_PATH = (
    REPO_ROOT / "docker" / "grafana" / "provisioning" / "alerting" / "ctr-alerts.yaml"
)
SLOW_STEP_SECONDS = 9999.0
POLL_INTERVAL_S = 10
# Grafana evaluates each rule once a minute, so a rule can take about 70 s to react.
FIRE_TIMEOUT_S = 240
RECOVER_TIMEOUT_S = 240
HTTP_TIMEOUT_S = 10
FIRING = "firing"
INACTIVE = "inactive"
DB_CONNECT_TIMEOUT_S = 3

CLEANUP_SQL: tuple[str, ...] = (
    "DELETE FROM pipeline_step_runs WHERE data_date = %s",
    "DELETE FROM feature_drift WHERE data_date = %s",
    "DELETE FROM model_runs WHERE data_date = %s",
)


@dataclass(frozen=True)
class Break:
    uid: str
    sql: str
    params: tuple[object, ...]


BREAKS: tuple[Break, ...] = (
    Break(
        uid="ctr-step-run-time",
        sql=(
            "INSERT INTO pipeline_step_runs "
            "(step, data_date, started_at, finished_at, duration_s, status) "
            "VALUES (%s, %s, now() - make_interval(secs => %s), now(), %s, 'success')"
        ),
        params=("alert-drill-slow", DRILL_DATE, SLOW_STEP_SECONDS, SLOW_STEP_SECONDS),
    ),
    Break(
        uid="ctr-step-failed",
        sql=(
            "INSERT INTO pipeline_step_runs "
            "(step, data_date, started_at, finished_at, duration_s, status, error) "
            "VALUES (%s, %s, now(), now(), 1, 'failed', %s)"
        ),
        params=("alert-drill-failed", DRILL_DATE, "alert drill"),
    ),
    Break(
        uid="ctr-row-count",
        sql=(
            "INSERT INTO pipeline_step_runs "
            "(step, data_date, started_at, finished_at, duration_s, status, rows_out) "
            "VALUES (%s, %s, now(), now(), 1, 'success', %s)"
        ),
        params=("build-training-set", DRILL_DATE, 10),
    ),
    Break(
        uid="ctr-feature-drift",
        sql=(
            "INSERT INTO feature_drift (data_date, reference_date, feature, distance) "
            "VALUES (%s, %s, %s, %s)"
        ),
        params=(DRILL_DATE, DRILL_REFERENCE_DATE, "alert_drill", 1.0),
    ),
    Break(
        uid="ctr-calibration",
        sql=(
            "INSERT INTO model_runs "
            "(mlflow_run_id, data_date, model_type, dataset_version, mean_pred, actual_ctr) "
            "VALUES (%s, %s, %s, %s, %s, %s)"
        ),
        params=("alert-drill", DRILL_DATE, "sklearn_logistic", "alert-drill", 0.5, 0.17),
    ),
)


@dataclass(frozen=True)
class DrillRow:
    title: str
    before: str
    fired_s: float | None
    recovered_s: float | None


def load_rule_titles(path: Path = ALERT_RULES_PATH) -> dict[str, str]:
    """Map each alert rule uid to its title, in the order the rule file lists them."""
    document = yaml.safe_load(path.read_text())
    return {rule["uid"]: rule["title"] for group in document["groups"] for rule in group["rules"]}


def rule_states(payload: dict) -> dict[str, str]:
    """Map each rule uid to its state ("inactive", "pending" or "firing") over every group."""
    states: dict[str, str] = {}
    for group in payload["data"]["groups"]:
        for rule in group["rules"]:
            states[rule["uid"]] = rule["state"]
    return states


def grafana_rules_url() -> str:
    port = os.environ.get("GRAFANA_PORT", "3000")
    return f"http://localhost:{port}/api/prometheus/grafana/api/v1/rules"


def grafana_auth_header() -> str:
    user = os.environ.get("GRAFANA_ADMIN_USER", "admin")
    password = os.environ.get("GRAFANA_ADMIN_PASSWORD")
    if not password:
        raise RuntimeError("GRAFANA_ADMIN_PASSWORD is not set; check .env")
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return f"Basic {token}"


def fetch_rule_states() -> dict[str, str]:
    # get_settings() loads .env into os.environ, which the Grafana helpers read.
    get_settings()
    request = urllib.request.Request(grafana_rules_url())
    request.add_header("Authorization", grafana_auth_header())
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
        payload = json.loads(response.read())
    return rule_states(payload)


def remove_drill_rows(dsn: str) -> None:
    """Delete every drill row in one transaction. The with block commits on success."""
    with psycopg.connect(dsn, connect_timeout=DB_CONNECT_TIMEOUT_S) as conn:
        for statement in CLEANUP_SQL:
            conn.execute(statement, (DRILL_DATE,))


def insert_breaks(dsn: str, breaks: tuple[Break, ...]) -> None:
    """Insert every broken row in one transaction, so either all of them land or none do."""
    with psycopg.connect(dsn, connect_timeout=DB_CONNECT_TIMEOUT_S) as conn:
        for item in breaks:
            conn.execute(item.sql, item.params)


def wait_for_states(
    uids: tuple[str, ...],
    wanted: str,
    since: float,
    timeout_s: float,
) -> dict[str, float]:
    """Poll Grafana until each rule reaches the wanted state or the timeout passes.

    Returns the seconds after `since` at which each rule first reached the state. Rules that
    never reached it are missing from the result.
    """
    reached: dict[str, float] = {}
    deadline = since + timeout_s
    while True:
        states = fetch_rule_states()
        now = time.monotonic()
        for uid in uids:
            if uid not in reached and states.get(uid) == wanted:
                reached[uid] = now - since
        if len(reached) == len(uids) or now >= deadline:
            return reached
        logger.info(
            "waiting for %d of %d rules to reach %s", len(uids) - len(reached), len(uids), wanted
        )
        time.sleep(POLL_INTERVAL_S)


def format_table(rows: list[DrillRow]) -> str:
    lines = [
        "| Alert | Before | Fired after | Back to normal after |",
        "|---|---|---|---|",
    ]
    for row in rows:
        fired = "did not fire" if row.fired_s is None else f"{round(row.fired_s)} s"
        recovered = "did not recover" if row.recovered_s is None else f"{round(row.recovered_s)} s"
        lines.append(f"| {row.title} | {row.before} | {fired} | {recovered} |")
    return "\n".join(lines)


def run_drill(dsn: str, titles: dict[str, str]) -> int:
    uids = tuple(titles)
    starting = fetch_rule_states()
    missing = [uid for uid in uids if uid not in starting]
    if missing:
        print(
            f"Grafana has no rule for: {', '.join(missing)}. Check that make up loaded the rules."
        )
        return 1
    before = {uid: starting[uid] for uid in uids}

    remove_drill_rows(dsn)
    try:
        logger.info("writing %d broken rows dated %s", len(BREAKS), DRILL_DATE)
        since_break = time.monotonic()
        insert_breaks(dsn, BREAKS)
        fired = wait_for_states(uids, FIRING, since_break, FIRE_TIMEOUT_S)
    finally:
        remove_drill_rows(dsn)
        logger.info("removed the broken rows")

    since_restore = time.monotonic()
    recovered = wait_for_states(uids, INACTIVE, since_restore, RECOVER_TIMEOUT_S)

    rows = [
        DrillRow(
            title=titles[uid],
            before=before[uid],
            fired_s=fired.get(uid),
            recovered_s=recovered.get(uid),
        )
        for uid in uids
    ]
    print(format_table(rows))
    passed = all(row.fired_s is not None and row.recovered_s is not None for row in rows)
    return 0 if passed else 1


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dsn = get_settings().monitoring_dsn
    titles = load_rule_titles()
    try:
        return run_drill(dsn, titles)
    except (OSError, psycopg.Error, RuntimeError) as exc:
        print(f"alert drill stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
