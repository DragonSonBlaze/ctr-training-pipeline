"""The alert drill: its pure helpers and the break list. No database and no network.

The script lives in scripts/, so it is loaded by file path. Importing it must run nothing.
"""

import importlib.util
import sys

from ctr_pipeline.config import LAST_DAY, REPO_ROOT

DRILL_SCRIPT = REPO_ROOT / "scripts" / "alert_drill.py"


def _load_drill():
    spec = importlib.util.spec_from_file_location("alert_drill", DRILL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so that dataclasses can resolve string annotations.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


drill = _load_drill()


def test_rule_states_reads_every_group():
    payload = {
        "data": {
            "groups": [
                {
                    "name": "first",
                    "rules": [
                        {"uid": "rule-a", "state": "inactive", "health": "ok"},
                        {"uid": "rule-b", "state": "firing", "health": "ok"},
                    ],
                },
                {
                    "name": "second",
                    "rules": [{"uid": "rule-c", "state": "pending", "health": "ok"}],
                },
            ]
        }
    }
    assert drill.rule_states(payload) == {
        "rule-a": "inactive",
        "rule-b": "firing",
        "rule-c": "pending",
    }


def test_breaks_cover_every_alert_rule():
    rule_uids = set(drill.load_rule_titles(drill.ALERT_RULES_PATH))
    break_uids = {item.uid for item in drill.BREAKS}
    assert break_uids == rule_uids


def test_every_break_uses_the_drill_date():
    for item in drill.BREAKS:
        assert drill.DRILL_DATE in item.params, item.uid


def test_drill_date_is_after_the_dataset():
    assert drill.DRILL_DATE > LAST_DAY


def test_format_table_marks_missing_fire_and_recovery():
    rows = [
        drill.DrillRow(
            title="Pipeline step took too long",
            before="inactive",
            fired_s=52.4,
            recovered_s=60.6,
        ),
        drill.DrillRow(
            title="Input feature drift",
            before="inactive",
            fired_s=None,
            recovered_s=61.0,
        ),
        drill.DrillRow(
            title="Predicted click rate off from actual",
            before="firing",
            fired_s=3.0,
            recovered_s=None,
        ),
    ]
    table = drill.format_table(rows).splitlines()
    assert table[0] == "| Alert | Before | Fired after | Back to normal after |"
    assert table[1] == "|---|---|---|---|"
    assert table[2] == "| Pipeline step took too long | inactive | 52 s | 61 s |"
    assert table[3] == "| Input feature drift | inactive | did not fire | 61 s |"
    assert table[4] == "| Predicted click rate off from actual | firing | 3 s | did not recover |"
