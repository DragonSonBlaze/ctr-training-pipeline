"""Checks on the Grafana dashboard file. These read the JSON only, no services are needed."""

import json

from ctr_pipeline.config import REPO_ROOT

DASHBOARD = REPO_ROOT / "docker" / "grafana" / "dashboards" / "ctr-pipeline.json"
PROMOTIONS_TITLE = "Promoted models (registry versions)"
KNOWN_TABLES = (
    "pipeline_step_runs",
    "data_quality_results",
    "model_runs",
    "feature_daily_stats",
    "feature_drift",
)


def load_dashboard() -> dict:
    return json.loads(DASHBOARD.read_text())


def test_dashboard_is_valid_json_with_unique_panel_ids():
    panels = load_dashboard()["panels"]
    ids = [panel["id"] for panel in panels]
    assert len(ids) == len(set(ids))


def test_promotions_panel_lists_registered_versions_only():
    panels = [p for p in load_dashboard()["panels"] if p["title"] == PROMOTIONS_TITLE]
    assert len(panels) == 1
    targets = panels[0]["targets"]
    assert len(targets) == 1
    sql = targets[0]["rawSql"]
    assert "WHERE promoted AND model_version IS NOT NULL" in sql
    assert "DISTINCT ON" not in sql


def test_every_panel_query_reads_a_known_table():
    for panel in load_dashboard()["panels"]:
        for target in panel["targets"]:
            sql = target["rawSql"]
            assert any(table in sql for table in KNOWN_TABLES), panel["title"]
