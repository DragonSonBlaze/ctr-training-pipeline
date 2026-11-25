import json

import pytest

from ctr_pipeline.config import get_settings
from ctr_pipeline.contracts import FEATURE_COLUMNS
from ctr_pipeline.errors import MissingInputError, PipelineError
from ctr_pipeline.features import build_training_set, feature_stats, load_training_day
from helpers import hourly_row, write_hourly

D = "2014-10-22"

FIXTURE = [
    ("r1", "14102122", 1, "a1", "s1", "i1", "m1"),
    ("r2", "14102122", 0, "a1", "s1", "i1", "m1"),
    ("r3", "14102123", 1, "a1", "s2", "i2", "m1"),
    ("r4", "14102200", 0, "a1", "s1", "i1", "m1"),
    ("r5", "14102200", 1, "a1", "s1", "i1", "m1"),
    ("r6", "14102201", 0, "a2", "s1", "i1", "m1"),
]


def _row(spec):
    row_id, hour, click, app, site, ip, model = spec
    return hourly_row(
        hour,
        click,
        id=row_id,
        app_id=app,
        site_id=site,
        device_ip=ip,
        device_model=model,
    )


@pytest.fixture
def hourly(monkeypatch, tmp_path, spark):
    monkeypatch.setenv("CTR_DATA_DIR", str(tmp_path))
    hourly_dir = get_settings().hourly_dir
    write_hourly(spark, hourly_dir, [_row(spec) for spec in FIXTURE])
    return hourly_dir


def _build(date: str = D, require_complete_history: bool = False):
    return build_training_set(date, require_complete_history=require_complete_history)


def _by_id(spark) -> dict[str, dict]:
    rows = load_training_day(D).collect()
    return {row["id"]: row.asDict() for row in rows}


def test_only_day_d_rows_are_output(hourly, spark):
    _build()
    assert sorted(_by_id(spark)) == ["r4", "r5", "r6"]


def test_counts_use_only_earlier_hours(hourly, spark):
    _build()
    rows = _by_id(spark)
    assert rows["r4"]["app_prior_impr"] == 3
    assert rows["r4"]["app_prior_clicks"] == 2
    assert rows["r4"]["site_prior_impr"] == 2
    assert rows["r4"]["site_prior_clicks"] == 1
    assert rows["r4"]["device_prior_impr"] == 2
    assert rows["r4"]["device_prior_clicks"] == 1
    assert rows["r4"]["global_prior_ctr"] == pytest.approx(2 / 3)

    assert rows["r5"]["app_prior_impr"] == 3
    assert rows["r5"]["app_prior_clicks"] == 2
    assert rows["r5"]["site_prior_impr"] == 2
    assert rows["r5"]["site_prior_clicks"] == 1
    assert rows["r5"]["device_prior_impr"] == 2
    assert rows["r5"]["device_prior_clicks"] == 1
    assert rows["r5"]["global_prior_ctr"] == pytest.approx(2 / 3)

    assert rows["r6"]["app_prior_impr"] == 0
    assert rows["r6"]["app_prior_clicks"] == 0
    assert rows["r6"]["site_prior_impr"] == 4
    assert rows["r6"]["site_prior_clicks"] == 2
    assert rows["r6"]["device_prior_impr"] == 4
    assert rows["r6"]["device_prior_clicks"] == 2
    assert rows["r6"]["global_prior_ctr"] == pytest.approx(3 / 5)


def test_same_hour_rows_do_not_see_each_other(hourly, spark):
    _build()
    rows = _by_id(spark)
    for column in ("app_prior_impr", "app_prior_clicks", "site_prior_impr", "device_prior_impr"):
        assert rows["r4"][column] == rows["r5"][column], column
    assert rows["r4"]["global_prior_ctr"] == rows["r5"]["global_prior_ctr"]


def test_smoothed_rate(hourly, spark):
    _build()
    rows = _by_id(spark)
    assert rows["r6"]["app_ctr_smooth"] == pytest.approx(0.6)
    assert rows["r4"]["app_ctr_smooth"] == pytest.approx((2 + 20 * 2 / 3) / (3 + 20))


def test_future_rows_do_not_change_features(hourly, spark):
    _build()
    before = _by_id(spark)

    later = [
        _row(("r7", "14102223", 1, "a1", "s1", "i1", "m1")),
        hourly_row("14102300", 1, id="r8", app_id="a1", site_id="s1", device_ip="i1"),
    ]
    write_hourly(spark, hourly, later)
    _build()
    after = _by_id(spark)

    for row_id in ("r4", "r5", "r6"):
        for column in FEATURE_COLUMNS:
            assert before[row_id][column] == after[row_id][column], (row_id, column)


def test_rebuild_is_identical(hourly, spark):
    first = _build()
    before = _by_id(spark)
    second = _build()
    after = _by_id(spark)

    assert first.rows == second.rows == 3
    assert before == after
    folders = sorted(p.name for p in (get_settings().training_dir).iterdir() if p.is_dir())
    assert folders == ["_manifests", f"dt={D}"]


def test_manifest_written(hourly):
    _build()
    manifest_path = get_settings().manifests_dir / f"{D}.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["rows"] == 3
    assert manifest["feature_version"] == "v1"
    assert manifest["data_date"] == D
    assert manifest["feature_columns"][0] == "app_prior_impr"
    assert not list(manifest_path.parent.glob("*.tmp"))


def test_first_hour_of_data_raises(monkeypatch, tmp_path, spark):
    monkeypatch.setenv("CTR_DATA_DIR", str(tmp_path))
    hourly_dir = get_settings().hourly_dir
    write_hourly(spark, hourly_dir, [hourly_row("14102100", 1, id="f1")])
    with pytest.raises(PipelineError):
        _build("2014-10-21")


def test_incomplete_history_raises(hourly):
    with pytest.raises(MissingInputError):
        _build(require_complete_history=True)


def test_feature_stats_reports_mean_and_null_fraction(hourly, spark):
    _build()
    stats = feature_stats(load_training_day(D))
    assert set(stats) == set(FEATURE_COLUMNS)
    assert stats["app_prior_impr"]["mean"] == pytest.approx((3 + 3 + 0) / 3)
    assert stats["app_prior_impr"]["null_fraction"] == 0.0
    assert stats["global_prior_ctr"]["p50"] is not None
