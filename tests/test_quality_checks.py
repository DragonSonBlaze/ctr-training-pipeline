import pytest

from ctr_pipeline.config import get_settings
from ctr_pipeline.quality_checks import QualityThresholds, run_quality_checks
from helpers import hourly_row, write_hourly

DAY = "2014-10-22"
THRESHOLDS = QualityThresholds(
    min_rows_per_hour=5,
    min_rows_per_day=100,
    click_rate_range=(0.10, 0.25),
    max_null_fraction=0.01,
)
CHECK_NAMES = [
    "schema_matches_contract",
    "hours_present",
    "min_rows_per_hour",
    "min_rows_per_day",
    "required_columns_not_null",
    "max_null_fraction",
    "click_values_binary",
    "click_rate_in_range",
]


@pytest.fixture
def hourly_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CTR_DATA_DIR", str(tmp_path))
    return get_settings().hourly_dir


def good_day_rows(hours=range(24), per_hour=10, overrides=None):
    """Each hour has per_hour rows with 2 clicks, so the click rate is 0.2."""
    rows = []
    for hour in hours:
        hour_text = f"141022{hour:02d}"
        for position in range(per_hour):
            click = 1 if position < 2 else 0
            rows.append(hourly_row(hour_text, click))
    return rows


def _by_name(results):
    return {result.name: result for result in results}


def _failed_names(results):
    return [result.name for result in results if not result.passed]


def test_good_day_passes_all(spark, hourly_dir):
    write_hourly(spark, hourly_dir, good_day_rows())
    results = run_quality_checks(DAY, THRESHOLDS)
    assert [r.name for r in results] == CHECK_NAMES
    assert all(r.passed for r in results)


def test_missing_hour_fails_hours_present(spark, hourly_dir):
    write_hourly(spark, hourly_dir, good_day_rows(hours=range(23)))
    results = _by_name(run_quality_checks(DAY, THRESHOLDS))
    assert results["hours_present"].passed is False
    assert results["hours_present"].observed == 23
    assert _failed_names(results.values()) == ["hours_present"]


def test_all_clicks_fail_click_rate(spark, hourly_dir):
    rows = [hourly_row(r["hour"], 1, id=r["id"]) for r in good_day_rows()]
    write_hourly(spark, hourly_dir, rows)
    results = _by_name(run_quality_checks(DAY, THRESHOLDS))
    assert _failed_names(results.values()) == ["click_rate_in_range"]


def test_non_binary_click_fails(spark, hourly_dir):
    rows = good_day_rows()
    rows[0] = hourly_row("14102200", 2)
    write_hourly(spark, hourly_dir, rows)
    results = _by_name(run_quality_checks(DAY, THRESHOLDS))
    assert results["click_values_binary"].passed is False
    assert results["click_values_binary"].observed == 1


def test_null_in_required_column_fails(spark, hourly_dir):
    rows = good_day_rows()
    rows[0] = hourly_row("14102200", 0, id=None)
    write_hourly(spark, hourly_dir, rows)
    results = _by_name(run_quality_checks(DAY, THRESHOLDS))
    assert results["required_columns_not_null"].passed is False


def test_nulls_in_optional_column(spark, hourly_dir):
    rows = good_day_rows()
    for i in range(5):
        rows[i] = hourly_row(rows[i]["hour"], rows[i]["click"], app_id=None)
    write_hourly(spark, hourly_dir, rows)
    results = _by_name(run_quality_checks(DAY, THRESHOLDS))
    assert results["max_null_fraction"].passed is False
    assert results["max_null_fraction"].observed == pytest.approx(0.0208, abs=1e-4)
    assert results["required_columns_not_null"].passed is True


def test_thin_hour_fails_min_rows_per_hour(spark, hourly_dir):
    rows = [r for r in good_day_rows() if r["hour"] != "14102207"]
    rows += [hourly_row("14102207", click) for click in (1, 0, 0)]
    write_hourly(spark, hourly_dir, rows)
    results = _by_name(run_quality_checks(DAY, THRESHOLDS))
    assert results["min_rows_per_hour"].passed is False
    assert results["min_rows_per_hour"].observed == 3


def test_empty_day_fails_everything_without_raising(spark, hourly_dir):
    write_hourly(spark, hourly_dir, good_day_rows())
    results = run_quality_checks("2014-10-25", THRESHOLDS)
    assert len(results) == 8
    assert all(not r.passed for r in results)
