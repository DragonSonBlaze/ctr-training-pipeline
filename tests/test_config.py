import os

import pytest

from ctr_pipeline.config import (
    day_range,
    get_settings,
    load_dotenv,
    previous_day,
    validate_date,
)


def test_day_range_inclusive():
    assert day_range("2014-10-21", "2014-10-23") == ["2014-10-21", "2014-10-22", "2014-10-23"]
    assert day_range("2014-10-23", "2014-10-21") == []


def test_previous_day_crosses_month():
    assert previous_day("2014-11-01") == "2014-10-31"


def test_validate_date_rejects_bad_input():
    with pytest.raises(ValueError):
        validate_date("2014-10-32")
    with pytest.raises(ValueError):
        validate_date("14102100")
    assert validate_date("2014-10-22") == "2014-10-22"


def test_settings_follow_ctr_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CTR_DATA_DIR", str(tmp_path))
    settings = get_settings()
    assert settings.hourly_dir == tmp_path / "hourly" / "events"
    assert settings.training_dir == tmp_path / "training" / "v1"


def test_load_dotenv_does_not_override(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\n\nCTR_TEST_FOO=from_file\nCTR_TEST_NEW =  from_file_new \n")
    monkeypatch.setenv("CTR_TEST_FOO", "from_env")
    monkeypatch.delenv("CTR_TEST_NEW", raising=False)

    load_dotenv(env_file)

    assert os.environ["CTR_TEST_FOO"] == "from_env"
    assert os.environ["CTR_TEST_NEW"] == "from_file_new"


def test_load_dotenv_missing_file_does_nothing(tmp_path):
    load_dotenv(tmp_path / "missing.env")
