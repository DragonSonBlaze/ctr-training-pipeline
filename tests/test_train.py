import mlflow
import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from ctr_pipeline.config import LR_EPOCHS, NUM_HASH_BUCKETS, REGISTERED_MODEL
from ctr_pipeline.contracts import MODEL_INPUT_COLUMNS
from ctr_pipeline.errors import MissingInputError, SkipStep
from ctr_pipeline.models.historical_ctr import HistoricalCtrModel
from ctr_pipeline.models.train import (
    dataset_version,
    load_eval_frame,
    predict_day,
    train_for_date,
)
from helpers import training_frame, write_training_day

EVAL_DAY = "2014-10-24"


@pytest.fixture
def days(mlflow_sandbox):
    write_training_day(mlflow_sandbox, "2014-10-22", training_frame(1000))
    write_training_day(mlflow_sandbox, "2014-10-23", training_frame(1000))
    write_training_day(mlflow_sandbox, EVAL_DAY, training_frame(1000))
    return mlflow_sandbox


def test_skip_on_first_training_day(days):
    with pytest.raises(SkipStep):
        train_for_date("2014-10-22")


def test_missing_manifest_raises(days):
    (days.manifests_dir / "2014-10-23.json").unlink()
    with pytest.raises(MissingInputError):
        train_for_date(EVAL_DAY)


def test_run_logged_with_dataset_version_params_and_metrics(days):
    result = train_for_date(EVAL_DAY, "historical_ctr")
    run = MlflowClient().get_run(result.run_id)

    assert run.data.tags["data_date"] == EVAL_DAY
    assert run.data.tags["dataset_version"].startswith("v1:2014-10-22..2014-10-23:")
    assert run.data.params["beta"] == "20.0"
    assert run.data.params["train_start"] == "2014-10-22"
    assert run.data.params["train_end"] == "2014-10-23"
    assert run.data.params["sample_fraction"] == "0.15"

    assert "log_loss" in run.data.metrics
    assert "auc" in run.data.metrics
    assert 0.1 < run.data.metrics["fitted_p0"] < 0.3
    assert 250 <= run.data.metrics["train_rows"] <= 350


def test_training_registers_no_version_and_model_loads_from_run(days):
    model = HistoricalCtrModel()
    result = train_for_date(EVAL_DAY, model=model)

    assert len(MlflowClient().search_model_versions(f"name='{REGISTERED_MODEL}'")) == 0

    loaded = mlflow.pyfunc.load_model(f"runs:/{result.run_id}/model")
    eval_frame = load_eval_frame(EVAL_DAY, MODEL_INPUT_COLUMNS)
    np.testing.assert_allclose(loaded.predict(eval_frame), model.predict(eval_frame))


def test_dataset_version_is_deterministic(days):
    days_used = ["2014-10-22", "2014-10-23"]
    assert dataset_version(days_used) == dataset_version(days_used)


def test_logistic_run_logged(days):
    result = train_for_date(EVAL_DAY, "logistic_hashed")
    run = MlflowClient().get_run(result.run_id)

    assert run.data.tags["model_type"] == "logistic_hashed"
    assert run.data.params["num_buckets"] == str(NUM_HASH_BUCKETS)
    assert run.data.params["epochs"] == str(LR_EPOCHS)
    assert "log_loss" in run.data.metrics


def test_unknown_model_type_raises(days):
    with pytest.raises(ValueError):
        train_for_date(EVAL_DAY, "no_such_model")


def test_batched_prediction_matches_whole_day(mlflow_sandbox):
    frame = training_frame(100)
    frame["site_prior_impr"] = np.arange(100)
    frame["site_prior_clicks"] = np.arange(100) // 3
    write_training_day(mlflow_sandbox, EVAL_DAY, frame)
    model = HistoricalCtrModel()
    model.p0 = 0.2

    labels, batched = predict_day(
        lambda batch: model.predict(batch[MODEL_INPUT_COLUMNS]),
        EVAL_DAY,
        [*MODEL_INPUT_COLUMNS, "click"],
        batch_rows=7,
    )

    assert len(labels) == 100
    np.testing.assert_allclose(batched, model.predict(frame))
