import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from ctr_pipeline.config import MLFLOW_EXPERIMENT
from ctr_pipeline.contracts import NUMERIC_FEATURES
from ctr_pipeline.models.sklearn_baseline import SklearnLogisticModel
from ctr_pipeline.models.train import MODEL_TYPES
from helpers import training_frame

BUCKETS = 2**12
NUM_CATEGORICAL = 20


def fitted_model(frame: pd.DataFrame) -> SklearnLogisticModel:
    model = SklearnLogisticModel(num_buckets=BUCKETS)
    model.fit(frame)
    return model


def test_design_matrix_shape_and_row_nnz():
    frame = training_frame(500)
    model = SklearnLogisticModel(num_buckets=BUCKETS)
    model.fit(frame)

    matrix = model.design_matrix(frame)

    assert matrix.shape == (500, BUCKETS + len(NUMERIC_FEATURES))
    nnz_per_row = np.diff(matrix.indptr)
    assert nnz_per_row.max() <= NUM_CATEGORICAL + len(NUMERIC_FEATURES)
    categorical_ones = matrix[:, :BUCKETS].sum(axis=1).A1
    assert (categorical_ones >= 1).all()


def test_learns_a_planted_signal():
    rng = np.random.default_rng(11)
    frame = training_frame(4000)
    is_target = rng.random(4000) < 0.5
    frame["site_id"] = np.where(is_target, "s0", "s1")
    flipped = rng.random(4000) < 0.05
    frame["click"] = (is_target ^ flipped).astype("int64")

    model = fitted_model(frame)

    assert roc_auc_score(frame["click"], model.predict(frame)) > 0.9


def test_predictions_are_probabilities():
    frame = training_frame(1000)
    predictions = fitted_model(frame).predict(frame)

    assert predictions.dtype == np.float64
    assert len(predictions) == len(frame)
    assert ((predictions >= 0) & (predictions <= 1)).all()


def test_mean_prediction_close_to_click_rate():
    frame = training_frame(4000)
    predictions = fitted_model(frame).predict(frame)

    assert abs(predictions.mean() - frame["click"].mean()) < 0.02


def test_empty_or_single_class_frame_raises():
    frame = training_frame(100)
    with pytest.raises(ValueError):
        SklearnLogisticModel(num_buckets=BUCKETS).fit(frame.head(0))
    single_class = frame.assign(click=0)
    with pytest.raises(ValueError):
        SklearnLogisticModel(num_buckets=BUCKETS).fit(single_class)


def test_pyfunc_round_trip(mlflow_sandbox):
    frame = training_frame(1000)
    model = fitted_model(frame)
    sample = frame.head(200)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            name="model",
            python_model=model.to_pyfunc(),
            pip_requirements=["numpy", "pandas", "scikit-learn", "scipy"],
        )
    loaded = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")

    np.testing.assert_allclose(loaded.predict(sample), model.predict(sample))


def test_train_for_date_accepts_the_new_type():
    assert "sklearn_logistic" in MODEL_TYPES
    assert isinstance(MODEL_TYPES["sklearn_logistic"](), SklearnLogisticModel)
