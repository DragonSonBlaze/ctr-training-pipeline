import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

from ctr_pipeline.config import MLFLOW_EXPERIMENT
from ctr_pipeline.contracts import CATEGORICAL_FIELDS, NUMERIC_FEATURES
from ctr_pipeline.models.logistic import HashedLogisticModel

ROWS = 20_000
BUCKETS = 2**12


def synthetic_frame() -> pd.DataFrame:
    """Half the rows have app "good" (click rate 0.5), half "bad" (click rate 0.05).

    Every other field is constant, so the app is the only signal the model can learn.
    """
    rng = np.random.default_rng(7)
    good = np.arange(ROWS) % 2 == 0
    rate = np.where(good, 0.5, 0.05)
    frame = pd.DataFrame(
        {
            "click": (rng.random(ROWS) < rate).astype("int64"),
            "app_id": np.where(good, "good", "bad"),
        }
    )
    for field in CATEGORICAL_FIELDS:
        if field != "app_id":
            frame[field] = "x"
    for name in NUMERIC_FEATURES:
        frame[name] = 0.17 if name.endswith("_ctr_smooth") else 100.0
    return frame


def small_model(seed: int = 42) -> HashedLogisticModel:
    return HashedLogisticModel(num_buckets=BUCKETS, epochs=3, batch_size=256, seed=seed)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return synthetic_frame()


@pytest.fixture(scope="module")
def fitted(frame) -> HashedLogisticModel:
    model = small_model()
    model.fit(frame)
    return model


def test_learns_field_effect(fitted, frame):
    predictions = fitted.predict(frame)
    good = frame["app_id"].to_numpy() == "good"
    assert predictions[good].mean() - predictions[~good].mean() >= 0.3


def test_beats_constant_predictor(fitted, frame):
    labels = frame["click"].to_numpy()
    constant = np.full(len(labels), labels.mean())
    assert log_loss(labels, fitted.predict(frame)) < log_loss(labels, constant)


def test_calibrated_on_average(fitted, frame):
    labels = frame["click"].to_numpy()
    assert abs(fitted.predict(frame).mean() - labels.mean()) <= 0.02


def test_same_seed_same_predictions(frame):
    first = small_model(seed=3)
    first.fit(frame)
    second = small_model(seed=3)
    second.fit(frame)
    np.testing.assert_array_equal(first.predict(frame), second.predict(frame))


def test_predict_before_fit_raises(frame):
    with pytest.raises(RuntimeError):
        small_model().predict(frame.head(10))


def test_pyfunc_round_trip(mlflow_sandbox, fitted, frame):
    sample = frame.head(500)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            name="model",
            python_model=fitted.to_pyfunc(),
            pip_requirements=["numpy", "pandas", "torch"],
        )
    loaded = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")

    np.testing.assert_allclose(loaded.predict(sample), fitted.predict(sample))
