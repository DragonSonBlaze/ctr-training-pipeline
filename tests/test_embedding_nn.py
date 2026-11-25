import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from ctr_pipeline.config import MLFLOW_EXPERIMENT
from ctr_pipeline.contracts import CATEGORICAL_FIELDS, NUMERIC_FEATURES
from ctr_pipeline.models.embedding_nn import HashedEmbeddingModel
from ctr_pipeline.models.logistic import HashedLogisticModel

ROWS = 20_000
BUCKETS = 2**12
EPOCHS = 10
BATCH_ROWS = 256


def xor_frame() -> pd.DataFrame:
    """Click rate 0.5 when site_id equals app_id, 0.05 otherwise.

    Each (site, app) pair is exactly a quarter of the rows, so each field alone carries no
    signal: only the pair does. A model that adds up per-field effects cannot separate them.
    """
    position = np.arange(ROWS)
    site = np.where(position % 2 == 0, "a", "b")
    app = np.where((position // 2) % 2 == 0, "a", "b")
    rng = np.random.default_rng(11)
    rate = np.where(site == app, 0.5, 0.05)
    frame = pd.DataFrame(
        {
            "click": (rng.random(ROWS) < rate).astype("int64"),
            "site_id": site,
            "app_id": app,
        }
    )
    for field in CATEGORICAL_FIELDS:
        if field not in ("site_id", "app_id"):
            frame[field] = "x"
    for name in NUMERIC_FEATURES:
        frame[name] = 0.17 if name.endswith("_ctr_smooth") else 100.0
    return frame


def small_network(seed: int = 42) -> HashedEmbeddingModel:
    return HashedEmbeddingModel(
        num_buckets=BUCKETS, epochs=EPOCHS, batch_size=BATCH_ROWS, seed=seed
    )


def small_baseline() -> HashedLogisticModel:
    return HashedLogisticModel(num_buckets=BUCKETS, epochs=EPOCHS, batch_size=BATCH_ROWS)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return xor_frame()


@pytest.fixture(scope="module")
def fitted_network(frame) -> HashedEmbeddingModel:
    model = small_network()
    model.fit(frame)
    return model


@pytest.fixture(scope="module")
def fitted_baseline(frame) -> HashedLogisticModel:
    model = small_baseline()
    model.fit(frame)
    return model


def test_learns_interaction_the_baseline_cannot(frame, fitted_network, fitted_baseline):
    labels = frame["click"].to_numpy()
    network_auc = roc_auc_score(labels, fitted_network.predict(frame))
    baseline_auc = roc_auc_score(labels, fitted_baseline.predict(frame))
    assert network_auc >= 0.75
    assert baseline_auc <= 0.60


def test_calibrated_on_average(frame, fitted_network):
    labels = frame["click"].to_numpy()
    assert abs(fitted_network.predict(frame).mean() - labels.mean()) <= 0.02


def test_same_seed_same_predictions(frame):
    sample = frame.head(4_000)
    first = small_network(seed=3)
    first.fit(sample)
    second = small_network(seed=3)
    second.fit(sample)
    np.testing.assert_array_equal(first.predict(sample), second.predict(sample))


def test_predict_before_fit_raises(frame):
    with pytest.raises(RuntimeError):
        small_network().predict(frame.head(10))


def test_pyfunc_round_trip(mlflow_sandbox, fitted_network, frame):
    sample = frame.head(500)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            name="model",
            python_model=fitted_network.to_pyfunc(),
            pip_requirements=["numpy", "pandas", "torch"],
        )
    loaded = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")

    np.testing.assert_allclose(loaded.predict(sample), fitted_network.predict(sample))


def test_params_include_network_shape():
    params = HashedEmbeddingModel(num_buckets=BUCKETS).params()
    assert params["embedding_dim"] == 16
    assert params["hidden_units"] == "64,32"
