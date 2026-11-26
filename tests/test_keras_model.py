import keras
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ctr_pipeline.config import MLFLOW_EXPERIMENT
from ctr_pipeline.models.keras_model import KerasEmbeddingModel, build_network
from helpers import training_frame

BUCKETS = 2**10
DIM = 4
HIDDEN = (8,)
BATCH = 256
EPOCHS = 3


def small_model(**overrides) -> KerasEmbeddingModel:
    settings = {
        "num_buckets": BUCKETS,
        "embedding_dim": DIM,
        "hidden_units": HIDDEN,
        "batch_size": BATCH,
        "max_epochs": EPOCHS,
    }
    settings.update(overrides)
    return KerasEmbeddingModel(**settings)


def fitted_model(frame: pd.DataFrame) -> KerasEmbeddingModel:
    model = small_model()
    model.fit(frame)
    return model


def test_network_shapes():
    network = build_network(BUCKETS, DIM, HIDDEN, initial_bias=0.0)

    assert network.output_shape == (None, 1)
    embedding = next(layer for layer in network.layers if isinstance(layer, keras.layers.Embedding))
    assert embedding.embeddings.shape == (BUCKETS, DIM)


def test_single_shared_embedding_table():
    network = build_network(BUCKETS, DIM, HIDDEN, initial_bias=0.0)

    embeddings = [layer for layer in network.layers if isinstance(layer, keras.layers.Embedding)]
    assert len(embeddings) == 1


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


def test_initial_bias_is_logit_of_click_rate():
    bias = float(np.log(0.2 / 0.8))
    network = build_network(BUCKETS, DIM, HIDDEN, initial_bias=bias)

    output_layer = network.layers[-1]
    np.testing.assert_allclose(output_layer.bias.numpy(), [bias], rtol=1e-6)


def test_early_stopping_restores_best_epoch():
    rng = np.random.default_rng(5)
    train = training_frame(2000)
    validation = training_frame(1000)
    validation["click"] = rng.integers(0, 2, size=1000).astype("int64")
    model = small_model()

    model.fit(train, validation)

    assert model.epochs_run <= EPOCHS
    assert len(model.history["val_loss"]) == model.epochs_run
    assert model.best_val_loss == min(model.history["val_loss"])


def test_no_validation_trains_max_epochs():
    model = small_model()

    model.fit(training_frame(1000))

    assert model.epochs_run == EPOCHS
    assert model.history["val_loss"] == []


def test_same_seed_same_predictions():
    frame = training_frame(2000)

    first = small_model(seed=7)
    first.fit(frame)
    second = small_model(seed=7)
    second.fit(frame)

    np.testing.assert_allclose(first.predict(frame), second.predict(frame), atol=1e-5)


def test_pyfunc_round_trip(mlflow_sandbox):
    frame = training_frame(1000)
    model = fitted_model(frame)
    sample = frame.head(200)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            name="model",
            python_model=model.to_pyfunc(),
            pip_requirements=["numpy", "pandas", "tensorflow", "scipy"],
        )
    loaded = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")

    np.testing.assert_allclose(loaded.predict(sample), model.predict(sample))
