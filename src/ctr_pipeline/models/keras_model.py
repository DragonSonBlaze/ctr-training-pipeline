"""TensorFlow Keras click model: hashed categorical slots, one shared embedding table and
numeric features in a small dense network.

The hashing runs in numpy before the data reaches TensorFlow, so the network takes integer
slots. One embedding table serves every categorical field: its memory is num_buckets times
the embedding size, where one table per field would be that many times over. The output
bias starts at the logit of the training click rate, so the first predictions are already
calibrated and training only has to learn the differences between rows.
"""

from __future__ import annotations

import logging

import keras
import mlflow.pyfunc
import numpy as np
import pandas as pd

from ctr_pipeline.config import (
    KERAS_BATCH_SIZE,
    KERAS_EMBEDDING_DIM,
    KERAS_HIDDEN_UNITS,
    KERAS_LEARNING_RATE,
    KERAS_MAX_EPOCHS,
    KERAS_NUM_BUCKETS,
    KERAS_PATIENCE,
    KERAS_SEED,
)
from ctr_pipeline.contracts import CATEGORICAL_FIELDS, NUMERIC_FEATURES
from ctr_pipeline.models.hashing import hash_frame
from ctr_pipeline.models.numeric import EPS, Standardizer, raw_numeric

logger = logging.getLogger(__name__)

SLOTS_INPUT = "slots"
NUMERIC_INPUT = "numeric"
EMBEDDING_LAYER = "shared_embedding"


def build_network(
    num_buckets: int,
    embedding_dim: int,
    hidden_units: tuple[int, ...],
    initial_bias: float,
) -> keras.Model:
    """Hashed slots and numeric features in, one click probability out."""
    slots = keras.Input(shape=(len(CATEGORICAL_FIELDS),), dtype="int32", name=SLOTS_INPUT)
    numeric = keras.Input(shape=(len(NUMERIC_FEATURES),), dtype="float32", name=NUMERIC_INPUT)

    embedded = keras.layers.Embedding(num_buckets, embedding_dim, name=EMBEDDING_LAYER)(slots)
    x = keras.layers.Concatenate()([keras.layers.Flatten()(embedded), numeric])
    for units in hidden_units:
        x = keras.layers.Dense(units, activation="relu")(x)
    output = keras.layers.Dense(
        1,
        activation="sigmoid",
        bias_initializer=keras.initializers.Constant(initial_bias),
        name="click",
    )(x)
    return keras.Model(inputs={SLOTS_INPUT: slots, NUMERIC_INPUT: numeric}, outputs=output)


def model_inputs(
    frame: pd.DataFrame, standardizer: Standardizer, num_buckets: int
) -> dict[str, np.ndarray]:
    """The two network inputs for every row of frame."""
    slots = hash_frame(frame, CATEGORICAL_FIELDS, num_buckets).astype(np.int32)
    numeric = standardizer.transform(raw_numeric(frame))
    return {SLOTS_INPUT: slots, NUMERIC_INPUT: numeric}


def predict_probabilities(
    network: keras.Model, inputs: dict[str, np.ndarray], batch_size: int
) -> np.ndarray:
    raw = network.predict(inputs, batch_size=batch_size, verbose=0)
    return np.asarray(raw, dtype=np.float64).reshape(-1)


def _logit(probability: float) -> float:
    clipped = min(max(probability, EPS), 1 - EPS)
    return float(np.log(clipped / (1 - clipped)))


def _check_training_frame(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("cannot fit on an empty training sample")
    if np.unique(frame["click"].to_numpy()).size < 2:
        raise ValueError("training sample has a single click class, cannot fit")


class KerasEmbeddingModel:
    model_type = "keras_embedding"
    training_columns: list[str] = CATEGORICAL_FIELDS + NUMERIC_FEATURES + ["click"]
    input_columns: list[str] = CATEGORICAL_FIELDS + NUMERIC_FEATURES

    def __init__(
        self,
        num_buckets: int = KERAS_NUM_BUCKETS,
        embedding_dim: int = KERAS_EMBEDDING_DIM,
        hidden_units: tuple[int, ...] = KERAS_HIDDEN_UNITS,
        learning_rate: float = KERAS_LEARNING_RATE,
        batch_size: int = KERAS_BATCH_SIZE,
        max_epochs: int = KERAS_MAX_EPOCHS,
        patience: int = KERAS_PATIENCE,
        seed: int = KERAS_SEED,
    ) -> None:
        self.num_buckets = num_buckets
        self.embedding_dim = embedding_dim
        self.hidden_units = tuple(hidden_units)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.standardizer: Standardizer | None = None
        self.network: keras.Model | None = None
        self.history: dict[str, list[float]] | None = None
        self.epochs_run: int | None = None
        self.best_val_loss: float | None = None
        self.train_click_rate: float | None = None

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame | None = None) -> None:
        _check_training_frame(train)
        if validation is not None and validation.empty:
            raise ValueError("validation frame is empty")
        # Seeding first makes the weight initialisation and the batch order repeatable.
        keras.utils.set_random_seed(self.seed)

        self.standardizer = Standardizer().fit(raw_numeric(train))
        labels = train["click"].to_numpy(dtype=np.float32)
        self.train_click_rate = float(labels.mean())
        self.network = build_network(
            self.num_buckets,
            self.embedding_dim,
            self.hidden_units,
            initial_bias=_logit(self.train_click_rate),
        )
        self.network.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=keras.losses.BinaryCrossentropy(),
        )

        callbacks: list[keras.callbacks.Callback] = []
        fit_kwargs: dict[str, object] = {}
        if validation is not None:
            # Validation rows use the training scaler, so the data the model sees is
            # scaled the same way as the data it was fitted on.
            fit_kwargs["validation_data"] = (
                self._inputs(validation),
                validation["click"].to_numpy(dtype=np.float32),
            )
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.patience,
                    restore_best_weights=True,
                )
            )

        fitted = self.network.fit(
            self._inputs(train),
            labels,
            batch_size=self.batch_size,
            epochs=self.max_epochs,
            callbacks=callbacks,
            verbose=0,
            **fit_kwargs,
        )
        self.history = {
            "loss": [float(v) for v in fitted.history["loss"]],
            "val_loss": [float(v) for v in fitted.history.get("val_loss", [])],
        }
        self.epochs_run = len(self.history["loss"])
        self.best_val_loss = min(self.history["val_loss"]) if self.history["val_loss"] else None
        logger.info(
            "fitted keras embedding model on %d rows for %d of %d epochs",
            len(labels),
            self.epochs_run,
            self.max_epochs,
        )

    def _inputs(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        if self.standardizer is None:
            raise RuntimeError("model inputs requested before fit()")
        return model_inputs(frame, self.standardizer, self.num_buckets)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.network is None:
            raise RuntimeError("predict() called before fit()")
        return predict_probabilities(self.network, self._inputs(frame), self.batch_size)

    def params(self) -> dict[str, float | int | str]:
        return {
            "model_type": self.model_type,
            "num_buckets": self.num_buckets,
            "embedding_dim": self.embedding_dim,
            "hidden_units": ",".join(str(units) for units in self.hidden_units),
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "seed": self.seed,
            "categorical_fields": ",".join(CATEGORICAL_FIELDS),
            "numeric_features": ",".join(NUMERIC_FEATURES),
        }

    def fitted_values(self) -> dict[str, float]:
        if self.train_click_rate is None or self.epochs_run is None:
            raise RuntimeError("fitted_values() called before fit()")
        values = {
            "train_click_rate": self.train_click_rate,
            "epochs_run": float(self.epochs_run),
        }
        if self.best_val_loss is not None:
            values["best_val_loss"] = self.best_val_loss
        return values

    def to_pyfunc(self) -> KerasEmbeddingPyfunc:
        if self.network is None or self.standardizer is None:
            raise RuntimeError("to_pyfunc() called before fit()")
        return KerasEmbeddingPyfunc(
            num_buckets=self.num_buckets,
            embedding_dim=self.embedding_dim,
            hidden_units=self.hidden_units,
            batch_size=self.batch_size,
            standardizer=self.standardizer,
            weights=self.network.get_weights(),
        )


class KerasEmbeddingPyfunc(mlflow.pyfunc.PythonModel):
    """MLflow wrapper: keeps the weights and scaler, and rebuilds the network on first use.

    The Keras object itself is never pickled. Only the weight arrays are saved, so the
    loaded model predicts with the same network code the fitted model used.
    """

    def __init__(
        self,
        num_buckets: int,
        embedding_dim: int,
        hidden_units: tuple[int, ...],
        batch_size: int,
        standardizer: Standardizer,
        weights: list[np.ndarray],
    ) -> None:
        self.num_buckets = num_buckets
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.standardizer = standardizer
        self.weights = weights
        self.network: keras.Model | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["network"] = None
        return state

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        if self.network is None:
            # The initial bias is overwritten by the saved weights, so any value works here.
            network = build_network(
                self.num_buckets, self.embedding_dim, self.hidden_units, initial_bias=0.0
            )
            network.set_weights(self.weights)
            self.network = network
        inputs = model_inputs(model_input, self.standardizer, self.num_buckets)
        return predict_probabilities(self.network, inputs, self.batch_size)
