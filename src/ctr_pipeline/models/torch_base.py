"""Shared base for the hashed PyTorch click models: hashing, scaling, training and prediction.

A subclass supplies the network (build_net), the output bias starting value (init_bias),
its hyperparameters (params) and its MLflow wrapper (to_pyfunc). Everything else is the
same for every subclass, so the models differ only in the network and the comparison
between them is fair by construction.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import torch

from ctr_pipeline.config import (
    LR_BATCH_SIZE,
    LR_EPOCHS,
    LR_LEARNING_RATE,
    NUM_HASH_BUCKETS,
    TORCH_SEED,
)
from ctr_pipeline.contracts import CATEGORICAL_FIELDS, NUMERIC_FEATURES
from ctr_pipeline.models.evaluate import EPS
from ctr_pipeline.models.hashing import hash_frame

logger = logging.getLogger(__name__)

PREDICT_CHUNK_ROWS = 100_000
# A numeric feature with no spread gets a standard deviation of 1, so dividing by it is safe.
STD_FLOOR = 1e-6


def logit(probability: float) -> float:
    clipped = min(max(probability, EPS), 1 - EPS)
    return math.log(clipped / (1 - clipped))


class HashedTorchModel:
    """Fit and predict on hashed categorical fields and numeric click-history features.

    Subclasses set model_type and implement build_net, init_bias, params and to_pyfunc.
    The network's forward(cat_idx, numeric) must return logits of shape (batch,).
    """

    model_type: str
    training_columns: list[str] = CATEGORICAL_FIELDS + NUMERIC_FEATURES + ["click"]
    input_columns: list[str] = CATEGORICAL_FIELDS + NUMERIC_FEATURES

    def __init__(
        self,
        num_buckets: int = NUM_HASH_BUCKETS,
        epochs: int = LR_EPOCHS,
        batch_size: int = LR_BATCH_SIZE,
        learning_rate: float = LR_LEARNING_RATE,
        seed: int = TORCH_SEED,
    ) -> None:
        self.num_buckets = num_buckets
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.net: torch.nn.Module | None = None
        self.numeric_mean: np.ndarray | None = None
        self.numeric_std: np.ndarray | None = None
        self.train_click_rate: float | None = None
        self.final_train_loss: float | None = None

    def build_net(self) -> torch.nn.Module:
        raise NotImplementedError

    def init_bias(self, net: torch.nn.Module, value: float) -> None:
        raise NotImplementedError

    def params(self) -> dict[str, float | int | str]:
        raise NotImplementedError

    def to_pyfunc(self):
        raise NotImplementedError

    def shared_params(self) -> dict[str, float | int | str]:
        """The hyperparameters every subclass logs."""
        return {
            "num_buckets": self.num_buckets,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "categorical_fields": ",".join(CATEGORICAL_FIELDS),
            "numeric_features": ",".join(NUMERIC_FEATURES),
        }

    def _raw_numeric(self, frame: pd.DataFrame) -> np.ndarray:
        """Click rates as log-odds and impression counts as log1p, one column per feature."""
        columns = []
        for name in NUMERIC_FEATURES:
            values = frame[name].to_numpy(dtype=np.float64)
            if name.endswith("_ctr_smooth"):
                clipped = np.clip(values, EPS, 1 - EPS)
                columns.append(np.log(clipped / (1 - clipped)))
            else:
                columns.append(np.log1p(values))
        return np.column_stack(columns)

    def numeric_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        if self.numeric_mean is None or self.numeric_std is None:
            raise RuntimeError("numeric_matrix() called before fit()")
        raw = self._raw_numeric(frame)
        return ((raw - self.numeric_mean) / self.numeric_std).astype(np.float32)

    def fit(self, train: pd.DataFrame) -> None:
        if train.empty:
            raise ValueError("cannot fit on an empty training sample")
        # Seeding here, not at import, so every fit with the same seed starts the same way.
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        cat = hash_frame(train, CATEGORICAL_FIELDS, self.num_buckets)
        raw = self._raw_numeric(train)
        self.numeric_mean = raw.mean(axis=0)
        std = raw.std(axis=0)
        std[std < STD_FLOOR] = 1.0
        self.numeric_std = std
        num = ((raw - self.numeric_mean) / self.numeric_std).astype(np.float32)
        labels = train["click"].to_numpy(dtype=np.float32)

        self.train_click_rate = float(labels.mean())
        net = self.build_net()
        # Starting the output bias at the training click rate means the model begins
        # calibrated, so training only has to learn the differences between rows.
        self.init_bias(net, logit(self.train_click_rate))
        self.final_train_loss = self._train(net, cat, num, labels)
        net.eval()
        self.net = net

    def _train(
        self, net: torch.nn.Module, cat: np.ndarray, num: np.ndarray, labels: np.ndarray
    ) -> float:
        cat_t = torch.from_numpy(cat)
        num_t = torch.from_numpy(num)
        labels_t = torch.from_numpy(labels)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        rows = len(labels)
        epoch_loss = float("nan")
        for epoch in range(self.epochs):
            generator = torch.Generator().manual_seed(self.seed + epoch)
            order = torch.randperm(rows, generator=generator)
            total = 0.0
            for start in range(0, rows, self.batch_size):
                batch = order[start : start + self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(net(cat_t[batch], num_t[batch]), labels_t[batch])
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(batch)
            epoch_loss = total / rows
            logger.info(
                "epoch %d of %d: mean training loss %.6f", epoch + 1, self.epochs, epoch_loss
            )
        return epoch_loss

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("predict() called before fit()")
        cat = hash_frame(frame, CATEGORICAL_FIELDS, self.num_buckets)
        num = self.numeric_matrix(frame)
        out = np.empty(len(frame), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(frame), PREDICT_CHUNK_ROWS):
                end = start + PREDICT_CHUNK_ROWS
                logits = self.net(
                    torch.from_numpy(cat[start:end]), torch.from_numpy(num[start:end])
                )
                out[start:end] = torch.sigmoid(logits).numpy()
        return out

    def fitted_values(self) -> dict[str, float]:
        if self.train_click_rate is None or self.final_train_loss is None:
            raise RuntimeError("fitted_values() called before fit()")
        return {
            "train_click_rate": self.train_click_rate,
            "final_train_loss": self.final_train_loss,
        }
