"""Stage 3: logistic regression on hashed categorical features, written in PyTorch.

Each (field, value) pair owns one learned weight in a table of NUM_HASH_BUCKETS slots.
A row's logit is the sum of the weights of its slots, plus a linear term on the numeric
click-history features and a bias. That is logistic regression on one-hot hashed features.
The training loop, scaling and prediction live in torch_base.py and are shared with the
embedding network.

The class names and attribute names here are kept as they were, because the models already
pickled in MLflow (including the champion) are loaded by these names.
"""

from __future__ import annotations

import mlflow.pyfunc
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
from ctr_pipeline.contracts import NUMERIC_FEATURES
from ctr_pipeline.models.torch_base import HashedTorchModel


class HashedLogisticNet(torch.nn.Module):
    def __init__(self, num_buckets: int, num_numeric: int) -> None:
        super().__init__()
        self.weights = torch.nn.Embedding(num_buckets, 1)
        torch.nn.init.zeros_(self.weights.weight)
        self.numeric = torch.nn.Linear(num_numeric, 1)
        torch.nn.init.zeros_(self.numeric.weight)

    def forward(self, cat_idx: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        # cat_idx: (batch, fields) int64, numeric: (batch, num_numeric) float32 -> logits (batch,)
        return self.weights(cat_idx).sum(dim=1).squeeze(-1) + self.numeric(numeric).squeeze(-1)


class HashedLogisticModel(HashedTorchModel):
    model_type = "logistic_hashed"

    def __init__(
        self,
        num_buckets: int = NUM_HASH_BUCKETS,
        epochs: int = LR_EPOCHS,
        batch_size: int = LR_BATCH_SIZE,
        learning_rate: float = LR_LEARNING_RATE,
        seed: int = TORCH_SEED,
    ) -> None:
        super().__init__(num_buckets, epochs, batch_size, learning_rate, seed)

    def build_net(self) -> HashedLogisticNet:
        return HashedLogisticNet(self.num_buckets, len(NUMERIC_FEATURES))

    def init_bias(self, net: torch.nn.Module, value: float) -> None:
        torch.nn.init.constant_(net.numeric.bias, value)

    def params(self) -> dict[str, float | int | str]:
        return self.shared_params()

    def to_pyfunc(self) -> HashedLogisticPyfunc:
        return HashedLogisticPyfunc(self)


class HashedLogisticPyfunc(mlflow.pyfunc.PythonModel):
    """MLflow wrapper: the fitted model is pickled with its weights and predicts from them."""

    def __init__(self, model: HashedLogisticModel) -> None:
        self.model = model

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        return self.model.predict(model_input)
