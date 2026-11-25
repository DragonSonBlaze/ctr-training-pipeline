"""Stage 3 (v3): embedding-based neural network in PyTorch on hashed categorical features.

Each hash slot owns a vector of NN_EMBEDDING_DIM numbers instead of one weight. The vectors of
the categorical fields are concatenated with the numeric features and passed through hidden
layers, so the network can learn combinations of values (this app on this site) that the
logistic baseline can only add up. The same inputs, training sample and evaluation day are
used for both models, so any difference in their scores comes from the model.
"""

from __future__ import annotations

import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch

from ctr_pipeline.config import (
    NN_BATCH_SIZE,
    NN_EMBEDDING_DIM,
    NN_EPOCHS,
    NN_HIDDEN_UNITS,
    NN_LEARNING_RATE,
    NUM_HASH_BUCKETS,
    TORCH_SEED,
)
from ctr_pipeline.contracts import CATEGORICAL_FIELDS, NUMERIC_FEATURES
from ctr_pipeline.models.torch_base import HashedTorchModel


class HashedEmbeddingNet(torch.nn.Module):
    def __init__(
        self,
        num_buckets: int,
        num_fields: int,
        num_numeric: int,
        embedding_dim: int,
        hidden_units: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.embeddings = torch.nn.Embedding(num_buckets, embedding_dim)
        torch.nn.init.normal_(self.embeddings.weight, std=0.01)
        layers: list[torch.nn.Module] = []
        width = num_fields * embedding_dim + num_numeric
        for units in hidden_units:
            layers += [torch.nn.Linear(width, units), torch.nn.ReLU()]
            width = units
        self.hidden = torch.nn.Sequential(*layers)
        self.output = torch.nn.Linear(width, 1)
        # Zero output weights make the first predictions depend only on the bias.
        torch.nn.init.zeros_(self.output.weight)

    def forward(self, cat_idx: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        # (batch, fields) -> (batch, fields, dim) -> (batch, fields * dim), then the numeric
        # features are appended to every row.
        x = torch.cat([self.embeddings(cat_idx).flatten(start_dim=1), numeric], dim=1)
        return self.output(self.hidden(x)).squeeze(-1)


class HashedEmbeddingModel(HashedTorchModel):
    model_type = "embedding_nn"

    def __init__(
        self,
        num_buckets: int = NUM_HASH_BUCKETS,
        epochs: int = NN_EPOCHS,
        batch_size: int = NN_BATCH_SIZE,
        learning_rate: float = NN_LEARNING_RATE,
        seed: int = TORCH_SEED,
        embedding_dim: int = NN_EMBEDDING_DIM,
        hidden_units: tuple[int, ...] = NN_HIDDEN_UNITS,
    ) -> None:
        super().__init__(num_buckets, epochs, batch_size, learning_rate, seed)
        self.embedding_dim = embedding_dim
        self.hidden_units = tuple(hidden_units)

    def build_net(self) -> HashedEmbeddingNet:
        return HashedEmbeddingNet(
            self.num_buckets,
            len(CATEGORICAL_FIELDS),
            len(NUMERIC_FEATURES),
            self.embedding_dim,
            self.hidden_units,
        )

    def init_bias(self, net: torch.nn.Module, value: float) -> None:
        torch.nn.init.constant_(net.output.bias, value)

    def params(self) -> dict[str, float | int | str]:
        return {
            **self.shared_params(),
            "embedding_dim": self.embedding_dim,
            "hidden_units": ",".join(str(units) for units in self.hidden_units),
        }

    def to_pyfunc(self) -> HashedEmbeddingPyfunc:
        return HashedEmbeddingPyfunc(self)


class HashedEmbeddingPyfunc(mlflow.pyfunc.PythonModel):
    """MLflow wrapper: the fitted model is pickled with its weights and predicts from them."""

    def __init__(self, model: HashedEmbeddingModel) -> None:
        self.model = model

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        return self.model.predict(model_input)
