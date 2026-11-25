"""Stable hashing of categorical values into a fixed number of slots.

Each (field, value) pair maps to one slot, so a value never seen in training still
gets a slot, no vocabulary has to be stored, and memory stays fixed. Two values that
share a slot collide, which the model accepts.

Python's built-in hash() is not used: it changes between processes, so a model saved
by one process would read other slots in another. pandas hashing with a fixed key
gives the same slots in every process.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NULL_TOKEN = "__null__"
_KEY_LENGTH = 16


def field_key(field: str) -> str:
    """The 16-character hash key of a field. Different fields get different keys, so the
    same value in two fields lands in different slots."""
    return (field + "_" * _KEY_LENGTH)[:_KEY_LENGTH]


def _as_text(values) -> np.ndarray:
    """Values as strings, with nulls replaced by NULL_TOKEN, in an object array."""
    text = pd.Series(values).astype("string").fillna(NULL_TOKEN)
    return text.to_numpy(dtype=object)


def hash_field(values, field: str, num_buckets: int) -> np.ndarray:
    """Slot of each value of one field, as int64 in [0, num_buckets)."""
    hashed = pd.util.hash_array(_as_text(values), hash_key=field_key(field), categorize=True)
    return (hashed % num_buckets).astype(np.int64)


def hash_frame(frame: pd.DataFrame, fields: list[str], num_buckets: int) -> np.ndarray:
    """Slots for every row and field, shape (rows, len(fields)).

    One column is converted at a time, so only one column of strings is in memory.
    """
    slots = np.empty((len(frame), len(fields)), dtype=np.int64)
    for position, field in enumerate(fields):
        slots[:, position] = hash_field(frame[field], field, num_buckets)
    return slots
