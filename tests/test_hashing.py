import subprocess
import sys

import numpy as np
import pandas as pd

from ctr_pipeline.models.hashing import field_key, hash_field, hash_frame

NUM_BUCKETS = 2**20
# Slot of "1005" in field C1, with the 2**20 buckets. Computed once with pandas hashing
# and fixed here so any change of hash function, key or library version fails this test.
KNOWN_C1_1005_SLOT = 645245


def test_hash_is_deterministic():
    values = np.array(["a", "b", "1005", None], dtype=object)
    first = hash_field(values, "site_id", NUM_BUCKETS)
    second = hash_field(values, "site_id", NUM_BUCKETS)
    np.testing.assert_array_equal(first, second)


def test_known_value():
    slot = hash_field(np.array(["1005"], dtype=object), "C1", NUM_BUCKETS)[0]
    assert slot == KNOWN_C1_1005_SLOT


def test_known_value_is_the_same_in_a_fresh_process():
    code = (
        "import numpy as np\n"
        "from ctr_pipeline.models.hashing import hash_field\n"
        "print(int(hash_field(np.array(['1005'], dtype=object), 'C1', 2**20)[0]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert int(result.stdout.strip()) == KNOWN_C1_1005_SLOT


def test_field_key_separates_fields():
    value = np.array(["abc"], dtype=object)
    site_slot = hash_field(value, "site_id", NUM_BUCKETS)[0]
    app_slot = hash_field(value, "app_id", NUM_BUCKETS)[0]
    assert site_slot != app_slot


def test_field_key_is_sixteen_characters():
    assert field_key("C1") == "C1" + "_" * 14
    assert len(field_key("device_model")) == 16
    assert field_key("a" * 40) == "a" * 16


def test_range_and_nulls():
    values = np.array(["x", "y", None, "1005", None], dtype=object)
    slots = hash_field(values, "app_domain", 2**12)
    assert slots.min() >= 0
    assert slots.max() < 2**12
    assert slots[2] == slots[4]
    assert slots[2] == hash_field(np.array(["__null__"], dtype=object), "app_domain", 2**12)[0]


def test_hash_frame_shape_and_columns_match_hash_field():
    frame = pd.DataFrame({"C1": [1005, 7], "site_id": ["a", None]})
    slots = hash_frame(frame, ["C1", "site_id"], NUM_BUCKETS)
    assert slots.shape == (2, 2)
    assert slots.dtype == np.int64
    np.testing.assert_array_equal(slots[:, 0], hash_field(frame["C1"], "C1", NUM_BUCKETS))
    np.testing.assert_array_equal(slots[:, 1], hash_field(frame["site_id"], "site_id", NUM_BUCKETS))
