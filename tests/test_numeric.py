import numpy as np
import pytest

from ctr_pipeline.contracts import NUMERIC_FEATURES
from ctr_pipeline.models.numeric import Standardizer, raw_numeric
from helpers import training_frame


def test_raw_numeric_hand_computed():
    rates = np.array([0.2, 0.5])
    impressions = np.array([3, 10])
    frame = training_frame(2)
    for name in NUMERIC_FEATURES:
        frame[name] = rates if name.endswith("_ctr_smooth") else impressions

    raw = raw_numeric(frame)

    for column, name in enumerate(NUMERIC_FEATURES):
        if name.endswith("_ctr_smooth"):
            expected = np.log(rates / (1 - rates))
        else:
            expected = np.log1p(impressions)
        np.testing.assert_allclose(raw[:, column], expected, rtol=0, atol=1e-12)


def test_standardizer_zero_spread_column_gets_std_one():
    raw = np.column_stack([np.arange(5.0), np.full(5, 3.0)])

    standardizer = Standardizer().fit(raw)

    assert standardizer.std[1] == 1.0
    np.testing.assert_allclose(standardizer.transform(raw)[:, 1], 0.0)


def test_standardizer_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        Standardizer().transform(np.zeros((3, 6)))
