import numpy as np
import pytest

from ctr_pipeline.models.logistic import HashedLogisticModel
from ctr_pipeline.models.numeric import Standardizer, raw_numeric
from helpers import training_frame


def test_raw_numeric_matches_torch_base():
    frame = training_frame(200)
    frame["app_ctr_smooth"] = np.linspace(0.0, 1.0, 200)
    frame["site_prior_impr"] = np.arange(200)
    torch_model = HashedLogisticModel(num_buckets=2**12)

    np.testing.assert_allclose(raw_numeric(frame), torch_model._raw_numeric(frame))


def test_standardizer_zero_spread_column_gets_std_one():
    raw = np.column_stack([np.arange(5.0), np.full(5, 3.0)])

    standardizer = Standardizer().fit(raw)

    assert standardizer.std[1] == 1.0
    np.testing.assert_allclose(standardizer.transform(raw)[:, 1], 0.0)


def test_standardizer_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        Standardizer().transform(np.zeros((3, 6)))
