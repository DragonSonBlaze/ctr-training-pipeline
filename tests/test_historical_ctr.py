import numpy as np
import pandas as pd
import pytest

from ctr_pipeline.contracts import MODEL_INPUT_COLUMNS
from ctr_pipeline.models.historical_ctr import HistoricalCtrModel


def _frame(**counts: int) -> pd.DataFrame:
    row = {column: 0 for column in MODEL_INPUT_COLUMNS}
    row.update(counts)
    return pd.DataFrame([row])


def test_predict_matches_hand_computed_value():
    model = HistoricalCtrModel(beta=20.0)
    model.p0 = 0.2
    frame = _frame(
        site_prior_impr=10,
        site_prior_clicks=5,
        device_prior_impr=30,
        device_prior_clicks=0,
    )
    # r = (5 + 20 * 0.2) / (10 + 20) = 0.3; q = r (no app history); p = (0 + 20 * 0.3) / (30 + 20)
    assert model.predict(frame)[0] == pytest.approx(0.12)


def test_no_history_predicts_p0():
    model = HistoricalCtrModel(beta=20.0)
    model.p0 = 0.173
    predictions = model.predict(_frame())
    assert np.allclose(predictions, 0.173)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        HistoricalCtrModel().predict(_frame())


def test_fit_uses_click_mean():
    model = HistoricalCtrModel()
    model.fit(pd.DataFrame({"click": [0, 0, 1, 1]}))
    assert model.fitted_values() == {"p0": 0.5}
