"""Module to test time_series functionality
"""
import numpy as np
import pandas as pd
import pytest
from time_series_test_utils import _return_model_names
from pycaret.time_series import TSForecastingExperiment
pytestmark = [pytest.mark.filterwarnings('ignore::UserWarning'), pytest.mark.tuning_random]
_model_names = _return_model_names()

@pytest.mark.parametrize('model', _model_names)
def test_tune_model_random(model, load_pos_and_neg_data):
    if False:
        i = 10
        return i + 15
    exp = TSForecastingExperiment()
    fh = 12
    fold = 2
    data = load_pos_and_neg_data
    exp.setup(data=data, fold=fold, fh=fh, fold_strategy='sliding')
    model_obj = exp.create_model(model)
    tuned_model_obj = exp.tune_model(model_obj)
    y_pred = exp.predict_model(tuned_model_obj)
    assert isinstance(y_pred, pd.DataFrame)
    expected_period_index = data.iloc[-fh:].index
    assert np.all(y_pred.index == expected_period_index)