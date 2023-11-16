"""Module to test time_series functionality
"""
import numpy as np
import pytest
from pandas.testing import assert_frame_equal
from sktime.transformations.series.date import DateTimeFeatures
from sktime.transformations.series.summarize import WindowSummarizer
from time_series_test_utils import assert_frame_not_equal
from pycaret.time_series import TSForecastingExperiment
pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')

def test_fe_target(load_pos_and_neg_data):
    if False:
        print('Hello World!')
    'Test custom feature engineering for target (applicable to reduced\n    regression models only)\n    '
    data = load_pos_and_neg_data
    kwargs = {'lag_feature': {'lag': [36, 24, 13, 12, 11, 9, 6, 3, 2, 1]}}
    fe_target_rr = [WindowSummarizer(n_jobs=1, truncate='bfill', **kwargs)]
    exp = TSForecastingExperiment()
    exp.setup(data=data, fh=12, fold=3, session_id=42)
    exp.create_model('lr_cds_dt')
    metrics1 = exp.pull()
    exp = TSForecastingExperiment()
    exp.setup(data=data, fh=12, fold=3, fe_target_rr=fe_target_rr, session_id=42)
    exp.create_model('lr_cds_dt')
    metrics2 = exp.pull()
    assert_frame_not_equal(metrics1, metrics2)

@pytest.mark.parametrize('model, expected_equal', [('arima', False), ('lr_cds_dt', False), ('naive', True)])
def test_fe_exogenous(load_uni_exo_data_target, model, expected_equal):
    if False:
        for i in range(10):
            print('nop')
    'Test custom feature engineering for exogenous variables (applicable to all models).'
    (data, target) = load_uni_exo_data_target
    fh = 12

    def num_above_thresh(x):
        if False:
            while True:
                i = 10
        'Count how many observations lie above threshold.'
        return np.sum((x > 0.7)[::-1])
    kwargs1 = {'lag_feature': {'lag': [0, 1], 'mean': [[0, 4]]}}
    kwargs2 = {'lag_feature': {'lag': [0, 1], num_above_thresh: [[0, 2]], 'mean': [[0, 4]], 'std': [[0, 4]]}}
    fe_exogenous = [('a', WindowSummarizer(n_jobs=1, target_cols=['Income'], truncate='bfill', **kwargs1)), ('b', WindowSummarizer(n_jobs=1, target_cols=['Unemployment', 'Production'], truncate='bfill', **kwargs2))]
    exp = TSForecastingExperiment()
    exp.setup(data=data, target=target, fh=fh, session_id=42)
    features1 = exp.get_config('X_transformed').columns
    _ = exp.create_model(model)
    metrics1 = exp.pull()
    exp = TSForecastingExperiment()
    exp.setup(data=data, target=target, fh=fh, fe_exogenous=fe_exogenous, session_id=42)
    features2 = exp.get_config('X_transformed').columns
    _ = exp.create_model(model)
    metrics2 = exp.pull()
    assert len(features1) != len(features2)
    if expected_equal:
        assert_frame_equal(metrics1, metrics2)
    else:
        assert_frame_not_equal(metrics1, metrics2)

def test_fe_exog_data_no_exo(load_pos_and_neg_data):
    if False:
        i = 10
        return i + 15
    'Test custom feature engineering for target and exogenous when data does\n    not have any exogenous variables. e.g. extracting DateTimeFeatures from Index.\n    '
    data = load_pos_and_neg_data
    kwargs = {'lag_feature': {'lag': [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]}}
    fe_target_rr = [WindowSummarizer(n_jobs=1, truncate='bfill', **kwargs)]
    exp = TSForecastingExperiment()
    exp.setup(data=data, fh=12, session_id=42)
    _ = exp.create_model('lr_cds_dt')
    metrics1 = exp.pull()
    exp = TSForecastingExperiment()
    exp.setup(data=data, fh=12, fe_target_rr=fe_target_rr, session_id=42)
    _ = exp.create_model('lr_cds_dt')
    metrics2 = exp.pull()
    assert_frame_equal(metrics1, metrics2)
    fe_target2 = fe_target_rr + [DateTimeFeatures(ts_freq='M')]
    exp = TSForecastingExperiment()
    exp.setup(data=data, fh=12, fe_target_rr=fe_target2, session_id=42)
    _ = exp.create_model('lr_cds_dt')
    metrics3 = exp.pull()
    assert_frame_not_equal(metrics1, metrics3)
    fe_exogenous = [DateTimeFeatures(ts_freq='M')]
    exp = TSForecastingExperiment()
    exp.setup(data=data, fh=12, fe_target_rr=fe_target_rr, fe_exogenous=fe_exogenous, session_id=42)
    _ = exp.create_model('lr_cds_dt')
    metrics4 = exp.pull()
    assert_frame_not_equal(metrics1, metrics4)