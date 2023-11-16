"""Module to test time_series forecasting pipeline utils
"""
import numpy as np
import pytest
from sktime.forecasting.naive import NaiveForecaster
from pycaret.time_series import TSForecastingExperiment
from pycaret.utils.time_series.forecasting.models import DummyForecaster
from pycaret.utils.time_series.forecasting.pipeline import _add_model_to_pipeline, _are_pipeline_tansformations_empty, _get_imputed_data, _transformations_present_X, _transformations_present_y
pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')

def test_get_imputed_data_noexo(load_pos_data_missing):
    if False:
        i = 10
        return i + 15
    'Tests _get_imputed_data WITHOUT exogenous variables'
    y = load_pos_data_missing
    exp = TSForecastingExperiment()
    FH = 12
    exp.setup(data=y, fh=FH, numeric_imputation_target='drift')
    (y_imputed, X_imputed) = _get_imputed_data(pipeline=exp.pipeline, y=y)
    assert not np.array_equal(y_imputed, y)
    assert X_imputed is None
    y_imputed_expected = y_imputed.copy()
    exp.setup(data=y, fh=FH, numeric_imputation_target='drift', transform_target='exp')
    (y_imputed, X_imputed) = _get_imputed_data(pipeline=exp.pipeline, y=y)
    assert not np.array_equal(y_imputed, y)
    assert np.array_equal(y_imputed, y_imputed_expected)
    assert X_imputed is None
    y_no_miss = y.copy()
    y_no_miss.fillna(10, inplace=True)
    exp.setup(data=y_no_miss, fh=FH)
    (y_imputed, X_imputed) = _get_imputed_data(pipeline=exp.pipeline, y=y_no_miss)
    assert np.array_equal(y_imputed, y_no_miss)
    assert X_imputed is None
    exp.setup(data=y_no_miss, fh=FH, numeric_imputation_target='drift')
    (y_imputed, X_imputed) = _get_imputed_data(pipeline=exp.pipeline, y=y_no_miss)
    assert np.array_equal(y_imputed, y_no_miss)
    assert X_imputed is None
    exp.setup(data=y_no_miss, fh=FH, numeric_imputation_target='drift', transform_target='exp')
    (y_imputed, X_imputed) = _get_imputed_data(pipeline=exp.pipeline, y=y_no_miss)
    assert np.array_equal(y_imputed, y_no_miss)
    assert X_imputed is None

def test_get_imputed_data_exo(load_uni_exo_data_target_missing):
    if False:
        while True:
            i = 10
    'Tests _get_imputed_data WITH exogenous variables'
    (data, target) = load_uni_exo_data_target_missing
    y = data[target]
    X = data.drop(columns=target)
    exp = TSForecastingExperiment()
    FH = 12
    exp.setup(data=data, target=target, fh=FH, numeric_imputation_target='drift', numeric_imputation_exogenous='drift')
    (y_imputed, X_imputed) = _get_imputed_data(pipeline=exp.pipeline, y=y, X=X)
    assert not np.array_equal(y_imputed, y)
    assert not X_imputed.equals(X)
    y_imputed_expected = y_imputed.copy()
    exp.setup(data=data, target=target, fh=FH, numeric_imputation_target='drift', numeric_imputation_exogenous='drift', transform_target='exp', transform_exogenous='exp')
    (y_imputed, X_imputed) = _get_imputed_data(pipeline=exp.pipeline, y=y, X=X)
    assert not np.array_equal(y_imputed, y)
    assert np.array_equal(y_imputed, y_imputed_expected)
    assert not X_imputed.equals(X)
    data_no_miss = data.copy()
    data_no_miss.fillna(10, inplace=True)
    y_no_miss = data_no_miss[target]
    X_no_miss = data_no_miss.drop(columns=target)
    exp.setup(data=data_no_miss, target=target, fh=FH)
    (y_imputed, X_imputed) = _get_imputed_data(pipeline=exp.pipeline, y=y_no_miss, X=X_no_miss)
    assert np.array_equal(y_imputed, y_no_miss)
    assert X_imputed.equals(X_no_miss)
    exp.setup(data=data_no_miss, target=target, fh=FH, numeric_imputation_target='drift', numeric_imputation_exogenous='drift')
    (y_imputed, X_imputed) = _get_imputed_data(pipeline=exp.pipeline, y=y_no_miss, X=X_no_miss)
    assert np.array_equal(y_imputed, y_no_miss)
    assert X_imputed.equals(X_no_miss)
    exp.setup(data=data_no_miss, target=target, fh=FH, numeric_imputation_target='drift', numeric_imputation_exogenous='drift', transform_target='exp', transform_exogenous='exp')
    (y_imputed, X_imputed) = _get_imputed_data(pipeline=exp.pipeline, y=y_no_miss, X=X_no_miss)
    assert np.array_equal(y_imputed, y_no_miss)
    assert X_imputed.equals(X_no_miss)

def test_are_pipeline_tansformations_empty_noexo(load_pos_data_missing):
    if False:
        print('Hello World!')
    'Tests _are_pipeline_tansformations_empty, _transformations_present_X, and\n    _transformations_present_y WITHOUT exogenous variables'
    y = load_pos_data_missing
    y_no_miss = y.copy()
    y_no_miss.fillna(10, inplace=True)
    exp = TSForecastingExperiment()
    FH = 12
    exp.setup(data=y, fh=FH, numeric_imputation_target='drift')
    assert not _transformations_present_X(pipeline=exp.pipeline)
    assert _transformations_present_y(pipeline=exp.pipeline)
    assert not _are_pipeline_tansformations_empty(pipeline=exp.pipeline)
    exp.setup(data=y_no_miss, fh=FH, numeric_imputation_target='drift')
    assert not _are_pipeline_tansformations_empty(pipeline=exp.pipeline)
    exp.setup(data=y_no_miss, fh=FH)
    assert not _transformations_present_X(pipeline=exp.pipeline)
    assert not _transformations_present_y(pipeline=exp.pipeline)
    assert _are_pipeline_tansformations_empty(pipeline=exp.pipeline)

def test_are_pipeline_tansformations_empty_exo(load_uni_exo_data_target_missing):
    if False:
        while True:
            i = 10
    'Tests _are_pipeline_tansformations_empty, _transformations_present_X, and\n    _transformations_present_y WITH exogenous variables'
    (data, target) = load_uni_exo_data_target_missing
    data_no_miss = data.copy()
    data_no_miss.fillna(10, inplace=True)
    exp = TSForecastingExperiment()
    FH = 12
    exp.setup(data=data, target=target, fh=FH, numeric_imputation_target='drift', numeric_imputation_exogenous='drift')
    assert _transformations_present_X(pipeline=exp.pipeline)
    assert _transformations_present_y(pipeline=exp.pipeline)
    assert not _are_pipeline_tansformations_empty(pipeline=exp.pipeline)
    exp.setup(data=data_no_miss, target=target, fh=FH, numeric_imputation_target='drift')
    assert not _transformations_present_X(pipeline=exp.pipeline)
    assert _transformations_present_y(pipeline=exp.pipeline)
    assert not _are_pipeline_tansformations_empty(pipeline=exp.pipeline)
    exp.setup(data=data_no_miss, target=target, fh=FH, numeric_imputation_exogenous='drift')
    assert _transformations_present_X(pipeline=exp.pipeline)
    assert not _transformations_present_y(pipeline=exp.pipeline)
    assert not _are_pipeline_tansformations_empty(pipeline=exp.pipeline)
    exp.setup(data=data_no_miss, target=target, fh=FH)
    assert not _transformations_present_X(pipeline=exp.pipeline)
    assert not _transformations_present_y(pipeline=exp.pipeline)
    assert _are_pipeline_tansformations_empty(pipeline=exp.pipeline)

def test_add_model_to_pipeline_noexo(load_pos_and_neg_data):
    if False:
        i = 10
        return i + 15
    'Tests _add_model_to_pipeline WITHOUT exogenous variables'
    y = load_pos_and_neg_data
    exp = TSForecastingExperiment()
    FH = 12
    model = NaiveForecaster()
    exp.setup(data=y, fh=FH)
    assert isinstance(exp.pipeline.steps[-1][1].steps[-1][1], DummyForecaster)
    pipeline = _add_model_to_pipeline(pipeline=exp.pipeline, model=model)
    assert isinstance(pipeline.steps[-1][1].steps[-1][1], NaiveForecaster)
    assert isinstance(pipeline.steps_[-1][1].steps_[-1][1], NaiveForecaster)
    assert isinstance(pipeline.steps[-1][1].steps_[-1][1], NaiveForecaster)
    assert isinstance(pipeline.steps_[-1][1].steps[-1][1], NaiveForecaster)
    assert len(exp.pipeline.steps) == len(pipeline.steps)
    assert len(exp.pipeline.steps_) == len(pipeline.steps_)
    for i in np.arange(len(exp.pipeline.steps_)):
        assert exp.pipeline.steps[i][1].__class__ is pipeline.steps[i][1].__class__
        assert exp.pipeline.steps_[i][1].__class__ is pipeline.steps_[i][1].__class__
    assert len(exp.pipeline.steps[-1][1].steps) == len(pipeline.steps[-1][1].steps)
    assert len(exp.pipeline.steps_[-1][1].steps_) == len(pipeline.steps_[-1][1].steps_)
    assert len(exp.pipeline.steps[-1][1].steps_) == len(pipeline.steps[-1][1].steps_)
    assert len(exp.pipeline.steps_[-1][1].steps) == len(pipeline.steps_[-1][1].steps)
    for i in np.arange(len(exp.pipeline.steps_[-1][1]) - 1):
        assert exp.pipeline.steps[-1][1].steps[i][1].__class__ is pipeline.steps[-1][1].steps[i][1].__class__
        assert exp.pipeline.steps_[-1][1].steps_[i][1].__class__ is pipeline.steps_[-1][1].steps_[i][1].__class__
        assert exp.pipeline.steps[-1][1].steps_[i][1].__class__ is pipeline.steps[-1][1].steps_[i][1].__class__
        assert exp.pipeline.steps_[-1][1].steps[i][1].__class__ is pipeline.steps_[-1][1].steps[i][1].__class__
    exp.setup(data=y, fh=FH, numeric_imputation_target='drift')
    assert isinstance(exp.pipeline.steps[-1][1].steps[-1][1], DummyForecaster)
    pipeline = _add_model_to_pipeline(pipeline=exp.pipeline, model=model)
    assert isinstance(pipeline.steps[-1][1].steps[-1][1], NaiveForecaster)
    assert isinstance(pipeline.steps_[-1][1].steps_[-1][1], NaiveForecaster)
    assert isinstance(pipeline.steps[-1][1].steps_[-1][1], NaiveForecaster)
    assert isinstance(pipeline.steps_[-1][1].steps[-1][1], NaiveForecaster)
    assert len(exp.pipeline.steps) == len(pipeline.steps)
    assert len(exp.pipeline.steps_) == len(pipeline.steps_)
    for i in np.arange(len(exp.pipeline.steps_)):
        assert exp.pipeline.steps[i][1].__class__ is pipeline.steps[i][1].__class__
        assert exp.pipeline.steps_[i][1].__class__ is pipeline.steps_[i][1].__class__
    assert len(exp.pipeline.steps[-1][1].steps) == len(pipeline.steps[-1][1].steps)
    assert len(exp.pipeline.steps_[-1][1].steps_) == len(pipeline.steps_[-1][1].steps_)
    assert len(exp.pipeline.steps[-1][1].steps_) == len(pipeline.steps[-1][1].steps_)
    assert len(exp.pipeline.steps_[-1][1].steps) == len(pipeline.steps_[-1][1].steps)
    for i in np.arange(len(exp.pipeline.steps_[-1][1]) - 1):
        assert exp.pipeline.steps[-1][1].steps[i][1].__class__ is pipeline.steps[-1][1].steps[i][1].__class__
        assert exp.pipeline.steps_[-1][1].steps_[i][1].__class__ is pipeline.steps_[-1][1].steps_[i][1].__class__
        assert exp.pipeline.steps[-1][1].steps_[i][1].__class__ is pipeline.steps[-1][1].steps_[i][1].__class__
        assert exp.pipeline.steps_[-1][1].steps[i][1].__class__ is pipeline.steps_[-1][1].steps[i][1].__class__

def test_add_model_to_pipeline_exo(load_uni_exo_data_target):
    if False:
        print('Hello World!')
    'Tests _add_model_to_pipeline WITH exogenous variables'
    (data, target) = load_uni_exo_data_target
    exp = TSForecastingExperiment()
    FH = 12
    model = NaiveForecaster()
    exp.setup(data=data, target=target, fh=FH)
    assert isinstance(exp.pipeline.steps[-1][1].steps[-1][1], DummyForecaster)
    pipeline = _add_model_to_pipeline(pipeline=exp.pipeline, model=model)
    assert isinstance(pipeline.steps[-1][1].steps[-1][1], NaiveForecaster)
    assert isinstance(pipeline.steps_[-1][1].steps_[-1][1], NaiveForecaster)
    assert isinstance(pipeline.steps[-1][1].steps_[-1][1], NaiveForecaster)
    assert isinstance(pipeline.steps_[-1][1].steps[-1][1], NaiveForecaster)
    assert len(exp.pipeline.steps) == len(pipeline.steps)
    assert len(exp.pipeline.steps_) == len(pipeline.steps_)
    for i in np.arange(len(exp.pipeline.steps_)):
        assert exp.pipeline.steps[i][1].__class__ is pipeline.steps[i][1].__class__
        assert exp.pipeline.steps_[i][1].__class__ is pipeline.steps_[i][1].__class__
    assert len(exp.pipeline.steps[-1][1].steps) == len(pipeline.steps[-1][1].steps)
    assert len(exp.pipeline.steps_[-1][1].steps_) == len(pipeline.steps_[-1][1].steps_)
    assert len(exp.pipeline.steps[-1][1].steps_) == len(pipeline.steps[-1][1].steps_)
    assert len(exp.pipeline.steps_[-1][1].steps) == len(pipeline.steps_[-1][1].steps)
    for i in np.arange(len(exp.pipeline.steps_[-1][1]) - 1):
        assert exp.pipeline.steps[-1][1].steps[i][1].__class__ is pipeline.steps[-1][1].steps[i][1].__class__
        assert exp.pipeline.steps_[-1][1].steps_[i][1].__class__ is pipeline.steps_[-1][1].steps_[i][1].__class__
        assert exp.pipeline.steps[-1][1].steps_[i][1].__class__ is pipeline.steps[-1][1].steps_[i][1].__class__
        assert exp.pipeline.steps_[-1][1].steps[i][1].__class__ is pipeline.steps_[-1][1].steps[i][1].__class__
    exp.setup(data=data, target=target, fh=FH, numeric_imputation_target='drift', numeric_imputation_exogenous='drift')
    assert isinstance(exp.pipeline.steps[-1][1].steps[-1][1], DummyForecaster)
    pipeline = _add_model_to_pipeline(pipeline=exp.pipeline, model=model)
    assert isinstance(pipeline.steps[-1][1].steps[-1][1], NaiveForecaster)
    assert isinstance(pipeline.steps_[-1][1].steps_[-1][1], NaiveForecaster)
    assert isinstance(pipeline.steps[-1][1].steps_[-1][1], NaiveForecaster)
    assert isinstance(pipeline.steps_[-1][1].steps[-1][1], NaiveForecaster)
    assert len(exp.pipeline.steps) == len(pipeline.steps)
    assert len(exp.pipeline.steps_) == len(pipeline.steps_)
    for i in np.arange(len(exp.pipeline.steps_)):
        assert exp.pipeline.steps[i][1].__class__ is pipeline.steps[i][1].__class__
        assert exp.pipeline.steps_[i][1].__class__ is pipeline.steps_[i][1].__class__
    assert len(exp.pipeline.steps[-1][1].steps) == len(pipeline.steps[-1][1].steps)
    assert len(exp.pipeline.steps_[-1][1].steps_) == len(pipeline.steps_[-1][1].steps_)
    assert len(exp.pipeline.steps[-1][1].steps_) == len(pipeline.steps[-1][1].steps_)
    assert len(exp.pipeline.steps_[-1][1].steps) == len(pipeline.steps_[-1][1].steps)
    for i in np.arange(len(exp.pipeline.steps_[-1][1]) - 1):
        assert exp.pipeline.steps[-1][1].steps[i][1].__class__ is pipeline.steps[-1][1].steps[i][1].__class__
        assert exp.pipeline.steps_[-1][1].steps_[i][1].__class__ is pipeline.steps_[-1][1].steps_[i][1].__class__
        assert exp.pipeline.steps[-1][1].steps_[i][1].__class__ is pipeline.steps[-1][1].steps_[i][1].__class__
        assert exp.pipeline.steps_[-1][1].steps[i][1].__class__ is pipeline.steps_[-1][1].steps[i][1].__class__