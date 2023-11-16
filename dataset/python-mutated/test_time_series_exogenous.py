"""Module to test time_series forecasting - univariate with exogenous variables
"""
import numpy as np
import pandas as pd
import pytest
from pycaret.time_series import TSForecastingExperiment
from pycaret.utils.time_series import TSApproachTypes, TSExogenousPresent
pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')

def test_create_tune_predict_finalize_model(load_uni_exo_data_target):
    if False:
        for i in range(10):
            print('nop')
    'test create_model, tune_model, predict_model and finalize_model\n    functionality using exogenous variables\n    '
    (data, target) = load_uni_exo_data_target
    fh = 12
    data_for_modeling = data.iloc[:-12]
    future_data = data.iloc[-12:]
    future_exog = future_data.drop(columns=target)
    exp = TSForecastingExperiment()
    exp.setup(data=data_for_modeling, target=target, fh=fh, session_id=42)
    model = exp.create_model('arima')
    expected_period_index = data_for_modeling.iloc[-fh:].index
    final_expected_period_index = future_exog.index
    y_pred = exp.predict_model(model)
    assert isinstance(y_pred, pd.DataFrame)
    assert np.all(y_pred.index == expected_period_index)
    tuned_model = exp.tune_model(model)
    y_pred = exp.predict_model(tuned_model)
    assert isinstance(y_pred, pd.DataFrame)
    assert np.all(y_pred.index == expected_period_index)
    final_model = exp.finalize_model(tuned_model)
    y_pred = exp.predict_model(final_model, X=future_exog)
    assert np.all(y_pred.index == final_expected_period_index)

def test_blend_models(load_uni_exo_data_target, load_models_uni_mix_exo_noexo):
    if False:
        return 10
    'test blending functionality.\n    NOTE: compare models does not enforce exog here for now.\n    TODO: Later when Reduced Regression Models also support exogenous variables,\n    we can add a test with only models that support exogenous variables (i.e.\n    with enforce_exogenous=True).\n    '
    (data, target) = load_uni_exo_data_target
    fh = 12
    data_for_modeling = data.iloc[:-12]
    future_data = data.iloc[-12:]
    future_exog = future_data.drop(columns=target)
    expected_period_index = data_for_modeling.iloc[-fh:].index
    final_expected_period_index = future_exog.index
    exp = TSForecastingExperiment()
    exp.setup(data=data_for_modeling, target=target, fh=fh, enforce_exogenous=False, session_id=42)
    models_to_include = load_models_uni_mix_exo_noexo
    best_models = exp.compare_models(include=models_to_include, n_select=3)
    blender = exp.blend_models(best_models)
    y_pred = exp.predict_model(blender)
    assert isinstance(y_pred, pd.DataFrame)
    assert np.all(y_pred.index == expected_period_index)
    final_model = exp.finalize_model(blender)
    y_pred = exp.predict_model(final_model, X=future_exog)
    assert np.all(y_pred.index == final_expected_period_index)

def test_setup():
    if False:
        return 10
    'Test the setup with exogenous variables'
    length = 100
    data = pd.DataFrame(np.random.rand(length, 7))
    data.columns = 'A B C D E F G'.split()
    data['B'] = pd.date_range('20130101', periods=length)
    target = 'A'
    index = 'B'
    exp = TSForecastingExperiment()
    approach_type = TSApproachTypes.UNI
    exogenous_present = TSExogenousPresent.NO
    exp.setup(data=data[target])
    assert exp.approach_type == approach_type
    assert exp.exogenous_present == exogenous_present
    assert exp.target_param == target
    assert exp.exogenous_variables == []
    exp.setup(data=pd.DataFrame(data[target]))
    assert exp.approach_type == approach_type
    assert exp.exogenous_present == exogenous_present
    assert exp.target_param == target
    assert exp.exogenous_variables == []
    exp.setup(data=data[target], target=target)
    assert exp.approach_type == approach_type
    assert exp.exogenous_present == exogenous_present
    assert exp.target_param == target
    assert exp.exogenous_variables == []
    approach_type = TSApproachTypes.UNI
    exogenous_present = TSExogenousPresent.YES
    exp.setup(data=data, target=target)
    assert exp.approach_type == approach_type
    assert exp.exogenous_present == exogenous_present
    assert exp.target_param == target
    assert exp.exogenous_variables == ['B', 'C', 'D', 'E', 'F', 'G']
    exp.setup(data=data, target=target, index=index)
    assert exp.approach_type == approach_type
    assert exp.exogenous_present == exogenous_present
    assert exp.target_param == target
    assert exp.exogenous_variables == ['C', 'D', 'E', 'F', 'G']
    exp.setup(data=data, target=target, index=index, ignore_features=['C', 'E'])
    assert exp.approach_type == approach_type
    assert exp.exogenous_present == exogenous_present
    assert exp.target_param == target
    assert exp.exogenous_variables == ['D', 'F', 'G']
    exp.setup(data=data, target=target, ignore_features=['C', 'E'])
    assert exp.approach_type == approach_type
    assert exp.exogenous_present == exogenous_present
    assert exp.target_param == target
    assert exp.exogenous_variables == ['B', 'D', 'F', 'G']

def test_setup_raises():
    if False:
        while True:
            i = 10
    'Test the setup with exogenous variables when it raises errors'
    length = 100
    data = pd.DataFrame(np.random.rand(length, 7))
    data.columns = 'A B C D E F G'.split()
    exp = TSForecastingExperiment()
    with pytest.raises(ValueError) as errmsg:
        exp.setup(data=data)
    exceptionmsg = errmsg.value.args[0]
    assert exceptionmsg == f'Data has {len(data.columns)} columns, but the target has not been specified.'
    target = 'WRONG'
    column = 'A'
    with pytest.raises(ValueError) as errmsg:
        exp.setup(data=data[column], target=target)
    exceptionmsg = errmsg.value.args[0]
    assert exceptionmsg == f"Target = '{target}', but data only has '{column}'. If you are passing a series (or a dataframe with 1 column) to setup, you can leave `target=None`"
    with pytest.raises(ValueError) as errmsg:
        exp.setup(data=data, target=target)
    exceptionmsg = errmsg.value.args[0]
    assert exceptionmsg == f"Target Column '{target}' is not present in the data."