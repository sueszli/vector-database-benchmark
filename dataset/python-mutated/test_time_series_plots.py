"""Module to test time_series plotting functionality
"""
import os
import sys
import numpy as np
import pytest
from time_series_test_utils import _ALL_PLOTS_DATA, _ALL_PLOTS_ESTIMATOR, _ALL_PLOTS_ESTIMATOR_NOT_DATA, _return_all_plots_estimator_ts_results, _return_data_with_without_period_index, _return_model_names_for_plots_stats
from pycaret.datasets import get_data
from pycaret.time_series import TSForecastingExperiment
pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')
os.environ['PYCARET_TESTING'] = '1'
if sys.platform == 'win32':
    pytest.skip('Skipping test module on Windows', allow_module_level=True)
_data_with_without_period_index = _return_data_with_without_period_index()
_model_names_for_plots = _return_model_names_for_plots_stats()
_all_plots_estimator_ts_results = _return_all_plots_estimator_ts_results()

@pytest.mark.parametrize('data', _data_with_without_period_index)
@pytest.mark.parametrize('plot', _ALL_PLOTS_DATA)
def test_plot_model_data(data, plot):
    if False:
        i = 10
        return i + 15
    'Tests the plot_model functionality on original dataset\n    NOTE: Want to show multiplicative plot here so can not take data with negative values\n    '
    exp = TSForecastingExperiment()
    fh = np.arange(1, 13)
    fold = 2
    exp.setup(data=data, fh=fh, fold=fold, fold_strategy='sliding', verbose=False, session_id=42)
    exp.plot_model(plot=plot)
    from pycaret.time_series import plot_model, setup
    _ = setup(data=data, fh=fh, fold=fold, fold_strategy='expanding', session_id=42, n_jobs=-1)
    plot_model(plot=plot)

@pytest.mark.parametrize('model_name', _model_names_for_plots)
@pytest.mark.parametrize('data', _data_with_without_period_index)
@pytest.mark.parametrize('plot', _ALL_PLOTS_ESTIMATOR)
def test_plot_model_estimator(model_name, data, plot):
    if False:
        while True:
            i = 10
    'Tests the plot_model functionality on estimators\n    NOTE: Want to show multiplicative plot here so can not take data with negative values\n    '
    exp = TSForecastingExperiment()
    fh = np.arange(1, 13)
    fold = 2
    exp.setup(data=data, fh=fh, fold=fold, fold_strategy='sliding', verbose=False, session_id=42)
    model = exp.create_model(model_name)
    exp.plot_model(estimator=model, plot=plot)
    from pycaret.time_series import create_model, plot_model, setup
    _ = setup(data=data, fh=fh, fold=fold, fold_strategy='expanding', session_id=42, n_jobs=-1)
    model = create_model(model_name)
    plot_model(estimator=model, plot=plot)

@pytest.mark.parametrize('plot', _ALL_PLOTS_ESTIMATOR_NOT_DATA)
def test_plot_model_data_raises(load_pos_and_neg_data, plot):
    if False:
        print('Hello World!')
    'Tests the plot_model functionality when it raises an exception\n    on data plots (i.e. estimator is not passed)\n    '
    exp = TSForecastingExperiment()
    fh = np.arange(1, 13)
    fold = 2
    exp.setup(data=load_pos_and_neg_data, fh=fh, fold=fold, fold_strategy='sliding', verbose=False, session_id=42)
    with pytest.raises(ValueError) as errmsg:
        exp.plot_model(plot=plot)
    exceptionmsg = errmsg.value.args[0]
    assert f"Plot type '{plot}' is not supported when estimator is not provided" in exceptionmsg

@pytest.mark.parametrize('data', _data_with_without_period_index)
def test_plot_model_customization(data):
    if False:
        while True:
            i = 10
    'Tests the customization of plot_model\n    NOTE: Want to show multiplicative plot here so can not take data with negative values\n    '
    exp = TSForecastingExperiment()
    fh = np.arange(1, 13)
    fold = 2
    exp.setup(data=data, fh=fh, fold=fold, fold_strategy='sliding', verbose=False, session_id=42)
    model = exp.create_model('naive')
    print('\n\n==== Testing Customization ON DATA ====')
    exp.plot_model(plot='pacf', data_kwargs={'nlags': 36}, fig_kwargs={'fig_size': [800, 500], 'fig_template': 'simple_white'})
    exp.plot_model(plot='decomp_classical', data_kwargs={'type': 'multiplicative'})
    print('\n\n====  Testing Customization ON ESTIMATOR ====')
    exp.plot_model(estimator=model, plot='forecast', data_kwargs={'fh': 24})

@pytest.mark.parametrize('data', _data_with_without_period_index)
@pytest.mark.parametrize('plot', _ALL_PLOTS_DATA)
def test_plot_model_return_data_original_data(data, plot):
    if False:
        print('Hello World!')
    'Tests whether the return_data parameter of the plot_model function works\n    properly or not for the original data\n    '
    exp = TSForecastingExperiment()
    fh = np.arange(1, 13)
    fold = 2
    exp.setup(data=data, fh=fh, fold=fold, fold_strategy='sliding', verbose=False, session_id=42)
    plot_data = exp.plot_model(plot=plot, return_data=True)
    assert isinstance(plot_data, dict) or plot_data is None

@pytest.mark.parametrize('data', _data_with_without_period_index)
@pytest.mark.parametrize('model_name', _model_names_for_plots)
@pytest.mark.parametrize('plot', _ALL_PLOTS_ESTIMATOR)
def test_plot_model_return_data_estimator(data, model_name, plot):
    if False:
        return 10
    'Tests whether the return_data parameter of the plot_model function works\n    properly or not for the estimator\n    '
    exp = TSForecastingExperiment()
    fh = np.arange(1, 13)
    fold = 2
    exp.setup(data=data, fh=fh, fold=fold, fold_strategy='sliding', verbose=False, session_id=42)
    model = exp.create_model(model_name)
    plot_data = exp.plot_model(estimator=model, plot=plot, return_data=True)
    assert isinstance(plot_data, dict) or plot_data is None

@pytest.mark.parametrize('plot, all_models_supported', _all_plots_estimator_ts_results)
def test_plot_multiple_model_overlays(load_pos_and_neg_data, plot, all_models_supported):
    if False:
        print('Hello World!')
    'Tests the plot_model functionality on estimators where the results from\n    multiple models get overlaid (time series plots)\n\n    Checks:\n        (1) Plots are correct even when the multiple models are of the same type\n        (2) Plot labels are correct when user provides custom labels\n        (3) When some models do not support certain plots, they are dropped appropriately\n        (4) When some models do not support certain plots, they are dropped appropriately\n            even when user provides custom labels\n        (5) When user provides custom labels, the number of labels must match number of models\n    '
    data = load_pos_and_neg_data
    exp = TSForecastingExperiment()
    fh = 12
    fold = 2
    exp.setup(data=data, fh=fh, fold=fold, fold_strategy='sliding')
    m1 = exp.create_model('exp_smooth')
    models = [m1, m1]
    fig_data = exp.plot_model(models, plot=plot, return_data=True)
    assert fig_data.get('overlay_data').shape[1] == len(models)
    labels = ['Model 1', 'Model 2']
    fig_data = exp.plot_model(models, plot=plot, data_kwargs={'labels': labels}, return_data=True)
    assert fig_data.get('overlay_data').shape[1] == len(models)
    assert np.all(fig_data.get('overlay_data').columns.to_list() == labels)
    if not all_models_supported:
        m2 = exp.create_model('lr_cds_dt')
        models = [m1, m2, m1]
        fig_data = exp.plot_model(models, plot=plot, return_data=True)
        assert fig_data.get('overlay_data').shape[1] == len(models) - 1
        labels = ['Model 1', 'Model 2', 'Model 3']
        fig_data = exp.plot_model(models, plot=plot, data_kwargs={'labels': labels}, return_data=True)
        assert fig_data.get('overlay_data').shape[1] == len(models) - 1
        labels.remove('Model 2')
        assert np.all(fig_data.get('overlay_data').columns.to_list() == labels)
    models = [m1, m1]
    labels = ['Model 1']
    with pytest.raises(ValueError) as errmsg:
        fig_data = exp.plot_model(models, plot=plot, data_kwargs={'labels': labels})
    exceptionmsg = errmsg.value.args[0]
    assert 'Please provide a label corresponding to each model to proceed.' in exceptionmsg
    labels = ['Model 1', 'Model 2', 'Model 3']
    with pytest.raises(ValueError) as errmsg:
        fig_data = exp.plot_model(models, plot=plot, data_kwargs={'labels': labels})
    exceptionmsg = errmsg.value.args[0]
    assert 'Please provide a label corresponding to each model to proceed.' in exceptionmsg

def test_plot_final_model_exo():
    if False:
        while True:
            i = 10
    'Tests running plot model after running finalize_model when exogenous\n    variables are present. Fix for https://github.com/pycaret/pycaret/issues/3565\n    '
    data = get_data('uschange')
    target = 'Consumption'
    FH = 3
    train = data.iloc[:int(len(data) - FH)]
    test = data.iloc[int(len(data)) - FH:]
    test = test.drop(columns=[target], axis=1)
    exp = TSForecastingExperiment()
    exp.setup(data=train, target=target, fh=FH, session_id=42)
    model = exp.create_model('arima')
    final_model = exp.finalize_model(model)
    exp.plot_model(final_model, data_kwargs={'X': test})
    exp.plot_model()