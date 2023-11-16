"""Helper functions for time series tests
"""
import random
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sktime.forecasting.base import ForecastingHorizon
from pycaret.containers.models.time_series import get_all_model_containers
from pycaret.datasets import get_data
from pycaret.time_series import TSForecastingExperiment
from pycaret.utils.time_series import SeasonalPeriod
_BLEND_TEST_MODELS = ['naive', 'poly_trend', 'arima', 'auto_ets', 'lr_cds_dt', 'en_cds_dt', 'knn_cds_dt', 'dt_cds_dt', 'lightgbm_cds_dt']
_ALL_DATA_TYPES = ['transformed', 'imputed', 'original']
_ALL_STATS_TESTS = ['summary', 'white_noise', 'adf', 'kpss', 'normality', 'stationarity', 'all']
_ALL_STATS_TESTS_MISSING_DATA = [('summary', True), ('white_noise', False), ('adf', False), ('kpss', False), ('normality', True), ('stationarity', False), ('all', False)]
_IMPUTE_METHODS_STR = ['drift', 'linear', 'nearest', 'mean', 'median', 'backfill', 'bfill', 'pad', 'ffill', 'random']
_TRANSFORMATION_METHODS = ['box-cox', 'log', 'sqrt', 'exp', 'cos']
_TRANSFORMATION_METHODS_NO_NEG = ['box-cox', 'log']
_SCALE_METHODS = ['zscore', 'minmax', 'maxabs', 'robust']

def _get_all_plots():
    if False:
        print('Hello World!')
    exp = TSForecastingExperiment()
    data = get_data('airline')
    exp.setup(data=data)
    all_plots = list(exp._available_plots.keys())
    all_plots = [None] + all_plots
    return all_plots

def _get_all_plots_data():
    if False:
        return 10
    exp = TSForecastingExperiment()
    data = get_data('airline')
    exp.setup(data=data)
    all_plots = exp._available_plots_data_keys
    all_plots = [None] + all_plots
    return all_plots

def _get_all_plots_estimator():
    if False:
        for i in range(10):
            print('nop')
    exp = TSForecastingExperiment()
    data = get_data('airline')
    exp.setup(data=data)
    all_plots = exp._available_plots_estimator_keys
    all_plots = [None] + all_plots
    return all_plots
_ALL_PLOTS = _get_all_plots()
_ALL_PLOTS_DATA = _get_all_plots_data()
_ALL_PLOTS_ESTIMATOR = _get_all_plots_estimator()
_ALL_PLOTS_ESTIMATOR_NOT_DATA = list(set(_ALL_PLOTS_ESTIMATOR) - set(_ALL_PLOTS_DATA))

def _return_all_plots_estimator_ts_results():
    if False:
        while True:
            i = 10
    'Returns all plots that look at model results in time series format.\n    Also returns whether the plot is supported by all models or not.\n    '
    return [('forecast', True), ('insample', False), ('residuals', False)]

def _get_all_metrics():
    if False:
        for i in range(10):
            print('nop')
    exp = TSForecastingExperiment()
    data = get_data('airline')
    exp.setup(data=data)
    all_metrics = exp.get_metrics()['Name'].to_list()
    return all_metrics
_ALL_METRICS = _get_all_metrics()

def _get_seasonal_values():
    if False:
        for i in range(10):
            print('nop')
    return [(k, v.value) for (k, v) in SeasonalPeriod.__members__.items()]

def _get_seasonal_values_alphanumeric():
    if False:
        return 10
    'Check if frequency is alphanumeric and process it as needed'
    choice_list = ['10', '20', '30', '40', '50', '60']
    return [(random.choice(choice_list), k, v.value) for (k, v) in SeasonalPeriod.__members__.items()]

def _check_windows():
    if False:
        i = 10
        return i + 15
    'Check if the system is Windows.'
    import sys
    platform = sys.platform
    is_windows = True if platform.startswith('win') else False
    return is_windows

def _return_model_names():
    if False:
        return 10
    'Return all model names.'
    data = get_data('airline')
    exp = TSForecastingExperiment()
    exp.setup(data=data, session_id=42)
    model_containers = get_all_model_containers(exp)
    models_to_ignore = ['prophet', 'ensemble_forecaster'] if _check_windows() else ['ensemble_forecaster']
    model_names_ = []
    for model_name in model_containers.keys():
        if model_name not in models_to_ignore:
            model_names_.append(model_name)
    return model_names_

def _return_model_parameters():
    if False:
        for i in range(10):
            print('nop')
    'Parameterize individual models.\n    Returns the model names and the corresponding forecast horizons.\n    Horizons are alternately picked to be either\n        (1) integers or\n        (2) numpy arrays (continuous)\n        (3) numpy arrays (with gaps)\n    '
    model_names = _return_model_names()
    parameters = []
    for (i, name) in enumerate(model_names):
        if i % 3 == 0:
            fh = random.randint(6, 24)
        elif i % 3 == 1:
            fh = np.arange(1, random.randint(13, 25))
        else:
            fh = np.arange(random.randint(6, 12), random.randint(13, 25))
        parameters.append((name, fh))
    return parameters

def _return_splitter_args():
    if False:
        for i in range(10):
            print('nop')
    'fold, fh, fold_strategy'
    parametrize_list = [(random.randint(2, 5), random.randint(5, 10), 'expanding'), (random.randint(2, 5), random.randint(5, 10), 'rolling'), (random.randint(2, 5), random.randint(5, 10), 'sliding'), (random.randint(2, 5), np.arange(1, random.randint(5, 10)), 'expanding'), (random.randint(2, 5), np.arange(1, random.randint(5, 10)), 'rolling'), (random.randint(2, 5), np.arange(1, random.randint(5, 10)), 'sliding'), (random.randint(2, 5), np.arange(random.randint(3, 5), random.randint(6, 10)), 'expanding'), (random.randint(2, 5), np.arange(random.randint(3, 5), random.randint(6, 10)), 'rolling'), (random.randint(2, 5), np.arange(random.randint(3, 5), random.randint(6, 10)), 'sliding'), (random.randint(2, 5), [1, 2, 3, 4], 'expanding'), (random.randint(2, 5), [1, 2, 3, 4], 'rolling'), (random.randint(2, 5), [1, 2, 3, 4], 'sliding'), (random.randint(2, 5), [3, 4], 'expanding'), (random.randint(2, 5), [3, 4], 'rolling'), (random.randint(2, 5), [3, 4], 'sliding'), (random.randint(2, 5), ForecastingHorizon(np.arange(1, random.randint(5, 10))), 'expanding'), (random.randint(2, 5), ForecastingHorizon(np.arange(1, random.randint(5, 10))), 'rolling'), (random.randint(2, 5), ForecastingHorizon(np.arange(1, random.randint(5, 10))), 'sliding'), (random.randint(2, 5), ForecastingHorizon(np.arange(random.randint(3, 5), random.randint(6, 10))), 'expanding'), (random.randint(2, 5), ForecastingHorizon(np.arange(random.randint(3, 5), random.randint(6, 10))), 'rolling'), (random.randint(2, 5), ForecastingHorizon(np.arange(random.randint(3, 5), random.randint(6, 10))), 'sliding')]
    return parametrize_list

def _return_compare_model_args():
    if False:
        return 10
    'Returns cross_validation, log_experiment parameters respectively'
    parametrize_list = [(False, False), (False, True), (True, False), (True, True)]
    return parametrize_list

def _return_setup_args_raises():
    if False:
        print('Hello World!')
    ' '
    setup_raises_list = [(random.randint(50, 100), random.randint(10, 20), 'expanding'), (random.randint(50, 100), random.randint(10, 20), 'rolling'), (random.randint(50, 100), random.randint(10, 20), 'sliding')]
    return setup_raises_list

def _return_data_with_without_period_index():
    if False:
        i = 10
        return i + 15
    'Returns one dataset with period index and one with int index'
    datasets = [get_data('airline'), get_data('10', folder='time_series/white_noise')]
    return datasets

def _return_model_names_for_plots_stats():
    if False:
        print('Hello World!')
    'Returns models to be used for testing plots. Needs\n    - 1 model that has prediction interval ("theta")\n    - 1 model that does not have prediction interval ("lr_cds_dt")\n    - 1 model that has in-sample forecasts ("theta")\n    - 1 model that does not have in-sample forecasts ("lr_cds_dt")\n    '
    model_names = ['theta', 'lr_cds_dt']
    return model_names

def _return_model_names_for_missing_data():
    if False:
        while True:
            i = 10
    'Returns models that do not support missing data'
    model_names = ['ets', 'theta', 'lr_cds_dt']
    return model_names

def assert_frame_not_equal(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'https://stackoverflow.com/a/38778401/8925915'
    try:
        assert_frame_equal(*args, **kwargs)
    except AssertionError:
        pass
    else:
        raise AssertionError('Frames are equal, but should not be.')

def _return_data_big_small():
    if False:
        for i in range(10):
            print('nop')
    'Returns one dataset with 144 data points and one with < 12 data points'
    data = get_data('airline')
    data = data - 400
    data_small = data[:11]
    datasets = [data, data_small]
    return datasets

def _return_data_seasonal_types_strictly_pos():
    if False:
        return 10
    'Returns data with additive and multiplicative seasonal types\n    (with strictly positive values only)'
    N = 100
    y_trend = np.arange(100, 100 + N)
    y_season = 100 * (1 + np.sin(y_trend))
    y_add = pd.Series(y_trend + y_season)
    y_mul = pd.Series(y_trend * y_season)
    datasets = [y_add, y_mul]
    return datasets