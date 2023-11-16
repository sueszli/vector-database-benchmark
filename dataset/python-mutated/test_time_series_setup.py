"""Module to test time_series "setup" functionality
"""
import math
import numpy as np
import pandas as pd
import pytest
from time_series_test_utils import _get_seasonal_values, _get_seasonal_values_alphanumeric, _return_data_seasonal_types_strictly_pos, _return_setup_args_raises, _return_splitter_args
from pycaret.datasets import get_data
from pycaret.time_series import TSForecastingExperiment
_splitter_args = _return_splitter_args()
_setup_args_raises = _return_setup_args_raises()
_data_seasonal_types_strictly_pos = _return_data_seasonal_types_strictly_pos()

@pytest.mark.parametrize('fold, fh, fold_strategy', _splitter_args)
def test_splitter_using_fold_and_fh(fold, fh, fold_strategy, load_pos_and_neg_data):
    if False:
        for i in range(10):
            print('nop')
    'Tests the splitter creation using fold, fh and a string value for fold_strategy.'
    from sktime.forecasting.model_selection._split import ExpandingWindowSplitter, SlidingWindowSplitter
    from pycaret.time_series import setup
    exp_name = setup(data=load_pos_and_neg_data, fold=fold, fh=fh, fold_strategy=fold_strategy)
    allowed_fold_strategies = ['expanding', 'rolling', 'sliding']
    if fold_strategy in allowed_fold_strategies:
        if fold_strategy == 'expanding' or fold_strategy == 'rolling':
            assert isinstance(exp_name.fold_generator, ExpandingWindowSplitter)
        elif fold_strategy == 'sliding':
            assert isinstance(exp_name.fold_generator, SlidingWindowSplitter)
        if isinstance(fh, int):
            assert np.all(exp_name.fold_generator.fh == np.arange(1, fh + 1))
            assert exp_name.fold_generator.step_length == fh
        else:
            assert np.all(exp_name.fold_generator.fh == fh)
            assert exp_name.fold_generator.step_length == len(fh)

def test_splitter_pass_cv_object(load_pos_and_neg_data):
    if False:
        print('Hello World!')
    'Tests the passing of a `sktime` cv splitter to fold_strategy'
    from sktime.forecasting.model_selection._split import ExpandingWindowSplitter
    from pycaret.time_series import setup
    fold = 3
    fh = np.arange(1, 13)
    fh_extended = np.arange(1, 25)
    fold_strategy = ExpandingWindowSplitter(initial_window=72, step_length=12, fh=fh)
    exp_name = setup(data=load_pos_and_neg_data, fold=fold, fh=fh_extended, fold_strategy=fold_strategy)
    assert exp_name.fold_generator.initial_window == fold_strategy.initial_window
    assert np.all(exp_name.fold_generator.fh == fold_strategy.fh)
    assert exp_name.fold_generator.step_length == fold_strategy.step_length
    num_folds = exp_name.get_config('fold_param')
    y_train = exp_name.get_config('y_train')
    expected = fold_strategy.get_n_splits(y=y_train)
    assert num_folds == expected

@pytest.mark.parametrize('fold, fh, fold_strategy', _setup_args_raises)
def test_setup_raises(fold, fh, fold_strategy, load_pos_and_neg_data):
    if False:
        i = 10
        return i + 15
    'Tests conditions that raise an error due to lack of data'
    from pycaret.time_series import setup
    with pytest.raises(ValueError) as errmsg:
        _ = setup(data=load_pos_and_neg_data, fold=fold, fh=fh, fold_strategy=fold_strategy)
    exceptionmsg = errmsg.value.args[0]
    assert exceptionmsg == 'Not Enough Data Points, set a lower number of folds or fh'

def test_enforce_pi(load_pos_and_neg_data):
    if False:
        i = 10
        return i + 15
    'Tests the enforcement of prediction interval'
    data = load_pos_and_neg_data
    exp1 = TSForecastingExperiment()
    exp1.setup(data=data, point_alpha=0.5)
    num_models1 = len(exp1.models())
    exp2 = TSForecastingExperiment()
    exp2.setup(data=data, point_alpha=None)
    num_models2 = len(exp2.models())
    assert num_models1 < num_models2

def test_enforce_exogenous_no_exo_data(load_pos_and_neg_data):
    if False:
        while True:
            i = 10
    'Tests the enforcement of exogenous variable support in models when\n    univariate data without exogenous variables is passed.'
    data = load_pos_and_neg_data
    exp1 = TSForecastingExperiment()
    exp1.setup(data=data, enforce_exogenous=True)
    num_models1 = len(exp1.models())
    exp2 = TSForecastingExperiment()
    exp2.setup(data=data, enforce_exogenous=False)
    num_models2 = len(exp2.models())
    assert num_models1 == num_models2

def test_enforce_exogenous_exo_data(load_uni_exo_data_target):
    if False:
        i = 10
        return i + 15
    'Tests the enforcement of exogenous variable support in models when\n    univariate data with exogenous variables is passed.'
    (data, target) = load_uni_exo_data_target
    exp1 = TSForecastingExperiment()
    exp1.setup(data=data, target=target, enforce_exogenous=True)
    num_models1 = len(exp1.models())
    exp2 = TSForecastingExperiment()
    exp2.setup(data=data, target=target, enforce_exogenous=False)
    num_models2 = len(exp2.models())
    assert num_models1 < num_models2

def test_sp_to_use_using_index_and_user_def():
    if False:
        while True:
            i = 10
    'Seasonal Period detection using Indices (used before 3.0.0rc5). Also\n    tests the user defined seasonal periods when used in conjunction with "index".\n    '
    exp = TSForecastingExperiment()
    data = get_data('airline', verbose=False)
    exp.setup(data=data, sp_detection='index', verbose=False, session_id=42)
    assert exp.seasonal_period is None
    assert exp.sp_detection == 'index'
    assert exp.ignore_seasonality_test is False
    assert exp.candidate_sps == [12]
    assert exp.significant_sps == [12]
    assert exp.significant_sps_no_harmonics == [12]
    assert exp.all_sps_to_use == [12]
    assert exp.primary_sp_to_use == 12
    exp.setup(data=data, sp_detection='index', verbose=False, session_id=42, seasonal_period=['M', 6], num_sps_to_use=-1)
    assert exp.seasonal_period == ['M', 6]
    assert exp.sp_detection == 'user_defined'
    assert exp.ignore_seasonality_test is False
    assert exp.candidate_sps == [12, 6]
    assert exp.significant_sps == [12, 6]
    assert exp.significant_sps_no_harmonics == [12]
    assert exp.all_sps_to_use == [12, 6]
    assert exp.primary_sp_to_use == 12
    data = get_data('1', folder='time_series/white_noise', verbose=False)
    exp.setup(data=data, sp_detection='index', seasonal_period=12, verbose=False, session_id=42)
    assert exp.seasonal_period == 12
    assert exp.sp_detection == 'user_defined'
    assert exp.ignore_seasonality_test is False
    assert exp.candidate_sps == [12]
    assert exp.significant_sps == [1]
    assert exp.significant_sps_no_harmonics == [1]
    assert exp.all_sps_to_use == [1]
    assert exp.primary_sp_to_use == 1
    data = get_data('1', folder='time_series/white_noise', verbose=False)
    exp.setup(data=data, sp_detection='index', seasonal_period=12, ignore_seasonality_test=True, verbose=False, session_id=42)
    assert exp.seasonal_period == 12
    assert exp.sp_detection == 'user_defined'
    assert exp.ignore_seasonality_test is True
    assert exp.candidate_sps == [12]
    assert exp.significant_sps == [12]
    assert exp.significant_sps_no_harmonics == [12]
    assert exp.all_sps_to_use == [12]
    assert exp.primary_sp_to_use == 12

def test_sp_to_use_using_auto_and_user_def():
    if False:
        return 10
    'Seasonal Period detection using Statistical tests (used on and after 3.0.0rc5).\n    Also tests the user defined seasonal periods when used in conjunction with "auto".\n    '
    exp = TSForecastingExperiment()
    data = get_data('airline', verbose=False)
    exp.setup(data=data, sp_detection='auto', verbose=False, session_id=42)
    assert exp.candidate_sps == [12, 24, 36, 11, 48]
    assert exp.significant_sps == [12, 24, 36, 11, 48]
    assert exp.significant_sps_no_harmonics == [48, 36, 11]
    assert exp.all_sps_to_use == [12]
    assert exp.primary_sp_to_use == 12
    exp.setup(data=data, sp_detection='auto', num_sps_to_use=2, verbose=False, session_id=42)
    assert exp.candidate_sps == [12, 24, 36, 11, 48]
    assert exp.significant_sps == [12, 24, 36, 11, 48]
    assert exp.significant_sps_no_harmonics == [48, 36, 11]
    assert exp.all_sps_to_use == [12, 24]
    assert exp.primary_sp_to_use == 12
    exp.setup(data=data, sp_detection='auto', num_sps_to_use=100, verbose=False, session_id=42)
    assert exp.candidate_sps == [12, 24, 36, 11, 48]
    assert exp.significant_sps == [12, 24, 36, 11, 48]
    assert exp.significant_sps_no_harmonics == [48, 36, 11]
    assert exp.all_sps_to_use == [12, 24, 36, 11, 48]
    assert exp.primary_sp_to_use == 12
    np.random.seed(42)
    sp = 60
    data = np.random.randint(0, 100, size=sp)
    data = pd.DataFrame(np.concatenate((np.tile(data, 2), [data[0]])))
    exp = TSForecastingExperiment()
    exp.setup(data=data)
    assert exp.primary_sp_to_use == sp
    exp = TSForecastingExperiment()
    exp.setup(data=data.iloc[:2 * sp])
    assert exp.primary_sp_to_use < sp
    sp = 19
    exp.setup(data=data, seasonal_period=sp, sp_detection='auto', verbose=False, session_id=42)
    assert exp.seasonal_period == sp
    assert exp.sp_detection == 'user_defined'
    assert exp.ignore_seasonality_test is False
    assert exp.candidate_sps == [sp]
    assert exp.significant_sps == [1]
    assert exp.significant_sps_no_harmonics == [1]
    assert exp.all_sps_to_use == [1]
    assert exp.primary_sp_to_use == 1
    exp.setup(data=data, seasonal_period=sp, ignore_seasonality_test=True, sp_detection='auto', verbose=False, session_id=42)
    assert exp.seasonal_period == sp
    assert exp.sp_detection == 'user_defined'
    assert exp.ignore_seasonality_test is True
    assert exp.candidate_sps == [sp]
    assert exp.significant_sps == [sp]
    assert exp.significant_sps_no_harmonics == [sp]
    assert exp.all_sps_to_use == [sp]
    assert exp.primary_sp_to_use == sp

def test_sp_to_use_upto_max_sp():
    if False:
        return 10
    'Seasonal Period detection upto a max seasonal period provided by user.'
    data = get_data('airline', verbose=False)
    exp = TSForecastingExperiment()
    exp.setup(data=data, fh=12, session_id=42, remove_harmonics=False, max_sp_to_consider=None)
    assert exp.candidate_sps == [12, 24, 36, 11, 48]
    assert exp.significant_sps == [12, 24, 36, 11, 48]
    assert exp.significant_sps_no_harmonics == [48, 36, 11]
    assert exp.all_sps_to_use == [12]
    assert exp.primary_sp_to_use == 12
    exp = TSForecastingExperiment()
    exp.setup(data=data, fh=12, session_id=42, remove_harmonics=False, max_sp_to_consider=24)
    assert exp.candidate_sps == [12, 24, 11]
    assert exp.significant_sps == [12, 24, 11]
    assert exp.significant_sps_no_harmonics == [24, 11]
    assert exp.all_sps_to_use == [12]
    assert exp.primary_sp_to_use == 12
    exp = TSForecastingExperiment()
    exp.setup(data=data, fh=12, session_id=42, remove_harmonics=True, max_sp_to_consider=24)
    assert exp.candidate_sps == [12, 24, 11]
    assert exp.significant_sps == [12, 24, 11]
    assert exp.significant_sps_no_harmonics == [24, 11]
    assert exp.all_sps_to_use == [24]
    assert exp.primary_sp_to_use == 24
    exp = TSForecastingExperiment()
    exp.setup(data=data, fh=12, session_id=42, remove_harmonics=False, max_sp_to_consider=2)
    assert exp.candidate_sps == []
    assert exp.significant_sps == [1]
    assert exp.significant_sps_no_harmonics == [1]
    assert exp.all_sps_to_use == [1]
    assert exp.primary_sp_to_use == 1
    exp = TSForecastingExperiment()
    exp.setup(data=data, fh=12, session_id=42, remove_harmonics=True, max_sp_to_consider=2)
    assert exp.candidate_sps == []
    assert exp.significant_sps == [1]
    assert exp.significant_sps_no_harmonics == [1]
    assert exp.all_sps_to_use == [1]
    assert exp.primary_sp_to_use == 1

@pytest.mark.parametrize('seasonal_key, seasonal_value', _get_seasonal_values())
def test_setup_seasonal_period_int(load_pos_and_neg_data, seasonal_key, seasonal_value):
    if False:
        for i in range(10):
            print('nop')
    exp = TSForecastingExperiment()
    fh = np.arange(1, 13)
    fold = 2
    data = load_pos_and_neg_data
    exp.setup(data=data, fh=fh, fold=fold, fold_strategy='sliding', verbose=False, seasonal_period=seasonal_value)
    assert exp.candidate_sps == [seasonal_value]

@pytest.mark.parametrize('seasonal_period, seasonal_value', _get_seasonal_values())
def test_setup_seasonal_period_str(load_pos_and_neg_data, seasonal_period, seasonal_value):
    if False:
        return 10
    exp = TSForecastingExperiment()
    fh = np.arange(1, 13)
    fold = 2
    data = load_pos_and_neg_data
    exp.setup(data=data, fh=fh, fold=fold, fold_strategy='sliding', verbose=False, session_id=42, seasonal_period=seasonal_period)
    assert exp.candidate_sps == [seasonal_value]

@pytest.mark.parametrize('prefix, seasonal_period, seasonal_value', _get_seasonal_values_alphanumeric())
def test_setup_seasonal_period_alphanumeric(load_pos_and_neg_data, prefix, seasonal_period, seasonal_value):
    if False:
        while True:
            i = 10
    'Tests the get_sp_from_str function with different values of frequency'
    seasonal_period = prefix + seasonal_period
    prefix = int(prefix)
    lcm = abs(seasonal_value * prefix) // math.gcd(seasonal_value, prefix)
    expected_candidate_sps = [int(lcm / prefix)]
    exp = TSForecastingExperiment()
    fh = np.arange(1, 13)
    fold = 2
    data = load_pos_and_neg_data
    exp.setup(data=data, fh=fh, fold=fold, fold_strategy='sliding', verbose=False, seasonal_period=seasonal_period)
    assert exp.candidate_sps == expected_candidate_sps

def test_train_test_split_uni_no_exo(load_pos_and_neg_data):
    if False:
        for i in range(10):
            print('nop')
    'Tests the train-test splits for univariate time series without exogenous variables'
    data = load_pos_and_neg_data
    exp = TSForecastingExperiment()
    fh = 12
    exp.setup(data=data, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[:len(data) - fh].index)
    assert np.all(exp.test.index == data.iloc[-fh:].index)
    assert exp.X is None
    assert np.all(exp.y.index == data.index)
    assert exp.X_train is None
    assert exp.X_test is None
    assert np.all(exp.y_train.index == data.iloc[:len(data) - fh].index)
    assert np.all(exp.y_test.index == data.iloc[-fh:].index)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(exp.train_transformed.index == data.iloc[:len(data) - fh].index)
    assert np.all(exp.test_transformed.index == data.iloc[-fh:].index)
    assert exp.X_transformed is None
    assert np.all(exp.y_transformed.index == data.index)
    assert exp.X_train_transformed is None
    assert exp.X_test_transformed is None
    assert np.all(exp.y_train_transformed.index == data.iloc[:len(data) - fh].index)
    assert np.all(exp.y_test_transformed.index == data.iloc[-fh:].index)
    exp = TSForecastingExperiment()
    fh = np.arange(1, 10)
    exp.setup(data=data, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[:len(data) - max(fh)].index)
    assert np.all(exp.test.index == data.iloc[-len(fh):].index)
    assert exp.X is None
    assert np.all(exp.y.index == data.index)
    assert exp.X_train is None
    assert exp.X_test is None
    assert np.all(exp.y_train.index == data.iloc[:len(data) - max(fh)].index)
    assert np.all(exp.y_test.index == data.iloc[-len(fh):].index)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(exp.train_transformed.index == data.iloc[:len(data) - max(fh)].index)
    assert np.all(exp.test_transformed.index == data.iloc[-len(fh):].index)
    assert exp.X_transformed is None
    assert np.all(exp.y_transformed.index == data.index)
    assert exp.X_train_transformed is None
    assert exp.X_test_transformed is None
    assert np.all(exp.y_train_transformed.index == data.iloc[:len(data) - max(fh)].index)
    assert np.all(exp.y_test_transformed.index == data.iloc[-len(fh):].index)
    exp = TSForecastingExperiment()
    fh = [1, 2, 3, 4, 5, 6]
    exp.setup(data=data, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[:len(data) - max(fh)].index)
    assert np.all(exp.test.index == data.iloc[-len(fh):].index)
    assert exp.X is None
    assert np.all(exp.y.index == data.index)
    assert exp.X_train is None
    assert exp.X_test is None
    assert np.all(exp.y_train.index == data.iloc[:len(data) - max(fh)].index)
    assert np.all(exp.y_test.index == data.iloc[-len(fh):].index)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(exp.train_transformed.index == data.iloc[:len(data) - max(fh)].index)
    assert np.all(exp.test_transformed.index == data.iloc[-len(fh):].index)
    assert exp.X_transformed is None
    assert np.all(exp.y_transformed.index == data.index)
    assert exp.X_train_transformed is None
    assert exp.X_test_transformed is None
    assert np.all(exp.y_train_transformed.index == data.iloc[:len(data) - max(fh)].index)
    assert np.all(exp.y_test_transformed.index == data.iloc[-len(fh):].index)
    exp = TSForecastingExperiment()
    fh = np.arange(7, 13)
    exp.setup(data=data, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.test) == len(fh)
    assert exp.X is None
    assert np.all(exp.y.index == data.index)
    assert exp.X_train is None
    assert exp.X_test is None
    assert np.all(exp.y_train.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.y_test) == len(fh)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(exp.train_transformed.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.test_transformed) == len(fh)
    assert exp.X_transformed is None
    assert np.all(exp.y_transformed.index == data.index)
    assert exp.X_train_transformed is None
    assert exp.X_test_transformed is None
    assert np.all(exp.y_train_transformed.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.y_test_transformed) == len(fh)
    exp = TSForecastingExperiment()
    fh = [4, 5, 6]
    exp.setup(data=data, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.test) == len(fh)
    assert exp.X is None
    assert np.all(exp.y.index == data.index)
    assert exp.X_train is None
    assert exp.X_test is None
    assert np.all(exp.y_train.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.y_test) == len(fh)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(exp.train_transformed.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.test_transformed) == len(fh)
    assert exp.X_transformed is None
    assert np.all(exp.y_transformed.index == data.index)
    assert exp.X_train_transformed is None
    assert exp.X_test_transformed is None
    assert np.all(exp.y_train_transformed.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.y_test_transformed) == len(fh)
    exp = TSForecastingExperiment()
    fh = np.array([4, 5, 6, 10, 11, 12])
    exp.setup(data=data, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.test) == len(fh)
    assert exp.X is None
    assert np.all(exp.y.index == data.index)
    assert exp.X_train is None
    assert exp.X_test is None
    assert np.all(exp.y_train.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.y_test) == len(fh)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(exp.train_transformed.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.test_transformed) == len(fh)
    assert exp.X_transformed is None
    assert np.all(exp.y_transformed.index == data.index)
    assert exp.X_train_transformed is None
    assert exp.X_test_transformed is None
    assert np.all(exp.y_train_transformed.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.y_test_transformed) == len(fh)
    exp = TSForecastingExperiment()
    fh = [4, 5, 6, 10, 11, 12]
    exp.setup(data=data, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.test) == len(fh)
    assert exp.X is None
    assert np.all(exp.y.index == data.index)
    assert exp.X_train is None
    assert exp.X_test is None
    assert np.all(exp.y_train.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.y_test) == len(fh)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(exp.train_transformed.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.test_transformed) == len(fh)
    assert exp.X_transformed is None
    assert np.all(exp.y_transformed.index == data.index)
    assert exp.X_train_transformed is None
    assert exp.X_test_transformed is None
    assert np.all(exp.y_train_transformed.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.y_test_transformed) == len(fh)

def test_train_test_split_uni_exo(load_uni_exo_data_target):
    if False:
        print('Hello World!')
    'Tests the train-test splits for univariate time series with exogenous variables'
    (data, target) = load_uni_exo_data_target
    exp = TSForecastingExperiment()
    fh = 12
    exp.setup(data=data, target=target, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[:len(data) - fh].index)
    assert np.all(exp.test.index == data.iloc[-fh:].index)
    assert np.all(exp.X.index == data.index)
    assert np.all(exp.y.index == data.index)
    assert np.all(exp.X_train.index == data.iloc[:len(data) - fh].index)
    assert np.all(exp.X_test.index == data.iloc[-fh:].index)
    assert np.all(exp.y_train.index == data.iloc[:len(data) - fh].index)
    assert np.all(exp.y_test.index == data.iloc[-fh:].index)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(exp.train_transformed.index == data.iloc[:len(data) - fh].index)
    assert np.all(exp.test_transformed.index == data.iloc[-fh:].index)
    assert np.all(exp.X_transformed.index == data.index)
    assert np.all(exp.y_transformed.index == data.index)
    assert np.all(exp.X_train_transformed.index == data.iloc[:len(data) - fh].index)
    assert np.all(exp.X_test_transformed.index == data.iloc[-fh:].index)
    assert np.all(exp.y_train_transformed.index == data.iloc[:len(data) - fh].index)
    assert np.all(exp.y_test_transformed.index == data.iloc[-fh:].index)
    exp = TSForecastingExperiment()
    fh = np.arange(1, 10)
    exp.setup(data=data, target=target, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[:len(data) - max(fh)].index)
    assert np.all(exp.test.index == data.iloc[-len(fh):].index)
    assert np.all(exp.X.index == data.index)
    assert np.all(exp.y.index == data.index)
    assert np.all(exp.X_train.index == data.iloc[:len(data) - max(fh)].index)
    assert np.all(exp.X_test.index == data.iloc[-len(fh):].index)
    assert np.all(exp.y_train.index == data.iloc[:len(data) - max(fh)].index)
    assert np.all(exp.y_test.index == data.iloc[-len(fh):].index)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(exp.train_transformed.index == data.iloc[:len(data) - max(fh)].index)
    assert np.all(exp.test_transformed.index == data.iloc[-len(fh):].index)
    assert np.all(exp.X_transformed.index == data.index)
    assert np.all(exp.y_transformed.index == data.index)
    assert np.all(exp.X_train_transformed.index == data.iloc[:len(data) - max(fh)].index)
    assert np.all(exp.X_test_transformed.index == data.iloc[-len(fh):].index)
    assert np.all(exp.y_train_transformed.index == data.iloc[:len(data) - max(fh)].index)
    assert np.all(exp.y_test_transformed.index == data.iloc[-len(fh):].index)
    exp = TSForecastingExperiment()
    fh = [1, 2, 3, 4, 5, 6]
    exp.setup(data=data, target=target, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[:len(data) - max(fh)].index)
    assert np.all(exp.test.index == data.iloc[-len(fh):].index)
    assert np.all(exp.X.index == data.index)
    assert np.all(exp.y.index == data.index)
    assert np.all(exp.X_train.index == data.iloc[:len(data) - max(fh)].index)
    assert np.all(exp.X_test.index == data.iloc[-len(fh):].index)
    assert np.all(exp.y_train.index == data.iloc[:len(data) - max(fh)].index)
    assert np.all(exp.y_test.index == data.iloc[-len(fh):].index)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(exp.train_transformed.index == data.iloc[:len(data) - max(fh)].index)
    assert np.all(exp.test_transformed.index == data.iloc[-len(fh):].index)
    assert np.all(exp.X_transformed.index == data.index)
    assert np.all(exp.y_transformed.index == data.index)
    assert np.all(exp.X_train_transformed.index == data.iloc[:len(data) - max(fh)].index)
    assert np.all(exp.X_test_transformed.index == data.iloc[-len(fh):].index)
    assert np.all(exp.y_train_transformed.index == data.iloc[:len(data) - max(fh)].index)
    assert np.all(exp.y_test_transformed.index == data.iloc[-len(fh):].index)
    exp = TSForecastingExperiment()
    fh = np.arange(7, 13)
    exp.setup(data=data, target=target, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.test) == len(fh)
    assert np.all(exp.X.index == data.index)
    assert np.all(exp.y.index == data.index)
    assert np.all(exp.X_train.index == data.iloc[:len(data) - max(fh)].index)
    assert np.all(exp.X_test.index == data.iloc[-max(fh):].index)
    assert np.all(exp.y_train.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.y_test) == len(fh)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(exp.train_transformed.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.test_transformed) == len(fh)
    assert np.all(exp.X_transformed.index == data.index)
    assert np.all(exp.y_transformed.index == data.index)
    exp = TSForecastingExperiment()
    fh = [4, 5, 6]
    exp.setup(data=data, target=target, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.test) == len(fh)
    assert np.all(exp.X.index == data.index)
    assert np.all(exp.y.index == data.index)
    assert np.all(exp.X_train.index == data.iloc[:len(data) - max(fh)].index)
    assert np.all(exp.X_test.index == data.iloc[-max(fh):].index)
    assert np.all(exp.y_train.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.y_test) == len(fh)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(exp.train_transformed.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.test_transformed) == len(fh)
    assert np.all(exp.X_transformed.index == data.index)
    assert np.all(exp.y_transformed.index == data.index)
    exp = TSForecastingExperiment()
    fh = np.array([4, 5, 6, 10, 11, 12])
    exp.setup(data=data, target=target, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.test) == len(fh)
    assert np.all(exp.X.index == data.index)
    assert np.all(exp.y.index == data.index)
    assert np.all(exp.X_train.index == data.iloc[:len(data) - max(fh)].index)
    assert np.all(exp.X_test.index == data.iloc[-max(fh):].index)
    assert np.all(exp.y_train.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.y_test) == len(fh)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(exp.train_transformed.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.test_transformed) == len(fh)
    assert np.all(exp.X_transformed.index == data.index)
    assert np.all(exp.y_transformed.index == data.index)
    assert np.all(exp.X_train_transformed.index == data.iloc[:len(data) - max(fh)].index)
    assert np.all(exp.X_test_transformed.index == data.iloc[-max(fh):].index)
    assert np.all(exp.y_train_transformed.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.y_test_transformed) == len(fh)
    exp = TSForecastingExperiment()
    fh = [4, 5, 6, 10, 11, 12]
    exp.setup(data=data, target=target, fh=fh, session_id=42)
    assert np.all(exp.dataset.index == data.index)
    assert np.all(exp.train.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.test) == len(fh)
    assert np.all(exp.X.index == data.index)
    assert np.all(exp.y.index == data.index)
    assert np.all(exp.X_train.index == data.iloc[:len(data) - max(fh)].index)
    assert np.all(exp.X_test.index == data.iloc[-max(fh):].index)
    assert np.all(exp.y_train.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.y_test) == len(fh)
    assert np.all(exp.dataset_transformed.index == data.index)
    assert np.all(exp.train_transformed.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.test_transformed) == len(fh)
    assert np.all(exp.X_transformed.index == data.index)
    assert np.all(exp.y_transformed.index == data.index)
    assert np.all(exp.X_train_transformed.index == data.iloc[:len(data) - max(fh)].index)
    assert np.all(exp.X_test_transformed.index == data.iloc[-max(fh):].index)
    assert np.all(exp.y_train_transformed.index == data.iloc[:len(data) - max(fh)].index)
    assert len(exp.y_test_transformed) == len(fh)

def test_missing_indices():
    if False:
        while True:
            i = 10
    'Tests setup when data has missing indices'
    data = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
    data['ds'] = pd.to_datetime(data['ds'])
    data.set_index('ds', inplace=True)
    data.index = data.index.to_period('D')
    data.info()
    exp = TSForecastingExperiment()
    with pytest.raises(ValueError) as errmsg:
        exp.setup(data=data, fh=365, session_id=42)
    exceptionmsg = errmsg.value.args[0]
    assert 'Data has missing indices!' in exceptionmsg

def test_hyperparameter_splits():
    if False:
        print('Hello World!')
    'Tests the splits to use to determine the hyperparameters'
    data = get_data('airline')
    FOLD = 1
    FH = 60
    TRAIN_SIZE = len(data) - FH
    data[:TRAIN_SIZE] = 1
    print('Experiment 1 ----')
    exp1 = TSForecastingExperiment()
    exp1.setup(data=data, fh=FH, fold=FOLD)
    print('Experiment 2 ----')
    exp2 = TSForecastingExperiment()
    exp2.setup(data=data, hyperparameter_split='train', fh=FH, fold=FOLD)
    assert exp1.primary_sp_to_use != exp2.primary_sp_to_use
    assert exp1.lowercase_d != exp2.lowercase_d
    assert exp1.white_noise != exp2.white_noise
    assert exp1.uppercase_d == exp2.uppercase_d
    data = get_data('airline')
    FOLD = 1
    FH = 36
    TRAIN_SIZE = len(data) - FH
    np.random.seed(42)
    indices = np.random.randint(1, int(TRAIN_SIZE / 2), 12)
    data.iloc[indices] = 200
    exp1 = TSForecastingExperiment()
    exp1.setup(data=data, fh=FH, fold=FOLD)
    exp2 = TSForecastingExperiment()
    exp2.setup(data=data, hyperparameter_split='train', fh=FH, fold=FOLD)
    assert exp1.uppercase_d != exp2.uppercase_d

@pytest.mark.parametrize('index', ['RangeIndex', 'DatetimeIndex'])
@pytest.mark.parametrize('seasonality_type', ['mul', 'add', 'auto'])
def test_seasonality_type_no_season(index: str, seasonality_type: str):
    if False:
        return 10
    'Tests the detection of the seasonality type with data that has no seasonality.\n\n    Parameters\n    ----------\n    index : str\n        Type of index. Options are: "RangeIndex" and "DatetimeIndex"\n    seasonality_type : str\n        The seasonality type to pass to setup\n    '
    N = 100
    y = pd.Series(np.arange(100, 100 + N))
    if index == 'DatetimeIndex':
        dates = pd.date_range(start='2020-01-01', periods=N, freq='MS')
        y.index = dates
    err_msg = 'Expected seasonality_type = None, but got something else.'
    exp = TSForecastingExperiment()
    exp.setup(data=y, seasonality_type=seasonality_type, session_id=42)
    assert exp.seasonality_type is None, err_msg

@pytest.mark.parametrize('index', ['RangeIndex', 'DatetimeIndex'])
@pytest.mark.parametrize('seasonality_type', ['mul', 'add', 'auto'])
@pytest.mark.parametrize('y', _data_seasonal_types_strictly_pos, ids=['data_add', 'data_mul'])
def test_seasonality_type_with_season_not_stricly_positive(index: str, seasonality_type: str, y: pd.Series):
    if False:
        for i in range(10):
            print('nop')
    'Tests the detection of the seasonality type with user defined type and\n    data that has seasonality and is not strictly positive.\n\n    Parameters\n    ----------\n    index : str\n        Type of index. Options are: "RangeIndex" and "DatetimeIndex"\n    seasonality_type : str\n        The seasonality type to pass to setup\n    y : pd.Series\n        Dataset to use\n    '
    y = y - y.max()
    if index == 'DatetimeIndex':
        dates = pd.date_range(start='2020-01-01', periods=len(y), freq='MS')
        y.index = dates
    err_msg = "Expected 'additive' seasonality, got something else"
    exp = TSForecastingExperiment()
    exp.setup(data=y, seasonality_type=seasonality_type, session_id=42)
    assert exp.seasonality_type == 'add', err_msg

@pytest.mark.parametrize('index', ['RangeIndex', 'DatetimeIndex'])
@pytest.mark.parametrize('seasonality_type', ['mul', 'add'])
@pytest.mark.parametrize('y', _data_seasonal_types_strictly_pos, ids=['data_add', 'data_mul'])
def test_seasonality_type_user_def_with_season_strictly_pos(index: str, seasonality_type: str, y: pd.Series):
    if False:
        while True:
            i = 10
    'Tests the detection of the seasonality type with user defined type and\n    data that has seasonality and is strictly positive.\n\n    Parameters\n    ----------\n    index : str\n        Type of index. Options are: "RangeIndex" and "DatetimeIndex"\n    seasonality_type : str\n        The seasonality type to pass to setup\n    y : pd.Series\n        Dataset to use\n    '
    if index == 'DatetimeIndex':
        dates = pd.date_range(start='2020-01-01', periods=len(y), freq='MS')
        y.index = dates
    err_msg = f"Expected '{seasonality_type}' seasonality, got something else"
    exp = TSForecastingExperiment()
    exp.setup(data=y, seasonality_type=seasonality_type, session_id=42)
    assert exp.seasonality_type == seasonality_type, err_msg

@pytest.mark.parametrize('index', ['RangeIndex', 'DatetimeIndex'])
@pytest.mark.parametrize('seasonality_type', ['auto'])
def test_seasonality_type_auto_with_season_strictly_pos(index: str, seasonality_type: str):
    if False:
        for i in range(10):
            print('nop')
    'Tests the detection of the seasonality type using the internal auto algorithm\n    when data that has seasonality and is strictly positive.\n\n    Tests various index types and tests for both additive and multiplicative\n    seasonality.\n\n    Parameters\n    ----------\n    index : str\n        Type of index. Options are: "RangeIndex" and "DatetimeIndex"\n    seasonality_type : str\n        The seasonality type to pass to setup\n    '
    N = 100
    y_trend = np.arange(100, 100 + N)
    y_season = 100 * (1 + np.sin(y_trend))
    y = pd.Series(y_trend + y_season)
    if index == 'DatetimeIndex':
        dates = pd.date_range(start='2020-01-01', periods=N, freq='MS')
        y.index = dates
    err_msg = 'Expected additive seasonality, got multiplicative'
    exp = TSForecastingExperiment()
    exp.setup(data=y, seasonality_type=seasonality_type, session_id=42)
    assert exp.seasonality_type == 'add', err_msg
    y = get_data('airline', verbose=False)
    err_msg = 'Expected multiplicative seasonality, got additive (2)'
    exp = TSForecastingExperiment()
    exp.setup(data=y, seasonality_type=seasonality_type, session_id=42)
    assert exp.seasonality_type == 'mul', err_msg