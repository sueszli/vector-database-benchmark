import numpy as np
import pytest
from numpy.testing import assert_allclose
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import lake
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen, _validate_fixed_params, _package_fixed_and_free_params_info, _stitch_fixed_and_free_params
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tools.tools import Bunch

@pytest.mark.low_precision('Test against Example 5.1.7 in Brockwell and Davis (2016)')
def test_brockwell_davis_example_517():
    if False:
        i = 10
        return i + 15
    endog = lake.copy()
    (hr, _) = hannan_rissanen(endog, ar_order=1, ma_order=1, demean=True, initial_ar_order=22, unbiased=False)
    assert_allclose(hr.ar_params, [0.6961], atol=0.0001)
    assert_allclose(hr.ma_params, [0.3788], atol=0.0001)
    (u, v) = arma_innovations(endog - endog.mean(), hr.ar_params, hr.ma_params, sigma2=1)
    tmp = u / v ** 0.5
    assert_allclose(np.inner(tmp, tmp) / len(u), 0.4774, atol=0.0001)

def test_itsmr():
    if False:
        return 10
    endog = lake.copy()
    (hr, _) = hannan_rissanen(endog, ar_order=1, ma_order=1, demean=True, initial_ar_order=22, unbiased=False)
    assert_allclose(hr.ar_params, [0.69607715], atol=0.0001)
    assert_allclose(hr.ma_params, [0.3787969217], atol=0.0001)
    (u, v) = arma_innovations(endog - endog.mean(), hr.ar_params, hr.ma_params, sigma2=1)
    tmp = u / v ** 0.5
    assert_allclose(np.inner(tmp, tmp) / len(u), 0.4773580109, atol=0.0001)

@pytest.mark.xfail(reason='TODO: improve checks on valid order parameters.')
def test_initial_order():
    if False:
        print('Hello World!')
    endog = np.arange(20) * 1.0
    hannan_rissanen(endog, ar_order=2, ma_order=0, initial_ar_order=1)
    hannan_rissanen(endog, ar_order=0, ma_order=2, initial_ar_order=1)
    hannan_rissanen(endog, ar_order=0, ma_order=2, initial_ar_order=20)

@pytest.mark.xfail(reason='TODO: improve checks on valid order parameters.')
def test_invalid_orders():
    if False:
        while True:
            i = 10
    endog = np.arange(2) * 1.0
    hannan_rissanen(endog, ar_order=2)
    hannan_rissanen(endog, ma_order=2)

@pytest.mark.todo('Improve checks on valid order parameters.')
@pytest.mark.smoke
def test_nonconsecutive_lags():
    if False:
        i = 10
        return i + 15
    endog = np.arange(20) * 1.0
    hannan_rissanen(endog, ar_order=[1, 4])
    hannan_rissanen(endog, ma_order=[1, 3])
    hannan_rissanen(endog, ar_order=[1, 4], ma_order=[1, 3])
    hannan_rissanen(endog, ar_order=[0, 0, 1])
    hannan_rissanen(endog, ma_order=[0, 0, 1])
    hannan_rissanen(endog, ar_order=[0, 0, 1], ma_order=[0, 0, 1])
    hannan_rissanen(endog, ar_order=0, ma_order=0)

def test_unbiased_error():
    if False:
        return 10
    endog = np.arange(1000) * 1.0
    with pytest.raises(ValueError, match='Cannot perform third step'):
        hannan_rissanen(endog, ar_order=1, ma_order=1, unbiased=True)

def test_set_default_unbiased():
    if False:
        for i in range(10):
            print('nop')
    endog = lake.copy()
    (p_1, other_results_2) = hannan_rissanen(endog, ar_order=1, ma_order=1, unbiased=None)
    (p_2, other_results_1) = hannan_rissanen(endog, ar_order=1, ma_order=1, unbiased=True)
    np.testing.assert_array_equal(p_1.ar_params, p_2.ar_params)
    np.testing.assert_array_equal(p_1.ma_params, p_2.ma_params)
    assert p_1.sigma2 == p_2.sigma2
    np.testing.assert_array_equal(other_results_1.resid, other_results_2.resid)
    (p_3, _) = hannan_rissanen(endog, ar_order=1, ma_order=1, unbiased=False)
    assert not np.array_equal(p_1.ar_params, p_3.ar_params)

@pytest.mark.parametrize('ar_order, ma_order, fixed_params, invalid_fixed_params', [(2, [1, 0, 1], None, None), ([0, 1], 0, {}, None), (1, 3, {'ar.L2': 1, 'ma.L2': 0}, ['ar.L2']), ([0, 1], [0, 0, 1], {'ma.L1': 0, 'sigma2': 1}, ['ma.L2', 'sigma2']), (0, 0, {'ma.L1': 0, 'ar.L1': 0}, ['ar.L1', 'ma.L1']), (5, [1, 0], {'random_param': 0, 'ar.L1': 0}, ['random_param']), (0, 2, {'ma.L1': -1, 'ma.L2': 1}, None), (1, 0, {'ar.L1': 0}, None), ([1, 0, 1], 3, {'ma.L2': 1, 'ar.L3': -1}, None), (2, 2, {'ma.L1': 1, 'ma.L2': 1, 'ar.L1': 1, 'ar.L2': 1}, None)])
def test_validate_fixed_params(ar_order, ma_order, fixed_params, invalid_fixed_params):
    if False:
        print('Hello World!')
    endog = np.random.normal(size=100)
    spec = SARIMAXSpecification(endog, ar_order=ar_order, ma_order=ma_order)
    if invalid_fixed_params is None:
        _validate_fixed_params(fixed_params, spec.param_names)
        hannan_rissanen(endog, ar_order=ar_order, ma_order=ma_order, fixed_params=fixed_params, unbiased=False)
    else:
        valid_params = sorted(list(set(spec.param_names) - {'sigma2'}))
        msg = f'Invalid fixed parameter(s): {invalid_fixed_params}. Please select among {valid_params}.'
        with pytest.raises(ValueError) as e:
            _validate_fixed_params(fixed_params, spec.param_names)
            assert e.msg == msg
        with pytest.raises(ValueError) as e:
            hannan_rissanen(endog, ar_order=ar_order, ma_order=ma_order, fixed_params=fixed_params, unbiased=False)
            assert e.msg == msg

@pytest.mark.parametrize('fixed_params, spec_ar_lags, spec_ma_lags, expected_bunch', [({}, [1], [], Bunch(fixed_ar_lags=[], fixed_ma_lags=[], free_ar_lags=[1], free_ma_lags=[], fixed_ar_ix=np.array([], dtype=int), fixed_ma_ix=np.array([], dtype=int), free_ar_ix=np.array([0], dtype=int), free_ma_ix=np.array([], dtype=int), fixed_ar_params=np.array([]), fixed_ma_params=np.array([]))), ({'ar.L2': 0.1, 'ma.L1': 0.2}, [2], [1, 3], Bunch(fixed_ar_lags=[2], fixed_ma_lags=[1], free_ar_lags=[], free_ma_lags=[3], fixed_ar_ix=np.array([1], dtype=int), fixed_ma_ix=np.array([0], dtype=int), free_ar_ix=np.array([], dtype=int), free_ma_ix=np.array([2], dtype=int), fixed_ar_params=np.array([0.1]), fixed_ma_params=np.array([0.2]))), ({'ma.L5': 0.1, 'ma.L10': 0.2}, [], [5, 10], Bunch(fixed_ar_lags=[], fixed_ma_lags=[5, 10], free_ar_lags=[], free_ma_lags=[], fixed_ar_ix=np.array([], dtype=int), fixed_ma_ix=np.array([4, 9], dtype=int), free_ar_ix=np.array([], dtype=int), free_ma_ix=np.array([], dtype=int), fixed_ar_params=np.array([]), fixed_ma_params=np.array([0.1, 0.2])))])
def test_package_fixed_and_free_params_info(fixed_params, spec_ar_lags, spec_ma_lags, expected_bunch):
    if False:
        return 10
    actual_bunch = _package_fixed_and_free_params_info(fixed_params, spec_ar_lags, spec_ma_lags)
    assert isinstance(actual_bunch, Bunch)
    assert len(actual_bunch) == len(expected_bunch)
    assert actual_bunch.keys() == expected_bunch.keys()
    lags = ['fixed_ar_lags', 'fixed_ma_lags', 'free_ar_lags', 'free_ma_lags']
    for k in lags:
        assert isinstance(actual_bunch[k], list)
        assert actual_bunch[k] == expected_bunch[k]
    ixs = ['fixed_ar_ix', 'fixed_ma_ix', 'free_ar_ix', 'free_ma_ix']
    for k in ixs:
        assert isinstance(actual_bunch[k], np.ndarray)
        assert actual_bunch[k].dtype in [np.int64, np.int32]
        np.testing.assert_array_equal(actual_bunch[k], expected_bunch[k])
    params = ['fixed_ar_params', 'fixed_ma_params']
    for k in params:
        assert isinstance(actual_bunch[k], np.ndarray)
        np.testing.assert_array_equal(actual_bunch[k], expected_bunch[k])

@pytest.mark.parametrize('fixed_lags, free_lags, fixed_params, free_params, spec_lags, expected_all_params', [([], [], [], [], [], []), ([2], [], [0.2], [], [2], [0.2]), ([], [1], [], [0.2], [1], [0.2]), ([1], [3], [0.2], [-0.2], [1, 3], [0.2, -0.2]), ([3], [1, 2], [0.2], [0.3, -0.2], [1, 2, 3], [0.3, -0.2, 0.2]), ([3, 1], [2, 4], [0.3, 0.1], [0.5, 0.0], [1, 2, 3, 4], [0.1, 0.5, 0.3, 0.0]), ([3, 10], [1, 2], [0.2, 0.5], [0.3, -0.2], [1, 2, 3, 10], [0.3, -0.2, 0.2, 0.5]), ([3, 10], [1, 2], [0.2, 0.5], [0.3, -0.2], [3, 1, 10, 2], [0.2, 0.3, 0.5, -0.2])])
def test_stitch_fixed_and_free_params(fixed_lags, free_lags, fixed_params, free_params, spec_lags, expected_all_params):
    if False:
        for i in range(10):
            print('nop')
    actual_all_params = _stitch_fixed_and_free_params(fixed_lags, fixed_params, free_lags, free_params, spec_lags)
    assert actual_all_params == expected_all_params

@pytest.mark.parametrize('fixed_params', [{'ar.L1': 0.69607715}, {'ma.L1': 0.37879692}, {'ar.L1': 0.69607715, 'ma.L1': 0.37879692}])
def test_itsmr_with_fixed_params(fixed_params):
    if False:
        print('Hello World!')
    endog = lake.copy()
    (hr, _) = hannan_rissanen(endog, ar_order=1, ma_order=1, demean=True, initial_ar_order=22, unbiased=False, fixed_params=fixed_params)
    assert_allclose(hr.ar_params, [0.69607715], atol=0.0001)
    assert_allclose(hr.ma_params, [0.3787969217], atol=0.0001)
    (u, v) = arma_innovations(endog - endog.mean(), hr.ar_params, hr.ma_params, sigma2=1)
    tmp = u / v ** 0.5
    assert_allclose(np.inner(tmp, tmp) / len(u), 0.4773580109, atol=0.0001)

def test_unbiased_error_with_fixed_params():
    if False:
        for i in range(10):
            print('nop')
    endog = np.random.normal(size=1000)
    msg = 'Third step of Hannan-Rissanen estimation to remove parameter bias is not yet implemented for the case with fixed parameters.'
    with pytest.raises(NotImplementedError, match=msg):
        hannan_rissanen(endog, ar_order=1, ma_order=1, unbiased=True, fixed_params={'ar.L1': 0})

def test_set_default_unbiased_with_fixed_params():
    if False:
        print('Hello World!')
    endog = np.random.normal(size=1000)
    (p_1, other_results_2) = hannan_rissanen(endog, ar_order=1, ma_order=1, unbiased=None, fixed_params={'ar.L1': 0.69607715})
    (p_2, other_results_1) = hannan_rissanen(endog, ar_order=1, ma_order=1, unbiased=False, fixed_params={'ar.L1': 0.69607715})
    np.testing.assert_array_equal(p_1.ar_params, p_2.ar_params)
    np.testing.assert_array_equal(p_1.ma_params, p_2.ma_params)
    assert p_1.sigma2 == p_2.sigma2
    np.testing.assert_array_equal(other_results_1.resid, other_results_2.resid)