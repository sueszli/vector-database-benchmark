import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_equal, assert_warns, assert_raises
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import lake, oshorts
from statsmodels.tsa.arima.estimators.gls import gls

@pytest.mark.low_precision('Test against Example 6.6.1 in Brockwell and Davis (2016)')
def test_brockwell_davis_example_661():
    if False:
        print('Hello World!')
    endog = oshorts.copy()
    exog = np.ones_like(endog)
    (res, _) = gls(endog, exog, order=(0, 0, 1), max_iter=1, tolerance=1)
    assert_allclose(res.exog_params, -4.745, atol=0.001)
    assert_allclose(res.ma_params, -0.818, atol=0.001)
    assert_allclose(res.sigma2, 2041, atol=1)
    (res, _) = gls(endog, exog, order=(0, 0, 1))
    assert_allclose(res.exog_params, -4.78, atol=0.001)
    assert_allclose(res.ma_params, -0.848, atol=0.001)

@pytest.mark.low_precision('Test against Example 6.6.2 in Brockwell and Davis (2016)')
def test_brockwell_davis_example_662():
    if False:
        print('Hello World!')
    endog = lake.copy()
    exog = np.c_[np.ones_like(endog), np.arange(1, len(endog) + 1) * 1.0]
    (res, _) = gls(endog, exog, order=(2, 0, 0))
    assert_allclose(res.exog_params, [10.091, -0.0216], atol=0.001)
    assert_allclose(res.ar_params, [1.005, -0.291], atol=0.001)
    assert_allclose(res.sigma2, 0.4571, atol=0.001)

def test_integrated():
    if False:
        return 10
    endog1 = lake.copy()
    exog1 = np.c_[np.ones_like(endog1), np.arange(1, len(endog1) + 1) * 1.0]
    endog2 = np.r_[0, np.cumsum(endog1)]
    exog2 = np.c_[[0, 0], np.cumsum(exog1, axis=0).T].T
    (p1, _) = gls(endog1, exog1, order=(1, 0, 0))
    with assert_warns(UserWarning):
        (p2, _) = gls(endog2, exog2, order=(1, 1, 0))
    assert_allclose(p1.params, p2.params)

def test_integrated_invalid():
    if False:
        i = 10
        return i + 15
    endog = lake.copy()
    exog = np.arange(1, len(endog) + 1) * 1.0
    assert_raises(ValueError, gls, endog, exog, order=(1, 1, 0), include_constant=True)

def test_results():
    if False:
        print('Hello World!')
    endog = lake.copy()
    exog = np.c_[np.ones_like(endog), np.arange(1, len(endog) + 1) * 1.0]
    (p, res) = gls(endog, exog, order=(1, 0, 0))
    assert_('params' in res)
    assert_('converged' in res)
    assert_('differences' in res)
    assert_('iterations' in res)
    assert_('arma_estimator' in res)
    assert_('arma_results' in res)
    assert_(res.converged)
    assert_(res.iterations > 0)
    assert_equal(res.arma_estimator, 'innovations_mle')
    assert_equal(len(res.params), res.iterations + 1)
    assert_equal(len(res.differences), res.iterations + 1)
    assert_equal(len(res.arma_results), res.iterations + 1)
    assert_equal(res.params[-1], p)

def test_iterations():
    if False:
        return 10
    endog = lake.copy()
    exog = np.c_[np.ones_like(endog), np.arange(1, len(endog) + 1) * 1.0]
    (_, res) = gls(endog, exog, order=(1, 0, 0), n_iter=1)
    assert_equal(res.iterations, 1)
    assert_equal(res.converged, None)

def test_misc():
    if False:
        while True:
            i = 10
    endog = lake.copy()
    exog = np.c_[np.ones_like(endog), np.arange(1, len(endog) + 1) * 1.0]
    assert_warns(UserWarning, gls, endog, exog, order=(2, 0, 0), max_iter=0)

@pytest.mark.todo('Low priority: test full GLS against another package')
@pytest.mark.smoke
def test_alternate_arma_estimators_valid():
    if False:
        while True:
            i = 10
    endog = lake.copy()
    exog = np.c_[np.ones_like(endog), np.arange(1, len(endog) + 1) * 1.0]
    (_, res_yw) = gls(endog, exog=exog, order=(1, 0, 0), arma_estimator='yule_walker', n_iter=1)
    assert_equal(res_yw.arma_estimator, 'yule_walker')
    (_, res_b) = gls(endog, exog=exog, order=(1, 0, 0), arma_estimator='burg', n_iter=1)
    assert_equal(res_b.arma_estimator, 'burg')
    (_, res_i) = gls(endog, exog=exog, order=(0, 0, 1), arma_estimator='innovations', n_iter=1)
    assert_equal(res_i.arma_estimator, 'innovations')
    (_, res_hr) = gls(endog, exog=exog, order=(1, 0, 1), arma_estimator='hannan_rissanen', n_iter=1)
    assert_equal(res_hr.arma_estimator, 'hannan_rissanen')
    (_, res_ss) = gls(endog, exog=exog, order=(1, 0, 1), arma_estimator='statespace', n_iter=1)
    assert_equal(res_ss.arma_estimator, 'statespace')
    (_, res_imle) = gls(endog, exog=exog, order=(1, 0, 1), n_iter=1)
    assert_equal(res_imle.arma_estimator, 'innovations_mle')

def test_alternate_arma_estimators_invalid():
    if False:
        return 10
    endog = lake.copy()
    exog = np.c_[np.ones_like(endog), np.arange(1, len(endog) + 1) * 1.0]
    assert_raises(ValueError, gls, endog, exog, order=(0, 0, 1), arma_estimator='invalid_estimator')
    assert_raises(ValueError, gls, endog, exog, order=(0, 0, 1), arma_estimator='yule_walker')
    assert_raises(ValueError, gls, endog, exog, order=(0, 0, 0), seasonal_order=(1, 0, 0, 4), arma_estimator='yule_walker')
    assert_raises(ValueError, gls, endog, exog, order=([0, 1], 0, 0), arma_estimator='yule_walker')
    assert_raises(ValueError, gls, endog, exog, order=(0, 0, 1), arma_estimator='burg')
    assert_raises(ValueError, gls, endog, exog, order=(0, 0, 0), seasonal_order=(1, 0, 0, 4), arma_estimator='burg')
    assert_raises(ValueError, gls, endog, exog, order=([0, 1], 0, 0), arma_estimator='burg')
    assert_raises(ValueError, gls, endog, exog, order=(1, 0, 0), arma_estimator='innovations')
    assert_raises(ValueError, gls, endog, exog, order=(0, 0, 0), seasonal_order=(0, 0, 1, 4), arma_estimator='innovations')
    assert_raises(ValueError, gls, endog, exog, order=(0, 0, [0, 1]), arma_estimator='innovations')
    assert_raises(ValueError, gls, endog, exog, order=(0, 0, 0), seasonal_order=(0, 0, 1, 4), arma_estimator='hannan_rissanen')

def test_arma_kwargs():
    if False:
        i = 10
        return i + 15
    endog = lake.copy()
    exog = np.c_[np.ones_like(endog), np.arange(1, len(endog) + 1) * 1.0]
    (_, res1_imle) = gls(endog, exog=exog, order=(1, 0, 1), n_iter=1)
    assert_equal(res1_imle.arma_estimator_kwargs, {})
    assert_equal(res1_imle.arma_results[1].minimize_results.message, 'Optimization terminated successfully.')
    arma_estimator_kwargs = {'minimize_kwargs': {'method': 'L-BFGS-B'}}
    (_, res2_imle) = gls(endog, exog=exog, order=(1, 0, 1), n_iter=1, arma_estimator_kwargs=arma_estimator_kwargs)
    assert_equal(res2_imle.arma_estimator_kwargs, arma_estimator_kwargs)
    msg = res2_imle.arma_results[1].minimize_results.message
    if isinstance(msg, bytes):
        msg = msg.decode('utf-8')
    assert_equal(msg, 'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH')