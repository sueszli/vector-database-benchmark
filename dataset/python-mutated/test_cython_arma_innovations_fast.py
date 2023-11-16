"""
Tests for fast version of ARMA innovations algorithm
"""
import numpy as np
import pytest
from numpy.testing import assert_equal, assert_allclose
from statsmodels.tsa.arima_process import arma_acovf
from statsmodels.tsa.innovations import _arma_innovations, arma_innovations
from statsmodels.tsa.statespace.sarimax import SARIMAX

def test_brockwell_davis_ex533():
    if False:
        i = 10
        return i + 15
    nobs = 10
    ar_params = np.array([0.2])
    ma_params = np.array([0.4])
    sigma2 = 8.92
    p = len(ar_params)
    q = len(ma_params)
    m = max(p, q)
    ar = np.r_[1, -ar_params]
    ma = np.r_[1, ma_params]
    arma_process_acovf = arma_acovf(ar, ma, nobs=nobs, sigma2=sigma2)
    unconditional_variance = sigma2 * (1 + 2 * ar_params[0] * ma_params[0] + ma_params[0] ** 2) / (1 - ar_params[0] ** 2)
    assert_allclose(arma_process_acovf[0], unconditional_variance)
    arma_process_acovf /= sigma2
    unconditional_variance /= sigma2
    transformed_acovf = _arma_innovations.darma_transformed_acovf_fast(ar, ma, arma_process_acovf)
    (acovf, acovf2) = (np.array(arr) for arr in transformed_acovf)
    assert_equal(acovf2.shape, (nobs - m,))
    assert_allclose(acovf2[0], 1 + ma_params[0] ** 2)
    assert_allclose(acovf2[1], ma_params[0])
    assert_allclose(acovf2[2:], 0)
    assert_equal(acovf.shape, (m * 2, m * 2))
    ix = np.diag_indices_from(acovf)
    ix_lower = (ix[0][:-1] + 1, ix[1][:-1])
    assert_allclose(acovf[ix][:m], unconditional_variance)
    assert_allclose(acovf[ix_lower][:m], ma_params[0])
    out = _arma_innovations.darma_innovations_algo_fast(nobs, ar_params, ma_params, acovf, acovf2)
    theta = np.array(out[0])
    v = np.array(out[1])
    desired_v = np.zeros(nobs)
    desired_v[0] = unconditional_variance
    for i in range(1, nobs):
        desired_v[i] = 1 + (1 - 1 / desired_v[i - 1]) * ma_params[0] ** 2
    assert_allclose(v, desired_v)
    assert_equal(theta.shape, (nobs, m + 1))
    desired_theta = np.zeros(nobs)
    for i in range(1, nobs):
        desired_theta[i] = ma_params[0] / desired_v[i - 1]
    assert_allclose(theta[:, 0], desired_theta)
    assert_allclose(theta[:, 1:], 0)
    endog = np.array([-1.1, 0.514, 0.116, -0.845, 0.872, -0.467, -0.977, -1.699, -1.228, -1.093])
    u = _arma_innovations.darma_innovations_filter(endog, ar_params, ma_params, theta)
    desired_hat = np.array([0, -0.54, 0.5068, -0.1321, -0.4539, 0.7046, -0.562, -0.3614, -0.8748, -0.3869])
    desired_u = endog - desired_hat
    assert_allclose(u, desired_u, atol=0.0001)

def test_brockwell_davis_ex534():
    if False:
        return 10
    nobs = 10
    ar_params = np.array([1, -0.24])
    ma_params = np.array([0.4, 0.2, 0.1])
    sigma2 = 1
    p = len(ar_params)
    q = len(ma_params)
    m = max(p, q)
    ar = np.r_[1, -ar_params]
    ma = np.r_[1, ma_params]
    arma_process_acovf = arma_acovf(ar, ma, nobs=nobs, sigma2=sigma2)
    assert_allclose(arma_process_acovf[:3], [7.17133, 6.44139, 5.06027], atol=1e-05)
    transformed_acovf = _arma_innovations.darma_transformed_acovf_fast(ar, ma, arma_process_acovf)
    (acovf, acovf2) = (np.array(arr) for arr in transformed_acovf)
    assert_equal(acovf.shape, (m * 2, m * 2))
    ix = np.diag_indices_from(acovf)
    ix_lower1 = (ix[0][:-1] + 1, ix[1][:-1])
    ix_lower2 = (ix[0][:-2] + 2, ix[1][:-2])
    ix_lower3 = (ix[0][:-3] + 3, ix[1][:-3])
    ix_lower4 = (ix[0][:-4] + 4, ix[1][:-4])
    assert_allclose(acovf[ix][:m], 7.17133, atol=1e-05)
    desired = [6.44139, 6.44139, 0.816]
    assert_allclose(acovf[ix_lower1][:m], desired, atol=1e-05)
    assert_allclose(acovf[ix_lower2][0], 5.06027, atol=1e-05)
    assert_allclose(acovf[ix_lower2][1:m], 0.34, atol=1e-05)
    assert_allclose(acovf[ix_lower3][:m], 0.1, atol=1e-05)
    assert_allclose(acovf[ix_lower4][:m], 0, atol=1e-05)
    assert_equal(acovf2.shape, (nobs - m,))
    assert_allclose(acovf2[:4], [1.21, 0.5, 0.24, 0.1])
    assert_allclose(acovf2[4:], 0)
    out = _arma_innovations.darma_innovations_algo_fast(nobs, ar_params, ma_params, acovf, acovf2)
    theta = np.array(out[0])
    v = np.array(out[1])
    desired_v = [7.1713, 1.3856, 1.0057, 1.0019, 1.0016, 1.0005, 1.0, 1.0, 1.0, 1.0]
    assert_allclose(v, desired_v, atol=0.0001)
    assert_equal(theta.shape, (nobs, m + 1))
    desired_theta = np.array([[0, 0.8982, 1.3685, 0.4008, 0.3998, 0.3992, 0.4, 0.4, 0.4, 0.4], [0, 0, 0.7056, 0.1806, 0.202, 0.1995, 0.1997, 0.2, 0.2, 0.2], [0, 0, 0, 0.0139, 0.0722, 0.0994, 0.0998, 0.0998, 0.0999, 0.1]]).T
    assert_allclose(theta[:, :m], desired_theta, atol=0.0001)
    assert_allclose(theta[:, m:], 0)
    endog = np.array([1.704, 0.527, 1.041, 0.942, 0.555, -1.002, -0.585, 0.01, -0.638, 0.525])
    u = _arma_innovations.darma_innovations_filter(endog, ar_params, ma_params, theta)
    desired_hat = np.array([0, 1.5305, -0.171, 1.2428, 0.7443, 0.3138, -1.7293, -0.1688, 0.3193, -0.8731])
    desired_u = endog - desired_hat
    assert_allclose(u, desired_u, atol=0.0001)

@pytest.mark.parametrize('ar_params,ma_params,sigma2', [(np.array([]), np.array([]), 1), (np.array([0.0]), np.array([0.0]), 1), (np.array([0.9]), np.array([]), 1), (np.array([]), np.array([0.9]), 1), (np.array([0.2, -0.4, 0.1, 0.1]), np.array([0.5, 0.1]), 1.123), (np.array([0.5, 0.1]), np.array([0.2, -0.4, 0.1, 0.1]), 1.123)])
def test_innovations_algo_filter_kalman_filter(ar_params, ma_params, sigma2):
    if False:
        while True:
            i = 10
    ar = np.r_[1, -ar_params]
    ma = np.r_[1, ma_params]
    endog = np.random.normal(size=10)
    nobs = len(endog)
    arma_process_acovf = arma_acovf(ar, ma, nobs=nobs, sigma2=sigma2)
    transformed_acov = _arma_innovations.darma_transformed_acovf_fast(ar, ma, arma_process_acovf / sigma2)
    (acovf, acovf2) = (np.array(mv) for mv in transformed_acov)
    (theta, r) = _arma_innovations.darma_innovations_algo_fast(nobs, ar_params, ma_params, acovf, acovf2)
    u = _arma_innovations.darma_innovations_filter(endog, ar_params, ma_params, theta)
    v = np.array(r) * sigma2
    u = np.array(u)
    llf_obs = -0.5 * u ** 2 / v - 0.5 * np.log(2 * np.pi * v)
    mod = SARIMAX(endog, order=(len(ar_params), 0, len(ma_params)))
    res = mod.filter(np.r_[ar_params, ma_params, sigma2])
    assert_allclose(u, res.forecasts_error[0])
    assert_allclose(llf_obs, res.llf_obs)
    llf_obs2 = _arma_innovations.darma_loglikeobs_fast(endog, ar_params, ma_params, sigma2)
    assert_allclose(llf_obs2, res.llf_obs)

@pytest.mark.parametrize('ar_params,ma_params,sigma2', [(np.array([]), np.array([]), 1), (np.array([0.0]), np.array([0.0]), 1), (np.array([0.9]), np.array([]), 1), (np.array([]), np.array([0.9]), 1), (np.array([0.2, -0.4, 0.1, 0.1]), np.array([0.5, 0.1]), 1.123), (np.array([0.5, 0.1]), np.array([0.2, -0.4, 0.1, 0.1]), 1.123)])
def test_innovations_algo_direct_filter_kalman_filter(ar_params, ma_params, sigma2):
    if False:
        while True:
            i = 10
    endog = np.random.normal(size=10)
    (u, r) = arma_innovations.arma_innovations(endog, ar_params, ma_params, sigma2)
    v = np.array(r) * sigma2
    u = np.array(u)
    llf_obs = -0.5 * u ** 2 / v - 0.5 * np.log(2 * np.pi * v)
    mod = SARIMAX(endog, order=(len(ar_params), 0, len(ma_params)))
    res = mod.filter(np.r_[ar_params, ma_params, sigma2])
    assert_allclose(u, res.forecasts_error[0])
    assert_allclose(llf_obs, res.llf_obs)
    llf_obs2 = _arma_innovations.darma_loglikeobs_fast(endog, ar_params, ma_params, sigma2)
    assert_allclose(llf_obs2, res.llf_obs)

@pytest.mark.parametrize('ar_params,diff,ma_params,sigma2', [(np.array([]), 1, np.array([]), 1), (np.array([0.0]), 1, np.array([0.0]), 1), (np.array([0.9]), 1, np.array([]), 1), (np.array([]), 1, np.array([0.9]), 1), (np.array([0.2, -0.4, 0.1, 0.1]), 1, np.array([0.5, 0.1]), 1.123), (np.array([0.5, 0.1]), 1, np.array([0.2, -0.4, 0.1, 0.1]), 1.123), (np.array([0.5, 0.1]), 2, np.array([0.2, -0.4, 0.1, 0.1]), 1.123)])
def test_integrated_process(ar_params, diff, ma_params, sigma2):
    if False:
        while True:
            i = 10
    nobs = 100
    endog = np.cumsum(np.random.normal(size=nobs))
    llf_obs = arma_innovations.arma_loglikeobs(np.diff(endog, diff), ar_params, ma_params, sigma2)
    mod = SARIMAX(endog, order=(len(ar_params), diff, len(ma_params)), simple_differencing=True)
    res = mod.filter(np.r_[ar_params, ma_params, sigma2])
    assert_allclose(llf_obs, res.llf_obs)

@pytest.mark.parametrize('ar_params,ma_params,sigma2', [(np.array([]), np.array([]), 1), (np.array([0.0]), np.array([0.0]), 1), (np.array([0.9]), np.array([]), 1), (np.array([]), np.array([0.9]), 1), (np.array([0.2, -0.4, 0.1, 0.1]), np.array([0.5, 0.1]), 1.123), (np.array([0.5, 0.1]), np.array([0.2, -0.4, 0.1, 0.1]), 1.123)])
def test_regression_with_arma_errors(ar_params, ma_params, sigma2):
    if False:
        print('Hello World!')
    nobs = 100
    eps = np.random.normal(nobs)
    exog = np.c_[np.ones(nobs), np.random.uniform(size=nobs)]
    beta = [5, -0.2]
    endog = np.dot(exog, beta) + eps
    beta_hat = np.squeeze(np.linalg.pinv(exog).dot(endog))
    demeaned = endog - np.dot(exog, beta_hat)
    llf_obs = arma_innovations.arma_loglikeobs(demeaned, ar_params, ma_params, sigma2)
    mod = SARIMAX(endog, exog=exog, order=(len(ar_params), 0, len(ma_params)))
    res = mod.filter(np.r_[beta_hat, ar_params, ma_params, sigma2])
    assert_allclose(llf_obs, res.llf_obs)