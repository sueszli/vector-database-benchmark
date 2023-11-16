"""
Tests for the generic MLEModel

Author: Chad Fulton
License: Simplified-BSD
"""
import os
import re
import warnings
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace import sarimax, varmax, kalman_filter, kalman_smoother
from statsmodels.tsa.statespace.mlemodel import MLEModel, MLEResultsWrapper
from statsmodels.datasets import nile
from numpy.testing import assert_, assert_almost_equal, assert_equal, assert_allclose, assert_raises
from statsmodels.tsa.statespace.tests.results import results_sarimax, results_var_misc
current_path = os.path.dirname(os.path.abspath(__file__))
kwargs = {'k_states': 1, 'design': [[1]], 'transition': [[1]], 'selection': [[1]], 'state_cov': [[1]], 'initialization': 'approximate_diffuse'}

def get_dummy_mod(fit=True, pandas=False):
    if False:
        for i in range(10):
            print('nop')
    endog = np.arange(100) * 1.0
    exog = 2 * endog
    if pandas:
        index = pd.date_range('1960-01-01', periods=100, freq='MS')
        endog = pd.Series(endog, index=index)
        exog = pd.Series(exog, index=index)
    mod = sarimax.SARIMAX(endog, exog=exog, order=(0, 0, 0), time_varying_regression=True, mle_regression=False, use_exact_diffuse=True)
    if fit:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = mod.fit(disp=-1)
    else:
        res = None
    return (mod, res)

def test_init_matrices_time_invariant():
    if False:
        return 10
    k_endog = 2
    k_states = 3
    k_posdef = 1
    endog = np.zeros((10, 2))
    obs_intercept = np.arange(k_endog) * 1.0
    design = np.reshape(np.arange(k_endog * k_states) * 1.0, (k_endog, k_states))
    obs_cov = np.reshape(np.arange(k_endog ** 2) * 1.0, (k_endog, k_endog))
    state_intercept = np.arange(k_states) * 1.0
    transition = np.reshape(np.arange(k_states ** 2) * 1.0, (k_states, k_states))
    selection = np.reshape(np.arange(k_states * k_posdef) * 1.0, (k_states, k_posdef))
    state_cov = np.reshape(np.arange(k_posdef ** 2) * 1.0, (k_posdef, k_posdef))
    mod = MLEModel(endog, k_states=k_states, k_posdef=k_posdef, obs_intercept=obs_intercept, design=design, obs_cov=obs_cov, state_intercept=state_intercept, transition=transition, selection=selection, state_cov=state_cov)
    assert_allclose(mod['obs_intercept'], obs_intercept)
    assert_allclose(mod['design'], design)
    assert_allclose(mod['obs_cov'], obs_cov)
    assert_allclose(mod['state_intercept'], state_intercept)
    assert_allclose(mod['transition'], transition)
    assert_allclose(mod['selection'], selection)
    assert_allclose(mod['state_cov'], state_cov)

def test_init_matrices_time_varying():
    if False:
        for i in range(10):
            print('nop')
    nobs = 10
    k_endog = 2
    k_states = 3
    k_posdef = 1
    endog = np.zeros((10, 2))
    obs_intercept = np.reshape(np.arange(k_endog * nobs) * 1.0, (k_endog, nobs))
    design = np.reshape(np.arange(k_endog * k_states * nobs) * 1.0, (k_endog, k_states, nobs))
    obs_cov = np.reshape(np.arange(k_endog ** 2 * nobs) * 1.0, (k_endog, k_endog, nobs))
    state_intercept = np.reshape(np.arange(k_states * nobs) * 1.0, (k_states, nobs))
    transition = np.reshape(np.arange(k_states ** 2 * nobs) * 1.0, (k_states, k_states, nobs))
    selection = np.reshape(np.arange(k_states * k_posdef * nobs) * 1.0, (k_states, k_posdef, nobs))
    state_cov = np.reshape(np.arange(k_posdef ** 2 * nobs) * 1.0, (k_posdef, k_posdef, nobs))
    mod = MLEModel(endog, k_states=k_states, k_posdef=k_posdef, obs_intercept=obs_intercept, design=design, obs_cov=obs_cov, state_intercept=state_intercept, transition=transition, selection=selection, state_cov=state_cov)
    assert_allclose(mod['obs_intercept'], obs_intercept)
    assert_allclose(mod['design'], design)
    assert_allclose(mod['obs_cov'], obs_cov)
    assert_allclose(mod['state_intercept'], state_intercept)
    assert_allclose(mod['transition'], transition)
    assert_allclose(mod['selection'], selection)
    assert_allclose(mod['state_cov'], state_cov)

def test_wrapping():
    if False:
        for i in range(10):
            print('nop')
    (mod, _) = get_dummy_mod(fit=False)
    assert_equal(mod['design', 0, 0], 2.0 * np.arange(100))
    mod['design', 0, 0, :] = 2
    assert_equal(mod.ssm['design', 0, 0, :], 2)
    assert_equal(mod.ssm['design'].shape, (1, 1, 100))
    mod['design'] = [[3.0]]
    assert_equal(mod.ssm['design', 0, 0], 3.0)
    assert_equal(mod.ssm['design'].shape, (1, 1))
    assert_equal(mod.loglikelihood_burn, 0)
    mod.loglikelihood_burn = 1
    assert_equal(mod.ssm.loglikelihood_burn, 1)
    assert_equal(mod.tolerance, mod.ssm.tolerance)
    mod.tolerance = 0.123
    assert_equal(mod.ssm.tolerance, 0.123)
    assert_equal(mod.initial_variance, 10000000000.0)
    mod.initial_variance = 1000000000000.0
    assert_equal(mod.ssm.initial_variance, 1000000000000.0)
    assert_equal(isinstance(mod.initialization, object), True)
    mod.initialize_default()
    mod.initialize_approximate_diffuse(100000.0)
    assert_equal(mod.initialization.initialization_type, 'approximate_diffuse')
    assert_equal(mod.initialization.approximate_diffuse_variance, 100000.0)
    mod.initialize_known([5.0], [[40]])
    assert_equal(mod.initialization.initialization_type, 'known')
    assert_equal(mod.initialization.constant, [5.0])
    assert_equal(mod.initialization.stationary_cov, [[40]])
    mod.initialize_stationary()
    assert_equal(mod.initialization.initialization_type, 'stationary')
    assert_equal(mod.ssm.filter_method, kalman_filter.FILTER_CONVENTIONAL)
    assert_equal(mod.ssm.stability_method, kalman_filter.STABILITY_FORCE_SYMMETRY)
    assert_equal(mod.ssm.conserve_memory, kalman_filter.MEMORY_STORE_ALL)
    assert_equal(mod.ssm.smoother_output, kalman_smoother.SMOOTHER_ALL)
    mod.ssm._initialize_filter()
    kf = mod.ssm._kalman_filter
    assert_equal(kf.filter_method, kalman_filter.FILTER_CONVENTIONAL)
    assert_equal(kf.stability_method, kalman_filter.STABILITY_FORCE_SYMMETRY)
    assert_equal(kf.conserve_memory, kalman_filter.MEMORY_STORE_ALL)
    mod.set_filter_method(100)
    mod.set_stability_method(101)
    mod.set_conserve_memory(102)
    mod.set_smoother_output(103)
    assert_equal(mod.ssm.filter_method, 100)
    assert_equal(mod.ssm.stability_method, 101)
    assert_equal(mod.ssm.conserve_memory, 102)
    assert_equal(mod.ssm.smoother_output, 103)
    assert_equal(kf.filter_method, kalman_filter.FILTER_CONVENTIONAL)
    assert_equal(kf.stability_method, kalman_filter.STABILITY_FORCE_SYMMETRY)
    assert_equal(kf.conserve_memory, kalman_filter.MEMORY_STORE_ALL)
    mod.set_filter_method(1)
    mod.ssm._initialize_filter()
    kf = mod.ssm._kalman_filter
    assert_equal(kf.filter_method, 1)
    assert_equal(kf.stability_method, 101)
    assert_equal(kf.conserve_memory, 102)

def test_fit_misc():
    if False:
        return 10
    true = results_sarimax.wpi1_stationary
    endog = np.diff(true['data'])[1:]
    mod = sarimax.SARIMAX(endog, order=(1, 0, 1), trend='c')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res1 = mod.fit(method='ncg', disp=0, optim_hessian='opg', optim_complex_step=False)
        res2 = mod.fit(method='ncg', disp=0, optim_hessian='oim', optim_complex_step=False)
    assert_allclose(res1.llf, res2.llf, rtol=0.01)
    (mod, _) = get_dummy_mod(fit=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res_params = mod.fit(disp=-1, return_params=True)
    assert_almost_equal(res_params, [0, 0], 5)

@pytest.mark.smoke
def test_score_misc():
    if False:
        print('Hello World!')
    (mod, res) = get_dummy_mod()
    mod.score(res.params)

def test_from_formula():
    if False:
        i = 10
        return i + 15
    assert_raises(NotImplementedError, lambda : MLEModel.from_formula(1, 2, 3))

def test_score_analytic_ar1():
    if False:
        for i in range(10):
            print('nop')
    mod = sarimax.SARIMAX([1, 0.5], order=(1, 0, 0))

    def partial_phi(phi, sigma2):
        if False:
            return 10
        return -0.5 * (phi ** 2 + 2 * phi * sigma2 - 1) / (sigma2 * (1 - phi ** 2))

    def partial_sigma2(phi, sigma2):
        if False:
            return 10
        return -0.5 * (2 * sigma2 + phi - 1.25) / sigma2 ** 2
    params = np.r_[0.0, 2]
    analytic_score = np.r_[partial_phi(params[0], params[1]), partial_sigma2(params[0], params[1])]
    approx_cs = mod.score(params, transformed=True, approx_complex_step=True)
    assert_allclose(approx_cs, analytic_score)
    approx_fd = mod.score(params, transformed=True, approx_complex_step=False)
    assert_allclose(approx_fd, analytic_score, atol=1e-05)
    approx_fd_centered = mod.score(params, transformed=True, approx_complex_step=False, approx_centered=True)
    assert_allclose(approx_fd, analytic_score, atol=1e-05)
    harvey_cs = mod.score(params, transformed=True, method='harvey', approx_complex_step=True)
    assert_allclose(harvey_cs, analytic_score)
    harvey_fd = mod.score(params, transformed=True, method='harvey', approx_complex_step=False)
    assert_allclose(harvey_fd, analytic_score, atol=1e-05)
    harvey_fd_centered = mod.score(params, transformed=True, method='harvey', approx_complex_step=False, approx_centered=True)
    assert_allclose(harvey_fd_centered, analytic_score, atol=1e-05)

    def partial_transform_phi(phi):
        if False:
            print('Hello World!')
        return -1.0 / (1 + phi ** 2) ** (3.0 / 2)

    def partial_transform_sigma2(sigma2):
        if False:
            while True:
                i = 10
        return 2.0 * sigma2
    uparams = mod.untransform_params(params)
    analytic_score = np.dot(np.diag(np.r_[partial_transform_phi(uparams[0]), partial_transform_sigma2(uparams[1])]), np.r_[partial_phi(params[0], params[1]), partial_sigma2(params[0], params[1])])
    approx_cs = mod.score(uparams, transformed=False, approx_complex_step=True)
    assert_allclose(approx_cs, analytic_score)
    approx_fd = mod.score(uparams, transformed=False, approx_complex_step=False)
    assert_allclose(approx_fd, analytic_score, atol=1e-05)
    approx_fd_centered = mod.score(uparams, transformed=False, approx_complex_step=False, approx_centered=True)
    assert_allclose(approx_fd_centered, analytic_score, atol=1e-05)
    harvey_cs = mod.score(uparams, transformed=False, method='harvey', approx_complex_step=True)
    assert_allclose(harvey_cs, analytic_score)
    harvey_fd = mod.score(uparams, transformed=False, method='harvey', approx_complex_step=False)
    assert_allclose(harvey_fd, analytic_score, atol=1e-05)
    harvey_fd_centered = mod.score(uparams, transformed=False, method='harvey', approx_complex_step=False, approx_centered=True)
    assert_allclose(harvey_fd_centered, analytic_score, atol=1e-05)
    params = np.r_[0.5, 1.0]

    def hessian(phi, sigma2):
        if False:
            i = 10
            return i + 15
        hessian = np.zeros((2, 2))
        hessian[0, 0] = (-phi ** 2 - 1) / (phi ** 2 - 1) ** 2
        hessian[1, 0] = hessian[0, 1] = -1 / (2 * sigma2 ** 2)
        hessian[1, 1] = (sigma2 + phi - 1.25) / sigma2 ** 3
        return hessian
    analytic_hessian = hessian(params[0], params[1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert_allclose(mod._hessian_complex_step(params) * 2, analytic_hessian, atol=0.1)
        assert_allclose(mod._hessian_finite_difference(params) * 2, analytic_hessian, atol=0.1)

def test_cov_params():
    if False:
        for i in range(10):
            print('nop')
    (mod, res) = get_dummy_mod()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = mod.fit(res.params, disp=-1, cov_type='none')
        assert_equal(res.cov_kwds['description'], 'Covariance matrix not calculated.')
        res = mod.fit(res.params, disp=-1, cov_type='approx')
        assert_equal(res.cov_type, 'approx')
        assert_equal(res.cov_kwds['description'], 'Covariance matrix calculated using numerical (complex-step) differentiation.')
        res = mod.fit(res.params, disp=-1, cov_type='oim')
        assert_equal(res.cov_type, 'oim')
        assert_equal(res.cov_kwds['description'], 'Covariance matrix calculated using the observed information matrix (complex-step) described in Harvey (1989).')
        res = mod.fit(res.params, disp=-1, cov_type='opg')
        assert_equal(res.cov_type, 'opg')
        assert_equal(res.cov_kwds['description'], 'Covariance matrix calculated using the outer product of gradients (complex-step).')
        res = mod.fit(res.params, disp=-1, cov_type='robust')
        assert_equal(res.cov_type, 'robust')
        assert_equal(res.cov_kwds['description'], 'Quasi-maximum likelihood covariance matrix used for robustness to some misspecifications; calculated using the observed information matrix (complex-step) described in Harvey (1989).')
        res = mod.fit(res.params, disp=-1, cov_type='robust_oim')
        assert_equal(res.cov_type, 'robust_oim')
        assert_equal(res.cov_kwds['description'], 'Quasi-maximum likelihood covariance matrix used for robustness to some misspecifications; calculated using the observed information matrix (complex-step) described in Harvey (1989).')
        res = mod.fit(res.params, disp=-1, cov_type='robust_approx')
        assert_equal(res.cov_type, 'robust_approx')
        assert_equal(res.cov_kwds['description'], 'Quasi-maximum likelihood covariance matrix used for robustness to some misspecifications; calculated using numerical (complex-step) differentiation.')
        with pytest.raises(NotImplementedError):
            mod.fit(res.params, disp=-1, cov_type='invalid_cov_type')

def test_transform():
    if False:
        i = 10
        return i + 15
    mod = MLEModel([1, 2], **kwargs)
    assert_allclose(mod.transform_params([2, 3]), [2, 3])
    assert_allclose(mod.untransform_params([2, 3]), [2, 3])
    mod.filter([], transformed=False)
    mod.update([], transformed=False)
    mod.loglike([], transformed=False)
    mod.loglikeobs([], transformed=False)
    (mod, _) = get_dummy_mod(fit=False)
    assert_allclose(mod.transform_params([2, 3]), [4, 9])
    assert_allclose(mod.untransform_params([4, 9]), [2, 3])
    res = mod.filter([2, 3], transformed=True)
    assert_allclose(res.params, [2, 3])
    res = mod.filter([2, 3], transformed=False)
    assert_allclose(res.params, [4, 9])

def test_filter():
    if False:
        for i in range(10):
            print('nop')
    endog = np.array([1.0, 2.0])
    mod = MLEModel(endog, **kwargs)
    res = mod.filter([], return_ssm=True)
    assert_equal(isinstance(res, kalman_filter.FilterResults), True)
    res = mod.filter([])
    assert_equal(isinstance(res, MLEResultsWrapper), True)
    assert_equal(res.cov_type, 'opg')
    res = mod.filter([], cov_type='oim')
    assert_equal(isinstance(res, MLEResultsWrapper), True)
    assert_equal(res.cov_type, 'oim')

def test_params():
    if False:
        for i in range(10):
            print('nop')
    mod = MLEModel([1, 2], **kwargs)
    assert_raises(NotImplementedError, lambda : mod.start_params)
    assert_equal(mod.param_names, [])
    mod._start_params = [1]
    mod._param_names = ['a']
    assert_equal(mod.start_params, [1])
    assert_equal(mod.param_names, ['a'])

def check_results(pandas):
    if False:
        return 10
    (mod, res) = get_dummy_mod(pandas=pandas)
    assert_almost_equal(res.fittedvalues[2:], mod.endog[2:].squeeze())
    assert_almost_equal(res.resid[2:], np.zeros(mod.nobs - 2))
    assert_equal(res.loglikelihood_burn, 0)

def test_results(pandas=False):
    if False:
        for i in range(10):
            print('nop')
    check_results(pandas=False)
    check_results(pandas=True)

def test_predict():
    if False:
        return 10
    dates = pd.date_range(start='1980-01-01', end='1981-01-01', freq='YS')
    endog = pd.Series([1, 2], index=dates)
    mod = MLEModel(endog, **kwargs)
    res = mod.filter([])
    predict = res.predict()
    assert_equal(predict.shape, (mod.nobs,))
    assert_allclose(res.get_prediction().predicted_mean, predict)
    assert_allclose(res.predict(dynamic='1981-01-01'), res.predict())
    mod = MLEModel([1, 2], **kwargs)
    res = mod.filter([])
    assert_raises(KeyError, res.predict, dynamic='string')

def test_forecast():
    if False:
        while True:
            i = 10
    mod = MLEModel([1, 2], **kwargs)
    res = mod.filter([])
    forecast = res.forecast(steps=10)
    assert_allclose(forecast, np.ones((10,)) * 2)
    assert_allclose(res.get_forecast(steps=10).predicted_mean, forecast)
    index = pd.date_range('1960-01-01', periods=2, freq='MS')
    mod = MLEModel(pd.Series([1, 2], index=index), **kwargs)
    res = mod.filter([])
    assert_allclose(res.forecast(steps=10), np.ones((10,)) * 2)
    assert_allclose(res.forecast(steps='1960-12-01'), np.ones((10,)) * 2)
    assert_allclose(res.get_forecast(steps=10).predicted_mean, np.ones((10,)) * 2)

def test_summary():
    if False:
        i = 10
        return i + 15
    dates = pd.date_range(start='1980-01-01', end='1984-01-01', freq='YS')
    endog = pd.Series([1, 2, 3, 4, 5], index=dates)
    mod = MLEModel(endog, **kwargs)
    res = mod.filter([])
    txt = str(res.summary())
    assert_equal(re.search('Sample:\\s+01-01-1980', txt) is not None, True)
    assert_equal(re.search('\\s+- 01-01-1984', txt) is not None, True)
    assert_equal(re.search('Model:\\s+MLEModel', txt) is not None, True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res.filter_results._standardized_forecasts_error[:] = np.nan
        res.summary()
        res.filter_results._standardized_forecasts_error = 1
        res.summary()
        res.filter_results._standardized_forecasts_error = 'a'
        res.summary()

def check_endog(endog, nobs=2, k_endog=1, **kwargs):
    if False:
        print('Hello World!')
    mod = MLEModel(endog, **kwargs)
    assert_equal(mod.endog.ndim, 2)
    assert_equal(mod.endog.flags['C_CONTIGUOUS'], True)
    assert_equal(mod.endog.shape, (nobs, k_endog))
    assert_equal(mod.ssm.endog.ndim, 2)
    assert_equal(mod.ssm.endog.flags['F_CONTIGUOUS'], True)
    assert_equal(mod.ssm.endog.shape, (k_endog, nobs))
    assert_equal(mod.ssm.endog.base is mod.endog, True)
    return mod

def test_basic_endog():
    if False:
        i = 10
        return i + 15
    assert_raises(ValueError, MLEModel, endog=1, k_states=1)
    assert_raises(ValueError, MLEModel, endog='a', k_states=1)
    assert_raises(ValueError, MLEModel, endog=True, k_states=1)
    mod = MLEModel([1], **kwargs)
    res = mod.filter([])
    assert_equal(res.filter_results.endog, [[1]])
    mod = MLEModel([1.0], **kwargs)
    res = mod.filter([])
    assert_equal(res.filter_results.endog, [[1]])
    mod = MLEModel([True], **kwargs)
    res = mod.filter([])
    assert_equal(res.filter_results.endog, [[1]])
    mod = MLEModel(['a'], **kwargs)
    assert_raises(ValueError, mod.filter, [])
    endog = [1.0, 2.0]
    mod = check_endog(endog, **kwargs)
    mod.filter([])
    endog = [[1.0], [2.0]]
    mod = check_endog(endog, **kwargs)
    mod.filter([])
    endog = (1.0, 2.0)
    mod = check_endog(endog, **kwargs)
    mod.filter([])

def test_numpy_endog():
    if False:
        for i in range(10):
            print('nop')
    endog = np.array([1.0, 2.0])
    mod = MLEModel(endog, **kwargs)
    assert_equal(mod.endog.base is not mod.data.orig_endog, True)
    assert_equal(mod.endog.base is not endog, True)
    assert_equal(mod.data.orig_endog.base is not endog, True)
    endog[0] = 2
    assert_equal(mod.endog, np.r_[1, 2].reshape(2, 1))
    assert_equal(mod.data.orig_endog, endog)
    endog = np.array(1.0)
    assert_raises(TypeError, check_endog, endog, **kwargs)
    endog = np.array([1.0, 2.0])
    assert_equal(endog.ndim, 1)
    assert_equal(endog.flags['C_CONTIGUOUS'], True)
    assert_equal(endog.flags['F_CONTIGUOUS'], True)
    assert_equal(endog.shape, (2,))
    mod = check_endog(endog, **kwargs)
    mod.filter([])
    endog = np.array([1.0, 2.0]).reshape(2, 1)
    assert_equal(endog.ndim, 2)
    assert_equal(endog.flags['C_CONTIGUOUS'], True)
    assert_equal(endog.shape, (2, 1))
    mod = check_endog(endog, **kwargs)
    mod.filter([])
    endog = np.array([1.0, 2.0]).reshape(1, 2)
    assert_equal(endog.ndim, 2)
    assert_equal(endog.flags['C_CONTIGUOUS'], True)
    assert_equal(endog.shape, (1, 2))
    assert_raises(ValueError, check_endog, endog, **kwargs)
    endog = np.array([1.0, 2.0]).reshape(1, 2).transpose()
    assert_equal(endog.ndim, 2)
    assert_equal(endog.flags['F_CONTIGUOUS'], True)
    assert_equal(endog.shape, (2, 1))
    mod = check_endog(endog, **kwargs)
    mod.filter([])
    endog = np.array([1.0, 2.0]).reshape(2, 1).transpose()
    assert_equal(endog.ndim, 2)
    assert_equal(endog.flags['F_CONTIGUOUS'], True)
    assert_equal(endog.shape, (1, 2))
    assert_raises(ValueError, check_endog, endog, **kwargs)
    endog = np.array([1.0, 2.0]).reshape(2, 1, 1)
    assert_raises(ValueError, check_endog, endog, **kwargs)
    kwargs2 = {'k_states': 1, 'design': [[1], [0.0]], 'obs_cov': [[1, 0], [0, 1]], 'transition': [[1]], 'selection': [[1]], 'state_cov': [[1]], 'initialization': 'approximate_diffuse'}
    endog = np.array([[1.0, 2.0], [3.0, 4.0]])
    mod = check_endog(endog, k_endog=2, **kwargs2)
    mod.filter([])

def test_pandas_endog():
    if False:
        while True:
            i = 10
    endog = pd.Series([1.0, 2.0])
    warnings.simplefilter('always')
    dates = pd.date_range(start='1980-01-01', end='1981-01-01', freq='YS')
    endog = pd.Series([1.0, 2.0], index=dates)
    mod = check_endog(endog, **kwargs)
    mod.filter([])
    endog = pd.Series(['a', 'b'], index=dates)
    assert_raises(ValueError, check_endog, endog, **kwargs)
    endog = pd.Series([1.0, 2.0], index=dates)
    mod = check_endog(endog, **kwargs)
    mod.filter([])
    endog = pd.DataFrame({'a': [1.0, 2.0]}, index=dates)
    mod = check_endog(endog, **kwargs)
    mod.filter([])
    endog = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]}, index=dates)
    assert_raises(ValueError, check_endog, endog, **kwargs)
    endog = pd.DataFrame({'a': [1.0, 2.0]}, index=dates)
    mod = check_endog(endog, **kwargs)
    assert_equal(mod.endog.base is not mod.data.orig_endog, True)
    assert_equal(mod.endog.base is not endog, True)
    assert_equal(mod.data.orig_endog.values.base is not endog, True)
    endog.iloc[0, 0] = 2
    assert_equal(mod.endog, np.r_[1, 2].reshape(2, 1))
    assert_allclose(mod.data.orig_endog, endog)
    kwargs2 = {'k_states': 1, 'design': [[1], [0.0]], 'obs_cov': [[1, 0], [0, 1]], 'transition': [[1]], 'selection': [[1]], 'state_cov': [[1]], 'initialization': 'approximate_diffuse'}
    endog = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]}, index=dates)
    mod = check_endog(endog, k_endog=2, **kwargs2)
    mod.filter([])

def test_diagnostics():
    if False:
        while True:
            i = 10
    (mod, res) = get_dummy_mod()
    shape = res.filter_results._standardized_forecasts_error.shape
    res.filter_results._standardized_forecasts_error = np.random.normal(size=shape)
    actual = res.test_normality(method=None)
    desired = res.test_normality(method='jarquebera')
    assert_allclose(actual, desired)
    assert_raises(NotImplementedError, res.test_normality, method='invalid')
    actual = res.test_heteroskedasticity(method=None)
    desired = res.test_heteroskedasticity(method='breakvar')
    assert_allclose(actual, desired)
    with pytest.raises(ValueError):
        res.test_heteroskedasticity(method=None, alternative='invalid')
    with pytest.raises(NotImplementedError):
        res.test_heteroskedasticity(method='invalid')
    actual = res.test_serial_correlation(method=None)
    desired = res.test_serial_correlation(method='ljungbox')
    assert_allclose(actual, desired)
    with pytest.raises(NotImplementedError):
        res.test_serial_correlation(method='invalid')
    res.test_heteroskedasticity(method=None, alternative='d', use_f=False)
    res.test_serial_correlation(method='boxpierce')

def test_small_sample_serial_correlation_test():
    if False:
        print('Hello World!')
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    niledata = nile.data.load_pandas().data
    niledata.index = pd.date_range('1871-01-01', '1970-01-01', freq='YS')
    mod = SARIMAX(endog=niledata['volume'], order=(1, 0, 1), trend='n', freq=niledata.index.freq)
    res = mod.fit()
    actual = res.test_serial_correlation(method='ljungbox', lags=10, df_adjust=True)[0, :, -1]
    assert_allclose(actual, [14.116, 0.0788], atol=0.001)

def test_diagnostics_nile_eviews():
    if False:
        for i in range(10):
            print('nop')
    niledata = nile.data.load_pandas().data
    niledata.index = pd.date_range('1871-01-01', '1970-01-01', freq='YS')
    mod = MLEModel(niledata['volume'], k_states=1, initialization='approximate_diffuse', initial_variance=1000000000000000.0, loglikelihood_burn=1)
    mod.ssm['design', 0, 0] = 1
    mod.ssm['obs_cov', 0, 0] = np.exp(9.60035)
    mod.ssm['transition', 0, 0] = 1
    mod.ssm['selection', 0, 0] = 1
    mod.ssm['state_cov', 0, 0] = np.exp(7.348705)
    res = mod.filter([])
    actual = res.test_serial_correlation(method='ljungbox', lags=10)[0, :, -1]
    assert_allclose(actual, [13.117, 0.217], atol=0.001)
    actual = res.test_normality(method='jarquebera')[0, :2]
    assert_allclose(actual, [0.041686, 0.979373], atol=1e-05)

def test_diagnostics_nile_durbinkoopman():
    if False:
        print('Hello World!')
    niledata = nile.data.load_pandas().data
    niledata.index = pd.date_range('1871-01-01', '1970-01-01', freq='YS')
    mod = MLEModel(niledata['volume'], k_states=1, initialization='approximate_diffuse', initial_variance=1000000000000000.0, loglikelihood_burn=1)
    mod.ssm['design', 0, 0] = 1
    mod.ssm['obs_cov', 0, 0] = 15099.0
    mod.ssm['transition', 0, 0] = 1
    mod.ssm['selection', 0, 0] = 1
    mod.ssm['state_cov', 0, 0] = 1469.1
    res = mod.filter([])
    actual = res.test_serial_correlation(method='ljungbox', lags=9)[0, 0, -1]
    assert_allclose(actual, [8.84], atol=0.01)
    norm = res.test_normality(method='jarquebera')[0]
    actual = [norm[0], norm[2], norm[3]]
    assert_allclose(actual, [0.05, -0.03, 3.09], atol=0.01)
    actual = res.test_heteroskedasticity(method='breakvar')[0, 0]
    assert_allclose(actual, [0.61], atol=0.01)

@pytest.mark.smoke
def test_prediction_results():
    if False:
        for i in range(10):
            print('nop')
    (mod, res) = get_dummy_mod()
    predict = res.get_prediction()
    predict.summary_frame()

def test_lutkepohl_information_criteria():
    if False:
        print('Hello World!')
    dta = pd.DataFrame(results_var_misc.lutkepohl_data, columns=['inv', 'inc', 'consump'], index=pd.date_range('1960-01-01', '1982-10-01', freq='QS'))
    dta['dln_inv'] = np.log(dta['inv']).diff()
    dta['dln_inc'] = np.log(dta['inc']).diff()
    dta['dln_consump'] = np.log(dta['consump']).diff()
    endog = dta.loc['1960-04-01':'1978-10-01', ['dln_inv', 'dln_inc', 'dln_consump']]
    true = results_var_misc.lutkepohl_ar1_lustats
    mod = sarimax.SARIMAX(endog['dln_inv'], order=(1, 0, 0), trend='c', loglikelihood_burn=1)
    res = mod.filter(true['params'])
    assert_allclose(res.llf, true['loglike'])
    aic = res.info_criteria('aic', method='lutkepohl') - 2 * 2 / res.nobs_effective
    bic = res.info_criteria('bic', method='lutkepohl') - 2 * np.log(res.nobs_effective) / res.nobs_effective
    hqic = res.info_criteria('hqic', method='lutkepohl') - 2 * 2 * np.log(np.log(res.nobs_effective)) / res.nobs_effective
    assert_allclose(aic, true['aic'])
    assert_allclose(bic, true['bic'])
    assert_allclose(hqic, true['hqic'])
    true = results_var_misc.lutkepohl_ar1
    aic = res.aic - 2
    bic = res.bic - np.log(res.nobs_effective)
    assert_allclose(aic, true['estat_aic'])
    assert_allclose(bic, true['estat_bic'])
    aic = res.info_criteria('aic') - 2
    bic = res.info_criteria('bic') - np.log(res.nobs_effective)
    assert_allclose(aic, true['estat_aic'])
    assert_allclose(bic, true['estat_bic'])
    true = results_var_misc.lutkepohl_var1_lustats
    mod = varmax.VARMAX(endog, order=(1, 0), trend='n', error_cov_type='unstructured', loglikelihood_burn=1)
    res = mod.filter(true['params'])
    assert_allclose(res.llf, true['loglike'])
    aic = res.info_criteria('aic', method='lutkepohl') - 2 * 6 / res.nobs_effective
    bic = res.info_criteria('bic', method='lutkepohl') - 6 * np.log(res.nobs_effective) / res.nobs_effective
    hqic = res.info_criteria('hqic', method='lutkepohl') - 2 * 6 * np.log(np.log(res.nobs_effective)) / res.nobs_effective
    assert_allclose(aic, true['aic'])
    assert_allclose(bic, true['bic'])
    assert_allclose(hqic, true['hqic'])
    true = results_var_misc.lutkepohl_var1
    aic = res.aic - 2 * 6
    bic = res.bic - 6 * np.log(res.nobs_effective)
    assert_allclose(aic, true['estat_aic'])
    assert_allclose(bic, true['estat_bic'])
    aic = res.info_criteria('aic') - 2 * 6
    bic = res.info_criteria('bic') - 6 * np.log(res.nobs_effective)
    assert_allclose(aic, true['estat_aic'])
    assert_allclose(bic, true['estat_bic'])

def test_append_extend_apply_invalid():
    if False:
        i = 10
        return i + 15
    niledata = nile.data.load_pandas().data['volume']
    niledata.index = pd.date_range('1871-01-01', '1970-01-01', freq='YS')
    endog1 = niledata.iloc[:20]
    endog2 = niledata.iloc[20:40]
    mod = sarimax.SARIMAX(endog1, order=(1, 0, 0), concentrate_scale=True)
    res1 = mod.smooth([0.5])
    assert_raises(ValueError, res1.append, endog2, fit_kwargs={'cov_type': 'approx'})
    assert_raises(ValueError, res1.extend, endog2, fit_kwargs={'cov_type': 'approx'})
    assert_raises(ValueError, res1.apply, endog2, fit_kwargs={'cov_type': 'approx'})
    assert_raises(ValueError, res1.append, endog2, fit_kwargs={'cov_kwds': {}})
    assert_raises(ValueError, res1.extend, endog2, fit_kwargs={'cov_kwds': {}})
    assert_raises(ValueError, res1.apply, endog2, fit_kwargs={'cov_kwds': {}})
    wrong_freq = niledata.iloc[20:40]
    wrong_freq.index = pd.date_range(start=niledata.index[0], periods=len(wrong_freq), freq='MS')
    message = 'Given `endog` does not have an index that extends the index of the model. Expected index frequency is'
    with pytest.raises(ValueError, match=message):
        res1.append(wrong_freq)
    with pytest.raises(ValueError, match=message):
        res1.extend(wrong_freq)
    message = 'Given `exog` does not have an index that extends the index of the model. Expected index frequency is'
    with pytest.raises(ValueError, match=message):
        res1.append(endog2, exog=wrong_freq)
    message = 'The indices for endog and exog are not aligned'
    with pytest.raises(ValueError, match=message):
        res1.extend(endog2, exog=wrong_freq)
    not_cts = niledata.iloc[21:41]
    message = 'Given `endog` does not have an index that extends the index of the model.$'
    with pytest.raises(ValueError, match=message):
        res1.append(not_cts)
    with pytest.raises(ValueError, match=message):
        res1.extend(not_cts)
    message = 'Given `exog` does not have an index that extends the index of the model.$'
    with pytest.raises(ValueError, match=message):
        res1.append(endog2, exog=not_cts)
    message = 'The indices for endog and exog are not aligned'
    with pytest.raises(ValueError, match=message):
        res1.extend(endog2, exog=not_cts)
    endog3 = pd.Series(niledata.iloc[:20].values)
    endog4 = pd.Series(niledata.iloc[:40].values)[20:]
    mod2 = sarimax.SARIMAX(endog3, order=(1, 0, 0), exog=endog3, concentrate_scale=True)
    res2 = mod2.smooth([0.2, 0.5])
    not_cts = pd.Series(niledata[:41].values)[21:]
    message = 'Given `endog` does not have an index that extends the index of the model.$'
    with pytest.raises(ValueError, match=message):
        res2.append(not_cts)
    with pytest.raises(ValueError, match=message):
        res2.extend(not_cts)
    message = 'Given `exog` does not have an index that extends the index of the model.$'
    with pytest.raises(ValueError, match=message):
        res2.append(endog4, exog=not_cts)
    message = 'The indices for endog and exog are not aligned'
    with pytest.raises(ValueError, match=message):
        res2.extend(endog4, exog=not_cts)

def test_integer_params():
    if False:
        return 10
    mod = sarimax.SARIMAX([1, 1, 1], order=(1, 0, 0), exog=[2, 2, 2], concentrate_scale=True)
    res = mod.filter([1, 0])
    p = res.predict(end=5, dynamic=True, exog=[3, 3, 4])
    assert_equal(p.dtype, np.float64)

def check_states_index(states, ix, predicted_ix, cols):
    if False:
        for i in range(10):
            print('nop')
    predicted_cov_ix = pd.MultiIndex.from_product([predicted_ix, cols]).swaplevel()
    filtered_cov_ix = pd.MultiIndex.from_product([ix, cols]).swaplevel()
    smoothed_cov_ix = pd.MultiIndex.from_product([ix, cols]).swaplevel()
    assert_(states.predicted.index.equals(predicted_ix))
    assert_(states.predicted.columns.equals(cols))
    assert_(states.predicted_cov.index.equals(predicted_cov_ix))
    assert_(states.predicted.columns.equals(cols))
    assert_(states.filtered.index.equals(ix))
    assert_(states.filtered.columns.equals(cols))
    assert_(states.filtered_cov.index.equals(filtered_cov_ix))
    assert_(states.filtered.columns.equals(cols))
    assert_(states.smoothed.index.equals(ix))
    assert_(states.smoothed.columns.equals(cols))
    assert_(states.smoothed_cov.index.equals(smoothed_cov_ix))
    assert_(states.smoothed.columns.equals(cols))

def test_states_index_periodindex():
    if False:
        print('Hello World!')
    nobs = 10
    ix = pd.period_range(start='2000', periods=nobs, freq='M')
    endog = pd.Series(np.zeros(nobs), index=ix)
    mod = sarimax.SARIMAX(endog, order=(2, 0, 0))
    res = mod.smooth([0.5, 0.1, 1.0])
    predicted_ix = pd.period_range(start=ix[0], periods=nobs + 1, freq='M')
    cols = pd.Index(['state.0', 'state.1'])
    check_states_index(res.states, ix, predicted_ix, cols)

def test_states_index_dateindex():
    if False:
        return 10
    nobs = 10
    ix = pd.date_range(start='2000', periods=nobs, freq='M')
    endog = pd.Series(np.zeros(nobs), index=ix)
    mod = sarimax.SARIMAX(endog, order=(2, 0, 0))
    res = mod.smooth([0.5, 0.1, 1.0])
    predicted_ix = pd.date_range(start=ix[0], periods=nobs + 1, freq='M')
    cols = pd.Index(['state.0', 'state.1'])
    check_states_index(res.states, ix, predicted_ix, cols)

def test_states_index_int64index():
    if False:
        for i in range(10):
            print('nop')
    nobs = 10
    ix = pd.Index(np.arange(10))
    endog = pd.Series(np.zeros(nobs), index=ix)
    mod = sarimax.SARIMAX(endog, order=(2, 0, 0))
    res = mod.smooth([0.5, 0.1, 1.0])
    predicted_ix = pd.Index(np.arange(11))
    cols = pd.Index(['state.0', 'state.1'])
    check_states_index(res.states, ix, predicted_ix, cols)

def test_states_index_rangeindex():
    if False:
        i = 10
        return i + 15
    nobs = 10
    ix = pd.RangeIndex(10)
    endog = pd.Series(np.zeros(nobs), index=ix)
    mod = sarimax.SARIMAX(endog, order=(2, 0, 0))
    res = mod.smooth([0.5, 0.1, 1.0])
    predicted_ix = pd.RangeIndex(11)
    cols = pd.Index(['state.0', 'state.1'])
    check_states_index(res.states, ix, predicted_ix, cols)
    ix = pd.RangeIndex(2, 32, 3)
    endog = pd.Series(np.zeros(nobs), index=ix)
    mod = sarimax.SARIMAX(endog, order=(2, 0, 0))
    res = mod.smooth([0.5, 0.1, 1.0])
    predicted_ix = pd.RangeIndex(2, 35, 3)
    cols = pd.Index(['state.0', 'state.1'])
    check_states_index(res.states, ix, predicted_ix, cols)

def test_invalid_kwargs():
    if False:
        print('Hello World!')
    endog = [0, 0, 1.0]
    sarimax.SARIMAX(endog)
    with pytest.warns(FutureWarning):
        sarimax.SARIMAX(endog, invalid_kwarg=True)