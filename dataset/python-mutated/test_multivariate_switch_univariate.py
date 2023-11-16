"""
Tests for automatic switching of the filter method from multivariate to
univariate when the forecast error covariance matrix is singular.

Author: Chad Fulton
License: Simplified-BSD

References
----------

Kim, Chang-Jin, and Charles R. Nelson. 1999.
"State-Space Models with Regime Switching:
Classical and Gibbs-Sampling Approaches with Applications".
MIT Press Books. The MIT Press.

Hamilton, James D. 1994.
Time Series Analysis.
Princeton, N.J.: Princeton University Press.
"""
import numpy as np
import pytest
from statsmodels.tsa.statespace import mlemodel, sarimax, structural, varmax, dynamic_factor
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from numpy.testing import assert_allclose

def get_model(univariate, missing=None, init=None):
    if False:
        print('Hello World!')
    if univariate:
        endog = np.array([0.5, 1.2, -0.2, 0.3, -0.1, 0.4, 1.4, 0.9])
        if missing == 'init':
            endog[0:2] = np.nan
        elif missing == 'mixed':
            endog[2:4] = np.nan
        elif missing == 'all':
            endog[:] = np.nan
        mod = mlemodel.MLEModel(endog, k_states=1, k_posdef=1)
        mod['design', 0, 0] = 1.0
        mod['transition', 0, 0] = 0.5
        mod['selection', 0, 0] = 1.0
        mod['state_cov', 0, 0] = 1.0
        mod['state_intercept', 0, 0] = 1.0
    else:
        endog = np.array([[0.5, 1.2, -0.2, 0.3, -0.1, 0.4, 1.4, 0.9], [-0.2, -0.3, -0.1, 0.1, 0.01, 0.05, -0.13, -0.2]]).T
        if missing == 'init':
            endog[0:2, :] = np.nan
        elif missing == 'mixed':
            endog[2:4, 0] = np.nan
            endog[3:6, 1] = np.nan
        elif missing == 'all':
            endog[:] = np.nan
        mod = mlemodel.MLEModel(endog, k_states=3, k_posdef=2)
        mod['obs_intercept'] = np.array([0.5, 0.2])
        mod['design'] = np.array([[0.1, -0.1, 0], [0.2, 0.3, 0]])
        mod['obs_cov'] = np.array([[5, -0.2], [-0.2, 3.0]])
        mod['transition', 0, 0] = 1
        mod['transition', 1:, 1:] = np.array([[0.5, -0.1], [1.0, 0.0]])
        mod['selection', :2, :2] = np.eye(2)
        mod['state_cov'] = np.array([[1.2, 0.2], [0.2, 2.5]])
        mod['state_intercept', :2] = np.array([1.0, -1.0])
    if init == 'diffuse':
        mod.ssm.initialize_diffuse()
    elif init == 'approximate_diffuse':
        mod.ssm.initialize_approximate_diffuse()
    elif init == 'stationary':
        mod.ssm.initialize_stationary()
    return mod

def check_filter_output(mod, periods, atol=0):
    if False:
        while True:
            i = 10
    if isinstance(mod, mlemodel.MLEModel):
        res_mv = mod.ssm.filter()
        mod.ssm.filter()
        kfilter = mod.ssm._kalman_filter
        kfilter.seek(0, True)
        kfilter.univariate_filter[periods] = 1
        for _ in range(mod.nobs):
            next(kfilter)
        res_switch = mod.ssm.results_class(mod.ssm)
        res_switch.update_representation(mod.ssm)
        res_switch.update_filter(kfilter)
        mod.ssm.filter_univariate = True
        res_uv = mod.ssm.filter()
    else:
        (res_mv, res_switch, res_uv) = mod
    assert_allclose(res_switch.llf, res_mv.llf)
    assert_allclose(res_switch.llf, res_uv.llf)
    assert_allclose(res_switch.scale, res_mv.scale)
    assert_allclose(res_switch.scale, res_uv.scale)
    attrs = ['forecasts_error_diffuse_cov', 'predicted_state', 'predicted_state_cov', 'predicted_diffuse_state_cov', 'filtered_state', 'filtered_state_cov', 'llf_obs']
    for attr in attrs:
        attr_mv = getattr(res_mv, attr)
        attr_uv = getattr(res_uv, attr)
        attr_switch = getattr(res_switch, attr)
        if attr_mv is None:
            continue
        assert_allclose(attr_switch, attr_mv, atol=atol)
        assert_allclose(attr_switch, attr_uv, atol=atol)
    attrs = ['forecasts_error', 'forecasts_error_cov', 'kalman_gain']
    for attr in attrs:
        actual = getattr(res_switch, attr).copy()
        desired = getattr(res_mv, attr).copy()
        actual[..., periods] = 0
        desired[..., periods] = 0
        assert_allclose(actual, desired, atol=atol)
        actual = getattr(res_switch, attr)[..., periods]
        desired = getattr(res_uv, attr)[..., periods]
        assert_allclose(actual, desired, atol=atol)

def check_smoother_output(mod, periods, atol=1e-12):
    if False:
        return 10
    if isinstance(mod, mlemodel.MLEModel):
        res_mv = mod.ssm.smooth()
        kfilter = mod.ssm._kalman_filter
        kfilter.seek(0, True)
        kfilter.univariate_filter[periods] = 1
        for _ in range(mod.nobs):
            next(kfilter)
        res_switch = mod.ssm.results_class(mod.ssm)
        res_switch.update_representation(mod.ssm)
        res_switch.update_filter(kfilter)
        mod.ssm._kalman_smoother.reset(True)
        smoother = mod.ssm._smooth()
        res_switch.update_smoother(smoother)
        mod.ssm.filter_univariate = True
        res_uv = mod.ssm.smooth()
    else:
        (res_mv, res_switch, res_uv) = mod
    attrs = ['scaled_smoothed_estimator', 'scaled_smoothed_estimator_cov', 'smoothed_state', 'smoothed_state_cov', 'smoothed_state_autocov', 'smoothed_state_disturbance', 'smoothed_state_disturbance_cov', 'innovations_transition']
    for attr in attrs:
        attr_mv = getattr(res_mv, attr)
        attr_uv = getattr(res_uv, attr)
        attr_switch = getattr(res_switch, attr)
        if attr_mv is None:
            continue
        assert_allclose(attr_uv, attr_mv, atol=atol)
        assert_allclose(attr_switch, attr_mv, atol=atol)
        assert_allclose(attr_switch, attr_uv, atol=atol)
    attrs = ['smoothing_error', 'smoothed_measurement_disturbance', 'smoothed_measurement_disturbance_cov']
    for attr in attrs:
        attr_mv = getattr(res_mv, attr)
        attr_uv = getattr(res_uv, attr)
        attr_switch = getattr(res_switch, attr)
        if attr_mv is None:
            continue
        actual = attr_switch.copy()
        desired = attr_mv.copy()
        actual[..., periods] = 0
        desired[..., periods] = 0
        assert_allclose(actual, desired)

@pytest.mark.parametrize('missing', [None, 'init', 'mixed', 'all'])
def test_basic(missing):
    if False:
        while True:
            i = 10
    mod = get_model(univariate=True, missing=missing)
    mod.initialize_known([0], [[0]])
    mod.ssm.filter()
    uf = np.array(mod.ssm._kalman_filter.univariate_filter)
    if missing in ['init', 'all']:
        assert_allclose(uf, 0)
    else:
        assert_allclose(uf[0], 1)
        assert_allclose(uf[1:], 0)

@pytest.mark.parametrize('univariate', [True, False])
@pytest.mark.parametrize('missing', [None, 'init', 'mixed', 'all'])
@pytest.mark.parametrize('init', ['stationary', 'diffuse', 'approximate_diffuse'])
@pytest.mark.parametrize('periods', [np.s_[0], np.s_[4:6], np.s_[:]])
def test_filter_output(univariate, missing, init, periods):
    if False:
        for i in range(10):
            print('nop')
    mod = get_model(univariate, missing, init)
    check_filter_output(mod, periods)

@pytest.mark.parametrize('univariate', [True, False])
@pytest.mark.parametrize('missing', [None, 'init', 'mixed', 'all'])
@pytest.mark.parametrize('init', ['stationary', 'diffuse', 'approximate_diffuse'])
@pytest.mark.parametrize('periods', [np.s_[0], np.s_[4:6], np.s_[:]])
@pytest.mark.parametrize('option', [None, 'alternate_timing'])
def test_smoother_output(univariate, missing, init, periods, option):
    if False:
        for i in range(10):
            print('nop')
    mod = get_model(univariate, missing, init)
    if option == 'alternate_timing':
        if init == 'diffuse':
            return
        mod.ssm.timing_init_filtered = True
    atol = 1e-12
    if missing == 'init' and init == 'approximate_diffuse':
        atol = 1e-06
    check_smoother_output(mod, periods, atol=atol)

def test_invalid_options():
    if False:
        i = 10
        return i + 15
    mod = get_model(univariate=True)
    mod.initialize_known([0], [[0]])
    mod.ssm.set_inversion_method(0, solve_lu=True)
    msg = 'Singular forecast error covariance matrix detected, but multivariate filter cannot fall back to univariate filter when the inversion method is set to anything other than INVERT_UNIVARIATE or SOLVE_CHOLESKY.'
    with pytest.raises(NotImplementedError, match=msg):
        mod.ssm.filter()
    mod = get_model(univariate=True)
    mod.initialize_known([0], [[0]])
    mod.ssm.smooth_classical = True
    msg = 'Cannot use classical smoothing when the multivariate filter has fallen back to univariate filtering.'
    with pytest.raises(NotImplementedError, match=msg):
        mod.ssm.smooth()
    mod = get_model(univariate=True)
    mod.initialize_known([0], [[0]])
    mod.ssm.smooth_alternative = True
    msg = 'Cannot use alternative smoothing when the multivariate filter has fallen back to univariate filtering.'
    with pytest.raises(NotImplementedError, match=msg):
        mod.ssm.smooth()

@pytest.mark.parametrize('missing', [None, 'init', 'mixed', 'all'])
@pytest.mark.parametrize('periods', [np.s_[0], np.s_[4:6], np.s_[:]])
@pytest.mark.parametrize('use_exact_diffuse', [False, True])
def test_sarimax(missing, periods, use_exact_diffuse):
    if False:
        while True:
            i = 10
    endog = np.array([0.5, 1.2, -0.2, 0.3, -0.1, 0.4, 1.4, 0.9])
    exog = np.ones_like(endog)
    if missing == 'init':
        endog[0:2] = np.nan
    elif missing == 'mixed':
        endog[2:4] = np.nan
    elif missing == 'all':
        endog[:] = np.nan
    mod = sarimax.SARIMAX(endog, order=(1, 1, 1), trend='t', seasonal_order=(1, 1, 1, 2), exog=exog, use_exact_diffuse=use_exact_diffuse)
    mod.update([0.1, 0.3, 0.5, 0.2, 0.05, -0.1, 1.0])
    check_filter_output(mod, periods, atol=1e-08)
    check_smoother_output(mod, periods)

@pytest.mark.parametrize('missing', [None, 'init', 'mixed', 'all'])
@pytest.mark.parametrize('periods', [np.s_[0], np.s_[4:6], np.s_[:]])
@pytest.mark.parametrize('use_exact_diffuse', [False, True])
def test_unobserved_components(missing, periods, use_exact_diffuse):
    if False:
        return 10
    endog = np.array([0.5, 1.2, -0.2, 0.3, -0.1, 0.4, 1.4, 0.9])
    exog = np.ones_like(endog)
    if missing == 'init':
        endog[0:2] = np.nan
    elif missing == 'mixed':
        endog[2:4] = np.nan
    elif missing == 'all':
        endog[:] = np.nan
    mod = structural.UnobservedComponents(endog, 'llevel', exog=exog, seasonal=2, autoregressive=1, use_exact_diffuse=use_exact_diffuse)
    mod.update([1.0, 0.1, 0.3, 0.05, 0.15, 0.5])
    check_filter_output(mod, periods)
    check_smoother_output(mod, periods)

@pytest.mark.parametrize('missing', [None, 'init', 'mixed', 'all'])
@pytest.mark.parametrize('periods', [np.s_[0], np.s_[4:6], np.s_[:]])
def test_varmax(missing, periods):
    if False:
        i = 10
        return i + 15
    endog = np.array([[0.5, 1.2, -0.2, 0.3, -0.1, 0.4, 1.4, 0.9], [-0.2, -0.3, -0.1, 0.1, 0.01, 0.05, -0.13, -0.2]]).T
    exog = np.ones_like(endog[:, 0])
    if missing == 'init':
        endog[0:2, :] = np.nan
    elif missing == 'mixed':
        endog[2:4, 0] = np.nan
        endog[3:6, 1] = np.nan
    elif missing == 'all':
        endog[:] = np.nan
    mod = varmax.VARMAX(endog, order=(1, 0), trend='t', exog=exog)
    mod.update([0.1, -0.1, 0.5, 0.1, -0.05, 0.2, 0.4, 0.25, 1.2, 0.4, 2.3])
    check_filter_output(mod, periods, atol=1e-12)
    check_smoother_output(mod, periods)

@pytest.mark.parametrize('missing', [None, 'init', 'mixed', 'all'])
@pytest.mark.parametrize('periods', [np.s_[0], np.s_[4:6], np.s_[:]])
def test_dynamic_factor(missing, periods):
    if False:
        i = 10
        return i + 15
    endog = np.array([[0.5, 1.2, -0.2, 0.3, -0.1, 0.4, 1.4, 0.9], [-0.2, -0.3, -0.1, 0.1, 0.01, 0.05, -0.13, -0.2]]).T
    exog = np.ones_like(endog[:, 0])
    if missing == 'init':
        endog[0:2, :] = np.nan
    elif missing == 'mixed':
        endog[2:4, 0] = np.nan
        endog[3:6, 1] = np.nan
    elif missing == 'all':
        endog[:] = np.nan
    mod = dynamic_factor.DynamicFactor(endog, k_factors=1, factor_order=2, exog=exog)
    mod.update([1.0, -0.5, 0.3, -0.1, 1.2, 2.3, 0.5, 0.1])
    check_filter_output(mod, periods)
    check_smoother_output(mod, periods)

@pytest.mark.parametrize('missing', [None, 'mixed'])
def test_simulation_smoothing(missing):
    if False:
        for i in range(10):
            print('nop')
    mod_switch = get_model(univariate=True, missing=missing)
    mod_switch.initialize_known([0], [[0]])
    sim_switch = mod_switch.simulation_smoother()
    mod_uv = get_model(univariate=True, missing=missing)
    mod_uv.initialize_known([0], [[0]])
    mod_uv.ssm.filter_univariate = True
    sim_uv = mod_uv.simulation_smoother()
    simulate_switch = mod_switch.simulate([], 10, random_state=1234)
    simulate_uv = mod_uv.simulate([], 10, random_state=1234)
    assert_allclose(simulate_switch, simulate_uv)
    sim_switch.simulate(random_state=1234)
    sim_uv.simulate(random_state=1234)
    kfilter = sim_switch._simulation_smoother.simulated_kfilter
    uf_switch = np.array(kfilter.univariate_filter, copy=True)
    assert_allclose(uf_switch[0], 1)
    assert_allclose(uf_switch[1:], 0)
    kfilter = sim_uv._simulation_smoother.simulated_kfilter.univariate_filter
    uf_uv = np.array(kfilter, copy=True)
    assert_allclose(uf_uv, 1)
    if missing == 'mixed':
        kfilter = sim_switch._simulation_smoother.secondary_simulated_kfilter.univariate_filter
        uf_switch = np.array(kfilter, copy=True)
        assert_allclose(uf_switch[0], 1)
        assert_allclose(uf_switch[1:], 0)
        kfilter = sim_uv._simulation_smoother.secondary_simulated_kfilter.univariate_filter
        uf_uv = np.array(kfilter, copy=True)
        assert_allclose(uf_uv, 1)
    attrs = ['generated_measurement_disturbance', 'generated_state_disturbance', 'generated_obs', 'generated_state', 'simulated_state', 'simulated_measurement_disturbance', 'simulated_state_disturbance']
    for attr in attrs:
        assert_allclose(getattr(sim_switch, attr), getattr(sim_uv, attr))

def test_time_varying_model(reset_randomstate):
    if False:
        print('Hello World!')
    endog = np.array([[0.5, 1.2, -0.2, 0.3, -0.1, 0.4, 1.4, 0.9], [-0.2, -0.3, -0.1, 0.1, 0.01, 0.05, -0.13, -0.2]]).T
    np.random.seed(1234)
    mod_switch = TVSS(endog)
    mod_switch['design', ..., 3] = 0
    mod_switch['obs_cov', ..., 3] = 0
    mod_switch['obs_cov', 1, 1, 3] = 1.0
    res_switch = mod_switch.ssm.smooth()
    kfilter = mod_switch.ssm._kalman_filter
    uf_switch = np.array(kfilter.univariate_filter, copy=True)
    np.random.seed(1234)
    mod_uv = TVSS(endog)
    mod_uv['design', ..., 3] = 0
    mod_uv['obs_cov', ..., 3] = 0
    mod_uv['obs_cov', 1, 1, 3] = 1.0
    mod_uv.ssm.filter_univariate = True
    res_uv = mod_uv.ssm.smooth()
    kfilter = mod_uv.ssm._kalman_filter
    uf_uv = np.array(kfilter.univariate_filter, copy=True)
    np.random.seed(1234)
    endog_mv = endog.copy()
    endog_mv[3, 0] = np.nan
    mod_mv = TVSS(endog_mv)
    mod_mv['design', ..., 3] = 0
    mod_mv['obs_cov', ..., 3] = 0
    mod_mv['obs_cov', 1, 1, 3] = 1.0
    res_mv = mod_mv.ssm.smooth()
    kfilter = mod_mv.ssm._kalman_filter
    uf_mv = np.array(kfilter.univariate_filter, copy=True)
    assert_allclose(uf_switch[:3], 0)
    assert_allclose(uf_switch[3], 1)
    assert_allclose(uf_switch[4:], 0)
    assert_allclose(uf_uv, 1)
    assert_allclose(uf_mv, 0)
    check_filter_output([res_mv, res_switch, res_uv], np.s_[3])
    check_smoother_output([res_mv, res_switch, res_uv], np.s_[3])