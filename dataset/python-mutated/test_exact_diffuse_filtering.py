"""
Tests for exact diffuse initialization

Notes
-----

These tests are against four sources:

- Koopman (1997)
- The R package KFAS (v1.3.1): test_exact_diffuse_filtering.R
- Stata: test_exact_diffuse_filtering_stata.do
- statsmodels state space models using approximate diffuse filtering

Koopman (1997) provides analytic results for a few cases that we can test
against. More comprehensive tests are available against the R package KFAS,
which also uses the Durbin and Koopman (2012) univariate diffuse filtering
method. However, there are apparently some bugs in the KFAS output (see notes
below), so some tests are run against Stata.

KFAS v1.3.1 appears to have the following bugs:

- Incorrect filtered covariance matrix (in their syntax, kf$Ptt). These
  matrices are not even symmetric, so they are clearly wrong.
- Loglikelihood computation appears to be incorrect for the diffuse part of
  the state. See the section with "Note: Apparent loglikelihood discrepancy"
  in the R file. It appears that KFAS does not include the constant term
  (-0.5 * log(2 pi)) for the diffuse observations, whereas the loglikelihood
  function as given in e.g. section 7.2.5 of Durbin and Koopman (2012) shows
  that it should be included. To confirm this, we also check against the
  loglikelihood value computed by Stata.

Stata uses the DeJong diffuse filtering method, which gives almost identical
results but does imply some numerical differences for output at the 6th or 7th
decimal place.

Finally, we have tests against the same model using approximate (rather than
exact) diffuse filtering. These will by definition have some discrepancies in
the diffuse observations.

Author: Chad Fulton
License: Simplified-BSD
"""
from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
import os
from statsmodels import datasets
from statsmodels.tsa.statespace.initialization import Initialization
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from numpy.testing import assert_equal, assert_allclose
from . import kfas_helpers
current_path = os.path.dirname(os.path.abspath(__file__))
macrodata = datasets.macrodata.load_pandas().data
macrodata.index = pd.period_range(start='1959Q1', end='2009Q3', freq='Q')

def model_local_level(endog=None, params=None, direct=False):
    if False:
        for i in range(10):
            print('nop')
    if endog is None:
        y1 = 10.2394
        endog = np.r_[y1, [1] * 9]
    if params is None:
        params = [1.993, 8.253]
    (sigma2_y, sigma2_mu) = params
    if direct:
        mod = None
        ssm = KalmanSmoother(k_endog=1, k_states=1, k_posdef=1)
        ssm.bind(endog)
        init = Initialization(ssm.k_states, initialization_type='diffuse')
        ssm.initialize(init)
        ssm['design', :] = 1
        ssm['obs_cov', :] = sigma2_y
        ssm['transition', :] = 1
        ssm['selection', :] = 1
        ssm['state_cov', :] = sigma2_mu
    else:
        mod = UnobservedComponents(endog, 'llevel')
        mod.update(params)
        ssm = mod.ssm
        ssm.initialize(Initialization(ssm.k_states, 'diffuse'))
    return (mod, ssm)

def model_local_linear_trend(endog=None, params=None, direct=False):
    if False:
        i = 10
        return i + 15
    if endog is None:
        y1 = 10.2394
        y2 = 4.2039
        y3 = 6.123123
        endog = np.r_[y1, y2, y3, [1] * 7]
    if params is None:
        params = [1.993, 8.253, 2.334]
    (sigma2_y, sigma2_mu, sigma2_beta) = params
    if direct:
        mod = None
        ssm = KalmanSmoother(k_endog=1, k_states=2, k_posdef=2)
        ssm.bind(endog)
        init = Initialization(ssm.k_states, initialization_type='diffuse')
        ssm.initialize(init)
        ssm['design', 0, 0] = 1
        ssm['obs_cov', 0, 0] = sigma2_y
        ssm['transition'] = np.array([[1, 1], [0, 1]])
        ssm['selection'] = np.eye(2)
        ssm['state_cov'] = np.diag([sigma2_mu, sigma2_beta])
    else:
        mod = UnobservedComponents(endog, 'lltrend')
        mod.update(params)
        ssm = mod.ssm
        ssm.initialize(Initialization(ssm.k_states, 'diffuse'))
    return (mod, ssm)

def model_common_level(endog=None, params=None, restricted=False):
    if False:
        i = 10
        return i + 15
    if endog is None:
        y11 = 10.2394
        y21 = 8.2304
        endog = np.column_stack([np.r_[y11, [1] * 9], np.r_[y21, [1] * 9]])
    if params is None:
        params = [0.1111, 3.2324]
    (theta, sigma2_mu) = params
    if not restricted:
        ssm = KalmanSmoother(k_endog=2, k_states=2, k_posdef=1)
        ssm.bind(endog.T)
        init = Initialization(ssm.k_states, initialization_type='diffuse')
        ssm.initialize(init)
        ssm['design'] = np.array([[1, 0], [theta, 1]])
        ssm['obs_cov'] = np.eye(2)
        ssm['transition'] = np.eye(2)
        ssm['selection', 0, 0] = 1
        ssm['state_cov', 0, 0] = sigma2_mu
    else:
        ssm = KalmanSmoother(k_endog=2, k_states=1, k_posdef=1)
        ssm.bind(endog.T)
        init = Initialization(ssm.k_states, initialization_type='diffuse')
        ssm.initialize(init)
        ssm['design'] = np.array([[1, theta]]).T
        ssm['obs_cov'] = np.eye(2)
        ssm['transition', :] = 1
        ssm['selection', :] = 1
        ssm['state_cov', :] = sigma2_mu
    return ssm

def model_var1(endog=None, params=None, measurement_error=False, init=None):
    if False:
        print('Hello World!')
    if endog is None:
        levels = macrodata[['realgdp', 'realcons']]
        endog = np.log(levels).iloc[:21].diff().iloc[1:] * 400
    if params is None:
        params = np.r_[0.5, 0.3, 0.2, 0.4, 2 ** 0.5, 0, 3 ** 0.5]
        if measurement_error:
            params = np.r_[params, 4, 5]
    mod = VARMAX(endog, order=(1, 0), trend='n', measurement_error=measurement_error)
    mod.update(params)
    ssm = mod.ssm
    if init is None:
        init = Initialization(ssm.k_states, 'diffuse')
    ssm.initialize(init)
    return (mod, ssm)

def model_dfm(endog=None, params=None, factor_order=2):
    if False:
        i = 10
        return i + 15
    if endog is None:
        levels = macrodata[['realgdp', 'realcons']]
        endog = np.log(levels).iloc[:21].diff().iloc[1:] * 400
    if params is None:
        params = np.r_[0.5, 1.0, 1.5, 2.0, 0.9, 0.1]
    mod = DynamicFactor(endog, k_factors=1, factor_order=factor_order)
    mod.update(params)
    ssm = mod.ssm
    ssm.filter_univariate = True
    init = Initialization(ssm.k_states, 'diffuse')
    ssm.initialize(init)
    return (mod, ssm)

class TestLocalLevelAnalytic:

    @classmethod
    def setup_class(cls, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        (cls.mod, cls.ssm) = model_local_level(**kwargs)
        cls.res = cls.ssm.smooth()

    def test_results(self):
        if False:
            print('Hello World!')
        ssm = self.ssm
        res = self.res
        y1 = ssm.endog[0, 0]
        sigma2_y = ssm['obs_cov', 0, 0]
        sigma2_mu = ssm['state_cov', 0, 0]
        assert_allclose(res.predicted_state_cov[0, 0, 0], 0)
        assert_allclose(res.predicted_diffuse_state_cov[0, 0, 0], 1)
        assert_allclose(res.forecasts_error[0, 0], y1)
        assert_allclose(res.forecasts_error_cov[0, 0, 0], sigma2_y)
        assert_allclose(res.forecasts_error_diffuse_cov[0, 0, 0], 1)
        assert_allclose(res.kalman_gain[0, 0, 0], 1)
        assert_allclose(res.predicted_state[0, 1], y1)
        assert_allclose(res.predicted_state_cov[0, 0, 1], sigma2_y + sigma2_mu)
        assert_allclose(res.predicted_diffuse_state_cov[0, 0, 1], 0)
        assert_equal(res.nobs_diffuse, 1)

class TestLocalLevelAnalyticDirect(TestLocalLevelAnalytic):

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        super(TestLocalLevelAnalyticDirect, cls).setup_class(direct=True)

class TestLocalLinearTrendAnalytic:

    @classmethod
    def setup_class(cls, **kwargs):
        if False:
            i = 10
            return i + 15
        (cls.mod, cls.ssm) = model_local_linear_trend(**kwargs)
        cls.res = cls.ssm.smooth()

    def test_results(self):
        if False:
            while True:
                i = 10
        ssm = self.ssm
        res = self.res
        (y1, y2, y3) = ssm.endog[0, :3]
        sigma2_y = ssm['obs_cov', 0, 0]
        (sigma2_mu, sigma2_beta) = np.diagonal(ssm['state_cov'])
        assert_allclose(res.predicted_state_cov[..., 0], np.zeros((2, 2)))
        assert_allclose(res.predicted_diffuse_state_cov[..., 0], np.eye(2))
        q_mu = sigma2_mu / sigma2_y
        q_beta = sigma2_beta / sigma2_y
        assert_allclose(res.forecasts_error[0, 0], y1)
        assert_allclose(res.kalman_gain[:, 0, 0], [1, 0])
        assert_allclose(res.predicted_state[:, 1], [y1, 0])
        P2 = sigma2_y * np.array([[1 + q_mu, 0], [0, q_beta]])
        assert_allclose(res.predicted_state_cov[:, :, 1], P2)
        assert_allclose(res.predicted_diffuse_state_cov[0, 0, 1], np.ones((2, 2)))
        assert_allclose(res.predicted_state[:, 2], [2 * y2 - y1, y2 - y1])
        P3 = sigma2_y * np.array([[5 + 2 * q_mu + q_beta, 3 + q_mu + q_beta], [3 + q_mu + q_beta, 2 + q_mu + 2 * q_beta]])
        assert_allclose(res.predicted_state_cov[:, :, 2], P3)
        assert_allclose(res.predicted_diffuse_state_cov[:, :, 2], np.zeros((2, 2)))
        assert_equal(res.nobs_diffuse, 2)

class TestLocalLinearTrendAnalyticDirect(TestLocalLinearTrendAnalytic):

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        super(TestLocalLinearTrendAnalyticDirect, cls).setup_class(direct=True)

class TestLocalLinearTrendAnalyticMissing(TestLocalLinearTrendAnalytic):

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        y1 = 10.2394
        y2 = np.nan
        y3 = 6.123123
        endog = np.r_[y1, y2, y3, [1] * 7]
        super(TestLocalLinearTrendAnalyticMissing, cls).setup_class(endog=endog)

    def test_results(self):
        if False:
            return 10
        ssm = self.ssm
        res = self.res
        (y1, y2, y3) = ssm.endog[0, :3]
        sigma2_y = ssm['obs_cov', 0, 0]
        (sigma2_mu, sigma2_beta) = np.diagonal(ssm['state_cov'])
        q_mu = sigma2_mu / sigma2_y
        q_beta = sigma2_beta / sigma2_y
        a4 = [1.5 * y3 - 0.5 * y1, 0.5 * y3 - 0.5 * y1]
        assert_allclose(res.predicted_state[:, 3], a4)
        P4 = sigma2_y * np.array([[2.5 + 1.5 * q_mu + 1.25 * q_beta, 1 + 0.5 * q_mu + 1.25 * q_beta], [1 + 0.5 * q_mu + 1.25 * q_beta, 0.5 + 0.5 * q_mu + 2.25 * q_beta]])
        assert_allclose(res.predicted_state_cov[:, :, 3], P4)
        assert_equal(res.nobs_diffuse, 3)

def test_common_level_analytic():
    if False:
        i = 10
        return i + 15
    mod = model_common_level()
    (y11, y21) = mod.endog[:, 0]
    theta = mod['design', 1, 0]
    sigma2_mu = mod['state_cov', 0, 0]
    res = mod.smooth()
    assert_allclose(res.predicted_state_cov[..., 0], np.zeros((2, 2)))
    assert_allclose(res.predicted_diffuse_state_cov[..., 0], np.eye(2))
    assert_allclose(res.predicted_state[:, 1], [y11, y21 - theta * y11])
    P2 = np.array([[1 + sigma2_mu, -theta], [-theta, 1 + theta ** 2]])
    assert_allclose(res.predicted_state_cov[..., 1], P2)
    assert_allclose(res.predicted_diffuse_state_cov[..., 1], np.zeros((2, 2)))
    assert_equal(res.nobs_diffuse, 1)

def test_common_level_restricted_analytic():
    if False:
        print('Hello World!')
    mod = model_common_level(restricted=True)
    (y11, y21) = mod.endog[:, 0]
    theta = mod['design', 1, 0]
    sigma2_mu = mod['state_cov', 0, 0]
    res = mod.smooth()
    assert_allclose(res.predicted_state_cov[..., 0], 0)
    assert_allclose(res.predicted_diffuse_state_cov[..., 0], 1)
    phi = 1 / (1 + theta ** 2)
    assert_allclose(res.predicted_state[:, 1], phi * (y11 + theta * y21))
    assert_allclose(res.predicted_state_cov[..., 1], phi + sigma2_mu)
    assert_allclose(res.predicted_diffuse_state_cov[..., 1], 0)
    assert_equal(res.nobs_diffuse, 1)

class CheckSSMResults:
    atol = 1e-14
    rtol = 1e-07
    atol_diffuse = 1e-07
    rtol_diffuse = None

    def check_object(self, actual, desired, rtol_diffuse):
        if False:
            return 10
        if actual is None or desired is None:
            return
        d = None
        if rtol_diffuse is None:
            rtol_diffuse = self.rtol_diffuse
        if rtol_diffuse is not None:
            d = self.d
            if rtol_diffuse != np.inf:
                assert_allclose(actual.T[:d], desired.T[:d], rtol=rtol_diffuse, atol=self.atol_diffuse)
        assert_allclose(actual.T[d:], desired.T[d:], rtol=self.rtol, atol=self.atol)

    def test_forecasts(self, rtol_diffuse=None):
        if False:
            return 10
        actual = self.results_a.forecasts
        desired = self.results_a.forecasts
        self.check_object(actual, desired, rtol_diffuse)

    def test_forecasts_error(self, rtol_diffuse=None):
        if False:
            i = 10
            return i + 15
        actual = self.results_a.forecasts_error
        desired = self.results_a.forecasts_error
        self.check_object(actual, desired, rtol_diffuse)

    def test_forecasts_error_cov(self, rtol_diffuse=None):
        if False:
            i = 10
            return i + 15
        actual = self.results_a.forecasts_error_cov
        desired = self.results_b.forecasts_error_cov
        self.check_object(actual, desired, rtol_diffuse)

    def test_filtered_state(self, rtol_diffuse=1e-05):
        if False:
            while True:
                i = 10
        actual = self.results_a.filtered_state
        desired = self.results_b.filtered_state
        self.check_object(actual, desired, rtol_diffuse)

    def test_filtered_state_cov(self, rtol_diffuse=None):
        if False:
            while True:
                i = 10
        actual = self.results_a.filtered_state_cov
        desired = self.results_b.filtered_state_cov
        self.check_object(actual, desired, rtol_diffuse)

    def test_predicted_state(self, rtol_diffuse=None):
        if False:
            while True:
                i = 10
        actual = self.results_a.predicted_state
        desired = self.results_b.predicted_state
        self.check_object(actual, desired, rtol_diffuse)

    def test_predicted_state_cov(self, rtol_diffuse=None):
        if False:
            for i in range(10):
                print('nop')
        actual = self.results_a.predicted_state_cov
        desired = self.results_b.predicted_state_cov
        self.check_object(actual, desired, rtol_diffuse)

    def test_kalman_gain(self, rtol_diffuse=None):
        if False:
            for i in range(10):
                print('nop')
        actual = self.results_a.kalman_gain
        desired = self.results_b.kalman_gain
        self.check_object(actual, desired, rtol_diffuse)

    def test_loglike(self, rtol_diffuse=None):
        if False:
            i = 10
            return i + 15
        if np.isscalar(self.results_b.llf_obs):
            actual = np.sum(self.results_a.llf_obs)
            desired = self.results_b.llf_obs
            assert_allclose(actual, desired)
        else:
            actual = self.results_a.llf_obs
            desired = self.results_b.llf_obs
            self.check_object(actual, desired, rtol_diffuse)

    def test_smoothed_state(self, rtol_diffuse=1e-05):
        if False:
            while True:
                i = 10
        actual = self.results_a.smoothed_state
        desired = self.results_b.smoothed_state
        self.check_object(actual, desired, rtol_diffuse)

    def test_smoothed_state_cov(self, rtol_diffuse=1e-05):
        if False:
            return 10
        actual = self.results_a.smoothed_state_cov
        desired = self.results_b.smoothed_state_cov
        self.check_object(actual, desired, rtol_diffuse)

    def test_smoothed_state_autocov(self, rtol_diffuse=None):
        if False:
            i = 10
            return i + 15
        actual = self.results_a.smoothed_state_autocov
        desired = self.results_b.smoothed_state_autocov
        self.check_object(actual, desired, rtol_diffuse)

    def test_smoothed_measurement_disturbance(self, rtol_diffuse=1e-05):
        if False:
            for i in range(10):
                print('nop')
        actual = self.results_a.smoothed_measurement_disturbance
        desired = self.results_b.smoothed_measurement_disturbance
        self.check_object(actual, desired, rtol_diffuse)

    def test_smoothed_measurement_disturbance_cov(self, rtol_diffuse=1e-05):
        if False:
            while True:
                i = 10
        actual = self.results_a.smoothed_measurement_disturbance_cov
        desired = self.results_b.smoothed_measurement_disturbance_cov
        self.check_object(actual, desired, rtol_diffuse)

    def test_smoothed_state_disturbance(self, rtol_diffuse=1e-05):
        if False:
            for i in range(10):
                print('nop')
        actual = self.results_a.smoothed_state_disturbance
        desired = self.results_b.smoothed_state_disturbance
        self.check_object(actual, desired, rtol_diffuse)

    def test_smoothed_state_disturbance_cov(self, rtol_diffuse=1e-05):
        if False:
            i = 10
            return i + 15
        actual = self.results_a.smoothed_state_disturbance_cov
        desired = self.results_b.smoothed_state_disturbance_cov
        self.check_object(actual, desired, rtol_diffuse)

    @pytest.mark.skip('This is not computed in the univariate method or by KFAS.')
    def test_smoothing_error(self, rtol_diffuse=None):
        if False:
            i = 10
            return i + 15
        actual = self.results_a.smoothing_error
        desired = self.results_b.smoothing_error
        self.check_object(actual, desired, rtol_diffuse)

    def test_scaled_smoothed_estimator(self, rtol_diffuse=1e-05):
        if False:
            while True:
                i = 10
        actual = self.results_a.scaled_smoothed_estimator
        desired = self.results_b.scaled_smoothed_estimator
        self.check_object(actual, desired, rtol_diffuse)

    def test_scaled_smoothed_estimator_cov(self, rtol_diffuse=1e-05):
        if False:
            i = 10
            return i + 15
        actual = self.results_a.scaled_smoothed_estimator_cov
        desired = self.results_b.scaled_smoothed_estimator_cov
        self.check_object(actual, desired, rtol_diffuse)

    def test_forecasts_error_diffuse_cov(self, rtol_diffuse=None):
        if False:
            for i in range(10):
                print('nop')
        actual = self.results_a.forecasts_error_diffuse_cov
        desired = self.results_b.forecasts_error_diffuse_cov
        self.check_object(actual, desired, rtol_diffuse)

    def test_predicted_diffuse_state_cov(self, rtol_diffuse=None):
        if False:
            for i in range(10):
                print('nop')
        actual = self.results_a.predicted_diffuse_state_cov
        desired = self.results_b.predicted_diffuse_state_cov
        self.check_object(actual, desired, rtol_diffuse)

    def test_scaled_smoothed_diffuse_estimator(self, rtol_diffuse=None):
        if False:
            print('Hello World!')
        actual = self.results_a.scaled_smoothed_diffuse_estimator
        desired = self.results_b.scaled_smoothed_diffuse_estimator
        self.check_object(actual, desired, rtol_diffuse)

    def test_scaled_smoothed_diffuse1_estimator_cov(self, rtol_diffuse=None):
        if False:
            for i in range(10):
                print('nop')
        actual = self.results_a.scaled_smoothed_diffuse1_estimator_cov
        desired = self.results_b.scaled_smoothed_diffuse1_estimator_cov
        self.check_object(actual, desired, rtol_diffuse)

    def test_scaled_smoothed_diffuse2_estimator_cov(self, rtol_diffuse=None):
        if False:
            while True:
                i = 10
        actual = self.results_a.scaled_smoothed_diffuse2_estimator_cov
        desired = self.results_b.scaled_smoothed_diffuse2_estimator_cov
        self.check_object(actual, desired, rtol_diffuse)

    @pytest.mark.xfail(reason='No sim_a attribute', raises=AttributeError, strict=True)
    def test_simulation_smoothed_state(self):
        if False:
            i = 10
            return i + 15
        assert_allclose(self.sim_a.simulated_state, self.sim_a.simulated_state)

    @pytest.mark.xfail(reason='No sim_a attribute', raises=AttributeError, strict=True)
    def test_simulation_smoothed_measurement_disturbance(self):
        if False:
            print('Hello World!')
        assert_allclose(self.sim_a.simulated_measurement_disturbance, self.sim_a.simulated_measurement_disturbance)

    @pytest.mark.xfail(reason='No sim_a attribute', raises=AttributeError, strict=True)
    def test_simulation_smoothed_state_disturbance(self):
        if False:
            i = 10
            return i + 15
        assert_allclose(self.sim_a.simulated_state_disturbance, self.sim_a.simulated_state_disturbance)

class CheckApproximateDiffuseMixin:
    """
    Test the exact diffuse initialization against the approximate diffuse
    initialization. By definition, the first few observations will be quite
    different between the exact and approximate approach for many quantities,
    so we do not test them here.
    """
    approximate_diffuse_variance = 1000000.0

    @classmethod
    def setup_class(cls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        init_approx = kwargs.pop('init_approx', None)
        super(CheckApproximateDiffuseMixin, cls).setup_class(*args, **kwargs)
        kappa = cls.approximate_diffuse_variance
        if init_approx is None:
            init_approx = Initialization(cls.ssm.k_states, 'approximate_diffuse', approximate_diffuse_variance=kappa)
        cls.ssm.initialize(init_approx)
        cls.results_b = cls.ssm.smooth()
        cls.rtol_diffuse = np.inf

    def test_initialization_approx(self):
        if False:
            i = 10
            return i + 15
        kappa = self.approximate_diffuse_variance
        assert_allclose(self.results_b.initial_state_cov, np.eye(self.ssm.k_states) * kappa)
        assert_equal(self.results_b.initial_diffuse_state_cov, None)

class CheckKFASMixin:
    """
    Test against values from KFAS
    """

    @classmethod
    def setup_class(cls, *args, **kwargs):
        if False:
            while True:
                i = 10
        kwargs.setdefault('filter_univariate', True)
        super(CheckKFASMixin, cls).setup_class(*args, **kwargs)
        cls.results_b = kfas_helpers.parse(cls.results_path, cls.ssm)
        cls.results_b.smoothed_state_autocov = None
        cls.results_b.kalman_gain = None
        cls.results_b.filtered_state_cov = None
        Finf = cls.results_b.forecasts_error_diffuse_cov.T
        Finf_nonsingular_obs = np.c_[[np.diag(Finf_t) for Finf_t in Finf]] > 0
        nonmissing = ~np.isnan(cls.ssm.endog).T
        constant = -0.5 * np.log(2 * np.pi) * (Finf_nonsingular_obs * nonmissing).sum(axis=1)
        cls.results_b.llf_obs += constant[:cls.results_a.nobs_diffuse].sum()

class CheckVAR1(CheckSSMResults):

    @classmethod
    def setup_class(cls, **kwargs):
        if False:
            print('Hello World!')
        filter_univariate = kwargs.pop('filter_univariate', False)
        (cls.mod, cls.ssm) = model_var1(**kwargs)
        if filter_univariate:
            cls.ssm.filter_univariate = True
        cls.results_a = cls.ssm.smooth()
        cls.d = cls.results_a.nobs_diffuse

    def test_nobs_diffuse(self):
        if False:
            i = 10
            return i + 15
        assert_allclose(self.d, 1)

    def test_initialization(self):
        if False:
            i = 10
            return i + 15
        assert_allclose(self.results_a.initial_state_cov, 0)
        assert_allclose(self.results_a.initial_diffuse_state_cov, np.eye(2))

class TestVAR1_Approx(CheckApproximateDiffuseMixin, CheckVAR1):
    pass

class TestVAR1_KFAS(CheckKFASMixin, CheckVAR1):
    results_path = os.path.join(current_path, 'results', 'results_exact_initial_var1_R.csv')

class CheckVAR1MeasurementError(CheckVAR1):

    @classmethod
    def setup_class(cls, **kwargs):
        if False:
            print('Hello World!')
        kwargs['measurement_error'] = True
        super(CheckVAR1MeasurementError, cls).setup_class(**kwargs)

class TestVAR1MeasurementError_Approx(CheckApproximateDiffuseMixin, CheckVAR1MeasurementError):
    approximate_diffuse_variance = 1000000000.0

    def test_smoothed_measurement_disturbance_cov(self, rtol_diffuse=None):
        if False:
            return 10
        super(TestVAR1MeasurementError_Approx, self).test_smoothed_measurement_disturbance_cov(rtol_diffuse=rtol_diffuse)

class TestVAR1MeasurementError_KFAS(CheckKFASMixin, CheckVAR1MeasurementError):
    results_path = os.path.join(current_path, 'results', 'results_exact_initial_var1_measurement_error_R.csv')

class CheckVAR1Missing(CheckVAR1):

    @classmethod
    def setup_class(cls, **kwargs):
        if False:
            while True:
                i = 10
        levels = macrodata[['realgdp', 'realcons']]
        endog = np.log(levels).iloc[:21].diff().iloc[1:] * 400
        endog.iloc[0:5, 0] = np.nan
        endog.iloc[8:12, :] = np.nan
        kwargs['endog'] = endog
        super(CheckVAR1Missing, cls).setup_class(**kwargs)

    def test_nobs_diffuse(self):
        if False:
            return 10
        assert_allclose(self.d, 2)

class TestVAR1Missing_Approx(CheckApproximateDiffuseMixin, CheckVAR1Missing):
    approximate_diffuse_variance = 10000000000.0

    def test_smoothed_state_cov(self, rtol_diffuse=None):
        if False:
            print('Hello World!')
        super(TestVAR1Missing_Approx, self).test_smoothed_state_cov(rtol_diffuse=rtol_diffuse)

class TestVAR1Missing_KFAS(CheckKFASMixin, CheckVAR1Missing):
    results_path = os.path.join(current_path, 'results', 'results_exact_initial_var1_missing_R.csv')

    def test_forecasts_error_cov(self):
        if False:
            for i in range(10):
                print('nop')
        bak = self.results_a.forecasts_error_cov[:]
        self.results_a.forecasts_error_cov[0, 1, :] = 0
        self.results_a.forecasts_error_cov[1, 0, :] = 0
        super(TestVAR1Missing_KFAS, self).test_forecasts_error_cov()
        self.results_a.forecasts_error_cov = bak

class CheckVAR1Mixed(CheckVAR1):

    @classmethod
    def setup_class(cls, **kwargs):
        if False:
            print('Hello World!')
        k_states = 2
        init = Initialization(k_states)
        init.set(0, 'diffuse')
        init.set(1, 'stationary')
        if kwargs.pop('approx', False):
            init_approx = Initialization(k_states)
            init_approx.set(0, 'approximate_diffuse')
            init_approx.set(1, 'stationary')
            kwargs['init_approx'] = init_approx
        super(CheckVAR1Mixed, cls).setup_class(init=init, **kwargs)

    def test_nobs_diffuse(self):
        if False:
            for i in range(10):
                print('nop')
        assert_allclose(self.d, 1)

    def test_initialization(self):
        if False:
            for i in range(10):
                print('nop')
        stationary_init = 3.5714285714285716
        assert_allclose(self.results_a.initial_state_cov, np.diag([0, stationary_init]))
        assert_allclose(self.results_a.initial_diffuse_state_cov, np.diag([1, 0]))

class TestVAR1Mixed_Approx(CheckVAR1Mixed, CheckApproximateDiffuseMixin, CheckVAR1):

    @classmethod
    def setup_class(cls, **kwargs):
        if False:
            i = 10
            return i + 15
        kwargs['approx'] = True
        super(TestVAR1Mixed_Approx, cls).setup_class(**kwargs)

    def test_initialization_approx(self):
        if False:
            while True:
                i = 10
        stationary_init = 3.5714285714285716
        kappa = self.approximate_diffuse_variance
        assert_allclose(self.results_b.initial_state_cov, np.diag([kappa, stationary_init]))
        assert_equal(self.results_b.initial_diffuse_state_cov, None)

class TestVAR1Mixed_KFAS(CheckVAR1Mixed, CheckKFASMixin, CheckVAR1):
    results_path = os.path.join(current_path, 'results', 'results_exact_initial_var1_mixed_R.csv')

    def test_predicted_state(self):
        if False:
            print('Hello World!')
        super(TestVAR1Mixed_KFAS, self).test_predicted_state(rtol_diffuse=np.inf)

    def test_filtered_state(self):
        if False:
            while True:
                i = 10
        super(TestVAR1Mixed_KFAS, self).test_filtered_state(rtol_diffuse=np.inf)

    def test_smoothed_state(self):
        if False:
            i = 10
            return i + 15
        super(TestVAR1Mixed_KFAS, self).test_smoothed_state(rtol_diffuse=np.inf)

class CheckDFM(CheckSSMResults):

    @classmethod
    def setup_class(cls, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        filter_univariate = kwargs.pop('filter_univariate', False)
        (cls.mod, cls.ssm) = model_dfm(**kwargs)
        if filter_univariate:
            cls.ssm.filter_univariate = True
        cls.results_a = cls.ssm.smooth()
        cls.d = cls.results_a.nobs_diffuse

    def test_nobs_diffuse(self):
        if False:
            for i in range(10):
                print('nop')
        assert_allclose(self.d, 2)

    def test_initialization(self):
        if False:
            return 10
        assert_allclose(self.results_a.initial_state_cov, 0)
        assert_allclose(self.results_a.initial_diffuse_state_cov, np.eye(2))

class TestDFM_Approx(CheckApproximateDiffuseMixin, CheckDFM):
    approximate_diffuse_variance = 50000000000.0

class TestDFM_KFAS(CheckKFASMixin, CheckDFM):
    results_path = os.path.join(current_path, 'results', 'results_exact_initial_dfm_R.csv')

    def test_predicted_state(self):
        if False:
            i = 10
            return i + 15
        super(TestDFM_KFAS, self).test_predicted_state(rtol_diffuse=np.inf)

    def test_filtered_state(self):
        if False:
            return 10
        super(TestDFM_KFAS, self).test_filtered_state(rtol_diffuse=np.inf)

    def test_smoothed_state(self):
        if False:
            return 10
        super(TestDFM_KFAS, self).test_smoothed_state(rtol_diffuse=np.inf)

class CheckDFMCollapsed(CheckSSMResults):

    @classmethod
    def setup_class(cls, **kwargs):
        if False:
            i = 10
            return i + 15
        filter_univariate = kwargs.pop('filter_univariate', True)
        (cls.mod, cls.ssm) = model_dfm(factor_order=1, **kwargs)
        if filter_univariate:
            cls.ssm.filter_univariate = True
        cls.ssm.filter_collapsed = True
        cls.results_a = cls.ssm.smooth()
        cls.d = cls.results_a.nobs_diffuse

    def test_nobs_diffuse(self):
        if False:
            return 10
        assert_allclose(self.d, 1)

    def test_initialization(self):
        if False:
            for i in range(10):
                print('nop')
        assert_allclose(self.results_a.initial_state_cov, 0)
        assert_allclose(self.results_a.initial_diffuse_state_cov, np.eye(1))

class TestDFMCollapsed_Approx(CheckApproximateDiffuseMixin, CheckDFMCollapsed):
    approximate_diffuse_variance = 1000000000.0

@pytest.mark.xfail
def test_irrelevant_state():
    if False:
        i = 10
        return i + 15
    endog = macrodata.infl
    spec = {'freq_seasonal': [{'period': 8, 'harmonics': 6}, {'period': 36, 'harmonics': 6}]}
    mod = UnobservedComponents(endog, 'llevel', **spec)
    mod.ssm.initialization = Initialization(mod.k_states, 'approximate_diffuse')
    res = mod.smooth([3.4, 7.2, 0.01, 0.01])
    mod2 = UnobservedComponents(endog, 'llevel', **spec)
    mod2.ssm.filter_univariate = True
    mod2.ssm.initialization = Initialization(mod2.k_states, 'diffuse')
    res2 = mod2.smooth([3.4, 7.2, 0.01, 0.01])
    assert_allclose(res.filtered_state[0, 25:], res2.filtered_state[0, 25:], atol=1e-05)

def test_nondiagonal_obs_cov(reset_randomstate):
    if False:
        return 10
    mod = TVSS(np.zeros((10, 2)))
    res1 = mod.smooth([])
    mod.ssm.filter_univariate = True
    res2 = mod.smooth([])
    atol = 0.002 if PLATFORM_WIN else 1e-05
    rtol = 0.002 if PLATFORM_WIN else 0.0001
    assert_allclose(res1.llf, res2.llf, rtol=rtol, atol=atol)
    assert_allclose(res1.forecasts[0], res2.forecasts[0], rtol=rtol, atol=atol)
    assert_allclose(res1.filtered_state, res2.filtered_state, rtol=rtol, atol=atol)
    assert_allclose(res1.filtered_state_cov, res2.filtered_state_cov, rtol=rtol, atol=atol)
    assert_allclose(res1.smoothed_state, res2.smoothed_state, rtol=rtol, atol=atol)
    assert_allclose(res1.smoothed_state_cov, res2.smoothed_state_cov, rtol=rtol, atol=atol)