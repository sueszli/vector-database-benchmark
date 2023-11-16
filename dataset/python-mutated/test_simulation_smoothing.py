"""
Tests for simulation smoothing

Author: Chad Fulton
License: Simplified-BSD
"""
import os
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, structural, varmax
from statsmodels.tsa.statespace.simulation_smoother import SIMULATION_STATE, SIMULATION_DISTURBANCE, SIMULATION_ALL
current_path = os.path.dirname(os.path.abspath(__file__))

class MultivariateVARKnown:
    """
    Tests for simulation smoothing values in a couple of special cases of
    variates. Both computed values and KFAS values are used for comparison
    against the simulation smoother output.
    """

    @classmethod
    def setup_class(cls, missing=None, test_against_KFAS=True, *args, **kwargs):
        if False:
            return 10
        cls.test_against_KFAS = test_against_KFAS
        dta = datasets.macrodata.load_pandas().data
        dta.index = pd.date_range(start='1959-01-01', end='2009-7-01', freq='QS')
        obs = np.log(dta[['realgdp', 'realcons', 'realinv']]).diff().iloc[1:]
        if missing == 'all':
            obs.iloc[0:50, :] = np.nan
        elif missing == 'partial':
            obs.iloc[0:50, 0] = np.nan
        elif missing == 'mixed':
            obs.iloc[0:50, 0] = np.nan
            obs.iloc[19:70, 1] = np.nan
            obs.iloc[39:90, 2] = np.nan
            obs.iloc[119:130, 0] = np.nan
            obs.iloc[119:130, 2] = np.nan
            obs.iloc[-10:, :] = np.nan
        if test_against_KFAS:
            obs = obs.iloc[:9]
        mod = mlemodel.MLEModel(obs, k_states=3, k_posdef=3, **kwargs)
        mod['design'] = np.eye(3)
        mod['obs_cov'] = np.array([[6.40649e-05, 0.0, 0.0], [0.0, 5.72802e-05, 0.0], [0.0, 0.0, 0.0017088585]])
        mod['transition'] = np.array([[-0.1119908792, 0.8441841604, 0.0238725303], [0.2629347724, 0.4996718412, -0.0173023305], [-3.2192369082, 4.1536028244, 0.4514379215]])
        mod['selection'] = np.eye(3)
        mod['state_cov'] = np.array([[6.40649e-05, 3.88496e-05, 0.0002148769], [3.88496e-05, 5.72802e-05, 1.555e-06], [0.0002148769, 1.555e-06, 0.0017088585]])
        mod.initialize_approximate_diffuse(1000000.0)
        mod.ssm.filter_univariate = True
        cls.model = mod
        cls.results = mod.smooth([], return_ssm=True)
        cls.sim = cls.model.simulation_smoother()

    def test_loglike(self):
        if False:
            i = 10
            return i + 15
        assert_allclose(np.sum(self.results.llf_obs), self.true_llf)

    def test_simulate_0(self):
        if False:
            while True:
                i = 10
        n = 10
        measurement_shocks = np.zeros((n, self.model.k_endog))
        state_shocks = np.zeros((n, self.model.ssm.k_posdef))
        initial_state = np.zeros(self.model.k_states)
        (obs, states) = self.model.ssm.simulate(nsimulations=n, measurement_shocks=measurement_shocks, state_shocks=state_shocks, initial_state=initial_state)
        assert_allclose(obs, np.zeros((n, self.model.k_endog)))
        assert_allclose(states, np.zeros((n, self.model.k_states)))

    def test_simulate_1(self):
        if False:
            return 10
        n = 10
        measurement_shocks = np.reshape(np.arange(n * self.model.k_endog) / 10.0, (n, self.model.k_endog))
        state_shocks = np.zeros((n, self.model.ssm.k_posdef))
        initial_state = np.zeros(self.model.k_states)
        (obs, states) = self.model.ssm.simulate(nsimulations=n, measurement_shocks=measurement_shocks, state_shocks=state_shocks, initial_state=initial_state)
        assert_allclose(obs, np.reshape(np.arange(n * self.model.k_endog) / 10.0, (n, self.model.k_endog)))
        assert_allclose(states, np.zeros((n, self.model.k_states)))

    def test_simulate_2(self):
        if False:
            i = 10
            return i + 15
        n = 10
        Z = self.model['design']
        T = self.model['transition']
        measurement_shocks = np.zeros((n, self.model.k_endog))
        state_shocks = np.ones((n, self.model.ssm.k_posdef))
        initial_state = np.ones(self.model.k_states) * 2.5
        (obs, states) = self.model.ssm.simulate(nsimulations=n, measurement_shocks=measurement_shocks, state_shocks=state_shocks, initial_state=initial_state)
        desired_obs = np.zeros((n, self.model.k_endog))
        desired_state = np.zeros((n, self.model.k_states))
        desired_state[0] = initial_state
        desired_obs[0] = np.dot(Z, initial_state)
        for i in range(1, n):
            desired_state[i] = np.dot(T, desired_state[i - 1]) + state_shocks[i]
            desired_obs[i] = np.dot(Z, desired_state[i])
        assert_allclose(obs, desired_obs)
        assert_allclose(states, desired_state)

    def test_simulation_smoothing_0(self):
        if False:
            for i in range(10):
                print('nop')
        sim = self.sim
        Z = self.model['design']
        nobs = self.model.nobs
        k_endog = self.model.k_endog
        k_posdef = self.model.ssm.k_posdef
        k_states = self.model.k_states
        sim.simulate(measurement_disturbance_variates=np.zeros(nobs * k_endog), state_disturbance_variates=np.zeros(nobs * k_posdef), initial_state_variates=np.zeros(k_states))
        assert_allclose(sim.generated_measurement_disturbance, 0)
        assert_allclose(sim.generated_state_disturbance, 0)
        assert_allclose(sim.generated_state, 0)
        assert_allclose(sim.generated_obs, 0)
        assert_allclose(sim.simulated_state, self.results.smoothed_state)
        if not self.model.ssm.filter_collapsed:
            assert_allclose(sim.simulated_measurement_disturbance, self.results.smoothed_measurement_disturbance)
        assert_allclose(sim.simulated_state_disturbance, self.results.smoothed_state_disturbance)
        if self.test_against_KFAS:
            path = os.path.join(current_path, 'results', 'results_simulation_smoothing0.csv')
            true = pd.read_csv(path)
            assert_allclose(sim.simulated_state, true[['state1', 'state2', 'state3']].T, atol=1e-07)
            assert_allclose(sim.simulated_measurement_disturbance, true[['eps1', 'eps2', 'eps3']].T, atol=1e-07)
            assert_allclose(sim.simulated_state_disturbance, true[['eta1', 'eta2', 'eta3']].T, atol=1e-07)
            signals = np.zeros((3, self.model.nobs))
            for t in range(self.model.nobs):
                signals[:, t] = np.dot(Z, sim.simulated_state[:, t])
            assert_allclose(signals, true[['signal1', 'signal2', 'signal3']].T, atol=1e-07)

    def test_simulation_smoothing_1(self):
        if False:
            print('Hello World!')
        sim = self.sim
        Z = self.model['design']
        nobs = self.model.nobs
        k_endog = self.model.k_endog
        k_posdef = self.model.ssm.k_posdef
        k_states = self.model.k_states
        measurement_disturbance_variates = np.reshape(np.arange(nobs * k_endog) / 10.0, (nobs, k_endog))
        state_disturbance_variates = np.zeros(nobs * k_posdef)
        generated_measurement_disturbance = np.zeros(measurement_disturbance_variates.shape)
        chol = np.linalg.cholesky(self.model['obs_cov'])
        for t in range(self.model.nobs):
            generated_measurement_disturbance[t] = np.dot(chol, measurement_disturbance_variates[t])
        y = generated_measurement_disturbance.copy()
        y[np.isnan(self.model.endog)] = np.nan
        generated_model = mlemodel.MLEModel(y, k_states=k_states, k_posdef=k_posdef)
        for name in ['design', 'obs_cov', 'transition', 'selection', 'state_cov']:
            generated_model[name] = self.model[name]
        generated_model.initialize_approximate_diffuse(1000000.0)
        generated_model.ssm.filter_univariate = True
        generated_res = generated_model.ssm.smooth()
        simulated_state = 0 - generated_res.smoothed_state + self.results.smoothed_state
        if not self.model.ssm.filter_collapsed:
            simulated_measurement_disturbance = generated_measurement_disturbance.T - generated_res.smoothed_measurement_disturbance + self.results.smoothed_measurement_disturbance
        simulated_state_disturbance = 0 - generated_res.smoothed_state_disturbance + self.results.smoothed_state_disturbance
        sim.simulate(measurement_disturbance_variates=measurement_disturbance_variates, state_disturbance_variates=state_disturbance_variates, initial_state_variates=np.zeros(k_states))
        assert_allclose(sim.generated_measurement_disturbance, generated_measurement_disturbance)
        assert_allclose(sim.generated_state_disturbance, 0)
        assert_allclose(sim.generated_state, 0)
        assert_allclose(sim.generated_obs, generated_measurement_disturbance.T)
        assert_allclose(sim.simulated_state, simulated_state)
        if not self.model.ssm.filter_collapsed:
            assert_allclose(sim.simulated_measurement_disturbance, simulated_measurement_disturbance)
        assert_allclose(sim.simulated_state_disturbance, simulated_state_disturbance)
        if self.test_against_KFAS:
            path = os.path.join(current_path, 'results', 'results_simulation_smoothing1.csv')
            true = pd.read_csv(path)
            assert_allclose(sim.simulated_state, true[['state1', 'state2', 'state3']].T, atol=1e-07)
            assert_allclose(sim.simulated_measurement_disturbance, true[['eps1', 'eps2', 'eps3']].T, atol=1e-07)
            assert_allclose(sim.simulated_state_disturbance, true[['eta1', 'eta2', 'eta3']].T, atol=1e-07)
            signals = np.zeros((3, self.model.nobs))
            for t in range(self.model.nobs):
                signals[:, t] = np.dot(Z, sim.simulated_state[:, t])
            assert_allclose(signals, true[['signal1', 'signal2', 'signal3']].T, atol=1e-07)

    def test_simulation_smoothing_2(self):
        if False:
            return 10
        sim = self.sim
        Z = self.model['design']
        T = self.model['transition']
        nobs = self.model.nobs
        k_endog = self.model.k_endog
        k_posdef = self.model.ssm.k_posdef
        k_states = self.model.k_states
        measurement_disturbance_variates = np.reshape(np.arange(nobs * k_endog) / 10.0, (nobs, k_endog))
        state_disturbance_variates = np.reshape(np.arange(nobs * k_posdef) / 10.0, (nobs, k_posdef))
        initial_state_variates = np.zeros(k_states)
        generated_measurement_disturbance = np.zeros(measurement_disturbance_variates.shape)
        chol = np.linalg.cholesky(self.model['obs_cov'])
        for t in range(self.model.nobs):
            generated_measurement_disturbance[t] = np.dot(chol, measurement_disturbance_variates[t])
        generated_state_disturbance = np.zeros(state_disturbance_variates.shape)
        chol = np.linalg.cholesky(self.model['state_cov'])
        for t in range(self.model.nobs):
            generated_state_disturbance[t] = np.dot(chol, state_disturbance_variates[t])
        generated_obs = np.zeros((self.model.k_endog, self.model.nobs))
        generated_state = np.zeros((self.model.k_states, self.model.nobs + 1))
        chol = np.linalg.cholesky(self.results.initial_state_cov)
        generated_state[:, 0] = self.results.initial_state + np.dot(chol, initial_state_variates)
        for t in range(self.model.nobs):
            generated_state[:, t + 1] = np.dot(T, generated_state[:, t]) + generated_state_disturbance.T[:, t]
            generated_obs[:, t] = np.dot(Z, generated_state[:, t]) + generated_measurement_disturbance.T[:, t]
        y = generated_obs.copy().T
        y[np.isnan(self.model.endog)] = np.nan
        generated_model = mlemodel.MLEModel(y, k_states=k_states, k_posdef=k_posdef)
        for name in ['design', 'obs_cov', 'transition', 'selection', 'state_cov']:
            generated_model[name] = self.model[name]
        generated_model.initialize_approximate_diffuse(1000000.0)
        generated_model.ssm.filter_univariate = True
        generated_res = generated_model.ssm.smooth()
        simulated_state = generated_state[:, :-1] - generated_res.smoothed_state + self.results.smoothed_state
        if not self.model.ssm.filter_collapsed:
            simulated_measurement_disturbance = generated_measurement_disturbance.T - generated_res.smoothed_measurement_disturbance + self.results.smoothed_measurement_disturbance
        simulated_state_disturbance = generated_state_disturbance.T - generated_res.smoothed_state_disturbance + self.results.smoothed_state_disturbance
        sim.simulate(measurement_disturbance_variates=measurement_disturbance_variates, state_disturbance_variates=state_disturbance_variates, initial_state_variates=np.zeros(k_states))
        assert_allclose(sim.generated_measurement_disturbance, generated_measurement_disturbance)
        assert_allclose(sim.generated_state_disturbance, generated_state_disturbance)
        assert_allclose(sim.generated_state, generated_state)
        assert_allclose(sim.generated_obs, generated_obs)
        assert_allclose(sim.simulated_state, simulated_state, atol=1e-07)
        if not self.model.ssm.filter_collapsed:
            assert_allclose(sim.simulated_measurement_disturbance.T, simulated_measurement_disturbance.T)
        assert_allclose(sim.simulated_state_disturbance, simulated_state_disturbance)
        if self.test_against_KFAS:
            path = os.path.join(current_path, 'results', 'results_simulation_smoothing2.csv')
            true = pd.read_csv(path)
            assert_allclose(sim.simulated_state.T, true[['state1', 'state2', 'state3']], atol=1e-07)
            assert_allclose(sim.simulated_measurement_disturbance, true[['eps1', 'eps2', 'eps3']].T, atol=1e-07)
            assert_allclose(sim.simulated_state_disturbance, true[['eta1', 'eta2', 'eta3']].T, atol=1e-07)
            signals = np.zeros((3, self.model.nobs))
            for t in range(self.model.nobs):
                signals[:, t] = np.dot(Z, sim.simulated_state[:, t])
            assert_allclose(signals, true[['signal1', 'signal2', 'signal3']].T, atol=1e-07)

class TestMultivariateVARKnown(MultivariateVARKnown):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(TestMultivariateVARKnown, cls).setup_class()
        cls.true_llf = 39.01246166

class TestMultivariateVARKnownMissingAll(MultivariateVARKnown):
    """
    Notes
    -----
    Cannot test against KFAS because they have a different behavior for
    missing entries. When an entry is missing, KFAS does not draw a simulation
    smoothed value for that entry, whereas we draw from the unconditional
    distribution. It appears there is nothing to definitively recommend one
    approach over the other, but it makes it difficult to line up the variates
    correctly in order to replicate results.
    """

    @classmethod
    def setup_class(cls, *args, **kwargs):
        if False:
            print('Hello World!')
        super(TestMultivariateVARKnownMissingAll, cls).setup_class(missing='all', test_against_KFAS=False)
        cls.true_llf = 1305.739288

class TestMultivariateVARKnownMissingPartial(MultivariateVARKnown):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(TestMultivariateVARKnownMissingPartial, cls).setup_class(missing='partial', test_against_KFAS=False)
        cls.true_llf = 1518.449598

class TestMultivariateVARKnownMissingMixed(MultivariateVARKnown):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(TestMultivariateVARKnownMissingMixed, cls).setup_class(missing='mixed', test_against_KFAS=False)
        cls.true_llf = 1117.265303

class TestDFM(TestMultivariateVARKnown):
    test_against_KFAS = False

    @classmethod
    def setup_class(cls, which='none', *args, **kwargs):
        if False:
            return 10
        dta = datasets.macrodata.load_pandas().data
        dta.index = pd.date_range(start='1959-01-01', end='2009-7-01', freq='QS')
        levels = dta[['realgdp', 'realcons', 'realinv']]
        obs = np.log(levels).diff().iloc[1:] * 400
        if which == 'all':
            obs.iloc[:50, :] = np.nan
            obs.iloc[119:130, :] = np.nan
        elif which == 'partial':
            obs.iloc[0:50, 0] = np.nan
            obs.iloc[119:130, 0] = np.nan
        elif which == 'mixed':
            obs.iloc[0:50, 0] = np.nan
            obs.iloc[19:70, 1] = np.nan
            obs.iloc[39:90, 2] = np.nan
            obs.iloc[119:130, 0] = np.nan
            obs.iloc[119:130, 2] = np.nan
        mod = mlemodel.MLEModel(obs, k_states=2, k_posdef=2, **kwargs)
        mod['design'] = np.array([[-32.47143586, 17.33779024], [-7.40264169, 1.69279859], [-209.04702853, 125.2879374]])
        mod['obs_cov'] = np.diag(np.array([0.0622668, 1.95666886, 58.37473642]))
        mod['transition'] = np.array([[0.29935707, 0.33289005], [-0.7639868, 1.2844237]])
        mod['selection'] = np.eye(2)
        mod['state_cov'] = np.array([[1.2, -0.25], [-0.25, 1.1]])
        mod.initialize_approximate_diffuse(1000000.0)
        mod.ssm.filter_univariate = True
        mod.ssm.filter_collapsed = True
        cls.model = mod
        cls.results = mod.smooth([], return_ssm=True)
        cls.sim = cls.model.simulation_smoother()

    def test_loglike(self):
        if False:
            while True:
                i = 10
        pass

class MultivariateVAR:
    """
    More generic tests for simulation smoothing; use actual N(0,1) variates
    """

    @classmethod
    def setup_class(cls, missing='none', *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        dta = datasets.macrodata.load_pandas().data
        dta.index = pd.date_range(start='1959-01-01', end='2009-7-01', freq='QS')
        obs = np.log(dta[['realgdp', 'realcons', 'realinv']]).diff().iloc[1:]
        if missing == 'all':
            obs.iloc[0:50, :] = np.nan
        elif missing == 'partial':
            obs.iloc[0:50, 0] = np.nan
        elif missing == 'mixed':
            obs.iloc[0:50, 0] = np.nan
            obs.iloc[19:70, 1] = np.nan
            obs.iloc[39:90, 2] = np.nan
            obs.iloc[119:130, 0] = np.nan
            obs.iloc[119:130, 2] = np.nan
            obs.iloc[-10:, :] = np.nan
        mod = mlemodel.MLEModel(obs, k_states=3, k_posdef=3, **kwargs)
        mod['design'] = np.eye(3)
        mod['obs_cov'] = np.array([[6.40649e-05, 0.0, 0.0], [0.0, 5.72802e-05, 0.0], [0.0, 0.0, 0.0017088585]])
        mod['transition'] = np.array([[-0.1119908792, 0.8441841604, 0.0238725303], [0.2629347724, 0.4996718412, -0.0173023305], [-3.2192369082, 4.1536028244, 0.4514379215]])
        mod['selection'] = np.eye(3)
        mod['state_cov'] = np.array([[6.40649e-05, 3.88496e-05, 0.0002148769], [3.88496e-05, 5.72802e-05, 1.555e-06], [0.0002148769, 1.555e-06, 0.0017088585]])
        mod.initialize_approximate_diffuse(1000000.0)
        mod.ssm.filter_univariate = True
        cls.model = mod
        cls.results = mod.smooth([], return_ssm=True)
        cls.sim = cls.model.simulation_smoother()

    def test_loglike(self):
        if False:
            while True:
                i = 10
        assert_allclose(np.sum(self.results.llf_obs), self.true_llf)

    def test_simulation_smoothing(self):
        if False:
            return 10
        sim = self.sim
        Z = self.model['design']
        nobs = self.model.nobs
        k_endog = self.model.k_endog
        sim.simulate(measurement_disturbance_variates=self.variates[:nobs * k_endog], state_disturbance_variates=self.variates[nobs * k_endog:-3], initial_state_variates=self.variates[-3:])
        assert_allclose(sim.simulated_state.T, self.true[['state1', 'state2', 'state3']], atol=1e-07)
        assert_allclose(sim.simulated_measurement_disturbance, self.true[['eps1', 'eps2', 'eps3']].T, atol=1e-07)
        assert_allclose(sim.simulated_state_disturbance, self.true[['eta1', 'eta2', 'eta3']].T, atol=1e-07)
        signals = np.zeros((3, self.model.nobs))
        for t in range(self.model.nobs):
            signals[:, t] = np.dot(Z, sim.simulated_state[:, t])
        assert_allclose(signals, self.true[['signal1', 'signal2', 'signal3']].T, atol=1e-07)

class TestMultivariateVAR(MultivariateVAR):

    @classmethod
    def setup_class(cls):
        if False:
            return 10
        super(TestMultivariateVAR, cls).setup_class()
        path = os.path.join(current_path, 'results', 'results_simulation_smoothing3_variates.csv')
        cls.variates = pd.read_csv(path).values.squeeze()
        path = os.path.join(current_path, 'results', 'results_simulation_smoothing3.csv')
        cls.true = pd.read_csv(path)
        cls.true_llf = 1695.34872

def test_misc():
    if False:
        for i in range(10):
            print('nop')
    dta = datasets.macrodata.load_pandas().data
    dta.index = pd.date_range(start='1959-01-01', end='2009-7-01', freq='QS')
    obs = np.log(dta[['realgdp', 'realcons', 'realinv']]).diff().iloc[1:]
    mod = sarimax.SARIMAX(obs['realgdp'], order=(1, 0, 0))
    mod['design', 0, 0] = 0.0
    mod['obs_cov', 0, 0] = 1.0
    mod.update(np.r_[1.0, 1.0])
    sim = mod.simulation_smoother()
    rs = np.random.RandomState(1234)
    n_disturbance_variates = mod.nobs * (mod.k_endog + mod.k_states)
    variates = rs.normal(size=n_disturbance_variates)
    rs = np.random.RandomState(1234)
    sim.simulate(random_state=rs)
    assert_allclose(sim.generated_measurement_disturbance[:, 0], variates[:mod.nobs])
    assert_allclose(sim.generated_state_disturbance[:, 0], variates[mod.nobs:])
    assert_equal(sim.simulation_output, mod.ssm.smoother_output)
    sim.simulation_output = 0
    assert_equal(sim.simulation_output, 0)
    sim.simulate_state = True
    assert_equal(sim.simulation_output, SIMULATION_STATE)
    sim.simulate_state = False
    assert_equal(sim.simulation_output, 0)
    sim.simulate_disturbance = True
    assert_equal(sim.simulation_output, SIMULATION_DISTURBANCE)
    sim.simulate_disturbance = False
    assert_equal(sim.simulation_output, 0)
    sim.simulate_all = True
    assert_equal(sim.simulation_output, SIMULATION_ALL)
    sim.simulate_all = False
    assert_equal(sim.simulation_output, 0)

def test_simulation_smoothing_obs_intercept():
    if False:
        print('Hello World!')
    nobs = 10
    intercept = 100
    endog = np.ones(nobs) * intercept
    mod = structural.UnobservedComponents(endog, 'rwalk', exog=np.ones(nobs))
    mod.update([1, intercept])
    sim = mod.simulation_smoother()
    sim.simulate(measurement_disturbance_variates=np.zeros(mod.nobs), state_disturbance_variates=np.zeros(mod.nobs), initial_state_variates=np.zeros(1))
    assert_equal(sim.simulated_state[0], 0)

def test_simulation_smoothing_state_intercept():
    if False:
        print('Hello World!')
    nobs = 10
    intercept = 100
    endog = np.ones(nobs) * intercept
    mod = sarimax.SARIMAX(endog, order=(0, 0, 0), trend='c', measurement_error=True)
    mod.initialize_known([100], [[0]])
    mod.update([intercept, 1.0, 1.0])
    sim = mod.simulation_smoother()
    sim.simulate(measurement_disturbance_variates=np.zeros(mod.nobs), state_disturbance_variates=np.zeros(mod.nobs), initial_state_variates=np.zeros(1))
    assert_equal(sim.simulated_state[0], intercept)

def test_simulation_smoothing_state_intercept_diffuse():
    if False:
        print('Hello World!')
    nobs = 10
    intercept = 100
    endog = np.ones(nobs) * intercept
    mod = sarimax.SARIMAX(endog, order=(0, 0, 0), trend='c', measurement_error=True, initialization='diffuse')
    mod.update([intercept, 1.0, 1.0])
    sim = mod.simulation_smoother()
    sim.simulate(measurement_disturbance_variates=np.zeros(mod.nobs), state_disturbance_variates=np.zeros(mod.nobs), initial_state_variates=np.zeros(1))
    assert_equal(sim.simulated_state[0], intercept)
    endog[5] = np.nan
    mod = sarimax.SARIMAX(endog, order=(0, 0, 0), trend='c', measurement_error=True, initialization='diffuse')
    mod.update([intercept, 1.0, 1.0])
    sim = mod.simulation_smoother()
    sim.simulate(measurement_disturbance_variates=np.zeros(mod.nobs), state_disturbance_variates=np.zeros(mod.nobs), initial_state_variates=np.zeros(1))
    assert_equal(sim.simulated_state[0], intercept)

def test_deprecated_arguments_univariate():
    if False:
        i = 10
        return i + 15
    nobs = 10
    intercept = 100
    endog = np.ones(nobs) * intercept
    mod = sarimax.SARIMAX(endog, order=(0, 0, 0), trend='c', measurement_error=True, initialization='diffuse')
    mod.update([intercept, 0.5, 2.0])
    mds = np.arange(10) / 10.0
    sds = np.arange(10)[::-1] / 20.0
    sim = mod.simulation_smoother()
    sim.simulate(measurement_disturbance_variates=mds, state_disturbance_variates=sds, initial_state_variates=np.zeros(1))
    desired = sim.simulated_state[0]
    with pytest.warns(FutureWarning):
        sim.simulate(disturbance_variates=np.r_[mds, sds], initial_state_variates=np.zeros(1))
    actual = sim.simulated_state[0]
    sim = mod.simulation_smoother()
    sim.simulate(measurement_disturbance_variates=mds, state_disturbance_variates=sds, initial_state_variates=np.zeros(1), pretransformed_measurement_disturbance_variates=True, pretransformed_state_disturbance_variates=True)
    desired = sim.simulated_state[0]
    with pytest.warns(FutureWarning):
        sim.simulate(measurement_disturbance_variates=mds, state_disturbance_variates=sds, pretransformed=True, initial_state_variates=np.zeros(1))
    actual = sim.simulated_state[0]
    assert_allclose(actual, desired)

def test_deprecated_arguments_multivariate():
    if False:
        for i in range(10):
            print('nop')
    endog = np.array([[0.3, 1.4], [-0.1, 0.6], [0.2, 0.7], [0.1, 0.9], [0.5, -0.1]])
    mod = varmax.VARMAX(endog, order=(1, 0, 0))
    mod.update([1.2, 0.5, 0.8, 0.1, -0.2, 0.5, 5.2, 0.5, 8.1])
    mds = np.arange(10).reshape(5, 2) / 10.0
    sds = np.arange(10).reshape(5, 2)[::-1] / 20.0
    sim = mod.simulation_smoother()
    sim.simulate(measurement_disturbance_variates=mds, state_disturbance_variates=sds, initial_state_variates=np.zeros(2))
    desired = sim.simulated_state[0]
    with pytest.warns(FutureWarning):
        sim.simulate(disturbance_variates=np.r_[mds.ravel(), sds.ravel()], initial_state_variates=np.zeros(2))
    actual = sim.simulated_state[0]
    sim = mod.simulation_smoother()
    sim.simulate(measurement_disturbance_variates=mds, state_disturbance_variates=sds, initial_state_variates=np.zeros(2), pretransformed_measurement_disturbance_variates=True, pretransformed_state_disturbance_variates=True)
    desired = sim.simulated_state[0]
    with pytest.warns(FutureWarning):
        sim.simulate(measurement_disturbance_variates=mds, state_disturbance_variates=sds, pretransformed=True, initial_state_variates=np.zeros(2))
    actual = sim.simulated_state[0]
    assert_allclose(actual, desired)

def test_nan():
    if False:
        for i in range(10):
            print('nop')
    '\n    This is a very slow test to check that the distribution of simulated states\n    (from the posterior) is correct in the presense of NaN values. Here, it\n    checks the marginal distribution of the drawn states against the values\n    computed from the smoother and prints the result.\n\n    With the fixed simulation smoother, it prints:\n\n    True values:\n    [1.         0.66666667 0.66666667 1.        ]\n    [0.         0.95238095 0.95238095 0.        ]\n\n    Simulated values:\n    [1.         0.66699187 0.66456719 1.        ]\n    [0.       0.953608 0.953198 0.      ]\n\n    Previously, it would have printed:\n\n    True values:\n    [1.         0.66666667 0.66666667 1.        ]\n    [0.         0.95238095 0.95238095 0.        ]\n\n    Simulated values:\n    [1.         0.66666667 0.66666667 1.        ]\n    [0. 0. 0. 0.]\n    '
    return
    mod = sarimax.SARIMAX([1, np.nan, np.nan, 1], order=(1, 0, 0), trend='c')
    res = mod.smooth([0, 0.5, 1.0])
    rs = np.random.RandomState(1234)
    sim = mod.simulation_smoother(random_state=rs)
    n = 1000000
    out = np.zeros((n, mod.nobs))
    for i in range(n):
        sim.simulate()
        out[i] = sim.simulated_state
    print('True values:')
    print(res.smoothed_state[0])
    print(res.smoothed_state_cov[0, 0])
    print()
    print('Simulated values:')
    print(np.mean(out, axis=0))
    print(np.var(out, axis=0).round(6))