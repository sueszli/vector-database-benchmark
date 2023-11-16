"""
Tests for CFA simulation smoothing

Author: Chad Fulton
License: BSD-3
"""
import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from scipy.linalg import cho_solve_banded
from statsmodels import datasets
from statsmodels.tsa.statespace import sarimax, structural, dynamic_factor, varmax
current_path = os.path.dirname(os.path.abspath(__file__))
dta = datasets.macrodata.load_pandas().data
dta.index = pd.period_range('1959Q1', '2009Q3', freq='Q')
dta = np.log(dta[['realcons', 'realgdp', 'cpi']]).diff().iloc[1:] * 400

class CheckPosteriorMoments:

    @classmethod
    def setup_class(cls, model_class, missing=None, mean_atol=0, cov_atol=0, use_complex=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        cls.mean_atol = mean_atol
        cls.cov_atol = cov_atol
        endog = dta.copy()
        if missing == 'all':
            endog.iloc[0:50, :] = np.nan
        elif missing == 'partial':
            endog.iloc[0:50, 0] = np.nan
        elif missing == 'mixed':
            endog.iloc[0:50, 0] = np.nan
            endog.iloc[19:70, 1] = np.nan
            endog.iloc[39:90, 2] = np.nan
            endog.iloc[119:130, 0] = np.nan
            endog.iloc[119:130, 2] = np.nan
            endog.iloc[-10:, :] = np.nan
        if model_class in [sarimax.SARIMAX, structural.UnobservedComponents]:
            endog = endog.iloc[:, 2]
        cls.mod = model_class(endog, *args, **kwargs)
        params = cls.mod.start_params
        if use_complex:
            params = params + 0j
        cls.res = cls.mod.smooth(params)
        cls.sim_cfa = cls.mod.simulation_smoother(method='cfa')
        cls.sim_cfa.simulate()
        prefix = 'z' if use_complex else 'd'
        cls._sim_cfa = cls.sim_cfa._simulation_smoothers[prefix]

    def test_posterior_mean(self):
        if False:
            i = 10
            return i + 15
        actual = np.array(self._sim_cfa.posterior_mean, copy=True)
        assert_allclose(actual, self.res.smoothed_state, atol=self.mean_atol)
        assert_allclose(self.sim_cfa.posterior_mean, self.res.smoothed_state, atol=self.mean_atol)

    def test_posterior_cov(self):
        if False:
            return 10
        inv_chol = np.array(self._sim_cfa.posterior_cov_inv_chol, copy=True)
        actual = cho_solve_banded((inv_chol, True), np.eye(inv_chol.shape[1]))
        for t in range(self.mod.nobs):
            tm = t * self.mod.k_states
            t1m = tm + self.mod.k_states
            assert_allclose(actual[tm:t1m, tm:t1m], self.res.smoothed_state_cov[..., t], atol=self.cov_atol)
        actual = self.sim_cfa.posterior_cov
        for t in range(self.mod.nobs):
            tm = t * self.mod.k_states
            t1m = tm + self.mod.k_states
            assert_allclose(actual[tm:t1m, tm:t1m], self.res.smoothed_state_cov[..., t], atol=self.cov_atol)

class TestDFM(CheckPosteriorMoments):

    @classmethod
    def setup_class(cls, missing=None, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        kwargs['k_factors'] = 1
        kwargs['factor_order'] = 1
        super().setup_class(dynamic_factor.DynamicFactor, *args, missing=missing, **kwargs)

class TestDFMComplex(CheckPosteriorMoments):

    @classmethod
    def setup_class(cls, missing=None, *args, **kwargs):
        if False:
            while True:
                i = 10
        kwargs['k_factors'] = 1
        kwargs['factor_order'] = 1
        super().setup_class(dynamic_factor.DynamicFactor, *args, missing=missing, use_complex=True, **kwargs)

class TestDFMAllMissing(TestDFM):

    def setup_class(cls, missing='all', *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().setup_class(*args, missing=missing, **kwargs)

class TestDFMPartialMissing(TestDFM):

    def setup_class(cls, missing='partial', *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().setup_class(*args, missing=missing, **kwargs)

class TestDFMMixedMissing(TestDFM):

    def setup_class(cls, missing='mixed', *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().setup_class(*args, missing=missing, **kwargs)

class TestVARME(CheckPosteriorMoments):

    @classmethod
    def setup_class(cls, missing=None, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        kwargs['order'] = (1, 0)
        kwargs['measurement_error'] = True
        super().setup_class(varmax.VARMAX, *args, missing=missing, **kwargs)

class TestVARMEAllMissing(TestVARME):

    def setup_class(cls, missing='all', *args, **kwargs):
        if False:
            while True:
                i = 10
        super().setup_class(*args, missing=missing, **kwargs)

class TestVARMEPartialMissing(TestVARME):

    def setup_class(cls, missing='partial', *args, **kwargs):
        if False:
            print('Hello World!')
        super().setup_class(*args, missing=missing, **kwargs)

class TestVARMEMixedMissing(TestVARME):

    def setup_class(cls, missing='mixed', *args, **kwargs):
        if False:
            return 10
        super().setup_class(*args, missing=missing, **kwargs)

class TestSARIMAXME(CheckPosteriorMoments):

    @classmethod
    def setup_class(cls, missing=None, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        kwargs['order'] = (1, 0, 0)
        kwargs['measurement_error'] = True
        super().setup_class(sarimax.SARIMAX, *args, missing=missing, **kwargs)

class TestSARIMAXMEMissing(TestSARIMAXME):

    def setup_class(cls, missing='mixed', *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().setup_class(*args, missing=missing, **kwargs)

class TestUnobservedComponents(CheckPosteriorMoments):

    @classmethod
    def setup_class(cls, missing=None, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        kwargs['level'] = 'llevel'
        kwargs['exog'] = np.arange(dta.shape[0])
        kwargs['autoregressive'] = 1
        super().setup_class(structural.UnobservedComponents, *args, missing=missing, **kwargs)

class TestUnobservedComponentsMissing(TestUnobservedComponents):

    def setup_class(cls, missing='mixed', *args, **kwargs):
        if False:
            return 10
        super().setup_class(*args, missing=missing, **kwargs)

def test_dfm(missing=None):
    if False:
        for i in range(10):
            print('nop')
    mod = dynamic_factor.DynamicFactor(dta, k_factors=2, factor_order=1)
    mod.update(mod.start_params)
    sim_cfa = mod.simulation_smoother(method='cfa')
    res = mod.ssm.smooth()
    sim_cfa.simulate(np.zeros((mod.k_states, mod.nobs)))
    assert_allclose(sim_cfa.simulated_state, res.smoothed_state)