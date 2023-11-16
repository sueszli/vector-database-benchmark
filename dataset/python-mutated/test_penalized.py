"""
Created on Sun May 10 12:39:33 2015

Author: Josef Perktold
License: BSD-3
"""
import warnings
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels.discrete.discrete_model import Poisson, Logit, Probit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import family
from statsmodels.sandbox.regression.penalized import TheilGLS
from statsmodels.base._penalized import PenalizedMixin
import statsmodels.base._penalties as smpen

class PoissonPenalized(PenalizedMixin, Poisson):
    pass

class LogitPenalized(PenalizedMixin, Logit):
    pass

class ProbitPenalized(PenalizedMixin, Probit):
    pass

class GLMPenalized(PenalizedMixin, GLM):
    pass

class CheckPenalizedPoisson:

    @classmethod
    def setup_class(cls):
        if False:
            i = 10
            return i + 15
        np.random.seed(987865)
        (nobs, k_vars) = (500, 10)
        k_nonzero = 4
        x = (np.random.rand(nobs, k_vars) + 0.5 * (np.random.rand(nobs, 1) - 0.5)) * 2 - 1
        x *= 1.2
        x[:, 0] = 1
        beta = np.zeros(k_vars)
        beta[:k_nonzero] = 1.0 / np.arange(1, k_nonzero + 1)
        linpred = x.dot(beta)
        y = cls._generate_endog(linpred)
        cls.k_nonzero = k_nonzero
        cls.x = x
        cls.y = y
        cls.rtol = 0.0001
        cls.atol = 1e-06
        cls.exog_index = slice(None, None, None)
        cls.k_params = k_vars
        cls.skip_hessian = False
        cls.penalty = smpen.SCADSmoothed(0.1, c0=0.0001)
        cls._initialize()

    @classmethod
    def _generate_endog(cls, linpred):
        if False:
            for i in range(10):
                print('nop')
        mu = np.exp(linpred)
        np.random.seed(999)
        y = np.random.poisson(mu)
        return y

    def test_params_table(self):
        if False:
            print('Hello World!')
        res1 = self.res1
        res2 = self.res2
        assert_equal((res1.params != 0).sum(), self.k_params)
        assert_allclose(res1.params[self.exog_index], res2.params, rtol=self.rtol, atol=self.atol)
        assert_allclose(res1.bse[self.exog_index], res2.bse, rtol=self.rtol, atol=self.atol)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            assert_allclose(res1.pvalues[self.exog_index], res2.pvalues, rtol=self.rtol, atol=self.atol)
        assert_allclose(res1.predict(), res2.predict(), rtol=0.05)

    @pytest.mark.smoke
    def test_summary(self):
        if False:
            i = 10
            return i + 15
        self.res1.summary()

    @pytest.mark.smoke
    def test_summary2(self):
        if False:
            while True:
                i = 10
        summ = self.res1.summary2()
        assert isinstance(summ.as_latex(), str)
        assert isinstance(summ.as_html(), str)
        assert isinstance(summ.as_text(), str)

    def test_numdiff(self):
        if False:
            return 10
        res1 = self.res1
        p = res1.params * 0.98
        kwds = {'scale': 1} if isinstance(res1.model, GLM) else {}
        assert_allclose(res1.model.score(p, **kwds)[self.exog_index], res1.model.score_numdiff(p, **kwds)[self.exog_index], rtol=0.025)
        if not self.skip_hessian:
            if isinstance(self.exog_index, slice):
                idx1 = idx2 = self.exog_index
            else:
                idx1 = self.exog_index[:, None]
                idx2 = self.exog_index
            h1 = res1.model.hessian(res1.params, **kwds)[idx1, idx2]
            h2 = res1.model.hessian_numdiff(res1.params, **kwds)[idx1, idx2]
            assert_allclose(h1, h2, rtol=0.02)

class TestPenalizedPoissonNonePenal(CheckPenalizedPoisson):

    @classmethod
    def _initialize(cls):
        if False:
            for i in range(10):
                print('nop')
        (y, x) = (cls.y, cls.x)
        modp = Poisson(y, x)
        cls.res2 = modp.fit(disp=0)
        mod = PoissonPenalized(y, x)
        mod.pen_weight = 0
        cls.res1 = mod.fit(method='bfgs', maxiter=100, disp=0)
        cls.atol = 5e-06

class TestPenalizedPoissonNoPenal(CheckPenalizedPoisson):

    @classmethod
    def _initialize(cls):
        if False:
            for i in range(10):
                print('nop')
        (y, x) = (cls.y, cls.x)
        modp = Poisson(y, x)
        cls.res2 = modp.fit(disp=0)
        mod = PoissonPenalized(y, x)
        mod.pen_weight = 0
        cls.res1 = mod.fit(method='bfgs', maxiter=100, disp=0)
        cls.atol = 5e-06

class TestPenalizedGLMPoissonNoPenal(CheckPenalizedPoisson):

    @classmethod
    def _initialize(cls):
        if False:
            print('Hello World!')
        (y, x) = (cls.y, cls.x)
        modp = GLM(y, x, family=family.Poisson())
        cls.res2 = modp.fit()
        mod = GLMPenalized(y, x, family=family.Poisson(), penal=cls.penalty)
        mod.pen_weight = 0
        cls.res1 = mod.fit(method='bfgs', maxiter=100, disp=0)
        cls.atol = 5e-06

class TestPenalizedPoissonOracle(CheckPenalizedPoisson):

    @classmethod
    def _initialize(cls):
        if False:
            i = 10
            return i + 15
        (y, x) = (cls.y, cls.x)
        modp = Poisson(y, x[:, :cls.k_nonzero])
        cls.res2 = modp.fit(disp=0)
        mod = PoissonPenalized(y, x, penal=cls.penalty)
        mod.pen_weight *= 1.5
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(method='bfgs', maxiter=100, disp=0)
        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.atol = 0.005

class TestPenalizedGLMPoissonOracle(CheckPenalizedPoisson):

    @classmethod
    def _initialize(cls):
        if False:
            for i in range(10):
                print('nop')
        (y, x) = (cls.y, cls.x)
        modp = GLM(y, x[:, :cls.k_nonzero], family=family.Poisson())
        cls.res2 = modp.fit()
        mod = GLMPenalized(y, x, family=family.Poisson(), penal=cls.penalty)
        mod.pen_weight *= 1.5
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(method='bfgs', maxiter=100)
        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.atol = 0.005

class TestPenalizedPoissonOracleHC(CheckPenalizedPoisson):

    @classmethod
    def _initialize(cls):
        if False:
            return 10
        (y, x) = (cls.y, cls.x)
        cov_type = 'HC0'
        modp = Poisson(y, x[:, :cls.k_nonzero])
        cls.res2 = modp.fit(cov_type=cov_type, method='bfgs', maxiter=100, disp=0)
        mod = PoissonPenalized(y, x, penal=cls.penalty)
        mod.pen_weight *= 1.5
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(cov_type=cov_type, method='bfgs', maxiter=100, disp=0)
        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.atol = 0.005

    def test_cov_type(self):
        if False:
            print('Hello World!')
        res1 = self.res1
        res2 = self.res2
        assert_equal(self.res1.cov_type, 'HC0')
        cov_kwds = {'description': 'Standard Errors are heteroscedasticity robust (HC0)', 'adjust_df': False, 'use_t': False, 'scaling_factor': None}
        assert_equal(self.res1.cov_kwds, cov_kwds)
        params = np.array([0.9681778757470111, 0.43674374940137434, 0.33096260487556745, 0.27415680046693747])
        bse = np.array([0.028126650444581985, 0.03309998456428315, 0.033184585514904545, 0.0342825041305033])
        assert_allclose(res2.params[:self.k_nonzero], params, atol=1e-05)
        assert_allclose(res2.bse[:self.k_nonzero], bse, rtol=1e-06)
        assert_allclose(res1.params[:self.k_nonzero], params, atol=self.atol)
        assert_allclose(res1.bse[:self.k_nonzero], bse, rtol=0.02)

class TestPenalizedGLMPoissonOracleHC(CheckPenalizedPoisson):

    @classmethod
    def _initialize(cls):
        if False:
            i = 10
            return i + 15
        (y, x) = (cls.y, cls.x)
        cov_type = 'HC0'
        modp = GLM(y, x[:, :cls.k_nonzero], family=family.Poisson())
        cls.res2 = modp.fit(cov_type=cov_type, method='bfgs', maxiter=100, disp=0)
        mod = GLMPenalized(y, x, family=family.Poisson(), penal=cls.penalty)
        mod.pen_weight *= 1.5
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(cov_type=cov_type, method='bfgs', maxiter=100, disp=0)
        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.atol = 0.005

class TestPenalizedPoissonGLMOracleHC(CheckPenalizedPoisson):

    @classmethod
    def _initialize(cls):
        if False:
            for i in range(10):
                print('nop')
        (y, x) = (cls.y, cls.x)
        cov_type = 'HC0'
        modp = PoissonPenalized(y, x, penal=cls.penalty)
        modp.pen_weight *= 1.5
        modp.penal.tau = 0.05
        cls.res2 = modp.fit(cov_type=cov_type, method='bfgs', maxiter=100, disp=0)
        mod = GLMPenalized(y, x, family=family.Poisson(), penal=cls.penalty)
        mod.pen_weight *= 1.5
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(cov_type=cov_type, method='bfgs', maxiter=100, disp=0)
        cls.exog_index = slice(None, None, None)
        cls.atol = 0.0001

class TestPenalizedPoissonOraclePenalized(CheckPenalizedPoisson):

    @classmethod
    def _initialize(cls):
        if False:
            while True:
                i = 10
        (y, x) = (cls.y, cls.x)
        modp = PoissonPenalized(y, x[:, :cls.k_nonzero], penal=cls.penalty)
        cls.res2 = modp.fit(method='bfgs', maxiter=100, disp=0)
        mod = PoissonPenalized(y, x, penal=cls.penalty)
        cls.res1 = mod.fit(method='bfgs', maxiter=100, trim=False, disp=0)
        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.atol = 0.001

class TestPenalizedPoissonOraclePenalized2(CheckPenalizedPoisson):

    @classmethod
    def _initialize(cls):
        if False:
            i = 10
            return i + 15
        (y, x) = (cls.y, cls.x)
        modp = PoissonPenalized(y, x[:, :cls.k_nonzero], penal=cls.penalty)
        modp.pen_weight *= 10
        modp.penal.tau = 0.05
        sp2 = np.array([0.96817921, 0.43673551, 0.33096011, 0.27416614])
        cls.res2 = modp.fit(start_params=sp2 * 0.5, method='bfgs', maxiter=100, disp=0)
        params_notrim = np.array([0.968178874, 0.436744981, 0.330965041, 0.274161883, -2.58988461e-06, -1.2435264e-06, 4.48584458e-08, -2.46876149e-06, -1.02471074e-05, -4.39248098e-06])
        mod = PoissonPenalized(y, x, penal=cls.penalty)
        mod.pen_weight *= 10
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(start_params=params_notrim * 0.5, method='bfgs', maxiter=100, trim=True, disp=0)
        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.atol = 1e-08
        cls.k_params = cls.k_nonzero

    def test_zeros(self):
        if False:
            print('Hello World!')
        assert_equal(self.res1.params[self.k_nonzero:], 0)
        assert_equal(self.res1.bse[self.k_nonzero:], 0)

class TestPenalizedPoissonOraclePenalized2HC(CheckPenalizedPoisson):

    @classmethod
    def _initialize(cls):
        if False:
            while True:
                i = 10
        (y, x) = (cls.y, cls.x)
        cov_type = 'HC0'
        modp = PoissonPenalized(y, x[:, :cls.k_nonzero], penal=cls.penalty)
        modp.pen_weight *= 10
        modp.penal.tau = 0.05
        sp2 = np.array([0.96817921, 0.43673551, 0.33096011, 0.27416614])
        cls.res2 = modp.fit(start_params=sp2 * 0.5, cov_type=cov_type, method='bfgs', maxiter=100, disp=0)
        params_notrim = np.array([0.968178874, 0.436744981, 0.330965041, 0.274161883, -2.58988461e-06, -1.2435264e-06, 4.48584458e-08, -2.46876149e-06, -1.02471074e-05, -4.39248098e-06])
        mod = PoissonPenalized(y, x, penal=cls.penalty)
        mod.pen_weight *= 10
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(start_params=params_notrim * 0.5, cov_type=cov_type, method='bfgs', maxiter=100, trim=True, disp=0)
        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.atol = 1e-12
        cls.k_params = cls.k_nonzero

    def test_cov_type(self):
        if False:
            print('Hello World!')
        res1 = self.res1
        res2 = self.res2
        assert_equal(self.res1.cov_type, 'HC0')
        assert_equal(self.res1.results_constrained.cov_type, 'HC0')
        cov_kwds = {'description': 'Standard Errors are heteroscedasticity robust (HC0)', 'adjust_df': False, 'use_t': False, 'scaling_factor': None}
        assert_equal(self.res1.cov_kwds, cov_kwds)
        assert_equal(self.res1.cov_kwds, self.res1.results_constrained.cov_kwds)
        params = np.array([0.9681778757470111, 0.43674374940137434, 0.33096260487556745, 0.27415680046693747])
        bse = np.array([0.028126650444581985, 0.03309998456428315, 0.033184585514904545, 0.0342825041305033])
        assert_allclose(res2.params[:self.k_nonzero], params, atol=1e-05)
        assert_allclose(res2.bse[:self.k_nonzero], bse, rtol=5e-06)
        assert_allclose(res1.params[:self.k_nonzero], params, atol=1e-05)
        assert_allclose(res1.bse[:self.k_nonzero], bse, rtol=5e-06)

class CheckPenalizedLogit(CheckPenalizedPoisson):

    @classmethod
    def _generate_endog(cls, linpred):
        if False:
            while True:
                i = 10
        mu = 1 / (1 + np.exp(-linpred + linpred.mean() - 0.5))
        np.random.seed(999)
        y = np.random.rand(len(mu)) < mu
        return y

class TestPenalizedLogitNoPenal(CheckPenalizedLogit):

    @classmethod
    def _initialize(cls):
        if False:
            for i in range(10):
                print('nop')
        (y, x) = (cls.y, cls.x)
        modp = Logit(y, x)
        cls.res2 = modp.fit(disp=0)
        mod = LogitPenalized(y, x, penal=cls.penalty)
        mod.pen_weight = 0
        cls.res1 = mod.fit(disp=0)
        cls.atol = 0.0001

class TestPenalizedLogitOracle(CheckPenalizedLogit):

    @classmethod
    def _initialize(cls):
        if False:
            return 10
        (y, x) = (cls.y, cls.x)
        modp = Logit(y, x[:, :cls.k_nonzero])
        cls.res2 = modp.fit(disp=0)
        mod = LogitPenalized(y, x, penal=cls.penalty)
        mod.pen_weight *= 0.5
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(method='bfgs', maxiter=100, disp=0)
        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.atol = 0.005

class TestPenalizedGLMLogitOracle(CheckPenalizedLogit):

    @classmethod
    def _initialize(cls):
        if False:
            while True:
                i = 10
        (y, x) = (cls.y, cls.x)
        modp = GLM(y, x[:, :cls.k_nonzero], family=family.Binomial())
        cls.res2 = modp.fit(disp=0)
        mod = GLMPenalized(y, x, family=family.Binomial(), penal=cls.penalty)
        mod.pen_weight *= 0.5
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(method='bfgs', maxiter=100, disp=0)
        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.atol = 0.005

class TestPenalizedLogitOraclePenalized(CheckPenalizedLogit):

    @classmethod
    def _initialize(cls):
        if False:
            while True:
                i = 10
        (y, x) = (cls.y, cls.x)
        modp = LogitPenalized(y, x[:, :cls.k_nonzero], penal=cls.penalty)
        cls.res2 = modp.fit(method='bfgs', maxiter=100, disp=0)
        mod = LogitPenalized(y, x, penal=cls.penalty)
        cls.res1 = mod.fit(method='bfgs', maxiter=100, trim=False, disp=0)
        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.atol = 0.001

class TestPenalizedLogitOraclePenalized2(CheckPenalizedLogit):

    @classmethod
    def _initialize(cls):
        if False:
            i = 10
            return i + 15
        (y, x) = (cls.y, cls.x)
        modp = LogitPenalized(y, x[:, :cls.k_nonzero], penal=cls.penalty)
        modp.pen_weight *= 0.5
        modp.penal.tau = 0.05
        cls.res2 = modp.fit(method='bfgs', maxiter=100, disp=0)
        mod = LogitPenalized(y, x, penal=cls.penalty)
        mod.pen_weight *= 0.5
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(method='bfgs', maxiter=100, trim=True, disp=0)
        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.atol = 1e-08
        cls.k_params = cls.k_nonzero

    def test_zeros(self):
        if False:
            for i in range(10):
                print('nop')
        assert_equal(self.res1.params[self.k_nonzero:], 0)
        assert_equal(self.res1.bse[self.k_nonzero:], 0)

class CheckPenalizedBinomCount(CheckPenalizedPoisson):

    @classmethod
    def _generate_endog(cls, linpred):
        if False:
            while True:
                i = 10
        mu = 1 / (1 + np.exp(-linpred + linpred.mean() - 0.5))
        np.random.seed(999)
        n_trials = 5 * np.ones(len(mu), int)
        n_trials[:len(mu) // 2] += 5
        y = np.random.binomial(n_trials, mu)
        return np.column_stack((y, n_trials - y))

class TestPenalizedGLMBinomCountNoPenal(CheckPenalizedBinomCount):

    @classmethod
    def _initialize(cls):
        if False:
            i = 10
            return i + 15
        (y, x) = (cls.y, cls.x)
        x = x[:, :4]
        offset = -0.25 * np.ones(len(y))
        modp = GLM(y, x, family=family.Binomial(), offset=offset)
        cls.res2 = modp.fit(method='bfgs', max_start_irls=100)
        mod = GLMPenalized(y, x, family=family.Binomial(), offset=offset, penal=cls.penalty)
        mod.pen_weight = 0
        cls.res1 = mod.fit(method='bfgs', max_start_irls=3, maxiter=100, disp=0, start_params=cls.res2.params * 0.9)
        cls.atol = 1e-10
        cls.k_params = 4

    def test_deriv(self):
        if False:
            return 10
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.model.score(res2.params * 0.98), res2.model.score(res2.params * 0.98), rtol=1e-10)
        assert_allclose(res1.model.score_obs(res2.params * 0.98), res2.model.score_obs(res2.params * 0.98), rtol=1e-10)

class TestPenalizedGLMBinomCountOracleHC(CheckPenalizedBinomCount):

    @classmethod
    def _initialize(cls):
        if False:
            while True:
                i = 10
        (y, x) = (cls.y, cls.x)
        offset = -0.25 * np.ones(len(y))
        cov_type = 'HC0'
        modp = GLM(y, x[:, :cls.k_nonzero], family=family.Binomial(), offset=offset)
        cls.res2 = modp.fit(cov_type=cov_type, method='newton', maxiter=1000, disp=0)
        mod = GLMPenalized(y, x, family=family.Binomial(), offset=offset, penal=cls.penalty)
        mod.pen_weight *= 1
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(cov_type=cov_type, method='bfgs', max_start_irls=0, maxiter=100, disp=0)
        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.atol = 0.001

class TestPenalizedGLMBinomCountOracleHC2(CheckPenalizedBinomCount):

    @classmethod
    def _initialize(cls):
        if False:
            while True:
                i = 10
        (y, x) = (cls.y, cls.x)
        offset = -0.25 * np.ones(len(y))
        cov_type = 'HC0'
        modp = GLM(y, x[:, :cls.k_nonzero], family=family.Binomial(), offset=offset)
        cls.res2 = modp.fit(cov_type=cov_type, method='newton', maxiter=1000, disp=0)
        mod = GLMPenalized(y, x, family=family.Binomial(), offset=offset, penal=cls.penalty)
        mod.pen_weight *= 1
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(cov_type=cov_type, method='bfgs', max_start_irls=0, maxiter=100, disp=0, trim=0.001)
        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.atol = 0.001
        cls.k_params = cls.k_nonzero

class CheckPenalizedGaussian(CheckPenalizedPoisson):

    @classmethod
    def _generate_endog(cls, linpred):
        if False:
            print('Hello World!')
        sig_e = np.sqrt(0.1)
        np.random.seed(999)
        y = linpred + sig_e * np.random.rand(len(linpred))
        return y

class TestPenalizedGLMGaussianOracleHC(CheckPenalizedGaussian):

    @classmethod
    def _initialize(cls):
        if False:
            return 10
        (y, x) = (cls.y, cls.x)
        y = y + 10
        cov_type = 'HC0'
        modp = GLM(y, x[:, :cls.k_nonzero], family=family.Gaussian())
        cls.res2 = modp.fit(cov_type=cov_type, method='bfgs', maxiter=100, disp=0)
        mod = GLMPenalized(y, x, family=family.Gaussian(), penal=cls.penalty)
        mod.pen_weight *= 1.5
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(cov_type=cov_type, method='bfgs', maxiter=100, disp=0)
        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.atol = 5e-06
        cls.rtol = 1e-06

class TestPenalizedGLMGaussianOracleHC2(CheckPenalizedGaussian):

    @classmethod
    def _initialize(cls):
        if False:
            while True:
                i = 10
        (y, x) = (cls.y, cls.x)
        y = y + 10
        cov_type = 'HC0'
        modp = GLM(y, x[:, :cls.k_nonzero], family=family.Gaussian())
        cls.res2 = modp.fit(cov_type=cov_type, method='bfgs', maxiter=100, disp=0)
        mod = GLMPenalized(y, x, family=family.Gaussian(), penal=cls.penalty)
        mod.pen_weight *= 1.5
        mod.penal.tau = 0.05
        cls.res1 = mod.fit(cov_type=cov_type, method='bfgs', maxiter=100, disp=0, trim=True)
        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.k_params = cls.k_nonzero
        cls.atol = 1e-05
        cls.rtol = 1e-05

class TestPenalizedGLMGaussianL2(CheckPenalizedGaussian):

    @classmethod
    def _initialize(cls):
        if False:
            while True:
                i = 10
        (y, x) = (cls.y, cls.x)
        y = y + 10
        cov_type = 'HC0'
        modp = GLM(y, x[:, :cls.k_nonzero], family=family.Gaussian())
        cls.res2 = modp.fit(cov_type=cov_type, method='bfgs', maxiter=100, disp=0)
        weights = (np.arange(x.shape[1]) >= 4).astype(float)
        mod = GLMPenalized(y, x, family=family.Gaussian(), penal=smpen.L2ConstraintsPenalty(weights=weights))
        mod.pen_weight *= 500
        cls.res1 = mod.fit(cov_type=cov_type, method='bfgs', maxiter=100, disp=0, trim=False)
        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.k_params = x.shape[1]
        cls.atol = 1e-05
        cls.rtol = 1e-05

class TestPenalizedGLMGaussianL2Theil(CheckPenalizedGaussian):

    @classmethod
    def _initialize(cls):
        if False:
            print('Hello World!')
        (y, x) = (cls.y, cls.x)
        y = y + 10
        k = x.shape[1]
        cov_type = 'HC0'
        restriction = np.eye(k)[2:]
        modp = TheilGLS(y, x, r_matrix=restriction)
        cls.res2 = modp.fit(pen_weight=120.74564413221599 * 1000, use_t=False)
        pen = smpen.L2ConstraintsPenalty(restriction=restriction)
        mod = GLMPenalized(y, x, family=family.Gaussian(), penal=pen)
        mod.pen_weight *= 1
        cls.res1 = mod.fit(cov_type=cov_type, method='bfgs', maxiter=100, disp=0, trim=False)
        cls.k_nonzero = k
        cls.exog_index = slice(None, cls.k_nonzero, None)
        cls.k_params = x.shape[1]
        cls.atol = 1e-05
        cls.rtol = 1e-05

    def test_params_table(self):
        if False:
            return 10
        res1 = self.res1
        res2 = self.res2
        assert_equal((res1.params != 0).sum(), self.k_params)
        assert_allclose(res1.params, res2.params, rtol=self.rtol, atol=self.atol)
        exog_index = slice(None, None, None)
        assert_allclose(res1.bse[exog_index], res2.bse[exog_index], rtol=0.1, atol=self.atol)
        assert_allclose(res1.tvalues[exog_index], res2.tvalues[exog_index], rtol=0.08, atol=0.005)
        assert_allclose(res1.pvalues[exog_index], res2.pvalues[exog_index], rtol=0.1, atol=0.005)
        assert_allclose(res1.predict(), res2.predict(), rtol=1e-05)