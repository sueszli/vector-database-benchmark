"""
Created on Fri May 30 16:22:29 2014

Author: Josef Perktold
License: BSD-3

"""
from io import StringIO
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import patsy
import pytest
from statsmodels import datasets
from statsmodels.base._constraints import fit_constrained
from statsmodels.discrete.discrete_model import Poisson, Logit
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.tools import add_constant
from .results import results_glm_logit_constrained as reslogit, results_poisson_constrained as results
spector_data = datasets.spector.load()
spector_data.endog = np.asarray(spector_data.endog)
spector_data.exog = np.asarray(spector_data.exog)
spector_data.exog = add_constant(spector_data.exog, prepend=False)
DEBUG = False
ss = 'agecat\tsmokes\tdeaths\tpyears\n1\t1\t32\t52407\n2\t1\t104\t43248\n3\t1\t206\t28612\n4\t1\t186\t12663\n5\t1\t102\t5317\n1\t0\t2\t18790\n2\t0\t12\t10673\n3\t0\t28\t5710\n4\t0\t28\t2585\n5\t0\t31\t1462'
data = pd.read_csv(StringIO(ss), delimiter='\t')
data = data.astype(int)
data['logpyears'] = np.log(data['pyears'])

class CheckPoissonConstrainedMixin:

    def test_basic(self):
        if False:
            for i in range(10):
                print('nop')
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1[0], res2.params[self.idx], rtol=1e-06)
        bse1 = np.sqrt(np.diag(res1[1]))
        mask = (bse1 == 0) & np.isnan(res2.bse[self.idx])
        assert_allclose(bse1[~mask], res2.bse[self.idx][~mask], rtol=1e-06)

    def test_basic_method(self):
        if False:
            i = 10
            return i + 15
        if hasattr(self, 'res1m'):
            res1 = self.res1m if not hasattr(self.res1m, '_results') else self.res1m._results
            res2 = self.res2
            assert_allclose(res1.params, res2.params[self.idx], rtol=1e-06)
            mask = (res1.bse == 0) & np.isnan(res2.bse[self.idx])
            assert_allclose(res1.bse[~mask], res2.bse[self.idx][~mask], rtol=1e-06)
            tvalues = res2.params_table[self.idx, 2]
            mask = np.isinf(res1.tvalues) & np.isnan(tvalues)
            assert_allclose(res1.tvalues[~mask], tvalues[~mask], rtol=1e-06)
            pvalues = res2.params_table[self.idx, 3]
            mask = (res1.pvalues == 0) & np.isnan(pvalues)
            assert_allclose(res1.pvalues[~mask], pvalues[~mask], rtol=5e-05)
            ci_low = res2.params_table[self.idx, 4]
            ci_upp = res2.params_table[self.idx, 5]
            ci = np.column_stack((ci_low, ci_upp))
            assert_allclose(res1.conf_int()[~np.isnan(ci)], ci[~np.isnan(ci)], rtol=5e-05)
            assert_allclose(res1.llf, res2.ll, rtol=1e-06)
            assert_equal(res1.df_model, res2.df_m)
            df_r = res2.N - res2.df_m - 1
            assert_equal(res1.df_resid, df_r)
        else:
            pytest.skip('not available yet')

    def test_other(self):
        if False:
            return 10
        if hasattr(self, 'res1m'):
            res1 = self.res1m
            res2 = self.res2
            if hasattr(res2, 'll_0'):
                assert_allclose(res1.llnull, res2.ll_0, rtol=1e-06)
            elif DEBUG:
                import warnings
                message = 'test: ll_0 not available, llnull=%6.4F' % res1.llnull
                warnings.warn(message)
        else:
            pytest.skip('not available yet')

class TestPoissonConstrained1a(CheckPoissonConstrainedMixin):

    @classmethod
    def setup_class(cls):
        if False:
            return 10
        cls.res2 = results.results_noexposure_constraint
        cls.idx = [7, 3, 4, 5, 6, 0, 1]
        formula = 'deaths ~ logpyears + smokes + C(agecat)'
        mod = Poisson.from_formula(formula, data=data)
        k_vars = len(mod.exog_names)
        start_params = np.zeros(k_vars)
        start_params[0] = np.log(mod.endog.mean())
        constr = 'C(agecat)[T.4] = C(agecat)[T.5]'
        lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constr)
        cls.res1 = fit_constrained(mod, lc.coefs, lc.constants, start_params=start_params, fit_kwds={'method': 'bfgs', 'disp': 0})
        cls.res1m = mod.fit_constrained(constr, start_params=start_params, method='bfgs', disp=0)

    @pytest.mark.smoke
    def test_summary(self):
        if False:
            return 10
        summ = self.res1m.summary()
        assert_('linear equality constraints' in summ.extra_txt)

    @pytest.mark.smoke
    def test_summary2(self):
        if False:
            while True:
                i = 10
        summ = self.res1m.summary2()
        assert_('linear equality constraints' in summ.extra_txt[0])

class TestPoissonConstrained1b(CheckPoissonConstrainedMixin):

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        cls.res2 = results.results_exposure_constraint
        cls.idx = [6, 2, 3, 4, 5, 0]
        formula = 'deaths ~ smokes + C(agecat)'
        mod = Poisson.from_formula(formula, data=data, exposure=data['pyears'].values)
        constr = 'C(agecat)[T.4] = C(agecat)[T.5]'
        lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constr)
        cls.res1 = fit_constrained(mod, lc.coefs, lc.constants, fit_kwds={'method': 'newton', 'disp': 0})
        cls.constraints = lc
        cls.res1m = mod.fit_constrained(constr, method='newton', disp=0)

class TestPoissonConstrained1c(CheckPoissonConstrainedMixin):

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        cls.res2 = results.results_exposure_constraint
        cls.idx = [6, 2, 3, 4, 5, 0]
        formula = 'deaths ~ smokes + C(agecat)'
        mod = Poisson.from_formula(formula, data=data, offset=np.log(data['pyears'].values))
        constr = 'C(agecat)[T.4] = C(agecat)[T.5]'
        lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constr)
        cls.res1 = fit_constrained(mod, lc.coefs, lc.constants, fit_kwds={'method': 'newton', 'disp': 0})
        cls.constraints = lc
        cls.res1m = mod.fit_constrained(constr, method='newton', disp=0)

class TestPoissonNoConstrained(CheckPoissonConstrainedMixin):

    @classmethod
    def setup_class(cls):
        if False:
            i = 10
            return i + 15
        cls.res2 = results.results_exposure_noconstraint
        cls.idx = [6, 2, 3, 4, 5, 0]
        formula = 'deaths ~ smokes + C(agecat)'
        mod = Poisson.from_formula(formula, data=data, offset=np.log(data['pyears'].values))
        res1 = mod.fit(disp=0)._results
        cls.res1 = (res1.params, res1.cov_params())
        cls.res1m = res1

class TestPoissonConstrained2a(CheckPoissonConstrainedMixin):

    @classmethod
    def setup_class(cls):
        if False:
            return 10
        cls.res2 = results.results_noexposure_constraint2
        cls.idx = [7, 3, 4, 5, 6, 0, 1]
        formula = 'deaths ~ logpyears + smokes + C(agecat)'
        mod = Poisson.from_formula(formula, data=data)
        k_vars = len(mod.exog_names)
        start_params = np.zeros(k_vars)
        start_params[0] = np.log(mod.endog.mean())
        constr = 'C(agecat)[T.5] - C(agecat)[T.4] = 0.5'
        lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constr)
        cls.res1 = fit_constrained(mod, lc.coefs, lc.constants, start_params=start_params, fit_kwds={'method': 'bfgs', 'disp': 0})
        cls.res1m = mod.fit_constrained(constr, start_params=start_params, method='bfgs', disp=0)

class TestPoissonConstrained2b(CheckPoissonConstrainedMixin):

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.res2 = results.results_exposure_constraint2
        cls.idx = [6, 2, 3, 4, 5, 0]
        formula = 'deaths ~ smokes + C(agecat)'
        mod = Poisson.from_formula(formula, data=data, exposure=data['pyears'].values)
        constr = 'C(agecat)[T.5] - C(agecat)[T.4] = 0.5'
        lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constr)
        cls.res1 = fit_constrained(mod, lc.coefs, lc.constants, fit_kwds={'method': 'newton', 'disp': 0})
        cls.constraints = lc
        cls.res1m = mod.fit_constrained(constr, method='bfgs', disp=0, start_params=cls.res1[0])

class TestPoissonConstrained2c(CheckPoissonConstrainedMixin):

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        cls.res2 = results.results_exposure_constraint2
        cls.idx = [6, 2, 3, 4, 5, 0]
        formula = 'deaths ~ smokes + C(agecat)'
        mod = Poisson.from_formula(formula, data=data, offset=np.log(data['pyears'].values))
        constr = 'C(agecat)[T.5] - C(agecat)[T.4] = 0.5'
        lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constr)
        cls.res1 = fit_constrained(mod, lc.coefs, lc.constants, fit_kwds={'method': 'newton', 'disp': 0})
        cls.constraints = lc
        cls.res1m = mod.fit_constrained(constr, method='bfgs', disp=0, start_params=cls.res1[0])

class TestGLMPoissonConstrained1a(CheckPoissonConstrainedMixin):

    @classmethod
    def setup_class(cls):
        if False:
            i = 10
            return i + 15
        from statsmodels.base._constraints import fit_constrained
        cls.res2 = results.results_noexposure_constraint
        cls.idx = [7, 3, 4, 5, 6, 0, 1]
        formula = 'deaths ~ logpyears + smokes + C(agecat)'
        mod = GLM.from_formula(formula, data=data, family=families.Poisson())
        constr = 'C(agecat)[T.4] = C(agecat)[T.5]'
        lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constr)
        cls.res1 = fit_constrained(mod, lc.coefs, lc.constants, fit_kwds={'atol': 1e-10})
        cls.constraints = lc
        cls.res1m = mod.fit_constrained(constr, atol=1e-10)

class TestGLMPoissonConstrained1b(CheckPoissonConstrainedMixin):

    @classmethod
    def setup_class(cls):
        if False:
            return 10
        from statsmodels.base._constraints import fit_constrained
        from statsmodels.genmod import families
        from statsmodels.genmod.generalized_linear_model import GLM
        cls.res2 = results.results_exposure_constraint
        cls.idx = [6, 2, 3, 4, 5, 0]
        formula = 'deaths ~ smokes + C(agecat)'
        mod = GLM.from_formula(formula, data=data, family=families.Poisson(), offset=np.log(data['pyears'].values))
        constr = 'C(agecat)[T.4] = C(agecat)[T.5]'
        lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constr)
        cls.res1 = fit_constrained(mod, lc.coefs, lc.constants, fit_kwds={'atol': 1e-10})
        cls.constraints = lc
        cls.res1m = mod.fit_constrained(constr, atol=1e-10)._results

    def test_compare_glm_poisson(self):
        if False:
            print('Hello World!')
        res1 = self.res1m
        res2 = self.res2
        formula = 'deaths ~ smokes + C(agecat)'
        mod = Poisson.from_formula(formula, data=data, exposure=data['pyears'].values)
        constr = 'C(agecat)[T.4] = C(agecat)[T.5]'
        res2 = mod.fit_constrained(constr, start_params=self.res1m.params, method='newton', warn_convergence=False, disp=0)
        assert_allclose(res1.params, res2.params, rtol=1e-12)
        assert_allclose(res1.bse, res2.bse, rtol=1e-11)
        predicted = res1.predict()
        assert_allclose(predicted, res2.predict(), rtol=1e-10)
        assert_allclose(res1.mu, predicted, rtol=1e-10)
        assert_allclose(res1.fittedvalues, predicted, rtol=1e-10)
        assert_allclose(res2.predict(which='linear'), res2.predict(which='linear'), rtol=1e-10)

class CheckGLMConstrainedMixin(CheckPoissonConstrainedMixin):

    def test_glm(self):
        if False:
            print('Hello World!')
        res2 = self.res2
        res1 = self.res1m
        assert_allclose(res1.aic, res2.infocrit[4], rtol=1e-10)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            assert_allclose(res1.bic, res2.bic, rtol=1e-10)
        assert_allclose(res1.deviance, res2.deviance, rtol=1e-10)

class TestGLMLogitConstrained1(CheckGLMConstrainedMixin):

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.idx = slice(None)
        cls.res2 = reslogit.results_constraint1
        mod1 = GLM(spector_data.endog, spector_data.exog, family=families.Binomial())
        constr = 'x1 = 2.8'
        cls.res1m = mod1.fit_constrained(constr)
        (R, q) = cls.res1m.constraints
        cls.res1 = fit_constrained(mod1, R, q)

class TestLogitConstrained1(CheckGLMConstrainedMixin):

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        cls.idx = slice(None)
        cls.res2 = reslogit.results_constraint1
        mod1 = Logit(spector_data.endog, spector_data.exog)
        constr = 'x1 = 2.8'
        cls.res1m = mod1.fit_constrained(constr, method='bfgs')
        (R, q) = (cls.res1m.constraints.coefs, cls.res1m.constraints.constants)
        cls.res1 = fit_constrained(mod1, R, q, fit_kwds={'method': 'bfgs'})

    @pytest.mark.skip(reason='not a GLM')
    def test_glm(self):
        if False:
            print('Hello World!')
        return

class TestGLMLogitConstrained2(CheckGLMConstrainedMixin):

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        cls.idx = slice(None)
        cls.res2 = reslogit.results_constraint2
        mod1 = GLM(spector_data.endog, spector_data.exog, family=families.Binomial())
        constr = 'x1 - x3 = 0'
        cls.res1m = mod1.fit_constrained(constr, atol=1e-10)
        (R, q) = (cls.res1m.constraints.coefs, cls.res1m.constraints.constants)
        cls.res1 = fit_constrained(mod1, R, q, fit_kwds={'atol': 1e-10})
        cls.constraints_rq = (R, q)

    def test_predict(self):
        if False:
            i = 10
            return i + 15
        res2 = self.res2
        res1 = self.res1m
        predicted = res1.predict()
        assert_allclose(predicted, res2.predict_mu, atol=1e-07)
        assert_allclose(res1.mu, predicted, rtol=1e-10)
        assert_allclose(res1.fittedvalues, predicted, rtol=1e-10)

    @pytest.mark.smoke
    def test_summary(self):
        if False:
            for i in range(10):
                print('nop')
        summ = self.res1m.summary()
        assert_('linear equality constraints' in summ.extra_txt)
        lc_string = str(self.res1m.constraints)
        assert lc_string == 'x1 - x3 = 0.0'

    @pytest.mark.smoke
    def test_summary2(self):
        if False:
            while True:
                i = 10
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            summ = self.res1m.summary2()
        assert_('linear equality constraints' in summ.extra_txt[0])

    def test_fit_constrained_wrap(self):
        if False:
            while True:
                i = 10
        res2 = self.res2
        from statsmodels.base._constraints import fit_constrained_wrap
        res_wrap = fit_constrained_wrap(self.res1m.model, self.constraints_rq)
        assert_allclose(res_wrap.params, res2.params, rtol=1e-06)
        assert_allclose(res_wrap.params, res2.params, rtol=1e-06)

class TestGLMLogitConstrained2HC(CheckGLMConstrainedMixin):

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        cls.idx = slice(None)
        cls.res2 = reslogit.results_constraint2_robust
        mod1 = GLM(spector_data.endog, spector_data.exog, family=families.Binomial())
        cov_type = 'HC0'
        cov_kwds = {'scaling_factor': 32 / 31}
        constr = 'x1 - x3 = 0'
        cls.res1m = mod1.fit_constrained(constr, cov_type=cov_type, cov_kwds=cov_kwds, atol=1e-10)
        (R, q) = cls.res1m.constraints
        cls.res1 = fit_constrained(mod1, R, q, fit_kwds={'atol': 1e-10, 'cov_type': cov_type, 'cov_kwds': cov_kwds})
        cls.constraints_rq = (R, q)

class TestLogitConstrained2HC(CheckGLMConstrainedMixin):

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        cls.idx = slice(None)
        cls.res2 = reslogit.results_constraint2_robust
        mod1 = Logit(spector_data.endog, spector_data.exog)
        cov_type = 'HC0'
        cov_kwds = {'scaling_factor': 32 / 31}
        constr = 'x1 - x3 = 0'
        cls.res1m = mod1.fit_constrained(constr, cov_type=cov_type, cov_kwds=cov_kwds, tol=1e-10)
        (R, q) = (cls.res1m.constraints.coefs, cls.res1m.constraints.constants)
        cls.res1 = fit_constrained(mod1, R, q, fit_kwds={'tol': 1e-10, 'cov_type': cov_type, 'cov_kwds': cov_kwds})
        cls.constraints_rq = (R, q)

    @pytest.mark.skip(reason='not a GLM')
    def test_glm(self):
        if False:
            return 10
        return

def junk():
    if False:
        return 10
    formula2 = 'deaths ~ C(agecat) + C(smokes) : C(agecat)'
    mod = Poisson.from_formula(formula2, data=data, exposure=data['pyears'].values)
    mod.fit()
    constraints = 'C(smokes)[T.1]:C(agecat)[3] = C(smokes)[T.1]:C(agec`at)[4]'
    import patsy
    lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constraints)
    (R, q) = (lc.coefs, lc.constants)
    mod.fit_constrained(R, q, fit_kwds={'method': 'bfgs'})
    formula1a = 'deaths ~ logpyears + smokes + C(agecat)'
    mod1a = Poisson.from_formula(formula1a, data=data)
    mod1a.fit()
    lc_1a = patsy.DesignInfo(mod1a.exog_names).linear_constraint('C(agecat)[T.4] = C(agecat)[T.5]')
    mod1a.fit_constrained(lc_1a.coefs, lc_1a.constants, fit_kwds={'method': 'newton'})