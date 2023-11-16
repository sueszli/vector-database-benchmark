"""Tests that use cross-checks for generic methods

Should be easy to check consistency across models
Does not cover tsa

Initial cases copied from test_shrink_pickle

Created on Wed Oct 30 14:01:27 2013

Author: Josef Perktold
"""
from statsmodels.compat.pytest import pytest_warns
from statsmodels.compat.pandas import assert_index_equal, assert_series_equal
from statsmodels.compat.platform import PLATFORM_LINUX32, PLATFORM_OSX, PLATFORM_WIN32
from statsmodels.compat.scipy import SCIPY_GT_14
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_array_equal, assert_equal
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.formula.api import glm, ols
import statsmodels.tools._testing as smt
from statsmodels.tools.sm_exceptions import HessianInversionWarning

class CheckGenericMixin:

    @classmethod
    def setup_class(cls):
        if False:
            return 10
        nobs = 500
        np.random.seed(987689)
        x = np.random.randn(nobs, 3)
        x = sm.add_constant(x)
        cls.exog = x
        cls.xf = 0.25 * np.ones((2, 4))
        cls.predict_kwds = {}
        cls.transform_index = None

    def test_ttest_tvalues(self):
        if False:
            while True:
                i = 10
        smt.check_ttest_tvalues(self.results)
        res = self.results
        mat = np.eye(len(res.params))
        tt = res.t_test(mat[0])
        string_confint = lambda alpha: '[%4.3F      %4.3F]' % (alpha / 2, 1 - alpha / 2)
        summ = tt.summary()
        assert_allclose(tt.pvalue, res.pvalues[0], rtol=5e-10)
        assert_(string_confint(0.05) in str(summ))
        summ = tt.summary(alpha=0.1)
        ss = '[0.05       0.95]'
        assert_(ss in str(summ))
        summf = tt.summary_frame(alpha=0.1)
        pvstring_use_t = 'P>|z|' if res.use_t is False else 'P>|t|'
        tstring_use_t = 'z' if res.use_t is False else 't'
        cols = ['coef', 'std err', tstring_use_t, pvstring_use_t, 'Conf. Int. Low', 'Conf. Int. Upp.']
        assert_array_equal(summf.columns.values, cols)

    def test_ftest_pvalues(self):
        if False:
            return 10
        smt.check_ftest_pvalues(self.results)

    def test_fitted(self):
        if False:
            for i in range(10):
                print('nop')
        smt.check_fitted(self.results)

    def test_predict_types(self):
        if False:
            print('Hello World!')
        smt.check_predict_types(self.results)

    def test_zero_constrained(self):
        if False:
            return 10
        if isinstance(self.results.model, sm.GEE):
            pytest.skip('GEE does not subclass LikelihoodModel')
        use_start_params = not isinstance(self.results.model, (sm.RLM, sm.OLS, sm.WLS))
        self.use_start_params = use_start_params
        keep_index = list(range(self.results.model.exog.shape[1]))
        keep_index_p = list(range(self.results.params.shape[0]))
        drop_index = [1]
        for i in drop_index:
            del keep_index[i]
            del keep_index_p[i]
        if use_start_params:
            res1 = self.results.model._fit_zeros(keep_index, maxiter=500, start_params=self.results.params)
        else:
            res1 = self.results.model._fit_zeros(keep_index, maxiter=500)
        res2 = self._get_constrained(keep_index, keep_index_p)
        assert_allclose(res1.params[keep_index_p], res2.params, rtol=1e-10, atol=1e-10)
        assert_equal(res1.params[drop_index], 0)
        assert_allclose(res1.bse[keep_index_p], res2.bse, rtol=1e-10, atol=1e-10)
        assert_equal(res1.bse[drop_index], 0)
        tol = 1e-08 if PLATFORM_OSX else 1e-10
        tvals1 = res1.tvalues[keep_index_p]
        assert_allclose(tvals1, res2.tvalues, rtol=tol, atol=tol)
        if PLATFORM_LINUX32 or SCIPY_GT_14:
            pvals1 = res1.pvalues[keep_index_p]
        else:
            pvals1 = res1.pvalues[keep_index_p]
        assert_allclose(pvals1, res2.pvalues, rtol=tol, atol=tol)
        if hasattr(res1, 'resid'):
            rtol = 1e-10
            atol = 1e-12
            if PLATFORM_OSX or PLATFORM_WIN32:
                rtol = 1e-08
                atol = 1e-10
            assert_allclose(res1.resid, res2.resid, rtol=rtol, atol=atol)
        ex = self.results.model.exog.mean(0)
        predicted1 = res1.predict(ex, **self.predict_kwds)
        predicted2 = res2.predict(ex[keep_index], **self.predict_kwds)
        assert_allclose(predicted1, predicted2, rtol=1e-10)
        ex = self.results.model.exog[:5]
        predicted1 = res1.predict(ex, **self.predict_kwds)
        predicted2 = res2.predict(ex[:, keep_index], **self.predict_kwds)
        assert_allclose(predicted1, predicted2, rtol=1e-10)

    def _get_constrained(self, keep_index, keep_index_p):
        if False:
            for i in range(10):
                print('nop')
        mod2 = self.results.model
        mod_cls = mod2.__class__
        init_kwds = mod2._get_init_kwds()
        mod = mod_cls(mod2.endog, mod2.exog[:, keep_index], **init_kwds)
        if self.use_start_params:
            res = mod.fit(start_params=self.results.params[keep_index_p], maxiter=500)
        else:
            res = mod.fit(maxiter=500)
        return res

    def test_zero_collinear(self):
        if False:
            print('Hello World!')
        if isinstance(self.results.model, sm.GEE):
            pytest.skip('Not completely generic yet')
        use_start_params = not isinstance(self.results.model, (sm.RLM, sm.OLS, sm.WLS, sm.GLM))
        self.use_start_params = use_start_params
        keep_index = list(range(self.results.model.exog.shape[1]))
        keep_index_p = list(range(self.results.params.shape[0]))
        drop_index = []
        for i in drop_index:
            del keep_index[i]
            del keep_index_p[i]
        keep_index_p = list(range(self.results.params.shape[0]))
        mod2 = self.results.model
        mod_cls = mod2.__class__
        init_kwds = mod2._get_init_kwds()
        ex = np.column_stack((mod2.exog, mod2.exog))
        mod = mod_cls(mod2.endog, ex, **init_kwds)
        keep_index = list(range(self.results.model.exog.shape[1]))
        keep_index_p = list(range(self.results.model.exog.shape[1]))
        k_vars = ex.shape[1]
        k_extra = 0
        if hasattr(mod, 'k_extra') and mod.k_extra > 0:
            keep_index_p += list(range(k_vars, k_vars + mod.k_extra))
            k_extra = mod.k_extra
        warn_cls = HessianInversionWarning if isinstance(mod, sm.GLM) else None
        cov_types = ['nonrobust', 'HC0']
        for cov_type in cov_types:
            if cov_type != 'nonrobust' and isinstance(self.results.model, sm.RLM):
                return
            if use_start_params:
                start_params = np.zeros(k_vars + k_extra)
                method = self.results.mle_settings['optimizer']
                sp = self.results.mle_settings['start_params'].copy()
                if self.transform_index is not None:
                    sp[self.transform_index] = np.exp(sp[self.transform_index])
                start_params[keep_index_p] = sp
                with pytest_warns(warn_cls):
                    res1 = mod._fit_collinear(cov_type=cov_type, start_params=start_params, method=method, disp=0)
                if cov_type != 'nonrobust':
                    with pytest_warns(warn_cls):
                        res2 = self.results.model.fit(cov_type=cov_type, start_params=sp, method=method, disp=0)
            else:
                with pytest_warns(warn_cls):
                    if isinstance(self.results.model, sm.RLM):
                        res1 = mod._fit_collinear()
                    else:
                        res1 = mod._fit_collinear(cov_type=cov_type)
                if cov_type != 'nonrobust':
                    res2 = self.results.model.fit(cov_type=cov_type)
            if cov_type == 'nonrobust':
                res2 = self.results
            if hasattr(res2, 'mle_settings'):
                assert_equal(res1.results_constrained.mle_settings['optimizer'], res2.mle_settings['optimizer'])
                if 'start_params' in res2.mle_settings:
                    spc = res1.results_constrained.mle_settings['start_params']
                    assert_allclose(spc, res2.mle_settings['start_params'], rtol=1e-10, atol=1e-20)
                    assert_equal(res1.mle_settings['optimizer'], res2.mle_settings['optimizer'])
                    assert_allclose(res1.mle_settings['start_params'], res2.mle_settings['start_params'], rtol=1e-10, atol=1e-20)
            assert_allclose(res1.params[keep_index_p], res2.params, rtol=1e-06)
            assert_allclose(res1.params[drop_index], 0, rtol=1e-10)
            assert_allclose(res1.bse[keep_index_p], res2.bse, rtol=1e-08)
            assert_allclose(res1.bse[drop_index], 0, rtol=1e-10)
            tvals1 = res1.tvalues[keep_index_p]
            assert_allclose(tvals1, res2.tvalues, rtol=5e-08)
            if PLATFORM_LINUX32 or SCIPY_GT_14:
                pvals1 = res1.pvalues[keep_index_p]
            else:
                pvals1 = res1.pvalues[keep_index_p]
            assert_allclose(pvals1, res2.pvalues, rtol=1e-06, atol=1e-30)
            if hasattr(res1, 'resid'):
                assert_allclose(res1.resid, res2.resid, rtol=1e-05, atol=1e-10)
            ex = res1.model.exog.mean(0)
            predicted1 = res1.predict(ex, **self.predict_kwds)
            predicted2 = res2.predict(ex[keep_index], **self.predict_kwds)
            assert_allclose(predicted1, predicted2, rtol=1e-08, atol=1e-11)
            ex = res1.model.exog[:5]
            kwds = getattr(self, 'predict_kwds_5', {})
            predicted1 = res1.predict(ex, **kwds)
            predicted2 = res2.predict(ex[:, keep_index], **kwds)
            assert_allclose(predicted1, predicted2, rtol=1e-08, atol=1e-11)

class TestGenericOLS(CheckGenericMixin):

    def setup_method(self):
        if False:
            return 10
        x = self.exog
        np.random.seed(987689)
        y = x.sum(1) + np.random.randn(x.shape[0])
        self.results = sm.OLS(y, self.exog).fit()

class TestGenericOLSOneExog(CheckGenericMixin):

    def setup_method(self):
        if False:
            return 10
        x = self.exog[:, 1]
        np.random.seed(987689)
        y = x + np.random.randn(x.shape[0])
        self.results = sm.OLS(y, x).fit()

    def test_zero_constrained(self):
        if False:
            for i in range(10):
                print('nop')
        pytest.skip('Override since cannot remove the only regressor')
        pass

class TestGenericWLS(CheckGenericMixin):

    def setup_method(self):
        if False:
            while True:
                i = 10
        x = self.exog
        np.random.seed(987689)
        y = x.sum(1) + np.random.randn(x.shape[0])
        self.results = sm.WLS(y, self.exog, weights=np.ones(len(y))).fit()

class TestGenericPoisson(CheckGenericMixin):

    def setup_method(self):
        if False:
            while True:
                i = 10
        x = self.exog
        np.random.seed(987689)
        y_count = np.random.poisson(np.exp(x.sum(1) - x.mean()))
        model = sm.Poisson(y_count, x)
        start_params = np.array([0.75334818, 0.99425553, 1.00494724, 1.00247112])
        self.results = model.fit(start_params=start_params, method='bfgs', disp=0)

class TestGenericPoissonOffset(CheckGenericMixin):

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        x = self.exog
        nobs = x.shape[0]
        np.random.seed(987689)
        y_count = np.random.poisson(np.exp(x.sum(1) - x.mean()))
        model = sm.Poisson(y_count, x, offset=0.01 * np.ones(nobs), exposure=np.ones(nobs))
        start_params = np.array([0.75334818, 0.99425553, 1.00494724, 1.00247112])
        self.results = model.fit(start_params=start_params, method='bfgs', disp=0)
        self.predict_kwds_5 = dict(exposure=0.01 * np.ones(5), offset=np.ones(5))
        self.predict_kwds = dict(exposure=1, offset=0)

class TestGenericNegativeBinomial(CheckGenericMixin):

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(987689)
        data = sm.datasets.randhie.load()
        data.exog = np.asarray(data.exog)
        data.endog = np.asarray(data.endog)
        exog = sm.add_constant(data.exog, prepend=False)
        mod = sm.NegativeBinomial(data.endog, exog)
        start_params = np.array([-0.05783623, -0.26655806, 0.04109148, -0.03815837, 0.2685168, 0.03811594, -0.04426238, 0.01614795, 0.17490962, 0.66461151, 1.2925957])
        self.results = mod.fit(start_params=start_params, disp=0, maxiter=500)
        self.transform_index = -1

class TestGenericLogit(CheckGenericMixin):

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        x = self.exog
        nobs = x.shape[0]
        np.random.seed(987689)
        y_bin = (np.random.rand(nobs) < 1.0 / (1 + np.exp(x.sum(1) - x.mean()))).astype(int)
        model = sm.Logit(y_bin, x)
        start_params = np.array([-0.73403806, -1.00901514, -0.97754543, -0.95648212])
        with pytest.warns(FutureWarning, match='Keyword arguments have been passed'):
            self.results = model.fit(start_params=start_params, method='bfgs', disp=0, tol=1e-05)

class TestGenericRLM(CheckGenericMixin):

    def setup_method(self):
        if False:
            print('Hello World!')
        x = self.exog
        np.random.seed(987689)
        y = x.sum(1) + np.random.randn(x.shape[0])
        self.results = sm.RLM(y, self.exog).fit()

class TestGenericGLM(CheckGenericMixin):

    def setup_method(self):
        if False:
            print('Hello World!')
        x = self.exog
        np.random.seed(987689)
        y = x.sum(1) + np.random.randn(x.shape[0])
        self.results = sm.GLM(y, self.exog).fit()

class TestGenericGLMPoissonOffset(CheckGenericMixin):

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        x = self.exog
        nobs = x.shape[0]
        np.random.seed(987689)
        y_count = np.random.poisson(np.exp(x.sum(1) - x.mean()))
        model = sm.GLM(y_count, x, family=sm.families.Poisson(), offset=0.01 * np.ones(nobs), exposure=np.ones(nobs))
        start_params = np.array([0.75334818, 0.99425553, 1.00494724, 1.00247112])
        self.results = model.fit(start_params=start_params, method='bfgs', disp=0)
        self.predict_kwds_5 = dict(exposure=0.01 * np.ones(5), offset=np.ones(5))
        self.predict_kwds = dict(exposure=1, offset=0)

class TestGenericGEEPoisson(CheckGenericMixin):

    def setup_method(self):
        if False:
            while True:
                i = 10
        x = self.exog
        np.random.seed(987689)
        y_count = np.random.poisson(np.exp(x.sum(1) - x.mean()))
        groups = np.random.randint(0, 4, size=x.shape[0])
        start_params = np.array([0.0, 1.0, 1.0, 1.0])
        vi = sm.cov_struct.Independence()
        family = sm.families.Poisson()
        self.results = sm.GEE(y_count, self.exog, groups, family=family, cov_struct=vi).fit(start_params=start_params)

class TestGenericGEEPoissonNaive(CheckGenericMixin):

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        x = self.exog
        np.random.seed(987689)
        y_count = np.random.poisson(np.exp(x.sum(1) - x.sum(1).mean(0)))
        groups = np.random.randint(0, 4, size=x.shape[0])
        start_params = np.array([0.0, 1.0, 1.0, 1.0])
        vi = sm.cov_struct.Independence()
        family = sm.families.Poisson()
        self.results = sm.GEE(y_count, self.exog, groups, family=family, cov_struct=vi).fit(start_params=start_params, cov_type='naive')

class TestGenericGEEPoissonBC(CheckGenericMixin):

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        x = self.exog
        np.random.seed(987689)
        y_count = np.random.poisson(np.exp(x.sum(1) - x.sum(1).mean(0)))
        groups = np.random.randint(0, 4, size=x.shape[0])
        start_params = np.array([0.0, 1.0, 1.0, 1.0])
        vi = sm.cov_struct.Independence()
        family = sm.families.Poisson()
        mod = sm.GEE(y_count, self.exog, groups, family=family, cov_struct=vi)
        self.results = mod.fit(start_params=start_params, cov_type='bias_reduced')

class CheckAnovaMixin:

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        import statsmodels.stats.tests.test_anova as ttmod
        test = ttmod.TestAnova3()
        test.setup_class()
        cls.data = test.data.drop([0, 1, 2])
        cls.initialize()

    def test_combined(self):
        if False:
            i = 10
            return i + 15
        res = self.res
        wa = res.wald_test_terms(skip_single=False, combine_terms=['Duration', 'Weight'], scalar=True)
        eye = np.eye(len(res.params))
        c_const = eye[0]
        c_w = eye[[2, 3]]
        c_d = eye[1]
        c_dw = eye[[4, 5]]
        c_weight = eye[2:6]
        c_duration = eye[[1, 4, 5]]
        compare_waldres(res, wa, [c_const, c_d, c_w, c_dw, c_duration, c_weight])

    def test_categories(self):
        if False:
            print('Hello World!')
        res = self.res
        wa = res.wald_test_terms(skip_single=True, scalar=True)
        eye = np.eye(len(res.params))
        c_w = eye[[2, 3]]
        c_dw = eye[[4, 5]]
        compare_waldres(res, wa, [c_w, c_dw])

def compare_waldres(res, wa, constrasts):
    if False:
        while True:
            i = 10
    for (i, c) in enumerate(constrasts):
        wt = res.wald_test(c, scalar=True)
        assert_allclose(wa.table.values[i, 0], wt.statistic)
        assert_allclose(wa.table.values[i, 1], wt.pvalue)
        df = c.shape[0] if c.ndim == 2 else 1
        assert_equal(wa.table.values[i, 2], df)
        assert_allclose(wa.statistic[i], wt.statistic)
        assert_allclose(wa.pvalues[i], wt.pvalue)
        assert_equal(wa.df_constraints[i], df)
        if res.use_t:
            assert_equal(wa.df_denom[i], res.df_resid)
    col_names = wa.col_names
    if res.use_t:
        assert_equal(wa.distribution, 'F')
        assert_equal(col_names[0], 'F')
        assert_equal(col_names[1], 'P>F')
    else:
        assert_equal(wa.distribution, 'chi2')
        assert_equal(col_names[0], 'chi2')
        assert_equal(col_names[1], 'P>chi2')
    wa.summary_frame()

class TestWaldAnovaOLS(CheckAnovaMixin):

    @classmethod
    def initialize(cls):
        if False:
            print('Hello World!')
        mod = ols('np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)', cls.data)
        cls.res = mod.fit(use_t=False)

    def test_noformula(self):
        if False:
            print('Hello World!')
        endog = self.res.model.endog
        exog = self.res.model.data.orig_exog
        exog = pd.DataFrame(exog)
        res = sm.OLS(endog, exog).fit()
        wa = res.wald_test_terms(skip_single=False, combine_terms=['Duration', 'Weight'], scalar=True)
        eye = np.eye(len(res.params))
        c_single = [row for row in eye]
        c_weight = eye[2:6]
        c_duration = eye[[1, 4, 5]]
        compare_waldres(res, wa, c_single + [c_duration, c_weight])
        df_constraints = [1] * len(c_single) + [3, 4]
        assert_equal(wa.df_constraints, df_constraints)

class TestWaldAnovaOLSF(CheckAnovaMixin):

    @classmethod
    def initialize(cls):
        if False:
            while True:
                i = 10
        mod = ols('np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)', cls.data)
        cls.res = mod.fit()

    def test_predict_missing(self):
        if False:
            return 10
        ex = self.data[:5].copy()
        ex.iloc[0, 1] = np.nan
        predicted1 = self.res.predict(ex)
        predicted2 = self.res.predict(ex[1:])
        assert_index_equal(predicted1.index, ex.index)
        assert_series_equal(predicted1.iloc[1:], predicted2)
        assert_equal(predicted1.values[0], np.nan)

class TestWaldAnovaGLM(CheckAnovaMixin):

    @classmethod
    def initialize(cls):
        if False:
            while True:
                i = 10
        mod = glm('np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)', cls.data)
        cls.res = mod.fit(use_t=False)

class TestWaldAnovaPoisson(CheckAnovaMixin):

    @classmethod
    def initialize(cls):
        if False:
            while True:
                i = 10
        from statsmodels.discrete.discrete_model import Poisson
        mod = Poisson.from_formula('Days ~ C(Duration, Sum)*C(Weight, Sum)', cls.data)
        cls.res = mod.fit(cov_type='HC0')

class TestWaldAnovaNegBin(CheckAnovaMixin):

    @classmethod
    def initialize(cls):
        if False:
            while True:
                i = 10
        from statsmodels.discrete.discrete_model import NegativeBinomial
        formula = 'Days ~ C(Duration, Sum)*C(Weight, Sum)'
        mod = NegativeBinomial.from_formula(formula, cls.data, loglike_method='nb2')
        cls.res = mod.fit()

class TestWaldAnovaNegBin1(CheckAnovaMixin):

    @classmethod
    def initialize(cls):
        if False:
            while True:
                i = 10
        from statsmodels.discrete.discrete_model import NegativeBinomial
        formula = 'Days ~ C(Duration, Sum)*C(Weight, Sum)'
        mod = NegativeBinomial.from_formula(formula, cls.data, loglike_method='nb1')
        cls.res = mod.fit(cov_type='HC0')

class CheckPairwise:

    def test_default(self):
        if False:
            i = 10
            return i + 15
        res = self.res
        tt = res.t_test(self.constraints)
        pw = res.t_test_pairwise(self.term_name)
        pw_frame = pw.result_frame
        assert_allclose(pw_frame.iloc[:, :6].values, tt.summary_frame().values)

class TestTTestPairwiseOLS(CheckPairwise):

    @classmethod
    def setup_class(cls):
        if False:
            i = 10
            return i + 15
        from statsmodels.formula.api import ols
        import statsmodels.stats.tests.test_anova as ttmod
        test = ttmod.TestAnova3()
        test.setup_class()
        cls.data = test.data.drop([0, 1, 2])
        mod = ols('np.log(Days+1) ~ C(Duration) + C(Weight)', cls.data)
        cls.res = mod.fit()
        cls.term_name = 'C(Weight)'
        cls.constraints = ['C(Weight)[T.2]', 'C(Weight)[T.3]', 'C(Weight)[T.3] - C(Weight)[T.2]']

    def test_alpha(self):
        if False:
            while True:
                i = 10
        pw1 = self.res.t_test_pairwise(self.term_name, method='hommel', factor_labels='A B C'.split())
        pw2 = self.res.t_test_pairwise(self.term_name, method='hommel', alpha=0.01)
        assert_allclose(pw1.result_frame.iloc[:, :7].values, pw2.result_frame.iloc[:, :7].values, rtol=1e-10)
        assert_equal(pw1.result_frame.iloc[:, -1].values, [True] * 3)
        assert_equal(pw2.result_frame.iloc[:, -1].values, [False, True, False])
        assert_equal(pw1.result_frame.index.values, np.array(['B-A', 'C-A', 'C-B'], dtype=object))

class TestTTestPairwiseOLS2(CheckPairwise):

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        from statsmodels.formula.api import ols
        import statsmodels.stats.tests.test_anova as ttmod
        test = ttmod.TestAnova3()
        test.setup_class()
        cls.data = test.data.drop([0, 1, 2])
        mod = ols('np.log(Days+1) ~ C(Weight) + C(Duration)', cls.data)
        cls.res = mod.fit()
        cls.term_name = 'C(Weight)'
        cls.constraints = ['C(Weight)[T.2]', 'C(Weight)[T.3]', 'C(Weight)[T.3] - C(Weight)[T.2]']

class TestTTestPairwiseOLS3(CheckPairwise):

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        from statsmodels.formula.api import ols
        import statsmodels.stats.tests.test_anova as ttmod
        test = ttmod.TestAnova3()
        test.setup_class()
        cls.data = test.data.drop([0, 1, 2])
        mod = ols('np.log(Days+1) ~ C(Weight) + C(Duration) - 1', cls.data)
        cls.res = mod.fit()
        cls.term_name = 'C(Weight)'
        cls.constraints = ['C(Weight)[2] - C(Weight)[1]', 'C(Weight)[3] - C(Weight)[1]', 'C(Weight)[3] - C(Weight)[2]']

class TestTTestPairwiseOLS4(CheckPairwise):

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        from statsmodels.formula.api import ols
        import statsmodels.stats.tests.test_anova as ttmod
        test = ttmod.TestAnova3()
        test.setup_class()
        cls.data = test.data.drop([0, 1, 2])
        mod = ols('np.log(Days+1) ~ C(Weight, Treatment(2)) + C(Duration)', cls.data)
        cls.res = mod.fit()
        cls.term_name = 'C(Weight, Treatment(2))'
        cls.constraints = ['-C(Weight, Treatment(2))[T.1]', 'C(Weight, Treatment(2))[T.3] - C(Weight, Treatment(2))[T.1]', 'C(Weight, Treatment(2))[T.3]']

class TestTTestPairwisePoisson(CheckPairwise):

    @classmethod
    def setup_class(cls):
        if False:
            i = 10
            return i + 15
        from statsmodels.discrete.discrete_model import Poisson
        import statsmodels.stats.tests.test_anova as ttmod
        test = ttmod.TestAnova3()
        test.setup_class()
        cls.data = test.data.drop([0, 1, 2])
        mod = Poisson.from_formula('Days ~ C(Duration) + C(Weight)', cls.data)
        cls.res = mod.fit(cov_type='HC0')
        cls.term_name = 'C(Weight)'
        cls.constraints = ['C(Weight)[T.2]', 'C(Weight)[T.3]', 'C(Weight)[T.3] - C(Weight)[T.2]']