"""
Test  for ordinal models
"""
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest
import scipy.stats as stats
from statsmodels.discrete.discrete_model import Logit
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.tools.sm_exceptions import HessianInversionWarning
from statsmodels.tools.tools import add_constant
from .results.results_ordinal_model import data_store as ds

class CheckOrdinalModelMixin:

    def test_basic(self):
        if False:
            while True:
                i = 10
        n_cat = ds.n_ordinal_cat
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.params[:-n_cat + 1], res2.coefficients_val, atol=0.0002)
        assert_allclose(res1.bse[:-n_cat + 1], res2.coefficients_stdE, rtol=0.003, atol=1e-05)
        assert_allclose(res1.tvalues[:-n_cat + 1], res2.coefficients_tval, rtol=0.003, atol=0.0007)
        assert_allclose(res1.pvalues[:-n_cat + 1], res2.coefficients_pval, rtol=0.009, atol=1e-05)
        assert_allclose(res1.model.transform_threshold_params(res1.params)[1:-1], res2.thresholds, atol=0.0004)
        assert_allclose(res1.predict()[:7, :], res2.prob_pred, atol=5e-05)

    def test_pandas(self):
        if False:
            return 10
        res1 = self.res1
        resp = self.resp
        assert_allclose(res1.params, resp.params, atol=1e-10)
        assert_allclose(res1.bse, resp.bse, atol=1e-10)
        assert_allclose(res1.model.endog, resp.model.endog, rtol=1e-10)
        assert_allclose(res1.model.exog, resp.model.exog, rtol=1e-10)

    def test_formula(self):
        if False:
            for i in range(10):
                print('nop')
        res1 = self.res1
        resf = self.resf
        assert_allclose(res1.params, resf.params, atol=5e-05)
        assert_allclose(res1.bse, resf.bse, atol=5e-05)
        assert_allclose(res1.model.endog, resf.model.endog, rtol=1e-10)
        assert_allclose(res1.model.exog, resf.model.exog, rtol=1e-10)

    def test_unordered(self):
        if False:
            for i in range(10):
                print('nop')
        res1 = self.res1
        resf = self.resu
        assert_allclose(res1.params, resf.params, atol=1e-10)
        assert_allclose(res1.bse, resf.bse, atol=1e-10)
        assert_allclose(res1.model.endog, resf.model.endog, rtol=1e-10)
        assert_allclose(res1.model.exog, resf.model.exog, rtol=1e-10)

    def test_results_other(self):
        if False:
            print('Hello World!')
        res1 = self.res1
        resp = self.resp
        param_names_np = ['x1', 'x2', 'x3', '0/1', '1/2']
        param_names_pd = ['pared', 'public', 'gpa', 'unlikely/somewhat likely', 'somewhat likely/very likely']
        assert res1.model.data.param_names == param_names_np
        assert self.resp.model.data.param_names == param_names_pd
        assert self.resp.model.endog_names == 'apply'
        if hasattr(self, 'pred_table'):
            table = res1.pred_table()
            assert_equal(table.values, self.pred_table)
        res1.summary()
        tt = res1.t_test(np.eye(len(res1.params)))
        assert_allclose(tt.pvalue, res1.pvalues, rtol=1e-13)
        tt = resp.t_test(['pared', 'public', 'gpa'])
        assert_allclose(tt.pvalue, res1.pvalues[:3], rtol=1e-13)
        pred = res1.predict(exog=res1.model.exog[-5:])
        fitted = res1.predict()
        assert_allclose(pred, fitted[-5:], rtol=1e-13)
        pred = resp.predict(exog=resp.model.data.orig_exog.iloc[-5:])
        fitted = resp.predict()
        assert_allclose(pred, fitted[-5:], rtol=1e-13)
        dataf = self.resf.model.data.frame
        dataf_df = pd.DataFrame.from_dict(dataf)
        pred = self.resf.predict(exog=dataf_df.iloc[-5:])
        fitted = self.resf.predict()
        assert_allclose(pred, fitted[-5:], rtol=1e-13)
        (n, k) = res1.model.exog.shape
        assert_equal(self.resf.df_resid, n - (k + 2))
        assert resp.params.index.tolist() == resp.model.exog_names
        assert resp.bse.index.tolist() == resp.model.exog_names

class TestLogitModel(CheckOrdinalModelMixin):

    @classmethod
    def setup_class(cls):
        if False:
            i = 10
            return i + 15
        data = ds.df
        data_unordered = ds.df_unordered
        mod = OrderedModel(data['apply'].values.codes, np.asarray(data[['pared', 'public', 'gpa']], float), distr='logit')
        res = mod.fit(method='bfgs', disp=False)
        modp = OrderedModel(data['apply'], data[['pared', 'public', 'gpa']], distr='logit')
        resp = modp.fit(method='bfgs', disp=False)
        modf = OrderedModel.from_formula('apply ~ pared + public + gpa - 1', data={'apply': data['apply'].values.codes, 'pared': data['pared'], 'public': data['public'], 'gpa': data['gpa']}, distr='logit')
        resf = modf.fit(method='bfgs', disp=False)
        modu = OrderedModel(data_unordered['apply'].values.codes, np.asarray(data_unordered[['pared', 'public', 'gpa']], float), distr='logit')
        resu = modu.fit(method='bfgs', disp=False)
        from .results.results_ordinal_model import res_ord_logit as res2
        cls.res2 = res2
        cls.res1 = res
        cls.resp = resp
        cls.resf = resf
        cls.resu = resu

    def test_postestimation(self):
        if False:
            for i in range(10):
                print('nop')
        res1 = self.res1
        res2 = self.res2
        resid_prob = res1.resid_prob
        assert_allclose(resid_prob[:len(res2.resid_prob)], res2.resid_prob, atol=0.0001)
        stats_prob = [resid_prob.mean(), resid_prob.min(), resid_prob.max(), resid_prob.var(ddof=1)]
        assert_allclose(stats_prob, res2.resid_prob_stats, atol=1e-05)
        chi2 = 20.958760713111
        df = 17
        p_value = 0.2281403796588
        import statsmodels.stats.diagnostic_gen as dia
        fitted = res1.predict()
        y_dummy = (res1.model.endog[:, None] == np.arange(3)).astype(int)
        sv = (fitted * np.arange(1, 3 + 1)).sum(1)
        dt = dia.test_chisquare_binning(y_dummy, fitted, sort_var=sv, bins=10, df=None, ordered=True, sort_method='stable')
        assert_allclose(dt.statistic, chi2, rtol=5e-05)
        assert_allclose(dt.pvalue, p_value, rtol=0.0001)
        assert_equal(dt.df, df)

class TestProbitModel(CheckOrdinalModelMixin):

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        data = ds.df
        data_unordered = ds.df_unordered
        mod = OrderedModel(data['apply'].values.codes, np.asarray(data[['pared', 'public', 'gpa']], float), distr='probit')
        res = mod.fit(method='bfgs', disp=False)
        modp = OrderedModel(data['apply'], data[['pared', 'public', 'gpa']], distr='probit')
        resp = modp.fit(method='bfgs', disp=False)
        modf = OrderedModel.from_formula('apply ~ pared + public + gpa - 1', data={'apply': data['apply'].values.codes, 'pared': data['pared'], 'public': data['public'], 'gpa': data['gpa']}, distr='probit')
        resf = modf.fit(method='bfgs', disp=False)
        modu = OrderedModel(data_unordered['apply'].values.codes, np.asarray(data_unordered[['pared', 'public', 'gpa']], float), distr='probit')
        resu = modu.fit(method='bfgs', disp=False)
        from .results.results_ordinal_model import res_ord_probit as res2
        cls.res2 = res2
        cls.res1 = res
        cls.resp = resp
        cls.resf = resf
        cls.resu = resu
        cls.pred_table = np.array([[202, 18, 0, 220], [112, 28, 0, 140], [27, 13, 0, 40], [341, 59, 0, 400]], dtype=np.int64)

    def test_loglikerelated(self):
        if False:
            i = 10
            return i + 15
        res1 = self.res1
        mod = res1.model
        fact = 1.1
        score1 = mod.score(res1.params * fact)
        score_obs_numdiff = mod.score_obs(res1.params * fact)
        score_obs_exog = mod.score_obs_(res1.params * fact)
        assert_allclose(score_obs_numdiff.sum(0), score1, atol=1e-06)
        assert_allclose(score_obs_exog.sum(0), score1[:mod.k_vars], atol=1e-06)
        mod_null = OrderedModel(mod.endog, None, offset=np.zeros(mod.nobs), distr=mod.distr)
        null_params = mod.start_params
        res_null = mod_null.fit(method='bfgs', disp=False)
        assert_allclose(res_null.params, null_params[mod.k_vars:], rtol=1e-08)
        assert_allclose(res1.llnull, res_null.llf, rtol=1e-08)

    def test_formula_categorical(self):
        if False:
            print('Hello World!')
        resp = self.resp
        data = ds.df
        formula = 'apply ~ pared + public + gpa - 1'
        modf2 = OrderedModel.from_formula(formula, data, distr='probit')
        resf2 = modf2.fit(method='bfgs', disp=False)
        assert_allclose(resf2.params, resp.params, atol=1e-08)
        assert modf2.exog_names == resp.model.exog_names
        assert modf2.data.ynames == resp.model.data.ynames
        assert hasattr(modf2.data, 'frame')
        assert not hasattr(modf2, 'frame')
        msg = 'Only ordered pandas Categorical'
        with pytest.raises(ValueError, match=msg):
            OrderedModel.from_formula('apply ~ pared + public + gpa - 1', data={'apply': np.asarray(data['apply']), 'pared': data['pared'], 'public': data['public'], 'gpa': data['gpa']}, distr='probit')

    def test_offset(self):
        if False:
            print('Hello World!')
        resp = self.resp
        data = ds.df
        offset = np.ones(len(data))
        formula = 'apply ~ pared + public + gpa - 1'
        modf2 = OrderedModel.from_formula(formula, data, offset=offset, distr='probit')
        resf2 = modf2.fit(method='bfgs', disp=False)
        resf2_params = np.asarray(resf2.params)
        resp_params = np.asarray(resp.params)
        assert_allclose(resf2_params[:3], resp_params[:3], atol=0.0002)
        assert_allclose(resf2_params[3], resp_params[3] + 1, atol=0.0002)
        fitted = resp.predict()
        fitted2 = resf2.predict()
        assert_allclose(fitted2, fitted, atol=0.0002)
        pred_ones = resf2.predict(data[:6], offset=np.ones(6))
        assert_allclose(pred_ones, fitted[:6], atol=0.0002)
        pred_zero1 = resf2.predict(data[:6])
        pred_zero2 = resf2.predict(data[:6], offset=0)
        assert_allclose(pred_zero1, pred_zero2, atol=0.0002)
        pred_zero = resp.predict(data[['pared', 'public', 'gpa']].iloc[:6], offset=-np.ones(6))
        assert_allclose(pred_zero1, pred_zero, atol=0.0002)
        params_adj = resp.params.copy()
        params_adj.iloc[3] += 1
        fitted_zero = resp.model.predict(params_adj)
        assert_allclose(pred_zero1, fitted_zero[:6], atol=0.0002)

class TestLogitModelFormula:

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        data = ds.df
        nobs = len(data)
        data['dummy'] = (np.arange(nobs) < nobs / 2).astype(float)
        data['C(dummy)[T.1.0]'] = data['dummy']
        cls.data = data
        columns = ['C(dummy)[T.1.0]', 'pared', 'public', 'gpa']
        mod = OrderedModel(data['apply'].values.codes, np.asarray(data[columns], float), distr='logit')
        cls.res = mod.fit(method='bfgs', disp=False)
        modp = OrderedModel(data['apply'], data[columns], distr='logit')
        cls.resp = modp.fit(method='bfgs', disp=False)

    def test_setup(self):
        if False:
            return 10
        data = self.data
        resp = self.resp
        fittedvalues = resp.predict()
        formulas = ['apply ~ 1 + pared + public + gpa + C(dummy)', 'apply ~ pared + public + gpa + C(dummy)']
        for formula in formulas:
            modf1 = OrderedModel.from_formula(formula, data, distr='logit')
            resf1 = modf1.fit(method='bfgs')
            summf1 = resf1.summary()
            summf1_str = str(summf1)
            assert resf1.model.exog_names == resp.model.exog_names
            assert resf1.model.data.param_names == resp.model.exog_names
            assert all((name in summf1_str for name in resp.model.data.param_names))
            assert_allclose(resf1.predict(data[:5]), fittedvalues[:5])
        formula = 'apply ~ 0 + pared + public + gpa + C(dummy)'
        with pytest.raises(ValueError, match='not be a constant'):
            OrderedModel.from_formula(formula, data, distr='logit')
        modf2 = OrderedModel.from_formula(formula, data, distr='logit', hasconst=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', HessianInversionWarning)
            resf2 = modf2.fit(method='bfgs')
        assert_allclose(resf2.predict(data[:5]), fittedvalues[:5], rtol=0.0001)

class TestCLogLogModel(CheckOrdinalModelMixin):

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        data = ds.df
        data_unordered = ds.df_unordered

        class CLogLog(stats.rv_continuous):

            def _ppf(self, q):
                if False:
                    for i in range(10):
                        print('nop')
                return np.log(-np.log(1 - q))

            def _cdf(self, x):
                if False:
                    print('Hello World!')
                return 1 - np.exp(-np.exp(x))
        cloglog = CLogLog()
        mod = OrderedModel(data['apply'].values.codes, np.asarray(data[['pared', 'public', 'gpa']], float), distr=cloglog)
        res = mod.fit(method='bfgs', disp=False)
        modp = OrderedModel(data['apply'], data[['pared', 'public', 'gpa']], distr=cloglog)
        resp = modp.fit(method='bfgs', disp=False)
        modf = OrderedModel.from_formula('apply ~ pared + public + gpa - 1', data={'apply': data['apply'].values.codes, 'pared': data['pared'], 'public': data['public'], 'gpa': data['gpa']}, distr=cloglog)
        resf = modf.fit(method='bfgs', disp=False)
        modu = OrderedModel(data_unordered['apply'].values.codes, np.asarray(data_unordered[['pared', 'public', 'gpa']], float), distr=cloglog)
        resu = modu.fit(method='bfgs', disp=False)
        from .results.results_ordinal_model import res_ord_cloglog as res2
        cls.res2 = res2
        cls.res1 = res
        cls.resp = resp
        cls.resf = resf
        cls.resu = resu

class TestLogitBinary:

    def test_attributes(self):
        if False:
            while True:
                i = 10
        data = ds.df
        mask_drop = data['apply'] == 'somewhat likely'
        data2 = data.loc[~mask_drop, :].copy()
        data2['apply'] = data2['apply'].cat.remove_categories('somewhat likely')
        modp = OrderedModel(data2['apply'], data2[['pared', 'public', 'gpa']], distr='logit')
        resp = modp.fit(method='bfgs', disp=False)
        exog = add_constant(data2[['pared', 'public', 'gpa']], prepend=False)
        mod_logit = Logit(data2['apply'].cat.codes, exog)
        res_logit = mod_logit.fit()
        attributes = 'bse df_resid llf aic bic llnull'.split()
        attributes += 'llnull llr llr_pvalue prsquared'.split()
        params = np.asarray(resp.params)
        logit_params = np.asarray(res_logit.params)
        assert_allclose(params[:3], logit_params[:3], rtol=1e-05)
        assert_allclose(params[3], -logit_params[3], rtol=1e-05)
        for attr in attributes:
            assert_allclose(getattr(resp, attr), getattr(res_logit, attr), rtol=0.0001)
        resp = modp.fit(method='bfgs', disp=False, cov_type='hac', cov_kwds={'maxlags': 2})
        res_logit = mod_logit.fit(method='bfgs', disp=False, cov_type='hac', cov_kwds={'maxlags': 2})
        for attr in attributes:
            assert_allclose(getattr(resp, attr), getattr(res_logit, attr), rtol=0.0001)
        resp = modp.fit(method='bfgs', disp=False, cov_type='hc1')
        res_logit = mod_logit.fit(method='bfgs', disp=False, cov_type='hc1')
        for attr in attributes:
            assert_allclose(getattr(resp, attr), getattr(res_logit, attr), rtol=0.0001)

def test_nan_endog_exceptions():
    if False:
        for i in range(10):
            print('nop')
    nobs = 15
    y = np.repeat(np.arange(3), nobs // 3)
    x = np.column_stack((np.ones(nobs), np.arange(nobs)))
    with pytest.raises(ValueError, match='not be a constant'):
        OrderedModel(y, x, distr='logit')
    y_nan = y.astype(float)
    y_nan[0] = np.nan
    with pytest.raises(ValueError, match='NaN in dependent variable'):
        OrderedModel(y_nan, x[:, 1:], distr='logit')
    if hasattr(pd, 'CategoricalDtype'):
        df = pd.DataFrame({'endog': pd.Series(y, dtype=pd.CategoricalDtype([1, 2, 3], ordered=True)), 'exog': x[:, 1]})
        msg = 'missing values in categorical endog'
        with pytest.raises(ValueError, match=msg):
            OrderedModel(df['endog'], df[['exog']])