import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels import datasets
from statsmodels.tools.tools import add_constant
from statsmodels.tools.testing import Holder
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.distributions.discrete import truncatedpoisson, truncatednegbin
from statsmodels.discrete.truncated_model import TruncatedLFPoisson, TruncatedLFNegativeBinomialP, HurdleCountModel
from statsmodels.sandbox.regression.tests.test_gmm_poisson import DATA
from .results.results_discrete import RandHIE
from .results import results_truncated as results_t
from .results import results_truncated_st as results_ts

class CheckResults:

    def test_params(self):
        if False:
            while True:
                i = 10
        assert_allclose(self.res1.params, self.res2.params, atol=1e-05, rtol=1e-05)

    def test_llf(self):
        if False:
            for i in range(10):
                print('nop')
        assert_allclose(self.res1.llf, self.res2.llf, atol=1e-05, rtol=1e-07)

    def test_conf_int(self):
        if False:
            while True:
                i = 10
        assert_allclose(self.res1.conf_int(), self.res2.conf_int, atol=0.001, rtol=1e-05)

    def test_bse(self):
        if False:
            for i in range(10):
                print('nop')
        assert_allclose(self.res1.bse, self.res2.bse, atol=0.001)

    def test_aic(self):
        if False:
            print('Hello World!')
        assert_allclose(self.res1.aic, self.res2.aic, atol=0.01, rtol=1e-12)

    def test_bic(self):
        if False:
            return 10
        assert_allclose(self.res1.bic, self.res2.bic, atol=0.01, rtol=1e-12)

    def test_fit_regularized(self):
        if False:
            while True:
                i = 10
        model = self.res1.model
        alpha = np.ones(len(self.res1.params))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=ConvergenceWarning)
            res_reg = model.fit_regularized(alpha=alpha * 0.01, disp=0)
        assert_allclose(res_reg.params, self.res1.params, rtol=0.001, atol=0.005)
        assert_allclose(res_reg.bse, self.res1.bse, rtol=0.001, atol=0.005)

class TestTruncatedLFPoissonModel(CheckResults):

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        data = datasets.randhie.load()
        exog = add_constant(np.asarray(data.exog)[:, :4], prepend=False)
        mod = TruncatedLFPoisson(data.endog, exog, truncation=5)
        cls.res1 = mod.fit(method='newton', maxiter=500)
        res2 = RandHIE()
        res2.truncated_poisson()
        cls.res2 = res2

class TestZeroTruncatedLFPoissonModel(CheckResults):

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        data = datasets.randhie.load()
        exog = add_constant(np.asarray(data.exog)[:, :4], prepend=False)
        mod = TruncatedLFPoisson(data.endog, exog, truncation=0)
        cls.res1 = mod.fit(maxiter=500)
        res2 = RandHIE()
        res2.zero_truncated_poisson()
        cls.res2 = res2

class TestZeroTruncatedNBPModel(CheckResults):

    @classmethod
    def setup_class(cls):
        if False:
            i = 10
            return i + 15
        data = datasets.randhie.load()
        exog = add_constant(np.asarray(data.exog)[:, :3], prepend=False)
        mod = TruncatedLFNegativeBinomialP(data.endog, exog, truncation=0)
        cls.res1 = mod.fit(maxiter=500)
        res2 = RandHIE()
        res2.zero_truncted_nbp()
        cls.res2 = res2

    def test_conf_int(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class TestTruncatedLFPoisson_predict:

    @classmethod
    def setup_class(cls):
        if False:
            i = 10
            return i + 15
        cls.expected_params = [1, 0.5]
        np.random.seed(123)
        nobs = 200
        exog = np.ones((nobs, 2))
        exog[:nobs // 2, 1] = 2
        mu_true = exog.dot(cls.expected_params)
        cls.endog = truncatedpoisson.rvs(mu_true, 0, size=mu_true.shape)
        model = TruncatedLFPoisson(cls.endog, exog, truncation=0)
        cls.res = model.fit(method='bfgs', maxiter=5000)

    def test_mean(self):
        if False:
            i = 10
            return i + 15
        assert_allclose(self.res.predict().mean(), self.endog.mean(), atol=0.2, rtol=0.2)

    def test_var(self):
        if False:
            while True:
                i = 10
        v = self.res.predict(which='var').mean()
        assert_allclose(v, self.endog.var(), atol=0.2, rtol=0.2)
        return
        assert_allclose(self.res.predict().mean() * self.res._dispersion_factor.mean(), self.endog.var(), atol=0.05, rtol=0.05)

    def test_predict_prob(self):
        if False:
            print('Hello World!')
        res = self.res
        pr = res.predict(which='prob')
        pr2 = truncatedpoisson.pmf(np.arange(8), res.predict(which='mean-main')[:, None], 0)
        assert_allclose(pr, pr2, rtol=1e-10, atol=1e-10)

class TestTruncatedNBP_predict:

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        cls.expected_params = [1, 0.5, 0.5]
        np.random.seed(1234)
        nobs = 200
        exog = np.ones((nobs, 2))
        exog[:nobs // 2, 1] = 2
        mu_true = np.exp(exog.dot(cls.expected_params[:-1]))
        cls.endog = truncatednegbin.rvs(mu_true, cls.expected_params[-1], 2, 0, size=mu_true.shape)
        model = TruncatedLFNegativeBinomialP(cls.endog, exog, truncation=0, p=2)
        cls.res = model.fit(method='nm', maxiter=5000, maxfun=5000)

    def test_mean(self):
        if False:
            i = 10
            return i + 15
        assert_allclose(self.res.predict().mean(), self.endog.mean(), atol=0.2, rtol=0.2)

    def test_var(self):
        if False:
            while True:
                i = 10
        v = self.res.predict(which='var').mean()
        assert_allclose(v, self.endog.var(), atol=0.1, rtol=0.01)
        return
        assert_allclose(self.res.predict().mean() * self.res._dispersion_factor.mean(), self.endog.var(), atol=0.05, rtol=0.05)

    def test_predict_prob(self):
        if False:
            i = 10
            return i + 15
        res = self.res
        pr = res.predict(which='prob')
        pr2 = truncatednegbin.pmf(np.arange(29), res.predict(which='mean-main')[:, None], res.params[-1], 2, 0)
        assert_allclose(pr, pr2, rtol=1e-10, atol=1e-10)

class CheckTruncatedST:

    def test_basic(self):
        if False:
            return 10
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.llf, res2.ll, rtol=1e-08)
        assert_allclose(res1.llnull, res2.ll_0, rtol=5e-06)
        pt2 = res2.params_table
        k = res1.model.exog.shape[1]
        assert_allclose(res1.params[:k], res2.params[:k], atol=1e-05)
        assert_allclose(res1.bse[:k], pt2[:k, 1], atol=1e-05)
        assert_allclose(res1.tvalues[:k], pt2[:k, 2], rtol=0.0005, atol=0.0005)
        assert_allclose(res1.pvalues[:k], pt2[:k, 3], rtol=0.0005, atol=1e-07)
        assert_equal(res1.df_model, res2.df_m)
        assert_allclose(res1.aic, res2.icr[-2], rtol=1e-08)
        assert_allclose(res1.bic, res2.icr[-1], rtol=1e-08)
        nobs = res1.model.endog.shape[0]
        assert_equal((res1.model.endog < 1).sum(), 0)
        assert_equal(res1.df_resid, nobs - len(res1.params))

    def test_predict(self):
        if False:
            for i in range(10):
                print('nop')
        res1 = self.res1
        res2 = self.res2
        rdf = res2.margins_means.table
        pred = res1.get_prediction(which='mean-main', average=True)
        assert_allclose(pred.predicted, rdf[0], rtol=5e-05)
        assert_allclose(pred.se, rdf[1], rtol=0.0005, atol=1e-10)
        ci = pred.conf_int()[0]
        assert_allclose(ci[0], rdf[4], rtol=1e-05, atol=1e-10)
        assert_allclose(ci[1], rdf[5], rtol=1e-05, atol=1e-10)
        ex = res1.model.exog.mean(0)
        rdf = res2.margins_atmeans.table
        pred = res1.get_prediction(ex, which='mean-main')
        assert_allclose(pred.predicted, rdf[0], rtol=5e-05)
        assert_allclose(pred.se, rdf[1], rtol=0.0005, atol=1e-10)
        ci = pred.conf_int()[0]
        assert_allclose(ci[0], rdf[4], rtol=5e-05, atol=1e-10)
        assert_allclose(ci[1], rdf[5], rtol=5e-05, atol=1e-10)
        rdf = res2.margins_cm.table
        try:
            pred = res1.get_prediction(average=True)
        except NotImplementedError:
            pred = None
        if pred is not None:
            assert_allclose(pred.predicted, rdf[0], rtol=5e-05)
            assert_allclose(pred.se, rdf[1], rtol=1e-05, atol=1e-10)
            ci = pred.conf_int()[0]
            assert_allclose(ci[0], rdf[4], rtol=1e-05, atol=1e-10)
            assert_allclose(ci[1], rdf[5], rtol=1e-05, atol=1e-10)
        ex = res1.model.exog.mean(0)
        rdf = res2.margins_cpr.table
        start_idx = res1.model.truncation + 1
        k = rdf.shape[0] + res1.model.truncation
        pred = res1.get_prediction(which='prob', average=True)
        assert_allclose(pred.predicted[start_idx:k], rdf[:-1, 0], rtol=5e-05)
        assert_allclose(pred.se[start_idx:k], rdf[:-1, 1], rtol=0.0005, atol=1e-10)
        ci = pred.conf_int()[start_idx:k]
        assert_allclose(ci[:, 0], rdf[:-1, 4], rtol=5e-05, atol=1e-10)
        assert_allclose(ci[:, 1], rdf[:-1, 5], rtol=5e-05, atol=1e-10)
        ex = res1.model.exog.mean(0)
        rdf = res2.margins_pr.table
        k = rdf.shape[0] - 1
        pred = res1.get_prediction(which='prob-base', average=True)
        assert_allclose(pred.predicted[:k], rdf[:-1, 0], rtol=5e-05)
        assert_allclose(pred.se[:k], rdf[:-1, 1], rtol=0.0008, atol=1e-10)
        ci = pred.conf_int()[:k]
        assert_allclose(ci[:, 0], rdf[:-1, 4], rtol=0.0005, atol=1e-10)
        assert_allclose(ci[:, 1], rdf[:-1, 5], rtol=0.0005, atol=1e-10)

class TestTruncatedLFPoissonSt(CheckTruncatedST):

    @classmethod
    def setup_class(cls):
        if False:
            i = 10
            return i + 15
        endog = DATA['docvis']
        exog_names = ['aget', 'totchr', 'const']
        exog = DATA[exog_names]
        cls.res1 = TruncatedLFPoisson(endog, exog).fit(method='bfgs', maxiter=300)
        cls.res2 = results_ts.results_trunc_poisson
        mod_offset = TruncatedLFPoisson(endog, exog, offset=DATA['aget'])
        cls.res_offset = mod_offset.fit(method='bfgs', maxiter=300)

    def test_offset(self):
        if False:
            return 10
        res1 = self.res1
        reso = self.res_offset
        paramso = np.asarray(reso.params)
        params1 = np.asarray(res1.params)
        assert_allclose(paramso[1:], params1[1:], rtol=1e-08)
        assert_allclose(paramso[0], params1[0] - 1, rtol=1e-08)
        pred1 = res1.predict()
        predo = reso.predict()
        assert_allclose(predo, pred1, rtol=1e-08)
        ex = res1.model.exog[:5]
        offs = reso.model.offset[:5]
        pred1 = res1.predict(ex, transform=False)
        predo = reso.predict(ex, offset=offs, transform=False)
        assert_allclose(predo, pred1, rtol=1e-08)

class TestTruncatedNegBinSt(CheckTruncatedST):

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        endog = DATA['docvis']
        exog_names = ['aget', 'totchr', 'const']
        exog = DATA[exog_names]
        cls.res1 = TruncatedLFNegativeBinomialP(endog, exog).fit(method='bfgs', maxiter=300)
        cls.res2 = results_ts.results_trunc_negbin
        mod_offset = TruncatedLFNegativeBinomialP(endog, exog, offset=DATA['aget'])
        cls.res_offset = mod_offset.fit(method='bfgs', maxiter=300)

    def test_offset(self):
        if False:
            i = 10
            return i + 15
        res1 = self.res1
        reso = self.res_offset
        paramso = np.asarray(reso.params)
        params1 = np.asarray(res1.params)
        assert_allclose(paramso[1:], params1[1:], rtol=1e-08)
        assert_allclose(paramso[0], params1[0] - 1, rtol=1e-08)
        pred1 = res1.predict()
        predo = reso.predict()
        assert_allclose(predo, pred1, rtol=1e-08)
        ex = res1.model.exog[:5]
        offs = reso.model.offset[:5]
        pred1 = res1.predict(ex, transform=False)
        predo = reso.predict(ex, offset=offs, transform=False)
        assert_allclose(predo, pred1, rtol=1e-08)

class TestTruncatedLFPoisson1St(CheckTruncatedST):

    @classmethod
    def setup_class(cls):
        if False:
            return 10
        endog = DATA['docvis']
        exog_names = ['aget', 'totchr', 'const']
        exog = DATA[exog_names]
        cls.res1 = TruncatedLFPoisson(endog, exog, truncation=1).fit(method='bfgs', maxiter=300)
        cls.res2 = results_ts.results_trunc_poisson1

class TestTruncatedNegBin1St(CheckTruncatedST):

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        endog = DATA['docvis']
        exog_names = ['aget', 'totchr', 'const']
        exog = DATA[exog_names]
        cls.res1 = TruncatedLFNegativeBinomialP(endog, exog, truncation=1).fit(method='newton', maxiter=300)
        cls.res2 = results_ts.results_trunc_negbin1

class TestHurdlePoissonR:

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        endog = DATA['docvis']
        exog_names = ['const', 'aget', 'totchr']
        exog = DATA[exog_names]
        cls.res1 = HurdleCountModel(endog, exog).fit(method='newton', maxiter=300)
        cls.res2 = results_t.hurdle_poisson

    def test_basic(self):
        if False:
            while True:
                i = 10
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.llf, res2.loglik, rtol=1e-08)
        pt2 = res2.params_table
        assert_allclose(res1.params, pt2[:, 0], atol=1e-05)
        assert_allclose(res1.bse, pt2[:, 1], atol=1e-05)
        assert_allclose(res1.tvalues, pt2[:, 2], rtol=0.0005, atol=0.0005)
        assert_allclose(res1.pvalues, pt2[:, 3], rtol=0.0005, atol=1e-07)
        assert_equal(res1.df_resid, res2.df_residual)
        assert_equal(res1.df_model, res2.df_null - res2.df_residual)
        assert_allclose(res1.aic, res2.aic, rtol=1e-08)
        idx = np.concatenate((np.arange(3, 6), np.arange(3)))
        vcov = res2.vcov[idx[:, None], idx]
        assert_allclose(np.asarray(res1.cov_params()), vcov, rtol=0.0001, atol=1e-08)

    def test_predict(self):
        if False:
            return 10
        res1 = self.res1
        res2 = self.res2
        ex = res1.model.exog.mean(0, keepdims=True)
        mu1 = res1.results_zero.predict(ex)
        prob_zero = np.exp(-mu1)
        prob_nz = 1 - prob_zero
        assert_allclose(prob_nz, res2.predict_zero, rtol=0.0005, atol=0.0005)
        prob_nz_ = res1.results_zero.model._prob_nonzero(mu1, res1.params[:4])
        assert_allclose(prob_nz_, res2.predict_zero, rtol=0.0005, atol=0.0005)
        mean_main = res1.results_count.predict(ex, which='mean-main')
        assert_allclose(mean_main, res2.predict_mean_main, rtol=0.0005, atol=0.0005)
        prob_main = res1.results_count.predict(ex, which='prob')[0] * prob_nz
        prob_main[0] = np.squeeze(prob_zero)
        assert_allclose(prob_main[:4], res2.predict_prob, rtol=0.0005, atol=0.0005)
        assert_allclose(mean_main * prob_nz, res2.predict_mean, rtol=0.001, atol=0.0005)
        m = res1.predict(ex)
        assert_allclose(m, res2.predict_mean, rtol=1e-06, atol=5e-07)
        mm = res1.predict(ex, which='mean-main')
        assert_allclose(mm, res2.predict_mean_main, rtol=1e-07, atol=1e-07)
        mnz = res1.predict(ex, which='mean-nonzero')
        assert_allclose(mnz, res2.predict_mean / (1 - res2.predict_prob[0]), rtol=5e-07, atol=5e-07)
        prob_main = res1.predict(ex, which='prob-main')
        pt = res1.predict(ex, which='prob-trunc')
        assert_allclose(prob_main / (1 - pt), res2.predict_zero, rtol=0.0005, atol=0.0005)
        probs = res1.predict(ex, which='prob')[0]
        assert_allclose(probs[:4], res2.predict_prob, rtol=1e-05, atol=1e-06)
        k_ex = 5
        ex5 = res1.model.exog[:k_ex]
        p1a = res1.predict(ex5, which='prob', y_values=np.arange(3))
        p1b = res1.get_prediction(ex5, which='prob', y_values=np.arange(3))
        assert_allclose(p1a, p1b.predicted, rtol=1e-10, atol=1e-10)
        p2a = res1.predict(which='prob', y_values=np.arange(3))
        p2b = res1.get_prediction(which='prob', y_values=np.arange(3), average=True)
        assert_allclose(p2a.mean(0), p2b.predicted, rtol=1e-10, atol=1e-10)
        for which in ['mean', 'mean-main', 'prob-main', 'prob-zero', 'linear']:
            p3a = res1.predict(ex5, which=which)
            p3b = res1.get_prediction(ex5, which=which)
            assert_allclose(p3a, p3b.predicted, rtol=1e-10, atol=1e-10)
            assert p3b.summary_frame().shape == (k_ex, 4)
        resid_p1 = res1.resid_pearson[:5]
        resid_p2 = np.asarray([-1.5892397298897, -0.3239276467705, -1.5878941800178, 0.6613236544236, -0.6690997162962])
        assert_allclose(resid_p1, resid_p2, rtol=1e-05, atol=1e-05)

class CheckHurdlePredict:

    def test_basic(self):
        if False:
            print('Hello World!')
        res1 = self.res1
        res2 = self.res2
        assert res1.df_model == res2.df_model
        assert res1.df_resid == res2.df_resid
        assert res1.model.k_extra == res2.k_extra
        assert len(res1.model.exog_names) == res2.k_params
        assert res1.model.exog_names == res2.exog_names
        res1.summary()

    def test_predict(self):
        if False:
            return 10
        res1 = self.res1
        endog = res1.model.endog
        exog = res1.model.exog
        pred_mean = res1.predict(which='mean').mean()
        assert_allclose(pred_mean, endog.mean(), rtol=0.01)
        mask_nz = endog > 0
        mean_nz = endog[mask_nz].mean()
        pred_mean_nz = res1.predict(which='mean-nonzero').mean()
        assert_allclose(pred_mean_nz, mean_nz, rtol=0.05)
        pred_mean_nnz = res1.predict(exog=exog[mask_nz], which='mean-nonzero').mean()
        assert_allclose(pred_mean_nnz, mean_nz, rtol=0.0005)
        pred_mean_nzm = res1.results_count.predict(which='mean').mean()
        assert_allclose(pred_mean_nzm, mean_nz, rtol=0.0005)
        assert_allclose(pred_mean_nzm, pred_mean_nnz, rtol=0.0001)
        pred_var = res1.predict(which='var').mean()
        assert_allclose(pred_var, res1.resid.var(), rtol=0.05)
        pred_var = res1.results_count.predict(which='var').mean()
        assert_allclose(pred_var, res1.resid[endog > 0].var(), rtol=0.05)
        freq = np.bincount(endog.astype(int)) / len(endog)
        pred_prob = res1.predict(which='prob').mean(0)
        assert_allclose(pred_prob, freq, rtol=0.005, atol=0.01)
        dia_hnb = res1.get_diagnostic()
        assert_allclose(dia_hnb.probs_predicted.mean(0), pred_prob, rtol=1e-10)
        try:
            dia_hnb.plot_probs()
        except ImportError:
            pass
        pred_prob0 = res1.predict(which='prob-zero').mean(0)
        assert_allclose(pred_prob0, freq[0], rtol=0.0001)
        assert_allclose(pred_prob0, pred_prob[0], rtol=1e-10)

class TestHurdleNegbinSimulated(CheckHurdlePredict):

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        nobs = 2000
        exog = np.column_stack((np.ones(nobs), np.linspace(0, 3, nobs)))
        y_fake = np.arange(nobs) // (nobs / 3)
        mod = HurdleCountModel(y_fake, exog, dist='negbin', zerodist='negbin')
        p_dgp = np.array([-0.4, 2, 0.5, 0.2, 0.5, 0.5])
        probs = mod.predict(p_dgp, which='prob', y_values=np.arange(50))
        cdf = probs.cumsum(1)
        n = cdf.shape[0]
        cdf = np.column_stack((cdf, np.ones(n)))
        rng = np.random.default_rng(987456348)
        u = rng.random((n, 1))
        endog = np.argmin(cdf < u, axis=1)
        mod_hnb = HurdleCountModel(endog, exog, dist='negbin', zerodist='negbin')
        cls.res1 = mod_hnb.fit(maxiter=300)
        df_null = 4
        cls.res2 = Holder(nobs=nobs, k_params=6, df_model=2, df_null=df_null, df_resid=nobs - 6, k_extra=df_null - 1, exog_names=['zm_const', 'zm_x1', 'zm_alpha', 'const', 'x1', 'alpha'])