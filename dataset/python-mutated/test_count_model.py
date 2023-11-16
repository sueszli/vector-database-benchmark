from statsmodels.compat.platform import PLATFORM_LINUX32
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_array_equal, assert_equal
import pandas as pd
import pytest
import statsmodels.api as sm
from .results.results_discrete import RandHIE
from .test_discrete import CheckModelMixin

class CheckGeneric(CheckModelMixin):

    def test_params(self):
        if False:
            i = 10
            return i + 15
        assert_allclose(self.res1.params, self.res2.params, atol=1e-05, rtol=1e-05)

    def test_llf(self):
        if False:
            print('Hello World!')
        assert_allclose(self.res1.llf, self.res2.llf, atol=1e-05, rtol=1e-05)

    def test_conf_int(self):
        if False:
            return 10
        assert_allclose(self.res1.conf_int(), self.res2.conf_int, atol=0.001, rtol=1e-05)

    def test_bse(self):
        if False:
            return 10
        assert_allclose(self.res1.bse, self.res2.bse, atol=0.001, rtol=0.001)

    def test_aic(self):
        if False:
            while True:
                i = 10
        assert_allclose(self.res1.aic, self.res2.aic, atol=0.01, rtol=0.01)

    def test_bic(self):
        if False:
            print('Hello World!')
        assert_allclose(self.res1.aic, self.res2.aic, atol=0.1, rtol=0.1)

    def test_t(self):
        if False:
            for i in range(10):
                print('nop')
        unit_matrix = np.identity(self.res1.params.size)
        t_test = self.res1.t_test(unit_matrix)
        assert_allclose(self.res1.tvalues, t_test.tvalue)

    def test_fit_regularized(self):
        if False:
            for i in range(10):
                print('nop')
        model = self.res1.model
        alpha = np.ones(len(self.res1.params))
        alpha[-2:] = 0
        res_reg = model.fit_regularized(alpha=alpha * 0.01, disp=False, maxiter=500)
        assert_allclose(res_reg.params[2:], self.res1.params[2:], atol=0.05, rtol=0.05)

    def test_init_keys(self):
        if False:
            i = 10
            return i + 15
        init_kwds = self.res1.model._get_init_kwds()
        assert_equal(set(init_kwds.keys()), set(self.init_keys))
        for (key, value) in self.init_kwds.items():
            assert_equal(init_kwds[key], value)

    def test_null(self):
        if False:
            i = 10
            return i + 15
        self.res1.llnull
        exog_null = self.res1.res_null.model.exog
        exog_infl_null = self.res1.res_null.model.exog_infl
        assert_array_equal(exog_infl_null.shape, (len(self.res1.model.exog), 1))
        assert_equal(np.ptp(exog_null), 0)
        assert_equal(np.ptp(exog_infl_null), 0)

    @pytest.mark.smoke
    def test_summary(self):
        if False:
            return 10
        summ = self.res1.summary()
        assert 'Covariance Type:' in str(summ)

class TestZeroInflatedModel_logit(CheckGeneric):

    @classmethod
    def setup_class(cls):
        if False:
            i = 10
            return i + 15
        data = sm.datasets.randhie.load()
        cls.endog = np.asarray(data.endog)
        data.exog = np.asarray(data.exog)
        exog = sm.add_constant(data.exog[:, 1:4], prepend=False)
        exog_infl = sm.add_constant(data.exog[:, 0], prepend=False)
        cls.res1 = sm.ZeroInflatedPoisson(data.endog, exog, exog_infl=exog_infl, inflation='logit').fit(method='newton', maxiter=500, disp=False)
        cls.res1._results._attach_nullmodel = True
        cls.init_keys = ['exog_infl', 'exposure', 'inflation', 'offset']
        cls.init_kwds = {'inflation': 'logit'}
        res2 = RandHIE.zero_inflated_poisson_logit
        cls.res2 = res2

class TestZeroInflatedModel_probit(CheckGeneric):

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        data = sm.datasets.randhie.load()
        cls.endog = np.asarray(data.endog)
        data.exog = np.asarray(data.exog)
        exog = sm.add_constant(data.exog[:, 1:4], prepend=False)
        exog_infl = sm.add_constant(data.exog[:, 0], prepend=False)
        cls.res1 = sm.ZeroInflatedPoisson(data.endog, exog, exog_infl=exog_infl, inflation='probit').fit(method='newton', maxiter=500, disp=False)
        cls.res1._results._attach_nullmodel = True
        cls.init_keys = ['exog_infl', 'exposure', 'inflation', 'offset']
        cls.init_kwds = {'inflation': 'probit'}
        res2 = RandHIE.zero_inflated_poisson_probit
        cls.res2 = res2

    @pytest.mark.skipif(PLATFORM_LINUX32, reason='Fails on 32-bit Linux')
    def test_fit_regularized(self):
        if False:
            print('Hello World!')
        super().test_fit_regularized()

class TestZeroInflatedModel_offset(CheckGeneric):

    @classmethod
    def setup_class(cls):
        if False:
            return 10
        data = sm.datasets.randhie.load()
        cls.endog = np.asarray(data.endog)
        data.exog = np.asarray(data.exog)
        exog = sm.add_constant(data.exog[:, 1:4], prepend=False)
        exog_infl = sm.add_constant(data.exog[:, 0], prepend=False)
        cls.res1 = sm.ZeroInflatedPoisson(data.endog, exog, exog_infl=exog_infl, offset=data.exog[:, 7]).fit(method='newton', maxiter=500, disp=False)
        cls.res1._results._attach_nullmodel = True
        cls.init_keys = ['exog_infl', 'exposure', 'inflation', 'offset']
        cls.init_kwds = {'inflation': 'logit'}
        res2 = RandHIE.zero_inflated_poisson_offset
        cls.res2 = res2

    def test_exposure(self):
        if False:
            for i in range(10):
                print('nop')
        model1 = self.res1.model
        offset = model1.offset
        model3 = sm.ZeroInflatedPoisson(model1.endog, model1.exog, exog_infl=model1.exog_infl, exposure=np.exp(offset))
        res3 = model3.fit(start_params=self.res1.params, method='newton', maxiter=500, disp=False)
        assert_allclose(res3.params, self.res1.params, atol=1e-06, rtol=1e-06)
        fitted1 = self.res1.predict()
        fitted3 = res3.predict()
        assert_allclose(fitted3, fitted1, atol=1e-06, rtol=1e-06)
        ex = model1.exog
        ex_infl = model1.exog_infl
        offset = model1.offset
        fitted1_0 = self.res1.predict(exog=ex, exog_infl=ex_infl, offset=offset.tolist())
        fitted3_0 = res3.predict(exog=ex, exog_infl=ex_infl, exposure=np.exp(offset))
        assert_allclose(fitted3_0, fitted1_0, atol=1e-06, rtol=1e-06)
        ex = model1.exog[:10:2]
        ex_infl = model1.exog_infl[:10:2]
        offset = offset[:10:2]
        fitted1_2 = self.res1.predict(exog=ex, exog_infl=ex_infl, offset=offset)
        fitted3_2 = res3.predict(exog=ex, exog_infl=ex_infl, exposure=np.exp(offset))
        assert_allclose(fitted3_2, fitted1_2, atol=1e-06, rtol=1e-06)
        assert_allclose(fitted1_2, fitted1[:10:2], atol=1e-06, rtol=1e-06)
        assert_allclose(fitted3_2, fitted1[:10:2], atol=1e-06, rtol=1e-06)
        fitted1_3 = self.res1.predict(exog=ex, exog_infl=ex_infl)
        fitted3_3 = res3.predict(exog=ex, exog_infl=ex_infl)
        assert_allclose(fitted3_3, fitted1_3, atol=1e-06, rtol=1e-06)

class TestZeroInflatedModelPandas(CheckGeneric):

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        data = sm.datasets.randhie.load_pandas()
        cls.endog = data.endog
        cls.data = data
        exog = sm.add_constant(data.exog.iloc[:, 1:4], prepend=False)
        exog_infl = sm.add_constant(data.exog.iloc[:, 0], prepend=False)
        start_params = np.asarray([0.10337834587498942, -1.0459825102508549, -0.08219794475894268, 0.00856917434709146, -0.026795737379474334, 1.4823632430107334])
        model = sm.ZeroInflatedPoisson(data.endog, exog, exog_infl=exog_infl, inflation='logit')
        cls.res1 = model.fit(start_params=start_params, method='newton', maxiter=500, disp=False)
        cls.res1._results._attach_nullmodel = True
        cls.init_keys = ['exog_infl', 'exposure', 'inflation', 'offset']
        cls.init_kwds = {'inflation': 'logit'}
        res2 = RandHIE.zero_inflated_poisson_logit
        cls.res2 = res2

    def test_names(self):
        if False:
            return 10
        param_names = ['inflate_lncoins', 'inflate_const', 'idp', 'lpi', 'fmde', 'const']
        assert_array_equal(self.res1.model.exog_names, param_names)
        assert_array_equal(self.res1.params.index.tolist(), param_names)
        assert_array_equal(self.res1.bse.index.tolist(), param_names)
        exog = sm.add_constant(self.data.exog.iloc[:, 1:4], prepend=True)
        exog_infl = sm.add_constant(self.data.exog.iloc[:, 0], prepend=True)
        param_names = ['inflate_const', 'inflate_lncoins', 'const', 'idp', 'lpi', 'fmde']
        model = sm.ZeroInflatedPoisson(self.data.endog, exog, exog_infl=exog_infl, inflation='logit')
        assert_array_equal(model.exog_names, param_names)

class TestZeroInflatedPoisson_predict:

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        expected_params = [1, 0.5]
        np.random.seed(999)
        nobs = 2000
        exog = np.ones((nobs, 2))
        exog[:nobs // 2, 1] = 2
        mu_true = exog.dot(expected_params)
        cls.endog = sm.distributions.zipoisson.rvs(mu_true, 0.05, size=mu_true.shape)
        model = sm.ZeroInflatedPoisson(cls.endog, exog)
        cls.res = model.fit(method='bfgs', maxiter=5000, disp=False)
        cls.params_true = [mu_true, 0.05, nobs]

    def test_mean(self):
        if False:
            for i in range(10):
                print('nop')

        def compute_conf_interval_95(mu, prob_infl, nobs):
            if False:
                print('Hello World!')
            dispersion_factor = 1 + prob_infl * mu
            var = (dispersion_factor * (1 - prob_infl) * mu).mean()
            var += (((1 - prob_infl) * mu) ** 2).mean()
            var -= ((1 - prob_infl) * mu).mean() ** 2
            std = np.sqrt(var)
            conf_intv_95 = 2 * std / np.sqrt(nobs)
            return conf_intv_95
        conf_interval_95 = compute_conf_interval_95(*self.params_true)
        assert_allclose(self.res.predict().mean(), self.endog.mean(), atol=conf_interval_95, rtol=0)

    def test_var(self):
        if False:
            for i in range(10):
                print('nop')

        def compute_mixture_var(dispersion_factor, prob_main, mu):
            if False:
                return 10
            prob_infl = 1 - prob_main
            var = (dispersion_factor * (1 - prob_infl) * mu).mean()
            var += (((1 - prob_infl) * mu) ** 2).mean()
            var -= ((1 - prob_infl) * mu).mean() ** 2
            return var
        res = self.res
        var_fitted = compute_mixture_var(res._dispersion_factor, res.predict(which='prob-main'), res.predict(which='mean-main'))
        assert_allclose(var_fitted.mean(), self.endog.var(), atol=0.05, rtol=0.05)

    def test_predict_prob(self):
        if False:
            while True:
                i = 10
        res = self.res
        pr = res.predict(which='prob')
        pr2 = sm.distributions.zipoisson.pmf(np.arange(pr.shape[1])[:, None], res.predict(), 0.05).T
        assert_allclose(pr, pr2, rtol=0.05, atol=0.05)

    def test_predict_options(self):
        if False:
            i = 10
            return i + 15
        res = self.res
        n = 5
        pr1 = res.predict(which='prob')
        pr0 = res.predict(exog=res.model.exog[:n], which='prob')
        assert_allclose(pr0, pr1[:n], rtol=1e-10)
        fitted1 = res.predict()
        fitted0 = res.predict(exog=res.model.exog[:n])
        assert_allclose(fitted0, fitted1[:n], rtol=1e-10)

@pytest.mark.slow
class TestZeroInflatedGeneralizedPoisson(CheckGeneric):

    @classmethod
    def setup_class(cls):
        if False:
            return 10
        data = sm.datasets.randhie.load()
        cls.endog = np.asarray(data.endog)
        data.exog = np.asarray(data.exog)
        exog = sm.add_constant(data.exog[:, 1:4], prepend=False)
        exog_infl = sm.add_constant(data.exog[:, 0], prepend=False)
        cls.res1 = sm.ZeroInflatedGeneralizedPoisson(data.endog, exog, exog_infl=exog_infl, p=1).fit(method='newton', maxiter=500, disp=False)
        cls.res1._results._attach_nullmodel = True
        cls.init_keys = ['exog_infl', 'exposure', 'inflation', 'offset', 'p']
        cls.init_kwds = {'inflation': 'logit', 'p': 1}
        res2 = RandHIE.zero_inflated_generalized_poisson
        cls.res2 = res2

    def test_bse(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_conf_int(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_bic(self):
        if False:
            return 10
        pass

    def test_t(self):
        if False:
            print('Hello World!')
        unit_matrix = np.identity(self.res1.params.size)
        t_test = self.res1.t_test(unit_matrix)
        assert_allclose(self.res1.tvalues, t_test.tvalue)

    def test_minimize(self, reset_randomstate):
        if False:
            while True:
                i = 10
        model = self.res1.model
        start_params = self.res1.mle_settings['start_params']
        res_ncg = model.fit(start_params=start_params, method='minimize', min_method='trust-ncg', maxiter=500, disp=False)
        assert_allclose(res_ncg.params, self.res2.params, atol=0.001, rtol=0.04)
        assert_allclose(res_ncg.bse, self.res2.bse, atol=0.001, rtol=0.6)
        assert_(res_ncg.mle_retvals['converged'] is True)
        res_dog = model.fit(start_params=start_params, method='minimize', min_method='dogleg', maxiter=500, disp=False)
        assert_allclose(res_dog.params, self.res2.params, atol=0.001, rtol=0.003)
        assert_allclose(res_dog.bse, self.res2.bse, atol=0.001, rtol=0.6)
        assert_(res_dog.mle_retvals['converged'] is True)
        random_state = np.random.RandomState(1)
        seed = {'seed': random_state}
        res_bh = model.fit(start_params=start_params, method='basinhopping', niter=500, stepsize=0.1, niter_success=None, disp=False, interval=1, **seed)
        assert_allclose(res_bh.params, self.res2.params, atol=0.0001, rtol=0.0001)
        assert_allclose(res_bh.bse, self.res2.bse, atol=0.001, rtol=0.6)

class TestZeroInflatedGeneralizedPoisson_predict:

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        expected_params = [1, 0.5, 0.5]
        np.random.seed(999)
        nobs = 2000
        exog = np.ones((nobs, 2))
        exog[:nobs // 2, 1] = 2
        mu_true = exog.dot(expected_params[:-1])
        cls.endog = sm.distributions.zigenpoisson.rvs(mu_true, expected_params[-1], 2, 0.5, size=mu_true.shape)
        model = sm.ZeroInflatedGeneralizedPoisson(cls.endog, exog, p=2)
        cls.res = model.fit(method='bfgs', maxiter=5000, disp=False)
        cls.params_true = [mu_true, expected_params[-1], 2, 0.5, nobs]

    def test_mean(self):
        if False:
            while True:
                i = 10

        def compute_conf_interval_95(mu, alpha, p, prob_infl, nobs):
            if False:
                print('Hello World!')
            p = p - 1
            dispersion_factor = (1 + alpha * mu ** p) ** 2 + prob_infl * mu
            var = (dispersion_factor * (1 - prob_infl) * mu).mean()
            var += (((1 - prob_infl) * mu) ** 2).mean()
            var -= ((1 - prob_infl) * mu).mean() ** 2
            std = np.sqrt(var)
            conf_intv_95 = 2 * std / np.sqrt(nobs)
            return conf_intv_95
        conf_interval_95 = compute_conf_interval_95(*self.params_true)
        assert_allclose(self.res.predict().mean(), self.endog.mean(), atol=conf_interval_95, rtol=0)

    def test_var(self):
        if False:
            i = 10
            return i + 15

        def compute_mixture_var(dispersion_factor, prob_main, mu):
            if False:
                print('Hello World!')
            prob_infl = 1 - prob_main
            var = (dispersion_factor * (1 - prob_infl) * mu).mean()
            var += (((1 - prob_infl) * mu) ** 2).mean()
            var -= ((1 - prob_infl) * mu).mean() ** 2
            return var
        res = self.res
        var_fitted = compute_mixture_var(res._dispersion_factor, res.predict(which='prob-main'), res.predict(which='mean-main'))
        assert_allclose(var_fitted.mean(), self.endog.var(), atol=0.05, rtol=0.1)

    def test_predict_prob(self):
        if False:
            for i in range(10):
                print('nop')
        res = self.res
        pr = res.predict(which='prob')
        pr2 = sm.distributions.zinegbin.pmf(np.arange(pr.shape[1])[:, None], res.predict(), 0.5, 2, 0.5).T
        assert_allclose(pr, pr2, rtol=0.08, atol=0.05)

class TestZeroInflatedNegativeBinomialP(CheckGeneric):

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        data = sm.datasets.randhie.load()
        cls.endog = np.asarray(data.endog)
        data.exog = np.asarray(data.exog)
        exog = sm.add_constant(data.exog[:, 1], prepend=False)
        exog_infl = sm.add_constant(data.exog[:, 0], prepend=False)
        sp = np.array([1.88, -10.28, -0.2, 1.14, 1.34])
        cls.res1 = sm.ZeroInflatedNegativeBinomialP(data.endog, exog, exog_infl=exog_infl, p=2).fit(start_params=sp, method='nm', xtol=1e-06, maxiter=5000, disp=False)
        cls.res1._results._attach_nullmodel = True
        cls.init_keys = ['exog_infl', 'exposure', 'inflation', 'offset', 'p']
        cls.init_kwds = {'inflation': 'logit', 'p': 2}
        res2 = RandHIE.zero_inflated_negative_binomial
        cls.res2 = res2

    def test_params(self):
        if False:
            i = 10
            return i + 15
        assert_allclose(self.res1.params, self.res2.params, atol=0.001, rtol=0.001)

    def test_conf_int(self):
        if False:
            print('Hello World!')
        pass

    def test_bic(self):
        if False:
            while True:
                i = 10
        pass

    def test_fit_regularized(self):
        if False:
            for i in range(10):
                print('nop')
        model = self.res1.model
        alpha = np.ones(len(self.res1.params))
        alpha[-2:] = 0
        res_reg = model.fit_regularized(alpha=alpha * 0.01, disp=False, maxiter=500)
        assert_allclose(res_reg.params[2:], self.res1.params[2:], atol=0.1, rtol=0.1)

    def test_minimize(self, reset_randomstate):
        if False:
            print('Hello World!')
        model = self.res1.model
        start_params = self.res1.mle_settings['start_params']
        res_ncg = model.fit(start_params=start_params, method='minimize', min_method='trust-ncg', maxiter=500, disp=False)
        assert_allclose(res_ncg.params, self.res2.params, atol=0.001, rtol=0.03)
        assert_allclose(res_ncg.bse, self.res2.bse, atol=0.001, rtol=0.06)
        assert_(res_ncg.mle_retvals['converged'] is True)
        res_dog = model.fit(start_params=start_params, method='minimize', min_method='dogleg', maxiter=500, disp=False)
        assert_allclose(res_dog.params, self.res2.params, atol=0.001, rtol=0.003)
        assert_allclose(res_dog.bse, self.res2.bse, atol=0.001, rtol=0.007)
        assert_(res_dog.mle_retvals['converged'] is True)
        res_bh = model.fit(start_params=start_params, method='basinhopping', maxiter=500, niter_success=3, disp=False)
        assert_allclose(res_bh.params, self.res2.params, atol=0.0001, rtol=0.0003)
        assert_allclose(res_bh.bse, self.res2.bse, atol=0.001, rtol=0.001)

class TestZeroInflatedNegativeBinomialP_predict:

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        expected_params = [1, 1, 0.5]
        np.random.seed(999)
        nobs = 5000
        exog = np.ones((nobs, 2))
        exog[:nobs // 2, 1] = 0
        prob_infl = 0.15
        mu_true = np.exp(exog.dot(expected_params[:-1]))
        cls.endog = sm.distributions.zinegbin.rvs(mu_true, expected_params[-1], 2, prob_infl, size=mu_true.shape)
        model = sm.ZeroInflatedNegativeBinomialP(cls.endog, exog, p=2)
        cls.res = model.fit(method='bfgs', maxiter=5000, disp=False)
        cls.prob_infl = prob_infl
        cls.params_true = [mu_true, expected_params[-1], 2, prob_infl, nobs]

    def test_mean(self):
        if False:
            while True:
                i = 10

        def compute_conf_interval_95(mu, alpha, p, prob_infl, nobs):
            if False:
                return 10
            dispersion_factor = 1 + alpha * mu ** (p - 1) + prob_infl * mu
            var = (dispersion_factor * (1 - prob_infl) * mu).mean()
            var += (((1 - prob_infl) * mu) ** 2).mean()
            var -= ((1 - prob_infl) * mu).mean() ** 2
            std = np.sqrt(var)
            conf_intv_95 = 2 * std / np.sqrt(nobs)
            return conf_intv_95
        conf_interval_95 = compute_conf_interval_95(*self.params_true)
        mean_true = ((1 - self.prob_infl) * self.params_true[0]).mean()
        assert_allclose(self.res.predict().mean(), mean_true, atol=conf_interval_95, rtol=0)

    def test_var(self):
        if False:
            print('Hello World!')

        def compute_mixture_var(dispersion_factor, prob_main, mu):
            if False:
                for i in range(10):
                    print('nop')
            prob_infl = 1 - prob_main
            var = (dispersion_factor * (1 - prob_infl) * mu).mean()
            var += (((1 - prob_infl) * mu) ** 2).mean()
            var -= ((1 - prob_infl) * mu).mean() ** 2
            return var
        res = self.res
        var_fitted = compute_mixture_var(res._dispersion_factor, res.predict(which='prob-main'), res.predict(which='mean-main'))
        assert_allclose(var_fitted.mean(), self.endog.var(), rtol=0.2)

    def test_predict_prob(self):
        if False:
            print('Hello World!')
        res = self.res
        endog = res.model.endog
        pr = res.predict(which='prob')
        pr2 = sm.distributions.zinegbin.pmf(np.arange(pr.shape[1])[:, None], res.predict(), 0.5, 2, self.prob_infl).T
        assert_allclose(pr, pr2, rtol=0.1, atol=0.1)
        prm = pr.mean(0)
        pr2m = pr2.mean(0)
        freq = np.bincount(endog.astype(int)) / len(endog)
        assert_allclose(((pr2m - prm) ** 2).mean(), 0, rtol=1e-10, atol=0.0005)
        assert_allclose(((prm - freq) ** 2).mean(), 0, rtol=1e-10, atol=0.0001)

    def test_predict_generic_zi(self):
        if False:
            i = 10
            return i + 15
        res = self.res
        endog = self.endog
        exog = self.res.model.exog
        prob_infl = self.prob_infl
        nobs = len(endog)
        freq = np.bincount(endog.astype(int)) / len(endog)
        probs = res.predict(which='prob')
        probsm = probs.mean(0)
        assert_allclose(freq, probsm, atol=0.02)
        probs_unique = res.predict(exog=[[1, 0], [1, 1]], exog_infl=np.asarray([[1], [1]]), which='prob')
        probs_unique2 = probs[[1, nobs - 1]]
        assert_allclose(probs_unique, probs_unique2, atol=1e-10)
        probs0_unique = res.predict(exog=[[1, 0], [1, 1]], exog_infl=np.asarray([[1], [1]]), which='prob-zero')
        assert_allclose(probs0_unique, probs_unique2[:, 0], rtol=1e-10)
        probs_main_unique = res.predict(exog=[[1, 0], [1, 1]], exog_infl=np.asarray([[1], [1]]), which='prob-main')
        probs_main = res.predict(which='prob-main')
        probs_main[[0, -1]]
        assert_allclose(probs_main_unique, probs_main[[0, -1]], rtol=1e-10)
        assert_allclose(probs_main_unique, 1 - prob_infl, atol=0.01)
        pred = res.predict(exog=[[1, 0], [1, 1]], exog_infl=np.asarray([[1], [1]]))
        pred1 = (endog[exog[:, 1] == 0].mean(), endog[exog[:, 1] == 1].mean())
        assert_allclose(pred, pred1, rtol=0.05)
        pred_main_unique = res.predict(exog=[[1, 0], [1, 1]], exog_infl=np.asarray([[1], [1]]), which='mean-main')
        assert_allclose(pred_main_unique, np.exp(np.cumsum(res.params[1:3])), rtol=1e-10)
        mean_nz = (endog[(exog[:, 1] == 0) & (endog > 0)].mean(), endog[(exog[:, 1] == 1) & (endog > 0)].mean())
        pred_nonzero_unique = res.predict(exog=[[1, 0], [1, 1]], exog_infl=np.asarray([[1], [1]]), which='mean-nonzero')
        assert_allclose(pred_nonzero_unique, mean_nz, rtol=0.05)
        pred_lin_unique = res.predict(exog=[[1, 0], [1, 1]], exog_infl=np.asarray([[1], [1]]), which='linear')
        assert_allclose(pred_lin_unique, np.cumsum(res.params[1:3]), rtol=1e-10)

class TestZeroInflatedNegativeBinomialP_predict2:

    @classmethod
    def setup_class(cls):
        if False:
            i = 10
            return i + 15
        data = sm.datasets.randhie.load()
        cls.endog = np.asarray(data.endog)
        data.exog = np.asarray(data.exog)
        exog = data.exog
        start_params = np.array([-2.83983767, -2.31595924, -3.9263248, -4.01816431, -5.52251843, -2.4351714, -4.61636366, -4.17959785, -0.12960256, -0.05653484, -0.21206673, 0.08782572, -0.02991995, 0.22901208, 0.0620983, 0.06809681, 0.0841814, 0.185506, 1.36527888])
        mod = sm.ZeroInflatedNegativeBinomialP(cls.endog, exog, exog_infl=exog, p=2)
        res = mod.fit(start_params=start_params, method='bfgs', maxiter=1000, disp=False)
        cls.res = res

    def test_mean(self):
        if False:
            i = 10
            return i + 15
        assert_allclose(self.res.predict().mean(), self.endog.mean(), atol=0.02)

    def test_zero_nonzero_mean(self):
        if False:
            for i in range(10):
                print('nop')
        mean1 = self.endog.mean()
        mean2 = (1 - self.res.predict(which='prob-zero').mean()) * self.res.predict(which='mean-nonzero').mean()
        assert_allclose(mean1, mean2, atol=0.2)

class TestPandasOffset:

    def test_pd_offset_exposure(self):
        if False:
            i = 10
            return i + 15
        endog = pd.DataFrame({'F': [0.0, 0.0, 0.0, 0.0, 1.0]})
        exog = pd.DataFrame({'I': [1.0, 1.0, 1.0, 1.0, 1.0], 'C': [0.0, 1.0, 0.0, 1.0, 0.0]})
        exposure = pd.Series([1.0, 1, 1, 2, 1])
        offset = pd.Series([1, 1, 1, 2, 1])
        sm.Poisson(endog=endog, exog=exog, offset=offset).fit()
        inflations = ['logit', 'probit']
        for inflation in inflations:
            sm.ZeroInflatedPoisson(endog=endog, exog=exog['I'], exposure=exposure, inflation=inflation).fit()