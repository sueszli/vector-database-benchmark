u"""
Beta regression for modeling rates and proportions.

References
----------
Grün, Bettina, Ioannis Kosmidis, and Achim Zeileis. Extended beta regression
in R: Shaken, stirred, mixed, and partitioned. No. 2011-22. Working Papers in
Economics and Statistics, 2011.

Smithson, Michael, and Jay Verkuilen. "A better lemon squeezer?
Maximum-likelihood regression with beta-distributed dependent variables."
Psychological methods 11.1 (2006): 54.
"""
import numpy as np
from scipy.special import gammaln as lgamma
import patsy
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import GenericLikelihoodModel, GenericLikelihoodModelResults, _LLRMixin
from statsmodels.genmod import families
_init_example = "\n\n    Beta regression with default of logit-link for exog and log-link\n    for precision.\n\n    >>> mod = BetaModel(endog, exog)\n    >>> rslt = mod.fit()\n    >>> print(rslt.summary())\n\n    We can also specify a formula and a specific structure and use the\n    identity-link for precision.\n\n    >>> from sm.families.links import identity\n    >>> Z = patsy.dmatrix('~ temp', dat, return_type='dataframe')\n    >>> mod = BetaModel.from_formula('iyield ~ C(batch, Treatment(10)) + temp',\n    ...                              dat, exog_precision=Z,\n    ...                              link_precision=identity())\n\n    In the case of proportion-data, we may think that the precision depends on\n    the number of measurements. E.g for sequence data, on the number of\n    sequence reads covering a site:\n\n    >>> Z = patsy.dmatrix('~ coverage', df)\n    >>> formula = 'methylation ~ disease + age + gender + coverage'\n    >>> mod = BetaModel.from_formula(formula, df, Z)\n    >>> rslt = mod.fit()\n\n"

class BetaModel(GenericLikelihoodModel):
    __doc__ = 'Beta Regression.\n\n    The Model is parameterized by mean and precision. Both can depend on\n    explanatory variables through link functions.\n\n    Parameters\n    ----------\n    endog : array_like\n        1d array of endogenous response variable.\n    exog : array_like\n        A nobs x k array where `nobs` is the number of observations and `k`\n        is the number of regressors. An intercept is not included by default\n        and should be added by the user (models specified using a formula\n        include an intercept by default). See `statsmodels.tools.add_constant`.\n    exog_precision : array_like\n        2d array of variables for the precision.\n    link : link\n        Any link in sm.families.links for mean, should have range in\n        interval [0, 1]. Default is logit-link.\n    link_precision : link\n        Any link in sm.families.links for precision, should have\n        range in positive line. Default is log-link.\n    **kwds : extra keywords\n        Keyword options that will be handled by super classes.\n        Not all general keywords will be supported in this class.\n\n    Notes\n    -----\n    Status: experimental, new in 0.13.\n    Core results are verified, but api can change and some extra results\n    specific to Beta regression are missing.\n\n    Examples\n    --------\n    {example}\n\n    See Also\n    --------\n    :ref:`links`\n\n    '.format(example=_init_example)

    def __init__(self, endog, exog, exog_precision=None, link=families.links.Logit(), link_precision=families.links.Log(), **kwds):
        if False:
            while True:
                i = 10
        etmp = np.array(endog)
        assert np.all((0 < etmp) & (etmp < 1))
        if exog_precision is None:
            extra_names = ['precision']
            exog_precision = np.ones((len(endog), 1), dtype='f')
        else:
            extra_names = ['precision-%s' % zc for zc in (exog_precision.columns if hasattr(exog_precision, 'columns') else range(1, exog_precision.shape[1] + 1))]
        kwds['extra_params_names'] = extra_names
        super(BetaModel, self).__init__(endog, exog, exog_precision=exog_precision, **kwds)
        self.link = link
        self.link_precision = link_precision
        self.nobs = self.endog.shape[0]
        self.k_extra = 1
        self.df_model = self.nparams - 2
        self.df_resid = self.nobs - self.nparams
        assert len(self.exog_precision) == len(self.endog)
        self.hess_type = 'oim'
        if 'exog_precision' not in self._init_keys:
            self._init_keys.extend(['exog_precision'])
        self._init_keys.extend(['link', 'link_precision'])
        self._null_drop_keys = ['exog_precision']
        del kwds['extra_params_names']
        self._check_kwargs(kwds)
        self.results_class = BetaResults
        self.results_class_wrapper = BetaResultsWrapper

    @classmethod
    def from_formula(cls, formula, data, exog_precision_formula=None, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if exog_precision_formula is not None:
            if 'subset' in kwargs:
                d = data.ix[kwargs['subset']]
                Z = patsy.dmatrix(exog_precision_formula, d)
            else:
                Z = patsy.dmatrix(exog_precision_formula, data)
            kwargs['exog_precision'] = Z
        return super(BetaModel, cls).from_formula(formula, data, *args, **kwargs)

    def _get_exogs(self):
        if False:
            while True:
                i = 10
        return (self.exog, self.exog_precision)

    def predict(self, params, exog=None, exog_precision=None, which='mean'):
        if False:
            i = 10
            return i + 15
        'Predict values for mean or precision\n\n        Parameters\n        ----------\n        params : array_like\n            The model parameters.\n        exog : array_like\n            Array of predictor variables for mean.\n        exog_precision : array_like\n            Array of predictor variables for precision parameter.\n        which : str\n\n            - "mean" : mean, conditional expectation E(endog | exog)\n            - "precision" : predicted precision\n            - "linear" : linear predictor for the mean function\n            - "linear-precision" : linear predictor for the precision parameter\n\n        Returns\n        -------\n        ndarray, predicted values\n        '
        if which == 'linpred':
            which = 'linear'
        if which in ['linpred_precision', 'linear_precision']:
            which = 'linear-precision'
        k_mean = self.exog.shape[1]
        if which in ['mean', 'linear']:
            if exog is None:
                exog = self.exog
            params_mean = params[:k_mean]
            linpred = np.dot(exog, params_mean)
            if which == 'mean':
                mu = self.link.inverse(linpred)
                res = mu
            else:
                res = linpred
        elif which in ['precision', 'linear-precision']:
            if exog_precision is None:
                exog_precision = self.exog_precision
            params_prec = params[k_mean:]
            linpred_prec = np.dot(exog_precision, params_prec)
            if which == 'precision':
                phi = self.link_precision.inverse(linpred_prec)
                res = phi
            else:
                res = linpred_prec
        elif which == 'var':
            res = self._predict_var(params, exog=exog, exog_precision=exog_precision)
        else:
            raise ValueError('which = %s is not available' % which)
        return res

    def _predict_precision(self, params, exog_precision=None):
        if False:
            return 10
        'Predict values for precision function for given exog_precision.\n\n        Parameters\n        ----------\n        params : array_like\n            The model parameters.\n        exog_precision : array_like\n            Array of predictor variables for precision.\n\n        Returns\n        -------\n        Predicted precision.\n        '
        if exog_precision is None:
            exog_precision = self.exog_precision
        k_mean = self.exog.shape[1]
        params_precision = params[k_mean:]
        linpred_prec = np.dot(exog_precision, params_precision)
        phi = self.link_precision.inverse(linpred_prec)
        return phi

    def _predict_var(self, params, exog=None, exog_precision=None):
        if False:
            i = 10
            return i + 15
        'predict values for conditional variance V(endog | exog)\n\n        Parameters\n        ----------\n        params : array_like\n            The model parameters.\n        exog : array_like\n            Array of predictor variables for mean.\n        exog_precision : array_like\n            Array of predictor variables for precision.\n\n        Returns\n        -------\n        Predicted conditional variance.\n        '
        mean = self.predict(params, exog=exog)
        precision = self._predict_precision(params, exog_precision=exog_precision)
        var_endog = mean * (1 - mean) / (1 + precision)
        return var_endog

    def loglikeobs(self, params):
        if False:
            return 10
        '\n        Loglikelihood for observations of the Beta regressionmodel.\n\n        Parameters\n        ----------\n        params : ndarray\n            The parameters of the model, coefficients for linear predictors\n            of the mean and of the precision function.\n\n        Returns\n        -------\n        loglike : ndarray\n            The log likelihood for each observation of the model evaluated\n            at `params`.\n        '
        return self._llobs(self.endog, self.exog, self.exog_precision, params)

    def _llobs(self, endog, exog, exog_precision, params):
        if False:
            return 10
        '\n        Loglikelihood for observations with data arguments.\n\n        Parameters\n        ----------\n        endog : ndarray\n            1d array of endogenous variable.\n        exog : ndarray\n            2d array of explanatory variables.\n        exog_precision : ndarray\n            2d array of explanatory variables for precision.\n        params : ndarray\n            The parameters of the model, coefficients for linear predictors\n            of the mean and of the precision function.\n\n        Returns\n        -------\n        loglike : ndarray\n            The log likelihood for each observation of the model evaluated\n            at `params`.\n        '
        (y, X, Z) = (endog, exog, exog_precision)
        nz = Z.shape[1]
        params_mean = params[:-nz]
        params_prec = params[-nz:]
        linpred = np.dot(X, params_mean)
        linpred_prec = np.dot(Z, params_prec)
        mu = self.link.inverse(linpred)
        phi = self.link_precision.inverse(linpred_prec)
        eps_lb = 1e-200
        alpha = np.clip(mu * phi, eps_lb, np.inf)
        beta = np.clip((1 - mu) * phi, eps_lb, np.inf)
        ll = lgamma(phi) - lgamma(alpha) - lgamma(beta) + (mu * phi - 1) * np.log(y) + ((1 - mu) * phi - 1) * np.log(1 - y)
        return ll

    def score(self, params):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the score vector of the log-likelihood.\n\n        http://www.tandfonline.com/doi/pdf/10.1080/00949650903389993\n\n        Parameters\n        ----------\n        params : ndarray\n            Parameter at which score is evaluated.\n\n        Returns\n        -------\n        score : ndarray\n            First derivative of loglikelihood function.\n        '
        (sf1, sf2) = self.score_factor(params)
        d1 = np.dot(sf1, self.exog)
        d2 = np.dot(sf2, self.exog_precision)
        return np.concatenate((d1, d2))

    def _score_check(self, params):
        if False:
            for i in range(10):
                print('nop')
        'Inherited score with finite differences\n\n        Parameters\n        ----------\n        params : ndarray\n            Parameter at which score is evaluated.\n\n        Returns\n        -------\n        score based on numerical derivatives\n        '
        return super(BetaModel, self).score(params)

    def score_factor(self, params, endog=None):
        if False:
            print('Hello World!')
        'Derivative of loglikelihood function w.r.t. linear predictors.\n\n        This needs to be multiplied with the exog to obtain the score_obs.\n\n        Parameters\n        ----------\n        params : ndarray\n            Parameter at which score is evaluated.\n\n        Returns\n        -------\n        score_factor : ndarray, 2-D\n            A 2d weight vector used in the calculation of the score_obs.\n\n        Notes\n        -----\n        The score_obs can be obtained from score_factor ``sf`` using\n\n            - d1 = sf[:, :1] * exog\n            - d2 = sf[:, 1:2] * exog_precision\n\n        '
        from scipy import special
        digamma = special.psi
        y = self.endog if endog is None else endog
        (X, Z) = (self.exog, self.exog_precision)
        nz = Z.shape[1]
        Xparams = params[:-nz]
        Zparams = params[-nz:]
        mu = self.link.inverse(np.dot(X, Xparams))
        phi = self.link_precision.inverse(np.dot(Z, Zparams))
        eps_lb = 1e-200
        alpha = np.clip(mu * phi, eps_lb, np.inf)
        beta = np.clip((1 - mu) * phi, eps_lb, np.inf)
        ystar = np.log(y / (1.0 - y))
        dig_beta = digamma(beta)
        mustar = digamma(alpha) - dig_beta
        yt = np.log(1 - y)
        mut = dig_beta - digamma(phi)
        t = 1.0 / self.link.deriv(mu)
        h = 1.0 / self.link_precision.deriv(phi)
        sf1 = phi * t * (ystar - mustar)
        sf2 = h * (mu * (ystar - mustar) + yt - mut)
        return (sf1, sf2)

    def score_hessian_factor(self, params, return_hessian=False, observed=True):
        if False:
            print('Hello World!')
        'Derivatives of loglikelihood function w.r.t. linear predictors.\n\n        This calculates score and hessian factors at the same time, because\n        there is a large overlap in calculations.\n\n        Parameters\n        ----------\n        params : ndarray\n            Parameter at which score is evaluated.\n        return_hessian : bool\n            If False, then only score_factors are returned\n            If True, the both score and hessian factors are returned\n        observed : bool\n            If True, then the observed Hessian is returned (default).\n            If False, then the expected information matrix is returned.\n\n        Returns\n        -------\n        score_factor : ndarray, 2-D\n            A 2d weight vector used in the calculation of the score_obs.\n        (-jbb, -jbg, -jgg) : tuple\n            A tuple with 3 hessian factors, corresponding to the upper\n            triangle of the Hessian matrix.\n            TODO: check why there are minus\n        '
        from scipy import special
        digamma = special.psi
        (y, X, Z) = (self.endog, self.exog, self.exog_precision)
        nz = Z.shape[1]
        Xparams = params[:-nz]
        Zparams = params[-nz:]
        mu = self.link.inverse(np.dot(X, Xparams))
        phi = self.link_precision.inverse(np.dot(Z, Zparams))
        eps_lb = 1e-200
        alpha = np.clip(mu * phi, eps_lb, np.inf)
        beta = np.clip((1 - mu) * phi, eps_lb, np.inf)
        ystar = np.log(y / (1.0 - y))
        dig_beta = digamma(beta)
        mustar = digamma(alpha) - dig_beta
        yt = np.log(1 - y)
        mut = dig_beta - digamma(phi)
        t = 1.0 / self.link.deriv(mu)
        h = 1.0 / self.link_precision.deriv(phi)
        ymu_star = ystar - mustar
        sf1 = phi * t * ymu_star
        sf2 = h * (mu * ymu_star + yt - mut)
        if return_hessian:
            trigamma = lambda x: special.polygamma(1, x)
            trig_beta = trigamma(beta)
            var_star = trigamma(alpha) + trig_beta
            var_t = trig_beta - trigamma(phi)
            c = -trig_beta
            s = self.link.deriv2(mu)
            q = self.link_precision.deriv2(phi)
            jbb = phi * t * var_star
            if observed:
                jbb += s * t ** 2 * ymu_star
            jbb *= t * phi
            jbg = phi * t * h * (mu * var_star + c)
            if observed:
                jbg -= ymu_star * t * h
            jgg = h ** 2 * (mu ** 2 * var_star + 2 * mu * c + var_t)
            if observed:
                jgg += (mu * ymu_star + yt - mut) * q * h ** 3
            return ((sf1, sf2), (-jbb, -jbg, -jgg))
        else:
            return (sf1, sf2)

    def score_obs(self, params):
        if False:
            while True:
                i = 10
        '\n        Score, first derivative of the loglikelihood for each observation.\n\n        Parameters\n        ----------\n        params : ndarray\n            Parameter at which score is evaluated.\n\n        Returns\n        -------\n        score_obs : ndarray, 2d\n            The first derivative of the loglikelihood function evaluated at\n            params for each observation.\n        '
        (sf1, sf2) = self.score_factor(params)
        d1 = sf1[:, None] * self.exog
        d2 = sf2[:, None] * self.exog_precision
        return np.column_stack((d1, d2))

    def hessian(self, params, observed=None):
        if False:
            while True:
                i = 10
        'Hessian, second derivative of loglikelihood function\n\n        Parameters\n        ----------\n        params : ndarray\n            Parameter at which Hessian is evaluated.\n        observed : bool\n            If True, then the observed Hessian is returned (default).\n            If False, then the expected information matrix is returned.\n\n        Returns\n        -------\n        hessian : ndarray\n            Hessian, i.e. observed information, or expected information matrix.\n        '
        if self.hess_type == 'eim':
            observed = False
        else:
            observed = True
        (_, hf) = self.score_hessian_factor(params, return_hessian=True, observed=observed)
        (hf11, hf12, hf22) = hf
        d11 = (self.exog.T * hf11).dot(self.exog)
        d12 = (self.exog.T * hf12).dot(self.exog_precision)
        d22 = (self.exog_precision.T * hf22).dot(self.exog_precision)
        return np.block([[d11, d12], [d12.T, d22]])

    def hessian_factor(self, params, observed=True):
        if False:
            return 10
        'Derivatives of loglikelihood function w.r.t. linear predictors.\n        '
        (_, hf) = self.score_hessian_factor(params, return_hessian=True, observed=observed)
        return hf

    def _start_params(self, niter=2, return_intermediate=False):
        if False:
            print('Hello World!')
        'find starting values\n\n        Parameters\n        ----------\n        niter : int\n            Number of iterations of WLS approximation\n        return_intermediate : bool\n            If False (default), then only the preliminary parameter estimate\n            will be returned.\n            If True, then also the two results instances of the WLS estimate\n            for mean parameters and for the precision parameters will be\n            returned.\n\n        Returns\n        -------\n        sp : ndarray\n            start parameters for the optimization\n        res_m2 : results instance (optional)\n            Results instance for the WLS regression of the mean function.\n        res_p2 : results instance (optional)\n            Results instance for the WLS regression of the precision function.\n\n        Notes\n        -----\n        This calculates a few iteration of weighted least squares. This is not\n        a full scoring algorithm.\n        '
        from statsmodels.regression.linear_model import OLS, WLS
        res_m = OLS(self.link(self.endog), self.exog).fit()
        fitted = self.link.inverse(res_m.fittedvalues)
        resid = self.endog - fitted
        prec_i = fitted * (1 - fitted) / np.maximum(np.abs(resid), 0.01) ** 2 - 1
        res_p = OLS(self.link_precision(prec_i), self.exog_precision).fit()
        prec_fitted = self.link_precision.inverse(res_p.fittedvalues)
        for _ in range(niter):
            y_var_inv = (1 + prec_fitted) / (fitted * (1 - fitted))
            ylink_var_inv = y_var_inv / self.link.deriv(fitted) ** 2
            res_m2 = WLS(self.link(self.endog), self.exog, weights=ylink_var_inv).fit()
            fitted = self.link.inverse(res_m2.fittedvalues)
            resid2 = self.endog - fitted
            prec_i2 = fitted * (1 - fitted) / np.maximum(np.abs(resid2), 0.01) ** 2 - 1
            w_p = 1.0 / self.link_precision.deriv(prec_fitted) ** 2
            res_p2 = WLS(self.link_precision(prec_i2), self.exog_precision, weights=w_p).fit()
            prec_fitted = self.link_precision.inverse(res_p2.fittedvalues)
            sp2 = np.concatenate((res_m2.params, res_p2.params))
        if return_intermediate:
            return (sp2, res_m2, res_p2)
        return sp2

    def fit(self, start_params=None, maxiter=1000, disp=False, method='bfgs', **kwds):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fit the model by maximum likelihood.\n\n        Parameters\n        ----------\n        start_params : array-like\n            A vector of starting values for the regression\n            coefficients.  If None, a default is chosen.\n        maxiter : integer\n            The maximum number of iterations\n        disp : bool\n            Show convergence stats.\n        method : str\n            The optimization method to use.\n        kwds :\n            Keyword arguments for the optimizer.\n\n        Returns\n        -------\n        BetaResults instance.\n        '
        if start_params is None:
            start_params = self._start_params()
        if 'cov_type' in kwds:
            if kwds['cov_type'].lower() == 'eim':
                self.hess_type = 'eim'
                del kwds['cov_type']
        else:
            self.hess_type = 'oim'
        res = super(BetaModel, self).fit(start_params=start_params, maxiter=maxiter, method=method, disp=disp, **kwds)
        if not isinstance(res, BetaResultsWrapper):
            res = BetaResultsWrapper(res)
        return res

    def _deriv_mean_dparams(self, params):
        if False:
            return 10
        '\n        Derivative of the expected endog with respect to the parameters.\n\n        not verified yet\n\n        Parameters\n        ----------\n        params : ndarray\n            parameter at which score is evaluated\n\n        Returns\n        -------\n        The value of the derivative of the expected endog with respect\n        to the parameter vector.\n        '
        link = self.link
        lin_pred = self.predict(params, which='linear')
        idl = link.inverse_deriv(lin_pred)
        dmat = self.exog * idl[:, None]
        return np.column_stack((dmat, np.zeros(self.exog_precision.shape)))

    def _deriv_score_obs_dendog(self, params):
        if False:
            print('Hello World!')
        'derivative of score_obs w.r.t. endog\n\n        Parameters\n        ----------\n        params : ndarray\n            parameter at which score is evaluated\n\n        Returns\n        -------\n        derivative : ndarray_2d\n            The derivative of the score_obs with respect to endog.\n        '
        from statsmodels.tools.numdiff import _approx_fprime_cs_scalar

        def f(y):
            if False:
                return 10
            if y.ndim == 2 and y.shape[1] == 1:
                y = y[:, 0]
            sf = self.score_factor(params, endog=y)
            return np.column_stack(sf)
        dsf = _approx_fprime_cs_scalar(self.endog[:, None], f)
        d1 = dsf[:, :1] * self.exog
        d2 = dsf[:, 1:2] * self.exog_precision
        return np.column_stack((d1, d2))

    def get_distribution_params(self, params, exog=None, exog_precision=None):
        if False:
            i = 10
            return i + 15
        '\n        Return distribution parameters converted from model prediction.\n\n        Parameters\n        ----------\n        params : array_like\n            The model parameters.\n        exog : array_like\n            Array of predictor variables for mean.\n        exog_precision : array_like\n            Array of predictor variables for mean.\n\n        Returns\n        -------\n        (alpha, beta) : tuple of ndarrays\n            Parameters for the scipy distribution to evaluate predictive\n            distribution.\n        '
        mean = self.predict(params, exog=exog)
        precision = self.predict(params, exog_precision=exog_precision, which='precision')
        return (precision * mean, precision * (1 - mean))

    def get_distribution(self, params, exog=None, exog_precision=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a instance of the predictive distribution.\n\n        Parameters\n        ----------\n        params : array_like\n            The model parameters.\n        exog : array_like\n            Array of predictor variables for mean.\n        exog_precision : array_like\n            Array of predictor variables for mean.\n\n        Returns\n        -------\n        Instance of a scipy frozen distribution based on estimated\n        parameters.\n\n        See Also\n        --------\n        predict\n\n        Notes\n        -----\n        This function delegates to the predict method to handle exog and\n        exog_precision, which in turn makes any required transformations.\n\n        Due to the behavior of ``scipy.stats.distributions objects``, the\n        returned random number generator must be called with ``gen.rvs(n)``\n        where ``n`` is the number of observations in the data set used\n        to fit the model.  If any other value is used for ``n``, misleading\n        results will be produced.\n        '
        from scipy import stats
        args = self.get_distribution_params(params, exog=exog, exog_precision=exog_precision)
        distr = stats.beta(*args)
        return distr

class BetaResults(GenericLikelihoodModelResults, _LLRMixin):
    """Results class for Beta regression

    This class inherits from GenericLikelihoodModelResults and not all
    inherited methods might be appropriate in this case.
    """

    @cache_readonly
    def fittedvalues(self):
        if False:
            for i in range(10):
                print('nop')
        'In-sample predicted mean, conditional expectation.'
        return self.model.predict(self.params)

    @cache_readonly
    def fitted_precision(self):
        if False:
            return 10
        'In-sample predicted precision'
        return self.model.predict(self.params, which='precision')

    @cache_readonly
    def resid(self):
        if False:
            while True:
                i = 10
        'Response residual'
        return self.model.endog - self.fittedvalues

    @cache_readonly
    def resid_pearson(self):
        if False:
            return 10
        'Pearson standardize residual'
        std = np.sqrt(self.model.predict(self.params, which='var'))
        return self.resid / std

    @cache_readonly
    def prsquared(self):
        if False:
            print('Hello World!')
        'Cox-Snell Likelihood-Ratio pseudo-R-squared.\n\n        1 - exp((llnull - .llf) * (2 / nobs))\n        '
        return self.pseudo_rsquared(kind='lr')

    def get_distribution_params(self, exog=None, exog_precision=None, transform=True):
        if False:
            print('Hello World!')
        '\n        Return distribution parameters converted from model prediction.\n\n        Parameters\n        ----------\n        params : array_like\n            The model parameters.\n        exog : array_like\n            Array of predictor variables for mean.\n        transform : bool\n            If transform is True and formulas have been used, then predictor\n            ``exog`` is passed through the formula processing. Default is True.\n\n        Returns\n        -------\n        (alpha, beta) : tuple of ndarrays\n            Parameters for the scipy distribution to evaluate predictive\n            distribution.\n        '
        mean = self.predict(exog=exog, transform=transform)
        precision = self.predict(exog_precision=exog_precision, which='precision', transform=transform)
        return (precision * mean, precision * (1 - mean))

    def get_distribution(self, exog=None, exog_precision=None, transform=True):
        if False:
            print('Hello World!')
        '\n        Return a instance of the predictive distribution.\n\n        Parameters\n        ----------\n        exog : array_like\n            Array of predictor variables for mean.\n        exog_precision : array_like\n            Array of predictor variables for mean.\n        transform : bool\n            If transform is True and formulas have been used, then predictor\n            ``exog`` is passed through the formula processing. Default is True.\n\n        Returns\n        -------\n        Instance of a scipy frozen distribution based on estimated\n        parameters.\n\n        See Also\n        --------\n        predict\n\n        Notes\n        -----\n        This function delegates to the predict method to handle exog and\n        exog_precision, which in turn makes any required transformations.\n\n        Due to the behavior of ``scipy.stats.distributions objects``, the\n        returned random number generator must be called with ``gen.rvs(n)``\n        where ``n`` is the number of observations in the data set used\n        to fit the model.  If any other value is used for ``n``, misleading\n        results will be produced.\n        '
        from scipy import stats
        args = self.get_distribution_params(exog=exog, exog_precision=exog_precision, transform=transform)
        args = (np.asarray(arg) for arg in args)
        distr = stats.beta(*args)
        return distr

    def get_influence(self):
        if False:
            print('Hello World!')
        '\n        Get an instance of MLEInfluence with influence and outlier measures\n\n        Returns\n        -------\n        infl : MLEInfluence instance\n            The instance has methods to calculate the main influence and\n            outlier measures as attributes.\n\n        See Also\n        --------\n        statsmodels.stats.outliers_influence.MLEInfluence\n\n        Notes\n        -----\n        Support for mutli-link and multi-exog models is still experimental\n        in MLEInfluence. Interface and some definitions might still change.\n\n        Note: Difference to R betareg: Betareg has the same general leverage\n        as this model. However, they use a linear approximation hat matrix\n        to scale and studentize influence and residual statistics.\n        MLEInfluence uses the generalized leverage as hat_matrix_diag.\n        Additionally, MLEInfluence uses pearson residuals for residual\n        analusis.\n\n        References\n        ----------\n        todo\n\n        '
        from statsmodels.stats.outliers_influence import MLEInfluence
        return MLEInfluence(self)

    def bootstrap(self, *args, **kwargs):
        if False:
            return 10
        raise NotImplementedError

class BetaResultsWrapper(lm.RegressionResultsWrapper):
    pass
wrap.populate_wrapper(BetaResultsWrapper, BetaResults)