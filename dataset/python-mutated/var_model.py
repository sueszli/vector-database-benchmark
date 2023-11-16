"""
Vector Autoregression (VAR) processes

References
----------
Lütkepohl (2005) New Introduction to Multiple Time Series Analysis
"""
from __future__ import annotations
from statsmodels.compat.python import lrange
from collections import defaultdict
from io import StringIO
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly, deprecated_alias
from statsmodels.tools.linalg import logdet_symm
from statsmodels.tools.sm_exceptions import OutputWarning
from statsmodels.tools.validation import array_like
from statsmodels.tsa.base.tsa_model import TimeSeriesModel, TimeSeriesResultsWrapper
import statsmodels.tsa.tsatools as tsa
from statsmodels.tsa.tsatools import duplication_matrix, unvec, vec
from statsmodels.tsa.vector_ar import output, plotting, util
from statsmodels.tsa.vector_ar.hypothesis_test_results import CausalityTestResults, NormalityTestResults, WhitenessTestResults
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.vector_ar.output import VARSummary

def ma_rep(coefs, maxn=10):
    if False:
        return 10
    '\n    MA(\\infty) representation of VAR(p) process\n\n    Parameters\n    ----------\n    coefs : ndarray (p x k x k)\n    maxn : int\n        Number of MA matrices to compute\n\n    Notes\n    -----\n    VAR(p) process as\n\n    .. math:: y_t = A_1 y_{t-1} + \\ldots + A_p y_{t-p} + u_t\n\n    can be equivalently represented as\n\n    .. math:: y_t = \\mu + \\sum_{i=0}^\\infty \\Phi_i u_{t-i}\n\n    e.g. can recursively compute the \\Phi_i matrices with \\Phi_0 = I_k\n\n    Returns\n    -------\n    phis : ndarray (maxn + 1 x k x k)\n    '
    (p, k, k) = coefs.shape
    phis = np.zeros((maxn + 1, k, k))
    phis[0] = np.eye(k)
    for i in range(1, maxn + 1):
        for j in range(1, i + 1):
            if j > p:
                break
            phis[i] += np.dot(phis[i - j], coefs[j - 1])
    return phis

def is_stable(coefs, verbose=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Determine stability of VAR(p) system by examining the eigenvalues of the\n    VAR(1) representation\n\n    Parameters\n    ----------\n    coefs : ndarray (p x k x k)\n\n    Returns\n    -------\n    is_stable : bool\n    '
    A_var1 = util.comp_matrix(coefs)
    eigs = np.linalg.eigvals(A_var1)
    if verbose:
        print('Eigenvalues of VAR(1) rep')
        for val in np.abs(eigs):
            print(val)
    return (np.abs(eigs) <= 1).all()

def var_acf(coefs, sig_u, nlags=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute autocovariance function ACF_y(h) up to nlags of stable VAR(p)\n    process\n\n    Parameters\n    ----------\n    coefs : ndarray (p x k x k)\n        Coefficient matrices A_i\n    sig_u : ndarray (k x k)\n        Covariance of white noise process u_t\n    nlags : int, optional\n        Defaults to order p of system\n\n    Notes\n    -----\n    Ref: Lütkepohl p.28-29\n\n    Returns\n    -------\n    acf : ndarray, (p, k, k)\n    '
    (p, k, _) = coefs.shape
    if nlags is None:
        nlags = p
    result = np.zeros((nlags + 1, k, k))
    result[:p] = _var_acf(coefs, sig_u)
    for h in range(p, nlags + 1):
        for j in range(p):
            result[h] += np.dot(coefs[j], result[h - j - 1])
    return result

def _var_acf(coefs, sig_u):
    if False:
        print('Hello World!')
    '\n    Compute autocovariance function ACF_y(h) for h=1,...,p\n\n    Notes\n    -----\n    Lütkepohl (2005) p.29\n    '
    (p, k, k2) = coefs.shape
    assert k == k2
    A = util.comp_matrix(coefs)
    SigU = np.zeros((k * p, k * p))
    SigU[:k, :k] = sig_u
    vecACF = np.linalg.solve(np.eye((k * p) ** 2) - np.kron(A, A), vec(SigU))
    acf = unvec(vecACF)
    acf = [acf[:k, k * i:k * (i + 1)] for i in range(p)]
    acf = np.array(acf)
    return acf

def forecast_cov(ma_coefs, sigma_u, steps):
    if False:
        return 10
    '\n    Compute theoretical forecast error variance matrices\n\n    Parameters\n    ----------\n    steps : int\n        Number of steps ahead\n\n    Notes\n    -----\n    .. math:: \\mathrm{MSE}(h) = \\sum_{i=0}^{h-1} \\Phi \\Sigma_u \\Phi^T\n\n    Returns\n    -------\n    forc_covs : ndarray (steps x neqs x neqs)\n    '
    neqs = len(sigma_u)
    forc_covs = np.zeros((steps, neqs, neqs))
    prior = np.zeros((neqs, neqs))
    for h in range(steps):
        phi = ma_coefs[h]
        var = phi @ sigma_u @ phi.T
        forc_covs[h] = prior = prior + var
    return forc_covs
mse = forecast_cov

def forecast(y, coefs, trend_coefs, steps, exog=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Produce linear minimum MSE forecast\n\n    Parameters\n    ----------\n    y : ndarray (k_ar x neqs)\n    coefs : ndarray (k_ar x neqs x neqs)\n    trend_coefs : ndarray (1 x neqs) or (neqs)\n    steps : int\n    exog : ndarray (trend_coefs.shape[1] x neqs)\n\n    Returns\n    -------\n    forecasts : ndarray (steps x neqs)\n\n    Notes\n    -----\n    Lütkepohl p. 37\n    '
    p = len(coefs)
    k = len(coefs[0])
    if y.shape[0] < p:
        raise ValueError(f'y must by have at least order ({p}) observations. Got {y.shape[0]}.')
    forcs = np.zeros((steps, k))
    if exog is not None and trend_coefs is not None:
        forcs += np.dot(exog, trend_coefs)
    elif exog is None and trend_coefs is not None:
        forcs += trend_coefs
    for h in range(1, steps + 1):
        f = forcs[h - 1]
        for i in range(1, p + 1):
            if h - i <= 0:
                prior_y = y[h - i - 1]
            else:
                prior_y = forcs[h - i - 1]
            f = f + np.dot(coefs[i - 1], prior_y)
        forcs[h - 1] = f
    return forcs

def _forecast_vars(steps, ma_coefs, sig_u):
    if False:
        print('Hello World!')
    '_forecast_vars function used by VECMResults. Note that the definition\n    of the local variable covs is the same as in VARProcess and as such it\n    differs from the one in VARResults!\n\n    Parameters\n    ----------\n    steps\n    ma_coefs\n    sig_u\n\n    Returns\n    -------\n    '
    covs = mse(ma_coefs, sig_u, steps)
    neqs = len(sig_u)
    inds = np.arange(neqs)
    return covs[:, inds, inds]

def forecast_interval(y, coefs, trend_coefs, sig_u, steps=5, alpha=0.05, exog=1):
    if False:
        return 10
    assert 0 < alpha < 1
    q = util.norm_signif_level(alpha)
    point_forecast = forecast(y, coefs, trend_coefs, steps, exog)
    ma_coefs = ma_rep(coefs, steps)
    sigma = np.sqrt(_forecast_vars(steps, ma_coefs, sig_u))
    forc_lower = point_forecast - q * sigma
    forc_upper = point_forecast + q * sigma
    return (point_forecast, forc_lower, forc_upper)

def var_loglike(resid, omega, nobs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the value of the VAR(p) log-likelihood.\n\n    Parameters\n    ----------\n    resid : ndarray (T x K)\n    omega : ndarray\n        Sigma hat matrix.  Each element i,j is the average product of the\n        OLS residual for variable i and the OLS residual for variable j or\n        np.dot(resid.T,resid)/nobs.  There should be no correction for the\n        degrees of freedom.\n    nobs : int\n\n    Returns\n    -------\n    llf : float\n        The value of the loglikelihood function for a VAR(p) model\n\n    Notes\n    -----\n    The loglikelihood function for the VAR(p) is\n\n    .. math::\n\n        -\\left(\\frac{T}{2}\\right)\n        \\left(\\ln\\left|\\Omega\\right|-K\\ln\\left(2\\pi\\right)-K\\right)\n    '
    logdet = logdet_symm(np.asarray(omega))
    neqs = len(omega)
    part1 = -(nobs * neqs / 2) * np.log(2 * np.pi)
    part2 = -(nobs / 2) * (logdet + neqs)
    return part1 + part2

def _reordered(self, order):
    if False:
        while True:
            i = 10
    endog = self.endog
    endog_lagged = self.endog_lagged
    params = self.params
    sigma_u = self.sigma_u
    names = self.names
    k_ar = self.k_ar
    endog_new = np.zeros_like(endog)
    endog_lagged_new = np.zeros_like(endog_lagged)
    params_new_inc = np.zeros_like(params)
    params_new = np.zeros_like(params)
    sigma_u_new_inc = np.zeros_like(sigma_u)
    sigma_u_new = np.zeros_like(sigma_u)
    num_end = len(self.params[0])
    names_new = []
    k = self.k_trend
    for (i, c) in enumerate(order):
        endog_new[:, i] = self.endog[:, c]
        if k > 0:
            params_new_inc[0, i] = params[0, i]
            endog_lagged_new[:, 0] = endog_lagged[:, 0]
        for j in range(k_ar):
            params_new_inc[i + j * num_end + k, :] = self.params[c + j * num_end + k, :]
            endog_lagged_new[:, i + j * num_end + k] = endog_lagged[:, c + j * num_end + k]
        sigma_u_new_inc[i, :] = sigma_u[c, :]
        names_new.append(names[c])
    for (i, c) in enumerate(order):
        params_new[:, i] = params_new_inc[:, c]
        sigma_u_new[:, i] = sigma_u_new_inc[:, c]
    return VARResults(endog=endog_new, endog_lagged=endog_lagged_new, params=params_new, sigma_u=sigma_u_new, lag_order=self.k_ar, model=self.model, trend='c', names=names_new, dates=self.dates)

def orth_ma_rep(results, maxn=10, P=None):
    if False:
        print('Hello World!')
    "Compute Orthogonalized MA coefficient matrices using P matrix such\n    that :math:`\\Sigma_u = PP^\\prime`. P defaults to the Cholesky\n    decomposition of :math:`\\Sigma_u`\n\n    Parameters\n    ----------\n    results : VARResults or VECMResults\n    maxn : int\n        Number of coefficient matrices to compute\n    P : ndarray (neqs x neqs), optional\n        Matrix such that Sigma_u = PP', defaults to the Cholesky decomposition.\n\n    Returns\n    -------\n    coefs : ndarray (maxn x neqs x neqs)\n    "
    if P is None:
        P = results._chol_sigma_u
    ma_mats = results.ma_rep(maxn=maxn)
    return np.array([np.dot(coefs, P) for coefs in ma_mats])

def test_normality(results, signif=0.05):
    if False:
        while True:
            i = 10
    '\n    Test assumption of normal-distributed errors using Jarque-Bera-style\n    omnibus Chi^2 test\n\n    Parameters\n    ----------\n    results : VARResults or statsmodels.tsa.vecm.vecm.VECMResults\n    signif : float\n        The test\'s significance level.\n\n    Notes\n    -----\n    H0 (null) : data are generated by a Gaussian-distributed process\n\n    Returns\n    -------\n    result : NormalityTestResults\n\n    References\n    ----------\n    .. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series*\n       *Analysis*. Springer.\n\n    .. [2] Kilian, L. & Demiroglu, U. (2000). "Residual-Based Tests for\n       Normality in Autoregressions: Asymptotic Theory and Simulation\n       Evidence." Journal of Business & Economic Statistics\n    '
    resid_c = results.resid - results.resid.mean(0)
    sig = np.dot(resid_c.T, resid_c) / results.nobs
    Pinv = np.linalg.inv(np.linalg.cholesky(sig))
    w = np.dot(Pinv, resid_c.T)
    b1 = (w ** 3).sum(1)[:, None] / results.nobs
    b2 = (w ** 4).sum(1)[:, None] / results.nobs - 3
    lam_skew = results.nobs * np.dot(b1.T, b1) / 6
    lam_kurt = results.nobs * np.dot(b2.T, b2) / 24
    lam_omni = float(np.squeeze(lam_skew + lam_kurt))
    omni_dist = stats.chi2(results.neqs * 2)
    omni_pvalue = float(omni_dist.sf(lam_omni))
    crit_omni = float(omni_dist.ppf(1 - signif))
    return NormalityTestResults(lam_omni, crit_omni, omni_pvalue, results.neqs * 2, signif)

class LagOrderResults:
    """
    Results class for choosing a model's lag order.

    Parameters
    ----------
    ics : dict
        The keys are the strings ``"aic"``, ``"bic"``, ``"hqic"``, and
        ``"fpe"``. A corresponding value is a list of information criteria for
        various numbers of lags.
    selected_orders : dict
        The keys are the strings ``"aic"``, ``"bic"``, ``"hqic"``, and
        ``"fpe"``. The corresponding value is an integer specifying the number
        of lags chosen according to a given criterion (key).
    vecm : bool, default: `False`
        `True` indicates that the model is a VECM. In case of a VAR model
        this argument must be `False`.

    Notes
    -----
    In case of a VECM the shown lags are lagged differences.
    """

    def __init__(self, ics, selected_orders, vecm=False):
        if False:
            print('Hello World!')
        self.title = ('VECM' if vecm else 'VAR') + ' Order Selection'
        self.title += ' (* highlights the minimums)'
        self.ics = ics
        self.selected_orders = selected_orders
        self.vecm = vecm
        self.aic = selected_orders['aic']
        self.bic = selected_orders['bic']
        self.hqic = selected_orders['hqic']
        self.fpe = selected_orders['fpe']

    def summary(self):
        if False:
            return 10
        cols = sorted(self.ics)
        str_data = np.array([['%#10.4g' % v for v in self.ics[c]] for c in cols], dtype=object).T
        for (i, col) in enumerate(cols):
            idx = (int(self.selected_orders[col]), i)
            str_data[idx] += '*'
        return SimpleTable(str_data, [col.upper() for col in cols], lrange(len(str_data)), title=self.title)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return f'<{self.__module__}.{self.__class__.__name__} object. Selected orders are: AIC -> {str(self.aic)}, BIC -> {str(self.bic)}, FPE -> {str(self.fpe)}, HQIC ->  {str(self.hqic)}>'

class VAR(TimeSeriesModel):
    """
    Fit VAR(p) process and do lag order selection

    .. math:: y_t = A_1 y_{t-1} + \\ldots + A_p y_{t-p} + u_t

    Parameters
    ----------
    endog : array_like
        2-d endogenous response variable. The independent variable.
    exog : array_like
        2-d exogenous variable.
    dates : array_like
        must match number of rows of endog

    References
    ----------
    Lütkepohl (2005) New Introduction to Multiple Time Series Analysis
    """
    y = deprecated_alias('y', 'endog', remove_version='0.11.0')

    def __init__(self, endog, exog=None, dates=None, freq=None, missing='none'):
        if False:
            print('Hello World!')
        super().__init__(endog, exog, dates, freq, missing=missing)
        if self.endog.ndim == 1:
            raise ValueError('Only gave one variable to VAR')
        self.neqs = self.endog.shape[1]
        self.n_totobs = len(endog)

    def predict(self, params, start=None, end=None, lags=1, trend='c'):
        if False:
            print('Hello World!')
        '\n        Returns in-sample predictions or forecasts\n        '
        params = np.array(params)
        if start is None:
            start = lags
        (start, end, out_of_sample, prediction_index) = self._get_prediction_index(start, end)
        if end < start:
            raise ValueError('end is before start')
        if end == start + out_of_sample:
            return np.array([])
        k_trend = util.get_trendorder(trend)
        k = self.neqs
        k_ar = lags
        predictedvalues = np.zeros((end + 1 - start + out_of_sample, k))
        if k_trend != 0:
            intercept = params[:k_trend]
            predictedvalues += intercept
        y = self.endog
        x = util.get_var_endog(y, lags, trend=trend, has_constant='raise')
        fittedvalues = np.dot(x, params)
        fv_start = start - k_ar
        pv_end = min(len(predictedvalues), len(fittedvalues) - fv_start)
        fv_end = min(len(fittedvalues), end - k_ar + 1)
        predictedvalues[:pv_end] = fittedvalues[fv_start:fv_end]
        if not out_of_sample:
            return predictedvalues
        y = y[-k_ar:]
        coefs = params[k_trend:].reshape((k_ar, k, k)).swapaxes(1, 2)
        predictedvalues[pv_end:] = forecast(y, coefs, intercept, out_of_sample)
        return predictedvalues

    def fit(self, maxlags: int | None=None, method='ols', ic=None, trend='c', verbose=False):
        if False:
            print('Hello World!')
        '\n        Fit the VAR model\n\n        Parameters\n        ----------\n        maxlags : {int, None}, default None\n            Maximum number of lags to check for order selection, defaults to\n            12 * (nobs/100.)**(1./4), see select_order function\n        method : {\'ols\'}\n            Estimation method to use\n        ic : {\'aic\', \'fpe\', \'hqic\', \'bic\', None}\n            Information criterion to use for VAR order selection.\n            aic : Akaike\n            fpe : Final prediction error\n            hqic : Hannan-Quinn\n            bic : Bayesian a.k.a. Schwarz\n        verbose : bool, default False\n            Print order selection output to the screen\n        trend : str {"c", "ct", "ctt", "n"}\n            "c" - add constant\n            "ct" - constant and trend\n            "ctt" - constant, linear and quadratic trend\n            "n" - co constant, no trend\n            Note that these are prepended to the columns of the dataset.\n\n        Returns\n        -------\n        VARResults\n            Estimation results\n\n        Notes\n        -----\n        See Lütkepohl pp. 146-153 for implementation details.\n        '
        lags = maxlags
        if trend not in ['c', 'ct', 'ctt', 'n']:
            raise ValueError("trend '{}' not supported for VAR".format(trend))
        if ic is not None:
            selections = self.select_order(maxlags=maxlags)
            if not hasattr(selections, ic):
                raise ValueError('%s not recognized, must be among %s' % (ic, sorted(selections)))
            lags = getattr(selections, ic)
            if verbose:
                print(selections)
                print('Using %d based on %s criterion' % (lags, ic))
        elif lags is None:
            lags = 1
        k_trend = util.get_trendorder(trend)
        orig_exog_names = self.exog_names
        self.exog_names = util.make_lag_names(self.endog_names, lags, k_trend)
        self.nobs = self.n_totobs - lags
        if self.exog is not None:
            if orig_exog_names:
                x_names_to_add = orig_exog_names
            else:
                x_names_to_add = ['exog%d' % i for i in range(self.exog.shape[1])]
            self.data.xnames = self.data.xnames[:k_trend] + x_names_to_add + self.data.xnames[k_trend:]
        self.data.cov_names = pd.MultiIndex.from_product((self.data.xnames, self.data.ynames))
        return self._estimate_var(lags, trend=trend)

    def _estimate_var(self, lags, offset=0, trend='c'):
        if False:
            return 10
        "\n        lags : int\n            Lags of the endogenous variable.\n        offset : int\n            Periods to drop from beginning-- for order selection so it's an\n            apples-to-apples comparison\n        trend : {str, None}\n            As per above\n        "
        self.k_trend = k_trend = util.get_trendorder(trend)
        if offset < 0:
            raise ValueError('offset must be >= 0')
        nobs = self.n_totobs - lags - offset
        endog = self.endog[offset:]
        exog = None if self.exog is None else self.exog[offset:]
        z = util.get_var_endog(endog, lags, trend=trend, has_constant='raise')
        if exog is not None:
            x = util.get_var_endog(exog[-nobs:], 0, trend='n', has_constant='raise')
            x_inst = exog[-nobs:]
            x = np.column_stack((x, x_inst))
            del x_inst
            temp_z = z
            z = np.empty((x.shape[0], x.shape[1] + z.shape[1]))
            z[:, :self.k_trend] = temp_z[:, :self.k_trend]
            z[:, self.k_trend:self.k_trend + x.shape[1]] = x
            z[:, self.k_trend + x.shape[1]:] = temp_z[:, self.k_trend:]
            del temp_z, x
        for i in range(self.k_trend):
            if (np.diff(z[:, i]) == 1).all():
                z[:, i] += lags
            if (np.diff(np.sqrt(z[:, i])) == 1).all():
                z[:, i] = (np.sqrt(z[:, i]) + lags) ** 2
        y_sample = endog[lags:]
        params = np.linalg.lstsq(z, y_sample, rcond=1e-15)[0]
        resid = y_sample - np.dot(z, params)
        avobs = len(y_sample)
        if exog is not None:
            k_trend += exog.shape[1]
        df_resid = avobs - (self.neqs * lags + k_trend)
        sse = np.dot(resid.T, resid)
        if df_resid:
            omega = sse / df_resid
        else:
            omega = np.full_like(sse, np.nan)
        varfit = VARResults(endog, z, params, omega, lags, names=self.endog_names, trend=trend, dates=self.data.dates, model=self, exog=self.exog)
        return VARResultsWrapper(varfit)

    def select_order(self, maxlags=None, trend='c'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute lag order selections based on each of the available information\n        criteria\n\n        Parameters\n        ----------\n        maxlags : int\n            if None, defaults to 12 * (nobs/100.)**(1./4)\n        trend : str {"n", "c", "ct", "ctt"}\n            * "n" - no deterministic terms\n            * "c" - constant term\n            * "ct" - constant and linear term\n            * "ctt" - constant, linear, and quadratic term\n\n        Returns\n        -------\n        selections : LagOrderResults\n        '
        ntrend = len(trend) if trend.startswith('c') else 0
        max_estimable = (self.n_totobs - self.neqs - ntrend) // (1 + self.neqs)
        if maxlags is None:
            maxlags = int(round(12 * (len(self.endog) / 100.0) ** (1 / 4.0)))
            maxlags = min(maxlags, max_estimable)
        elif maxlags > max_estimable:
            raise ValueError('maxlags is too large for the number of observations and the number of equations. The largest model cannot be estimated.')
        ics = defaultdict(list)
        p_min = 0 if self.exog is not None or trend != 'n' else 1
        for p in range(p_min, maxlags + 1):
            result = self._estimate_var(p, offset=maxlags - p, trend=trend)
            for (k, v) in result.info_criteria.items():
                ics[k].append(v)
        selected_orders = dict(((k, np.array(v).argmin() + p_min) for (k, v) in ics.items()))
        return LagOrderResults(ics, selected_orders, vecm=False)

    @classmethod
    def from_formula(cls, formula, data, subset=None, drop_cols=None, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Not implemented. Formulas are not supported for VAR models.\n        '
        raise NotImplementedError('formulas are not supported for VAR models.')

class VARProcess:
    """
    Class represents a known VAR(p) process

    Parameters
    ----------
    coefs : ndarray (p x k x k)
        coefficients for lags of endog, part or params reshaped
    coefs_exog : ndarray
        parameters for trend and user provided exog
    sigma_u : ndarray (k x k)
        residual covariance
    names : sequence (length k)
    _params_info : dict
        internal dict to provide information about the composition of `params`,
        specifically `k_trend` (trend order) and `k_exog_user` (the number of
        exog variables provided by the user).
        If it is None, then coefs_exog are assumed to be for the intercept and
        trend.
    """

    def __init__(self, coefs, coefs_exog, sigma_u, names=None, _params_info=None):
        if False:
            for i in range(10):
                print('nop')
        self.k_ar = len(coefs)
        self.neqs = coefs.shape[1]
        self.coefs = coefs
        self.coefs_exog = coefs_exog
        self.sigma_u = sigma_u
        self.names = names
        if _params_info is None:
            _params_info = {}
        self.k_exog_user = _params_info.get('k_exog_user', 0)
        if self.coefs_exog is not None:
            k_ex = self.coefs_exog.shape[0] if self.coefs_exog.ndim != 1 else 1
            k_c = k_ex - self.k_exog_user
        else:
            k_c = 0
        self.k_trend = _params_info.get('k_trend', k_c)
        self.k_exog = self.k_trend + self.k_exog_user
        if self.k_trend > 0:
            if coefs_exog.ndim == 2:
                self.intercept = coefs_exog[:, 0]
            else:
                self.intercept = coefs_exog
        else:
            self.intercept = np.zeros(self.neqs)

    def get_eq_index(self, name):
        if False:
            while True:
                i = 10
        'Return integer position of requested equation name'
        return util.get_index(self.names, name)

    def __str__(self):
        if False:
            return 10
        output = 'VAR(%d) process for %d-dimensional response y_t' % (self.k_ar, self.neqs)
        output += '\nstable: %s' % self.is_stable()
        output += '\nmean: %s' % self.mean()
        return output

    def is_stable(self, verbose=False):
        if False:
            return 10
        'Determine stability based on model coefficients\n\n        Parameters\n        ----------\n        verbose : bool\n            Print eigenvalues of the VAR(1) companion\n\n        Notes\n        -----\n        Checks if det(I - Az) = 0 for any mod(z) <= 1, so all the eigenvalues of\n        the companion matrix must lie outside the unit circle\n        '
        return is_stable(self.coefs, verbose=verbose)

    def simulate_var(self, steps=None, offset=None, seed=None, initial_values=None, nsimulations=None):
        if False:
            print('Hello World!')
        '\n        simulate the VAR(p) process for the desired number of steps\n\n        Parameters\n        ----------\n        steps : None or int\n            number of observations to simulate, this includes the initial\n            observations to start the autoregressive process.\n            If offset is not None, then exog of the model are used if they were\n            provided in the model\n        offset : None or ndarray (steps, neqs)\n            If not None, then offset is added as an observation specific\n            intercept to the autoregression. If it is None and either trend\n            (including intercept) or exog were used in the VAR model, then\n            the linear predictor of those components will be used as offset.\n            This should have the same number of rows as steps, and the same\n            number of columns as endogenous variables (neqs).\n        seed : {None, int}\n            If seed is not None, then it will be used with for the random\n            variables generated by numpy.random.\n        initial_values : array_like, optional\n            Initial values for use in the simulation. Shape should be\n            (nlags, neqs) or (neqs,). Values should be ordered from less to\n            most recent. Note that this values will be returned by the\n            simulation as the first values of `endog_simulated` and they\n            will count for the total number of steps.\n        nsimulations : {None, int}\n            Number of simulations to perform. If `nsimulations` is None it will\n            perform one simulation and return value will have shape (steps, neqs).\n\n        Returns\n        -------\n        endog_simulated : nd_array\n            Endog of the simulated VAR process. Shape will be (nsimulations, steps, neqs)\n            or (steps, neqs) if `nsimulations` is None.\n        '
        steps_ = None
        if offset is None:
            if self.k_exog_user > 0 or self.k_trend > 1:
                offset = self.endog_lagged[:, :self.k_exog].dot(self.coefs_exog.T)
                steps_ = self.endog_lagged.shape[0]
            else:
                offset = self.intercept
        else:
            steps_ = offset.shape[0]
        if steps is None:
            if steps_ is None:
                steps = 1000
            else:
                steps = steps_
        elif steps_ is not None and steps != steps_:
            raise ValueError('if exog or offset are used, then steps mustbe equal to their length or None')
        y = util.varsim(self.coefs, offset, self.sigma_u, steps=steps, seed=seed, initial_values=initial_values, nsimulations=nsimulations)
        return y

    def plotsim(self, steps=None, offset=None, seed=None):
        if False:
            i = 10
            return i + 15
        '\n        Plot a simulation from the VAR(p) process for the desired number of\n        steps\n        '
        y = self.simulate_var(steps=steps, offset=offset, seed=seed)
        return plotting.plot_mts(y)

    def intercept_longrun(self):
        if False:
            return 10
        '\n        Long run intercept of stable VAR process\n\n        Lütkepohl eq. 2.1.23\n\n        .. math:: \\mu = (I - A_1 - \\dots - A_p)^{-1} \\alpha\n\n        where \\alpha is the intercept (parameter of the constant)\n        '
        return np.linalg.solve(self._char_mat, self.intercept)

    def mean(self):
        if False:
            while True:
                i = 10
        '\n        Long run intercept of stable VAR process\n\n        Warning: trend and exog except for intercept are ignored for this.\n        This might change in future versions.\n\n        Lütkepohl eq. 2.1.23\n\n        .. math:: \\mu = (I - A_1 - \\dots - A_p)^{-1} \\alpha\n\n        where \\alpha is the intercept (parameter of the constant)\n        '
        return self.intercept_longrun()

    def ma_rep(self, maxn=10):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute MA(:math:`\\infty`) coefficient matrices\n\n        Parameters\n        ----------\n        maxn : int\n            Number of coefficient matrices to compute\n\n        Returns\n        -------\n        coefs : ndarray (maxn x k x k)\n        '
        return ma_rep(self.coefs, maxn=maxn)

    def orth_ma_rep(self, maxn=10, P=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Compute orthogonalized MA coefficient matrices using P matrix such\n        that :math:`\\Sigma_u = PP^\\prime`. P defaults to the Cholesky\n        decomposition of :math:`\\Sigma_u`\n\n        Parameters\n        ----------\n        maxn : int\n            Number of coefficient matrices to compute\n        P : ndarray (k x k), optional\n            Matrix such that Sigma_u = PP', defaults to Cholesky descomp\n\n        Returns\n        -------\n        coefs : ndarray (maxn x k x k)\n        "
        return orth_ma_rep(self, maxn, P)

    def long_run_effects(self):
        if False:
            for i in range(10):
                print('nop')
        'Compute long-run effect of unit impulse\n\n        .. math::\n\n            \\Psi_\\infty = \\sum_{i=0}^\\infty \\Phi_i\n        '
        return np.linalg.inv(self._char_mat)

    @cache_readonly
    def _chol_sigma_u(self):
        if False:
            return 10
        return np.linalg.cholesky(self.sigma_u)

    @cache_readonly
    def _char_mat(self):
        if False:
            while True:
                i = 10
        'Characteristic matrix of the VAR'
        return np.eye(self.neqs) - self.coefs.sum(0)

    def acf(self, nlags=None):
        if False:
            for i in range(10):
                print('nop')
        'Compute theoretical autocovariance function\n\n        Returns\n        -------\n        acf : ndarray (p x k x k)\n        '
        return var_acf(self.coefs, self.sigma_u, nlags=nlags)

    def acorr(self, nlags=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Autocorrelation function\n\n        Parameters\n        ----------\n        nlags : int or None\n            The number of lags to include in the autocovariance function. The\n            default is the number of lags included in the model.\n\n        Returns\n        -------\n        acorr : ndarray\n            Autocorrelation and cross correlations (nlags, neqs, neqs)\n        '
        return util.acf_to_acorr(self.acf(nlags=nlags))

    def plot_acorr(self, nlags=10, linewidth=8):
        if False:
            return 10
        'Plot theoretical autocorrelation function'
        fig = plotting.plot_full_acorr(self.acorr(nlags=nlags), linewidth=linewidth)
        return fig

    def forecast(self, y, steps, exog_future=None):
        if False:
            return 10
        'Produce linear minimum MSE forecasts for desired number of steps\n        ahead, using prior values y\n\n        Parameters\n        ----------\n        y : ndarray (p x k)\n        steps : int\n\n        Returns\n        -------\n        forecasts : ndarray (steps x neqs)\n\n        Notes\n        -----\n        Lütkepohl pp 37-38\n        '
        if self.exog is None and exog_future is not None:
            raise ValueError('No exog in model, so no exog_future supported in forecast method.')
        if self.exog is not None and exog_future is None:
            raise ValueError('Please provide an exog_future argument to the forecast method.')
        exog_future = array_like(exog_future, 'exog_future', optional=True, ndim=2)
        if exog_future is not None:
            if exog_future.shape[0] != steps:
                err_msg = f'exog_future only has {exog_future.shape[0]} observations. It must have steps ({steps}) observations.\n'
                raise ValueError(err_msg)
        trend_coefs = None if self.coefs_exog.size == 0 else self.coefs_exog.T
        exogs = []
        if self.trend.startswith('c'):
            exogs.append(np.ones(steps))
        exog_lin_trend = np.arange(self.n_totobs + 1, self.n_totobs + 1 + steps)
        if 't' in self.trend:
            exogs.append(exog_lin_trend)
        if 'tt' in self.trend:
            exogs.append(exog_lin_trend ** 2)
        if exog_future is not None:
            exogs.append(exog_future)
        if not exogs:
            exog_future = None
        else:
            exog_future = np.column_stack(exogs)
        return forecast(y, self.coefs, trend_coefs, steps, exog_future)

    def mse(self, steps):
        if False:
            print('Hello World!')
        '\n        Compute theoretical forecast error variance matrices\n\n        Parameters\n        ----------\n        steps : int\n            Number of steps ahead\n\n        Notes\n        -----\n        .. math:: \\mathrm{MSE}(h) = \\sum_{i=0}^{h-1} \\Phi \\Sigma_u \\Phi^T\n\n        Returns\n        -------\n        forc_covs : ndarray (steps x neqs x neqs)\n        '
        ma_coefs = self.ma_rep(steps)
        k = len(self.sigma_u)
        forc_covs = np.zeros((steps, k, k))
        prior = np.zeros((k, k))
        for h in range(steps):
            phi = ma_coefs[h]
            var = phi @ self.sigma_u @ phi.T
            forc_covs[h] = prior = prior + var
        return forc_covs
    forecast_cov = mse

    def _forecast_vars(self, steps):
        if False:
            i = 10
            return i + 15
        covs = self.forecast_cov(steps)
        inds = np.arange(self.neqs)
        return covs[:, inds, inds]

    def forecast_interval(self, y, steps, alpha=0.05, exog_future=None):
        if False:
            return 10
        '\n        Construct forecast interval estimates assuming the y are Gaussian\n\n        Parameters\n        ----------\n        y : {ndarray, None}\n            The initial values to use for the forecasts. If None,\n            the last k_ar values of the original endogenous variables are\n            used.\n        steps : int\n            Number of steps ahead to forecast\n        alpha : float, optional\n            The significance level for the confidence intervals.\n        exog_future : ndarray, optional\n            Forecast values of the exogenous variables. Should include\n            constant, trend, etc. as needed, including extrapolating out\n            of sample.\n        Returns\n        -------\n        point : ndarray\n            Mean value of forecast\n        lower : ndarray\n            Lower bound of confidence interval\n        upper : ndarray\n            Upper bound of confidence interval\n\n        Notes\n        -----\n        Lütkepohl pp. 39-40\n        '
        if not 0 < alpha < 1:
            raise ValueError('alpha must be between 0 and 1')
        q = util.norm_signif_level(alpha)
        point_forecast = self.forecast(y, steps, exog_future=exog_future)
        sigma = np.sqrt(self._forecast_vars(steps))
        forc_lower = point_forecast - q * sigma
        forc_upper = point_forecast + q * sigma
        return (point_forecast, forc_lower, forc_upper)

    def to_vecm(self):
        if False:
            for i in range(10):
                print('nop')
        'to_vecm'
        k = self.coefs.shape[1]
        p = self.coefs.shape[0]
        A = self.coefs
        pi = -(np.identity(k) - np.sum(A, 0))
        gamma = np.zeros((p - 1, k, k))
        for i in range(p - 1):
            gamma[i] = -np.sum(A[i + 1:], 0)
        gamma = np.concatenate(gamma, 1)
        return {'Gamma': gamma, 'Pi': pi}

class VARResults(VARProcess):
    """Estimate VAR(p) process with fixed number of lags

    Parameters
    ----------
    endog : ndarray
    endog_lagged : ndarray
    params : ndarray
    sigma_u : ndarray
    lag_order : int
    model : VAR model instance
    trend : str {'n', 'c', 'ct'}
    names : array_like
        List of names of the endogenous variables in order of appearance in
        `endog`.
    dates
    exog : ndarray

    Attributes
    ----------
    params : ndarray (p x K x K)
        Estimated A_i matrices, A_i = coefs[i-1]
    dates
    endog
    endog_lagged
    k_ar : int
        Order of VAR process
    k_trend : int
    model
    names
    neqs : int
        Number of variables (equations)
    nobs : int
    n_totobs : int
    params : ndarray (Kp + 1) x K
        A_i matrices and intercept in stacked form [int A_1 ... A_p]
    names : list
        variables names
    sigma_u : ndarray (K x K)
        Estimate of white noise process variance Var[u_t]
    """
    _model_type = 'VAR'

    def __init__(self, endog, endog_lagged, params, sigma_u, lag_order, model=None, trend='c', names=None, dates=None, exog=None):
        if False:
            for i in range(10):
                print('nop')
        self.model = model
        self.endog = endog
        self.endog_lagged = endog_lagged
        self.dates = dates
        (self.n_totobs, neqs) = self.endog.shape
        self.nobs = self.n_totobs - lag_order
        self.trend = trend
        k_trend = util.get_trendorder(trend)
        self.exog_names = util.make_lag_names(names, lag_order, k_trend, model.data.orig_exog)
        self.params = params
        self.exog = exog
        endog_start = k_trend
        if exog is not None:
            k_exog_user = exog.shape[1]
            endog_start += k_exog_user
        else:
            k_exog_user = 0
        reshaped = self.params[endog_start:]
        reshaped = reshaped.reshape((lag_order, neqs, neqs))
        coefs = reshaped.swapaxes(1, 2).copy()
        self.coefs_exog = params[:endog_start].T
        self.k_exog = self.coefs_exog.shape[1]
        self.k_exog_user = k_exog_user
        _params_info = {'k_trend': k_trend, 'k_exog_user': k_exog_user, 'k_ar': lag_order}
        super().__init__(coefs, self.coefs_exog, sigma_u, names=names, _params_info=_params_info)

    def plot(self):
        if False:
            print('Hello World!')
        'Plot input time series'
        return plotting.plot_mts(self.endog, names=self.names, index=self.dates)

    @property
    def df_model(self):
        if False:
            print('Hello World!')
        '\n        Number of estimated parameters per variable, including the intercept / trends\n        '
        return self.neqs * self.k_ar + self.k_exog

    @property
    def df_resid(self):
        if False:
            for i in range(10):
                print('nop')
        'Number of observations minus number of estimated parameters'
        return self.nobs - self.df_model

    @cache_readonly
    def fittedvalues(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The predicted insample values of the response variables of the model.\n        '
        return np.dot(self.endog_lagged, self.params)

    @cache_readonly
    def resid(self):
        if False:
            while True:
                i = 10
        '\n        Residuals of response variable resulting from estimated coefficients\n        '
        return self.endog[self.k_ar:] - self.fittedvalues

    def sample_acov(self, nlags=1):
        if False:
            for i in range(10):
                print('nop')
        'Sample acov'
        return _compute_acov(self.endog[self.k_ar:], nlags=nlags)

    def sample_acorr(self, nlags=1):
        if False:
            return 10
        'Sample acorr'
        acovs = self.sample_acov(nlags=nlags)
        return _acovs_to_acorrs(acovs)

    def plot_sample_acorr(self, nlags=10, linewidth=8):
        if False:
            i = 10
            return i + 15
        '\n        Plot sample autocorrelation function\n\n        Parameters\n        ----------\n        nlags : int\n            The number of lags to use in compute the autocorrelation. Does\n            not count the zero lag, which will be returned.\n        linewidth : int\n            The linewidth for the plots.\n\n        Returns\n        -------\n        Figure\n            The figure that contains the plot axes.\n        '
        fig = plotting.plot_full_acorr(self.sample_acorr(nlags=nlags), linewidth=linewidth)
        return fig

    def resid_acov(self, nlags=1):
        if False:
            while True:
                i = 10
        '\n        Compute centered sample autocovariance (including lag 0)\n\n        Parameters\n        ----------\n        nlags : int\n\n        Returns\n        -------\n        '
        return _compute_acov(self.resid, nlags=nlags)

    def resid_acorr(self, nlags=1):
        if False:
            print('Hello World!')
        '\n        Compute sample autocorrelation (including lag 0)\n\n        Parameters\n        ----------\n        nlags : int\n\n        Returns\n        -------\n        '
        acovs = self.resid_acov(nlags=nlags)
        return _acovs_to_acorrs(acovs)

    @cache_readonly
    def resid_corr(self):
        if False:
            print('Hello World!')
        '\n        Centered residual correlation matrix\n        '
        return self.resid_acorr(0)[0]

    @cache_readonly
    def sigma_u_mle(self):
        if False:
            i = 10
            return i + 15
        '(Biased) maximum likelihood estimate of noise process covariance'
        if not self.df_resid:
            return np.zeros_like(self.sigma_u)
        return self.sigma_u * self.df_resid / self.nobs

    def cov_params(self):
        if False:
            print('Hello World!')
        'Estimated variance-covariance of model coefficients\n\n        Notes\n        -----\n        Covariance of vec(B), where B is the matrix\n        [params_for_deterministic_terms, A_1, ..., A_p] with the shape\n        (K x (Kp + number_of_deterministic_terms))\n        Adjusted to be an unbiased estimator\n        Ref: Lütkepohl p.74-75\n        '
        z = self.endog_lagged
        return np.kron(np.linalg.inv(z.T @ z), self.sigma_u)

    def cov_ybar(self):
        if False:
            return 10
        'Asymptotically consistent estimate of covariance of the sample mean\n\n        .. math::\n\n            \\sqrt(T) (\\bar{y} - \\mu) \\rightarrow\n                  {\\cal N}(0, \\Sigma_{\\bar{y}}) \\\\\n\n            \\Sigma_{\\bar{y}} = B \\Sigma_u B^\\prime, \\text{where }\n                  B = (I_K - A_1 - \\cdots - A_p)^{-1}\n\n        Notes\n        -----\n        Lütkepohl Proposition 3.3\n        '
        Ainv = np.linalg.inv(np.eye(self.neqs) - self.coefs.sum(0))
        return Ainv @ self.sigma_u @ Ainv.T

    @cache_readonly
    def _zz(self):
        if False:
            for i in range(10):
                print('nop')
        return np.dot(self.endog_lagged.T, self.endog_lagged)

    @property
    def _cov_alpha(self):
        if False:
            i = 10
            return i + 15
        '\n        Estimated covariance matrix of model coefficients w/o exog\n        '
        kn = self.k_exog * self.neqs
        return self.cov_params()[kn:, kn:]

    @cache_readonly
    def _cov_sigma(self):
        if False:
            while True:
                i = 10
        '\n        Estimated covariance matrix of vech(sigma_u)\n        '
        D_K = tsa.duplication_matrix(self.neqs)
        D_Kinv = np.linalg.pinv(D_K)
        sigxsig = np.kron(self.sigma_u, self.sigma_u)
        return 2 * D_Kinv @ sigxsig @ D_Kinv.T

    @cache_readonly
    def llf(self):
        if False:
            return 10
        'Compute VAR(p) loglikelihood'
        return var_loglike(self.resid, self.sigma_u_mle, self.nobs)

    @cache_readonly
    def stderr(self):
        if False:
            i = 10
            return i + 15
        'Standard errors of coefficients, reshaped to match in size'
        stderr = np.sqrt(np.diag(self.cov_params()))
        return stderr.reshape((self.df_model, self.neqs), order='C')
    bse = stderr

    @cache_readonly
    def stderr_endog_lagged(self):
        if False:
            return 10
        'Stderr_endog_lagged'
        start = self.k_exog
        return self.stderr[start:]

    @cache_readonly
    def stderr_dt(self):
        if False:
            for i in range(10):
                print('nop')
        'Stderr_dt'
        end = self.k_exog
        return self.stderr[:end]

    @cache_readonly
    def tvalues(self):
        if False:
            print('Hello World!')
        '\n        Compute t-statistics. Use Student-t(T - Kp - 1) = t(df_resid) to\n        test significance.\n        '
        return self.params / self.stderr

    @cache_readonly
    def tvalues_endog_lagged(self):
        if False:
            return 10
        'tvalues_endog_lagged'
        start = self.k_exog
        return self.tvalues[start:]

    @cache_readonly
    def tvalues_dt(self):
        if False:
            for i in range(10):
                print('nop')
        'tvalues_dt'
        end = self.k_exog
        return self.tvalues[:end]

    @cache_readonly
    def pvalues(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Two-sided p-values for model coefficients from Student t-distribution\n        '
        return 2 * stats.norm.sf(np.abs(self.tvalues))

    @cache_readonly
    def pvalues_endog_lagged(self):
        if False:
            i = 10
            return i + 15
        'pvalues_endog_laggd'
        start = self.k_exog
        return self.pvalues[start:]

    @cache_readonly
    def pvalues_dt(self):
        if False:
            i = 10
            return i + 15
        'pvalues_dt'
        end = self.k_exog
        return self.pvalues[:end]

    def plot_forecast(self, steps, alpha=0.05, plot_stderr=True):
        if False:
            print('Hello World!')
        '\n        Plot forecast\n        '
        (mid, lower, upper) = self.forecast_interval(self.endog[-self.k_ar:], steps, alpha=alpha)
        fig = plotting.plot_var_forc(self.endog, mid, lower, upper, names=self.names, plot_stderr=plot_stderr)
        return fig

    def forecast_cov(self, steps=1, method='mse'):
        if False:
            print('Hello World!')
        'Compute forecast covariance matrices for desired number of steps\n\n        Parameters\n        ----------\n        steps : int\n\n        Notes\n        -----\n        .. math:: \\Sigma_{\\hat y}(h) = \\Sigma_y(h) + \\Omega(h) / T\n\n        Ref: Lütkepohl pp. 96-97\n\n        Returns\n        -------\n        covs : ndarray (steps x k x k)\n        '
        fc_cov = self.mse(steps)
        if method == 'mse':
            pass
        elif method == 'auto':
            if self.k_exog == 1 and self.k_trend < 2:
                fc_cov += self._omega_forc_cov(steps) / self.nobs
                import warnings
                warnings.warn('forecast cov takes parameter uncertainty intoaccount', OutputWarning, stacklevel=2)
        else:
            raise ValueError("method has to be either 'mse' or 'auto'")
        return fc_cov

    def irf_errband_mc(self, orth=False, repl=1000, steps=10, signif=0.05, seed=None, burn=100, cum=False):
        if False:
            print('Hello World!')
        '\n        Compute Monte Carlo integrated error bands assuming normally\n        distributed for impulse response functions\n\n        Parameters\n        ----------\n        orth : bool, default False\n            Compute orthogonalized impulse response error bands\n        repl : int\n            number of Monte Carlo replications to perform\n        steps : int, default 10\n            number of impulse response periods\n        signif : float (0 < signif <1)\n            Significance level for error bars, defaults to 95% CI\n        seed : int\n            np.random.seed for replications\n        burn : int\n            number of initial observations to discard for simulation\n        cum : bool, default False\n            produce cumulative irf error bands\n\n        Notes\n        -----\n        Lütkepohl (2005) Appendix D\n\n        Returns\n        -------\n        Tuple of lower and upper arrays of ma_rep monte carlo standard errors\n        '
        ma_coll = self.irf_resim(orth=orth, repl=repl, steps=steps, seed=seed, burn=burn, cum=cum)
        ma_sort = np.sort(ma_coll, axis=0)
        low_idx = int(round(signif / 2 * repl) - 1)
        upp_idx = int(round((1 - signif / 2) * repl) - 1)
        lower = ma_sort[low_idx, :, :, :]
        upper = ma_sort[upp_idx, :, :, :]
        return (lower, upper)

    def irf_resim(self, orth=False, repl=1000, steps=10, seed=None, burn=100, cum=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Simulates impulse response function, returning an array of simulations.\n        Used for Sims-Zha error band calculation.\n\n        Parameters\n        ----------\n        orth : bool, default False\n            Compute orthogonalized impulse response error bands\n        repl : int\n            number of Monte Carlo replications to perform\n        steps : int, default 10\n            number of impulse response periods\n        signif : float (0 < signif <1)\n            Significance level for error bars, defaults to 95% CI\n        seed : int\n            np.random.seed for replications\n        burn : int\n            number of initial observations to discard for simulation\n        cum : bool, default False\n            produce cumulative irf error bands\n\n        Notes\n        -----\n        .. [*] Sims, Christoper A., and Tao Zha. 1999. "Error Bands for Impulse\n           Response." Econometrica 67: 1113-1155.\n\n        Returns\n        -------\n        Array of simulated impulse response functions\n        '
        neqs = self.neqs
        k_ar = self.k_ar
        coefs = self.coefs
        sigma_u = self.sigma_u
        intercept = self.intercept
        nobs = self.nobs
        nobs_original = nobs + k_ar
        ma_coll = np.zeros((repl, steps + 1, neqs, neqs))

        def fill_coll(sim):
            if False:
                print('Hello World!')
            ret = VAR(sim, exog=self.exog).fit(maxlags=k_ar, trend=self.trend)
            ret = ret.orth_ma_rep(maxn=steps) if orth else ret.ma_rep(maxn=steps)
            return ret.cumsum(axis=0) if cum else ret
        for i in range(repl):
            sim = util.varsim(coefs, intercept, sigma_u, seed=seed, steps=nobs_original + burn)
            sim = sim[burn:]
            ma_coll[i, :, :, :] = fill_coll(sim)
        return ma_coll

    def _omega_forc_cov(self, steps):
        if False:
            for i in range(10):
                print('nop')
        G = self._zz
        Ginv = np.linalg.inv(G)
        B = self._bmat_forc_cov()
        _B = {}

        def bpow(i):
            if False:
                while True:
                    i = 10
            if i not in _B:
                _B[i] = np.linalg.matrix_power(B, i)
            return _B[i]
        phis = self.ma_rep(steps)
        sig_u = self.sigma_u
        omegas = np.zeros((steps, self.neqs, self.neqs))
        for h in range(1, steps + 1):
            if h == 1:
                omegas[h - 1] = self.df_model * self.sigma_u
                continue
            om = omegas[h - 1]
            for i in range(h):
                for j in range(h):
                    Bi = bpow(h - 1 - i)
                    Bj = bpow(h - 1 - j)
                    mult = np.trace(Bi.T @ Ginv @ Bj @ G)
                    om += mult * phis[i] @ sig_u @ phis[j].T
            omegas[h - 1] = om
        return omegas

    def _bmat_forc_cov(self):
        if False:
            return 10
        upper = np.zeros((self.k_exog, self.df_model))
        upper[:, :self.k_exog] = np.eye(self.k_exog)
        lower_dim = self.neqs * (self.k_ar - 1)
        eye = np.eye(lower_dim)
        lower = np.column_stack((np.zeros((lower_dim, self.k_exog)), eye, np.zeros((lower_dim, self.neqs))))
        return np.vstack((upper, self.params.T, lower))

    def summary(self):
        if False:
            i = 10
            return i + 15
        'Compute console output summary of estimates\n\n        Returns\n        -------\n        summary : VARSummary\n        '
        return VARSummary(self)

    def irf(self, periods=10, var_decomp=None, var_order=None):
        if False:
            i = 10
            return i + 15
        "Analyze impulse responses to shocks in system\n\n        Parameters\n        ----------\n        periods : int\n        var_decomp : ndarray (k x k), lower triangular\n            Must satisfy Omega = P P', where P is the passed matrix. Defaults\n            to Cholesky decomposition of Omega\n        var_order : sequence\n            Alternate variable order for Cholesky decomposition\n\n        Returns\n        -------\n        irf : IRAnalysis\n        "
        if var_order is not None:
            raise NotImplementedError('alternate variable order not implemented (yet)')
        return IRAnalysis(self, P=var_decomp, periods=periods)

    def fevd(self, periods=10, var_decomp=None):
        if False:
            while True:
                i = 10
        '\n        Compute forecast error variance decomposition ("fevd")\n\n        Returns\n        -------\n        fevd : FEVD instance\n        '
        return FEVD(self, P=var_decomp, periods=periods)

    def reorder(self, order):
        if False:
            while True:
                i = 10
        'Reorder variables for structural specification'
        if len(order) != len(self.params[0, :]):
            raise ValueError('Reorder specification length should match number of endogenous variables')
        if isinstance(order[0], str):
            order_new = []
            for (i, nam) in enumerate(order):
                order_new.append(self.names.index(order[i]))
            order = order_new
        return _reordered(self, order)

    def test_causality(self, caused, causing=None, kind='f', signif=0.05):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test Granger causality\n\n        Parameters\n        ----------\n        caused : int or str or sequence of int or str\n            If int or str, test whether the variable specified via this index\n            (int) or name (str) is Granger-caused by the variable(s) specified\n            by `causing`.\n            If a sequence of int or str, test whether the corresponding\n            variables are Granger-caused by the variable(s) specified\n            by `causing`.\n        causing : int or str or sequence of int or str or None, default: None\n            If int or str, test whether the variable specified via this index\n            (int) or name (str) is Granger-causing the variable(s) specified by\n            `caused`.\n            If a sequence of int or str, test whether the corresponding\n            variables are Granger-causing the variable(s) specified by\n            `caused`.\n            If None, `causing` is assumed to be the complement of `caused`.\n        kind : {\'f\', \'wald\'}\n            Perform F-test or Wald (chi-sq) test\n        signif : float, default 5%\n            Significance level for computing critical values for test,\n            defaulting to standard 0.05 level\n\n        Notes\n        -----\n        Null hypothesis is that there is no Granger-causality for the indicated\n        variables. The degrees of freedom in the F-test are based on the\n        number of variables in the VAR system, that is, degrees of freedom\n        are equal to the number of equations in the VAR times degree of freedom\n        of a single equation.\n\n        Test for Granger-causality as described in chapter 7.6.3 of [1]_.\n        Test H0: "`causing` does not Granger-cause the remaining variables of\n        the system" against  H1: "`causing` is Granger-causal for the\n        remaining variables".\n\n        Returns\n        -------\n        results : CausalityTestResults\n\n        References\n        ----------\n        .. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series*\n           *Analysis*. Springer.\n        '
        if not 0 < signif < 1:
            raise ValueError('signif has to be between 0 and 1')
        allowed_types = (str, int)
        if isinstance(caused, allowed_types):
            caused = [caused]
        if not all((isinstance(c, allowed_types) for c in caused)):
            raise TypeError('caused has to be of type string or int (or a sequence of these types).')
        caused = [self.names[c] if type(c) is int else c for c in caused]
        caused_ind = [util.get_index(self.names, c) for c in caused]
        if causing is not None:
            if isinstance(causing, allowed_types):
                causing = [causing]
            if not all((isinstance(c, allowed_types) for c in causing)):
                raise TypeError('causing has to be of type string or int (or a sequence of these types) or None.')
            causing = [self.names[c] if type(c) is int else c for c in causing]
            causing_ind = [util.get_index(self.names, c) for c in causing]
        else:
            causing_ind = [i for i in range(self.neqs) if i not in caused_ind]
            causing = [self.names[c] for c in caused_ind]
        (k, p) = (self.neqs, self.k_ar)
        if p == 0:
            err = 'Cannot test Granger Causality in a model with 0 lags.'
            raise RuntimeError(err)
        num_restr = len(causing) * len(caused) * p
        num_det_terms = self.k_exog
        C = np.zeros((num_restr, k * num_det_terms + k ** 2 * p), dtype=float)
        cols_det = k * num_det_terms
        row = 0
        for j in range(p):
            for ing_ind in causing_ind:
                for ed_ind in caused_ind:
                    C[row, cols_det + ed_ind + k * ing_ind + k ** 2 * j] = 1
                    row += 1
        Cb = np.dot(C, vec(self.params.T))
        middle = np.linalg.inv(C @ self.cov_params() @ C.T)
        lam_wald = statistic = Cb @ middle @ Cb
        if kind.lower() == 'wald':
            df = num_restr
            dist = stats.chi2(df)
        elif kind.lower() == 'f':
            statistic = lam_wald / num_restr
            df = (num_restr, k * self.df_resid)
            dist = stats.f(*df)
        else:
            raise ValueError('kind %s not recognized' % kind)
        pvalue = dist.sf(statistic)
        crit_value = dist.ppf(1 - signif)
        return CausalityTestResults(causing, caused, statistic, crit_value, pvalue, df, signif, test='granger', method=kind)

    def test_inst_causality(self, causing, signif=0.05):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test for instantaneous causality\n\n        Parameters\n        ----------\n        causing :\n            If int or str, test whether the corresponding variable is causing\n            the variable(s) specified in caused.\n            If sequence of int or str, test whether the corresponding\n            variables are causing the variable(s) specified in caused.\n        signif : float between 0 and 1, default 5 %\n            Significance level for computing critical values for test,\n            defaulting to standard 0.05 level\n        verbose : bool\n            If True, print a table with the results.\n\n        Returns\n        -------\n        results : dict\n            A dict holding the test\'s results. The dict\'s keys are:\n\n            "statistic" : float\n              The calculated test statistic.\n\n            "crit_value" : float\n              The critical value of the Chi^2-distribution.\n\n            "pvalue" : float\n              The p-value corresponding to the test statistic.\n\n            "df" : float\n              The degrees of freedom of the Chi^2-distribution.\n\n            "conclusion" : str {"reject", "fail to reject"}\n              Whether H0 can be rejected or not.\n\n            "signif" : float\n              Significance level\n\n        Notes\n        -----\n        Test for instantaneous causality as described in chapters 3.6.3 and\n        7.6.4 of [1]_.\n        Test H0: "No instantaneous causality between caused and causing"\n        against H1: "Instantaneous causality between caused and causing\n        exists".\n\n        Instantaneous causality is a symmetric relation (i.e. if causing is\n        "instantaneously causing" caused, then also caused is "instantaneously\n        causing" causing), thus the naming of the parameters (which is chosen\n        to be in accordance with test_granger_causality()) may be misleading.\n\n        This method is not returning the same result as JMulTi. This is\n        because the test is based on a VAR(k_ar) model in statsmodels\n        (in accordance to pp. 104, 320-321 in [1]_) whereas JMulTi seems\n        to be using a VAR(k_ar+1) model.\n\n        References\n        ----------\n        .. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series*\n           *Analysis*. Springer.\n        '
        if not 0 < signif < 1:
            raise ValueError('signif has to be between 0 and 1')
        allowed_types = (str, int)
        if isinstance(causing, allowed_types):
            causing = [causing]
        if not all((isinstance(c, allowed_types) for c in causing)):
            raise TypeError('causing has to be of type string or int (or a ' + 'a sequence of these types).')
        causing = [self.names[c] if type(c) is int else c for c in causing]
        causing_ind = [util.get_index(self.names, c) for c in causing]
        caused_ind = [i for i in range(self.neqs) if i not in causing_ind]
        caused = [self.names[c] for c in caused_ind]
        (k, t, p) = (self.neqs, self.nobs, self.k_ar)
        num_restr = len(causing) * len(caused)
        sigma_u = self.sigma_u
        vech_sigma_u = util.vech(sigma_u)
        sig_mask = np.zeros(sigma_u.shape)
        sig_mask[causing_ind, caused_ind] = 1
        sig_mask[caused_ind, causing_ind] = 1
        vech_sig_mask = util.vech(sig_mask)
        inds = np.nonzero(vech_sig_mask)[0]
        C = np.zeros((num_restr, len(vech_sigma_u)), dtype=float)
        for row in range(num_restr):
            C[row, inds[row]] = 1
        Cs = np.dot(C, vech_sigma_u)
        d = np.linalg.pinv(duplication_matrix(k))
        Cd = np.dot(C, d)
        middle = np.linalg.inv(Cd @ np.kron(sigma_u, sigma_u) @ Cd.T) / 2
        wald_statistic = t * (Cs.T @ middle @ Cs)
        df = num_restr
        dist = stats.chi2(df)
        pvalue = dist.sf(wald_statistic)
        crit_value = dist.ppf(1 - signif)
        return CausalityTestResults(causing, caused, wald_statistic, crit_value, pvalue, df, signif, test='inst', method='wald')

    def test_whiteness(self, nlags=10, signif=0.05, adjusted=False):
        if False:
            while True:
                i = 10
        '\n        Residual whiteness tests using Portmanteau test\n\n        Parameters\n        ----------\n        nlags : int > 0\n            The number of lags tested must be larger than the number of lags\n            included in the VAR model.\n        signif : float, between 0 and 1\n            The significance level of the test.\n        adjusted : bool, default False\n            Flag indicating to apply small-sample adjustments.\n\n        Returns\n        -------\n        WhitenessTestResults\n            The test results.\n\n        Notes\n        -----\n        Test the whiteness of the residuals using the Portmanteau test as\n        described in [1]_, chapter 4.4.3.\n\n        References\n        ----------\n        .. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series*\n           *Analysis*. Springer.\n        '
        if nlags - self.k_ar <= 0:
            raise ValueError(f'The whiteness test can only be used when nlags is larger than the number of lags included in the model ({self.k_ar}).')
        statistic = 0
        u = np.asarray(self.resid)
        acov_list = _compute_acov(u, nlags)
        cov0_inv = np.linalg.inv(acov_list[0])
        for t in range(1, nlags + 1):
            ct = acov_list[t]
            to_add = np.trace(ct.T @ cov0_inv @ ct @ cov0_inv)
            if adjusted:
                to_add /= self.nobs - t
            statistic += to_add
        statistic *= self.nobs ** 2 if adjusted else self.nobs
        df = self.neqs ** 2 * (nlags - self.k_ar)
        dist = stats.chi2(df)
        pvalue = dist.sf(statistic)
        crit_value = dist.ppf(1 - signif)
        return WhitenessTestResults(statistic, crit_value, pvalue, df, signif, nlags, adjusted)

    def plot_acorr(self, nlags=10, resid=True, linewidth=8):
        if False:
            while True:
                i = 10
        '\n        Plot autocorrelation of sample (endog) or residuals\n\n        Sample (Y) or Residual autocorrelations are plotted together with the\n        standard :math:`2 / \\sqrt{T}` bounds.\n\n        Parameters\n        ----------\n        nlags : int\n            number of lags to display (excluding 0)\n        resid : bool\n            If True, then the autocorrelation of the residuals is plotted\n            If False, then the autocorrelation of endog is plotted.\n        linewidth : int\n            width of vertical bars\n\n        Returns\n        -------\n        Figure\n            Figure instance containing the plot.\n        '
        if resid:
            acorrs = self.resid_acorr(nlags)
        else:
            acorrs = self.sample_acorr(nlags)
        bound = 2 / np.sqrt(self.nobs)
        fig = plotting.plot_full_acorr(acorrs[1:], xlabel=np.arange(1, nlags + 1), err_bound=bound, linewidth=linewidth)
        fig.suptitle('ACF plots for residuals with $2 / \\sqrt{T}$ bounds ')
        return fig

    def test_normality(self, signif=0.05):
        if False:
            return 10
        '\n        Test assumption of normal-distributed errors using Jarque-Bera-style\n        omnibus Chi^2 test.\n\n        Parameters\n        ----------\n        signif : float\n            Test significance level.\n\n        Returns\n        -------\n        result : NormalityTestResults\n\n        Notes\n        -----\n        H0 (null) : data are generated by a Gaussian-distributed process\n        '
        return test_normality(self, signif=signif)

    @cache_readonly
    def detomega(self):
        if False:
            while True:
                i = 10
        '\n        Return determinant of white noise covariance with degrees of freedom\n        correction:\n\n        .. math::\n\n            \\hat \\Omega = \\frac{T}{T - Kp - 1} \\hat \\Omega_{\\mathrm{MLE}}\n        '
        return np.linalg.det(self.sigma_u)

    @cache_readonly
    def info_criteria(self):
        if False:
            print('Hello World!')
        'information criteria for lagorder selection'
        nobs = self.nobs
        neqs = self.neqs
        lag_order = self.k_ar
        free_params = lag_order * neqs ** 2 + neqs * self.k_exog
        if self.df_resid:
            ld = logdet_symm(self.sigma_u_mle)
        else:
            ld = -np.inf
        aic = ld + 2.0 / nobs * free_params
        bic = ld + np.log(nobs) / nobs * free_params
        hqic = ld + 2.0 * np.log(np.log(nobs)) / nobs * free_params
        if self.df_resid:
            fpe = ((nobs + self.df_model) / self.df_resid) ** neqs * np.exp(ld)
        else:
            fpe = np.inf
        return {'aic': aic, 'bic': bic, 'hqic': hqic, 'fpe': fpe}

    @property
    def aic(self):
        if False:
            while True:
                i = 10
        'Akaike information criterion'
        return self.info_criteria['aic']

    @property
    def fpe(self):
        if False:
            print('Hello World!')
        'Final Prediction Error (FPE)\n\n        Lütkepohl p. 147, see info_criteria\n        '
        return self.info_criteria['fpe']

    @property
    def hqic(self):
        if False:
            return 10
        'Hannan-Quinn criterion'
        return self.info_criteria['hqic']

    @property
    def bic(self):
        if False:
            print('Hello World!')
        'Bayesian a.k.a. Schwarz info criterion'
        return self.info_criteria['bic']

    @cache_readonly
    def roots(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The roots of the VAR process are the solution to\n        (I - coefs[0]*z - coefs[1]*z**2 ... - coefs[p-1]*z**k_ar) = 0.\n        Note that the inverse roots are returned, and stability requires that\n        the roots lie outside the unit circle.\n        '
        neqs = self.neqs
        k_ar = self.k_ar
        p = neqs * k_ar
        arr = np.zeros((p, p))
        arr[:neqs, :] = np.column_stack(self.coefs)
        arr[neqs:, :-neqs] = np.eye(p - neqs)
        roots = np.linalg.eig(arr)[0] ** (-1)
        idx = np.argsort(np.abs(roots))[::-1]
        return roots[idx]

class VARResultsWrapper(wrap.ResultsWrapper):
    _attrs = {'bse': 'columns_eq', 'cov_params': 'cov', 'params': 'columns_eq', 'pvalues': 'columns_eq', 'tvalues': 'columns_eq', 'sigma_u': 'cov_eq', 'sigma_u_mle': 'cov_eq', 'stderr': 'columns_eq'}
    _wrap_attrs = wrap.union_dicts(TimeSeriesResultsWrapper._wrap_attrs, _attrs)
    _methods = {'conf_int': 'multivariate_confint'}
    _wrap_methods = wrap.union_dicts(TimeSeriesResultsWrapper._wrap_methods, _methods)
wrap.populate_wrapper(VARResultsWrapper, VARResults)

class FEVD:
    """
    Compute and plot Forecast error variance decomposition and asymptotic
    standard errors
    """

    def __init__(self, model, P=None, periods=None):
        if False:
            return 10
        self.periods = periods
        self.model = model
        self.neqs = model.neqs
        self.names = model.model.endog_names
        self.irfobj = model.irf(var_decomp=P, periods=periods)
        self.orth_irfs = self.irfobj.orth_irfs
        irfs = (self.orth_irfs[:periods] ** 2).cumsum(axis=0)
        rng = lrange(self.neqs)
        mse = self.model.mse(periods)[:, rng, rng]
        fevd = np.empty_like(irfs)
        for i in range(periods):
            fevd[i] = (irfs[i].T / mse[i]).T
        self.decomp = fevd.swapaxes(0, 1)

    def summary(self):
        if False:
            while True:
                i = 10
        buf = StringIO()
        rng = lrange(self.periods)
        for i in range(self.neqs):
            ppm = output.pprint_matrix(self.decomp[i], rng, self.names)
            buf.write('FEVD for %s\n' % self.names[i])
            buf.write(ppm + '\n')
        print(buf.getvalue())

    def cov(self):
        if False:
            print('Hello World!')
        'Compute asymptotic standard errors\n\n        Returns\n        -------\n        '
        raise NotImplementedError

    def plot(self, periods=None, figsize=(10, 10), **plot_kwds):
        if False:
            print('Hello World!')
        'Plot graphical display of FEVD\n\n        Parameters\n        ----------\n        periods : int, default None\n            Defaults to number originally specified. Can be at most that number\n        '
        import matplotlib.pyplot as plt
        k = self.neqs
        periods = periods or self.periods
        (fig, axes) = plt.subplots(nrows=k, figsize=figsize)
        fig.suptitle('Forecast error variance decomposition (FEVD)')
        colors = [str(c) for c in np.arange(k, dtype=float) / k]
        ticks = np.arange(periods)
        limits = self.decomp.cumsum(2)
        ax = axes[0]
        for i in range(k):
            ax = axes[i]
            this_limits = limits[i].T
            handles = []
            for j in range(k):
                lower = this_limits[j - 1] if j > 0 else 0
                upper = this_limits[j]
                handle = ax.bar(ticks, upper - lower, bottom=lower, color=colors[j], label=self.names[j], **plot_kwds)
                handles.append(handle)
            ax.set_title(self.names[i])
        (handles, labels) = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        plotting.adjust_subplots(right=0.85)
        return fig

def _compute_acov(x, nlags=1):
    if False:
        i = 10
        return i + 15
    x = x - x.mean(0)
    result = []
    for lag in range(nlags + 1):
        if lag > 0:
            r = np.dot(x[lag:].T, x[:-lag])
        else:
            r = np.dot(x.T, x)
        result.append(r)
    return np.array(result) / len(x)

def _acovs_to_acorrs(acovs):
    if False:
        i = 10
        return i + 15
    sd = np.sqrt(np.diag(acovs[0]))
    return acovs / np.outer(sd, sd)