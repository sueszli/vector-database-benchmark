"""
Quantile regression model

Model parameters are estimated using iterated reweighted least squares. The
asymptotic covariance matrix estimated using kernel density estimation.

Author: Vincent Arel-Bundock
License: BSD-3
Created: 2013-03-19

The original IRLS function was written for Matlab by Shapour Mohammadi,
University of Tehran, 2008 (shmohammadi@gmail.com), with some lines based on
code written by James P. Lesage in Applied Econometrics Using MATLAB(1999).PP.
73-4.  Translated to python with permission from original author by Christian
Prinoth (christian at prinoth dot name).
"""
import numpy as np
import warnings
import scipy.stats as stats
from numpy.linalg import pinv
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import RegressionModel, RegressionResults, RegressionResultsWrapper
from statsmodels.tools.sm_exceptions import ConvergenceWarning, IterationLimitWarning

class QuantReg(RegressionModel):
    """Quantile Regression

    Estimate a quantile regression model using iterative reweighted least
    squares.

    Parameters
    ----------
    endog : array or dataframe
        endogenous/response variable
    exog : array or dataframe
        exogenous/explanatory variable(s)

    Notes
    -----
    The Least Absolute Deviation (LAD) estimator is a special case where
    quantile is set to 0.5 (q argument of the fit method).

    The asymptotic covariance matrix is estimated following the procedure in
    Greene (2008, p.407-408), using either the logistic or gaussian kernels
    (kernel argument of the fit method).

    References
    ----------
    General:

    * Birkes, D. and Y. Dodge(1993). Alternative Methods of Regression, John Wiley and Sons.
    * Green,W. H. (2008). Econometric Analysis. Sixth Edition. International Student Edition.
    * Koenker, R. (2005). Quantile Regression. New York: Cambridge University Press.
    * LeSage, J. P.(1999). Applied Econometrics Using MATLAB,

    Kernels (used by the fit method):

    * Green (2008) Table 14.2

    Bandwidth selection (used by the fit method):

    * Bofinger, E. (1975). Estimation of a density function using order statistics. Australian Journal of Statistics 17: 1-17.
    * Chamberlain, G. (1994). Quantile regression, censoring, and the structure of wages. In Advances in Econometrics, Vol. 1: Sixth World Congress, ed. C. A. Sims, 171-209. Cambridge: Cambridge University Press.
    * Hall, P., and S. Sheather. (1988). On the distribution of the Studentized quantile. Journal of the Royal Statistical Society, Series B 50: 381-391.

    Keywords: Least Absolute Deviation(LAD) Regression, Quantile Regression,
    Regression, Robust Estimation.
    """

    def __init__(self, endog, exog, **kwargs):
        if False:
            return 10
        self._check_kwargs(kwargs)
        super(QuantReg, self).__init__(endog, exog, **kwargs)

    def whiten(self, data):
        if False:
            return 10
        '\n        QuantReg model whitener does nothing: returns data.\n        '
        return data

    def fit(self, q=0.5, vcov='robust', kernel='epa', bandwidth='hsheather', max_iter=1000, p_tol=1e-06, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Solve by Iterative Weighted Least Squares\n\n        Parameters\n        ----------\n        q : float\n            Quantile must be strictly between 0 and 1\n        vcov : str, method used to calculate the variance-covariance matrix\n            of the parameters. Default is ``robust``:\n\n            - robust : heteroskedasticity robust standard errors (as suggested\n              in Greene 6th edition)\n            - iid : iid errors (as in Stata 12)\n\n        kernel : str, kernel to use in the kernel density estimation for the\n            asymptotic covariance matrix:\n\n            - epa: Epanechnikov\n            - cos: Cosine\n            - gau: Gaussian\n            - par: Parzene\n\n        bandwidth : str, Bandwidth selection method in kernel density\n            estimation for asymptotic covariance estimate (full\n            references in QuantReg docstring):\n\n            - hsheather: Hall-Sheather (1988)\n            - bofinger: Bofinger (1975)\n            - chamberlain: Chamberlain (1994)\n        '
        if q <= 0 or q >= 1:
            raise Exception('q must be strictly between 0 and 1')
        kern_names = ['biw', 'cos', 'epa', 'gau', 'par']
        if kernel not in kern_names:
            raise Exception('kernel must be one of ' + ', '.join(kern_names))
        else:
            kernel = kernels[kernel]
        if bandwidth == 'hsheather':
            bandwidth = hall_sheather
        elif bandwidth == 'bofinger':
            bandwidth = bofinger
        elif bandwidth == 'chamberlain':
            bandwidth = chamberlain
        else:
            raise Exception("bandwidth must be in 'hsheather', 'bofinger', 'chamberlain'")
        endog = self.endog
        exog = self.exog
        nobs = self.nobs
        exog_rank = np.linalg.matrix_rank(self.exog)
        self.rank = exog_rank
        self.df_model = float(self.rank - self.k_constant)
        self.df_resid = self.nobs - self.rank
        n_iter = 0
        xstar = exog
        beta = np.ones(exog.shape[1])
        diff = 10
        cycle = False
        history = dict(params=[], mse=[])
        while n_iter < max_iter and diff > p_tol and (not cycle):
            n_iter += 1
            beta0 = beta
            xtx = np.dot(xstar.T, exog)
            xty = np.dot(xstar.T, endog)
            beta = np.dot(pinv(xtx), xty)
            resid = endog - np.dot(exog, beta)
            mask = np.abs(resid) < 1e-06
            resid[mask] = ((resid[mask] >= 0) * 2 - 1) * 1e-06
            resid = np.where(resid < 0, q * resid, (1 - q) * resid)
            resid = np.abs(resid)
            xstar = exog / resid[:, np.newaxis]
            diff = np.max(np.abs(beta - beta0))
            history['params'].append(beta)
            history['mse'].append(np.mean(resid * resid))
            if n_iter >= 300 and n_iter % 100 == 0:
                for ii in range(2, 10):
                    if np.all(beta == history['params'][-ii]):
                        cycle = True
                        warnings.warn('Convergence cycle detected', ConvergenceWarning)
                        break
        if n_iter == max_iter:
            warnings.warn('Maximum number of iterations (' + str(max_iter) + ') reached.', IterationLimitWarning)
        e = endog - np.dot(exog, beta)
        iqre = stats.scoreatpercentile(e, 75) - stats.scoreatpercentile(e, 25)
        h = bandwidth(nobs, q)
        h = min(np.std(endog), iqre / 1.34) * (norm.ppf(q + h) - norm.ppf(q - h))
        fhat0 = 1.0 / (nobs * h) * np.sum(kernel(e / h))
        if vcov == 'robust':
            d = np.where(e > 0, (q / fhat0) ** 2, ((1 - q) / fhat0) ** 2)
            xtxi = pinv(np.dot(exog.T, exog))
            xtdx = np.dot(exog.T * d[np.newaxis, :], exog)
            vcov = xtxi @ xtdx @ xtxi
        elif vcov == 'iid':
            vcov = (1.0 / fhat0) ** 2 * q * (1 - q) * pinv(np.dot(exog.T, exog))
        else:
            raise Exception("vcov must be 'robust' or 'iid'")
        lfit = QuantRegResults(self, beta, normalized_cov_params=vcov)
        lfit.q = q
        lfit.iterations = n_iter
        lfit.sparsity = 1.0 / fhat0
        lfit.bandwidth = h
        lfit.history = history
        return RegressionResultsWrapper(lfit)

def _parzen(u):
    if False:
        print('Hello World!')
    z = np.where(np.abs(u) <= 0.5, 4.0 / 3 - 8.0 * u ** 2 + 8.0 * np.abs(u) ** 3, 8.0 * (1 - np.abs(u)) ** 3 / 3.0)
    z[np.abs(u) > 1] = 0
    return z
kernels = {}
kernels['biw'] = lambda u: 15.0 / 16 * (1 - u ** 2) ** 2 * np.where(np.abs(u) <= 1, 1, 0)
kernels['cos'] = lambda u: np.where(np.abs(u) <= 0.5, 1 + np.cos(2 * np.pi * u), 0)
kernels['epa'] = lambda u: 3.0 / 4 * (1 - u ** 2) * np.where(np.abs(u) <= 1, 1, 0)
kernels['gau'] = norm.pdf
kernels['par'] = _parzen

def hall_sheather(n, q, alpha=0.05):
    if False:
        while True:
            i = 10
    z = norm.ppf(q)
    num = 1.5 * norm.pdf(z) ** 2.0
    den = 2.0 * z ** 2.0 + 1.0
    h = n ** (-1.0 / 3) * norm.ppf(1.0 - alpha / 2.0) ** (2.0 / 3) * (num / den) ** (1.0 / 3)
    return h

def bofinger(n, q):
    if False:
        return 10
    num = 9.0 / 2 * norm.pdf(2 * norm.ppf(q)) ** 4
    den = (2 * norm.ppf(q) ** 2 + 1) ** 2
    h = n ** (-1.0 / 5) * (num / den) ** (1.0 / 5)
    return h

def chamberlain(n, q, alpha=0.05):
    if False:
        while True:
            i = 10
    return norm.ppf(1 - alpha / 2) * np.sqrt(q * (1 - q) / n)

class QuantRegResults(RegressionResults):
    """Results instance for the QuantReg model"""

    @cache_readonly
    def prsquared(self):
        if False:
            print('Hello World!')
        q = self.q
        endog = self.model.endog
        e = self.resid
        e = np.where(e < 0, (1 - q) * e, q * e)
        e = np.abs(e)
        ered = endog - stats.scoreatpercentile(endog, q * 100)
        ered = np.where(ered < 0, (1 - q) * ered, q * ered)
        ered = np.abs(ered)
        return 1 - np.sum(e) / np.sum(ered)

    def scale(self):
        if False:
            print('Hello World!')
        return 1.0

    @cache_readonly
    def bic(self):
        if False:
            i = 10
            return i + 15
        return np.nan

    @cache_readonly
    def aic(self):
        if False:
            i = 10
            return i + 15
        return np.nan

    @cache_readonly
    def llf(self):
        if False:
            for i in range(10):
                print('nop')
        return np.nan

    @cache_readonly
    def rsquared(self):
        if False:
            print('Hello World!')
        return np.nan

    @cache_readonly
    def rsquared_adj(self):
        if False:
            print('Hello World!')
        return np.nan

    @cache_readonly
    def mse(self):
        if False:
            for i in range(10):
                print('nop')
        return np.nan

    @cache_readonly
    def mse_model(self):
        if False:
            return 10
        return np.nan

    @cache_readonly
    def mse_total(self):
        if False:
            i = 10
            return i + 15
        return np.nan

    @cache_readonly
    def centered_tss(self):
        if False:
            print('Hello World!')
        return np.nan

    @cache_readonly
    def uncentered_tss(self):
        if False:
            print('Hello World!')
        return np.nan

    @cache_readonly
    def HC0_se(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    @cache_readonly
    def HC1_se(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    @cache_readonly
    def HC2_se(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    @cache_readonly
    def HC3_se(self):
        if False:
            return 10
        raise NotImplementedError

    def summary(self, yname=None, xname=None, title=None, alpha=0.05):
        if False:
            return 10
        'Summarize the Regression Results\n\n        Parameters\n        ----------\n        yname : str, optional\n            Default is `y`\n        xname : list[str], optional\n            Names for the exogenous variables. Default is `var_##` for ## in\n            the number of regressors. Must match the number of parameters\n            in the model\n        title : str, optional\n            Title for the top table. If not None, then this replaces the\n            default title\n        alpha : float\n            significance level for the confidence intervals\n\n        Returns\n        -------\n        smry : Summary instance\n            this holds the summary tables and text, which can be printed or\n            converted to various output formats.\n\n        See Also\n        --------\n        statsmodels.iolib.summary.Summary : class to hold summary results\n        '
        eigvals = self.eigenvals
        condno = self.condition_number
        top_left = [('Dep. Variable:', None), ('Model:', None), ('Method:', ['Least Squares']), ('Date:', None), ('Time:', None)]
        top_right = [('Pseudo R-squared:', ['%#8.4g' % self.prsquared]), ('Bandwidth:', ['%#8.4g' % self.bandwidth]), ('Sparsity:', ['%#8.4g' % self.sparsity]), ('No. Observations:', None), ('Df Residuals:', None), ('Df Model:', None)]
        if title is None:
            title = self.model.__class__.__name__ + ' ' + 'Regression Results'
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right, yname=yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha, use_t=self.use_t)
        etext = []
        if eigvals[-1] < 1e-10:
            wstr = 'The smallest eigenvalue is %6.3g. This might indicate '
            wstr += 'that there are\n'
            wstr += 'strong multicollinearity problems or that the design '
            wstr += 'matrix is singular.'
            wstr = wstr % eigvals[-1]
            etext.append(wstr)
        elif condno > 1000:
            wstr = 'The condition number is large, %6.3g. This might '
            wstr += 'indicate that there are\n'
            wstr += 'strong multicollinearity or other numerical '
            wstr += 'problems.'
            wstr = wstr % condno
            etext.append(wstr)
        if etext:
            smry.add_extra_txt(etext)
        return smry