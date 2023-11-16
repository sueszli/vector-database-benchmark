"""
Implements Lilliefors corrected Kolmogorov-Smirnov tests for normal and
exponential distributions.

`kstest_fit` is provided as a top-level function to access both tests.
`kstest_normal` and `kstest_exponential` are provided as convenience functions
with the appropriate test as the default.
`lilliefors` is provided as an alias for `kstest_fit`.

Created on Sat Oct 01 13:16:49 2011

Author: Josef Perktold
License: BSD-3

pvalues for Lilliefors test are based on formula and table in

An Analytic Approximation to the Distribution of Lilliefors's Test Statistic
for Normality
Author(s): Gerard E. Dallal and Leland WilkinsonSource: The American
Statistician, Vol. 40, No. 4 (Nov., 1986), pp. 294-296
Published by: American Statistical Association
Stable URL: http://www.jstor.org/stable/2684607 .

On the Kolmogorov-Smirnov Test for Normality with Mean and Variance Unknown
Hubert W. Lilliefors
Journal of the American Statistical Association, Vol. 62, No. 318.
(Jun., 1967), pp. 399-402.

---

Updated 2017-07-23
Jacob C. Kimmel

Ref:
Lilliefors, H.W.
On the Kolmogorov-Smirnov test for the exponential distribution with mean
unknown. Journal of the American Statistical Association, Vol 64, No. 325.
(1969), pp. 387–389.
"""
from functools import partial
import numpy as np
from scipy import stats
from statsmodels.tools.validation import string_like
from ._lilliefors_critical_values import critical_values, asymp_critical_values, PERCENTILES
from .tabledist import TableDist

def _make_asymptotic_function(params):
    if False:
        return 10
    '\n    Generates an asymptotic distribution callable from a param matrix\n\n    Polynomial is a[0] * x**(-1/2) + a[1] * x**(-1) + a[2] * x**(-3/2)\n\n    Parameters\n    ----------\n    params : ndarray\n        Array with shape (nalpha, 3) where nalpha is the number of\n        significance levels\n    '

    def f(n):
        if False:
            i = 10
            return i + 15
        poly = np.array([1, np.log(n), np.log(n) ** 2])
        return np.exp(poly.dot(params.T))
    return f

def ksstat(x, cdf, alternative='two_sided', args=()):
    if False:
        i = 10
        return i + 15
    '\n    Calculate statistic for the Kolmogorov-Smirnov test for goodness of fit\n\n    This calculates the test statistic for a test of the distribution G(x) of\n    an observed variable against a given distribution F(x). Under the null\n    hypothesis the two distributions are identical, G(x)=F(x). The\n    alternative hypothesis can be either \'two_sided\' (default), \'less\'\n    or \'greater\'. The KS test is only valid for continuous distributions.\n\n    Parameters\n    ----------\n    x : array_like, 1d\n        array of observations\n    cdf : str or callable\n        string: name of a distribution in scipy.stats\n        callable: function to evaluate cdf\n    alternative : \'two_sided\' (default), \'less\' or \'greater\'\n        defines the alternative hypothesis (see explanation)\n    args : tuple, sequence\n        distribution parameters for call to cdf\n\n\n    Returns\n    -------\n    D : float\n        KS test statistic, either D, D+ or D-\n\n    See Also\n    --------\n    scipy.stats.kstest\n\n    Notes\n    -----\n\n    In the one-sided test, the alternative is that the empirical\n    cumulative distribution function of the random variable is "less"\n    or "greater" than the cumulative distribution function F(x) of the\n    hypothesis, G(x)<=F(x), resp. G(x)>=F(x).\n\n    In contrast to scipy.stats.kstest, this function only calculates the\n    statistic which can be used either as distance measure or to implement\n    case specific p-values.\n    '
    nobs = float(len(x))
    if isinstance(cdf, str):
        cdf = getattr(stats.distributions, cdf).cdf
    elif hasattr(cdf, 'cdf'):
        cdf = getattr(cdf, 'cdf')
    x = np.sort(x)
    cdfvals = cdf(x, *args)
    d_plus = (np.arange(1.0, nobs + 1) / nobs - cdfvals).max()
    d_min = (cdfvals - np.arange(0.0, nobs) / nobs).max()
    if alternative == 'greater':
        return d_plus
    elif alternative == 'less':
        return d_min
    return np.max([d_plus, d_min])

def get_lilliefors_table(dist='norm'):
    if False:
        print('Hello World!')
    "\n    Generates tables for significance levels of Lilliefors test statistics\n\n    Tables for available normal and exponential distribution testing,\n    as specified in Lilliefors references above\n\n    Parameters\n    ----------\n    dist : str\n        distribution being tested in set {'norm', 'exp'}.\n\n    Returns\n    -------\n    lf : TableDist object.\n        table of critical values\n    "
    alpha = 1 - np.array(PERCENTILES) / 100.0
    alpha = alpha[::-1]
    dist = 'normal' if dist == 'norm' else dist
    if dist not in critical_values:
        raise ValueError("Invalid dist parameter. Must be 'norm' or 'exp'")
    cv_data = critical_values[dist]
    acv_data = asymp_critical_values[dist]
    size = np.array(sorted(cv_data), dtype=float)
    crit_lf = np.array([cv_data[key] for key in sorted(cv_data)])
    crit_lf = crit_lf[:, ::-1]
    asym_params = np.array([acv_data[key] for key in sorted(acv_data)])
    asymp_fn = _make_asymptotic_function(asym_params[::-1])
    lf = TableDist(alpha, size, crit_lf, asymptotic=asymp_fn)
    return lf
lilliefors_table_norm = get_lilliefors_table(dist='norm')
lilliefors_table_expon = get_lilliefors_table(dist='exp')

def pval_lf(d_max, n):
    if False:
        for i in range(10):
            print('nop')
    '\n    Approximate pvalues for Lilliefors test\n\n    This is only valid for pvalues smaller than 0.1 which is not checked in\n    this function.\n\n    Parameters\n    ----------\n    d_max : array_like\n        two-sided Kolmogorov-Smirnov test statistic\n    n : int or float\n        sample size\n\n    Returns\n    -------\n    p-value : float or ndarray\n        pvalue according to approximation formula of Dallal and Wilkinson.\n\n    Notes\n    -----\n    This is mainly a helper function where the calling code should dispatch\n    on bound violations. Therefore it does not check whether the pvalue is in\n    the valid range.\n\n    Precision for the pvalues is around 2 to 3 decimals. This approximation is\n    also used by other statistical packages (e.g. R:fBasics) but might not be\n    the most precise available.\n\n    References\n    ----------\n    DallalWilkinson1986\n    '
    if n > 100:
        d_max *= (n / 100.0) ** 0.49
        n = 100
    pval = np.exp(-7.01256 * d_max ** 2 * (n + 2.78019) + 2.99587 * d_max * np.sqrt(n + 2.78019) - 0.122119 + 0.974598 / np.sqrt(n) + 1.67997 / n)
    return pval

def kstest_fit(x, dist='norm', pvalmethod='table'):
    if False:
        while True:
            i = 10
    "\n    Test assumed normal or exponential distribution using Lilliefors' test.\n\n    Lilliefors' test is a Kolmogorov-Smirnov test with estimated parameters.\n\n    Parameters\n    ----------\n    x : array_like, 1d\n        Data to test.\n    dist : {'norm', 'exp'}, optional\n        The assumed distribution.\n    pvalmethod : {'approx', 'table'}, optional\n        The method used to compute the p-value of the test statistic. In\n        general, 'table' is preferred and makes use of a very large simulation.\n        'approx' is only valid for normality. if `dist = 'exp'` `table` is\n        always used. 'approx' uses the approximation formula of Dalal and\n        Wilkinson, valid for pvalues < 0.1. If the pvalue is larger than 0.1,\n        then the result of `table` is returned.\n\n    Returns\n    -------\n    ksstat : float\n        Kolmogorov-Smirnov test statistic with estimated mean and variance.\n    pvalue : float\n        If the pvalue is lower than some threshold, e.g. 0.05, then we can\n        reject the Null hypothesis that the sample comes from a normal\n        distribution.\n\n    Notes\n    -----\n    'table' uses an improved table based on 10,000,000 simulations. The\n    critical values are approximated using\n    log(cv_alpha) = b_alpha + c[0] log(n) + c[1] log(n)**2\n    where cv_alpha is the critical value for a test with size alpha,\n    b_alpha is an alpha-specific intercept term and c[1] and c[2] are\n    coefficients that are shared all alphas.\n    Values in the table are linearly interpolated. Values outside the\n    range are be returned as bounds, 0.990 for large and 0.001 for small\n    pvalues.\n\n    For implementation details, see  lilliefors_critical_value_simulation.py in\n    the test directory.\n    "
    pvalmethod = string_like(pvalmethod, 'pvalmethod', options=('approx', 'table'))
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    elif x.ndim != 1:
        raise ValueError('Invalid parameter `x`: must be a one-dimensional array-like or a single-column DataFrame')
    nobs = len(x)
    if dist == 'norm':
        z = (x - x.mean()) / x.std(ddof=1)
        test_d = stats.norm.cdf
        lilliefors_table = lilliefors_table_norm
    elif dist == 'exp':
        z = x / x.mean()
        test_d = stats.expon.cdf
        lilliefors_table = lilliefors_table_expon
        pvalmethod = 'table'
    else:
        raise ValueError("Invalid dist parameter, must be 'norm' or 'exp'")
    min_nobs = 4 if dist == 'norm' else 3
    if nobs < min_nobs:
        raise ValueError('Test for distribution {0} requires at least {1} observations'.format(dist, min_nobs))
    d_ks = ksstat(z, test_d, alternative='two_sided')
    if pvalmethod == 'approx':
        pval = pval_lf(d_ks, nobs)
        if pval > 0.1:
            pval = lilliefors_table.prob(d_ks, nobs)
    else:
        pval = lilliefors_table.prob(d_ks, nobs)
    return (d_ks, pval)
lilliefors = kstest_fit
kstest_normal = kstest_fit
kstest_exponential = partial(kstest_fit, dist='exp')