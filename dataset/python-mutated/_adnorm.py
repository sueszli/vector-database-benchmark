"""
Created on Sun Sep 25 21:23:38 2011

Author: Josef Perktold and Scipy developers
License : BSD-3
"""
import numpy as np
from scipy import stats
from statsmodels.tools.validation import array_like, bool_like, int_like

def anderson_statistic(x, dist='norm', fit=True, params=(), axis=0):
    if False:
        return 10
    "\n    Calculate the Anderson-Darling a2 statistic.\n\n    Parameters\n    ----------\n    x : array_like\n        The data to test.\n    dist : {'norm', callable}\n        The assumed distribution under the null of test statistic.\n    fit : bool\n        If True, then the distribution parameters are estimated.\n        Currently only for 1d data x, except in case dist='norm'.\n    params : tuple\n        The optional distribution parameters if fit is False.\n    axis : int\n        If dist is 'norm' or fit is False, then data can be an n-dimensional\n        and axis specifies the axis of a variable.\n\n    Returns\n    -------\n    {float, ndarray}\n        The Anderson-Darling statistic.\n    "
    x = array_like(x, 'x', ndim=None)
    fit = bool_like(fit, 'fit')
    axis = int_like(axis, 'axis')
    y = np.sort(x, axis=axis)
    nobs = y.shape[axis]
    if fit:
        if dist == 'norm':
            xbar = np.expand_dims(np.mean(x, axis=axis), axis)
            s = np.expand_dims(np.std(x, ddof=1, axis=axis), axis)
            w = (y - xbar) / s
            z = stats.norm.cdf(w)
        elif callable(dist):
            params = dist.fit(x)
            z = dist.cdf(y, *params)
        else:
            raise ValueError("dist must be 'norm' or a Callable")
    elif callable(dist):
        z = dist.cdf(y, *params)
    else:
        raise ValueError('if fit is false, then dist must be callable')
    i = np.arange(1, nobs + 1)
    sl1 = [None] * x.ndim
    sl1[axis] = slice(None)
    sl1 = tuple(sl1)
    sl2 = [slice(None)] * x.ndim
    sl2[axis] = slice(None, None, -1)
    sl2 = tuple(sl2)
    s = np.sum((2 * i[sl1] - 1.0) / nobs * (np.log(z) + np.log1p(-z[sl2])), axis=axis)
    a2 = -nobs - s
    return a2

def normal_ad(x, axis=0):
    if False:
        i = 10
        return i + 15
    '\n    Anderson-Darling test for normal distribution unknown mean and variance.\n\n    Parameters\n    ----------\n    x : array_like\n        The data array.\n    axis : int\n        The axis to perform the test along.\n\n    Returns\n    -------\n    ad2 : float\n        Anderson Darling test statistic.\n    pval : float\n        The pvalue for hypothesis that the data comes from a normal\n        distribution with unknown mean and variance.\n\n    See Also\n    --------\n    statsmodels.stats.diagnostic.anderson_statistic\n        The Anderson-Darling a2 statistic.\n    statsmodels.stats.diagnostic.kstest_fit\n        Kolmogorov-Smirnov test with estimated parameters for Normal or\n        Exponential distributions.\n    '
    ad2 = anderson_statistic(x, dist='norm', fit=True, axis=axis)
    n = x.shape[axis]
    ad2a = ad2 * (1 + 0.75 / n + 2.25 / n ** 2)
    if np.size(ad2a) == 1:
        if ad2a >= 0.0 and ad2a < 0.2:
            pval = 1 - np.exp(-13.436 + 101.14 * ad2a - 223.73 * ad2a ** 2)
        elif ad2a < 0.34:
            pval = 1 - np.exp(-8.318 + 42.796 * ad2a - 59.938 * ad2a ** 2)
        elif ad2a < 0.6:
            pval = np.exp(0.9177 - 4.279 * ad2a - 1.38 * ad2a ** 2)
        elif ad2a <= 13:
            pval = np.exp(1.2937 - 5.709 * ad2a + 0.0186 * ad2a ** 2)
        else:
            pval = 0.0
    else:
        bounds = np.array([0.0, 0.2, 0.34, 0.6])
        pval0 = lambda ad2a: np.nan * np.ones_like(ad2a)
        pval1 = lambda ad2a: 1 - np.exp(-13.436 + 101.14 * ad2a - 223.73 * ad2a ** 2)
        pval2 = lambda ad2a: 1 - np.exp(-8.318 + 42.796 * ad2a - 59.938 * ad2a ** 2)
        pval3 = lambda ad2a: np.exp(0.9177 - 4.279 * ad2a - 1.38 * ad2a ** 2)
        pval4 = lambda ad2a: np.exp(1.2937 - 5.709 * ad2a + 0.0186 * ad2a ** 2)
        pvalli = [pval0, pval1, pval2, pval3, pval4]
        idx = np.searchsorted(bounds, ad2a, side='right')
        pval = np.nan * np.ones_like(ad2a)
        for i in range(5):
            mask = idx == i
            pval[mask] = pvalli[i](ad2a[mask])
    return (ad2, pval)