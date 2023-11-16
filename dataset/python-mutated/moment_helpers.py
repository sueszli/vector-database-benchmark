"""helper functions conversion between moments

contains:

* conversion between central and non-central moments, skew, kurtosis and
  cummulants
* cov2corr : convert covariance matrix to correlation matrix


Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from scipy.special import comb

def _convert_to_multidim(x):
    if False:
        for i in range(10):
            print('nop')
    if any([isinstance(x, list), isinstance(x, tuple)]):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    else:
        return x

def _convert_from_multidim(x, totype=list):
    if False:
        i = 10
        return i + 15
    if len(x.shape) < 2:
        return totype(x)
    return x.T

def mc2mnc(mc):
    if False:
        while True:
            i = 10
    'convert central to non-central moments, uses recursive formula\n    optionally adjusts first moment to return mean\n    '
    x = _convert_to_multidim(mc)

    def _local_counts(mc):
        if False:
            while True:
                i = 10
        mean = mc[0]
        mc = [1] + list(mc)
        mc[1] = 0
        mnc = [1, mean]
        for (nn, m) in enumerate(mc[2:]):
            n = nn + 2
            mnc.append(0)
            for k in range(n + 1):
                mnc[n] += comb(n, k, exact=True) * mc[k] * mean ** (n - k)
        return mnc[1:]
    res = np.apply_along_axis(_local_counts, 0, x)
    return _convert_from_multidim(res)

def mnc2mc(mnc, wmean=True):
    if False:
        for i in range(10):
            print('nop')
    'convert non-central to central moments, uses recursive formula\n    optionally adjusts first moment to return mean\n    '
    X = _convert_to_multidim(mnc)

    def _local_counts(mnc):
        if False:
            while True:
                i = 10
        mean = mnc[0]
        mnc = [1] + list(mnc)
        mu = []
        for (n, m) in enumerate(mnc):
            mu.append(0)
            for k in range(n + 1):
                sgn_comb = (-1) ** (n - k) * comb(n, k, exact=True)
                mu[n] += sgn_comb * mnc[k] * mean ** (n - k)
        if wmean:
            mu[1] = mean
        return mu[1:]
    res = np.apply_along_axis(_local_counts, 0, X)
    return _convert_from_multidim(res)

def cum2mc(kappa):
    if False:
        print('Hello World!')
    'convert non-central moments to cumulants\n    recursive formula produces as many cumulants as moments\n\n    References\n    ----------\n    Kenneth Lange: Numerical Analysis for Statisticians, page 40\n    '
    X = _convert_to_multidim(kappa)

    def _local_counts(kappa):
        if False:
            for i in range(10):
                print('nop')
        mc = [1, 0.0]
        kappa0 = kappa[0]
        kappa = [1] + list(kappa)
        for (nn, m) in enumerate(kappa[2:]):
            n = nn + 2
            mc.append(0)
            for k in range(n - 1):
                mc[n] += comb(n - 1, k, exact=True) * kappa[n - k] * mc[k]
        mc[1] = kappa0
        return mc[1:]
    res = np.apply_along_axis(_local_counts, 0, X)
    return _convert_from_multidim(res)

def mnc2cum(mnc):
    if False:
        return 10
    'convert non-central moments to cumulants\n    recursive formula produces as many cumulants as moments\n\n    https://en.wikipedia.org/wiki/Cumulant#Cumulants_and_moments\n    '
    X = _convert_to_multidim(mnc)

    def _local_counts(mnc):
        if False:
            return 10
        mnc = [1] + list(mnc)
        kappa = [1]
        for (nn, m) in enumerate(mnc[1:]):
            n = nn + 1
            kappa.append(m)
            for k in range(1, n):
                num_ways = comb(n - 1, k - 1, exact=True)
                kappa[n] -= num_ways * kappa[k] * mnc[n - k]
        return kappa[1:]
    res = np.apply_along_axis(_local_counts, 0, X)
    return _convert_from_multidim(res)

def mc2cum(mc):
    if False:
        i = 10
        return i + 15
    '\n    just chained because I have still the test case\n    '
    first_step = mc2mnc(mc)
    if isinstance(first_step, np.ndarray):
        first_step = first_step.T
    return mnc2cum(first_step)

def mvsk2mc(args):
    if False:
        return 10
    'convert mean, variance, skew, kurtosis to central moments'
    X = _convert_to_multidim(args)

    def _local_counts(args):
        if False:
            print('Hello World!')
        (mu, sig2, sk, kur) = args
        cnt = [None] * 4
        cnt[0] = mu
        cnt[1] = sig2
        cnt[2] = sk * sig2 ** 1.5
        cnt[3] = (kur + 3.0) * sig2 ** 2.0
        return tuple(cnt)
    res = np.apply_along_axis(_local_counts, 0, X)
    return _convert_from_multidim(res, tuple)

def mvsk2mnc(args):
    if False:
        print('Hello World!')
    'convert mean, variance, skew, kurtosis to non-central moments'
    X = _convert_to_multidim(args)

    def _local_counts(args):
        if False:
            return 10
        (mc, mc2, skew, kurt) = args
        mnc = mc
        mnc2 = mc2 + mc * mc
        mc3 = skew * mc2 ** 1.5
        mnc3 = mc3 + 3 * mc * mc2 + mc ** 3
        mc4 = (kurt + 3.0) * mc2 ** 2.0
        mnc4 = mc4 + 4 * mc * mc3 + 6 * mc * mc * mc2 + mc ** 4
        return (mnc, mnc2, mnc3, mnc4)
    res = np.apply_along_axis(_local_counts, 0, X)
    return _convert_from_multidim(res, tuple)

def mc2mvsk(args):
    if False:
        print('Hello World!')
    'convert central moments to mean, variance, skew, kurtosis'
    X = _convert_to_multidim(args)

    def _local_counts(args):
        if False:
            i = 10
            return i + 15
        (mc, mc2, mc3, mc4) = args
        skew = np.divide(mc3, mc2 ** 1.5)
        kurt = np.divide(mc4, mc2 ** 2.0) - 3.0
        return (mc, mc2, skew, kurt)
    res = np.apply_along_axis(_local_counts, 0, X)
    return _convert_from_multidim(res, tuple)

def mnc2mvsk(args):
    if False:
        print('Hello World!')
    'convert central moments to mean, variance, skew, kurtosis\n    '
    X = _convert_to_multidim(args)

    def _local_counts(args):
        if False:
            i = 10
            return i + 15
        (mnc, mnc2, mnc3, mnc4) = args
        mc = mnc
        mc2 = mnc2 - mnc * mnc
        mc3 = mnc3 - (3 * mc * mc2 + mc ** 3)
        mc4 = mnc4 - (4 * mc * mc3 + 6 * mc * mc * mc2 + mc ** 4)
        return mc2mvsk((mc, mc2, mc3, mc4))
    res = np.apply_along_axis(_local_counts, 0, X)
    return _convert_from_multidim(res, tuple)

def cov2corr(cov, return_std=False):
    if False:
        return 10
    '\n    convert covariance matrix to correlation matrix\n\n    Parameters\n    ----------\n    cov : array_like, 2d\n        covariance matrix, see Notes\n\n    Returns\n    -------\n    corr : ndarray (subclass)\n        correlation matrix\n    return_std : bool\n        If this is true then the standard deviation is also returned.\n        By default only the correlation matrix is returned.\n\n    Notes\n    -----\n    This function does not convert subclasses of ndarrays. This requires that\n    division is defined elementwise. np.ma.array and np.matrix are allowed.\n    '
    cov = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)
    if return_std:
        return (corr, std_)
    else:
        return corr

def corr2cov(corr, std):
    if False:
        return 10
    '\n    convert correlation matrix to covariance matrix given standard deviation\n\n    Parameters\n    ----------\n    corr : array_like, 2d\n        correlation matrix, see Notes\n    std : array_like, 1d\n        standard deviation\n\n    Returns\n    -------\n    cov : ndarray (subclass)\n        covariance matrix\n\n    Notes\n    -----\n    This function does not convert subclasses of ndarrays. This requires\n    that multiplication is defined elementwise. np.ma.array are allowed, but\n    not matrices.\n    '
    corr = np.asanyarray(corr)
    std_ = np.asanyarray(std)
    cov = corr * np.outer(std_, std_)
    return cov

def se_cov(cov):
    if False:
        print('Hello World!')
    '\n    get standard deviation from covariance matrix\n\n    just a shorthand function np.sqrt(np.diag(cov))\n\n    Parameters\n    ----------\n    cov : array_like, square\n        covariance matrix\n\n    Returns\n    -------\n    std : ndarray\n        standard deviation from diagonal of cov\n    '
    return np.sqrt(np.diag(cov))