"""Correlation and Covariance Structures

Created on Sat Dec 17 20:46:05 2011

Author: Josef Perktold
License: BSD-3


Reference
---------
quick reading of some section on mixed effects models in S-plus and of
outline for GEE.

"""
import numpy as np
from statsmodels.regression.linear_model import yule_walker
from statsmodels.stats.moment_helpers import cov2corr

def corr_equi(k_vars, rho):
    if False:
        while True:
            i = 10
    'create equicorrelated correlation matrix with rho on off diagonal\n\n    Parameters\n    ----------\n    k_vars : int\n        number of variables, correlation matrix will be (k_vars, k_vars)\n    rho : float\n        correlation between any two random variables\n\n    Returns\n    -------\n    corr : ndarray (k_vars, k_vars)\n        correlation matrix\n\n    '
    corr = np.empty((k_vars, k_vars))
    corr.fill(rho)
    corr[np.diag_indices_from(corr)] = 1
    return corr

def corr_ar(k_vars, ar):
    if False:
        for i in range(10):
            print('nop')
    'create autoregressive correlation matrix\n\n    This might be MA, not AR, process if used for residual process - check\n\n    Parameters\n    ----------\n    ar : array_like, 1d\n        AR lag-polynomial including 1 for lag 0\n\n\n    '
    from scipy.linalg import toeplitz
    if len(ar) < k_vars:
        ar_ = np.zeros(k_vars)
        ar_[:len(ar)] = ar
        ar = ar_
    return toeplitz(ar)

def corr_arma(k_vars, ar, ma):
    if False:
        print('Hello World!')
    'create arma correlation matrix\n\n    converts arma to autoregressive lag-polynomial with k_var lags\n\n    ar and arma might need to be switched for generating residual process\n\n    Parameters\n    ----------\n    ar : array_like, 1d\n        AR lag-polynomial including 1 for lag 0\n    ma : array_like, 1d\n        MA lag-polynomial\n\n    '
    from scipy.linalg import toeplitz
    from statsmodels.tsa.arima_process import arma2ar
    ar = arma2ar(ar, ma, lags=k_vars)[:k_vars]
    return toeplitz(ar)

def corr2cov(corr, std):
    if False:
        return 10
    'convert correlation matrix to covariance matrix\n\n    Parameters\n    ----------\n    corr : ndarray, (k_vars, k_vars)\n        correlation matrix\n    std : ndarray, (k_vars,) or scalar\n        standard deviation for the vector of random variables. If scalar, then\n        it is assumed that all variables have the same scale given by std.\n\n    '
    if np.size(std) == 1:
        std = std * np.ones(corr.shape[0])
    cov = corr * std[:, None] * std[None, :]
    return cov

def whiten_ar(x, ar_coefs, order):
    if False:
        for i in range(10):
            print('nop')
    '\n    Whiten a series of columns according to an AR(p) covariance structure.\n\n    This drops the initial conditions (Cochran-Orcut ?)\n    Uses loop, so for short ar polynomials only, use lfilter otherwise\n\n    This needs to improve, option on method, full additional to conditional\n\n    Parameters\n    ----------\n    x : array_like, (nobs,) or (nobs, k_vars)\n        The data to be whitened along axis 0\n    ar_coefs : ndarray\n        coefficients of AR lag- polynomial,   TODO: ar or ar_coefs?\n    order : int\n\n    Returns\n    -------\n    x_new : ndarray\n        transformed array\n    '
    rho = ar_coefs
    x = np.array(x, np.float64)
    _x = x.copy()
    if x.ndim == 2:
        rho = rho[:, None]
    for i in range(order):
        _x[i + 1:] = _x[i + 1:] - rho[i] * x[0:-(i + 1)]
    return _x[order:]

def yule_walker_acov(acov, order=1, method='unbiased', df=None, inv=False):
    if False:
        i = 10
        return i + 15
    '\n    Estimate AR(p) parameters from acovf using Yule-Walker equation.\n\n\n    Parameters\n    ----------\n    acov : array_like, 1d\n        auto-covariance\n    order : int, optional\n        The order of the autoregressive process.  Default is 1.\n    inv : bool\n        If inv is True the inverse of R is also returned.  Default is False.\n\n    Returns\n    -------\n    rho : ndarray\n        The estimated autoregressive coefficients\n    sigma\n        TODO\n    Rinv : ndarray\n        inverse of the Toepliz matrix\n    '
    return yule_walker(acov, order=order, method=method, df=df, inv=inv, demean=False)

class ARCovariance:
    """
    experimental class for Covariance of AR process
    classmethod? staticmethods?
    """

    def __init__(self, ar=None, ar_coefs=None, sigma=1.0):
        if False:
            for i in range(10):
                print('nop')
        if ar is not None:
            self.ar = ar
            self.ar_coefs = -ar[1:]
            self.k_lags = len(ar)
        elif ar_coefs is not None:
            self.arcoefs = ar_coefs
            self.ar = np.hstack(([1], -ar_coefs))
            self.k_lags = len(self.ar)

    @classmethod
    def fit(cls, cov, order, **kwds):
        if False:
            while True:
                i = 10
        (rho, sigma) = yule_walker_acov(cov, order=order, **kwds)
        return cls(ar_coefs=rho)

    def whiten(self, x):
        if False:
            i = 10
            return i + 15
        return whiten_ar(x, self.ar_coefs, order=self.order)

    def corr(self, k_vars=None):
        if False:
            for i in range(10):
                print('nop')
        if k_vars is None:
            k_vars = len(self.ar)
        return corr_ar(k_vars, self.ar)

    def cov(self, k_vars=None):
        if False:
            print('Hello World!')
        return cov2corr(self.corr(k_vars=None), self.sigma)