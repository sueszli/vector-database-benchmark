"""some measures for evaluation of prediction, tests and model selection

Created on Tue Nov 08 15:23:20 2011
Updated on Wed Jun 03 10:42:20 2020

Authors: Josef Perktold & Peter Prescott
License: BSD-3

"""
import numpy as np
from statsmodels.tools.validation import array_like

def mse(x1, x2, axis=0):
    if False:
        print('Hello World!')
    'mean squared error\n\n    Parameters\n    ----------\n    x1, x2 : array_like\n       The performance measure depends on the difference between these two\n       arrays.\n    axis : int\n       axis along which the summary statistic is calculated\n\n    Returns\n    -------\n    mse : ndarray or float\n       mean squared error along given axis.\n\n    Notes\n    -----\n    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.\n    This uses ``numpy.asanyarray`` to convert the input. Whether this is the\n    desired result or not depends on the array subclass, for example\n    numpy matrices will silently produce an incorrect result.\n    '
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.mean((x1 - x2) ** 2, axis=axis)

def rmse(x1, x2, axis=0):
    if False:
        for i in range(10):
            print('nop')
    'root mean squared error\n\n    Parameters\n    ----------\n    x1, x2 : array_like\n       The performance measure depends on the difference between these two\n       arrays.\n    axis : int\n       axis along which the summary statistic is calculated\n\n    Returns\n    -------\n    rmse : ndarray or float\n       root mean squared error along given axis.\n\n    Notes\n    -----\n    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.\n    This uses ``numpy.asanyarray`` to convert the input. Whether this is the\n    desired result or not depends on the array subclass, for example\n    numpy matrices will silently produce an incorrect result.\n    '
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.sqrt(mse(x1, x2, axis=axis))

def rmspe(y, y_hat, axis=0, zeros=np.nan):
    if False:
        return 10
    '\n    Root Mean Squared Percentage Error\n\n    Parameters\n    ----------\n    y : array_like\n      The actual value.\n    y_hat : array_like\n       The predicted value.\n    axis : int\n       Axis along which the summary statistic is calculated\n    zeros : float\n       Value to assign to error where y is zero\n\n    Returns\n    -------\n    rmspe : ndarray or float\n       Root Mean Squared Percentage Error along given axis.\n    '
    y_hat = np.asarray(y_hat)
    y = np.asarray(y)
    error = y - y_hat
    loc = y != 0
    loc = loc.ravel()
    percentage_error = np.full_like(error, zeros)
    percentage_error.flat[loc] = error.flat[loc] / y.flat[loc]
    mspe = np.nanmean(percentage_error ** 2, axis=axis) * 100
    return np.sqrt(mspe)

def maxabs(x1, x2, axis=0):
    if False:
        while True:
            i = 10
    'maximum absolute error\n\n    Parameters\n    ----------\n    x1, x2 : array_like\n       The performance measure depends on the difference between these two\n       arrays.\n    axis : int\n       axis along which the summary statistic is calculated\n\n    Returns\n    -------\n    maxabs : ndarray or float\n       maximum absolute difference along given axis.\n\n    Notes\n    -----\n    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.\n    This uses ``numpy.asanyarray`` to convert the input. Whether this is the\n    desired result or not depends on the array subclass.\n    '
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.max(np.abs(x1 - x2), axis=axis)

def meanabs(x1, x2, axis=0):
    if False:
        print('Hello World!')
    'mean absolute error\n\n    Parameters\n    ----------\n    x1, x2 : array_like\n       The performance measure depends on the difference between these two\n       arrays.\n    axis : int\n       axis along which the summary statistic is calculated\n\n    Returns\n    -------\n    meanabs : ndarray or float\n       mean absolute difference along given axis.\n\n    Notes\n    -----\n    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.\n    This uses ``numpy.asanyarray`` to convert the input. Whether this is the\n    desired result or not depends on the array subclass.\n    '
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.mean(np.abs(x1 - x2), axis=axis)

def medianabs(x1, x2, axis=0):
    if False:
        return 10
    'median absolute error\n\n    Parameters\n    ----------\n    x1, x2 : array_like\n       The performance measure depends on the difference between these two\n       arrays.\n    axis : int\n       axis along which the summary statistic is calculated\n\n    Returns\n    -------\n    medianabs : ndarray or float\n       median absolute difference along given axis.\n\n    Notes\n    -----\n    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.\n    This uses ``numpy.asanyarray`` to convert the input. Whether this is the\n    desired result or not depends on the array subclass.\n    '
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.median(np.abs(x1 - x2), axis=axis)

def bias(x1, x2, axis=0):
    if False:
        for i in range(10):
            print('nop')
    'bias, mean error\n\n    Parameters\n    ----------\n    x1, x2 : array_like\n       The performance measure depends on the difference between these two\n       arrays.\n    axis : int\n       axis along which the summary statistic is calculated\n\n    Returns\n    -------\n    bias : ndarray or float\n       bias, or mean difference along given axis.\n\n    Notes\n    -----\n    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.\n    This uses ``numpy.asanyarray`` to convert the input. Whether this is the\n    desired result or not depends on the array subclass.\n    '
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.mean(x1 - x2, axis=axis)

def medianbias(x1, x2, axis=0):
    if False:
        i = 10
        return i + 15
    'median bias, median error\n\n    Parameters\n    ----------\n    x1, x2 : array_like\n       The performance measure depends on the difference between these two\n       arrays.\n    axis : int\n       axis along which the summary statistic is calculated\n\n    Returns\n    -------\n    medianbias : ndarray or float\n       median bias, or median difference along given axis.\n\n    Notes\n    -----\n    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.\n    This uses ``numpy.asanyarray`` to convert the input. Whether this is the\n    desired result or not depends on the array subclass.\n    '
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.median(x1 - x2, axis=axis)

def vare(x1, x2, ddof=0, axis=0):
    if False:
        print('Hello World!')
    'variance of error\n\n    Parameters\n    ----------\n    x1, x2 : array_like\n       The performance measure depends on the difference between these two\n       arrays.\n    axis : int\n       axis along which the summary statistic is calculated\n\n    Returns\n    -------\n    vare : ndarray or float\n       variance of difference along given axis.\n\n    Notes\n    -----\n    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.\n    This uses ``numpy.asanyarray`` to convert the input. Whether this is the\n    desired result or not depends on the array subclass.\n    '
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.var(x1 - x2, ddof=ddof, axis=axis)

def stde(x1, x2, ddof=0, axis=0):
    if False:
        i = 10
        return i + 15
    'standard deviation of error\n\n    Parameters\n    ----------\n    x1, x2 : array_like\n       The performance measure depends on the difference between these two\n       arrays.\n    axis : int\n       axis along which the summary statistic is calculated\n\n    Returns\n    -------\n    stde : ndarray or float\n       standard deviation of difference along given axis.\n\n    Notes\n    -----\n    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.\n    This uses ``numpy.asanyarray`` to convert the input. Whether this is the\n    desired result or not depends on the array subclass.\n    '
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.std(x1 - x2, ddof=ddof, axis=axis)

def iqr(x1, x2, axis=0):
    if False:
        return 10
    '\n    Interquartile range of error\n\n    Parameters\n    ----------\n    x1 : array_like\n       One of the inputs into the IQR calculation.\n    x2 : array_like\n       The other input into the IQR calculation.\n    axis : {None, int}\n       axis along which the summary statistic is calculated\n\n    Returns\n    -------\n    irq : {float, ndarray}\n       Interquartile range along given axis.\n\n    Notes\n    -----\n    If ``x1`` and ``x2`` have different shapes, then they must broadcast.\n    '
    x1 = array_like(x1, 'x1', dtype=None, ndim=None)
    x2 = array_like(x2, 'x1', dtype=None, ndim=None)
    if axis is None:
        x1 = x1.ravel()
        x2 = x2.ravel()
        axis = 0
    xdiff = np.sort(x1 - x2, axis=axis)
    nobs = x1.shape[axis]
    idx = np.round((nobs - 1) * np.array([0.25, 0.75])).astype(int)
    sl = [slice(None)] * xdiff.ndim
    sl[axis] = idx
    iqr = np.diff(xdiff[tuple(sl)], axis=axis)
    iqr = np.squeeze(iqr)
    return iqr

def aic(llf, nobs, df_modelwc):
    if False:
        return 10
    '\n    Akaike information criterion\n\n    Parameters\n    ----------\n    llf : {float, array_like}\n        value of the loglikelihood\n    nobs : int\n        number of observations\n    df_modelwc : int\n        number of parameters including constant\n\n    Returns\n    -------\n    aic : float\n        information criterion\n\n    References\n    ----------\n    https://en.wikipedia.org/wiki/Akaike_information_criterion\n    '
    return -2.0 * llf + 2.0 * df_modelwc

def aicc(llf, nobs, df_modelwc):
    if False:
        print('Hello World!')
    '\n    Akaike information criterion (AIC) with small sample correction\n\n    Parameters\n    ----------\n    llf : {float, array_like}\n        value of the loglikelihood\n    nobs : int\n        number of observations\n    df_modelwc : int\n        number of parameters including constant\n\n    Returns\n    -------\n    aicc : float\n        information criterion\n\n    References\n    ----------\n    https://en.wikipedia.org/wiki/Akaike_information_criterion#AICc\n\n    Notes\n    -----\n    Returns +inf if the effective degrees of freedom, defined as\n    ``nobs - df_modelwc - 1.0``, is <= 0.\n    '
    dof_eff = nobs - df_modelwc - 1.0
    if dof_eff > 0:
        return -2.0 * llf + 2.0 * df_modelwc * nobs / dof_eff
    else:
        return np.inf

def bic(llf, nobs, df_modelwc):
    if False:
        for i in range(10):
            print('nop')
    '\n    Bayesian information criterion (BIC) or Schwarz criterion\n\n    Parameters\n    ----------\n    llf : {float, array_like}\n        value of the loglikelihood\n    nobs : int\n        number of observations\n    df_modelwc : int\n        number of parameters including constant\n\n    Returns\n    -------\n    bic : float\n        information criterion\n\n    References\n    ----------\n    https://en.wikipedia.org/wiki/Bayesian_information_criterion\n    '
    return -2.0 * llf + np.log(nobs) * df_modelwc

def hqic(llf, nobs, df_modelwc):
    if False:
        i = 10
        return i + 15
    '\n    Hannan-Quinn information criterion (HQC)\n\n    Parameters\n    ----------\n    llf : {float, array_like}\n        value of the loglikelihood\n    nobs : int\n        number of observations\n    df_modelwc : int\n        number of parameters including constant\n\n    Returns\n    -------\n    hqic : float\n        information criterion\n\n    References\n    ----------\n    Wikipedia does not say much\n    '
    return -2.0 * llf + 2 * np.log(np.log(nobs)) * df_modelwc

def aic_sigma(sigma2, nobs, df_modelwc, islog=False):
    if False:
        while True:
            i = 10
    '\n    Akaike information criterion\n\n    Parameters\n    ----------\n    sigma2 : float\n        estimate of the residual variance or determinant of Sigma_hat in the\n        multivariate case. If islog is true, then it is assumed that sigma\n        is already log-ed, for example logdetSigma.\n    nobs : int\n        number of observations\n    df_modelwc : int\n        number of parameters including constant\n\n    Returns\n    -------\n    aic : float\n        information criterion\n\n    Notes\n    -----\n    A constant has been dropped in comparison to the loglikelihood base\n    information criteria. The information criteria should be used to compare\n    only comparable models.\n\n    For example, AIC is defined in terms of the loglikelihood as\n\n    :math:`-2 llf + 2 k`\n\n    in terms of :math:`\\hat{\\sigma}^2`\n\n    :math:`log(\\hat{\\sigma}^2) + 2 k / n`\n\n    in terms of the determinant of :math:`\\hat{\\Sigma}`\n\n    :math:`log(\\|\\hat{\\Sigma}\\|) + 2 k / n`\n\n    Note: In our definition we do not divide by n in the log-likelihood\n    version.\n\n    TODO: Latex math\n\n    reference for example lecture notes by Herman Bierens\n\n    See Also\n    --------\n\n    References\n    ----------\n    https://en.wikipedia.org/wiki/Akaike_information_criterion\n    '
    if not islog:
        sigma2 = np.log(sigma2)
    return sigma2 + aic(0, nobs, df_modelwc) / nobs

def aicc_sigma(sigma2, nobs, df_modelwc, islog=False):
    if False:
        return 10
    '\n    Akaike information criterion (AIC) with small sample correction\n\n    Parameters\n    ----------\n    sigma2 : float\n        estimate of the residual variance or determinant of Sigma_hat in the\n        multivariate case. If islog is true, then it is assumed that sigma\n        is already log-ed, for example logdetSigma.\n    nobs : int\n        number of observations\n    df_modelwc : int\n        number of parameters including constant\n\n    Returns\n    -------\n    aicc : float\n        information criterion\n\n    Notes\n    -----\n    A constant has been dropped in comparison to the loglikelihood base\n    information criteria. These should be used to compare for comparable\n    models.\n\n    References\n    ----------\n    https://en.wikipedia.org/wiki/Akaike_information_criterion#AICc\n    '
    if not islog:
        sigma2 = np.log(sigma2)
    return sigma2 + aicc(0, nobs, df_modelwc) / nobs

def bic_sigma(sigma2, nobs, df_modelwc, islog=False):
    if False:
        print('Hello World!')
    'Bayesian information criterion (BIC) or Schwarz criterion\n\n    Parameters\n    ----------\n    sigma2 : float\n        estimate of the residual variance or determinant of Sigma_hat in the\n        multivariate case. If islog is true, then it is assumed that sigma\n        is already log-ed, for example logdetSigma.\n    nobs : int\n        number of observations\n    df_modelwc : int\n        number of parameters including constant\n\n    Returns\n    -------\n    bic : float\n        information criterion\n\n    Notes\n    -----\n    A constant has been dropped in comparison to the loglikelihood base\n    information criteria. These should be used to compare for comparable\n    models.\n\n    References\n    ----------\n    https://en.wikipedia.org/wiki/Bayesian_information_criterion\n    '
    if not islog:
        sigma2 = np.log(sigma2)
    return sigma2 + bic(0, nobs, df_modelwc) / nobs

def hqic_sigma(sigma2, nobs, df_modelwc, islog=False):
    if False:
        for i in range(10):
            print('nop')
    'Hannan-Quinn information criterion (HQC)\n\n    Parameters\n    ----------\n    sigma2 : float\n        estimate of the residual variance or determinant of Sigma_hat in the\n        multivariate case. If islog is true, then it is assumed that sigma\n        is already log-ed, for example logdetSigma.\n    nobs : int\n        number of observations\n    df_modelwc : int\n        number of parameters including constant\n\n    Returns\n    -------\n    hqic : float\n        information criterion\n\n    Notes\n    -----\n    A constant has been dropped in comparison to the loglikelihood base\n    information criteria. These should be used to compare for comparable\n    models.\n\n    References\n    ----------\n    xxx\n    '
    if not islog:
        sigma2 = np.log(sigma2)
    return sigma2 + hqic(0, nobs, df_modelwc) / nobs
__all__ = [maxabs, meanabs, medianabs, medianbias, mse, rmse, rmspe, stde, vare, aic, aic_sigma, aicc, aicc_sigma, bias, bic, bic_sigma, hqic, hqic_sigma, iqr]