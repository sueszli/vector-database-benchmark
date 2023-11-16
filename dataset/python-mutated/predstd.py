"""Additional functions

prediction standard errors and confidence intervals


A: josef pktd
"""
import numpy as np
from scipy import stats

def atleast_2dcol(x):
    if False:
        return 10
    ' convert array_like to 2d from 1d or 0d\n\n    not tested because not used\n    '
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim == 0:
        x = np.atleast_2d(x)
    elif x.ndim > 0:
        raise ValueError('too many dimensions')
    return x

def wls_prediction_std(res, exog=None, weights=None, alpha=0.05):
    if False:
        while True:
            i = 10
    'calculate standard deviation and confidence interval for prediction\n\n    applies to WLS and OLS, not to general GLS,\n    that is independently but not identically distributed observations\n\n    Parameters\n    ----------\n    res : regression result instance\n        results of WLS or OLS regression required attributes see notes\n    exog : array_like (optional)\n        exogenous variables for points to predict\n    weights : scalar or array_like (optional)\n        weights as defined for WLS (inverse of variance of observation)\n    alpha : float (default: alpha = 0.05)\n        confidence level for two-sided hypothesis\n\n    Returns\n    -------\n    predstd : array_like, 1d\n        standard error of prediction\n        same length as rows of exog\n    interval_l, interval_u : array_like\n        lower und upper confidence bounds\n\n    Notes\n    -----\n    The result instance needs to have at least the following\n    res.model.predict() : predicted values or\n    res.fittedvalues : values used in estimation\n    res.cov_params() : covariance matrix of parameter estimates\n\n    If exog is 1d, then it is interpreted as one observation,\n    i.e. a row vector.\n\n    testing status: not compared with other packages\n\n    References\n    ----------\n\n    Greene p.111 for OLS, extended to WLS by analogy\n\n    '
    covb = res.cov_params()
    if exog is None:
        exog = res.model.exog
        predicted = res.fittedvalues
        if weights is None:
            weights = res.model.weights
    else:
        exog = np.atleast_2d(exog)
        if covb.shape[1] != exog.shape[1]:
            raise ValueError('wrong shape of exog')
        predicted = res.model.predict(res.params, exog)
        if weights is None:
            weights = 1.0
        else:
            weights = np.asarray(weights)
            if weights.size > 1 and len(weights) != exog.shape[0]:
                raise ValueError('weights and exog do not have matching shape')
    predvar = res.mse_resid / weights + (exog * np.dot(covb, exog.T).T).sum(1)
    predstd = np.sqrt(predvar)
    tppf = stats.t.isf(alpha / 2.0, res.df_resid)
    interval_u = predicted + tppf * predstd
    interval_l = predicted - tppf * predstd
    return (predstd, interval_l, interval_u)