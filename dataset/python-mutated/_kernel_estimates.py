import numpy as np
from statsmodels.duration.hazard_regression import PHReg

def _kernel_cumincidence(time, status, exog, kfunc, freq_weights, dimred=True):
    if False:
        i = 10
        return i + 15
    '\n    Calculates cumulative incidence functions using kernels.\n\n    Parameters\n    ----------\n    time : array_like\n        The observed time values\n    status : array_like\n        The status values.  status == 0 indicates censoring,\n        status == 1, 2, ... are the events.\n    exog : array_like\n        Covariates such that censoring becomes independent of\n        outcome times conditioned on the covariate values.\n    kfunc : function\n        A kernel function\n    freq_weights : array_like\n        Optional frequency weights\n    dimred : bool\n        If True, proportional hazards regression models are used to\n        reduce exog to two columns by predicting overall events and\n        censoring in two separate models.  If False, exog is used\n        directly for calculating kernel weights without dimension\n        reduction.\n    '
    ii = np.argsort(time)
    time = time[ii]
    status = status[ii]
    exog = exog[ii, :]
    nobs = len(time)
    (utime, rtime) = np.unique(time, return_inverse=True)
    ie = np.searchsorted(time, utime, side='right') - 1
    ngrp = int(status.max())
    statusa = (status >= 1).astype(np.float64)
    if freq_weights is not None:
        freq_weights = freq_weights / freq_weights.sum()
    ip = []
    sp = [None] * nobs
    n_risk = [None] * nobs
    kd = [None] * nobs
    for k in range(ngrp):
        status0 = (status == k + 1).astype(np.float64)
        if dimred:
            sfe = PHReg(time, exog, status0).fit()
            fitval_e = sfe.predict().predicted_values
            sfc = PHReg(time, exog, 1 - status0).fit()
            fitval_c = sfc.predict().predicted_values
            exog2d = np.hstack((fitval_e[:, None], fitval_c[:, None]))
            exog2d -= exog2d.mean(0)
            exog2d /= exog2d.std(0)
        else:
            exog2d = exog
        ip0 = 0
        for i in range(nobs):
            if k == 0:
                kd1 = exog2d - exog2d[i, :]
                kd1 = kfunc(kd1)
                kd[i] = kd1
            if k == 0:
                denom = np.cumsum(kd[i][::-1])[::-1]
                num = kd[i] * statusa
                rat = num / denom
                tr = 1e-15
                ii = np.flatnonzero((denom < tr) & (num < tr))
                rat[ii] = 0
                ratc = 1 - rat
                ratc = np.clip(ratc, 1e-10, np.inf)
                lrat = np.log(ratc)
                prat = np.cumsum(lrat)[ie]
                sf = np.exp(prat)
                sp[i] = np.r_[1, sf[:-1]]
                n_risk[i] = denom[ie]
            d0 = np.bincount(rtime, weights=status0 * kd[i], minlength=len(utime))
            ip1 = np.cumsum(sp[i] * d0 / n_risk[i])
            jj = len(ip1) - np.searchsorted(n_risk[i][::-1], 1)
            if jj < len(ip1):
                ip1[jj:] = ip1[jj - 1]
            if freq_weights is None:
                ip0 += ip1
            else:
                ip0 += freq_weights[i] * ip1
        if freq_weights is None:
            ip0 /= nobs
        ip.append(ip0)
    return (utime, ip)

def _kernel_survfunc(time, status, exog, kfunc, freq_weights):
    if False:
        print('Hello World!')
    '\n    Estimate the marginal survival function under dependent censoring.\n\n    Parameters\n    ----------\n    time : array_like\n        The observed times for each subject\n    status : array_like\n        The status for each subject (1 indicates event, 0 indicates\n        censoring)\n    exog : array_like\n        Covariates such that censoring is independent conditional on\n        exog\n    kfunc : function\n        Kernel function\n    freq_weights : array_like\n        Optional frequency weights\n\n    Returns\n    -------\n    probs : array_like\n        The estimated survival probabilities\n    times : array_like\n        The times at which the survival probabilities are estimated\n\n    References\n    ----------\n    Zeng, Donglin 2004. Estimating Marginal Survival Function by\n    Adjusting for Dependent Censoring Using Many Covariates. The\n    Annals of Statistics 32 (4): 1533 55.\n    doi:10.1214/009053604000000508.\n    https://arxiv.org/pdf/math/0409180.pdf\n    '
    sfe = PHReg(time, exog, status).fit()
    fitval_e = sfe.predict().predicted_values
    sfc = PHReg(time, exog, 1 - status).fit()
    fitval_c = sfc.predict().predicted_values
    exog2d = np.hstack((fitval_e[:, None], fitval_c[:, None]))
    n = len(time)
    ixd = np.flatnonzero(status == 1)
    utime = np.unique(time[ixd])
    ii = np.argsort(time)
    time = time[ii]
    status = status[ii]
    exog2d = exog2d[ii, :]
    ie = np.searchsorted(time, utime, side='right') - 1
    if freq_weights is not None:
        freq_weights = freq_weights / freq_weights.sum()
    sprob = 0.0
    for i in range(n):
        kd = exog2d - exog2d[i, :]
        kd = kfunc(kd)
        denom = np.cumsum(kd[::-1])[::-1]
        num = kd * status
        rat = num / denom
        tr = 1e-15
        ii = np.flatnonzero((denom < tr) & (num < tr))
        rat[ii] = 0
        ratc = 1 - rat
        ratc = np.clip(ratc, 1e-12, np.inf)
        lrat = np.log(ratc)
        prat = np.cumsum(lrat)[ie]
        prat = np.exp(prat)
        if freq_weights is None:
            sprob += prat
        else:
            sprob += prat * freq_weights[i]
    if freq_weights is None:
        sprob /= n
    return (sprob, utime)