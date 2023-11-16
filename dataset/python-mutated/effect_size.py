"""
Created on Mon Oct  5 12:36:54 2020

Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from scipy import special
from statsmodels.stats.base import Holder

def _noncentrality_chisquare(chi2_stat, df, alpha=0.05):
    if False:
        while True:
            i = 10
    'noncentrality parameter for chi-square statistic\n\n    `nc` is zero-truncated umvue\n\n    Parameters\n    ----------\n    chi2_stat : float\n        Chisquare-statistic, for example from a hypothesis test\n    df : int or float\n        Degrees of freedom\n    alpha : float in (0, 1)\n        Significance level for the confidence interval, covarage is 1 - alpha.\n\n    Returns\n    -------\n    HolderTuple\n        The main attributes are\n\n        - ``nc`` : estimate of noncentrality parameter\n        - ``confint`` : lower and upper bound of confidence interval for `nc``\n\n        Other attributes are estimates for nc by different methods.\n\n    References\n    ----------\n    .. [1] Kubokawa, T., C.P. Robert, and A.K.Md.E. Saleh. 1993. “Estimation of\n        Noncentrality Parameters.”\n        Canadian Journal of Statistics 21 (1): 45–57.\n        https://doi.org/10.2307/3315657.\n\n    .. [2] Li, Qizhai, Junjian Zhang, and Shuai Dai. 2009. “On Estimating the\n        Non-Centrality Parameter of a Chi-Squared Distribution.”\n        Statistics & Probability Letters 79 (1): 98–104.\n        https://doi.org/10.1016/j.spl.2008.07.025.\n\n    '
    alpha_half = alpha / 2
    nc_umvue = chi2_stat - df
    nc = np.maximum(nc_umvue, 0)
    nc_lzd = np.maximum(nc_umvue, chi2_stat / (df + 1))
    nc_krs = np.maximum(nc_umvue, chi2_stat * 2 / (df + 2))
    nc_median = special.chndtrinc(chi2_stat, df, 0.5)
    ci = special.chndtrinc(chi2_stat, df, [1 - alpha_half, alpha_half])
    res = Holder(nc=nc, confint=ci, nc_umvue=nc_umvue, nc_lzd=nc_lzd, nc_krs=nc_krs, nc_median=nc_median, name='Noncentrality for chisquare-distributed random variable')
    return res

def _noncentrality_f(f_stat, df1, df2, alpha=0.05):
    if False:
        while True:
            i = 10
    'noncentrality parameter for f statistic\n\n    `nc` is zero-truncated umvue\n\n    Parameters\n    ----------\n    fstat : float\n        f-statistic, for example from a hypothesis test\n        df : int or float\n        Degrees of freedom\n    alpha : float in (0, 1)\n        Significance level for the confidence interval, covarage is 1 - alpha.\n\n    Returns\n    -------\n    HolderTuple\n        The main attributes are\n\n        - ``nc`` : estimate of noncentrality parameter\n        - ``confint`` : lower and upper bound of confidence interval for `nc``\n\n        Other attributes are estimates for nc by different methods.\n\n    References\n    ----------\n    .. [1] Kubokawa, T., C.P. Robert, and A.K.Md.E. Saleh. 1993. “Estimation of\n       Noncentrality Parameters.” Canadian Journal of Statistics 21 (1): 45–57.\n       https://doi.org/10.2307/3315657.\n    '
    alpha_half = alpha / 2
    x_s = f_stat * df1 / df2
    nc_umvue = (df2 - 2) * x_s - df1
    nc = np.maximum(nc_umvue, 0)
    nc_krs = np.maximum(nc_umvue, x_s * 2 * (df2 - 1) / (df1 + 2))
    nc_median = special.ncfdtrinc(df1, df2, 0.5, f_stat)
    ci = special.ncfdtrinc(df1, df2, [1 - alpha_half, alpha_half], f_stat)
    res = Holder(nc=nc, confint=ci, nc_umvue=nc_umvue, nc_krs=nc_krs, nc_median=nc_median, name='Noncentrality for F-distributed random variable')
    return res

def _noncentrality_t(t_stat, df, alpha=0.05):
    if False:
        while True:
            i = 10
    'noncentrality parameter for t statistic\n\n    Parameters\n    ----------\n    fstat : float\n        f-statistic, for example from a hypothesis test\n        df : int or float\n        Degrees of freedom\n    alpha : float in (0, 1)\n        Significance level for the confidence interval, covarage is 1 - alpha.\n\n    Returns\n    -------\n    HolderTuple\n        The main attributes are\n\n        - ``nc`` : estimate of noncentrality parameter\n        - ``confint`` : lower and upper bound of confidence interval for `nc``\n\n        Other attributes are estimates for nc by different methods.\n\n    References\n    ----------\n    .. [1] Hedges, Larry V. 2016. “Distribution Theory for Glass’s Estimator of\n       Effect Size and Related Estimators:”\n       Journal of Educational Statistics, November.\n       https://doi.org/10.3102/10769986006002107.\n\n    '
    alpha_half = alpha / 2
    gfac = np.exp(special.gammaln(df / 2.0 - 0.5) - special.gammaln(df / 2.0))
    c11 = np.sqrt(df / 2.0) * gfac
    nc = t_stat / c11
    nc_median = special.nctdtrinc(df, 0.5, t_stat)
    ci = special.nctdtrinc(df, [1 - alpha_half, alpha_half], t_stat)
    res = Holder(nc=nc, confint=ci, nc_median=nc_median, name='Noncentrality for t-distributed random variable')
    return res