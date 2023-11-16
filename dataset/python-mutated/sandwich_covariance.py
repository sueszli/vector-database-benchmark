"""Sandwich covariance estimators


Created on Sun Nov 27 14:10:57 2011

Author: Josef Perktold
Author: Skipper Seabold for HCxxx in linear_model.RegressionResults
License: BSD-3

Notes
-----

for calculating it, we have two versions

version 1: use pinv
pinv(x) scale pinv(x)   used currently in linear_model, with scale is
1d (or diagonal matrix)
(x'x)^(-1) x' scale x (x'x)^(-1),  scale in general is (nobs, nobs) so
pretty large general formulas for scale in cluster case are in [4],
which can be found (as of 2017-05-20) at
http://www.tandfonline.com/doi/abs/10.1198/jbes.2010.07136
This paper also has the second version.

version 2:
(x'x)^(-1) S (x'x)^(-1)    with S = x' scale x,    S is (kvar,kvars),
(x'x)^(-1) is available as normalized_covparams.



S = sum (x*u) dot (x*u)' = sum x*u*u'*x'  where sum here can aggregate
over observations or groups. u is regression residual.

x is (nobs, k_var)
u is (nobs, 1)
x*u is (nobs, k_var)


For cluster robust standard errors, we first sum (x*w) over other groups
(including time) and then take the dot product (sum of outer products)

S = sum_g(x*u)' dot sum_g(x*u)
For HAC by clusters, we first sum over groups for each time period, and then
use HAC on the group sums of (x*w).
If we have several groups, we have to sum first over all relevant groups, and
then take the outer product sum. This can be done by summing using indicator
functions or matrices or with explicit loops. Alternatively we calculate
separate covariance matrices for each group, sum them and subtract the
duplicate counted intersection.

Not checked in details yet: degrees of freedom or small sample correction
factors, see (two) references (?)


This is the general case for MLE and GMM also

in MLE     hessian H, outerproduct of jacobian S,   cov_hjjh = HJJH,
which reduces to the above in the linear case, but can be used
generally, e.g. in discrete, and is misnomed in GenericLikelihoodModel

in GMM it's similar but I would have to look up the details, (it comes
out in sandwich form by default, it's in the sandbox), standard Newey
West or similar are on the covariance matrix of the moment conditions

quasi-MLE: MLE with mis-specified model where parameter estimates are
fine (consistent ?) but cov_params needs to be adjusted similar or
same as in sandwiches. (I did not go through any details yet.)

TODO
----
* small sample correction factors, Done for cluster, not yet for HAC
* automatic lag-length selection for Newey-West HAC,
  -> added: nlag = floor[4(T/100)^(2/9)]  Reference: xtscc paper, Newey-West
     note this will not be optimal in the panel context, see Peterson
* HAC should maybe return the chosen nlags
* get consistent notation, varies by paper, S, scale, sigma?
* replace diag(hat_matrix) calculations in cov_hc2, cov_hc3


References
----------
[1] John C. Driscoll and Aart C. Kraay, “Consistent Covariance Matrix Estimation
with Spatially Dependent Panel Data,” Review of Economics and Statistics 80,
no. 4 (1998): 549-560.

[2] Daniel Hoechle, "Robust Standard Errors for Panel Regressions with
Cross-Sectional Dependence", The Stata Journal

[3] Mitchell A. Petersen, “Estimating Standard Errors in Finance Panel Data
Sets: Comparing Approaches,” Review of Financial Studies 22, no. 1
(January 1, 2009): 435 -480.

[4] A. Colin Cameron, Jonah B. Gelbach, and Douglas L. Miller, “Robust Inference
With Multiway Clustering,” Journal of Business and Economic Statistics 29
(April 2011): 238-249.


not used yet:
A.C. Cameron, J.B. Gelbach, and D.L. Miller, “Bootstrap-based improvements
for inference with clustered errors,” The Review of Economics and
Statistics 90, no. 3 (2008): 414–427.

"""
import numpy as np
from statsmodels.tools.grouputils import combine_indices, group_sums
from statsmodels.stats.moment_helpers import se_cov
__all__ = ['cov_cluster', 'cov_cluster_2groups', 'cov_hac', 'cov_nw_panel', 'cov_white_simple', 'cov_hc0', 'cov_hc1', 'cov_hc2', 'cov_hc3', 'se_cov', 'weights_bartlett', 'weights_uniform']
"\n    HC0_se\n        White's (1980) heteroskedasticity robust standard errors.\n        Defined as sqrt(diag(X.T X)^(-1)X.T diag(e_i^(2)) X(X.T X)^(-1)\n        where e_i = resid[i]\n        HC0_se is a property.  It is not evaluated until it is called.\n        When it is called the RegressionResults instance will then have\n        another attribute cov_HC0, which is the full heteroskedasticity\n        consistent covariance matrix and also `het_scale`, which is in\n        this case just resid**2.  HCCM matrices are only appropriate for OLS.\n    HC1_se\n        MacKinnon and White's (1985) alternative heteroskedasticity robust\n        standard errors.\n        Defined as sqrt(diag(n/(n-p)*HC_0)\n        HC1_se is a property.  It is not evaluated until it is called.\n        When it is called the RegressionResults instance will then have\n        another attribute cov_HC1, which is the full HCCM and also `het_scale`,\n        which is in this case n/(n-p)*resid**2.  HCCM matrices are only\n        appropriate for OLS.\n    HC2_se\n        MacKinnon and White's (1985) alternative heteroskedasticity robust\n        standard errors.\n        Defined as (X.T X)^(-1)X.T diag(e_i^(2)/(1-h_ii)) X(X.T X)^(-1)\n        where h_ii = x_i(X.T X)^(-1)x_i.T\n        HC2_se is a property.  It is not evaluated until it is called.\n        When it is called the RegressionResults instance will then have\n        another attribute cov_HC2, which is the full HCCM and also `het_scale`,\n        which is in this case is resid^(2)/(1-h_ii).  HCCM matrices are only\n        appropriate for OLS.\n    HC3_se\n        MacKinnon and White's (1985) alternative heteroskedasticity robust\n        standard errors.\n        Defined as (X.T X)^(-1)X.T diag(e_i^(2)/(1-h_ii)^(2)) X(X.T X)^(-1)\n        where h_ii = x_i(X.T X)^(-1)x_i.T\n        HC3_se is a property.  It is not evaluated until it is called.\n        When it is called the RegressionResults instance will then have\n        another attribute cov_HC3, which is the full HCCM and also `het_scale`,\n        which is in this case is resid^(2)/(1-h_ii)^(2).  HCCM matrices are\n        only appropriate for OLS.\n\n"

def _HCCM(results, scale):
    if False:
        while True:
            i = 10
    "\n    sandwich with pinv(x) * diag(scale) * pinv(x).T\n\n    where pinv(x) = (X'X)^(-1) X\n    and scale is (nobs,)\n    "
    H = np.dot(results.model.pinv_wexog, scale[:, None] * results.model.pinv_wexog.T)
    return H

def cov_hc0(results):
    if False:
        i = 10
        return i + 15
    '\n    See statsmodels.RegressionResults\n    '
    het_scale = results.resid ** 2
    cov_hc0 = _HCCM(results, het_scale)
    return cov_hc0

def cov_hc1(results):
    if False:
        return 10
    '\n    See statsmodels.RegressionResults\n    '
    het_scale = results.nobs / results.df_resid * results.resid ** 2
    cov_hc1 = _HCCM(results, het_scale)
    return cov_hc1

def cov_hc2(results):
    if False:
        for i in range(10):
            print('nop')
    '\n    See statsmodels.RegressionResults\n    '
    h = np.diag(np.dot(results.model.exog, np.dot(results.normalized_cov_params, results.model.exog.T)))
    het_scale = results.resid ** 2 / (1 - h)
    cov_hc2_ = _HCCM(results, het_scale)
    return cov_hc2_

def cov_hc3(results):
    if False:
        i = 10
        return i + 15
    '\n    See statsmodels.RegressionResults\n    '
    h = np.diag(np.dot(results.model.exog, np.dot(results.normalized_cov_params, results.model.exog.T)))
    het_scale = (results.resid / (1 - h)) ** 2
    cov_hc3_ = _HCCM(results, het_scale)
    return cov_hc3_

def _get_sandwich_arrays(results, cov_type=''):
    if False:
        while True:
            i = 10
    'Helper function to get scores from results\n\n    Parameters\n    '
    if isinstance(results, tuple):
        (jac, hessian_inv) = results
        xu = jac = np.asarray(jac)
        hessian_inv = np.asarray(hessian_inv)
    elif hasattr(results, 'model'):
        if hasattr(results, '_results'):
            results = results._results
        if hasattr(results.model, 'jac'):
            xu = results.model.jac(results.params)
            hessian_inv = np.linalg.inv(results.model.hessian(results.params))
        elif hasattr(results.model, 'score_obs'):
            xu = results.model.score_obs(results.params)
            hessian_inv = np.linalg.inv(results.model.hessian(results.params))
        else:
            xu = results.model.wexog * results.wresid[:, None]
            hessian_inv = np.asarray(results.normalized_cov_params)
        if hasattr(results.model, 'freq_weights') and (not cov_type == 'clu'):
            xu /= np.sqrt(np.asarray(results.model.freq_weights)[:, None])
    else:
        raise ValueError('need either tuple of (jac, hessian_inv) or results' + 'instance')
    return (xu, hessian_inv)

def _HCCM1(results, scale):
    if False:
        i = 10
        return i + 15
    "\n    sandwich with pinv(x) * scale * pinv(x).T\n\n    where pinv(x) = (X'X)^(-1) X\n    and scale is (nobs, nobs), or (nobs,) with diagonal matrix diag(scale)\n\n    Parameters\n    ----------\n    results : result instance\n       need to contain regression results, uses results.model.pinv_wexog\n    scale : ndarray (nobs,) or (nobs, nobs)\n       scale matrix, treated as diagonal matrix if scale is one-dimensional\n\n    Returns\n    -------\n    H : ndarray (k_vars, k_vars)\n        robust covariance matrix for the parameter estimates\n\n    "
    if scale.ndim == 1:
        H = np.dot(results.model.pinv_wexog, scale[:, None] * results.model.pinv_wexog.T)
    else:
        H = np.dot(results.model.pinv_wexog, np.dot(scale, results.model.pinv_wexog.T))
    return H

def _HCCM2(hessian_inv, scale):
    if False:
        return 10
    "\n    sandwich with (X'X)^(-1) * scale * (X'X)^(-1)\n\n    scale is (kvars, kvars)\n    this uses results.normalized_cov_params for (X'X)^(-1)\n\n    Parameters\n    ----------\n    results : result instance\n       need to contain regression results, uses results.normalized_cov_params\n    scale : ndarray (k_vars, k_vars)\n       scale matrix\n\n    Returns\n    -------\n    H : ndarray (k_vars, k_vars)\n        robust covariance matrix for the parameter estimates\n\n    "
    if scale.ndim == 1:
        scale = scale[:, None]
    xxi = hessian_inv
    H = np.dot(np.dot(xxi, scale), xxi.T)
    return H

def weights_bartlett(nlags):
    if False:
        return 10
    'Bartlett weights for HAC\n\n    this will be moved to another module\n\n    Parameters\n    ----------\n    nlags : int\n       highest lag in the kernel window, this does not include the zero lag\n\n    Returns\n    -------\n    kernel : ndarray, (nlags+1,)\n        weights for Bartlett kernel\n\n    '
    return 1 - np.arange(nlags + 1) / (nlags + 1.0)

def weights_uniform(nlags):
    if False:
        print('Hello World!')
    'uniform weights for HAC\n\n    this will be moved to another module\n\n    Parameters\n    ----------\n    nlags : int\n       highest lag in the kernel window, this does not include the zero lag\n\n    Returns\n    -------\n    kernel : ndarray, (nlags+1,)\n        weights for uniform kernel\n\n    '
    return np.ones(nlags + 1)
kernel_dict = {'bartlett': weights_bartlett, 'uniform': weights_uniform}

def S_hac_simple(x, nlags=None, weights_func=weights_bartlett):
    if False:
        for i in range(10):
            print('nop')
    'inner covariance matrix for HAC (Newey, West) sandwich\n\n    assumes we have a single time series with zero axis consecutive, equal\n    spaced time periods\n\n\n    Parameters\n    ----------\n    x : ndarray (nobs,) or (nobs, k_var)\n        data, for HAC this is array of x_i * u_i\n    nlags : int or None\n        highest lag to include in kernel window. If None, then\n        nlags = floor(4(T/100)^(2/9)) is used.\n    weights_func : callable\n        weights_func is called with nlags as argument to get the kernel\n        weights. default are Bartlett weights\n\n    Returns\n    -------\n    S : ndarray, (k_vars, k_vars)\n        inner covariance matrix for sandwich\n\n    Notes\n    -----\n    used by cov_hac_simple\n\n    options might change when other kernels besides Bartlett are available.\n\n    '
    if x.ndim == 1:
        x = x[:, None]
    n_periods = x.shape[0]
    if nlags is None:
        nlags = int(np.floor(4 * (n_periods / 100.0) ** (2.0 / 9.0)))
    weights = weights_func(nlags)
    S = weights[0] * np.dot(x.T, x)
    for lag in range(1, nlags + 1):
        s = np.dot(x[lag:].T, x[:-lag])
        S += weights[lag] * (s + s.T)
    return S

def S_white_simple(x):
    if False:
        while True:
            i = 10
    'inner covariance matrix for White heteroscedastistity sandwich\n\n\n    Parameters\n    ----------\n    x : ndarray (nobs,) or (nobs, k_var)\n        data, for HAC this is array of x_i * u_i\n\n    Returns\n    -------\n    S : ndarray, (k_vars, k_vars)\n        inner covariance matrix for sandwich\n\n    Notes\n    -----\n    this is just dot(X.T, X)\n\n    '
    if x.ndim == 1:
        x = x[:, None]
    return np.dot(x.T, x)

def S_hac_groupsum(x, time, nlags=None, weights_func=weights_bartlett):
    if False:
        for i in range(10):
            print('nop')
    'inner covariance matrix for HAC over group sums sandwich\n\n    This assumes we have complete equal spaced time periods.\n    The number of time periods per group need not be the same, but we need\n    at least one observation for each time period\n\n    For a single categorical group only, or a everything else but time\n    dimension. This first aggregates x over groups for each time period, then\n    applies HAC on the sum per period.\n\n    Parameters\n    ----------\n    x : ndarray (nobs,) or (nobs, k_var)\n        data, for HAC this is array of x_i * u_i\n    time : ndarray, (nobs,)\n        timeindes, assumed to be integers range(n_periods)\n    nlags : int or None\n        highest lag to include in kernel window. If None, then\n        nlags = floor[4(T/100)^(2/9)] is used.\n    weights_func : callable\n        weights_func is called with nlags as argument to get the kernel\n        weights. default are Bartlett weights\n\n    Returns\n    -------\n    S : ndarray, (k_vars, k_vars)\n        inner covariance matrix for sandwich\n\n    References\n    ----------\n    Daniel Hoechle, xtscc paper\n    Driscoll and Kraay\n\n    '
    x_group_sums = group_sums(x, time).T
    return S_hac_simple(x_group_sums, nlags=nlags, weights_func=weights_func)

def S_crosssection(x, group):
    if False:
        return 10
    'inner covariance matrix for White on group sums sandwich\n\n    I guess for a single categorical group only,\n    categorical group, can also be the product/intersection of groups\n\n    This is used by cov_cluster and indirectly verified\n\n    '
    x_group_sums = group_sums(x, group).T
    return S_white_simple(x_group_sums)

def cov_crosssection_0(results, group):
    if False:
        print('Hello World!')
    'this one is still wrong, use cov_cluster instead'
    scale = S_crosssection(results.resid[:, None], group)
    scale = np.squeeze(scale)
    cov = _HCCM1(results, scale)
    return cov

def cov_cluster(results, group, use_correction=True):
    if False:
        for i in range(10):
            print('nop')
    'cluster robust covariance matrix\n\n    Calculates sandwich covariance matrix for a single cluster, i.e. grouped\n    variables.\n\n    Parameters\n    ----------\n    results : result instance\n       result of a regression, uses results.model.exog and results.resid\n       TODO: this should use wexog instead\n    use_correction : bool\n       If true (default), then the small sample correction factor is used.\n\n    Returns\n    -------\n    cov : ndarray, (k_vars, k_vars)\n        cluster robust covariance matrix for parameter estimates\n\n    Notes\n    -----\n    same result as Stata in UCLA example and same as Peterson\n\n    '
    (xu, hessian_inv) = _get_sandwich_arrays(results, cov_type='clu')
    if not hasattr(group, 'dtype') or group.dtype != np.dtype('int'):
        (clusters, group) = np.unique(group, return_inverse=True)
    else:
        clusters = np.unique(group)
    scale = S_crosssection(xu, group)
    (nobs, k_params) = xu.shape
    n_groups = len(clusters)
    cov_c = _HCCM2(hessian_inv, scale)
    if use_correction:
        cov_c *= n_groups / (n_groups - 1.0) * ((nobs - 1.0) / float(nobs - k_params))
    return cov_c

def cov_cluster_2groups(results, group, group2=None, use_correction=True):
    if False:
        return 10
    "cluster robust covariance matrix for two groups/clusters\n\n    Parameters\n    ----------\n    results : result instance\n       result of a regression, uses results.model.exog and results.resid\n       TODO: this should use wexog instead\n    use_correction : bool\n       If true (default), then the small sample correction factor is used.\n\n    Returns\n    -------\n    cov_both : ndarray, (k_vars, k_vars)\n        cluster robust covariance matrix for parameter estimates, for both\n        clusters\n    cov_0 : ndarray, (k_vars, k_vars)\n        cluster robust covariance matrix for parameter estimates for first\n        cluster\n    cov_1 : ndarray, (k_vars, k_vars)\n        cluster robust covariance matrix for parameter estimates for second\n        cluster\n\n    Notes\n    -----\n\n    verified against Peterson's table, (4 decimal print precision)\n    "
    if group2 is None:
        if group.ndim != 2 or group.shape[1] != 2:
            raise ValueError('if group2 is not given, then groups needs to be ' + 'an array with two columns')
        group0 = group[:, 0]
        group1 = group[:, 1]
    else:
        group0 = group
        group1 = group2
        group = (group0, group1)
    cov0 = cov_cluster(results, group0, use_correction=use_correction)
    cov1 = cov_cluster(results, group1, use_correction=use_correction)
    cov01 = cov_cluster(results, combine_indices(group)[0], use_correction=use_correction)
    cov_both = cov0 + cov1 - cov01
    return (cov_both, cov0, cov1)

def cov_white_simple(results, use_correction=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    heteroscedasticity robust covariance matrix (White)\n\n    Parameters\n    ----------\n    results : result instance\n       result of a regression, uses results.model.exog and results.resid\n       TODO: this should use wexog instead\n\n    Returns\n    -------\n    cov : ndarray, (k_vars, k_vars)\n        heteroscedasticity robust covariance matrix for parameter estimates\n\n    Notes\n    -----\n    This produces the same result as cov_hc0, and does not include any small\n    sample correction.\n\n    verified (against LinearRegressionResults and Peterson)\n\n    See Also\n    --------\n    cov_hc1, cov_hc2, cov_hc3 : heteroscedasticity robust covariance matrices\n        with small sample corrections\n\n    '
    (xu, hessian_inv) = _get_sandwich_arrays(results)
    sigma = S_white_simple(xu)
    cov_w = _HCCM2(hessian_inv, sigma)
    if use_correction:
        (nobs, k_params) = xu.shape
        cov_w *= nobs / float(nobs - k_params)
    return cov_w

def cov_hac_simple(results, nlags=None, weights_func=weights_bartlett, use_correction=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    heteroscedasticity and autocorrelation robust covariance matrix (Newey-West)\n\n    Assumes we have a single time series with zero axis consecutive, equal\n    spaced time periods\n\n\n    Parameters\n    ----------\n    results : result instance\n       result of a regression, uses results.model.exog and results.resid\n       TODO: this should use wexog instead\n    nlags : int or None\n        highest lag to include in kernel window. If None, then\n        nlags = floor[4(T/100)^(2/9)] is used.\n    weights_func : callable\n        weights_func is called with nlags as argument to get the kernel\n        weights. default are Bartlett weights\n\n    Returns\n    -------\n    cov : ndarray, (k_vars, k_vars)\n        HAC robust covariance matrix for parameter estimates\n\n    Notes\n    -----\n    verified only for nlags=0, which is just White\n    just guessing on correction factor, need reference\n\n    options might change when other kernels besides Bartlett are available.\n\n    '
    (xu, hessian_inv) = _get_sandwich_arrays(results)
    sigma = S_hac_simple(xu, nlags=nlags, weights_func=weights_func)
    cov_hac = _HCCM2(hessian_inv, sigma)
    if use_correction:
        (nobs, k_params) = xu.shape
        cov_hac *= nobs / float(nobs - k_params)
    return cov_hac
cov_hac = cov_hac_simple

def lagged_groups(x, lag, groupidx):
    if False:
        i = 10
        return i + 15
    '\n    assumes sorted by time, groupidx is tuple of start and end values\n    not optimized, just to get a working version, loop over groups\n    '
    out0 = []
    out_lagged = []
    for (l, u) in groupidx:
        if l + lag < u:
            out0.append(x[l + lag:u])
            out_lagged.append(x[l:u - lag])
    if out0 == []:
        raise ValueError('all groups are empty taking lags')
    return (np.vstack(out0), np.vstack(out_lagged))

def S_nw_panel(xw, weights, groupidx):
    if False:
        print('Hello World!')
    'inner covariance matrix for HAC for panel data\n\n    no denominator nobs used\n\n    no reference for this, just accounting for time indices\n    '
    nlags = len(weights) - 1
    S = weights[0] * np.dot(xw.T, xw)
    for lag in range(1, nlags + 1):
        (xw0, xwlag) = lagged_groups(xw, lag, groupidx)
        s = np.dot(xw0.T, xwlag)
        S += weights[lag] * (s + s.T)
    return S

def cov_nw_panel(results, nlags, groupidx, weights_func=weights_bartlett, use_correction='hac'):
    if False:
        while True:
            i = 10
    "Panel HAC robust covariance matrix\n\n    Assumes we have a panel of time series with consecutive, equal spaced time\n    periods. Data is assumed to be in long format with time series of each\n    individual stacked into one array. Panel can be unbalanced.\n\n    Parameters\n    ----------\n    results : result instance\n       result of a regression, uses results.model.exog and results.resid\n       TODO: this should use wexog instead\n    nlags : int or None\n        Highest lag to include in kernel window. Currently, no default\n        because the optimal length will depend on the number of observations\n        per cross-sectional unit.\n    groupidx : list of tuple\n        each tuple should contain the start and end index for an individual.\n        (groupidx might change in future).\n    weights_func : callable\n        weights_func is called with nlags as argument to get the kernel\n        weights. default are Bartlett weights\n    use_correction : 'cluster' or 'hac' or False\n        If False, then no small sample correction is used.\n        If 'cluster' (default), then the same correction as in cov_cluster is\n        used.\n        If 'hac', then the same correction as in single time series, cov_hac\n        is used.\n\n\n    Returns\n    -------\n    cov : ndarray, (k_vars, k_vars)\n        HAC robust covariance matrix for parameter estimates\n\n    Notes\n    -----\n    For nlags=0, this is just White covariance, cov_white.\n    If kernel is uniform, `weights_uniform`, with nlags equal to the number\n    of observations per unit in a balance panel, then cov_cluster and\n    cov_hac_panel are identical.\n\n    Tested against STATA `newey` command with same defaults.\n\n    Options might change when other kernels besides Bartlett and uniform are\n    available.\n\n    "
    if nlags == 0:
        weights = [1, 0]
    else:
        weights = weights_func(nlags)
    (xu, hessian_inv) = _get_sandwich_arrays(results)
    S_hac = S_nw_panel(xu, weights, groupidx)
    cov_hac = _HCCM2(hessian_inv, S_hac)
    if use_correction:
        (nobs, k_params) = xu.shape
        if use_correction == 'hac':
            cov_hac *= nobs / float(nobs - k_params)
        elif use_correction in ['c', 'clu', 'cluster']:
            n_groups = len(groupidx)
            cov_hac *= n_groups / (n_groups - 1.0)
            cov_hac *= (nobs - 1.0) / float(nobs - k_params)
    return cov_hac

def cov_nw_groupsum(results, nlags, time, weights_func=weights_bartlett, use_correction=0):
    if False:
        return 10
    "Driscoll and Kraay Panel robust covariance matrix\n\n    Robust covariance matrix for panel data of Driscoll and Kraay.\n\n    Assumes we have a panel of time series where the time index is available.\n    The time index is assumed to represent equal spaced periods. At least one\n    observation per period is required.\n\n    Parameters\n    ----------\n    results : result instance\n       result of a regression, uses results.model.exog and results.resid\n       TODO: this should use wexog instead\n    nlags : int or None\n        Highest lag to include in kernel window. Currently, no default\n        because the optimal length will depend on the number of observations\n        per cross-sectional unit.\n    time : ndarray of int\n        this should contain the coding for the time period of each observation.\n        time periods should be integers in range(maxT) where maxT is obs of i\n    weights_func : callable\n        weights_func is called with nlags as argument to get the kernel\n        weights. default are Bartlett weights\n    use_correction : 'cluster' or 'hac' or False\n        If False, then no small sample correction is used.\n        If 'hac' (default), then the same correction as in single time series, cov_hac\n        is used.\n        If 'cluster', then the same correction as in cov_cluster is\n        used.\n\n    Returns\n    -------\n    cov : ndarray, (k_vars, k_vars)\n        HAC robust covariance matrix for parameter estimates\n\n    Notes\n    -----\n    Tested against STATA xtscc package, which uses no small sample correction\n\n    This first averages relevant variables for each time period over all\n    individuals/groups, and then applies the same kernel weighted averaging\n    over time as in HAC.\n\n    Warning:\n    In the example with a short panel (few time periods and many individuals)\n    with mainly across individual variation this estimator did not produce\n    reasonable results.\n\n    Options might change when other kernels besides Bartlett and uniform are\n    available.\n\n    References\n    ----------\n    Daniel Hoechle, xtscc paper\n    Driscoll and Kraay\n\n    "
    (xu, hessian_inv) = _get_sandwich_arrays(results)
    S_hac = S_hac_groupsum(xu, time, nlags=nlags, weights_func=weights_func)
    cov_hac = _HCCM2(hessian_inv, S_hac)
    if use_correction:
        (nobs, k_params) = xu.shape
        if use_correction == 'hac':
            cov_hac *= nobs / float(nobs - k_params)
        elif use_correction in ['c', 'cluster']:
            n_groups = len(np.unique(time))
            cov_hac *= n_groups / (n_groups - 1.0)
            cov_hac *= (nobs - 1.0) / float(nobs - k_params)
    return cov_hac