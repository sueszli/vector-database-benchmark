"""Tools for multivariate analysis


Author : Josef Perktold
License : BSD-3



TODO:

- names of functions, currently just "working titles"

"""
import numpy as np
from statsmodels.tools.tools import Bunch

def partial_project(endog, exog):
    if False:
        return 10
    'helper function to get linear projection or partialling out of variables\n\n    endog variables are projected on exog variables\n\n    Parameters\n    ----------\n    endog : ndarray\n        array of variables where the effect of exog is partialled out.\n    exog : ndarray\n        array of variables on which the endog variables are projected.\n\n    Returns\n    -------\n    res : instance of Bunch with\n\n        - params : OLS parameter estimates from projection of endog on exog\n        - fittedvalues : predicted values of endog given exog\n        - resid : residual of the regression, values of endog with effect of\n          exog partialled out\n\n    Notes\n    -----\n    This is no-frills mainly for internal calculations, no error checking or\n    array conversion is performed, at least for now.\n\n    '
    (x1, x2) = (endog, exog)
    params = np.linalg.pinv(x2).dot(x1)
    predicted = x2.dot(params)
    residual = x1 - predicted
    res = Bunch(params=params, fittedvalues=predicted, resid=residual)
    return res

def cancorr(x1, x2, demean=True, standardize=False):
    if False:
        return 10
    'canonical correlation coefficient beween 2 arrays\n\n    Parameters\n    ----------\n    x1, x2 : ndarrays, 2_D\n        two 2-dimensional data arrays, observations in rows, variables in columns\n    demean : bool\n         If demean is true, then the mean is subtracted from each variable\n    standardize : bool\n         If standardize is true, then each variable is demeaned and divided by\n         its standard deviation. Rescaling does not change the canonical\n         correlation coefficients.\n\n    Returns\n    -------\n    ccorr : ndarray, 1d\n        canonical correlation coefficients, sorted from largest to smallest.\n        Note, that these are the square root of the eigenvalues.\n\n    Notes\n    -----\n    This is a helper function for other statistical functions. It only\n    calculates the canonical correlation coefficients and does not do a full\n    canoncial correlation analysis\n\n    The canonical correlation coefficient is calculated with the generalized\n    matrix inverse and does not raise an exception if one of the data arrays\n    have less than full column rank.\n\n    See Also\n    --------\n    cc_ranktest\n    cc_stats\n    CCA not yet\n\n    '
    if demean or standardize:
        x1 = x1 - x1.mean(0)
        x2 = x2 - x2.mean(0)
    if standardize:
        x1 /= x1.std(0)
        x2 /= x2.std(0)
    t1 = np.linalg.pinv(x1).dot(x2)
    t2 = np.linalg.pinv(x2).dot(x1)
    m = t1.dot(t2)
    cc = np.sqrt(np.linalg.eigvals(m))
    cc.sort()
    return cc[::-1]

def cc_ranktest(x1, x2, demean=True, fullrank=False):
    if False:
        i = 10
        return i + 15
    "rank tests based on smallest canonical correlation coefficients\n\n    Anderson canonical correlations test (LM test) and\n    Cragg-Donald test (Wald test)\n    Assumes homoskedasticity and independent observations, overrejects if\n    there is heteroscedasticity or autocorrelation.\n\n    The Null Hypothesis is that the rank is k - 1, the alternative hypothesis\n    is that the rank is at least k.\n\n\n    Parameters\n    ----------\n    x1, x2 : ndarrays, 2_D\n        two 2-dimensional data arrays, observations in rows, variables in columns\n    demean : bool\n         If demean is true, then the mean is subtracted from each variable.\n    fullrank : bool\n         If true, then only the test that the matrix has full rank is returned.\n         If false, the test for all possible ranks are returned. However, no\n         the p-values are not corrected for the multiplicity of tests.\n\n    Returns\n    -------\n    value : float\n        value of the test statistic\n    p-value : float\n        p-value for the test Null Hypothesis tha the smallest canonical\n        correlation coefficient is zero. based on chi-square distribution\n    df : int\n        degrees of freedom for thechi-square distribution in the hypothesis test\n    ccorr : ndarray, 1d\n        All canonical correlation coefficients sorted from largest to smallest.\n\n    Notes\n    -----\n    Degrees of freedom for the distribution of the test statistic are based on\n    number of columns of x1 and x2 and not on their matrix rank.\n    (I'm not sure yet what the interpretation of the test is if x1 or x2 are of\n    reduced rank.)\n\n    See Also\n    --------\n    cancorr\n    cc_stats\n\n    "
    from scipy import stats
    (nobs1, k1) = x1.shape
    (nobs2, k2) = x2.shape
    cc = cancorr(x1, x2, demean=demean)
    cc2 = cc * cc
    if fullrank:
        df = np.abs(k1 - k2) + 1
        value = nobs1 * cc2[-1]
        w_value = nobs1 * (cc2[-1] / (1.0 - cc2[-1]))
        return (value, stats.chi2.sf(value, df), df, cc, w_value, stats.chi2.sf(w_value, df))
    else:
        r = np.arange(min(k1, k2))[::-1]
        df = (k1 - r) * (k2 - r)
        values = nobs1 * cc2[::-1].cumsum()
        w_values = nobs1 * (cc2 / (1.0 - cc2))[::-1].cumsum()
        return (values, stats.chi2.sf(values, df), df, cc, w_values, stats.chi2.sf(w_values, df))

def cc_stats(x1, x2, demean=True):
    if False:
        return 10
    "MANOVA statistics based on canonical correlation coefficient\n\n    Calculates Pillai's Trace, Wilk's Lambda, Hotelling's Trace and\n    Roy's Largest Root.\n\n    Parameters\n    ----------\n    x1, x2 : ndarrays, 2_D\n        two 2-dimensional data arrays, observations in rows, variables in columns\n    demean : bool\n         If demean is true, then the mean is subtracted from each variable.\n\n    Returns\n    -------\n    res : dict\n        Dictionary containing the test statistics.\n\n    Notes\n    -----\n\n    same as `canon` in Stata\n\n    missing: F-statistics and p-values\n\n    TODO: should return a results class instead\n    produces nans sometimes, singular, perfect correlation of x1, x2 ?\n\n    "
    (nobs1, k1) = x1.shape
    (nobs2, k2) = x2.shape
    cc = cancorr(x1, x2, demean=demean)
    cc2 = cc ** 2
    lam = cc2 / (1 - cc2)
    df_model = k1 * k2
    df_resid = k1 * (nobs1 - k2 - demean)
    s = min(df_model, k1)
    m = 0.5 * (df_model - k1)
    n = 0.5 * (df_resid - k1 - 1)
    df1 = k1 * df_model
    df2 = k2
    pt_value = cc2.sum()
    wl_value = np.product(1 / (1 + lam))
    ht_value = lam.sum()
    rm_value = lam.max()
    res = {}
    res['canonical correlation coefficient'] = cc
    res['eigenvalues'] = lam
    res["Pillai's Trace"] = pt_value
    res["Wilk's Lambda"] = wl_value
    res["Hotelling's Trace"] = ht_value
    res["Roy's Largest Root"] = rm_value
    res['df_resid'] = df_resid
    res['df_m'] = m
    return res