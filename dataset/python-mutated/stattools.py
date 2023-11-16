"""
Statistical tests to be used in conjunction with the models

Notes
-----
These functions have not been formally tested.
"""
from scipy import stats
import numpy as np
from statsmodels.tools.sm_exceptions import ValueWarning

def durbin_watson(resids, axis=0):
    if False:
        return 10
    '\n    Calculates the Durbin-Watson statistic.\n\n    Parameters\n    ----------\n    resids : array_like\n        Data for which to compute the Durbin-Watson statistic. Usually\n        regression model residuals.\n    axis : int, optional\n        Axis to use if data has more than 1 dimension. Default is 0.\n\n    Returns\n    -------\n    dw : float, array_like\n        The Durbin-Watson statistic.\n\n    Notes\n    -----\n    The null hypothesis of the test is that there is no serial correlation\n    in the residuals.\n    The Durbin-Watson test statistic is defined as:\n\n    .. math::\n\n       \\sum_{t=2}^T((e_t - e_{t-1})^2)/\\sum_{t=1}^Te_t^2\n\n    The test statistic is approximately equal to 2*(1-r) where ``r`` is the\n    sample autocorrelation of the residuals. Thus, for r == 0, indicating no\n    serial correlation, the test statistic equals 2. This statistic will\n    always be between 0 and 4. The closer to 0 the statistic, the more\n    evidence for positive serial correlation. The closer to 4, the more\n    evidence for negative serial correlation.\n    '
    resids = np.asarray(resids)
    diff_resids = np.diff(resids, 1, axis=axis)
    dw = np.sum(diff_resids ** 2, axis=axis) / np.sum(resids ** 2, axis=axis)
    return dw

def omni_normtest(resids, axis=0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Omnibus test for normality\n\n    Parameters\n    ----------\n    resid : array_like\n    axis : int, optional\n        Default is 0\n\n    Returns\n    -------\n    Chi^2 score, two-tail probability\n    '
    resids = np.asarray(resids)
    n = resids.shape[axis]
    if n < 8:
        from warnings import warn
        warn('omni_normtest is not valid with less than 8 observations; %i samples were given.' % int(n), ValueWarning)
        return (np.nan, np.nan)
    return stats.normaltest(resids, axis=axis)

def jarque_bera(resids, axis=0):
    if False:
        print('Hello World!')
    '\n    The Jarque-Bera test of normality.\n\n    Parameters\n    ----------\n    resids : array_like\n        Data to test for normality. Usually regression model residuals that\n        are mean 0.\n    axis : int, optional\n        Axis to use if data has more than 1 dimension. Default is 0.\n\n    Returns\n    -------\n    JB : {float, ndarray}\n        The Jarque-Bera test statistic.\n    JBpv : {float, ndarray}\n        The pvalue of the test statistic.\n    skew : {float, ndarray}\n        Estimated skewness of the data.\n    kurtosis : {float, ndarray}\n        Estimated kurtosis of the data.\n\n    Notes\n    -----\n    Each output returned has 1 dimension fewer than data\n\n    The Jarque-Bera test statistic tests the null that the data is normally\n    distributed against an alternative that the data follow some other\n    distribution. The test statistic is based on two moments of the data,\n    the skewness, and the kurtosis, and has an asymptotic :math:`\\chi^2_2`\n    distribution.\n\n    The test statistic is defined\n\n    .. math:: JB = n(S^2/6+(K-3)^2/24)\n\n    where n is the number of data points, S is the sample skewness, and K is\n    the sample kurtosis of the data.\n    '
    resids = np.atleast_1d(np.asarray(resids, dtype=float))
    if resids.size < 2:
        raise ValueError('resids must contain at least 2 elements')
    skew = stats.skew(resids, axis=axis)
    kurtosis = 3 + stats.kurtosis(resids, axis=axis)
    n = resids.shape[axis]
    jb = n / 6.0 * (skew ** 2 + 1 / 4.0 * (kurtosis - 3) ** 2)
    jb_pv = stats.chi2.sf(jb, 2)
    return (jb, jb_pv, skew, kurtosis)

def robust_skewness(y, axis=0):
    if False:
        print('Hello World!')
    '\n    Calculates the four skewness measures in Kim & White\n\n    Parameters\n    ----------\n    y : array_like\n        Data to compute use in the estimator.\n    axis : int or None, optional\n        Axis along which the skewness measures are computed.  If `None`, the\n        entire array is used.\n\n    Returns\n    -------\n    sk1 : ndarray\n          The standard skewness estimator.\n    sk2 : ndarray\n          Skewness estimator based on quartiles.\n    sk3 : ndarray\n          Skewness estimator based on mean-median difference, standardized by\n          absolute deviation.\n    sk4 : ndarray\n          Skewness estimator based on mean-median difference, standardized by\n          standard deviation.\n\n    Notes\n    -----\n    The robust skewness measures are defined\n\n    .. math::\n\n        SK_{2}=\\frac{\\left(q_{.75}-q_{.5}\\right)\n        -\\left(q_{.5}-q_{.25}\\right)}{q_{.75}-q_{.25}}\n\n    .. math::\n\n        SK_{3}=\\frac{\\mu-\\hat{q}_{0.5}}\n        {\\hat{E}\\left[\\left|y-\\hat{\\mu}\\right|\\right]}\n\n    .. math::\n\n        SK_{4}=\\frac{\\mu-\\hat{q}_{0.5}}{\\hat{\\sigma}}\n\n    .. [*] Tae-Hwan Kim and Halbert White, "On more robust estimation of\n       skewness and kurtosis," Finance Research Letters, vol. 1, pp. 56-73,\n       March 2004.\n    '
    if axis is None:
        y = y.ravel()
        axis = 0
    y = np.sort(y, axis)
    (q1, q2, q3) = np.percentile(y, [25.0, 50.0, 75.0], axis=axis)
    mu = y.mean(axis)
    shape = (y.size,)
    if axis is not None:
        shape = list(mu.shape)
        shape.insert(axis, 1)
        shape = tuple(shape)
    mu_b = np.reshape(mu, shape)
    q2_b = np.reshape(q2, shape)
    sigma = np.sqrt(np.mean((y - mu_b) ** 2, axis))
    sk1 = stats.skew(y, axis=axis)
    sk2 = (q1 + q3 - 2.0 * q2) / (q3 - q1)
    sk3 = (mu - q2) / np.mean(abs(y - q2_b), axis=axis)
    sk4 = (mu - q2) / sigma
    return (sk1, sk2, sk3, sk4)

def _kr3(y, alpha=5.0, beta=50.0):
    if False:
        while True:
            i = 10
    '\n    KR3 estimator from Kim & White\n\n    Parameters\n    ----------\n    y : array_like, 1-d\n        Data to compute use in the estimator.\n    alpha : float, optional\n        Lower cut-off for measuring expectation in tail.\n    beta :  float, optional\n        Lower cut-off for measuring expectation in center.\n\n    Returns\n    -------\n    kr3 : float\n        Robust kurtosis estimator based on standardized lower- and upper-tail\n        expected values\n\n    Notes\n    -----\n    .. [*] Tae-Hwan Kim and Halbert White, "On more robust estimation of\n       skewness and kurtosis," Finance Research Letters, vol. 1, pp. 56-73,\n       March 2004.\n    '
    perc = (alpha, 100.0 - alpha, beta, 100.0 - beta)
    (lower_alpha, upper_alpha, lower_beta, upper_beta) = np.percentile(y, perc)
    l_alpha = np.mean(y[y < lower_alpha])
    u_alpha = np.mean(y[y > upper_alpha])
    l_beta = np.mean(y[y < lower_beta])
    u_beta = np.mean(y[y > upper_beta])
    return (u_alpha - l_alpha) / (u_beta - l_beta)

def expected_robust_kurtosis(ab=(5.0, 50.0), dg=(2.5, 25.0)):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculates the expected value of the robust kurtosis measures in Kim and\n    White assuming the data are normally distributed.\n\n    Parameters\n    ----------\n    ab : iterable, optional\n        Contains 100*(alpha, beta) in the kr3 measure where alpha is the tail\n        quantile cut-off for measuring the extreme tail and beta is the central\n        quantile cutoff for the standardization of the measure\n    db : iterable, optional\n        Contains 100*(delta, gamma) in the kr4 measure where delta is the tail\n        quantile for measuring extreme values and gamma is the central quantile\n        used in the the standardization of the measure\n\n    Returns\n    -------\n    ekr : ndarray, 4-element\n        Contains the expected values of the 4 robust kurtosis measures\n\n    Notes\n    -----\n    See `robust_kurtosis` for definitions of the robust kurtosis measures\n    '
    (alpha, beta) = ab
    (delta, gamma) = dg
    expected_value = np.zeros(4)
    ppf = stats.norm.ppf
    pdf = stats.norm.pdf
    (q1, q2, q3, q5, q6, q7) = ppf(np.array((1.0, 2.0, 3.0, 5.0, 6.0, 7.0)) / 8)
    expected_value[0] = 3
    expected_value[1] = (q7 - q5 + (q3 - q1)) / (q6 - q2)
    (q_alpha, q_beta) = ppf(np.array((alpha / 100.0, beta / 100.0)))
    expected_value[2] = 2 * pdf(q_alpha) / alpha / (2 * pdf(q_beta) / beta)
    (q_delta, q_gamma) = ppf(np.array((delta / 100.0, gamma / 100.0)))
    expected_value[3] = -2.0 * q_delta / (-2.0 * q_gamma)
    return expected_value

def robust_kurtosis(y, axis=0, ab=(5.0, 50.0), dg=(2.5, 25.0), excess=True):
    if False:
        while True:
            i = 10
    '\n    Calculates the four kurtosis measures in Kim & White\n\n    Parameters\n    ----------\n    y : array_like\n        Data to compute use in the estimator.\n    axis : int or None, optional\n        Axis along which the kurtosis are computed.  If `None`, the\n        entire array is used.\n    a iterable, optional\n        Contains 100*(alpha, beta) in the kr3 measure where alpha is the tail\n        quantile cut-off for measuring the extreme tail and beta is the central\n        quantile cutoff for the standardization of the measure\n    db : iterable, optional\n        Contains 100*(delta, gamma) in the kr4 measure where delta is the tail\n        quantile for measuring extreme values and gamma is the central quantile\n        used in the the standardization of the measure\n    excess : bool, optional\n        If true (default), computed values are excess of those for a standard\n        normal distribution.\n\n    Returns\n    -------\n    kr1 : ndarray\n          The standard kurtosis estimator.\n    kr2 : ndarray\n          Kurtosis estimator based on octiles.\n    kr3 : ndarray\n          Kurtosis estimators based on exceedance expectations.\n    kr4 : ndarray\n          Kurtosis measure based on the spread between high and low quantiles.\n\n    Notes\n    -----\n    The robust kurtosis measures are defined\n\n    .. math::\n\n        KR_{2}=\\frac{\\left(\\hat{q}_{.875}-\\hat{q}_{.625}\\right)\n        +\\left(\\hat{q}_{.375}-\\hat{q}_{.125}\\right)}\n        {\\hat{q}_{.75}-\\hat{q}_{.25}}\n\n    .. math::\n\n        KR_{3}=\\frac{\\hat{E}\\left(y|y>\\hat{q}_{1-\\alpha}\\right)\n        -\\hat{E}\\left(y|y<\\hat{q}_{\\alpha}\\right)}\n        {\\hat{E}\\left(y|y>\\hat{q}_{1-\\beta}\\right)\n        -\\hat{E}\\left(y|y<\\hat{q}_{\\beta}\\right)}\n\n    .. math::\n\n        KR_{4}=\\frac{\\hat{q}_{1-\\delta}-\\hat{q}_{\\delta}}\n        {\\hat{q}_{1-\\gamma}-\\hat{q}_{\\gamma}}\n\n    where :math:`\\hat{q}_{p}` is the estimated quantile at :math:`p`.\n\n    .. [*] Tae-Hwan Kim and Halbert White, "On more robust estimation of\n       skewness and kurtosis," Finance Research Letters, vol. 1, pp. 56-73,\n       March 2004.\n    '
    if axis is None or (y.squeeze().ndim == 1 and y.ndim != 1):
        y = y.ravel()
        axis = 0
    (alpha, beta) = ab
    (delta, gamma) = dg
    perc = (12.5, 25.0, 37.5, 62.5, 75.0, 87.5, delta, 100.0 - delta, gamma, 100.0 - gamma)
    (e1, e2, e3, e5, e6, e7, fd, f1md, fg, f1mg) = np.percentile(y, perc, axis=axis)
    expected_value = expected_robust_kurtosis(ab, dg) if excess else np.zeros(4)
    kr1 = stats.kurtosis(y, axis, False) - expected_value[0]
    kr2 = (e7 - e5 + (e3 - e1)) / (e6 - e2) - expected_value[1]
    if y.ndim == 1:
        kr3 = _kr3(y, alpha, beta)
    else:
        kr3 = np.apply_along_axis(_kr3, axis, y, alpha, beta)
    kr3 -= expected_value[2]
    kr4 = (f1md - fd) / (f1mg - fg) - expected_value[3]
    return (kr1, kr2, kr3, kr4)

def _medcouple_1d(y):
    if False:
        i = 10
        return i + 15
    '\n    Calculates the medcouple robust measure of skew.\n\n    Parameters\n    ----------\n    y : array_like, 1-d\n        Data to compute use in the estimator.\n\n    Returns\n    -------\n    mc : float\n        The medcouple statistic\n\n    Notes\n    -----\n    The current algorithm requires a O(N**2) memory allocations, and so may\n    not work for very large arrays (N>10000).\n\n    .. [*] M. Hubert and E. Vandervieren, "An adjusted boxplot for skewed\n       distributions" Computational Statistics & Data Analysis, vol. 52, pp.\n       5186-5201, August 2008.\n    '
    y = np.squeeze(np.asarray(y))
    if y.ndim != 1:
        raise ValueError('y must be squeezable to a 1-d array')
    y = np.sort(y)
    n = y.shape[0]
    if n % 2 == 0:
        mf = (y[n // 2 - 1] + y[n // 2]) / 2
    else:
        mf = y[(n - 1) // 2]
    z = y - mf
    lower = z[z <= 0.0]
    upper = z[z >= 0.0]
    upper = upper[:, None]
    standardization = upper - lower
    is_zero = np.logical_and(lower == 0.0, upper == 0.0)
    standardization[is_zero] = np.inf
    spread = upper + lower
    h = spread / standardization
    num_ties = np.sum(lower == 0.0)
    if num_ties:
        replacements = np.ones((num_ties, num_ties)) - np.eye(num_ties)
        replacements -= 2 * np.triu(replacements)
        replacements = np.fliplr(replacements)
        h[:num_ties, -num_ties:] = replacements
    return np.median(h)

def medcouple(y, axis=0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculate the medcouple robust measure of skew.\n\n    Parameters\n    ----------\n    y : array_like\n        Data to compute use in the estimator.\n    axis : {int, None}\n        Axis along which the medcouple statistic is computed.  If `None`, the\n        entire array is used.\n\n    Returns\n    -------\n    mc : ndarray\n        The medcouple statistic with the same shape as `y`, with the specified\n        axis removed.\n\n    Notes\n    -----\n    The current algorithm requires a O(N**2) memory allocations, and so may\n    not work for very large arrays (N>10000).\n\n    .. [*] M. Hubert and E. Vandervieren, "An adjusted boxplot for skewed\n       distributions" Computational Statistics & Data Analysis, vol. 52, pp.\n       5186-5201, August 2008.\n    '
    y = np.asarray(y, dtype=np.double)
    if axis is None:
        return _medcouple_1d(y.ravel())
    return np.apply_along_axis(_medcouple_1d, axis, y)