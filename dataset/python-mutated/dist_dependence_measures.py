""" Distance dependence measure and the dCov test.

Implementation of Székely et al. (2007) calculation of distance
dependence statistics, including the Distance covariance (dCov) test
for independence of random vectors of arbitrary length.

Author: Ron Itzikovitch

References
----------
.. Szekely, G.J., Rizzo, M.L., and Bakirov, N.K. (2007)
   "Measuring and testing dependence by correlation of distances".
   Annals of Statistics, Vol. 35 No. 6, pp. 2769-2794.

"""
from collections import namedtuple
import warnings
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
DistDependStat = namedtuple('DistDependStat', ['test_statistic', 'distance_correlation', 'distance_covariance', 'dvar_x', 'dvar_y', 'S'])

def distance_covariance_test(x, y, B=None, method='auto'):
    if False:
        while True:
            i = 10
    'The Distance Covariance (dCov) test\n\n    Apply the Distance Covariance (dCov) test of independence to `x` and `y`.\n    This test was introduced in [1]_, and is based on the distance covariance\n    statistic. The test is applicable to random vectors of arbitrary length\n    (see the notes section for more details).\n\n    Parameters\n    ----------\n    x : array_like, 1-D or 2-D\n        If `x` is 1-D than it is assumed to be a vector of observations of a\n        single random variable. If `x` is 2-D than the rows should be\n        observations and the columns are treated as the components of a\n        random vector, i.e., each column represents a different component of\n        the random vector `x`.\n    y : array_like, 1-D or 2-D\n        Same as `x`, but only the number of observation has to match that of\n        `x`. If `y` is 2-D note that the number of columns of `y` (i.e., the\n        number of components in the random vector) does not need to match\n        the number of columns in `x`.\n    B : int, optional, default=`None`\n        The number of iterations to perform when evaluating the null\n        distribution of the test statistic when the `emp` method is\n        applied (see below). if `B` is `None` than as in [1]_ we set\n        `B` to be ``B = 200 + 5000/n``, where `n` is the number of\n        observations.\n    method : {\'auto\', \'emp\', \'asym\'}, optional, default=auto\n        The method by which to obtain the p-value for the test.\n\n        - `auto` : Default method. The number of observations will be used to\n          determine the method.\n        - `emp` : Empirical evaluation of the p-value using permutations of\n          the rows of `y` to obtain the null distribution.\n        - `asym` : An asymptotic approximation of the distribution of the test\n          statistic is used to find the p-value.\n\n    Returns\n    -------\n    test_statistic : float\n        The value of the test statistic used in the test.\n    pval : float\n        The p-value.\n    chosen_method : str\n        The method that was used to obtain the p-value. Mostly relevant when\n        the function is called with `method=\'auto\'`.\n\n    Notes\n    -----\n    The test applies to random vectors of arbitrary dimensions, i.e., `x`\n    can be a 1-D vector of observations for a single random variable while\n    `y` can be a `k` by `n` 2-D array (where `k > 1`). In other words, it\n    is also possible for `x` and `y` to both be 2-D arrays and have the\n    same number of rows (observations) while differing in the number of\n    columns.\n\n    As noted in [1]_ the statistics are sensitive to all types of departures\n    from independence, including nonlinear or nonmonotone dependence\n    structure.\n\n    References\n    ----------\n    .. [1] Szekely, G.J., Rizzo, M.L., and Bakirov, N.K. (2007)\n       "Measuring and testing by correlation of distances".\n       Annals of Statistics, Vol. 35 No. 6, pp. 2769-2794.\n\n    Examples\n    --------\n    >>> from statsmodels.stats.dist_dependence_measures import\n    ... distance_covariance_test\n    >>> data = np.random.rand(1000, 10)\n    >>> x, y = data[:, :3], data[:, 3:]\n    >>> x.shape\n    (1000, 3)\n    >>> y.shape\n    (1000, 7)\n    >>> distance_covariance_test(x, y)\n    (1.0426404792714983, 0.2971148340813543, \'asym\')\n    # (test_statistic, pval, chosen_method)\n\n    '
    (x, y) = _validate_and_tranform_x_and_y(x, y)
    n = x.shape[0]
    stats = distance_statistics(x, y)
    if method == 'auto' and n <= 500 or method == 'emp':
        chosen_method = 'emp'
        (test_statistic, pval) = _empirical_pvalue(x, y, B, n, stats)
    elif method == 'auto' and n > 500 or method == 'asym':
        chosen_method = 'asym'
        (test_statistic, pval) = _asymptotic_pvalue(stats)
    else:
        raise ValueError("Unknown 'method' parameter: {}".format(method))
    if chosen_method == 'emp' and pval in [0, 1]:
        msg = f'p-value was {pval} when using the empirical method. The asymptotic approximation will be used instead'
        warnings.warn(msg, HypothesisTestWarning)
        (_, pval) = _asymptotic_pvalue(stats)
    return (test_statistic, pval, chosen_method)

def _validate_and_tranform_x_and_y(x, y):
    if False:
        print('Hello World!')
    'Ensure `x` and `y` have proper shape and transform/reshape them if\n    required.\n\n    Parameters\n    ----------\n    x : array_like, 1-D or 2-D\n        If `x` is 1-D than it is assumed to be a vector of observations of a\n        single random variable. If `x` is 2-D than the rows should be\n        observations and the columns are treated as the components of a\n        random vector, i.e., each column represents a different component of\n        the random vector `x`.\n    y : array_like, 1-D or 2-D\n        Same as `x`, but only the number of observation has to match that of\n        `x`. If `y` is 2-D note that the number of columns of `y` (i.e., the\n        number of components in the random vector) does not need to match\n        the number of columns in `x`.\n\n    Returns\n    -------\n    x : array_like, 1-D or 2-D\n    y : array_like, 1-D or 2-D\n\n    Raises\n    ------\n    ValueError\n        If `x` and `y` have a different number of observations.\n\n    '
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    if x.shape[0] != y.shape[0]:
        raise ValueError('x and y must have the same number of observations (rows).')
    if len(x.shape) == 1:
        x = x.reshape((x.shape[0], 1))
    if len(y.shape) == 1:
        y = y.reshape((y.shape[0], 1))
    return (x, y)

def _empirical_pvalue(x, y, B, n, stats):
    if False:
        print('Hello World!')
    "Calculate the empirical p-value based on permutations of `y`'s rows\n\n    Parameters\n    ----------\n    x : array_like, 1-D or 2-D\n        If `x` is 1-D than it is assumed to be a vector of observations of a\n        single random variable. If `x` is 2-D than the rows should be\n        observations and the columns are treated as the components of a\n        random vector, i.e., each column represents a different component of\n        the random vector `x`.\n    y : array_like, 1-D or 2-D\n        Same as `x`, but only the number of observation has to match that of\n        `x`. If `y` is 2-D note that the number of columns of `y` (i.e., the\n        number of components in the random vector) does not need to match\n        the number of columns in `x`.\n    B : int\n        The number of iterations when evaluating the null distribution.\n    n : Number of observations found in each of `x` and `y`.\n    stats: namedtuple\n        The result obtained from calling ``distance_statistics(x, y)``.\n\n    Returns\n    -------\n    test_statistic : float\n        The empirical test statistic.\n    pval : float\n        The empirical p-value.\n\n    "
    B = int(B) if B else int(np.floor(200 + 5000 / n))
    empirical_dist = _get_test_statistic_distribution(x, y, B)
    pval = 1 - np.searchsorted(sorted(empirical_dist), stats.test_statistic) / len(empirical_dist)
    test_statistic = stats.test_statistic
    return (test_statistic, pval)

def _asymptotic_pvalue(stats):
    if False:
        while True:
            i = 10
    'Calculate the p-value based on an approximation of the distribution of\n    the test statistic under the null.\n\n    Parameters\n    ----------\n    stats: namedtuple\n        The result obtained from calling ``distance_statistics(x, y)``.\n\n    Returns\n    -------\n    test_statistic : float\n        The test statistic.\n    pval : float\n        The asymptotic p-value.\n\n    '
    test_statistic = np.sqrt(stats.test_statistic / stats.S)
    pval = (1 - norm.cdf(test_statistic)) * 2
    return (test_statistic, pval)

def _get_test_statistic_distribution(x, y, B):
    if False:
        while True:
            i = 10
    '\n    Parameters\n    ----------\n    x : array_like, 1-D or 2-D\n        If `x` is 1-D than it is assumed to be a vector of observations of a\n        single random variable. If `x` is 2-D than the rows should be\n        observations and the columns are treated as the components of a\n        random vector, i.e., each column represents a different component of\n        the random vector `x`.\n    y : array_like, 1-D or 2-D\n        Same as `x`, but only the number of observation has to match that of\n        `x`. If `y` is 2-D note that the number of columns of `y` (i.e., the\n        number of components in the random vector) does not need to match\n        the number of columns in `x`.\n    B : int\n        The number of iterations to perform when evaluating the null\n        distribution.\n\n    Returns\n    -------\n    emp_dist : array_like\n        The empirical distribution of the test statistic.\n\n    '
    y = y.copy()
    emp_dist = np.zeros(B)
    x_dist = squareform(pdist(x, 'euclidean'))
    for i in range(B):
        np.random.shuffle(y)
        emp_dist[i] = distance_statistics(x, y, x_dist=x_dist).test_statistic
    return emp_dist

def distance_statistics(x, y, x_dist=None, y_dist=None):
    if False:
        return 10
    'Calculate various distance dependence statistics.\n\n    Calculate several distance dependence statistics as described in [1]_.\n\n    Parameters\n    ----------\n    x : array_like, 1-D or 2-D\n        If `x` is 1-D than it is assumed to be a vector of observations of a\n        single random variable. If `x` is 2-D than the rows should be\n        observations and the columns are treated as the components of a\n        random vector, i.e., each column represents a different component of\n        the random vector `x`.\n    y : array_like, 1-D or 2-D\n        Same as `x`, but only the number of observation has to match that of\n        `x`. If `y` is 2-D note that the number of columns of `y` (i.e., the\n        number of components in the random vector) does not need to match\n        the number of columns in `x`.\n    x_dist : array_like, 2-D, optional\n        A square 2-D array_like object whose values are the euclidean\n        distances between `x`\'s rows.\n    y_dist : array_like, 2-D, optional\n        A square 2-D array_like object whose values are the euclidean\n        distances between `y`\'s rows.\n\n    Returns\n    -------\n    namedtuple\n        A named tuple of distance dependence statistics (DistDependStat) with\n        the following values:\n\n        - test_statistic : float - The "basic" test statistic (i.e., the one\n          used when the `emp` method is chosen when calling\n          ``distance_covariance_test()``\n        - distance_correlation : float - The distance correlation\n          between `x` and `y`.\n        - distance_covariance : float - The distance covariance of\n          `x` and `y`.\n        - dvar_x : float - The distance variance of `x`.\n        - dvar_y : float - The distance variance of `y`.\n        - S : float - The mean of the euclidean distances in `x` multiplied\n          by those of `y`. Mostly used internally.\n\n    References\n    ----------\n    .. [1] Szekely, G.J., Rizzo, M.L., and Bakirov, N.K. (2007)\n       "Measuring and testing dependence by correlation of distances".\n       Annals of Statistics, Vol. 35 No. 6, pp. 2769-2794.\n\n    Examples\n    --------\n\n    >>> from statsmodels.stats.dist_dependence_measures import\n    ... distance_statistics\n    >>> distance_statistics(np.random.random(1000), np.random.random(1000))\n    DistDependStat(test_statistic=0.07948284320205831,\n    distance_correlation=0.04269511890990793,\n    distance_covariance=0.008915315092696293,\n    dvar_x=0.20719027438266704, dvar_y=0.21044934264957588,\n    S=0.10892061635588891)\n\n    '
    (x, y) = _validate_and_tranform_x_and_y(x, y)
    n = x.shape[0]
    a = x_dist if x_dist is not None else squareform(pdist(x, 'euclidean'))
    b = y_dist if y_dist is not None else squareform(pdist(y, 'euclidean'))
    a_row_means = a.mean(axis=0, keepdims=True)
    b_row_means = b.mean(axis=0, keepdims=True)
    a_col_means = a.mean(axis=1, keepdims=True)
    b_col_means = b.mean(axis=1, keepdims=True)
    a_mean = a.mean()
    b_mean = b.mean()
    A = a - a_row_means - a_col_means + a_mean
    B = b - b_row_means - b_col_means + b_mean
    S = a_mean * b_mean
    dcov = np.sqrt(np.multiply(A, B).mean())
    dvar_x = np.sqrt(np.multiply(A, A).mean())
    dvar_y = np.sqrt(np.multiply(B, B).mean())
    dcor = dcov / np.sqrt(dvar_x * dvar_y)
    test_statistic = n * dcov ** 2
    return DistDependStat(test_statistic=test_statistic, distance_correlation=dcor, distance_covariance=dcov, dvar_x=dvar_x, dvar_y=dvar_y, S=S)

def distance_covariance(x, y):
    if False:
        while True:
            i = 10
    'Distance covariance.\n\n    Calculate the empirical distance covariance as described in [1]_.\n\n    Parameters\n    ----------\n    x : array_like, 1-D or 2-D\n        If `x` is 1-D than it is assumed to be a vector of observations of a\n        single random variable. If `x` is 2-D than the rows should be\n        observations and the columns are treated as the components of a\n        random vector, i.e., each column represents a different component of\n        the random vector `x`.\n    y : array_like, 1-D or 2-D\n        Same as `x`, but only the number of observation has to match that of\n        `x`. If `y` is 2-D note that the number of columns of `y` (i.e., the\n        number of components in the random vector) does not need to match\n        the number of columns in `x`.\n\n    Returns\n    -------\n    float\n        The empirical distance covariance between `x` and `y`.\n\n    References\n    ----------\n    .. [1] Szekely, G.J., Rizzo, M.L., and Bakirov, N.K. (2007)\n       "Measuring and testing dependence by correlation of distances".\n       Annals of Statistics, Vol. 35 No. 6, pp. 2769-2794.\n\n    Examples\n    --------\n\n    >>> from statsmodels.stats.dist_dependence_measures import\n    ... distance_covariance\n    >>> distance_covariance(np.random.random(1000), np.random.random(1000))\n    0.007575063951951362\n\n    '
    return distance_statistics(x, y).distance_covariance

def distance_variance(x):
    if False:
        print('Hello World!')
    'Distance variance.\n\n    Calculate the empirical distance variance as described in [1]_.\n\n    Parameters\n    ----------\n    x : array_like, 1-D or 2-D\n        If `x` is 1-D than it is assumed to be a vector of observations of a\n        single random variable. If `x` is 2-D than the rows should be\n        observations and the columns are treated as the components of a\n        random vector, i.e., each column represents a different component of\n        the random vector `x`.\n\n    Returns\n    -------\n    float\n        The empirical distance variance of `x`.\n\n    References\n    ----------\n    .. [1] Szekely, G.J., Rizzo, M.L., and Bakirov, N.K. (2007)\n       "Measuring and testing dependence by correlation of distances".\n       Annals of Statistics, Vol. 35 No. 6, pp. 2769-2794.\n\n    Examples\n    --------\n\n    >>> from statsmodels.stats.dist_dependence_measures import\n    ... distance_variance\n    >>> distance_variance(np.random.random(1000))\n    0.21732609190659702\n\n    '
    return distance_covariance(x, x)

def distance_correlation(x, y):
    if False:
        return 10
    'Distance correlation.\n\n    Calculate the empirical distance correlation as described in [1]_.\n    This statistic is analogous to product-moment correlation and describes\n    the dependence between `x` and `y`, which are random vectors of\n    arbitrary length. The statistics\' values range between 0 (implies\n    independence) and 1 (implies complete dependence).\n\n    Parameters\n    ----------\n    x : array_like, 1-D or 2-D\n        If `x` is 1-D than it is assumed to be a vector of observations of a\n        single random variable. If `x` is 2-D than the rows should be\n        observations and the columns are treated as the components of a\n        random vector, i.e., each column represents a different component of\n        the random vector `x`.\n    y : array_like, 1-D or 2-D\n        Same as `x`, but only the number of observation has to match that of\n        `x`. If `y` is 2-D note that the number of columns of `y` (i.e., the\n        number of components in the random vector) does not need to match\n        the number of columns in `x`.\n\n    Returns\n    -------\n    float\n        The empirical distance correlation between `x` and `y`.\n\n    References\n    ----------\n    .. [1] Szekely, G.J., Rizzo, M.L., and Bakirov, N.K. (2007)\n       "Measuring and testing dependence by correlation of distances".\n       Annals of Statistics, Vol. 35 No. 6, pp. 2769-2794.\n\n    Examples\n    --------\n\n    >>> from statsmodels.stats.dist_dependence_measures import\n    ... distance_correlation\n    >>> distance_correlation(np.random.random(1000), np.random.random(1000))\n    0.04060497840149489\n\n    '
    return distance_statistics(x, y).distance_correlation