"""
Univariate lowess function, like in R.

References
----------
Hastie, Tibshirani, Friedman. (2009) The Elements of Statistical Learning: Data Mining, Inference, and Prediction, Second Edition: Chapter 6.

Cleveland, W.S. (1979) "Robust Locally Weighted Regression and Smoothing Scatterplots". Journal of the American Statistical Association 74 (368): 829-836.
"""
import numpy as np
from numpy.linalg import lstsq

def lowess(endog, exog, frac=2.0 / 3, it=3):
    if False:
        i = 10
        return i + 15
    '\n    LOWESS (Locally Weighted Scatterplot Smoothing)\n\n    A lowess function that outs smoothed estimates of endog\n    at the given exog values from points (exog, endog)\n\n    Parameters\n    ----------\n    endog : 1-D numpy array\n        The y-values of the observed points\n    exog : 1-D numpy array\n        The x-values of the observed points\n    frac : float\n        Between 0 and 1. The fraction of the data used\n        when estimating each y-value.\n    it : int\n        The number of residual-based reweightings\n        to perform.\n\n    Returns\n    -------\n    out: numpy array\n        A numpy array with two columns. The first column\n        is the sorted x values and the second column the\n        associated estimated y-values.\n\n    Notes\n    -----\n    This lowess function implements the algorithm given in the\n    reference below using local linear estimates.\n\n    Suppose the input data has N points. The algorithm works by\n    estimating the true ``y_i`` by taking the frac*N closest points\n    to ``(x_i,y_i)`` based on their x values and estimating ``y_i``\n    using a weighted linear regression. The weight for ``(x_j,y_j)``\n    is `_lowess_tricube` function applied to ``|x_i-x_j|``.\n\n    If ``iter > 0``, then further weighted local linear regressions\n    are performed, where the weights are the same as above\n    times the `_lowess_bisquare` function of the residuals. Each iteration\n    takes approximately the same amount of time as the original fit,\n    so these iterations are expensive. They are most useful when\n    the noise has extremely heavy tails, such as Cauchy noise.\n    Noise with less heavy-tails, such as t-distributions with ``df > 2``,\n    are less problematic. The weights downgrade the influence of\n    points with large residuals. In the extreme case, points whose\n    residuals are larger than 6 times the median absolute residual\n    are given weight 0.\n\n    Some experimentation is likely required to find a good\n    choice of frac and iter for a particular dataset.\n\n    References\n    ----------\n    Cleveland, W.S. (1979) "Robust Locally Weighted Regression\n    and Smoothing Scatterplots". Journal of the American Statistical\n    Association 74 (368): 829-836.\n\n    Examples\n    --------\n    The below allows a comparison between how different the fits from\n    `lowess` for different values of frac can be.\n\n    >>> import numpy as np\n    >>> import statsmodels.api as sm\n    >>> lowess = sm.nonparametric.lowess\n    >>> x = np.random.uniform(low=-2*np.pi, high=2*np.pi, size=500)\n    >>> y = np.sin(x) + np.random.normal(size=len(x))\n    >>> z = lowess(y, x)\n    >>> w = lowess(y, x, frac=1./3)\n\n    This gives a similar comparison for when it is 0 vs not.\n\n    >>> import scipy.stats as stats\n    >>> x = np.random.uniform(low=-2*np.pi, high=2*np.pi, size=500)\n    >>> y = np.sin(x) + stats.cauchy.rvs(size=len(x))\n    >>> z = lowess(y, x, frac= 1./3, it=0)\n    >>> w = lowess(y, x, frac=1./3)\n    '
    x = exog
    if exog.ndim != 1:
        raise ValueError('exog must be a vector')
    if endog.ndim != 1:
        raise ValueError('endog must be a vector')
    if endog.shape[0] != x.shape[0]:
        raise ValueError('exog and endog must have same length')
    n = exog.shape[0]
    fitted = np.zeros(n)
    k = int(frac * n)
    index_array = np.argsort(exog)
    x_copy = np.array(exog[index_array])
    y_copy = endog[index_array]
    (fitted, weights) = _lowess_initial_fit(x_copy, y_copy, k, n)
    for i in range(it):
        _lowess_robustify_fit(x_copy, y_copy, fitted, weights, k, n)
    out = np.array([x_copy, fitted]).T
    out.shape = (n, 2)
    return out

def _lowess_initial_fit(x_copy, y_copy, k, n):
    if False:
        return 10
    '\n    The initial weighted local linear regression for lowess.\n\n    Parameters\n    ----------\n    x_copy : 1-d ndarray\n        The x-values/exogenous part of the data being smoothed\n    y_copy : 1-d ndarray\n        The y-values/ endogenous part of the data being smoothed\n   k : int\n        The number of data points which affect the linear fit for\n        each estimated point\n    n : int\n        The total number of points\n\n    Returns\n    -------\n    fitted : 1-d ndarray\n        The fitted y-values\n    weights : 2-d ndarray\n        An n by k array. The contribution to the weights in the\n        local linear fit coming from the distances between the\n        x-values\n\n   '
    weights = np.zeros((n, k), dtype=x_copy.dtype)
    nn_indices = [0, k]
    X = np.ones((k, 2))
    fitted = np.zeros(n)
    for i in range(n):
        left_width = x_copy[i] - x_copy[nn_indices[0]]
        right_width = x_copy[nn_indices[1] - 1] - x_copy[i]
        width = max(left_width, right_width)
        _lowess_wt_standardize(weights[i, :], x_copy[nn_indices[0]:nn_indices[1]], x_copy[i], width)
        _lowess_tricube(weights[i, :])
        weights[i, :] = np.sqrt(weights[i, :])
        X[:, 1] = x_copy[nn_indices[0]:nn_indices[1]]
        y_i = weights[i, :] * y_copy[nn_indices[0]:nn_indices[1]]
        beta = lstsq(weights[i, :].reshape(k, 1) * X, y_i, rcond=-1)[0]
        fitted[i] = beta[0] + beta[1] * x_copy[i]
        _lowess_update_nn(x_copy, nn_indices, i + 1)
    return (fitted, weights)

def _lowess_wt_standardize(weights, new_entries, x_copy_i, width):
    if False:
        print('Hello World!')
    "\n    The initial phase of creating the weights.\n    Subtract the current x_i and divide by the width.\n\n    Parameters\n    ----------\n    weights : ndarray\n        The memory where (new_entries - x_copy_i)/width will be placed\n    new_entries : ndarray\n        The x-values of the k closest points to x[i]\n    x_copy_i : float\n        x[i], the i'th point in the (sorted) x values\n    width : float\n        The maximum distance between x[i] and any point in new_entries\n\n    Returns\n    -------\n    Nothing. The modifications are made to weight in place.\n    "
    weights[:] = new_entries
    weights -= x_copy_i
    weights /= width

def _lowess_robustify_fit(x_copy, y_copy, fitted, weights, k, n):
    if False:
        i = 10
        return i + 15
    '\n    Additional weighted local linear regressions, performed if\n    iter>0. They take into account the sizes of the residuals,\n    to eliminate the effect of extreme outliers.\n\n    Parameters\n    ----------\n    x_copy : 1-d ndarray\n        The x-values/exogenous part of the data being smoothed\n    y_copy : 1-d ndarray\n        The y-values/ endogenous part of the data being smoothed\n    fitted : 1-d ndarray\n        The fitted y-values from the previous iteration\n    weights : 2-d ndarray\n        An n by k array. The contribution to the weights in the\n        local linear fit coming from the distances between the\n        x-values\n    k : int\n        The number of data points which affect the linear fit for\n        each estimated point\n    n : int\n        The total number of points\n\n   Returns\n    -------\n    Nothing. The fitted values are modified in place.\n    '
    nn_indices = [0, k]
    X = np.ones((k, 2))
    residual_weights = np.copy(y_copy)
    residual_weights.shape = (n,)
    residual_weights -= fitted
    residual_weights = np.absolute(residual_weights)
    s = np.median(residual_weights)
    residual_weights /= 6 * s
    too_big = residual_weights >= 1
    _lowess_bisquare(residual_weights)
    residual_weights[too_big] = 0
    for i in range(n):
        total_weights = weights[i, :] * np.sqrt(residual_weights[nn_indices[0]:nn_indices[1]])
        X[:, 1] = x_copy[nn_indices[0]:nn_indices[1]]
        y_i = total_weights * y_copy[nn_indices[0]:nn_indices[1]]
        total_weights.shape = (k, 1)
        beta = lstsq(total_weights * X, y_i, rcond=-1)[0]
        fitted[i] = beta[0] + beta[1] * x_copy[i]
        _lowess_update_nn(x_copy, nn_indices, i + 1)

def _lowess_update_nn(x, cur_nn, i):
    if False:
        while True:
            i = 10
    '\n    Update the endpoints of the nearest neighbors to\n    the ith point.\n\n    Parameters\n    ----------\n    x : iterable\n        The sorted points of x-values\n    cur_nn : list of length 2\n        The two current indices between which are the\n        k closest points to x[i]. (The actual value of\n        k is irrelevant for the algorithm.\n    i : int\n        The index of the current value in x for which\n        the k closest points are desired.\n\n    Returns\n    -------\n    Nothing. It modifies cur_nn in place.\n    '
    while True:
        if cur_nn[1] < x.size:
            left_dist = x[i] - x[cur_nn[0]]
            new_right_dist = x[cur_nn[1]] - x[i]
            if new_right_dist < left_dist:
                cur_nn[0] = cur_nn[0] + 1
                cur_nn[1] = cur_nn[1] + 1
            else:
                break
        else:
            break

def _lowess_tricube(t):
    if False:
        i = 10
        return i + 15
    '\n    The _tricube function applied to a numpy array.\n    The tricube function is (1-abs(t)**3)**3.\n\n    Parameters\n    ----------\n    t : ndarray\n        Array the tricube function is applied to elementwise and\n        in-place.\n\n    Returns\n    -------\n    Nothing\n    '
    t[:] = np.absolute(t)
    _lowess_mycube(t)
    t[:] = np.negative(t)
    t += 1
    _lowess_mycube(t)

def _lowess_mycube(t):
    if False:
        while True:
            i = 10
    '\n    Fast matrix cube\n\n    Parameters\n    ----------\n    t : ndarray\n        Array that is cubed, elementwise and in-place\n\n    Returns\n    -------\n    Nothing\n    '
    t2 = t * t
    t *= t2

def _lowess_bisquare(t):
    if False:
        return 10
    '\n    The bisquare function applied to a numpy array.\n    The bisquare function is (1-t**2)**2.\n\n    Parameters\n    ----------\n    t : ndarray\n        array bisquare function is applied to, element-wise and in-place.\n\n    Returns\n    -------\n    Nothing\n    '
    t *= t
    t[:] = np.negative(t)
    t += 1
    t *= t