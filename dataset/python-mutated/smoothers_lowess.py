"""Lowess - wrapper for cythonized extension

Author : Chris Jordan-Squire
Author : Carl Vogel
Author : Josef Perktold

"""
import numpy as np
from ._smoothers_lowess import lowess as _lowess

def lowess(endog, exog, frac=2.0 / 3.0, it=3, delta=0.0, xvals=None, is_sorted=False, missing='drop', return_sorted=True):
    if False:
        while True:
            i = 10
    'LOWESS (Locally Weighted Scatterplot Smoothing)\n\n    A lowess function that outs smoothed estimates of endog\n    at the given exog values from points (exog, endog)\n\n    Parameters\n    ----------\n    endog : 1-D numpy array\n        The y-values of the observed points\n    exog : 1-D numpy array\n        The x-values of the observed points\n    frac : float\n        Between 0 and 1. The fraction of the data used\n        when estimating each y-value.\n    it : int\n        The number of residual-based reweightings\n        to perform.\n    delta : float\n        Distance within which to use linear-interpolation\n        instead of weighted regression.\n    xvals: 1-D numpy array\n        Values of the exogenous variable at which to evaluate the regression.\n        If supplied, cannot use delta.\n    is_sorted : bool\n        If False (default), then the data will be sorted by exog before\n        calculating lowess. If True, then it is assumed that the data is\n        already sorted by exog. If xvals is specified, then it too must be\n        sorted if is_sorted is True.\n    missing : str\n        Available options are \'none\', \'drop\', and \'raise\'. If \'none\', no nan\n        checking is done. If \'drop\', any observations with nans are dropped.\n        If \'raise\', an error is raised. Default is \'drop\'.\n    return_sorted : bool\n        If True (default), then the returned array is sorted by exog and has\n        missing (nan or infinite) observations removed.\n        If False, then the returned array is in the same length and the same\n        sequence of observations as the input array.\n\n    Returns\n    -------\n    out : {ndarray, float}\n        The returned array is two-dimensional if return_sorted is True, and\n        one dimensional if return_sorted is False.\n        If return_sorted is True, then a numpy array with two columns. The\n        first column contains the sorted x (exog) values and the second column\n        the associated estimated y (endog) values.\n        If return_sorted is False, then only the fitted values are returned,\n        and the observations will be in the same order as the input arrays.\n        If xvals is provided, then return_sorted is ignored and the returned\n        array is always one dimensional, containing the y values fitted at\n        the x values provided by xvals.\n\n    Notes\n    -----\n    This lowess function implements the algorithm given in the\n    reference below using local linear estimates.\n\n    Suppose the input data has N points. The algorithm works by\n    estimating the `smooth` y_i by taking the frac*N closest points\n    to (x_i,y_i) based on their x values and estimating y_i\n    using a weighted linear regression. The weight for (x_j,y_j)\n    is tricube function applied to abs(x_i-x_j).\n\n    If it > 1, then further weighted local linear regressions\n    are performed, where the weights are the same as above\n    times the _lowess_bisquare function of the residuals. Each iteration\n    takes approximately the same amount of time as the original fit,\n    so these iterations are expensive. They are most useful when\n    the noise has extremely heavy tails, such as Cauchy noise.\n    Noise with less heavy-tails, such as t-distributions with df>2,\n    are less problematic. The weights downgrade the influence of\n    points with large residuals. In the extreme case, points whose\n    residuals are larger than 6 times the median absolute residual\n    are given weight 0.\n\n    `delta` can be used to save computations. For each `x_i`, regressions\n    are skipped for points closer than `delta`. The next regression is\n    fit for the farthest point within delta of `x_i` and all points in\n    between are estimated by linearly interpolating between the two\n    regression fits.\n\n    Judicious choice of delta can cut computation time considerably\n    for large data (N > 5000). A good choice is ``delta = 0.01 * range(exog)``.\n\n    If `xvals` is provided, the regression is then computed at those points\n    and the fit values are returned. Otherwise, the regression is run\n    at points of `exog`.\n\n    Some experimentation is likely required to find a good\n    choice of `frac` and `iter` for a particular dataset.\n\n    References\n    ----------\n    Cleveland, W.S. (1979) "Robust Locally Weighted Regression\n    and Smoothing Scatterplots". Journal of the American Statistical\n    Association 74 (368): 829-836.\n\n    Examples\n    --------\n    The below allows a comparison between how different the fits from\n    lowess for different values of frac can be.\n\n    >>> import numpy as np\n    >>> import statsmodels.api as sm\n    >>> lowess = sm.nonparametric.lowess\n    >>> x = np.random.uniform(low = -2*np.pi, high = 2*np.pi, size=500)\n    >>> y = np.sin(x) + np.random.normal(size=len(x))\n    >>> z = lowess(y, x)\n    >>> w = lowess(y, x, frac=1./3)\n\n    This gives a similar comparison for when it is 0 vs not.\n\n    >>> import numpy as np\n    >>> import scipy.stats as stats\n    >>> import statsmodels.api as sm\n    >>> lowess = sm.nonparametric.lowess\n    >>> x = np.random.uniform(low = -2*np.pi, high = 2*np.pi, size=500)\n    >>> y = np.sin(x) + stats.cauchy.rvs(size=len(x))\n    >>> z = lowess(y, x, frac= 1./3, it=0)\n    >>> w = lowess(y, x, frac=1./3)\n\n    '
    endog = np.asarray(endog, float)
    exog = np.asarray(exog, float)
    given_xvals = xvals is not None
    if exog.ndim != 1:
        raise ValueError('exog must be a vector')
    if endog.ndim != 1:
        raise ValueError('endog must be a vector')
    if endog.shape[0] != exog.shape[0]:
        raise ValueError('exog and endog must have same length')
    if xvals is not None:
        xvals = np.ascontiguousarray(xvals)
        if xvals.ndim != 1:
            raise ValueError('exog_predict must be a vector')
    if missing in ['drop', 'raise']:
        mask_valid = np.isfinite(exog) & np.isfinite(endog)
        all_valid = np.all(mask_valid)
        if all_valid:
            y = endog
            x = exog
        elif missing == 'drop':
            x = exog[mask_valid]
            y = endog[mask_valid]
        else:
            raise ValueError('nan or inf found in data')
    elif missing == 'none':
        y = endog
        x = exog
        all_valid = True
    else:
        raise ValueError("missing can only be 'none', 'drop' or 'raise'")
    if not is_sorted:
        sort_index = np.argsort(x)
        x = np.array(x[sort_index])
        y = np.array(y[sort_index])
    if not given_xvals:
        xvals = exog
        xvalues = x
        xvals_all_valid = all_valid
        if missing == 'drop':
            xvals_mask_valid = mask_valid
    else:
        if delta != 0.0:
            raise ValueError("Cannot have non-zero 'delta' and 'xvals' values")
        mask_valid = np.isfinite(xvals)
        if missing == 'raise':
            raise ValueError("NaN values in xvals with missing='raise'")
        elif missing == 'drop':
            xvals_mask_valid = mask_valid
        xvalues = xvals
        xvals_all_valid = True if missing == 'none' else np.all(mask_valid)
        return_sorted = False
        if missing in ['drop', 'raise']:
            xvals_mask_valid = np.isfinite(xvals)
            xvals_all_valid = np.all(xvals_mask_valid)
            if xvals_all_valid:
                xvalues = xvals
            elif missing == 'drop':
                xvalues = xvals[xvals_mask_valid]
            else:
                raise ValueError('nan or inf found in xvals')
        if not is_sorted:
            sort_index = np.argsort(xvalues)
            xvalues = np.array(xvalues[sort_index])
        else:
            xvals_all_valid = True
    y = np.ascontiguousarray(y)
    x = np.ascontiguousarray(x)
    if not given_xvals:
        (res, _) = _lowess(y, x, x, np.ones_like(x), frac=frac, it=it, delta=delta, given_xvals=False)
    else:
        if it > 0:
            (_, weights) = _lowess(y, x, x, np.ones_like(x), frac=frac, it=it - 1, delta=delta, given_xvals=False)
        else:
            weights = np.ones_like(x)
        xvalues = np.ascontiguousarray(xvalues, dtype=float)
        (res, _) = _lowess(y, x, xvalues, weights, frac=frac, it=0, delta=delta, given_xvals=True)
    (_, yfitted) = res.T
    if return_sorted:
        return res
    else:
        if not is_sorted:
            yfitted_ = np.empty_like(xvalues)
            yfitted_.fill(np.nan)
            yfitted_[sort_index] = yfitted
            yfitted = yfitted_
        else:
            yfitted = yfitted
        if not xvals_all_valid:
            yfitted_ = np.empty_like(xvals)
            yfitted_.fill(np.nan)
            yfitted_[xvals_mask_valid] = yfitted
            yfitted = yfitted_
        return yfitted