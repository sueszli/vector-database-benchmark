import warnings
import numpy as np
import scipy.stats._stats_py
from . import distributions
from .._lib._bunch import _make_tuple_bunch
from ._stats_pythran import siegelslopes as siegelslopes_pythran
__all__ = ['_find_repeats', 'linregress', 'theilslopes', 'siegelslopes']
LinregressResult = _make_tuple_bunch('LinregressResult', ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr'], extra_field_names=['intercept_stderr'])
TheilslopesResult = _make_tuple_bunch('TheilslopesResult', ['slope', 'intercept', 'low_slope', 'high_slope'])
SiegelslopesResult = _make_tuple_bunch('SiegelslopesResult', ['slope', 'intercept'])

def linregress(x, y=None, alternative='two-sided'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculate a linear least-squares regression for two sets of measurements.\n\n    Parameters\n    ----------\n    x, y : array_like\n        Two sets of measurements.  Both arrays should have the same length.  If\n        only `x` is given (and ``y=None``), then it must be a two-dimensional\n        array where one dimension has length 2.  The two sets of measurements\n        are then found by splitting the array along the length-2 dimension. In\n        the case where ``y=None`` and `x` is a 2x2 array, ``linregress(x)`` is\n        equivalent to ``linregress(x[0], x[1])``.\n    alternative : {\'two-sided\', \'less\', \'greater\'}, optional\n        Defines the alternative hypothesis. Default is \'two-sided\'.\n        The following options are available:\n\n        * \'two-sided\': the slope of the regression line is nonzero\n        * \'less\': the slope of the regression line is less than zero\n        * \'greater\':  the slope of the regression line is greater than zero\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    result : ``LinregressResult`` instance\n        The return value is an object with the following attributes:\n\n        slope : float\n            Slope of the regression line.\n        intercept : float\n            Intercept of the regression line.\n        rvalue : float\n            The Pearson correlation coefficient. The square of ``rvalue``\n            is equal to the coefficient of determination.\n        pvalue : float\n            The p-value for a hypothesis test whose null hypothesis is\n            that the slope is zero, using Wald Test with t-distribution of\n            the test statistic. See `alternative` above for alternative\n            hypotheses.\n        stderr : float\n            Standard error of the estimated slope (gradient), under the\n            assumption of residual normality.\n        intercept_stderr : float\n            Standard error of the estimated intercept, under the assumption\n            of residual normality.\n\n    See Also\n    --------\n    scipy.optimize.curve_fit :\n        Use non-linear least squares to fit a function to data.\n    scipy.optimize.leastsq :\n        Minimize the sum of squares of a set of equations.\n\n    Notes\n    -----\n    Missing values are considered pair-wise: if a value is missing in `x`,\n    the corresponding value in `y` is masked.\n\n    For compatibility with older versions of SciPy, the return value acts\n    like a ``namedtuple`` of length 5, with fields ``slope``, ``intercept``,\n    ``rvalue``, ``pvalue`` and ``stderr``, so one can continue to write::\n\n        slope, intercept, r, p, se = linregress(x, y)\n\n    With that style, however, the standard error of the intercept is not\n    available.  To have access to all the computed values, including the\n    standard error of the intercept, use the return value as an object\n    with attributes, e.g.::\n\n        result = linregress(x, y)\n        print(result.intercept, result.intercept_stderr)\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> import matplotlib.pyplot as plt\n    >>> from scipy import stats\n    >>> rng = np.random.default_rng()\n\n    Generate some data:\n\n    >>> x = rng.random(10)\n    >>> y = 1.6*x + rng.random(10)\n\n    Perform the linear regression:\n\n    >>> res = stats.linregress(x, y)\n\n    Coefficient of determination (R-squared):\n\n    >>> print(f"R-squared: {res.rvalue**2:.6f}")\n    R-squared: 0.717533\n\n    Plot the data along with the fitted line:\n\n    >>> plt.plot(x, y, \'o\', label=\'original data\')\n    >>> plt.plot(x, res.intercept + res.slope*x, \'r\', label=\'fitted line\')\n    >>> plt.legend()\n    >>> plt.show()\n\n    Calculate 95% confidence interval on slope and intercept:\n\n    >>> # Two-sided inverse Students t-distribution\n    >>> # p - probability, df - degrees of freedom\n    >>> from scipy.stats import t\n    >>> tinv = lambda p, df: abs(t.ppf(p/2, df))\n\n    >>> ts = tinv(0.05, len(x)-2)\n    >>> print(f"slope (95%): {res.slope:.6f} +/- {ts*res.stderr:.6f}")\n    slope (95%): 1.453392 +/- 0.743465\n    >>> print(f"intercept (95%): {res.intercept:.6f}"\n    ...       f" +/- {ts*res.intercept_stderr:.6f}")\n    intercept (95%): 0.616950 +/- 0.544475\n\n    '
    TINY = 1e-20
    if y is None:
        x = np.asarray(x)
        if x.shape[0] == 2:
            (x, y) = x
        elif x.shape[1] == 2:
            (x, y) = x.T
        else:
            raise ValueError(f'If only `x` is given as input, it has to be of shape (2, N) or (N, 2); provided shape was {x.shape}.')
    else:
        x = np.asarray(x)
        y = np.asarray(y)
    if x.size == 0 or y.size == 0:
        raise ValueError('Inputs must not be empty.')
    if np.amax(x) == np.amin(x) and len(x) > 1:
        raise ValueError('Cannot calculate a linear regression if all x values are identical')
    n = len(x)
    xmean = np.mean(x, None)
    ymean = np.mean(y, None)
    (ssxm, ssxym, _, ssym) = np.cov(x, y, bias=1).flat
    if ssxm == 0.0 or ssym == 0.0:
        r = 0.0
    else:
        r = ssxym / np.sqrt(ssxm * ssym)
        if r > 1.0:
            r = 1.0
        elif r < -1.0:
            r = -1.0
    slope = ssxym / ssxm
    intercept = ymean - slope * xmean
    if n == 2:
        if y[0] == y[1]:
            prob = 1.0
        else:
            prob = 0.0
        slope_stderr = 0.0
        intercept_stderr = 0.0
    else:
        df = n - 2
        t = r * np.sqrt(df / ((1.0 - r + TINY) * (1.0 + r + TINY)))
        (t, prob) = scipy.stats._stats_py._ttest_finish(df, t, alternative)
        slope_stderr = np.sqrt((1 - r ** 2) * ssym / ssxm / df)
        intercept_stderr = slope_stderr * np.sqrt(ssxm + xmean ** 2)
    return LinregressResult(slope=slope, intercept=intercept, rvalue=r, pvalue=prob, stderr=slope_stderr, intercept_stderr=intercept_stderr)

def theilslopes(y, x=None, alpha=0.95, method='separate'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Computes the Theil-Sen estimator for a set of points (x, y).\n\n    `theilslopes` implements a method for robust linear regression.  It\n    computes the slope as the median of all slopes between paired values.\n\n    Parameters\n    ----------\n    y : array_like\n        Dependent variable.\n    x : array_like or None, optional\n        Independent variable. If None, use ``arange(len(y))`` instead.\n    alpha : float, optional\n        Confidence degree between 0 and 1. Default is 95% confidence.\n        Note that `alpha` is symmetric around 0.5, i.e. both 0.1 and 0.9 are\n        interpreted as "find the 90% confidence interval".\n    method : {\'joint\', \'separate\'}, optional\n        Method to be used for computing estimate for intercept.\n        Following methods are supported,\n\n            * \'joint\': Uses np.median(y - slope * x) as intercept.\n            * \'separate\': Uses np.median(y) - slope * np.median(x)\n                          as intercept.\n\n        The default is \'separate\'.\n\n        .. versionadded:: 1.8.0\n\n    Returns\n    -------\n    result : ``TheilslopesResult`` instance\n        The return value is an object with the following attributes:\n\n        slope : float\n            Theil slope.\n        intercept : float\n            Intercept of the Theil line.\n        low_slope : float\n            Lower bound of the confidence interval on `slope`.\n        high_slope : float\n            Upper bound of the confidence interval on `slope`.\n\n    See Also\n    --------\n    siegelslopes : a similar technique using repeated medians\n\n    Notes\n    -----\n    The implementation of `theilslopes` follows [1]_. The intercept is\n    not defined in [1]_, and here it is defined as ``median(y) -\n    slope*median(x)``, which is given in [3]_. Other definitions of\n    the intercept exist in the literature such as  ``median(y - slope*x)``\n    in [4]_. The approach to compute the intercept can be determined by the\n    parameter ``method``. A confidence interval for the intercept is not\n    given as this question is not addressed in [1]_.\n\n    For compatibility with older versions of SciPy, the return value acts\n    like a ``namedtuple`` of length 4, with fields ``slope``, ``intercept``,\n    ``low_slope``, and ``high_slope``, so one can continue to write::\n\n        slope, intercept, low_slope, high_slope = theilslopes(y, x)\n\n    References\n    ----------\n    .. [1] P.K. Sen, "Estimates of the regression coefficient based on\n           Kendall\'s tau", J. Am. Stat. Assoc., Vol. 63, pp. 1379-1389, 1968.\n    .. [2] H. Theil, "A rank-invariant method of linear and polynomial\n           regression analysis I, II and III",  Nederl. Akad. Wetensch., Proc.\n           53:, pp. 386-392, pp. 521-525, pp. 1397-1412, 1950.\n    .. [3] W.L. Conover, "Practical nonparametric statistics", 2nd ed.,\n           John Wiley and Sons, New York, pp. 493.\n    .. [4] https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy import stats\n    >>> import matplotlib.pyplot as plt\n\n    >>> x = np.linspace(-5, 5, num=150)\n    >>> y = x + np.random.normal(size=x.size)\n    >>> y[11:15] += 10  # add outliers\n    >>> y[-5:] -= 7\n\n    Compute the slope, intercept and 90% confidence interval.  For comparison,\n    also compute the least-squares fit with `linregress`:\n\n    >>> res = stats.theilslopes(y, x, 0.90, method=\'separate\')\n    >>> lsq_res = stats.linregress(x, y)\n\n    Plot the results. The Theil-Sen regression line is shown in red, with the\n    dashed red lines illustrating the confidence interval of the slope (note\n    that the dashed red lines are not the confidence interval of the regression\n    as the confidence interval of the intercept is not included). The green\n    line shows the least-squares fit for comparison.\n\n    >>> fig = plt.figure()\n    >>> ax = fig.add_subplot(111)\n    >>> ax.plot(x, y, \'b.\')\n    >>> ax.plot(x, res[1] + res[0] * x, \'r-\')\n    >>> ax.plot(x, res[1] + res[2] * x, \'r--\')\n    >>> ax.plot(x, res[1] + res[3] * x, \'r--\')\n    >>> ax.plot(x, lsq_res[1] + lsq_res[0] * x, \'g-\')\n    >>> plt.show()\n\n    '
    if method not in ['joint', 'separate']:
        raise ValueError("method must be either 'joint' or 'separate'.'{}' is invalid.".format(method))
    y = np.array(y).flatten()
    if x is None:
        x = np.arange(len(y), dtype=float)
    else:
        x = np.array(x, dtype=float).flatten()
        if len(x) != len(y):
            raise ValueError('Incompatible lengths ! (%s<>%s)' % (len(y), len(x)))
    deltax = x[:, np.newaxis] - x
    deltay = y[:, np.newaxis] - y
    slopes = deltay[deltax > 0] / deltax[deltax > 0]
    if not slopes.size:
        msg = 'All `x` coordinates are identical.'
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    slopes.sort()
    medslope = np.median(slopes)
    if method == 'joint':
        medinter = np.median(y - medslope * x)
    else:
        medinter = np.median(y) - medslope * np.median(x)
    if alpha > 0.5:
        alpha = 1.0 - alpha
    z = distributions.norm.ppf(alpha / 2.0)
    (_, nxreps) = _find_repeats(x)
    (_, nyreps) = _find_repeats(y)
    nt = len(slopes)
    ny = len(y)
    sigsq = 1 / 18.0 * (ny * (ny - 1) * (2 * ny + 5) - sum((k * (k - 1) * (2 * k + 5) for k in nxreps)) - sum((k * (k - 1) * (2 * k + 5) for k in nyreps)))
    try:
        sigma = np.sqrt(sigsq)
        Ru = min(int(np.round((nt - z * sigma) / 2.0)), len(slopes) - 1)
        Rl = max(int(np.round((nt + z * sigma) / 2.0)) - 1, 0)
        delta = slopes[[Rl, Ru]]
    except (ValueError, IndexError):
        delta = (np.nan, np.nan)
    return TheilslopesResult(slope=medslope, intercept=medinter, low_slope=delta[0], high_slope=delta[1])

def _find_repeats(arr):
    if False:
        for i in range(10):
            print('nop')
    if len(arr) == 0:
        return (np.array(0, np.float64), np.array(0, np.intp))
    arr = np.asarray(arr, np.float64).ravel()
    arr.sort()
    change = np.concatenate(([True], arr[1:] != arr[:-1]))
    unique = arr[change]
    change_idx = np.concatenate(np.nonzero(change) + ([arr.size],))
    freq = np.diff(change_idx)
    atleast2 = freq > 1
    return (unique[atleast2], freq[atleast2])

def siegelslopes(y, x=None, method='hierarchical'):
    if False:
        i = 10
        return i + 15
    '\n    Computes the Siegel estimator for a set of points (x, y).\n\n    `siegelslopes` implements a method for robust linear regression\n    using repeated medians (see [1]_) to fit a line to the points (x, y).\n    The method is robust to outliers with an asymptotic breakdown point\n    of 50%.\n\n    Parameters\n    ----------\n    y : array_like\n        Dependent variable.\n    x : array_like or None, optional\n        Independent variable. If None, use ``arange(len(y))`` instead.\n    method : {\'hierarchical\', \'separate\'}\n        If \'hierarchical\', estimate the intercept using the estimated\n        slope ``slope`` (default option).\n        If \'separate\', estimate the intercept independent of the estimated\n        slope. See Notes for details.\n\n    Returns\n    -------\n    result : ``SiegelslopesResult`` instance\n        The return value is an object with the following attributes:\n\n        slope : float\n            Estimate of the slope of the regression line.\n        intercept : float\n            Estimate of the intercept of the regression line.\n\n    See Also\n    --------\n    theilslopes : a similar technique without repeated medians\n\n    Notes\n    -----\n    With ``n = len(y)``, compute ``m_j`` as the median of\n    the slopes from the point ``(x[j], y[j])`` to all other `n-1` points.\n    ``slope`` is then the median of all slopes ``m_j``.\n    Two ways are given to estimate the intercept in [1]_ which can be chosen\n    via the parameter ``method``.\n    The hierarchical approach uses the estimated slope ``slope``\n    and computes ``intercept`` as the median of ``y - slope*x``.\n    The other approach estimates the intercept separately as follows: for\n    each point ``(x[j], y[j])``, compute the intercepts of all the `n-1`\n    lines through the remaining points and take the median ``i_j``.\n    ``intercept`` is the median of the ``i_j``.\n\n    The implementation computes `n` times the median of a vector of size `n`\n    which can be slow for large vectors. There are more efficient algorithms\n    (see [2]_) which are not implemented here.\n\n    For compatibility with older versions of SciPy, the return value acts\n    like a ``namedtuple`` of length 2, with fields ``slope`` and\n    ``intercept``, so one can continue to write::\n\n        slope, intercept = siegelslopes(y, x)\n\n    References\n    ----------\n    .. [1] A. Siegel, "Robust Regression Using Repeated Medians",\n           Biometrika, Vol. 69, pp. 242-244, 1982.\n\n    .. [2] A. Stein and M. Werman, "Finding the repeated median regression\n           line", Proceedings of the Third Annual ACM-SIAM Symposium on\n           Discrete Algorithms, pp. 409-413, 1992.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy import stats\n    >>> import matplotlib.pyplot as plt\n\n    >>> x = np.linspace(-5, 5, num=150)\n    >>> y = x + np.random.normal(size=x.size)\n    >>> y[11:15] += 10  # add outliers\n    >>> y[-5:] -= 7\n\n    Compute the slope and intercept.  For comparison, also compute the\n    least-squares fit with `linregress`:\n\n    >>> res = stats.siegelslopes(y, x)\n    >>> lsq_res = stats.linregress(x, y)\n\n    Plot the results. The Siegel regression line is shown in red. The green\n    line shows the least-squares fit for comparison.\n\n    >>> fig = plt.figure()\n    >>> ax = fig.add_subplot(111)\n    >>> ax.plot(x, y, \'b.\')\n    >>> ax.plot(x, res[1] + res[0] * x, \'r-\')\n    >>> ax.plot(x, lsq_res[1] + lsq_res[0] * x, \'g-\')\n    >>> plt.show()\n\n    '
    if method not in ['hierarchical', 'separate']:
        raise ValueError("method can only be 'hierarchical' or 'separate'")
    y = np.asarray(y).ravel()
    if x is None:
        x = np.arange(len(y), dtype=float)
    else:
        x = np.asarray(x, dtype=float).ravel()
        if len(x) != len(y):
            raise ValueError('Incompatible lengths ! (%s<>%s)' % (len(y), len(x)))
    dtype = np.result_type(x, y, np.float32)
    (y, x) = (y.astype(dtype), x.astype(dtype))
    (medslope, medinter) = siegelslopes_pythran(y, x, method)
    return SiegelslopesResult(slope=medslope, intercept=medinter)