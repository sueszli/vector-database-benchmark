"""
Additional statistics functions with support for masked arrays.

"""
__all__ = ['compare_medians_ms', 'hdquantiles', 'hdmedian', 'hdquantiles_sd', 'idealfourths', 'median_cihs', 'mjci', 'mquantiles_cimj', 'rsh', 'trimmed_mean_ci']
import numpy as np
from numpy import float64, ndarray
import numpy.ma as ma
from numpy.ma import MaskedArray
from . import _mstats_basic as mstats
from scipy.stats.distributions import norm, beta, t, binom

def hdquantiles(data, prob=list([0.25, 0.5, 0.75]), axis=None, var=False):
    if False:
        i = 10
        return i + 15
    '\n    Computes quantile estimates with the Harrell-Davis method.\n\n    The quantile estimates are calculated as a weighted linear combination\n    of order statistics.\n\n    Parameters\n    ----------\n    data : array_like\n        Data array.\n    prob : sequence, optional\n        Sequence of probabilities at which to compute the quantiles.\n    axis : int or None, optional\n        Axis along which to compute the quantiles. If None, use a flattened\n        array.\n    var : bool, optional\n        Whether to return the variance of the estimate.\n\n    Returns\n    -------\n    hdquantiles : MaskedArray\n        A (p,) array of quantiles (if `var` is False), or a (2,p) array of\n        quantiles and variances (if `var` is True), where ``p`` is the\n        number of quantiles.\n\n    See Also\n    --------\n    hdquantiles_sd\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.stats.mstats import hdquantiles\n    >>>\n    >>> # Sample data\n    >>> data = np.array([1.2, 2.5, 3.7, 4.0, 5.1, 6.3, 7.0, 8.2, 9.4])\n    >>>\n    >>> # Probabilities at which to compute quantiles\n    >>> probabilities = [0.25, 0.5, 0.75]\n    >>>\n    >>> # Compute Harrell-Davis quantile estimates\n    >>> quantile_estimates = hdquantiles(data, prob=probabilities)\n    >>>\n    >>> # Display the quantile estimates\n    >>> for i, quantile in enumerate(probabilities):\n    ...     print(f"{int(quantile * 100)}th percentile: {quantile_estimates[i]}")\n    25th percentile: 3.1505820231763066\n    50th percentile: 5.194344084883956\n    75th percentile: 7.430626414674935\n\n    '

    def _hd_1D(data, prob, var):
        if False:
            print('Hello World!')
        'Computes the HD quantiles for a 1D array. Returns nan for invalid data.'
        xsorted = np.squeeze(np.sort(data.compressed().view(ndarray)))
        n = xsorted.size
        hd = np.empty((2, len(prob)), float64)
        if n < 2:
            hd.flat = np.nan
            if var:
                return hd
            return hd[0]
        v = np.arange(n + 1) / float(n)
        betacdf = beta.cdf
        for (i, p) in enumerate(prob):
            _w = betacdf(v, (n + 1) * p, (n + 1) * (1 - p))
            w = _w[1:] - _w[:-1]
            hd_mean = np.dot(w, xsorted)
            hd[0, i] = hd_mean
            hd[1, i] = np.dot(w, (xsorted - hd_mean) ** 2)
        hd[0, prob == 0] = xsorted[0]
        hd[0, prob == 1] = xsorted[-1]
        if var:
            hd[1, prob == 0] = hd[1, prob == 1] = np.nan
            return hd
        return hd[0]
    data = ma.array(data, copy=False, dtype=float64)
    p = np.array(prob, copy=False, ndmin=1)
    if axis is None or data.ndim == 1:
        result = _hd_1D(data, p, var)
    else:
        if data.ndim > 2:
            raise ValueError("Array 'data' must be at most two dimensional, but got data.ndim = %d" % data.ndim)
        result = ma.apply_along_axis(_hd_1D, axis, data, p, var)
    return ma.fix_invalid(result, copy=False)

def hdmedian(data, axis=-1, var=False):
    if False:
        print('Hello World!')
    '\n    Returns the Harrell-Davis estimate of the median along the given axis.\n\n    Parameters\n    ----------\n    data : ndarray\n        Data array.\n    axis : int, optional\n        Axis along which to compute the quantiles. If None, use a flattened\n        array.\n    var : bool, optional\n        Whether to return the variance of the estimate.\n\n    Returns\n    -------\n    hdmedian : MaskedArray\n        The median values.  If ``var=True``, the variance is returned inside\n        the masked array.  E.g. for a 1-D array the shape change from (1,) to\n        (2,).\n\n    '
    result = hdquantiles(data, [0.5], axis=axis, var=var)
    return result.squeeze()

def hdquantiles_sd(data, prob=list([0.25, 0.5, 0.75]), axis=None):
    if False:
        print('Hello World!')
    '\n    The standard error of the Harrell-Davis quantile estimates by jackknife.\n\n    Parameters\n    ----------\n    data : array_like\n        Data array.\n    prob : sequence, optional\n        Sequence of quantiles to compute.\n    axis : int, optional\n        Axis along which to compute the quantiles. If None, use a flattened\n        array.\n\n    Returns\n    -------\n    hdquantiles_sd : MaskedArray\n        Standard error of the Harrell-Davis quantile estimates.\n\n    See Also\n    --------\n    hdquantiles\n\n    '

    def _hdsd_1D(data, prob):
        if False:
            return 10
        'Computes the std error for 1D arrays.'
        xsorted = np.sort(data.compressed())
        n = len(xsorted)
        hdsd = np.empty(len(prob), float64)
        if n < 2:
            hdsd.flat = np.nan
        vv = np.arange(n) / float(n - 1)
        betacdf = beta.cdf
        for (i, p) in enumerate(prob):
            _w = betacdf(vv, n * p, n * (1 - p))
            w = _w[1:] - _w[:-1]
            mx_ = np.zeros_like(xsorted)
            mx_[1:] = np.cumsum(w * xsorted[:-1])
            mx_[:-1] += np.cumsum(w[::-1] * xsorted[:0:-1])[::-1]
            hdsd[i] = np.sqrt(mx_.var() * (n - 1))
        return hdsd
    data = ma.array(data, copy=False, dtype=float64)
    p = np.array(prob, copy=False, ndmin=1)
    if axis is None:
        result = _hdsd_1D(data, p)
    else:
        if data.ndim > 2:
            raise ValueError("Array 'data' must be at most two dimensional, but got data.ndim = %d" % data.ndim)
        result = ma.apply_along_axis(_hdsd_1D, axis, data, p)
    return ma.fix_invalid(result, copy=False).ravel()

def trimmed_mean_ci(data, limits=(0.2, 0.2), inclusive=(True, True), alpha=0.05, axis=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Selected confidence interval of the trimmed mean along the given axis.\n\n    Parameters\n    ----------\n    data : array_like\n        Input data.\n    limits : {None, tuple}, optional\n        None or a two item tuple.\n        Tuple of the percentages to cut on each side of the array, with respect\n        to the number of unmasked data, as floats between 0. and 1. If ``n``\n        is the number of unmasked data before trimming, then\n        (``n * limits[0]``)th smallest data and (``n * limits[1]``)th\n        largest data are masked.  The total number of unmasked data after\n        trimming is ``n * (1. - sum(limits))``.\n        The value of one limit can be set to None to indicate an open interval.\n\n        Defaults to (0.2, 0.2).\n    inclusive : (2,) tuple of boolean, optional\n        If relative==False, tuple indicating whether values exactly equal to\n        the absolute limits are allowed.\n        If relative==True, tuple indicating whether the number of data being\n        masked on each side should be rounded (True) or truncated (False).\n\n        Defaults to (True, True).\n    alpha : float, optional\n        Confidence level of the intervals.\n\n        Defaults to 0.05.\n    axis : int, optional\n        Axis along which to cut. If None, uses a flattened version of `data`.\n\n        Defaults to None.\n\n    Returns\n    -------\n    trimmed_mean_ci : (2,) ndarray\n        The lower and upper confidence intervals of the trimmed data.\n\n    '
    data = ma.array(data, copy=False)
    trimmed = mstats.trimr(data, limits=limits, inclusive=inclusive, axis=axis)
    tmean = trimmed.mean(axis)
    tstde = mstats.trimmed_stde(data, limits=limits, inclusive=inclusive, axis=axis)
    df = trimmed.count(axis) - 1
    tppf = t.ppf(1 - alpha / 2.0, df)
    return np.array((tmean - tppf * tstde, tmean + tppf * tstde))

def mjci(data, prob=[0.25, 0.5, 0.75], axis=None):
    if False:
        print('Hello World!')
    '\n    Returns the Maritz-Jarrett estimators of the standard error of selected\n    experimental quantiles of the data.\n\n    Parameters\n    ----------\n    data : ndarray\n        Data array.\n    prob : sequence, optional\n        Sequence of quantiles to compute.\n    axis : int or None, optional\n        Axis along which to compute the quantiles. If None, use a flattened\n        array.\n\n    '

    def _mjci_1D(data, p):
        if False:
            while True:
                i = 10
        data = np.sort(data.compressed())
        n = data.size
        prob = (np.array(p) * n + 0.5).astype(int)
        betacdf = beta.cdf
        mj = np.empty(len(prob), float64)
        x = np.arange(1, n + 1, dtype=float64) / n
        y = x - 1.0 / n
        for (i, m) in enumerate(prob):
            W = betacdf(x, m - 1, n - m) - betacdf(y, m - 1, n - m)
            C1 = np.dot(W, data)
            C2 = np.dot(W, data ** 2)
            mj[i] = np.sqrt(C2 - C1 ** 2)
        return mj
    data = ma.array(data, copy=False)
    if data.ndim > 2:
        raise ValueError("Array 'data' must be at most two dimensional, but got data.ndim = %d" % data.ndim)
    p = np.array(prob, copy=False, ndmin=1)
    if axis is None:
        return _mjci_1D(data, p)
    else:
        return ma.apply_along_axis(_mjci_1D, axis, data, p)

def mquantiles_cimj(data, prob=[0.25, 0.5, 0.75], alpha=0.05, axis=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Computes the alpha confidence interval for the selected quantiles of the\n    data, with Maritz-Jarrett estimators.\n\n    Parameters\n    ----------\n    data : ndarray\n        Data array.\n    prob : sequence, optional\n        Sequence of quantiles to compute.\n    alpha : float, optional\n        Confidence level of the intervals.\n    axis : int or None, optional\n        Axis along which to compute the quantiles.\n        If None, use a flattened array.\n\n    Returns\n    -------\n    ci_lower : ndarray\n        The lower boundaries of the confidence interval.  Of the same length as\n        `prob`.\n    ci_upper : ndarray\n        The upper boundaries of the confidence interval.  Of the same length as\n        `prob`.\n\n    '
    alpha = min(alpha, 1 - alpha)
    z = norm.ppf(1 - alpha / 2.0)
    xq = mstats.mquantiles(data, prob, alphap=0, betap=0, axis=axis)
    smj = mjci(data, prob, axis=axis)
    return (xq - z * smj, xq + z * smj)

def median_cihs(data, alpha=0.05, axis=None):
    if False:
        while True:
            i = 10
    '\n    Computes the alpha-level confidence interval for the median of the data.\n\n    Uses the Hettmasperger-Sheather method.\n\n    Parameters\n    ----------\n    data : array_like\n        Input data. Masked values are discarded. The input should be 1D only,\n        or `axis` should be set to None.\n    alpha : float, optional\n        Confidence level of the intervals.\n    axis : int or None, optional\n        Axis along which to compute the quantiles. If None, use a flattened\n        array.\n\n    Returns\n    -------\n    median_cihs\n        Alpha level confidence interval.\n\n    '

    def _cihs_1D(data, alpha):
        if False:
            while True:
                i = 10
        data = np.sort(data.compressed())
        n = len(data)
        alpha = min(alpha, 1 - alpha)
        k = int(binom._ppf(alpha / 2.0, n, 0.5))
        gk = binom.cdf(n - k, n, 0.5) - binom.cdf(k - 1, n, 0.5)
        if gk < 1 - alpha:
            k -= 1
            gk = binom.cdf(n - k, n, 0.5) - binom.cdf(k - 1, n, 0.5)
        gkk = binom.cdf(n - k - 1, n, 0.5) - binom.cdf(k, n, 0.5)
        I = (gk - 1 + alpha) / (gk - gkk)
        lambd = (n - k) * I / float(k + (n - 2 * k) * I)
        lims = (lambd * data[k] + (1 - lambd) * data[k - 1], lambd * data[n - k - 1] + (1 - lambd) * data[n - k])
        return lims
    data = ma.array(data, copy=False)
    if axis is None:
        result = _cihs_1D(data, alpha)
    else:
        if data.ndim > 2:
            raise ValueError("Array 'data' must be at most two dimensional, but got data.ndim = %d" % data.ndim)
        result = ma.apply_along_axis(_cihs_1D, axis, data, alpha)
    return result

def compare_medians_ms(group_1, group_2, axis=None):
    if False:
        return 10
    '\n    Compares the medians from two independent groups along the given axis.\n\n    The comparison is performed using the McKean-Schrader estimate of the\n    standard error of the medians.\n\n    Parameters\n    ----------\n    group_1 : array_like\n        First dataset.  Has to be of size >=7.\n    group_2 : array_like\n        Second dataset.  Has to be of size >=7.\n    axis : int, optional\n        Axis along which the medians are estimated. If None, the arrays are\n        flattened.  If `axis` is not None, then `group_1` and `group_2`\n        should have the same shape.\n\n    Returns\n    -------\n    compare_medians_ms : {float, ndarray}\n        If `axis` is None, then returns a float, otherwise returns a 1-D\n        ndarray of floats with a length equal to the length of `group_1`\n        along `axis`.\n\n    Examples\n    --------\n\n    >>> from scipy import stats\n    >>> a = [1, 2, 3, 4, 5, 6, 7]\n    >>> b = [8, 9, 10, 11, 12, 13, 14]\n    >>> stats.mstats.compare_medians_ms(a, b, axis=None)\n    1.0693225866553746e-05\n\n    The function is vectorized to compute along a given axis.\n\n    >>> import numpy as np\n    >>> rng = np.random.default_rng()\n    >>> x = rng.random(size=(3, 7))\n    >>> y = rng.random(size=(3, 8))\n    >>> stats.mstats.compare_medians_ms(x, y, axis=1)\n    array([0.36908985, 0.36092538, 0.2765313 ])\n\n    References\n    ----------\n    .. [1] McKean, Joseph W., and Ronald M. Schrader. "A comparison of methods\n       for studentizing the sample median." Communications in\n       Statistics-Simulation and Computation 13.6 (1984): 751-773.\n\n    '
    (med_1, med_2) = (ma.median(group_1, axis=axis), ma.median(group_2, axis=axis))
    (std_1, std_2) = (mstats.stde_median(group_1, axis=axis), mstats.stde_median(group_2, axis=axis))
    W = np.abs(med_1 - med_2) / ma.sqrt(std_1 ** 2 + std_2 ** 2)
    return 1 - norm.cdf(W)

def idealfourths(data, axis=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns an estimate of the lower and upper quartiles.\n\n    Uses the ideal fourths algorithm.\n\n    Parameters\n    ----------\n    data : array_like\n        Input array.\n    axis : int, optional\n        Axis along which the quartiles are estimated. If None, the arrays are\n        flattened.\n\n    Returns\n    -------\n    idealfourths : {list of floats, masked array}\n        Returns the two internal values that divide `data` into four parts\n        using the ideal fourths algorithm either along the flattened array\n        (if `axis` is None) or along `axis` of `data`.\n\n    '

    def _idf(data):
        if False:
            print('Hello World!')
        x = data.compressed()
        n = len(x)
        if n < 3:
            return [np.nan, np.nan]
        (j, h) = divmod(n / 4.0 + 5 / 12.0, 1)
        j = int(j)
        qlo = (1 - h) * x[j - 1] + h * x[j]
        k = n - j
        qup = (1 - h) * x[k] + h * x[k - 1]
        return [qlo, qup]
    data = ma.sort(data, axis=axis).view(MaskedArray)
    if axis is None:
        return _idf(data)
    else:
        return ma.apply_along_axis(_idf, axis, data)

def rsh(data, points=None):
    if False:
        print('Hello World!')
    "\n    Evaluates Rosenblatt's shifted histogram estimators for each data point.\n\n    Rosenblatt's estimator is a centered finite-difference approximation to the\n    derivative of the empirical cumulative distribution function.\n\n    Parameters\n    ----------\n    data : sequence\n        Input data, should be 1-D. Masked values are ignored.\n    points : sequence or None, optional\n        Sequence of points where to evaluate Rosenblatt shifted histogram.\n        If None, use the data.\n\n    "
    data = ma.array(data, copy=False)
    if points is None:
        points = data
    else:
        points = np.array(points, copy=False, ndmin=1)
    if data.ndim != 1:
        raise AttributeError('The input array should be 1D only !')
    n = data.count()
    r = idealfourths(data, axis=None)
    h = 1.2 * (r[-1] - r[0]) / n ** (1.0 / 5)
    nhi = (data[:, None] <= points[None, :] + h).sum(0)
    nlo = (data[:, None] < points[None, :] - h).sum(0)
    return (nhi - nlo) / (2.0 * n * h)