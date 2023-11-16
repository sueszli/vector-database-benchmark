"""get versions of mstats percentile functions that also work with non-masked arrays

uses dispatch to mstats version for difficult cases:
  - data is masked array
  - data requires nan handling (masknan=True)
  - data should be trimmed (limit is non-empty)
handle simple cases directly, which does not require apply_along_axis
changes compared to mstats: plotting_positions for n-dim with axis argument
addition: plotting_positions_w1d: with weights, 1d ndarray only

TODO:
consistency with scipy.stats versions not checked
docstrings from mstats not updated yet
code duplication, better solutions (?)
convert examples to tests
rename alphap, betap for consistency
timing question: one additional argsort versus apply_along_axis
weighted plotting_positions
- I have not figured out nd version of weighted plotting_positions
- add weighted quantiles


"""
import numpy as np
from numpy import ma
from scipy import stats

def quantiles(a, prob=list([0.25, 0.5, 0.75]), alphap=0.4, betap=0.4, axis=None, limit=(), masknan=False):
    if False:
        return 10
    "\n    Computes empirical quantiles for a data array.\n\n    Samples quantile are defined by :math:`Q(p) = (1-g).x[i] +g.x[i+1]`,\n    where :math:`x[j]` is the *j*th order statistic, and\n    `i = (floor(n*p+m))`, `m=alpha+p*(1-alpha-beta)` and `g = n*p + m - i`.\n\n    Typical values of (alpha,beta) are:\n        - (0,1)    : *p(k) = k/n* : linear interpolation of cdf (R, type 4)\n        - (.5,.5)  : *p(k) = (k+1/2.)/n* : piecewise linear\n          function (R, type 5)\n        - (0,0)    : *p(k) = k/(n+1)* : (R type 6)\n        - (1,1)    : *p(k) = (k-1)/(n-1)*. In this case, p(k) = mode[F(x[k])].\n          That's R default (R type 7)\n        - (1/3,1/3): *p(k) = (k-1/3)/(n+1/3)*. Then p(k) ~ median[F(x[k])].\n          The resulting quantile estimates are approximately median-unbiased\n          regardless of the distribution of x. (R type 8)\n        - (3/8,3/8): *p(k) = (k-3/8)/(n+1/4)*. Blom.\n          The resulting quantile estimates are approximately unbiased\n          if x is normally distributed (R type 9)\n        - (.4,.4)  : approximately quantile unbiased (Cunnane)\n        - (.35,.35): APL, used with PWM ?? JP\n        - (0.35, 0.65): PWM   ?? JP  p(k) = (k-0.35)/n\n\n    Parameters\n    ----------\n    a : array_like\n        Input data, as a sequence or array of dimension at most 2.\n    prob : array_like, optional\n        List of quantiles to compute.\n    alpha : float, optional\n        Plotting positions parameter, default is 0.4.\n    beta : float, optional\n        Plotting positions parameter, default is 0.4.\n    axis : int, optional\n        Axis along which to perform the trimming.\n        If None (default), the input array is first flattened.\n    limit : tuple\n        Tuple of (lower, upper) values.\n        Values of `a` outside this closed interval are ignored.\n\n    Returns\n    -------\n    quants : MaskedArray\n        An array containing the calculated quantiles.\n\n    Examples\n    --------\n    >>> from scipy.stats.mstats import mquantiles\n    >>> a = np.array([6., 47., 49., 15., 42., 41., 7., 39., 43., 40., 36.])\n    >>> mquantiles(a)\n    array([ 19.2,  40. ,  42.8])\n\n    Using a 2D array, specifying axis and limit.\n\n    >>> data = np.array([[   6.,    7.,    1.],\n                         [  47.,   15.,    2.],\n                         [  49.,   36.,    3.],\n                         [  15.,   39.,    4.],\n                         [  42.,   40., -999.],\n                         [  41.,   41., -999.],\n                         [   7., -999., -999.],\n                         [  39., -999., -999.],\n                         [  43., -999., -999.],\n                         [  40., -999., -999.],\n                         [  36., -999., -999.]])\n    >>> mquantiles(data, axis=0, limit=(0, 50))\n    array([[ 19.2 ,  14.6 ,   1.45],\n           [ 40.  ,  37.5 ,   2.5 ],\n           [ 42.8 ,  40.05,   3.55]])\n\n    >>> data[:, 2] = -999.\n    >>> mquantiles(data, axis=0, limit=(0, 50))\n    masked_array(data =\n     [[19.2 14.6 --]\n     [40.0 37.5 --]\n     [42.8 40.05 --]],\n                 mask =\n     [[False False  True]\n      [False False  True]\n      [False False  True]],\n           fill_value = 1e+20)\n    "
    if isinstance(a, np.ma.MaskedArray):
        return stats.mstats.mquantiles(a, prob=prob, alphap=alphap, betap=alphap, axis=axis, limit=limit)
    if limit:
        marr = stats.mstats.mquantiles(a, prob=prob, alphap=alphap, betap=alphap, axis=axis, limit=limit)
        return ma.filled(marr, fill_value=np.nan)
    if masknan:
        nanmask = np.isnan(a)
        if nanmask.any():
            marr = ma.array(a, mask=nanmask)
            marr = stats.mstats.mquantiles(marr, prob=prob, alphap=alphap, betap=alphap, axis=axis, limit=limit)
            return ma.filled(marr, fill_value=np.nan)
    data = np.asarray(a)
    p = np.array(prob, copy=False, ndmin=1)
    m = alphap + p * (1.0 - alphap - betap)
    isrolled = False
    if axis is None:
        data = data.ravel()
        axis = 0
    else:
        axis = np.arange(data.ndim)[axis]
        data = np.rollaxis(data, axis)
        isrolled = True
    x = np.sort(data, axis=0)
    n = x.shape[0]
    returnshape = list(data.shape)
    returnshape[axis] = p
    if n == 0:
        return np.empty(len(p), dtype=float)
    elif n == 1:
        return np.resize(x, p.shape)
    aleph = n * p + m
    k = np.floor(aleph.clip(1, n - 1)).astype(int)
    ind = [None] * x.ndim
    ind[0] = slice(None)
    gamma = (aleph - k).clip(0, 1)[ind]
    q = (1.0 - gamma) * x[k - 1] + gamma * x[k]
    if isrolled:
        return np.rollaxis(q, 0, axis + 1)
    else:
        return q

def scoreatpercentile(data, per, limit=(), alphap=0.4, betap=0.4, axis=0, masknan=None):
    if False:
        while True:
            i = 10
    "Calculate the score at the given 'per' percentile of the\n    sequence a.  For example, the score at per=50 is the median.\n\n    This function is a shortcut to mquantile\n    "
    per = np.asarray(per, float)
    if (per < 0).any() or (per > 100.0).any():
        raise ValueError('The percentile should be between 0. and 100. ! (got %s)' % per)
    return quantiles(data, prob=[per / 100.0], alphap=alphap, betap=betap, limit=limit, axis=axis, masknan=masknan).squeeze()

def plotting_positions(data, alpha=0.4, beta=0.4, axis=0, masknan=False):
    if False:
        print('Hello World!')
    'Returns the plotting positions (or empirical percentile points) for the\n    data.\n    Plotting positions are defined as (i-alpha)/(n+1-alpha-beta), where:\n        - i is the rank order statistics (starting at 1)\n        - n is the number of unmasked values along the given axis\n        - alpha and beta are two parameters.\n\n    Typical values for alpha and beta are:\n        - (0,1)    : *p(k) = k/n* : linear interpolation of cdf (R, type 4)\n        - (.5,.5)  : *p(k) = (k-1/2.)/n* : piecewise linear function (R, type 5)\n          (Bliss 1967: "Rankit")\n        - (0,0)    : *p(k) = k/(n+1)* : Weibull (R type 6), (Van der Waerden 1952)\n        - (1,1)    : *p(k) = (k-1)/(n-1)*. In this case, p(k) = mode[F(x[k])].\n          That\'s R default (R type 7)\n        - (1/3,1/3): *p(k) = (k-1/3)/(n+1/3)*. Then p(k) ~ median[F(x[k])].\n          The resulting quantile estimates are approximately median-unbiased\n          regardless of the distribution of x. (R type 8), (Tukey 1962)\n        - (3/8,3/8): *p(k) = (k-3/8)/(n+1/4)*.\n          The resulting quantile estimates are approximately unbiased\n          if x is normally distributed (R type 9) (Blom 1958)\n        - (.4,.4)  : approximately quantile unbiased (Cunnane)\n        - (.35,.35): APL, used with PWM\n\n    Parameters\n    ----------\n    x : sequence\n        Input data, as a sequence or array of dimension at most 2.\n    prob : sequence\n        List of quantiles to compute.\n    alpha : {0.4, float} optional\n        Plotting positions parameter.\n    beta : {0.4, float} optional\n        Plotting positions parameter.\n\n    Notes\n    -----\n    I think the adjustments assume that there are no ties in order to be a reasonable\n    approximation to a continuous density function. TODO: check this\n\n    References\n    ----------\n    unknown,\n    dates to original papers from Beasley, Erickson, Allison 2009 Behav Genet\n    '
    if isinstance(data, np.ma.MaskedArray):
        if axis is None or data.ndim == 1:
            return stats.mstats.plotting_positions(data, alpha=alpha, beta=beta)
        else:
            return ma.apply_along_axis(stats.mstats.plotting_positions, axis, data, alpha=alpha, beta=beta)
    if masknan:
        nanmask = np.isnan(data)
        if nanmask.any():
            marr = ma.array(data, mask=nanmask)
            if axis is None or data.ndim == 1:
                marr = stats.mstats.plotting_positions(marr, alpha=alpha, beta=beta)
            else:
                marr = ma.apply_along_axis(stats.mstats.plotting_positions, axis, marr, alpha=alpha, beta=beta)
            return ma.filled(marr, fill_value=np.nan)
    data = np.asarray(data)
    if data.size == 1:
        data = np.atleast_1d(data)
        axis = 0
    if axis is None:
        data = data.ravel()
        axis = 0
    n = data.shape[axis]
    if data.ndim == 1:
        plpos = np.empty(data.shape, dtype=float)
        plpos[data.argsort()] = (np.arange(1, n + 1) - alpha) / (n + 1.0 - alpha - beta)
    else:
        plpos = (data.argsort(axis).argsort(axis) + 1.0 - alpha) / (n + 1.0 - alpha - beta)
    return plpos
meppf = plotting_positions

def plotting_positions_w1d(data, weights=None, alpha=0.4, beta=0.4, method='notnormed'):
    if False:
        print('Hello World!')
    'Weighted plotting positions (or empirical percentile points) for the data.\n\n    observations are weighted and the plotting positions are defined as\n    (ws-alpha)/(n-alpha-beta), where:\n        - ws is the weighted rank order statistics or cumulative weighted sum,\n          normalized to n if method is "normed"\n        - n is the number of values along the given axis if method is "normed"\n          and total weight otherwise\n        - alpha and beta are two parameters.\n\n    wtd.quantile in R package Hmisc seems to use the "notnormed" version.\n    notnormed coincides with unweighted segment in example, drop "normed" version ?\n\n\n    See Also\n    --------\n    plotting_positions : unweighted version that works also with more than one\n        dimension and has other options\n    '
    x = np.atleast_1d(data)
    if x.ndim > 1:
        raise ValueError('currently implemented only for 1d')
    if weights is None:
        weights = np.ones(x.shape)
    else:
        weights = np.array(weights, float, copy=False, ndmin=1)
        if weights.shape != x.shape:
            raise ValueError('if weights is given, it needs to be the sameshape as data')
    n = len(x)
    xargsort = x.argsort()
    ws = weights[xargsort].cumsum()
    res = np.empty(x.shape)
    if method == 'normed':
        res[xargsort] = (1.0 * ws / ws[-1] * n - alpha) / (n + 1.0 - alpha - beta)
    else:
        res[xargsort] = (1.0 * ws - alpha) / (ws[-1] + 1.0 - alpha - beta)
    return res

def edf_normal_inverse_transformed(x, alpha=3.0 / 8, beta=3.0 / 8, axis=0):
    if False:
        for i in range(10):
            print('nop')
    'rank based normal inverse transformed cdf\n    '
    from scipy import stats
    ranks = plotting_positions(x, alpha=alpha, beta=alpha, axis=0, masknan=False)
    ranks_transf = stats.norm.ppf(ranks)
    return ranks_transf
if __name__ == '__main__':
    x = np.arange(5)
    print(plotting_positions(x))
    x = np.arange(10).reshape(-1, 2)
    print(plotting_positions(x))
    print(quantiles(x, axis=0))
    print(quantiles(x, axis=None))
    print(quantiles(x, axis=1))
    xm = ma.array(x)
    x2 = x.astype(float)
    x2[1, 0] = np.nan
    print(plotting_positions(xm, axis=0))
    for sl1 in [slice(None), 0]:
        print((plotting_positions(xm[sl1, 0]) == plotting_positions(x[sl1, 0])).all())
        print((quantiles(xm[sl1, 0]) == quantiles(x[sl1, 0])).all())
        print((stats.mstats.mquantiles(ma.fix_invalid(x2[sl1, 0])) == quantiles(x2[sl1, 0], masknan=1)).all())
    for ax in [0, 1, None, -1]:
        print((plotting_positions(xm, axis=ax) == plotting_positions(x, axis=ax)).all())
        print((quantiles(xm, axis=ax) == quantiles(x, axis=ax)).all())
        print((stats.mstats.mquantiles(ma.fix_invalid(x2), axis=ax) == quantiles(x2, axis=ax, masknan=1)).all())
    print((stats.mstats.plotting_positions(ma.fix_invalid(x2)) == plotting_positions(x2, axis=None, masknan=1)).all())
    x3 = np.dstack((x, x)).T
    for ax in [1, 2]:
        print((plotting_positions(x3, axis=ax)[0] == plotting_positions(x.T, axis=ax - 1)).all())
    np.testing.assert_equal(plotting_positions(np.arange(10), alpha=0.35, beta=1 - 0.35), (1 + np.arange(10) - 0.35) / 10)
    np.testing.assert_equal(plotting_positions(np.arange(10), alpha=0.4, beta=0.4), (1 + np.arange(10) - 0.4) / (10 + 0.2))
    np.testing.assert_equal(plotting_positions(np.arange(10)), (1 + np.arange(10) - 0.4) / (10 + 0.2))
    print('')
    print(scoreatpercentile(x, [10, 90]))
    print(plotting_positions_w1d(x[:, 0]))
    print((plotting_positions_w1d(x[:, 0]) == plotting_positions(x[:, 0])).all())
    w1 = [1, 1, 2, 1, 1]
    plotexample = 1
    if plotexample:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title('ppf, cdf values on horizontal axis')
        plt.step(plotting_positions_w1d(x[:, 0], weights=w1, method='0'), x[:, 0], where='post')
        plt.step(stats.mstats.plotting_positions(np.repeat(x[:, 0], w1, axis=0)), np.repeat(x[:, 0], w1, axis=0), where='post')
        plt.plot(plotting_positions_w1d(x[:, 0], weights=w1, method='0'), x[:, 0], '-o')
        plt.plot(stats.mstats.plotting_positions(np.repeat(x[:, 0], w1, axis=0)), np.repeat(x[:, 0], w1, axis=0), '-o')
        plt.figure()
        plt.title('cdf, cdf values on vertical axis')
        plt.step(x[:, 0], plotting_positions_w1d(x[:, 0], weights=w1, method='0'), where='post')
        plt.step(np.repeat(x[:, 0], w1, axis=0), stats.mstats.plotting_positions(np.repeat(x[:, 0], w1, axis=0)), where='post')
        plt.plot(x[:, 0], plotting_positions_w1d(x[:, 0], weights=w1, method='0'), '-o')
        plt.plot(np.repeat(x[:, 0], w1, axis=0), stats.mstats.plotting_positions(np.repeat(x[:, 0], w1, axis=0)), '-o')
    plt.show()