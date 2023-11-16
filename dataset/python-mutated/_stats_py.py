import cupy

def _first(arr, axis):
    if False:
        return 10
    'Return arr[..., 0:1, ...] where 0:1 is in the `axis` position\n\n    '
    return cupy.take_along_axis(arr, cupy.array(0, ndmin=arr.ndim), axis)

def _isconst(x):
    if False:
        while True:
            i = 10
    'Check if all values in x are the same.  nans are ignored.\n    x must be a 1d array. The return value is a 1d array\n    with length 1, so it can be used in cupy.apply_along_axis.\n\n    '
    y = x[~cupy.isnan(x)]
    if y.size == 0:
        return cupy.array([True])
    else:
        return (y[0] == y).all(keepdims=True)

def zscore(a, axis=0, ddof=0, nan_policy='propagate'):
    if False:
        i = 10
        return i + 15
    "Compute the z-score.\n\n    Compute the z-score of each value in the sample, relative to\n    the sample mean and standard deviation.\n\n    Parameters\n    ----------\n    a : array-like\n        An array like object containing the sample data\n    axis : int or None, optional\n        Axis along which to operate. Default is 0. If None,\n        compute over the whole arrsy `a`\n    ddof : int, optional\n        Degrees of freedom correction in the calculation of the\n        standard deviation. Default is 0\n    nan_policy : {'propagate', 'raise', 'omit'}, optional\n        Defines how to handle when input contains nan. 'propagate'\n        returns nan, 'raise' throws an error, 'omit' performs\n        the calculations ignoring nan values. Default is\n        'propagate'. Note that when the value is 'omit',\n        nans in the input also propagate to the output,\n        but they do not affect the z-scores computed\n        for the non-nan values\n\n    Returns\n    -------\n    zscore : array-like\n        The z-scores, standardized by mean and standard deviation of\n        input array `a`\n\n    "
    return zmap(a, a, axis=axis, ddof=ddof, nan_policy=nan_policy)

def zmap(scores, compare, axis=0, ddof=0, nan_policy='propagate'):
    if False:
        for i in range(10):
            print('nop')
    "Calculate the relative z-scores.\n\n    Return an array of z-scores, i.e., scores that are standardized\n    to zero mean and unit variance, where mean and variance are\n    calculated from the comparison array.\n\n    Parameters\n    ----------\n    scores : array-like\n        The input for which z-scores are calculated\n    compare : array-like\n        The input from which the mean and standard deviation of\n        the normalization are taken; assumed to have the same\n        dimension as `scores`\n    axis : int or None, optional\n        Axis over which mean and variance of `compare` are calculated.\n        Default is 0. If None, compute over the whole array `scores`\n    ddof : int, optional\n        Degrees of freedom correction in the calculation of the\n        standard deviation. Default is 0\n    nan_policy : {'propagate', 'raise', 'omit'}, optional\n        Defines how to handle the occurrence of nans in `compare`.\n        'propagate' returns nan, 'raise' raises an exception, 'omit'\n        performs the calculations ignoring nan values. Default is\n        'propagate'. Note that when the value is 'omit', nans in `scores`\n        also propagate to the output, but they do not affect the z-scores\n        computed for the non-nan values\n\n    Returns\n    -------\n    zscore : array-like\n        Z-scores, in the same shape as `scores`\n\n    "
    policies = ['propagate', 'raise', 'omit']
    if nan_policy not in policies:
        raise ValueError('nan_policy must be one of {%s}' % ', '.join(("'%s'" % s for s in policies)))
    a = compare
    if a.size == 0:
        return cupy.empty(a.shape)
    if nan_policy == 'raise':
        contains_nan = cupy.isnan(cupy.sum(a))
        if contains_nan:
            raise ValueError('The input contains nan values')
    if nan_policy == 'omit':
        if axis is None:
            mn = cupy.nanmean(a.ravel())
            std = cupy.nanstd(a.ravel(), ddof=ddof)
            isconst = _isconst(a.ravel())
        else:
            mn = cupy.nanmean(a, axis=axis, keepdims=True)
            std = cupy.nanstd(a, axis=axis, keepdims=True, ddof=ddof)
            isconst = (_first(a, axis) == a).all(axis=axis, keepdims=True)
    else:
        mn = a.mean(axis=axis, keepdims=True)
        std = a.std(axis=axis, ddof=ddof, keepdims=True)
        if axis is None:
            isconst = (a.ravel()[0] == a).all()
        else:
            isconst = (_first(a, axis) == a).all(axis=axis, keepdims=True)
    std[isconst] = 1.0
    z = (scores - mn) / std
    z[cupy.broadcast_to(isconst, z.shape)] = cupy.nan
    return z