import cupy as cp

def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    if False:
        print('Hello World!')
    'Compute the log of the sum of exponentials of input elements.\n\n    Parameters\n    ----------\n    a : cupy.ndarray\n        Input array\n    axis : None or int or tuple of ints, optional\n        Axis or axes over which the sum is taken. By default\n        `axis` is None, and all elements are summed\n    keepdims : bool, optional\n        If this is set to True, the axes which are reduced\n        are left in the result as dimensions with size one. With\n        this option, the result will broadcast correctly\n        against the original array\n    b : cupy.ndarray, optional\n        Scaling factor for exp(`a`) must be of the same shape as `a` or\n        broadcastable to `a`. These values may be negative in order to\n        implement subtraction\n    return_sign : bool, optional\n        If this is set to True, the result will be a pair containing sign\n        information; if False, results that are negative will be returned\n        as NaN. Default is False\n\n    Returns\n    -------\n    res : cupy.ndarray\n        The result, ``cp.log(cp.sum(cp.exp(a)))`` calculated\n        in a numerically more stable way. If `b` is given then\n        ``cp.log(cp.sum(b*cp.exp(a)))`` is returned\n    sgn : cupy.ndarray\n        If return_sign is True, this will be an array of floating-point\n        numbers matching res and +1, 0, or -1 depending on the sign of\n        the result. If False, only onw result is returned\n\n    See Also\n    --------\n    scipy.special.logsumexp\n\n    '
    if b is not None:
        (a, b) = cp.broadcast_arrays(a, b)
        if cp.any(b == 0):
            a = a + 0.0
            a[b == 0] = -cp.inf
    a_max = cp.max(a, axis=axis, keepdims=True)
    if a_max.ndim > 0:
        a_max[~cp.isfinite(a_max)] = 0
    elif not cp.isfinite(a_max):
        a_max = 0
    if b is not None:
        tmp = b * cp.exp(a - a_max)
    else:
        tmp = cp.exp(a - a_max)
    s = cp.sum(tmp, axis=axis, keepdims=keepdims)
    if return_sign:
        sgn = cp.sign(s)
        s *= sgn
    out = cp.log(s)
    if not keepdims:
        a_max = cp.squeeze(a_max, axis=axis)
    out += a_max
    if return_sign:
        return (out, sgn)
    else:
        return out