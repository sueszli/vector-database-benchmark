import cupy as cp
_log_softmax_kernel = cp._core.ReductionKernel('T x1', 'T y', 'exp(x1)', 'a + b', 'y = log(a)', '0', name='log_softmax')

def log_softmax(x, axis=None):
    if False:
        while True:
            i = 10
    'Compute logarithm of softmax function\n\n    Parameters\n    ----------\n    x : array-like\n        Input array\n    axis : int or tuple of ints, optional\n        Axis to compute values along. Default is None and softmax\n        will be  computed over the entire array `x`\n\n    Returns\n    -------\n    s : cupy.ndarry\n        An array with the same shape as `x`. Exponential of the\n        result will sum to 1 along the specified axis. If `x` is a\n        scalar, a scalar is returned\n\n    '
    x_max = cp.amax(x, axis=axis, keepdims=True)
    if x_max.ndim > 0:
        x_max[~cp.isfinite(x_max)] = 0
    elif not cp.isfinite(x_max):
        x_max = 0
    tmp = x - x_max
    if tmp.dtype.kind in 'iu':
        for out_dtype in [cp.float16, cp.float32, cp.float64]:
            if cp.can_cast(tmp.dtype, out_dtype):
                tmp = tmp.astype(out_dtype)
                break
    out = _log_softmax_kernel(tmp, axis=axis, keepdims=True)
    out = tmp - out
    return out