import cupy

def boxcox_llf(lmb, data):
    if False:
        i = 10
        return i + 15
    'The boxcox log-likelihood function.\n\n    Parameters\n    ----------\n    lmb : scalar\n        Parameter for Box-Cox transformation\n    data : array-like\n        Data to calculate Box-Cox log-likelihood for. If\n        `data` is multi-dimensional, the log-likelihood\n        is calculated along the first axis\n\n    Returns\n    -------\n    llf : float or cupy.ndarray\n        Box-Cox log-likelihood of `data` given `lmb`. A float\n        for 1-D `data`, an array otherwise\n\n    See Also\n    --------\n    scipy.stats.boxcox_llf\n\n    '
    if data.ndim == 1 and data.dtype == cupy.float16:
        data = data.astype(cupy.float64)
    if data.ndim == 1 and data.dtype == cupy.float32:
        data = data.astype(cupy.float64)
    if data.ndim == 1 and data.dtype == cupy.complex64:
        data = data.astype(cupy.complex128)
    N = data.shape[0]
    if N == 0:
        return cupy.array(cupy.nan)
    logdata = cupy.log(data)
    if lmb == 0:
        variance = cupy.var(logdata, axis=0)
    else:
        variance = cupy.var(data ** lmb / lmb, axis=0)
    return (lmb - 1) * cupy.sum(logdata, axis=0) - N / 2 * cupy.log(variance)