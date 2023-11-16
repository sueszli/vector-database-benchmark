import functools
import warnings
import numpy
import cupy
from cupy import _core

def corrcoef(a, y=None, rowvar=True, bias=None, ddof=None, *, dtype=None):
    if False:
        while True:
            i = 10
    'Returns the Pearson product-moment correlation coefficients of an array.\n\n    Args:\n        a (cupy.ndarray): Array to compute the Pearson product-moment\n            correlation coefficients.\n        y (cupy.ndarray): An additional set of variables and observations.\n        rowvar (bool): If ``True``, then each row represents a variable, with\n            observations in the columns. Otherwise, the relationship is\n            transposed.\n        bias (None): Has no effect, do not use.\n        ddof (None): Has no effect, do not use.\n        dtype: Data type specifier. By default, the return data-type will have\n            at least `numpy.float64` precision.\n\n    Returns:\n        cupy.ndarray: The Pearson product-moment correlation coefficients of\n        the input array.\n\n    .. seealso:: :func:`numpy.corrcoef`\n\n    '
    if bias is not None or ddof is not None:
        warnings.warn('bias and ddof have no effect and are deprecated', DeprecationWarning)
    out = cov(a, y, rowvar, dtype=dtype)
    try:
        d = cupy.diag(out)
    except ValueError:
        return out / out
    stddev = cupy.sqrt(d.real)
    out /= stddev[:, None]
    out /= stddev[None, :]
    cupy.clip(out.real, -1, 1, out=out.real)
    if cupy.iscomplexobj(out):
        cupy.clip(out.imag, -1, 1, out=out.imag)
    return out

def correlate(a, v, mode='valid'):
    if False:
        while True:
            i = 10
    'Returns the cross-correlation of two 1-dimensional sequences.\n\n    Args:\n        a (cupy.ndarray): first 1-dimensional input.\n        v (cupy.ndarray): second 1-dimensional input.\n        mode (str, optional): `valid`, `same`, `full`\n\n    Returns:\n        cupy.ndarray: Discrete cross-correlation of a and v.\n\n    .. seealso:: :func:`numpy.correlate`\n\n    '
    if a.size == 0 or v.size == 0:
        raise ValueError('Array arguments cannot be empty')
    if a.ndim != 1 or v.ndim != 1:
        raise ValueError('object too deep for desired array')
    method = cupy._math.misc._choose_conv_method(a, v, mode)
    if method == 'direct':
        out = cupy._math.misc._dot_convolve(a, v.conj()[::-1], mode)
    elif method == 'fft':
        out = cupy._math.misc._fft_convolve(a, v.conj()[::-1], mode)
    else:
        raise ValueError('Unsupported method')
    return out

def cov(a, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None, *, dtype=None):
    if False:
        print('Hello World!')
    'Returns the covariance matrix of an array.\n\n    This function currently does not support ``fweights`` and ``aweights``\n    options.\n\n    Args:\n        a (cupy.ndarray): Array to compute covariance matrix.\n        y (cupy.ndarray): An additional set of variables and observations.\n        rowvar (bool): If ``True``, then each row represents a variable, with\n            observations in the columns. Otherwise, the relationship is\n            transposed.\n        bias (bool): If ``False``, normalization is by ``(N - 1)``, where N is\n            the number of observations given (unbiased estimate). If ``True``,\n            then normalization is by ``N``.\n        ddof (int): If not ``None`` the default value implied by bias is\n            overridden. Note that ``ddof=1`` will return the unbiased estimate\n            and ``ddof=0`` will return the simple average.\n\n        fweights (cupy.ndarray, int): 1-D array of integer frequency weights.\n            the number of times each observation vector should be repeated.\n            It is required that fweights >= 0. However, the function will not\n            error when fweights < 0 for performance reasons.\n        aweights (cupy.ndarray): 1-D array of observation vector weights.\n            These relative weights are typically large for observations\n            considered "important" and smaller for observations considered\n            less "important". If ``ddof=0`` the array of weights can be used\n            to assign probabilities to observation vectors.\n            It is required that aweights >= 0. However, the function will not\n            error when aweights < 0 for performance reasons.\n        dtype: Data type specifier. By default, the return data-type will have\n            at least `numpy.float64` precision.\n\n    Returns:\n        cupy.ndarray: The covariance matrix of the input array.\n\n    .. seealso:: :func:`numpy.cov`\n\n    '
    if ddof is not None and ddof != int(ddof):
        raise ValueError('ddof must be integer')
    if a.ndim > 2:
        raise ValueError('Input must be <= 2-d')
    if dtype is None:
        if y is None:
            dtype = numpy.promote_types(a.dtype, numpy.float64)
        else:
            if y.ndim > 2:
                raise ValueError('y must be <= 2-d')
            dtype = functools.reduce(numpy.promote_types, (a.dtype, y.dtype, numpy.float64))
    X = cupy.array(a, ndmin=2, dtype=dtype)
    if not rowvar and X.shape[0] != 1:
        X = X.T
    if X.shape[0] == 0:
        return cupy.array([]).reshape(0, 0)
    if y is not None:
        y = cupy.array(y, copy=False, ndmin=2, dtype=dtype)
        if not rowvar and y.shape[0] != 1:
            y = y.T
        X = _core.concatenate_method((X, y), axis=0)
    if ddof is None:
        ddof = 0 if bias else 1
    w = None
    if fweights is not None:
        if not isinstance(fweights, cupy.ndarray):
            raise TypeError('fweights must be a cupy.ndarray')
        if fweights.dtype.char not in 'bBhHiIlLqQ':
            raise TypeError('fweights must be integer')
        fweights = fweights.astype(dtype=float)
        if fweights.ndim > 1:
            raise RuntimeError('cannot handle multidimensional fweights')
        if fweights.shape[0] != X.shape[1]:
            raise RuntimeError('incompatible numbers of samples and fweights')
        w = fweights
    if aweights is not None:
        if not isinstance(aweights, cupy.ndarray):
            raise TypeError('aweights must be a cupy.ndarray')
        aweights = aweights.astype(dtype=float)
        if aweights.ndim > 1:
            raise RuntimeError('cannot handle multidimensional aweights')
        if aweights.shape[0] != X.shape[1]:
            raise RuntimeError('incompatible numbers of samples and aweights')
        if w is None:
            w = aweights
        else:
            w *= aweights
    (avg, w_sum) = cupy.average(X, axis=1, weights=w, returned=True)
    w_sum = w_sum[0]
    if w is None:
        fact = X.shape[1] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * sum(w * aweights) / w_sum
    if fact <= 0:
        warnings.warn('Degrees of freedom <= 0 for slice', RuntimeWarning, stacklevel=2)
        fact = 0.0
    X -= X.mean(axis=1)[:, None]
    if w is None:
        X_T = X.T
    else:
        X_T = (X * w).T
    out = X.dot(X_T.conj()) * (1 / cupy.float64(fact))
    return out.squeeze()