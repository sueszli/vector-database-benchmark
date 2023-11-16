import functools
import numpy
import cupy
from cupy._core import _routines_statistics as _statistics

def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    if False:
        while True:
            i = 10
    'Compute the median along the specified axis.\n\n    Returns the median of the array elements.\n\n    Args:\n        a (cupy.ndarray): Array to compute the median.\n        axis (int, sequence of int or None): Axis along which the medians are\n             computed. The flattened array is used by default.\n        out (cupy.ndarray): Output array.\n        overwrite_input (bool): If ``True``, then allow use of memory of input\n            array a for calculations. The input array will be modified by the\n            call to median. This will save memory when you do not need to\n            preserve the contents of the input array. Treat the input as\n            undefined, but it will probably be fully or partially sorted.\n            Default is ``False``. If ``overwrite_input`` is ``True`` and ``a``\n            is not already an ndarray, an error will be raised.\n        keepdims (bool): If ``True``, the axis is remained as an axis of size\n            one.\n\n    Returns:\n        cupy.ndarray: The median of ``a``, along the axis if specified.\n\n    .. seealso:: :func:`numpy.median`\n\n    '
    return _statistics._median(a, axis, out, overwrite_input, keepdims)

def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    if False:
        i = 10
        return i + 15
    'Compute the median along the specified axis, while ignoring NaNs.\n\n    Returns the median of the array elements.\n\n    Args:\n        a (cupy.ndarray): Array to compute the median.\n        axis (int, sequence of int or None): Axis along which the medians are\n            computed. The flattened array is used by default.\n        out (cupy.ndarray): Output array.\n        overwrite_input (bool): If ``True``, then allow use of memory of input\n            array a for calculations. The input array will be modified by the\n            call to median. This will save memory when you do not need to\n            preserve the contents of the input array. Treat the input as\n            undefined, but it will probably be fully or partially sorted.\n            Default is ``False``. If ``overwrite_input`` is ``True`` and ``a``\n            is not already an ndarray, an error will be raised.\n        keepdims (bool): If ``True``, the axis is remained as an axis of size\n            one.\n\n    Returns:\n        cupy.ndarray: The median of ``a``, along the axis if specified.\n\n    .. seealso:: :func:`numpy.nanmedian`\n\n    '
    if a.dtype.char in 'efdFD':
        return _statistics._nanmedian(a, axis, out, overwrite_input, keepdims)
    else:
        return median(a, axis=axis, out=out, overwrite_input=overwrite_input, keepdims=keepdims)

def average(a, axis=None, weights=None, returned=False, *, keepdims=False):
    if False:
        i = 10
        return i + 15
    'Returns the weighted average along an axis.\n\n    Args:\n        a (cupy.ndarray): Array to compute average.\n        axis (int): Along which axis to compute average. The flattened array\n            is used by default.\n        weights (cupy.ndarray): Array of weights where each element\n            corresponds to the value in ``a``. If ``None``, all the values\n            in ``a`` have a weight equal to one.\n        returned (bool): If ``True``, a tuple of the average and the sum\n            of weights is returned, otherwise only the average is returned.\n        keepdims (bool): If ``True``, the axis is remained as an axis of size\n            one.\n\n    Returns:\n        cupy.ndarray or tuple of cupy.ndarray: The average of the input array\n        along the axis and the sum of weights.\n\n    .. warning::\n\n        This function may synchronize the device if ``weight`` is given.\n\n    .. seealso:: :func:`numpy.average`\n    '
    a = cupy.asarray(a)
    if weights is None:
        avg = a.mean(axis=axis, keepdims=keepdims)
        scl = avg.dtype.type(a.size / avg.size)
    else:
        wgt = cupy.asarray(weights)
        if issubclass(a.dtype.type, (numpy.integer, numpy.bool_)):
            result_dtype = functools.reduce(numpy.promote_types, (a.dtype, wgt.dtype, 'f8'))
        else:
            result_dtype = numpy.promote_types(a.dtype, wgt.dtype)
        if a.shape != wgt.shape:
            if axis is None:
                raise TypeError('Axis must be specified when shapes of a and weights differ.')
            if wgt.ndim != 1:
                raise TypeError('1D weights expected when shapes of a and weights differ.')
            if wgt.shape[0] != a.shape[axis]:
                raise ValueError('Length of weights not compatible with specified axis.')
            wgt = cupy.broadcast_to(wgt, (a.ndim - 1) * (1,) + wgt.shape)
            wgt = wgt.swapaxes(-1, axis)
        scl = wgt.sum(axis=axis, dtype=result_dtype, keepdims=keepdims)
        if cupy.any(scl == 0.0):
            raise ZeroDivisionError("Weights sum to zero, can't be normalized")
        avg = cupy.multiply(a, wgt, dtype=result_dtype).sum(axis, keepdims=keepdims) / scl
    if returned:
        if scl.shape != avg.shape:
            scl = cupy.broadcast_to(cupy.array(scl), avg.shape).copy()
        return (avg, scl)
    else:
        return avg

def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    if False:
        for i in range(10):
            print('nop')
    'Returns the arithmetic mean along an axis.\n\n    Args:\n        a (cupy.ndarray): Array to compute mean.\n        axis (int, sequence of int or None): Along which axis to compute mean.\n            The flattened array is used by default.\n        dtype: Data type specifier.\n        out (cupy.ndarray): Output array.\n        keepdims (bool): If ``True``, the axis is remained as an axis of\n            size one.\n\n    Returns:\n        cupy.ndarray: The mean of the input array along the axis.\n\n    .. seealso:: :func:`numpy.mean`\n\n    '
    return a.mean(axis=axis, dtype=dtype, out=out, keepdims=keepdims)

def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if False:
        print('Hello World!')
    'Returns the variance along an axis.\n\n    Args:\n        a (cupy.ndarray): Array to compute variance.\n        axis (int): Along which axis to compute variance. The flattened array\n            is used by default.\n        dtype: Data type specifier.\n        out (cupy.ndarray): Output array.\n        keepdims (bool): If ``True``, the axis is remained as an axis of\n            size one.\n\n    Returns:\n        cupy.ndarray: The variance of the input array along the axis.\n\n    .. seealso:: :func:`numpy.var`\n\n    '
    return a.var(axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)

def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if False:
        for i in range(10):
            print('nop')
    'Returns the standard deviation along an axis.\n\n    Args:\n        a (cupy.ndarray): Array to compute standard deviation.\n        axis (int): Along which axis to compute standard deviation. The\n            flattened array is used by default.\n        dtype: Data type specifier.\n        out (cupy.ndarray): Output array.\n        keepdims (bool): If ``True``, the axis is remained as an axis of\n            size one.\n\n    Returns:\n        cupy.ndarray: The standard deviation of the input array along the axis.\n\n    .. seealso:: :func:`numpy.std`\n\n    '
    return a.std(axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)

def nanmean(a, axis=None, dtype=None, out=None, keepdims=False):
    if False:
        return 10
    'Returns the arithmetic mean along an axis ignoring NaN values.\n\n    Args:\n        a (cupy.ndarray): Array to compute mean.\n        axis (int, sequence of int or None): Along which axis to compute mean.\n            The flattened array is used by default.\n        dtype: Data type specifier.\n        out (cupy.ndarray): Output array.\n        keepdims (bool): If ``True``, the axis is remained as an axis of\n            size one.\n\n    Returns:\n        cupy.ndarray: The mean of the input array along the axis ignoring NaNs.\n\n    .. seealso:: :func:`numpy.nanmean`\n\n    '
    if a.dtype.kind in 'biu':
        return a.mean(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    return _statistics._nanmean(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if False:
        while True:
            i = 10
    'Returns the variance along an axis ignoring NaN values.\n\n    Args:\n        a (cupy.ndarray): Array to compute variance.\n        axis (int): Along which axis to compute variance. The flattened array\n            is used by default.\n        dtype: Data type specifier.\n        out (cupy.ndarray): Output array.\n        keepdims (bool): If ``True``, the axis is remained as an axis of\n            size one.\n\n    Returns:\n        cupy.ndarray: The variance of the input array along the axis.\n\n    .. seealso:: :func:`numpy.nanvar`\n\n    '
    if a.dtype.kind in 'biu':
        return a.var(axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)
    return _statistics._nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)

def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if False:
        print('Hello World!')
    'Returns the standard deviation along an axis ignoring NaN values.\n\n    Args:\n        a (cupy.ndarray): Array to compute standard deviation.\n        axis (int): Along which axis to compute standard deviation. The\n            flattened array is used by default.\n        dtype: Data type specifier.\n        out (cupy.ndarray): Output array.\n        keepdims (bool): If ``True``, the axis is remained as an axis of\n            size one.\n\n    Returns:\n        cupy.ndarray: The standard deviation of the input array along the axis.\n\n    .. seealso:: :func:`numpy.nanstd`\n\n    '
    if a.dtype.kind in 'biu':
        return a.std(axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)
    return _statistics._nanstd(a, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)