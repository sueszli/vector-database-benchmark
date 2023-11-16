import warnings
import numpy
import cupy
from cupy._core import _routines_math as _math
from cupy._core import _fusion_thread_local
from cupy._core import internal

def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    if False:
        i = 10
        return i + 15
    'Returns the sum of an array along given axes.\n\n    Args:\n        a (cupy.ndarray): Array to take sum.\n        axis (int or sequence of ints): Axes along which the sum is taken.\n        dtype: Data type specifier.\n        out (cupy.ndarray): Output array.\n        keepdims (bool): If ``True``, the specified axes are remained as axes\n            of length one.\n\n    Returns:\n        cupy.ndarray: The result array.\n\n    .. seealso:: :func:`numpy.sum`\n\n    '
    if _fusion_thread_local.is_fusing():
        if keepdims:
            raise NotImplementedError('cupy.sum does not support `keepdims` in fusion yet.')
        if dtype is None:
            func = _math.sum_auto_dtype
        else:
            func = _math._sum_keep_dtype
        return _fusion_thread_local.call_reduction(func, a, axis=axis, dtype=dtype, out=out)
    return a.sum(axis, dtype, out, keepdims)

def prod(a, axis=None, dtype=None, out=None, keepdims=False):
    if False:
        print('Hello World!')
    'Returns the product of an array along given axes.\n\n    Args:\n        a (cupy.ndarray): Array to take product.\n        axis (int or sequence of ints): Axes along which the product is taken.\n        dtype: Data type specifier.\n        out (cupy.ndarray): Output array.\n        keepdims (bool): If ``True``, the specified axes are remained as axes\n            of length one.\n\n    Returns:\n        cupy.ndarray: The result array.\n\n    .. seealso:: :func:`numpy.prod`\n\n    '
    if _fusion_thread_local.is_fusing():
        if keepdims:
            raise NotImplementedError('cupy.prod does not support `keepdims` in fusion yet.')
        if dtype is None:
            func = _math._prod_auto_dtype
        else:
            func = _math._prod_keep_dtype
        return _fusion_thread_local.call_reduction(func, a, axis=axis, dtype=dtype, out=out)
    return a.prod(axis, dtype, out, keepdims)

def nansum(a, axis=None, dtype=None, out=None, keepdims=False):
    if False:
        return 10
    'Returns the sum of an array along given axes treating Not a Numbers\n    (NaNs) as zero.\n\n    Args:\n        a (cupy.ndarray): Array to take sum.\n        axis (int or sequence of ints): Axes along which the sum is taken.\n        dtype: Data type specifier.\n        out (cupy.ndarray): Output array.\n        keepdims (bool): If ``True``, the specified axes are remained as axes\n            of length one.\n\n    Returns:\n        cupy.ndarray: The result array.\n\n    .. seealso:: :func:`numpy.nansum`\n\n    '
    if _fusion_thread_local.is_fusing():
        if keepdims:
            raise NotImplementedError('cupy.nansum does not support `keepdims` in fusion yet.')
        if a.dtype.char in 'FD':
            func = _math._nansum_complex_dtype
        elif dtype is None:
            func = _math._nansum_auto_dtype
        else:
            func = _math._nansum_keep_dtype
        return _fusion_thread_local.call_reduction(func, a, axis=axis, dtype=dtype, out=out)
    return _math._nansum(a, axis, dtype, out, keepdims)

def nanprod(a, axis=None, dtype=None, out=None, keepdims=False):
    if False:
        i = 10
        return i + 15
    'Returns the product of an array along given axes treating Not a Numbers\n    (NaNs) as zero.\n\n    Args:\n        a (cupy.ndarray): Array to take product.\n        axis (int or sequence of ints): Axes along which the product is taken.\n        dtype: Data type specifier.\n        out (cupy.ndarray): Output array.\n        keepdims (bool): If ``True``, the specified axes are remained as axes\n            of length one.\n\n    Returns:\n        cupy.ndarray: The result array.\n\n    .. seealso:: :func:`numpy.nanprod`\n\n    '
    if _fusion_thread_local.is_fusing():
        if keepdims:
            raise NotImplementedError('cupy.nanprod does not support `keepdims` in fusion yet.')
        if dtype is None:
            func = _math._nanprod_auto_dtype
        else:
            func = _math._nanprod_keep_dtype
        return _fusion_thread_local.call_reduction(func, a, axis=axis, dtype=dtype, out=out)
    return _math._nanprod(a, axis, dtype, out, keepdims)

def cumsum(a, axis=None, dtype=None, out=None):
    if False:
        return 10
    'Returns the cumulative sum of an array along a given axis.\n\n    Args:\n        a (cupy.ndarray): Input array.\n        axis (int): Axis along which the cumulative sum is taken. If it is not\n            specified, the input is flattened.\n        dtype: Data type specifier.\n        out (cupy.ndarray): Output array.\n\n    Returns:\n        cupy.ndarray: The result array.\n\n    .. seealso:: :func:`numpy.cumsum`\n\n    '
    return _math.scan_core(a, axis, _math.scan_op.SCAN_SUM, dtype, out)

def cumprod(a, axis=None, dtype=None, out=None):
    if False:
        print('Hello World!')
    'Returns the cumulative product of an array along a given axis.\n\n    Args:\n        a (cupy.ndarray): Input array.\n        axis (int): Axis along which the cumulative product is taken. If it is\n            not specified, the input is flattened.\n        dtype: Data type specifier.\n        out (cupy.ndarray): Output array.\n\n    Returns:\n        cupy.ndarray: The result array.\n\n    .. seealso:: :func:`numpy.cumprod`\n\n    '
    return _math.scan_core(a, axis, _math.scan_op.SCAN_PROD, dtype, out)

def nancumsum(a, axis=None, dtype=None, out=None):
    if False:
        return 10
    'Returns the cumulative sum of an array along a given axis treating Not a\n    Numbers (NaNs) as zero.\n\n    Args:\n        a (cupy.ndarray): Input array.\n        axis (int): Axis along which the cumulative sum is taken. If it is not\n            specified, the input is flattened.\n        dtype: Data type specifier.\n        out (cupy.ndarray): Output array.\n\n    Returns:\n        cupy.ndarray: The result array.\n\n    .. seealso:: :func:`numpy.nancumsum`\n    '
    a = _replace_nan(a, 0, out=out)
    return cumsum(a, axis=axis, dtype=dtype, out=out)

def nancumprod(a, axis=None, dtype=None, out=None):
    if False:
        return 10
    'Returns the cumulative product of an array along a given axis treating\n    Not a Numbers (NaNs) as one.\n\n    Args:\n        a (cupy.ndarray): Input array.\n        axis (int): Axis along which the cumulative product is taken. If it is\n            not specified, the input is flattened.\n        dtype: Data type specifier.\n        out (cupy.ndarray): Output array.\n\n    Returns:\n        cupy.ndarray: The result array.\n\n    .. seealso:: :func:`numpy.nancumprod`\n    '
    a = _replace_nan(a, 1, out=out)
    return cumprod(a, axis=axis, dtype=dtype, out=out)
_replace_nan_kernel = cupy._core._kernel.ElementwiseKernel('T a, T val', 'T out', 'if (a == a) {out = a;} else {out = val;}', 'cupy_replace_nan')

def _replace_nan(a, val, out=None):
    if False:
        for i in range(10):
            print('nop')
    if out is None or a.dtype != out.dtype:
        out = cupy.empty_like(a)
    _replace_nan_kernel(a, val, out)
    return out

def diff(a, n=1, axis=-1, prepend=None, append=None):
    if False:
        print('Hello World!')
    'Calculate the n-th discrete difference along the given axis.\n\n    Args:\n        a (cupy.ndarray): Input array.\n        n (int): The number of times values are differenced. If zero, the input\n            is returned as-is.\n        axis (int): The axis along which the difference is taken, default is\n            the last axis.\n        prepend (int, float, cupy.ndarray): Value to prepend to ``a``.\n        append (int, float, cupy.ndarray): Value to append to ``a``.\n\n    Returns:\n        cupy.ndarray: The result array.\n\n    .. seealso:: :func:`numpy.diff`\n    '
    if n == 0:
        return a
    if n < 0:
        raise ValueError('order must be non-negative but got ' + repr(n))
    a = cupy.asanyarray(a)
    nd = a.ndim
    axis = internal._normalize_axis_index(axis, nd)
    combined = []
    if prepend is not None:
        prepend = cupy.asanyarray(prepend)
        if prepend.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            prepend = cupy.broadcast_to(prepend, tuple(shape))
        combined.append(prepend)
    combined.append(a)
    if append is not None:
        append = cupy.asanyarray(append)
        if append.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            append = cupy.broadcast_to(append, tuple(shape))
        combined.append(append)
    if len(combined) > 1:
        a = cupy.concatenate(combined, axis)
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)
    op = cupy.not_equal if a.dtype == numpy.bool_ else cupy.subtract
    for _ in range(n):
        a = op(a[slice1], a[slice2])
    return a

def gradient(f, *varargs, axis=None, edge_order=1):
    if False:
        for i in range(10):
            print('nop')
    'Return the gradient of an N-dimensional array.\n\n    The gradient is computed using second order accurate central differences\n    in the interior points and either first or second order accurate one-sides\n    (forward or backwards) differences at the boundaries.\n    The returned gradient hence has the same shape as the input array.\n\n    Args:\n        f (cupy.ndarray): An N-dimensional array containing samples of a scalar\n            function.\n        varargs (list of scalar or array, optional): Spacing between f values.\n            Default unitary spacing for all dimensions. Spacing can be\n            specified using:\n\n            1. single scalar to specify a sample distance for all dimensions.\n            2. N scalars to specify a constant sample distance for each\n               dimension. i.e. `dx`, `dy`, `dz`, ...\n            3. N arrays to specify the coordinates of the values along each\n               dimension of F. The length of the array must match the size of\n               the corresponding dimension\n            4. Any combination of N scalars/arrays with the meaning of 2. and\n               3.\n\n            If `axis` is given, the number of varargs must equal the number of\n            axes. Default: 1.\n        edge_order ({1, 2}, optional): The gradient is calculated using N-th\n            order accurate differences at the boundaries. Default: 1.\n        axis (None or int or tuple of ints, optional): The gradient is\n            calculated only along the given axis or axes. The default\n            (axis = None) is to calculate the gradient for all the axes of the\n            input array. axis may be negative, in which case it counts from the\n            last to the first axis.\n\n    Returns:\n        gradient (cupy.ndarray or list of cupy.ndarray): A set of ndarrays\n        (or a single ndarray if there is only one dimension) corresponding\n        to the derivatives of f with respect to each dimension. Each\n        derivative has the same shape as f.\n\n    .. seealso:: :func:`numpy.gradient`\n    '
    f = cupy.asanyarray(f)
    ndim = f.ndim
    axes = internal._normalize_axis_indices(axis, ndim, sort_axes=False)
    len_axes = len(axes)
    n = len(varargs)
    if n == 0:
        dx = [1.0] * len_axes
    elif n == 1 and cupy.ndim(varargs[0]) == 0:
        dx = varargs * len_axes
    elif n == len_axes:
        dx = list(varargs)
        for (i, distances) in enumerate(dx):
            if cupy.ndim(distances) == 0:
                continue
            elif cupy.ndim(distances) != 1:
                raise ValueError('distances must be either scalars or 1d')
            if len(distances) != f.shape[axes[i]]:
                raise ValueError('when 1d, distances must match the length of the corresponding dimension')
            if numpy.issubdtype(distances.dtype, numpy.integer):
                distances = distances.astype(numpy.float64)
            diffx = cupy.diff(distances)
            if (diffx == diffx[0]).all():
                diffx = diffx[0]
            dx[i] = diffx
    else:
        raise TypeError('invalid number of arguments')
    if edge_order > 2:
        raise ValueError("'edge_order' greater than 2 not supported")
    outvals = []
    slice1 = [slice(None)] * ndim
    slice2 = [slice(None)] * ndim
    slice3 = [slice(None)] * ndim
    slice4 = [slice(None)] * ndim
    otype = f.dtype
    if numpy.issubdtype(otype, numpy.inexact):
        pass
    else:
        if numpy.issubdtype(otype, numpy.integer):
            f = f.astype(numpy.float64)
        otype = numpy.float64
    for (axis, ax_dx) in zip(axes, dx):
        if f.shape[axis] < edge_order + 1:
            raise ValueError('Shape of array too small to calculate a numerical gradient, at least (edge_order + 1) elements are required.')
        out = cupy.empty_like(f, dtype=otype)
        uniform_spacing = cupy.ndim(ax_dx) == 0
        slice1[axis] = slice(1, -1)
        slice2[axis] = slice(None, -2)
        slice3[axis] = slice(1, -1)
        slice4[axis] = slice(2, None)
        if uniform_spacing:
            out[tuple(slice1)] = (f[tuple(slice4)] - f[tuple(slice2)]) / (2.0 * ax_dx)
        else:
            dx1 = ax_dx[0:-1]
            dx2 = ax_dx[1:]
            dx_sum = dx1 + dx2
            a = -dx2 / (dx1 * dx_sum)
            b = (dx2 - dx1) / (dx1 * dx2)
            c = dx1 / (dx2 * dx_sum)
            shape = [1] * ndim
            shape[axis] = -1
            a.shape = b.shape = c.shape = tuple(shape)
            out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
        if edge_order == 1:
            slice1[axis] = 0
            slice2[axis] = 1
            slice3[axis] = 0
            dx_0 = ax_dx if uniform_spacing else ax_dx[0]
            out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_0
            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2
            dx_n = ax_dx if uniform_spacing else ax_dx[-1]
            out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_n
        else:
            slice1[axis] = 0
            slice2[axis] = 0
            slice3[axis] = 1
            slice4[axis] = 2
            if uniform_spacing:
                a = -1.5 / ax_dx
                b = 2.0 / ax_dx
                c = -0.5 / ax_dx
            else:
                dx1 = ax_dx[0]
                dx2 = ax_dx[1]
                dx_sum = dx1 + dx2
                a = -(2.0 * dx1 + dx2) / (dx1 * dx_sum)
                b = dx_sum / (dx1 * dx2)
                c = -dx1 / (dx2 * dx_sum)
            out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
            slice1[axis] = -1
            slice2[axis] = -3
            slice3[axis] = -2
            slice4[axis] = -1
            if uniform_spacing:
                a = 0.5 / ax_dx
                b = -2.0 / ax_dx
                c = 1.5 / ax_dx
            else:
                dx1 = ax_dx[-2]
                dx2 = ax_dx[-1]
                dx_sum = dx1 + dx2
                a = dx2 / (dx1 * dx_sum)
                b = -dx_sum / (dx1 * dx2)
                c = (2.0 * dx2 + dx1) / (dx2 * dx_sum)
            out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
        outvals.append(out)
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)
    if len_axes == 1:
        return outvals[0]
    else:
        return outvals

def ediff1d(arr, to_end=None, to_begin=None):
    if False:
        while True:
            i = 10
    '\n    Calculates the difference between consecutive elements of an array.\n\n    Args:\n        arr (cupy.ndarray): Input array.\n        to_end (cupy.ndarray, optional): Numbers to append at the end\n            of the returend differences.\n        to_begin (cupy.ndarray, optional): Numbers to prepend at the\n            beginning of the returned differences.\n\n    Returns:\n        cupy.ndarray: New array consisting differences among succeeding\n        elements.\n\n    .. seealso:: :func:`numpy.ediff1d`\n    '
    if not isinstance(arr, cupy.ndarray):
        raise TypeError('`arr` should be of type cupy.ndarray')
    arr = arr.ravel()
    dtype_req = arr.dtype
    if to_begin is None and to_end is None:
        return arr[1:] - arr[:-1]
    if to_begin is None:
        l_begin = 0
    else:
        if not isinstance(to_begin, cupy.ndarray):
            raise TypeError('`to_begin` should be of type cupy.ndarray')
        if not cupy.can_cast(to_begin, dtype_req, casting='same_kind'):
            raise TypeError('dtype of `to_begin` must be compatible with input `arr` under the `same_kind` rule.')
        to_begin = to_begin.ravel()
        l_begin = len(to_begin)
    if to_end is None:
        l_end = 0
    else:
        if not isinstance(to_end, cupy.ndarray):
            raise TypeError('`to_end` should be of type cupy.ndarray')
        if not cupy.can_cast(to_end, dtype_req, casting='same_kind'):
            raise TypeError('dtype of `to_end` must be compatible with input `arr` under the `same_kind` rule.')
        to_end = to_end.ravel()
        l_end = len(to_end)
    l_diff = max(len(arr) - 1, 0)
    result = cupy.empty(l_diff + l_begin + l_end, dtype=arr.dtype)
    if l_begin > 0:
        result[:l_begin] = to_begin
    if l_end > 0:
        result[l_begin + l_diff:] = to_end
    cupy.subtract(arr[1:], arr[:-1], result[l_begin:l_begin + l_diff])
    return result

def trapz(y, x=None, dx=1.0, axis=-1):
    if False:
        print('Hello World!')
    '\n    Integrate along the given axis using the composite trapezoidal rule.\n    Integrate `y` (`x`) along the given axis.\n\n    Args:\n        y (cupy.ndarray): Input array to integrate.\n        x (cupy.ndarray): Sample points over which to integrate. If None equal\n            spacing `dx` is assumed.\n        dx (float): Spacing between sample points, used if `x` is None, default\n            is 1.\n        axis (int): The axis along which the integral is taken, default is\n            the last axis.\n\n    Returns:\n        cupy.ndarray: Definite integral as approximated by the trapezoidal\n        rule.\n\n    .. seealso:: :func:`numpy.trapz`\n    '
    if not isinstance(y, cupy.ndarray):
        raise TypeError('`y` should be of type cupy.ndarray')
    if x is None:
        d = dx
    else:
        if not isinstance(x, cupy.ndarray):
            raise TypeError('`x` should be of type cupy.ndarray')
        if x.ndim == 1:
            d = diff(x)
            shape = [1] * y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = diff(x, axis=axis)
    nd = y.ndim
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    product = d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0
    try:
        ret = product.sum(axis)
    except ValueError:
        ret = cupy.add.reduce(product, axis)
    return ret

def product(a, axis=None, dtype=None, out=None, keepdims=False):
    if False:
        print('Hello World!')
    warnings.warn('Please use `prod` instead.', DeprecationWarning)
    return prod(a, axis, dtype, out, keepdims)

def cumproduct(a, axis=None, dtype=None, out=None):
    if False:
        for i in range(10):
            print('nop')
    warnings.warn('Please use `cumprod` instead.', DeprecationWarning)
    return cumprod(a, axis, dtype, out)