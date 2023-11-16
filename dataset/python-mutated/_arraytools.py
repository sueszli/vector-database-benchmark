"""
Functions for acting on a axis of an array.
"""
import cupy

def axis_slice(a, start=None, stop=None, step=None, axis=-1):
    if False:
        i = 10
        return i + 15
    "Take a slice along axis 'axis' from 'a'.\n\n    Parameters\n    ----------\n    a : cupy.ndarray\n        The array to be sliced.\n    start, stop, step : int or None\n        The slice parameters.\n    axis : int, optional\n        The axis of `a` to be sliced.\n\n    Examples\n    --------\n    >>> a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\n    >>> axis_slice(a, start=0, stop=1, axis=1)\n    array([[1],\n           [4],\n           [7]])\n    >>> axis_slice(a, start=1, axis=0)\n    array([[4, 5, 6],\n           [7, 8, 9]])\n\n    Notes\n    -----\n    The keyword arguments start, stop and step are used by calling\n    slice(start, stop, step). This implies axis_slice() does not\n    handle its arguments the exactly the same as indexing. To select\n    a single index k, for example, use\n        axis_slice(a, start=k, stop=k+1)\n    In this case, the length of the axis 'axis' in the result will\n    be 1; the trivial dimension is not removed. (Use cupy.squeeze()\n    to remove trivial axes.)\n    "
    a_slice = [slice(None)] * a.ndim
    a_slice[axis] = slice(start, stop, step)
    b = a[tuple(a_slice)]
    return b

def axis_assign(a, b, start=None, stop=None, step=None, axis=-1):
    if False:
        i = 10
        return i + 15
    "Take a slice along axis 'axis' from 'a' and set it to 'b' in-place.\n\n    Parameters\n    ----------\n    a : numpy.ndarray\n        The array to be sliced.\n    b : cupy.ndarray\n        The array to be assigned.\n    start, stop, step : int or None\n        The slice parameters.\n    axis : int, optional\n        The axis of `a` to be sliced.\n\n    Examples\n    --------\n    >>> a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\n    >>> b1 = array([[-1], [-4], [-7]])\n    >>> axis_assign(a, b1, start=0, stop=1, axis=1)\n    array([[-1, 2, 3],\n           [-4, 5, 6],\n           [-7, 8, 9]])\n\n    Notes\n    -----\n    The keyword arguments start, stop and step are used by calling\n    slice(start, stop, step). This implies axis_assign() does not\n    handle its arguments the exactly the same as indexing. To assign\n    a single index k, for example, use\n        axis_assign(a, start=k, stop=k+1)\n    In this case, the length of the axis 'axis' in the result will\n    be 1; the trivial dimension is not removed. (Use numpy.squeeze()\n    to remove trivial axes.)\n\n    This function works in-place and will modify the values contained in `a`\n    "
    a_slice = [slice(None)] * a.ndim
    a_slice[axis] = slice(start, stop, step)
    a[tuple(a_slice)] = b
    return a

def axis_reverse(a, axis=-1):
    if False:
        while True:
            i = 10
    'Reverse the 1-D slices of `a` along axis `axis`.\n\n    Returns axis_slice(a, step=-1, axis=axis).\n    '
    return axis_slice(a, step=-1, axis=axis)

def odd_ext(x, n, axis=-1):
    if False:
        for i in range(10):
            print('nop')
    '\n    Odd extension at the boundaries of an array\n\n    Generate a new ndarray by making an odd extension of `x` along an axis.\n\n    Parameters\n    ----------\n    x : ndarray\n        The array to be extended.\n    n : int\n        The number of elements by which to extend `x` at each end of the axis.\n    axis : int, optional\n        The axis along which to extend `x`. Default is -1.\n\n    Examples\n    --------\n    >>> from cupyx.scipy.signal._arraytools import odd_ext\n    >>> a = cupy.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])\n    >>> odd_ext(a, 2)\n    array([[-1,  0,  1,  2,  3,  4,  5,  6,  7],\n           [-4, -1,  0,  1,  4,  9, 16, 23, 28]])\n    '
    if n < 1:
        return x
    if n > x.shape[axis] - 1:
        raise ValueError(('The extension length n (%d) is too big. ' + 'It must not exceed x.shape[axis]-1, which is %d.') % (n, x.shape[axis] - 1))
    left_end = axis_slice(x, start=0, stop=1, axis=axis)
    left_ext = axis_slice(x, start=n, stop=0, step=-1, axis=axis)
    right_end = axis_slice(x, start=-1, axis=axis)
    right_ext = axis_slice(x, start=-2, stop=-(n + 2), step=-1, axis=axis)
    ext = cupy.concatenate((2 * left_end - left_ext, x, 2 * right_end - right_ext), axis=axis)
    return ext

def even_ext(x, n, axis=-1):
    if False:
        print('Hello World!')
    '\n    Even extension at the boundaries of an array\n\n    Generate a new ndarray by making an even extension of `x` along an axis.\n\n    Parameters\n    ----------\n    x : ndarray\n        The array to be extended.\n    n : int\n        The number of elements by which to extend `x` at each end of the axis.\n    axis : int, optional\n        The axis along which to extend `x`. Default is -1.\n\n    Examples\n    --------\n    >>> from cupyx.scipy.signal._arraytools import even_ext\n    >>> a = cupy.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])\n    >>> even_ext(a, 2)\n    array([[ 3,  2,  1,  2,  3,  4,  5,  4,  3],\n           [ 4,  1,  0,  1,  4,  9, 16,  9,  4]])\n\n    '
    if n < 1:
        return x
    if n > x.shape[axis] - 1:
        raise ValueError(('The extension length n (%d) is too big. ' + 'It must not exceed x.shape[axis]-1, which is %d.') % (n, x.shape[axis] - 1))
    left_ext = axis_slice(x, start=n, stop=0, step=-1, axis=axis)
    right_ext = axis_slice(x, start=-2, stop=-(n + 2), step=-1, axis=axis)
    ext = cupy.concatenate((left_ext, x, right_ext), axis=axis)
    return ext

def const_ext(x, n, axis=-1):
    if False:
        return 10
    '\n    Constant extension at the boundaries of an array\n\n    Generate a new ndarray that is a constant extension of `x` along an axis.\n    The extension repeats the values at the first and last element of\n    the axis.\n\n    Parameters\n    ----------\n    x : ndarray\n        The array to be extended.\n    n : int\n        The number of elements by which to extend `x` at each end of the axis.\n    axis : int, optional\n        The axis along which to extend `x`. Default is -1.\n\n    Examples\n    --------\n    >>> from cupyx.scipy.signal._arraytools import const_ext\n    >>> a = cupy.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])\n    >>> const_ext(a, 2)\n    array([[ 1,  1,  1,  2,  3,  4,  5,  5,  5],\n           [ 0,  0,  0,  1,  4,  9, 16, 16, 16]])\n    '
    if n < 1:
        return x
    left_end = axis_slice(x, start=0, stop=1, axis=axis)
    ones_shape = [1] * x.ndim
    ones_shape[axis] = n
    ones = cupy.ones(ones_shape, dtype=x.dtype)
    left_ext = ones * left_end
    right_end = axis_slice(x, start=-1, axis=axis)
    right_ext = ones * right_end
    ext = cupy.concatenate((left_ext, x, right_ext), axis=axis)
    return ext

def zero_ext(x, n, axis=-1):
    if False:
        print('Hello World!')
    '\n    Zero padding at the boundaries of an array\n\n    Generate a new ndarray that is a zero-padded extension of `x` along\n    an axis.\n\n    Parameters\n    ----------\n    x : ndarray\n        The array to be extended.\n    n : int\n        The number of elements by which to extend `x` at each end of the\n        axis.\n    axis : int, optional\n        The axis along which to extend `x`. Default is -1.\n\n    Examples\n    --------\n    >>> from cupyx.scipy.signal._arraytools import zero_ext\n    >>> a = cupy.array([[1, 2, 3, 4, 5], [0, 1, 4, 9, 16]])\n    >>> zero_ext(a, 2)\n    array([[ 0,  0,  1,  2,  3,  4,  5,  0,  0],\n           [ 0,  0,  0,  1,  4,  9, 16,  0,  0]])\n    '
    if n < 1:
        return x
    zeros_shape = list(x.shape)
    zeros_shape[axis] = n
    zeros = cupy.zeros(zeros_shape, dtype=x.dtype)
    ext = cupy.concatenate((zeros, x, zeros), axis=axis)
    return ext

def _as_strided(x, shape=None, strides=None):
    if False:
        return 10
    '\n    Create a view into the array with the given shape and strides.\n    .. warning:: This function has to be used with extreme care, see notes.\n\n    Parameters\n    ----------\n    x : ndarray\n        Array to create a new.\n    shape : sequence of int, optional\n        The shape of the new array. Defaults to ``x.shape``.\n    strides : sequence of int, optional\n        The strides of the new array. Defaults to ``x.strides``.\n\n    Returns\n    -------\n    view : ndarray\n\n    Notes\n    -----\n    ``as_strided`` creates a view into the array given the exact strides\n    and shape. This means it manipulates the internal data structure of\n    ndarray and, if done incorrectly, the array elements can point to\n    invalid memory and can corrupt results or crash your program.\n    '
    shape = x.shape if shape is None else tuple(shape)
    strides = x.strides if strides is None else tuple(strides)
    return cupy.ndarray(shape=shape, dtype=x.dtype, memptr=x.data, strides=strides)