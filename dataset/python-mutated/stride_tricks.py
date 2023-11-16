import cupy as _cupy
import numpy as np

def as_strided(x, shape=None, strides=None):
    if False:
        return 10
    '\n    Create a view into the array with the given shape and strides.\n\n    .. warning:: This function has to be used with extreme care, see notes.\n\n    Parameters\n    ----------\n    x : ndarray\n        Array to create a new.\n    shape : sequence of int, optional\n        The shape of the new array. Defaults to ``x.shape``.\n    strides : sequence of int, optional\n        The strides of the new array. Defaults to ``x.strides``.\n\n    Returns\n    -------\n    view : ndarray\n\n    See also\n    --------\n    numpy.lib.stride_tricks.as_strided\n    reshape : reshape an array.\n\n    Notes\n    -----\n    ``as_strided`` creates a view into the array given the exact strides\n    and shape. This means it manipulates the internal data structure of\n    ndarray and, if done incorrectly, the array elements can point to\n    invalid memory and can corrupt results or crash your program.\n    '
    shape = x.shape if shape is None else tuple(shape)
    strides = x.strides if strides is None else tuple(strides)
    return _cupy.ndarray(shape=shape, dtype=x.dtype, memptr=x.data, strides=strides)

def sliding_window_view(x, window_shape, axis=None, *, subok=False, writeable=False):
    if False:
        print('Hello World!')
    '\n    Create a sliding window view into the array with the given window shape.\n\n    Also known as rolling or moving window, the window slides across all\n    dimensions of the array and extracts subsets of the array at all window\n    positions.\n\n\n    Parameters\n    ----------\n    x : array_like\n        Array to create the sliding window view from.\n    window_shape : int or tuple of int\n        Size of window over each axis that takes part in the sliding window.\n        If `axis` is not present, must have same length as the number of input\n        array dimensions. Single integers `i` are treated as if they were the\n        tuple `(i,)`.\n    axis : int or tuple of int, optional\n        Axis or axes along which the sliding window is applied.\n        By default, the sliding window is applied to all axes and\n        `window_shape[i]` will refer to axis `i` of `x`.\n        If `axis` is given as a `tuple of int`, `window_shape[i]` will refer to\n        the axis `axis[i]` of `x`.\n        Single integers `i` are treated as if they were the tuple `(i,)`.\n    subok : bool, optional\n        If True, sub-classes will be passed-through, otherwise the returned\n        array will be forced to be a base-class array (default).\n    writeable : bool, optional -- not supported\n        When true, allow writing to the returned view. The default is false,\n        as this should be used with caution: the returned view contains the\n        same memory location multiple times, so writing to one location will\n        cause others to change.\n\n    Returns\n    -------\n    view : ndarray\n        Sliding window view of the array. The sliding window dimensions are\n        inserted at the end, and the original dimensions are trimmed as\n        required by the size of the sliding window.\n        That is, ``view.shape = x_shape_trimmed + window_shape``, where\n        ``x_shape_trimmed`` is ``x.shape`` with every entry reduced by one less\n        than the corresponding window size.\n\n\n    See also\n    --------\n    numpy.lib.stride_tricks.as_strided\n\n    Notes\n    --------\n    This function is adapted from numpy.lib.stride_tricks.as_strided.\n\n    Examples\n    --------\n    >>> x = _cupy.arange(6)\n    >>> x.shape\n    (6,)\n    >>> v = sliding_window_view(x, 3)\n    >>> v.shape\n    (4, 3)\n    >>> v\n    array([[0, 1, 2],\n           [1, 2, 3],\n           [2, 3, 4],\n           [3, 4, 5]])\n\n    '
    window_shape = tuple(window_shape) if np.iterable(window_shape) else (window_shape,)
    if writeable:
        raise NotImplementedError('Writeable views are not supported.')
    x = _cupy.array(x, copy=False, subok=subok)
    window_shape_array = _cupy.array(window_shape)
    for dim in window_shape_array:
        if dim < 0:
            raise ValueError('`window_shape` cannot contain negative values')
    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(f'Since axis is `None`, must provide window_shape for all dimensions of `x`; got {len(window_shape)} window_shape elements and `x.ndim` is {x.ndim}.')
    else:
        axis = _cupy._core.internal._normalize_axis_indices(axis, x.ndim)
        if len(window_shape) != len(axis):
            raise ValueError(f'Must provide matching length window_shape and axis; got {len(window_shape)} window_shape elements and {len(axis)} axes elements.')
    out_strides = x.strides + tuple((x.strides[ax] for ax in axis))
    x_shape_trimmed = list(x.shape)
    for (ax, dim) in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError('window shape cannot be larger than input array shape')
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape
    return as_strided(x, strides=out_strides, shape=out_shape)