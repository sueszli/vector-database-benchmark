"""
The arraypad module contains a group of functions to pad values onto the edges
of an n-dimensional array.

"""
import numpy as np
from numpy._core.overrides import array_function_dispatch
from numpy.lib._index_tricks_impl import ndindex
__all__ = ['pad']

def _round_if_needed(arr, dtype):
    if False:
        return 10
    '\n    Rounds arr inplace if destination dtype is integer.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array.\n    dtype : dtype\n        The dtype of the destination array.\n    '
    if np.issubdtype(dtype, np.integer):
        arr.round(out=arr)

def _slice_at_axis(sl, axis):
    if False:
        i = 10
        return i + 15
    '\n    Construct tuple of slices to slice an array in the given dimension.\n\n    Parameters\n    ----------\n    sl : slice\n        The slice for the given dimension.\n    axis : int\n        The axis to which `sl` is applied. All other dimensions are left\n        "unsliced".\n\n    Returns\n    -------\n    sl : tuple of slices\n        A tuple with slices matching `shape` in length.\n\n    Examples\n    --------\n    >>> _slice_at_axis(slice(None, 3, -1), 1)\n    (slice(None, None, None), slice(None, 3, -1), (...,))\n    '
    return (slice(None),) * axis + (sl,) + (...,)

def _view_roi(array, original_area_slice, axis):
    if False:
        print('Hello World!')
    '\n    Get a view of the current region of interest during iterative padding.\n\n    When padding multiple dimensions iteratively corner values are\n    unnecessarily overwritten multiple times. This function reduces the\n    working area for the first dimensions so that corners are excluded.\n\n    Parameters\n    ----------\n    array : ndarray\n        The array with the region of interest.\n    original_area_slice : tuple of slices\n        Denotes the area with original values of the unpadded array.\n    axis : int\n        The currently padded dimension assuming that `axis` is padded before\n        `axis` + 1.\n\n    Returns\n    -------\n    roi : ndarray\n        The region of interest of the original `array`.\n    '
    axis += 1
    sl = (slice(None),) * axis + original_area_slice[axis:]
    return array[sl]

def _pad_simple(array, pad_width, fill_value=None):
    if False:
        print('Hello World!')
    '\n    Pad array on all sides with either a single value or undefined values.\n\n    Parameters\n    ----------\n    array : ndarray\n        Array to grow.\n    pad_width : sequence of tuple[int, int]\n        Pad width on both sides for each dimension in `arr`.\n    fill_value : scalar, optional\n        If provided the padded area is filled with this value, otherwise\n        the pad area left undefined.\n\n    Returns\n    -------\n    padded : ndarray\n        The padded array with the same dtype as`array`. Its order will default\n        to C-style if `array` is not F-contiguous.\n    original_area_slice : tuple\n        A tuple of slices pointing to the area of the original array.\n    '
    new_shape = tuple((left + size + right for (size, (left, right)) in zip(array.shape, pad_width)))
    order = 'F' if array.flags.fnc else 'C'
    padded = np.empty(new_shape, dtype=array.dtype, order=order)
    if fill_value is not None:
        padded.fill(fill_value)
    original_area_slice = tuple((slice(left, left + size) for (size, (left, right)) in zip(array.shape, pad_width)))
    padded[original_area_slice] = array
    return (padded, original_area_slice)

def _set_pad_area(padded, axis, width_pair, value_pair):
    if False:
        while True:
            i = 10
    '\n    Set empty-padded area in given dimension.\n\n    Parameters\n    ----------\n    padded : ndarray\n        Array with the pad area which is modified inplace.\n    axis : int\n        Dimension with the pad area to set.\n    width_pair : (int, int)\n        Pair of widths that mark the pad area on both sides in the given\n        dimension.\n    value_pair : tuple of scalars or ndarrays\n        Values inserted into the pad area on each side. It must match or be\n        broadcastable to the shape of `arr`.\n    '
    left_slice = _slice_at_axis(slice(None, width_pair[0]), axis)
    padded[left_slice] = value_pair[0]
    right_slice = _slice_at_axis(slice(padded.shape[axis] - width_pair[1], None), axis)
    padded[right_slice] = value_pair[1]

def _get_edges(padded, axis, width_pair):
    if False:
        return 10
    '\n    Retrieve edge values from empty-padded array in given dimension.\n\n    Parameters\n    ----------\n    padded : ndarray\n        Empty-padded array.\n    axis : int\n        Dimension in which the edges are considered.\n    width_pair : (int, int)\n        Pair of widths that mark the pad area on both sides in the given\n        dimension.\n\n    Returns\n    -------\n    left_edge, right_edge : ndarray\n        Edge values of the valid area in `padded` in the given dimension. Its\n        shape will always match `padded` except for the dimension given by\n        `axis` which will have a length of 1.\n    '
    left_index = width_pair[0]
    left_slice = _slice_at_axis(slice(left_index, left_index + 1), axis)
    left_edge = padded[left_slice]
    right_index = padded.shape[axis] - width_pair[1]
    right_slice = _slice_at_axis(slice(right_index - 1, right_index), axis)
    right_edge = padded[right_slice]
    return (left_edge, right_edge)

def _get_linear_ramps(padded, axis, width_pair, end_value_pair):
    if False:
        print('Hello World!')
    '\n    Construct linear ramps for empty-padded array in given dimension.\n\n    Parameters\n    ----------\n    padded : ndarray\n        Empty-padded array.\n    axis : int\n        Dimension in which the ramps are constructed.\n    width_pair : (int, int)\n        Pair of widths that mark the pad area on both sides in the given\n        dimension.\n    end_value_pair : (scalar, scalar)\n        End values for the linear ramps which form the edge of the fully padded\n        array. These values are included in the linear ramps.\n\n    Returns\n    -------\n    left_ramp, right_ramp : ndarray\n        Linear ramps to set on both sides of `padded`.\n    '
    edge_pair = _get_edges(padded, axis, width_pair)
    (left_ramp, right_ramp) = (np.linspace(start=end_value, stop=edge.squeeze(axis), num=width, endpoint=False, dtype=padded.dtype, axis=axis) for (end_value, edge, width) in zip(end_value_pair, edge_pair, width_pair))
    right_ramp = right_ramp[_slice_at_axis(slice(None, None, -1), axis)]
    return (left_ramp, right_ramp)

def _get_stats(padded, axis, width_pair, length_pair, stat_func):
    if False:
        print('Hello World!')
    '\n    Calculate statistic for the empty-padded array in given dimension.\n\n    Parameters\n    ----------\n    padded : ndarray\n        Empty-padded array.\n    axis : int\n        Dimension in which the statistic is calculated.\n    width_pair : (int, int)\n        Pair of widths that mark the pad area on both sides in the given\n        dimension.\n    length_pair : 2-element sequence of None or int\n        Gives the number of values in valid area from each side that is\n        taken into account when calculating the statistic. If None the entire\n        valid area in `padded` is considered.\n    stat_func : function\n        Function to compute statistic. The expected signature is\n        ``stat_func(x: ndarray, axis: int, keepdims: bool) -> ndarray``.\n\n    Returns\n    -------\n    left_stat, right_stat : ndarray\n        Calculated statistic for both sides of `padded`.\n    '
    left_index = width_pair[0]
    right_index = padded.shape[axis] - width_pair[1]
    max_length = right_index - left_index
    (left_length, right_length) = length_pair
    if left_length is None or max_length < left_length:
        left_length = max_length
    if right_length is None or max_length < right_length:
        right_length = max_length
    if (left_length == 0 or right_length == 0) and stat_func in {np.amax, np.amin}:
        raise ValueError('stat_length of 0 yields no value for padding')
    left_slice = _slice_at_axis(slice(left_index, left_index + left_length), axis)
    left_chunk = padded[left_slice]
    left_stat = stat_func(left_chunk, axis=axis, keepdims=True)
    _round_if_needed(left_stat, padded.dtype)
    if left_length == right_length == max_length:
        return (left_stat, left_stat)
    right_slice = _slice_at_axis(slice(right_index - right_length, right_index), axis)
    right_chunk = padded[right_slice]
    right_stat = stat_func(right_chunk, axis=axis, keepdims=True)
    _round_if_needed(right_stat, padded.dtype)
    return (left_stat, right_stat)

def _set_reflect_both(padded, axis, width_pair, method, include_edge=False):
    if False:
        i = 10
        return i + 15
    "\n    Pad `axis` of `arr` with reflection.\n\n    Parameters\n    ----------\n    padded : ndarray\n        Input array of arbitrary shape.\n    axis : int\n        Axis along which to pad `arr`.\n    width_pair : (int, int)\n        Pair of widths that mark the pad area on both sides in the given\n        dimension.\n    method : str\n        Controls method of reflection; options are 'even' or 'odd'.\n    include_edge : bool\n        If true, edge value is included in reflection, otherwise the edge\n        value forms the symmetric axis to the reflection.\n\n    Returns\n    -------\n    pad_amt : tuple of ints, length 2\n        New index positions of padding to do along the `axis`. If these are\n        both 0, padding is done in this dimension.\n    "
    (left_pad, right_pad) = width_pair
    old_length = padded.shape[axis] - right_pad - left_pad
    if include_edge:
        edge_offset = 1
    else:
        edge_offset = 0
        old_length -= 1
    if left_pad > 0:
        chunk_length = min(old_length, left_pad)
        stop = left_pad - edge_offset
        start = stop + chunk_length
        left_slice = _slice_at_axis(slice(start, stop, -1), axis)
        left_chunk = padded[left_slice]
        if method == 'odd':
            edge_slice = _slice_at_axis(slice(left_pad, left_pad + 1), axis)
            left_chunk = 2 * padded[edge_slice] - left_chunk
        start = left_pad - chunk_length
        stop = left_pad
        pad_area = _slice_at_axis(slice(start, stop), axis)
        padded[pad_area] = left_chunk
        left_pad -= chunk_length
    if right_pad > 0:
        chunk_length = min(old_length, right_pad)
        start = -right_pad + edge_offset - 2
        stop = start - chunk_length
        right_slice = _slice_at_axis(slice(start, stop, -1), axis)
        right_chunk = padded[right_slice]
        if method == 'odd':
            edge_slice = _slice_at_axis(slice(-right_pad - 1, -right_pad), axis)
            right_chunk = 2 * padded[edge_slice] - right_chunk
        start = padded.shape[axis] - right_pad
        stop = start + chunk_length
        pad_area = _slice_at_axis(slice(start, stop), axis)
        padded[pad_area] = right_chunk
        right_pad -= chunk_length
    return (left_pad, right_pad)

def _set_wrap_both(padded, axis, width_pair, original_period):
    if False:
        while True:
            i = 10
    '\n    Pad `axis` of `arr` with wrapped values.\n\n    Parameters\n    ----------\n    padded : ndarray\n        Input array of arbitrary shape.\n    axis : int\n        Axis along which to pad `arr`.\n    width_pair : (int, int)\n        Pair of widths that mark the pad area on both sides in the given\n        dimension.\n    original_period : int\n        Original length of data on `axis` of `arr`.\n\n    Returns\n    -------\n    pad_amt : tuple of ints, length 2\n        New index positions of padding to do along the `axis`. If these are\n        both 0, padding is done in this dimension.\n    '
    (left_pad, right_pad) = width_pair
    period = padded.shape[axis] - right_pad - left_pad
    period = period // original_period * original_period
    new_left_pad = 0
    new_right_pad = 0
    if left_pad > 0:
        slice_end = left_pad + period
        slice_start = slice_end - min(period, left_pad)
        right_slice = _slice_at_axis(slice(slice_start, slice_end), axis)
        right_chunk = padded[right_slice]
        if left_pad > period:
            pad_area = _slice_at_axis(slice(left_pad - period, left_pad), axis)
            new_left_pad = left_pad - period
        else:
            pad_area = _slice_at_axis(slice(None, left_pad), axis)
        padded[pad_area] = right_chunk
    if right_pad > 0:
        slice_start = -right_pad - period
        slice_end = slice_start + min(period, right_pad)
        left_slice = _slice_at_axis(slice(slice_start, slice_end), axis)
        left_chunk = padded[left_slice]
        if right_pad > period:
            pad_area = _slice_at_axis(slice(-right_pad, -right_pad + period), axis)
            new_right_pad = right_pad - period
        else:
            pad_area = _slice_at_axis(slice(-right_pad, None), axis)
        padded[pad_area] = left_chunk
    return (new_left_pad, new_right_pad)

def _as_pairs(x, ndim, as_index=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Broadcast `x` to an array with the shape (`ndim`, 2).\n\n    A helper function for `pad` that prepares and validates arguments like\n    `pad_width` for iteration in pairs.\n\n    Parameters\n    ----------\n    x : {None, scalar, array-like}\n        The object to broadcast to the shape (`ndim`, 2).\n    ndim : int\n        Number of pairs the broadcasted `x` will have.\n    as_index : bool, optional\n        If `x` is not None, try to round each element of `x` to an integer\n        (dtype `np.intp`) and ensure every element is positive.\n\n    Returns\n    -------\n    pairs : nested iterables, shape (`ndim`, 2)\n        The broadcasted version of `x`.\n\n    Raises\n    ------\n    ValueError\n        If `as_index` is True and `x` contains negative elements.\n        Or if `x` is not broadcastable to the shape (`ndim`, 2).\n    '
    if x is None:
        return ((None, None),) * ndim
    x = np.array(x)
    if as_index:
        x = np.round(x).astype(np.intp, copy=False)
    if x.ndim < 3:
        if x.size == 1:
            x = x.ravel()
            if as_index and x < 0:
                raise ValueError("index can't contain negative values")
            return ((x[0], x[0]),) * ndim
        if x.size == 2 and x.shape != (2, 1):
            x = x.ravel()
            if as_index and (x[0] < 0 or x[1] < 0):
                raise ValueError("index can't contain negative values")
            return ((x[0], x[1]),) * ndim
    if as_index and x.min() < 0:
        raise ValueError("index can't contain negative values")
    return np.broadcast_to(x, (ndim, 2)).tolist()

def _pad_dispatcher(array, pad_width, mode=None, **kwargs):
    if False:
        while True:
            i = 10
    return (array,)

@array_function_dispatch(_pad_dispatcher, module='numpy')
def pad(array, pad_width, mode='constant', **kwargs):
    if False:
        while True:
            i = 10
    "\n    Pad an array.\n\n    Parameters\n    ----------\n    array : array_like of rank N\n        The array to pad.\n    pad_width : {sequence, array_like, int}\n        Number of values padded to the edges of each axis.\n        ``((before_1, after_1), ... (before_N, after_N))`` unique pad widths\n        for each axis.\n        ``(before, after)`` or ``((before, after),)`` yields same before\n        and after pad for each axis.\n        ``(pad,)`` or ``int`` is a shortcut for before = after = pad width\n        for all axes.\n    mode : str or function, optional\n        One of the following string values or a user supplied function.\n\n        'constant' (default)\n            Pads with a constant value.\n        'edge'\n            Pads with the edge values of array.\n        'linear_ramp'\n            Pads with the linear ramp between end_value and the\n            array edge value.\n        'maximum'\n            Pads with the maximum value of all or part of the\n            vector along each axis.\n        'mean'\n            Pads with the mean value of all or part of the\n            vector along each axis.\n        'median'\n            Pads with the median value of all or part of the\n            vector along each axis.\n        'minimum'\n            Pads with the minimum value of all or part of the\n            vector along each axis.\n        'reflect'\n            Pads with the reflection of the vector mirrored on\n            the first and last values of the vector along each\n            axis.\n        'symmetric'\n            Pads with the reflection of the vector mirrored\n            along the edge of the array.\n        'wrap'\n            Pads with the wrap of the vector along the axis.\n            The first values are used to pad the end and the\n            end values are used to pad the beginning.\n        'empty'\n            Pads with undefined values.\n\n            .. versionadded:: 1.17\n\n        <function>\n            Padding function, see Notes.\n    stat_length : sequence or int, optional\n        Used in 'maximum', 'mean', 'median', and 'minimum'.  Number of\n        values at edge of each axis used to calculate the statistic value.\n\n        ``((before_1, after_1), ... (before_N, after_N))`` unique statistic\n        lengths for each axis.\n\n        ``(before, after)`` or ``((before, after),)`` yields same before\n        and after statistic lengths for each axis.\n\n        ``(stat_length,)`` or ``int`` is a shortcut for\n        ``before = after = statistic`` length for all axes.\n\n        Default is ``None``, to use the entire axis.\n    constant_values : sequence or scalar, optional\n        Used in 'constant'.  The values to set the padded values for each\n        axis.\n\n        ``((before_1, after_1), ... (before_N, after_N))`` unique pad constants\n        for each axis.\n\n        ``(before, after)`` or ``((before, after),)`` yields same before\n        and after constants for each axis.\n\n        ``(constant,)`` or ``constant`` is a shortcut for\n        ``before = after = constant`` for all axes.\n\n        Default is 0.\n    end_values : sequence or scalar, optional\n        Used in 'linear_ramp'.  The values used for the ending value of the\n        linear_ramp and that will form the edge of the padded array.\n\n        ``((before_1, after_1), ... (before_N, after_N))`` unique end values\n        for each axis.\n\n        ``(before, after)`` or ``((before, after),)`` yields same before\n        and after end values for each axis.\n\n        ``(constant,)`` or ``constant`` is a shortcut for\n        ``before = after = constant`` for all axes.\n\n        Default is 0.\n    reflect_type : {'even', 'odd'}, optional\n        Used in 'reflect', and 'symmetric'.  The 'even' style is the\n        default with an unaltered reflection around the edge value.  For\n        the 'odd' style, the extended part of the array is created by\n        subtracting the reflected values from two times the edge value.\n\n    Returns\n    -------\n    pad : ndarray\n        Padded array of rank equal to `array` with shape increased\n        according to `pad_width`.\n\n    Notes\n    -----\n    .. versionadded:: 1.7.0\n\n    For an array with rank greater than 1, some of the padding of later\n    axes is calculated from padding of previous axes.  This is easiest to\n    think about with a rank 2 array where the corners of the padded array\n    are calculated by using padded values from the first axis.\n\n    The padding function, if used, should modify a rank 1 array in-place. It\n    has the following signature::\n\n        padding_func(vector, iaxis_pad_width, iaxis, kwargs)\n\n    where\n\n    vector : ndarray\n        A rank 1 array already padded with zeros.  Padded values are\n        vector[:iaxis_pad_width[0]] and vector[-iaxis_pad_width[1]:].\n    iaxis_pad_width : tuple\n        A 2-tuple of ints, iaxis_pad_width[0] represents the number of\n        values padded at the beginning of vector where\n        iaxis_pad_width[1] represents the number of values padded at\n        the end of vector.\n    iaxis : int\n        The axis currently being calculated.\n    kwargs : dict\n        Any keyword arguments the function requires.\n\n    Examples\n    --------\n    >>> a = [1, 2, 3, 4, 5]\n    >>> np.pad(a, (2, 3), 'constant', constant_values=(4, 6))\n    array([4, 4, 1, ..., 6, 6, 6])\n\n    >>> np.pad(a, (2, 3), 'edge')\n    array([1, 1, 1, ..., 5, 5, 5])\n\n    >>> np.pad(a, (2, 3), 'linear_ramp', end_values=(5, -4))\n    array([ 5,  3,  1,  2,  3,  4,  5,  2, -1, -4])\n\n    >>> np.pad(a, (2,), 'maximum')\n    array([5, 5, 1, 2, 3, 4, 5, 5, 5])\n\n    >>> np.pad(a, (2,), 'mean')\n    array([3, 3, 1, 2, 3, 4, 5, 3, 3])\n\n    >>> np.pad(a, (2,), 'median')\n    array([3, 3, 1, 2, 3, 4, 5, 3, 3])\n\n    >>> a = [[1, 2], [3, 4]]\n    >>> np.pad(a, ((3, 2), (2, 3)), 'minimum')\n    array([[1, 1, 1, 2, 1, 1, 1],\n           [1, 1, 1, 2, 1, 1, 1],\n           [1, 1, 1, 2, 1, 1, 1],\n           [1, 1, 1, 2, 1, 1, 1],\n           [3, 3, 3, 4, 3, 3, 3],\n           [1, 1, 1, 2, 1, 1, 1],\n           [1, 1, 1, 2, 1, 1, 1]])\n\n    >>> a = [1, 2, 3, 4, 5]\n    >>> np.pad(a, (2, 3), 'reflect')\n    array([3, 2, 1, 2, 3, 4, 5, 4, 3, 2])\n\n    >>> np.pad(a, (2, 3), 'reflect', reflect_type='odd')\n    array([-1,  0,  1,  2,  3,  4,  5,  6,  7,  8])\n\n    >>> np.pad(a, (2, 3), 'symmetric')\n    array([2, 1, 1, 2, 3, 4, 5, 5, 4, 3])\n\n    >>> np.pad(a, (2, 3), 'symmetric', reflect_type='odd')\n    array([0, 1, 1, 2, 3, 4, 5, 5, 6, 7])\n\n    >>> np.pad(a, (2, 3), 'wrap')\n    array([4, 5, 1, 2, 3, 4, 5, 1, 2, 3])\n\n    >>> def pad_with(vector, pad_width, iaxis, kwargs):\n    ...     pad_value = kwargs.get('padder', 10)\n    ...     vector[:pad_width[0]] = pad_value\n    ...     vector[-pad_width[1]:] = pad_value\n    >>> a = np.arange(6)\n    >>> a = a.reshape((2, 3))\n    >>> np.pad(a, 2, pad_with)\n    array([[10, 10, 10, 10, 10, 10, 10],\n           [10, 10, 10, 10, 10, 10, 10],\n           [10, 10,  0,  1,  2, 10, 10],\n           [10, 10,  3,  4,  5, 10, 10],\n           [10, 10, 10, 10, 10, 10, 10],\n           [10, 10, 10, 10, 10, 10, 10]])\n    >>> np.pad(a, 2, pad_with, padder=100)\n    array([[100, 100, 100, 100, 100, 100, 100],\n           [100, 100, 100, 100, 100, 100, 100],\n           [100, 100,   0,   1,   2, 100, 100],\n           [100, 100,   3,   4,   5, 100, 100],\n           [100, 100, 100, 100, 100, 100, 100],\n           [100, 100, 100, 100, 100, 100, 100]])\n    "
    array = np.asarray(array)
    pad_width = np.asarray(pad_width)
    if not pad_width.dtype.kind == 'i':
        raise TypeError('`pad_width` must be of integral type.')
    pad_width = _as_pairs(pad_width, array.ndim, as_index=True)
    if callable(mode):
        function = mode
        (padded, _) = _pad_simple(array, pad_width, fill_value=0)
        for axis in range(padded.ndim):
            view = np.moveaxis(padded, axis, -1)
            inds = ndindex(view.shape[:-1])
            inds = (ind + (Ellipsis,) for ind in inds)
            for ind in inds:
                function(view[ind], pad_width[axis], axis, kwargs)
        return padded
    allowed_kwargs = {'empty': [], 'edge': [], 'wrap': [], 'constant': ['constant_values'], 'linear_ramp': ['end_values'], 'maximum': ['stat_length'], 'mean': ['stat_length'], 'median': ['stat_length'], 'minimum': ['stat_length'], 'reflect': ['reflect_type'], 'symmetric': ['reflect_type']}
    try:
        unsupported_kwargs = set(kwargs) - set(allowed_kwargs[mode])
    except KeyError:
        raise ValueError("mode '{}' is not supported".format(mode)) from None
    if unsupported_kwargs:
        raise ValueError("unsupported keyword arguments for mode '{}': {}".format(mode, unsupported_kwargs))
    stat_functions = {'maximum': np.amax, 'minimum': np.amin, 'mean': np.mean, 'median': np.median}
    (padded, original_area_slice) = _pad_simple(array, pad_width)
    axes = range(padded.ndim)
    if mode == 'constant':
        values = kwargs.get('constant_values', 0)
        values = _as_pairs(values, padded.ndim)
        for (axis, width_pair, value_pair) in zip(axes, pad_width, values):
            roi = _view_roi(padded, original_area_slice, axis)
            _set_pad_area(roi, axis, width_pair, value_pair)
    elif mode == 'empty':
        pass
    elif array.size == 0:
        for (axis, width_pair) in zip(axes, pad_width):
            if array.shape[axis] == 0 and any(width_pair):
                raise ValueError("can't extend empty axis {} using modes other than 'constant' or 'empty'".format(axis))
    elif mode == 'edge':
        for (axis, width_pair) in zip(axes, pad_width):
            roi = _view_roi(padded, original_area_slice, axis)
            edge_pair = _get_edges(roi, axis, width_pair)
            _set_pad_area(roi, axis, width_pair, edge_pair)
    elif mode == 'linear_ramp':
        end_values = kwargs.get('end_values', 0)
        end_values = _as_pairs(end_values, padded.ndim)
        for (axis, width_pair, value_pair) in zip(axes, pad_width, end_values):
            roi = _view_roi(padded, original_area_slice, axis)
            ramp_pair = _get_linear_ramps(roi, axis, width_pair, value_pair)
            _set_pad_area(roi, axis, width_pair, ramp_pair)
    elif mode in stat_functions:
        func = stat_functions[mode]
        length = kwargs.get('stat_length', None)
        length = _as_pairs(length, padded.ndim, as_index=True)
        for (axis, width_pair, length_pair) in zip(axes, pad_width, length):
            roi = _view_roi(padded, original_area_slice, axis)
            stat_pair = _get_stats(roi, axis, width_pair, length_pair, func)
            _set_pad_area(roi, axis, width_pair, stat_pair)
    elif mode in {'reflect', 'symmetric'}:
        method = kwargs.get('reflect_type', 'even')
        include_edge = True if mode == 'symmetric' else False
        for (axis, (left_index, right_index)) in zip(axes, pad_width):
            if array.shape[axis] == 1 and (left_index > 0 or right_index > 0):
                edge_pair = _get_edges(padded, axis, (left_index, right_index))
                _set_pad_area(padded, axis, (left_index, right_index), edge_pair)
                continue
            roi = _view_roi(padded, original_area_slice, axis)
            while left_index > 0 or right_index > 0:
                (left_index, right_index) = _set_reflect_both(roi, axis, (left_index, right_index), method, include_edge)
    elif mode == 'wrap':
        for (axis, (left_index, right_index)) in zip(axes, pad_width):
            roi = _view_roi(padded, original_area_slice, axis)
            original_period = padded.shape[axis] - right_index - left_index
            while left_index > 0 or right_index > 0:
                (left_index, right_index) = _set_wrap_both(roi, axis, (left_index, right_index), original_period)
    return padded