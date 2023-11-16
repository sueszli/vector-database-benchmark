"""
The arraypad module contains a group of functions to pad values onto the edges
of an n-dimensional array.

"""
from __future__ import division, absolute_import, print_function
import numpy as np
from numpy.core.overrides import array_function_dispatch
__all__ = ['pad']

def _arange_ndarray(arr, shape, axis, reverse=False):
    if False:
        print('Hello World!')
    '\n    Create an ndarray of `shape` with increments along specified `axis`\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    shape : tuple of ints\n        Shape of desired array. Should be equivalent to `arr.shape` except\n        `shape[axis]` which may have any positive value.\n    axis : int\n        Axis to increment along.\n    reverse : bool\n        If False, increment in a positive fashion from 1 to `shape[axis]`,\n        inclusive. If True, the bounds are the same but the order reversed.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array sized to pad `arr` along `axis`, with linear range from\n        1 to `shape[axis]` along specified `axis`.\n\n    Notes\n    -----\n    The range is deliberately 1-indexed for this specific use case. Think of\n    this algorithm as broadcasting `np.arange` to a single `axis` of an\n    arbitrarily shaped ndarray.\n\n    '
    initshape = tuple((1 if i != axis else shape[axis] for (i, x) in enumerate(arr.shape)))
    if not reverse:
        padarr = np.arange(1, shape[axis] + 1)
    else:
        padarr = np.arange(shape[axis], 0, -1)
    padarr = padarr.reshape(initshape)
    for (i, dim) in enumerate(shape):
        if padarr.shape[i] != dim:
            padarr = padarr.repeat(dim, axis=i)
    return padarr

def _round_ifneeded(arr, dtype):
    if False:
        i = 10
        return i + 15
    '\n    Rounds arr inplace if destination dtype is integer.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array.\n    dtype : dtype\n        The dtype of the destination array.\n\n    '
    if np.issubdtype(dtype, np.integer):
        arr.round(out=arr)

def _slice_at_axis(shape, sl, axis):
    if False:
        for i in range(10):
            print('nop')
    '\n    Construct a slice tuple the length of shape, with sl at the specified axis\n    '
    slice_tup = (slice(None),)
    return slice_tup * axis + (sl,) + slice_tup * (len(shape) - axis - 1)

def _slice_first(shape, n, axis):
    if False:
        return 10
    ' Construct a slice tuple to take the first n elements along axis '
    return _slice_at_axis(shape, slice(0, n), axis=axis)

def _slice_last(shape, n, axis):
    if False:
        return 10
    ' Construct a slice tuple to take the last n elements along axis '
    dim = shape[axis]
    return _slice_at_axis(shape, slice(dim - n, dim), axis=axis)

def _do_prepend(arr, pad_chunk, axis):
    if False:
        for i in range(10):
            print('nop')
    return np.concatenate((pad_chunk.astype(arr.dtype, copy=False), arr), axis=axis)

def _do_append(arr, pad_chunk, axis):
    if False:
        i = 10
        return i + 15
    return np.concatenate((arr, pad_chunk.astype(arr.dtype, copy=False)), axis=axis)

def _prepend_const(arr, pad_amt, val, axis=-1):
    if False:
        print('Hello World!')
    '\n    Prepend constant `val` along `axis` of `arr`.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to prepend.\n    val : scalar\n        Constant value to use. For best results should be of type `arr.dtype`;\n        if not `arr.dtype` will be cast to `arr.dtype`.\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt` constant `val` prepended along `axis`.\n\n    '
    if pad_amt == 0:
        return arr
    padshape = tuple((x if i != axis else pad_amt for (i, x) in enumerate(arr.shape)))
    return _do_prepend(arr, np.full(padshape, val, dtype=arr.dtype), axis)

def _append_const(arr, pad_amt, val, axis=-1):
    if False:
        while True:
            i = 10
    '\n    Append constant `val` along `axis` of `arr`.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to append.\n    val : scalar\n        Constant value to use. For best results should be of type `arr.dtype`;\n        if not `arr.dtype` will be cast to `arr.dtype`.\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt` constant `val` appended along `axis`.\n\n    '
    if pad_amt == 0:
        return arr
    padshape = tuple((x if i != axis else pad_amt for (i, x) in enumerate(arr.shape)))
    return _do_append(arr, np.full(padshape, val, dtype=arr.dtype), axis)

def _prepend_edge(arr, pad_amt, axis=-1):
    if False:
        while True:
            i = 10
    '\n    Prepend `pad_amt` to `arr` along `axis` by extending edge values.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to prepend.\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, extended by `pad_amt` edge values appended along `axis`.\n\n    '
    if pad_amt == 0:
        return arr
    edge_slice = _slice_first(arr.shape, 1, axis=axis)
    edge_arr = arr[edge_slice]
    return _do_prepend(arr, edge_arr.repeat(pad_amt, axis=axis), axis)

def _append_edge(arr, pad_amt, axis=-1):
    if False:
        print('Hello World!')
    '\n    Append `pad_amt` to `arr` along `axis` by extending edge values.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to append.\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, extended by `pad_amt` edge values prepended along\n        `axis`.\n\n    '
    if pad_amt == 0:
        return arr
    edge_slice = _slice_last(arr.shape, 1, axis=axis)
    edge_arr = arr[edge_slice]
    return _do_append(arr, edge_arr.repeat(pad_amt, axis=axis), axis)

def _prepend_ramp(arr, pad_amt, end, axis=-1):
    if False:
        i = 10
        return i + 15
    '\n    Prepend linear ramp along `axis`.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to prepend.\n    end : scalar\n        Constal value to use. For best results should be of type `arr.dtype`;\n        if not `arr.dtype` will be cast to `arr.dtype`.\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt` values prepended along `axis`. The\n        prepended region ramps linearly from the edge value to `end`.\n\n    '
    if pad_amt == 0:
        return arr
    padshape = tuple((x if i != axis else pad_amt for (i, x) in enumerate(arr.shape)))
    ramp_arr = _arange_ndarray(arr, padshape, axis, reverse=True).astype(np.float64)
    edge_slice = _slice_first(arr.shape, 1, axis=axis)
    edge_pad = arr[edge_slice].repeat(pad_amt, axis)
    slope = (end - edge_pad) / float(pad_amt)
    ramp_arr = ramp_arr * slope
    ramp_arr += edge_pad
    _round_ifneeded(ramp_arr, arr.dtype)
    return _do_prepend(arr, ramp_arr, axis)

def _append_ramp(arr, pad_amt, end, axis=-1):
    if False:
        return 10
    '\n    Append linear ramp along `axis`.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to append.\n    end : scalar\n        Constal value to use. For best results should be of type `arr.dtype`;\n        if not `arr.dtype` will be cast to `arr.dtype`.\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt` values appended along `axis`. The\n        appended region ramps linearly from the edge value to `end`.\n\n    '
    if pad_amt == 0:
        return arr
    padshape = tuple((x if i != axis else pad_amt for (i, x) in enumerate(arr.shape)))
    ramp_arr = _arange_ndarray(arr, padshape, axis, reverse=False).astype(np.float64)
    edge_slice = _slice_last(arr.shape, 1, axis=axis)
    edge_pad = arr[edge_slice].repeat(pad_amt, axis)
    slope = (end - edge_pad) / float(pad_amt)
    ramp_arr = ramp_arr * slope
    ramp_arr += edge_pad
    _round_ifneeded(ramp_arr, arr.dtype)
    return _do_append(arr, ramp_arr, axis)

def _prepend_max(arr, pad_amt, num, axis=-1):
    if False:
        for i in range(10):
            print('nop')
    '\n    Prepend `pad_amt` maximum values along `axis`.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to prepend.\n    num : int\n        Depth into `arr` along `axis` to calculate maximum.\n        Range: [1, `arr.shape[axis]`] or None (entire axis)\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt` values appended along `axis`. The\n        prepended region is the maximum of the first `num` values along\n        `axis`.\n\n    '
    if pad_amt == 0:
        return arr
    if num == 1:
        return _prepend_edge(arr, pad_amt, axis)
    if num is not None:
        if num >= arr.shape[axis]:
            num = None
    max_slice = _slice_first(arr.shape, num, axis=axis)
    max_chunk = arr[max_slice].max(axis=axis, keepdims=True)
    return _do_prepend(arr, max_chunk.repeat(pad_amt, axis=axis), axis)

def _append_max(arr, pad_amt, num, axis=-1):
    if False:
        return 10
    '\n    Pad one `axis` of `arr` with the maximum of the last `num` elements.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to append.\n    num : int\n        Depth into `arr` along `axis` to calculate maximum.\n        Range: [1, `arr.shape[axis]`] or None (entire axis)\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt` values appended along `axis`. The\n        appended region is the maximum of the final `num` values along `axis`.\n\n    '
    if pad_amt == 0:
        return arr
    if num == 1:
        return _append_edge(arr, pad_amt, axis)
    if num is not None:
        if num >= arr.shape[axis]:
            num = None
    if num is not None:
        max_slice = _slice_last(arr.shape, num, axis=axis)
    else:
        max_slice = tuple((slice(None) for x in arr.shape))
    max_chunk = arr[max_slice].max(axis=axis, keepdims=True)
    return _do_append(arr, max_chunk.repeat(pad_amt, axis=axis), axis)

def _prepend_mean(arr, pad_amt, num, axis=-1):
    if False:
        i = 10
        return i + 15
    '\n    Prepend `pad_amt` mean values along `axis`.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to prepend.\n    num : int\n        Depth into `arr` along `axis` to calculate mean.\n        Range: [1, `arr.shape[axis]`] or None (entire axis)\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt` values prepended along `axis`. The\n        prepended region is the mean of the first `num` values along `axis`.\n\n    '
    if pad_amt == 0:
        return arr
    if num == 1:
        return _prepend_edge(arr, pad_amt, axis)
    if num is not None:
        if num >= arr.shape[axis]:
            num = None
    mean_slice = _slice_first(arr.shape, num, axis=axis)
    mean_chunk = arr[mean_slice].mean(axis, keepdims=True)
    _round_ifneeded(mean_chunk, arr.dtype)
    return _do_prepend(arr, mean_chunk.repeat(pad_amt, axis), axis=axis)

def _append_mean(arr, pad_amt, num, axis=-1):
    if False:
        print('Hello World!')
    '\n    Append `pad_amt` mean values along `axis`.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to append.\n    num : int\n        Depth into `arr` along `axis` to calculate mean.\n        Range: [1, `arr.shape[axis]`] or None (entire axis)\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt` values appended along `axis`. The\n        appended region is the maximum of the final `num` values along `axis`.\n\n    '
    if pad_amt == 0:
        return arr
    if num == 1:
        return _append_edge(arr, pad_amt, axis)
    if num is not None:
        if num >= arr.shape[axis]:
            num = None
    if num is not None:
        mean_slice = _slice_last(arr.shape, num, axis=axis)
    else:
        mean_slice = tuple((slice(None) for x in arr.shape))
    mean_chunk = arr[mean_slice].mean(axis=axis, keepdims=True)
    _round_ifneeded(mean_chunk, arr.dtype)
    return _do_append(arr, mean_chunk.repeat(pad_amt, axis), axis=axis)

def _prepend_med(arr, pad_amt, num, axis=-1):
    if False:
        while True:
            i = 10
    '\n    Prepend `pad_amt` median values along `axis`.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to prepend.\n    num : int\n        Depth into `arr` along `axis` to calculate median.\n        Range: [1, `arr.shape[axis]`] or None (entire axis)\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt` values prepended along `axis`. The\n        prepended region is the median of the first `num` values along `axis`.\n\n    '
    if pad_amt == 0:
        return arr
    if num == 1:
        return _prepend_edge(arr, pad_amt, axis)
    if num is not None:
        if num >= arr.shape[axis]:
            num = None
    med_slice = _slice_first(arr.shape, num, axis=axis)
    med_chunk = np.median(arr[med_slice], axis=axis, keepdims=True)
    _round_ifneeded(med_chunk, arr.dtype)
    return _do_prepend(arr, med_chunk.repeat(pad_amt, axis), axis=axis)

def _append_med(arr, pad_amt, num, axis=-1):
    if False:
        return 10
    '\n    Append `pad_amt` median values along `axis`.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to append.\n    num : int\n        Depth into `arr` along `axis` to calculate median.\n        Range: [1, `arr.shape[axis]`] or None (entire axis)\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt` values appended along `axis`. The\n        appended region is the median of the final `num` values along `axis`.\n\n    '
    if pad_amt == 0:
        return arr
    if num == 1:
        return _append_edge(arr, pad_amt, axis)
    if num is not None:
        if num >= arr.shape[axis]:
            num = None
    if num is not None:
        med_slice = _slice_last(arr.shape, num, axis=axis)
    else:
        med_slice = tuple((slice(None) for x in arr.shape))
    med_chunk = np.median(arr[med_slice], axis=axis, keepdims=True)
    _round_ifneeded(med_chunk, arr.dtype)
    return _do_append(arr, med_chunk.repeat(pad_amt, axis), axis=axis)

def _prepend_min(arr, pad_amt, num, axis=-1):
    if False:
        while True:
            i = 10
    '\n    Prepend `pad_amt` minimum values along `axis`.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to prepend.\n    num : int\n        Depth into `arr` along `axis` to calculate minimum.\n        Range: [1, `arr.shape[axis]`] or None (entire axis)\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt` values prepended along `axis`. The\n        prepended region is the minimum of the first `num` values along\n        `axis`.\n\n    '
    if pad_amt == 0:
        return arr
    if num == 1:
        return _prepend_edge(arr, pad_amt, axis)
    if num is not None:
        if num >= arr.shape[axis]:
            num = None
    min_slice = _slice_first(arr.shape, num, axis=axis)
    min_chunk = arr[min_slice].min(axis=axis, keepdims=True)
    return _do_prepend(arr, min_chunk.repeat(pad_amt, axis), axis=axis)

def _append_min(arr, pad_amt, num, axis=-1):
    if False:
        while True:
            i = 10
    '\n    Append `pad_amt` median values along `axis`.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to append.\n    num : int\n        Depth into `arr` along `axis` to calculate minimum.\n        Range: [1, `arr.shape[axis]`] or None (entire axis)\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt` values appended along `axis`. The\n        appended region is the minimum of the final `num` values along `axis`.\n\n    '
    if pad_amt == 0:
        return arr
    if num == 1:
        return _append_edge(arr, pad_amt, axis)
    if num is not None:
        if num >= arr.shape[axis]:
            num = None
    if num is not None:
        min_slice = _slice_last(arr.shape, num, axis=axis)
    else:
        min_slice = tuple((slice(None) for x in arr.shape))
    min_chunk = arr[min_slice].min(axis=axis, keepdims=True)
    return _do_append(arr, min_chunk.repeat(pad_amt, axis), axis=axis)

def _pad_ref(arr, pad_amt, method, axis=-1):
    if False:
        while True:
            i = 10
    "\n    Pad `axis` of `arr` by reflection.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : tuple of ints, length 2\n        Padding to (prepend, append) along `axis`.\n    method : str\n        Controls method of reflection; options are 'even' or 'odd'.\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt[0]` values prepended and `pad_amt[1]`\n        values appended along `axis`. Both regions are padded with reflected\n        values from the original array.\n\n    Notes\n    -----\n    This algorithm does not pad with repetition, i.e. the edges are not\n    repeated in the reflection. For that behavior, use `mode='symmetric'`.\n\n    The modes 'reflect', 'symmetric', and 'wrap' must be padded with a\n    single function, lest the indexing tricks in non-integer multiples of the\n    original shape would violate repetition in the final iteration.\n\n    "
    if pad_amt[0] == 0 and pad_amt[1] == 0:
        return arr
    ref_slice = _slice_at_axis(arr.shape, slice(pad_amt[0], 0, -1), axis=axis)
    ref_chunk1 = arr[ref_slice]
    if 'odd' in method and pad_amt[0] > 0:
        edge_slice1 = _slice_first(arr.shape, 1, axis=axis)
        edge_chunk = arr[edge_slice1]
        ref_chunk1 = 2 * edge_chunk - ref_chunk1
        del edge_chunk
    start = arr.shape[axis] - pad_amt[1] - 1
    end = arr.shape[axis] - 1
    ref_slice = _slice_at_axis(arr.shape, slice(start, end), axis=axis)
    rev_idx = _slice_at_axis(arr.shape, slice(None, None, -1), axis=axis)
    ref_chunk2 = arr[ref_slice][rev_idx]
    if 'odd' in method:
        edge_slice2 = _slice_last(arr.shape, 1, axis=axis)
        edge_chunk = arr[edge_slice2]
        ref_chunk2 = 2 * edge_chunk - ref_chunk2
        del edge_chunk
    return np.concatenate((ref_chunk1, arr, ref_chunk2), axis=axis)

def _pad_sym(arr, pad_amt, method, axis=-1):
    if False:
        while True:
            i = 10
    "\n    Pad `axis` of `arr` by symmetry.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : tuple of ints, length 2\n        Padding to (prepend, append) along `axis`.\n    method : str\n        Controls method of symmetry; options are 'even' or 'odd'.\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt[0]` values prepended and `pad_amt[1]`\n        values appended along `axis`. Both regions are padded with symmetric\n        values from the original array.\n\n    Notes\n    -----\n    This algorithm DOES pad with repetition, i.e. the edges are repeated.\n    For padding without repeated edges, use `mode='reflect'`.\n\n    The modes 'reflect', 'symmetric', and 'wrap' must be padded with a\n    single function, lest the indexing tricks in non-integer multiples of the\n    original shape would violate repetition in the final iteration.\n\n    "
    if pad_amt[0] == 0 and pad_amt[1] == 0:
        return arr
    sym_slice = _slice_first(arr.shape, pad_amt[0], axis=axis)
    rev_idx = _slice_at_axis(arr.shape, slice(None, None, -1), axis=axis)
    sym_chunk1 = arr[sym_slice][rev_idx]
    if 'odd' in method and pad_amt[0] > 0:
        edge_slice1 = _slice_first(arr.shape, 1, axis=axis)
        edge_chunk = arr[edge_slice1]
        sym_chunk1 = 2 * edge_chunk - sym_chunk1
        del edge_chunk
    sym_slice = _slice_last(arr.shape, pad_amt[1], axis=axis)
    sym_chunk2 = arr[sym_slice][rev_idx]
    if 'odd' in method:
        edge_slice2 = _slice_last(arr.shape, 1, axis=axis)
        edge_chunk = arr[edge_slice2]
        sym_chunk2 = 2 * edge_chunk - sym_chunk2
        del edge_chunk
    return np.concatenate((sym_chunk1, arr, sym_chunk2), axis=axis)

def _pad_wrap(arr, pad_amt, axis=-1):
    if False:
        i = 10
        return i + 15
    "\n    Pad `axis` of `arr` via wrapping.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : tuple of ints, length 2\n        Padding to (prepend, append) along `axis`.\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt[0]` values prepended and `pad_amt[1]`\n        values appended along `axis`. Both regions are padded wrapped values\n        from the opposite end of `axis`.\n\n    Notes\n    -----\n    This method of padding is also known as 'tile' or 'tiling'.\n\n    The modes 'reflect', 'symmetric', and 'wrap' must be padded with a\n    single function, lest the indexing tricks in non-integer multiples of the\n    original shape would violate repetition in the final iteration.\n\n    "
    if pad_amt[0] == 0 and pad_amt[1] == 0:
        return arr
    wrap_slice = _slice_last(arr.shape, pad_amt[0], axis=axis)
    wrap_chunk1 = arr[wrap_slice]
    wrap_slice = _slice_first(arr.shape, pad_amt[1], axis=axis)
    wrap_chunk2 = arr[wrap_slice]
    return np.concatenate((wrap_chunk1, arr, wrap_chunk2), axis=axis)

def _as_pairs(x, ndim, as_index=False):
    if False:
        i = 10
        return i + 15
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

def _pad_dispatcher(array, pad_width, mode, **kwargs):
    if False:
        return 10
    return (array,)

@array_function_dispatch(_pad_dispatcher, module='numpy')
def pad(array, pad_width, mode, **kwargs):
    if False:
        return 10
    "\n    Pads an array.\n\n    Parameters\n    ----------\n    array : array_like of rank N\n        Input array\n    pad_width : {sequence, array_like, int}\n        Number of values padded to the edges of each axis.\n        ((before_1, after_1), ... (before_N, after_N)) unique pad widths\n        for each axis.\n        ((before, after),) yields same before and after pad for each axis.\n        (pad,) or int is a shortcut for before = after = pad width for all\n        axes.\n    mode : str or function\n        One of the following string values or a user supplied function.\n\n        'constant'\n            Pads with a constant value.\n        'edge'\n            Pads with the edge values of array.\n        'linear_ramp'\n            Pads with the linear ramp between end_value and the\n            array edge value.\n        'maximum'\n            Pads with the maximum value of all or part of the\n            vector along each axis.\n        'mean'\n            Pads with the mean value of all or part of the\n            vector along each axis.\n        'median'\n            Pads with the median value of all or part of the\n            vector along each axis.\n        'minimum'\n            Pads with the minimum value of all or part of the\n            vector along each axis.\n        'reflect'\n            Pads with the reflection of the vector mirrored on\n            the first and last values of the vector along each\n            axis.\n        'symmetric'\n            Pads with the reflection of the vector mirrored\n            along the edge of the array.\n        'wrap'\n            Pads with the wrap of the vector along the axis.\n            The first values are used to pad the end and the\n            end values are used to pad the beginning.\n        <function>\n            Padding function, see Notes.\n    stat_length : sequence or int, optional\n        Used in 'maximum', 'mean', 'median', and 'minimum'.  Number of\n        values at edge of each axis used to calculate the statistic value.\n\n        ((before_1, after_1), ... (before_N, after_N)) unique statistic\n        lengths for each axis.\n\n        ((before, after),) yields same before and after statistic lengths\n        for each axis.\n\n        (stat_length,) or int is a shortcut for before = after = statistic\n        length for all axes.\n\n        Default is ``None``, to use the entire axis.\n    constant_values : sequence or int, optional\n        Used in 'constant'.  The values to set the padded values for each\n        axis.\n\n        ((before_1, after_1), ... (before_N, after_N)) unique pad constants\n        for each axis.\n\n        ((before, after),) yields same before and after constants for each\n        axis.\n\n        (constant,) or int is a shortcut for before = after = constant for\n        all axes.\n\n        Default is 0.\n    end_values : sequence or int, optional\n        Used in 'linear_ramp'.  The values used for the ending value of the\n        linear_ramp and that will form the edge of the padded array.\n\n        ((before_1, after_1), ... (before_N, after_N)) unique end values\n        for each axis.\n\n        ((before, after),) yields same before and after end values for each\n        axis.\n\n        (constant,) or int is a shortcut for before = after = end value for\n        all axes.\n\n        Default is 0.\n    reflect_type : {'even', 'odd'}, optional\n        Used in 'reflect', and 'symmetric'.  The 'even' style is the\n        default with an unaltered reflection around the edge value.  For\n        the 'odd' style, the extended part of the array is created by\n        subtracting the reflected values from two times the edge value.\n\n    Returns\n    -------\n    pad : ndarray\n        Padded array of rank equal to `array` with shape increased\n        according to `pad_width`.\n\n    Notes\n    -----\n    .. versionadded:: 1.7.0\n\n    For an array with rank greater than 1, some of the padding of later\n    axes is calculated from padding of previous axes.  This is easiest to\n    think about with a rank 2 array where the corners of the padded array\n    are calculated by using padded values from the first axis.\n\n    The padding function, if used, should return a rank 1 array equal in\n    length to the vector argument with padded values replaced. It has the\n    following signature::\n\n        padding_func(vector, iaxis_pad_width, iaxis, kwargs)\n\n    where\n\n        vector : ndarray\n            A rank 1 array already padded with zeros.  Padded values are\n            vector[:pad_tuple[0]] and vector[-pad_tuple[1]:].\n        iaxis_pad_width : tuple\n            A 2-tuple of ints, iaxis_pad_width[0] represents the number of\n            values padded at the beginning of vector where\n            iaxis_pad_width[1] represents the number of values padded at\n            the end of vector.\n        iaxis : int\n            The axis currently being calculated.\n        kwargs : dict\n            Any keyword arguments the function requires.\n\n    Examples\n    --------\n    >>> a = [1, 2, 3, 4, 5]\n    >>> np.pad(a, (2,3), 'constant', constant_values=(4, 6))\n    array([4, 4, 1, 2, 3, 4, 5, 6, 6, 6])\n\n    >>> np.pad(a, (2, 3), 'edge')\n    array([1, 1, 1, 2, 3, 4, 5, 5, 5, 5])\n\n    >>> np.pad(a, (2, 3), 'linear_ramp', end_values=(5, -4))\n    array([ 5,  3,  1,  2,  3,  4,  5,  2, -1, -4])\n\n    >>> np.pad(a, (2,), 'maximum')\n    array([5, 5, 1, 2, 3, 4, 5, 5, 5])\n\n    >>> np.pad(a, (2,), 'mean')\n    array([3, 3, 1, 2, 3, 4, 5, 3, 3])\n\n    >>> np.pad(a, (2,), 'median')\n    array([3, 3, 1, 2, 3, 4, 5, 3, 3])\n\n    >>> a = [[1, 2], [3, 4]]\n    >>> np.pad(a, ((3, 2), (2, 3)), 'minimum')\n    array([[1, 1, 1, 2, 1, 1, 1],\n           [1, 1, 1, 2, 1, 1, 1],\n           [1, 1, 1, 2, 1, 1, 1],\n           [1, 1, 1, 2, 1, 1, 1],\n           [3, 3, 3, 4, 3, 3, 3],\n           [1, 1, 1, 2, 1, 1, 1],\n           [1, 1, 1, 2, 1, 1, 1]])\n\n    >>> a = [1, 2, 3, 4, 5]\n    >>> np.pad(a, (2, 3), 'reflect')\n    array([3, 2, 1, 2, 3, 4, 5, 4, 3, 2])\n\n    >>> np.pad(a, (2, 3), 'reflect', reflect_type='odd')\n    array([-1,  0,  1,  2,  3,  4,  5,  6,  7,  8])\n\n    >>> np.pad(a, (2, 3), 'symmetric')\n    array([2, 1, 1, 2, 3, 4, 5, 5, 4, 3])\n\n    >>> np.pad(a, (2, 3), 'symmetric', reflect_type='odd')\n    array([0, 1, 1, 2, 3, 4, 5, 5, 6, 7])\n\n    >>> np.pad(a, (2, 3), 'wrap')\n    array([4, 5, 1, 2, 3, 4, 5, 1, 2, 3])\n\n    >>> def pad_with(vector, pad_width, iaxis, kwargs):\n    ...     pad_value = kwargs.get('padder', 10)\n    ...     vector[:pad_width[0]] = pad_value\n    ...     vector[-pad_width[1]:] = pad_value\n    ...     return vector\n    >>> a = np.arange(6)\n    >>> a = a.reshape((2, 3))\n    >>> np.pad(a, 2, pad_with)\n    array([[10, 10, 10, 10, 10, 10, 10],\n           [10, 10, 10, 10, 10, 10, 10],\n           [10, 10,  0,  1,  2, 10, 10],\n           [10, 10,  3,  4,  5, 10, 10],\n           [10, 10, 10, 10, 10, 10, 10],\n           [10, 10, 10, 10, 10, 10, 10]])\n    >>> np.pad(a, 2, pad_with, padder=100)\n    array([[100, 100, 100, 100, 100, 100, 100],\n           [100, 100, 100, 100, 100, 100, 100],\n           [100, 100,   0,   1,   2, 100, 100],\n           [100, 100,   3,   4,   5, 100, 100],\n           [100, 100, 100, 100, 100, 100, 100],\n           [100, 100, 100, 100, 100, 100, 100]])\n    "
    if not np.asarray(pad_width).dtype.kind == 'i':
        raise TypeError('`pad_width` must be of integral type.')
    narray = np.array(array)
    pad_width = _as_pairs(pad_width, narray.ndim, as_index=True)
    allowedkwargs = {'constant': ['constant_values'], 'edge': [], 'linear_ramp': ['end_values'], 'maximum': ['stat_length'], 'mean': ['stat_length'], 'median': ['stat_length'], 'minimum': ['stat_length'], 'reflect': ['reflect_type'], 'symmetric': ['reflect_type'], 'wrap': []}
    kwdefaults = {'stat_length': None, 'constant_values': 0, 'end_values': 0, 'reflect_type': 'even'}
    if isinstance(mode, np.compat.basestring):
        for key in kwargs:
            if key not in allowedkwargs[mode]:
                raise ValueError('%s keyword not in allowed keywords %s' % (key, allowedkwargs[mode]))
        for kw in allowedkwargs[mode]:
            kwargs.setdefault(kw, kwdefaults[kw])
        for i in kwargs:
            if i == 'stat_length':
                kwargs[i] = _as_pairs(kwargs[i], narray.ndim, as_index=True)
            if i in ['end_values', 'constant_values']:
                kwargs[i] = _as_pairs(kwargs[i], narray.ndim)
    else:
        function = mode
        rank = list(range(narray.ndim))
        total_dim_increase = [np.sum(pad_width[i]) for i in rank]
        offset_slices = tuple((slice(pad_width[i][0], pad_width[i][0] + narray.shape[i]) for i in rank))
        new_shape = np.array(narray.shape) + total_dim_increase
        newmat = np.zeros(new_shape, narray.dtype)
        newmat[offset_slices] = narray
        for iaxis in rank:
            np.apply_along_axis(function, iaxis, newmat, pad_width[iaxis], iaxis, kwargs)
        return newmat
    newmat = narray.copy()
    if mode == 'constant':
        for (axis, ((pad_before, pad_after), (before_val, after_val))) in enumerate(zip(pad_width, kwargs['constant_values'])):
            newmat = _prepend_const(newmat, pad_before, before_val, axis)
            newmat = _append_const(newmat, pad_after, after_val, axis)
    elif mode == 'edge':
        for (axis, (pad_before, pad_after)) in enumerate(pad_width):
            newmat = _prepend_edge(newmat, pad_before, axis)
            newmat = _append_edge(newmat, pad_after, axis)
    elif mode == 'linear_ramp':
        for (axis, ((pad_before, pad_after), (before_val, after_val))) in enumerate(zip(pad_width, kwargs['end_values'])):
            newmat = _prepend_ramp(newmat, pad_before, before_val, axis)
            newmat = _append_ramp(newmat, pad_after, after_val, axis)
    elif mode == 'maximum':
        for (axis, ((pad_before, pad_after), (chunk_before, chunk_after))) in enumerate(zip(pad_width, kwargs['stat_length'])):
            newmat = _prepend_max(newmat, pad_before, chunk_before, axis)
            newmat = _append_max(newmat, pad_after, chunk_after, axis)
    elif mode == 'mean':
        for (axis, ((pad_before, pad_after), (chunk_before, chunk_after))) in enumerate(zip(pad_width, kwargs['stat_length'])):
            newmat = _prepend_mean(newmat, pad_before, chunk_before, axis)
            newmat = _append_mean(newmat, pad_after, chunk_after, axis)
    elif mode == 'median':
        for (axis, ((pad_before, pad_after), (chunk_before, chunk_after))) in enumerate(zip(pad_width, kwargs['stat_length'])):
            newmat = _prepend_med(newmat, pad_before, chunk_before, axis)
            newmat = _append_med(newmat, pad_after, chunk_after, axis)
    elif mode == 'minimum':
        for (axis, ((pad_before, pad_after), (chunk_before, chunk_after))) in enumerate(zip(pad_width, kwargs['stat_length'])):
            newmat = _prepend_min(newmat, pad_before, chunk_before, axis)
            newmat = _append_min(newmat, pad_after, chunk_after, axis)
    elif mode == 'reflect':
        for (axis, (pad_before, pad_after)) in enumerate(pad_width):
            if narray.shape[axis] == 0:
                if pad_before > 0 or pad_after > 0:
                    raise ValueError("There aren't any elements to reflect in axis {} of `array`".format(axis))
                continue
            if (pad_before > 0 or pad_after > 0) and newmat.shape[axis] == 1:
                newmat = _prepend_edge(newmat, pad_before, axis)
                newmat = _append_edge(newmat, pad_after, axis)
                continue
            method = kwargs['reflect_type']
            safe_pad = newmat.shape[axis] - 1
            while pad_before > safe_pad or pad_after > safe_pad:
                pad_iter_b = min(safe_pad, safe_pad * (pad_before // safe_pad))
                pad_iter_a = min(safe_pad, safe_pad * (pad_after // safe_pad))
                newmat = _pad_ref(newmat, (pad_iter_b, pad_iter_a), method, axis)
                pad_before -= pad_iter_b
                pad_after -= pad_iter_a
                safe_pad += pad_iter_b + pad_iter_a
            newmat = _pad_ref(newmat, (pad_before, pad_after), method, axis)
    elif mode == 'symmetric':
        for (axis, (pad_before, pad_after)) in enumerate(pad_width):
            method = kwargs['reflect_type']
            safe_pad = newmat.shape[axis]
            while pad_before > safe_pad or pad_after > safe_pad:
                pad_iter_b = min(safe_pad, safe_pad * (pad_before // safe_pad))
                pad_iter_a = min(safe_pad, safe_pad * (pad_after // safe_pad))
                newmat = _pad_sym(newmat, (pad_iter_b, pad_iter_a), method, axis)
                pad_before -= pad_iter_b
                pad_after -= pad_iter_a
                safe_pad += pad_iter_b + pad_iter_a
            newmat = _pad_sym(newmat, (pad_before, pad_after), method, axis)
    elif mode == 'wrap':
        for (axis, (pad_before, pad_after)) in enumerate(pad_width):
            safe_pad = newmat.shape[axis]
            while pad_before > safe_pad or pad_after > safe_pad:
                pad_iter_b = min(safe_pad, safe_pad * (pad_before // safe_pad))
                pad_iter_a = min(safe_pad, safe_pad * (pad_after // safe_pad))
                newmat = _pad_wrap(newmat, (pad_iter_b, pad_iter_a), axis)
                pad_before -= pad_iter_b
                pad_after -= pad_iter_a
                safe_pad += pad_iter_b + pad_iter_a
            newmat = _pad_wrap(newmat, (pad_before, pad_after), axis)
    return newmat