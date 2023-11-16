""" A set of NumPy functions to apply per chunk """
from __future__ import annotations
import contextlib
from collections.abc import Container, Iterable, Sequence
from functools import wraps
from numbers import Integral
import numpy as np
from tlz import concat
from dask.core import flatten

def keepdims_wrapper(a_callable):
    if False:
        i = 10
        return i + 15
    "\n    A wrapper for functions that don't provide keepdims to ensure that they do.\n    "

    @wraps(a_callable)
    def keepdims_wrapped_callable(x, axis=None, keepdims=None, *args, **kwargs):
        if False:
            return 10
        r = a_callable(x, *args, axis=axis, **kwargs)
        if not keepdims:
            return r
        axes = axis
        if axes is None:
            axes = range(x.ndim)
        if not isinstance(axes, (Container, Iterable, Sequence)):
            axes = [axes]
        r_slice = tuple()
        for each_axis in range(x.ndim):
            if each_axis in axes:
                r_slice += (None,)
            else:
                r_slice += (slice(None),)
        r = r[r_slice]
        return r
    return keepdims_wrapped_callable
sum = np.sum
prod = np.prod
min = np.min
max = np.max
argmin = keepdims_wrapper(np.argmin)
nanargmin = keepdims_wrapper(np.nanargmin)
argmax = keepdims_wrapper(np.argmax)
nanargmax = keepdims_wrapper(np.nanargmax)
any = np.any
all = np.all
nansum = np.nansum
nanprod = np.nanprod
nancumprod = np.nancumprod
nancumsum = np.nancumsum
nanmin = np.nanmin
nanmax = np.nanmax
mean = np.mean
with contextlib.suppress(AttributeError):
    nanmean = np.nanmean
var = np.var
with contextlib.suppress(AttributeError):
    nanvar = np.nanvar
std = np.std
with contextlib.suppress(AttributeError):
    nanstd = np.nanstd

def coarsen(reduction, x, axes, trim_excess=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Coarsen array by applying reduction to fixed size neighborhoods\n\n    Parameters\n    ----------\n    reduction: function\n        Function like np.sum, np.mean, etc...\n    x: np.ndarray\n        Array to be coarsened\n    axes: dict\n        Mapping of axis to coarsening factor\n\n    Examples\n    --------\n    >>> x = np.array([1, 2, 3, 4, 5, 6])\n    >>> coarsen(np.sum, x, {0: 2})\n    array([ 3,  7, 11])\n    >>> coarsen(np.max, x, {0: 3})\n    array([3, 6])\n\n    Provide dictionary of scale per dimension\n\n    >>> x = np.arange(24).reshape((4, 6))\n    >>> x\n    array([[ 0,  1,  2,  3,  4,  5],\n           [ 6,  7,  8,  9, 10, 11],\n           [12, 13, 14, 15, 16, 17],\n           [18, 19, 20, 21, 22, 23]])\n\n    >>> coarsen(np.min, x, {0: 2, 1: 3})\n    array([[ 0,  3],\n           [12, 15]])\n\n    You must avoid excess elements explicitly\n\n    >>> x = np.array([1, 2, 3, 4, 5, 6, 7, 8])\n    >>> coarsen(np.min, x, {0: 3}, trim_excess=True)\n    array([1, 4])\n    '
    for i in range(x.ndim):
        if i not in axes:
            axes[i] = 1
    if trim_excess:
        ind = tuple((slice(0, -(d % axes[i])) if d % axes[i] else slice(None, None) for (i, d) in enumerate(x.shape)))
        x = x[ind]
    newshape = tuple(concat([(x.shape[i] // axes[i], axes[i]) for i in range(x.ndim)]))
    return reduction(x.reshape(newshape), axis=tuple(range(1, x.ndim * 2, 2)), **kwargs)

def trim(x, axes=None):
    if False:
        while True:
            i = 10
    'Trim boundaries off of array\n\n    >>> x = np.arange(24).reshape((4, 6))\n    >>> trim(x, axes={0: 0, 1: 1})\n    array([[ 1,  2,  3,  4],\n           [ 7,  8,  9, 10],\n           [13, 14, 15, 16],\n           [19, 20, 21, 22]])\n\n    >>> trim(x, axes={0: 1, 1: 1})\n    array([[ 7,  8,  9, 10],\n           [13, 14, 15, 16]])\n    '
    if isinstance(axes, Integral):
        axes = [axes] * x.ndim
    if isinstance(axes, dict):
        axes = [axes.get(i, 0) for i in range(x.ndim)]
    return x[tuple((slice(ax, -ax if ax else None) for ax in axes))]

def topk(a, k, axis, keepdims):
    if False:
        i = 10
        return i + 15
    'Chunk and combine function of topk\n\n    Extract the k largest elements from a on the given axis.\n    If k is negative, extract the -k smallest elements instead.\n    Note that, unlike in the parent function, the returned elements\n    are not sorted internally.\n    '
    assert keepdims is True
    axis = axis[0]
    if abs(k) >= a.shape[axis]:
        return a
    a = np.partition(a, -k, axis=axis)
    k_slice = slice(-k, None) if k > 0 else slice(-k)
    return a[tuple((k_slice if i == axis else slice(None) for i in range(a.ndim)))]

def topk_aggregate(a, k, axis, keepdims):
    if False:
        while True:
            i = 10
    'Final aggregation function of topk\n\n    Invoke topk one final time and then sort the results internally.\n    '
    assert keepdims is True
    a = topk(a, k, axis, keepdims)
    axis = axis[0]
    a = np.sort(a, axis=axis)
    if k < 0:
        return a
    return a[tuple((slice(None, None, -1) if i == axis else slice(None) for i in range(a.ndim)))]

def argtopk_preprocess(a, idx):
    if False:
        i = 10
        return i + 15
    'Preparatory step for argtopk\n\n    Put data together with its original indices in a tuple.\n    '
    return (a, idx)

def argtopk(a_plus_idx, k, axis, keepdims):
    if False:
        i = 10
        return i + 15
    'Chunk and combine function of argtopk\n\n    Extract the indices of the k largest elements from a on the given axis.\n    If k is negative, extract the indices of the -k smallest elements instead.\n    Note that, unlike in the parent function, the returned elements\n    are not sorted internally.\n    '
    assert keepdims is True
    axis = axis[0]
    if isinstance(a_plus_idx, list):
        a_plus_idx = list(flatten(a_plus_idx))
        a = np.concatenate([ai for (ai, _) in a_plus_idx], axis)
        idx = np.concatenate([np.broadcast_to(idxi, ai.shape) for (ai, idxi) in a_plus_idx], axis)
    else:
        (a, idx) = a_plus_idx
    if abs(k) >= a.shape[axis]:
        return a_plus_idx
    idx2 = np.argpartition(a, -k, axis=axis)
    k_slice = slice(-k, None) if k > 0 else slice(-k)
    idx2 = idx2[tuple((k_slice if i == axis else slice(None) for i in range(a.ndim)))]
    return (np.take_along_axis(a, idx2, axis), np.take_along_axis(idx, idx2, axis))

def argtopk_aggregate(a_plus_idx, k, axis, keepdims):
    if False:
        for i in range(10):
            print('nop')
    'Final aggregation function of argtopk\n\n    Invoke argtopk one final time, sort the results internally, drop the data\n    and return the index only.\n    '
    assert keepdims is True
    a_plus_idx = a_plus_idx if len(a_plus_idx) > 1 else a_plus_idx[0]
    (a, idx) = argtopk(a_plus_idx, k, axis, keepdims)
    axis = axis[0]
    idx2 = np.argsort(a, axis=axis)
    idx = np.take_along_axis(idx, idx2, axis)
    if k < 0:
        return idx
    return idx[tuple((slice(None, None, -1) if i == axis else slice(None) for i in range(idx.ndim)))]

def arange(start, stop, step, length, dtype, like=None):
    if False:
        while True:
            i = 10
    from dask.array.utils import arange_safe
    res = arange_safe(start, stop, step, dtype, like=like)
    return res[:-1] if len(res) > length else res

def linspace(start, stop, num, endpoint=True, dtype=None):
    if False:
        print('Hello World!')
    from dask.array.core import Array
    if isinstance(start, Array):
        start = start.compute()
    if isinstance(stop, Array):
        stop = stop.compute()
    return np.linspace(start, stop, num, endpoint=endpoint, dtype=dtype)

def astype(x, astype_dtype=None, **kwargs):
    if False:
        return 10
    return x.astype(astype_dtype, **kwargs)

def view(x, dtype, order='C'):
    if False:
        for i in range(10):
            print('nop')
    if order == 'C':
        try:
            x = np.ascontiguousarray(x, like=x)
        except TypeError:
            x = np.ascontiguousarray(x)
        return x.view(dtype)
    else:
        try:
            x = np.asfortranarray(x, like=x)
        except TypeError:
            x = np.asfortranarray(x)
        return x.T.view(dtype).T

def slice_with_int_dask_array(x, idx, offset, x_size, axis):
    if False:
        i = 10
        return i + 15
    'Chunk function of `slice_with_int_dask_array_on_axis`.\n    Slice one chunk of x by one chunk of idx.\n\n    Parameters\n    ----------\n    x: ndarray, any dtype, any shape\n        i-th chunk of x\n    idx: ndarray, ndim=1, dtype=any integer\n        j-th chunk of idx (cartesian product with the chunks of x)\n    offset: ndarray, shape=(1, ), dtype=int64\n        Index of the first element along axis of the current chunk of x\n    x_size: int\n        Total size of the x da.Array along axis\n    axis: int\n        normalized axis to take elements from (0 <= axis < x.ndim)\n\n    Returns\n    -------\n    x sliced along axis, using only the elements of idx that fall inside the\n    current chunk.\n    '
    from dask.array.utils import asarray_safe, meta_from_array
    idx = asarray_safe(idx, like=meta_from_array(x))
    idx = idx.astype(np.int64)
    idx = np.where(idx < 0, idx + x_size, idx)
    idx = idx - offset
    idx_filter = (idx >= 0) & (idx < x.shape[axis])
    idx = idx[idx_filter]
    return x[tuple((idx if i == axis else slice(None) for i in range(x.ndim)))]

def slice_with_int_dask_array_aggregate(idx, chunk_outputs, x_chunks, axis):
    if False:
        while True:
            i = 10
    'Final aggregation function of `slice_with_int_dask_array_on_axis`.\n    Aggregate all chunks of x by one chunk of idx, reordering the output of\n    `slice_with_int_dask_array`.\n\n    Note that there is no combine function, as a recursive aggregation (e.g.\n    with split_every) would not give any benefit.\n\n    Parameters\n    ----------\n    idx: ndarray, ndim=1, dtype=any integer\n        j-th chunk of idx\n    chunk_outputs: ndarray\n        concatenation along axis of the outputs of `slice_with_int_dask_array`\n        for all chunks of x and the j-th chunk of idx\n    x_chunks: tuple\n        dask chunks of the x da.Array along axis, e.g. ``(3, 3, 2)``\n    axis: int\n        normalized axis to take elements from (0 <= axis < x.ndim)\n\n    Returns\n    -------\n    Selection from all chunks of x for the j-th chunk of idx, in the correct\n    order\n    '
    idx = idx.astype(np.int64)
    idx = np.where(idx < 0, idx + sum(x_chunks), idx)
    x_chunk_offset = 0
    chunk_output_offset = 0
    idx_final = np.zeros_like(idx)
    for x_chunk in x_chunks:
        idx_filter = (idx >= x_chunk_offset) & (idx < x_chunk_offset + x_chunk)
        idx_cum = np.cumsum(idx_filter)
        idx_final += np.where(idx_filter, idx_cum - 1 + chunk_output_offset, 0)
        x_chunk_offset += x_chunk
        if idx_cum.size > 0:
            chunk_output_offset += idx_cum[-1]
    return chunk_outputs[tuple((idx_final if i == axis else slice(None) for i in range(chunk_outputs.ndim)))]

def getitem(obj, index):
    if False:
        i = 10
        return i + 15
    'Getitem function\n\n    This function creates a copy of the desired selection for array-like\n    inputs when the selection is smaller than half of the original array. This\n    avoids excess memory usage when extracting a small portion from a large array.\n    For more information, see\n    https://numpy.org/doc/stable/reference/arrays.indexing.html#basic-slicing-and-indexing.\n\n    Parameters\n    ----------\n    obj: ndarray, string, tuple, list\n        Object to get item from.\n    index: int, list[int], slice()\n        Desired selection to extract from obj.\n\n    Returns\n    -------\n    Selection obj[index]\n\n    '
    try:
        result = obj[index]
    except IndexError as e:
        raise ValueError('Array chunk size or shape is unknown. Possible solution with x.compute_chunk_sizes()') from e
    try:
        if not result.flags.owndata and obj.size >= 2 * result.size:
            result = result.copy()
    except AttributeError:
        pass
    return result