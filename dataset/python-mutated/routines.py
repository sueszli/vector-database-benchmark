from __future__ import annotations
import math
import warnings
from collections.abc import Iterable
from functools import partial, reduce, wraps
from numbers import Integral, Real
import numpy as np
from tlz import concat, interleave, sliding_window
from dask.array import chunk
from dask.array.core import Array, asanyarray, asarray, blockwise, broadcast_arrays, broadcast_shapes, broadcast_to, concatenate, elemwise, from_array, implements, is_scalar_for_elemwise, map_blocks, stack, tensordot_lookup
from dask.array.creation import arange, diag, empty, indices, tri
from dask.array.einsumfuncs import einsum
from dask.array.reductions import reduction
from dask.array.ufunc import multiply, sqrt
from dask.array.utils import array_safe, asarray_safe, meta_from_array, safe_wraps, validate_axis
from dask.array.wrap import ones
from dask.base import is_dask_collection, tokenize
from dask.core import flatten
from dask.delayed import Delayed, unpack_collections
from dask.highlevelgraph import HighLevelGraph
from dask.utils import apply, derived_from, funcname, is_arraylike, is_cupy_type
_range = range

@derived_from(np)
def array(x, dtype=None, ndmin=None, *, like=None):
    if False:
        print('Hello World!')
    x = asarray(x, like=like)
    while ndmin is not None and x.ndim < ndmin:
        x = x[None, :]
    if dtype is not None and x.dtype != dtype:
        x = x.astype(dtype)
    return x

@derived_from(np)
def result_type(*args):
    if False:
        for i in range(10):
            print('nop')
    args = [a if is_scalar_for_elemwise(a) else a.dtype for a in args]
    return np.result_type(*args)

@derived_from(np)
def atleast_3d(*arys):
    if False:
        for i in range(10):
            print('nop')
    new_arys = []
    for x in arys:
        x = asanyarray(x)
        if x.ndim == 0:
            x = x[None, None, None]
        elif x.ndim == 1:
            x = x[None, :, None]
        elif x.ndim == 2:
            x = x[:, :, None]
        new_arys.append(x)
    if len(new_arys) == 1:
        return new_arys[0]
    else:
        return new_arys

@derived_from(np)
def atleast_2d(*arys):
    if False:
        while True:
            i = 10
    new_arys = []
    for x in arys:
        x = asanyarray(x)
        if x.ndim == 0:
            x = x[None, None]
        elif x.ndim == 1:
            x = x[None, :]
        new_arys.append(x)
    if len(new_arys) == 1:
        return new_arys[0]
    else:
        return new_arys

@derived_from(np)
def atleast_1d(*arys):
    if False:
        i = 10
        return i + 15
    new_arys = []
    for x in arys:
        x = asanyarray(x)
        if x.ndim == 0:
            x = x[None]
        new_arys.append(x)
    if len(new_arys) == 1:
        return new_arys[0]
    else:
        return new_arys

@derived_from(np)
def vstack(tup, allow_unknown_chunksizes=False):
    if False:
        i = 10
        return i + 15
    if isinstance(tup, Array):
        raise NotImplementedError('``vstack`` expects a sequence of arrays as the first argument')
    tup = tuple((atleast_2d(x) for x in tup))
    return concatenate(tup, axis=0, allow_unknown_chunksizes=allow_unknown_chunksizes)

@derived_from(np)
def hstack(tup, allow_unknown_chunksizes=False):
    if False:
        print('Hello World!')
    if isinstance(tup, Array):
        raise NotImplementedError('``hstack`` expects a sequence of arrays as the first argument')
    if all((x.ndim == 1 for x in tup)):
        return concatenate(tup, axis=0, allow_unknown_chunksizes=allow_unknown_chunksizes)
    else:
        return concatenate(tup, axis=1, allow_unknown_chunksizes=allow_unknown_chunksizes)

@derived_from(np)
def dstack(tup, allow_unknown_chunksizes=False):
    if False:
        print('Hello World!')
    if isinstance(tup, Array):
        raise NotImplementedError('``dstack`` expects a sequence of arrays as the first argument')
    tup = tuple((atleast_3d(x) for x in tup))
    return concatenate(tup, axis=2, allow_unknown_chunksizes=allow_unknown_chunksizes)

@derived_from(np)
def swapaxes(a, axis1, axis2):
    if False:
        for i in range(10):
            print('nop')
    if axis1 == axis2:
        return a
    if axis1 < 0:
        axis1 = axis1 + a.ndim
    if axis2 < 0:
        axis2 = axis2 + a.ndim
    ind = list(range(a.ndim))
    out = list(ind)
    (out[axis1], out[axis2]) = (axis2, axis1)
    return blockwise(np.swapaxes, out, a, ind, axis1=axis1, axis2=axis2, dtype=a.dtype)

@derived_from(np)
def transpose(a, axes=None):
    if False:
        i = 10
        return i + 15
    if axes:
        if len(axes) != a.ndim:
            raise ValueError("axes don't match array")
        axes = tuple((d + a.ndim if d < 0 else d for d in axes))
    else:
        axes = tuple(range(a.ndim))[::-1]
    return blockwise(np.transpose, axes, a, tuple(range(a.ndim)), dtype=a.dtype, axes=axes)

def flip(m, axis=None):
    if False:
        return 10
    '\n    Reverse element order along axis.\n\n    Parameters\n    ----------\n    m : array_like\n        Input array.\n    axis : None or int or tuple of ints, optional\n        Axis or axes to reverse element order of. None will reverse all axes.\n\n    Returns\n    -------\n    dask.array.Array\n        The flipped array.\n    '
    m = asanyarray(m)
    sl = m.ndim * [slice(None)]
    if axis is None:
        axis = range(m.ndim)
    if not isinstance(axis, Iterable):
        axis = (axis,)
    try:
        for ax in axis:
            sl[ax] = slice(None, None, -1)
    except IndexError as e:
        raise ValueError(f'`axis` of {str(axis)} invalid for {str(m.ndim)}-D array') from e
    sl = tuple(sl)
    return m[sl]

@derived_from(np)
def flipud(m):
    if False:
        return 10
    return flip(m, 0)

@derived_from(np)
def fliplr(m):
    if False:
        return 10
    return flip(m, 1)

@derived_from(np)
def rot90(m, k=1, axes=(0, 1)):
    if False:
        return 10
    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError('len(axes) must be 2.')
    m = asanyarray(m)
    if axes[0] == axes[1] or np.absolute(axes[0] - axes[1]) == m.ndim:
        raise ValueError('Axes must be different.')
    if axes[0] >= m.ndim or axes[0] < -m.ndim or axes[1] >= m.ndim or (axes[1] < -m.ndim):
        raise ValueError(f'Axes={axes} out of range for array of ndim={m.ndim}.')
    k %= 4
    if k == 0:
        return m[:]
    if k == 2:
        return flip(flip(m, axes[0]), axes[1])
    axes_list = list(range(0, m.ndim))
    (axes_list[axes[0]], axes_list[axes[1]]) = (axes_list[axes[1]], axes_list[axes[0]])
    if k == 1:
        return transpose(flip(m, axes[1]), axes_list)
    else:
        return flip(transpose(m, axes_list), axes[1])

def _tensordot(a, b, axes, is_sparse):
    if False:
        while True:
            i = 10
    x = max([a, b], key=lambda x: x.__array_priority__)
    tensordot = tensordot_lookup.dispatch(type(x))
    x = tensordot(a, b, axes=axes)
    if is_sparse and len(axes[0]) == 1:
        return x
    else:
        ind = [slice(None, None)] * x.ndim
        for a in sorted(axes[0]):
            ind.insert(a, None)
        x = x[tuple(ind)]
        return x

def _tensordot_is_sparse(x):
    if False:
        print('Hello World!')
    is_sparse = 'sparse' in str(type(x._meta))
    if is_sparse:
        is_sparse = 'sparse._coo.core.COO' not in str(type(x._meta))
    return is_sparse

@derived_from(np)
def tensordot(lhs, rhs, axes=2):
    if False:
        print('Hello World!')
    if not isinstance(lhs, Array):
        lhs = from_array(lhs)
    if not isinstance(rhs, Array):
        rhs = from_array(rhs)
    if isinstance(axes, Iterable):
        (left_axes, right_axes) = axes
    else:
        left_axes = tuple(range(lhs.ndim - axes, lhs.ndim))
        right_axes = tuple(range(0, axes))
    if isinstance(left_axes, Integral):
        left_axes = (left_axes,)
    if isinstance(right_axes, Integral):
        right_axes = (right_axes,)
    if isinstance(left_axes, list):
        left_axes = tuple(left_axes)
    if isinstance(right_axes, list):
        right_axes = tuple(right_axes)
    is_sparse = _tensordot_is_sparse(lhs) or _tensordot_is_sparse(rhs)
    if is_sparse and len(left_axes) == 1:
        concatenate = True
    else:
        concatenate = False
    dt = np.promote_types(lhs.dtype, rhs.dtype)
    left_index = list(range(lhs.ndim))
    right_index = list(range(lhs.ndim, lhs.ndim + rhs.ndim))
    out_index = left_index + right_index
    adjust_chunks = {}
    for (l, r) in zip(left_axes, right_axes):
        out_index.remove(right_index[r])
        right_index[r] = left_index[l]
        if concatenate:
            out_index.remove(left_index[l])
        else:
            adjust_chunks[left_index[l]] = lambda c: 1
    intermediate = blockwise(_tensordot, out_index, lhs, left_index, rhs, right_index, dtype=dt, concatenate=concatenate, adjust_chunks=adjust_chunks, axes=(left_axes, right_axes), is_sparse=is_sparse)
    if concatenate:
        return intermediate
    else:
        return intermediate.sum(axis=left_axes)

@derived_from(np, ua_args=['out'])
def dot(a, b):
    if False:
        return 10
    return tensordot(a, b, axes=((a.ndim - 1,), (b.ndim - 2,)))

@derived_from(np)
def vdot(a, b):
    if False:
        while True:
            i = 10
    return dot(a.conj().ravel(), b.ravel())

def _chunk_sum(a, axis=None, dtype=None, keepdims=None):
    if False:
        i = 10
        return i + 15
    if type(a) is list:
        out = reduce(partial(np.add, dtype=dtype), a)
    else:
        out = a
    if keepdims:
        return out
    else:
        return out.squeeze(axis[0])

def _sum_wo_cat(a, axis=None, dtype=None):
    if False:
        i = 10
        return i + 15
    if dtype is None:
        dtype = getattr(np.zeros(1, dtype=a.dtype).sum(), 'dtype', object)
    if a.shape[axis] == 1:
        return a.squeeze(axis)
    return reduction(a, _chunk_sum, _chunk_sum, axis=axis, dtype=dtype, concatenate=False)

def _matmul(a, b):
    if False:
        i = 10
        return i + 15
    xp = np
    if is_cupy_type(a):
        import cupy
        xp = cupy
    chunk = xp.matmul(a, b)
    return chunk[..., xp.newaxis, :]

@derived_from(np)
def matmul(a, b):
    if False:
        for i in range(10):
            print('nop')
    a = asanyarray(a)
    b = asanyarray(b)
    if a.ndim == 0 or b.ndim == 0:
        raise ValueError('`matmul` does not support scalars.')
    a_is_1d = False
    if a.ndim == 1:
        a_is_1d = True
        a = a[np.newaxis, :]
    b_is_1d = False
    if b.ndim == 1:
        b_is_1d = True
        b = b[:, np.newaxis]
    if a.ndim < b.ndim:
        a = a[(b.ndim - a.ndim) * (np.newaxis,)]
    elif a.ndim > b.ndim:
        b = b[(a.ndim - b.ndim) * (np.newaxis,)]
    out_ind = tuple(range(a.ndim + 1))
    lhs_ind = tuple(range(a.ndim))
    rhs_ind = tuple(range(a.ndim - 2)) + (lhs_ind[-1], a.ndim)
    out = blockwise(_matmul, out_ind, a, lhs_ind, b, rhs_ind, adjust_chunks={lhs_ind[-1]: 1}, dtype=result_type(a, b), concatenate=False)
    out = _sum_wo_cat(out, axis=-2)
    if a_is_1d:
        out = out.squeeze(-2)
    if b_is_1d:
        out = out.squeeze(-1)
    return out

@derived_from(np)
def outer(a, b):
    if False:
        return 10
    a = a.flatten()
    b = b.flatten()
    dtype = np.outer(a.dtype.type(), b.dtype.type()).dtype
    return blockwise(np.outer, 'ij', a, 'i', b, 'j', dtype=dtype)

def _inner_apply_along_axis(arr, func1d, func1d_axis, func1d_args, func1d_kwargs):
    if False:
        for i in range(10):
            print('nop')
    return np.apply_along_axis(func1d, func1d_axis, arr, *func1d_args, **func1d_kwargs)

@derived_from(np)
def apply_along_axis(func1d, axis, arr, *args, dtype=None, shape=None, **kwargs):
    if False:
        return 10
    '\n    This is a blocked variant of :func:`numpy.apply_along_axis` implemented via\n    :func:`dask.array.map_blocks`\n\n    Notes\n    -----\n    If either of `dtype` or `shape` are not provided, Dask attempts to\n    determine them by calling `func1d` on a dummy array. This may produce\n    incorrect values for `dtype` or `shape`, so we recommend providing them.\n    '
    arr = asarray(arr)
    axis = len(arr.shape[:axis])
    if shape is None or dtype is None:
        test_data = np.ones((1,), dtype=arr.dtype)
        test_result = np.array(func1d(test_data, *args, **kwargs))
        if shape is None:
            shape = test_result.shape
        if dtype is None:
            dtype = test_result.dtype
    arr = arr.rechunk(arr.chunks[:axis] + (arr.shape[axis:axis + 1],) + arr.chunks[axis + 1:])
    result = arr.map_blocks(_inner_apply_along_axis, name=funcname(func1d) + '-along-axis', dtype=dtype, chunks=arr.chunks[:axis] + shape + arr.chunks[axis + 1:], drop_axis=axis, new_axis=list(range(axis, axis + len(shape), 1)), func1d=func1d, func1d_axis=axis, func1d_args=args, func1d_kwargs=kwargs)
    return result

@derived_from(np)
def apply_over_axes(func, a, axes):
    if False:
        for i in range(10):
            print('nop')
    a = asarray(a)
    try:
        axes = tuple(axes)
    except TypeError:
        axes = (axes,)
    sl = a.ndim * (slice(None),)
    result = a
    for i in axes:
        result = apply_along_axis(func, i, result, 0)
        if result.ndim == a.ndim - 1:
            result = result[sl[:i] + (None,)]
        elif result.ndim != a.ndim:
            raise ValueError('func must either preserve dimensionality of the input or reduce it by one.')
    return result

@derived_from(np)
def ptp(a, axis=None):
    if False:
        for i in range(10):
            print('nop')
    return a.max(axis=axis) - a.min(axis=axis)

@derived_from(np)
def diff(a, n=1, axis=-1, prepend=None, append=None):
    if False:
        return 10
    a = asarray(a)
    n = int(n)
    axis = int(axis)
    if n == 0:
        return a
    if n < 0:
        raise ValueError('order must be non-negative but got %d' % n)
    combined = []
    if prepend is not None:
        prepend = asarray_safe(prepend, like=meta_from_array(a))
        if prepend.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            prepend = broadcast_to(prepend, tuple(shape))
        combined.append(prepend)
    combined.append(a)
    if append is not None:
        append = asarray_safe(append, like=meta_from_array(a))
        if append.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            append = np.broadcast_to(append, tuple(shape))
        combined.append(append)
    if len(combined) > 1:
        a = concatenate(combined, axis)
    sl_1 = a.ndim * [slice(None)]
    sl_2 = a.ndim * [slice(None)]
    sl_1[axis] = slice(1, None)
    sl_2[axis] = slice(None, -1)
    sl_1 = tuple(sl_1)
    sl_2 = tuple(sl_2)
    r = a
    for _ in range(n):
        r = r[sl_1] - r[sl_2]
    return r

@derived_from(np)
def ediff1d(ary, to_end=None, to_begin=None):
    if False:
        print('Hello World!')
    ary = asarray(ary)
    aryf = ary.flatten()
    r = aryf[1:] - aryf[:-1]
    r = [r]
    if to_begin is not None:
        r = [asarray(to_begin).flatten()] + r
    if to_end is not None:
        r = r + [asarray(to_end).flatten()]
    r = concatenate(r)
    return r

def _gradient_kernel(x, block_id, coord, axis, array_locs, grad_kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    x: nd-array\n        array of one block\n    coord: 1d-array or scalar\n        coordinate along which the gradient is computed.\n    axis: int\n        axis along which the gradient is computed\n    array_locs:\n        actual location along axis. None if coordinate is scalar\n    grad_kwargs:\n        keyword to be passed to np.gradient\n    '
    block_loc = block_id[axis]
    if array_locs is not None:
        coord = coord[array_locs[0][block_loc]:array_locs[1][block_loc]]
    grad = np.gradient(x, coord, axis=axis, **grad_kwargs)
    return grad

@derived_from(np)
def gradient(f, *varargs, axis=None, **kwargs):
    if False:
        print('Hello World!')
    f = asarray(f)
    kwargs['edge_order'] = math.ceil(kwargs.get('edge_order', 1))
    if kwargs['edge_order'] > 2:
        raise ValueError('edge_order must be less than or equal to 2.')
    drop_result_list = False
    if axis is None:
        axis = tuple(range(f.ndim))
    elif isinstance(axis, Integral):
        drop_result_list = True
        axis = (axis,)
    axis = validate_axis(axis, f.ndim)
    if len(axis) != len(set(axis)):
        raise ValueError('duplicate axes not allowed')
    axis = tuple((ax % f.ndim for ax in axis))
    if varargs == ():
        varargs = (1,)
    if len(varargs) == 1:
        varargs = len(axis) * varargs
    if len(varargs) != len(axis):
        raise TypeError('Spacing must either be a single scalar, or a scalar / 1d-array per axis')
    if issubclass(f.dtype.type, (np.bool_, Integral)):
        f = f.astype(float)
    elif issubclass(f.dtype.type, Real) and f.dtype.itemsize < 4:
        f = f.astype(float)
    results = []
    for (i, ax) in enumerate(axis):
        for c in f.chunks[ax]:
            if np.min(c) < kwargs['edge_order'] + 1:
                raise ValueError('Chunk size must be larger than edge_order + 1. Minimum chunk for axis {} is {}. Rechunk to proceed.'.format(ax, np.min(c)))
        if np.isscalar(varargs[i]):
            array_locs = None
        else:
            if isinstance(varargs[i], Array):
                raise NotImplementedError('dask array coordinated is not supported.')
            chunk = np.array(f.chunks[ax])
            array_loc_stop = np.cumsum(chunk) + 1
            array_loc_start = array_loc_stop - chunk - 2
            array_loc_stop[-1] -= 1
            array_loc_start[0] = 0
            array_locs = (array_loc_start, array_loc_stop)
        results.append(f.map_overlap(_gradient_kernel, dtype=f.dtype, depth={j: 1 if j == ax else 0 for j in range(f.ndim)}, boundary='none', coord=varargs[i], axis=ax, array_locs=array_locs, grad_kwargs=kwargs))
    if drop_result_list:
        results = results[0]
    return results

def _bincount_agg(bincounts, dtype, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(bincounts, list):
        return bincounts
    n = max(map(len, bincounts))
    out = np.zeros_like(bincounts[0], shape=n, dtype=dtype)
    for b in bincounts:
        out[:len(b)] += b
    return out

@derived_from(np)
def bincount(x, weights=None, minlength=0, split_every=None):
    if False:
        return 10
    if x.ndim != 1:
        raise ValueError('Input array must be one dimensional. Try using x.ravel()')
    if weights is not None:
        if weights.chunks != x.chunks:
            raise ValueError('Chunks of input array x and weights must match.')
    token = tokenize(x, weights, minlength)
    args = [x, 'i']
    if weights is not None:
        meta = array_safe(np.bincount([1], weights=[1]), like=meta_from_array(x))
        args.extend([weights, 'i'])
    else:
        meta = array_safe(np.bincount([]), like=meta_from_array(x))
    if minlength == 0:
        output_size = (np.nan,)
    else:
        output_size = (minlength,)
    chunked_counts = blockwise(partial(np.bincount, minlength=minlength), 'i', *args, token=token, meta=meta)
    chunked_counts._chunks = (output_size * len(chunked_counts.chunks[0]), *chunked_counts.chunks[1:])
    from dask.array.reductions import _tree_reduce
    output = _tree_reduce(chunked_counts, aggregate=partial(_bincount_agg, dtype=meta.dtype), axis=(0,), keepdims=True, dtype=meta.dtype, split_every=split_every, concatenate=False)
    output._chunks = (output_size, *chunked_counts.chunks[1:])
    output._meta = meta
    return output

@derived_from(np)
def digitize(a, bins, right=False):
    if False:
        print('Hello World!')
    bins = asarray_safe(bins, like=meta_from_array(a))
    dtype = np.digitize(asarray_safe([0], like=bins), bins, right=False).dtype
    return a.map_blocks(np.digitize, dtype=dtype, bins=bins, right=right)

def _searchsorted_block(x, y, side):
    if False:
        return 10
    res = np.searchsorted(x, y, side=side)
    res[res == 0] = -1
    return res[np.newaxis, :]

@derived_from(np)
def searchsorted(a, v, side='left', sorter=None):
    if False:
        print('Hello World!')
    if a.ndim != 1:
        raise ValueError('Input array a must be one dimensional')
    if sorter is not None:
        raise NotImplementedError('da.searchsorted with a sorter argument is not supported')
    meta = np.searchsorted(a._meta, v._meta)
    out = blockwise(_searchsorted_block, list(range(v.ndim + 1)), a, [0], v, list(range(1, v.ndim + 1)), side, None, meta=meta, adjust_chunks={0: 1})
    a_chunk_sizes = array_safe((0, *a.chunks[0]), like=meta_from_array(a))
    a_chunk_offsets = np.cumsum(a_chunk_sizes)[:-1]
    a_chunk_offsets = a_chunk_offsets[(Ellipsis,) + v.ndim * (np.newaxis,)]
    a_offsets = asarray(a_chunk_offsets, chunks=1)
    out = where(out < 0, out, out + a_offsets)
    out = out.max(axis=0)
    out[out == -1] = 0
    return out

def _linspace_from_delayed(start, stop, num=50):
    if False:
        return 10
    linspace_name = 'linspace-' + tokenize(start, stop, num)
    ((start_ref, stop_ref, num_ref), deps) = unpack_collections([start, stop, num])
    if len(deps) == 0:
        return np.linspace(start, stop, num=num)
    linspace_dsk = {(linspace_name, 0): (np.linspace, start_ref, stop_ref, num_ref)}
    linspace_graph = HighLevelGraph.from_collections(linspace_name, linspace_dsk, dependencies=deps)
    chunks = ((np.nan,),) if is_dask_collection(num) else ((num,),)
    return Array(linspace_graph, linspace_name, chunks, dtype=float)

def _block_hist(x, bins, range=None, weights=None):
    if False:
        for i in range(10):
            print('nop')
    return np.histogram(x, bins, range=range, weights=weights)[0][np.newaxis]

def histogram(a, bins=None, range=None, normed=False, weights=None, density=None):
    if False:
        while True:
            i = 10
    '\n    Blocked variant of :func:`numpy.histogram`.\n\n    Parameters\n    ----------\n    a : dask.array.Array\n        Input data; the histogram is computed over the flattened\n        array. If the ``weights`` argument is used, the chunks of\n        ``a`` are accessed to check chunking compatibility between\n        ``a`` and ``weights``. If ``weights`` is ``None``, a\n        :py:class:`dask.dataframe.Series` object can be passed as\n        input data.\n    bins : int or sequence of scalars, optional\n        Either an iterable specifying the ``bins`` or the number of ``bins``\n        and a ``range`` argument is required as computing ``min`` and ``max``\n        over blocked arrays is an expensive operation that must be performed\n        explicitly.\n        If `bins` is an int, it defines the number of equal-width\n        bins in the given range (10, by default). If `bins` is a\n        sequence, it defines a monotonically increasing array of bin edges,\n        including the rightmost edge, allowing for non-uniform bin widths.\n    range : (float, float), optional\n        The lower and upper range of the bins.  If not provided, range\n        is simply ``(a.min(), a.max())``.  Values outside the range are\n        ignored. The first element of the range must be less than or\n        equal to the second. `range` affects the automatic bin\n        computation as well. While bin width is computed to be optimal\n        based on the actual data within `range`, the bin count will fill\n        the entire range including portions containing no data.\n    normed : bool, optional\n        This is equivalent to the ``density`` argument, but produces incorrect\n        results for unequal bin widths. It should not be used.\n    weights : dask.array.Array, optional\n        A dask.array.Array of weights, of the same block structure as ``a``.  Each value in\n        ``a`` only contributes its associated weight towards the bin count\n        (instead of 1). If ``density`` is True, the weights are\n        normalized, so that the integral of the density over the range\n        remains 1.\n    density : bool, optional\n        If ``False``, the result will contain the number of samples in\n        each bin. If ``True``, the result is the value of the\n        probability *density* function at the bin, normalized such that\n        the *integral* over the range is 1. Note that the sum of the\n        histogram values will not be equal to 1 unless bins of unity\n        width are chosen; it is not a probability *mass* function.\n        Overrides the ``normed`` keyword if given.\n        If ``density`` is True, ``bins`` cannot be a single-number delayed\n        value. It must be a concrete number, or a (possibly-delayed)\n        array/sequence of the bin edges.\n\n    Returns\n    -------\n    hist : dask Array\n        The values of the histogram. See `density` and `weights` for a\n        description of the possible semantics.\n    bin_edges : dask Array of dtype float\n        Return the bin edges ``(length(hist)+1)``.\n\n    Examples\n    --------\n    Using number of bins and range:\n\n    >>> import dask.array as da\n    >>> import numpy as np\n    >>> x = da.from_array(np.arange(10000), chunks=10)\n    >>> h, bins = da.histogram(x, bins=10, range=[0, 10000])\n    >>> bins\n    array([    0.,  1000.,  2000.,  3000.,  4000.,  5000.,  6000.,  7000.,\n            8000.,  9000., 10000.])\n    >>> h.compute()\n    array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000])\n\n    Explicitly specifying the bins:\n\n    >>> h, bins = da.histogram(x, bins=np.array([0, 5000, 10000]))\n    >>> bins\n    array([    0,  5000, 10000])\n    >>> h.compute()\n    array([5000, 5000])\n    '
    if isinstance(bins, Array):
        scalar_bins = bins.ndim == 0
    elif isinstance(bins, Delayed):
        scalar_bins = bins._length is None or bins._length == 1
    else:
        scalar_bins = np.ndim(bins) == 0
    if bins is None or (scalar_bins and range is None):
        raise ValueError('dask.array.histogram requires either specifying bins as an iterable or specifying both a range and the number of bins')
    if weights is not None and weights.chunks != a.chunks:
        raise ValueError('Input array and weights must have the same chunked structure')
    if normed is not False:
        raise ValueError('The normed= keyword argument has been deprecated. Please use density instead. See the numpy.histogram docstring for more information.')
    if density and scalar_bins and isinstance(bins, (Array, Delayed)):
        raise NotImplementedError('When `density` is True, `bins` cannot be a scalar Dask object. It must be a concrete number or a (possibly-delayed) array/sequence of bin edges.')
    for (argname, val) in [('bins', bins), ('range', range), ('weights', weights)]:
        if not isinstance(bins, (Array, Delayed)) and is_dask_collection(bins):
            raise TypeError('Dask types besides Array and Delayed are not supported for `histogram`. For argument `{}`, got: {!r}'.format(argname, val))
    if range is not None:
        try:
            if len(range) != 2:
                raise ValueError(f'range must be a sequence or array of length 2, but got {len(range)} items')
            if isinstance(range, (Array, np.ndarray)) and range.shape != (2,):
                raise ValueError(f'range must be a 1-dimensional array of two items, but got an array of shape {range.shape}')
        except TypeError:
            raise TypeError(f'Expected a sequence or array for range, not {range}') from None
    token = tokenize(a, bins, range, weights, density)
    name = 'histogram-sum-' + token
    if scalar_bins:
        bins = _linspace_from_delayed(range[0], range[1], bins + 1)
    else:
        if not isinstance(bins, (Array, np.ndarray)):
            bins = asarray(bins)
        if bins.ndim != 1:
            raise ValueError(f'bins must be a 1-dimensional array or sequence, got shape {bins.shape}')
    ((bins_ref, range_ref), deps) = unpack_collections([bins, range])
    if weights is None:
        dsk = {(name, i, 0): (_block_hist, k, bins_ref, range_ref) for (i, k) in enumerate(flatten(a.__dask_keys__()))}
        dtype = np.histogram([])[0].dtype
    else:
        a_keys = flatten(a.__dask_keys__())
        w_keys = flatten(weights.__dask_keys__())
        dsk = {(name, i, 0): (_block_hist, k, bins_ref, range_ref, w) for (i, (k, w)) in enumerate(zip(a_keys, w_keys))}
        dtype = weights.dtype
    deps = (a,) + deps
    if weights is not None:
        deps += (weights,)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=deps)
    nchunks = len(list(flatten(a.__dask_keys__())))
    nbins = bins.size - 1
    chunks = ((1,) * nchunks, (nbins,))
    mapped = Array(graph, name, chunks, dtype=dtype)
    n = mapped.sum(axis=0)
    if density is not None:
        if density:
            db = asarray(np.diff(bins).astype(float), chunks=n.chunks)
            return (n / db / n.sum(), bins)
        else:
            return (n, bins)
    else:
        return (n, bins)

def histogram2d(x, y, bins=10, range=None, normed=None, weights=None, density=None):
    if False:
        i = 10
        return i + 15
    'Blocked variant of :func:`numpy.histogram2d`.\n\n    Parameters\n    ----------\n    x : dask.array.Array\n        An array containing the `x`-coordinates of the points to be\n        histogrammed.\n    y : dask.array.Array\n        An array containing the `y`-coordinates of the points to be\n        histogrammed.\n    bins : sequence of arrays describing bin edges, int, or sequence of ints\n        The bin specification. See the `bins` argument description for\n        :py:func:`histogramdd` for a complete description of all\n        possible bin configurations (this function is a 2D specific\n        version of histogramdd).\n    range : tuple of pairs, optional.\n        The leftmost and rightmost edges of the bins along each\n        dimension when integers are passed to `bins`; of the form:\n        ((xmin, xmax), (ymin, ymax)).\n    normed : bool, optional\n        An alias for the density argument that behaves identically. To\n        avoid confusion with the broken argument in the `histogram`\n        function, `density` should be preferred.\n    weights : dask.array.Array, optional\n        An array of values weighing each sample in the input data. The\n        chunks of the weights must be identical to the chunking along\n        the 0th (row) axis of the data sample.\n    density : bool, optional\n        If False (the default) return the number of samples in each\n        bin. If True, the returned array represents the probability\n        density function at each bin.\n\n    Returns\n    -------\n    dask.array.Array\n        The values of the histogram.\n    dask.array.Array\n        The edges along the `x`-dimension.\n    dask.array.Array\n        The edges along the `y`-dimension.\n\n    See Also\n    --------\n    histogram\n    histogramdd\n\n    Examples\n    --------\n    >>> import dask.array as da\n    >>> x = da.array([2, 4, 2, 4, 2, 4])\n    >>> y = da.array([2, 2, 4, 4, 2, 4])\n    >>> bins = 2\n    >>> range = ((0, 6), (0, 6))\n    >>> h, xedges, yedges = da.histogram2d(x, y, bins=bins, range=range)\n    >>> h\n    dask.array<sum-aggregate, shape=(2, 2), dtype=float64, chunksize=(2, 2), chunktype=numpy.ndarray>\n    >>> xedges\n    dask.array<array, shape=(3,), dtype=float64, chunksize=(3,), chunktype=numpy.ndarray>\n    >>> h.compute()\n    array([[2., 1.],\n           [1., 2.]])\n    '
    (counts, edges) = histogramdd((x, y), bins=bins, range=range, normed=normed, weights=weights, density=density)
    return (counts, edges[0], edges[1])

def _block_histogramdd_rect(sample, bins, range, weights):
    if False:
        while True:
            i = 10
    'Call numpy.histogramdd for a blocked/chunked calculation.\n\n    Slurps the result into an additional outer axis; this new axis\n    will be used to stack chunked calls of the numpy function and add\n    them together later.\n\n    Returns\n    -------\n    :py:object:`np.ndarray`\n        NumPy array with an additional outer dimension.\n\n    '
    return np.histogramdd(sample, bins, range=range, weights=weights)[0:1]

def _block_histogramdd_multiarg(*args):
    if False:
        i = 10
        return i + 15
    'Call numpy.histogramdd for a multi argument blocked/chunked calculation.\n\n    Slurps the result into an additional outer axis; this new axis\n    will be used to stack chunked calls of the numpy function and add\n    them together later.\n\n    The last three arguments _must be_ (bins, range, weights).\n\n    The difference between this function and\n    _block_histogramdd_rect is that here we expect the sample\n    to be composed of multiple arguments (multiple 1D arrays, each one\n    representing a coordinate), while _block_histogramdd_rect\n    expects a single rectangular (2D array where columns are\n    coordinates) sample.\n\n    '
    (bins, range, weights) = args[-3:]
    sample = args[:-3]
    return np.histogramdd(sample, bins=bins, range=range, weights=weights)[0:1]

def histogramdd(sample, bins, range=None, normed=None, weights=None, density=None):
    if False:
        print('Hello World!')
    'Blocked variant of :func:`numpy.histogramdd`.\n\n    Chunking of the input data (``sample``) is only allowed along the\n    0th (row) axis (the axis corresponding to the total number of\n    samples). Data chunked along the 1st axis (column) axis is not\n    compatible with this function. If weights are used, they must be\n    chunked along the 0th axis identically to the input sample.\n\n    An example setup for a three dimensional histogram, where the\n    sample shape is ``(8, 3)`` and weights are shape ``(8,)``, sample\n    chunks would be ``((4, 4), (3,))`` and the weights chunks would be\n    ``((4, 4),)`` a table of the structure:\n\n    +-------+-----------------------+-----------+\n    |       |      sample (8 x 3)   |  weights  |\n    +=======+=====+=====+=====+=====+=====+=====+\n    | chunk | row | `x` | `y` | `z` | row | `w` |\n    +-------+-----+-----+-----+-----+-----+-----+\n    |       |   0 |   5 |   6 |   6 |   0 | 0.5 |\n    |       +-----+-----+-----+-----+-----+-----+\n    |       |   1 |   8 |   9 |   2 |   1 | 0.8 |\n    |   0   +-----+-----+-----+-----+-----+-----+\n    |       |   2 |   3 |   3 |   1 |   2 | 0.3 |\n    |       +-----+-----+-----+-----+-----+-----+\n    |       |   3 |   2 |   5 |   6 |   3 | 0.7 |\n    +-------+-----+-----+-----+-----+-----+-----+\n    |       |   4 |   3 |   1 |   1 |   4 | 0.3 |\n    |       +-----+-----+-----+-----+-----+-----+\n    |       |   5 |   3 |   2 |   9 |   5 | 1.3 |\n    |   1   +-----+-----+-----+-----+-----+-----+\n    |       |   6 |   8 |   1 |   5 |   6 | 0.8 |\n    |       +-----+-----+-----+-----+-----+-----+\n    |       |   7 |   3 |   5 |   3 |   7 | 0.7 |\n    +-------+-----+-----+-----+-----+-----+-----+\n\n    If the sample 0th dimension and weight 0th (row) dimension are\n    chunked differently, a ``ValueError`` will be raised. If\n    coordinate groupings ((x, y, z) trios) are separated by a chunk\n    boundry, then a ``ValueError`` will be raised. We suggest that you\n    rechunk your data if it is of that form.\n\n    The chunks property of the data (and optional weights) are used to\n    check for compatibility with the blocked algorithm (as described\n    above); therefore, you must call `to_dask_array` on a collection\n    from ``dask.dataframe``, i.e. :class:`dask.dataframe.Series` or\n    :class:`dask.dataframe.DataFrame`.\n\n    The function is also compatible with `x`, `y`, and `z` being\n    individual 1D arrays with equal chunking. In that case, the data\n    should be passed as a tuple: ``histogramdd((x, y, z), ...)``\n\n    Parameters\n    ----------\n    sample : dask.array.Array (N, D) or sequence of dask.array.Array\n        Multidimensional data to be histogrammed.\n\n        Note the unusual interpretation of a sample when it is a\n        sequence of dask Arrays:\n\n        * When a (N, D) dask Array, each row is an entry in the sample\n          (coordinate in D dimensional space).\n        * When a sequence of dask Arrays, each element in the sequence\n          is the array of values for a single coordinate.\n    bins : sequence of arrays describing bin edges, int, or sequence of ints\n        The bin specification.\n\n        The possible binning configurations are:\n\n        * A sequence of arrays describing the monotonically increasing\n          bin edges along each dimension.\n        * A single int describing the total number of bins that will\n          be used in each dimension (this requires the ``range``\n          argument to be defined).\n        * A sequence of ints describing the total number of bins to be\n          used in each dimension (this requires the ``range`` argument\n          to be defined).\n\n        When bins are described by arrays, the rightmost edge is\n        included. Bins described by arrays also allows for non-uniform\n        bin widths.\n    range : sequence of pairs, optional\n        A sequence of length D, each a (min, max) tuple giving the\n        outer bin edges to be used if the edges are not given\n        explicitly in `bins`. If defined, this argument is required to\n        have an entry for each dimension. Unlike\n        :func:`numpy.histogramdd`, if `bins` does not define bin\n        edges, this argument is required (this function will not\n        automatically use the min and max of of the value in a given\n        dimension because the input data may be lazy in dask).\n    normed : bool, optional\n        An alias for the density argument that behaves identically. To\n        avoid confusion with the broken argument to `histogram`,\n        `density` should be preferred.\n    weights : dask.array.Array, optional\n        An array of values weighing each sample in the input data. The\n        chunks of the weights must be identical to the chunking along\n        the 0th (row) axis of the data sample.\n    density : bool, optional\n        If ``False`` (default), the returned array represents the\n        number of samples in each bin. If ``True``, the returned array\n        represents the probability density function at each bin.\n\n    See Also\n    --------\n    histogram\n\n    Returns\n    -------\n    dask.array.Array\n        The values of the histogram.\n    list(dask.array.Array)\n        Sequence of arrays representing the bin edges along each\n        dimension.\n\n    Examples\n    --------\n    Computing the histogram in 5 blocks using different bin edges\n    along each dimension:\n\n    >>> import dask.array as da\n    >>> x = da.random.uniform(0, 1, size=(1000, 3), chunks=(200, 3))\n    >>> edges = [\n    ...     np.linspace(0, 1, 5), # 4 bins in 1st dim\n    ...     np.linspace(0, 1, 6), # 5 in the 2nd\n    ...     np.linspace(0, 1, 4), # 3 in the 3rd\n    ... ]\n    >>> h, edges = da.histogramdd(x, bins=edges)\n    >>> result = h.compute()\n    >>> result.shape\n    (4, 5, 3)\n\n    Defining the bins by total number and their ranges, along with\n    using weights:\n\n    >>> bins = (4, 5, 3)\n    >>> ranges = ((0, 1),) * 3  # expands to ((0, 1), (0, 1), (0, 1))\n    >>> w = da.random.uniform(0, 1, size=(1000,), chunks=x.chunksize[0])\n    >>> h, edges = da.histogramdd(x, bins=bins, range=ranges, weights=w)\n    >>> np.isclose(h.sum().compute(), w.sum().compute())\n    True\n\n    Using a sequence of 1D arrays as the input:\n\n    >>> x = da.array([2, 4, 2, 4, 2, 4])\n    >>> y = da.array([2, 2, 4, 4, 2, 4])\n    >>> z = da.array([4, 2, 4, 2, 4, 2])\n    >>> bins = ([0, 3, 6],) * 3\n    >>> h, edges = da.histogramdd((x, y, z), bins)\n    >>> h\n    dask.array<sum-aggregate, shape=(2, 2, 2), dtype=float64, chunksize=(2, 2, 2), chunktype=numpy.ndarray>\n    >>> edges[0]\n    dask.array<array, shape=(3,), dtype=int64, chunksize=(3,), chunktype=numpy.ndarray>\n    >>> h.compute()\n    array([[[0., 2.],\n            [0., 1.]],\n    <BLANKLINE>\n           [[1., 0.],\n            [2., 0.]]])\n    >>> edges[0].compute()\n    array([0, 3, 6])\n    >>> edges[1].compute()\n    array([0, 3, 6])\n    >>> edges[2].compute()\n    array([0, 3, 6])\n\n    '
    if normed is None:
        if density is None:
            density = False
    elif density is None:
        density = normed
    else:
        raise TypeError("Cannot specify both 'normed' and 'density'")
    dc_bins = is_dask_collection(bins)
    if isinstance(bins, (list, tuple)):
        dc_bins = dc_bins or any([is_dask_collection(b) for b in bins])
    dc_range = any([is_dask_collection(r) for r in range]) if range is not None else False
    if dc_bins or dc_range:
        raise NotImplementedError('Passing dask collections to bins=... or range=... is not supported.')
    token = tokenize(sample, bins, range, weights, density)
    name = f'histogramdd-sum-{token}'
    if hasattr(sample, 'shape'):
        if len(sample.shape) != 2:
            raise ValueError('Single array input to histogramdd should be columnar')
        else:
            (_, D) = sample.shape
        n_chunks = sample.numblocks[0]
        rectangular_sample = True
        if sample.shape[1:] != sample.chunksize[1:]:
            raise ValueError('Input array can only be chunked along the 0th axis.')
    elif isinstance(sample, (tuple, list)):
        rectangular_sample = False
        D = len(sample)
        n_chunks = sample[0].numblocks[0]
        for i in _range(1, D):
            if sample[i].chunks != sample[0].chunks:
                raise ValueError('All coordinate arrays must be chunked identically.')
    else:
        raise ValueError('Incompatible sample. Must be a 2D array or a sequence of 1D arrays.')
    for (argname, val) in [('bins', bins), ('range', range), ('weights', weights)]:
        if not isinstance(bins, (Array, Delayed)) and is_dask_collection(bins):
            raise TypeError('Dask types besides Array and Delayed are not supported for `histogramdd`. For argument `{}`, got: {!r}'.format(argname, val))
    if weights is not None:
        if rectangular_sample and weights.chunks[0] != sample.chunks[0]:
            raise ValueError('Input array and weights must have the same shape and chunk structure along the first dimension.')
        elif not rectangular_sample and weights.numblocks[0] != n_chunks:
            raise ValueError('Input arrays and weights must have the same shape and chunk structure.')
    if isinstance(bins, (list, tuple)):
        if len(bins) != D:
            raise ValueError('The dimension of bins must be equal to the dimension of the sample.')
    if range is not None:
        if len(range) != D:
            raise ValueError('range argument requires one entry, a min max pair, per dimension.')
        if not all((len(r) == 2 for r in range)):
            raise ValueError('range argument should be a sequence of pairs')
    if isinstance(bins, int):
        bins = (bins,) * D
    if all((isinstance(b, int) for b in bins)) and all((len(r) == 2 for r in range)):
        edges = [np.linspace(r[0], r[1], b + 1) for (b, r) in zip(bins, range)]
    else:
        edges = [np.asarray(b) for b in bins]
    if rectangular_sample:
        deps = (sample,)
    else:
        deps = tuple(sample)
    if weights is not None:
        w_keys = flatten(weights.__dask_keys__())
        deps += (weights,)
        dtype = weights.dtype
    else:
        w_keys = (None,) * n_chunks
        dtype = np.histogramdd([])[0].dtype
    column_zeros = tuple((0 for _ in _range(D)))
    if rectangular_sample:
        sample_keys = flatten(sample.__dask_keys__())
        dsk = {(name, i, *column_zeros): (_block_histogramdd_rect, k, bins, range, w) for (i, (k, w)) in enumerate(zip(sample_keys, w_keys))}
    else:
        sample_keys = [list(flatten(sample[i].__dask_keys__())) for i in _range(len(sample))]
        fused_on_chunk_keys = [tuple((sample_keys[j][i] for j in _range(D))) for i in _range(n_chunks)]
        dsk = {(name, i, *column_zeros): (_block_histogramdd_multiarg, *(*k, bins, range, w)) for (i, (k, w)) in enumerate(zip(fused_on_chunk_keys, w_keys))}
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=deps)
    all_nbins = tuple(((b.size - 1,) for b in edges))
    stacked_chunks = ((1,) * n_chunks, *all_nbins)
    mapped = Array(graph, name, stacked_chunks, dtype=dtype)
    n = mapped.sum(axis=0)
    if density:
        width_divider = np.ones(n.shape)
        for i in _range(D):
            shape = np.ones(D, int)
            shape[i] = width_divider.shape[i]
            width_divider *= np.diff(edges[i]).reshape(shape)
        width_divider = asarray(width_divider, chunks=n.chunks)
        return (n / width_divider / n.sum(), edges)
    return (n, [asarray(entry) for entry in edges])

@derived_from(np)
def cov(m, y=None, rowvar=1, bias=0, ddof=None):
    if False:
        for i in range(10):
            print('nop')
    if ddof is not None and ddof != int(ddof):
        raise ValueError('ddof must be integer')
    m = asarray(m)
    if y is None:
        dtype = np.result_type(m, np.float64)
    else:
        y = asarray(y)
        dtype = np.result_type(m, y, np.float64)
    X = array(m, ndmin=2, dtype=dtype)
    if X.shape[0] == 1:
        rowvar = 1
    if rowvar:
        N = X.shape[1]
        axis = 0
    else:
        N = X.shape[0]
        axis = 1
    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0
    fact = float(N - ddof)
    if fact <= 0:
        warnings.warn('Degrees of freedom <= 0 for slice', RuntimeWarning)
        fact = 0.0
    if y is not None:
        y = array(y, ndmin=2, dtype=dtype)
        X = concatenate((X, y), axis)
    X = X - X.mean(axis=1 - axis, keepdims=True)
    if not rowvar:
        return (dot(X.T, X.conj()) / fact).squeeze()
    else:
        return (dot(X, X.T.conj()) / fact).squeeze()

@derived_from(np)
def corrcoef(x, y=None, rowvar=1):
    if False:
        return 10
    c = cov(x, y, rowvar)
    if c.shape == ():
        return c / c
    d = diag(c)
    d = d.reshape((d.shape[0], 1))
    sqr_d = sqrt(d)
    return c / sqr_d / sqr_d.T

@implements(np.round)
@derived_from(np)
def round(a, decimals=0):
    if False:
        while True:
            i = 10
    return a.map_blocks(np.round, decimals=decimals, dtype=a.dtype)

@implements(np.ndim)
@derived_from(np)
def ndim(a):
    if False:
        print('Hello World!')
    return a.ndim

@implements(np.iscomplexobj)
@derived_from(np)
def iscomplexobj(x):
    if False:
        print('Hello World!')
    return issubclass(x.dtype.type, np.complexfloating)

def _unique_internal(ar, indices, counts, return_inverse=False):
    if False:
        while True:
            i = 10
    '\n    Helper/wrapper function for :func:`numpy.unique`.\n\n    Uses :func:`numpy.unique` to find the unique values for the array chunk.\n    Given this chunk may not represent the whole array, also take the\n    ``indices`` and ``counts`` that are in 1-to-1 correspondence to ``ar``\n    and reduce them in the same fashion as ``ar`` is reduced. Namely sum\n    any counts that correspond to the same value and take the smallest\n    index that corresponds to the same value.\n\n    To handle the inverse mapping from the unique values to the original\n    array, simply return a NumPy array created with ``arange`` with enough\n    values to correspond 1-to-1 to the unique values. While there is more\n    work needed to be done to create the full inverse mapping for the\n    original array, this provides enough information to generate the\n    inverse mapping in Dask.\n\n    Given Dask likes to have one array returned from functions like\n    ``blockwise``, some formatting is done to stuff all of the resulting arrays\n    into one big NumPy structured array. Dask is then able to handle this\n    object and can split it apart into the separate results on the Dask side,\n    which then can be passed back to this function in concatenated chunks for\n    further reduction or can be return to the user to perform other forms of\n    analysis.\n\n    By handling the problem in this way, it does not matter where a chunk\n    is in a larger array or how big it is. The chunk can still be computed\n    on the same way. Also it does not matter if the chunk is the result of\n    other chunks being run through this function multiple times. The end\n    result will still be just as accurate using this strategy.\n    '
    return_index = indices is not None
    return_counts = counts is not None
    u = np.unique(ar)
    dt = [('values', u.dtype)]
    if return_index:
        dt.append(('indices', np.intp))
    if return_inverse:
        dt.append(('inverse', np.intp))
    if return_counts:
        dt.append(('counts', np.intp))
    r = np.empty(u.shape, dtype=dt)
    r['values'] = u
    if return_inverse:
        r['inverse'] = np.arange(len(r), dtype=np.intp)
    if return_index or return_counts:
        for (i, v) in enumerate(r['values']):
            m = ar == v
            if return_index:
                indices[m].min(keepdims=True, out=r['indices'][i:i + 1])
            if return_counts:
                counts[m].sum(keepdims=True, out=r['counts'][i:i + 1])
    return r

def unique_no_structured_arr(ar, return_index=False, return_inverse=False, return_counts=False):
    if False:
        for i in range(10):
            print('nop')
    if return_index is not False or return_inverse is not False or return_counts is not False:
        raise ValueError("dask.array.unique does not support `return_index`, `return_inverse` or `return_counts` with array types that don't support structured arrays.")
    ar = ar.ravel()
    args = [ar, 'i']
    meta = meta_from_array(ar)
    out = blockwise(np.unique, 'i', *args, meta=meta)
    out._chunks = tuple(((np.nan,) * len(c) for c in out.chunks))
    out_parts = [out]
    name = 'unique-aggregate-' + out.name
    dsk = {(name, 0): (np.unique,) + tuple(((np.concatenate, o.__dask_keys__()) if hasattr(o, '__dask_keys__') else o for o in out_parts))}
    dependencies = [o for o in out_parts if hasattr(o, '__dask_keys__')]
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=dependencies)
    chunks = ((np.nan,),)
    out = Array(graph, name, chunks, meta=meta)
    result = [out]
    if len(result) == 1:
        result = result[0]
    else:
        result = tuple(result)
    return result

@derived_from(np)
def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    if False:
        i = 10
        return i + 15
    try:
        meta = meta_from_array(ar)
        np.empty_like(meta, dtype=[('a', int), ('b', float)])
    except TypeError:
        return unique_no_structured_arr(ar, return_index=return_index, return_inverse=return_inverse, return_counts=return_counts)
    ar = ar.ravel()
    args = [ar, 'i']
    out_dtype = [('values', ar.dtype)]
    if return_index:
        args.extend([arange(ar.shape[0], dtype=np.intp, chunks=ar.chunks[0]), 'i'])
        out_dtype.append(('indices', np.intp))
    else:
        args.extend([None, None])
    if return_counts:
        args.extend([ones((ar.shape[0],), dtype=np.intp, chunks=ar.chunks[0]), 'i'])
        out_dtype.append(('counts', np.intp))
    else:
        args.extend([None, None])
    out = blockwise(_unique_internal, 'i', *args, dtype=out_dtype, return_inverse=False)
    out._chunks = tuple(((np.nan,) * len(c) for c in out.chunks))
    out_parts = [out['values']]
    if return_index:
        out_parts.append(out['indices'])
    else:
        out_parts.append(None)
    if return_counts:
        out_parts.append(out['counts'])
    else:
        out_parts.append(None)
    name = 'unique-aggregate-' + out.name
    dsk = {(name, 0): (_unique_internal,) + tuple(((np.concatenate, o.__dask_keys__()) if hasattr(o, '__dask_keys__') else o for o in out_parts)) + (return_inverse,)}
    out_dtype = [('values', ar.dtype)]
    if return_index:
        out_dtype.append(('indices', np.intp))
    if return_inverse:
        out_dtype.append(('inverse', np.intp))
    if return_counts:
        out_dtype.append(('counts', np.intp))
    dependencies = [o for o in out_parts if hasattr(o, '__dask_keys__')]
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=dependencies)
    chunks = ((np.nan,),)
    out = Array(graph, name, chunks, out_dtype)
    result = [out['values']]
    if return_index:
        result.append(out['indices'])
    if return_inverse:
        mtches = (ar[:, None] == out['values'][None, :]).astype(np.intp)
        result.append((mtches * out['inverse']).sum(axis=1))
    if return_counts:
        result.append(out['counts'])
    if len(result) == 1:
        result = result[0]
    else:
        result = tuple(result)
    return result

def _isin_kernel(element, test_elements, assume_unique=False):
    if False:
        while True:
            i = 10
    values = np.isin(element.ravel(), test_elements, assume_unique=assume_unique)
    return values.reshape(element.shape + (1,) * test_elements.ndim)

@safe_wraps(getattr(np, 'isin', None))
def isin(element, test_elements, assume_unique=False, invert=False):
    if False:
        return 10
    element = asarray(element)
    test_elements = asarray(test_elements)
    element_axes = tuple(range(element.ndim))
    test_axes = tuple((i + element.ndim for i in range(test_elements.ndim)))
    mapped = blockwise(_isin_kernel, element_axes + test_axes, element, element_axes, test_elements, test_axes, adjust_chunks={axis: lambda _: 1 for axis in test_axes}, dtype=bool, assume_unique=assume_unique)
    result = mapped.any(axis=test_axes)
    if invert:
        result = ~result
    return result

@derived_from(np)
def roll(array, shift, axis=None):
    if False:
        for i in range(10):
            print('nop')
    result = array
    if axis is None:
        result = ravel(result)
        if not isinstance(shift, Integral):
            raise TypeError('Expect `shift` to be an instance of Integral when `axis` is None.')
        shift = (shift,)
        axis = (0,)
    else:
        try:
            len(shift)
        except TypeError:
            shift = (shift,)
        try:
            len(axis)
        except TypeError:
            axis = (axis,)
    if len(shift) != len(axis):
        raise ValueError('Must have the same number of shifts as axes.')
    for (i, s) in zip(axis, shift):
        shape = result.shape[i]
        s = 0 if shape == 0 else -s % shape
        sl1 = result.ndim * [slice(None)]
        sl2 = result.ndim * [slice(None)]
        sl1[i] = slice(s, None)
        sl2[i] = slice(None, s)
        sl1 = tuple(sl1)
        sl2 = tuple(sl2)
        result = concatenate([result[sl1], result[sl2]], axis=i)
    result = result.reshape(array.shape)
    result = result.copy() if result is array else result
    return result

@derived_from(np)
def shape(array):
    if False:
        while True:
            i = 10
    return array.shape

@derived_from(np)
def union1d(ar1, ar2):
    if False:
        print('Hello World!')
    return unique(concatenate((ar1.ravel(), ar2.ravel())))

@derived_from(np)
def ravel(array_like):
    if False:
        i = 10
        return i + 15
    return asanyarray(array_like).reshape((-1,))

@derived_from(np)
def expand_dims(a, axis):
    if False:
        print('Hello World!')
    if type(axis) not in (tuple, list):
        axis = (axis,)
    out_ndim = len(axis) + a.ndim
    axis = validate_axis(axis, out_ndim)
    shape_it = iter(a.shape)
    shape = [1 if ax in axis else next(shape_it) for ax in range(out_ndim)]
    return a.reshape(shape)

@derived_from(np)
def squeeze(a, axis=None):
    if False:
        i = 10
        return i + 15
    if axis is None:
        axis = tuple((i for (i, d) in enumerate(a.shape) if d == 1))
    elif not isinstance(axis, tuple):
        axis = (axis,)
    if any((a.shape[i] != 1 for i in axis)):
        raise ValueError('cannot squeeze axis with size other than one')
    axis = validate_axis(axis, a.ndim)
    sl = tuple((0 if i in axis else slice(None) for (i, s) in enumerate(a.shape)))
    if all((s == 0 for s in sl)) and all((s == 1 for s in a.shape)):
        return a.map_blocks(np.squeeze, meta=a._meta, drop_axis=tuple(range(len(a.shape))))
    a = a[sl]
    return a

@derived_from(np)
def compress(condition, a, axis=None):
    if False:
        while True:
            i = 10
    if not is_arraylike(condition):
        condition = np.asarray(condition)
    condition = condition.astype(bool)
    a = asarray(a)
    if condition.ndim != 1:
        raise ValueError('Condition must be one dimensional')
    if axis is None:
        a = a.ravel()
        axis = 0
    axis = validate_axis(axis, a.ndim)
    a = a[tuple((slice(None, len(condition)) if i == axis else slice(None) for i in range(a.ndim)))]
    a = a[tuple((condition if i == axis else slice(None) for i in range(a.ndim)))]
    return a

@derived_from(np)
def extract(condition, arr):
    if False:
        while True:
            i = 10
    condition = asarray(condition).astype(bool)
    arr = asarray(arr)
    return compress(condition.ravel(), arr.ravel())

@derived_from(np)
def take(a, indices, axis=0):
    if False:
        return 10
    axis = validate_axis(axis, a.ndim)
    if isinstance(a, np.ndarray) and isinstance(indices, Array):
        return _take_dask_array_from_numpy(a, indices, axis)
    else:
        return a[(slice(None),) * axis + (indices,)]

def _take_dask_array_from_numpy(a, indices, axis):
    if False:
        print('Hello World!')
    assert isinstance(a, np.ndarray)
    assert isinstance(indices, Array)
    return indices.map_blocks(lambda block: np.take(a, block, axis), chunks=indices.chunks, dtype=a.dtype)

@derived_from(np)
def around(x, decimals=0):
    if False:
        i = 10
        return i + 15
    return map_blocks(partial(np.around, decimals=decimals), x, dtype=x.dtype)

def _asarray_isnull(values):
    if False:
        for i in range(10):
            print('nop')
    import pandas as pd
    return np.asarray(pd.isnull(values))

def isnull(values):
    if False:
        i = 10
        return i + 15
    'pandas.isnull for dask arrays'
    import pandas as pd
    return elemwise(_asarray_isnull, values, dtype='bool')

def notnull(values):
    if False:
        print('Hello World!')
    'pandas.notnull for dask arrays'
    return ~isnull(values)

@derived_from(np)
def isclose(arr1, arr2, rtol=1e-05, atol=1e-08, equal_nan=False):
    if False:
        while True:
            i = 10
    func = partial(np.isclose, rtol=rtol, atol=atol, equal_nan=equal_nan)
    return elemwise(func, arr1, arr2, dtype='bool')

@derived_from(np)
def allclose(arr1, arr2, rtol=1e-05, atol=1e-08, equal_nan=False):
    if False:
        i = 10
        return i + 15
    return isclose(arr1, arr2, rtol=rtol, atol=atol, equal_nan=equal_nan).all()

def variadic_choose(a, *choices):
    if False:
        i = 10
        return i + 15
    return np.choose(a, choices)

@derived_from(np)
def choose(a, choices):
    if False:
        while True:
            i = 10
    return elemwise(variadic_choose, a, *choices)

def _isnonzero_vec(v):
    if False:
        i = 10
        return i + 15
    return bool(np.count_nonzero(v))
_isnonzero_vec = np.vectorize(_isnonzero_vec, otypes=[bool])

def isnonzero(a):
    if False:
        return 10
    if a.dtype.kind in {'U', 'S'}:
        return a.map_blocks(_isnonzero_vec, dtype=bool)
    try:
        np.zeros(tuple(), dtype=a.dtype).astype(bool)
    except ValueError:
        return a.map_blocks(_isnonzero_vec, dtype=bool)
    else:
        return a.astype(bool)

@derived_from(np)
def argwhere(a):
    if False:
        print('Hello World!')
    a = asarray(a)
    nz = isnonzero(a).flatten()
    ind = indices(a.shape, dtype=np.intp, chunks=a.chunks)
    if ind.ndim > 1:
        ind = stack([ind[i].ravel() for i in range(len(ind))], axis=1)
    ind = compress(nz, ind, axis=0)
    return ind

@derived_from(np)
def where(condition, x=None, y=None):
    if False:
        return 10
    if (x is None) != (y is None):
        raise ValueError('either both or neither of x and y should be given')
    if x is None and y is None:
        return nonzero(condition)
    if np.isscalar(condition):
        dtype = result_type(x, y)
        x = asarray(x)
        y = asarray(y)
        shape = broadcast_shapes(x.shape, y.shape)
        out = x if condition else y
        return broadcast_to(out, shape).astype(dtype)
    else:
        return elemwise(np.where, condition, x, y)

@derived_from(np)
def count_nonzero(a, axis=None):
    if False:
        return 10
    return isnonzero(asarray(a)).astype(np.intp).sum(axis=axis)

@derived_from(np)
def flatnonzero(a):
    if False:
        return 10
    return argwhere(asarray(a).ravel())[:, 0]

@derived_from(np)
def nonzero(a):
    if False:
        for i in range(10):
            print('nop')
    ind = argwhere(a)
    if ind.ndim > 1:
        return tuple((ind[:, i] for i in range(ind.shape[1])))
    else:
        return (ind,)

def _unravel_index_kernel(indices, func_kwargs):
    if False:
        while True:
            i = 10
    return np.stack(np.unravel_index(indices, **func_kwargs))

@derived_from(np)
def unravel_index(indices, shape, order='C'):
    if False:
        while True:
            i = 10
    if shape and indices.size:
        unraveled_indices = tuple(indices.map_blocks(_unravel_index_kernel, dtype=np.intp, chunks=((len(shape),),) + indices.chunks, new_axis=0, func_kwargs={'shape': shape, 'order': order}))
    else:
        unraveled_indices = tuple((empty((0,), dtype=np.intp, chunks=1) for i in shape))
    return unraveled_indices

@wraps(np.ravel_multi_index)
def ravel_multi_index(multi_index, dims, mode='raise', order='C'):
    if False:
        i = 10
        return i + 15
    if np.isscalar(dims):
        dims = (dims,)
    if is_dask_collection(dims) or any((is_dask_collection(d) for d in dims)):
        raise NotImplementedError(f'Dask types are not supported in the `dims` argument: {dims!r}')
    if is_arraylike(multi_index):
        index_stack = asarray(multi_index)
    else:
        multi_index_arrs = broadcast_arrays(*multi_index)
        index_stack = stack(multi_index_arrs)
    if not np.isnan(index_stack.shape).any() and len(index_stack) != len(dims):
        raise ValueError(f'parameter multi_index must be a sequence of length {len(dims)}')
    if not np.issubdtype(index_stack.dtype, np.signedinteger):
        raise TypeError('only int indices permitted')
    return index_stack.map_blocks(np.ravel_multi_index, dtype=np.intp, chunks=index_stack.chunks[1:], drop_axis=0, dims=dims, mode=mode, order=order)

def _int_piecewise(x, *condlist, **kwargs):
    if False:
        while True:
            i = 10
    return np.piecewise(x, list(condlist), kwargs['funclist'], *kwargs['func_args'], **kwargs['func_kw'])

@derived_from(np)
def piecewise(x, condlist, funclist, *args, **kw):
    if False:
        print('Hello World!')
    return map_blocks(_int_piecewise, x, *condlist, dtype=x.dtype, name='piecewise', funclist=funclist, func_args=args, func_kw=kw)

def _select(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    This is a version of :func:`numpy.select` that acceptes an arbitrary number of arguments and\n    splits them in half to create ``condlist`` and ``choicelist`` params.\n    '
    split_at = len(args) // 2
    condlist = args[:split_at]
    choicelist = args[split_at:]
    return np.select(condlist, choicelist, **kwargs)

@derived_from(np)
def select(condlist, choicelist, default=0):
    if False:
        return 10
    if len(condlist) != len(choicelist):
        raise ValueError('list of cases must be same length as list of conditions')
    if len(condlist) == 0:
        raise ValueError('select with an empty condition list is not possible')
    choicelist = [asarray(choice) for choice in choicelist]
    try:
        intermediate_dtype = result_type(*choicelist)
    except TypeError as e:
        msg = 'Choicelist elements do not have a common dtype.'
        raise TypeError(msg) from e
    blockwise_shape = tuple(range(choicelist[0].ndim))
    condargs = [arg for elem in condlist for arg in (elem, blockwise_shape)]
    choiceargs = [arg for elem in choicelist for arg in (elem, blockwise_shape)]
    return blockwise(_select, blockwise_shape, *condargs, *choiceargs, dtype=intermediate_dtype, name='select', default=default)

def _partition(total: int, divisor: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if False:
        while True:
            i = 10
    'Given a total and a divisor, return two tuples: A tuple containing `divisor`\n    repeated the number of times it divides `total`, and length-1 or empty tuple\n    containing the remainder when `total` is divided by `divisor`. If `divisor` factors\n    `total`, i.e. if the remainder is 0, then `remainder` is empty.\n    '
    multiples = (divisor,) * (total // divisor)
    remainder = (total % divisor,) if total % divisor else ()
    return (multiples, remainder)

def aligned_coarsen_chunks(chunks: list[int], multiple: int) -> tuple[int, ...]:
    if False:
        while True:
            i = 10
    '\n    Returns a new chunking aligned with the coarsening multiple.\n    Any excess is at the end of the array.\n\n    Examples\n    --------\n    >>> aligned_coarsen_chunks(chunks=(1, 2, 3), multiple=4)\n    (4, 2)\n    >>> aligned_coarsen_chunks(chunks=(1, 20, 3, 4), multiple=4)\n    (4, 20, 4)\n    >>> aligned_coarsen_chunks(chunks=(20, 10, 15, 23, 24), multiple=10)\n    (20, 10, 20, 20, 20, 2)\n    '
    overflow = np.array(chunks) % multiple
    excess = overflow.sum()
    new_chunks = np.array(chunks) - overflow
    chunk_validity = new_chunks == chunks
    (valid_inds, invalid_inds) = (np.where(chunk_validity)[0], np.where(~chunk_validity)[0])
    chunk_modification_order = [*invalid_inds[np.argsort(new_chunks[invalid_inds])], *valid_inds[np.argsort(new_chunks[valid_inds])]]
    (partitioned_excess, remainder) = _partition(excess, multiple)
    for (idx, extra) in enumerate(partitioned_excess):
        new_chunks[chunk_modification_order[idx]] += extra
    new_chunks = np.array([*new_chunks, *remainder])
    new_chunks = new_chunks[new_chunks > 0]
    return tuple(new_chunks)

@wraps(chunk.coarsen)
def coarsen(reduction, x, axes, trim_excess=False, **kwargs):
    if False:
        i = 10
        return i + 15
    if not trim_excess and (not all((x.shape[i] % div == 0 for (i, div) in axes.items()))):
        msg = f'Coarsening factors {axes} do not align with array shape {x.shape}.'
        raise ValueError(msg)
    if reduction.__module__.startswith('dask.'):
        reduction = getattr(np, reduction.__name__)
    new_chunks = {}
    for (i, div) in axes.items():
        aligned = aligned_coarsen_chunks(x.chunks[i], div)
        if aligned != x.chunks[i]:
            new_chunks[i] = aligned
    if new_chunks:
        x = x.rechunk(new_chunks)
    name = 'coarsen-' + tokenize(reduction, x, axes, trim_excess)
    dsk = {(name,) + key[1:]: (apply, chunk.coarsen, [reduction, key, axes, trim_excess], kwargs) for key in flatten(x.__dask_keys__())}
    coarsen_dim = lambda dim, ax: int(dim // axes.get(ax, 1))
    chunks = tuple((tuple((coarsen_dim(bd, i) for bd in bds if coarsen_dim(bd, i) > 0)) for (i, bds) in enumerate(x.chunks)))
    meta = reduction(np.empty((1,) * x.ndim, dtype=x.dtype), **kwargs)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[x])
    return Array(graph, name, chunks, meta=meta)

def split_at_breaks(array, breaks, axis=0):
    if False:
        print('Hello World!')
    'Split an array into a list of arrays (using slices) at the given breaks\n\n    >>> split_at_breaks(np.arange(6), [3, 5])\n    [array([0, 1, 2]), array([3, 4]), array([5])]\n    '
    padded_breaks = concat([[None], breaks, [None]])
    slices = [slice(i, j) for (i, j) in sliding_window(2, padded_breaks)]
    preslice = (slice(None),) * axis
    split_array = [array[preslice + (s,)] for s in slices]
    return split_array

@derived_from(np)
def insert(arr, obj, values, axis):
    if False:
        print('Hello World!')
    axis = validate_axis(axis, arr.ndim)
    if isinstance(obj, slice):
        obj = np.arange(*obj.indices(arr.shape[axis]))
    obj = np.asarray(obj)
    scalar_obj = obj.ndim == 0
    if scalar_obj:
        obj = np.atleast_1d(obj)
    obj = np.where(obj < 0, obj + arr.shape[axis], obj)
    if (np.diff(obj) < 0).any():
        raise NotImplementedError('da.insert only implemented for monotonic ``obj`` argument')
    split_arr = split_at_breaks(arr, np.unique(obj), axis)
    if getattr(values, 'ndim', 0) == 0:
        name = 'values-' + tokenize(values)
        dtype = getattr(values, 'dtype', type(values))
        values = Array({(name,): values}, name, chunks=(), dtype=dtype)
        values_shape = tuple((len(obj) if axis == n else s for (n, s) in enumerate(arr.shape)))
        values = broadcast_to(values, values_shape)
    elif scalar_obj:
        values = values[(slice(None),) * axis + (None,)]
    values_chunks = tuple((values_bd if axis == n else arr_bd for (n, (arr_bd, values_bd)) in enumerate(zip(arr.chunks, values.chunks))))
    values = values.rechunk(values_chunks)
    counts = np.bincount(obj)[:-1]
    values_breaks = np.cumsum(counts[counts > 0])
    split_values = split_at_breaks(values, values_breaks, axis)
    interleaved = list(interleave([split_arr, split_values]))
    interleaved = [i for i in interleaved if i.nbytes]
    return concatenate(interleaved, axis=axis)

@derived_from(np)
def delete(arr, obj, axis):
    if False:
        i = 10
        return i + 15
    '\n    NOTE: If ``obj`` is a dask array it is implicitly computed when this function\n    is called.\n    '
    axis = validate_axis(axis, arr.ndim)
    if isinstance(obj, slice):
        tmp = np.arange(*obj.indices(arr.shape[axis]))
        obj = tmp[::-1] if obj.step and obj.step < 0 else tmp
    else:
        obj = np.asarray(obj)
        obj = np.where(obj < 0, obj + arr.shape[axis], obj)
        obj = np.unique(obj)
    target_arr = split_at_breaks(arr, obj, axis)
    target_arr = [arr[tuple((slice(1, None) if axis == n else slice(None) for n in range(arr.ndim)))] if i != 0 else arr for (i, arr) in enumerate(target_arr)]
    return concatenate(target_arr, axis=axis)

@derived_from(np)
def append(arr, values, axis=None):
    if False:
        print('Hello World!')
    arr = asanyarray(arr)
    if axis is None:
        if arr.ndim != 1:
            arr = arr.ravel()
        values = ravel(asanyarray(values))
        axis = arr.ndim - 1
    return concatenate((arr, values), axis=axis)

def _average(a, axis=None, weights=None, returned=False, is_masked=False, keepdims=False):
    if False:
        i = 10
        return i + 15
    a = asanyarray(a)
    if weights is None:
        avg = a.mean(axis, keepdims=keepdims)
        scl = avg.dtype.type(a.size / avg.size)
    else:
        wgt = asanyarray(weights)
        if issubclass(a.dtype.type, (np.integer, np.bool_)):
            result_dtype = result_type(a.dtype, wgt.dtype, 'f8')
        else:
            result_dtype = result_type(a.dtype, wgt.dtype)
        if a.shape != wgt.shape:
            if axis is None:
                raise TypeError('Axis must be specified when shapes of a and weights differ.')
            if wgt.ndim != 1:
                raise TypeError('1D weights expected when shapes of a and weights differ.')
            if wgt.shape[0] != a.shape[axis]:
                raise ValueError('Length of weights not compatible with specified axis.')
            wgt = broadcast_to(wgt, (a.ndim - 1) * (1,) + wgt.shape)
            wgt = wgt.swapaxes(-1, axis)
        if is_masked:
            from dask.array.ma import getmaskarray
            wgt = wgt * ~getmaskarray(a)
        scl = wgt.sum(axis=axis, dtype=result_dtype, keepdims=keepdims)
        avg = multiply(a, wgt, dtype=result_dtype).sum(axis, keepdims=keepdims) / scl
    if returned:
        if scl.shape != avg.shape:
            scl = broadcast_to(scl, avg.shape).copy()
        return (avg, scl)
    else:
        return avg

@derived_from(np)
def average(a, axis=None, weights=None, returned=False, keepdims=False):
    if False:
        return 10
    return _average(a, axis, weights, returned, is_masked=False, keepdims=keepdims)

@derived_from(np)
def tril(m, k=0):
    if False:
        i = 10
        return i + 15
    m = asarray_safe(m, like=m)
    mask = tri(*m.shape[-2:], k=k, dtype=bool, chunks=m.chunks[-2:], like=meta_from_array(m))
    return where(mask, m, np.zeros_like(m, shape=(1,)))

@derived_from(np)
def triu(m, k=0):
    if False:
        for i in range(10):
            print('nop')
    m = asarray_safe(m, like=m)
    mask = tri(*m.shape[-2:], k=k - 1, dtype=bool, chunks=m.chunks[-2:], like=meta_from_array(m))
    return where(mask, np.zeros_like(m, shape=(1,)), m)

@derived_from(np)
def tril_indices(n, k=0, m=None, chunks='auto'):
    if False:
        return 10
    return nonzero(tri(n, m, k=k, dtype=bool, chunks=chunks))

@derived_from(np)
def tril_indices_from(arr, k=0):
    if False:
        for i in range(10):
            print('nop')
    if arr.ndim != 2:
        raise ValueError('input array must be 2-d')
    return tril_indices(arr.shape[-2], k=k, m=arr.shape[-1], chunks=arr.chunks)

@derived_from(np)
def triu_indices(n, k=0, m=None, chunks='auto'):
    if False:
        i = 10
        return i + 15
    return nonzero(~tri(n, m, k=k - 1, dtype=bool, chunks=chunks))

@derived_from(np)
def triu_indices_from(arr, k=0):
    if False:
        print('Hello World!')
    if arr.ndim != 2:
        raise ValueError('input array must be 2-d')
    return triu_indices(arr.shape[-2], k=k, m=arr.shape[-1], chunks=arr.chunks)