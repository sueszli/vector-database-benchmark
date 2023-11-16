from __future__ import annotations
import warnings
from numbers import Integral, Number
import numpy as np
from tlz import concat, get, partial
from tlz.curried import map
from dask.array import chunk
from dask.array.core import Array, concatenate, map_blocks, unify_chunks
from dask.array.creation import empty_like, full_like
from dask.array.numpy_compat import normalize_axis_tuple
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.layers import ArrayOverlapLayer
from dask.utils import derived_from

def _overlap_internal_chunks(original_chunks, axes):
    if False:
        for i in range(10):
            print('nop')
    'Get new chunks for array with overlap.'
    chunks = []
    for (i, bds) in enumerate(original_chunks):
        depth = axes.get(i, 0)
        if isinstance(depth, tuple):
            left_depth = depth[0]
            right_depth = depth[1]
        else:
            left_depth = depth
            right_depth = depth
        if len(bds) == 1:
            chunks.append(bds)
        else:
            left = [bds[0] + right_depth]
            right = [bds[-1] + left_depth]
            mid = []
            for bd in bds[1:-1]:
                mid.append(bd + left_depth + right_depth)
            chunks.append(left + mid + right)
    return chunks

def overlap_internal(x, axes):
    if False:
        print('Hello World!')
    'Share boundaries between neighboring blocks\n\n    Parameters\n    ----------\n\n    x: da.Array\n        A dask array\n    axes: dict\n        The size of the shared boundary per axis\n\n    The axes input informs how many cells to overlap between neighboring blocks\n    {0: 2, 2: 5} means share two cells in 0 axis, 5 cells in 2 axis\n    '
    token = tokenize(x, axes)
    name = 'overlap-' + token
    graph = ArrayOverlapLayer(name=x.name, axes=axes, chunks=x.chunks, numblocks=x.numblocks, token=token)
    graph = HighLevelGraph.from_collections(name, graph, dependencies=[x])
    chunks = _overlap_internal_chunks(x.chunks, axes)
    return Array(graph, name, chunks, meta=x)

def trim_overlap(x, depth, boundary=None):
    if False:
        print('Hello World!')
    'Trim sides from each block.\n\n    This couples well with the ``map_overlap`` operation which may leave\n    excess data on each block.\n\n    See also\n    --------\n    dask.array.overlap.map_overlap\n\n    '
    axes = coerce_depth(x.ndim, depth)
    return trim_internal(x, axes=axes, boundary=boundary)

def trim_internal(x, axes, boundary=None):
    if False:
        while True:
            i = 10
    'Trim sides from each block\n\n    This couples well with the overlap operation, which may leave excess data on\n    each block\n\n    See also\n    --------\n    dask.array.chunk.trim\n    dask.array.map_blocks\n    '
    boundary = coerce_boundary(x.ndim, boundary)
    olist = []
    for (i, bd) in enumerate(x.chunks):
        bdy = boundary.get(i, 'none')
        overlap = axes.get(i, 0)
        ilist = []
        for (j, d) in enumerate(bd):
            if bdy != 'none':
                if isinstance(overlap, tuple):
                    d = d - sum(overlap)
                else:
                    d = d - overlap * 2
            elif isinstance(overlap, tuple):
                d = d - overlap[0] if j != 0 else d
                d = d - overlap[1] if j != len(bd) - 1 else d
            else:
                d = d - overlap if j != 0 else d
                d = d - overlap if j != len(bd) - 1 else d
            ilist.append(d)
        olist.append(tuple(ilist))
    chunks = tuple(olist)
    return map_blocks(partial(_trim, axes=axes, boundary=boundary), x, chunks=chunks, dtype=x.dtype, meta=x._meta)

def _trim(x, axes, boundary, block_info):
    if False:
        i = 10
        return i + 15
    'Similar to dask.array.chunk.trim but requires one to specificy the\n    boundary condition.\n\n    ``axes``, and ``boundary`` are assumed to have been coerced.\n\n    '
    axes = [axes.get(i, 0) for i in range(x.ndim)]
    axes_front = (ax[0] if isinstance(ax, tuple) else ax for ax in axes)
    axes_back = (-ax[1] if isinstance(ax, tuple) and ax[1] else -ax if isinstance(ax, Integral) and ax else None for ax in axes)
    trim_front = (0 if chunk_location == 0 and boundary.get(i, 'none') == 'none' else ax for (i, (chunk_location, ax)) in enumerate(zip(block_info[0]['chunk-location'], axes_front)))
    trim_back = (None if chunk_location == chunks - 1 and boundary.get(i, 'none') == 'none' else ax for (i, (chunks, chunk_location, ax)) in enumerate(zip(block_info[0]['num-chunks'], block_info[0]['chunk-location'], axes_back)))
    ind = tuple((slice(front, back) for (front, back) in zip(trim_front, trim_back)))
    return x[ind]

def periodic(x, axis, depth):
    if False:
        print('Hello World!')
    'Copy a slice of an array around to its other side\n\n    Useful to create periodic boundary conditions for overlap\n    '
    left = (slice(None, None, None),) * axis + (slice(0, depth),) + (slice(None, None, None),) * (x.ndim - axis - 1)
    right = (slice(None, None, None),) * axis + (slice(-depth, None),) + (slice(None, None, None),) * (x.ndim - axis - 1)
    l = x[left]
    r = x[right]
    (l, r) = _remove_overlap_boundaries(l, r, axis, depth)
    return concatenate([r, x, l], axis=axis)

def reflect(x, axis, depth):
    if False:
        for i in range(10):
            print('nop')
    'Reflect boundaries of array on the same side\n\n    This is the converse of ``periodic``\n    '
    if depth == 1:
        left = (slice(None, None, None),) * axis + (slice(0, 1),) + (slice(None, None, None),) * (x.ndim - axis - 1)
    else:
        left = (slice(None, None, None),) * axis + (slice(depth - 1, None, -1),) + (slice(None, None, None),) * (x.ndim - axis - 1)
    right = (slice(None, None, None),) * axis + (slice(-1, -depth - 1, -1),) + (slice(None, None, None),) * (x.ndim - axis - 1)
    l = x[left]
    r = x[right]
    (l, r) = _remove_overlap_boundaries(l, r, axis, depth)
    return concatenate([l, x, r], axis=axis)

def nearest(x, axis, depth):
    if False:
        print('Hello World!')
    'Each reflect each boundary value outwards\n\n    This mimics what the skimage.filters.gaussian_filter(... mode="nearest")\n    does.\n    '
    left = (slice(None, None, None),) * axis + (slice(0, 1),) + (slice(None, None, None),) * (x.ndim - axis - 1)
    right = (slice(None, None, None),) * axis + (slice(-1, -2, -1),) + (slice(None, None, None),) * (x.ndim - axis - 1)
    l = concatenate([x[left]] * depth, axis=axis)
    r = concatenate([x[right]] * depth, axis=axis)
    (l, r) = _remove_overlap_boundaries(l, r, axis, depth)
    return concatenate([l, x, r], axis=axis)

def constant(x, axis, depth, value):
    if False:
        i = 10
        return i + 15
    'Add constant slice to either side of array'
    chunks = list(x.chunks)
    chunks[axis] = (depth,)
    c = full_like(x, value, shape=tuple(map(sum, chunks)), chunks=tuple(chunks), dtype=x.dtype)
    return concatenate([c, x, c], axis=axis)

def _remove_overlap_boundaries(l, r, axis, depth):
    if False:
        i = 10
        return i + 15
    lchunks = list(l.chunks)
    lchunks[axis] = (depth,)
    rchunks = list(r.chunks)
    rchunks[axis] = (depth,)
    l = l.rechunk(tuple(lchunks))
    r = r.rechunk(tuple(rchunks))
    return (l, r)

def boundaries(x, depth=None, kind=None):
    if False:
        i = 10
        return i + 15
    'Add boundary conditions to an array before overlaping\n\n    See Also\n    --------\n    periodic\n    constant\n    '
    if not isinstance(kind, dict):
        kind = {i: kind for i in range(x.ndim)}
    if not isinstance(depth, dict):
        depth = {i: depth for i in range(x.ndim)}
    for i in range(x.ndim):
        d = depth.get(i, 0)
        if d == 0:
            continue
        this_kind = kind.get(i, 'none')
        if this_kind == 'none':
            continue
        elif this_kind == 'periodic':
            x = periodic(x, i, d)
        elif this_kind == 'reflect':
            x = reflect(x, i, d)
        elif this_kind == 'nearest':
            x = nearest(x, i, d)
        elif i in kind:
            x = constant(x, i, d, kind[i])
    return x

def ensure_minimum_chunksize(size, chunks):
    if False:
        i = 10
        return i + 15
    'Determine new chunks to ensure that every chunk >= size\n\n    Parameters\n    ----------\n    size: int\n        The maximum size of any chunk.\n    chunks: tuple\n        Chunks along one axis, e.g. ``(3, 3, 2)``\n\n    Examples\n    --------\n    >>> ensure_minimum_chunksize(10, (20, 20, 1))\n    (20, 11, 10)\n    >>> ensure_minimum_chunksize(3, (1, 1, 3))\n    (5,)\n\n    See Also\n    --------\n    overlap\n    '
    if size <= min(chunks):
        return chunks
    output = []
    new = 0
    for c in chunks:
        if c < size:
            if new > size + (size - c):
                output.append(new - (size - c))
                new = size
            else:
                new += c
        if new >= size:
            output.append(new)
            new = 0
        if c >= size:
            new += c
    if new >= size:
        output.append(new)
    elif len(output) >= 1:
        output[-1] += new
    else:
        raise ValueError(f'The overlapping depth {size} is larger than your array {sum(chunks)}.')
    return tuple(output)

def overlap(x, depth, boundary, *, allow_rechunk=True):
    if False:
        i = 10
        return i + 15
    "Share boundaries between neighboring blocks\n\n    Parameters\n    ----------\n\n    x: da.Array\n        A dask array\n    depth: dict\n        The size of the shared boundary per axis\n    boundary: dict\n        The boundary condition on each axis. Options are 'reflect', 'periodic',\n        'nearest', 'none', or an array value.  Such a value will fill the\n        boundary with that value.\n    allow_rechunk: bool, keyword only\n        Allows rechunking, otherwise chunk sizes need to match and core\n        dimensions are to consist only of one chunk.\n\n    The depth input informs how many cells to overlap between neighboring\n    blocks ``{0: 2, 2: 5}`` means share two cells in 0 axis, 5 cells in 2 axis.\n    Axes missing from this input will not be overlapped.\n\n    Any axis containing chunks smaller than depth will be rechunked if\n    possible, provided the keyword ``allow_rechunk`` is True (recommended).\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> import dask.array as da\n\n    >>> x = np.arange(64).reshape((8, 8))\n    >>> d = da.from_array(x, chunks=(4, 4))\n    >>> d.chunks\n    ((4, 4), (4, 4))\n\n    >>> g = da.overlap.overlap(d, depth={0: 2, 1: 1},\n    ...                       boundary={0: 100, 1: 'reflect'})\n    >>> g.chunks\n    ((8, 8), (6, 6))\n\n    >>> np.array(g)\n    array([[100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],\n           [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],\n           [  0,   0,   1,   2,   3,   4,   3,   4,   5,   6,   7,   7],\n           [  8,   8,   9,  10,  11,  12,  11,  12,  13,  14,  15,  15],\n           [ 16,  16,  17,  18,  19,  20,  19,  20,  21,  22,  23,  23],\n           [ 24,  24,  25,  26,  27,  28,  27,  28,  29,  30,  31,  31],\n           [ 32,  32,  33,  34,  35,  36,  35,  36,  37,  38,  39,  39],\n           [ 40,  40,  41,  42,  43,  44,  43,  44,  45,  46,  47,  47],\n           [ 16,  16,  17,  18,  19,  20,  19,  20,  21,  22,  23,  23],\n           [ 24,  24,  25,  26,  27,  28,  27,  28,  29,  30,  31,  31],\n           [ 32,  32,  33,  34,  35,  36,  35,  36,  37,  38,  39,  39],\n           [ 40,  40,  41,  42,  43,  44,  43,  44,  45,  46,  47,  47],\n           [ 48,  48,  49,  50,  51,  52,  51,  52,  53,  54,  55,  55],\n           [ 56,  56,  57,  58,  59,  60,  59,  60,  61,  62,  63,  63],\n           [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],\n           [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]])\n    "
    depth2 = coerce_depth(x.ndim, depth)
    boundary2 = coerce_boundary(x.ndim, boundary)
    depths = [max(d) if isinstance(d, tuple) else d for d in depth2.values()]
    if allow_rechunk:
        new_chunks = tuple((ensure_minimum_chunksize(size, c) for (size, c) in zip(depths, x.chunks)))
        x1 = x.rechunk(new_chunks)
    else:
        original_chunks_too_small = any([min(c) < d for (d, c) in zip(depths, x.chunks)])
        if original_chunks_too_small:
            raise ValueError(f'Overlap depth is larger than smallest chunksize.\nPlease set allow_rechunk=True to rechunk automatically.\nOverlap depths required: {depths}\nInput chunks: {x.chunks}\n')
        x1 = x
    x2 = boundaries(x1, depth2, boundary2)
    x3 = overlap_internal(x2, depth2)
    trim = {k: v * 2 if boundary2.get(k, 'none') != 'none' else 0 for (k, v) in depth2.items()}
    x4 = chunk.trim(x3, trim)
    return x4

def add_dummy_padding(x, depth, boundary):
    if False:
        print('Hello World!')
    "\n    Pads an array which has 'none' as the boundary type.\n    Used to simplify trimming arrays which use 'none'.\n\n    >>> import dask.array as da\n    >>> x = da.arange(6, chunks=3)\n    >>> add_dummy_padding(x, {0: 1}, {0: 'none'}).compute()  # doctest: +NORMALIZE_WHITESPACE\n    array([..., 0, 1, 2, 3, 4, 5, ...])\n    "
    for (k, v) in boundary.items():
        d = depth.get(k, 0)
        if v == 'none' and d > 0:
            empty_shape = list(x.shape)
            empty_shape[k] = d
            empty_chunks = list(x.chunks)
            empty_chunks[k] = (d,)
            empty = empty_like(getattr(x, '_meta', x), shape=empty_shape, chunks=empty_chunks, dtype=x.dtype)
            out_chunks = list(x.chunks)
            ax_chunks = list(out_chunks[k])
            ax_chunks[0] += d
            ax_chunks[-1] += d
            out_chunks[k] = tuple(ax_chunks)
            x = concatenate([empty, x, empty], axis=k)
            x = x.rechunk(out_chunks)
    return x

def map_overlap(func, *args, depth=None, boundary=None, trim=True, align_arrays=True, allow_rechunk=True, **kwargs):
    if False:
        i = 10
        return i + 15
    "Map a function over blocks of arrays with some overlap\n\n    We share neighboring zones between blocks of the array, map a\n    function, and then trim away the neighboring strips. If depth is\n    larger than any chunk along a particular axis, then the array is\n    rechunked.\n\n    Note that this function will attempt to automatically determine the output\n    array type before computing it, please refer to the ``meta`` keyword argument\n    in ``map_blocks`` if you expect that the function will not succeed when\n    operating on 0-d arrays.\n\n    Parameters\n    ----------\n    func: function\n        The function to apply to each extended block.\n        If multiple arrays are provided, then the function should expect to\n        receive chunks of each array in the same order.\n    args : dask arrays\n    depth: int, tuple, dict or list, keyword only\n        The number of elements that each block should share with its neighbors\n        If a tuple or dict then this can be different per axis.\n        If a list then each element of that list must be an int, tuple or dict\n        defining depth for the corresponding array in `args`.\n        Asymmetric depths may be specified using a dict value of (-/+) tuples.\n        Note that asymmetric depths are currently only supported when\n        ``boundary`` is 'none'.\n        The default value is 0.\n    boundary: str, tuple, dict or list, keyword only\n        How to handle the boundaries.\n        Values include 'reflect', 'periodic', 'nearest', 'none',\n        or any constant value like 0 or np.nan.\n        If a list then each element must be a str, tuple or dict defining the\n        boundary for the corresponding array in `args`.\n        The default value is 'reflect'.\n    trim: bool, keyword only\n        Whether or not to trim ``depth`` elements from each block after\n        calling the map function.\n        Set this to False if your mapping function already does this for you\n    align_arrays: bool, keyword only\n        Whether or not to align chunks along equally sized dimensions when\n        multiple arrays are provided.  This allows for larger chunks in some\n        arrays to be broken into smaller ones that match chunk sizes in other\n        arrays such that they are compatible for block function mapping. If\n        this is false, then an error will be thrown if arrays do not already\n        have the same number of blocks in each dimension.\n    allow_rechunk: bool, keyword only\n        Allows rechunking, otherwise chunk sizes need to match and core\n        dimensions are to consist only of one chunk.\n    **kwargs:\n        Other keyword arguments valid in ``map_blocks``\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> import dask.array as da\n\n    >>> x = np.array([1, 1, 2, 3, 3, 3, 2, 1, 1])\n    >>> x = da.from_array(x, chunks=5)\n    >>> def derivative(x):\n    ...     return x - np.roll(x, 1)\n\n    >>> y = x.map_overlap(derivative, depth=1, boundary=0)\n    >>> y.compute()\n    array([ 1,  0,  1,  1,  0,  0, -1, -1,  0])\n\n    >>> x = np.arange(16).reshape((4, 4))\n    >>> d = da.from_array(x, chunks=(2, 2))\n    >>> d.map_overlap(lambda x: x + x.size, depth=1, boundary='reflect').compute()\n    array([[16, 17, 18, 19],\n           [20, 21, 22, 23],\n           [24, 25, 26, 27],\n           [28, 29, 30, 31]])\n\n    >>> func = lambda x: x + x.size\n    >>> depth = {0: 1, 1: 1}\n    >>> boundary = {0: 'reflect', 1: 'none'}\n    >>> d.map_overlap(func, depth, boundary).compute()  # doctest: +NORMALIZE_WHITESPACE\n    array([[12,  13,  14,  15],\n           [16,  17,  18,  19],\n           [20,  21,  22,  23],\n           [24,  25,  26,  27]])\n\n    The ``da.map_overlap`` function can also accept multiple arrays.\n\n    >>> func = lambda x, y: x + y\n    >>> x = da.arange(8).reshape(2, 4).rechunk((1, 2))\n    >>> y = da.arange(4).rechunk(2)\n    >>> da.map_overlap(func, x, y, depth=1, boundary='reflect').compute() # doctest: +NORMALIZE_WHITESPACE\n    array([[ 0,  2,  4,  6],\n           [ 4,  6,  8,  10]])\n\n    When multiple arrays are given, they do not need to have the\n    same number of dimensions but they must broadcast together.\n    Arrays are aligned block by block (just as in ``da.map_blocks``)\n    so the blocks must have a common chunk size.  This common chunking\n    is determined automatically as long as ``align_arrays`` is True.\n\n    >>> x = da.arange(8, chunks=4)\n    >>> y = da.arange(8, chunks=2)\n    >>> r = da.map_overlap(func, x, y, depth=1, boundary='reflect', align_arrays=True)\n    >>> len(r.to_delayed())\n    4\n\n    >>> da.map_overlap(func, x, y, depth=1, boundary='reflect', align_arrays=False).compute()\n    Traceback (most recent call last):\n        ...\n    ValueError: Shapes do not align {'.0': {2, 4}}\n\n    Note also that this function is equivalent to ``map_blocks``\n    by default.  A non-zero ``depth`` must be defined for any\n    overlap to appear in the arrays provided to ``func``.\n\n    >>> func = lambda x: x.sum()\n    >>> x = da.ones(10, dtype='int')\n    >>> block_args = dict(chunks=(), drop_axis=0)\n    >>> da.map_blocks(func, x, **block_args).compute()\n    10\n    >>> da.map_overlap(func, x, **block_args, boundary='reflect').compute()\n    10\n    >>> da.map_overlap(func, x, **block_args, depth=1, boundary='reflect').compute()\n    12\n\n    For functions that may not handle 0-d arrays, it's also possible to specify\n    ``meta`` with an empty array matching the type of the expected result. In\n    the example below, ``func`` will result in an ``IndexError`` when computing\n    ``meta``:\n\n    >>> x = np.arange(16).reshape((4, 4))\n    >>> d = da.from_array(x, chunks=(2, 2))\n    >>> y = d.map_overlap(lambda x: x + x[2], depth=1, boundary='reflect', meta=np.array(()))\n    >>> y\n    dask.array<_trim, shape=(4, 4), dtype=float64, chunksize=(2, 2), chunktype=numpy.ndarray>\n    >>> y.compute()\n    array([[ 4,  6,  8, 10],\n           [ 8, 10, 12, 14],\n           [20, 22, 24, 26],\n           [24, 26, 28, 30]])\n\n    Similarly, it's possible to specify a non-NumPy array to ``meta``:\n\n    >>> import cupy  # doctest: +SKIP\n    >>> x = cupy.arange(16).reshape((4, 4))  # doctest: +SKIP\n    >>> d = da.from_array(x, chunks=(2, 2))  # doctest: +SKIP\n    >>> y = d.map_overlap(lambda x: x + x[2], depth=1, boundary='reflect', meta=cupy.array(()))  # doctest: +SKIP\n    >>> y  # doctest: +SKIP\n    dask.array<_trim, shape=(4, 4), dtype=float64, chunksize=(2, 2), chunktype=cupy.ndarray>\n    >>> y.compute()  # doctest: +SKIP\n    array([[ 4,  6,  8, 10],\n           [ 8, 10, 12, 14],\n           [20, 22, 24, 26],\n           [24, 26, 28, 30]])\n    "
    if isinstance(func, Array) and callable(args[0]):
        warnings.warn('The use of map_overlap(array, func, **kwargs) is deprecated since dask 2.17.0 and will be an error in a future release. To silence this warning, use the syntax map_overlap(func, array0,[ array1, ...,] **kwargs) instead.', FutureWarning)
        sig = ['func', 'depth', 'boundary', 'trim']
        depth = get(sig.index('depth'), args, depth)
        boundary = get(sig.index('boundary'), args, boundary)
        trim = get(sig.index('trim'), args, trim)
        (func, args) = (args[0], [func])
    if not callable(func):
        raise TypeError('First argument must be callable function, not {}\nUsage:   da.map_overlap(function, x)\n   or:   da.map_overlap(function, x, y, z)'.format(type(func).__name__))
    if not all((isinstance(x, Array) for x in args)):
        raise TypeError('All variadic arguments must be arrays, not {}\nUsage:   da.map_overlap(function, x)\n   or:   da.map_overlap(function, x, y, z)'.format([type(x).__name__ for x in args]))

    def coerce(xs, arg, fn):
        if False:
            i = 10
            return i + 15
        if not isinstance(arg, list):
            arg = [arg] * len(xs)
        return [fn(x.ndim, a) for (x, a) in zip(xs, arg)]
    depth = coerce(args, depth, coerce_depth)
    boundary = coerce(args, boundary, coerce_boundary)
    if align_arrays:
        inds = [list(reversed(range(x.ndim))) for x in args]
        (_, args) = unify_chunks(*list(concat(zip(args, inds))), warn=False)
    if all([all((depth_val == 0 for depth_val in d.values())) for d in depth]):
        return map_blocks(func, *args, **kwargs)
    for (i, x) in enumerate(args):
        for j in range(x.ndim):
            if isinstance(depth[i][j], tuple) and boundary[i][j] != 'none':
                raise NotImplementedError("Asymmetric overlap is currently only implemented for boundary='none', however boundary for dimension {} in array argument {} is {}".format(j, i, boundary[i][j]))

    def assert_int_chunksize(xs):
        if False:
            i = 10
            return i + 15
        assert all((type(c) is int for x in xs for cc in x.chunks for c in cc))
    assert_int_chunksize(args)
    if not trim and 'chunks' not in kwargs:
        kwargs['chunks'] = args[0].chunks
    args = [overlap(x, depth=d, boundary=b, allow_rechunk=allow_rechunk) for (x, d, b) in zip(args, depth, boundary)]
    assert_int_chunksize(args)
    x = map_blocks(func, *args, **kwargs)
    assert_int_chunksize([x])
    if trim:
        i = sorted(enumerate(args), key=lambda v: (v[1].ndim, -v[0]))[-1][0]
        depth = depth[i]
        boundary = boundary[i]
        drop_axis = kwargs.pop('drop_axis', None)
        if drop_axis is not None:
            if isinstance(drop_axis, Number):
                drop_axis = [drop_axis]
            ndim_out = max((a.ndim for a in args if isinstance(a, Array)))
            drop_axis = [d % ndim_out for d in drop_axis]
            kept_axes = tuple((ax for ax in range(args[i].ndim) if ax not in drop_axis))
            depth = {n: depth[ax] for (n, ax) in enumerate(kept_axes)}
            boundary = {n: boundary[ax] for (n, ax) in enumerate(kept_axes)}
        return trim_internal(x, depth, boundary)
    else:
        return x

def coerce_depth(ndim, depth):
    if False:
        i = 10
        return i + 15
    default = 0
    if depth is None:
        depth = default
    if isinstance(depth, Integral):
        depth = (depth,) * ndim
    if isinstance(depth, tuple):
        depth = dict(zip(range(ndim), depth))
    if isinstance(depth, dict):
        depth = {ax: depth.get(ax, default) for ax in range(ndim)}
    return coerce_depth_type(ndim, depth)

def coerce_depth_type(ndim, depth):
    if False:
        while True:
            i = 10
    for i in range(ndim):
        if isinstance(depth[i], tuple):
            depth[i] = tuple((int(d) for d in depth[i]))
        else:
            depth[i] = int(depth[i])
    return depth

def coerce_boundary(ndim, boundary):
    if False:
        i = 10
        return i + 15
    default = 'none'
    if boundary is None:
        boundary = default
    if not isinstance(boundary, (tuple, dict)):
        boundary = (boundary,) * ndim
    if isinstance(boundary, tuple):
        boundary = dict(zip(range(ndim), boundary))
    if isinstance(boundary, dict):
        boundary = {ax: boundary.get(ax, default) for ax in range(ndim)}
    return boundary

@derived_from(np.lib.stride_tricks)
def sliding_window_view(x, window_shape, axis=None):
    if False:
        return 10
    window_shape = tuple(window_shape) if np.iterable(window_shape) else (window_shape,)
    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array <= 0):
        raise ValueError('`window_shape` must contain values > 0')
    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(f'Since axis is `None`, must provide window_shape for all dimensions of `x`; got {len(window_shape)} window_shape elements and `x.ndim` is {x.ndim}.')
    else:
        axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
        if len(window_shape) != len(axis):
            raise ValueError(f'Must provide matching length window_shape and axis; got {len(window_shape)} window_shape elements and {len(axis)} axes elements.')
    depths = [0] * x.ndim
    for (ax, window) in zip(axis, window_shape):
        depths[ax] += window - 1
    safe_chunks = tuple((ensure_minimum_chunksize(d + 1, c) for (d, c) in zip(depths, x.chunks)))
    x = x.rechunk(safe_chunks)
    newchunks = tuple((c[:-1] + (c[-1] - d,) for (d, c) in zip(depths, x.chunks))) + tuple(((window,) for window in window_shape))
    return map_overlap(np.lib.stride_tricks.sliding_window_view, x, depth=tuple(((0, d) for d in depths)), boundary='none', meta=x._meta, new_axis=range(x.ndim, x.ndim + len(axis)), chunks=newchunks, trim=False, align_arrays=False, window_shape=window_shape, axis=axis)