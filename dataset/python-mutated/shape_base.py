__all__ = ['atleast_1d', 'atleast_2d', 'atleast_3d', 'block', 'hstack', 'stack', 'vstack']
import functools
import itertools
import operator
import warnings
from . import numeric as _nx
from . import overrides
from .multiarray import array, asanyarray, normalize_axis_index
from . import fromnumeric as _from_nx
array_function_dispatch = functools.partial(overrides.array_function_dispatch, module='numpy')

def _atleast_1d_dispatcher(*arys):
    if False:
        print('Hello World!')
    return arys

@array_function_dispatch(_atleast_1d_dispatcher)
def atleast_1d(*arys):
    if False:
        i = 10
        return i + 15
    '\n    Convert inputs to arrays with at least one dimension.\n\n    Scalar inputs are converted to 1-dimensional arrays, whilst\n    higher-dimensional inputs are preserved.\n\n    Parameters\n    ----------\n    arys1, arys2, ... : array_like\n        One or more input arrays.\n\n    Returns\n    -------\n    ret : ndarray\n        An array, or list of arrays, each with ``a.ndim >= 1``.\n        Copies are made only if necessary.\n\n    See Also\n    --------\n    atleast_2d, atleast_3d\n\n    Examples\n    --------\n    >>> np.atleast_1d(1.0)\n    array([1.])\n\n    >>> x = np.arange(9.0).reshape(3,3)\n    >>> np.atleast_1d(x)\n    array([[0., 1., 2.],\n           [3., 4., 5.],\n           [6., 7., 8.]])\n    >>> np.atleast_1d(x) is x\n    True\n\n    >>> np.atleast_1d(1, [3, 4])\n    [array([1]), array([3, 4])]\n\n    '
    res = []
    for ary in arys:
        ary = asanyarray(ary)
        if ary.ndim == 0:
            result = ary.reshape(1)
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res

def _atleast_2d_dispatcher(*arys):
    if False:
        return 10
    return arys

@array_function_dispatch(_atleast_2d_dispatcher)
def atleast_2d(*arys):
    if False:
        return 10
    '\n    View inputs as arrays with at least two dimensions.\n\n    Parameters\n    ----------\n    arys1, arys2, ... : array_like\n        One or more array-like sequences.  Non-array inputs are converted\n        to arrays.  Arrays that already have two or more dimensions are\n        preserved.\n\n    Returns\n    -------\n    res, res2, ... : ndarray\n        An array, or list of arrays, each with ``a.ndim >= 2``.\n        Copies are avoided where possible, and views with two or more\n        dimensions are returned.\n\n    See Also\n    --------\n    atleast_1d, atleast_3d\n\n    Examples\n    --------\n    >>> np.atleast_2d(3.0)\n    array([[3.]])\n\n    >>> x = np.arange(3.0)\n    >>> np.atleast_2d(x)\n    array([[0., 1., 2.]])\n    >>> np.atleast_2d(x).base is x\n    True\n\n    >>> np.atleast_2d(1, [1, 2], [[1, 2]])\n    [array([[1]]), array([[1, 2]]), array([[1, 2]])]\n\n    '
    res = []
    for ary in arys:
        ary = asanyarray(ary)
        if ary.ndim == 0:
            result = ary.reshape(1, 1)
        elif ary.ndim == 1:
            result = ary[_nx.newaxis, :]
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res

def _atleast_3d_dispatcher(*arys):
    if False:
        print('Hello World!')
    return arys

@array_function_dispatch(_atleast_3d_dispatcher)
def atleast_3d(*arys):
    if False:
        for i in range(10):
            print('nop')
    '\n    View inputs as arrays with at least three dimensions.\n\n    Parameters\n    ----------\n    arys1, arys2, ... : array_like\n        One or more array-like sequences.  Non-array inputs are converted to\n        arrays.  Arrays that already have three or more dimensions are\n        preserved.\n\n    Returns\n    -------\n    res1, res2, ... : ndarray\n        An array, or list of arrays, each with ``a.ndim >= 3``.  Copies are\n        avoided where possible, and views with three or more dimensions are\n        returned.  For example, a 1-D array of shape ``(N,)`` becomes a view\n        of shape ``(1, N, 1)``, and a 2-D array of shape ``(M, N)`` becomes a\n        view of shape ``(M, N, 1)``.\n\n    See Also\n    --------\n    atleast_1d, atleast_2d\n\n    Examples\n    --------\n    >>> np.atleast_3d(3.0)\n    array([[[3.]]])\n\n    >>> x = np.arange(3.0)\n    >>> np.atleast_3d(x).shape\n    (1, 3, 1)\n\n    >>> x = np.arange(12.0).reshape(4,3)\n    >>> np.atleast_3d(x).shape\n    (4, 3, 1)\n    >>> np.atleast_3d(x).base is x.base  # x is a reshape, so not base itself\n    True\n\n    >>> for arr in np.atleast_3d([1, 2], [[1, 2]], [[[1, 2]]]):\n    ...     print(arr, arr.shape) # doctest: +SKIP\n    ...\n    [[[1]\n      [2]]] (1, 2, 1)\n    [[[1]\n      [2]]] (1, 2, 1)\n    [[[1 2]]] (1, 1, 2)\n\n    '
    res = []
    for ary in arys:
        ary = asanyarray(ary)
        if ary.ndim == 0:
            result = ary.reshape(1, 1, 1)
        elif ary.ndim == 1:
            result = ary[_nx.newaxis, :, _nx.newaxis]
        elif ary.ndim == 2:
            result = ary[:, :, _nx.newaxis]
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res

def _arrays_for_stack_dispatcher(arrays):
    if False:
        while True:
            i = 10
    if not hasattr(arrays, '__getitem__'):
        raise TypeError('arrays to stack must be passed as a "sequence" type such as list or tuple.')
    return tuple(arrays)

def _vhstack_dispatcher(tup, *, dtype=None, casting=None):
    if False:
        print('Hello World!')
    return _arrays_for_stack_dispatcher(tup)

@array_function_dispatch(_vhstack_dispatcher)
def vstack(tup, *, dtype=None, casting='same_kind'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Stack arrays in sequence vertically (row wise).\n\n    This is equivalent to concatenation along the first axis after 1-D arrays\n    of shape `(N,)` have been reshaped to `(1,N)`. Rebuilds arrays divided by\n    `vsplit`.\n\n    This function makes most sense for arrays with up to 3 dimensions. For\n    instance, for pixel-data with a height (first axis), width (second axis),\n    and r/g/b channels (third axis). The functions `concatenate`, `stack` and\n    `block` provide more general stacking and concatenation operations.\n\n    Parameters\n    ----------\n    tup : sequence of ndarrays\n        The arrays must have the same shape along all but the first axis.\n        1-D arrays must have the same length.\n\n    dtype : str or dtype\n        If provided, the destination array will have this dtype. Cannot be\n        provided together with `out`.\n\n        .. versionadded:: 1.24\n\n    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional\n        Controls what kind of data casting may occur. Defaults to 'same_kind'.\n\n        .. versionadded:: 1.24\n\n    Returns\n    -------\n    stacked : ndarray\n        The array formed by stacking the given arrays, will be at least 2-D.\n\n    See Also\n    --------\n    concatenate : Join a sequence of arrays along an existing axis.\n    stack : Join a sequence of arrays along a new axis.\n    block : Assemble an nd-array from nested lists of blocks.\n    hstack : Stack arrays in sequence horizontally (column wise).\n    dstack : Stack arrays in sequence depth wise (along third axis).\n    column_stack : Stack 1-D arrays as columns into a 2-D array.\n    vsplit : Split an array into multiple sub-arrays vertically (row-wise).\n\n    Examples\n    --------\n    >>> a = np.array([1, 2, 3])\n    >>> b = np.array([4, 5, 6])\n    >>> np.vstack((a,b))\n    array([[1, 2, 3],\n           [4, 5, 6]])\n\n    >>> a = np.array([[1], [2], [3]])\n    >>> b = np.array([[4], [5], [6]])\n    >>> np.vstack((a,b))\n    array([[1],\n           [2],\n           [3],\n           [4],\n           [5],\n           [6]])\n\n    "
    arrs = atleast_2d(*tup)
    if not isinstance(arrs, list):
        arrs = [arrs]
    return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)

@array_function_dispatch(_vhstack_dispatcher)
def hstack(tup, *, dtype=None, casting='same_kind'):
    if False:
        i = 10
        return i + 15
    "\n    Stack arrays in sequence horizontally (column wise).\n\n    This is equivalent to concatenation along the second axis, except for 1-D\n    arrays where it concatenates along the first axis. Rebuilds arrays divided\n    by `hsplit`.\n\n    This function makes most sense for arrays with up to 3 dimensions. For\n    instance, for pixel-data with a height (first axis), width (second axis),\n    and r/g/b channels (third axis). The functions `concatenate`, `stack` and\n    `block` provide more general stacking and concatenation operations.\n\n    Parameters\n    ----------\n    tup : sequence of ndarrays\n        The arrays must have the same shape along all but the second axis,\n        except 1-D arrays which can be any length.\n\n    dtype : str or dtype\n        If provided, the destination array will have this dtype. Cannot be\n        provided together with `out`.\n\n        .. versionadded:: 1.24\n\n    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional\n        Controls what kind of data casting may occur. Defaults to 'same_kind'.\n\n        .. versionadded:: 1.24\n\n    Returns\n    -------\n    stacked : ndarray\n        The array formed by stacking the given arrays.\n\n    See Also\n    --------\n    concatenate : Join a sequence of arrays along an existing axis.\n    stack : Join a sequence of arrays along a new axis.\n    block : Assemble an nd-array from nested lists of blocks.\n    vstack : Stack arrays in sequence vertically (row wise).\n    dstack : Stack arrays in sequence depth wise (along third axis).\n    column_stack : Stack 1-D arrays as columns into a 2-D array.\n    hsplit : Split an array into multiple sub-arrays \n             horizontally (column-wise).\n\n    Examples\n    --------\n    >>> a = np.array((1,2,3))\n    >>> b = np.array((4,5,6))\n    >>> np.hstack((a,b))\n    array([1, 2, 3, 4, 5, 6])\n    >>> a = np.array([[1],[2],[3]])\n    >>> b = np.array([[4],[5],[6]])\n    >>> np.hstack((a,b))\n    array([[1, 4],\n           [2, 5],\n           [3, 6]])\n\n    "
    arrs = atleast_1d(*tup)
    if not isinstance(arrs, list):
        arrs = [arrs]
    if arrs and arrs[0].ndim == 1:
        return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)
    else:
        return _nx.concatenate(arrs, 1, dtype=dtype, casting=casting)

def _stack_dispatcher(arrays, axis=None, out=None, *, dtype=None, casting=None):
    if False:
        return 10
    arrays = _arrays_for_stack_dispatcher(arrays)
    if out is not None:
        arrays = list(arrays)
        arrays.append(out)
    return arrays

@array_function_dispatch(_stack_dispatcher)
def stack(arrays, axis=0, out=None, *, dtype=None, casting='same_kind'):
    if False:
        while True:
            i = 10
    "\n    Join a sequence of arrays along a new axis.\n\n    The ``axis`` parameter specifies the index of the new axis in the\n    dimensions of the result. For example, if ``axis=0`` it will be the first\n    dimension and if ``axis=-1`` it will be the last dimension.\n\n    .. versionadded:: 1.10.0\n\n    Parameters\n    ----------\n    arrays : sequence of array_like\n        Each array must have the same shape.\n\n    axis : int, optional\n        The axis in the result array along which the input arrays are stacked.\n\n    out : ndarray, optional\n        If provided, the destination to place the result. The shape must be\n        correct, matching that of what stack would have returned if no\n        out argument were specified.\n\n    dtype : str or dtype\n        If provided, the destination array will have this dtype. Cannot be\n        provided together with `out`.\n\n        .. versionadded:: 1.24\n\n    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional\n        Controls what kind of data casting may occur. Defaults to 'same_kind'.\n\n        .. versionadded:: 1.24\n\n\n    Returns\n    -------\n    stacked : ndarray\n        The stacked array has one more dimension than the input arrays.\n\n    See Also\n    --------\n    concatenate : Join a sequence of arrays along an existing axis.\n    block : Assemble an nd-array from nested lists of blocks.\n    split : Split array into a list of multiple sub-arrays of equal size.\n\n    Examples\n    --------\n    >>> arrays = [np.random.randn(3, 4) for _ in range(10)]\n    >>> np.stack(arrays, axis=0).shape\n    (10, 3, 4)\n\n    >>> np.stack(arrays, axis=1).shape\n    (3, 10, 4)\n\n    >>> np.stack(arrays, axis=2).shape\n    (3, 4, 10)\n\n    >>> a = np.array([1, 2, 3])\n    >>> b = np.array([4, 5, 6])\n    >>> np.stack((a, b))\n    array([[1, 2, 3],\n           [4, 5, 6]])\n\n    >>> np.stack((a, b), axis=-1)\n    array([[1, 4],\n           [2, 5],\n           [3, 6]])\n\n    "
    arrays = [asanyarray(arr) for arr in arrays]
    if not arrays:
        raise ValueError('need at least one array to stack')
    shapes = {arr.shape for arr in arrays}
    if len(shapes) != 1:
        raise ValueError('all input arrays must have the same shape')
    result_ndim = arrays[0].ndim + 1
    axis = normalize_axis_index(axis, result_ndim)
    sl = (slice(None),) * axis + (_nx.newaxis,)
    expanded_arrays = [arr[sl] for arr in arrays]
    return _nx.concatenate(expanded_arrays, axis=axis, out=out, dtype=dtype, casting=casting)
_size = getattr(_from_nx.size, '__wrapped__', _from_nx.size)
_ndim = getattr(_from_nx.ndim, '__wrapped__', _from_nx.ndim)
_concatenate = getattr(_from_nx.concatenate, '__wrapped__', _from_nx.concatenate)

def _block_format_index(index):
    if False:
        print('Hello World!')
    '\n    Convert a list of indices ``[0, 1, 2]`` into ``"arrays[0][1][2]"``.\n    '
    idx_str = ''.join(('[{}]'.format(i) for i in index if i is not None))
    return 'arrays' + idx_str

def _block_check_depths_match(arrays, parent_index=[]):
    if False:
        i = 10
        return i + 15
    '\n    Recursive function checking that the depths of nested lists in `arrays`\n    all match. Mismatch raises a ValueError as described in the block\n    docstring below.\n\n    The entire index (rather than just the depth) needs to be calculated\n    for each innermost list, in case an error needs to be raised, so that\n    the index of the offending list can be printed as part of the error.\n\n    Parameters\n    ----------\n    arrays : nested list of arrays\n        The arrays to check\n    parent_index : list of int\n        The full index of `arrays` within the nested lists passed to\n        `_block_check_depths_match` at the top of the recursion.\n\n    Returns\n    -------\n    first_index : list of int\n        The full index of an element from the bottom of the nesting in\n        `arrays`. If any element at the bottom is an empty list, this will\n        refer to it, and the last index along the empty axis will be None.\n    max_arr_ndim : int\n        The maximum of the ndims of the arrays nested in `arrays`.\n    final_size: int\n        The number of elements in the final array. This is used the motivate\n        the choice of algorithm used using benchmarking wisdom.\n\n    '
    if type(arrays) is tuple:
        raise TypeError('{} is a tuple. Only lists can be used to arrange blocks, and np.block does not allow implicit conversion from tuple to ndarray.'.format(_block_format_index(parent_index)))
    elif type(arrays) is list and len(arrays) > 0:
        idxs_ndims = (_block_check_depths_match(arr, parent_index + [i]) for (i, arr) in enumerate(arrays))
        (first_index, max_arr_ndim, final_size) = next(idxs_ndims)
        for (index, ndim, size) in idxs_ndims:
            final_size += size
            if ndim > max_arr_ndim:
                max_arr_ndim = ndim
            if len(index) != len(first_index):
                raise ValueError('List depths are mismatched. First element was at depth {}, but there is an element at depth {} ({})'.format(len(first_index), len(index), _block_format_index(index)))
            if index[-1] is None:
                first_index = index
        return (first_index, max_arr_ndim, final_size)
    elif type(arrays) is list and len(arrays) == 0:
        return (parent_index + [None], 0, 0)
    else:
        size = _size(arrays)
        return (parent_index, _ndim(arrays), size)

def _atleast_nd(a, ndim):
    if False:
        for i in range(10):
            print('nop')
    return array(a, ndmin=ndim, copy=False, subok=True)

def _accumulate(values):
    if False:
        while True:
            i = 10
    return list(itertools.accumulate(values))

def _concatenate_shapes(shapes, axis):
    if False:
        for i in range(10):
            print('nop')
    'Given array shapes, return the resulting shape and slices prefixes.\n\n    These help in nested concatenation.\n\n    Returns\n    -------\n    shape: tuple of int\n        This tuple satisfies::\n\n            shape, _ = _concatenate_shapes([arr.shape for shape in arrs], axis)\n            shape == concatenate(arrs, axis).shape\n\n    slice_prefixes: tuple of (slice(start, end), )\n        For a list of arrays being concatenated, this returns the slice\n        in the larger array at axis that needs to be sliced into.\n\n        For example, the following holds::\n\n            ret = concatenate([a, b, c], axis)\n            _, (sl_a, sl_b, sl_c) = concatenate_slices([a, b, c], axis)\n\n            ret[(slice(None),) * axis + sl_a] == a\n            ret[(slice(None),) * axis + sl_b] == b\n            ret[(slice(None),) * axis + sl_c] == c\n\n        These are called slice prefixes since they are used in the recursive\n        blocking algorithm to compute the left-most slices during the\n        recursion. Therefore, they must be prepended to rest of the slice\n        that was computed deeper in the recursion.\n\n        These are returned as tuples to ensure that they can quickly be added\n        to existing slice tuple without creating a new tuple every time.\n\n    '
    shape_at_axis = [shape[axis] for shape in shapes]
    first_shape = shapes[0]
    first_shape_pre = first_shape[:axis]
    first_shape_post = first_shape[axis + 1:]
    if any((shape[:axis] != first_shape_pre or shape[axis + 1:] != first_shape_post for shape in shapes)):
        raise ValueError('Mismatched array shapes in block along axis {}.'.format(axis))
    shape = first_shape_pre + (sum(shape_at_axis),) + first_shape[axis + 1:]
    offsets_at_axis = _accumulate(shape_at_axis)
    slice_prefixes = [(slice(start, end),) for (start, end) in zip([0] + offsets_at_axis, offsets_at_axis)]
    return (shape, slice_prefixes)

def _block_info_recursion(arrays, max_depth, result_ndim, depth=0):
    if False:
        print('Hello World!')
    '\n    Returns the shape of the final array, along with a list\n    of slices and a list of arrays that can be used for assignment inside the\n    new array\n\n    Parameters\n    ----------\n    arrays : nested list of arrays\n        The arrays to check\n    max_depth : list of int\n        The number of nested lists\n    result_ndim : int\n        The number of dimensions in thefinal array.\n\n    Returns\n    -------\n    shape : tuple of int\n        The shape that the final array will take on.\n    slices: list of tuple of slices\n        The slices into the full array required for assignment. These are\n        required to be prepended with ``(Ellipsis, )`` to obtain to correct\n        final index.\n    arrays: list of ndarray\n        The data to assign to each slice of the full array\n\n    '
    if depth < max_depth:
        (shapes, slices, arrays) = zip(*[_block_info_recursion(arr, max_depth, result_ndim, depth + 1) for arr in arrays])
        axis = result_ndim - max_depth + depth
        (shape, slice_prefixes) = _concatenate_shapes(shapes, axis)
        slices = [slice_prefix + the_slice for (slice_prefix, inner_slices) in zip(slice_prefixes, slices) for the_slice in inner_slices]
        arrays = functools.reduce(operator.add, arrays)
        return (shape, slices, arrays)
    else:
        arr = _atleast_nd(arrays, result_ndim)
        return (arr.shape, [()], [arr])

def _block(arrays, max_depth, result_ndim, depth=0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Internal implementation of block based on repeated concatenation.\n    `arrays` is the argument passed to\n    block. `max_depth` is the depth of nested lists within `arrays` and\n    `result_ndim` is the greatest of the dimensions of the arrays in\n    `arrays` and the depth of the lists in `arrays` (see block docstring\n    for details).\n    '
    if depth < max_depth:
        arrs = [_block(arr, max_depth, result_ndim, depth + 1) for arr in arrays]
        return _concatenate(arrs, axis=-(max_depth - depth))
    else:
        return _atleast_nd(arrays, result_ndim)

def _block_dispatcher(arrays):
    if False:
        return 10
    if type(arrays) is list:
        for subarrays in arrays:
            yield from _block_dispatcher(subarrays)
    else:
        yield arrays

@array_function_dispatch(_block_dispatcher)
def block(arrays):
    if False:
        return 10
    '\n    Assemble an nd-array from nested lists of blocks.\n\n    Blocks in the innermost lists are concatenated (see `concatenate`) along\n    the last dimension (-1), then these are concatenated along the\n    second-last dimension (-2), and so on until the outermost list is reached.\n\n    Blocks can be of any dimension, but will not be broadcasted using\n    the normal rules. Instead, leading axes of size 1 are inserted, \n    to make ``block.ndim`` the same for all blocks. This is primarily useful\n    for working with scalars, and means that code like ``np.block([v, 1])``\n    is valid, where ``v.ndim == 1``.\n\n    When the nested list is two levels deep, this allows block matrices to be\n    constructed from their components.\n\n    .. versionadded:: 1.13.0\n\n    Parameters\n    ----------\n    arrays : nested list of array_like or scalars (but not tuples)\n        If passed a single ndarray or scalar (a nested list of depth 0), this\n        is returned unmodified (and not copied).\n\n        Elements shapes must match along the appropriate axes (without\n        broadcasting), but leading 1s will be prepended to the shape as\n        necessary to make the dimensions match.\n\n    Returns\n    -------\n    block_array : ndarray\n        The array assembled from the given blocks.\n\n        The dimensionality of the output is equal to the greatest of:\n\n        * the dimensionality of all the inputs\n        * the depth to which the input list is nested\n\n    Raises\n    ------\n    ValueError\n        * If list depths are mismatched - for instance, ``[[a, b], c]`` is\n          illegal, and should be spelt ``[[a, b], [c]]``\n        * If lists are empty - for instance, ``[[a, b], []]``\n\n    See Also\n    --------\n    concatenate : Join a sequence of arrays along an existing axis.\n    stack : Join a sequence of arrays along a new axis.\n    vstack : Stack arrays in sequence vertically (row wise).\n    hstack : Stack arrays in sequence horizontally (column wise).\n    dstack : Stack arrays in sequence depth wise (along third axis).\n    column_stack : Stack 1-D arrays as columns into a 2-D array.\n    vsplit : Split an array into multiple sub-arrays vertically (row-wise).\n\n    Notes\n    -----\n\n    When called with only scalars, ``np.block`` is equivalent to an ndarray\n    call. So ``np.block([[1, 2], [3, 4]])`` is equivalent to\n    ``np.array([[1, 2], [3, 4]])``.\n\n    This function does not enforce that the blocks lie on a fixed grid.\n    ``np.block([[a, b], [c, d]])`` is not restricted to arrays of the form::\n\n        AAAbb\n        AAAbb\n        cccDD\n\n    But is also allowed to produce, for some ``a, b, c, d``::\n\n        AAAbb\n        AAAbb\n        cDDDD\n\n    Since concatenation happens along the last axis first, `block` is *not*\n    capable of producing the following directly::\n\n        AAAbb\n        cccbb\n        cccDD\n\n    Matlab\'s "square bracket stacking", ``[A, B, ...; p, q, ...]``, is\n    equivalent to ``np.block([[A, B, ...], [p, q, ...]])``.\n\n    Examples\n    --------\n    The most common use of this function is to build a block matrix\n\n    >>> A = np.eye(2) * 2\n    >>> B = np.eye(3) * 3\n    >>> np.block([\n    ...     [A,               np.zeros((2, 3))],\n    ...     [np.ones((3, 2)), B               ]\n    ... ])\n    array([[2., 0., 0., 0., 0.],\n           [0., 2., 0., 0., 0.],\n           [1., 1., 3., 0., 0.],\n           [1., 1., 0., 3., 0.],\n           [1., 1., 0., 0., 3.]])\n\n    With a list of depth 1, `block` can be used as `hstack`\n\n    >>> np.block([1, 2, 3])              # hstack([1, 2, 3])\n    array([1, 2, 3])\n\n    >>> a = np.array([1, 2, 3])\n    >>> b = np.array([4, 5, 6])\n    >>> np.block([a, b, 10])             # hstack([a, b, 10])\n    array([ 1,  2,  3,  4,  5,  6, 10])\n\n    >>> A = np.ones((2, 2), int)\n    >>> B = 2 * A\n    >>> np.block([A, B])                 # hstack([A, B])\n    array([[1, 1, 2, 2],\n           [1, 1, 2, 2]])\n\n    With a list of depth 2, `block` can be used in place of `vstack`:\n\n    >>> a = np.array([1, 2, 3])\n    >>> b = np.array([4, 5, 6])\n    >>> np.block([[a], [b]])             # vstack([a, b])\n    array([[1, 2, 3],\n           [4, 5, 6]])\n\n    >>> A = np.ones((2, 2), int)\n    >>> B = 2 * A\n    >>> np.block([[A], [B]])             # vstack([A, B])\n    array([[1, 1],\n           [1, 1],\n           [2, 2],\n           [2, 2]])\n\n    It can also be used in places of `atleast_1d` and `atleast_2d`\n\n    >>> a = np.array(0)\n    >>> b = np.array([1])\n    >>> np.block([a])                    # atleast_1d(a)\n    array([0])\n    >>> np.block([b])                    # atleast_1d(b)\n    array([1])\n\n    >>> np.block([[a]])                  # atleast_2d(a)\n    array([[0]])\n    >>> np.block([[b]])                  # atleast_2d(b)\n    array([[1]])\n\n\n    '
    (arrays, list_ndim, result_ndim, final_size) = _block_setup(arrays)
    if list_ndim * final_size > 2 * 512 * 512:
        return _block_slicing(arrays, list_ndim, result_ndim)
    else:
        return _block_concatenate(arrays, list_ndim, result_ndim)

def _block_setup(arrays):
    if False:
        i = 10
        return i + 15
    '\n    Returns\n    (`arrays`, list_ndim, result_ndim, final_size)\n    '
    (bottom_index, arr_ndim, final_size) = _block_check_depths_match(arrays)
    list_ndim = len(bottom_index)
    if bottom_index and bottom_index[-1] is None:
        raise ValueError('List at {} cannot be empty'.format(_block_format_index(bottom_index)))
    result_ndim = max(arr_ndim, list_ndim)
    return (arrays, list_ndim, result_ndim, final_size)

def _block_slicing(arrays, list_ndim, result_ndim):
    if False:
        return 10
    (shape, slices, arrays) = _block_info_recursion(arrays, list_ndim, result_ndim)
    dtype = _nx.result_type(*[arr.dtype for arr in arrays])
    F_order = all((arr.flags['F_CONTIGUOUS'] for arr in arrays))
    C_order = all((arr.flags['C_CONTIGUOUS'] for arr in arrays))
    order = 'F' if F_order and (not C_order) else 'C'
    result = _nx.empty(shape=shape, dtype=dtype, order=order)
    for (the_slice, arr) in zip(slices, arrays):
        result[(Ellipsis,) + the_slice] = arr
    return result

def _block_concatenate(arrays, list_ndim, result_ndim):
    if False:
        while True:
            i = 10
    result = _block(arrays, list_ndim, result_ndim)
    if list_ndim == 0:
        result = result.copy()
    return result