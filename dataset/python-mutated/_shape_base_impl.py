import functools
import warnings
import numpy._core.numeric as _nx
from numpy._core.numeric import asarray, zeros, zeros_like, array, asanyarray
from numpy._core.fromnumeric import reshape, transpose
from numpy._core.multiarray import normalize_axis_index
from numpy._core import overrides
from numpy._core import vstack, atleast_3d
from numpy._core.numeric import normalize_axis_tuple
from numpy._core.overrides import set_module
from numpy._core.shape_base import _arrays_for_stack_dispatcher
from numpy.lib._index_tricks_impl import ndindex
from numpy.matrixlib.defmatrix import matrix
__all__ = ['column_stack', 'row_stack', 'dstack', 'array_split', 'split', 'hsplit', 'vsplit', 'dsplit', 'apply_over_axes', 'expand_dims', 'apply_along_axis', 'kron', 'tile', 'take_along_axis', 'put_along_axis']
array_function_dispatch = functools.partial(overrides.array_function_dispatch, module='numpy')

def _make_along_axis_idx(arr_shape, indices, axis):
    if False:
        for i in range(10):
            print('nop')
    if not _nx.issubdtype(indices.dtype, _nx.integer):
        raise IndexError('`indices` must be an integer array')
    if len(arr_shape) != indices.ndim:
        raise ValueError('`indices` and `arr` must have the same number of dimensions')
    shape_ones = (1,) * indices.ndim
    dest_dims = list(range(axis)) + [None] + list(range(axis + 1, indices.ndim))
    fancy_index = []
    for (dim, n) in zip(dest_dims, arr_shape):
        if dim is None:
            fancy_index.append(indices)
        else:
            ind_shape = shape_ones[:dim] + (-1,) + shape_ones[dim + 1:]
            fancy_index.append(_nx.arange(n).reshape(ind_shape))
    return tuple(fancy_index)

def _take_along_axis_dispatcher(arr, indices, axis):
    if False:
        for i in range(10):
            print('nop')
    return (arr, indices)

@array_function_dispatch(_take_along_axis_dispatcher)
def take_along_axis(arr, indices, axis):
    if False:
        while True:
            i = 10
    '\n    Take values from the input array by matching 1d index and data slices.\n\n    This iterates over matching 1d slices oriented along the specified axis in\n    the index and data arrays, and uses the former to look up values in the\n    latter. These slices can be different lengths.\n\n    Functions returning an index along an axis, like `argsort` and\n    `argpartition`, produce suitable indices for this function.\n\n    .. versionadded:: 1.15.0\n\n    Parameters\n    ----------\n    arr : ndarray (Ni..., M, Nk...)\n        Source array\n    indices : ndarray (Ni..., J, Nk...)\n        Indices to take along each 1d slice of `arr`. This must match the\n        dimension of arr, but dimensions Ni and Nj only need to broadcast\n        against `arr`.\n    axis : int\n        The axis to take 1d slices along. If axis is None, the input array is\n        treated as if it had first been flattened to 1d, for consistency with\n        `sort` and `argsort`.\n\n    Returns\n    -------\n    out: ndarray (Ni..., J, Nk...)\n        The indexed result.\n\n    Notes\n    -----\n    This is equivalent to (but faster than) the following use of `ndindex` and\n    `s_`, which sets each of ``ii`` and ``kk`` to a tuple of indices::\n\n        Ni, M, Nk = a.shape[:axis], a.shape[axis], a.shape[axis+1:]\n        J = indices.shape[axis]  # Need not equal M\n        out = np.empty(Ni + (J,) + Nk)\n\n        for ii in ndindex(Ni):\n            for kk in ndindex(Nk):\n                a_1d       = a      [ii + s_[:,] + kk]\n                indices_1d = indices[ii + s_[:,] + kk]\n                out_1d     = out    [ii + s_[:,] + kk]\n                for j in range(J):\n                    out_1d[j] = a_1d[indices_1d[j]]\n\n    Equivalently, eliminating the inner loop, the last two lines would be::\n\n                out_1d[:] = a_1d[indices_1d]\n\n    See Also\n    --------\n    take : Take along an axis, using the same indices for every 1d slice\n    put_along_axis :\n        Put values into the destination array by matching 1d index and data slices\n\n    Examples\n    --------\n\n    For this sample array\n\n    >>> a = np.array([[10, 30, 20], [60, 40, 50]])\n\n    We can sort either by using sort directly, or argsort and this function\n\n    >>> np.sort(a, axis=1)\n    array([[10, 20, 30],\n           [40, 50, 60]])\n    >>> ai = np.argsort(a, axis=1)\n    >>> ai\n    array([[0, 2, 1],\n           [1, 2, 0]])\n    >>> np.take_along_axis(a, ai, axis=1)\n    array([[10, 20, 30],\n           [40, 50, 60]])\n\n    The same works for max and min, if you maintain the trivial dimension\n    with ``keepdims``:\n\n    >>> np.max(a, axis=1, keepdims=True)\n    array([[30],\n           [60]])\n    >>> ai = np.argmax(a, axis=1, keepdims=True)\n    >>> ai\n    array([[1],\n           [0]])\n    >>> np.take_along_axis(a, ai, axis=1)\n    array([[30],\n           [60]])\n\n    If we want to get the max and min at the same time, we can stack the\n    indices first\n\n    >>> ai_min = np.argmin(a, axis=1, keepdims=True)\n    >>> ai_max = np.argmax(a, axis=1, keepdims=True)\n    >>> ai = np.concatenate([ai_min, ai_max], axis=1)\n    >>> ai\n    array([[0, 1],\n           [1, 0]])\n    >>> np.take_along_axis(a, ai, axis=1)\n    array([[10, 30],\n           [40, 60]])\n    '
    if axis is None:
        arr = arr.flat
        arr_shape = (len(arr),)
        axis = 0
    else:
        axis = normalize_axis_index(axis, arr.ndim)
        arr_shape = arr.shape
    return arr[_make_along_axis_idx(arr_shape, indices, axis)]

def _put_along_axis_dispatcher(arr, indices, values, axis):
    if False:
        while True:
            i = 10
    return (arr, indices, values)

@array_function_dispatch(_put_along_axis_dispatcher)
def put_along_axis(arr, indices, values, axis):
    if False:
        print('Hello World!')
    '\n    Put values into the destination array by matching 1d index and data slices.\n\n    This iterates over matching 1d slices oriented along the specified axis in\n    the index and data arrays, and uses the former to place values into the\n    latter. These slices can be different lengths.\n\n    Functions returning an index along an axis, like `argsort` and\n    `argpartition`, produce suitable indices for this function.\n\n    .. versionadded:: 1.15.0\n\n    Parameters\n    ----------\n    arr : ndarray (Ni..., M, Nk...)\n        Destination array.\n    indices : ndarray (Ni..., J, Nk...)\n        Indices to change along each 1d slice of `arr`. This must match the\n        dimension of arr, but dimensions in Ni and Nj may be 1 to broadcast\n        against `arr`.\n    values : array_like (Ni..., J, Nk...)\n        values to insert at those indices. Its shape and dimension are\n        broadcast to match that of `indices`.\n    axis : int\n        The axis to take 1d slices along. If axis is None, the destination\n        array is treated as if a flattened 1d view had been created of it.\n\n    Notes\n    -----\n    This is equivalent to (but faster than) the following use of `ndindex` and\n    `s_`, which sets each of ``ii`` and ``kk`` to a tuple of indices::\n\n        Ni, M, Nk = a.shape[:axis], a.shape[axis], a.shape[axis+1:]\n        J = indices.shape[axis]  # Need not equal M\n\n        for ii in ndindex(Ni):\n            for kk in ndindex(Nk):\n                a_1d       = a      [ii + s_[:,] + kk]\n                indices_1d = indices[ii + s_[:,] + kk]\n                values_1d  = values [ii + s_[:,] + kk]\n                for j in range(J):\n                    a_1d[indices_1d[j]] = values_1d[j]\n\n    Equivalently, eliminating the inner loop, the last two lines would be::\n\n                a_1d[indices_1d] = values_1d\n\n    See Also\n    --------\n    take_along_axis :\n        Take values from the input array by matching 1d index and data slices\n\n    Examples\n    --------\n\n    For this sample array\n\n    >>> a = np.array([[10, 30, 20], [60, 40, 50]])\n\n    We can replace the maximum values with:\n\n    >>> ai = np.argmax(a, axis=1, keepdims=True)\n    >>> ai\n    array([[1],\n           [0]])\n    >>> np.put_along_axis(a, ai, 99, axis=1)\n    >>> a\n    array([[10, 99, 20],\n           [99, 40, 50]])\n\n    '
    if axis is None:
        arr = arr.flat
        axis = 0
        arr_shape = (len(arr),)
    else:
        axis = normalize_axis_index(axis, arr.ndim)
        arr_shape = arr.shape
    arr[_make_along_axis_idx(arr_shape, indices, axis)] = values

def _apply_along_axis_dispatcher(func1d, axis, arr, *args, **kwargs):
    if False:
        return 10
    return (arr,)

@array_function_dispatch(_apply_along_axis_dispatcher)
def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Apply a function to 1-D slices along the given axis.\n\n    Execute `func1d(a, *args, **kwargs)` where `func1d` operates on 1-D arrays\n    and `a` is a 1-D slice of `arr` along `axis`.\n\n    This is equivalent to (but faster than) the following use of `ndindex` and\n    `s_`, which sets each of ``ii``, ``jj``, and ``kk`` to a tuple of indices::\n\n        Ni, Nk = a.shape[:axis], a.shape[axis+1:]\n        for ii in ndindex(Ni):\n            for kk in ndindex(Nk):\n                f = func1d(arr[ii + s_[:,] + kk])\n                Nj = f.shape\n                for jj in ndindex(Nj):\n                    out[ii + jj + kk] = f[jj]\n\n    Equivalently, eliminating the inner loop, this can be expressed as::\n\n        Ni, Nk = a.shape[:axis], a.shape[axis+1:]\n        for ii in ndindex(Ni):\n            for kk in ndindex(Nk):\n                out[ii + s_[...,] + kk] = func1d(arr[ii + s_[:,] + kk])\n\n    Parameters\n    ----------\n    func1d : function (M,) -> (Nj...)\n        This function should accept 1-D arrays. It is applied to 1-D\n        slices of `arr` along the specified axis.\n    axis : integer\n        Axis along which `arr` is sliced.\n    arr : ndarray (Ni..., M, Nk...)\n        Input array.\n    args : any\n        Additional arguments to `func1d`.\n    kwargs : any\n        Additional named arguments to `func1d`.\n\n        .. versionadded:: 1.9.0\n\n\n    Returns\n    -------\n    out : ndarray  (Ni..., Nj..., Nk...)\n        The output array. The shape of `out` is identical to the shape of\n        `arr`, except along the `axis` dimension. This axis is removed, and\n        replaced with new dimensions equal to the shape of the return value\n        of `func1d`. So if `func1d` returns a scalar `out` will have one\n        fewer dimensions than `arr`.\n\n    See Also\n    --------\n    apply_over_axes : Apply a function repeatedly over multiple axes.\n\n    Examples\n    --------\n    >>> def my_func(a):\n    ...     """Average first and last element of a 1-D array"""\n    ...     return (a[0] + a[-1]) * 0.5\n    >>> b = np.array([[1,2,3], [4,5,6], [7,8,9]])\n    >>> np.apply_along_axis(my_func, 0, b)\n    array([4., 5., 6.])\n    >>> np.apply_along_axis(my_func, 1, b)\n    array([2.,  5.,  8.])\n\n    For a function that returns a 1D array, the number of dimensions in\n    `outarr` is the same as `arr`.\n\n    >>> b = np.array([[8,1,7], [4,3,9], [5,2,6]])\n    >>> np.apply_along_axis(sorted, 1, b)\n    array([[1, 7, 8],\n           [3, 4, 9],\n           [2, 5, 6]])\n\n    For a function that returns a higher dimensional array, those dimensions\n    are inserted in place of the `axis` dimension.\n\n    >>> b = np.array([[1,2,3], [4,5,6], [7,8,9]])\n    >>> np.apply_along_axis(np.diag, -1, b)\n    array([[[1, 0, 0],\n            [0, 2, 0],\n            [0, 0, 3]],\n           [[4, 0, 0],\n            [0, 5, 0],\n            [0, 0, 6]],\n           [[7, 0, 0],\n            [0, 8, 0],\n            [0, 0, 9]]])\n    '
    arr = asanyarray(arr)
    nd = arr.ndim
    axis = normalize_axis_index(axis, nd)
    in_dims = list(range(nd))
    inarr_view = transpose(arr, in_dims[:axis] + in_dims[axis + 1:] + [axis])
    inds = ndindex(inarr_view.shape[:-1])
    inds = (ind + (Ellipsis,) for ind in inds)
    try:
        ind0 = next(inds)
    except StopIteration:
        raise ValueError('Cannot apply_along_axis when any iteration dimensions are 0') from None
    res = asanyarray(func1d(inarr_view[ind0], *args, **kwargs))
    if not isinstance(res, matrix):
        buff = zeros_like(res, shape=inarr_view.shape[:-1] + res.shape)
    else:
        buff = zeros(inarr_view.shape[:-1] + res.shape, dtype=res.dtype)
    buff_dims = list(range(buff.ndim))
    buff_permute = buff_dims[0:axis] + buff_dims[buff.ndim - res.ndim:buff.ndim] + buff_dims[axis:buff.ndim - res.ndim]
    buff[ind0] = res
    for ind in inds:
        buff[ind] = asanyarray(func1d(inarr_view[ind], *args, **kwargs))
    if not isinstance(res, matrix):
        buff = res.__array_wrap__(buff)
        return transpose(buff, buff_permute)
    else:
        out_arr = transpose(buff, buff_permute)
        return res.__array_wrap__(out_arr)

def _apply_over_axes_dispatcher(func, a, axes):
    if False:
        i = 10
        return i + 15
    return (a,)

@array_function_dispatch(_apply_over_axes_dispatcher)
def apply_over_axes(func, a, axes):
    if False:
        i = 10
        return i + 15
    '\n    Apply a function repeatedly over multiple axes.\n\n    `func` is called as `res = func(a, axis)`, where `axis` is the first\n    element of `axes`.  The result `res` of the function call must have\n    either the same dimensions as `a` or one less dimension.  If `res`\n    has one less dimension than `a`, a dimension is inserted before\n    `axis`.  The call to `func` is then repeated for each axis in `axes`,\n    with `res` as the first argument.\n\n    Parameters\n    ----------\n    func : function\n        This function must take two arguments, `func(a, axis)`.\n    a : array_like\n        Input array.\n    axes : array_like\n        Axes over which `func` is applied; the elements must be integers.\n\n    Returns\n    -------\n    apply_over_axis : ndarray\n        The output array.  The number of dimensions is the same as `a`,\n        but the shape can be different.  This depends on whether `func`\n        changes the shape of its output with respect to its input.\n\n    See Also\n    --------\n    apply_along_axis :\n        Apply a function to 1-D slices of an array along the given axis.\n\n    Notes\n    -----\n    This function is equivalent to tuple axis arguments to reorderable ufuncs\n    with keepdims=True. Tuple axis arguments to ufuncs have been available since\n    version 1.7.0.\n\n    Examples\n    --------\n    >>> a = np.arange(24).reshape(2,3,4)\n    >>> a\n    array([[[ 0,  1,  2,  3],\n            [ 4,  5,  6,  7],\n            [ 8,  9, 10, 11]],\n           [[12, 13, 14, 15],\n            [16, 17, 18, 19],\n            [20, 21, 22, 23]]])\n\n    Sum over axes 0 and 2. The result has same number of dimensions\n    as the original array:\n\n    >>> np.apply_over_axes(np.sum, a, [0,2])\n    array([[[ 60],\n            [ 92],\n            [124]]])\n\n    Tuple axis arguments to ufuncs are equivalent:\n\n    >>> np.sum(a, axis=(0,2), keepdims=True)\n    array([[[ 60],\n            [ 92],\n            [124]]])\n\n    '
    val = asarray(a)
    N = a.ndim
    if array(axes).ndim == 0:
        axes = (axes,)
    for axis in axes:
        if axis < 0:
            axis = N + axis
        args = (val, axis)
        res = func(*args)
        if res.ndim == val.ndim:
            val = res
        else:
            res = expand_dims(res, axis)
            if res.ndim == val.ndim:
                val = res
            else:
                raise ValueError('function is not returning an array of the correct shape')
    return val

def _expand_dims_dispatcher(a, axis):
    if False:
        print('Hello World!')
    return (a,)

@array_function_dispatch(_expand_dims_dispatcher)
def expand_dims(a, axis):
    if False:
        for i in range(10):
            print('nop')
    '\n    Expand the shape of an array.\n\n    Insert a new axis that will appear at the `axis` position in the expanded\n    array shape.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array.\n    axis : int or tuple of ints\n        Position in the expanded axes where the new axis (or axes) is placed.\n\n        .. deprecated:: 1.13.0\n            Passing an axis where ``axis > a.ndim`` will be treated as\n            ``axis == a.ndim``, and passing ``axis < -a.ndim - 1`` will\n            be treated as ``axis == 0``. This behavior is deprecated.\n\n        .. versionchanged:: 1.18.0\n            A tuple of axes is now supported.  Out of range axes as\n            described above are now forbidden and raise an\n            `~exceptions.AxisError`.\n\n    Returns\n    -------\n    result : ndarray\n        View of `a` with the number of dimensions increased.\n\n    See Also\n    --------\n    squeeze : The inverse operation, removing singleton dimensions\n    reshape : Insert, remove, and combine dimensions, and resize existing ones\n    atleast_1d, atleast_2d, atleast_3d\n\n    Examples\n    --------\n    >>> x = np.array([1, 2])\n    >>> x.shape\n    (2,)\n\n    The following is equivalent to ``x[np.newaxis, :]`` or ``x[np.newaxis]``:\n\n    >>> y = np.expand_dims(x, axis=0)\n    >>> y\n    array([[1, 2]])\n    >>> y.shape\n    (1, 2)\n\n    The following is equivalent to ``x[:, np.newaxis]``:\n\n    >>> y = np.expand_dims(x, axis=1)\n    >>> y\n    array([[1],\n           [2]])\n    >>> y.shape\n    (2, 1)\n\n    ``axis`` may also be a tuple:\n\n    >>> y = np.expand_dims(x, axis=(0, 1))\n    >>> y\n    array([[[1, 2]]])\n\n    >>> y = np.expand_dims(x, axis=(2, 0))\n    >>> y\n    array([[[1],\n            [2]]])\n\n    Note that some examples may use ``None`` instead of ``np.newaxis``.  These\n    are the same objects:\n\n    >>> np.newaxis is None\n    True\n\n    '
    if isinstance(a, matrix):
        a = asarray(a)
    else:
        a = asanyarray(a)
    if type(axis) not in (tuple, list):
        axis = (axis,)
    out_ndim = len(axis) + a.ndim
    axis = normalize_axis_tuple(axis, out_ndim)
    shape_it = iter(a.shape)
    shape = [1 if ax in axis else next(shape_it) for ax in range(out_ndim)]
    return a.reshape(shape)

@set_module('numpy')
def row_stack(tup, *, dtype=None, casting='same_kind'):
    if False:
        return 10
    warnings.warn('`row_stack` alias is deprecated. Use `np.vstack` directly.', DeprecationWarning, stacklevel=2)
    return vstack(tup, dtype=dtype, casting=casting)
row_stack.__doc__ = vstack.__doc__

def _column_stack_dispatcher(tup):
    if False:
        i = 10
        return i + 15
    return _arrays_for_stack_dispatcher(tup)

@array_function_dispatch(_column_stack_dispatcher)
def column_stack(tup):
    if False:
        while True:
            i = 10
    '\n    Stack 1-D arrays as columns into a 2-D array.\n\n    Take a sequence of 1-D arrays and stack them as columns\n    to make a single 2-D array. 2-D arrays are stacked as-is,\n    just like with `hstack`.  1-D arrays are turned into 2-D columns\n    first.\n\n    Parameters\n    ----------\n    tup : sequence of 1-D or 2-D arrays.\n        Arrays to stack. All of them must have the same first dimension.\n\n    Returns\n    -------\n    stacked : 2-D array\n        The array formed by stacking the given arrays.\n\n    See Also\n    --------\n    stack, hstack, vstack, concatenate\n\n    Examples\n    --------\n    >>> a = np.array((1,2,3))\n    >>> b = np.array((2,3,4))\n    >>> np.column_stack((a,b))\n    array([[1, 2],\n           [2, 3],\n           [3, 4]])\n\n    '
    arrays = []
    for v in tup:
        arr = asanyarray(v)
        if arr.ndim < 2:
            arr = array(arr, copy=False, subok=True, ndmin=2).T
        arrays.append(arr)
    return _nx.concatenate(arrays, 1)

def _dstack_dispatcher(tup):
    if False:
        for i in range(10):
            print('nop')
    return _arrays_for_stack_dispatcher(tup)

@array_function_dispatch(_dstack_dispatcher)
def dstack(tup):
    if False:
        print('Hello World!')
    '\n    Stack arrays in sequence depth wise (along third axis).\n\n    This is equivalent to concatenation along the third axis after 2-D arrays\n    of shape `(M,N)` have been reshaped to `(M,N,1)` and 1-D arrays of shape\n    `(N,)` have been reshaped to `(1,N,1)`. Rebuilds arrays divided by\n    `dsplit`.\n\n    This function makes most sense for arrays with up to 3 dimensions. For\n    instance, for pixel-data with a height (first axis), width (second axis),\n    and r/g/b channels (third axis). The functions `concatenate`, `stack` and\n    `block` provide more general stacking and concatenation operations.\n\n    Parameters\n    ----------\n    tup : sequence of arrays\n        The arrays must have the same shape along all but the third axis.\n        1-D or 2-D arrays must have the same shape.\n\n    Returns\n    -------\n    stacked : ndarray\n        The array formed by stacking the given arrays, will be at least 3-D.\n\n    See Also\n    --------\n    concatenate : Join a sequence of arrays along an existing axis.\n    stack : Join a sequence of arrays along a new axis.\n    block : Assemble an nd-array from nested lists of blocks.\n    vstack : Stack arrays in sequence vertically (row wise).\n    hstack : Stack arrays in sequence horizontally (column wise).\n    column_stack : Stack 1-D arrays as columns into a 2-D array.\n    dsplit : Split array along third axis.\n\n    Examples\n    --------\n    >>> a = np.array((1,2,3))\n    >>> b = np.array((2,3,4))\n    >>> np.dstack((a,b))\n    array([[[1, 2],\n            [2, 3],\n            [3, 4]]])\n\n    >>> a = np.array([[1],[2],[3]])\n    >>> b = np.array([[2],[3],[4]])\n    >>> np.dstack((a,b))\n    array([[[1, 2]],\n           [[2, 3]],\n           [[3, 4]]])\n\n    '
    arrs = atleast_3d(*tup)
    if not isinstance(arrs, list):
        arrs = [arrs]
    return _nx.concatenate(arrs, 2)

def _replace_zero_by_x_arrays(sub_arys):
    if False:
        print('Hello World!')
    for i in range(len(sub_arys)):
        if _nx.ndim(sub_arys[i]) == 0:
            sub_arys[i] = _nx.empty(0, dtype=sub_arys[i].dtype)
        elif _nx.sometrue(_nx.equal(_nx.shape(sub_arys[i]), 0)):
            sub_arys[i] = _nx.empty(0, dtype=sub_arys[i].dtype)
    return sub_arys

def _array_split_dispatcher(ary, indices_or_sections, axis=None):
    if False:
        return 10
    return (ary, indices_or_sections)

@array_function_dispatch(_array_split_dispatcher)
def array_split(ary, indices_or_sections, axis=0):
    if False:
        return 10
    '\n    Split an array into multiple sub-arrays.\n\n    Please refer to the ``split`` documentation.  The only difference\n    between these functions is that ``array_split`` allows\n    `indices_or_sections` to be an integer that does *not* equally\n    divide the axis. For an array of length l that should be split\n    into n sections, it returns l % n sub-arrays of size l//n + 1\n    and the rest of size l//n.\n\n    See Also\n    --------\n    split : Split array into multiple sub-arrays of equal size.\n\n    Examples\n    --------\n    >>> x = np.arange(8.0)\n    >>> np.array_split(x, 3)\n    [array([0.,  1.,  2.]), array([3.,  4.,  5.]), array([6.,  7.])]\n\n    >>> x = np.arange(9)\n    >>> np.array_split(x, 4)\n    [array([0, 1, 2]), array([3, 4]), array([5, 6]), array([7, 8])]\n\n    '
    try:
        Ntotal = ary.shape[axis]
    except AttributeError:
        Ntotal = len(ary)
    try:
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError:
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError('number sections must be larger than 0.') from None
        (Neach_section, extras) = divmod(Ntotal, Nsections)
        section_sizes = [0] + extras * [Neach_section + 1] + (Nsections - extras) * [Neach_section]
        div_points = _nx.array(section_sizes, dtype=_nx.intp).cumsum()
    sub_arys = []
    sary = _nx.swapaxes(ary, axis, 0)
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_arys.append(_nx.swapaxes(sary[st:end], axis, 0))
    return sub_arys

def _split_dispatcher(ary, indices_or_sections, axis=None):
    if False:
        return 10
    return (ary, indices_or_sections)

@array_function_dispatch(_split_dispatcher)
def split(ary, indices_or_sections, axis=0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Split an array into multiple sub-arrays as views into `ary`.\n\n    Parameters\n    ----------\n    ary : ndarray\n        Array to be divided into sub-arrays.\n    indices_or_sections : int or 1-D array\n        If `indices_or_sections` is an integer, N, the array will be divided\n        into N equal arrays along `axis`.  If such a split is not possible,\n        an error is raised.\n\n        If `indices_or_sections` is a 1-D array of sorted integers, the entries\n        indicate where along `axis` the array is split.  For example,\n        ``[2, 3]`` would, for ``axis=0``, result in\n\n        - ary[:2]\n        - ary[2:3]\n        - ary[3:]\n\n        If an index exceeds the dimension of the array along `axis`,\n        an empty sub-array is returned correspondingly.\n    axis : int, optional\n        The axis along which to split, default is 0.\n\n    Returns\n    -------\n    sub-arrays : list of ndarrays\n        A list of sub-arrays as views into `ary`.\n\n    Raises\n    ------\n    ValueError\n        If `indices_or_sections` is given as an integer, but\n        a split does not result in equal division.\n\n    See Also\n    --------\n    array_split : Split an array into multiple sub-arrays of equal or\n                  near-equal size.  Does not raise an exception if\n                  an equal division cannot be made.\n    hsplit : Split array into multiple sub-arrays horizontally (column-wise).\n    vsplit : Split array into multiple sub-arrays vertically (row wise).\n    dsplit : Split array into multiple sub-arrays along the 3rd axis (depth).\n    concatenate : Join a sequence of arrays along an existing axis.\n    stack : Join a sequence of arrays along a new axis.\n    hstack : Stack arrays in sequence horizontally (column wise).\n    vstack : Stack arrays in sequence vertically (row wise).\n    dstack : Stack arrays in sequence depth wise (along third dimension).\n\n    Examples\n    --------\n    >>> x = np.arange(9.0)\n    >>> np.split(x, 3)\n    [array([0.,  1.,  2.]), array([3.,  4.,  5.]), array([6.,  7.,  8.])]\n\n    >>> x = np.arange(8.0)\n    >>> np.split(x, [3, 5, 6, 10])\n    [array([0.,  1.,  2.]),\n     array([3.,  4.]),\n     array([5.]),\n     array([6.,  7.]),\n     array([], dtype=float64)]\n\n    '
    try:
        len(indices_or_sections)
    except TypeError:
        sections = indices_or_sections
        N = ary.shape[axis]
        if N % sections:
            raise ValueError('array split does not result in an equal division') from None
    return array_split(ary, indices_or_sections, axis)

def _hvdsplit_dispatcher(ary, indices_or_sections):
    if False:
        i = 10
        return i + 15
    return (ary, indices_or_sections)

@array_function_dispatch(_hvdsplit_dispatcher)
def hsplit(ary, indices_or_sections):
    if False:
        print('Hello World!')
    '\n    Split an array into multiple sub-arrays horizontally (column-wise).\n\n    Please refer to the `split` documentation.  `hsplit` is equivalent\n    to `split` with ``axis=1``, the array is always split along the second\n    axis except for 1-D arrays, where it is split at ``axis=0``.\n\n    See Also\n    --------\n    split : Split an array into multiple sub-arrays of equal size.\n\n    Examples\n    --------\n    >>> x = np.arange(16.0).reshape(4, 4)\n    >>> x\n    array([[ 0.,   1.,   2.,   3.],\n           [ 4.,   5.,   6.,   7.],\n           [ 8.,   9.,  10.,  11.],\n           [12.,  13.,  14.,  15.]])\n    >>> np.hsplit(x, 2)\n    [array([[  0.,   1.],\n           [  4.,   5.],\n           [  8.,   9.],\n           [12.,  13.]]),\n     array([[  2.,   3.],\n           [  6.,   7.],\n           [10.,  11.],\n           [14.,  15.]])]\n    >>> np.hsplit(x, np.array([3, 6]))\n    [array([[ 0.,   1.,   2.],\n           [ 4.,   5.,   6.],\n           [ 8.,   9.,  10.],\n           [12.,  13.,  14.]]),\n     array([[ 3.],\n           [ 7.],\n           [11.],\n           [15.]]),\n     array([], shape=(4, 0), dtype=float64)]\n\n    With a higher dimensional array the split is still along the second axis.\n\n    >>> x = np.arange(8.0).reshape(2, 2, 2)\n    >>> x\n    array([[[0.,  1.],\n            [2.,  3.]],\n           [[4.,  5.],\n            [6.,  7.]]])\n    >>> np.hsplit(x, 2)\n    [array([[[0.,  1.]],\n           [[4.,  5.]]]),\n     array([[[2.,  3.]],\n           [[6.,  7.]]])]\n\n    With a 1-D array, the split is along axis 0.\n\n    >>> x = np.array([0, 1, 2, 3, 4, 5])\n    >>> np.hsplit(x, 2)\n    [array([0, 1, 2]), array([3, 4, 5])]\n\n    '
    if _nx.ndim(ary) == 0:
        raise ValueError('hsplit only works on arrays of 1 or more dimensions')
    if ary.ndim > 1:
        return split(ary, indices_or_sections, 1)
    else:
        return split(ary, indices_or_sections, 0)

@array_function_dispatch(_hvdsplit_dispatcher)
def vsplit(ary, indices_or_sections):
    if False:
        while True:
            i = 10
    '\n    Split an array into multiple sub-arrays vertically (row-wise).\n\n    Please refer to the ``split`` documentation.  ``vsplit`` is equivalent\n    to ``split`` with `axis=0` (default), the array is always split along the\n    first axis regardless of the array dimension.\n\n    See Also\n    --------\n    split : Split an array into multiple sub-arrays of equal size.\n\n    Examples\n    --------\n    >>> x = np.arange(16.0).reshape(4, 4)\n    >>> x\n    array([[ 0.,   1.,   2.,   3.],\n           [ 4.,   5.,   6.,   7.],\n           [ 8.,   9.,  10.,  11.],\n           [12.,  13.,  14.,  15.]])\n    >>> np.vsplit(x, 2)\n    [array([[0., 1., 2., 3.],\n           [4., 5., 6., 7.]]), array([[ 8.,  9., 10., 11.],\n           [12., 13., 14., 15.]])]\n    >>> np.vsplit(x, np.array([3, 6]))\n    [array([[ 0.,  1.,  2.,  3.],\n           [ 4.,  5.,  6.,  7.],\n           [ 8.,  9., 10., 11.]]), array([[12., 13., 14., 15.]]), array([], shape=(0, 4), dtype=float64)]\n\n    With a higher dimensional array the split is still along the first axis.\n\n    >>> x = np.arange(8.0).reshape(2, 2, 2)\n    >>> x\n    array([[[0.,  1.],\n            [2.,  3.]],\n           [[4.,  5.],\n            [6.,  7.]]])\n    >>> np.vsplit(x, 2)\n    [array([[[0., 1.],\n            [2., 3.]]]), array([[[4., 5.],\n            [6., 7.]]])]\n\n    '
    if _nx.ndim(ary) < 2:
        raise ValueError('vsplit only works on arrays of 2 or more dimensions')
    return split(ary, indices_or_sections, 0)

@array_function_dispatch(_hvdsplit_dispatcher)
def dsplit(ary, indices_or_sections):
    if False:
        print('Hello World!')
    '\n    Split array into multiple sub-arrays along the 3rd axis (depth).\n\n    Please refer to the `split` documentation.  `dsplit` is equivalent\n    to `split` with ``axis=2``, the array is always split along the third\n    axis provided the array dimension is greater than or equal to 3.\n\n    See Also\n    --------\n    split : Split an array into multiple sub-arrays of equal size.\n\n    Examples\n    --------\n    >>> x = np.arange(16.0).reshape(2, 2, 4)\n    >>> x\n    array([[[ 0.,   1.,   2.,   3.],\n            [ 4.,   5.,   6.,   7.]],\n           [[ 8.,   9.,  10.,  11.],\n            [12.,  13.,  14.,  15.]]])\n    >>> np.dsplit(x, 2)\n    [array([[[ 0.,  1.],\n            [ 4.,  5.]],\n           [[ 8.,  9.],\n            [12., 13.]]]), array([[[ 2.,  3.],\n            [ 6.,  7.]],\n           [[10., 11.],\n            [14., 15.]]])]\n    >>> np.dsplit(x, np.array([3, 6]))\n    [array([[[ 0.,   1.,   2.],\n            [ 4.,   5.,   6.]],\n           [[ 8.,   9.,  10.],\n            [12.,  13.,  14.]]]),\n     array([[[ 3.],\n            [ 7.]],\n           [[11.],\n            [15.]]]),\n    array([], shape=(2, 2, 0), dtype=float64)]\n    '
    if _nx.ndim(ary) < 3:
        raise ValueError('dsplit only works on arrays of 3 or more dimensions')
    return split(ary, indices_or_sections, 2)

def get_array_wrap(*args):
    if False:
        print('Hello World!')
    'Find the wrapper for the array with the highest priority.\n\n    In case of ties, leftmost wins. If no wrapper is found, return None.\n\n    .. deprecated:: 2.0\n    '
    warnings.warn('`get_array_wrap` is deprecated. (deprecated in NumPy 2.0)', DeprecationWarning, stacklevel=2)
    wrappers = sorted(((getattr(x, '__array_priority__', 0), -i, x.__array_wrap__) for (i, x) in enumerate(args) if hasattr(x, '__array_wrap__')))
    if wrappers:
        return wrappers[-1][-1]
    return None

def _kron_dispatcher(a, b):
    if False:
        return 10
    return (a, b)

@array_function_dispatch(_kron_dispatcher)
def kron(a, b):
    if False:
        while True:
            i = 10
    '\n    Kronecker product of two arrays.\n\n    Computes the Kronecker product, a composite array made of blocks of the\n    second array scaled by the first.\n\n    Parameters\n    ----------\n    a, b : array_like\n\n    Returns\n    -------\n    out : ndarray\n\n    See Also\n    --------\n    outer : The outer product\n\n    Notes\n    -----\n    The function assumes that the number of dimensions of `a` and `b`\n    are the same, if necessary prepending the smallest with ones.\n    If ``a.shape = (r0,r1,..,rN)`` and ``b.shape = (s0,s1,...,sN)``,\n    the Kronecker product has shape ``(r0*s0, r1*s1, ..., rN*SN)``.\n    The elements are products of elements from `a` and `b`, organized\n    explicitly by::\n\n        kron(a,b)[k0,k1,...,kN] = a[i0,i1,...,iN] * b[j0,j1,...,jN]\n\n    where::\n\n        kt = it * st + jt,  t = 0,...,N\n\n    In the common 2-D case (N=1), the block structure can be visualized::\n\n        [[ a[0,0]*b,   a[0,1]*b,  ... , a[0,-1]*b  ],\n         [  ...                              ...   ],\n         [ a[-1,0]*b,  a[-1,1]*b, ... , a[-1,-1]*b ]]\n\n\n    Examples\n    --------\n    >>> np.kron([1,10,100], [5,6,7])\n    array([  5,   6,   7, ..., 500, 600, 700])\n    >>> np.kron([5,6,7], [1,10,100])\n    array([  5,  50, 500, ...,   7,  70, 700])\n\n    >>> np.kron(np.eye(2), np.ones((2,2)))\n    array([[1.,  1.,  0.,  0.],\n           [1.,  1.,  0.,  0.],\n           [0.,  0.,  1.,  1.],\n           [0.,  0.,  1.,  1.]])\n\n    >>> a = np.arange(100).reshape((2,5,2,5))\n    >>> b = np.arange(24).reshape((2,3,4))\n    >>> c = np.kron(a,b)\n    >>> c.shape\n    (2, 10, 6, 20)\n    >>> I = (1,3,0,2)\n    >>> J = (0,2,1)\n    >>> J1 = (0,) + J             # extend to ndim=4\n    >>> S1 = (1,) + b.shape\n    >>> K = tuple(np.array(I) * np.array(S1) + np.array(J1))\n    >>> c[K] == a[I]*b[J]\n    True\n\n    '
    b = asanyarray(b)
    a = array(a, copy=False, subok=True, ndmin=b.ndim)
    is_any_mat = isinstance(a, matrix) or isinstance(b, matrix)
    (ndb, nda) = (b.ndim, a.ndim)
    nd = max(ndb, nda)
    if nda == 0 or ndb == 0:
        return _nx.multiply(a, b)
    as_ = a.shape
    bs = b.shape
    if not a.flags.contiguous:
        a = reshape(a, as_)
    if not b.flags.contiguous:
        b = reshape(b, bs)
    as_ = (1,) * max(0, ndb - nda) + as_
    bs = (1,) * max(0, nda - ndb) + bs
    a_arr = expand_dims(a, axis=tuple(range(ndb - nda)))
    b_arr = expand_dims(b, axis=tuple(range(nda - ndb)))
    a_arr = expand_dims(a_arr, axis=tuple(range(1, nd * 2, 2)))
    b_arr = expand_dims(b_arr, axis=tuple(range(0, nd * 2, 2)))
    result = _nx.multiply(a_arr, b_arr, subok=not is_any_mat)
    result = result.reshape(_nx.multiply(as_, bs))
    return result if not is_any_mat else matrix(result, copy=False)

def _tile_dispatcher(A, reps):
    if False:
        for i in range(10):
            print('nop')
    return (A, reps)

@array_function_dispatch(_tile_dispatcher)
def tile(A, reps):
    if False:
        i = 10
        return i + 15
    "\n    Construct an array by repeating A the number of times given by reps.\n\n    If `reps` has length ``d``, the result will have dimension of\n    ``max(d, A.ndim)``.\n\n    If ``A.ndim < d``, `A` is promoted to be d-dimensional by prepending new\n    axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication,\n    or shape (1, 1, 3) for 3-D replication. If this is not the desired\n    behavior, promote `A` to d-dimensions manually before calling this\n    function.\n\n    If ``A.ndim > d``, `reps` is promoted to `A`.ndim by prepending 1's to it.\n    Thus for an `A` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as\n    (1, 1, 2, 2).\n\n    Note : Although tile may be used for broadcasting, it is strongly\n    recommended to use numpy's broadcasting operations and functions.\n\n    Parameters\n    ----------\n    A : array_like\n        The input array.\n    reps : array_like\n        The number of repetitions of `A` along each axis.\n\n    Returns\n    -------\n    c : ndarray\n        The tiled output array.\n\n    See Also\n    --------\n    repeat : Repeat elements of an array.\n    broadcast_to : Broadcast an array to a new shape\n\n    Examples\n    --------\n    >>> a = np.array([0, 1, 2])\n    >>> np.tile(a, 2)\n    array([0, 1, 2, 0, 1, 2])\n    >>> np.tile(a, (2, 2))\n    array([[0, 1, 2, 0, 1, 2],\n           [0, 1, 2, 0, 1, 2]])\n    >>> np.tile(a, (2, 1, 2))\n    array([[[0, 1, 2, 0, 1, 2]],\n           [[0, 1, 2, 0, 1, 2]]])\n\n    >>> b = np.array([[1, 2], [3, 4]])\n    >>> np.tile(b, 2)\n    array([[1, 2, 1, 2],\n           [3, 4, 3, 4]])\n    >>> np.tile(b, (2, 1))\n    array([[1, 2],\n           [3, 4],\n           [1, 2],\n           [3, 4]])\n\n    >>> c = np.array([1,2,3,4])\n    >>> np.tile(c,(4,1))\n    array([[1, 2, 3, 4],\n           [1, 2, 3, 4],\n           [1, 2, 3, 4],\n           [1, 2, 3, 4]])\n    "
    try:
        tup = tuple(reps)
    except TypeError:
        tup = (reps,)
    d = len(tup)
    if all((x == 1 for x in tup)) and isinstance(A, _nx.ndarray):
        return _nx.array(A, copy=True, subok=True, ndmin=d)
    else:
        c = _nx.array(A, copy=False, subok=True, ndmin=d)
    if d < c.ndim:
        tup = (1,) * (c.ndim - d) + tup
    shape_out = tuple((s * t for (s, t) in zip(c.shape, tup)))
    n = c.size
    if n > 0:
        for (dim_in, nrep) in zip(c.shape, tup):
            if nrep != 1:
                c = c.reshape(-1, n).repeat(nrep, 0)
            n //= dim_in
    return c.reshape(shape_out)