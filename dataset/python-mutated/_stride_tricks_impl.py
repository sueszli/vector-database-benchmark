"""
Utilities that manipulate strides to achieve desirable effects.

An explanation of strides can be found in the :ref:`arrays.ndarray`.

"""
import numpy as np
from numpy._core.numeric import normalize_axis_tuple
from numpy._core.overrides import array_function_dispatch, set_module
__all__ = ['broadcast_to', 'broadcast_arrays', 'broadcast_shapes']

class DummyArray:
    """Dummy object that just exists to hang __array_interface__ dictionaries
    and possibly keep alive a reference to a base array.
    """

    def __init__(self, interface, base=None):
        if False:
            for i in range(10):
                print('nop')
        self.__array_interface__ = interface
        self.base = base

def _maybe_view_as_subclass(original_array, new_array):
    if False:
        return 10
    if type(original_array) is not type(new_array):
        new_array = new_array.view(type=type(original_array))
        if new_array.__array_finalize__:
            new_array.__array_finalize__(original_array)
    return new_array

def as_strided(x, shape=None, strides=None, subok=False, writeable=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a view into the array with the given shape and strides.\n\n    .. warning:: This function has to be used with extreme care, see notes.\n\n    Parameters\n    ----------\n    x : ndarray\n        Array to create a new.\n    shape : sequence of int, optional\n        The shape of the new array. Defaults to ``x.shape``.\n    strides : sequence of int, optional\n        The strides of the new array. Defaults to ``x.strides``.\n    subok : bool, optional\n        .. versionadded:: 1.10\n\n        If True, subclasses are preserved.\n    writeable : bool, optional\n        .. versionadded:: 1.12\n\n        If set to False, the returned array will always be readonly.\n        Otherwise it will be writable if the original array was. It\n        is advisable to set this to False if possible (see Notes).\n\n    Returns\n    -------\n    view : ndarray\n\n    See also\n    --------\n    broadcast_to : broadcast an array to a given shape.\n    reshape : reshape an array.\n    lib.stride_tricks.sliding_window_view :\n        userfriendly and safe function for a creation of sliding window views.\n\n    Notes\n    -----\n    ``as_strided`` creates a view into the array given the exact strides\n    and shape. This means it manipulates the internal data structure of\n    ndarray and, if done incorrectly, the array elements can point to\n    invalid memory and can corrupt results or crash your program.\n    It is advisable to always use the original ``x.strides`` when\n    calculating new strides to avoid reliance on a contiguous memory\n    layout.\n\n    Furthermore, arrays created with this function often contain self\n    overlapping memory, so that two elements are identical.\n    Vectorized write operations on such arrays will typically be\n    unpredictable. They may even give different results for small, large,\n    or transposed arrays.\n\n    Since writing to these arrays has to be tested and done with great\n    care, you may want to use ``writeable=False`` to avoid accidental write\n    operations.\n\n    For these reasons it is advisable to avoid ``as_strided`` when\n    possible.\n    '
    x = np.array(x, copy=False, subok=subok)
    interface = dict(x.__array_interface__)
    if shape is not None:
        interface['shape'] = tuple(shape)
    if strides is not None:
        interface['strides'] = tuple(strides)
    array = np.asarray(DummyArray(interface, base=x))
    array.dtype = x.dtype
    view = _maybe_view_as_subclass(x, array)
    if view.flags.writeable and (not writeable):
        view.flags.writeable = False
    return view

def _sliding_window_view_dispatcher(x, window_shape, axis=None, *, subok=None, writeable=None):
    if False:
        for i in range(10):
            print('nop')
    return (x,)

@array_function_dispatch(_sliding_window_view_dispatcher)
def sliding_window_view(x, window_shape, axis=None, *, subok=False, writeable=False):
    if False:
        return 10
    '\n    Create a sliding window view into the array with the given window shape.\n\n    Also known as rolling or moving window, the window slides across all\n    dimensions of the array and extracts subsets of the array at all window\n    positions.\n    \n    .. versionadded:: 1.20.0\n\n    Parameters\n    ----------\n    x : array_like\n        Array to create the sliding window view from.\n    window_shape : int or tuple of int\n        Size of window over each axis that takes part in the sliding window.\n        If `axis` is not present, must have same length as the number of input\n        array dimensions. Single integers `i` are treated as if they were the\n        tuple `(i,)`.\n    axis : int or tuple of int, optional\n        Axis or axes along which the sliding window is applied.\n        By default, the sliding window is applied to all axes and\n        `window_shape[i]` will refer to axis `i` of `x`.\n        If `axis` is given as a `tuple of int`, `window_shape[i]` will refer to\n        the axis `axis[i]` of `x`.\n        Single integers `i` are treated as if they were the tuple `(i,)`.\n    subok : bool, optional\n        If True, sub-classes will be passed-through, otherwise the returned\n        array will be forced to be a base-class array (default).\n    writeable : bool, optional\n        When true, allow writing to the returned view. The default is false,\n        as this should be used with caution: the returned view contains the\n        same memory location multiple times, so writing to one location will\n        cause others to change.\n\n    Returns\n    -------\n    view : ndarray\n        Sliding window view of the array. The sliding window dimensions are\n        inserted at the end, and the original dimensions are trimmed as\n        required by the size of the sliding window.\n        That is, ``view.shape = x_shape_trimmed + window_shape``, where\n        ``x_shape_trimmed`` is ``x.shape`` with every entry reduced by one less\n        than the corresponding window size.\n\n    See Also\n    --------\n    lib.stride_tricks.as_strided: A lower-level and less safe routine for\n        creating arbitrary views from custom shape and strides.\n    broadcast_to: broadcast an array to a given shape.\n\n    Notes\n    -----\n    For many applications using a sliding window view can be convenient, but\n    potentially very slow. Often specialized solutions exist, for example:\n\n    - `scipy.signal.fftconvolve`\n\n    - filtering functions in `scipy.ndimage`\n\n    - moving window functions provided by\n      `bottleneck <https://github.com/pydata/bottleneck>`_.\n\n    As a rough estimate, a sliding window approach with an input size of `N`\n    and a window size of `W` will scale as `O(N*W)` where frequently a special\n    algorithm can achieve `O(N)`. That means that the sliding window variant\n    for a window size of 100 can be a 100 times slower than a more specialized\n    version.\n\n    Nevertheless, for small window sizes, when no custom algorithm exists, or\n    as a prototyping and developing tool, this function can be a good solution.\n\n    Examples\n    --------\n    >>> x = np.arange(6)\n    >>> x.shape\n    (6,)\n    >>> v = sliding_window_view(x, 3)\n    >>> v.shape\n    (4, 3)\n    >>> v\n    array([[0, 1, 2],\n           [1, 2, 3],\n           [2, 3, 4],\n           [3, 4, 5]])\n\n    This also works in more dimensions, e.g.\n\n    >>> i, j = np.ogrid[:3, :4]\n    >>> x = 10*i + j\n    >>> x.shape\n    (3, 4)\n    >>> x\n    array([[ 0,  1,  2,  3],\n           [10, 11, 12, 13],\n           [20, 21, 22, 23]])\n    >>> shape = (2,2)\n    >>> v = sliding_window_view(x, shape)\n    >>> v.shape\n    (2, 3, 2, 2)\n    >>> v\n    array([[[[ 0,  1],\n             [10, 11]],\n            [[ 1,  2],\n             [11, 12]],\n            [[ 2,  3],\n             [12, 13]]],\n           [[[10, 11],\n             [20, 21]],\n            [[11, 12],\n             [21, 22]],\n            [[12, 13],\n             [22, 23]]]])\n\n    The axis can be specified explicitly:\n\n    >>> v = sliding_window_view(x, 3, 0)\n    >>> v.shape\n    (1, 4, 3)\n    >>> v\n    array([[[ 0, 10, 20],\n            [ 1, 11, 21],\n            [ 2, 12, 22],\n            [ 3, 13, 23]]])\n\n    The same axis can be used several times. In that case, every use reduces\n    the corresponding original dimension:\n\n    >>> v = sliding_window_view(x, (2, 3), (1, 1))\n    >>> v.shape\n    (3, 1, 2, 3)\n    >>> v\n    array([[[[ 0,  1,  2],\n             [ 1,  2,  3]]],\n           [[[10, 11, 12],\n             [11, 12, 13]]],\n           [[[20, 21, 22],\n             [21, 22, 23]]]])\n\n    Combining with stepped slicing (`::step`), this can be used to take sliding\n    views which skip elements:\n\n    >>> x = np.arange(7)\n    >>> sliding_window_view(x, 5)[:, ::2]\n    array([[0, 2, 4],\n           [1, 3, 5],\n           [2, 4, 6]])\n\n    or views which move by multiple elements\n\n    >>> x = np.arange(7)\n    >>> sliding_window_view(x, 3)[::2, :]\n    array([[0, 1, 2],\n           [2, 3, 4],\n           [4, 5, 6]])\n\n    A common application of `sliding_window_view` is the calculation of running\n    statistics. The simplest example is the\n    `moving average <https://en.wikipedia.org/wiki/Moving_average>`_:\n\n    >>> x = np.arange(6)\n    >>> x.shape\n    (6,)\n    >>> v = sliding_window_view(x, 3)\n    >>> v.shape\n    (4, 3)\n    >>> v\n    array([[0, 1, 2],\n           [1, 2, 3],\n           [2, 3, 4],\n           [3, 4, 5]])\n    >>> moving_average = v.mean(axis=-1)\n    >>> moving_average\n    array([1., 2., 3., 4.])\n\n    Note that a sliding window approach is often **not** optimal (see Notes).\n    '
    window_shape = tuple(window_shape) if np.iterable(window_shape) else (window_shape,)
    x = np.array(x, copy=False, subok=subok)
    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError('`window_shape` cannot contain negative values')
    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(f'Since axis is `None`, must provide window_shape for all dimensions of `x`; got {len(window_shape)} window_shape elements and `x.ndim` is {x.ndim}.')
    else:
        axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
        if len(window_shape) != len(axis):
            raise ValueError(f'Must provide matching length window_shape and axis; got {len(window_shape)} window_shape elements and {len(axis)} axes elements.')
    out_strides = x.strides + tuple((x.strides[ax] for ax in axis))
    x_shape_trimmed = list(x.shape)
    for (ax, dim) in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError('window shape cannot be larger than input array shape')
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape
    return as_strided(x, strides=out_strides, shape=out_shape, subok=subok, writeable=writeable)

def _broadcast_to(array, shape, subok, readonly):
    if False:
        while True:
            i = 10
    shape = tuple(shape) if np.iterable(shape) else (shape,)
    array = np.array(array, copy=False, subok=subok)
    if not shape and array.shape:
        raise ValueError('cannot broadcast a non-scalar to a scalar array')
    if any((size < 0 for size in shape)):
        raise ValueError('all elements of broadcast shape must be non-negative')
    extras = []
    it = np.nditer((array,), flags=['multi_index', 'refs_ok', 'zerosize_ok'] + extras, op_flags=['readonly'], itershape=shape, order='C')
    with it:
        broadcast = it.itviews[0]
    result = _maybe_view_as_subclass(array, broadcast)
    if not readonly and array.flags._writeable_no_warn:
        result.flags.writeable = True
        result.flags._warn_on_write = True
    return result

def _broadcast_to_dispatcher(array, shape, subok=None):
    if False:
        for i in range(10):
            print('nop')
    return (array,)

@array_function_dispatch(_broadcast_to_dispatcher, module='numpy')
def broadcast_to(array, shape, subok=False):
    if False:
        while True:
            i = 10
    "Broadcast an array to a new shape.\n\n    Parameters\n    ----------\n    array : array_like\n        The array to broadcast.\n    shape : tuple or int\n        The shape of the desired array. A single integer ``i`` is interpreted\n        as ``(i,)``.\n    subok : bool, optional\n        If True, then sub-classes will be passed-through, otherwise\n        the returned array will be forced to be a base-class array (default).\n\n    Returns\n    -------\n    broadcast : array\n        A readonly view on the original array with the given shape. It is\n        typically not contiguous. Furthermore, more than one element of a\n        broadcasted array may refer to a single memory location.\n\n    Raises\n    ------\n    ValueError\n        If the array is not compatible with the new shape according to NumPy's\n        broadcasting rules.\n\n    See Also\n    --------\n    broadcast\n    broadcast_arrays\n    broadcast_shapes\n\n    Notes\n    -----\n    .. versionadded:: 1.10.0\n\n    Examples\n    --------\n    >>> x = np.array([1, 2, 3])\n    >>> np.broadcast_to(x, (3, 3))\n    array([[1, 2, 3],\n           [1, 2, 3],\n           [1, 2, 3]])\n    "
    return _broadcast_to(array, shape, subok=subok, readonly=True)

def _broadcast_shape(*args):
    if False:
        while True:
            i = 10
    'Returns the shape of the arrays that would result from broadcasting the\n    supplied arrays against each other.\n    '
    b = np.broadcast(*args[:32])
    for pos in range(32, len(args), 31):
        b = broadcast_to(0, b.shape)
        b = np.broadcast(b, *args[pos:pos + 31])
    return b.shape

@set_module('numpy')
def broadcast_shapes(*args):
    if False:
        while True:
            i = 10
    "\n    Broadcast the input shapes into a single shape.\n\n    :ref:`Learn more about broadcasting here <basics.broadcasting>`.\n\n    .. versionadded:: 1.20.0\n\n    Parameters\n    ----------\n    *args : tuples of ints, or ints\n        The shapes to be broadcast against each other.\n\n    Returns\n    -------\n    tuple\n        Broadcasted shape.\n\n    Raises\n    ------\n    ValueError\n        If the shapes are not compatible and cannot be broadcast according\n        to NumPy's broadcasting rules.\n\n    See Also\n    --------\n    broadcast\n    broadcast_arrays\n    broadcast_to\n\n    Examples\n    --------\n    >>> np.broadcast_shapes((1, 2), (3, 1), (3, 2))\n    (3, 2)\n\n    >>> np.broadcast_shapes((6, 7), (5, 6, 1), (7,), (5, 1, 7))\n    (5, 6, 7)\n    "
    arrays = [np.empty(x, dtype=[]) for x in args]
    return _broadcast_shape(*arrays)

def _broadcast_arrays_dispatcher(*args, subok=None):
    if False:
        while True:
            i = 10
    return args

@array_function_dispatch(_broadcast_arrays_dispatcher, module='numpy')
def broadcast_arrays(*args, subok=False):
    if False:
        print('Hello World!')
    '\n    Broadcast any number of arrays against each other.\n\n    Parameters\n    ----------\n    *args : array_likes\n        The arrays to broadcast.\n\n    subok : bool, optional\n        If True, then sub-classes will be passed-through, otherwise\n        the returned arrays will be forced to be a base-class array (default).\n\n    Returns\n    -------\n    broadcasted : list of arrays\n        These arrays are views on the original arrays.  They are typically\n        not contiguous.  Furthermore, more than one element of a\n        broadcasted array may refer to a single memory location. If you need\n        to write to the arrays, make copies first. While you can set the\n        ``writable`` flag True, writing to a single output value may end up\n        changing more than one location in the output array.\n\n        .. deprecated:: 1.17\n            The output is currently marked so that if written to, a deprecation\n            warning will be emitted. A future version will set the\n            ``writable`` flag False so writing to it will raise an error.\n\n    See Also\n    --------\n    broadcast\n    broadcast_to\n    broadcast_shapes\n\n    Examples\n    --------\n    >>> x = np.array([[1,2,3]])\n    >>> y = np.array([[4],[5]])\n    >>> np.broadcast_arrays(x, y)\n    [array([[1, 2, 3],\n           [1, 2, 3]]), array([[4, 4, 4],\n           [5, 5, 5]])]\n\n    Here is a useful idiom for getting contiguous copies instead of\n    non-contiguous views.\n\n    >>> [np.array(a) for a in np.broadcast_arrays(x, y)]\n    [array([[1, 2, 3],\n           [1, 2, 3]]), array([[4, 4, 4],\n           [5, 5, 5]])]\n\n    '
    args = [np.array(_m, copy=False, subok=subok) for _m in args]
    shape = _broadcast_shape(*args)
    if all((array.shape == shape for array in args)):
        return args
    return [_broadcast_to(array, shape, subok=subok, readonly=False) for array in args]