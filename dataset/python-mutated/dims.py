import cupy
from cupy import _core
import cupy._core._routines_manipulation as _manipulation
_atleast_nd_shape_map = {(1, 0): lambda shape: (1,), (2, 0): lambda shape: (1, 1), (2, 1): lambda shape: (1,) + shape, (3, 0): lambda shape: (1, 1, 1), (3, 1): lambda shape: (1,) + shape + (1,), (3, 2): lambda shape: shape + (1,)}

def _atleast_nd_helper(n, arys):
    if False:
        i = 10
        return i + 15
    'Helper function for atleast_nd functions.'
    res = []
    for a in arys:
        a = cupy.asarray(a)
        if a.ndim < n:
            new_shape = _atleast_nd_shape_map[n, a.ndim](a.shape)
            a = a.reshape(*new_shape)
        res.append(a)
    if len(res) == 1:
        (res,) = res
    return res

def atleast_1d(*arys):
    if False:
        for i in range(10):
            print('nop')
    'Converts arrays to arrays with dimensions >= 1.\n\n    Args:\n        arys (tuple of arrays): Arrays to be converted. All arguments must be\n            :class:`cupy.ndarray` objects. Only zero-dimensional array is\n            affected.\n\n    Returns:\n        If there are only one input, then it returns its converted version.\n        Otherwise, it returns a list of converted arrays.\n\n    .. seealso:: :func:`numpy.atleast_1d`\n\n    '
    return _atleast_nd_helper(1, arys)

def atleast_2d(*arys):
    if False:
        i = 10
        return i + 15
    'Converts arrays to arrays with dimensions >= 2.\n\n    If an input array has dimensions less than two, then this function inserts\n    new axes at the head of dimensions to make it have two dimensions.\n\n    Args:\n        arys (tuple of arrays): Arrays to be converted. All arguments must be\n            :class:`cupy.ndarray` objects.\n\n    Returns:\n        If there are only one input, then it returns its converted version.\n        Otherwise, it returns a list of converted arrays.\n\n    .. seealso:: :func:`numpy.atleast_2d`\n\n    '
    return _atleast_nd_helper(2, arys)

def atleast_3d(*arys):
    if False:
        while True:
            i = 10
    'Converts arrays to arrays with dimensions >= 3.\n\n    If an input array has dimensions less than three, then this function\n    inserts new axes to make it have three dimensions. The place of the new\n    axes are following:\n\n    - If its shape is ``()``, then the shape of output is ``(1, 1, 1)``.\n    - If its shape is ``(N,)``, then the shape of output is ``(1, N, 1)``.\n    - If its shape is ``(M, N)``, then the shape of output is ``(M, N, 1)``.\n    - Otherwise, the output is the input array itself.\n\n    Args:\n        arys (tuple of arrays): Arrays to be converted. All arguments must be\n            :class:`cupy.ndarray` objects.\n\n    Returns:\n        If there are only one input, then it returns its converted version.\n        Otherwise, it returns a list of converted arrays.\n\n    .. seealso:: :func:`numpy.atleast_3d`\n\n    '
    return _atleast_nd_helper(3, arys)
broadcast = _core.broadcast

def broadcast_arrays(*args):
    if False:
        for i in range(10):
            print('nop')
    'Broadcasts given arrays.\n\n    Args:\n        args (tuple of arrays): Arrays to broadcast for each other.\n\n    Returns:\n        list: A list of broadcasted arrays.\n\n    .. seealso:: :func:`numpy.broadcast_arrays`\n\n    '
    return list(broadcast(*args).values)

def broadcast_to(array, shape):
    if False:
        i = 10
        return i + 15
    'Broadcast an array to a given shape.\n\n    Args:\n        array (cupy.ndarray): Array to broadcast.\n        shape (tuple of int): The shape of the desired array.\n\n    Returns:\n        cupy.ndarray: Broadcasted view.\n\n    .. seealso:: :func:`numpy.broadcast_to`\n\n    '
    return _core.broadcast_to(array, shape)

def expand_dims(a, axis):
    if False:
        for i in range(10):
            print('nop')
    'Expands given arrays.\n\n    Args:\n        a (cupy.ndarray): Array to be expanded.\n        axis (int): Position where new axis is to be inserted.\n\n    Returns:\n        cupy.ndarray: The number of dimensions is one greater than that of\n        the input array.\n\n    .. seealso:: :func:`numpy.expand_dims`\n\n    '
    if type(axis) not in (tuple, list):
        axis = (axis,)
    return _manipulation._expand_dims(a, axis)

def squeeze(a, axis=None):
    if False:
        while True:
            i = 10
    'Removes size-one axes from the shape of an array.\n\n    Args:\n        a (cupy.ndarray): Array to be reshaped.\n        axis (int or tuple of ints): Axes to be removed. This function removes\n            all size-one axes by default. If one of the specified axes is not\n            of size one, an exception is raised.\n\n    Returns:\n        cupy.ndarray: An array without (specified) size-one axes.\n\n    .. seealso:: :func:`numpy.squeeze`\n\n    '
    return a.squeeze(axis)