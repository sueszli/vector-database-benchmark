import cupy
from cupy import _core

def tile(A, reps):
    if False:
        i = 10
        return i + 15
    'Construct an array by repeating A the number of times given by reps.\n\n    Args:\n        A (cupy.ndarray): Array to transform.\n        reps (int or tuple): The number of repeats.\n\n    Returns:\n        cupy.ndarray: Transformed array with repeats.\n\n    .. seealso:: :func:`numpy.tile`\n\n    '
    try:
        tup = tuple(reps)
    except TypeError:
        tup = (reps,)
    d = len(tup)
    if tup.count(1) == len(tup) and isinstance(A, cupy.ndarray):
        return cupy.array(A, copy=True, ndmin=d)
    else:
        c = cupy.array(A, copy=False, ndmin=d)
    if d < c.ndim:
        tup = (1,) * (c.ndim - d) + tup
    shape_out = tuple((s * t for (s, t) in zip(c.shape, tup)))
    if c.size == 0:
        return cupy.empty(shape_out, dtype=c.dtype)
    c_shape = []
    ret_shape = []
    for (dim_in, nrep) in zip(c.shape, tup):
        if nrep == 1:
            c_shape.append(dim_in)
            ret_shape.append(dim_in)
        elif dim_in == 1:
            c_shape.append(dim_in)
            ret_shape.append(nrep)
        else:
            c_shape.append(1)
            c_shape.append(dim_in)
            ret_shape.append(nrep)
            ret_shape.append(dim_in)
    ret = cupy.empty(ret_shape, dtype=c.dtype)
    if ret.size:
        _core.elementwise_copy(c.reshape(c_shape), ret)
    return ret.reshape(shape_out)

def repeat(a, repeats, axis=None):
    if False:
        for i in range(10):
            print('nop')
    'Repeat arrays along an axis.\n\n    Args:\n        a (cupy.ndarray): Array to transform.\n        repeats (int, list or tuple): The number of repeats.\n        axis (int): The axis to repeat.\n\n    Returns:\n        cupy.ndarray: Transformed array with repeats.\n\n    .. seealso:: :func:`numpy.repeat`\n\n    '
    return a.repeat(repeats, axis)