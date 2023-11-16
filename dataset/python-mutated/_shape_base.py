from numpy.lib import index_tricks
import cupy
from cupy._core import internal

def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    'Apply a function to 1-D slices along the given axis.\n\n    Args:\n        func1d (function (M,) -> (Nj...)): This function should accept 1-D\n            arrays. It is applied to 1-D slices of ``arr`` along the specified\n            axis. It must return a 1-D ``cupy.ndarray``.\n        axis (integer): Axis along which ``arr`` is sliced.\n        arr (cupy.ndarray (Ni..., M, Nk...)): Input array.\n        args: Additional arguments for ``func1d``.\n        kwargs: Additional keyword arguments for ``func1d``.\n\n    Returns:\n        cupy.ndarray: The output array. The shape of ``out`` is identical to\n        the shape of ``arr``, except along the ``axis`` dimension. This\n        axis is removed, and replaced with new dimensions equal to the\n        shape of the return value of ``func1d``. So if ``func1d`` returns a\n        scalar ``out`` will have one fewer dimensions than ``arr``.\n\n    .. seealso:: :func:`numpy.apply_along_axis`\n    '
    ndim = arr.ndim
    axis = internal._normalize_axis_index(axis, ndim)
    inarr_view = cupy.moveaxis(arr, axis, -1)
    inds = index_tricks.ndindex(inarr_view.shape[:-1])
    inds = (ind + (Ellipsis,) for ind in inds)
    try:
        ind0 = next(inds)
    except StopIteration:
        raise ValueError('Cannot apply_along_axis when any iteration dimensions are 0')
    res = func1d(inarr_view[ind0], *args, **kwargs)
    res = cupy.asarray(res)
    buff = cupy.empty(inarr_view.shape[:-1] + res.shape, res.dtype)
    buff[ind0] = res
    for ind in inds:
        out = func1d(inarr_view[ind], *args, **kwargs)
        buff[ind] = cupy.asarray(out)
    for i in range(res.ndim):
        buff = cupy.moveaxis(buff, -1, axis)
    return buff