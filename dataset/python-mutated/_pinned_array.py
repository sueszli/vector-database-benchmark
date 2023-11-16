import numpy
from cupy import cuda
from cupy._creation.basic import _new_like_order_and_strides
from cupy._core import internal

def _update_shape(a, shape):
    if False:
        while True:
            i = 10
    if shape is None and a is not None:
        shape = a.shape
    elif isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)
    return shape

def empty_pinned(shape, dtype=float, order='C'):
    if False:
        print('Hello World!')
    "Returns a new, uninitialized NumPy array with the given shape\n    and dtype.\n\n    This is a convenience function which is just :func:`numpy.empty`,\n    except that the underlying memory is pinned/pagelocked.\n\n    Args:\n        shape (int or tuple of ints): Dimensionalities of the array.\n        dtype: Data type specifier.\n        order ({'C', 'F'}): Row-major (C-style) or column-major\n            (Fortran-style) order.\n\n    Returns:\n        numpy.ndarray: A new array with elements not initialized.\n\n    .. seealso:: :func:`numpy.empty`\n\n    "
    shape = _update_shape(None, shape)
    nbytes = internal.prod(shape) * numpy.dtype(dtype).itemsize
    mem = cuda.alloc_pinned_memory(nbytes)
    out = numpy.ndarray(shape, dtype=dtype, buffer=mem, order=order)
    return out

def empty_like_pinned(a, dtype=None, order='K', subok=None, shape=None):
    if False:
        print('Hello World!')
    "Returns a new, uninitialized NumPy array with the same shape and dtype\n    as those of the given array.\n\n    This is a convenience function which is just :func:`numpy.empty_like`,\n    except that the underlying memory is pinned/pagelocked.\n\n    This function currently does not support ``subok`` option.\n\n    Args:\n        a (numpy.ndarray or cupy.ndarray): Base array.\n        dtype: Data type specifier. The data type of ``a`` is used by default.\n        order ({'C', 'F', 'A', or 'K'}): Overrides the memory layout of the\n            result. ``'C'`` means C-order, ``'F'`` means F-order, ``'A'`` means\n            ``'F'`` if ``a`` is Fortran contiguous, ``'C'`` otherwise.\n            ``'K'`` means match the layout of ``a`` as closely as possible.\n        subok: Not supported yet, must be None.\n        shape (int or tuple of ints): Overrides the shape of the result. If\n            ``order='K'`` and the number of dimensions is unchanged, will try\n            to keep order, otherwise, ``order='C'`` is implied.\n\n    Returns:\n        numpy.ndarray: A new array with same shape and dtype of ``a`` with\n        elements not initialized.\n\n    .. seealso:: :func:`numpy.empty_like`\n\n    "
    if subok is not None:
        raise TypeError('subok is not supported yet')
    if dtype is None:
        dtype = a.dtype
    shape = _update_shape(a, shape)
    (order, strides, _) = _new_like_order_and_strides(a, dtype, order, shape, get_memptr=False)
    nbytes = internal.prod(shape) * numpy.dtype(dtype).itemsize
    mem = cuda.alloc_pinned_memory(nbytes)
    out = numpy.ndarray(shape, dtype=dtype, buffer=mem, strides=strides, order=order)
    return out

def zeros_pinned(shape, dtype=float, order='C'):
    if False:
        print('Hello World!')
    "Returns a new, zero-initialized NumPy array with the given shape\n    and dtype.\n\n    This is a convenience function which is just :func:`numpy.zeros`,\n    except that the underlying memory is pinned/pagelocked.\n\n    Args:\n        shape (int or tuple of ints): Dimensionalities of the array.\n        dtype: Data type specifier.\n        order ({'C', 'F'}): Row-major (C-style) or column-major\n            (Fortran-style) order.\n\n    Returns:\n        numpy.ndarray: An array filled with zeros.\n\n    .. seealso:: :func:`numpy.zeros`\n\n    "
    out = empty_pinned(shape, dtype, order)
    numpy.copyto(out, 0, casting='unsafe')
    return out

def zeros_like_pinned(a, dtype=None, order='K', subok=None, shape=None):
    if False:
        for i in range(10):
            print('nop')
    "Returns a new, zero-initialized NumPy array with the same shape and dtype\n    as those of the given array.\n\n    This is a convenience function which is just :func:`numpy.zeros_like`,\n    except that the underlying memory is pinned/pagelocked.\n\n    This function currently does not support ``subok`` option.\n\n    Args:\n        a (numpy.ndarray or cupy.ndarray): Base array.\n        dtype: Data type specifier. The dtype of ``a`` is used by default.\n        order ({'C', 'F', 'A', or 'K'}): Overrides the memory layout of the\n            result. ``'C'`` means C-order, ``'F'`` means F-order, ``'A'`` means\n            ``'F'`` if ``a`` is Fortran contiguous, ``'C'`` otherwise.\n            ``'K'`` means match the layout of ``a`` as closely as possible.\n        subok: Not supported yet, must be None.\n        shape (int or tuple of ints): Overrides the shape of the result. If\n            ``order='K'`` and the number of dimensions is unchanged, will try\n            to keep order, otherwise, ``order='C'`` is implied.\n\n    Returns:\n        numpy.ndarray: An array filled with zeros.\n\n    .. seealso:: :func:`numpy.zeros_like`\n\n    "
    out = empty_like_pinned(a, dtype, order, subok, shape)
    numpy.copyto(out, 0, casting='unsafe')
    return out