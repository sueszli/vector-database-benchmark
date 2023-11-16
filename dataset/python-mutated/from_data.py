import numpy
from cupy import _core
from cupy._core import fusion

def array(obj, dtype=None, copy=True, order='K', subok=False, ndmin=0):
    if False:
        while True:
            i = 10
    "Creates an array on the current device.\n\n    This function currently does not support the ``subok`` option.\n\n    Args:\n        obj: :class:`cupy.ndarray` object or any other object that can be\n            passed to :func:`numpy.array`.\n        dtype: Data type specifier.\n        copy (bool): If ``False``, this function returns ``obj`` if possible.\n            Otherwise this function always returns a new array.\n        order ({'C', 'F', 'A', 'K'}): Row-major (C-style) or column-major\n            (Fortran-style) order.\n            When ``order`` is ``'A'``, it uses ``'F'`` if ``a`` is column-major\n            and uses ``'C'`` otherwise.\n            And when ``order`` is ``'K'``, it keeps strides as closely as\n            possible.\n            If ``obj`` is :class:`numpy.ndarray`, the function returns ``'C'``\n            or ``'F'`` order array.\n        subok (bool): If ``True``, then sub-classes will be passed-through,\n            otherwise the returned array will be forced to be a base-class\n            array (default).\n        ndmin (int): Minimum number of dimensions. Ones are inserted to the\n            head of the shape if needed.\n\n    Returns:\n        cupy.ndarray: An array on the current device.\n\n    .. note::\n       This method currently does not support ``subok`` argument.\n\n    .. note::\n       If ``obj`` is an `numpy.ndarray` instance that contains big-endian data,\n       this function automatically swaps its byte order to little-endian,\n       which is the NVIDIA and AMD GPU architecture's native use.\n\n    .. seealso:: :func:`numpy.array`\n\n    "
    return _core.array(obj, dtype, copy, order, subok, ndmin)

def asarray(a, dtype=None, order=None):
    if False:
        i = 10
        return i + 15
    "Converts an object to array.\n\n    This is equivalent to ``array(a, dtype, copy=False, order=order)``.\n\n    Args:\n        a: The source object.\n        dtype: Data type specifier. It is inferred from the input by default.\n        order ({'C', 'F', 'A', 'K'}):\n            Whether to use row-major (C-style) or column-major (Fortran-style)\n            memory representation. Defaults to ``'K'``. ``order`` is ignored\n            for objects that are not :class:`cupy.ndarray`, but have the\n            ``__cuda_array_interface__`` attribute.\n\n    Returns:\n        cupy.ndarray: An array on the current device. If ``a`` is already on\n        the device, no copy is performed.\n\n    .. note::\n       If ``a`` is an `numpy.ndarray` instance that contains big-endian data,\n       this function automatically swaps its byte order to little-endian,\n       which is the NVIDIA and AMD GPU architecture's native use.\n\n    .. seealso:: :func:`numpy.asarray`\n\n    "
    return _core.array(a, dtype, False, order)

def asanyarray(a, dtype=None, order=None):
    if False:
        return 10
    'Converts an object to array.\n\n    This is currently equivalent to :func:`cupy.asarray`, since there is no\n    subclass of :class:`cupy.ndarray` in CuPy. Note that the original\n    :func:`numpy.asanyarray` returns the input array as is if it is an instance\n    of a subtype of :class:`numpy.ndarray`.\n\n    .. seealso:: :func:`cupy.asarray`, :func:`numpy.asanyarray`\n\n    '
    return _core.array(a, dtype, False, order)

def ascontiguousarray(a, dtype=None):
    if False:
        return 10
    'Returns a C-contiguous array.\n\n    Args:\n        a (cupy.ndarray): Source array.\n        dtype: Data type specifier.\n\n    Returns:\n        cupy.ndarray: If no copy is required, it returns ``a``. Otherwise, it\n        returns a copy of ``a``.\n\n    .. seealso:: :func:`numpy.ascontiguousarray`\n\n    '
    return _core.ascontiguousarray(a, dtype)

def copy(a, order='K'):
    if False:
        for i in range(10):
            print('nop')
    "Creates a copy of a given array on the current device.\n\n    This function allocates the new array on the current device. If the given\n    array is allocated on the different device, then this function tries to\n    copy the contents over the devices.\n\n    Args:\n        a (cupy.ndarray): The source array.\n        order ({'C', 'F', 'A', 'K'}): Row-major (C-style) or column-major\n            (Fortran-style) order.\n            When ``order`` is ``'A'``, it uses ``'F'`` if ``a`` is column-major\n            and uses ``'C'`` otherwise.\n            And when ``order`` is ``'K'``, it keeps strides as closely as\n            possible.\n\n    Returns:\n        cupy.ndarray: The copy of ``a`` on the current device.\n\n    .. seealso:: :func:`numpy.copy`, :meth:`cupy.ndarray.copy`\n\n    "
    if fusion._is_fusing():
        if order != 'K':
            raise NotImplementedError('cupy.copy does not support `order` in fusion yet.')
        return fusion._call_ufunc(_core.elementwise_copy, a)
    return a.copy(order=order)

def frombuffer(*args, **kwargs):
    if False:
        print('Hello World!')
    "Interpret a buffer as a 1-dimensional array.\n\n    .. note::\n        Uses NumPy's ``frombuffer`` and coerces the result to a CuPy array.\n\n    .. seealso:: :func:`numpy.frombuffer`\n\n    "
    return asarray(numpy.frombuffer(*args, **kwargs))

def fromfile(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "Reads an array from a file.\n\n    .. note::\n        Uses NumPy's ``fromfile`` and coerces the result to a CuPy array.\n\n    .. note::\n       If you let NumPy's ``fromfile`` read the file in big-endian, CuPy\n       automatically swaps its byte order to little-endian, which is the NVIDIA\n       and AMD GPU architecture's native use.\n\n    .. seealso:: :func:`numpy.fromfile`\n\n    "
    return asarray(numpy.fromfile(*args, **kwargs))

def fromfunction(*args, **kwargs):
    if False:
        return 10
    "Construct an array by executing a function over each coordinate.\n\n    .. note::\n        Uses NumPy's ``fromfunction`` and coerces the result to a CuPy array.\n\n    .. seealso:: :func:`numpy.fromfunction`\n    "
    return asarray(numpy.fromfunction(*args, **kwargs))

def fromiter(*args, **kwargs):
    if False:
        print('Hello World!')
    "Create a new 1-dimensional array from an iterable object.\n\n    .. note::\n        Uses NumPy's ``fromiter`` and coerces the result to a CuPy array.\n\n    .. seealso:: :func:`numpy.fromiter`\n    "
    return asarray(numpy.fromiter(*args, **kwargs))

def fromstring(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "A new 1-D array initialized from text data in a string.\n\n    .. note::\n        Uses NumPy's ``fromstring`` and coerces the result to a CuPy array.\n\n    .. seealso:: :func:`numpy.fromstring`\n    "
    return asarray(numpy.fromstring(*args, **kwargs))

def loadtxt(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "Load data from a text file.\n\n    .. note::\n        Uses NumPy's ``loadtxt`` and coerces the result to a CuPy array.\n\n    .. seealso:: :func:`numpy.loadtxt`\n    "
    return asarray(numpy.loadtxt(*args, **kwargs))

def genfromtxt(*args, **kwargs):
    if False:
        return 10
    "Load data from text file, with missing values handled as specified.\n\n    .. note::\n        Uses NumPy's ``genfromtxt`` and coerces the result to a CuPy array.\n\n    .. seealso:: :func:`numpy.genfromtxt`\n    "
    return asarray(numpy.genfromtxt(*args, **kwargs))