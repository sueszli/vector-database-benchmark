from __future__ import annotations
import numpy as np
_HANDLED_CHUNK_TYPES = [np.ndarray, np.ma.MaskedArray]

def register_chunk_type(type):
    if False:
        while True:
            i = 10
    'Register the given type as a valid chunk and downcast array type\n\n    Parameters\n    ----------\n    type : type\n        Duck array type to be registered as a type Dask can safely wrap as a chunk and\n        to which Dask does not defer in arithmetic operations and NumPy\n        functions/ufuncs.\n\n    Notes\n    -----\n    A :py:class:`dask.array.Array` can contain any sufficiently "NumPy-like" array in\n    its chunks. These are also referred to as "duck arrays" since they match the most\n    important parts of NumPy\'s array API, and so, behave the same way when relying on\n    duck typing.\n\n    However, for multiple duck array types to interoperate properly, they need to\n    properly defer to each other in arithmetic operations and NumPy functions/ufuncs\n    according to a well-defined type casting hierarchy (\n    `see NEP 13 <https://numpy.org/neps/nep-0013-ufunc-overrides.html#type-casting-hierarchy>`__\n    ). In an effort to maintain this hierarchy, Dask defers to all other duck array\n    types except those in its internal registry. By default, this registry contains\n\n    * :py:class:`numpy.ndarray`\n    * :py:class:`numpy.ma.MaskedArray`\n    * :py:class:`cupy.ndarray`\n    * :py:class:`sparse.SparseArray`\n    * :py:class:`scipy.sparse.spmatrix`\n\n    This function exists to append any other types to this registry. If a type is not\n    in this registry, and yet is a downcast type (it comes below\n    :py:class:`dask.array.Array` in the type casting hierarchy), a ``TypeError`` will\n    be raised due to all operand types returning ``NotImplemented``.\n\n    Examples\n    --------\n    Using a mock ``FlaggedArray`` class as an example chunk type unknown to Dask with\n    minimal duck array API:\n\n    >>> import numpy.lib.mixins\n    >>> class FlaggedArray(numpy.lib.mixins.NDArrayOperatorsMixin):\n    ...     def __init__(self, a, flag=False):\n    ...         self.a = a\n    ...         self.flag = flag\n    ...     def __repr__(self):\n    ...         return f"Flag: {self.flag}, Array: " + repr(self.a)\n    ...     def __array__(self):\n    ...         return np.asarray(self.a)\n    ...     def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):\n    ...         if method == \'__call__\':\n    ...             downcast_inputs = []\n    ...             flag = False\n    ...             for input in inputs:\n    ...                 if isinstance(input, self.__class__):\n    ...                     flag = flag or input.flag\n    ...                     downcast_inputs.append(input.a)\n    ...                 elif isinstance(input, np.ndarray):\n    ...                     downcast_inputs.append(input)\n    ...                 else:\n    ...                     return NotImplemented\n    ...             return self.__class__(ufunc(*downcast_inputs, **kwargs), flag)\n    ...         else:\n    ...             return NotImplemented\n    ...     @property\n    ...     def shape(self):\n    ...         return self.a.shape\n    ...     @property\n    ...     def ndim(self):\n    ...         return self.a.ndim\n    ...     @property\n    ...     def dtype(self):\n    ...         return self.a.dtype\n    ...     def __getitem__(self, key):\n    ...         return type(self)(self.a[key], self.flag)\n    ...     def __setitem__(self, key, value):\n    ...         self.a[key] = value\n\n    Before registering ``FlaggedArray``, both types will attempt to defer to the\n    other:\n\n    >>> import dask.array as da\n    >>> da.ones(5) - FlaggedArray(np.ones(5), True)\n    Traceback (most recent call last):\n    ...\n    TypeError: operand type(s) all returned NotImplemented ...\n\n    However, once registered, Dask will be able to handle operations with this new\n    type:\n\n    >>> da.register_chunk_type(FlaggedArray)\n    >>> x = da.ones(5) - FlaggedArray(np.ones(5), True)\n    >>> x\n    dask.array<sub, shape=(5,), dtype=float64, chunksize=(5,), chunktype=dask.FlaggedArray>\n    >>> x.compute()\n    Flag: True, Array: array([0., 0., 0., 0., 0.])\n    '
    _HANDLED_CHUNK_TYPES.append(type)
try:
    import cupy
    register_chunk_type(cupy.ndarray)
except ImportError:
    pass
try:
    from cupyx.scipy.sparse import spmatrix
    register_chunk_type(spmatrix)
except ImportError:
    pass
try:
    import sparse
    register_chunk_type(sparse.SparseArray)
except ImportError:
    pass
try:
    import scipy.sparse
    register_chunk_type(scipy.sparse.spmatrix)
except ImportError:
    pass

def is_valid_chunk_type(type):
    if False:
        i = 10
        return i + 15
    'Check if given type is a valid chunk and downcast array type'
    try:
        return type in _HANDLED_CHUNK_TYPES or issubclass(type, tuple(_HANDLED_CHUNK_TYPES))
    except TypeError:
        return False

def is_valid_array_chunk(array):
    if False:
        return 10
    'Check if given array is of a valid type to operate with'
    return array is None or isinstance(array, tuple(_HANDLED_CHUNK_TYPES))