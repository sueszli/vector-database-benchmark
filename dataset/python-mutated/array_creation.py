"""Module houses array creation methods for Modin's NumPy API."""
import numpy
from modin.error_message import ErrorMessage
from .arr import array

def _create_array(dtype, shape, order, subok, numpy_method):
    if False:
        while True:
            i = 10
    if order not in ['K', 'C']:
        ErrorMessage.single_warning("Array order besides 'C' is not currently supported in Modin. Defaulting to 'C' order.")
    if not subok:
        ErrorMessage.single_warning('Subclassing types is not currently supported in Modin. Defaulting to the same base dtype.')
    ErrorMessage.single_warning(f'np.{numpy_method}_like defaulting to NumPy.')
    return array(getattr(numpy, numpy_method)(shape, dtype=dtype))

def zeros_like(a, dtype=None, order='K', subok=True, shape=None):
    if False:
        while True:
            i = 10
    if not isinstance(a, array):
        ErrorMessage.bad_type_for_numpy_op('zeros_like', type(a))
        return numpy.zeros_like(a, dtype=dtype, order=order, subok=subok, shape=shape)
    dtype = a.dtype if dtype is None else dtype
    shape = a.shape if shape is None else shape
    return _create_array(dtype, shape, order, subok, 'zeros')

def ones_like(a, dtype=None, order='K', subok=True, shape=None):
    if False:
        while True:
            i = 10
    if not isinstance(a, array):
        ErrorMessage.bad_type_for_numpy_op('ones_like', type(a))
        return numpy.ones_like(a, dtype=dtype, order=order, subok=subok, shape=shape)
    dtype = a.dtype if dtype is None else dtype
    shape = a.shape if shape is None else shape
    return _create_array(dtype, shape, order, subok, 'ones')

def tri(N, M=None, k=0, dtype=float, like=None):
    if False:
        i = 10
        return i + 15
    if like is not None:
        ErrorMessage.single_warning('Modin NumPy does not support the `like` argument for np.tri. Defaulting to `like=None`.')
    ErrorMessage.single_warning('np.tri defaulting to NumPy.')
    return array(numpy.tri(N, M=M, k=k, dtype=dtype))