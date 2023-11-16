"""Module houses array shaping methods for Modin's NumPy API."""
import numpy
from modin.error_message import ErrorMessage
from .arr import array
from .utils import try_convert_from_interoperable_type

def ravel(a, order='C'):
    if False:
        for i in range(10):
            print('nop')
    a = try_convert_from_interoperable_type(a)
    if not isinstance(a, array):
        ErrorMessage.bad_type_for_numpy_op('ravel', type(a))
        return numpy.ravel(a, order=order)
    if order != 'C':
        ErrorMessage.single_warning("Array order besides 'C' is not currently supported in Modin. Defaulting to 'C' order.")
    return a.flatten(order)

def shape(a):
    if False:
        while True:
            i = 10
    a = try_convert_from_interoperable_type(a)
    if not isinstance(a, array):
        ErrorMessage.bad_type_for_numpy_op('shape', type(a))
        return numpy.shape(a)
    return a.shape

def transpose(a, axes=None):
    if False:
        return 10
    a = try_convert_from_interoperable_type(a)
    if not isinstance(a, array):
        ErrorMessage.bad_type_for_numpy_op('transpose', type(a))
        return numpy.transpose(a, axes=axes)
    if axes is not None:
        raise NotImplementedError('Modin does not support arrays higher than 2-dimensions. Please use `transpose` with `axis=None` on a 2-dimensional or lower object.')
    return a.transpose()

def split(arr, indices, axis=0):
    if False:
        while True:
            i = 10
    arr = try_convert_from_interoperable_type(arr)
    if not isinstance(arr, array):
        ErrorMessage.bad_type_for_numpy_op('split', type(arr))
        return numpy.split(arr, indices, axis=axis)
    return arr.split(indices, axis)

def hstack(tup, dtype=None, casting='same_kind'):
    if False:
        i = 10
        return i + 15
    a = try_convert_from_interoperable_type(tup[0])
    if not isinstance(a, array):
        ErrorMessage.bad_type_for_numpy_op('hstack', type(a))
        return numpy.hstack(tup, dtype=dtype, casting=casting)
    return a.hstack(tup[1:], dtype, casting)

def append(arr, values, axis=None):
    if False:
        while True:
            i = 10
    arr = try_convert_from_interoperable_type(arr)
    if not isinstance(arr, array):
        ErrorMessage.bad_type_for_numpy_op('append', type(arr))
        return numpy.append(arr, values, axis=axis)
    return arr.append(values, axis)