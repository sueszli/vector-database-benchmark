import ivy
import numbers
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back, from_zero_dim_arrays_to_scalar, handle_numpy_out
import ivy.functional.frontends.numpy as np_frontend

@handle_numpy_out
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def all(a, axis=None, out=None, keepdims=False, *, where=None):
    if False:
        i = 10
        return i + 15
    axis = tuple(axis) if isinstance(axis, list) else axis
    if where is not None:
        a = ivy.where(where, a, True)
    ret = ivy.all(a, axis=axis, keepdims=keepdims, out=out)
    return ret

@handle_numpy_out
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def any(a, axis=None, out=None, keepdims=False, *, where=None):
    if False:
        while True:
            i = 10
    axis = tuple(axis) if isinstance(axis, list) else axis
    if where is not None:
        a = ivy.where(where, a, False)
    ret = ivy.any(a, axis=axis, keepdims=keepdims, out=out)
    return ret

@to_ivy_arrays_and_back
def iscomplex(x):
    if False:
        while True:
            i = 10
    return ivy.bitwise_invert(ivy.isreal(x))

@to_ivy_arrays_and_back
def iscomplexobj(x):
    if False:
        print('Hello World!')
    if x.ndim == 0:
        return ivy.is_complex_dtype(ivy.dtype(x))
    for ele in x:
        if ivy.is_complex_dtype(ivy.dtype(ele)):
            return True
        else:
            return False

@to_ivy_arrays_and_back
def isfortran(a):
    if False:
        return 10
    return a.flags.fnc

@to_ivy_arrays_and_back
def isreal(x):
    if False:
        while True:
            i = 10
    return ivy.isreal(x)

@to_ivy_arrays_and_back
def isrealobj(x: any):
    if False:
        print('Hello World!')
    return not ivy.is_complex_dtype(ivy.dtype(x))

@to_ivy_arrays_and_back
def isscalar(element):
    if False:
        for i in range(10):
            print('nop')
    return isinstance(element, (int, float, complex, bool, bytes, str, memoryview, numbers.Number, np_frontend.generic))