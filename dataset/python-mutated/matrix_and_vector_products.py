import ivy
from ivy.functional.frontends.numpy import promote_types_of_numpy_inputs
from ivy import with_unsupported_dtypes
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back, handle_numpy_casting, handle_numpy_dtype, from_zero_dim_arrays_to_scalar, handle_numpy_out

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _matmul(x1, x2, /, out=None, *, casting='same_kind', order='K', dtype=None, subok=True):
    if False:
        i = 10
        return i + 15
    return ivy.matmul(x1, x2, out=out)

@to_ivy_arrays_and_back
def cross(a, b, *, axisa=-1, axisb=-1, axisc=-1, axis=None):
    if False:
        while True:
            i = 10
    return ivy.cross(a, b, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)

@handle_numpy_out
@to_ivy_arrays_and_back
def dot(a, b, out=None):
    if False:
        print('Hello World!')
    (a, b) = promote_types_of_numpy_inputs(a, b)
    return ivy.matmul(a, b, out=out)

@handle_numpy_out
@to_ivy_arrays_and_back
def einsum(subscripts, *operands, out=None, dtype=None, order='K', casting='safe', optimize=False):
    if False:
        print('Hello World!')
    return ivy.einsum(subscripts, *operands, out=out)

@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def inner(a, b, /):
    if False:
        print('Hello World!')
    (a, b) = promote_types_of_numpy_inputs(a, b)
    return ivy.inner(a, b)

@to_ivy_arrays_and_back
def kron(a, b):
    if False:
        return 10
    (a, b) = promote_types_of_numpy_inputs(a, b)
    return ivy.kron(a, b)

@to_ivy_arrays_and_back
def matrix_power(a, n):
    if False:
        print('Hello World!')
    return ivy.matrix_power(a, n)

@with_unsupported_dtypes({'2.0.0 and below': ('float16',)}, 'torch')
@handle_numpy_out
@to_ivy_arrays_and_back
def multi_dot(arrays, *, out=None):
    if False:
        return 10
    return ivy.multi_dot(arrays, out=out)

@handle_numpy_out
@to_ivy_arrays_and_back
def outer(a, b, out=None):
    if False:
        return 10
    (a, b) = promote_types_of_numpy_inputs(a, b)
    return ivy.outer(a, b, out=out)

@to_ivy_arrays_and_back
def tensordot(a, b, axes=2):
    if False:
        while True:
            i = 10
    return ivy.tensordot(a, b, axes=axes)

@to_ivy_arrays_and_back
def tensorsolve(a, b, axes=2):
    if False:
        for i in range(10):
            print('nop')
    return ivy.tensorsolve(a, b, axes=axes)