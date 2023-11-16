import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back, from_zero_dim_arrays_to_scalar
from ivy.func_wrapper import with_unsupported_dtypes

@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def det(a):
    if False:
        for i in range(10):
            print('nop')
    return ivy.det(a)

@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def matrix_rank(A, tol=None, hermitian=False):
    if False:
        for i in range(10):
            print('nop')
    return ivy.matrix_rank(A, atol=tol, hermitian=hermitian)

@with_unsupported_dtypes({'1.26.2 and below': ('float16',)}, 'numpy')
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def norm(x, ord=None, axis=None, keepdims=False):
    if False:
        return 10
    if axis is None and ord is not None:
        if x.ndim not in (1, 2):
            raise ValueError('Improper number of dimensions to norm.')
        elif x.ndim == 1:
            ret = ivy.vector_norm(x, axis=axis, keepdims=keepdims, ord=ord)
        else:
            ret = ivy.matrix_norm(x, axis=axis, keepdims=keepdims, ord=ord)
    elif axis is None and ord is None:
        x = ivy.flatten(x)
        ret = ivy.vector_norm(x, axis=0, keepdims=keepdims, ord=2)
    if isinstance(axis, int):
        ret = ivy.vector_norm(x, axis=axis, keepdims=keepdims, ord=ord)
    elif isinstance(axis, tuple):
        ret = ivy.matrix_norm(x, axis=axis, keepdims=keepdims, ord=ord)
    return ret

@with_unsupported_dtypes({'1.26.2 and below': ('float16',)}, 'numpy')
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def slogdet(a):
    if False:
        print('Hello World!')
    (sign, logabsdet) = ivy.slogdet(a)
    return (sign, logabsdet)

@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def trace(a, offset=0, axis1=0, axis2=1, out=None):
    if False:
        return 10
    ret = ivy.trace(a, offset=offset, axis1=axis1, axis2=axis2, out=out)
    return ret