import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.numpy import promote_types_of_numpy_inputs
from ivy.functional.frontends.numpy.linalg.norms_and_other_numbers import matrix_rank

@with_unsupported_dtypes({'1.26.2 and below': ('float16',)}, 'numpy')
@to_ivy_arrays_and_back
def inv(a):
    if False:
        i = 10
        return i + 15
    return ivy.inv(a)

@to_ivy_arrays_and_back
@with_unsupported_dtypes({'1.26.2 and below': ('float16',)}, 'numpy')
def lstsq(a, b, rcond='warn'):
    if False:
        return 10
    solution = ivy.matmul(ivy.pinv(a, rtol=1e-15).astype(ivy.float64), b.astype(ivy.float64))
    svd = ivy.svd(a, compute_uv=False)
    rank = matrix_rank(a).astype(ivy.int32)
    residuals = ivy.sum((b - ivy.matmul(a, solution)) ** 2).astype(ivy.float64)
    return (solution, residuals, rank, svd[0])

@with_unsupported_dtypes({'1.26.2 and below': ('float16',)}, 'numpy')
@to_ivy_arrays_and_back
def pinv(a, rcond=1e-15, hermitian=False):
    if False:
        return 10
    return ivy.pinv(a, rtol=rcond)

@with_unsupported_dtypes({'1.26.2 and below': ('float16',)}, 'numpy')
@to_ivy_arrays_and_back
def solve(a, b):
    if False:
        print('Hello World!')
    (a, b) = promote_types_of_numpy_inputs(a, b)
    return ivy.solve(a, b)

@to_ivy_arrays_and_back
@with_unsupported_dtypes({'1.26.2 and below': ('float16', 'blfloat16')}, 'numpy')
def tensorinv(a, ind=2):
    if False:
        for i in range(10):
            print('nop')
    old_shape = ivy.shape(a)
    prod = 1
    if ind > 0:
        invshape = old_shape[ind:] + old_shape[:ind]
        for k in old_shape[ind:]:
            prod *= k
    else:
        raise ValueError('Invalid ind argument.')
    a = ivy.reshape(a, shape=(prod, -1))
    ia = ivy.inv(a)
    new_shape = tuple([*invshape])
    return ivy.reshape(ia, shape=new_shape)