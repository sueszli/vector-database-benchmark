import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back, from_zero_dim_arrays_to_scalar

@to_ivy_arrays_and_back
def eig(a):
    if False:
        while True:
            i = 10
    return ivy.eig(a)

@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def eigh(a, /, UPLO='L'):
    if False:
        print('Hello World!')
    return ivy.eigh(a, UPLO=UPLO)

@to_ivy_arrays_and_back
def eigvals(a):
    if False:
        print('Hello World!')
    return ivy.eig(a)[0]

@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def eigvalsh(a, /, UPLO='L'):
    if False:
        while True:
            i = 10
    return ivy.eigvalsh(a, UPLO=UPLO)