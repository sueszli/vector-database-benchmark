import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back

@to_ivy_arrays_and_back
def repeat(a, repeats, axis=None):
    if False:
        i = 10
        return i + 15
    return ivy.repeat(a, repeats, axis=axis)

@to_ivy_arrays_and_back
def tile(A, reps):
    if False:
        for i in range(10):
            print('nop')
    return ivy.tile(A, reps)