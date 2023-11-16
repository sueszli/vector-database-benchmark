import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back

@to_ivy_arrays_and_back
def atleast_1d(*arys):
    if False:
        i = 10
        return i + 15
    return ivy.atleast_1d(*arys)

@to_ivy_arrays_and_back
def atleast_2d(*arys):
    if False:
        return 10
    return ivy.atleast_2d(*arys)

@to_ivy_arrays_and_back
def atleast_3d(*arys):
    if False:
        i = 10
        return i + 15
    return ivy.atleast_3d(*arys)

@to_ivy_arrays_and_back
def broadcast_arrays(*args):
    if False:
        while True:
            i = 10
    return ivy.broadcast_arrays(*args)

@to_ivy_arrays_and_back
def expand_dims(a, axis):
    if False:
        while True:
            i = 10
    return ivy.expand_dims(a, axis=axis)

@to_ivy_arrays_and_back
def squeeze(a, axis=None):
    if False:
        for i in range(10):
            print('nop')
    return ivy.squeeze(a, axis=axis)