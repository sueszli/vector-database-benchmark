import ivy
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back

@to_ivy_arrays_and_back
def imag(x):
    if False:
        return 10
    return ivy.imag(x)

@to_ivy_arrays_and_back
def is_complex(x):
    if False:
        print('Hello World!')
    return ivy.is_complex_dtype(x)

@to_ivy_arrays_and_back
def is_floating_point(x):
    if False:
        for i in range(10):
            print('nop')
    return ivy.is_float_dtype(x)

@to_ivy_arrays_and_back
def is_integer(x):
    if False:
        while True:
            i = 10
    return ivy.is_int_dtype(x)

@to_ivy_arrays_and_back
def rank(input):
    if False:
        return 10
    return ivy.get_num_dims(input)

@to_ivy_arrays_and_back
def real(x):
    if False:
        while True:
            i = 10
    return ivy.real(x)