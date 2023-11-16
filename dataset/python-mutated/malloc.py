import random
from cython.cimports.libc.stdlib import malloc, free

def random_noise(number: cython.int=1):
    if False:
        return 10
    i: cython.int
    my_array: cython.p_double = cython.cast(cython.p_double, malloc(number * cython.sizeof(cython.double)))
    if not my_array:
        raise MemoryError()
    try:
        ran = random.normalvariate
        for i in range(number):
            my_array[i] = ran(0, 1)
        return [x for x in my_array[:number]]
    finally:
        free(my_array)