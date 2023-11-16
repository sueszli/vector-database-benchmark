import numpy as np

def multiply_by_10(arr):
    if False:
        while True:
            i = 10
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    arr_memview: cython.double[::1] = arr
    multiply_by_10_in_C(cython.address(arr_memview[0]), arr_memview.shape[0])
    return arr
a = np.ones(5, dtype=np.double)
print(multiply_by_10(a))
b = np.ones(10, dtype=np.double)
b = b[::2]
print(multiply_by_10(b))