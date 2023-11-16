import ivy.functional.frontends.numpy as np_frontend
import ivy

def asmatrix(data, dtype=None):
    if False:
        while True:
            i = 10
    return np_frontend.matrix(ivy.array(data), dtype=dtype, copy=False)

def asscalar(a):
    if False:
        i = 10
        return i + 15
    return a.item()