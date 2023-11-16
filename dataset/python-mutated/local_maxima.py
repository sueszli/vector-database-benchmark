import numpy as np

def wrap(pos, offset, bound):
    if False:
        while True:
            i = 10
    return (pos + offset) % bound

def clamp(pos, offset, bound):
    if False:
        return 10
    return min(bound - 1, max(0, pos + offset))

def reflect(pos, offset, bound):
    if False:
        for i in range(10):
            print('nop')
    idx = pos + offset
    return min(2 * (bound - 1) - idx, max(idx, -idx))

def local_maxima(data, mode=wrap):
    if False:
        for i in range(10):
            print('nop')
    wsize = data.shape
    result = np.ones(data.shape, bool)
    for pos in np.ndindex(data.shape):
        myval = data[pos]
        for offset in np.ndindex(wsize):
            neighbor_idx = tuple((mode(p, o - w // 2, w) for (p, o, w) in zip(pos, offset, wsize, strict=True)))
            result[pos] &= data[neighbor_idx] <= myval
    return result