import numba
import numpy as np
rescale_functions = {}

def rescale_clip_source(xx, scale, offset, vmin, vmax, yy):
    if False:
        while True:
            i = 10
    for i in range(xx.size):
        val = (xx[i] - offset) * scale
        yy[i] = min(max(val, vmin), vmax)

def rescaleData(data, scale, offset, dtype, clip):
    if False:
        while True:
            i = 10
    data_out = np.empty_like(data, dtype=dtype)
    key = (data.dtype.name, data_out.dtype.name)
    func = rescale_functions.get(key)
    if func is None:
        func = numba.guvectorize([f'{key[0]}[:],f8,f8,f8,f8,{key[1]}[:]'], '(n),(),(),(),()->(n)', nopython=True)(rescale_clip_source)
        rescale_functions[key] = func
    func(data, scale, offset, clip[0], clip[1], out=data_out)
    return data_out

@numba.jit(nopython=True)
def _rescale_and_lookup1d_function(data, scale, offset, lut, out):
    if False:
        i = 10
        return i + 15
    (vmin, vmax) = (0, lut.shape[0] - 1)
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            val = (data[r, c] - offset) * scale
            val = min(max(val, vmin), vmax)
            out[r, c] = lut[int(val)]

def rescale_and_lookup1d(data, scale, offset, lut):
    if False:
        return 10
    data_out = np.empty_like(data, dtype=lut.dtype)
    _rescale_and_lookup1d_function(data, float(scale), float(offset), lut, data_out)
    return data_out

@numba.jit(nopython=True)
def numba_take(lut, data):
    if False:
        return 10
    return np.take(lut, data)