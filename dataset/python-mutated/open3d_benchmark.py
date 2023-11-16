import numpy as np
import open3d as o3d
import open3d.core as o3c

def list_tensor_sizes():
    if False:
        print('Hello World!')
    return [100000000]

def list_non_bool_dtypes():
    if False:
        i = 10
        return i + 15
    return [o3c.int8, o3c.uint8, o3c.int16, o3c.uint16, o3c.int32, o3c.uint32, o3c.int64, o3c.uint64, o3c.float32, o3c.float64]

def list_float_dtypes():
    if False:
        for i in range(10):
            print('nop')
    return [o3c.float32, o3c.float64]

def to_numpy_dtype(dtype: o3c.Dtype):
    if False:
        while True:
            i = 10
    conversions = {o3c.bool8: np.bool8, o3c.bool: np.bool8, o3c.int8: np.int8, o3c.uint8: np.uint8, o3c.int16: np.int16, o3c.uint16: np.uint16, o3c.int32: np.int32, o3c.uint32: np.uint32, o3c.int64: np.int64, o3c.uint64: np.uint64, o3c.float32: np.float32, o3c.float64: np.float64}
    return conversions[dtype]

def list_devices():
    if False:
        while True:
            i = 10
    devices = [o3c.Device('CPU:0')]
    if o3c.cuda.is_available():
        devices.append(o3c.Device('CUDA:0'))
    return devices