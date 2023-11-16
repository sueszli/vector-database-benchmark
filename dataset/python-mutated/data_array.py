import numpy as np
from cinn import common, runtime
from cinn.common import BFloat16, Bool, Float, Float16, Int, UInt

class DataArray:
    """
    Provides Python encapsulation of the cinn_buffer_t
    data interface in the CINN RunTime module.
    """

    def __init__(self, shape: list, dtype: common.Type=common.Float(32), data: runtime.cinn_buffer_t=None) -> None:
        if False:
            return 10
        self.shape = shape
        self.dtype = dtype
        self.data = data

    def to_numpy(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert DataArray to numpy array\n        '
        np_dtype = 'unk'
        if self.dtype.is_bfloat16():
            np_dtype = 'uint16'
        elif self.dtype.is_float16():
            np_dtype = 'float16'
        elif self.dtype.is_float(32, common.Type.specific_type_t.UNK):
            np_dtype = 'float32'
        elif self.dtype.is_float(64, common.Type.specific_type_t.UNK):
            np_dtype = 'float64'
        elif self.dtype.is_int(8):
            np_dtype = 'int8'
        elif self.dtype.is_int(16):
            np_dtype = 'int16'
        elif self.dtype.is_int(32):
            np_dtype = 'int32'
        elif self.dtype.is_int(64):
            np_dtype = 'int64'
        elif self.dtype.is_uint(8):
            np_dtype = 'uint8'
        elif self.dtype.is_uint(32):
            np_dtype = 'uint32'
        elif self.dtype.is_uint(64):
            np_dtype = 'uint64'
        elif self.dtype.is_bool():
            np_dtype = 'bool'
        else:
            raise TypeError(f'no support {self.dtype} in CINN')
        np_arr = np.empty(self.shape, np_dtype)
        assert np_arr.flags['C_CONTIGUOUS']
        self.data.copy_to(np_arr)
        return np_arr

    @staticmethod
    def from_numpy(np_array, target=common.DefaultHostTarget()):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create DataArray form numpy array\n        '
        assert isinstance(np_array, np.ndarray)
        data = runtime.cinn_buffer_t(np_array, target)
        dtype_np_to_common = {'uint16': BFloat16(), 'bfloat16': BFloat16(), 'float16': Float16(), 'float32': Float(32), 'float64': Float(64), 'int8': Int(8), 'int16': Int(16), 'int32': Int(32), 'int64': Int(64), 'uint8': UInt(8), 'uint32': UInt(32), 'uint64': UInt(64), 'bool': Bool()}
        dtype_np = str(np_array.dtype).split('.')[-1]
        assert str(dtype_np) in dtype_np_to_common, str(dtype_np) + ' not support in CINN'
        assert dtype_np in dtype_np_to_common.keys()
        return DataArray(np_array.shape, dtype_np_to_common[dtype_np], data)