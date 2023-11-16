"""Double op is a user's defined op for testing purpose."""
from tensorflow.lite.python.testdata import double_op_wrapper
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
_double_op = load_library.load_op_library(resource_loader.get_path_to_datafile('_double_op.so'))

def double(input_tensor):
    if False:
        return 10
    'Double op applies element-wise double to input data.'
    if input_tensor.dtype != dtypes.int32 and input_tensor.dtype != dtypes.float32:
        raise ValueError('Double op only accept int32 or float32 values.')
    return double_op_wrapper.double(input_tensor)