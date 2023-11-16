"""ndarray class."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.ops.numpy_ops import np_dtypes

def convert_to_tensor(value, dtype=None, dtype_hint=None):
    if False:
        for i in range(10):
            print('nop')
    'Wrapper over `tf.convert_to_tensor`.\n\n  Args:\n    value: value to convert\n    dtype: (optional) the type we would like it to be converted to.\n    dtype_hint: (optional) soft preference for the type we would like it to be\n      converted to. `tf.convert_to_tensor` will attempt to convert value to this\n      type first, but will not fail if conversion is not possible falling back\n      to inferring the type instead.\n\n  Returns:\n    Value converted to tf.Tensor.\n  '
    if dtype is None and isinstance(value, int) and (value >= 2 ** 63):
        dtype = dtypes.uint64
    elif dtype is None and dtype_hint is None and isinstance(value, float):
        dtype = np_dtypes.default_float_type()
    return tensor_conversion.convert_to_tensor_v2_with_dispatch(value, dtype=dtype, dtype_hint=dtype_hint)
ndarray = tensor.Tensor