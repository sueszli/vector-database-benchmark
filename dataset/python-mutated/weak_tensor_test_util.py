"""Utils for WeakTensor related tests."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework.weak_tensor import WeakTensor

def get_test_input_for_op(val, dtype):
    if False:
        while True:
            i = 10
    'Returns a list containing all the possible inputs with a given dtype.\n\n  Args:\n    val: value to convert to test input.\n    dtype: a tuple of format (tf.Dtype, bool) where the bool value represents\n      whether the dtype is "weak" or not.\n\n  Returns:\n    A list of all possible inputs given a value and a dtype.\n  '
    python_inferred_types = {(dtypes.int32, True): 1, (dtypes.float32, True): 1.0, (dtypes.complex128, True): 1j}
    (dtype, weak) = dtype
    inputs = []
    if weak:
        inputs.append(convert_to_input_type(val, 'WeakTensor', dtype))
        if dtype in python_inferred_types:
            val_in_dtype = val * python_inferred_types[dtype]
            inputs.append(val_in_dtype)
            inputs.append(convert_to_input_type(val_in_dtype, 'Tensor', None))
    else:
        inputs.append(convert_to_input_type(val, 'Tensor', dtype))
        inputs.append(convert_to_input_type(val, 'NumPy', dtype))
    return inputs

def convert_to_input_type(base_input, input_type, dtype=None):
    if False:
        print('Hello World!')
    if input_type == 'WeakTensor':
        return WeakTensor.from_tensor(constant_op.constant(base_input, dtype=dtype))
    elif input_type == 'Tensor':
        return constant_op.constant(base_input, dtype=dtype)
    elif input_type == 'NumPy':
        dtype = dtype.as_numpy_dtype if isinstance(dtype, dtypes.DType) else dtype
        return np.array(base_input, dtype=dtype)
    elif input_type == 'Python':
        return base_input
    else:
        raise ValueError(f'The provided input_type {input_type} is not supported.')

def get_weak_tensor(*args, **kwargs):
    if False:
        print('Hello World!')
    return WeakTensor.from_tensor(constant_op.constant(*args, **kwargs))

class DtypeConversionTestEnv:
    """Test environment for different dtype conversion semantics."""

    def __init__(self, promo_mode):
        if False:
            return 10
        self._old_promo_mode = ops.promo_mode_enum_to_string(ops.get_dtype_conversion_mode())
        self._new_promo_mode = promo_mode

    def __enter__(self):
        if False:
            while True:
                i = 10
        ops.set_dtype_conversion_mode(self._new_promo_mode)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if False:
            i = 10
            return i + 15
        ops.set_dtype_conversion_mode(self._old_promo_mode)