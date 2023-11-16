"""Config functions for TF NumPy."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import weak_tensor_ops
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import tf_export

@tf_export.tf_export('experimental.numpy.experimental_enable_numpy_behavior', v1=[])
def enable_numpy_behavior(prefer_float32=False, dtype_conversion_mode='legacy'):
    if False:
        print('Hello World!')
    'Enable NumPy behavior on Tensors.\n\n  Enabling NumPy behavior has three effects:\n  * It adds to `tf.Tensor` some common NumPy methods such as `T`,\n    `reshape` and `ravel`.\n  * It changes dtype promotion in `tf.Tensor` operators to be\n    compatible with NumPy. For example,\n    `tf.ones([], tf.int32) + tf.ones([], tf.float32)` used to throw a\n    "dtype incompatible" error, but after this it will return a\n    float64 tensor (obeying NumPy\'s promotion rules).\n  * It enhances `tf.Tensor`\'s indexing capability to be on par with\n    [NumPy\'s](https://numpy.org/doc/stable/reference/arrays.indexing.html).\n\n  Args:\n    prefer_float32: Controls whether dtype inference will use float32 for Python\n      floats, or float64 (the default and the NumPy-compatible behavior).\n    dtype_conversion_mode: a string that specifies promotion mode. This string\n      corresponds to a PromoMode Enum and can be \'off\', \'legacy\', \'safe\', or\n      \'all\'. \'safe\' or \'all\' mode enables the auto dtype conversion semantics.\n  '
    if dtype_conversion_mode == 'safe' or dtype_conversion_mode == 'all':
        tf_logging.warning('UserWarning: enabling the new type promotion must happen at the beginning of the program. Please ensure no TF APIs have been used yet.')
    ops.set_dtype_conversion_mode(dtype_conversion_mode)
    ops.enable_numpy_style_slicing()
    np_math_ops.enable_numpy_methods_on_tensor()
    np_dtypes.set_prefer_float32(prefer_float32)