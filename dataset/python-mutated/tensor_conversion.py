"""Tensor conversion functions."""
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import tf_export

def convert_to_tensor_v1(value, dtype=None, name=None, preferred_dtype=None, dtype_hint=None) -> tensor_lib.Tensor:
    if False:
        print('Hello World!')
    'Converts the given `value` to a `Tensor` (with the TF1 API).'
    preferred_dtype = deprecation.deprecated_argument_lookup('dtype_hint', dtype_hint, 'preferred_dtype', preferred_dtype)
    return convert_to_tensor_v2(value, dtype, preferred_dtype, name)

@tf_export.tf_export(v1=['convert_to_tensor'])
@dispatch.add_dispatch_support
def convert_to_tensor_v1_with_dispatch(value, dtype=None, name=None, preferred_dtype=None, dtype_hint=None) -> tensor_lib.Tensor:
    if False:
        return 10
    'Converts the given `value` to a `Tensor`.\n\n  This function converts Python objects of various types to `Tensor`\n  objects. It accepts `Tensor` objects, numpy arrays, Python lists,\n  and Python scalars. For example:\n\n  ```python\n  import numpy as np\n\n  def my_func(arg):\n    arg = tf.convert_to_tensor(arg, dtype=tf.float32)\n    return tf.matmul(arg, arg) + arg\n\n  # The following calls are equivalent.\n  value_1 = my_func(tf.constant([[1.0, 2.0], [3.0, 4.0]]))\n  value_2 = my_func([[1.0, 2.0], [3.0, 4.0]])\n  value_3 = my_func(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))\n  ```\n\n  This function can be useful when composing a new operation in Python\n  (such as `my_func` in the example above). All standard Python op\n  constructors apply this function to each of their Tensor-valued\n  inputs, which allows those ops to accept numpy arrays, Python lists,\n  and scalars in addition to `Tensor` objects.\n\n  Note: This function diverges from default Numpy behavior for `float` and\n    `string` types when `None` is present in a Python list or scalar. Rather\n    than silently converting `None` values, an error will be thrown.\n\n  Args:\n    value: An object whose type has a registered `Tensor` conversion function.\n    dtype: Optional element type for the returned tensor. If missing, the type\n      is inferred from the type of `value`.\n    name: Optional name to use if a new `Tensor` is created.\n    preferred_dtype: Optional element type for the returned tensor, used when\n      dtype is None. In some cases, a caller may not have a dtype in mind when\n      converting to a tensor, so preferred_dtype can be used as a soft\n      preference.  If the conversion to `preferred_dtype` is not possible, this\n      argument has no effect.\n    dtype_hint: same meaning as preferred_dtype, and overrides it.\n\n  Returns:\n    A `Tensor` based on `value`.\n\n  Raises:\n    TypeError: If no conversion function is registered for `value` to `dtype`.\n    RuntimeError: If a registered conversion function returns an invalid value.\n    ValueError: If the `value` is a tensor not of given `dtype` in graph mode.\n  '
    return convert_to_tensor_v1(value, dtype=dtype, name=name, preferred_dtype=preferred_dtype, dtype_hint=dtype_hint)

@tf_export.tf_export('convert_to_tensor', v1=[])
@dispatch.add_dispatch_support
def convert_to_tensor_v2_with_dispatch(value, dtype=None, dtype_hint=None, name=None) -> tensor_lib.Tensor:
    if False:
        return 10
    'Converts the given `value` to a `Tensor`.\n\n  This function converts Python objects of various types to `Tensor`\n  objects. It accepts `Tensor` objects, numpy arrays, Python lists,\n  and Python scalars.\n\n  For example:\n\n  >>> import numpy as np\n  >>> def my_func(arg):\n  ...   arg = tf.convert_to_tensor(arg, dtype=tf.float32)\n  ...   return arg\n\n  >>> # The following calls are equivalent.\n  ...\n  >>> value_1 = my_func(tf.constant([[1.0, 2.0], [3.0, 4.0]]))\n  >>> print(value_1)\n  tf.Tensor(\n    [[1. 2.]\n     [3. 4.]], shape=(2, 2), dtype=float32)\n  >>> value_2 = my_func([[1.0, 2.0], [3.0, 4.0]])\n  >>> print(value_2)\n  tf.Tensor(\n    [[1. 2.]\n     [3. 4.]], shape=(2, 2), dtype=float32)\n  >>> value_3 = my_func(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))\n  >>> print(value_3)\n  tf.Tensor(\n    [[1. 2.]\n     [3. 4.]], shape=(2, 2), dtype=float32)\n\n  This function can be useful when composing a new operation in Python\n  (such as `my_func` in the example above). All standard Python op\n  constructors apply this function to each of their Tensor-valued\n  inputs, which allows those ops to accept numpy arrays, Python lists,\n  and scalars in addition to `Tensor` objects.\n\n  Note: This function diverges from default Numpy behavior for `float` and\n    `string` types when `None` is present in a Python list or scalar. Rather\n    than silently converting `None` values, an error will be thrown.\n\n  Args:\n    value: An object whose type has a registered `Tensor` conversion function.\n    dtype: Optional element type for the returned tensor. If missing, the type\n      is inferred from the type of `value`.\n    dtype_hint: Optional element type for the returned tensor, used when dtype\n      is None. In some cases, a caller may not have a dtype in mind when\n      converting to a tensor, so dtype_hint can be used as a soft preference. If\n      the conversion to `dtype_hint` is not possible, this argument has no\n      effect.\n    name: Optional name to use if a new `Tensor` is created.\n\n  Returns:\n    A `Tensor` based on `value`.\n\n  Raises:\n    TypeError: If no conversion function is registered for `value` to `dtype`.\n    RuntimeError: If a registered conversion function returns an invalid value.\n    ValueError: If the `value` is a tensor not of given `dtype` in graph mode.\n  '
    return convert_to_tensor_v2(value, dtype=dtype, dtype_hint=dtype_hint, name=name)

def convert_to_tensor_v2(value, dtype=None, dtype_hint=None, name=None) -> tensor_lib.Tensor:
    if False:
        for i in range(10):
            print('nop')
    'Converts the given `value` to a `Tensor`.'
    return tensor_conversion_registry.convert(value, dtype, name, preferred_dtype=dtype_hint)