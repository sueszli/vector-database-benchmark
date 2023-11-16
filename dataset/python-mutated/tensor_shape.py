"""Helper classes for tensor shape inference."""
import functools
import operator
from typing import Optional, Sequence, Type, Union
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import monitoring
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import trace
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
_TENSORSHAPE_V2_OVERRIDE = None
_api_usage_gauge = monitoring.BoolGauge('/tensorflow/api/v2_tensorshape', 'Whether tensor_shape.enable_v2_tensorshape() is called.')

@tf_export(v1=['enable_v2_tensorshape'])
def enable_v2_tensorshape():
    if False:
        while True:
            i = 10
    'In TensorFlow 2.0, iterating over a TensorShape instance returns values.\n\n  This enables the new behavior.\n\n  Concretely, `tensor_shape[i]` returned a Dimension instance in V1, but\n  it V2 it returns either an integer, or None.\n\n  Examples:\n\n  ```\n  #######################\n  # If you had this in V1:\n  value = tensor_shape[i].value\n\n  # Do this in V2 instead:\n  value = tensor_shape[i]\n\n  #######################\n  # If you had this in V1:\n  for dim in tensor_shape:\n    value = dim.value\n    print(value)\n\n  # Do this in V2 instead:\n  for value in tensor_shape:\n    print(value)\n\n  #######################\n  # If you had this in V1:\n  dim = tensor_shape[i]\n  dim.assert_is_compatible_with(other_shape)  # or using any other shape method\n\n  # Do this in V2 instead:\n  if tensor_shape.rank is None:\n    dim = Dimension(None)\n  else:\n    dim = tensor_shape.dims[i]\n  dim.assert_is_compatible_with(other_shape)  # or using any other shape method\n\n  # The V2 suggestion above is more explicit, which will save you from\n  # the following trap (present in V1):\n  # you might do in-place modifications to `dim` and expect them to be reflected\n  # in `tensor_shape[i]`, but they would not be.\n  ```\n  '
    global _TENSORSHAPE_V2_OVERRIDE
    _TENSORSHAPE_V2_OVERRIDE = True
    logging.vlog(1, 'Enabling v2 tensorshape')
    _api_usage_gauge.get_cell().set(True)

@tf_export(v1=['disable_v2_tensorshape'])
def disable_v2_tensorshape():
    if False:
        return 10
    'Disables the V2 TensorShape behavior and reverts to V1 behavior.\n\n  See docstring for `enable_v2_tensorshape` for details about the new behavior.\n  '
    global _TENSORSHAPE_V2_OVERRIDE
    _TENSORSHAPE_V2_OVERRIDE = False
    logging.vlog(1, 'Disabling v2 tensorshape')
    _api_usage_gauge.get_cell().set(False)

@tf_export('compat.dimension_value', v1=['dimension_value', 'compat.dimension_value'])
def dimension_value(dimension: Union['Dimension', int, None]) -> Union[int, None]:
    if False:
        for i in range(10):
            print('nop')
    'Compatibility utility required to allow for both V1 and V2 behavior in TF.\n\n  Until the release of TF 2.0, we need the legacy behavior of `TensorShape` to\n  coexist with the new behavior. This utility is a bridge between the two.\n\n  When accessing the value of a TensorShape dimension,\n  use this utility, like this:\n\n  ```\n  # If you had this in your V1 code:\n  value = tensor_shape[i].value\n\n  # Use `dimension_value` as direct replacement compatible with both V1 & V2:\n  value = dimension_value(tensor_shape[i])\n\n  # This would be the V2 equivalent:\n  value = tensor_shape[i]  # Warning: this will return the dim value in V2!\n  ```\n\n  Args:\n    dimension: Either a `Dimension` instance, an integer, or None.\n\n  Returns:\n    A plain value, i.e. an integer or None.\n  '
    if isinstance(dimension, Dimension):
        return dimension.value
    return dimension

@tf_export('compat.dimension_at_index', v1=['dimension_at_index', 'compat.dimension_at_index'])
def dimension_at_index(shape, index) -> 'Dimension':
    if False:
        for i in range(10):
            print('nop')
    'Compatibility utility required to allow for both V1 and V2 behavior in TF.\n\n  Until the release of TF 2.0, we need the legacy behavior of `TensorShape` to\n  coexist with the new behavior. This utility is a bridge between the two.\n\n  If you want to retrieve the Dimension instance corresponding to a certain\n  index in a TensorShape instance, use this utility, like this:\n\n  ```\n  # If you had this in your V1 code:\n  dim = tensor_shape[i]\n\n  # Use `dimension_at_index` as direct replacement compatible with both V1 & V2:\n  dim = dimension_at_index(tensor_shape, i)\n\n  # Another possibility would be this, but WARNING: it only works if the\n  # tensor_shape instance has a defined rank.\n  dim = tensor_shape.dims[i]  # `dims` may be None if the rank is undefined!\n\n  # In native V2 code, we recommend instead being more explicit:\n  if tensor_shape.rank is None:\n    dim = Dimension(None)\n  else:\n    dim = tensor_shape.dims[i]\n\n  # Being more explicit will save you from the following trap (present in V1):\n  # you might do in-place modifications to `dim` and expect them to be reflected\n  # in `tensor_shape[i]`, but they would not be (as the Dimension object was\n  # instantiated on the fly.\n  ```\n\n  Args:\n    shape: A TensorShape instance.\n    index: An integer index.\n\n  Returns:\n    A dimension object.\n  '
    assert isinstance(shape, TensorShape)
    if shape.rank is None:
        return Dimension(None)
    else:
        return shape.dims[index]

@tf_export(v1=['Dimension'])
class Dimension(object):
    """Represents the value of one dimension in a TensorShape.

  @compatibility(TF2)
  In TF2, members of a `TensorShape` object are integers. The `Dimension` class
  is not part of TF2's data model.

  Please refer to the [TensorShape section of the migration guide]
  (https://www.tensorflow.org/guide/migrate/index#tensorshape) on common code
  patterns adapting Dimension objects to a TF2 syntax.
  @end_compatibility
  """
    __slots__ = ['_value']

    def __init__(self, value):
        if False:
            return 10
        'Creates a new Dimension with the given value.'
        if isinstance(value, int):
            if value < 0:
                raise ValueError('Dimension %d must be >= 0' % value)
            self._value = value
        elif value is None:
            self._value = None
        elif isinstance(value, Dimension):
            self._value = value._value
        else:
            try:
                self._value = int(value.__index__())
            except AttributeError:
                raise TypeError("Dimension value must be integer or None or have an __index__ method, got value '{0!r}' with type '{1!r}'".format(value, type(value))) from None
            if self._value < 0:
                raise ValueError('Dimension %d must be >= 0' % self._value)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'Dimension(%s)' % repr(self._value)

    def __str__(self):
        if False:
            return 10
        value = self._value
        return '?' if value is None else str(value)

    def __eq__(self, other):
        if False:
            return 10
        'Returns true if `other` has the same known value as this Dimension.'
        try:
            other = as_dimension(other)
        except (TypeError, ValueError):
            return NotImplemented
        if self._value is None or other.value is None:
            return None
        return self._value == other.value

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        'Returns true if `other` has a different known value from `self`.'
        try:
            other = as_dimension(other)
        except (TypeError, ValueError):
            return NotImplemented
        if self._value is None or other.value is None:
            return None
        return self._value != other.value

    def __bool__(self):
        if False:
            for i in range(10):
                print('nop')
        'Equivalent to `bool(self.value)`.'
        return bool(self._value)

    def __int__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._value

    def __long__(self):
        if False:
            i = 10
            return i + 15
        return self._value

    def __index__(self):
        if False:
            i = 10
            return i + 15
        return self._value

    @property
    def value(self):
        if False:
            return 10
        'The value of this dimension, or None if it is unknown.'
        return self._value

    def is_compatible_with(self, other):
        if False:
            return 10
        'Returns true if `other` is compatible with this Dimension.\n\n    Two known Dimensions are compatible if they have the same value.\n    An unknown Dimension is compatible with all other Dimensions.\n\n    Args:\n      other: Another Dimension.\n\n    Returns:\n      True if this Dimension and `other` are compatible.\n    '
        other = as_dimension(other)
        return self._value is None or other.value is None or self._value == other.value

    def assert_is_compatible_with(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Raises an exception if `other` is not compatible with this Dimension.\n\n    Args:\n      other: Another Dimension.\n\n    Raises:\n      ValueError: If `self` and `other` are not compatible (see\n        is_compatible_with).\n    '
        if not self.is_compatible_with(other):
            raise ValueError('Dimensions %s and %s are not compatible' % (self, other))

    def merge_with(self, other):
        if False:
            return 10
        'Returns a Dimension that combines the information in `self` and `other`.\n\n    Dimensions are combined as follows:\n\n    ```python\n    tf.compat.v1.Dimension(n)   .merge_with(tf.compat.v1.Dimension(n))     ==\n    tf.compat.v1.Dimension(n)\n    tf.compat.v1.Dimension(n)   .merge_with(tf.compat.v1.Dimension(None))  ==\n    tf.compat.v1.Dimension(n)\n    tf.compat.v1.Dimension(None).merge_with(tf.compat.v1.Dimension(n))     ==\n    tf.compat.v1.Dimension(n)\n    # equivalent to tf.compat.v1.Dimension(None)\n    tf.compat.v1.Dimension(None).merge_with(tf.compat.v1.Dimension(None))\n\n    # raises ValueError for n != m\n    tf.compat.v1.Dimension(n)   .merge_with(tf.compat.v1.Dimension(m))\n    ```\n\n    Args:\n      other: Another Dimension.\n\n    Returns:\n      A Dimension containing the combined information of `self` and\n      `other`.\n\n    Raises:\n      ValueError: If `self` and `other` are not compatible (see\n        is_compatible_with).\n    '
        other = as_dimension(other)
        self.assert_is_compatible_with(other)
        if self._value is None:
            return Dimension(other.value)
        else:
            return Dimension(self._value)

    def __add__(self, other):
        if False:
            while True:
                i = 10
        'Returns the sum of `self` and `other`.\n\n    Dimensions are summed as follows:\n\n    ```python\n    tf.compat.v1.Dimension(m)    + tf.compat.v1.Dimension(n)     ==\n    tf.compat.v1.Dimension(m + n)\n    tf.compat.v1.Dimension(m)    + tf.compat.v1.Dimension(None)  # equiv. to\n    tf.compat.v1.Dimension(None)\n    tf.compat.v1.Dimension(None) + tf.compat.v1.Dimension(n)     # equiv. to\n    tf.compat.v1.Dimension(None)\n    tf.compat.v1.Dimension(None) + tf.compat.v1.Dimension(None)  # equiv. to\n    tf.compat.v1.Dimension(None)\n    ```\n\n    Args:\n      other: Another Dimension, or a value accepted by `as_dimension`.\n\n    Returns:\n      A Dimension whose value is the sum of `self` and `other`.\n    '
        try:
            other = as_dimension(other)
        except (TypeError, ValueError):
            return NotImplemented
        if self._value is None or other.value is None:
            return Dimension(None)
        else:
            return Dimension(self._value + other.value)

    def __radd__(self, other):
        if False:
            print('Hello World!')
        'Returns the sum of `other` and `self`.\n\n    Args:\n      other: Another Dimension, or a value accepted by `as_dimension`.\n\n    Returns:\n      A Dimension whose value is the sum of `self` and `other`.\n    '
        return self + other

    def __sub__(self, other):
        if False:
            while True:
                i = 10
        'Returns the subtraction of `other` from `self`.\n\n    Dimensions are subtracted as follows:\n\n    ```python\n    tf.compat.v1.Dimension(m)    - tf.compat.v1.Dimension(n)     ==\n    tf.compat.v1.Dimension(m - n)\n    tf.compat.v1.Dimension(m)    - tf.compat.v1.Dimension(None)  # equiv. to\n    tf.compat.v1.Dimension(None)\n    tf.compat.v1.Dimension(None) - tf.compat.v1.Dimension(n)     # equiv. to\n    tf.compat.v1.Dimension(None)\n    tf.compat.v1.Dimension(None) - tf.compat.v1.Dimension(None)  # equiv. to\n    tf.compat.v1.Dimension(None)\n    ```\n\n    Args:\n      other: Another Dimension, or a value accepted by `as_dimension`.\n\n    Returns:\n      A Dimension whose value is the subtraction of `other` from `self`.\n    '
        try:
            other = as_dimension(other)
        except (TypeError, ValueError):
            return NotImplemented
        if self._value is None or other.value is None:
            return Dimension(None)
        else:
            return Dimension(self._value - other.value)

    def __rsub__(self, other):
        if False:
            while True:
                i = 10
        'Returns the subtraction of `self` from `other`.\n\n    Args:\n      other: Another Dimension, or a value accepted by `as_dimension`.\n\n    Returns:\n      A Dimension whose value is the subtraction of `self` from `other`.\n    '
        other = as_dimension(other)
        if self._value is None or other.value is None:
            return Dimension(None)
        else:
            return Dimension(other.value - self._value)

    def __mul__(self, other):
        if False:
            i = 10
            return i + 15
        'Returns the product of `self` and `other`.\n\n    Dimensions are summed as follows:\n\n    ```python\n    tf.compat.v1.Dimension(m)    * tf.compat.v1.Dimension(n)     ==\n    tf.compat.v1.Dimension(m * n)\n    tf.compat.v1.Dimension(m)    * tf.compat.v1.Dimension(None)  # equiv. to\n    tf.compat.v1.Dimension(None)\n    tf.compat.v1.Dimension(None) * tf.compat.v1.Dimension(n)     # equiv. to\n    tf.compat.v1.Dimension(None)\n    tf.compat.v1.Dimension(None) * tf.compat.v1.Dimension(None)  # equiv. to\n    tf.compat.v1.Dimension(None)\n    ```\n\n    Args:\n      other: Another Dimension, or a value accepted by `as_dimension`.\n\n    Returns:\n      A Dimension whose value is the product of `self` and `other`.\n    '
        try:
            other = as_dimension(other)
        except (TypeError, ValueError):
            return NotImplemented
        if self._value is None or other.value is None:
            return Dimension(None)
        else:
            return Dimension(self._value * other.value)

    def __rmul__(self, other):
        if False:
            while True:
                i = 10
        'Returns the product of `self` and `other`.\n\n    Args:\n      other: Another Dimension, or a value accepted by `as_dimension`.\n\n    Returns:\n      A Dimension whose value is the product of `self` and `other`.\n    '
        return self * other

    def __floordiv__(self, other):
        if False:
            print('Hello World!')
        'Returns the quotient of `self` and `other` rounded down.\n\n    Dimensions are divided as follows:\n\n    ```python\n    tf.compat.v1.Dimension(m)    // tf.compat.v1.Dimension(n)     ==\n    tf.compat.v1.Dimension(m // n)\n    tf.compat.v1.Dimension(m)    // tf.compat.v1.Dimension(None)  # equiv. to\n    tf.compat.v1.Dimension(None)\n    tf.compat.v1.Dimension(None) // tf.compat.v1.Dimension(n)     # equiv. to\n    tf.compat.v1.Dimension(None)\n    tf.compat.v1.Dimension(None) // tf.compat.v1.Dimension(None)  # equiv. to\n    tf.compat.v1.Dimension(None)\n    ```\n\n    Args:\n      other: Another Dimension, or a value accepted by `as_dimension`.\n\n    Returns:\n      A `Dimension` whose value is the integer quotient of `self` and `other`.\n    '
        try:
            other = as_dimension(other)
        except (TypeError, ValueError):
            return NotImplemented
        if self._value is None or other.value is None:
            return Dimension(None)
        else:
            return Dimension(self._value // other.value)

    def __rfloordiv__(self, other):
        if False:
            return 10
        'Returns the quotient of `other` and `self` rounded down.\n\n    Args:\n      other: Another Dimension, or a value accepted by `as_dimension`.\n\n    Returns:\n      A `Dimension` whose value is the integer quotient of `self` and `other`.\n    '
        other = as_dimension(other)
        if self._value is None or other.value is None:
            return Dimension(None)
        else:
            return Dimension(other.value // self._value)

    def __div__(self, other):
        if False:
            return 10
        'DEPRECATED: Use `__floordiv__` via `x // y` instead.\n\n    This function exists only for backwards compatibility purposes; new code\n    should use `__floordiv__` via the syntax `x // y`.  Using `x // y`\n    communicates clearly that the result rounds down, and is forward compatible\n    to Python 3.\n\n    Args:\n      other: Another `Dimension`.\n\n    Returns:\n      A `Dimension` whose value is the integer quotient of `self` and `other`.\n    '
        return self // other

    def __rdiv__(self, other):
        if False:
            while True:
                i = 10
        "Use `__floordiv__` via `x // y` instead.\n\n    This function exists only to have a better error message. Instead of:\n    `TypeError: unsupported operand type(s) for /: 'int' and 'Dimension'`,\n    this function will explicitly call for usage of `//` instead.\n\n    Args:\n      other: Another `Dimension`.\n\n    Raises:\n      TypeError.\n    "
        raise TypeError("unsupported operand type(s) for /: '{}' and 'Dimension', please use // instead".format(type(other).__name__))

    def __truediv__(self, other):
        if False:
            for i in range(10):
                print('nop')
        "Use `__floordiv__` via `x // y` instead.\n\n    This function exists only to have a better error message. Instead of:\n    `TypeError: unsupported operand type(s) for /: 'Dimension' and 'int'`,\n    this function will explicitly call for usage of `//` instead.\n\n    Args:\n      other: Another `Dimension`.\n\n    Raises:\n      TypeError.\n    "
        raise TypeError("unsupported operand type(s) for /: 'Dimension' and '{}', please use // instead".format(type(other).__name__))

    def __rtruediv__(self, other):
        if False:
            return 10
        "Use `__floordiv__` via `x // y` instead.\n\n    This function exists only to have a better error message. Instead of:\n    `TypeError: unsupported operand type(s) for /: 'int' and 'Dimension'`,\n    this function will explicitly call for usage of `//` instead.\n\n    Args:\n      other: Another `Dimension`.\n\n    Raises:\n      TypeError.\n    "
        raise TypeError("unsupported operand type(s) for /: '{}' and 'Dimension', please use // instead".format(type(other).__name__))

    def __mod__(self, other):
        if False:
            i = 10
            return i + 15
        'Returns `self` modulo `other`.\n\n    Dimension modulo are computed as follows:\n\n    ```python\n    tf.compat.v1.Dimension(m)    % tf.compat.v1.Dimension(n)     ==\n    tf.compat.v1.Dimension(m % n)\n    tf.compat.v1.Dimension(m)    % tf.compat.v1.Dimension(None)  # equiv. to\n    tf.compat.v1.Dimension(None)\n    tf.compat.v1.Dimension(None) % tf.compat.v1.Dimension(n)     # equiv. to\n    tf.compat.v1.Dimension(None)\n    tf.compat.v1.Dimension(None) % tf.compat.v1.Dimension(None)  # equiv. to\n    tf.compat.v1.Dimension(None)\n    ```\n\n    Args:\n      other: Another Dimension, or a value accepted by `as_dimension`.\n\n    Returns:\n      A Dimension whose value is `self` modulo `other`.\n    '
        other = as_dimension(other)
        if self._value is None or other.value is None:
            return Dimension(None)
        else:
            return Dimension(self._value % other.value)

    def __rmod__(self, other):
        if False:
            return 10
        'Returns `other` modulo `self`.\n\n    Args:\n      other: Another Dimension, or a value accepted by `as_dimension`.\n\n    Returns:\n      A Dimension whose value is `other` modulo `self`.\n    '
        other = as_dimension(other)
        return other % self

    def __lt__(self, other):
        if False:
            print('Hello World!')
        'Returns True if `self` is known to be less than `other`.\n\n    Dimensions are compared as follows:\n\n    ```python\n    (tf.compat.v1.Dimension(m)    < tf.compat.v1.Dimension(n))    == (m < n)\n    (tf.compat.v1.Dimension(m)    < tf.compat.v1.Dimension(None)) == None\n    (tf.compat.v1.Dimension(None) < tf.compat.v1.Dimension(n))    == None\n    (tf.compat.v1.Dimension(None) < tf.compat.v1.Dimension(None)) == None\n    ```\n\n    Args:\n      other: Another Dimension.\n\n    Returns:\n      The value of `self.value < other.value` if both are known, otherwise\n      None.\n    '
        other = as_dimension(other)
        if self._value is None or other.value is None:
            return None
        else:
            return self._value < other.value

    def __le__(self, other):
        if False:
            return 10
        'Returns True if `self` is known to be less than or equal to `other`.\n\n    Dimensions are compared as follows:\n\n    ```python\n    (tf.compat.v1.Dimension(m)    <= tf.compat.v1.Dimension(n))    == (m <= n)\n    (tf.compat.v1.Dimension(m)    <= tf.compat.v1.Dimension(None)) == None\n    (tf.compat.v1.Dimension(None) <= tf.compat.v1.Dimension(n))    == None\n    (tf.compat.v1.Dimension(None) <= tf.compat.v1.Dimension(None)) == None\n    ```\n\n    Args:\n      other: Another Dimension.\n\n    Returns:\n      The value of `self.value <= other.value` if both are known, otherwise\n      None.\n    '
        other = as_dimension(other)
        if self._value is None or other.value is None:
            return None
        else:
            return self._value <= other.value

    def __gt__(self, other):
        if False:
            while True:
                i = 10
        'Returns True if `self` is known to be greater than `other`.\n\n    Dimensions are compared as follows:\n\n    ```python\n    (tf.compat.v1.Dimension(m)    > tf.compat.v1.Dimension(n))    == (m > n)\n    (tf.compat.v1.Dimension(m)    > tf.compat.v1.Dimension(None)) == None\n    (tf.compat.v1.Dimension(None) > tf.compat.v1.Dimension(n))    == None\n    (tf.compat.v1.Dimension(None) > tf.compat.v1.Dimension(None)) == None\n    ```\n\n    Args:\n      other: Another Dimension.\n\n    Returns:\n      The value of `self.value > other.value` if both are known, otherwise\n      None.\n    '
        other = as_dimension(other)
        if self._value is None or other.value is None:
            return None
        else:
            return self._value > other.value

    def __ge__(self, other):
        if False:
            print('Hello World!')
        'Returns True if `self` is known to be greater than or equal to `other`.\n\n    Dimensions are compared as follows:\n\n    ```python\n    (tf.compat.v1.Dimension(m)    >= tf.compat.v1.Dimension(n))    == (m >= n)\n    (tf.compat.v1.Dimension(m)    >= tf.compat.v1.Dimension(None)) == None\n    (tf.compat.v1.Dimension(None) >= tf.compat.v1.Dimension(n))    == None\n    (tf.compat.v1.Dimension(None) >= tf.compat.v1.Dimension(None)) == None\n    ```\n\n    Args:\n      other: Another Dimension.\n\n    Returns:\n      The value of `self.value >= other.value` if both are known, otherwise\n      None.\n    '
        other = as_dimension(other)
        if self._value is None or other.value is None:
            return None
        else:
            return self._value >= other.value

    def __reduce__(self):
        if False:
            print('Hello World!')
        return (Dimension, (self._value,))

def as_dimension(value):
    if False:
        return 10
    'Converts the given value to a Dimension.\n\n  A Dimension input will be returned unmodified.\n  An input of `None` will be converted to an unknown Dimension.\n  An integer input will be converted to a Dimension with that value.\n\n  Args:\n    value: The value to be converted.\n\n  Returns:\n    A Dimension corresponding to the given value.\n  '
    if isinstance(value, Dimension):
        return value
    else:
        return Dimension(value)

@tf_export('TensorShape')
class TensorShape(trace.TraceType, trace_type.Serializable):
    """Represents the shape of a `Tensor`.

  >>> t = tf.constant([[1,2,3],[4,5,6]])
  >>> t.shape
  TensorShape([2, 3])

  `TensorShape` is the *static* shape representation of a Tensor.
  During eager execution a Tensor always has a fully specified shape but
  when tracing a `tf.function` it may be one of the following:

  * *Fully-known shape:* has a known number of dimensions and a known size
    for each dimension. e.g. `TensorShape([16, 256])`
  * *Partially-known shape:* has a known number of dimensions, and an unknown
    size for one or more dimension. e.g. `TensorShape([None, 256])`
  * *Unknown shape:* has an unknown number of dimensions, and an unknown
    size in all dimensions. e.g. `TensorShape(None)`

  During function tracing `t.shape` will return a `TensorShape` object
  representing the shape of Tensor as it is known during tracing.
  This static representation will be partially defined in cases where the
  exact shape depends on the values within the tensors. To get the
  *dynamic* representation, please use `tf.shape(t)`
  which will return Tensor representing the fully defined shape of `t`.
  This way, you can express logic that manipulates the shapes of tensors by
  building other tensors that depend on the dynamic shape of `t`.

  Note: `tf.RaggedTensor.shape` also returns a `tf.TensorShape`,
  the lengths of any ragged dimensions are unknown (`None`).

  For example, this function prints the `TensorShape' (`t.shape`), when you
  trace the function, and returns a tensor `tf.shape(t)` for given input `t`:

  >>> @tf.function
  ... def get_dynamic_shape(t):
  ...   print("tracing...")
  ...   print(f"static shape is {t.shape}")
  ...   return tf.shape(t)

  Just calling the function traces it with a fully-specified static shape:

  >>> result = get_dynamic_shape(tf.constant([[1, 1, 1], [0, 0, 0]]))
  tracing...
  static shape is (2, 3)
  >>> result.numpy()
  array([2, 3], dtype=int32)

  But `tf.function` can also trace the function with a partially specified
  (or even unspecified) shape:

  >>> cf1 = get_dynamic_shape.get_concrete_function(tf.TensorSpec(
  ...                                               shape=[None, 2]))
  tracing...
  static shape is (None, 2)
  >>> cf1(tf.constant([[1., 0],[1, 0],[1, 0]])).numpy()
  array([3, 2], dtype=int32)

  >>> cf2 = get_dynamic_shape.get_concrete_function(tf.TensorSpec(shape=None))
  tracing...
  static shape is <unknown>
  >>> cf2(tf.constant([[[[[1., 0]]]]])).numpy()
  array([1, 1, 1, 1, 2], dtype=int32)

  If a tensor is produced by an operation of type `"Foo"`, its shape
  may be inferred if there is a registered shape function for
  `"Foo"`. See [Shape
  functions](https://www.tensorflow.org/guide/create_op#shape_functions_in_c)
  for details of shape functions and how to register them. Alternatively,
  you may set the shape explicitly using `tf.Tensor.ensure_shape`.
  """
    __slots__ = ['_dims']

    def __init__(self, dims):
        if False:
            return 10
        'Creates a new TensorShape with the given dimensions.\n\n    Args:\n      dims: A list of Dimensions, or None if the shape is unspecified.\n\n    Raises:\n      TypeError: If dims cannot be converted to a list of dimensions.\n    '
        if isinstance(dims, (tuple, list)):
            self._dims = tuple((as_dimension(d).value for d in dims))
        elif dims is None:
            self._dims = None
        elif isinstance(dims, tensor_shape_pb2.TensorShapeProto):
            if dims.unknown_rank:
                self._dims = None
            else:
                self._dims = tuple((dim.size if dim.size != -1 else None for dim in dims.dim))
        elif isinstance(dims, TensorShape):
            self._dims = dims._dims
        else:
            try:
                dims_iter = iter(dims)
            except TypeError:
                self._dims = (as_dimension(dims).value,)
            else:
                self._dims = []
                for d in dims_iter:
                    try:
                        self._dims.append(as_dimension(d).value)
                    except TypeError as e:
                        raise TypeError("Failed to convert '{0!r}' to a shape: '{1!r}'could not be converted to a dimension. A shape should either be single dimension (e.g. 10), or an iterable of dimensions (e.g. [1, 10, None]).".format(dims, d)) from e
                self._dims = tuple(self._dims)

    @property
    def _v2_behavior(self):
        if False:
            i = 10
            return i + 15
        if _TENSORSHAPE_V2_OVERRIDE is None:
            return tf2.enabled()
        return _TENSORSHAPE_V2_OVERRIDE

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        if self._v2_behavior:
            if self._dims is not None:
                return f'TensorShape({list(self._dims)})'
            else:
                return 'TensorShape(None)'
        else:
            return f'TensorShape({self.dims})'

    def __str__(self):
        if False:
            print('Hello World!')
        if self.rank is None:
            return '<unknown>'
        elif self.rank == 1:
            if self._v2_behavior:
                return '(%s,)' % self._dims[0]
            else:
                return '(%s,)' % self.dims[0]
        elif self._v2_behavior:
            return '(%s)' % ', '.join((str(d) for d in self._dims))
        else:
            return '(%s)' % ', '.join((str(d) for d in self.dims))

    @property
    def rank(self):
        if False:
            print('Hello World!')
        'Returns the rank of this shape, or None if it is unspecified.'
        if self._dims is not None:
            return len(self._dims)
        return None

    @property
    def dims(self):
        if False:
            while True:
                i = 10
        'Deprecated.  Returns list of dimensions for this shape.\n\n    Suggest `TensorShape.as_list` instead.\n\n    Returns:\n      A list containing `tf.compat.v1.Dimension`s, or None if the shape is\n      unspecified.\n    '
        if self._dims is None:
            return None
        return [as_dimension(d) for d in self._dims]

    @property
    def ndims(self):
        if False:
            return 10
        'Deprecated accessor for `rank`.'
        return self.rank

    def __len__(self):
        if False:
            while True:
                i = 10
        'Returns the rank of this shape, or raises ValueError if unspecified.'
        if self._dims is None:
            raise ValueError('Cannot take the length of shape with unknown rank.')
        return len(self._dims)

    def __bool__(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns True if this shape contains non-zero information.'
        return self._dims is not None
    __nonzero__ = __bool__

    def __iter__(self):
        if False:
            while True:
                i = 10
        'Returns `self.dims` if the rank is known, otherwise raises ValueError.'
        if self._dims is None:
            raise ValueError('Cannot iterate over a shape with unknown rank.')
        elif self._v2_behavior:
            return iter((d for d in self._dims))
        else:
            return iter((d for d in self.dims))

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        'Returns the value of a dimension or a shape, depending on the key.\n\n    Args:\n      key: If `key` is an integer, returns the dimension at that index;\n        otherwise if `key` is a slice, returns a TensorShape whose dimensions\n        are those selected by the slice from `self`.\n\n    Returns:\n      An integer if `key` is an integer, or a `TensorShape` if `key` is a\n      slice.\n\n    Raises:\n      ValueError: If `key` is a slice and `self` is completely unknown and\n        the step is set.\n    '
        if self._dims is not None:
            if isinstance(key, slice):
                return TensorShape(self._dims[key])
            elif self._v2_behavior:
                return self._dims[key]
            else:
                return self.dims[key]
        elif isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop
            if key.step is not None:
                raise ValueError('Steps are not yet handled')
            if stop is None:
                return unknown_shape()
            elif start < 0 or stop < 0:
                return unknown_shape()
            else:
                return unknown_shape(rank=stop - start)
        elif self._v2_behavior:
            return None
        else:
            return Dimension(None)

    def num_elements(self):
        if False:
            while True:
                i = 10
        'Returns the total number of elements, or none for incomplete shapes.'
        if self.is_fully_defined():
            return functools.reduce(operator.mul, self.as_list(), 1)
        else:
            return None

    def merge_with(self, other):
        if False:
            i = 10
            return i + 15
        'Returns a `TensorShape` combining the information in `self` and `other`.\n\n    The dimensions in `self` and `other` are merged element-wise,\n    according to the rules below:\n\n    ```python\n    Dimension(n).merge_with(Dimension(None)) == Dimension(n)\n    Dimension(None).merge_with(Dimension(n)) == Dimension(n)\n    Dimension(None).merge_with(Dimension(None)) == Dimension(None)\n    # raises ValueError for n != m\n    Dimension(n).merge_with(Dimension(m))\n    ```\n    >> ts = tf.TensorShape([1,2])\n    >> ot1 = tf.TensorShape([1,2])\n    >> ts.merge_with(ot).as_list()\n    [1,2]\n\n    >> ot2 = tf.TensorShape([1,None])\n    >> ts.merge_with(ot2).as_list()\n    [1,2]\n\n    >> ot3 = tf.TensorShape([None, None])\n    >> ot3.merge_with(ot2).as_list()\n    [1, None]\n\n    Args:\n      other: Another `TensorShape`.\n\n    Returns:\n      A `TensorShape` containing the combined information of `self` and\n      `other`.\n\n    Raises:\n      ValueError: If `self` and `other` are not compatible.\n    '
        other = as_shape(other)
        if self.dims is None:
            return other
        if other.dims is None:
            return self
        else:
            try:
                self.assert_same_rank(other)
                new_dims = [dim.merge_with(other_dim) for (dim, other_dim) in zip(self.dims, other.dims)]
                return TensorShape(new_dims)
            except ValueError:
                raise ValueError('Shapes %s and %s are not compatible' % (self, other))

    def __add__(self, other):
        if False:
            return 10
        return self.concatenate(other)

    def __radd__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, TensorShape):
            other = TensorShape(other)
        return other.concatenate(self)

    def concatenate(self, other):
        if False:
            while True:
                i = 10
        'Returns the concatenation of the dimension in `self` and `other`.\n\n    *N.B.* If either `self` or `other` is completely unknown,\n    concatenation will discard information about the other shape. In\n    future, we might support concatenation that preserves this\n    information for use with slicing.\n\n    Args:\n      other: Another `TensorShape`.\n\n    Returns:\n      A `TensorShape` whose dimensions are the concatenation of the\n      dimensions in `self` and `other`.\n    '
        other = as_shape(other)
        if self.dims is None or other.dims is None:
            return unknown_shape()
        else:
            return TensorShape(self.dims + other.dims)

    def assert_same_rank(self, other):
        if False:
            i = 10
            return i + 15
        'Raises an exception if `self` and `other` do not have compatible ranks.\n\n    Args:\n      other: Another `TensorShape`.\n\n    Raises:\n      ValueError: If `self` and `other` do not represent shapes with the\n        same rank.\n    '
        other = as_shape(other)
        if self.rank is not None and other.rank is not None:
            if self.rank != other.rank:
                raise ValueError('Shapes %s and %s must have the same rank' % (self, other))

    def assert_has_rank(self, rank):
        if False:
            for i in range(10):
                print('nop')
        'Raises an exception if `self` is not compatible with the given `rank`.\n\n    Args:\n      rank: An integer.\n\n    Raises:\n      ValueError: If `self` does not represent a shape with the given `rank`.\n    '
        if self.rank not in (None, rank):
            raise ValueError('Shape %s must have rank %d' % (self, rank))

    def with_rank(self, rank):
        if False:
            for i in range(10):
                print('nop')
        'Returns a shape based on `self` with the given rank.\n\n    This method promotes a completely unknown shape to one with a\n    known rank.\n\n    Args:\n      rank: An integer.\n\n    Returns:\n      A shape that is at least as specific as `self` with the given rank.\n\n    Raises:\n      ValueError: If `self` does not represent a shape with the given `rank`.\n    '
        try:
            return self.merge_with(unknown_shape(rank=rank))
        except ValueError:
            raise ValueError('Shape %s must have rank %d' % (self, rank))

    def with_rank_at_least(self, rank):
        if False:
            print('Hello World!')
        'Returns a shape based on `self` with at least the given rank.\n\n    Args:\n      rank: An integer.\n\n    Returns:\n      A shape that is at least as specific as `self` with at least the given\n      rank.\n\n    Raises:\n      ValueError: If `self` does not represent a shape with at least the given\n        `rank`.\n    '
        if self.rank is not None and self.rank < rank:
            raise ValueError('Shape %s must have rank at least %d' % (self, rank))
        else:
            return self

    def with_rank_at_most(self, rank):
        if False:
            for i in range(10):
                print('nop')
        'Returns a shape based on `self` with at most the given rank.\n\n    Args:\n      rank: An integer.\n\n    Returns:\n      A shape that is at least as specific as `self` with at most the given\n      rank.\n\n    Raises:\n      ValueError: If `self` does not represent a shape with at most the given\n        `rank`.\n    '
        if self.rank is not None and self.rank > rank:
            raise ValueError('Shape %s must have rank at most %d' % (self, rank))
        else:
            return self

    def is_subtype_of(self, other: trace.TraceType) -> bool:
        if False:
            print('Hello World!')
        'Returns True iff `self` is subtype of `other`.\n\n    Shape A is a subtype of shape B if shape B can successfully represent it:\n\n    * A `TensorShape` of any rank is a subtype of `TensorShape(None)`.\n\n    *  TensorShapes of equal ranks are covariant, i.e.\n      `TensorShape([A1, A2, ..])` is a subtype of\n      `TensorShape([B1, B2, ..])` iff An is a subtype of Bn.\n\n      An is subtype of Bn iff An == Bn or Bn is None.\n\n    * TensorShapes of different defined ranks have no subtyping relation.\n\n    The subtyping relation is reflexive and transitive, but not symmetric.\n\n    Some examples:\n    * `TensorShape([32, 784])` is a subtype of `TensorShape(None)`, and\n      `TensorShape([4, 4])` is also a subtype of `TensorShape(None)` but\n      `TensorShape([32, 784])` and `TensorShape([4, 4])` are not subtypes of\n      each other.\n\n    * All two-dimensional shapes are subtypes of `TensorShape([None, None])`,\n      such as `TensorShape([32, 784])`. There is no subtype relationship with,\n      for example, `TensorShape([None])` or `TensorShape([None, None, None])`.\n\n    * `TensorShape([32, None])` is also a subtype of `TensorShape([None, None])`\n      and `TensorShape(None)`. It is not a subtype of, for example,\n      `TensorShape([32])`, `TensorShape([32, None, 1])`,\n      `TensorShape([64, None])` or `TensorShape([None, 32])`.\n\n    * `TensorShape([32, 784])` is a subtype of itself, and also\n      `TensorShape([32, None])`, `TensorShape([None, 784])`,\n      `TensorShape([None, None])` and `TensorShape(None)`.\n      It has no subtype relation with, for example, `TensorShape([32, 1, 784])`\n      or `TensorShape([None])`.\n\n    Args:\n      other: Another `TensorShape`.\n\n    Returns:\n      True iff `self` is subtype of `other`.\n\n    '
        if not isinstance(other, TensorShape):
            return False
        if other.rank is None:
            return True
        if self.rank != other.rank:
            return False
        return all((o is None or s == o for (s, o) in zip(self._dims, other._dims)))

    def most_specific_common_supertype(self, others: Sequence[trace.TraceType]) -> Optional['TensorShape']:
        if False:
            for i in range(10):
                print('nop')
        'Returns the most specific supertype `TensorShape` of self and others.\n\n    * `TensorShape([None, 1])` is the most specific `TensorShape` supertyping\n      both `TensorShape([2, 1])` and `TensorShape([5, 1])`. Note that\n      `TensorShape(None)` is also a supertype but it is not "most specific".\n\n    * `TensorShape([1, 2, 3])` is the most specific `TensorShape` supertyping\n      both `TensorShape([1, 2, 3])` and `TensorShape([1, 2, 3]`). There are\n      other less specific TensorShapes that supertype above mentioned\n      TensorShapes, e.g. `TensorShape([1, 2, None])`, `TensorShape(None)`.\n\n     * `TensorShape([None, None])` is the most specific `TensorShape`\n       supertyping both `TensorShape([2, None])` and `TensorShape([None, 3])`.\n       As always, `TensorShape(None)` is also a supertype but not the most\n       specific one.\n\n     * `TensorShape(None`) is the only `TensorShape` supertyping both\n       `TensorShape([1, 2, 3])` and `TensorShape([1, 2])`. In general, any two\n       shapes that have different ranks will only have `TensorShape(None)`\n       as a common supertype.\n\n     * `TensorShape(None)` is the only `TensorShape` supertyping both\n       `TensorShape([1, 2, 3])` and `TensorShape(None)`. In general, the common\n       supertype of any shape with `TensorShape(None)` is `TensorShape(None)`.\n\n    Args:\n      others: Sequence of `TensorShape`.\n\n    Returns:\n      A `TensorShape` which is the most specific supertype shape of `self`\n      and `others`. None if it does not exist.\n    '
        if any((not isinstance(other, TensorShape) for other in others)):
            return None
        if self.rank is None:
            return unknown_shape()
        if any((other.dims is None or self.rank != other.rank for other in others)):
            return unknown_shape()
        dims = [dim if all((dim == other._dims[i] for other in others)) else None for (i, dim) in enumerate(self._dims)]
        return TensorShape(dims)

    @doc_controls.do_not_doc_inheritable
    def placeholder_value(self, placeholder_context):
        if False:
            i = 10
            return i + 15
        'See tf.types.experimental.TraceType base class.'
        return super().placeholder_value(placeholder_context)

    @doc_controls.do_not_doc_inheritable
    def from_tensors(self, tensors):
        if False:
            i = 10
            return i + 15
        'See tf.types.experimental.TraceType base class.'
        return super().from_tensors(tensors)

    @doc_controls.do_not_doc_inheritable
    def to_tensors(self, value):
        if False:
            print('Hello World!')
        'See tf.types.experimental.TraceType base class.'
        return super().to_tensors(value)

    @doc_controls.do_not_doc_inheritable
    def flatten(self):
        if False:
            i = 10
            return i + 15
        'See tf.types.experimental.TraceType base class.'
        return super().flatten()

    @doc_controls.do_not_doc_inheritable
    def cast(self, value, cast_context):
        if False:
            return 10
        'See tf.types.experimental.TraceType base class.'
        return super().cast(value, cast_context)

    @classmethod
    def experimental_type_proto(cls) -> Type[tensor_shape_pb2.TensorShapeProto]:
        if False:
            while True:
                i = 10
        'Returns the type of proto associated with TensorShape serialization.'
        return tensor_shape_pb2.TensorShapeProto

    @classmethod
    def experimental_from_proto(cls, proto: tensor_shape_pb2.TensorShapeProto) -> 'TensorShape':
        if False:
            print('Hello World!')
        'Returns a TensorShape instance based on the serialized proto.'
        return TensorShape(proto)

    def experimental_as_proto(self) -> tensor_shape_pb2.TensorShapeProto:
        if False:
            return 10
        'Returns a proto representation of the TensorShape instance.'
        return self.as_proto()

    def is_compatible_with(self, other):
        if False:
            print('Hello World!')
        'Returns True iff `self` is compatible with `other`.\n\n    Two possibly-partially-defined shapes are compatible if there\n    exists a fully-defined shape that both shapes can represent. Thus,\n    compatibility allows the shape inference code to reason about\n    partially-defined shapes. For example:\n\n    * TensorShape(None) is compatible with all shapes.\n\n    * TensorShape([None, None]) is compatible with all two-dimensional\n      shapes, such as TensorShape([32, 784]), and also TensorShape(None). It is\n      not compatible with, for example, TensorShape([None]) or\n      TensorShape([None, None, None]).\n\n    * TensorShape([32, None]) is compatible with all two-dimensional shapes\n      with size 32 in the 0th dimension, and also TensorShape([None, None])\n      and TensorShape(None). It is not compatible with, for example,\n      TensorShape([32]), TensorShape([32, None, 1]) or TensorShape([64, None]).\n\n    * TensorShape([32, 784]) is compatible with itself, and also\n      TensorShape([32, None]), TensorShape([None, 784]), TensorShape([None,\n      None]) and TensorShape(None). It is not compatible with, for example,\n      TensorShape([32, 1, 784]) or TensorShape([None]).\n\n    The compatibility relation is reflexive and symmetric, but not\n    transitive. For example, TensorShape([32, 784]) is compatible with\n    TensorShape(None), and TensorShape(None) is compatible with\n    TensorShape([4, 4]), but TensorShape([32, 784]) is not compatible with\n    TensorShape([4, 4]).\n\n    Args:\n      other: Another TensorShape.\n\n    Returns:\n      True iff `self` is compatible with `other`.\n\n    '
        other = as_shape(other)
        if self.dims is not None and other.dims is not None:
            if self.rank != other.rank:
                return False
            for (x_dim, y_dim) in zip(self.dims, other.dims):
                if not x_dim.is_compatible_with(y_dim):
                    return False
        return True

    def assert_is_compatible_with(self, other):
        if False:
            print('Hello World!')
        'Raises exception if `self` and `other` do not represent the same shape.\n\n    This method can be used to assert that there exists a shape that both\n    `self` and `other` represent.\n\n    Args:\n      other: Another TensorShape.\n\n    Raises:\n      ValueError: If `self` and `other` do not represent the same shape.\n    '
        if not self.is_compatible_with(other):
            raise ValueError('Shapes %s and %s are incompatible' % (self, other))

    def most_specific_compatible_shape(self, other) -> 'TensorShape':
        if False:
            return 10
        'Returns the most specific TensorShape compatible with `self` and `other`.\n\n    * TensorShape([None, 1]) is the most specific TensorShape compatible with\n      both TensorShape([2, 1]) and TensorShape([5, 1]). Note that\n      TensorShape(None) is also compatible with above mentioned TensorShapes.\n\n    * TensorShape([1, 2, 3]) is the most specific TensorShape compatible with\n      both TensorShape([1, 2, 3]) and TensorShape([1, 2, 3]). There are more\n      less specific TensorShapes compatible with above mentioned TensorShapes,\n      e.g. TensorShape([1, 2, None]), TensorShape(None).\n\n    Args:\n      other: Another `TensorShape`.\n\n    Returns:\n      A `TensorShape` which is the most specific compatible shape of `self`\n      and `other`.\n    '
        other = as_shape(other)
        if self.dims is None or other.dims is None or self.rank != other.rank:
            return unknown_shape()
        dims = [d1 if d1 is not None and d2 is not None and (d1 == d2) else None for (d1, d2) in zip(self.dims, other.dims)]
        return TensorShape(dims)

    def is_fully_defined(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns True iff `self` is fully defined in every dimension.'
        return self._dims is not None and all((dim is not None for dim in self._dims))

    def assert_is_fully_defined(self):
        if False:
            return 10
        'Raises an exception if `self` is not fully defined in every dimension.\n\n    Raises:\n      ValueError: If `self` does not have a known value for every dimension.\n    '
        if not self.is_fully_defined():
            raise ValueError('Shape %s is not fully defined' % self)

    def as_list(self):
        if False:
            while True:
                i = 10
        'Returns a list of integers or `None` for each dimension.\n\n    Returns:\n      A list of integers or `None` for each dimension.\n\n    Raises:\n      ValueError: If `self` is an unknown shape with an unknown rank.\n    '
        if self._dims is None:
            raise ValueError('as_list() is not defined on an unknown TensorShape.')
        return list(self._dims)

    def as_proto(self):
        if False:
            i = 10
            return i + 15
        'Returns this shape as a `TensorShapeProto`.'
        if self._dims is None:
            return tensor_shape_pb2.TensorShapeProto(unknown_rank=True)
        else:
            return tensor_shape_pb2.TensorShapeProto(dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=-1 if d is None else d) for d in self._dims])

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Returns True if `self` is equivalent to `other`.\n\n    It first tries to convert `other` to `TensorShape`. `TypeError` is thrown\n    when the conversion fails. Otherwise, it compares each element in the\n    TensorShape dimensions.\n\n    * Two *Fully known* shapes, return True iff each element is equal.\n    >>> t_a = tf.TensorShape([1,2])\n    >>> a = [1, 2]\n    >>> t_b = tf.TensorShape([1,2])\n    >>> t_c = tf.TensorShape([1,2,3])\n    >>> t_a.__eq__(a)\n    True\n    >>> t_a.__eq__(t_b)\n    True\n    >>> t_a.__eq__(t_c)\n    False\n\n    * Two *Partially-known* shapes, return True iff each element is equal.\n    >>> p_a = tf.TensorShape([1,None])\n    >>> p_b = tf.TensorShape([1,None])\n    >>> p_c = tf.TensorShape([2,None])\n    >>> p_a.__eq__(p_b)\n    True\n    >>> t_a.__eq__(p_a)\n    False\n    >>> p_a.__eq__(p_c)\n    False\n\n    * Two *Unknown shape*, return True.\n    >>> unk_a = tf.TensorShape(None)\n    >>> unk_b = tf.TensorShape(None)\n    >>> unk_a.__eq__(unk_b)\n    True\n    >>> unk_a.__eq__(t_a)\n    False\n\n    Args:\n      other: A `TensorShape` or type that can be converted to `TensorShape`.\n\n    Returns:\n      True if the dimensions are all equal.\n\n    Raises:\n      TypeError if `other` can not be converted to `TensorShape`.\n    '
        try:
            other = as_shape(other)
        except TypeError:
            return NotImplemented
        return self._dims == other._dims

    def __hash__(self):
        if False:
            return 10
        return hash(self._dims)

    def __reduce__(self):
        if False:
            print('Hello World!')
        return (TensorShape, (self.dims,))

    def __concat__(self, other):
        if False:
            i = 10
            return i + 15
        return self.concatenate(other)
trace_type.register_serializable(TensorShape)

class _TensorShapeCodec:
    """Codec for `TensorShape`."""

    def can_encode(self, pyobj):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(pyobj, TensorShape)

    def do_encode(self, tensor_shape_value, encode_fn):
        if False:
            return 10
        del encode_fn
        encoded_tensor_shape = struct_pb2.StructuredValue()
        encoded_tensor_shape.tensor_shape_value.CopyFrom(tensor_shape_value.as_proto())
        return encoded_tensor_shape

    def can_decode(self, value):
        if False:
            print('Hello World!')
        return value.HasField('tensor_shape_value')

    def do_decode(self, value, decode_fn):
        if False:
            while True:
                i = 10
        del decode_fn
        return TensorShape(value.tensor_shape_value)
nested_structure_coder.register_codec(_TensorShapeCodec())

def as_shape(shape) -> 'TensorShape':
    if False:
        for i in range(10):
            print('nop')
    'Converts the given object to a TensorShape.'
    if isinstance(shape, TensorShape):
        return shape
    else:
        return TensorShape(shape)

def unknown_shape(rank=None, **kwargs) -> 'TensorShape':
    if False:
        return 10
    'Returns an unknown TensorShape, optionally with a known rank.\n\n  Args:\n    rank: (Optional) If specified, the number of dimensions in the shape.\n    **kwargs: For backwards compatibility.\n\n  Returns:\n    An unknown TensorShape.\n\n  Raises:\n    TypeError: In case of invalid arguments.\n  '
    if rank is None and 'ndims' in kwargs:
        rank = kwargs.pop('ndims')
    if kwargs:
        raise TypeError('Unknown argument: %s' % kwargs)
    if rank is None:
        return TensorShape(None)
    else:
        return TensorShape([Dimension(None)] * rank)