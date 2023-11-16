"""A type for representing values that may or may not exist."""
import abc
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.data.util import structure
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import gen_optional_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

@tf_export('experimental.Optional', 'data.experimental.Optional')
@deprecation.deprecated_endpoints('data.experimental.Optional')
class Optional(composite_tensor.CompositeTensor, metaclass=abc.ABCMeta):
    """Represents a value that may or may not be present.

  A `tf.experimental.Optional` can represent the result of an operation that may
  fail as a value, rather than raising an exception and halting execution. For
  example, `tf.data.Iterator.get_next_as_optional()` returns a
  `tf.experimental.Optional` that either contains the next element of an
  iterator if one exists, or an "empty" value that indicates the end of the
  sequence has been reached.

  `tf.experimental.Optional` can only be used with values that are convertible
  to `tf.Tensor` or `tf.CompositeTensor`.

  One can create a `tf.experimental.Optional` from a value using the
  `from_value()` method:

  >>> optional = tf.experimental.Optional.from_value(42)
  >>> print(optional.has_value())
  tf.Tensor(True, shape=(), dtype=bool)
  >>> print(optional.get_value())
  tf.Tensor(42, shape=(), dtype=int32)

  or without a value using the `empty()` method:

  >>> optional = tf.experimental.Optional.empty(
  ...   tf.TensorSpec(shape=(), dtype=tf.int32, name=None))
  >>> print(optional.has_value())
  tf.Tensor(False, shape=(), dtype=bool)
  """

    @abc.abstractmethod
    def has_value(self, name=None):
        if False:
            print('Hello World!')
        'Returns a tensor that evaluates to `True` if this optional has a value.\n\n    >>> optional = tf.experimental.Optional.from_value(42)\n    >>> print(optional.has_value())\n    tf.Tensor(True, shape=(), dtype=bool)\n\n    Args:\n      name: (Optional.) A name for the created operation.\n\n    Returns:\n      A scalar `tf.Tensor` of type `tf.bool`.\n    '
        raise NotImplementedError('Optional.has_value()')

    @abc.abstractmethod
    def get_value(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns the value wrapped by this optional.\n\n    If this optional does not have a value (i.e. `self.has_value()` evaluates to\n    `False`), this operation will raise `tf.errors.InvalidArgumentError` at\n    runtime.\n\n    >>> optional = tf.experimental.Optional.from_value(42)\n    >>> print(optional.get_value())\n    tf.Tensor(42, shape=(), dtype=int32)\n\n    Args:\n      name: (Optional.) A name for the created operation.\n\n    Returns:\n      The wrapped value.\n    '
        raise NotImplementedError('Optional.get_value()')

    @abc.abstractproperty
    def element_spec(self):
        if False:
            while True:
                i = 10
        'The type specification of an element of this optional.\n\n    >>> optional = tf.experimental.Optional.from_value(42)\n    >>> print(optional.element_spec)\n    tf.TensorSpec(shape=(), dtype=tf.int32, name=None)\n\n    Returns:\n      A (nested) structure of `tf.TypeSpec` objects matching the structure of an\n      element of this optional, specifying the type of individual components.\n    '
        raise NotImplementedError('Optional.element_spec')

    @staticmethod
    def empty(element_spec):
        if False:
            for i in range(10):
                print('nop')
        'Returns an `Optional` that has no value.\n\n    NOTE: This method takes an argument that defines the structure of the value\n    that would be contained in the returned `Optional` if it had a value.\n\n    >>> optional = tf.experimental.Optional.empty(\n    ...   tf.TensorSpec(shape=(), dtype=tf.int32, name=None))\n    >>> print(optional.has_value())\n    tf.Tensor(False, shape=(), dtype=bool)\n\n    Args:\n      element_spec: A (nested) structure of `tf.TypeSpec` objects matching the\n        structure of an element of this optional.\n\n    Returns:\n      A `tf.experimental.Optional` with no value.\n    '
        return _OptionalImpl(gen_optional_ops.optional_none(), element_spec)

    @staticmethod
    def from_value(value):
        if False:
            for i in range(10):
                print('nop')
        'Returns a `tf.experimental.Optional` that wraps the given value.\n\n    >>> optional = tf.experimental.Optional.from_value(42)\n    >>> print(optional.has_value())\n    tf.Tensor(True, shape=(), dtype=bool)\n    >>> print(optional.get_value())\n    tf.Tensor(42, shape=(), dtype=int32)\n\n    Args:\n      value: A value to wrap. The value must be convertible to `tf.Tensor` or\n        `tf.CompositeTensor`.\n\n    Returns:\n      A `tf.experimental.Optional` that wraps `value`.\n    '
        with ops.name_scope('optional') as scope:
            with ops.name_scope('value'):
                element_spec = structure.type_spec_from_value(value)
                encoded_value = structure.to_tensor_list(element_spec, value)
        return _OptionalImpl(gen_optional_ops.optional_from_value(encoded_value, name=scope), element_spec)

class _OptionalImpl(Optional):
    """Concrete implementation of `tf.experimental.Optional`.

  NOTE(mrry): This implementation is kept private, to avoid defining
  `Optional.__init__()` in the public API.
  """

    def __init__(self, variant_tensor, element_spec):
        if False:
            while True:
                i = 10
        super().__init__()
        self._variant_tensor = variant_tensor
        self._element_spec = element_spec

    def has_value(self, name=None):
        if False:
            return 10
        with ops.colocate_with(self._variant_tensor):
            return gen_optional_ops.optional_has_value(self._variant_tensor, name=name)

    def get_value(self, name=None):
        if False:
            return 10
        with ops.name_scope(name, 'OptionalGetValue', [self._variant_tensor]) as scope:
            with ops.colocate_with(self._variant_tensor):
                result = gen_optional_ops.optional_get_value(self._variant_tensor, name=scope, output_types=structure.get_flat_tensor_types(self._element_spec), output_shapes=structure.get_flat_tensor_shapes(self._element_spec))
            return structure.from_tensor_list(self._element_spec, result)

    @property
    def element_spec(self):
        if False:
            return 10
        return self._element_spec

    @property
    def _type_spec(self):
        if False:
            print('Hello World!')
        return OptionalSpec.from_value(self)

@tf_export('OptionalSpec', v1=['OptionalSpec', 'data.experimental.OptionalStructure'])
class OptionalSpec(type_spec.TypeSpec):
    """Type specification for `tf.experimental.Optional`.

  For instance, `tf.OptionalSpec` can be used to define a tf.function that takes
  `tf.experimental.Optional` as an input argument:

  >>> @tf.function(input_signature=[tf.OptionalSpec(
  ...   tf.TensorSpec(shape=(), dtype=tf.int32, name=None))])
  ... def maybe_square(optional):
  ...   if optional.has_value():
  ...     x = optional.get_value()
  ...     return x * x
  ...   return -1
  >>> optional = tf.experimental.Optional.from_value(5)
  >>> print(maybe_square(optional))
  tf.Tensor(25, shape=(), dtype=int32)

  Attributes:
    element_spec: A (nested) structure of `TypeSpec` objects that represents the
      type specification of the optional element.
  """
    __slots__ = ['_element_spec']

    def __init__(self, element_spec):
        if False:
            while True:
                i = 10
        super().__init__()
        self._element_spec = element_spec

    @property
    def value_type(self):
        if False:
            i = 10
            return i + 15
        return _OptionalImpl

    def _serialize(self):
        if False:
            for i in range(10):
                print('nop')
        return (self._element_spec,)

    @property
    def _component_specs(self):
        if False:
            while True:
                i = 10
        return [tensor_spec.TensorSpec((), dtypes.variant)]

    def _to_components(self, value):
        if False:
            while True:
                i = 10
        return [value._variant_tensor]

    def _from_components(self, flat_value):
        if False:
            for i in range(10):
                print('nop')
        return _OptionalImpl(flat_value[0], self._element_spec)

    @staticmethod
    def from_value(value):
        if False:
            while True:
                i = 10
        return OptionalSpec(value.element_spec)

    def _to_legacy_output_types(self):
        if False:
            return 10
        return self

    def _to_legacy_output_shapes(self):
        if False:
            print('Hello World!')
        return self

    def _to_legacy_output_classes(self):
        if False:
            while True:
                i = 10
        return self
nested_structure_coder.register_codec(nested_structure_coder.BuiltInTypeSpecCodec(OptionalSpec, struct_pb2.TypeSpecProto.OPTIONAL_SPEC))