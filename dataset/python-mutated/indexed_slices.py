"""Indexed slices."""
import collections
import warnings
import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import internal
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export

class IndexedSlicesCompositeTensorGradient(composite_tensor_gradient.CompositeTensorGradient):
    """CompositeTensorGradient for IndexedSlices."""

    def get_gradient_components(self, value):
        if False:
            print('Hello World!')
        return value

    def replace_gradient_components(self, value, component_grads):
        if False:
            print('Hello World!')
        return component_grads

@tf_export('IndexedSlices')
class IndexedSlices(internal.IndexedSlices, internal.NativeObject, composite_tensor.CompositeTensor):
    """A sparse representation of a set of tensor slices at given indices.

  This class is a simple wrapper for a pair of `Tensor` objects:

  * `values`: A `Tensor` of any dtype with shape `[D0, D1, ..., Dn]`.
  * `indices`: A 1-D integer `Tensor` with shape `[D0]`.

  An `IndexedSlices` is typically used to represent a subset of a larger
  tensor `dense` of shape `[LARGE0, D1, .. , DN]` where `LARGE0 >> D0`.
  The values in `indices` are the indices in the first dimension of
  the slices that have been extracted from the larger tensor.

  The dense tensor `dense` represented by an `IndexedSlices` `slices` has

  ```python
  dense[slices.indices[i], :, :, :, ...] = slices.values[i, :, :, :, ...]
  ```

  The `IndexedSlices` class is used principally in the definition of
  gradients for operations that have sparse gradients
  (e.g. `tf.gather`).

  >>> v = tf.Variable([[0.,1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8]])
  >>> with tf.GradientTape() as tape:
  ...   r = tf.gather(v, [1,3])
  >>> index_slices = tape.gradient(r,v)
  >>> index_slices
  <...IndexedSlices object ...>
  >>> index_slices.indices.numpy()
  array([1, 3], dtype=int32)
  >>> index_slices.values.numpy()
  array([[1., 1., 1.],
         [1., 1., 1.]], dtype=float32)

  Contrast this representation with
  `tf.sparse.SparseTensor`,
  which uses multi-dimensional indices and scalar values.
  """

    def __init__(self, values, indices, dense_shape=None):
        if False:
            i = 10
            return i + 15
        'Creates an `IndexedSlices`.'
        self._values = values
        self._indices = indices
        self._dense_shape = dense_shape

    @property
    def values(self):
        if False:
            while True:
                i = 10
        'A `Tensor` containing the values of the slices.'
        return self._values

    @property
    def indices(self):
        if False:
            for i in range(10):
                print('nop')
        'A 1-D `Tensor` containing the indices of the slices.'
        return self._indices

    @property
    def dense_shape(self):
        if False:
            i = 10
            return i + 15
        'A 1-D `Tensor` containing the shape of the corresponding dense tensor.'
        return self._dense_shape

    @property
    def shape(self):
        if False:
            for i in range(10):
                print('nop')
        'Gets the `tf.TensorShape` representing the shape of the dense tensor.\n\n    Returns:\n      A `tf.TensorShape` object.\n    '
        if self._dense_shape is None:
            return tensor_shape.TensorShape(None)
        return tensor_util.constant_value_as_shape(self._dense_shape)

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        'The name of this `IndexedSlices`.'
        return self.values.name

    @property
    def device(self):
        if False:
            return 10
        'The name of the device on which `values` will be produced, or `None`.'
        return self.values.device

    @property
    def op(self) -> ops.Operation:
        if False:
            print('Hello World!')
        'The `Operation` that produces `values` as an output.'
        return self.values.op

    @property
    def dtype(self):
        if False:
            while True:
                i = 10
        'The `DType` of elements in this tensor.'
        return self.values.dtype

    @property
    def graph(self) -> ops.Graph:
        if False:
            print('Hello World!')
        'The `Graph` that contains the values, indices, and shape tensors.'
        return self._values.graph

    def __str__(self):
        if False:
            return 10
        return 'IndexedSlices(indices=%s, values=%s%s)' % (self._indices, self._values, ', dense_shape=%s' % (self._dense_shape,) if self._dense_shape is not None else '')

    def __neg__(self):
        if False:
            while True:
                i = 10
        return IndexedSlices(-self.values, self.indices, self.dense_shape)
    __composite_gradient__ = IndexedSlicesCompositeTensorGradient()

    @property
    def _type_spec(self):
        if False:
            return 10
        indices_shape = self._indices.shape.merge_with(self._values.shape[:1])
        dense_shape = tensor_shape.TensorShape([None]).concatenate(self._values.shape[1:])
        if self._dense_shape is not None:
            dense_shape_dtype = self._dense_shape.dtype
            dense_shape = dense_shape.merge_with(tensor_util.constant_value_as_shape(self._dense_shape))
        else:
            dense_shape_dtype = None
        return IndexedSlicesSpec(dense_shape, self.dtype, self._indices.dtype, dense_shape_dtype, indices_shape)

    def _shape_invariant_to_type_spec(self, shape):
        if False:
            for i in range(10):
                print('nop')
        indices_shape = shape[:1]
        dense_shape = tensor_shape.TensorShape([None]).concatenate(shape[1:])
        if self._dense_shape is None:
            dense_shape_dtype = None
        else:
            dense_shape_dtype = self._dense_shape.dtype
        return IndexedSlicesSpec(dense_shape, self.dtype, self._indices.dtype, dense_shape_dtype, indices_shape)

    def consumers(self):
        if False:
            while True:
                i = 10
        return self._consumers()
IndexedSlicesValue = collections.namedtuple('IndexedSlicesValue', ['values', 'indices', 'dense_shape'])

@tf_export('IndexedSlicesSpec')
class IndexedSlicesSpec(type_spec.TypeSpec):
    """Type specification for a `tf.IndexedSlices`."""
    __slots__ = ['_shape', '_values_dtype', '_indices_dtype', '_dense_shape_dtype', '_indices_shape']
    value_type = property(lambda self: IndexedSlices)

    def __init__(self, shape=None, dtype=dtypes.float32, indices_dtype=dtypes.int64, dense_shape_dtype=None, indices_shape=None):
        if False:
            print('Hello World!')
        'Constructs a type specification for a `tf.IndexedSlices`.\n\n    Args:\n      shape: The dense shape of the `IndexedSlices`, or `None` to allow any\n        dense shape.\n      dtype: `tf.DType` of values in the `IndexedSlices`.\n      indices_dtype: `tf.DType` of the `indices` in the `IndexedSlices`.  One\n        of `tf.int32` or `tf.int64`.\n      dense_shape_dtype: `tf.DType` of the `dense_shape` in the `IndexedSlices`.\n        One of `tf.int32`, `tf.int64`, or `None` (if the `IndexedSlices` has\n        no `dense_shape` tensor).\n      indices_shape: The shape of the `indices` component, which indicates\n        how many slices are in the `IndexedSlices`.\n    '
        self._shape = tensor_shape.as_shape(shape)
        self._values_dtype = dtypes.as_dtype(dtype)
        self._indices_dtype = dtypes.as_dtype(indices_dtype)
        if dense_shape_dtype is None:
            self._dense_shape_dtype = None
        else:
            self._dense_shape_dtype = dtypes.as_dtype(dense_shape_dtype)
        self._indices_shape = tensor_shape.as_shape(indices_shape).with_rank(1)

    def _serialize(self):
        if False:
            i = 10
            return i + 15
        return (self._shape, self._values_dtype, self._indices_dtype, self._dense_shape_dtype, self._indices_shape)

    @property
    def _component_specs(self):
        if False:
            return 10
        value_shape = self._indices_shape.concatenate(self._shape[1:])
        specs = [tensor_spec.TensorSpec(value_shape, self._values_dtype), tensor_spec.TensorSpec(self._indices_shape, self._indices_dtype)]
        if self._dense_shape_dtype is not None:
            specs.append(tensor_spec.TensorSpec([self._shape.ndims], self._dense_shape_dtype))
        return tuple(specs)

    def _to_components(self, value):
        if False:
            return 10
        if value.dense_shape is None:
            return (value.values, value.indices)
        else:
            return (value.values, value.indices, value.dense_shape)

    def _from_components(self, tensor_list):
        if False:
            while True:
                i = 10
        if all((isinstance(t, np.ndarray) for t in tensor_list)) and (not tf2.enabled()):
            if len(tensor_list) == 2:
                return IndexedSlicesValue(tensor_list[0], tensor_list[1], None)
            else:
                return IndexedSlicesValue(*tensor_list)
        else:
            return IndexedSlices(*tensor_list)
nested_structure_coder.register_codec(nested_structure_coder.BuiltInTypeSpecCodec(IndexedSlicesSpec, struct_pb2.TypeSpecProto.INDEXED_SLICES_SPEC))

@tf_export(v1=['convert_to_tensor_or_indexed_slices'])
def convert_to_tensor_or_indexed_slices(value, dtype=None, name=None):
    if False:
        return 10
    'Converts the given object to a `Tensor` or an `IndexedSlices`.\n\n  If `value` is an `IndexedSlices` or `SparseTensor` it is returned\n  unmodified. Otherwise, it is converted to a `Tensor` using\n  `convert_to_tensor()`.\n\n  Args:\n    value: An `IndexedSlices`, `SparseTensor`, or an object that can be consumed\n      by `convert_to_tensor()`.\n    dtype: (Optional.) The required `DType` of the returned `Tensor` or\n      `IndexedSlices`.\n    name: (Optional.) A name to use if a new `Tensor` is created.\n\n  Returns:\n    A `Tensor`, `IndexedSlices`, or `SparseTensor` based on `value`.\n\n  Raises:\n    ValueError: If `dtype` does not match the element type of `value`.\n  '
    return internal_convert_to_tensor_or_indexed_slices(value=value, dtype=dtype, name=name, as_ref=False)

def internal_convert_to_tensor_or_indexed_slices(value, dtype=None, name=None, as_ref=False):
    if False:
        for i in range(10):
            print('nop')
    'Converts the given object to a `Tensor` or an `IndexedSlices`.\n\n  If `value` is an `IndexedSlices` or `SparseTensor` it is returned\n  unmodified. Otherwise, it is converted to a `Tensor` using\n  `convert_to_tensor()`.\n\n  Args:\n    value: An `IndexedSlices`, `SparseTensor`, or an object that can be consumed\n      by `convert_to_tensor()`.\n    dtype: (Optional.) The required `DType` of the returned `Tensor` or\n      `IndexedSlices`.\n    name: (Optional.) A name to use if a new `Tensor` is created.\n    as_ref: True if the caller wants the results as ref tensors.\n\n  Returns:\n    A `Tensor`, `IndexedSlices`, or `SparseTensor` based on `value`.\n\n  Raises:\n    ValueError: If `dtype` does not match the element type of `value`.\n  '
    if isinstance(value, ops.EagerTensor) and (not context.executing_eagerly()):
        return ops.convert_to_tensor(value, dtype=dtype, name=name, as_ref=as_ref)
    elif isinstance(value, internal.NativeObject):
        if dtype and (not dtypes.as_dtype(dtype).is_compatible_with(value.dtype)):
            raise ValueError(f'Incompatible tensor conversion requested to `dtype` {dtypes.as_dtype(dtype).name} for `value` ({value}) with dtype {value.dtype.name}.')
        return value
    else:
        return ops.convert_to_tensor(value, dtype=dtype, name=name, as_ref=as_ref)

def internal_convert_n_to_tensor_or_indexed_slices(values, dtype=None, name=None, as_ref=False):
    if False:
        return 10
    "Converts `values` to a list of `Tensor` or `IndexedSlices` objects.\n\n  Any `IndexedSlices` or `SparseTensor` objects in `values` are returned\n  unmodified.\n\n  Args:\n    values: An iterable of `None`, `IndexedSlices`, `SparseTensor`, or objects\n      that can be consumed by `convert_to_tensor()`.\n    dtype: (Optional.) The required `DType` of the returned `Tensor` or\n      `IndexedSlices`.\n    name: (Optional.) A name prefix to used when a new `Tensor` is created, in\n      which case element `i` will be given the name `name + '_' + i`.\n    as_ref: True if the caller wants the results as ref tensors.\n\n  Returns:\n    A list of `Tensor`, `IndexedSlices`, `SparseTensor` and/or `None` objects.\n\n  Raises:\n    TypeError: If no conversion function is registered for an element in\n      `values`.\n    RuntimeError: If a registered conversion function returns an invalid\n      value.\n  "
    if not isinstance(values, collections_abc.Iterable):
        raise TypeError('Argument `values` must be iterable.')
    ret = []
    for (i, value) in enumerate(values):
        if value is None:
            ret.append(value)
        else:
            n = None if name is None else '%s_%d' % (name, i)
            ret.append(internal_convert_to_tensor_or_indexed_slices(value, dtype=dtype, name=n, as_ref=as_ref))
    return ret

def convert_n_to_tensor_or_indexed_slices(values, dtype=None, name=None):
    if False:
        i = 10
        return i + 15
    "Converts `values` to a list of `Output` or `IndexedSlices` objects.\n\n  Any `IndexedSlices` or `SparseTensor` objects in `values` are returned\n  unmodified.\n\n  Args:\n    values: A list of `None`, `IndexedSlices`, `SparseTensor`, or objects that\n      can be consumed by `convert_to_tensor()`.\n    dtype: (Optional.) The required `DType` of the returned `Tensor`\n      `IndexedSlices`.\n    name: (Optional.) A name prefix to used when a new `Tensor` is created, in\n      which case element `i` will be given the name `name + '_' + i`.\n\n  Returns:\n    A list of `Tensor`, `IndexedSlices`, and/or `SparseTensor` objects.\n\n  Raises:\n    TypeError: If no conversion function is registered for an element in\n      `values`.\n    RuntimeError: If a registered conversion function returns an invalid\n      value.\n  "
    return internal_convert_n_to_tensor_or_indexed_slices(values=values, dtype=dtype, name=name, as_ref=False)
_LARGE_SPARSE_NUM_ELEMENTS = 100000000

def _indexed_slices_to_tensor(value, dtype=None, name=None, as_ref=False):
    if False:
        while True:
            i = 10
    'Converts an IndexedSlices object `value` to a Tensor.\n\n  NOTE(mrry): This function is potentially expensive.\n\n  Args:\n    value: An ops.IndexedSlices object.\n    dtype: The dtype of the Tensor to be returned.\n    name: Optional name to use for the returned Tensor.\n    as_ref: True if a ref is requested.\n\n  Returns:\n    A dense Tensor representing the values in the given IndexedSlices.\n\n  Raises:\n    ValueError: If the IndexedSlices does not have the same dtype.\n  '
    _ = as_ref
    if dtype and (not dtype.is_compatible_with(value.dtype)):
        raise ValueError(f'Incompatible tensor conversion requested to `dtype` {dtype.name} for IndexedSlices ({value}) with dtype {value.dtype.name}')
    if value.dense_shape is None:
        raise ValueError(f'Tensor conversion requested for IndexedSlices for argument `value` without dense_shape: {value!s}')
    if not context.executing_eagerly():
        dense_shape_value = tensor_util.constant_value(value.dense_shape)
        if dense_shape_value is not None:
            num_elements = np.prod(dense_shape_value)
            if num_elements >= _LARGE_SPARSE_NUM_ELEMENTS:
                warnings.warn('Converting sparse IndexedSlices to a dense Tensor with %d elements. This may consume a large amount of memory.' % num_elements)
    return gen_math_ops.unsorted_segment_sum(value.values, value.indices, value.dense_shape[0], name=name)
tensor_conversion_registry.register_tensor_conversion_function(IndexedSlices, _indexed_slices_to_tensor)