"""Sparse tensors."""
import collections
import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python import tf2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import internal
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util.tf_export import tf_export
_eval_using_default_session = tensor._eval_using_default_session
_override_helper = tensor._override_helper

@tf_export('sparse.SparseTensor', 'SparseTensor')
class SparseTensor(internal.NativeObject, composite_tensor.CompositeTensor):
    """Represents a sparse tensor.

  TensorFlow represents a sparse tensor as three separate dense tensors:
  `indices`, `values`, and `dense_shape`.  In Python, the three tensors are
  collected into a `SparseTensor` class for ease of use.  If you have separate
  `indices`, `values`, and `dense_shape` tensors, wrap them in a `SparseTensor`
  object before passing to the ops below.

  Concretely, the sparse tensor `SparseTensor(indices, values, dense_shape)`
  comprises the following components, where `N` and `ndims` are the number
  of values and number of dimensions in the `SparseTensor`, respectively:

  * `indices`: A 2-D int64 tensor of shape `[N, ndims]`, which specifies the
    indices of the elements in the sparse tensor that contain nonzero values
    (elements are zero-indexed). For example, `indices=[[1,3], [2,4]]` specifies
    that the elements with indexes of [1,3] and [2,4] have nonzero values.

  * `values`: A 1-D tensor of any type and shape `[N]`, which supplies the
    values for each element in `indices`. For example, given `indices=[[1,3],
    [2,4]]`, the parameter `values=[18, 3.6]` specifies that element [1,3] of
    the sparse tensor has a value of 18, and element [2,4] of the tensor has a
    value of 3.6.

  * `dense_shape`: A 1-D int64 tensor of shape `[ndims]`, which specifies the
    dense_shape of the sparse tensor. Takes a list indicating the number of
    elements in each dimension. For example, `dense_shape=[3,6]` specifies a
    two-dimensional 3x6 tensor, `dense_shape=[2,3,4]` specifies a
    three-dimensional 2x3x4 tensor, and `dense_shape=[9]` specifies a
    one-dimensional tensor with 9 elements.

  The corresponding dense tensor satisfies:

  ```python
  dense.shape = dense_shape
  dense[tuple(indices[i])] = values[i]
  ```

  By convention, `indices` should be sorted in row-major order (or equivalently
  lexicographic order on the tuples `indices[i]`). This is not enforced when
  `SparseTensor` objects are constructed, but most ops assume correct ordering.
  If the ordering of sparse tensor `st` is wrong, a fixed version can be
  obtained by calling `tf.sparse.reorder(st)`.

  Example: The sparse tensor

  ```python
  SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
  ```

  represents the dense tensor

  ```python
  [[1, 0, 0, 0]
   [0, 0, 2, 0]
   [0, 0, 0, 0]]
  ```
  """

    @classmethod
    def from_value(cls, sparse_tensor_value):
        if False:
            while True:
                i = 10
        if not is_sparse(sparse_tensor_value):
            raise TypeError(f'Argument sparse_tensor_value={sparse_tensor_value} is neither a SparseTensor nor SparseTensorValue.')
        return SparseTensor(indices=sparse_tensor_value.indices, values=sparse_tensor_value.values, dense_shape=sparse_tensor_value.dense_shape)

    def __init__(self, indices, values, dense_shape):
        if False:
            return 10
        'Creates a `SparseTensor`.\n\n    Args:\n      indices: A 2-D int64 tensor of shape `[N, ndims]`.\n      values: A 1-D tensor of any type and shape `[N]`.\n      dense_shape: A 1-D int64 tensor of shape `[ndims]`.\n\n    Raises:\n      ValueError: When building an eager SparseTensor if `dense_shape` is\n        unknown or contains unknown elements (None or -1).\n    '
        with ops.name_scope(None, 'SparseTensor', [indices, values, dense_shape]):
            indices = ops.convert_to_tensor(indices, name='indices', dtype=dtypes.int64)
            values = ops.convert_to_tensor(values, name='values')
            dense_shape = ops.convert_to_tensor(dense_shape, name='dense_shape', dtype=dtypes.int64)
            dense_shape_default = tensor_util.constant_value_as_shape(dense_shape)
        self._indices = indices
        self._values = values
        self._dense_shape = dense_shape
        self._dense_shape_default = dense_shape_default
        indices_shape = indices.shape.with_rank(2)
        values_shape = values.shape.with_rank(1)
        dense_shape_shape = dense_shape.shape.with_rank(1)
        indices_shape.dims[0].assert_is_compatible_with(values_shape.dims[0])
        indices_shape.dims[1].assert_is_compatible_with(dense_shape_shape.dims[0])

    def get_shape(self) -> tensor_shape.TensorShape:
        if False:
            print('Hello World!')
        'Get the `TensorShape` representing the shape of the dense tensor.\n\n    Returns:\n      A `TensorShape` object.\n    '
        return self._dense_shape_default

    @property
    def indices(self):
        if False:
            for i in range(10):
                print('nop')
        'The indices of non-zero values in the represented dense tensor.\n\n    Returns:\n      A 2-D Tensor of int64 with dense_shape `[N, ndims]`, where `N` is the\n        number of non-zero values in the tensor, and `ndims` is the rank.\n    '
        return self._indices

    @property
    def values(self):
        if False:
            return 10
        'The non-zero values in the represented dense tensor.\n\n    Returns:\n      A 1-D Tensor of any data type.\n    '
        return self._values

    def with_values(self, new_values):
        if False:
            while True:
                i = 10
        'Returns a copy of `self` with `values` replaced by `new_values`.\n\n    This method produces a new `SparseTensor` that has the same nonzero\n    `indices` and same `dense_shape`, but updated values.\n\n    Args:\n      new_values: The values of the new `SparseTensor`. Needs to have the same\n        shape as the current `.values` `Tensor`. May have a different type than\n        the current `values`.\n\n    Returns:\n      A `SparseTensor` with identical indices and shape but updated values.\n\n    Example usage:\n\n    >>> st = tf.sparse.from_dense([[1, 0, 2, 0], [3, 0, 0, 4]])\n    >>> tf.sparse.to_dense(st.with_values([10, 20, 30, 40]))  # 4 nonzero values\n    <tf.Tensor: shape=(2, 4), dtype=int32, numpy=\n    array([[10,  0, 20,  0],\n           [30,  0,  0, 40]], dtype=int32)>\n\n    '
        return SparseTensor(self._indices, new_values, self._dense_shape)

    @property
    def op(self) -> ops.Operation:
        if False:
            for i in range(10):
                print('nop')
        'The `Operation` that produces `values` as an output.'
        return self._values.op

    @property
    def dtype(self):
        if False:
            print('Hello World!')
        'The `DType` of elements in this tensor.'
        return self._values.dtype

    @property
    def dense_shape(self):
        if False:
            print('Hello World!')
        'A 1-D Tensor of int64 representing the shape of the dense tensor.'
        return self._dense_shape

    @property
    def shape(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the `TensorShape` representing the shape of the dense tensor.\n\n    Returns:\n      A `TensorShape` object.\n    '
        return self._dense_shape_default

    def set_shape(self, shape):
        if False:
            i = 10
            return i + 15
        "Updates the `TensorShape` representing the shape of the dense tensor.\n\n    With eager execution this operates as a shape assertion.\n    Here the shapes match:\n\n    >>> st = tf.SparseTensor(\n    ...   indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])\n    >>> st.set_shape([3, 4])\n\n    Passing a `None` in the new shape allows any value for that axis:\n\n    >>> st.set_shape([3, None])\n\n    An error is raised if an incompatible shape is passed.\n\n    >>> st.set_shape([1, 4])\n    Traceback (most recent call last):\n    ...\n    ValueError: Tensor's shape (3, 4) is not compatible with supplied\n    shape [1, 4]\n\n    When executing in a `tf.function`, or building a model using\n    `tf.keras.Input`, `SparseTensor.set_shape` will *merge* the given `shape`\n    with the current shape of this tensor, and set the tensor's shape to the\n    merged value (see `tf.TensorShape.merge_with` for details):\n\n    >>> st = tf.keras.Input(shape=[None, None, 3], sparse=True)\n    >>> print(st.shape)\n    (None, None, None, 3)\n\n    Dimensions set to `None` are not updated:\n\n    >>> st.set_shape([None, 224, 224, None])\n    >>> print(st.shape)\n    (None, 224, 224, 3)\n\n    The main use case for this is to provide additional shape information\n    that cannot be inferred from the graph alone.\n\n    Caution: `set_shape` ensures that the applied shape is compatible with\n    the existing shape, but it does not check at runtime. Setting\n    incorrect shapes can result in inconsistencies between the\n    statically-known graph and the runtime value of tensors.\n\n    Args:\n      shape: A `TensorShape` representing the shape of this tensor, a\n        `TensorShapeProto`, a list, a tuple, or None.\n\n    Raises:\n      ValueError: If `shape` is not compatible with the current shape of\n        this tensor.\n    "
        if not isinstance(shape, tensor_shape.TensorShape):
            shape = tensor_shape.TensorShape(shape)
        self._dense_shape_default = self._dense_shape_default.merge_with(shape)

    @property
    def graph(self):
        if False:
            return 10
        'The `Graph` that contains the index, value, and dense_shape tensors.'
        return self._indices.graph

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'SparseTensor(indices=%s, values=%s, dense_shape=%s)' % (self._indices, self._values, self._dense_shape)

    def eval(self, feed_dict=None, session=None):
        if False:
            return 10
        'Evaluates this sparse tensor in a `Session`.\n\n    Calling this method will execute all preceding operations that\n    produce the inputs needed for the operation that produces this\n    tensor.\n\n    *N.B.* Before invoking `SparseTensor.eval()`, its graph must have been\n    launched in a session, and either a default session must be\n    available, or `session` must be specified explicitly.\n\n    Args:\n      feed_dict: A dictionary that maps `Tensor` objects to feed values. See\n        `tf.Session.run` for a description of the valid feed values.\n      session: (Optional.) The `Session` to be used to evaluate this sparse\n        tensor. If none, the default session will be used.\n\n    Returns:\n      A `SparseTensorValue` object.\n    '
        (indices, values, dense_shape) = _eval_using_default_session([self.indices, self.values, self.dense_shape], feed_dict, self.graph, session)
        return SparseTensorValue(indices, values, dense_shape)

    @staticmethod
    def _override_operator(operator, func):
        if False:
            for i in range(10):
                print('nop')
        _override_helper(SparseTensor, operator, func)

    @property
    def _type_spec(self):
        if False:
            while True:
                i = 10
        return SparseTensorSpec(self.shape, self.dtype)

    def _shape_invariant_to_type_spec(self, shape):
        if False:
            i = 10
            return i + 15
        if shape.ndims is not None and shape.ndims != 1:
            raise ValueError(f'Expected a shape with 1 dimension. Obtained: {shape} which has {shape.ndims} dimensions.')
        rank = tensor_shape.dimension_value(shape[0])
        return SparseTensorSpec(tensor_shape.unknown_shape(rank), self.dtype)

    def consumers(self):
        if False:
            return 10
        return self._consumers()

    def _numpy(self):
        if False:
            print('Hello World!')
        'Returns a numpy `array` with the values for this `SparseTensor`.\n\n    Requires that this `SparseTensor` was constructed in eager execution mode.\n    '
        if not self._is_eager():
            raise ValueError('SparseTensor.numpy() is only supported in eager mode.')
        arr = np.zeros(self.dense_shape, dtype=self.dtype.as_numpy_dtype())
        for (i, v) in zip(self.indices, self.values):
            arr[tuple(i)] = v
        return arr

    def _is_eager(self):
        if False:
            i = 10
            return i + 15
        'Returns True if this `SparseTensor` was constructed in eager execution.\n\n    Requires that each individual component of `SparseTensor`\n    (`indices`, `values` and `dense_shape`) is an instance of `EagerTensor`.\n    '
        return all((isinstance(t, ops.EagerTensor) for t in (self.indices, self.values, self.dense_shape)))
SparseTensorValue = collections.namedtuple('SparseTensorValue', ['indices', 'values', 'dense_shape'])
tf_export(v1=['SparseTensorValue'])(SparseTensorValue)
_pywrap_utils.RegisterType('SparseTensorValue', SparseTensorValue)

@tf_export('SparseTensorSpec')
@type_spec_registry.register('tf.SparseTensorSpec')
class SparseTensorSpec(type_spec.BatchableTypeSpec):
    """Type specification for a `tf.sparse.SparseTensor`."""
    __slots__ = ['_shape', '_dtype']
    value_type = property(lambda self: SparseTensor)

    def __init__(self, shape=None, dtype=dtypes.float32):
        if False:
            for i in range(10):
                print('nop')
        'Constructs a type specification for a `tf.sparse.SparseTensor`.\n\n    Args:\n      shape: The dense shape of the `SparseTensor`, or `None` to allow any dense\n        shape.\n      dtype: `tf.DType` of values in the `SparseTensor`.\n    '
        self._shape = tensor_shape.as_shape(shape)
        self._dtype = dtypes.as_dtype(dtype)

    def _serialize(self):
        if False:
            return 10
        return (self._shape, self._dtype)

    @property
    def dtype(self):
        if False:
            print('Hello World!')
        'The `tf.dtypes.DType` specified by this type for the SparseTensor.'
        return self._dtype

    @property
    def shape(self):
        if False:
            return 10
        'The `tf.TensorShape` specified by this type for the SparseTensor.'
        return self._shape

    @property
    def _component_specs(self):
        if False:
            return 10
        rank = self._shape.ndims
        num_values = None
        return [tensor_spec.TensorSpec([num_values, rank], dtypes.int64), tensor_spec.TensorSpec([num_values], self._dtype), tensor_spec.TensorSpec([rank], dtypes.int64)]

    def _to_components(self, value):
        if False:
            print('Hello World!')
        if isinstance(value, SparseTensorValue):
            value = SparseTensor.from_value(value)
        return [value.indices, value.values, value.dense_shape]

    def _from_components(self, tensor_list):
        if False:
            return 10
        if all((isinstance(t, np.ndarray) for t in tensor_list)) and (not tf2.enabled()):
            return SparseTensorValue(*tensor_list)
        else:
            result = SparseTensor(*tensor_list)
            result._dense_shape_default = result._dense_shape_default.merge_with(self._shape)
            return result

    @property
    def _flat_tensor_specs(self):
        if False:
            print('Hello World!')
        return [tensor_spec.TensorSpec(None, dtypes.variant)]

    def _to_tensor_list(self, value):
        if False:
            print('Hello World!')
        value = SparseTensor.from_value(value)
        return [gen_sparse_ops.serialize_sparse(value.indices, value.values, value.dense_shape, out_type=dtypes.variant)]

    def _to_batched_tensor_list(self, value):
        if False:
            return 10
        dense_shape = tensor_util.constant_value_as_shape(value.dense_shape)
        if self._shape.merge_with(dense_shape).ndims == 0:
            raise ValueError(f'Unbatching a sparse tensor is only supported for rank >= 1. Obtained input: {value}.')
        return [gen_sparse_ops.serialize_many_sparse(value.indices, value.values, value.dense_shape, out_type=dtypes.variant)]

    def _from_compatible_tensor_list(self, tensor_list):
        if False:
            i = 10
            return i + 15
        tensor_list = gen_sparse_ops.deserialize_sparse(tensor_list[0], self._dtype)
        (indices, values, dense_shape) = tensor_list
        rank = self._shape.ndims
        indices.set_shape([None, rank])
        if self._shape.is_fully_defined():
            dense_shape = ops.convert_to_tensor(self._shape, dtype=dtypes.int64, name='shape')
        elif self._shape.rank is not None and any((dim.value is not None for dim in self._shape.dims)):
            pieces = array_ops_stack.unstack(dense_shape, num=self._shape.rank)
            for (i, dim) in enumerate(self._shape.dims):
                if dim.value is not None:
                    pieces[i] = constant_op.constant(dim.value, dense_shape.dtype)
            dense_shape = array_ops_stack.stack(pieces)
        else:
            dense_shape.set_shape([rank])
        return SparseTensor(indices, values, dense_shape)

    def _batch(self, batch_size):
        if False:
            for i in range(10):
                print('nop')
        return SparseTensorSpec(tensor_shape.TensorShape([batch_size]).concatenate(self._shape), self._dtype)

    def _unbatch(self):
        if False:
            for i in range(10):
                print('nop')
        if self._shape.ndims == 0:
            raise ValueError('Unbatching a tensor is only supported for rank >= 1')
        return SparseTensorSpec(self._shape[1:], self._dtype)

    def _to_legacy_output_types(self):
        if False:
            for i in range(10):
                print('nop')
        return self._dtype

    def _to_legacy_output_shapes(self):
        if False:
            print('Hello World!')
        return self._shape

    def _to_legacy_output_classes(self):
        if False:
            print('Hello World!')
        return SparseTensor

    @classmethod
    def from_value(cls, value):
        if False:
            i = 10
            return i + 15
        if isinstance(value, SparseTensor):
            return cls(value.shape, value.dtype)
        if isinstance(value, SparseTensorValue):
            if isinstance(value.values, np.ndarray):
                return cls(value.dense_shape, value.values.dtype)
            else:
                return cls.from_value(SparseTensor.from_value(value))
        else:
            raise TypeError(f'Expected SparseTensor or SparseTensorValue. Received: {value} of type {type(value).__name__}.')
nested_structure_coder.register_codec(nested_structure_coder.BuiltInTypeSpecCodec(SparseTensorSpec, struct_pb2.TypeSpecProto.SPARSE_TENSOR_SPEC))
type_spec.register_type_spec_from_value_converter(SparseTensor, SparseTensorSpec.from_value)
type_spec.register_type_spec_from_value_converter(SparseTensorValue, SparseTensorSpec.from_value)

@tf_export(v1=['convert_to_tensor_or_sparse_tensor'])
def convert_to_tensor_or_sparse_tensor(value, dtype=None, name=None):
    if False:
        return 10
    'Converts value to a `SparseTensor` or `Tensor`.\n\n  Args:\n    value: A `SparseTensor`, `SparseTensorValue`, or an object whose type has a\n      registered `Tensor` conversion function.\n    dtype: Optional element type for the returned tensor. If missing, the type\n      is inferred from the type of `value`.\n    name: Optional name to use if a new `Tensor` is created.\n\n  Returns:\n    A `SparseTensor` or `Tensor` based on `value`.\n\n  Raises:\n    RuntimeError: If result type is incompatible with `dtype`.\n  '
    if dtype is not None:
        dtype = dtypes.as_dtype(dtype)
    if isinstance(value, SparseTensorValue):
        value = SparseTensor.from_value(value)
    if isinstance(value, SparseTensor):
        if dtype and (not dtype.is_compatible_with(value.dtype)):
            raise RuntimeError(f'Sparse dtype mismatch. Requested: {dtype.name},  Actual: {value.dtype.name}')
        return value
    return ops.convert_to_tensor(value, dtype=dtype, name=name)

def is_sparse(x):
    if False:
        i = 10
        return i + 15
    'Check whether `x` is sparse.\n\n  Check whether an object is a `tf.sparse.SparseTensor` or\n  `tf.compat.v1.SparseTensorValue`.\n\n  Args:\n    x: A python object to check.\n\n  Returns:\n    `True` iff `x` is a `tf.sparse.SparseTensor` or\n    `tf.compat.v1.SparseTensorValue`.\n  '
    return isinstance(x, (SparseTensor, SparseTensorValue))