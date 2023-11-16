"""Operations that generate constants.

See the [constants guide](https://tensorflow.org/api_guides/python/constant_op).
"""
from typing import Union
import numpy as np
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.profiler import trace
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util.tf_export import tf_export

def _eager_reshape(tensor, shape, ctx):
    if False:
        while True:
            i = 10
    'Eager-only version of Reshape op; requires tensor is an eager Tensor.'
    attr_t = tensor._datatype_enum()
    (attr_tshape, (shape,)) = execute.args_to_matching_eager([shape], ctx, [dtypes.int32, dtypes.int64], dtypes.int32)
    inputs_flat = [tensor, shape]
    attrs = ('T', attr_t, 'Tshape', attr_tshape)
    [result] = execute.execute(b'Reshape', 1, inputs=inputs_flat, attrs=attrs, ctx=ctx)
    return result

def _eager_fill(dims, value, ctx):
    if False:
        for i in range(10):
            print('nop')
    'Eager-only version of Fill op; requires value is an eager Tensor.'
    attr_t = value.dtype.as_datatype_enum
    dims = convert_to_eager_tensor(dims, ctx, dtypes.int32)
    inputs_flat = [dims, value]
    attrs = ('T', attr_t, 'index_type', types_pb2.DT_INT32)
    [result] = execute.execute(b'Fill', 1, inputs=inputs_flat, attrs=attrs, ctx=ctx)
    return result

def _eager_identity(tensor, ctx):
    if False:
        i = 10
        return i + 15
    'Eager-only version of Identity op; requires tensor is an eager Tensor.'
    attrs = ('T', tensor.dtype.as_datatype_enum)
    [result] = execute.execute(b'Identity', 1, inputs=[tensor], attrs=attrs, ctx=ctx)
    return result

def convert_to_eager_tensor(value, ctx, dtype=None) -> ops._EagerTensorBase:
    if False:
        i = 10
        return i + 15
    'Converts the given `value` to an `EagerTensor`.\n\n  Note that this function could return cached copies of created constants for\n  performance reasons.\n\n  Args:\n    value: value to convert to EagerTensor.\n    ctx: value of context.context().\n    dtype: optional desired dtype of the converted EagerTensor.\n\n  Returns:\n    EagerTensor created from value.\n\n  Raises:\n    TypeError: if `dtype` is not compatible with the type of t.\n  '
    if isinstance(value, np.ndarray):
        value = value.copy()
    if isinstance(value, ops.EagerTensor):
        if dtype is not None and value.dtype != dtype:
            raise TypeError(f'Expected tensor {value} with dtype {dtype!r}, but got dtype {value.dtype!r}.')
        return value
    if dtype is not None:
        try:
            dtype = dtype.as_datatype_enum
        except AttributeError:
            dtype = dtypes.as_dtype(dtype).as_datatype_enum
    ctx.ensure_initialized()
    return ops.EagerTensor(value, ctx.device_name, dtype)

@tf_export(v1=['constant'])
def constant_v1(value, dtype=None, shape=None, name='Const', verify_shape=False) -> Union[ops.Operation, ops._EagerTensorBase]:
    if False:
        for i in range(10):
            print('nop')
    'Creates a constant tensor.\n\n  The resulting tensor is populated with values of type `dtype`, as\n  specified by arguments `value` and (optionally) `shape` (see examples\n  below).\n\n  The argument `value` can be a constant value, or a list of values of type\n  `dtype`. If `value` is a list, then the length of the list must be less\n  than or equal to the number of elements implied by the `shape` argument (if\n  specified). In the case where the list length is less than the number of\n  elements specified by `shape`, the last element in the list will be used\n  to fill the remaining entries.\n\n  The argument `shape` is optional. If present, it specifies the dimensions of\n  the resulting tensor. If not present, the shape of `value` is used.\n\n  If the argument `dtype` is not specified, then the type is inferred from\n  the type of `value`.\n\n  For example:\n\n  ```python\n  # Constant 1-D Tensor populated with value list.\n  tensor = tf.constant([1, 2, 3, 4, 5, 6, 7]) => [1 2 3 4 5 6 7]\n\n  # Constant 2-D tensor populated with scalar value -1.\n  tensor = tf.constant(-1.0, shape=[2, 3]) => [[-1. -1. -1.]\n                                               [-1. -1. -1.]]\n  ```\n\n  `tf.constant` differs from `tf.fill` in a few ways:\n\n  *   `tf.constant` supports arbitrary constants, not just uniform scalar\n      Tensors like `tf.fill`.\n  *   `tf.constant` creates a `Const` node in the computation graph with the\n      exact value at graph construction time. On the other hand, `tf.fill`\n      creates an Op in the graph that is expanded at runtime.\n  *   Because `tf.constant` only embeds constant values in the graph, it does\n      not support dynamic shapes based on other runtime Tensors, whereas\n      `tf.fill` does.\n\n  Args:\n    value:          A constant value (or list) of output type `dtype`.\n\n    dtype:          The type of the elements of the resulting tensor.\n\n    shape:          Optional dimensions of resulting tensor.\n\n    name:           Optional name for the tensor.\n\n    verify_shape:   Boolean that enables verification of a shape of values.\n\n  Returns:\n    A Constant Tensor.\n\n  Raises:\n    TypeError: if shape is incorrectly specified or unsupported.\n  '
    return _constant_impl(value, dtype, shape, name, verify_shape=verify_shape, allow_broadcast=False)

@tf_export('constant', v1=[])
def constant(value, dtype=None, shape=None, name='Const') -> Union[ops.Operation, ops._EagerTensorBase]:
    if False:
        return 10
    'Creates a constant tensor from a tensor-like object.\n\n  Note: All eager `tf.Tensor` values are immutable (in contrast to\n  `tf.Variable`). There is nothing especially _constant_ about the value\n  returned from `tf.constant`. This function is not fundamentally different from\n  `tf.convert_to_tensor`. The name `tf.constant` comes from the `value` being\n  embedded in a `Const` node in the `tf.Graph`. `tf.constant` is useful\n  for asserting that the value can be embedded that way.\n\n  If the argument `dtype` is not specified, then the type is inferred from\n  the type of `value`.\n\n  >>> # Constant 1-D Tensor from a python list.\n  >>> tf.constant([1, 2, 3, 4, 5, 6])\n  <tf.Tensor: shape=(6,), dtype=int32,\n      numpy=array([1, 2, 3, 4, 5, 6], dtype=int32)>\n  >>> # Or a numpy array\n  >>> a = np.array([[1, 2, 3], [4, 5, 6]])\n  >>> tf.constant(a)\n  <tf.Tensor: shape=(2, 3), dtype=int64, numpy=\n    array([[1, 2, 3],\n           [4, 5, 6]])>\n\n  If `dtype` is specified, the resulting tensor values are cast to the requested\n  `dtype`.\n\n  >>> tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.float64)\n  <tf.Tensor: shape=(6,), dtype=float64,\n      numpy=array([1., 2., 3., 4., 5., 6.])>\n\n  If `shape` is set, the `value` is reshaped to match. Scalars are expanded to\n  fill the `shape`:\n\n  >>> tf.constant(0, shape=(2, 3))\n    <tf.Tensor: shape=(2, 3), dtype=int32, numpy=\n    array([[0, 0, 0],\n           [0, 0, 0]], dtype=int32)>\n  >>> tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])\n  <tf.Tensor: shape=(2, 3), dtype=int32, numpy=\n    array([[1, 2, 3],\n           [4, 5, 6]], dtype=int32)>\n\n  `tf.constant` has no effect if an eager Tensor is passed as the `value`, it\n  even transmits gradients:\n\n  >>> v = tf.Variable([0.0])\n  >>> with tf.GradientTape() as g:\n  ...     loss = tf.constant(v + v)\n  >>> g.gradient(loss, v).numpy()\n  array([2.], dtype=float32)\n\n  But, since `tf.constant` embeds the value in the `tf.Graph` this fails for\n  symbolic tensors:\n\n  >>> with tf.compat.v1.Graph().as_default():\n  ...   i = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.float32)\n  ...   t = tf.constant(i)\n  Traceback (most recent call last):\n  ...\n  TypeError: ...\n\n  `tf.constant` will create tensors on the current device. Inputs which are\n  already tensors maintain their placements unchanged.\n\n  Related Ops:\n\n  * `tf.convert_to_tensor` is similar but:\n    * It has no `shape` argument.\n    * Symbolic tensors are allowed to pass through.\n\n    >>> with tf.compat.v1.Graph().as_default():\n    ...   i = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.float32)\n    ...   t = tf.convert_to_tensor(i)\n\n  * `tf.fill`: differs in a few ways:\n    *   `tf.constant` supports arbitrary constants, not just uniform scalar\n        Tensors like `tf.fill`.\n    *   `tf.fill` creates an Op in the graph that is expanded at runtime, so it\n        can efficiently represent large tensors.\n    *   Since `tf.fill` does not embed the value, it can produce dynamically\n        sized outputs.\n\n  Args:\n    value: A constant value (or list) of output type `dtype`.\n    dtype: The type of the elements of the resulting tensor.\n    shape: Optional dimensions of resulting tensor.\n    name: Optional name for the tensor.\n\n  Returns:\n    A Constant Tensor.\n\n  Raises:\n    TypeError: if shape is incorrectly specified or unsupported.\n    ValueError: if called on a symbolic tensor.\n  '
    return _constant_impl(value, dtype, shape, name, verify_shape=False, allow_broadcast=True)

def _constant_impl(value, dtype, shape, name, verify_shape, allow_broadcast) -> Union[ops.Operation, ops._EagerTensorBase]:
    if False:
        i = 10
        return i + 15
    'Implementation of constant.'
    ctx = context.context()
    if ctx.executing_eagerly():
        if trace.enabled:
            with trace.Trace('tf.constant'):
                return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
        return _constant_eager_impl(ctx, value, dtype, shape, verify_shape)
    const_tensor = ops._create_graph_constant(value, dtype, shape, name, verify_shape, allow_broadcast)
    return const_tensor

def _constant_eager_impl(ctx, value, dtype, shape, verify_shape) -> ops._EagerTensorBase:
    if False:
        print('Hello World!')
    'Creates a constant on the current device.'
    t = convert_to_eager_tensor(value, ctx, dtype)
    if shape is None:
        return t
    shape = tensor_shape.as_shape(shape)
    if shape == t.shape:
        return t
    if verify_shape:
        raise TypeError(f'Expected Tensor {t} (converted from {value}) with shape {tuple(shape)}, but got shape {tuple(t.shape)}.')
    num_t = t.shape.num_elements()
    if num_t == shape.num_elements():
        return _eager_reshape(t, shape.as_list(), ctx)
    if num_t == 1:
        if t.dtype == dtypes.bool:
            with ops.device('/device:CPU:0'):
                x = _eager_fill(shape.as_list(), _eager_identity(t, ctx), ctx)
            return _eager_identity(x, ctx)
        else:
            return _eager_fill(shape.as_list(), t, ctx)
    raise TypeError(f'Eager execution of tf.constant with unsupported shape. Tensor {t} (converted from {value}) has {num_t:d} elements, but got `shape` {shape} with {shape.num_elements()} elements).')

def is_constant(tensor_or_op):
    if False:
        return 10
    if isinstance(tensor_or_op, tensor_lib.Tensor):
        op = tensor_or_op.op
    else:
        op = tensor_or_op
    return op.type == 'Const'

def _constant_tensor_conversion_function(v, dtype=None, name=None, as_ref=False):
    if False:
        for i in range(10):
            print('nop')
    _ = as_ref
    return constant(v, dtype=dtype, name=name)
tensor_conversion_registry.register_tensor_conversion_function_internal(tensor_conversion_registry._CONSTANT_OP_CONVERTIBLES, _constant_tensor_conversion_function, 0)
tensor_conversion_registry.register_tensor_conversion_function((list, tuple), _constant_tensor_conversion_function, 100)
tensor_conversion_registry.register_tensor_conversion_function(object, _constant_tensor_conversion_function, 200)

def _tensor_shape_tensor_conversion_function(s, dtype=None, name=None, as_ref=False):
    if False:
        for i in range(10):
            print('nop')
    'Function to convert TensorShape to Tensor.'
    _ = as_ref
    if not s.is_fully_defined():
        raise ValueError(f'Cannot convert a partially known TensorShape {s} to a Tensor.')
    s_list = s.as_list()
    int64_value = 0
    for dim in s_list:
        if dim >= 2 ** 31:
            int64_value = dim
            break
    if dtype is not None:
        if dtype not in (dtypes.int32, dtypes.int64):
            raise TypeError(f'Cannot convert TensorShape {s} to dtype {dtype}. Allowed dtypes are tf.int32 and tf.int64.')
        if dtype == dtypes.int32 and int64_value:
            raise ValueError(f'Cannot convert TensorShape {s} to dtype int32; a dimension is too large. Consider using tf.int64.')
    else:
        dtype = dtypes.int64 if int64_value else dtypes.int32
    if name is None:
        name = 'shape_as_tensor'
    return constant(s_list, dtype=dtype, name=name)
tensor_conversion_registry.register_tensor_conversion_function(tensor_shape.TensorShape, _tensor_shape_tensor_conversion_function, 100)

def _dimension_tensor_conversion_function(d, dtype=None, name=None, as_ref=False):
    if False:
        for i in range(10):
            print('nop')
    'Function to convert Dimension to Tensor.'
    _ = as_ref
    if d.value is None:
        raise ValueError(f'Cannot convert unknown Dimension {d} to a Tensor.')
    if dtype is not None:
        if dtype not in (dtypes.int32, dtypes.int64):
            raise TypeError(f'Cannot convert Dimension {d} to dtype {dtype}. Allowed dtypes are tf.int32 and tf.int64.')
    else:
        dtype = dtypes.int32
    if name is None:
        name = 'shape_as_tensor'
    return constant(d.value, dtype=dtype, name=name)
tensor_conversion_registry.register_tensor_conversion_function(tensor_shape.Dimension, _dimension_tensor_conversion_function, 100)

class _ConstantTensorCodec:
    """Codec for Tensor."""

    def can_encode(self, pyobj):
        if False:
            i = 10
            return i + 15
        return isinstance(pyobj, tensor_lib.Tensor)

    def do_encode(self, tensor_value, encode_fn):
        if False:
            while True:
                i = 10
        'Returns an encoded `TensorProto` for the given `tf.Tensor`.'
        del encode_fn
        encoded_tensor = struct_pb2.StructuredValue()
        if isinstance(tensor_value, ops.EagerTensor):
            encoded_tensor.tensor_value.CopyFrom(tensor_util.make_tensor_proto(tensor_value.numpy()))
        elif tensor_value.op.type == 'Const':
            encoded_tensor.tensor_value.CopyFrom(tensor_value.op.get_attr('value'))
        else:
            raise nested_structure_coder.NotEncodableError(f'No encoder for object {str(tensor_value)} of type {type(tensor_value)}.')
        return encoded_tensor

    def can_decode(self, value):
        if False:
            while True:
                i = 10
        return value.HasField('tensor_value')

    def do_decode(self, value, decode_fn):
        if False:
            print('Hello World!')
        'Returns the `tf.Tensor` encoded by the proto `value`.'
        del decode_fn
        tensor_proto = value.tensor_value
        tensor = constant(tensor_util.MakeNdarray(tensor_proto))
        return tensor
nested_structure_coder.register_codec(_ConstantTensorCodec())

class _NumpyCodec:
    """Codec for Numpy."""

    def can_encode(self, pyobj):
        if False:
            while True:
                i = 10
        return isinstance(pyobj, np.ndarray)

    def do_encode(self, numpy_value, encode_fn):
        if False:
            i = 10
            return i + 15
        'Returns an encoded `TensorProto` for `np.ndarray`.'
        del encode_fn
        encoded_numpy = struct_pb2.StructuredValue()
        encoded_numpy.numpy_value.CopyFrom(tensor_util.make_tensor_proto(numpy_value))
        return encoded_numpy

    def can_decode(self, value):
        if False:
            for i in range(10):
                print('nop')
        return value.HasField('numpy_value')

    def do_decode(self, value, decode_fn):
        if False:
            while True:
                i = 10
        'Returns the `np.ndarray` encoded by the proto `value`.'
        del decode_fn
        tensor_proto = value.numpy_value
        numpy = tensor_util.MakeNdarray(tensor_proto)
        return numpy
nested_structure_coder.register_codec(_NumpyCodec())