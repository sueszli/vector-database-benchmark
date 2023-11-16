"""Shared utilities related to backprop."""
from tensorflow.core.config import flags
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops

def _DTypeFromTensor(tensor):
    if False:
        for i in range(10):
            print('nop')
    'Extract either `tensor.dtype` or the unanimous sub-type of a variant.'
    dtype = tensor.dtype
    if dtype.base_dtype == dtypes.variant:
        if isinstance(tensor, ops.EagerTensor):
            handle_data = tensor._handle_data
        else:
            handle_data = handle_data_util.get_resource_handle_data(tensor)
        if handle_data is not None and handle_data.is_set and handle_data.shape_and_type:
            first_type = handle_data.shape_and_type[0].dtype
            if first_type != types_pb2.DT_INVALID and all((shape_and_type.dtype == first_type for shape_and_type in handle_data.shape_and_type)):
                return first_type
    return dtype

def IsTrainable(tensor_or_dtype):
    if False:
        return 10
    'Determines whether a tensor or dtype supports infinitesimal changes.'
    if tensor_util.is_tf_type(tensor_or_dtype):
        dtype = _DTypeFromTensor(tensor_or_dtype)
    else:
        dtype = tensor_or_dtype
    dtype = dtypes.as_dtype(dtype)
    trainable_dtypes = [dtypes.float16, dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128, dtypes.resource, dtypes.variant, dtypes.bfloat16]
    if flags.config().enable_quantized_dtypes_training.value():
        trainable_dtypes.extend([dtypes.qint8, dtypes.qint16, dtypes.qint32, dtypes.quint8, dtypes.quint16])
    return dtype.base_dtype in trainable_dtypes

def FlattenNestedIndexedSlices(grad):
    if False:
        return 10
    assert isinstance(grad, indexed_slices.IndexedSlices)
    if isinstance(grad.values, tensor_lib.Tensor):
        return grad
    else:
        assert isinstance(grad.values, indexed_slices.IndexedSlices)
        g = FlattenNestedIndexedSlices(grad.values)
        return indexed_slices.IndexedSlices(g.values, array_ops.gather(grad.indices, g.indices), g.dense_shape)

def AggregateIndexedSlicesGradients(grads):
    if False:
        print('Hello World!')
    'Aggregates gradients containing `IndexedSlices`s.'
    if len(grads) < 1:
        return None
    if len(grads) == 1:
        return grads[0]
    grads = [g for g in grads if g is not None]
    if any((isinstance(g, tensor_lib.Tensor) for g in grads)):
        return math_ops.add_n(grads)
    grads = math_ops._as_indexed_slices_list(grads)
    grads = [FlattenNestedIndexedSlices(x) for x in grads]
    concat_grad = indexed_slices.IndexedSlices(array_ops.concat([x.values for x in grads], axis=0), array_ops.concat([x.indices for x in grads], axis=0), grads[0].dense_shape)
    return concat_grad