"""Ops to convert between RaggedTensors and other tensor types."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_ragged_conversion_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor

def from_tensor(tensor, lengths=None, padding=None, ragged_rank=1, row_splits_dtype=dtypes.int64, name=None):
    if False:
        i = 10
        return i + 15
    if ragged_tensor.is_ragged(tensor):
        return tensor
    else:
        return ragged_tensor.RaggedTensor.from_tensor(tensor, lengths=lengths, padding=padding, ragged_rank=ragged_rank, row_splits_dtype=row_splits_dtype, name=name)

def to_tensor(rt_input, default_value=None, name=None):
    if False:
        return 10
    if ragged_tensor.is_ragged(rt_input):
        return rt_input.to_tensor(default_value, name)
    else:
        return rt_input

def ragged_to_dense(rt_input, default_value=None, shape=None):
    if False:
        while True:
            i = 10
    'Create a dense tensor from a ragged tensor.'
    return rt_input.to_tensor(default_value=default_value, shape=shape)

@ops.RegisterGradient('RaggedTensorToTensor')
def _ragged_tensor_to_tensor_grad(op, grad):
    if False:
        i = 10
        return i + 15
    'Gradient for RaggedToTensor op.'
    flat_values = op.inputs[1]
    default_value = op.inputs[2]
    row_partition_tensors = op.inputs[3:]
    row_partition_types = op.get_attr('row_partition_types')
    flat_value_shape = array_ops.shape(flat_values)
    ragged_rank = sum((1 for typ in row_partition_types if typ != b'FIRST_DIM_SIZE'))
    indices = gen_ragged_conversion_ops.ragged_tensor_to_tensor(shape=array_ops.shape(grad)[:1 + ragged_rank], values=math_ops.range(flat_value_shape[0]), default_value=-1, row_partition_types=row_partition_types, row_partition_tensors=row_partition_tensors)
    mask = math_ops.not_equal(indices, -1)
    values_grad = indexed_slices.IndexedSlices(values=array_ops.boolean_mask(grad, mask), indices=array_ops.boolean_mask(indices, mask), dense_shape=flat_value_shape)
    default_grads = array_ops.boolean_mask(grad, ~mask)
    dims_to_reduce = math_ops.range(array_ops.rank(default_grads) - _rank_ignoring_leading_dims_with_size_1(default_value))
    default_grad = math_ops.reduce_sum(default_grads, axis=dims_to_reduce)
    default_grad = array_ops.reshape(default_grad, array_ops.shape(default_value))
    return [None, values_grad, default_grad] + [None for _ in row_partition_tensors]

def _rank_ignoring_leading_dims_with_size_1(value):
    if False:
        return 10
    'Returns `rank(value)`, ignoring any leading dimensions with size 1.'
    if value.shape.rank is not None:
        ndims = value.shape.rank
        for dim in value.shape.dims:
            if dim.value == 1:
                ndims -= 1
            elif dim.value is None:
                ndims = None
                break
            else:
                break
        if ndims is not None:
            return ndims
    shape = array_ops.shape(value)
    dim_is_one = math_ops.cast(math_ops.equal(shape, 1), dtypes.int32)
    leading_ones = math_ops.cumprod(dim_is_one)
    num_leading_ones = math_ops.reduce_sum(leading_ones)
    return array_ops.rank(value) - num_leading_ones

def to_sparse(rt_input, name=None):
    if False:
        for i in range(10):
            print('nop')
    return rt_input.to_sparse(name)

def from_sparse(st_input, name=None):
    if False:
        i = 10
        return i + 15
    return ragged_tensor.RaggedTensor.from_sparse(st_input, name)

@ops.RegisterGradient('RaggedTensorFromVariant')
def _ragged_tensor_from_variant_grad(op, *grads):
    if False:
        return 10
    'Gradient for RaggedTensorFromVariant op.'
    variant_rank = op.inputs[0].shape.rank
    if variant_rank == 0:
        batched_input = False
    elif variant_rank == 1:
        batched_input = True
    elif variant_rank is None:
        batched_input = op.get_attr('output_ragged_rank') > 0
    else:
        raise ValueError('Unable to compute gradient: RaggedTensorToVariant can currently only generate 0D or 1D output.')
    return [gen_ragged_conversion_ops.ragged_tensor_to_variant(rt_nested_splits=op.outputs[:-1], rt_dense_values=grads[-1], batched_input=batched_input)]

@ops.RegisterGradient('RaggedTensorToVariant')
def _ragged_tensor_to_variant_grad(op, encoded_ragged_grad):
    if False:
        i = 10
        return i + 15
    'Gradient for RaggedTensorToVariant op.'
    dense_values = op.inputs[-1]
    ragged_rank = len(op.inputs) - 1
    row_splits = 0 if ragged_rank == 0 else op.inputs[0]
    values_grad = gen_ragged_conversion_ops.ragged_tensor_to_variant_gradient(encoded_ragged_grad=encoded_ragged_grad, row_splits=row_splits, dense_values_shape=array_ops.shape(dense_values), Tvalues=op.inputs[-1].dtype)
    result = [None] * ragged_rank + [values_grad]
    return result