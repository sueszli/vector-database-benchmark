"""Gradients for operators defined in data_flow_ops.py."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops

@ops.RegisterGradient('DynamicPartition')
def _DynamicPartitionGrads(op, *grads):
    if False:
        i = 10
        return i + 15
    'Gradients for DynamicPartition.'
    data = op.inputs[0]
    indices = op.inputs[1]
    num_partitions = op.get_attr('num_partitions')
    prefix_shape = array_ops.shape(indices)
    original_indices = array_ops.reshape(math_ops.range(math_ops.reduce_prod(prefix_shape)), prefix_shape)
    partitioned_indices = data_flow_ops.dynamic_partition(original_indices, indices, num_partitions)
    reconstructed = data_flow_ops.parallel_dynamic_stitch(partitioned_indices, grads)
    reconstructed = array_ops.reshape(reconstructed, array_ops.shape(data))
    return [reconstructed, None]

@ops.RegisterGradient('DynamicStitch')
@ops.RegisterGradient('ParallelDynamicStitch')
def _DynamicStitchGrads(op, grad):
    if False:
        return 10
    'Gradients for DynamicStitch and ParallelDynamicStitch.'
    num_values = len(op.inputs) // 2
    indices_grad = [None] * num_values

    def AsInt32(x):
        if False:
            return 10
        return x if op.inputs[0].dtype == dtypes.int32 else math_ops.cast(x, dtypes.int32)
    inputs = [AsInt32(op.inputs[i]) for i in range(num_values)]
    if isinstance(grad, indexed_slices.IndexedSlices):
        output_shape = array_ops.shape(op.outputs[0])
        output_rows = output_shape[0]
        grad = math_ops.unsorted_segment_sum(grad.values, grad.indices, output_rows)
    values_grad = [array_ops.gather(grad, inp) for inp in inputs]
    return indices_grad + values_grad
ops.NotDifferentiable('Queue')
ops.NotDifferentiable('QueueEnqueue')
ops.NotDifferentiable('QueueEnqueueMany')
ops.NotDifferentiable('QueueDequeue')
ops.NotDifferentiable('QueueDequeueMany')
ops.NotDifferentiable('QueueDequeueUpTo')
ops.NotDifferentiable('QueueClose')
ops.NotDifferentiable('QueueSize')
ops.NotDifferentiable('Stack')
ops.NotDifferentiable('StackPush')
ops.NotDifferentiable('StackPop')
ops.NotDifferentiable('StackClose')
ops.NotDifferentiable('GetSessionHandle')
ops.NotDifferentiable('GetSessionHandleV2')
ops.NotDifferentiable('GetSessionTensor')
ops.NotDifferentiable('DeleteSessionTensor')