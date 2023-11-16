"""Gradients for operators defined in math_ops.py."""
import numpy as np
from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices as indexed_slices_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops

@ops.RegisterGradient('ArgMax')
def _ArgMaxGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    del op, grad
    return [None, None]

@ops.RegisterGradient('ArgMin')
def _ArgMinGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    del op, grad
    return [None, None]

@ops.RegisterGradient('EuclideanNorm')
def _EuclideanNormGrad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    'Gradient for EuclideanNorm.'
    output = op.outputs[0]
    if not op.get_attr('keep_dims'):
        output_shape_kept_dims = math_ops.reduced_shape(array_ops.shape(op.inputs[0]), op.inputs[1])
        output = array_ops.reshape(output, output_shape_kept_dims)
        grad = array_ops.reshape(grad, output_shape_kept_dims)
    return (math_ops.truediv(op.inputs[0], output / grad), None)

def SmartBroadcastGradientArgs(x, y, grad=None):
    if False:
        for i in range(10):
            print('nop')
    'Version of `BroadcastGradientArgs` optimized for partially-known shapes.\n\n  Args:\n    x: The first argument of a broadcasting binary op.\n    y: The second argument of a broadcasting binary op.\n    grad: Deprecated.\n\n  Returns:\n    A pair of triples, one per argument with\n      * Shape of the argument (tensor);\n      * Reduction axes for the argument (list or tensor);\n      * Boolean indicating whether the reduction must be applied.\n  '
    del grad
    x_shape = array_ops.shape(x)
    y_shape = array_ops.shape(y)
    if not context.executing_eagerly() and isinstance(x, tensor.Tensor) and isinstance(y, tensor.Tensor):
        (x_axes, y_axes) = _InferGradientReductionAxes(x.shape, y.shape)
    else:
        (x_axes, y_axes) = (None, None)
    if x_axes is None or y_axes is None:
        (x_axes, y_axes) = gen_array_ops.broadcast_gradient_args(x_shape, y_shape)
        x_must_reduce = True
        y_must_reduce = True
    else:
        x_must_reduce = x_axes or x.shape.rank < y.shape.rank
        y_must_reduce = y_axes or y.shape.rank < x.shape.rank
    return ((x_shape, x_axes, x_must_reduce), (y_shape, y_axes, y_must_reduce))

def _InferGradientReductionAxes(x_shape, y_shape):
    if False:
        while True:
            i = 10
    'Infers the sets of axes that might have been broadcasted.'
    x_rank = x_shape.rank
    y_rank = y_shape.rank
    if x_rank is None or y_rank is None:
        return (None, None)
    x_shape = x_shape.as_list()
    y_shape = y_shape.as_list()
    b_rank = max(x_rank, y_rank)
    x_axes = []
    y_axes = []
    for axis in range(b_rank):
        x_dim = 1 if axis < b_rank - x_rank else x_shape[axis - (b_rank - x_rank)]
        y_dim = 1 if axis < b_rank - y_rank else y_shape[axis - (b_rank - y_rank)]
        if x_dim == 1 and y_dim != 1:
            x_axes.append(axis)
        elif y_dim == 1 and x_dim != 1:
            y_axes.append(axis)
        elif x_dim is None or y_dim is None:
            return (None, None)
    return (x_axes, y_axes)

def _ReduceGradientArg(grad, shape_axes_must_reduce):
    if False:
        print('Hello World!')
    'Reduces gradients of one of the arguments of a broadcasting binary op.'
    (shape, axes, must_reduce) = shape_axes_must_reduce
    if grad is not None and must_reduce:
        grad = math_ops.reduce_sum(grad, axes, keepdims=True)
        grad = array_ops.reshape(grad, shape)
    return grad

def _ReduceGradientArgs(x, y, gx, gy):
    if False:
        print('Hello World!')
    'Reduces gradients of both arguments of a broadcasting binary op.'
    if gx is not None or gy is not None:
        (bx, by) = SmartBroadcastGradientArgs(x, y)
        gx = _ReduceGradientArg(gx, bx)
        gy = _ReduceGradientArg(gy, by)
    return (gx, gy)
_EMPTY_TUPLE = ()

def _IsScalar(x):
    if False:
        return 10
    return x._shape_tuple() is _EMPTY_TUPLE

def _SafeShapeDiv(x, y):
    if False:
        print('Hello World!')
    'Divides `x / y` assuming `x, y >= 0`, treating `0 / 0 = 0`.'
    return x // math_ops.maximum(y, 1)

@ops.RegisterGradient('Sum')
def _SumGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    'Gradient for Sum.'
    input_0_shape = op.inputs[0]._shape_tuple()
    if input_0_shape is not None:
        axes = tensor_util.constant_value(op.inputs[1])
        if axes is not None:
            rank = len(input_0_shape)
            if np.array_equal(axes, np.arange(rank)):
                if context.executing_eagerly():
                    ctx = context.context()
                    new_shape = ctx.ones_rank_cache().get(rank)
                    if new_shape is None:
                        new_shape = constant_op.constant([1] * rank, dtype=dtypes.int32)
                        ctx.ones_rank_cache().put(rank, new_shape)
                else:
                    new_shape = [1] * rank
                grad = array_ops.reshape(grad, new_shape)
                if None not in input_0_shape:
                    input_shape = constant_op.constant(input_0_shape, dtype=dtypes.int32)
                else:
                    input_shape = array_ops.shape(op.inputs[0])
                return [array_ops.tile(grad, input_shape), None]
            elif None not in input_0_shape and (not context.executing_eagerly()):
                graph = ops.get_default_graph()
                axes = tuple(axes.reshape(-1))
                try:
                    (output_shape_kept_dims, tile_scaling) = graph._reduced_shape_cache[input_0_shape, axes]
                except KeyError:

                    def EvaluateAsTuple(t):
                        if False:
                            i = 10
                            return i + 15
                        if tensor_util.is_tf_type(t):
                            value = tensor_util.try_evaluate_constant(t)
                            assert value is not None
                        else:
                            value = t
                        return tuple(value)
                    output_shape_kept_dims = EvaluateAsTuple(math_ops.reduced_shape(input_0_shape, axes))
                    tile_scaling = EvaluateAsTuple(_SafeShapeDiv(input_0_shape, output_shape_kept_dims))
                    graph._reduced_shape_cache[input_0_shape, axes] = (output_shape_kept_dims, tile_scaling)
                grad = array_ops.reshape(grad, output_shape_kept_dims)
                return [array_ops.tile(grad, tile_scaling), None]
    input_shape = array_ops.shape(op.inputs[0])
    if not op.get_attr('keep_dims'):
        with ops.colocate_with(input_shape):
            output_shape_kept_dims = math_ops.reduced_shape(input_shape, op.inputs[1])
        grad = array_ops.reshape(grad, output_shape_kept_dims)
    return [array_ops.broadcast_to(grad, input_shape), None]

def _MinOrMaxGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    "Gradient for Min or Max. Amazingly it's precisely the same code."
    input_shape = array_ops.shape(op.inputs[0])
    y = op.outputs[0]
    if not op.get_attr('keep_dims'):
        output_shape_kept_dims = math_ops.reduced_shape(input_shape, op.inputs[1])
        y = array_ops.reshape(y, output_shape_kept_dims)
        grad = array_ops.reshape(grad, output_shape_kept_dims)
    else:
        output_shape_kept_dims = array_ops.shape(y)
    indicators = math_ops.cast(math_ops.equal(y, op.inputs[0]), grad.dtype)
    num_selected = array_ops.reshape(math_ops.reduce_sum(indicators, op.inputs[1]), output_shape_kept_dims)
    return [math_ops.divide(indicators, num_selected) * grad, None]

@ops.RegisterGradient('Max')
def _MaxGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Gradient for Max.'
    return _MinOrMaxGrad(op, grad)

@ops.RegisterGradient('Min')
def _MinGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    return _MinOrMaxGrad(op, grad)

@ops.RegisterGradient('Mean')
def _MeanGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Gradient for Mean.'
    sum_grad = _SumGrad(op, grad)[0]
    input_shape = op.inputs[0]._shape_tuple()
    output_shape = op.outputs[0]._shape_tuple()
    if input_shape is not None and output_shape is not None and (None not in input_shape) and (None not in output_shape):
        input_size = np.prod(input_shape)
        output_size = np.prod(output_shape)
        factor = input_size // max(output_size, 1)
        factor = constant_op.constant(factor, dtype=sum_grad.dtype)
    else:
        input_shape = array_ops.shape(op.inputs[0])
        input_rank = array_ops.size(input_shape)
        axes = (op.inputs[1] + input_rank) % input_rank
        factor = math_ops.reduce_prod(array_ops.gather(input_shape, axes))
    return (math_ops.truediv(sum_grad, math_ops.cast(factor, sum_grad.dtype)), None)

@ops.RegisterGradient('Prod')
def _ProdGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'Gradient for Prod.'
    input_shape = array_ops.shape(op.inputs[0])
    reduction_indices = array_ops.reshape(op.inputs[1], [-1])
    if not op.get_attr('keep_dims'):
        output_shape_kept_dims = math_ops.reduced_shape(input_shape, op.inputs[1])
        grad = array_ops.reshape(grad, output_shape_kept_dims)
    grad = array_ops.broadcast_to(grad, input_shape)
    with ops.device('/cpu:0'):
        rank = array_ops.rank(op.inputs[0])
        reduction_indices = (reduction_indices + rank) % rank
        reduced = math_ops.cast(reduction_indices, dtypes.int32)
        idx = math_ops.range(0, rank)
        (other, _) = gen_array_ops.list_diff(idx, reduced, dtypes.int32)
        perm = array_ops.concat([reduced, other], 0)
        reduced_num = math_ops.reduce_prod(array_ops.gather(input_shape, reduced))
        other_num = math_ops.reduce_prod(array_ops.gather(input_shape, other))
    permuted = array_ops.transpose(op.inputs[0], perm)
    permuted_shape = array_ops.shape(permuted)
    reshaped = array_ops.reshape(permuted, (reduced_num, other_num))
    left = math_ops.cumprod(reshaped, axis=0, exclusive=True)
    right = math_ops.cumprod(reshaped, axis=0, exclusive=True, reverse=True)
    y = array_ops.reshape(math_ops.conj(left) * math_ops.conj(right), permuted_shape)
    out = grad * array_ops.transpose(y, array_ops.invert_permutation(perm))
    return (array_ops.reshape(out, input_shape), None)

@ops.RegisterGradient('SegmentSum')
def _SegmentSumGrad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    'Gradient for SegmentSum.'
    return (array_ops.gather(grad, op.inputs[1]), None)

@ops.RegisterGradient('SegmentMean')
def _SegmentMeanGrad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    'Gradient for SegmentMean.'
    input_rank = array_ops.rank(op.inputs[0])
    ones_shape = array_ops.concat([array_ops.shape(op.inputs[1]), array_ops.ones(array_ops.expand_dims(input_rank - 1, 0), dtype=dtypes.int32)], 0)
    ones = array_ops.ones(ones_shape, dtype=grad.dtype)
    scaled_grad = math_ops.divide(grad, math_ops.segment_sum(ones, op.inputs[1]))
    return (array_ops.gather(scaled_grad, op.inputs[1]), None)

def _SparseSegmentReduceGradV2(op, grad, norm=None):
    if False:
        print('Hello World!')
    'Sparse gradient for SparseSegment(Sum|Mean|SqrtN)[WithNumSegments].'
    assert norm is None or norm == 'mean' or norm == 'sqrtn'
    data = op.inputs[0]
    indices = op.inputs[1]
    segment_ids = op.inputs[2]
    data_shape = array_ops.shape(op.inputs[0])
    dense_output_dim0 = data_shape[0]
    grad_fn = math_ops.sparse_segment_mean_grad_v2 if norm == 'mean' else math_ops.sparse_segment_sqrt_n_grad_v2 if norm == 'sqrtn' else math_ops.sparse_segment_sum_grad_v2
    (grad_values, sorted_unique_indices) = grad_fn(grad, indices, segment_ids, dense_output_dim0)
    return indexed_slices_lib.IndexedSlices(grad_values, sorted_unique_indices, data_shape)

def _GetOpAttrOrNone(op, name):
    if False:
        i = 10
        return i + 15
    'Returns the value of the attr of `op` with the given `name`, or None if no\n\n  such attr exists.\n  '
    try:
        return op.get_attr(name)
    except ValueError:
        return None

@ops.RegisterGradient('SparseSegmentSum')
def _SparseSegmentSumGrad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    'Gradient for SparseSegmentSum.'
    if _GetOpAttrOrNone(op, 'sparse_gradient'):
        return (_SparseSegmentReduceGradV2(op, grad), None, None)
    dim0 = array_ops.shape(op.inputs[0])[0]
    if compat.forward_compatible(2021, 6, 10):
        return (math_ops.sparse_segment_sum_grad(grad, op.inputs[1], op.inputs[2], dim0), None, None)
    else:
        return (math_ops.unsorted_segment_sum(array_ops.gather(grad, op.inputs[2]), op.inputs[1], dim0), None, None)

@ops.RegisterGradient('SparseSegmentSumWithNumSegments')
def _SparseSegmentSumWithNumSegmentsGrad(op: ops.Operation, grad):
    if False:
        return 10
    'Gradient for SparseSegmentSumWithNumSegments.'
    if _GetOpAttrOrNone(op, 'sparse_gradient'):
        return (_SparseSegmentReduceGradV2(op, grad), None, None, None)
    dim0 = array_ops.shape(op.inputs[0])[0]
    if compat.forward_compatible(2021, 6, 10):
        return (math_ops.sparse_segment_sum_grad(grad, op.inputs[1], op.inputs[2], dim0), None, None, None)
    else:
        return (math_ops.unsorted_segment_sum(array_ops.gather(grad, op.inputs[2]), op.inputs[1], dim0), None, None, None)

@ops.RegisterGradient('SparseSegmentMean')
def _SparseSegmentMeanGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'Gradient for SparseSegmentMean.'
    if _GetOpAttrOrNone(op, 'sparse_gradient'):
        return (_SparseSegmentReduceGradV2(op, grad, 'mean'), None, None)
    dim0 = array_ops.shape(op.inputs[0])[0]
    return (math_ops.sparse_segment_mean_grad(grad, op.inputs[1], op.inputs[2], dim0), None, None)

@ops.RegisterGradient('SparseSegmentMeanWithNumSegments')
def _SparseSegmentMeanWithNumSegmentsGrad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    'Gradient for SparseSegmentMeanWithNumSegments.'
    if _GetOpAttrOrNone(op, 'sparse_gradient'):
        return (_SparseSegmentReduceGradV2(op, grad, 'mean'), None, None, None)
    dim0 = array_ops.shape(op.inputs[0])[0]
    return (math_ops.sparse_segment_mean_grad(grad, op.inputs[1], op.inputs[2], dim0), None, None, None)

@ops.RegisterGradient('SparseSegmentSqrtN')
def _SparseSegmentSqrtNGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'Gradient for SparseSegmentSqrtN.'
    if _GetOpAttrOrNone(op, 'sparse_gradient'):
        return (_SparseSegmentReduceGradV2(op, grad, 'sqrtn'), None, None)
    dim0 = array_ops.shape(op.inputs[0])[0]
    return (math_ops.sparse_segment_sqrt_n_grad(grad, op.inputs[1], op.inputs[2], dim0), None, None)

@ops.RegisterGradient('SparseSegmentSqrtNWithNumSegments')
def _SparseSegmentSqrtNWithNumSegmentsGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Gradient for SparseSegmentSqrtNWithNumSegments.'
    if _GetOpAttrOrNone(op, 'sparse_gradient'):
        return (_SparseSegmentReduceGradV2(op, grad, 'sqrtn'), None, None, None)
    dim0 = array_ops.shape(op.inputs[0])[0]
    return (math_ops.sparse_segment_sqrt_n_grad(grad, op.inputs[1], op.inputs[2], dim0), None, None, None)

def _SegmentMinOrMaxGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    ' Gradient for SegmentMin and SegmentMax. '
    zeros = array_ops.zeros_like(op.inputs[0], dtype=op.inputs[0].dtype)
    gathered_outputs = array_ops.gather(op.outputs[0], op.inputs[1])
    is_selected = math_ops.equal(op.inputs[0], gathered_outputs)
    num_selected = math_ops.segment_sum(math_ops.cast(is_selected, grad.dtype), op.inputs[1])
    weighted_grads = math_ops.divide(grad, num_selected)
    gathered_grads = array_ops.gather(weighted_grads, op.inputs[1])
    return (array_ops.where_v2(is_selected, gathered_grads, zeros), None)

@ops.RegisterGradient('SegmentMin')
def _SegmentMinGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    'Gradient for SegmentMin.'
    return _SegmentMinOrMaxGrad(op, grad)

@ops.RegisterGradient('SegmentMax')
def _SegmentMaxGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Gradient for SegmentMax.'
    return _SegmentMinOrMaxGrad(op, grad)

@ops.RegisterGradient('SegmentProd')
def _SegmentProdGrad(op: ops.Operation, grad):
    if False:
        return 10
    "Gradient for SegmentProd.\n\n  The gradient can be expressed for each segment by dividing the segment's\n  product by each element of the segment input tensor, but this approach can't\n  deal with zeros in the input.\n  Unlike reduce_prod we can't use cumsum here as individual segments may have\n  a different number of elements. Therefore we consider three cases:\n  1) A segment input contains no zeros and we can safely divide by the input\n     tensor.\n  2) A segment contains exactly one zero. Then the gradient of each input of\n     the segment is zero except for the 0-input, there the gradient is\n     the product of the remaining segment entries.\n  3) A segment contains at least two zeros. The gradient is zero for all\n     segment inputs.\n  "
    data = op.inputs[0]
    segment_ids = op.inputs[1]
    is_zero = math_ops.equal(data, 0)
    num_zeros = gen_math_ops.segment_sum(math_ops.cast(is_zero, dtype=dtypes.int32), segment_ids)
    grad = array_ops.where_v2(math_ops.greater(num_zeros, 1), array_ops.zeros_like(grad), grad)
    non_zero_data = array_ops.where_v2(is_zero, array_ops.ones_like(data), data)
    non_zero_prod = gen_math_ops.segment_prod(non_zero_data, segment_ids)
    gathered_prod = array_ops.gather(op.outputs[0], segment_ids)
    gathered_non_zero_prod = array_ops.gather(non_zero_prod, segment_ids)
    prod_divided_by_el = gathered_prod / non_zero_data
    partial_derivative = array_ops.where_v2(is_zero, gathered_non_zero_prod, prod_divided_by_el)
    gathered_grad = array_ops.gather(grad, segment_ids)
    return (gathered_grad * partial_derivative, None)

def _GatherDropNegatives(params, ids, zero_clipped_indices=None, is_positive=None):
    if False:
        i = 10
        return i + 15
    ' Helper function for unsorted segment ops.\n\n  Gathers params for\n      positive segment ids and gathers 0 for inputs with negative segment id.\n      Also returns the clipped indices and a boolean mask with the same shape\n      as ids where a positive id is masked as true. With this, the latter two\n      can be passed as arguments to this function to reuse them.\n  '
    if zero_clipped_indices is None:
        zero_clipped_indices = math_ops.maximum(ids, array_ops.zeros_like(ids))
    gathered = array_ops.gather(params, zero_clipped_indices)
    if is_positive is None:
        is_positive = math_ops.greater_equal(ids, 0)
        is_positive_shape = array_ops.shape(is_positive)
        broadcastable_shape = array_ops.concat([is_positive_shape, array_ops.ones([array_ops.rank(gathered) - array_ops.rank(is_positive)], dtype=is_positive_shape.dtype)], axis=0)
        is_positive = array_ops.reshape(is_positive, broadcastable_shape)
        is_positive = is_positive & array_ops.ones_like(gathered, dtype=dtypes.bool)
    zero_slice = array_ops.zeros_like(gathered)
    return (array_ops.where_v2(is_positive, gathered, zero_slice), zero_clipped_indices, is_positive)

def _UnsortedSegmentMinOrMaxGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Gradient for UnsortedSegmentMin and UnsortedSegmentMax.'
    (gathered_outputs, zero_clipped_indices, is_positive) = _GatherDropNegatives(op.outputs[0], op.inputs[1])
    is_selected = math_ops.equal(op.inputs[0], gathered_outputs)
    is_selected = math_ops.logical_and(is_selected, is_positive)
    num_selected = math_ops.unsorted_segment_sum(math_ops.cast(is_selected, grad.dtype), op.inputs[1], op.inputs[2])
    weighted_grads = math_ops.divide(grad, num_selected)
    (gathered_grads, _, _) = _GatherDropNegatives(weighted_grads, None, zero_clipped_indices, is_positive)
    zeros = array_ops.zeros_like(gathered_grads)
    return (array_ops.where_v2(is_selected, gathered_grads, zeros), None, None)

@ops.RegisterGradient('UnsortedSegmentSum')
def _UnsortedSegmentSumGrad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    'Gradient for UnsortedSegmentSum.'
    return (_GatherDropNegatives(grad, op.inputs[1])[0], None, None)

@ops.RegisterGradient('UnsortedSegmentMax')
def _UnsortedSegmentMaxGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    ' Gradient for UnsortedSegmentMax. '
    return _UnsortedSegmentMinOrMaxGrad(op, grad)

@ops.RegisterGradient('UnsortedSegmentMin')
def _UnsortedSegmentMinGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    ' Gradient for UnsortedSegmentMin. '
    return _UnsortedSegmentMinOrMaxGrad(op, grad)

@ops.RegisterGradient('UnsortedSegmentProd')
def _UnsortedSegmentProdGrad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    " Gradient for UnsortedSegmentProd.\n\n  The gradient can be expressed for each segment by dividing the segment's\n  product by each element of the segment input tensor, but this approach can't\n  deal with zeros in the input.\n  Unlike reduce_prod we can't use cumsum here as individual segments may have\n  a different number of elements. Therefore we consider three cases:\n  1) A segment input contains no zeros and we can safely divide by the input\n     tensor.\n  2) A segment contains exactly one zero. Then the gradient of each input of\n     the segment is zero except for the 0-input, there the gradient is\n     the product of the remaining segment entries.\n  3) A segment contains at least two zeros. The gradient is zero for all\n     segment inputs.\n  "
    is_zero = math_ops.equal(op.inputs[0], 0)
    num_zeros = gen_math_ops.unsorted_segment_sum(math_ops.cast(is_zero, dtype=dtypes.int32), op.inputs[1], op.inputs[2])
    grad = array_ops.where_v2(math_ops.greater(num_zeros, 1), array_ops.zeros_like(grad), grad)
    non_zero_data = array_ops.where_v2(is_zero, array_ops.ones_like(op.inputs[0]), op.inputs[0])
    non_zero_prod = gen_math_ops.unsorted_segment_prod(non_zero_data, op.inputs[1], op.inputs[2])
    zero_clipped_indices = math_ops.maximum(op.inputs[1], array_ops.zeros_like(op.inputs[1]))
    gathered_prod = array_ops.gather(op.outputs[0], zero_clipped_indices)
    gathered_non_zero_prod = array_ops.gather(non_zero_prod, zero_clipped_indices)
    prod_divided_by_el = gathered_prod / op.inputs[0]
    partial_derivative = array_ops.where_v2(is_zero, gathered_non_zero_prod, prod_divided_by_el)
    gathered_grad = _GatherDropNegatives(grad, op.inputs[1], zero_clipped_indices)[0]
    return (gathered_grad * partial_derivative, None, None)

@ops.RegisterGradient('Abs')
def _AbsGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    x = op.inputs[0]
    return grad * math_ops.sign(x)

@ops.RegisterGradient('Neg')
def _NegGrad(_, grad):
    if False:
        for i in range(10):
            print('nop')
    'Returns -grad.'
    return -grad

@ops.RegisterGradient('Inv')
def _InvGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'Returns -grad * (1 / x^2).'
    y = op.outputs[0]
    return gen_math_ops.reciprocal_grad(y, grad)

@ops.RegisterGradient('Reciprocal')
def _ReciprocalGrad(op: ops.Operation, grad):
    if False:
        return 10
    'Returns -grad * (1 / x^2).'
    y = op.outputs[0]
    return gen_math_ops.reciprocal_grad(y, grad)

@ops.RegisterGradient('InvGrad')
def _InvGradGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    b = op.inputs[1]
    with ops.control_dependencies([grad]):
        ca = math_ops.conj(op.inputs[0])
        cg = math_ops.conj(grad)
        return (cg * -2.0 * b * ca, gen_math_ops.reciprocal_grad(ca, grad))

@ops.RegisterGradient('ReciprocalGrad')
def _ReciprocalGradGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    b = op.inputs[1]
    with ops.control_dependencies([grad]):
        ca = math_ops.conj(op.inputs[0])
        cg = math_ops.conj(grad)
        return (cg * -2.0 * b * ca, gen_math_ops.reciprocal_grad(ca, grad))

@ops.RegisterGradient('Square')
def _SquareGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        x = math_ops.conj(x)
        y = constant_op.constant(2.0, dtype=x.dtype)
        return math_ops.multiply(grad, math_ops.multiply(x, y))

@ops.RegisterGradient('Sqrt')
def _SqrtGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    y = op.outputs[0]
    return gen_math_ops.sqrt_grad(y, grad)

@ops.RegisterGradient('SqrtGrad')
def _SqrtGradGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    a = op.inputs[0]
    y = op.outputs[0]
    with ops.control_dependencies([grad]):
        ga = grad / a
        return (-math_ops.conj(ga) * y, 0.5 * ga)

@ops.RegisterGradient('Rsqrt')
def _RsqrtGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    'Returns -0.5 * grad * conj(y)^3.'
    y = op.outputs[0]
    return gen_math_ops.rsqrt_grad(y, grad)

@ops.RegisterGradient('RsqrtGrad')
def _RsqrtGradGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    'Returns backprop gradient for f(a,b) = -0.5 * b * conj(a)^3.'
    a = op.inputs[0]
    b = op.inputs[1]
    with ops.control_dependencies([grad]):
        ca = math_ops.conj(a)
        cg = math_ops.conj(grad)
        grad_a = -1.5 * cg * b * math_ops.square(ca)
        grad_b = gen_math_ops.rsqrt_grad(ca, grad)
        return (grad_a, grad_b)

@ops.RegisterGradient('Exp')
def _ExpGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    'Returns grad * exp(x).'
    y = op.outputs[0]
    with ops.control_dependencies([grad]):
        y = math_ops.conj(y)
        return grad * y

@ops.RegisterGradient('Expm1')
def _Expm1Grad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Returns grad * exp(x).'
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        x = math_ops.conj(x)
        y = math_ops.exp(x)
        return grad * y

@ops.RegisterGradient('Log')
def _LogGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Returns grad * (1/x).'
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        x = math_ops.conj(x)
        return grad * math_ops.reciprocal(x)

@ops.RegisterGradient('Log1p')
def _Log1pGrad(op: ops.Operation, grad):
    if False:
        return 10
    'Returns grad * (1/(1 + x)).'
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        x = math_ops.conj(x)
        return grad * math_ops.reciprocal(1 + x)

@ops.RegisterGradient('Xlogy')
def _XLogyGrad(op: ops.Operation, grad):
    if False:
        return 10
    'Returns gradient of xlogy(x, y) with respect to x and y.'
    x = op.inputs[0]
    y = op.inputs[1]
    sx = array_ops.shape(x)
    sy = array_ops.shape(y)
    (rx, ry) = gen_array_ops.broadcast_gradient_args(sx, sy)
    with ops.control_dependencies([grad]):
        not_zero_x = math_ops.cast(math_ops.not_equal(x, math_ops.cast(0.0, dtype=x.dtype)), dtype=x.dtype)
        partial_x = gen_math_ops.xlogy(not_zero_x, y)
        partial_y = gen_math_ops.xdivy(x, y)
        return (array_ops.reshape(math_ops.reduce_sum(partial_x * grad, rx), sx), array_ops.reshape(math_ops.reduce_sum(partial_y * grad, ry), sy))

@ops.RegisterGradient('Xlog1py')
def _XLog1pyGrad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    'Returns gradient of xlog1py(x, y) with respect to x and y.'
    x = op.inputs[0]
    y = op.inputs[1]
    sx = array_ops.shape(x)
    sy = array_ops.shape(y)
    (rx, ry) = gen_array_ops.broadcast_gradient_args(sx, sy)
    with ops.control_dependencies([grad]):
        not_zero_x = math_ops.cast(math_ops.not_equal(x, math_ops.cast(0.0, dtype=x.dtype)), dtype=x.dtype)
        partial_x = gen_math_ops.xlog1py(not_zero_x, y)
        partial_y = gen_math_ops.xdivy(x, y + 1.0)
        return (array_ops.reshape(math_ops.reduce_sum(partial_x * grad, rx), sx), array_ops.reshape(math_ops.reduce_sum(partial_y * grad, ry), sy))

@ops.RegisterGradient('Xdivy')
def _XDivyGrad(op: ops.Operation, grad):
    if False:
        return 10
    'Returns gradient of xdivy(x, y) with respect to x and y.'
    x = op.inputs[0]
    y = op.inputs[1]
    sx = array_ops.shape(x)
    sy = array_ops.shape(y)
    (rx, ry) = gen_array_ops.broadcast_gradient_args(sx, sy)
    with ops.control_dependencies([grad]):
        not_zero_x = math_ops.cast(math_ops.not_equal(x, math_ops.cast(0.0, dtype=x.dtype)), dtype=x.dtype)
        partial_x = gen_math_ops.xdivy(not_zero_x, y)
        partial_y = gen_math_ops.xdivy(math_ops.negative(x), y ** 2)
        return (array_ops.reshape(math_ops.reduce_sum(partial_x * grad, rx), sx), array_ops.reshape(math_ops.reduce_sum(partial_y * grad, ry), sy))

@ops.RegisterGradient('Sinh')
def _SinhGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Returns grad * cosh(x).'
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        x = math_ops.conj(x)
        return grad * math_ops.cosh(x)

@ops.RegisterGradient('Cosh')
def _CoshGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'Returns grad * sinh(x).'
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        x = math_ops.conj(x)
        return grad * math_ops.sinh(x)

@ops.RegisterGradient('Tanh')
def _TanhGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Returns grad * (1 - tanh(x) * tanh(x)).'
    y = op.outputs[0]
    with ops.control_dependencies([grad]):
        y = math_ops.conj(y)
        return gen_math_ops.tanh_grad(y, grad)

@ops.RegisterGradient('Asinh')
def _AsinhGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Returns grad * 1/cosh(y).'
    y = op.outputs[0]
    with ops.control_dependencies([grad]):
        y = math_ops.conj(y)
        return grad / math_ops.cosh(y)

@ops.RegisterGradient('Acosh')
def _AcoshGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Returns grad * 1/sinh(y).'
    y = op.outputs[0]
    with ops.control_dependencies([grad]):
        y = math_ops.conj(y)
        return grad / math_ops.sinh(y)

@ops.RegisterGradient('Atanh')
def _AtanhGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    'Returns grad * 1/ (1 - x^2).'
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        x = math_ops.conj(x)
        x2 = math_ops.square(x)
        one = constant_op.constant(1, dtype=grad.dtype)
        inv = math_ops.reciprocal(math_ops.subtract(one, x2))
        return grad * inv

@ops.RegisterGradient('TanhGrad')
def _TanhGradGrad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    with ops.control_dependencies([grad]):
        a = math_ops.conj(op.inputs[0])
        b = math_ops.conj(op.inputs[1])
        return (grad * -2.0 * b * a, gen_math_ops.tanh_grad(a, grad))

@ops.RegisterGradient('Erf')
def _ErfGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Returns grad * 2/sqrt(pi) * exp(-x**2).'
    x = op.inputs[0]
    two_over_root_pi = constant_op.constant(2 / np.sqrt(np.pi), dtype=grad.dtype)
    with ops.control_dependencies([grad]):
        x = math_ops.conj(x)
        return grad * two_over_root_pi * math_ops.exp(-math_ops.square(x))

@ops.RegisterGradient('Erfc')
def _ErfcGrad(op: ops.Operation, grad):
    if False:
        return 10
    'Returns -grad * 2/sqrt(pi) * exp(-x**2).'
    x = op.inputs[0]
    minus_two_over_root_pi = constant_op.constant(-2 / np.sqrt(np.pi), dtype=grad.dtype)
    with ops.control_dependencies([grad]):
        x = math_ops.conj(x)
        return grad * minus_two_over_root_pi * math_ops.exp(-math_ops.square(x))

@ops.RegisterGradient('Erfinv')
def _ErfinvGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'Returns grad * sqrt(pi) / 2 * exp(erfinv(x)**2).'
    root_pi_over_two = constant_op.constant(np.sqrt(np.pi) / 2, dtype=grad.dtype)
    with ops.control_dependencies([grad]):
        return grad * root_pi_over_two * math_ops.exp(math_ops.square(op.outputs[0]))

@ops.RegisterGradient('Ndtri')
def _NdtriGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Returns grad * sqrt(2 * pi) * exp(ndtri(x)**2 / 2).'
    root_two_pi = constant_op.constant(np.sqrt(2 * np.pi), dtype=grad.dtype)
    with ops.control_dependencies([grad]):
        return grad * root_two_pi * math_ops.exp(math_ops.square(op.outputs[0]) / 2.0)

@ops.RegisterGradient('Lgamma')
def _LgammaGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Returns grad * digamma(x).'
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        x = math_ops.conj(x)
        return grad * math_ops.digamma(x)

@ops.RegisterGradient('Digamma')
def _DigammaGrad(op: ops.Operation, grad):
    if False:
        return 10
    'Compute gradient of the digamma function with respect to its argument.'
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        x = math_ops.conj(x)
        partial_x = math_ops.polygamma(array_ops.constant(1, dtype=x.dtype), x)
        return grad * partial_x

@ops.RegisterGradient('Dawsn')
def _DawsnGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    'Compute gradient of dawsn(x) with respect to its argument.'
    x = op.inputs[0]
    y = op.outputs[0]
    with ops.control_dependencies([grad]):
        return grad * (1.0 - 2 * x * y)

@ops.RegisterGradient('Expint')
def _ExpintGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    'Compute gradient of expint(x) with respect to its argument.'
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        return grad * math_ops.exp(x) / x

@ops.RegisterGradient('FresnelCos')
def _FresnelCosGrad(op: ops.Operation, grad):
    if False:
        return 10
    'Compute gradient of fresnel_cos(x) with respect to its argument.'
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        return grad * math_ops.cos(np.pi / 2.0 * math_ops.square(x))

@ops.RegisterGradient('FresnelSin')
def _FresnelSinGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    'Compute gradient of fresnel_sin(x) with respect to its argument.'
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        return grad * math_ops.sin(np.pi / 2.0 * math_ops.square(x))

@ops.RegisterGradient('Spence')
def _SpenceGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'Compute gradient of spence(x) with respect to its argument.'
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        partial_x = math_ops.log(x) / (1 - x)
        partial_x = array_ops.where(math_ops.equal(x, 1.0), -array_ops.ones_like(x), partial_x)
        return grad * partial_x

@ops.RegisterGradient('BesselI0')
def _BesselI0Grad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    'Compute gradient of bessel_i0(x) with respect to its argument.'
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        partial_x = special_math_ops.bessel_i1(x)
        return grad * partial_x

@ops.RegisterGradient('BesselI0e')
def _BesselI0eGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Compute gradient of bessel_i0e(x) with respect to its argument.'
    x = op.inputs[0]
    y = op.outputs[0]
    with ops.control_dependencies([grad]):
        partial_x = special_math_ops.bessel_i1e(x) - math_ops.sign(x) * y
        return grad * partial_x

@ops.RegisterGradient('BesselI1')
def _BesselI1Grad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Compute gradient of bessel_i1(x) with respect to its argument.'
    x = op.inputs[0]
    y = op.outputs[0]
    with ops.control_dependencies([grad]):
        dy_dx = array_ops.where_v2(math_ops.equal(x, 0.0), math_ops.cast(1.0, x.dtype), special_math_ops.bessel_i0(x) - math_ops.div(y, x))
        return grad * dy_dx

@ops.RegisterGradient('BesselI1e')
def _BesselI1eGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    'Compute gradient of bessel_i1e(x) with respect to its argument.'
    x = op.inputs[0]
    y = op.outputs[0]
    with ops.control_dependencies([grad]):
        dy_dx = array_ops.where_v2(math_ops.equal(x, 0.0), math_ops.cast(0.5, x.dtype), special_math_ops.bessel_i0e(x) - y * (math_ops.sign(x) + math_ops.reciprocal(x)))
        return grad * dy_dx

@ops.RegisterGradient('BesselK0')
def _BesselK0Grad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    'Compute gradient of bessel_k0(x) with respect to its argument.'
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        partial_x = -special_math_ops.bessel_k1(x)
        return grad * partial_x

@ops.RegisterGradient('BesselK0e')
def _BesselK0eGrad(op: ops.Operation, grad):
    if False:
        return 10
    'Compute gradient of bessel_k0e(x) with respect to its argument.'
    x = op.inputs[0]
    y = op.outputs[0]
    with ops.control_dependencies([grad]):
        partial_x = y - special_math_ops.bessel_k1e(x)
        return grad * partial_x

@ops.RegisterGradient('BesselK1')
def _BesselK1Grad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    'Compute gradient of bessel_k1(x) with respect to its argument.'
    x = op.inputs[0]
    y = op.outputs[0]
    with ops.control_dependencies([grad]):
        partial_x = -special_math_ops.bessel_k0(x) - math_ops.div(y, x)
        return grad * partial_x

@ops.RegisterGradient('BesselK1e')
def _BesselK1eGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'Compute gradient of bessel_k1e(x) with respect to its argument.'
    x = op.inputs[0]
    y = op.outputs[0]
    with ops.control_dependencies([grad]):
        partial_x = y * (1.0 - math_ops.reciprocal(x)) - special_math_ops.bessel_k0e(x)
        return grad * partial_x

@ops.RegisterGradient('BesselJ0')
def _BesselJ0Grad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    'Compute gradient of bessel_j0(x) with respect to its argument.'
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        partial_x = -special_math_ops.bessel_j1(x)
        return grad * partial_x

@ops.RegisterGradient('BesselJ1')
def _BesselJ1Grad(op: ops.Operation, grad):
    if False:
        return 10
    'Compute gradient of bessel_j1(x) with respect to its argument.'
    x = op.inputs[0]
    y = op.outputs[0]
    with ops.control_dependencies([grad]):
        dy_dx = array_ops.where_v2(math_ops.equal(x, 0.0), math_ops.cast(0.5, x.dtype), special_math_ops.bessel_j0(x) - math_ops.div(y, x))
        return grad * dy_dx

@ops.RegisterGradient('BesselY0')
def _BesselY0Grad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    'Compute gradient of bessel_y0(x) with respect to its argument.'
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        partial_x = -special_math_ops.bessel_y1(x)
        return grad * partial_x

@ops.RegisterGradient('BesselY1')
def _BesselY1Grad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Compute gradient of bessel_y1(x) with respect to its argument.'
    x = op.inputs[0]
    y = op.outputs[0]
    with ops.control_dependencies([grad]):
        partial_x = special_math_ops.bessel_y0(x) - math_ops.div(y, x)
        return grad * partial_x

@ops.RegisterGradient('Igamma')
def _IgammaGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    'Returns gradient of igamma(a, x) with respect to a and x.'
    a = op.inputs[0]
    x = op.inputs[1]
    sa = array_ops.shape(a)
    sx = array_ops.shape(x)
    (ra, rx) = gen_array_ops.broadcast_gradient_args(sa, sx)
    with ops.control_dependencies([grad]):
        partial_a = gen_math_ops.igamma_grad_a(a, x)
        partial_x = math_ops.exp(-x + (a - 1) * math_ops.log(x) - math_ops.lgamma(a))
        return (array_ops.reshape(math_ops.reduce_sum(partial_a * grad, ra), sa), array_ops.reshape(math_ops.reduce_sum(partial_x * grad, rx), sx))

@ops.RegisterGradient('Igammac')
def _IgammacGrad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    'Returns gradient of igammac(a, x) = 1 - igamma(a, x) w.r.t. a and x.'
    (igamma_grad_a, igamma_grad_x) = _IgammaGrad(op, grad)
    return (-igamma_grad_a, -igamma_grad_x)

@ops.RegisterGradient('Betainc')
def _BetaincGrad(op: ops.Operation, grad):
    if False:
        return 10
    'Returns gradient of betainc(a, b, x) with respect to x.'
    (a, b, x) = op.inputs
    sa = array_ops.shape(a)
    sx = array_ops.shape(x)
    (_, rx) = gen_array_ops.broadcast_gradient_args(sa, sx)
    log_beta = gen_math_ops.lgamma(a) + gen_math_ops.lgamma(b) - gen_math_ops.lgamma(a + b)
    partial_x = math_ops.exp(math_ops.xlog1py(b - 1, -x) + math_ops.xlogy(a - 1, x) - log_beta)
    return (None, None, array_ops.reshape(math_ops.reduce_sum(partial_x * grad, rx), sx))

@ops.RegisterGradient('Zeta')
def _ZetaGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'Returns gradient of zeta(x, q) with respect to x and q.'
    x = op.inputs[0]
    q = op.inputs[1]
    sx = array_ops.shape(x)
    sq = array_ops.shape(q)
    (unused_rx, rq) = gen_array_ops.broadcast_gradient_args(sx, sq)
    with ops.control_dependencies([grad]):
        x = math_ops.conj(x)
        q = math_ops.conj(q)
        partial_q = -x * math_ops.zeta(x + 1, q)
        return (None, array_ops.reshape(math_ops.reduce_sum(partial_q * grad, rq), sq))

@ops.RegisterGradient('Polygamma')
def _PolygammaGrad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    'Returns gradient of psi(n, x) with respect to n and x.'
    n = op.inputs[0]
    x = op.inputs[1]
    sn = array_ops.shape(n)
    sx = array_ops.shape(x)
    (unused_rn, rx) = gen_array_ops.broadcast_gradient_args(sn, sx)
    with ops.control_dependencies([grad]):
        n = math_ops.conj(n)
        x = math_ops.conj(x)
        partial_x = math_ops.polygamma(n + 1, x)
        return (None, array_ops.reshape(math_ops.reduce_sum(partial_x * grad, rx), sx))

@ops.RegisterGradient('Sigmoid')
def _SigmoidGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Returns grad * sigmoid(x) * (1 - sigmoid(x)).'
    y = op.outputs[0]
    with ops.control_dependencies([grad]):
        y = math_ops.conj(y)
        return gen_math_ops.sigmoid_grad(y, grad)

@ops.RegisterGradient('SigmoidGrad')
def _SigmoidGradGrad(op: ops.Operation, grad):
    if False:
        return 10
    with ops.control_dependencies([grad]):
        a = math_ops.conj(op.inputs[0])
        b = math_ops.conj(op.inputs[1])
        gb = grad * b
        return (gb - 2.0 * gb * a, gen_math_ops.sigmoid_grad(a, grad))

@ops.RegisterGradient('Sign')
def _SignGrad(op: ops.Operation, _):
    if False:
        print('Hello World!')
    'Returns 0.'
    x = op.inputs[0]
    return array_ops.zeros_like(x)

@ops.RegisterGradient('Sin')
def _SinGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'Returns grad * cos(x).'
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        x = math_ops.conj(x)
        return grad * math_ops.cos(x)

@ops.RegisterGradient('Cos')
def _CosGrad(op: ops.Operation, grad):
    if False:
        return 10
    'Returns grad * -sin(x).'
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        x = math_ops.conj(x)
        return -grad * math_ops.sin(x)

@ops.RegisterGradient('Tan')
def _TanGrad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    'Returns grad * 1/sec^2(x).'
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        x = math_ops.conj(x)
        secx = math_ops.reciprocal(math_ops.cos(x))
        secx2 = math_ops.square(secx)
        return secx2 * grad

@ops.RegisterGradient('Asin')
def _AsinGrad(op: ops.Operation, grad):
    if False:
        return 10
    'Returns grad * 1/sqrt(1-x^2).'
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        x = math_ops.conj(x)
        x2 = math_ops.square(x)
        one = constant_op.constant(1, dtype=grad.dtype)
        den = math_ops.sqrt(math_ops.subtract(one, x2))
        inv = math_ops.reciprocal(den)
        return grad * inv

@ops.RegisterGradient('Acos')
def _AcosGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'Returns grad * -1/sqrt(1-x^2).'
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        x = math_ops.conj(x)
        x2 = math_ops.square(x)
        one = constant_op.constant(1, dtype=grad.dtype)
        den = math_ops.sqrt(math_ops.subtract(one, x2))
        inv = math_ops.reciprocal(den)
        return -grad * inv

@ops.RegisterGradient('Atan')
def _AtanGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    'Returns grad * 1/ (1 + x^2).'
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        x = math_ops.conj(x)
        x2 = math_ops.square(x)
        one = constant_op.constant(1, dtype=grad.dtype)
        inv = math_ops.reciprocal(math_ops.add(one, x2))
        return grad * inv

@ops.RegisterGradient('Atan2')
def _Atan2Grad(op: ops.Operation, grad):
    if False:
        return 10
    'Returns grad * x / (y^2 + x^2), grad * -y / (y^2 + x^2).'
    y = op.inputs[0]
    x = op.inputs[1]
    with ops.control_dependencies([grad]):
        grad_inv = grad / (math_ops.square(y) + math_ops.square(x))
        gy = x * grad_inv
        gx = -y * grad_inv
        return _ReduceGradientArgs(y, x, gy, gx)

@ops.RegisterGradient('AddN')
def _AddNGrad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    'Copies the gradient to all inputs.'
    return [grad] * len(op.inputs)

def _ShapesFullySpecifiedAndEqual(x, y, grad):
    if False:
        for i in range(10):
            print('nop')
    x_shape = x._shape_tuple()
    y_shape = y._shape_tuple()
    grad_shape = grad._shape_tuple()
    return x_shape == y_shape and x_shape == grad_shape and (x_shape is not None) and (None not in x_shape)

@ops.RegisterGradient('Add')
@ops.RegisterGradient('AddV2')
def _AddGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'Gradient for Add.'
    y = op.inputs[1]
    try:
        skip_input_indices = op.skip_input_indices or ()
        if 1 in skip_input_indices and _IsScalar(y):
            return (grad, None)
    except AttributeError:
        skip_input_indices = ()
    x = op.inputs[0]
    if isinstance(grad, tensor.Tensor) and _ShapesFullySpecifiedAndEqual(x, y, grad):
        return (grad, grad)
    gx = None if 0 in skip_input_indices else grad
    gy = None if 1 in skip_input_indices else grad
    return _ReduceGradientArgs(x, y, gx, gy)

@ops.RegisterGradient('Sub')
def _SubGrad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    'Gradient for Sub.'
    y = op.inputs[1]
    try:
        skip_input_indices = op.skip_input_indices or ()
        if 1 in skip_input_indices and _IsScalar(y):
            return (grad, None)
    except AttributeError:
        skip_input_indices = ()
    x = op.inputs[0]
    if isinstance(grad, tensor.Tensor) and _ShapesFullySpecifiedAndEqual(x, y, grad):
        return (grad, -grad)
    gx = None if 0 in skip_input_indices else grad
    gy = None if 1 in skip_input_indices else -grad
    return _ReduceGradientArgs(x, y, gx, gy)

@ops.RegisterGradient('Mul')
def _MulGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'The gradient of scalar multiplication.'
    y = op.inputs[1]
    try:
        skip_input_indices = op.skip_input_indices or ()
        if 1 in skip_input_indices and _IsScalar(y):
            return (gen_math_ops.mul(grad, math_ops.conj(y)), None)
    except AttributeError:
        skip_input_indices = ()
    x = op.inputs[0]
    if isinstance(grad, tensor.Tensor) and _ShapesFullySpecifiedAndEqual(x, y, grad) and (grad.dtype in (dtypes.int32, dtypes.float32)):
        return (gen_math_ops.mul(grad, y), gen_math_ops.mul(grad, x))
    assert x.dtype.base_dtype == y.dtype.base_dtype, (x.dtype, ' vs. ', y.dtype)
    if 0 in skip_input_indices:
        gx = None
    else:
        gx = gen_math_ops.mul(grad, math_ops.conj(y))
    if 1 in skip_input_indices:
        gy = None
    else:
        gy = gen_math_ops.mul(math_ops.conj(x), grad)
    return _ReduceGradientArgs(x, y, gx, gy)

@ops.RegisterGradient('MulNoNan')
def _MulNoNanGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'The gradient of scalar multiplication with NaN-suppression.'
    x = op.inputs[0]
    y = op.inputs[1]
    if isinstance(grad, tensor.Tensor) and _ShapesFullySpecifiedAndEqual(x, y, grad):
        return (gen_math_ops.mul_no_nan(grad, y), gen_math_ops.mul_no_nan(x, grad))
    assert x.dtype.base_dtype == y.dtype.base_dtype, (x.dtype, ' vs. ', y.dtype)
    gx = gen_math_ops.mul_no_nan(grad, y)
    gy = gen_math_ops.mul_no_nan(x, grad)
    return _ReduceGradientArgs(x, y, gx, gy)

@ops.RegisterGradient('Div')
def _DivGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    'The gradient for the Div operator.'
    x = op.inputs[0]
    y = op.inputs[1]
    cx = math_ops.conj(x)
    cy = math_ops.conj(y)
    gx = math_ops.divide(grad, cy)
    gy = grad * math_ops.divide(math_ops.divide(-cx, cy), cy)
    return _ReduceGradientArgs(x, y, gx, gy)

@ops.RegisterGradient('FloorDiv')
def _FloorDivGrad(_, unused_grad):
    if False:
        i = 10
        return i + 15
    'The gradient for the FloorDiv operator.'
    return (None, None)

@ops.RegisterGradient('FloorMod')
def _FloorModGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'Returns grad * (1, -floor(x/y)).'
    x = math_ops.conj(op.inputs[0])
    y = math_ops.conj(op.inputs[1])
    floor_xy = math_ops.floor_div(x, y)
    gx = grad
    gy = grad * math_ops.negative(floor_xy)
    return _ReduceGradientArgs(x, y, gx, gy)

@ops.RegisterGradient('TruncateDiv')
def _TruncateDivGrad(_, unused_grad):
    if False:
        i = 10
        return i + 15
    return (None, None)

@ops.RegisterGradient('RealDiv')
def _RealDivGrad(op: ops.Operation, grad):
    if False:
        return 10
    'RealDiv op gradient.'
    x = op.inputs[0]
    y = op.inputs[1]
    cx = math_ops.conj(op.inputs[0])
    cy = math_ops.conj(op.inputs[1])
    gx = math_ops.realdiv(grad, cy)
    gy = grad * math_ops.realdiv(math_ops.realdiv(-cx, cy), cy)
    return _ReduceGradientArgs(x, y, gx, gy)

@ops.RegisterGradient('DivNoNan')
def _DivNoNanGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    'DivNoNan op gradient.'
    x = math_ops.conj(op.inputs[0])
    y = math_ops.conj(op.inputs[1])
    gx = math_ops.div_no_nan(grad, y)
    gy = grad * math_ops.div_no_nan(math_ops.div_no_nan(-x, y), y)
    return _ReduceGradientArgs(x, y, gx, gy)

@ops.RegisterGradient('Pow')
def _PowGrad(op: ops.Operation, grad):
    if False:
        return 10
    'Returns grad * (y*x^(y-1), z*log(x)).'
    x = op.inputs[0]
    y = op.inputs[1]
    cx = math_ops.conj(x)
    cy = math_ops.conj(y)
    try:
        skip_input_indices = op.skip_input_indices or ()
        if 1 in skip_input_indices and _IsScalar(y):
            return (grad * cy * math_ops.pow(cx, cy - 1), None)
    except AttributeError:
        skip_input_indices = ()
    if 0 in skip_input_indices:
        gx = None
    else:
        gx = grad * cy * math_ops.pow(cx, cy - 1)
    if 1 in skip_input_indices:
        gy = None
    else:
        if x.dtype.is_complex:
            mask = math_ops.not_equal(cx, 0)
        else:
            mask = cx > 0
        safe_x = array_ops.where(mask, cx, array_ops.ones_like(x))
        log_x = array_ops.where(mask, math_ops.log(safe_x), array_ops.zeros_like(x))
        gy = grad * math_ops.conj(op.outputs[0]) * log_x
    return _ReduceGradientArgs(x, y, gx, gy)

def _MaximumMinimumGradInputOnly(op: ops.Operation, grad, selector_op):
    if False:
        print('Hello World!')
    x = op.inputs[0]
    y = op.inputs[1]
    zeros = array_ops.zeros_like(grad)
    xmask = selector_op(x, y)
    xgrad = array_ops.where_v2(xmask, grad, zeros)
    ygrad = None
    return (xgrad, ygrad)

def _MaximumMinimumGrad(op: ops.Operation, grad, selector_op):
    if False:
        print('Hello World!')
    'Factor out the code for the gradient of Maximum or Minimum.'
    y = op.inputs[1]
    try:
        skip_input_indices = op.skip_input_indices or ()
        if 1 in skip_input_indices and _IsScalar(y):
            return _MaximumMinimumGradInputOnly(op, grad, selector_op)
    except AttributeError:
        skip_input_indices = ()
    x = op.inputs[0]
    zeros = array_ops.zeros_like(grad)
    xmask = selector_op(x, y)
    if 0 in skip_input_indices:
        gx = None
    else:
        gx = array_ops.where_v2(xmask, grad, zeros)
    if 1 in skip_input_indices:
        gy = None
    else:
        gy = array_ops.where_v2(xmask, zeros, grad)
    return _ReduceGradientArgs(x, y, gx, gy)

@ops.RegisterGradient('Maximum')
def _MaximumGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Returns grad*(x >= y, x < y) with type of grad.'
    return _MaximumMinimumGrad(op, grad, math_ops.greater_equal)

@ops.RegisterGradient('Minimum')
def _MinimumGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Returns grad*(x <= y, x > y) with type of grad.'
    return _MaximumMinimumGrad(op, grad, math_ops.less_equal)

@ops.RegisterGradient('SquaredDifference')
def _SquaredDifferenceGrad(op: ops.Operation, grad):
    if False:
        return 10
    'Returns the gradient for (x-y)^2.'
    x = op.inputs[0]
    y = op.inputs[1]
    try:
        skip_input_indices = op.skip_input_indices or ()
    except AttributeError:
        skip_input_indices = ()
    with ops.control_dependencies([grad]):
        x_grad = math_ops.scalar_mul(2.0, grad) * (x - y)
    if isinstance(grad, tensor.Tensor) and _ShapesFullySpecifiedAndEqual(x, y, grad):
        return (x_grad, -x_grad)
    gx = None if 0 in skip_input_indices else x_grad
    gy = None if 1 in skip_input_indices else -x_grad
    return _ReduceGradientArgs(x, y, gx, gy)
ops.NotDifferentiable('Less')
ops.NotDifferentiable('LessEqual')
ops.NotDifferentiable('Greater')
ops.NotDifferentiable('GreaterEqual')
ops.NotDifferentiable('Equal')
ops.NotDifferentiable('ApproximateEqual')
ops.NotDifferentiable('NotEqual')
ops.NotDifferentiable('LogicalAnd')
ops.NotDifferentiable('LogicalOr')
ops.NotDifferentiable('LogicalNot')

@ops.RegisterGradient('Select')
def _SelectGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    c = op.inputs[0]
    x = op.inputs[1]
    zeros = array_ops.zeros_like(x)
    return (None, array_ops.where(c, grad, zeros), array_ops.where(c, zeros, grad))

@ops.RegisterGradient('SelectV2')
def _SelectGradV2(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    c = op.inputs[0]
    x = op.inputs[1]
    y = op.inputs[2]
    z = op.outputs[0]
    zeros = array_ops.zeros([], dtype=grad.dtype.base_dtype)
    gx = array_ops.where_v2(c, grad, zeros)
    gy = array_ops.where_v2(c, zeros, grad)
    (gx, _) = _ReduceGradientArgs(x, z, gx, None)
    (gy, _) = _ReduceGradientArgs(y, z, gy, None)
    return (None, gx, gy)

def _MatMulGradAgainstFirstOnly(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    'Gradient for MatMul, only for the first input.'
    t_a = op.get_attr('transpose_a')
    t_b = op.get_attr('transpose_b')
    b = math_ops.conj(op.inputs[1])
    if not t_a and (not t_b):
        grad_a = gen_math_ops.mat_mul(grad, b, transpose_b=True)
    elif not t_a and t_b:
        grad_a = gen_math_ops.mat_mul(grad, b)
    elif t_a and (not t_b):
        grad_a = gen_math_ops.mat_mul(b, grad, transpose_b=True)
    elif t_a and t_b:
        grad_a = gen_math_ops.mat_mul(b, grad, transpose_a=True, transpose_b=True)
    return (grad_a, None)

def _MatMulGradAgainstSecondOnly(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    'Gradient for MatMul, only for the second input.'
    t_a = op.get_attr('transpose_a')
    t_b = op.get_attr('transpose_b')
    a = math_ops.conj(op.inputs[0])
    if not t_a and (not t_b):
        grad_b = gen_math_ops.mat_mul(a, grad, transpose_a=True)
    elif not t_a and t_b:
        grad_b = gen_math_ops.mat_mul(grad, a, transpose_a=True)
    elif t_a and (not t_b):
        grad_b = gen_math_ops.mat_mul(a, grad)
    elif t_a and t_b:
        grad_b = gen_math_ops.mat_mul(grad, a, transpose_a=True, transpose_b=True)
    return (None, grad_b)

@ops.RegisterGradient('MatMul')
def _MatMulGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Gradient for MatMul.'
    try:
        skip_input_indices = op.skip_input_indices
        if skip_input_indices is not None:
            if 1 in skip_input_indices:
                return _MatMulGradAgainstFirstOnly(op, grad)
            elif 0 in skip_input_indices:
                return _MatMulGradAgainstSecondOnly(op, grad)
    except AttributeError:
        pass
    t_a = op.get_attr('transpose_a')
    t_b = op.get_attr('transpose_b')
    a = math_ops.conj(op.inputs[0])
    b = math_ops.conj(op.inputs[1])
    if not t_a and (not t_b):
        grad_a = gen_math_ops.mat_mul(grad, b, transpose_b=True)
        grad_b = gen_math_ops.mat_mul(a, grad, transpose_a=True)
    elif not t_a and t_b:
        grad_a = gen_math_ops.mat_mul(grad, b)
        grad_b = gen_math_ops.mat_mul(grad, a, transpose_a=True)
    elif t_a and (not t_b):
        grad_a = gen_math_ops.mat_mul(b, grad, transpose_b=True)
        grad_b = gen_math_ops.mat_mul(a, grad)
    elif t_a and t_b:
        grad_a = gen_math_ops.mat_mul(b, grad, transpose_a=True, transpose_b=True)
        grad_b = gen_math_ops.mat_mul(grad, a, transpose_a=True, transpose_b=True)
    return (grad_a, grad_b)

@ops.RegisterGradient('SparseMatMul')
def _SparseMatMulGrad(op: ops.Operation, grad):
    if False:
        return 10
    'Gradient for SparseMatMul.'
    t_a = op.get_attr('transpose_a')
    t_b = op.get_attr('transpose_b')
    is_sparse = {}
    is_sparse[op.inputs[0].ref()] = op.get_attr('a_is_sparse')
    is_sparse[op.inputs[1].ref()] = op.get_attr('b_is_sparse')
    is_sparse[grad.ref()] = not context.executing_eagerly() and grad.op.type == 'ReluGrad'

    def _SparseMatMul(t1, t2, out_dtype, transpose_a=False, transpose_b=False):
        if False:
            i = 10
            return i + 15
        'Helper function to create SparseMatMul op.'
        assert t1.ref() in is_sparse and t2.ref() in is_sparse
        t1_sparse = is_sparse[t1.ref()]
        t2_sparse = is_sparse[t2.ref()]
        if transpose_b:
            t2 = array_ops.transpose(t2)
            transpose_b = False
        prod = math_ops.matmul(t1, t2, transpose_a=transpose_a, transpose_b=transpose_b, a_is_sparse=t1_sparse, b_is_sparse=t2_sparse)
        if prod.dtype != out_dtype:
            prod = math_ops.cast(prod, out_dtype)
        return prod
    dtype_a = op.inputs[0].dtype
    dtype_b = op.inputs[1].dtype
    if not t_a and (not t_b):
        return (_SparseMatMul(grad, op.inputs[1], dtype_a, transpose_b=True), _SparseMatMul(op.inputs[0], grad, dtype_b, transpose_a=True))
    elif not t_a and t_b:
        return (_SparseMatMul(grad, op.inputs[1], dtype_a), _SparseMatMul(grad, op.inputs[0], dtype_b, transpose_a=True))
    elif t_a and (not t_b):
        return (_SparseMatMul(op.inputs[1], grad, dtype_a, transpose_b=True), _SparseMatMul(op.inputs[0], grad, dtype_b))
    elif t_a and t_b:
        return (_SparseMatMul(op.inputs[1], grad, dtype_a, transpose_a=True, transpose_b=True), _SparseMatMul(grad, op.inputs[0], dtype_b, transpose_a=True, transpose_b=True))

@ops.RegisterGradient('Floor')
def _FloorGrad(_, unused_grad):
    if False:
        while True:
            i = 10
    return [None]

@ops.RegisterGradient('Ceil')
def _CeilGrad(_, unused_grad):
    if False:
        return 10
    return [None]

@ops.RegisterGradient('Round')
def _RoundGrad(_, unused_grad):
    if False:
        i = 10
        return i + 15
    return [None]

@ops.RegisterGradient('Rint')
def _RintGrad(_, unused_grad):
    if False:
        i = 10
        return i + 15
    return [None]

@ops.RegisterGradient('BatchMatMul')
def _BatchMatMul(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Returns the gradient of x and y given the gradient of x * y.'
    x = op.inputs[0]
    y = op.inputs[1]
    adj_x = op.get_attr('adj_x')
    adj_y = op.get_attr('adj_y')
    if not adj_x:
        if not adj_y:
            grad_x = math_ops.matmul(grad, y, adjoint_a=False, adjoint_b=True)
            grad_y = math_ops.matmul(x, grad, adjoint_a=True, adjoint_b=False)
        else:
            grad_x = math_ops.matmul(grad, y, adjoint_a=False, adjoint_b=False)
            grad_y = math_ops.matmul(grad, x, adjoint_a=True, adjoint_b=False)
    elif not adj_y:
        grad_x = math_ops.matmul(y, grad, adjoint_a=False, adjoint_b=True)
        grad_y = math_ops.matmul(x, grad, adjoint_a=False, adjoint_b=False)
    else:
        grad_x = math_ops.matmul(y, grad, adjoint_a=True, adjoint_b=True)
        grad_y = math_ops.matmul(grad, x, adjoint_a=True, adjoint_b=True)
    return (grad_x, grad_y)

@ops.RegisterGradient('BatchMatMulV2')
@ops.RegisterGradient('BatchMatMulV3')
def _BatchMatMulV2(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    'Returns the gradient of x and y given the gradient of x * y.'
    x = op.inputs[0]
    y = op.inputs[1]
    adj_x = op.get_attr('adj_x')
    adj_y = op.get_attr('adj_y')
    if not adj_x:
        if not adj_y:
            grad_x = math_ops.matmul(grad, y, adjoint_a=False, adjoint_b=True)
            grad_y = math_ops.matmul(x, grad, adjoint_a=True, adjoint_b=False)
        else:
            grad_x = math_ops.matmul(grad, y, adjoint_a=False, adjoint_b=False)
            grad_y = math_ops.matmul(grad, x, adjoint_a=True, adjoint_b=False)
    elif not adj_y:
        grad_x = math_ops.matmul(y, grad, adjoint_a=False, adjoint_b=True)
        grad_y = math_ops.matmul(x, grad, adjoint_a=False, adjoint_b=False)
    else:
        grad_x = math_ops.matmul(y, grad, adjoint_a=True, adjoint_b=True)
        grad_y = math_ops.matmul(grad, x, adjoint_a=True, adjoint_b=True)
    shape_x_static = x.get_shape()
    shape_y_static = y.get_shape()
    output_may_have_non_empty_batch_shape = (shape_x_static.rank is None or shape_x_static.rank > 2) or (shape_y_static.rank is None or shape_y_static.rank > 2)
    batch_shapes_match = shape_x_static[:-2].is_fully_defined() and shape_y_static[:-2].is_fully_defined() and (shape_x_static[:-2] == shape_y_static[:-2])
    if not output_may_have_non_empty_batch_shape or batch_shapes_match:
        return (grad_x, grad_y)
    sx = array_ops.shape(x)
    sy = array_ops.shape(y)
    (rx, ry) = gen_array_ops.broadcast_gradient_args(sx[:-2], sy[:-2])
    grad_x = array_ops.reshape(math_ops.reduce_sum(grad_x, rx), sx)
    grad_y = array_ops.reshape(math_ops.reduce_sum(grad_y, ry), sy)
    return (grad_x, grad_y)
ops.NotDifferentiable('Range')
ops.NotDifferentiable('LinSpace')

@ops.RegisterGradient('Complex')
def _ComplexGrad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    "Returns the real and imaginary components of 'grad', respectively."
    x = op.inputs[0]
    y = op.inputs[1]
    gx = math_ops.real(grad)
    gy = math_ops.imag(grad)
    return _ReduceGradientArgs(x, y, gx, gy)

@ops.RegisterGradient('Real')
def _RealGrad(_, grad):
    if False:
        return 10
    "Returns 'grad' as the real part and set the imaginary part 0."
    zero = constant_op.constant(0, dtype=grad.dtype)
    return math_ops.complex(grad, zero)

@ops.RegisterGradient('Imag')
def _ImagGrad(_, grad):
    if False:
        while True:
            i = 10
    "Returns 'grad' as the imaginary part and set the real part 0."
    zero = constant_op.constant(0, dtype=grad.dtype)
    return math_ops.complex(zero, grad)

@ops.RegisterGradient('Angle')
def _AngleGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Returns `-grad / (Im(x) + i Re(x))`.'
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        re = math_ops.real(x)
        im = math_ops.imag(x)
        z = math_ops.reciprocal(math_ops.complex(im, re))
        zero = constant_op.constant(0, dtype=grad.dtype)
        complex_grad = math_ops.complex(grad, zero)
        return -complex_grad * z

@ops.RegisterGradient('Conj')
def _ConjGrad(_, grad):
    if False:
        while True:
            i = 10
    'Returns the complex conjugate of grad.'
    return math_ops.conj(grad)

@ops.RegisterGradient('ComplexAbs')
def _ComplexAbsGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Returns the gradient of ComplexAbs.'
    return math_ops.div_no_nan(math_ops.complex(grad, array_ops.zeros_like(grad)) * op.inputs[0], math_ops.complex(op.outputs[0], array_ops.zeros_like(op.outputs[0])))

@ops.RegisterGradient('Cast')
def _CastGrad(op: ops.Operation, grad):
    if False:
        return 10
    t = [dtypes.float16, dtypes.float32, dtypes.float64, dtypes.bfloat16, dtypes.complex64, dtypes.complex128]
    src_type = op.inputs[0].dtype.base_dtype
    dst_type = grad.dtype.base_dtype
    if src_type in t and dst_type in t:
        return math_ops.cast(grad, src_type)
    else:
        return None

@ops.RegisterGradient('Cross')
def _CrossGrad(op: ops.Operation, grad):
    if False:
        for i in range(10):
            print('nop')
    u = op.inputs[0]
    v = op.inputs[1]
    return (math_ops.cross(v, grad), math_ops.cross(grad, u))

@ops.RegisterGradient('Cumsum')
def _CumsumGrad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    axis = op.inputs[1]
    exclusive = op.get_attr('exclusive')
    reverse = op.get_attr('reverse')
    return [math_ops.cumsum(grad, axis, exclusive=exclusive, reverse=not reverse), None]

@ops.RegisterGradient('Cumprod')
def _CumprodGrad(op: ops.Operation, grad):
    if False:
        print('Hello World!')
    x = op.inputs[0]
    axis = op.inputs[1]
    exclusive = op.get_attr('exclusive')
    reverse = op.get_attr('reverse')
    prod = math_ops.cumprod(x, axis, exclusive=exclusive, reverse=reverse)
    out = math_ops.cumsum(prod * grad, axis, exclusive=exclusive, reverse=not reverse)
    return [math_ops.div_no_nan(out, x), None]

@ops.RegisterGradient('CumulativeLogsumexp')
def _CumulativeLogsumexpGrad(op: ops.Operation, grad):
    if False:
        while True:
            i = 10
    x = op.inputs[0]
    axis = op.inputs[1]
    cumulative_logsumexp = op.outputs[0]
    exclusive = op.get_attr('exclusive')
    reverse = op.get_attr('reverse')
    log_grad_positive = array_ops.where_v2(math_ops.greater(grad, 0), math_ops.log(grad), grad.dtype.min)
    log_grad_negative = array_ops.where_v2(math_ops.less(grad, 0), math_ops.log(-grad), grad.dtype.min)
    output_pos = math_ops.exp(math_ops.cumulative_logsumexp(log_grad_positive - cumulative_logsumexp, axis=axis, reverse=not reverse, exclusive=exclusive) + x)
    output_neg = math_ops.exp(math_ops.cumulative_logsumexp(log_grad_negative - cumulative_logsumexp, axis=axis, reverse=not reverse, exclusive=exclusive) + x)
    return [output_pos - output_neg, None]

@ops.RegisterGradient('NextAfter')
def _NextAfterGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Returns gradient of nextafter(x1, x2) with respect to x1 and x2.'
    x1 = op.inputs[0]
    x2 = op.inputs[1]
    s_x1 = array_ops.shape(x1)
    s_x2 = array_ops.shape(x2)
    (r_x1, r_x2) = gen_array_ops.broadcast_gradient_args(s_x1, s_x2)
    with ops.control_dependencies([grad]):
        partial_x1 = array_ops.ones(s_x1, dtype=x1.dtype)
        partial_x2 = array_ops.zeros(s_x2, dtype=x2.dtype)
        return (array_ops.reshape(math_ops.reduce_sum(partial_x1 * grad, r_x1), s_x1), array_ops.reshape(math_ops.reduce_sum(partial_x2 * grad, r_x2), s_x2))