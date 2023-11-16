"""Operator Squeeze for RaggedTensors."""
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch

@dispatch.dispatch_for_api(array_ops.squeeze_v2)
def squeeze(input: ragged_tensor.Ragged, axis=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Ragged compatible squeeze.\n\n  If `input` is a `tf.Tensor`, then this calls `tf.squeeze`.\n\n  If `input` is a `tf.RaggedTensor`, then this operation takes `O(N)` time,\n  where `N` is the number of elements in the squeezed dimensions.\n\n  Args:\n    input: A potentially ragged tensor. The input to squeeze.\n    axis: An optional list of ints. Defaults to `None`. If the `input` is\n      ragged, it only squeezes the dimensions listed. It fails if `input` is\n      ragged and axis is []. If `input` is not ragged it calls tf.squeeze. Note\n      that it is an error to squeeze a dimension that is not 1. It must be in\n      the range of [-rank(input), rank(input)).\n   name: A name for the operation (optional).\n\n  Returns:\n    A potentially ragged tensor. Contains the same data as input,\n    but has one or more dimensions of size 1 removed.\n  '
    with ops.name_scope(name, 'RaggedSqueeze', [input]):
        input = ragged_tensor.convert_to_tensor_or_ragged_tensor(input)
        if isinstance(input, tensor.Tensor):
            return array_ops.squeeze(input, axis, name)
        if axis is None:
            raise ValueError('Ragged.squeeze must have an axis argument.')
        if isinstance(axis, int):
            axis = [axis]
        elif not isinstance(axis, (list, tuple)) or not all((isinstance(d, int) for d in axis)):
            raise TypeError('Axis must be a list or tuple of integers.')
        dense_dims = []
        ragged_dims = []
        axis = [array_ops.get_positive_axis(d, input.shape.ndims, 'axis[%d]' % i, 'rank(input)') for (i, d) in enumerate(axis)]
        for dim in axis:
            if dim > input.ragged_rank:
                dense_dims.append(dim - input.ragged_rank)
            else:
                ragged_dims.append(dim)
        assertion_list = []
        scalar_tensor_one = constant_op.constant(1, dtype=input.row_splits.dtype)
        for (i, r) in enumerate(input.nested_row_lengths()):
            if i + 1 in ragged_dims:
                assertion_list.append(control_flow_assert.Assert(math_ops.reduce_all(math_ops.equal(r, scalar_tensor_one)), ['the given axis (axis = %d) is not squeezable!' % (i + 1)]))
        if 0 in ragged_dims:
            scalar_tensor_two = constant_op.constant(2, dtype=dtypes.int32)
            assertion_list.append(control_flow_assert.Assert(math_ops.equal(array_ops.size(input.row_splits), scalar_tensor_two), ['the given axis (axis = 0) is not squeezable!']))
        squeezed_rt = None
        squeezed_rt = control_flow_ops.with_dependencies(assertion_list, input.flat_values)
        if dense_dims:
            squeezed_rt = array_ops.squeeze(squeezed_rt, dense_dims)
        remaining_row_splits = []
        remaining_row_splits = list()
        for (i, row_split) in enumerate(input.nested_row_splits):
            if i + 1 not in ragged_dims:
                remaining_row_splits.append(row_split)
        if remaining_row_splits and 0 in ragged_dims:
            remaining_row_splits.pop(0)
        squeezed_rt = RaggedTensor.from_nested_row_splits(squeezed_rt, remaining_row_splits)
        if set(range(0, input.ragged_rank + 1)).issubset(set(ragged_dims)):
            squeezed_rt = array_ops.squeeze(squeezed_rt, [0], name)
        return squeezed_rt

@dispatch.dispatch_for_api(array_ops.squeeze)
def _ragged_squeeze_v1(input: ragged_tensor.Ragged, axis=None, name=None, squeeze_dims=None):
    if False:
        print('Hello World!')
    axis = deprecation.deprecated_argument_lookup('axis', axis, 'squeeze_dims', squeeze_dims)
    return squeeze(input, axis, name)