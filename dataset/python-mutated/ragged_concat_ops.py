"""Concat and stack operations for RaggedTensors."""
import typing
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_gather_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

@dispatch.dispatch_for_api(array_ops.concat)
def concat(values: typing.List[ragged_tensor.RaggedOrDense], axis, name=None):
    if False:
        print('Hello World!')
    'Concatenates potentially ragged tensors along one dimension.\n\n  Given a list of tensors with the same rank `K` (`K >= axis`), returns a\n  rank-`K` `RaggedTensor` `result` such that `result[i0...iaxis]` is the\n  concatenation of `[rt[i0...iaxis] for rt in values]`.\n\n  Args:\n    values: A list of potentially ragged tensors.  May not be empty. All\n      `values` must have the same rank and the same dtype; but unlike\n      `tf.concat`, they can have arbitrary shapes.\n    axis: A python integer, indicating the dimension along which to concatenate.\n      (Note: Unlike `tf.concat`, the `axis` parameter must be statically known.)\n        Negative values are supported only if the rank of at least one\n        `values` value is statically known.\n    name: A name prefix for the returned tensor (optional).\n\n  Returns:\n    A `RaggedTensor` with rank `K`.\n    `result.ragged_rank=max(axis, max(rt.ragged_rank for rt in values]))`.\n\n  Raises:\n    ValueError: If `values` is empty, if `axis` is out of bounds or if\n      the input tensors have different ranks.\n\n  #### Example:\n\n  >>> t1 = tf.ragged.constant([[1, 2], [3, 4, 5]])\n  >>> t2 = tf.ragged.constant([[6], [7, 8, 9]])\n  >>> tf.concat([t1, t2], axis=0)\n  <tf.RaggedTensor [[1, 2], [3, 4, 5], [6], [7, 8, 9]]>\n  >>> tf.concat([t1, t2], axis=1)\n  <tf.RaggedTensor [[1, 2, 6], [3, 4, 5, 7, 8, 9]]>\n  '
    if not isinstance(values, (list, tuple)):
        values = [values]
    with ops.name_scope(name, 'RaggedConcat', values):
        return _ragged_stack_concat_helper(values, axis, stack_values=False)

@tf_export('ragged.stack')
@dispatch.add_dispatch_support
@dispatch.dispatch_for_api(array_ops_stack.stack)
def stack(values: typing.List[ragged_tensor.RaggedOrDense], axis=0, name=None):
    if False:
        return 10
    'Stacks a list of rank-`R` tensors into one rank-`(R+1)` `RaggedTensor`.\n\n  Given a list of tensors or ragged tensors with the same rank `R`\n  (`R >= axis`), returns a rank-`R+1` `RaggedTensor` `result` such that\n  `result[i0...iaxis]` is `[value[i0...iaxis] for value in values]`.\n\n  #### Examples:\n\n  >>> # Stacking two ragged tensors.\n  >>> t1 = tf.ragged.constant([[1, 2], [3, 4, 5]])\n  >>> t2 = tf.ragged.constant([[6], [7, 8, 9]])\n  >>> tf.ragged.stack([t1, t2], axis=0)\n  <tf.RaggedTensor [[[1, 2], [3, 4, 5]], [[6], [7, 8, 9]]]>\n  >>> tf.ragged.stack([t1, t2], axis=1)\n  <tf.RaggedTensor [[[1, 2], [6]], [[3, 4, 5], [7, 8, 9]]]>\n\n  >>> # Stacking two dense tensors with different sizes.\n  >>> t3 = tf.constant([[1, 2, 3], [4, 5, 6]])\n  >>> t4 = tf.constant([[5], [6], [7]])\n  >>> tf.ragged.stack([t3, t4], axis=0)\n  <tf.RaggedTensor [[[1, 2, 3], [4, 5, 6]], [[5], [6], [7]]]>\n\n  Args:\n    values: A list of `tf.Tensor` or `tf.RaggedTensor`.  May not be empty. All\n      `values` must have the same rank and the same dtype; but unlike\n      `tf.stack`, they can have arbitrary dimension sizes.\n    axis: A python integer, indicating the dimension along which to stack.\n      (Note: Unlike `tf.stack`, the `axis` parameter must be statically known.)\n      Negative values are supported only if the rank of at least one\n      `values` value is statically known.\n    name: A name prefix for the returned tensor (optional).\n\n  Returns:\n    A `RaggedTensor` with rank `R+1` (if `R>0`).\n    If `R==0`, then the result will be returned as a 1D `Tensor`, since\n    `RaggedTensor` can only be used when `rank>1`.\n    `result.ragged_rank=1+max(axis, max(rt.ragged_rank for rt in values]))`.\n\n  Raises:\n    ValueError: If `values` is empty, if `axis` is out of bounds or if\n      the input tensors have different ranks.\n  '
    if not isinstance(values, (list, tuple)):
        values = [values]
    with ops.name_scope(name, 'RaggedConcat', values):
        return _ragged_stack_concat_helper(values, axis, stack_values=True)

def _ragged_stack_concat_helper(rt_inputs, axis, stack_values):
    if False:
        print('Hello World!')
    'Helper function to concatenate or stack ragged tensors.\n\n  Args:\n    rt_inputs: A list of RaggedTensors or Tensors to combine.\n    axis: The axis along which to concatenate or stack.\n    stack_values: A boolean -- if true, then stack values; otherwise,\n      concatenate them.\n\n  Returns:\n    A RaggedTensor.\n  Raises:\n    ValueError: If rt_inputs is empty, or if axis is out of range.\n  '
    if not rt_inputs:
        raise ValueError('rt_inputs may not be empty.')
    rt_inputs = [ragged_tensor.convert_to_tensor_or_ragged_tensor(rt_input, name='rt_input') for rt_input in rt_inputs]
    (row_splits_dtype, rt_inputs) = ragged_tensor.match_row_splits_dtypes(*rt_inputs, return_dtype=True)
    rt_inputs = list(rt_inputs)
    if len(rt_inputs) == 1 and (not stack_values):
        return rt_inputs[0]
    ndims = None
    for rt in rt_inputs:
        if ndims is None:
            ndims = rt.shape.ndims
        else:
            rt.shape.assert_has_rank(ndims)
    out_ndims = ndims if ndims is None or not stack_values else ndims + 1
    axis = array_ops.get_positive_axis(axis, out_ndims)
    if stack_values and ndims == 1 and (axis == 0):
        return ragged_tensor.RaggedTensor.from_row_lengths(values=array_ops.concat(rt_inputs, axis=0), row_lengths=array_ops.concat([array_ops.shape(r) for r in rt_inputs], axis=0))
    if all((not ragged_tensor.is_ragged(rt) for rt in rt_inputs)):
        if ndims is not None and (axis == out_ndims - 1 or axis == ndims - 1):
            if stack_values:
                return array_ops_stack.stack(rt_inputs, axis)
            else:
                return array_ops.concat(rt_inputs, axis)
    for i in range(len(rt_inputs)):
        if not ragged_tensor.is_ragged(rt_inputs[i]):
            rt_inputs[i] = ragged_tensor.RaggedTensor.from_tensor(rt_inputs[i], ragged_rank=1, row_splits_dtype=row_splits_dtype)
    ragged_rank = max(max((rt.ragged_rank for rt in rt_inputs)), 1)
    rt_inputs = [_increase_ragged_rank_to(rt, ragged_rank, row_splits_dtype) for rt in rt_inputs]
    if axis == 0:
        return _ragged_stack_concat_axis_0(rt_inputs, stack_values)
    elif axis == 1:
        return _ragged_stack_concat_axis_1(rt_inputs, stack_values)
    else:
        values = [rt.values for rt in rt_inputs]
        splits = [[rt_input.row_splits] for rt_input in rt_inputs]
        with ops.control_dependencies(ragged_util.assert_splits_match(splits)):
            return ragged_tensor.RaggedTensor.from_row_splits(_ragged_stack_concat_helper(values, axis - 1, stack_values), splits[0][0], validate=False)

def _ragged_stack_concat_axis_0(rt_inputs, stack_values):
    if False:
        i = 10
        return i + 15
    'Helper function to concatenate or stack ragged tensors along axis 0.\n\n  Args:\n    rt_inputs: A list of RaggedTensors, all with the same rank and ragged_rank.\n    stack_values: Boolean.  If true, then stack values; otherwise, concatenate\n      them.\n\n  Returns:\n    A RaggedTensor.\n  '
    flat_values = [rt.flat_values for rt in rt_inputs]
    concatenated_flat_values = array_ops.concat(flat_values, axis=0)
    nested_splits = [rt.nested_row_splits for rt in rt_inputs]
    ragged_rank = rt_inputs[0].ragged_rank
    concatenated_nested_splits = [_concat_ragged_splits([ns[dim] for ns in nested_splits]) for dim in range(ragged_rank)]
    if stack_values:
        stack_lengths = array_ops_stack.stack([rt.nrows() for rt in rt_inputs])
        stack_splits = ragged_util.lengths_to_splits(stack_lengths)
        concatenated_nested_splits.insert(0, stack_splits)
    return ragged_tensor.RaggedTensor.from_nested_row_splits(concatenated_flat_values, concatenated_nested_splits, validate=False)

def _ragged_stack_concat_axis_1(rt_inputs, stack_values):
    if False:
        print('Hello World!')
    'Helper function to concatenate or stack ragged tensors along axis 1.\n\n  Args:\n    rt_inputs: A list of RaggedTensors, all with the same rank and ragged_rank.\n    stack_values: Boolean.  If true, then stack values; otherwise, concatenate\n      them.\n\n  Returns:\n    A RaggedTensor.\n  '
    num_inputs = len(rt_inputs)
    nrows_checks = []
    rt_nrows = rt_inputs[0].nrows()
    for (index, rt) in enumerate(rt_inputs[1:]):
        nrows_checks.append(check_ops.assert_equal(rt_nrows, rt.nrows(), message=f'Input tensors at index 0 (=x) and {index + 1} (=y) have incompatible shapes.'))
    with ops.control_dependencies(nrows_checks):
        concatenated_rt = _ragged_stack_concat_axis_0(rt_inputs, stack_values=False)
        row_indices = math_ops.range(rt_nrows * num_inputs)
        row_index_matrix = array_ops.reshape(row_indices, [num_inputs, -1])
        transposed_row_index_matrix = array_ops.transpose(row_index_matrix)
        row_permutation = array_ops.reshape(transposed_row_index_matrix, [-1])
        permuted_rt = ragged_gather_ops.gather(concatenated_rt, row_permutation)
        if stack_values:
            stack_splits = math_ops.range(0, rt_nrows * num_inputs + 1, num_inputs)
            _copy_row_shape(rt_inputs, stack_splits)
            return ragged_tensor.RaggedTensor.from_row_splits(permuted_rt, stack_splits, validate=False)
        else:
            concat_splits = permuted_rt.row_splits[::num_inputs]
            _copy_row_shape(rt_inputs, concat_splits)
            return ragged_tensor.RaggedTensor.from_row_splits(permuted_rt.values, concat_splits, validate=False)

def _copy_row_shape(rt_inputs, splits):
    if False:
        while True:
            i = 10
    'Sets splits.shape to [rt[shape[0]+1] for each rt in rt_inputs.'
    for rt in rt_inputs:
        if rt.shape[0] is not None:
            splits.set_shape(tensor_shape.TensorShape(rt.shape[0] + 1))

def _increase_ragged_rank_to(rt_input, ragged_rank, row_splits_dtype):
    if False:
        return 10
    'Adds ragged dimensions to `rt_input` so it has the desired ragged rank.'
    if ragged_rank > 0:
        if not ragged_tensor.is_ragged(rt_input):
            rt_input = ragged_tensor.RaggedTensor.from_tensor(rt_input, row_splits_dtype=row_splits_dtype)
        if rt_input.ragged_rank < ragged_rank:
            rt_input = rt_input.with_values(_increase_ragged_rank_to(rt_input.values, ragged_rank - 1, row_splits_dtype))
    return rt_input

def _concat_ragged_splits(splits_list):
    if False:
        i = 10
        return i + 15
    'Concatenates a list of RaggedTensor splits to form a single splits.'
    pieces = [splits_list[0]]
    splits_offset = splits_list[0][-1]
    for splits in splits_list[1:]:
        pieces.append(splits[1:] + splits_offset)
        splits_offset += splits[-1]
    return array_ops.concat(pieces, axis=0)