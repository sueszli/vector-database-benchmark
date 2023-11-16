"""Private convenience functions for RaggedTensors.

None of these methods are exposed in the main "ragged" package.
"""
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import math_ops

def assert_splits_match(nested_splits_lists):
    if False:
        return 10
    'Checks that the given splits lists are identical.\n\n  Performs static tests to ensure that the given splits lists are identical,\n  and returns a list of control dependency op tensors that check that they are\n  fully identical.\n\n  Args:\n    nested_splits_lists: A list of nested_splits_lists, where each split_list is\n      a list of `splits` tensors from a `RaggedTensor`, ordered from outermost\n      ragged dimension to innermost ragged dimension.\n\n  Returns:\n    A list of control dependency op tensors.\n  Raises:\n    ValueError: If the splits are not identical.\n  '
    error_msg = 'Inputs must have identical ragged splits'
    for splits_list in nested_splits_lists:
        if len(splits_list) != len(nested_splits_lists[0]):
            raise ValueError(error_msg)
    return [check_ops.assert_equal(s1, s2, message=error_msg) for splits_list in nested_splits_lists[1:] for (s1, s2) in zip(nested_splits_lists[0], splits_list)]
get_positive_axis = array_ops.get_positive_axis
convert_to_int_tensor = array_ops.convert_to_int_tensor
repeat = array_ops.repeat_with_axis

def lengths_to_splits(lengths):
    if False:
        return 10
    'Returns splits corresponding to the given lengths.'
    return array_ops.concat([[0], math_ops.cumsum(lengths)], axis=-1)

def repeat_ranges(params, splits, repeats):
    if False:
        i = 10
        return i + 15
    "Repeats each range of `params` (as specified by `splits`) `repeats` times.\n\n  Let the `i`th range of `params` be defined as\n  `params[splits[i]:splits[i + 1]]`.  Then this function returns a tensor\n  containing range 0 repeated `repeats[0]` times, followed by range 1 repeated\n  `repeats[1]`, ..., followed by the last range repeated `repeats[-1]` times.\n\n  Args:\n    params: The `Tensor` whose values should be repeated.\n    splits: A splits tensor indicating the ranges of `params` that should be\n      repeated. Elements should be non-negative integers.\n    repeats: The number of times each range should be repeated. Supports\n      broadcasting from a scalar value. Elements should be non-negative\n      integers.\n\n  Returns:\n    A `Tensor` with the same rank and type as `params`.\n\n  #### Example:\n\n  >>> print(repeat_ranges(\n  ...     params=tf.constant(['a', 'b', 'c']),\n  ...     splits=tf.constant([0, 2, 3]),\n  ...     repeats=tf.constant(3)))\n  tf.Tensor([b'a' b'b' b'a' b'b' b'a' b'b' b'c' b'c' b'c'],\n      shape=(9,), dtype=string)\n  "
    splits_checks = [check_ops.assert_non_negative(splits, message="Input argument 'splits' must be non-negative"), check_ops.assert_integer(splits, message=f"Input argument 'splits' must be integer, but got {splits.dtype} instead")]
    repeats_checks = [check_ops.assert_non_negative(repeats, message="Input argument 'repeats' must be non-negative"), check_ops.assert_integer(repeats, message=f"Input argument 'repeats' must be integer, but got {repeats.dtype} instead")]
    splits = control_flow_ops.with_dependencies(splits_checks, splits)
    repeats = control_flow_ops.with_dependencies(repeats_checks, repeats)
    if repeats.shape.ndims != 0:
        repeated_starts = repeat(splits[:-1], repeats, axis=0)
        repeated_limits = repeat(splits[1:], repeats, axis=0)
    else:
        repeated_splits = repeat(splits, repeats, axis=0)
        n_splits = array_ops.shape(repeated_splits, out_type=repeats.dtype)[0]
        repeated_starts = repeated_splits[:n_splits - repeats]
        repeated_limits = repeated_splits[repeats:]
    one = array_ops.ones((), repeated_starts.dtype)
    offsets = gen_ragged_math_ops.ragged_range(repeated_starts, repeated_limits, one)
    return array_ops.gather(params, offsets.rt_dense_values)