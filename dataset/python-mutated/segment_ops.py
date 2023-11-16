"""Element wise ops acting on segments of arrays."""
import tensorflow.compat.v2 as tf
from tf_quant_finance.math import diff_ops

def segment_diff(x, segment_ids, order=1, exclusive=False, dtype=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    "Computes difference of successive elements in a segment.\n\n  For a complete description of segment_* ops see documentation of\n  `tf.segment_max`. This op extends the `diff` functionality to segmented\n  inputs.\n\n  The behaviour of this op is the same as that of the op `diff` within each\n  segment. The result is effectively a concatenation of the results of `diff`\n  applied to each segment.\n\n  #### Example\n\n  ```python\n    x = tf.constant([2, 5, 1, 7, 9] + [32, 10, 12, 3] + [4, 8, 5])\n    segments = tf.constant([0, 0, 0, 0, 0] + [1, 1, 1, 1] + [2, 2, 2])\n    # First order diff. Expected result: [3, -4, 6, 2, -22, 2, -9, 4, -3]\n    dx1 = segment_diff(\n        x, segment_ids=segments, order=1, exclusive=True)\n    # Non-exclusive, second order diff.\n    # Expected result: [2, 5, -1, 2, 8, 32, 10, -20, -7, 4, 8, 1]\n    dx2 = segment_diff(\n        x, segment_ids=segments, order=2, exclusive=False)\n  ```\n\n  Args:\n    x: A rank 1 `Tensor` of any dtype for which arithmetic operations are\n      permitted.\n    segment_ids: A `Tensor`. Must be one of the following types: int32, int64. A\n      1-D tensor whose size is equal to the size of `x`. Values should be sorted\n      and can be repeated.\n    order: Positive Python int. The order of the difference to compute. `order =\n      1` corresponds to the difference between successive elements.\n      Default value: 1\n    exclusive: Python bool. See description above.\n      Default value: False\n    dtype: Optional `tf.Dtype`. If supplied, the dtype for `x` to use when\n      converting to `Tensor`.\n      Default value: None which maps to the default dtype inferred by TF.\n    name: Python `str` name prefixed to Ops created by this class.\n      Default value: None which is mapped to the default name 'segment_diff'.\n\n  Returns:\n    diffs: A `Tensor` of the same dtype as `x`. Assuming that each segment is\n      of length greater than or equal to order, if `exclusive` is True,\n      then the size is `n-order*k` where `n` is the size of x,\n      `k` is the number of different segment ids supplied if `segment_ids` is\n      not None or 1 if `segment_ids` is None. If any of the segments is of\n      length less than the order, then the size is:\n      `n-sum(min(order, length(segment_j)), j)` where the sum is over segments.\n      If `exclusive` is False, then the size is `n`.\n  "
    with tf.compat.v1.name_scope(name, default_name='segment_diff', values=[x]):
        x = tf.convert_to_tensor(x, dtype=dtype)
        raw_diffs = diff_ops.diff(x, order=order, exclusive=exclusive)
        if segment_ids is None:
            return raw_diffs
        has_segment_changed = tf.concat([[False], tf.not_equal(segment_ids[1:] - segment_ids[:-1], 0)], axis=0)
        segment_start_index = tf.cast(tf.where(has_segment_changed), dtype=tf.int32)
        segment_end_index = tf.concat([tf.reshape(segment_start_index, [-1])[1:], [tf.size(segment_ids)]], axis=0)
        segment_end_index = tf.reshape(segment_end_index, [-1, 1])
        fix_indices = segment_start_index + tf.range(order, dtype=segment_start_index.dtype)
        in_bounds = tf.where(fix_indices < segment_end_index)
        fix_indices = tf.reshape(tf.gather_nd(fix_indices, in_bounds), [-1, 1])
        needs_fix = tf.scatter_nd(fix_indices, tf.reshape(tf.ones_like(fix_indices, dtype=tf.int32), [-1]), shape=tf.shape(x))
        needs_fix = tf.cast(needs_fix, dtype=tf.bool)
        if not exclusive:
            return tf.where(needs_fix, x, raw_diffs)
        return tf.boolean_mask(raw_diffs, tf.logical_not(needs_fix[order:]))

def segment_cumsum(x, segment_ids, exclusive=False, dtype=None, name=None):
    if False:
        i = 10
        return i + 15
    "Computes cumulative sum of elements in a segment.\n\n  For a complete description of segment_* ops see documentation of\n  `tf.segment_sum`. This op extends the `tf.math.cumsum` functionality to\n  segmented inputs.\n\n  The behaviour of this op is the same as that of the op `tf.math.cumsum` within\n  each segment. The result is effectively a concatenation of the results of\n  `tf.math.cumsum` applied to each segment with the same interpretation for the\n  argument `exclusive`.\n\n  #### Example\n\n  ```python\n    x = tf.constant([2, 5, 1, 7, 9] + [32, 10, 12, 3] + [4, 8, 5])\n    segments = tf.constant([0, 0, 0, 0, 0] + [1, 1, 1, 1] + [2, 2, 2])\n    # Inclusive cumulative sum.\n    # Expected result: [2, 7, 8, 15, 24, 32, 42, 54, 57, 4, 12, 17]\n    cumsum1 = segment_cumsum(\n        x, segment_ids=segments, exclusive=False)\n    # Exclusive cumsum.\n    # Expected result: [0, 2, 7, 8, 15, 0, 32, 42, 54, 0, 4, 12]\n    cumsum2 = segment_cumsum(\n        x, segment_ids=segments, exclusive=True)\n  ```\n\n  Args:\n    x: A rank 1 `Tensor` of any dtype for which arithmetic operations are\n      permitted.\n    segment_ids: A `Tensor`. Must be one of the following types: int32, int64. A\n      1-D tensor whose size is equal to the size of `x`. Values should be sorted\n      and can be repeated. Values must range from `0` to `num segments - 1`.\n    exclusive: Python bool. See description above.\n      Default value: False\n    dtype: Optional `tf.Dtype`. If supplied, the dtype for `x` to use when\n      converting to `Tensor`.\n      Default value: None which maps to the default dtype inferred by TF.\n    name: Python `str` name prefixed to Ops created by this class.\n      Default value: None which is mapped to the default name 'segment_cumsum'.\n\n  Returns:\n    cumsums: A `Tensor` of the same dtype as `x`. Assuming that each segment is\n      of length greater than or equal to order, if `exclusive` is True,\n      then the size is `n-order*k` where `n` is the size of x,\n      `k` is the number of different segment ids supplied if `segment_ids` is\n      not None or 1 if `segment_ids` is None. If any of the segments is of\n      length less than the order, then the size is:\n      `n-sum(min(order, length(segment_j)), j)` where the sum is over segments.\n      If `exclusive` is False, then the size is `n`.\n  "
    with tf.compat.v1.name_scope(name, default_name='segment_cumsum', values=[x]):
        x = tf.convert_to_tensor(x, dtype=dtype)
        raw_cumsum = tf.math.cumsum(x, exclusive=exclusive)
        if segment_ids is None:
            return raw_cumsum

        def scanner(accumulators, args):
            if False:
                i = 10
                return i + 15
            (cumsum, prev_segment, prev_value) = accumulators
            (value, segment) = args
            if exclusive:
                (initial_value, inc_value) = (tf.zeros_like(value), cumsum + prev_value)
            else:
                (initial_value, inc_value) = (value, cumsum + value)
            next_cumsum = tf.where(tf.equal(prev_segment, segment), inc_value, initial_value)
            return (next_cumsum, segment, value)
        return tf.scan(scanner, (x, segment_ids), initializer=(tf.zeros_like(x[0]), tf.zeros_like(segment_ids[0]) - 1, tf.zeros_like(x[0])))[0]
__all__ = ['segment_cumsum', 'segment_diff']