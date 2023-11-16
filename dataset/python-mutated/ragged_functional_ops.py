"""Support for ragged tensors."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops.ragged import ragged_config
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

@tf_export('ragged.map_flat_values')
@dispatch.add_dispatch_support
def map_flat_values(op, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    "Applies `op` to the `flat_values` of one or more RaggedTensors.\n\n  Replaces any `RaggedTensor` in `args` or `kwargs` with its `flat_values`\n  tensor (which collapses all ragged dimensions), and then calls `op`.  Returns\n  a `RaggedTensor` that is constructed from the input `RaggedTensor`s'\n  `nested_row_splits` and the value returned by the `op`.\n\n  If the input arguments contain multiple `RaggedTensor`s, then they must have\n  identical `nested_row_splits`.\n\n  This operation is generally used to apply elementwise operations to each value\n  in a `RaggedTensor`.\n\n  Warning: `tf.ragged.map_flat_values` does *not* apply `op` to each row of a\n  ragged tensor.  This difference is important for non-elementwise operations,\n  such as `tf.reduce_sum`.  If you wish to apply a non-elementwise operation to\n  each row of a ragged tensor, use `tf.map_fn` instead.  (You may need to\n  specify an `output_signature` when using `tf.map_fn` with ragged tensors.)\n\n  Examples:\n\n  >>> rt = tf.ragged.constant([[1, 2, 3], [], [4, 5], [6]])\n  >>> tf.ragged.map_flat_values(tf.ones_like, rt)\n  <tf.RaggedTensor [[1, 1, 1], [], [1, 1], [1]]>\n  >>> tf.ragged.map_flat_values(tf.multiply, rt, rt)\n  <tf.RaggedTensor [[1, 4, 9], [], [16, 25], [36]]>\n  >>> tf.ragged.map_flat_values(tf.add, rt, 5)\n  <tf.RaggedTensor [[6, 7, 8], [], [9, 10], [11]]>\n\n  Example with a non-elementwise operation (note that `map_flat_values` and\n  `map_fn` return different results):\n\n  >>> rt = tf.ragged.constant([[1.0, 3.0], [], [3.0, 6.0, 3.0]])\n  >>> def normalized(x):\n  ...   return x / tf.reduce_sum(x)\n  >>> tf.ragged.map_flat_values(normalized, rt)\n  <tf.RaggedTensor [[0.0625, 0.1875], [], [0.1875, 0.375, 0.1875]]>\n  >>> tf.map_fn(normalized, rt)\n  <tf.RaggedTensor [[0.25, 0.75], [], [0.25, 0.5, 0.25]]>\n\n  Args:\n    op: The operation that should be applied to the RaggedTensor `flat_values`.\n      `op` is typically an element-wise operation (such as math_ops.add), but\n      any operation that preserves the size of the outermost dimension can be\n      used.  I.e., `shape[0]` of the value returned by `op` must match\n      `shape[0]` of the `RaggedTensor`s' `flat_values` tensors.\n    *args: Arguments for `op`.\n    **kwargs: Keyword arguments for `op`.\n\n  Returns:\n    A `RaggedTensor` whose `ragged_rank` matches the `ragged_rank` of all\n    input `RaggedTensor`s.\n  Raises:\n    ValueError: If args contains no `RaggedTensors`, or if the `nested_splits`\n      of the input `RaggedTensor`s are not identical.\n  "
    partition_lists = []
    flat_values_nrows = []
    inner_args = _replace_ragged_with_flat_values(args, partition_lists, flat_values_nrows)
    inner_kwargs = _replace_ragged_with_flat_values(kwargs, partition_lists, flat_values_nrows)
    if not partition_lists:
        return op(*args, **kwargs)
    if flat_values_nrows:
        flat_values_nrows = set(flat_values_nrows)
        if len(flat_values_nrows) != 1:
            raise ValueError("Input RaggedTensors' flat_values must all have the same outer-dimension size.  Got sizes: %s" % flat_values_nrows)
        flat_values_nrows = flat_values_nrows.pop()
    else:
        flat_values_nrows = None
    partition_dtypes = set((p[0].dtype for p in partition_lists))
    if len(partition_dtypes) > 1:
        if not ragged_config.auto_cast_partition_dtype():
            raise ValueError('Input RaggedTensors have mismatched row partition dtypes; use RaggedTensor.with_row_splits_dtype() to convert them to compatible dtypes.')
        partition_lists = [[p.with_dtype(dtypes.int64) for p in partition_list] for partition_list in partition_lists]
    op_output = op(*inner_args, **inner_kwargs)
    if flat_values_nrows is not None:
        if not op_output.shape[:1].is_compatible_with([flat_values_nrows]):
            raise ValueError('tf.ragged.map_flat_values requires that the output of `op` have the same outer-dimension size as flat_values of any ragged inputs. (output shape: %s; expected outer dimension size: %s)' % (op_output.shape, flat_values_nrows))
    return ragged_tensor.RaggedTensor._from_nested_row_partitions(op_output, _merge_partition_lists(partition_lists), validate=False)

def _replace_ragged_with_flat_values(value, partition_lists, flat_values_nrows):
    if False:
        return 10
    "Replace RaggedTensors with their flat_values, and record their partitions.\n\n  Returns a copy of `value`, with any nested `RaggedTensor`s replaced by their\n  `flat_values` tensor.  Looks inside lists, tuples, and dicts.\n\n  Appends each `RaggedTensor`'s `RowPartition`s to `partition_lists`.\n\n  Args:\n    value: The value that should be transformed by replacing `RaggedTensors`.\n    partition_lists: An output parameter used to record the row partitions\n      for any `RaggedTensors` that were replaced.\n    flat_values_nrows: An output parameter used to record the outer dimension\n      size for each replacement `flat_values` (when known).  Contains a list of\n      int.\n\n  Returns:\n    A copy of `value` with nested `RaggedTensors` replaced by their `values`.\n  "
    if ragged_tensor.is_ragged(value):
        value = ragged_tensor.convert_to_tensor_or_ragged_tensor(value)
        partition_lists.append(value._nested_row_partitions)
        nrows = tensor_shape.dimension_at_index(value.flat_values.shape, 0).value
        if nrows is not None:
            flat_values_nrows.append(nrows)
        return value.flat_values

    def recurse(v):
        if False:
            for i in range(10):
                print('nop')
        return _replace_ragged_with_flat_values(v, partition_lists, flat_values_nrows)
    if isinstance(value, list):
        return [recurse(v) for v in value]
    elif isinstance(value, tuple):
        return tuple((recurse(v) for v in value))
    elif isinstance(value, dict):
        return dict(((k, recurse(v)) for (k, v) in value.items()))
    else:
        return value

def _merge_partition_lists(partition_lists):
    if False:
        while True:
            i = 10
    'Merges the given list of lists of RowPartitions.\n\n  Args:\n    partition_lists: A list of lists of RowPartition.\n\n  Returns:\n    A list of RowPartitions, where `result[i]` is formed by merging\n    `partition_lists[j][i]` for all `j`, using\n    `RowPartition._merge_precomputed_encodings`.\n  '
    dst = list(partition_lists[0])
    for src in partition_lists[1:]:
        if len(src) != len(dst):
            raise ValueError('All ragged inputs must have the same ragged_rank.')
        for i in range(len(dst)):
            dst[i] = dst[i]._merge_precomputed_encodings(src[i])
    return dst