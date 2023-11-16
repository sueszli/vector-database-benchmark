"""Array operations for RaggedTensors."""
from typing import Optional
from typing import Union
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_ragged_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.types import core as core_types
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

@tf_export('ragged.boolean_mask')
@dispatch.add_dispatch_support
def boolean_mask(data, mask, name=None):
    if False:
        while True:
            i = 10
    "Applies a boolean mask to `data` without flattening the mask dimensions.\n\n  Returns a potentially ragged tensor that is formed by retaining the elements\n  in `data` where the corresponding value in `mask` is `True`.\n\n  * `output[a1...aA, i, b1...bB] = data[a1...aA, j, b1...bB]`\n\n     Where `j` is the `i`th `True` entry of `mask[a1...aA]`.\n\n  Note that `output` preserves the mask dimensions `a1...aA`; this differs\n  from `tf.boolean_mask`, which flattens those dimensions.\n\n  Args:\n    data: A potentially ragged tensor.\n    mask: A potentially ragged boolean tensor.  `mask`'s shape must be a prefix\n      of `data`'s shape.  `rank(mask)` must be known statically.\n    name: A name prefix for the returned tensor (optional).\n\n  Returns:\n    A potentially ragged tensor that is formed by retaining the elements in\n    `data` where the corresponding value in `mask` is `True`.\n\n    * `rank(output) = rank(data)`.\n    * `output.ragged_rank = max(data.ragged_rank, rank(mask) - 1)`.\n\n  Raises:\n    ValueError: if `rank(mask)` is not known statically; or if `mask.shape` is\n      not a prefix of `data.shape`.\n\n  #### Examples:\n\n  >>> # Aliases for True & False so data and mask line up.\n  >>> T, F = (True, False)\n\n  >>> tf.ragged.boolean_mask(  # Mask a 2D Tensor.\n  ...     data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],\n  ...     mask=[[T, F, T], [F, F, F], [T, F, F]]).to_list()\n  [[1, 3], [], [7]]\n\n  >>> tf.ragged.boolean_mask(  # Mask a 2D RaggedTensor.\n  ...     tf.ragged.constant([[1, 2, 3], [4], [5, 6]]),\n  ...     tf.ragged.constant([[F, F, T], [F], [T, T]])).to_list()\n  [[3], [], [5, 6]]\n\n  >>> tf.ragged.boolean_mask(  # Mask rows of a 2D RaggedTensor.\n  ...     tf.ragged.constant([[1, 2, 3], [4], [5, 6]]),\n  ...     tf.ragged.constant([True, False, True])).to_list()\n  [[1, 2, 3], [5, 6]]\n  "
    with ops.name_scope(name, 'RaggedMask', [data, mask]):
        data = ragged_tensor.convert_to_tensor_or_ragged_tensor(data, name='data')
        mask = ragged_tensor.convert_to_tensor_or_ragged_tensor(mask, dtypes.bool, name='mask')
        (row_splits_dtype, (data, mask)) = ragged_tensor.match_row_splits_dtypes(data, mask, return_dtype=True)
        if mask.shape.ndims is None:
            raise ValueError('mask.shape.ndims must be known statically.')
        elif mask.shape.ndims == 0:
            raise ValueError('mask cannot be scalar.')
        if ragged_tensor.is_ragged(mask):
            if not ragged_tensor.is_ragged(data):
                data = ragged_tensor.RaggedTensor.from_tensor(data, ragged_rank=mask.ragged_rank, row_splits_dtype=mask.row_splits.dtype)
            splits_list = [mask.nested_row_splits, data.nested_row_splits[:mask.ragged_rank]]
            with ops.control_dependencies(ragged_util.assert_splits_match(splits_list)):
                splits = []
                while ragged_tensor.is_ragged(mask):
                    if mask.shape.ndims > 2:
                        splits.append(mask.row_splits)
                    else:
                        int_mask = ragged_functional_ops.map_flat_values(math_ops.cast, mask, dtype=row_splits_dtype)
                        masked_row_lengths = ragged_math_ops.reduce_sum(int_mask, axis=1)
                        splits.append(ragged_util.lengths_to_splits(masked_row_lengths))
                    mask = mask.values
                    data = data.values
                masked_values = boolean_mask(data, mask)
                masked_values = ragged_tensor.RaggedTensor.from_nested_row_splits(masked_values, splits, validate=False)
                return masked_values
        elif ragged_tensor.is_ragged(data) and mask.shape.ndims == 1:
            lengths = data.row_lengths()
            masked_lengths = array_ops.boolean_mask(lengths, mask)
            masked_splits = ragged_util.lengths_to_splits(masked_lengths)
            segment_ids = segment_id_ops.row_splits_to_segment_ids(data.row_splits)
            segment_mask = array_ops.gather(mask, segment_ids)
            masked_values = boolean_mask(data.values, segment_mask)
            return ragged_tensor.RaggedTensor.from_row_splits(masked_values, masked_splits, validate=False)
        if ragged_tensor.is_ragged(data):
            mask = ragged_tensor.RaggedTensor.from_tensor(mask, ragged_rank=min(data.ragged_rank, mask.shape.ndims - 1), row_splits_dtype=data.row_splits.dtype)
            return boolean_mask(data, mask)
        else:
            masked_values = array_ops.boolean_mask(data, mask)
            if mask.shape.ndims >= 2:
                masked_lengths = math_ops.count_nonzero(mask, axis=-1, dtype=row_splits_dtype)
                flattened_masked_lengths = array_ops.reshape(masked_lengths, [-1])
                masked_values = ragged_tensor.RaggedTensor.from_row_lengths(masked_values, flattened_masked_lengths, validate=False)
                if mask.shape.ndims > 2:
                    mask_shape = array_ops.shape(mask, out_type=row_splits_dtype)
                    split_size = math_ops.cumprod(mask_shape) + 1
                    for dim in range(mask.shape.ndims - 3, -1, -1):
                        elt_size = mask_shape[dim + 1]
                        masked_splits = math_ops.range(split_size[dim]) * elt_size
                        masked_values = ragged_tensor.RaggedTensor.from_row_splits(masked_values, masked_splits, validate=False)
            return masked_values

@dispatch.dispatch_for_api(array_ops.tile)
def tile(input: ragged_tensor.Ragged, multiples, name=None):
    if False:
        while True:
            i = 10
    'Constructs a `RaggedTensor` by tiling a given `RaggedTensor`.\n\n  The values of `input` are replicated `multiples[i]` times along the\n  `i`th dimension (for each dimension `i`).  For every dimension `axis` in\n  `input`, the length of each output element in that dimension is the\n  length of corresponding input element multiplied by `multiples[axis]`.\n\n  Args:\n    input: A `RaggedTensor`.\n    multiples: A 1-D integer `Tensor`.  Length must be the same as the number of\n      dimensions in `input`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `RaggedTensor` with the same type, rank, and ragged_rank as `input`.\n\n  #### Example:\n\n  >>> rt = tf.ragged.constant([[1, 2], [3]])\n  >>> tf.tile(rt, [3, 2]).to_list()\n  [[1, 2, 1, 2], [3, 3], [1, 2, 1, 2], [3, 3], [1, 2, 1, 2], [3, 3]]\n  '
    with ops.name_scope(name, 'RaggedTile', [input, multiples]):
        input = ragged_tensor.convert_to_tensor_or_ragged_tensor(input, name='input')
        if not ragged_tensor.is_ragged(input):
            return array_ops.tile(input, multiples, name)
        multiples = ragged_util.convert_to_int_tensor(multiples, name='multiples', dtype=input.row_splits.dtype)
        multiples.shape.assert_has_rank(1)
        const_multiples = tensor_util.constant_value(multiples)
        return ragged_tensor.RaggedTensor.from_nested_row_splits(_tile_ragged_values(input, multiples, const_multiples), _tile_ragged_splits(input, multiples, const_multiples), validate=False)

def _tile_ragged_values(rt_input, multiples, const_multiples=None):
    if False:
        print('Hello World!')
    'Builds flat_values tensor for a tiled `RaggedTensor`.\n\n  Returns a tensor that repeats the values in\n  `rt_input.flat_values` in the\n  appropriate pattern to construct a `RaggedTensor` that tiles `rt_input` as\n  specified by `multiples`.\n\n  Args:\n    rt_input: The `RaggedTensor` whose values should be repeated.\n    multiples: A 1-D integer `tensor`, indicating how many times each dimension\n      should be repeated.\n    const_multiples: Optional constant value for multiples.  Used to skip tiling\n      dimensions where `multiples=1`.\n\n  Returns:\n    A `Tensor` with the same type and rank as `rt_input.flat_values`.\n\n  #### Example:\n\n  >>> rt = tf.ragged.constant([[1, 2], [3]])\n  >>> _tile_ragged_values(rt, tf.constant([3, 2])).numpy()\n  array([1, 2, 1, 2, 3, 3, 1, 2, 1, 2, 3, 3, 1, 2, 1, 2, 3, 3], dtype=int32)\n  '
    ragged_rank = rt_input.ragged_rank
    nested_splits = rt_input.nested_row_splits
    inner_value_ids = math_ops.range(nested_splits[-1][-1])
    prev_splits = None
    for axis in range(ragged_rank, 0, -1):
        splits = nested_splits[axis - 1]
        if prev_splits is not None:
            splits = array_ops.gather(prev_splits * multiples[axis + 1], splits)
        if const_multiples is None or const_multiples[axis] != 1:
            inner_value_ids = ragged_util.repeat_ranges(inner_value_ids, splits, multiples[axis])
        prev_splits = splits
    ragged_tiled_values = array_ops.gather(rt_input.flat_values, inner_value_ids)
    inner_repeats = array_ops.concat([multiples[:1], multiples[ragged_rank + 1:]], axis=0)
    return array_ops.tile(ragged_tiled_values, inner_repeats)

def _tile_ragged_splits(rt_input, multiples, const_multiples=None):
    if False:
        print('Hello World!')
    'Builds nested_split tensors for a tiled `RaggedTensor`.\n\n  Returns a list of split tensors that can be used to construct the\n  `RaggedTensor` that tiles `rt_input` as specified by `multiples`.\n\n  Args:\n    rt_input: The `RaggedTensor` that is being tiled.\n    multiples: A 1-D integer `tensor`, indicating how many times each dimension\n      should be repeated.\n    const_multiples: Optional constant value for multiples.  Used to skip tiling\n      dimensions where `multiples=1`.\n\n  Returns:\n    A list of 1-D integer `Tensor`s (one for each ragged dimension in\n    `rt_input`).\n\n  #### Example:\n\n  >>> rt = tf.ragged.constant([[1, 2], [3]])\n  >>> _tile_ragged_splits(rt, [3, 2])\n  [<tf.Tensor: shape=(7,), dtype=int64,\n  numpy=array([ 0,  4,  6, 10, 12, 16, 18])>]\n  '
    ragged_rank = rt_input.ragged_rank
    nested_splits = rt_input.nested_row_splits
    projected_splits = [{i: nested_splits[i]} for i in range(ragged_rank)]
    for src_axis in range(ragged_rank):
        for dst_axis in range(src_axis + 1, ragged_rank - 1):
            projected_splits[src_axis][dst_axis] = array_ops.gather(nested_splits[dst_axis], projected_splits[src_axis][dst_axis - 1])
    result_splits = []
    for axis in range(ragged_rank):
        input_lengths = nested_splits[axis][1:] - nested_splits[axis][:-1]
        output_lengths = input_lengths * multiples[axis + 1]
        repeats = 1
        for d in range(axis - 1, -1, -1):
            if const_multiples is None or const_multiples[d + 1] != 1:
                splits = projected_splits[d][axis - 1] * repeats
                output_lengths = ragged_util.repeat_ranges(output_lengths, splits, multiples[d + 1])
            repeats *= multiples[d + 1]
        output_lengths = array_ops.tile(output_lengths, multiples[:1])
        result_splits.append(ragged_util.lengths_to_splits(output_lengths))
    return result_splits

@dispatch.dispatch_for_api(array_ops.expand_dims_v2)
def expand_dims(input: ragged_tensor.Ragged, axis, name=None):
    if False:
        i = 10
        return i + 15
    "Inserts a dimension with shape 1 into a potentially ragged tensor's shape.\n\n  Given a potentially ragged tenor `input`, this operation inserts a\n  dimension with size 1 at the dimension `axis` of `input`'s shape.\n\n  The following table gives some examples showing how `ragged.expand_dims`\n  impacts the shapes of different input tensors.  Ragged dimensions are\n  indicated by enclosing them in parentheses.\n\n  input.shape             | axis | result.shape\n  ----------------------- | ---- | -----------------------------\n  `[D1, D2]`              |  `0` | `[1, D1, D2]`\n  `[D1, D2]`              |  `1` | `[D1, 1, D2]`\n  `[D1, D2]`              |  `2` | `[D1, D2, 1]`\n  `[D1, (D2), (D3), D4]`  |  `0` | `[1, D1, (D2), (D3), D4]`\n  `[D1, (D2), (D3), D4]`  |  `1` | `[D1, 1, (D2), (D3), D4]`\n  `[D1, (D2), (D3), D4]`  |  `2` | `[D1, (D2), 1, (D3), D4]`\n  `[D1, (D2), (D3), D4]`  |  `3` | `[D1, (D2), (D3), 1, D4]`\n  `[D1, (D2), (D3), D4]`  |  `4` | `[D1, (D2), (D3), D4, 1]`\n\n  Args:\n    input: The potentially tensor that should be expanded with a new dimension.\n    axis: An integer constant indicating where the new dimension should be\n      inserted.\n    name: A name for the operation (optional).\n\n  Returns:\n    A tensor with the same values as `input`, with an added dimension of\n    size 1 at `axis`.\n\n  #### Examples:\n\n  >>> rt = tf.ragged.constant([[1, 2], [3]])\n  >>> print(rt.shape)\n  (2, None)\n\n  >>> expanded = tf.expand_dims(rt, axis=0)\n  >>> print(expanded.shape, expanded)\n  (1, 2, None) <tf.RaggedTensor [[[1, 2], [3]]]>\n\n  >>> expanded = tf.expand_dims(rt, axis=1)\n  >>> print(expanded.shape, expanded)\n  (2, 1, None) <tf.RaggedTensor [[[1, 2]], [[3]]]>\n\n  >>> expanded = tf.expand_dims(rt, axis=2)\n  >>> print(expanded.shape, expanded)\n  (2, None, 1) <tf.RaggedTensor [[[1], [2]], [[3]]]>\n  "
    with ops.name_scope(name, 'RaggedExpandDims', [input]):
        input = ragged_tensor.convert_to_tensor_or_ragged_tensor(input, name='input')
        if not ragged_tensor.is_ragged(input):
            return array_ops.expand_dims(input, axis)
        ndims = None if input.shape.ndims is None else input.shape.ndims + 1
        axis = array_ops.get_positive_axis(axis, ndims, ndims_name='rank(input)')
        if axis == 0:
            return ragged_tensor.RaggedTensor.from_uniform_row_length(input, uniform_row_length=input.nrows(), nrows=1, validate=False)
        elif axis == 1:
            return ragged_tensor.RaggedTensor.from_uniform_row_length(input, uniform_row_length=1, nrows=input.nrows(), validate=False)
        elif ragged_tensor.is_ragged(input.values):
            return input.with_values(expand_dims(input.values, axis - 1))
        else:
            return input.with_values(array_ops.expand_dims(input.values, axis - 1))

@dispatch.dispatch_for_api(array_ops.expand_dims)
def _ragged_expand_dims_v1(input: ragged_tensor.Ragged, axis=None, name=None, dim=None):
    if False:
        print('Hello World!')
    if dim is not None:
        axis = dim
    return expand_dims(input=input, axis=axis, name=name)

@dispatch.dispatch_for_api(array_ops.size_v2)
def size(input: ragged_tensor.Ragged, out_type=dtypes.int32, name=None):
    if False:
        return 10
    'Returns the size of a potentially ragged tensor.\n\n  The size of a ragged tensor is the size of its inner values.\n\n  #### Example:\n\n  >>> tf.size(tf.ragged.constant([[1, 2], [3]])).numpy()\n  3\n\n  Args:\n    input: A potentially ragged `Tensor`.\n    out_type: The numeric output type for the operation.\n    name: A name for the operation (optional).\n\n  Returns:\n    A Tensor of type `out_type`.\n  '
    if ragged_tensor.is_ragged(input):
        return array_ops.size(input.flat_values, out_type=out_type, name=name)
    else:
        return array_ops.size(input, out_type=out_type, name=name)

@dispatch.dispatch_for_api(array_ops.size)
def _ragged_size_v1(input: ragged_tensor.Ragged, name=None, out_type=dtypes.int32):
    if False:
        return 10
    return size(input=input, out_type=out_type, name=name)

@dispatch.dispatch_for_api(array_ops.rank)
def rank(input: ragged_tensor.Ragged, name=None):
    if False:
        while True:
            i = 10
    "Returns the rank of a RaggedTensor.\n\n  Returns a 0-D `int32` `Tensor` representing the rank of `input`.\n\n  #### Example:\n\n  >>> # shape of tensor 't' is [2, None, None]\n  >>> t = tf.ragged.constant([[[1], [2, 2]], [[3, 3, 3], [4, 4, 4, 4]]])\n  >>> tf.rank(t).numpy()\n  3\n\n  Args:\n    input: A `RaggedTensor`\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` of type `int32`.\n  "
    with ops.name_scope(name, 'RaggedRank', [input]) as name:
        if not ragged_tensor.is_ragged(input):
            return array_ops.rank(input, name)
        return input.ragged_rank + array_ops.rank(input.flat_values)

@dispatch.dispatch_for_api(array_ops.one_hot)
def ragged_one_hot(indices: ragged_tensor.Ragged, depth, on_value=None, off_value=None, axis=None, dtype=None, name=None):
    if False:
        print('Hello World!')
    'Applies tf.one_hot along the values of a RaggedTensor.'
    if isinstance(axis, int) and axis >= 0:
        if axis <= indices.ragged_rank:
            raise ValueError('axis (%d) must be greater than indices.ragged_rank (%d).' % (axis, indices.ragged_rank))
        axis -= indices.ragged_rank
    with ops.name_scope(name, 'RaggedOneHot', [indices, depth, on_value, off_value, axis]):
        indices = ragged_tensor.convert_to_tensor_or_ragged_tensor(indices, name='indices')
        return indices.with_flat_values(array_ops.one_hot(indices.flat_values, depth, on_value, off_value, axis, dtype, name))

@tf_export('ragged.stack_dynamic_partitions')
@dispatch.add_dispatch_support
def stack_dynamic_partitions(data, partitions, num_partitions, name=None):
    if False:
        i = 10
        return i + 15
    "Stacks dynamic partitions of a Tensor or RaggedTensor.\n\n  Returns a RaggedTensor `output` with `num_partitions` rows, where the row\n  `output[i]` is formed by stacking all slices `data[j1...jN]` such that\n  `partitions[j1...jN] = i`.  Slices of `data` are stacked in row-major\n  order.\n\n  If `num_partitions` is an `int` (not a `Tensor`), then this is equivalent to\n  `tf.ragged.stack(tf.dynamic_partition(data, partitions, num_partitions))`.\n\n  #### Example:\n\n  >>> data           = ['a', 'b', 'c', 'd', 'e']\n  >>> partitions     = [  3,   0,   2,   2,   3]\n  >>> num_partitions = 5\n  >>> tf.ragged.stack_dynamic_partitions(data, partitions, num_partitions)\n  <tf.RaggedTensor [[b'b'], [], [b'c', b'd'], [b'a', b'e'], []]>\n\n  Args:\n    data: A `Tensor` or `RaggedTensor` containing the values to stack.\n    partitions: An `int32` or `int64` `Tensor` or `RaggedTensor` specifying the\n      partition that each slice of `data` should be added to. `partitions.shape`\n      must be a prefix of `data.shape`.  Values must be greater than or equal to\n      zero, and less than `num_partitions`. `partitions` is not required to be\n      sorted.\n    num_partitions: An `int32` or `int64` scalar specifying the number of\n      partitions to output.  This determines the number of rows in `output`.\n    name: A name prefix for the returned tensor (optional).\n\n  Returns:\n    A `RaggedTensor` containing the stacked partitions.  The returned tensor\n    has the same dtype as `data`, and its shape is\n    `[num_partitions, (D)] + data.shape[partitions.rank:]`, where `(D)` is a\n    ragged dimension whose length is the number of data slices stacked for\n    each `partition`.\n  "
    with ops.name_scope(name, 'SegmentStack', [data, partitions, num_partitions]):
        data = ragged_tensor.convert_to_tensor_or_ragged_tensor(data, name='data')
        row_splits_dtype = data.row_splits.dtype if isinstance(data, ragged_tensor.RaggedTensor) else None
        partitions = ragged_tensor.convert_to_tensor_or_ragged_tensor(partitions, name='partitions', preferred_dtype=row_splits_dtype)
        num_partitions = ops.convert_to_tensor(num_partitions, name='num_partitions', preferred_dtype=partitions.dtype)
        if row_splits_dtype is not None:
            partitions = math_ops.cast(partitions, row_splits_dtype)
        num_partitions = math_ops.cast(num_partitions, partitions.dtype)
        partitions_rank = partitions.shape.ndims
        if partitions_rank is None:
            raise ValueError('partitions must have known rank.')
        num_partitions.shape.assert_has_rank(0)
        partitions.shape.assert_is_compatible_with(data.shape[:partitions_rank])
        if partitions_rank == 0:
            return ragged_tensor.RaggedTensor.from_value_rowids(values=array_ops_stack.stack([data]), value_rowids=array_ops_stack.stack([partitions]), nrows=num_partitions, validate=False)
        elif partitions_rank == 1:
            permutation = sort_ops.argsort(partitions, stable=True)
            value_rowids = array_ops.gather(partitions, permutation)
            values = array_ops.gather(data, permutation)
            checks = [check_ops.assert_less(value_rowids[-1:], num_partitions, message='partitions must be less than num_partitions'), check_ops.assert_non_negative(partitions, message='partitions must be non-negative.')]
            with ops.control_dependencies(checks):
                return ragged_tensor.RaggedTensor.from_value_rowids(values, value_rowids, nrows=num_partitions, validate=False)
        else:
            if not isinstance(data, ragged_tensor.RaggedTensor):
                data = ragged_tensor.RaggedTensor.from_tensor(data, row_splits_dtype=partitions.dtype, ragged_rank=1)
            if not isinstance(partitions, ragged_tensor.RaggedTensor):
                partitions = ragged_tensor.RaggedTensor.from_tensor(partitions, row_splits_dtype=partitions.dtype, ragged_rank=max(data.ragged_rank, partitions_rank - 1))
            check = check_ops.assert_equal(data.row_splits, partitions.row_splits, message='data and partitions have incompatible ragged shapes')
            with ops.control_dependencies([check]):
                return stack_dynamic_partitions(data.values, partitions.values, num_partitions)

@dispatch.dispatch_for_api(array_ops.reverse)
def reverse(tensor: ragged_tensor.Ragged, axis, name=None):
    if False:
        for i in range(10):
            print('nop')
    "Reverses a RaggedTensor along the specified axes.\n\n  #### Example:\n\n  >>> data = tf.ragged.constant([\n  ...   [[1, 2], [3, 4]], [[5, 6]], [[7, 8], [9, 10], [11, 12]]])\n  >>> tf.reverse(data, axis=[0, 2])\n  <tf.RaggedTensor [[[8, 7], [10, 9], [12, 11]], [[6, 5]], [[2, 1], [4, 3]]]>\n\n  Args:\n    tensor: A 'RaggedTensor' to reverse.\n    axis: A list or tuple of 'int' or a constant 1D 'tf.Tensor'. The indices of\n      the axes to reverse.\n    name: A name prefix for the returned tensor (optional).\n\n  Returns:\n    A 'RaggedTensor'.\n  "
    type_error_msg = '`axis` must be a list of int or a constant tensorwhen reversing axes in a ragged tensor'
    with ops.name_scope(name, 'Reverse', [tensor, axis]):
        if isinstance(axis, tensor_lib.Tensor):
            axis = tensor_util.constant_value(axis)
            if axis is None:
                raise TypeError(type_error_msg)
        elif not (isinstance(axis, (list, tuple)) and all((isinstance(dim, int) for dim in axis))):
            raise TypeError(type_error_msg)
        tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(tensor, name='tensor')
        axis = [array_ops.get_positive_axis(dim, tensor.shape.rank, 'axis[%d]' % i, 'rank(tensor)') for (i, dim) in enumerate(axis)]
        slices = [slice(None)] * (max(axis) + 1 if axis else 0)
        for dim in axis:
            slices[dim] = slice(None, None, -1)
        return tensor[tuple(slices)]

@tf_export('ragged.cross')
@dispatch.add_dispatch_support
def cross(inputs, name=None):
    if False:
        return 10
    "Generates feature cross from a list of tensors.\n\n  The input tensors must have `rank=2`, and must all have the same number of\n  rows.  The result is a `RaggedTensor` with the same number of rows as the\n  inputs, where `result[row]` contains a list of all combinations of values\n  formed by taking a single value from each input's corresponding row\n  (`inputs[i][row]`).  Values are combined by joining their strings with '_X_'.\n  E.g.:\n\n  >>> tf.ragged.cross([tf.ragged.constant([['a'], ['b', 'c']]),\n  ...                  tf.ragged.constant([['d'], ['e']]),\n  ...                  tf.ragged.constant([['f'], ['g']])])\n  <tf.RaggedTensor [[b'a_X_d_X_f'], [b'b_X_e_X_g', b'c_X_e_X_g']]>\n\n  Args:\n    inputs: A list of `RaggedTensor` or `Tensor` or `SparseTensor`.\n    name: Optional name for the op.\n\n  Returns:\n    A 2D `RaggedTensor` of type `string`.\n  "
    return _cross_internal(inputs=inputs, hashed_output=False, name=name)

@tf_export('ragged.cross_hashed')
@dispatch.add_dispatch_support
def cross_hashed(inputs, num_buckets=0, hash_key=None, name=None):
    if False:
        while True:
            i = 10
    "Generates hashed feature cross from a list of tensors.\n\n  The input tensors must have `rank=2`, and must all have the same number of\n  rows.  The result is a `RaggedTensor` with the same number of rows as the\n  inputs, where `result[row]` contains a list of all combinations of values\n  formed by taking a single value from each input's corresponding row\n  (`inputs[i][row]`).  Values are combined by hashing together their\n  fingerprints. E.g.:\n\n  >>> tf.ragged.cross_hashed([tf.ragged.constant([['a'], ['b', 'c']]),\n  ...                         tf.ragged.constant([['d'], ['e']]),\n  ...                         tf.ragged.constant([['f'], ['g']])],\n  ...                        num_buckets=100)\n  <tf.RaggedTensor [[78], [66, 74]]>\n\n  Args:\n    inputs: A list of `RaggedTensor` or `Tensor` or `SparseTensor`.\n    num_buckets: A non-negative `int` that used to bucket the hashed values. If\n      `num_buckets != 0`, then `output = hashed_value % num_buckets`.\n    hash_key: Integer hash_key that will be used by the `FingerprintCat64`\n      function. If not given, a default key is used.\n    name: Optional name for the op.\n\n  Returns:\n    A 2D `RaggedTensor` of type `int64`.\n  "
    return _cross_internal(inputs=inputs, hashed_output=True, num_buckets=num_buckets, hash_key=hash_key, name=name)
_DEFAULT_CROSS_HASH_KEY = 956888297470

def _cross_internal(inputs, hashed_output=False, num_buckets=0, hash_key=None, name=None):
    if False:
        while True:
            i = 10
    'Generates feature cross from a list of ragged and dense tensors.'
    if not isinstance(inputs, (tuple, list)):
        raise TypeError('Inputs must be a list')
    if hash_key is None:
        hash_key = _DEFAULT_CROSS_HASH_KEY
    ragged_inputs = []
    sparse_inputs = []
    dense_inputs = []
    input_order = []
    with ops.name_scope(name, 'RaggedCross', inputs):
        for (i, t) in enumerate(inputs):
            if sparse_tensor.is_sparse(t):
                t = sparse_tensor.SparseTensor.from_value(t)
            else:
                t = ragged_tensor.convert_to_tensor_or_ragged_tensor(t)
            if t.dtype.is_integer:
                t = math_ops.cast(t, dtypes.int64)
            elif t.dtype != dtypes.string:
                raise ValueError('Unexpected dtype for inputs[%d]: %s' % (i, t.dtype))
            if isinstance(t, ragged_tensor.RaggedTensor):
                if t.ragged_rank != 1:
                    raise ValueError('tf.ragged.cross only supports inputs with rank=2')
                ragged_inputs.append(t)
                input_order.append('R')
            elif isinstance(t, sparse_tensor.SparseTensor):
                sparse_inputs.append(t)
                input_order.append('S')
            else:
                dense_inputs.append(t)
                input_order.append('D')
        out_values_type = dtypes.int64 if hashed_output else dtypes.string
        if ragged_inputs and all((t.row_splits.dtype == dtypes.int32 for t in ragged_inputs)):
            out_row_splits_type = dtypes.int32
        else:
            out_row_splits_type = dtypes.int64
        if hash_key > 2 ** 63:
            hash_key -= 2 ** 64
        (values_out, splits_out) = gen_ragged_array_ops.ragged_cross(ragged_values=[rt.values for rt in ragged_inputs], ragged_row_splits=[rt.row_splits for rt in ragged_inputs], sparse_indices=[st.indices for st in sparse_inputs], sparse_values=[st.values for st in sparse_inputs], sparse_shape=[st.dense_shape for st in sparse_inputs], dense_inputs=dense_inputs, input_order=''.join(input_order), hashed_output=hashed_output, num_buckets=num_buckets, hash_key=hash_key, out_values_type=out_values_type.as_datatype_enum, out_row_splits_type=out_row_splits_type.as_datatype_enum, name=name)
        return ragged_tensor.RaggedTensor.from_row_splits(values_out, splits_out, validate=False)

def fill_empty_rows(ragged_input, default_value, name=None):
    if False:
        i = 10
        return i + 15
    'Fills empty rows in the input `RaggedTensor` with rank 2 with a default\n\n  value.\n\n  This op adds entries with the specified `default_value` for any row in the\n  input that does not already have a value.\n\n  The op also returns an indicator vector such that\n\n      empty_row_indicator[i] = True iff row i was an empty row.\n\n  Args:\n    ragged_input: A `RaggedTensor` with rank 2.\n    default_value: The value to fill for empty rows, with the same type as\n      `ragged_input.`\n    name: A name prefix for the returned tensors (optional)\n\n  Returns:\n    ragged_ordered_output: A `RaggedTensor`with all empty rows filled in with\n      `default_value`.\n    empty_row_indicator: A bool vector indicating whether each input row was\n      empty.\n\n  Raises:\n    TypeError: If `ragged_input` is not a `RaggedTensor`.\n  '
    with ops.name_scope(name, 'RaggedFillEmptyRows', [ragged_input]):
        if not isinstance(ragged_input, ragged_tensor.RaggedTensor):
            raise TypeError(f'ragged_input must be RaggedTensor,             got {type(ragged_input)}')
        default_value = ops.convert_to_tensor(default_value, dtype=ragged_input.dtype)
        (output_value_rowids, output_values, empty_row_indicator, unused_reverse_index_map) = gen_ragged_array_ops.ragged_fill_empty_rows(value_rowids=ragged_input.value_rowids(), values=ragged_input.values, nrows=ragged_input.nrows(), default_value=default_value)
        return (ragged_tensor.RaggedTensor.from_value_rowids(values=output_values, value_rowids=output_value_rowids, validate=False), empty_row_indicator)

@ops.RegisterGradient('RaggedFillEmptyRows')
def _ragged_fill_empty_rows_grad(op, unused_grad_output_indices, output_grad_values, unused_grad_empty_row_indicator, unused_grad_reverse_index_map):
    if False:
        return 10
    'Gradients for RaggedFillEmptyRows.'
    reverse_index_map = op.outputs[3]
    (d_values, d_default_value) = gen_ragged_array_ops.ragged_fill_empty_rows_grad(reverse_index_map=reverse_index_map, grad_values=output_grad_values)
    return [None, d_values, None, d_default_value]

@dispatch.dispatch_for_api(data_flow_ops.dynamic_partition)
def dynamic_partition(data: ragged_tensor.RaggedOrDense, partitions: ragged_tensor.RaggedOrDense, num_partitions, name=None):
    if False:
        while True:
            i = 10
    'RaggedTensor dispatch override for tf.dynamic_partition.'
    if not isinstance(num_partitions, int) or num_partitions < 0:
        raise TypeError('num_partitions must be a non-negative integer')
    result = stack_dynamic_partitions(data, partitions, num_partitions, name)
    return [result[i] for i in range(num_partitions)]

@dispatch.dispatch_for_api(array_ops.split)
def split(value: ragged_tensor.Ragged, num_or_size_splits, axis=0, num=None, name=None):
    if False:
        print('Hello World!')
    "Splits a RaggedTensor `value` into a list of sub RaggedTensors.\n\n  If `num_or_size_splits` is an `int`,  then it splits `value` along the\n  dimension `axis` into `num_or_size_splits` smaller RaggedTensors. This\n  requires that `value.shape[axis]` is divisible by `num_or_size_splits`.\n\n  If `num_or_size_splits` is a 1-D Tensor (or list), then `value` is split into\n  `len(num_or_size_splits)` elements. The shape of the `i`-th element has the\n  same size as the `value` except along dimension `axis` where the size is\n  `num_or_size_splits[i]`.\n\n  Splits along a ragged dimension is not allowed.\n\n  For example:\n\n  >>> rt = tf.RaggedTensor.from_row_lengths(\n  ...      np.arange(6 * 3).reshape(6, 3), row_lengths=[1, 2, 2, 1])\n  >>> rt.shape\n  TensorShape([4, None, 3])\n  >>>\n  >>> rt1, rt2 = tf.split(rt, 2)  # uniform splits\n  >>> rt1.shape\n  TensorShape([2, None, 3])\n  >>> rt2.shape\n  TensorShape([2, None, 3])\n  >>>\n  >>> rt3, rt4, rt5 = tf.split(rt, [1, 2, 1])  # ragged splits\n  >>> rt3.shape\n  TensorShape([1, None, 3])\n  >>> rt4.shape\n  TensorShape([2, None, 3])\n  >>> rt5.shape\n  TensorShape([1, None, 3])\n  >>>\n  >>> rt6, rt7 = tf.split(rt, [1, 2], axis=2)  # splits along axis 2\n  >>> rt6.shape\n  TensorShape([4, None, 1])\n  >>> rt7.shape\n  TensorShape([4, None, 2])\n\n  Args:\n    value: The `RaggedTensor` to split.\n    num_or_size_splits: Either an `int` indicating the number of splits\n      along `axis` or a 1-D integer `Tensor` or Python list containing the sizes\n      of each output tensor along `axis`. If a Python int, then it must evenly\n      divide `value.shape[axis]`; otherwise the sum of sizes along the split\n      axis must match that of the `value`.\n    axis: An `int` or scalar `int32` `Tensor`. The dimension along which\n      to split. Must be in the range `[-rank(value), rank(value))`. Defaults to\n      0.\n    num: An `int` used to specify the number of outputs when\n      `num_or_size_splits` is a 1-D list or `Tensor` and its length is\n      statically unknown, e.g., specifying `tf.TensorSepc(None)` with\n      the `input_signature` argument of `tf.function` (optional).\n    name: A name for the operation (optional).\n\n  Returns:\n    if `num_or_size_splits` is an `int` returns a list of `num_or_size_splits`\n    `RaggedTensor` objects; if `num_or_size_splits` is a 1-D Tensor returns\n    `num_or_size_splits.get_shape[0]` `RaggedTensor` objects resulting from\n    splitting `value`.\n\n  Raises:\n    ValueError: If the dimension `axis` of `value` is a ragged dimension.\n    ValueError: If `num` is unspecified and cannot be inferred.\n    ValueError: If `num` is specified but doesn't match the length of\n      `num_or_size_splits`.\n    ValueError: If `num_or_size_splits` is an `int` and less than 1.\n    TypeError: If `num_or_size_splits` is not an `int` or 1-D\n      list or 1-D `Tensor`.\n    InvalidArgumentError: If the `axis` of `value` cannot be exactly splitted\n      by `num_or_size_splits`.\n    InvalidArgumentError: If `num_or_size_splits` is contains negative integers.\n    InvalidArgumentError: If `num_or_size_splits`'s static shape is unknown and\n      its dynamic shape is inconsistent `num`.\n    InvalidArgumentError: If `num_or_size_splits`'s static rank is unknown and\n      `axis` is a negative integer.\n  "
    with ops.name_scope(name, 'RaggedSplit'):
        value = ragged_tensor.convert_to_tensor_or_ragged_tensor(value, name='value')
        if isinstance(num_or_size_splits, int) and num_or_size_splits == 1:
            return [value]
        check_ops.assert_integer_v2(num_or_size_splits, message='`num_or_size_splits` must be an `int` or 1-D list or `Tensor` of integers.')
        value_shape = dynamic_ragged_shape.DynamicRaggedShape.from_tensor(value)
        axis = array_ops.get_positive_axis(axis, value_shape.rank)
        try:
            dim_size = value_shape[axis]
        except ValueError:
            raise ValueError(f'Cannot split a ragged dimension. Got `value` with shape {value_shape} and `axis` {axis}.')
        if isinstance(num_or_size_splits, int):
            num_splits = num_or_size_splits
            if num_splits < 1:
                raise ValueError(f'`num_or_size_splits` must be >=1 if it is an `int`.Received {num_or_size_splits}.')
            split_length = math_ops.floordiv(dim_size, num_splits)
            split_lengths = array_ops.repeat(split_length, num_splits)
        else:
            num_splits = None
            split_lengths = ops.convert_to_tensor(num_or_size_splits)
            if split_lengths.shape.ndims is not None:
                if split_lengths.shape.ndims != 1:
                    raise TypeError(f'`num_or_size_splits` must be an `int` or 1-D list or `Tensor`. Received {num_or_size_splits}.')
                num_splits = tensor_shape.dimension_value(split_lengths.shape[0])
            if num_splits is None:
                if num is None:
                    raise ValueError(f'`num` must be specified as an `int` when the size of `num_or_size_split` is statically unknown. Received `num`: {num} and `num_or_size_split`: {num_or_size_splits}.')
                num_splits = num
            elif num is not None and num != num_splits:
                raise ValueError(f'`num` does not match the size of `num_or_size_split`. Received `num`: {num} and size of `num_or_size_split`: {num_splits}.')
        splits = array_ops.concat([[0], math_ops.cumsum(split_lengths)], axis=0)
        checks = []
        checks.append(check_ops.assert_non_negative_v2(num_or_size_splits, message='`num_or_size_splits` must be non-negative.'))
        checks.append(check_ops.assert_equal_v2(num_splits, array_ops.shape(split_lengths)[0], message='`num` is inconsistent with `num_or_size_split.shape[0]`.'))
        checks.append(check_ops.assert_equal_v2(math_ops.cast(dim_size, splits.dtype), splits[-1], message='Cannot exactly split the `axis` dimension of `value` with the given `num_or_size_split`.'))
        splits = control_flow_ops.with_dependencies(checks, splits)
        splited_rts = []
        slices = [slice(None)] * (axis + 1)
        for i in range(num_splits):
            slices[-1] = slice(splits[i], splits[i + 1])
            splited_rts.append(value[tuple(slices)])
        return splited_rts

@dispatch.dispatch_for_api(array_ops.reshape)
def ragged_reshape(tensor: ragged_tensor.RaggedOrDense, shape: dynamic_ragged_shape.DenseOrRaggedShape) -> Union[ragged_tensor.RaggedTensor, tensor_lib.Tensor]:
    if False:
        for i in range(10):
            print('nop')
    'Reshapes a tensor or ragged tensor.'
    tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(tensor, name='tensor')
    if isinstance(tensor, ragged_tensor.RaggedTensor):
        tensor = tensor.values
    if isinstance(shape, dynamic_ragged_shape.DynamicRaggedShape):
        flat_values = array_ops.reshape(tensor, shape.inner_shape)
        return ragged_tensor.RaggedTensor._from_nested_row_partitions(flat_values, shape.row_partitions, validate=False)
    else:
        shape = ops.convert_to_tensor(shape, name='shape')
        return array_ops.reshape(tensor, shape)

@dispatch.dispatch_for_api(array_ops.broadcast_to)
def broadcast_to(input: ragged_tensor.RaggedOrDense, shape: dynamic_ragged_shape.DynamicRaggedShape) -> Union[ragged_tensor.RaggedTensor, tensor_lib.Tensor]:
    if False:
        for i in range(10):
            print('nop')
    'Broadcasts a potentially ragged tensor to a ragged shape.\n\n  Tiles `input` as necessary to match the given shape.\n\n  Behavior is undefined if `input` is not broadcast-compatible with `shape`.\n\n  Args:\n    input: The potentially ragged tensor to broadcast.\n    shape: A `DynamicRaggedShape`\n\n  Returns:\n    A potentially ragged tensor whose values are taken from\n    `input`, and whose shape matches `shape`.\n  '
    return dynamic_ragged_shape.broadcast_to(input, shape)

@dispatch.dispatch_for_api(array_ops.shape)
def ragged_shape(input: ragged_tensor.Ragged, name: Optional[str]=None, out_type=dtypes.int32) -> dynamic_ragged_shape.DynamicRaggedShape:
    if False:
        while True:
            i = 10
    'Returns the shape of a RaggedTensor.\n\n  Args:\n    input: A `RaggedTensor`\n    name: A name for the operation (optional).\n    out_type: dtype used to encode the shape.\n\n  Returns:\n    A `tf.experimental.DynamicRaggedShape`\n  '
    with ops.name_scope(name, 'RaggedShape', [input]):
        return dynamic_ragged_shape.DynamicRaggedShape.from_tensor(input, out_type)

@dispatch.dispatch_for_api(array_ops.broadcast_dynamic_shape)
def broadcast_dynamic_shape(shape_x: dynamic_ragged_shape.DenseOrRaggedShape, shape_y: dynamic_ragged_shape.DenseOrRaggedShape) -> dynamic_ragged_shape.DynamicRaggedShape:
    if False:
        print('Hello World!')
    "Returns the shape formed by broadcasting two shapes to be compatible.\n\n  1. If shape_x and shape_y both have row_partitions, then fail if their dtypes\n     don't match.\n  2. If neither has row_partitions and they have different dtypes,\n     go with int64.\n  3. If one has row_partitions, go with that dtype.\n\n  Args:\n    shape_x: A `DynamicRaggedShape`\n    shape_y: A `DynamicRaggedShape`\n\n  Returns:\n    A `DynamicRaggedShape`.\n  Raises:\n    ValueError: If `shape_x` and `shape_y` are not broadcast-compatible.\n  "
    if not isinstance(shape_x, dynamic_ragged_shape.DynamicRaggedShape):
        shape_x = dynamic_ragged_shape.DynamicRaggedShape([], shape_x)
    if not isinstance(shape_y, dynamic_ragged_shape.DynamicRaggedShape):
        shape_y = dynamic_ragged_shape.DynamicRaggedShape([], shape_y)
    return dynamic_ragged_shape.broadcast_dynamic_shape(shape_x, shape_y)

@dispatch.dispatch_for_api(array_ops.ones)
def ones(shape: dynamic_ragged_shape.DynamicRaggedShape, dtype=dtypes.float32, name=None, layout=None) -> ragged_tensor.RaggedOrDense:
    if False:
        return 10
    'Returns ones shaped like x.'
    if layout is not None and (not layout.is_fully_replicated()):
        raise ValueError(f'RaggedTensor only allows replicated layout. got {layout}')
    flat_values = array_ops.ones(shape.inner_shape, dtype=dtype, name=name, layout=layout)
    return shape._add_row_partitions(flat_values)

@dispatch.dispatch_for_api(array_ops.zeros)
def zeros(shape: dynamic_ragged_shape.DynamicRaggedShape, dtype=dtypes.float32, name=None, layout=None) -> ragged_tensor.RaggedOrDense:
    if False:
        while True:
            i = 10
    'Returns ones shaped like x.'
    if layout is not None and (not layout.is_fully_replicated()):
        raise ValueError(f'RaggedTensor only allows replicated layout. got {layout}')
    flat_values = array_ops.zeros(shape.inner_shape, dtype=dtype, name=name, layout=layout)
    return shape._add_row_partitions(flat_values)

@dispatch.dispatch_for_api(array_ops.fill)
def fill(dims: dynamic_ragged_shape.DynamicRaggedShape, value: core_types.TensorLike, name: Optional[str]=None, layout=None) -> ragged_tensor.RaggedOrDense:
    if False:
        print('Hello World!')
    'Creates a tensor with shape `dims` and fills it with `value`.'
    if layout is not None and (not layout.is_fully_replicated()):
        raise ValueError(f'RaggedTensor only allows replicated layout. got {layout}')
    flat_values = array_ops.fill(dims.inner_shape, value, name=name, layout=layout)
    return dims._add_row_partitions(flat_values)

@dispatch.dispatch_for_api(array_ops.bitcast)
def bitcast(input: ragged_tensor.RaggedOrDense, type, name=None) -> ragged_tensor.RaggedOrDense:
    if False:
        return 10
    'RaggedTensor dispatch override for tf.bitcast.'
    type = dtypes.as_dtype(type)
    with ops.name_scope(name, 'Bitcast', [input]):
        input = ragged_tensor.convert_to_tensor_or_ragged_tensor(input, name='input')
        if input.dtype.size < type.size and input.flat_values.shape.rank < 2:
            raise ValueError(f'`input.flat_values` is required to have rank >= 2 when input.dtype.size < type.size. Actual rank: {input.flat_values.shape.rank}')
        return input.with_flat_values(array_ops.bitcast(input.flat_values, type))