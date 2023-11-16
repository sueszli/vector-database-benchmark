"""Support for ragged tensors."""
import functools
import typing
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

@tf_export('ragged.range')
@dispatch.add_dispatch_support
def range(starts, limits=None, deltas=1, dtype=None, name=None, row_splits_dtype=dtypes.int64):
    if False:
        while True:
            i = 10
    "Returns a `RaggedTensor` containing the specified sequences of numbers.\n\n  Each row of the returned `RaggedTensor` contains a single sequence:\n\n  ```python\n  ragged.range(starts, limits, deltas)[i] ==\n      tf.range(starts[i], limits[i], deltas[i])\n  ```\n\n  If `start[i] < limits[i] and deltas[i] > 0`, then `output[i]` will be an\n  empty list.  Similarly, if `start[i] > limits[i] and deltas[i] < 0`, then\n  `output[i]` will be an empty list.  This behavior is consistent with the\n  Python `range` function, but differs from the `tf.range` op, which returns\n  an error for these cases.\n\n  Examples:\n\n  >>> tf.ragged.range([3, 5, 2]).to_list()\n  [[0, 1, 2], [0, 1, 2, 3, 4], [0, 1]]\n  >>> tf.ragged.range([0, 5, 8], [3, 3, 12]).to_list()\n  [[0, 1, 2], [], [8, 9, 10, 11]]\n  >>> tf.ragged.range([0, 5, 8], [3, 3, 12], 2).to_list()\n  [[0, 2], [], [8, 10]]\n\n  The input tensors `starts`, `limits`, and `deltas` may be scalars or vectors.\n  The vector inputs must all have the same size.  Scalar inputs are broadcast\n  to match the size of the vector inputs.\n\n  Args:\n    starts: Vector or scalar `Tensor`.  Specifies the first entry for each range\n      if `limits` is not `None`; otherwise, specifies the range limits, and the\n      first entries default to `0`.\n    limits: Vector or scalar `Tensor`.  Specifies the exclusive upper limits for\n      each range.\n    deltas: Vector or scalar `Tensor`.  Specifies the increment for each range.\n      Defaults to `1`.\n    dtype: The type of the elements of the resulting tensor.  If not specified,\n      then a value is chosen based on the other args.\n    name: A name for the operation.\n    row_splits_dtype: `dtype` for the returned `RaggedTensor`'s `row_splits`\n      tensor.  One of `tf.int32` or `tf.int64`.\n\n  Returns:\n    A `RaggedTensor` of type `dtype` with `ragged_rank=1`.\n  "
    row_splits_dtype = dtypes.as_dtype(row_splits_dtype)
    if limits is None:
        (starts, limits) = (0, starts)
    with ops.name_scope(name, 'RaggedRange', [starts, limits, deltas]) as name:
        starts = ops.convert_to_tensor(starts, dtype=dtype, name='starts')
        limits = ops.convert_to_tensor(limits, dtype=dtype, name='limits')
        deltas = ops.convert_to_tensor(deltas, dtype=dtype, name='deltas')
        if dtype is None:
            (starts, limits, deltas) = _infer_matching_dtype([starts, limits, deltas], [dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64])
        result = gen_ragged_math_ops.ragged_range(starts, limits, deltas, Tsplits=row_splits_dtype, name=name)
        return ragged_tensor.RaggedTensor.from_row_splits(result.rt_dense_values, result.rt_nested_splits, validate=False)

def _infer_matching_dtype(tensors, dtype_hierarchy):
    if False:
        print('Hello World!')
    'Infers a matching dtype for tensors, and casts them to that dtype.'
    assert all((t.dtype in dtype_hierarchy for t in tensors))
    inferred_dtype = max([t.dtype for t in tensors], key=dtype_hierarchy.index)
    return [math_ops.cast(t, inferred_dtype) for t in tensors]
ops.no_gradient('RaggedRange')
_RAGGED_SEGMENT_DOCSTRING = 'Computes the %(combination)s along segments of a RaggedTensor.\n\n  Returns a RaggedTensor `output` with `num_segments` rows, where the row\n  `output[i]` is formed by taking the %(combination)s of all rows of `data`\n  whose corresponding `segment_id` is `i`.\n\n  The length of the row `output[i]` will be the maximum of the lengths of\n  all rows of `data` whose corresponding `segment_id` is `i`.  If no `data`\n  rows correspond to a given segment ID, then the output row for that segment\n  ID will be empty.\n\n  Args:\n    data: A `RaggedTensor` containing the values to combine.\n    segment_ids: A `Tensor` or `RaggedTensor`.  Must have type `int64` or\n      `int32`.  `segment_ids.shape` must be a prefix of `data.shape`.\n      Must be greater than or equal to zero, and less than `num_segments`.\n      `segment_ids` is not required to be sorted.\n    num_segments: An `int32` or `int64` scalar specifying the number of\n      distinct segment ids.\n    name: A name prefix for the returned tensor (optional).\n  Returns:\n    A `RaggedTensor` containing the %(combined)s values.  The returned tensor\n    has the same dtype as `data`, and its shape is\n    `[num_segments] + data.shape[segment_ids.rank:]`.\n  Raises:\n    ValueError: If `segment_ids.shape` is not a prefix of `data.shape`.\n'

def _ragged_segment_aggregate(unsorted_segment_op, data, segment_ids, num_segments, separator=None, name=None):
    if False:
        return 10
    'Aggregates along segments of a RaggedTensor using `unsorted_segment_op`.\n\n  Returns a RaggedTensor `output` with `num_segments` rows, where the row\n  `output[i]` is formed by combining all rows of `data` whose corresponding\n  `segment_id` is `i`.  The values in each row are combined using\n  `unsorted_segment_op`.\n\n  The length of the row `output[i]` will be the maximum of the lengths of\n  all rows of `data` whose corresponding `segment_id` is `i`.  If no `data`\n  rows correspond to a given segment ID, then the output row for that segment\n  ID will be empty.\n\n  Args:\n    unsorted_segment_op: The tensorflow `op` that should be used to combine\n      values in each row.  Must have the same signature and basic behavior as\n      `unsorted_segment_sum`, `unsorted_segment_max`, etc.\n    data: A `RaggedTensor` containing the values to be combined.\n    segment_ids: A `Tensor` or `RaggedTensor`.  Must have type `int64` or\n      `int32`.  `segment_ids.shape` must be a prefix of `data.shape`.\n      `segment_ids` is not required to be sorted.\n    num_segments: An `int32` or `int64` scalar.\n    separator: An optional string. Defaults to None. The separator to use when\n      joining. Only used for string types.\n    name: A name prefix for the returned tensor (optional).\n\n  Returns:\n    A `RaggedTensor` containing the aggregated values.  The returned tensor\n    has the same dtype as `data`, and its shape is\n    `[num_segments] + data.shape[segment_ids.rank:]`.\n  Raises:\n    ValueError: If segment_ids.shape is not a prefix of data.shape.\n  '
    if not (ragged_tensor.is_ragged(data) or ragged_tensor.is_ragged(segment_ids)):
        if separator is not None:
            return unsorted_segment_op(data, segment_ids, num_segments, separator, name)
        else:
            return unsorted_segment_op(data, segment_ids, num_segments, name)
    with ops.name_scope(name, 'RaggedSegment', [data, segment_ids, num_segments]) as name:
        data = ragged_tensor.convert_to_tensor_or_ragged_tensor(data, name='data')
        segment_ids = ragged_tensor.convert_to_tensor_or_ragged_tensor(segment_ids, name='segment_ids')
        (data, segment_ids) = ragged_tensor.match_row_splits_dtypes(data, segment_ids)
        if segment_ids.dtype not in (dtypes.int32, dtypes.int64):
            raise ValueError('segment_ids must have dtype int32 or int64.')
        if ragged_tensor.is_ragged(segment_ids):
            if not ragged_tensor.is_ragged(data):
                raise ValueError('segment_ids.shape must be a prefix of data.shape, but segment_ids is ragged and data is not.')
            check_splits = check_ops.assert_equal(segment_ids.row_splits, data.row_splits, message='segment_ids.shape must be a prefix of data.shape')
            with ops.control_dependencies([check_splits]):
                return _ragged_segment_aggregate(unsorted_segment_op, data.values, segment_ids.values, num_segments, separator)
        data_row_lengths = data.row_splits[1:] - data.row_splits[:-1]
        output_row_lengths = math_ops.maximum(math_ops.unsorted_segment_max(data_row_lengths, segment_ids, num_segments), 0)
        output_splits = array_ops.concat([array_ops.zeros([1], output_row_lengths.dtype), math_ops.cumsum(output_row_lengths)], axis=0)
        data_row_to_out_row_start = array_ops.gather(output_splits, segment_ids)
        data_row_to_out_row_limit = data_row_to_out_row_start + data_row_lengths
        data_val_to_out_val_index = range(data_row_to_out_row_start, data_row_to_out_row_limit).values
        output_values = _ragged_segment_aggregate(unsorted_segment_op, data.values, data_val_to_out_val_index, output_splits[-1], separator)
        return ragged_tensor.RaggedTensor.from_row_splits(output_values, output_splits, validate=False)

@dispatch.dispatch_for_api(math_ops.unsorted_segment_sum)
def segment_sum(data: ragged_tensor.RaggedOrDense, segment_ids: ragged_tensor.RaggedOrDense, num_segments, name=None):
    if False:
        print('Hello World!')
    return _ragged_segment_aggregate(math_ops.unsorted_segment_sum, data=data, segment_ids=segment_ids, num_segments=num_segments, name=name or 'RaggedSegmentSum')

@dispatch.dispatch_for_api(math_ops.unsorted_segment_prod)
def segment_prod(data: ragged_tensor.RaggedOrDense, segment_ids: ragged_tensor.RaggedOrDense, num_segments, name=None):
    if False:
        for i in range(10):
            print('nop')
    return _ragged_segment_aggregate(math_ops.unsorted_segment_prod, data=data, segment_ids=segment_ids, num_segments=num_segments, name=name or 'RaggedSegmentProd')

@dispatch.dispatch_for_api(math_ops.unsorted_segment_min)
def segment_min(data: ragged_tensor.RaggedOrDense, segment_ids: ragged_tensor.RaggedOrDense, num_segments, name=None):
    if False:
        i = 10
        return i + 15
    return _ragged_segment_aggregate(math_ops.unsorted_segment_min, data=data, segment_ids=segment_ids, num_segments=num_segments, name=name or 'RaggedSegmentMin')

@dispatch.dispatch_for_api(math_ops.unsorted_segment_max)
def segment_max(data: ragged_tensor.RaggedOrDense, segment_ids: ragged_tensor.RaggedOrDense, num_segments, name=None):
    if False:
        print('Hello World!')
    return _ragged_segment_aggregate(math_ops.unsorted_segment_max, data=data, segment_ids=segment_ids, num_segments=num_segments, name=name or 'RaggedSegmentMax')

@dispatch.dispatch_for_api(math_ops.unsorted_segment_mean)
def segment_mean(data: ragged_tensor.RaggedOrDense, segment_ids: ragged_tensor.RaggedOrDense, num_segments, name=None):
    if False:
        print('Hello World!')
    'For docs, see: _RAGGED_SEGMENT_DOCSTRING.'
    with ops.name_scope(name, 'RaggedSegmentMean', [data, segment_ids, num_segments]):
        total = segment_sum(data, segment_ids, num_segments)
        ones = ragged_tensor.RaggedTensor.from_nested_row_splits(array_ops.ones_like(data.flat_values), data.nested_row_splits, validate=False)
        count = segment_sum(ones, segment_ids, num_segments)
        if ragged_tensor.is_ragged(total):
            return total.with_flat_values(total.flat_values / count.flat_values)
        else:
            return total / count

@dispatch.dispatch_for_api(math_ops.unsorted_segment_sqrt_n)
def segment_sqrt_n(data: ragged_tensor.RaggedOrDense, segment_ids: ragged_tensor.RaggedOrDense, num_segments, name=None):
    if False:
        print('Hello World!')
    'For docs, see: _RAGGED_SEGMENT_DOCSTRING.'
    with ops.name_scope(name, 'RaggedSegmentSqrtN', [data, segment_ids, num_segments]):
        total = segment_sum(data, segment_ids, num_segments)
        ones = ragged_tensor.RaggedTensor.from_nested_row_splits(array_ops.ones_like(data.flat_values), data.nested_row_splits, validate=False)
        count = segment_sum(ones, segment_ids, num_segments)
        if ragged_tensor.is_ragged(total):
            return total.with_flat_values(total.flat_values / math_ops.sqrt(count.flat_values))
        else:
            return total / math_ops.sqrt(count)

def _set_ragged_segment_docstring(func, combination, combined):
    if False:
        print('Hello World!')
    func.__doc__ = _RAGGED_SEGMENT_DOCSTRING % dict(combination=combination, combined=combined)
_set_ragged_segment_docstring(segment_sum, 'sum', 'summed')
_set_ragged_segment_docstring(segment_prod, 'product', 'multiplied')
_set_ragged_segment_docstring(segment_min, 'minimum', 'minimized')
_set_ragged_segment_docstring(segment_max, 'maximum', 'maximized')
_set_ragged_segment_docstring(segment_mean, 'mean', 'averaged')
_set_ragged_segment_docstring(segment_sqrt_n, 'sum divided by sqrt(N)', 'summed')
_RAGGED_REDUCE_DOCSTRING = 'Computes the %(combination)s of elements across dimensions of a `RaggedTensor`.\n\n  Reduces `input_tensor` along the dimensions given in `axis` by taking the\n  %(combination)s of values.  If a reduced dimension has no elements for\n  some index, then the value for that index will be %(default)s.\n\n  The rank of the tensor is reduced by `1` for each entry in `axis`.  If\n  `axis` is not specified, then all dimensions are reduced, and a scalar\n  value is returned.\n  Args:\n    input_tensor: A `RaggedTensor` containing the values to be %(combined)s.\n    axis: The dimensions to reduce.  May be `None` (to reduce all axes), an\n      `int` (to reduce a single axis), a `list` or `tuple` of `int` (to reduce\n      a given set of axes), or a `Tensor` with a constant value.  Must be in\n      the range `[0, input_tensor.rank]`.\n    name: A name prefix for the returned tensor (optional).\n  Returns:\n    A `RaggedTensor` containing the %(combined)s values.  The returned tensor\n    has the same dtype as `data`, and its shape is given by removing the\n    dimensions specified in `axis` from `input_tensor.shape`.  The `ragged_rank`\n    of the returned tensor is given by substracting any ragged dimensions\n    specified in `axis` from `input_tensor.ragged_rank`.\n  Raises:\n    ValueError: If `axis` contains a `Tensor` whose value is not constant.\n  ####Example:\n    %(example)s\n'
_RAGGED_REDUCE_SUM_EXAMPLE = '\n    >>> rt = tf.ragged.constant([[3, 1, 4], [1, 5], [9], [2, 6]])\n    >>> tf.reduce_sum(rt, axis=0).numpy()  # = [3+1+9+2, 1+5+6, 4]\n    array([15, 12, 4], dtype=int32)\n    >>> tf.reduce_sum(rt, axis=1).numpy()  # = [3+1+4, 1+5, 9, 2+6]\n    array([8, 6, 9, 8], dtype=int32)\n'
_RAGGED_REDUCE_PROD_EXAMPLE = '\n    >>> rt = tf.ragged.constant([[3, 1, 4], [1, 5], [9], [2, 6]])\n    >>> tf.reduce_prod(rt, axis=0).numpy()  # = [3*1*9*2, 1*5*6, 4]\n    array([54, 30, 4], dtype=int32)\n    >>> tf.reduce_prod(rt, axis=1).numpy()  # = [3*1*4, 1*5, 9, 2*6]\n    array([12, 5, 9, 12], dtype=int32)\n'
_RAGGED_REDUCE_MIN_EXAMPLE = '\n    >>> rt = tf.ragged.constant([[3, 1, 4], [1, 5], [9], [2, 6]])\n    >>> tf.reduce_min(rt, axis=0).numpy()\n    array([1, 1, 4], dtype=int32)\n    >>> tf.reduce_min(rt, axis=1).numpy()\n    array([1, 1, 9, 2], dtype=int32)\n'
_RAGGED_REDUCE_MAX_EXAMPLE = '\n    >>> rt = tf.ragged.constant([[3, 1, 4], [1, 5], [9], [2, 6]])\n    >>> tf.reduce_max(rt, axis=0).numpy()\n    array([9, 6, 4], dtype=int32)\n    >>> tf.reduce_max(rt, axis=1).numpy()\n    array([4, 5, 9, 6], dtype=int32)\n'
_RAGGED_REDUCE_MEAN_EXAMPLE = '\n    >>> rt = tf.ragged.constant([[3, 1, 4], [1, 5], [9], [2, 6]])\n    >>> tf.reduce_mean(rt, axis=0).numpy()\n    array([3.75, 4.  , 4. ])\n    >>> tf.reduce_mean(rt, axis=1).numpy()\n    array([2.66666667, 3.  , 9.  , 4.  ])\n'
_RAGGED_REDUCE_VARIANCE_EXAMPLE = '\n    >>> rt = tf.ragged.constant([[1, 1, 4], [2, 1], [3], [4, 1]],\n    ...                         dtype=tf.float64)\n    >>> tf.math.reduce_variance(rt, axis=0).numpy()\n    array([1.25, 0., 0.])\n    >>> tf.math.reduce_variance(rt, axis=1).numpy()\n    array([2., 0.25, 0., 2.25])\n'
_RAGGED_REDUCE_STD_EXAMPLE = '\n    >>> rt = tf.ragged.constant([[1, 0], [2, 1], [3], [4, 1]],\n    ...                         dtype=tf.float64)\n    >>> tf.math.reduce_std(rt, axis=0).numpy()\n    array([1.11803399, 0.47140452])\n    >>> tf.math.reduce_std(rt, axis=1).numpy()\n    array([0.5, 0.5, 0., 1.5])\n'
_RAGGED_REDUCE_ALL_EXAMPLE = '\n    >>> rt = tf.ragged.constant([[True, True], [True, True, False, True], [False, True]])\n    >>> tf.reduce_all(rt, axis=0).numpy()\n    array([False,  True, False,  True])\n    >>> tf.reduce_all(rt, axis=1).numpy()\n    array([ True, False, False])\n'
_RAGGED_REDUCE_ANY_EXAMPLE = '\n    >>> rt = tf.ragged.constant([[True, True], [True, True, False, True], [False, True]])\n    >>> tf.reduce_any(rt, axis=0).numpy()\n    array([ True,  True, False,  True])\n    >>> tf.reduce_any(rt, axis=1).numpy()\n    array([ True,  True,  True])\n'

def ragged_reduce_aggregate(reduce_op, unsorted_segment_op, rt_input, axis, keepdims, separator=None, name=None):
    if False:
        i = 10
        return i + 15
    'Aggregates across axes of a RaggedTensor using the given `Tensor` ops.\n\n  Reduces `rt_input` along the dimensions given in `axis`.  The rank of the\n  tensor is reduced by 1 for each entry in `axis`.  If `axis` is not specified,\n  then all dimensions are reduced, and a scalar value is returned.\n\n  This op assumes that `reduce_op` and `unsorted_segment_op` are associative;\n  if not, then reducing multiple axes will return incorrect results.  (In\n  particular, reducing multiple axes is currently implemented by reducing the\n  axes one at a time.)\n\n  Args:\n    reduce_op: The tensorflow `op` that should be used to reduce values in\n      uniform dimensions.  Must have the same signature and basic behavior as\n      `reduce_sum`, `reduce_max`, etc.\n    unsorted_segment_op: The tensorflow `op` that should be used to combine\n      values in ragged dimensions.  Must have the same signature and basic\n      behavior as `unsorted_segment_sum`, `unsorted_segment_max`, etc.\n    rt_input: A `Tensor` or `RaggedTensor` containing the values to be reduced.\n    axis: The axis or axes to reduce.  May be `None` (to reduce all axes), an\n      `int` (to reduce a single axis), a `list` or `tuple` of `int` (to reduce a\n      given set of axes), or a `Tensor` with a constant value.  Must be in the\n      range `[0, rt_input.rank)`.\n    keepdims: If true, retains reduced dimensions with length 1.\n    separator: An optional string. Defaults to None. The separator to use when\n      joining. The separator must not be set for non-string data types. (i.e. if\n      separator is not None then it uses string ops)\n    name: A name prefix for the returned tensor (optional).\n\n  Returns:\n    A `RaggedTensor` containing the reduced values.  The returned tensor\n    has the same dtype as `data`, and its shape is given by removing the\n    dimensions specified in `axis` from `rt_input.shape`.  The `ragged_rank`\n    of the returned tensor is given by substracting any ragged dimensions\n    specified in `axis` from `rt_input.ragged_rank`.\n  Raises:\n    ValueError: If `axis` contains a `Tensor` whose value is not constant.\n  '
    if separator is None:
        maybe_separator = {}
    else:
        maybe_separator = {'separator': separator}
    if not ragged_tensor.is_ragged(rt_input):
        return reduce_op(rt_input, axis, keepdims=keepdims, name=name, **maybe_separator)
    if isinstance(axis, tensor.Tensor):
        axis = tensor_util.constant_value(axis)
        if axis is None:
            raise ValueError('axis must be known at graph construction time.')
        if isinstance(axis, np.ndarray):
            axis = axis.tolist()
    if axis is None:
        result = reduce_op(rt_input.flat_values, None, keepdims=keepdims, name=name, **maybe_separator)
        if keepdims:
            for _ in rt_input.shape[1:]:
                result = array_ops.expand_dims(result, axis=0)
        return result
    with ops.name_scope(name, 'RaggedReduce', [rt_input, axis]):
        if isinstance(axis, (tuple, list)):
            if not axis:
                return rt_input
            elif len(axis) == 1:
                axis = axis[0]
            else:
                axis = [array_ops.get_positive_axis(a, rt_input.shape.ndims, 'axis[%s]' % i, 'rank(input_tensor)') for (i, a) in enumerate(axis)]
                axis = sorted(axis)
                inner_reduced = ragged_reduce_aggregate(reduce_op, unsorted_segment_op, rt_input, axis[-1], keepdims, separator)
                return ragged_reduce_aggregate(reduce_op, unsorted_segment_op, inner_reduced, axis[:-1], keepdims, separator)
        rt_input = ragged_tensor.convert_to_tensor_or_ragged_tensor(rt_input, name='rt_input')
        axis = array_ops.get_positive_axis(axis, rt_input.shape.ndims, ndims_name='rank(input_tensor)')
        if axis == 0:
            row_lengths = rt_input.row_splits[1:] - rt_input.row_splits[:-1]
            num_segments = math_ops.maximum(math_ops.reduce_max(row_lengths), 0)
            segment_ids = range(row_lengths).values
            result = _ragged_segment_aggregate(unsorted_segment_op, rt_input.values, segment_ids, num_segments, separator)
            if keepdims:
                result = array_ops.expand_dims(result, axis=0)
            return result
        elif axis == 1:
            num_segments = array_ops.shape(rt_input.row_splits)[0] - 1
            segment_ids = segment_id_ops.row_splits_to_segment_ids(rt_input.row_splits)
            result = _ragged_segment_aggregate(unsorted_segment_op, rt_input.values, segment_ids, num_segments, separator)
            if keepdims:
                result = array_ops.expand_dims(result, axis=1)
            return result
        else:
            return rt_input.with_values(ragged_reduce_aggregate(reduce_op, unsorted_segment_op, rt_input.values, axis - 1, keepdims, separator))

@dispatch.dispatch_for_api(math_ops.reduce_sum)
def reduce_sum(input_tensor: ragged_tensor.Ragged, axis=None, keepdims=None, name=None):
    if False:
        while True:
            i = 10
    'For docs, see: _RAGGED_REDUCE_DOCSTRING.'
    return ragged_reduce_aggregate(reduce_op=math_ops.reduce_sum, unsorted_segment_op=math_ops.unsorted_segment_sum, rt_input=input_tensor, axis=axis, keepdims=keepdims, name=name or 'RaggedReduceSum')

@dispatch.dispatch_for_api(math_ops.reduce_prod)
def reduce_prod(input_tensor: ragged_tensor.Ragged, axis=None, keepdims=None, name=None):
    if False:
        print('Hello World!')
    'For docs, see: _RAGGED_REDUCE_DOCSTRING.'
    return ragged_reduce_aggregate(reduce_op=math_ops.reduce_prod, unsorted_segment_op=math_ops.unsorted_segment_prod, rt_input=input_tensor, axis=axis, keepdims=keepdims, name=name or 'RaggedReduceProd')

@dispatch.dispatch_for_api(math_ops.reduce_min)
def reduce_min(input_tensor: ragged_tensor.Ragged, axis=None, keepdims=None, name=None):
    if False:
        i = 10
        return i + 15
    'For docs, see: _RAGGED_REDUCE_DOCSTRING.'
    return ragged_reduce_aggregate(reduce_op=math_ops.reduce_min, unsorted_segment_op=math_ops.unsorted_segment_min, rt_input=input_tensor, axis=axis, keepdims=keepdims, name=name or 'RaggedReduceMin')

@dispatch.dispatch_for_api(math_ops.reduce_max)
def reduce_max(input_tensor: ragged_tensor.Ragged, axis=None, keepdims=None, name=None):
    if False:
        print('Hello World!')
    'For docs, see: _RAGGED_REDUCE_DOCSTRING.'
    return ragged_reduce_aggregate(reduce_op=math_ops.reduce_max, unsorted_segment_op=math_ops.unsorted_segment_max, rt_input=input_tensor, axis=axis, keepdims=keepdims, name=name or 'RaggedReduceMax')

@dispatch.dispatch_for_api(math_ops.reduce_mean)
def reduce_mean(input_tensor: ragged_tensor.Ragged, axis=None, keepdims=None, name=None):
    if False:
        print('Hello World!')
    'For docs, see: _RAGGED_REDUCE_DOCSTRING.'
    with ops.name_scope(name, 'RaggedReduceMean', [input_tensor, axis]):
        total = reduce_sum(input_tensor, axis, keepdims)
        if ragged_tensor.is_ragged(input_tensor):
            ones = ragged_tensor.RaggedTensor.from_nested_row_splits(array_ops.ones_like(input_tensor.flat_values), input_tensor.nested_row_splits, validate=False)
        else:
            ones = array_ops.ones_like(input_tensor)
        count = reduce_sum(ones, axis, keepdims)
        if ragged_tensor.is_ragged(total):
            return ragged_tensor.RaggedTensor.from_nested_row_splits(total.flat_values / count.flat_values, total.nested_row_splits, validate=False)
        else:
            return total / count

@dispatch.dispatch_for_api(math_ops.reduce_variance)
def reduce_variance(input_tensor: ragged_tensor.Ragged, axis=None, keepdims=False, name=None):
    if False:
        i = 10
        return i + 15
    'For docs, see: _RAGGED_REDUCE_DOCSTRING.'
    with ops.name_scope(name, 'RaggedReduceVariance', [input_tensor, axis]):
        input_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(input_tensor, name='input_tensor')
        if input_tensor.dtype.is_complex:
            raise ValueError('reduce_variance is not supported for RaggedTensors with complex dtypes.')
        square_of_input = math_ops.square(input_tensor)
        mean_of_square = reduce_mean(square_of_input, axis=axis, keepdims=keepdims)
        mean = reduce_mean(input_tensor, axis=axis, keepdims=keepdims)
        square_of_mean = math_ops.square(mean)
        return math_ops.maximum(mean_of_square - square_of_mean, 0)

@dispatch.dispatch_for_api(math_ops.reduce_std)
def reduce_std(input_tensor: ragged_tensor.Ragged, axis=None, keepdims=False, name=None):
    if False:
        while True:
            i = 10
    'For docs, see: _RAGGED_REDUCE_DOCSTRING.'
    with ops.name_scope(name, 'RaggedReduceStd', [input_tensor, axis]):
        variance = reduce_variance(input_tensor, axis=axis, keepdims=keepdims)
        return math_ops.sqrt(variance)

def _cast(input_tensor, dtype):
    if False:
        i = 10
        return i + 15
    return ragged_functional_ops.map_flat_values(math_ops.cast, input_tensor, dtype)

@dispatch.dispatch_for_api(math_ops.reduce_all)
def reduce_all(input_tensor: ragged_tensor.Ragged, axis=None, keepdims=None, name=None):
    if False:
        return 10
    'For docs, see: _RAGGED_REDUCE_DOCSTRING.'
    with ops.name_scope(name, 'RaggedReduceAll', [input_tensor, axis]):
        return _cast(reduce_prod(_cast(input_tensor, dtypes.int32), axis, keepdims), dtypes.bool)

@dispatch.dispatch_for_api(math_ops.reduce_any)
def reduce_any(input_tensor: ragged_tensor.Ragged, axis=None, keepdims=None, name=None):
    if False:
        print('Hello World!')
    'For docs, see: _RAGGED_REDUCE_DOCSTRING.'
    with ops.name_scope(name, 'RaggedReduceAny', [input_tensor, axis]):
        return _cast(reduce_sum(_cast(input_tensor, dtypes.int32), axis, keepdims), dtypes.bool)

def _set_ragged_reduce_docstring(func, combination, combined, default, example):
    if False:
        return 10
    func.__doc__ = _RAGGED_REDUCE_DOCSTRING % dict(combination=combination, combined=combined, default=default, example=example)
_set_ragged_reduce_docstring(reduce_sum, 'sum', 'summed', '0', _RAGGED_REDUCE_SUM_EXAMPLE)
_set_ragged_reduce_docstring(reduce_prod, 'product', 'multiplied', '1', _RAGGED_REDUCE_PROD_EXAMPLE)
_set_ragged_reduce_docstring(reduce_min, 'minimum', 'minimized', '`input_tensor.dtype.min`', _RAGGED_REDUCE_MIN_EXAMPLE)
_set_ragged_reduce_docstring(reduce_max, 'maximum', 'maximized', '`input_tensor.dtype.max`', _RAGGED_REDUCE_MAX_EXAMPLE)
_set_ragged_reduce_docstring(reduce_mean, 'mean', 'averaged', 'NaN', _RAGGED_REDUCE_MEAN_EXAMPLE)
_set_ragged_reduce_docstring(reduce_variance, 'variance', 'averaged', 'NaN', _RAGGED_REDUCE_VARIANCE_EXAMPLE)
_set_ragged_reduce_docstring(reduce_std, 'std', 'averaged', 'NaN', _RAGGED_REDUCE_STD_EXAMPLE)
_set_ragged_reduce_docstring(reduce_all, 'logical and', 'and-ed', 'True', _RAGGED_REDUCE_ALL_EXAMPLE)
_set_ragged_reduce_docstring(reduce_any, 'logical or', 'or-ed', 'False', _RAGGED_REDUCE_ANY_EXAMPLE)

@dispatch.dispatch_for_api(math_ops.matmul)
def matmul(a: ragged_tensor.RaggedOrDense, b: ragged_tensor.RaggedOrDense, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False, output_type=None, name=None):
    if False:
        print('Hello World!')
    'Multiplies matrix `a` by matrix `b`.\n\n  If all transpose or adjoint attributes are `False` then:\n\n  ```\n  output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j]), for all indices i, j.\n  ```\n\n  The inputs `a` and `b` must have `rank >= 2`, where the outermost `rank - 2`\n  dimensions are batch dimensions.  The inputs must have the same dtype.  See\n  `tf.matmul` for more information.\n\n  Args:\n    a: `tf.Tensor` or `RaggedTensor` with `rank > 1`.\n    b: `tf.Tensor` or `RaggedTensor` with same type and rank as `a`.\n    transpose_a: If `True`, `a` is transposed before multiplication.\n    transpose_b: If `True`, `b` is transposed before multiplication.\n    adjoint_a: If `True`, `a` is conjugated & transposed before multiplication.\n    adjoint_b: If `True`, `b` is conjugated & transposed before multiplication.\n    a_is_sparse: If `True`, optimize assuming `a` is mostly zero.\n    b_is_sparse: If `True`, optimize assuming `b` is mostly zero.\n    output_type: The output datatype (optional).\n    name: Name for the operation (optional).\n\n  Returns:\n    A `Tensor` or `RaggedTensor` with the same rank and shape as `a`, where\n    each inner-most matrix is the product of the corresponding matrices in `a`\n    and `b`.\n  '
    if transpose_a and adjoint_a:
        raise ValueError('Only one of transpose_a and adjoint_a can be True.')
    if transpose_b and adjoint_b:
        raise ValueError('Only one of transpose_b and adjoint_b can be True.')
    kwargs = dict(transpose_a=transpose_a, transpose_b=transpose_b, adjoint_a=adjoint_a, adjoint_b=adjoint_b, a_is_sparse=a_is_sparse, b_is_sparse=b_is_sparse, output_type=output_type)
    with ops.name_scope(name, 'RaggedMatMul', [a, b]) as name:
        a = ragged_tensor.convert_to_tensor_or_ragged_tensor(a, name='a')
        b = ragged_tensor.convert_to_tensor_or_ragged_tensor(b, name='b')
        a_is_ragged = isinstance(a, ragged_tensor.RaggedTensor)
        b_is_ragged = isinstance(b, ragged_tensor.RaggedTensor)
        if not (a_is_ragged or b_is_ragged):
            return math_ops.matmul(a, b, **kwargs)
        if a.dtype != b.dtype:
            raise ValueError('`a` and `b` must have the same dtype.')
        if a.shape.rank is None:
            if b.shape.rank is None:
                raise ValueError('matmul requires at least one input to have known rank if either input is ragged.')
            rank = b.shape.rank
        else:
            if b.shape.rank is not None and a.shape.rank != b.shape.rank:
                raise ValueError('`a` and `b` must have the same rank.')
            rank = a.shape.rank
        if rank < 2:
            raise ValueError('`a` and `b` must have the same rank.')
        if rank > 3:
            shape_err = 'Batch dimensions of `a` and `b` do not have the same size.'
            if not a_is_ragged:
                a = ragged_tensor.RaggedTensor.from_tensor(a, ragged_rank=1)
            if not b_is_ragged:
                b = ragged_tensor.RaggedTensor.from_tensor(b, ragged_rank=1)
            with ops.control_dependencies([check_ops.assert_equal(a.row_splits, b.row_splits, message=shape_err)]):
                flat_result = matmul(a.values, b.values, **kwargs)
                return a.with_values(flat_result)
        if rank == 2:
            return _matmul_2d(a, b, **kwargs)
        assert rank == 3
        a_ragged_rank = a.ragged_rank if a_is_ragged else 0
        if a_ragged_rank == 1 and (not (b_is_ragged or transpose_a or adjoint_a)):
            return _matmul_3d_with_batch_dim_folding(a, b, **kwargs)
        else:
            return _matmul_3d_with_map_fn(a, b, **kwargs)

def _matmul_2d(a, b, **kwargs):
    if False:
        while True:
            i = 10
    'Multiplies potentially ragged 2D tensors.\n\n  Args:\n    a: A 2D Tensor or RaggedTensor with `shape=[I, J]`\n    b: A 2D Tensor or RaggedTensor with `shape=[J, K]`\n    **kwargs: Additional arguments for `tf.matmul` (e.g. transpose_a).\n\n  Returns:\n    A 2D Tensor with `shape=[I, K]`.\n  '
    ragged_err = 'The matrices in `a` and `b` may not be ragged in their innermost dimension.'
    checks = []
    if isinstance(a, ragged_tensor.RaggedTensor):
        original_size = array_ops.size(a.flat_values)
        a = a.to_tensor()
        checks.append(check_ops.assert_equal(original_size, array_ops.size(a), message=ragged_err))
    if isinstance(b, ragged_tensor.RaggedTensor):
        original_size = array_ops.size(b.flat_values)
        b = b.to_tensor()
        checks.append(check_ops.assert_equal(original_size, array_ops.size(b), message=ragged_err))
    with ops.control_dependencies(checks):
        return math_ops.matmul(a, b, **kwargs)

def _matmul_3d_with_map_fn(a, b, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Multiplies batches of 2D matrices using map_fn.\n\n  `output[n, i, k]` = sum_j (a[n, i, j] * b[n, j, k])` (for all `n`, `i`, `k`).\n\n  Requires that `a[n, i].nrows()` == `b[n].nrows()` (for all `n` and `i`).\n\n  Args:\n    a: A 3D Tensor or RaggedTensor with `shape=[B, I, J]`, where dimensions `I`\n      and `J` may be ragged.\n    b: A 3D Tensor or RaggedTensor with `shape=[B, J, K]`, where dimensions `J`\n      and `K` may be ragged.\n    **kwargs: Additional arguments for `tf.matmul` (e.g. transpose_a).\n\n  Returns:\n    A 3D RaggedTensor with `shape=[B, (I), (K)]`.\n  '
    if isinstance(b, ragged_tensor.RaggedTensor) and (b.ragged_rank == 2 or kwargs.get('transpose_b') or kwargs.get('adjoint_b')):
        output_ragged_rank = 2
    else:
        output_ragged_rank = 1

    def single_batch_matmul(x):
        if False:
            print('Hello World!')
        out = _matmul_2d(x[0], x[1], **kwargs)
        if output_ragged_rank == 2:
            out = ragged_tensor.RaggedTensor.from_tensor(out)
        return out
    fn_out_shape = None
    row_splits_dtype = a.row_splits.dtype if isinstance(a, ragged_tensor.RaggedTensor) else b.row_splits.dtype
    output_type = kwargs['output_type']
    if output_type is None:
        output_type = a.dtype
    spec = ragged_tensor.RaggedTensorSpec(shape=fn_out_shape, dtype=output_type, ragged_rank=output_ragged_rank - 1, row_splits_dtype=row_splits_dtype)
    result = map_fn.map_fn(single_batch_matmul, elems=(a, b), fn_output_signature=spec)
    if kwargs.get('transpose_a') or kwargs.get('adjoint_a'):
        result._set_shape(a.shape[:-2] + a.shape[-1:] + [None])
    else:
        result._set_shape(a.shape[:-2] + a.shape[-2:-1] + [None])
    if kwargs.get('transpose_b') or kwargs.get('adjoint_b'):
        result._set_shape(b.shape[:-2] + [None] + b.shape[-2:-1])
    else:
        result._set_shape(b.shape[:-2] + [None] + b.shape[-1:])
    return result

def _matmul_3d_with_batch_dim_folding(a, b, **kwargs):
    if False:
        return 10
    'Multiply batches of 2D matrices where only `a.shape[1]` is ragged.\n\n  Args:\n    a: A RaggedTensor with `shape=[B, (I), J]`.  (ragged_rank must be 1.)\n    b: A Tensor with `shape=[B, J, K]`\n    **kwargs: Additional arguments for `tf.matmul` (e.g. transpose_a).\n      transpose_a and adjoint_a must not be true.\n\n  Returns:\n    A RaggedTensor with `shape=[B, (I), K].\n  '
    reshaped_a = array_ops.expand_dims(a.values, 1)
    reshaped_b = array_ops.repeat(b, a.row_lengths(), axis=0)
    flat_result = math_ops.matmul(reshaped_a, reshaped_b, **kwargs)
    return a.with_values(array_ops.squeeze(flat_result, axis=1))

@dispatch.dispatch_for_api(nn_ops.softmax_v2)
def softmax(logits: ragged_tensor.Ragged, axis=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Computes softmax activations.\n\n  Used for multi-class predictions. The sum of all outputs generated by softmax\n  is 1.\n\n  This function performs the equivalent of\n\n      softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)\n\n  Example usage:\n\n  >>> softmax = tf.nn.softmax([-1, 0., 1.])\n  >>> softmax\n  <tf.Tensor: shape=(3,), dtype=float32,\n  numpy=array([0.09003057, 0.24472848, 0.66524094], dtype=float32)>\n  >>> sum(softmax)\n  <tf.Tensor: shape=(), dtype=float32, numpy=1.0>\n\n  Args:\n    logits: A non-empty `Tensor`. Must be one of the following types: `half`,\n      `float32`, `float64`.\n    axis: The dimension softmax would be performed on. The default is -1 which\n      indicates the last dimension.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor`. Has the same type and shape as `logits`.\n\n  Raises:\n    InvalidArgumentError: if `logits` is empty or `axis` is beyond the last\n      dimension of `logits`.\n  '
    if axis is None:
        axis = -1
    with ops.name_scope(name, 'RaggedSoftmax', [logits]) as name:
        max_input = reduce_max(logits, axis=axis, keepdims=True)
        logits_exp = math_ops.exp(math_ops.subtract(logits, max_input))
        denominator = reduce_sum(logits_exp, axis=axis, keepdims=True)
        return math_ops.divide(logits_exp, denominator)

@dispatch.dispatch_for_api(math_ops.add_n)
def add_n(inputs: typing.List[ragged_tensor.RaggedOrDense], name=None):
    if False:
        for i in range(10):
            print('nop')
    'RaggedTensor implementation for tf.math.add_n.'
    if len(inputs) < 0:
        raise ValueError('tf.add_n: expected at least one input.')
    with ops.name_scope(name, 'RaggedAddN', inputs):
        return ragged_functional_ops.map_flat_values(math_ops.add_n, inputs)

@dispatch.dispatch_for_api(nn_ops.dropout)
def dropout_v1(x: ragged_tensor.Ragged, keep_prob=None, noise_shape=None, seed=None, name=None, rate=None):
    if False:
        return 10
    'Ragged dispatch target for tf.nn.dropout.'
    if noise_shape is not None:
        raise ValueError('noise_shape is not supported yet for RaggedTensor x')
    with ops.name_scope(name, 'RaggedNNDropout', [x, rate]):
        x = ragged_tensor.convert_to_tensor_or_ragged_tensor(x, name='x')
        return x.with_flat_values(nn_ops.dropout(x.flat_values, keep_prob=keep_prob, seed=seed, rate=rate))

@dispatch.dispatch_for_api(nn_ops.dropout_v2)
def dropout_v2(x: ragged_tensor.Ragged, rate, noise_shape=None, seed=None, name=None):
    if False:
        return 10
    'Ragged dispatch target for tf.nn.dropout.'
    if noise_shape is not None:
        raise ValueError('noise_shape is not supported yet for RaggedTensor x')
    with ops.name_scope(name, 'RaggedNNDropout', [x, rate]):
        x = ragged_tensor.convert_to_tensor_or_ragged_tensor(x, name='x')
        return x.with_flat_values(nn_ops.dropout_v2(x.flat_values, rate=rate, seed=seed))

@dispatch.dispatch_for_api(nn_ops.stateless_dropout)
def stateless_dropout(x: ragged_tensor.Ragged, rate, seed, rng_alg=None, noise_shape=None, name=None):
    if False:
        while True:
            i = 10
    'Ragged dispatch target for tf.nn.experimental.stateless_dropout.'
    if noise_shape is not None:
        raise ValueError('noise_shape is not supported yet for RaggedTensor x')
    with ops.name_scope(name, 'RaggedNNStatelessDropout', [x, rate]):
        x = ragged_tensor.convert_to_tensor_or_ragged_tensor(x, name='x')
        return x.with_flat_values(nn_ops.stateless_dropout(x.flat_values, rate=rate, seed=seed, rng_alg=rng_alg))

@dispatch.dispatch_for_api(math_ops.tensor_equals)
def tensor_equals(self: ragged_tensor.RaggedOrDense, other: ragged_tensor.RaggedOrDense):
    if False:
        print('Hello World!')
    'Ragged version of the operation invoked by `Tensor.__eq__`.'
    if other is None:
        return False
    elif _use_legacy_mode_for_tensor_equality(self):
        return self is other
    else:
        try:
            return math_ops.equal(self, other)
        except (errors.InvalidArgumentError, ValueError):
            return False

@dispatch.dispatch_for_api(math_ops.tensor_not_equals)
def tensor_not_equals(self: ragged_tensor.RaggedOrDense, other: ragged_tensor.RaggedOrDense):
    if False:
        print('Hello World!')
    'Ragged version of the operation invoked by `Tensor.__ne__`.'
    if other is None:
        return False
    elif _use_legacy_mode_for_tensor_equality(self):
        return self is not other
    else:
        try:
            return math_ops.not_equal(self, other)
        except (errors.InvalidArgumentError, ValueError):
            return True

def _use_legacy_mode_for_tensor_equality(self):
    if False:
        return 10
    g = getattr(self, 'graph', None)
    return not (tensor.Tensor._USE_EQUALITY and ops.executing_eagerly_outside_functions() and (g is None or g.building_function))

def _cumsum_flat_values_at_ragged_rank(last_rp, flat_values, exclusive=False, reverse=False):
    if False:
        print('Hello World!')
    'Calculate flat_values for math_ops.cumsum when axis==ragged_rank.'
    if not exclusive:
        partial = _cumsum_flat_values_at_ragged_rank(last_rp, flat_values, exclusive=True, reverse=reverse)
        return partial + flat_values
    if reverse:
        youngest_sibling = array_ops.gather(params=last_rp.row_splits(), indices=last_rp.value_rowids() + 1) - 1
        new_flat_values = math_ops.cumsum(flat_values, exclusive=True, reverse=True)
        initial_values = array_ops.gather(params=new_flat_values, indices=youngest_sibling)
        return new_flat_values - initial_values
    else:
        eldest_sibling = array_ops.gather(params=last_rp.row_splits(), indices=last_rp.value_rowids())
        new_flat_values = math_ops.cumsum(flat_values, exclusive=True)
        initial_values = array_ops.gather(params=new_flat_values, indices=eldest_sibling)
        return new_flat_values - initial_values

@dispatch.dispatch_for_api(math_ops.cumsum)
def ragged_cumsum(x: ragged_tensor.Ragged, axis: int=0, exclusive: bool=False, reverse: bool=False, name: typing.Optional[str]=None):
    if False:
        for i in range(10):
            print('nop')
    'Calculate math_ops.cumsum for a RaggedTensor.\n\n  Given a ragged tensor `x`, the `result` is a ragged tensor with the same\n  shape. One can calculate the value of `result[i_1...i_k]` as follows:\n  ```\n  dense_result=tf.math.cumsum(rt.to_tensor(), axis=axis, exclusive=exclusive,\n                              reverse=reverse)\n  result[i_1...i_k]=dense_result[i_1...i_k]\n  ```\n\n  Args:\n    x: the original ragged tensor to sum.\n    axis: the axis along which to sum, can range -rank<=axis<rank.\n    exclusive: is the sum exclusive or inclusive? If True, then result[0]=0.\n        If False, then result[0]=x[0].\n    reverse: If True, sum from back to front.\n    name: the name of the op.\n  Returns:\n    the cumulative sum.\n  '
    with ops.name_scope(name, 'RaggedCumSum', [x, axis, exclusive, reverse]):
        axis = array_ops.get_positive_axis(axis, x.shape.rank, ndims_name='rank')
        if axis == x.ragged_rank:
            last_rp = x._nested_row_partitions[-1]
            return x.with_flat_values(_cumsum_flat_values_at_ragged_rank(last_rp, x.flat_values, exclusive=exclusive, reverse=reverse))
        elif axis > x.ragged_rank:
            new_axis = axis - x.ragged_rank
            cumsum_bound = functools.partial(math_ops.cumsum, axis=new_axis, exclusive=exclusive, reverse=reverse)
            return ragged_functional_ops.map_flat_values(cumsum_bound, x)
        else:
            dense_version = x.to_tensor()
            result = math_ops.cumsum(dense_version, axis, exclusive=exclusive, reverse=reverse, name=name)
            return ragged_tensor.RaggedTensor.from_tensor(result, lengths=x.nested_row_lengths())