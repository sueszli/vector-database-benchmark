"""Gather operations for RaggedTensors."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_ragged_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch

@dispatch.dispatch_for_api(array_ops.gather_v2)
def gather(params: ragged_tensor.RaggedOrDense, indices: ragged_tensor.RaggedOrDense, validate_indices=None, axis=None, batch_dims=0, name=None):
    if False:
        for i in range(10):
            print('nop')
    "Gathers ragged slices from `params` axis `0` according to `indices`.\n\n  See `tf.gather` for full documentation.  (This version has the same API\n  as `tf.gather`, but supports ragged `params` and `indices`.)\n\n  Examples:\n\n  >>> params = tf.constant(['a', 'b', 'c', 'd', 'e'])\n  >>> indices = tf.constant([3, 1, 2, 1, 0])\n  >>> ragged_params = tf.ragged.constant([['a', 'b', 'c'], ['d'], [], ['e']])\n  >>> ragged_indices = tf.ragged.constant([[3, 1, 2], [1], [], [0]])\n\n  >>> tf.gather(params, ragged_indices)\n  <tf.RaggedTensor [[b'd', b'b', b'c'], [b'b'], [], [b'a']]>\n\n  >>> tf.gather(ragged_params, indices)\n  <tf.RaggedTensor [[b'e'], [b'd'], [], [b'd'], [b'a', b'b', b'c']]>\n\n  >>> tf.gather(ragged_params, ragged_indices)\n  <tf.RaggedTensor [[[b'e'], [b'd'], []], [[b'd']], [], [[b'a', b'b', b'c']]]>\n\n  Args:\n    params: The potentially ragged tensor from which to gather values. Must be\n      at least rank 1.\n    indices: The potentially ragged tensor indicating which values to gather.\n      Must have dtype `int32` or `int64`.  Values must be in the range `[0,\n      params.shape[0]]`.\n    validate_indices: Ignored.\n    axis: The axis in `params` to gather `indices` from.\n    batch_dims: The number of batch dimensions.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `RaggedTensor`, where `output.dtype=params.dtype` and\n    `output.shape=indices.shape + params.shape[1:]` and\n    `output.ragged_rank=indices.shape.ndims + params.ragged_rank`.\n\n  Raises:\n    ValueError: If indices.shape.ndims is not known statically.\n  "
    del validate_indices
    with ops.name_scope(name, 'RaggedGather', [params, indices]):
        params = ragged_tensor.convert_to_tensor_or_ragged_tensor(params, name='params')
        indices = ragged_tensor.convert_to_tensor_or_ragged_tensor(indices, name='indices')
        (params, indices) = ragged_tensor.match_row_splits_dtypes(params, indices)
        if batch_dims != indices.shape.rank:
            batch_dims = array_ops.get_positive_axis(batch_dims, indices.shape.rank, axis_name='batch_dims', ndims_name='rank(indices)')
        if params.shape.rank is not None and batch_dims >= params.shape.rank:
            raise ValueError('batch_dims must be less than rank(params)')
        if axis is None:
            axis = batch_dims
        axis = array_ops.get_positive_axis(axis, params.shape.rank, ndims_name='rank(params)')
        if axis < batch_dims:
            raise ValueError('axis must be greater than or equal to batch_dims')
        if indices.shape.rank is not None:
            if not 0 <= batch_dims <= indices.shape.rank:
                raise ValueError('batch_dims=%s must be between 0 and rank(indices)=%s' % (batch_dims, indices.shape.rank))
        return _gather(params, indices, axis, batch_dims)

def _gather(params, indices, axis, batch_dims):
    if False:
        return 10
    'Helper that implements the body for ragged gather().\n\n  Assumes that `params` and `indices` have been converted to tensors or\n  ragged tensors, and that `axis` and `batch_dims` have been normalized to\n  be positive.  (So these conversions & normalizations can be skipped in\n  recursive calls to _gather).\n\n  Args:\n    params: The tensor from which to gather values.\n    indices: The indices of values to gather.\n    axis: The axis in `params` to gather `indices` from.\n    batch_dims: The number of batch dimensions.\n\n  Returns:\n    A potentially ragged tensor.\n  '
    params_is_ragged = ragged_tensor.is_ragged(params)
    indices_is_ragged = ragged_tensor.is_ragged(indices)
    if not (params_is_ragged or indices_is_ragged):
        return array_ops.gather(params, indices, axis=axis, batch_dims=batch_dims)
    if batch_dims > 0:
        return _batch_gather(params, indices, axis, batch_dims)
    if axis > 0:
        return _axis_gather(params, indices, axis)
    if indices_is_ragged:
        return indices.with_values(_gather(params, indices.values, 0, 0))
    if indices.shape.ndims is None:
        raise ValueError('rank(indices) must be known statically')
    out_ragged_rank = indices.shape.ndims + len(params.nested_row_splits) - 1
    result = gen_ragged_array_ops.ragged_gather(indices=indices, params_dense_values=params.flat_values, params_nested_splits=params.nested_row_splits, OUTPUT_RAGGED_RANK=out_ragged_rank)
    result = ragged_tensor.RaggedTensor.from_nested_row_splits(result.output_dense_values, result.output_nested_splits, validate=False)
    if indices.shape.ndims > 1:
        target = result
        indices_shape = array_ops.shape(indices, out_type=params.row_splits.dtype)
        shape_cumprod = math_ops.cumprod(indices_shape)
        for dim in range(indices.shape.ndims - 1):
            target._cached_nrows = shape_cumprod[dim]
            target._uniform_row_length = indices_shape[dim + 1]
            target = target.values
    return result

def _batch_gather(params, indices, axis, batch_dims):
    if False:
        while True:
            i = 10
    'Helper that implements the body for ragged gather() when batch_dims>0.\n\n  Args:\n    params: The tensor from which to gather values.\n    indices: The indices of values to gather.\n    axis: The axis in `params` to gather `indices` from.\n    batch_dims: The number of batch dimensions.\n\n  Returns:\n    A potentially ragged tensor.\n  '
    if not params.shape[:batch_dims].is_compatible_with(indices.shape[:batch_dims]):
        raise ValueError('batch shape from indices %s does not match params shape %s' % (indices.shape[:batch_dims], params.shape))
    if batch_dims > 1:
        if not isinstance(params, ragged_tensor.RaggedTensor):
            if indices.uniform_row_length is None:
                raise ValueError('batch shape from indices does not match params shape: ragged indices dimension corresponds to uniform params dimension')
            params = ragged_tensor.RaggedTensor.from_tensor(params, ragged_rank=1, row_splits_dtype=indices.row_splits.dtype)
        if not isinstance(indices, ragged_tensor.RaggedTensor):
            if params.uniform_row_length is None:
                raise ValueError('batch shape from indices does not match params shape: ragged params dimension corresponds to uniform indices dimension')
            indices = ragged_tensor.RaggedTensor.from_tensor(indices, ragged_rank=1, row_splits_dtype=params.row_splits.dtype)
        return params.with_values(_gather(params.values, indices.values, axis - 1, batch_dims - 1))
    if axis > 1:
        if not isinstance(indices, ragged_tensor.RaggedTensor):
            adjusted_indices = params.with_values(array_ops.repeat(indices, params.row_lengths(), 0))
        else:
            if not isinstance(params, ragged_tensor.RaggedTensor):
                params = ragged_tensor.RaggedTensor.from_tensor(params, ragged_rank=1, row_splits_dtype=indices.row_splits.dtype)
            adjusted_indices = _gather(indices, params.with_values(array_ops.repeat(math_ops.range(params.nrows()), params.row_lengths())), 0, 0)
        return _batch_gather(params, adjusted_indices, axis, batch_dims + 1)
    if indices.shape.rank is None:
        raise ValueError('rank(indices) must be known statically')
    assert batch_dims == 1
    flat_params = _flatten_dims_0_and_1(params)
    adjustments = _row_starts(params, indices.dtype)
    adjustments = _increase_rank_to(adjustments, indices.shape.ndims)
    adjusted_indices = indices + adjustments
    return _gather(flat_params, adjusted_indices, axis - 1, 0)

def _axis_gather(params, indices, axis):
    if False:
        i = 10
        return i + 15
    'Helper that implements ragged gather when axis>0 and batch_dims==0.\n\n  Args:\n    params: The tensor from which to gather values.\n    indices: The indices of values to gather.\n    axis: The axis in `params` to gather `indices` from.\n\n  Returns:\n    A potentially ragged tensor.\n  '
    if axis > 1:
        if not isinstance(params, ragged_tensor.RaggedTensor):
            params = ragged_tensor.RaggedTensor.from_tensor(params, ragged_rank=1, row_splits_dtype=indices.row_splits.dtype)
        return params.with_values(_gather(params.values, indices, axis - 1, 0))
    if indices.shape.rank is None:
        raise ValueError('rank(indices) must be known statically')
    assert axis == 1
    flat_params = _flatten_dims_0_and_1(params)
    adjustments = _row_starts(params, indices.dtype)
    adjustments = _increase_rank_to(adjustments, indices.shape.ndims + 1)
    adjusted_indices = indices + adjustments
    return _gather(flat_params, adjusted_indices, axis - 1, 0)

def _flatten_dims_0_and_1(t):
    if False:
        for i in range(10):
            print('nop')
    'Returns a copy of `t` with the outer two dimensions merged.'
    if isinstance(t, ragged_tensor.RaggedTensor):
        return t.values
    else:
        t_shape = array_ops.shape(t)
        return array_ops.reshape(t, array_ops.concat([[-1], t_shape[2:]], axis=0))

def _row_starts(t, dtype):
    if False:
        for i in range(10):
            print('nop')
    'Returns the start indices for the rows in `t`.'
    if isinstance(t, ragged_tensor.RaggedTensor):
        return math_ops.cast(t.row_starts(), dtype)
    else:
        t_shape = array_ops.shape(t, out_type=dtype)
        return math_ops.range(t_shape[0]) * t_shape[1]

def _increase_rank_to(t, rank):
    if False:
        i = 10
        return i + 15
    'Adds *trailing* size-1 dimensions to `t` until it has the given rank.'
    if isinstance(t, ragged_tensor.RaggedTensor):
        return t.with_values(_increase_rank_to(t, rank - 1))
    else:
        old_dims = array_ops.shape(t)
        new_dims = array_ops.ones([rank - array_ops.rank(t)], old_dims.dtype)
        new_shape = array_ops.concat([old_dims, new_dims], axis=0)
        return array_ops.reshape(t, new_shape)

@dispatch.dispatch_for_api(array_ops.gather)
def _ragged_gather_v1(params: ragged_tensor.RaggedOrDense, indices: ragged_tensor.RaggedOrDense, validate_indices=None, name=None, axis=0, batch_dims=0):
    if False:
        while True:
            i = 10
    return gather(params, indices, validate_indices, axis, batch_dims, name)

@dispatch.dispatch_for_api(array_ops.gather_nd_v2)
def gather_nd(params: ragged_tensor.RaggedOrDense, indices: ragged_tensor.RaggedOrDense, batch_dims=0, name=None):
    if False:
        while True:
            i = 10
    "Gather slices from `params` using `n`-dimensional indices.\n\n  This operation is similar to `gather`, but it uses the innermost dimension\n  of `indices` to define a slice into `params`.  In particular, if:\n\n  * `indices` has shape `[A1...AN, I]`\n  * `params` has shape `[B1...BM]`\n\n  Then:\n\n  * `result` has shape `[A1...AN, B_{I+1}...BM]`.\n  * `result[a1...aN] = params[indices[a1...aN, :]]`\n\n  Args:\n    params: A potentially ragged tensor with shape `[A1...AN, I]`.\n    indices: A potentially ragged tensor with shape `[B1...BM]`.\n    batch_dims: Must be zero.\n    name: A name for the operation (optional).\n\n  Returns:\n    A potentially ragged tensor with shape `[A1...AN, B_{I+1}...BM]`.\n\n  #### Examples:\n\n  >>> params = tf.ragged.constant(\n  ...     [ [ ['000', '001'], ['010'              ]          ],\n  ...       [ ['100'       ], ['110', '111', '112'], ['120'] ],\n  ...       [ [            ], ['210'              ]          ] ])\n\n  >>> # Gather 2D slices from a 3D tensor\n  >>> tf.gather_nd(params, [[2], [0]])\n  <tf.RaggedTensor [[[], [b'210']], [[b'000', b'001'], [b'010']]]>\n\n  >>> # Gather 1D slices from a 3D tensor\n  >>> tf.gather_nd(params, [[2, 1], [0, 0]])\n  <tf.RaggedTensor [[b'210'], [b'000', b'001']]>\n\n  >>> # Gather scalars from a 3D tensor\n  >>> tf.gather_nd(params, [[0, 0, 1], [1, 1, 2]]).numpy()\n  array([b'001', b'112'], dtype=object)\n  "
    if not isinstance(batch_dims, int) or batch_dims != 0:
        raise ValueError('batch_dims != 0 is not supported for ragged gather yet.')
    if not (ragged_tensor.is_ragged(params) or ragged_tensor.is_ragged(indices)):
        return array_ops.gather_nd(params, indices, name)
    with ops.name_scope(name, 'RaggedGatherNd', [params, indices]):
        params = ragged_tensor.convert_to_tensor_or_ragged_tensor(params, name='params')
        indices = ragged_tensor.convert_to_tensor_or_ragged_tensor(indices, name='indices')
        (params, indices) = ragged_tensor.match_row_splits_dtypes(params, indices)
        indices_shape = indices.shape
        indices_ndims = indices_shape.ndims
        if indices_ndims is None:
            raise ValueError('indices.rank be statically known.')
        if indices_ndims == 0:
            raise ValueError('indices.rank must be at least 1.')
        if ragged_tensor.is_ragged(indices) and indices_ndims == indices.ragged_rank + 1:
            raise ValueError('The innermost dimension of indices may not be ragged')
        index_size = tensor_shape.dimension_value(indices_shape[-1])
        if index_size is None:
            raise ValueError('indices.shape[-1] must be statically known.')
        if indices_ndims > 2:
            indices_is_dense = not ragged_tensor.is_ragged(indices)
            if indices_is_dense:
                indices = ragged_tensor.RaggedTensor.from_tensor(indices, ragged_rank=indices_ndims - 2, row_splits_dtype=params.row_splits.dtype)
            result = indices.with_flat_values(gather_nd(params, indices.flat_values))
            if indices_is_dense and ragged_tensor.is_ragged(result) and (result.ragged_rank == indices_ndims - 2):
                result = ragged_tensor.RaggedTensor.to_tensor(result)
            return result
        assert not ragged_tensor.is_ragged(indices)
        assert ragged_tensor.is_ragged(params)
        if index_size == 0:
            params_ndims = params.ragged_rank + array_ops.rank(params.flat_values)
            for dim in range(indices_ndims - 1):
                params = ragged_array_ops.expand_dims(params, axis=0)
            multiples = array_ops.concat([array_ops.shape(indices)[:-1], array_ops.ones([params_ndims], dtypes.int32)], axis=0)
            return ragged_array_ops.tile(params, multiples)
        elif index_size == 1:
            flattened_index_tuples = array_ops.reshape(indices, [-1])
            return gather(params, flattened_index_tuples)
        else:
            indices = math_ops.cast(indices, params.row_splits.dtype)
            flattened_index_tuples = array_ops.gather(params.row_splits, indices[..., 0])
            flattened_index_tuples += indices[..., 1]
            flattened_params = params.values
            for dim in range(2, index_size):
                if not ragged_tensor.is_ragged(flattened_params):
                    flattened_index_tuples = array_ops.expand_dims(flattened_index_tuples, axis=1)
                    flattened_index_tuples = array_ops.concat([flattened_index_tuples, indices[..., dim:]], axis=1)
                    return array_ops.gather_nd(flattened_params, flattened_index_tuples)
                flattened_index_tuples = array_ops.gather(flattened_params.row_starts(), flattened_index_tuples)
                flattened_index_tuples += indices[..., dim]
                flattened_params = flattened_params.values
            return gather(flattened_params, flattened_index_tuples)

@dispatch.dispatch_for_api(array_ops.gather_nd)
def _ragged_gather_nd_v1(params: ragged_tensor.RaggedOrDense, indices: ragged_tensor.RaggedOrDense, name=None, batch_dims=0):
    if False:
        return 10
    return gather_nd(params, indices, batch_dims, name)

@ops.RegisterGradient('RaggedGather')
def _ragged_gather_grad(op, *grads):
    if False:
        i = 10
        return i + 15
    'Gradient for RaggedGather op.'
    param_nested_splits = op.inputs[:-2]
    param_inner_values = op.inputs[-2]
    indices = op.inputs[-1]
    grad_inner_values = grads[-1]
    combined_splits = param_nested_splits[0]
    for row_splits in param_nested_splits[1:]:
        combined_splits = array_ops.gather(row_splits, combined_splits)
    flat_indices = array_ops.reshape(indices, [-1])
    grad_indices = ragged_math_ops.range(array_ops.gather(combined_splits, flat_indices), array_ops.gather(combined_splits[1:], flat_indices)).values
    param_inner_values_grad = indexed_slices.IndexedSlices(values=grad_inner_values, indices=grad_indices, dense_shape=array_ops.shape(param_inner_values))
    return [None for _ in param_nested_splits] + [param_inner_values_grad, None]