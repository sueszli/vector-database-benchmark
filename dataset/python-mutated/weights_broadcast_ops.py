"""Weight broadcasting operations.

In `tf.losses` and `tf.metrics`, we support limited weight broadcasting. This
file includes operations for those broadcasting rules.
"""
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sets
from tensorflow.python.util.tf_export import tf_export

def _has_valid_dims(weights_shape, values_shape):
    if False:
        while True:
            i = 10
    with ops.name_scope(None, 'has_invalid_dims', (weights_shape, values_shape)) as scope:
        values_shape_2d = array_ops.expand_dims(values_shape, -1)
        valid_dims = array_ops.concat((values_shape_2d, array_ops.ones_like(values_shape_2d)), axis=1)
        weights_shape_2d = array_ops.expand_dims(weights_shape, -1)
        invalid_dims = sets.set_difference(weights_shape_2d, valid_dims)
        num_invalid_dims = array_ops.size(invalid_dims.values, name='num_invalid_dims')
        return math_ops.equal(0, num_invalid_dims, name=scope)

def _has_valid_nonscalar_shape(weights_rank, weights_shape, values_rank, values_shape):
    if False:
        while True:
            i = 10
    with ops.name_scope(None, 'has_valid_nonscalar_shape', (weights_rank, weights_shape, values_rank, values_shape)) as scope:
        is_same_rank = math_ops.equal(values_rank, weights_rank, name='is_same_rank')
        return cond.cond(is_same_rank, lambda : _has_valid_dims(weights_shape, values_shape), lambda : is_same_rank, name=scope)
_ASSERT_BROADCASTABLE_ERROR_PREFIX = 'weights can not be broadcast to values.'

def assert_broadcastable(weights, values):
    if False:
        i = 10
        return i + 15
    'Asserts `weights` can be broadcast to `values`.\n\n  In `tf.losses` and `tf.metrics`, we support limited weight broadcasting. We\n  let weights be either scalar, or the same rank as the target values, with each\n  dimension either 1, or the same as the corresponding values dimension.\n\n  Args:\n    weights: `Tensor` of weights.\n    values: `Tensor` of values to which weights are applied.\n\n  Returns:\n    `Operation` raising `InvalidArgumentError` if `weights` has incorrect shape.\n    `no_op` if static checks determine `weights` has correct shape.\n\n  Raises:\n    ValueError:  If static checks determine `weights` has incorrect shape.\n  '
    with ops.name_scope(None, 'assert_broadcastable', (weights, values)) as scope:
        with ops.name_scope(None, 'weights', (weights,)) as weights_scope:
            weights = ops.convert_to_tensor(weights, name=weights_scope)
            weights_shape = array_ops.shape(weights, name='shape')
            weights_rank = array_ops.rank(weights, name='rank')
        weights_rank_static = tensor_util.constant_value(weights_rank)
        with ops.name_scope(None, 'values', (values,)) as values_scope:
            values = ops.convert_to_tensor(values, name=values_scope)
            values_shape = array_ops.shape(values, name='shape')
            values_rank = array_ops.rank(values, name='rank')
        values_rank_static = tensor_util.constant_value(values_rank)
        if weights_rank_static is not None and values_rank_static is not None:
            if weights_rank_static == 0:
                return control_flow_ops.no_op(name='static_scalar_check_success')
            if weights_rank_static != values_rank_static:
                raise ValueError(f'{_ASSERT_BROADCASTABLE_ERROR_PREFIX} values.rank={values_rank_static}. weights.rank={weights_rank_static}. values.shape={values.shape}. weights.shape={weights.shape}. Received weights={weights}, values={values}')
            weights_shape_static = tensor_util.constant_value(weights_shape)
            values_shape_static = tensor_util.constant_value(values_shape)
            if weights_shape_static is not None and values_shape_static is not None:
                ndims = len(values_shape_static)
                assert ndims == len(weights_shape_static)
                for i in range(ndims):
                    if weights_shape_static[i] not in (1, values_shape_static[i]):
                        raise ValueError(f'{_ASSERT_BROADCASTABLE_ERROR_PREFIX} Mismatch at dim {i}. values.shape={values_shape_static}, weights.shape={weights_shape_static}. Received weights={weights}, values={values}')
                return control_flow_ops.no_op(name='static_dims_check_success')
        is_scalar = math_ops.equal(0, weights_rank, name='is_scalar')
        data = (_ASSERT_BROADCASTABLE_ERROR_PREFIX, 'weights.shape=', weights.name, weights_shape, 'values.shape=', values.name, values_shape, 'is_scalar=', is_scalar)
        is_valid_shape = cond.cond(is_scalar, lambda : is_scalar, lambda : _has_valid_nonscalar_shape(weights_rank, weights_shape, values_rank, values_shape), name='is_valid_shape')
        return control_flow_assert.Assert(is_valid_shape, data, name=scope)

@tf_export('__internal__.ops.broadcast_weights', v1=[])
def broadcast_weights(weights, values):
    if False:
        i = 10
        return i + 15
    'Broadcast `weights` to the same shape as `values`.\n\n  This returns a version of `weights` following the same broadcast rules as\n  `mul(weights, values)`, but limited to the weights shapes allowed by\n  `assert_broadcastable`. When computing a weighted average, use this function\n  to broadcast `weights` before summing them; e.g.,\n  `reduce_sum(w * v) / reduce_sum(_broadcast_weights(w, v))`.\n\n  Args:\n    weights: `Tensor` whose shape is broadcastable to `values` according to the\n      rules of `assert_broadcastable`.\n    values: `Tensor` of any shape.\n\n  Returns:\n    `weights` broadcast to `values` shape according to the rules of\n      `assert_broadcastable`.\n  '
    with ops.name_scope(None, 'broadcast_weights', (weights, values)) as scope:
        values = ops.convert_to_tensor(values, name='values')
        weights = ops.convert_to_tensor(weights, dtype=values.dtype.base_dtype, name='weights')
        weights_shape = weights.get_shape()
        values_shape = values.get_shape()
        if weights_shape.is_fully_defined() and values_shape.is_fully_defined() and weights_shape.is_compatible_with(values_shape):
            return weights
        if control_flow_ops.get_enclosing_xla_context() is not None:
            return math_ops.multiply(weights, array_ops.ones_like(values), name=scope)
        with ops.control_dependencies((assert_broadcastable(weights, values),)):
            return math_ops.multiply(weights, array_ops.ones_like(values), name=scope)