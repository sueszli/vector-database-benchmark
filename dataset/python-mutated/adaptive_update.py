"""Update function for intervals in adaptive numeric integration."""
from typing import Optional
import tensorflow.compat.v2 as tf
from tf_quant_finance import types

def update(lower: types.FloatTensor, upper: types.FloatTensor, estimate: types.FloatTensor, error: types.FloatTensor, tolerance: float, dtype: Optional[tf.DType]=None, name: Optional[str]=None) -> (types.FloatTensor, types.FloatTensor, types.FloatTensor):
    if False:
        i = 10
        return i + 15
    "Calculates new values for the limits for any adaptive quadrature.\n\n  Checks which intervals have estimated results that are within the provided\n  tolerance. The values for these intervals are added to the sum of good\n  estimations. The other intervals get divided in half.\n\n  #### Example\n  ```python\n    l = tf.constant([[[0.0], [1.0]]])\n    u = tf.constant([[[1.0], [2.0]]])\n    estimate = tf.constant([[[3.0], [4.0]]])\n    err = tf.constant([[[0.01], [0.02]]])\n    tol = 0.004\n    update(l, u, estimate, err, tol)\n    # tf.constant([[1.0, 1.5]]), tf.constant([[1.5, 2.0]]), tf.constant([3.0])\n  ```\n\n  Args:\n    lower: Represents the lower limits of integration. Must be a 2-dimensional\n      tensor of shape `[batch_dim, n]` (where `n` is defined by the algorithm\n      and represents the number of subintervals).\n    upper: Same shape and dtype as `lower` representing the upper limits of\n      intergation.\n    estimate: Same shape and dtype as `lower` representing the integration\n      results calculated with some quadrature method for the corresponding\n      limits.\n    error: Same shape and dtype as `lower` representing the estimated\n      integration error for corresponding `estimate` values.\n    tolerance: Represents the tolerance for the estimated error of the integral\n      estimation, at which to stop further dividing the intervals.\n    dtype: If supplied, the dtype for the `lower` and `upper`. Result will have\n      the same dtype. Default value: None which maps to dtype of `lower`.\n    name: The name to give to the ops created by this function. Default value:\n      None which maps to 'adaptive_update'.\n\n  Returns:\n    A tuple:\n      * `Tensor` of shape `[batch_dim, new_n]`, containing values of the new\n      lower limits,\n      * `Tensor` of shape `[batch_dim, new_n]`, containing values of the new\n      upper limits,\n      * `Tensor` of shape `[batch_dim]`, containing sum values of the quadrature\n      method results of the good intervals.\n  "
    with tf.name_scope(name=name or 'adaptive_update'):
        lower = tf.convert_to_tensor(lower, dtype=dtype, name='lower')
        dtype = lower.dtype
        upper = tf.convert_to_tensor(upper, dtype=dtype, name='upper')
        relative_error = error / estimate
        condition = relative_error > tolerance
        num_bad_sub_intervals = tf.reduce_max(tf.math.count_nonzero(condition, axis=1, dtype=tf.int32), axis=0)
        indices = tf.math.top_k(relative_error, k=num_bad_sub_intervals, sorted=False).indices
        sum_all = tf.reduce_sum(estimate, axis=-1)
        sum_bad = tf.reduce_sum(tf.gather(estimate, indices, batch_dims=-1), axis=-1)
        sum_goods = sum_all - sum_bad
        filtered_lower = tf.gather(lower, indices, batch_dims=-1)
        filtered_upper = tf.gather(upper, indices, batch_dims=-1)
        mid_points = (filtered_lower + filtered_upper) / 2
        new_lower = tf.concat([filtered_lower, mid_points], axis=-1)
        new_upper = tf.concat([mid_points, filtered_upper], axis=-1)
        return (new_lower, new_upper, sum_goods)