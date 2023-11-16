"""Composite Simpson's algorithm for numeric integration."""
from typing import Callable, Optional
import tensorflow.compat.v2 as tf
from tf_quant_finance import types

def simpson(func: Callable[[types.FloatTensor], types.FloatTensor], lower: types.FloatTensor, upper: types.FloatTensor, num_points: types.IntTensor=1001, dtype: Optional[tf.DType]=None, name: Optional[str]=None) -> types.FloatTensor:
    if False:
        print('Hello World!')
    'Evaluates definite integral using composite Simpson\'s 1/3 rule.\n\n  Integrates `func` using composite Simpson\'s 1/3 rule [1].\n\n  Evaluates function at points of evenly spaced grid of `num_points` points,\n  then uses obtained values to interpolate `func` with quadratic polynomials\n  and integrates these polynomials.\n\n  #### References\n  [1] Weisstein, Eric W. "Simpson\'s Rule." From MathWorld - A Wolfram Web\n      Resource. http://mathworld.wolfram.com/SimpsonsRule.html\n\n  #### Example\n  ```python\n    f = lambda x: x*x\n    a = tf.constant(0.0)\n    b = tf.constant(3.0)\n    simpson(f, a, b, num_points=1001) # 9.0\n  ```\n\n  Args:\n    func: Represents a function to be integrated. It must be a callable of a\n      single `Tensor` parameter and return a `Tensor` of the same shape and\n      dtype as its input. It will be called with a `Tensor` of shape\n      `lower.shape + [n]` (where n is integer number of points) and of the same\n      `dtype` as `lower`.\n    lower: Represents the lower limits of integration. `func` will be integrated\n      between each pair of points defined by `lower` and `upper`.\n    upper: Same shape and dtype as `lower` representing the upper limits of\n      intergation.\n    num_points: Number of points at which function `func` will be evaluated.\n      Must be odd and at least 3. Default value: 1001.\n    dtype: If supplied, the dtype for the `lower` and `upper`. Result will have\n      the same dtype.\n      Default value: None which maps to dtype of `lower`.\n    name: The name to give to the ops created by this function.\n      Default value: None which maps to \'integrate_simpson_composite\'.\n\n  Returns:\n    `Tensor` of shape `func_batch_shape + limits_batch_shape`, containing\n      value of the definite integral.\n\n  '
    with tf.compat.v1.name_scope(name, default_name='integrate_simpson_composite', values=[lower, upper]):
        lower = tf.convert_to_tensor(lower, dtype=dtype, name='lower')
        dtype = lower.dtype
        upper = tf.convert_to_tensor(upper, dtype=dtype, name='upper')
        num_points = tf.convert_to_tensor(num_points, dtype=tf.int32, name='num_points')
        assertions = [tf.debugging.assert_greater_equal(num_points, 3), tf.debugging.assert_equal(num_points % 2, 1)]
        with tf.compat.v1.control_dependencies(assertions):
            dx = (upper - lower) / (tf.cast(num_points, dtype=dtype) - 1)
            dx_expand = tf.expand_dims(dx, -1)
            lower_exp = tf.expand_dims(lower, -1)
            grid = lower_exp + dx_expand * tf.cast(tf.range(num_points), dtype=dtype)
            weights_first = tf.constant([1.0], dtype=dtype)
            weights_mid = tf.tile(tf.constant([4.0, 2.0], dtype=dtype), [(num_points - 3) // 2])
            weights_last = tf.constant([4.0, 1.0], dtype=dtype)
            weights = tf.concat([weights_first, weights_mid, weights_last], axis=0)
        return tf.reduce_sum(func(grid) * weights, axis=-1) * dx / 3