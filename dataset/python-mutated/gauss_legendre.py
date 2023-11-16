"""Gauss-Legendre quadrature algorithm for numeric integration."""
from typing import Callable, Optional
import tensorflow.compat.v2 as tf
from tf_quant_finance import types
from tf_quant_finance.math.integration import gauss_constants

def gauss_legendre(func: Callable[[types.FloatTensor], types.FloatTensor], lower: types.FloatTensor, upper: types.FloatTensor, num_points: int=32, dtype: Optional[tf.DType]=None, name: Optional[str]=None) -> types.FloatTensor:
    if False:
        while True:
            i = 10
    "Evaluates definite integral using Gauss-Legendre quadrature.\n\n  Integrates `func` using Gauss-Legendre quadrature [1].\n\n  Applies change of variables to the function to obtain the [-1,1] integration\n  interval.\n  Takes the sum of values obtained from evaluating the new function at points\n  given by the roots of the Legendre polynomial of degree `num_points`,\n  multiplied with corresponding precalculated coefficients.\n\n  #### References\n  [1] https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature\n\n  #### Example\n  ```python\n    f = lambda x: x*x\n    a = tf.constant(0.0)\n    b = tf.constant(3.0)\n    gauss_legendre(f, a, b, num_points=15) # 9.0\n  ```\n\n  Args:\n    func: Represents a function to be integrated. It must be a callable of a\n      single `Tensor` parameter and return a `Tensor` of the same shape and\n      dtype as its input. It will be called with a `Tensor` of shape\n      `lower.shape + [n]` (where n is integer number of points) and of the same\n      `dtype` as `lower`.\n    lower: Represents the lower limits of integration. `func` will be integrated\n      between each pair of points defined by `lower` and `upper`.\n    upper: Same shape and dtype as `lower` representing the upper limits of\n      intergation.\n    num_points: Number of points at which the function `func` will be evaluated.\n      Implemented for 2-15,20,32.\n      Default value: 32.\n    dtype: If supplied, the dtype for the `lower` and `upper`. Result will have\n      the same dtype.\n      Default value: None which maps to dtype of `lower`.\n    name: The name to give to the ops created by this function.\n      Default value: None which maps to 'gauss_legendre'.\n\n  Returns:\n    `Tensor` of shape `func_batch_shape + limits_batch_shape`, containing\n      value of the definite integral.\n\n  "
    with tf.name_scope(name=name or 'gauss_legendre'):
        lower = tf.convert_to_tensor(lower, dtype=dtype, name='lower')
        dtype = lower.dtype
        upper = tf.convert_to_tensor(upper, dtype=dtype, name='upper')
        roots = gauss_constants.legendre_roots.get(num_points, None)
        if roots is None:
            raise ValueError(f'Unsupported value for `num_points`: {num_points}')
        coefficients = gauss_constants.legendre_weights
        lower = tf.expand_dims(lower, -1)
        upper = tf.expand_dims(upper, -1)
        roots = tf.constant(roots, dtype=dtype)
        grid = ((upper - lower) * roots + upper + lower) / 2
        weights = tf.constant(coefficients[num_points], dtype=dtype)
        result = tf.reduce_sum(func(grid) * (upper - lower) * weights / 2, axis=-1)
        return result