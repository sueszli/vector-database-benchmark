"""Adaptive Gauss-Kronrod quadrature algorithm for numeric integration."""
from typing import Callable, Optional
import tensorflow.compat.v2 as tf
from tf_quant_finance import types
from tf_quant_finance import utils
from tf_quant_finance.math.integration import adaptive_update
from tf_quant_finance.math.integration import gauss_constants

def _non_adaptive_gauss_kronrod(func: Callable[[types.FloatTensor], types.FloatTensor], lower: types.FloatTensor, upper: types.FloatTensor, num_points: int=15, dtype: Optional[tf.DType]=None, name: Optional[str]=None) -> (types.FloatTensor, types.FloatTensor):
    if False:
        while True:
            i = 10
    "Evaluates definite integral using non-adaptive Gauss-Kronrod quadrature.\n\n  Integrates `func` using non-adaptive Gauss-Kronrod quadrature [1].\n\n  Applies change of variables to the function to obtain the [-1,1] integration\n  interval.\n  Takes the sum of values obtained from evaluating the new function at points\n  given by the roots of the Legendre polynomial of degree `(num_points-1)//2`\n  and the roots of the Stieltjes polynomial of degree `(num_points+1)//2`,\n  multiplied with corresponding precalculated coefficients.\n\n  #### References\n  [1] https://en.wikipedia.org/wiki/Gauss%E2%80%93Kronrod_quadrature_formula\n\n  #### Example\n  ```python\n    f = lambda x: x*x\n    a = tf.constant([0.0])\n    b = tf.constant([3.0])\n    num_points = 21\n    _non_adaptive_gauss_kronrod(f, a, b, num_points) # [9.0]\n  ```\n\n  Args:\n    func: Represents a function to be integrated. It must be a callable of a\n      single `Tensor` parameter and return a `Tensor` of the same shape and\n      dtype as its input. It will be called with a `Tensor` of shape\n      `lower.shape + [n]` (where n is integer number of points) and of the same\n      `dtype` as `lower`.\n    lower: Represents the lower limits of integration. `func` will be integrated\n      between each pair of points defined by `lower` and `upper`.\n    upper: Same shape and dtype as `lower` representing the upper limits of\n      intergation.\n    num_points: Number of points at which the function `func` will be evaluated.\n      Implemented for 15,21,31.\n      Default value: 15.\n    dtype: If supplied, the dtype for the `lower` and `upper`. Result will have\n      the same dtype. Default value: None which maps to dtype of `lower`.\n    name: The name to give to the ops created by this function. Default value:\n      None which maps to 'non_adaptive_gauss_kronrod'.\n\n  Returns:\n    A tuple:\n      * `Tensor` of shape `batch_shape`, containing value of the definite\n      integral,\n      * `Tensor` of shape `batch_shape + [legendre_num_points]`, containing\n      values of the function evaluated at the Legendre polynomial root points.\n  "
    with tf.name_scope(name=name or 'non_adaptive_gauss_kronrod'):
        lower = tf.convert_to_tensor(lower, dtype=dtype, name='lower')
        dtype = lower.dtype
        upper = tf.convert_to_tensor(upper, dtype=dtype, name='upper')
        legendre_num_points = (num_points - 1) // 2
        legendre_roots = gauss_constants.legendre_roots.get(legendre_num_points, None)
        stieltjes_roots = gauss_constants.stieltjes_roots.get(num_points, None)
        if legendre_roots is None:
            raise ValueError(f'Unsupported value for `num_points`: {num_points}')
        if stieltjes_roots is None:
            raise ValueError(f'Unsupported value for `num_points`: {num_points}')
        lower = tf.expand_dims(lower, -1)
        upper = tf.expand_dims(upper, -1)
        roots = legendre_roots + stieltjes_roots
        roots = tf.constant(roots, dtype=dtype)
        grid = ((upper - lower) * roots + upper + lower) / 2
        func_results = func(grid)
        weights = gauss_constants.kronrod_weights.get(num_points, None)
        result = tf.reduce_sum(func_results * (upper - lower) * weights / 2, axis=-1)
        return (result, func_results)

def gauss_kronrod(func: Callable[[types.FloatTensor], types.FloatTensor], lower: types.FloatTensor, upper: types.FloatTensor, tolerance: float, num_points: int=21, max_depth: int=20, dtype: Optional[tf.DType]=None, name: Optional[str]=None) -> types.FloatTensor:
    if False:
        while True:
            i = 10
    "Evaluates definite integral using adaptive Gauss-Kronrod quadrature.\n\n  Integrates `func` using adaptive Gauss-Kronrod quadrature [1].\n\n  Applies change of variables to the function to obtain the [-1,1] integration\n  interval.\n  Takes the sum of values obtained from evaluating the new function at points\n  given by the roots of the Legendre polynomial of degree `(num_points-1)//2`\n  and the roots of the Stieltjes polynomial of degree `(num_points+1)//2`,\n  multiplied with corresponding precalculated coefficients.\n  Repeats procedure if not accurate enough by halving the intervals and dividing\n  these into the same number of subintervals.\n\n  #### References\n  [1] https://en.wikipedia.org/wiki/Gauss%E2%80%93Kronrod_quadrature_formula\n\n  #### Example\n  ```python\n    f = lambda x: x*x\n    a = tf.constant([0.0])\n    b = tf.constant([3.0])\n    tol = 1e-5\n    num_points = 21\n    max_depth = 10\n    gauss_kronrod(f, a, b, tol, num_points, max_depth) # [9.0]\n  ```\n\n  Args:\n    func: Represents a function to be integrated. It must be a callable of a\n      single `Tensor` parameter and return a `Tensor` of the same shape and\n      dtype as its input. It will be called with a `Tensor` of shape\n      `lower.shape + [n,  num_points]` (where `n` is defined by the algorithm\n      and represents the number of subintervals) and of the same `dtype` as\n      `lower`.\n    lower: Represents the lower limits of integration. `func` will be integrated\n      between each pair of points defined by `lower` and `upper`. Must be a\n      1-dimensional tensor of shape `[batch_dim]`.\n    upper: Same shape and dtype as `lower` representing the upper limits of\n      intergation.\n    tolerance: Represents the tolerance for the estimated error of the integral\n      estimation, at which to stop further dividing the intervals.\n    num_points: Number of points at which the function `func` will be evaluated.\n      Implemented for 15,21,31. Default value: 21.\n    max_depth: Maximum number of times to divide intervals into two parts and\n      recalculate Gauss-Kronrod on them. Default value: 20.\n    dtype: If supplied, the dtype for the `lower` and `upper`. Result will have\n      the same dtype. Default value: None which maps to dtype of `lower`.\n    name: The name to give to the ops created by this function. Default value:\n      None which maps to 'gauss_kronrod'.\n\n  Returns:\n    `Tensor` of shape `[batch_dim]`, containing value of the definite integral.\n  "
    with tf.name_scope(name=name or 'gauss_kronrod'):
        lower = tf.convert_to_tensor(lower, dtype=dtype, name='lower')
        dtype = lower.dtype
        upper = tf.convert_to_tensor(upper, dtype=dtype, name='upper')
        legendre_num_points = (num_points - 1) // 2

        def cond(lower, upper, sum_estimates):
            if False:
                for i in range(10):
                    print('nop')
            del upper, sum_estimates
            return tf.size(lower) > 0

        def body(lower, upper, sum_estimates):
            if False:
                for i in range(10):
                    print('nop')
            (kronrod_result, func_results) = _non_adaptive_gauss_kronrod(func, lower, upper, num_points, dtype, name)
            legendre_func_results = func_results[..., :legendre_num_points]
            legendre_weights = tf.constant(gauss_constants.legendre_weights[legendre_num_points], dtype=dtype)
            lower_exp = tf.expand_dims(lower, -1)
            upper_exp = tf.expand_dims(upper, -1)
            legendre_result = tf.reduce_sum(legendre_func_results * (upper_exp - lower_exp) * legendre_weights / 2, axis=-1)
            error = tf.abs(kronrod_result - legendre_result)
            (new_lower, new_upper, sum_good_estimates) = adaptive_update.update(lower, upper, kronrod_result, error, tolerance, dtype)
            sum_estimates += sum_good_estimates
            return (new_lower, new_upper, sum_estimates)
        sum_estimates = tf.zeros_like(lower, dtype=dtype)
        lower = tf.expand_dims(lower, -1)
        upper = tf.expand_dims(upper, -1)
        loop_vars = (lower, upper, sum_estimates)
        (lower, upper) = utils.broadcast_tensors(lower, upper)
        batch_shape = lower.shape[:-1]
        (_, _, estimate_result) = tf.while_loop(cond=cond, body=body, loop_vars=loop_vars, maximum_iterations=max_depth, shape_invariants=(tf.TensorShape(batch_shape + [None]), tf.TensorShape(batch_shape + [None]), tf.TensorShape(batch_shape)))
        return estimate_result