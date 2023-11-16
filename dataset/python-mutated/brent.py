"""Root search functions."""
from typing import Callable
import tensorflow.compat.v2 as tf
from tf_quant_finance import types
from tf_quant_finance import utils as tff_utils
from tf_quant_finance.math.root_search import utils
__all__ = ['BrentResults', 'brentq']

@tff_utils.dataclass
class BrentResults:
    """Brent root search results.

  Attributes:
    estimated_root: A `Tensor` containing the best estimate. If the search was
      successful, this estimate is a root of the objective function.
    objective_at_estimated_root: A `Tensor` containing the value of the
      objective function at  the best estimate. If the search was successful,
      then this is close to 0.
    num_iterations: A `Tensor` containing number of iterations performed for
      each pair of starting points
    converged: A boolean `Tensor` indicating whether the best estimate is a root
      within the tolerance specified for the search.
  """
    estimated_root: types.RealTensor
    objective_at_estimated_root: types.RealTensor
    num_iterations: types.IntTensor
    converged: types.BoolTensor

@tff_utils.dataclass
class _BrentSearchConstants:
    """Values which remain fixed across all root searches."""
    false: types.BoolTensor
    zero: types.RealTensor
    zero_value: types.RealTensor

@tff_utils.dataclass
class _BrentSearchState:
    """Values which are updated during the root search."""
    best_estimate: types.RealTensor
    value_at_best_estimate: types.RealTensor
    last_estimate: types.RealTensor
    value_at_last_estimate: types.RealTensor
    contrapoint: types.RealTensor
    value_at_contrapoint: types.RealTensor
    step_to_best_estimate: types.RealTensor
    step_to_last_estimate: types.RealTensor
    num_iterations: types.IntTensor
    finished: types.BoolTensor

@tff_utils.dataclass
class _BrentSearchParams:
    """Values which remain fixed for a given root search."""
    objective_fn: Callable[[types.BoolTensor], types.BoolTensor]
    max_iterations: types.IntTensor
    absolute_root_tolerance: types.RealTensor
    relative_root_tolerance: types.RealTensor
    function_tolerance: types.RealTensor
    stopping_policy_fn: Callable[[types.BoolTensor], types.BoolTensor]

def _swap_where(condition, x, y):
    if False:
        print('Hello World!')
    'Swaps the elements of `x` and `y` based on `condition`.\n\n  Args:\n    condition: A `Tensor` of dtype bool.\n    x: A `Tensor` with the same shape as `condition`.\n    y: A `Tensor` with the same shape and dtype as `x`.\n\n  Returns:\n    Two `Tensors` with the same shape as `x` and `y`.\n  '
    return (tf.where(condition, y, x), tf.where(condition, x, y))

def _secant_step(x1, x2, y1, y2):
    if False:
        for i in range(10):
            print('nop')
    'Returns the step size at the current position if using the secant method.\n\n  This function is meant for exclusive use by the `_brent_loop_body` function:\n  - It does not guard against divisions by zero, and instead assumes that `y1`\n    is distinct from `y2`. The `_brent_loop_body` function guarantees this\n    property.\n  - It does not guard against overflows which may occur if the difference\n    between `y1` and `y2` is small while that between `x1` and `x2` is not.\n    In this case, the resulting step size will be larger than `bisection_step`\n    and thus ignored by the `_brent_loop_body` function.\n\n  Args:\n    x1: `Tensor` containing the current position.\n    x2: `Tensor` containing the previous position.\n    y1: `Tensor` containing the value of `objective_fn` at `x1`.\n    y2: `Tensor` containing the value of `objective_fn` at `x2`.\n\n  Returns:\n    A `Tensor` with the same shape and dtype as `current`.\n  '
    x_difference = x1 - x2
    y_difference = y1 - y2
    return -y1 * x_difference / y_difference

def _quadratic_interpolation_step(x1, x2, x3, y1, y2, y3):
    if False:
        print('Hello World!')
    'Returns the step size to use when using quadratic interpolation.\n\n  This function is meant for exclusive use by the `_brent_loop_body` function.\n  It does not guard against divisions by zero, and instead assumes that `y1` is\n  distinct from `y2` and `y3`. The `_brent_loop_body` function guarantees this\n  property.\n\n  Args:\n    x1: `Tensor` of any shape and real dtype containing the first position used\n      for extrapolation.\n    x2: `Tensor` of the same shape and dtype as `x1` containing the second\n      position used for extrapolation.\n    x3: `Tensor` of the same shape and dtype as `x1` containing the third\n      position used for extrapolation.\n    y1: `Tensor` containing the value of the interpolated function at `x1`.\n    y2: `Tensor` containing the value of interpolated function at `x2`.\n    y3: `Tensor` containing the value of interpolated function at `x3`.\n\n  Returns:\n    A `Tensor` with the same shape and dtype as `x1`.\n  '
    r2 = (x2 - x1) / (y2 - y1)
    r3 = (x3 - x1) / (y3 - y1)
    return -x1 * tf.math.divide_no_nan(x3 * r3 - x2 * r2, r3 * r2 * (x3 - x2))

def _should_stop(state, stopping_policy_fn):
    if False:
        while True:
            i = 10
    'Indicates whether the overall Brent search should continue.\n\n  Args:\n    state: A Python `_BrentSearchState` namedtuple.\n    stopping_policy_fn: Python `callable` controlling the algorithm termination.\n\n  Returns:\n    A boolean value indicating whether the overall search should continue.\n  '
    return tf.convert_to_tensor(stopping_policy_fn(state.finished), name='should_stop', dtype=tf.bool)

def _brent_loop_body(state, params, constants):
    if False:
        i = 10
        return i + 15
    'Performs one iteration of the Brent root-finding algorithm.\n\n  Args:\n    state: A Python `_BrentSearchState` namedtuple.\n    params: A Python `_BrentSearchParams` namedtuple.\n    constants: A Python `_BrentSearchConstants` namedtuple.\n\n  Returns:\n    The `Tensor`s to use for the next iteration of the algorithm.\n  '
    best_estimate = state.best_estimate
    last_estimate = state.last_estimate
    contrapoint = state.contrapoint
    value_at_best_estimate = state.value_at_best_estimate
    value_at_last_estimate = state.value_at_last_estimate
    value_at_contrapoint = state.value_at_contrapoint
    step_to_best_estimate = state.step_to_best_estimate
    step_to_last_estimate = state.step_to_last_estimate
    num_iterations = state.num_iterations
    finished = state.finished
    replace_contrapoint = ~finished & (value_at_last_estimate * value_at_best_estimate < constants.zero_value)
    contrapoint = tf.where(replace_contrapoint, last_estimate, contrapoint)
    value_at_contrapoint = tf.where(replace_contrapoint, value_at_last_estimate, value_at_contrapoint)
    step_to_last_estimate = tf.where(replace_contrapoint, best_estimate - last_estimate, step_to_last_estimate)
    step_to_best_estimate = tf.where(replace_contrapoint, step_to_last_estimate, step_to_best_estimate)
    replace_best_estimate = tf.where(finished, constants.false, tf.math.abs(value_at_contrapoint) < tf.math.abs(value_at_best_estimate))
    last_estimate = tf.where(replace_best_estimate, best_estimate, last_estimate)
    best_estimate = tf.where(replace_best_estimate, contrapoint, best_estimate)
    contrapoint = tf.where(replace_best_estimate, last_estimate, contrapoint)
    value_at_last_estimate = tf.where(replace_best_estimate, value_at_best_estimate, value_at_last_estimate)
    value_at_best_estimate = tf.where(replace_best_estimate, value_at_contrapoint, value_at_best_estimate)
    value_at_contrapoint = tf.where(replace_best_estimate, value_at_last_estimate, value_at_contrapoint)
    root_tolerance = 0.5 * (params.absolute_root_tolerance + params.relative_root_tolerance * tf.math.abs(best_estimate))
    bisection_step = 0.5 * (contrapoint - best_estimate)
    finished |= (num_iterations >= params.max_iterations) | (tf.math.abs(bisection_step) < root_tolerance) | ~tf.math.is_finite(value_at_best_estimate) | (tf.math.abs(value_at_best_estimate) <= params.function_tolerance)
    compute_short_step = tf.where(finished, constants.false, (root_tolerance < tf.math.abs(step_to_last_estimate)) & (tf.math.abs(value_at_best_estimate) < tf.math.abs(value_at_last_estimate)))
    short_step = tf.where(compute_short_step, tf.where(tf.equal(last_estimate, contrapoint), _secant_step(best_estimate, last_estimate, value_at_best_estimate, value_at_last_estimate), _quadratic_interpolation_step(value_at_best_estimate, value_at_last_estimate, value_at_contrapoint, best_estimate, last_estimate, contrapoint)), constants.zero)
    use_short_step = tf.where(compute_short_step, 2 * tf.math.abs(short_step) < tf.minimum(3 * tf.math.abs(bisection_step) - root_tolerance, tf.math.abs(step_to_last_estimate)), constants.false)
    step_to_last_estimate = tf.where(use_short_step, step_to_best_estimate, bisection_step)
    step_to_best_estimate = tf.where(finished, constants.zero, tf.where(use_short_step, short_step, bisection_step))
    last_estimate = tf.where(finished, last_estimate, best_estimate)
    best_estimate += tf.where(finished, constants.zero, tf.where(root_tolerance < tf.math.abs(step_to_best_estimate), step_to_best_estimate, tf.where(bisection_step > 0, root_tolerance, -root_tolerance)))
    value_at_last_estimate = tf.where(finished, value_at_last_estimate, value_at_best_estimate)
    value_at_best_estimate = tf.where(finished, value_at_best_estimate, params.objective_fn(best_estimate))
    num_iterations = tf.where(finished, num_iterations, num_iterations + 1)
    return [_BrentSearchState(best_estimate=best_estimate, last_estimate=last_estimate, contrapoint=contrapoint, value_at_best_estimate=value_at_best_estimate, value_at_last_estimate=value_at_last_estimate, value_at_contrapoint=value_at_contrapoint, step_to_best_estimate=step_to_best_estimate, step_to_last_estimate=step_to_last_estimate, num_iterations=num_iterations, finished=finished)]

def _prepare_brent_args(objective_fn, left_bracket, right_bracket, value_at_left_bracket, value_at_right_bracket, absolute_root_tolerance=2e-07, relative_root_tolerance=None, function_tolerance=2e-07, max_iterations=100, stopping_policy_fn=None):
    if False:
        return 10
    "Prepares arguments for root search using Brent's method.\n\n  Args:\n    objective_fn: Python callable for which roots are searched. It must be a\n      callable of a single `Tensor` parameter and return a `Tensor` of the same\n      shape and dtype as `left_bracket`.\n    left_bracket: `Tensor` or Python float representing the first starting\n      points. The function will search for roots between each pair of points\n      defined by `left_bracket` and `right_bracket`. The shape of `left_bracket`\n      should match that of the input to `objective_fn`.\n    right_bracket: `Tensor` of the same shape and dtype as `left_bracket` or\n      Python float representing the second starting points. The function will\n      search for roots between each pair of points defined by `left_bracket` and\n      `right_bracket`. This argument must have the same shape as `left_bracket`.\n    value_at_left_bracket: Optional `Tensor` or Python float representing the\n      value of `objective_fn` at `left_bracket`. If specified, this argument\n      must have the same shape as `left_bracket`. If not specified, the value\n      will be evaluated during the search.\n      Default value: None.\n    value_at_right_bracket: Optional `Tensor` or Python float representing the\n      value of `objective_fn` at `right_bracket`. If specified, this argument\n      must have the same shape as `right_bracket`. If not specified, the value\n      will be evaluated during the search.\n      Default value: None.\n    absolute_root_tolerance: Optional `Tensor` representing the absolute\n      tolerance for estimated roots, with the total tolerance being calculated\n      as `(absolute_root_tolerance + relative_root_tolerance * |root|) / 2`. If\n      specified, this argument must be positive, broadcast with the shape of\n      `left_bracket` and have the same dtype.\n      Default value: `2e-7`.\n    relative_root_tolerance: Optional `Tensor` representing the relative\n      tolerance for estimated roots, with the total tolerance being calculated\n      as `(absolute_root_tolerance + relative_root_tolerance * |root|) / 2`. If\n      specified, this argument must be positive, broadcast with the shape of\n      `left_bracket` and have the same dtype.\n      Default value: `None` which translates to `4 *\n        numpy.finfo(left_bracket.dtype.as_numpy_dtype).eps`.\n    function_tolerance: Optional `Tensor` representing the tolerance used to\n      check for roots. If the absolute value of `objective_fn` is smaller than\n      or equal to `function_tolerance` at a given estimate, then that estimate\n      is considered a root for the function. If specified, this argument must\n      broadcast with the shape of `left_bracket` and have the same dtype. Set to\n      zero to match Brent's original algorithm and to continue the search until\n      an exact root is found.\n      Default value: `2e-7`.\n    max_iterations: Optional `Tensor` of an integral dtype or Python integer\n      specifying the maximum number of steps to perform for each initial point.\n      Must broadcast with the shape of `left_bracket`. If an element is set to\n      zero, the function will not search for any root for the corresponding\n      points in `left_bracket` and `right_bracket`. Instead, it will return the\n      best estimate from the inputs.\n      Default value: `100`.\n    stopping_policy_fn: Python `callable` controlling the algorithm termination.\n      It must be a callable accepting a `Tensor` of booleans with the shape of\n      `left_bracket` (each denoting whether the search is finished for each\n      starting point), and returning a scalar boolean `Tensor` (indicating\n      whether the overall search should stop). Typical values are\n      `tf.reduce_all` (which returns only when the search is finished for all\n      pairs of points), and `tf.reduce_any` (which returns as soon as the search\n      is finished for any pair of points).\n      Default value: `None` which translates to `tf.reduce_all`.\n\n  Returns:\n    A tuple of 3 Python objects containing the state, parameters, and constants\n    to use for the search.\n  "
    stopping_policy_fn = stopping_policy_fn or tf.reduce_all
    if not callable(stopping_policy_fn):
        raise ValueError('stopping_policy_fn must be callable')
    left_bracket = tf.convert_to_tensor(left_bracket, name='left_bracket')
    right_bracket = tf.convert_to_tensor(right_bracket, name='right_bracket', dtype=left_bracket.dtype)
    if value_at_left_bracket is None:
        value_at_left_bracket = objective_fn(left_bracket)
    if value_at_right_bracket is None:
        value_at_right_bracket = objective_fn(right_bracket)
    value_at_left_bracket = tf.convert_to_tensor(value_at_left_bracket, name='value_at_left_bracket', dtype=left_bracket.dtype.base_dtype)
    value_at_right_bracket = tf.convert_to_tensor(value_at_right_bracket, name='value_at_right_bracket', dtype=left_bracket.dtype.base_dtype)
    if relative_root_tolerance is None:
        relative_root_tolerance = utils.default_relative_root_tolerance(left_bracket.dtype.base_dtype)
    absolute_root_tolerance = tf.convert_to_tensor(absolute_root_tolerance, name='absolute_root_tolerance', dtype=left_bracket.dtype)
    relative_root_tolerance = tf.convert_to_tensor(relative_root_tolerance, name='relative_root_tolerance', dtype=left_bracket.dtype)
    function_tolerance = tf.convert_to_tensor(function_tolerance, name='function_tolerance', dtype=left_bracket.dtype)
    max_iterations = tf.broadcast_to(tf.convert_to_tensor(max_iterations), name='max_iterations', shape=left_bracket.shape)
    num_iterations = tf.zeros_like(max_iterations)
    false = tf.constant(False, shape=left_bracket.shape)
    zero = tf.zeros_like(left_bracket)
    contrapoint = zero
    step_to_last_estimate = zero
    step_to_best_estimate = zero
    zero_value = tf.zeros_like(value_at_left_bracket)
    value_at_contrapoint = zero_value
    swap_positions = tf.math.abs(value_at_left_bracket) < tf.math.abs(value_at_right_bracket)
    (best_estimate, last_estimate) = _swap_where(swap_positions, right_bracket, left_bracket)
    (value_at_best_estimate, value_at_last_estimate) = _swap_where(swap_positions, value_at_right_bracket, value_at_left_bracket)
    finished = (num_iterations >= max_iterations) | ~tf.math.is_finite(value_at_last_estimate) | ~tf.math.is_finite(value_at_best_estimate) | (tf.math.abs(value_at_best_estimate) <= function_tolerance)
    return (_BrentSearchState(best_estimate=best_estimate, last_estimate=last_estimate, contrapoint=contrapoint, value_at_best_estimate=value_at_best_estimate, value_at_last_estimate=value_at_last_estimate, value_at_contrapoint=value_at_contrapoint, step_to_best_estimate=step_to_best_estimate, step_to_last_estimate=step_to_last_estimate, num_iterations=num_iterations, finished=finished), _BrentSearchParams(objective_fn=objective_fn, max_iterations=max_iterations, absolute_root_tolerance=absolute_root_tolerance, relative_root_tolerance=relative_root_tolerance, function_tolerance=function_tolerance, stopping_policy_fn=stopping_policy_fn), _BrentSearchConstants(false=false, zero=zero, zero_value=zero_value))

def _brent(objective_fn, left_bracket, right_bracket, value_at_left_bracket=None, value_at_right_bracket=None, absolute_root_tolerance=2e-07, relative_root_tolerance=None, function_tolerance=2e-07, max_iterations=100, stopping_policy_fn=None, validate_args=False, name=None):
    if False:
        print('Hello World!')
    "Finds root(s) of a function of a single variable using Brent's method.\n\n  [Brent's method](https://en.wikipedia.org/wiki/Brent%27s_method) is a\n  root-finding algorithm combining the bisection method, the secant method and\n  extrapolation. Like bisection it is guaranteed to converge towards a root if\n  one exists, but that convergence is superlinear and on par with less reliable\n  methods.\n\n  This implementation is a translation of the algorithm described in the\n  [original article](https://academic.oup.com/comjnl/article/14/4/422/325237).\n\n  Args:\n    objective_fn: Python callable for which roots are searched. It must be a\n      callable of a single `Tensor` parameter and return a `Tensor` of the same\n      shape and dtype as `left_bracket`.\n    left_bracket: `Tensor` or Python float representing the first starting\n      points. The function will search for roots between each pair of points\n      defined by `left_bracket` and `right_bracket`. The shape of `left_bracket`\n      should match that of the input to `objective_fn`.\n    right_bracket: `Tensor` of the same shape and dtype as `left_bracket` or\n      Python float representing the second starting points. The function will\n      search for roots between each pair of points defined by `left_bracket` and\n      `right_bracket`. This argument must have the same shape as `left_bracket`.\n    value_at_left_bracket: Optional `Tensor` or Python float representing the\n      value of `objective_fn` at `left_bracket`. If specified, this argument\n      must have the same shape as `left_bracket`. If not specified, the value\n      will be evaluated during the search.\n      Default value: None.\n    value_at_right_bracket: Optional `Tensor` or Python float representing the\n      value of `objective_fn` at `right_bracket`. If specified, this argument\n      must have the same shape as `right_bracket`. If not specified, the value\n      will be evaluated during the search.\n      Default value: None.\n    absolute_root_tolerance: Optional `Tensor` representing the absolute\n      tolerance for estimated roots, with the total tolerance being calculated\n      as `(absolute_root_tolerance + relative_root_tolerance * |root|) / 2`. If\n      specified, this argument must be positive, broadcast with the shape of\n      `left_bracket` and have the same dtype.\n      Default value: `2e-7`.\n    relative_root_tolerance: Optional `Tensor` representing the relative\n      tolerance for estimated roots, with the total tolerance being calculated\n      as `(absolute_root_tolerance + relative_root_tolerance * |root|) / 2`. If\n      specified, this argument must be positive, broadcast with the shape of\n      `left_bracket` and have the same dtype.\n      Default value: `None` which translates to `4 *\n        numpy.finfo(left_bracket.dtype.as_numpy_dtype).eps`.\n    function_tolerance: Optional `Tensor` representing the tolerance used to\n      check for roots. If the absolute value of `objective_fn` is smaller than\n      or equal to `function_tolerance` at a given estimate, then that estimate\n      is considered a root for the function. If specified, this argument must\n      broadcast with the shape of `left_bracket` and have the same dtype. Set to\n      zero to match Brent's original algorithm and to continue the search until\n      an exact root is found.\n      Default value: `2e-7`.\n    max_iterations: Optional `Tensor` of an integral dtype or Python integer\n      specifying the maximum number of steps to perform for each initial point.\n      Must broadcast with the shape of `left_bracket`. If an element is set to\n      zero, the function will not search for any root for the corresponding\n      points in `left_bracket` and `right_bracket`. Instead, it will return the\n      best estimate from the inputs.\n      Default value: `100`.\n    stopping_policy_fn: Python `callable` controlling the algorithm termination.\n      It must be a callable accepting a `Tensor` of booleans with the shape of\n      `left_bracket` (each denoting whether the search is finished for each\n      starting point), and returning a scalar boolean `Tensor` (indicating\n      whether the overall search should stop). Typical values are\n      `tf.reduce_all` (which returns only when the search is finished for all\n      pairs of points), and `tf.reduce_any` (which returns as soon as the search\n      is finished for any pair of points).\n      Default value: `None` which translates to `tf.reduce_all`.\n    validate_args: Python `bool` indicating whether to validate arguments such\n      as `left_bracket`, `right_bracket`, `absolute_root_tolerance`,\n      `relative_root_tolerance`, `function_tolerance`, and `max_iterations`.\n      Default value: `False`.\n    name: Python `str` name prefixed to ops created by this function.\n\n  Returns:\n    brent_results: A Python object containing the following attributes:\n      estimated_root: `Tensor` containing the best estimate explored. If the\n        search was successful within the specified tolerance, this estimate is\n        a root of the objective function.\n      objective_at_estimated_root: `Tensor` containing the value of the\n        objective function at `estimated_root`. If the search was successful\n        within the specified tolerance, then this is close to 0. It has the\n        same dtype and shape as `estimated_root`.\n      num_iterations: `Tensor` containing the number of iterations performed.\n        It has the same dtype as `max_iterations` and shape as `estimated_root`.\n      converged: Scalar boolean `Tensor` indicating whether `estimated_root` is\n        a root within the tolerance specified for the search. It has the same\n        shape as `estimated_root`.\n\n  Raises:\n    ValueError: if the `stopping_policy_fn` is not callable.\n  "
    with tf.name_scope(name or 'brent_root'):
        (state, params, constants) = _prepare_brent_args(objective_fn, left_bracket, right_bracket, value_at_left_bracket, value_at_right_bracket, absolute_root_tolerance, relative_root_tolerance, function_tolerance, max_iterations, stopping_policy_fn)
        assertions = []
        if validate_args:
            assertions += [tf.Assert(tf.reduce_all(state.value_at_last_estimate * state.value_at_best_estimate <= constants.zero_value), [state.value_at_last_estimate, state.value_at_best_estimate]), tf.Assert(tf.reduce_all(params.absolute_root_tolerance > constants.zero), [params.absolute_root_tolerance]), tf.Assert(tf.reduce_all(params.relative_root_tolerance > constants.zero), [params.relative_root_tolerance]), tf.Assert(tf.reduce_all(params.function_tolerance >= constants.zero), [params.function_tolerance]), tf.Assert(tf.reduce_all(params.max_iterations >= state.num_iterations), [params.max_iterations])]
        with tf.compat.v1.control_dependencies(assertions):
            result = tf.while_loop(lambda loop_vars: ~_should_stop(loop_vars, params.stopping_policy_fn), lambda state: _brent_loop_body(state, params, constants), loop_vars=[state], maximum_iterations=max_iterations)
    state = result[0]
    converged = tf.math.abs(state.value_at_best_estimate) <= function_tolerance
    return BrentResults(estimated_root=state.best_estimate, objective_at_estimated_root=state.value_at_best_estimate, num_iterations=state.num_iterations, converged=converged)

def brentq(objective_fn: Callable[[types.RealTensor], types.RealTensor], left_bracket: types.RealTensor, right_bracket: types.RealTensor, value_at_left_bracket: types.RealTensor=None, value_at_right_bracket: types.RealTensor=None, absolute_root_tolerance: types.RealTensor=2e-07, relative_root_tolerance: types.RealTensor=None, function_tolerance: types.RealTensor=2e-07, max_iterations: types.IntTensor=100, stopping_policy_fn: Callable[[types.BoolTensor], types.BoolTensor]=None, validate_args: bool=False, name: str=None) -> BrentResults:
    if False:
        for i in range(10):
            print('nop')
    "Finds root(s) of a function of single variable using Brent's method.\n\n  [Brent's method](https://en.wikipedia.org/wiki/Brent%27s_method) is a\n  root-finding algorithm combining the bisection method, the secant method and\n  extrapolation. Like bisection it is guaranteed to converge towards a root if\n  one exists, but that convergence is superlinear and on par with less reliable\n  methods.\n\n  This implementation is a translation of the algorithm described in the\n  [original article](https://academic.oup.com/comjnl/article/14/4/422/325237).\n\n  #### Examples\n\n  ```python\n  import tensorflow as tf\n  import tf_quant_finance as tff\n\n  # Example 1: Roots of a single function for two pairs of starting points.\n\n  f = lambda x: 63 * x**5 - 70 * x**3 + 15 * x + 2\n  x1 = tf.constant([-10, 1], dtype=tf.float64)\n  x2 = tf.constant([10, -1], dtype=tf.float64)\n\n  tf.math.brentq(objective_fn=f, left_bracket=x1, right_bracket=x2)\n  # ==> BrentResults(\n  #    estimated_root=array([-0.14823253, -0.14823253]),\n  #    objective_at_estimated_root=array([3.27515792e-15, 0.]),\n  #    num_iterations=array([11, 6]),\n  #    converged=array([True, True]))\n\n  tff.math.root_search.brentq(objective_fn=f,\n                              left_bracket=x1,\n                              right_bracket=x2,\n                              stopping_policy_fn=tf.reduce_any)\n  # ==> BrentResults(\n  #    estimated_root=array([-2.60718234, -0.14823253]),\n  #    objective_at_estimated_root=array([-6.38579115e+03, 2.39763764e-11]),\n  #    num_iterations=array([7, 6]),\n  #    converged=array([False, True]))\n  ```\n\n  # Example 2: Roots of a multiplex function for one pair of starting points.\n\n  def f(x):\n    return tf.constant([0., 63.], dtype=tf.float64) * x**5 \\\n        + tf.constant([5., -70.], dtype=tf.float64) * x**3 \\\n        + tf.constant([-3., 15.], dtype=tf.float64) * x \\\n        + 2\n\n  x1 = tf.constant([-5, -5], dtype=tf.float64)\n  x2 = tf.constant([5, 5], dtype=tf.float64)\n\n  tff.math.root_search.brentq(objective_fn=f, left_bracket=x1, right_bracket=x2)\n  # ==> BrentResults(\n  #    estimated_root=array([-1., -0.14823253]),\n  #    objective_at_estimated_root=array([0., 2.08721929e-14]),\n  #    num_iterations=array([13, 11]),\n  #    converged=array([True, True]))\n\n  # Example 3: Roots of a multiplex function for two pairs of starting points.\n\n  def f(x):\n    return tf.constant([0., 63.], dtype=tf.float64) * x**5 \\\n        + tf.constant([5., -70.], dtype=tf.float64) * x**3 \\\n        + tf.constant([-3., 15.], dtype=tf.float64) * x \\\n        + 2\n\n  x1 = tf.constant([[-5, -5], [10, 10]], dtype=tf.float64)\n  x2 = tf.constant([[5, 5], [-10, -10]], dtype=tf.float64)\n\n  tff.math.root_search.brentq(objective_fn=f, left_bracket=x1, right_bracket=x2)\n  # ==> BrentResults(\n  #    estimated_root=array([\n  #        [-1, -0.14823253],\n  #        [-1, -0.14823253]]),\n  #    objective_at_estimated_root=array([\n  #        [0., 2.08721929e-14],\n  #        [0., 2.08721929e-14]]),\n  #    num_iterations=array([\n  #        [13, 11],\n  #        [15, 11]]),\n  #    converged=array([\n  #        [True, True],\n  #        [True, True]]))\n  ```\n\n  Args:\n    objective_fn: Python callable for which roots are searched. It must be a\n      callable of a single `Tensor` parameter and return a `Tensor` of the same\n      shape and dtype as `left_bracket`.\n    left_bracket: `Tensor` or Python float representing the first starting\n      points. The function will search for roots between each pair of points\n      defined by `left_bracket` and `right_bracket`. The shape of `left_bracket`\n      should match that of the input to `objective_fn`.\n    right_bracket: `Tensor` of the same shape and dtype as `left_bracket` or\n      Python float representing the second starting points. The function will\n      search for roots between each pair of points defined by `left_bracket` and\n      `right_bracket`. This argument must have the same shape as `left_bracket`.\n    value_at_left_bracket: Optional `Tensor` or Python float representing the\n      value of `objective_fn` at `left_bracket`. If specified, this argument\n      must have the same shape as `left_bracket`. If not specified, the value\n      will be evaluated during the search.\n      Default value: None.\n    value_at_right_bracket: Optional `Tensor` or Python float representing the\n      value of `objective_fn` at `right_bracket`. If specified, this argument\n      must have the same shape as `right_bracket`. If not specified, the value\n      will be evaluated during the search.\n      Default value: None.\n    absolute_root_tolerance: Optional `Tensor` representing the absolute\n      tolerance for estimated roots, with the total tolerance being calculated\n      as `(absolute_root_tolerance + relative_root_tolerance * |root|) / 2`. If\n      specified, this argument must be positive, broadcast with the shape of\n      `left_bracket` and have the same dtype.\n      Default value: `2e-7`.\n    relative_root_tolerance: Optional `Tensor` representing the relative\n      tolerance for estimated roots, with the total tolerance being calculated\n      as `(absolute_root_tolerance + relative_root_tolerance * |root|) / 2`. If\n      specified, this argument must be positive, broadcast with the shape of\n      `left_bracket` and have the same dtype.\n      Default value: `None` which translates to `4 *\n        numpy.finfo(left_bracket.dtype.as_numpy_dtype).eps`.\n    function_tolerance: Optional `Tensor` representing the tolerance used to\n      check for roots. If the absolute value of `objective_fn` is smaller than\n      or equal to `function_tolerance` at a given estimate, then that estimate\n      is considered a root for the function. If specified, this argument must\n      broadcast with the shape of `left_bracket` and have the same dtype. Set to\n      zero to match Brent's original algorithm and to continue the search until\n      an exact root is found.\n      Default value: `2e-7`.\n    max_iterations: Optional `Tensor` of an integral dtype or Python integer\n      specifying the maximum number of steps to perform for each initial point.\n      Must broadcast with the shape of `left_bracket`. If an element is set to\n      zero, the function will not search for any root for the corresponding\n      points in `left_bracket` and `right_bracket`. Instead, it will return the\n      best estimate from the inputs.\n      Default value: `100`.\n    stopping_policy_fn: Python `callable` controlling the algorithm termination.\n      It must be a callable accepting a `Tensor` of booleans with the shape of\n      `left_bracket` (each denoting whether the search is finished for each\n      starting point), and returning a scalar boolean `Tensor` (indicating\n      whether the overall search should stop). Typical values are\n      `tf.reduce_all` (which returns only when the search is finished for all\n      pairs of points), and `tf.reduce_any` (which returns as soon as the search\n      is finished for any pair of points).\n      Default value: `None` which translates to `tf.reduce_all`.\n    validate_args: Python `bool` indicating whether to validate arguments such\n      as `left_bracket`, `right_bracket`, `absolute_root_tolerance`,\n      `relative_root_tolerance`, `function_tolerance`, and `max_iterations`.\n      Default value: `False`.\n    name: Python `str` name prefixed to ops created by this function.\n\n  Returns:\n    brent_results: A Python object containing the following attributes:\n      estimated_root: `Tensor` containing the best estimate explored. If the\n        search was successful within the specified tolerance, this estimate is\n        a root of the objective function.\n      objective_at_estimated_root: `Tensor` containing the value of the\n        objective function at `estimated_root`. If the search was successful\n        within the specified tolerance, then this is close to 0. It has the\n        same dtype and shape as `estimated_root`.\n      num_iterations: `Tensor` containing the number of iterations performed.\n        It has the same dtype as `max_iterations` and shape as `estimated_root`.\n      converged: Scalar boolean `Tensor` indicating whether `estimated_root` is\n        a root within the tolerance specified for the search. It has the same\n        shape as `estimated_root`.\n\n  Raises:\n    ValueError: if the `stopping_policy_fn` is not callable.\n  "
    return _brent(objective_fn, left_bracket, right_bracket, value_at_left_bracket=value_at_left_bracket, value_at_right_bracket=value_at_right_bracket, absolute_root_tolerance=absolute_root_tolerance, relative_root_tolerance=relative_root_tolerance, function_tolerance=function_tolerance, max_iterations=max_iterations, stopping_policy_fn=stopping_policy_fn, validate_args=validate_args, name=name)