"""Implementation of the regression MC of Longstaff and Schwartz.

This version of the algorithm is XLA-compatible but might be slower in some
cases than the previous version without XLA compilation. The following major
differences with the previous version are:

1. All tensors inside `_continuation_value_fn`, `_expected_exercise_fn` and
  `_updated_cashflow` are now of static shape. This increases the memory
  consumption of the algorithm but avoids dynamically shaped tensors.
2. `make_polynomial_basis` outputs the function that has an additional
  `time_index` argument. This is necessary to avoid dynamically shaped tensors.
"""
import collections
import tensorflow.compat.v2 as tf
LsmLoopVars = collections.namedtuple('LsmLoopVars', ['exercise_index', 'cashflow'])

def make_polynomial_basis(degree):
    if False:
        print('Hello World!')
    'Produces a callable from samples to polynomial basis for use in regression.\n\n  The output callable accepts a scalar `Tensor` `t` and a `Tensor` `X` of\n  shape `[num_samples, dim]`, computes a centered value\n  `Y = X - mean(X, axis=0)` and outputs a `Tensor` of shape\n  `[degree * dim, num_samples]`, where\n  ```\n  Z[i*j, k] = X[k, j]**(degree - i) * X[k, j]**i, 0<=i<degree - 1, 0<=j<dim\n  ```\n  For example, if `degree` and `dim` are both equal to 2, the polynomial basis\n  is `1, X, X**2, Y, Y**2, X * Y, X**2 * Y, X * Y**2`, where `X` and `Y` are\n  the spatial axes.\n\n  #### Example\n  ```python\n  basis = tff.experimental.lsm_algorithm.make_polynomial_basis_v2(2)\n  x = [[1.0], [2.0], [3.0], [4.0]]\n  x = tf.expand_dims(x, axis=-1)\n  basis(x, tf.constant(0, dtype=np.int32))\n  # Expected result:\n  [[ 1.  ,  1.  ,  1.  ,  1.  ], [-1.5 , -0.5 ,  0.5 ,  1.5 ],\n  [ 2.25,  0.25,  0.25,  2.25]]\n  ```\n\n  Args:\n    degree: An `int32` scalar `Tensor`. The degree of the desired polynomial\n      basis.\n\n  Returns:\n    A callable from `Tensor`s of shape `[batch_size, num_samples, dim]` to\n    `Tensor`s of shape `[batch_size, (degree + 1)**dim, num_samples]`.\n\n  Raises:\n    ValueError: If `degree` is less than `1`.\n  '
    tf.debugging.assert_greater_equal(degree, 0, message='Degree of polynomial basis can not be negative.')

    def basis(sample_paths, time_index):
        if False:
            print('Hello World!')
        'Computes polynomial basis expansion at the given sample points.\n\n    Args:\n      sample_paths: A `Tensor` of either `flaot32` or `float64` dtype and of\n        either shape `[num_samples, num_times, dim]` or\n        `[batch_size, num_samples, num_times, dim]`.\n      time_index: An integer scalar `Tensor` that corresponds to the time\n        coordinate at which the basis function is computed.\n\n    Returns:\n      A `Tensor`s of shape `[batch_size, (degree + 1)**dim, num_samples]`.\n    '
        sample_paths = tf.convert_to_tensor(sample_paths, name='sample_paths')
        if sample_paths.shape.rank == 3:
            sample_paths = tf.expand_dims(sample_paths, axis=0)
        shape = tf.shape(sample_paths)
        num_samples = shape[1]
        batch_size = shape[0]
        dim = sample_paths.shape[-1]
        slice_samples = tf.slice(sample_paths, [0, 0, time_index, 0], [batch_size, num_samples, 1, dim])
        samples_centered = slice_samples - tf.math.reduce_mean(slice_samples, axis=1, keepdims=True)
        grid = tf.range(degree + 1, dtype=samples_centered.dtype)
        grid = tf.meshgrid(*dim * [grid])
        grid = tf.reshape(tf.stack(grid, -1), [-1, dim])
        basis_expansion = tf.reduce_prod(samples_centered ** grid, axis=-1)
        return tf.transpose(basis_expansion, [0, 2, 1])
    return basis

def least_square_mc(sample_paths, exercise_times, payoff_fn, basis_fn, discount_factors=None, num_calibration_samples=None, dtype=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    "Values Amercian style options using the LSM algorithm.\n\n  The Least-Squares Monte-Carlo (LSM) algorithm is a Monte-Carlo approach to\n  valuation of American style options. Using the sample paths of underlying\n  assets, and a user supplied payoff function it attempts to find the optimal\n  exercise point along each sample path. With optimal exercise points known,\n  the option is valued as the average payoff assuming optimal exercise\n  discounted to present value.\n\n  #### Example. American put option price through Monte Carlo\n  ```python\n  # Let the underlying model be a Black-Scholes process\n  # dS_t / S_t = rate dt + sigma**2 dW_t, S_0 = 1.0\n  # with `rate = 0.1`, and volatility `sigma = 1.0`.\n  # Define drift and volatility functions for log(S_t)\n  rate = 0.1\n  def drift_fn(_, x):\n    return rate - tf.ones_like(x) / 2.\n  def vol_fn(_, x):\n    return tf.expand_dims(tf.ones_like(x), axis=-1)\n  # Use Euler scheme to propagate 100000 paths for 1 year into the future\n  times = np.linspace(0., 1, num=50)\n  num_samples = 100000\n  log_paths = tf.function(tff.models.euler_sampling.sample)(\n          dim=1,\n          drift_fn=drift_fn, volatility_fn=vol_fn,\n          random_type=tff.math.random.RandomType.PSEUDO_ANTITHETIC,\n          times=times, num_samples=num_samples, seed=42, time_step=0.01)\n  # Compute exponent to get samples of `S_t`\n  paths = tf.math.exp(log_paths)\n  # American put option price for strike 1.1 and expiry 1 (assuming actual day\n  # count convention and no settlement adjustment)\n  strike = [1.1]\n  exercise_times = tf.range(times.shape[-1])\n  discount_factors = tf.exp(-rate * times)\n  payoff_fn = make_basket_put_payoff(strike)\n  basis_fn = make_polynomial_basis(10)\n  least_square_mc(paths, exercise_times, payoff_fn, basis_fn,\n                  discount_factors=discount_factors)\n  # Expected value: [0.397]\n  # European put option price\n  tff.black_scholes.option_price(volatilities=[1], strikes=strikes,\n                                 expiries=[1], spots=[1.],\n                                 discount_factors=discount_factors[-1],\n                                 is_call_options=False,\n                                 dtype=tf.float64)\n  # Expected value: [0.379]\n  ```\n  #### References\n\n  [1] Longstaff, F.A. and Schwartz, E.S., 2001. Valuing American options by\n  simulation: a simple least-squares approach. The review of financial studies,\n  14(1), pp.113-147.\n\n  Args:\n    sample_paths: A `Tensor` of either shape `[num_samples, num_times, dim]` or\n      `[batch_size, num_samples, num_times, dim]`, the sample paths of the\n      underlying ito process of dimension `dim` at `num_times` different points.\n      The `batch_size` allows multiple options to be valued in parallel.\n    exercise_times: An `int32` `Tensor` of shape `[num_exercise_times]`.\n      Contents must be a subset of the integers `[0,...,num_times - 1]`,\n      representing the time indices at which the option may be exercised.\n    payoff_fn: Callable from a `Tensor` of shape `[num_samples, S, dim]`\n      (where S <= num_times) to a `Tensor` of shape `[num_samples, batch_size]`\n      of the same dtype as `samples`. The output represents the payout resulting\n      from exercising the option at time `S`. The `batch_size` allows multiple\n      options to be valued in parallel.\n    basis_fn: Callable from a `Tensor` of the same shape and `dtype` as\n      `sample_paths` and a positive integer `Tenor` (representing a current\n      time index) to a `Tensor` of shape `[batch_size, basis_size, num_samples]`\n      of the same dtype as `sample_paths`. The result being the design matrix\n      used in regression of the continuation value of options.\n    discount_factors: A `Tensor` of shape `[num_exercise_times]` or of rank 3\n      and compatible with `[num_samples, batch_size, num_exercise_times]`.\n      The `dtype` should be the same as of `samples`.\n      Default value: `None` which maps to a one-`Tensor` of the same `dtype`\n        as `samples` and shape `[num_exercise_times]`.\n    num_calibration_samples: An optional integer less or equal to `num_samples`.\n      The number of sampled trajectories used for the LSM regression step.\n      Note that only the last`num_samples - num_calibration_samples` of the\n      sampled paths are used to determine the price of the option.\n      Default value: `None`, which means that all samples are used for\n        regression and option pricing.\n    dtype: Optional `dtype`. Either `tf.float32` or `tf.float64`. The `dtype`\n      If supplied, represents the `dtype` for the input and output `Tensor`s.\n      Default value: `None`, which means that the `dtype` inferred by TensorFlow\n      is used.\n    name: Python `str` name prefixed to Ops created by this function.\n      Default value: `None` which is mapped to the default name\n      'least_square_mc'.\n\n  Returns:\n    A `Tensor` of shape `[num_samples, batch_size]` of the same dtype as\n    `samples`.\n  "
    name = name or 'least_square_mc'
    with tf.name_scope(name):
        sample_paths = tf.convert_to_tensor(sample_paths, dtype=dtype, name='sample_paths')
        dtype = sample_paths.dtype
        exercise_times = tf.convert_to_tensor(exercise_times, name='exercise_times')
        num_times = exercise_times.shape.as_list()[-1]
        if discount_factors is None:
            discount_factors = tf.ones(shape=exercise_times.shape, dtype=dtype, name='discount_factors')
        else:
            discount_factors = tf.convert_to_tensor(discount_factors, dtype=dtype, name='discount_factors')
        if discount_factors.shape.rank == 1:
            discount_factors = tf.reshape(discount_factors, [1, 1, -1])
        rank_discount_factors = discount_factors.shape.rank
        discount_factors = tf.pad(discount_factors, (rank_discount_factors - 1) * [[0, 0]] + [[1, 0]], constant_values=1)
        time_index = exercise_times[num_times - 1]
        exercise_value = payoff_fn(sample_paths, time_index)
        zeros = tf.zeros(exercise_value.shape + [num_times - 1], dtype=dtype)
        exercise_value = tf.expand_dims(exercise_value, axis=-1)
        cashflow = tf.concat([zeros, exercise_value], axis=-1)
        calibration_indices = None
        if num_calibration_samples is not None:
            calibration_indices = tf.range(num_calibration_samples)
        lsm_loop_vars = LsmLoopVars(exercise_index=num_times - 1, cashflow=cashflow)

        def loop_body(exercise_index, cashflow):
            if False:
                while True:
                    i = 10
            return _lsm_loop_body(sample_paths=sample_paths, exercise_times=exercise_times, discount_factors=discount_factors, payoff_fn=payoff_fn, basis_fn=basis_fn, num_times=num_times, exercise_index=exercise_index, cashflow=cashflow, calibration_indices=calibration_indices)
        max_iterations = tf.shape(exercise_times)[-1]
        loop_value = tf.while_loop(_lsm_loop_cond, loop_body, lsm_loop_vars, maximum_iterations=max_iterations)
        present_values = _continuation_value_fn(loop_value.cashflow, discount_factors, 0)
        if num_calibration_samples is not None:
            present_values = present_values[num_calibration_samples:]
        return tf.math.reduce_mean(present_values, axis=0)

def _lsm_loop_cond(exercise_index, cashflow):
    if False:
        while True:
            i = 10
    'Condition to exit a countdown loop when the exercise date hits zero.'
    del cashflow
    return exercise_index > 0

def _continuation_value_fn(cashflow, discount_factors, exercise_index):
    if False:
        return 10
    'Returns the discounted value of the right hand part of the cashflow tensor.\n\n  Args:\n    cashflow: A real `Tensor` of shape\n      `[num_samples, batch_size, num_exercise]`. Tracks the optimal cashflow of\n       each sample path for each payoff dimension at each exercise time.\n    discount_factors: A `Tensor` of shape `[num_exercise_times]` or of rank 3\n      and compatible with `[num_samples, batch_size, num_exercise_times]`.\n      The `dtype` should be the same as of `samples`\n    exercise_index: An integer scalar `Tensor` representing the index of the\n      exercise time of interest. Should be less than `num_exercise_times`.\n\n  Returns:\n    A `[num_samples, batch_size]` `Tensor` whose entries represent the sum of\n    those elements to the right of `exercise_index` in `cashflow`, discounted to\n    the time indexed by `exercise_index`. When `exercise_index` is zero, the\n    return represents the sum of the cashflow discounted to present value for\n    each sample path.\n  '
    (_, _, num_cashflow) = cashflow.shape.as_list()
    disc_factors_are_used = tf.range(num_cashflow + 1) >= exercise_index + 1
    discount_factors_slice = tf.transpose(tf.transpose(discount_factors)[exercise_index])
    total_discount_factors = tf.where(disc_factors_are_used, discount_factors, 0) / tf.expand_dims(discount_factors_slice, axis=-1)
    cashflows_are_used = tf.range(num_cashflow) >= exercise_index
    cashflow_masked = tf.where(cashflows_are_used, cashflow, 0)
    total_discount_factors_slice = tf.transpose(tf.transpose(total_discount_factors)[1:])
    return tf.math.reduce_sum(cashflow_masked * total_discount_factors_slice, axis=2)

def _expected_exercise_fn(design, calibration_indices, continuation_value, exercise_value):
    if False:
        print('Hello World!')
    'Returns the expected continuation value for each path.\n\n  Args:\n    design: A real `Tensor` of shape `[batch_size, basis_size, num_samples]`.\n    calibration_indices: A rank 1 integer `Tensor` denoting indices of samples\n      used for regression.\n    continuation_value: A `Tensor` of shape `[num_samples, batch_size]` and of\n      the same dtype as `design`. The optimal value of the option conditional on\n      not exercising now or earlier, taking future information into account.\n    exercise_value: A `Tensor` of the same shape and dtype as\n      `continuation_value`. Value of the option if exercised immideately at\n      the current time\n\n  Returns:\n    A `Tensor` of the same shape and dtype as `continuation_value` whose\n    `(n, v)`-th entry represents the expected continuation value of sample path\n    `n` under the `v`-th payoff scheme.\n  '
    mask = exercise_value > 0
    design_t = tf.transpose(design, [0, 2, 1])
    masked = tf.where(tf.expand_dims(tf.transpose(mask), axis=-1), design_t, tf.zeros_like(design_t))
    if calibration_indices is None:
        submask = masked
        mask_cont_value = continuation_value
    else:
        submask = tf.gather(masked, calibration_indices, axis=1)
        mask_cont_value = tf.gather(continuation_value, calibration_indices)
    lhs = tf.matmul(submask, submask, transpose_a=True)
    lhs_pinv = tf.linalg.pinv(lhs)
    rhs = tf.matmul(submask, tf.expand_dims(tf.transpose(mask_cont_value), axis=-1), transpose_a=True)
    beta = tf.matmul(lhs_pinv, rhs)
    continuation = tf.matmul(design_t, beta)
    return tf.nn.relu(tf.transpose(tf.squeeze(continuation, axis=-1)))

def _updated_cashflow(num_times, exercise_index, exercise_value, expected_continuation, cashflow):
    if False:
        print('Hello World!')
    'Revises the cashflow tensor where options will be exercised earlier.'
    do_exercise_bool = exercise_value > expected_continuation
    scaled_do_exercise = tf.where(do_exercise_bool, exercise_value, 0)
    new_samp_masked = tf.expand_dims(scaled_do_exercise, axis=2)
    cashflow_update = tf.where(tf.expand_dims(do_exercise_bool, axis=-1), tf.constant(0, dtype=cashflow.dtype), cashflow)
    new_mask = tf.range(0, num_times) >= exercise_index
    return tf.where(new_mask, cashflow_update, new_samp_masked)

def _lsm_loop_body(sample_paths, exercise_times, discount_factors, payoff_fn, basis_fn, num_times, exercise_index, cashflow, calibration_indices):
    if False:
        for i in range(10):
            print('nop')
    'Finds the optimal exercise point and updates `cashflow`.'
    time_index = exercise_times[exercise_index - 1]
    exercise_value = payoff_fn(sample_paths, time_index)
    continuation_value = _continuation_value_fn(cashflow, discount_factors, exercise_index)
    design = basis_fn(sample_paths, time_index)
    expected_continuation = _expected_exercise_fn(design, calibration_indices, continuation_value, exercise_value)
    rev_cash = _updated_cashflow(num_times, exercise_index, exercise_value, expected_continuation, cashflow)
    return LsmLoopVars(exercise_index=exercise_index - 1, cashflow=rev_cash)