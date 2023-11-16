"""Utility functions for the Geometric Brownian Motion models."""
import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff

def convert_to_ndarray(test_obj, a):
    if False:
        print('Hello World!')
    'Converts the input `a` into an ndarray.\n\n  Args:\n    test_obj: An object which has the `evaluate` method. Used to evaluate `a` if\n      `a` is a Tensor.\n    a: Object to be converted to an ndarray.\n\n  Returns:\n    An ndarray containing the values of `a`.\n  '
    if tf.is_tensor(a):
        a = test_obj.evaluate(a)
    if not isinstance(a, np.ndarray):
        return np.array(a)
    return a

def arrays_all_close(test_obj, a, b, atol, msg=None):
    if False:
        while True:
            i = 10
    'Check if two arrays are within a tolerance specified per element.\n\n  Checks that a, b and atol have the same shape and that\n    `abs(a_i - b_i) <= atol_i` for all elements of `a` and `b`.\n  This function differs from np.testing.assert_allclose() as\n  np.testing.assert_allclose() applies the same `atol` to all of the elements,\n  whereas this function takes a `ndarray` specifying a `atol` for each element.\n\n  Args:\n    test_obj: An object which has the `evaluate` method. Used to evaluate `a` if\n      `a` is a Tensor.\n    a: The expected numpy `ndarray`, or anything that can be converted into a\n       numpy `ndarray` (including Tensor), or any arbitrarily nested of\n       structure of these.\n    b: The actual numpy `ndarray`, or anything that can be converted into a\n       numpy `ndarray` (including Tensor), or any arbitrarily nested of\n       structure of these.\n    atol: absolute tolerance as a numpy `ndarray` of the same shape as `a` and\n       `b`.\n    msg: Optional message to include in the error message.\n  Raises:\n    ValueError: If `a`, `b` and `atol` do not have the same shape.\n    AssertionError: If any of the elements are outside the tolerance.\n  '
    a = convert_to_ndarray(test_obj, a)
    b = convert_to_ndarray(test_obj, b)
    atol = convert_to_ndarray(test_obj, atol)
    if a.shape != b.shape:
        raise ValueError('Mismatched shapes a.shape() = {}'.format(a.shape) + ', b.shape= {}'.format(b.shape) + ', atol.shape = {}'.format(atol.shape) + '. ({}).'.format(msg))
    abs_diff = np.abs(a - b)
    if np.any(abs_diff >= atol):
        raise ValueError('Expected and actual values differ by more than the ' + 'tolerance.\n a = {}'.format(a) + '\n b = {}'.format(b) + '\n abs_diff = {}'.format(abs_diff) + '\n atol = {}'.format(atol) + '\n When {}.'.format(msg))
    return

def generate_sample_paths(mu, sigma, times, initial_state, supply_draws, num_samples, dtype):
    if False:
        return 10
    'Returns the sample paths for the process with the given parameters.\n\n  Args:\n    mu: Scalar real `Tensor` broadcastable to [`batch_shape`, 1] or an instance\n      of left-continuous `PiecewiseConstantFunc` of [`batch_shape`]\n      dimensions. Where `batch_shape` is the larger of `mu.shape` and\n      `sigma.shape`. Corresponds to the mean drift of the Ito process.\n    sigma: Scalar real `Tensor` broadcastable to [`batch_shape`, 1] or an\n      instance of left-continuous `PiecewiseConstantFunc` of the same `dtype`\n      and `batch_shape` as set by `mu`. Where `batch_shape` is the larger of\n      `mu.shape` and `sigma.shape`. Corresponds to the volatility of the\n      process and should be positive.\n    times: A `Tensor` of positive real values of a shape [`T`, `num_times`],\n      where `T` is either empty or a shape which is broadcastable to\n      `batch_shape` (as defined by the shape of `mu` or `sigma`. The times at\n      which the path points are to be evaluated.\n    initial_state: A `Tensor` of the same `dtype` as `times` and of shape\n      broadcastable to `[batch_shape, num_samples]`. Represents the initial\n      state of the Ito process.\n    supply_draws: Boolean set to true if the `normal_draws` should be generated\n      and then passed to the pricing function.\n    num_samples: Positive scalar `int`. The number of paths to draw.\n    dtype: The default dtype to use when converting values to `Tensor`s.\n\n  Returns:\n    A Tensor containing the the sample paths of shape\n    [batch_shape, num_samples, num_times, 1].\n  '
    process = tff.models.GeometricBrownianMotion(mu, sigma, dtype=dtype)
    normal_draws = None
    if supply_draws:
        total_dimension = tf.zeros(times.shape[-1], dtype=dtype)
        normal_draws = tff.math.random.mv_normal_sample([num_samples], mean=total_dimension, random_type=tff.math.random.RandomType.SOBOL, seed=[4, 2])
        normal_draws = tf.expand_dims(normal_draws, axis=-1)
    return process.sample_paths(times=times, initial_state=initial_state, random_type=tff.math.random.RandomType.STATELESS, num_samples=num_samples, normal_draws=normal_draws, seed=[1234, 5])

def calculate_mean_and_variance_from_sample_paths(samples, num_samples, dtype):
    if False:
        print('Hello World!')
    'Returns the mean and variance of log(`samples`).\n\n  Args:\n    samples: A real `Tensor` of shape [batch_shape, `num_samples`, num_times, 1]\n      containing the samples of random paths drawn from an Ito process.\n    num_samples: A scalar integer. The number of sample paths in `samples`.\n    dtype: The default dtype to use when converting values to `Tensor`s.\n\n  Returns:\n    A tuple of (mean, variance, standard_error of the mean,\n    standard_error of the variance) of the log of the samples.  Where the\n    components of the tuple have shape [batch_shape, num_times].\n  '
    log_s = tf.math.log(samples)
    mean = tf.reduce_mean(log_s, axis=-3, keepdims=True)
    var = tf.reduce_mean((log_s - mean) ** 2, axis=-3, keepdims=True)
    mean = tf.squeeze(mean, axis=[-1, -3])
    var = tf.squeeze(var, axis=[-1, -3])
    std_err_mean = tf.math.sqrt(var / num_samples)
    std_err_var = var * tf.math.sqrt(tf.constant(2.0, dtype=dtype) / (tf.constant(num_samples, dtype=dtype) - tf.constant(1.0, dtype=dtype)))
    return (mean, var, std_err_mean, std_err_var)

def calculate_sample_paths_mean_and_variance(test_obj, mu, sigma, times, initial_state, supply_draws, num_samples, dtype):
    if False:
        i = 10
        return i + 15
    'Returns the mean and variance of the log of the sample paths for a process.\n\n  Generates a set of sample paths for a univariate geometric brownian motion\n  and calculates the mean and variance of the log of the paths. Also returns the\n  standard error of the mean and variance.\n\n  Args:\n    test_obj: An object which has the `evaluate` method. Used to evaluate `a` if\n      `a` is a Tensor.\n    mu: Scalar real `Tensor` broadcastable to [`batch_shape`] or an instance\n      of left-continuous `PiecewiseConstantFunc` of [`batch_shape`]\n      dimensions. Where `batch_shape` is the larger of `mu.shape` and\n      `sigma.shape`. Corresponds to the mean drift of the Ito process.\n    sigma: Scalar real `Tensor` broadcastable to [`batch_shape`] or an\n      instance of left-continuous `PiecewiseConstantFunc` of the same `dtype`\n      and `batch_shape` as set by `mu`. Where `batch_shape` is the larger of\n      `mu.shape` and `sigma.shape`. Corresponds to the volatility of the\n      process and should be positive.\n    times: Rank 1 `Tensor` of positive real values. The times at which the\n      path points are to be evaluated.\n    initial_state: A `Tensor` of the same `dtype` as `times` and of shape\n      broadcastable to `[batch_shape, num_samples]`. Represents the initial\n      state of the Ito process.\n    supply_draws: Boolean set to true if the `normal_draws` should be generated\n      and then passed to the pricing function.\n    num_samples: Positive scalar `int`. The number of paths to draw.\n    dtype: The default dtype to use when converting values to `Tensor`s.\n\n  Returns:\n    A tuple of (mean, variance, standard_error of the mean,\n      standard_error of the variance).\n  '
    samples = generate_sample_paths(mu, sigma, times, initial_state, supply_draws, num_samples, dtype)
    return test_obj.evaluate(calculate_mean_and_variance_from_sample_paths(samples, num_samples, dtype))