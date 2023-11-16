"""Methods for brownian bridges.

These can be used in Monte-Carlo simulation for payoff with continuous barrier.
Indeed, the Monte-Carlo simulation is inherently discrete in time, and to
improve convergence (w.r.t. the number of time steps) for payoff with continuous
barrier, adjustment with brownian bridge can be made.

## References

[1] Emmanuel Gobet. Advanced Monte Carlo methods for barrier and related
exotic options.
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1265669
"""
import tensorflow.compat.v2 as tf

def brownian_bridge_double(*, x_start, x_end, variance, upper_barrier, lower_barrier, n_cutoff=3, dtype=None, name=None):
    if False:
        return 10
    "Computes probability of not touching the barriers for a 1D Brownian Bridge.\n\n  The Brownian bridge starts at `x_start`, ends at `x_end` and has a variance\n  `variance`. The no-touch probabilities are calculated assuming that `x_start`\n  and `x_end` are within the barriers 'lower_barrier' and 'upper_barrier'.\n  This can be used in Monte Carlo pricing for adjusting probability of\n  touching the barriers from discrete case to continuous case.\n  Typically in practice, the tensors `x_start`, `x_end` and `variance` should be\n  of rank 2 (with time steps and paths being the 2 dimensions).\n\n  #### Example\n\n  ```python\n  x_start = np.asarray([[4.5, 4.5, 4.5], [4.5, 4.6, 4.7]])\n  x_end = np.asarray([[5.0, 4.9, 4.8], [4.8, 4.9, 5.0]])\n  variance = np.asarray([[0.1, 0.2, 0.1], [0.3, 0.1, 0.2]])\n  upper_barrier = 5.1\n  lower_barrier = 4.4\n\n  no_touch_proba = brownian_bridge_double(\n    x_start=x_start,\n    x_end=x_end,\n    variance=variance,\n    upper_barrier=upper_barrier,\n    lower_barrier=lower_barrier,\n    n_cutoff=3,\n    )\n  # Expected print output of no_touch_proba:\n  #[[0.45842169 0.21510919 0.52704599]\n  #[0.09394963 0.73302813 0.22595022]]\n  ```\n\n  #### References\n\n  [1] Emmanuel Gobet. Advanced Monte Carlo methods for barrier and related\n  exotic options.\n  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1265669\n\n  Args:\n    x_start: A real `Tensor` of any shape and dtype.\n    x_end: A real `Tensor` of the same dtype and compatible shape as\n      `x_start`.\n    variance: A real `Tensor` of the same dtype and compatible shape as\n      `x_start`.\n    upper_barrier: A scalar `Tensor` of the same dtype as `x_start`. Stands for\n      the upper boundary for the Brownian Bridge.\n    lower_barrier: A scalar `Tensor` of the same dtype as `x_start`. Stands for\n      lower the boundary for the Brownian Bridge.\n    n_cutoff: A positive scalar int32 `Tensor`. This controls when to cutoff\n      the sum which would otherwise have an infinite number of terms.\n      Default value: 3.\n    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion\n      of any supplied non-`Tensor` arguments to `Tensor`.\n      Default value: None which maps to the default dtype inferred by\n      TensorFlow.\n    name: str. The name for the ops created by this function.\n      Default value: None which is mapped to the default name\n      `brownian_bridge_double`.\n\n  Returns:\n      A `Tensor` of the same shape as the input data which is the probability\n      of not touching the upper and lower barrier.\n  "
    with tf.name_scope(name or 'brownian_bridge_double'):
        x_start = tf.convert_to_tensor(x_start, dtype=dtype, name='x_start')
        dtype = x_start.dtype
        variance = tf.convert_to_tensor(variance, dtype=dtype, name='variance')
        x_end = tf.convert_to_tensor(x_end, dtype=dtype, name='x_end')
        barrier_diff = upper_barrier - lower_barrier
        x_start = tf.expand_dims(x_start, -1)
        x_end = tf.expand_dims(x_end, -1)
        variance = tf.expand_dims(variance, -1)
        k = tf.expand_dims(tf.range(-n_cutoff, n_cutoff + 1, dtype=dtype), 0)
        a = k * barrier_diff * (k * barrier_diff + (x_end - x_start))
        b = k * barrier_diff + x_start - upper_barrier
        b *= k * barrier_diff + (x_end - upper_barrier)
        output = tf.math.exp(-2 * a / variance) - tf.math.exp(-2 * b / variance)
        return tf.reduce_sum(output, axis=-1)

def brownian_bridge_single(*, x_start, x_end, variance, barrier, dtype=None, name=None):
    if False:
        while True:
            i = 10
    'Computes proba of not touching the barrier for a 1D Brownian Bridge.\n\n  The Brownian bridge starts at `x_start`, ends at `x_end` and has a variance\n  `variance`. The no-touch probabilities are calculated assuming that `x_start`\n  and `x_end` are the same side of the barrier (either both above or both\n  below).\n  This can be used in Monte Carlo pricing for adjusting probability of\n  touching the barrier from discrete case to continuous case.\n  Typically in practise, the tensors `x_start`, `x_end` and `variance` should be\n  bi-dimensional (with time steps and paths being the 2 dimensions).\n\n  #### Example\n\n  ```python\n  x_start = np.asarray([[4.5, 4.5, 4.5], [4.5, 4.6, 4.7]])\n  x_end = np.asarray([[5.0, 4.9, 4.8], [4.8, 4.9, 5.0]])\n  variance = np.asarray([[0.1, 0.2, 0.1], [0.3, 0.1, 0.2]])\n  barrier = 5.1\n\n  no_touch_proba = brownian_bridge_single(\n    x_start=x_start,\n    x_end=x_end,\n    variance=variance,\n    barrier=barrier)\n  # Expected print output of no_touch_proba:\n  # [[0.69880579 0.69880579 0.97267628]\n  #  [0.69880579 0.86466472 0.32967995]]\n  ```\n\n  #### References\n\n  [1] Emmanuel Gobet. Advanced Monte Carlo methods for barrier and related\n  exotic options.\n  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1265669\n\n  Args:\n    x_start: A real `Tensor` of any shape and dtype.\n    x_end: A real `Tensor` of the same dtype and compatible shape as\n      `x_start`.\n    variance: A real `Tensor` of the same dtype and compatible shape as\n      `x_start`.\n    barrier: A scalar `Tensor` of the same dtype as `x_start`. Stands for the\n      boundary for the Brownian Bridge.\n    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion\n      of any supplied non-`Tensor` arguments to `Tensor`.\n      Default value: None which maps to the default dtype inferred by\n      TensorFlow.\n    name: str. The name for the ops created by this function.\n      Default value: None which is mapped to the default name\n      `brownian_bridge_single`.\n\n  Returns:\n      A `Tensor` of the same shape as the input data which is the probability\n      of not touching the barrier.\n  '
    with tf.name_scope(name or 'brownian_bridge_single'):
        x_start = tf.convert_to_tensor(x_start, dtype=dtype, name='x_start')
        dtype = x_start.dtype
        variance = tf.convert_to_tensor(variance, dtype=dtype, name='variance')
        x_end = tf.convert_to_tensor(x_end, dtype=dtype, name='x_end')
        a = (x_start - barrier) * (x_end - barrier)
        return 1 - tf.math.exp(-2 * a / variance)