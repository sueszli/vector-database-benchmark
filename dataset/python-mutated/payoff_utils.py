"""Payoff functions."""
import functools
from typing import Callable
import tensorflow.compat.v2 as tf
from tf_quant_finance import types
__all__ = ['make_basket_put_payoff']

def make_basket_put_payoff(strikes: types.RealTensor, dtype: tf.DType=None, name: str=None) -> Callable[[types.RealTensor], types.RealTensor]:
    if False:
        i = 10
        return i + 15
    "Produces a callable from samples to payoff of a simple basket put option.\n\n  Args:\n    strikes: A `Tensor` of `dtype` consistent with `samples` and shape\n      `[num_samples, batch_size]`.\n    dtype: Optional `dtype`. Either `tf.float32` or `tf.float64`. If supplied,\n      represents the `dtype` for the 'strikes' as well as for the input\n      argument of the output payoff callable.\n      Default value: `None`, which means that the `dtype` inferred from\n      `strikes` is used.\n    name: Python `str` name prefixed to Ops created by the callable created\n      by this function.\n      Default value: `None` which is mapped to the default name 'put_valuer'\n\n  Returns:\n    A callable from `Tensor` of shape\n    `[batch_size, num_samples, num_exercise_times, dim]`\n    and a scalar `Tensor` representing current time to a `Tensor` of shape\n    `[num_samples, batch_size]`.\n  "
    name = name or 'put_valuer'
    with tf.name_scope(name):
        strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
        dtype = dtype or strikes.dtype
        put_valuer = functools.partial(_put_valuer, strikes=strikes, dtype=dtype)
    return put_valuer

def _put_valuer(sample_paths, time_index, strikes, dtype=None):
    if False:
        while True:
            i = 10
    "Produces a callable from samples to payoff of a simple basket put option.\n\n  Args:\n    sample_paths: A `Tensor` of either `float32` or `float64` dtype and of\n      either shape `[num_samples, num_times, dim]` or\n      `[batch_size, num_samples, num_times, dim]`.\n    time_index: An integer scalar `Tensor` that corresponds to the time\n      coordinate at which the basis function is computed.\n    strikes: A `Tensor` of the same `dtype` as `sample_paths` and shape\n      compatible with `[num_samples, batch_size]`.\n    dtype: Optional `dtype`. Either `tf.float32` or `tf.float64`. The `dtype`\n      If supplied, represents the `dtype` for the 'strikes' as well as\n      for the input argument of the output payoff callable.\n      Default value: `None`, which means that the `dtype` inferred by TensorFlow\n      is used.\n  Returns:\n    A callable from `Tensor` of shape `sample_paths.shape`\n    and a scalar `Tensor` representing current time to a `Tensor` of shape\n    `[num_samples, batch_size]`.\n  "
    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    sample_paths = tf.convert_to_tensor(sample_paths, dtype=dtype, name='sample_paths')
    if sample_paths.shape.rank == 3:
        sample_paths = tf.expand_dims(sample_paths, axis=1)
    else:
        sample_paths = tf.transpose(sample_paths, [1, 0, 2, 3])
    (num_samples, batch_size, _, dim) = sample_paths.shape.as_list()
    slice_sample_paths = tf.slice(sample_paths, [0, 0, time_index, 0], [num_samples, batch_size, 1, dim])
    slice_sample_paths = tf.squeeze(slice_sample_paths, 2)
    average = tf.math.reduce_mean(slice_sample_paths, axis=-1)
    return tf.nn.relu(strikes - average)