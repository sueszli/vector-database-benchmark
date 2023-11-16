"""Collection of utility functions for pricing options on zero coupon bonds."""
from typing import Callable, Tuple
import tensorflow.compat.v2 as tf
from tf_quant_finance import types
from tf_quant_finance import utils as tff_utils
from tf_quant_finance.models import utils
__all__ = ['options_price_from_samples']

def options_price_from_samples(strikes: types.RealTensor, expiries: types.RealTensor, maturities: types.RealTensor, is_call_options: types.BoolTensor, sample_discount_curve_paths_fn: Callable[..., Tuple[types.RealTensor, types.RealTensor]], num_samples: types.IntTensor, time_step: types.RealTensor, dtype: tf.DType=None, name: str=None) -> types.RealTensor:
    if False:
        print('Hello World!')
    'Computes the zero coupon bond options price from simulated discount curves.\n\n  Args:\n    strikes: A real `Tensor` of any shape and dtype. The strike price of the\n      options. The shape of this input determines the number (and shape) of the\n      options to be priced and the output.\n    expiries: A real `Tensor` of the same dtype and compatible shape as\n      `strikes`.  The time to expiry of each bond option.\n    maturities: A real `Tensor` of the same dtype and compatible shape as\n      `strikes`.  The time to maturity of the underlying zero coupon bonds.\n    is_call_options: A boolean `Tensor` of a shape compatible with `strikes`.\n      Indicates whether the option is a call (if True) or a put (if False).\n    sample_discount_curve_paths_fn: Callable which takes the following args:\n\n      1) times: Rank 1 `Tensor` of positive real values, specifying the times at\n        which the path points are to be evaluated.\n      2) curve_times: Rank 1 `Tensor` of positive real values, specifying the\n        maturities at which the discount curve is to be computed at each\n        simulation time.\n      3) num_samples: Positive scalar integer specifying the number of paths to\n        draw.\n\n      and returns two `Tensor`s, the first being a Rank-4 tensor of shape\n      `[num_samples, m, k, dim]` containing the simulated zero coupon bond\n      curves, and the second being a `Tensor` of shape `[num_samples, k, dim]`\n      containing the simulated short rate paths. Here, `m` is the size of\n      `curve_times`, `k` is the size of `times`, and `dim` is the dimensionality\n      of the paths.\n\n    num_samples: Positive scalar `int32` `Tensor`. The number of simulation\n      paths during Monte-Carlo valuation.\n    time_step: Scalar real `Tensor`. Maximal distance between time grid points\n      in Euler scheme. Relevant when Euler scheme is used for simulation.\n    dtype: The default dtype to use when converting values to `Tensor`s.\n      Default value: `None` which means that default dtypes inferred by\n        TensorFlow are used.\n    name: Python string. The name to give to the ops created by this function.\n      Default value: `None` which maps to the default name\n      `options_price_from_samples`.\n\n  Returns:\n    A `Tensor` of real dtype and shape `strikes.shape + [dim]` containing the\n    computed option prices.\n  '
    name = name or 'options_price_from_samples'
    with tf.name_scope(name):
        (sim_times, _) = tf.unique(tf.reshape(expiries, shape=[-1]))
        longest_expiry = tf.reduce_max(sim_times)
        (sim_times, _) = tf.unique(tf.concat([sim_times, tf.range(time_step, longest_expiry, time_step)], axis=0))
        sim_times = tf.sort(sim_times, name='sort_sim_times')
        tau = maturities - expiries
        (curve_times_builder, _) = tf.unique(tf.reshape(tau, shape=[-1]))
        curve_times = tf.sort(curve_times_builder, name='sort_curve_times')
        (p_t_tau, r_t) = sample_discount_curve_paths_fn(times=sim_times, curve_times=curve_times, num_samples=num_samples)
        dim = p_t_tau.shape[-1]
        dt_builder = tf.concat(axis=0, values=[tf.convert_to_tensor([0.0], dtype=dtype), sim_times[1:] - sim_times[:-1]])
        dt = tf.expand_dims(tf.expand_dims(dt_builder, axis=-1), axis=0)
        discount_factors_builder = tf.math.exp(-r_t * dt)
        discount_factors_builder = tf.transpose(utils.cumprod_using_matvec(tf.transpose(discount_factors_builder, [0, 2, 1])), [0, 2, 1])
        discount_factors_builder = tf.expand_dims(discount_factors_builder, axis=1)
        discount_factors_simulated = tf.repeat(discount_factors_builder, p_t_tau.shape.as_list()[1], axis=1)
        sim_time_index = tf.searchsorted(sim_times, tf.reshape(expiries, [-1]))
        curve_time_index = tf.searchsorted(curve_times, tf.reshape(tau, [-1]))
        (curve_time_index, sim_time_index) = tff_utils.broadcast_tensors(curve_time_index, sim_time_index)
        gather_index = _prepare_indices(tf.range(0, num_samples), curve_time_index, sim_time_index, tf.range(0, dim))
        payoff_discount_factors_builder = tf.gather_nd(discount_factors_simulated, gather_index)
        payoff_discount_factors = tf.reshape(payoff_discount_factors_builder, [num_samples] + strikes.shape + [dim])
        payoff_bond_price_builder = tf.gather_nd(p_t_tau, gather_index)
        payoff_bond_price = tf.reshape(payoff_bond_price_builder, [num_samples] + strikes.shape + [dim])
        is_call_options = tf.reshape(tf.broadcast_to(is_call_options, strikes.shape), [1] + strikes.shape + [1])
        strikes = tf.reshape(strikes, [1] + strikes.shape + [1])
        payoff = tf.where(is_call_options, tf.math.maximum(payoff_bond_price - strikes, 0.0), tf.math.maximum(strikes - payoff_bond_price, 0.0))
        option_value = tf.math.reduce_mean(payoff_discount_factors * payoff, axis=0)
        return option_value

def _prepare_indices(idx0, idx1, idx2, idx3):
    if False:
        print('Hello World!')
    'Prepare indices to get relevant slice from discount curve simulations.'
    len0 = idx0.shape.as_list()[0]
    len1 = idx1.shape.as_list()[0]
    len3 = idx3.shape.as_list()[0]
    idx0 = tf.repeat(idx0, len1 * len3)
    idx1 = tf.tile(tf.repeat(idx1, len3), [len0])
    idx2 = tf.tile(tf.repeat(idx2, len3), [len0])
    idx3 = tf.tile(idx3, [len0 * len1])
    return tf.stack([idx0, idx1, idx2, idx3], axis=-1)