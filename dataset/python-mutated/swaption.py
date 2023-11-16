"""Pricing of Interest rate Swaptions using the Hull-White model."""
from typing import Callable, Union
import numpy as np
import tensorflow.compat.v2 as tf
from tf_quant_finance import types
from tf_quant_finance.math import pde
from tf_quant_finance.math import random
from tf_quant_finance.math.interpolation import linear
from tf_quant_finance.math.root_search import brent
from tf_quant_finance.models import utils
from tf_quant_finance.models.hjm import swaption_util
from tf_quant_finance.models.hull_white import one_factor
from tf_quant_finance.models.hull_white import zero_coupon_bond_option as zcb
from tf_quant_finance.models.longstaff_schwartz import lsm
__all__ = ['swaption_price', 'bermudan_swaption_price']
_PDE_TIME_GRID_TOL = 1e-07

def swaption_price(*, expiries: types.RealTensor, floating_leg_start_times: types.RealTensor, floating_leg_end_times: types.RealTensor, fixed_leg_payment_times: types.RealTensor, floating_leg_daycount_fractions: types.RealTensor, fixed_leg_daycount_fractions: types.RealTensor, fixed_leg_coupon: types.RealTensor, reference_rate_fn: Callable[..., types.RealTensor], mean_reversion: Union[types.RealTensor, Callable[..., types.RealTensor]], volatility: Union[types.RealTensor, Callable[..., types.RealTensor]], notional: types.RealTensor=None, is_payer_swaption: types.BoolTensor=True, use_analytic_pricing: bool=True, num_samples: types.IntTensor=100, random_type: random.RandomType=None, seed: types.IntTensor=None, skip: types.IntTensor=0, time_step: types.RealTensor=None, dtype: tf.DType=None, name: str=None) -> types.RealTensor:
    if False:
        while True:
            i = 10
    "Calculates the price of European Swaptions using the Hull-White model.\n\n  A European Swaption is a contract that gives the holder an option to enter a\n  swap contract at a future date at a prespecified fixed rate. A swaption that\n  grants the holder to pay fixed rate and receive floating rate is called a\n  payer swaption while the swaption that grants the holder to receive fixed and\n  pay floating payments is called the receiver swaption. Typically the start\n  date (or the inception date) of the swap concides with the expiry of the\n  swaption. Mid-curve swaptions are currently not supported (b/160061740).\n\n  Analytic pricing of swaptions is performed using the Jamshidian decomposition\n  [1].\n\n  #### References:\n    [1]: D. Brigo, F. Mercurio. Interest Rate Models-Theory and Practice.\n    Second Edition. 2007.\n\n  #### Example\n  The example shows how value a batch of 1y x 1y and 1y x 2y swaptions using the\n  Hull-White model.\n\n  ````python\n  import numpy as np\n  import tensorflow.compat.v2 as tf\n  import tf_quant_finance as tff\n\n  dtype = tf.float64\n\n  expiries = [1.0, 1.0]\n  float_leg_start_times = [[1.0, 1.25, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0],\n                            [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]]\n  float_leg_end_times = [[1.25, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0, 2.0],\n                          [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]]\n  fixed_leg_payment_times = [[1.25, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0, 2.0],\n                          [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]]\n  float_leg_daycount_fractions = [[0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0],\n                              [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]]\n  fixed_leg_daycount_fractions = [[0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0],\n                              [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]]\n  fixed_leg_coupon = [[0.011, 0.011, 0.011, 0.011, 0.0, 0.0, 0.0, 0.0],\n                      [0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011]]\n  zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)\n  price = tff.models.hull_white.swaption_price(\n      expiries=expiries,\n      floating_leg_start_times=float_leg_start_times,\n      floating_leg_end_times=float_leg_end_times,\n      fixed_leg_payment_times=fixed_leg_payment_times,\n      floating_leg_daycount_fractions=float_leg_daycount_fractions,\n      fixed_leg_daycount_fractions=fixed_leg_daycount_fractions,\n      fixed_leg_coupon=fixed_leg_coupon,\n      reference_rate_fn=zero_rate_fn,\n      notional=100.,\n      dim=1,\n      mean_reversion=[0.03],\n      volatility=[0.02],\n      dtype=dtype)\n  # Expected value: [0.7163243383624043, 1.4031415262337608] # shape = (2,1)\n  ````\n\n  Args:\n    expiries: A real `Tensor` of any shape and dtype. The time to\n      expiration of the swaptions. The shape of this input determines the number\n      (and shape) of swaptions to be priced and the shape of the output.\n    floating_leg_start_times: A real `Tensor` of the same dtype as `expiries`.\n      The times when accrual begins for each payment in the floating leg. The\n      shape of this input should be `expiries.shape + [m]` where `m` denotes\n      the number of floating payments in each leg.\n    floating_leg_end_times: A real `Tensor` of the same dtype as `expiries`.\n      The times when accrual ends for each payment in the floating leg. The\n      shape of this input should be `expiries.shape + [m]` where `m` denotes\n      the number of floating payments in each leg.\n    fixed_leg_payment_times: A real `Tensor` of the same dtype as `expiries`.\n      The payment times for each payment in the fixed leg. The shape of this\n      input should be `expiries.shape + [n]` where `n` denotes the number of\n      fixed payments in each leg.\n    floating_leg_daycount_fractions: A real `Tensor` of the same dtype and\n      compatible shape as `floating_leg_start_times`. The daycount fractions\n      for each payment in the floating leg.\n    fixed_leg_daycount_fractions: A real `Tensor` of the same dtype and\n      compatible shape as `fixed_leg_payment_times`. The daycount fractions\n      for each payment in the fixed leg.\n    fixed_leg_coupon: A real `Tensor` of the same dtype and compatible shape\n      as `fixed_leg_payment_times`. The fixed rate for each payment in the\n      fixed leg.\n    reference_rate_fn: A Python callable that accepts expiry time as a real\n      `Tensor` and returns a `Tensor` of either shape `input_shape` or\n      `input_shape`. Returns the continuously compounded zero rate at\n      the present time for the input expiry time.\n    mean_reversion: A real positive scalar `Tensor` or a Python callable. The\n      callable can be one of the following:\n      (a) A left-continuous piecewise constant object (e.g.,\n      `tff.math.piecewise.PiecewiseConstantFunc`) that has a property\n      `is_piecewise_constant` set to `True`. In this case the object should\n      have a method `jump_locations(self)` that returns a `Tensor` of shape\n      `[num_jumps]`. The return value of `mean_reversion(t)` should return a\n      `Tensor` of shape `t.shape`, `t` is a rank 1 `Tensor` of the same `dtype`\n      as the output. See example in the class docstring.\n      (b) A callable that accepts scalars (stands for time `t`) and returns a\n      scalar `Tensor` of the same `dtype` as `strikes`.\n      Corresponds to the mean reversion rate.\n    volatility: A real positive `Tensor` of the same `dtype` as\n      `mean_reversion` or a callable with the same specs as above.\n      Corresponds to the long run price variance.\n    notional: An optional `Tensor` of same dtype and compatible shape as\n      `strikes`specifying the notional amount for the underlying swap.\n       Default value: None in which case the notional is set to 1.\n    is_payer_swaption: A boolean `Tensor` of a shape compatible with `expiries`.\n      Indicates whether the swaption is a payer (if True) or a receiver\n      (if False) swaption. If not supplied, payer swaptions are assumed.\n    use_analytic_pricing: A Python boolean specifying if analytic valuation\n      should be performed. Analytic valuation is only supported for constant\n      `mean_reversion` and piecewise constant `volatility`. If the input is\n      `False`, then valuation using Monte-Carlo simulations is performed.\n      Default value: The default value is `True`.\n    num_samples: Positive scalar `int32` `Tensor`. The number of simulation\n      paths during Monte-Carlo valuation. This input is ignored during analytic\n      valuation.\n      Default value: The default value is 1.\n    random_type: Enum value of `RandomType`. The type of (quasi)-random\n      number generator to use to generate the simulation paths. This input is\n      relevant only for Monte-Carlo valuation and ignored during analytic\n      valuation.\n      Default value: `None` which maps to the standard pseudo-random numbers.\n    seed: Seed for the random number generator. The seed is only relevant if\n      `random_type` is one of\n      `[STATELESS, PSEUDO, HALTON_RANDOMIZED, PSEUDO_ANTITHETIC,\n        STATELESS_ANTITHETIC]`. For `PSEUDO`, `PSEUDO_ANTITHETIC` and\n      `HALTON_RANDOMIZED` the seed should be an Python integer. For\n      `STATELESS` and  `STATELESS_ANTITHETIC `must be supplied as an integer\n      `Tensor` of shape `[2]`. This input is relevant only for Monte-Carlo\n      valuation and ignored during analytic valuation.\n      Default value: `None` which means no seed is set.\n    skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or\n      Halton sequence to skip. Used only when `random_type` is 'SOBOL',\n      'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.\n      Default value: `0`.\n    time_step: Scalar real `Tensor`. Maximal distance between time grid points\n      in Euler scheme. Relevant when Euler scheme is used for simulation. This\n      input is ignored during analytic valuation.\n      Default value: `None`.\n    dtype: The default dtype to use when converting values to `Tensor`s.\n      Default value: `None` which means that default dtypes inferred by\n      TensorFlow are used.\n    name: Python string. The name to give to the ops created by this function.\n      Default value: `None` which maps to the default name\n      `hw_swaption_price`.\n\n  Returns:\n    A `Tensor` of real dtype and shape  `expiries.shape` containing the\n    computed swaption prices. For swaptions that have. reset in the past\n    (expiries<0), the function sets the corresponding option prices to 0.0.\n  "
    name = name or 'hw_swaption_price'
    del floating_leg_daycount_fractions
    with tf.name_scope(name):
        expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
        dtype = dtype or expiries.dtype
        float_leg_start_times = tf.convert_to_tensor(floating_leg_start_times, dtype=dtype, name='float_leg_start_times')
        float_leg_end_times = tf.convert_to_tensor(floating_leg_end_times, dtype=dtype, name='float_leg_end_times')
        fixed_leg_payment_times = tf.convert_to_tensor(fixed_leg_payment_times, dtype=dtype, name='fixed_leg_payment_times')
        fixed_leg_daycount_fractions = tf.convert_to_tensor(fixed_leg_daycount_fractions, dtype=dtype, name='fixed_leg_daycount_fractions')
        fixed_leg_coupon = tf.convert_to_tensor(fixed_leg_coupon, dtype=dtype, name='fixed_leg_coupon')
        notional = tf.convert_to_tensor(notional, dtype=dtype, name='notional')
        is_payer_swaption = tf.convert_to_tensor(is_payer_swaption, dtype=tf.bool, name='is_payer_swaption')
        if expiries.shape.rank < fixed_leg_payment_times.shape.rank - 1:
            raise ValueError('Swaption expiries not specified for all swaptions in the batch. Expected rank {} but received {}.'.format(fixed_leg_payment_times.shape.rank - 1, expiries.shape.rank))
        expiries = tf.expand_dims(expiries, axis=-1)
        expiries = tf.repeat(expiries, tf.shape(fixed_leg_payment_times)[-1], axis=-1)
        if use_analytic_pricing:
            return _analytic_valuation(expiries, float_leg_start_times, float_leg_end_times, fixed_leg_payment_times, fixed_leg_daycount_fractions, fixed_leg_coupon, reference_rate_fn, mean_reversion, volatility, notional, is_payer_swaption, dtype, name + '_analytic_valuation')
        if time_step is None:
            raise ValueError('`time_step` must be provided for simulation based bond option valuation.')
        model = one_factor.HullWhiteModel1F(mean_reversion, volatility, initial_discount_rate_fn=reference_rate_fn, dtype=dtype)

        def _sample_discount_curve_path_fn(times, curve_times, num_samples):
            if False:
                while True:
                    i = 10
            (p_t_tau, r_t) = model.sample_discount_curve_paths(times=times, curve_times=curve_times, num_samples=num_samples, random_type=random_type, seed=seed, skip=skip)
            return (p_t_tau, r_t, None)
        (sim_times, _) = tf.unique(tf.reshape(expiries, shape=[-1]))
        longest_expiry = tf.reduce_max(sim_times)
        sim_times = tf.concat([sim_times, tf.range(time_step, longest_expiry, time_step)], axis=0)
        sim_times = tf.sort(sim_times, name='sort_sim_times')
        (payoff_discount_factors, payoff_bond_price) = swaption_util.discount_factors_and_bond_prices_from_samples(expiries=expiries, payment_times=fixed_leg_payment_times, sample_discount_curve_paths_fn=_sample_discount_curve_path_fn, num_samples=num_samples, times=sim_times, dtype=dtype)
        fixed_leg_pv = fixed_leg_coupon * fixed_leg_daycount_fractions * tf.squeeze(payoff_bond_price, axis=-1)
        fixed_leg_pv = tf.math.reduce_sum(fixed_leg_pv, axis=-1)
        float_leg_pv = 1.0 - tf.squeeze(payoff_bond_price, axis=-1)[..., -1]
        payoff_swap = tf.squeeze(payoff_discount_factors, axis=-1)[..., -1] * (float_leg_pv - fixed_leg_pv)
        payoff_swap = tf.where(is_payer_swaption, payoff_swap, -1.0 * payoff_swap)
        payoff_swaption = tf.math.maximum(payoff_swap, 0.0)
        option_value = tf.math.reduce_mean(payoff_swaption, axis=0)
        return notional * option_value

def bermudan_swaption_price(*, exercise_times: types.RealTensor, floating_leg_start_times: types.RealTensor, floating_leg_end_times: types.RealTensor, fixed_leg_payment_times: types.RealTensor, floating_leg_daycount_fractions: types.RealTensor, fixed_leg_daycount_fractions: types.RealTensor, fixed_leg_coupon: types.RealTensor, reference_rate_fn: Callable[..., types.RealTensor], mean_reversion: Union[types.RealTensor, Callable[..., types.RealTensor]], volatility: Union[types.RealTensor, Callable[..., types.RealTensor]], notional: types.RealTensor=None, is_payer_swaption: types.BoolTensor=True, use_finite_difference: bool=False, lsm_basis: Callable[..., types.RealTensor]=None, num_samples: types.IntTensor=100, random_type: random.RandomType=None, seed: types.IntTensor=None, skip: types.IntTensor=0, time_step: types.RealTensor=None, time_step_finite_difference: types.IntTensor=None, num_grid_points_finite_difference: types.IntTensor=101, dtype: tf.DType=None, name: str=None) -> types.RealTensor:
    if False:
        while True:
            i = 10
    "Calculates the price of Bermudan Swaptions using the Hull-White model.\n\n  A Bermudan Swaption is a contract that gives the holder an option to enter a\n  swap contract on a set of future exercise dates. The exercise dates are\n  typically the fixing dates (or a subset thereof) of the underlying swap. If\n  `T_N` denotes the final payoff date and `T_i, i = {1,...,n}` denote the set\n  of exercise dates, then if the option is exercised at `T_i`, the holder is\n  left with a swap with first fixing date equal to `T_i` and maturity `T_N`.\n\n  Simulation based pricing of Bermudan swaptions is performed using the least\n  squares Monte-carlo approach [1].\n\n  #### References:\n    [1]: D. Brigo, F. Mercurio. Interest Rate Models-Theory and Practice.\n    Second Edition. 2007.\n\n  #### Example\n  The example shows how value a batch of 5-no-call-1 and 5-no-call-2\n  swaptions using the Hull-White model.\n\n  ````python\n  import numpy as np\n  import tensorflow.compat.v2 as tf\n  import tf_quant_finance as tff\n\n  dtype = tf.float64\n\n  exercise_swaption_1 = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]\n  exercise_swaption_2 = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.0]\n  exercise_times = [exercise_swaption_1, exercise_swaption_2]\n\n  float_leg_start_times_1y = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]\n  float_leg_start_times_18m = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]\n  float_leg_start_times_2y = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.0]\n  float_leg_start_times_30m = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.0, 5.0]\n  float_leg_start_times_3y = [3.0, 3.5, 4.0, 4.5, 5.0, 5.0, 5.0, 5.0]\n  float_leg_start_times_42m = [3.5, 4.0, 4.5, 5.0, 5.0, 5.0, 5.0, 5.0]\n  float_leg_start_times_4y = [4.0, 4.5, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]\n  float_leg_start_times_54m = [4.5, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]\n  float_leg_start_times_5y = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]\n\n  float_leg_start_times_swaption_1 = [float_leg_start_times_1y,\n                                      float_leg_start_times_18m,\n                                      float_leg_start_times_2y,\n                                      float_leg_start_times_30m,\n                                      float_leg_start_times_3y,\n                                      float_leg_start_times_42m,\n                                      float_leg_start_times_4y,\n                                      float_leg_start_times_54m]\n\n  float_leg_start_times_swaption_2 = [float_leg_start_times_2y,\n                                      float_leg_start_times_30m,\n                                      float_leg_start_times_3y,\n                                      float_leg_start_times_42m,\n                                      float_leg_start_times_4y,\n                                      float_leg_start_times_54m,\n                                      float_leg_start_times_5y,\n                                      float_leg_start_times_5y]\n  float_leg_start_times = [float_leg_start_times_swaption_1,\n                         float_leg_start_times_swaption_2]\n\n  float_leg_end_times = np.clip(np.array(float_leg_start_times) + 0.5, 0.0, 5.0)\n\n  fixed_leg_payment_times = float_leg_end_times\n  float_leg_daycount_fractions = (np.array(float_leg_end_times) -\n                                  np.array(float_leg_start_times))\n  fixed_leg_daycount_fractions = float_leg_daycount_fractions\n  fixed_leg_coupon = 0.011 * np.ones_like(fixed_leg_payment_times)\n  zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)\n  price = bermudan_swaption_price(\n      exercise_times=exercise_times,\n      floating_leg_start_times=float_leg_start_times,\n      floating_leg_end_times=float_leg_end_times,\n      fixed_leg_payment_times=fixed_leg_payment_times,\n      floating_leg_daycount_fractions=float_leg_daycount_fractions,\n      fixed_leg_daycount_fractions=fixed_leg_daycount_fractions,\n      fixed_leg_coupon=fixed_leg_coupon,\n      reference_rate_fn=zero_rate_fn,\n      notional=100.,\n      mean_reversion=[0.03],\n      volatility=[0.01],\n      num_samples=1000000,\n      time_step=0.1,\n      random_type=tff.math.random.RandomType.PSEUDO_ANTITHETIC,\n      seed=0,\n      dtype=dtype)\n  # Expected value: [1.8913050118443016, 1.6618681421434984] # shape = (2,)\n  ````\n\n  Args:\n    exercise_times: A real `Tensor` of any shape `[batch_size, num_exercise]`\n      `and real dtype. The times corresponding to exercise dates of the\n      swaptions. `num_exercise` corresponds to the number of exercise dates for\n      the Bermudan swaption. The shape of this input determines the number (and\n      shape) of Bermudan swaptions to be priced and the shape of the output.\n    floating_leg_start_times: A real `Tensor` of the same dtype as\n      `exercise_times`. The times when accrual begins for each payment in the\n      floating leg upon exercise of the option. The shape of this input should\n      be `exercise_times.shape + [m]` where `m` denotes the number of floating\n      payments in each leg of the underlying swap until the swap maturity.\n    floating_leg_end_times: A real `Tensor` of the same dtype as\n      `exercise_times`. The times when accrual ends for each payment in the\n      floating leg upon exercise of the option. The shape of this input should\n      be `exercise_times.shape + [m]` where `m` denotes the number of floating\n      payments in each leg of the underlying swap until the swap maturity.\n    fixed_leg_payment_times: A real `Tensor` of the same dtype as\n      `exercise_times`. The payment times for each payment in the fixed leg.\n      The shape of this input should be `exercise_times.shape + [n]` where `n`\n      denotes the number of fixed payments in each leg of the underlying swap\n      until the swap maturity.\n    floating_leg_daycount_fractions: A real `Tensor` of the same dtype and\n      compatible shape as `floating_leg_start_times`. The daycount fractions\n      for each payment in the floating leg.\n    fixed_leg_daycount_fractions: A real `Tensor` of the same dtype and\n      compatible shape as `fixed_leg_payment_times`. The daycount fractions\n      for each payment in the fixed leg.\n    fixed_leg_coupon: A real `Tensor` of the same dtype and compatible shape\n      as `fixed_leg_payment_times`. The fixed rate for each payment in the\n      fixed leg.\n    reference_rate_fn: A Python callable that accepts expiry time as a real\n      `Tensor` and returns a `Tensor` of either shape `input_shape` or\n      `input_shape`. Returns the continuously compounded zero rate at\n      the present time for the input expiry time.\n    mean_reversion: A real positive scalar `Tensor` or a Python callable. The\n      callable can be one of the following:\n      (a) A left-continuous piecewise constant object (e.g.,\n      `tff.math.piecewise.PiecewiseConstantFunc`) that has a property\n      `is_piecewise_constant` set to `True`. In this case the object should\n      have a method `jump_locations(self)` that returns a `Tensor` of shape\n      `[num_jumps]`. The return value of `mean_reversion(t)` should return a\n      `Tensor` of shape `t.shape`, `t` is a rank 1 `Tensor` of the same `dtype`\n      as the output. See example in the class docstring.\n      (b) A callable that accepts scalars (stands for time `t`) and returns a\n      scalar `Tensor` of the same `dtype` as `strikes`.\n      Corresponds to the mean reversion rate.\n    volatility: A real positive `Tensor` of the same `dtype` as\n      `mean_reversion` or a callable with the same specs as above.\n      Corresponds to the long run price variance.\n    notional: An optional `Tensor` of same dtype and compatible shape as\n      `strikes`specifying the notional amount for the underlying swap.\n       Default value: None in which case the notional is set to 1.\n    is_payer_swaption: A boolean `Tensor` of a shape compatible with `expiries`.\n      Indicates whether the swaption is a payer (if True) or a receiver\n      (if False) swaption. If not supplied, payer swaptions are assumed.\n    use_finite_difference: A Python boolean specifying if the valuation should\n      be performed using the finite difference and PDE.\n      Default value: `False`, in which case valuation is performed using least\n      squares monte-carlo method.\n    lsm_basis: A Python callable specifying the basis to be used in the LSM\n      algorithm. The callable must accept a `Tensor`s of shape\n      `[num_samples, dim]` and output `Tensor`s of shape `[m, num_samples]`\n      where `m` is the nimber of basis functions used. This input is only used\n      for valuation using LSM.\n      Default value: `None`, in which case a polynomial basis of order 2 is\n      used.\n    num_samples: Positive scalar `int32` `Tensor`. The number of simulation\n      paths during Monte-Carlo valuation. This input is only used for valuation\n      using LSM.\n      Default value: The default value is 100.\n    random_type: Enum value of `RandomType`. The type of (quasi)-random\n      number generator to use to generate the simulation paths. This input is\n      only used for valuation using LSM.\n      Default value: `None` which maps to the standard pseudo-random numbers.\n    seed: Seed for the random number generator. The seed is only relevant if\n      `random_type` is one of\n      `[STATELESS, PSEUDO, HALTON_RANDOMIZED, PSEUDO_ANTITHETIC,\n        STATELESS_ANTITHETIC]`. For `PSEUDO`, `PSEUDO_ANTITHETIC` and\n      `HALTON_RANDOMIZED` the seed should be an Python integer. For\n      `STATELESS` and  `STATELESS_ANTITHETIC `must be supplied as an integer\n      `Tensor` of shape `[2]`. This input is only used for valuation using LSM.\n      Default value: `None` which means no seed is set.\n    skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or\n      Halton sequence to skip. Used only when `random_type` is 'SOBOL',\n      'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored. This input is only\n      used for valuation using LSM.\n      Default value: `0`.\n    time_step: Scalar real `Tensor`. Maximal distance between time grid points\n      in Euler scheme. Relevant when Euler scheme is used for simulation.\n      This input is only used for valuation using LSM.\n      Default value: `None`.\n    time_step_finite_difference: Scalar real `Tensor`. Spacing between time\n      grid points in finite difference discretization. This input is only\n      relevant for valuation using finite difference.\n      Default value: `None`, in which case a `time_step` corresponding to 100\n      discrete steps is used.\n    num_grid_points_finite_difference: Scalar real `Tensor`. Number of spatial\n      grid points for discretization. This input is only relevant for valuation\n      using finite difference.\n      Default value: 100.\n    dtype: The default dtype to use when converting values to `Tensor`s.\n      Default value: `None` which means that default dtypes inferred by\n      TensorFlow are used.\n    name: Python string. The name to give to the ops created by this function.\n      Default value: `None` which maps to the default name\n      `hw_bermudan_swaption_price`.\n\n  Returns:\n    A `Tensor` of real dtype and shape  `[batch_size]` containing the\n    computed swaption prices.\n\n  Raises:\n    (a) `ValueError` if exercise_times.rank is less than\n    floating_leg_start_times.rank - 1, which would mean exercise times are not\n    specified for all swaptions.\n    (b) `ValueError` if `time_step` is not specified for Monte-Carlo\n    simulations.\n  "
    name = name or 'hw_bermudan_swaption_price'
    del floating_leg_daycount_fractions, floating_leg_start_times
    del floating_leg_end_times
    with tf.name_scope(name):
        exercise_times = tf.convert_to_tensor(exercise_times, dtype=dtype, name='exercise_times')
        dtype = dtype or exercise_times.dtype
        fixed_leg_payment_times = tf.convert_to_tensor(fixed_leg_payment_times, dtype=dtype, name='fixed_leg_payment_times')
        fixed_leg_daycount_fractions = tf.convert_to_tensor(fixed_leg_daycount_fractions, dtype=dtype, name='fixed_leg_daycount_fractions')
        fixed_leg_coupon = tf.convert_to_tensor(fixed_leg_coupon, dtype=dtype, name='fixed_leg_coupon')
        notional = tf.convert_to_tensor(notional, dtype=dtype, name='notional')
        is_payer_swaption = True
        is_payer_swaption = tf.convert_to_tensor(is_payer_swaption, dtype=tf.bool, name='is_payer_swaption')
        if lsm_basis is None:
            basis_fn = lsm.make_polynomial_basis(2)
        else:
            basis_fn = lsm_basis
        batch_shape = exercise_times.shape.as_list()[:-1]
        (unique_exercise_times, exercise_time_index) = tf.unique(tf.reshape(exercise_times, shape=[-1]))
        exercise_time_index = tf.reshape(exercise_time_index, shape=exercise_times.shape)
        if exercise_times.shape.rank < fixed_leg_payment_times.shape.rank - 1:
            raise ValueError('Swaption exercise times not specified for all swaptions in the batch. Expected rank {} but received {}.'.format(fixed_leg_payment_times.shape.rank - 1, exercise_times.shape.rank))
        exercise_times = tf.expand_dims(exercise_times, axis=-1)
        exercise_times = tf.repeat(exercise_times, tf.shape(fixed_leg_payment_times)[-1], axis=-1)
        model = one_factor.HullWhiteModel1F(mean_reversion, volatility, initial_discount_rate_fn=reference_rate_fn, dtype=dtype)
        if use_finite_difference:
            return _bermudan_swaption_fd(batch_shape, model, exercise_times, unique_exercise_times, fixed_leg_payment_times, fixed_leg_daycount_fractions, fixed_leg_coupon, notional, is_payer_swaption, time_step_finite_difference, num_grid_points_finite_difference, name + '_fd', dtype)
        if time_step is None:
            raise ValueError('`time_step` must be provided for LSM valuation.')
        sim_times = unique_exercise_times
        longest_exercise_time = sim_times[-1]
        (sim_times, _) = tf.unique(tf.concat([sim_times, tf.range(time_step, longest_exercise_time, time_step)], axis=0))
        sim_times = tf.sort(sim_times, name='sort_sim_times')
        maturities = fixed_leg_payment_times
        maturities_shape = maturities.shape
        tau = maturities - exercise_times
        (curve_times_builder, _) = tf.unique(tf.reshape(tau, shape=[-1]))
        curve_times = tf.sort(curve_times_builder, name='sort_curve_times')
        (p_t_tau, r_t) = model.sample_discount_curve_paths(times=sim_times, curve_times=curve_times, num_samples=num_samples, random_type=random_type, seed=seed, skip=skip)
        dt = tf.concat([tf.convert_to_tensor([0.0], dtype=dtype), sim_times[1:] - sim_times[:-1]], axis=0)
        dt = tf.expand_dims(tf.expand_dims(dt, axis=-1), axis=0)
        discount_factors_builder = tf.math.exp(-r_t * dt)
        discount_factors_builder = tf.transpose(utils.cumprod_using_matvec(tf.transpose(discount_factors_builder, [0, 2, 1])), [0, 2, 1])
        discount_factors_builder = tf.expand_dims(discount_factors_builder, axis=1)
        discount_factors_simulated = tf.repeat(discount_factors_builder, tf.shape(p_t_tau)[1], axis=1)
        sim_time_index = tf.searchsorted(sim_times, tf.reshape(exercise_times, [-1]))
        curve_time_index = tf.searchsorted(curve_times, tf.reshape(tau, [-1]))
        gather_index = _prepare_indices_ijjk(tf.range(0, num_samples), curve_time_index, sim_time_index, tf.range(0, 1))
        payoff_bond_price_builder = tf.gather_nd(p_t_tau, gather_index)
        payoff_bond_price = tf.reshape(payoff_bond_price_builder, [num_samples] + maturities_shape + [1])
        fixed_leg_pv = tf.expand_dims(fixed_leg_coupon * fixed_leg_daycount_fractions, axis=-1) * payoff_bond_price
        fixed_leg_pv = tf.math.reduce_sum(fixed_leg_pv, axis=-2)
        float_leg_pv = 1.0 - payoff_bond_price[..., -1, :]
        payoff_swap = float_leg_pv - fixed_leg_pv
        payoff_swap = tf.where(is_payer_swaption, payoff_swap, -1.0 * payoff_swap)
        sim_time_index = tf.searchsorted(sim_times, unique_exercise_times)
        short_rate = tf.gather(r_t, sim_time_index, axis=1)
        (is_exercise_time, payoff_swap) = _map_payoff_to_sim_times(exercise_time_index, payoff_swap, num_samples)
        perm = [is_exercise_time.shape.rank - 1] + list(range(is_exercise_time.shape.rank - 1))
        is_exercise_time = tf.transpose(is_exercise_time, perm=perm)
        payoff_swap = tf.transpose(payoff_swap, perm=perm)

        def _payoff_fn(rt, time_index):
            if False:
                for i in range(10):
                    print('nop')
            del rt
            result = tf.where(is_exercise_time[time_index] > 0, tf.nn.relu(payoff_swap[time_index]), 0.0)
            if batch_shape:
                return result
            else:
                return tf.expand_dims(result, axis=-1)
        discount_factors_simulated = tf.gather(discount_factors_simulated, sim_time_index, axis=2)
        option_value = lsm.least_square_mc(short_rate, tf.range(0, tf.shape(short_rate)[1]), _payoff_fn, basis_fn, discount_factors=discount_factors_simulated[:, -1:, :, 0], dtype=dtype)
        option_value = notional * option_value
        if batch_shape:
            return option_value
        else:
            return tf.squeeze(option_value)

def _jamshidian_decomposition(hw_model, expiries, maturities, coefficients, dtype, name=None):
    if False:
        while True:
            i = 10
    'Jamshidian decomposition for European swaption valuation.\n\n  Jamshidian decomposition is a widely used technique for the valuation of\n  European swaptions (and options on coupon bearing bonds) when the underlying\n  models for the term structure are short rate models (such as Hull-White\n  model). The method transforms the swaption valuation to the valuation of a\n  portfolio of call (put) options on zero-coupon bonds.\n\n  Consider the following swaption payoff(assuming unit notional) at the\n  exipration time (under a single curve valuation):\n\n  ```None\n  payoff = max(1 - P(T0, TN, r(T0)) - sum_1^N tau_i * X_i * P(T0, Ti, r(T0)), 0)\n         = max(1 - sum_0^N alpha_i * P(T0, Ti, r(T0)), 0)\n  ```\n\n  where `T0` denotes the swaption expiry, P(T0, Ti, r(T0)) denotes the price\n  of the zero coupon bond at `T0` with maturity `Ti` and `r(T0)` is the short\n  rate at time `T0`. If `r*` (or breakeven short rate) is the solution of the\n  following equation:\n\n  ```None\n  1 - sum_0^N alpha_i * P(T0, Ti, r*) = 0            (1)\n  ```\n\n  Then the swaption payoff can be expressed as the following (Ref. [1]):\n\n  ```None\n  payoff = sum_1^N alpha_i max(P(T0, Ti, r*) - P(T0, Ti), 0)\n  ```\n  where in the above formulation the swaption payoff is the same as that of\n  a portfolio of bond options with strikes `P(T0, Ti, r*)`.\n\n  The function accepts relevant inputs for the above computation and returns\n  the strikes of the bond options computed using the Jamshidian decomposition.\n\n  #### References:\n    [1]: Leif B. G. Andersen and Vladimir V. Piterbarg. Interest Rate Modeling.\n    Volume II: Term Structure Models. Chapter 10.\n\n\n  Args:\n    hw_model: An instance of `VectorHullWhiteModel`. The model used for the\n      valuation.\n    expiries: A real `Tensor` of shape `batch_shape + [n]`, where `n` denotes\n      the number of payments in the fixed leg of the underlying swaps. The time\n      to expiration of the swaptions.\n    maturities: A real `Tensor` of the same shape and dtype as `expiries`. The\n      payment times for fixed payments of the underlying swaptions.\n    coefficients: A real `Tensor` of the same shape and dtype as `expiries`\n    dtype: The default dtype to use when converting values to `Tensor`s.\n    name: Python string. The name to give to the ops created by this function.\n      Default value: `None` which maps to the default name\n      `jamshidian_decomposition`.\n\n  Returns:\n    A real `Tensor` of shape `expiries.shape + [1]` containing the forward bond\n    prices computed at the breakeven short rate using the Jamshidian\n    decomposition.\n  '
    with tf.name_scope(name):
        coefficients = tf.expand_dims(coefficients, axis=-1)

        def _zero_fun(x):
            if False:
                while True:
                    i = 10
            p_t0_t = hw_model.discount_bond_price(x, expiries, maturities)
            return_value = tf.reduce_sum(coefficients * p_t0_t, axis=-2, keepdims=True) + [1.0]
            return return_value
        swap_shape = expiries.shape.as_list()[:-1] + [1] + [1]
        lower_bound = -1 * tf.ones(swap_shape, dtype=dtype)
        upper_bound = 1 * tf.ones(swap_shape, dtype=dtype)
        brent_results = brent.brentq(_zero_fun, lower_bound, upper_bound)
        breakeven_short_rate = brent_results.estimated_root
        return hw_model.discount_bond_price(breakeven_short_rate, expiries, maturities)

def _prepare_swaption_indices(tensor_shape):
    if False:
        i = 10
        return i + 15
    'Indices for `gather_nd` for analytic valuation.\n\n  For a `Tensor` x of shape `tensor_shape` = [n] + batch_shape + [n], this\n  function returns indices for tf.gather_nd to get `x[i,...,i]`\n\n  Args:\n    tensor_shape: A list of length `k` representing shape of the `Tensor`.\n\n  Returns:\n    A `Tensor` of shape (num_elements, k) where num_elements= n * batch_size\n    of dtype tf.int64.\n  '
    tensor_shape = np.array(tensor_shape, dtype=np.int64)
    batch_shape = tensor_shape[1:-1]
    batch_size = np.prod(batch_shape)
    index_list = []
    for i in range(len(tensor_shape)):
        index = np.arange(0, tensor_shape[i], dtype=np.int64)
        if i == 0 or i == len(tensor_shape) - 1:
            index = tf.tile(index, [batch_size])
        else:
            index = np.tile(np.repeat(index, np.prod(tensor_shape[i + 1:])), [np.prod(tensor_shape[1:i])])
        index_list.append(index)
    return tf.stack(index_list, axis=-1)

def _prepare_indices_ijjk(idx0, idx1, idx2, idx3):
    if False:
        i = 10
        return i + 15
    'Prepares indices to get x[i, j, j, k].'
    len0 = tf.shape(idx0)[0]
    len1 = tf.shape(idx1)[0]
    len3 = tf.shape(idx3)[0]
    idx0 = tf.repeat(idx0, len1 * len3)
    idx1 = tf.tile(tf.repeat(idx1, len3), [len0])
    idx2 = tf.tile(tf.repeat(idx2, len3), [len0])
    idx3 = tf.tile(idx3, [len0 * len1])
    return tf.stack([idx0, idx1, idx2, idx3], axis=-1)

def _prepare_indices_ijj(idx0, idx1, idx2):
    if False:
        i = 10
        return i + 15
    'Prepares indices to get x[i, j, j].'
    len0 = tf.shape(idx0)[0]
    len1 = tf.shape(idx1)[0]
    idx0 = tf.repeat(idx0, len1)
    idx1 = tf.tile(idx1, [len0])
    idx2 = tf.tile(idx2, [len0])
    return tf.stack([idx0, idx1, idx2], axis=-1)

def _map_payoff_to_sim_times(indices, payoff, num_samples):
    if False:
        return 10
    "Maps the swaption payoffs to short rate simulation times.\n\n  Swaption payoffs are calculated on bermudan swaption's expiries. However, for\n  the LSM algorithm, we need short rate simulations and swaption payoffs at\n  the union of all exercise times in the batch of swaptions. This function\n  takes the payoff of individual swaption at their respective exercise times\n  and maps it to all simulation times. This is done by setting the payoff to\n  -1 whenever the simulation time is not equal to the swaption exercise time.\n\n  Args:\n    indices: A `Tensor` of shape `batch_shape + num_exercise_times` containing\n      the index of exercise time in the vector of simulation times.\n    payoff: A real tensor of shape\n      `[num_samples] + batch_shape + num_exercise_times` containing the\n      exercise value of the underlying swap on each exercise time.\n    num_samples: A scalar `Tensor` specifying the number of samples on which\n      swaption payoff is computed.\n\n  Returns:\n    A tuple of `Tensors`. The first tensor is a integer `Tensor` of shape\n    `[num_samples] + batch_shape + [num_simulation_times]` and contains `1`\n    if the corresponding simulation time is one of the exercise times for the\n    swaption. The second `Tensor` is a real `Tensor` of same shape and contains\n    the exercise value of the swaption if the corresponding simulation time is\n    an exercise time for the swaption or -1 otherwise.\n  "
    indices = tf.expand_dims(indices, axis=0)
    indices = tf.repeat(indices, num_samples, axis=0)
    index_list = list()
    tensor_shape = np.array(indices.shape.as_list())
    output_shape = indices.shape.as_list()[:-1] + [tf.math.reduce_max(indices) + 1]
    num_elements = np.prod(tensor_shape)
    for (dim, _) in enumerate(tensor_shape[:-1]):
        idx = tf.range(0, tensor_shape[dim], dtype=indices.dtype)
        idx = tf.tile(tf.repeat(idx, np.prod(tensor_shape[dim + 1:])), [np.prod(tensor_shape[:dim])])
        index_list.append(idx)
    index_list.append(tf.reshape(indices, [-1]))
    sparse_indices = tf.cast(tf.stack(index_list, axis=-1), dtype=np.int64)
    is_exercise_time = tf.sparse.to_dense(tf.sparse.SparseTensor(sparse_indices, tf.ones(shape=num_elements), output_shape), validate_indices=False)
    payoff = tf.sparse.to_dense(tf.sparse.SparseTensor(sparse_indices, tf.reshape(payoff, [-1]), output_shape), validate_indices=False)
    return (is_exercise_time, payoff)

def _analytic_valuation(expiries, floating_leg_start_times, floating_leg_end_times, fixed_leg_payment_times, fixed_leg_daycount_fractions, fixed_leg_coupon, reference_rate_fn, mean_reversion, volatility, notional, is_payer_swaption, dtype, name):
    if False:
        return 10
    'Helper function for analytic valuation.'
    del floating_leg_start_times, floating_leg_end_times
    with tf.name_scope(name):
        is_call_options = tf.where(is_payer_swaption, tf.convert_to_tensor(False, dtype=tf.bool), tf.convert_to_tensor(True, dtype=tf.bool))
        model = one_factor.HullWhiteModel1F(mean_reversion, volatility, initial_discount_rate_fn=reference_rate_fn, dtype=dtype)
        coefficients = fixed_leg_daycount_fractions * fixed_leg_coupon
        jamshidian_coefficients = tf.concat([-coefficients[..., :-1], tf.expand_dims(-1.0 - coefficients[..., -1], axis=-1)], axis=-1)
        breakeven_bond_option_strikes = _jamshidian_decomposition(model, expiries, fixed_leg_payment_times, jamshidian_coefficients, dtype, name=name + '_jamshidian_decomposition')[..., 0]
        bond_option_prices = zcb.bond_option_price(strikes=breakeven_bond_option_strikes, expiries=expiries, maturities=fixed_leg_payment_times, discount_rate_fn=reference_rate_fn, mean_reversion=mean_reversion, volatility=volatility, is_call_options=is_call_options, use_analytic_pricing=True, dtype=dtype, name=name + '_bond_option')
        swaption_values = tf.reduce_sum(bond_option_prices * coefficients, axis=-1) + bond_option_prices[..., -1]
        return notional * swaption_values

def _bermudan_swaption_fd(batch_shape, model, exercise_times, unique_exercise_times, fixed_leg_payment_times, fixed_leg_daycount_fractions, fixed_leg_coupon, notional, is_payer_swaption, time_step_fd, num_grid_points_fd, name, dtype):
    if False:
        while True:
            i = 10
    'Price Bermudan swaptions using finite difference.'
    with tf.name_scope(name):
        longest_exercise_time = unique_exercise_times[-1]
        if time_step_fd is None:
            time_step_fd = longest_exercise_time / 100.0
        short_rate_min = -0.2
        short_rate_max = 0.2
        grid = pde.grids.uniform_grid(minimums=[short_rate_min], maximums=[short_rate_max], sizes=[num_grid_points_fd], dtype=dtype)
        pde_time_grid = tf.concat([unique_exercise_times, tf.range(0.0, longest_exercise_time, time_step_fd, dtype=dtype)], axis=0)
        pde_time_grid = tf.sort(pde_time_grid, name='sort_pde_time_grid')
        pde_time_grid_dt = pde_time_grid[1:] - pde_time_grid[:-1]
        pde_time_grid_dt = tf.concat([[100.0], pde_time_grid_dt], axis=-1)
        mask = tf.math.greater(pde_time_grid_dt, _PDE_TIME_GRID_TOL)
        pde_time_grid = tf.boolean_mask(pde_time_grid, mask)
        pde_time_grid_dt = tf.boolean_mask(pde_time_grid_dt, mask)
        maturities = fixed_leg_payment_times
        maturities_shape = maturities.shape
        (unique_maturities, _) = tf.unique(tf.reshape(maturities, shape=[-1]))
        unique_maturities = tf.sort(unique_maturities, name='sort_maturities')
        num_exercise_times = tf.shape(pde_time_grid)[-1]
        num_maturities = tf.shape(unique_maturities)[-1]
        short_rates = tf.reshape(grid[0], grid[0].shape + [1, 1])
        broadcasted_exercise_times = tf.reshape(pde_time_grid, [1] + pde_time_grid.shape + [1])
        broadcasted_maturities = tf.reshape(unique_maturities, [1, 1] + unique_maturities.shape)
        short_rates = tf.broadcast_to(short_rates, grid[0].shape + [num_exercise_times, num_maturities])
        broadcasted_exercise_times = tf.broadcast_to(broadcasted_exercise_times, grid[0].shape + [num_exercise_times, num_maturities])
        broadcasted_maturities = tf.broadcast_to(broadcasted_maturities, grid[0].shape + [num_exercise_times, num_maturities])
        zcb_curve = model.discount_bond_price(tf.expand_dims(short_rates, axis=-1), broadcasted_exercise_times, broadcasted_maturities)[..., 0]
        exercise_times_index = tf.searchsorted(pde_time_grid, tf.reshape(exercise_times, [-1]))
        maturities_index = tf.searchsorted(unique_maturities, tf.reshape(maturities, [-1]))
        gather_index = _prepare_indices_ijj(tf.range(0, num_grid_points_fd), exercise_times_index, maturities_index)
        zcb_curve = tf.gather_nd(zcb_curve, gather_index)
        zcb_curve = tf.reshape(zcb_curve, [num_grid_points_fd] + maturities_shape)
        fixed_leg = tf.math.reduce_sum(fixed_leg_coupon * fixed_leg_daycount_fractions * zcb_curve, axis=-1)
        float_leg = 1.0 - zcb_curve[..., -1]
        payoff_at_exercise = float_leg - fixed_leg
        payoff_at_exercise = tf.where(is_payer_swaption, payoff_at_exercise, -payoff_at_exercise)
        unrepeated_exercise_times = exercise_times[..., -1]
        exercise_times_index = tf.searchsorted(pde_time_grid, tf.reshape(unrepeated_exercise_times, [-1]))
        (_, payoff_swap) = _map_payoff_to_sim_times(tf.reshape(exercise_times_index, unrepeated_exercise_times.shape), payoff_at_exercise, num_grid_points_fd)
        payoff_swap = tf.transpose(payoff_swap)

        def _get_index(t, tensor_to_search):
            if False:
                i = 10
                return i + 15
            t = tf.expand_dims(t, axis=-1)
            index = tf.searchsorted(tensor_to_search, t - _PDE_TIME_GRID_TOL, 'right')
            y = tf.gather(tensor_to_search, index)
            return tf.where(tf.math.abs(t - y) < _PDE_TIME_GRID_TOL, index, -1)[0]

        def _second_order_coeff_fn(t, grid):
            if False:
                for i in range(10):
                    print('nop')
            del grid
            return [[model.volatility(t) ** 2 / 2]]

        def _first_order_coeff_fn(t, grid):
            if False:
                for i in range(10):
                    print('nop')
            s = grid[0]
            return [model.drift_fn()(t, s)]

        def _zeroth_order_coeff_fn(t, grid):
            if False:
                print('Hello World!')
            del t
            return -grid[0]

        @pde.boundary_conditions.dirichlet
        def _lower_boundary_fn(t, grid):
            if False:
                return 10
            del grid
            index = _get_index(t, pde_time_grid)
            result = tf.where(index > -1, payoff_swap[index, ..., 0], 0.0)
            return tf.where(is_payer_swaption, 0.0, result)

        @pde.boundary_conditions.dirichlet
        def _upper_boundary_fn(t, grid):
            if False:
                for i in range(10):
                    print('nop')
            del grid
            index = _get_index(t, pde_time_grid)
            result = tf.where(index > -1, payoff_swap[index, ..., 0], 0.0)
            return tf.where(is_payer_swaption, result, 0.0)

        def _final_value():
            if False:
                i = 10
                return i + 15
            return tf.nn.relu(payoff_swap[-1])

        def _values_transform_fn(t, grid, value_grid):
            if False:
                for i in range(10):
                    print('nop')
            index = _get_index(t, pde_time_grid)
            v_star = tf.where(index > -1, tf.nn.relu(payoff_swap[index]), 0.0)
            return (grid, tf.maximum(value_grid, v_star))

        def _pde_time_step(t):
            if False:
                print('Hello World!')
            index = _get_index(t, pde_time_grid)
            dt = pde_time_grid_dt[index]
            return dt
        res = pde.fd_solvers.solve_backward(longest_exercise_time, 0.0, grid, values_grid=_final_value(), time_step=_pde_time_step, boundary_conditions=[(_lower_boundary_fn, _upper_boundary_fn)], values_transform_fn=_values_transform_fn, second_order_coeff_fn=_second_order_coeff_fn, first_order_coeff_fn=_first_order_coeff_fn, zeroth_order_coeff_fn=_zeroth_order_coeff_fn, dtype=dtype)
        r0 = model.instant_forward_rate(0.0)
        option_value = linear.interpolate(r0, res[1], res[0])
        return tf.reshape(notional * tf.transpose(option_value), batch_shape)