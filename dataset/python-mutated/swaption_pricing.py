"""Pricing of the Interest rate Swaption using the HJM model."""
from typing import Callable, Union
import tensorflow.compat.v2 as tf
from tf_quant_finance import types
from tf_quant_finance.math import pde
from tf_quant_finance.math import random
from tf_quant_finance.models import utils
from tf_quant_finance.models import valuation_method as vm
from tf_quant_finance.models.hjm import gaussian_hjm
from tf_quant_finance.models.hjm import quasi_gaussian_hjm
from tf_quant_finance.models.hjm import swaption_util
__all__ = ['price']
_PDE_TIME_GRID_TOL = 1e-07

def price(*, expiries: types.RealTensor, fixed_leg_payment_times: types.RealTensor, fixed_leg_daycount_fractions: types.RealTensor, fixed_leg_coupon: types.RealTensor, reference_rate_fn: Callable[..., types.RealTensor], num_hjm_factors: int, mean_reversion: types.RealTensor, volatility: Union[types.RealTensor, Callable[..., types.RealTensor]], times: types.RealTensor=None, time_step: types.RealTensor=None, num_time_steps: types.IntTensor=None, curve_times: types.RealTensor=None, corr_matrix: types.RealTensor=None, notional: types.RealTensor=None, is_payer_swaption: types.BoolTensor=None, valuation_method: vm.ValuationMethod=vm.ValuationMethod.MONTE_CARLO, num_samples: types.IntTensor=1, random_type: random.RandomType=None, seed: types.IntTensor=None, skip: types.IntTensor=0, time_step_finite_difference: types.IntTensor=None, num_time_steps_finite_difference: types.IntTensor=None, num_grid_points_finite_difference: types.IntTensor=101, dtype: tf.DType=None, name: str=None) -> types.RealTensor:
    if False:
        i = 10
        return i + 15
    "Calculates the price of European swaptions using the HJM model.\n\n  A European Swaption is a contract that gives the holder an option to enter a\n  swap contract at a future date at a prespecified fixed rate. A swaption that\n  grants the holder the right to pay fixed rate and receive floating rate is\n  called a payer swaption while the swaption that grants the holder the right to\n  receive fixed and pay floating payments is called the receiver swaption.\n  Typically the start date (or the inception date) of the swap coincides with\n  the expiry of the swaption. Mid-curve swaptions are currently not supported\n  (b/160061740).\n\n  This implementation uses the HJM model to numerically value European\n  swaptions. For more information on the formulation of the HJM model, see\n  quasi_gaussian_hjm.py.\n\n  #### Example\n\n  ````python\n  import numpy as np\n  import tensorflow.compat.v2 as tf\n  import tf_quant_finance as tff\n\n  dtype = tf.float64\n\n  # Price 1y x 1y swaption with quarterly payments using Monte Carlo\n  # simulations.\n  expiries = np.array([1.0])\n  fixed_leg_payment_times = np.array([1.25, 1.5, 1.75, 2.0])\n  fixed_leg_daycount_fractions = 0.25 * np.ones_like(fixed_leg_payment_times)\n  fixed_leg_coupon = 0.011 * np.ones_like(fixed_leg_payment_times)\n  zero_rate_fn = lambda x: 0.01 * tf.ones_like(x, dtype=dtype)\n  mean_reversion = [0.03]\n  volatility = [0.02]\n\n  price = tff.models.hjm.swaption_price(\n      expiries=expiries,\n      fixed_leg_payment_times=fixed_leg_payment_times,\n      fixed_leg_daycount_fractions=fixed_leg_daycount_fractions,\n      fixed_leg_coupon=fixed_leg_coupon,\n      reference_rate_fn=zero_rate_fn,\n      notional=100.,\n      num_hjm_factors=1,\n      mean_reversion=mean_reversion,\n      volatility=volatility,\n      valuation_method=tff.model.ValuationMethod.MONTE_CARLO,\n      num_samples=500000,\n      time_step=0.1,\n      random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,\n      seed=[1, 2])\n  # Expected value: [[0.716]]\n  ````\n\n\n  #### References:\n    [1]: D. Brigo, F. Mercurio. Interest Rate Models-Theory and Practice.\n    Second Edition. 2007. Section 6.7, page 237.\n\n  Args:\n    expiries: A real `Tensor` of any shape and dtype. The time to expiration of\n      the swaptions. The shape of this input along with the batch shape of the\n      HJM model determines the number (and shape) of swaptions to be priced and\n      the shape of the output. If the batch shape of HJM models is\n      `model_batch_shape`, then the leading dimensions of `expiries` must be\n      broadcastable to `model_batch_shape`. For example, if the rank of\n      `model_batch_shape` is `n` and the rank of `expiries.shape` is `m`, then\n      `m>=n` and the leading `n` dimensions of `expiries.shape` must be\n      broadcastable to `model_batch_shape`.\n    fixed_leg_payment_times: A real `Tensor` of the same dtype as `expiries`.\n      The payment times for each payment in the fixed leg. The shape of this\n      input should be `expiries.shape + [n]` where `n` denotes the number of\n      fixed payments in each leg. The `fixed_leg_payment_times` should be\n      greater-than or equal-to the corresponding expiries.\n    fixed_leg_daycount_fractions: A real `Tensor` of the same dtype and\n      compatible shape as `fixed_leg_payment_times`. The daycount fractions for\n      each payment in the fixed leg.\n    fixed_leg_coupon: A real `Tensor` of the same dtype and compatible shape as\n      `fixed_leg_payment_times`. The fixed rate for each payment in the fixed\n      leg.\n    reference_rate_fn: A Python callable that accepts expiry time as a real\n      `Tensor` and returns a `Tensor` of shape\n      `model_batch_shape + input_shape`. Returns the continuously compounded\n      zero rate at the present time for the input expiry time.\n    num_hjm_factors: A Python scalar which corresponds to the number of factors\n      in the batch of HJM models to be used for pricing.\n    mean_reversion: A real positive `Tensor` of shape\n      `model_batch_shape + [num_hjm_factors]`.\n      Corresponds to the mean reversion rate of each factor in the batch.\n    volatility: A real positive `Tensor` of the same `dtype` and shape as\n        `mean_reversion` or a callable with the following properties:\n        (a)  The callable should accept a scalar `Tensor` `t` and a `Tensor`\n        `r(t)` of shape `batch_shape + [num_samples]` and returns a `Tensor` of\n        shape compatible with `batch_shape + [num_samples, dim]`. The variable\n        `t`  stands for time and `r(t)` is the short rate at time `t`. The\n        function returns instantaneous volatility `sigma(t) = sigma(t, r(t))`.\n        When `volatility` is specified as a real `Tensor`, each factor is\n        assumed to have a constant instantaneous volatility  and the  model is\n        effectively a Gaussian HJM model.\n        Corresponds to the instantaneous volatility of each factor.\n    times: An optional rank 1 `Tensor` of increasing positive real values. The\n      times at which Monte Carlo simulations are performed. Relevant when\n      swaption valuation is done using Monte Calro simulations.\n      Default value: `None` in which case simulation times are computed based\n      on either `time_step` or `num_time_steps` inputs.\n    time_step: Optional scalar real `Tensor`. Maximal distance between time\n      grid points in Euler scheme. Relevant when Euler scheme is used for\n      simulation. This input or `num_time_steps` are required when valuation\n      method is Monte Carlo.\n      Default Value: `None`.\n    num_time_steps: An optional scalar integer `Tensor` - a total number of\n      time steps during Monte Carlo simulations. The maximal distance betwen\n      points in grid is bounded by\n      `times[-1] / (num_time_steps - times.shape[0])`.\n      Either this or `time_step` should be supplied when the valuation method\n      is Monte Carlo.\n      Default value: `None`.\n    curve_times: An optional rank 1 `Tensor` of positive and increasing real\n      values. The maturities at which spot discount curve is computed during\n      simulations.\n      Default value: `None` in which case `curve_times` is computed based on\n      swaption expiries and `fixed_leg_payments_times` inputs.\n    corr_matrix: A `Tensor` of shape `[num_hjm_factors, num_hjm_factors]` and\n      the same `dtype` as `mean_reversion`. Specifies the correlation between\n      HJM factors.\n      Default value: `None` in which case the factors are assumed to be\n        uncorrelated.\n    notional: An optional `Tensor` of same dtype and compatible shape as\n      `strikes`specifying the notional amount for the underlying swaps.\n       Default value: None in which case the notional is set to 1.\n    is_payer_swaption: A boolean `Tensor` of a shape compatible with `expiries`.\n      Indicates whether the swaption is a payer (if True) or a receiver (if\n      False) swaption. If not supplied, payer swaptions are assumed.\n    valuation_method: An enum of type `ValuationMethod` specifying\n      the method to be used for swaption valuation. Currently the valuation is\n      supported using `MONTE_CARLO` and `FINITE_DIFFERENCE` methods. Valuation\n      using finite difference is only supported for Gaussian HJM models, i.e.\n      for models with constant mean-reversion rate and time-dependent\n      volatility.\n      Default value: `ValuationMethod.MONTE_CARLO`, in which case\n      swaption valuation is done using Monte Carlo simulations.\n    num_samples: Positive scalar `int32` `Tensor`. The number of simulation\n      paths during Monte-Carlo valuation. This input is ignored during analytic\n      valuation.\n      Default value: The default value is 1.\n    random_type: Enum value of `RandomType`. The type of (quasi)-random number\n      generator to use to generate the simulation paths. This input is relevant\n      only for Monte-Carlo valuation and ignored during analytic valuation.\n      Default value: `None` which maps to the standard pseudo-random numbers.\n    seed: Seed for the random number generator. The seed is only relevant if\n      `random_type` is one of `[STATELESS, PSEUDO, HALTON_RANDOMIZED,\n      PSEUDO_ANTITHETIC, STATELESS_ANTITHETIC]`. For `PSEUDO`,\n      `PSEUDO_ANTITHETIC` and `HALTON_RANDOMIZED` the seed should be an Python\n      integer. For `STATELESS` and  `STATELESS_ANTITHETIC` must be supplied as\n      an integer `Tensor` of shape `[2]`. This input is relevant only for\n      Monte-Carlo valuation and ignored during analytic valuation.\n      Default value: `None` which means no seed is set.\n    skip: `int32` 0-d `Tensor`. The number of initial points of the Sobol or\n      Halton sequence to skip. Used only when `random_type` is 'SOBOL',\n      'HALTON', or 'HALTON_RANDOMIZED', otherwise ignored.\n      Default value: `0`.\n    time_step_finite_difference: Optional scalar real `Tensor`. Spacing between\n      time grid points in finite difference discretization. This input is only\n      relevant for valuation using finite difference.\n      Default value: `None`. If `num_time_steps_finite_difference` is also\n      unspecified then a `time_step` corresponding to 100 discretization steps\n      is used.\n    num_time_steps_finite_difference: Optional scalar real `Tensor`. Number of\n      time grid points in finite difference discretization. This input is only\n      relevant for valuation using finite difference.\n      Default value: `None`. If `time_step_finite_difference` is also\n      unspecified, then 100 time steps are used.\n    num_grid_points_finite_difference: Optional scalar real `Tensor`. Number of\n      spatial grid points per dimension. Currently, we construct an uniform grid\n      for spatial discretization. This input is only relevant for valuation\n      using finite difference.\n      Default value: 101.\n    dtype: The default dtype to use when converting values to `Tensor`s.\n      Default value: `None` which means that default dtypes inferred by\n        TensorFlow are used.\n    name: Python string. The name to give to the ops created by this function.\n      Default value: `None` which maps to the default name `hjm_swaption_price`.\n\n  Returns:\n    A `Tensor` of real dtype and shape derived from `model_batch_shape` and\n    expiries.shape containing the computed swaption prices. The shape of the\n    output is as follows:\n      * If the `model_batch_shape` is [], then the shape of the output is\n        expiries.shape\n      * Otherwise, the shape of the output is\n        `model_batch_shape + expiries.shape[model_batch_shape.rank:]`\n    For swaptions that have reset in the past (expiries<0), the function sets\n    the corresponding option prices to 0.0.\n  "
    name = name or 'hjm_swaption_price'
    with tf.name_scope(name):
        expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
        dtype = dtype or expiries.dtype
        fixed_leg_payment_times = tf.convert_to_tensor(fixed_leg_payment_times, dtype=dtype, name='fixed_leg_payment_times')
        fixed_leg_daycount_fractions = tf.convert_to_tensor(fixed_leg_daycount_fractions, dtype=dtype, name='fixed_leg_daycount_fractions')
        fixed_leg_coupon = tf.convert_to_tensor(fixed_leg_coupon, dtype=dtype, name='fixed_leg_coupon')
        notional = tf.convert_to_tensor(notional, dtype=dtype, name='notional')
        if is_payer_swaption is None:
            is_payer_swaption = True
        is_payer_swaption = tf.convert_to_tensor(is_payer_swaption, dtype=tf.bool, name='is_payer_swaption')
        if expiries.shape.rank < fixed_leg_payment_times.shape.rank - 1:
            raise ValueError('Swaption expiries not specified for all swaptions in the batch. Expected rank {} but received {}.'.format(fixed_leg_payment_times.shape.rank - 1, expiries.shape.rank))
        expiries = tf.expand_dims(expiries, axis=-1)
        expiries = tf.repeat(expiries, tf.shape(fixed_leg_payment_times)[-1], axis=-1)
        if valuation_method == vm.ValuationMethod.FINITE_DIFFERENCE:
            model = gaussian_hjm.GaussianHJM(num_hjm_factors, mean_reversion=mean_reversion, volatility=volatility, initial_discount_rate_fn=reference_rate_fn, corr_matrix=corr_matrix, dtype=dtype)
            if reference_rate_fn(tf.constant([0.0], dtype=dtype)).shape.rank > 1:
                raise ValueError('Pricing swaptions using a batch of HJM models with finite differences is not currently supported.')
            instrument_batch_shape = expiries.shape.as_list()[:-1] or [1]
            return _european_swaption_fd(instrument_batch_shape, model, tf.expand_dims(expiries, axis=-2), fixed_leg_payment_times, fixed_leg_daycount_fractions, fixed_leg_coupon, notional, is_payer_swaption, time_step_finite_difference, num_time_steps_finite_difference, num_grid_points_finite_difference, name + '_fd', dtype)
        elif valuation_method == vm.ValuationMethod.MONTE_CARLO:
            model = quasi_gaussian_hjm.QuasiGaussianHJM(num_hjm_factors, mean_reversion=mean_reversion, volatility=volatility, initial_discount_rate_fn=reference_rate_fn, corr_matrix=corr_matrix, dtype=dtype)
            return _european_swaption_mc(model, expiries, fixed_leg_payment_times, fixed_leg_daycount_fractions, fixed_leg_coupon, notional, is_payer_swaption, times, time_step, num_time_steps, curve_times, num_samples, random_type, skip, seed, dtype, name + '_mc')
        else:
            raise ValueError('Swaption Valuation using {} is not supported'.format(str(valuation_method)))

def _european_swaption_mc(model, expiries, fixed_leg_payment_times, fixed_leg_daycount_fractions, fixed_leg_coupon, notional, is_payer_swaption, times, time_step, num_time_steps, curve_times, num_samples, random_type, skip, seed, dtype, name):
    if False:
        for i in range(10):
            print('nop')
    'Price European swaptions using Monte-Carlo.'
    with tf.name_scope(name):
        if times is None and time_step is None and (num_time_steps is None):
            raise ValueError('One of `times`, `time_step` or `num_time_steps` must be provided for simulation based swaption valuation.')

        def _sample_discount_curve_path_fn(times, curve_times, num_samples):
            if False:
                i = 10
                return i + 15
            (p_t_tau, r_t, df) = model.sample_discount_curve_paths(times=times, curve_times=curve_times, num_samples=num_samples, random_type=random_type, time_step=time_step, num_time_steps=num_time_steps, seed=seed, skip=skip)
            p_t_tau = tf.expand_dims(p_t_tau, axis=-1)
            r_t = tf.expand_dims(r_t, axis=-1)
            df = tf.expand_dims(df, axis=-1)
            return (p_t_tau, r_t, df)
        (payoff_discount_factors, payoff_bond_price) = swaption_util.discount_factors_and_bond_prices_from_samples(expiries=expiries, payment_times=fixed_leg_payment_times, sample_discount_curve_paths_fn=_sample_discount_curve_path_fn, num_samples=num_samples, times=times, curve_times=curve_times, dtype=dtype)
        fixed_leg_pv = tf.expand_dims(fixed_leg_coupon * fixed_leg_daycount_fractions, axis=-1) * payoff_bond_price
        fixed_leg_pv = tf.math.reduce_sum(fixed_leg_pv, axis=-2)
        float_leg_pv = 1.0 - payoff_bond_price[..., -1, :]
        payoff_swap = payoff_discount_factors[..., -1, :] * (float_leg_pv - fixed_leg_pv)
        payoff_swap = tf.where(is_payer_swaption, payoff_swap, -1.0 * payoff_swap)
        payoff_swaption = tf.math.maximum(payoff_swap, 0.0)
        option_value = tf.math.reduce_mean(payoff_swaption, axis=0)
        return notional * tf.squeeze(option_value, axis=-1)

def _european_swaption_fd(batch_shape, model, exercise_times, fixed_leg_payment_times, fixed_leg_daycount_fractions, fixed_leg_coupon, notional, is_payer_swaption, time_step_fd, num_time_steps_fd, num_grid_points_fd, name, dtype):
    if False:
        for i in range(10):
            print('nop')
    'Price European swaptions using finite difference.'
    with tf.name_scope(name):
        dim = model.dim()
        x_min = -0.5
        x_max = 0.5
        grid = pde.grids.uniform_grid(minimums=[x_min] * dim, maximums=[x_max] * dim, sizes=[num_grid_points_fd] * dim, dtype=dtype)
        (pde_time_grid, pde_time_grid_dt) = _create_pde_time_grid(exercise_times, time_step_fd, num_time_steps_fd, dtype)
        (maturities, unique_maturities, maturities_shape) = _create_term_structure_maturities(fixed_leg_payment_times)
        num_maturities = tf.shape(unique_maturities)[-1]
        x_meshgrid = _coord_grid_to_mesh_grid(grid)
        meshgrid_shape = tf.shape(x_meshgrid)
        broadcasted_maturities = tf.expand_dims(unique_maturities, axis=0)
        num_grid_points = tf.math.reduce_prod(meshgrid_shape[1:])
        shape_to_broadcast = tf.concat([meshgrid_shape, [num_maturities]], axis=0)
        state_x = tf.expand_dims(x_meshgrid, axis=-1)
        state_x = tf.broadcast_to(state_x, shape_to_broadcast)
        broadcasted_maturities = tf.broadcast_to(broadcasted_maturities, shape_to_broadcast[1:])

        def _get_swap_payoff(payoff_time):
            if False:
                return 10
            broadcasted_exercise_times = tf.broadcast_to(payoff_time, shape_to_broadcast[1:])
            zcb_curve = model.discount_bond_price(tf.transpose(tf.reshape(state_x, [dim, num_grid_points * num_maturities])), tf.reshape(broadcasted_exercise_times, [-1]), tf.reshape(broadcasted_maturities, [-1]))
            zcb_curve = tf.reshape(zcb_curve, [num_grid_points, num_maturities])
            maturities_index = tf.searchsorted(unique_maturities, tf.reshape(maturities, [-1]))
            zcb_curve = tf.gather(zcb_curve, maturities_index, axis=-1)
            zcb_curve = tf.reshape(zcb_curve, tf.concat([[num_grid_points], maturities_shape], axis=0))
            fixed_leg = tf.math.reduce_sum(fixed_leg_coupon * fixed_leg_daycount_fractions * zcb_curve, axis=-1)
            float_leg = 1.0 - zcb_curve[..., -1]
            payoff_swap = float_leg - fixed_leg
            payoff_swap = tf.where(is_payer_swaption, payoff_swap, -payoff_swap)
            return tf.reshape(tf.transpose(payoff_swap), tf.concat([batch_shape, meshgrid_shape[1:]], axis=0))

        def _get_index(t, tensor_to_search):
            if False:
                while True:
                    i = 10
            t = tf.expand_dims(t, axis=-1)
            index = tf.searchsorted(tensor_to_search, t - _PDE_TIME_GRID_TOL, 'right')
            y = tf.gather(tensor_to_search, index)
            return tf.where(tf.math.abs(t - y) < _PDE_TIME_GRID_TOL, index, -1)[0]
        sum_x_meshgrid = tf.math.reduce_sum(x_meshgrid, axis=0)

        def _is_exercise_time(t):
            if False:
                for i in range(10):
                    print('nop')
            return tf.reduce_any(tf.where(tf.math.abs(exercise_times[..., -1] - t) < _PDE_TIME_GRID_TOL, True, False), axis=-1)

        def _discounting_fn(t, grid):
            if False:
                for i in range(10):
                    print('nop')
            del grid
            f_0_t = model._instant_forward_rate_fn(t)
            return sum_x_meshgrid + f_0_t

        def _final_value():
            if False:
                while True:
                    i = 10
            t = pde_time_grid[-1]
            payoff_swap = tf.nn.relu(_get_swap_payoff(t))
            is_ex_time = _is_exercise_time(t)
            return tf.where(tf.reshape(is_ex_time, tf.concat([batch_shape, [1] * dim], axis=0)), payoff_swap, 0.0)

        def _values_transform_fn(t, grid, value_grid):
            if False:
                i = 10
                return i + 15
            zero = tf.zeros_like(value_grid)
            is_ex_time = _is_exercise_time(t)

            def _at_least_one_swaption_pays():
                if False:
                    for i in range(10):
                        print('nop')
                payoff_swap = tf.nn.relu(_get_swap_payoff(t))
                return tf.where(tf.reshape(is_ex_time, tf.concat([batch_shape, [1] * dim], axis=0)), payoff_swap, zero)
            v_star = tf.cond(tf.math.reduce_any(is_ex_time), _at_least_one_swaption_pays, lambda : zero)
            return (grid, tf.maximum(value_grid, v_star))

        def _pde_time_step(t):
            if False:
                while True:
                    i = 10
            index = _get_index(t, pde_time_grid)
            return pde_time_grid_dt[index]
        boundary_conditions = [(None, None) for i in range(dim)]
        res = model.fd_solver_backward(pde_time_grid[-1], 0.0, grid, values_grid=_final_value(), time_step=_pde_time_step, boundary_conditions=boundary_conditions, values_transform_fn=_values_transform_fn, discounting=_discounting_fn, dtype=dtype)
        idx = tf.searchsorted(tf.convert_to_tensor(grid), tf.expand_dims(tf.convert_to_tensor([0.0] * dim, dtype=dtype), axis=-1))
        idx = tf.squeeze(idx) if dim > 1 else tf.reshape(idx, shape=[1])
        slices = [slice(None)] + [slice(i, i + 1) for i in tf.unstack(idx)]
        option_value = res[0][slices]
        option_value = tf.squeeze(option_value, axis=list(range(-dim, 0)))
        return notional * tf.reshape(option_value, batch_shape)

def _coord_grid_to_mesh_grid(coord_grid):
    if False:
        while True:
            i = 10
    if len(coord_grid) == 1:
        return tf.expand_dims(coord_grid[0], 0)
    x_meshgrid = tf.stack(values=tf.meshgrid(*coord_grid, indexing='ij'), axis=-1)
    perm = [len(coord_grid)] + list(range(len(coord_grid)))
    return tf.transpose(x_meshgrid, perm=perm)

def _create_pde_time_grid(exercise_times, time_step_fd, num_time_steps_fd, dtype):
    if False:
        for i in range(10):
            print('nop')
    'Create PDE time grid.'
    with tf.name_scope('create_pde_time_grid'):
        (exercise_times, _) = tf.unique(tf.reshape(exercise_times, shape=[-1]))
        if num_time_steps_fd is not None:
            num_time_steps_fd = tf.convert_to_tensor(num_time_steps_fd, dtype=tf.int32, name='num_time_steps_fd')
            time_step_fd = tf.math.reduce_max(exercise_times) / tf.cast(num_time_steps_fd, dtype=dtype)
        if time_step_fd is None and num_time_steps_fd is None:
            num_time_steps_fd = 100
        (pde_time_grid, _, _) = utils.prepare_grid(times=exercise_times, time_step=time_step_fd, dtype=dtype, num_time_steps=num_time_steps_fd)
        pde_time_grid_dt = pde_time_grid[1:] - pde_time_grid[:-1]
        pde_time_grid_dt = tf.concat([[100.0], pde_time_grid_dt], axis=-1)
        return (pde_time_grid, pde_time_grid_dt)

def _create_term_structure_maturities(fixed_leg_payment_times):
    if False:
        i = 10
        return i + 15
    'Create maturities needed for termstructure simulations.'
    with tf.name_scope('create_termstructure_maturities'):
        maturities = fixed_leg_payment_times
        maturities_shape = tf.shape(maturities)
        (unique_maturities, _) = tf.unique(tf.reshape(maturities, shape=[-1]))
        unique_maturities = tf.sort(unique_maturities, name='sort_maturities')
        return (maturities, unique_maturities, maturities_shape)