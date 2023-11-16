"""Calculating American option prices with Andersen-Lake approximation."""
import tensorflow.compat.v2 as tf
from tf_quant_finance import types
from tf_quant_finance import utils
from tf_quant_finance.black_scholes import vanilla_prices
from tf_quant_finance.experimental.american_option_pricing import common
from tf_quant_finance.experimental.american_option_pricing import exercise_boundary
from tf_quant_finance.math.integration import gauss_kronrod
calculate_exercise_boundary = exercise_boundary.exercise_boundary
standard_normal_cdf = common.standard_normal_cdf
d_plus = common.d_plus
d_minus = common.d_minus
machine_eps = common.machine_eps
divide_with_positive_denominator = common.divide_with_positive_denominator

def andersen_lake(*, volatilities: types.RealTensor, strikes: types.RealTensor, expiries: types.RealTensor, spots: types.RealTensor=None, forwards: types.RealTensor=None, discount_rates: types.RealTensor=None, discount_factors: types.RealTensor=None, dividend_rates: types.RealTensor=None, is_call_options: types.BoolTensor=None, grid_num_points: int=10, integration_num_points_kronrod: int=31, integration_num_points_legendre: int=32, max_iterations_exercise_boundary: int=30, max_depth_kronrod: int=30, tolerance_exercise_boundary: types.RealTensor=1e-08, tolerance_kronrod: types.RealTensor=1e-08, dtype: tf.DType=None, name: str=None) -> types.RealTensor:
    if False:
        return 10
    "Computes American option prices using the Andersen-Lake approximation.\n\n  #### Example\n\n  ```python\n  volatilities = [0.1, 0.15]\n  strikes = [3, 2]\n  expiries = [1, 2]\n  spots = [8.0, 9.0]\n  discount_rates = [0.01, 0.02]\n  dividend_rates = [0.01, 0.02]\n  is_call_options = [True, False]\n  grid_num_points = 40\n  integration_num_points_kronrod = 31\n  integration_num_points_legendre = 32\n  max_iterations_exercise_boundary = 500\n  max_depth_kronrod = 50\n  tolerance_exercise_boundary = 1e-11\n  tolerance_kronrod = 1e-11\n  computed_prices = andersen_lake(\n      volatilities=volatilities,\n      strikes=strikes,\n      expiries=expiries,\n      spots=spots,\n      discount_rates=discount_rates,\n      dividend_rates=dividend_rates,\n      is_call_options=is_call_options,\n      grid_num_points=grid_num_points,\n      integration_num_points_kronrod=integration_num_points_kronrod,\n      integration_num_points_legendre=integration_num_points_legendre,\n      max_iterations_exercise_boundary=max_iterations_exercise_boundary,\n      max_depth_kronrod=max_depth_kronrod,\n      tolerance_exercise_boundary=tolerance_exercise_boundary,\n      tolerance_kronrod=tolerance_kronrod\n      dtype=tf.float64)\n  # Expected print output of computed prices:\n  # [4.950249e+00, 7.768513e-14]\n  ```\n\n  #### References:\n  [1] Leif Andersen, Mark Lake and Dimitri Offengenden. High-performance\n  American option pricing. 2015\n  https://engineering.nyu.edu/sites/default/files/2019-03/Carr-adjusting-exponential-levy-models.pdf#page=46\n\n  Args:\n    volatilities: Real `Tensor` of any real dtype and shape `[num_options]`.\n      The volatilities to expiry of the options to price.\n    strikes: A real `Tensor` of the same dtype and same shape as `volatilities`.\n      The strikes of the options to be priced.\n    expiries: A real `Tensor` of same dtype and same shape as `volatilities`.\n      The expiry of each option. The units should be such that\n      `expiry * volatility**2` is dimensionless.\n    spots: A real `Tensor` of same shape as `volatilities`. The current spot\n      price of the underlying. Either this argument or the `forwards` (but not\n      both) must be supplied.\n    forwards: A real `Tensor` of same shape as `volatilities`. The forwards to\n      maturity. Either this argument or the `spots` must be supplied but both\n      must not be supplied.\n    discount_rates: An optional real `Tensor` of same shape and dtype as the\n      `volatilities`. If not `None`, discount factors are calculated as e^(-rT),\n      where r are the discount rates, or risk free rates.\n      Default value: `None`, which maps to `-log(discount_factors) / expiries`\n        if `discount_factors` is not `None`, or maps to `0` when\n        `discount_factors` is also `None`.\n    discount_factors: An optional real `Tensor` of same shape and dtype as the\n      `volatilities`. If not `None`, these are the discount factors to expiry\n      (i.e. e^(-rT)). Mutually exclusive with `discount_rate`. If neither is\n      given, no discounting is applied (i.e. the undiscounted option price is\n      returned). If `spots` is supplied and `discount_factors` is not `None`\n      then this is also used to compute the forwards to expiry.\n      Default value: `None`.\n    dividend_rates: An optional real `Tensor` of same shape and dtype as the\n      `volatilities`. The continuous dividend rate on the underliers. May be\n      negative (to indicate costs of holding the underlier).\n      Default value: `None`, equivalent to zero dividends.\n    is_call_options: A boolean `Tensor` of a shape compatible with\n      `volatilities`. Indicates whether the option is a call (if True) or a put\n      (if False). If not supplied, call options are assumed.\n    grid_num_points: positive `int`. The number of equidistant points to divide\n      the values given in `expiries` into in the grid of `tau_grid`.\n      Default value: 10.\n    integration_num_points_kronrod: positive `int`. The number of points used in\n      the Gauss-Kronrod integration approximation method used for\n      calculating the option prices.\n      Default value: 31.\n    integration_num_points_legendre: positive `int`. The number of points used\n      in the Gauss-Legendre integration approximation method used for\n      calculating the exercise boundary function used for pricing the options.\n      Default value: 32.\n    max_iterations_exercise_boundary: positive `int`. Maximum number of\n      iterations for calculating the exercise boundary if it doesn't converge\n      earlier.\n      Default value: 30.\n    max_depth_kronrod: positive `int`. Maximum number of iterations for\n      calculating the Gauss-Kronrod integration approximation.\n      Default value: 30.\n    tolerance_exercise_boundary: Positive scalar `Tensor`. The tolerance for the\n      convergence of calculating the exercise boundary function.\n      Default value: 1e-8.\n    tolerance_kronrod: Positive scalar `Tensor`. The tolerance for the\n      convergence of calculating the Gauss-Kronrod integration approximation.\n      Default value: 1e-8.\n    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion\n      of any supplied non-`Tensor` arguments to `Tensor`.\n      Default value: None which maps to the default dtype inferred by\n        TensorFlow.\n    name: str. The name for the ops created by this function.\n      Default value: None which is mapped to the default name `andersen_lake`.\n\n  Returns:\n    `Tensor` of shape `[num_options]`, containing the calculated American option\n    prices.\n\n  Raises:\n    ValueError:\n      (a) If both `forwards` and `spots` are supplied or if neither is supplied.\n  "
    if (spots is None) == (forwards is None):
        raise ValueError('Either spots or forwards must be supplied but not both.')
    if discount_rates is not None and discount_factors is not None:
        raise ValueError('At most one of discount_rates and discount_factors may be supplied')
    with tf.name_scope(name or 'andersen_lake'):
        volatilities = tf.convert_to_tensor(volatilities, dtype=dtype, name='volatilities')
        dtype = volatilities.dtype
        strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
        expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
        if discount_rates is not None:
            discount_rates = tf.convert_to_tensor(discount_rates, dtype=dtype, name='discount_rates')
        elif discount_factors is not None:
            discount_factors = tf.convert_to_tensor(discount_factors, dtype=dtype, name='discount_factors')
            discount_rates = tf.math.divide_no_nan(-tf.math.log(discount_factors), expiries)
        else:
            discount_rates = tf.constant([0.0], dtype=dtype, name='discount_rates')
        if dividend_rates is not None:
            dividend_rates = tf.convert_to_tensor(dividend_rates, dtype=dtype, name='dividend_rates')
        else:
            dividend_rates = tf.constant([0.0], dtype=dtype, name='dividend_rates')
        if forwards is not None:
            spots = tf.convert_to_tensor(forwards * tf.exp(-(discount_rates - dividend_rates) * expiries), dtype=dtype, name='spots')
        else:
            spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
        if is_call_options is not None:
            is_call_options = tf.convert_to_tensor(is_call_options, dtype=tf.bool, name='is_call_options')
        else:
            is_call_options = tf.constant(True, name='is_call_options')
        am_prices = _andersen_lake(sigma=volatilities, k_temp=strikes, tau=expiries, s_temp=spots, r_temp=discount_rates, q_temp=dividend_rates, is_call_options=is_call_options, grid_num_points=grid_num_points, integration_num_points_kronrod=integration_num_points_kronrod, integration_num_points_legendre=integration_num_points_legendre, max_iterations_exercise_boundary=max_iterations_exercise_boundary, max_depth_kronrod=max_depth_kronrod, tolerance_exercise_boundary=tolerance_exercise_boundary, tolerance_kronrod=tolerance_kronrod, dtype=dtype)
        return am_prices

def _andersen_lake(*, sigma, k_temp, tau, s_temp, r_temp, q_temp, is_call_options, grid_num_points, integration_num_points_kronrod, integration_num_points_legendre, max_iterations_exercise_boundary, max_depth_kronrod, tolerance_exercise_boundary, tolerance_kronrod, dtype):
    if False:
        for i in range(10):
            print('nop')
    'Computes American option prices using the Andersen-Lake formula.'
    eu_prices = vanilla_prices.option_price(volatilities=sigma, strikes=k_temp, expiries=tau, spots=s_temp, discount_rates=r_temp, dividend_rates=q_temp, is_call_options=is_call_options, dtype=dtype)
    k = tf.where(is_call_options, s_temp, k_temp)
    s = tf.where(is_call_options, k_temp, s_temp)
    r = tf.where(is_call_options, q_temp, r_temp)
    q = tf.where(is_call_options, r_temp, q_temp)
    tau_grid = tf.linspace(tau / grid_num_points, tau, grid_num_points, axis=-1)
    epsilon = machine_eps(dtype)
    r_e_b = tf.where(tf.math.abs(r) < epsilon, tf.where(tf.math.abs(q) < epsilon, tf.constant(0.1, dtype=dtype), r), r)
    q_e_b = tf.where(tf.math.abs(q) < epsilon, tf.where(tf.math.abs(r) < epsilon, tf.constant(0.1, dtype=dtype), q), q)
    exercise_boundary_fn = calculate_exercise_boundary(tau_grid, k, r_e_b, q_e_b, sigma, max_iterations_exercise_boundary, tolerance_exercise_boundary, integration_num_points_legendre, dtype)
    k_exp = k[:, tf.newaxis, tf.newaxis]
    r_exp = r[:, tf.newaxis, tf.newaxis]
    q_exp = q[:, tf.newaxis, tf.newaxis]
    sigma_exp = sigma[:, tf.newaxis, tf.newaxis]
    s_exp = s[:, tf.newaxis, tf.newaxis]
    tau_exp = tau[:, tf.newaxis, tf.newaxis]

    def get_ratio(u):
        if False:
            for i in range(10):
                print('nop')
        u_shape = utils.get_shape(u)
        u_reshaped = tf.reshape(u, [u_shape[0], u_shape[1] * u_shape[2]])
        return divide_with_positive_denominator(s_exp, tf.reshape(exercise_boundary_fn(u_reshaped), [u_shape[0], u_shape[1], u_shape[2]]))

    def func_1(u):
        if False:
            while True:
                i = 10
        ratio = get_ratio(u)
        norm = standard_normal_cdf(-d_minus(tau_exp - u, ratio, r_exp, q_exp, sigma_exp))
        return r_exp * k_exp * tf.math.exp(-r_exp * (tau_exp - u)) * norm
    term1 = gauss_kronrod(func=func_1, lower=tf.zeros_like(tau), upper=tau, tolerance=tolerance_kronrod, num_points=integration_num_points_kronrod, max_depth=max_depth_kronrod, dtype=dtype)

    def func_2(u):
        if False:
            i = 10
            return i + 15
        ratio = get_ratio(u)
        norm = standard_normal_cdf(-d_plus(tau_exp - u, ratio, r_exp, q_exp, sigma_exp))
        return q_exp * s_exp * tf.math.exp(-q_exp * (tau_exp - u)) * norm
    term2 = gauss_kronrod(func=func_2, lower=tf.zeros_like(tau), upper=tau, tolerance=tolerance_kronrod, num_points=integration_num_points_kronrod, max_depth=max_depth_kronrod, dtype=dtype)
    return eu_prices + term1 - term2