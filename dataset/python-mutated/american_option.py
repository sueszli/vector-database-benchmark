"""Approximate formulas for American option pricing."""
from typing import Tuple
import numpy as np
import tensorflow.compat.v2 as tf
from tf_quant_finance import types
from tf_quant_finance.black_scholes import vanilla_prices
from tf_quant_finance.math import gradient
from tf_quant_finance.math import integration
from tf_quant_finance.math.root_search import newton as root_finder_newton
__all__ = ['adesi_whaley', 'bjerksund_stensland']
_SQRT_2_PI = np.sqrt(2 * np.pi, dtype=np.float64)
_SQRT_2 = np.sqrt(2.0, dtype=np.float64)

def adesi_whaley(*, volatilities: types.RealTensor, strikes: types.RealTensor, expiries: types.RealTensor, spots: types.RealTensor=None, forwards: types.RealTensor=None, discount_rates: types.RealTensor=None, dividend_rates: types.RealTensor=None, discount_factors: types.RealTensor=None, is_call_options: types.BoolTensor=None, max_iterations: int=100, tolerance: types.RealTensor=1e-08, dtype: tf.DType=None, name: str=None) -> Tuple[types.RealTensor, types.BoolTensor, types.BoolTensor]:
    if False:
        while True:
            i = 10
    "Computes American option prices using the Baron-Adesi Whaley approximation.\n\n  #### Example\n\n  ```python\n  spots = [80.0, 90.0, 100.0, 110.0, 120.0]\n  strikes = [100.0, 100.0, 100.0, 100.0, 100.0]\n  volatilities = [0.2, 0.2, 0.2, 0.2, 0.2]\n  expiries = 0.25\n  dividends = 0.12\n  discount_rates = 0.08\n  computed_prices = adesi_whaley(\n      volatilities=volatilities,\n      strikes=strikes,\n      expiries=expiries,\n      discount_rates=discount_rates,\n      dividend_rates=dividends,\n      spots=spots,\n      dtype=tf.float64)\n  # Expected print output of computed prices:\n  # [0.03, 0.59, 3.52, 10.31, 20.0]\n  ```\n\n  #### References:\n  [1] Baron-Adesi, Whaley, Efficient Analytic Approximation of American Option\n    Values, The Journal of Finance, Vol XLII, No. 2, June 1987\n    https://deriscope.com/docs/Barone_Adesi_Whaley_1987.pdf\n\n  Args:\n    volatilities: Real `Tensor` of any shape and real dtype. The volatilities to\n      expiry of the options to price.\n    strikes: A real `Tensor` of the same dtype and compatible shape as\n      `volatilities`. The strikes of the options to be priced.\n    expiries: A real `Tensor` of same dtype and compatible shape as\n      `volatilities`. The expiry of each option. The units should be such that\n      `expiry * volatility**2` is dimensionless.\n    spots: A real `Tensor` of any shape that broadcasts to the shape of the\n      `volatilities`. The current spot price of the underlying. Either this\n      argument or the `forwards` (but not both) must be supplied.\n    forwards: A real `Tensor` of any shape that broadcasts to the shape of\n      `volatilities`. The forwards to maturity. Either this argument or the\n      `spots` must be supplied but both must not be supplied.\n    discount_rates: An optional real `Tensor` of same dtype as the\n      `volatilities`. If not `None`, discount factors are calculated as e^(-rT),\n      where r are the discount rates, or risk free rates.\n      Default value: `None`, equivalent to r = 0 and discount factors = 1 when\n        discount_factors also not given.\n    dividend_rates: An optional real `Tensor` of same dtype as the\n      `volatilities`. The continuous dividend rate on the underliers. May be\n      negative (to indicate costs of holding the underlier).\n      Default value: `None`, equivalent to zero dividends.\n    discount_factors: An optional real `Tensor` of same dtype as the\n      `volatilities`. If not `None`, these are the discount factors to expiry\n      (i.e. e^(-rT)). Mutually exclusive with discount_rate. If neither is\n      given, no discounting is applied (i.e. the undiscounted option price is\n      returned). If `spots` is supplied and `discount_factors` is not `None`\n      then this is also used to compute the forwards to expiry. At most one of\n      discount_rates and discount_factors can be supplied.\n      Default value: `None`, which maps to `-log(discount_factors) / expiries`\n    is_call_options: A boolean `Tensor` of a shape compatible with\n      `volatilities`. Indicates whether the option is a call (if True) or a put\n      (if False). If not supplied, call options are assumed.\n    max_iterations: positive `int`. The maximum number of iterations of Newton's\n      root finding method to find the critical spot price above and below which\n      the pricing formula is different.\n      Default value: 100\n    tolerance: Positive scalar `Tensor`. As with `max_iterations`, used with the\n      Newton root finder to find the critical spot price. The root finder will\n      judge an element to have converged if `|f(x_n) - a|` is less than\n      `tolerance` (where `f` is the target function as defined in [1] and `x_n`\n      is the estimated critical value), or if `x_n` becomes `nan`. When an\n      element is judged to have converged it will no longer be updated. If all\n      elements converge before `max_iterations` is reached then the root finder\n      will return early.\n      Default value: 1e-8\n    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion\n      of any supplied non-`Tensor` arguments to `Tensor`.\n      Default value: None which maps to the default dtype inferred by\n        TensorFlow.\n    name: str. The name for the ops created by this function.\n      Default value: None which is mapped to the default name `adesi_whaley`.\n\n  Returns:\n    A 3-tuple containing the following items in order:\n       (a) option_prices: A `Tensor` of the same shape as `forwards`. The Black\n         Scholes price of the options.\n       (b) converged: A boolean `Tensor` of the same shape as `option_prices`\n         above. Indicates whether the corresponding adesi-whaley approximation\n         has converged to within tolerance.\n       (c) failed: A boolean `Tensor` of the same shape as `option_prices`\n         above. Indicates whether the corresponding options price is NaN or not\n         a finite number. Note that converged being True implies that failed\n         will be false. However, it may happen that converged is False but\n         failed is not True. This indicates the search did not converge in the\n         permitted number of iterations but may converge if the iterations are\n         increased.\n\n  Raises:\n    ValueError:\n      (a) If both `forwards` and `spots` are supplied or if neither is supplied.\n  "
    if (spots is None) == (forwards is None):
        raise ValueError('Either spots or forwards must be supplied but not both.')
    if discount_rates is not None and discount_factors is not None:
        raise ValueError('At most one of discount_rates and discount_factors may be supplied')
    with tf.name_scope(name or 'adesi_whaley'):
        volatilities = tf.convert_to_tensor(volatilities, dtype=dtype, name='volatilities')
        dtype = volatilities.dtype
        strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
        expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
        if discount_rates is not None:
            discount_rates = tf.convert_to_tensor(discount_rates, dtype=dtype, name='discount_rates')
        elif discount_factors is not None:
            discount_factors = tf.convert_to_tensor(discount_factors, dtype=dtype, name='discount_factors')
            discount_rates = -tf.math.log(discount_factors) / expiries
        else:
            discount_rates = tf.constant(0.0, dtype=dtype, name='discount_rates')
        if dividend_rates is not None:
            dividend_rates = tf.convert_to_tensor(dividend_rates, dtype=dtype, name='dividend_rates')
        else:
            dividend_rates = 0
        if forwards is not None:
            spots = tf.convert_to_tensor(forwards * tf.exp(-(discount_rates - dividend_rates) * expiries), dtype=dtype, name='spots')
        else:
            spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
        if is_call_options is not None:
            is_call_options = tf.convert_to_tensor(is_call_options, dtype=tf.bool, name='is_call_options')
        else:
            is_call_options = tf.constant(True, name='is_call_options')
        (am_prices, converged, failed) = _adesi_whaley(sigma=volatilities, x=strikes, t=expiries, s=spots, r=discount_rates, d=dividend_rates, is_call_options=is_call_options, dtype=dtype, max_iterations=max_iterations, tolerance=tolerance)
        eu_prices = vanilla_prices.option_price(volatilities=volatilities, strikes=strikes, expiries=expiries, spots=spots, discount_rates=discount_rates, dividend_rates=dividend_rates, dtype=dtype, name=name)
        calculate_eu = is_call_options & (dividend_rates <= 0)
        converged = tf.where(calculate_eu, True, converged)
        failed = tf.where(calculate_eu, False, failed)
        return (tf.where(calculate_eu, eu_prices, am_prices), converged, failed)

def _adesi_whaley(*, sigma, x, t, r, d, s, is_call_options, max_iterations, tolerance, dtype):
    if False:
        for i in range(10):
            print('nop')
    'Computes American option prices using the Baron-Adesi Whaley formula.'
    sign = tf.where(is_call_options, tf.constant(1, dtype=dtype), tf.constant(-1, dtype=dtype))
    (q2, a2, s_crit, converged, failed) = _adesi_whaley_critical_values(sigma=sigma, x=x, t=t, r=r, d=d, sign=sign, is_call_options=is_call_options, max_iterations=max_iterations, tolerance=tolerance, dtype=dtype)
    eu_prices = vanilla_prices.option_price(volatilities=sigma, strikes=x, expiries=t, spots=s, discount_rates=r, dividend_rates=d, is_call_options=is_call_options, dtype=dtype)
    condition = tf.where(is_call_options, s < s_crit, s > s_crit)
    american_prices = tf.where(t > 0, tf.where(condition, eu_prices + a2 * (s / s_crit) ** q2, (s - x) * sign), tf.where(is_call_options, tf.math.maximum(s - x, 0), tf.math.maximum(x - s, 0)))
    return (american_prices, converged, failed)

def _adesi_whaley_critical_values(*, sigma, x, t, r, d, sign, is_call_options, max_iterations=20, tolerance=1e-08, dtype):
    if False:
        while True:
            i = 10
    'Computes critical value for the Baron-Adesi Whaley approximation.'
    m = 2 * r / sigma ** 2
    n = 2 * (r - d) / sigma ** 2
    k = 1 - tf.exp(-r * t)
    q = _calc_q(n, m, sign, k)

    def value_fn(s_crit):
        if False:
            return 10
        return vanilla_prices.option_price(volatilities=sigma, strikes=x, expiries=t, spots=s_crit, discount_rates=r, dividend_rates=d, is_call_options=is_call_options, dtype=dtype) + sign * (1 - tf.math.exp(-d * t) * _ncdf(sign * _calc_d1(s_crit, x, sigma, r - d, t))) * tf.math.divide_no_nan(s_crit, q) - sign * (s_crit - x)

    def value_and_gradient_func(price):
        if False:
            print('Hello World!')
        return gradient.value_and_gradient(value_fn, price)
    q_inf = _calc_q(n, m, sign)
    s_inf = tf.math.divide_no_nan(x, 1 - tf.math.divide_no_nan(tf.constant(1, dtype=dtype), q_inf))
    h = -(sign * (r - d) * t + 2 * sigma * tf.math.sqrt(t)) * sign * tf.math.divide_no_nan(x, s_inf - x)
    if is_call_options is None:
        s_seed = x + (s_inf - x) * (1 - tf.math.exp(h))
    else:
        s_seed = tf.where(is_call_options, x + (s_inf - x) * (1 - tf.math.exp(h)), s_inf + (x - s_inf) * tf.math.exp(h))
    (s_crit, converged, failed) = root_finder_newton.root_finder(value_and_grad_func=value_and_gradient_func, initial_values=s_seed, max_iterations=max_iterations, tolerance=tolerance, dtype=dtype)
    a = sign * tf.math.divide_no_nan(s_crit, q) * (1 - tf.math.exp(-d * t) * _ncdf(sign * _calc_d1(s_crit, x, sigma, r - d, t)))
    return (q, a, s_crit, converged, failed)

def _calc_d1(s, x, sigma, b, t):
    if False:
        print('Hello World!')
    return tf.math.divide_no_nan(tf.math.log(s / x) + (b + sigma ** 2 / 2) * t, sigma * tf.math.sqrt(t))

def _calc_q(n, m, sign, k=1):
    if False:
        print('Hello World!')
    return (1 - n + sign * tf.math.sqrt((n - 1) ** 2 + tf.math.divide_no_nan(4 * m, k))) / 2

def bjerksund_stensland(*, volatilities: types.RealTensor, strikes: types.RealTensor, expiries: types.RealTensor, spots: types.RealTensor=None, forwards: types.RealTensor=None, discount_rates: types.RealTensor=None, dividend_rates: types.RealTensor=None, discount_factors: types.RealTensor=None, is_call_options: types.BoolTensor=None, modified_boundary: bool=True, dtype: tf.DType=None, name: str=None) -> types.RealTensor:
    if False:
        print('Hello World!')
    'Computes prices of a batch of American options using Bjerksund-Stensland.\n\n  #### Example\n\n  ```python\n    import tf_quant_finance as tff\n    # Price a batch of 5 american call options.\n    volatilities = [0.2, 0.2, 0.2, 0.2, 0.2]\n    forwards = [80.0, 90.0, 100.0, 110.0, 120.0]\n    # Strikes will automatically be broadcasted to shape [5].\n    strikes = np.array([100.0])\n    # Expiries will be broadcast to shape [5], i.e. each option has strike=100\n    # and expiry = 0.25.\n    expiries = 0.25\n    discount_rates = 0.08\n    dividend_rates = 0.12\n    computed_prices = tff.black_scholes.approximations.bjerksund_stensland(\n        volatilities=volatilities,\n        strikes=strikes,\n        expiries=expiries,\n        discount_rates=discount_rates,\n        dividend_rates=dividend_rates,\n        forwards=forwards,\n        is_call_options=True\n        modified_boundary=True)\n  # Expected print output of computed prices:\n  # [ 0.03931201  0.70745419  4.01937524 11.31429842 21.20602005]\n  ```\n\n  #### References:\n  [1] Bjerksund, P. and Stensland G., Closed Form Valuation of American Options,\n      2002\n      https://core.ac.uk/download/pdf/30824897.pdf\n\n  Args:\n    volatilities: Real `Tensor` of any shape and real dtype. The volatilities to\n      expiry of the options to price.\n    strikes: A real `Tensor` of the same dtype and compatible shape as\n      `volatilities`. The strikes of the options to be priced.\n    expiries: A real `Tensor` of same dtype and compatible shape as\n      `volatilities`. The expiry of each option. The units should be such that\n      `expiry * volatility**2` is dimensionless.\n    spots: A real `Tensor` of any shape that broadcasts to the shape of the\n      `volatilities`. The current spot price of the underlying. Either this\n      argument or the `forwards` (but not both) must be supplied.\n    forwards: A real `Tensor` of any shape that broadcasts to the shape of\n      `volatilities`. The forwards to maturity. Either this argument or the\n      `spots` must be supplied but both must not be supplied.\n    discount_rates: An optional real `Tensor` of same dtype as the\n      `volatilities` and of the shape that broadcasts with `volatilities`.\n      If not `None`, discount factors are calculated as e^(-rT),\n      where r are the discount rates, or risk free rates. At most one of\n      discount_rates and discount_factors can be supplied.\n      Default value: `None`, equivalent to r = 0 and discount factors = 1 when\n      discount_factors also not given.\n    dividend_rates: An optional real `Tensor` of same dtype as the\n      `volatilities`. The continuous dividend rate on the underliers. May be\n      negative (to indicate costs of holding the underlier).\n      Default value: `None`, equivalent to zero dividends.\n    discount_factors: An optional real `Tensor` of same dtype as the\n      `volatilities`. If not `None`, these are the discount factors to expiry\n      (i.e. e^(-rT)). Mutually exclusive with discount_rate and cost_of_carry.\n      If neither is given, no discounting is applied (i.e. the undiscounted\n      option price is returned). If `spots` is supplied and `discount_factors`\n      is not `None` then this is also used to compute the forwards to expiry.\n      At most one of discount_rates and discount_factors can be supplied.\n      Default value: `None`, which maps to e^(-rT) calculated from\n      discount_rates.\n    is_call_options: A boolean `Tensor` of a shape compatible with\n      `volatilities`. Indicates whether the option is a call (if True) or a put\n      (if False). If not supplied, call options are assumed.\n    modified_boundary: Python `bool`. Indicates whether the Bjerksund-Stensland\n      1993 algorithm (single boundary) if False or Bjerksund-Stensland 2002\n      algorithm (modified boundary) if True, is to be used.\n    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion\n      of any supplied non-`Tensor` arguments to `Tensor`.\n      Default value: `None` which maps to the default dtype inferred by\n        TensorFlow.\n    name: str. The name for the ops created by this function.\n      Default value: `None` which is mapped to the default name `option_price`.\n\n  Returns:\n    A `Tensor` of the same shape as `forwards`.\n\n  Raises:\n    ValueError: If both `forwards` and `spots` are supplied or if neither is\n      supplied.\n    ValueError: If both `discount_rates` and `discount_factors` is supplied.\n  '
    if (spots is None) == (forwards is None):
        raise ValueError('Either spots or forwards must be supplied but not both.')
    if discount_rates is not None and discount_factors is not None:
        raise ValueError('At most one of discount_rates and discount_factors may be supplied')
    with tf.name_scope(name or 'option_price'):
        strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
        dtype = strikes.dtype
        volatilities = tf.convert_to_tensor(volatilities, dtype=dtype, name='volatilities')
        expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
        if discount_rates is not None:
            discount_rates = tf.convert_to_tensor(discount_rates, dtype=dtype, name='discount_rates')
            discount_factors = tf.exp(-discount_rates * expiries)
        elif discount_factors is not None:
            discount_factors = tf.convert_to_tensor(discount_factors, dtype=dtype, name='discount_factors')
            discount_rates = -tf.math.log(discount_factors) / expiries
        else:
            discount_rates = tf.convert_to_tensor(0.0, dtype=dtype, name='discount_rates')
            discount_factors = tf.convert_to_tensor(1.0, dtype=dtype, name='discount_factors')
        if dividend_rates is None:
            dividend_rates = tf.convert_to_tensor(0.0, dtype=dtype, name='dividend_rates')
        cost_of_carries = discount_rates - dividend_rates
        if forwards is not None:
            forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
            spots = tf.convert_to_tensor(forwards * tf.exp(-cost_of_carries * expiries), dtype=dtype, name='spots')
        else:
            spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
            forwards = spots * tf.exp(cost_of_carries * expiries)
        if is_call_options is not None:
            is_call_options = tf.convert_to_tensor(is_call_options, dtype=tf.bool, name='is_call_options')
        else:
            is_call_options = tf.constant(True, name='is_call_options')
        if modified_boundary:
            bjerksund_stensland_model = _call_2002
        else:
            bjerksund_stensland_model = _call_1993
        american_prices = tf.where(tf.math.logical_and(cost_of_carries >= discount_rates, is_call_options), vanilla_prices.option_price(volatilities=volatilities, strikes=strikes, expiries=expiries, spots=spots, discount_rates=discount_rates, dividend_rates=dividend_rates, is_call_options=is_call_options), tf.where(is_call_options, tf.where(expiries > 0, bjerksund_stensland_model(spots, strikes, expiries, discount_rates, cost_of_carries, volatilities), tf.math.maximum(spots - strikes, 0)), tf.where(expiries > 0, bjerksund_stensland_model(strikes, spots, expiries, discount_rates - cost_of_carries, -cost_of_carries, volatilities), tf.math.maximum(strikes - spots, 0))))
        return american_prices

def _call_2002(s, k, t, r, b, sigma):
    if False:
        return 10
    'Calculates the approximate value of an American call option.'
    dtype = sigma.dtype
    t1 = 0.5 * (tf.math.sqrt(tf.constant(5.0, dtype=dtype)) - 1) * t
    beta = 0.5 - b / sigma ** 2 + tf.math.sqrt((b / sigma ** 2 - 0.5) ** 2 + 2 * r / sigma ** 2)
    x1 = _boundary1_2002(k, t, r, b, sigma, beta)
    x2 = _boundary2_2002(k, t, r, b, sigma, beta)
    alpha1 = (x1 - k) * x1 ** (-beta)
    alpha2 = (x2 - k) * x2 ** (-beta)
    return tf.where(s >= x2, s - k, alpha2 * s ** beta - alpha2 * _phi_1993(s, t1, beta, x2, x2, r, b, sigma) + _phi_1993(s, t1, 1, x2, x2, r, b, sigma) - _phi_1993(s, t1, 1, x1, x2, r, b, sigma) - k * _phi_1993(s, t1, 0, x2, x2, r, b, sigma) + k * _phi_1993(s, t1, 0, x1, x2, r, b, sigma) + alpha1 * _phi_1993(s, t1, beta, x1, x2, r, b, sigma) - alpha1 * _psi_2002(s, t, beta, x1, x2, x1, t1, r, b, sigma) + _psi_2002(s, t, 1, x1, x2, x1, t1, r, b, sigma) - _psi_2002(s, t, 1, k, x2, x1, t1, r, b, sigma) - k * _psi_2002(s, t, 0, x1, x2, x1, t1, r, b, sigma) + k * _psi_2002(s, t, 0, k, x2, x1, t1, r, b, sigma))

def _psi_2002(s, t, gamma, h, x2, x1, t1, r, b, sigma):
    if False:
        print('Hello World!')
    'Compute the psi function.'
    d1 = -(tf.math.log(s / x1) + (b + (gamma - 0.5) * sigma ** 2) * t1) / (sigma * tf.math.sqrt(t1))
    d2 = -(tf.math.log(x2 ** 2 / (s * x1)) + (b + (gamma - 0.5) * sigma ** 2) * t1) / (sigma * tf.math.sqrt(t1))
    d3 = -(tf.math.log(s / x1) - (b + (gamma - 0.5) * sigma ** 2) * t1) / (sigma * tf.math.sqrt(t1))
    d4 = -(tf.math.log(x2 ** 2 / (s * x1)) - (b + (gamma - 0.5) * sigma ** 2) * t1) / (sigma * tf.math.sqrt(t1))
    f1 = -(tf.math.log(s / h) + (b + (gamma - 0.5) * sigma ** 2) * t) / (sigma * tf.math.sqrt(t))
    f2 = -(tf.math.log(x2 ** 2 / (s * h)) + (b + (gamma - 0.5) * sigma ** 2) * t) / (sigma * tf.math.sqrt(t))
    f3 = -(tf.math.log(x1 ** 2 / (s * h)) + (b + (gamma - 0.5) * sigma ** 2) * t) / (sigma * tf.math.sqrt(t))
    f4 = -(tf.math.log(s * x1 ** 2 / (h * x2 ** 2)) + (b + (gamma - 0.5) * sigma ** 2) * t) / (sigma * tf.math.sqrt(t))
    kappa = 2 * b / sigma ** 2 + (2 * gamma - 1)
    lambd = -r + gamma * b + 0.5 * gamma * (gamma - 1) * sigma ** 2
    return tf.math.exp(lambd * t) * s ** gamma * (_cbnd(d1, f1, tf.math.sqrt(t1 / t)) - (x2 / s) ** kappa * _cbnd(d2, f2, tf.math.sqrt(t1 / t)) - (x1 / s) ** kappa * _cbnd(d3, f3, -tf.math.sqrt(t1 / t)) + (x1 / x2) ** kappa * _cbnd(d4, f4, -tf.math.sqrt(t1 / t)))

def _boundary1_2002(k, t, r, b, sigma, beta):
    if False:
        i = 10
        return i + 15
    'Computes the first boundary or trigger price for the Bjerksund-Stensland 2002 algorithm.'
    dtype = sigma.dtype
    t1 = 0.5 * (tf.math.sqrt(tf.constant(5.0, dtype=dtype)) - 1) * t
    b0 = tf.math.maximum(k, r / (r - b) * k)
    binfinity = beta / (beta - 1) * k
    ht1 = -(b * (t - t1) + 2 * sigma * tf.math.sqrt(t - t1)) * (k ** 2 / ((binfinity - b0) * b0))
    return b0 + (binfinity - b0) * (1 - tf.math.exp(ht1))

def _boundary2_2002(k, t, r, b, sigma, beta):
    if False:
        i = 10
        return i + 15
    'Computes the second boundary or trigger price for the Bjerksund-Stensland 2002 algorithm.'
    b0 = tf.math.maximum(k, r / (r - b) * k)
    binfinity = beta / (beta - 1) * k
    ht2 = -(b * t + 2 * sigma * tf.math.sqrt(t)) * (k ** 2 / ((binfinity - b0) * b0))
    return b0 + (binfinity - b0) * (1 - tf.math.exp(ht2))

def _call_1993(s, k, t, r, b, sigma):
    if False:
        for i in range(10):
            print('nop')
    'Calculates the approximate value of an American call option.'
    beta = 0.5 - b / sigma ** 2 + tf.math.sqrt((b / sigma ** 2 - 0.5) ** 2 + 2 * r / sigma ** 2)
    alpha = (_boundary_1993(k, t, r, b, sigma, beta) - k) * _boundary_1993(k, t, r, b, sigma, beta) ** (-beta)
    return tf.where(s >= _boundary_1993(k, t, r, b, sigma, beta), s - k, alpha * s ** beta - alpha * _phi_1993(s, t, beta, _boundary_1993(k, t, r, b, sigma, beta), _boundary_1993(k, t, r, b, sigma, beta), r, b, sigma) + _phi_1993(s, t, 1, _boundary_1993(k, t, r, b, sigma, beta), _boundary_1993(k, t, r, b, sigma, beta), r, b, sigma) - _phi_1993(s, t, 1, k, _boundary_1993(k, t, r, b, sigma, beta), r, b, sigma) - k * _phi_1993(s, t, 0, _boundary_1993(k, t, r, b, sigma, beta), _boundary_1993(k, t, r, b, sigma, beta), r, b, sigma) + k * _phi_1993(s, t, 0, k, _boundary_1993(k, t, r, b, sigma, beta), r, b, sigma))

def _phi_1993(s, t, gamma, h, x, r, b, sigma):
    if False:
        i = 10
        return i + 15
    'Computes the value of the Phi (7) function in reference [1].'
    kappa = 2 * b / sigma ** 2 + (2 * gamma - 1)
    d1 = -(tf.math.log(s / h) + (b + (gamma - 0.5) * sigma ** 2) * t) / (sigma * tf.math.sqrt(t))
    d2 = -(tf.math.log(x ** 2 / (s * h)) + (b + (gamma - 0.5) * sigma ** 2) * t) / (sigma * tf.math.sqrt(t))
    lambd = -r + gamma * b + 0.5 * gamma * (gamma - 1) * sigma ** 2
    return tf.math.exp(lambd * t) * s ** gamma * (_ncdf(d1) - (x / s) ** kappa * _ncdf(d2))

def _boundary_1993(k, t, r, b, sigma, beta):
    if False:
        print('Hello World!')
    'Computes the early exercise boundary (10) in reference [1].'
    b0 = tf.math.maximum(k, r / (r - b) * k)
    binfinity = beta / (beta - 1) * k
    ht = -(b * t + 2 * sigma * tf.math.sqrt(t)) * (k ** 2 / ((binfinity - b0) * b0))
    return b0 + (binfinity - b0) * (1 - tf.math.exp(ht))

def _ncdf(x):
    if False:
        i = 10
        return i + 15
    return (tf.math.erf(x / _SQRT_2) + 1) / 2

def _npdf(x):
    if False:
        for i in range(10):
            print('nop')
    return tf.math.exp(-0.5 * x ** 2) / _SQRT_2_PI

def _cbnd(dh, dk, rho):
    if False:
        return 10
    'Computes values for the cumulative standard bivariate normal distribution.\n\n  More specifically, compultes `P(x > dh, y > dk)` where `x` and `y` are\n  standard normal variables with correlation `rho`.\n\n  Args:\n    dh: A real `Tensor` representing lower integration limits for `x`.\n    dk: A `Tensor` of the same dtype as `dh` and of compatible shape\n      representing lower integration limits `y`.\n    rho: A `Tensor` of the same dtype as `dh` and of compatible shape\n      representing correlation coefficients.\n\n  Returns:\n    A `Tensor` of cumulative distribution function values.\n\n  #### References:\n  [1] Genz, A., Numerical Computation of Rectangular Bivariate and Trivariate\n        Normal and t Probabilities, 2004\n    http://www.math.wsu.edu/faculty/genz/papers/bvnt.pdf\n  '
    dtype = rho.dtype
    h = tf.cast(-dh, dtype=dtype)
    k = tf.cast(-dk, dtype=dtype)
    hk = h * k
    bvn = tf.zeros_like(hk)
    hs = (h * h + k * k) / 2
    asr = tf.math.asin(rho)

    def transformed_bvn(hk, hs, asr):
        if False:
            for i in range(10):
                print('nop')

        def transformed_bvn_distribution(x):
            if False:
                i = 10
                return i + 15
            hk_exp = tf.expand_dims(hk, axis=-1)
            hs_exp = tf.expand_dims(hs, axis=-1)
            asr_exp = tf.expand_dims(asr, axis=-1)
            sn1 = tf.math.sin(asr_exp * (1 * x + 1) / 2)
            return tf.math.exp((hk_exp * sn1 - hs_exp) / (1 - sn1 * sn1))
        ones = tf.ones_like(hk)
        res = integration.gauss_legendre(func=transformed_bvn_distribution, lower=-ones, upper=ones, num_points=20, dtype=dtype)
        return res
    bvn = bvn + transformed_bvn(hk, hs, asr)
    bvn = bvn * asr / (4 * np.pi)
    bvn = bvn + _ncdf(-h) * _ncdf(-k)
    return bvn