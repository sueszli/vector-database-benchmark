"""Calculation of the Black-Scholes implied volatility via Newton's method."""
import numpy as np
import tensorflow.compat.v2 as tf
from tf_quant_finance.black_scholes import implied_vol_approximation as approx
from tf_quant_finance.black_scholes import implied_vol_utils as utils
from tf_quant_finance.math.root_search import newton
_SQRT_2 = np.sqrt(2.0, dtype=np.float64)
_SQRT_2_PI = np.sqrt(2 * np.pi, dtype=np.float64)
_NORM_PDF_AT_ZERO = 1.0 / _SQRT_2_PI

def _cdf(x):
    if False:
        i = 10
        return i + 15
    return (tf.math.erf(x / _SQRT_2) + 1) / 2

def _pdf(x):
    if False:
        i = 10
        return i + 15
    return tf.math.exp(-0.5 * x ** 2) / _SQRT_2_PI

def implied_vol(*, prices, strikes, expiries, spots=None, forwards=None, discount_factors=None, is_call_options=None, initial_volatilities=None, underlying_distribution=utils.UnderlyingDistribution.LOG_NORMAL, tolerance=1e-08, max_iterations=20, validate_args=False, dtype=None, name=None):
    if False:
        print('Hello World!')
    "Computes implied volatilities from given call or put option prices.\n\n  This method applies a Newton root search algorithm to back out the implied\n  volatility given the price of either a put or a call option.\n\n  The implementation assumes that each cell in the supplied tensors corresponds\n  to an independent volatility to find.\n\n  Args:\n    prices: A real `Tensor` of any shape. The prices of the options whose\n      implied vol is to be calculated.\n    strikes: A real `Tensor` of the same dtype as `prices` and a shape that\n      broadcasts with `prices`. The strikes of the options.\n    expiries: A real `Tensor` of the same dtype as `prices` and a shape that\n      broadcasts with `prices`. The expiry for each option. The units should be\n      such that `expiry * volatility**2` is dimensionless.\n    spots: A real `Tensor` of any shape that broadcasts to the shape of the\n      `prices`. The current spot price of the underlying. Either this argument\n      or the `forwards` (but not both) must be supplied.\n      Default value: None.\n    forwards: A real `Tensor` of any shape that broadcasts to the shape of\n      `prices`. The forwards to maturity. Either this argument or the `spots`\n      must be supplied but both must not be supplied.\n      Default value: None.\n    discount_factors: An optional real `Tensor` of same dtype as the `prices`.\n      If not None, these are the discount factors to expiry (i.e. e^(-rT)). If\n      None, no discounting is applied (i.e. it is assumed that the undiscounted\n      option prices are provided ). If `spots` is supplied and\n      `discount_factors` is not None then this is also used to compute the\n      forwards to expiry.\n      Default value: None, equivalent to discount factors = 1.\n    is_call_options: A boolean `Tensor` of a shape compatible with `prices`.\n      Indicates whether the option is a call (if True) or a put (if False). If\n      not supplied, call options are assumed.\n      Default value: None.\n    initial_volatilities: A real `Tensor` of the same shape and dtype as\n      `forwards`. The starting positions for Newton's method.\n      Default value: None. If not supplied, the starting point is chosen using\n        the Stefanica-Radoicic scheme. See `polya_approx.implied_vol` for\n        details.\n      Default value: None.\n    underlying_distribution: Enum value of ImpliedVolUnderlyingDistribution to\n      select the distribution of the underlying.\n      Default value: UnderlyingDistribution.LOG_NORMAL\n    tolerance: `float`. The root finder will stop where this tolerance is\n      crossed.\n    max_iterations: `int`. The maximum number of iterations of Newton's method.\n      Default value: 20.\n    validate_args: A Python bool. If True, indicates that arguments should be\n      checked for correctness before performing the computation. The checks\n      performed are: (1) Forwards and strikes are positive. (2) The prices\n        satisfy the arbitrage bounds (i.e. for call options, checks the\n        inequality `max(F-K, 0) <= Price <= F` and for put options, checks that\n        `max(K-F, 0) <= Price <= K`.). (3) Checks that the prices are not too\n        close to the bounds. It is numerically unstable to compute the implied\n        vols from options too far in the money or out of the money.\n      Default value: False.\n    dtype: `tf.Dtype` to use when converting arguments to `Tensor`s. If not\n      supplied, the default TensorFlow conversion will take place. Note that\n      this argument does not do any casting for `Tensor`s or numpy arrays.\n      Default value: None.\n    name: (Optional) Python str. The name prefixed to the ops created by this\n      function. If not supplied, the default name 'implied_vol' is used.\n      Default value: None.\n\n  Returns:\n    A 3-tuple containing the following items in order:\n       (a) implied_vols: A `Tensor` of the same dtype as `prices` and shape as\n         the common broadcasted shape of\n         `(prices, spots/forwards, strikes, expiries)`. The implied vols as\n         inferred by the algorithm. It is possible that the search may not have\n         converged or may have produced NaNs. This can be checked for using the\n         following return values.\n       (b) converged: A boolean `Tensor` of the same shape as `implied_vols`\n         above. Indicates whether the corresponding vol has converged to within\n         tolerance.\n       (c) failed: A boolean `Tensor` of the same shape as `implied_vols` above.\n         Indicates whether the corresponding vol is NaN or not a finite number.\n         Note that converged being True implies that failed will be false.\n         However, it may happen that converged is False but failed is not True.\n         This indicates the search did not converge in the permitted number of\n         iterations but may converge if the iterations are increased.\n\n  Raises:\n    ValueError: If both `forwards` and `spots` are supplied or if neither is\n      supplied.\n  "
    if (spots is None) == (forwards is None):
        raise ValueError('Either spots or forwards must be supplied but not both.')
    with tf.compat.v1.name_scope(name, default_name='implied_vol', values=[prices, spots, forwards, strikes, expiries, discount_factors, is_call_options, initial_volatilities]):
        prices = tf.convert_to_tensor(prices, dtype=dtype, name='prices')
        dtype = prices.dtype
        strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
        expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
        if discount_factors is None:
            discount_factors = tf.convert_to_tensor(1.0, dtype=dtype, name='discount_factors')
        else:
            discount_factors = tf.convert_to_tensor(discount_factors, dtype=dtype, name='discount_factors')
        if forwards is not None:
            forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
        else:
            spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
            forwards = spots / discount_factors
        if initial_volatilities is None:
            if underlying_distribution is utils.UnderlyingDistribution.LOG_NORMAL:
                initial_volatilities = approx.implied_vol(prices=prices, strikes=strikes, expiries=expiries, forwards=forwards, discount_factors=discount_factors, is_call_options=is_call_options, validate_args=validate_args)
            elif underlying_distribution is utils.UnderlyingDistribution.NORMAL:
                initial_volatilities = prices / _NORM_PDF_AT_ZERO
        else:
            initial_volatilities = tf.convert_to_tensor(initial_volatilities, dtype=dtype, name='initial_volatilities')
        (implied_vols, converged, failed) = _newton_implied_vol(prices, strikes, expiries, forwards, discount_factors, is_call_options, initial_volatilities, underlying_distribution, tolerance, max_iterations)
        return (implied_vols, converged, failed)

def _newton_implied_vol(prices, strikes, expiries, forwards, discount_factors, is_call_options, initial_volatilities, underlying_distribution, tolerance, max_iterations):
    if False:
        for i in range(10):
            print('nop')
    "Uses Newton's method to find Black Scholes implied volatilities of options.\n\n  Finds the volatility implied under the Black Scholes option pricing scheme for\n  a set of European options given observed market prices. The implied volatility\n  is found via application of Newton's algorithm for locating the root of a\n  differentiable function.\n\n  The implementation assumes that each cell in the supplied tensors corresponds\n  to an independent volatility to find.\n\n  Args:\n    prices: A real `Tensor` of any shape. The prices of the options whose\n      implied vol is to be calculated.\n    strikes: A real `Tensor` of the same dtype as `prices` and a shape that\n      broadcasts with `prices`. The strikes of the options.\n    expiries: A real `Tensor` of the same dtype as `prices` and a shape that\n      broadcasts with `prices`. The expiry for each option. The units should be\n      such that `expiry * volatility**2` is dimensionless.\n    forwards: A real `Tensor` of any shape that broadcasts to the shape of\n      `prices`. The forwards to maturity.\n    discount_factors: An optional real `Tensor` of same dtype as the `prices`.\n      If not None, these are the discount factors to expiry (i.e. e^(-rT)). If\n      None, no discounting is applied (i.e. it is assumed that the undiscounted\n      option prices are provided ).\n    is_call_options: A boolean `Tensor` of a shape compatible with `prices`.\n      Indicates whether the option is a call (if True) or a put (if False). If\n      not supplied, call options are assumed.\n    initial_volatilities: A real `Tensor` of the same shape and dtype as\n      `forwards`. The starting positions for Newton's method.\n    underlying_distribution: Enum value of ImpliedVolUnderlyingDistribution to\n      select the distribution of the underlying.\n    tolerance: `float`. The root finder will stop where this tolerance is\n      crossed.\n    max_iterations: `int`. The maximum number of iterations of Newton's method.\n\n  Returns:\n    A three tuple of `Tensor`s, each the same shape as `forwards`. It\n    contains the implied volatilities (same dtype as `forwards`), a boolean\n    `Tensor` indicating whether the corresponding implied volatility converged,\n    and a boolean `Tensor` which is true where the corresponding implied\n    volatility is not a finite real number.\n  "
    if underlying_distribution is utils.UnderlyingDistribution.LOG_NORMAL:
        pricer = _make_black_lognormal_objective_and_vega_func(prices, forwards, strikes, expiries, is_call_options, discount_factors)
    elif underlying_distribution is utils.UnderlyingDistribution.NORMAL:
        pricer = _make_bachelier_objective_and_vega_func(prices, forwards, strikes, expiries, is_call_options, discount_factors)
    results = newton.root_finder(pricer, initial_volatilities, max_iterations=max_iterations, tolerance=tolerance)
    return results

def _get_normalizations(prices, forwards, strikes, discount_factors):
    if False:
        for i in range(10):
            print('nop')
    "Returns the normalized prices, normalization factors, and discount_factors.\n\n  The normalization factors is the larger of strikes and forwards.\n  If `discount_factors` is not None, these are the discount factors to expiry.\n  If None, no discounting is applied and 1's are returned.\n\n  Args:\n    prices: A real `Tensor` of any shape. The observed market prices of the\n      assets.\n    forwards: A real `Tensor` of the same shape and dtype as `prices`. The\n      current forward prices to expiry.\n    strikes: A real `Tensor` of the same shape and dtype as `prices`. The strike\n      prices of the options.\n    discount_factors: A real `Tensor` of same dtype as the `prices`.\n\n  Returns:\n    the normalized prices, normalization factors, and discount_factors.\n  "
    strikes_abs = tf.abs(strikes)
    forwards_abs = tf.abs(forwards)
    orientations = strikes_abs >= forwards_abs
    normalization = tf.where(orientations, strikes_abs, forwards_abs)
    normalization = tf.where(tf.equal(normalization, 0), tf.ones_like(normalization), normalization)
    normalized_prices = prices / normalization
    if discount_factors is not None:
        normalized_prices /= discount_factors
    else:
        discount_factors = tf.ones_like(normalized_prices)
    return (normalized_prices, normalization, discount_factors)

def _make_black_lognormal_objective_and_vega_func(prices, forwards, strikes, expiries, is_call_options, discount_factors):
    if False:
        return 10
    "Produces an objective and vega function for the Black Scholes model.\n\n  The returned function maps volatilities to a tuple of objective function\n  values and their gradients with respect to the volatilities. The objective\n  function is the difference between Black Scholes prices and observed market\n  prices, whereas the gradient is called vega of the option. That is:\n\n  ```\n  g(s) = (f(s) - a, f'(s))\n  ```\n\n  Where `g` is the returned function taking volatility parameter `s`, `f` the\n  Black Scholes price with all other variables curried and `f'` its derivative,\n  and `a` the observed market prices of the options. Hence `g` calculates the\n  information necessary for finding the volatility implied by observed market\n  prices for options with given terms using first order methods.\n\n  #### References\n  [1] Hull, J., 2018. Options, Futures, and Other Derivatives. Harlow, England.\n  Pearson. (p.358 - 361)\n\n  Args:\n    prices: A real `Tensor` of any shape. The observed market prices of the\n      assets.\n    forwards: A real `Tensor` of the same shape and dtype as `prices`. The\n      current forward prices to expiry.\n    strikes: A real `Tensor` of the same shape and dtype as `prices`. The strike\n      prices of the options.\n    expiries: A real `Tensor` of same shape and dtype as `forwards`. The expiry\n      for each option. The units should be such that `expiry * volatility**2` is\n      dimensionless.\n    is_call_options: A boolean `Tensor` of same shape and dtype as `forwards`.\n      Positive one where option is a call, negative one where option is a put.\n    discount_factors: A real `Tensor` of the same shape and dtype as `forwards`.\n      The total discount factors to apply.\n\n  Returns:\n    A function from volatilities to a Black Scholes objective and its\n    derivative (which is coincident with Vega).\n  "
    (normalized_prices, normalization, discount_factors) = _get_normalizations(prices, forwards, strikes, discount_factors)
    norm_forwards = forwards / normalization
    norm_strikes = strikes / normalization
    lnz = tf.math.log(forwards) - tf.math.log(strikes)
    sqrt_t = tf.sqrt(expiries)
    if is_call_options is not None:
        is_call_options = tf.convert_to_tensor(is_call_options, dtype=tf.bool, name='is_call_options')

    def _black_objective_and_vega(volatilities):
        if False:
            return 10
        'Calculate the Black Scholes price and vega for a given volatility.\n\n    This method returns normalized results.\n\n    Args:\n      volatilities: A real `Tensor` of same shape and dtype as `forwards`. The\n        volatility to expiry.\n\n    Returns:\n      A tuple containing (value, gradient) of the black scholes price, both of\n        which are `Tensor`s of the same shape and dtype as `volatilities`.\n    '
        vol_t = volatilities * sqrt_t
        d1 = lnz / vol_t + vol_t / 2
        d2 = d1 - vol_t
        implied_prices = norm_forwards * _cdf(d1) - norm_strikes * _cdf(d2)
        if is_call_options is not None:
            put_prices = implied_prices - norm_forwards + norm_strikes
            implied_prices = tf.where(tf.broadcast_to(is_call_options, tf.shape(put_prices)), implied_prices, put_prices)
        vega = norm_forwards * _pdf(d1) * sqrt_t / discount_factors
        return (implied_prices - normalized_prices, vega)
    return _black_objective_and_vega

def _make_bachelier_objective_and_vega_func(prices, forwards, strikes, expiries, is_call_options, discount_factors):
    if False:
        return 10
    "Produces an objective and vega function for the Bachelier model.\n\n  The returned function maps volatilities to a tuple of objective function\n  values and their gradients with respect to the volatilities. The objective\n  function is the difference between model implied prices and observed market\n  prices, whereas the gradient is called vega of the option. That is:\n\n  ```\n  g(s) = (f(s) - a, f'(s))\n  ```\n\n  Where `g` is the returned function taking volatility parameter `s`, `f` the\n  Black Scholes price with all other variables curried and `f'` its derivative,\n  and `a` the observed market prices of the options. Hence `g` calculates the\n  information necessary for finding the volatility implied by observed market\n  prices for options with given terms using first order methods.\n\n  #### References\n  [1] Wenqing H., 2013. Risk Measures with Normal Distributed Black Options\n  Pricing Model. Mälardalen University, Sweden. (p. 10 - 17)\n\n  Args:\n    prices: A real `Tensor` of any shape. The observed market prices of the\n      options.\n    forwards: A real `Tensor` of the same shape and dtype as `prices`. The\n      current forward prices to expiry.\n    strikes: A real `Tensor` of the same shape and dtype as `prices`. The strike\n      prices of the options.\n    expiries: A real `Tensor` of same shape and dtype as `forwards`. The expiry\n      for each option. The units should be such that `expiry * volatility**2` is\n      dimensionless.\n    is_call_options: A boolean `Tensor` of same shape and dtype as `forwards`.\n      `True` for call options and `False` for put options.\n    discount_factors: A real `Tensor` of the same shape and dtype as `forwards`.\n      The total discount factors to option expiry.\n\n  Returns:\n    A function from volatilities to a Black Scholes objective and its\n    derivative (which is coincident with Vega).\n  "
    (normalized_prices, normalization, discount_factors) = _get_normalizations(prices, forwards, strikes, discount_factors)
    norm_forwards = forwards / normalization
    norm_strikes = strikes / normalization
    sqrt_t = tf.sqrt(expiries)
    if is_call_options is not None:
        is_call_options = tf.convert_to_tensor(is_call_options, dtype=tf.bool, name='is_call_options')

    def _objective_and_vega(volatilities):
        if False:
            for i in range(10):
                print('nop')
        'Calculate the Bachelier price and vega for a given volatility.\n\n    This method returns normalized results.\n\n    Args:\n      volatilities: A real `Tensor` of same shape and dtype as `forwards`. The\n        volatility to expiry.\n\n    Returns:\n      A tuple containing (value, gradient) of the black scholes price, both of\n        which are `Tensor`s of the same shape and dtype as `volatilities`.\n    '
        vols = volatilities * sqrt_t / normalization
        d1 = (norm_forwards - norm_strikes) / vols
        implied_prices = (norm_forwards - norm_strikes) * _cdf(d1) + vols * _pdf(d1)
        if is_call_options is not None:
            put_prices = implied_prices - norm_forwards + norm_strikes
            implied_prices = tf.where(is_call_options, implied_prices, put_prices)
        vega = _pdf(d1) * sqrt_t / discount_factors / normalization
        return (implied_prices - normalized_prices, vega)
    return _objective_and_vega