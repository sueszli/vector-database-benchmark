"""Methods to approximate the implied vol of options from market prices."""
import numpy as np
import tensorflow.compat.v2 as tf

def implied_vol(*, prices, strikes, expiries, spots=None, forwards=None, discount_factors=None, is_call_options=None, validate_args=False, polya_factor=2 / np.pi, dtype=None, name=None):
    if False:
        print('Hello World!')
    "Approximates the implied vol using the Stefanica-Radiocic algorithm.\n\n  Finds an approximation to the implied vol using the Polya approximation for\n  the Normal CDF. This algorithm was described by Stefanica and Radiocic in\n  ref [1]. They show that if the Normal CDFs appearing in the Black Scholes\n  formula for the option price are replaced with Polya's approximation, the\n  implied vol can be solved for analytically. The Polya approximation produces\n  absolute errors of less than 0.003 and the resulting implied vol is fairly\n  close to the true value. For practical purposes, this may not be accurate\n  enough so this result should be used as a starting point for some method with\n  controllable tolerance (e.g. a root finder).\n\n  #### References:\n  [1]: Dan Stefanica and Rados Radoicic. An explicit implied volatility formula.\n    International Journal of Theoretical and Applied Finance,\n    Vol. 20, no. 7, 2017.\n    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2908494\n  [2]: Omar Eidous, Samar Al-Salman. One-term approximation for Normal\n    distribution function. Mathematics and Statistics 4(1), 2016.\n    http://www.hrpub.org/download/20160229/MS2-13405192.pdf\n\n  Args:\n    prices: A real `Tensor` of any shape. The prices of the options whose\n      implied vol is to be calculated.\n    strikes: A real `Tensor` of the same dtype as `prices` and a shape that\n      broadcasts with `prices`. The strikes of the options.\n    expiries: A real `Tensor` of the same dtype as `prices` and a shape that\n      broadcasts with `prices`. The expiry for each option. The units should\n      be such that `expiry * volatility**2` is dimensionless.\n    spots: A real `Tensor` of any shape that broadcasts to the shape\n      of the `prices`. The current spot price of the underlying. Either this\n      argument or the `forwards` (but not both) must be supplied.\n    forwards: A real `Tensor` of any shape that broadcasts to the shape of\n      `prices`. The forwards to maturity. Either this argument or the `spots`\n      must be supplied but both must not be supplied.\n    discount_factors: An optional real `Tensor` of same dtype as the `prices`.\n      If not None, these are the discount factors to expiry (i.e. e^(-rT)).\n      If None, no discounting is applied (i.e. it is assumed that the\n      undiscounted option prices are provided ). If `spots` is supplied and\n      `discount_factors` is not None then this is also used to compute the\n      forwards to expiry.\n      Default value: None, equivalent to discount factors = 1.\n    is_call_options: A boolean `Tensor` of a shape compatible with `prices`.\n      Indicates whether the option is a call (if True) or a put (if False).\n      If not supplied, call options are assumed.\n    validate_args: A Python bool. If True, indicates that arguments should be\n      checked for correctness before performing the computation. The checks\n      performed are: (1) Forwards/spots and strikes are positive. (2) The prices\n        satisfy the arbitrage bounds (i.e. for call options, checks the\n        inequality `max(F-K, 0) <= Price <= F` and for put options, checks that\n        `max(K-F, 0) <= Price <= K`.). (3) Checks that the prices are not too\n        close to the bounds. It is numerically unstable to compute the implied\n        vols from options too far in the money or out of the money.\n      Default value: False\n    polya_factor: A real scalar. The coefficient to use in the\n      approximation for the Normal CDF. The approximation is: `N(x) ~ 0.5 + 0.5\n        * sign(x) * sqrt[ 1 - exp(-k * x**2) ]` where `k` is the coefficient\n        supplied with `polya_factor`. The original Polya approximation has the\n        value `2 / pi` and this is approximation used in Ref [1]. However, as\n        described in Ref [2], a slightly more accurate approximation is achieved\n        if we use the value of `k=5/8`).\n    dtype: `tf.Dtype` to use when converting arguments to `Tensor`s. If not\n      supplied, the default TensorFlow conversion will take place. Note that\n      this argument does not do any casting for `Tensor`s or numpy arrays.\n      Default value: None.\n    name: (Optional) Python str. The name prefixed to the ops created by this\n      function. If not supplied, the default name 'implied_vol' is\n      used.\n      Default value: None\n\n  Returns:\n    implied_vols: A `Tensor` of the same dtype as `prices` and shape as the\n      common broadcasted shape of `(prices, spots/forwards, strikes, expiries)`.\n      The approximate implied total volatilities computed using the Polya\n      approximation method.\n\n  Raises:\n    ValueError: If both `forwards` and `spots` are supplied or if neither is\n      supplied.\n  "
    if (spots is None) == (forwards is None):
        raise ValueError('Either spots or forwards must be supplied but not both.')
    name = name or 'implied_vol'
    with tf.name_scope(name):
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
        control_inputs = None
        if validate_args:
            control_inputs = _validate_args_control_deps(prices, forwards, strikes, expiries, discount_factors, is_call_options)
        with tf.compat.v1.control_dependencies(control_inputs):
            adjusted_strikes = strikes * discount_factors
            normalized_prices = prices / adjusted_strikes
            normalized_forwards = forwards / strikes
            return _approx_implied_vol_polya(normalized_prices, normalized_forwards, expiries, is_call_options, polya_factor)

def _validate_args_control_deps(prices, forwards, strikes, expiries, discount_factors, is_call_options):
    if False:
        for i in range(10):
            print('nop')
    'Returns assertions for no-arbitrage conditions on the prices.'
    epsilon = tf.convert_to_tensor(1e-08, dtype=prices.dtype)
    forwards_positive = tf.debugging.assert_positive(forwards, message='Forwards positive')
    strikes_positive = tf.debugging.assert_positive(strikes, message='Strikes positive')
    expiries_positive = tf.debugging.assert_non_negative(expiries, message='Expiries positive')
    put_lower_bounds = tf.nn.relu(strikes - forwards)
    call_lower_bounds = tf.nn.relu(forwards - strikes)
    if is_call_options is not None:
        is_call_options = tf.convert_to_tensor(is_call_options, dtype=tf.bool, name='is_call_options')
        lower_bounds = tf.where(is_call_options, x=call_lower_bounds, y=put_lower_bounds)
        upper_bounds = tf.where(is_call_options, x=forwards, y=strikes)
    else:
        lower_bounds = call_lower_bounds
        upper_bounds = forwards
    undiscounted_prices = prices / discount_factors
    bounds_satisfied = [tf.debugging.assert_less_equal(lower_bounds, undiscounted_prices, message='Price lower bound'), tf.debugging.assert_greater_equal(upper_bounds, undiscounted_prices, message='Price upper bound')]
    not_too_close_to_bounds = [tf.debugging.assert_greater(tf.math.abs(undiscounted_prices - lower_bounds), epsilon, message='Close to lower bound'), tf.debugging.assert_greater(tf.math.abs(undiscounted_prices - upper_bounds), epsilon, message='Close to upper bound')]
    return [expiries_positive, forwards_positive, strikes_positive] + bounds_satisfied + not_too_close_to_bounds

def _approx_implied_vol_polya(normalized_prices, normalized_forwards, expiries, is_call_options, polya_factor):
    if False:
        print('Hello World!')
    'Computes approximate implied vol using the Stefanica-Radoicic algorithm.\n\n  ## Implementation Notes\n  The mapping between the notation used in the reference paper and the code\n  below is as follows:\n    y -> log_normalized_forwards\n    alpha_c -> normalized_prices\n  This notation is used in the in-line comments.\n\n  Args:\n    normalized_prices: `Tensor` of real dtype and any shape. The prices of the\n      options to be inverted. Normalization means that the raw price is divided\n      by the strike discounted to the present.\n    normalized_forwards: `Tensor` or same dtype and shape as `normalized_prices`\n      The forwards divided by the strike of the options.\n    expiries: A real `Tensor` of same shape and dtype as `normalized_forwards`.\n      The expiry for each option.\n    is_call_options: Boolean `Tensor` of same shape as `normalized_prices` or\n      None. Indicates whether a price is for a call option (if True) or a put\n      option (if False). If None is specified, it is assumed that all the\n      options are call options.\n    polya_factor: Scalar `Tensor` of same dtype as `normalized_prices`. This is\n      the factor to use for approximating the normal CDF in a Polya-like\n      expression. Polya approximation is: N(x) ~ 0.5 + sign(x) sqrt(1-e^(-k\n        x^2)) with k = 2 / pi. However, it has been found that other values for\n        `k` may be more accurate. The value that minimizes the absolute error\n        over the range [-10, 10] is 0.62305051 (approximately 5/8).\n\n  Returns:\n    implied_vols: A `Tensor` of same shape and dtype as\n      `undiscounted_prices`. The approximate implied volatilities\n      computed using the Polya approximation for the normal CDF.\n  '
    if polya_factor is None:
        polya_factor = tf.convert_to_tensor(2.0 / np.pi, dtype=normalized_prices.dtype)
    floored_forwards = tf.math.maximum(normalized_forwards, 1)
    capped_forwards = tf.math.minimum(normalized_forwards, 1)
    log_normalized_forwards = tf.math.log(normalized_forwards)
    sign_log_forward = tf.math.sign(log_normalized_forwards)
    if is_call_options is not None:
        is_call_options = tf.convert_to_tensor(is_call_options, dtype=tf.bool, name='is_call_options')
        ones = tf.ones_like(is_call_options, dtype=normalized_forwards.dtype)
        option_signs = tf.where(is_call_options, ones, -ones)
    else:
        option_signs = 1
    signs = option_signs * sign_log_forward
    cdfs = 0.5 + 0.5 * signs * tf.math.sqrt(-tf.math.expm1(-2 * polya_factor * tf.math.abs(log_normalized_forwards)))
    threshold = signs * (floored_forwards * cdfs - capped_forwards / 2)
    (a, b, lnc) = _get_quadratic_coeffs(normalized_prices, normalized_forwards, log_normalized_forwards, option_signs, polya_factor)
    c = tf.math.exp(lnc)
    lntwo = tf.convert_to_tensor(np.log(2.0), dtype=normalized_forwards.dtype)
    lnbeta = lntwo + lnc - tf.math.log(b + tf.math.sqrt(b * b + 4 * a * c))
    gamma = -lnbeta / polya_factor
    term1 = tf.math.sqrt(gamma + log_normalized_forwards)
    term2 = tf.math.sqrt(gamma - log_normalized_forwards)
    sqrt_var = tf.where(normalized_prices <= threshold, sign_log_forward * (term1 - term2), term1 + term2)
    return sqrt_var / tf.math.sqrt(expiries)

def _get_quadratic_coeffs(normalized_prices, normalized_forwards, log_normalized_forwards, option_signs, polya_factor):
    if False:
        for i in range(10):
            print('nop')
    'Computes the coefficients of the quadratic in Stefanica-Radiocic method.\n\n  Computes the coefficients described in Table 3 in Ref [1].\n\n  Args:\n    normalized_prices: `Tensor` of real dtype and any shape. The prices of the\n      options to be inverted. Normalization means that the raw price is divided\n      by the strike discounted to the present.\n    normalized_forwards: `Tensor` or same dtype and shape as `normalized_prices`\n      The forwards divided by the strike of the options.\n    log_normalized_forwards: `Tensor` or same dtype and shape as\n      `normalized_prices`. Log of the normalized forwards.\n    option_signs: Real `Tensor` of same shape and dtype as `normalized_prices`.\n      Should be +1 for a Call option and -1 for a put option.\n    polya_factor: Scalar `Tensor` of same dtype as `normalized_prices`. This is\n      the factor to use for approximating the normal CDF in a Polya-like\n      expression. Polya approximation is (here `k` is the `polya_factor`) N(x) ~\n      0.5 + sign(x) sqrt(1-e^(-k x^2)) with k = 2 / pi. However, it has been\n      found that other values for `k` may be more accurate. The value that\n      minimizes the absolute error over the range [-10, 10] is 0.62305051\n      (approximately 5/8).\n\n  Returns:\n    (A, B, ln(C)): A 3-tuple of coefficients in terms of which the approximate\n      implied vol is calculated.\n  '
    q1 = normalized_forwards - 1
    q2 = normalized_forwards + 1
    r = 2 * normalized_prices - option_signs * q1
    f1 = tf.math.pow(normalized_forwards, -polya_factor)
    f2 = 1 / f1
    g1 = f1 * normalized_forwards
    g2 = 1 / g1
    a = tf.math.square(g1 - g2)
    h = tf.math.square(normalized_forwards)
    r2 = tf.math.square(r)
    b = 4 * (f1 + f2) - 2 * (g1 + g2) * (1 + h - r2) / normalized_forwards
    lnc1 = tf.math.log(4.0 * normalized_prices) + tf.math.log(normalized_prices - option_signs * q1)
    lnc2 = tf.math.log(q2 - r) + tf.math.log(q2 + r)
    lnc = lnc1 + lnc2 - 2.0 * log_normalized_forwards
    return (a, b, lnc)