"""Black Scholes prices of a batch of European options."""
import numpy as np
import tensorflow.compat.v2 as tf
from tf_quant_finance import types
__all__ = ['option_price', 'barrier_price', 'binary_price', 'asset_or_nothing_price', 'swaption_price']

def option_price(*, volatilities: types.RealTensor, strikes: types.RealTensor, expiries: types.RealTensor, spots: types.RealTensor=None, forwards: types.RealTensor=None, discount_rates: types.RealTensor=None, dividend_rates: types.RealTensor=None, discount_factors: types.RealTensor=None, is_call_options: types.BoolTensor=None, is_normal_volatility: bool=False, dtype: tf.DType=None, name: str=None) -> types.RealTensor:
    if False:
        print('Hello World!')
    'Computes the Black Scholes price for a batch of call or put options.\n\n  #### Example\n\n  ```python\n    # Price a batch of 5 vanilla call options.\n    volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4])\n    forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])\n    # Strikes will automatically be broadcasted to shape [5].\n    strikes = np.array([3.0])\n    # Expiries will be broadcast to shape [5], i.e. each option has strike=3\n    # and expiry = 1.\n    expiries = 1.0\n    computed_prices = tff.black_scholes.option_price(\n        volatilities=volatilities,\n        strikes=strikes,\n        expiries=expiries,\n        forwards=forwards)\n  # Expected print output of computed prices:\n  # [ 0.          2.          2.04806848  1.00020297  2.07303131]\n  ```\n\n  #### References:\n  [1] Hull, John C., Options, Futures and Other Derivatives. Pearson, 2018.\n  [2] Wikipedia contributors. Black-Scholes model. Available at:\n    https://en.wikipedia.org/w/index.php?title=Black%E2%80%93Scholes_model\n\n  Args:\n    volatilities: Real `Tensor` of any shape and dtype. The volatilities to\n      expiry of the options to price.\n    strikes: A real `Tensor` of the same dtype and compatible shape as\n      `volatilities`. The strikes of the options to be priced.\n    expiries: A real `Tensor` of same dtype and compatible shape as\n      `volatilities`. The expiry of each option. The units should be such that\n      `expiry * volatility**2` is dimensionless.\n    spots: A real `Tensor` of any shape that broadcasts to the shape of the\n      `volatilities`. The current spot price of the underlying. Either this\n      argument or the `forwards` (but not both) must be supplied.\n    forwards: A real `Tensor` of any shape that broadcasts to the shape of\n      `volatilities`. The forwards to maturity. Either this argument or the\n      `spots` must be supplied but both must not be supplied.\n    discount_rates: An optional real `Tensor` of same dtype as the\n      `volatilities` and of the shape that broadcasts with `volatilities`.\n      If not `None`, discount factors are calculated as e^(-rT),\n      where r are the discount rates, or risk free rates. At most one of\n      `discount_rates` and `discount_factors` can be supplied.\n      Default value: `None`, equivalent to r = 0 and discount factors = 1 when\n      `discount_factors` also not given.\n    dividend_rates: An optional real `Tensor` of same dtype as the\n      `volatilities` and of the shape that broadcasts with `volatilities`.\n      Default value: `None`, equivalent to q = 0.\n    discount_factors: An optional real `Tensor` of same dtype as the\n      `volatilities`. If not `None`, these are the discount factors to expiry\n      (i.e. e^(-rT)). Mutually exclusive with `discount_rates`. If neither is\n      given, no discounting is applied (i.e. the undiscounted option price is\n      returned). If `spots` is supplied and `discount_factors` is not `None`\n      then this is also used to compute the forwards to expiry. At most one of\n      `discount_rates` and `discount_factors` can be supplied.\n      Default value: `None`, which maps to e^(-rT) calculated from\n      discount_rates.\n    is_call_options: A boolean `Tensor` of a shape compatible with\n      `volatilities`. Indicates whether the option is a call (if True) or a put\n      (if False). If not supplied, call options are assumed.\n    is_normal_volatility: An optional Python boolean specifying whether the\n      `volatilities` correspond to lognormal Black volatility (if False) or\n      normal Black volatility (if True).\n      Default value: False, which corresponds to lognormal volatility.\n    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion\n      of any supplied non-`Tensor` arguments to `Tensor`.\n      Default value: `None` which maps to the default dtype inferred by\n        TensorFlow.\n    name: str. The name for the ops created by this function.\n      Default value: `None` which is mapped to the default name `option_price`.\n\n  Returns:\n    option_prices: A `Tensor` of the same shape as `forwards`. The Black\n    Scholes price of the options.\n\n  Raises:\n    ValueError: If both `forwards` and `spots` are supplied or if neither is\n      supplied.\n    ValueError: If both `discount_rates` and `discount_factors` is supplied.\n  '
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
        if forwards is not None:
            forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
        else:
            spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
            forwards = spots * tf.exp((discount_rates - dividend_rates) * expiries)
        sqrt_var = volatilities * tf.math.sqrt(expiries)
        if not is_normal_volatility:
            d1 = tf.math.divide_no_nan(tf.math.log(forwards / strikes), sqrt_var) + sqrt_var / 2
            d2 = d1 - sqrt_var
            undiscounted_calls = tf.where(sqrt_var > 0, forwards * _ncdf(d1) - strikes * _ncdf(d2), tf.math.maximum(forwards - strikes, 0.0))
        else:
            d1 = tf.math.divide_no_nan(forwards - strikes, sqrt_var)
            undiscounted_calls = tf.where(sqrt_var > 0.0, (forwards - strikes) * _ncdf(d1) + sqrt_var * tf.math.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi), tf.math.maximum(forwards - strikes, 0.0))
        if is_call_options is None:
            return discount_factors * undiscounted_calls
        undiscounted_forward = forwards - strikes
        undiscounted_puts = undiscounted_calls - undiscounted_forward
        predicate = tf.broadcast_to(is_call_options, tf.shape(undiscounted_calls))
        return discount_factors * tf.where(predicate, undiscounted_calls, undiscounted_puts)

def barrier_price(*, volatilities: types.RealTensor, strikes: types.RealTensor, expiries: types.RealTensor, spots: types.RealTensor, barriers: types.RealTensor, rebates: types.RealTensor=None, discount_rates: types.RealTensor=None, dividend_rates: types.RealTensor=None, is_barrier_down: types.BoolTensor=None, is_knock_out: types.BoolTensor=None, is_call_options: types.BoolTensor=None, dtype: tf.DType=None, name: str=None) -> types.RealTensor:
    if False:
        return 10
    'Prices barrier options in a Black-Scholes Model.\n\n  Computes the prices of options with a single barrier in Black-Scholes world as\n  described in Ref. [1]. Note that the barrier is applied continuously.\n\n  #### Example\n\n  This example is taken from Ref. [2], Page 154.\n\n  ```python\n  import tf_quant_finance as tff\n\n  dtype = np.float32\n  discount_rates = np.array([.08, .08])\n  dividend_rates = np.array([.04, .04])\n  spots = np.array([100., 100.])\n  strikes = np.array([90., 90.])\n  barriers = np.array([95. 95.])\n  rebates = np.array([3. 3.])\n  volatilities = np.array([.25, .25])\n  expiries = np.array([.5, .5])\n  barriers_type = np.array([5, 1])\n  is_barrier_down = np.array([True, False])\n  is_knock_out = np.array([False, False])\n  is_call_option = np.array([True, True])\n\n  price = tff.black_scholes.barrier_price(\n    discount_rates, dividend_rates, spots, strikes,\n    barriers, rebates, volatilities,\n    expiries, is_barrier_down, is_knock_out, is_call_options)\n\n  # Expected output\n  #  `Tensor` with values [9.024, 7.7627]\n  ```\n\n  #### References\n\n  [1]: Lee Clewlow, Javier Llanos, Chris Strickland, Caracas Venezuela\n    Pricing Exotic Options in a Black-Scholes World, 1994\n    https://warwick.ac.uk/fac/soc/wbs/subjects/finance/research/wpaperseries/1994/94-54.pdf\n  [2]: Espen Gaarder Haug, The Complete Guide to Option Pricing Formulas,\n    2nd Edition, 1997\n\n  Args:\n    volatilities: Real `Tensor` of any shape and dtype. The volatilities to\n      expiry of the options to price.\n    strikes: A real `Tensor` of the same dtype and compatible shape as\n      `volatilities`. The strikes of the options to be priced.\n    expiries: A real `Tensor` of same dtype and compatible shape as\n      `volatilities`. The expiry of each option. The units should be such that\n      `expiry * volatility**2` is dimensionless.\n    spots: A real `Tensor` of any shape that broadcasts to the shape of the\n      `volatilities`. The current spot price of the underlying.\n    barriers: A real `Tensor` of same dtype as the `volatilities` and of the\n      shape that broadcasts with `volatilities`. The barriers of each option.\n    rebates: A real `Tensor` of same dtype as the `volatilities` and of the\n      shape that broadcasts with `volatilities`. For knockouts, this is a\n      fixed cash payout in case the barrier is breached. For knockins, this is a\n      fixed cash payout in case the barrier level is not breached. In the former\n      case, the rebate is paid immediately on breach whereas in the latter, the\n      rebate is paid at the expiry of the option.\n      Default value: `None` which maps to no rebates.\n    discount_rates: A real `Tensor` of same dtype as the\n      `volatilities` and of the shape that broadcasts with `volatilities`.\n      Discount rates, or risk free rates.\n      Default value: `None`, equivalent to discount_rate = 0.\n    dividend_rates: A real `Tensor` of same dtype as the\n      `volatilities` and of the shape that broadcasts with `volatilities`. A\n      continuous dividend rate paid by the underlier. If `None`, then\n      defaults to zero dividends.\n      Default value: `None`, equivalent to zero dividends.\n    is_barrier_down: A real `Tensor` of `boolean` values and of the shape\n      that broadcasts with `volatilities`. True if barrier is below asset\n      price at expiration.\n      Default value: `True`.\n    is_knock_out: A real `Tensor` of `boolean` values and of the shape\n      that broadcasts with `volatilities`. True if option is knock out\n      else false.\n      Default value: `True`.\n    is_call_options: A real `Tensor` of `boolean` values and of the shape\n      that broadcasts with `volatilities`. True if option is call else\n      false.\n      Default value: `True`.\n    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion\n      of any supplied non-`Tensor` arguments to `Tensor`.\n      Default value: `None` which maps to the default dtype inferred by\n      TensorFlow.\n    name: str. The name for the ops created by this function.\n      Default value: `None` which is mapped to the default name `barrier_price`.\n  Returns:\n    option_prices: A `Tensor` of same shape as `spots`. The approximate price of\n    the barriers option under black scholes.\n  '
    with tf.name_scope(name or 'barrier_price'):
        spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
        dtype = spots.dtype
        strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
        volatilities = tf.convert_to_tensor(volatilities, dtype=dtype, name='volatilities')
        expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
        barriers = tf.convert_to_tensor(barriers, dtype=dtype, name='barriers')
        if rebates is not None:
            rebates = tf.convert_to_tensor(rebates, dtype=dtype, name='rebates')
        else:
            rebates = tf.zeros_like(spots, dtype=dtype, name='rebates')
        if discount_rates is not None:
            discount_rates = tf.convert_to_tensor(discount_rates, dtype=dtype, name='discount_rates')
        else:
            discount_rates = tf.convert_to_tensor(0.0, dtype=dtype, name='discount_rates')
        if dividend_rates is not None:
            dividend_rates = tf.convert_to_tensor(dividend_rates, dtype=dtype, name='dividend_rates')
        else:
            dividend_rates = tf.convert_to_tensor(0.0, dtype=dtype, name='dividend_rates')
        if is_barrier_down is None:
            is_barrier_down = tf.constant(1, name='is_barrier_down')
        else:
            is_barrier_down = tf.convert_to_tensor(is_barrier_down, dtype=tf.bool, name='is_barrier_down')
            is_barrier_down = tf.where(is_barrier_down, 1, 0)
        if is_knock_out is None:
            is_knock_out = tf.constant(1, name='is_knock_out')
        else:
            is_knock_out = tf.convert_to_tensor(is_knock_out, dtype=tf.bool, name='is_knock_out')
            is_knock_out = tf.where(is_knock_out, 1, 0)
        if is_call_options is None:
            is_call_options = tf.constant(1, name='is_call_options')
        else:
            is_call_options = tf.convert_to_tensor(is_call_options, dtype=tf.bool, name='is_call_options')
            is_call_options = tf.where(is_call_options, 1, 0)
        indices = tf.bitwise.left_shift(is_barrier_down, 2) + tf.bitwise.left_shift(is_knock_out, 1) + is_call_options
        mask_matrix_greater_strike = tf.constant([[1, 1, -1, -1, 0, 0, 1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 1, 1, -1, -1, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0], [1, 1, -1, -1, 1, 1, -1, -1, 0, 0, 1, 1], [1, 1, 0, 0, -1, -1, 0, 0, 0, 0, 1, 1]])
        mask_matrix_lower_strike = tf.constant([[0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0], [0, 0, 1, 1, -1, -1, 1, 1, 1, 1, 0, 0], [1, 1, 0, 0, -1, -1, 0, 0, 0, 0, 1, 1], [1, 1, -1, -1, 1, 1, -1, -1, 0, 0, 1, 1], [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], [1, 1, -1, -1, 0, 0, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1]])
        masks_lower = tf.gather(mask_matrix_lower_strike, indices, axis=0)
        masks_greater = tf.gather(mask_matrix_greater_strike, indices, axis=0)
        strikes_greater = tf.expand_dims(strikes > barriers, axis=-1)
        masks = tf.where(strikes_greater, masks_greater, masks_lower)
        masks = tf.cast(masks, dtype=dtype)
        one = tf.constant(1, dtype=dtype)
        call_or_put = tf.cast(tf.where(tf.equal(is_call_options, 0), -one, one), dtype=dtype)
        below_or_above = tf.cast(tf.where(tf.equal(is_barrier_down, 0), -one, one), dtype=dtype)
        sqrt_var = volatilities * tf.math.sqrt(expiries)
        mu = discount_rates - dividend_rates - volatilities ** 2 / 2
        lamda = 1 + mu / volatilities ** 2
        x = tf.math.log(spots / strikes) / sqrt_var + lamda * sqrt_var
        x1 = tf.math.log(spots / barriers) / sqrt_var + lamda * sqrt_var
        y = tf.math.log(barriers ** 2 / (spots * strikes)) / sqrt_var + lamda * sqrt_var
        y1 = tf.math.log(barriers / spots) / sqrt_var + lamda * sqrt_var
        b = (mu ** 2 + 2 * volatilities ** 2 * discount_rates) / volatilities ** 2
        z = tf.math.log(barriers / spots) / sqrt_var + b * sqrt_var
        a = mu / volatilities ** 2
        discount_factors = tf.math.exp(-discount_rates * expiries, name='discount_factors')
        barriers_ratio = tf.math.divide(barriers, spots, name='barriers_ratio')
        spots_term = call_or_put * spots * tf.math.exp(-dividend_rates * expiries)
        strikes_term = call_or_put * strikes * discount_factors
        strike_rank = strikes.shape.rank
        terms_mat = tf.stack((spots_term, -strikes_term, spots_term, -strikes_term, spots_term * barriers_ratio ** (2 * lamda), -strikes_term * barriers_ratio ** (2 * lamda - 2), spots_term * barriers_ratio ** (2 * lamda), -strikes_term * barriers_ratio ** (2 * lamda - 2), rebates * discount_factors, -rebates * discount_factors * barriers_ratio ** (2 * lamda - 2), rebates * barriers_ratio ** (a + b), rebates * barriers_ratio ** (a - b)), name='term_matrix', axis=strike_rank)
        cdf_mat = tf.stack((call_or_put * x, call_or_put * (x - sqrt_var), call_or_put * x1, call_or_put * (x1 - sqrt_var), below_or_above * y, below_or_above * (y - sqrt_var), below_or_above * y1, below_or_above * (y1 - sqrt_var), below_or_above * (x1 - sqrt_var), below_or_above * (y1 - sqrt_var), below_or_above * z, below_or_above * (z - 2 * b * sqrt_var)), name='cdf_matrix', axis=strike_rank)
        cdf_mat = _ncdf(cdf_mat)
        return tf.reduce_sum(masks * terms_mat * cdf_mat, axis=strike_rank)

def binary_price(*, volatilities: types.RealTensor, strikes: types.RealTensor, expiries: types.RealTensor, spots: types.RealTensor=None, forwards: types.RealTensor=None, discount_rates: types.RealTensor=None, dividend_rates: types.RealTensor=None, discount_factors: types.RealTensor=None, is_call_options: types.BoolTensor=None, is_normal_volatility: bool=False, dtype: tf.DType=None, name: str=None) -> types.RealTensor:
    if False:
        i = 10
        return i + 15
    'Computes the Black Scholes price for a batch of binary call or put options.\n\n  The binary call (resp. put) option priced here is that which pays off a unit\n  of cash if the underlying asset has a value greater (resp. smaller) than the\n  strike price at expiry. Hence the binary option price is the discounted\n  probability that the asset will end up higher (resp. lower) than the\n  strike price at expiry.\n\n  #### Example\n\n  ```python\n    # Price a batch of 5 binary call options.\n    volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4])\n    forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])\n    # Strikes will automatically be broadcasted to shape [5].\n    strikes = np.array([3.0])\n    # Expiries will be broadcast to shape [5], i.e. each option has strike=3\n    # and expiry = 1.\n    expiries = 1.0\n    computed_prices = tff.black_scholes.binary_price(\n        volatilities=volatilities,\n        strikes=strikes,\n        expiries=expiries,\n        forwards=forwards)\n  # Expected print output of prices:\n  # [0.         0.         0.15865525 0.99764937 0.85927418]\n  ```\n\n  #### References:\n\n  [1] Hull, John C., Options, Futures and Other Derivatives. Pearson, 2018.\n  [2] Wikipedia contributors. Binary option. Available at:\n  https://en.wikipedia.org/w/index.php?title=Binary_option\n\n  Args:\n    volatilities: Real `Tensor` of any shape and dtype. The volatilities to\n      expiry of the options to price.\n    strikes: A real `Tensor` of the same dtype and compatible shape as\n      `volatilities`. The strikes of the options to be priced.\n    expiries: A real `Tensor` of same dtype and compatible shape as\n      `volatilities`. The expiry of each option. The units should be such that\n      `expiry * volatility**2` is dimensionless.\n    spots: A real `Tensor` of any shape that broadcasts to the shape of the\n      `volatilities`. The current spot price of the underlying. Either this\n      argument or the `forwards` (but not both) must be supplied.\n    forwards: A real `Tensor` of any shape that broadcasts to the shape of\n      `volatilities`. The forwards to maturity. Either this argument or the\n      `spots` must be supplied but both must not be supplied.\n    discount_rates: An optional real `Tensor` of same dtype as the\n      `volatilities` and of the shape that broadcasts with `volatilities`.\n      If not `None`, discount factors are calculated as e^(-rT),\n      where r are the discount rates, or risk free rates. At most one of\n      discount_rates and discount_factors can be supplied.\n      Default value: `None`, equivalent to r = 0 and discount factors = 1 when\n      discount_factors also not given.\n    dividend_rates: An optional real `Tensor` of same dtype as the\n      `volatilities` and of the shape that broadcasts with `volatilities`.\n      Default value: `None`, equivalent to q = 0.\n    discount_factors: An optional real `Tensor` of same dtype as the\n      `volatilities`. If not None, these are the discount factors to expiry\n      (i.e. e^(-rT)). If None, no discounting is applied (i.e. the undiscounted\n      option price is returned). If `spots` is supplied and `discount_factors`\n      is not None then this is also used to compute the forwards to expiry.\n      Default value: None, equivalent to discount factors = 1.\n    is_call_options: A boolean `Tensor` of a shape compatible with\n      `volatilities`. Indicates whether the option is a call (if True) or a put\n      (if False). If not supplied, call options are assumed.\n    is_normal_volatility: An optional Python boolean specifying whether the\n      `volatilities` correspond to lognormal Black volatility (if False) or\n      normal Black volatility (if True).\n      Default value: False, which corresponds to lognormal volatility.\n    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion\n      of any supplied non-`Tensor` arguments to `Tensor`.\n      Default value: None which maps to the default dtype inferred by TensorFlow\n        (float32).\n    name: str. The name for the ops created by this function.\n      Default value: None which is mapped to the default name `binary_price`.\n\n  Returns:\n    binary_prices: A `Tensor` of the same shape as `forwards`. The Black\n    Scholes price of the binary options.\n\n  Raises:\n    ValueError: If both `forwards` and `spots` are supplied or if neither is\n      supplied.\n    ValueError: If both `discount_rates` and `discount_factors` is supplied.\n  '
    if (spots is None) == (forwards is None):
        raise ValueError('Either spots or forwards must be supplied but not both.')
    if discount_rates is not None and discount_factors is not None:
        raise ValueError('At most one of discount_rates and discount_factors may be supplied')
    with tf.name_scope(name or 'binary_price'):
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
        if forwards is not None:
            forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
        else:
            spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
            forwards = spots / discount_factors
        sqrt_var = volatilities * tf.math.sqrt(expiries)
        if is_normal_volatility:
            d2 = (forwards - strikes) / sqrt_var
        else:
            d2 = tf.math.log(forwards / strikes) / sqrt_var - sqrt_var / 2
        zero_volatility_call_payoff = tf.where(forwards > strikes, tf.ones_like(strikes, dtype=dtype), tf.zeros_like(strikes, dtype=dtype))
        undiscounted_calls = tf.where(sqrt_var > 0, _ncdf(d2), zero_volatility_call_payoff)
        if is_call_options is None:
            return discount_factors * undiscounted_calls
        undiscounted_puts = 1 - undiscounted_calls
        predicate = tf.broadcast_to(is_call_options, tf.shape(undiscounted_calls))
        return discount_factors * tf.where(predicate, undiscounted_calls, undiscounted_puts)

def asset_or_nothing_price(*, volatilities: types.RealTensor, strikes: types.RealTensor, expiries: types.RealTensor, spots: types.RealTensor=None, forwards: types.RealTensor=None, discount_rates: types.RealTensor=None, dividend_rates: types.RealTensor=None, discount_factors: types.RealTensor=None, is_call_options: types.BoolTensor=None, is_normal_volatility: bool=False, dtype: tf.DType=None, name: str=None) -> types.RealTensor:
    if False:
        return 10
    'Computes the Black Scholes price for a batch of asset-or-nothing options.\n\n  The asset-or-nothing call (resp. put) pays out one unit of the underlying\n  asset if the spot is above (resp. below) the strike at maturity.\n\n  #### Example\n\n  ```python\n    # Price a batch of 5 asset_or_nothing call and put options.\n    volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4])\n    forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])\n    # Strikes will automatically be broadcasted to shape [5].\n    strikes = np.array([3.0])\n    # Expiries will be broadcast to shape [5], i.e. each option has strike=3\n    # and expiry = 1.\n    expiries = 1.0\n    computed_prices = tff.black_scholes.asset_or_nothing_price(\n        volatilities=volatilities,\n        strikes=strikes,\n        expiries=expiries,\n        forwards=forwards)\n  # Expected print output of prices:\n  # [0., 2., 2.52403424, 3.99315108, 4.65085383]\n  ```\n\n  #### References:\n\n  [1] Hull, John C., Options, Futures and Other Derivatives. Pearson, 2018.\n  [2] https://en.wikipedia.org/wiki/Binary_option#Asset-or-nothing_call\n\n  Args:\n    volatilities: Real `Tensor` of any shape and dtype. The volatilities to\n      expiry of the options to price.\n    strikes: A real `Tensor` of the same dtype and compatible shape as\n      `volatilities`. The strikes of the options to be priced.\n    expiries: A real `Tensor` of same dtype and compatible shape as\n      `volatilities`. The expiry of each option.\n    spots: A real `Tensor` of any shape that broadcasts to the shape of the\n      `volatilities`. The current spot price of the underlying. Either this\n      argument or the `forwards` (but not both) must be supplied.\n    forwards: A real `Tensor` of any shape that broadcasts to the shape of\n      `volatilities`. The forwards to maturity. Either this argument or the\n      `spots` must be supplied but both must not be supplied.\n    discount_rates: An optional real `Tensor` of same dtype as the\n      `volatilities` and of the shape that broadcasts with `volatilities`. If\n      not `None`, discount factors are calculated as e^(-rT), where r are the\n      discount rates, or risk free rates. At most one of discount_rates and\n      discount_factors can be supplied.\n      Default value: `None`, equivalent to r = 0 and discount factors = 1 when\n        discount_factors also not given.\n    dividend_rates: An optional real `Tensor` of same dtype as the\n      `volatilities` and of the shape that broadcasts with `volatilities`.\n      Default value: `None`, equivalent to q = 0.\n    discount_factors: An optional real `Tensor` of same dtype as the\n      `volatilities`. If not `None`, these are the discount factors to expiry\n      (i.e. e^(-rT)). Mutually exclusive with discount_rates. If neither is\n      given, no discounting is applied (i.e. the undiscounted option price is\n      returned). If `spots` is supplied and `discount_factors` is not `None`\n      then this is also used to compute the forwards to expiry. At most one of\n      `discount_rates` and `discount_factors` can be supplied.\n      Default value: `None`, which maps to e^(-rT) calculated from\n        discount_rates.\n    is_call_options: A boolean `Tensor` of a shape compatible with\n      `volatilities`. Indicates whether the option is a call (if True) or a put\n      (if False). If not supplied, call options are assumed.\n    is_normal_volatility: An optional Python boolean specifying whether the\n      `volatilities` correspond to lognormal Black volatility (if False) or\n      normal Black volatility (if True).\n      Default value: False, which corresponds to lognormal volatility.\n    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion\n      of any supplied non-`Tensor` arguments to `Tensor`.\n      Default value: `None` which maps to the default dtype inferred by\n        TensorFlow.\n    name: str. The name for the ops created by this function.\n      Default value: `None`, which is mapped to the default name\n        `asset_or_nothing_price`.\n\n  Returns:\n    option_prices: A `Tensor` of the same shape as `forwards`. The Black\n    Scholes price of the options.\n\n  Raises:\n    ValueError: If both `forwards` and `spots` are supplied or if neither is\n      supplied.\n    ValueError: If both `discount_rates` and `discount_factors` is supplied.\n  '
    if (spots is None) == (forwards is None):
        raise ValueError('Either spots or forwards must be supplied but not both.')
    if discount_rates is not None and discount_factors is not None:
        raise ValueError('At most one of discount_rates and discount_factors may be supplied')
    with tf.name_scope(name or 'asset_or_nothing_price'):
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
        if forwards is not None:
            forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
        else:
            spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
            forwards = spots * tf.exp((discount_rates - dividend_rates) * expiries)
        sqrt_var = volatilities * tf.math.sqrt(expiries)
        if not is_normal_volatility:
            d1 = tf.math.divide_no_nan(tf.math.log(forwards / strikes), sqrt_var) + sqrt_var / 2
            undiscounted_calls = tf.where(sqrt_var > 0, forwards * _ncdf(d1), tf.where(forwards > strikes, forwards, 0.0))
        else:
            d1 = tf.math.divide_no_nan(forwards - strikes, sqrt_var)
            undiscounted_calls = tf.where(sqrt_var > 0.0, forwards * _ncdf(d1) + sqrt_var * tf.math.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi), tf.where(forwards > strikes, forwards, 0.0))
        if is_call_options is None:
            return discount_factors * undiscounted_calls
        undiscounted_puts = forwards - undiscounted_calls
        predicate = tf.broadcast_to(is_call_options, tf.shape(undiscounted_calls))
        return discount_factors * tf.where(predicate, undiscounted_calls, undiscounted_puts)

def swaption_price(*, volatilities: types.RealTensor, expiries: types.RealTensor, floating_leg_start_times: types.RealTensor, floating_leg_end_times: types.RealTensor, fixed_leg_payment_times: types.RealTensor, floating_leg_daycount_fractions: types.RealTensor, fixed_leg_daycount_fractions: types.RealTensor, fixed_leg_coupon: types.RealTensor, floating_leg_start_times_discount_factors: types.RealTensor, floating_leg_end_times_discount_factors: types.RealTensor, fixed_leg_payment_times_discount_factors: types.RealTensor, notional: types.RealTensor=None, is_payer_swaption: types.BoolTensor=None, is_normal_volatility: bool=True, dtype: tf.DType=None, name: str=None) -> types.RealTensor:
    if False:
        return 10
    'Calculates the price of European Swaptions using the Black model.\n\n  A European Swaption is a contract that gives the holder an option to enter a\n  swap contract at a future date at a prespecified fixed rate. A swaption that\n  grants the holder to pay fixed rate and receive floating rate is called a\n  payer swaption while the swaption that grants the holder to receive fixed and\n  pay floating payments is called the receiver swaption. Typically the start\n  date (or the inception date) of the swap coincides with the expiry of the\n  swaption.\n\n  #### Example\n  The example shows how value a batch of 1y x 1y and 1y x 2y swaptions using the\n  Black (normal) model for the swap rate.\n\n  ````python\n  import numpy as np\n  import tensorflow.compat.v2 as tf\n  import tf_quant_finance as tff\n\n  dtype = tf.float64\n\n  volatilities = [0.01, 0.005]\n  expiries = [1.0, 1.0]\n  float_leg_start_times = [[1.0, 1.25, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0],\n                            [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]]\n  float_leg_end_times = [[1.25, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0, 2.0],\n                          [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]]\n  fixed_leg_payment_times = [[1.25, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0, 2.0],\n                              [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]]\n  float_leg_daycount_fractions = [[0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0],\n                                   [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,\n                                   0.25]]\n  fixed_leg_daycount_fractions = [[0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0],\n                                   [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,\n                                   0.25]]\n  fixed_leg_coupon = [0.011, 0.011]\n  discount_fn = lambda x: np.exp(-0.01 * np.array(x))\n  price = self.evaluate(\n  tff.black_scholes.swaption_price(\n      volatilities=volatilities,\n      expiries=expiries,\n      floating_leg_start_times=float_leg_start_times,\n      floating_leg_end_times=float_leg_end_times,\n      fixed_leg_payment_times=fixed_leg_payment_times,\n      floating_leg_daycount_fractions=float_leg_daycount_fractions,\n      fixed_leg_daycount_fractions=fixed_leg_daycount_fractions,\n      fixed_leg_coupon=fixed_leg_coupon,\n      floating_leg_start_times_discount_factors=discount_fn(\n          float_leg_start_times),\n      floating_leg_end_times_discount_factors=discount_fn(\n          float_leg_end_times),\n      fixed_leg_payment_times_discount_factors=discount_fn(\n          fixed_leg_payment_times),\n      is_normal_volatility=is_normal_model,\n      notional=100.,\n      dtype=dtype))\n  # Expected value: [0.3458467885511461, 0.3014786656395892] # shape = (2,)\n  ````\n\n  Args:\n    volatilities: Real `Tensor` of any shape and dtype. The Black volatilities\n      of the swaptions to price. The shape of this input determines the number\n      (and shape) of swaptions to be priced and the shape of the output.\n    expiries: A real `Tensor` of same shape and dtype as `volatilities`. The\n      time to expiration of the swaptions.\n    floating_leg_start_times: A real `Tensor` of the same dtype as\n      `volatilities`. The times when accrual begins for each payment in the\n      floating leg. The shape of this input should be `expiries.shape + [m]` or\n      `batch_shape + [m]` where `m` denotes the number of floating payments in\n      each leg.\n    floating_leg_end_times: A real `Tensor` of the same dtype as `volatilities`.\n      The times when accrual ends for each payment in the floating leg. The\n      shape of this input should be `batch_shape + [m]` where `m` denotes\n      the number of floating payments in each leg.\n    fixed_leg_payment_times: A real `Tensor` of the same dtype as\n      `volatilities`.  The payment times for each payment in the fixed leg.\n      The shape of this input should be `batch_shape + [n]` where `n` denotes\n      the number of fixed payments in each leg.\n    floating_leg_daycount_fractions: A real `Tensor` of the same dtype and\n      compatible shape as `floating_leg_start_times`. The daycount fractions\n      for each payment in the floating leg.\n    fixed_leg_daycount_fractions: A real `Tensor` of the same dtype and\n      compatible shape as `fixed_leg_payment_times`. The daycount fractions\n      for each payment in the fixed leg.\n    fixed_leg_coupon: A real `Tensor` of the same dtype and shape compatible\n      to `batch_shape`. The fixed coupon rate for each payment in the fixed leg.\n    floating_leg_start_times_discount_factors: A real `Tensor` of the same\n      shape and dtype as `floating_leg_start_times`. The discount factors\n      corresponding to `floating_leg_start_times`.\n    floating_leg_end_times_discount_factors: A real `Tensor` of the same\n      shape and dtype as `floating_leg_end_times`. The discount factors\n      corresponding to `floating_leg_end_times`.\n    fixed_leg_payment_times_discount_factors: A real `Tensor` of the same\n      shape and dtype as `fixed_leg_payment_times`. The discount factors\n      corresponding to `fixed_leg_payment_times`.\n    notional: An optional `Tensor` of same dtype and compatible shape as\n      `volatilities` specifying the notional amount for the underlying swap.\n       Default value: None in which case the notional is set to 1.\n    is_payer_swaption: A boolean `Tensor` of a shape compatible with `expiries`.\n      Indicates whether the swaption is a payer (if True) or a receiver\n      (if False) swaption. If not supplied, payer swaptions are assumed.\n    is_normal_volatility: An optional Python boolean specifying whether the\n      `volatilities` correspond to normal Black volatility (if True) or\n      lognormal Black volatility (if False).\n      Default value: True, which corresponds to normal volatility.\n    dtype: The default dtype to use when converting values to `Tensor`s.\n      Default value: `None` which means that default dtypes inferred by\n      TensorFlow are used.\n    name: Python string. The name to give to the ops created by this function.\n      Default value: `None` which maps to the default name\n      `hw_swaption_price`.\n\n  Returns:\n    A `Tensor` of real dtype and shape `batch_shape` containing the\n    computed swaption prices.\n  '
    name = name or 'black_swaption_price'
    del floating_leg_daycount_fractions
    with tf.name_scope(name):
        volatilities = tf.convert_to_tensor(volatilities, dtype=dtype, name='volatilities')
        dtype = dtype or volatilities.dtype
        expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
        floating_leg_start_times = tf.convert_to_tensor(floating_leg_start_times, dtype=dtype, name='float_leg_start_times')
        floating_leg_end_times = tf.convert_to_tensor(floating_leg_end_times, dtype=dtype, name='float_leg_end_times')
        fixed_leg_payment_times = tf.convert_to_tensor(fixed_leg_payment_times, dtype=dtype, name='fixed_leg_payment_times')
        fixed_leg_daycount_fractions = tf.convert_to_tensor(fixed_leg_daycount_fractions, dtype=dtype, name='fixed_leg_daycount_fractions')
        fixed_leg_coupon = tf.convert_to_tensor(fixed_leg_coupon, dtype=dtype, name='fixed_leg_coupon')
        float_leg_start_times_discount_factors = tf.convert_to_tensor(floating_leg_start_times_discount_factors, dtype=dtype, name='float_leg_start_times_discount_factors')
        float_leg_end_times_discount_factors = tf.convert_to_tensor(floating_leg_end_times_discount_factors, dtype=dtype, name='float_leg_end_times_discount_factors')
        fixed_leg_payment_times_discount_factors = tf.convert_to_tensor(fixed_leg_payment_times_discount_factors, dtype=dtype, name='fixed_leg_payment_times_discount_factors')
        notional = tf.convert_to_tensor(notional, dtype=dtype, name='notional')
        if is_payer_swaption is None:
            is_payer_swaption = True
        is_payer_swaption = tf.convert_to_tensor(is_payer_swaption, dtype=tf.bool, name='is_payer_swaption')
        swap_annuity = tf.math.reduce_sum(fixed_leg_daycount_fractions * fixed_leg_payment_times_discount_factors, axis=-1)
        forward_swap_rate = tf.math.reduce_sum(float_leg_start_times_discount_factors - float_leg_end_times_discount_factors, axis=-1) / swap_annuity
        swaption_value = option_price(volatilities=volatilities, strikes=fixed_leg_coupon, expiries=expiries, forwards=forward_swap_rate, is_call_options=is_payer_swaption, is_normal_volatility=is_normal_volatility, dtype=dtype, name=name + '_option_price')
        return notional * swap_annuity * swaption_value

def _ncdf(x):
    if False:
        for i in range(10):
            print('nop')
    return (tf.math.erf(x / _SQRT_2) + 1) / 2
_SQRT_2 = np.sqrt(2.0, dtype=np.float64)