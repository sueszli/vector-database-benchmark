"""Black Scholes prices of a batch of Asian options."""
import enum
from typing import Optional
import numpy as np
import tensorflow.compat.v2 as tf
from tf_quant_finance import types
from tf_quant_finance.black_scholes import vanilla_prices
__all__ = ['asian_option_price']

@enum.unique
class AveragingType(enum.Enum):
    """Averaging types for asian options.

  * `GEOMETRIC`: C = ( \\prod S(t_i) ) ^ {\\frac{1}{n}}
  * `ARITHMETIC`: C = \\frac{1}{n} \\sum S(t_i)
  """
    GEOMETRIC = 1
    ARITHMETIC = 2

@enum.unique
class AveragingFrequency(enum.Enum):
    """Averaging types for asian options.

  * `DISCRETE`: Option samples on discrete times
  * `CONTINUOUS`: Option samples continuously throughout lifetime
  """
    DISCRETE = 1
    CONTINUOUS = 2

def asian_option_price(*, volatilities: types.RealTensor, strikes: types.RealTensor, expiries: types.RealTensor, spots: Optional[types.RealTensor]=None, forwards: Optional[types.RealTensor]=None, sampling_times: Optional[types.RealTensor]=None, past_fixings: Optional[types.RealTensor]=None, discount_rates: Optional[types.RealTensor]=None, dividend_rates: Optional[types.RealTensor]=None, discount_factors: Optional[types.RealTensor]=None, is_call_options: Optional[types.BoolTensor]=None, is_normal_volatility: bool=False, averaging_type: AveragingType=AveragingType.GEOMETRIC, averaging_frequency: AveragingFrequency=AveragingFrequency.DISCRETE, dtype: tf.DType=None, name: str=None) -> types.RealTensor:
    if False:
        return 10
    'Computes the Black Scholes price for a batch of asian options.\n\n  In Black-Scholes, the marginal distribution of the underlying at each sampling\n  date is lognormal. The product of a sequence of lognormal variables is also\n  lognormal so we can re-express these options as vanilla options with modified\n  parameters and use the vanilla pricer to price them.\n\n  TODO(b/261568763): support volatility term structures\n\n\n  #### Example\n\n  ```python\n    # Price a batch of 5 seasoned discrete geometric Asian options.\n    volatilities = np.array([0.0001, 102.0, 2.0, 0.1, 0.4])\n    forwards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])\n    # Strikes will automatically be broadcasted to shape [5].\n    strikes = np.array([3.0])\n    # Expiries will be broadcast to shape [5], i.e. each option has strike=3\n    # and expiry = 1.\n    expiries = 1.0\n    sampling_times = np.array([[0.5, 0.5, 0.5, 0.5, 0.5],\n                               [1.0, 1.0, 1.0, 1.0, 1.0]])\n    past_fixings = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])\n    computed_prices = tff.black_scholes.asian_option_price(\n        volatilities=volatilities,\n        strikes=strikes,\n        expiries=expiries,\n        forwards=forwards,\n        sampling_times=sampling_times,\n        past_fixings=past_fixings)\n  # Expected print output of computed prices:\n  # [ 0.0, 0.0, 0.52833763, 0.99555802, 1.91452834]\n  ```\n\n  #### References:\n  [1] Haug, E. G., The Complete Guide to Option Pricing Formulas. McGraw-Hill.\n\n  Args:\n    volatilities: Real `Tensor` of any shape compatible with a `batch_shape` and\n      and anyy real dtype. The volatilities to expiry of the options to price.\n      Here `batch_shape` corresponds to a batch of priced options.\n    strikes: A real `Tensor` of the same dtype and compatible shape as\n      `volatilities`. The strikes of the options to be priced.\n    expiries: A real `Tensor` of same dtype and compatible shape as\n      `volatilities`. The expiry of each option. The units should be such that\n      `expiry * volatility**2` is dimensionless.\n    spots: A real `Tensor` of any shape that broadcasts to the shape of the\n      `volatilities`. The current spot price of the underlying. Either this\n      argument or the `forwards` (but not both) must be supplied.\n    forwards: A real `Tensor` of any shape that broadcasts to the shape of\n      `volatilities`. The forwards to maturity. Either this argument or the\n      `spots` must be supplied but both must not be supplied.\n    sampling_times: A real `Tensor` of same dtype as expiries and shape `[n] +\n      batch_shape` where n is the number of sampling times for the Asian options\n      Default value: `None`, which will raise an error for discrete sampling\n      Asian options\n    past_fixings: A real `Tensor` of same dtype as spots or forwards and shape\n      `[n] + batch_shape` where n is the number of past fixings that have\n      already been observed.\n      Default value: `None`, equivalent to no past fixings (ie. unseasoned)\n    discount_rates: An optional real `Tensor` of same dtype as the\n      `volatilities` and of the shape that broadcasts with `volatilities`. If\n      not `None`, discount factors are calculated as e^(-rT), where r are the\n      discount rates, or risk free rates. At most one of `discount_rates` and\n      `discount_factors` can be supplied.\n      Default value: `None`, equivalent to `r = 0` and `discount factors = 1`\n      when `discount_factors` also not given.\n    dividend_rates: An optional real `Tensor` of same dtype as the\n      `volatilities` and of the shape that broadcasts with `volatilities`.\n      Default value: `None`, equivalent to q = 0.\n    discount_factors: An optional real `Tensor` of same dtype as the\n      `volatilities`. If not `None`, these are the discount factors to expiry\n      (i.e. e^(-rT)). Mutually exclusive with `discount_rates`. If neither is\n      given, no discounting is applied (i.e. the undiscounted option price is\n      returned). If `spots` is supplied and `discount_factors` is not `None`\n      then this is also used to compute the forwards to expiry. At most one of\n      `discount_rates` and `discount_factors` can be supplied.\n      Default value: `None`, which maps to e^(-rT) calculated from\n      `discount_rates`.\n    is_call_options: A boolean `Tensor` of a shape compatible with\n      `volatilities`. Indicates whether the option is a call (if True) or a put\n      (if False). If not supplied, call options are assumed.\n    is_normal_volatility: An optional Python boolean specifying whether the\n      `volatilities` correspond to lognormal Black volatility (if False) or\n      normal Black volatility (if True).\n      Default value: False, which corresponds to lognormal volatility.\n    averaging_type: Enum value of AveragingType to select the averaging method\n      for the payoff calculation.\n      Default value: AveragingType.GEOMETRIC\n    averaging_frequency: Enum value of AveragingFrequency to select the\n      averaging type for the payoff calculation (discrete vs continuous)\n      Default value: AveragingFrequency.DISCRETE\n    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion\n      of any supplied non-`Tensor` arguments to `Tensor`.\n      Default value: `None` which maps to the default dtype inferred by\n      TensorFlow.\n    name: str. The name for the ops created by this function.\n      Default value: `None` which is mapped to the default name\n      `asian_option_price`.\n\n  Returns:\n    option_prices: A `Tensor` of shape `batch_shape` and the same dtype as\n    `volatilities`. The Black Scholes price of the Asian options.\n\n  Raises:\n    ValueError: If both `forwards` and `spots` are supplied or if neither is\n      supplied.\n    ValueError: If both `discount_rates` and `discount_factors` is supplied.\n    ValueError: If `is_normal_volatility` is true and option is geometric, or\n      `is_normal_volatility` is false (ie. lognormal) and option is arithmetic.\n    ValueError: If option is discrete averaging and `sampling_dates` is None of\n      if last sampling date is later than option expiry date.\n    NotImplementedError: if option is continuous averaging.\n    NotImplementedError: if option is arithmetic.\n  '
    if (spots is None) == (forwards is None):
        raise ValueError('Either spots or forwards must be supplied but not both.')
    if discount_rates is not None and discount_factors is not None:
        raise ValueError('At most one of discount_rates and discount_factors may be supplied')
    if is_normal_volatility and averaging_type == AveragingType.GEOMETRIC:
        raise ValueError('Cannot price geometric averaging asians analytically under normal volatility')
    if not is_normal_volatility and averaging_type == AveragingType.ARITHMETIC:
        raise ValueError('Cannot price arithmetic averaging asians analytically under lognormal volatility')
    if averaging_frequency == AveragingFrequency.DISCRETE:
        if sampling_times is None:
            raise ValueError('Sampling times required for discrete sampling asians')
        if not np.all(np.maximum(sampling_times[-1], expiries) == expiries):
            raise ValueError('Sampling times cannot occur after expiry times')
    if averaging_frequency == AveragingFrequency.CONTINUOUS:
        raise NotImplementedError('Pricing continuous averaging asians not yet supported')
    if averaging_type == AveragingType.ARITHMETIC:
        raise NotImplementedError('Pricing arithmetic Asians not yet supported')
    with tf.name_scope(name or 'asian_option_price'):
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
            spots = forwards * tf.exp(-(discount_rates - dividend_rates) * expiries)
        else:
            spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
            forwards = spots * tf.exp((discount_rates - dividend_rates) * expiries)
        if past_fixings is None:
            running_accumulator = tf.convert_to_tensor(1.0, dtype=dtype)
            fixing_count = 0
        else:
            running_accumulator = tf.reduce_prod(past_fixings, 0)
            fixing_count = past_fixings.shape[0]
        sample_count = sampling_times.shape[0] + fixing_count
        sampling_time_forwards = spots * tf.exp((discount_rates - dividend_rates) * sampling_times)
        t1 = tf.reduce_sum(sampling_times, 0) / sample_count
        t2 = tf.reduce_sum(tf.vectorized_map(lambda x: tf.minimum(*tf.meshgrid(x, tf.transpose(x))), tf.transpose(sampling_times), fallback_to_while_loop=False), [1, 2]) / sample_count ** 2
        asian_forwards = tf.math.pow(running_accumulator * tf.reduce_prod(sampling_time_forwards, axis=0), 1 / sample_count) * tf.math.exp(-0.5 * volatilities * volatilities * (t1 - t2))
        effective_volatilities = volatilities * tf.math.sqrt(t2 / expiries)
        effective_dividend_rates = discount_rates - tf.math.log(asian_forwards / spots) / expiries
        return vanilla_prices.option_price(volatilities=effective_volatilities, strikes=strikes, expiries=expiries, forwards=asian_forwards, dividend_rates=effective_dividend_rates, discount_factors=discount_factors, is_call_options=is_call_options, dtype=dtype)