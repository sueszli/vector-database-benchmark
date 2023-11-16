"""Variance swap pricing using replicating portfolio approach."""
import tensorflow.compat.v2 as tf
from tf_quant_finance.black_scholes import vanilla_prices
from tf_quant_finance.math import diff_ops

def replicating_weights(ordered_strikes, reference_strikes, expiries, validate_args=False, dtype=None, name=None):
    if False:
        print('Hello World!')
    "Calculates the weights for options to recreate the variance swap payoff.\n\n  This implements the approach in Appendix A of Demeterfi et al for calculating\n  the weight of European options required to replicate the payoff of a variance\n  swap given traded strikes. In particular this function calculates the weights\n  for the put option part of the portfolio (when `ordered_strikes` is descending\n  ) or for the call option part of the portfolio (when `ordered_strikes`\n  is ascending). See the fair strike docstring for further details on variance\n  swaps.\n\n  #### Example\n\n  ```python\n  dtype = tf.float64\n  ordered_put_strikes = [100, 95, 90, 85]\n  reference_strikes = ordered_put_strikes[0]\n  expiries = 0.25\n  # Contains weights for put options at ordered_put_strikes[:-1]\n  put_weights = variance_replicating_weights(\n    ordered_put_strikes, reference_strikes, expiries, dtype=dtype)\n  # [0.00206927, 0.00443828, 0.00494591]\n  ```\n\n  #### References\n\n  [1] Demeterfi, K., Derman, E., Kamal, M. and Zou, J., 1999. More Than You Ever\n    Wanted To Know About Volatility Swaps. Goldman Sachs Quantitative Strategies\n    Research Notes.\n\n  Args:\n    ordered_strikes: A real `Tensor` of liquidly traded strikes of shape\n      `batch_shape + [num_strikes]`. The last entry will not receive a weight in\n      the portfolio. The values must be sorted ascending if the strikes are for\n      calls, or descending if the strikes are for puts. The final value in\n      `ordered_strikes` will not itself receive a weight.\n    reference_strikes: A `Tensor` of the same dtype as `ordered_strikes` and of\n      shape compatible with `batch_shape`. An arbitrarily chosen strike\n      representing an at the money strike price.\n    expiries: A `Tensor` of the same dtype as `ordered_strikes` and of shape\n      compatible with `batch_shape`. Represents the time to maturity of the\n      options.\n    validate_args: Python `bool`. When `True`, input `Tensor`s are checked for\n      validity. The checks verify that `ordered_strikes` is indeed ordered. When\n      `False` invalid inputs may silently render incorrect outputs, yet runtime\n      performance may be improved.\n      Default value: False.\n    dtype: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.\n      Default value: None leading to use of `ordered_strikes.dtype`.\n    name: Python str. The name to give to the ops created by this function.\n      Default value: `None` which maps to 'variance_replicating_weights'.\n\n  Returns:\n    A `Tensor` of shape `batch_shape + [num_strikes - 1]` representing the\n    weight which should be given to each strike in the replicating portfolio,\n    save for the final strike which is not represented.\n  "
    with tf.name_scope(name or 'replicating_weights'):
        ordered_strikes = tf.convert_to_tensor(ordered_strikes, dtype=dtype, name='ordered_strikes')
        dtype = dtype or ordered_strikes.dtype
        reference_strikes = tf.expand_dims(tf.convert_to_tensor(reference_strikes, dtype=dtype, name='reference_strikes'), -1)
        expiries = tf.expand_dims(tf.convert_to_tensor(expiries, dtype=dtype, name='expiries'), -1)
        strike_diff = diff_ops.diff(ordered_strikes, order=1, exclusive=True)
        strikes_descending = tf.math.reduce_all(strike_diff < 0)
        control_dependencies = []
        if validate_args:
            strikes_ascending = tf.math.reduce_all(strike_diff > 0)
            control_dependencies.append(tf.compat.v1.debugging.Assert(tf.math.logical_or(strikes_descending, strikes_ascending), [strike_diff]))
        with tf.control_dependencies(control_dependencies):
            term_lin = (ordered_strikes - reference_strikes) / reference_strikes
            term_log = tf.math.log(ordered_strikes) - tf.math.log(reference_strikes)
            payoff = 2.0 / expiries * (term_lin - term_log)
            payoff_diff = diff_ops.diff(payoff, order=1, exclusive=True)
            r_vals = tf.math.divide_no_nan(payoff_diff, strike_diff)
            zero = tf.zeros(r_vals.shape[:-1] + [1], dtype=r_vals.dtype)
            r_vals_diff = diff_ops.diff(tf.concat([zero, r_vals], axis=-1), order=1, exclusive=True)
            return tf.where(strikes_descending, -r_vals_diff, r_vals_diff)

def fair_strike(put_strikes, put_volatilities, call_strikes, call_volatilities, expiries, discount_rates, spots, reference_strikes, validate_args=False, dtype=None, name=None):
    if False:
        while True:
            i = 10
    "Calculates the fair value strike for a variance swap contract.\n\n  This implements the approach in Appendix A of Demeterfi et al (1999), where a\n  variance swap is defined as a forward contract on the square of annualized\n  realized volatility (though the approach assumes continuous sampling). The\n  variance swap payoff is, then:\n\n  `notional * (realized_volatility^2 - variance_strike)`\n\n  The method calculates the weight of each European option required to\n  approximately replicate such a payoff using the discrete range of strike\n  prices and implied volatilities of European options traded on the market. The\n  fair value `variance_strike` is that which is expected to produce zero payoff.\n\n  #### Example\n\n  ```python\n  dtype = tf.float64\n  call_strikes = tf.constant([[100, 105, 110, 115], [1000, 1100, 1200, 1300]],\n    dtype=dtype)\n  call_vols = 0.2 * tf.ones((2, 4), dtype=dtype)\n  put_strikes = tf.constant([[100, 95, 90, 85], [1000, 900, 800, 700]],\n    dtype=dtype)\n  put_vols = 0.2 * tf.ones((2, 4), dtype=dtype)\n  reference_strikes = tf.constant([100.0, 1000.0], dtype=dtype)\n  expiries = tf.constant([0.25, 0.25], dtype=dtype)\n  discount_rates = tf.constant([0.05, 0.05], dtype=dtype)\n  variance_swap_price(\n    put_strikes,\n    put_vols,\n    call_strikes,\n    put_vols,\n    expiries,\n    discount_rates,\n    reference_strikes,\n    reference_strikes,\n    dtype=tf.float64)\n  # [0.03825004, 0.04659269]\n  ```\n\n  #### References\n\n  [1] Demeterfi, K., Derman, E., Kamal, M. and Zou, J., 1999. More Than You Ever\n    Wanted To Know About Volatility Swaps. Goldman Sachs Quantitative Strategies\n    Research Notes.\n\n  Args:\n    put_strikes: A real `Tensor` of shape  `batch_shape + [num_put_strikes]`\n      containing the strike values of traded puts. This must be supplied in\n      **descending** order, and its elements should be less than or equal to the\n      `reference_strike`.\n    put_volatilities: A real `Tensor` of shape  `batch_shape +\n      [num_put_strikes]` containing the market volatility for each strike in\n      `put_strikes. The final value is unused.\n    call_strikes: A real `Tensor` of shape  `batch_shape + [num_call_strikes]`\n      containing the strike values of traded calls. This must be supplied in\n      **ascending** order, and its elements should be greater than or equal to\n      the `reference_strike`.\n    call_volatilities: A real `Tensor` of shape  `batch_shape +\n      [num_call_strikes]` containing the market volatility for each strike in\n      `call_strikes`. The final value is unused.\n    expiries: A real `Tensor` of shape compatible with `batch_shape` containing\n      the time to expiries of the contracts.\n    discount_rates: A real `Tensor` of shape compatible with `batch_shape`\n      containing the discount rate to be applied.\n    spots: A real `Tensor` of shape compatible with `batch_shape` containing the\n      current spot price of the asset.\n    reference_strikes: A real `Tensor` of shape compatible with `batch_shape`\n      containing an arbitrary value demarcating the atm boundary between liquid\n      calls and puts. Typically either the spot price or the (common) first\n      value of `put_strikes` or `call_strikes`.\n    validate_args: Python `bool`. When `True`, input `Tensor`s are checked for\n      validity. The checks verify the the matching length of strikes and\n      volatilties. When `False` invalid inputs may silently render incorrect\n      outputs, yet runtime performance will be improved.\n      Default value: False.\n    dtype: `tf.Dtype`. If supplied the dtype for the input and output `Tensor`s.\n      Default value: None, leading to the default value inferred by Tensorflow.\n    name: Python str. The name to give to the ops created by this function.\n      Default value: `None` which maps to 'variance_swap_price'.\n\n  Returns:\n    A `Tensor` of shape `batch_shape` containing the fair value of variance for\n    each item in the batch. Note this is on the decimal rather than square\n    percentage scale.\n  "
    with tf.name_scope(name or 'variance_swap_price'):
        put_strikes = tf.convert_to_tensor(put_strikes, dtype=dtype, name='put_strikes')
        dtype = dtype or put_strikes.dtype
        put_volatilities = tf.convert_to_tensor(put_volatilities, dtype=dtype, name='put_volatilities')
        call_strikes = tf.convert_to_tensor(call_strikes, dtype=dtype, name='call_strikes')
        call_volatilities = tf.convert_to_tensor(call_volatilities, dtype=dtype, name='call_volatilities')
        expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
        discount_rates = tf.expand_dims(tf.convert_to_tensor(discount_rates, dtype=dtype, name='discount_rates'), -1)
        spots = tf.expand_dims(tf.convert_to_tensor(spots, dtype=dtype, name='spots'), -1)
        reference_strikes = tf.convert_to_tensor(reference_strikes, dtype=dtype, name='reference_strikes')
        control_dependencies = []
        if validate_args:
            control_dependencies.append(tf.math.reduce_all(tf.shape(put_strikes)[-1] == tf.shape(put_volatilities)[-1]))
            control_dependencies.append(tf.math.reduce_all(tf.shape(call_strikes)[-1] == tf.shape(call_volatilities)[-1]))
        with tf.control_dependencies(control_dependencies):
            put_weights = replicating_weights(put_strikes, reference_strikes, expiries, validate_args=validate_args)
            call_weights = replicating_weights(call_strikes, reference_strikes, expiries, validate_args=validate_args)
            expiries = tf.expand_dims(expiries, -1)
            reference_strikes = tf.expand_dims(reference_strikes, -1)
            put_prices = vanilla_prices.option_price(volatilities=put_volatilities[..., :-1], strikes=put_strikes[..., :-1], expiries=expiries, spots=spots, discount_rates=discount_rates, is_call_options=False)
            call_prices = vanilla_prices.option_price(volatilities=call_volatilities[..., :-1], strikes=call_strikes[..., :-1], expiries=expiries, spots=spots, discount_rates=discount_rates, is_call_options=True)
            effective_rate = expiries * discount_rates
            discount_factor = tf.math.exp(effective_rate)
            s_ratio = spots / reference_strikes
            centrality_term = 2.0 / expiries * (effective_rate - discount_factor * s_ratio + 1 + tf.math.log(s_ratio))
            options_value = discount_factor * (tf.math.reduce_sum(put_weights * put_prices, axis=-1, keepdims=True) + tf.math.reduce_sum(call_weights * call_prices, axis=-1, keepdims=True))
            return tf.squeeze(options_value + centrality_term, axis=-1)