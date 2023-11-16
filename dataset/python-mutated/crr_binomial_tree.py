"""Black Scholes prices of options using CRR binomial trees."""
import tensorflow.compat.v2 as tf

def option_price_binomial(*, volatilities, strikes, expiries, spots, discount_rates=None, dividend_rates=None, is_call_options=None, is_american=None, num_steps=100, dtype=None, name=None):
    if False:
        while True:
            i = 10
    'Computes the BS price for a batch of European or American options.\n\n  Uses the Cox-Ross-Rubinstein version of the binomial tree method to compute\n  the price of American or European options. Supports batching of the options\n  and allows mixing of European and American style exercises in a batch.\n  For more information about the binomial tree method and the\n  Cox-Ross-Rubinstein method in particular see the references below.\n\n  #### Example\n\n  ```python\n  # Prices 5 options with a mix of Call/Put, American/European features\n  # in a single batch.\n  dtype = np.float64\n  spots = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)\n  strikes = np.array([3.0, 3.0, 3.0, 3.0, 3.0], dtype=dtype)\n  volatilities = np.array([0.1, 0.22, 0.32, 0.01, 0.4], dtype=dtype)\n  is_call_options = np.array([True, True, False, False, False])\n  is_american = np.array([False, True, True, False, True])\n  discount_rates = np.array(0.035, dtype=dtype)\n  dividend_rates = np.array([0.02, 0.0, 0.07, 0.01, 0.0], dtype=dtype)\n  expiries = np.array(1.0, dtype=dtype)\n\n  prices = option_price_binomial(\n      volatilities=volatilities,\n      strikes=strikes,\n      expiries=expiries,\n      spots=spots,\n      discount_rates=discount_rates,\n      dividend_rates=dividend_rates,\n      is_call_options=is_call_options,\n      is_american=is_american,\n      dtype=dtype)\n  # Prints [0., 0.0098847, 0.41299509, 0., 0.06046989]\n  ```\n\n  #### References\n\n  [1] Hull, John C., Options, Futures and Other Derivatives. Pearson, 2018.\n  [2] Wikipedia contributors. Binomial Options Pricing Model. Available at:\n    https://en.wikipedia.org/wiki/Binomial_options_pricing_model\n\n  Args:\n    volatilities: Real `Tensor` of any shape and dtype. The volatilities to\n      expiry of the options to price.\n    strikes: A real `Tensor` of the same dtype and compatible shape as\n      `volatilities`. The strikes of the options to be priced.\n    expiries: A real `Tensor` of same dtype and compatible shape as\n      `volatilities`. The expiry of each option. The units should be such that\n      `expiry * volatility**2` is dimensionless.\n    spots: A real `Tensor` of any shape that broadcasts to the shape of the\n      `volatilities`. The current spot price of the underlying.\n    discount_rates: An optional real `Tensor` of same dtype as the\n      `volatilities`. The risk free discount rate. If None the rate is assumed\n      to be 0.\n      Default value: None, equivalent to discount rates = 0..\n    dividend_rates: An optional real `Tensor` of same dtype as the\n      `volatilities`. If None the rate is assumed to be 0.\n      Default value: None, equivalent to discount rates = 1.\n    is_call_options: A boolean `Tensor` of a shape compatible with\n      `volatilities`. Indicates whether the option is a call (if True) or a put\n      (if False). If not supplied, call options are assumed.\n      Default value: None, equivalent to is_call_options = True.\n    is_american: A boolean `Tensor` of a shape compatible with `volatilities`.\n      Indicates whether the option exercise style is American (if True) or\n      European (if False). If not supplied, European style exercise is assumed.\n      Default value: None, equivalent to is_american = False.\n    num_steps: A positive scalar int32 `Tensor`. The size of the time\n      discretization to use.\n      Default value: 100.\n    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion\n      of any supplied non-`Tensor` arguments to `Tensor`.\n      Default value: None which maps to the default dtype inferred by TensorFlow\n        (float32).\n    name: str. The name for the ops created by this function.\n      Default value: None which is mapped to the default name `option_price`.\n\n  Returns:\n    A `Tensor` of the same shape as the inferred batch shape of the input data.\n    The Black Scholes price of the options computed on a binomial tree.\n  '
    with tf.name_scope(name or 'crr_option_price'):
        strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
        dtype = strikes.dtype
        volatilities = tf.convert_to_tensor(volatilities, dtype=dtype, name='volatilities')
        expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
        spots = tf.convert_to_tensor(spots, dtype=dtype, name='spots')
        if discount_rates is None:
            discount_rates = tf.zeros_like(volatilities)
        else:
            discount_rates = tf.convert_to_tensor(discount_rates, dtype=dtype, name='discount_rates')
        if dividend_rates is None:
            dividend_rates = tf.zeros_like(volatilities)
        else:
            dividend_rates = tf.convert_to_tensor(dividend_rates, dtype=dtype, name='dividend_rates')
        if is_call_options is None:
            is_call_options = tf.ones_like(volatilities, dtype=tf.bool, name='is_call_options')
        else:
            is_call_options = tf.convert_to_tensor(is_call_options, dtype=tf.bool, name='is_call_options')
        if is_american is None:
            is_american = tf.zeros_like(volatilities, dtype=tf.bool, name='is_american')
        else:
            is_american = tf.convert_to_tensor(is_american, dtype=tf.bool, name='is_american')
        num_steps = tf.cast(num_steps, dtype=dtype)
        dt = expiries / num_steps
        ln_up = volatilities * tf.math.sqrt(dt)
        ln_dn = -ln_up
        grid_idx = tf.range(num_steps + 1)
        log_spot_grid_1 = tf.expand_dims(tf.math.log(spots) + ln_up * num_steps, axis=-1)
        log_spot_grid_2 = tf.expand_dims(ln_dn - ln_up, axis=-1) * grid_idx
        log_spot_grid = log_spot_grid_1 + log_spot_grid_2
        payoff_fn = _get_payoff_fn(tf.expand_dims(strikes, axis=-1), tf.expand_dims(is_call_options, axis=-1))
        value_mod_fn = _get_value_modifier(tf.expand_dims(is_american, axis=-1), payoff_fn)
        values_grid = payoff_fn(tf.math.exp(log_spot_grid))
        p_up = tf.math.exp((discount_rates - dividend_rates) * dt + ln_up) - 1
        p_up /= tf.math.exp(2 * ln_up) - 1
        p_up = tf.expand_dims(p_up, axis=-1)
        p_dn = 1 - p_up
        discount_factors = tf.expand_dims(tf.math.exp(-discount_rates * dt), axis=-1)
        ln_up = tf.expand_dims(ln_up, axis=-1)

        def one_step_back(current_values, current_log_spot_grid):
            if False:
                return 10
            next_values = current_values[..., 1:] * p_dn + current_values[..., :-1] * p_up
            next_log_spot_grid = current_log_spot_grid[..., :-1] - ln_up
            next_values = value_mod_fn(next_values, tf.math.exp(next_log_spot_grid))
            return (discount_factors * next_values, next_log_spot_grid)

        def should_continue(current_values, current_log_spot_grid):
            if False:
                return 10
            del current_values, current_log_spot_grid
            return True
        batch_shape = values_grid.shape[:-1]
        (pv, _) = tf.while_loop(should_continue, one_step_back, (values_grid, log_spot_grid), maximum_iterations=tf.cast(num_steps, dtype=tf.int32), shape_invariants=(tf.TensorShape(batch_shape + [None]), tf.TensorShape(batch_shape + [None])))
        return tf.where(expiries > 0, tf.squeeze(pv, axis=-1), tf.where(is_call_options, tf.math.maximum(spots - strikes, 0), tf.math.maximum(strikes - spots, 0)))

def _get_payoff_fn(strikes, is_call_options):
    if False:
        return 10
    'Constructs the payoff functions.'
    option_signs = tf.cast(is_call_options, dtype=strikes.dtype) * 2 - 1

    def payoff(spots):
        if False:
            for i in range(10):
                print('nop')
        'Computes payff for the specified options given the spot grid.\n\n    Args:\n      spots: Tensor of shape [batch_size, grid_size, 1]. The spot values at some\n        time.\n\n    Returns:\n      Payoffs for exercise at the specified strikes.\n    '
        return tf.nn.relu((spots - strikes) * option_signs)
    return payoff

def _get_value_modifier(is_american, payoff_fn):
    if False:
        while True:
            i = 10
    'Constructs the value modifier for american style exercise.'

    def modifier(values, spots):
        if False:
            return 10
        immediate_exercise_value = payoff_fn(spots)
        return tf.where(is_american, tf.math.maximum(immediate_exercise_value, values), values)
    return modifier