"""Local Volatility Model."""
import functools
from typing import Any, Callable, List, Optional, Union
import tensorflow.compat.v2 as tf
from tf_quant_finance import black_scholes
from tf_quant_finance import datetime
from tf_quant_finance import math
from tf_quant_finance import types
from tf_quant_finance.experimental.pricing_platform.framework.market_data import volatility_surface
from tf_quant_finance.math import interpolation
from tf_quant_finance.math import piecewise
from tf_quant_finance.math import random
from tf_quant_finance.models import generic_ito_process
interpolation_2d = interpolation.interpolation_2d
cubic = interpolation.cubic
linear = interpolation.linear

def _dupire_local_volatility_prices(time, spot_price, initial_spot_price, implied_volatility_surface, discount_factor_fn, dividend_yield):
    if False:
        i = 10
        return i + 15
    "Constructs local volatility function using Dupire's formula.\n\n  Args:\n    time: A real `Tensor` of shape compatible with `spot_price` specifying the\n      times at which local volatility function is computed.\n    spot_price: A real `Tensor` specifying the underlying price at which local\n      volatility function is computed.\n    initial_spot_price: A real `Tensor` of shape compatible with `spot_price`\n      specifying the underlying spot price at t=0.\n    implied_volatility_surface: A Python callable which implements the\n      interpolation of market implied volatilities. The callable should have the\n      interface `implied_volatility_surface(strike, expiry_times)` which takes\n      real `Tensor`s corresponding to option strikes and time to expiry and\n      returns a real `Tensor` containing the corresponding market implied\n      volatility. The shape of `strike` is `(n,dim)` where `dim` is the\n      dimensionality of the local volatility process and `t` is a scalar tensor.\n      The output from the callable is a `Tensor` of shape `(n,dim)` containing\n      the interpolated implied volatilties.\n    discount_factor_fn: A python callable accepting one real `Tensor` argument\n      time t. It should return a `Tensor` specifying the discount factor to time\n      t.\n    dividend_yield: A real `Tensor` of shape compatible with `spot_price`\n      specifying the (continuously compounded) dividend yield.\n\n  Returns:\n    A real `Tensor` of same shape as `spot_price` containing the local\n    volatility computed at `(spot_price,time)` using Dupire's\n    construction of local volatility.\n  "
    dtype = time.dtype
    risk_free_rate_fn = _get_risk_free_rate_from_discount_factor(discount_factor_fn)
    risk_free_rate = tf.convert_to_tensor(risk_free_rate_fn(time), dtype=dtype)

    def _option_price(expiry_time, strike):
        if False:
            return 10
        discount_factors = tf.convert_to_tensor(discount_factor_fn(expiry_time), dtype=dtype)
        vols = implied_volatility_surface(strike=strike, expiry_times=expiry_time)
        c_k_t = black_scholes.option_price(volatilities=vols, strikes=strike, expiries=expiry_time, spots=initial_spot_price, dividend_rates=dividend_yield, discount_factors=discount_factors, dtype=dtype)
        return c_k_t
    dcdk_fn = lambda x: _option_price(time, x)
    dcdt_fn = lambda x: _option_price(x, spot_price)
    d2cdk2_fn = lambda x: math.fwd_gradient(dcdk_fn, x)
    numerator = math.fwd_gradient(dcdt_fn, time) + (risk_free_rate - dividend_yield) * spot_price * math.fwd_gradient(dcdk_fn, spot_price) + dividend_yield * _option_price(time, spot_price)
    denominator = math.fwd_gradient(d2cdk2_fn, spot_price) * spot_price ** 2
    local_volatility_squared = tf.nn.relu(2 * tf.math.divide_no_nan(numerator, denominator))
    return tf.math.sqrt(local_volatility_squared)

def _dupire_local_volatility_iv(time, spot_price, initial_spot_price, implied_volatility_surface, discount_factor_fn, dividend_yield):
    if False:
        return 10
    'Similar to _dupire_local_volatility_prices, but uses implied vols.'
    dtype = time.dtype
    risk_free_rate_fn = _get_risk_free_rate_from_discount_factor(discount_factor_fn)
    risk_free_rate = tf.convert_to_tensor(risk_free_rate_fn(time), dtype=dtype)

    def _implied_vol(expiry_time, strike):
        if False:
            while True:
                i = 10
        return implied_volatility_surface(strike=strike, expiry_times=expiry_time)
    theta = _implied_vol(time, spot_price)
    d1 = tf.math.divide_no_nan((tf.math.log(initial_spot_price / spot_price) + (risk_free_rate - dividend_yield + 0.5 * theta ** 2) * time) / theta, tf.math.sqrt(time))
    spot_fn = lambda x: _implied_vol(time, x)
    time_fn = lambda t: _implied_vol(t, spot_price)
    dtheta_dt = lambda t: math.fwd_gradient(time_fn, t)
    dtheta_dspot = lambda x: math.fwd_gradient(spot_fn, x)
    d2theta_dspot2 = lambda x: math.fwd_gradient(dtheta_dspot, x)
    numerator = theta ** 2 + 2 * time * theta * dtheta_dt(time) + 2 * (risk_free_rate - dividend_yield) * spot_price * time * theta * dtheta_dspot(spot_price)
    denominator = (1 + spot_price * d1 * time * dtheta_dspot(spot_price)) ** 2 + spot_price ** 2 * time * theta * (d2theta_dspot2(spot_price) - d1 * time * dtheta_dspot(spot_price) ** 2)
    local_volatility_squared = tf.nn.relu(tf.math.divide_no_nan(numerator, denominator))
    return tf.math.sqrt(local_volatility_squared)

def _dupire_local_volatility_iv_precomputed(initial_spot_price: types.RealTensor, implied_volatility_surface: Any, discount_factor_fn: Callable[..., types.RealTensor], dividend_yield: Callable[..., types.RealTensor], times_grid: types.RealTensor, spot_grid: types.RealTensor, dim: int, dtype: tf.DType) -> Callable[..., types.RealTensor]:
    if False:
        i = 10
        return i + 15
    "Constructs local volatility function using Dupire's formula.\n\n  Returns a function similar to _dupire_local_volatility_iv using precomputed\n  spline coefficients for IV.\n\n  Args:\n    initial_spot_price: A real `Tensor` specifying the underlying spot price at\n      t=0.\n    implied_volatility_surface: A Python callable which implements the\n      interpolation of market implied volatilities. The callable should have the\n      interface `implied_volatility_surface(strike, expiry_times)` which takes\n      real `Tensor`s corresponding to option strikes and time to expiry and\n      returns a real `Tensor` containing the corresponding market implied\n      volatility. The shape of `strike` is `(n,dim)` where `dim` is the\n      dimensionality of the local volatility process and `t` is a scalar tensor.\n      The output from the callable is a `Tensor` of shape `(n,dim)` containing\n      the interpolated implied volatilties.\n    discount_factor_fn: A python callable accepting one real `Tensor` argument\n      time t. It should return a `Tensor` specifying the discount factor to time\n      t.\n    dividend_yield: A real `Tensor` of shape compatible with `spot_price`\n      specifying the (continuously compounded) dividend yield.\n    times_grid: A `Tensor` of shape `[num_time_samples]`. The grid on which to\n      do interpolation over time.\n    spot_grid: A `Tensor` of shape `[num_spot_samples]`. The grid on which to do\n      interpolation over spots.\n    dim: An int. The model dimension.\n    dtype: The default dtype to use when converting values to `Tensor`s.\n\n  Returns:\n    A python callable f(spot_price,time) returning a real `Tensor` of same shape\n    as `spot_price` containing the local volatility computed at\n    `(spot_price,time)` using Dupire's construction of local volatility.\n  "
    times_grid = tf.convert_to_tensor(times_grid, dtype=dtype)
    spot_grid = tf.convert_to_tensor(spot_grid, dtype=dtype)
    times_grid = tf.expand_dims(times_grid, 0)
    spot_grid = tf.expand_dims(spot_grid, 0)

    def map_times(times):
        if False:
            print('Hello World!')

        def wrap_implied_vol(time):
            if False:
                while True:
                    i = 10
            spots = tf.transpose(spot_grid)
            return implied_volatility_surface(strike=spots, expiry_times=time)
        return tf.vectorized_map(wrap_implied_vol, tf.transpose(times))
    implied_vol_grid = map_times(times_grid)
    implied_vol_grid = tf.transpose(implied_vol_grid, [0, 2, 1])

    def map_times_grad(times):
        if False:
            return 10

        def wrap_implied_vol(time):
            if False:
                return 10
            spots = tf.transpose(spot_grid)
            return implied_volatility_surface(strike=spots, expiry_times=time)
        fn_grad = lambda time: math.fwd_gradient(wrap_implied_vol, time)
        return tf.vectorized_map(fn_grad, tf.transpose(times))
    grad_implied_vol_grid = map_times_grad(times_grid)
    grad_implied_vol_grid = tf.transpose(grad_implied_vol_grid, [0, 2, 1])
    times_grid = tf.squeeze(times_grid)
    spot_grid = tf.squeeze(spot_grid)
    spot_grid_2d = tf.broadcast_to(spot_grid, [times_grid.shape[0], dim, spot_grid.shape[0]])
    spline_info = cubic.build_spline(spot_grid_2d, implied_vol_grid, name='spline_spots', validate_args=True)
    jump_locations = tf.slice(times_grid, [1], [times_grid.shape[0] - 1])
    spline_coeffs_fn = piecewise.PiecewiseConstantFunc(jump_locations, spline_info.spline_coeffs)
    spot_grid_2d_fn = piecewise.PiecewiseConstantFunc(jump_locations, spot_grid_2d, dtype=dtype)
    implied_vol_grid_fn = piecewise.PiecewiseConstantFunc(jump_locations, implied_vol_grid, dtype=dtype)
    spline_grad_info = cubic.build_spline(spot_grid_2d, grad_implied_vol_grid, name='spline_grad_spots', validate_args=True)
    grad_implied_vol_grid_fn = piecewise.PiecewiseConstantFunc(jump_locations, grad_implied_vol_grid, dtype=dtype)
    spline_grad_coeffs_fn = piecewise.PiecewiseConstantFunc(jump_locations, spline_grad_info.spline_coeffs)

    def local_variance(time, spot_price):
        if False:
            i = 10
            return i + 15
        risk_free_rate_fn = _get_risk_free_rate_from_discount_factor(discount_factor_fn)
        risk_free_rate = tf.convert_to_tensor(risk_free_rate_fn(time), dtype=dtype)
        spot_grid_2d = tf.squeeze(spot_grid_2d_fn([time]), 0)
        implied_vol_grid = tf.squeeze(implied_vol_grid_fn([time]), 0)
        spline_coeffs = tf.squeeze(spline_coeffs_fn([time]), 0)
        spline = math.interpolation.cubic.SplineParameters(spot_grid_2d, implied_vol_grid, spline_coeffs)

        def theta_fn(spot):
            if False:
                return 10
            return tf.transpose(math.interpolation.cubic.interpolate(tf.transpose(spot), spline, dtype=dtype))

        def grad_spots_fn(spot):
            if False:
                i = 10
                return i + 15
            return math.fwd_gradient(theta_fn, spot, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        theta = theta_fn(spot_price)
        grad_implied_vol_grid = tf.squeeze(grad_implied_vol_grid_fn([time]), 0)
        spline_grad_coeffs = tf.squeeze(spline_grad_coeffs_fn([time]), 0)
        spline_grad = math.interpolation.cubic.SplineParameters(spot_grid_2d, grad_implied_vol_grid, spline_grad_coeffs)
        grad_times = tf.transpose(math.interpolation.cubic.interpolate(tf.transpose(spot_price), spline_grad, dtype=dtype))
        d1 = tf.math.divide_no_nan((tf.math.log(tf.math.divide_no_nan(initial_spot_price, spot_price)) + (risk_free_rate - dividend_yield + 0.5 * theta ** 2) * time) / theta, tf.math.sqrt(time))
        spots_t = spot_price
        grad_spots = grad_spots_fn(spots_t)
        grad_grad_spots = math.fwd_gradient(grad_spots_fn, spots_t, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        local_var = (theta ** 2 + 2 * time * theta * grad_times + 2 * (risk_free_rate - dividend_yield) * spots_t * time * theta * grad_spots) / ((1 + spots_t * d1 * time * grad_spots) ** 2 + spots_t ** 2 * time * theta * (grad_grad_spots - d1 * time * grad_spots ** 2))
        local_var = tf.nn.relu(local_var)
        return tf.math.sqrt(local_var)
    return local_variance

class LocalVolatilityModel(generic_ito_process.GenericItoProcess):
    """Local volatility model for smile modeling.

  Local volatility (LV) model specifies that the dynamics of an asset is
  governed by the following stochastic differential equation:

  ```None
    dS(t) / S(t) =  mu(t, S(t)) dt + sigma(t, S(t)) * dW(t)
  ```
  where `mu(t, S(t))` is the drift and `sigma(t, S(t))` is the instantaneous
  volatility. The local volatility function `sigma(t, S(t))` is state dependent
  and is computed by calibrating against a given implied volatility surface
  `sigma_iv(T, K)` using the Dupire's formula [1]:

  ```
  sigma(T,K)^2 = 2 * (dC(T,K)/dT + (r - q)K dC(T,K)/dK + qC(T,K)) /
                     (K^2 d2C(T,K)/dK2)
  ```
  where the derivatives above are the partial derivatives. The LV model provides
  a flexible framework to model any (arbitrage free) volatility surface.

  #### Example: Simulation of local volatility process.

  ```python
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tf_quant_finance as tff

  dtype = tf.float64
  dim = 2
  year = dim * [[2021, 2022]]
  month = dim * [[1, 1]]
  day = dim * [[1, 1]]
  expiries = tff.datetime.dates_from_year_month_day(year, month, day)
  valuation_date = [(2020, 1, 1)]
  expiry_times = tff.datetime.daycount_actual_365_fixed(
      start_date=valuation_date, end_date=expiries, dtype=dtype)
  strikes = dim * [[[0.1, 0.9, 1.0, 1.1, 3], [0.1, 0.9, 1.0, 1.1, 3]]]
  iv = dim * [[[0.135, 0.13, 0.1, 0.11, 0.13],
               [0.135, 0.13, 0.1, 0.11, 0.13]]]
  spot = dim * [1.0]
  risk_free_rate = [0.02]
  r = tf.convert_to_tensor(risk_free_rate, dtype=dtype)
  df = lambda t: tf.math.exp(-r * t)

  lv = tff.models.LocalVolatilityModel.from_market_data(
      dim, val_date, expiries, strikes, iv, spot, df, [0.0], dtype=dtype)
  num_samples = 10000
  paths = lv.sample_paths(
      [1.0, 1.5, 2.0],
      num_samples=num_samples,
      initial_state=spot,
      time_step=0.1,
      random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC,
      seed=[1, 2])
  # paths.shape = (10000, 3, 2)

  #### References:
    [1]: Iain J. Clark. Foreign exchange option pricing - A Practitioner's
    guide. Chapter 5, Section 5.3.2. 2011.
  """

    def __init__(self, dim: int, risk_free_rate: Optional[Union[types.RealTensor, Optional[types.RealTensor]]]=None, dividend_yield: Optional[Union[types.RealTensor, Optional[types.RealTensor]]]=None, local_volatility_fn: Optional[Callable[..., types.RealTensor]]=None, corr_matrix: Optional[types.RealTensor]=None, times_grid: Optional[types.RealTensor]=None, spot_grid: Optional[types.RealTensor]=None, precompute_iv: Optional[bool]=False, dtype: Optional[tf.DType]=None, name: Optional[str]=None):
        if False:
            while True:
                i = 10
        "Initializes the Local volatility model.\n\n    If `precompute_iv` is True and `times_grid` and `spot_grid` are supplied, an\n    interpolated version that pre-computes spline coefficients is used.\n\n    Args:\n      dim: A Python scalar which corresponds to the number of underlying assets\n        comprising the model.\n      risk_free_rate: One of the following: (a) An optional real `Tensor` of\n        shape compatible with `[dim]` specifying the (continuously compounded)\n        risk free interest rate. (b) A python callable accepting one real\n        `Tensor` argument time t returning a `Tensor` of shape compatible with\n        `[dim]`. If the underlying is an FX rate, then use this input to specify\n        the domestic interest rate.\n        Default value: `None` in which case the input is set to Zero.\n      dividend_yield: A real `Tensor` of shape compatible with `spot_price`\n        specifying the (continuously compounded) dividend yield. If the\n        underlying is an FX rate, then use this input to specify the foreign\n        interest rate.\n        Default value: `None` in which case the input is set to Zero.\n      local_volatility_fn: A Python callable which returns instantaneous\n        volatility as a function of state and time. The function must accept a\n        scalar `Tensor` corresponding to time 't' and a real `Tensor` of shape\n        `[num_samples, dim]` corresponding to the underlying price (S) as inputs\n        and return a real `Tensor` of shape `[num_samples, dim]` containing the\n        local volatility computed at (S,t).\n      corr_matrix: A `Tensor` of shape `[dim, dim]` and the same `dtype` as\n        `risk_free_rate`. Corresponds to the instantaneous correlation between\n        the underlying assets.\n      times_grid: A `Tensor` of shape `[num_time_samples]`. The grid on which to\n        do interpolation over time. Must be jointly specified with `spot_grid`.\n        Default value: `None`.\n      spot_grid: A `Tensor` of shape `[num_spot_samples]`. The grid on which to\n        do interpolation over spots. Must be jointly specified with\n        `times_grid`.\n        Default value: `None`.\n      precompute_iv: A bool. Whether or not to precompute implied volatility\n        spline coefficients when using Dupire. If True, then `times_grid` and\n        `spot_grid` must be supplied.\n        Default value: False.\n      dtype: The default dtype to use when converting values to `Tensor`s.\n        Default value: `None` which means that default dtypes inferred by\n          TensorFlow are used.\n      name: Python string. The name to give to the ops created by this class.\n        Default value: `None` which maps to the default name\n          `local_volatility_model`.\n\n    Raises:\n      ValueError: If `precompute_iv` is True, but grids are not supplied.\n    "
        self._name = name or 'local_volatility_model'
        self._times_grid = None
        self._local_volatility_fn = local_volatility_fn
        self._precompute_iv = precompute_iv
        if precompute_iv:
            if times_grid is None or spot_grid is None:
                raise ValueError('When `precompute_iv` is True, both `times_grid` and `spot_grid` must be supplied')
            self._times_grid = times_grid
        with tf.name_scope(self._name):
            self._dtype = dtype
            risk_free_rate = [0.0] if risk_free_rate is None else risk_free_rate
            dividend_yield = [0.0] if dividend_yield is None else dividend_yield
            self._domestic_rate = _convert_to_tensor_fn(risk_free_rate, dtype, 'risk_free_rate')
            self._foreign_rate = _convert_to_tensor_fn(dividend_yield, dtype, 'dividend_yield')
            corr_matrix = corr_matrix or tf.eye(dim, dim, dtype=self._dtype)
            self._rho = tf.convert_to_tensor(corr_matrix, dtype=self._dtype, name='rho')
            self._sqrt_rho = tf.linalg.cholesky(self._rho)

            def _vol_fn(t, log_spot):
                if False:
                    for i in range(10):
                        print('nop')
                'Volatility function of LV model.'
                lv = self._local_volatility_fn(t, tf.math.exp(log_spot))
                lv = tf.expand_dims(lv, axis=-1)
                return lv * self._sqrt_rho

            def _drift_fn(t, log_spot):
                if False:
                    while True:
                        i = 10
                'Drift function of LV model.'
                domestic_rate = self._domestic_rate(t)
                foreign_rate = self._foreign_rate(t)
                lv = self._local_volatility_fn(t, tf.math.exp(log_spot))
                return domestic_rate - foreign_rate - lv * lv / 2
            super(LocalVolatilityModel, self).__init__(dim, _drift_fn, _vol_fn, dtype, name)

    def local_volatility_fn(self):
        if False:
            while True:
                i = 10
        'Local volatility function.'
        return self._local_volatility_fn

    def precompute_iv(self) -> bool:
        if False:
            while True:
                i = 10
        "Whether or not to precompute IV in Dupire's formula."
        return self._precompute_iv

    def sample_paths(self, times: types.RealTensor, num_samples: Optional[int]=1, initial_state: Optional[types.RealTensor]=None, random_type: Optional[random.RandomType]=None, seed: Optional[types.IntTensor]=None, swap_memory: Optional[bool]=True, time_step: Optional[types.RealTensor]=None, num_time_steps: Optional[types.IntTensor]=None, skip: Optional[types.IntTensor]=0, precompute_normal_draws: Optional[bool]=True, times_grid: Optional[types.RealTensor]=None, normal_draws: Optional[types.RealTensor]=None, watch_params: Optional[List[types.RealTensor]]=None, validate_args: Optional[bool]=False, name: Optional[str]=None) -> types.RealTensor:
        if False:
            while True:
                i = 10
        'Returns samples from the LV process.\n\n    See GenericItoProcess.sample_paths. If `times_grid` is supplied to\n    `__init__`, then `times_grid` cannot be supplied here.\n\n    Raises:\n      ValueError: If `precompute_iv` is True, but `time_step`, `num_time_steps`\n        or `times_grid` are given.\n    '
        name = name or self._name + '_log_sample_path'
        with tf.name_scope(name):
            if initial_state is not None:
                initial_state = tf.math.log(tf.convert_to_tensor(initial_state, dtype_hint=tf.float64))
            if self.precompute_iv():
                if time_step is not None or num_time_steps is not None or times_grid is not None:
                    raise ValueError('`time_step`, `num_time_steps`, or `times_grid` cannot be usedwith the interpolated LVM')
                times_grid = self._times_grid
            return tf.math.exp(super(LocalVolatilityModel, self).sample_paths(times=times, num_samples=num_samples, initial_state=initial_state, random_type=random_type, seed=seed, swap_memory=swap_memory, name=name, time_step=time_step, num_time_steps=num_time_steps, skip=skip, precompute_normal_draws=precompute_normal_draws, times_grid=times_grid, normal_draws=normal_draws, watch_params=watch_params, validate_args=validate_args))

    @classmethod
    def from_market_data(cls, dim: int, valuation_date: types.DateTensor, expiry_dates: types.DateTensor, strikes: types.RealTensor, implied_volatilities: types.RealTensor, spot: types.RealTensor, discount_factor_fn: Callable[..., types.RealTensor], dividend_yield: Optional[Callable[..., types.RealTensor]]=None, local_volatility_from_iv: Optional[bool]=True, times_grid: Optional[types.RealTensor]=None, spot_grid: Optional[types.RealTensor]=None, precompute_iv: Optional[bool]=False, dtype: Optional[tf.DType]=None, name: Optional[str]=None) -> generic_ito_process.GenericItoProcess:
        if False:
            while True:
                i = 10
        "Creates a `LocalVolatilityModel` from market data.\n\n    Args:\n      dim: A Python scalar which corresponds to the number of underlying assets\n        comprising the model.\n      valuation_date: A `DateTensor` specifying the valuation (or settlement)\n        date for the market data.\n      expiry_dates: A `DateTensor` of shape `(dim, num_expiries)` containing the\n        expiry dates on which the implied volatilities are specified.\n      strikes: A `Tensor` of real dtype and shape `(dim, num_expiries,\n        num_strikes)`specifying the strike prices at which implied volatilities\n        are specified.\n      implied_volatilities: A `Tensor` of real dtype and shape `(dim,\n        num_expiries, num_strikes)` specifying the implied volatilities.\n      spot: A real `Tensor` of shape `(dim,)` specifying the underlying spot\n        price on the valuation date.\n      discount_factor_fn: A python callable accepting one real `Tensor` argument\n        time t. It should return a `Tensor` specifying the discount factor to\n        time t.\n      dividend_yield: A real `Tensor` of shape compatible with `spot_price`\n        specifying the (continuously compounded) dividend yield. If the\n        underlying is an FX rate, then use this input to specify the foreign\n        interest rate.\n        Default value: `None` in which case the input is set to Zero.\n      local_volatility_from_iv: A bool. If True, calculates the Dupire local\n        volatility function from implied volatilities. Otherwise, it computes\n        the local volatility using option prices.\n        Default value: True.\n      times_grid: A `Tensor` of shape `[num_time_samples]`. The grid on which to\n        do interpolation over time. Must be jointly specified with `spot_grid`\n        or `None`.\n        Default value: `None`.\n      spot_grid: A `Tensor` of shape `[num_spot_samples]`. The grid on which to\n        do interpolation over spots. Must be jointly specified with `times_grid`\n        or `None`.\n        Default value: `None`.\n      precompute_iv: A bool. Whether or not to precompute implied volatility\n        spline coefficients when using Dupire's formula. This is done by\n        precomputing values of implied volatility on the grid\n        `times_grid x spot_grid`. The algorithm then steps through `times_grid`\n        in `sample_paths`.\n        Default value: False.\n      dtype: The default dtype to use when converting values to `Tensor`s.\n        Default value: `None` which means that default dtypes inferred by\n          TensorFlow are used.\n      name: Python string. The name to give to the ops created by this class.\n        Default value: `None` which maps to the default name `from_market_data`.\n\n    Returns:\n      An instance of `LocalVolatilityModel` constructed using the input data.\n    "
        name = name or 'from_market_data'
        if precompute_iv and (times_grid is None or spot_grid is None):
            raise ValueError('When `precompute_iv` is True, both `times_grid` and `spot_grid` must be supplied')
        with tf.name_scope(name):
            spot = tf.convert_to_tensor(spot, dtype=dtype)
            dtype = dtype or spot.dtype
            dividend_yield = [0.0] if dividend_yield is None else dividend_yield
            dividend_yield = tf.convert_to_tensor(dividend_yield, dtype=dtype)
            risk_free_rate_fn = _get_risk_free_rate_from_discount_factor(discount_factor_fn)
            valuation_date = datetime.convert_to_date_tensor(valuation_date)
            expiry_dates = datetime.convert_to_date_tensor(expiry_dates)
            expiry_times = tf.cast(valuation_date.days_until(expiry_dates), dtype=dtype) / 365.0
            strikes = tf.convert_to_tensor(strikes, dtype=dtype)
            implied_volatilities = tf.convert_to_tensor(implied_volatilities, dtype=dtype)

            def _log_forward_moneyness(times, strikes):
                if False:
                    return 10
                risk_free_rate = tf.squeeze(risk_free_rate_fn(times))
                log_forward_moneyness = tf.math.log(tf.math.divide_no_nan(strikes, tf.reshape(spot, [dim, 1, 1]))) - tf.expand_dims((risk_free_rate - dividend_yield) * times, axis=-1)
                return log_forward_moneyness
            interpolator = interpolation_2d.Interpolation2D(expiry_times, _log_forward_moneyness(expiry_times, strikes), implied_volatilities, dtype=dtype)

            def _log_moneyness_2d_interpolator(times, strikes):
                if False:
                    return 10
                risk_free_rate = risk_free_rate_fn(times)
                log_forward_moneyness = tf.math.log(strikes / spot) - (risk_free_rate - dividend_yield) * times
                moneyness_transposed = tf.transpose(log_forward_moneyness)
                times = tf.broadcast_to(times, moneyness_transposed.shape)
                return tf.transpose(interpolator.interpolate(times, moneyness_transposed))
            vs = volatility_surface.VolatilitySurface(valuation_date, expiry_dates, strikes, implied_volatilities, interpolator=_log_moneyness_2d_interpolator, dtype=dtype)
            if precompute_iv:
                local_volatility_fn = _dupire_local_volatility_iv_precomputed(initial_spot_price=spot, discount_factor_fn=discount_factor_fn, dividend_yield=dividend_yield, implied_volatility_surface=vs.volatility, times_grid=times_grid, spot_grid=spot_grid, dim=dim, dtype=dtype)
            else:
                local_volatility_fn = functools.partial(_dupire_local_volatility_iv if local_volatility_from_iv else _dupire_local_volatility_prices, initial_spot_price=spot, discount_factor_fn=discount_factor_fn, dividend_yield=dividend_yield, implied_volatility_surface=vs.volatility)
            return LocalVolatilityModel(dim, risk_free_rate=risk_free_rate_fn, dividend_yield=dividend_yield, local_volatility_fn=local_volatility_fn, times_grid=times_grid, spot_grid=spot_grid, precompute_iv=precompute_iv, dtype=dtype)

    @classmethod
    def from_volatility_surface(cls, dim: int, spot: types.RealTensor, implied_volatility_surface: Any, discount_factor_fn: Callable[..., types.RealTensor], dividend_yield: Optional[Callable[..., types.RealTensor]]=None, local_volatility_from_iv: Optional[bool]=True, times_grid: Optional[types.RealTensor]=None, spot_grid: Optional[types.RealTensor]=None, precompute_iv: Optional[bool]=False, dtype: Optional[tf.DType]=None, name: Optional[str]=None) -> generic_ito_process.GenericItoProcess:
        if False:
            i = 10
            return i + 15
        "Creates a `LocalVolatilityModel` from implied volatility data.\n\n    Args:\n      dim: A Python scalar which corresponds to the number of underlying assets\n        comprising the model.\n      spot: A real `Tensor` of shape `(dim,)` specifying the underlying spot\n        price on the valuation date.\n      implied_volatility_surface: Either an instance of\n        `processed_market_data.VolatilitySurface` or a Python object containing\n        the implied volatility market data. If the input is a Python object,\n        then the object must implement a function `volatility(strike,\n        expiry_times)` which takes real `Tensor`s corresponding to option\n        strikes and time to expiry and returns a real `Tensor` containing the\n        corresponding market implied volatility. The shape of `strike` is\n        `(n,dim)` where `dim` is the dimensionality of the local volatility\n        process and `t` is a scalar tensor. The output from the callable is a\n        `Tensor` of shape `(n,dim)` containing the interpolated implied\n        volatilties.\n      discount_factor_fn: A python callable accepting one real `Tensor` argument\n        time t. It should return a `Tensor` specifying the discount factor to\n        time t.\n      dividend_yield: A real `Tensor` of shape compatible with `spot_price`\n        specifying the (continuously compounded) dividend yield. If the\n        underlying is an FX rate, then use this input to specify the foreign\n        interest rate.\n        Default value: `None` in which case the input is set to Zero.\n      local_volatility_from_iv: A bool. If True, calculates the Dupire local\n        volatility function from implied volatilities. Otherwise, it computes\n        the local volatility using option prices.\n        Default value: True.\n      times_grid: A `Tensor` of shape `[num_time_samples]`. The grid on which to\n        do interpolation over time. Must be jointly specified with `spot_grid`\n        or `None`.\n        Default value: `None`.\n      spot_grid: A `Tensor` of shape `[num_spot_samples]`. The grid on which to\n        do interpolation over spots. Must be jointly specified with `times_grid`\n        or `None`.\n        Default value: `None`.\n      precompute_iv: A bool. Whether or not to precompute implied volatility\n        spline coefficients when using Dupire's formula. This is done by\n        precomputing values of implied volatility on the grid\n        `times_grid x spot_grid`. The algorithm then steps through `times_grid`\n        in `sample_paths`.\n        Default value: False.\n      dtype: The default dtype to use when converting values to `Tensor`s.\n        Default value: `None` which means that default dtypes inferred by\n          TensorFlow are used.\n      name: Python string. The name to give to the ops created by this class.\n        Default value: `None` which maps to the default name\n          `from_volatility_surface`.\n\n    Returns:\n      An instance of `LocalVolatilityModel` constructed using the input data.\n    "
        name = name or 'from_volatility_surface'
        if precompute_iv and (times_grid is None or spot_grid is None):
            raise ValueError('When `precompute_iv` is True, both `times_grid` and `spot_grid` must be supplied')
        with tf.name_scope(name):
            dividend_yield = [0.0] if dividend_yield is None else dividend_yield
            dividend_yield = tf.convert_to_tensor(dividend_yield, dtype=dtype)
            risk_free_rate_fn = _get_risk_free_rate_from_discount_factor(discount_factor_fn)
            if precompute_iv:
                local_volatility_fn = _dupire_local_volatility_iv_precomputed(initial_spot_price=spot, discount_factor_fn=discount_factor_fn, dividend_yield=dividend_yield, implied_volatility_surface=implied_volatility_surface.volatility, times_grid=times_grid, spot_grid=spot_grid, dim=dim, dtype=dtype)
            else:
                local_volatility_fn = functools.partial(_dupire_local_volatility_iv if local_volatility_from_iv else _dupire_local_volatility_prices, initial_spot_price=spot, discount_factor_fn=discount_factor_fn, dividend_yield=dividend_yield, implied_volatility_surface=implied_volatility_surface.volatility)
            return LocalVolatilityModel(dim, risk_free_rate=risk_free_rate_fn, dividend_yield=dividend_yield, local_volatility_fn=local_volatility_fn, times_grid=times_grid, spot_grid=spot_grid, precompute_iv=precompute_iv, dtype=dtype)

def _convert_to_tensor_fn(x, dtype, name):
    if False:
        for i in range(10):
            print('nop')
    if callable(x):
        return x
    else:
        return lambda t: tf.convert_to_tensor(x, dtype, name=name)

def _get_risk_free_rate_from_discount_factor(discount_factor_fn):
    if False:
        for i in range(10):
            print('nop')
    'Returns r(t) given a discount factor function.'

    def risk_free_rate_fn(t):
        if False:
            for i in range(10):
                print('nop')
        logdf = lambda x: -tf.math.log(discount_factor_fn(x))
        return math.fwd_gradient(logdf, t, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    return risk_free_rate_fn