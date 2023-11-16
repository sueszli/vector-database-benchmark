"""Implementation of VolatilitySurface object."""
from typing import Optional, Callable
import tensorflow.compat.v2 as tf
from tf_quant_finance import datetime as dateslib
from tf_quant_finance import math
from tf_quant_finance.experimental.pricing_platform.framework.core import daycount_conventions
from tf_quant_finance.experimental.pricing_platform.framework.core import implied_volatility_type
from tf_quant_finance.experimental.pricing_platform.framework.core import processed_market_data as pmd
from tf_quant_finance.experimental.pricing_platform.framework.core import types
from tf_quant_finance.experimental.pricing_platform.framework.market_data import utils
_DayCountConventions = daycount_conventions.DayCountConventions
_DayCountConventionsProtoType = types.DayCountConventionsProtoType
interpolation_2d = math.interpolation.interpolation_2d

class VolatilitySurface(pmd.VolatilitySurface):
    """Represents a volatility surface."""

    def __init__(self, valuation_date: types.DateTensor, expiries: types.DateTensor, strikes: types.FloatTensor, volatilities: types.FloatTensor, daycount_convention: Optional[_DayCountConventionsProtoType]=None, interpolator: Optional[Callable[[types.FloatTensor, types.FloatTensor], types.FloatTensor]]=None, dtype: Optional[tf.DType]=None, name: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        "Initializes the volatility surface.\n\n    Args:\n      valuation_date: A `DateTensor` specifying the valuation (or\n        settlement) date for the curve.\n      expiries: A `DateTensor` containing the expiry dates on which the\n        implied volatilities are specified. Should have a compatible shape with\n        valuation_date.\n      strikes: A `Tensor` of real dtype specifying the strikes corresponding to\n        the input maturities. The shape of this input should match the shape of\n        `expiries`.\n      volatilities: A `Tensor` of real dtype specifying the volatilities\n        corresponding to  the input maturities. The shape of this input should\n        match the shape of `expiries`.\n      daycount_convention: `DayCountConventions` to use for the interpolation\n        purpose.\n        Default value: `None` which maps to actual/365 day count convention.\n      interpolator: An optional Python callable implementing the interpolation\n        to be used. The callable should accept two real `Tensor`s specifying\n        the strikes and expiry times and should return a real `Tensor` of\n        same dtype as the inputs containing the interpolated implied\n        volatilities.\n        Default value: `None` in which case `Interpolation2D` is used.\n      dtype: `tf.Dtype`. Optional input specifying the dtype of the `rates`\n        input.\n      name: Python str. The name to give to the ops created by this function.\n        Default value: `None` which maps to 'rate_curve'.\n    "
        self._name = name or 'VolatilitySurface'
        self._dtype = dtype or tf.float64
        with tf.name_scope(self._name):
            self._daycount_convention = daycount_convention or _DayCountConventions.ACTUAL_365
            self._day_count_fn = utils.get_daycount_fn(self._daycount_convention)
            self._valuation_date = dateslib.convert_to_date_tensor(valuation_date)
            self._expiries = dateslib.convert_to_date_tensor(expiries)
            self._strikes = tf.convert_to_tensor(strikes, dtype=self._dtype, name='strikes')
            self._volatilities = tf.convert_to_tensor(volatilities, dtype=self._dtype, name='volatilities')
            expiry_times = self._day_count_fn(start_date=self._valuation_date, end_date=self._expiries, dtype=self._dtype)
            if interpolator is None:
                interpolator_obj = interpolation_2d.Interpolation2D(expiry_times, strikes, volatilities, dtype=self._dtype)

                def interpolator_fn(t, x):
                    if False:
                        i = 10
                        return i + 15
                    return interpolator_obj.interpolate(t, x)
                self._interpolator = interpolator_fn
            else:
                self._interpolator = interpolator

    def volatility(self, strike: types.FloatTensor, expiry_dates: Optional[types.DateTensor]=None, expiry_times: Optional[types.FloatTensor]=None, term: Optional[types.Period]=None) -> types.FloatTensor:
        if False:
            while True:
                i = 10
        'Returns the interpolated volatility on a specified set of expiries.\n\n    Args:\n      strike: The strikes for which the interpolation is desired.\n      expiry_dates: Optional input specifying the expiry dates for which\n        interpolation is desired. The user should supply either `expiry_dates`\n        or `expiry_times` for interpolation.\n      expiry_times: Optional real `Tensor` containing the time to expiration\n        for which interpolation is desired. The user should supply either\n        `expiry_dates` or `expiry_times` for interpolation.\n      term: Optional input specifying the term of the underlying rate for\n        which the interpolation is desired. Relevant for interest rate implied\n        volatility data.\n\n    Returns:\n      A `Tensor` of the same shape as `expiry` with the interpolated volatility\n      from the volatility surface.\n\n    Raises:\n      ValueError is both `expiry_dates` and `expiry_times`  are specified.\n    '
        del term
        if expiry_dates is not None and expiry_times is not None:
            raise ValueError('Unexpected inputs: Both expiry_dates and expiry times are specified')
        if expiry_times is None:
            expiry_dates = dateslib.convert_to_date_tensor(expiry_dates)
            expiries = self._day_count_fn(start_date=self._valuation_date, end_date=expiry_dates, dtype=self._dtype)
        else:
            expiries = tf.convert_to_tensor(expiry_times, dtype=self._dtype)
        strike = tf.convert_to_tensor(strike, dtype=self._dtype, name='strike')
        return self._interpolator(expiries, strike)

    def settlement_date(self) -> types.DateTensor:
        if False:
            for i in range(10):
                print('nop')
        'Returns the valuation/settlement date.'
        return self._valuation_date

    def volatility_type(self) -> implied_volatility_type.ImpliedVolatilityType:
        if False:
            while True:
                i = 10
        'Returns the type of implied volatility.'
        pass

    def node_expiries(self) -> types.DateTensor:
        if False:
            print('Hello World!')
        'Expiry dates at which the implied volatilities are specified.'
        return self._expiries

    def node_strikes(self) -> tf.Tensor:
        if False:
            for i in range(10):
                print('nop')
        'Strikes at which the implied volatilities are specified.'
        return self._strikes

    def node_volatilities(self) -> tf.Tensor:
        if False:
            print('Hello World!')
        'Market implied volatilities.'
        return self._volatilities

    @property
    def daycount_convention(self) -> _DayCountConventionsProtoType:
        if False:
            return 10
        return self._daycount_convention

    def node_terms(self) -> types.Period:
        if False:
            for i in range(10):
                print('nop')
        'Rate terms corresponding to the specified implied volatilities.'
        pass

    def interpolator(self):
        if False:
            print('Hello World!')
        return self._interpolator
__all__ = ['VolatilitySurface']