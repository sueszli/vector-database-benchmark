"""Implementation of RateCurve object."""
from typing import Optional, Tuple
import tensorflow.compat.v2 as tf
from tf_quant_finance import datetime as dateslib
from tf_quant_finance import math
from tf_quant_finance import rates as rates_lib
from tf_quant_finance.experimental.pricing_platform.framework.core import curve_types
from tf_quant_finance.experimental.pricing_platform.framework.core import daycount_conventions
from tf_quant_finance.experimental.pricing_platform.framework.core import interpolation_method
from tf_quant_finance.experimental.pricing_platform.framework.core import processed_market_data as pmd
from tf_quant_finance.experimental.pricing_platform.framework.core import types
from tf_quant_finance.experimental.pricing_platform.framework.market_data import utils
_DayCountConventions = daycount_conventions.DayCountConventions
_InterpolationMethod = interpolation_method.InterpolationMethod
_DayCountConventionsProtoType = types.DayCountConventionsProtoType

class RateCurve(pmd.RateCurve):
    """Represents an interest rate curve."""

    def __init__(self, maturity_dates: types.DateTensor, discount_factors: tf.Tensor, valuation_date: types.DateTensor, interpolator: Optional[_InterpolationMethod]=None, interpolate_rates: Optional[bool]=True, daycount_convention: Optional[_DayCountConventionsProtoType]=None, curve_type: Optional[curve_types.CurveType]=None, dtype: Optional[tf.DType]=None, name: Optional[str]=None):
        if False:
            return 10
        "Initializes the interest rate curve.\n\n    Args:\n      maturity_dates: A `DateTensor` containing the maturity dates on which the\n        curve is specified.\n      discount_factors: A `Tensor` of real dtype specifying the discount factors\n        corresponding to the input maturities. The shape of this input should\n        match the shape of `maturity_dates`.\n      valuation_date: A scalar `DateTensor` specifying the valuation (or\n        settlement) date for the curve.\n      interpolator: An instance of `InterpolationMethod`.\n        Default value: `None` in which case cubic interpolation is used.\n      interpolate_rates: A boolean specifying whether the interpolation should\n        be done in discount rates or discount factors space.\n        Default value: `True`, i.e., interpolation is done in the discount\n        factors space.\n      daycount_convention: `DayCountConventions` to use for the interpolation\n        purpose.\n        Default value: `None` which maps to actual/365 day count convention.\n      curve_type: An instance of `CurveTypes` to mark the rate curve.\n        Default value: `None` which means that the curve does not have the\n          marker.\n      dtype: `tf.Dtype`. Optional input specifying the dtype of the `rates`\n        input.\n      name: Python str. The name to give to the ops created by this function.\n        Default value: `None` which maps to 'rate_curve'.\n    "
        self._name = name or 'rate_curve'
        with tf.compat.v1.name_scope(self._name):
            self._discount_factor_nodes = tf.convert_to_tensor(discount_factors, dtype=dtype, name='curve_discount_factors')
            self._dtype = dtype or self._discount_factor_nodes.dtype
            if interpolator is None or interpolator == _InterpolationMethod.CUBIC:

                def cubic_interpolator(xi, x, y):
                    if False:
                        i = 10
                        return i + 15
                    spline_coeffs = math.interpolation.cubic.build_spline(x, y)
                    return math.interpolation.cubic.interpolate(xi, spline_coeffs, dtype=dtype)
                interpolator = cubic_interpolator
                self._interpolation_method = _InterpolationMethod.CUBIC
            elif interpolator == _InterpolationMethod.LINEAR:

                def linear_interpolator(xi, x, y):
                    if False:
                        print('Hello World!')
                    return math.interpolation.linear.interpolate(xi, x, y, dtype=dtype)
                interpolator = linear_interpolator
                self._interpolation_method = _InterpolationMethod.LINEAR
            elif interpolator == _InterpolationMethod.CONSTANT_FORWARD:

                def constant_fwd(xi, x, y):
                    if False:
                        while True:
                            i = 10
                    return rates_lib.constant_fwd.interpolate(xi, x, y, dtype=dtype)
                interpolator = constant_fwd
                self._interpolation_method = _InterpolationMethod.CONSTANT_FORWARD
            else:
                raise ValueError(f'Unknown interpolation method {interpolator}.')
            self._dates = dateslib.convert_to_date_tensor(maturity_dates)
            self._valuation_date = dateslib.convert_to_date_tensor(valuation_date)
            self._daycount_convention = daycount_convention or _DayCountConventions.ACTUAL_365
            self._day_count_fn = utils.get_daycount_fn(self._daycount_convention)
            self._times = self._get_time(self._dates)
            self._interpolator = interpolator
            self._interpolate_rates = interpolate_rates
            self._curve_type = curve_type

    @property
    def daycount_convention(self) -> types.DayCountConventionsProtoType:
        if False:
            i = 10
            return i + 15
        'Daycount convention.'
        return self._daycount_convention

    def daycount_fn(self):
        if False:
            return 10
        'Daycount function.'
        return self._day_count_fn

    @property
    def discount_factor_nodes(self) -> types.FloatTensor:
        if False:
            while True:
                i = 10
        'Discount factors at the interpolation nodes.'
        return self._discount_factor_nodes

    @property
    def node_dates(self) -> types.DateTensor:
        if False:
            while True:
                i = 10
        'Dates at which the discount factors and rates are specified.'
        return self._dates

    @property
    def discount_rate_nodes(self) -> types.FloatTensor:
        if False:
            return 10
        'Discount rates at the interpolation nodes.'
        discount_rates = tf.math.divide_no_nan(-tf.math.log(self.discount_factor_nodes), self._times, name='discount_rate_nodes')
        return discount_rates

    def set_discount_factor_nodes(self, values: types.FloatTensor):
        if False:
            for i in range(10):
                print('nop')
        'Update discount factors at the interpolation nodes with new values.'
        values = tf.convert_to_tensor(values, dtype=self._dtype)
        values_shape = values.shape.as_list()
        nodes_shape = self.discount_factor_nodes.shape.as_list()
        if values_shape != nodes_shape:
            raise ValueError('New values should have shape {0} but are of shape {1}'.format(nodes_shape, values_shape))
        self._discount_factor_nodes = values

    def discount_rate(self, interpolation_dates: Optional[types.DateTensor]=None, interpolation_times: Optional[types.FloatTensor]=None, name: Optional[str]=None):
        if False:
            print('Hello World!')
        'Returns interpolated rates at `interpolation_dates`.'
        if interpolation_dates is None and interpolation_times is None:
            raise ValueError('Either interpolation_dates or interpolation times must be supplied.')
        if interpolation_dates is not None:
            interpolation_dates = dateslib.convert_to_date_tensor(interpolation_dates)
            times = self._get_time(interpolation_dates)
        else:
            times = tf.convert_to_tensor(interpolation_times, self._dtype)
        rates = self._interpolator(times, self._times, self.discount_rate_nodes)
        if self._interpolate_rates:
            rates = self._interpolator(times, self._times, self.discount_rate_nodes)
        else:
            discount_factor = self._interpolator(times, self._times, self.discount_factor_nodes)
            rates = -tf.math.divide_no_nan(tf.math.log(discount_factor), times)
        return tf.identity(rates, name=name or 'discount_rate')

    def discount_factor(self, interpolation_dates: Optional[types.DateTensor]=None, interpolation_times: Optional[types.FloatTensor]=None, name: Optional[str]=None):
        if False:
            print('Hello World!')
        'Returns discount factors at `interpolation_dates`.'
        if interpolation_dates is None and interpolation_times is None:
            raise ValueError('Either interpolation_dates or interpolation times must be supplied.')
        if interpolation_dates is not None:
            interpolation_dates = dateslib.convert_to_date_tensor(interpolation_dates)
            times = self._get_time(interpolation_dates)
        else:
            times = tf.convert_to_tensor(interpolation_times, self._dtype)
        if self._interpolate_rates:
            rates = self._interpolator(times, self._times, self.discount_rate_nodes)
            discount_factor = tf.math.exp(-rates * times)
        else:
            discount_factor = self._interpolator(times, self._times, self.discount_factor_nodes)
        return tf.identity(discount_factor, name=name or 'discount_factor')

    def forward_rate(self, start_date: Optional[types.DateTensor]=None, maturity_date: Optional[types.DateTensor]=None, start_time: Optional[types.FloatTensor]=None, maturity_time: Optional[types.FloatTensor]=None, day_count_fraction: Optional[tf.Tensor]=None):
        if False:
            i = 10
            return i + 15
        "Returns the simply accrued forward rate between [start_dt, maturity_dt].\n\n    Args:\n      start_date: A `DateTensor` specifying the start of the accrual period\n        for the forward rate. The function expects either `start_date` or\n        `start_time` to be specified.\n      maturity_date: A `DateTensor` specifying the end of the accrual period\n        for the forward rate. The shape of `end_date` must be broadcastable\n        with the shape of `start_date`. The function expects either `end_date`\n        or `end_time` to be specified.\n      start_time: A real `Tensor` specifying the start of the accrual period\n        for the forward rate. The function expects either `start_date` or\n        `start_time` to be specified.\n      maturity_time: A real `Tensor` specifying the end of the accrual period\n        for the forward rate. The shape of `end_date` must be broadcastable\n        with the shape of `start_date`. The function expects either `end_date`\n        or `end_time` to be specified.\n      day_count_fraction: An optional `Tensor` of real dtype specifying the\n        time between `start_date` and `maturity_date` in years computed using\n        the forward rate's day count basis. The shape of the input should be\n        the same as that of `start_date` and `maturity_date`.\n        Default value: `None`, in which case the daycount fraction is computed\n          using `daycount_convention`.\n\n    Returns:\n      A real `Tensor` of same shape as the inputs containing the simply\n      compounded forward rate.\n    "
        if start_date is None and start_time is None:
            raise ValueError('Either start_date or start_times must be supplied.')
        if maturity_date is None and maturity_time is None:
            raise ValueError('Either maturity_date or maturity_time must be supplied.')
        if start_date is not None and maturity_date is not None:
            start_date = dateslib.convert_to_date_tensor(start_date)
            maturity_date = dateslib.convert_to_date_tensor(maturity_date)
            if day_count_fraction is None:
                day_count_fn = self._day_count_fn
                day_count_fraction = day_count_fn(start_date=start_date, end_date=maturity_date, dtype=self._dtype)
            else:
                day_count_fraction = tf.convert_to_tensor(day_count_fraction, self._dtype, name='day_count_fraction')
            start_time = self._get_time(start_date)
            maturity_time = self._get_time(maturity_date)
        else:
            start_time = tf.convert_to_tensor(start_time, dtype=self._dtype)
            maturity_time = tf.convert_to_tensor(maturity_time, dtype=self._dtype)
            day_count_fraction = maturity_time - start_time
        dfstart = self.discount_factor(interpolation_times=start_time)
        dfmaturity = self.discount_factor(interpolation_times=maturity_time)
        return tf.math.divide_no_nan(tf.math.divide_no_nan(dfstart, dfmaturity) - 1.0, day_count_fraction)

    @property
    def valuation_date(self) -> types.DateTensor:
        if False:
            while True:
                i = 10
        return self._valuation_date

    @property
    def interpolation_method(self) -> _InterpolationMethod:
        if False:
            return 10
        return self._interpolation_method

    def _get_time(self, dates: types.DateTensor) -> types.FloatTensor:
        if False:
            print('Hello World!')
        "Computes the year fraction from the curve's valuation date."
        return self._day_count_fn(start_date=self._valuation_date, end_date=dates, dtype=self._dtype)

    @property
    def curve_type(self) -> curve_types.CurveType:
        if False:
            print('Hello World!')
        return self._curve_type

    def discount_factors_and_dates(self) -> Tuple[types.FloatTensor, types.DateTensor]:
        if False:
            return 10
        'Returns discount factors and dates at which the discount curve is fitted.\n    '
        return (self._discount_factor_nodes, self._dates)

    @property
    def dtype(self) -> types.Dtype:
        if False:
            print('Hello World!')
        return self._dtype

    @property
    def interpolate_rates(self) -> bool:
        if False:
            while True:
                i = 10
        'Returns `True` if the interpolation is on rates and not on discounts.'
        return self._interpolate_rates
__all__ = ['RateCurve']