"""Interface for the Market data."""
import abc
import datetime
from typing import Any, List, Tuple, Callable, Optional
import tensorflow.compat.v2 as tf
from tf_quant_finance.experimental.pricing_platform.framework.core import curve_types
from tf_quant_finance.experimental.pricing_platform.framework.core import daycount_conventions
from tf_quant_finance.experimental.pricing_platform.framework.core import implied_volatility_type
from tf_quant_finance.experimental.pricing_platform.framework.core import interpolation_method
from tf_quant_finance.experimental.pricing_platform.framework.core import types

class RateCurve(abc.ABC):
    """Interface for interest rate curves."""

    @abc.abstractmethod
    def discount_factor(self, date: Optional[types.DateTensor]=None, time: Optional[types.FloatTensor]=None, **kwargs) -> tf.Tensor:
        if False:
            for i in range(10):
                print('nop')
        'Returns the discount factor to a specified set of dates.\n\n    Args:\n      date: Optional input specifying the dates at which to evaluate the\n        discount factors. The function expects either `date` or `time` to be\n        specified.\n      time: Optional input specifying the times at which to evaluate the\n        discount factors. The function expects either `date` or `time` to be\n        specified.\n      **kwargs: The context object, e.g., curve_type.\n\n    Returns:\n      A `Tensor` of the same shape as `dates` with the corresponding discount\n      factors.\n    '
        pass

    @abc.abstractmethod
    def forward_rate(self, start_date: Optional[types.DateTensor]=None, end_date: Optional[types.DateTensor]=None, start_time: Optional[types.FloatTensor]=None, end_time: Optional[types.FloatTensor]=None, **kwargs) -> tf.Tensor:
        if False:
            print('Hello World!')
        'Returns the simply accrued forward rate between dates.\n\n    Args:\n      start_date: A `DateTensor` specifying the start of the accrual period\n        for the forward rate. The function expects either `start_date` or\n        `start_time` to be specified.\n      end_date: A `DateTensor` specifying the end of the accrual period\n        for the forward rate. The shape of `end_date` must be broadcastable\n        with the shape of `start_date`. The function expects either `end_date`\n        or `end_time` to be specified.\n      start_time: A real `Tensor` specifying the start of the accrual period\n        for the forward rate. The function expects either `start_date` or\n        `start_time` to be specified.\n      end_time: A real `Tensor` specifying the end of the accrual period\n        for the forward rate. The shape of `end_date` must be broadcastable\n        with the shape of `start_date`. The function expects either `end_date`\n        or `end_time` to be specified.\n      **kwargs: The context object, e.g., curve_type.\n\n    Returns:\n      A `Tensor` with the corresponding forward rates.\n    '
        pass

    @abc.abstractmethod
    def discount_rate(self, date: Optional[types.DateTensor]=None, time: Optional[types.FloatTensor]=None, context=None) -> tf.Tensor:
        if False:
            while True:
                i = 10
        'Returns the discount rates to a specified set of dates.\n\n    Args:\n      date: A `DateTensor` specifying the dates at which to evaluate the\n        discount rates. The function expects either `date` or `time` to be\n        specified.\n      time: A real `Tensor` specifying the times at which to evaluate the\n        discount rates. The function expects either `date` or `time` to be\n        specified.\n      context: The context object, e.g., curve_type.\n\n    Returns:\n      A `Tensor` of the same shape as `dates` with the corresponding discount\n      rates.\n    '
        pass

    @property
    @abc.abstractmethod
    def curve_type(self) -> Any:
        if False:
            for i in range(10):
                print('nop')
        'Returns type of the curve.'
        pass

    @abc.abstractmethod
    def interpolation_method(self) -> interpolation_method.InterpolationMethod:
        if False:
            return 10
        'Interpolation method used for this discount curve.'
        pass

    @abc.abstractmethod
    def discount_factors_and_dates(self) -> Tuple[types.FloatTensor, types.DateTensor]:
        if False:
            return 10
        'Returns discount factors and dates at which the discount curve is fitted.\n    '
        pass

    @abc.abstractproperty
    def discount_factor_nodes(self) -> types.FloatTensor:
        if False:
            print('Hello World!')
        'Discount factors at the interpolation nodes.'
        pass

    @abc.abstractmethod
    def set_discount_factor_nodes(self, values: types.FloatTensor) -> types.FloatTensor:
        if False:
            return 10
        'Update discount factors at the interpolation nodes with new values.'
        pass

    @abc.abstractproperty
    def discount_rate_nodes(self) -> types.FloatTensor:
        if False:
            i = 10
            return i + 15
        'Discount rates at the interpolation nodes.'
        pass

    @abc.abstractproperty
    def node_dates(self) -> types.DateTensor:
        if False:
            i = 10
            return i + 15
        'Dates at which the discount factors and rates are specified.'
        return self._dates

    @abc.abstractproperty
    def daycount_convention(self) -> types.DayCountConventionsProtoType:
        if False:
            while True:
                i = 10
        'Daycount convention.'
        raise NotImplementedError

    @abc.abstractmethod
    def daycount_fn(self) -> Callable[..., Any]:
        if False:
            for i in range(10):
                print('nop')
        'Daycount function.'
        raise NotImplementedError

class VolatilitySurface(abc.ABC):
    """Interface for implied volatility surface."""

    @abc.abstractmethod
    def volatility(self, strike: types.FloatTensor, expiry_dates: Optional[types.DateTensor]=None, expiry_times: Optional[types.FloatTensor]=None, term: Optional[types.Period]=None) -> types.FloatTensor:
        if False:
            print('Hello World!')
        'Returns the interpolated volatility on a specified set of expiries.\n\n    Args:\n      strike: The strikes for which the interpolation is desired.\n      expiry_dates: Optional input specifying the expiry dates for which\n        interpolation is desired. The user should supply either `expiry_dates`\n        or `expiry_times` for interpolation.\n      expiry_times: Optional real `Tensor` containing the time to expiration\n        for which interpolation is desired. The user should supply either\n        `expiry_dates` or `expiry_times` for interpolation.\n      term: Optional input specifying the term of the underlying rate for\n        which the interpolation is desired. Relevant for interest rate implied\n        volatility data.\n\n    Returns:\n      A `Tensor` of the same shape as `expiry` with the interpolated volatility\n      from the volatility surface.\n    '
        pass

    @property
    @abc.abstractmethod
    def volatility_type(self) -> implied_volatility_type.ImpliedVolatilityType:
        if False:
            for i in range(10):
                print('nop')
        'Returns the type of implied volatility.'
        pass

    @property
    @abc.abstractmethod
    def node_expiries(self) -> types.DateTensor:
        if False:
            return 10
        'Expiry dates at which the implied volatilities are specified.'
        return self._expiries

    @property
    @abc.abstractmethod
    def node_strikes(self) -> tf.Tensor:
        if False:
            for i in range(10):
                print('nop')
        'Striks at which the implied volatilities are specified.'
        return self._strikes

    @property
    @abc.abstractmethod
    def node_terms(self) -> types.Period:
        if False:
            i = 10
            return i + 15
        'Rate terms corresponding to the specified implied volatilities.'
        return self._terms

class ProcessedMarketData(abc.ABC):
    """Market data snapshot used by pricing library."""

    @abc.abstractproperty
    def date(self) -> datetime.date:
        if False:
            while True:
                i = 10
        'The date of the market data.'
        pass

    @abc.abstractproperty
    def time(self) -> datetime.time:
        if False:
            i = 10
            return i + 15
        'The time of the snapshot.'
        pass

    @abc.abstractmethod
    def yield_curve(self, curve_type: curve_types.CurveType) -> RateCurve:
        if False:
            return 10
        'The yield curve object.'
        pass

    @abc.abstractmethod
    def fixings(self, date: types.DateTensor, fixing_type: curve_types.RateIndexCurve) -> Tuple[tf.Tensor, daycount_conventions.DayCountConventions]:
        if False:
            i = 10
            return i + 15
        'Returns past fixings of the market rates at the specified dates.'
        pass

    @abc.abstractmethod
    def spot(self, asset: str, data: types.DateTensor) -> tf.Tensor:
        if False:
            print('Hello World!')
        'The spot price of an asset.'
        pass

    @abc.abstractmethod
    def volatility_surface(self, asset: str) -> VolatilitySurface:
        if False:
            return 10
        'The volatility surface object for an asset.'
        pass

    @abc.abstractmethod
    def forward_curve(self, asset: str) -> RateCurve:
        if False:
            i = 10
            return i + 15
        'The forward curve of the asset prices object.'
        pass

    @abc.abstractproperty
    def supported_currencies(self) -> List[str]:
        if False:
            return 10
        'List of supported currencies.'
        pass

    @abc.abstractmethod
    def supported_assets(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        'List of supported assets.'
        pass

    @abc.abstractproperty
    def dtype(self) -> types.Dtype:
        if False:
            for i in range(10):
                print('nop')
        'Type of the float calculations.'
        pass
__all__ = ['RateCurve', 'ProcessedMarketData']