"""Cashflow streams objects."""
from typing import Optional, Tuple, Callable, Any, List, Union
import numpy as np
import tensorflow.compat.v2 as tf
from tf_quant_finance import datetime as dateslib
from tf_quant_finance.experimental.pricing_platform.framework.core import curve_types as curve_types_lib
from tf_quant_finance.experimental.pricing_platform.framework.core import processed_market_data as pmd
from tf_quant_finance.experimental.pricing_platform.framework.core import types
from tf_quant_finance.experimental.pricing_platform.framework.market_data import rate_curve
from tf_quant_finance.experimental.pricing_platform.framework.market_data import utils as market_data_utils
from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments import coupon_specs
from tf_quant_finance.experimental.pricing_platform.instrument_protos import period_pb2
from tf_quant_finance.math import pad
_CurveType = curve_types_lib.CurveType

class FixedCashflowStream:
    """Represents a batch of fixed stream of cashflows."""

    def __init__(self, coupon_spec: coupon_specs.FixedCouponSpecs, discount_curve_type: Union[_CurveType, List[_CurveType]], start_date: types.DateTensor=None, end_date: types.DateTensor=None, discount_curve_mask: types.IntTensor=None, first_coupon_date: Optional[types.DateTensor]=None, penultimate_coupon_date: Optional[types.DateTensor]=None, schedule_fn: Optional[Callable[..., Any]]=None, schedule: Optional[types.DateTensor]=None, dtype: Optional[types.Dtype]=None, name: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        "Initializes a batch of fixed cashflow streams.\n\n    Args:\n      coupon_spec: An instance of `FixedCouponSpecs` specifying the\n        details of the coupon payment for the cashflow stream.\n      discount_curve_type: An instance of `CurveType` or a list of those.\n        If supplied as a list and `discount_curve_mask` is not supplied,\n        the size of the list should be the same as the number of priced\n        instruments. Defines discount curves for the instruments.\n      start_date: A `DateTensor` of `batch_shape` specifying the starting dates\n        of the accrual of the first coupon of the cashflow stream. The shape of\n        the input correspond to the number of streams being created.\n        When passed as an integet `Tensor`, should be of shape\n        `batch_shape + [3]` and contain `[year, month, day]` for each date.\n        Either this of `schedule` should be supplied\n        Default value: `None`\n      end_date: A `DateTensor` of `batch_shape`specifying the end dates for\n        accrual of the last coupon in each cashflow stream. The shape of the\n        input should be the same as that of `start_date`.\n        Either this of `schedule` should be supplied\n        When passed as an integet `Tensor`, should be of shape\n        `batch_shape + [3]` and contain `[year, month, day]` for each date.\n        Default value: `None`\n      discount_curve_mask: An optional integer `Tensor` of values ranging from\n        `0` to `len(discount_curve_type) - 1` and of shape `batch_shape`.\n        Identifies a mapping between `discount_curve_type` list and the\n        underlying instruments.\n        Default value: `None`.\n      first_coupon_date: An optional `DateTensor` specifying the payment dates\n        of the first coupon of the cashflow stream. Use this input for cashflows\n        with irregular first coupon. Should be of the same shape as\n        `start_date`.\n        When passed as an integet `Tensor`, should be of shape\n        `batch_shape + [3]` and contain `[year, month, day]` for each date.\n        Default value: None which implies regular first coupon.\n      penultimate_coupon_date: An optional `DateTensor` specifying the payment\n        dates of the penultimate (next to last) coupon of the cashflow\n        stream. Use this input for cashflows with irregular last coupon.\n        Should be of the same shape as `end_date`.\n        When passed as an integet `Tensor`, should be of shape\n        `batch_shape + [3]` and contain `[year, month, day]` for each date.\n        Default value: None which implies regular last coupon.\n      schedule_fn: A callable that accepts `start_date`, `end_date`,\n        `coupon_frequency`, `settlement_days`, `first_coupon_date`, and\n        `penultimate_coupon_date` as `Tensor`s and returns coupon payment\n        days.\n        Default value: `None`.\n      schedule: A `DateTensor` of coupon payment dates including the start and\n        end dates of the cashflows.\n        Default value: `None`.\n      dtype: `tf.Dtype` of the input and output real `Tensor`s.\n        Default value: None which maps to the default dtype inferred by\n        TensorFlow.\n      name: Python str. The name to give to the ops created by this class.\n        Default value: `None` which maps to 'fixed_cashflow_stream'.\n    "
        self._name = name or 'fixed_cashflow_stream'
        with tf.name_scope(self._name):
            curve_list = to_list(discount_curve_type)
            [self._discount_curve_type, self._mask] = process_curve_types(curve_list, discount_curve_mask)
            if schedule is None:
                if start_date is None or end_date is None:
                    raise ValueError('If `schedule` is not supplied both `start_date` and `end_date` should be supplied')
                if isinstance(start_date, tf.Tensor):
                    self._start_date = dateslib.dates_from_tensor(start_date)
                else:
                    self._start_date = dateslib.convert_to_date_tensor(start_date)
                if isinstance(start_date, tf.Tensor):
                    self._end_date = dateslib.dates_from_tensor(end_date)
                else:
                    self._end_date = dateslib.convert_to_date_tensor(end_date)
                self._first_coupon_date = first_coupon_date
                self._penultimate_coupon_date = penultimate_coupon_date
                if self._first_coupon_date is not None:
                    if isinstance(start_date, tf.Tensor):
                        self._first_coupon_date = dateslib.dates_from_tensor(first_coupon_date)
                    else:
                        self._first_coupon_date = dateslib.convert_to_date_tensor(first_coupon_date)
                if self._penultimate_coupon_date is not None:
                    if isinstance(start_date, tf.Tensor):
                        self._penultimate_coupon_date = dateslib.dates_from_tensor(penultimate_coupon_date)
                    else:
                        self._penultimate_coupon_date = dateslib.convert_to_date_tensor(penultimate_coupon_date)
            coupon_frequency = _get_attr(coupon_spec, 'coupon_frequency')
            if isinstance(coupon_frequency, period_pb2.Period):
                coupon_frequency = market_data_utils.get_period(_get_attr(coupon_spec, 'coupon_frequency'))
            if isinstance(coupon_frequency, (list, tuple)):
                coupon_frequency = market_data_utils.period_from_list(*_get_attr(coupon_spec, 'coupon_frequency'))
            if isinstance(coupon_frequency, dict):
                coupon_frequency = market_data_utils.period_from_dict(_get_attr(coupon_spec, 'coupon_frequency'))
            businessday_rule = coupon_spec.businessday_rule
            (roll_convention, eom) = market_data_utils.get_business_day_convention(businessday_rule)
            notional = tf.convert_to_tensor(_get_attr(coupon_spec, 'notional_amount'), dtype=dtype, name='notional')
            self._dtype = dtype or notional.dtype
            fixed_rate = tf.convert_to_tensor(_get_attr(coupon_spec, 'fixed_rate'), dtype=self._dtype, name='fixed_rate')
            daycount_fn = market_data_utils.get_daycount_fn(_get_attr(coupon_spec, 'daycount_convention'), self._dtype)
            self._settlement_days = tf.convert_to_tensor(_get_attr(coupon_spec, 'settlement_days'), dtype=tf.int32, name='settlement_days')
            if schedule is not None:
                if isinstance(schedule, tf.Tensor):
                    coupon_dates = dateslib.dates_from_tensor(schedule)
                else:
                    coupon_dates = dateslib.convert_to_date_tensor(schedule)
                self._start_date = coupon_dates[..., 0]
            elif schedule_fn is None:
                calendar = dateslib.create_holiday_calendar(weekend_mask=dateslib.WeekendMask.SATURDAY_SUNDAY)
                self._calendar = calendar
                coupon_dates = _generate_schedule(start_date=self._start_date, end_date=self._end_date, coupon_frequency=coupon_frequency, roll_convention=roll_convention, calendar=calendar, settlement_days=self._settlement_days, end_of_month=eom, first_coupon_date=self._first_coupon_date, penultimate_coupon_date=self._penultimate_coupon_date)
                self._start_date = coupon_dates[..., 0]
            else:
                if first_coupon_date is not None:
                    first_coupon_date = self._first_coupon_date.to_tensor()
                if penultimate_coupon_date is not None:
                    penultimate_coupon_date = self._penultimate_coupon_date.to_tensor()
                    coupon_dates = schedule_fn(start_date=self._start_date.to_tensor(), end_date=self._end_date.to_tensor(), coupon_frequency=coupon_frequency.quantity(), settlement_days=self._settlement_days, first_coupon_date=first_coupon_date, penultimate_coupon_date=penultimate_coupon_date)
            coupon_dates = dateslib.convert_to_date_tensor(coupon_dates)
            self._batch_shape = tf.shape(coupon_dates.ordinal())[:-1]
            payment_dates = coupon_dates[..., 1:]
            daycount_fractions = daycount_fn(start_date=coupon_dates[..., :-1], end_date=coupon_dates[..., 1:])
            coupon_rate = tf.expand_dims(fixed_rate, axis=-1)
            self._num_cashflows = tf.shape(payment_dates.ordinal())[-1]
            self._payment_dates = payment_dates
            self._notional = notional
            self._daycount_fractions = daycount_fractions
            self._coupon_rate = coupon_rate
            self._fixed_rate = tf.convert_to_tensor(fixed_rate, dtype=self._dtype)
            self._daycount_fn = daycount_fn

    def daycount_fn(self) -> Callable[..., Any]:
        if False:
            for i in range(10):
                print('nop')
        return self._daycount_fn

    @property
    def daycount_fractions(self) -> types.FloatTensor:
        if False:
            while True:
                i = 10
        return self._daycount_fractions

    @property
    def fixed_rate(self) -> types.FloatTensor:
        if False:
            for i in range(10):
                print('nop')
        return self._fixed_rate

    @property
    def notional(self) -> types.FloatTensor:
        if False:
            while True:
                i = 10
        return self._notional

    @property
    def discount_curve_type(self) -> _CurveType:
        if False:
            while True:
                i = 10
        return self._discount_curve_type

    @property
    def batch_shape(self) -> types.StringTensor:
        if False:
            for i in range(10):
                print('nop')
        return self._batch_shape

    @property
    def cashflow_dates(self) -> types.DateTensor:
        if False:
            i = 10
            return i + 15
        return self._payment_dates

    def cashflows(self, market: pmd.ProcessedMarketData, name: Optional[str]=None) -> Tuple[types.DateTensor, types.FloatTensor]:
        if False:
            while True:
                i = 10
        "Returns cashflows for the fixed leg.\n\n    Args:\n      market: An instance of `ProcessedMarketData`.\n      name: Python str. The name to give to the ops created by this function.\n        Default value: `None` which maps to 'cashflows'.\n\n    Returns:\n      A tuple of two `Tensor`s of shape `batch_shape + [num_cashflows]` and\n      containing the dates and the corresponding cashflows price for each\n      stream based on the input market data.\n    "
        name = name or self._name + '_cashflows'
        with tf.name_scope(name):
            valuation_date = dateslib.convert_to_date_tensor(market.date)
            future_cashflows = tf.cast(self._payment_dates >= valuation_date, dtype=self._dtype)
            notional = tf.expand_dims(self._notional, axis=-1)
            cashflows = notional * (future_cashflows * self._daycount_fractions * self._coupon_rate)
            return (self._payment_dates, cashflows)

    def price(self, market: pmd.ProcessedMarketData, name: Optional[str]=None):
        if False:
            i = 10
            return i + 15
        "Returns the present value of the stream on the valuation date.\n\n    Args:\n      market: An instance of `ProcessedMarketData`.\n      name: Python str. The name to give to the ops created by this function.\n        Default value: `None` which maps to 'price'.\n\n    Returns:\n      A `Tensor` of shape `batch_shape`  containing the modeled price of each\n      stream based on the input market data.\n    "
        name = name or self._name + '_price'
        with tf.name_scope(name):
            discount_curve = get_discount_curve(self._discount_curve_type, market, self._mask)
            discount_factors = discount_curve.discount_factor(self._payment_dates)
            (_, cashflows) = self.cashflows(market)
            cashflow_pvs = cashflows * discount_factors
            return tf.math.reduce_sum(cashflow_pvs, axis=1)

class FloatingCashflowStream:
    """Represents a batch of cashflows indexed to a floating rate."""

    def __init__(self, coupon_spec: coupon_specs.FloatCouponSpecs, discount_curve_type: Union[_CurveType, List[_CurveType]], start_date: types.DateTensor=None, end_date: types.DateTensor=None, discount_curve_mask: types.IntTensor=None, rate_index_curves: Union[curve_types_lib.RateIndexCurve, List[curve_types_lib.RateIndexCurve]]=None, reference_mask: types.IntTensor=None, first_coupon_date: Optional[types.DateTensor]=None, penultimate_coupon_date: Optional[types.DateTensor]=None, schedule_fn: Optional[Callable[..., Any]]=None, schedule: Optional[types.DateTensor]=None, past_fixing: Optional[types.FloatTensor]=None, dtype: Optional[types.Dtype]=None, name: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        "Initializes a batch of floating cashflow streams.\n\n    Args:\n      coupon_spec: An instance of `FloatCouponSpecs` specifying the\n        details of the coupon payment for the cashflow stream.\n      discount_curve_type: An instance of `CurveType` or a list of those.\n        If supplied as a list and `discount_curve_mask` is not supplied,\n        the size of the list should be the same as the number of priced\n        instruments. Defines discount curves for the instruments.\n      start_date: A `DateTensor` of `batch_shape` specifying the starting dates\n        of the accrual of the first coupon of the cashflow stream. The shape of\n        the input correspond to the number of streams being created.\n        Either this of `schedule` should be supplied.\n        When passed as an integet `Tensor`, should be of shape\n        `batch_shape + [3]` and contain `[year, month, day]` for each date.\n        Default value: `None`\n      end_date: A `DateTensor` of `batch_shape`specifying the end dates for\n        accrual of the last coupon in each cashflow stream. The shape of the\n        input should be the same as that of `start_date`.\n        Either this of `schedule` should be supplied.\n        When passed as an integet `Tensor`, should be of shape\n        `batch_shape + [3]` and contain `[year, month, day]` for each date.\n        Default value: `None`\n      discount_curve_mask: An optional integer `Tensor` of values ranging from\n        `0` to `len(discount_curve_type) - 1` and of shape `batch_shape`.\n        Identifies a mapping between `discount_curve_type` list and the\n        underlying instruments.\n        Default value: `None`.\n      rate_index_curves: An instance of `RateIndexCurve` or a list of those.\n        If supplied as a list and `reference_mask` is not supplid,\n        the size of the list should be the same as the number of priced\n        instruments. Defines the index curves for each instrument. If not\n        supplied, `coupon_spec.floating_rate_type` is used to identify the\n        curves.\n        Default value: `None`.\n      reference_mask: An optional integer `Tensor` of values ranging from\n        `0` to `len(rate_index_curves) - 1` and of shape `batch_shape`.\n        Identifies a mapping between `rate_index_curves` list and the underlying\n        instruments.\n        Default value: `None`.\n      first_coupon_date: An optional `DateTensor` specifying the payment dates\n        of the first coupon of the cashflow stream. Use this input for cashflows\n        with irregular first coupon. Should be of the same shape as\n        `start_date`.\n        When passed as an integet `Tensor`, should be of shape\n        `batch_shape + [3]` and contain `[year, month, day]` for each date.\n        Default value: None which implies regular first coupon.\n      penultimate_coupon_date: An optional `DateTensor` specifying the payment\n        dates of the penultimate (next to last) coupon of the cashflow\n        stream. Use this input for cashflows with irregular last coupon.\n        Should be of the same shape as `end_date`.\n        When passed as an integet `Tensor`, should be of shape\n        `batch_shape + [3]` and contain `[year, month, day]` for each date.\n        Default value: None which implies regular last coupon.\n      schedule_fn: A callable that accepts `start_date`, `end_date`,\n        `coupon_frequency`, `settlement_days`, `first_coupon_date`, and\n        `penultimate_coupon_date` as `Tensor`s and returns coupon payment\n        days.\n        Default value: `None`.\n      schedule: A `DateTensor` of coupon payment dates including the start and\n        end dates of the cashflows.\n        Default value: `None`.\n      past_fixing: An optional `Tensor` of shape compatible with\n        `batch_shape + [1]`. Represents the fixings for the cashflows as\n        observed at `market.date`.\n      dtype: `tf.Dtype` of the input and output real `Tensor`s.\n        Default value: None which maps to the default dtype inferred by\n        TensorFlow.\n      name: Python str. The name to give to the ops created by this class.\n        Default value: `None` which maps to 'floating_cashflow_stream'.\n    "
        self._name = name or 'floating_cashflow_stream'
        with tf.name_scope(self._name):
            curve_list = to_list(discount_curve_type)
            [self._discount_curve_type, self._mask] = process_curve_types(curve_list, discount_curve_mask)
            self._first_coupon_date = None
            self._penultimate_coupon_date = None
            if schedule is None:
                if start_date is None or end_date is None:
                    raise ValueError('If `schedule` is not supplied both `start_date` and `end_date` should be supplied')
            if schedule is None:
                if isinstance(start_date, tf.Tensor):
                    self._start_date = dateslib.dates_from_tensor(start_date)
                else:
                    self._start_date = dateslib.convert_to_date_tensor(start_date)
                if isinstance(start_date, tf.Tensor):
                    self._end_date = dateslib.dates_from_tensor(end_date)
                else:
                    self._end_date = dateslib.convert_to_date_tensor(end_date)
                self._first_coupon_date = first_coupon_date
                self._penultimate_coupon_date = penultimate_coupon_date
                if self._first_coupon_date is not None:
                    if isinstance(start_date, tf.Tensor):
                        self._first_coupon_date = dateslib.dates_from_tensor(first_coupon_date)
                    else:
                        self._first_coupon_date = dateslib.convert_to_date_tensor(first_coupon_date)
                if self._penultimate_coupon_date is not None:
                    if isinstance(start_date, tf.Tensor):
                        self._penultimate_coupon_date = dateslib.dates_from_tensor(penultimate_coupon_date)
                    else:
                        self._penultimate_coupon_date = dateslib.convert_to_date_tensor(penultimate_coupon_date)
            coupon_frequency = _get_attr(coupon_spec, 'coupon_frequency')
            if isinstance(coupon_frequency, period_pb2.Period):
                coupon_frequency = market_data_utils.get_period(_get_attr(coupon_spec, 'coupon_frequency'))
            if isinstance(coupon_frequency, (list, tuple)):
                coupon_frequency = market_data_utils.period_from_list(*_get_attr(coupon_spec, 'coupon_frequency'))
            if isinstance(coupon_frequency, dict):
                coupon_frequency = market_data_utils.period_from_dict(_get_attr(coupon_spec, 'coupon_frequency'))
            reset_frequency = _get_attr(coupon_spec, 'reset_frequency')
            if isinstance(reset_frequency, period_pb2.Period):
                reset_frequency = market_data_utils.get_period(_get_attr(coupon_spec, 'reset_frequency'))
            if isinstance(reset_frequency, (list, tuple)):
                reset_frequency = market_data_utils.period_from_list(*_get_attr(coupon_spec, 'reset_frequency'))
            if isinstance(reset_frequency, dict):
                reset_frequency = market_data_utils.period_from_dict(_get_attr(coupon_spec, 'reset_frequency'))
            self._reset_frequency = reset_frequency
            businessday_rule = _get_attr(coupon_spec, 'businessday_rule')
            (roll_convention, eom) = market_data_utils.get_business_day_convention(businessday_rule)
            notional = tf.convert_to_tensor(_get_attr(coupon_spec, 'notional_amount'), dtype=dtype, name='notional')
            self._dtype = dtype or notional.dtype
            daycount_convention = _get_attr(coupon_spec, 'daycount_convention')
            daycount_fn = market_data_utils.get_daycount_fn(_get_attr(coupon_spec, 'daycount_convention'), self._dtype)
            self._daycount_convention = daycount_convention
            self._settlement_days = tf.convert_to_tensor(_get_attr(coupon_spec, 'settlement_days'), dtype=tf.int32, name='settlement_days')
            spread = tf.convert_to_tensor(_get_attr(coupon_spec, 'spread'), dtype=self._dtype, name='spread')
            if schedule is not None:
                if isinstance(schedule, tf.Tensor):
                    coupon_dates = dateslib.dates_from_tensor(schedule)
                else:
                    coupon_dates = dateslib.convert_to_date_tensor(schedule)
                self._start_date = coupon_dates[..., 0]
            elif schedule_fn is None:
                calendar = dateslib.create_holiday_calendar(weekend_mask=dateslib.WeekendMask.SATURDAY_SUNDAY)
                self._calendar = calendar
                coupon_dates = _generate_schedule(start_date=self._start_date, end_date=self._end_date, coupon_frequency=coupon_frequency, roll_convention=roll_convention, calendar=calendar, settlement_days=self._settlement_days, end_of_month=eom, first_coupon_date=self._first_coupon_date, penultimate_coupon_date=self._penultimate_coupon_date)
                self._start_date = coupon_dates[..., 0]
            else:
                if first_coupon_date is not None:
                    first_coupon_date = self._first_coupon_date.to_tensor()
                if penultimate_coupon_date is not None:
                    penultimate_coupon_date = self._penultimate_coupon_date.to_tensor()
                    coupon_dates = schedule_fn(start_date=self._start_date.to_tensor(), end_date=self._end_date.to_tensor(), coupon_frequency=coupon_frequency.quantity(), settlement_days=self._settlement_days, first_coupon_date=first_coupon_date, penultimate_coupon_date=penultimate_coupon_date)
            coupon_dates = dateslib.convert_to_date_tensor(coupon_dates)
            self._batch_shape = tf.shape(coupon_dates.ordinal())[:-1]
            accrual_start_dates = coupon_dates[..., :-1]
            coupon_start_dates = coupon_dates[..., :-1]
            coupon_end_dates = coupon_dates[..., 1:]
            accrual_end_dates = accrual_start_dates + reset_frequency.expand_dims(axis=-1)
            accrual_end_dates = dateslib.DateTensor.concat([coupon_end_dates[..., :1], accrual_end_dates[..., 1:-1], coupon_end_dates[..., -1:]], axis=-1)
            daycount_fractions = daycount_fn(start_date=coupon_start_dates, end_date=coupon_end_dates)
            self._num_cashflows = tf.shape(daycount_fractions)[-1]
            self._coupon_start_dates = coupon_start_dates
            self._coupon_end_dates = coupon_end_dates
            self._accrual_start_date = accrual_start_dates
            self._accrual_end_date = accrual_end_dates
            self._notional = notional
            self._daycount_fractions = daycount_fractions
            self._spread = spread
            self._currency = _get_attr(coupon_spec, 'currency')
            self._daycount_fn = daycount_fn
            self._floating_rate_type = to_list(_get_attr(coupon_spec, 'floating_rate_type'))
            self._currency = to_list(self._currency)
            if rate_index_curves is None:
                rate_index_curves = []
                for (currency, floating_rate_type) in zip(self._currency, self._floating_rate_type):
                    rate_index_curves.append(curve_types_lib.RateIndexCurve(currency=currency, index=floating_rate_type))
            [self._reference_curve_type, self._reference_mask] = process_curve_types(rate_index_curves, reference_mask)
            self._past_fixing = past_fixing

    def daycount_fn(self) -> Callable[..., Any]:
        if False:
            for i in range(10):
                print('nop')
        return self._daycount_fn

    @property
    def notional(self) -> types.FloatTensor:
        if False:
            while True:
                i = 10
        return self._notional

    @property
    def discount_curve_type(self) -> _CurveType:
        if False:
            i = 10
            return i + 15
        return self._discount_curve_type

    @property
    def reference_curve_type(self) -> _CurveType:
        if False:
            return 10
        return self._reference_curve_type

    @property
    def batch_shape(self) -> types.StringTensor:
        if False:
            i = 10
            return i + 15
        return self._batch_shape

    @property
    def daycount_fractions(self) -> types.FloatTensor:
        if False:
            i = 10
            return i + 15
        return self._daycount_fractions

    @property
    def cashflow_dates(self) -> types.DateTensor:
        if False:
            while True:
                i = 10
        return self._coupon_end_dates

    @property
    def coupon_start_dates(self) -> types.DateTensor:
        if False:
            return 10
        return self._coupon_start_dates

    @property
    def coupon_end_dates(self) -> types.DateTensor:
        if False:
            print('Hello World!')
        return self._coupon_end_dates

    def forward_rates(self, market: pmd.ProcessedMarketData, past_fixing: Optional[types.FloatTensor]=None, name: Optional[str]=None) -> Tuple[types.DateTensor, types.FloatTensor]:
        if False:
            return 10
        "Returns forward rates for the floating leg.\n\n    Args:\n      market: An instance of `ProcessedMarketData`.\n      past_fixing: An optional `Tensor` of shape compatible with\n        `batch_shape + [1]`. Represents the fixings for the cashflows as\n        observed at `market.date`.\n      name: Python str. The name to give to the ops created by this function.\n        Default value: `None` which maps to 'forward_rates'.\n\n    Returns:\n      A tuple of two `Tensor`s of shape `batch_shape + [num_cashflows]`\n      containing the dates and the corresponding forward rates for each stream\n      based on the input market data.\n    "
        name = name or self._name + '_forward_rates'
        with tf.name_scope(name):
            reference_curve = get_discount_curve(self._reference_curve_type, market, self._reference_mask)
            valuation_date = dateslib.convert_to_date_tensor(market.date)
            coupon_start_date_ord = self._coupon_start_dates.ordinal()
            coupon_end_date_ord = self._coupon_end_dates.ordinal()
            valuation_date_ord = valuation_date.ordinal()
            batch_shape = tf.shape(coupon_start_date_ord)[:-1]
            valuation_date_ord += tf.expand_dims(tf.zeros(batch_shape, dtype=tf.int32), axis=-1)
            ind = tf.maximum(tf.searchsorted(coupon_start_date_ord, valuation_date_ord) - 1, 0)
            fixing_dates_ord = tf.gather(coupon_start_date_ord, ind, batch_dims=len(coupon_start_date_ord.shape) - 1)
            fixing_end_dates_ord = tf.gather(coupon_end_date_ord, ind, batch_dims=len(coupon_start_date_ord.shape) - 1)
            fixing_dates = dateslib.dates_from_ordinals(fixing_dates_ord)
            fixing_end_dates = dateslib.dates_from_ordinals(fixing_end_dates_ord)
            if past_fixing is None:
                past_fixing = _get_fixings(fixing_dates, fixing_end_dates, self._reference_curve_type, self._reference_mask, market)
            else:
                past_fixing = tf.convert_to_tensor(past_fixing, dtype=self._dtype, name='past_fixing')
            forward_rates = reference_curve.forward_rate(self._accrual_start_date, self._accrual_end_date, day_count_fraction=self._daycount_fractions)
            forward_rates = tf.where(self._daycount_fractions > 0.0, forward_rates, tf.zeros_like(forward_rates))
            forward_rates = tf.where(self._coupon_end_dates < valuation_date, tf.constant(0, dtype=self._dtype), tf.where(self._coupon_start_dates >= valuation_date, forward_rates, past_fixing))
            return (self._coupon_end_dates, forward_rates)

    def cashflows(self, market: pmd.ProcessedMarketData, past_fixing: Optional[types.FloatTensor]=None, name: Optional[str]=None) -> Tuple[types.DateTensor, types.FloatTensor]:
        if False:
            for i in range(10):
                print('nop')
        "Returns cashflows for the floating leg.\n\n    Args:\n      market: An instance of `ProcessedMarketData`.\n      past_fixing: An optional `Tensor` of shape compatible with\n        `batch_shape + [1]`. Represents the fixings for the cashflows as\n        observed at `market.date`.\n      name: Python str. The name to give to the ops created by this function.\n        Default value: `None` which maps to 'cashflows'.\n\n    Returns:\n      A tuple of two `Tensor`s of shape `batch_shape + [num_cashflows]` and\n      containing the dates and the corresponding cashflows price for each\n      stream based on the input market data.\n    "
        name = name or self._name + '_cashflows'
        with tf.name_scope(name):
            (_, forward_rates) = self.forward_rates(market, past_fixing=past_fixing)
            coupon_rate = forward_rates + tf.expand_dims(self._spread, axis=-1)
            notional = tf.expand_dims(self._notional, axis=-1)
            cashflows = notional * (self._daycount_fractions * coupon_rate)
            return (self._coupon_end_dates, cashflows)

    def price(self, market: pmd.ProcessedMarketData, name: Optional[str]=None) -> types.FloatTensor:
        if False:
            for i in range(10):
                print('nop')
        "Returns the present value of the stream on the valuation date.\n\n    Args:\n      market: An instance of `ProcessedMarketData`.\n      name: Python str. The name to give to the ops created by this function.\n        Default value: `None` which maps to 'price'.\n\n    Returns:\n      A `Tensor` of shape `batch_shape`  containing the modeled price of each\n      stream based on the input market data.\n    "
        name = name or self._name + '_price'
        with tf.name_scope(name):
            discount_curve = get_discount_curve(self._discount_curve_type, market, self._mask)
            discount_factors = discount_curve.discount_factor(self._coupon_end_dates)
            (_, cashflows) = self.cashflows(market, past_fixing=self._past_fixing)
            cashflow_pvs = cashflows * discount_factors
            return tf.math.reduce_sum(cashflow_pvs, axis=1)

def _generate_schedule(start_date: dateslib.DateTensor, end_date: dateslib.DateTensor, coupon_frequency: dateslib.PeriodTensor, calendar: dateslib.HolidayCalendar, roll_convention: dateslib.BusinessDayConvention, settlement_days: tf.Tensor, end_of_month: bool=False, first_coupon_date: Optional[dateslib.DateTensor]=None, penultimate_coupon_date: Optional[dateslib.DateTensor]=None) -> tf.Tensor:
    if False:
        i = 10
        return i + 15
    'Method to generate coupon dates.\n\n  Args:\n    start_date: Starting dates of schedule.\n    end_date: End dates of the schedule.\n    coupon_frequency: A `PeriodTensor` specifying the frequency of coupon\n      payments.\n    calendar: calendar: An instance of `BankHolidays`.\n    roll_convention: Business day roll convention of the schedule.\n    settlement_days: An integer `Tensor` with the shape compatible with\n      `start_date` and `end_date` specifying the number of settlement days.\n    end_of_month: Python `bool`. If `True`, shifts all dates in schedule to\n      the ends of corresponding months, if `start_date` or `end_date` (\n      depending on `backward`) is at the end of a month. The shift is applied\n      before applying `roll_convention`.\n    first_coupon_date: First day of the irregular coupon, if any.\n    penultimate_coupon_date: Penultimate day of the coupon, if any.\n\n  Returns:\n    A `DateTensor` containing the generated date schedule of shape\n    `batch_shape + [max_num_coupon_days]`, where `max_num_coupon_days` is the\n    number of coupon days for the longest living swap in the batch. The coupon\n    days for the rest of the swaps are padded with their final coupon day.\n  '
    if first_coupon_date is not None and penultimate_coupon_date is not None:
        raise ValueError('Only first or last coupon dates can be specified  for an irregular coupon.')
    start_date = first_coupon_date or start_date
    start_date = calendar.add_business_days(start_date, settlement_days, roll_convention=roll_convention)
    if penultimate_coupon_date is None:
        backward = False
    else:
        backward = True
        end_date = end_date or penultimate_coupon_date
    end_date = calendar.add_business_days(end_date, settlement_days, roll_convention=roll_convention)
    coupon_dates = dateslib.PeriodicSchedule(start_date=start_date, end_date=end_date, tenor=coupon_frequency, roll_convention=roll_convention, backward=backward, end_of_month=end_of_month).dates()
    coupon_dates = dateslib.DateTensor.concat([start_date.expand_dims(-1), coupon_dates, end_date.expand_dims(-1)], axis=-1)
    return coupon_dates

def get_discount_curve(discount_curve_types: List[Union[curve_types_lib.RiskFreeCurve, curve_types_lib.RateIndexCurve]], market: pmd.ProcessedMarketData, mask: List[int]) -> rate_curve.RateCurve:
    if False:
        i = 10
        return i + 15
    'Builds a batched discount curve.\n\n  Given a list of discount curve an integer mask, creates a discount curve\n  object to compute discount factors against the list of discount curves.\n\n  #### Example\n  ```none\n  curve_types = [RiskFreeCurve("USD"), RiskFreeCurve("AUD")]\n  # A mask to price a batch of 7 instruments with the corresponding discount\n  # curves ["USD", "AUD", "AUD", "AUD" "USD", "USD", "AUD"].\n  mask = [0, 1, 1, 1, 0, 0, 1]\n  market = MarketDataDict(...)\n  get_discount_curve(curve_types, market, mask)\n  # Returns a RateCurve object that can compute a discount factors for a\n  # batch of 7 dates.\n  ```\n\n  Args:\n    discount_curve_types: A list of curve types.\n    market: an instance of the processed market data.\n    mask: An integer mask.\n\n  Returns:\n    An instance of `RateCurve`.\n  '
    discount_curves = [market.yield_curve(curve_type) for curve_type in discount_curve_types]
    discounts = []
    dates = []
    interpolation_method = None
    interpolate_rates = None
    for curve in discount_curves:
        (discount, date) = curve.discount_factors_and_dates()
        discounts.append(discount)
        dates.append(date)
        interpolation_method = curve.interpolation_method
        interpolate_rates = curve.interpolate_rates
    all_discounts = tf.stack(pad.pad_tensors(discounts), axis=0)
    all_dates = pad.pad_date_tensors(dates)
    all_dates = dateslib.DateTensor.stack(dates, axis=0)
    prepare_discounts = tf.gather(all_discounts, mask)
    prepare_dates = dateslib.dates_from_ordinals(tf.gather(all_dates.ordinal(), mask))
    discount_curve = rate_curve.RateCurve(prepare_dates, prepare_discounts, market.date, interpolator=interpolation_method, interpolate_rates=interpolate_rates)
    return discount_curve

def _get_fixings(start_dates, end_dates, reference_curve_types, reference_mask, market):
    if False:
        return 10
    'Computes fixings for a list of reference curves.'
    num_curves = len(reference_curve_types)
    if num_curves > 1:
        split_indices = [tf.squeeze(tf.where(tf.equal(reference_mask, i)), -1) for i in range(num_curves)]
    else:
        split_indices = [0]
    fixings = []
    start_dates_ordinal = start_dates.ordinal()
    end_dates_ordinal = end_dates.ordinal()
    for (idx, reference_curve_type) in zip(split_indices, reference_curve_types):
        if num_curves > 1:
            start_date = dateslib.dates_from_ordinals(tf.gather(start_dates_ordinal, idx))
            end_date = dateslib.dates_from_ordinals(tf.gather(end_dates_ordinal, idx))
        else:
            start_date = start_dates
            end_date = end_dates
        (fixing, fixing_daycount) = market.fixings(start_date, reference_curve_type)
        if fixing_daycount is not None:
            fixing_daycount = market_data_utils.get_daycount_fn(fixing_daycount, dtype=market.dtype)
            year_fraction = fixing_daycount(start_date=start_date, end_date=end_date)
        else:
            year_fraction = 0.0
        fixings.append(fixing * year_fraction)
    fixings = pad.pad_tensors(fixings)
    all_indices = tf.concat(split_indices, axis=0)
    all_fixings = tf.concat(fixings, axis=0)
    if num_curves > 1:
        return tf.gather(all_fixings, tf.argsort(all_indices))
    else:
        return all_fixings

def process_curve_types(curve_types: List[Union[curve_types_lib.RiskFreeCurve, curve_types_lib.RateIndexCurve]], mask=None) -> Tuple[List[Union[curve_types_lib.RiskFreeCurve, curve_types_lib.RateIndexCurve]], List[int]]:
    if False:
        return 10
    'Extracts unique curves and computes an integer mask.\n\n  #### Example\n  ```python\n  curve_types = [RiskFreeCurve("USD"), RiskFreeCurve("AUD"),\n                 RiskFreeCurve("USD")]\n  process_curve_types(curve_types)\n  # Returns [RiskFreeCurve("AUD"), RiskFreeCurve("USD")], [1, 0, 1]\n  ```\n  Args:\n    curve_types: A list of either `RiskFreeCurve` or `RateIndexCurve`.\n    mask: An optional integer mask for the sorted curve type sequence. If\n      supplied, the function returns does not do anything and returns\n      `(curve_types, mask)`.\n\n  Returns:\n    A Tuple of `(curve_list, mask)` where  `curve_list` is  a list of unique\n    curves in `curve_types` and `mask` is a list of integers which is the\n    mask for `curve_types`.\n  '

    def _get_signature(curve):
        if False:
            return 10
        'Converts curve information to a string.'
        if isinstance(curve, curve_types_lib.RiskFreeCurve):
            return curve.currency.value
        elif isinstance(curve, curve_types_lib.RateIndexCurve):
            return curve.currency.value + '_' + curve.index.type.value + '_' + '_'.join(curve.index.source) + '_' + '_'.join(curve.index.name)
        else:
            raise ValueError(f'{type(curve)} is not supported.')
    curve_list = to_list(curve_types)
    if mask is not None:
        return (curve_list, mask)
    curve_hash = [_get_signature(curve_type) for curve_type in curve_list]
    hash_discount_map = {_get_signature(curve_type): curve_type for curve_type in curve_list}
    (mask, mask_map, num_unique_discounts) = create_mask(curve_hash)
    discount_curve_types = [hash_discount_map[mask_map[i]] for i in range(num_unique_discounts)]
    return (discount_curve_types, mask)

def create_mask(x):
    if False:
        for i in range(10):
            print('nop')
    'Given a list of object creates integer mask for unique values in the list.\n\n  Args:\n    x: 1-d numpy array.\n\n  Returns:\n    A tuple of three objects:\n      * A list of integers that is the mask for `x`,\n      * A dictionary map between  entries of `x` and the list\n      * The number of unique elements.\n  '
    unique = np.unique(x)
    num_unique_elems = len(unique)
    keys = range(num_unique_elems)
    d = dict(zip(unique, keys))
    mask_map = dict(zip(keys, unique))
    return ([d[el] for el in x], mask_map, num_unique_elems)

def to_list(x):
    if False:
        for i in range(10):
            print('nop')
    'Converts input to a list if necessary.'
    if isinstance(x, (list, tuple)):
        return x
    else:
        return [x]

def _get_attr(obj, key):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(obj, dict):
        return obj[key]
    else:
        return obj.__getattribute__(key)
__all__ = ['FixedCashflowStream', 'FloatingCashflowStream']