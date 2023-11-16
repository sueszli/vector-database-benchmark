"""Forward Rate Agreement."""
from typing import Any, Optional, List, Dict, Union
import dataclasses
import tensorflow.compat.v2 as tf
from tf_quant_finance import datetime as dateslib
from tf_quant_finance.experimental.pricing_platform.framework.core import curve_types as curve_types_lib
from tf_quant_finance.experimental.pricing_platform.framework.core import instrument
from tf_quant_finance.experimental.pricing_platform.framework.core import processed_market_data as pmd
from tf_quant_finance.experimental.pricing_platform.framework.core import rate_indices
from tf_quant_finance.experimental.pricing_platform.framework.core import types
from tf_quant_finance.experimental.pricing_platform.framework.market_data import utils as market_data_utils
from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments import cashflow_streams
from tf_quant_finance.experimental.pricing_platform.framework.rate_instruments.forward_rate_agreement import proto_utils
from tf_quant_finance.experimental.pricing_platform.instrument_protos import forward_rate_agreement_pb2 as fra
from tf_quant_finance.experimental.pricing_platform.instrument_protos import period_pb2

@dataclasses.dataclass(frozen=True)
class ForwardRateAgreementConfig:
    discounting_curve: Dict[types.CurrencyProtoType, curve_types_lib.CurveType] = dataclasses.field(default_factory=dict)
    model: str = ''

class ForwardRateAgreement(instrument.Instrument):
    """Represents a batch of Forward Rate Agreements (FRA).

  An FRA is a contract for the period [T, T+tau] where the holder exchanges a
  fixed rate (agreed at the start of the contract) against a floating payment
  determined at time T based on the spot Libor rate for term `tau`. The
  cashflows are exchanged at the settlement time T_s, which is either equal to T
  or close to T. See, e.g., [1].

  The ForwardRateAgreement class can be used to create and price multiple FRAs
  simultaneously. However all FRAs within an FRA object must be priced using
  a common reference and discount curve.

  #### Example:
  The following example illustrates the construction of an FRA instrument and
  calculating its price.

  ```python
  RateIndex = instrument_protos.rate_indices.RateIndex

  fra = fra_pb2.ForwardRateAgreement(
      short_position=True,
      fixing_date=date_pb2.Date(year=2021, month=5, day=21),
      currency=Currency.USD(),
      fixed_rate=decimal_pb2.Decimal(nanos=31340000),
      notional_amount=decimal_pb2.Decimal(units=10000),
      daycount_convention=DayCountConventions.ACTUAL_360(),
      business_day_convention=BusinessDayConvention.MODIFIED_FOLLOWING(),
      floating_rate_term=fra_pb2.FloatingRateTerm(
          floating_rate_type=RateIndex(type="LIBOR_3M"),
          term = period_pb2.Period(type="MONTH", amount=3)),
      settlement_days=2)
  date = [[2021, 2, 8], [2022, 2, 8], [2023, 2, 8], [2025, 2, 8],
          [2027, 2, 8], [2030, 2, 8], [2050, 2, 8]]
  discount = [0.97197441, 0.94022746, 0.91074031, 0.85495089, 0.8013675,
              0.72494879, 0.37602059]
  market_data_dict = {
      "rates": {
          "USD": {
              "risk_free_curve": {
                  "dates": dates,
                  "discounts": discounts,
              },
              "LIBOR_3M": {
                  "dates": dates,
                  "discounts": discounts,
              }
          }
      },
      "reference_date": [(2020, 2, 8)],
  }
  market = market_data.MarketDataDict(market_data_dict)
  fra_portfolio = forward_rate_agreement.ForwardRateAgreement.from_protos([fra])
  fra_portfolio[0].price(market)
  # Expected result: [4.05463257]
  ```

  #### References:
  [1]: Leif B.G. Andersen and Vladimir V. Piterbarg. Interest Rate Modeling,
      Volume I: Foundations and Vanilla Models. Chapter 5. 2010.
  """

    def __init__(self, short_position: types.BoolTensor, currency: types.CurrencyProtoType, fixing_date: types.DateTensor, fixed_rate: types.FloatTensor, notional_amount: types.FloatTensor, daycount_convention: types.DayCountConventionsProtoType, business_day_convention: types.BusinessDayConventionProtoType, calendar: types.BankHolidaysProtoType, rate_term: period_pb2.Period, rate_index: rate_indices.RateIndex, settlement_days: Optional[types.IntTensor]=0, discount_curve_type: curve_types_lib.CurveType=None, discount_curve_mask: types.IntTensor=None, rate_index_curves: curve_types_lib.RateIndexCurve=None, reference_mask: types.IntTensor=None, config: Union[ForwardRateAgreementConfig, Dict[str, Any]]=None, batch_names: Optional[types.StringTensor]=None, dtype: Optional[types.Dtype]=None, name: Optional[str]=None):
        if False:
            i = 10
            return i + 15
        "Initializes the batch of FRA contracts.\n\n    Args:\n      short_position: Whether the contract holder lends or borrows the money.\n        Default value: `True` which means that the contract holder lends the\n        money at the fixed rate.\n      currency: The denominated currency.\n      fixing_date: A `DateTensor` specifying the dates on which forward\n        rate will be fixed.\n      fixed_rate: A `Tensor` of real dtype specifying the fixed rate\n        payment agreed at the initiation of the individual contracts. The shape\n        should be broadcastable with `fixed_rate`.\n      notional_amount: A `Tensor` of real dtype broadcastable with fixed_rate\n        specifying the notional amount for each contract. When the notional is\n        specified as a scalar, it is assumed that all contracts have the same\n        notional.\n      daycount_convention: A `DayCountConvention` to determine how cashflows\n        are accrued for each contract. Daycount is assumed to be the same for\n        all contracts in a given batch.\n      business_day_convention: A business count convention.\n      calendar: A calendar to specify the weekend mask and bank holidays.\n      rate_term: A tenor of the rate (usually Libor) that determines the\n        floating cashflow.\n      rate_index: A type of the floating leg. An instance of\n        `core.rate_indices.RateIndex`.\n      settlement_days: An integer `Tensor` of the shape broadcastable with the\n        shape of `fixing_date`.\n      discount_curve_type: An optional instance of `CurveType` or a list of\n        those. If supplied as a list and `discount_curve_mask` is not supplied,\n        the size of the list should be the same as the number of priced\n        instruments. Defines discount curves for the instruments.\n        Default value: `None`, meaning that discount curves are inferred\n        from `currency` and `config`.\n      discount_curve_mask: An optional integer `Tensor` of values ranging from\n        `0` to `len(discount_curve_type) - 1` and of shape `batch_shape`.\n        Identifies a mapping between `discount_curve_type` list and the\n        underlying instruments.\n        Default value: `None`.\n      rate_index_curves: An instance of `RateIndexCurve` or a list of those.\n        If supplied as a list and `reference_mask` is not supplid,\n        the size of the list should be the same as the number of priced\n        instruments. Defines the index curves for each instrument. If not\n        supplied, `coupon_spec.floating_rate_type` is used to identify the\n        curves.\n        Default value: `None`.\n      reference_mask: An optional integer `Tensor` of values ranging from\n        `0` to `len(rate_index_curves) - 1` and of shape `batch_shape`.\n        Identifies a mapping between `rate_index_curves` list and the underlying\n        instruments.\n        Default value: `None`.\n      config: Optional `ForwardRateAgreementConfig` or a dictionary.\n        If dictionary, then the keys should be the same as the field names of\n        `ForwardRateAgreementConfig`.\n      batch_names: A string `Tensor` of instrument names. Should be of shape\n        `batch_shape + [2]` specying name and instrument type. This is useful\n        when the `from_protos` method is used and the user needs to identify\n        which instruments got batched together.\n      dtype: `tf.Dtype` of the input and output real `Tensor`s.\n        Default value: `None` which maps to `float64`.\n      name: Python str. The name to give to the ops created by this class.\n        Default value: `None` which maps to 'forward_rate_agreement'.\n    "
        self._name = name or 'forward_rate_agreement'
        with tf.name_scope(self._name):
            if batch_names is not None:
                self._names = tf.convert_to_tensor(batch_names, name='batch_names')
            else:
                self._names = None
            self._dtype = dtype or tf.float64
            ones = tf.constant(1, dtype=self._dtype)
            self._short_position = tf.where(short_position, ones, -ones, name='short_position')
            self._notional_amount = tf.convert_to_tensor(notional_amount, dtype=self._dtype, name='notional_amount')
            self._fixed_rate = tf.convert_to_tensor(fixed_rate, dtype=self._dtype, name='fixed_rate')
            settlement_days = tf.convert_to_tensor(settlement_days)
            (roll_convention, eom) = market_data_utils.get_business_day_convention(business_day_convention)
            calendar = dateslib.create_holiday_calendar(weekend_mask=dateslib.WeekendMask.SATURDAY_SUNDAY)
            if isinstance(fixing_date, types.IntTensor):
                self._fixing_date = dateslib.dates_from_tensor(fixing_date)
            else:
                self._fixing_date = dateslib.convert_to_date_tensor(fixing_date)
            self._accrual_start_date = calendar.add_business_days(self._fixing_date, settlement_days, roll_convention=roll_convention)
            self._day_count_fn = market_data_utils.get_daycount_fn(daycount_convention)
            period = rate_term
            if isinstance(rate_term, period_pb2.Period):
                period = market_data_utils.get_period(rate_term)
            if isinstance(rate_term, dict):
                period = market_data_utils.period_from_dict(rate_term)
            self._accrual_end_date = calendar.add_period_and_roll(self._accrual_start_date, period, roll_convention=roll_convention)
            if eom:
                self._accrual_end_date = self._accrual_end_date.to_end_of_month()
            self._daycount_fractions = self._day_count_fn(start_date=self._accrual_start_date, end_date=self._accrual_end_date, dtype=self._dtype)
            self._settlement_days = settlement_days
            self._roll_convention = roll_convention
            self._currency = cashflow_streams.to_list(currency)
            self._rate_index = cashflow_streams.to_list(rate_index)
            if rate_index_curves is None:
                rate_index_curves = []
                if len(self._currency) != len(self._rate_index):
                    raise ValueError('When rate_index_curves` is not supplied, number of currencies and rate indices should be the same `but it is {0} and {1}'.format(len(self._currency), len(self._rate_index)))
                for (currency, rate_index) in zip(self._currency, self._rate_index):
                    rate_index_curves.append(curve_types_lib.RateIndexCurve(currency=currency, index=rate_index))
            [self._reference_curve_type, self._reference_mask] = cashflow_streams.process_curve_types(rate_index_curves, reference_mask)
            self._config = _process_config(config)
            if discount_curve_type is None:
                curve_list = []
                for currency in self._currency:
                    if currency in self._config.discounting_curve:
                        discount_curve_type = self._config.discounting_curve[currency]
                    else:
                        discount_curve_type = curve_types_lib.RiskFreeCurve(currency=currency)
                    curve_list.append(discount_curve_type)
            else:
                curve_list = cashflow_streams.to_list(discount_curve_type)
            [self._discount_curve_type, self._mask] = cashflow_streams.process_curve_types(curve_list, discount_curve_mask)
            self._batch_shape = self._daycount_fractions.shape.as_list()[:-1]

    @classmethod
    def create_constructor_args(cls, proto_list: List[fra.ForwardRateAgreement], config: ForwardRateAgreementConfig=None) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        'Creates a dictionary to initialize ForwardRateAgreement.\n\n    The output dictionary is such that the instruments can be initialized\n    as follows:\n    ```\n    initializer = create_constructor_args(proto_list, config)\n    fras = [ForwardRateAgreement(**data) for data in initializer.values()]\n    ```\n\n    The keys of the output dictionary are unique identifiers of the batched\n    instruments. This is useful for identifying an existing graph that could be\n    reused for the instruments without the need of rebuilding the graph.\n\n    Args:\n      proto_list: A list of protos for which the initialization arguments are\n        constructed.\n      config: An instance of `ForwardRateAgreementConfig`.\n\n    Returns:\n      A possibly nested dictionary such that each value provides initialization\n      arguments for the ForwardRateAgreement.\n    '
        fra_data = proto_utils.from_protos_v2(proto_list, config)
        res = {}
        for key in fra_data:
            tensor_repr = proto_utils.tensor_repr(fra_data[key])
            res[key] = tensor_repr
        return res

    @classmethod
    def from_protos(cls, proto_list: List[fra.ForwardRateAgreement], config: ForwardRateAgreementConfig=None) -> List['ForwardRateAgreement']:
        if False:
            while True:
                i = 10
        proto_dict = proto_utils.from_protos_v2(proto_list, config)
        instruments = []
        for kwargs in proto_dict.values():
            kwargs['rate_term'] = market_data_utils.period_from_list(kwargs['rate_term'])
            instruments.append(cls(**kwargs))
        return instruments

    @classmethod
    def group_protos(cls, proto_list: List[fra.ForwardRateAgreement], config: ForwardRateAgreementConfig=None) -> Dict[str, List['ForwardRateAgreement']]:
        if False:
            return 10
        return proto_utils.group_protos_v2(proto_list, config)

    def price(self, market: pmd.ProcessedMarketData, name: Optional[str]=None) -> types.FloatTensor:
        if False:
            print('Hello World!')
        "Returns the present value of the stream on the valuation date.\n\n    Args:\n      market: An instance of `ProcessedMarketData`.\n      name: Python str. The name to give to the ops created by this function.\n        Default value: `None` which maps to 'price'.\n\n    Returns:\n      A `Tensor` of shape `batch_shape`  containing the modeled price of each\n      FRA contract based on the input market data.\n    "
        name = name or self._name + '_price'
        with tf.name_scope(name):
            discount_curve = cashflow_streams.get_discount_curve(self._discount_curve_type, market, self._mask)
            reference_curve = cashflow_streams.get_discount_curve(self._reference_curve_type, market, self._reference_mask)
            daycount_fractions = tf.expand_dims(self._daycount_fractions, axis=-1)
            fwd_rate = reference_curve.forward_rate(self._accrual_start_date.expand_dims(axis=-1), self._accrual_end_date.expand_dims(axis=-1), day_count_fraction=daycount_fractions)
            discount_at_settlement = discount_curve.discount_factor(self._accrual_start_date.expand_dims(axis=-1))
            discount_at_settlement = tf.where(daycount_fractions > 0.0, discount_at_settlement, tf.zeros_like(discount_at_settlement))
            discount_at_settlement = tf.squeeze(discount_at_settlement, axis=-1)
            fwd_rate = tf.squeeze(fwd_rate, axis=-1)
            return self._short_position * discount_at_settlement * self._notional_amount * (fwd_rate - self._fixed_rate) * self._daycount_fractions / (1.0 + self._daycount_fractions * fwd_rate)

    @property
    def batch_shape(self) -> tf.Tensor:
        if False:
            while True:
                i = 10
        return self._batch_shape

    @property
    def names(self) -> tf.Tensor:
        if False:
            print('Hello World!')
        'Returns a string tensor of names and instrument types.\n\n    The shape of the output is  [batch_shape, 2].\n    '
        return self._names

def _process_config(config: Union[ForwardRateAgreementConfig, Dict[str, Any], None]) -> ForwardRateAgreementConfig:
    if False:
        return 10
    'Converts config to ForwardRateAgreementConfig.'
    if config is None:
        return ForwardRateAgreementConfig()
    if isinstance(config, ForwardRateAgreementConfig):
        return config
    model = config.get('model', '')
    discounting_curve = config.get('discounting_curve', dict())
    return ForwardRateAgreementConfig(discounting_curve=discounting_curve, model=model)
__all__ = ['ForwardRateAgreementConfig', 'ForwardRateAgreement']