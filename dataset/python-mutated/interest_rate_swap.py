"""Interest rate swap."""
import tensorflow.compat.v2 as tf
from tf_quant_finance import datetime as dates
from tf_quant_finance.experimental.instruments import cashflow_stream as cs
from tf_quant_finance.experimental.instruments import rates_common as rc

class InterestRateSwap:
    """Represents a batch of Interest Rate Swaps (IRS).

  An Interest rate swap (IRS) is a contract between two counterparties for an
  exchange of a series of payments over a period of time. The payments are made
  periodically (for example quarterly or semi-annually) where the last payment
  is made at the maturity (or termination) of the contract. In the case of
  fixed-for-floating IRS, one counterparty pays a fixed rate while the other
  counterparty's payments are linked to a floating index, most commonly the
  LIBOR rate. On the other hand, in the case of interest rate basis swap, the
  payments of both counterparties are linked to a floating index. Typically, the
  floating rate is observed (or fixed) at the beginning of each period while the
  payments are made at the end of each period [1].

  For example, consider a vanilla swap with the starting date T_0 and maturity
  date T_n and equally spaced coupon payment dates T_1, T_2, ..., T_n such that

  T_0 < T_1 < T_2 < ... < T_n and dt_i = T_(i+1) - T_i    (A)

  The floating rate is fixed on T_0, T_1, ..., T_(n-1) and both the fixed and
  floating payments are made on T_1, T_2, ..., T_n (payment dates).

  The InterestRateSwap class can be used to create and price multiple IRS
  simultaneously. The class supports vanilla fixed-for-floating swaps as
  well as basis swaps. However all IRS within an IRS object must be priced using
  a common reference and discount curve.

  #### Example (non batch):
  The following example illustrates the construction of an IRS instrument and
  calculating its price.

  ```python
  import numpy as np
  import tensorflow as tf
  import tf_quant_finance as tff
  dates = tff.datetime
  instruments = tff.experimental.instruments

  dtype = np.float64
  start_date = dates.convert_to_date_tensor([(2020, 2, 8)])
  maturity_date = dates.convert_to_date_tensor([(2022, 2, 8)])
  valuation_date = dates.convert_to_date_tensor([(2020, 2, 8)])
  period_3m = dates.periods.months(3)
  period_6m = dates.periods.months(6)
  fix_spec = instruments.FixedCouponSpecs(
              coupon_frequency=period_6m, currency='usd',
              notional=1., coupon_rate=0.03134,
              daycount_convention=instruments.DayCountConvention.ACTUAL_365,
              businessday_rule=dates.BusinessDayConvention.NONE)

  flt_spec = instruments.FloatCouponSpecs(
              coupon_frequency=period_3m, reference_rate_term=period_3m,
              reset_frequency=period_3m, currency='usd', notional=1.,
              businessday_rule=dates.BusinessDayConvention.NONE,
              coupon_basis=0., coupon_multiplier=1.,
              daycount_convention=instruments.DayCountConvention.ACTUAL_365)

  swap = instruments.InterestRateSwap([(2020,2,2)], [(2023,2,2)], [fix_spec],
                                      [flt_spec], dtype=np.float64)

  curve_dates = valuation_date + dates.periods.years([1, 2, 3, 5, 7, 10, 30])
  reference_curve = instruments.RateCurve(
      curve_dates,
      np.array([
        0.02834814, 0.03077457, 0.03113739, 0.03130794, 0.03160892,
        0.03213901, 0.03257991
        ], dtype=dtype),
      valuation_date=valuation_date,
      dtype=dtype)
  market = instruments.InterestRateMarket(
      reference_curve=reference_curve, discount_curve=reference_curve)

  price = swap.price(valuation_date, market)
  # Expected result: 1e-7
  ```

  #### Example (batch):
  The following example illustrates the construction and pricing of IRS using
  batches.

  ```python
  import numpy as np
  import tensorflow as tf
  import tf_quant_finance as tff
  dates = tff.datetime
  instruments = tff.experimental.instruments

  dtype = np.float64
  notional = 1.0
  maturity_date = dates.convert_to_date_tensor([(2023, 2, 8), (2027, 2, 8)])
  start_date = dates.convert_to_date_tensor([(2020, 2, 8), (2020, 2, 8)])
  valuation_date = dates.convert_to_date_tensor([(2020, 2, 8)])

  period3m = dates.periods.months([3, 3])
  period6m = dates.periods.months([6, 6])
  fix_spec = instruments.FixedCouponSpecs(
      coupon_frequency=period6m, currency='usd',
      notional=notional,
      coupon_rate=[0.03134, 0.03181],
      daycount_convention=instruments.DayCountConvention.ACTUAL_365,
      businessday_rule=dates.BusinessDayConvention.NONE)
  flt_spec = instruments.FloatCouponSpecs(
      coupon_frequency=period3m, reference_rate_term=period3m,
      reset_frequency=period3m, currency='usd',
      notional=notional,
      businessday_rule=dates.BusinessDayConvention.NONE,
      coupon_basis=0.0, coupon_multiplier=1.0,
      daycount_convention=instruments.DayCountConvention.ACTUAL_365)

  swap = instruments.InterestRateSwap(start_date, maturity_date,
                                      fix_spec, flt_spec,
                                      dtype=dtype)
  curve_dates = valuation_date + dates.periods.years([1, 2, 3, 5, 7, 10, 30])
  reference_curve = instruments.RateCurve(
      curve_dates,
      np.array([
        0.02834814, 0.03077457, 0.03113739, 0.03130794, 0.03160892,
        0.03213901, 0.03257991
        ], dtype=dtype),
        valuation_date=valuation_date,
      dtype=dtype)
  market = instruments.InterestRateMarket(
      reference_curve=reference_curve, discount_curve=reference_curve)

  price = swap.price(valuation_date, market)
  # Expected result: [1.0e-7, 1.0e-7]
  ```

  #### References:
  [1]: Leif B.G. Andersen and Vladimir V. Piterbarg. Interest Rate Modeling,
      Volume I: Foundations and Vanilla Models. Chapter 5. 2010.
  """

    def __init__(self, start_date, maturity_date, pay_leg, receive_leg, holiday_calendar=None, dtype=None, name=None):
        if False:
            print('Hello World!')
        "Initialize a batch of IRS contracts.\n\n    Args:\n      start_date: A rank 1 `DateTensor` specifying the dates for the inception\n        (start of the accrual) of the swap contracts. The shape of the input\n        correspond to the number of instruments being created.\n      maturity_date: A rank 1 `DateTensor` specifying the maturity dates for\n        each contract. The shape of the input should be the same as that of\n        `start_date`.\n      pay_leg: A scalar or a list of either `FixedCouponSpecs` or\n        `FloatCouponSpecs` specifying the coupon payments for the payment leg\n        of the swap. If specified as a list then the length of the list should\n        be the same as the number of instruments being created. If specified as\n        a scalar, then the elements of the namedtuple must be of the same shape\n        as (or compatible to) the shape of `start_date`.\n      receive_leg: A scalar or a list of either `FixedCouponSpecs` or\n        `FloatCouponSpecs` specifying the coupon payments for the receiving leg\n        of the swap. If specified as a list then the length of the list should\n        be the same as the number of instruments being created. If specified as\n        a scalar, then the elements of the namedtuple must be of the same shape\n        as (or compatible with) the shape of `start_date`.\n      holiday_calendar: An instance of `dates.HolidayCalendar` to specify\n        weekends and holidays.\n        Default value: None in which case a holiday calendar would be created\n        with Saturday and Sunday being the holidays.\n      dtype: `tf.Dtype`. If supplied the dtype for the real variables or ops\n        either supplied to the IRS object or created by the IRS object.\n        Default value: None which maps to the default dtype inferred by\n        TensorFlow.\n      name: Python str. The name to give to the ops created by this class.\n        Default value: `None` which maps to 'interest_rate_swap'.\n    "
        self._name = name or 'interest_rate_swap'
        if holiday_calendar is None:
            holiday_calendar = dates.create_holiday_calendar(weekend_mask=dates.WeekendMask.SATURDAY_SUNDAY)
        with tf.name_scope(self._name):
            self._dtype = dtype
            self._start_date = dates.convert_to_date_tensor(start_date)
            self._maturity_date = dates.convert_to_date_tensor(maturity_date)
            self._holiday_calendar = holiday_calendar
            self._floating_leg = None
            self._fixed_leg = None
            self._pay_leg = self._setup_leg(pay_leg)
            self._receive_leg = self._setup_leg(receive_leg)
            self._is_payer = isinstance(self._pay_leg, cs.FixedCashflowStream)

    def price(self, valuation_date, market, model=None, pricing_context=None, name=None):
        if False:
            for i in range(10):
                print('nop')
        "Returns the present value of the instrument on the valuation date.\n\n    Args:\n      valuation_date: A scalar `DateTensor` specifying the date on which\n        valuation is being desired.\n      market: A namedtuple of type `InterestRateMarket` which contains the\n        necessary information for pricing the interest rate swap.\n      model: Reserved for future use.\n      pricing_context: Additional context relevant for pricing.\n      name: Python str. The name to give to the ops created by this function.\n        Default value: `None` which maps to 'price'.\n\n    Returns:\n      A Rank 1 `Tensor` of real type containing the modeled price of each IRS\n      contract based on the input market data.\n    "
        name = name or self._name + '_price'
        with tf.name_scope(name):
            valuation_date = dates.convert_to_date_tensor(valuation_date)
            pay_cf = self._pay_leg.price(valuation_date, market, model, pricing_context)
            receive_cf = self._receive_leg.price(valuation_date, market, model, pricing_context)
            return receive_cf - pay_cf

    def annuity(self, valuation_date, market, model=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns the annuity of each swap on the vauation date.'
        valuation_date = dates.convert_to_date_tensor(valuation_date)
        return self._annuity(valuation_date, market, model, True)

    def par_rate(self, valuation_date, market, model=None):
        if False:
            i = 10
            return i + 15
        'Returns the par swap rate for the swap.'
        valuation_date = dates.convert_to_date_tensor(valuation_date)
        swap_annuity = self._annuity(valuation_date, market, model, False)
        float_pv = self._floating_leg.price(valuation_date, market, model)
        return float_pv / swap_annuity

    @property
    def term(self):
        if False:
            for i in range(10):
                print('nop')
        return tf.cast(self._start_date.days_until(self._maturity_date), dtype=self._dtype) / 365.0

    @property
    def fixed_rate(self):
        if False:
            i = 10
            return i + 15
        return self._fixed_leg.fixed_rate

    @property
    def notional(self):
        if False:
            return 10
        return self._floating_leg.notional

    @property
    def is_payer(self):
        if False:
            for i in range(10):
                print('nop')
        return self._is_payer

    def _setup_leg(self, leg):
        if False:
            print('Hello World!')
        'Setup swap legs.'
        leg_instance = leg[0] if isinstance(leg, list) else leg
        if isinstance(leg_instance, rc.FixedCouponSpecs):
            leg_ = cs.FixedCashflowStream(self._start_date, self._maturity_date, leg, dtype=self._dtype)
            self._fixed_leg = leg_
        elif isinstance(leg_instance, rc.FloatCouponSpecs):
            leg_ = cs.FloatingCashflowStream(self._start_date, self._maturity_date, leg, dtype=self._dtype)
            self._floating_leg = leg_
        else:
            raise ValueError('Unreconized leg type.')
        return leg_

    def _annuity(self, valuation_date, market, model=None, unit_notional=True):
        if False:
            for i in range(10):
                print('nop')
        'Returns the annuity of each swap on the vauation date.'
        del valuation_date, model
        if unit_notional:
            notional = 1.0
        else:
            notional = self._fixed_leg.notional
        if self._fixed_leg is not None:
            discount_curve = market.discount_curve
            discount_factors = discount_curve.get_discount_factor(self._fixed_leg.payment_dates)
            return tf.math.segment_sum(notional * discount_factors * self._fixed_leg.daycount_fractions, self._fixed_leg.contract_index)
        else:
            return 0.0