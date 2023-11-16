"""Constant maturity swaps."""
import itertools
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tf_quant_finance import black_scholes
from tf_quant_finance import datetime as dates
from tf_quant_finance.experimental.instruments import cashflow_stream as cs
from tf_quant_finance.experimental.instruments import interest_rate_swap as irs
from tf_quant_finance.experimental.instruments import rates_common as rc
from tf_quant_finance.math import integration

class CMSCashflowStream(cs.CashflowStream):
    """Represents a batch of cashflows indexed to a CMS rate.

  #### Example:
  ```python
  import numpy as np
  import tensorflow as tf
  import tf_quant_finance as tff
  dates = tff.datetime
  instruments = tff.experimental.instruments

  start_date = dates.convert_to_date_tensor([(2021, 1, 1)])
  maturity_date = dates.convert_to_date_tensor([(2023, 1, 1)])
  valuation_date = dates.convert_to_date_tensor([(2021, 1, 1)])
  p3m = dates.periods.months(3)
  p6m = dates.periods.months(6)
  p1y = dates.periods.year()
  fix_spec = instruments.FixedCouponSpecs(
      coupon_frequency=p6m,
      currency='usd',
      notional=1.,
      coupon_rate=0.0,  # Not needed
      daycount_convention=instruments.DayCountConvention.ACTUAL_365,
      businessday_rule=dates.BusinessDayConvention.NONE)
  flt_spec = instruments.FloatCouponSpecs(
      coupon_frequency=p3m,
      reference_rate_term=p3m,
      reset_frequency=p3m,
      currency='usd',
      notional=1.,
      businessday_rule=dates.BusinessDayConvention.NONE,
      coupon_basis=0.,
      coupon_multiplier=1.,
      daycount_convention=instruments.DayCountConvention.ACTUAL_365)
  cms_spec = instruments.CMSCouponSpecs(
      coupon_frequency=p3m,
      tenor=p1y,
      float_leg=flt_spec,
      fixed_leg=fix_spec,
      notional=1.,
      coupon_basis=0.,
      coupon_multiplier=1.,
      businessday_rule=None,
      daycount_convention=instruments.DayCountConvention.ACTUAL_365)

  cms = instruments.CMSCashflowStream(
      start_date, maturity_date, [cms_spec], dtype=dtype)

  curve_dates = valuation_date + dates.periods.years([0, 1, 2, 3, 5])
  reference_curve = instruments.RateCurve(
      curve_dates,
      np.array([
          0.02, 0.02, 0.025, 0.03, 0.035
      ], dtype=np.float64),
      valuation_date=valuation_date,
      dtype=np.float64)
  market = instruments.InterestRateMarket(
      reference_curve=reference_curve, discount_curve=reference_curve)

  price = cms.price(valuation_date, market)
  # Expected result: 55512.6295434207
  ```
  """

    def __init__(self, start_date, end_date, coupon_spec, dtype=None, name=None):
        if False:
            for i in range(10):
                print('nop')
        "Initialize a batch of CMS cashflow streams.\n\n    Args:\n      start_date: A rank 1 `DateTensor` specifying the starting dates of the\n        accrual of the first coupon of the cashflow stream. The shape of the\n        input correspond to the numbercof streams being created.\n      end_date: A rank 1 `DateTensor` specifying the end dates for accrual of\n        the last coupon in each cashflow stream. The shape of the input should\n        be the same as that of `start_date`.\n      coupon_spec: A list of `CMSCouponSpecs` specifying the details of the\n        coupon payment for the cashflow stream. The length of the list should\n        be the same as the number of streams being created. Each coupon within\n        the list must have the same daycount_convention and businessday_rule.\n      dtype: `tf.Dtype`. If supplied the dtype for the real variables or ops\n        either supplied to the FloatingCashflowStream object or created by the\n        object.\n        Default value: None which maps to the default dtype inferred by\n        TensorFlow.\n      name: Python str. The name to give to the ops created by this class.\n        Default value: `None` which maps to 'floating_cashflow_stream'.\n    "
        super(CMSCashflowStream, self).__init__()
        self._name = name or 'cms_cashflow_stream'
        with tf.name_scope(self._name):
            self._start_date = dates.convert_to_date_tensor(start_date)
            self._end_date = dates.convert_to_date_tensor(end_date)
            self._batch_size = self._start_date.shape[0]
            self._first_coupon_date = None
            self._penultimate_coupon_date = None
            self._dtype = dtype
            self._setup(coupon_spec)

    def price(self, valuation_date, market, model=None, pricing_context=None, name=None):
        if False:
            return 10
        "Returns the present value of the stream on the valuation date.\n\n    Args:\n      valuation_date: A scalar `DateTensor` specifying the date on which\n        valuation is being desired.\n      market: A namedtuple of type `InterestRateMarket` which contains the\n        necessary information for pricing the cashflow stream.\n      model: An optional input of type `InterestRateModelType` to specify which\n        model to use for pricing.\n        Default value: `None` in which case `NORMAL_RATE` model is used.\n      pricing_context: An optional input to provide additional parameters (such\n        as model parameters) relevant for pricing.\n      name: Python str. The name to give to the ops created by this function.\n        Default value: `None` which maps to 'price'.\n\n    Returns:\n      A Rank 1 `Tensor` of real type containing the modeled price of each stream\n      contract based on the input market data.\n    "
        name = name or self._name + '_price'
        with tf.name_scope(name):
            valuation_date = dates.convert_to_date_tensor(valuation_date)
            discount_curve = market.discount_curve
            past_fixing = rc.get_rate_index(market, self._start_date, rc.RateIndexType.SWAP, dtype=self._dtype)
            past_fixing = tf.repeat(tf.convert_to_tensor(past_fixing, dtype=self._dtype), self._num_cashflows)
            discount_factors = discount_curve.get_discount_factor(self._payment_dates)
            cms_rates = self._swap.par_rate(valuation_date, market, model)
            cms_rates = tf.where(self._daycount_fractions > 0.0, cms_rates, tf.zeros_like(cms_rates))
            cms_rates = tf.where(self._coupon_end_dates < valuation_date, tf.constant(0.0, dtype=self._dtype), tf.where(self._coupon_start_dates < valuation_date, past_fixing, cms_rates))
            cms_rates = self._adjust_convexity(valuation_date, market, model, pricing_context, cms_rates, discount_factors)
            coupon_rate = self._coupon_multiplier * (cms_rates + self._coupon_basis)
            cashflow_pvs = self._notional * (self._daycount_fractions * coupon_rate * discount_factors)
            return tf.math.segment_sum(cashflow_pvs, self._contract_index)

    def _adjust_convexity(self, valuation_date, market, model, pricing_context, cms_rates, discount_factors):
        if False:
            return 10
        'Computes the convexity adjusted cms rate.'
        if model is None:
            return cms_rates
        elif model in (rc.InterestRateModelType.LOGNORMAL_SMILE_CONSISTENT_REPLICATION, rc.InterestRateModelType.NORMAL_SMILE_CONSISTENT_REPLICATION):
            return self._convexity_smile_replication(valuation_date, market, model, cms_rates, pricing_context)
        else:
            level = self._swap.annuity(valuation_date, market, None)
            expiry_time = dates.daycount_actual_365_fixed(start_date=valuation_date, end_date=self._coupon_start_dates, dtype=self._dtype)
            with tf.GradientTape() as g:
                g.watch(cms_rates)
                fx = self._fs(cms_rates)
            dfx = tf.squeeze(g.gradient(fx, cms_rates))
            swap_vol = tf.convert_to_tensor(pricing_context, dtype=self._dtype)
            if model == rc.InterestRateModelType.LOGNORMAL_RATE:
                cms_rates = cms_rates + dfx * level * cms_rates ** 2 * (tf.math.exp(swap_vol ** 2 * expiry_time) - 1.0) / discount_factors
            else:
                cms_rates = cms_rates + dfx * level * (swap_vol ** 2 * expiry_time) / discount_factors
            return cms_rates

    def _convexity_smile_replication(self, valuation_date, market, model, cms_rates, pricing_context):
        if False:
            print('Hello World!')
        'Calculate CMS convexity correction by static replication.'
        normal_model = model == rc.InterestRateModelType.NORMAL_SMILE_CONSISTENT_REPLICATION
        swap_vol = tf.convert_to_tensor(pricing_context, dtype=self._dtype)
        expiry_time = dates.daycount_actual_365_fixed(start_date=valuation_date, end_date=self._coupon_start_dates, dtype=self._dtype)
        lower = tf.zeros_like(cms_rates) + 1e-06
        rate_limit = 2000.0
        upper = rate_limit * cms_rates
        num_points = 10001

        def _call_replication():
            if False:
                for i in range(10):
                    print('nop')

            def _intfun_call(x):
                if False:
                    return 10
                d2fx = self._f_atm_second_derivative(x, cms_rates)
                forwards = tf.broadcast_to(tf.expand_dims(cms_rates, -1), x.shape)
                expiries = tf.broadcast_to(tf.expand_dims(expiry_time, -1), x.shape)
                option_val = _option_prices(volatilities=swap_vol, strikes=x, expiries=expiries, forwards=forwards, is_normal_model=normal_model, dtype=self._dtype)
                return d2fx * option_val
            intval_c = integration.integrate(_intfun_call, cms_rates, upper, num_points=num_points)
            dfk = self._f_atm_first_derivative(cms_rates, cms_rates)
            c_k = _option_prices(volatilities=swap_vol, strikes=cms_rates, expiries=expiry_time, forwards=cms_rates, is_normal_model=normal_model, dtype=self._dtype)
            return (1.0 + dfk) * c_k + intval_c

        def _put_replication():
            if False:
                for i in range(10):
                    print('nop')

            def _intfun_put(x):
                if False:
                    while True:
                        i = 10
                d2fx = self._f_atm_second_derivative(x, cms_rates)
                forwards = tf.broadcast_to(tf.expand_dims(cms_rates, -1), x.shape)
                expiries = tf.broadcast_to(tf.expand_dims(expiry_time, -1), x.shape)
                option_val = _option_prices(volatilities=swap_vol, strikes=x, expiries=expiries, forwards=forwards, is_call_options=False, is_normal_model=normal_model, dtype=self._dtype)
                return d2fx * option_val
            intval_p = integration.integrate(_intfun_put, lower, cms_rates, num_points=num_points)
            dfk = self._f_atm_first_derivative(cms_rates, cms_rates)
            p_k = _option_prices(volatilities=swap_vol, strikes=cms_rates, expiries=expiry_time, forwards=cms_rates, is_call_options=False, is_normal_model=normal_model, dtype=self._dtype)
            return (1.0 + dfk) * p_k - intval_p
        call_rep = _call_replication()
        put_rep = _put_replication()
        return cms_rates + (call_rep - put_rep)

    def _setup(self, coupon_spec):
        if False:
            for i in range(10):
                print('nop')
        'Setup tensors for efficient computations.'
        cpn_frequency = dates.PeriodTensor.stack([x.coupon_frequency for x in coupon_spec], axis=0)
        (cpn_dates, _) = self._generate_schedule(cpn_frequency, coupon_spec[-1].businessday_rule)
        cms_start_dates = cpn_dates[:, :-1]
        cms_term = dates.PeriodTensor.stack([x.tenor for x in coupon_spec], axis=0)
        cms_end_dates = cpn_dates[:, :-1] + cms_term.expand_dims(axis=-1).broadcast_to(cms_start_dates.shape)
        coupon_start_dates = cpn_dates[:, :-1]
        coupon_end_dates = cpn_dates[:, 1:]
        payment_dates = cpn_dates[:, 1:]
        daycount_fractions = rc.get_daycount_fraction(coupon_start_dates, coupon_end_dates, coupon_spec[-1].daycount_convention, dtype=self._dtype)
        notional = tf.repeat(tf.convert_to_tensor([x.notional for x in coupon_spec], dtype=self._dtype), payment_dates.shape.as_list()[-1])
        coupon_basis = tf.repeat(tf.convert_to_tensor([x.coupon_basis for x in coupon_spec], dtype=self._dtype), payment_dates.shape.as_list()[-1])
        coupon_multiplier = tf.repeat(tf.convert_to_tensor([x.coupon_multiplier for x in coupon_spec], dtype=self._dtype), payment_dates.shape.as_list()[-1])
        contract_index = tf.repeat(tf.range(0, len(coupon_spec)), payment_dates.shape.as_list()[-1])
        cms_fixed_leg = [x.fixed_leg for x in coupon_spec]
        cms_float_leg = [x.float_leg for x in coupon_spec]
        self._num_cashflows = daycount_fractions.shape.as_list()[-1]
        self._swap = irs.InterestRateSwap(cms_start_dates.reshape([-1]), cms_end_dates.reshape([-1]), list(itertools.chain.from_iterable((itertools.repeat(i, self._num_cashflows) for i in cms_fixed_leg))), list(itertools.chain.from_iterable((itertools.repeat(i, self._num_cashflows) for i in cms_float_leg))), dtype=self._dtype)
        self._coupon_start_dates = coupon_start_dates.reshape([-1])
        self._coupon_end_dates = coupon_end_dates.reshape([-1])
        self._payment_dates = payment_dates.reshape([-1])
        self._notional = notional
        self._daycount_fractions = tf.reshape(daycount_fractions, [-1])
        self._coupon_basis = coupon_basis
        self._coupon_multiplier = coupon_multiplier
        self._contract_index = contract_index

        def term_to_years(t):
            if False:
                for i in range(10):
                    print('nop')
            frac = tf.where(t.period_type() == dates.PeriodType.MONTH, tf.constant(1.0 / 12.0, dtype=self._dtype), tf.where(t.period_type() == dates.PeriodType.YEAR, tf.constant(1.0, dtype=self._dtype), tf.constant(0.0, dtype=self._dtype)))
            return frac * tf.cast(t.quantity(), dtype=self._dtype)
        cms_fixed_leg_frequency = dates.PeriodTensor.stack([x.fixed_leg.coupon_frequency for x in coupon_spec], axis=0)
        self._delta = term_to_years(cpn_frequency)
        self._tau = term_to_years(cms_fixed_leg_frequency)
        self._cms_periods = term_to_years(cms_term) / self._tau

    def _fs(self, s):
        if False:
            print('Hello World!')
        "Equation 2.13(a) from Hagen's paper."
        g = tf.where(s == 0.0, self._tau * self._cms_periods, 1 / s * (1.0 - 1.0 / (1.0 + self._tau * s) ** self._cms_periods))
        return 1.0 / (g * (1.0 + self._tau * s) ** (self._delta / self._tau))

    def _f_atm(self, s, cms_rates):
        if False:
            for i in range(10):
                print('nop')
        "Equation 2.19(b) from Hagan's paper."
        return (s - tf.expand_dims(cms_rates, -1)) * (self._fs(s) / self._fs(tf.expand_dims(cms_rates, -1)) - 1.0)

    def _f_atm_first_derivative(self, s, cms_rates):
        if False:
            return 10
        'Computes first order derivative of _f_atm.'
        with tf.GradientTape() as g:
            g.watch(s)
            fx = self._f_atm(s, cms_rates)
        dfx = tf.squeeze(g.gradient(fx, s))
        return dfx

    def _f_atm_second_derivative(self, s, cms_rates):
        if False:
            return 10
        'Computes second order derivative of _f_atm.'
        with tf.GradientTape() as g:
            g.watch(s)
            with tf.GradientTape() as gg:
                gg.watch(s)
                fx = self._f_atm(s, cms_rates)
            dfx = tf.squeeze(gg.gradient(fx, s))
        d2fx = tf.squeeze(g.gradient(dfx, s))
        return d2fx

class CMSSwap(irs.InterestRateSwap):
    """Represents a batch of CMS Swaps.

  A CMS swap is a swap contract where the floating leg payments are based on the
  constant maturity swap (CMS) rate. The CMS rate refers to a future fixing of
  swap rate of a fixed maturity, i.e. the breakeven swap rate on a standard
  fixed-to-float swap of the specified maturity [1].

  Let S_(i,m) denote a swap rate with maturity `m` (years) on the fixing date
  `T_i. Consider a CMS swap with the starting date T_0 and maturity date T_n
  and regularly spaced coupon payment dates T_1, T_2, ..., T_n such that

  T_0 < T_1 < T_2 < ... < T_n and dt_i = T_(i+1) - T_i    (A)

  The CMS rate, S_(i, m), is fixed on T_0, T_1, ..., T_(n-1) and floating
  payments made are on T_1, T_2, ..., T_n (payment dates) with the i-th payment
  being equal to tau_i * S_(i, m) where tau_i is the year fraction between
  [T_i, T_(i+1)].

  The CMSSwap class can be used to create and price multiple CMS swaps
  simultaneously. However all CMS swaps within an object must be priced using
  a common reference and discount curve.

  #### Example:
  The following example illustrates the construction of an CMS swap and
  calculating its price.

  ```python
  import numpy as np
  import tensorflow as tf
  import tf_quant_finance as tff
  dates = tff.datetime
  instruments = tff.experimental.instruments
  rc = tff.experimental.instruments.rates_common

  dtype = np.float64
  start_date = dates.convert_to_date_tensor([(2021, 1, 1)])
  maturity_date = dates.convert_to_date_tensor([(2023, 1, 1)])
  valuation_date = dates.convert_to_date_tensor([(2021, 1, 1)])
  p3m = dates.months(3)
  p6m = dates.months(6)
  p1y = dates.year()
  fix_spec = instruments.FixedCouponSpecs(
      coupon_frequency=p6m,
      currency='usd',
      notional=1.,
      coupon_rate=0.02,
      daycount_convention=instruments.DayCountConvention.ACTUAL_365,
      businessday_rule=dates.BusinessDayConvention.NONE)
  flt_spec = instruments.FloatCouponSpecs(
      coupon_frequency=p3m,
      reference_rate_term=p3m,
      reset_frequency=p3m,
      currency='usd',
      notional=1.,
      businessday_rule=dates.BusinessDayConvention.NONE,
      coupon_basis=0.,
      coupon_multiplier=1.,
      daycount_convention=instruments.DayCountConvention.ACTUAL_365)
  cms_spec = instruments.CMSCouponSpecs(
      coupon_frequency=p3m,
      tenor=p1y,
      float_leg=flt_spec,
      fixed_leg=fix_spec,
      notional=1.e6,
      coupon_basis=0.,
      coupon_multiplier=1.,
      businessday_rule=None,
      daycount_convention=instruments.DayCountConvention.ACTUAL_365)

  cms = instruments.CMSSwap(
      start_date, maturity_date, [fix_spec],
      [cms_spec], dtype=dtype)

  curve_dates = valuation_date + dates.years([0, 1, 2, 3, 5])
  reference_curve = instruments.RateCurve(
      curve_dates,
      np.array([
          0.02, 0.02, 0.025, 0.03, 0.035
      ], dtype=np.float64),
      valuation_date=valuation_date,
      dtype=np.float64)
  market = instruments.InterestRateMarket(
      reference_curve=reference_curve, discount_curve=reference_curve)

  price = cms.price(valuation_date, market)
  # Expected result: 16629.820479418966
  ```

  #### References:
  [1]: Leif B.G. Andersen and Vladimir V. Piterbarg. Interest Rate Modeling,
      Volume I: Foundations and Vanilla Models. Chapter 5. 2010.
  """

    def __init__(self, start_date, maturity_date, pay_leg, receive_leg, holiday_calendar=None, dtype=None, name=None):
        if False:
            return 10
        "Initialize a batch of CMS swap contracts.\n\n    Args:\n      start_date: A rank 1 `DateTensor` specifying the dates for the inception\n        (start of the accrual) of the swap cpntracts. The shape of the input\n        correspond to the numbercof instruments being created.\n      maturity_date: A rank 1 `DateTensor` specifying the maturity dates for\n        each contract. The shape of the input should be the same as that of\n        `start_date`.\n      pay_leg: A list of either `FixedCouponSpecs`, `FloatCouponSpecs` or\n        `CMSCouponSpecs` specifying the coupon payments for the payment leg of\n        the swap. The length of the list should be the same as the number of\n        instruments being created.\n      receive_leg: A list of either `FixedCouponSpecs` or `FloatCouponSpecs` or\n        `CMSCouponSpecs` specifying the coupon payments for the receiving leg\n        of the swap. The length of the list should be the same as the number of\n        instruments being created.\n      holiday_calendar: An instance of `dates.HolidayCalendar` to specify\n        weekends and holidays.\n        Default value: None in which case a holiday calendar would be created\n        with Saturday and Sunday being the holidays.\n      dtype: `tf.Dtype`. If supplied the dtype for the real variables or ops\n        either supplied to the IRS object or created by the IRS object.\n        Default value: None which maps to the default dtype inferred by\n        TensorFlow.\n      name: Python str. The name to give to the ops created by this class.\n        Default value: `None` which maps to 'cms_swap'.\n    "
        self._name = name or 'cms_swap'
        if holiday_calendar is None:
            holiday_calendar = dates.create_holiday_calendar(weekend_mask=dates.WeekendMask.SATURDAY_SUNDAY)
        with tf.name_scope(self._name):
            self._dtype = dtype
            self._start_date = dates.convert_to_date_tensor(start_date)
            self._maturity_date = dates.convert_to_date_tensor(maturity_date)
            self._holiday_calendar = holiday_calendar
            self._floating_leg = None
            self._fixed_leg = None
            self._cms_leg = None
            self._pay_leg = self._setup_leg(pay_leg)
            self._receive_leg = self._setup_leg(receive_leg)

    def price(self, valuation_date, market, model=None, pricing_context=None, name=None):
        if False:
            return 10
        "Returns the present value of the instrument on the valuation date.\n\n    Args:\n      valuation_date: A scalar `DateTensor` specifying the date on which\n        valuation is being desired.\n      market: A namedtuple of type `InterestRateMarket` which contains the\n        necessary information for pricing the interest rate swap.\n      model: An optional input of type `InterestRateModelType` to specify the\n        model to use for `convexity correction` while pricing individual\n        swaplets of the cms swap. When `model` is\n        `InterestRateModelType.LOGNORMAL_SMILE_CONSISTENT_REPLICATION` or\n        `InterestRateModelType.NORMAL_SMILE_CONSISTENT_REPLICATION`, the\n        function uses static replication (from lognormal and normal swaption\n        implied volatility data respectively) as described in [1]. When `model`\n        is `InterestRateModelType.LOGNORMAL_RATE` or\n        `InterestRateModelType.NORMAL_RATE`, the function uses analytic\n        approximations for the convexity adjustment based on lognormal and\n        normal swaption rate dyanmics respectively [1].\n        Default value: `None` in which case convexity correction is not used.\n      pricing_context: Additional context relevant for pricing.\n      name: Python str. The name to give to the ops created by this function.\n        Default value: `None` which maps to 'price'.\n\n    Returns:\n      A Rank 1 `Tensor` of real type containing the modeled price of each IRS\n      contract based on the input market data.\n\n    #### References:\n    [1]: Patrick S. Hagan. Convexity conundrums: Pricing cms swaps, caps and\n    floors. WILMOTT magazine.\n    "
        name = name or self._name + '_price'
        with tf.name_scope(name):
            return super(CMSSwap, self).price(valuation_date, market, model, pricing_context, name)

    def _setup_leg(self, leg):
        if False:
            i = 10
            return i + 15
        'Setup swap legs.'
        if isinstance(leg[0], rc.CMSCouponSpecs):
            new_leg = CMSCashflowStream(self._start_date, self._maturity_date, leg, dtype=self._dtype)
            self._cms_leg = new_leg
        else:
            new_leg = super(CMSSwap, self)._setup_leg(leg)
        return new_leg

def _option_prices(*, volatilities=None, strikes=None, forwards=None, expiries=None, is_call_options=True, is_normal_model=True, dtype=None):
    if False:
        while True:
            i = 10
    'Computes prices of European options using normal model.\n\n  Args:\n    volatilities: Real `Tensor` of any shape and dtype. The volatilities to\n      expiry of the options to price.\n    strikes: A real `Tensor` of the same dtype and compatible shape as\n      `volatilities`. The strikes of the options to be priced.\n    forwards: A real `Tensor` of any shape that broadcasts to the shape of\n      `volatilities`. The forwards to maturity. Either this argument or the\n    expiries: A real `Tensor` of same dtype and compatible shape as\n      `volatilities`. The expiry of each option. The units should be such that\n      `expiry * volatility**2` is dimensionless.\n    is_call_options: A boolean `Tensor` of a shape compatible with\n      `volatilities`. Indicates whether the option is a call (if True) or a put\n      (if False). If not supplied, call options are assumed.\n    is_normal_model: A boolean `Tensor` of a shape compatible with\n      `volatilities`. Indicates whether the options should be priced using\n      normal model (if True) or lognormal model (if False). If not supplied,\n      normal model is assumed.\n    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion\n      of any supplied non-`Tensor` arguments to `Tensor`.\n      Default value: `None` which maps to the default dtype inferred by\n        TensorFlow.\n\n  Returns:\n    Options prices computed using normal model for the underlying.\n  '
    dtype = dtype or tf.constant(0.0).dtype

    def _ncdf(x):
        if False:
            for i in range(10):
                print('nop')
        sqrt_2 = tf.math.sqrt(tf.constant(2.0, dtype=dtype))
        return (tf.math.erf(x / sqrt_2) + 1) / 2
    sqrt_var = tf.math.sqrt(expiries) * volatilities
    d = (forwards - strikes) / sqrt_var
    mu = tf.constant(0.0, dtype=dtype)
    loc = tf.constant(1.0, dtype=dtype)
    value = tf.where(is_normal_model, tf.where(is_call_options, (forwards - strikes) * _ncdf(d) + sqrt_var * tfp.distributions.Normal(mu, loc).prob(d), (strikes - forwards) * _ncdf(-d) + sqrt_var * tfp.distributions.Normal(mu, loc).prob(d)), black_scholes.option_price(volatilities=volatilities, strikes=strikes, expiries=expiries, forwards=forwards, is_call_options=is_call_options, dtype=dtype))
    value = tf.where(expiries > 0, value, tf.where(is_call_options, tf.maximum(forwards - strikes, 0.0), tf.maximum(strikes - forwards, 0.0)))
    return value