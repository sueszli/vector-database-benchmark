"""Forward Rate Agreement."""
import tensorflow.compat.v2 as tf
from tf_quant_finance import datetime as dates
from tf_quant_finance.experimental.instruments import rates_common as rc

class ForwardRateAgreement:
    """Represents a batch of Forward Rate Agreements (FRA).

  An FRA is a contract for the period [T, T+tau] where the holder exchanges a
  fixed rate (agreed at the start of the contract) against a floating payment
  determined at time T based on the spot Libor rate for term `tau`. The
  cashflows are exchanged at the settlement time T_s, which is either equal to T
  or close to T. The FRA are structured so that the payments are made in T+tau
  dollars (ref [1]).

  The ForwardRateAgreement class can be used to create and price multiple FRAs
  simultaneously. However all FRAs within a FRA object must be priced using
  a common reference and discount curve.

  #### Example:
  The following example illustrates the construction of a FRA instrument and
  calculating its price.

  ```python
  import numpy as np
  import tensorflow as tf
  import tf_quant_finance as tff
  dates = tff.datetime

  dtype = np.float64
  notional = 1.
  settlement_date = dates.convert_to_date_tensor([(2021, 2, 8)])
  fixing_date = dates.convert_to_date_tensor([(2021, 2, 8)])
  valuation_date = dates.convert_to_date_tensor([(2020, 2, 8)])
  fixed_rate = 0.02
  rate_term = rate_term = dates.periods.months(3)

  fra = tff.experimental.instruments.ForwardRateAgreement(
        notional, settlement_date, fixing_date, fixed_rate,
        rate_term=rate_term, dtype=dtype)
  curve_dates = valuation_date + dates.periods.months([1, 2, 3, 12, 24, 60])
  reference_curve = tff.experimental.instruments.RateCurve(
      curve_dates,
      np.array([0.02, 0.025, 0.0275, 0.03, 0.035, 0.0325], dtype=dtype),
      dtype=dtype)
  market = tff.experimental.instruments.InterestRateMarket(
      reference_curve=reference_curve, discount_curve=reference_curve)

  price = fra.price(valuation_date, market)
  # Expected result: 0.00378275
  ```

  #### References:
  [1]: Leif B.G. Andersen and Vladimir V. Piterbarg. Interest Rate Modeling,
      Volume I: Foundations and Vanilla Models. Chapter 5. 2010.
  """

    def __init__(self, settlement_date, fixing_date, fixed_rate, notional=1.0, daycount_convention=None, rate_term=None, maturity_date=None, dtype=None, name=None):
        if False:
            i = 10
            return i + 15
        "Initialize the batch of FRA contracts.\n\n    Args:\n      settlement_date: A rank 1 `DateTensor` specifying the dates on which\n        cashflows are settled. The shape of the input correspond to the number\n        of instruments being created.\n      fixing_date: A rank 1 `DateTensor` specifying the dates on which forward\n        rate will be fixed. The shape of the inout should be the same as that of\n        `settlement_date`.\n      fixed_rate: A rank 1 `Tensor` of real dtype specifying the fixed rate\n        payment agreed at the initiation of the individual contracts. The shape\n        should be the same as that of `settlement_date`.\n      notional: A scalar or a rank 1 `Tensor` of real dtype specifying the\n        notional amount for each contract. When the notional is specified as a\n        scalar, it is assumed that all contracts have the same notional. If the\n        notional is in the form of a `Tensor`, then the shape must be the same\n        as `settlement_date`.\n        Default value: 1.0\n      daycount_convention: An optional `DayCountConvention` to determine\n        how cashflows are accrued for each contract. Daycount is assumed to be\n        the same for all contracts in a given batch.\n        Default value: None in which case the daycount convention will default\n        to DayCountConvention.ACTUAL_360 for all contracts.\n      rate_term: An optional rank 1 `PeriodTensor` specifying the term (or the\n        tenor) of the Libor rate that determines the floating cashflow. The\n        shape of the input should be the same as `settlement_date`.\n        Default value: `None` in which case the the forward rate is determined\n        for the period [settlement_date, maturity_date].\n      maturity_date: An optional rank 1 `DateTensor` specifying the maturity of\n        the underlying forward rate for each contract. This input is only used\n        if the input `rate_term` is `None`.\n        Default value: `None`\n      dtype: `tf.Dtype`. If supplied the dtype for the real variables or ops\n        either supplied to the FRA object or created by the FRA object.\n        Default value: None which maps to the default dtype inferred by\n        TensorFlow.\n      name: Python str. The name to give to the ops created by this class.\n        Default value: `None` which maps to 'forward_rate_agreement'.\n\n    Raises:\n      ValueError: If both `maturity_date` and `rate_term` are unspecified.\n    "
        self._name = name or 'forward_rate_agreement'
        if rate_term is None and maturity_date is None:
            raise ValueError('Error creating FRA. Either rate_term or maturity_date is required.')
        with tf.name_scope(self._name):
            self._dtype = dtype
            self._notional = tf.convert_to_tensor(notional, dtype=self._dtype)
            self._fixing_date = dates.convert_to_date_tensor(fixing_date)
            self._settlement_date = dates.convert_to_date_tensor(settlement_date)
            self._accrual_start_date = dates.convert_to_date_tensor(settlement_date)
            if rate_term is None:
                self._accrual_end_date = dates.convert_to_date_tensor(maturity_date)
            else:
                self._accrual_end_date = self._accrual_start_date + rate_term
            if daycount_convention is None:
                daycount_convention = rc.DayCountConvention.ACTUAL_360
            self._fixed_rate = tf.convert_to_tensor(fixed_rate, dtype=self._dtype, name='fixed_rate')
            self._daycount_convention = daycount_convention
            self._daycount_fraction = rc.get_daycount_fraction(self._accrual_start_date, self._accrual_end_date, self._daycount_convention, self._dtype)

    def price(self, valuation_date, market, model=None):
        if False:
            return 10
        'Returns the present value of the instrument on the valuation date.\n\n    Args:\n      valuation_date: A scalar `DateTensor` specifying the date on which\n        valuation is being desired.\n      market: A namedtuple of type `InterestRateMarket` which contains the\n        necessary information for pricing the FRA instrument.\n      model: Reserved for future use.\n\n    Returns:\n      A Rank 1 `Tensor` of real type containing the modeled price of each FRA\n      contract based on the input market data.\n    '
        del model, valuation_date
        reference_curve = market.reference_curve
        discount_curve = market.discount_curve
        fwd_rate = reference_curve.get_forward_rate(self._accrual_start_date, self._accrual_end_date, self._daycount_fraction)
        discount_at_settlement = discount_curve.get_discount_factor(self._settlement_date)
        return discount_at_settlement * self._notional * (fwd_rate - self._fixed_rate) * self._daycount_fraction / (1.0 + self._daycount_fraction * fwd_rate)