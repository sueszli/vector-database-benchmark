"""Eurodollar futures contract."""
import tensorflow.compat.v2 as tf
from tf_quant_finance import datetime as dates
from tf_quant_finance.experimental.instruments import rates_common as rc

class EurodollarFutures:
    """Represents a collection of Eurodollar futures contracts.

  Interest rate futures are exchange traded futures contracts on Libor rates
  liquidly traded on exchanges such as Chicago Mercantile Exchange (CME) or
  London International Financial Futures and Options Exchange (LIFFE). Contracts
  on CME on a US Dollar spot Libor rate are called Eurodollar (ED) Futures.
  An ED future contract at maturity T settles at the price
  100 * (1 - F(T, T1, T2))
  where F(T, T1, T2) is the spot Libor rate at time T with start T1 and
  maturity T2 (ref [1]).

  The EurodollarFutures class can used to create and price multiple contracts
  simultaneously. However all contracts within an object must be priced using a
  common reference curve.

  #### Example:
  The following example illustrates the construction of an ED future instrument
  and calculating its price.

  ```python

  import numpy as np
  import tensorflow as tf
  import tf_quant_finance as tff

  dates = tff.datetime
  instruments = tff.experimental.instruments

  dtype = np.float64
  notional = 1.
  expiry_date = dates.convert_to_date_tensor([(2021, 2, 8)])
  valuation_date = dates.convert_to_date_tensor([(2020, 2, 8)])
  rate_term = dates.periods.months(3)

  edfuture = instruments.EurodollarFutures(
      expiry_date, notional, rate_term=rate_term, dtype=dtype)

  curve_dates = valuation_date + dates.periods.months([1, 2, 3, 12, 24, 60])
  reference_curve = instruments.RateCurve(
      curve_dates,
      np.array([0.02, 0.025, 0.0275, 0.03, 0.035, 0.0325], dtype=dtype),
      dtype=dtype)

  market = instruments.InterestRateMarket(reference_curve=reference_curve,
                                          discount_curve=None)

  price = self.evaluate(edfuture.price(valuation_date, market))

  #### References:
  [1]: Leif B.G. Andersen and Vladimir V. Piterbarg. Interest Rate Modeling,
      Volume I: Foundations and Vanilla Models. Chapter 5. 2010.
  """

    def __init__(self, expiry_date, contract_notional=1.0, daycount_convention=None, rate_term=None, maturity_date=None, dtype=None, name=None):
        if False:
            for i in range(10):
                print('nop')
        "Initialize the Eurodollar futures object.\n\n    Args:\n      expiry_date: A Rank 1 `DateTensor` specifying the dates on which the\n        futures contracts expire.\n      contract_notional: An optional scalar or Rank 1 `Tensor` of real dtype\n        specifying the unit (or size) for the contract. For example for\n        eurodollar futures traded on CME, the contract notional is $2500. If\n        `contract_notional` is entered as a scalar, it is assumed that the input\n        is the same for all of the contracts.\n        Default value: 1.0\n      daycount_convention: An optional `DayCountConvention` corresponding\n        to the day count convention for the underlying rate for each contract.\n        Daycount is assumed to be the same for all contracts in a given batch.\n        Default value: None in which case each the day count convention of\n        DayCountConvention.ACTUAL_360 is used for each contract.\n      rate_term: An optional Rank 1 `PeriodTensor` specifying the term (or\n        tenor) of the rate that determines the settlement of each contract.\n        Default value: `None` in which case the the rate is assumed to be for\n        the period [expiry_date, maturity_date].\n      maturity_date: An optional Rank 1 `DateTensor` specifying the maturity of\n        the underlying forward rate for each contract. This input should be\n        specified if the input `rate_term` is `None`. If both `maturity_date`\n        and `rate_term` are specified, an error is raised.\n        Default value: `None`\n      dtype: `tf.Dtype`. If supplied the dtype for the real variables or ops\n        either supplied to the EurodollarFuture object or created by the\n        EurodollarFuture object.\n        Default value: None which maps to the default dtype inferred by\n        TensorFlow.\n      name: Python str. The name to give to the ops created by this class.\n        Default value: `None` which maps to 'eurodollar_future'.\n\n    Raises:\n      ValueError: If both `maturity_date` and `rate_term` are unspecified or\n      if both `maturity_date` and `rate_term` are specified.\n    "
        self._name = name or 'eurodollar_futures'
        if (rate_term is None) == (maturity_date is None):
            msg = 'Error creating the EurodollarFutures contract. Either rate_term or maturity_date is required.'
            raise ValueError(msg)
        if rate_term is not None and maturity_date is not None:
            msg = 'Error creating the EurodollarFutures contract. Both rate_term or maturity_date are specified.'
            raise ValueError(msg)
        with tf.name_scope(self._name):
            self._dtype = dtype
            self._contract_notional = tf.convert_to_tensor(contract_notional, dtype=self._dtype)
            self._expiry_date = dates.convert_to_date_tensor(expiry_date)
            self._accrual_start_date = self._expiry_date
            if rate_term is None:
                self._accrual_end_date = dates.convert_to_date_tensor(maturity_date)
            else:
                self._accrual_end_date = self._accrual_start_date + rate_term
            if daycount_convention is None:
                daycount_convention = rc.DayCountConvention.ACTUAL_360
            self._daycount_convention = daycount_convention
            self._daycount_fraction = rc.get_daycount_fraction(self._accrual_start_date, self._accrual_end_date, self._daycount_convention, self._dtype)

    def price(self, valuation_date, market, model=None):
        if False:
            print('Hello World!')
        'Returns the price of the contract on the valuation date.\n\n    Args:\n      valuation_date: A scalar `DateTensor` specifying the date on which\n        valuation is being desired.\n      market: A namedtuple of type `InterestRateMarket` which contains the\n        necessary information for pricing the FRA instrument.\n      model: Reserved for future use.\n\n    Returns:\n      A Rank 1 `Tensor` of real type containing the modeled price of each\n      futures contract based on the input market data.\n    '
        del model, valuation_date
        reference_curve = market.reference_curve
        fwd_rate = reference_curve.get_forward_rate(self._accrual_start_date, self._accrual_end_date, self._daycount_fraction)
        return 100.0 * self._contract_notional * (1.0 - fwd_rate)