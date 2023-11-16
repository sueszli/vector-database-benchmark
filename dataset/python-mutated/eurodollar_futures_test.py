"""Tests for eurodollar_futures.py."""
import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util
dates = tff.datetime

@test_util.run_all_in_graph_and_eager_modes
class EurodollarFuturesTest(tf.test.TestCase):

    def test_edf_correctness(self):
        if False:
            i = 10
            return i + 15
        dtype = np.float64
        notional = 1.0
        expiry_date = tff.datetime.convert_to_date_tensor([(2021, 2, 8)])
        valuation_date = tff.datetime.convert_to_date_tensor([(2020, 2, 8)])
        rate_term = dates.periods.months(3)
        edfuture = tff.experimental.instruments.EurodollarFutures(expiry_date, contract_notional=notional, rate_term=rate_term, dtype=dtype)
        curve_dates = valuation_date + dates.periods.months([1, 2, 3, 12, 24, 60])
        reference_curve = tff.experimental.instruments.RateCurve(curve_dates, np.array([0.02, 0.025, 0.0275, 0.03, 0.035, 0.0325], dtype=dtype), valuation_date=valuation_date, dtype=dtype)
        market = tff.experimental.instruments.InterestRateMarket(reference_curve=reference_curve, discount_curve=None)
        price = self.evaluate(edfuture.price(valuation_date, market))
        np.testing.assert_allclose(price, 96.41051344, atol=1e-06)

    def test_edf_explicit_maturity(self):
        if False:
            for i in range(10):
                print('nop')
        dtype = np.float64
        notional = 1.0
        expiry_date = tff.datetime.convert_to_date_tensor([(2021, 2, 8)])
        valuation_date = tff.datetime.convert_to_date_tensor([(2020, 2, 8)])
        maturity_date = tff.datetime.convert_to_date_tensor([(2021, 5, 8)])
        edfuture = tff.experimental.instruments.EurodollarFutures(expiry_date, contract_notional=notional, maturity_date=maturity_date, dtype=dtype)
        curve_dates = valuation_date + dates.periods.months([1, 2, 3, 12, 24, 60])
        reference_curve = tff.experimental.instruments.RateCurve(curve_dates, np.array([0.02, 0.025, 0.0275, 0.03, 0.035, 0.0325], dtype=dtype), valuation_date=valuation_date, dtype=dtype)
        market = tff.experimental.instruments.InterestRateMarket(reference_curve=reference_curve, discount_curve=None)
        price = self.evaluate(edfuture.price(valuation_date, market))
        np.testing.assert_allclose(price, 96.41051344, atol=1e-06)

    def test_edf_many(self):
        if False:
            print('Hello World!')
        dtype = np.float64
        notional = 1.0
        expiry_date = tff.datetime.convert_to_date_tensor([(2021, 2, 8), (2021, 2, 8)])
        valuation_date = tff.datetime.convert_to_date_tensor([(2020, 2, 8)])
        maturity_date = tff.datetime.convert_to_date_tensor([(2021, 5, 8), (2021, 5, 8)])
        edfuture = tff.experimental.instruments.EurodollarFutures(expiry_date, contract_notional=notional, maturity_date=maturity_date, dtype=dtype)
        curve_dates = valuation_date + dates.periods.months([1, 2, 3, 12, 24, 60])
        reference_curve = tff.experimental.instruments.RateCurve(curve_dates, np.array([0.02, 0.025, 0.0275, 0.03, 0.035, 0.0325], dtype=dtype), valuation_date=valuation_date, dtype=dtype)
        market = tff.experimental.instruments.InterestRateMarket(reference_curve=reference_curve, discount_curve=None)
        price = self.evaluate(edfuture.price(valuation_date, market))
        np.testing.assert_allclose(price, [96.41051344, 96.41051344], atol=1e-06)
if __name__ == '__main__':
    tf.test.main()