"""Tests for bond.py."""
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util
dates = tff.datetime
instruments = tff.experimental.instruments

@test_util.run_all_in_graph_and_eager_modes
class BondTest(tf.test.TestCase, parameterized.TestCase):

    @parameterized.named_parameters(('DoublePrecision', np.float64))
    def test_bond_correctness(self, dtype):
        if False:
            return 10
        settlement_date = dates.convert_to_date_tensor([(2014, 1, 15)])
        maturity_date = dates.convert_to_date_tensor([(2015, 1, 15)])
        valuation_date = dates.convert_to_date_tensor([(2014, 1, 15)])
        period_6m = dates.periods.months(6)
        fix_spec = instruments.FixedCouponSpecs(coupon_frequency=period_6m, currency='usd', notional=100.0, coupon_rate=0.06, daycount_convention=instruments.DayCountConvention.ACTUAL_365, businessday_rule=dates.BusinessDayConvention.NONE)
        bond_inst = instruments.Bond(settlement_date, maturity_date, [fix_spec], dtype=dtype)
        curve_dates = valuation_date + dates.periods.months([0, 6, 12])
        reference_curve = instruments.RateCurve(curve_dates, np.array([0.0, 0.005, 0.007], dtype=dtype), valuation_date=valuation_date, dtype=dtype)
        market = instruments.InterestRateMarket(discount_curve=reference_curve)
        price = self.evaluate(bond_inst.price(valuation_date, market))
        np.testing.assert_allclose(price, 105.27397754, atol=1e-06)

    @parameterized.named_parameters(('DoublePrecision', np.float64))
    def test_bond_many(self, dtype):
        if False:
            while True:
                i = 10
        settlement_date = dates.convert_to_date_tensor([(2014, 1, 15), (2014, 1, 15)])
        maturity_date = dates.convert_to_date_tensor([(2015, 1, 15), (2015, 1, 15)])
        valuation_date = dates.convert_to_date_tensor([(2014, 1, 15)])
        period_6m = dates.periods.months(6)
        fix_spec = instruments.FixedCouponSpecs(coupon_frequency=period_6m, currency='usd', notional=100.0, coupon_rate=0.06, daycount_convention=instruments.DayCountConvention.ACTUAL_365, businessday_rule=dates.BusinessDayConvention.NONE)
        bond_inst = instruments.Bond(settlement_date, maturity_date, [fix_spec, fix_spec], dtype=dtype)
        curve_dates = valuation_date + dates.periods.months([0, 6, 12])
        reference_curve = instruments.RateCurve(curve_dates, np.array([0.0, 0.005, 0.007], dtype=dtype), valuation_date=valuation_date, dtype=dtype)
        market = instruments.InterestRateMarket(discount_curve=reference_curve)
        price = self.evaluate(bond_inst.price(valuation_date, market))
        np.testing.assert_allclose(price, [105.27397754, 105.27397754], atol=1e-06)

    @parameterized.named_parameters(('DoublePrecision', np.float64))
    def test_bond_stub_begin(self, dtype):
        if False:
            while True:
                i = 10
        settlement_date = dates.convert_to_date_tensor([(2020, 1, 1)])
        maturity_date = dates.convert_to_date_tensor([(2021, 2, 1)])
        first_coupon_date = dates.convert_to_date_tensor([(2020, 2, 1)])
        valuation_date = dates.convert_to_date_tensor([(2020, 1, 1)])
        period_6m = dates.periods.months(6)
        fix_spec = instruments.FixedCouponSpecs(coupon_frequency=period_6m, currency='usd', notional=100.0, coupon_rate=0.06, daycount_convention=instruments.DayCountConvention.ACTUAL_365, businessday_rule=dates.BusinessDayConvention.NONE)
        bond_inst = instruments.Bond(settlement_date, maturity_date, [fix_spec], first_coupon_date=first_coupon_date, dtype=dtype)
        curve_dates = valuation_date + dates.periods.months([0, 6, 12, 24])
        reference_curve = instruments.RateCurve(curve_dates, np.array([0.0, 0.025, 0.03, 0.035], dtype=dtype), valuation_date=valuation_date, dtype=dtype)
        market = instruments.InterestRateMarket(discount_curve=reference_curve)
        price = self.evaluate(bond_inst.price(valuation_date, market))
        np.testing.assert_allclose(price, [103.12756228], atol=1e-06)
        expected_coupon_dates = dates.convert_to_date_tensor([(2020, 2, 1), (2020, 8, 1), (2021, 2, 1)])
        self.assertAllEqual(expected_coupon_dates.ordinal(), bond_inst._cashflows.payment_dates.ordinal())

    @parameterized.named_parameters(('DoublePrecision', np.float64))
    def test_bond_stub_end(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        settlement_date = dates.convert_to_date_tensor([(2020, 1, 1)])
        maturity_date = dates.convert_to_date_tensor([(2021, 2, 1)])
        last_coupon_date = dates.convert_to_date_tensor([(2021, 1, 1)])
        valuation_date = dates.convert_to_date_tensor([(2020, 1, 1)])
        period_6m = dates.periods.months(6)
        fix_spec = instruments.FixedCouponSpecs(coupon_frequency=period_6m, currency='usd', notional=100.0, coupon_rate=0.06, daycount_convention=instruments.DayCountConvention.ACTUAL_365, businessday_rule=dates.BusinessDayConvention.NONE)
        bond_inst = instruments.Bond(settlement_date, maturity_date, [fix_spec], penultimate_coupon_date=last_coupon_date, dtype=dtype)
        curve_dates = valuation_date + dates.periods.months([0, 6, 12, 24])
        reference_curve = instruments.RateCurve(curve_dates, np.array([0.0, 0.025, 0.03, 0.035], dtype=dtype), valuation_date=valuation_date, dtype=dtype)
        market = instruments.InterestRateMarket(discount_curve=reference_curve)
        price = self.evaluate(bond_inst.price(valuation_date, market))
        np.testing.assert_allclose(price, [103.12769595], atol=1e-06)
        expected_coupon_dates = dates.convert_to_date_tensor([(2020, 7, 1), (2021, 1, 1), (2021, 2, 1)])
        self.assertAllEqual(expected_coupon_dates.ordinal(), bond_inst._cashflows.payment_dates.ordinal())
if __name__ == '__main__':
    tf.test.main()