"""Tests for daycounts.py."""
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util
dateslib = tff.datetime

@test_util.run_all_in_graph_and_eager_modes
class DayCountsTest(tf.test.TestCase):

    def test_actual_360(self):
        if False:
            print('Hello World!')
        start_date = dateslib.dates_from_tuples([(2019, 12, 17)])
        end_date = dateslib.dates_from_tuples([(2021, 2, 11)])
        yf = self.evaluate(dateslib.daycount_actual_360(start_date=start_date, end_date=end_date))
        self.assertAllClose(yf, [1.17222], atol=1e-05)

    def test_actual_365_fixed(self):
        if False:
            for i in range(10):
                print('nop')
        start_date = dateslib.dates_from_tuples([(2019, 12, 17)])
        end_date = dateslib.dates_from_tuples([(2021, 2, 11)])
        yf = self.evaluate(dateslib.daycount_actual_365_fixed(start_date=start_date, end_date=end_date))
        self.assertAllClose(yf, [1.156164], atol=1e-05)

    def test_actual_365_actual_with_leap_day(self):
        if False:
            for i in range(10):
                print('nop')
        start_date = dateslib.dates_from_tuples([(2019, 12, 17)])
        end_date = dateslib.dates_from_tuples([(2021, 2, 11)])
        yf = self.evaluate(dateslib.daycount_actual_365_actual(start_date=start_date, end_date=end_date))
        self.assertAllClose(yf, [1.153005], atol=1e-05)

    def test_actual_365_actual_no_leap_day(self):
        if False:
            i = 10
            return i + 15
        start_date = dateslib.dates_from_tuples([(2020, 3, 17)])
        end_date = dateslib.dates_from_tuples([(2021, 2, 17)])
        yf = self.evaluate(dateslib.daycount_actual_365_actual(start_date=start_date, end_date=end_date))
        self.assertAllClose(yf, [0.923288], atol=1e-05)

    def test_actual_365_actual_with_leap_day_on_start(self):
        if False:
            return 10
        start_date = dateslib.dates_from_tuples([(2020, 2, 29)])
        end_date = dateslib.dates_from_tuples([(2021, 2, 11)])
        yf = self.evaluate(dateslib.daycount_actual_365_actual(start_date=start_date, end_date=end_date))
        self.assertAllClose(yf, [0.953424], atol=1e-05)

    def test_actual_365_actual_with_leap_day_on_end(self):
        if False:
            i = 10
            return i + 15
        start_date = dateslib.dates_from_tuples([(2019, 2, 11)])
        end_date = dateslib.dates_from_tuples([(2020, 2, 29)])
        yf = self.evaluate(dateslib.daycount_actual_365_actual(start_date=start_date, end_date=end_date))
        self.assertAllClose(yf, [1.046448], atol=1e-05)

    def test_thirty_360_isda_no_leap_year(self):
        if False:
            while True:
                i = 10
        'Test 30/360 ISDA on start / last dates without leap year.\n\n    The test cases and results are from\n    https://www.isda.org/2008/12/22/30-360-day-count-conventions\n    '
        start_date = dateslib.dates_from_tuples([(2007, 1, 15), (2007, 1, 15), (2007, 1, 15), (2007, 9, 30), (2007, 9, 30), (2007, 9, 30)])
        end_date = dateslib.dates_from_tuples([(2007, 1, 30), (2007, 2, 15), (2007, 7, 15), (2008, 3, 31), (2007, 10, 31), (2008, 9, 30)])
        yf = self.evaluate(dateslib.daycount_thirty_360_isda(start_date=start_date, end_date=end_date))
        self.assertAllClose(yf, [0.041666667, 0.083333333, 0.5, 0.5, 0.083333333, 1.0], atol=1e-05)

    def test_thirty_360_isda_with_leap_year_on_start(self):
        if False:
            i = 10
            return i + 15
        'Test 30/360 ISDA on start dates in leap year.\n\n    The test cases and results are from\n    https://www.isda.org/2008/12/22/30-360-day-count-conventions\n    '
        start_date = dateslib.dates_from_tuples([(2008, 2, 29), (2008, 2, 29), (2008, 2, 29)])
        end_date = dateslib.dates_from_tuples([(2009, 2, 28), (2008, 3, 30), (2008, 3, 31)])
        yf = self.evaluate(dateslib.daycount_thirty_360_isda(start_date=start_date, end_date=end_date))
        self.assertAllClose(yf, [0.997222222, 0.086111111, 0.088888889], atol=1e-05)

    def test_thirty_360_isda_with_leap_year_on_end(self):
        if False:
            while True:
                i = 10
        'Test 30/360 ISDA on last dates in leap year.\n\n    The test cases and results are from\n    https://www.isda.org/2008/12/22/30-360-day-count-conventions\n    '
        start_date = dateslib.dates_from_tuples([(2007, 2, 26), (2007, 8, 31)])
        end_date = dateslib.dates_from_tuples([(2008, 2, 29), (2008, 2, 29)])
        yf = self.evaluate(dateslib.daycount_thirty_360_isda(start_date=start_date, end_date=end_date))
        self.assertAllClose(yf, [1.008333333, 0.497222222], atol=1e-05)

    def test_actual_actual_isda(self):
        if False:
            print('Hello World!')
        'Test Actual/Actual ISDA day count convention.\n\n    The test cases are benchmarked against Actual/Actual daycount of QuantLib\n    '
        start_date = dateslib.dates_from_tuples([(2019, 9, 21), (2007, 1, 15), (2020, 6, 15), (2020, 12, 31)])
        end_date = dateslib.dates_from_tuples([(2020, 9, 21), (2007, 2, 15), (2020, 6, 16), (2023, 12, 31)])
        yf = self.evaluate(dateslib.daycount_actual_actual_isda(start_date=start_date, end_date=end_date, dtype=tf.float64))
        self.assertAllClose(yf, [1.0007635302043567, 0.08493150684931511, 0.0027322404371585285, 2.9999925144097612], atol=1e-14, rtol=1e-14)
if __name__ == '__main__':
    tf.test.main()