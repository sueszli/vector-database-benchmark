"""Tests for bermudan swaptions."""
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util

class HullWhiteBermudanSwaptionTest(parameterized.TestCase, tf.test.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.mean_reversion_1d = 0.03
        self.volatility_1d = 0.01
        self.volatility_time_dep_1d = [0.01, 0.02]
        self.mean_reversion_2d = [0.03, 0.03]
        self.volatility_2d = [0.01, 0.015]
        self.exercise_swaption_1 = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        self.exercise_swaption_2 = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.0]
        self.exercise_times = [self.exercise_swaption_1, self.exercise_swaption_2]
        self.float_leg_start_times_1y = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        self.float_leg_start_times_18m = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        self.float_leg_start_times_2y = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.0]
        self.float_leg_start_times_30m = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.0, 5.0]
        self.float_leg_start_times_3y = [3.0, 3.5, 4.0, 4.5, 5.0, 5.0, 5.0, 5.0]
        self.float_leg_start_times_42m = [3.5, 4.0, 4.5, 5.0, 5.0, 5.0, 5.0, 5.0]
        self.float_leg_start_times_4y = [4.0, 4.5, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        self.float_leg_start_times_54m = [4.5, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        self.float_leg_start_times_5y = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        self.float_leg_start_times_swaption_1 = [self.float_leg_start_times_1y, self.float_leg_start_times_18m, self.float_leg_start_times_2y, self.float_leg_start_times_30m, self.float_leg_start_times_3y, self.float_leg_start_times_42m, self.float_leg_start_times_4y, self.float_leg_start_times_54m]
        self.float_leg_start_times_swaption_2 = [self.float_leg_start_times_2y, self.float_leg_start_times_30m, self.float_leg_start_times_3y, self.float_leg_start_times_42m, self.float_leg_start_times_4y, self.float_leg_start_times_54m, self.float_leg_start_times_5y, self.float_leg_start_times_5y]
        self.float_leg_start_times = [self.float_leg_start_times_swaption_1, self.float_leg_start_times_swaption_2]
        self.float_leg_end_times = np.clip(np.array(self.float_leg_start_times) + 0.5, 0.0, 5.0)
        self.fixed_leg_payment_times = self.float_leg_end_times
        self.float_leg_daycount_fractions = np.array(self.float_leg_end_times) - np.array(self.float_leg_start_times)
        self.fixed_leg_daycount_fractions = self.float_leg_daycount_fractions
        self.fixed_leg_coupon = 0.011 * np.ones_like(self.fixed_leg_payment_times)
        zero_rate_fn = lambda x: 0.01 * tf.ones_like(x)
        self.zero_rate_fn = zero_rate_fn
        super(HullWhiteBermudanSwaptionTest, self).setUp()

    @parameterized.named_parameters({'testcase_name': 'float64_lsm', 'dtype': tf.float64, 'use_fd': False, 'expected': 1.8892, 'tol': 0.01}, {'testcase_name': 'float64_fd', 'dtype': tf.float64, 'use_fd': True, 'expected': 1.8892, 'tol': 0.1})
    def test_correctness(self, dtype, use_fd, expected, tol):
        if False:
            while True:
                i = 10
        'Tests model with constant parameters in 1 dimension.'
        price = tff.models.hull_white.bermudan_swaption_price(exercise_times=self.exercise_times[0], floating_leg_start_times=self.float_leg_start_times[0], floating_leg_end_times=self.float_leg_end_times[0], fixed_leg_payment_times=self.fixed_leg_payment_times[0], floating_leg_daycount_fractions=self.float_leg_daycount_fractions[0], fixed_leg_daycount_fractions=self.fixed_leg_daycount_fractions[0], fixed_leg_coupon=self.fixed_leg_coupon[0], reference_rate_fn=self.zero_rate_fn, notional=100.0, mean_reversion=self.mean_reversion_1d, volatility=self.volatility_1d, num_samples=10000, time_step=0.1, random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC, seed=[0, 0], use_finite_difference=use_fd, time_step_finite_difference=0.05, num_grid_points_finite_difference=500, dtype=dtype)
        self.assertEqual(price.dtype, dtype)
        self.assertAllEqual(price.shape, [])
        price = self.evaluate(price)
        self.assertAllClose(price, expected, rtol=tol, atol=tol)

    @parameterized.named_parameters({'testcase_name': 'float64_lsm', 'dtype': tf.float64, 'use_fd': False, 'expected': [1.8892, 1.6633], 'tol': 0.005}, {'testcase_name': 'float64_fd', 'dtype': tf.float64, 'use_fd': True, 'expected': [1.8892, 1.6633], 'tol': 0.1})
    def test_correctness_batch(self, dtype, use_fd, expected, tol):
        if False:
            return 10
        'Tests model with constant parameters in 1 dimension.'
        price = tff.models.hull_white.bermudan_swaption_price(exercise_times=self.exercise_times, floating_leg_start_times=self.float_leg_start_times, floating_leg_end_times=self.float_leg_end_times, fixed_leg_payment_times=self.fixed_leg_payment_times, floating_leg_daycount_fractions=self.float_leg_daycount_fractions, fixed_leg_daycount_fractions=self.fixed_leg_daycount_fractions, fixed_leg_coupon=self.fixed_leg_coupon, reference_rate_fn=self.zero_rate_fn, notional=100.0, mean_reversion=self.mean_reversion_1d, volatility=self.volatility_1d, num_samples=50000, time_step=0.1, random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC, seed=[0, 0], use_finite_difference=use_fd, time_step_finite_difference=0.05, num_grid_points_finite_difference=500, dtype=dtype)
        self.assertEqual(price.dtype, dtype)
        self.assertAllEqual(price.shape, [2])
        price = self.evaluate(price)
        self.assertAllClose(price, expected, rtol=tol, atol=tol)

    @parameterized.named_parameters({'testcase_name': 'float64_lsm', 'dtype': tf.float64, 'use_fd': False}, {'testcase_name': 'float64_fd', 'dtype': tf.float64, 'use_fd': True})
    def test_correctness_with_european(self, dtype, use_fd):
        if False:
            i = 10
            return i + 15
        'Tests model with constant parameters in 1 dimension.'
        price_berm = tff.models.hull_white.bermudan_swaption_price(exercise_times=[self.exercise_times[0][0]], floating_leg_start_times=self.float_leg_start_times[0][0], floating_leg_end_times=self.float_leg_end_times[0][0], fixed_leg_payment_times=self.fixed_leg_payment_times[0][0], floating_leg_daycount_fractions=self.float_leg_daycount_fractions[0][0], fixed_leg_daycount_fractions=self.fixed_leg_daycount_fractions[0][0], fixed_leg_coupon=self.fixed_leg_coupon[0][0], reference_rate_fn=self.zero_rate_fn, notional=100.0, mean_reversion=self.mean_reversion_1d, volatility=self.volatility_1d, num_samples=50000, time_step=0.1, random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC, seed=[0, 0], use_finite_difference=use_fd, time_step_finite_difference=0.05, num_grid_points_finite_difference=500, dtype=dtype)
        price_euro = tff.models.hull_white.swaption_price(expiries=self.exercise_times[0][0], floating_leg_start_times=self.float_leg_start_times[0][0], floating_leg_end_times=self.float_leg_end_times[0][0], fixed_leg_payment_times=self.fixed_leg_payment_times[0][0], floating_leg_daycount_fractions=self.float_leg_daycount_fractions[0][0], fixed_leg_daycount_fractions=self.fixed_leg_daycount_fractions[0][0], fixed_leg_coupon=self.fixed_leg_coupon[0][0], reference_rate_fn=self.zero_rate_fn, notional=100.0, mean_reversion=self.mean_reversion_1d, volatility=self.volatility_1d, dtype=dtype)
        self.assertAllClose(self.evaluate(price_berm), self.evaluate(price_euro), rtol=0.001, atol=0.001)

    @parameterized.named_parameters({'testcase_name': 'float64_lsm', 'dtype': tf.float64, 'use_fd': False}, {'testcase_name': 'float64_fd', 'dtype': tf.float64, 'use_fd': True})
    def test_correctness_with_quantlib(self, dtype, use_fd):
        if False:
            for i in range(10):
                print('nop')
        'Tests model with constant parameters in 1 dimension.'
        exercise_times = [[0.498630137, 1.002739726], [1.002739726, 2.002739726]]
        float_leg_start_times = [[[0.498630137, 1.002739726, 1.498630137, 2.002739726, 2.498630137], [1.002739726, 1.498630137, 2.002739726, 2.498630137, 3.002739726]], [[1.002739726, 1.498630137, 2.002739726, 2.498630137, 3.002739726], [2.002739726, 2.498630137, 3.002739726, 3.002739726, 3.002739726]]]
        float_leg_end_times = [[[1.002739726, 1.498630137, 2.002739726, 2.498630137, 3.002739726], [1.498630137, 2.002739726, 2.498630137, 3.002739726, 3.002739726]], [[1.498630137, 2.002739726, 2.498630137, 3.002739726, 3.002739726], [2.498630137, 3.002739726, 3.002739726, 3.002739726, 3.002739726]]]
        fixed_leg_payment_times = float_leg_end_times
        float_leg_daycount_fractions = np.array(float_leg_end_times) - np.array(float_leg_start_times)
        fixed_leg_daycount_fractions = float_leg_daycount_fractions
        fixed_leg_coupon = 0.008 * np.ones_like(fixed_leg_payment_times)
        price_tff = tff.models.hull_white.bermudan_swaption_price(exercise_times=exercise_times, floating_leg_start_times=float_leg_start_times, floating_leg_end_times=float_leg_end_times, fixed_leg_payment_times=fixed_leg_payment_times, floating_leg_daycount_fractions=float_leg_daycount_fractions, fixed_leg_daycount_fractions=fixed_leg_daycount_fractions, fixed_leg_coupon=fixed_leg_coupon, reference_rate_fn=self.zero_rate_fn, notional=100.0, mean_reversion=self.mean_reversion_1d, volatility=self.volatility_1d, num_samples=10000, time_step=0.1, random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC, seed=[0, 0], use_finite_difference=use_fd, time_step_finite_difference=0.05, num_grid_points_finite_difference=500, dtype=dtype)
        price_ql = [1.1016533104878343, 1.0586289990592714]
        self.assertAllClose(self.evaluate(price_tff), price_ql, rtol=0.001, atol=0.001)
if __name__ == '__main__':
    tf.test.main()