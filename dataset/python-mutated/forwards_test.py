"""Tests for rate forwards."""
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util

@test_util.run_all_in_graph_and_eager_modes
class ForwardRatesTest(tf.test.TestCase, parameterized.TestCase):

    @parameterized.named_parameters({'testcase_name': 'SinglePrecision', 'dtype': np.float32}, {'testcase_name': 'DoublePrecision', 'dtype': np.float64})
    def test_forward_rates(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        df_start_dates = [[0.95, 0.9, 0.75], [0.95, 0.99, 0.85]]
        df_end_dates = [[0.8, 0.6, 0.5], [0.8, 0.9, 0.5]]
        daycount_fractions = [[0.5, 1.0, 2], [0.6, 0.4, 4.0]]
        forward_rates = self.evaluate(tff.rates.analytics.forwards.forward_rates(df_start_dates, df_end_dates, daycount_fractions, dtype=dtype))
        expected_forward_rates = np.array([[0.375, 0.5, 0.25], [0.3125, 0.25, 0.175]], dtype=dtype)
        np.testing.assert_allclose(forward_rates, expected_forward_rates, atol=1e-06)

    @parameterized.named_parameters({'testcase_name': 'SinglePrecision', 'dtype': np.float32}, {'testcase_name': 'DoublePrecision', 'dtype': np.float64})
    def test_forward_rates_from_yields(self, dtype):
        if False:
            i = 10
            return i + 15
        groups = np.array([0, 0, 0, 1, 1, 1, 1])
        times = np.array([0.25, 0.5, 1.0, 0.25, 0.5, 1.0, 1.5], dtype=dtype)
        rates = np.array([0.04, 0.041, 0.044, 0.022, 0.025, 0.028, 0.036], dtype=dtype)
        forward_rates = self.evaluate(tff.rates.analytics.forwards.forward_rates_from_yields(rates, times, groups=groups, dtype=dtype))
        expected_forward_rates = np.array([0.04, 0.042, 0.047, 0.022, 0.028, 0.031, 0.052], dtype=dtype)
        np.testing.assert_allclose(forward_rates, expected_forward_rates, atol=1e-06)

    @parameterized.named_parameters({'testcase_name': 'SinglePrecision', 'dtype': np.float32}, {'testcase_name': 'DoublePrecision', 'dtype': np.float64})
    def test_forward_rates_from_yields_no_batches(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        times = np.array([0.25, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5], dtype=dtype)
        rates = np.array([0.04, 0.041, 0.044, 0.046, 0.046, 0.047, 0.05], dtype=dtype)
        forward_rates = self.evaluate(tff.rates.analytics.forwards.forward_rates_from_yields(rates, times, dtype=dtype))
        expected_forward_rates = np.array([0.04, 0.042, 0.047, 0.054, 0.046, 0.05, 0.062], dtype=dtype)
        np.testing.assert_allclose(forward_rates, expected_forward_rates, atol=1e-06)

    @parameterized.named_parameters({'testcase_name': 'SinglePrecision', 'dtype': np.float32}, {'testcase_name': 'DoublePrecision', 'dtype': np.float64})
    def test_yields_from_forwards(self, dtype):
        if False:
            print('Hello World!')
        groups = np.array([0, 0, 0, 1, 1, 1, 1])
        times = np.array([0.25, 0.5, 1.0, 0.25, 0.5, 1.0, 1.5], dtype=dtype)
        forward_rates = np.array([0.04, 0.042, 0.047, 0.022, 0.028, 0.031, 0.052], dtype=dtype)
        expected_rates = np.array([0.04, 0.041, 0.044, 0.022, 0.025, 0.028, 0.036], dtype=dtype)
        actual_rates = self.evaluate(tff.rates.analytics.forwards.yields_from_forward_rates(forward_rates, times, groups=groups, dtype=dtype))
        np.testing.assert_allclose(actual_rates, expected_rates, atol=1e-06)

    @parameterized.named_parameters({'testcase_name': 'SinglePrecision', 'dtype': np.float32}, {'testcase_name': 'DoublePrecision', 'dtype': np.float64})
    def test_yields_from_forward_rates_no_batches(self, dtype):
        if False:
            print('Hello World!')
        times = np.array([0.25, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5], dtype=dtype)
        forward_rates = np.array([0.04, 0.042, 0.047, 0.054, 0.046, 0.05, 0.062], dtype=dtype)
        expected_rates = np.array([0.04, 0.041, 0.044, 0.046, 0.046, 0.047, 0.05], dtype=dtype)
        actual_rates = self.evaluate(tff.rates.analytics.forwards.yields_from_forward_rates(forward_rates, times, dtype=dtype))
        np.testing.assert_allclose(actual_rates, expected_rates, atol=1e-06)
if __name__ == '__main__':
    tf.test.main()