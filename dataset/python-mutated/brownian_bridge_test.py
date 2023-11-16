"""Tests for Brownian Bridge method."""
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util

@test_util.run_all_in_graph_and_eager_modes
class BrownianBridgeTest(parameterized.TestCase, tf.test.TestCase):
    """Tests for Brownian Bridge method."""

    @parameterized.named_parameters({'testcase_name': 'SinglePrecision', 'dtype': np.float32}, {'testcase_name': 'DoublePrecision', 'dtype': np.float64})
    def test_brownian_bridge_double(self, dtype):
        if False:
            while True:
                i = 10

        def brownian_bridge_numpy(x_start, x_end, upper_b, lower_b, variance, n_cutoff):
            if False:
                i = 10
                return i + 15

            def f(k):
                if False:
                    i = 10
                    return i + 15
                a = np.exp(-2 * k * (upper_b - lower_b) * (k * (upper_b - lower_b) + (x_end - x_start)) / variance)
                b = np.exp(-2 * (k * (upper_b - lower_b) + x_start - upper_b) * (k * (upper_b - lower_b) + (x_end - upper_b)) / variance)
                return a - b
            return np.sum([f(k) for k in range(-n_cutoff, n_cutoff + 1)], axis=0)
        x_start = np.asarray([[1.0, 1.1, 1.1], [1.05, 1.11, 1.11]], dtype=dtype)
        x_end = np.asarray([[2.0, 2.1, 2.8], [2.05, 2.11, 2.11]], dtype=dtype)
        variance = np.asarray([1.0, 1.0, 1.1], dtype=dtype)
        n_cutoff = 3
        upper_barrier = 3.0
        lower_barrier = 0.5
        np_values = brownian_bridge_numpy(x_start, x_end, upper_barrier, lower_barrier, variance, n_cutoff=n_cutoff)
        tff_values = self.evaluate(tff.black_scholes.brownian_bridge_double(x_start=x_start, x_end=x_end, variance=variance, dtype=dtype, upper_barrier=upper_barrier, lower_barrier=lower_barrier, n_cutoff=n_cutoff))
        self.assertEqual(tff_values.shape, np_values.shape)
        self.assertArrayNear(tff_values.flatten(), np_values.flatten(), 1e-07)

    @parameterized.named_parameters({'testcase_name': 'SinglePrecision', 'dtype': np.float32}, {'testcase_name': 'DoublePrecision', 'dtype': np.float64})
    def test_brownian_bridge_single(self, dtype):
        if False:
            print('Hello World!')

        def brownian_bridge_numpy(x_start, x_end, barrier, variance):
            if False:
                for i in range(10):
                    print('nop')
            return 1 - np.exp(-2 * (x_start - barrier) * (x_end - barrier) / variance)
        x_start = np.asarray([[1.0, 1.1, 1.1], [1.05, 1.11, 1.11]], dtype=dtype)
        x_end = np.asarray([[2.0, 2.1, 2.8], [2.05, 2.11, 2.11]], dtype=dtype)
        variance = np.asarray([1.0, 1.0, 1.1], dtype=dtype)
        barrier = 3.0
        np_values = brownian_bridge_numpy(x_start, x_end, barrier, variance)
        tff_values = self.evaluate(tff.black_scholes.brownian_bridge_single(x_start=x_start, x_end=x_end, variance=variance, dtype=dtype, barrier=barrier))
        self.assertEqual(tff_values.shape, np_values.shape)
        self.assertArrayNear(tff_values.flatten(), np_values.flatten(), 1e-07)
if __name__ == '__main__':
    tf.test.main()