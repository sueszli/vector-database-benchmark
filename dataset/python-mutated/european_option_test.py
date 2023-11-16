"""Tests for Heston Price method."""
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util

@test_util.run_all_in_graph_and_eager_modes
class HestonPriceTest(parameterized.TestCase, tf.test.TestCase):
    """Tests for Heston Price method."""

    @parameterized.named_parameters({'testcase_name': 'DoublePrecision', 'dtype': np.float64}, {'testcase_name': 'SinglePrecision', 'dtype': np.float32})
    def test_docstring(self, dtype):
        if False:
            i = 10
            return i + 15
        prices = tff.models.heston.approximations.european_option_price(variances=0.11, strikes=102.0, expiries=1.2, forwards=100.0, is_call_options=True, mean_reversion=2.0, theta=0.5, volvol=0.15, rho=0.3, discount_factors=1.0, dtype=dtype)
        self.assertAllClose(prices, 24.822196, rtol=1e-05, atol=1e-05)

    @parameterized.named_parameters({'testcase_name': 'DoublePrecisionUseForwards', 'use_forwards': True, 'dtype': np.float64}, {'testcase_name': 'DoublePrecisionUseSpots', 'use_forwards': False, 'dtype': np.float64}, {'testcase_name': 'SinglePrecisionUseForwards', 'use_forwards': True, 'dtype': np.float32})
    def test_heston_price(self, dtype, use_forwards):
        if False:
            for i in range(10):
                print('nop')
        mean_reversion = np.array([0.1, 10.0], dtype=dtype)
        theta = np.array([0.1, 0.5], dtype=dtype)
        variances = np.array([0.1, 0.5], dtype=dtype)
        discount_factors = np.array([0.99, 0.98], dtype=dtype)
        expiries = np.array([1.0], dtype=dtype)
        forwards = np.array([10.0], dtype=dtype)
        if use_forwards:
            spots = None
        else:
            spots = forwards * discount_factors
            forwards = None
        volvol = np.array([1.0, 0.9], dtype=dtype)
        strikes = np.array([9.7, 10.0], dtype=dtype)
        rho = np.array([0.5, 0.1], dtype=dtype)
        tff_prices = self.evaluate(tff.models.heston.approximations.european_option_price(mean_reversion=mean_reversion, theta=theta, volvol=volvol, rho=rho, variances=variances, forwards=forwards, spots=spots, expiries=expiries, strikes=strikes, discount_factors=discount_factors, is_call_options=np.asarray([True, False], dtype=bool)))
        scipy_prices = [1.07475678, 2.708217]
        np.testing.assert_allclose(tff_prices, scipy_prices, rtol=1e-05, atol=1e-05)
if __name__ == '__main__':
    tf.test.main()