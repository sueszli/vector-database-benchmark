"""Tests for implied_vol_approx."""
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util
bs = tff.black_scholes

@test_util.run_all_in_graph_and_eager_modes
class ApproxImpliedVolTest(parameterized.TestCase, tf.test.TestCase):
    """Tests for methods in implied_vol module."""

    def test_approx_implied_vol(self):
        if False:
            i = 10
            return i + 15
        'Basic test of the implied vol calculation.'
        np.random.seed(6589)
        n = 100
        dtypes = [np.float32, np.float64]
        for dtype in dtypes:
            volatilities = np.exp(np.random.randn(n) / 2)
            forwards = np.exp(np.random.randn(n))
            strikes = forwards * (1 + (np.random.rand(n) - 0.5) * 0.2)
            expiries = np.exp(np.random.randn(n))
            prices = self.evaluate(bs.option_price(volatilities=volatilities, strikes=strikes, expiries=expiries, forwards=forwards, dtype=dtype))
            implied_vols = self.evaluate(bs.implied_vol_approx(prices=prices, strikes=strikes, expiries=expiries, forwards=forwards, dtype=dtype))
            self.assertArrayNear(volatilities, implied_vols, 0.6)

    def test_approx_implied_vol_validate(self):
        if False:
            while True:
                i = 10
        "Test the Radiocic-Polya approx doesn't raise where it shouldn't."
        np.random.seed(6589)
        n = 100
        dtypes = [np.float32, np.float64]
        for dtype in dtypes:
            volatilities = np.exp(np.random.randn(n) / 2)
            forwards = np.exp(np.random.randn(n))
            strikes = forwards * (1 + (np.random.rand(n) - 0.5) * 0.2)
            expiries = np.exp(np.random.randn(n))
            prices = self.evaluate(bs.option_price(volatilities=volatilities, strikes=strikes, expiries=expiries, forwards=forwards, dtype=dtype))
            implied_vols = self.evaluate(bs.implied_vol_approx(prices=prices, strikes=strikes, expiries=expiries, forwards=forwards, validate_args=True, dtype=dtype))
            self.assertArrayNear(volatilities, implied_vols, 0.6)

    @parameterized.named_parameters(('forwards_positive', 1.0, -1.0, 1.0, 1.0, True, 'Forwards positive'), ('strikes_positive', 1.0, 1.0, -1.0, 1.0, True, 'Strikes positive'), ('call_lower', -1e-07, 1.0, 1.0, 1.0, True, 'Price lower bound'), ('call_upper', 1.0 + 1e-07, 1.0, 1.0, 1.0, True, 'Price upper bound'), ('put_lower', -1e-07, 1.0, 1.0, 1.0, False, 'Price lower bound'), ('put_upper', 1.0 + 1e-07, 1e-07, 1.0, 1.0, False, 'Price upper bound'))
    def test_approx_implied_vol_validate_raises(self, price, forward, strike, expiry, is_call_option, regex):
        if False:
            while True:
                i = 10
        'Test the Radiocic-Polya approximation raises appropriately.'
        dtypes = [np.float32, np.float64]
        for dtype in dtypes:
            prices = np.array([price]).astype(dtype)
            forwards = np.array([forward]).astype(dtype)
            strikes = np.array([strike]).astype(dtype)
            expiries = np.array([expiry]).astype(dtype)
            is_call_options = np.array([is_call_option])
            with self.assertRaisesRegex(tf.errors.InvalidArgumentError, regex):
                implied_vols = bs.implied_vol_approx(prices=prices, strikes=strikes, expiries=expiries, forwards=forwards, is_call_options=is_call_options, validate_args=True, dtype=dtype)
                self.evaluate(implied_vols)
if __name__ == '__main__':
    tf.test.main()