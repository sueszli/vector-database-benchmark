"""Tests for implied_volatility.newton_vol."""
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util
ImpliedVolUnderlyingDistribution = tff.black_scholes.ImpliedVolUnderlyingDistribution

@test_util.run_all_in_graph_and_eager_modes
class ImpliedVolNewtonTest(parameterized.TestCase, tf.test.TestCase):
    """Tests for methods in newton_vol module."""

    def test_basic_newton_finder(self):
        if False:
            print('Hello World!')
        'Tests the Newton root finder recovers the volatility on a few cases.'
        forwards = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        strikes = np.array([1.0, 2.0, 1.0, 0.5, 1.0, 1.0])
        expiries = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 2.0])
        discounts = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        init_vols = np.array([2.0, 0.5, 2.0, 0.5, 1.5, 1.5])
        is_call_options = np.array([True, True, False, False, True, True])
        volatilities = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        prices = np.array([0.38292492, 0.19061012, 0.38292492, 0.09530506, 0.27632639, 0.52049988])
        (implied_vols, converged, failed) = self.evaluate(tff.black_scholes.implied_vol_newton(prices=prices, strikes=strikes, expiries=expiries, forwards=forwards, discount_factors=discounts, is_call_options=is_call_options, initial_volatilities=init_vols, max_iterations=100))
        self.assertTrue(np.all(converged))
        self.assertFalse(np.any(failed))
        self.assertArrayNear(volatilities, implied_vols, 1e-07)

    def test_basic_radiocic_newton_combination_finder(self):
        if False:
            print('Hello World!')
        'Tests the Newton root finder recovers the volatility on a few cases.'
        forwards = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        strikes = np.array([1.0, 2.0, 1.0, 0.5, 1.0, 1.0])
        expiries = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 2.0])
        discounts = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        is_call_options = np.array([True, True, False, False, True, True])
        volatilities = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        prices = np.array([0.38292492, 0.19061012, 0.38292492, 0.09530506, 0.27632639, 0.52049988])
        (implied_vols, converged, failed) = self.evaluate(tff.black_scholes.implied_vol_newton(prices=prices, strikes=strikes, expiries=expiries, forwards=forwards, discount_factors=discounts, is_call_options=is_call_options))
        self.assertTrue(np.all(converged))
        self.assertFalse(np.any(failed))
        self.assertArrayNear(volatilities, implied_vols, 1e-07)

    def test_bachelier_positive_underlying(self):
        if False:
            return 10
        'Tests the Newton root finder recovers the volatility on Bachelier Model.\n\n    This are the cases with positive underlying and strike.\n    '
        forwards = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        strikes = np.array([1.0, 2.0, 1.0, 0.5, 1.0, 1.0])
        expiries = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 2.0])
        discounts = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        init_vols = np.array([2.0, 0.5, 2.0, 0.5, 1.5, 1.5])
        is_call_options = np.array([True, True, False, False, True, True])
        volatilities = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        prices = np.array([0.3989423, 0.0833155, 0.3989423, 0.1977966, 0.2820948, 0.5641896])
        (implied_vols, converged, failed) = self.evaluate(tff.black_scholes.implied_vol_newton(prices=prices, strikes=strikes, expiries=expiries, forwards=forwards, discount_factors=discounts, is_call_options=is_call_options, initial_volatilities=init_vols, underlying_distribution=ImpliedVolUnderlyingDistribution.NORMAL, max_iterations=100))
        self.assertTrue(np.all(converged))
        self.assertFalse(np.any(failed))
        self.assertArrayNear(volatilities, implied_vols, 1e-06)

    def test_bachelier_negative_underlying(self):
        if False:
            return 10
        'Tests the Newton root finder recovers the volatility on Bachelier Model.\n\n    These are the cases with negative underlying and strike.\n    '
        forwards = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        strikes = np.array([1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -0.5])
        expiries = np.array([1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 1.0])
        discounts = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        init_vols = np.array([2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        is_call_options = np.array([True, True, True, True, True, False, False])
        volatilities = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        prices = np.array([0.0084907, 0.0833155, 0.3989423, 0.5641896, 0.2820948, 0.3989423, 0.6977965])
        (implied_vols, converged, failed) = self.evaluate(tff.black_scholes.implied_vol_newton(prices=prices, strikes=strikes, expiries=expiries, forwards=forwards, discount_factors=discounts, is_call_options=is_call_options, initial_volatilities=init_vols, underlying_distribution=ImpliedVolUnderlyingDistribution.NORMAL, max_iterations=100))
        self.assertTrue(np.all(converged))
        self.assertFalse(np.any(failed))
        self.assertArrayNear(volatilities, implied_vols, 1e-06)

    def test_bachelier_at_the_money(self):
        if False:
            return 10
        'Tests the Newton root finder recovers the volatility on Bachelier Model.\n\n    These are the cases for at the money (forward = strike).\n    '
        forwards = np.array([1.0, 0.0, -1.0, 1.0, 0.0, -1.0])
        strikes = np.array([1.0, 0.0, -1.0, 1.0, 0.0, -1.0])
        expiries = np.array([1.0, 1.0, 2.0, 1.0, 1.0, 2.0])
        discounts = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        init_vols = np.array([2.0, 1.0, 1.0, 2.0, 1.0, 1.0])
        is_call_options = np.array([True, True, True, False, False, False])
        volatilities = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        prices = np.array([0.3989423, 0.3989423, 0.5641896, 0.3989423, 0.3989423, 0.5641896])
        (implied_vols, converged, failed) = self.evaluate(tff.black_scholes.implied_vol_newton(prices=prices, strikes=strikes, expiries=expiries, forwards=forwards, discount_factors=discounts, is_call_options=is_call_options, initial_volatilities=init_vols, underlying_distribution=ImpliedVolUnderlyingDistribution.NORMAL, max_iterations=100))
        print('converged', converged)
        print('failed', failed)
        print('implied_vols', implied_vols)
        self.assertTrue(np.all(converged))
        self.assertFalse(np.any(failed))
        self.assertArrayNear(volatilities, implied_vols, 1e-06)

    @parameterized.named_parameters(('call_lower', 0.0, 1.0, 1.0, 1.0, 1.0), ('call_upper', 1.0, 1.0, 1.0, 1.0, 1.0), ('put_lower', 1.0, 1.0, 1.0, 1.0, -1.0), ('put_upper', 0.0, 1.0, 1.0, 1.0, -1.0))
    def test_implied_vol_validate_raises(self, price, forward, strike, expiry, option_sign):
        if False:
            while True:
                i = 10
        'Tests validation errors raised where BS model assumptions violated.'
        prices = np.array([price])
        forwards = np.array([forward])
        strikes = np.array([strike])
        expiries = np.array([expiry])
        is_call_options = np.array([option_sign > 0])
        discounts = np.array([1.0])
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.evaluate(tff.black_scholes.implied_vol_newton(prices=prices, strikes=strikes, expiries=expiries, forwards=forwards, discount_factors=discounts, validate_args=True, is_call_options=is_call_options))

    def test_implied_vol_extensive(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(135)
        num_examples = 1000
        expiries = np.linspace(0.8, 1.2, num_examples)
        rates = np.linspace(0.03, 0.08, num_examples)
        discount_factors = np.exp(-rates * expiries)
        spots = np.ones(num_examples)
        forwards = spots / discount_factors
        strikes = np.linspace(0.8, 1.2, num_examples)
        volatilities = np.ones_like(forwards)
        call_options = np.random.binomial(n=1, p=0.5, size=num_examples)
        is_call_options = np.array(call_options, dtype=bool)
        prices = self.evaluate(tff.black_scholes.option_price(volatilities=volatilities, strikes=strikes, expiries=expiries, forwards=forwards, is_call_options=is_call_options, discount_factors=discount_factors, dtype=tf.float64))
        implied_vols = self.evaluate(tff.black_scholes.implied_vol_newton(prices=prices, strikes=strikes, expiries=expiries, forwards=forwards, discount_factors=discount_factors, is_call_options=is_call_options, dtype=tf.float64, max_iterations=1000, tolerance=1e-08))[0]
        self.assertArrayNear(volatilities, implied_vols, 1e-07)

    def test_discount_factor_correctness(self):
        if False:
            i = 10
            return i + 15
        dtype = np.float64
        expiries = np.array([1.0], dtype=dtype)
        rates = np.array([0.05], dtype=dtype)
        discount_factors = np.exp(-rates * expiries)
        spots = np.array([1.0], dtype=dtype)
        strikes = np.array([0.9], dtype=dtype)
        volatilities = np.array([0.13], dtype=dtype)
        is_call_options = True
        prices = self.evaluate(tff.black_scholes.option_price(volatilities=volatilities, strikes=strikes, expiries=expiries, spots=spots, is_call_options=is_call_options, discount_factors=discount_factors, dtype=tf.float64))
        implied_vols = self.evaluate(tff.black_scholes.implied_vol_newton(prices=prices, strikes=strikes, expiries=expiries, spots=spots, discount_factors=discount_factors, is_call_options=is_call_options, dtype=tf.float64, max_iterations=1000, tolerance=1e-08))[0]
        self.assertArrayNear(volatilities, implied_vols, 1e-07)

    def test_bachelier_tricky(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests the Newton root finder recovers the volatility on a few cases.'
        forwards = np.array([0.00982430235191995])
        strikes = np.array([0.00982430235191995])
        expiries = np.array([0.5])
        discounts = np.array([1.0])
        is_call_options = np.array([True])
        volatilities = np.array([0.01])
        prices = np.array([0.002820947917738782])
        (implied_vols, converged, failed) = self.evaluate(tff.black_scholes.implied_vol_newton(prices=prices, strikes=strikes, expiries=expiries, forwards=forwards, discount_factors=discounts, is_call_options=is_call_options, underlying_distribution=ImpliedVolUnderlyingDistribution.NORMAL, max_iterations=100, dtype=np.float64))
        self.assertTrue(np.all(converged))
        self.assertFalse(np.any(failed))
        self.assertArrayNear(volatilities, implied_vols, 1e-07)
if __name__ == '__main__':
    tf.test.main()