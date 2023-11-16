"""Tests for svi.parameterizations."""
import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util

@test_util.run_all_in_graph_and_eager_modes
class SviParameterizationsTest(tf.test.TestCase):

    def test_variance_correctness(self):
        if False:
            i = 10
            return i + 15
        a = -0.02
        b = 0.15
        rho = 0.3
        m = 0.2
        sigma = 0.4
        parameters = np.array([a, b, rho, m, sigma])
        forwards = np.array([8.0])
        strikes = np.linspace(4.0, 16.0, 10)
        actual = self.evaluate(tff.experimental.svi.total_variance_from_raw_svi_parameters(svi_parameters=parameters, forwards=forwards, strikes=strikes))
        expected = np.array([0.08660251, 0.06160364, 0.04579444, 0.03808204, 0.03832965, 0.04537031, 0.05669857, 0.06997256, 0.0837822, 0.09743798])
        self.assertAllClose(actual, expected, 1e-08)

    def test_volatility_correctness(self):
        if False:
            for i in range(10):
                print('nop')
        a = 0.03
        b = 0.1
        rho = -0.2
        m = 0.05
        sigma = 0.5
        k = np.linspace(-1.25, 1.25, 11)
        parameters = np.array([[a, b, rho, m, sigma]])
        expiries = np.array([2.0])
        actual = self.evaluate(tff.experimental.svi.implied_volatility_from_raw_svi_parameters(svi_parameters=parameters, log_moneyness=k, expiries=expiries))
        expected = np.array([[0.31247711, 0.28922053, 0.26489603, 0.24013574, 0.21715147, 0.20155567, 0.19981447, 0.21008108, 0.22585754, 0.2432638, 0.2607681]])
        self.assertAllClose(actual, expected, 1e-08)

    def test_raises_when_bad_combination_of_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(ValueError, 'Either both `forwards` and `strikes` must be supplied, or neither.'):
            self.evaluate(tff.experimental.svi.total_variance_from_raw_svi_parameters(svi_parameters=None, forwards=np.array([5.0])))
        with self.assertRaisesRegex(ValueError, 'Exactly one of `log_moneyness` or `forwards` must be provided.'):
            self.evaluate(tff.experimental.svi.total_variance_from_raw_svi_parameters(svi_parameters=None, log_moneyness=np.array([0.0]), forwards=np.array([5.0]), strikes=np.array([6.0])))
if __name__ == '__main__':
    tf.test.main()