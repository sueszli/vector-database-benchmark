"""Tests for nelson_svensson_interpolation."""
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util

@test_util.run_all_in_graph_and_eager_modes
class NelsonSvenssonInterpolationTest(tf.test.TestCase, parameterized.TestCase):

    def test_all_beta_0(self):
        if False:
            i = 10
            return i + 15
        interpolation_times = [5.0, 10.0, 15.0, 20.0]
        s_p = tff.rates.nelson_seigel_svensson.SvenssonParameters(beta_0=0.0, beta_1=0.0, beta_2=0.0, beta_3=0.0, tau_1=100.0, tau_2=10.0)
        output = self.evaluate(tff.rates.nelson_seigel_svensson.interpolate(interpolation_times, s_p))
        expected_output = [0.0, 0.0, 0.0, 0.0]
        np.testing.assert_allclose(output, expected_output)

    def test_custom_input(self):
        if False:
            i = 10
            return i + 15
        interpolation_times = [5.0, 10.0, 15.0, 20.0]
        s_p = tff.rates.nelson_seigel_svensson.SvenssonParameters(beta_0=0.05, beta_1=-0.01, beta_2=0.3, beta_3=0.02, tau_1=1.5, tau_2=20.0)
        output = self.evaluate(tff.rates.nelson_seigel_svensson.interpolate(interpolation_times, s_p))
        expected_output = [0.12531409, 0.09667101, 0.08360796, 0.0770343]
        np.testing.assert_allclose(output, expected_output, atol=1e-05, rtol=1e-05)

    def test_batch_input(self):
        if False:
            while True:
                i = 10
        interpolation_times = [[1.0, 2.0, 3.0, 4.0], [5.0, 10.0, 15.0, 20.0], [30.0, 40.0, 50.0, 60.0]]
        s_p = tff.rates.nelson_seigel_svensson.SvenssonParameters(beta_0=0.25, beta_1=-0.4, beta_2=-0.02, beta_3=0.05, tau_1=4.5, tau_2=10)
        output = self.evaluate(tff.rates.nelson_seigel_svensson.interpolate(interpolation_times, s_p))
        expected_output = [[-0.10825214, -0.07188015, -0.04012282, -0.0123332], [0.01203921, 0.09686097, 0.14394756, 0.1716945], [0.20045316, 0.21411455, 0.22179659, 0.22668882]]
        np.testing.assert_allclose(output, expected_output, atol=1e-05, rtol=1e-05)
if __name__ == '__main__':
    tf.test.main()