"""Tests for tensorflow.python.ops.special_math_ops."""
from absl.testing import parameterized
import numpy as np
import opt_einsum
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

class LBetaTest(test.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def test_one_dimensional_arg(self):
        if False:
            while True:
                i = 10
        x_one = [1, 1.0]
        x_one_half = [2, 1.0]
        with self.session():
            self.assertAllClose(1, self.evaluate(math_ops.exp(special_math_ops.lbeta(x_one))))
            self.assertAllClose(0.5, self.evaluate(math_ops.exp(special_math_ops.lbeta(x_one_half))))
            self.assertEqual([], special_math_ops.lbeta(x_one).get_shape())

    @test_util.run_deprecated_v1
    def test_one_dimensional_arg_dynamic(self):
        if False:
            i = 10
            return i + 15
        x_one = [1, 1.0]
        x_one_half = [2, 1.0]
        with self.session():
            ph = array_ops.placeholder(dtypes.float32)
            beta_ph = math_ops.exp(special_math_ops.lbeta(ph))
            self.assertAllClose(1, beta_ph.eval(feed_dict={ph: x_one}))
            self.assertAllClose(0.5, beta_ph.eval(feed_dict={ph: x_one_half}))

    @test_util.run_deprecated_v1
    def test_four_dimensional_arg_with_partial_shape_dynamic(self):
        if False:
            for i in range(10):
                print('nop')
        x_ = np.ones((3, 2, 3, 4))
        expected_beta_x = 1 / 6 * np.ones((3, 2, 3))
        with self.session():
            x_ph = array_ops.placeholder(dtypes.float32, [3, 2, 3, None])
            beta_ph = math_ops.exp(special_math_ops.lbeta(x_ph))
            self.assertAllClose(expected_beta_x, beta_ph.eval(feed_dict={x_ph: x_}))

    @test_util.run_in_graph_and_eager_modes
    def test_two_dimensional_arg(self):
        if False:
            i = 10
            return i + 15
        x_one_half = [[2, 1.0], [2, 1.0]]
        with self.session():
            self.assertAllClose([0.5, 0.5], self.evaluate(math_ops.exp(special_math_ops.lbeta(x_one_half))))
            self.assertEqual((2,), special_math_ops.lbeta(x_one_half).get_shape())

    @test_util.run_deprecated_v1
    def test_two_dimensional_arg_dynamic(self):
        if False:
            i = 10
            return i + 15
        x_one_half = [[2, 1.0], [2, 1.0]]
        with self.session():
            ph = array_ops.placeholder(dtypes.float32)
            beta_ph = math_ops.exp(special_math_ops.lbeta(ph))
            self.assertAllClose([0.5, 0.5], beta_ph.eval(feed_dict={ph: x_one_half}))

    @test_util.run_in_graph_and_eager_modes
    def test_two_dimensional_proper_shape(self):
        if False:
            print('Hello World!')
        x_one_half = [[2, 1.0], [2, 1.0]]
        with self.session():
            self.assertAllClose([0.5, 0.5], self.evaluate(math_ops.exp(special_math_ops.lbeta(x_one_half))))
            self.assertEqual((2,), self.evaluate(array_ops.shape(special_math_ops.lbeta(x_one_half))))
            self.assertEqual(tensor_shape.TensorShape([2]), special_math_ops.lbeta(x_one_half).get_shape())

    @test_util.run_in_graph_and_eager_modes
    def test_complicated_shape(self):
        if False:
            i = 10
            return i + 15
        with self.session():
            x = ops.convert_to_tensor(np.random.rand(3, 2, 2))
            self.assertAllEqual((3, 2), self.evaluate(array_ops.shape(special_math_ops.lbeta(x))))
            self.assertEqual(tensor_shape.TensorShape([3, 2]), special_math_ops.lbeta(x).get_shape())

    @test_util.run_in_graph_and_eager_modes
    def test_length_1_last_dimension_results_in_one(self):
        if False:
            while True:
                i = 10
        x_a = [5.5]
        x_b = [0.1]
        with self.session():
            self.assertAllClose(1, self.evaluate(math_ops.exp(special_math_ops.lbeta(x_a))), rtol=3e-06)
            self.assertAllClose(1, self.evaluate(math_ops.exp(special_math_ops.lbeta(x_b))))
            self.assertEqual((), special_math_ops.lbeta(x_a).get_shape())

    @test_util.run_in_graph_and_eager_modes
    def test_empty_rank1_returns_negative_infinity(self):
        if False:
            while True:
                i = 10
        with self.session():
            x = constant_op.constant([], shape=[0])
            lbeta_x = special_math_ops.lbeta(x)
            expected_result = constant_op.constant(-np.inf, shape=())
            self.assertAllEqual(self.evaluate(expected_result), self.evaluate(lbeta_x))
            self.assertEqual(expected_result.get_shape(), lbeta_x.get_shape())

    @test_util.run_in_graph_and_eager_modes
    def test_empty_rank2_with_zero_last_dim_returns_negative_infinity(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session():
            event_size = 0
            for batch_size in [0, 1, 2]:
                x = constant_op.constant([], shape=[batch_size, event_size])
                lbeta_x = special_math_ops.lbeta(x)
                expected_result = constant_op.constant(-np.inf, shape=[batch_size])
                self.assertAllEqual(self.evaluate(expected_result), self.evaluate(lbeta_x))
                self.assertEqual(expected_result.get_shape(), lbeta_x.get_shape())

    @test_util.run_in_graph_and_eager_modes
    def test_empty_rank2_with_zero_batch_dim_returns_empty(self):
        if False:
            return 10
        with self.session():
            batch_size = 0
            for event_size in [0, 1, 2]:
                x = constant_op.constant([], shape=[batch_size, event_size])
                lbeta_x = special_math_ops.lbeta(x)
                expected_result = constant_op.constant([], shape=[batch_size])
                self.assertAllEqual(self.evaluate(expected_result), self.evaluate(lbeta_x))
                self.assertEqual(expected_result.get_shape(), lbeta_x.get_shape())

@test_util.run_all_in_graph_and_eager_modes
class DawsnTest(test.TestCase, parameterized.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def test_dawsn_boundary(self):
        if False:
            return 10
        self.assertAllClose(0.0, special_math_ops.dawsn(0.0))
        self.assertTrue(np.isnan(self.evaluate(special_math_ops.dawsn(np.nan))))

    @parameterized.parameters(np.float32, np.float64)
    def test_dawsn_odd(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        x = np.random.uniform(-100.0, 100.0, size=int(10000.0)).astype(dtype)
        self.assertAllClose(self.evaluate(special_math_ops.dawsn(x)), self.evaluate(-special_math_ops.dawsn(-x)))

    @parameterized.parameters(np.float32, np.float64)
    def test_dawsn_small(self, dtype):
        if False:
            while True:
                i = 10
        x = np.random.uniform(-1.0, 1.0, size=int(10000.0)).astype(dtype)
        try:
            from scipy import special
            self.assertAllClose(special.dawsn(x), self.evaluate(special_math_ops.dawsn(x)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    @parameterized.parameters(np.float32, np.float64)
    def test_dawsn_larger(self, dtype):
        if False:
            print('Hello World!')
        x = np.random.uniform(1.0, 100.0, size=int(10000.0)).astype(dtype)
        try:
            from scipy import special
            self.assertAllClose(special.dawsn(x), self.evaluate(special_math_ops.dawsn(x)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    def test_dawsn_gradient(self):
        if False:
            while True:
                i = 10
        inputs = [np.random.uniform(-50.0, 50.0, size=int(100.0))]
        (analytical, numerical) = gradient_checker_v2.compute_gradient(special_math_ops.dawsn, inputs)
        self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.0001)

@test_util.run_all_in_graph_and_eager_modes
class ExpintTest(test.TestCase, parameterized.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def test_expint_boundary(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertAllClose(-np.inf, special_math_ops.expint(0.0))
        self.assertTrue(np.isnan(self.evaluate(special_math_ops.expint(np.nan))))
        self.assertTrue(np.all(np.isnan(self.evaluate(special_math_ops.expint(np.random.uniform(-20.0, -1.0, size=int(1000.0)))))))

    @parameterized.parameters(np.float32, np.float64)
    def test_expint_small(self, dtype):
        if False:
            print('Hello World!')
        x = np.random.uniform(0.0, 1.0, size=int(10000.0)).astype(dtype)
        try:
            from scipy import special
            self.assertAllClose(special.expi(x), self.evaluate(special_math_ops.expint(x)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    @parameterized.parameters(np.float32, np.float64)
    def test_expint_larger(self, dtype):
        if False:
            while True:
                i = 10
        x = np.random.uniform(1.0, 50.0, size=int(10000.0)).astype(dtype)
        try:
            from scipy import special
            self.assertAllClose(special.expi(x), self.evaluate(special_math_ops.expint(x)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    def test_expint_gradient(self):
        if False:
            i = 10
            return i + 15
        inputs = [np.random.uniform(1.0, 10.0, size=int(100.0))]
        (analytical, numerical) = gradient_checker_v2.compute_gradient(special_math_ops.expint, inputs)
        self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.005)

@test_util.run_all_in_graph_and_eager_modes
class FresnelCosTest(test.TestCase, parameterized.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def test_fresnel_cos_boundary(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertAllClose(0.0, special_math_ops.fresnel_cos(0.0))
        self.assertTrue(np.isnan(self.evaluate(special_math_ops.fresnel_cos(np.nan))))

    @parameterized.parameters(np.float32, np.float64)
    def test_fresnel_cos_odd(self, dtype):
        if False:
            return 10
        x = np.random.uniform(-100.0, 100.0, size=int(10000.0)).astype(dtype)
        self.assertAllClose(self.evaluate(special_math_ops.fresnel_cos(x)), self.evaluate(-special_math_ops.fresnel_cos(-x)))

    @parameterized.parameters(np.float32, np.float64)
    def test_fresnel_cos_small(self, dtype):
        if False:
            return 10
        x = np.random.uniform(0.0, 1.0, size=int(10000.0)).astype(dtype)
        try:
            from scipy import special
            self.assertAllClose(special.fresnel(x)[1], self.evaluate(special_math_ops.fresnel_cos(x)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    @parameterized.parameters(np.float32, np.float64)
    def test_fresnel_cos_larger(self, dtype):
        if False:
            return 10
        x = np.random.uniform(1.0, 100.0, size=int(10000.0)).astype(dtype)
        try:
            from scipy import special
            self.assertAllClose(special.fresnel(x)[1], self.evaluate(special_math_ops.fresnel_cos(x)), rtol=1e-05)
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    def test_fresnel_cos_gradient(self):
        if False:
            print('Hello World!')
        inputs = [np.random.uniform(1.0, 50.0, size=int(100.0))]
        (analytical, numerical) = gradient_checker_v2.compute_gradient(special_math_ops.fresnel_cos, inputs)
        self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.005)

@test_util.run_all_in_graph_and_eager_modes
class FresnelSinTest(test.TestCase, parameterized.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def test_fresnel_sin_boundary(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertAllClose(0.0, special_math_ops.fresnel_sin(0.0))
        self.assertTrue(np.isnan(self.evaluate(special_math_ops.fresnel_sin(np.nan))))

    @parameterized.parameters(np.float32, np.float64)
    def test_fresnel_sin_odd(self, dtype):
        if False:
            i = 10
            return i + 15
        x = np.random.uniform(-100.0, 100.0, size=int(10000.0)).astype(dtype)
        self.assertAllClose(self.evaluate(special_math_ops.fresnel_sin(x)), self.evaluate(-special_math_ops.fresnel_sin(-x)))

    @parameterized.parameters(np.float32, np.float64)
    def test_fresnel_sin_small(self, dtype):
        if False:
            i = 10
            return i + 15
        x = np.random.uniform(0.0, 1.0, size=int(10000.0)).astype(dtype)
        try:
            from scipy import special
            self.assertAllClose(special.fresnel(x)[0], self.evaluate(special_math_ops.fresnel_sin(x)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    @parameterized.parameters(np.float32, np.float64)
    def test_fresnel_sin_larger(self, dtype):
        if False:
            while True:
                i = 10
        x = np.random.uniform(1.0, 100.0, size=int(10000.0)).astype(dtype)
        try:
            from scipy import special
            self.assertAllClose(special.fresnel(x)[0], self.evaluate(special_math_ops.fresnel_sin(x)), rtol=1e-05)
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    def test_fresnel_sin_gradient(self):
        if False:
            for i in range(10):
                print('nop')
        inputs = [np.random.uniform(1.0, 50.0, size=int(100.0))]
        (analytical, numerical) = gradient_checker_v2.compute_gradient(special_math_ops.fresnel_sin, inputs)
        self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.005)

@test_util.run_all_in_graph_and_eager_modes
class SpenceTest(test.TestCase, parameterized.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def test_spence_boundary(self):
        if False:
            print('Hello World!')
        self.assertAllClose(np.pi ** 2 / 6.0, special_math_ops.spence(0.0))
        self.assertAllClose(0.0, special_math_ops.spence(1.0))
        self.assertTrue(np.isnan(self.evaluate(special_math_ops.spence(np.nan))))
        self.assertTrue(np.all(np.isnan(self.evaluate(special_math_ops.spence(np.random.uniform(-20.0, -1.0, size=int(1000.0)))))))

    @parameterized.parameters(np.float32, np.float64)
    def test_spence_small(self, dtype):
        if False:
            while True:
                i = 10
        x = np.random.uniform(0.0, 1.0, size=int(10000.0)).astype(dtype)
        try:
            from scipy import special
            self.assertAllClose(special.spence(x), self.evaluate(special_math_ops.spence(x)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    @parameterized.parameters(np.float32, np.float64)
    def test_spence_larger(self, dtype):
        if False:
            while True:
                i = 10
        x = np.random.uniform(1.0, 100.0, size=int(10000.0)).astype(dtype)
        try:
            from scipy import special
            self.assertAllClose(special.spence(x), self.evaluate(special_math_ops.spence(x)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    def test_spence_gradient(self):
        if False:
            i = 10
            return i + 15
        inputs = [np.random.uniform(1.0, 50.0, size=int(100.0))]
        (analytical, numerical) = gradient_checker_v2.compute_gradient(special_math_ops.spence, inputs)
        self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.0001)

    def test_spence_gradient_at_one(self):
        if False:
            while True:
                i = 10
        (analytical, _) = gradient_checker_v2.compute_gradient(special_math_ops.spence, [1.0])
        self.assertAllClose([[[-1.0]]], analytical)

@test_util.run_all_in_graph_and_eager_modes
class BesselTest(test.TestCase, parameterized.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def test_besseli_boundary(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertAllClose(1.0, special_math_ops.bessel_i0(0.0))
        self.assertAllClose(1.0, special_math_ops.bessel_i0e(0.0))
        self.assertAllClose(0.0, special_math_ops.bessel_i1(0.0))
        self.assertAllClose(0.0, special_math_ops.bessel_i1e(0.0))
        self.assertTrue(np.isnan(self.evaluate(special_math_ops.bessel_i0(np.nan))))
        self.assertTrue(np.isnan(self.evaluate(special_math_ops.bessel_i0e(np.nan))))
        self.assertTrue(np.isnan(self.evaluate(special_math_ops.bessel_i1(np.nan))))
        self.assertTrue(np.isnan(self.evaluate(special_math_ops.bessel_i1e(np.nan))))

    @test_util.run_in_graph_and_eager_modes
    def test_besselj_boundary(self):
        if False:
            print('Hello World!')
        self.assertAllClose(1.0, special_math_ops.bessel_j0(0.0))
        self.assertAllClose(0.0, special_math_ops.bessel_j1(0.0))
        self.assertTrue(np.isnan(self.evaluate(special_math_ops.bessel_j0(np.nan))))
        self.assertTrue(np.isnan(self.evaluate(special_math_ops.bessel_j1(np.nan))))

    @test_util.run_in_graph_and_eager_modes
    def test_besselk_boundary(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(np.isinf(self.evaluate(special_math_ops.bessel_k0(0.0))))
        self.assertTrue(np.isinf(self.evaluate(special_math_ops.bessel_k0e(0.0))))
        self.assertTrue(np.isinf(self.evaluate(special_math_ops.bessel_k1(0.0))))
        self.assertTrue(np.isinf(self.evaluate(special_math_ops.bessel_k1e(0.0))))
        self.assertTrue(np.isnan(self.evaluate(special_math_ops.bessel_k0(np.nan))))
        self.assertTrue(np.isnan(self.evaluate(special_math_ops.bessel_k0e(np.nan))))
        self.assertTrue(np.isnan(self.evaluate(special_math_ops.bessel_k1(np.nan))))
        self.assertTrue(np.isnan(self.evaluate(special_math_ops.bessel_k1e(np.nan))))

    @parameterized.parameters(np.float32, np.float64)
    def test_i0j0_even(self, dtype):
        if False:
            i = 10
            return i + 15
        x = np.random.uniform(-100.0, 100.0, size=int(10000.0)).astype(dtype)
        self.assertAllClose(self.evaluate(special_math_ops.bessel_i0(x)), self.evaluate(special_math_ops.bessel_i0(-x)))
        self.assertAllClose(self.evaluate(special_math_ops.bessel_i0e(x)), self.evaluate(special_math_ops.bessel_i0e(-x)))
        self.assertAllClose(self.evaluate(special_math_ops.bessel_j0(x)), self.evaluate(special_math_ops.bessel_j0(-x)))

    @parameterized.parameters(np.float32, np.float64)
    def test_i1j1_odd(self, dtype):
        if False:
            while True:
                i = 10
        x = np.random.uniform(-100.0, 100.0, size=int(10000.0)).astype(dtype)
        self.assertAllClose(self.evaluate(special_math_ops.bessel_i1(x)), self.evaluate(-special_math_ops.bessel_i1(-x)))
        self.assertAllClose(self.evaluate(special_math_ops.bessel_i1e(x)), self.evaluate(-special_math_ops.bessel_i1e(-x)))
        self.assertAllClose(self.evaluate(special_math_ops.bessel_j1(x)), self.evaluate(-special_math_ops.bessel_j1(-x)))

    @parameterized.parameters(np.float32, np.float64)
    def test_besseli_small(self, dtype):
        if False:
            return 10
        x = np.random.uniform(-1.0, 1.0, size=int(10000.0)).astype(dtype)
        try:
            from scipy import special
            self.assertAllClose(special.i0(x), self.evaluate(special_math_ops.bessel_i0(x)))
            self.assertAllClose(special.i1(x), self.evaluate(special_math_ops.bessel_i1(x)))
            self.assertAllClose(special.i0e(x), self.evaluate(special_math_ops.bessel_i0e(x)))
            self.assertAllClose(special.i1e(x), self.evaluate(special_math_ops.bessel_i1e(x)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    @parameterized.parameters(np.float32, np.float64)
    def test_besselj_small(self, dtype):
        if False:
            return 10
        x = np.random.uniform(-1.0, 1.0, size=int(10000.0)).astype(dtype)
        try:
            from scipy import special
            self.assertAllClose(special.j0(x), self.evaluate(special_math_ops.bessel_j0(x)))
            self.assertAllClose(special.j1(x), self.evaluate(special_math_ops.bessel_j1(x)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    @parameterized.parameters(np.float32, np.float64)
    def test_besselk_small(self, dtype):
        if False:
            while True:
                i = 10
        x = np.random.uniform(np.finfo(dtype).eps, 1.0, size=int(10000.0)).astype(dtype)
        try:
            from scipy import special
            self.assertAllClose(special.k0(x), self.evaluate(special_math_ops.bessel_k0(x)))
            self.assertAllClose(special.k0e(x), self.evaluate(special_math_ops.bessel_k0e(x)))
            self.assertAllClose(special.k1(x), self.evaluate(special_math_ops.bessel_k1(x)))
            self.assertAllClose(special.k1e(x), self.evaluate(special_math_ops.bessel_k1e(x)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    @parameterized.parameters(np.float32, np.float64)
    def test_bessely_small(self, dtype):
        if False:
            while True:
                i = 10
        x = np.random.uniform(np.finfo(dtype).eps, 1.0, size=int(10000.0)).astype(dtype)
        try:
            from scipy import special
            self.assertAllClose(special.y0(x), self.evaluate(special_math_ops.bessel_y0(x)))
            self.assertAllClose(special.y1(x), self.evaluate(special_math_ops.bessel_y1(x)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    @parameterized.parameters(np.float32, np.float64)
    def test_besseli_larger(self, dtype):
        if False:
            i = 10
            return i + 15
        x = np.random.uniform(1.0, 20.0, size=int(10000.0)).astype(dtype)
        try:
            from scipy import special
            self.assertAllClose(special.i0e(x), self.evaluate(special_math_ops.bessel_i0e(x)))
            self.assertAllClose(special.i1e(x), self.evaluate(special_math_ops.bessel_i1e(x)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    @parameterized.parameters(np.float32, np.float64)
    def test_besselj_larger(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        x = np.random.uniform(1.0, 30.0, size=int(10000.0)).astype(dtype)
        try:
            from scipy import special
            self.assertAllClose(special.j0(x), self.evaluate(special_math_ops.bessel_j0(x)))
            self.assertAllClose(special.j1(x), self.evaluate(special_math_ops.bessel_j1(x)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    @parameterized.parameters(np.float32, np.float64)
    def test_besselk_larger(self, dtype):
        if False:
            i = 10
            return i + 15
        x = np.random.uniform(1.0, 30.0, size=int(10000.0)).astype(dtype)
        try:
            from scipy import special
            self.assertAllClose(special.k0(x), self.evaluate(special_math_ops.bessel_k0(x)))
            self.assertAllClose(special.k0e(x), self.evaluate(special_math_ops.bessel_k0e(x)))
            self.assertAllClose(special.k1(x), self.evaluate(special_math_ops.bessel_k1(x)))
            self.assertAllClose(special.k1e(x), self.evaluate(special_math_ops.bessel_k1e(x)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    @parameterized.parameters(np.float32, np.float64)
    def test_bessely_larger(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        x = np.random.uniform(1.0, 30.0, size=int(10000.0)).astype(dtype)
        try:
            from scipy import special
            self.assertAllClose(special.y0(x), self.evaluate(special_math_ops.bessel_y0(x)))
            self.assertAllClose(special.y1(x), self.evaluate(special_math_ops.bessel_y1(x)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    def test_besseli_gradient(self):
        if False:
            print('Hello World!')
        inputs = [np.random.uniform(-10.0, 10.0, size=int(100.0))]
        (analytical, numerical) = gradient_checker_v2.compute_gradient(special_math_ops.bessel_i0, inputs)
        self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.001)
        (analytical, numerical) = gradient_checker_v2.compute_gradient(special_math_ops.bessel_i0e, inputs)
        self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.0001)
        (analytical, numerical) = gradient_checker_v2.compute_gradient(special_math_ops.bessel_i1, inputs)
        self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.001)
        (analytical, numerical) = gradient_checker_v2.compute_gradient(special_math_ops.bessel_i1e, inputs)
        self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.0001)

    def test_besselj_gradient(self):
        if False:
            return 10
        inputs = [np.random.uniform(-50.0, 50.0, size=int(100.0))]
        (analytical, numerical) = gradient_checker_v2.compute_gradient(special_math_ops.bessel_j0, inputs)
        self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.0001)
        (analytical, numerical) = gradient_checker_v2.compute_gradient(special_math_ops.bessel_j1, inputs)
        self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.0001)

    def test_besselk_gradient(self):
        if False:
            print('Hello World!')
        inputs = [np.random.uniform(1.0, 50.0, size=int(100.0))]
        (analytical, numerical) = gradient_checker_v2.compute_gradient(special_math_ops.bessel_k0, inputs)
        self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.0001)
        (analytical, numerical) = gradient_checker_v2.compute_gradient(special_math_ops.bessel_k0e, inputs)
        self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.0001)
        (analytical, numerical) = gradient_checker_v2.compute_gradient(special_math_ops.bessel_k1, inputs)
        self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.0001)
        (analytical, numerical) = gradient_checker_v2.compute_gradient(special_math_ops.bessel_k1e, inputs)
        self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.0001)

    def test_bessely_gradient(self):
        if False:
            return 10
        inputs = [np.random.uniform(1.0, 50.0, size=int(100.0))]
        (analytical, numerical) = gradient_checker_v2.compute_gradient(special_math_ops.bessel_y0, inputs)
        self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.0001)
        (analytical, numerical) = gradient_checker_v2.compute_gradient(special_math_ops.bessel_y1, inputs)
        self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.0001)

@test_util.run_all_in_graph_and_eager_modes
@test_util.run_all_without_tensor_float_32('Tests einsum, which sometimes does a matmul with cuBLAS')
class EinsumTest(test.TestCase):

    def _check(self, s, *input_shapes, **kwargs):
        if False:
            i = 10
            return i + 15
        dtype = kwargs.pop('dtype', np.float32)
        r = np.random.RandomState(0)
        inputs = []
        for shape in input_shapes:
            arr = np.array(r.randn(*shape)).astype(dtype)
            if dtype == np.complex64 or dtype == np.complex128:
                arr += 1j * np.array(r.randn(*shape)).astype(dtype)
            inputs.append(arr)
        input_tensors = [constant_op.constant(x, shape=x.shape) for x in inputs]
        a = np.einsum(s, *inputs)
        b = self.evaluate(special_math_ops.einsum(s, *input_tensors))
        self.assertAllClose(a, b, atol=0.0001, rtol=0.0001)

    def test_invalid_keyword_arguments(self):
        if False:
            print('Hello World!')
        r = np.random.RandomState(0)
        a = array_ops.placeholder_with_default(r.randn(2, 3), shape=(2, 3))
        b = array_ops.placeholder_with_default(r.randn(3, 4), shape=(3, 4))
        with self.assertRaises(TypeError):
            _ = special_math_ops.einsum('ij,jk->ik', a, b, name='name', invalid1='value1', invalid2='value2')

    def test_unary(self):
        if False:
            print('Hello World!')
        self._check('a', (3,))
        self._check('aa', (3, 3))
        self._check('ab->', (3, 3))
        self._check('ab->ab', (3, 3))
        self._check('abc->b', (3, 4, 5))
        self._check('abc->ca', (3, 4, 5))
        self._check('abc->cab', (3, 4, 5))
        self._check('', ())
        self._check('->', ())
        self._check('aa->', (3, 3))
        self._check('aa->a', (3, 3))
        self._check('aaa->', (3, 3, 3))
        self._check('aaa->a', (3, 3, 3))
        self._check('aab->a', (3, 3, 4))
        self._check('aabcc->a', (3, 3, 5, 4, 4))
        self._check('aabcc->ac', (3, 3, 5, 4, 4))
        self._check('aabcd->ad', (3, 3, 5, 4, 4))

    def test_unary_ellipsis(self):
        if False:
            return 10
        self._check('...->', ())
        self._check('...ijk->...ki', (3, 4, 5))
        self._check('...ijk->...ki', (1, 3, 4, 5))
        self._check('...ijk->...ki', (2, 2, 3, 4, 5))
        self._check('...ij->...ji', (5, 2, 3))
        self._check('...ij->...', (5, 2, 3))
        self._check('...->...', ())
        self._check('->...', ())
        self._check('i...ii->...i', (3, 2, 3, 3))
        self._check('i...i->i...', (2, 2))
        self._check('i...i->', (2, 2))
        self._check('i...i->...', (2, 5, 1, 2))
        self._check('i...i->i...', (2, 1, 2))
        self._check('i...i->i...', (2, 3, 4, 5, 2))

    def test_binary_simple(self):
        if False:
            for i in range(10):
                print('nop')
        self._check(',->', (), ())
        self._check('a,a->', (3,), (3,))
        self._check('a,a->a', (3,), (3,))
        self._check('ab,b->a', (3, 4), (4,))
        self._check('ab,ab->', (3, 4), (3, 4))
        self._check('ab,bc->ac', (3, 4), (4, 5))
        self._check('nij,jk->nik', (5, 2, 3), (3, 4))
        self._check('abc,bad->abcd', (1, 2, 3), (2, 1, 4))
        self._check('sa,shb->shab', (2, 1), (2, 3, 4))
        self._check('ab,b', (3, 4), (4,))
        self._check('cab,b', (1, 3, 4), (4,))

    def test_reduced_indices(self):
        if False:
            while True:
                i = 10
        self._check('ba,b->', (3, 2), (3,))
        self._check('ab,ab->', (3, 4), (3, 4))

    def test_repeated_indices(self):
        if False:
            for i in range(10):
                print('nop')
        self._check('ijj,k->ik', (2, 3, 3), (4,))
        self._check('aba,a->b', (3, 4, 3), (3,))
        self._check('aab,bc->ac', (2, 2, 3), (3, 4))
        self._check('aab,bcc->ac', (2, 2, 3), (3, 4, 4))

    def test_binary_ellipsis(self):
        if False:
            i = 10
            return i + 15
        self._check('...mk,...kn->...mn', (5, 1, 2, 3), (5, 1, 3, 4))
        self._check('...mk,...kn->...mn', (2, 3), (3, 4))
        self._check('...ija,aijb...->ba...ij', (1, 2, 2, 3, 1), (1, 2, 3, 4, 1, 2))
        self._check('...mk,...kn->mn', (2, 3), (3, 4))
        self._check('...mk,kn->mn', (2, 3), (3, 4))
        self._check('mk,...kn->mn', (2, 3), (3, 4))
        self._check('...,...->...', (2, 3), (2, 3))
        self._check('...i,...j->...ij', (5, 2), (5, 3))

    def test_broadcasting(self):
        if False:
            i = 10
            return i + 15
        self._check('...ij,...jk->...ik', (1, 2, 3), (3, 5))
        self._check('...ij,...jk->...ik', (2, 3), (1, 3, 5))
        self._check('...ij,...jk->...ik', (5, 2, 3), (3, 5))
        self._check('...ij,...jk->...ik', (2, 3), (5, 3, 5))
        self._check('...ij,...jk->...ik', (3, 1, 2, 3), (1, 1, 7, 3, 5))
        self._check('i...j,j...k->...ik', (2, 1, 3, 1, 3), (3, 1, 7, 5))
        self._check('ij,jk...k->i...', (3, 2), (2, 4, 1, 4))
        self._check('ij,jk...k->...i', (3, 2), (2, 4, 5, 4))
        self._check('ijj,jk...k->i...', (3, 2, 2), (2, 4, 1, 4))
        self._check('i...jj,jk...k->i...', (3, 3, 1, 2, 2), (2, 4, 1, 5, 4))
        self._check('...abc,...abcd->...d', (1, 1, 2, 3, 4), (5, 2, 3, 4, 6))
        self._check('ab...,b->ab...', (2, 3, 1, 1, 5), (3,))

    def test_dtypes(self):
        if False:
            print('Hello World!')
        dtypes = [np.float64, np.float32, np.complex64, np.complex128]
        for dtype in dtypes:
            self._check('ij,jk->ik', (2, 2), (2, 2), dtype=dtype)
            self._check('ji,jk->ik', (2, 2), (2, 2), dtype=dtype)
            self._check('ji,kj->ik', (2, 2), (2, 2), dtype=dtype)
            self._check('ij,jk->ki', (2, 2), (2, 2), dtype=dtype)
            self._check('ji,kj->ki', (2, 2), (2, 2), dtype=dtype)

    def test_multiple_inputs(self):
        if False:
            while True:
                i = 10
        self._check('ijk,ijl,ikl->i', (1, 2, 3), (1, 2, 4), (1, 3, 4))
        self._check('i,ijk,j->k', (1,), (1, 2, 4), (2,))
        self._check('ij,ij,jk,kl->il', (1, 2), (1, 2), (2, 3), (3, 4))
        self._check('a,b,c', (5,), (7,), (9,))
        self._check('ab,ab,c->c', (5, 6), (5, 6), (2,))

    @test_util.disable_xla('b/131919749')
    def test_placeholder(self):
        if False:
            i = 10
            return i + 15

        def check(equation, *input_and_placeholder_shapes):
            if False:
                for i in range(10):
                    print('nop')
            r = np.random.RandomState(0)
            inputs = []
            input_placeholders = []
            for (actual_shape, placeholder_shape) in input_and_placeholder_shapes:
                input_np = np.array(r.randn(*actual_shape))
                inputs.append(input_np)
                input_placeholders.append(array_ops.placeholder_with_default(input_np, placeholder_shape))
            a = np.einsum(equation, *inputs)
            b = self.evaluate(special_math_ops.einsum(equation, *input_placeholders))
            self.assertAllClose(a, b, atol=0.0001, rtol=0.0001)
        check('bijl,bjkm->bik', ((9, 2, 3, 5), (None, None, None, 5)), ((9, 3, 4, 7), (None, None, 4, None)))
        check('...ij,...->...i', ((4, 3, 1, 2), (None, 3, None, 2)), ((4, 3), (None, 3)))
        check('bijl,bjkm->bik', ((9, 2, 3, 5), None), ((9, 3, 4, 7), None))
        check('...ij,...jk->...ik', ((3, 1, 2, 3), None), ((1, 7, 3, 4), None))

    def test_numpy_input(self):
        if False:
            print('Hello World!')
        r = np.random.RandomState(0)
        s = 'ijk,ijl,ikl->i'
        x = r.randn(1, 2, 3)
        y = r.randn(1, 2, 4)
        z = r.randn(1, 3, 4)
        a = np.einsum(s, x, y, z)
        b = self.evaluate(special_math_ops.einsum(s, x, y, z))
        self.assertAllClose(a, b, atol=0.0001, rtol=0.0001)

    def test_long_cases(self):
        if False:
            i = 10
            return i + 15
        cases = ['efc,dbc,acf,fd->abe', 'ea,fb,gc,hd,abcd->efgh', 'abhe,hidj,jgba,hiab,gab->ed', 'efc, dbc, acf, fd -> abe', 'abhe, hidj, jgba, hiab, gab', 'ea,fb,abcd,gc,hd->efgh', 'ea,fb,abcd,gc,hd->efgh']
        dimension_map = dict(((c, ord(c) - ord('a') + 1) for c in 'abcdefghij'))
        for equation in cases:
            inputs = equation.split('->')[0].replace(' ', '')
            input_shapes = []
            for input_str in inputs.split(','):
                input_shapes.append(tuple([dimension_map[c] for c in input_str]))
            self._check(equation, *input_shapes)

    def test_opt_einsum_cached(self):
        if False:
            i = 10
            return i + 15
        if not context.executing_eagerly():
            return
        input_1 = ('ijk,ijl,ikl->i', (1, 2, 3), (1, 2, 4), (1, 3, 4))
        input_2 = ('ij,ij,jk,kl->il', (1, 2), (1, 2), (2, 3), (3, 4))
        with test.mock.patch.object(opt_einsum, 'contract_path', wraps=opt_einsum.contract_path) as mock_contract_path:
            special_math_ops._get_opt_einsum_contract_path.cache_clear()
            self.assertEqual(mock_contract_path.call_count, 0)
            self._check(*input_1)
            self.assertEqual(mock_contract_path.call_count, 1)
            self._check(*input_1)
            self.assertEqual(mock_contract_path.call_count, 1)
            self._check(*input_2)
            self.assertEqual(mock_contract_path.call_count, 2)
            self._check(*input_1)
            self._check(*input_2)
            self._check(*input_1)
            self.assertEqual(mock_contract_path.call_count, 2)

    @test_util.disable_xla('b/131919749')
    def test_long_cases_with_repeated_labels(self):
        if False:
            while True:
                i = 10
        cases = ['fdf,cdd,ccd,afe->ae', 'fff,fae,bef,def->abd']
        dimension_map = dict(((c, ord(c) - ord('a') + 1) for c in 'abcdefghij'))
        for equation in cases:
            inputs = equation.split('->')[0].replace(' ', '')
            input_shapes = []
            for input_str in inputs.split(','):
                input_shapes.append(tuple([dimension_map[c] for c in input_str]))
            self._check(equation, *input_shapes)

    @test_util.disable_xla('b/131919749')
    @test_util.run_in_graph_and_eager_modes
    def test_invalid_equation(self):
        if False:
            while True:
                i = 10
        r = np.random.RandomState(0)
        cases = [('a0->a', r.randn(5, 3)), ('a->a,a', r.randn(5)), ('a->a->a', r.randn(5)), ('ijk ijk', r.randn(1, 2, 3), r.randn(1, 2, 3)), ('ij.jk->ik', r.randn(2, 3), r.randn(3, 4)), ('a->b', r.randn(5)), ('ij,jk->im', r.randn(2, 3), r.randn(3, 4)), ('ij,jk->ik', r.randn(1, 2, 3), r.randn(3, 4)), ('ij,jk->ik', r.randn(2, 3), r.randn(4, 4)), ('ij,jk->iik', r.randn(2, 3), r.randn(3, 4)), ('...ij...,jk...->ik...', r.randn(2, 3), r.randn(3, 4)), ('...ij,jk...->...ik...', r.randn(2, 3), r.randn(3, 4)), ('...ij,...jk->...ik', r.randn(5, 2, 3), r.randn(7, 3, 4)), ('...ij,...jk->ik', r.randn(2, 2, 3), r.randn(3, 4))]
        for args in cases:
            with self.assertRaises((ValueError, errors.InvalidArgumentError)):
                _ = special_math_ops.einsum(*args)
            placeholders = [array_ops.placeholder_with_default(x, shape=None) for x in args[1:]]
            with self.assertRaises((ValueError, errors.InvalidArgumentError)):
                _ = self.evaluate(special_math_ops.einsum(args[0], *placeholders))

    @test_util.disable_xla('b/131919749')
    def test_empty(self):
        if False:
            while True:
                i = 10

        def check(equation, input_shapes, output_shape):
            if False:
                print('Hello World!')
            r = np.random.RandomState(0)
            inputs = [np.array(r.randn(*shape)) for shape in input_shapes]
            input_tensors = [constant_op.constant(x, shape=x.shape) for x in inputs]
            output = self.evaluate(special_math_ops.einsum(equation, *input_tensors))
            self.assertAllClose(output, np.zeros(output_shape), atol=0.0001, rtol=0.0001)
        check('ab,bc->ac', [(0, 10), (10, 10)], (0, 10))
        check('ibnd,ijbn->jnd', [(1, 0, 5, 10), (1, 1, 0, 5)], (1, 5, 10))
        check('aab,bc->ac', [(0, 0, 10), (10, 10)], (0, 10))
        check('aaab,bc->c', [(0, 0, 0, 3), (3, 4)], (4,))

@test_util.run_all_in_graph_and_eager_modes
class EinsumGradTest(test.TestCase):

    def _check_gradient(self, s, *input_shapes):
        if False:
            return 10
        with self.cached_session():
            r = np.random.RandomState(0)
            inputs = [np.array(r.randn(*shape)) for shape in input_shapes]
            input_tensors = [constant_op.constant(x, shape=x.shape) for x in inputs]
            (analytical, numerical) = gradient_checker_v2.compute_gradient(lambda *xs: special_math_ops.einsum(s, *xs), input_tensors)
            self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.0001)

    @test_util.disable_xla('b/131919749')
    def test_unary(self):
        if False:
            for i in range(10):
                print('nop')
        self._check_gradient('->', ())
        self._check_gradient('aaa->a', (3, 3, 3))
        self._check_gradient('aabcd->ad', (3, 3, 5, 4, 4))
        self._check_gradient('abcd->da', (3, 5, 4, 2))

    @test_util.disable_xla('b/131919749')
    def test_unary_ellipsis(self):
        if False:
            for i in range(10):
                print('nop')
        self._check_gradient('...->...', ())
        self._check_gradient('...->', ())
        self._check_gradient('->...', ())
        self._check_gradient('a...a->a...', (2, 2))
        self._check_gradient('a...a->', (2, 2))
        self._check_gradient('a...a->...', (2, 5, 1, 2))
        self._check_gradient('a...a->a...', (2, 1, 2))
        self._check_gradient('a...a->a...', (2, 3, 4, 5, 2))
        self._check_gradient('...ijk->...ki', (3, 4, 5))
        self._check_gradient('...ijk->...ki', (1, 3, 4, 5))
        self._check_gradient('...ijk->...ki', (2, 2, 3, 4, 5))
        self._check_gradient('ab...cd->da...', (3, 5, 2, 3, 4, 2))

    def test_binary_simple(self):
        if False:
            return 10
        self._check_gradient(',->', (), ())
        self._check_gradient('a,a->', (3,), (3,))
        self._check_gradient('a,a->a', (3,), (3,))
        self._check_gradient('ab,b->a', (3, 4), (4,))
        self._check_gradient('ab,ab->', (3, 4), (3, 4))
        self._check_gradient('ab,bc->ac', (3, 4), (4, 5))
        self._check_gradient('nij,jk->nik', (5, 2, 3), (3, 4))
        self._check_gradient('abc,bad->abcd', (1, 2, 3), (2, 1, 4))
        self._check_gradient('sa,shb->shab', (2, 1), (2, 3, 4))

    def test_empty(self):
        if False:
            i = 10
            return i + 15
        self._check_gradient('ibnd,ijbn->jnd', (1, 0, 5, 10), (1, 1, 0, 5))

    @test_util.disable_xla('b/131919749')
    def test_reduced_indices(self):
        if False:
            i = 10
            return i + 15
        self._check_gradient('ba,b->', (3, 2), (3,))
        self._check_gradient('ab,ab->', (3, 4), (3, 4))
        self._check_gradient('abce,badf->abcd', (1, 2, 3, 4), (2, 1, 4, 3))

    @test_util.disable_xla('b/131919749')
    def test_repeated_indices(self):
        if False:
            print('Hello World!')
        self._check_gradient('aba,a->b', (3, 4, 3), (3,))
        self._check_gradient('ijj,k->ik', (2, 3, 3), (4,))
        self._check_gradient('ill,k->ik', (2, 3, 3), (4,))
        self._check_gradient('aab,bc->ac', (1, 1, 3), (3, 4))
        self._check_gradient('aab,bcc->ac', (2, 2, 3), (3, 4, 4))

    @test_util.disable_xla('b/131919749')
    def test_empty_with_repeated_indices(self):
        if False:
            return 10
        self._check_gradient('aab,bc->ac', (0, 0, 10), (10, 10))
        self._check_gradient('aab,bc->ac', (1, 1, 0), (0, 10))
        self._check_gradient('aaab,bc->c', (0, 0, 0, 3), (3, 4))

    @test_util.disable_xla('b/131919749')
    def test_broadcasting(self):
        if False:
            for i in range(10):
                print('nop')
        self._check_gradient('...ij,...jk->...ik', (3, 2), (2, 4))
        self._check_gradient('ij...,jk...->ik...', (3, 2, 1), (2, 4))
        self._check_gradient('...ij,...jk->...ik', (3, 1, 3, 2), (1, 5, 2, 4))
        self._check_gradient('ij,jk...k->i...', (3, 2), (2, 4, 1, 4))
        self._check_gradient('aab,b...c->a...c', (1, 1, 3), (3, 1, 1, 4))
        self._check_gradient('...i,...j,...k->...ijk', (1, 4, 1, 2), (5, 1, 1, 3), (1, 1, 1, 1, 9))
        self._check_gradient('...i,...j,...k->...ijk', (1,), (1,), (1,))

    def test_long_cases(self):
        if False:
            print('Hello World!')
        cases = ['abhe,hidj,jgba,hiab,gab->ed', 'ea,fb,abcd,gc,hd->efgh']
        dimension_map = dict(((c, (ord(c) - ord('a')) % 3 + 1) for c in 'abcdefghij'))
        for equation in cases:
            inputs = equation.split('->')[0].replace(' ', '')
            input_shapes = []
            for input_str in inputs.split(','):
                input_shapes.append(tuple([dimension_map[c] for c in input_str]))
            self._check_gradient(equation, *input_shapes)

    @test_util.disable_xla('b/131919749')
    def test_long_cases_with_repeated_labels(self):
        if False:
            for i in range(10):
                print('nop')
        cases = ['fdf,cdd,ccd,afe->ae', 'fff,fae,bef,def->abd']
        dimension_map = dict(((c, (ord(c) - ord('a')) % 3 + 1) for c in 'abcdefghij'))
        for equation in cases:
            inputs = equation.split('->')[0].replace(' ', '')
            input_shapes = []
            for input_str in inputs.split(','):
                input_shapes.append(tuple([dimension_map[c] for c in input_str]))
            self._check_gradient(equation, *input_shapes)

class EinsumBenchmark(test.Benchmark):
    cases = [['ijk->i', 100], ['ijk->kji', 100], ['ij,jk->ik', 500], ['ji,kj->ik', 500], ['bij,bjk->bik', 100], ['bji,bjk->bki', 100], ['ikl,kji->kl', 100], ['klj,lki->ij', 100], ['ijk,ilj->kli', 100], ['ijk,jklm->il', 50], ['efabc,eabcd->efd', 20], ['fabec,abcde->fde', 20], ['efabc,edabc->efd', 20], ['eadbf,dfebc->ecfad', 20], ['abcdef,bcdfg->abcdeg', 20], ['ij,jk,kl->il', 1000], ['ea,fb,abcd,gc,hd->efgh', 10], ['bca,cdb,dbf,afc->', 10], ['efc,dbc,acf,fd->abe', 10], ['abhe,hidj,jgba,hiab,gab->ed', 10]]

    def benchmark_einsum(self):
        if False:
            for i in range(10):
                print('nop')
        for (equation, dim) in self.cases:
            with ops.Graph().as_default(), session.Session(config=benchmark.benchmark_config()) as sess, ops.device('/cpu:0'):
                r = np.random.RandomState(0)
                input_subscripts = equation.split('->')[0].split(',')
                input_vars = []
                for subscript in input_subscripts:
                    input_shape = (dim,) * len(subscript)
                    input_vars.append(variables.Variable(np.array(r.randn(*input_shape), np.float32)))
                self.evaluate(variables.global_variables_initializer())
                if len(input_vars) <= 2:
                    self.run_op_benchmark(sess, special_math_ops.einsum(equation, *input_vars), min_iters=50, name='einsum_cpu_({})_{}'.format(equation, dim))
                else:
                    for optimize in ['greedy', 'auto']:
                        self.run_op_benchmark(sess, special_math_ops.einsum(equation, *input_vars, optimize=optimize), min_iters=50, name='einsum_cpu_({})_{}_{}'.format(equation, optimize, dim))
if __name__ == '__main__':
    test.main()