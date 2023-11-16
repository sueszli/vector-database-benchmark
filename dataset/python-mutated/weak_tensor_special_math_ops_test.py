"""Tests for tensorflow.python.ops.special_math_ops on WeakTensor."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework.weak_tensor import WeakTensor
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import weak_tensor_ops
from tensorflow.python.ops import weak_tensor_test_util
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
_get_weak_tensor = weak_tensor_test_util.get_weak_tensor

@test_util.run_all_in_graph_and_eager_modes
class DawsnTest(test.TestCase, parameterized.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def test_dawsn_boundary(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertAllClose(0.0, special_math_ops.dawsn(0.0))
        self.assertTrue(np.isnan(self.evaluate(special_math_ops.dawsn(np.nan))))

    @parameterized.parameters(np.float32, np.float64)
    def test_dawsn_odd(self, dtype):
        if False:
            print('Hello World!')
        x = _get_weak_tensor(np.random.uniform(-100.0, 100.0, size=int(10000.0)).astype(dtype))
        y = special_math_ops.dawsn(x)
        neg_y = -special_math_ops.dawsn(-x)
        self.assertIsInstance(y, WeakTensor)
        self.assertIsInstance(neg_y, WeakTensor)
        self.assertAllClose(self.evaluate(y), self.evaluate(neg_y))

    @parameterized.parameters(np.float32, np.float64)
    def test_dawsn_small(self, dtype):
        if False:
            while True:
                i = 10
        x = np.random.uniform(-1.0, 1.0, size=int(10000.0)).astype(dtype)
        x_wt = _get_weak_tensor(x)
        y_wt = special_math_ops.dawsn(x_wt)
        self.assertIsInstance(y_wt, WeakTensor)
        try:
            from scipy import special
            self.assertAllClose(special.dawsn(x), self.evaluate(y_wt))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    @parameterized.parameters(np.float32, np.float64)
    def test_dawsn_larger(self, dtype):
        if False:
            i = 10
            return i + 15
        x = np.random.uniform(1.0, 100.0, size=int(10000.0)).astype(dtype)
        x_wt = _get_weak_tensor(x)
        y_wt = special_math_ops.dawsn(x_wt)
        self.assertIsInstance(y_wt, WeakTensor)
        try:
            from scipy import special
            self.assertAllClose(special.dawsn(x), y_wt)
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    def test_dawsn_gradient(self):
        if False:
            print('Hello World!')
        inputs = [_get_weak_tensor(np.random.uniform(-50.0, 50.0, size=int(100.0)))]
        (analytical, numerical) = gradient_checker_v2.compute_gradient(special_math_ops.dawsn, inputs)
        self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.0001)

@test_util.run_all_in_graph_and_eager_modes
class ExpintTest(test.TestCase, parameterized.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def test_expint_boundary(self):
        if False:
            print('Hello World!')
        self.assertAllClose(-np.inf, special_math_ops.expint(0.0))
        self.assertTrue(np.isnan(self.evaluate(special_math_ops.expint(np.nan))))
        self.assertTrue(np.all(np.isnan(self.evaluate(special_math_ops.expint(np.random.uniform(-20.0, -1.0, size=int(1000.0)))))))

    @parameterized.parameters(np.float32, np.float64)
    def test_expint_small(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        x = np.random.uniform(0.0, 1.0, size=int(10000.0)).astype(dtype)
        x_wt = _get_weak_tensor(x)
        y_wt = special_math_ops.expint(x_wt)
        self.assertIsInstance(y_wt, WeakTensor)
        try:
            from scipy import special
            self.assertAllClose(special.expi(x), self.evaluate(special_math_ops.expint(x_wt)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    @parameterized.parameters(np.float32, np.float64)
    def test_expint_larger(self, dtype):
        if False:
            return 10
        x = np.random.uniform(1.0, 50.0, size=int(10000.0)).astype(dtype)
        x_wt = _get_weak_tensor(x)
        try:
            from scipy import special
            self.assertAllClose(special.expi(x), self.evaluate(special_math_ops.expint(x_wt)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    def test_expint_gradient(self):
        if False:
            return 10
        inputs = [_get_weak_tensor(np.random.uniform(1.0, 10.0, size=int(100.0)))]
        (analytical, numerical) = gradient_checker_v2.compute_gradient(special_math_ops.expint, inputs)
        self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.005)

@test_util.run_all_in_graph_and_eager_modes
class FresnelCosTest(test.TestCase, parameterized.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def test_fresnel_cos_boundary(self):
        if False:
            return 10
        self.assertAllClose(0.0, special_math_ops.fresnel_cos(0.0))
        self.assertTrue(np.isnan(self.evaluate(special_math_ops.fresnel_cos(np.nan))))

    @parameterized.parameters(np.float32, np.float64)
    def test_fresnel_cos_odd(self, dtype):
        if False:
            while True:
                i = 10
        x = _get_weak_tensor(np.random.uniform(-100.0, 100.0, size=int(10000.0)).astype(dtype))
        y = special_math_ops.fresnel_cos(x)
        neg_y = -special_math_ops.fresnel_cos(-x)
        self.assertIsInstance(y, WeakTensor)
        self.assertIsInstance(neg_y, WeakTensor)
        self.assertAllClose(self.evaluate(y), self.evaluate(neg_y))

    @parameterized.parameters(np.float32, np.float64)
    def test_fresnel_cos_small(self, dtype):
        if False:
            while True:
                i = 10
        x = np.random.uniform(0.0, 1.0, size=int(10000.0)).astype(dtype)
        x_wt = _get_weak_tensor(x)
        y_wt = special_math_ops.fresnel_cos(x_wt)
        self.assertIsInstance(y_wt, WeakTensor)
        try:
            from scipy import special
            self.assertAllClose(special.fresnel(x)[1], self.evaluate(y_wt))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    @parameterized.parameters(np.float32, np.float64)
    def test_fresnel_cos_larger(self, dtype):
        if False:
            i = 10
            return i + 15
        x = np.random.uniform(1.0, 100.0, size=int(10000.0)).astype(dtype)
        x_wt = _get_weak_tensor(x)
        try:
            from scipy import special
            self.assertAllClose(special.fresnel(x)[1], self.evaluate(special_math_ops.fresnel_cos(x_wt)), rtol=1e-05)
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    def test_fresnel_cos_gradient(self):
        if False:
            for i in range(10):
                print('nop')
        inputs = [_get_weak_tensor(np.random.uniform(1.0, 50.0, size=int(100.0)))]
        (analytical, numerical) = gradient_checker_v2.compute_gradient(special_math_ops.fresnel_cos, inputs)
        self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.005)

@test_util.run_all_in_graph_and_eager_modes
class FresnelSinTest(test.TestCase, parameterized.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def test_fresnel_sin_boundary(self):
        if False:
            return 10
        self.assertAllClose(0.0, special_math_ops.fresnel_sin(0.0))
        self.assertTrue(np.isnan(self.evaluate(special_math_ops.fresnel_sin(np.nan))))

    @parameterized.parameters(np.float32, np.float64)
    def test_fresnel_sin_odd(self, dtype):
        if False:
            print('Hello World!')
        x = _get_weak_tensor(np.random.uniform(-100.0, 100.0, size=int(10000.0)).astype(dtype))
        y = special_math_ops.fresnel_sin(x)
        neg_y = -special_math_ops.fresnel_sin(-x)
        self.assertIsInstance(y, WeakTensor)
        self.assertIsInstance(neg_y, WeakTensor)
        self.assertAllClose(y, neg_y)

    @parameterized.parameters(np.float32, np.float64)
    def test_fresnel_sin_small(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        x = np.random.uniform(0.0, 1.0, size=int(10000.0)).astype(dtype)
        x_wt = _get_weak_tensor(x)
        try:
            from scipy import special
            self.assertAllClose(special.fresnel(x)[0], self.evaluate(special_math_ops.fresnel_sin(x_wt)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    @parameterized.parameters(np.float32, np.float64)
    def test_fresnel_sin_larger(self, dtype):
        if False:
            return 10
        x = np.random.uniform(1.0, 100.0, size=int(10000.0)).astype(dtype)
        x_wt = _get_weak_tensor(x)
        try:
            from scipy import special
            self.assertAllClose(special.fresnel(x)[0], self.evaluate(special_math_ops.fresnel_sin(x_wt)), rtol=1e-05)
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    def test_fresnel_sin_gradient(self):
        if False:
            for i in range(10):
                print('nop')
        inputs = [_get_weak_tensor(np.random.uniform(1.0, 50.0, size=int(100.0)))]
        (analytical, numerical) = gradient_checker_v2.compute_gradient(special_math_ops.fresnel_sin, inputs)
        self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.005)

@test_util.run_all_in_graph_and_eager_modes
class SpenceTest(test.TestCase, parameterized.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def test_spence_boundary(self):
        if False:
            i = 10
            return i + 15
        self.assertAllClose(np.pi ** 2 / 6.0, special_math_ops.spence(0.0))
        self.assertAllClose(0.0, special_math_ops.spence(1.0))
        self.assertTrue(np.isnan(self.evaluate(special_math_ops.spence(np.nan))))
        self.assertTrue(np.all(np.isnan(self.evaluate(special_math_ops.spence(np.random.uniform(-20.0, -1.0, size=int(1000.0)))))))

    @parameterized.parameters(np.float32, np.float64)
    def test_spence_small(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        x = np.random.uniform(0.0, 1.0, size=int(10000.0)).astype(dtype)
        x_wt = _get_weak_tensor(x)
        y_wt = special_math_ops.spence(x_wt)
        self.assertIsInstance(y_wt, WeakTensor)
        try:
            from scipy import special
            self.assertAllClose(special.spence(x), self.evaluate(y_wt))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    @parameterized.parameters(np.float32, np.float64)
    def test_spence_larger(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        x = np.random.uniform(1.0, 100.0, size=int(10000.0)).astype(dtype)
        x_wt = _get_weak_tensor(x)
        try:
            from scipy import special
            self.assertAllClose(special.spence(x), self.evaluate(special_math_ops.spence(x_wt)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    def test_spence_gradient(self):
        if False:
            i = 10
            return i + 15
        inputs = [_get_weak_tensor(np.random.uniform(1.0, 50.0, size=int(100.0)))]
        (analytical, numerical) = gradient_checker_v2.compute_gradient(special_math_ops.spence, inputs)
        self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.0001)

    def test_spence_gradient_at_one(self):
        if False:
            i = 10
            return i + 15
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
            for i in range(10):
                print('nop')
        self.assertAllClose(1.0, special_math_ops.bessel_j0(0.0))
        self.assertAllClose(0.0, special_math_ops.bessel_j1(0.0))
        self.assertTrue(np.isnan(self.evaluate(special_math_ops.bessel_j0(np.nan))))
        self.assertTrue(np.isnan(self.evaluate(special_math_ops.bessel_j1(np.nan))))

    @test_util.run_in_graph_and_eager_modes
    def test_besselk_boundary(self):
        if False:
            i = 10
            return i + 15
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
            print('Hello World!')
        x = _get_weak_tensor(np.random.uniform(-100.0, 100.0, size=int(10000.0)).astype(dtype))
        self.assertAllClose(self.evaluate(special_math_ops.bessel_i0(x)), self.evaluate(special_math_ops.bessel_i0(-x)))
        self.assertAllClose(self.evaluate(special_math_ops.bessel_i0e(x)), self.evaluate(special_math_ops.bessel_i0e(-x)))
        self.assertAllClose(self.evaluate(special_math_ops.bessel_j0(x)), self.evaluate(special_math_ops.bessel_j0(-x)))

    @parameterized.parameters(np.float32, np.float64)
    def test_i1j1_odd(self, dtype):
        if False:
            print('Hello World!')
        x = _get_weak_tensor(np.random.uniform(-100.0, 100.0, size=int(10000.0)).astype(dtype))
        self.assertAllClose(self.evaluate(special_math_ops.bessel_i1(x)), self.evaluate(-special_math_ops.bessel_i1(-x)))
        self.assertAllClose(self.evaluate(special_math_ops.bessel_i1e(x)), self.evaluate(-special_math_ops.bessel_i1e(-x)))
        self.assertAllClose(self.evaluate(special_math_ops.bessel_j1(x)), self.evaluate(-special_math_ops.bessel_j1(-x)))

    @parameterized.parameters(np.float32, np.float64)
    def test_besseli_small(self, dtype):
        if False:
            i = 10
            return i + 15
        x = np.random.uniform(-1.0, 1.0, size=int(10000.0)).astype(dtype)
        x_wt = _get_weak_tensor(x)
        try:
            from scipy import special
            self.assertAllClose(special.i0(x), self.evaluate(special_math_ops.bessel_i0(x_wt)))
            self.assertAllClose(special.i1(x), self.evaluate(special_math_ops.bessel_i1(x_wt)))
            self.assertAllClose(special.i0e(x), self.evaluate(special_math_ops.bessel_i0e(x_wt)))
            self.assertAllClose(special.i1e(x), self.evaluate(special_math_ops.bessel_i1e(x_wt)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    @parameterized.parameters(np.float32, np.float64)
    def test_besselj_small(self, dtype):
        if False:
            print('Hello World!')
        x = np.random.uniform(-1.0, 1.0, size=int(10000.0)).astype(dtype)
        x_wt = _get_weak_tensor(x)
        try:
            from scipy import special
            self.assertAllClose(special.j0(x), self.evaluate(special_math_ops.bessel_j0(x_wt)))
            self.assertAllClose(special.j1(x), self.evaluate(special_math_ops.bessel_j1(x_wt)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    @parameterized.parameters(np.float32, np.float64)
    def test_besselk_small(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        x = np.random.uniform(np.finfo(dtype).eps, 1.0, size=int(10000.0)).astype(dtype)
        x_wt = _get_weak_tensor(x)
        try:
            from scipy import special
            self.assertAllClose(special.k0(x), self.evaluate(special_math_ops.bessel_k0(x_wt)))
            self.assertAllClose(special.k0e(x), self.evaluate(special_math_ops.bessel_k0e(x_wt)))
            self.assertAllClose(special.k1(x), self.evaluate(special_math_ops.bessel_k1(x_wt)))
            self.assertAllClose(special.k1e(x), self.evaluate(special_math_ops.bessel_k1e(x_wt)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    @parameterized.parameters(np.float32, np.float64)
    def test_bessely_small(self, dtype):
        if False:
            print('Hello World!')
        x = np.random.uniform(np.finfo(dtype).eps, 1.0, size=int(10000.0)).astype(dtype)
        x_wt = _get_weak_tensor(x)
        try:
            from scipy import special
            self.assertAllClose(special.y0(x), self.evaluate(special_math_ops.bessel_y0(x_wt)))
            self.assertAllClose(special.y1(x), self.evaluate(special_math_ops.bessel_y1(x_wt)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    @parameterized.parameters(np.float32, np.float64)
    def test_besseli_larger(self, dtype):
        if False:
            return 10
        x = np.random.uniform(1.0, 20.0, size=int(10000.0)).astype(dtype)
        x_wt = _get_weak_tensor(x)
        try:
            from scipy import special
            self.assertAllClose(special.i0e(x), self.evaluate(special_math_ops.bessel_i0e(x_wt)))
            self.assertAllClose(special.i1e(x), self.evaluate(special_math_ops.bessel_i1e(x_wt)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    @parameterized.parameters(np.float32, np.float64)
    def test_besselj_larger(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        x = np.random.uniform(1.0, 30.0, size=int(10000.0)).astype(dtype)
        x_wt = _get_weak_tensor(x)
        try:
            from scipy import special
            self.assertAllClose(special.j0(x), self.evaluate(special_math_ops.bessel_j0(x_wt)))
            self.assertAllClose(special.j1(x), self.evaluate(special_math_ops.bessel_j1(x_wt)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    @parameterized.parameters(np.float32, np.float64)
    def test_besselk_larger(self, dtype):
        if False:
            return 10
        x = np.random.uniform(1.0, 30.0, size=int(10000.0)).astype(dtype)
        x_wt = _get_weak_tensor(x)
        try:
            from scipy import special
            self.assertAllClose(special.k0(x), self.evaluate(special_math_ops.bessel_k0(x_wt)))
            self.assertAllClose(special.k0e(x), self.evaluate(special_math_ops.bessel_k0e(x_wt)))
            self.assertAllClose(special.k1(x), self.evaluate(special_math_ops.bessel_k1(x_wt)))
            self.assertAllClose(special.k1e(x), self.evaluate(special_math_ops.bessel_k1e(x_wt)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    @parameterized.parameters(np.float32, np.float64)
    def test_bessely_larger(self, dtype):
        if False:
            return 10
        x = np.random.uniform(1.0, 30.0, size=int(10000.0)).astype(dtype)
        x_wt = _get_weak_tensor(x)
        try:
            from scipy import special
            self.assertAllClose(special.y0(x), self.evaluate(special_math_ops.bessel_y0(x_wt)))
            self.assertAllClose(special.y1(x), self.evaluate(special_math_ops.bessel_y1(x_wt)))
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))

    def test_besseli_gradient(self):
        if False:
            return 10
        inputs = [_get_weak_tensor(np.random.uniform(-10.0, 10.0, size=int(100.0)))]
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
            while True:
                i = 10
        inputs = [_get_weak_tensor(np.random.uniform(-50.0, 50.0, size=int(100.0)))]
        (analytical, numerical) = gradient_checker_v2.compute_gradient(special_math_ops.bessel_j0, inputs)
        self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.0001)
        (analytical, numerical) = gradient_checker_v2.compute_gradient(special_math_ops.bessel_j1, inputs)
        self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.0001)

    def test_besselk_gradient(self):
        if False:
            while True:
                i = 10
        inputs = [_get_weak_tensor(np.random.uniform(1.0, 50.0, size=int(100.0)))]
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
            while True:
                i = 10
        inputs = [_get_weak_tensor(np.random.uniform(1.0, 50.0, size=int(100.0)))]
        (analytical, numerical) = gradient_checker_v2.compute_gradient(special_math_ops.bessel_y0, inputs)
        self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.0001)
        (analytical, numerical) = gradient_checker_v2.compute_gradient(special_math_ops.bessel_y1, inputs)
        self.assertLess(gradient_checker_v2.max_error(analytical, numerical), 0.0001)
if __name__ == '__main__':
    ops.set_dtype_conversion_mode('all')
    test.main()