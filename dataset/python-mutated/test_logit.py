import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_allclose
from scipy.special import logit, expit, log_expit

class TestLogit:

    def check_logit_out(self, dtype, expected):
        if False:
            return 10
        a = np.linspace(0, 1, 10)
        a = np.array(a, dtype=dtype)
        with np.errstate(divide='ignore'):
            actual = logit(a)
        assert_almost_equal(actual, expected)
        assert_equal(actual.dtype, np.dtype(dtype))

    def test_float32(self):
        if False:
            for i in range(10):
                print('nop')
        expected = np.array([-np.inf, -2.07944155, -1.25276291, -0.69314718, -0.22314353, 0.22314365, 0.6931473, 1.25276303, 2.07944155, np.inf], dtype=np.float32)
        self.check_logit_out('f4', expected)

    def test_float64(self):
        if False:
            while True:
                i = 10
        expected = np.array([-np.inf, -2.07944154, -1.25276297, -0.69314718, -0.22314355, 0.22314355, 0.69314718, 1.25276297, 2.07944154, np.inf])
        self.check_logit_out('f8', expected)

    def test_nan(self):
        if False:
            return 10
        expected = np.array([np.nan] * 4)
        with np.errstate(invalid='ignore'):
            actual = logit(np.array([-3.0, -2.0, 2.0, 3.0]))
        assert_equal(expected, actual)

class TestExpit:

    def check_expit_out(self, dtype, expected):
        if False:
            i = 10
            return i + 15
        a = np.linspace(-4, 4, 10)
        a = np.array(a, dtype=dtype)
        actual = expit(a)
        assert_almost_equal(actual, expected)
        assert_equal(actual.dtype, np.dtype(dtype))

    def test_float32(self):
        if False:
            for i in range(10):
                print('nop')
        expected = np.array([0.01798621, 0.04265125, 0.09777259, 0.20860852, 0.39068246, 0.60931754, 0.79139149, 0.9022274, 0.95734876, 0.98201376], dtype=np.float32)
        self.check_expit_out('f4', expected)

    def test_float64(self):
        if False:
            return 10
        expected = np.array([0.01798621, 0.04265125, 0.0977726, 0.20860853, 0.39068246, 0.60931754, 0.79139147, 0.9022274, 0.95734875, 0.98201379])
        self.check_expit_out('f8', expected)

    def test_large(self):
        if False:
            while True:
                i = 10
        for dtype in (np.float32, np.float64, np.longdouble):
            for n in (88, 89, 709, 710, 11356, 11357):
                n = np.array(n, dtype=dtype)
                assert_allclose(expit(n), 1.0, atol=1e-20)
                assert_allclose(expit(-n), 0.0, atol=1e-20)
                assert_equal(expit(n).dtype, dtype)
                assert_equal(expit(-n).dtype, dtype)

class TestLogExpit:

    def test_large_negative(self):
        if False:
            while True:
                i = 10
        x = np.array([-10000.0, -750.0, -500.0, -35.0])
        y = log_expit(x)
        assert_equal(y, x)

    def test_large_positive(self):
        if False:
            i = 10
            return i + 15
        x = np.array([750.0, 1000.0, 10000.0])
        y = log_expit(x)
        assert_equal(y, np.array([-0.0, -0.0, -0.0]))

    def test_basic_float64(self):
        if False:
            print('Hello World!')
        x = np.array([-32, -20, -10, -3, -1, -0.1, -1e-09, 0, 1e-09, 0.1, 1, 10, 100, 500, 710, 725, 735])
        y = log_expit(x)
        expected = [-32.000000000000014, -20.000000002061153, -10.000045398899218, -3.048587351573742, -1.3132616875182228, -0.7443966600735709, -0.6931471810599453, -0.6931471805599453, -0.6931471800599454, -0.6443966600735709, -0.3132616875182228, -4.539889921686465e-05, -3.720075976020836e-44, -7.124576406741286e-218, -4.47628622567513e-309, -1.36930634e-315, -6.217e-320]
        assert_allclose(y, expected, rtol=1e-15)

    def test_basic_float32(self):
        if False:
            while True:
                i = 10
        x = np.array([-32, -20, -10, -3, -1, -0.1, -1e-09, 0, 1e-09, 0.1, 1, 10, 100], dtype=np.float32)
        y = log_expit(x)
        expected = np.array([-32.0, -20.0, -10.000046, -3.0485873, -1.3132616, -0.7443967, -0.6931472, -0.6931472, -0.6931472, -0.64439666, -0.3132617, -4.5398898e-05, -3.8e-44], dtype=np.float32)
        assert_allclose(y, expected, rtol=5e-07)