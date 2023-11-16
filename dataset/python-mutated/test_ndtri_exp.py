import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from scipy.special import log_ndtr, ndtri_exp
from scipy.special._testutils import assert_func_equal

def log_ndtr_ndtri_exp(y):
    if False:
        for i in range(10):
            print('nop')
    return log_ndtr(ndtri_exp(y))

@pytest.fixture(scope='class')
def uniform_random_points():
    if False:
        print('Hello World!')
    random_state = np.random.RandomState(1234)
    points = random_state.random_sample(1000)
    return points

class TestNdtriExp:
    """Tests that ndtri_exp is sufficiently close to an inverse of log_ndtr.

    We have separate tests for the five intervals (-inf, -10),
    [-10, -2), [-2, -0.14542), [-0.14542, -1e-6), and [-1e-6, 0).
    ndtri_exp(y) is computed in three different ways depending on if y
    is in (-inf, -2), [-2, log(1 - exp(-2))], or [log(1 - exp(-2), 0).
    Each of these intervals is given its own test with two additional tests
    for handling very small values and values very close to zero.
    """

    @pytest.mark.parametrize('test_input', [-10.0, -100.0, -10000000000.0, -1e+20, -np.finfo(float).max])
    def test_very_small_arg(self, test_input, uniform_random_points):
        if False:
            return 10
        scale = test_input
        points = scale * (0.5 * uniform_random_points + 0.5)
        assert_func_equal(log_ndtr_ndtri_exp, lambda y: y, points, rtol=1e-14, nan_ok=True)

    @pytest.mark.parametrize('interval,expected_rtol', [((-10, -2), 1e-14), ((-2, -0.14542), 1e-12), ((-0.14542, -1e-06), 1e-10), ((-1e-06, 0), 1e-06)])
    def test_in_interval(self, interval, expected_rtol, uniform_random_points):
        if False:
            while True:
                i = 10
        (left, right) = interval
        points = (right - left) * uniform_random_points + left
        assert_func_equal(log_ndtr_ndtri_exp, lambda y: y, points, rtol=expected_rtol, nan_ok=True)

    def test_extreme(self):
        if False:
            return 10
        bigneg = np.nextafter.reduce([np.finfo(float).min, 0, 0, 0, 0])
        tinyneg = -np.finfo(float).tiny
        x = np.array([tinyneg, bigneg])
        result = log_ndtr_ndtri_exp(x)
        assert_allclose(result, x, rtol=1e-12)

    def test_asymptotes(self):
        if False:
            print('Hello World!')
        assert_equal(ndtri_exp([-np.inf, 0.0]), [-np.inf, np.inf])

    def test_outside_domain(self):
        if False:
            while True:
                i = 10
        assert np.isnan(ndtri_exp(1.0))