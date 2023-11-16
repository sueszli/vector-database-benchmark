import pytest
import numpy as np
from numpy.testing import assert_allclose
import scipy.special as sc

class TestExp1:

    def test_branch_cut(self):
        if False:
            print('Hello World!')
        assert np.isnan(sc.exp1(-1))
        assert sc.exp1(complex(-1, 0)).imag == -sc.exp1(complex(-1, -0.0)).imag
        assert_allclose(sc.exp1(complex(-1, 0)), sc.exp1(-1 + 1e-20j), atol=0, rtol=1e-15)
        assert_allclose(sc.exp1(complex(-1, -0.0)), sc.exp1(-1 - 1e-20j), atol=0, rtol=1e-15)

    def test_834(self):
        if False:
            while True:
                i = 10
        a = sc.exp1(-complex(19.999999))
        b = sc.exp1(-complex(19.9999991))
        assert_allclose(a.imag, b.imag, atol=0, rtol=1e-15)

class TestScaledExp1:

    @pytest.mark.parametrize('x, expected', [(0, 0), (np.inf, 1)])
    def test_limits(self, x, expected):
        if False:
            while True:
                i = 10
        y = sc._ufuncs._scaled_exp1(x)
        assert y == expected

    @pytest.mark.parametrize('x, expected', [(1e-25, 5.698741165994961e-24), (0.1, 0.20146425447084518), (0.9995, 0.5962509885831002), (1.0, 0.5963473623231941), (1.0005, 0.5964436833238044), (2.5, 0.7588145912149602), (10.0, 0.9156333393978808), (100.0, 0.9901942286733019), (500.0, 0.9980079523802055), (1000.0, 0.9990019940238807), (1249.5, 0.9992009578306811), (1250.0, 0.9992012769377913), (1250.25, 0.9992014363957858), (2000.0, 0.9995004992514963), (10000.0, 0.9999000199940024), (10000000000.0, 0.9999999999), (1000000000000000.0, 0.999999999999999)])
    def test_scaled_exp1(self, x, expected):
        if False:
            i = 10
            return i + 15
        y = sc._ufuncs._scaled_exp1(x)
        assert_allclose(y, expected, rtol=2e-15)

class TestExpi:

    @pytest.mark.parametrize('result', [sc.expi(complex(-1, 0)), sc.expi(complex(-1, -0.0)), sc.expi(-1)])
    def test_branch_cut(self, result):
        if False:
            while True:
                i = 10
        desired = -0.21938393439552029
        assert_allclose(result, desired, atol=0, rtol=1e-14)

    def test_near_branch_cut(self):
        if False:
            print('Hello World!')
        lim_from_above = sc.expi(-1 + 1e-20j)
        lim_from_below = sc.expi(-1 - 1e-20j)
        assert_allclose(lim_from_above.real, lim_from_below.real, atol=0, rtol=1e-15)
        assert_allclose(lim_from_above.imag, -lim_from_below.imag, atol=0, rtol=1e-15)

    def test_continuity_on_positive_real_axis(self):
        if False:
            for i in range(10):
                print('nop')
        assert_allclose(sc.expi(complex(1, 0)), sc.expi(complex(1, -0.0)), atol=0, rtol=1e-15)

class TestExpn:

    def test_out_of_domain(self):
        if False:
            for i in range(10):
                print('nop')
        assert all(np.isnan([sc.expn(-1, 1.0), sc.expn(1, -1.0)]))