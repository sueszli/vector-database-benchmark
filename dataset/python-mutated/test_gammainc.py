import cupy as cp
import pytest
import cupyx.scipy.special as sc
from cupy.testing import assert_allclose, assert_array_equal
INVALID_POINTS = [(1, -1), (0, 0), (-1, 1), (cp.nan, 1), (1, cp.nan)]

class TestGammainc(object):

    @pytest.mark.skipif(cp.cuda.runtime.is_hip and cp.cuda.runtime.runtimeGetVersion() < 50000000, reason='ROCm/HIP fails in ROCm 4.x')
    @pytest.mark.parametrize('a, x', INVALID_POINTS)
    def test_domain(self, a, x):
        if False:
            print('Hello World!')
        assert cp.isnan(sc.gammainc(a, x))

    def test_a_eq_0_x_gt_0(self):
        if False:
            while True:
                i = 10
        assert sc.gammainc(0, 1) == 1

    @pytest.mark.parametrize('a, x, desired', [(cp.inf, 1, 0), (cp.inf, 0, 0), (cp.inf, cp.inf, cp.nan), (1, cp.inf, 1)])
    def test_infinite_arguments(self, a, x, desired):
        if False:
            while True:
                i = 10
        result = sc.gammainc(a, x)
        if cp.isnan(desired):
            assert cp.isnan(result)
        else:
            assert result == desired

    @pytest.mark.skipif(cp.cuda.runtime.is_hip and cp.cuda.runtime.runtimeGetVersion() < 50000000, reason='ROCm/HIP fails in ROCm 4.x')
    def test_infinite_limits(self):
        if False:
            for i in range(10):
                print('nop')
        assert_allclose(sc.gammainc(1000, 100), sc.gammainc(cp.inf, 100), atol=1e-200, rtol=0)
        assert sc.gammainc(100, 1000) == sc.gammainc(100, cp.inf)

    def test_x_zero(self):
        if False:
            i = 10
            return i + 15
        a = cp.arange(1, 10)
        assert_array_equal(sc.gammainc(a, 0), 0)

    @pytest.mark.skipif(cp.cuda.runtime.is_hip and cp.cuda.runtime.runtimeGetVersion() < 50000000, reason='ROCm/HIP fails in ROCm 4.x')
    def test_limit_check(self):
        if False:
            return 10
        result = sc.gammainc(1e-10, 1)
        limit = sc.gammainc(0, 1)
        assert cp.isclose(result, limit)

    def gammainc_line(self, x):
        if False:
            for i in range(10):
                print('nop')
        c = cp.asarray([-1 / 3, -1 / 540, 25 / 6048, 101 / 155520, -3184811 / 3695155200, -2745493 / 8151736420])
        res = 0
        xfac = 1
        for ck in c:
            res -= ck * xfac
            xfac /= x
        res /= cp.sqrt(2 * cp.pi * x)
        res += 0.5
        return res

    @pytest.mark.skipif(cp.cuda.runtime.is_hip and cp.cuda.runtime.runtimeGetVersion() < 50000000, reason='ROCm/HIP fails in ROCm 4.x')
    def test_roundtrip(self):
        if False:
            i = 10
            return i + 15
        a = cp.logspace(-5, 10, 100)
        x = cp.logspace(-5, 10, 100)
        y = sc.gammaincinv(a, sc.gammainc(a, x))
        assert_allclose(x, y, rtol=1e-10)

class TestGammaincc(object):

    @pytest.mark.parametrize('a, x', INVALID_POINTS)
    def test_domain(self, a, x):
        if False:
            print('Hello World!')
        assert cp.isnan(sc.gammaincc(a, x))

    def test_a_eq_0_x_gt_0(self):
        if False:
            print('Hello World!')
        assert sc.gammaincc(0, 1) == 0

    @pytest.mark.parametrize('a, x, desired', [(cp.inf, 1, 1), (cp.inf, 0, 1), (cp.inf, cp.inf, cp.nan), (1, cp.inf, 0)])
    def test_infinite_arguments(self, a, x, desired):
        if False:
            for i in range(10):
                print('nop')
        result = sc.gammaincc(a, x)
        if cp.isnan(desired):
            assert cp.isnan(result)
        else:
            assert result == desired

    @pytest.mark.skipif(cp.cuda.runtime.is_hip and cp.cuda.runtime.runtimeGetVersion() < 50000000, reason='ROCm/HIP fails in ROCm 4.x')
    def test_infinite_limits(self):
        if False:
            for i in range(10):
                print('nop')
        assert sc.gammaincc(1000, 100) == sc.gammaincc(cp.inf, 100)
        assert_allclose(sc.gammaincc(100, 1000), sc.gammaincc(100, cp.inf), atol=1e-200, rtol=0)

    def test_limit_check(self):
        if False:
            print('Hello World!')
        result = sc.gammaincc(1e-10, 1)
        limit = sc.gammaincc(0, 1)
        assert cp.isclose(result, limit)

    def test_x_zero(self):
        if False:
            print('Hello World!')
        a = cp.arange(1, 10)
        assert_array_equal(sc.gammaincc(a, 0), 1)

    @pytest.mark.skipif(cp.cuda.runtime.is_hip and cp.cuda.runtime.runtimeGetVersion() < 50000000, reason='ROCm/HIP fails in ROCm 4.x')
    def test_roundtrip(self):
        if False:
            print('Hello World!')
        a = cp.logspace(-5, 10, 100)
        x = cp.logspace(-5, 10, 100)
        y = sc.gammainccinv(a, sc.gammaincc(a, x))
        assert_allclose(x, y, rtol=1e-14)