import pytest
import cupy
from cupy import testing
from cupy.cuda import runtime
import numpy as _np
import cupyx.scipy.interpolate as csi
try:
    from scipy import interpolate
except ImportError:
    pass

@pytest.mark.skipif(runtime.is_hip, reason='csrlsvqr not available')
@testing.with_requires('scipy')
class TestInterp:

    def get_xy(self, xp):
        if False:
            print('Hello World!')
        xx = xp.linspace(0.0, 2.0 * cupy.pi, 11)
        yy = xp.sin(xx)
        return (xx, yy)

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=TypeError)
    def test_non_int_order(self, xp, scp):
        if False:
            i = 10
            return i + 15
        (x, y) = self.get_xy(xp)
        return scp.interpolate.make_interp_spline(x, y, k=2.5)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_order_0(self, xp, scp):
        if False:
            while True:
                i = 10
        (x, y) = self.get_xy(xp)
        return (scp.interpolate.make_interp_spline(x, y, k=0)(x), scp.interpolate.make_interp_spline(x, y, k=0, axis=-1)(x))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_linear(self, xp, scp):
        if False:
            while True:
                i = 10
        (x, y) = self.get_xy(xp)
        return (scp.interpolate.make_interp_spline(x, y, k=1)(x), scp.interpolate.make_interp_spline(x, y, k=1, axis=-1)(x))

    @testing.with_requires('scipy >= 1.10')
    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    @pytest.mark.parametrize('k', [0, 1, 2, 3])
    def test_incompatible_x_y(self, xp, scp, k):
        if False:
            return 10
        x = [0, 1, 2, 3, 4, 5]
        y = [0, 1, 2, 3, 4, 5, 6, 7]
        scp.interpolate.make_interp_spline(x, y, k=k)

    @testing.with_requires('scipy >= 1.10')
    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    @pytest.mark.parametrize('k', [0, 1, 2, 3])
    def test_broken_x(self, xp, scp, k):
        if False:
            print('Hello World!')
        x = [0, 1, 1, 2, 3, 4]
        y = [0, 1, 2, 3, 4, 5]
        scp.interpolate.make_interp_spline(x, y, k=k)

    @testing.with_requires('scipy >= 1.10')
    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    @pytest.mark.parametrize('k', [0, 1, 2, 3])
    def test_broken_x_2(self, xp, scp, k):
        if False:
            for i in range(10):
                print('nop')
        x = [0, 2, 1, 3, 4, 5]
        y = [0, 1, 2, 3, 4, 5]
        scp.interpolate.make_interp_spline(x, y, k=k)

    @testing.with_requires('scipy >= 1.10')
    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    @pytest.mark.parametrize('k', [0, 1, 2, 3])
    def test_broken_x_3(self, xp, scp, k):
        if False:
            i = 10
            return i + 15
        x = xp.asarray([0, 1, 2, 3, 4, 5]).reshape((1, -1))
        y = [0, 1, 2, 3, 4, 5]
        scp.interpolate.make_interp_spline(x, y, k=1)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    @pytest.mark.parametrize('k', [3, 5])
    def test_not_a_knot(self, xp, scp, k):
        if False:
            i = 10
            return i + 15
        (x, y) = self.get_xy(xp)
        return scp.interpolate.make_interp_spline(x, y, k=k)(x)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    @pytest.mark.parametrize('k', [0, 1, 3, 5])
    def test_int_xy(self, xp, scp, k):
        if False:
            for i in range(10):
                print('nop')
        x = xp.arange(10).astype(int)
        y = xp.arange(10).astype(int)
        return scp.interpolate.make_interp_spline(x, y, k=k)(x)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize('k', [0, 1, 2, 3])
    def test_sliced_input(self, xp, scp, k):
        if False:
            print('Hello World!')
        xx = xp.linspace(-1, 1, 100)
        x = xx[::5]
        y = xx[::5]
        return scp.interpolate.make_interp_spline(x, y, k=k)(x)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    @pytest.mark.parametrize('k', [1, 2, 3, 5])
    def test_list_input(self, xp, scp, k):
        if False:
            print('Hello World!')
        x = list(range(10))
        y = [a ** 2 for a in x]
        return scp.interpolate.make_interp_spline(x, y, k=k)(x)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_quadratic_deriv_right(self, xp, scp):
        if False:
            return 10
        (x, y) = self.get_xy(xp)
        der = [(1, 8.0)]
        b = scp.interpolate.make_interp_spline(x, y, k=2, bc_type=(None, der))
        return (b(x), b(x[-1], 1))

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_quadratic_deriv_left(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        (x, y) = self.get_xy(xp)
        der = [(1, 8.0)]
        b = scp.interpolate.make_interp_spline(x, y, k=2, bc_type=(der, None))
        return (b(x), b(x[0], 1))

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_cubic_deriv_deriv(self, xp, scp):
        if False:
            i = 10
            return i + 15
        (x, y) = self.get_xy(xp)
        (der_l, der_r) = ([(1, 3.0)], [(1, 4.0)])
        b = scp.interpolate.make_interp_spline(x, y, k=3, bc_type=(der_l, der_r))
        return (b(x), b(x[0], 1), b(x[-1], 1))

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_cubic_deriv_natural(self, xp, scp):
        if False:
            return 10
        (x, y) = self.get_xy(xp)
        (der_l, der_r) = ([(2, 0)], [(2, 0)])
        b = scp.interpolate.make_interp_spline(x, y, k=3, bc_type=(der_l, der_r))
        return (b(x), b(x[0], 2), b(x[-1], 2))

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_quintic_derivs(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        (k, n) = (5, 7)
        x = xp.arange(n).astype(xp.float_)
        y = xp.sin(x)
        der_l = [(1, -12.0), (2, 1)]
        der_r = [(1, 8.0), (2, 3.0)]
        b = scp.interpolate.make_interp_spline(x, y, k=k, bc_type=(der_l, der_r))
        return (b(x), b(x[0], 1), b(x[0], 2), b(x[-1], 1), b(x[-1], 2))

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_knots_not_data_sites(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        k = 2
        (x, y) = self.get_xy(xp)
        t = xp.r_[(x[0],) * (k + 1), (x[1:] + x[:-1]) / 2.0, (x[-1],) * (k + 1)]
        b = scp.interpolate.make_interp_spline(x, y, k, t, bc_type=([(2, 0)], [(2, 0)]))
        return (b(x), b(x[0], 2), b(x[-1], 2))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_minimum_points_and_deriv(self, xp, scp):
        if False:
            while True:
                i = 10
        k = 3
        x = [0.0, 1.0]
        y = [0.0, 1.0]
        b = scp.interpolate.make_interp_spline(x, y, k, bc_type=([(1, 0.0)], [(1, 3.0)]))
        xx = xp.linspace(0.0, 1.0)
        return b(xx)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_complex(self, xp, scp):
        if False:
            i = 10
            return i + 15
        (x, y) = self.get_xy(xp)
        y = y + 1j * y
        (der_l, der_r) = ([(1, 3j)], [(1, 4.0 + 2j)])
        b = scp.interpolate.make_interp_spline(x, y, k=3, bc_type=(der_l, der_r))
        return (b(x), b(x[0], 1), b(x[-1], 1))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_complex_01(self, xp, scp):
        if False:
            print('Hello World!')
        (x, y) = self.get_xy(xp)
        y = y + 1j * y
        return (scp.interpolate.make_interp_spline(x, y, k=0)(x), scp.interpolate.make_interp_spline(x, y, k=1)(x))

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_multiple_rhs(self, xp, scp):
        if False:
            i = 10
            return i + 15
        (x, _) = self.get_xy(xp)
        yy = xp.c_[xp.sin(x), xp.cos(x)]
        der_l = [(1, [1.0, 2.0])]
        der_r = [(1, [3.0, 4.0])]
        b = scp.interpolate.make_interp_spline(x, yy, k=3, bc_type=(der_l, der_r))
        return (b(x), b(x[0], 1), b(x[-1], 1))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_shapes(self, xp, scp):
        if False:
            print('Hello World!')
        xp.random.seed(1234)
        (k, n) = (3, 22)
        x = xp.sort(xp.random.random(size=n))
        y = xp.random.random(size=(n, 5, 6, 7))
        b1 = scp.interpolate.make_interp_spline(x, y, k)
        assert b1.c.shape == (n, 5, 6, 7)
        d_l = [(1, xp.random.random((5, 6, 7)))]
        d_r = [(1, xp.random.random((5, 6, 7)))]
        b2 = scp.interpolate.make_interp_spline(x, y, k, bc_type=(d_l, d_r))
        assert b2.c.shape == (n + k - 1, 5, 6, 7)
        return b1.c.shape + b2.c.shape

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_string_aliases_1(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        (x, _) = self.get_xy(xp)
        y = xp.sin(x)
        b1 = scp.interpolate.make_interp_spline(x, y, k=3, bc_type='natural')
        b2 = scp.interpolate.make_interp_spline(x, y, k=3, bc_type=([(2, 0)], [(2, 0)]))
        return (b1.c, b2.c)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_string_aliases_2(self, xp, scp):
        if False:
            while True:
                i = 10
        (x, _) = self.get_xy(xp)
        y = xp.sin(x)
        b1 = scp.interpolate.make_interp_spline(x, y, k=3, bc_type=('natural', 'clamped'))
        b2 = scp.interpolate.make_interp_spline(x, y, k=3, bc_type=([(2, 0)], [(1, 0)]))
        return (b1.c, b2.c)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_string_aliases_3(self, xp, scp):
        if False:
            i = 10
            return i + 15
        (x, _) = self.get_xy(xp)
        y = xp.sin(x)
        b1 = scp.interpolate.make_interp_spline(x, y, k=2, bc_type=(None, 'clamped'))
        b2 = scp.interpolate.make_interp_spline(x, y, k=2, bc_type=(None, [(1, 0.0)]))
        return (b1.c, b2.c)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_string_aliases_4(self, xp, scp):
        if False:
            i = 10
            return i + 15
        (x, _) = self.get_xy(xp)
        y = xp.sin(x)
        b1 = scp.interpolate.make_interp_spline(x, y, k=3, bc_type='not-a-knot')
        b2 = scp.interpolate.make_interp_spline(x, y, k=3, bc_type=None)
        return (b1.c, b2.c)

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_string_aliases_5(self, xp, scp):
        if False:
            i = 10
            return i + 15
        (x, _) = self.get_xy(xp)
        y = xp.sin(x)
        scp.interpolate.make_interp_spline(x, y, k=3, bc_type='typo')

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_string_aliases_6(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        (x, _) = self.get_xy(xp)
        yy = xp.c_[xp.sin(x), xp.cos(x)]
        der_l = [(1, [0.0, 0.0])]
        der_r = [(2, [0.0, 0.0])]
        b2 = scp.interpolate.make_interp_spline(x, yy, k=3, bc_type=(der_l, der_r))
        b1 = scp.interpolate.make_interp_spline(x, yy, k=3, bc_type=('clamped', 'natural'))
        return (b1.c, b2.c)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_string_aliases_7(self, xp, scp):
        if False:
            while True:
                i = 10
        rng = _np.random.RandomState(1234)
        (k, n) = (3, 22)
        x = _np.sort(rng.uniform(size=n))
        y = rng.uniform(size=(n, 5, 6, 7))
        if xp is cupy:
            x = cupy.asarray(x)
            y = cupy.asarray(y)
        d_l = [(1, xp.zeros((5, 6, 7)))]
        d_r = [(1, xp.zeros((5, 6, 7)))]
        b1 = scp.interpolate.make_interp_spline(x, y, k, bc_type=(d_l, d_r))
        b2 = scp.interpolate.make_interp_spline(x, y, k, bc_type='clamped')
        return (b1.c, b2.c)

    def test_deriv_spec(self):
        if False:
            i = 10
            return i + 15
        x = y = [1.0, 2, 3, 4, 5, 6]
        with pytest.raises(ValueError):
            csi.make_interp_spline(x, y, bc_type=([(1, 0.0)], None))
        with pytest.raises(ValueError):
            csi.make_interp_spline(x, y, bc_type=(1, 0.0))
        with pytest.raises(ValueError):
            csi.make_interp_spline(x, y, bc_type=[(1, 0.0)])
        with pytest.raises(ValueError):
            csi.make_interp_spline(x, y, bc_type=42)
        (l, r) = ((1, 0.0), (1, 0.0))
        with pytest.raises(ValueError):
            csi.make_interp_spline(x, y, bc_type=(l, r))

    def test_full_matrix(self):
        if False:
            print('Hello World!')
        from cupyx.scipy.interpolate._bspline2 import _make_interp_spline_full_matrix
        cupy.random.seed(1234)
        (k, n) = (3, 7)
        x = cupy.sort(cupy.random.random(size=n))
        y = cupy.random.random(size=n)
        b = csi.make_interp_spline(x, y, k=3)
        bf = _make_interp_spline_full_matrix(x, y, k, b.t, bc_type=None)
        cupy.testing.assert_allclose(b.c, bf.c, atol=1e-14, rtol=1e-14)
        b = csi.make_interp_spline(x, y, k=3, bc_type='natural')
        bf = _make_interp_spline_full_matrix(x, y, k, b.t, bc_type='natural')
        cupy.testing.assert_allclose(b.c, bf.c, atol=1e-13)

@testing.with_requires('scipy>=1.7')
@pytest.mark.skipif(runtime.is_hip, reason='csrlsvqr not available')
class TestInterpPeriodic:

    def get_xy(self, xp):
        if False:
            i = 10
            return i + 15
        xx = xp.linspace(0.0, 2.0 * cupy.pi, 11)
        yy = xp.sin(xx)
        return (xx, yy)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_periodic(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        (x, y) = self.get_xy(xp)
        b = scp.interpolate.make_interp_spline(x, y, k=5, bc_type='periodic')
        for i in range(1, 5):
            xp.testing.assert_allclose(b(x[0], nu=i), b(x[-1], nu=i), atol=1e-11)
        return b(x)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_periodic_axis1(self, xp, scp):
        if False:
            return 10
        (x, y) = self.get_xy(xp)
        b = scp.interpolate.make_interp_spline(x, y, k=5, bc_type='periodic', axis=-1)
        for i in range(1, 5):
            xp.testing.assert_allclose(b(x[0], nu=i), b(x[-1], nu=i), atol=1e-11)
        return b(x)

    @pytest.mark.parametrize('k', [2, 3, 4, 5, 6, 7])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_periodic_random(self, xp, scp, k):
        if False:
            print('Hello World!')
        n = 15
        _np.random.seed(1234)
        x = _np.sort(_np.random.random_sample(n) * 10)
        y = _np.random.random_sample(n) * 100
        if xp is cupy:
            x = cupy.asarray(x)
            y = cupy.asarray(y)
        y[0] = y[-1]
        b = scp.interpolate.make_interp_spline(x, y, k=k, bc_type='periodic')
        return b(x)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_periodic_axis(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        (x, y) = self.get_xy(xp)
        n = x.shape[0]
        _np.random.seed(1234)
        x = _np.random.random_sample(n) * 2 * _np.pi
        x = _np.sort(x)
        if xp is cupy:
            x = cupy.asarray(x)
        x[0] = 0.0
        x[-1] = 2 * xp.pi
        y = xp.zeros((2, n))
        y[0] = xp.sin(x)
        y[1] = xp.cos(x)
        b = scp.interpolate.make_interp_spline(x, y, k=5, bc_type='periodic', axis=1)
        return b(x)

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_periodic_points_exception(self, xp, scp):
        if False:
            return 10
        (n, k) = (8, 5)
        x = xp.linspace(0, n, n)
        y = x
        return scp.interpolate.make_interp_spline(x, y, k=k, bc_type='periodic')

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_periodic_knots_exception(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        (n, k) = (7, 3)
        x = xp.linspace(0, n, n)
        y = x ** 2
        t = xp.zeros(n + 2 * k)
        return scp.interpolate.make_interp_spline(x, y, k, t, 'periodic')

    def test_periodic_cubic(self):
        if False:
            i = 10
            return i + 15
        n = 3
        cupy.random.seed(1234)
        x = cupy.sort(cupy.random.random_sample(n) * 10)
        y = cupy.random.random_sample(n) * 100
        y[0] = y[-1]
        b = csi.make_interp_spline(x, y, k=3, bc_type='periodic')
        cub = interpolate.CubicSpline(x.get(), y.get(), bc_type='periodic')
        cupy.testing.assert_allclose(b(x), cub(x.get()), atol=1e-14)