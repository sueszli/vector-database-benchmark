import pytest
import numpy
import numpy as np
from operator import attrgetter
import cupy
from cupy import testing
from cupy.testing import assert_allclose
from cupy_backends.cuda.api import driver
from cupy_backends.cuda.api import runtime
import cupyx.scipy
import cupyx.scipy.interpolate
try:
    import scipy
    from scipy import interpolate as sc_interpolate
    from scipy import special as sc_special
except ImportError:
    pass
interpolate_cls = ['PPoly', 'BPoly']

@testing.with_requires('scipy')
class TestPPolyCommon:
    """Test basic functionality for PPoly and BPoly."""

    @pytest.mark.parametrize('cls', interpolate_cls)
    def test_sort_check(self, cls):
        if False:
            while True:
                i = 10
        for (xp, scp) in [(numpy, scipy), (cupy, cupyx.scipy)]:
            c = xp.array([[1, 4], [2, 5], [3, 6]])
            x = xp.array([0, 1, 0.5])
            cls1 = getattr(scp.interpolate, cls)
            try:
                cls1(c, x)
            except ValueError:
                pass

    @pytest.mark.parametrize('cls', interpolate_cls)
    def test_ctor_c(self, cls):
        if False:
            i = 10
            return i + 15
        for (xp, scp) in [(numpy, scipy), (cupy, cupyx.scipy)]:
            cls1 = getattr(scp.interpolate, cls)
            try:
                cls1(c=xp.asarray([1, 2]), x=xp.asarray([0, 1]))
            except ValueError:
                pass

    @pytest.mark.parametrize('cls', interpolate_cls)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_extend(self, xp, scp, cls):
        if False:
            return 10
        cls = getattr(scp.interpolate, cls)
        numpy.random.seed(1234)
        order = 3
        x = numpy.unique(numpy.r_[0, 10 * numpy.random.rand(30), 10])
        x = xp.asarray(x)
        c = 2 * numpy.random.rand(order + 1, len(x) - 1, 2, 3) - 1
        c = xp.asarray(c)
        pp = cls(c[:, :9], x[:10])
        pp.extend(c[:, 9:], x[10:])
        pp2 = cls(c[:, 10:], x[10:])
        pp2.extend(c[:, :10], x[:10])
        pp3 = cls(c, x)
        return (pp.c, pp.x, pp2.c, pp2.x, pp3.c, pp3.x)

    @pytest.mark.parametrize('cls', interpolate_cls)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_extend_diff_orders(self, xp, scp, cls):
        if False:
            print('Hello World!')
        cls = getattr(scp.interpolate, cls)
        numpy.random.seed(1234)
        x = xp.linspace(0, 1, 6)
        c = xp.asarray(numpy.random.rand(2, 5))
        x2 = xp.linspace(1, 2, 6)
        c2 = xp.asarray(numpy.random.rand(4, 5))
        pp1 = cls(c, x)
        pp2 = cls(c2, x2)
        pp_comb = cls(c, x)
        pp_comb.extend(c2, x2[1:])
        xi1 = xp.linspace(0, 1, 300, endpoint=False)
        xi2 = xp.linspace(1, 2, 300)
        return (pp1(xi1), pp_comb(xi1), pp2(xi2), pp_comb(xi2))

    @pytest.mark.parametrize('cls', interpolate_cls)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_extend_descending(self, xp, scp, cls):
        if False:
            print('Hello World!')
        cls = getattr(scp.interpolate, cls)
        numpy.random.seed(0)
        order = 3
        x = numpy.sort(numpy.random.uniform(0, 10, 20))
        x = xp.asarray(x)
        c = numpy.random.rand(order + 1, x.shape[0] - 1, 2, 3)
        c = xp.asarray(c)
        p1 = cls(c[:, :9], x[:10])
        p1.extend(c[:, 9:], x[10:])
        p2 = cls(c[:, 10:], x[10:])
        p2.extend(c[:, :10], x[:10])
        return (p1.c, p1.x, p2.c, p2.x)

    @pytest.mark.parametrize('cls', interpolate_cls)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_shape(self, xp, scp, cls):
        if False:
            i = 10
            return i + 15
        cls = getattr(scp.interpolate, cls)
        numpy.random.seed(1234)
        c = numpy.random.rand(8, 12, 5, 6, 7)
        c = xp.asarray(c)
        x = numpy.sort(numpy.random.rand(13))
        x = xp.asarray(x)
        xpts = numpy.random.rand(3, 4)
        xpts = xp.asarray(xpts)
        p = cls(c, x)
        return p(xpts).shape

    @pytest.mark.parametrize('cls', interpolate_cls)
    def test_shape_2(self, cls):
        if False:
            i = 10
            return i + 15
        for (xp, scp) in [(numpy, scipy), (cupy, cupyx.scipy)]:
            cls1 = getattr(scp.interpolate, cls)
            numpy.random.seed(1234)
            c = numpy.random.rand(8, 12, 5, 6, 7)
            c = xp.asarray(c)
            x = numpy.sort(numpy.random.rand(13))
            x = xp.asarray(x)
            p = cls1(c[..., 0, 0, 0], x)
            assert p(0.5).shape == ()
            assert p(xp.array(0.5)).shape == ()
            with pytest.raises(ValueError):
                xxx = xp.array([[0.1, 0.2], [0.4]], dtype=object)
                p(xxx)

    @pytest.mark.parametrize('cls', interpolate_cls)
    @pytest.mark.xfail(runtime.is_hip and driver.get_build_version() < 50000000, reason='Fails on ROCm 4.3')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_complex_coef(self, xp, scp, cls):
        if False:
            i = 10
            return i + 15
        cls = getattr(scp.interpolate, cls)
        numpy.random.seed(12345)
        x = numpy.sort(numpy.random.random(13))
        x = xp.array(x)
        c = numpy.random.random((8, 12)) * (1.0 + 0.3j)
        c = xp.array(c)
        xpt = xp.array(numpy.random.random(5))
        p = cls(c, x)
        return [p(xpt, nu) for nu in [0, 1, 2]]

    @pytest.mark.parametrize('cls', interpolate_cls)
    @pytest.mark.parametrize('axis', [0, 1, 2, 3])
    def test_axis(self, cls, axis):
        if False:
            while True:
                i = 10
        for (xp, scp) in [(numpy, scipy), (cupy, cupyx.scipy)]:
            cls1 = getattr(scp.interpolate, cls)
            numpy.random.seed(12345)
            c = numpy.random.rand(3, 4, 5, 6, 7, 8)
            c = xp.asarray(c)
            c_s = c.shape
            xpt = numpy.random.random((1, 2))
            xpt = xp.asarray(xpt)
            m = c.shape[axis + 1]
            x = numpy.sort(numpy.random.rand(m + 1))
            x = xp.asarray(x)
            p = cls1(c, x, axis=axis)
            assert p.c.shape == c_s[axis:axis + 2] + c_s[:axis] + c_s[axis + 2:]
            res = p(xpt)
            targ_shape = c_s[:axis] + xpt.shape + c_s[2 + axis:]
            assert res.shape == targ_shape
            for p1 in [cls1(c, x, axis=axis).derivative(), cls1(c, x, axis=axis).derivative(2), cls1(c, x, axis=axis).antiderivative(), cls1(c, x, axis=axis).antiderivative(2)]:
                assert p1.axis == p.axis

    @pytest.mark.parametrize('cls', interpolate_cls)
    @pytest.mark.parametrize('axis', [-1, 4, 5, 6])
    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_axis_2(self, xp, scp, cls, axis):
        if False:
            for i in range(10):
                print('nop')
        cls = getattr(scp.interpolate, cls)
        numpy.random.seed(12345)
        c = numpy.random.rand(3, 4, 5, 6, 7, 8)
        c = xp.asarray(c)
        x = numpy.sort(numpy.random.rand(c.shape[-1]))
        x = xp.asarray(x)
        instance = cls(c=c, x=x, axis=axis)
        return (instance.c, instance.x, instance.axis)

@testing.with_requires('scipy')
class TestPPoly:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_simple(self, xp, scp):
        if False:
            i = 10
            return i + 15
        c = xp.array([[1, 4], [2, 5], [3, 6]])
        x = xp.array([0, 0.5, 1])
        p = scp.interpolate.PPoly(c, x)
        return (p(0.3), p(0.7))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_periodic(self, xp, scp):
        if False:
            print('Hello World!')
        c = xp.array([[1, 4], [2, 5], [3, 6]])
        x = xp.array([0, 0.5, 1])
        p = scp.interpolate.PPoly(c, x, extrapolate='periodic')
        return (p(1.3), p(-0.3), p(1.3, 1), p(-0.3, 1))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_read_only(self, xp, scp):
        if False:
            print('Hello World!')
        c = xp.array([[1, 4], [2, 5], [3, 6]])
        x = xp.array([0, 0.5, 1])
        xnew = xp.array([0, 0.1, 0.2])
        scp.interpolate.PPoly(c, x, extrapolate='periodic')
        lst = []
        for writeable in (True, False):
            x.flags.writeable = writeable
            f = scp.interpolate.PPoly(c, x)
            vals = f(xnew)
            assert xp.isfinite(vals).all()
            lst.append(vals)
        return vals

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_multi_shape(self, xp, scp):
        if False:
            while True:
                i = 10
        c = numpy.random.rand(6, 2, 1, 2, 3)
        c = xp.asarray(c)
        x = xp.array([0, 0.5, 1])
        p = scp.interpolate.PPoly(c, x)
        assert p.x.shape == x.shape
        assert p.c.shape == c.shape
        assert p(0.3).shape == c.shape[2:]
        assert p(xp.random.rand(5, 6)).shape == (5, 6) + c.shape[2:]
        dp = p.derivative()
        assert dp.c.shape == (5, 2, 1, 2, 3)
        ip = p.antiderivative()
        assert ip.c.shape == (7, 2, 1, 2, 3)
        return True

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_construct_fast(self, xp, scp):
        if False:
            return 10
        c = xp.array([[1, 4], [2, 5], [3, 6]], dtype=float)
        x = xp.array([0, 0.5, 1])
        p = scp.interpolate.PPoly.construct_fast(c, x)
        return (p(0.3), p(0.7))

    def test_from_spline(self):
        if False:
            for i in range(10):
                print('nop')
        numpy.random.seed(1234)
        x = numpy.sort(numpy.r_[0, numpy.random.rand(11), 1])
        y = numpy.random.rand(len(x))
        spl = sc_interpolate.splrep(x, y, s=0)
        spl = (cupy.asarray(spl[0]), cupy.asarray(spl[1]), spl[2])
        sc_spl = tuple([x.get() if isinstance(x, cupy.ndarray) else x for x in spl])
        pp = cupyx.scipy.interpolate.PPoly.from_spline(spl)
        xi = np.linspace(0, 1, 200)
        testing.assert_allclose(pp(cupy.asarray(xi)), sc_interpolate.splev(xi, sc_spl))
        b = cupyx.scipy.interpolate.BSpline(*spl)
        ppp = cupyx.scipy.interpolate.PPoly.from_spline(b)
        testing.assert_allclose(ppp(xi), b(xi))
        (t, c, k) = spl
        for extrap in (None, True, False):
            b = cupyx.scipy.interpolate.BSpline(t, c, k, extrapolate=extrap)
            p = cupyx.scipy.interpolate.PPoly.from_spline(b)
            assert p.extrapolate == b.extrapolate

    def test_derivative_simple(self):
        if False:
            for i in range(10):
                print('nop')
        c = cupy.array([[4, 3, 2, 1]]).T
        dc = cupy.array([[3 * 4, 2 * 3, 2]]).T
        ddc = cupy.array([[2 * 3 * 4, 1 * 2 * 3]]).T
        x = cupy.array([0, 1])
        pp = cupyx.scipy.interpolate.PPoly(c, x)
        dpp = cupyx.scipy.interpolate.PPoly(dc, x)
        ddpp = cupyx.scipy.interpolate.PPoly(ddc, x)
        testing.assert_allclose(pp.derivative().c, dpp.c)
        testing.assert_allclose(pp.derivative(2).c, ddpp.c)

    def test_derivative_eval(self):
        if False:
            for i in range(10):
                print('nop')
        numpy.random.seed(1234)
        x = numpy.sort(numpy.r_[0, numpy.random.rand(11), 1])
        y = numpy.random.rand(len(x))
        spl = sc_interpolate.splrep(x, y, s=0)
        spl_cupy = (cupy.asarray(spl[0]), cupy.asarray(spl[1]), spl[2])
        pp = cupyx.scipy.interpolate.PPoly.from_spline(spl_cupy)
        xi = cupy.linspace(0, 1, 200)
        for dx in range(0, 3):
            testing.assert_allclose(pp(xi, dx), sc_interpolate.splev(xi.get(), spl, dx))

    def test_derivative(self):
        if False:
            for i in range(10):
                print('nop')
        numpy.random.seed(1234)
        x = numpy.sort(numpy.r_[0, numpy.random.rand(11), 1])
        y = numpy.random.rand(len(x))
        spl = sc_interpolate.splrep(x, y, s=0, k=5)
        spl_cupy = (cupy.asarray(spl[0]), cupy.asarray(spl[1]), spl[2])
        pp = cupyx.scipy.interpolate.PPoly.from_spline(spl_cupy)
        xi = cupy.linspace(0, 1, 200)
        for dx in range(0, 10):
            testing.assert_allclose(pp(xi, dx), pp.derivative(dx)(xi), err_msg='dx=%d' % (dx,))

    def test_antiderivative_of_constant(self):
        if False:
            return 10
        PPoly = cupyx.scipy.interpolate.PPoly
        p = PPoly(cupy.asarray([[1.0]]), cupy.asarray([0, 1]))
        testing.assert_allclose(p.antiderivative().c, PPoly(c=cupy.asarray([[1], [0]]), x=cupy.asarray([0, 1])).c, atol=1e-15)
        testing.assert_allclose(p.antiderivative().x, PPoly(c=cupy.asarray([[1], [0]]), x=cupy.asarray([0, 1])).x, atol=1e-15)

    def test_antiderivative_regression_4355(self):
        if False:
            i = 10
            return i + 15
        PPoly = cupyx.scipy.interpolate.PPoly
        p = PPoly(cupy.asarray([[1.0, 0.5]]), cupy.asarray([0, 1, 2]))
        q = p.antiderivative()
        testing.assert_allclose(q.c, cupy.asarray([[1, 0.5], [0, 1]]))
        testing.assert_allclose(q.x, cupy.asarray([0, 1, 2]))
        testing.assert_allclose(p.integrate(0, 2), 1.5)
        testing.assert_allclose(q(2) - q(0), 1.5)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_antiderivative_simple(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        c = xp.array([[3, 2, 1], [0, 0, 1.6875]]).T
        x = xp.array([0, 0.25, 1])
        pp = scp.interpolate.PPoly(c, x)
        ipp = pp.antiderivative()
        iipp = pp.antiderivative(2)
        iipp2 = ipp.antiderivative()
        return (ipp.x, ipp.c, iipp.x, iipp.c, iipp2.x, iipp2.c)

    def test_antiderivative_vs_derivative(self):
        if False:
            print('Hello World!')
        numpy.random.seed(1234)
        x = numpy.linspace(0, 1, 30) ** 2
        y = numpy.random.rand(len(x))
        spl = sc_interpolate.splrep(x, y, s=0, k=5)
        spl = (cupy.asarray(spl[0]), cupy.asarray(spl[1]), spl[2])
        pp = cupyx.scipy.interpolate.PPoly.from_spline(spl)
        for dx in range(0, 10):
            ipp = pp.antiderivative(dx)
            pp2 = ipp.derivative(dx)
            assert_allclose(pp.c, pp2.c)
            for k in range(dx):
                pp2 = ipp.derivative(k)
                r = 1e-13
                endpoint = r * pp2.x[:-1] + (1 - r) * pp2.x[1:]
                assert_allclose(pp2(pp2.x[1:]), pp2(endpoint), rtol=1e-07, err_msg='dx=%d k=%d' % (dx, k))

    def test_antiderivative_vs_spline(self):
        if False:
            for i in range(10):
                print('nop')
        numpy.random.seed(1234)
        x = numpy.sort(numpy.r_[0, numpy.random.rand(11), 1])
        y = numpy.random.rand(len(x))
        spl = sc_interpolate.splrep(x, y, s=0, k=5)
        spl_cupy = (cupy.asarray(spl[0]), cupy.asarray(spl[1]), spl[2])
        pp = cupyx.scipy.interpolate.PPoly.from_spline(spl_cupy)
        for dx in range(0, 10):
            pp2 = pp.antiderivative(dx)
            spl2 = cupyx.scipy.interpolate.splantider(spl_cupy, dx)
            spl2_np = (spl2[0].get(), spl2[1].get(), spl2[2])
            xi = cupy.linspace(0, 1, 200)
            testing.assert_allclose(pp2(xi), sc_interpolate.splev(xi.get(), spl2_np), rtol=1e-07)

    def test_antiderivative_continuity(self):
        if False:
            while True:
                i = 10
        c = cupy.array([[2, 1, 2, 2], [2, 1, 3, 3]]).T
        x = cupy.array([0, 0.5, 1])
        p = cupyx.scipy.interpolate.PPoly(c, x)
        ip = p.antiderivative()
        testing.assert_allclose(ip(0.5 - 1e-09), ip(0.5 + 1e-09), rtol=1e-08)
        p2 = ip.derivative()
        testing.assert_allclose(p2.c, p.c)

    def test_integrate(self):
        if False:
            while True:
                i = 10
        numpy.random.seed(1234)
        x = numpy.sort(numpy.r_[0, numpy.random.rand(11), 1])
        y = numpy.random.rand(len(x))
        spl = sc_interpolate.splrep(x, y, s=0, k=5)
        cp_spl = tuple([cupy.asarray(x) if isinstance(x, np.ndarray) else x for x in spl])
        pp = cupyx.scipy.interpolate.PPoly.from_spline(cp_spl)
        (a, b) = (0.3, 0.9)
        ig = pp.integrate(a, b)
        testing.assert_allclose(ig, sc_interpolate.splint(a, b, spl))
        (a, b) = (-0.3, 0.9)
        ig = pp.integrate(a, b, extrapolate=True)
        assert cupy.isnan(pp.integrate(a, b, extrapolate=False)).all()

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_integrate_readonly(self, xp, scp):
        if False:
            i = 10
            return i + 15
        x = xp.array([1, 2, 4])
        c = xp.array([[0.0, 0.0], [-1.0, -1.0], [2.0, -0.0], [1.0, 2.0]])
        ret = []
        for writeable in (True, False):
            x.flags.writeable = writeable
            P = scp.interpolate.PPoly(c, x)
            vals = P.integrate(1, 4)
            ret.append(vals)
        return ret

    def test_integrate_periodic(self):
        if False:
            while True:
                i = 10
        x = cupy.array([1, 2, 4])
        c = cupy.array([[0.0, 0.0], [-1.0, -1.0], [2.0, -0.0], [1.0, 2.0]])
        P = cupyx.scipy.interpolate.PPoly(c, x, extrapolate='periodic')
        poly_int = P.antiderivative()
        period_int = poly_int(4) - poly_int(1)
        assert_allclose(P.integrate(1, 4), period_int)
        assert_allclose(P.integrate(-10, -7), period_int)
        assert_allclose(P.integrate(-10, -4), 2 * period_int)
        assert_allclose(P.integrate(1.5, 2.5), poly_int(2.5) - poly_int(1.5))
        assert_allclose(P.integrate(3.5, 5), poly_int(2) - poly_int(1) + poly_int(4) - poly_int(3.5))
        assert_allclose(P.integrate(3.5 + 12, 5 + 12), poly_int(2) - poly_int(1) + poly_int(4) - poly_int(3.5))
        assert_allclose(P.integrate(3.5, 5 + 12), poly_int(2) - poly_int(1) + poly_int(4) - poly_int(3.5) + 4 * period_int)
        assert_allclose(P.integrate(0, -1), poly_int(2) - poly_int(3))
        assert_allclose(P.integrate(-9, -10), poly_int(2) - poly_int(3))
        assert_allclose(P.integrate(0, -10), poly_int(2) - poly_int(3) - 3 * period_int)

    @pytest.mark.skip(reason='There is not an asymmetric eigenvalue solver')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_roots(self, xp, scp):
        if False:
            return 10
        x = xp.linspace(0, 1, 31) ** 2
        y = xp.sin(30 * x)
        if xp is cupy:
            spl = sc_interpolate.splrep(x.get(), y.get(), s=0, k=3)
            spl = (cupy.asarray(spl[0]), cupy.asarray(spl[1]), spl[2])
        else:
            spl = sc_interpolate.splrep(x, y, s=0, k=3)
        pp = scp.interpolate.PPoly.from_spline(spl)
        r = pp.roots()
        return r

    @pytest.mark.skip(reason='There is not an asymmetric eigenvalue solver')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_roots_idzero(self, xp, scp):
        if False:
            while True:
                i = 10
        c = xp.array([[-1, 0.25], [0, 0], [-1, 0.25]]).T
        x = xp.array([0, 0.4, 0.6, 1.0])
        pp = scp.interpolate.PPoly(c, x)
        const = 2.0
        c1 = c.copy()
        c1[1, :] += const
        pp1 = scp.interpolate.PPoly(c1, x)
        return (pp.roots(), pp1.roots())

    @pytest.mark.skip(reason='There is not an asymmetric eigenvalue solver')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_roots_all_zero(self, xp, scp):
        if False:
            while True:
                i = 10
        c = xp.asarray([[0], [0]])
        x = xp.asarray([0, 1])
        p = scp.interpolate.PPoly(c, x)
        return (p.roots(), p.solve(0), p.solve(1))

    @pytest.mark.skip(reason='There is not an asymmetric eigenvalue solver')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_roots_all_zero_1(self, xp, scp):
        if False:
            i = 10
            return i + 15
        c = xp.asarray([[0, 0], [0, 0]])
        x = xp.asarray([0, 1, 2])
        p = scp.interpolate.PPoly(c, x)
        return (p.roots(), p.solve(0), p.solve(1))

    @pytest.mark.skip(reason='There is not an asymmetric eigenvalue solver')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_roots_repeated(self, xp, scp):
        if False:
            while True:
                i = 10
        c = xp.array([[1, 0, -1], [-1, 0, 0]]).T
        x = xp.array([-1, 0, 1])
        pp = scp.interpolate.PPoly(c, x)
        return (pp.roots(), pp.roots(extrapolate=False))

    @pytest.mark.skip(reason='There is not an asymmetric eigenvalue solver')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_roots_discont(self, xp, scp):
        if False:
            i = 10
            return i + 15
        c = xp.array([[1], [-1]]).T
        x = xp.array([0, 0.5, 1])
        pp = scp.interpolate.PPoly(c, x)
        return (pp.roots(), pp.roots(discontinuity=False), pp.solve(0.5), pp.solve(0.5, discontinuity=False), pp.solve(1.5), pp.solve(1.5, discontinuity=False))

    @pytest.mark.skip(reason='There is not an asymmetric eigenvalue solver')
    def test_roots_random(self):
        if False:
            return 10
        numpy.random.seed(1234)
        num = 0
        for extrapolate in (True, False):
            for order in range(0, 20):
                x = numpy.unique(numpy.r_[0, 10 * numpy.random.rand(30), 10])
                x = cupy.asarray(x)
                c = 2 * numpy.random.rand(order + 1, len(x) - 1, 2, 3) - 1
                c = cupy.asarray(c)
                pp = cupyx.scipy.interpolate.PPoly(c, x)
                for y in [0, numpy.random.random()]:
                    r = pp.solve(y, discontinuity=False, extrapolate=extrapolate)
                    for i in range(2):
                        for j in range(3):
                            r1 = r[i]
                            rr = r1[j]
                            if rr.size > 0:
                                num += rr.size
                                val = pp(rr, extrapolate=extrapolate)[:, i, j]
                                cmpval = pp(rr, nu=1, extrapolate=extrapolate)[:, i, j]
                                msg = '(%r) r = %s' % (extrapolate, repr(rr))
                                assert_allclose((val - y) / cmpval, 0, atol=1e-07, err_msg=msg)
        assert num > 100, repr(num)

    @pytest.mark.skip(reason='There is not a complex root solver available')
    def test_roots_croots(self):
        if False:
            i = 10
            return i + 15
        testing.shaped_seed(1234)
        for k in range(1, 15):
            c = testing.shaped_rand(k, 1, 130)
            if k == 3:
                c[:, 0, 0] = (1, 2, 1)
            for y in [0, testing.shaped_random()]:
                w = np.empty(c.shape, dtype=complex)
                if k == 1:
                    assert np.isnan(w).all()
                    continue
                res = 0
                cres = 0
                for i in range(k):
                    res += c[i, None] * w ** (k - 1 - i)
                    cres += abs(c[i, None] * w ** (k - 1 - i))
                with np.errstate(invalid='ignore'):
                    res /= cres
                res = res.ravel()
                res = res[~np.isnan(res)]
                assert_allclose(res, 0, atol=1e-10)

    @pytest.mark.parametrize('extrapolate', [True, False, None])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_extrapolate_attr(self, xp, scp, extrapolate):
        if False:
            while True:
                i = 10
        c = xp.array([[-1, 0, 1]]).T
        x = xp.array([0, 1])
        pp = scp.interpolate.PPoly(c, x, extrapolate=extrapolate)
        pp_d = pp.derivative()
        pp_i = pp.antiderivative()
        xx = xp.asarray([-0.1, 1.1])
        return (pp(xx), pp_i(xx), pp_d(xx))

    def binom_matrix(self, power, xp):
        if False:
            for i in range(10):
                print('nop')
        n = numpy.arange(power + 1).reshape(-1, 1)
        k = numpy.arange(power + 1)
        B = sc_special.binom(n, k)
        if xp is cupy:
            B = cupy.asarray(B)
        return B[::-1, ::-1]

    def _prepare_descending(self, m, xp, scp):
        if False:
            while True:
                i = 10
        power = 3
        x = numpy.sort(numpy.random.uniform(0, 10, m + 1))
        x = xp.asarray(x)
        ca = numpy.random.uniform(-2, 2, size=(power + 1, m))
        ca = xp.asarray(ca)
        h = xp.diff(x)
        h_powers = h[None, :] ** xp.arange(power + 1)[::-1, None]
        B = self.binom_matrix(power, xp)
        cap = ca * h_powers
        cdp = xp.dot(B.T, cap)
        cd = cdp / h_powers
        pa = scp.interpolate.PPoly(ca, x, extrapolate=True)
        pd = scp.interpolate.PPoly(cd[:, ::-1], x[::-1], extrapolate=True)
        return (pa, pd)

    @pytest.mark.parametrize('m', [10, 20, 30])
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13)
    def test_descending(self, m, xp, scp):
        if False:
            return 10
        numpy.random.seed(0)
        (pa, pd) = self._prepare_descending(m, xp, scp)
        x_test = numpy.random.uniform(-10, 20, 100)
        x_test = xp.asarray(x_test)
        return (pa(x_test), pa(x_test, 1))

    @pytest.mark.parametrize('m', [10, 20, 30])
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-13)
    def test_descending_derivative(self, m, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        numpy.random.seed(0)
        (pa, pd) = self._prepare_descending(m, xp, scp)
        pa_d = pa.derivative()
        pd_d = pd.derivative()
        x_test = numpy.random.uniform(-10, 20, 100)
        x_test = xp.asarray(x_test)
        return (pa_d(x_test), pd_d(x_test))

    @pytest.mark.parametrize('m', [10, 20, 30])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_descending_antiderivative(self, m, xp, scp):
        if False:
            i = 10
            return i + 15
        numpy.random.seed(0)
        (pa, pd) = self._prepare_descending(m, xp, scp)
        pa_i = pa.antiderivative()
        pd_i = pd.antiderivative()
        results = []
        for (a, b) in numpy.random.uniform(-10, 20, (5, 2)):
            int_a = pa.integrate(a, b)
            int_d = pd.integrate(a, b)
            results += [int_a, int_d]
            results += [pa_i(b) - pa_i(a), pd_i(b) - pd_i(a)]
        return results

    @pytest.mark.skip(reason='There is not an asymmetric eigenvalue solver')
    @pytest.mark.parametrize('m', [10, 20, 30])
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-12)
    def test_descending_roots(self, m, xp, scp):
        if False:
            print('Hello World!')
        numpy.random.seed(0)
        (pa, pd) = self._prepare_descending(m, xp, scp)
        roots_d = pd.roots()
        roots_a = pa.roots()
        return (roots_a, roots_d)

@testing.with_requires('scipy')
class TestBPoly:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_simple(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        x = xp.asarray([0, 1])
        c = xp.asarray([[3]])
        bp = scp.interpolate.BPoly(c, x)
        return bp(0.1)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_simple2(self, xp, scp):
        if False:
            i = 10
            return i + 15
        x = xp.asarray([0, 1])
        c = xp.asarray([[3], [1]])
        bp = scp.interpolate.BPoly(c, x)
        return bp(0.1)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_simple3(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        x = xp.asarray([0, 1])
        c = xp.asarray([[3], [1], [4]])
        bp = scp.interpolate.BPoly(c, x)
        return bp(0.2)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_simple4(self, xp, scp):
        if False:
            return 10
        x = xp.asarray([0, 1])
        c = xp.asarray([[1], [1], [1], [2]])
        bp = scp.interpolate.BPoly(c, x)
        return bp(0.3)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_simple5(self, xp, scp):
        if False:
            while True:
                i = 10
        x = xp.asarray([0, 1])
        c = xp.asarray([[1], [1], [8], [2], [1]])
        bp = scp.interpolate.BPoly(c, x)
        return bp(0.3)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_periodic(self, xp, scp):
        if False:
            i = 10
            return i + 15
        x = xp.asarray([0, 1, 3])
        c = xp.asarray([[3, 0], [0, 0], [0, 2]])
        bp = scp.interpolate.BPoly(c, x, extrapolate='periodic')
        return (bp(3.4), bp(-1.3), bp(3.4, 1), bp(-1.3, 1))

    @pytest.mark.parametrize('m', [10, 20, 30])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_descending(self, xp, scp, m):
        if False:
            for i in range(10):
                print('nop')
        res = []
        power = 3
        x = xp.sort(testing.shaped_random((m + 1,), xp))
        ca = testing.shaped_random((power + 1, m), xp)
        ca = ca * 2 - 1
        cd = ca[::-1].copy()
        pa = scp.interpolate.BPoly(ca, x, extrapolate=True)
        pd = scp.interpolate.BPoly(cd[:, ::-1], x[::-1], extrapolate=True)
        x_test = testing.shaped_random((100,), xp)
        x_test = x_test * 30 - 10
        res += [pa(x_test), pd(x_test)]
        res += [pa(x_test, 1), pd(x_test, 1)]
        pa_d = pa.derivative()
        pd_d = pd.derivative()
        res += [pa_d(x_test), pd_d(x_test)]
        pa_i = pa.antiderivative()
        pd_i = pd.antiderivative()
        points = testing.shaped_random((5, 2), xp)
        points = points * 30 - 10
        for (a, b) in points:
            int_a = pa.integrate(a, b)
            int_d = pd.integrate(a, b)
            res += [int_a, int_d]
            res += [pa_i(b) - pa_i(a), pd_i(b) - pd_i(a)]
        return res

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_multi_shape(self, xp, scp):
        if False:
            while True:
                i = 10
        c = testing.shaped_random((6, 2, 1, 2, 3), xp)
        x = xp.array([0, 0.5, 1])
        p = scp.interpolate.BPoly(c, x)
        x1 = testing.shaped_random((5, 6), xp)
        dp = p.derivative()
        return (p(x1), dp(x1))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_interval_length(self, xp, scp):
        if False:
            print('Hello World!')
        x = xp.asarray([0, 2])
        c = xp.asarray([[3], [1], [4]])
        bp = scp.interpolate.BPoly(c, x)
        xval = xp.asarray([0.1])
        return bp(xval)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_two_intervals(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        x = xp.asarray([0, 1, 3])
        c = xp.asarray([[3, 0], [0, 0], [0, 2]])
        bp = scp.interpolate.BPoly(c, x)
        return bp(xp.asarray([0.4, 1.7]))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize('extrapolate', [True, False, None])
    def test_extrapolate_attr(self, xp, scp, extrapolate):
        if False:
            i = 10
            return i + 15
        x = xp.asarray([0, 2])
        c = xp.asarray([[3], [1], [4]])
        x1 = xp.asarray([-0.1, 2.1])
        bp = scp.interpolate.BPoly(c, x, extrapolate=extrapolate)
        bp_d = bp.derivative()
        return (bp(x1), bp_d(x1))

@testing.with_requires('scipy')
class TestBPolyCalculus:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_derivative(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        res = []
        x = xp.asarray([0, 1, 3])
        c = xp.asarray([[3, 0], [0, 0], [0, 2]])
        bp = scp.interpolate.BPoly(c, x)
        bp_der = bp.derivative()
        res += [bp_der(0.4), bp_der(1.7)]
        res += [bp(0.4, nu=1), bp(0.4, nu=2), bp(0.4, nu=3)]
        res += [bp(1.7, nu=1), bp(1.7, nu=2), bp(1.7, nu=3)]
        return res

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=0.001)
    def test_derivative_ppoly(self, xp, scp):
        if False:
            i = 10
            return i + 15
        (m, k) = (5, 8)
        x = xp.sort(testing.shaped_random((m,), xp))
        c = testing.shaped_random((k, m - 1), xp)
        bp = scp.interpolate.BPoly(c, x)
        pp = scp.interpolate.PPoly.from_bernstein_basis(bp)
        res = []
        for _ in range(k):
            bp = bp.derivative()
            pp = pp.derivative()
            xi = xp.linspace(x[0], x[-1], 21)
            res += [bp(xi), pp(xi)]
        return res

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=0.001)
    def test_deriv_inplace(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        (m, k) = (5, 8)
        x = xp.sort(testing.shaped_random((m,), xp))
        c = testing.shaped_random((k, m - 1), xp)
        res = []
        for cc in [c.copy(), c * (1.0 + 2j)]:
            if runtime.is_hip and driver.get_build_version() < 50000000 and (cc.dtype.kind == 'c'):
                continue
            bp = scp.interpolate.BPoly(cc, x)
            xi = xp.linspace(x[0], x[-1], 21)
            for i in range(k):
                res += [bp(xi, i), bp.derivative(i)(xi)]
        return res

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_antiderivative_simple(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        x = xp.asarray([0, 1, 3])
        c = xp.asarray([[0, 0], [1, 1]])
        bp = scp.interpolate.BPoly(c, x)
        bi = bp.antiderivative()
        xx = xp.linspace(0, 3, 11)
        return bi(xx)

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=4e-06)
    def test_der_antider(self, xp, scp):
        if False:
            while True:
                i = 10
        x = xp.sort(testing.shaped_random((11,), xp))
        c = testing.shaped_random((4, 10, 2, 3), xp)
        bp = scp.interpolate.BPoly(c, x)
        xx = xp.linspace(x[0], x[-1], 100)
        return (bp.antiderivative().derivative()(xx), bp(xx))

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=3e-07)
    def test_antider_ppoly(self, xp, scp):
        if False:
            while True:
                i = 10
        x = xp.sort(testing.shaped_random((11,), xp))
        c = testing.shaped_random((4, 10, 2, 3), xp)
        bp = scp.interpolate.BPoly(c, x)
        pp = scp.interpolate.PPoly.from_bernstein_basis(bp)
        xx = xp.linspace(x[0], x[-1], 10)
        return (bp.antiderivative(2)(xx), pp.antiderivative(2)(xx))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_antider_continuous(self, xp, scp):
        if False:
            while True:
                i = 10
        x = xp.sort(testing.shaped_random((11,), xp))
        c = testing.shaped_random((4, 10), xp)
        bp = scp.interpolate.BPoly(c, x).antiderivative()
        xx = bp.x[1:-1]
        return (bp(xx - 1e-14), bp(xx + 1e-14))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_integrate(self, xp, scp):
        if False:
            i = 10
            return i + 15
        x = xp.sort(testing.shaped_random((11,), xp))
        c = testing.shaped_random((4, 10), xp)
        bp = scp.interpolate.BPoly(c, x)
        pp = scp.interpolate.PPoly.from_bernstein_basis(bp)
        return (bp.integrate(0, 1), pp.integrate(0, 1))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_integrate_extrap(self, xp, scp):
        if False:
            while True:
                i = 10
        c = xp.asarray([[1]])
        x = xp.asarray([0, 1])
        b = scp.interpolate.BPoly(c, x)
        b1 = scp.interpolate.BPoly(c, x, extrapolate=False)
        return (b.integrate(0, 2), b1.integrate(0, 2), b1.integrate(0, 2, extrapolate=True))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_integrate_periodic(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        res = []
        x = xp.asarray([1, 2, 4])
        c = xp.asarray([[0.0, 0.0], [-1.0, -1.0], [2.0, -0.0], [1.0, 2.0]])
        P = scp.interpolate.BPoly.from_power_basis(scp.interpolate.PPoly(c, x), extrapolate='periodic')
        res += [P.integrate(1, 4), P.integrate(-10, -7), P.integrate(-10, -4), P.integrate(1.5, 2.5), P.integrate(3.5, 5), P.integrate(3.5 + 12, 5 + 12), P.integrate(3.5, 5 + 12), P.integrate(0, -1), P.integrate(-9, -10), P.integrate(0, -10)]
        return res

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_antider_neg(self, xp, scp):
        if False:
            return 10
        c = xp.asarray([[1]])
        x = xp.asarray([0, 1])
        b = scp.interpolate.BPoly(c, x)
        xx = xp.linspace(0, 1, 21)
        return (b.derivative(-1)(xx), b.antiderivative()(xx), b.derivative(1)(xx), b.antiderivative(-1)(xx))

@testing.with_requires('scipy')
class TestPolyConversions:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bp_from_pp(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        x = xp.asarray([0, 1, 3])
        c = xp.asarray([[3, 2], [1, 8], [4, 3]])
        pp = scp.interpolate.PPoly(c, x)
        bp = scp.interpolate.BPoly.from_power_basis(pp)
        pp1 = scp.interpolate.PPoly.from_bernstein_basis(bp)
        x1 = xp.asarray([0.1, 1.4])
        return (pp(x1), bp(x1), pp(x1), pp1(x1))

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=6e-07)
    def test_bp_from_pp_random(self, xp, scp):
        if False:
            print('Hello World!')
        (m, k) = (5, 8)
        x = xp.sort(testing.shaped_random((m,), xp))
        c = testing.shaped_random((k, m - 1), xp)
        pp = scp.interpolate.PPoly(c, x)
        bp = scp.interpolate.BPoly.from_power_basis(pp)
        pp1 = scp.interpolate.PPoly.from_bernstein_basis(bp)
        x1 = xp.linspace(x[0], x[-1], 21)
        return (pp(x1), bp(x1), pp(x1), pp1(x1))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_pp_from_bp(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        x = xp.asarray([0, 1, 3])
        c = xp.asarray([[3, 3], [1, 1], [4, 2]])
        bp = scp.interpolate.BPoly(c, x)
        pp = scp.interpolate.PPoly.from_bernstein_basis(bp)
        bp1 = scp.interpolate.BPoly.from_power_basis(pp)
        x1 = xp.asarray([0.1, 1.4])
        return (bp(x1), pp(x1), bp(x1), bp1(x1))

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=True)
    @pytest.mark.parametrize('comb', [(attrgetter('PPoly'), attrgetter('PPoly.from_bernstein_basis')), (attrgetter('BPoly'), attrgetter('PPoly.from_power_basis'))])
    def test_broken_conversions(self, xp, scp, comb):
        if False:
            i = 10
            return i + 15
        x = xp.asarray([0, 1, 3])
        c = xp.asarray([[3, 3], [1, 1], [4, 2]])
        (get_interp_cls, get_conv_meth) = comb
        pp = get_interp_cls(scp.interpolate)(c, x)
        get_conv_meth(scp.interpolate)(pp)

@testing.with_requires('scipy')
class TestBPolyFromDerivatives:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_make_poly_1(self, xp, scp):
        if False:
            i = 10
            return i + 15
        c1 = scp.interpolate.BPoly._construct_from_derivatives(0, 1, [2], [3])
        return c1

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_make_poly_2(self, xp, scp):
        if False:
            return 10
        c1 = scp.interpolate.BPoly._construct_from_derivatives(0, 1, [1, 0], [1])
        c2 = scp.interpolate.BPoly._construct_from_derivatives(0, 1, [2, 3], [1])
        c3 = scp.interpolate.BPoly._construct_from_derivatives(0, 1, [2], [1, 3])
        return (c1, c2, c3)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_make_poly_3(self, xp, scp):
        if False:
            return 10
        c1 = scp.interpolate.BPoly._construct_from_derivatives(0, 1, [1, 2, 3], [4])
        c2 = scp.interpolate.BPoly._construct_from_derivatives(0, 1, [1], [4, 2, 3])
        c3 = scp.interpolate.BPoly._construct_from_derivatives(0, 1, [1, 2], [4, 3])
        return (c1, c2, c3)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_make_poly_12(self, xp, scp):
        if False:
            while True:
                i = 10
        ya = xp.r_[0, testing.shaped_random((5,), xp)]
        yb = xp.r_[0, testing.shaped_random((5,), xp)]
        c = scp.interpolate.BPoly._construct_from_derivatives(0, 1, ya, yb)
        pp = scp.interpolate.BPoly(c[:, None], xp.asarray([0, 1]))
        res = []
        for _ in range(6):
            res += [pp(0.0), pp(1.0)]
            pp = pp.derivative()
        return res

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_raise_degree(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        x = xp.asarray([0, 1])
        (k, d) = (8, 5)
        c = testing.shaped_random((k, 1, 2, 3, 4), xp)
        bp = scp.interpolate.BPoly(c, x)
        c1 = scp.interpolate.BPoly._raise_degree(c, d)
        bp1 = scp.interpolate.BPoly(c1, x)
        x1 = xp.linspace(0, 1, 11)
        return (bp(x1), bp1(x1))

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_xi_yi(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        scp.interpolate.BPoly.from_derivatives(xp.asarray([0, 1]), xp.asarray([0]))

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_coords_order(self, xp, scp):
        if False:
            print('Hello World!')
        xi = xp.asarray([0, 0, 1])
        yi = xp.asarray([[0], [0], [0]])
        scp.interpolate.BPoly.from_derivatives(xi, yi)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zeros(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        xi = xp.asarray([0, 1, 2, 3])
        yi = [[0, 0], [0], [0, 0], [0, 0]]
        pp = scp.interpolate.BPoly.from_derivatives(xi, yi)
        assert pp.c.shape == (4, 3)
        ppd = pp.derivative()
        res = []
        for x1 in [0.0, 0.1, 1.0, 1.1, 1.9, 2.0, 2.5]:
            res += [pp(x1), ppd(x1)]
        return res

    def _make_random_mk(self, m, k, xp):
        if False:
            i = 10
            return i + 15
        xi = xp.asarray([1.0 * j ** 2 for j in range(m + 1)])
        yi = [testing.shaped_random((k,), xp) for j in range(m + 1)]
        return (xi, yi)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_random_12(self, xp, scp):
        if False:
            while True:
                i = 10
        (m, k) = (5, 12)
        (xi, yi) = self._make_random_mk(m, k, xp)
        pp = scp.interpolate.BPoly.from_derivatives(xi, yi)
        res = []
        for _ in range(k // 2):
            res.append(pp(xi))
            pp = pp.derivative()
        return res

    @testing.numpy_cupy_allclose(scipy_name='scp', accept_error=ValueError)
    def test_order_zero(self, xp, scp):
        if False:
            print('Hello World!')
        (m, k) = (5, 12)
        (xi, yi) = self._make_random_mk(m, k, xp)
        scp.interpolate.BPoly.from_derivatives(xi, yi, 0)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_orders_too_high(self, xp, scp):
        if False:
            while True:
                i = 10
        (m, k) = (5, 12)
        (xi, yi) = self._make_random_mk(m, k, xp)
        scp.interpolate.BPoly.from_derivatives(xi, yi, orders=2 * k - 1)
        with pytest.raises(ValueError):
            scp.interpolate.BPoly.from_derivatives(xi, yi, orders=2 * k)
        return True

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_orders_global(self, xp, scp):
        if False:
            while True:
                i = 10
        (m, k) = (5, 12)
        (xi, yi) = self._make_random_mk(m, k, xp)
        order = 5
        pp = scp.interpolate.BPoly.from_derivatives(xi, yi, orders=order)
        res = []
        for _ in range(order // 2 + 1):
            res += [pp(xi[1:-1] - 1e-12), pp(xi[1:-1] + 1e-12)]
            pp = pp.derivative()
        order = 6
        pp = scp.interpolate.BPoly.from_derivatives(xi, yi, orders=order)
        for _ in range(order // 2):
            res += [pp(xi[1:-1] - 1e-12), pp(xi[1:-1] + 1e-12)]
            pp = pp.derivative()
        return res

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_orders_local(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        (m, k) = (7, 12)
        (xi, yi) = self._make_random_mk(m, k, xp)
        orders = [o + 1 for o in range(m)]
        res = []
        for (i, x) in enumerate(xi[1:-1]):
            pp = scp.interpolate.BPoly.from_derivatives(xi, yi, orders=orders)
            for _ in range(orders[i] // 2 + 1):
                res += [pp(x - 1e-12), pp(x + 1e-12)]
                pp = pp.derivative()
        return res

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_yi_trailing_dims(self, xp, scp):
        if False:
            i = 10
            return i + 15
        (m, k) = (7, 5)
        xi = xp.sort(testing.shaped_random((m + 1,), xp))
        yi = testing.shaped_random((m + 1, k, 6, 7, 8), xp)
        pp = scp.interpolate.BPoly.from_derivatives(xi, yi)
        return pp.c.shape

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_scipy_gh_5430(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        res = []
        orders = xp.int32(1)
        p = scp.interpolate.BPoly.from_derivatives(xp.asarray([0, 1]), [[0], [0]], orders=orders)
        res.append(p(0))
        orders = xp.int64(1)
        p = scp.interpolate.BPoly.from_derivatives(xp.asarray([0, 1]), [[0], [0]], orders=orders)
        res.append(p(0))
        orders = 1
        p = scp.interpolate.BPoly.from_derivatives(xp.asarray([0, 1]), [[0], [0]], orders=orders)
        res.append(p(0))
        orders = 1
        return res

@testing.with_requires('scipy')
class TestNdPPoly:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_simple_1d(self, xp, scp):
        if False:
            return 10
        c = testing.shaped_random((4, 5), xp)
        x = xp.linspace(0, 1, 5 + 1)
        xi = testing.shaped_random((200,), xp)
        p = scp.interpolate.NdPPoly(c, (x,))
        v1 = p((xi,))
        return v1

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_simple_2d(self, xp, scp):
        if False:
            return 10
        c = testing.shaped_random((4, 5, 6, 7), xp)
        x = xp.linspace(0, 1, 6 + 1)
        y = xp.linspace(0, 1, 7 + 1) ** 2
        xi = testing.shaped_random((200,), xp)
        yi = testing.shaped_random((200,), xp)
        p = scp.interpolate.NdPPoly(c, (x, y))
        result = []
        for nu in (None, (0, 0), (0, 1), (1, 0), (2, 3), (9, 2)):
            result.append(p(xp.c_[xi, yi], nu=nu))
        return result

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_simple_3d(self, xp, scp):
        if False:
            print('Hello World!')
        c = testing.shaped_random((4, 5, 6, 7, 8, 9), xp)
        x = xp.linspace(0, 1, 7 + 1)
        y = xp.linspace(0, 1, 8 + 1) ** 2
        z = xp.linspace(0, 1, 9 + 1) ** 3
        xi = testing.shaped_random((40,), xp)
        yi = testing.shaped_random((40,), xp)
        zi = testing.shaped_random((40,), xp)
        p = scp.interpolate.NdPPoly(c, (x, y, z))
        result = []
        for nu in (None, (0, 0, 0), (0, 1, 0), (1, 0, 0), (2, 3, 0), (6, 0, 2)):
            result.append(p((xi, yi, zi), nu=nu))
        return result

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_simple_4d(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        c = testing.shaped_random((4, 5, 6, 7, 8, 9, 10, 11), xp)
        x = xp.linspace(0, 1, 8 + 1)
        y = xp.linspace(0, 1, 9 + 1) ** 2
        z = xp.linspace(0, 1, 10 + 1) ** 3
        u = xp.linspace(0, 1, 11 + 1) ** 4
        xi = testing.shaped_random((20,), xp)
        yi = testing.shaped_random((20,), xp)
        zi = testing.shaped_random((20,), xp)
        ui = testing.shaped_random((20,), xp)
        p = scp.interpolate.NdPPoly(c, (x, y, z, u))
        v1 = p((xi, yi, zi, ui))
        return v1

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_deriv_1d(self, xp, scp):
        if False:
            print('Hello World!')
        c = testing.shaped_random((4, 5), xp)
        x = xp.linspace(0, 1, 5 + 1)
        p = scp.interpolate.NdPPoly(c, (x,))
        dp = p.derivative(nu=[1])
        ip = p.antiderivative(nu=[2])
        return (dp.c, ip.c)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_deriv_3d(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        c = testing.shaped_random((4, 5, 6, 7, 8, 9), xp)
        x = xp.linspace(0, 1, 7 + 1)
        y = xp.linspace(0, 1, 8 + 1) ** 2
        z = xp.linspace(0, 1, 9 + 1) ** 3
        p = scp.interpolate.NdPPoly(c, (x, y, z))
        dpx = p.derivative(nu=[2])
        dpy = p.antiderivative(nu=[0, 1, 0])
        dpz = p.derivative(nu=[0, 0, 3])
        return (dpx.c, dpy.c, dpz.c)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_deriv_3d_simple(self, xp, scp):
        if False:
            return 10
        c = xp.ones((1, 1, 1, 3, 4, 5))
        x = xp.linspace(0, 1, 3 + 1) ** 1
        y = xp.linspace(0, 1, 4 + 1) ** 2
        z = xp.linspace(0, 1, 5 + 1) ** 3
        p = scp.interpolate.NdPPoly(c, (x, y, z))
        ip = p.antiderivative((1, 0, 4))
        ip = ip.antiderivative((0, 2, 0))
        xi = testing.shaped_random((20,), xp)
        yi = testing.shaped_random((20,), xp)
        zi = testing.shaped_random((20,), xp)
        return ip((xi, yi, zi))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_integrate_2d(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        c = testing.shaped_random((4, 5, 16, 17), xp, dtype=xp.float64)
        x = xp.linspace(0, 1, 16 + 1) ** 1
        y = xp.linspace(0, 1, 17 + 1) ** 2
        fix_continuity_mod = attrgetter('interpolate._interpolate._fix_continuity')
        if xp is not cupy:
            fix_continuity_mod = attrgetter('interpolate._ppoly.fix_continuity')
        fix_continuity = fix_continuity_mod(scp)
        c = c.transpose(0, 2, 1, 3)
        cx = c.reshape(c.shape[0], c.shape[1], -1).copy()
        fix_continuity(cx, x, 2)
        c = cx.reshape(c.shape)
        c = c.transpose(0, 2, 1, 3)
        c = c.transpose(1, 3, 0, 2)
        cx = c.reshape(c.shape[0], c.shape[1], -1).copy()
        fix_continuity(cx, y, 2)
        c = cx.reshape(c.shape)
        c = c.transpose(2, 0, 3, 1).copy()
        p = scp.interpolate.NdPPoly(c, (x, y))
        result = []
        for ranges in [[(0, 1), (0, 1)], [(0, 0.5), (0, 1)], [(0, 1), (0, 0.5)], [(0.3, 0.7), (0.6, 0.2)]]:
            ig = p.integrate(ranges)
            result.append(ig)
        return result

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_integrate_1d(self, xp, scp):
        if False:
            return 10
        c = testing.shaped_random((4, 5, 6, 16, 17, 18), xp)
        x = xp.linspace(0, 1, 16 + 1) ** 1
        y = xp.linspace(0, 1, 17 + 1) ** 2
        z = xp.linspace(0, 1, 18 + 1) ** 3
        p = scp.interpolate.NdPPoly(c, (x, y, z))
        u = testing.shaped_random((200,), xp)
        v = testing.shaped_random((200,), xp)
        (a, b) = (0.2, 0.7)
        result = []
        px = p.integrate_1d(a, b, axis=0)
        pax = p.antiderivative((1, 0, 0))
        result += [px((u, v)), pax((b, u, v)) - pax((a, u, v))]
        py = p.integrate_1d(a, b, axis=1)
        pay = p.antiderivative((0, 1, 0))
        result += [py((u, v)), pay((u, b, v)) - pay((u, a, v))]
        pz = p.integrate_1d(a, b, axis=2)
        paz = p.antiderivative((0, 0, 1))
        result += [pz((u, v)), paz((u, v, b)) - paz((u, v, a))]
        return result