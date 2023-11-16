import os
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_almost_equal, suppress_warnings
from pytest import raises as assert_raises
import pytest
import scipy.interpolate.interpnd as interpnd
import scipy.spatial._qhull as qhull
import pickle

def data_file(basename):
    if False:
        for i in range(10):
            print('nop')
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', basename)

class TestLinearNDInterpolation:

    def test_smoketest(self):
        if False:
            return 10
        x = np.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.25, 0.3)], dtype=np.float64)
        y = np.arange(x.shape[0], dtype=np.float64)
        yi = interpnd.LinearNDInterpolator(x, y)(x)
        assert_almost_equal(y, yi)

    def test_smoketest_alternate(self):
        if False:
            i = 10
            return i + 15
        x = np.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.25, 0.3)], dtype=np.float64)
        y = np.arange(x.shape[0], dtype=np.float64)
        yi = interpnd.LinearNDInterpolator((x[:, 0], x[:, 1]), y)(x[:, 0], x[:, 1])
        assert_almost_equal(y, yi)

    def test_complex_smoketest(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.25, 0.3)], dtype=np.float64)
        y = np.arange(x.shape[0], dtype=np.float64)
        y = y - 3j * y
        yi = interpnd.LinearNDInterpolator(x, y)(x)
        assert_almost_equal(y, yi)

    def test_tri_input(self):
        if False:
            i = 10
            return i + 15
        x = np.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.25, 0.3)], dtype=np.float64)
        y = np.arange(x.shape[0], dtype=np.float64)
        y = y - 3j * y
        tri = qhull.Delaunay(x)
        yi = interpnd.LinearNDInterpolator(tri, y)(x)
        assert_almost_equal(y, yi)

    def test_square(self):
        if False:
            return 10
        points = np.array([(0, 0), (0, 1), (1, 1), (1, 0)], dtype=np.float64)
        values = np.array([1.0, 2.0, -3.0, 5.0], dtype=np.float64)

        def ip(x, y):
            if False:
                return 10
            t1 = x + y <= 1
            t2 = ~t1
            x1 = x[t1]
            y1 = y[t1]
            x2 = x[t2]
            y2 = y[t2]
            z = 0 * x
            z[t1] = values[0] * (1 - x1 - y1) + values[1] * y1 + values[3] * x1
            z[t2] = values[2] * (x2 + y2 - 1) + values[1] * (1 - x2) + values[3] * (1 - y2)
            return z
        (xx, yy) = np.broadcast_arrays(np.linspace(0, 1, 14)[:, None], np.linspace(0, 1, 14)[None, :])
        xx = xx.ravel()
        yy = yy.ravel()
        xi = np.array([xx, yy]).T.copy()
        zi = interpnd.LinearNDInterpolator(points, values)(xi)
        assert_almost_equal(zi, ip(xx, yy))

    def test_smoketest_rescale(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.array([(0, 0), (-5, -5), (-5, 5), (5, 5), (2.5, 3)], dtype=np.float64)
        y = np.arange(x.shape[0], dtype=np.float64)
        yi = interpnd.LinearNDInterpolator(x, y, rescale=True)(x)
        assert_almost_equal(y, yi)

    def test_square_rescale(self):
        if False:
            return 10
        points = np.array([(0, 0), (0, 100), (10, 100), (10, 0)], dtype=np.float64)
        values = np.array([1.0, 2.0, -3.0, 5.0], dtype=np.float64)
        (xx, yy) = np.broadcast_arrays(np.linspace(0, 10, 14)[:, None], np.linspace(0, 100, 14)[None, :])
        xx = xx.ravel()
        yy = yy.ravel()
        xi = np.array([xx, yy]).T.copy()
        zi = interpnd.LinearNDInterpolator(points, values)(xi)
        zi_rescaled = interpnd.LinearNDInterpolator(points, values, rescale=True)(xi)
        assert_almost_equal(zi, zi_rescaled)

    def test_tripoints_input_rescale(self):
        if False:
            return 10
        x = np.array([(0, 0), (-5, -5), (-5, 5), (5, 5), (2.5, 3)], dtype=np.float64)
        y = np.arange(x.shape[0], dtype=np.float64)
        y = y - 3j * y
        tri = qhull.Delaunay(x)
        yi = interpnd.LinearNDInterpolator(tri.points, y)(x)
        yi_rescale = interpnd.LinearNDInterpolator(tri.points, y, rescale=True)(x)
        assert_almost_equal(yi, yi_rescale)

    def test_tri_input_rescale(self):
        if False:
            while True:
                i = 10
        x = np.array([(0, 0), (-5, -5), (-5, 5), (5, 5), (2.5, 3)], dtype=np.float64)
        y = np.arange(x.shape[0], dtype=np.float64)
        y = y - 3j * y
        tri = qhull.Delaunay(x)
        match = 'Rescaling is not supported when passing a Delaunay triangulation as ``points``.'
        with pytest.raises(ValueError, match=match):
            interpnd.LinearNDInterpolator(tri, y, rescale=True)(x)

    def test_pickle(self):
        if False:
            print('Hello World!')
        np.random.seed(1234)
        x = np.random.rand(30, 2)
        y = np.random.rand(30) + 1j * np.random.rand(30)
        ip = interpnd.LinearNDInterpolator(x, y)
        ip2 = pickle.loads(pickle.dumps(ip))
        assert_almost_equal(ip(0.5, 0.5), ip2(0.5, 0.5))

class TestEstimateGradients2DGlobal:

    def test_smoketest(self):
        if False:
            i = 10
            return i + 15
        x = np.array([(0, 0), (0, 2), (1, 0), (1, 2), (0.25, 0.75), (0.6, 0.8)], dtype=float)
        tri = qhull.Delaunay(x)
        funcs = [(lambda x, y: 0 * x + 1, (0, 0)), (lambda x, y: 0 + x, (1, 0)), (lambda x, y: -2 + y, (0, 1)), (lambda x, y: 3 + 3 * x + 14.15 * y, (3, 14.15))]
        for (j, (func, grad)) in enumerate(funcs):
            z = func(x[:, 0], x[:, 1])
            dz = interpnd.estimate_gradients_2d_global(tri, z, tol=1e-06)
            assert_equal(dz.shape, (6, 2))
            assert_allclose(dz, np.array(grad)[None, :] + 0 * dz, rtol=1e-05, atol=1e-05, err_msg='item %d' % j)

    def test_regression_2359(self):
        if False:
            i = 10
            return i + 15
        points = np.load(data_file('estimate_gradients_hang.npy'))
        values = np.random.rand(points.shape[0])
        tri = qhull.Delaunay(points)
        with suppress_warnings() as sup:
            sup.filter(interpnd.GradientEstimationWarning, 'Gradient estimation did not converge')
            interpnd.estimate_gradients_2d_global(tri, values, maxiter=1)

class TestCloughTocher2DInterpolator:

    def _check_accuracy(self, func, x=None, tol=1e-06, alternate=False, rescale=False, **kw):
        if False:
            i = 10
            return i + 15
        np.random.seed(1234)
        if x is None:
            x = np.array([(0, 0), (0, 1), (1, 0), (1, 1), (0.25, 0.75), (0.6, 0.8), (0.5, 0.2)], dtype=float)
        if not alternate:
            ip = interpnd.CloughTocher2DInterpolator(x, func(x[:, 0], x[:, 1]), tol=1e-06, rescale=rescale)
        else:
            ip = interpnd.CloughTocher2DInterpolator((x[:, 0], x[:, 1]), func(x[:, 0], x[:, 1]), tol=1e-06, rescale=rescale)
        p = np.random.rand(50, 2)
        if not alternate:
            a = ip(p)
        else:
            a = ip(p[:, 0], p[:, 1])
        b = func(p[:, 0], p[:, 1])
        try:
            assert_allclose(a, b, **kw)
        except AssertionError:
            print('_check_accuracy: abs(a-b):', abs(a - b))
            print('ip.grad:', ip.grad)
            raise

    def test_linear_smoketest(self):
        if False:
            while True:
                i = 10
        funcs = [lambda x, y: 0 * x + 1, lambda x, y: 0 + x, lambda x, y: -2 + y, lambda x, y: 3 + 3 * x + 14.15 * y]
        for (j, func) in enumerate(funcs):
            self._check_accuracy(func, tol=1e-13, atol=1e-07, rtol=1e-07, err_msg='Function %d' % j)
            self._check_accuracy(func, tol=1e-13, atol=1e-07, rtol=1e-07, alternate=True, err_msg='Function (alternate) %d' % j)
            self._check_accuracy(func, tol=1e-13, atol=1e-07, rtol=1e-07, err_msg='Function (rescaled) %d' % j, rescale=True)
            self._check_accuracy(func, tol=1e-13, atol=1e-07, rtol=1e-07, alternate=True, rescale=True, err_msg='Function (alternate, rescaled) %d' % j)

    def test_quadratic_smoketest(self):
        if False:
            print('Hello World!')
        funcs = [lambda x, y: x ** 2, lambda x, y: y ** 2, lambda x, y: x ** 2 - y ** 2, lambda x, y: x * y]
        for (j, func) in enumerate(funcs):
            self._check_accuracy(func, tol=1e-09, atol=0.22, rtol=0, err_msg='Function %d' % j)
            self._check_accuracy(func, tol=1e-09, atol=0.22, rtol=0, err_msg='Function %d' % j, rescale=True)

    def test_tri_input(self):
        if False:
            while True:
                i = 10
        x = np.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.25, 0.3)], dtype=np.float64)
        y = np.arange(x.shape[0], dtype=np.float64)
        y = y - 3j * y
        tri = qhull.Delaunay(x)
        yi = interpnd.CloughTocher2DInterpolator(tri, y)(x)
        assert_almost_equal(y, yi)

    def test_tri_input_rescale(self):
        if False:
            print('Hello World!')
        x = np.array([(0, 0), (-5, -5), (-5, 5), (5, 5), (2.5, 3)], dtype=np.float64)
        y = np.arange(x.shape[0], dtype=np.float64)
        y = y - 3j * y
        tri = qhull.Delaunay(x)
        match = 'Rescaling is not supported when passing a Delaunay triangulation as ``points``.'
        with pytest.raises(ValueError, match=match):
            interpnd.CloughTocher2DInterpolator(tri, y, rescale=True)(x)

    def test_tripoints_input_rescale(self):
        if False:
            return 10
        x = np.array([(0, 0), (-5, -5), (-5, 5), (5, 5), (2.5, 3)], dtype=np.float64)
        y = np.arange(x.shape[0], dtype=np.float64)
        y = y - 3j * y
        tri = qhull.Delaunay(x)
        yi = interpnd.CloughTocher2DInterpolator(tri.points, y)(x)
        yi_rescale = interpnd.CloughTocher2DInterpolator(tri.points, y, rescale=True)(x)
        assert_almost_equal(yi, yi_rescale)

    def test_dense(self):
        if False:
            return 10
        funcs = [lambda x, y: x ** 2, lambda x, y: y ** 2, lambda x, y: x ** 2 - y ** 2, lambda x, y: x * y, lambda x, y: np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)]
        np.random.seed(4321)
        grid = np.r_[np.array([(0, 0), (0, 1), (1, 0), (1, 1)], dtype=float), np.random.rand(30 * 30, 2)]
        for (j, func) in enumerate(funcs):
            self._check_accuracy(func, x=grid, tol=1e-09, atol=0.005, rtol=0.01, err_msg='Function %d' % j)
            self._check_accuracy(func, x=grid, tol=1e-09, atol=0.005, rtol=0.01, err_msg='Function %d' % j, rescale=True)

    def test_wrong_ndim(self):
        if False:
            i = 10
            return i + 15
        x = np.random.randn(30, 3)
        y = np.random.randn(30)
        assert_raises(ValueError, interpnd.CloughTocher2DInterpolator, x, y)

    def test_pickle(self):
        if False:
            print('Hello World!')
        np.random.seed(1234)
        x = np.random.rand(30, 2)
        y = np.random.rand(30) + 1j * np.random.rand(30)
        ip = interpnd.CloughTocher2DInterpolator(x, y)
        ip2 = pickle.loads(pickle.dumps(ip))
        assert_almost_equal(ip(0.5, 0.5), ip2(0.5, 0.5))

    def test_boundary_tri_symmetry(self):
        if False:
            for i in range(10):
                print('nop')
        points = np.array([(0, 0), (1, 0), (0.5, np.sqrt(3) / 2)])
        values = np.array([1, 0, 0])
        ip = interpnd.CloughTocher2DInterpolator(points, values)
        ip.grad[...] = 0
        alpha = 0.3
        p1 = np.array([0.5 * np.cos(alpha), 0.5 * np.sin(alpha)])
        p2 = np.array([0.5 * np.cos(np.pi / 3 - alpha), 0.5 * np.sin(np.pi / 3 - alpha)])
        v1 = ip(p1)
        v2 = ip(p2)
        assert_allclose(v1, v2)
        np.random.seed(1)
        A = np.random.randn(2, 2)
        b = np.random.randn(2)
        points = A.dot(points.T).T + b[None, :]
        p1 = A.dot(p1) + b
        p2 = A.dot(p2) + b
        ip = interpnd.CloughTocher2DInterpolator(points, values)
        ip.grad[...] = 0
        w1 = ip(p1)
        w2 = ip(p2)
        assert_allclose(w1, v1)
        assert_allclose(w2, v2)