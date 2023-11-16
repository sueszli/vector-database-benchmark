"""Tests for hermite module.

"""
from functools import reduce
import numpy as np
import numpy.polynomial.hermite as herm
from numpy.polynomial.polynomial import polyval
from numpy.testing import assert_almost_equal, assert_raises, assert_equal, assert_
H0 = np.array([1])
H1 = np.array([0, 2])
H2 = np.array([-2, 0, 4])
H3 = np.array([0, -12, 0, 8])
H4 = np.array([12, 0, -48, 0, 16])
H5 = np.array([0, 120, 0, -160, 0, 32])
H6 = np.array([-120, 0, 720, 0, -480, 0, 64])
H7 = np.array([0, -1680, 0, 3360, 0, -1344, 0, 128])
H8 = np.array([1680, 0, -13440, 0, 13440, 0, -3584, 0, 256])
H9 = np.array([0, 30240, 0, -80640, 0, 48384, 0, -9216, 0, 512])
Hlist = [H0, H1, H2, H3, H4, H5, H6, H7, H8, H9]

def trim(x):
    if False:
        while True:
            i = 10
    return herm.hermtrim(x, tol=1e-06)

class TestConstants:

    def test_hermdomain(self):
        if False:
            i = 10
            return i + 15
        assert_equal(herm.hermdomain, [-1, 1])

    def test_hermzero(self):
        if False:
            while True:
                i = 10
        assert_equal(herm.hermzero, [0])

    def test_hermone(self):
        if False:
            i = 10
            return i + 15
        assert_equal(herm.hermone, [1])

    def test_hermx(self):
        if False:
            while True:
                i = 10
        assert_equal(herm.hermx, [0, 0.5])

class TestArithmetic:
    x = np.linspace(-3, 3, 100)

    def test_hermadd(self):
        if False:
            while True:
                i = 10
        for i in range(5):
            for j in range(5):
                msg = f'At i={i}, j={j}'
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = herm.hermadd([0] * i + [1], [0] * j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_hermsub(self):
        if False:
            while True:
                i = 10
        for i in range(5):
            for j in range(5):
                msg = f'At i={i}, j={j}'
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = herm.hermsub([0] * i + [1], [0] * j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_hermmulx(self):
        if False:
            while True:
                i = 10
        assert_equal(herm.hermmulx([0]), [0])
        assert_equal(herm.hermmulx([1]), [0, 0.5])
        for i in range(1, 5):
            ser = [0] * i + [1]
            tgt = [0] * (i - 1) + [i, 0, 0.5]
            assert_equal(herm.hermmulx(ser), tgt)

    def test_hermmul(self):
        if False:
            while True:
                i = 10
        for i in range(5):
            pol1 = [0] * i + [1]
            val1 = herm.hermval(self.x, pol1)
            for j in range(5):
                msg = f'At i={i}, j={j}'
                pol2 = [0] * j + [1]
                val2 = herm.hermval(self.x, pol2)
                pol3 = herm.hermmul(pol1, pol2)
                val3 = herm.hermval(self.x, pol3)
                assert_(len(pol3) == i + j + 1, msg)
                assert_almost_equal(val3, val1 * val2, err_msg=msg)

    def test_hermdiv(self):
        if False:
            return 10
        for i in range(5):
            for j in range(5):
                msg = f'At i={i}, j={j}'
                ci = [0] * i + [1]
                cj = [0] * j + [1]
                tgt = herm.hermadd(ci, cj)
                (quo, rem) = herm.hermdiv(tgt, ci)
                res = herm.hermadd(herm.hermmul(quo, ci), rem)
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_hermpow(self):
        if False:
            print('Hello World!')
        for i in range(5):
            for j in range(5):
                msg = f'At i={i}, j={j}'
                c = np.arange(i + 1)
                tgt = reduce(herm.hermmul, [c] * j, np.array([1]))
                res = herm.hermpow(c, j)
                assert_equal(trim(res), trim(tgt), err_msg=msg)

class TestEvaluation:
    c1d = np.array([2.5, 1.0, 0.75])
    c2d = np.einsum('i,j->ij', c1d, c1d)
    c3d = np.einsum('i,j,k->ijk', c1d, c1d, c1d)
    x = np.random.random((3, 5)) * 2 - 1
    y = polyval(x, [1.0, 2.0, 3.0])

    def test_hermval(self):
        if False:
            for i in range(10):
                print('nop')
        assert_equal(herm.hermval([], [1]).size, 0)
        x = np.linspace(-1, 1)
        y = [polyval(x, c) for c in Hlist]
        for i in range(10):
            msg = f'At i={i}'
            tgt = y[i]
            res = herm.hermval(x, [0] * i + [1])
            assert_almost_equal(res, tgt, err_msg=msg)
        for i in range(3):
            dims = [2] * i
            x = np.zeros(dims)
            assert_equal(herm.hermval(x, [1]).shape, dims)
            assert_equal(herm.hermval(x, [1, 0]).shape, dims)
            assert_equal(herm.hermval(x, [1, 0, 0]).shape, dims)

    def test_hermval2d(self):
        if False:
            for i in range(10):
                print('nop')
        (x1, x2, x3) = self.x
        (y1, y2, y3) = self.y
        assert_raises(ValueError, herm.hermval2d, x1, x2[:2], self.c2d)
        tgt = y1 * y2
        res = herm.hermval2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)
        z = np.ones((2, 3))
        res = herm.hermval2d(z, z, self.c2d)
        assert_(res.shape == (2, 3))

    def test_hermval3d(self):
        if False:
            for i in range(10):
                print('nop')
        (x1, x2, x3) = self.x
        (y1, y2, y3) = self.y
        assert_raises(ValueError, herm.hermval3d, x1, x2, x3[:2], self.c3d)
        tgt = y1 * y2 * y3
        res = herm.hermval3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)
        z = np.ones((2, 3))
        res = herm.hermval3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3))

    def test_hermgrid2d(self):
        if False:
            i = 10
            return i + 15
        (x1, x2, x3) = self.x
        (y1, y2, y3) = self.y
        tgt = np.einsum('i,j->ij', y1, y2)
        res = herm.hermgrid2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)
        z = np.ones((2, 3))
        res = herm.hermgrid2d(z, z, self.c2d)
        assert_(res.shape == (2, 3) * 2)

    def test_hermgrid3d(self):
        if False:
            while True:
                i = 10
        (x1, x2, x3) = self.x
        (y1, y2, y3) = self.y
        tgt = np.einsum('i,j,k->ijk', y1, y2, y3)
        res = herm.hermgrid3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)
        z = np.ones((2, 3))
        res = herm.hermgrid3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3) * 3)

class TestIntegral:

    def test_hermint(self):
        if False:
            print('Hello World!')
        assert_raises(TypeError, herm.hermint, [0], 0.5)
        assert_raises(ValueError, herm.hermint, [0], -1)
        assert_raises(ValueError, herm.hermint, [0], 1, [0, 0])
        assert_raises(ValueError, herm.hermint, [0], lbnd=[0])
        assert_raises(ValueError, herm.hermint, [0], scl=[0])
        assert_raises(TypeError, herm.hermint, [0], axis=0.5)
        for i in range(2, 5):
            k = [0] * (i - 2) + [1]
            res = herm.hermint([0], m=i, k=k)
            assert_almost_equal(res, [0, 0.5])
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [1 / scl]
            hermpol = herm.poly2herm(pol)
            hermint = herm.hermint(hermpol, m=1, k=[i])
            res = herm.herm2poly(hermint)
            assert_almost_equal(trim(res), trim(tgt))
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            hermpol = herm.poly2herm(pol)
            hermint = herm.hermint(hermpol, m=1, k=[i], lbnd=-1)
            assert_almost_equal(herm.hermval(-1, hermint), i)
        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [2 / scl]
            hermpol = herm.poly2herm(pol)
            hermint = herm.hermint(hermpol, m=1, k=[i], scl=2)
            res = herm.herm2poly(hermint)
            assert_almost_equal(trim(res), trim(tgt))
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = herm.hermint(tgt, m=1)
                res = herm.hermint(pol, m=j)
                assert_almost_equal(trim(res), trim(tgt))
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = herm.hermint(tgt, m=1, k=[k])
                res = herm.hermint(pol, m=j, k=list(range(j)))
                assert_almost_equal(trim(res), trim(tgt))
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = herm.hermint(tgt, m=1, k=[k], lbnd=-1)
                res = herm.hermint(pol, m=j, k=list(range(j)), lbnd=-1)
                assert_almost_equal(trim(res), trim(tgt))
        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = herm.hermint(tgt, m=1, k=[k], scl=2)
                res = herm.hermint(pol, m=j, k=list(range(j)), scl=2)
                assert_almost_equal(trim(res), trim(tgt))

    def test_hermint_axis(self):
        if False:
            return 10
        c2d = np.random.random((3, 4))
        tgt = np.vstack([herm.hermint(c) for c in c2d.T]).T
        res = herm.hermint(c2d, axis=0)
        assert_almost_equal(res, tgt)
        tgt = np.vstack([herm.hermint(c) for c in c2d])
        res = herm.hermint(c2d, axis=1)
        assert_almost_equal(res, tgt)
        tgt = np.vstack([herm.hermint(c, k=3) for c in c2d])
        res = herm.hermint(c2d, k=3, axis=1)
        assert_almost_equal(res, tgt)

class TestDerivative:

    def test_hermder(self):
        if False:
            for i in range(10):
                print('nop')
        assert_raises(TypeError, herm.hermder, [0], 0.5)
        assert_raises(ValueError, herm.hermder, [0], -1)
        for i in range(5):
            tgt = [0] * i + [1]
            res = herm.hermder(tgt, m=0)
            assert_equal(trim(res), trim(tgt))
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = herm.hermder(herm.hermint(tgt, m=j), m=j)
                assert_almost_equal(trim(res), trim(tgt))
        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = herm.hermder(herm.hermint(tgt, m=j, scl=2), m=j, scl=0.5)
                assert_almost_equal(trim(res), trim(tgt))

    def test_hermder_axis(self):
        if False:
            while True:
                i = 10
        c2d = np.random.random((3, 4))
        tgt = np.vstack([herm.hermder(c) for c in c2d.T]).T
        res = herm.hermder(c2d, axis=0)
        assert_almost_equal(res, tgt)
        tgt = np.vstack([herm.hermder(c) for c in c2d])
        res = herm.hermder(c2d, axis=1)
        assert_almost_equal(res, tgt)

class TestVander:
    x = np.random.random((3, 5)) * 2 - 1

    def test_hermvander(self):
        if False:
            print('Hello World!')
        x = np.arange(3)
        v = herm.hermvander(x, 3)
        assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0] * i + [1]
            assert_almost_equal(v[..., i], herm.hermval(x, coef))
        x = np.array([[1, 2], [3, 4], [5, 6]])
        v = herm.hermvander(x, 3)
        assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0] * i + [1]
            assert_almost_equal(v[..., i], herm.hermval(x, coef))

    def test_hermvander2d(self):
        if False:
            i = 10
            return i + 15
        (x1, x2, x3) = self.x
        c = np.random.random((2, 3))
        van = herm.hermvander2d(x1, x2, [1, 2])
        tgt = herm.hermval2d(x1, x2, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)
        van = herm.hermvander2d([x1], [x2], [1, 2])
        assert_(van.shape == (1, 5, 6))

    def test_hermvander3d(self):
        if False:
            for i in range(10):
                print('nop')
        (x1, x2, x3) = self.x
        c = np.random.random((2, 3, 4))
        van = herm.hermvander3d(x1, x2, x3, [1, 2, 3])
        tgt = herm.hermval3d(x1, x2, x3, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)
        van = herm.hermvander3d([x1], [x2], [x3], [1, 2, 3])
        assert_(van.shape == (1, 5, 24))

class TestFitting:

    def test_hermfit(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return x * (x - 1) * (x - 2)

        def f2(x):
            if False:
                i = 10
                return i + 15
            return x ** 4 + x ** 2 + 1
        assert_raises(ValueError, herm.hermfit, [1], [1], -1)
        assert_raises(TypeError, herm.hermfit, [[1]], [1], 0)
        assert_raises(TypeError, herm.hermfit, [], [1], 0)
        assert_raises(TypeError, herm.hermfit, [1], [[[1]]], 0)
        assert_raises(TypeError, herm.hermfit, [1, 2], [1], 0)
        assert_raises(TypeError, herm.hermfit, [1], [1, 2], 0)
        assert_raises(TypeError, herm.hermfit, [1], [1], 0, w=[[1]])
        assert_raises(TypeError, herm.hermfit, [1], [1], 0, w=[1, 1])
        assert_raises(ValueError, herm.hermfit, [1], [1], [-1])
        assert_raises(ValueError, herm.hermfit, [1], [1], [2, -1, 6])
        assert_raises(TypeError, herm.hermfit, [1], [1], [])
        x = np.linspace(0, 2)
        y = f(x)
        coef3 = herm.hermfit(x, y, 3)
        assert_equal(len(coef3), 4)
        assert_almost_equal(herm.hermval(x, coef3), y)
        coef3 = herm.hermfit(x, y, [0, 1, 2, 3])
        assert_equal(len(coef3), 4)
        assert_almost_equal(herm.hermval(x, coef3), y)
        coef4 = herm.hermfit(x, y, 4)
        assert_equal(len(coef4), 5)
        assert_almost_equal(herm.hermval(x, coef4), y)
        coef4 = herm.hermfit(x, y, [0, 1, 2, 3, 4])
        assert_equal(len(coef4), 5)
        assert_almost_equal(herm.hermval(x, coef4), y)
        coef4 = herm.hermfit(x, y, [2, 3, 4, 1, 0])
        assert_equal(len(coef4), 5)
        assert_almost_equal(herm.hermval(x, coef4), y)
        coef2d = herm.hermfit(x, np.array([y, y]).T, 3)
        assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
        coef2d = herm.hermfit(x, np.array([y, y]).T, [0, 1, 2, 3])
        assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
        w = np.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        y[0::2] = 0
        wcoef3 = herm.hermfit(x, yw, 3, w=w)
        assert_almost_equal(wcoef3, coef3)
        wcoef3 = herm.hermfit(x, yw, [0, 1, 2, 3], w=w)
        assert_almost_equal(wcoef3, coef3)
        wcoef2d = herm.hermfit(x, np.array([yw, yw]).T, 3, w=w)
        assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
        wcoef2d = herm.hermfit(x, np.array([yw, yw]).T, [0, 1, 2, 3], w=w)
        assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
        x = [1, 1j, -1, -1j]
        assert_almost_equal(herm.hermfit(x, x, 1), [0, 0.5])
        assert_almost_equal(herm.hermfit(x, x, [0, 1]), [0, 0.5])
        x = np.linspace(-1, 1)
        y = f2(x)
        coef1 = herm.hermfit(x, y, 4)
        assert_almost_equal(herm.hermval(x, coef1), y)
        coef2 = herm.hermfit(x, y, [0, 2, 4])
        assert_almost_equal(herm.hermval(x, coef2), y)
        assert_almost_equal(coef1, coef2)

class TestCompanion:

    def test_raises(self):
        if False:
            return 10
        assert_raises(ValueError, herm.hermcompanion, [])
        assert_raises(ValueError, herm.hermcompanion, [1])

    def test_dimensions(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(1, 5):
            coef = [0] * i + [1]
            assert_(herm.hermcompanion(coef).shape == (i, i))

    def test_linear_root(self):
        if False:
            print('Hello World!')
        assert_(herm.hermcompanion([1, 2])[0, 0] == -0.25)

class TestGauss:

    def test_100(self):
        if False:
            while True:
                i = 10
        (x, w) = herm.hermgauss(100)
        v = herm.hermvander(x, 99)
        vv = np.dot(v.T * w, v)
        vd = 1 / np.sqrt(vv.diagonal())
        vv = vd[:, None] * vv * vd
        assert_almost_equal(vv, np.eye(100))
        tgt = np.sqrt(np.pi)
        assert_almost_equal(w.sum(), tgt)

class TestMisc:

    def test_hermfromroots(self):
        if False:
            print('Hello World!')
        res = herm.hermfromroots([])
        assert_almost_equal(trim(res), [1])
        for i in range(1, 5):
            roots = np.cos(np.linspace(-np.pi, 0, 2 * i + 1)[1::2])
            pol = herm.hermfromroots(roots)
            res = herm.hermval(roots, pol)
            tgt = 0
            assert_(len(pol) == i + 1)
            assert_almost_equal(herm.herm2poly(pol)[-1], 1)
            assert_almost_equal(res, tgt)

    def test_hermroots(self):
        if False:
            return 10
        assert_almost_equal(herm.hermroots([1]), [])
        assert_almost_equal(herm.hermroots([1, 1]), [-0.5])
        for i in range(2, 5):
            tgt = np.linspace(-1, 1, i)
            res = herm.hermroots(herm.hermfromroots(tgt))
            assert_almost_equal(trim(res), trim(tgt))

    def test_hermtrim(self):
        if False:
            for i in range(10):
                print('nop')
        coef = [2, -1, 1, 0]
        assert_raises(ValueError, herm.hermtrim, coef, -1)
        assert_equal(herm.hermtrim(coef), coef[:-1])
        assert_equal(herm.hermtrim(coef, 1), coef[:-3])
        assert_equal(herm.hermtrim(coef, 2), [0])

    def test_hermline(self):
        if False:
            print('Hello World!')
        assert_equal(herm.hermline(3, 4), [3, 2])

    def test_herm2poly(self):
        if False:
            while True:
                i = 10
        for i in range(10):
            assert_almost_equal(herm.herm2poly([0] * i + [1]), Hlist[i])

    def test_poly2herm(self):
        if False:
            i = 10
            return i + 15
        for i in range(10):
            assert_almost_equal(herm.poly2herm(Hlist[i]), [0] * i + [1])

    def test_weight(self):
        if False:
            print('Hello World!')
        x = np.linspace(-5, 5, 11)
        tgt = np.exp(-x ** 2)
        res = herm.hermweight(x)
        assert_almost_equal(res, tgt)