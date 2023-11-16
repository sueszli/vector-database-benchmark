"""
Test SciPy functions versus mpmath, if available.

"""
import numpy as np
from numpy.testing import assert_, assert_allclose
from numpy import pi
import pytest
import itertools
from scipy._lib import _pep440
import scipy.special as sc
from scipy.special._testutils import MissingModule, check_version, FuncData, assert_func_equal
from scipy.special._mptestutils import Arg, FixedArg, ComplexArg, IntArg, assert_mpmath_equal, nonfunctional_tooslow, trace_args, time_limited, exception_to_nan, inf_to_nan
from scipy.special._ufuncs import _sinpi, _cospi, _lgam1p, _lanczos_sum_expg_scaled, _log1pmx, _igam_fac
try:
    import mpmath
except ImportError:
    mpmath = MissingModule('mpmath')

@check_version(mpmath, '0.10')
def test_expi_complex():
    if False:
        print('Hello World!')
    dataset = []
    for r in np.logspace(-99, 2, 10):
        for p in np.linspace(0, 2 * np.pi, 30):
            z = r * np.exp(1j * p)
            dataset.append((z, complex(mpmath.ei(z))))
    dataset = np.array(dataset, dtype=np.cdouble)
    FuncData(sc.expi, dataset, 0, 1).check()

@check_version(mpmath, '0.19')
def test_expn_large_n():
    if False:
        for i in range(10):
            print('nop')
    dataset = []
    for n in [50, 51]:
        for x in np.logspace(0, 4, 200):
            with mpmath.workdps(100):
                dataset.append((n, x, float(mpmath.expint(n, x))))
    dataset = np.asarray(dataset)
    FuncData(sc.expn, dataset, (0, 1), 2, rtol=1e-13).check()

@check_version(mpmath, '0.19')
def test_hyp0f1_gh5764():
    if False:
        i = 10
        return i + 15
    dataset = []
    axis = [-99.5, -9.5, -0.5, 0.5, 9.5, 99.5]
    for v in axis:
        for x in axis:
            for y in axis:
                z = x + 1j * y
                with mpmath.workdps(120):
                    res = complex(mpmath.hyp0f1(v, z))
                dataset.append((v, z, res))
    dataset = np.array(dataset)
    FuncData(lambda v, z: sc.hyp0f1(v.real, z), dataset, (0, 1), 2, rtol=1e-13).check()

@check_version(mpmath, '0.19')
def test_hyp0f1_gh_1609():
    if False:
        print('Hello World!')
    vv = np.linspace(150, 180, 21)
    af = sc.hyp0f1(vv, 0.5)
    mf = np.array([mpmath.hyp0f1(v, 0.5) for v in vv])
    assert_allclose(af, mf.astype(float), rtol=1e-12)

@check_version(mpmath, '1.1.0')
def test_hyperu_around_0():
    if False:
        return 10
    dataset = []
    for n in np.arange(-5, 5):
        for b in np.linspace(-5, 5, 20):
            a = -n
            dataset.append((a, b, 0, float(mpmath.hyperu(a, b, 0))))
            a = -n + b - 1
            dataset.append((a, b, 0, float(mpmath.hyperu(a, b, 0))))
    for a in [-10.5, -1.5, -0.5, 0, 0.5, 1, 10]:
        for b in [-1.0, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]:
            dataset.append((a, b, 0, float(mpmath.hyperu(a, b, 0))))
    dataset = np.array(dataset)
    FuncData(sc.hyperu, dataset, (0, 1, 2), 3, rtol=1e-15, atol=5e-13).check()

@check_version(mpmath, '1.0.0')
def test_hyp2f1_strange_points():
    if False:
        i = 10
        return i + 15
    pts = [(2, -1, -1, 0.7), (2, -2, -2, 0.7)]
    pts += list(itertools.product([2, 1, -0.7, -1000], repeat=4))
    pts = [(a, b, c, x) for (a, b, c, x) in pts if b == c and round(b) == b and (b < 0) and (b != -1000)]
    kw = dict(eliminate=True)
    dataset = [p + (float(mpmath.hyp2f1(*p, **kw)),) for p in pts]
    dataset = np.array(dataset, dtype=np.float64)
    FuncData(sc.hyp2f1, dataset, (0, 1, 2, 3), 4, rtol=1e-10).check()

@check_version(mpmath, '0.13')
def test_hyp2f1_real_some_points():
    if False:
        for i in range(10):
            print('nop')
    pts = [(1, 2, 3, 0), (1.0 / 3, 2.0 / 3, 5.0 / 6, 27.0 / 32), (1.0 / 4, 1.0 / 2, 3.0 / 4, 80.0 / 81), (2, -2, -3, 3), (2, -3, -2, 3), (2, -1.5, -1.5, 3), (1, 2, 3, 0), (0.7235, -1, -5, 0.3), (0.25, 1.0 / 3, 2, 0.999), (0.25, 1.0 / 3, 2, -1), (2, 3, 5, 0.99), (3.0 / 2, -0.5, 3, 0.99), (2, 2.5, -3.25, 0.999), (-8, 18.016500331508873, 10.805295997850628, 0.90875647507), (-10, 900, -10.5, 0.99), (-10, 900, 10.5, 0.99), (-1, 2, 1, 1.0), (-1, 2, 1, -1.0), (-3, 13, 5, 1.0), (-3, 13, 5, -1.0), (0.5, 1 - 270.5, 1.5, 0.999 ** 2)]
    dataset = [p + (float(mpmath.hyp2f1(*p)),) for p in pts]
    dataset = np.array(dataset, dtype=np.float64)
    with np.errstate(invalid='ignore'):
        FuncData(sc.hyp2f1, dataset, (0, 1, 2, 3), 4, rtol=1e-10).check()

@check_version(mpmath, '0.14')
def test_hyp2f1_some_points_2():
    if False:
        return 10
    pts = [(112, (51, 10), (-9, 10), -0.99999), (10, -900, 10.5, 0.99), (10, -900, -10.5, 0.99)]

    def fev(x):
        if False:
            return 10
        if isinstance(x, tuple):
            return float(x[0]) / x[1]
        else:
            return x
    dataset = [tuple(map(fev, p)) + (float(mpmath.hyp2f1(*p)),) for p in pts]
    dataset = np.array(dataset, dtype=np.float64)
    FuncData(sc.hyp2f1, dataset, (0, 1, 2, 3), 4, rtol=1e-10).check()

@check_version(mpmath, '0.13')
def test_hyp2f1_real_some():
    if False:
        i = 10
        return i + 15
    dataset = []
    for a in [-10, -5, -1.8, 1.8, 5, 10]:
        for b in [-2.5, -1, 1, 7.4]:
            for c in [-9, -1.8, 5, 20.4]:
                for z in [-10, -1.01, -0.99, 0, 0.6, 0.95, 1.5, 10]:
                    try:
                        v = float(mpmath.hyp2f1(a, b, c, z))
                    except Exception:
                        continue
                    dataset.append((a, b, c, z, v))
    dataset = np.array(dataset, dtype=np.float64)
    with np.errstate(invalid='ignore'):
        FuncData(sc.hyp2f1, dataset, (0, 1, 2, 3), 4, rtol=1e-09, ignore_inf_sign=True).check()

@check_version(mpmath, '0.12')
@pytest.mark.slow
def test_hyp2f1_real_random():
    if False:
        while True:
            i = 10
    npoints = 500
    dataset = np.zeros((npoints, 5), np.float64)
    np.random.seed(1234)
    dataset[:, 0] = np.random.pareto(1.5, npoints)
    dataset[:, 1] = np.random.pareto(1.5, npoints)
    dataset[:, 2] = np.random.pareto(1.5, npoints)
    dataset[:, 3] = 2 * np.random.rand(npoints) - 1
    dataset[:, 0] *= (-1) ** np.random.randint(2, npoints)
    dataset[:, 1] *= (-1) ** np.random.randint(2, npoints)
    dataset[:, 2] *= (-1) ** np.random.randint(2, npoints)
    for ds in dataset:
        if mpmath.__version__ < '0.14':
            if abs(ds[:2]).max() > abs(ds[2]):
                ds[2] = abs(ds[:2]).max()
        ds[4] = float(mpmath.hyp2f1(*tuple(ds[:4])))
    FuncData(sc.hyp2f1, dataset, (0, 1, 2, 3), 4, rtol=1e-09).check()

@check_version(mpmath, '0.14')
def test_erf_complex():
    if False:
        while True:
            i = 10
    (old_dps, old_prec) = (mpmath.mp.dps, mpmath.mp.prec)
    try:
        mpmath.mp.dps = 70
        (x1, y1) = np.meshgrid(np.linspace(-10, 1, 31), np.linspace(-10, 1, 11))
        (x2, y2) = np.meshgrid(np.logspace(-80, 0.8, 31), np.logspace(-80, 0.8, 11))
        points = np.r_[x1.ravel(), x2.ravel()] + 1j * np.r_[y1.ravel(), y2.ravel()]
        assert_func_equal(sc.erf, lambda x: complex(mpmath.erf(x)), points, vectorized=False, rtol=1e-13)
        assert_func_equal(sc.erfc, lambda x: complex(mpmath.erfc(x)), points, vectorized=False, rtol=1e-13)
    finally:
        (mpmath.mp.dps, mpmath.mp.prec) = (old_dps, old_prec)

@check_version(mpmath, '0.15')
def test_lpmv():
    if False:
        return 10
    pts = []
    for x in [-0.99, -0.557, 1e-06, 0.132, 1]:
        pts.extend([(1, 1, x), (1, -1, x), (-1, 1, x), (-1, -2, x), (1, 1.7, x), (1, -1.7, x), (-1, 1.7, x), (-1, -2.7, x), (1, 10, x), (1, 11, x), (3, 8, x), (5, 11, x), (-3, 8, x), (-5, 11, x), (3, -8, x), (5, -11, x), (-3, -8, x), (-5, -11, x), (3, 8.3, x), (5, 11.3, x), (-3, 8.3, x), (-5, 11.3, x), (3, -8.3, x), (5, -11.3, x), (-3, -8.3, x), (-5, -11.3, x)])

    def mplegenp(nu, mu, x):
        if False:
            for i in range(10):
                print('nop')
        if mu == int(mu) and x == 1:
            if mu == 0:
                return 1
            else:
                return 0
        return mpmath.legenp(nu, mu, x)
    dataset = [p + (mplegenp(p[1], p[0], p[2]),) for p in pts]
    dataset = np.array(dataset, dtype=np.float64)

    def evf(mu, nu, x):
        if False:
            i = 10
            return i + 15
        return sc.lpmv(mu.astype(int), nu, x)
    with np.errstate(invalid='ignore'):
        FuncData(evf, dataset, (0, 1, 2), 3, rtol=1e-10, atol=1e-14).check()

@check_version(mpmath, '0.15')
def test_beta():
    if False:
        return 10
    np.random.seed(1234)
    b = np.r_[np.logspace(-200, 200, 4), np.logspace(-10, 10, 4), np.logspace(-1, 1, 4), np.arange(-10, 11, 1), np.arange(-10, 11, 1) + 0.5, -1, -2.3, -3, -100.3, -10003.4]
    a = b
    ab = np.array(np.broadcast_arrays(a[:, None], b[None, :])).reshape(2, -1).T
    (old_dps, old_prec) = (mpmath.mp.dps, mpmath.mp.prec)
    try:
        mpmath.mp.dps = 400
        assert_func_equal(sc.beta, lambda a, b: float(mpmath.beta(a, b)), ab, vectorized=False, rtol=1e-10, ignore_inf_sign=True)
        assert_func_equal(sc.betaln, lambda a, b: float(mpmath.log(abs(mpmath.beta(a, b)))), ab, vectorized=False, rtol=1e-10)
    finally:
        (mpmath.mp.dps, mpmath.mp.prec) = (old_dps, old_prec)
LOGGAMMA_TAYLOR_RADIUS = 0.2

@check_version(mpmath, '0.19')
def test_loggamma_taylor_transition():
    if False:
        for i in range(10):
            print('nop')
    r = LOGGAMMA_TAYLOR_RADIUS + np.array([-0.1, -0.01, 0, 0.01, 0.1])
    theta = np.linspace(0, 2 * np.pi, 20)
    (r, theta) = np.meshgrid(r, theta)
    dz = r * np.exp(1j * theta)
    z = np.r_[1 + dz, 2 + dz].flatten()
    dataset = [(z0, complex(mpmath.loggamma(z0))) for z0 in z]
    dataset = np.array(dataset)
    FuncData(sc.loggamma, dataset, 0, 1, rtol=5e-14).check()

@check_version(mpmath, '0.19')
def test_loggamma_taylor():
    if False:
        print('Hello World!')
    r = np.logspace(-16, np.log10(LOGGAMMA_TAYLOR_RADIUS), 10)
    theta = np.linspace(0, 2 * np.pi, 20)
    (r, theta) = np.meshgrid(r, theta)
    dz = r * np.exp(1j * theta)
    z = np.r_[1 + dz, 2 + dz].flatten()
    dataset = [(z0, complex(mpmath.loggamma(z0))) for z0 in z]
    dataset = np.array(dataset)
    FuncData(sc.loggamma, dataset, 0, 1, rtol=5e-14).check()

@check_version(mpmath, '0.19')
@pytest.mark.slow
def test_rgamma_zeros():
    if False:
        return 10
    dx = np.r_[-np.logspace(-1, -13, 3), 0, np.logspace(-13, -1, 3)]
    dy = dx.copy()
    (dx, dy) = np.meshgrid(dx, dy)
    dz = dx + 1j * dy
    zeros = np.arange(0, -170, -1).reshape(1, 1, -1)
    z = (zeros + np.dstack((dz,) * zeros.size)).flatten()
    with mpmath.workdps(100):
        dataset = [(z0, complex(mpmath.rgamma(z0))) for z0 in z]
    dataset = np.array(dataset)
    FuncData(sc.rgamma, dataset, 0, 1, rtol=1e-12).check()

@check_version(mpmath, '0.19')
@pytest.mark.slow
def test_digamma_roots():
    if False:
        i = 10
        return i + 15
    root = mpmath.findroot(mpmath.digamma, 1.5)
    roots = [float(root)]
    root = mpmath.findroot(mpmath.digamma, -0.5)
    roots.append(float(root))
    roots = np.array(roots)
    dx = np.r_[-0.24, -np.logspace(-1, -15, 10), 0, np.logspace(-15, -1, 10), 0.24]
    dy = dx.copy()
    (dx, dy) = np.meshgrid(dx, dy)
    dz = dx + 1j * dy
    z = (roots + np.dstack((dz,) * roots.size)).flatten()
    with mpmath.workdps(30):
        dataset = [(z0, complex(mpmath.digamma(z0))) for z0 in z]
    dataset = np.array(dataset)
    FuncData(sc.digamma, dataset, 0, 1, rtol=1e-14).check()

@check_version(mpmath, '0.19')
def test_digamma_negreal():
    if False:
        while True:
            i = 10
    digamma = exception_to_nan(mpmath.digamma)
    x = -np.logspace(300, -30, 100)
    y = np.r_[-np.logspace(0, -3, 5), 0, np.logspace(-3, 0, 5)]
    (x, y) = np.meshgrid(x, y)
    z = (x + 1j * y).flatten()
    with mpmath.workdps(40):
        dataset = [(z0, complex(digamma(z0))) for z0 in z]
    dataset = np.asarray(dataset)
    FuncData(sc.digamma, dataset, 0, 1, rtol=1e-13).check()

@check_version(mpmath, '0.19')
def test_digamma_boundary():
    if False:
        while True:
            i = 10
    x = -np.logspace(300, -30, 100)
    y = np.array([-6.1, -5.9, 5.9, 6.1])
    (x, y) = np.meshgrid(x, y)
    z = (x + 1j * y).flatten()
    with mpmath.workdps(30):
        dataset = [(z0, complex(mpmath.digamma(z0))) for z0 in z]
    dataset = np.asarray(dataset)
    FuncData(sc.digamma, dataset, 0, 1, rtol=1e-13).check()

@check_version(mpmath, '0.19')
@pytest.mark.slow
def test_gammainc_boundary():
    if False:
        for i in range(10):
            print('nop')
    small = 20
    a = np.linspace(0.5 * small, 2 * small, 50)
    x = a.copy()
    (a, x) = np.meshgrid(a, x)
    (a, x) = (a.flatten(), x.flatten())
    with mpmath.workdps(100):
        dataset = [(a0, x0, float(mpmath.gammainc(a0, b=x0, regularized=True))) for (a0, x0) in zip(a, x)]
    dataset = np.array(dataset)
    FuncData(sc.gammainc, dataset, (0, 1), 2, rtol=1e-12).check()

@check_version(mpmath, '0.19')
@pytest.mark.slow
def test_spence_circle():
    if False:
        return 10

    def spence(z):
        if False:
            i = 10
            return i + 15
        return complex(mpmath.polylog(2, 1 - z))
    r = np.linspace(0.5, 1.5)
    theta = np.linspace(0, 2 * pi)
    z = (1 + np.outer(r, np.exp(1j * theta))).flatten()
    dataset = np.asarray([(z0, spence(z0)) for z0 in z])
    FuncData(sc.spence, dataset, 0, 1, rtol=1e-14).check()

@check_version(mpmath, '0.19')
def test_sinpi_zeros():
    if False:
        return 10
    eps = np.finfo(float).eps
    dx = np.r_[-np.logspace(0, -13, 3), 0, np.logspace(-13, 0, 3)]
    dy = dx.copy()
    (dx, dy) = np.meshgrid(dx, dy)
    dz = dx + 1j * dy
    zeros = np.arange(-100, 100, 1).reshape(1, 1, -1)
    z = (zeros + np.dstack((dz,) * zeros.size)).flatten()
    dataset = np.asarray([(z0, complex(mpmath.sinpi(z0))) for z0 in z])
    FuncData(_sinpi, dataset, 0, 1, rtol=2 * eps).check()

@check_version(mpmath, '0.19')
def test_cospi_zeros():
    if False:
        return 10
    eps = np.finfo(float).eps
    dx = np.r_[-np.logspace(0, -13, 3), 0, np.logspace(-13, 0, 3)]
    dy = dx.copy()
    (dx, dy) = np.meshgrid(dx, dy)
    dz = dx + 1j * dy
    zeros = (np.arange(-100, 100, 1) + 0.5).reshape(1, 1, -1)
    z = (zeros + np.dstack((dz,) * zeros.size)).flatten()
    dataset = np.asarray([(z0, complex(mpmath.cospi(z0))) for z0 in z])
    FuncData(_cospi, dataset, 0, 1, rtol=2 * eps).check()

@check_version(mpmath, '0.19')
def test_dn_quarter_period():
    if False:
        while True:
            i = 10

    def dn(u, m):
        if False:
            while True:
                i = 10
        return sc.ellipj(u, m)[2]

    def mpmath_dn(u, m):
        if False:
            for i in range(10):
                print('nop')
        return float(mpmath.ellipfun('dn', u=u, m=m))
    m = np.linspace(0, 1, 20)
    du = np.r_[-np.logspace(-1, -15, 10), 0, np.logspace(-15, -1, 10)]
    dataset = []
    for m0 in m:
        u0 = float(mpmath.ellipk(m0))
        for du0 in du:
            p = u0 + du0
            dataset.append((p, m0, mpmath_dn(p, m0)))
    dataset = np.asarray(dataset)
    FuncData(dn, dataset, (0, 1), 2, rtol=1e-10).check()

def _mpmath_wrightomega(z, dps):
    if False:
        i = 10
        return i + 15
    with mpmath.workdps(dps):
        z = mpmath.mpc(z)
        unwind = mpmath.ceil((z.imag - mpmath.pi) / (2 * mpmath.pi))
        res = mpmath.lambertw(mpmath.exp(z), unwind)
    return res

@pytest.mark.slow
@check_version(mpmath, '0.19')
def test_wrightomega_branch():
    if False:
        print('Hello World!')
    x = -np.logspace(10, 0, 25)
    picut_above = [np.nextafter(np.pi, np.inf)]
    picut_below = [np.nextafter(np.pi, -np.inf)]
    npicut_above = [np.nextafter(-np.pi, np.inf)]
    npicut_below = [np.nextafter(-np.pi, -np.inf)]
    for i in range(50):
        picut_above.append(np.nextafter(picut_above[-1], np.inf))
        picut_below.append(np.nextafter(picut_below[-1], -np.inf))
        npicut_above.append(np.nextafter(npicut_above[-1], np.inf))
        npicut_below.append(np.nextafter(npicut_below[-1], -np.inf))
    y = np.hstack((picut_above, picut_below, npicut_above, npicut_below))
    (x, y) = np.meshgrid(x, y)
    z = (x + 1j * y).flatten()
    dataset = np.asarray([(z0, complex(_mpmath_wrightomega(z0, 25))) for z0 in z])
    FuncData(sc.wrightomega, dataset, 0, 1, rtol=1e-08).check()

@pytest.mark.slow
@check_version(mpmath, '0.19')
def test_wrightomega_region1():
    if False:
        i = 10
        return i + 15
    x = np.linspace(-2, 1)
    y = np.linspace(1, 2 * np.pi)
    (x, y) = np.meshgrid(x, y)
    z = (x + 1j * y).flatten()
    dataset = np.asarray([(z0, complex(_mpmath_wrightomega(z0, 25))) for z0 in z])
    FuncData(sc.wrightomega, dataset, 0, 1, rtol=1e-15).check()

@pytest.mark.slow
@check_version(mpmath, '0.19')
def test_wrightomega_region2():
    if False:
        print('Hello World!')
    x = np.linspace(-2, 1)
    y = np.linspace(-2 * np.pi, -1)
    (x, y) = np.meshgrid(x, y)
    z = (x + 1j * y).flatten()
    dataset = np.asarray([(z0, complex(_mpmath_wrightomega(z0, 25))) for z0 in z])
    FuncData(sc.wrightomega, dataset, 0, 1, rtol=1e-15).check()

@pytest.mark.slow
@check_version(mpmath, '0.19')
def test_lambertw_smallz():
    if False:
        return 10
    (x, y) = (np.linspace(-1, 1, 25), np.linspace(-1, 1, 25))
    (x, y) = np.meshgrid(x, y)
    z = (x + 1j * y).flatten()
    dataset = np.asarray([(z0, complex(mpmath.lambertw(z0))) for z0 in z])
    FuncData(sc.lambertw, dataset, 0, 1, rtol=1e-13).check()
HYPERKW = dict(maxprec=200, maxterms=200)

@pytest.mark.slow
@check_version(mpmath, '0.17')
class TestSystematic:

    def test_airyai(self):
        if False:
            print('Hello World!')
        assert_mpmath_equal(lambda z: sc.airy(z)[0], mpmath.airyai, [Arg(-100000000.0, 100000000.0)], rtol=1e-05)
        assert_mpmath_equal(lambda z: sc.airy(z)[0], mpmath.airyai, [Arg(-1000.0, 1000.0)])

    def test_airyai_complex(self):
        if False:
            for i in range(10):
                print('nop')
        assert_mpmath_equal(lambda z: sc.airy(z)[0], mpmath.airyai, [ComplexArg()])

    def test_airyai_prime(self):
        if False:
            for i in range(10):
                print('nop')
        assert_mpmath_equal(lambda z: sc.airy(z)[1], lambda z: mpmath.airyai(z, derivative=1), [Arg(-100000000.0, 100000000.0)], rtol=1e-05)
        assert_mpmath_equal(lambda z: sc.airy(z)[1], lambda z: mpmath.airyai(z, derivative=1), [Arg(-1000.0, 1000.0)])

    def test_airyai_prime_complex(self):
        if False:
            while True:
                i = 10
        assert_mpmath_equal(lambda z: sc.airy(z)[1], lambda z: mpmath.airyai(z, derivative=1), [ComplexArg()])

    def test_airybi(self):
        if False:
            for i in range(10):
                print('nop')
        assert_mpmath_equal(lambda z: sc.airy(z)[2], lambda z: mpmath.airybi(z), [Arg(-100000000.0, 100000000.0)], rtol=1e-05)
        assert_mpmath_equal(lambda z: sc.airy(z)[2], lambda z: mpmath.airybi(z), [Arg(-1000.0, 1000.0)])

    def test_airybi_complex(self):
        if False:
            for i in range(10):
                print('nop')
        assert_mpmath_equal(lambda z: sc.airy(z)[2], lambda z: mpmath.airybi(z), [ComplexArg()])

    def test_airybi_prime(self):
        if False:
            while True:
                i = 10
        assert_mpmath_equal(lambda z: sc.airy(z)[3], lambda z: mpmath.airybi(z, derivative=1), [Arg(-100000000.0, 100000000.0)], rtol=1e-05)
        assert_mpmath_equal(lambda z: sc.airy(z)[3], lambda z: mpmath.airybi(z, derivative=1), [Arg(-1000.0, 1000.0)])

    def test_airybi_prime_complex(self):
        if False:
            i = 10
            return i + 15
        assert_mpmath_equal(lambda z: sc.airy(z)[3], lambda z: mpmath.airybi(z, derivative=1), [ComplexArg()])

    def test_bei(self):
        if False:
            i = 10
            return i + 15
        assert_mpmath_equal(sc.bei, exception_to_nan(lambda z: mpmath.bei(0, z, **HYPERKW)), [Arg(-1000.0, 1000.0)])

    def test_ber(self):
        if False:
            return 10
        assert_mpmath_equal(sc.ber, exception_to_nan(lambda z: mpmath.ber(0, z, **HYPERKW)), [Arg(-1000.0, 1000.0)])

    def test_bernoulli(self):
        if False:
            while True:
                i = 10
        assert_mpmath_equal(lambda n: sc.bernoulli(int(n))[int(n)], lambda n: float(mpmath.bernoulli(int(n))), [IntArg(0, 13000)], rtol=1e-09, n=13000)

    def test_besseli(self):
        if False:
            i = 10
            return i + 15
        assert_mpmath_equal(sc.iv, exception_to_nan(lambda v, z: mpmath.besseli(v, z, **HYPERKW)), [Arg(-1e+100, 1e+100), Arg()], atol=1e-270)

    def test_besseli_complex(self):
        if False:
            return 10
        assert_mpmath_equal(lambda v, z: sc.iv(v.real, z), exception_to_nan(lambda v, z: mpmath.besseli(v, z, **HYPERKW)), [Arg(-1e+100, 1e+100), ComplexArg()])

    def test_besselj(self):
        if False:
            i = 10
            return i + 15
        assert_mpmath_equal(sc.jv, exception_to_nan(lambda v, z: mpmath.besselj(v, z, **HYPERKW)), [Arg(-1e+100, 1e+100), Arg(-1000.0, 1000.0)], ignore_inf_sign=True)
        assert_mpmath_equal(sc.jv, exception_to_nan(lambda v, z: mpmath.besselj(v, z, **HYPERKW)), [Arg(-1e+100, 1e+100), Arg(-100000000.0, 100000000.0)], ignore_inf_sign=True, rtol=1e-05)

    def test_besselj_complex(self):
        if False:
            i = 10
            return i + 15
        assert_mpmath_equal(lambda v, z: sc.jv(v.real, z), exception_to_nan(lambda v, z: mpmath.besselj(v, z, **HYPERKW)), [Arg(), ComplexArg()])

    def test_besselk(self):
        if False:
            print('Hello World!')
        assert_mpmath_equal(sc.kv, mpmath.besselk, [Arg(-200, 200), Arg(0, np.inf)], nan_ok=False, rtol=1e-12)

    def test_besselk_int(self):
        if False:
            for i in range(10):
                print('nop')
        assert_mpmath_equal(sc.kn, mpmath.besselk, [IntArg(-200, 200), Arg(0, np.inf)], nan_ok=False, rtol=1e-12)

    def test_besselk_complex(self):
        if False:
            print('Hello World!')
        assert_mpmath_equal(lambda v, z: sc.kv(v.real, z), exception_to_nan(lambda v, z: mpmath.besselk(v, z, **HYPERKW)), [Arg(-1e+100, 1e+100), ComplexArg()])

    def test_bessely(self):
        if False:
            return 10

        def mpbessely(v, x):
            if False:
                return 10
            r = float(mpmath.bessely(v, x, **HYPERKW))
            if abs(r) > 1e+305:
                r = np.inf * np.sign(r)
            if abs(r) == 0 and x == 0:
                return np.nan
            return r
        assert_mpmath_equal(sc.yv, exception_to_nan(mpbessely), [Arg(-1e+100, 1e+100), Arg(-100000000.0, 100000000.0)], n=5000)

    def test_bessely_complex(self):
        if False:
            i = 10
            return i + 15

        def mpbessely(v, x):
            if False:
                while True:
                    i = 10
            r = complex(mpmath.bessely(v, x, **HYPERKW))
            if abs(r) > 1e+305:
                with np.errstate(invalid='ignore'):
                    r = np.inf * np.sign(r)
            return r
        assert_mpmath_equal(lambda v, z: sc.yv(v.real, z), exception_to_nan(mpbessely), [Arg(), ComplexArg()], n=15000)

    def test_bessely_int(self):
        if False:
            while True:
                i = 10

        def mpbessely(v, x):
            if False:
                for i in range(10):
                    print('nop')
            r = float(mpmath.bessely(v, x))
            if abs(r) == 0 and x == 0:
                return np.nan
            return r
        assert_mpmath_equal(lambda v, z: sc.yn(int(v), z), exception_to_nan(mpbessely), [IntArg(-1000, 1000), Arg(-100000000.0, 100000000.0)])

    def test_beta(self):
        if False:
            i = 10
            return i + 15
        bad_points = []

        def beta(a, b, nonzero=False):
            if False:
                return 10
            if a < -1000000000000.0 or b < -1000000000000.0:
                return np.nan
            if (a < 0 or b < 0) and abs(float(a + b)) % 1 == 0:
                if nonzero:
                    bad_points.append((float(a), float(b)))
                    return np.nan
            return mpmath.beta(a, b)
        assert_mpmath_equal(sc.beta, lambda a, b: beta(a, b, nonzero=True), [Arg(), Arg()], dps=400, ignore_inf_sign=True)
        assert_mpmath_equal(sc.beta, beta, np.array(bad_points), dps=400, ignore_inf_sign=True, atol=1e-11)

    def test_betainc(self):
        if False:
            while True:
                i = 10
        assert_mpmath_equal(sc.betainc, time_limited()(exception_to_nan(lambda a, b, x: mpmath.betainc(a, b, 0, x, regularized=True))), [Arg(), Arg(), Arg()])

    def test_betaincc(self):
        if False:
            return 10
        assert_mpmath_equal(sc.betaincc, time_limited()(exception_to_nan(lambda a, b, x: mpmath.betainc(a, b, x, 1, regularized=True))), [Arg(), Arg(), Arg()], dps=400)

    def test_binom(self):
        if False:
            while True:
                i = 10
        bad_points = []

        def binomial(n, k, nonzero=False):
            if False:
                print('Hello World!')
            if abs(k) > 100000000.0 * (abs(n) + 1):
                return np.nan
            if n < k and abs(float(n - k) - np.round(float(n - k))) < 1e-15:
                if nonzero:
                    bad_points.append((float(n), float(k)))
                    return np.nan
            return mpmath.binomial(n, k)
        assert_mpmath_equal(sc.binom, lambda n, k: binomial(n, k, nonzero=True), [Arg(), Arg()], dps=400)
        assert_mpmath_equal(sc.binom, binomial, np.array(bad_points), dps=400, atol=1e-14)

    def test_chebyt_int(self):
        if False:
            i = 10
            return i + 15
        assert_mpmath_equal(lambda n, x: sc.eval_chebyt(int(n), x), exception_to_nan(lambda n, x: mpmath.chebyt(n, x, **HYPERKW)), [IntArg(), Arg()], dps=50)

    @pytest.mark.xfail(run=False, reason='some cases in hyp2f1 not fully accurate')
    def test_chebyt(self):
        if False:
            for i in range(10):
                print('nop')
        assert_mpmath_equal(sc.eval_chebyt, lambda n, x: time_limited()(exception_to_nan(mpmath.chebyt))(n, x, **HYPERKW), [Arg(-101, 101), Arg()], n=10000)

    def test_chebyu_int(self):
        if False:
            return 10
        assert_mpmath_equal(lambda n, x: sc.eval_chebyu(int(n), x), exception_to_nan(lambda n, x: mpmath.chebyu(n, x, **HYPERKW)), [IntArg(), Arg()], dps=50)

    @pytest.mark.xfail(run=False, reason='some cases in hyp2f1 not fully accurate')
    def test_chebyu(self):
        if False:
            i = 10
            return i + 15
        assert_mpmath_equal(sc.eval_chebyu, lambda n, x: time_limited()(exception_to_nan(mpmath.chebyu))(n, x, **HYPERKW), [Arg(-101, 101), Arg()])

    def test_chi(self):
        if False:
            while True:
                i = 10

        def chi(x):
            if False:
                for i in range(10):
                    print('nop')
            return sc.shichi(x)[1]
        assert_mpmath_equal(chi, mpmath.chi, [Arg()])
        assert_mpmath_equal(chi, mpmath.chi, [FixedArg([88 - 1e-09, 88, 88 + 1e-09])])

    def test_chi_complex(self):
        if False:
            i = 10
            return i + 15

        def chi(z):
            if False:
                i = 10
                return i + 15
            return sc.shichi(z)[1]
        assert_mpmath_equal(chi, mpmath.chi, [ComplexArg(complex(-np.inf, -100000000.0), complex(np.inf, 100000000.0))], rtol=1e-12)

    def test_ci(self):
        if False:
            i = 10
            return i + 15

        def ci(x):
            if False:
                for i in range(10):
                    print('nop')
            return sc.sici(x)[1]
        assert_mpmath_equal(ci, mpmath.ci, [Arg(-100000000.0, 100000000.0)])

    def test_ci_complex(self):
        if False:
            while True:
                i = 10

        def ci(z):
            if False:
                return 10
            return sc.sici(z)[1]
        assert_mpmath_equal(ci, mpmath.ci, [ComplexArg(complex(-100000000.0, -np.inf), complex(100000000.0, np.inf))], rtol=1e-08)

    def test_cospi(self):
        if False:
            while True:
                i = 10
        eps = np.finfo(float).eps
        assert_mpmath_equal(_cospi, mpmath.cospi, [Arg()], nan_ok=False, rtol=2 * eps)

    def test_cospi_complex(self):
        if False:
            for i in range(10):
                print('nop')
        assert_mpmath_equal(_cospi, mpmath.cospi, [ComplexArg()], nan_ok=False, rtol=1e-13)

    def test_digamma(self):
        if False:
            i = 10
            return i + 15
        assert_mpmath_equal(sc.digamma, exception_to_nan(mpmath.digamma), [Arg()], rtol=1e-12, dps=50)

    def test_digamma_complex(self):
        if False:
            print('Hello World!')

        def param_filter(z):
            if False:
                while True:
                    i = 10
            return np.where((z.real < 0) & (np.abs(z.imag) < 1.12), False, True)
        assert_mpmath_equal(sc.digamma, exception_to_nan(mpmath.digamma), [ComplexArg()], rtol=1e-13, dps=40, param_filter=param_filter)

    def test_e1(self):
        if False:
            return 10
        assert_mpmath_equal(sc.exp1, mpmath.e1, [Arg()], rtol=1e-14)

    def test_e1_complex(self):
        if False:
            return 10
        assert_mpmath_equal(sc.exp1, mpmath.e1, [ComplexArg(complex(-np.inf, -100000000.0), complex(np.inf, 100000000.0))], rtol=1e-11)
        assert_mpmath_equal(sc.exp1, mpmath.e1, (np.linspace(-50, 50, 171)[:, None] + np.r_[0, np.logspace(-3, 2, 61), -np.logspace(-3, 2, 11)] * 1j).ravel(), rtol=1e-11)
        assert_mpmath_equal(sc.exp1, mpmath.e1, np.linspace(-50, -35, 10000) + 0j, rtol=1e-11)

    def test_exprel(self):
        if False:
            print('Hello World!')
        assert_mpmath_equal(sc.exprel, lambda x: mpmath.expm1(x) / x if x != 0 else mpmath.mpf('1.0'), [Arg(a=-np.log(np.finfo(np.float64).max), b=np.log(np.finfo(np.float64).max))])
        assert_mpmath_equal(sc.exprel, lambda x: mpmath.expm1(x) / x if x != 0 else mpmath.mpf('1.0'), np.array([1e-12, 1e-24, 0, 1000000000000.0, 1e+24, np.inf]), rtol=1e-11)
        assert_(np.isinf(sc.exprel(np.inf)))
        assert_(sc.exprel(-np.inf) == 0)

    def test_expm1_complex(self):
        if False:
            for i in range(10):
                print('nop')
        assert_mpmath_equal(sc.expm1, mpmath.expm1, [ComplexArg(complex(-np.inf, -10000000.0), complex(np.inf, 10000000.0))])

    def test_log1p_complex(self):
        if False:
            print('Hello World!')
        assert_mpmath_equal(sc.log1p, lambda x: mpmath.log(x + 1), [ComplexArg()], dps=60)

    def test_log1pmx(self):
        if False:
            print('Hello World!')
        assert_mpmath_equal(_log1pmx, lambda x: mpmath.log(x + 1) - x, [Arg()], dps=60, rtol=1e-14)

    def test_ei(self):
        if False:
            while True:
                i = 10
        assert_mpmath_equal(sc.expi, mpmath.ei, [Arg()], rtol=1e-11)

    def test_ei_complex(self):
        if False:
            while True:
                i = 10
        assert_mpmath_equal(sc.expi, mpmath.ei, [ComplexArg(complex(-np.inf, -100000000.0), complex(np.inf, 100000000.0))], rtol=1e-09)

    def test_ellipe(self):
        if False:
            i = 10
            return i + 15
        assert_mpmath_equal(sc.ellipe, mpmath.ellipe, [Arg(b=1.0)])

    def test_ellipeinc(self):
        if False:
            i = 10
            return i + 15
        assert_mpmath_equal(sc.ellipeinc, mpmath.ellipe, [Arg(-1000.0, 1000.0), Arg(b=1.0)])

    def test_ellipeinc_largephi(self):
        if False:
            print('Hello World!')
        assert_mpmath_equal(sc.ellipeinc, mpmath.ellipe, [Arg(), Arg()])

    def test_ellipf(self):
        if False:
            print('Hello World!')
        assert_mpmath_equal(sc.ellipkinc, mpmath.ellipf, [Arg(-1000.0, 1000.0), Arg()])

    def test_ellipf_largephi(self):
        if False:
            return 10
        assert_mpmath_equal(sc.ellipkinc, mpmath.ellipf, [Arg(), Arg()])

    def test_ellipk(self):
        if False:
            while True:
                i = 10
        assert_mpmath_equal(sc.ellipk, mpmath.ellipk, [Arg(b=1.0)])
        assert_mpmath_equal(sc.ellipkm1, lambda m: mpmath.ellipk(1 - m), [Arg(a=0.0)], dps=400)

    def test_ellipkinc(self):
        if False:
            return 10

        def ellipkinc(phi, m):
            if False:
                print('Hello World!')
            return mpmath.ellippi(0, phi, m)
        assert_mpmath_equal(sc.ellipkinc, ellipkinc, [Arg(-1000.0, 1000.0), Arg(b=1.0)], ignore_inf_sign=True)

    def test_ellipkinc_largephi(self):
        if False:
            return 10

        def ellipkinc(phi, m):
            if False:
                for i in range(10):
                    print('nop')
            return mpmath.ellippi(0, phi, m)
        assert_mpmath_equal(sc.ellipkinc, ellipkinc, [Arg(), Arg(b=1.0)], ignore_inf_sign=True)

    def test_ellipfun_sn(self):
        if False:
            print('Hello World!')

        def sn(u, m):
            if False:
                while True:
                    i = 10
            if u == 0:
                return 0
            else:
                return mpmath.ellipfun('sn', u=u, m=m)
        assert_mpmath_equal(lambda u, m: sc.ellipj(u, m)[0], sn, [Arg(-1000000.0, 1000000.0), Arg(a=0, b=1)], rtol=1e-08)

    def test_ellipfun_cn(self):
        if False:
            i = 10
            return i + 15
        assert_mpmath_equal(lambda u, m: sc.ellipj(u, m)[1], lambda u, m: mpmath.ellipfun('cn', u=u, m=m), [Arg(-1000000.0, 1000000.0), Arg(a=0, b=1)], rtol=1e-08)

    def test_ellipfun_dn(self):
        if False:
            while True:
                i = 10
        assert_mpmath_equal(lambda u, m: sc.ellipj(u, m)[2], lambda u, m: mpmath.ellipfun('dn', u=u, m=m), [Arg(-1000000.0, 1000000.0), Arg(a=0, b=1)], rtol=1e-08)

    def test_erf(self):
        if False:
            return 10
        assert_mpmath_equal(sc.erf, lambda z: mpmath.erf(z), [Arg()])

    def test_erf_complex(self):
        if False:
            i = 10
            return i + 15
        assert_mpmath_equal(sc.erf, lambda z: mpmath.erf(z), [ComplexArg()], n=200)

    def test_erfc(self):
        if False:
            i = 10
            return i + 15
        assert_mpmath_equal(sc.erfc, exception_to_nan(lambda z: mpmath.erfc(z)), [Arg()], rtol=1e-13)

    def test_erfc_complex(self):
        if False:
            return 10
        assert_mpmath_equal(sc.erfc, exception_to_nan(lambda z: mpmath.erfc(z)), [ComplexArg()], n=200)

    def test_erfi(self):
        if False:
            return 10
        assert_mpmath_equal(sc.erfi, mpmath.erfi, [Arg()], n=200)

    def test_erfi_complex(self):
        if False:
            for i in range(10):
                print('nop')
        assert_mpmath_equal(sc.erfi, mpmath.erfi, [ComplexArg()], n=200)

    def test_ndtr(self):
        if False:
            i = 10
            return i + 15
        assert_mpmath_equal(sc.ndtr, exception_to_nan(lambda z: mpmath.ncdf(z)), [Arg()], n=200)

    def test_ndtr_complex(self):
        if False:
            return 10
        assert_mpmath_equal(sc.ndtr, lambda z: mpmath.erfc(-z / np.sqrt(2.0)) / 2.0, [ComplexArg(a=complex(-10000, -10000), b=complex(10000, 10000))], n=400)

    def test_log_ndtr(self):
        if False:
            i = 10
            return i + 15
        assert_mpmath_equal(sc.log_ndtr, exception_to_nan(lambda z: mpmath.log(mpmath.ncdf(z))), [Arg()], n=600, dps=300, rtol=1e-13)

    def test_log_ndtr_complex(self):
        if False:
            i = 10
            return i + 15
        assert_mpmath_equal(sc.log_ndtr, exception_to_nan(lambda z: mpmath.log(mpmath.erfc(-z / np.sqrt(2.0)) / 2.0)), [ComplexArg(a=complex(-10000, -100), b=complex(10000, 100))], n=200, dps=300)

    def test_eulernum(self):
        if False:
            i = 10
            return i + 15
        assert_mpmath_equal(lambda n: sc.euler(n)[-1], mpmath.eulernum, [IntArg(1, 10000)], n=10000)

    def test_expint(self):
        if False:
            i = 10
            return i + 15
        assert_mpmath_equal(sc.expn, mpmath.expint, [IntArg(0, 200), Arg(0, np.inf)], rtol=1e-13, dps=160)

    def test_fresnels(self):
        if False:
            while True:
                i = 10

        def fresnels(x):
            if False:
                for i in range(10):
                    print('nop')
            return sc.fresnel(x)[0]
        assert_mpmath_equal(fresnels, mpmath.fresnels, [Arg()])

    def test_fresnelc(self):
        if False:
            print('Hello World!')

        def fresnelc(x):
            if False:
                while True:
                    i = 10
            return sc.fresnel(x)[1]
        assert_mpmath_equal(fresnelc, mpmath.fresnelc, [Arg()])

    def test_gamma(self):
        if False:
            while True:
                i = 10
        assert_mpmath_equal(sc.gamma, exception_to_nan(mpmath.gamma), [Arg()])

    def test_gamma_complex(self):
        if False:
            print('Hello World!')
        assert_mpmath_equal(sc.gamma, exception_to_nan(mpmath.gamma), [ComplexArg()], rtol=5e-13)

    def test_gammainc(self):
        if False:
            for i in range(10):
                print('nop')
        assert_mpmath_equal(sc.gammainc, lambda z, b: mpmath.gammainc(z, b=b, regularized=True), [Arg(0, 10000.0, inclusive_a=False), Arg(0, 10000.0)], nan_ok=False, rtol=1e-11)

    def test_gammaincc(self):
        if False:
            i = 10
            return i + 15
        assert_mpmath_equal(sc.gammaincc, lambda z, a: mpmath.gammainc(z, a=a, regularized=True), [Arg(0, 10000.0, inclusive_a=False), Arg(0, 10000.0)], nan_ok=False, rtol=1e-11)

    def test_gammaln(self):
        if False:
            for i in range(10):
                print('nop')

        def f(z):
            if False:
                return 10
            return mpmath.loggamma(z).real
        assert_mpmath_equal(sc.gammaln, exception_to_nan(f), [Arg()])

    @pytest.mark.xfail(run=False)
    def test_gegenbauer(self):
        if False:
            print('Hello World!')
        assert_mpmath_equal(sc.eval_gegenbauer, exception_to_nan(mpmath.gegenbauer), [Arg(-1000.0, 1000.0), Arg(), Arg()])

    def test_gegenbauer_int(self):
        if False:
            i = 10
            return i + 15

        def gegenbauer(n, a, x):
            if False:
                i = 10
                return i + 15
            if abs(a) > 1e+100:
                return np.nan
            if n == 0:
                r = 1.0
            elif n == 1:
                r = 2 * a * x
            else:
                r = mpmath.gegenbauer(n, a, x)
            if float(r) == 0 and a < -1 and (float(a) == int(float(a))):
                r = mpmath.gegenbauer(n, a + mpmath.mpf('1e-50'), x)
                if abs(r) < mpmath.mpf('1e-50'):
                    r = mpmath.mpf('0.0')
            if abs(r) > 1e+270:
                return np.inf
            return r

        def sc_gegenbauer(n, a, x):
            if False:
                i = 10
                return i + 15
            r = sc.eval_gegenbauer(int(n), a, x)
            if abs(r) > 1e+270:
                return np.inf
            return r
        assert_mpmath_equal(sc_gegenbauer, exception_to_nan(gegenbauer), [IntArg(0, 100), Arg(-1000000000.0, 1000000000.0), Arg()], n=40000, dps=100, ignore_inf_sign=True, rtol=1e-06)
        assert_mpmath_equal(sc_gegenbauer, exception_to_nan(gegenbauer), [IntArg(0, 100), Arg(), FixedArg(np.logspace(-30, -4, 30))], dps=100, ignore_inf_sign=True)

    @pytest.mark.xfail(run=False)
    def test_gegenbauer_complex(self):
        if False:
            return 10
        assert_mpmath_equal(lambda n, a, x: sc.eval_gegenbauer(int(n), a.real, x), exception_to_nan(mpmath.gegenbauer), [IntArg(0, 100), Arg(), ComplexArg()])

    @nonfunctional_tooslow
    def test_gegenbauer_complex_general(self):
        if False:
            while True:
                i = 10
        assert_mpmath_equal(lambda n, a, x: sc.eval_gegenbauer(n.real, a.real, x), exception_to_nan(mpmath.gegenbauer), [Arg(-1000.0, 1000.0), Arg(), ComplexArg()])

    def test_hankel1(self):
        if False:
            while True:
                i = 10
        assert_mpmath_equal(sc.hankel1, exception_to_nan(lambda v, x: mpmath.hankel1(v, x, **HYPERKW)), [Arg(-1e+20, 1e+20), Arg()])

    def test_hankel2(self):
        if False:
            for i in range(10):
                print('nop')
        assert_mpmath_equal(sc.hankel2, exception_to_nan(lambda v, x: mpmath.hankel2(v, x, **HYPERKW)), [Arg(-1e+20, 1e+20), Arg()])

    @pytest.mark.xfail(run=False, reason='issues at intermediately large orders')
    def test_hermite(self):
        if False:
            i = 10
            return i + 15
        assert_mpmath_equal(lambda n, x: sc.eval_hermite(int(n), x), exception_to_nan(mpmath.hermite), [IntArg(0, 10000), Arg()])

    def test_hyp0f1(self):
        if False:
            while True:
                i = 10
        KW = dict(maxprec=400, maxterms=1500)
        assert_mpmath_equal(sc.hyp0f1, lambda a, x: mpmath.hyp0f1(a, x, **KW), [Arg(-10000000.0, 10000000.0), Arg(0, 100000.0)], n=5000)

    def test_hyp0f1_complex(self):
        if False:
            return 10
        assert_mpmath_equal(lambda a, z: sc.hyp0f1(a.real, z), exception_to_nan(lambda a, x: mpmath.hyp0f1(a, x, **HYPERKW)), [Arg(-10, 10), ComplexArg(complex(-120, -120), complex(120, 120))])

    def test_hyp1f1(self):
        if False:
            for i in range(10):
                print('nop')

        def mpmath_hyp1f1(a, b, x):
            if False:
                return 10
            try:
                return mpmath.hyp1f1(a, b, x)
            except ZeroDivisionError:
                return np.inf
        assert_mpmath_equal(sc.hyp1f1, mpmath_hyp1f1, [Arg(-50, 50), Arg(1, 50, inclusive_a=False), Arg(-50, 50)], n=500, nan_ok=False)

    @pytest.mark.xfail(run=False)
    def test_hyp1f1_complex(self):
        if False:
            print('Hello World!')
        assert_mpmath_equal(inf_to_nan(lambda a, b, x: sc.hyp1f1(a.real, b.real, x)), exception_to_nan(lambda a, b, x: mpmath.hyp1f1(a, b, x, **HYPERKW)), [Arg(-1000.0, 1000.0), Arg(-1000.0, 1000.0), ComplexArg()], n=2000)

    @nonfunctional_tooslow
    def test_hyp2f1_complex(self):
        if False:
            while True:
                i = 10
        assert_mpmath_equal(lambda a, b, c, x: sc.hyp2f1(a.real, b.real, c.real, x), exception_to_nan(lambda a, b, c, x: mpmath.hyp2f1(a, b, c, x, **HYPERKW)), [Arg(-100.0, 100.0), Arg(-100.0, 100.0), Arg(-100.0, 100.0), ComplexArg()], n=10)

    @pytest.mark.xfail(run=False)
    def test_hyperu(self):
        if False:
            print('Hello World!')
        assert_mpmath_equal(sc.hyperu, exception_to_nan(lambda a, b, x: mpmath.hyperu(a, b, x, **HYPERKW)), [Arg(), Arg(), Arg()])

    @pytest.mark.xfail_on_32bit('mpmath issue gh-342: unsupported operand mpz, long for pow')
    def test_igam_fac(self):
        if False:
            for i in range(10):
                print('nop')

        def mp_igam_fac(a, x):
            if False:
                return 10
            return mpmath.power(x, a) * mpmath.exp(-x) / mpmath.gamma(a)
        assert_mpmath_equal(_igam_fac, mp_igam_fac, [Arg(0, 100000000000000.0, inclusive_a=False), Arg(0, 100000000000000.0)], rtol=1e-10)

    def test_j0(self):
        if False:
            for i in range(10):
                print('nop')
        assert_mpmath_equal(sc.j0, mpmath.j0, [Arg(-1000.0, 1000.0)])
        assert_mpmath_equal(sc.j0, mpmath.j0, [Arg(-100000000.0, 100000000.0)], rtol=1e-05)

    def test_j1(self):
        if False:
            for i in range(10):
                print('nop')
        assert_mpmath_equal(sc.j1, mpmath.j1, [Arg(-1000.0, 1000.0)])
        assert_mpmath_equal(sc.j1, mpmath.j1, [Arg(-100000000.0, 100000000.0)], rtol=1e-05)

    @pytest.mark.xfail(run=False)
    def test_jacobi(self):
        if False:
            print('Hello World!')
        assert_mpmath_equal(sc.eval_jacobi, exception_to_nan(lambda a, b, c, x: mpmath.jacobi(a, b, c, x, **HYPERKW)), [Arg(), Arg(), Arg(), Arg()])
        assert_mpmath_equal(lambda n, b, c, x: sc.eval_jacobi(int(n), b, c, x), exception_to_nan(lambda a, b, c, x: mpmath.jacobi(a, b, c, x, **HYPERKW)), [IntArg(), Arg(), Arg(), Arg()])

    def test_jacobi_int(self):
        if False:
            return 10

        def jacobi(n, a, b, x):
            if False:
                return 10
            if n == 0:
                return 1.0
            return mpmath.jacobi(n, a, b, x)
        assert_mpmath_equal(lambda n, a, b, x: sc.eval_jacobi(int(n), a, b, x), lambda n, a, b, x: exception_to_nan(jacobi)(n, a, b, x, **HYPERKW), [IntArg(), Arg(), Arg(), Arg()], n=20000, dps=50)

    def test_kei(self):
        if False:
            for i in range(10):
                print('nop')

        def kei(x):
            if False:
                for i in range(10):
                    print('nop')
            if x == 0:
                return -pi / 4
            return exception_to_nan(mpmath.kei)(0, x, **HYPERKW)
        assert_mpmath_equal(sc.kei, kei, [Arg(-1e+30, 1e+30)], n=1000)

    def test_ker(self):
        if False:
            while True:
                i = 10
        assert_mpmath_equal(sc.ker, exception_to_nan(lambda x: mpmath.ker(0, x, **HYPERKW)), [Arg(-1e+30, 1e+30)], n=1000)

    @nonfunctional_tooslow
    def test_laguerre(self):
        if False:
            i = 10
            return i + 15
        assert_mpmath_equal(trace_args(sc.eval_laguerre), lambda n, x: exception_to_nan(mpmath.laguerre)(n, x, **HYPERKW), [Arg(), Arg()])

    def test_laguerre_int(self):
        if False:
            i = 10
            return i + 15
        assert_mpmath_equal(lambda n, x: sc.eval_laguerre(int(n), x), lambda n, x: exception_to_nan(mpmath.laguerre)(n, x, **HYPERKW), [IntArg(), Arg()], n=20000)

    @pytest.mark.xfail_on_32bit('see gh-3551 for bad points')
    def test_lambertw_real(self):
        if False:
            for i in range(10):
                print('nop')
        assert_mpmath_equal(lambda x, k: sc.lambertw(x, int(k.real)), lambda x, k: mpmath.lambertw(x, int(k.real)), [ComplexArg(-np.inf, np.inf), IntArg(0, 10)], rtol=1e-13, nan_ok=False)

    def test_lanczos_sum_expg_scaled(self):
        if False:
            while True:
                i = 10
        maxgamma = 171.6243769563027
        e = np.exp(1)
        g = 6.02468004077673

        def gamma(x):
            if False:
                i = 10
                return i + 15
            with np.errstate(over='ignore'):
                fac = ((x + g - 0.5) / e) ** (x - 0.5)
                if fac != np.inf:
                    res = fac * _lanczos_sum_expg_scaled(x)
                else:
                    fac = ((x + g - 0.5) / e) ** (0.5 * (x - 0.5))
                    res = fac * _lanczos_sum_expg_scaled(x)
                    res *= fac
            return res
        assert_mpmath_equal(gamma, mpmath.gamma, [Arg(0, maxgamma, inclusive_a=False)], rtol=1e-13)

    @nonfunctional_tooslow
    def test_legendre(self):
        if False:
            return 10
        assert_mpmath_equal(sc.eval_legendre, mpmath.legendre, [Arg(), Arg()])

    def test_legendre_int(self):
        if False:
            while True:
                i = 10
        assert_mpmath_equal(lambda n, x: sc.eval_legendre(int(n), x), lambda n, x: exception_to_nan(mpmath.legendre)(n, x, **HYPERKW), [IntArg(), Arg()], n=20000)
        assert_mpmath_equal(lambda n, x: sc.eval_legendre(int(n), x), lambda n, x: exception_to_nan(mpmath.legendre)(n, x, **HYPERKW), [IntArg(), FixedArg(np.logspace(-30, -4, 20))])

    def test_legenp(self):
        if False:
            i = 10
            return i + 15

        def lpnm(n, m, z):
            if False:
                while True:
                    i = 10
            try:
                v = sc.lpmn(m, n, z)[0][-1, -1]
            except ValueError:
                return np.nan
            if abs(v) > 1e+306:
                v = np.inf * np.sign(v.real)
            return v

        def lpnm_2(n, m, z):
            if False:
                print('Hello World!')
            v = sc.lpmv(m, n, z)
            if abs(v) > 1e+306:
                v = np.inf * np.sign(v.real)
            return v

        def legenp(n, m, z):
            if False:
                for i in range(10):
                    print('nop')
            if (z == 1 or z == -1) and int(n) == n:
                if m == 0:
                    if n < 0:
                        n = -n - 1
                    return mpmath.power(mpmath.sign(z), n)
                else:
                    return 0
            if abs(z) < 1e-15:
                return np.nan
            typ = 2 if abs(z) < 1 else 3
            v = exception_to_nan(mpmath.legenp)(n, m, z, type=typ)
            if abs(v) > 1e+306:
                v = mpmath.inf * mpmath.sign(v.real)
            return v
        assert_mpmath_equal(lpnm, legenp, [IntArg(-100, 100), IntArg(-100, 100), Arg()])
        assert_mpmath_equal(lpnm_2, legenp, [IntArg(-100, 100), Arg(-100, 100), Arg(-1, 1)], atol=1e-10)

    def test_legenp_complex_2(self):
        if False:
            i = 10
            return i + 15

        def clpnm(n, m, z):
            if False:
                for i in range(10):
                    print('nop')
            try:
                return sc.clpmn(m.real, n.real, z, type=2)[0][-1, -1]
            except ValueError:
                return np.nan

        def legenp(n, m, z):
            if False:
                return 10
            if abs(z) < 1e-15:
                return np.nan
            return exception_to_nan(mpmath.legenp)(int(n.real), int(m.real), z, type=2)
        x = np.array([-2, -0.99, -0.5, 0, 1e-05, 0.5, 0.99, 20, 2000.0])
        y = np.array([-1000.0, -0.5, 0.5, 1.3])
        z = (x[:, None] + 1j * y[None, :]).ravel()
        assert_mpmath_equal(clpnm, legenp, [FixedArg([-2, -1, 0, 1, 2, 10]), FixedArg([-2, -1, 0, 1, 2, 10]), FixedArg(z)], rtol=1e-06, n=500)

    def test_legenp_complex_3(self):
        if False:
            return 10

        def clpnm(n, m, z):
            if False:
                print('Hello World!')
            try:
                return sc.clpmn(m.real, n.real, z, type=3)[0][-1, -1]
            except ValueError:
                return np.nan

        def legenp(n, m, z):
            if False:
                while True:
                    i = 10
            if abs(z) < 1e-15:
                return np.nan
            return exception_to_nan(mpmath.legenp)(int(n.real), int(m.real), z, type=3)
        x = np.array([-2, -0.99, -0.5, 0, 1e-05, 0.5, 0.99, 20, 2000.0])
        y = np.array([-1000.0, -0.5, 0.5, 1.3])
        z = (x[:, None] + 1j * y[None, :]).ravel()
        assert_mpmath_equal(clpnm, legenp, [FixedArg([-2, -1, 0, 1, 2, 10]), FixedArg([-2, -1, 0, 1, 2, 10]), FixedArg(z)], rtol=1e-06, n=500)

    @pytest.mark.xfail(run=False, reason='apparently picks wrong function at |z| > 1')
    def test_legenq(self):
        if False:
            while True:
                i = 10

        def lqnm(n, m, z):
            if False:
                return 10
            return sc.lqmn(m, n, z)[0][-1, -1]

        def legenq(n, m, z):
            if False:
                i = 10
                return i + 15
            if abs(z) < 1e-15:
                return np.nan
            return exception_to_nan(mpmath.legenq)(n, m, z, type=2)
        assert_mpmath_equal(lqnm, legenq, [IntArg(0, 100), IntArg(0, 100), Arg()])

    @nonfunctional_tooslow
    def test_legenq_complex(self):
        if False:
            print('Hello World!')

        def lqnm(n, m, z):
            if False:
                for i in range(10):
                    print('nop')
            return sc.lqmn(int(m.real), int(n.real), z)[0][-1, -1]

        def legenq(n, m, z):
            if False:
                for i in range(10):
                    print('nop')
            if abs(z) < 1e-15:
                return np.nan
            return exception_to_nan(mpmath.legenq)(int(n.real), int(m.real), z, type=2)
        assert_mpmath_equal(lqnm, legenq, [IntArg(0, 100), IntArg(0, 100), ComplexArg()], n=100)

    def test_lgam1p(self):
        if False:
            print('Hello World!')

        def param_filter(x):
            if False:
                print('Hello World!')
            return np.where((np.floor(x) == x) & (x <= 0), False, True)

        def mp_lgam1p(z):
            if False:
                return 10
            return mpmath.loggamma(1 + z).real
        assert_mpmath_equal(_lgam1p, mp_lgam1p, [Arg()], rtol=1e-13, dps=100, param_filter=param_filter)

    def test_loggamma(self):
        if False:
            i = 10
            return i + 15

        def mpmath_loggamma(z):
            if False:
                for i in range(10):
                    print('nop')
            try:
                res = mpmath.loggamma(z)
            except ValueError:
                res = complex(np.nan, np.nan)
            return res
        assert_mpmath_equal(sc.loggamma, mpmath_loggamma, [ComplexArg()], nan_ok=False, distinguish_nan_and_inf=False, rtol=5e-14)

    @pytest.mark.xfail(run=False)
    def test_pcfd(self):
        if False:
            for i in range(10):
                print('nop')

        def pcfd(v, x):
            if False:
                for i in range(10):
                    print('nop')
            return sc.pbdv(v, x)[0]
        assert_mpmath_equal(pcfd, exception_to_nan(lambda v, x: mpmath.pcfd(v, x, **HYPERKW)), [Arg(), Arg()])

    @pytest.mark.xfail(run=False, reason="it's not the same as the mpmath function --- maybe different definition?")
    def test_pcfv(self):
        if False:
            while True:
                i = 10

        def pcfv(v, x):
            if False:
                for i in range(10):
                    print('nop')
            return sc.pbvv(v, x)[0]
        assert_mpmath_equal(pcfv, lambda v, x: time_limited()(exception_to_nan(mpmath.pcfv))(v, x, **HYPERKW), [Arg(), Arg()], n=1000)

    def test_pcfw(self):
        if False:
            i = 10
            return i + 15

        def pcfw(a, x):
            if False:
                for i in range(10):
                    print('nop')
            return sc.pbwa(a, x)[0]

        def dpcfw(a, x):
            if False:
                for i in range(10):
                    print('nop')
            return sc.pbwa(a, x)[1]

        def mpmath_dpcfw(a, x):
            if False:
                print('Hello World!')
            return mpmath.diff(mpmath.pcfw, (a, x), (0, 1))
        assert_mpmath_equal(pcfw, mpmath.pcfw, [Arg(-5, 5), Arg(-5, 5)], rtol=2e-08, n=100)
        assert_mpmath_equal(dpcfw, mpmath_dpcfw, [Arg(-5, 5), Arg(-5, 5)], rtol=2e-09, n=100)

    @pytest.mark.xfail(run=False, reason='issues at large arguments (atol OK, rtol not) and <eps-close to z=0')
    def test_polygamma(self):
        if False:
            return 10
        assert_mpmath_equal(sc.polygamma, time_limited()(exception_to_nan(mpmath.polygamma)), [IntArg(0, 1000), Arg()])

    def test_rgamma(self):
        if False:
            print('Hello World!')
        assert_mpmath_equal(sc.rgamma, mpmath.rgamma, [Arg(-8000, np.inf)], n=5000, nan_ok=False, ignore_inf_sign=True)

    def test_rgamma_complex(self):
        if False:
            print('Hello World!')
        assert_mpmath_equal(sc.rgamma, exception_to_nan(mpmath.rgamma), [ComplexArg()], rtol=5e-13)

    @pytest.mark.xfail(reason='see gh-3551 for bad points on 32 bit systems and gh-8095 for another bad point')
    def test_rf(self):
        if False:
            while True:
                i = 10
        if _pep440.parse(mpmath.__version__) >= _pep440.Version('1.0.0'):
            mppoch = mpmath.rf
        else:

            def mppoch(a, m):
                if False:
                    return 10
                if float(a + m) == int(a + m) and float(a + m) <= 0:
                    a = mpmath.mpf(a)
                    m = int(a + m) - a
                return mpmath.rf(a, m)
        assert_mpmath_equal(sc.poch, mppoch, [Arg(), Arg()], dps=400)

    def test_sinpi(self):
        if False:
            i = 10
            return i + 15
        eps = np.finfo(float).eps
        assert_mpmath_equal(_sinpi, mpmath.sinpi, [Arg()], nan_ok=False, rtol=2 * eps)

    def test_sinpi_complex(self):
        if False:
            i = 10
            return i + 15
        assert_mpmath_equal(_sinpi, mpmath.sinpi, [ComplexArg()], nan_ok=False, rtol=2e-14)

    def test_shi(self):
        if False:
            while True:
                i = 10

        def shi(x):
            if False:
                print('Hello World!')
            return sc.shichi(x)[0]
        assert_mpmath_equal(shi, mpmath.shi, [Arg()])
        assert_mpmath_equal(shi, mpmath.shi, [FixedArg([88 - 1e-09, 88, 88 + 1e-09])])

    def test_shi_complex(self):
        if False:
            for i in range(10):
                print('nop')

        def shi(z):
            if False:
                while True:
                    i = 10
            return sc.shichi(z)[0]
        assert_mpmath_equal(shi, mpmath.shi, [ComplexArg(complex(-np.inf, -100000000.0), complex(np.inf, 100000000.0))], rtol=1e-12)

    def test_si(self):
        if False:
            return 10

        def si(x):
            if False:
                return 10
            return sc.sici(x)[0]
        assert_mpmath_equal(si, mpmath.si, [Arg()])

    def test_si_complex(self):
        if False:
            while True:
                i = 10

        def si(z):
            if False:
                for i in range(10):
                    print('nop')
            return sc.sici(z)[0]
        assert_mpmath_equal(si, mpmath.si, [ComplexArg(complex(-100000000.0, -np.inf), complex(100000000.0, np.inf))], rtol=1e-12)

    def test_spence(self):
        if False:
            print('Hello World!')

        def dilog(x):
            if False:
                print('Hello World!')
            return mpmath.polylog(2, 1 - x)
        assert_mpmath_equal(sc.spence, exception_to_nan(dilog), [Arg(0, np.inf)], rtol=1e-14)

    def test_spence_complex(self):
        if False:
            return 10

        def dilog(z):
            if False:
                while True:
                    i = 10
            return mpmath.polylog(2, 1 - z)
        assert_mpmath_equal(sc.spence, exception_to_nan(dilog), [ComplexArg()], rtol=1e-14)

    def test_spherharm(self):
        if False:
            return 10

        def spherharm(l, m, theta, phi):
            if False:
                for i in range(10):
                    print('nop')
            if m > l:
                return np.nan
            return sc.sph_harm(m, l, phi, theta)
        assert_mpmath_equal(spherharm, mpmath.spherharm, [IntArg(0, 100), IntArg(0, 100), Arg(a=0, b=pi), Arg(a=0, b=2 * pi)], atol=1e-08, n=6000, dps=150)

    def test_struveh(self):
        if False:
            return 10
        assert_mpmath_equal(sc.struve, exception_to_nan(mpmath.struveh), [Arg(-10000.0, 10000.0), Arg(0, 10000.0)], rtol=5e-10)

    def test_struvel(self):
        if False:
            i = 10
            return i + 15

        def mp_struvel(v, z):
            if False:
                i = 10
                return i + 15
            if v < 0 and z < -v and (abs(v) > 1000):
                old_dps = mpmath.mp.dps
                try:
                    mpmath.mp.dps = 300
                    return mpmath.struvel(v, z)
                finally:
                    mpmath.mp.dps = old_dps
            return mpmath.struvel(v, z)
        assert_mpmath_equal(sc.modstruve, exception_to_nan(mp_struvel), [Arg(-10000.0, 10000.0), Arg(0, 10000.0)], rtol=5e-10, ignore_inf_sign=True)

    def test_wrightomega_real(self):
        if False:
            for i in range(10):
                print('nop')

        def mpmath_wrightomega_real(x):
            if False:
                for i in range(10):
                    print('nop')
            return mpmath.lambertw(mpmath.exp(x), mpmath.mpf('-0.5'))
        assert_mpmath_equal(sc.wrightomega, mpmath_wrightomega_real, [Arg(-1000, 1e+21)], rtol=5e-15, atol=0, nan_ok=False)

    def test_wrightomega(self):
        if False:
            for i in range(10):
                print('nop')
        assert_mpmath_equal(sc.wrightomega, lambda z: _mpmath_wrightomega(z, 25), [ComplexArg()], rtol=1e-14, nan_ok=False)

    def test_hurwitz_zeta(self):
        if False:
            return 10
        assert_mpmath_equal(sc.zeta, exception_to_nan(mpmath.zeta), [Arg(a=1, b=10000000000.0, inclusive_a=False), Arg(a=0, inclusive_a=False)])

    def test_riemann_zeta(self):
        if False:
            while True:
                i = 10
        assert_mpmath_equal(sc.zeta, lambda x: mpmath.zeta(x) if x != 1 else mpmath.inf, [Arg(-100, 100)], nan_ok=False, rtol=5e-13)

    def test_zetac(self):
        if False:
            for i in range(10):
                print('nop')
        assert_mpmath_equal(sc.zetac, lambda x: mpmath.zeta(x) - 1 if x != 1 else mpmath.inf, [Arg(-100, 100)], nan_ok=False, dps=45, rtol=5e-13)

    def test_boxcox(self):
        if False:
            return 10

        def mp_boxcox(x, lmbda):
            if False:
                i = 10
                return i + 15
            x = mpmath.mp.mpf(x)
            lmbda = mpmath.mp.mpf(lmbda)
            if lmbda == 0:
                return mpmath.mp.log(x)
            else:
                return mpmath.mp.powm1(x, lmbda) / lmbda
        assert_mpmath_equal(sc.boxcox, exception_to_nan(mp_boxcox), [Arg(a=0, inclusive_a=False), Arg()], n=200, dps=60, rtol=1e-13)

    def test_boxcox1p(self):
        if False:
            while True:
                i = 10

        def mp_boxcox1p(x, lmbda):
            if False:
                print('Hello World!')
            x = mpmath.mp.mpf(x)
            lmbda = mpmath.mp.mpf(lmbda)
            one = mpmath.mp.mpf(1)
            if lmbda == 0:
                return mpmath.mp.log(one + x)
            else:
                return mpmath.mp.powm1(one + x, lmbda) / lmbda
        assert_mpmath_equal(sc.boxcox1p, exception_to_nan(mp_boxcox1p), [Arg(a=-1, inclusive_a=False), Arg()], n=200, dps=60, rtol=1e-13)

    def test_spherical_jn(self):
        if False:
            for i in range(10):
                print('nop')

        def mp_spherical_jn(n, z):
            if False:
                return 10
            arg = mpmath.mpmathify(z)
            out = mpmath.besselj(n + mpmath.mpf(1) / 2, arg) / mpmath.sqrt(2 * arg / mpmath.pi)
            if arg.imag == 0:
                return out.real
            else:
                return out
        assert_mpmath_equal(lambda n, z: sc.spherical_jn(int(n), z), exception_to_nan(mp_spherical_jn), [IntArg(0, 200), Arg(-100000000.0, 100000000.0)], dps=300)

    def test_spherical_jn_complex(self):
        if False:
            return 10

        def mp_spherical_jn(n, z):
            if False:
                for i in range(10):
                    print('nop')
            arg = mpmath.mpmathify(z)
            out = mpmath.besselj(n + mpmath.mpf(1) / 2, arg) / mpmath.sqrt(2 * arg / mpmath.pi)
            if arg.imag == 0:
                return out.real
            else:
                return out
        assert_mpmath_equal(lambda n, z: sc.spherical_jn(int(n.real), z), exception_to_nan(mp_spherical_jn), [IntArg(0, 200), ComplexArg()])

    def test_spherical_yn(self):
        if False:
            print('Hello World!')

        def mp_spherical_yn(n, z):
            if False:
                while True:
                    i = 10
            arg = mpmath.mpmathify(z)
            out = mpmath.bessely(n + mpmath.mpf(1) / 2, arg) / mpmath.sqrt(2 * arg / mpmath.pi)
            if arg.imag == 0:
                return out.real
            else:
                return out
        assert_mpmath_equal(lambda n, z: sc.spherical_yn(int(n), z), exception_to_nan(mp_spherical_yn), [IntArg(0, 200), Arg(-10000000000.0, 10000000000.0)], dps=100)

    def test_spherical_yn_complex(self):
        if False:
            return 10

        def mp_spherical_yn(n, z):
            if False:
                for i in range(10):
                    print('nop')
            arg = mpmath.mpmathify(z)
            out = mpmath.bessely(n + mpmath.mpf(1) / 2, arg) / mpmath.sqrt(2 * arg / mpmath.pi)
            if arg.imag == 0:
                return out.real
            else:
                return out
        assert_mpmath_equal(lambda n, z: sc.spherical_yn(int(n.real), z), exception_to_nan(mp_spherical_yn), [IntArg(0, 200), ComplexArg()])

    def test_spherical_in(self):
        if False:
            while True:
                i = 10

        def mp_spherical_in(n, z):
            if False:
                for i in range(10):
                    print('nop')
            arg = mpmath.mpmathify(z)
            out = mpmath.besseli(n + mpmath.mpf(1) / 2, arg) / mpmath.sqrt(2 * arg / mpmath.pi)
            if arg.imag == 0:
                return out.real
            else:
                return out
        assert_mpmath_equal(lambda n, z: sc.spherical_in(int(n), z), exception_to_nan(mp_spherical_in), [IntArg(0, 200), Arg()], dps=200, atol=10 ** (-278))

    def test_spherical_in_complex(self):
        if False:
            for i in range(10):
                print('nop')

        def mp_spherical_in(n, z):
            if False:
                print('Hello World!')
            arg = mpmath.mpmathify(z)
            out = mpmath.besseli(n + mpmath.mpf(1) / 2, arg) / mpmath.sqrt(2 * arg / mpmath.pi)
            if arg.imag == 0:
                return out.real
            else:
                return out
        assert_mpmath_equal(lambda n, z: sc.spherical_in(int(n.real), z), exception_to_nan(mp_spherical_in), [IntArg(0, 200), ComplexArg()])

    def test_spherical_kn(self):
        if False:
            for i in range(10):
                print('nop')

        def mp_spherical_kn(n, z):
            if False:
                print('Hello World!')
            out = mpmath.besselk(n + mpmath.mpf(1) / 2, z) * mpmath.sqrt(mpmath.pi / (2 * mpmath.mpmathify(z)))
            if mpmath.mpmathify(z).imag == 0:
                return out.real
            else:
                return out
        assert_mpmath_equal(lambda n, z: sc.spherical_kn(int(n), z), exception_to_nan(mp_spherical_kn), [IntArg(0, 150), Arg()], dps=100)

    @pytest.mark.xfail(run=False, reason='Accuracy issues near z = -1 inherited from kv.')
    def test_spherical_kn_complex(self):
        if False:
            while True:
                i = 10

        def mp_spherical_kn(n, z):
            if False:
                return 10
            arg = mpmath.mpmathify(z)
            out = mpmath.besselk(n + mpmath.mpf(1) / 2, arg) / mpmath.sqrt(2 * arg / mpmath.pi)
            if arg.imag == 0:
                return out.real
            else:
                return out
        assert_mpmath_equal(lambda n, z: sc.spherical_kn(int(n.real), z), exception_to_nan(mp_spherical_kn), [IntArg(0, 200), ComplexArg()], dps=200)