from math import sqrt, pi
import cupy
import cupyx.scipy.signal as signal
from cupyx.scipy.signal._iir_filter_conversions import _cplxreal
from cupy import testing
from cupy.testing import assert_array_almost_equal
import numpy as np
import pytest
from pytest import raises as assert_raises

@testing.with_requires('scipy')
class TestBilinear_zpk:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, xp, scp):
        if False:
            return 10
        z = [-2j, +2j]
        p = [-0.75, -0.5 - 0.5j, -0.5 + 0.5j]
        k = 3
        (z_d, p_d, k_d) = scp.signal.bilinear_zpk(z, p, k, 10)
        return (z_d, p_d, k_d)
        '\n        assert_allclose(sort(z_d), sort([(20-2j)/(20+2j), (20+2j)/(20-2j),\n                                         -1]))\n        assert_allclose(sort(p_d), sort([77/83,\n                                         (1j/2 + 39/2) / (41/2 - 1j/2),\n                                         (39/2 - 1j/2) / (1j/2 + 41/2), ]))\n        assert_allclose(k_d, 9696/69803)\n        '

@testing.with_requires('scipy')
class TestBilinear:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, xp, scp):
        if False:
            i = 10
            return i + 15
        b = [0.14879732743343033]
        a = [1, 0.5455223688052221, 0.14879732743343033]
        (b_z, a_z) = scp.signal.bilinear(b, a, 0.5)
        return (b_z, a_z)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic_2(self, xp, scp):
        if False:
            while True:
                i = 10
        b = [1, 0, 0.17407467530697837]
        a = [1, 0.1846057532615225, 0.17407467530697837]
        (b_z, a_z) = scp.signal.bilinear(b, a, 0.5)
        return (b_z, a_z)

@testing.with_requires('scipy')
class TestNormalize:

    def test_allclose(self):
        if False:
            return 10
        'Test for false positive on allclose in normalize() in\n        filter_design.py'
        b_matlab = cupy.array([2.150733144728282e-11, 1.720586515782626e-10, 6.02205280523919e-10, 1.204410561047838e-09, 1.505513201309798e-09, 1.204410561047838e-09, 6.02205280523919e-10, 1.720586515782626e-10, 2.150733144728282e-11])
        a_matlab = cupy.array([1.0, -7.782402035027959, 26.54354569747454, -51.82182531666387, 63.34127355102684, -49.63358186631157, 24.34862182949389, -6.836925348604676, 0.841293494444914])
        b_norm_in = cupy.array([1.5543135865293012e-06, 1.2434508692234413e-05, 4.352078042282045e-05, 8.70415608456409e-05, 0.00010880195105705122, 8.704156084564097e-05, 4.352078042282045e-05, 1.2434508692234413e-05, 1.5543135865293012e-06])
        a_norm_in = cupy.array([72269.02590912717, -562426.6143046797, 1918276.1917308895, -3745112.8364682454, 4577612.139376277, -3586970.6138592605, 1759651.1818472347, -494097.93515707983, 60799.46134721965])
        (b_output, a_output) = signal.normalize(b_norm_in, a_norm_in)
        assert_array_almost_equal(b_matlab, b_output, decimal=13)
        assert_array_almost_equal(a_matlab, a_output, decimal=13)

    def test_errors(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the error cases.'
        assert_raises(ValueError, signal.normalize, [1, 2], 0)
        assert_raises(ValueError, signal.normalize, [1, 2], [[1]])
        assert_raises(ValueError, signal.normalize, [[[1, 2]]], 1)

@testing.with_requires('scipy')
class TestLp2lp:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, xp, scp):
        if False:
            return 10
        b = [1]
        a = [1, float(xp.sqrt(2)), 1]
        (b_lp, a_lp) = scp.signal.lp2lp(b, a, 0.3857425662711212)
        return (b_lp, a_lp)

@testing.with_requires('scipy')
class TestLp2hp:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, xp, scp):
        if False:
            i = 10
            return i + 15
        b = [0.2505943232519002]
        a = [1, 0.5972404165413486, 0.9283480575752417, 0.2505943232519002]
        (b_hp, a_hp) = scp.signal.lp2hp(b, a, 2 * pi * 5000)
        return (b_hp, a_hp)

@testing.with_requires('scipy')
class TestLp2bp:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, xp, scp):
        if False:
            i = 10
            return i + 15
        b = [1]
        a = [1, 2, 2, 1]
        (b_bp, a_bp) = scp.signal.lp2bp(b, a, 2 * pi * 4000, 2 * pi * 2000)
        return (b_bp, a_bp)

@testing.with_requires('scipy')
class TestLp2bs:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        b = [1]
        a = [1, 1]
        (b_bs, a_bs) = scp.signal.lp2bs(b, a, 0.41722257286366754, 0.1846057532615225)
        return (b_bs, a_bs)

@testing.with_requires('scipy')
class TestLp2lp_zpk:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, xp, scp):
        if False:
            return 10
        z = []
        p = [(-1 + 1j) / sqrt(2), (-1 - 1j) / sqrt(2)]
        k = 1
        (z_lp, p_lp, k_lp) = scp.signal.lp2lp_zpk(z, p, k, 5)
        return (z_lp, p_lp, k_lp)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic_2(self, xp, scp):
        if False:
            print('Hello World!')
        z = [-2j, +2j]
        p = [-0.75, -0.5 - 0.5j, -0.5 + 0.5j]
        k = 3
        (z_lp, p_lp, k_lp) = scp.signal.lp2lp_zpk(z, p, k, 20)
        return (z_lp, p_lp, k_lp)

@testing.with_requires('scipy')
class TestLp2hp_zpk:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, xp, scp):
        if False:
            i = 10
            return i + 15
        z = []
        p = [(-1 + 1j) / np.sqrt(2), (-1 - 1j) / np.sqrt(2)]
        k = 1
        (z_hp, p_hp, k_hp) = scp.signal.lp2hp_zpk(z, p, k, 5)
        return (z_hp, p_hp, k_hp)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic_2(self, xp, scp):
        if False:
            return 10
        z = [-2j, +2j]
        p = [-0.75, -0.5 - 0.5j, -0.5 + 0.5j]
        k = 3
        (z_hp, p_hp, k_hp) = scp.signal.lp2hp_zpk(z, p, k, 6)
        return (z_hp, p_hp, k_hp)

@testing.with_requires('scipy')
class TestLp2bp_zpk:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, xp, scp):
        if False:
            while True:
                i = 10
        z = [-2j, +2j]
        p = [-0.75, -0.5 - 0.5j, -0.5 + 0.5j]
        k = 3
        (z_bp, p_bp, k_bp) = scp.signal.lp2bp_zpk(z, p, k, 15, 8)
        return (z_bp, p_bp, k_bp)

@testing.with_requires('scipy')
class TestLp2bs_zpk:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, xp, scp):
        if False:
            print('Hello World!')
        z = [-2j, +2j]
        p = [-0.75, -0.5 - 0.5j, -0.5 + 0.5j]
        k = 3
        (z_bs, p_bs, k_bs) = scp.signal.lp2bs_zpk(z, p, k, 35, 12)
        z_bs_s = z_bs[xp.argsort(z_bs.imag)]
        p_bs_s = p_bs[xp.argsort(p_bs.imag)]
        return (z_bs_s, p_bs_s, k_bs)

@testing.with_requires('scipy >= 1.8.0')
class TestZpk2Sos:

    @pytest.mark.parametrize('dt', 'fdFD')
    @pytest.mark.parametrize('pairing, analog', [('nearest', False), ('keep_odd', False), ('minimal', False), ('minimal', True)])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_dtypes(self, dt, pairing, analog, xp, scp):
        if False:
            print('Hello World!')
        z = xp.array([-1, -1]).astype(dt)
        ct = dt.upper()
        p = xp.array([0.57149 + 0.2936j, 0.57149 - 0.2936j]).astype(ct)
        k = xp.array(1).astype(dt)
        sos = scp.signal.zpk2sos(z, p, k, pairing=pairing, analog=analog)
        return sos

    @pytest.mark.parametrize('case', [([-1, -1], [0.57149 + 0.2936j, 0.57149 - 0.2936j], 1), ([1j, -1j], [0.9, -0.9, 0.7j, -0.7j], 1), ([], [0.8, -0.5 + 0.25j, -0.5 - 0.25j], 1), ([1.0, 1.0, 0.9j, -0.9j], [0.99 + 0.01j, 0.99 - 0.01j, 0.1 + 0.9j, 0.1 - 0.9j], 1), ([0.9 + 0.1j, 0.9 - 0.1j, -0.9], [0.75 + 0.25j, 0.75 - 0.25j, 0.9], 1), ([-0.309 + 0.9511j, -0.309 - 0.9511j, 0.809 + 0.5878j, +0.809 - 0.5878j, -1.0 + 0j], [-0.3026 + 0.9312j, -0.3026 - 0.9312j, 0.7922 + 0.5755j, +0.7922 - 0.5755j, -0.9791 + 0j], 1), ([-1 - 1.4142j, -1 + 1.4142j, -0.625 - 1.0533j, -0.625 + 1.0533j], [-0.2 - 0.6782j, -0.2 + 0.6782j, -0.1 - 0.5385j, -0.1 + 0.5385j], 4), ([], [0.2, -0.5 + 0.25j, -0.5 - 0.25j], 1.0)])
    @pytest.mark.parametrize('pairing', ['nearest', 'keep_odd'])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-12)
    def test_basic(self, case, pairing, xp, scp):
        if False:
            return 10
        (z, p, k) = case
        z = xp.asarray(z)
        p = xp.asarray(p)
        sos = scp.signal.zpk2sos(z, p, k, pairing=pairing)
        return sos

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-12)
    def test_basic_2(self, xp, scp):
        if False:
            while True:
                i = 10
        deg2rad = np.pi / 180.0
        k = 1.0
        thetas = [22.5, 45, 77.5]
        mags = [0.8, 0.6, 0.9]
        z = xp.array([xp.exp(theta * deg2rad * 1j) for theta in thetas])
        z = xp.concatenate((z, z.conj()))
        p = xp.array([mag * xp.exp(theta * deg2rad * 1j) for (theta, mag) in zip(thetas, mags)])
        p = xp.concatenate((p, p.conj()))
        sos_1 = scp.signal.zpk2sos(z, p, k)
        z = xp.array([xp.exp(theta * deg2rad * 1j) for theta in (85.0, 10.0)])
        z = xp.concatenate((z, z.conj(), xp.array([1, -1])))
        sos_2 = scp.signal.zpk2sos(z, p, k)
        return (sos_1, sos_2)

    @pytest.mark.parametrize('pairing', ['nearest', 'keep_odd', 'minimal'])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-12)
    def test_pairing(self, pairing, xp, scp):
        if False:
            while True:
                i = 10
        z1 = xp.array([-1, -0.5 - 0.5j, -0.5 + 0.5j])
        p1 = xp.array([0.75, 0.8 + 0.1j, 0.8 - 0.1j])
        sos2 = scp.signal.zpk2sos(z1, p1, 1, pairing=pairing)
        return sos2

    @pytest.mark.parametrize('p', [[-1, 1, -0.1, 0.1], [-0.7071 + 0.7071j, -0.7071 - 0.7071j, -0.1j, 0.1j]])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-12)
    def test_analog(self, p, xp, scp):
        if False:
            return 10
        p = xp.asarray(p)
        sos2_dt = scp.signal.zpk2sos([], p, 1, pairing='minimal', analog=False)
        sos2_ct = scp.signal.zpk2sos([], p, 1, pairing='minimal', analog=True)
        return (sos2_dt, sos2_ct)

    def test_bad_args(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValueError, match='pairing must be one of'):
            signal.zpk2sos(cupy.array([1]), cupy.array([2]), 1, pairing='no_such_pairing')
        with pytest.raises(ValueError, match='.*pairing must be "minimal"'):
            signal.zpk2sos(cupy.array([1]), cupy.array([2]), 1, pairing='keep_odd', analog=True)
        with pytest.raises(ValueError, match='.*must have len\\(p\\)>=len\\(z\\)'):
            signal.zpk2sos(cupy.array([1, 1]), cupy.array([2]), 1, analog=True)
        with pytest.raises(ValueError, match='k must be real'):
            signal.zpk2sos(cupy.array([1]), cupy.array([2]), k=1j)

@testing.with_requires('scipy')
class TestTf2zpk:

    @pytest.mark.parametrize('dt', ('float64', 'complex128'))
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_simple(self, xp, scp, dt):
        if False:
            for i in range(10):
                print('nop')
        z_r = xp.array([0.5, -0.5])
        p_r = xp.array([1j / sqrt(2), -1j / sqrt(2)])
        b = xp.poly(z_r).astype(dt)
        a = xp.poly(p_r).real.astype(dt)
        (z, p, k) = scp.signal.tf2zpk(b, a)
        z.sort()
        p = p[xp.argsort(p.imag)]
        return (z, p, k)

@testing.with_requires('scipy')
class TestSS2TF:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, xp, scp):
        if False:
            return 10
        b = xp.array([1.0, 3.0, 5.0])
        a = xp.array([1.0, 2.0, 3.0])
        (A, B, C, D) = scp.signal.tf2ss(b, a)
        return (A, B, C, D)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic_2(self, xp, scp):
        if False:
            return 10
        b = xp.array([1.0, 3.0, 5.0])
        a = xp.array([1.0, 2.0, 3.0])
        (A, B, C, D) = scp.signal.tf2ss(b, a)
        (bb, aa) = scp.signal.ss2tf(A, B, C, D)
        return (bb, aa)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zero_order_round_trip(self, xp, scp):
        if False:
            while True:
                i = 10
        tf = (2, 1)
        (A, B, C, D) = scp.signal.tf2ss(*tf)
        return (A, B, C, D)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zero_order_round_trip_2(self, xp, scp):
        if False:
            i = 10
            return i + 15
        tf = (2, 1)
        (A, B, C, D) = scp.signal.tf2ss(*tf)
        (num, den) = scp.signal.ss2tf(A, B, C, D)
        return (num, den)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zero_order_round_trip_3(self, xp, scp):
        if False:
            while True:
                i = 10
        tf = (xp.asarray([[5], [2]]), 1)
        (A, B, C, D) = scp.signal.tf2ss(*tf)
        return (A, B, C, D)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zero_order_round_trip_4(self, xp, scp):
        if False:
            print('Hello World!')
        tf = (xp.asarray([[5], [2]]), 1)
        (A, B, C, D) = scp.signal.tf2ss(*tf)
        (num, den) = scp.signal.ss2tf(A, B, C, D)
        return (num, den)

    @pytest.mark.parametrize('tf', [([[1, 2], [1, 1]], [1, 2]), ([[1, 0, 1], [1, 1, 1]], [1, 1, 1]), ([[1, 2, 3], [1, 2, 3]], [1, 2, 3, 4])])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_simo_round_trip(self, xp, scp, tf):
        if False:
            print('Hello World!')
        tf = tuple((xp.asarray(x) for x in tf))
        (A, B, C, D) = scp.signal.tf2ss(*tf)
        return (A, B, C, D)

    @pytest.mark.parametrize('tf', [([[1, 2], [1, 1]], [1, 2]), ([[1, 0, 1], [1, 1, 1]], [1, 1, 1]), ([[1, 2, 3], [1, 2, 3]], [1, 2, 3, 4])])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_simo_round_trip_2(self, xp, scp, tf):
        if False:
            for i in range(10):
                print('nop')
        tf = tuple((xp.asarray(x) for x in tf))
        (A, B, C, D) = scp.signal.tf2ss(*tf)
        (num, den) = scp.signal.ss2tf(A, B, C, D)
        return (num, den)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_all_int_arrays(self, xp, scp):
        if False:
            return 10
        A = xp.asarray([[0, 1, 0], [0, 0, 1], [-3, -4, -2]])
        B = xp.asarray([[0], [0], [1]])
        C = xp.asarray([[5, 1, 0]])
        D = xp.asarray([[0]])
        (num, den) = scp.signal.ss2tf(A, B, C, D)
        return (num, den)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_multioutput(self, xp, scp):
        if False:
            while True:
                i = 10
        A = xp.array([[-1.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 2.0, 0.0], [-4.0, 0.0, 3.0, 0.0], [-8.0, 8.0, 0.0, 4.0]])
        B = xp.array([[0.3], [0.0], [7.0], [0.0]])
        C = xp.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [8.0, 8.0, 0.0, 0.0]])
        D = xp.array([[0.0], [0.0], [1.0]])
        (b_all, a) = scp.signal.ss2tf(A, B, C, D)
        return (b_all, a)

@testing.with_requires('scipy')
class TestSos2Zpk:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, xp, scp):
        if False:
            while True:
                i = 10
        sos = xp.asarray([[1, 0, 1, 1, 0, -0.81], [1, 0, 0, 1, 0, +0.49]])
        (z, p, k) = scp.signal.sos2zpk(sos)
        return (z, p, k)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic_2(self, xp, scp):
        if False:
            while True:
                i = 10
        sos = [[1.0, +0.61803, 1.0, 1.0, +0.60515, 0.95873], [1.0, -1.61803, 1.0, 1.0, -1.5843, 0.95873], [1.0, +1.0, 0.0, 1.0, +0.97915, 0.0]]
        sos = xp.asarray(sos)
        (z, p, k) = scp.signal.sos2zpk(sos)
        return (z, p, k)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic_3(self, xp, scp):
        if False:
            print('Hello World!')
        sos = xp.array([[1, 2, 3, 1, 0.2, 0.3], [4, 5, 6, 1, 0.4, 0.5]])
        (z2, p2, k2) = scp.signal.sos2zpk(sos)
        return (z2, p2, k2)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_fewer_zeros(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        'Test not the expected number of p/z (effectively at origin).'
        sos = scp.signal.butter(3, 0.1, output='sos')
        (z, p, k) = scp.signal.sos2zpk(sos)
        return (z, p, k)

    def test_fewer_zeros_2(self):
        if False:
            while True:
                i = 10
        sos = signal.butter(12, [5.0, 30.0], 'bandpass', fs=1200.0, analog=False, output='sos')
        with pytest.warns(signal.BadCoefficients, match='Badly conditioned'):
            (z, p, k) = signal.sos2zpk(sos)
        assert len(z) == 24
        assert len(p) == 24

@testing.with_requires('scipy')
class TestSos2Tf:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, xp, scp):
        if False:
            i = 10
            return i + 15
        sos = xp.array([[1, 1, 1, 1, 0, -1], [-2, 3, 1, 1, 10, 1]])
        (b, a) = scp.signal.sos2tf(sos)
        return (b, a)

@testing.with_requires('scipy')
class TestCplxReal:

    def test_trivial_input(self):
        if False:
            print('Hello World!')
        assert all((x.size == 0 for x in _cplxreal([])))
        cplx1 = _cplxreal(1)
        assert cplx1[0].size == 0
        testing.assert_allclose(cplx1[1], cupy.array([1]))

    def test_output_order(self):
        if False:
            while True:
                i = 10
        eps = cupy.finfo(float).eps
        a = [0 + 1j, 0 - 1j, eps + 1j, eps - 1j, -eps + 1j, -eps - 1j, 1, 4, 2, 3, 0, 0, 2 + 3j, 2 - 3j, 1 - eps + 1j, 1 + 2j, 1 - 2j, 1 + eps - 1j, 3 + 1j, 3 + 1j, 3 + 1j, 3 - 1j, 3 - 1j, 3 - 1j, 2 - 3j, 2 + 3j]
        a = cupy.array(a)
        (zc, zr) = _cplxreal(a)
        testing.assert_allclose(zc, [1j, 1j, 1j, 1 + 1j, 1 + 2j, 2 + 3j, 2 + 3j, 3 + 1j, 3 + 1j, 3 + 1j])
        testing.assert_allclose(zr, [0, 0, 1, 2, 3, 4])
        z = cupy.array([1 - eps + 1j, 1 + 2j, 1 - 2j, 1 + eps - 1j, 1 + eps + 3j, 1 - 2 * eps - 3j, 0 + 1j, 0 - 1j, 2 + 4j, 2 - 4j, 2 + 3j, 2 - 3j, 3 + 7j, 3 - 7j, 4 - eps + 1j, 4 + eps - 2j, 4 - 1j, 4 - eps + 2j])
        (zc, zr) = _cplxreal(z)
        testing.assert_allclose(zc, [1j, 1 + 1j, 1 + 2j, 1 + 3j, 2 + 3j, 2 + 4j, 3 + 7j, 4 + 1j, 4 + 2j])
        assert zr.size == 0

    def test_unmatched_conjugates(self):
        if False:
            for i in range(10):
                print('nop')
        assert_raises(ValueError, _cplxreal, [1 + 3j, 1 - 3j, 1 + 2j])
        assert_raises(ValueError, _cplxreal, [1 + 3j, 1 - 3j, 1 + 2j, 1 - 3j])
        assert_raises(ValueError, _cplxreal, [1 + 3j, 1 - 3j, 1 + 3j])
        assert_raises(ValueError, _cplxreal, [1 + 3j])
        assert_raises(ValueError, _cplxreal, [1 - 3j])

    def test_real_integer_input(self):
        if False:
            while True:
                i = 10
        (zc, zr) = _cplxreal([2, 0, 1, 4])
        assert zc.size == 0
        testing.assert_allclose(zr, [0, 1, 2, 4], atol=1e-15)

@testing.with_requires('scipy')
class TestLowLevelAP:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_buttap(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        return scp.signal.buttap(3)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_cheb1ap(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        return scp.signal.cheb1ap(3, 1)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_cheb2ap(self, xp, scp):
        if False:
            print('Hello World!')
        return scp.signal.cheb2ap(3, 1)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=0.0002, rtol=0.0002)
    def test_ellipap(self, xp, scp):
        if False:
            print('Hello World!')
        return scp.signal.ellipap(7, 1, 10)