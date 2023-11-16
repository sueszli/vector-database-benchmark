import pytest
from pytest import raises as assert_raises
import cupy
from cupy import testing
from cupyx.scipy import signal
from cupyx.scipy.signal import iirdesign
try:
    import scipy.signal
except ImportError:
    pass
nimpl = pytest.mark.xfail(reason='not implemented')
prec_loss = pytest.mark.xfail(reason='zpk2tf loses precision')

@testing.with_requires('scipy>=1.8')
class TestIIRFilter:

    @pytest.mark.parametrize('N', list(range(1, 25)))
    @pytest.mark.parametrize('ftype', ['butter', pytest.param('bessel', marks=nimpl), 'cheby1', 'cheby2', 'ellip'])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-05, rtol=1e-06)
    def test_symmetry(self, N, ftype, xp, scp):
        if False:
            while True:
                i = 10
        (z, p, k) = scp.signal.iirfilter(N, 1.1, 1, 20, 'low', analog=True, ftype=ftype, output='zpk')
        return (z, p, k)

    @pytest.mark.parametrize('N', list(range(1, 25)))
    @pytest.mark.parametrize('ftype', ['butter', pytest.param('bessel', marks=nimpl), 'cheby1', 'cheby2', 'ellip'])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-05, rtol=1e-05)
    def test_symmetry_2(self, N, ftype, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        (b, a) = scp.signal.iirfilter(N, 1.1, 1, 20, 'low', analog=True, ftype=ftype, output='ba')
        return (b, a)

    @pytest.mark.xfail(reason='bessel IIR filter not implemented')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_int_inputs(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        (z, p, k) = scp.signal.iirfilter(24, 100, btype='low', analog=True, ftype='bessel', output='zpk')
        return (z, p, k)

    def test_invalid_wn_size(self):
        if False:
            print('Hello World!')
        assert_raises(ValueError, signal.iirfilter, 1, [0.1, 0.9], btype='low')
        assert_raises(ValueError, signal.iirfilter, 1, [0.2, 0.5], btype='high')
        assert_raises(ValueError, signal.iirfilter, 1, 0.2, btype='bp')
        assert_raises(ValueError, signal.iirfilter, 1, 400, btype='bs', analog=True)

    def test_invalid_wn_range(self):
        if False:
            i = 10
            return i + 15
        assert_raises(ValueError, signal.iirfilter, 1, 2, btype='low')
        assert_raises(ValueError, signal.iirfilter, 1, [0.5, 1], btype='band')
        assert_raises(ValueError, signal.iirfilter, 1, [0.0, 0.5], btype='band')
        assert_raises(ValueError, signal.iirfilter, 1, -1, btype='high')
        assert_raises(ValueError, signal.iirfilter, 1, [1, 2], btype='band')
        assert_raises(ValueError, signal.iirfilter, 1, [10, 20], btype='stop')

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_analog_sos(self, xp, scp):
        if False:
            i = 10
            return i + 15
        sos2 = scp.signal.iirfilter(N=1, Wn=1, btype='low', analog=True, output='sos')
        return sos2

    def test_wn1_ge_wn0(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValueError, match='Wn\\[0\\] must be less than Wn\\[1\\]'):
            signal.iirfilter(2, [0.5, 0.5])
        with pytest.raises(ValueError, match='Wn\\[0\\] must be less than Wn\\[1\\]'):
            signal.iirfilter(2, [0.6, 0.5])

@testing.with_requires('scipy')
class TestButter:

    @pytest.mark.parametrize('arg', [(0, 1), (1, 1)])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_degenerate(self, xp, scp, arg):
        if False:
            for i in range(10):
                print('nop')
        (b, a) = scp.signal.butter(*arg, analog=True)
        return (b, a)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_degenerate_1(self, xp, scp):
        if False:
            i = 10
            return i + 15
        (z, p, k) = scp.signal.butter(1, 0.3, output='zpk')
        return (z, p, k)

    @pytest.mark.parametrize('N', list(range(25)))
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, xp, scp, N):
        if False:
            return 10
        wn = 0.01
        (z, p, k) = scp.signal.butter(N, wn, 'low', analog=True, output='zpk')
        return (z, p, k)

    @pytest.mark.parametrize('N', list(range(25)))
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic_1(self, xp, scp, N):
        if False:
            i = 10
            return i + 15
        wn = 0.01
        (z, p, k) = scp.signal.butter(N, wn, 'high', analog=False, output='zpk')
        return (z, p, k)

    @pytest.mark.parametrize('arg, analog', [((2, 1), True), ((5, 1), True), ((10, 1), True), ((19, 1.0441379169150726), True), ((5, 0.4), False)])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic_2(self, xp, scp, arg, analog):
        if False:
            while True:
                i = 10
        (b, a) = scp.signal.butter(*arg, analog=analog)
        return (b, a)

    @pytest.mark.parametrize('arg', [(28, 0.43), (27, 0.56)])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_highpass(self, xp, scp, arg):
        if False:
            for i in range(10):
                print('nop')
        (z, p, k) = scp.signal.butter(*arg, 'high', output='zpk')
        return (z, p, k)

    @pytest.mark.parametrize('format', ['zpk', 'ba'])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-12)
    def test_bandpass(self, xp, scp, format):
        if False:
            return 10
        output = scp.signal.butter(8, [0.25, 0.33], 'band', output=format)
        return output

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-12)
    def test_bandpass_analog(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        output = scp.signal.butter(4, [90.5, 110.5], 'bp', analog=True, output='zpk')
        return output

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bandstop(self, xp, scp):
        if False:
            return 10
        (z, p, k) = scp.signal.butter(7, [0.45, 0.56], 'stop', output='zpk')
        z.sort()
        p.sort()
        return (z, p, k)

    @pytest.mark.parametrize('outp', ['zpk', 'sos', pytest.param('ba', marks=prec_loss)])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    @testing.with_requires('scipy>=1.8')
    def test_ba_output(self, xp, scp, outp):
        if False:
            for i in range(10):
                print('nop')
        outp = scp.signal.butter(4, [100, 300], 'bandpass', analog=True, output=outp)
        return outp

    def test_fs_param(self):
        if False:
            for i in range(10):
                print('nop')
        for fs in (900, 900.1, 1234.567):
            for N in (0, 1, 2, 3, 10):
                for fc in (100, 100.1, 432.12345):
                    for btype in ('lp', 'hp'):
                        ba1 = signal.butter(N, fc, btype, fs=fs)
                        ba2 = signal.butter(N, fc / (fs / 2), btype)
                        testing.assert_allclose(ba1[0], ba2[0])
                        testing.assert_allclose(ba1[1], ba2[1])
                for fc in ((100, 200), (100.1, 200.2), (321.123, 432.123)):
                    for btype in ('bp', 'bs'):
                        ba1 = signal.butter(N, fc, btype, fs=fs)
                        fcnorm = cupy.array([f / (fs / 2) for f in fc])
                        ba2 = signal.butter(N, fcnorm, btype)
                        testing.assert_allclose(ba1[0], ba2[0])
                        testing.assert_allclose(ba1[0], ba2[0])

@testing.with_requires('scipy')
class TestCheby1:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_degenerate(self, xp, scp):
        if False:
            return 10
        (b, a) = scp.signal.cheby1(0, 10 * xp.log10(2), 1, analog=True)
        return (b, a)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_degenerate_1(self, xp, scp):
        if False:
            while True:
                i = 10
        (b, a) = scp.signal.cheby1(1, 10 * xp.log10(2), 1, analog=True)
        return (b, a)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_degenerate_2(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        (z, p, k) = scp.signal.cheby1(1, 0.1, 0.3, output='zpk')
        return (z, p, k)

    @pytest.mark.parametrize('N', list(range(25)))
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, xp, scp, N):
        if False:
            i = 10
            return i + 15
        wn = 0.01
        (z, p, k) = scp.signal.cheby1(N, 1, wn, 'low', analog=True, output='zpk')
        return (z, p, k)

    @pytest.mark.parametrize('N', list(range(25)))
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic_1(self, xp, scp, N):
        if False:
            print('Hello World!')
        wn = 0.01
        (z, p, k) = scp.signal.cheby1(N, 1, wn, 'high', analog=False, output='zpk')
        return (z, p, k)

    @pytest.mark.parametrize('arg, kwd', [((8, 0.5, 0.048), {}), ((4, 1, [0.4, 0.7]), {'btype': 'band'}), ((5, 3, 1), {'analog': True}), ((8, 0.5, 0.1), {}), ((8, 0.5, 0.25), {})])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_basic_2(self, xp, scp, arg, kwd):
        if False:
            for i in range(10):
                print('nop')
        (b, a) = scp.signal.cheby1(*arg, **kwd)
        return (b, a)

    @pytest.mark.parametrize('arg, kwd', [((24, 0.7, 0.2), {'output': 'zpk'}), ((23, 0.8, 0.3), {'output': 'zpk'}), ((10, 1, 1000), {'analog': True, 'output': 'zpk'})])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_highpass(self, xp, scp, arg, kwd):
        if False:
            while True:
                i = 10
        (z, p, k) = scp.signal.cheby1(*arg, 'high', **kwd)
        return (z, p, k)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bandpass(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        (z, p, k) = scp.signal.cheby1(8, 1, [0.3, 0.4], 'bp', output='zpk')
        return (z, p, k)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bandstop(self, xp, scp):
        if False:
            print('Hello World!')
        (z, p, k) = scp.signal.cheby1(7, 1, [0.5, 0.6], 'stop', output='zpk')
        z = z[xp.argsort(z.imag)]
        p = p[xp.argsort(p.imag)]
        return (z, p, k)

    @pytest.mark.xfail(reason='zpk2tf loses precision (cf TestButter)')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-10)
    def test_ba_output(self, xp, scp):
        if False:
            return 10
        (b, a) = scp.signal.cheby1(5, 0.9, [210, 310], 'stop', analog=True)
        return (b, a)

@testing.with_requires('scipy')
class TestCheby2:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_degenerate(self, xp, scp):
        if False:
            while True:
                i = 10
        (b, a) = scp.signal.cheby2(0, 123.456, 1, analog=True)
        return (b, a)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_degenerate_1(self, xp, scp):
        if False:
            return 10
        (b, a) = scp.signal.cheby2(1, 10 * xp.log10(2), 1, analog=True)
        return (b, a)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_degenerate_2(self, xp, scp):
        if False:
            i = 10
            return i + 15
        (z, p, k) = scp.signal.cheby2(1, 50, 0.3, output='zpk')
        return (z, p, k)

    @pytest.mark.parametrize('N', list(range(25)))
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, xp, scp, N):
        if False:
            return 10
        wn = 0.01
        (z, p, k) = scp.signal.cheby2(N, 40, wn, 'low', analog=True, output='zpk')
        return (z, p, k)

    @pytest.mark.parametrize('N', list(range(25)))
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic_1(self, xp, scp, N):
        if False:
            print('Hello World!')
        wn = 0.01
        (z, p, k) = scp.signal.cheby2(N, 40, wn, 'high', analog=False, output='zpk')
        return (z, p, k)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic_2(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        (B, A) = scp.signal.cheby2(18, 100, 0.5)
        return (B, A)

    @pytest.mark.parametrize('arg, kwd', [((26, 60, 0.3), {'output': 'zpk'}), ((25, 80, 0.5), {'output': 'zpk'})])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_highpass(self, xp, scp, arg, kwd):
        if False:
            print('Hello World!')
        (z, p, k) = scp.signal.cheby2(*arg, 'high', **kwd)
        return (z, p, k)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bandpass(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        (z, p, k) = scp.signal.cheby2(9, 40, [0.07, 0.2], 'pass', output='zpk')
        return (z, p, k)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_bandstop(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        (z, p, k) = scp.signal.cheby2(6, 55, [0.1, 0.9], 'stop', output='zpk')
        z = z[xp.argsort(xp.angle(z))]
        p = p[xp.argsort(xp.angle(p))]
        return (z, p, k)

    @pytest.mark.xfail(reason='zpk2tf loses precision')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_ba_output(self, xp, scp):
        if False:
            i = 10
            return i + 15
        (b, a) = scp.signal.cheby2(5, 20, [2010, 2100], 'stop', True)
        return (b, a)

@testing.with_requires('scipy>=1.8')
class TestEllip:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_degenerate(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        (b, a) = scp.signal.ellip(0, 10 * xp.log10(2), 123.456, 1, analog=True)
        return (b, a)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_degenerate_1(self, xp, scp):
        if False:
            print('Hello World!')
        (b, a) = scp.signal.ellip(1, 10 * xp.log10(2), 1, 1, analog=True)
        return (b, a)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_degenerate_2(self, xp, scp):
        if False:
            while True:
                i = 10
        (z, p, k) = scp.signal.ellip(1, 1, 55, 0.3, output='zpk')
        return (z, p, k)

    @pytest.mark.parametrize('N', list(range(25)))
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, xp, scp, N):
        if False:
            return 10
        wn = 0.01
        (z, p, k) = scp.signal.ellip(N, 1, 40, wn, 'low', analog=True, output='zpk')
        return (z, p, k)

    @pytest.mark.parametrize('N', list(range(20)))
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic_1(self, xp, scp, N):
        if False:
            i = 10
            return i + 15
        wn = 0.01
        (z, p, k) = scp.signal.ellip(N, 1, 40, wn, 'high', analog=False, output='zpk')
        return (z, p, k)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14, rtol=1e-14)
    def test_basic_2(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        (b3, a3) = scp.signal.ellip(5, 3, 26, 1, analog=True)
        return (b3, a3)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic_3(self, xp, scp):
        if False:
            return 10
        (b, a) = scp.signal.ellip(3, 1, 60, [0.4, 0.7], 'stop')
        return (b, a)

    @pytest.mark.parametrize('arg', [(24, 1, 80, 0.3, 'high'), (23, 1, 70, 0.5, 'high')])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_highpass(self, xp, scp, arg):
        if False:
            for i in range(10):
                print('nop')
        (z, p, k) = scp.signal.ellip(*arg, output='zpk')
        return (z, p, k)

    @pytest.mark.parametrize('arg', [(7, 1, 40, [0.07, 0.2], 'pass'), (5, 1, 75, [90.5, 110.5], 'pass', True)])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bandpass(self, xp, scp, arg):
        if False:
            for i in range(10):
                print('nop')
        (z, p, k) = scp.signal.ellip(7, 1, 40, [0.07, 0.2], 'pass', output='zpk')
        return (z, p, k)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bandstop(self, xp, scp):
        if False:
            while True:
                i = 10
        (z, p, k) = scp.signal.ellip(8, 1, 65, [0.2, 0.4], 'stop', output='zpk')
        z = z[xp.argsort(xp.angle(z))]
        p = p[xp.argsort(xp.angle(p))]
        return (z, p, k)

    @pytest.mark.xfail(reason='zpk2tf loses precision')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_ba_output(self, xp, scp):
        if False:
            return 10
        (b, a) = scp.signal.ellip(5, 1, 40, [201, 240], 'stop', True)
        return (b, a)

@testing.with_requires('scipy')
class TestButtord:

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-08)
    def test_lowpass(self, xp, scp):
        if False:
            while True:
                i = 10
        wp = 0.2
        ws = 0.3
        rp = 3
        rs = 60
        (N, Wn) = scp.signal.buttord(wp, ws, rp, rs, False)
        (b, a) = scp.signal.butter(N, Wn, 'lowpass', False)
        (w, h) = scp.signal.freqz(b, a)
        return (w, h)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-10)
    def test_highpass(self, xp, scp):
        if False:
            print('Hello World!')
        wp = 0.3
        ws = 0.2
        rp = 3
        rs = 70
        (N, Wn) = scp.signal.buttord(wp, ws, rp, rs, False)
        (b, a) = scp.signal.butter(N, Wn, 'highpass', False)
        (w, h) = scp.signal.freqz(b, a)
        return (w, h)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bandpass(self, xp, scp):
        if False:
            print('Hello World!')
        wp = [0.2, 0.5]
        ws = [0.1, 0.6]
        rp = 3
        rs = 80
        (N, Wn) = scp.signal.buttord(wp, ws, rp, rs, False)
        return (N, Wn)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bandstop(self, xp, scp):
        if False:
            while True:
                i = 10
        wp = [0.1, 0.6]
        ws = [0.2, 0.5]
        rp = 3
        rs = 90
        (N, Wn) = scp.signal.buttord(wp, ws, rp, rs, False)
        return (N, Wn)

    @pytest.mark.xfail(reason='TODO: freqs')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_analog(self, xp, scp):
        if False:
            while True:
                i = 10
        wp = 200
        ws = 600
        rp = 3
        rs = 60
        (N, Wn) = scp.signal.buttord(wp, ws, rp, rs, True)
        (b, a) = scp.signal.butter(N, Wn, 'lowpass', True)
        (w, h) = scp.signal.freqs(b, a)
        return (w, h)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_fs_param(self, xp, scp):
        if False:
            return 10
        wp = [4410, 11025]
        ws = [2205, 13230]
        rp = 3
        rs = 80
        fs = 44100
        (N, Wn) = scp.signal.buttord(wp, ws, rp, rs, False, fs=fs)
        return (N, Wn)

    def test_invalid_input(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValueError) as exc_info:
            signal.buttord([20, 50], [14, 60], 3, 2)
        assert 'gpass should be smaller than gstop' in str(exc_info.value)
        with pytest.raises(ValueError) as exc_info:
            signal.buttord([20, 50], [14, 60], -1, 2)
        assert 'gpass should be larger than 0.0' in str(exc_info.value)
        with pytest.raises(ValueError) as exc_info:
            signal.buttord([20, 50], [14, 60], 1, -2)
        assert 'gstop should be larger than 0.0' in str(exc_info.value)

    def test_runtime_warnings(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.warns(RuntimeWarning, match='Order is zero'):
            signal.buttord(0.0, 1.0, 3, 60)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_ellip_butter(self, xp, scp):
        if False:
            return 10
        (n, wn) = scp.signal.buttord([0.1, 0.6], [0.2, 0.5], 3, 60)
        return (n, wn)

@testing.with_requires('scipy')
class TestCheb1ord:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_lowpass(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        wp = 0.2
        ws = 0.3
        rp = 3
        rs = 60
        (N, Wn) = scp.signal.cheb1ord(wp, ws, rp, rs, False)
        return (N, Wn)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_highpass(self, xp, scp):
        if False:
            print('Hello World!')
        wp = 0.3
        ws = 0.2
        rp = 3
        rs = 70
        (N, Wn) = scp.signal.cheb1ord(wp, ws, rp, rs, False)
        return (N, Wn)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bandpass(self, xp, scp):
        if False:
            i = 10
            return i + 15
        wp = [0.2, 0.5]
        ws = [0.1, 0.6]
        rp = 3
        rs = 80
        (N, Wn) = scp.signal.cheb1ord(wp, ws, rp, rs, False)
        return (N, Wn)

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=3e-07)
    def test_bandstop(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        wp = [0.1, 0.6]
        ws = [0.2, 0.5]
        rp = 3
        rs = 90
        (N, Wn) = scp.signal.cheb1ord(wp, ws, rp, rs, False)
        return (N, Wn)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_analog(self, xp, scp):
        if False:
            i = 10
            return i + 15
        wp = 700
        ws = 100
        rp = 3
        rs = 70
        (N, Wn) = scp.signal.cheb1ord(wp, ws, rp, rs, True)
        return (N, Wn)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_fs_param(self, xp, scp):
        if False:
            while True:
                i = 10
        wp = 4800
        ws = 7200
        rp = 3
        rs = 60
        fs = 48000
        (N, Wn) = scp.signal.cheb1ord(wp, ws, rp, rs, False, fs=fs)
        return (N, Wn)

    def test_invalid_input(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValueError) as exc_info:
            signal.cheb1ord(0.2, 0.3, 3, 2)
        assert 'gpass should be smaller than gstop' in str(exc_info.value)
        with pytest.raises(ValueError) as exc_info:
            signal.cheb1ord(0.2, 0.3, -1, 2)
        assert 'gpass should be larger than 0.0' in str(exc_info.value)
        with pytest.raises(ValueError) as exc_info:
            signal.cheb1ord(0.2, 0.3, 1, -2)
        assert 'gstop should be larger than 0.0' in str(exc_info.value)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_ellip_butter(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        (n, wn) = scp.signal.cheb1ord([0.1, 0.6], [0.2, 0.5], 3, 60)
        return (n, wn)

@testing.with_requires('scipy')
class TestCheb2ord:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_lowpass(self, xp, scp):
        if False:
            while True:
                i = 10
        wp = 0.2
        ws = 0.3
        rp = 3
        rs = 60
        (N, Wn) = scp.signal.cheb2ord(wp, ws, rp, rs, False)
        return (N, Wn)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_highpass(self, xp, scp):
        if False:
            print('Hello World!')
        wp = 0.3
        ws = 0.2
        rp = 3
        rs = 70
        (N, Wn) = scp.signal.cheb2ord(wp, ws, rp, rs, False)
        return (N, Wn)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bandpass(self, xp, scp):
        if False:
            print('Hello World!')
        wp = [0.2, 0.5]
        ws = [0.1, 0.6]
        rp = 3
        rs = 80
        (N, Wn) = scp.signal.cheb2ord(wp, ws, rp, rs, False)
        return (N, Wn)

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=3e-07)
    def test_bandstop(self, xp, scp):
        if False:
            print('Hello World!')
        wp = [0.1, 0.6]
        ws = [0.2, 0.5]
        rp = 3
        rs = 90
        (N, Wn) = scp.signal.cheb2ord(wp, ws, rp, rs, False)
        return (N, Wn)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_analog(self, xp, scp):
        if False:
            return 10
        wp = [20, 50]
        ws = [10, 60]
        rp = 3
        rs = 80
        (N, Wn) = scp.signal.cheb2ord(wp, ws, rp, rs, True)
        return (N, Wn)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_fs_param(self, xp, scp):
        if False:
            while True:
                i = 10
        wp = 150
        ws = 100
        rp = 3
        rs = 70
        fs = 1000
        (N, Wn) = scp.signal.cheb2ord(wp, ws, rp, rs, False, fs=fs)
        return (N, Wn)

    def test_invalid_input(self):
        if False:
            print('Hello World!')
        with pytest.raises(ValueError) as exc_info:
            signal.cheb2ord([0.1, 0.6], [0.2, 0.5], 3, 2)
        assert 'gpass should be smaller than gstop' in str(exc_info.value)
        with pytest.raises(ValueError) as exc_info:
            signal.cheb2ord([0.1, 0.6], [0.2, 0.5], -1, 2)
        assert 'gpass should be larger than 0.0' in str(exc_info.value)
        with pytest.raises(ValueError) as exc_info:
            signal.cheb2ord([0.1, 0.6], [0.2, 0.5], 1, -2)
        assert 'gstop should be larger than 0.0' in str(exc_info.value)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_ellip_butter(self, xp, scp):
        if False:
            i = 10
            return i + 15
        (n, wn) = scp.signal.cheb2ord([0.1, 0.6], [0.2, 0.5], 3, 60)
        return (n, wn)

@testing.with_requires('scipy')
class TestEllipord:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_lowpass(self, xp, scp):
        if False:
            return 10
        wp = 0.2
        ws = 0.3
        rp = 3
        rs = 60
        (N, Wn) = scp.signal.ellipord(wp, ws, rp, rs, False)
        return (N, Wn)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    @testing.with_requires('scipy>=1.7')
    def test_lowpass_1000dB(self, xp, scp):
        if False:
            print('Hello World!')
        wp = 0.2
        ws = 0.3
        rp = 3
        rs = 1000
        (N, Wn) = scp.signal.ellipord(wp, ws, rp, rs, False)
        return (N, Wn)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_highpass(self, xp, scp):
        if False:
            print('Hello World!')
        wp = 0.3
        ws = 0.2
        rp = 3
        rs = 70
        (N, Wn) = scp.signal.ellipord(wp, ws, rp, rs, False)
        return (N, Wn)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bandpass(self, xp, scp):
        if False:
            i = 10
            return i + 15
        wp = xp.array([0.2, 0.5])
        ws = xp.array([0.1, 0.6])
        rp = 3
        rs = 80
        (N, Wn) = scp.signal.ellipord(wp, ws, rp, rs, False)
        return (N, Wn)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bandstop(self, xp, scp):
        if False:
            i = 10
            return i + 15
        wp = xp.array([0.1, 0.6])
        ws = xp.array([0.2, 0.5])
        rp = 3
        rs = 90
        (N, Wn) = scp.signal.ellipord(wp, ws, rp, rs, False)
        return (N, Wn)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_analog(self, xp, scp):
        if False:
            while True:
                i = 10
        wp = [1000, 6000]
        ws = [2000, 5000]
        rp = 3
        rs = 90
        (N, Wn) = scp.signal.ellipord(wp, ws, rp, rs, True)
        return (N, Wn)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_fs_param(self, xp, scp):
        if False:
            i = 10
            return i + 15
        wp = [400, 2400]
        ws = [800, 2000]
        rp = 3
        rs = 90
        fs = 8000
        (N, Wn) = scp.signal.ellipord(wp, ws, rp, rs, False, fs=fs)
        return (N, Wn)

    def test_invalid_input(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValueError) as exc_info:
            signal.ellipord(0.2, 0.5, 3, 2)
        assert 'gpass should be smaller than gstop' in str(exc_info.value)
        with pytest.raises(ValueError) as exc_info:
            signal.ellipord(0.2, 0.5, -1, 2)
        assert 'gpass should be larger than 0.0' in str(exc_info.value)
        with pytest.raises(ValueError) as exc_info:
            signal.ellipord(0.2, 0.5, 1, -2)
        assert 'gstop should be larger than 0.0' in str(exc_info.value)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_ellip_butter(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        (n, wn) = scp.signal.ellipord([0.1, 0.6], [0.2, 0.5], 3, 60)
        return (n, wn)

@testing.with_requires('scipy')
class TestIIRDesign:

    def test_exceptions(self):
        if False:
            return 10
        with pytest.raises(ValueError, match='the same shape'):
            iirdesign(0.2, [0.1, 0.3], 1, 40)
        with pytest.raises(ValueError, match='the same shape'):
            iirdesign(cupy.array([[0.3, 0.6], [0.3, 0.6]]), cupy.array([[0.4, 0.5], [0.4, 0.5]]), 1, 40)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign(0, 0.5, 1, 40)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign(-0.1, 0.5, 1, 40)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign(0.1, 0, 1, 40)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign(0.1, -0.5, 1, 40)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign([0, 0.3], [0.1, 0.5], 1, 40)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign([-0.1, 0.3], [0.1, 0.5], 1, 40)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign([0.1, 0], [0.1, 0.5], 1, 40)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign([0.1, -0.3], [0.1, 0.5], 1, 40)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign([0.1, 0.3], [0, 0.5], 1, 40)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign([0.1, 0.3], [-0.1, 0.5], 1, 40)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign([0.1, 0.3], [0.1, 0], 1, 40)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign([0.1, 0.3], [0.1, -0.5], 1, 40)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign(-0.1, 0.5, 1, 40, analog=True)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign(0.1, -0.5, 1, 40, analog=True)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign([-0.1, 0.3], [0.1, 0.5], 1, 40, analog=True)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign([0.1, -0.3], [0.1, 0.5], 1, 40, analog=True)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign([0.1, 0.3], [-0.1, 0.5], 1, 40, analog=True)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign([0.1, 0.3], [0.1, -0.5], 1, 40, analog=True)
        with pytest.raises(ValueError, match='must be less than 1'):
            iirdesign(1, 0.5, 1, 40)
        with pytest.raises(ValueError, match='must be less than 1'):
            iirdesign(1.1, 0.5, 1, 40)
        with pytest.raises(ValueError, match='must be less than 1'):
            iirdesign(0.1, 1, 1, 40)
        with pytest.raises(ValueError, match='must be less than 1'):
            iirdesign(0.1, 1.5, 1, 40)
        with pytest.raises(ValueError, match='must be less than 1'):
            iirdesign([1, 0.3], [0.1, 0.5], 1, 40)
        with pytest.raises(ValueError, match='must be less than 1'):
            iirdesign([1.1, 0.3], [0.1, 0.5], 1, 40)
        with pytest.raises(ValueError, match='must be less than 1'):
            iirdesign([0.1, 1], [0.1, 0.5], 1, 40)
        with pytest.raises(ValueError, match='must be less than 1'):
            iirdesign([0.1, 1.1], [0.1, 0.5], 1, 40)
        with pytest.raises(ValueError, match='must be less than 1'):
            iirdesign([0.1, 0.3], [1, 0.5], 1, 40)
        with pytest.raises(ValueError, match='must be less than 1'):
            iirdesign([0.1, 0.3], [1.1, 0.5], 1, 40)
        with pytest.raises(ValueError, match='must be less than 1'):
            iirdesign([0.1, 0.3], [0.1, 1], 1, 40)
        with pytest.raises(ValueError, match='must be less than 1'):
            iirdesign([0.1, 0.3], [0.1, 1.5], 1, 40)
        iirdesign(100, 500, 1, 40, fs=2000)
        iirdesign(500, 100, 1, 40, fs=2000)
        iirdesign([200, 400], [100, 500], 1, 40, fs=2000)
        iirdesign([100, 500], [200, 400], 1, 40, fs=2000)
        with pytest.raises(ValueError, match='must be less than fs/2'):
            iirdesign(1000, 400, 1, 40, fs=2000)
        with pytest.raises(ValueError, match='must be less than fs/2'):
            iirdesign(1100, 500, 1, 40, fs=2000)
        with pytest.raises(ValueError, match='must be less than fs/2'):
            iirdesign(100, 1000, 1, 40, fs=2000)
        with pytest.raises(ValueError, match='must be less than fs/2'):
            iirdesign(100, 1100, 1, 40, fs=2000)
        with pytest.raises(ValueError, match='must be less than fs/2'):
            iirdesign([1000, 400], [100, 500], 1, 40, fs=2000)
        with pytest.raises(ValueError, match='must be less than fs/2'):
            iirdesign([1100, 400], [100, 500], 1, 40, fs=2000)
        with pytest.raises(ValueError, match='must be less than fs/2'):
            iirdesign([200, 1000], [100, 500], 1, 40, fs=2000)
        with pytest.raises(ValueError, match='must be less than fs/2'):
            iirdesign([200, 1100], [100, 500], 1, 40, fs=2000)
        with pytest.raises(ValueError, match='must be less than fs/2'):
            iirdesign([200, 400], [1000, 500], 1, 40, fs=2000)
        with pytest.raises(ValueError, match='must be less than fs/2'):
            iirdesign([200, 400], [1100, 500], 1, 40, fs=2000)
        with pytest.raises(ValueError, match='must be less than fs/2'):
            iirdesign([200, 400], [100, 1000], 1, 40, fs=2000)
        with pytest.raises(ValueError, match='must be less than fs/2'):
            iirdesign([200, 400], [100, 1100], 1, 40, fs=2000)
        with pytest.raises(ValueError, match='strictly inside stopband'):
            iirdesign([0.1, 0.4], [0.5, 0.6], 1, 40)
        with pytest.raises(ValueError, match='strictly inside stopband'):
            iirdesign([0.5, 0.6], [0.1, 0.4], 1, 40)
        with pytest.raises(ValueError, match='strictly inside stopband'):
            iirdesign([0.3, 0.6], [0.4, 0.7], 1, 40)
        with pytest.raises(ValueError, match='strictly inside stopband'):
            iirdesign([0.4, 0.7], [0.3, 0.6], 1, 40)

@testing.with_requires('scipy')
class TestIIRNotch:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_ba_output(self, xp, scp):
        if False:
            i = 10
            return i + 15
        (b, a) = scp.signal.iirnotch(0.06, 30)
        return (b, a)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_frequency_response(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        (b, a) = scp.signal.iirnotch(0.3, 30)
        return (b, a)

    def test_errors(self):
        if False:
            for i in range(10):
                print('nop')
        assert_raises(ValueError, signal.iirnotch, w0=2, Q=30)
        assert_raises(ValueError, signal.iirnotch, w0=-1, Q=30)
        assert_raises(ValueError, signal.iirnotch, w0='blabla', Q=30)
        assert_raises(TypeError, signal.iirnotch, w0=-1, Q=[1, 2, 3])

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_fs_param(self, xp, scp):
        if False:
            while True:
                i = 10
        (b, a) = scp.signal.iirnotch(1500, 30, fs=10000)
        return (b, a)

@testing.with_requires('scipy')
class TestIIRPeak:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_ba_output(self, xp, scp):
        if False:
            while True:
                i = 10
        (b, a) = scp.signal.iirpeak(0.06, 30)
        return (b, a)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_frequency_response(self, xp, scp):
        if False:
            i = 10
            return i + 15
        (b, a) = scp.signal.iirpeak(0.3, 30)
        return (b, a)

    def test_errors(self):
        if False:
            i = 10
            return i + 15
        assert_raises(ValueError, signal.iirpeak, w0=2, Q=30)
        assert_raises(ValueError, signal.iirpeak, w0=-1, Q=30)
        assert_raises(ValueError, signal.iirpeak, w0='blabla', Q=30)
        assert_raises(TypeError, signal.iirpeak, w0=-1, Q=[1, 2, 3])

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_fs_param(self, xp, scp):
        if False:
            print('Hello World!')
        (b, a) = scp.signal.iirpeak(1200, 30, fs=8000)
        return (b, a)

@testing.with_requires('scipy')
class TestIIRComb:

    def test_invalid_input(self):
        if False:
            return 10
        fs = 1000
        for args in [(-fs, 30), (0, 35), (fs / 2, 40), (fs, 35)]:
            with pytest.raises(ValueError, match='w0 must be between '):
                signal.iircomb(*args, fs=fs)
        for args in [(120, 30), (157, 35)]:
            with pytest.raises(ValueError, match='fs must be divisible '):
                signal.iircomb(*args, fs=fs)
        with pytest.raises(ValueError, match='fs must be divisible '):
            signal.iircomb(w0=49.999 / int(44100 / 2), Q=30)
        with pytest.raises(ValueError, match='fs must be divisible '):
            signal.iircomb(w0=49.999, Q=30, fs=44100)
        for args in [(0.2, 30, 'natch'), (0.5, 35, 'comb')]:
            with pytest.raises(ValueError, match='ftype must be '):
                signal.iircomb(*args)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize('ftype', ('notch', 'peak'))
    def test_frequency_response(self, ftype, xp, scp):
        if False:
            return 10
        (b, a) = scp.signal.iircomb(1000, 30, ftype=ftype, fs=10000)
        return (b, a)

    @testing.with_requires('scipy>=1.9.0')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize('ftype,pass_zero,peak,notch', [('peak', True, 123.45, 61.725), ('peak', False, 61.725, 123.45), ('peak', None, 61.725, 123.45), ('notch', None, 61.725, 123.45), ('notch', True, 123.45, 61.725), ('notch', False, 61.725, 123.45)])
    def test_pass_zero(self, ftype, pass_zero, peak, notch, xp, scp):
        if False:
            return 10
        (b, a) = scp.signal.iircomb(123.45, 30, ftype=ftype, fs=1234.5, pass_zero=pass_zero)
        return (b, a)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_iir_symmetry(self, xp, scp):
        if False:
            i = 10
            return i + 15
        (b, a) = scp.signal.iircomb(400, 30, fs=24000)
        return (b, a)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_ba_output(self, xp, scp):
        if False:
            print('Hello World!')
        (b, a) = scp.signal.iircomb(60, 35, ftype='notch', fs=600)
        return (b, a)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_ba_output_2(self, xp, scp):
        if False:
            i = 10
            return i + 15
        (b, a) = scp.signal.iircomb(60, 35, ftype='peak', fs=600)
        return (b, a)

    @testing.with_requires('scipy>=1.9.0')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_nearest_divisor(self, xp, scp):
        if False:
            i = 10
            return i + 15
        (b, a) = scp.signal.iircomb(50 / int(44100 / 2), 50.0, ftype='notch')
        return (b, a)

@testing.with_requires('scipy')
class TestZpk2Tf:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_identity(self, xp, scp):
        if False:
            while True:
                i = 10
        'Test the identity transfer function.'
        z = xp.array([])
        p = xp.array([])
        k = 1.0
        (b, a) = scp.signal.zpk2tf(z, p, k)
        return (b, a)