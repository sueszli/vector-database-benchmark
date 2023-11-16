import cupy
from cupy import testing
import cupyx.scipy.signal
import cupyx.scipy.signal.windows
try:
    import scipy.signal
    import scipy.signal.windows
except ImportError:
    pass
import numpy as np
import pytest
from math import gcd
padtype_options = ['mean', 'median', 'minimum', 'maximum', 'line']
_upfirdn_modes = ['constant', 'wrap', 'edge', 'smooth', 'symmetric', 'reflect', 'antisymmetric', 'antireflect', 'line']
padtype_options += _upfirdn_modes

@testing.with_requires('scipy')
class TestResample:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        sig = xp.arange(128)
        num = 256
        win = scp.signal.get_window(('kaiser', 8.0), 160)
        with pytest.raises(ValueError):
            scp.signal.resample(sig, num, window=win)
        with pytest.raises(ValueError):
            scp.signal.resample_poly(sig, 'yo', 1)
        with pytest.raises(ValueError):
            scp.signal.resample_poly(sig, 1, 0)
        sig2 = xp.tile(xp.arange(160), (2, 1))
        return scp.signal.resample(sig2, num, axis=-1, window=win).copy()

    @pytest.mark.parametrize('window', (None, 'hamming'))
    @pytest.mark.parametrize('N', (20, 19))
    @pytest.mark.parametrize('num', (100, 101, 10, 11))
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-07)
    def test_rfft(self, N, num, window, xp, scp):
        if False:
            return 10
        results = []
        x = xp.linspace(0, 10, N, endpoint=False)
        y = xp.cos(-x ** 2 / 6.0)
        results.append(scp.signal.resample(y, num, window=window).copy())
        results.append(scp.signal.resample(y + 0j, num, window=window).real.copy())
        y = xp.array([xp.cos(-x ** 2 / 6.0), xp.sin(-x ** 2 / 6.0)])
        y_complex = y + 0j
        results.append(scp.signal.resample(y, num, axis=1, window=window).copy())
        results.append(scp.signal.resample(y_complex, num, axis=1, window=window).real.copy())
        return results

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-07)
    def test_input_domain(self, xp, scp):
        if False:
            return 10
        tsig = xp.arange(256) + 0j
        fsig = xp.fft.fft(tsig)
        num = 256
        return (scp.signal.resample(fsig, num, domain='freq'), scp.signal.resample(tsig, num, domain='time'))

    @pytest.mark.parametrize('nx', (1, 2, 3, 5, 8))
    @pytest.mark.parametrize('ny', (1, 2, 3, 5, 8))
    @pytest.mark.parametrize('dtype', ('float', 'complex'))
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_dc(self, nx, ny, dtype, xp, scp):
        if False:
            i = 10
            return i + 15
        x = xp.array([1] * nx, dtype)
        y = scp.signal.resample(x, ny)
        return y.copy()

    @pytest.mark.parametrize('padtype', padtype_options)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.skip(reason='cval and mode is not supported on upfirdn')
    def test_mutable_window(self, padtype, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        impulse = xp.zeros(3)
        window = xp.random.RandomState(0).randn(2)
        scp.signal.resample_poly(impulse, 5, 1, window=window, padtype=padtype)
        return window

    @pytest.mark.parametrize('padtype', padtype_options)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-05, rtol=1e-05)
    @pytest.mark.skip(reason='cval and mode is not supported on upfirdn')
    def test_output_float32(self, padtype, xp, scp):
        if False:
            i = 10
            return i + 15
        x = xp.arange(10, dtype=xp.float32)
        h = xp.array([1, 1, 1], dtype=xp.float32)
        y = scp.signal.resample_poly(x, 1, 2, window=h, padtype=padtype)
        return y

    @pytest.mark.parametrize('padtype', padtype_options)
    @pytest.mark.parametrize('dtype', [cupy.float32, cupy.float64])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-05, rtol=1e-05)
    @pytest.mark.skip(reason='cval and mode is not supported on upfirdn')
    def test_output_match_dtype(self, padtype, dtype, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        x = xp.arange(10, dtype=dtype)
        y = scp.signal.resample_poly(x, 1, 2, padtype=padtype)
        return y

    @pytest.mark.parametrize('method, ext, padtype', [('fft', False, None)])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-07)
    def test_resample_methods(self, method, ext, padtype, xp, scp):
        if False:
            i = 10
            return i + 15
        rate = 100
        rates_to = [49, 50, 51, 99, 100, 101, 199, 200, 201]
        t = xp.arange(rate) / float(rate)
        freqs = xp.array((1.0, 10.0, 40.0))[:, xp.newaxis]
        x = xp.sin(2 * xp.pi * freqs * t) * scp.signal.windows.hann(rate)
        results = []
        for rate_to in rates_to:
            if method == 'fft':
                y_resamps = scp.signal.resample(x, rate_to, axis=-1)
            else:
                if ext and rate_to != rate:
                    g = gcd(rate_to, rate)
                    up = rate_to // g
                    down = rate // g
                    max_rate = max(up, down)
                    f_c = 1.0 / max_rate
                    half_len = 10 * max_rate
                    window = scp.signal.firwin(2 * half_len + 1, f_c, window=('kaiser', 5.0))
                    polyargs = {'window': window, 'padtype': padtype}
                else:
                    polyargs = {'padtype': padtype}
                y_resamps = scp.signal.resample_poly(x, rate_to, rate, axis=-1, **polyargs)
            results.append(y_resamps.copy())
        if method == 'fft':
            x1 = xp.array([1.0 + 0j, 0.0 + 0j])
            y1_test = scp.signal.resample(x1, 4)
            x2 = xp.array([1.0, 0.5, 0.0, 0.5])
            y2_test = scp.signal.resample(x2, 2)
            results.append(y1_test.copy())
            results.append(y2_test.copy())
        return results

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_poly_vs_filtfilt(self, xp, scp):
        if False:
            i = 10
            return i + 15
        try_types = (int, xp.float32, xp.complex64, float, complex)
        size = 10000
        down_factors = [2, 11, 79]
        results = []
        for dtype in try_types:
            x = testing.shaped_random((size,), xp, dtype, scale=1.0)
            if dtype in (xp.complex64, xp.complex128):
                x += 1j * testing.shaped_random((size,), xp, xp.float32, scale=1.0)
            x[0] = 0
            x[-1] = 0
            for down in down_factors:
                h = scp.signal.firwin(31, 1.0 / down, window='hamming')
                hc = scp.signal.convolve(h, h[::-1])
                y = scp.signal.resample_poly(x, 1, down, window=hc)
                results.append(y)
        return results

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_correlate1d(self, xp, scp):
        if False:
            return 10
        results = []
        for down in [2, 4]:
            for nx in range(1, 40, down):
                for nweights in (32, 33):
                    x = testing.shaped_random((nx,), xp, xp.float64, scale=1.0)
                    weights = testing.shaped_random((nweights,), xp, xp.float64, scale=1.0)
                    y_s = scp.signal.resample_poly(x, up=1, down=down, window=weights)
                    results.append(y_s)
        return results

@testing.with_requires('scipy')
class TestDecimate:

    @pytest.mark.parametrize('mod', [(cupy, cupyx.scipy), (np, scipy)])
    def test_bad_args(self, mod):
        if False:
            i = 10
            return i + 15
        (xp, scp) = mod
        x = xp.arange(12)
        with pytest.raises(TypeError):
            scp.signal.decimate(x, q=0.5, n=1)
        with pytest.raises(TypeError):
            scp.signal.decimate(x, q=2, n=0.5)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic_IIR(self, xp, scp):
        if False:
            while True:
                i = 10
        x = xp.arange(12)
        y = scp.signal.decimate(x, 2, n=1, ftype='iir', zero_phase=False).round()
        return y

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_basic_FIR(self, xp, scp):
        if False:
            while True:
                i = 10
        x = xp.arange(12)
        y = scp.signal.decimate(x, 2, n=1, ftype='fir', zero_phase=False).round()
        return y

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_shape(self, xp, scp):
        if False:
            print('Hello World!')
        z = xp.zeros((30, 30))
        d0 = scp.signal.decimate(z, 2, axis=0, zero_phase=False)
        d1 = scp.signal.decimate(z, 2, axis=1, zero_phase=False)
        return (d0, d1)

    @pytest.mark.xfail(reason='Sometimes it fails depending on hardware')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_phaseshift_FIR(self, xp, scp):
        if False:
            i = 10
            return i + 15
        return self._test_phaseshift(xp, scp, method='fir', zero_phase=False)

    @pytest.mark.xfail(reason='Sometimes it fails depending on hardware')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zero_phase_FIR(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        return self._test_phaseshift(xp, scp, method='fir', zero_phase=True)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=0.0001, rtol=0.0001)
    def test_phaseshift_IIR(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        return self._test_phaseshift(xp, scp, method='iir', zero_phase=False)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=0.0001, rtol=0.0001)
    def test_zero_phase_IIR(self, xp, scp):
        if False:
            while True:
                i = 10
        return self._test_phaseshift(xp, scp, method='iir', zero_phase=True)

    def _test_phaseshift(self, xp, scp, method, zero_phase):
        if False:
            return 10
        rate = 120
        rates_to = [15, 20, 30, 40]
        t_tot = int(100)
        t = xp.arange(rate * t_tot + 1) / float(rate)
        freqs = xp.array(rates_to) * 0.8 / 2
        d = xp.exp(1j * 2 * xp.pi * freqs[:, xp.newaxis] * t) * scp.signal.windows.tukey(t.size, 0.1)
        results = []
        for rate_to in rates_to:
            q = rate // rate_to
            t_to = xp.arange(rate_to * t_tot + 1) / float(rate_to)
            d_tos = xp.exp(1j * 2 * xp.pi * freqs[:, xp.newaxis] * t_to) * scp.signal.windows.tukey(t_to.size, 0.1)
            if method == 'fir':
                n = 30
                system = scp.signal.dlti(scp.signal.firwin(n + 1, 1.0 / q, window='hamming'), 1.0)
            elif method == 'iir':
                n = 8
                wc = 0.8 * xp.pi / q
                system = scp.signal.dlti(*scp.signal.cheby1(n, 0.05, wc / xp.pi))
            if zero_phase is False:
                (_, h_resps) = scp.signal.freqz(system.num, system.den, freqs / rate * 2 * xp.pi)
                h_resps /= xp.abs(h_resps)
            else:
                h_resps = xp.ones_like(freqs)
            y_resamps = scp.signal.decimate(d.real, q, n, ftype=system, zero_phase=zero_phase)
            results.append(y_resamps)
            h_resamps = xp.sum(d_tos.conj() * y_resamps, axis=-1)
            h_resamps /= xp.abs(h_resamps)
        return results

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-07)
    def test_auto_n(self, xp, scp):
        if False:
            while True:
                i = 10
        sfreq = 100.0
        n = 1000
        t = xp.arange(n) / sfreq
        x = xp.sqrt(2.0 / n) * xp.sin(2 * xp.pi * (sfreq / 30.0) * t)
        x_out = scp.signal.decimate(x, 30, ftype='fir')
        return x_out

    @testing.with_requires('scipy>=1.10')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=5e-05, rtol=5e-05)
    def test_long_float32(self, xp, scp):
        if False:
            return 10
        x = scp.signal.decimate(xp.ones(10000, dtype=np.float32), 10)
        return x

    @testing.with_requires('scipy>=1.10')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_float16_upcast(self, xp, scp):
        if False:
            return 10
        x = scp.signal.decimate(xp.ones(100, dtype=xp.float16), 10)
        return x

    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.skip(reason='zpk2tf is returning real outputs instead of complex ones')
    def test_complex_iir_dlti(self, xp, scp):
        if False:
            while True:
                i = 10
        fcentre = 50
        fwidth = 5
        fs = 1000.0
        (z, p, k) = scp.signal.butter(2, 2 * xp.pi * fwidth / 2, output='zpk', fs=fs)
        z = z.astype(complex) * xp.exp(2j * xp.pi * fcentre / fs)
        p = p.astype(complex) * xp.exp(2j * xp.pi * fcentre / fs)
        system = scp.signal.dlti(z, p, k)
        t = xp.arange(200) / fs
        u = xp.exp(2j * xp.pi * fcentre * t) + 0.5 * xp.exp(-2j * xp.pi * fcentre * t)
        ynzp = scp.signal.decimate(u, 2, ftype=system, zero_phase=False)
        yzp = scp.signal.decimate(u, 2, ftype=system, zero_phase=True)
        return (ynzp, yzp)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.skip(reason='roots does not support non-symmetric inputs')
    def test_complex_fir_dlti(self, xp, scp):
        if False:
            print('Hello World!')
        fcentre = 50
        fwidth = 5
        fs = 1000.0
        numtaps = 20
        bbase = scp.signal.firwin(numtaps, fwidth / 2, fs=fs)
        zbase = xp.roots(bbase)
        zrot = zbase * xp.exp(2j * xp.pi * fcentre / fs)
        bz = bbase[0] * xp.poly(zrot)
        system = scp.signal.dlti(bz, 1)
        t = xp.arange(200) / fs
        u = xp.exp(2j * xp.pi * fcentre * t) + 0.5 * xp.exp(-2j * xp.pi * fcentre * t)
        ynzp = scp.signal.decimate(u, 2, ftype=system, zero_phase=False)
        yzp = scp.signal.decimate(u, 2, ftype=system, zero_phase=True)
        return (ynzp, yzp)