import cupy
import cupyx.scipy.signal as signal
from cupy import testing
import pytest
from pytest import raises as assert_raises

@testing.with_requires('scipy')
class TestKaiser:

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_kaiser_beta(self, xp, scp):
        if False:
            return 10
        k = scp.signal.kaiser_beta
        return (k(58.7), k(22.0), k(21.0), k(10.0))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_kaiser_atten(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        k = scp.signal.kaiser_atten
        return (k(1, 1.0), k(2, 1.0 / xp.pi))

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_kaiserord(self, xp, scp):
        if False:
            i = 10
            return i + 15
        assert_raises(ValueError, scp.signal.kaiserord, 1.0, 1.0)
        return scp.signal.kaiserord(2.285 + 7.95 - 0.001, 1 / xp.pi)

@testing.with_requires('scipy')
class TestFirwin:

    @pytest.mark.parametrize('args, kwds', [((51, 0.5), dict()), ((52, 0.5), dict(window='nuttall')), ((53, 0.5), dict(pass_zero=False)), ((54, [0.2, 0.4]), dict(pass_zero=False)), ((55, [0.2, 0.4]), dict()), ((56, [0.2, 0.4, 0.6, 0.8]), dict(pass_zero=False, scale=False)), ((57, [0.2, 0.4, 0.6, 0.8]), dict()), ((58, 0.1), dict(width=0.03)), ((59, 0.1), dict(pass_zero=False))])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_response(self, xp, scp, args, kwds):
        if False:
            while True:
                i = 10
        h = scp.signal.firwin(*args, **kwds)
        return h

    @pytest.mark.parametrize('case', [([0.5], True, (0, 1)), ([0.2, 0.6], False, (0.4, 1)), ([0.5], False, (1, 1))])
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test_scaling(self, xp, scp, case):
        if False:
            i = 10
            return i + 15
        '\n        For one lowpass, bandpass, and highpass example filter, this test\n        checks two things:\n          - the mean squared error over the frequency domain of the unscaled\n            filter is smaller than the scaled filter (true for rectangular\n            window)\n          - the response of the scaled filter is exactly unity at the center\n            of the first passband\n        '
        N = 11
        (cutoff, pass_zero, expected_responce) = case
        fw = scp.signal.firwin
        h = fw(N, cutoff, scale=False, pass_zero=pass_zero, window='ones')
        hs = fw(N, cutoff, scale=True, pass_zero=pass_zero, window='ones')
        return (h, hs)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_lowpass(self, xp, scp):
        if False:
            print('Hello World!')
        width = 0.04
        (ntaps, beta) = scp.signal.kaiserord(120, width)
        kwargs = dict(cutoff=0.5, window=('kaiser', beta), scale=False)
        taps = scp.signal.firwin(ntaps, **kwargs)
        return taps

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_highpass(self, xp, scp):
        if False:
            return 10
        width = 0.04
        (ntaps, beta) = scp.signal.kaiserord(120, width)
        ntaps |= 1
        kwargs = dict(cutoff=0.5, window=('kaiser', beta), scale=False)
        taps = scp.signal.firwin(ntaps, pass_zero=False, **kwargs)
        return taps

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bandpass(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        width = 0.04
        (ntaps, beta) = scp.signal.kaiserord(120, width)
        kwargs = dict(cutoff=[0.3, 0.7], window=('kaiser', beta), scale=False)
        taps = scp.signal.firwin(ntaps, pass_zero=False, **kwargs)
        return taps

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_bandstop_multi(self, xp, scp):
        if False:
            return 10
        width = 0.04
        (ntaps, beta) = scp.signal.kaiserord(120, width)
        kwargs = dict(cutoff=[0.2, 0.5, 0.8], window=('kaiser', beta), scale=False)
        taps = scp.signal.firwin(ntaps, **kwargs)
        return taps

    def test_bad_cutoff(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that invalid cutoff argument raises ValueError.'
        assert_raises(ValueError, signal.firwin, 99, -0.5)
        assert_raises(ValueError, signal.firwin, 99, 1.5)
        assert_raises(ValueError, signal.firwin, 99, [0, 0.5])
        assert_raises(ValueError, signal.firwin, 99, [0.5, 1])
        assert_raises(ValueError, signal.firwin, 99, [0.1, 0.5, 0.2])
        assert_raises(ValueError, signal.firwin, 99, [0.1, 0.5, 0.5])
        assert_raises(ValueError, signal.firwin, 99, [])
        assert_raises(ValueError, signal.firwin, 99, [[0.1, 0.2], [0.3, 0.4]])
        assert_raises(ValueError, signal.firwin, 99, 50.0, fs=80)
        assert_raises(ValueError, signal.firwin, 99, [10, 20, 30], fs=50)

    def test_even_highpass_raises_value_error(self):
        if False:
            return 10
        'Test that attempt to create a highpass filter with an even number\n        of taps raises a ValueError exception.'
        assert_raises(ValueError, signal.firwin, 40, 0.5, pass_zero=False)
        assert_raises(ValueError, signal.firwin, 40, [0.25, 0.5])

    def test_bad_pass_zero(self):
        if False:
            i = 10
            return i + 15
        'Test degenerate pass_zero cases.'
        with assert_raises(ValueError, match='pass_zero must be'):
            signal.firwin(41, 0.5, pass_zero='foo')
        with assert_raises(TypeError):
            signal.firwin(41, 0.5, pass_zero=1.0)
        for pass_zero in ('lowpass', 'highpass'):
            with assert_raises(ValueError, match='cutoff must have one'):
                signal.firwin(41, [0.5, 0.6], pass_zero=pass_zero)
        for pass_zero in ('bandpass', 'bandstop'):
            with assert_raises(ValueError, match='must have at least two'):
                signal.firwin(41, [0.5], pass_zero=pass_zero)

@testing.with_requires('scipy')
class TestFirwin2:

    def test_invalid_args(self):
        if False:
            return 10
        with assert_raises(ValueError, match='must be of same length'):
            signal.firwin2(50, [0, 0.5, 1], [0.0, 1.0])
        with assert_raises(ValueError, match='ntaps must be less than nfreqs'):
            signal.firwin2(50, [0, 0.5, 1], [0.0, 1.0, 1.0], nfreqs=33)
        with assert_raises(ValueError, match='must be nondecreasing'):
            signal.firwin2(50, [0, 0.5, 0.4, 1.0], [0, 0.25, 0.5, 1.0])
        with assert_raises(ValueError, match='must not occur more than twice'):
            signal.firwin2(50, [0, 0.1, 0.1, 0.1, 1.0], [0.0, 0.5, 0.75, 1.0, 1.0])
        with assert_raises(ValueError, match='start with 0'):
            signal.firwin2(50, [0.5, 1.0], [0.0, 1.0])
        with assert_raises(ValueError, match='end with fs/2'):
            signal.firwin2(50, [0.0, 0.5], [0.0, 1.0])
        with assert_raises(ValueError, match='0 must not be repeated'):
            signal.firwin2(50, [0.0, 0.0, 0.5, 1.0], [1.0, 1.0, 0.0, 0.0])
        with assert_raises(ValueError, match='fs/2 must not be repeated'):
            signal.firwin2(50, [0.0, 0.5, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0])
        with assert_raises(ValueError, match='cannot contain numbers that are too close'):
            eps = cupy.finfo(float).eps
            signal.firwin2(50, [0.0, 0.5 - eps * 0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 0.0, 0.0])
        with assert_raises(ValueError, match='Type II filter'):
            signal.firwin2(16, [0.0, 0.5, 1.0], [0.0, 1.0, 1.0])
        with assert_raises(ValueError, match='Type III filter'):
            signal.firwin2(17, [0.0, 0.5, 1.0], [0.0, 1.0, 1.0], antisymmetric=True)
        with assert_raises(ValueError, match='Type III filter'):
            signal.firwin2(17, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0], antisymmetric=True)
        with assert_raises(ValueError, match='Type III filter'):
            signal.firwin2(17, [0.0, 0.5, 1.0], [1.0, 1.0, 1.0], antisymmetric=True)
        with assert_raises(ValueError, match='Type IV filter'):
            signal.firwin2(16, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0], antisymmetric=True)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test01(self, xp, scp):
        if False:
            print('Hello World!')
        beta = 12.0
        ntaps = 400
        freq = xp.asarray([0.0, 0.5, 1.0])
        gain = xp.asarray([1.0, 1.0, 0.0])
        taps = scp.signal.firwin2(ntaps, freq, gain, window=('kaiser', beta))
        return taps

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test02(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        beta = 12.0
        ntaps = 401
        freq = xp.asarray([0.0, 0.5, 0.5, 1.0])
        gain = xp.asarray([0.0, 0.0, 1.0, 1.0])
        taps = scp.signal.firwin2(ntaps, freq, gain, window=('kaiser', beta))
        return taps

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test03(self, xp, scp):
        if False:
            print('Hello World!')
        width = 0.02
        (ntaps, beta) = scp.signal.kaiserord(120, width)
        ntaps = int(ntaps) | 1
        freq = xp.asarray([0.0, 0.4, 0.4, 0.5, 0.5, 1.0])
        gain = xp.asarray([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
        taps = scp.signal.firwin2(ntaps, freq, gain, window=('kaiser', beta))
        return taps

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-14)
    def test04(self, xp, scp):
        if False:
            print('Hello World!')
        'Test firwin2 when window=None.'
        ntaps = 5
        freq = xp.asarray([0.0, 0.5, 0.5, 1.0])
        gain = xp.asarray([1.0, 1.0, 0.0, 0.0])
        taps = scp.signal.firwin2(ntaps, freq, gain, window=None, nfreqs=8193)
        return taps

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test05(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        'Test firwin2 for calculating Type IV filters'
        ntaps = 1500
        freq = xp.asarray([0.0, 1.0])
        gain = xp.asarray([0.0, 1.0])
        taps = scp.signal.firwin2(ntaps, freq, gain, window=None, antisymmetric=True)
        return taps

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test06(self, xp, scp):
        if False:
            return 10
        'Test firwin2 for calculating Type III filters'
        ntaps = 1501
        freq = xp.asarray([0.0, 0.5, 0.55, 1.0])
        gain = xp.asarray([0.0, 0.5, 0.0, 0.0])
        taps = scp.signal.firwin2(ntaps, freq, gain, window=None, antisymmetric=True)
        return taps

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_fs_nyq(self, xp, scp):
        if False:
            i = 10
            return i + 15
        taps1 = scp.signal.firwin2(80, xp.asarray([0.0, 0.5, 1.0]), xp.asarray([1.0, 1.0, 0.0]))
        taps2 = scp.signal.firwin2(80, xp.asarray([0.0, 30.0, 60.0]), xp.asarray([1.0, 1.0, 0.0]), fs=120.0)
        return (taps1, taps2)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_tuple(self, xp, scp):
        if False:
            while True:
                i = 10
        taps1 = scp.signal.firwin2(150, xp.asarray((0.0, 0.5, 0.5, 1.0)), xp.asarray((1.0, 1.0, 0.0, 0.0)))
        taps2 = scp.signal.firwin2(150, xp.asarray([0.0, 0.5, 0.5, 1.0]), xp.asarray([1.0, 1.0, 0.0, 0.0]))
        return (taps1, taps2)

    def test_input_modyfication(self):
        if False:
            print('Hello World!')
        freq1 = cupy.array([0.0, 0.5, 0.5, 1.0])
        freq2 = cupy.array(freq1, copy=True)
        signal.firwin2(80, freq1, cupy.array([1.0, 1.0, 0.0, 0.0]))
        assert (freq1 == freq2).all()

@testing.with_requires('scipy')
class TestFirls:

    def test_bad_args(self):
        if False:
            i = 10
            return i + 15
        firls = signal.firls
        assert_raises(ValueError, firls, 10, [0.1, 0.2], [0, 0])
        assert_raises(ValueError, firls, 11, [0.1, 0.2, 0.4], [0, 0, 0])
        assert_raises(ValueError, firls, 11, [0.1, 0.2, 0.3, 0.4], [0, 0, 0])
        assert_raises(ValueError, firls, 11, [0.1, 0.2], [0, 0], [1, 2])

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-13)
    def test_firls(self, xp, scp):
        if False:
            i = 10
            return i + 15
        N = 11
        a = 0.1
        h = scp.signal.firls(N, [0, a, 0.5 - a, 0.5], [1, 1, 0, 0], fs=1.0)
        return h

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_firls_freqz(self, xp, scp):
        if False:
            return 10
        N = 11
        a = 0.1
        h = scp.signal.firls(N, [0, a, 0.5 - a, 0.5], [1, 1, 0, 0], fs=1.0)
        (w, H) = scp.signal.freqz(h, 1)
        return (w, H)

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_compare(self, xp, scp):
        if False:
            while True:
                i = 10
        taps = scp.signal.firls(9, [0, 0.5, 0.55, 1], [1, 1, 0, 0], [1, 2])
        return taps

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_compare_2(self, xp, scp):
        if False:
            while True:
                i = 10
        taps = scp.signal.firls(11, [0, 0.5, 0.5, 1], [1, 1, 0, 0], [1, 2])
        return taps

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_compare_3(self, xp, scp):
        if False:
            i = 10
            return i + 15
        taps = scp.signal.firls(7, (0, 1, 2, 3, 4, 5), [1, 0, 0, 1, 1, 0], fs=20)
        return taps

    @pytest.mark.xfail(reason='https://github.com/scipy/scipy/issues/18533')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_rank_deficient(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        x = scp.signal.firls(21, [0, 0.1, 0.9, 1], [1, 1, 0, 0])
        (w, h) = scp.signal.freqz(x, fs=2.0)
        return (x, w, h)

    @pytest.mark.xfail(reason='https://github.com/scipy/scipy/issues/18533')
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_rank_deficient_2(self, xp, scp):
        if False:
            return 10
        x = scp.signal.firls(101, [0, 0.01, 0.99, 1], [1, 1, 0, 0])
        (w, h) = scp.signal.freqz(x, fs=2.0)
        return (x, w, h)

    def test_rank_deficient_3(self):
        if False:
            print('Hello World!')
        x = signal.firls(101, [0, 0.01, 0.99, 1], [1, 1, 0, 0])
        (w, h) = signal.freqz(x, fs=2.0)
        mask = w < 0.01
        assert mask.sum() > 3
        testing.assert_allclose(cupy.abs(h[mask]), 1.0, atol=0.0001)
        mask = w > 0.99
        assert mask.sum() > 3
        testing.assert_allclose(cupy.abs(h[mask]), 0.0, atol=0.0001)

@testing.with_requires('scipy')
class TestMinimumPhase:

    def test_bad_args(self):
        if False:
            return 10
        assert_raises(ValueError, signal.minimum_phase, cupy.array([1.0]))
        assert_raises(ValueError, signal.minimum_phase, cupy.array([1.0, 1.0]))
        assert_raises(ValueError, signal.minimum_phase, cupy.full(10, 1j))
        assert_raises((AttributeError, ValueError), signal.minimum_phase, 'foo')
        assert_raises(ValueError, signal.minimum_phase, cupy.ones(10), n_fft=8)
        assert_raises(ValueError, signal.minimum_phase, cupy.ones(10), method='foo')

    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_homomorphic(self, xp, scp):
        if False:
            while True:
                i = 10
        h = xp.asarray([1, -1])
        h_new = scp.signal.minimum_phase(xp.convolve(h, h[::-1]))
        return h_new

    @pytest.mark.parametrize('n', [2, 3, 10, 11, 15, 16, 17, 20, 21, 100, 101])
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_homomorphic_2(self, xp, scp, n):
        if False:
            return 10
        rng = cupy.random.RandomState(0)
        h = rng.randn(n)
        if xp != cupy:
            h = h.get()
        h_new = scp.signal.minimum_phase(xp.convolve(h, h[::-1]))
        return h_new

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=2e-05)
    def test_hilbert(self, xp, scp):
        if False:
            for i in range(10):
                print('nop')
        from scipy.signal import remez
        h_linear = remez(151, [0, 0.2, 0.3, 1.0], [1, 0], fs=2)
        if xp == cupy:
            h_linear = cupy.asarray(h_linear)
        return scp.signal.minimum_phase(h_linear, method='hilbert')