"""Tools for spectral analysis.
"""
import numpy as np
from scipy import fft as sp_fft
from . import _signaltools
from .windows import get_window
from ._spectral import _lombscargle
from ._arraytools import const_ext, even_ext, odd_ext, zero_ext
import warnings
__all__ = ['periodogram', 'welch', 'lombscargle', 'csd', 'coherence', 'spectrogram', 'stft', 'istft', 'check_COLA', 'check_NOLA']

def lombscargle(x, y, freqs, precenter=False, normalize=False):
    if False:
        return 10
    '\n    lombscargle(x, y, freqs)\n\n    Computes the Lomb-Scargle periodogram.\n\n    The Lomb-Scargle periodogram was developed by Lomb [1]_ and further\n    extended by Scargle [2]_ to find, and test the significance of weak\n    periodic signals with uneven temporal sampling.\n\n    When *normalize* is False (default) the computed periodogram\n    is unnormalized, it takes the value ``(A**2) * N/4`` for a harmonic\n    signal with amplitude A for sufficiently large N.\n\n    When *normalize* is True the computed periodogram is normalized by\n    the residuals of the data around a constant reference model (at zero).\n\n    Input arrays should be 1-D and will be cast to float64.\n\n    Parameters\n    ----------\n    x : array_like\n        Sample times.\n    y : array_like\n        Measurement values.\n    freqs : array_like\n        Angular frequencies for output periodogram.\n    precenter : bool, optional\n        Pre-center measurement values by subtracting the mean.\n    normalize : bool, optional\n        Compute normalized periodogram.\n\n    Returns\n    -------\n    pgram : array_like\n        Lomb-Scargle periodogram.\n\n    Raises\n    ------\n    ValueError\n        If the input arrays `x` and `y` do not have the same shape.\n\n    See Also\n    --------\n    istft: Inverse Short Time Fourier Transform\n    check_COLA: Check whether the Constant OverLap Add (COLA) constraint is met\n    welch: Power spectral density by Welch\'s method\n    spectrogram: Spectrogram by Welch\'s method\n    csd: Cross spectral density by Welch\'s method\n\n    Notes\n    -----\n    This subroutine calculates the periodogram using a slightly\n    modified algorithm due to Townsend [3]_ which allows the\n    periodogram to be calculated using only a single pass through\n    the input arrays for each frequency.\n\n    The algorithm running time scales roughly as O(x * freqs) or O(N^2)\n    for a large number of samples and frequencies.\n\n    References\n    ----------\n    .. [1] N.R. Lomb "Least-squares frequency analysis of unequally spaced\n           data", Astrophysics and Space Science, vol 39, pp. 447-462, 1976\n\n    .. [2] J.D. Scargle "Studies in astronomical time series analysis. II -\n           Statistical aspects of spectral analysis of unevenly spaced data",\n           The Astrophysical Journal, vol 263, pp. 835-853, 1982\n\n    .. [3] R.H.D. Townsend, "Fast calculation of the Lomb-Scargle\n           periodogram using graphics processing units.", The Astrophysical\n           Journal Supplement Series, vol 191, pp. 247-253, 2010\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> import matplotlib.pyplot as plt\n    >>> rng = np.random.default_rng()\n\n    First define some input parameters for the signal:\n\n    >>> A = 2.\n    >>> w0 = 1.  # rad/sec\n    >>> nin = 150\n    >>> nout = 100000\n\n    Randomly generate sample times:\n\n    >>> x = rng.uniform(0, 10*np.pi, nin)\n\n    Plot a sine wave for the selected times:\n\n    >>> y = A * np.cos(w0*x)\n\n    Define the array of frequencies for which to compute the periodogram:\n\n    >>> w = np.linspace(0.01, 10, nout)\n\n    Calculate Lomb-Scargle periodogram:\n\n    >>> import scipy.signal as signal\n    >>> pgram = signal.lombscargle(x, y, w, normalize=True)\n\n    Now make a plot of the input data:\n\n    >>> fig, (ax_t, ax_w) = plt.subplots(2, 1, constrained_layout=True)\n    >>> ax_t.plot(x, y, \'b+\')\n    >>> ax_t.set_xlabel(\'Time [s]\')\n\n    Then plot the normalized periodogram:\n\n    >>> ax_w.plot(w, pgram)\n    >>> ax_w.set_xlabel(\'Angular frequency [rad/s]\')\n    >>> ax_w.set_ylabel(\'Normalized amplitude\')\n    >>> plt.show()\n\n    '
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    freqs = np.ascontiguousarray(freqs, dtype=np.float64)
    assert x.ndim == 1
    assert y.ndim == 1
    assert freqs.ndim == 1
    if precenter:
        pgram = _lombscargle(x, y - y.mean(), freqs)
    else:
        pgram = _lombscargle(x, y, freqs)
    if normalize:
        pgram *= 2 / np.dot(y, y)
    return pgram

def periodogram(x, fs=1.0, window='boxcar', nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1):
    if False:
        return 10
    "\n    Estimate power spectral density using a periodogram.\n\n    Parameters\n    ----------\n    x : array_like\n        Time series of measurement values\n    fs : float, optional\n        Sampling frequency of the `x` time series. Defaults to 1.0.\n    window : str or tuple or array_like, optional\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be equal to the length\n        of the axis over which the periodogram is computed. Defaults\n        to 'boxcar'.\n    nfft : int, optional\n        Length of the FFT used. If `None` the length of `x` will be\n        used.\n    detrend : str or function or `False`, optional\n        Specifies how to detrend each segment. If `detrend` is a\n        string, it is passed as the `type` argument to the `detrend`\n        function. If it is a function, it takes a segment and returns a\n        detrended segment. If `detrend` is `False`, no detrending is\n        done. Defaults to 'constant'.\n    return_onesided : bool, optional\n        If `True`, return a one-sided spectrum for real data. If\n        `False` return a two-sided spectrum. Defaults to `True`, but for\n        complex data, a two-sided spectrum is always returned.\n    scaling : { 'density', 'spectrum' }, optional\n        Selects between computing the power spectral density ('density')\n        where `Pxx` has units of V**2/Hz and computing the power\n        spectrum ('spectrum') where `Pxx` has units of V**2, if `x`\n        is measured in V and `fs` is measured in Hz. Defaults to\n        'density'\n    axis : int, optional\n        Axis along which the periodogram is computed; the default is\n        over the last axis (i.e. ``axis=-1``).\n\n    Returns\n    -------\n    f : ndarray\n        Array of sample frequencies.\n    Pxx : ndarray\n        Power spectral density or power spectrum of `x`.\n\n    See Also\n    --------\n    welch: Estimate power spectral density using Welch's method\n    lombscargle: Lomb-Scargle periodogram for unevenly sampled data\n\n    Notes\n    -----\n    .. versionadded:: 0.12.0\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy import signal\n    >>> import matplotlib.pyplot as plt\n    >>> rng = np.random.default_rng()\n\n    Generate a test signal, a 2 Vrms sine wave at 1234 Hz, corrupted by\n    0.001 V**2/Hz of white noise sampled at 10 kHz.\n\n    >>> fs = 10e3\n    >>> N = 1e5\n    >>> amp = 2*np.sqrt(2)\n    >>> freq = 1234.0\n    >>> noise_power = 0.001 * fs / 2\n    >>> time = np.arange(N) / fs\n    >>> x = amp*np.sin(2*np.pi*freq*time)\n    >>> x += rng.normal(scale=np.sqrt(noise_power), size=time.shape)\n\n    Compute and plot the power spectral density.\n\n    >>> f, Pxx_den = signal.periodogram(x, fs)\n    >>> plt.semilogy(f, Pxx_den)\n    >>> plt.ylim([1e-7, 1e2])\n    >>> plt.xlabel('frequency [Hz]')\n    >>> plt.ylabel('PSD [V**2/Hz]')\n    >>> plt.show()\n\n    If we average the last half of the spectral density, to exclude the\n    peak, we can recover the noise power on the signal.\n\n    >>> np.mean(Pxx_den[25000:])\n    0.000985320699252543\n\n    Now compute and plot the power spectrum.\n\n    >>> f, Pxx_spec = signal.periodogram(x, fs, 'flattop', scaling='spectrum')\n    >>> plt.figure()\n    >>> plt.semilogy(f, np.sqrt(Pxx_spec))\n    >>> plt.ylim([1e-4, 1e1])\n    >>> plt.xlabel('frequency [Hz]')\n    >>> plt.ylabel('Linear spectrum [V RMS]')\n    >>> plt.show()\n\n    The peak height in the power spectrum is an estimate of the RMS\n    amplitude.\n\n    >>> np.sqrt(Pxx_spec.max())\n    2.0077340678640727\n\n    "
    x = np.asarray(x)
    if x.size == 0:
        return (np.empty(x.shape), np.empty(x.shape))
    if window is None:
        window = 'boxcar'
    if nfft is None:
        nperseg = x.shape[axis]
    elif nfft == x.shape[axis]:
        nperseg = nfft
    elif nfft > x.shape[axis]:
        nperseg = x.shape[axis]
    elif nfft < x.shape[axis]:
        s = [np.s_[:]] * len(x.shape)
        s[axis] = np.s_[:nfft]
        x = x[tuple(s)]
        nperseg = nfft
        nfft = None
    if hasattr(window, 'size'):
        if window.size != nperseg:
            raise ValueError('the size of the window must be the same size of the input on the specified axis')
    return welch(x, fs=fs, window=window, nperseg=nperseg, noverlap=0, nfft=nfft, detrend=detrend, return_onesided=return_onesided, scaling=scaling, axis=axis)

def welch(x, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean'):
    if False:
        while True:
            i = 10
    '\n    Estimate power spectral density using Welch\'s method.\n\n    Welch\'s method [1]_ computes an estimate of the power spectral\n    density by dividing the data into overlapping segments, computing a\n    modified periodogram for each segment and averaging the\n    periodograms.\n\n    Parameters\n    ----------\n    x : array_like\n        Time series of measurement values\n    fs : float, optional\n        Sampling frequency of the `x` time series. Defaults to 1.0.\n    window : str or tuple or array_like, optional\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg. Defaults\n        to a Hann window.\n    nperseg : int, optional\n        Length of each segment. Defaults to None, but if window is str or\n        tuple, is set to 256, and if window is array_like, is set to the\n        length of the window.\n    noverlap : int, optional\n        Number of points to overlap between segments. If `None`,\n        ``noverlap = nperseg // 2``. Defaults to `None`.\n    nfft : int, optional\n        Length of the FFT used, if a zero padded FFT is desired. If\n        `None`, the FFT length is `nperseg`. Defaults to `None`.\n    detrend : str or function or `False`, optional\n        Specifies how to detrend each segment. If `detrend` is a\n        string, it is passed as the `type` argument to the `detrend`\n        function. If it is a function, it takes a segment and returns a\n        detrended segment. If `detrend` is `False`, no detrending is\n        done. Defaults to \'constant\'.\n    return_onesided : bool, optional\n        If `True`, return a one-sided spectrum for real data. If\n        `False` return a two-sided spectrum. Defaults to `True`, but for\n        complex data, a two-sided spectrum is always returned.\n    scaling : { \'density\', \'spectrum\' }, optional\n        Selects between computing the power spectral density (\'density\')\n        where `Pxx` has units of V**2/Hz and computing the power\n        spectrum (\'spectrum\') where `Pxx` has units of V**2, if `x`\n        is measured in V and `fs` is measured in Hz. Defaults to\n        \'density\'\n    axis : int, optional\n        Axis along which the periodogram is computed; the default is\n        over the last axis (i.e. ``axis=-1``).\n    average : { \'mean\', \'median\' }, optional\n        Method to use when averaging periodograms. Defaults to \'mean\'.\n\n        .. versionadded:: 1.2.0\n\n    Returns\n    -------\n    f : ndarray\n        Array of sample frequencies.\n    Pxx : ndarray\n        Power spectral density or power spectrum of x.\n\n    See Also\n    --------\n    periodogram: Simple, optionally modified periodogram\n    lombscargle: Lomb-Scargle periodogram for unevenly sampled data\n\n    Notes\n    -----\n    An appropriate amount of overlap will depend on the choice of window\n    and on your requirements. For the default Hann window an overlap of\n    50% is a reasonable trade off between accurately estimating the\n    signal power, while not over counting any of the data. Narrower\n    windows may require a larger overlap.\n\n    If `noverlap` is 0, this method is equivalent to Bartlett\'s method\n    [2]_.\n\n    .. versionadded:: 0.12.0\n\n    References\n    ----------\n    .. [1] P. Welch, "The use of the fast Fourier transform for the\n           estimation of power spectra: A method based on time averaging\n           over short, modified periodograms", IEEE Trans. Audio\n           Electroacoust. vol. 15, pp. 70-73, 1967.\n    .. [2] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",\n           Biometrika, vol. 37, pp. 1-16, 1950.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy import signal\n    >>> import matplotlib.pyplot as plt\n    >>> rng = np.random.default_rng()\n\n    Generate a test signal, a 2 Vrms sine wave at 1234 Hz, corrupted by\n    0.001 V**2/Hz of white noise sampled at 10 kHz.\n\n    >>> fs = 10e3\n    >>> N = 1e5\n    >>> amp = 2*np.sqrt(2)\n    >>> freq = 1234.0\n    >>> noise_power = 0.001 * fs / 2\n    >>> time = np.arange(N) / fs\n    >>> x = amp*np.sin(2*np.pi*freq*time)\n    >>> x += rng.normal(scale=np.sqrt(noise_power), size=time.shape)\n\n    Compute and plot the power spectral density.\n\n    >>> f, Pxx_den = signal.welch(x, fs, nperseg=1024)\n    >>> plt.semilogy(f, Pxx_den)\n    >>> plt.ylim([0.5e-3, 1])\n    >>> plt.xlabel(\'frequency [Hz]\')\n    >>> plt.ylabel(\'PSD [V**2/Hz]\')\n    >>> plt.show()\n\n    If we average the last half of the spectral density, to exclude the\n    peak, we can recover the noise power on the signal.\n\n    >>> np.mean(Pxx_den[256:])\n    0.0009924865443739191\n\n    Now compute and plot the power spectrum.\n\n    >>> f, Pxx_spec = signal.welch(x, fs, \'flattop\', 1024, scaling=\'spectrum\')\n    >>> plt.figure()\n    >>> plt.semilogy(f, np.sqrt(Pxx_spec))\n    >>> plt.xlabel(\'frequency [Hz]\')\n    >>> plt.ylabel(\'Linear spectrum [V RMS]\')\n    >>> plt.show()\n\n    The peak height in the power spectrum is an estimate of the RMS\n    amplitude.\n\n    >>> np.sqrt(Pxx_spec.max())\n    2.0077340678640727\n\n    If we now introduce a discontinuity in the signal, by increasing the\n    amplitude of a small portion of the signal by 50, we can see the\n    corruption of the mean average power spectral density, but using a\n    median average better estimates the normal behaviour.\n\n    >>> x[int(N//2):int(N//2)+10] *= 50.\n    >>> f, Pxx_den = signal.welch(x, fs, nperseg=1024)\n    >>> f_med, Pxx_den_med = signal.welch(x, fs, nperseg=1024, average=\'median\')\n    >>> plt.semilogy(f, Pxx_den, label=\'mean\')\n    >>> plt.semilogy(f_med, Pxx_den_med, label=\'median\')\n    >>> plt.ylim([0.5e-3, 1])\n    >>> plt.xlabel(\'frequency [Hz]\')\n    >>> plt.ylabel(\'PSD [V**2/Hz]\')\n    >>> plt.legend()\n    >>> plt.show()\n\n    '
    (freqs, Pxx) = csd(x, x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=detrend, return_onesided=return_onesided, scaling=scaling, axis=axis, average=average)
    return (freqs, Pxx.real)

def csd(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean'):
    if False:
        i = 10
        return i + 15
    '\n    Estimate the cross power spectral density, Pxy, using Welch\'s method.\n\n    Parameters\n    ----------\n    x : array_like\n        Time series of measurement values\n    y : array_like\n        Time series of measurement values\n    fs : float, optional\n        Sampling frequency of the `x` and `y` time series. Defaults\n        to 1.0.\n    window : str or tuple or array_like, optional\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg. Defaults\n        to a Hann window.\n    nperseg : int, optional\n        Length of each segment. Defaults to None, but if window is str or\n        tuple, is set to 256, and if window is array_like, is set to the\n        length of the window.\n    noverlap: int, optional\n        Number of points to overlap between segments. If `None`,\n        ``noverlap = nperseg // 2``. Defaults to `None`.\n    nfft : int, optional\n        Length of the FFT used, if a zero padded FFT is desired. If\n        `None`, the FFT length is `nperseg`. Defaults to `None`.\n    detrend : str or function or `False`, optional\n        Specifies how to detrend each segment. If `detrend` is a\n        string, it is passed as the `type` argument to the `detrend`\n        function. If it is a function, it takes a segment and returns a\n        detrended segment. If `detrend` is `False`, no detrending is\n        done. Defaults to \'constant\'.\n    return_onesided : bool, optional\n        If `True`, return a one-sided spectrum for real data. If\n        `False` return a two-sided spectrum. Defaults to `True`, but for\n        complex data, a two-sided spectrum is always returned.\n    scaling : { \'density\', \'spectrum\' }, optional\n        Selects between computing the cross spectral density (\'density\')\n        where `Pxy` has units of V**2/Hz and computing the cross spectrum\n        (\'spectrum\') where `Pxy` has units of V**2, if `x` and `y` are\n        measured in V and `fs` is measured in Hz. Defaults to \'density\'\n    axis : int, optional\n        Axis along which the CSD is computed for both inputs; the\n        default is over the last axis (i.e. ``axis=-1``).\n    average : { \'mean\', \'median\' }, optional\n        Method to use when averaging periodograms. If the spectrum is\n        complex, the average is computed separately for the real and\n        imaginary parts. Defaults to \'mean\'.\n\n        .. versionadded:: 1.2.0\n\n    Returns\n    -------\n    f : ndarray\n        Array of sample frequencies.\n    Pxy : ndarray\n        Cross spectral density or cross power spectrum of x,y.\n\n    See Also\n    --------\n    periodogram: Simple, optionally modified periodogram\n    lombscargle: Lomb-Scargle periodogram for unevenly sampled data\n    welch: Power spectral density by Welch\'s method. [Equivalent to\n           csd(x,x)]\n    coherence: Magnitude squared coherence by Welch\'s method.\n\n    Notes\n    -----\n    By convention, Pxy is computed with the conjugate FFT of X\n    multiplied by the FFT of Y.\n\n    If the input series differ in length, the shorter series will be\n    zero-padded to match.\n\n    An appropriate amount of overlap will depend on the choice of window\n    and on your requirements. For the default Hann window an overlap of\n    50% is a reasonable trade off between accurately estimating the\n    signal power, while not over counting any of the data. Narrower\n    windows may require a larger overlap.\n\n    .. versionadded:: 0.16.0\n\n    References\n    ----------\n    .. [1] P. Welch, "The use of the fast Fourier transform for the\n           estimation of power spectra: A method based on time averaging\n           over short, modified periodograms", IEEE Trans. Audio\n           Electroacoust. vol. 15, pp. 70-73, 1967.\n    .. [2] Rabiner, Lawrence R., and B. Gold. "Theory and Application of\n           Digital Signal Processing" Prentice-Hall, pp. 414-419, 1975\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy import signal\n    >>> import matplotlib.pyplot as plt\n    >>> rng = np.random.default_rng()\n\n    Generate two test signals with some common features.\n\n    >>> fs = 10e3\n    >>> N = 1e5\n    >>> amp = 20\n    >>> freq = 1234.0\n    >>> noise_power = 0.001 * fs / 2\n    >>> time = np.arange(N) / fs\n    >>> b, a = signal.butter(2, 0.25, \'low\')\n    >>> x = rng.normal(scale=np.sqrt(noise_power), size=time.shape)\n    >>> y = signal.lfilter(b, a, x)\n    >>> x += amp*np.sin(2*np.pi*freq*time)\n    >>> y += rng.normal(scale=0.1*np.sqrt(noise_power), size=time.shape)\n\n    Compute and plot the magnitude of the cross spectral density.\n\n    >>> f, Pxy = signal.csd(x, y, fs, nperseg=1024)\n    >>> plt.semilogy(f, np.abs(Pxy))\n    >>> plt.xlabel(\'frequency [Hz]\')\n    >>> plt.ylabel(\'CSD [V**2/Hz]\')\n    >>> plt.show()\n\n    '
    (freqs, _, Pxy) = _spectral_helper(x, y, fs, window, nperseg, noverlap, nfft, detrend, return_onesided, scaling, axis, mode='psd')
    if len(Pxy.shape) >= 2 and Pxy.size > 0:
        if Pxy.shape[-1] > 1:
            if average == 'median':
                bias = _median_bias(Pxy.shape[-1])
                if np.iscomplexobj(Pxy):
                    Pxy = np.median(np.real(Pxy), axis=-1) + 1j * np.median(np.imag(Pxy), axis=-1)
                else:
                    Pxy = np.median(Pxy, axis=-1)
                Pxy /= bias
            elif average == 'mean':
                Pxy = Pxy.mean(axis=-1)
            else:
                raise ValueError('average must be "median" or "mean", got %s' % (average,))
        else:
            Pxy = np.reshape(Pxy, Pxy.shape[:-1])
    return (freqs, Pxy)

def spectrogram(x, fs=1.0, window=('tukey', 0.25), nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='psd'):
    if False:
        return 10
    'Compute a spectrogram with consecutive Fourier transforms.\n\n    Spectrograms can be used as a way of visualizing the change of a\n    nonstationary signal\'s frequency content over time.\n\n    .. legacy:: function\n\n        :class:`ShortTimeFFT` is a newer STFT / ISTFT implementation with more\n        features also including a :meth:`~ShortTimeFFT.spectrogram` method.\n        A :ref:`comparison <tutorial_stft_legacy_stft>` between the\n        implementations can be found in the :ref:`tutorial_stft` section of\n        the :ref:`user_guide`.\n\n    Parameters\n    ----------\n    x : array_like\n        Time series of measurement values\n    fs : float, optional\n        Sampling frequency of the `x` time series. Defaults to 1.0.\n    window : str or tuple or array_like, optional\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg.\n        Defaults to a Tukey window with shape parameter of 0.25.\n    nperseg : int, optional\n        Length of each segment. Defaults to None, but if window is str or\n        tuple, is set to 256, and if window is array_like, is set to the\n        length of the window.\n    noverlap : int, optional\n        Number of points to overlap between segments. If `None`,\n        ``noverlap = nperseg // 8``. Defaults to `None`.\n    nfft : int, optional\n        Length of the FFT used, if a zero padded FFT is desired. If\n        `None`, the FFT length is `nperseg`. Defaults to `None`.\n    detrend : str or function or `False`, optional\n        Specifies how to detrend each segment. If `detrend` is a\n        string, it is passed as the `type` argument to the `detrend`\n        function. If it is a function, it takes a segment and returns a\n        detrended segment. If `detrend` is `False`, no detrending is\n        done. Defaults to \'constant\'.\n    return_onesided : bool, optional\n        If `True`, return a one-sided spectrum for real data. If\n        `False` return a two-sided spectrum. Defaults to `True`, but for\n        complex data, a two-sided spectrum is always returned.\n    scaling : { \'density\', \'spectrum\' }, optional\n        Selects between computing the power spectral density (\'density\')\n        where `Sxx` has units of V**2/Hz and computing the power\n        spectrum (\'spectrum\') where `Sxx` has units of V**2, if `x`\n        is measured in V and `fs` is measured in Hz. Defaults to\n        \'density\'.\n    axis : int, optional\n        Axis along which the spectrogram is computed; the default is over\n        the last axis (i.e. ``axis=-1``).\n    mode : str, optional\n        Defines what kind of return values are expected. Options are\n        [\'psd\', \'complex\', \'magnitude\', \'angle\', \'phase\']. \'complex\' is\n        equivalent to the output of `stft` with no padding or boundary\n        extension. \'magnitude\' returns the absolute magnitude of the\n        STFT. \'angle\' and \'phase\' return the complex angle of the STFT,\n        with and without unwrapping, respectively.\n\n    Returns\n    -------\n    f : ndarray\n        Array of sample frequencies.\n    t : ndarray\n        Array of segment times.\n    Sxx : ndarray\n        Spectrogram of x. By default, the last axis of Sxx corresponds\n        to the segment times.\n\n    See Also\n    --------\n    periodogram: Simple, optionally modified periodogram\n    lombscargle: Lomb-Scargle periodogram for unevenly sampled data\n    welch: Power spectral density by Welch\'s method.\n    csd: Cross spectral density by Welch\'s method.\n    ShortTimeFFT: Newer STFT/ISTFT implementation providing more features,\n                  which also includes a :meth:`~ShortTimeFFT.spectrogram`\n                  method.\n\n    Notes\n    -----\n    An appropriate amount of overlap will depend on the choice of window\n    and on your requirements. In contrast to welch\'s method, where the\n    entire data stream is averaged over, one may wish to use a smaller\n    overlap (or perhaps none at all) when computing a spectrogram, to\n    maintain some statistical independence between individual segments.\n    It is for this reason that the default window is a Tukey window with\n    1/8th of a window\'s length overlap at each end.\n\n\n    .. versionadded:: 0.16.0\n\n    References\n    ----------\n    .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck\n           "Discrete-Time Signal Processing", Prentice Hall, 1999.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy import signal\n    >>> from scipy.fft import fftshift\n    >>> import matplotlib.pyplot as plt\n    >>> rng = np.random.default_rng()\n\n    Generate a test signal, a 2 Vrms sine wave whose frequency is slowly\n    modulated around 3kHz, corrupted by white noise of exponentially\n    decreasing magnitude sampled at 10 kHz.\n\n    >>> fs = 10e3\n    >>> N = 1e5\n    >>> amp = 2 * np.sqrt(2)\n    >>> noise_power = 0.01 * fs / 2\n    >>> time = np.arange(N) / float(fs)\n    >>> mod = 500*np.cos(2*np.pi*0.25*time)\n    >>> carrier = amp * np.sin(2*np.pi*3e3*time + mod)\n    >>> noise = rng.normal(scale=np.sqrt(noise_power), size=time.shape)\n    >>> noise *= np.exp(-time/5)\n    >>> x = carrier + noise\n\n    Compute and plot the spectrogram.\n\n    >>> f, t, Sxx = signal.spectrogram(x, fs)\n    >>> plt.pcolormesh(t, f, Sxx, shading=\'gouraud\')\n    >>> plt.ylabel(\'Frequency [Hz]\')\n    >>> plt.xlabel(\'Time [sec]\')\n    >>> plt.show()\n\n    Note, if using output that is not one sided, then use the following:\n\n    >>> f, t, Sxx = signal.spectrogram(x, fs, return_onesided=False)\n    >>> plt.pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0), shading=\'gouraud\')\n    >>> plt.ylabel(\'Frequency [Hz]\')\n    >>> plt.xlabel(\'Time [sec]\')\n    >>> plt.show()\n\n    '
    modelist = ['psd', 'complex', 'magnitude', 'angle', 'phase']
    if mode not in modelist:
        raise ValueError('unknown value for mode {}, must be one of {}'.format(mode, modelist))
    (window, nperseg) = _triage_segments(window, nperseg, input_length=x.shape[axis])
    if noverlap is None:
        noverlap = nperseg // 8
    if mode == 'psd':
        (freqs, time, Sxx) = _spectral_helper(x, x, fs, window, nperseg, noverlap, nfft, detrend, return_onesided, scaling, axis, mode='psd')
    else:
        (freqs, time, Sxx) = _spectral_helper(x, x, fs, window, nperseg, noverlap, nfft, detrend, return_onesided, scaling, axis, mode='stft')
        if mode == 'magnitude':
            Sxx = np.abs(Sxx)
        elif mode in ['angle', 'phase']:
            Sxx = np.angle(Sxx)
            if mode == 'phase':
                if axis < 0:
                    axis -= 1
                Sxx = np.unwrap(Sxx, axis=axis)
    return (freqs, time, Sxx)

def check_COLA(window, nperseg, noverlap, tol=1e-10):
    if False:
        print('Hello World!')
    'Check whether the Constant OverLap Add (COLA) constraint is met.\n\n    Parameters\n    ----------\n    window : str or tuple or array_like\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg.\n    nperseg : int\n        Length of each segment.\n    noverlap : int\n        Number of points to overlap between segments.\n    tol : float, optional\n        The allowed variance of a bin\'s weighted sum from the median bin\n        sum.\n\n    Returns\n    -------\n    verdict : bool\n        `True` if chosen combination satisfies COLA within `tol`,\n        `False` otherwise\n\n    See Also\n    --------\n    check_NOLA: Check whether the Nonzero Overlap Add (NOLA) constraint is met\n    stft: Short Time Fourier Transform\n    istft: Inverse Short Time Fourier Transform\n\n    Notes\n    -----\n    In order to enable inversion of an STFT via the inverse STFT in\n    `istft`, it is sufficient that the signal windowing obeys the constraint of\n    "Constant OverLap Add" (COLA). This ensures that every point in the input\n    data is equally weighted, thereby avoiding aliasing and allowing full\n    reconstruction.\n\n    Some examples of windows that satisfy COLA:\n        - Rectangular window at overlap of 0, 1/2, 2/3, 3/4, ...\n        - Bartlett window at overlap of 1/2, 3/4, 5/6, ...\n        - Hann window at 1/2, 2/3, 3/4, ...\n        - Any Blackman family window at 2/3 overlap\n        - Any window with ``noverlap = nperseg-1``\n\n    A very comprehensive list of other windows may be found in [2]_,\n    wherein the COLA condition is satisfied when the "Amplitude\n    Flatness" is unity.\n\n    .. versionadded:: 0.19.0\n\n    References\n    ----------\n    .. [1] Julius O. Smith III, "Spectral Audio Signal Processing", W3K\n           Publishing, 2011,ISBN 978-0-9745607-3-1.\n    .. [2] G. Heinzel, A. Ruediger and R. Schilling, "Spectrum and\n           spectral density estimation by the Discrete Fourier transform\n           (DFT), including a comprehensive list of window functions and\n           some new at-top windows", 2002,\n           http://hdl.handle.net/11858/00-001M-0000-0013-557A-5\n\n    Examples\n    --------\n    >>> from scipy import signal\n\n    Confirm COLA condition for rectangular window of 75% (3/4) overlap:\n\n    >>> signal.check_COLA(signal.windows.boxcar(100), 100, 75)\n    True\n\n    COLA is not true for 25% (1/4) overlap, though:\n\n    >>> signal.check_COLA(signal.windows.boxcar(100), 100, 25)\n    False\n\n    "Symmetrical" Hann window (for filter design) is not COLA:\n\n    >>> signal.check_COLA(signal.windows.hann(120, sym=True), 120, 60)\n    False\n\n    "Periodic" or "DFT-even" Hann window (for FFT analysis) is COLA for\n    overlap of 1/2, 2/3, 3/4, etc.:\n\n    >>> signal.check_COLA(signal.windows.hann(120, sym=False), 120, 60)\n    True\n\n    >>> signal.check_COLA(signal.windows.hann(120, sym=False), 120, 80)\n    True\n\n    >>> signal.check_COLA(signal.windows.hann(120, sym=False), 120, 90)\n    True\n\n    '
    nperseg = int(nperseg)
    if nperseg < 1:
        raise ValueError('nperseg must be a positive integer')
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    noverlap = int(noverlap)
    if isinstance(window, str) or type(window) is tuple:
        win = get_window(window, nperseg)
    else:
        win = np.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if win.shape[0] != nperseg:
            raise ValueError('window must have length of nperseg')
    step = nperseg - noverlap
    binsums = sum((win[ii * step:(ii + 1) * step] for ii in range(nperseg // step)))
    if nperseg % step != 0:
        binsums[:nperseg % step] += win[-(nperseg % step):]
    deviation = binsums - np.median(binsums)
    return np.max(np.abs(deviation)) < tol

def check_NOLA(window, nperseg, noverlap, tol=1e-10):
    if False:
        return 10
    'Check whether the Nonzero Overlap Add (NOLA) constraint is met.\n\n    Parameters\n    ----------\n    window : str or tuple or array_like\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg.\n    nperseg : int\n        Length of each segment.\n    noverlap : int\n        Number of points to overlap between segments.\n    tol : float, optional\n        The allowed variance of a bin\'s weighted sum from the median bin\n        sum.\n\n    Returns\n    -------\n    verdict : bool\n        `True` if chosen combination satisfies the NOLA constraint within\n        `tol`, `False` otherwise\n\n    See Also\n    --------\n    check_COLA: Check whether the Constant OverLap Add (COLA) constraint is met\n    stft: Short Time Fourier Transform\n    istft: Inverse Short Time Fourier Transform\n\n    Notes\n    -----\n    In order to enable inversion of an STFT via the inverse STFT in\n    `istft`, the signal windowing must obey the constraint of "nonzero\n    overlap add" (NOLA):\n\n    .. math:: \\sum_{t}w^{2}[n-tH] \\ne 0\n\n    for all :math:`n`, where :math:`w` is the window function, :math:`t` is the\n    frame index, and :math:`H` is the hop size (:math:`H` = `nperseg` -\n    `noverlap`).\n\n    This ensures that the normalization factors in the denominator of the\n    overlap-add inversion equation are not zero. Only very pathological windows\n    will fail the NOLA constraint.\n\n    .. versionadded:: 1.2.0\n\n    References\n    ----------\n    .. [1] Julius O. Smith III, "Spectral Audio Signal Processing", W3K\n           Publishing, 2011,ISBN 978-0-9745607-3-1.\n    .. [2] G. Heinzel, A. Ruediger and R. Schilling, "Spectrum and\n           spectral density estimation by the Discrete Fourier transform\n           (DFT), including a comprehensive list of window functions and\n           some new at-top windows", 2002,\n           http://hdl.handle.net/11858/00-001M-0000-0013-557A-5\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy import signal\n\n    Confirm NOLA condition for rectangular window of 75% (3/4) overlap:\n\n    >>> signal.check_NOLA(signal.windows.boxcar(100), 100, 75)\n    True\n\n    NOLA is also true for 25% (1/4) overlap:\n\n    >>> signal.check_NOLA(signal.windows.boxcar(100), 100, 25)\n    True\n\n    "Symmetrical" Hann window (for filter design) is also NOLA:\n\n    >>> signal.check_NOLA(signal.windows.hann(120, sym=True), 120, 60)\n    True\n\n    As long as there is overlap, it takes quite a pathological window to fail\n    NOLA:\n\n    >>> w = np.ones(64, dtype="float")\n    >>> w[::2] = 0\n    >>> signal.check_NOLA(w, 64, 32)\n    False\n\n    If there is not enough overlap, a window with zeros at the ends will not\n    work:\n\n    >>> signal.check_NOLA(signal.windows.hann(64), 64, 0)\n    False\n    >>> signal.check_NOLA(signal.windows.hann(64), 64, 1)\n    False\n    >>> signal.check_NOLA(signal.windows.hann(64), 64, 2)\n    True\n\n    '
    nperseg = int(nperseg)
    if nperseg < 1:
        raise ValueError('nperseg must be a positive integer')
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg')
    if noverlap < 0:
        raise ValueError('noverlap must be a nonnegative integer')
    noverlap = int(noverlap)
    if isinstance(window, str) or type(window) is tuple:
        win = get_window(window, nperseg)
    else:
        win = np.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if win.shape[0] != nperseg:
            raise ValueError('window must have length of nperseg')
    step = nperseg - noverlap
    binsums = sum((win[ii * step:(ii + 1) * step] ** 2 for ii in range(nperseg // step)))
    if nperseg % step != 0:
        binsums[:nperseg % step] += win[-(nperseg % step):] ** 2
    return np.min(binsums) > tol

def stft(x, fs=1.0, window='hann', nperseg=256, noverlap=None, nfft=None, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=-1, scaling='spectrum'):
    if False:
        for i in range(10):
            print('nop')
    'Compute the Short Time Fourier Transform (STFT).\n\n    STFTs can be used as a way of quantifying the change of a\n    nonstationary signal\'s frequency and phase content over time.\n\n    .. legacy:: function\n\n        `ShortTimeFFT` is a newer STFT / ISTFT implementation with more\n        features. A :ref:`comparison <tutorial_stft_legacy_stft>` between the\n        implementations can be found in the :ref:`tutorial_stft` section of the\n        :ref:`user_guide`.\n\n    Parameters\n    ----------\n    x : array_like\n        Time series of measurement values\n    fs : float, optional\n        Sampling frequency of the `x` time series. Defaults to 1.0.\n    window : str or tuple or array_like, optional\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg. Defaults\n        to a Hann window.\n    nperseg : int, optional\n        Length of each segment. Defaults to 256.\n    noverlap : int, optional\n        Number of points to overlap between segments. If `None`,\n        ``noverlap = nperseg // 2``. Defaults to `None`. When\n        specified, the COLA constraint must be met (see Notes below).\n    nfft : int, optional\n        Length of the FFT used, if a zero padded FFT is desired. If\n        `None`, the FFT length is `nperseg`. Defaults to `None`.\n    detrend : str or function or `False`, optional\n        Specifies how to detrend each segment. If `detrend` is a\n        string, it is passed as the `type` argument to the `detrend`\n        function. If it is a function, it takes a segment and returns a\n        detrended segment. If `detrend` is `False`, no detrending is\n        done. Defaults to `False`.\n    return_onesided : bool, optional\n        If `True`, return a one-sided spectrum for real data. If\n        `False` return a two-sided spectrum. Defaults to `True`, but for\n        complex data, a two-sided spectrum is always returned.\n    boundary : str or None, optional\n        Specifies whether the input signal is extended at both ends, and\n        how to generate the new values, in order to center the first\n        windowed segment on the first input point. This has the benefit\n        of enabling reconstruction of the first input point when the\n        employed window function starts at zero. Valid options are\n        ``[\'even\', \'odd\', \'constant\', \'zeros\', None]``. Defaults to\n        \'zeros\', for zero padding extension. I.e. ``[1, 2, 3, 4]`` is\n        extended to ``[0, 1, 2, 3, 4, 0]`` for ``nperseg=3``.\n    padded : bool, optional\n        Specifies whether the input signal is zero-padded at the end to\n        make the signal fit exactly into an integer number of window\n        segments, so that all of the signal is included in the output.\n        Defaults to `True`. Padding occurs after boundary extension, if\n        `boundary` is not `None`, and `padded` is `True`, as is the\n        default.\n    axis : int, optional\n        Axis along which the STFT is computed; the default is over the\n        last axis (i.e. ``axis=-1``).\n    scaling: {\'spectrum\', \'psd\'}\n        The default \'spectrum\' scaling allows each frequency line of `Zxx` to\n        be interpreted as a magnitude spectrum. The \'psd\' option scales each\n        line to a power spectral density - it allows to calculate the signal\'s\n        energy by numerically integrating over ``abs(Zxx)**2``.\n\n        .. versionadded:: 1.9.0\n\n    Returns\n    -------\n    f : ndarray\n        Array of sample frequencies.\n    t : ndarray\n        Array of segment times.\n    Zxx : ndarray\n        STFT of `x`. By default, the last axis of `Zxx` corresponds\n        to the segment times.\n\n    See Also\n    --------\n    istft: Inverse Short Time Fourier Transform\n    ShortTimeFFT: Newer STFT/ISTFT implementation providing more features.\n    check_COLA: Check whether the Constant OverLap Add (COLA) constraint\n                is met\n    check_NOLA: Check whether the Nonzero Overlap Add (NOLA) constraint is met\n    welch: Power spectral density by Welch\'s method.\n    spectrogram: Spectrogram by Welch\'s method.\n    csd: Cross spectral density by Welch\'s method.\n    lombscargle: Lomb-Scargle periodogram for unevenly sampled data\n\n    Notes\n    -----\n    In order to enable inversion of an STFT via the inverse STFT in\n    `istft`, the signal windowing must obey the constraint of "Nonzero\n    OverLap Add" (NOLA), and the input signal must have complete\n    windowing coverage (i.e. ``(x.shape[axis] - nperseg) %\n    (nperseg-noverlap) == 0``). The `padded` argument may be used to\n    accomplish this.\n\n    Given a time-domain signal :math:`x[n]`, a window :math:`w[n]`, and a hop\n    size :math:`H` = `nperseg - noverlap`, the windowed frame at time index\n    :math:`t` is given by\n\n    .. math:: x_{t}[n]=x[n]w[n-tH]\n\n    The overlap-add (OLA) reconstruction equation is given by\n\n    .. math:: x[n]=\\frac{\\sum_{t}x_{t}[n]w[n-tH]}{\\sum_{t}w^{2}[n-tH]}\n\n    The NOLA constraint ensures that every normalization term that appears\n    in the denomimator of the OLA reconstruction equation is nonzero. Whether a\n    choice of `window`, `nperseg`, and `noverlap` satisfy this constraint can\n    be tested with `check_NOLA`.\n\n\n    .. versionadded:: 0.19.0\n\n    References\n    ----------\n    .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck\n           "Discrete-Time Signal Processing", Prentice Hall, 1999.\n    .. [2] Daniel W. Griffin, Jae S. Lim "Signal Estimation from\n           Modified Short-Time Fourier Transform", IEEE 1984,\n           10.1109/TASSP.1984.1164317\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy import signal\n    >>> import matplotlib.pyplot as plt\n    >>> rng = np.random.default_rng()\n\n    Generate a test signal, a 2 Vrms sine wave whose frequency is slowly\n    modulated around 3kHz, corrupted by white noise of exponentially\n    decreasing magnitude sampled at 10 kHz.\n\n    >>> fs = 10e3\n    >>> N = 1e5\n    >>> amp = 2 * np.sqrt(2)\n    >>> noise_power = 0.01 * fs / 2\n    >>> time = np.arange(N) / float(fs)\n    >>> mod = 500*np.cos(2*np.pi*0.25*time)\n    >>> carrier = amp * np.sin(2*np.pi*3e3*time + mod)\n    >>> noise = rng.normal(scale=np.sqrt(noise_power),\n    ...                    size=time.shape)\n    >>> noise *= np.exp(-time/5)\n    >>> x = carrier + noise\n\n    Compute and plot the STFT\'s magnitude.\n\n    >>> f, t, Zxx = signal.stft(x, fs, nperseg=1000)\n    >>> plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading=\'gouraud\')\n    >>> plt.title(\'STFT Magnitude\')\n    >>> plt.ylabel(\'Frequency [Hz]\')\n    >>> plt.xlabel(\'Time [sec]\')\n    >>> plt.show()\n\n    Compare the energy of the signal `x` with the energy of its STFT:\n\n    >>> E_x = sum(x**2) / fs  # Energy of x\n    >>> # Calculate a two-sided STFT with PSD scaling:\n    >>> f, t, Zxx = signal.stft(x, fs, nperseg=1000, return_onesided=False,\n    ...                         scaling=\'psd\')\n    >>> # Integrate numerically over abs(Zxx)**2:\n    >>> df, dt = f[1] - f[0], t[1] - t[0]\n    >>> E_Zxx = sum(np.sum(Zxx.real**2 + Zxx.imag**2, axis=0) * df) * dt\n    >>> # The energy is the same, but the numerical errors are quite large:\n    >>> np.isclose(E_x, E_Zxx, rtol=1e-2)\n    True\n\n    '
    if scaling == 'psd':
        scaling = 'density'
    elif scaling != 'spectrum':
        raise ValueError(f"Parameter scaling={scaling!r} not in ['spectrum', 'psd']!")
    (freqs, time, Zxx) = _spectral_helper(x, x, fs, window, nperseg, noverlap, nfft, detrend, return_onesided, scaling=scaling, axis=axis, mode='stft', boundary=boundary, padded=padded)
    return (freqs, time, Zxx)

def istft(Zxx, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, input_onesided=True, boundary=True, time_axis=-1, freq_axis=-2, scaling='spectrum'):
    if False:
        i = 10
        return i + 15
    'Perform the inverse Short Time Fourier transform (iSTFT).\n\n    .. legacy:: function\n\n        `ShortTimeFFT` is a newer STFT / ISTFT implementation with more\n        features. A :ref:`comparison <tutorial_stft_legacy_stft>` between the\n        implementations can be found in the :ref:`tutorial_stft` section of the\n        :ref:`user_guide`.\n\n    Parameters\n    ----------\n    Zxx : array_like\n        STFT of the signal to be reconstructed. If a purely real array\n        is passed, it will be cast to a complex data type.\n    fs : float, optional\n        Sampling frequency of the time series. Defaults to 1.0.\n    window : str or tuple or array_like, optional\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg. Defaults\n        to a Hann window. Must match the window used to generate the\n        STFT for faithful inversion.\n    nperseg : int, optional\n        Number of data points corresponding to each STFT segment. This\n        parameter must be specified if the number of data points per\n        segment is odd, or if the STFT was padded via ``nfft >\n        nperseg``. If `None`, the value depends on the shape of\n        `Zxx` and `input_onesided`. If `input_onesided` is `True`,\n        ``nperseg=2*(Zxx.shape[freq_axis] - 1)``. Otherwise,\n        ``nperseg=Zxx.shape[freq_axis]``. Defaults to `None`.\n    noverlap : int, optional\n        Number of points to overlap between segments. If `None`, half\n        of the segment length. Defaults to `None`. When specified, the\n        COLA constraint must be met (see Notes below), and should match\n        the parameter used to generate the STFT. Defaults to `None`.\n    nfft : int, optional\n        Number of FFT points corresponding to each STFT segment. This\n        parameter must be specified if the STFT was padded via ``nfft >\n        nperseg``. If `None`, the default values are the same as for\n        `nperseg`, detailed above, with one exception: if\n        `input_onesided` is True and\n        ``nperseg==2*Zxx.shape[freq_axis] - 1``, `nfft` also takes on\n        that value. This case allows the proper inversion of an\n        odd-length unpadded STFT using ``nfft=None``. Defaults to\n        `None`.\n    input_onesided : bool, optional\n        If `True`, interpret the input array as one-sided FFTs, such\n        as is returned by `stft` with ``return_onesided=True`` and\n        `numpy.fft.rfft`. If `False`, interpret the input as a a\n        two-sided FFT. Defaults to `True`.\n    boundary : bool, optional\n        Specifies whether the input signal was extended at its\n        boundaries by supplying a non-`None` ``boundary`` argument to\n        `stft`. Defaults to `True`.\n    time_axis : int, optional\n        Where the time segments of the STFT is located; the default is\n        the last axis (i.e. ``axis=-1``).\n    freq_axis : int, optional\n        Where the frequency axis of the STFT is located; the default is\n        the penultimate axis (i.e. ``axis=-2``).\n    scaling: {\'spectrum\', \'psd\'}\n        The default \'spectrum\' scaling allows each frequency line of `Zxx` to\n        be interpreted as a magnitude spectrum. The \'psd\' option scales each\n        line to a power spectral density - it allows to calculate the signal\'s\n        energy by numerically integrating over ``abs(Zxx)**2``.\n\n    Returns\n    -------\n    t : ndarray\n        Array of output data times.\n    x : ndarray\n        iSTFT of `Zxx`.\n\n    See Also\n    --------\n    stft: Short Time Fourier Transform\n    ShortTimeFFT: Newer STFT/ISTFT implementation providing more features.\n    check_COLA: Check whether the Constant OverLap Add (COLA) constraint\n                is met\n    check_NOLA: Check whether the Nonzero Overlap Add (NOLA) constraint is met\n\n    Notes\n    -----\n    In order to enable inversion of an STFT via the inverse STFT with\n    `istft`, the signal windowing must obey the constraint of "nonzero\n    overlap add" (NOLA):\n\n    .. math:: \\sum_{t}w^{2}[n-tH] \\ne 0\n\n    This ensures that the normalization factors that appear in the denominator\n    of the overlap-add reconstruction equation\n\n    .. math:: x[n]=\\frac{\\sum_{t}x_{t}[n]w[n-tH]}{\\sum_{t}w^{2}[n-tH]}\n\n    are not zero. The NOLA constraint can be checked with the `check_NOLA`\n    function.\n\n    An STFT which has been modified (via masking or otherwise) is not\n    guaranteed to correspond to a exactly realizible signal. This\n    function implements the iSTFT via the least-squares estimation\n    algorithm detailed in [2]_, which produces a signal that minimizes\n    the mean squared error between the STFT of the returned signal and\n    the modified STFT.\n\n\n    .. versionadded:: 0.19.0\n\n    References\n    ----------\n    .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck\n           "Discrete-Time Signal Processing", Prentice Hall, 1999.\n    .. [2] Daniel W. Griffin, Jae S. Lim "Signal Estimation from\n           Modified Short-Time Fourier Transform", IEEE 1984,\n           10.1109/TASSP.1984.1164317\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy import signal\n    >>> import matplotlib.pyplot as plt\n    >>> rng = np.random.default_rng()\n\n    Generate a test signal, a 2 Vrms sine wave at 50Hz corrupted by\n    0.001 V**2/Hz of white noise sampled at 1024 Hz.\n\n    >>> fs = 1024\n    >>> N = 10*fs\n    >>> nperseg = 512\n    >>> amp = 2 * np.sqrt(2)\n    >>> noise_power = 0.001 * fs / 2\n    >>> time = np.arange(N) / float(fs)\n    >>> carrier = amp * np.sin(2*np.pi*50*time)\n    >>> noise = rng.normal(scale=np.sqrt(noise_power),\n    ...                    size=time.shape)\n    >>> x = carrier + noise\n\n    Compute the STFT, and plot its magnitude\n\n    >>> f, t, Zxx = signal.stft(x, fs=fs, nperseg=nperseg)\n    >>> plt.figure()\n    >>> plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading=\'gouraud\')\n    >>> plt.ylim([f[1], f[-1]])\n    >>> plt.title(\'STFT Magnitude\')\n    >>> plt.ylabel(\'Frequency [Hz]\')\n    >>> plt.xlabel(\'Time [sec]\')\n    >>> plt.yscale(\'log\')\n    >>> plt.show()\n\n    Zero the components that are 10% or less of the carrier magnitude,\n    then convert back to a time series via inverse STFT\n\n    >>> Zxx = np.where(np.abs(Zxx) >= amp/10, Zxx, 0)\n    >>> _, xrec = signal.istft(Zxx, fs)\n\n    Compare the cleaned signal with the original and true carrier signals.\n\n    >>> plt.figure()\n    >>> plt.plot(time, x, time, xrec, time, carrier)\n    >>> plt.xlim([2, 2.1])\n    >>> plt.xlabel(\'Time [sec]\')\n    >>> plt.ylabel(\'Signal\')\n    >>> plt.legend([\'Carrier + Noise\', \'Filtered via STFT\', \'True Carrier\'])\n    >>> plt.show()\n\n    Note that the cleaned signal does not start as abruptly as the original,\n    since some of the coefficients of the transient were also removed:\n\n    >>> plt.figure()\n    >>> plt.plot(time, x, time, xrec, time, carrier)\n    >>> plt.xlim([0, 0.1])\n    >>> plt.xlabel(\'Time [sec]\')\n    >>> plt.ylabel(\'Signal\')\n    >>> plt.legend([\'Carrier + Noise\', \'Filtered via STFT\', \'True Carrier\'])\n    >>> plt.show()\n\n    '
    Zxx = np.asarray(Zxx) + 0j
    freq_axis = int(freq_axis)
    time_axis = int(time_axis)
    if Zxx.ndim < 2:
        raise ValueError('Input stft must be at least 2d!')
    if freq_axis == time_axis:
        raise ValueError('Must specify differing time and frequency axes!')
    nseg = Zxx.shape[time_axis]
    if input_onesided:
        n_default = 2 * (Zxx.shape[freq_axis] - 1)
    else:
        n_default = Zxx.shape[freq_axis]
    if nperseg is None:
        nperseg = n_default
    else:
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')
    if nfft is None:
        if input_onesided and nperseg == n_default + 1:
            nfft = nperseg
        else:
            nfft = n_default
    elif nfft < nperseg:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    else:
        nfft = int(nfft)
    if noverlap is None:
        noverlap = nperseg // 2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    nstep = nperseg - noverlap
    if time_axis != Zxx.ndim - 1 or freq_axis != Zxx.ndim - 2:
        if freq_axis < 0:
            freq_axis = Zxx.ndim + freq_axis
        if time_axis < 0:
            time_axis = Zxx.ndim + time_axis
        zouter = list(range(Zxx.ndim))
        for ax in sorted([time_axis, freq_axis], reverse=True):
            zouter.pop(ax)
        Zxx = np.transpose(Zxx, zouter + [freq_axis, time_axis])
    if isinstance(window, str) or type(window) is tuple:
        win = get_window(window, nperseg)
    else:
        win = np.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if win.shape[0] != nperseg:
            raise ValueError(f'window must have length of {nperseg}')
    ifunc = sp_fft.irfft if input_onesided else sp_fft.ifft
    xsubs = ifunc(Zxx, axis=-2, n=nfft)[..., :nperseg, :]
    outputlength = nperseg + (nseg - 1) * nstep
    x = np.zeros(list(Zxx.shape[:-2]) + [outputlength], dtype=xsubs.dtype)
    norm = np.zeros(outputlength, dtype=xsubs.dtype)
    if np.result_type(win, xsubs) != xsubs.dtype:
        win = win.astype(xsubs.dtype)
    if scaling == 'spectrum':
        xsubs *= win.sum()
    elif scaling == 'psd':
        xsubs *= np.sqrt(fs * sum(win ** 2))
    else:
        raise ValueError(f"Parameter scaling={scaling!r} not in ['spectrum', 'psd']!")
    for ii in range(nseg):
        x[..., ii * nstep:ii * nstep + nperseg] += xsubs[..., ii] * win
        norm[..., ii * nstep:ii * nstep + nperseg] += win ** 2
    if boundary:
        x = x[..., nperseg // 2:-(nperseg // 2)]
        norm = norm[..., nperseg // 2:-(nperseg // 2)]
    if np.sum(norm > 1e-10) != len(norm):
        warnings.warn('NOLA condition failed, STFT may not be invertible.' + (' Possibly due to missing boundary' if not boundary else ''))
    x /= np.where(norm > 1e-10, norm, 1.0)
    if input_onesided:
        x = x.real
    if x.ndim > 1:
        if time_axis != Zxx.ndim - 1:
            if freq_axis < time_axis:
                time_axis -= 1
            x = np.moveaxis(x, -1, time_axis)
    time = np.arange(x.shape[0]) / float(fs)
    return (time, x)

def coherence(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', axis=-1):
    if False:
        for i in range(10):
            print('nop')
    '\n    Estimate the magnitude squared coherence estimate, Cxy, of\n    discrete-time signals X and Y using Welch\'s method.\n\n    ``Cxy = abs(Pxy)**2/(Pxx*Pyy)``, where `Pxx` and `Pyy` are power\n    spectral density estimates of X and Y, and `Pxy` is the cross\n    spectral density estimate of X and Y.\n\n    Parameters\n    ----------\n    x : array_like\n        Time series of measurement values\n    y : array_like\n        Time series of measurement values\n    fs : float, optional\n        Sampling frequency of the `x` and `y` time series. Defaults\n        to 1.0.\n    window : str or tuple or array_like, optional\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg. Defaults\n        to a Hann window.\n    nperseg : int, optional\n        Length of each segment. Defaults to None, but if window is str or\n        tuple, is set to 256, and if window is array_like, is set to the\n        length of the window.\n    noverlap: int, optional\n        Number of points to overlap between segments. If `None`,\n        ``noverlap = nperseg // 2``. Defaults to `None`.\n    nfft : int, optional\n        Length of the FFT used, if a zero padded FFT is desired. If\n        `None`, the FFT length is `nperseg`. Defaults to `None`.\n    detrend : str or function or `False`, optional\n        Specifies how to detrend each segment. If `detrend` is a\n        string, it is passed as the `type` argument to the `detrend`\n        function. If it is a function, it takes a segment and returns a\n        detrended segment. If `detrend` is `False`, no detrending is\n        done. Defaults to \'constant\'.\n    axis : int, optional\n        Axis along which the coherence is computed for both inputs; the\n        default is over the last axis (i.e. ``axis=-1``).\n\n    Returns\n    -------\n    f : ndarray\n        Array of sample frequencies.\n    Cxy : ndarray\n        Magnitude squared coherence of x and y.\n\n    See Also\n    --------\n    periodogram: Simple, optionally modified periodogram\n    lombscargle: Lomb-Scargle periodogram for unevenly sampled data\n    welch: Power spectral density by Welch\'s method.\n    csd: Cross spectral density by Welch\'s method.\n\n    Notes\n    -----\n    An appropriate amount of overlap will depend on the choice of window\n    and on your requirements. For the default Hann window an overlap of\n    50% is a reasonable trade off between accurately estimating the\n    signal power, while not over counting any of the data. Narrower\n    windows may require a larger overlap.\n\n    .. versionadded:: 0.16.0\n\n    References\n    ----------\n    .. [1] P. Welch, "The use of the fast Fourier transform for the\n           estimation of power spectra: A method based on time averaging\n           over short, modified periodograms", IEEE Trans. Audio\n           Electroacoust. vol. 15, pp. 70-73, 1967.\n    .. [2] Stoica, Petre, and Randolph Moses, "Spectral Analysis of\n           Signals" Prentice Hall, 2005\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy import signal\n    >>> import matplotlib.pyplot as plt\n    >>> rng = np.random.default_rng()\n\n    Generate two test signals with some common features.\n\n    >>> fs = 10e3\n    >>> N = 1e5\n    >>> amp = 20\n    >>> freq = 1234.0\n    >>> noise_power = 0.001 * fs / 2\n    >>> time = np.arange(N) / fs\n    >>> b, a = signal.butter(2, 0.25, \'low\')\n    >>> x = rng.normal(scale=np.sqrt(noise_power), size=time.shape)\n    >>> y = signal.lfilter(b, a, x)\n    >>> x += amp*np.sin(2*np.pi*freq*time)\n    >>> y += rng.normal(scale=0.1*np.sqrt(noise_power), size=time.shape)\n\n    Compute and plot the coherence.\n\n    >>> f, Cxy = signal.coherence(x, y, fs, nperseg=1024)\n    >>> plt.semilogy(f, Cxy)\n    >>> plt.xlabel(\'frequency [Hz]\')\n    >>> plt.ylabel(\'Coherence\')\n    >>> plt.show()\n\n    '
    (freqs, Pxx) = welch(x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=detrend, axis=axis)
    (_, Pyy) = welch(y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=detrend, axis=axis)
    (_, Pxy) = csd(x, y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=detrend, axis=axis)
    Cxy = np.abs(Pxy) ** 2 / Pxx / Pyy
    return (freqs, Cxy)

def _spectral_helper(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='psd', boundary=None, padded=False):
    if False:
        for i in range(10):
            print('nop')
    "Calculate various forms of windowed FFTs for PSD, CSD, etc.\n\n    This is a helper function that implements the commonality between\n    the stft, psd, csd, and spectrogram functions. It is not designed to\n    be called externally. The windows are not averaged over; the result\n    from each window is returned.\n\n    Parameters\n    ----------\n    x : array_like\n        Array or sequence containing the data to be analyzed.\n    y : array_like\n        Array or sequence containing the data to be analyzed. If this is\n        the same object in memory as `x` (i.e. ``_spectral_helper(x,\n        x, ...)``), the extra computations are spared.\n    fs : float, optional\n        Sampling frequency of the time series. Defaults to 1.0.\n    window : str or tuple or array_like, optional\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg. Defaults\n        to a Hann window.\n    nperseg : int, optional\n        Length of each segment. Defaults to None, but if window is str or\n        tuple, is set to 256, and if window is array_like, is set to the\n        length of the window.\n    noverlap : int, optional\n        Number of points to overlap between segments. If `None`,\n        ``noverlap = nperseg // 2``. Defaults to `None`.\n    nfft : int, optional\n        Length of the FFT used, if a zero padded FFT is desired. If\n        `None`, the FFT length is `nperseg`. Defaults to `None`.\n    detrend : str or function or `False`, optional\n        Specifies how to detrend each segment. If `detrend` is a\n        string, it is passed as the `type` argument to the `detrend`\n        function. If it is a function, it takes a segment and returns a\n        detrended segment. If `detrend` is `False`, no detrending is\n        done. Defaults to 'constant'.\n    return_onesided : bool, optional\n        If `True`, return a one-sided spectrum for real data. If\n        `False` return a two-sided spectrum. Defaults to `True`, but for\n        complex data, a two-sided spectrum is always returned.\n    scaling : { 'density', 'spectrum' }, optional\n        Selects between computing the cross spectral density ('density')\n        where `Pxy` has units of V**2/Hz and computing the cross\n        spectrum ('spectrum') where `Pxy` has units of V**2, if `x`\n        and `y` are measured in V and `fs` is measured in Hz.\n        Defaults to 'density'\n    axis : int, optional\n        Axis along which the FFTs are computed; the default is over the\n        last axis (i.e. ``axis=-1``).\n    mode: str {'psd', 'stft'}, optional\n        Defines what kind of return values are expected. Defaults to\n        'psd'.\n    boundary : str or None, optional\n        Specifies whether the input signal is extended at both ends, and\n        how to generate the new values, in order to center the first\n        windowed segment on the first input point. This has the benefit\n        of enabling reconstruction of the first input point when the\n        employed window function starts at zero. Valid options are\n        ``['even', 'odd', 'constant', 'zeros', None]``. Defaults to\n        `None`.\n    padded : bool, optional\n        Specifies whether the input signal is zero-padded at the end to\n        make the signal fit exactly into an integer number of window\n        segments, so that all of the signal is included in the output.\n        Defaults to `False`. Padding occurs after boundary extension, if\n        `boundary` is not `None`, and `padded` is `True`.\n\n    Returns\n    -------\n    freqs : ndarray\n        Array of sample frequencies.\n    t : ndarray\n        Array of times corresponding to each data segment\n    result : ndarray\n        Array of output data, contents dependent on *mode* kwarg.\n\n    Notes\n    -----\n    Adapted from matplotlib.mlab\n\n    .. versionadded:: 0.16.0\n    "
    if mode not in ['psd', 'stft']:
        raise ValueError("Unknown value for mode %s, must be one of: {'psd', 'stft'}" % mode)
    boundary_funcs = {'even': even_ext, 'odd': odd_ext, 'constant': const_ext, 'zeros': zero_ext, None: None}
    if boundary not in boundary_funcs:
        raise ValueError("Unknown boundary option '{}', must be one of: {}".format(boundary, list(boundary_funcs.keys())))
    same_data = y is x
    if not same_data and mode != 'psd':
        raise ValueError("x and y must be equal if mode is 'stft'")
    axis = int(axis)
    x = np.asarray(x)
    if not same_data:
        y = np.asarray(y)
        outdtype = np.result_type(x, y, np.complex64)
    else:
        outdtype = np.result_type(x, np.complex64)
    if not same_data:
        xouter = list(x.shape)
        youter = list(y.shape)
        xouter.pop(axis)
        youter.pop(axis)
        try:
            outershape = np.broadcast(np.empty(xouter), np.empty(youter)).shape
        except ValueError as e:
            raise ValueError('x and y cannot be broadcast together.') from e
    if same_data:
        if x.size == 0:
            return (np.empty(x.shape), np.empty(x.shape), np.empty(x.shape))
    elif x.size == 0 or y.size == 0:
        outshape = outershape + (min([x.shape[axis], y.shape[axis]]),)
        emptyout = np.moveaxis(np.empty(outshape), -1, axis)
        return (emptyout, emptyout, emptyout)
    if x.ndim > 1:
        if axis != -1:
            x = np.moveaxis(x, axis, -1)
            if not same_data and y.ndim > 1:
                y = np.moveaxis(y, axis, -1)
    if not same_data:
        if x.shape[-1] != y.shape[-1]:
            if x.shape[-1] < y.shape[-1]:
                pad_shape = list(x.shape)
                pad_shape[-1] = y.shape[-1] - x.shape[-1]
                x = np.concatenate((x, np.zeros(pad_shape)), -1)
            else:
                pad_shape = list(y.shape)
                pad_shape[-1] = x.shape[-1] - y.shape[-1]
                y = np.concatenate((y, np.zeros(pad_shape)), -1)
    if nperseg is not None:
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')
    (win, nperseg) = _triage_segments(window, nperseg, input_length=x.shape[-1])
    if nfft is None:
        nfft = nperseg
    elif nfft < nperseg:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    else:
        nfft = int(nfft)
    if noverlap is None:
        noverlap = nperseg // 2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    nstep = nperseg - noverlap
    if boundary is not None:
        ext_func = boundary_funcs[boundary]
        x = ext_func(x, nperseg // 2, axis=-1)
        if not same_data:
            y = ext_func(y, nperseg // 2, axis=-1)
    if padded:
        nadd = -(x.shape[-1] - nperseg) % nstep % nperseg
        zeros_shape = list(x.shape[:-1]) + [nadd]
        x = np.concatenate((x, np.zeros(zeros_shape)), axis=-1)
        if not same_data:
            zeros_shape = list(y.shape[:-1]) + [nadd]
            y = np.concatenate((y, np.zeros(zeros_shape)), axis=-1)
    if not detrend:

        def detrend_func(d):
            if False:
                for i in range(10):
                    print('nop')
            return d
    elif not hasattr(detrend, '__call__'):

        def detrend_func(d):
            if False:
                return 10
            return _signaltools.detrend(d, type=detrend, axis=-1)
    elif axis != -1:

        def detrend_func(d):
            if False:
                while True:
                    i = 10
            d = np.moveaxis(d, -1, axis)
            d = detrend(d)
            return np.moveaxis(d, axis, -1)
    else:
        detrend_func = detrend
    if np.result_type(win, np.complex64) != outdtype:
        win = win.astype(outdtype)
    if scaling == 'density':
        scale = 1.0 / (fs * (win * win).sum())
    elif scaling == 'spectrum':
        scale = 1.0 / win.sum() ** 2
    else:
        raise ValueError('Unknown scaling: %r' % scaling)
    if mode == 'stft':
        scale = np.sqrt(scale)
    if return_onesided:
        if np.iscomplexobj(x):
            sides = 'twosided'
            warnings.warn('Input data is complex, switching to return_onesided=False')
        else:
            sides = 'onesided'
            if not same_data:
                if np.iscomplexobj(y):
                    sides = 'twosided'
                    warnings.warn('Input data is complex, switching to return_onesided=False')
    else:
        sides = 'twosided'
    if sides == 'twosided':
        freqs = sp_fft.fftfreq(nfft, 1 / fs)
    elif sides == 'onesided':
        freqs = sp_fft.rfftfreq(nfft, 1 / fs)
    result = _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides)
    if not same_data:
        result_y = _fft_helper(y, win, detrend_func, nperseg, noverlap, nfft, sides)
        result = np.conjugate(result) * result_y
    elif mode == 'psd':
        result = np.conjugate(result) * result
    result *= scale
    if sides == 'onesided' and mode == 'psd':
        if nfft % 2:
            result[..., 1:] *= 2
        else:
            result[..., 1:-1] *= 2
    time = np.arange(nperseg / 2, x.shape[-1] - nperseg / 2 + 1, nperseg - noverlap) / float(fs)
    if boundary is not None:
        time -= nperseg / 2 / fs
    result = result.astype(outdtype)
    if same_data and mode != 'stft':
        result = result.real
    if axis < 0:
        axis -= 1
    result = np.moveaxis(result, -1, axis)
    return (freqs, time, result)

def _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculate windowed FFT, for internal use by\n    `scipy.signal._spectral_helper`.\n\n    This is a helper function that does the main FFT calculation for\n    `_spectral helper`. All input validation is performed there, and the\n    data axis is assumed to be the last axis of x. It is not designed to\n    be called externally. The windows are not averaged over; the result\n    from each window is returned.\n\n    Returns\n    -------\n    result : ndarray\n        Array of FFT data\n\n    Notes\n    -----\n    Adapted from matplotlib.mlab\n\n    .. versionadded:: 0.16.0\n    '
    if nperseg == 1 and noverlap == 0:
        result = x[..., np.newaxis]
    else:
        step = nperseg - noverlap
        shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, nperseg)
        strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    result = detrend_func(result)
    result = win * result
    if sides == 'twosided':
        func = sp_fft.fft
    else:
        result = result.real
        func = sp_fft.rfft
    result = func(result, n=nfft)
    return result

def _triage_segments(window, nperseg, input_length):
    if False:
        i = 10
        return i + 15
    '\n    Parses window and nperseg arguments for spectrogram and _spectral_helper.\n    This is a helper function, not meant to be called externally.\n\n    Parameters\n    ----------\n    window : string, tuple, or ndarray\n        If window is specified by a string or tuple and nperseg is not\n        specified, nperseg is set to the default of 256 and returns a window of\n        that length.\n        If instead the window is array_like and nperseg is not specified, then\n        nperseg is set to the length of the window. A ValueError is raised if\n        the user supplies both an array_like window and a value for nperseg but\n        nperseg does not equal the length of the window.\n\n    nperseg : int\n        Length of each segment\n\n    input_length: int\n        Length of input signal, i.e. x.shape[-1]. Used to test for errors.\n\n    Returns\n    -------\n    win : ndarray\n        window. If function was called with string or tuple than this will hold\n        the actual array used as a window.\n\n    nperseg : int\n        Length of each segment. If window is str or tuple, nperseg is set to\n        256. If window is array_like, nperseg is set to the length of the\n        window.\n    '
    if isinstance(window, str) or isinstance(window, tuple):
        if nperseg is None:
            nperseg = 256
        if nperseg > input_length:
            warnings.warn('nperseg = {0:d} is greater than input length  = {1:d}, using nperseg = {1:d}'.format(nperseg, input_length))
            nperseg = input_length
        win = get_window(window, nperseg)
    else:
        win = np.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if input_length < win.shape[-1]:
            raise ValueError('window is longer than input signal')
        if nperseg is None:
            nperseg = win.shape[0]
        elif nperseg is not None:
            if nperseg != win.shape[0]:
                raise ValueError('value specified for nperseg is different from length of window')
    return (win, nperseg)

def _median_bias(n):
    if False:
        while True:
            i = 10
    '\n    Returns the bias of the median of a set of periodograms relative to\n    the mean.\n\n    See Appendix B from [1]_ for details.\n\n    Parameters\n    ----------\n    n : int\n        Numbers of periodograms being averaged.\n\n    Returns\n    -------\n    bias : float\n        Calculated bias.\n\n    References\n    ----------\n    .. [1] B. Allen, W.G. Anderson, P.R. Brady, D.A. Brown, J.D.E. Creighton.\n           "FINDCHIRP: an algorithm for detection of gravitational waves from\n           inspiraling compact binaries", Physical Review D 85, 2012,\n           :arxiv:`gr-qc/0509116`\n    '
    ii_2 = 2 * np.arange(1.0, (n - 1) // 2 + 1)
    return 1 + np.sum(1.0 / (ii_2 + 1) - 1.0 / ii_2)