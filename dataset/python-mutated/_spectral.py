"""
Spectral analysis functions and utilities.

Some of the functions defined here were ported directly from CuSignal under
terms of the Apache license, under the following notice

Copyright (c) 2019-2020, NVIDIA CORPORATION.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import warnings
import cupy
from cupyx.scipy.signal.windows._windows import get_window
from cupyx.scipy.signal._spectral_impl import _lombscargle, _spectral_helper, _median_bias, _triage_segments

def lombscargle(x, y, freqs, precenter=False, normalize=False):
    if False:
        print('Hello World!')
    '\n    lombscargle(x, y, freqs)\n\n    Computes the Lomb-Scargle periodogram.\n\n    The Lomb-Scargle periodogram was developed by Lomb [1]_ and further\n    extended by Scargle [2]_ to find, and test the significance of weak\n    periodic signals with uneven temporal sampling.\n\n    When *normalize* is False (default) the computed periodogram\n    is unnormalized, it takes the value ``(A**2) * N/4`` for a harmonic\n    signal with amplitude A for sufficiently large N.\n\n    When *normalize* is True the computed periodogram is normalized by\n    the residuals of the data around a constant reference model (at zero).\n\n    Input arrays should be one-dimensional and will be cast to float64.\n\n    Parameters\n    ----------\n    x : array_like\n        Sample times.\n    y : array_like\n        Measurement values.\n    freqs : array_like\n        Angular frequencies for output periodogram.\n    precenter : bool, optional\n        Pre-center amplitudes by subtracting the mean.\n    normalize : bool, optional\n        Compute normalized periodogram.\n\n    Returns\n    -------\n    pgram : array_like\n        Lomb-Scargle periodogram.\n\n    Raises\n    ------\n    ValueError\n        If the input arrays `x` and `y` do not have the same shape.\n\n    Notes\n    -----\n    This subroutine calculates the periodogram using a slightly\n    modified algorithm due to Townsend [3]_ which allows the\n    periodogram to be calculated using only a single pass through\n    the input arrays for each frequency.\n    The algorithm running time scales roughly as O(x * freqs) or O(N^2)\n    for a large number of samples and frequencies.\n\n    References\n    ----------\n    .. [1] N.R. Lomb "Least-squares frequency analysis of unequally spaced\n           data", Astrophysics and Space Science, vol 39, pp. 447-462, 1976\n    .. [2] J.D. Scargle "Studies in astronomical time series analysis. II -\n           Statistical aspects of spectral analysis of unevenly spaced data",\n           The Astrophysical Journal, vol 263, pp. 835-853, 1982\n    .. [3] R.H.D. Townsend, "Fast calculation of the Lomb-Scargle\n           periodogram using graphics processing units.", The Astrophysical\n           Journal Supplement Series, vol 191, pp. 247-253, 2010\n\n    See Also\n    --------\n    istft: Inverse Short Time Fourier Transform\n    check_COLA: Check whether the Constant OverLap Add (COLA) constraint is met\n    welch: Power spectral density by Welch\'s method\n    spectrogram: Spectrogram by Welch\'s method\n    csd: Cross spectral density by Welch\'s method\n    '
    x = cupy.asarray(x, dtype=cupy.float64)
    y = cupy.asarray(y, dtype=cupy.float64)
    freqs = cupy.asarray(freqs, dtype=cupy.float64)
    pgram = cupy.empty(freqs.shape[0], dtype=cupy.float64)
    assert x.ndim == 1
    assert y.ndim == 1
    assert freqs.ndim == 1
    if x.shape[0] != y.shape[0]:
        raise ValueError('Input arrays do not have the same size.')
    y_dot = cupy.zeros(1, dtype=cupy.float64)
    if normalize:
        cupy.dot(y, y, out=y_dot)
    if precenter:
        y_in = y - y.mean()
    else:
        y_in = y
    _lombscargle(x, y_in, freqs, pgram, y_dot)
    return pgram

def periodogram(x, fs=1.0, window='boxcar', nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1):
    if False:
        return 10
    "\n    Estimate power spectral density using a periodogram.\n\n    Parameters\n    ----------\n    x : array_like\n        Time series of measurement values\n    fs : float, optional\n        Sampling frequency of the `x` time series. Defaults to 1.0.\n    window : str or tuple or array_like, optional\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg. Defaults\n        to 'boxcar'.\n    nfft : int, optional\n        Length of the FFT used. If `None` the length of `x` will be\n        used.\n    detrend : str or function or `False`, optional\n        Specifies how to detrend each segment. If `detrend` is a\n        string, it is passed as the `type` argument to the `detrend`\n        function. If it is a function, it takes a segment and returns a\n        detrended segment. If `detrend` is `False`, no detrending is\n        done. Defaults to 'constant'.\n    return_onesided : bool, optional\n        If `True`, return a one-sided spectrum for real data. If\n        `False` return a two-sided spectrum. Defaults to `True`, but for\n        complex data, a two-sided spectrum is always returned.\n    scaling : { 'density', 'spectrum' }, optional\n        Selects between computing the power spectral density ('density')\n        where `Pxx` has units of V**2/Hz and computing the power\n        spectrum ('spectrum') where `Pxx` has units of V**2, if `x`\n        is measured in V and `fs` is measured in Hz. Defaults to\n        'density'\n    axis : int, optional\n        Axis along which the periodogram is computed; the default is\n        over the last axis (i.e. ``axis=-1``).\n\n    Returns\n    -------\n    f : ndarray\n        Array of sample frequencies.\n    Pxx : ndarray\n        Power spectral density or power spectrum of `x`.\n\n    See Also\n    --------\n    welch: Estimate power spectral density using Welch's method\n    lombscargle: Lomb-Scargle periodogram for unevenly sampled data\n    "
    x = cupy.asarray(x)
    if x.size == 0:
        return (cupy.empty(x.shape), cupy.empty(x.shape))
    if window is None:
        window = 'boxcar'
    if nfft is None:
        nperseg = x.shape[axis]
    elif nfft == x.shape[axis]:
        nperseg = nfft
    elif nfft > x.shape[axis]:
        nperseg = x.shape[axis]
    elif nfft < x.shape[axis]:
        s = [cupy.s_[:]] * len(x.shape)
        s[axis] = cupy.s_[:nfft]
        x = cupy.asarray(x[tuple(s)])
        nperseg = nfft
        nfft = None
    return welch(x, fs=fs, window=window, nperseg=nperseg, noverlap=0, nfft=nfft, detrend=detrend, return_onesided=return_onesided, scaling=scaling, axis=axis)

def welch(x, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean'):
    if False:
        while True:
            i = 10
    '\n    Estimate power spectral density using Welch\'s method.\n\n    Welch\'s method [1]_ computes an estimate of the power spectral\n    density by dividing the data into overlapping segments, computing a\n    modified periodogram for each segment and averaging the\n    periodograms.\n\n    Parameters\n    ----------\n    x : array_like\n        Time series of measurement values\n    fs : float, optional\n        Sampling frequency of the `x` time series. Defaults to 1.0.\n    window : str or tuple or array_like, optional\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg. Defaults\n        to a Hann window.\n    nperseg : int, optional\n        Length of each segment. Defaults to None, but if window is str or\n        tuple, is set to 256, and if window is array_like, is set to the\n        length of the window.\n    noverlap : int, optional\n        Number of points to overlap between segments. If `None`,\n        ``noverlap = nperseg // 2``. Defaults to `None`.\n    nfft : int, optional\n        Length of the FFT used, if a zero padded FFT is desired. If\n        `None`, the FFT length is `nperseg`. Defaults to `None`.\n    detrend : str or function or `False`, optional\n        Specifies how to detrend each segment. If `detrend` is a\n        string, it is passed as the `type` argument to the `detrend`\n        function. If it is a function, it takes a segment and returns a\n        detrended segment. If `detrend` is `False`, no detrending is\n        done. Defaults to \'constant\'.\n    return_onesided : bool, optional\n        If `True`, return a one-sided spectrum for real data. If\n        `False` return a two-sided spectrum. Defaults to `True`, but for\n        complex data, a two-sided spectrum is always returned.\n    scaling : { \'density\', \'spectrum\' }, optional\n        Selects between computing the power spectral density (\'density\')\n        where `Pxx` has units of V**2/Hz and computing the power\n        spectrum (\'spectrum\') where `Pxx` has units of V**2, if `x`\n        is measured in V and `fs` is measured in Hz. Defaults to\n        \'density\'\n    axis : int, optional\n        Axis along which the periodogram is computed; the default is\n        over the last axis (i.e. ``axis=-1``).\n    average : { \'mean\', \'median\' }, optional\n        Method to use when averaging periodograms. Defaults to \'mean\'.\n\n\n    Returns\n    -------\n    f : ndarray\n        Array of sample frequencies.\n    Pxx : ndarray\n        Power spectral density or power spectrum of x.\n\n    See Also\n    --------\n    periodogram: Simple, optionally modified periodogram\n    lombscargle: Lomb-Scargle periodogram for unevenly sampled data\n\n    Notes\n    -----\n    An appropriate amount of overlap will depend on the choice of window\n    and on your requirements. For the default Hann window an overlap of\n    50% is a reasonable trade off between accurately estimating the\n    signal power, while not over counting any of the data. Narrower\n    windows may require a larger overlap.\n\n    If `noverlap` is 0, this method is equivalent to Bartlett\'s method\n    [2]_.\n\n    References\n    ----------\n    .. [1] P. Welch, "The use of the fast Fourier transform for the\n           estimation of power spectra: A method based on time averaging\n           over short, modified periodograms", IEEE Trans. Audio\n           Electroacoust. vol. 15, pp. 70-73, 1967.\n    .. [2] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",\n           Biometrika, vol. 37, pp. 1-16, 1950.\n    '
    (freqs, Pxx) = csd(x, x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=detrend, return_onesided=return_onesided, scaling=scaling, axis=axis, average=average)
    return (freqs, Pxx.real)

def csd(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean'):
    if False:
        print('Hello World!')
    "\n    Estimate the cross power spectral density, Pxy, using Welch's\n    method.\n\n    Parameters\n    ----------\n    x : array_like\n        Time series of measurement values\n    y : array_like\n        Time series of measurement values\n    fs : float, optional\n        Sampling frequency of the `x` and `y` time series. Defaults\n        to 1.0.\n    window : str or tuple or array_like, optional\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg. Defaults\n        to a Hann window.\n    nperseg : int, optional\n        Length of each segment. Defaults to None, but if window is str or\n        tuple, is set to 256, and if window is array_like, is set to the\n        length of the window.\n    noverlap: int, optional\n        Number of points to overlap between segments. If `None`,\n        ``noverlap = nperseg // 2``. Defaults to `None`.\n    nfft : int, optional\n        Length of the FFT used, if a zero padded FFT is desired. If\n        `None`, the FFT length is `nperseg`. Defaults to `None`.\n    detrend : str or function or `False`, optional\n        Specifies how to detrend each segment. If `detrend` is a\n        string, it is passed as the `type` argument to the `detrend`\n        function. If it is a function, it takes a segment and returns a\n        detrended segment. If `detrend` is `False`, no detrending is\n        done. Defaults to 'constant'.\n    return_onesided : bool, optional\n        If `True`, return a one-sided spectrum for real data. If\n        `False` return a two-sided spectrum. Defaults to `True`, but for\n        complex data, a two-sided spectrum is always returned.\n    scaling : { 'density', 'spectrum' }, optional\n        Selects between computing the cross spectral density ('density')\n        where `Pxy` has units of V**2/Hz and computing the cross spectrum\n        ('spectrum') where `Pxy` has units of V**2, if `x` and `y` are\n        measured in V and `fs` is measured in Hz. Defaults to 'density'\n    axis : int, optional\n        Axis along which the CSD is computed for both inputs; the\n        default is over the last axis (i.e. ``axis=-1``).\n    average : { 'mean', 'median' }, optional\n        Method to use when averaging periodograms. Defaults to 'mean'.\n\n\n    Returns\n    -------\n    f : ndarray\n        Array of sample frequencies.\n    Pxy : ndarray\n        Cross spectral density or cross power spectrum of x,y.\n\n    See Also\n    --------\n    periodogram: Simple, optionally modified periodogram\n    lombscargle: Lomb-Scargle periodogram for unevenly sampled data\n    welch: Power spectral density by Welch's method. [Equivalent to\n           csd(x,x)]\n    coherence: Magnitude squared coherence by Welch's method.\n\n    Notes\n    -----\n    By convention, Pxy is computed with the conjugate FFT of X\n    multiplied by the FFT of Y.\n\n    If the input series differ in length, the shorter series will be\n    zero-padded to match.\n\n    An appropriate amount of overlap will depend on the choice of window\n    and on your requirements. For the default Hann window an overlap of\n    50% is a reasonable trade off between accurately estimating the\n    signal power, while not over counting any of the data. Narrower\n    windows may require a larger overlap.\n\n    "
    x = cupy.asarray(x)
    y = cupy.asarray(y)
    (freqs, _, Pxy) = _spectral_helper(x, y, fs, window, nperseg, noverlap, nfft, detrend, return_onesided, scaling, axis, mode='psd')
    if len(Pxy.shape) >= 2 and Pxy.size > 0:
        if Pxy.shape[-1] > 1:
            if average == 'median':
                Pxy = cupy.median(Pxy, axis=-1) / _median_bias(Pxy.shape[-1])
            elif average == 'mean':
                Pxy = Pxy.mean(axis=-1)
            else:
                raise ValueError('average must be "median" or "mean", got %s' % (average,))
        else:
            Pxy = cupy.reshape(Pxy, Pxy.shape[:-1])
    return (freqs, Pxy)

def check_COLA(window, nperseg, noverlap, tol=1e-10):
    if False:
        while True:
            i = 10
    'Check whether the Constant OverLap Add (COLA) constraint is met.\n\n    Parameters\n    ----------\n    window : str or tuple or array_like\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg.\n    nperseg : int\n        Length of each segment.\n    noverlap : int\n        Number of points to overlap between segments.\n    tol : float, optional\n        The allowed variance of a bin\'s weighted sum from the median bin\n        sum.\n\n    Returns\n    -------\n    verdict : bool\n        `True` if chosen combination satisfies COLA within `tol`,\n        `False` otherwise\n\n    See Also\n    --------\n    check_NOLA: Check whether the Nonzero Overlap Add (NOLA) constraint is met\n    stft: Short Time Fourier Transform\n    istft: Inverse Short Time Fourier Transform\n\n    Notes\n    -----\n    In order to enable inversion of an STFT via the inverse STFT in\n    `istft`, it is sufficient that the signal windowing obeys the constraint of\n    "Constant OverLap Add" (COLA). This ensures that every point in the input\n    data is equally weighted, thereby avoiding aliasing and allowing full\n    reconstruction.\n\n    Some examples of windows that satisfy COLA:\n        - Rectangular window at overlap of 0, 1/2, 2/3, 3/4, ...\n        - Bartlett window at overlap of 1/2, 3/4, 5/6, ...\n        - Hann window at 1/2, 2/3, 3/4, ...\n        - Any Blackman family window at 2/3 overlap\n        - Any window with ``noverlap = nperseg-1``\n\n    A very comprehensive list of other windows may be found in [2]_,\n    wherein the COLA condition is satisfied when the "Amplitude\n    Flatness" is unity. See [1]_ for more information.\n\n    References\n    ----------\n    .. [1] Julius O. Smith III, "Spectral Audio Signal Processing", W3K\n           Publishing, 2011,ISBN 978-0-9745607-3-1.\n    .. [2] G. Heinzel, A. Ruediger and R. Schilling, "Spectrum and\n           spectral density estimation by the Discrete Fourier transform\n           (DFT), including a comprehensive list of window functions and\n           some new at-top windows", 2002,\n           http://hdl.handle.net/11858/00-001M-0000-0013-557A-5\n\n    '
    nperseg = int(nperseg)
    if nperseg < 1:
        raise ValueError('nperseg must be a positive integer')
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    noverlap = int(noverlap)
    if isinstance(window, str) or type(window) is tuple:
        win = get_window(window, nperseg)
    else:
        win = cupy.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if win.shape[0] != nperseg:
            raise ValueError('window must have length of nperseg')
    step = nperseg - noverlap
    binsums = sum((win[ii * step:(ii + 1) * step] for ii in range(nperseg // step)))
    if nperseg % step != 0:
        binsums[:nperseg % step] += win[-(nperseg % step):]
    deviation = binsums - cupy.median(binsums)
    return cupy.max(cupy.abs(deviation)) < tol

def check_NOLA(window, nperseg, noverlap, tol=1e-10):
    if False:
        return 10
    'Check whether the Nonzero Overlap Add (NOLA) constraint is met.\n\n    Parameters\n    ----------\n    window : str or tuple or array_like\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg.\n    nperseg : int\n        Length of each segment.\n    noverlap : int\n        Number of points to overlap between segments.\n    tol : float, optional\n        The allowed variance of a bin\'s weighted sum from the median bin\n        sum.\n\n    Returns\n    -------\n    verdict : bool\n        `True` if chosen combination satisfies the NOLA constraint within\n        `tol`, `False` otherwise\n\n    See Also\n    --------\n    check_COLA: Check whether the Constant OverLap Add (COLA) constraint is met\n    stft: Short Time Fourier Transform\n    istft: Inverse Short Time Fourier Transform\n\n    Notes\n    -----\n    In order to enable inversion of an STFT via the inverse STFT in\n    `istft`, the signal windowing must obey the constraint of "nonzero\n    overlap add" (NOLA):\n\n    .. math:: \\sum_{t}w^{2}[n-tH] \\ne 0\n\n    for all :math:`n`, where :math:`w` is the window function, :math:`t` is the\n    frame index, and :math:`H` is the hop size (:math:`H` = `nperseg` -\n    `noverlap`).\n\n    This ensures that the normalization factors in the denominator of the\n    overlap-add inversion equation are not zero. Only very pathological windows\n    will fail the NOLA constraint.\n\n    See [1]_, [2]_ for more information.\n\n    References\n    ----------\n    .. [1] Julius O. Smith III, "Spectral Audio Signal Processing", W3K\n           Publishing, 2011,ISBN 978-0-9745607-3-1.\n    .. [2] G. Heinzel, A. Ruediger and R. Schilling, "Spectrum and\n           spectral density estimation by the Discrete Fourier transform\n           (DFT), including a comprehensive list of window functions and\n           some new at-top windows", 2002,\n           http://hdl.handle.net/11858/00-001M-0000-0013-557A-5\n\n    '
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
        win = cupy.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if win.shape[0] != nperseg:
            raise ValueError('window must have length of nperseg')
    step = nperseg - noverlap
    binsums = sum((win[ii * step:(ii + 1) * step] ** 2 for ii in range(nperseg // step)))
    if nperseg % step != 0:
        binsums[:nperseg % step] += win[-(nperseg % step):] ** 2
    return cupy.min(binsums) > tol

def stft(x, fs=1.0, window='hann', nperseg=256, noverlap=None, nfft=None, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=-1, scaling='spectrum'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the Short Time Fourier Transform (STFT).\n\n    STFTs can be used as a way of quantifying the change of a\n    nonstationary signal\'s frequency and phase content over time.\n\n    Parameters\n    ----------\n    x : array_like\n        Time series of measurement values\n    fs : float, optional\n        Sampling frequency of the `x` time series. Defaults to 1.0.\n    window : str or tuple or array_like, optional\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg. Defaults\n        to a Hann window.\n    nperseg : int, optional\n        Length of each segment. Defaults to 256.\n    noverlap : int, optional\n        Number of points to overlap between segments. If `None`,\n        ``noverlap = nperseg // 2``. Defaults to `None`. When\n        specified, the COLA constraint must be met (see Notes below).\n    nfft : int, optional\n        Length of the FFT used, if a zero padded FFT is desired. If\n        `None`, the FFT length is `nperseg`. Defaults to `None`.\n    detrend : str or function or `False`, optional\n        Specifies how to detrend each segment. If `detrend` is a\n        string, it is passed as the `type` argument to the `detrend`\n        function. If it is a function, it takes a segment and returns a\n        detrended segment. If `detrend` is `False`, no detrending is\n        done. Defaults to `False`.\n    return_onesided : bool, optional\n        If `True`, return a one-sided spectrum for real data. If\n        `False` return a two-sided spectrum. Defaults to `True`, but for\n        complex data, a two-sided spectrum is always returned.\n    boundary : str or None, optional\n        Specifies whether the input signal is extended at both ends, and\n        how to generate the new values, in order to center the first\n        windowed segment on the first input point. This has the benefit\n        of enabling reconstruction of the first input point when the\n        employed window function starts at zero. Valid options are\n        ``[\'even\', \'odd\', \'constant\', \'zeros\', None]``. Defaults to\n        \'zeros\', for zero padding extension. I.e. ``[1, 2, 3, 4]`` is\n        extended to ``[0, 1, 2, 3, 4, 0]`` for ``nperseg=3``.\n    padded : bool, optional\n        Specifies whether the input signal is zero-padded at the end to\n        make the signal fit exactly into an integer number of window\n        segments, so that all of the signal is included in the output.\n        Defaults to `True`. Padding occurs after boundary extension, if\n        `boundary` is not `None`, and `padded` is `True`, as is the\n        default.\n    axis : int, optional\n        Axis along which the STFT is computed; the default is over the\n        last axis (i.e. ``axis=-1``).\n    scaling: {\'spectrum\', \'psd\'}\n        The default \'spectrum\' scaling allows each frequency line of `Zxx` to\n        be interpreted as a magnitude spectrum. The \'psd\' option scales each\n        line to a power spectral density - it allows to calculate the signal\'s\n        energy by numerically integrating over ``abs(Zxx)**2``.\n\n    Returns\n    -------\n    f : ndarray\n        Array of sample frequencies.\n    t : ndarray\n        Array of segment times.\n    Zxx : ndarray\n        STFT of `x`. By default, the last axis of `Zxx` corresponds\n        to the segment times.\n\n    See Also\n    --------\n    welch: Power spectral density by Welch\'s method.\n    spectrogram: Spectrogram by Welch\'s method.\n    csd: Cross spectral density by Welch\'s method.\n    lombscargle: Lomb-Scargle periodogram for unevenly sampled data\n\n    Notes\n    -----\n    In order to enable inversion of an STFT via the inverse STFT in\n    `istft`, the signal windowing must obey the constraint of "Nonzero\n    OverLap Add" (NOLA), and the input signal must have complete\n    windowing coverage (i.e. ``(x.shape[axis] - nperseg) %\n    (nperseg-noverlap) == 0``). The `padded` argument may be used to\n    accomplish this.\n\n    Given a time-domain signal :math:`x[n]`, a window :math:`w[n]`, and a hop\n    size :math:`H` = `nperseg - noverlap`, the windowed frame at time index\n    :math:`t` is given by\n\n    .. math:: x_{t}[n]=x[n]w[n-tH]\n\n    The overlap-add (OLA) reconstruction equation is given by\n\n    .. math:: x[n]=\\frac{\\sum_{t}x_{t}[n]w[n-tH]}{\\sum_{t}w^{2}[n-tH]}\n\n    The NOLA constraint ensures that every normalization term that appears\n    in the denomimator of the OLA reconstruction equation is nonzero. Whether a\n    choice of `window`, `nperseg`, and `noverlap` satisfy this constraint can\n    be tested with `check_NOLA`.\n\n    See [1]_, [2]_ for more information.\n\n    References\n    ----------\n    .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck\n           "Discrete-Time Signal Processing", Prentice Hall, 1999.\n    .. [2] Daniel W. Griffin, Jae S. Lim "Signal Estimation from\n           Modified Short-Time Fourier Transform", IEEE 1984,\n           10.1109/TASSP.1984.1164317\n\n    Examples\n    --------\n    >>> import cupy\n    >>> import cupyx.scipy.signal import stft\n    >>> import matplotlib.pyplot as plt\n\n    Generate a test signal, a 2 Vrms sine wave whose frequency is slowly\n    modulated around 3kHz, corrupted by white noise of exponentially\n    decreasing magnitude sampled at 10 kHz.\n\n    >>> fs = 10e3\n    >>> N = 1e5\n    >>> amp = 2 * cupy.sqrt(2)\n    >>> noise_power = 0.01 * fs / 2\n    >>> time = cupy.arange(N) / float(fs)\n    >>> mod = 500*cupy.cos(2*cupy.pi*0.25*time)\n    >>> carrier = amp * cupy.sin(2*cupy.pi*3e3*time + mod)\n    >>> noise = cupy.random.normal(scale=cupy.sqrt(noise_power),\n    ...                            size=time.shape)\n    >>> noise *= cupy.exp(-time/5)\n    >>> x = carrier + noise\n\n    Compute and plot the STFT\'s magnitude.\n\n    >>> f, t, Zxx = stft(x, fs, nperseg=1000)\n    >>> plt.pcolormesh(cupy.asnumpy(t), cupy.asnumpy(f),\n    ...                cupy.asnumpy(cupy.abs(Zxx)), vmin=0, vmax=amp)\n    >>> plt.title(\'STFT Magnitude\')\n    >>> plt.ylabel(\'Frequency [Hz]\')\n    >>> plt.xlabel(\'Time [sec]\')\n    >>> plt.show()\n    '
    if scaling == 'psd':
        scaling = 'density'
    elif scaling != 'spectrum':
        raise ValueError(f"Parameter scaling={scaling!r} not in ['spectrum', 'psd']!")
    (freqs, time, Zxx) = _spectral_helper(x, x, fs, window, nperseg, noverlap, nfft, detrend, return_onesided, scaling=scaling, axis=axis, mode='stft', boundary=boundary, padded=padded)
    return (freqs, time, Zxx)

def istft(Zxx, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, input_onesided=True, boundary=True, time_axis=-1, freq_axis=-2, scaling='spectrum'):
    if False:
        while True:
            i = 10
    '\n    Perform the inverse Short Time Fourier transform (iSTFT).\n\n    Parameters\n    ----------\n    Zxx : array_like\n        STFT of the signal to be reconstructed. If a purely real array\n        is passed, it will be cast to a complex data type.\n    fs : float, optional\n        Sampling frequency of the time series. Defaults to 1.0.\n    window : str or tuple or array_like, optional\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg. Defaults\n        to a Hann window. Must match the window used to generate the\n        STFT for faithful inversion.\n    nperseg : int, optional\n        Number of data points corresponding to each STFT segment. This\n        parameter must be specified if the number of data points per\n        segment is odd, or if the STFT was padded via ``nfft >\n        nperseg``. If `None`, the value depends on the shape of\n        `Zxx` and `input_onesided`. If `input_onesided` is `True`,\n        ``nperseg=2*(Zxx.shape[freq_axis] - 1)``. Otherwise,\n        ``nperseg=Zxx.shape[freq_axis]``. Defaults to `None`.\n    noverlap : int, optional\n        Number of points to overlap between segments. If `None`, half\n        of the segment length. Defaults to `None`. When specified, the\n        COLA constraint must be met (see Notes below), and should match\n        the parameter used to generate the STFT. Defaults to `None`.\n    nfft : int, optional\n        Number of FFT points corresponding to each STFT segment. This\n        parameter must be specified if the STFT was padded via ``nfft >\n        nperseg``. If `None`, the default values are the same as for\n        `nperseg`, detailed above, with one exception: if\n        `input_onesided` is True and\n        ``nperseg==2*Zxx.shape[freq_axis] - 1``, `nfft` also takes on\n        that value. This case allows the proper inversion of an\n        odd-length unpadded STFT using ``nfft=None``. Defaults to\n        `None`.\n    input_onesided : bool, optional\n        If `True`, interpret the input array as one-sided FFTs, such\n        as is returned by `stft` with ``return_onesided=True`` and\n        `numpy.fft.rfft`. If `False`, interpret the input as a\n        two-sided FFT. Defaults to `True`.\n    boundary : bool, optional\n        Specifies whether the input signal was extended at its\n        boundaries by supplying a non-`None` ``boundary`` argument to\n        `stft`. Defaults to `True`.\n    time_axis : int, optional\n        Where the time segments of the STFT is located; the default is\n        the last axis (i.e. ``axis=-1``).\n    freq_axis : int, optional\n        Where the frequency axis of the STFT is located; the default is\n        the penultimate axis (i.e. ``axis=-2``).\n    scaling: {\'spectrum\', \'psd\'}\n        The default \'spectrum\' scaling allows each frequency line of `Zxx` to\n        be interpreted as a magnitude spectrum. The \'psd\' option scales each\n        line to a power spectral density - it allows to calculate the signal\'s\n        energy by numerically integrating over ``abs(Zxx)**2``.\n\n    Returns\n    -------\n    t : ndarray\n        Array of output data times.\n    x : ndarray\n        iSTFT of `Zxx`.\n\n    See Also\n    --------\n    stft: Short Time Fourier Transform\n    check_COLA: Check whether the Constant OverLap Add (COLA) constraint\n                is met\n    check_NOLA: Check whether the Nonzero Overlap Add (NOLA) constraint is met\n\n    Notes\n    -----\n    In order to enable inversion of an STFT via the inverse STFT with\n    `istft`, the signal windowing must obey the constraint of "nonzero\n    overlap add" (NOLA):\n\n    .. math:: \\sum_{t}w^{2}[n-tH] \\ne 0\n\n    This ensures that the normalization factors that appear in the denominator\n    of the overlap-add reconstruction equation\n\n    .. math:: x[n]=\\frac{\\sum_{t}x_{t}[n]w[n-tH]}{\\sum_{t}w^{2}[n-tH]}\n\n    are not zero. The NOLA constraint can be checked with the `check_NOLA`\n    function.\n\n    An STFT which has been modified (via masking or otherwise) is not\n    guaranteed to correspond to a exactly realizible signal. This\n    function implements the iSTFT via the least-squares estimation\n    algorithm detailed in [2]_, which produces a signal that minimizes\n    the mean squared error between the STFT of the returned signal and\n    the modified STFT.\n\n    See [1]_, [2]_ for more information.\n\n    References\n    ----------\n    .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck\n           "Discrete-Time Signal Processing", Prentice Hall, 1999.\n    .. [2] Daniel W. Griffin, Jae S. Lim "Signal Estimation from\n           Modified Short-Time Fourier Transform", IEEE 1984,\n           10.1109/TASSP.1984.1164317\n\n    Examples\n    --------\n    >>> import cupy\n    >>> from cupyx.scipy.signal import stft, istft\n    >>> import matplotlib.pyplot as plt\n\n    Generate a test signal, a 2 Vrms sine wave at 50Hz corrupted by\n    0.001 V**2/Hz of white noise sampled at 1024 Hz.\n\n    >>> fs = 1024\n    >>> N = 10*fs\n    >>> nperseg = 512\n    >>> amp = 2 * np.sqrt(2)\n    >>> noise_power = 0.001 * fs / 2\n    >>> time = cupy.arange(N) / float(fs)\n    >>> carrier = amp * cupy.sin(2*cupy.pi*50*time)\n    >>> noise = cupy.random.normal(scale=cupy.sqrt(noise_power),\n    ...                          size=time.shape)\n    >>> x = carrier + noise\n\n    Compute the STFT, and plot its magnitude\n\n    >>> f, t, Zxx = cusignal.stft(x, fs=fs, nperseg=nperseg)\n    >>> f = cupy.asnumpy(f)\n    >>> t = cupy.asnumpy(t)\n    >>> Zxx = cupy.asnumpy(Zxx)\n    >>> plt.figure()\n    >>> plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading=\'gouraud\')\n    >>> plt.ylim([f[1], f[-1]])\n    >>> plt.title(\'STFT Magnitude\')\n    >>> plt.ylabel(\'Frequency [Hz]\')\n    >>> plt.xlabel(\'Time [sec]\')\n    >>> plt.yscale(\'log\')\n    >>> plt.show()\n\n    Zero the components that are 10% or less of the carrier magnitude,\n    then convert back to a time series via inverse STFT\n\n    >>> Zxx = cupy.where(cupy.abs(Zxx) >= amp/10, Zxx, 0)\n    >>> _, xrec = cusignal.istft(Zxx, fs)\n    >>> xrec = cupy.asnumpy(xrec)\n    >>> x = cupy.asnumpy(x)\n    >>> time = cupy.asnumpy(time)\n    >>> carrier = cupy.asnumpy(carrier)\n\n    Compare the cleaned signal with the original and true carrier signals.\n\n    >>> plt.figure()\n    >>> plt.plot(time, x, time, xrec, time, carrier)\n    >>> plt.xlim([2, 2.1])*+\n    >>> plt.xlabel(\'Time [sec]\')\n    >>> plt.ylabel(\'Signal\')\n    >>> plt.legend([\'Carrier + Noise\', \'Filtered via STFT\', \'True Carrier\'])\n    >>> plt.show()\n\n    Note that the cleaned signal does not start as abruptly as the original,\n    since some of the coefficients of the transient were also removed:\n\n    >>> plt.figure()\n    >>> plt.plot(time, x, time, xrec, time, carrier)\n    >>> plt.xlim([0, 0.1])\n    >>> plt.xlabel(\'Time [sec]\')\n    >>> plt.ylabel(\'Signal\')\n    >>> plt.legend([\'Carrier + Noise\', \'Filtered via STFT\', \'True Carrier\'])\n    >>> plt.show()\n    '
    Zxx = cupy.asarray(Zxx) + 0j
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
        Zxx = cupy.transpose(Zxx, zouter + [freq_axis, time_axis])
    if isinstance(window, str) or type(window) is tuple:
        win = get_window(window, nperseg)
    else:
        win = cupy.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if win.shape[0] != nperseg:
            raise ValueError('window must have length of {0}'.format(nperseg))
    ifunc = cupy.fft.irfft if input_onesided else cupy.fft.ifft
    xsubs = ifunc(Zxx, axis=-2, n=nfft)[..., :nperseg, :]
    outputlength = nperseg + (nseg - 1) * nstep
    x = cupy.zeros(list(Zxx.shape[:-2]) + [outputlength], dtype=xsubs.dtype)
    norm = cupy.zeros(outputlength, dtype=xsubs.dtype)
    if cupy.result_type(win, xsubs) != xsubs.dtype:
        win = win.astype(xsubs.dtype)
    if scaling == 'spectrum':
        xsubs *= win.sum()
    elif scaling == 'psd':
        xsubs *= cupy.sqrt(fs * cupy.sum(win ** 2))
    else:
        raise ValueError(f"Parameter scaling={scaling!r} not in ['spectrum', 'psd']!")
    for ii in range(nseg):
        x[..., ii * nstep:ii * nstep + nperseg] += xsubs[..., ii] * win
        norm[..., ii * nstep:ii * nstep + nperseg] += win ** 2
    if boundary:
        x = x[..., nperseg // 2:-(nperseg // 2)]
        norm = norm[..., nperseg // 2:-(nperseg // 2)]
    if cupy.sum(norm > 1e-10) != len(norm):
        warnings.warn('NOLA condition failed, STFT may not be invertible')
    x /= cupy.where(norm > 1e-10, norm, 1.0)
    if input_onesided:
        x = x.real
    if x.ndim > 1:
        if time_axis != Zxx.ndim - 1:
            if freq_axis < time_axis:
                time_axis -= 1
            x = cupy.rollaxis(x, -1, time_axis)
    time = cupy.arange(x.shape[0]) / float(fs)
    return (time, x)

def spectrogram(x, fs=1.0, window=('tukey', 0.25), nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='psd'):
    if False:
        i = 10
        return i + 15
    '\n    Compute a spectrogram with consecutive Fourier transforms.\n\n    Spectrograms can be used as a way of visualizing the change of a\n    nonstationary signal\'s frequency content over time.\n\n    Parameters\n    ----------\n    x : array_like\n        Time series of measurement values\n    fs : float, optional\n        Sampling frequency of the `x` time series. Defaults to 1.0.\n    window : str or tuple or array_like, optional\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg.\n        Defaults to a Tukey window with shape parameter of 0.25.\n    nperseg : int, optional\n        Length of each segment. Defaults to None, but if window is str or\n        tuple, is set to 256, and if window is array_like, is set to the\n        length of the window.\n    noverlap : int, optional\n        Number of points to overlap between segments. If `None`,\n        ``noverlap = nperseg // 8``. Defaults to `None`.\n    nfft : int, optional\n        Length of the FFT used, if a zero padded FFT is desired. If\n        `None`, the FFT length is `nperseg`. Defaults to `None`.\n    detrend : str or function or `False`, optional\n        Specifies how to detrend each segment. If `detrend` is a\n        string, it is passed as the `type` argument to the `detrend`\n        function. If it is a function, it takes a segment and returns a\n        detrended segment. If `detrend` is `False`, no detrending is\n        done. Defaults to \'constant\'.\n    return_onesided : bool, optional\n        If `True`, return a one-sided spectrum for real data. If\n        `False` return a two-sided spectrum. Defaults to `True`, but for\n        complex data, a two-sided spectrum is always returned.\n    scaling : { \'density\', \'spectrum\' }, optional\n        Selects between computing the power spectral density (\'density\')\n        where `Sxx` has units of V**2/Hz and computing the power\n        spectrum (\'spectrum\') where `Sxx` has units of V**2, if `x`\n        is measured in V and `fs` is measured in Hz. Defaults to\n        \'density\'.\n    axis : int, optional\n        Axis along which the spectrogram is computed; the default is over\n        the last axis (i.e. ``axis=-1``).\n    mode : str, optional\n        Defines what kind of return values are expected. Options are\n        [\'psd\', \'complex\', \'magnitude\', \'angle\', \'phase\']. \'complex\' is\n        equivalent to the output of `stft` with no padding or boundary\n        extension. \'magnitude\' returns the absolute magnitude of the\n        STFT. \'angle\' and \'phase\' return the complex angle of the STFT,\n        with and without unwrapping, respectively.\n\n    Returns\n    -------\n    f : ndarray\n        Array of sample frequencies.\n    t : ndarray\n        Array of segment times.\n    Sxx : ndarray\n        Spectrogram of x. By default, the last axis of Sxx corresponds\n        to the segment times.\n\n    See Also\n    --------\n    periodogram: Simple, optionally modified periodogram\n    lombscargle: Lomb-Scargle periodogram for unevenly sampled data\n    welch: Power spectral density by Welch\'s method.\n    csd: Cross spectral density by Welch\'s method.\n\n    Notes\n    -----\n    An appropriate amount of overlap will depend on the choice of window\n    and on your requirements. In contrast to welch\'s method, where the\n    entire data stream is averaged over, one may wish to use a smaller\n    overlap (or perhaps none at all) when computing a spectrogram, to\n    maintain some statistical independence between individual segments.\n    It is for this reason that the default window is a Tukey window with\n    1/8th of a window\'s length overlap at each end. See [1]_ for more\n    information.\n\n    References\n    ----------\n    .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck\n           "Discrete-Time Signal Processing", Prentice Hall, 1999.\n\n    Examples\n    --------\n    >>> import cupy\n    >>> from cupyx.scipy.signal import spectrogram\n    >>> import matplotlib.pyplot as plt\n\n    Generate a test signal, a 2 Vrms sine wave whose frequency is slowly\n    modulated around 3kHz, corrupted by white noise of exponentially\n    decreasing magnitude sampled at 10 kHz.\n\n    >>> fs = 10e3\n    >>> N = 1e5\n    >>> amp = 2 * cupy.sqrt(2)\n    >>> noise_power = 0.01 * fs / 2\n    >>> time = cupy.arange(N) / float(fs)\n    >>> mod = 500*cupy.cos(2*cupy.pi*0.25*time)\n    >>> carrier = amp * cupy.sin(2*cupy.pi*3e3*time + mod)\n    >>> noise = cupy.random.normal(\n    ...     scale=cupy.sqrt(noise_power), size=time.shape)\n    >>> noise *= cupy.exp(-time/5)\n    >>> x = carrier + noise\n\n    Compute and plot the spectrogram.\n\n    >>> f, t, Sxx = spectrogram(x, fs)\n    >>> plt.pcolormesh(cupy.asnumpy(t), cupy.asnumpy(f), cupy.asnumpy(Sxx))\n    >>> plt.ylabel(\'Frequency [Hz]\')\n    >>> plt.xlabel(\'Time [sec]\')\n    >>> plt.show()\n\n    Note, if using output that is not one sided, then use the following:\n\n    >>> f, t, Sxx = spectrogram(x, fs, return_onesided=False)\n    >>> plt.pcolormesh(cupy.asnumpy(t), cupy.fft.fftshift(f),         cupy.fft.fftshift(Sxx, axes=0))\n    >>> plt.ylabel(\'Frequency [Hz]\')\n    >>> plt.xlabel(\'Time [sec]\')\n    >>> plt.show()\n    '
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
            Sxx = cupy.abs(Sxx)
        elif mode in ['angle', 'phase']:
            Sxx = cupy.angle(Sxx)
            if mode == 'phase':
                if axis < 0:
                    axis -= 1
                Sxx = cupy.unwrap(Sxx, axis=axis)
    return (freqs, time, Sxx)

def coherence(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', axis=-1):
    if False:
        return 10
    '\n    Estimate the magnitude squared coherence estimate, Cxy, of\n    discrete-time signals X and Y using Welch\'s method.\n\n    ``Cxy = abs(Pxy)**2/(Pxx*Pyy)``, where `Pxx` and `Pyy` are power\n    spectral density estimates of X and Y, and `Pxy` is the cross\n    spectral density estimate of X and Y.\n\n    Parameters\n    ----------\n    x : array_like\n        Time series of measurement values\n    y : array_like\n        Time series of measurement values\n    fs : float, optional\n        Sampling frequency of the `x` and `y` time series. Defaults\n        to 1.0.\n    window : str or tuple or array_like, optional\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg. Defaults\n        to a Hann window.\n    nperseg : int, optional\n        Length of each segment. Defaults to None, but if window is str or\n        tuple, is set to 256, and if window is array_like, is set to the\n        length of the window.\n    noverlap: int, optional\n        Number of points to overlap between segments. If `None`,\n        ``noverlap = nperseg // 2``. Defaults to `None`.\n    nfft : int, optional\n        Length of the FFT used, if a zero padded FFT is desired. If\n        `None`, the FFT length is `nperseg`. Defaults to `None`.\n    detrend : str or function or `False`, optional\n        Specifies how to detrend each segment. If `detrend` is a\n        string, it is passed as the `type` argument to the `detrend`\n        function. If it is a function, it takes a segment and returns a\n        detrended segment. If `detrend` is `False`, no detrending is\n        done. Defaults to \'constant\'.\n    axis : int, optional\n        Axis along which the coherence is computed for both inputs; the\n        default is over the last axis (i.e. ``axis=-1``).\n\n    Returns\n    -------\n    f : ndarray\n        Array of sample frequencies.\n    Cxy : ndarray\n        Magnitude squared coherence of x and y.\n\n    See Also\n    --------\n    periodogram: Simple, optionally modified periodogram\n    lombscargle: Lomb-Scargle periodogram for unevenly sampled data\n    welch: Power spectral density by Welch\'s method.\n    csd: Cross spectral density by Welch\'s method.\n\n    Notes\n    -----\n    An appropriate amount of overlap will depend on the choice of window\n    and on your requirements. For the default Hann window an overlap of\n    50% is a reasonable trade off between accurately estimating the\n    signal power, while not over counting any of the data. Narrower\n    windows may require a larger overlap. See [1]_ and [2]_ for more\n    information.\n\n    References\n    ----------\n    .. [1] P. Welch, "The use of the fast Fourier transform for the\n           estimation of power spectra: A method based on time averaging\n           over short, modified periodograms", IEEE Trans. Audio\n           Electroacoust. vol. 15, pp. 70-73, 1967.\n    .. [2] Stoica, Petre, and Randolph Moses, "Spectral Analysis of\n           Signals" Prentice Hall, 2005\n\n    Examples\n    --------\n    >>> import cupy as cp\n    >>> from cupyx.scipy.signal import butter, lfilter, coherence\n    >>> import matplotlib.pyplot as plt\n\n    Generate two test signals with some common features.\n\n    >>> fs = 10e3\n    >>> N = 1e5\n    >>> amp = 20\n    >>> freq = 1234.0\n    >>> noise_power = 0.001 * fs / 2\n    >>> time = cupy.arange(N) / fs\n    >>> b, a = butter(2, 0.25, \'low\')\n    >>> x = cupy.random.normal(\n    ...         scale=cupy.sqrt(noise_power), size=time.shape)\n    >>> y = lfilter(b, a, x)\n    >>> x += amp * cupy.sin(2*cupy.pi*freq*time)\n    >>> y += cupy.random.normal(\n    ...         scale=0.1*cupy.sqrt(noise_power), size=time.shape)\n\n    Compute and plot the coherence.\n\n    >>> f, Cxy = coherence(x, y, fs, nperseg=1024)\n    >>> plt.semilogy(cupy.asnumpy(f), cupy.asnumpy(Cxy))\n    >>> plt.xlabel(\'frequency [Hz]\')\n    >>> plt.ylabel(\'Coherence\')\n    >>> plt.show()\n    '
    (freqs, Pxx) = welch(x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=detrend, axis=axis)
    (_, Pyy) = welch(y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=detrend, axis=axis)
    (_, Pxy) = csd(x, y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=detrend, axis=axis)
    Cxy = cupy.abs(Pxy) ** 2 / Pxx / Pyy
    return (freqs, Cxy)

def vectorstrength(events, period):
    if False:
        while True:
            i = 10
    '\n    Determine the vector strength of the events corresponding to the given\n    period.\n\n    The vector strength is a measure of phase synchrony, how well the\n    timing of the events is synchronized to a single period of a periodic\n    signal.\n\n    If multiple periods are used, calculate the vector strength of each.\n    This is called the "resonating vector strength".\n\n    Parameters\n    ----------\n    events : 1D array_like\n        An array of time points containing the timing of the events.\n    period : float or array_like\n        The period of the signal that the events should synchronize to.\n        The period is in the same units as `events`.  It can also be an array\n        of periods, in which case the outputs are arrays of the same length.\n\n    Returns\n    -------\n    strength : float or 1D array\n        The strength of the synchronization.  1.0 is perfect synchronization\n        and 0.0 is no synchronization.  If `period` is an array, this is also\n        an array with each element containing the vector strength at the\n        corresponding period.\n    phase : float or array\n        The phase that the events are most strongly synchronized to in radians.\n        If `period` is an array, this is also an array with each element\n        containing the phase for the corresponding period.\n\n    Notes\n    -----\n    See [1]_, [2]_ and [3]_ for more information.\n\n    References\n    ----------\n    .. [1] van Hemmen, JL, Longtin, A, and Vollmayr, AN. Testing resonating\n           vector strength: Auditory system, electric fish, and noise.\n           Chaos 21, 047508 (2011).\n    .. [2] van Hemmen, JL. Vector strength after Goldberg, Brown, and\n           von Mises: biological and mathematical perspectives.  Biol Cybern.\n           2013 Aug;107(4):385-96.\n    .. [3] van Hemmen, JL and Vollmayr, AN.  Resonating vector strength:\n           what happens when we vary the "probing" frequency while keeping\n           the spike times fixed.  Biol Cybern. 2013 Aug;107(4):491-94.\n    '
    events = cupy.asarray(events)
    period = cupy.asarray(period)
    if events.ndim > 1:
        raise ValueError('events cannot have dimensions more than 1')
    if period.ndim > 1:
        raise ValueError('period cannot have dimensions more than 1')
    scalarperiod = not period.ndim
    events = cupy.atleast_2d(events)
    period = cupy.atleast_2d(period)
    if (period <= 0).any():
        raise ValueError('periods must be positive')
    vectors = cupy.exp(cupy.dot(2j * cupy.pi / period.T, events))
    vectormean = cupy.mean(vectors, axis=1)
    strength = cupy.abs(vectormean)
    phase = cupy.angle(vectormean)
    if scalarperiod:
        strength = strength[0]
        phase = phase[0]
    return (strength, phase)