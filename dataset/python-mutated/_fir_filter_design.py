"""Functions for FIR filter design."""
import math
from cupy.fft import fft, ifft
from cupy.linalg import solve, lstsq, LinAlgError
from cupyx.scipy.linalg import toeplitz, hankel
import cupyx
from cupyx.scipy.signal.windows import get_window
import cupy
import numpy
__all__ = ['firls', 'minimum_phase']

def kaiser_beta(a):
    if False:
        for i in range(10):
            print('nop')
    'Compute the Kaiser parameter `beta`, given the attenuation `a`.\n\n    Parameters\n    ----------\n    a : float\n        The desired attenuation in the stopband and maximum ripple in\n        the passband, in dB.  This should be a *positive* number.\n\n    Returns\n    -------\n    beta : float\n        The `beta` parameter to be used in the formula for a Kaiser window.\n\n    References\n    ----------\n    Oppenheim, Schafer, "Discrete-Time Signal Processing", p.475-476.\n\n    See Also\n    --------\n    scipy.signal.kaiser_beta\n\n    '
    if a > 50:
        beta = 0.1102 * (a - 8.7)
    elif a > 21:
        beta = 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21)
    else:
        beta = 0.0
    return beta

def kaiser_atten(numtaps, width):
    if False:
        for i in range(10):
            print('nop')
    "Compute the attenuation of a Kaiser FIR filter.\n\n    Given the number of taps `N` and the transition width `width`, compute the\n    attenuation `a` in dB, given by Kaiser's formula:\n\n        a = 2.285 * (N - 1) * pi * width + 7.95\n\n    Parameters\n    ----------\n    numtaps : int\n        The number of taps in the FIR filter.\n    width : float\n        The desired width of the transition region between passband and\n        stopband (or, in general, at any discontinuity) for the filter,\n        expressed as a fraction of the Nyquist frequency.\n\n    Returns\n    -------\n    a : float\n        The attenuation of the ripple, in dB.\n\n    See Also\n    --------\n    scipy.signal.kaiser_atten\n    "
    a = 2.285 * (numtaps - 1) * cupy.pi * width + 7.95
    return a

def kaiserord(ripple, width):
    if False:
        return 10
    "\n    Determine the filter window parameters for the Kaiser window method.\n\n    The parameters returned by this function are generally used to create\n    a finite impulse response filter using the window method, with either\n    `firwin` or `firwin2`.\n\n    Parameters\n    ----------\n    ripple : float\n        Upper bound for the deviation (in dB) of the magnitude of the\n        filter's frequency response from that of the desired filter (not\n        including frequencies in any transition intervals). That is, if w\n        is the frequency expressed as a fraction of the Nyquist frequency,\n        A(w) is the actual frequency response of the filter and D(w) is the\n        desired frequency response, the design requirement is that::\n\n            abs(A(w) - D(w))) < 10**(-ripple/20)\n\n        for 0 <= w <= 1 and w not in a transition interval.\n    width : float\n        Width of transition region, normalized so that 1 corresponds to pi\n        radians / sample. That is, the frequency is expressed as a fraction\n        of the Nyquist frequency.\n\n    Returns\n    -------\n    numtaps : int\n        The length of the Kaiser window.\n    beta : float\n        The beta parameter for the Kaiser window.\n\n    See Also\n    --------\n    scipy.signal.kaiserord\n\n\n    "
    A = abs(ripple)
    if A < 8:
        raise ValueError('Requested maximum ripple attentuation %f is too small for the Kaiser formula.' % A)
    beta = kaiser_beta(A)
    numtaps = (A - 7.95) / 2.285 / (cupy.pi * width) + 1
    return (int(numpy.ceil(numtaps)), beta)
_firwin_kernel = cupy.ElementwiseKernel('float64 win, int32 numtaps, raw float64 bands, int32 steps, bool scale', 'float64 h, float64 hc', '\n    const double m { static_cast<double>( i ) - alpha ?\n        static_cast<double>( i ) - alpha : 1.0e-20 };\n\n    double temp {};\n    double left {};\n    double right {};\n\n    for ( int s = 0; s < steps; s++ ) {\n        left = bands[s * 2 + 0] ? bands[s * 2 + 0] : 1.0e-20;\n        right = bands[s * 2 + 1] ? bands[s * 2 + 1] : 1.0e-20;\n\n        temp += right * ( sin( right * m * M_PI ) / ( right * m * M_PI ) );\n        temp -= left * ( sin( left * m * M_PI ) / ( left * m * M_PI ) );\n    }\n\n    temp *= win;\n    h = temp;\n\n    double scale_frequency {};\n\n    if ( scale ) {\n        left = bands[0];\n        right = bands[1];\n\n        if ( left == 0 ) {\n            scale_frequency = 0.0;\n        } else if ( right == 1 ) {\n            scale_frequency = 1.0;\n        } else {\n            scale_frequency = 0.5 * ( left + right );\n        }\n        double c { cos( M_PI * m * scale_frequency ) };\n        hc = temp * c;\n    }\n    ', '_firwin_kernel', options=('-std=c++11',), loop_prep='const double alpha { 0.5 * ( numtaps - 1 ) };')

def firwin(numtaps, cutoff, width=None, window='hamming', pass_zero=True, scale=True, fs=2):
    if False:
        return 10
    '\n    FIR filter design using the window method.\n\n    This function computes the coefficients of a finite impulse response\n    filter.  The filter will have linear phase; it will be Type I if\n    `numtaps` is odd and Type II if `numtaps` is even.\n\n    Type II filters always have zero response at the Nyquist frequency, so a\n    ValueError exception is raised if firwin is called with `numtaps` even and\n    having a passband whose right end is at the Nyquist frequency.\n\n    Parameters\n    ----------\n    numtaps : int\n        Length of the filter (number of coefficients, i.e. the filter\n        order + 1).  `numtaps` must be odd if a passband includes the\n        Nyquist frequency.\n    cutoff : float or 1D array_like\n        Cutoff frequency of filter (expressed in the same units as `fs`)\n        OR an array of cutoff frequencies (that is, band edges). In the\n        latter case, the frequencies in `cutoff` should be positive and\n        monotonically increasing between 0 and `fs/2`.  The values 0 and\n        `fs/2` must not be included in `cutoff`.\n    width : float or None, optional\n        If `width` is not None, then assume it is the approximate width\n        of the transition region (expressed in the same units as `fs`)\n        for use in Kaiser FIR filter design.  In this case, the `window`\n        argument is ignored.\n    window : string or tuple of string and parameter values, optional\n        Desired window to use. See `cusignal.get_window` for a list\n        of windows and required parameters.\n    pass_zero : {True, False, \'bandpass\', \'lowpass\', \'highpass\', \'bandstop\'},\n        optional\n        If True, the gain at the frequency 0 (i.e. the "DC gain") is 1.\n        If False, the DC gain is 0. Can also be a string argument for the\n        desired filter type (equivalent to ``btype`` in IIR design functions).\n    scale : bool, optional\n        Set to True to scale the coefficients so that the frequency\n        response is exactly unity at a certain frequency.\n        That frequency is either:\n\n        - 0 (DC) if the first passband starts at 0 (i.e. pass_zero\n          is True)\n        - `fs/2` (the Nyquist frequency) if the first passband ends at\n          `fs/2` (i.e the filter is a single band highpass filter);\n          center of first passband otherwise\n    fs : float, optional\n        The sampling frequency of the signal.  Each frequency in `cutoff`\n        must be between 0 and ``fs/2``.  Default is 2.\n\n    Returns\n    -------\n    h : (numtaps,) ndarray\n        Coefficients of length `numtaps` FIR filter.\n\n    Raises\n    ------\n    ValueError\n        If any value in `cutoff` is less than or equal to 0 or greater\n        than or equal to ``fs/2``, if the values in `cutoff` are not strictly\n        monotonically increasing, or if `numtaps` is even but a passband\n        includes the Nyquist frequency.\n\n    See Also\n    --------\n    firwin2\n    firls\n    minimum_phase\n    remez\n\n    Examples\n    --------\n    Low-pass from 0 to f:\n\n    >>> import cusignal\n    >>> numtaps = 3\n    >>> f = 0.1\n    >>> cusignal.firwin(numtaps, f)\n    array([ 0.06799017,  0.86401967,  0.06799017])\n\n    Use a specific window function:\n\n    >>> cusignal.firwin(numtaps, f, window=\'nuttall\')\n    array([  3.56607041e-04,   9.99286786e-01,   3.56607041e-04])\n\n    High-pass (\'stop\' from 0 to f):\n\n    >>> cusignal.firwin(numtaps, f, pass_zero=False)\n    array([-0.00859313,  0.98281375, -0.00859313])\n\n    Band-pass:\n\n    >>> f1, f2 = 0.1, 0.2\n    >>> cusignal.firwin(numtaps, [f1, f2], pass_zero=False)\n    array([ 0.06301614,  0.88770441,  0.06301614])\n\n    Band-stop:\n\n    >>> cusignal.firwin(numtaps, [f1, f2])\n    array([-0.00801395,  1.0160279 , -0.00801395])\n\n    Multi-band (passbands are [0, f1], [f2, f3] and [f4, 1]):\n\n    >>> f3, f4 = 0.3, 0.4\n    >>> cusignal.firwin(numtaps, [f1, f2, f3, f4])\n    array([-0.01376344,  1.02752689, -0.01376344])\n\n    Multi-band (passbands are [f1, f2] and [f3,f4]):\n\n    >>> cusignal.firwin(numtaps, [f1, f2, f3, f4], pass_zero=False)\n    array([ 0.04890915,  0.91284326,  0.04890915])\n\n    '
    nyq = 0.5 * fs
    cutoff = cupy.atleast_1d(cutoff) / float(nyq)
    if cutoff.ndim > 1:
        raise ValueError('The cutoff argument must be at most one-dimensional.')
    if cutoff.size == 0:
        raise ValueError('At least one cutoff frequency must be given.')
    if cutoff.min() <= 0 or cutoff.max() >= 1:
        raise ValueError('Invalid cutoff frequency: frequencies must be greater than 0 and less than nyq.')
    if cupy.any(cupy.diff(cutoff) <= 0):
        raise ValueError('Invalid cutoff frequencies: the frequencies must be strictly increasing.')
    if width is not None:
        atten = kaiser_atten(numtaps, float(width) / nyq)
        beta = kaiser_beta(atten)
        window = ('kaiser', beta)
    if isinstance(pass_zero, str):
        if pass_zero in ('bandstop', 'lowpass'):
            if pass_zero == 'lowpass':
                if cutoff.size != 1:
                    raise ValueError('cutoff must have one element if pass_zero=="lowpass", got %s' % (cutoff.shape,))
            elif cutoff.size <= 1:
                raise ValueError('cutoff must have at least two elements if pass_zero=="bandstop", got %s' % (cutoff.shape,))
            pass_zero = True
        elif pass_zero in ('bandpass', 'highpass'):
            if pass_zero == 'highpass':
                if cutoff.size != 1:
                    raise ValueError('cutoff must have one element if pass_zero=="highpass", got %s' % (cutoff.shape,))
            elif cutoff.size <= 1:
                raise ValueError('cutoff must have at least two elements if pass_zero=="bandpass", got %s' % (cutoff.shape,))
            pass_zero = False
        else:
            raise ValueError('pass_zero must be True, False, "bandpass", "lowpass", "highpass", or "bandstop", got {}'.format(pass_zero))
    pass_nyquist = bool(cutoff.size & 1) ^ pass_zero
    if pass_nyquist and numtaps % 2 == 0:
        raise ValueError('A filter with an even number of coefficients must have zero response at the Nyquist rate.')
    cutoff = cupy.hstack(([0.0] * pass_zero, cutoff, [1.0] * pass_nyquist))
    bands = cutoff.reshape(-1, 2)
    win = get_window(window, numtaps, fftbins=False)
    (h, hc) = _firwin_kernel(win, numtaps, bands, bands.shape[0], scale)
    if scale:
        s = cupy.sum(hc)
        h /= s
        alpha = 0.5 * (numtaps - 1)
        m = cupy.arange(0, numtaps) - alpha
        h = 0
        for (left, right) in bands:
            h += right * cupy.sinc(right * m)
            h -= left * cupy.sinc(left * m)
        h *= win
        if scale:
            (left, right) = bands[0]
            if left == 0:
                scale_frequency = 0.0
            elif right == 1:
                scale_frequency = 1.0
            else:
                scale_frequency = 0.5 * (left + right)
            c = cupy.cos(cupy.pi * m * scale_frequency)
            s = cupy.sum(h * c)
            h /= s
    return h

def firwin2(numtaps, freq, gain, nfreqs=None, window='hamming', nyq=None, antisymmetric=False, fs=2.0):
    if False:
        print('Hello World!')
    '\n    FIR filter design using the window method.\n\n    From the given frequencies `freq` and corresponding gains `gain`,\n    this function constructs an FIR filter with linear phase and\n    (approximately) the given frequency response.\n\n    Parameters\n    ----------\n    numtaps : int\n        The number of taps in the FIR filter.  `numtaps` must be less than\n        `nfreqs`.\n    freq : array_like, 1-D\n        The frequency sampling points. Typically 0.0 to 1.0 with 1.0 being\n        Nyquist.  The Nyquist frequency is half `fs`.\n        The values in `freq` must be nondecreasing. A value can be repeated\n        once to implement a discontinuity. The first value in `freq` must\n        be 0, and the last value must be ``fs/2``. Values 0 and ``fs/2`` must\n        not be repeated.\n    gain : array_like\n        The filter gains at the frequency sampling points. Certain\n        constraints to gain values, depending on the filter type, are applied,\n        see Notes for details.\n    nfreqs : int, optional\n        The size of the interpolation mesh used to construct the filter.\n        For most efficient behavior, this should be a power of 2 plus 1\n        (e.g, 129, 257, etc). The default is one more than the smallest\n        power of 2 that is not less than `numtaps`. `nfreqs` must be greater\n        than `numtaps`.\n    window : string or (string, float) or float, or None, optional\n        Window function to use. Default is "hamming". See\n        `scipy.signal.get_window` for the complete list of possible values.\n        If None, no window function is applied.\n    antisymmetric : bool, optional\n        Whether resulting impulse response is symmetric/antisymmetric.\n        See Notes for more details.\n    fs : float, optional\n        The sampling frequency of the signal. Each frequency in `cutoff`\n        must be between 0 and ``fs/2``. Default is 2.\n\n    Returns\n    -------\n    taps : ndarray\n        The filter coefficients of the FIR filter, as a 1-D array of length\n        `numtaps`.\n\n    See Also\n    --------\n    scipy.signal.firwin2\n    firls\n    firwin\n    minimum_phase\n    remez\n\n    Notes\n    -----\n    From the given set of frequencies and gains, the desired response is\n    constructed in the frequency domain. The inverse FFT is applied to the\n    desired response to create the associated convolution kernel, and the\n    first `numtaps` coefficients of this kernel, scaled by `window`, are\n    returned.\n    The FIR filter will have linear phase. The type of filter is determined by\n    the value of \'numtaps` and `antisymmetric` flag.\n    There are four possible combinations:\n\n       - odd  `numtaps`, `antisymmetric` is False, type I filter is produced\n       - even `numtaps`, `antisymmetric` is False, type II filter is produced\n       - odd  `numtaps`, `antisymmetric` is True, type III filter is produced\n       - even `numtaps`, `antisymmetric` is True, type IV filter is produced\n\n    Magnitude response of all but type I filters are subjects to following\n    constraints:\n\n       - type II  -- zero at the Nyquist frequency\n       - type III -- zero at zero and Nyquist frequencies\n       - type IV  -- zero at zero frequency\n    '
    nyq = 0.5 * fs
    if len(freq) != len(gain):
        raise ValueError('freq and gain must be of same length.')
    if nfreqs is not None and numtaps >= nfreqs:
        raise ValueError('ntaps must be less than nfreqs, but firwin2 was called with ntaps=%d and nfreqs=%s' % (numtaps, nfreqs))
    if freq[0] != 0 or freq[-1] != nyq:
        raise ValueError('freq must start with 0 and end with fs/2.')
    d = cupy.diff(freq)
    if (d < 0).any():
        raise ValueError('The values in freq must be nondecreasing.')
    d2 = d[:-1] + d[1:]
    if (d2 == 0).any():
        raise ValueError('A value in freq must not occur more than twice.')
    if freq[1] == 0:
        raise ValueError('Value 0 must not be repeated in freq')
    if freq[-2] == nyq:
        raise ValueError('Value fs/2 must not be repeated in freq')
    if antisymmetric:
        if numtaps % 2 == 0:
            ftype = 4
        else:
            ftype = 3
    elif numtaps % 2 == 0:
        ftype = 2
    else:
        ftype = 1
    if ftype == 2 and gain[-1] != 0.0:
        raise ValueError('A Type II filter must have zero gain at the Nyquist frequency.')
    elif ftype == 3 and (gain[0] != 0.0 or gain[-1] != 0.0):
        raise ValueError('A Type III filter must have zero gain at zero and Nyquist frequencies.')
    elif ftype == 4 and gain[0] != 0.0:
        raise ValueError('A Type IV filter must have zero gain at zero frequency.')
    if nfreqs is None:
        nfreqs = 1 + 2 ** int(math.ceil(math.log(numtaps, 2)))
    if (d == 0).any():
        freq = cupy.array(freq, copy=True)
        eps = cupy.finfo(float).eps * nyq
        for k in range(len(freq) - 1):
            if freq[k] == freq[k + 1]:
                freq[k] = freq[k] - eps
                freq[k + 1] = freq[k + 1] + eps
        d = cupy.diff(freq)
        if (d <= 0).any():
            raise ValueError('freq cannot contain numbers that are too close (within eps * (fs/2): {}) to a repeated value'.format(eps))
    x = cupy.linspace(0.0, nyq, nfreqs)
    fx = cupy.interp(x, freq, gain)
    shift = cupy.exp(-(numtaps - 1) / 2.0 * 1j * math.pi * x / nyq)
    if ftype > 2:
        shift *= 1j
    fx2 = fx * shift
    out_full = cupy.fft.irfft(fx2)
    if window is not None:
        wind = get_window(window, numtaps, fftbins=False)
    else:
        wind = 1
    out = out_full[:numtaps] * wind
    if ftype == 3:
        out[out.size // 2] = 0.0
    return out

def firls(numtaps, bands, desired, weight=None, fs=2):
    if False:
        print('Hello World!')
    '\n    FIR filter design using least-squares error minimization.\n\n    Calculate the filter coefficients for the linear-phase finite\n    impulse response (FIR) filter which has the best approximation\n    to the desired frequency response described by `bands` and\n    `desired` in the least squares sense (i.e., the integral of the\n    weighted mean-squared error within the specified bands is\n    minimized).\n\n    Parameters\n    ----------\n    numtaps : int\n        The number of taps in the FIR filter. `numtaps` must be odd.\n    bands : array_like\n        A monotonic nondecreasing sequence containing the band edges in\n        Hz. All elements must be non-negative and less than or equal to\n        the Nyquist frequency given by `fs`/2. The bands are specified as\n        frequency pairs, thus, if using a 1D array, its length must be\n        even, e.g., `cupy.array([0, 1, 2, 3, 4, 5])`. Alternatively, the\n        bands can be specified as an nx2 sized 2D array, where n is the\n        number of bands, e.g, `cupy.array([[0, 1], [2, 3], [4, 5]])`.\n        All elements of `bands` must be monotonically nondecreasing, have\n        width > 0, and must not overlap. (This is not checked by the routine).\n    desired : array_like\n        A sequence the same size as `bands` containing the desired gain\n        at the start and end point of each band.\n        All elements must be non-negative (this is not checked by the routine).\n    weight : array_like, optional\n        A relative weighting to give to each band region when solving\n        the least squares problem. `weight` has to be half the size of\n        `bands`.\n        All elements must be non-negative (this is not checked by the routine).\n    fs : float, optional\n        The sampling frequency of the signal. Each frequency in `bands`\n        must be between 0 and ``fs/2`` (inclusive). Default is 2.\n\n    Returns\n    -------\n    coeffs : ndarray\n        Coefficients of the optimal (in a least squares sense) FIR filter.\n\n    See Also\n    --------\n    firwin\n    firwin2\n    minimum_phase\n    remez\n    scipy.signal.firls\n    '
    nyq = 0.5 * fs
    numtaps = int(numtaps)
    if numtaps % 2 == 0 or numtaps < 1:
        raise ValueError('numtaps must be odd and >= 1')
    M = (numtaps - 1) // 2
    nyq = float(nyq)
    if nyq <= 0:
        raise ValueError('nyq must be positive, got %s <= 0.' % nyq)
    bands = cupy.asarray(bands).flatten() / nyq
    if len(bands) % 2 != 0:
        raise ValueError('bands must contain frequency pairs.')
    if (bands < 0).any() or (bands > 1).any():
        raise ValueError('bands must be between 0 and 1 relative to Nyquist')
    bands.shape = (-1, 2)
    desired = cupy.asarray(desired).flatten()
    if bands.size != desired.size:
        raise ValueError('desired must have one entry per frequency, got %s gains for %s frequencies.' % (desired.size, bands.size))
    desired.shape = (-1, 2)
    if weight is None:
        weight = cupy.ones(len(desired))
    weight = cupy.asarray(weight).flatten()
    if len(weight) != len(desired):
        raise ValueError('weight must be the same size as the number of band pairs ({}).'.format(len(bands)))
    n = cupy.arange(numtaps)[:, cupy.newaxis, cupy.newaxis]
    q = cupy.dot(cupy.diff(cupy.sinc(bands * n) * bands, axis=2)[:, :, 0], weight)
    Q1 = toeplitz(q[:M + 1])
    Q2 = hankel(q[:M + 1], q[M:])
    Q = Q1 + Q2
    n = n[:M + 1]
    m = cupy.diff(desired, axis=1) / cupy.diff(bands, axis=1)
    c = desired[:, [0]] - bands[:, [0]] * m
    b = bands * (m * bands + c) * cupy.sinc(bands * n)
    b[0] -= m * bands * bands / 2.0
    b[1:] += m * cupy.cos(n[1:] * cupy.pi * bands) / (cupy.pi * n[1:]) ** 2
    b = cupy.diff(b, axis=2)[:, :, 0] @ weight
    with cupyx.errstate(linalg='raise'):
        try:
            a = solve(Q, b)
        except LinAlgError:
            a = lstsq(Q, b, rcond=None)[0]
    coeffs = cupy.hstack((a[:0:-1], 2 * a[0], a[1:]))
    return coeffs

def _dhtm(mag):
    if False:
        return 10
    'Compute the modified 1-D discrete Hilbert transform\n\n    Parameters\n    ----------\n    mag : ndarray\n        The magnitude spectrum. Should be 1-D with an even length, and\n        preferably a fast length for FFT/IFFT.\n    '
    sig = cupy.zeros(len(mag))
    midpt = len(mag) // 2
    sig[1:midpt] = 1
    sig[midpt + 1:] = -1
    recon = ifft(mag * cupy.exp(fft(sig * ifft(cupy.log(mag))))).real
    return recon

def minimum_phase(h, method='homomorphic', n_fft=None):
    if False:
        for i in range(10):
            print('nop')
    'Convert a linear-phase FIR filter to minimum phase\n\n    Parameters\n    ----------\n    h : array\n        Linear-phase FIR filter coefficients.\n    method : {\'hilbert\', \'homomorphic\'}\n        The method to use:\n\n            \'homomorphic\' (default)\n                This method [4]_ [5]_ works best with filters with an\n                odd number of taps, and the resulting minimum phase filter\n                will have a magnitude response that approximates the square\n                root of the original filter\'s magnitude response.\n\n            \'hilbert\'\n                This method [1]_ is designed to be used with equiripple\n                filters (e.g., from `remez`) with unity or zero gain\n                regions.\n\n    n_fft : int\n        The number of points to use for the FFT. Should be at least a\n        few times larger than the signal length (see Notes).\n\n    Returns\n    -------\n    h_minimum : array\n        The minimum-phase version of the filter, with length\n        ``(length(h) + 1) // 2``.\n\n    See Also\n    --------\n    scipy.signal.minimum_phase\n\n    Notes\n    -----\n    Both the Hilbert [1]_ or homomorphic [4]_ [5]_ methods require selection\n    of an FFT length to estimate the complex cepstrum of the filter.\n\n    In the case of the Hilbert method, the deviation from the ideal\n    spectrum ``epsilon`` is related to the number of stopband zeros\n    ``n_stop`` and FFT length ``n_fft`` as::\n\n        epsilon = 2. * n_stop / n_fft\n\n    For example, with 100 stopband zeros and a FFT length of 2048,\n    ``epsilon = 0.0976``. If we conservatively assume that the number of\n    stopband zeros is one less than the filter length, we can take the FFT\n    length to be the next power of 2 that satisfies ``epsilon=0.01`` as::\n\n        n_fft = 2 ** int(np.ceil(np.log2(2 * (len(h) - 1) / 0.01)))\n\n    This gives reasonable results for both the Hilbert and homomorphic\n    methods, and gives the value used when ``n_fft=None``.\n\n    Alternative implementations exist for creating minimum-phase filters,\n    including zero inversion [2]_ and spectral factorization [3]_ [4]_ [5]_.\n    For more information, see:\n\n        http://dspguru.com/dsp/howtos/how-to-design-minimum-phase-fir-filters\n\n    References\n    ----------\n    .. [1] N. Damera-Venkata and B. L. Evans, "Optimal design of real and\n           complex minimum phase digital FIR filters," Acoustics, Speech,\n           and Signal Processing, 1999. Proceedings., 1999 IEEE International\n           Conference on, Phoenix, AZ, 1999, pp. 1145-1148 vol.3.\n           DOI:10.1109/ICASSP.1999.756179\n    .. [2] X. Chen and T. W. Parks, "Design of optimal minimum phase FIR\n           filters by direct factorization," Signal Processing,\n           vol. 10, no. 4, pp. 369-383, Jun. 1986.\n    .. [3] T. Saramaki, "Finite Impulse Response Filter Design," in\n           Handbook for Digital Signal Processing, chapter 4,\n           New York: Wiley-Interscience, 1993.\n    .. [4] J. S. Lim, Advanced Topics in Signal Processing.\n           Englewood Cliffs, N.J.: Prentice Hall, 1988.\n    .. [5] A. V. Oppenheim, R. W. Schafer, and J. R. Buck,\n           "Discrete-Time Signal Processing," 2nd edition.\n           Upper Saddle River, N.J.: Prentice Hall, 1999.\n\n    '
    if cupy.iscomplexobj(h):
        raise ValueError('Complex filters not supported')
    if h.ndim != 1 or h.size <= 2:
        raise ValueError('h must be 1-D and at least 2 samples long')
    n_half = len(h) // 2
    if not cupy.allclose(h[-n_half:][::-1], h[:n_half]):
        import warnings
        warnings.warn('h does not appear to by symmetric, conversion may fail', RuntimeWarning)
    if not isinstance(method, str) or method not in ('homomorphic', 'hilbert'):
        raise ValueError('method must be "homomorphic" or "hilbert", got %r' % (method,))
    if n_fft is None:
        n_fft = 2 ** int(cupy.ceil(cupy.log2(2 * (len(h) - 1) / 0.01)))
    n_fft = int(n_fft)
    if n_fft < len(h):
        raise ValueError('n_fft must be at least len(h)==%s' % len(h))
    if method == 'hilbert':
        w = cupy.arange(n_fft) * (2 * cupy.pi / n_fft * n_half)
        H = cupy.real(fft(h, n_fft) * cupy.exp(1j * w))
        dp = max(H) - 1
        ds = 0 - min(H)
        S = 4.0 / (cupy.sqrt(1 + dp + ds) + cupy.sqrt(1 - dp + ds)) ** 2
        H += ds
        H *= S
        H = cupy.sqrt(H, out=H)
        H += 1e-10
        h_minimum = _dhtm(H)
    else:
        h_temp = cupy.abs(fft(h, n_fft))
        h_temp += 1e-07 * h_temp[h_temp > 0].min()
        cupy.log(h_temp, out=h_temp)
        h_temp *= 0.5
        h_temp = ifft(h_temp).real
        win = cupy.zeros(n_fft)
        win[0] = 1
        stop = (len(h) + 1) // 2
        win[1:stop] = 2
        if len(h) % 2:
            win[stop] = 1
        h_temp *= win
        h_temp = ifft(cupy.exp(fft(h_temp)))
        h_minimum = h_temp.real
    n_out = n_half + len(h) % 2
    return h_minimum[:n_out]