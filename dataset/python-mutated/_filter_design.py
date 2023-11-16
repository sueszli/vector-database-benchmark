import operator
from math import pi
import warnings
import cupy
from cupy.polynomial.polynomial import polyval as npp_polyval, polyvalfromroots as npp_polyvalfromroots
import cupyx.scipy.fft as sp_fft
from cupyx import jit
from cupyx.scipy._lib._util import float_factorial
from cupyx.scipy.signal._polyutils import roots
EPSILON = 2e-16

def _try_convert_to_int(x):
    if False:
        for i in range(10):
            print('nop')
    'Return an integer for ``5`` and ``array(5)``, fail if not an\n       integer scalar.\n\n    NB: would be easier if ``operator.index(cupy.array(5))`` worked\n    (numpy.array(5) does)\n    '
    if isinstance(x, cupy.ndarray):
        if x.ndim == 0:
            value = x.item()
        else:
            return (x, False)
    else:
        value = x
    try:
        return (operator.index(value), True)
    except TypeError:
        return (value, False)

def findfreqs(num, den, N, kind='ba'):
    if False:
        print('Hello World!')
    '\n    Find array of frequencies for computing the response of an analog filter.\n\n    Parameters\n    ----------\n    num, den : array_like, 1-D\n        The polynomial coefficients of the numerator and denominator of the\n        transfer function of the filter or LTI system, where the coefficients\n        are ordered from highest to lowest degree. Or, the roots  of the\n        transfer function numerator and denominator (i.e., zeroes and poles).\n    N : int\n        The length of the array to be computed.\n    kind : str {\'ba\', \'zp\'}, optional\n        Specifies whether the numerator and denominator are specified by their\n        polynomial coefficients (\'ba\'), or their roots (\'zp\').\n\n    Returns\n    -------\n    w : (N,) ndarray\n        A 1-D array of frequencies, logarithmically spaced.\n\n    Warning\n    -------\n    This function may synchronize the device.\n\n    See Also\n    --------\n    scipy.signal.find_freqs\n\n    Examples\n    --------\n    Find a set of nine frequencies that span the "interesting part" of the\n    frequency response for the filter with the transfer function\n\n        H(s) = s / (s^2 + 8s + 25)\n\n    >>> from scipy import signal\n    >>> signal.findfreqs([1, 0], [1, 8, 25], N=9)\n    array([  1.00000000e-02,   3.16227766e-02,   1.00000000e-01,\n             3.16227766e-01,   1.00000000e+00,   3.16227766e+00,\n             1.00000000e+01,   3.16227766e+01,   1.00000000e+02])\n    '
    if kind == 'ba':
        ep = cupy.atleast_1d(roots(den)) + 0j
        tz = cupy.atleast_1d(roots(num)) + 0j
    elif kind == 'zp':
        ep = cupy.atleast_1d(den) + 0j
        tz = cupy.atleast_1d(num) + 0j
    else:
        raise ValueError("input must be one of {'ba', 'zp'}")
    if len(ep) == 0:
        ep = cupy.atleast_1d(-1000) + 0j
    ez = cupy.r_[cupy.compress(ep.imag >= 0, ep, axis=-1), cupy.compress((abs(tz) < 100000.0) & (tz.imag >= 0), tz, axis=-1)]
    integ = cupy.abs(ez) < 1e-10
    hfreq = cupy.around(cupy.log10(cupy.max(3 * cupy.abs(ez.real + integ) + 1.5 * ez.imag)) + 0.5)
    lfreq = cupy.around(cupy.log10(0.1 * cupy.min(cupy.abs((ez + integ).real) + 2 * ez.imag)) - 0.5)
    w = cupy.logspace(lfreq, hfreq, N)
    return w

def freqs(b, a, worN=200, plot=None):
    if False:
        while True:
            i = 10
    '\n    Compute frequency response of analog filter.\n\n    Given the M-order numerator `b` and N-order denominator `a` of an analog\n    filter, compute its frequency response::\n\n             b[0]*(jw)**M + b[1]*(jw)**(M-1) + ... + b[M]\n     H(w) = ----------------------------------------------\n             a[0]*(jw)**N + a[1]*(jw)**(N-1) + ... + a[N]\n\n    Parameters\n    ----------\n    b : array_like\n        Numerator of a linear filter.\n    a : array_like\n        Denominator of a linear filter.\n    worN : {None, int, array_like}, optional\n        If None, then compute at 200 frequencies around the interesting parts\n        of the response curve (determined by pole-zero locations). If a single\n        integer, then compute at that many frequencies. Otherwise, compute the\n        response at the angular frequencies (e.g., rad/s) given in `worN`.\n    plot : callable, optional\n        A callable that takes two arguments. If given, the return parameters\n        `w` and `h` are passed to plot. Useful for plotting the frequency\n        response inside `freqs`.\n\n    Returns\n    -------\n    w : ndarray\n        The angular frequencies at which `h` was computed.\n    h : ndarray\n        The frequency response.\n\n    See Also\n    --------\n    scipy.signal.freqs\n    freqz : Compute the frequency response of a digital filter.\n\n    '
    if worN is None:
        w = findfreqs(b, a, 200)
    else:
        (N, _is_int) = _try_convert_to_int(worN)
        if _is_int:
            w = findfreqs(b, a, N)
        else:
            w = cupy.atleast_1d(worN)
    s = 1j * w
    h = cupy.polyval(b, s) / cupy.polyval(a, s)
    if plot is not None:
        plot(w, h)
    return (w, h)

def freqs_zpk(z, p, k, worN=200):
    if False:
        return 10
    '\n    Compute frequency response of analog filter.\n\n    Given the zeros `z`, poles `p`, and gain `k` of a filter, compute its\n    frequency response::\n\n                (jw-z[0]) * (jw-z[1]) * ... * (jw-z[-1])\n     H(w) = k * ----------------------------------------\n                (jw-p[0]) * (jw-p[1]) * ... * (jw-p[-1])\n\n    Parameters\n    ----------\n    z : array_like\n        Zeroes of a linear filter\n    p : array_like\n        Poles of a linear filter\n    k : scalar\n        Gain of a linear filter\n    worN : {None, int, array_like}, optional\n        If None, then compute at 200 frequencies around the interesting parts\n        of the response curve (determined by pole-zero locations). If a single\n        integer, then compute at that many frequencies. Otherwise, compute the\n        response at the angular frequencies (e.g., rad/s) given in `worN`.\n\n    Returns\n    -------\n    w : ndarray\n        The angular frequencies at which `h` was computed.\n    h : ndarray\n        The frequency response.\n\n    See Also\n    --------\n    scipy.signal.freqs_zpk\n\n    '
    k = cupy.asarray(k)
    if k.size > 1:
        raise ValueError('k must be a single scalar gain')
    if worN is None:
        w = findfreqs(z, p, 200, kind='zp')
    else:
        (N, _is_int) = _try_convert_to_int(worN)
        if _is_int:
            w = findfreqs(z, p, worN, kind='zp')
        else:
            w = worN
    w = cupy.atleast_1d(w)
    s = 1j * w
    num = npp_polyvalfromroots(s, z)
    den = npp_polyvalfromroots(s, p)
    h = k * num / den
    return (w, h)

def _is_int_type(x):
    if False:
        return 10
    '\n    Check if input is of a scalar integer type (so ``5`` and ``array(5)`` will\n    pass, while ``5.0`` and ``array([5])`` will fail.\n    '
    if cupy.ndim(x) != 0:
        return False
    try:
        operator.index(x)
    except TypeError:
        return False
    else:
        return True

def group_delay(system, w=512, whole=False, fs=2 * cupy.pi):
    if False:
        while True:
            i = 10
    'Compute the group delay of a digital filter.\n\n    The group delay measures by how many samples amplitude envelopes of\n    various spectral components of a signal are delayed by a filter.\n    It is formally defined as the derivative of continuous (unwrapped) phase::\n\n               d        jw\n     D(w) = - -- arg H(e)\n              dw\n\n    Parameters\n    ----------\n    system : tuple of array_like (b, a)\n        Numerator and denominator coefficients of a filter transfer function.\n    w : {None, int, array_like}, optional\n        If a single integer, then compute at that many frequencies (default is\n        N=512).\n\n        If an array_like, compute the delay at the frequencies given. These\n        are in the same units as `fs`.\n    whole : bool, optional\n        Normally, frequencies are computed from 0 to the Nyquist frequency,\n        fs/2 (upper-half of unit-circle). If `whole` is True, compute\n        frequencies from 0 to fs. Ignored if w is array_like.\n    fs : float, optional\n        The sampling frequency of the digital system. Defaults to 2*pi\n        radians/sample (so w is from 0 to pi).\n\n    Returns\n    -------\n    w : ndarray\n        The frequencies at which group delay was computed, in the same units\n        as `fs`.  By default, `w` is normalized to the range [0, pi)\n        (radians/sample).\n    gd : ndarray\n        The group delay.\n\n    See Also\n    --------\n    freqz : Frequency response of a digital filter\n\n    Notes\n    -----\n    The similar function in MATLAB is called `grpdelay`.\n\n    If the transfer function :math:`H(z)` has zeros or poles on the unit\n    circle, the group delay at corresponding frequencies is undefined.\n    When such a case arises the warning is raised and the group delay\n    is set to 0 at those frequencies.\n\n    For the details of numerical computation of the group delay refer to [1]_.\n\n    References\n    ----------\n    .. [1] Richard G. Lyons, "Understanding Digital Signal Processing,\n           3rd edition", p. 830.\n\n    '
    if w is None:
        w = 512
    if _is_int_type(w):
        if whole:
            w = cupy.linspace(0, 2 * cupy.pi, w, endpoint=False)
        else:
            w = cupy.linspace(0, cupy.pi, w, endpoint=False)
    else:
        w = cupy.atleast_1d(w)
        w = 2 * cupy.pi * w / fs
    (b, a) = map(cupy.atleast_1d, system)
    c = cupy.convolve(b, a[::-1])
    cr = c * cupy.arange(c.size)
    z = cupy.exp(-1j * w)
    num = cupy.polyval(cr[::-1], z)
    den = cupy.polyval(c[::-1], z)
    gd = cupy.real(num / den) - a.size + 1
    singular = ~cupy.isfinite(gd)
    gd[singular] = 0
    w = w * fs / (2 * cupy.pi)
    return (w, gd)

def freqz(b, a=1, worN=512, whole=False, plot=None, fs=2 * pi, include_nyquist=False):
    if False:
        i = 10
        return i + 15
    "\n    Compute the frequency response of a digital filter.\n\n    Given the M-order numerator `b` and N-order denominator `a` of a digital\n    filter, compute its frequency response::\n\n                 jw                 -jw              -jwM\n        jw    B(e  )    b[0] + b[1]e    + ... + b[M]e\n     H(e  ) = ------ = -----------------------------------\n                 jw                 -jw              -jwN\n              A(e  )    a[0] + a[1]e    + ... + a[N]e\n\n    Parameters\n    ----------\n    b : array_like\n        Numerator of a linear filter. If `b` has dimension greater than 1,\n        it is assumed that the coefficients are stored in the first dimension,\n        and ``b.shape[1:]``, ``a.shape[1:]``, and the shape of the frequencies\n        array must be compatible for broadcasting.\n    a : array_like\n        Denominator of a linear filter. If `b` has dimension greater than 1,\n        it is assumed that the coefficients are stored in the first dimension,\n        and ``b.shape[1:]``, ``a.shape[1:]``, and the shape of the frequencies\n        array must be compatible for broadcasting.\n    worN : {None, int, array_like}, optional\n        If a single integer, then compute at that many frequencies (default is\n        N=512). This is a convenient alternative to::\n\n            cupy.linspace(0, fs if whole else fs/2, N,\n                          endpoint=include_nyquist)\n\n        Using a number that is fast for FFT computations can result in\n        faster computations (see Notes).\n\n        If an array_like, compute the response at the frequencies given.\n        These are in the same units as `fs`.\n    whole : bool, optional\n        Normally, frequencies are computed from 0 to the Nyquist frequency,\n        fs/2 (upper-half of unit-circle). If `whole` is True, compute\n        frequencies from 0 to fs. Ignored if worN is array_like.\n    plot : callable\n        A callable that takes two arguments. If given, the return parameters\n        `w` and `h` are passed to plot. Useful for plotting the frequency\n        response inside `freqz`.\n    fs : float, optional\n        The sampling frequency of the digital system. Defaults to 2*pi\n        radians/sample (so w is from 0 to pi).\n    include_nyquist : bool, optional\n        If `whole` is False and `worN` is an integer, setting `include_nyquist`\n        to True will include the last frequency (Nyquist frequency) and is\n        otherwise ignored.\n\n    Returns\n    -------\n    w : ndarray\n        The frequencies at which `h` was computed, in the same units as `fs`.\n        By default, `w` is normalized to the range [0, pi) (radians/sample).\n    h : ndarray\n        The frequency response, as complex numbers.\n\n    See Also\n    --------\n    freqz_zpk\n    sosfreqz\n    scipy.signal.freqz\n\n\n    Notes\n    -----\n    Using Matplotlib's :func:`matplotlib.pyplot.plot` function as the callable\n    for `plot` produces unexpected results, as this plots the real part of the\n    complex transfer function, not the magnitude.\n    Try ``lambda w, h: plot(w, cupy.abs(h))``.\n\n    A direct computation via (R)FFT is used to compute the frequency response\n    when the following conditions are met:\n\n    1. An integer value is given for `worN`.\n    2. `worN` is fast to compute via FFT (i.e.,\n       `next_fast_len(worN) <scipy.fft.next_fast_len>` equals `worN`).\n    3. The denominator coefficients are a single value (``a.shape[0] == 1``).\n    4. `worN` is at least as long as the numerator coefficients\n       (``worN >= b.shape[0]``).\n    5. If ``b.ndim > 1``, then ``b.shape[-1] == 1``.\n\n    For long FIR filters, the FFT approach can have lower error and be much\n    faster than the equivalent direct polynomial calculation.\n    "
    b = cupy.atleast_1d(b)
    a = cupy.atleast_1d(a)
    if worN is None:
        worN = 512
    h = None
    (N, _is_int) = _try_convert_to_int(worN)
    if _is_int:
        if N < 0:
            raise ValueError(f'worN must be nonnegative, got {N}')
        lastpoint = 2 * pi if whole else pi
        w = cupy.linspace(0, lastpoint, N, endpoint=include_nyquist and (not whole))
        use_fft = a.size == 1 and N >= b.shape[0] and (sp_fft.next_fast_len(N) == N) and (b.ndim == 1 or b.shape[-1] == 1)
        if use_fft:
            n_fft = N if whole else N * 2
            if cupy.isrealobj(b) and cupy.isrealobj(a):
                fft_func = sp_fft.rfft
            else:
                fft_func = sp_fft.fft
            h = fft_func(b, n=n_fft, axis=0)[:N]
            h /= a
            if fft_func is sp_fft.rfft and whole:
                stop = -1 if n_fft % 2 == 1 else -2
                h_flip = slice(stop, 0, -1)
                h = cupy.concatenate((h, h[h_flip].conj()))
            if b.ndim > 1:
                h = h[..., 0]
                h = cupy.moveaxis(h, 0, -1)
    else:
        w = cupy.atleast_1d(worN)
        w = 2 * pi * w / fs
    if h is None:
        zm1 = cupy.exp(-1j * w)
        h = npp_polyval(zm1, b, tensor=False) / npp_polyval(zm1, a, tensor=False)
    w = w * fs / (2 * pi)
    if plot is not None:
        plot(w, h)
    return (w, h)

def freqz_zpk(z, p, k, worN=512, whole=False, fs=2 * pi):
    if False:
        i = 10
        return i + 15
    '\n    Compute the frequency response of a digital filter in ZPK form.\n\n    Given the Zeros, Poles and Gain of a digital filter, compute its frequency\n    response:\n\n    :math:`H(z)=k \\prod_i (z - Z[i]) / \\prod_j (z - P[j])`\n\n    where :math:`k` is the `gain`, :math:`Z` are the `zeros` and :math:`P` are\n    the `poles`.\n\n    Parameters\n    ----------\n    z : array_like\n        Zeroes of a linear filter\n    p : array_like\n        Poles of a linear filter\n    k : scalar\n        Gain of a linear filter\n    worN : {None, int, array_like}, optional\n        If a single integer, then compute at that many frequencies (default is\n        N=512).\n\n        If an array_like, compute the response at the frequencies given.\n        These are in the same units as `fs`.\n    whole : bool, optional\n        Normally, frequencies are computed from 0 to the Nyquist frequency,\n        fs/2 (upper-half of unit-circle). If `whole` is True, compute\n        frequencies from 0 to fs. Ignored if w is array_like.\n    fs : float, optional\n        The sampling frequency of the digital system. Defaults to 2*pi\n        radians/sample (so w is from 0 to pi).\n\n    Returns\n    -------\n    w : ndarray\n        The frequencies at which `h` was computed, in the same units as `fs`.\n        By default, `w` is normalized to the range [0, pi) (radians/sample).\n    h : ndarray\n        The frequency response, as complex numbers.\n\n    See Also\n    --------\n    freqs : Compute the frequency response of an analog filter in TF form\n    freqs_zpk : Compute the frequency response of an analog filter in ZPK form\n    freqz : Compute the frequency response of a digital filter in TF form\n    scipy.signal.freqz_zpk\n\n    '
    (z, p) = map(cupy.atleast_1d, (z, p))
    if whole:
        lastpoint = 2 * pi
    else:
        lastpoint = pi
    if worN is None:
        w = cupy.linspace(0, lastpoint, 512, endpoint=False)
    else:
        (N, _is_int) = _try_convert_to_int(worN)
        if _is_int:
            w = cupy.linspace(0, lastpoint, N, endpoint=False)
        else:
            w = cupy.atleast_1d(worN)
            w = 2 * pi * w / fs
    zm1 = cupy.exp(1j * w)
    h = k * npp_polyvalfromroots(zm1, z) / npp_polyvalfromroots(zm1, p)
    w = w * fs / (2 * pi)
    return (w, h)

def _validate_sos(sos):
    if False:
        while True:
            i = 10
    'Helper to validate a SOS input'
    sos = cupy.atleast_2d(sos)
    if sos.ndim != 2:
        raise ValueError('sos array must be 2D')
    (n_sections, m) = sos.shape
    if m != 6:
        raise ValueError('sos array must be shape (n_sections, 6)')
    if (sos[:, 3] - 1 > 1e-15).any():
        raise ValueError('sos[:, 3] should be all ones')
    return (sos, n_sections)

def sosfreqz(sos, worN=512, whole=False, fs=2 * pi):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the frequency response of a digital filter in SOS format.\n\n    Given `sos`, an array with shape (n, 6) of second order sections of\n    a digital filter, compute the frequency response of the system function::\n\n               B0(z)   B1(z)         B{n-1}(z)\n        H(z) = ----- * ----- * ... * ---------\n               A0(z)   A1(z)         A{n-1}(z)\n\n    for z = exp(omega*1j), where B{k}(z) and A{k}(z) are numerator and\n    denominator of the transfer function of the k-th second order section.\n\n    Parameters\n    ----------\n    sos : array_like\n        Array of second-order filter coefficients, must have shape\n        ``(n_sections, 6)``. Each row corresponds to a second-order\n        section, with the first three columns providing the numerator\n        coefficients and the last three providing the denominator\n        coefficients.\n    worN : {None, int, array_like}, optional\n        If a single integer, then compute at that many frequencies (default is\n        N=512).  Using a number that is fast for FFT computations can result\n        in faster computations (see Notes of `freqz`).\n\n        If an array_like, compute the response at the frequencies given (must\n        be 1-D). These are in the same units as `fs`.\n    whole : bool, optional\n        Normally, frequencies are computed from 0 to the Nyquist frequency,\n        fs/2 (upper-half of unit-circle). If `whole` is True, compute\n        frequencies from 0 to fs.\n    fs : float, optional\n        The sampling frequency of the digital system. Defaults to 2*pi\n        radians/sample (so w is from 0 to pi).\n\n        .. versionadded:: 1.2.0\n\n    Returns\n    -------\n    w : ndarray\n        The frequencies at which `h` was computed, in the same units as `fs`.\n        By default, `w` is normalized to the range [0, pi) (radians/sample).\n    h : ndarray\n        The frequency response, as complex numbers.\n\n    See Also\n    --------\n    freqz, sosfilt\n    scipy.signal.sosfreqz\n    '
    (sos, n_sections) = _validate_sos(sos)
    if n_sections == 0:
        raise ValueError('Cannot compute frequencies with no sections')
    h = 1.0
    for row in sos:
        (w, rowh) = freqz(row[:3], row[3:], worN=worN, whole=whole, fs=fs)
        h *= rowh
    return (w, h)

def _hz_to_erb(hz):
    if False:
        return 10
    '\n    Utility for converting from frequency (Hz) to the\n    Equivalent Rectangular Bandwidth (ERB) scale\n    ERB = frequency / EarQ + minBW\n    '
    EarQ = 9.26449
    minBW = 24.7
    return hz / EarQ + minBW

@jit.rawkernel()
def _gammatone_iir_kernel(fs, freq, b, a):
    if False:
        while True:
            i = 10
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    EarQ = 9.26449
    minBW = 24.7
    erb = freq / EarQ + minBW
    T = 1.0 / fs
    bw = 2 * cupy.pi * 1.019 * erb
    fr = 2 * freq * cupy.pi * T
    bwT = bw * T
    g1 = -2 * cupy.exp(2j * fr) * T
    g2 = 2 * cupy.exp(-bwT + 1j * fr) * T
    g3 = cupy.sqrt(3 + 2 ** (3 / 2)) * cupy.sin(fr)
    g4 = cupy.sqrt(3 - 2 ** (3 / 2)) * cupy.sin(fr)
    g5 = cupy.exp(2j * fr)
    g = g1 + g2 * (cupy.cos(fr) - g4)
    g *= g1 + g2 * (cupy.cos(fr) + g4)
    g *= g1 + g2 * (cupy.cos(fr) - g3)
    g *= g1 + g2 * (cupy.cos(fr) + g3)
    g /= (-2 / cupy.exp(2 * bwT) - 2 * g5 + 2 * (1 + g5) / cupy.exp(bwT)) ** 4
    g_act = cupy.abs(g)
    if tid == 0:
        b[tid] = T ** 4 / g_act
        a[tid] = 1
    elif tid == 1:
        b[tid] = -4 * T ** 4 * cupy.cos(fr) / cupy.exp(bw * T) / g_act
        a[tid] = -8 * cupy.cos(fr) / cupy.exp(bw * T)
    elif tid == 2:
        b[tid] = 6 * T ** 4 * cupy.cos(2 * fr) / cupy.exp(2 * bw * T) / g_act
        a[tid] = 4 * (4 + 3 * cupy.cos(2 * fr)) / cupy.exp(2 * bw * T)
    elif tid == 3:
        b[tid] = -4 * T ** 4 * cupy.cos(3 * fr) / cupy.exp(3 * bw * T) / g_act
        a[tid] = -8 * (6 * cupy.cos(fr) + cupy.cos(3 * fr))
        a[tid] /= cupy.exp(3 * bw * T)
    elif tid == 4:
        b[tid] = T ** 4 * cupy.cos(4 * fr) / cupy.exp(4 * bw * T) / g_act
        a[tid] = 2 * (18 + 16 * cupy.cos(2 * fr) + cupy.cos(4 * fr))
        a[tid] /= cupy.exp(4 * bw * T)
    elif tid == 5:
        a[tid] = -8 * (6 * cupy.cos(fr) + cupy.cos(3 * fr))
        a[tid] /= cupy.exp(5 * bw * T)
    elif tid == 6:
        a[tid] = 4 * (4 + 3 * cupy.cos(2 * fr)) / cupy.exp(6 * bw * T)
    elif tid == 7:
        a[tid] = -8 * cupy.cos(fr) / cupy.exp(7 * bw * T)
    elif tid == 8:
        a[tid] = cupy.exp(-8 * bw * T)

def gammatone(freq, ftype, order=None, numtaps=None, fs=None):
    if False:
        i = 10
        return i + 15
    '\n    Gammatone filter design.\n\n    This function computes the coefficients of an FIR or IIR gammatone\n    digital filter [1]_.\n\n    Parameters\n    ----------\n    freq : float\n        Center frequency of the filter (expressed in the same units\n        as `fs`).\n    ftype : {\'fir\', \'iir\'}\n        The type of filter the function generates. If \'fir\', the function\n        will generate an Nth order FIR gammatone filter. If \'iir\', the\n        function will generate an 8th order digital IIR filter, modeled as\n        as 4th order gammatone filter.\n    order : int, optional\n        The order of the filter. Only used when ``ftype=\'fir\'``.\n        Default is 4 to model the human auditory system. Must be between\n        0 and 24.\n    numtaps : int, optional\n        Length of the filter. Only used when ``ftype=\'fir\'``.\n        Default is ``fs*0.015`` if `fs` is greater than 1000,\n        15 if `fs` is less than or equal to 1000.\n    fs : float, optional\n        The sampling frequency of the signal. `freq` must be between\n        0 and ``fs/2``. Default is 2.\n\n    Returns\n    -------\n    b, a : ndarray, ndarray\n        Numerator (``b``) and denominator (``a``) polynomials of the filter.\n\n    Raises\n    ------\n    ValueError\n        If `freq` is less than or equal to 0 or greater than or equal to\n        ``fs/2``, if `ftype` is not \'fir\' or \'iir\', if `order` is less than\n        or equal to 0 or greater than 24 when ``ftype=\'fir\'``\n\n    See Also\n    --------\n    firwin\n    iirfilter\n\n    References\n    ----------\n    .. [1] Slaney, Malcolm, "An Efficient Implementation of the\n        Patterson-Holdsworth Auditory Filter Bank", Apple Computer\n        Technical Report 35, 1993, pp.3-8, 34-39.\n    '
    freq = float(freq)
    if fs is None:
        fs = 2
    fs = float(fs)
    ftype = ftype.lower()
    filter_types = ['fir', 'iir']
    if not 0 < freq < fs / 2:
        raise ValueError('The frequency must be between 0 and {} (nyquist), but given {}.'.format(fs / 2, freq))
    if ftype not in filter_types:
        raise ValueError('ftype must be either fir or iir.')
    if ftype == 'fir':
        if order is None:
            order = 4
        order = operator.index(order)
        if numtaps is None:
            numtaps = max(int(fs * 0.015), 15)
        numtaps = operator.index(numtaps)
        if not 0 < order <= 24:
            raise ValueError('Invalid order: order must be > 0 and <= 24.')
        t = cupy.arange(numtaps) / fs
        bw = 1.019 * _hz_to_erb(freq)
        b = t ** (order - 1) * cupy.exp(-2 * cupy.pi * bw * t)
        b *= cupy.cos(2 * cupy.pi * freq * t)
        scale_factor = 2 * (2 * cupy.pi * bw) ** order
        scale_factor /= float_factorial(order - 1)
        scale_factor /= fs
        b *= scale_factor
        a = [1.0]
    elif ftype == 'iir':
        if order is not None:
            warnings.warn('order is not used for IIR gammatone filter.')
        if numtaps is not None:
            warnings.warn('numtaps is not used for IIR gammatone filter.')
        b = cupy.empty(5)
        a = cupy.empty(9)
        _gammatone_iir_kernel((9,), (1,), (fs, freq, b, a))
    return (b, a)