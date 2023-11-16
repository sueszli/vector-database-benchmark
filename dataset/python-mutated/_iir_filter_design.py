"""IIR filter design APIs"""
from math import pi
import math
import cupy
from cupyx.scipy.signal._iir_filter_conversions import lp2bp_zpk, lp2lp_zpk, lp2hp_zpk, lp2bs_zpk, bilinear_zpk, zpk2tf, zpk2sos
from cupyx.scipy.signal._iir_filter_conversions import buttap, cheb1ap, cheb2ap, ellipap, buttord, ellipord, cheb1ord, cheb2ord, _validate_gpass_gstop

def besselap():
    if False:
        while True:
            i = 10
    raise NotImplementedError
bessel_norms = {'fix': 'me'}

def iirfilter(N, Wn, rp=None, rs=None, btype='band', analog=False, ftype='butter', output='ba', fs=None):
    if False:
        i = 10
        return i + 15
    "\n    IIR digital and analog filter design given order and critical points.\n\n    Design an Nth-order digital or analog filter and return the filter\n    coefficients.\n\n    Parameters\n    ----------\n    N : int\n        The order of the filter.\n    Wn : array_like\n        A scalar or length-2 sequence giving the critical frequencies.\n\n        For digital filters, `Wn` are in the same units as `fs`. By default,\n        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,\n        where 1 is the Nyquist frequency. (`Wn` is thus in\n        half-cycles / sample.)\n\n        For analog filters, `Wn` is an angular frequency (e.g., rad/s).\n\n        When Wn is a length-2 sequence, ``Wn[0]`` must be less than ``Wn[1]``.\n    rp : float, optional\n        For Chebyshev and elliptic filters, provides the maximum ripple\n        in the passband. (dB)\n    rs : float, optional\n        For Chebyshev and elliptic filters, provides the minimum attenuation\n        in the stop band. (dB)\n    btype : {'bandpass', 'lowpass', 'highpass', 'bandstop'}, optional\n        The type of filter.  Default is 'bandpass'.\n    analog : bool, optional\n        When True, return an analog filter, otherwise a digital filter is\n        returned.\n    ftype : str, optional\n        The type of IIR filter to design:\n\n            - Butterworth   : 'butter'\n            - Chebyshev I   : 'cheby1'\n            - Chebyshev II  : 'cheby2'\n            - Cauer/elliptic: 'ellip'\n            - Bessel/Thomson: 'bessel'\n\n    output : {'ba', 'zpk', 'sos'}, optional\n        Filter form of the output:\n\n            - second-order sections (recommended): 'sos'\n            - numerator/denominator (default)    : 'ba'\n            - pole-zero                          : 'zpk'\n\n        In general the second-order sections ('sos') form  is\n        recommended because inferring the coefficients for the\n        numerator/denominator form ('ba') suffers from numerical\n        instabilities. For reasons of backward compatibility the default\n        form is the numerator/denominator form ('ba'), where the 'b'\n        and the 'a' in 'ba' refer to the commonly used names of the\n        coefficients used.\n\n        Note: Using the second-order sections form ('sos') is sometimes\n        associated with additional computational costs: for\n        data-intense use cases it is therefore recommended to also\n        investigate the numerator/denominator form ('ba').\n\n    fs : float, optional\n        The sampling frequency of the digital system.\n\n    Returns\n    -------\n    b, a : ndarray, ndarray\n        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.\n        Only returned if ``output='ba'``.\n    z, p, k : ndarray, ndarray, float\n        Zeros, poles, and system gain of the IIR filter transfer\n        function.  Only returned if ``output='zpk'``.\n    sos : ndarray\n        Second-order sections representation of the IIR filter.\n        Only returned if ``output='sos'``.\n\n    See Also\n    --------\n    butter : Filter design using order and critical points\n    cheby1, cheby2, ellip, bessel\n    buttord : Find order and critical points from passband and stopband spec\n    cheb1ord, cheb2ord, ellipord\n    iirdesign : General filter design using passband and stopband spec\n    scipy.signal.iirfilter\n\n    "
    (ftype, btype, output) = [x.lower() for x in (ftype, btype, output)]
    Wn = cupy.asarray(Wn)
    if Wn.size > 1 and (not Wn[0] < Wn[1]):
        raise ValueError('Wn[0] must be less than Wn[1]')
    if fs is not None:
        if analog:
            raise ValueError('fs cannot be specified for an analog filter')
        Wn = 2 * Wn / fs
    try:
        btype = band_dict[btype]
    except KeyError as e:
        raise ValueError("'%s' is an invalid bandtype for filter." % btype) from e
    try:
        typefunc = filter_dict[ftype][0]
    except KeyError as e:
        raise ValueError("'%s' is not a valid basic IIR filter." % ftype) from e
    if output not in ['ba', 'zpk', 'sos']:
        raise ValueError("'%s' is not a valid output form." % output)
    if rp is not None and rp < 0:
        raise ValueError('passband ripple (rp) must be positive')
    if rs is not None and rs < 0:
        raise ValueError('stopband attenuation (rs) must be positive')
    if typefunc == buttap:
        (z, p, k) = typefunc(N)
    elif typefunc == besselap:
        (z, p, k) = typefunc(N, norm=bessel_norms[ftype])
    elif typefunc == cheb1ap:
        if rp is None:
            raise ValueError('passband ripple (rp) must be provided to design a Chebyshev I filter.')
        (z, p, k) = typefunc(N, rp)
    elif typefunc == cheb2ap:
        if rs is None:
            raise ValueError('stopband attenuation (rs) must be provided to design an Chebyshev II filter.')
        (z, p, k) = typefunc(N, rs)
    elif typefunc == ellipap:
        if rs is None or rp is None:
            raise ValueError('Both rp and rs must be provided to design an elliptic filter.')
        (z, p, k) = typefunc(N, rp, rs)
    else:
        raise NotImplementedError("'%s' not implemented in iirfilter." % ftype)
    if not analog:
        if cupy.any(Wn <= 0) or cupy.any(Wn >= 1):
            if fs is not None:
                raise ValueError(f'Digital filter critical frequencies must be 0 < Wn < fs/2 (fs={fs} -> fs/2={fs / 2})')
            raise ValueError('Digital filter critical frequencies must be 0 < Wn < 1')
        fs = 2.0
        warped = 2 * fs * cupy.tan(pi * Wn / fs)
    else:
        warped = Wn
    if btype in ('lowpass', 'highpass'):
        if cupy.size(Wn) != 1:
            raise ValueError('Must specify a single critical frequency Wn for lowpass or highpass filter')
        if btype == 'lowpass':
            (z, p, k) = lp2lp_zpk(z, p, k, wo=warped)
        elif btype == 'highpass':
            (z, p, k) = lp2hp_zpk(z, p, k, wo=warped)
    elif btype in ('bandpass', 'bandstop'):
        try:
            bw = warped[1] - warped[0]
            wo = cupy.sqrt(warped[0] * warped[1])
        except IndexError as e:
            raise ValueError('Wn must specify start and stop frequencies for bandpass or bandstop filter') from e
        if btype == 'bandpass':
            (z, p, k) = lp2bp_zpk(z, p, k, wo=wo, bw=bw)
        elif btype == 'bandstop':
            (z, p, k) = lp2bs_zpk(z, p, k, wo=wo, bw=bw)
    else:
        raise NotImplementedError("'%s' not implemented in iirfilter." % btype)
    if not analog:
        (z, p, k) = bilinear_zpk(z, p, k, fs=fs)
    if output == 'zpk':
        return (z, p, k)
    elif output == 'ba':
        return zpk2tf(z, p, k)
    elif output == 'sos':
        return zpk2sos(z, p, k, analog=analog)

def butter(N, Wn, btype='low', analog=False, output='ba', fs=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Butterworth digital and analog filter design.\n\n    Design an Nth-order digital or analog Butterworth filter and return\n    the filter coefficients.\n\n    Parameters\n    ----------\n    N : int\n        The order of the filter. For \'bandpass\' and \'bandstop\' filters,\n        the resulting order of the final second-order sections (\'sos\')\n        matrix is ``2*N``, with `N` the number of biquad sections\n        of the desired system.\n    Wn : array_like\n        The critical frequency or frequencies. For lowpass and highpass\n        filters, Wn is a scalar; for bandpass and bandstop filters,\n        Wn is a length-2 sequence.\n\n        For a Butterworth filter, this is the point at which the gain\n        drops to 1/sqrt(2) that of the passband (the "-3 dB point").\n\n        For digital filters, if `fs` is not specified, `Wn` units are\n        normalized from 0 to 1, where 1 is the Nyquist frequency (`Wn` is\n        thus in half cycles / sample and defined as 2*critical frequencies\n        / `fs`). If `fs` is specified, `Wn` is in the same units as `fs`.\n\n        For analog filters, `Wn` is an angular frequency (e.g. rad/s).\n    btype : {\'lowpass\', \'highpass\', \'bandpass\', \'bandstop\'}, optional\n        The type of filter.  Default is \'lowpass\'.\n    analog : bool, optional\n        When True, return an analog filter, otherwise a digital filter is\n        returned.\n    output : {\'ba\', \'zpk\', \'sos\'}, optional\n        Type of output:  numerator/denominator (\'ba\'), pole-zero (\'zpk\'), or\n        second-order sections (\'sos\'). Default is \'ba\' for backwards\n        compatibility, but \'sos\' should be used for general-purpose filtering.\n    fs : float, optional\n        The sampling frequency of the digital system.\n\n    Returns\n    -------\n    b, a : ndarray, ndarray\n        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.\n        Only returned if ``output=\'ba\'``.\n    z, p, k : ndarray, ndarray, float\n        Zeros, poles, and system gain of the IIR filter transfer\n        function.  Only returned if ``output=\'zpk\'``.\n    sos : ndarray\n        Second-order sections representation of the IIR filter.\n        Only returned if ``output=\'sos\'``.\n\n    See Also\n    --------\n    buttord, buttap\n    iirfilter\n    scipy.signal.butter\n\n\n    Notes\n    -----\n    The Butterworth filter has maximally flat frequency response in the\n    passband.\n\n    If the transfer function form ``[b, a]`` is requested, numerical\n    problems can occur since the conversion between roots and\n    the polynomial coefficients is a numerically sensitive operation,\n    even for N >= 4. It is recommended to work with the SOS\n    representation.\n\n    .. warning::\n        Designing high-order and narrowband IIR filters in TF form can\n        result in unstable or incorrect filtering due to floating point\n        numerical precision issues. Consider inspecting output filter\n        characteristics `freqz` or designing the filters with second-order\n        sections via ``output=\'sos\'``.\n    '
    return iirfilter(N, Wn, btype=btype, analog=analog, output=output, ftype='butter', fs=fs)

def cheby1(N, rp, Wn, btype='low', analog=False, output='ba', fs=None):
    if False:
        print('Hello World!')
    "\n    Chebyshev type I digital and analog filter design.\n\n    Design an Nth-order digital or analog Chebyshev type I filter and\n    return the filter coefficients.\n\n    Parameters\n    ----------\n    N : int\n        The order of the filter.\n    rp : float\n        The maximum ripple allowed below unity gain in the passband.\n        Specified in decibels, as a positive number.\n    Wn : array_like\n        A scalar or length-2 sequence giving the critical frequencies.\n        For Type I filters, this is the point in the transition band at which\n        the gain first drops below -`rp`.\n\n        For digital filters, `Wn` are in the same units as `fs`. By default,\n        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,\n        where 1 is the Nyquist frequency. (`Wn` is thus in\n        half-cycles / sample.)\n\n        For analog filters, `Wn` is an angular frequency (e.g., rad/s).\n    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional\n        The type of filter.  Default is 'lowpass'.\n    analog : bool, optional\n        When True, return an analog filter, otherwise a digital filter is\n        returned.\n    output : {'ba', 'zpk', 'sos'}, optional\n        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or\n        second-order sections ('sos'). Default is 'ba' for backwards\n        compatibility, but 'sos' should be used for general-purpose filtering.\n    fs : float, optional\n        The sampling frequency of the digital system.\n\n    Returns\n    -------\n    b, a : ndarray, ndarray\n        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.\n        Only returned if ``output='ba'``.\n    z, p, k : ndarray, ndarray, float\n        Zeros, poles, and system gain of the IIR filter transfer\n        function.  Only returned if ``output='zpk'``.\n    sos : ndarray\n        Second-order sections representation of the IIR filter.\n        Only returned if ``output='sos'``.\n\n    See Also\n    --------\n    cheb1ord, cheb1ap\n    iirfilter\n    scipy.signal.cheby1\n\n    Notes\n    -----\n    The Chebyshev type I filter maximizes the rate of cutoff between the\n    frequency response's passband and stopband, at the expense of ripple in\n    the passband and increased ringing in the step response.\n\n    Type I filters roll off faster than Type II (`cheby2`), but Type II\n    filters do not have any ripple in the passband.\n\n    The equiripple passband has N maxima or minima (for example, a\n    5th-order filter has 3 maxima and 2 minima). Consequently, the DC gain is\n    unity for odd-order filters, or -rp dB for even-order filters.\n    "
    return iirfilter(N, Wn, rp=rp, btype=btype, analog=analog, output=output, ftype='cheby1', fs=fs)

def cheby2(N, rs, Wn, btype='low', analog=False, output='ba', fs=None):
    if False:
        while True:
            i = 10
    "\n    Chebyshev type II digital and analog filter design.\n\n    Design an Nth-order digital or analog Chebyshev type II filter and\n    return the filter coefficients.\n\n    Parameters\n    ----------\n    N : int\n        The order of the filter.\n    rs : float\n        The minimum attenuation required in the stop band.\n        Specified in decibels, as a positive number.\n    Wn : array_like\n        A scalar or length-2 sequence giving the critical frequencies.\n        For Type II filters, this is the point in the transition band at which\n        the gain first reaches -`rs`.\n\n        For digital filters, `Wn` are in the same units as `fs`. By default,\n        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,\n        where 1 is the Nyquist frequency. (`Wn` is thus in\n        half-cycles / sample.)\n\n        For analog filters, `Wn` is an angular frequency (e.g., rad/s).\n    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional\n        The type of filter.  Default is 'lowpass'.\n    analog : bool, optional\n        When True, return an analog filter, otherwise a digital filter is\n        returned.\n    output : {'ba', 'zpk', 'sos'}, optional\n        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or\n        second-order sections ('sos'). Default is 'ba' for backwards\n        compatibility, but 'sos' should be used for general-purpose filtering.\n    fs : float, optional\n        The sampling frequency of the digital system.\n\n    Returns\n    -------\n    b, a : ndarray, ndarray\n        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.\n        Only returned if ``output='ba'``.\n    z, p, k : ndarray, ndarray, float\n        Zeros, poles, and system gain of the IIR filter transfer\n        function.  Only returned if ``output='zpk'``.\n    sos : ndarray\n        Second-order sections representation of the IIR filter.\n        Only returned if ``output='sos'``.\n\n    See Also\n    --------\n    cheb2ord, cheb2ap\n    iirfilter\n    scipy.signal.cheby2\n\n    Notes\n    -----\n    The Chebyshev type II filter maximizes the rate of cutoff between the\n    frequency response's passband and stopband, at the expense of ripple in\n    the stopband and increased ringing in the step response.\n\n    Type II filters do not roll off as fast as Type I (`cheby1`).\n    "
    return iirfilter(N, Wn, rs=rs, btype=btype, analog=analog, output=output, ftype='cheby2', fs=fs)

def ellip(N, rp, rs, Wn, btype='low', analog=False, output='ba', fs=None):
    if False:
        i = 10
        return i + 15
    "\n    Elliptic (Cauer) digital and analog filter design.\n\n    Design an Nth-order digital or analog elliptic filter and return\n    the filter coefficients.\n\n    Parameters\n    ----------\n    N : int\n        The order of the filter.\n    rp : float\n        The maximum ripple allowed below unity gain in the passband.\n        Specified in decibels, as a positive number.\n    rs : float\n        The minimum attenuation required in the stop band.\n        Specified in decibels, as a positive number.\n    Wn : array_like\n        A scalar or length-2 sequence giving the critical frequencies.\n        For elliptic filters, this is the point in the transition band at\n        which the gain first drops below -`rp`.\n\n        For digital filters, `Wn` are in the same units as `fs`. By default,\n        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,\n        where 1 is the Nyquist frequency. (`Wn` is thus in\n        half-cycles / sample.)\n\n        For analog filters, `Wn` is an angular frequency (e.g., rad/s).\n    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional\n        The type of filter. Default is 'lowpass'.\n    analog : bool, optional\n        When True, return an analog filter, otherwise a digital filter is\n        returned.\n    output : {'ba', 'zpk', 'sos'}, optional\n        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or\n        second-order sections ('sos'). Default is 'ba' for backwards\n        compatibility, but 'sos' should be used for general-purpose filtering.\n    fs : float, optional\n        The sampling frequency of the digital system.\n\n    Returns\n    -------\n    b, a : ndarray, ndarray\n        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.\n        Only returned if ``output='ba'``.\n    z, p, k : ndarray, ndarray, float\n        Zeros, poles, and system gain of the IIR filter transfer\n        function.  Only returned if ``output='zpk'``.\n    sos : ndarray\n        Second-order sections representation of the IIR filter.\n        Only returned if ``output='sos'``.\n\n    See Also\n    --------\n    ellipord, ellipap\n    iirfilter\n    scipy.signal.ellip\n\n    Notes\n    -----\n    Also known as Cauer or Zolotarev filters, the elliptical filter maximizes\n    the rate of transition between the frequency response's passband and\n    stopband, at the expense of ripple in both, and increased ringing in the\n    step response.\n\n    As `rp` approaches 0, the elliptical filter becomes a Chebyshev\n    type II filter (`cheby2`). As `rs` approaches 0, it becomes a Chebyshev\n    type I filter (`cheby1`). As both approach 0, it becomes a Butterworth\n    filter (`butter`).\n\n    The equiripple passband has N maxima or minima (for example, a\n    5th-order filter has 3 maxima and 2 minima). Consequently, the DC gain is\n    unity for odd-order filters, or -rp dB for even-order filters.\n    "
    return iirfilter(N, Wn, rs=rs, rp=rp, btype=btype, analog=analog, output=output, ftype='elliptic', fs=fs)

def iirdesign(wp, ws, gpass, gstop, analog=False, ftype='ellip', output='ba', fs=None):
    if False:
        i = 10
        return i + 15
    "Complete IIR digital and analog filter design.\n\n    Given passband and stopband frequencies and gains, construct an analog or\n    digital IIR filter of minimum order for a given basic type. Return the\n    output in numerator, denominator ('ba'), pole-zero ('zpk') or second order\n    sections ('sos') form.\n\n    Parameters\n    ----------\n    wp, ws : float or array like, shape (2,)\n        Passband and stopband edge frequencies. Possible values are scalars\n        (for lowpass and highpass filters) or ranges (for bandpass and bandstop\n        filters).\n        For digital filters, these are in the same units as `fs`. By default,\n        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,\n        where 1 is the Nyquist frequency. For example:\n\n            - Lowpass:   wp = 0.2,          ws = 0.3\n            - Highpass:  wp = 0.3,          ws = 0.2\n            - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]\n            - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]\n\n        For analog filters, `wp` and `ws` are angular frequencies\n        (e.g., rad/s). Note, that for bandpass and bandstop filters passband\n        must lie strictly inside stopband or vice versa.\n    gpass : float\n        The maximum loss in the passband (dB).\n    gstop : float\n        The minimum attenuation in the stopband (dB).\n    analog : bool, optional\n        When True, return an analog filter, otherwise a digital filter is\n        returned.\n    ftype : str, optional\n        The type of IIR filter to design:\n\n            - Butterworth   : 'butter'\n            - Chebyshev I   : 'cheby1'\n            - Chebyshev II  : 'cheby2'\n            - Cauer/elliptic: 'ellip'\n\n    output : {'ba', 'zpk', 'sos'}, optional\n        Filter form of the output:\n\n            - second-order sections (recommended): 'sos'\n            - numerator/denominator (default)    : 'ba'\n            - pole-zero                          : 'zpk'\n\n        In general the second-order sections ('sos') form  is\n        recommended because inferring the coefficients for the\n        numerator/denominator form ('ba') suffers from numerical\n        instabilities. For reasons of backward compatibility the default\n        form is the numerator/denominator form ('ba'), where the 'b'\n        and the 'a' in 'ba' refer to the commonly used names of the\n        coefficients used.\n\n        Note: Using the second-order sections form ('sos') is sometimes\n        associated with additional computational costs: for\n        data-intense use cases it is therefore recommended to also\n        investigate the numerator/denominator form ('ba').\n\n    fs : float, optional\n        The sampling frequency of the digital system.\n\n        .. versionadded:: 1.2.0\n\n    Returns\n    -------\n    b, a : ndarray, ndarray\n        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.\n        Only returned if ``output='ba'``.\n    z, p, k : ndarray, ndarray, float\n        Zeros, poles, and system gain of the IIR filter transfer\n        function.  Only returned if ``output='zpk'``.\n    sos : ndarray\n        Second-order sections representation of the IIR filter.\n        Only returned if ``output='sos'``.\n\n    See Also\n    --------\n    scipy.signal.iirdesign\n    butter : Filter design using order and critical points\n    cheby1, cheby2, ellip, bessel\n    buttord : Find order and critical points from passband and stopband spec\n    cheb1ord, cheb2ord, ellipord\n    iirfilter : General filter design using order and critical frequencies\n    "
    try:
        ordfunc = filter_dict[ftype][1]
    except KeyError as e:
        raise ValueError('Invalid IIR filter type: %s' % ftype) from e
    except IndexError as e:
        raise ValueError('%s does not have order selection. Use iirfilter function.' % ftype) from e
    _validate_gpass_gstop(gpass, gstop)
    wp = cupy.atleast_1d(wp)
    ws = cupy.atleast_1d(ws)
    if wp.shape[0] != ws.shape[0] or wp.shape not in [(1,), (2,)]:
        raise ValueError('wp and ws must have one or two elements each, andthe same shape, got %s and %s' % (wp.shape, ws.shape))
    if any(wp <= 0) or any(ws <= 0):
        raise ValueError('Values for wp, ws must be greater than 0')
    if not analog:
        if fs is None:
            if any(wp >= 1) or any(ws >= 1):
                raise ValueError('Values for wp, ws must be less than 1')
        elif any(wp >= fs / 2) or any(ws >= fs / 2):
            raise ValueError('Values for wp, ws must be less than fs/2 (fs={} -> fs/2={})'.format(fs, fs / 2))
    if wp.shape[0] == 2:
        if not (ws[0] < wp[0] and wp[1] < ws[1] or (wp[0] < ws[0] and ws[1] < wp[1])):
            raise ValueError('Passband must lie strictly inside stopband or vice versa')
    band_type = 2 * (len(wp) - 1)
    band_type += 1
    if wp[0] >= ws[0]:
        band_type += 1
    btype = {1: 'lowpass', 2: 'highpass', 3: 'bandstop', 4: 'bandpass'}[band_type]
    (N, Wn) = ordfunc(wp, ws, gpass, gstop, analog=analog, fs=fs)
    return iirfilter(N, Wn, rp=gpass, rs=gstop, analog=analog, btype=btype, ftype=ftype, output=output, fs=fs)

def iircomb(w0, Q, ftype='notch', fs=2.0, *, pass_zero=False):
    if False:
        while True:
            i = 10
    '\n    Design IIR notching or peaking digital comb filter.\n\n    A notching comb filter consists of regularly-spaced band-stop filters with\n    a narrow bandwidth (high quality factor). Each rejects a narrow frequency\n    band and leaves the rest of the spectrum little changed.\n\n    A peaking comb filter consists of regularly-spaced band-pass filters with\n    a narrow bandwidth (high quality factor). Each rejects components outside\n    a narrow frequency band.\n\n    Parameters\n    ----------\n    w0 : float\n        The fundamental frequency of the comb filter (the spacing between its\n        peaks). This must evenly divide the sampling frequency. If `fs` is\n        specified, this is in the same units as `fs`. By default, it is\n        a normalized scalar that must satisfy  ``0 < w0 < 1``, with\n        ``w0 = 1`` corresponding to half of the sampling frequency.\n    Q : float\n        Quality factor. Dimensionless parameter that characterizes\n        notch filter -3 dB bandwidth ``bw`` relative to its center\n        frequency, ``Q = w0/bw``.\n    ftype : {\'notch\', \'peak\'}\n        The type of comb filter generated by the function. If \'notch\', then\n        the Q factor applies to the notches. If \'peak\', then the Q factor\n        applies to the peaks.  Default is \'notch\'.\n    fs : float, optional\n        The sampling frequency of the signal. Default is 2.0.\n    pass_zero : bool, optional\n        If False (default), the notches (nulls) of the filter are centered on\n        frequencies [0, w0, 2*w0, ...], and the peaks are centered on the\n        midpoints [w0/2, 3*w0/2, 5*w0/2, ...].  If True, the peaks are centered\n        on [0, w0, 2*w0, ...] (passing zero frequency) and vice versa.\n\n    Returns\n    -------\n    b, a : ndarray, ndarray\n        Numerator (``b``) and denominator (``a``) polynomials\n        of the IIR filter.\n\n    Raises\n    ------\n    ValueError\n        If `w0` is less than or equal to 0 or greater than or equal to\n        ``fs/2``, if `fs` is not divisible by `w0`, if `ftype`\n        is not \'notch\' or \'peak\'\n\n    See Also\n    --------\n    scipy.signal.iircomb\n    iirnotch\n    iirpeak\n\n    Notes\n    -----\n    The TF implementation of the\n    comb filter is numerically stable even at higher orders due to the\n    use of a single repeated pole, which won\'t suffer from precision loss.\n\n    References\n    ----------\n    Sophocles J. Orfanidis, "Introduction To Signal Processing",\n         Prentice-Hall, 1996, ch. 11, "Digital Filter Design"\n    '
    w0 = float(w0)
    Q = float(Q)
    fs = float(fs)
    ftype = ftype.lower()
    if not 0 < w0 < fs / 2:
        raise ValueError('w0 must be between 0 and {} (nyquist), but given {}.'.format(fs / 2, w0))
    if ftype not in ('notch', 'peak'):
        raise ValueError('ftype must be either notch or peak.')
    N = round(fs / w0)
    if abs(w0 - fs / N) / fs > 1e-14:
        raise ValueError('fs must be divisible by w0.')
    w0 = 2 * pi * w0 / fs
    w_delta = w0 / Q
    if ftype == 'notch':
        (G0, G) = (1, 0)
    elif ftype == 'peak':
        (G0, G) = (0, 1)
    GB = 1 / math.sqrt(2)
    beta = math.sqrt((GB ** 2 - G0 ** 2) / (G ** 2 - GB ** 2)) * math.tan(N * w_delta / 4)
    ax = (1 - beta) / (1 + beta)
    bx = (G0 + G * beta) / (1 + beta)
    cx = (G0 - G * beta) / (1 + beta)
    negative_coef = ftype == 'peak' and pass_zero or (ftype == 'notch' and (not pass_zero))
    b = cupy.zeros(N + 1)
    b[0] = bx
    if negative_coef:
        b[-1] = -cx
    else:
        b[-1] = +cx
    a = cupy.zeros(N + 1)
    a[0] = 1
    if negative_coef:
        a[-1] = -ax
    else:
        a[-1] = +ax
    return (b, a)

def iirnotch(w0, Q, fs=2.0):
    if False:
        return 10
    '\n    Design second-order IIR notch digital filter.\n\n    A notch filter is a band-stop filter with a narrow bandwidth\n    (high quality factor). It rejects a narrow frequency band and\n    leaves the rest of the spectrum little changed.\n\n    Parameters\n    ----------\n    w0 : float\n        Frequency to remove from a signal. If `fs` is specified, this is in\n        the same units as `fs`. By default, it is a normalized scalar that must\n        satisfy  ``0 < w0 < 1``, with ``w0 = 1`` corresponding to half of the\n        sampling frequency.\n    Q : float\n        Quality factor. Dimensionless parameter that characterizes\n        notch filter -3 dB bandwidth ``bw`` relative to its center\n        frequency, ``Q = w0/bw``.\n    fs : float, optional\n        The sampling frequency of the digital system.\n\n    Returns\n    -------\n    b, a : ndarray, ndarray\n        Numerator (``b``) and denominator (``a``) polynomials\n        of the IIR filter.\n\n    See Also\n    --------\n    scipy.signal.iirnotch\n\n    References\n    ----------\n    Sophocles J. Orfanidis, "Introduction To Signal Processing",\n         Prentice-Hall, 1996\n    '
    return _design_notch_peak_filter(w0, Q, 'notch', fs)

def iirpeak(w0, Q, fs=2.0):
    if False:
        print('Hello World!')
    '\n    Design second-order IIR peak (resonant) digital filter.\n\n    A peak filter is a band-pass filter with a narrow bandwidth\n    (high quality factor). It rejects components outside a narrow\n    frequency band.\n\n    Parameters\n    ----------\n    w0 : float\n        Frequency to be retained in a signal. If `fs` is specified, this is in\n        the same units as `fs`. By default, it is a normalized scalar that must\n        satisfy  ``0 < w0 < 1``, with ``w0 = 1`` corresponding to half of the\n        sampling frequency.\n    Q : float\n        Quality factor. Dimensionless parameter that characterizes\n        peak filter -3 dB bandwidth ``bw`` relative to its center\n        frequency, ``Q = w0/bw``.\n    fs : float, optional\n        The sampling frequency of the digital system.\n\n\n    Returns\n    -------\n    b, a : ndarray, ndarray\n        Numerator (``b``) and denominator (``a``) polynomials\n        of the IIR filter.\n\n    See Also\n    --------\n    scpy.signal.iirpeak\n\n    References\n    ----------\n    Sophocles J. Orfanidis, "Introduction To Signal Processing",\n       Prentice-Hall, 1996\n    '
    return _design_notch_peak_filter(w0, Q, 'peak', fs)

def _design_notch_peak_filter(w0, Q, ftype, fs=2.0):
    if False:
        i = 10
        return i + 15
    '\n    Design notch or peak digital filter.\n\n    Parameters\n    ----------\n    w0 : float\n        Normalized frequency to remove from a signal. If `fs` is specified,\n        this is in the same units as `fs`. By default, it is a normalized\n        scalar that must satisfy  ``0 < w0 < 1``, with ``w0 = 1``\n        corresponding to half of the sampling frequency.\n    Q : float\n        Quality factor. Dimensionless parameter that characterizes\n        notch filter -3 dB bandwidth ``bw`` relative to its center\n        frequency, ``Q = w0/bw``.\n    ftype : str\n        The type of IIR filter to design:\n\n            - notch filter : ``notch``\n            - peak filter  : ``peak``\n    fs : float, optional\n        The sampling frequency of the digital system.\n\n    Returns\n    -------\n    b, a : ndarray, ndarray\n        Numerator (``b``) and denominator (``a``) polynomials\n        of the IIR filter.\n    '
    w0 = float(w0)
    Q = float(Q)
    w0 = 2 * w0 / fs
    if w0 > 1.0 or w0 < 0.0:
        raise ValueError('w0 should be such that 0 < w0 < 1')
    bw = w0 / Q
    bw = bw * pi
    w0 = w0 * pi
    gb = 1 / math.sqrt(2)
    if ftype == 'notch':
        beta = math.sqrt(1.0 - gb ** 2.0) / gb * math.tan(bw / 2.0)
    elif ftype == 'peak':
        beta = gb / math.sqrt(1.0 - gb ** 2.0) * math.tan(bw / 2.0)
    else:
        raise ValueError('Unknown ftype.')
    gain = 1.0 / (1.0 + beta)
    if ftype == 'notch':
        b = [gain * x for x in (1.0, -2.0 * math.cos(w0), 1.0)]
    else:
        b = [(1.0 - gain) * x for x in (1.0, 0.0, -1.0)]
    a = [1.0, -2.0 * gain * math.cos(w0), 2.0 * gain - 1.0]
    a = cupy.asarray(a)
    b = cupy.asarray(b)
    return (b, a)
filter_dict = {'butter': [buttap, buttord], 'butterworth': [buttap, buttord], 'cauer': [ellipap, ellipord], 'elliptic': [ellipap, ellipord], 'ellip': [ellipap, ellipord], 'bessel': [besselap], 'bessel_phase': [besselap], 'bessel_delay': [besselap], 'bessel_mag': [besselap], 'cheby1': [cheb1ap, cheb1ord], 'chebyshev1': [cheb1ap, cheb1ord], 'chebyshevi': [cheb1ap, cheb1ord], 'cheby2': [cheb2ap, cheb2ord], 'chebyshev2': [cheb2ap, cheb2ord], 'chebyshevii': [cheb2ap, cheb2ord]}
band_dict = {'band': 'bandpass', 'bandpass': 'bandpass', 'pass': 'bandpass', 'bp': 'bandpass', 'bs': 'bandstop', 'bandstop': 'bandstop', 'bands': 'bandstop', 'stop': 'bandstop', 'l': 'lowpass', 'low': 'lowpass', 'lowpass': 'lowpass', 'lp': 'lowpass', 'high': 'highpass', 'highpass': 'highpass', 'h': 'highpass', 'hp': 'highpass'}