"""
Discrete Fourier Transforms

Routines in this module:

fft(a, n=None, axis=-1, norm="backward")
ifft(a, n=None, axis=-1, norm="backward")
rfft(a, n=None, axis=-1, norm="backward")
irfft(a, n=None, axis=-1, norm="backward")
hfft(a, n=None, axis=-1, norm="backward")
ihfft(a, n=None, axis=-1, norm="backward")
fftn(a, s=None, axes=None, norm="backward")
ifftn(a, s=None, axes=None, norm="backward")
rfftn(a, s=None, axes=None, norm="backward")
irfftn(a, s=None, axes=None, norm="backward")
fft2(a, s=None, axes=(-2,-1), norm="backward")
ifft2(a, s=None, axes=(-2, -1), norm="backward")
rfft2(a, s=None, axes=(-2,-1), norm="backward")
irfft2(a, s=None, axes=(-2, -1), norm="backward")

i = inverse transform
r = transform of purely real data
h = Hermite transform
n = n-dimensional transform
2 = 2-dimensional transform
(Note: 2D routines are just nD routines with different default
behavior.)

"""
__all__ = ['fft', 'ifft', 'rfft', 'irfft', 'hfft', 'ihfft', 'rfftn', 'irfftn', 'rfft2', 'irfft2', 'fft2', 'ifft2', 'fftn', 'ifftn']
import functools
from numpy.lib.array_utils import normalize_axis_index
from numpy._core import asarray, zeros, swapaxes, conjugate, take, sqrt
from . import _pocketfft_internal as pfi
from numpy._core import overrides
array_function_dispatch = functools.partial(overrides.array_function_dispatch, module='numpy.fft')

def _raw_fft(a, n, axis, is_real, is_forward, inv_norm):
    if False:
        while True:
            i = 10
    axis = normalize_axis_index(axis, a.ndim)
    if n is None:
        n = a.shape[axis]
    fct = 1 / inv_norm
    if a.shape[axis] != n:
        s = list(a.shape)
        index = [slice(None)] * len(s)
        if s[axis] > n:
            index[axis] = slice(0, n)
            a = a[tuple(index)]
        else:
            index[axis] = slice(0, s[axis])
            s[axis] = n
            z = zeros(s, a.dtype.char)
            z[tuple(index)] = a
            a = z
    if axis == a.ndim - 1:
        r = pfi.execute(a, is_real, is_forward, fct)
    else:
        a = swapaxes(a, axis, -1)
        r = pfi.execute(a, is_real, is_forward, fct)
        r = swapaxes(r, axis, -1)
    return r

def _get_forward_norm(n, norm):
    if False:
        for i in range(10):
            print('nop')
    if n < 1:
        raise ValueError(f'Invalid number of FFT data points ({n}) specified.')
    if norm is None or norm == 'backward':
        return 1
    elif norm == 'ortho':
        return sqrt(n)
    elif norm == 'forward':
        return n
    raise ValueError(f'Invalid norm value {norm}; should be "backward","ortho" or "forward".')

def _get_backward_norm(n, norm):
    if False:
        for i in range(10):
            print('nop')
    if n < 1:
        raise ValueError(f'Invalid number of FFT data points ({n}) specified.')
    if norm is None or norm == 'backward':
        return n
    elif norm == 'ortho':
        return sqrt(n)
    elif norm == 'forward':
        return 1
    raise ValueError(f'Invalid norm value {norm}; should be "backward", "ortho" or "forward".')
_SWAP_DIRECTION_MAP = {'backward': 'forward', None: 'forward', 'ortho': 'ortho', 'forward': 'backward'}

def _swap_direction(norm):
    if False:
        i = 10
        return i + 15
    try:
        return _SWAP_DIRECTION_MAP[norm]
    except KeyError:
        raise ValueError(f'Invalid norm value {norm}; should be "backward", "ortho" or "forward".') from None

def _fft_dispatcher(a, n=None, axis=None, norm=None):
    if False:
        print('Hello World!')
    return (a,)

@array_function_dispatch(_fft_dispatcher)
def fft(a, n=None, axis=-1, norm=None):
    if False:
        print('Hello World!')
    '\n    Compute the one-dimensional discrete Fourier Transform.\n\n    This function computes the one-dimensional *n*-point discrete Fourier\n    Transform (DFT) with the efficient Fast Fourier Transform (FFT)\n    algorithm [CT].\n\n    Parameters\n    ----------\n    a : array_like\n        Input array, can be complex.\n    n : int, optional\n        Length of the transformed axis of the output.\n        If `n` is smaller than the length of the input, the input is cropped.\n        If it is larger, the input is padded with zeros.  If `n` is not given,\n        the length of the input along the axis specified by `axis` is used.\n    axis : int, optional\n        Axis over which to compute the FFT.  If not given, the last axis is\n        used.\n    norm : {"backward", "ortho", "forward"}, optional\n        .. versionadded:: 1.10.0\n\n        Normalization mode (see `numpy.fft`). Default is "backward".\n        Indicates which direction of the forward/backward pair of transforms\n        is scaled and with what normalization factor.\n\n        .. versionadded:: 1.20.0\n\n            The "backward", "forward" values were added.\n\n    Returns\n    -------\n    out : complex ndarray\n        The truncated or zero-padded input, transformed along the axis\n        indicated by `axis`, or the last one if `axis` is not specified.\n\n    Raises\n    ------\n    IndexError\n        If `axis` is not a valid axis of `a`.\n\n    See Also\n    --------\n    numpy.fft : for definition of the DFT and conventions used.\n    ifft : The inverse of `fft`.\n    fft2 : The two-dimensional FFT.\n    fftn : The *n*-dimensional FFT.\n    rfftn : The *n*-dimensional FFT of real input.\n    fftfreq : Frequency bins for given FFT parameters.\n\n    Notes\n    -----\n    FFT (Fast Fourier Transform) refers to a way the discrete Fourier\n    Transform (DFT) can be calculated efficiently, by using symmetries in the\n    calculated terms.  The symmetry is highest when `n` is a power of 2, and\n    the transform is therefore most efficient for these sizes.\n\n    The DFT is defined, with the conventions used in this implementation, in\n    the documentation for the `numpy.fft` module.\n\n    References\n    ----------\n    .. [CT] Cooley, James W., and John W. Tukey, 1965, "An algorithm for the\n            machine calculation of complex Fourier series," *Math. Comput.*\n            19: 297-301.\n\n    Examples\n    --------\n    >>> np.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))\n    array([-2.33486982e-16+1.14423775e-17j,  8.00000000e+00-1.25557246e-15j,\n            2.33486982e-16+2.33486982e-16j,  0.00000000e+00+1.22464680e-16j,\n           -1.14423775e-17+2.33486982e-16j,  0.00000000e+00+5.20784380e-16j,\n            1.14423775e-17+1.14423775e-17j,  0.00000000e+00+1.22464680e-16j])\n\n    In this example, real input has an FFT which is Hermitian, i.e., symmetric\n    in the real part and anti-symmetric in the imaginary part, as described in\n    the `numpy.fft` documentation:\n\n    >>> import matplotlib.pyplot as plt\n    >>> t = np.arange(256)\n    >>> sp = np.fft.fft(np.sin(t))\n    >>> freq = np.fft.fftfreq(t.shape[-1])\n    >>> plt.plot(freq, sp.real, freq, sp.imag)\n    [<matplotlib.lines.Line2D object at 0x...>, <matplotlib.lines.Line2D object at 0x...>]\n    >>> plt.show()\n\n    '
    a = asarray(a)
    if n is None:
        n = a.shape[axis]
    inv_norm = _get_forward_norm(n, norm)
    output = _raw_fft(a, n, axis, False, True, inv_norm)
    return output

@array_function_dispatch(_fft_dispatcher)
def ifft(a, n=None, axis=-1, norm=None):
    if False:
        i = 10
        return i + 15
    '\n    Compute the one-dimensional inverse discrete Fourier Transform.\n\n    This function computes the inverse of the one-dimensional *n*-point\n    discrete Fourier transform computed by `fft`.  In other words,\n    ``ifft(fft(a)) == a`` to within numerical accuracy.\n    For a general description of the algorithm and definitions,\n    see `numpy.fft`.\n\n    The input should be ordered in the same way as is returned by `fft`,\n    i.e.,\n\n    * ``a[0]`` should contain the zero frequency term,\n    * ``a[1:n//2]`` should contain the positive-frequency terms,\n    * ``a[n//2 + 1:]`` should contain the negative-frequency terms, in\n      increasing order starting from the most negative frequency.\n\n    For an even number of input points, ``A[n//2]`` represents the sum of\n    the values at the positive and negative Nyquist frequencies, as the two\n    are aliased together. See `numpy.fft` for details.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array, can be complex.\n    n : int, optional\n        Length of the transformed axis of the output.\n        If `n` is smaller than the length of the input, the input is cropped.\n        If it is larger, the input is padded with zeros.  If `n` is not given,\n        the length of the input along the axis specified by `axis` is used.\n        See notes about padding issues.\n    axis : int, optional\n        Axis over which to compute the inverse DFT.  If not given, the last\n        axis is used.\n    norm : {"backward", "ortho", "forward"}, optional\n        .. versionadded:: 1.10.0\n\n        Normalization mode (see `numpy.fft`). Default is "backward".\n        Indicates which direction of the forward/backward pair of transforms\n        is scaled and with what normalization factor.\n\n        .. versionadded:: 1.20.0\n\n            The "backward", "forward" values were added.\n\n    Returns\n    -------\n    out : complex ndarray\n        The truncated or zero-padded input, transformed along the axis\n        indicated by `axis`, or the last one if `axis` is not specified.\n\n    Raises\n    ------\n    IndexError\n        If `axis` is not a valid axis of `a`.\n\n    See Also\n    --------\n    numpy.fft : An introduction, with definitions and general explanations.\n    fft : The one-dimensional (forward) FFT, of which `ifft` is the inverse\n    ifft2 : The two-dimensional inverse FFT.\n    ifftn : The n-dimensional inverse FFT.\n\n    Notes\n    -----\n    If the input parameter `n` is larger than the size of the input, the input\n    is padded by appending zeros at the end.  Even though this is the common\n    approach, it might lead to surprising results.  If a different padding is\n    desired, it must be performed before calling `ifft`.\n\n    Examples\n    --------\n    >>> np.fft.ifft([0, 4, 0, 0])\n    array([ 1.+0.j,  0.+1.j, -1.+0.j,  0.-1.j]) # may vary\n\n    Create and plot a band-limited signal with random phases:\n\n    >>> import matplotlib.pyplot as plt\n    >>> t = np.arange(400)\n    >>> n = np.zeros((400,), dtype=complex)\n    >>> n[40:60] = np.exp(1j*np.random.uniform(0, 2*np.pi, (20,)))\n    >>> s = np.fft.ifft(n)\n    >>> plt.plot(t, s.real, label=\'real\')\n    [<matplotlib.lines.Line2D object at ...>]\n    >>> plt.plot(t, s.imag, \'--\', label=\'imaginary\')\n    [<matplotlib.lines.Line2D object at ...>]\n    >>> plt.legend()\n    <matplotlib.legend.Legend object at ...>\n    >>> plt.show()\n\n    '
    a = asarray(a)
    if n is None:
        n = a.shape[axis]
    inv_norm = _get_backward_norm(n, norm)
    output = _raw_fft(a, n, axis, False, False, inv_norm)
    return output

@array_function_dispatch(_fft_dispatcher)
def rfft(a, n=None, axis=-1, norm=None):
    if False:
        return 10
    '\n    Compute the one-dimensional discrete Fourier Transform for real input.\n\n    This function computes the one-dimensional *n*-point discrete Fourier\n    Transform (DFT) of a real-valued array by means of an efficient algorithm\n    called the Fast Fourier Transform (FFT).\n\n    Parameters\n    ----------\n    a : array_like\n        Input array\n    n : int, optional\n        Number of points along transformation axis in the input to use.\n        If `n` is smaller than the length of the input, the input is cropped.\n        If it is larger, the input is padded with zeros. If `n` is not given,\n        the length of the input along the axis specified by `axis` is used.\n    axis : int, optional\n        Axis over which to compute the FFT. If not given, the last axis is\n        used.\n    norm : {"backward", "ortho", "forward"}, optional\n        .. versionadded:: 1.10.0\n\n        Normalization mode (see `numpy.fft`). Default is "backward".\n        Indicates which direction of the forward/backward pair of transforms\n        is scaled and with what normalization factor.\n\n        .. versionadded:: 1.20.0\n\n            The "backward", "forward" values were added.\n\n    Returns\n    -------\n    out : complex ndarray\n        The truncated or zero-padded input, transformed along the axis\n        indicated by `axis`, or the last one if `axis` is not specified.\n        If `n` is even, the length of the transformed axis is ``(n/2)+1``.\n        If `n` is odd, the length is ``(n+1)/2``.\n\n    Raises\n    ------\n    IndexError\n        If `axis` is not a valid axis of `a`.\n\n    See Also\n    --------\n    numpy.fft : For definition of the DFT and conventions used.\n    irfft : The inverse of `rfft`.\n    fft : The one-dimensional FFT of general (complex) input.\n    fftn : The *n*-dimensional FFT.\n    rfftn : The *n*-dimensional FFT of real input.\n\n    Notes\n    -----\n    When the DFT is computed for purely real input, the output is\n    Hermitian-symmetric, i.e. the negative frequency terms are just the complex\n    conjugates of the corresponding positive-frequency terms, and the\n    negative-frequency terms are therefore redundant.  This function does not\n    compute the negative frequency terms, and the length of the transformed\n    axis of the output is therefore ``n//2 + 1``.\n\n    When ``A = rfft(a)`` and fs is the sampling frequency, ``A[0]`` contains\n    the zero-frequency term 0*fs, which is real due to Hermitian symmetry.\n\n    If `n` is even, ``A[-1]`` contains the term representing both positive\n    and negative Nyquist frequency (+fs/2 and -fs/2), and must also be purely\n    real. If `n` is odd, there is no term at fs/2; ``A[-1]`` contains\n    the largest positive frequency (fs/2*(n-1)/n), and is complex in the\n    general case.\n\n    If the input `a` contains an imaginary part, it is silently discarded.\n\n    Examples\n    --------\n    >>> np.fft.fft([0, 1, 0, 0])\n    array([ 1.+0.j,  0.-1.j, -1.+0.j,  0.+1.j]) # may vary\n    >>> np.fft.rfft([0, 1, 0, 0])\n    array([ 1.+0.j,  0.-1.j, -1.+0.j]) # may vary\n\n    Notice how the final element of the `fft` output is the complex conjugate\n    of the second element, for real input. For `rfft`, this symmetry is\n    exploited to compute only the non-negative frequency terms.\n\n    '
    a = asarray(a)
    if n is None:
        n = a.shape[axis]
    inv_norm = _get_forward_norm(n, norm)
    output = _raw_fft(a, n, axis, True, True, inv_norm)
    return output

@array_function_dispatch(_fft_dispatcher)
def irfft(a, n=None, axis=-1, norm=None):
    if False:
        while True:
            i = 10
    '\n    Computes the inverse of `rfft`.\n\n    This function computes the inverse of the one-dimensional *n*-point\n    discrete Fourier Transform of real input computed by `rfft`.\n    In other words, ``irfft(rfft(a), len(a)) == a`` to within numerical\n    accuracy. (See Notes below for why ``len(a)`` is necessary here.)\n\n    The input is expected to be in the form returned by `rfft`, i.e. the\n    real zero-frequency term followed by the complex positive frequency terms\n    in order of increasing frequency.  Since the discrete Fourier Transform of\n    real input is Hermitian-symmetric, the negative frequency terms are taken\n    to be the complex conjugates of the corresponding positive frequency terms.\n\n    Parameters\n    ----------\n    a : array_like\n        The input array.\n    n : int, optional\n        Length of the transformed axis of the output.\n        For `n` output points, ``n//2+1`` input points are necessary.  If the\n        input is longer than this, it is cropped.  If it is shorter than this,\n        it is padded with zeros.  If `n` is not given, it is taken to be\n        ``2*(m-1)`` where ``m`` is the length of the input along the axis\n        specified by `axis`.\n    axis : int, optional\n        Axis over which to compute the inverse FFT. If not given, the last\n        axis is used.\n    norm : {"backward", "ortho", "forward"}, optional\n        .. versionadded:: 1.10.0\n\n        Normalization mode (see `numpy.fft`). Default is "backward".\n        Indicates which direction of the forward/backward pair of transforms\n        is scaled and with what normalization factor.\n\n        .. versionadded:: 1.20.0\n\n            The "backward", "forward" values were added.\n\n    Returns\n    -------\n    out : ndarray\n        The truncated or zero-padded input, transformed along the axis\n        indicated by `axis`, or the last one if `axis` is not specified.\n        The length of the transformed axis is `n`, or, if `n` is not given,\n        ``2*(m-1)`` where ``m`` is the length of the transformed axis of the\n        input. To get an odd number of output points, `n` must be specified.\n\n    Raises\n    ------\n    IndexError\n        If `axis` is not a valid axis of `a`.\n\n    See Also\n    --------\n    numpy.fft : For definition of the DFT and conventions used.\n    rfft : The one-dimensional FFT of real input, of which `irfft` is inverse.\n    fft : The one-dimensional FFT.\n    irfft2 : The inverse of the two-dimensional FFT of real input.\n    irfftn : The inverse of the *n*-dimensional FFT of real input.\n\n    Notes\n    -----\n    Returns the real valued `n`-point inverse discrete Fourier transform\n    of `a`, where `a` contains the non-negative frequency terms of a\n    Hermitian-symmetric sequence. `n` is the length of the result, not the\n    input.\n\n    If you specify an `n` such that `a` must be zero-padded or truncated, the\n    extra/removed values will be added/removed at high frequencies. One can\n    thus resample a series to `m` points via Fourier interpolation by:\n    ``a_resamp = irfft(rfft(a), m)``.\n\n    The correct interpretation of the hermitian input depends on the length of\n    the original data, as given by `n`. This is because each input shape could\n    correspond to either an odd or even length signal. By default, `irfft`\n    assumes an even output length which puts the last entry at the Nyquist\n    frequency; aliasing with its symmetric counterpart. By Hermitian symmetry,\n    the value is thus treated as purely real. To avoid losing information, the\n    correct length of the real input **must** be given.\n\n    Examples\n    --------\n    >>> np.fft.ifft([1, -1j, -1, 1j])\n    array([0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j]) # may vary\n    >>> np.fft.irfft([1, -1j, -1])\n    array([0.,  1.,  0.,  0.])\n\n    Notice how the last term in the input to the ordinary `ifft` is the\n    complex conjugate of the second term, and the output has zero imaginary\n    part everywhere.  When calling `irfft`, the negative frequencies are not\n    specified, and the output array is purely real.\n\n    '
    a = asarray(a)
    if n is None:
        n = (a.shape[axis] - 1) * 2
    inv_norm = _get_backward_norm(n, norm)
    output = _raw_fft(a, n, axis, True, False, inv_norm)
    return output

@array_function_dispatch(_fft_dispatcher)
def hfft(a, n=None, axis=-1, norm=None):
    if False:
        print('Hello World!')
    '\n    Compute the FFT of a signal that has Hermitian symmetry, i.e., a real\n    spectrum.\n\n    Parameters\n    ----------\n    a : array_like\n        The input array.\n    n : int, optional\n        Length of the transformed axis of the output. For `n` output\n        points, ``n//2 + 1`` input points are necessary.  If the input is\n        longer than this, it is cropped.  If it is shorter than this, it is\n        padded with zeros.  If `n` is not given, it is taken to be ``2*(m-1)``\n        where ``m`` is the length of the input along the axis specified by\n        `axis`.\n    axis : int, optional\n        Axis over which to compute the FFT. If not given, the last\n        axis is used.\n    norm : {"backward", "ortho", "forward"}, optional\n        .. versionadded:: 1.10.0\n\n        Normalization mode (see `numpy.fft`). Default is "backward".\n        Indicates which direction of the forward/backward pair of transforms\n        is scaled and with what normalization factor.\n\n        .. versionadded:: 1.20.0\n\n            The "backward", "forward" values were added.\n\n    Returns\n    -------\n    out : ndarray\n        The truncated or zero-padded input, transformed along the axis\n        indicated by `axis`, or the last one if `axis` is not specified.\n        The length of the transformed axis is `n`, or, if `n` is not given,\n        ``2*m - 2`` where ``m`` is the length of the transformed axis of\n        the input. To get an odd number of output points, `n` must be\n        specified, for instance as ``2*m - 1`` in the typical case,\n\n    Raises\n    ------\n    IndexError\n        If `axis` is not a valid axis of `a`.\n\n    See also\n    --------\n    rfft : Compute the one-dimensional FFT for real input.\n    ihfft : The inverse of `hfft`.\n\n    Notes\n    -----\n    `hfft`/`ihfft` are a pair analogous to `rfft`/`irfft`, but for the\n    opposite case: here the signal has Hermitian symmetry in the time\n    domain and is real in the frequency domain. So here it\'s `hfft` for\n    which you must supply the length of the result if it is to be odd.\n\n    * even: ``ihfft(hfft(a, 2*len(a) - 2)) == a``, within roundoff error,\n    * odd: ``ihfft(hfft(a, 2*len(a) - 1)) == a``, within roundoff error.\n\n    The correct interpretation of the hermitian input depends on the length of\n    the original data, as given by `n`. This is because each input shape could\n    correspond to either an odd or even length signal. By default, `hfft`\n    assumes an even output length which puts the last entry at the Nyquist\n    frequency; aliasing with its symmetric counterpart. By Hermitian symmetry,\n    the value is thus treated as purely real. To avoid losing information, the\n    shape of the full signal **must** be given.\n\n    Examples\n    --------\n    >>> signal = np.array([1, 2, 3, 4, 3, 2])\n    >>> np.fft.fft(signal)\n    array([15.+0.j,  -4.+0.j,   0.+0.j,  -1.-0.j,   0.+0.j,  -4.+0.j]) # may vary\n    >>> np.fft.hfft(signal[:4]) # Input first half of signal\n    array([15.,  -4.,   0.,  -1.,   0.,  -4.])\n    >>> np.fft.hfft(signal, 6)  # Input entire signal and truncate\n    array([15.,  -4.,   0.,  -1.,   0.,  -4.])\n\n\n    >>> signal = np.array([[1, 1.j], [-1.j, 2]])\n    >>> np.conj(signal.T) - signal   # check Hermitian symmetry\n    array([[ 0.-0.j,  -0.+0.j], # may vary\n           [ 0.+0.j,  0.-0.j]])\n    >>> freq_spectrum = np.fft.hfft(signal)\n    >>> freq_spectrum\n    array([[ 1.,  1.],\n           [ 2., -2.]])\n\n    '
    a = asarray(a)
    if n is None:
        n = (a.shape[axis] - 1) * 2
    new_norm = _swap_direction(norm)
    output = irfft(conjugate(a), n, axis, norm=new_norm)
    return output

@array_function_dispatch(_fft_dispatcher)
def ihfft(a, n=None, axis=-1, norm=None):
    if False:
        return 10
    '\n    Compute the inverse FFT of a signal that has Hermitian symmetry.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array.\n    n : int, optional\n        Length of the inverse FFT, the number of points along\n        transformation axis in the input to use.  If `n` is smaller than\n        the length of the input, the input is cropped.  If it is larger,\n        the input is padded with zeros. If `n` is not given, the length of\n        the input along the axis specified by `axis` is used.\n    axis : int, optional\n        Axis over which to compute the inverse FFT. If not given, the last\n        axis is used.\n    norm : {"backward", "ortho", "forward"}, optional\n        .. versionadded:: 1.10.0\n\n        Normalization mode (see `numpy.fft`). Default is "backward".\n        Indicates which direction of the forward/backward pair of transforms\n        is scaled and with what normalization factor.\n\n        .. versionadded:: 1.20.0\n\n            The "backward", "forward" values were added.\n\n    Returns\n    -------\n    out : complex ndarray\n        The truncated or zero-padded input, transformed along the axis\n        indicated by `axis`, or the last one if `axis` is not specified.\n        The length of the transformed axis is ``n//2 + 1``.\n\n    See also\n    --------\n    hfft, irfft\n\n    Notes\n    -----\n    `hfft`/`ihfft` are a pair analogous to `rfft`/`irfft`, but for the\n    opposite case: here the signal has Hermitian symmetry in the time\n    domain and is real in the frequency domain. So here it\'s `hfft` for\n    which you must supply the length of the result if it is to be odd:\n\n    * even: ``ihfft(hfft(a, 2*len(a) - 2)) == a``, within roundoff error,\n    * odd: ``ihfft(hfft(a, 2*len(a) - 1)) == a``, within roundoff error.\n\n    Examples\n    --------\n    >>> spectrum = np.array([ 15, -4, 0, -1, 0, -4])\n    >>> np.fft.ifft(spectrum)\n    array([1.+0.j,  2.+0.j,  3.+0.j,  4.+0.j,  3.+0.j,  2.+0.j]) # may vary\n    >>> np.fft.ihfft(spectrum)\n    array([ 1.-0.j,  2.-0.j,  3.-0.j,  4.-0.j]) # may vary\n\n    '
    a = asarray(a)
    if n is None:
        n = a.shape[axis]
    new_norm = _swap_direction(norm)
    output = conjugate(rfft(a, n, axis, norm=new_norm))
    return output

def _cook_nd_args(a, s=None, axes=None, invreal=0):
    if False:
        for i in range(10):
            print('nop')
    if s is None:
        shapeless = 1
        if axes is None:
            s = list(a.shape)
        else:
            s = take(a.shape, axes)
    else:
        shapeless = 0
    s = list(s)
    if axes is None:
        axes = list(range(-len(s), 0))
    if len(s) != len(axes):
        raise ValueError('Shape and axes have different lengths.')
    if invreal and shapeless:
        s[-1] = (a.shape[axes[-1]] - 1) * 2
    return (s, axes)

def _raw_fftnd(a, s=None, axes=None, function=fft, norm=None):
    if False:
        print('Hello World!')
    a = asarray(a)
    (s, axes) = _cook_nd_args(a, s, axes)
    itl = list(range(len(axes)))
    itl.reverse()
    for ii in itl:
        a = function(a, n=s[ii], axis=axes[ii], norm=norm)
    return a

def _fftn_dispatcher(a, s=None, axes=None, norm=None):
    if False:
        print('Hello World!')
    return (a,)

@array_function_dispatch(_fftn_dispatcher)
def fftn(a, s=None, axes=None, norm=None):
    if False:
        print('Hello World!')
    '\n    Compute the N-dimensional discrete Fourier Transform.\n\n    This function computes the *N*-dimensional discrete Fourier Transform over\n    any number of axes in an *M*-dimensional array by means of the Fast Fourier\n    Transform (FFT).\n\n    Parameters\n    ----------\n    a : array_like\n        Input array, can be complex.\n    s : sequence of ints, optional\n        Shape (length of each transformed axis) of the output\n        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).\n        This corresponds to ``n`` for ``fft(x, n)``.\n        Along any axis, if the given shape is smaller than that of the input,\n        the input is cropped.  If it is larger, the input is padded with zeros.\n        if `s` is not given, the shape of the input along the axes specified\n        by `axes` is used.\n    axes : sequence of ints, optional\n        Axes over which to compute the FFT.  If not given, the last ``len(s)``\n        axes are used, or all axes if `s` is also not specified.\n        Repeated indices in `axes` means that the transform over that axis is\n        performed multiple times.\n    norm : {"backward", "ortho", "forward"}, optional\n        .. versionadded:: 1.10.0\n\n        Normalization mode (see `numpy.fft`). Default is "backward".\n        Indicates which direction of the forward/backward pair of transforms\n        is scaled and with what normalization factor.\n\n        .. versionadded:: 1.20.0\n\n            The "backward", "forward" values were added.\n\n    Returns\n    -------\n    out : complex ndarray\n        The truncated or zero-padded input, transformed along the axes\n        indicated by `axes`, or by a combination of `s` and `a`,\n        as explained in the parameters section above.\n\n    Raises\n    ------\n    ValueError\n        If `s` and `axes` have different length.\n    IndexError\n        If an element of `axes` is larger than than the number of axes of `a`.\n\n    See Also\n    --------\n    numpy.fft : Overall view of discrete Fourier transforms, with definitions\n        and conventions used.\n    ifftn : The inverse of `fftn`, the inverse *n*-dimensional FFT.\n    fft : The one-dimensional FFT, with definitions and conventions used.\n    rfftn : The *n*-dimensional FFT of real input.\n    fft2 : The two-dimensional FFT.\n    fftshift : Shifts zero-frequency terms to centre of array\n\n    Notes\n    -----\n    The output, analogously to `fft`, contains the term for zero frequency in\n    the low-order corner of all axes, the positive frequency terms in the\n    first half of all axes, the term for the Nyquist frequency in the middle\n    of all axes and the negative frequency terms in the second half of all\n    axes, in order of decreasingly negative frequency.\n\n    See `numpy.fft` for details, definitions and conventions used.\n\n    Examples\n    --------\n    >>> a = np.mgrid[:3, :3, :3][0]\n    >>> np.fft.fftn(a, axes=(1, 2))\n    array([[[ 0.+0.j,   0.+0.j,   0.+0.j], # may vary\n            [ 0.+0.j,   0.+0.j,   0.+0.j],\n            [ 0.+0.j,   0.+0.j,   0.+0.j]],\n           [[ 9.+0.j,   0.+0.j,   0.+0.j],\n            [ 0.+0.j,   0.+0.j,   0.+0.j],\n            [ 0.+0.j,   0.+0.j,   0.+0.j]],\n           [[18.+0.j,   0.+0.j,   0.+0.j],\n            [ 0.+0.j,   0.+0.j,   0.+0.j],\n            [ 0.+0.j,   0.+0.j,   0.+0.j]]])\n    >>> np.fft.fftn(a, (2, 2), axes=(0, 1))\n    array([[[ 2.+0.j,  2.+0.j,  2.+0.j], # may vary\n            [ 0.+0.j,  0.+0.j,  0.+0.j]],\n           [[-2.+0.j, -2.+0.j, -2.+0.j],\n            [ 0.+0.j,  0.+0.j,  0.+0.j]]])\n\n    >>> import matplotlib.pyplot as plt\n    >>> [X, Y] = np.meshgrid(2 * np.pi * np.arange(200) / 12,\n    ...                      2 * np.pi * np.arange(200) / 34)\n    >>> S = np.sin(X) + np.cos(Y) + np.random.uniform(0, 1, X.shape)\n    >>> FS = np.fft.fftn(S)\n    >>> plt.imshow(np.log(np.abs(np.fft.fftshift(FS))**2))\n    <matplotlib.image.AxesImage object at 0x...>\n    >>> plt.show()\n\n    '
    return _raw_fftnd(a, s, axes, fft, norm)

@array_function_dispatch(_fftn_dispatcher)
def ifftn(a, s=None, axes=None, norm=None):
    if False:
        i = 10
        return i + 15
    '\n    Compute the N-dimensional inverse discrete Fourier Transform.\n\n    This function computes the inverse of the N-dimensional discrete\n    Fourier Transform over any number of axes in an M-dimensional array by\n    means of the Fast Fourier Transform (FFT).  In other words,\n    ``ifftn(fftn(a)) == a`` to within numerical accuracy.\n    For a description of the definitions and conventions used, see `numpy.fft`.\n\n    The input, analogously to `ifft`, should be ordered in the same way as is\n    returned by `fftn`, i.e. it should have the term for zero frequency\n    in all axes in the low-order corner, the positive frequency terms in the\n    first half of all axes, the term for the Nyquist frequency in the middle\n    of all axes and the negative frequency terms in the second half of all\n    axes, in order of decreasingly negative frequency.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array, can be complex.\n    s : sequence of ints, optional\n        Shape (length of each transformed axis) of the output\n        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).\n        This corresponds to ``n`` for ``ifft(x, n)``.\n        Along any axis, if the given shape is smaller than that of the input,\n        the input is cropped.  If it is larger, the input is padded with zeros.\n        if `s` is not given, the shape of the input along the axes specified\n        by `axes` is used.  See notes for issue on `ifft` zero padding.\n    axes : sequence of ints, optional\n        Axes over which to compute the IFFT.  If not given, the last ``len(s)``\n        axes are used, or all axes if `s` is also not specified.\n        Repeated indices in `axes` means that the inverse transform over that\n        axis is performed multiple times.\n    norm : {"backward", "ortho", "forward"}, optional\n        .. versionadded:: 1.10.0\n\n        Normalization mode (see `numpy.fft`). Default is "backward".\n        Indicates which direction of the forward/backward pair of transforms\n        is scaled and with what normalization factor.\n\n        .. versionadded:: 1.20.0\n\n            The "backward", "forward" values were added.\n\n    Returns\n    -------\n    out : complex ndarray\n        The truncated or zero-padded input, transformed along the axes\n        indicated by `axes`, or by a combination of `s` or `a`,\n        as explained in the parameters section above.\n\n    Raises\n    ------\n    ValueError\n        If `s` and `axes` have different length.\n    IndexError\n        If an element of `axes` is larger than than the number of axes of `a`.\n\n    See Also\n    --------\n    numpy.fft : Overall view of discrete Fourier transforms, with definitions\n         and conventions used.\n    fftn : The forward *n*-dimensional FFT, of which `ifftn` is the inverse.\n    ifft : The one-dimensional inverse FFT.\n    ifft2 : The two-dimensional inverse FFT.\n    ifftshift : Undoes `fftshift`, shifts zero-frequency terms to beginning\n        of array.\n\n    Notes\n    -----\n    See `numpy.fft` for definitions and conventions used.\n\n    Zero-padding, analogously with `ifft`, is performed by appending zeros to\n    the input along the specified dimension.  Although this is the common\n    approach, it might lead to surprising results.  If another form of zero\n    padding is desired, it must be performed before `ifftn` is called.\n\n    Examples\n    --------\n    >>> a = np.eye(4)\n    >>> np.fft.ifftn(np.fft.fftn(a, axes=(0,)), axes=(1,))\n    array([[1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j], # may vary\n           [0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],\n           [0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],\n           [0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j]])\n\n\n    Create and plot an image with band-limited frequency content:\n\n    >>> import matplotlib.pyplot as plt\n    >>> n = np.zeros((200,200), dtype=complex)\n    >>> n[60:80, 20:40] = np.exp(1j*np.random.uniform(0, 2*np.pi, (20, 20)))\n    >>> im = np.fft.ifftn(n).real\n    >>> plt.imshow(im)\n    <matplotlib.image.AxesImage object at 0x...>\n    >>> plt.show()\n\n    '
    return _raw_fftnd(a, s, axes, ifft, norm)

@array_function_dispatch(_fftn_dispatcher)
def fft2(a, s=None, axes=(-2, -1), norm=None):
    if False:
        return 10
    '\n    Compute the 2-dimensional discrete Fourier Transform.\n\n    This function computes the *n*-dimensional discrete Fourier Transform\n    over any axes in an *M*-dimensional array by means of the\n    Fast Fourier Transform (FFT).  By default, the transform is computed over\n    the last two axes of the input array, i.e., a 2-dimensional FFT.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array, can be complex\n    s : sequence of ints, optional\n        Shape (length of each transformed axis) of the output\n        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).\n        This corresponds to ``n`` for ``fft(x, n)``.\n        Along each axis, if the given shape is smaller than that of the input,\n        the input is cropped.  If it is larger, the input is padded with zeros.\n        if `s` is not given, the shape of the input along the axes specified\n        by `axes` is used.\n    axes : sequence of ints, optional\n        Axes over which to compute the FFT.  If not given, the last two\n        axes are used.  A repeated index in `axes` means the transform over\n        that axis is performed multiple times.  A one-element sequence means\n        that a one-dimensional FFT is performed.\n    norm : {"backward", "ortho", "forward"}, optional\n        .. versionadded:: 1.10.0\n\n        Normalization mode (see `numpy.fft`). Default is "backward".\n        Indicates which direction of the forward/backward pair of transforms\n        is scaled and with what normalization factor.\n\n        .. versionadded:: 1.20.0\n\n            The "backward", "forward" values were added.\n\n    Returns\n    -------\n    out : complex ndarray\n        The truncated or zero-padded input, transformed along the axes\n        indicated by `axes`, or the last two axes if `axes` is not given.\n\n    Raises\n    ------\n    ValueError\n        If `s` and `axes` have different length, or `axes` not given and\n        ``len(s) != 2``.\n    IndexError\n        If an element of `axes` is larger than than the number of axes of `a`.\n\n    See Also\n    --------\n    numpy.fft : Overall view of discrete Fourier transforms, with definitions\n         and conventions used.\n    ifft2 : The inverse two-dimensional FFT.\n    fft : The one-dimensional FFT.\n    fftn : The *n*-dimensional FFT.\n    fftshift : Shifts zero-frequency terms to the center of the array.\n        For two-dimensional input, swaps first and third quadrants, and second\n        and fourth quadrants.\n\n    Notes\n    -----\n    `fft2` is just `fftn` with a different default for `axes`.\n\n    The output, analogously to `fft`, contains the term for zero frequency in\n    the low-order corner of the transformed axes, the positive frequency terms\n    in the first half of these axes, the term for the Nyquist frequency in the\n    middle of the axes and the negative frequency terms in the second half of\n    the axes, in order of decreasingly negative frequency.\n\n    See `fftn` for details and a plotting example, and `numpy.fft` for\n    definitions and conventions used.\n\n\n    Examples\n    --------\n    >>> a = np.mgrid[:5, :5][0]\n    >>> np.fft.fft2(a)\n    array([[ 50.  +0.j        ,   0.  +0.j        ,   0.  +0.j        , # may vary\n              0.  +0.j        ,   0.  +0.j        ],\n           [-12.5+17.20477401j,   0.  +0.j        ,   0.  +0.j        ,\n              0.  +0.j        ,   0.  +0.j        ],\n           [-12.5 +4.0614962j ,   0.  +0.j        ,   0.  +0.j        ,\n              0.  +0.j        ,   0.  +0.j        ],\n           [-12.5 -4.0614962j ,   0.  +0.j        ,   0.  +0.j        ,\n              0.  +0.j        ,   0.  +0.j        ],\n           [-12.5-17.20477401j,   0.  +0.j        ,   0.  +0.j        ,\n              0.  +0.j        ,   0.  +0.j        ]])\n\n    '
    return _raw_fftnd(a, s, axes, fft, norm)

@array_function_dispatch(_fftn_dispatcher)
def ifft2(a, s=None, axes=(-2, -1), norm=None):
    if False:
        return 10
    '\n    Compute the 2-dimensional inverse discrete Fourier Transform.\n\n    This function computes the inverse of the 2-dimensional discrete Fourier\n    Transform over any number of axes in an M-dimensional array by means of\n    the Fast Fourier Transform (FFT).  In other words, ``ifft2(fft2(a)) == a``\n    to within numerical accuracy.  By default, the inverse transform is\n    computed over the last two axes of the input array.\n\n    The input, analogously to `ifft`, should be ordered in the same way as is\n    returned by `fft2`, i.e. it should have the term for zero frequency\n    in the low-order corner of the two axes, the positive frequency terms in\n    the first half of these axes, the term for the Nyquist frequency in the\n    middle of the axes and the negative frequency terms in the second half of\n    both axes, in order of decreasingly negative frequency.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array, can be complex.\n    s : sequence of ints, optional\n        Shape (length of each axis) of the output (``s[0]`` refers to axis 0,\n        ``s[1]`` to axis 1, etc.).  This corresponds to `n` for ``ifft(x, n)``.\n        Along each axis, if the given shape is smaller than that of the input,\n        the input is cropped.  If it is larger, the input is padded with zeros.\n        if `s` is not given, the shape of the input along the axes specified\n        by `axes` is used.  See notes for issue on `ifft` zero padding.\n    axes : sequence of ints, optional\n        Axes over which to compute the FFT.  If not given, the last two\n        axes are used.  A repeated index in `axes` means the transform over\n        that axis is performed multiple times.  A one-element sequence means\n        that a one-dimensional FFT is performed.\n    norm : {"backward", "ortho", "forward"}, optional\n        .. versionadded:: 1.10.0\n\n        Normalization mode (see `numpy.fft`). Default is "backward".\n        Indicates which direction of the forward/backward pair of transforms\n        is scaled and with what normalization factor.\n\n        .. versionadded:: 1.20.0\n\n            The "backward", "forward" values were added.\n\n    Returns\n    -------\n    out : complex ndarray\n        The truncated or zero-padded input, transformed along the axes\n        indicated by `axes`, or the last two axes if `axes` is not given.\n\n    Raises\n    ------\n    ValueError\n        If `s` and `axes` have different length, or `axes` not given and\n        ``len(s) != 2``.\n    IndexError\n        If an element of `axes` is larger than than the number of axes of `a`.\n\n    See Also\n    --------\n    numpy.fft : Overall view of discrete Fourier transforms, with definitions\n         and conventions used.\n    fft2 : The forward 2-dimensional FFT, of which `ifft2` is the inverse.\n    ifftn : The inverse of the *n*-dimensional FFT.\n    fft : The one-dimensional FFT.\n    ifft : The one-dimensional inverse FFT.\n\n    Notes\n    -----\n    `ifft2` is just `ifftn` with a different default for `axes`.\n\n    See `ifftn` for details and a plotting example, and `numpy.fft` for\n    definition and conventions used.\n\n    Zero-padding, analogously with `ifft`, is performed by appending zeros to\n    the input along the specified dimension.  Although this is the common\n    approach, it might lead to surprising results.  If another form of zero\n    padding is desired, it must be performed before `ifft2` is called.\n\n    Examples\n    --------\n    >>> a = 4 * np.eye(4)\n    >>> np.fft.ifft2(a)\n    array([[1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j], # may vary\n           [0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j],\n           [0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],\n           [0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j]])\n\n    '
    return _raw_fftnd(a, s, axes, ifft, norm)

@array_function_dispatch(_fftn_dispatcher)
def rfftn(a, s=None, axes=None, norm=None):
    if False:
        i = 10
        return i + 15
    '\n    Compute the N-dimensional discrete Fourier Transform for real input.\n\n    This function computes the N-dimensional discrete Fourier Transform over\n    any number of axes in an M-dimensional real array by means of the Fast\n    Fourier Transform (FFT).  By default, all axes are transformed, with the\n    real transform performed over the last axis, while the remaining\n    transforms are complex.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array, taken to be real.\n    s : sequence of ints, optional\n        Shape (length along each transformed axis) to use from the input.\n        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).\n        The final element of `s` corresponds to `n` for ``rfft(x, n)``, while\n        for the remaining axes, it corresponds to `n` for ``fft(x, n)``.\n        Along any axis, if the given shape is smaller than that of the input,\n        the input is cropped.  If it is larger, the input is padded with zeros.\n        if `s` is not given, the shape of the input along the axes specified\n        by `axes` is used.\n    axes : sequence of ints, optional\n        Axes over which to compute the FFT.  If not given, the last ``len(s)``\n        axes are used, or all axes if `s` is also not specified.\n    norm : {"backward", "ortho", "forward"}, optional\n        .. versionadded:: 1.10.0\n\n        Normalization mode (see `numpy.fft`). Default is "backward".\n        Indicates which direction of the forward/backward pair of transforms\n        is scaled and with what normalization factor.\n\n        .. versionadded:: 1.20.0\n\n            The "backward", "forward" values were added.\n\n    Returns\n    -------\n    out : complex ndarray\n        The truncated or zero-padded input, transformed along the axes\n        indicated by `axes`, or by a combination of `s` and `a`,\n        as explained in the parameters section above.\n        The length of the last axis transformed will be ``s[-1]//2+1``,\n        while the remaining transformed axes will have lengths according to\n        `s`, or unchanged from the input.\n\n    Raises\n    ------\n    ValueError\n        If `s` and `axes` have different length.\n    IndexError\n        If an element of `axes` is larger than than the number of axes of `a`.\n\n    See Also\n    --------\n    irfftn : The inverse of `rfftn`, i.e. the inverse of the n-dimensional FFT\n         of real input.\n    fft : The one-dimensional FFT, with definitions and conventions used.\n    rfft : The one-dimensional FFT of real input.\n    fftn : The n-dimensional FFT.\n    rfft2 : The two-dimensional FFT of real input.\n\n    Notes\n    -----\n    The transform for real input is performed over the last transformation\n    axis, as by `rfft`, then the transform over the remaining axes is\n    performed as by `fftn`.  The order of the output is as for `rfft` for the\n    final transformation axis, and as for `fftn` for the remaining\n    transformation axes.\n\n    See `fft` for details, definitions and conventions used.\n\n    Examples\n    --------\n    >>> a = np.ones((2, 2, 2))\n    >>> np.fft.rfftn(a)\n    array([[[8.+0.j,  0.+0.j], # may vary\n            [0.+0.j,  0.+0.j]],\n           [[0.+0.j,  0.+0.j],\n            [0.+0.j,  0.+0.j]]])\n\n    >>> np.fft.rfftn(a, axes=(2, 0))\n    array([[[4.+0.j,  0.+0.j], # may vary\n            [4.+0.j,  0.+0.j]],\n           [[0.+0.j,  0.+0.j],\n            [0.+0.j,  0.+0.j]]])\n\n    '
    a = asarray(a)
    (s, axes) = _cook_nd_args(a, s, axes)
    a = rfft(a, s[-1], axes[-1], norm)
    for ii in range(len(axes) - 1):
        a = fft(a, s[ii], axes[ii], norm)
    return a

@array_function_dispatch(_fftn_dispatcher)
def rfft2(a, s=None, axes=(-2, -1), norm=None):
    if False:
        while True:
            i = 10
    '\n    Compute the 2-dimensional FFT of a real array.\n\n    Parameters\n    ----------\n    a : array\n        Input array, taken to be real.\n    s : sequence of ints, optional\n        Shape of the FFT.\n    axes : sequence of ints, optional\n        Axes over which to compute the FFT.\n    norm : {"backward", "ortho", "forward"}, optional\n        .. versionadded:: 1.10.0\n\n        Normalization mode (see `numpy.fft`). Default is "backward".\n        Indicates which direction of the forward/backward pair of transforms\n        is scaled and with what normalization factor.\n\n        .. versionadded:: 1.20.0\n\n            The "backward", "forward" values were added.\n\n    Returns\n    -------\n    out : ndarray\n        The result of the real 2-D FFT.\n\n    See Also\n    --------\n    rfftn : Compute the N-dimensional discrete Fourier Transform for real\n            input.\n\n    Notes\n    -----\n    This is really just `rfftn` with different default behavior.\n    For more details see `rfftn`.\n\n    Examples\n    --------\n    >>> a = np.mgrid[:5, :5][0]\n    >>> np.fft.rfft2(a)\n    array([[ 50.  +0.j        ,   0.  +0.j        ,   0.  +0.j        ],\n           [-12.5+17.20477401j,   0.  +0.j        ,   0.  +0.j        ],\n           [-12.5 +4.0614962j ,   0.  +0.j        ,   0.  +0.j        ],\n           [-12.5 -4.0614962j ,   0.  +0.j        ,   0.  +0.j        ],\n           [-12.5-17.20477401j,   0.  +0.j        ,   0.  +0.j        ]])\n    '
    return rfftn(a, s, axes, norm)

@array_function_dispatch(_fftn_dispatcher)
def irfftn(a, s=None, axes=None, norm=None):
    if False:
        i = 10
        return i + 15
    '\n    Computes the inverse of `rfftn`.\n\n    This function computes the inverse of the N-dimensional discrete\n    Fourier Transform for real input over any number of axes in an\n    M-dimensional array by means of the Fast Fourier Transform (FFT).  In\n    other words, ``irfftn(rfftn(a), a.shape) == a`` to within numerical\n    accuracy. (The ``a.shape`` is necessary like ``len(a)`` is for `irfft`,\n    and for the same reason.)\n\n    The input should be ordered in the same way as is returned by `rfftn`,\n    i.e. as for `irfft` for the final transformation axis, and as for `ifftn`\n    along all the other axes.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array.\n    s : sequence of ints, optional\n        Shape (length of each transformed axis) of the output\n        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.). `s` is also the\n        number of input points used along this axis, except for the last axis,\n        where ``s[-1]//2+1`` points of the input are used.\n        Along any axis, if the shape indicated by `s` is smaller than that of\n        the input, the input is cropped.  If it is larger, the input is padded\n        with zeros. If `s` is not given, the shape of the input along the axes\n        specified by axes is used. Except for the last axis which is taken to\n        be ``2*(m-1)`` where ``m`` is the length of the input along that axis.\n    axes : sequence of ints, optional\n        Axes over which to compute the inverse FFT. If not given, the last\n        `len(s)` axes are used, or all axes if `s` is also not specified.\n        Repeated indices in `axes` means that the inverse transform over that\n        axis is performed multiple times.\n    norm : {"backward", "ortho", "forward"}, optional\n        .. versionadded:: 1.10.0\n\n        Normalization mode (see `numpy.fft`). Default is "backward".\n        Indicates which direction of the forward/backward pair of transforms\n        is scaled and with what normalization factor.\n\n        .. versionadded:: 1.20.0\n\n            The "backward", "forward" values were added.\n\n    Returns\n    -------\n    out : ndarray\n        The truncated or zero-padded input, transformed along the axes\n        indicated by `axes`, or by a combination of `s` or `a`,\n        as explained in the parameters section above.\n        The length of each transformed axis is as given by the corresponding\n        element of `s`, or the length of the input in every axis except for the\n        last one if `s` is not given.  In the final transformed axis the length\n        of the output when `s` is not given is ``2*(m-1)`` where ``m`` is the\n        length of the final transformed axis of the input.  To get an odd\n        number of output points in the final axis, `s` must be specified.\n\n    Raises\n    ------\n    ValueError\n        If `s` and `axes` have different length.\n    IndexError\n        If an element of `axes` is larger than than the number of axes of `a`.\n\n    See Also\n    --------\n    rfftn : The forward n-dimensional FFT of real input,\n            of which `ifftn` is the inverse.\n    fft : The one-dimensional FFT, with definitions and conventions used.\n    irfft : The inverse of the one-dimensional FFT of real input.\n    irfft2 : The inverse of the two-dimensional FFT of real input.\n\n    Notes\n    -----\n    See `fft` for definitions and conventions used.\n\n    See `rfft` for definitions and conventions used for real input.\n\n    The correct interpretation of the hermitian input depends on the shape of\n    the original data, as given by `s`. This is because each input shape could\n    correspond to either an odd or even length signal. By default, `irfftn`\n    assumes an even output length which puts the last entry at the Nyquist\n    frequency; aliasing with its symmetric counterpart. When performing the\n    final complex to real transform, the last value is thus treated as purely\n    real. To avoid losing information, the correct shape of the real input\n    **must** be given.\n\n    Examples\n    --------\n    >>> a = np.zeros((3, 2, 2))\n    >>> a[0, 0, 0] = 3 * 2 * 2\n    >>> np.fft.irfftn(a)\n    array([[[1.,  1.],\n            [1.,  1.]],\n           [[1.,  1.],\n            [1.,  1.]],\n           [[1.,  1.],\n            [1.,  1.]]])\n\n    '
    a = asarray(a)
    (s, axes) = _cook_nd_args(a, s, axes, invreal=1)
    for ii in range(len(axes) - 1):
        a = ifft(a, s[ii], axes[ii], norm)
    a = irfft(a, s[-1], axes[-1], norm)
    return a

@array_function_dispatch(_fftn_dispatcher)
def irfft2(a, s=None, axes=(-2, -1), norm=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Computes the inverse of `rfft2`.\n\n    Parameters\n    ----------\n    a : array_like\n        The input array\n    s : sequence of ints, optional\n        Shape of the real output to the inverse FFT.\n    axes : sequence of ints, optional\n        The axes over which to compute the inverse fft.\n        Default is the last two axes.\n    norm : {"backward", "ortho", "forward"}, optional\n        .. versionadded:: 1.10.0\n\n        Normalization mode (see `numpy.fft`). Default is "backward".\n        Indicates which direction of the forward/backward pair of transforms\n        is scaled and with what normalization factor.\n\n        .. versionadded:: 1.20.0\n\n            The "backward", "forward" values were added.\n\n    Returns\n    -------\n    out : ndarray\n        The result of the inverse real 2-D FFT.\n\n    See Also\n    --------\n    rfft2 : The forward two-dimensional FFT of real input,\n            of which `irfft2` is the inverse.\n    rfft : The one-dimensional FFT for real input.\n    irfft : The inverse of the one-dimensional FFT of real input.\n    irfftn : Compute the inverse of the N-dimensional FFT of real input.\n\n    Notes\n    -----\n    This is really `irfftn` with different defaults.\n    For more details see `irfftn`.\n\n    Examples\n    --------\n    >>> a = np.mgrid[:5, :5][0]\n    >>> A = np.fft.rfft2(a)\n    >>> np.fft.irfft2(A, s=a.shape)\n    array([[0., 0., 0., 0., 0.],\n           [1., 1., 1., 1., 1.],\n           [2., 2., 2., 2., 2.],\n           [3., 3., 3., 3., 3.],\n           [4., 4., 4., 4., 4.]])\n    '
    return irfftn(a, s, axes, norm)