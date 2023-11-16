"""
Numerical Python functions written for compatibility with MATLAB
commands with the same names. Most numerical Python functions can be found in
the `NumPy`_ and `SciPy`_ libraries. What remains here is code for performing
spectral computations and kernel density estimations.

.. _NumPy: https://numpy.org
.. _SciPy: https://www.scipy.org

Spectral functions
------------------

`cohere`
    Coherence (normalized cross spectral density)

`csd`
    Cross spectral density using Welch's average periodogram

`detrend`
    Remove the mean or best fit line from an array

`psd`
    Power spectral density using Welch's average periodogram

`specgram`
    Spectrogram (spectrum over segments of time)

`complex_spectrum`
    Return the complex-valued frequency spectrum of a signal

`magnitude_spectrum`
    Return the magnitude of the frequency spectrum of a signal

`angle_spectrum`
    Return the angle (wrapped phase) of the frequency spectrum of a signal

`phase_spectrum`
    Return the phase (unwrapped angle) of the frequency spectrum of a signal

`detrend_mean`
    Remove the mean from a line.

`detrend_linear`
    Remove the best fit line from a line.

`detrend_none`
    Return the original line.
"""
import functools
from numbers import Number
import numpy as np
from matplotlib import _api, _docstring, cbook

def window_hanning(x):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return *x* times the Hanning (or Hann) window of len(*x*).\n\n    See Also\n    --------\n    window_none : Another window algorithm.\n    '
    return np.hanning(len(x)) * x

def window_none(x):
    if False:
        while True:
            i = 10
    '\n    No window function; simply return *x*.\n\n    See Also\n    --------\n    window_hanning : Another window algorithm.\n    '
    return x

def detrend(x, key=None, axis=None):
    if False:
        i = 10
        return i + 15
    "\n    Return *x* with its trend removed.\n\n    Parameters\n    ----------\n    x : array or sequence\n        Array or sequence containing the data.\n\n    key : {'default', 'constant', 'mean', 'linear', 'none'} or function\n        The detrending algorithm to use. 'default', 'mean', and 'constant' are\n        the same as `detrend_mean`. 'linear' is the same as `detrend_linear`.\n        'none' is the same as `detrend_none`. The default is 'mean'. See the\n        corresponding functions for more details regarding the algorithms. Can\n        also be a function that carries out the detrend operation.\n\n    axis : int\n        The axis along which to do the detrending.\n\n    See Also\n    --------\n    detrend_mean : Implementation of the 'mean' algorithm.\n    detrend_linear : Implementation of the 'linear' algorithm.\n    detrend_none : Implementation of the 'none' algorithm.\n    "
    if key is None or key in ['constant', 'mean', 'default']:
        return detrend(x, key=detrend_mean, axis=axis)
    elif key == 'linear':
        return detrend(x, key=detrend_linear, axis=axis)
    elif key == 'none':
        return detrend(x, key=detrend_none, axis=axis)
    elif callable(key):
        x = np.asarray(x)
        if axis is not None and axis + 1 > x.ndim:
            raise ValueError(f'axis(={axis}) out of bounds')
        if axis is None and x.ndim == 0 or (not axis and x.ndim == 1):
            return key(x)
        try:
            return key(x, axis=axis)
        except TypeError:
            return np.apply_along_axis(key, axis=axis, arr=x)
    else:
        raise ValueError(f"Unknown value for key: {key!r}, must be one of: 'default', 'constant', 'mean', 'linear', or a function")

def detrend_mean(x, axis=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return *x* minus the mean(*x*).\n\n    Parameters\n    ----------\n    x : array or sequence\n        Array or sequence containing the data\n        Can have any dimensionality\n\n    axis : int\n        The axis along which to take the mean.  See `numpy.mean` for a\n        description of this argument.\n\n    See Also\n    --------\n    detrend_linear : Another detrend algorithm.\n    detrend_none : Another detrend algorithm.\n    detrend : A wrapper around all the detrend algorithms.\n    '
    x = np.asarray(x)
    if axis is not None and axis + 1 > x.ndim:
        raise ValueError('axis(=%s) out of bounds' % axis)
    return x - x.mean(axis, keepdims=True)

def detrend_none(x, axis=None):
    if False:
        return 10
    '\n    Return *x*: no detrending.\n\n    Parameters\n    ----------\n    x : any object\n        An object containing the data\n\n    axis : int\n        This parameter is ignored.\n        It is included for compatibility with detrend_mean\n\n    See Also\n    --------\n    detrend_mean : Another detrend algorithm.\n    detrend_linear : Another detrend algorithm.\n    detrend : A wrapper around all the detrend algorithms.\n    '
    return x

def detrend_linear(y):
    if False:
        print('Hello World!')
    "\n    Return *x* minus best fit line; 'linear' detrending.\n\n    Parameters\n    ----------\n    y : 0-D or 1-D array or sequence\n        Array or sequence containing the data\n\n    See Also\n    --------\n    detrend_mean : Another detrend algorithm.\n    detrend_none : Another detrend algorithm.\n    detrend : A wrapper around all the detrend algorithms.\n    "
    y = np.asarray(y)
    if y.ndim > 1:
        raise ValueError('y cannot have ndim > 1')
    if not y.ndim:
        return np.array(0.0, dtype=y.dtype)
    x = np.arange(y.size, dtype=float)
    C = np.cov(x, y, bias=1)
    b = C[0, 1] / C[0, 0]
    a = y.mean() - b * x.mean()
    return y - (b * x + a)

def _spectral_helper(x, y=None, NFFT=None, Fs=None, detrend_func=None, window=None, noverlap=None, pad_to=None, sides=None, scale_by_freq=None, mode=None):
    if False:
        while True:
            i = 10
    '\n    Private helper implementing the common parts between the psd, csd,\n    spectrogram and complex, magnitude, angle, and phase spectrums.\n    '
    if y is None:
        same_data = True
    else:
        same_data = y is x
    if Fs is None:
        Fs = 2
    if noverlap is None:
        noverlap = 0
    if detrend_func is None:
        detrend_func = detrend_none
    if window is None:
        window = window_hanning
    if NFFT is None:
        NFFT = 256
    if noverlap >= NFFT:
        raise ValueError('noverlap must be less than NFFT')
    if mode is None or mode == 'default':
        mode = 'psd'
    _api.check_in_list(['default', 'psd', 'complex', 'magnitude', 'angle', 'phase'], mode=mode)
    if not same_data and mode != 'psd':
        raise ValueError("x and y must be equal if mode is not 'psd'")
    x = np.asarray(x)
    if not same_data:
        y = np.asarray(y)
    if sides is None or sides == 'default':
        if np.iscomplexobj(x):
            sides = 'twosided'
        else:
            sides = 'onesided'
    _api.check_in_list(['default', 'onesided', 'twosided'], sides=sides)
    if len(x) < NFFT:
        n = len(x)
        x = np.resize(x, NFFT)
        x[n:] = 0
    if not same_data and len(y) < NFFT:
        n = len(y)
        y = np.resize(y, NFFT)
        y[n:] = 0
    if pad_to is None:
        pad_to = NFFT
    if mode != 'psd':
        scale_by_freq = False
    elif scale_by_freq is None:
        scale_by_freq = True
    if sides == 'twosided':
        numFreqs = pad_to
        if pad_to % 2:
            freqcenter = (pad_to - 1) // 2 + 1
        else:
            freqcenter = pad_to // 2
        scaling_factor = 1.0
    elif sides == 'onesided':
        if pad_to % 2:
            numFreqs = (pad_to + 1) // 2
        else:
            numFreqs = pad_to // 2 + 1
        scaling_factor = 2.0
    if not np.iterable(window):
        window = window(np.ones(NFFT, x.dtype))
    if len(window) != NFFT:
        raise ValueError("The window length must match the data's first dimension")
    result = np.lib.stride_tricks.sliding_window_view(x, NFFT, axis=0)[::NFFT - noverlap].T
    result = detrend(result, detrend_func, axis=0)
    result = result * window.reshape((-1, 1))
    result = np.fft.fft(result, n=pad_to, axis=0)[:numFreqs, :]
    freqs = np.fft.fftfreq(pad_to, 1 / Fs)[:numFreqs]
    if not same_data:
        resultY = np.lib.stride_tricks.sliding_window_view(y, NFFT, axis=0)[::NFFT - noverlap].T
        resultY = detrend(resultY, detrend_func, axis=0)
        resultY = resultY * window.reshape((-1, 1))
        resultY = np.fft.fft(resultY, n=pad_to, axis=0)[:numFreqs, :]
        result = np.conj(result) * resultY
    elif mode == 'psd':
        result = np.conj(result) * result
    elif mode == 'magnitude':
        result = np.abs(result) / window.sum()
    elif mode == 'angle' or mode == 'phase':
        result = np.angle(result)
    elif mode == 'complex':
        result /= window.sum()
    if mode == 'psd':
        if not NFFT % 2:
            slc = slice(1, -1, None)
        else:
            slc = slice(1, None, None)
        result[slc] *= scaling_factor
        if scale_by_freq:
            result /= Fs
            result /= (window ** 2).sum()
        else:
            result /= window.sum() ** 2
    t = np.arange(NFFT / 2, len(x) - NFFT / 2 + 1, NFFT - noverlap) / Fs
    if sides == 'twosided':
        freqs = np.roll(freqs, -freqcenter, axis=0)
        result = np.roll(result, -freqcenter, axis=0)
    elif not pad_to % 2:
        freqs[-1] *= -1
    if mode == 'phase':
        result = np.unwrap(result, axis=0)
    return (result, freqs, t)

def _single_spectrum_helper(mode, x, Fs=None, window=None, pad_to=None, sides=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Private helper implementing the commonality between the complex, magnitude,\n    angle, and phase spectrums.\n    '
    _api.check_in_list(['complex', 'magnitude', 'angle', 'phase'], mode=mode)
    if pad_to is None:
        pad_to = len(x)
    (spec, freqs, _) = _spectral_helper(x=x, y=None, NFFT=len(x), Fs=Fs, detrend_func=detrend_none, window=window, noverlap=0, pad_to=pad_to, sides=sides, scale_by_freq=False, mode=mode)
    if mode != 'complex':
        spec = spec.real
    if spec.ndim == 2 and spec.shape[1] == 1:
        spec = spec[:, 0]
    return (spec, freqs)
_docstring.interpd.update(Spectral="Fs : float, default: 2\n    The sampling frequency (samples per time unit).  It is used to calculate\n    the Fourier frequencies, *freqs*, in cycles per time unit.\n\nwindow : callable or ndarray, default: `.window_hanning`\n    A function or a vector of length *NFFT*.  To create window vectors see\n    `.window_hanning`, `.window_none`, `numpy.blackman`, `numpy.hamming`,\n    `numpy.bartlett`, `scipy.signal`, `scipy.signal.get_window`, etc.  If a\n    function is passed as the argument, it must take a data segment as an\n    argument and return the windowed version of the segment.\n\nsides : {'default', 'onesided', 'twosided'}, optional\n    Which sides of the spectrum to return. 'default' is one-sided for real\n    data and two-sided for complex data. 'onesided' forces the return of a\n    one-sided spectrum, while 'twosided' forces two-sided.", Single_Spectrum='pad_to : int, optional\n    The number of points to which the data segment is padded when performing\n    the FFT.  While not increasing the actual resolution of the spectrum (the\n    minimum distance between resolvable peaks), this can give more points in\n    the plot, allowing for more detail. This corresponds to the *n* parameter\n    in the call to `~numpy.fft.fft`.  The default is None, which sets *pad_to*\n    equal to the length of the input signal (i.e. no padding).', PSD="pad_to : int, optional\n    The number of points to which the data segment is padded when performing\n    the FFT.  This can be different from *NFFT*, which specifies the number\n    of data points used.  While not increasing the actual resolution of the\n    spectrum (the minimum distance between resolvable peaks), this can give\n    more points in the plot, allowing for more detail. This corresponds to\n    the *n* parameter in the call to `~numpy.fft.fft`. The default is None,\n    which sets *pad_to* equal to *NFFT*\n\nNFFT : int, default: 256\n    The number of data points used in each block for the FFT.  A power 2 is\n    most efficient.  This should *NOT* be used to get zero padding, or the\n    scaling of the result will be incorrect; use *pad_to* for this instead.\n\ndetrend : {'none', 'mean', 'linear'} or callable, default: 'none'\n    The function applied to each segment before fft-ing, designed to remove\n    the mean or linear trend.  Unlike in MATLAB, where the *detrend* parameter\n    is a vector, in Matplotlib it is a function.  The :mod:`~matplotlib.mlab`\n    module defines `.detrend_none`, `.detrend_mean`, and `.detrend_linear`,\n    but you can use a custom function as well.  You can also use a string to\n    choose one of the functions: 'none' calls `.detrend_none`. 'mean' calls\n    `.detrend_mean`. 'linear' calls `.detrend_linear`.\n\nscale_by_freq : bool, default: True\n    Whether the resulting density values should be scaled by the scaling\n    frequency, which gives density in units of 1/Hz.  This allows for\n    integration over the returned frequency values.  The default is True for\n    MATLAB compatibility.")

@_docstring.dedent_interpd
def psd(x, NFFT=None, Fs=None, detrend=None, window=None, noverlap=None, pad_to=None, sides=None, scale_by_freq=None):
    if False:
        return 10
    "\n    Compute the power spectral density.\n\n    The power spectral density :math:`P_{xx}` by Welch's average\n    periodogram method.  The vector *x* is divided into *NFFT* length\n    segments.  Each segment is detrended by function *detrend* and\n    windowed by function *window*.  *noverlap* gives the length of\n    the overlap between segments.  The :math:`|\\mathrm{fft}(i)|^2`\n    of each segment :math:`i` are averaged to compute :math:`P_{xx}`.\n\n    If len(*x*) < *NFFT*, it will be zero padded to *NFFT*.\n\n    Parameters\n    ----------\n    x : 1-D array or sequence\n        Array or sequence containing the data\n\n    %(Spectral)s\n\n    %(PSD)s\n\n    noverlap : int, default: 0 (no overlap)\n        The number of points of overlap between segments.\n\n    Returns\n    -------\n    Pxx : 1-D array\n        The values for the power spectrum :math:`P_{xx}` (real valued)\n\n    freqs : 1-D array\n        The frequencies corresponding to the elements in *Pxx*\n\n    References\n    ----------\n    Bendat & Piersol -- Random Data: Analysis and Measurement Procedures, John\n    Wiley & Sons (1986)\n\n    See Also\n    --------\n    specgram\n        `specgram` differs in the default overlap; in not returning the mean of\n        the segment periodograms; and in returning the times of the segments.\n\n    magnitude_spectrum : returns the magnitude spectrum.\n\n    csd : returns the spectral density between two signals.\n    "
    (Pxx, freqs) = csd(x=x, y=None, NFFT=NFFT, Fs=Fs, detrend=detrend, window=window, noverlap=noverlap, pad_to=pad_to, sides=sides, scale_by_freq=scale_by_freq)
    return (Pxx.real, freqs)

@_docstring.dedent_interpd
def csd(x, y, NFFT=None, Fs=None, detrend=None, window=None, noverlap=None, pad_to=None, sides=None, scale_by_freq=None):
    if False:
        i = 10
        return i + 15
    "\n    Compute the cross-spectral density.\n\n    The cross spectral density :math:`P_{xy}` by Welch's average\n    periodogram method.  The vectors *x* and *y* are divided into\n    *NFFT* length segments.  Each segment is detrended by function\n    *detrend* and windowed by function *window*.  *noverlap* gives\n    the length of the overlap between segments.  The product of\n    the direct FFTs of *x* and *y* are averaged over each segment\n    to compute :math:`P_{xy}`, with a scaling to correct for power\n    loss due to windowing.\n\n    If len(*x*) < *NFFT* or len(*y*) < *NFFT*, they will be zero\n    padded to *NFFT*.\n\n    Parameters\n    ----------\n    x, y : 1-D arrays or sequences\n        Arrays or sequences containing the data\n\n    %(Spectral)s\n\n    %(PSD)s\n\n    noverlap : int, default: 0 (no overlap)\n        The number of points of overlap between segments.\n\n    Returns\n    -------\n    Pxy : 1-D array\n        The values for the cross spectrum :math:`P_{xy}` before scaling (real\n        valued)\n\n    freqs : 1-D array\n        The frequencies corresponding to the elements in *Pxy*\n\n    References\n    ----------\n    Bendat & Piersol -- Random Data: Analysis and Measurement Procedures, John\n    Wiley & Sons (1986)\n\n    See Also\n    --------\n    psd : equivalent to setting ``y = x``.\n    "
    if NFFT is None:
        NFFT = 256
    (Pxy, freqs, _) = _spectral_helper(x=x, y=y, NFFT=NFFT, Fs=Fs, detrend_func=detrend, window=window, noverlap=noverlap, pad_to=pad_to, sides=sides, scale_by_freq=scale_by_freq, mode='psd')
    if Pxy.ndim == 2:
        if Pxy.shape[1] > 1:
            Pxy = Pxy.mean(axis=1)
        else:
            Pxy = Pxy[:, 0]
    return (Pxy, freqs)
_single_spectrum_docs = 'Compute the {quantity} of *x*.\nData is padded to a length of *pad_to* and the windowing function *window* is\napplied to the signal.\n\nParameters\n----------\nx : 1-D array or sequence\n    Array or sequence containing the data\n\n{Spectral}\n\n{Single_Spectrum}\n\nReturns\n-------\nspectrum : 1-D array\n    The {quantity}.\nfreqs : 1-D array\n    The frequencies corresponding to the elements in *spectrum*.\n\nSee Also\n--------\npsd\n    Returns the power spectral density.\ncomplex_spectrum\n    Returns the complex-valued frequency spectrum.\nmagnitude_spectrum\n    Returns the absolute value of the `complex_spectrum`.\nangle_spectrum\n    Returns the angle of the `complex_spectrum`.\nphase_spectrum\n    Returns the phase (unwrapped angle) of the `complex_spectrum`.\nspecgram\n    Can return the complex spectrum of segments within the signal.\n'
complex_spectrum = functools.partial(_single_spectrum_helper, 'complex')
complex_spectrum.__doc__ = _single_spectrum_docs.format(quantity='complex-valued frequency spectrum', **_docstring.interpd.params)
magnitude_spectrum = functools.partial(_single_spectrum_helper, 'magnitude')
magnitude_spectrum.__doc__ = _single_spectrum_docs.format(quantity='magnitude (absolute value) of the frequency spectrum', **_docstring.interpd.params)
angle_spectrum = functools.partial(_single_spectrum_helper, 'angle')
angle_spectrum.__doc__ = _single_spectrum_docs.format(quantity='angle of the frequency spectrum (wrapped phase spectrum)', **_docstring.interpd.params)
phase_spectrum = functools.partial(_single_spectrum_helper, 'phase')
phase_spectrum.__doc__ = _single_spectrum_docs.format(quantity='phase of the frequency spectrum (unwrapped phase spectrum)', **_docstring.interpd.params)

@_docstring.dedent_interpd
def specgram(x, NFFT=None, Fs=None, detrend=None, window=None, noverlap=None, pad_to=None, sides=None, scale_by_freq=None, mode=None):
    if False:
        while True:
            i = 10
    "\n    Compute a spectrogram.\n\n    Compute and plot a spectrogram of data in *x*.  Data are split into\n    *NFFT* length segments and the spectrum of each section is\n    computed.  The windowing function *window* is applied to each\n    segment, and the amount of overlap of each segment is\n    specified with *noverlap*.\n\n    Parameters\n    ----------\n    x : array-like\n        1-D array or sequence.\n\n    %(Spectral)s\n\n    %(PSD)s\n\n    noverlap : int, default: 128\n        The number of points of overlap between blocks.\n    mode : str, default: 'psd'\n        What sort of spectrum to use:\n            'psd'\n                Returns the power spectral density.\n            'complex'\n                Returns the complex-valued frequency spectrum.\n            'magnitude'\n                Returns the magnitude spectrum.\n            'angle'\n                Returns the phase spectrum without unwrapping.\n            'phase'\n                Returns the phase spectrum with unwrapping.\n\n    Returns\n    -------\n    spectrum : array-like\n        2D array, columns are the periodograms of successive segments.\n\n    freqs : array-like\n        1-D array, frequencies corresponding to the rows in *spectrum*.\n\n    t : array-like\n        1-D array, the times corresponding to midpoints of segments\n        (i.e the columns in *spectrum*).\n\n    See Also\n    --------\n    psd : differs in the overlap and in the return values.\n    complex_spectrum : similar, but with complex valued frequencies.\n    magnitude_spectrum : similar single segment when *mode* is 'magnitude'.\n    angle_spectrum : similar to single segment when *mode* is 'angle'.\n    phase_spectrum : similar to single segment when *mode* is 'phase'.\n\n    Notes\n    -----\n    *detrend* and *scale_by_freq* only apply when *mode* is set to 'psd'.\n\n    "
    if noverlap is None:
        noverlap = 128
    if NFFT is None:
        NFFT = 256
    if len(x) <= NFFT:
        _api.warn_external(f'Only one segment is calculated since parameter NFFT (={NFFT}) >= signal length (={len(x)}).')
    (spec, freqs, t) = _spectral_helper(x=x, y=None, NFFT=NFFT, Fs=Fs, detrend_func=detrend, window=window, noverlap=noverlap, pad_to=pad_to, sides=sides, scale_by_freq=scale_by_freq, mode=mode)
    if mode != 'complex':
        spec = spec.real
    return (spec, freqs, t)

@_docstring.dedent_interpd
def cohere(x, y, NFFT=256, Fs=2, detrend=detrend_none, window=window_hanning, noverlap=0, pad_to=None, sides='default', scale_by_freq=None):
    if False:
        while True:
            i = 10
    '\n    The coherence between *x* and *y*.  Coherence is the normalized\n    cross spectral density:\n\n    .. math::\n\n        C_{xy} = \\frac{|P_{xy}|^2}{P_{xx}P_{yy}}\n\n    Parameters\n    ----------\n    x, y\n        Array or sequence containing the data\n\n    %(Spectral)s\n\n    %(PSD)s\n\n    noverlap : int, default: 0 (no overlap)\n        The number of points of overlap between segments.\n\n    Returns\n    -------\n    Cxy : 1-D array\n        The coherence vector.\n    freqs : 1-D array\n            The frequencies for the elements in *Cxy*.\n\n    See Also\n    --------\n    :func:`psd`, :func:`csd` :\n        For information about the methods used to compute :math:`P_{xy}`,\n        :math:`P_{xx}` and :math:`P_{yy}`.\n    '
    if len(x) < 2 * NFFT:
        raise ValueError('Coherence is calculated by averaging over *NFFT* length segments.  Your signal is too short for your choice of *NFFT*.')
    (Pxx, f) = psd(x, NFFT, Fs, detrend, window, noverlap, pad_to, sides, scale_by_freq)
    (Pyy, f) = psd(y, NFFT, Fs, detrend, window, noverlap, pad_to, sides, scale_by_freq)
    (Pxy, f) = csd(x, y, NFFT, Fs, detrend, window, noverlap, pad_to, sides, scale_by_freq)
    Cxy = np.abs(Pxy) ** 2 / (Pxx * Pyy)
    return (Cxy, f)

class GaussianKDE:
    """
    Representation of a kernel-density estimate using Gaussian kernels.

    Parameters
    ----------
    dataset : array-like
        Datapoints to estimate from. In case of univariate data this is a 1-D
        array, otherwise a 2D array with shape (# of dims, # of data).
    bw_method : str, scalar or callable, optional
        The method used to calculate the estimator bandwidth.  This can be
        'scott', 'silverman', a scalar constant or a callable.  If a
        scalar, this will be used directly as `kde.factor`.  If a
        callable, it should take a `GaussianKDE` instance as only
        parameter and return a scalar. If None (default), 'scott' is used.

    Attributes
    ----------
    dataset : ndarray
        The dataset passed to the constructor.
    dim : int
        Number of dimensions.
    num_dp : int
        Number of datapoints.
    factor : float
        The bandwidth factor, obtained from `kde.covariance_factor`, with which
        the covariance matrix is multiplied.
    covariance : ndarray
        The covariance matrix of *dataset*, scaled by the calculated bandwidth
        (`kde.factor`).
    inv_cov : ndarray
        The inverse of *covariance*.

    Methods
    -------
    kde.evaluate(points) : ndarray
        Evaluate the estimated pdf on a provided set of points.
    kde(points) : ndarray
        Same as kde.evaluate(points)
    """

    def __init__(self, dataset, bw_method=None):
        if False:
            for i in range(10):
                print('nop')
        self.dataset = np.atleast_2d(dataset)
        if not np.array(self.dataset).size > 1:
            raise ValueError('`dataset` input should have multiple elements.')
        (self.dim, self.num_dp) = np.array(self.dataset).shape
        if bw_method is None:
            pass
        elif cbook._str_equal(bw_method, 'scott'):
            self.covariance_factor = self.scotts_factor
        elif cbook._str_equal(bw_method, 'silverman'):
            self.covariance_factor = self.silverman_factor
        elif isinstance(bw_method, Number):
            self._bw_method = 'use constant'
            self.covariance_factor = lambda : bw_method
        elif callable(bw_method):
            self._bw_method = bw_method
            self.covariance_factor = lambda : self._bw_method(self)
        else:
            raise ValueError("`bw_method` should be 'scott', 'silverman', a scalar or a callable")
        self.factor = self.covariance_factor()
        if not hasattr(self, '_data_inv_cov'):
            self.data_covariance = np.atleast_2d(np.cov(self.dataset, rowvar=1, bias=False))
            self.data_inv_cov = np.linalg.inv(self.data_covariance)
        self.covariance = self.data_covariance * self.factor ** 2
        self.inv_cov = self.data_inv_cov / self.factor ** 2
        self.norm_factor = np.sqrt(np.linalg.det(2 * np.pi * self.covariance)) * self.num_dp

    def scotts_factor(self):
        if False:
            print('Hello World!')
        return np.power(self.num_dp, -1.0 / (self.dim + 4))

    def silverman_factor(self):
        if False:
            for i in range(10):
                print('nop')
        return np.power(self.num_dp * (self.dim + 2.0) / 4.0, -1.0 / (self.dim + 4))
    covariance_factor = scotts_factor

    def evaluate(self, points):
        if False:
            i = 10
            return i + 15
        '\n        Evaluate the estimated pdf on a set of points.\n\n        Parameters\n        ----------\n        points : (# of dimensions, # of points)-array\n            Alternatively, a (# of dimensions,) vector can be passed in and\n            treated as a single point.\n\n        Returns\n        -------\n        (# of points,)-array\n            The values at each point.\n\n        Raises\n        ------\n        ValueError : if the dimensionality of the input points is different\n                     than the dimensionality of the KDE.\n\n        '
        points = np.atleast_2d(points)
        (dim, num_m) = np.array(points).shape
        if dim != self.dim:
            raise ValueError(f'points have dimension {dim}, dataset has dimension {self.dim}')
        result = np.zeros(num_m)
        if num_m >= self.num_dp:
            for i in range(self.num_dp):
                diff = self.dataset[:, i, np.newaxis] - points
                tdiff = np.dot(self.inv_cov, diff)
                energy = np.sum(diff * tdiff, axis=0) / 2.0
                result = result + np.exp(-energy)
        else:
            for i in range(num_m):
                diff = self.dataset - points[:, i, np.newaxis]
                tdiff = np.dot(self.inv_cov, diff)
                energy = np.sum(diff * tdiff, axis=0) / 2.0
                result[i] = np.sum(np.exp(-energy), axis=0)
        result = result / self.norm_factor
        return result
    __call__ = evaluate