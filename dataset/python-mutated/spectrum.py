"""Utilities for spectral processing"""
import warnings
import numpy as np
import scipy
import scipy.ndimage
import scipy.signal
import scipy.interpolate
from numba import jit
from . import convert
from .fft import get_fftlib
from .audio import resample
from .._cache import cache
from .. import util
from ..util.exceptions import ParameterError
from ..filters import get_window, semitone_filterbank
from ..filters import window_sumsquare
from numpy.typing import DTypeLike
from typing import Any, Callable, Optional, Tuple, List, Union, overload
from typing_extensions import Literal
from .._typing import _WindowSpec, _PadMode, _PadModeSTFT
__all__ = ['stft', 'istft', 'magphase', 'iirt', 'reassigned_spectrogram', 'phase_vocoder', 'perceptual_weighting', 'power_to_db', 'db_to_power', 'amplitude_to_db', 'db_to_amplitude', 'fmt', 'pcen', 'griffinlim']

@cache(level=20)
def stft(y: np.ndarray, *, n_fft: int=2048, hop_length: Optional[int]=None, win_length: Optional[int]=None, window: _WindowSpec='hann', center: bool=True, dtype: Optional[DTypeLike]=None, pad_mode: _PadModeSTFT='constant', out: Optional[np.ndarray]=None) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Short-time Fourier transform (STFT).\n\n    The STFT represents a signal in the time-frequency domain by\n    computing discrete Fourier transforms (DFT) over short overlapping\n    windows.\n\n    This function returns a complex-valued matrix D such that\n\n    - ``np.abs(D[..., f, t])`` is the magnitude of frequency bin ``f``\n      at frame ``t``, and\n\n    - ``np.angle(D[..., f, t])`` is the phase of frequency bin ``f``\n      at frame ``t``.\n\n    The integers ``t`` and ``f`` can be converted to physical units by means\n    of the utility functions `frames_to_samples` and `fft_frequencies`.\n\n    Parameters\n    ----------\n    y : np.ndarray [shape=(..., n)], real-valued\n        input signal. Multi-channel is supported.\n\n    n_fft : int > 0 [scalar]\n        length of the windowed signal after padding with zeros.\n        The number of rows in the STFT matrix ``D`` is ``(1 + n_fft/2)``.\n        The default value, ``n_fft=2048`` samples, corresponds to a physical\n        duration of 93 milliseconds at a sample rate of 22050 Hz, i.e. the\n        default sample rate in librosa. This value is well adapted for music\n        signals. However, in speech processing, the recommended value is 512,\n        corresponding to 23 milliseconds at a sample rate of 22050 Hz.\n        In any case, we recommend setting ``n_fft`` to a power of two for\n        optimizing the speed of the fast Fourier transform (FFT) algorithm.\n\n    hop_length : int > 0 [scalar]\n        number of audio samples between adjacent STFT columns.\n\n        Smaller values increase the number of columns in ``D`` without\n        affecting the frequency resolution of the STFT.\n\n        If unspecified, defaults to ``win_length // 4`` (see below).\n\n    win_length : int <= n_fft [scalar]\n        Each frame of audio is windowed by ``window`` of length ``win_length``\n        and then padded with zeros to match ``n_fft``.  Padding is added on\n        both the left- and the right-side of the window so that the window\n        is centered within the frame.\n\n        Smaller values improve the temporal resolution of the STFT (i.e. the\n        ability to discriminate impulses that are closely spaced in time)\n        at the expense of frequency resolution (i.e. the ability to discriminate\n        pure tones that are closely spaced in frequency). This effect is known\n        as the time-frequency localization trade-off and needs to be adjusted\n        according to the properties of the input signal ``y``.\n\n        If unspecified, defaults to ``win_length = n_fft``.\n\n    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]\n        Either:\n\n        - a window specification (string, tuple, or number);\n          see `scipy.signal.get_window`\n        - a window function, such as `scipy.signal.windows.hann`\n        - a vector or array of length ``n_fft``\n\n        Defaults to a raised cosine window (`\'hann\'`), which is adequate for\n        most applications in audio signal processing.\n\n        .. see also:: `filters.get_window`\n\n    center : boolean\n        If ``True``, the signal ``y`` is padded so that frame\n        ``D[:, t]`` is centered at ``y[t * hop_length]``.\n\n        If ``False``, then ``D[:, t]`` begins at ``y[t * hop_length]``.\n\n        Defaults to ``True``,  which simplifies the alignment of ``D`` onto a\n        time grid by means of `librosa.frames_to_samples`.\n        Note, however, that ``center`` must be set to `False` when analyzing\n        signals with `librosa.stream`.\n\n        .. see also:: `librosa.stream`\n\n    dtype : np.dtype, optional\n        Complex numeric type for ``D``.  Default is inferred to match the\n        precision of the input signal.\n\n    pad_mode : string or function\n        If ``center=True``, this argument is passed to `np.pad` for padding\n        the edges of the signal ``y``. By default (``pad_mode="constant"``),\n        ``y`` is padded on both sides with zeros.\n\n        .. note:: Not all padding modes supported by `numpy.pad` are supported here.\n            `wrap`, `mean`, `maximum`, `median`, and `minimum` are not supported.\n\n            Other modes that depend at most on input values at the edges of the\n            signal (e.g., `constant`, `edge`, `linear_ramp`) are supported.\n\n        If ``center=False``,  this argument is ignored.\n\n        .. see also:: `numpy.pad`\n\n    out : np.ndarray or None\n        A pre-allocated, complex-valued array to store the STFT results.\n        This must be of compatible shape and dtype for the given input parameters.\n\n        If `out` is larger than necessary for the provided input signal, then only\n        a prefix slice of `out` will be used.\n\n        If not provided, a new array is allocated and returned.\n\n    Returns\n    -------\n    D : np.ndarray [shape=(..., 1 + n_fft/2, n_frames), dtype=dtype]\n        Complex-valued matrix of short-term Fourier transform\n        coefficients.\n\n        If a pre-allocated `out` array is provided, then `D` will be\n        a reference to `out`.\n\n        If `out` is larger than necessary, then `D` will be a sliced\n        view: `D = out[..., :n_frames]`.\n\n    See Also\n    --------\n    istft : Inverse STFT\n    reassigned_spectrogram : Time-frequency reassigned spectrogram\n\n    Notes\n    -----\n    This function caches at level 20.\n\n    Examples\n    --------\n    >>> y, sr = librosa.load(librosa.ex(\'trumpet\'))\n    >>> S = np.abs(librosa.stft(y))\n    >>> S\n    array([[5.395e-03, 3.332e-03, ..., 9.862e-07, 1.201e-05],\n           [3.244e-03, 2.690e-03, ..., 9.536e-07, 1.201e-05],\n           ...,\n           [7.523e-05, 3.722e-05, ..., 1.188e-04, 1.031e-03],\n           [7.640e-05, 3.944e-05, ..., 5.180e-04, 1.346e-03]],\n          dtype=float32)\n\n    Use left-aligned frames, instead of centered frames\n\n    >>> S_left = librosa.stft(y, center=False)\n\n    Use a shorter hop length\n\n    >>> D_short = librosa.stft(y, hop_length=64)\n\n    Display a spectrogram\n\n    >>> import matplotlib.pyplot as plt\n    >>> fig, ax = plt.subplots()\n    >>> img = librosa.display.specshow(librosa.amplitude_to_db(S,\n    ...                                                        ref=np.max),\n    ...                                y_axis=\'log\', x_axis=\'time\', ax=ax)\n    >>> ax.set_title(\'Power spectrogram\')\n    >>> fig.colorbar(img, ax=ax, format="%+2.0f dB")\n    '
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = int(win_length // 4)
    elif not util.is_positive_int(hop_length):
        raise ParameterError(f'hop_length={hop_length} must be a positive integer')
    util.valid_audio(y, mono=False)
    fft_window = get_window(window, win_length, fftbins=True)
    fft_window = util.pad_center(fft_window, size=n_fft)
    fft_window = util.expand_to(fft_window, ndim=1 + y.ndim, axes=-2)
    if center:
        if pad_mode in ('wrap', 'maximum', 'mean', 'median', 'minimum'):
            raise ParameterError(f"pad_mode='{pad_mode}' is not supported by librosa.stft")
        if n_fft > y.shape[-1]:
            warnings.warn(f'n_fft={n_fft} is too large for input signal of length={y.shape[-1]}')
        padding = [(0, 0) for _ in range(y.ndim)]
        start_k = int(np.ceil(n_fft // 2 / hop_length))
        tail_k = (y.shape[-1] + n_fft // 2 - n_fft) // hop_length + 1
        if tail_k <= start_k:
            start = 0
            extra = 0
            padding[-1] = (n_fft // 2, n_fft // 2)
            y = np.pad(y, padding, mode=pad_mode)
        else:
            start = start_k * hop_length - n_fft // 2
            padding[-1] = (n_fft // 2, 0)
            y_pre = np.pad(y[..., :(start_k - 1) * hop_length - n_fft // 2 + n_fft + 1], padding, mode=pad_mode)
            y_frames_pre = util.frame(y_pre, frame_length=n_fft, hop_length=hop_length)
            y_frames_pre = y_frames_pre[..., :start_k]
            extra = y_frames_pre.shape[-1]
            if tail_k * hop_length - n_fft // 2 + n_fft <= y.shape[-1] + n_fft // 2:
                padding[-1] = (0, n_fft // 2)
                y_post = np.pad(y[..., tail_k * hop_length - n_fft // 2:], padding, mode=pad_mode)
                y_frames_post = util.frame(y_post, frame_length=n_fft, hop_length=hop_length)
                extra += y_frames_post.shape[-1]
            else:
                post_shape = list(y_frames_pre.shape)
                post_shape[-1] = 0
                y_frames_post = np.empty_like(y_frames_pre, shape=post_shape)
    else:
        if n_fft > y.shape[-1]:
            raise ParameterError(f'n_fft={n_fft} is too large for uncentered analysis of input signal of length={y.shape[-1]}')
        start = 0
        extra = 0
    fft = get_fftlib()
    if dtype is None:
        dtype = util.dtype_r2c(y.dtype)
    y_frames = util.frame(y[..., start:], frame_length=n_fft, hop_length=hop_length)
    shape = list(y_frames.shape)
    shape[-2] = 1 + n_fft // 2
    shape[-1] += extra
    if out is None:
        stft_matrix = np.zeros(shape, dtype=dtype, order='F')
    elif not (np.allclose(out.shape[:-1], shape[:-1]) and out.shape[-1] >= shape[-1]):
        raise ParameterError(f'Shape mismatch for provided output array out.shape={out.shape} and target shape={shape}')
    elif not np.iscomplexobj(out):
        raise ParameterError(f'output with dtype={out.dtype} is not of complex type')
    elif np.allclose(shape, out.shape):
        stft_matrix = out
    else:
        stft_matrix = out[..., :shape[-1]]
    if center and extra > 0:
        off_start = y_frames_pre.shape[-1]
        stft_matrix[..., :off_start] = fft.rfft(fft_window * y_frames_pre, axis=-2)
        off_end = y_frames_post.shape[-1]
        if off_end > 0:
            stft_matrix[..., -off_end:] = fft.rfft(fft_window * y_frames_post, axis=-2)
    else:
        off_start = 0
    n_columns = int(util.MAX_MEM_BLOCK // (np.prod(y_frames.shape[:-1]) * y_frames.itemsize))
    n_columns = max(n_columns, 1)
    for bl_s in range(0, y_frames.shape[-1], n_columns):
        bl_t = min(bl_s + n_columns, y_frames.shape[-1])
        stft_matrix[..., bl_s + off_start:bl_t + off_start] = fft.rfft(fft_window * y_frames[..., bl_s:bl_t], axis=-2)
    return stft_matrix

@cache(level=30)
def istft(stft_matrix: np.ndarray, *, hop_length: Optional[int]=None, win_length: Optional[int]=None, n_fft: Optional[int]=None, window: _WindowSpec='hann', center: bool=True, dtype: Optional[DTypeLike]=None, length: Optional[int]=None, out: Optional[np.ndarray]=None) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    '\n    Inverse short-time Fourier transform (ISTFT).\n\n    Converts a complex-valued spectrogram ``stft_matrix`` to time-series ``y``\n    by minimizing the mean squared error between ``stft_matrix`` and STFT of\n    ``y`` as described in [#]_ up to Section 2 (reconstruction from MSTFT).\n\n    In general, window function, hop length and other parameters should be same\n    as in stft, which mostly leads to perfect reconstruction of a signal from\n    unmodified ``stft_matrix``.\n\n    .. [#] D. W. Griffin and J. S. Lim,\n        "Signal estimation from modified short-time Fourier transform,"\n        IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.\n\n    Parameters\n    ----------\n    stft_matrix : np.ndarray [shape=(..., 1 + n_fft//2, t)]\n        STFT matrix from ``stft``\n\n    hop_length : int > 0 [scalar]\n        Number of frames between STFT columns.\n        If unspecified, defaults to ``win_length // 4``.\n\n    win_length : int <= n_fft = 2 * (stft_matrix.shape[0] - 1)\n        When reconstructing the time series, each frame is windowed\n        and each sample is normalized by the sum of squared window\n        according to the ``window`` function (see below).\n\n        If unspecified, defaults to ``n_fft``.\n\n    n_fft : int > 0 or None\n        The number of samples per frame in the input spectrogram.\n        By default, this will be inferred from the shape of ``stft_matrix``.\n        However, if an odd frame length was used, you can specify the correct\n        length by setting ``n_fft``.\n\n    window : string, tuple, number, function, np.ndarray [shape=(n_fft,)]\n        - a window specification (string, tuple, or number);\n          see `scipy.signal.get_window`\n        - a window function, such as `scipy.signal.windows.hann`\n        - a user-specified window vector of length ``n_fft``\n\n        .. see also:: `filters.get_window`\n\n    center : boolean\n        - If ``True``, ``D`` is assumed to have centered frames.\n        - If ``False``, ``D`` is assumed to have left-aligned frames.\n\n    dtype : numeric type\n        Real numeric type for ``y``.  Default is to match the numerical\n        precision of the input spectrogram.\n\n    length : int > 0, optional\n        If provided, the output ``y`` is zero-padded or clipped to exactly\n        ``length`` samples.\n\n    out : np.ndarray or None\n        A pre-allocated, complex-valued array to store the reconstructed signal\n        ``y``.  This must be of the correct shape for the given input parameters.\n\n        If not provided, a new array is allocated and returned.\n\n    Returns\n    -------\n    y : np.ndarray [shape=(..., n)]\n        time domain signal reconstructed from ``stft_matrix``.\n        If ``stft_matrix`` contains more than two axes\n        (e.g., from a stereo input signal), then ``y`` will match shape on the leading dimensions.\n\n    See Also\n    --------\n    stft : Short-time Fourier Transform\n\n    Notes\n    -----\n    This function caches at level 30.\n\n    Examples\n    --------\n    >>> y, sr = librosa.load(librosa.ex(\'trumpet\'))\n    >>> D = librosa.stft(y)\n    >>> y_hat = librosa.istft(D)\n    >>> y_hat\n    array([-1.407e-03, -4.461e-04, ...,  5.131e-06, -1.417e-05],\n          dtype=float32)\n\n    Exactly preserving length of the input signal requires explicit padding.\n    Otherwise, a partial frame at the end of ``y`` will not be represented.\n\n    >>> n = len(y)\n    >>> n_fft = 2048\n    >>> y_pad = librosa.util.fix_length(y, size=n + n_fft // 2)\n    >>> D = librosa.stft(y_pad, n_fft=n_fft)\n    >>> y_out = librosa.istft(D, length=n)\n    >>> np.max(np.abs(y - y_out))\n    8.940697e-08\n    '
    if n_fft is None:
        n_fft = 2 * (stft_matrix.shape[-2] - 1)
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = int(win_length // 4)
    ifft_window = get_window(window, win_length, fftbins=True)
    ifft_window = util.pad_center(ifft_window, size=n_fft)
    ifft_window = util.expand_to(ifft_window, ndim=stft_matrix.ndim, axes=-2)
    if length:
        if center:
            padded_length = length + 2 * (n_fft // 2)
        else:
            padded_length = length
        n_frames = min(stft_matrix.shape[-1], int(np.ceil(padded_length / hop_length)))
    else:
        n_frames = stft_matrix.shape[-1]
    if dtype is None:
        dtype = util.dtype_c2r(stft_matrix.dtype)
    shape = list(stft_matrix.shape[:-2])
    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    if length:
        expected_signal_len = length
    elif center:
        expected_signal_len -= 2 * (n_fft // 2)
    shape.append(expected_signal_len)
    if out is None:
        y = np.zeros(shape, dtype=dtype)
    elif not np.allclose(out.shape, shape):
        raise ParameterError(f'Shape mismatch for provided output array out.shape={out.shape} != {shape}')
    else:
        y = out
        y.fill(0.0)
    fft = get_fftlib()
    if center:
        start_frame = int(np.ceil(n_fft // 2 / hop_length))
        ytmp = ifft_window * fft.irfft(stft_matrix[..., :start_frame], n=n_fft, axis=-2)
        shape[-1] = n_fft + hop_length * (start_frame - 1)
        head_buffer = np.zeros(shape, dtype=dtype)
        __overlap_add(head_buffer, ytmp, hop_length)
        if y.shape[-1] < shape[-1] - n_fft // 2:
            y[..., :] = head_buffer[..., n_fft // 2:y.shape[-1] + n_fft // 2]
        else:
            y[..., :shape[-1] - n_fft // 2] = head_buffer[..., n_fft // 2:]
        offset = start_frame * hop_length - n_fft // 2
    else:
        start_frame = 0
        offset = 0
    n_columns = int(util.MAX_MEM_BLOCK // (np.prod(stft_matrix.shape[:-1]) * stft_matrix.itemsize))
    n_columns = max(n_columns, 1)
    frame = 0
    for bl_s in range(start_frame, n_frames, n_columns):
        bl_t = min(bl_s + n_columns, n_frames)
        ytmp = ifft_window * fft.irfft(stft_matrix[..., bl_s:bl_t], n=n_fft, axis=-2)
        __overlap_add(y[..., frame * hop_length + offset:], ytmp, hop_length)
        frame += bl_t - bl_s
    ifft_window_sum = window_sumsquare(window=window, n_frames=n_frames, win_length=win_length, n_fft=n_fft, hop_length=hop_length, dtype=dtype)
    if center:
        start = n_fft // 2
    else:
        start = 0
    ifft_window_sum = util.fix_length(ifft_window_sum[..., start:], size=y.shape[-1])
    approx_nonzero_indices = ifft_window_sum > util.tiny(ifft_window_sum)
    y[..., approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]
    return y

@jit(nopython=True, cache=True)
def __overlap_add(y, ytmp, hop_length):
    if False:
        for i in range(10):
            print('nop')
    n_fft = ytmp.shape[-2]
    N = n_fft
    for frame in range(ytmp.shape[-1]):
        sample = frame * hop_length
        if N > y.shape[-1] - sample:
            N = y.shape[-1] - sample
        y[..., sample:sample + N] += ytmp[..., :N, frame]

def __reassign_frequencies(y: np.ndarray, sr: float=22050, S: Optional[np.ndarray]=None, n_fft: int=2048, hop_length: Optional[int]=None, win_length: Optional[int]=None, window: _WindowSpec='hann', center: bool=True, dtype: Optional[DTypeLike]=None, pad_mode: _PadModeSTFT='constant') -> Tuple[np.ndarray, np.ndarray]:
    if False:
        i = 10
        return i + 15
    'Instantaneous frequencies based on a spectrogram representation.\n\n    The reassignment vector is calculated using equation 5.20 in Flandrin,\n    Auger, & Chassande-Mottin 2002::\n\n        omega_reassigned = omega - np.imag(S_dh/S_h)\n\n    where ``S_h`` is the complex STFT calculated using the original window, and\n    ``S_dh`` is the complex STFT calculated using the derivative of the original\n    window.\n\n    See `reassigned_spectrogram` for references.\n\n    It is recommended to use ``pad_mode="wrap"`` or else ``center=False``, rather\n    than the defaults. Frequency reassignment assumes that the energy in each\n    FFT bin is associated with exactly one signal component. Reflection padding\n    at the edges of the signal may invalidate the reassigned estimates in the\n    boundary frames.\n\n    Parameters\n    ----------\n    y : np.ndarray [shape=(..., n,)], real-valued\n        audio time series. Multi-channel is supported.\n\n    sr : number > 0 [scalar]\n        sampling rate of ``y``\n\n    S : np.ndarray [shape=(..., d, t)] or None\n        (optional) complex STFT calculated using the other arguments provided\n        to `__reassign_frequencies`\n\n    n_fft : int > 0 [scalar]\n        FFT window size. Defaults to 2048.\n\n    hop_length : int > 0 [scalar]\n        hop length, number samples between subsequent frames.\n        If not supplied, defaults to ``win_length // 4``.\n\n    win_length : int > 0, <= n_fft\n        Window length. Defaults to ``n_fft``.\n        See ``stft`` for details.\n\n    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]\n        - a window specification (string, tuple, number);\n          see `scipy.signal.get_window`\n        - a window function, such as `scipy.signal.windows.hann`\n        - a user-specified window vector of length ``n_fft``\n\n        See `stft` for details.\n\n        .. see also:: `filters.get_window`\n\n    center : boolean\n        - If ``True``, the signal ``y`` is padded so that frame\n          ``S[:, t]`` is centered at ``y[t * hop_length]``.\n        - If ``False``, then ``S[:, t]`` begins at ``y[t * hop_length]``.\n\n    dtype : numeric type\n        Complex numeric type for ``S``. Default is inferred to match\n        the numerical precision of the input signal.\n\n    pad_mode : string\n        If ``center=True``, the padding mode to use at the edges of the signal.\n        By default, STFT uses zero padding.\n\n    Returns\n    -------\n    freqs : np.ndarray [shape=(..., 1 + n_fft/2, t), dtype=real]\n        Instantaneous frequencies:\n        ``freqs[f, t]`` is the frequency for bin ``f``, frame ``t``.\n    S : np.ndarray [shape=(..., 1 + n_fft/2, t), dtype=complex]\n        Short-time Fourier transform\n\n    Warns\n    -----\n    RuntimeWarning\n        Frequencies with zero support will produce a divide-by-zero warning and\n        will be returned as `np.nan`.\n\n    See Also\n    --------\n    stft : Short-time Fourier Transform\n    reassigned_spectrogram : Time-frequency reassigned spectrogram\n\n    Examples\n    --------\n    >>> y, sr = librosa.load(librosa.ex(\'trumpet\'))\n    >>> frequencies, S = librosa.core.spectrum.__reassign_frequencies(y, sr=sr)\n    >>> frequencies\n    array([[0.000e+00, 0.000e+00, ..., 0.000e+00, 0.000e+00],\n           [3.628e+00, 4.698e+00, ..., 1.239e+01, 1.072e+01],\n           ...,\n           [1.101e+04, 1.102e+04, ..., 1.105e+04, 1.102e+04],\n           [1.102e+04, 1.102e+04, ..., 1.102e+04, 1.102e+04]])\n    '
    if win_length is None:
        win_length = n_fft
    window = get_window(window, win_length, fftbins=True)
    window = util.pad_center(window, size=n_fft)
    if S is None:
        if dtype is None:
            dtype = util.dtype_r2c(y.dtype)
        S_h = stft(y=y, n_fft=n_fft, hop_length=hop_length, window=window, center=center, dtype=dtype, pad_mode=pad_mode)
    else:
        if dtype is None:
            dtype = S.dtype
        S_h = S
    window_derivative = util.cyclic_gradient(window)
    S_dh = stft(y=y, n_fft=n_fft, hop_length=hop_length, window=window_derivative, center=center, dtype=dtype, pad_mode=pad_mode)
    correction = -np.imag(S_dh / S_h)
    freqs = convert.fft_frequencies(sr=sr, n_fft=n_fft)
    freqs = util.expand_to(freqs, ndim=correction.ndim, axes=-2) + correction * (0.5 * sr / np.pi)
    return (freqs, S_h)

def __reassign_times(y: np.ndarray, sr: float=22050, S: Optional[np.ndarray]=None, n_fft: int=2048, hop_length: Optional[int]=None, win_length: Optional[int]=None, window: _WindowSpec='hann', center: bool=True, dtype: Optional[DTypeLike]=None, pad_mode: _PadModeSTFT='constant') -> Tuple[np.ndarray, np.ndarray]:
    if False:
        print('Hello World!')
    'Time reassignments based on a spectrogram representation.\n\n    The reassignment vector is calculated using equation 5.23 in Flandrin,\n    Auger, & Chassande-Mottin 2002::\n\n        t_reassigned = t + np.real(S_th/S_h)\n\n    where ``S_h`` is the complex STFT calculated using the original window, and\n    ``S_th`` is the complex STFT calculated using the original window multiplied\n    by the time offset from the window center.\n\n    See `reassigned_spectrogram` for references.\n\n    It is recommended to use ``pad_mode="constant"`` (zero padding) or else\n    ``center=False``, rather than the defaults. Time reassignment assumes that\n    the energy in each FFT bin is associated with exactly one impulse event.\n    Reflection padding at the edges of the signal may invalidate the reassigned\n    estimates in the boundary frames.\n\n    Parameters\n    ----------\n    y : np.ndarray [shape=(..., n,)], real-valued\n        audio time series. Multi-channel is supported.\n\n    sr : number > 0 [scalar]\n        sampling rate of ``y``\n\n    S : np.ndarray [shape=(..., d, t)] or None\n        (optional) complex STFT calculated using the other arguments provided\n        to `__reassign_times`\n\n    n_fft : int > 0 [scalar]\n        FFT window size. Defaults to 2048.\n\n    hop_length : int > 0 [scalar]\n        hop length, number samples between subsequent frames.\n        If not supplied, defaults to ``win_length // 4``.\n\n    win_length : int > 0, <= n_fft\n        Window length. Defaults to ``n_fft``.\n        See `stft` for details.\n\n    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]\n        - a window specification (string, tuple, number);\n          see `scipy.signal.get_window`\n        - a window function, such as `scipy.signal.windows.hann`\n        - a user-specified window vector of length ``n_fft``\n\n        See `stft` for details.\n\n        .. see also:: `filters.get_window`\n\n    center : boolean\n        - If ``True``, the signal ``y`` is padded so that frame\n          ``S[:, t]`` is centered at ``y[t * hop_length]``.\n        - If ``False``, then ``S[:, t]`` begins at ``y[t * hop_length]``.\n\n    dtype : numeric type\n        Complex numeric type for ``S``. Default is inferred to match\n        the precision of the input signal.\n\n    pad_mode : string\n        If ``center=True``, the padding mode to use at the edges of the signal.\n        By default, STFT uses zero padding.\n\n    Returns\n    -------\n    times : np.ndarray [shape=(..., 1 + n_fft/2, t), dtype=real]\n        Reassigned times:\n        ``times[f, t]`` is the time for bin ``f``, frame ``t``.\n    S : np.ndarray [shape=(..., 1 + n_fft/2, t), dtype=complex]\n        Short-time Fourier transform\n\n    Warns\n    -----\n    RuntimeWarning\n        Time estimates with zero support will produce a divide-by-zero warning\n        and will be returned as `np.nan`.\n\n    See Also\n    --------\n    stft : Short-time Fourier Transform\n    reassigned_spectrogram : Time-frequency reassigned spectrogram\n\n    Examples\n    --------\n    >>> y, sr = librosa.load(librosa.ex(\'trumpet\'))\n    >>> times, S = librosa.core.spectrum.__reassign_times(y, sr=sr)\n    >>> times\n    array([[ 2.268e-05,  1.144e-02, ...,  5.332e+00,  5.333e+00],\n           [ 2.268e-05,  1.451e-02, ...,  5.334e+00,  5.333e+00],\n           ...,\n           [ 2.268e-05, -6.177e-04, ...,  5.368e+00,  5.327e+00],\n           [ 2.268e-05,  1.420e-03, ...,  5.307e+00,  5.328e+00]])\n    '
    if win_length is None:
        win_length = n_fft
    window = get_window(window, win_length, fftbins=True)
    window = util.pad_center(window, size=n_fft)
    if hop_length is None:
        hop_length = int(win_length // 4)
    if S is None:
        if dtype is None:
            dtype = util.dtype_r2c(y.dtype)
        S_h = stft(y=y, n_fft=n_fft, hop_length=hop_length, window=window, center=center, dtype=dtype, pad_mode=pad_mode)
    else:
        if dtype is None:
            dtype = S.dtype
        S_h = S
    half_width = n_fft // 2
    window_times: np.ndarray
    if n_fft % 2:
        window_times = np.arange(-half_width, half_width + 1)
    else:
        window_times = np.arange(0.5 - half_width, half_width)
    window_time_weighted = window * window_times
    S_th = stft(y=y, n_fft=n_fft, hop_length=hop_length, window=window_time_weighted, center=center, dtype=dtype, pad_mode=pad_mode)
    correction = np.real(S_th / S_h)
    if center:
        pad_length = None
    else:
        pad_length = n_fft
    times = convert.frames_to_time(np.arange(S_h.shape[-1]), sr=sr, hop_length=hop_length, n_fft=pad_length)
    times = util.expand_to(times, ndim=correction.ndim, axes=-1) + correction / sr
    return (times, S_h)

def reassigned_spectrogram(y: np.ndarray, *, sr: float=22050, S: Optional[np.ndarray]=None, n_fft: int=2048, hop_length: Optional[int]=None, win_length: Optional[int]=None, window: _WindowSpec='hann', center: bool=True, reassign_frequencies: bool=True, reassign_times: bool=True, ref_power: Union[float, Callable]=1e-06, fill_nan: bool=False, clip: bool=True, dtype: Optional[DTypeLike]=None, pad_mode: _PadModeSTFT='constant') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    'Time-frequency reassigned spectrogram.\n\n    The reassignment vectors are calculated using equations 5.20 and 5.23 in\n    [#]_::\n\n        t_reassigned = t + np.real(S_th/S_h)\n        omega_reassigned = omega - np.imag(S_dh/S_h)\n\n    where ``S_h`` is the complex STFT calculated using the original window,\n    ``S_dh`` is the complex STFT calculated using the derivative of the original\n    window, and ``S_th`` is the complex STFT calculated using the original window\n    multiplied by the time offset from the window center. See [#]_ for\n    additional algorithms, and [#]_ and [#]_ for history and discussion of the\n    method.\n\n    .. [#] Flandrin, P., Auger, F., & Chassande-Mottin, E. (2002).\n        Time-Frequency reassignment: From principles to algorithms. In\n        Applications in Time-Frequency Signal Processing (Vol. 10, pp.\n        179-204). CRC Press.\n\n    .. [#] Fulop, S. A., & Fitz, K. (2006). Algorithms for computing the\n        time-corrected instantaneous frequency (reassigned) spectrogram, with\n        applications. The Journal of the Acoustical Society of America, 119(1),\n        360. doi:10.1121/1.2133000\n\n    .. [#] Auger, F., Flandrin, P., Lin, Y.-T., McLaughlin, S., Meignen, S.,\n        Oberlin, T., & Wu, H.-T. (2013). Time-Frequency Reassignment and\n        Synchrosqueezing: An Overview. IEEE Signal Processing Magazine, 30(6),\n        32-41. doi:10.1109/MSP.2013.2265316\n\n    .. [#] Hainsworth, S., Macleod, M. (2003). Time-frequency reassignment: a\n        review and analysis. Tech. Rep. CUED/FINFENG/TR.459, Cambridge\n        University Engineering Department\n\n    Parameters\n    ----------\n    y : np.ndarray [shape=(..., n)], real-valued\n        audio time series. Multi-channel is supported.\n\n    sr : number > 0 [scalar]\n        sampling rate of ``y``\n\n    S : np.ndarray [shape=(..., d, t)] or None\n        (optional) complex STFT calculated using the other arguments provided\n        to ``reassigned_spectrogram``\n\n    n_fft : int > 0 [scalar]\n        FFT window size. Defaults to 2048.\n\n    hop_length : int > 0 [scalar]\n        hop length, number samples between subsequent frames.\n        If not supplied, defaults to ``win_length // 4``.\n\n    win_length : int > 0, <= n_fft\n        Window length. Defaults to ``n_fft``.\n        See `stft` for details.\n\n    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]\n        - a window specification (string, tuple, number);\n          see `scipy.signal.get_window`\n        - a window function, such as `scipy.signal.windows.hann`\n        - a user-specified window vector of length ``n_fft``\n\n        See `stft` for details.\n\n        .. see also:: `filters.get_window`\n\n    center : boolean\n        - If ``True`` (default), the signal ``y`` is padded so that frame\n          ``S[:, t]`` is centered at ``y[t * hop_length]``. See `Notes` for\n          recommended usage in this function.\n        - If ``False``, then ``S[:, t]`` begins at ``y[t * hop_length]``.\n\n    reassign_frequencies : boolean\n        - If ``True`` (default), the returned frequencies will be instantaneous\n          frequency estimates.\n        - If ``False``, the returned frequencies will be a read-only view of the\n          STFT bin frequencies for all frames.\n\n    reassign_times : boolean\n        - If ``True`` (default), the returned times will be corrected\n          (reassigned) time estimates for each bin.\n        - If ``False``, the returned times will be a read-only view of the STFT\n          frame times for all bins.\n\n    ref_power : float >= 0 or callable\n        Minimum power threshold for estimating time-frequency reassignments.\n        Any bin with ``np.abs(S[f, t])**2 < ref_power`` will be returned as\n        `np.nan` in both frequency and time, unless ``fill_nan`` is ``True``. If 0\n        is provided, then only bins with zero power will be returned as\n        `np.nan` (unless ``fill_nan=True``).\n\n    fill_nan : boolean\n        - If ``False`` (default), the frequency and time reassignments for bins\n          below the power threshold provided in ``ref_power`` will be returned as\n          `np.nan`.\n        - If ``True``, the frequency and time reassignments for these bins will\n          be returned as the bin center frequencies and frame times.\n\n    clip : boolean\n        - If ``True`` (default), estimated frequencies outside the range\n          `[0, 0.5 * sr]` or times outside the range `[0, len(y) / sr]` will be\n          clipped to those ranges.\n        - If ``False``, estimated frequencies and times beyond the bounds of the\n          spectrogram may be returned.\n\n    dtype : numeric type\n        Complex numeric type for STFT calculation. Default is inferred to match\n        the precision of the input signal.\n\n    pad_mode : string\n        If ``center=True``, the padding mode to use at the edges of the signal.\n        By default, STFT uses zero padding.\n\n    Returns\n    -------\n    freqs, times, mags : np.ndarray [shape=(..., 1 + n_fft/2, t), dtype=real]\n        Instantaneous frequencies:\n            ``freqs[..., f, t]`` is the frequency for bin ``f``, frame ``t``.\n            If ``reassign_frequencies=False``, this will instead be a read-only array\n            of the same shape containing the bin center frequencies for all frames.\n\n        Reassigned times:\n            ``times[..., f, t]`` is the time for bin ``f``, frame ``t``.\n            If ``reassign_times=False``, this will instead be a read-only array of\n            the same shape containing the frame times for all bins.\n\n        Magnitudes from short-time Fourier transform:\n            ``mags[..., f, t]`` is the magnitude for bin ``f``, frame ``t``.\n\n    Warns\n    -----\n    RuntimeWarning\n        Frequency or time estimates with zero support will produce a\n        divide-by-zero warning, and will be returned as `np.nan` unless\n        ``fill_nan=True``.\n\n    See Also\n    --------\n    stft : Short-time Fourier Transform\n\n    Notes\n    -----\n    It is recommended to use ``center=False`` with this function rather than the\n    librosa default ``True``. Unlike ``stft``, reassigned times are not aligned to\n    the left or center of each frame, so padding the signal does not affect the\n    meaning of the reassigned times. However, reassignment assumes that the\n    energy in each FFT bin is associated with exactly one signal component and\n    impulse event.\n\n    If ``reassign_times`` is ``False``, the frame times that are returned will be\n    aligned to the left or center of the frame, depending on the value of\n    ``center``. In this case, if ``center`` is ``True``, then ``pad_mode="wrap"`` is\n    recommended for valid estimation of the instantaneous frequencies in the\n    boundary frames.\n\n    Examples\n    --------\n    >>> import matplotlib.pyplot as plt\n    >>> amin = 1e-10\n    >>> n_fft = 64\n    >>> sr = 4000\n    >>> y = 1e-3 * librosa.clicks(times=[0.3], sr=sr, click_duration=1.0,\n    ...                           click_freq=1200.0, length=8000) +\\\n    ...     1e-3 * librosa.clicks(times=[1.5], sr=sr, click_duration=0.5,\n    ...                           click_freq=400.0, length=8000) +\\\n    ...     1e-3 * librosa.chirp(fmin=200, fmax=1600, sr=sr, duration=2.0) +\\\n    ...     1e-6 * np.random.randn(2*sr)\n    >>> freqs, times, mags = librosa.reassigned_spectrogram(y=y, sr=sr,\n    ...                                                     n_fft=n_fft)\n    >>> mags_db = librosa.amplitude_to_db(mags, ref=np.max)\n\n    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)\n    >>> img = librosa.display.specshow(mags_db, x_axis="s", y_axis="linear", sr=sr,\n    ...                          hop_length=n_fft//4, ax=ax[0])\n    >>> ax[0].set(title="Spectrogram", xlabel=None)\n    >>> ax[0].label_outer()\n    >>> ax[1].scatter(times, freqs, c=mags_db, cmap="magma", alpha=0.1, s=5)\n    >>> ax[1].set_title("Reassigned spectrogram")\n    >>> fig.colorbar(img, ax=ax, format="%+2.f dB")\n    '
    if not callable(ref_power) and ref_power < 0:
        raise ParameterError('ref_power must be non-negative or callable.')
    if not reassign_frequencies and (not reassign_times):
        raise ParameterError('reassign_frequencies or reassign_times must be True.')
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = int(win_length // 4)
    if reassign_frequencies:
        (freqs, S) = __reassign_frequencies(y=y, sr=sr, S=S, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, dtype=dtype, pad_mode=pad_mode)
    if reassign_times:
        (times, S) = __reassign_times(y=y, sr=sr, S=S, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, dtype=dtype, pad_mode=pad_mode)
    assert S is not None
    mags: np.ndarray = np.abs(S)
    if fill_nan or not reassign_frequencies or (not reassign_times):
        if center:
            pad_length = None
        else:
            pad_length = n_fft
        bin_freqs = convert.fft_frequencies(sr=sr, n_fft=n_fft)
        frame_times = convert.frames_to_time(frames=np.arange(S.shape[-1]), sr=sr, hop_length=hop_length, n_fft=pad_length)
    if callable(ref_power):
        ref_p = ref_power(mags ** 2)
    else:
        ref_p = ref_power
    mags_low = np.less(mags, ref_p ** 0.5, where=~np.isnan(mags))
    if reassign_frequencies:
        if ref_p > 0:
            freqs[mags_low] = np.nan
        if fill_nan:
            freqs = np.where(np.isnan(freqs), bin_freqs[:, np.newaxis], freqs)
        if clip:
            np.clip(freqs, 0, sr / 2.0, out=freqs)
    else:
        freqs = np.broadcast_to(bin_freqs[:, np.newaxis], S.shape)
    if reassign_times:
        if ref_p > 0:
            times[mags_low] = np.nan
        if fill_nan:
            times = np.where(np.isnan(times), frame_times[np.newaxis, :], times)
        if clip:
            np.clip(times, 0, y.shape[-1] / float(sr), out=times)
    else:
        times = np.broadcast_to(frame_times[np.newaxis, :], S.shape)
    return (freqs, times, mags)

def magphase(D: np.ndarray, *, power: float=1) -> Tuple[np.ndarray, np.ndarray]:
    if False:
        print('Hello World!')
    "Separate a complex-valued spectrogram D into its magnitude (S)\n    and phase (P) components, so that ``D = S * P``.\n\n    Parameters\n    ----------\n    D : np.ndarray [shape=(..., d, t), dtype=complex]\n        complex-valued spectrogram\n    power : float > 0\n        Exponent for the magnitude spectrogram,\n        e.g., 1 for energy, 2 for power, etc.\n\n    Returns\n    -------\n    D_mag : np.ndarray [shape=(..., d, t), dtype=real]\n        magnitude of ``D``, raised to ``power``\n    D_phase : np.ndarray [shape=(..., d, t), dtype=complex]\n        ``exp(1.j * phi)`` where ``phi`` is the phase of ``D``\n\n    Examples\n    --------\n    >>> y, sr = librosa.load(librosa.ex('trumpet'))\n    >>> D = librosa.stft(y)\n    >>> magnitude, phase = librosa.magphase(D)\n    >>> magnitude\n    array([[5.395e-03, 3.332e-03, ..., 9.862e-07, 1.201e-05],\n           [3.244e-03, 2.690e-03, ..., 9.536e-07, 1.201e-05],\n           ...,\n           [7.523e-05, 3.722e-05, ..., 1.188e-04, 1.031e-03],\n           [7.640e-05, 3.944e-05, ..., 5.180e-04, 1.346e-03]],\n          dtype=float32)\n    >>> phase\n    array([[ 1.   +0.000e+00j,  1.   +0.000e+00j, ...,\n            -1.   -8.742e-08j, -1.   -8.742e-08j],\n           [-1.   -8.742e-08j, -0.775-6.317e-01j, ...,\n            -0.885-4.648e-01j,  0.472-8.815e-01j],\n           ...,\n           [ 1.   -4.342e-12j,  0.028-9.996e-01j, ...,\n            -0.222-9.751e-01j, -0.75 -6.610e-01j],\n           [-1.   -8.742e-08j, -1.   -8.742e-08j, ...,\n             1.   +0.000e+00j,  1.   +0.000e+00j]], dtype=complex64)\n\n    Or get the phase angle (in radians)\n\n    >>> np.angle(phase)\n    array([[ 0.000e+00,  0.000e+00, ..., -3.142e+00, -3.142e+00],\n           [-3.142e+00, -2.458e+00, ..., -2.658e+00, -1.079e+00],\n           ...,\n           [-4.342e-12, -1.543e+00, ..., -1.794e+00, -2.419e+00],\n           [-3.142e+00, -3.142e+00, ...,  0.000e+00,  0.000e+00]],\n          dtype=float32)\n    "
    mag = np.abs(D)
    zeros_to_ones = mag == 0
    mag_nonzero = mag + zeros_to_ones
    phase = np.empty_like(D, dtype=util.dtype_r2c(D.dtype))
    phase.real = D.real / mag_nonzero + zeros_to_ones
    phase.imag = D.imag / mag_nonzero
    mag **= power
    return (mag, phase)

def phase_vocoder(D: np.ndarray, *, rate: float, hop_length: Optional[int]=None, n_fft: Optional[int]=None) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Phase vocoder.  Given an STFT matrix D, speed up by a factor of ``rate``\n\n    Based on the implementation provided by [#]_.\n\n    This is a simplified implementation, intended primarily for\n    reference and pedagogical purposes.  It makes no attempt to\n    handle transients, and is likely to produce many audible\n    artifacts.  For a higher quality implementation, we recommend\n    the RubberBand library [#]_ and its Python wrapper `pyrubberband`.\n\n    .. [#] Ellis, D. P. W. "A phase vocoder in Matlab."\n        Columbia University, 2002.\n        https://www.ee.columbia.edu/~dpwe/resources/matlab/pvoc/\n\n    .. [#] https://breakfastquay.com/rubberband/\n\n    Examples\n    --------\n    >>> # Play at double speed\n    >>> y, sr   = librosa.load(librosa.ex(\'trumpet\'))\n    >>> D       = librosa.stft(y, n_fft=2048, hop_length=512)\n    >>> D_fast  = librosa.phase_vocoder(D, rate=2.0, hop_length=512)\n    >>> y_fast  = librosa.istft(D_fast, hop_length=512)\n\n    >>> # Or play at 1/3 speed\n    >>> y, sr   = librosa.load(librosa.ex(\'trumpet\'))\n    >>> D       = librosa.stft(y, n_fft=2048, hop_length=512)\n    >>> D_slow  = librosa.phase_vocoder(D, rate=1./3, hop_length=512)\n    >>> y_slow  = librosa.istft(D_slow, hop_length=512)\n\n    Parameters\n    ----------\n    D : np.ndarray [shape=(..., d, t), dtype=complex]\n        STFT matrix\n\n    rate : float > 0 [scalar]\n        Speed-up factor: ``rate > 1`` is faster, ``rate < 1`` is slower.\n\n    hop_length : int > 0 [scalar] or None\n        The number of samples between successive columns of ``D``.\n\n        If None, defaults to ``n_fft//4 = (D.shape[0]-1)//2``\n\n    n_fft : int > 0 or None\n        The number of samples per frame in D.\n        By default (None), this will be inferred from the shape of D.\n        However, if D was constructed using an odd-length window, the correct\n        frame length can be specified here.\n\n    Returns\n    -------\n    D_stretched : np.ndarray [shape=(..., d, t / rate), dtype=complex]\n        time-stretched STFT\n\n    See Also\n    --------\n    pyrubberband\n    '
    if n_fft is None:
        n_fft = 2 * (D.shape[-2] - 1)
    if hop_length is None:
        hop_length = int(n_fft // 4)
    time_steps = np.arange(0, D.shape[-1], rate, dtype=np.float64)
    shape = list(D.shape)
    shape[-1] = len(time_steps)
    d_stretch = np.zeros_like(D, shape=shape)
    phi_advance = np.linspace(0, np.pi * hop_length, D.shape[-2])
    phase_acc = np.angle(D[..., 0])
    padding = [(0, 0) for _ in D.shape]
    padding[-1] = (0, 2)
    D = np.pad(D, padding, mode='constant')
    for (t, step) in enumerate(time_steps):
        columns = D[..., int(step):int(step + 2)]
        alpha = np.mod(step, 1.0)
        mag = (1.0 - alpha) * np.abs(columns[..., 0]) + alpha * np.abs(columns[..., 1])
        d_stretch[..., t] = util.phasor(phase_acc, mag=mag)
        dphase = np.angle(columns[..., 1]) - np.angle(columns[..., 0]) - phi_advance
        dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))
        phase_acc += phi_advance + dphase
    return d_stretch

@cache(level=20)
def iirt(y: np.ndarray, *, sr: float=22050, win_length: int=2048, hop_length: Optional[int]=None, center: bool=True, tuning: float=0.0, pad_mode: _PadMode='constant', flayout: str='sos', res_type: str='soxr_hq', **kwargs: Any) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    'Time-frequency representation using IIR filters\n\n    This function will return a time-frequency representation\n    using a multirate filter bank consisting of IIR filters. [#]_\n\n    First, ``y`` is resampled as needed according to the provided ``sample_rates``.\n\n    Then, a filterbank with with ``n`` band-pass filters is designed.\n\n    The resampled input signals are processed by the filterbank as a whole.\n    (`scipy.signal.filtfilt` resp. `sosfiltfilt` is used to make the phase linear.)\n    The output of the filterbank is cut into frames.\n    For each band, the short-time mean-square power (STMSP) is calculated by\n    summing ``win_length`` subsequent filtered time samples.\n\n    When called with the default set of parameters, it will generate the TF-representation\n    (pitch filterbank):\n\n        * 85 filters with MIDI pitches [24, 108] as ``center_freqs``.\n        * each filter having a bandwidth of one semitone.\n\n    .. [#] Müller, Meinard.\n           "Information Retrieval for Music and Motion."\n           Springer Verlag. 2007.\n\n    Parameters\n    ----------\n    y : np.ndarray [shape=(..., n)]\n        audio time series. Multi-channel is supported.\n    sr : number > 0 [scalar]\n        sampling rate of ``y``\n    win_length : int > 0, <= n_fft\n        Window length.\n    hop_length : int > 0 [scalar]\n        Hop length, number samples between subsequent frames.\n        If not supplied, defaults to ``win_length // 4``.\n    center : boolean\n        - If ``True``, the signal ``y`` is padded so that frame\n          ``D[..., :, t]`` is centered at ``y[t * hop_length]``.\n        - If ``False``, then `D[..., :, t]`` begins at ``y[t * hop_length]``\n    tuning : float [scalar]\n        Tuning deviation from A440 in fractions of a bin.\n    pad_mode : string\n        If ``center=True``, the padding mode to use at the edges of the signal.\n        By default, this function uses zero padding.\n    flayout : string\n        - If `sos` (default), a series of second-order filters is used for filtering with `scipy.signal.sosfiltfilt`.\n          Minimizes numerical precision errors for high-order filters, but is slower.\n        - If `ba`, the standard difference equation is used for filtering with `scipy.signal.filtfilt`.\n          Can be unstable for high-order filters.\n    res_type : string\n        The resampling mode.  See `librosa.resample` for details.\n    **kwargs : additional keyword arguments\n        Additional arguments for `librosa.filters.semitone_filterbank`\n        (e.g., could be used to provide another set of ``center_freqs`` and ``sample_rates``).\n\n    Returns\n    -------\n    bands_power : np.ndarray [shape=(..., n, t), dtype=dtype]\n        Short-time mean-square power for the input signal.\n\n    Raises\n    ------\n    ParameterError\n        If ``flayout`` is not None, `ba`, or `sos`.\n\n    See Also\n    --------\n    librosa.filters.semitone_filterbank\n    librosa.filters.mr_frequencies\n    librosa.cqt\n    scipy.signal.filtfilt\n    scipy.signal.sosfiltfilt\n\n    Examples\n    --------\n    >>> import matplotlib.pyplot as plt\n    >>> y, sr = librosa.load(librosa.ex(\'trumpet\'), duration=3)\n    >>> D = np.abs(librosa.iirt(y))\n    >>> C = np.abs(librosa.cqt(y=y, sr=sr))\n    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)\n    >>> img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),\n    ...                                y_axis=\'cqt_hz\', x_axis=\'time\', ax=ax[0])\n    >>> ax[0].set(title=\'Constant-Q transform\')\n    >>> ax[0].label_outer()\n    >>> img = librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),\n    ...                                y_axis=\'cqt_hz\', x_axis=\'time\', ax=ax[1])\n    >>> ax[1].set_title(\'Semitone spectrogram (iirt)\')\n    >>> fig.colorbar(img, ax=ax, format="%+2.0f dB")\n    '
    if flayout not in ('ba', 'sos'):
        raise ParameterError(f'Unsupported flayout={flayout}')
    util.valid_audio(y, mono=False)
    if hop_length is None:
        hop_length = win_length // 4
    if center:
        padding = [(0, 0) for _ in y.shape]
        padding[-1] = (win_length // 2, win_length // 2)
        y = np.pad(y, padding, mode=pad_mode)
    (filterbank_ct, sample_rates) = semitone_filterbank(tuning=tuning, flayout=flayout, **kwargs)
    y_resampled = []
    y_srs = np.unique(sample_rates)
    for cur_sr in y_srs:
        y_resampled.append(resample(y, orig_sr=sr, target_sr=cur_sr, res_type=res_type))
    n_frames = int(1 + (y.shape[-1] - win_length) // hop_length)
    shape = list(y.shape)
    shape[-1] = n_frames
    shape.insert(-1, len(filterbank_ct))
    bands_power = np.empty_like(y, shape=shape)
    slices: List[Union[int, slice]] = [slice(None) for _ in bands_power.shape]
    for (i, (cur_sr, cur_filter)) in enumerate(zip(sample_rates, filterbank_ct)):
        slices[-2] = i
        cur_sr_idx = np.flatnonzero(y_srs == cur_sr)[0]
        if flayout == 'ba':
            cur_filter_output = scipy.signal.filtfilt(cur_filter[0], cur_filter[1], y_resampled[cur_sr_idx], axis=-1)
        elif flayout == 'sos':
            cur_filter_output = scipy.signal.sosfiltfilt(cur_filter, y_resampled[cur_sr_idx], axis=-1)
        factor = sr / cur_sr
        hop_length_STMSP = hop_length / factor
        win_length_STMSP_round = int(round(win_length / factor))
        start_idx = np.arange(0, cur_filter_output.shape[-1] - win_length_STMSP_round, hop_length_STMSP)
        if len(start_idx) < n_frames:
            min_length = int(np.ceil(n_frames * hop_length_STMSP)) + win_length_STMSP_round
            cur_filter_output = util.fix_length(cur_filter_output, size=min_length)
            start_idx = np.arange(0, cur_filter_output.shape[-1] - win_length_STMSP_round, hop_length_STMSP)
        start_idx = np.round(start_idx).astype(int)[:n_frames]
        idx = np.add.outer(start_idx, np.arange(win_length_STMSP_round))
        bands_power[tuple(slices)] = factor * np.sum(cur_filter_output[..., idx] ** 2, axis=-1)
    return bands_power

@cache(level=30)
def power_to_db(S: np.ndarray, *, ref: Union[float, Callable]=1.0, amin: float=1e-10, top_db: Optional[float]=80.0) -> np.ndarray:
    if False:
        return 10
    'Convert a power spectrogram (amplitude squared) to decibel (dB) units\n\n    This computes the scaling ``10 * log10(S / ref)`` in a numerically\n    stable way.\n\n    Parameters\n    ----------\n    S : np.ndarray\n        input power\n\n    ref : scalar or callable\n        If scalar, the amplitude ``abs(S)`` is scaled relative to ``ref``::\n\n            10 * log10(S / ref)\n\n        Zeros in the output correspond to positions where ``S == ref``.\n\n        If callable, the reference value is computed as ``ref(S)``.\n\n    amin : float > 0 [scalar]\n        minimum threshold for ``abs(S)`` and ``ref``\n\n    top_db : float >= 0 [scalar]\n        threshold the output at ``top_db`` below the peak:\n        ``max(10 * log10(S/ref)) - top_db``\n\n    Returns\n    -------\n    S_db : np.ndarray\n        ``S_db ~= 10 * log10(S) - 10 * log10(ref)``\n\n    See Also\n    --------\n    perceptual_weighting\n    db_to_power\n    amplitude_to_db\n    db_to_amplitude\n\n    Notes\n    -----\n    This function caches at level 30.\n\n    Examples\n    --------\n    Get a power spectrogram from a waveform ``y``\n\n    >>> y, sr = librosa.load(librosa.ex(\'trumpet\'))\n    >>> S = np.abs(librosa.stft(y))\n    >>> librosa.power_to_db(S**2)\n    array([[-41.809, -41.809, ..., -41.809, -41.809],\n           [-41.809, -41.809, ..., -41.809, -41.809],\n           ...,\n           [-41.809, -41.809, ..., -41.809, -41.809],\n           [-41.809, -41.809, ..., -41.809, -41.809]], dtype=float32)\n\n    Compute dB relative to peak power\n\n    >>> librosa.power_to_db(S**2, ref=np.max)\n    array([[-80., -80., ..., -80., -80.],\n           [-80., -80., ..., -80., -80.],\n           ...,\n           [-80., -80., ..., -80., -80.],\n           [-80., -80., ..., -80., -80.]], dtype=float32)\n\n    Or compare to median power\n\n    >>> librosa.power_to_db(S**2, ref=np.median)\n    array([[16.578, 16.578, ..., 16.578, 16.578],\n           [16.578, 16.578, ..., 16.578, 16.578],\n           ...,\n           [16.578, 16.578, ..., 16.578, 16.578],\n           [16.578, 16.578, ..., 16.578, 16.578]], dtype=float32)\n\n    And plot the results\n\n    >>> import matplotlib.pyplot as plt\n    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)\n    >>> imgpow = librosa.display.specshow(S**2, sr=sr, y_axis=\'log\', x_axis=\'time\',\n    ...                                   ax=ax[0])\n    >>> ax[0].set(title=\'Power spectrogram\')\n    >>> ax[0].label_outer()\n    >>> imgdb = librosa.display.specshow(librosa.power_to_db(S**2, ref=np.max),\n    ...                                  sr=sr, y_axis=\'log\', x_axis=\'time\', ax=ax[1])\n    >>> ax[1].set(title=\'Log-Power spectrogram\')\n    >>> fig.colorbar(imgpow, ax=ax[0])\n    >>> fig.colorbar(imgdb, ax=ax[1], format="%+2.0f dB")\n    '
    S = np.asarray(S)
    if amin <= 0:
        raise ParameterError('amin must be strictly positive')
    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn('power_to_db was called on complex input so phase information will be discarded. To suppress this warning, call power_to_db(np.abs(D)**2) instead.', stacklevel=2)
        magnitude = np.abs(S)
    else:
        magnitude = S
    if callable(ref):
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)
    log_spec: np.ndarray = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))
    if top_db is not None:
        if top_db < 0:
            raise ParameterError('top_db must be non-negative')
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    return log_spec

@cache(level=30)
def db_to_power(S_db: np.ndarray, *, ref: float=1.0) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    'Convert a dB-scale spectrogram to a power spectrogram.\n\n    This effectively inverts ``power_to_db``::\n\n        db_to_power(S_db) ~= ref * 10.0**(S_db / 10)\n\n    Parameters\n    ----------\n    S_db : np.ndarray\n        dB-scaled spectrogram\n    ref : number > 0\n        Reference power: output will be scaled by this value\n\n    Returns\n    -------\n    S : np.ndarray\n        Power spectrogram\n\n    Notes\n    -----\n    This function caches at level 30.\n    '
    return ref * np.power(10.0, 0.1 * S_db)

@cache(level=30)
def amplitude_to_db(S: np.ndarray, *, ref: Union[float, Callable]=1.0, amin: float=1e-05, top_db: Optional[float]=80.0) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    'Convert an amplitude spectrogram to dB-scaled spectrogram.\n\n    This is equivalent to ``power_to_db(S**2, ref=ref**2, amin=amin**2, top_db=top_db)``,\n    but is provided for convenience.\n\n    Parameters\n    ----------\n    S : np.ndarray\n        input amplitude\n\n    ref : scalar or callable\n        If scalar, the amplitude ``abs(S)`` is scaled relative to ``ref``:\n        ``20 * log10(S / ref)``.\n        Zeros in the output correspond to positions where ``S == ref``.\n\n        If callable, the reference value is computed as ``ref(S)``.\n\n    amin : float > 0 [scalar]\n        minimum threshold for ``S`` and ``ref``\n\n    top_db : float >= 0 [scalar]\n        threshold the output at ``top_db`` below the peak:\n        ``max(20 * log10(S/ref)) - top_db``\n\n    Returns\n    -------\n    S_db : np.ndarray\n        ``S`` measured in dB\n\n    See Also\n    --------\n    power_to_db, db_to_amplitude\n\n    Notes\n    -----\n    This function caches at level 30.\n    '
    S = np.asarray(S)
    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn('amplitude_to_db was called on complex input so phase information will be discarded. To suppress this warning, call amplitude_to_db(np.abs(S)) instead.', stacklevel=2)
    magnitude = np.abs(S)
    if callable(ref):
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)
    power = np.square(magnitude, out=magnitude)
    return power_to_db(power, ref=ref_value ** 2, amin=amin ** 2, top_db=top_db)

@cache(level=30)
def db_to_amplitude(S_db: np.ndarray, *, ref: float=1.0) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    'Convert a dB-scaled spectrogram to an amplitude spectrogram.\n\n    This effectively inverts `amplitude_to_db`::\n\n        db_to_amplitude(S_db) ~= 10.0**(0.5 * S_db/10 + log10(ref))\n\n    Parameters\n    ----------\n    S_db : np.ndarray\n        dB-scaled spectrogram\n    ref : number > 0\n        Optional reference power.\n\n    Returns\n    -------\n    S : np.ndarray\n        Linear magnitude spectrogram\n\n    Notes\n    -----\n    This function caches at level 30.\n    '
    return db_to_power(S_db, ref=ref ** 2) ** 0.5

@cache(level=30)
def perceptual_weighting(S: np.ndarray, frequencies: np.ndarray, *, kind: str='A', **kwargs: Any) -> np.ndarray:
    if False:
        print('Hello World!')
    'Perceptual weighting of a power spectrogram::\n\n        S_p[..., f, :] = frequency_weighting(f, \'A\') + 10*log(S[..., f, :] / ref)\n\n    Parameters\n    ----------\n    S : np.ndarray [shape=(..., d, t)]\n        Power spectrogram\n    frequencies : np.ndarray [shape=(d,)]\n        Center frequency for each row of` `S``\n    kind : str\n        The frequency weighting curve to use.\n        e.g. `\'A\'`, `\'B\'`, `\'C\'`, `\'D\'`, `None or \'Z\'`\n    **kwargs : additional keyword arguments\n        Additional keyword arguments to `power_to_db`.\n\n    Returns\n    -------\n    S_p : np.ndarray [shape=(..., d, t)]\n        perceptually weighted version of ``S``\n\n    See Also\n    --------\n    power_to_db\n\n    Notes\n    -----\n    This function caches at level 30.\n\n    Examples\n    --------\n    Re-weight a CQT power spectrum, using peak power as reference\n\n    >>> y, sr = librosa.load(librosa.ex(\'trumpet\'))\n    >>> C = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz(\'A1\')))\n    >>> freqs = librosa.cqt_frequencies(C.shape[0],\n    ...                                 fmin=librosa.note_to_hz(\'A1\'))\n    >>> perceptual_CQT = librosa.perceptual_weighting(C**2,\n    ...                                               freqs,\n    ...                                               ref=np.max)\n    >>> perceptual_CQT\n    array([[ -96.528,  -97.101, ..., -108.561, -108.561],\n           [ -95.88 ,  -96.479, ..., -107.551, -107.551],\n           ...,\n           [ -65.142,  -53.256, ...,  -80.098,  -80.098],\n           [ -71.542,  -53.197, ...,  -80.311,  -80.311]])\n\n    >>> import matplotlib.pyplot as plt\n    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)\n    >>> img = librosa.display.specshow(librosa.amplitude_to_db(C,\n    ...                                                        ref=np.max),\n    ...                                fmin=librosa.note_to_hz(\'A1\'),\n    ...                                y_axis=\'cqt_hz\', x_axis=\'time\',\n    ...                                ax=ax[0])\n    >>> ax[0].set(title=\'Log CQT power\')\n    >>> ax[0].label_outer()\n    >>> imgp = librosa.display.specshow(perceptual_CQT, y_axis=\'cqt_hz\',\n    ...                                 fmin=librosa.note_to_hz(\'A1\'),\n    ...                                 x_axis=\'time\', ax=ax[1])\n    >>> ax[1].set(title=\'Perceptually weighted log CQT\')\n    >>> fig.colorbar(img, ax=ax[0], format="%+2.0f dB")\n    >>> fig.colorbar(imgp, ax=ax[1], format="%+2.0f dB")\n    '
    offset = convert.frequency_weighting(frequencies, kind=kind).reshape((-1, 1))
    result: np.ndarray = offset + power_to_db(S, **kwargs)
    return result

@cache(level=30)
def fmt(y: np.ndarray, *, t_min: float=0.5, n_fmt: Optional[int]=None, kind: str='cubic', beta: float=0.5, over_sample: float=1, axis: int=-1) -> np.ndarray:
    if False:
        return 10
    'Fast Mellin transform (FMT)\n\n    The Mellin of a signal `y` is performed by interpolating `y` on an exponential time\n    axis, applying a polynomial window, and then taking the discrete Fourier transform.\n\n    When the Mellin parameter (beta) is 1/2, it is also known as the scale transform. [#]_\n    The scale transform can be useful for audio analysis because its magnitude is invariant\n    to scaling of the domain (e.g., time stretching or compression).  This is analogous\n    to the magnitude of the Fourier transform being invariant to shifts in the input domain.\n\n    .. [#] De Sena, Antonio, and Davide Rocchesso.\n        "A fast Mellin and scale transform."\n        EURASIP Journal on Applied Signal Processing 2007.1 (2007): 75-75.\n\n    .. [#] Cohen, L.\n        "The scale representation."\n        IEEE Transactions on Signal Processing 41, no. 12 (1993): 3275-3292.\n\n    Parameters\n    ----------\n    y : np.ndarray, real-valued\n        The input signal(s).  Can be multidimensional.\n        The target axis must contain at least 3 samples.\n\n    t_min : float > 0\n        The minimum time spacing (in samples).\n        This value should generally be less than 1 to preserve as much information as\n        possible.\n\n    n_fmt : int > 2 or None\n        The number of scale transform bins to use.\n        If None, then ``n_bins = over_sample * ceil(n * log((n-1)/t_min))`` is taken,\n        where ``n = y.shape[axis]``\n\n    kind : str\n        The type of interpolation to use when re-sampling the input.\n        See `scipy.interpolate.interp1d` for possible values.\n\n        Note that the default is to use high-precision (cubic) interpolation.\n        This can be slow in practice; if speed is preferred over accuracy,\n        then consider using ``kind=\'linear\'``.\n\n    beta : float\n        The Mellin parameter.  ``beta=0.5`` provides the scale transform.\n\n    over_sample : float >= 1\n        Over-sampling factor for exponential resampling.\n\n    axis : int\n        The axis along which to transform ``y``\n\n    Returns\n    -------\n    x_scale : np.ndarray [dtype=complex]\n        The scale transform of ``y`` along the ``axis`` dimension.\n\n    Raises\n    ------\n    ParameterError\n        if ``n_fmt < 2`` or ``t_min <= 0``\n        or if ``y`` is not finite\n        or if ``y.shape[axis] < 3``.\n\n    Notes\n    -----\n    This function caches at level 30.\n\n    Examples\n    --------\n    >>> # Generate a signal and time-stretch it (with energy normalization)\n    >>> scale = 1.25\n    >>> freq = 3.0\n    >>> x1 = np.linspace(0, 1, num=1024, endpoint=False)\n    >>> x2 = np.linspace(0, 1, num=int(scale * len(x1)), endpoint=False)\n    >>> y1 = np.sin(2 * np.pi * freq * x1)\n    >>> y2 = np.sin(2 * np.pi * freq * x2) / np.sqrt(scale)\n    >>> # Verify that the two signals have the same energy\n    >>> np.sum(np.abs(y1)**2), np.sum(np.abs(y2)**2)\n        (255.99999999999997, 255.99999999999969)\n    >>> scale1 = librosa.fmt(y1, n_fmt=512)\n    >>> scale2 = librosa.fmt(y2, n_fmt=512)\n\n    >>> # And plot the results\n    >>> import matplotlib.pyplot as plt\n    >>> fig, ax = plt.subplots(nrows=2)\n    >>> ax[0].plot(y1, label=\'Original\')\n    >>> ax[0].plot(y2, linestyle=\'--\', label=\'Stretched\')\n    >>> ax[0].set(xlabel=\'time (samples)\', title=\'Input signals\')\n    >>> ax[0].legend()\n    >>> ax[1].semilogy(np.abs(scale1), label=\'Original\')\n    >>> ax[1].semilogy(np.abs(scale2), linestyle=\'--\', label=\'Stretched\')\n    >>> ax[1].set(xlabel=\'scale coefficients\', title=\'Scale transform magnitude\')\n    >>> ax[1].legend()\n\n    >>> # Plot the scale transform of an onset strength autocorrelation\n    >>> y, sr = librosa.load(librosa.ex(\'choice\'))\n    >>> odf = librosa.onset.onset_strength(y=y, sr=sr)\n    >>> # Auto-correlate with up to 10 seconds lag\n    >>> odf_ac = librosa.autocorrelate(odf, max_size=10 * sr // 512)\n    >>> # Normalize\n    >>> odf_ac = librosa.util.normalize(odf_ac, norm=np.inf)\n    >>> # Compute the scale transform\n    >>> odf_ac_scale = librosa.fmt(librosa.util.normalize(odf_ac), n_fmt=512)\n    >>> # Plot the results\n    >>> fig, ax = plt.subplots(nrows=3)\n    >>> ax[0].plot(odf, label=\'Onset strength\')\n    >>> ax[0].set(xlabel=\'Time (frames)\', title=\'Onset strength\')\n    >>> ax[1].plot(odf_ac, label=\'Onset autocorrelation\')\n    >>> ax[1].set(xlabel=\'Lag (frames)\', title=\'Onset autocorrelation\')\n    >>> ax[2].semilogy(np.abs(odf_ac_scale), label=\'Scale transform magnitude\')\n    >>> ax[2].set(xlabel=\'scale coefficients\')\n    '
    n = y.shape[axis]
    if n < 3:
        raise ParameterError(f'y.shape[{axis}]=={n} < 3')
    if t_min <= 0:
        raise ParameterError(f't_min={t_min} must be a positive number')
    if n_fmt is None:
        if over_sample < 1:
            raise ParameterError(f'over_sample={over_sample} must be >= 1')
        log_base = np.log(n - 1) - np.log(n - 2)
        n_fmt = int(np.ceil(over_sample * (np.log(n - 1) - np.log(t_min)) / log_base))
    elif n_fmt < 3:
        raise ParameterError(f'n_fmt=={n_fmt} < 3')
    else:
        log_base = (np.log(n_fmt - 1) - np.log(n_fmt - 2)) / over_sample
    if not np.all(np.isfinite(y)):
        raise ParameterError('y must be finite everywhere')
    base = np.exp(log_base)
    x = np.linspace(0, 1, num=n, endpoint=False)
    f_interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=axis)
    n_over = int(np.ceil(over_sample))
    x_exp = np.logspace((np.log(t_min) - np.log(n)) / log_base, 0, num=n_fmt + n_over, endpoint=False, base=base)[:-n_over]
    if x_exp[0] < t_min or x_exp[-1] > float(n - 1.0) / n:
        x_exp = np.clip(x_exp, float(t_min) / n, x[-1])
    if len(np.unique(x_exp)) != len(x_exp):
        raise ParameterError('Redundant sample positions in Mellin transform')
    y_res = f_interp(x_exp)
    shape = [1] * y_res.ndim
    shape[axis] = -1
    fft = get_fftlib()
    result: np.ndarray = fft.rfft(y_res * ((x_exp ** beta).reshape(shape) * np.sqrt(n) / n_fmt), axis=axis)
    return result

@overload
def pcen(S: np.ndarray, *, sr: float=..., hop_length: int=..., gain: float=..., bias: float=..., power: float=..., time_constant: float=..., eps: float=..., b: Optional[float]=..., max_size: int=..., ref: Optional[np.ndarray]=..., axis: int=..., max_axis: Optional[int]=..., zi: Optional[np.ndarray]=..., return_zf: Literal[False]=...) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    ...

@overload
def pcen(S: np.ndarray, *, sr: float=..., hop_length: int=..., gain: float=..., bias: float=..., power: float=..., time_constant: float=..., eps: float=..., b: Optional[float]=..., max_size: int=..., ref: Optional[np.ndarray]=..., axis: int=..., max_axis: Optional[int]=..., zi: Optional[np.ndarray]=..., return_zf: Literal[True]) -> Tuple[np.ndarray, np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def pcen(S: np.ndarray, *, sr: float=..., hop_length: int=..., gain: float=..., bias: float=..., power: float=..., time_constant: float=..., eps: float=..., b: Optional[float]=..., max_size: int=..., ref: Optional[np.ndarray]=..., axis: int=..., max_axis: Optional[int]=..., zi: Optional[np.ndarray]=..., return_zf: bool=...) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if False:
        while True:
            i = 10
    ...

@cache(level=30)
def pcen(S: np.ndarray, *, sr: float=22050, hop_length: int=512, gain: float=0.98, bias: float=2, power: float=0.5, time_constant: float=0.4, eps: float=1e-06, b: Optional[float]=None, max_size: int=1, ref: Optional[np.ndarray]=None, axis: int=-1, max_axis: Optional[int]=None, zi: Optional[np.ndarray]=None, return_zf: bool=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if False:
        return 10
    'Per-channel energy normalization (PCEN)\n\n    This function normalizes a time-frequency representation ``S`` by\n    performing automatic gain control, followed by nonlinear compression [#]_ ::\n\n        P[f, t] = (S / (eps + M[f, t])**gain + bias)**power - bias**power\n\n    IMPORTANT: the default values of eps, gain, bias, and power match the\n    original publication, in which ``S`` is a 40-band mel-frequency\n    spectrogram with 25 ms windowing, 10 ms frame shift, and raw audio values\n    in the interval [-2**31; 2**31-1[. If you use these default values, we\n    recommend to make sure that the raw audio is properly scaled to this\n    interval, and not to [-1, 1[ as is most often the case.\n\n    The matrix ``M`` is the result of applying a low-pass, temporal IIR filter\n    to ``S``::\n\n        M[f, t] = (1 - b) * M[f, t - 1] + b * S[f, t]\n\n    If ``b`` is not provided, it is calculated as::\n\n        b = (sqrt(1 + 4* T**2) - 1) / (2 * T**2)\n\n    where ``T = time_constant * sr / hop_length``. [#]_\n\n    This normalization is designed to suppress background noise and\n    emphasize foreground signals, and can be used as an alternative to\n    decibel scaling (`amplitude_to_db`).\n\n    This implementation also supports smoothing across frequency bins\n    by specifying ``max_size > 1``.  If this option is used, the filtered\n    spectrogram ``M`` is computed as::\n\n        M[f, t] = (1 - b) * M[f, t - 1] + b * R[f, t]\n\n    where ``R`` has been max-filtered along the frequency axis, similar to\n    the SuperFlux algorithm implemented in `onset.onset_strength`::\n\n        R[f, t] = max(S[f - max_size//2: f + max_size//2, t])\n\n    This can be used to perform automatic gain control on signals that cross\n    or span multiple frequency bans, which may be desirable for spectrograms\n    with high frequency resolution.\n\n    .. [#] Wang, Y., Getreuer, P., Hughes, T., Lyon, R. F., & Saurous, R. A.\n       (2017, March). Trainable frontend for robust and far-field keyword spotting.\n       In Acoustics, Speech and Signal Processing (ICASSP), 2017\n       IEEE International Conference on (pp. 5670-5674). IEEE.\n\n    .. [#] Lostanlen, V., Salamon, J., McFee, B., Cartwright, M., Farnsworth, A.,\n       Kelling, S., and Bello, J. P. Per-Channel Energy Normalization: Why and How.\n       IEEE Signal Processing Letters, 26(1), 39-43.\n\n    Parameters\n    ----------\n    S : np.ndarray (non-negative)\n        The input (magnitude) spectrogram\n\n    sr : number > 0 [scalar]\n        The audio sampling rate\n\n    hop_length : int > 0 [scalar]\n        The hop length of ``S``, expressed in samples\n\n    gain : number >= 0 [scalar]\n        The gain factor.  Typical values should be slightly less than 1.\n\n    bias : number >= 0 [scalar]\n        The bias point of the nonlinear compression (default: 2)\n\n    power : number >= 0 [scalar]\n        The compression exponent.  Typical values should be between 0 and 0.5.\n        Smaller values of ``power`` result in stronger compression.\n        At the limit ``power=0``, polynomial compression becomes logarithmic.\n\n    time_constant : number > 0 [scalar]\n        The time constant for IIR filtering, measured in seconds.\n\n    eps : number > 0 [scalar]\n        A small constant used to ensure numerical stability of the filter.\n\n    b : number in [0, 1]  [scalar]\n        The filter coefficient for the low-pass filter.\n        If not provided, it will be inferred from ``time_constant``.\n\n    max_size : int > 0 [scalar]\n        The width of the max filter applied to the frequency axis.\n        If left as `1`, no filtering is performed.\n\n    ref : None or np.ndarray (shape=S.shape)\n        An optional pre-computed reference spectrum (``R`` in the above).\n        If not provided it will be computed from ``S``.\n\n    axis : int [scalar]\n        The (time) axis of the input spectrogram.\n\n    max_axis : None or int [scalar]\n        The frequency axis of the input spectrogram.\n        If `None`, and ``S`` is two-dimensional, it will be inferred\n        as the opposite from ``axis``.\n        If ``S`` is not two-dimensional, and ``max_size > 1``, an error\n        will be raised.\n\n    zi : np.ndarray\n        The initial filter delay values.\n\n        This may be the ``zf`` (final delay values) of a previous call to ``pcen``, or\n        computed by `scipy.signal.lfilter_zi`.\n\n    return_zf : bool\n        If ``True``, return the final filter delay values along with the PCEN output ``P``.\n        This is primarily useful in streaming contexts, where the final state of one\n        block of processing should be used to initialize the next block.\n\n        If ``False`` (default) only the PCEN values ``P`` are returned.\n\n    Returns\n    -------\n    P : np.ndarray, non-negative [shape=(n, m)]\n        The per-channel energy normalized version of ``S``.\n    zf : np.ndarray (optional)\n        The final filter delay values.  Only returned if ``return_zf=True``.\n\n    See Also\n    --------\n    amplitude_to_db\n    librosa.onset.onset_strength\n\n    Examples\n    --------\n    Compare PCEN to log amplitude (dB) scaling on Mel spectra\n\n    >>> import matplotlib.pyplot as plt\n    >>> y, sr = librosa.load(librosa.ex(\'robin\'))\n\n    >>> # We recommend scaling y to the range [-2**31, 2**31[ before applying\n    >>> # PCEN\'s default parameters. Furthermore, we use power=1 to get a\n    >>> # magnitude spectrum instead of a power spectrum.\n    >>> S = librosa.feature.melspectrogram(y=y, sr=sr, power=1)\n    >>> log_S = librosa.amplitude_to_db(S, ref=np.max)\n    >>> pcen_S = librosa.pcen(S * (2**31))\n    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)\n    >>> img = librosa.display.specshow(log_S, x_axis=\'time\', y_axis=\'mel\', ax=ax[0])\n    >>> ax[0].set(title=\'log amplitude (dB)\', xlabel=None)\n    >>> ax[0].label_outer()\n    >>> imgpcen = librosa.display.specshow(pcen_S, x_axis=\'time\', y_axis=\'mel\', ax=ax[1])\n    >>> ax[1].set(title=\'Per-channel energy normalization\')\n    >>> fig.colorbar(img, ax=ax[0], format="%+2.0f dB")\n    >>> fig.colorbar(imgpcen, ax=ax[1])\n\n    Compare PCEN with and without max-filtering\n\n    >>> pcen_max = librosa.pcen(S * (2**31), max_size=3)\n    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)\n    >>> librosa.display.specshow(pcen_S, x_axis=\'time\', y_axis=\'mel\', ax=ax[0])\n    >>> ax[0].set(title=\'Per-channel energy normalization (no max-filter)\')\n    >>> ax[0].label_outer()\n    >>> img = librosa.display.specshow(pcen_max, x_axis=\'time\', y_axis=\'mel\', ax=ax[1])\n    >>> ax[1].set(title=\'Per-channel energy normalization (max_size=3)\')\n    >>> fig.colorbar(img, ax=ax)\n    '
    if power < 0:
        raise ParameterError(f'power={power} must be nonnegative')
    if gain < 0:
        raise ParameterError(f'gain={gain} must be non-negative')
    if bias < 0:
        raise ParameterError(f'bias={bias} must be non-negative')
    if eps <= 0:
        raise ParameterError(f'eps={eps} must be strictly positive')
    if time_constant <= 0:
        raise ParameterError(f'time_constant={time_constant} must be strictly positive')
    if not util.is_positive_int(max_size):
        raise ParameterError(f'max_size={max_size} must be a positive integer')
    if b is None:
        t_frames = time_constant * sr / float(hop_length)
        b = (np.sqrt(1 + 4 * t_frames ** 2) - 1) / (2 * t_frames ** 2)
    if not 0 <= b <= 1:
        raise ParameterError(f'b={b} must be between 0 and 1')
    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn('pcen was called on complex input so phase information will be discarded. To suppress this warning, call pcen(np.abs(D)) instead.', stacklevel=2)
        S = np.abs(S)
    if ref is None:
        if max_size == 1:
            ref = S
        elif S.ndim == 1:
            raise ParameterError('Max-filtering cannot be applied to 1-dimensional input')
        else:
            if max_axis is None:
                if S.ndim != 2:
                    raise ParameterError(f'Max-filtering a {S.ndim:d}-dimensional spectrogram requires you to specify max_axis')
                max_axis = np.mod(1 - axis, 2)
            ref = scipy.ndimage.maximum_filter1d(S, max_size, axis=max_axis)
    if zi is None:
        shape = tuple([1] * ref.ndim)
        zi = np.empty(shape)
        zi[:] = scipy.signal.lfilter_zi([b], [1, b - 1])[:]
    S_smooth: np.ndarray
    zf: np.ndarray
    (S_smooth, zf) = scipy.signal.lfilter([b], [1, b - 1], ref, zi=zi, axis=axis)
    smooth = np.exp(-gain * (np.log(eps) + np.log1p(S_smooth / eps)))
    S_out: np.ndarray
    if power == 0:
        S_out = np.log1p(S * smooth)
    elif bias == 0:
        S_out = np.exp(power * (np.log(S) + np.log(smooth)))
    else:
        S_out = bias ** power * np.expm1(power * np.log1p(S * smooth / bias))
    if return_zf:
        return (S_out, zf)
    else:
        return S_out

def griffinlim(S: np.ndarray, *, n_iter: int=32, hop_length: Optional[int]=None, win_length: Optional[int]=None, n_fft: Optional[int]=None, window: _WindowSpec='hann', center: bool=True, dtype: Optional[DTypeLike]=None, length: Optional[int]=None, pad_mode: _PadModeSTFT='constant', momentum: float=0.99, init: Optional[str]='random', random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]]=None) -> np.ndarray:
    if False:
        return 10
    'Approximate magnitude spectrogram inversion using the "fast" Griffin-Lim algorithm.\n\n    Given a short-time Fourier transform magnitude matrix (``S``), the algorithm randomly\n    initializes phase estimates, and then alternates forward- and inverse-STFT\n    operations. [#]_\n\n    Note that this assumes reconstruction of a real-valued time-domain signal, and\n    that ``S`` contains only the non-negative frequencies (as computed by\n    `stft`).\n\n    The "fast" GL method [#]_ uses a momentum parameter to accelerate convergence.\n\n    .. [#] D. W. Griffin and J. S. Lim,\n        "Signal estimation from modified short-time Fourier transform,"\n        IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.\n\n    .. [#] Perraudin, N., Balazs, P., & Søndergaard, P. L.\n        "A fast Griffin-Lim algorithm,"\n        IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (pp. 1-4),\n        Oct. 2013.\n\n    Parameters\n    ----------\n    S : np.ndarray [shape=(..., n_fft // 2 + 1, t), non-negative]\n        An array of short-time Fourier transform magnitudes as produced by\n        `stft`.\n\n    n_iter : int > 0\n        The number of iterations to run\n\n    hop_length : None or int > 0\n        The hop length of the STFT.  If not provided, it will default to ``n_fft // 4``\n\n    win_length : None or int > 0\n        The window length of the STFT.  By default, it will equal ``n_fft``\n\n    n_fft : None or int > 0\n        The number of samples per frame.\n        By default, this will be inferred from the shape of ``S`` as an even number.\n        However, if an odd frame length was used, you can explicitly set ``n_fft``.\n\n    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]\n        A window specification as supported by `stft` or `istft`\n\n    center : boolean\n        If ``True``, the STFT is assumed to use centered frames.\n        If ``False``, the STFT is assumed to use left-aligned frames.\n\n    dtype : np.dtype\n        Real numeric type for the time-domain signal.  Default is inferred\n        to match the precision of the input spectrogram.\n\n    length : None or int > 0\n        If provided, the output ``y`` is zero-padded or clipped to exactly ``length``\n        samples.\n\n    pad_mode : string\n        If ``center=True``, the padding mode to use at the edges of the signal.\n        By default, STFT uses zero padding.\n\n    momentum : number >= 0\n        The momentum parameter for fast Griffin-Lim.\n        Setting this to 0 recovers the original Griffin-Lim method [1]_.\n        Values near 1 can lead to faster convergence, but above 1 may not converge.\n\n    init : None or \'random\' [default]\n        If \'random\' (the default), then phase values are initialized randomly\n        according to ``random_state``.  This is recommended when the input ``S`` is\n        a magnitude spectrogram with no initial phase estimates.\n\n        If `None`, then the phase is initialized from ``S``.  This is useful when\n        an initial guess for phase can be provided, or when you want to resume\n        Griffin-Lim from a previous output.\n\n    random_state : None, int, np.random.RandomState, or np.random.Generator\n        If int, random_state is the seed used by the random number generator\n        for phase initialization.\n\n        If `np.random.RandomState` or `np.random.Generator` instance, the random number\n        generator itself.\n\n        If `None`, defaults to the `np.random.default_rng()` object.\n\n    Returns\n    -------\n    y : np.ndarray [shape=(..., n)]\n        time-domain signal reconstructed from ``S``\n\n    See Also\n    --------\n    stft\n    istft\n    magphase\n    filters.get_window\n\n    Examples\n    --------\n    A basic STFT inverse example\n\n    >>> y, sr = librosa.load(librosa.ex(\'trumpet\'))\n    >>> # Get the magnitude spectrogram\n    >>> S = np.abs(librosa.stft(y))\n    >>> # Invert using Griffin-Lim\n    >>> y_inv = librosa.griffinlim(S)\n    >>> # Invert without estimating phase\n    >>> y_istft = librosa.istft(S)\n\n    Wave-plot the results\n\n    >>> import matplotlib.pyplot as plt\n    >>> fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)\n    >>> librosa.display.waveshow(y, sr=sr, color=\'b\', ax=ax[0])\n    >>> ax[0].set(title=\'Original\', xlabel=None)\n    >>> ax[0].label_outer()\n    >>> librosa.display.waveshow(y_inv, sr=sr, color=\'g\', ax=ax[1])\n    >>> ax[1].set(title=\'Griffin-Lim reconstruction\', xlabel=None)\n    >>> ax[1].label_outer()\n    >>> librosa.display.waveshow(y_istft, sr=sr, color=\'r\', ax=ax[2])\n    >>> ax[2].set_title(\'Magnitude-only istft reconstruction\')\n    '
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.RandomState(seed=random_state)
    elif isinstance(random_state, (np.random.RandomState, np.random.Generator)):
        rng = random_state
    else:
        raise ParameterError(f'Unsupported random_state={random_state!r}')
    if momentum > 1:
        warnings.warn(f'Griffin-Lim with momentum={momentum} > 1 can be unstable. Proceed with caution!', stacklevel=2)
    elif momentum < 0:
        raise ParameterError(f'griffinlim() called with momentum={momentum} < 0')
    if n_fft is None:
        n_fft = 2 * (S.shape[-2] - 1)
    angles = np.empty(S.shape, dtype=util.dtype_r2c(S.dtype))
    eps = util.tiny(angles)
    if init == 'random':
        angles[:] = util.phasor(2 * np.pi * rng.random(size=S.shape))
    elif init is None:
        angles[:] = 1.0
    else:
        raise ParameterError(f"init={init} must either None or 'random'")
    rebuilt = None
    tprev = None
    inverse = None
    angles *= S
    for _ in range(n_iter):
        inverse = istft(angles, hop_length=hop_length, win_length=win_length, n_fft=n_fft, window=window, center=center, dtype=dtype, length=length, out=inverse)
        rebuilt = stft(inverse, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode, out=rebuilt)
        angles[:] = rebuilt
        if tprev is not None:
            angles -= momentum / (1 + momentum) * tprev
        angles /= np.abs(angles) + eps
        angles *= S
        (rebuilt, tprev) = (tprev, rebuilt)
    return istft(angles, hop_length=hop_length, win_length=win_length, n_fft=n_fft, window=window, center=center, dtype=dtype, length=length, out=inverse)

def _spectrogram(*, y: Optional[np.ndarray]=None, S: Optional[np.ndarray]=None, n_fft: Optional[int]=2048, hop_length: Optional[int]=512, power: float=1, win_length: Optional[int]=None, window: _WindowSpec='hann', center: bool=True, pad_mode: _PadModeSTFT='constant') -> Tuple[np.ndarray, int]:
    if False:
        i = 10
        return i + 15
    'Retrieve a magnitude spectrogram.\n\n    This is primarily used in feature extraction functions that can operate on\n    either audio time-series or spectrogram input.\n\n    Parameters\n    ----------\n    y : None or np.ndarray\n        If provided, an audio time series\n\n    S : None or np.ndarray\n        Spectrogram input, optional\n\n    n_fft : int > 0\n        STFT window size\n\n    hop_length : int > 0\n        STFT hop length\n\n    power : float > 0\n        Exponent for the magnitude spectrogram,\n        e.g., 1 for energy, 2 for power, etc.\n\n    win_length : int <= n_fft [scalar]\n        Each frame of audio is windowed by ``window``.\n        The window will be of length ``win_length`` and then padded\n        with zeros to match ``n_fft``.\n\n        If unspecified, defaults to ``win_length = n_fft``.\n\n    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]\n        - a window specification (string, tuple, or number);\n          see `scipy.signal.get_window`\n        - a window function, such as `scipy.signal.windows.hann`\n        - a vector or array of length ``n_fft``\n\n        .. see also:: `filters.get_window`\n\n    center : boolean\n        - If ``True``, the signal ``y`` is padded so that frame\n          ``t`` is centered at ``y[t * hop_length]``.\n        - If ``False``, then frame ``t`` begins at ``y[t * hop_length]``\n\n    pad_mode : string\n        If ``center=True``, the padding mode to use at the edges of the signal.\n        By default, STFT uses zero padding.\n\n    Returns\n    -------\n    S_out : np.ndarray [dtype=np.float]\n        - If ``S`` is provided as input, then ``S_out == S``\n        - Else, ``S_out = |stft(y, ...)|**power``\n    n_fft : int > 0\n        - If ``S`` is provided, then ``n_fft`` is inferred from ``S``\n        - Else, copied from input\n    '
    if S is not None:
        if n_fft is None or n_fft // 2 + 1 != S.shape[-2]:
            n_fft = 2 * (S.shape[-2] - 1)
    else:
        if n_fft is None:
            raise ParameterError(f'Unable to compute spectrogram with n_fft={n_fft}')
        if y is None:
            raise ParameterError('Input signal must be provided to compute a spectrogram')
        S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=center, window=window, pad_mode=pad_mode)) ** power
    return (S, n_fft)