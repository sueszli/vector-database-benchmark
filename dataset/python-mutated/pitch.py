"""Pitch-tracking and tuning estimation"""
import warnings
import numpy as np
import scipy
import numba
from .spectrum import _spectrogram
from . import convert
from .._cache import cache
from .. import util
from .. import sequence
from ..util.exceptions import ParameterError
from numpy.typing import ArrayLike
from typing import Any, Callable, Optional, Tuple, Union
from .._typing import _WindowSpec, _PadMode, _PadModeSTFT
__all__ = ['estimate_tuning', 'pitch_tuning', 'piptrack', 'yin', 'pyin']

def estimate_tuning(*, y: Optional[np.ndarray]=None, sr: float=22050, S: Optional[np.ndarray]=None, n_fft: Optional[int]=2048, resolution: float=0.01, bins_per_octave: int=12, **kwargs: Any) -> float:
    if False:
        return 10
    "Estimate the tuning of an audio time series or spectrogram input.\n\n    Parameters\n    ----------\n    y : np.ndarray [shape=(..., n)] or None\n        audio signal. Multi-channel is supported..\n    sr : number > 0 [scalar]\n        audio sampling rate of ``y``\n    S : np.ndarray [shape=(..., d, t)] or None\n        magnitude or power spectrogram\n    n_fft : int > 0 [scalar] or None\n        number of FFT bins to use, if ``y`` is provided.\n    resolution : float in `(0, 1)`\n        Resolution of the tuning as a fraction of a bin.\n        0.01 corresponds to measurements in cents.\n    bins_per_octave : int > 0 [scalar]\n        How many frequency bins per octave\n    **kwargs : additional keyword arguments\n        Additional arguments passed to `piptrack`\n\n    Returns\n    -------\n    tuning: float in `[-0.5, 0.5)`\n        estimated tuning deviation (fractions of a bin).\n\n        Note that if multichannel input is provided, a single tuning estimate is provided spanning all\n        channels.\n\n    See Also\n    --------\n    piptrack : Pitch tracking by parabolic interpolation\n\n    Examples\n    --------\n    With time-series input\n\n    >>> y, sr = librosa.load(librosa.ex('trumpet'))\n    >>> librosa.estimate_tuning(y=y, sr=sr)\n    -0.08000000000000002\n\n    In tenths of a cent\n\n    >>> librosa.estimate_tuning(y=y, sr=sr, resolution=1e-3)\n    -0.016000000000000014\n\n    Using spectrogram input\n\n    >>> S = np.abs(librosa.stft(y))\n    >>> librosa.estimate_tuning(S=S, sr=sr)\n    -0.08000000000000002\n\n    Using pass-through arguments to `librosa.piptrack`\n\n    >>> librosa.estimate_tuning(y=y, sr=sr, n_fft=8192,\n    ...                         fmax=librosa.note_to_hz('G#9'))\n    -0.08000000000000002\n    "
    (pitch, mag) = piptrack(y=y, sr=sr, S=S, n_fft=n_fft, **kwargs)
    pitch_mask = pitch > 0
    if pitch_mask.any():
        threshold = np.median(mag[pitch_mask])
    else:
        threshold = 0.0
    return pitch_tuning(pitch[(mag >= threshold) & pitch_mask], resolution=resolution, bins_per_octave=bins_per_octave)

def pitch_tuning(frequencies: ArrayLike, *, resolution: float=0.01, bins_per_octave: int=12) -> float:
    if False:
        return 10
    "Given a collection of pitches, estimate its tuning offset\n    (in fractions of a bin) relative to A440=440.0Hz.\n\n    Parameters\n    ----------\n    frequencies : array-like, float\n        A collection of frequencies detected in the signal.\n        See `piptrack`\n    resolution : float in `(0, 1)`\n        Resolution of the tuning as a fraction of a bin.\n        0.01 corresponds to cents.\n    bins_per_octave : int > 0 [scalar]\n        How many frequency bins per octave\n\n    Returns\n    -------\n    tuning: float in `[-0.5, 0.5)`\n        estimated tuning deviation (fractions of a bin)\n\n    See Also\n    --------\n    estimate_tuning : Estimating tuning from time-series or spectrogram input\n\n    Examples\n    --------\n    >>> # Generate notes at +25 cents\n    >>> freqs = librosa.cqt_frequencies(n_bins=24, fmin=55, tuning=0.25)\n    >>> librosa.pitch_tuning(freqs)\n    0.25\n\n    >>> # Track frequencies from a real spectrogram\n    >>> y, sr = librosa.load(librosa.ex('trumpet'))\n    >>> freqs, times, mags = librosa.reassigned_spectrogram(y, sr=sr,\n    ...                                                     fill_nan=True)\n    >>> # Select out pitches with high energy\n    >>> freqs = freqs[mags > np.median(mags)]\n    >>> librosa.pitch_tuning(freqs)\n    -0.07\n    "
    frequencies = np.atleast_1d(frequencies)
    frequencies = frequencies[frequencies > 0]
    if not np.any(frequencies):
        warnings.warn('Trying to estimate tuning from empty frequency set.', stacklevel=2)
        return 0.0
    residual = np.mod(bins_per_octave * convert.hz_to_octs(frequencies), 1.0)
    residual[residual >= 0.5] -= 1.0
    bins = np.linspace(-0.5, 0.5, int(np.ceil(1.0 / resolution)) + 1)
    (counts, tuning) = np.histogram(residual, bins)
    tuning_est: float = tuning[np.argmax(counts)]
    return tuning_est

@cache(level=30)
def piptrack(*, y: Optional[np.ndarray]=None, sr: float=22050, S: Optional[np.ndarray]=None, n_fft: Optional[int]=2048, hop_length: Optional[int]=None, fmin: float=150.0, fmax: float=4000.0, threshold: float=0.1, win_length: Optional[int]=None, window: _WindowSpec='hann', center: bool=True, pad_mode: _PadModeSTFT='constant', ref: Optional[Union[float, Callable]]=None) -> Tuple[np.ndarray, np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    "Pitch tracking on thresholded parabolically-interpolated STFT.\n\n    This implementation uses the parabolic interpolation method described by [#]_.\n\n    .. [#] https://ccrma.stanford.edu/~jos/sasp/Sinusoidal_Peak_Interpolation.html\n\n    Parameters\n    ----------\n    y : np.ndarray [shape=(..., n)] or None\n        audio signal. Multi-channel is supported..\n\n    sr : number > 0 [scalar]\n        audio sampling rate of ``y``\n\n    S : np.ndarray [shape=(..., d, t)] or None\n        magnitude or power spectrogram\n\n    n_fft : int > 0 [scalar] or None\n        number of FFT bins to use, if ``y`` is provided.\n\n    hop_length : int > 0 [scalar] or None\n        number of samples to hop\n\n    threshold : float in `(0, 1)`\n        A bin in spectrum ``S`` is considered a pitch when it is greater than\n        ``threshold * ref(S)``.\n\n        By default, ``ref(S)`` is taken to be ``max(S, axis=0)`` (the maximum value in\n        each column).\n\n    fmin : float > 0 [scalar]\n        lower frequency cutoff.\n\n    fmax : float > 0 [scalar]\n        upper frequency cutoff.\n\n    win_length : int <= n_fft [scalar]\n        Each frame of audio is windowed by ``window``.\n        The window will be of length `win_length` and then padded\n        with zeros to match ``n_fft``.\n\n        If unspecified, defaults to ``win_length = n_fft``.\n\n    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]\n        - a window specification (string, tuple, or number);\n          see `scipy.signal.get_window`\n        - a window function, such as `scipy.signal.windows.hann`\n        - a vector or array of length ``n_fft``\n\n        .. see also:: `filters.get_window`\n\n    center : boolean\n        - If ``True``, the signal ``y`` is padded so that frame\n          ``t`` is centered at ``y[t * hop_length]``.\n        - If ``False``, then frame ``t`` begins at ``y[t * hop_length]``\n\n    pad_mode : string\n        If ``center=True``, the padding mode to use at the edges of the signal.\n        By default, STFT uses zero-padding.\n\n        See also: `np.pad`.\n\n    ref : scalar or callable [default=np.max]\n        If scalar, the reference value against which ``S`` is compared for determining\n        pitches.\n\n        If callable, the reference value is computed as ``ref(S, axis=0)``.\n\n    Returns\n    -------\n    pitches, magnitudes : np.ndarray [shape=(..., d, t)]\n        Where ``d`` is the subset of FFT bins within ``fmin`` and ``fmax``.\n\n        ``pitches[..., f, t]`` contains instantaneous frequency at bin\n        ``f``, time ``t``\n\n        ``magnitudes[..., f, t]`` contains the corresponding magnitudes.\n\n        Both ``pitches`` and ``magnitudes`` take value 0 at bins\n        of non-maximal magnitude.\n\n    Notes\n    -----\n    This function caches at level 30.\n\n    One of ``S`` or ``y`` must be provided.\n    If ``S`` is not given, it is computed from ``y`` using\n    the default parameters of `librosa.stft`.\n\n    Examples\n    --------\n    Computing pitches from a waveform input\n\n    >>> y, sr = librosa.load(librosa.ex('trumpet'))\n    >>> pitches, magnitudes = librosa.piptrack(y=y, sr=sr)\n\n    Or from a spectrogram input\n\n    >>> S = np.abs(librosa.stft(y))\n    >>> pitches, magnitudes = librosa.piptrack(S=S, sr=sr)\n\n    Or with an alternate reference value for pitch detection, where\n    values above the mean spectral energy in each frame are counted as pitches\n\n    >>> pitches, magnitudes = librosa.piptrack(S=S, sr=sr, threshold=1,\n    ...                                        ref=np.mean)\n    "
    (S, n_fft) = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode)
    S = np.abs(S)
    fmin = np.maximum(fmin, 0)
    fmax = np.minimum(fmax, float(sr) / 2)
    fft_freqs = convert.fft_frequencies(sr=sr, n_fft=n_fft)
    avg = np.gradient(S, axis=-2)
    shift = _parabolic_interpolation(S, axis=-2)
    dskew = 0.5 * avg * shift
    pitches = np.zeros_like(S)
    mags = np.zeros_like(S)
    freq_mask = (fmin <= fft_freqs) & (fft_freqs < fmax)
    freq_mask = util.expand_to(freq_mask, ndim=S.ndim, axes=-2)
    if ref is None:
        ref = np.max
    if callable(ref):
        ref_value = threshold * ref(S, axis=-2)
        ref_value = np.expand_dims(ref_value, -2)
    else:
        ref_value = np.abs(ref)
    idx = np.nonzero(freq_mask & util.localmax(S * (S > ref_value), axis=-2))
    pitches[idx] = (idx[-2] + shift[idx]) * float(sr) / n_fft
    mags[idx] = S[idx] + dskew[idx]
    return (pitches, mags)

def _cumulative_mean_normalized_difference(y_frames: np.ndarray, frame_length: int, win_length: int, min_period: int, max_period: int) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Cumulative mean normalized difference function (equation 8 in [#]_)\n\n    .. [#] De Cheveigné, Alain, and Hideki Kawahara.\n        "YIN, a fundamental frequency estimator for speech and music."\n        The Journal of the Acoustical Society of America 111.4 (2002): 1917-1930.\n\n    Parameters\n    ----------\n    y_frames : np.ndarray [shape=(frame_length, n_frames)]\n        framed audio time series.\n    frame_length : int > 0 [scalar]\n        length of the frames in samples.\n    win_length : int > 0 [scalar]\n        length of the window for calculating autocorrelation in samples.\n    min_period : int > 0 [scalar]\n        minimum period.\n    max_period : int > 0 [scalar]\n        maximum period.\n\n    Returns\n    -------\n    yin_frames : np.ndarray [shape=(max_period-min_period+1,n_frames)]\n        Cumulative mean normalized difference function for each frame.\n    '
    a = np.fft.rfft(y_frames, frame_length, axis=-2)
    b = np.fft.rfft(y_frames[..., win_length:0:-1, :], frame_length, axis=-2)
    acf_frames = np.fft.irfft(a * b, frame_length, axis=-2)[..., win_length:, :]
    acf_frames[np.abs(acf_frames) < 1e-06] = 0
    energy_frames = np.cumsum(y_frames ** 2, axis=-2)
    energy_frames = energy_frames[..., win_length:, :] - energy_frames[..., :-win_length, :]
    energy_frames[np.abs(energy_frames) < 1e-06] = 0
    yin_frames = energy_frames[..., :1, :] + energy_frames - 2 * acf_frames
    yin_numerator = yin_frames[..., min_period:max_period + 1, :]
    tau_range = util.expand_to(np.arange(1, max_period + 1), ndim=yin_frames.ndim, axes=-2)
    cumulative_mean = np.cumsum(yin_frames[..., 1:max_period + 1, :], axis=-2) / tau_range
    yin_denominator = cumulative_mean[..., min_period - 1:max_period, :]
    yin_frames: np.ndarray = yin_numerator / (yin_denominator + util.tiny(yin_denominator))
    return yin_frames

@numba.stencil
def _pi_stencil(x: np.ndarray) -> np.ndarray:
    if False:
        return 10
    'Stencil to compute local parabolic interpolation'
    a = x[1] + x[-1] - 2 * x[0]
    b = (x[1] - x[-1]) / 2
    if np.abs(b) >= np.abs(a):
        return 0
    return -b / a

@numba.guvectorize(['void(float32[:], float32[:])', 'void(float64[:], float64[:])'], '(n)->(n)', cache=True, nopython=True)
def _pi_wrapper(x: np.ndarray, y: np.ndarray) -> None:
    if False:
        return 10
    'Vectorized wrapper for the parabolic interpolation stencil'
    y[:] = _pi_stencil(x)

def _parabolic_interpolation(x: np.ndarray, *, axis: int=-2) -> np.ndarray:
    if False:
        return 10
    'Piecewise parabolic interpolation for yin and pyin.\n\n    Parameters\n    ----------\n    x : np.ndarray\n        array to interpolate\n    axis : int\n        axis along which to interpolate\n\n    Returns\n    -------\n    parabolic_shifts : np.ndarray [shape=x.shape]\n        position of the parabola optima (relative to bin indices)\n\n        Note: the shift at bin `n` is determined as 0 if the estimated\n        optimum is outside the range `[n-1, n+1]`.\n    '
    xi = x.swapaxes(-1, axis)
    shifts = np.empty_like(x)
    shiftsi = shifts.swapaxes(-1, axis)
    _pi_wrapper(xi, shiftsi)
    shiftsi[..., -1] = 0
    shiftsi[..., 0] = 0
    return shifts

def yin(y: np.ndarray, *, fmin: float, fmax: float, sr: float=22050, frame_length: int=2048, win_length: Optional[int]=None, hop_length: Optional[int]=None, trough_threshold: float=0.1, center: bool=True, pad_mode: _PadMode='constant') -> np.ndarray:
    if False:
        print('Hello World!')
    'Fundamental frequency (F0) estimation using the YIN algorithm.\n\n    YIN is an autocorrelation based method for fundamental frequency estimation [#]_.\n    First, a normalized difference function is computed over short (overlapping) frames of audio.\n    Next, the first minimum in the difference function below ``trough_threshold`` is selected as\n    an estimate of the signal\'s period.\n    Finally, the estimated period is refined using parabolic interpolation before converting\n    into the corresponding frequency.\n\n    .. [#] De Cheveigné, Alain, and Hideki Kawahara.\n        "YIN, a fundamental frequency estimator for speech and music."\n        The Journal of the Acoustical Society of America 111.4 (2002): 1917-1930.\n\n    Parameters\n    ----------\n    y : np.ndarray [shape=(..., n)]\n        audio time series. Multi-channel is supported..\n    fmin : number > 0 [scalar]\n        minimum frequency in Hertz.\n        The recommended minimum is ``librosa.note_to_hz(\'C2\')`` (~65 Hz)\n        though lower values may be feasible.\n    fmax : number > fmin, <= sr/2 [scalar]\n        maximum frequency in Hertz.\n        The recommended maximum is ``librosa.note_to_hz(\'C7\')`` (~2093 Hz)\n        though higher values may be feasible.\n    sr : number > 0 [scalar]\n        sampling rate of ``y`` in Hertz.\n    frame_length : int > 0 [scalar]\n        length of the frames in samples.\n        By default, ``frame_length=2048`` corresponds to a time scale of about 93 ms at\n        a sampling rate of 22050 Hz.\n    win_length : None or int > 0 [scalar]\n        length of the window for calculating autocorrelation in samples.\n        If ``None``, defaults to ``frame_length // 2``\n    hop_length : None or int > 0 [scalar]\n        number of audio samples between adjacent YIN predictions.\n        If ``None``, defaults to ``frame_length // 4``.\n    trough_threshold : number > 0 [scalar]\n        absolute threshold for peak estimation.\n    center : boolean\n        If ``True``, the signal `y` is padded so that frame\n        ``D[:, t]`` is centered at `y[t * hop_length]`.\n        If ``False``, then ``D[:, t]`` begins at ``y[t * hop_length]``.\n        Defaults to ``True``,  which simplifies the alignment of ``D`` onto a\n        time grid by means of ``librosa.core.frames_to_samples``.\n    pad_mode : string or function\n        If ``center=True``, this argument is passed to ``np.pad`` for padding\n        the edges of the signal ``y``. By default (``pad_mode="constant"``),\n        ``y`` is padded on both sides with zeros.\n        If ``center=False``,  this argument is ignored.\n        .. see also:: `np.pad`\n\n    Returns\n    -------\n    f0: np.ndarray [shape=(..., n_frames)]\n        time series of fundamental frequencies in Hertz.\n\n        If multi-channel input is provided, f0 curves are estimated separately for each channel.\n\n    See Also\n    --------\n    librosa.pyin :\n        Fundamental frequency (F0) estimation using probabilistic YIN (pYIN).\n\n    Examples\n    --------\n    Computing a fundamental frequency (F0) curve from an audio input\n\n    >>> y = librosa.chirp(fmin=440, fmax=880, duration=5.0)\n    >>> librosa.yin(y, fmin=440, fmax=880)\n    array([442.66354675, 441.95299983, 441.58010963, ...,\n        871.161732  , 873.99001454, 877.04297681])\n    '
    if fmin is None or fmax is None:
        raise ParameterError('both "fmin" and "fmax" must be provided')
    if win_length is None:
        win_length = frame_length // 2
    __check_yin_params(sr=sr, fmax=fmax, fmin=fmin, frame_length=frame_length, win_length=win_length)
    if hop_length is None:
        hop_length = frame_length // 4
    util.valid_audio(y, mono=False)
    if center:
        padding = [(0, 0)] * y.ndim
        padding[-1] = (frame_length // 2, frame_length // 2)
        y = np.pad(y, padding, mode=pad_mode)
    y_frames = util.frame(y, frame_length=frame_length, hop_length=hop_length)
    min_period = int(np.floor(sr / fmax))
    max_period = min(int(np.ceil(sr / fmin)), frame_length - win_length - 1)
    yin_frames = _cumulative_mean_normalized_difference(y_frames, frame_length, win_length, min_period, max_period)
    parabolic_shifts = _parabolic_interpolation(yin_frames)
    is_trough = util.localmin(yin_frames, axis=-2)
    is_trough[..., 0, :] = yin_frames[..., 0, :] < yin_frames[..., 1, :]
    is_threshold_trough = np.logical_and(is_trough, yin_frames < trough_threshold)
    target_shape = list(yin_frames.shape)
    target_shape[-2] = 1
    global_min = np.argmin(yin_frames, axis=-2)
    yin_period = np.argmax(is_threshold_trough, axis=-2)
    global_min = global_min.reshape(target_shape)
    yin_period = yin_period.reshape(target_shape)
    no_trough_below_threshold = np.all(~is_threshold_trough, axis=-2, keepdims=True)
    yin_period[no_trough_below_threshold] = global_min[no_trough_below_threshold]
    yin_period = (min_period + yin_period + np.take_along_axis(parabolic_shifts, yin_period, axis=-2))[..., 0, :]
    f0: np.ndarray = sr / yin_period
    return f0

def pyin(y: np.ndarray, *, fmin: float, fmax: float, sr: float=22050, frame_length: int=2048, win_length: Optional[int]=None, hop_length: Optional[int]=None, n_thresholds: int=100, beta_parameters: Tuple[float, float]=(2, 18), boltzmann_parameter: float=2, resolution: float=0.1, max_transition_rate: float=35.92, switch_prob: float=0.01, no_trough_prob: float=0.01, fill_na: Optional[float]=np.nan, center: bool=True, pad_mode: _PadMode='constant') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if False:
        i = 10
        return i + 15
    'Fundamental frequency (F0) estimation using probabilistic YIN (pYIN).\n\n    pYIN [#]_ is a modificatin of the YIN algorithm [#]_ for fundamental frequency (F0) estimation.\n    In the first step of pYIN, F0 candidates and their probabilities are computed using the YIN algorithm.\n    In the second step, Viterbi decoding is used to estimate the most likely F0 sequence and voicing flags.\n\n    .. [#] Mauch, Matthias, and Simon Dixon.\n        "pYIN: A fundamental frequency estimator using probabilistic threshold distributions."\n        2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2014.\n\n    .. [#] De Cheveigné, Alain, and Hideki Kawahara.\n        "YIN, a fundamental frequency estimator for speech and music."\n        The Journal of the Acoustical Society of America 111.4 (2002): 1917-1930.\n\n    Parameters\n    ----------\n    y : np.ndarray [shape=(..., n)]\n        audio time series. Multi-channel is supported.\n    fmin : number > 0 [scalar]\n        minimum frequency in Hertz.\n        The recommended minimum is ``librosa.note_to_hz(\'C2\')`` (~65 Hz)\n        though lower values may be feasible.\n    fmax : number > fmin, <= sr/2 [scalar]\n        maximum frequency in Hertz.\n        The recommended maximum is ``librosa.note_to_hz(\'C7\')`` (~2093 Hz)\n        though higher values may be feasible.\n    sr : number > 0 [scalar]\n        sampling rate of ``y`` in Hertz.\n    frame_length : int > 0 [scalar]\n        length of the frames in samples.\n        By default, ``frame_length=2048`` corresponds to a time scale of about 93 ms at\n        a sampling rate of 22050 Hz.\n    win_length : None or int > 0 [scalar]\n        length of the window for calculating autocorrelation in samples.\n        If ``None``, defaults to ``frame_length // 2``\n    hop_length : None or int > 0 [scalar]\n        number of audio samples between adjacent pYIN predictions.\n        If ``None``, defaults to ``frame_length // 4``.\n    n_thresholds : int > 0 [scalar]\n        number of thresholds for peak estimation.\n    beta_parameters : tuple\n        shape parameters for the beta distribution prior over thresholds.\n    boltzmann_parameter : number > 0 [scalar]\n        shape parameter for the Boltzmann distribution prior over troughs.\n        Larger values will assign more mass to smaller periods.\n    resolution : float in `(0, 1)`\n        Resolution of the pitch bins.\n        0.01 corresponds to cents.\n    max_transition_rate : float > 0\n        maximum pitch transition rate in octaves per second.\n    switch_prob : float in ``(0, 1)``\n        probability of switching from voiced to unvoiced or vice versa.\n    no_trough_prob : float in ``(0, 1)``\n        maximum probability to add to global minimum if no trough is below threshold.\n    fill_na : None, float, or ``np.nan``\n        default value for unvoiced frames of ``f0``.\n        If ``None``, the unvoiced frames will contain a best guess value.\n    center : boolean\n        If ``True``, the signal ``y`` is padded so that frame\n        ``D[:, t]`` is centered at ``y[t * hop_length]``.\n        If ``False``, then ``D[:, t]`` begins at ``y[t * hop_length]``.\n        Defaults to ``True``,  which simplifies the alignment of ``D`` onto a\n        time grid by means of ``librosa.core.frames_to_samples``.\n    pad_mode : string or function\n        If ``center=True``, this argument is passed to ``np.pad`` for padding\n        the edges of the signal ``y``. By default (``pad_mode="constant"``),\n        ``y`` is padded on both sides with zeros.\n        If ``center=False``,  this argument is ignored.\n        .. see also:: `np.pad`\n\n    Returns\n    -------\n    f0: np.ndarray [shape=(..., n_frames)]\n        time series of fundamental frequencies in Hertz.\n    voiced_flag: np.ndarray [shape=(..., n_frames)]\n        time series containing boolean flags indicating whether a frame is voiced or not.\n    voiced_prob: np.ndarray [shape=(..., n_frames)]\n        time series containing the probability that a frame is voiced.\n    .. note:: If multi-channel input is provided, f0 and voicing are estimated separately for each channel.\n\n    See Also\n    --------\n    librosa.yin :\n        Fundamental frequency (F0) estimation using the YIN algorithm.\n\n    Examples\n    --------\n    Computing a fundamental frequency (F0) curve from an audio input\n\n    >>> y, sr = librosa.load(librosa.ex(\'trumpet\'))\n    >>> f0, voiced_flag, voiced_probs = librosa.pyin(y,\n    ...                                              fmin=librosa.note_to_hz(\'C2\'),\n    ...                                              fmax=librosa.note_to_hz(\'C7\'))\n    >>> times = librosa.times_like(f0)\n\n    Overlay F0 over a spectrogram\n\n    >>> import matplotlib.pyplot as plt\n    >>> D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)\n    >>> fig, ax = plt.subplots()\n    >>> img = librosa.display.specshow(D, x_axis=\'time\', y_axis=\'log\', ax=ax)\n    >>> ax.set(title=\'pYIN fundamental frequency estimation\')\n    >>> fig.colorbar(img, ax=ax, format="%+2.f dB")\n    >>> ax.plot(times, f0, label=\'f0\', color=\'cyan\', linewidth=3)\n    >>> ax.legend(loc=\'upper right\')\n    '
    if fmin is None or fmax is None:
        raise ParameterError('both "fmin" and "fmax" must be provided')
    if win_length is None:
        win_length = frame_length // 2
    __check_yin_params(sr=sr, fmax=fmax, fmin=fmin, frame_length=frame_length, win_length=win_length)
    if hop_length is None:
        hop_length = frame_length // 4
    util.valid_audio(y, mono=False)
    if center:
        padding = [(0, 0) for _ in y.shape]
        padding[-1] = (frame_length // 2, frame_length // 2)
        y = np.pad(y, padding, mode=pad_mode)
    y_frames = util.frame(y, frame_length=frame_length, hop_length=hop_length)
    min_period = int(np.floor(sr / fmax))
    max_period = min(int(np.ceil(sr / fmin)), frame_length - win_length - 1)
    yin_frames = _cumulative_mean_normalized_difference(y_frames, frame_length, win_length, min_period, max_period)
    parabolic_shifts = _parabolic_interpolation(yin_frames)
    thresholds = np.linspace(0, 1, n_thresholds + 1)
    beta_cdf = scipy.stats.beta.cdf(thresholds, beta_parameters[0], beta_parameters[1])
    beta_probs = np.diff(beta_cdf)
    n_bins_per_semitone = int(np.ceil(1.0 / resolution))
    n_pitch_bins = int(np.floor(12 * n_bins_per_semitone * np.log2(fmax / fmin))) + 1

    def _helper(a, b):
        if False:
            for i in range(10):
                print('nop')
        return __pyin_helper(a, b, sr, thresholds, boltzmann_parameter, beta_probs, no_trough_prob, min_period, fmin, n_pitch_bins, n_bins_per_semitone)
    helper = np.vectorize(_helper, signature='(f,t),(k,t)->(1,d,t),(j,t)')
    (observation_probs, voiced_prob) = helper(yin_frames, parabolic_shifts)
    max_semitones_per_frame = round(max_transition_rate * 12 * hop_length / sr)
    transition_width = max_semitones_per_frame * n_bins_per_semitone + 1
    transition = sequence.transition_local(n_pitch_bins, transition_width, window='triangle', wrap=False)
    t_switch = sequence.transition_loop(2, 1 - switch_prob)
    transition = np.kron(t_switch, transition)
    p_init = np.zeros(2 * n_pitch_bins)
    p_init[n_pitch_bins:] = 1 / n_pitch_bins
    states = sequence.viterbi(observation_probs, transition, p_init=p_init)
    freqs = fmin * 2 ** (np.arange(n_pitch_bins) / (12 * n_bins_per_semitone))
    f0 = freqs[states % n_pitch_bins]
    voiced_flag = states < n_pitch_bins
    if fill_na is not None:
        f0[~voiced_flag] = fill_na
    return (f0[..., 0, :], voiced_flag[..., 0, :], voiced_prob[..., 0, :])

def __pyin_helper(yin_frames, parabolic_shifts, sr, thresholds, boltzmann_parameter, beta_probs, no_trough_prob, min_period, fmin, n_pitch_bins, n_bins_per_semitone):
    if False:
        for i in range(10):
            print('nop')
    yin_probs = np.zeros_like(yin_frames)
    for (i, yin_frame) in enumerate(yin_frames.T):
        is_trough = util.localmin(yin_frame)
        is_trough[0] = yin_frame[0] < yin_frame[1]
        (trough_index,) = np.nonzero(is_trough)
        if len(trough_index) == 0:
            continue
        trough_heights = yin_frame[trough_index]
        trough_thresholds = np.less.outer(trough_heights, thresholds[1:])
        trough_positions = np.cumsum(trough_thresholds, axis=0) - 1
        n_troughs = np.count_nonzero(trough_thresholds, axis=0)
        trough_prior = scipy.stats.boltzmann.pmf(trough_positions, boltzmann_parameter, n_troughs)
        trough_prior[~trough_thresholds] = 0
        probs = trough_prior.dot(beta_probs)
        global_min = np.argmin(trough_heights)
        n_thresholds_below_min = np.count_nonzero(~trough_thresholds[global_min, :])
        probs[global_min] += no_trough_prob * np.sum(beta_probs[:n_thresholds_below_min])
        yin_probs[trough_index, i] = probs
    (yin_period, frame_index) = np.nonzero(yin_probs)
    period_candidates = min_period + yin_period
    period_candidates = period_candidates + parabolic_shifts[yin_period, frame_index]
    f0_candidates = sr / period_candidates
    bin_index = 12 * n_bins_per_semitone * np.log2(f0_candidates / fmin)
    bin_index = np.clip(np.round(bin_index), 0, n_pitch_bins).astype(int)
    observation_probs = np.zeros((2 * n_pitch_bins, yin_frames.shape[1]))
    observation_probs[bin_index, frame_index] = yin_probs[yin_period, frame_index]
    voiced_prob = np.clip(np.sum(observation_probs[:n_pitch_bins, :], axis=0, keepdims=True), 0, 1)
    observation_probs[n_pitch_bins:, :] = (1 - voiced_prob) / n_pitch_bins
    return (observation_probs[np.newaxis], voiced_prob)

def __check_yin_params(*, sr: float, fmax: float, fmin: float, frame_length: int, win_length: int):
    if False:
        for i in range(10):
            print('nop')
    'Check the feasibility of yin/pyin parameters against\n    the following conditions:\n\n    1. 0 < fmin < fmax <= sr/2\n    2. frame_length - win_length - 1 > sr/fmax\n    '
    if fmax > sr / 2:
        raise ParameterError(f'fmax={fmax:.3f} cannot exceed Nyquist frequency {sr / 2}')
    if fmin >= fmax:
        raise ParameterError(f'fmin={fmin:.3f} must be less than fmax={fmax:.3f}')
    if fmin <= 0:
        raise ParameterError(f'fmin={fmin:.3f} must be strictly positive')
    if win_length >= frame_length:
        raise ParameterError(f'win_length={win_length} must be less than frame_length={frame_length}')
    if frame_length - win_length - 1 <= sr // fmax:
        fmax_feasible = sr / (frame_length - win_length - 1)
        frame_length_feasible = int(np.ceil(sr / fmax) + win_length + 1)
        raise ParameterError(f'fmax={fmax:.3f} is too small for frame_length={frame_length}, win_length={win_length}, and sr={sr}. Either increase to fmax={fmax_feasible:.3f} or frame_length={frame_length_feasible}')