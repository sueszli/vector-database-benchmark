"""
Effects
=======

Harmonic-percussive source separation
-------------------------------------
.. autosummary::
    :toctree: generated/

    hpss
    harmonic
    percussive

Time and frequency
------------------
.. autosummary::
    :toctree: generated/

    time_stretch
    pitch_shift

Miscellaneous
-------------
.. autosummary::
    :toctree: generated/

    remix
    trim
    split
    preemphasis
    deemphasis
"""
import numpy as np
import scipy.signal
from . import core
from . import decompose
from . import feature
from . import util
from .util.exceptions import ParameterError
from typing import Any, Callable, Iterable, Optional, Tuple, Union, overload
from typing_extensions import Literal
from numpy.typing import ArrayLike
__all__ = ['hpss', 'harmonic', 'percussive', 'time_stretch', 'pitch_shift', 'remix', 'trim', 'split']

def hpss(y: np.ndarray, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    if False:
        return 10
    "Decompose an audio time series into harmonic and percussive components.\n\n    This function automates the STFT->HPSS->ISTFT pipeline, and ensures that\n    the output waveforms have equal length to the input waveform ``y``.\n\n    Parameters\n    ----------\n    y : np.ndarray [shape=(..., n)]\n        audio time series. Multi-channel is supported.\n    **kwargs : additional keyword arguments.\n        See `librosa.decompose.hpss` for details.\n\n    Returns\n    -------\n    y_harmonic : np.ndarray [shape=(..., n)]\n        audio time series of the harmonic elements\n    y_percussive : np.ndarray [shape=(..., n)]\n        audio time series of the percussive elements\n\n    See Also\n    --------\n    harmonic : Extract only the harmonic component\n    percussive : Extract only the percussive component\n    librosa.decompose.hpss : HPSS on spectrograms\n\n    Examples\n    --------\n    >>> # Extract harmonic and percussive components\n    >>> y, sr = librosa.load(librosa.ex('choice'))\n    >>> y_harmonic, y_percussive = librosa.effects.hpss(y)\n\n    >>> # Get a more isolated percussive component by widening its margin\n    >>> y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(1.0,5.0))\n    "
    stft = core.stft(y)
    (stft_harm, stft_perc) = decompose.hpss(stft, **kwargs)
    y_harm = core.istft(stft_harm, dtype=y.dtype, length=y.shape[-1])
    y_perc = core.istft(stft_perc, dtype=y.dtype, length=y.shape[-1])
    return (y_harm, y_perc)

def harmonic(y: np.ndarray, **kwargs: Any) -> np.ndarray:
    if False:
        while True:
            i = 10
    "Extract harmonic elements from an audio time-series.\n\n    Parameters\n    ----------\n    y : np.ndarray [shape=(..., n)]\n        audio time series. Multi-channel is supported.\n    **kwargs : additional keyword arguments.\n        See `librosa.decompose.hpss` for details.\n\n    Returns\n    -------\n    y_harmonic : np.ndarray [shape=(..., n)]\n        audio time series of just the harmonic portion\n\n    See Also\n    --------\n    hpss : Separate harmonic and percussive components\n    percussive : Extract only the percussive component\n    librosa.decompose.hpss : HPSS for spectrograms\n\n    Examples\n    --------\n    >>> # Extract harmonic component\n    >>> y, sr = librosa.load(librosa.ex('choice'))\n    >>> y_harmonic = librosa.effects.harmonic(y)\n\n    >>> # Use a margin > 1.0 for greater harmonic separation\n    >>> y_harmonic = librosa.effects.harmonic(y, margin=3.0)\n    "
    stft = core.stft(y)
    stft_harm = decompose.hpss(stft, **kwargs)[0]
    y_harm = core.istft(stft_harm, dtype=y.dtype, length=y.shape[-1])
    return y_harm

def percussive(y: np.ndarray, **kwargs: Any) -> np.ndarray:
    if False:
        print('Hello World!')
    "Extract percussive elements from an audio time-series.\n\n    Parameters\n    ----------\n    y : np.ndarray [shape=(..., n)]\n        audio time series. Multi-channel is supported.\n    **kwargs : additional keyword arguments.\n        See `librosa.decompose.hpss` for details.\n\n    Returns\n    -------\n    y_percussive : np.ndarray [shape=(..., n)]\n        audio time series of just the percussive portion\n\n    See Also\n    --------\n    hpss : Separate harmonic and percussive components\n    harmonic : Extract only the harmonic component\n    librosa.decompose.hpss : HPSS for spectrograms\n\n    Examples\n    --------\n    >>> # Extract percussive component\n    >>> y, sr = librosa.load(librosa.ex('choice'))\n    >>> y_percussive = librosa.effects.percussive(y)\n\n    >>> # Use a margin > 1.0 for greater percussive separation\n    >>> y_percussive = librosa.effects.percussive(y, margin=3.0)\n    "
    stft = core.stft(y)
    stft_perc = decompose.hpss(stft, **kwargs)[1]
    y_perc = core.istft(stft_perc, dtype=y.dtype, length=y.shape[-1])
    return y_perc

def time_stretch(y: np.ndarray, *, rate: float, **kwargs: Any) -> np.ndarray:
    if False:
        return 10
    "Time-stretch an audio series by a fixed rate.\n\n    Parameters\n    ----------\n    y : np.ndarray [shape=(..., n)]\n        audio time series. Multi-channel is supported.\n    rate : float > 0 [scalar]\n        Stretch factor.  If ``rate > 1``, then the signal is sped up.\n        If ``rate < 1``, then the signal is slowed down.\n    **kwargs : additional keyword arguments.\n        See `librosa.decompose.stft` for details.\n\n    Returns\n    -------\n    y_stretch : np.ndarray [shape=(..., round(n/rate))]\n        audio time series stretched by the specified rate\n\n    See Also\n    --------\n    pitch_shift :\n        pitch shifting\n    librosa.phase_vocoder :\n        spectrogram phase vocoder\n    pyrubberband.pyrb.time_stretch :\n        high-quality time stretching using RubberBand\n\n    Examples\n    --------\n    Compress to be twice as fast\n\n    >>> y, sr = librosa.load(librosa.ex('choice'))\n    >>> y_fast = librosa.effects.time_stretch(y, rate=2.0)\n\n    Or half the original speed\n\n    >>> y_slow = librosa.effects.time_stretch(y, rate=0.5)\n    "
    if rate <= 0:
        raise ParameterError('rate must be a positive number')
    stft = core.stft(y, **kwargs)
    stft_stretch = core.phase_vocoder(stft, rate=rate, hop_length=kwargs.get('hop_length', None), n_fft=kwargs.get('n_fft', None))
    len_stretch = int(round(y.shape[-1] / rate))
    y_stretch = core.istft(stft_stretch, dtype=y.dtype, length=len_stretch, **kwargs)
    return y_stretch

def pitch_shift(y: np.ndarray, *, sr: float, n_steps: float, bins_per_octave: int=12, res_type: str='soxr_hq', scale: bool=False, **kwargs: Any) -> np.ndarray:
    if False:
        print('Hello World!')
    "Shift the pitch of a waveform by ``n_steps`` steps.\n\n    A step is equal to a semitone if ``bins_per_octave`` is set to 12.\n\n    Parameters\n    ----------\n    y : np.ndarray [shape=(..., n)]\n        audio time series. Multi-channel is supported.\n\n    sr : number > 0 [scalar]\n        audio sampling rate of ``y``\n\n    n_steps : float [scalar]\n        how many (fractional) steps to shift ``y``\n\n    bins_per_octave : int > 0 [scalar]\n        how many steps per octave\n\n    res_type : string\n        Resample type. By default, 'soxr_hq' is used.\n\n        See `librosa.resample` for more information.\n\n    scale : bool\n        Scale the resampled signal so that ``y`` and ``y_hat`` have approximately\n        equal total energy.\n\n    **kwargs : additional keyword arguments.\n        See `librosa.decompose.stft` for details.\n\n    Returns\n    -------\n    y_shift : np.ndarray [shape=(..., n)]\n        The pitch-shifted audio time-series\n\n    See Also\n    --------\n    time_stretch :\n        time stretching\n    librosa.phase_vocoder :\n        spectrogram phase vocoder\n    pyrubberband.pyrb.pitch_shift :\n        high-quality pitch shifting using RubberBand\n\n    Examples\n    --------\n    Shift up by a major third (four steps if ``bins_per_octave`` is 12)\n\n    >>> y, sr = librosa.load(librosa.ex('choice'))\n    >>> y_third = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)\n\n    Shift down by a tritone (six steps if ``bins_per_octave`` is 12)\n\n    >>> y_tritone = librosa.effects.pitch_shift(y, sr=sr, n_steps=-6)\n\n    Shift up by 3 quarter-tones\n\n    >>> y_three_qt = librosa.effects.pitch_shift(y, sr=sr, n_steps=3,\n    ...                                          bins_per_octave=24)\n    "
    if not util.is_positive_int(bins_per_octave):
        raise ParameterError(f'bins_per_octave={bins_per_octave} must be a positive integer.')
    rate = 2.0 ** (-float(n_steps) / bins_per_octave)
    y_shift = core.resample(time_stretch(y, rate=rate, **kwargs), orig_sr=float(sr) / rate, target_sr=sr, res_type=res_type, scale=scale)
    return util.fix_length(y_shift, size=y.shape[-1])

def remix(y: np.ndarray, intervals: Iterable[Tuple[int, int]], *, align_zeros: bool=True) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    "Remix an audio signal by re-ordering time intervals.\n\n    Parameters\n    ----------\n    y : np.ndarray [shape=(..., t)]\n        Audio time series. Multi-channel is supported.\n    intervals : iterable of tuples (start, end)\n        An iterable (list-like or generator) where the ``i``th item\n        ``intervals[i]`` indicates the start and end (in samples)\n        of a slice of ``y``.\n    align_zeros : boolean\n        If ``True``, interval boundaries are mapped to the closest\n        zero-crossing in ``y``.  If ``y`` is stereo, zero-crossings\n        are computed after converting to mono.\n\n    Returns\n    -------\n    y_remix : np.ndarray [shape=(..., d)]\n        ``y`` remixed in the order specified by ``intervals``\n\n    Examples\n    --------\n    Load in the example track and reverse the beats\n\n    >>> y, sr = librosa.load(librosa.ex('choice'))\n\n    Compute beats\n\n    >>> _, beat_frames = librosa.beat.beat_track(y=y, sr=sr,\n    ...                                          hop_length=512)\n\n    Convert from frames to sample indices\n\n    >>> beat_samples = librosa.frames_to_samples(beat_frames)\n\n    Generate intervals from consecutive events\n\n    >>> intervals = librosa.util.frame(beat_samples, frame_length=2,\n    ...                                hop_length=1).T\n\n    Reverse the beat intervals\n\n    >>> y_out = librosa.effects.remix(y, intervals[::-1])\n    "
    y_out = []
    if align_zeros:
        y_mono = core.to_mono(y)
        zeros = np.nonzero(core.zero_crossings(y_mono))[-1]
        zeros = np.append(zeros, [len(y_mono)])
    for interval in intervals:
        if align_zeros:
            interval = zeros[util.match_events(interval, zeros)]
        y_out.append(y[..., interval[0]:interval[1]])
    return np.concatenate(y_out, axis=-1)

def _signal_to_frame_nonsilent(y: np.ndarray, frame_length: int=2048, hop_length: int=512, top_db: float=60, ref: Union[Callable, float]=np.max, aggregate: Callable=np.max) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    'Frame-wise non-silent indicator for audio input.\n\n    This is a helper function for `trim` and `split`.\n\n    Parameters\n    ----------\n    y : np.ndarray\n        Audio signal, mono or stereo\n\n    frame_length : int > 0\n        The number of samples per frame\n\n    hop_length : int > 0\n        The number of samples between frames\n\n    top_db : number > 0\n        The threshold (in decibels) below reference to consider as\n        silence\n\n    ref : callable or float\n        The reference amplitude\n\n    aggregate : callable [default: np.max]\n        Function to aggregate dB measurements across channels (if y.ndim > 1)\n\n        Note: for multiple leading axes, this is performed using ``np.apply_over_axes``.\n\n    Returns\n    -------\n    non_silent : np.ndarray, shape=(m,), dtype=bool\n        Indicator of non-silent frames\n    '
    mse = feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
    db = core.amplitude_to_db(mse[..., 0, :], ref=ref, top_db=None)
    if db.ndim > 1:
        db = np.apply_over_axes(aggregate, db, range(db.ndim - 1))
        db = np.squeeze(db, axis=tuple(range(db.ndim - 1)))
    return db > -top_db

def trim(y: np.ndarray, *, top_db: float=60, ref: Union[float, Callable]=np.max, frame_length: int=2048, hop_length: int=512, aggregate: Callable=np.max) -> Tuple[np.ndarray, np.ndarray]:
    if False:
        while True:
            i = 10
    "Trim leading and trailing silence from an audio signal.\n\n    Parameters\n    ----------\n    y : np.ndarray, shape=(..., n)\n        Audio signal. Multi-channel is supported.\n    top_db : number > 0\n        The threshold (in decibels) below reference to consider as\n        silence\n    ref : number or callable\n        The reference amplitude.  By default, it uses `np.max` and compares\n        to the peak amplitude in the signal.\n    frame_length : int > 0\n        The number of samples per analysis frame\n    hop_length : int > 0\n        The number of samples between analysis frames\n    aggregate : callable [default: np.max]\n        Function to aggregate across channels (if y.ndim > 1)\n\n    Returns\n    -------\n    y_trimmed : np.ndarray, shape=(..., m)\n        The trimmed signal\n    index : np.ndarray, shape=(2,)\n        the interval of ``y`` corresponding to the non-silent region:\n        ``y_trimmed = y[index[0]:index[1]]`` (for mono) or\n        ``y_trimmed = y[:, index[0]:index[1]]`` (for stereo).\n\n    Examples\n    --------\n    >>> # Load some audio\n    >>> y, sr = librosa.load(librosa.ex('choice'))\n    >>> # Trim the beginning and ending silence\n    >>> yt, index = librosa.effects.trim(y)\n    >>> # Print the durations\n    >>> print(librosa.get_duration(y), librosa.get_duration(yt))\n    25.025986394557822 25.007891156462584\n    "
    non_silent = _signal_to_frame_nonsilent(y, frame_length=frame_length, hop_length=hop_length, ref=ref, top_db=top_db, aggregate=aggregate)
    nonzero = np.flatnonzero(non_silent)
    if nonzero.size > 0:
        start = int(core.frames_to_samples(nonzero[0], hop_length=hop_length))
        end = min(y.shape[-1], int(core.frames_to_samples(nonzero[-1] + 1, hop_length=hop_length)))
    else:
        (start, end) = (0, 0)
    full_index = [slice(None)] * y.ndim
    full_index[-1] = slice(start, end)
    return (y[tuple(full_index)], np.asarray([start, end]))

def split(y: np.ndarray, *, top_db: float=60, ref: Union[float, Callable]=np.max, frame_length: int=2048, hop_length: int=512, aggregate: Callable=np.max) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    'Split an audio signal into non-silent intervals.\n\n    Parameters\n    ----------\n    y : np.ndarray, shape=(..., n)\n        An audio signal. Multi-channel is supported.\n    top_db : number > 0\n        The threshold (in decibels) below reference to consider as\n        silence\n    ref : number or callable\n        The reference amplitude.  By default, it uses `np.max` and compares\n        to the peak amplitude in the signal.\n    frame_length : int > 0\n        The number of samples per analysis frame\n    hop_length : int > 0\n        The number of samples between analysis frames\n    aggregate : callable [default: np.max]\n        Function to aggregate across channels (if y.ndim > 1)\n\n    Returns\n    -------\n    intervals : np.ndarray, shape=(m, 2)\n        ``intervals[i] == (start_i, end_i)`` are the start and end time\n        (in samples) of non-silent interval ``i``.\n    '
    non_silent = _signal_to_frame_nonsilent(y, frame_length=frame_length, hop_length=hop_length, ref=ref, top_db=top_db, aggregate=aggregate)
    edges = np.flatnonzero(np.diff(non_silent.astype(int)))
    edges = [edges + 1]
    if non_silent[0]:
        edges.insert(0, np.array([0]))
    if non_silent[-1]:
        edges.append(np.array([len(non_silent)]))
    edges = core.frames_to_samples(np.concatenate(edges), hop_length=hop_length)
    edges = np.minimum(edges, y.shape[-1])
    edges = edges.reshape((-1, 2))
    return edges

@overload
def preemphasis(y: np.ndarray, *, coef: float=..., zi: Optional[ArrayLike]=..., return_zf: Literal[False]=...) -> np.ndarray:
    if False:
        print('Hello World!')
    ...

@overload
def preemphasis(y: np.ndarray, *, coef: float=..., zi: Optional[ArrayLike]=..., return_zf: Literal[True]) -> Tuple[np.ndarray, np.ndarray]:
    if False:
        print('Hello World!')
    ...

@overload
def preemphasis(y: np.ndarray, *, coef: float=..., zi: Optional[ArrayLike]=..., return_zf: bool) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if False:
        return 10
    ...

def preemphasis(y: np.ndarray, *, coef: float=0.97, zi: Optional[ArrayLike]=None, return_zf: bool=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if False:
        return 10
    'Pre-emphasize an audio signal with a first-order differencing filter:\n\n        y[n] -> y[n] - coef * y[n-1]\n\n    Parameters\n    ----------\n    y : np.ndarray [shape=(..., n)]\n        Audio signal. Multi-channel is supported.\n\n    coef : positive number\n        Pre-emphasis coefficient.  Typical values of ``coef`` are between 0 and 1.\n\n        At the limit ``coef=0``, the signal is unchanged.\n\n        At ``coef=1``, the result is the first-order difference of the signal.\n\n        The default (0.97) matches the pre-emphasis filter used in the HTK\n        implementation of MFCCs [#]_.\n\n        .. [#] https://htk.eng.cam.ac.uk/\n\n    zi : number\n        Initial filter state.  When making successive calls to non-overlapping\n        frames, this can be set to the ``zf`` returned from the previous call.\n        (See example below.)\n\n        By default ``zi`` is initialized as ``2*y[0] - y[1]``.\n\n    return_zf : boolean\n        If ``True``, return the final filter state.\n        If ``False``, only return the pre-emphasized signal.\n\n    Returns\n    -------\n    y_out : np.ndarray\n        pre-emphasized signal\n    zf : number\n        if ``return_zf=True``, the final filter state is also returned\n\n    Examples\n    --------\n    Apply a standard pre-emphasis filter\n\n    >>> import matplotlib.pyplot as plt\n    >>> y, sr = librosa.load(librosa.ex(\'trumpet\'))\n    >>> y_filt = librosa.effects.preemphasis(y)\n    >>> # and plot the results for comparison\n    >>> S_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max, top_db=None)\n    >>> S_preemph = librosa.amplitude_to_db(np.abs(librosa.stft(y_filt)), ref=np.max, top_db=None)\n    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)\n    >>> librosa.display.specshow(S_orig, y_axis=\'log\', x_axis=\'time\', ax=ax[0])\n    >>> ax[0].set(title=\'Original signal\')\n    >>> ax[0].label_outer()\n    >>> img = librosa.display.specshow(S_preemph, y_axis=\'log\', x_axis=\'time\', ax=ax[1])\n    >>> ax[1].set(title=\'Pre-emphasized signal\')\n    >>> fig.colorbar(img, ax=ax, format="%+2.f dB")\n\n    Apply pre-emphasis in pieces for block streaming.  Note that the second block\n    initializes ``zi`` with the final state ``zf`` returned by the first call.\n\n    >>> y_filt_1, zf = librosa.effects.preemphasis(y[:1000], return_zf=True)\n    >>> y_filt_2, zf = librosa.effects.preemphasis(y[1000:], zi=zf, return_zf=True)\n    >>> np.allclose(y_filt, np.concatenate([y_filt_1, y_filt_2]))\n    True\n\n    See Also\n    --------\n    deemphasis\n    '
    b = np.asarray([1.0, -coef], dtype=y.dtype)
    a = np.asarray([1.0], dtype=y.dtype)
    if zi is None:
        zi = 2 * y[..., 0:1] - y[..., 1:2]
    zi = np.atleast_1d(zi)
    y_out: np.ndarray
    z_f: np.ndarray
    (y_out, z_f) = scipy.signal.lfilter(b, a, y, zi=np.asarray(zi, dtype=y.dtype))
    if return_zf:
        return (y_out, z_f)
    return y_out

@overload
def deemphasis(y: np.ndarray, *, coef: float=..., zi: Optional[ArrayLike]=..., return_zf: Literal[False]=...) -> np.ndarray:
    if False:
        return 10
    ...

@overload
def deemphasis(y: np.ndarray, *, coef: float=..., zi: Optional[ArrayLike]=..., return_zf: Literal[True]) -> Tuple[np.ndarray, np.ndarray]:
    if False:
        i = 10
        return i + 15
    ...

def deemphasis(y: np.ndarray, *, coef: float=0.97, zi: Optional[ArrayLike]=None, return_zf: bool=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if False:
        for i in range(10):
            print('nop')
    "De-emphasize an audio signal with the inverse operation of preemphasis():\n\n    If y = preemphasis(x, coef=coef, zi=zi), the deemphasis is:\n\n    >>> x[i] = y[i] + coef * x[i-1]\n    >>> x = deemphasis(y, coef=coef, zi=zi)\n\n    Parameters\n    ----------\n    y : np.ndarray [shape=(..., n)]\n        Audio signal. Multi-channel is supported.\n\n    coef : positive number\n        Pre-emphasis coefficient.  Typical values of ``coef`` are between 0 and 1.\n\n        At the limit ``coef=0``, the signal is unchanged.\n\n        At ``coef=1``, the result is the first-order difference of the signal.\n\n        The default (0.97) matches the pre-emphasis filter used in the HTK\n        implementation of MFCCs [#]_.\n\n        .. [#] https://htk.eng.cam.ac.uk/\n\n    zi : number\n        Initial filter state. If inverting a previous preemphasis(), the same value should be used.\n\n        By default ``zi`` is initialized as\n        ``((2 - coef) * y[0] - y[1]) / (3 - coef)``. This\n        value corresponds to the transformation of the default initialization of ``zi`` in ``preemphasis()``,\n        ``2*x[0] - x[1]``.\n\n    return_zf : boolean\n        If ``True``, return the final filter state.\n        If ``False``, only return the pre-emphasized signal.\n\n    Returns\n    -------\n    y_out : np.ndarray\n        de-emphasized signal\n    zf : number\n        if ``return_zf=True``, the final filter state is also returned\n\n    Examples\n    --------\n    Apply a standard pre-emphasis filter and invert it with de-emphasis\n\n    >>> y, sr = librosa.load(librosa.ex('trumpet'))\n    >>> y_filt = librosa.effects.preemphasis(y)\n    >>> y_deemph = librosa.effects.deemphasis(y_filt)\n    >>> np.allclose(y, y_deemph)\n    True\n\n    See Also\n    --------\n    preemphasis\n    "
    b = np.array([1.0, -coef], dtype=y.dtype)
    a = np.array([1.0], dtype=y.dtype)
    y_out: np.ndarray
    zf: np.ndarray
    if zi is None:
        zi = np.zeros(list(y.shape[:-1]) + [1], dtype=y.dtype)
        (y_out, zf) = scipy.signal.lfilter(a, b, y, zi=zi)
        y_out -= ((2 - coef) * y[..., 0:1] - y[..., 1:2]) / (3 - coef) * coef ** np.arange(y.shape[-1])
    else:
        zi = np.atleast_1d(zi)
        (y_out, zf) = scipy.signal.lfilter(a, b, y, zi=zi.astype(y.dtype))
    if return_zf:
        return (y_out, zf)
    else:
        return y_out