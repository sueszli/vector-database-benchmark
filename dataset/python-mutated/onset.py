"""
Onset detection
===============
.. autosummary::
    :toctree: generated/

    onset_detect
    onset_backtrack
    onset_strength
    onset_strength_multi
"""
import numpy as np
import scipy
from ._cache import cache
from . import core
from . import util
from .util.exceptions import ParameterError
from .feature.spectral import melspectrogram
from typing import Any, Callable, Iterable, Optional, Union, Sequence
__all__ = ['onset_detect', 'onset_strength', 'onset_strength_multi', 'onset_backtrack']

def onset_detect(*, y: Optional[np.ndarray]=None, sr: float=22050, onset_envelope: Optional[np.ndarray]=None, hop_length: int=512, backtrack: bool=False, energy: Optional[np.ndarray]=None, units: str='frames', normalize: bool=True, **kwargs: Any) -> np.ndarray:
    if False:
        while True:
            i = 10
    "Locate note onset events by picking peaks in an onset strength envelope.\n\n    The `peak_pick` parameters were chosen by large-scale hyper-parameter\n    optimization over the dataset provided by [#]_.\n\n    .. [#] https://github.com/CPJKU/onset_db\n\n    Parameters\n    ----------\n    y : np.ndarray [shape=(n,)]\n        audio time series, must be monophonic\n\n    sr : number > 0 [scalar]\n        sampling rate of ``y``\n\n    onset_envelope : np.ndarray [shape=(m,)]\n        (optional) pre-computed onset strength envelope\n\n    hop_length : int > 0 [scalar]\n        hop length (in samples)\n\n    units : {'frames', 'samples', 'time'}\n        The units to encode detected onset events in.\n        By default, 'frames' are used.\n\n    backtrack : bool\n        If ``True``, detected onset events are backtracked to the nearest\n        preceding minimum of ``energy``.\n\n        This is primarily useful when using onsets as slice points for segmentation.\n\n    energy : np.ndarray [shape=(m,)] (optional)\n        An energy function to use for backtracking detected onset events.\n        If none is provided, then ``onset_envelope`` is used.\n\n    normalize : bool\n        If ``True`` (default), normalize the onset envelope to have minimum of 0 and\n        maximum of 1 prior to detection.  This is helpful for standardizing the\n        parameters of `librosa.util.peak_pick`.\n\n        Otherwise, the onset envelope is left unnormalized.\n\n    **kwargs : additional keyword arguments\n        Additional parameters for peak picking.\n\n        See `librosa.util.peak_pick` for details.\n\n    Returns\n    -------\n    onsets : np.ndarray [shape=(n_onsets,)]\n        estimated positions of detected onsets, in whichever units\n        are specified.  By default, frame indices.\n\n        .. note::\n            If no onset strength could be detected, onset_detect returns\n            an empty list.\n\n    Raises\n    ------\n    ParameterError\n        if neither ``y`` nor ``onsets`` are provided\n\n        or if ``units`` is not one of 'frames', 'samples', or 'time'\n\n    See Also\n    --------\n    onset_strength : compute onset strength per-frame\n    onset_backtrack : backtracking onset events\n    librosa.util.peak_pick : pick peaks from a time series\n\n    Examples\n    --------\n    Get onset times from a signal\n\n    >>> y, sr = librosa.load(librosa.ex('trumpet'))\n    >>> librosa.onset.onset_detect(y=y, sr=sr, units='time')\n    array([0.07 , 0.232, 0.395, 0.604, 0.743, 0.929, 1.045, 1.115,\n           1.416, 1.672, 1.881, 2.043, 2.206, 2.368, 2.554, 3.019])\n\n    Or use a pre-computed onset envelope\n\n    >>> o_env = librosa.onset.onset_strength(y=y, sr=sr)\n    >>> times = librosa.times_like(o_env, sr=sr)\n    >>> onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)\n\n    >>> import matplotlib.pyplot as plt\n    >>> D = np.abs(librosa.stft(y))\n    >>> fig, ax = plt.subplots(nrows=2, sharex=True)\n    >>> librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),\n    ...                          x_axis='time', y_axis='log', ax=ax[0])\n    >>> ax[0].set(title='Power spectrogram')\n    >>> ax[0].label_outer()\n    >>> ax[1].plot(times, o_env, label='Onset strength')\n    >>> ax[1].vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,\n    ...            linestyle='--', label='Onsets')\n    >>> ax[1].legend()\n    "
    if onset_envelope is None:
        if y is None:
            raise ParameterError('y or onset_envelope must be provided')
        onset_envelope = onset_strength(y=y, sr=sr, hop_length=hop_length)
    if normalize:
        onset_envelope = onset_envelope - np.min(onset_envelope)
        onset_envelope /= np.max(onset_envelope) + util.tiny(onset_envelope)
    assert onset_envelope is not None
    if not onset_envelope.any() or not np.all(np.isfinite(onset_envelope)):
        onsets = np.array([], dtype=int)
    else:
        kwargs.setdefault('pre_max', 0.03 * sr // hop_length)
        kwargs.setdefault('post_max', 0.0 * sr // hop_length + 1)
        kwargs.setdefault('pre_avg', 0.1 * sr // hop_length)
        kwargs.setdefault('post_avg', 0.1 * sr // hop_length + 1)
        kwargs.setdefault('wait', 0.03 * sr // hop_length)
        kwargs.setdefault('delta', 0.07)
        onsets = util.peak_pick(onset_envelope, **kwargs)
        if backtrack:
            if energy is None:
                energy = onset_envelope
            assert energy is not None
            onsets = onset_backtrack(onsets, energy)
    if units == 'frames':
        pass
    elif units == 'samples':
        onsets = core.frames_to_samples(onsets, hop_length=hop_length)
    elif units == 'time':
        onsets = core.frames_to_time(onsets, hop_length=hop_length, sr=sr)
    else:
        raise ParameterError(f'Invalid unit type: {units}')
    return onsets

def onset_strength(*, y: Optional[np.ndarray]=None, sr: float=22050, S: Optional[np.ndarray]=None, lag: int=1, max_size: int=1, ref: Optional[np.ndarray]=None, detrend: bool=False, center: bool=True, feature: Optional[Callable]=None, aggregate: Optional[Union[Callable, bool]]=None, **kwargs: Any) -> np.ndarray:
    if False:
        return 10
    'Compute a spectral flux onset strength envelope.\n\n    Onset strength at time ``t`` is determined by::\n\n        mean_f max(0, S[f, t] - ref[f, t - lag])\n\n    where ``ref`` is ``S`` after local max filtering along the frequency\n    axis [#]_.\n\n    By default, if a time series ``y`` is provided, S will be the\n    log-power Mel spectrogram.\n\n    .. [#] Böck, Sebastian, and Gerhard Widmer.\n           "Maximum filter vibrato suppression for onset detection."\n           16th International Conference on Digital Audio Effects,\n           Maynooth, Ireland. 2013.\n\n    Parameters\n    ----------\n    y : np.ndarray [shape=(..., n)]\n        audio time-series. Multi-channel is supported.\n\n    sr : number > 0 [scalar]\n        sampling rate of ``y``\n\n    S : np.ndarray [shape=(..., d, m)]\n        pre-computed (log-power) spectrogram\n\n    lag : int > 0\n        time lag for computing differences\n\n    max_size : int > 0\n        size (in frequency bins) of the local max filter.\n        set to `1` to disable filtering.\n\n    ref : None or np.ndarray [shape=(..., d, m)]\n        An optional pre-computed reference spectrum, of the same shape as ``S``.\n        If not provided, it will be computed from ``S``.\n        If provided, it will override any local max filtering governed by ``max_size``.\n\n    detrend : bool [scalar]\n        Filter the onset strength to remove the DC component\n\n    center : bool [scalar]\n        Shift the onset function by ``n_fft // (2 * hop_length)`` frames.\n        This corresponds to using a centered frame analysis in the short-time Fourier\n        transform.\n\n    feature : function\n        Function for computing time-series features, eg, scaled spectrograms.\n        By default, uses `librosa.feature.melspectrogram` with ``fmax=sr/2``\n\n    aggregate : function\n        Aggregation function to use when combining onsets\n        at different frequency bins.\n\n        Default: `np.mean`\n\n    **kwargs : additional keyword arguments\n        Additional parameters to ``feature()``, if ``S`` is not provided.\n\n    Returns\n    -------\n    onset_envelope : np.ndarray [shape=(..., m,)]\n        vector containing the onset strength envelope.\n        If the input contains multiple channels, then onset envelope is computed for each channel.\n\n    Raises\n    ------\n    ParameterError\n        if neither ``(y, sr)`` nor ``S`` are provided\n\n        or if ``lag`` or ``max_size`` are not positive integers\n\n    See Also\n    --------\n    onset_detect\n    onset_strength_multi\n\n    Examples\n    --------\n    First, load some audio and plot the spectrogram\n\n    >>> import matplotlib.pyplot as plt\n    >>> y, sr = librosa.load(librosa.ex(\'trumpet\'), duration=3)\n    >>> D = np.abs(librosa.stft(y))\n    >>> times = librosa.times_like(D)\n    >>> fig, ax = plt.subplots(nrows=2, sharex=True)\n    >>> librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),\n    ...                          y_axis=\'log\', x_axis=\'time\', ax=ax[0])\n    >>> ax[0].set(title=\'Power spectrogram\')\n    >>> ax[0].label_outer()\n\n    Construct a standard onset function\n\n    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr)\n    >>> ax[1].plot(times, 2 + onset_env / onset_env.max(), alpha=0.8,\n    ...            label=\'Mean (mel)\')\n\n    Median aggregation, and custom mel options\n\n    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr,\n    ...                                          aggregate=np.median,\n    ...                                          fmax=8000, n_mels=256)\n    >>> ax[1].plot(times, 1 + onset_env / onset_env.max(), alpha=0.8,\n    ...            label=\'Median (custom mel)\')\n\n    Constant-Q spectrogram instead of Mel\n\n    >>> C = np.abs(librosa.cqt(y=y, sr=sr))\n    >>> onset_env = librosa.onset.onset_strength(sr=sr, S=librosa.amplitude_to_db(C, ref=np.max))\n    >>> ax[1].plot(times, onset_env / onset_env.max(), alpha=0.8,\n    ...          label=\'Mean (CQT)\')\n    >>> ax[1].legend()\n    >>> ax[1].set(ylabel=\'Normalized strength\', yticks=[])\n    '
    if aggregate is False:
        raise ParameterError(f'aggregate parameter cannot be False when computing full-spectrum onset strength.')
    odf_all = onset_strength_multi(y=y, sr=sr, S=S, lag=lag, max_size=max_size, ref=ref, detrend=detrend, center=center, feature=feature, aggregate=aggregate, channels=None, **kwargs)
    return odf_all[..., 0, :]

def onset_backtrack(events: np.ndarray, energy: np.ndarray) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    'Backtrack detected onset events to the nearest preceding local\n    minimum of an energy function.\n\n    This function can be used to roll back the timing of detected onsets\n    from a detected peak amplitude to the preceding minimum.\n\n    This is most useful when using onsets to determine slice points for\n    segmentation, as described by [#]_.\n\n    .. [#] Jehan, Tristan.\n           "Creating music by listening"\n           Doctoral dissertation\n           Massachusetts Institute of Technology, 2005.\n\n    Parameters\n    ----------\n    events : np.ndarray, dtype=int\n        List of onset event frame indices, as computed by `onset_detect`\n    energy : np.ndarray, shape=(m,)\n        An energy function\n\n    Returns\n    -------\n    events_backtracked : np.ndarray, shape=events.shape\n        The input events matched to nearest preceding minima of ``energy``.\n\n    Examples\n    --------\n    Backtrack the events using the onset envelope\n\n    >>> y, sr = librosa.load(librosa.ex(\'trumpet\'), duration=3)\n    >>> oenv = librosa.onset.onset_strength(y=y, sr=sr)\n    >>> times = librosa.times_like(oenv)\n    >>> # Detect events without backtracking\n    >>> onset_raw = librosa.onset.onset_detect(onset_envelope=oenv,\n    ...                                        backtrack=False)\n    >>> onset_bt = librosa.onset.onset_backtrack(onset_raw, oenv)\n\n    Backtrack the events using the RMS values\n\n    >>> S = np.abs(librosa.stft(y=y))\n    >>> rms = librosa.feature.rms(S=S)\n    >>> onset_bt_rms = librosa.onset.onset_backtrack(onset_raw, rms[0])\n\n    Plot the results\n\n    >>> import matplotlib.pyplot as plt\n    >>> fig, ax = plt.subplots(nrows=3, sharex=True)\n    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),\n    ...                          y_axis=\'log\', x_axis=\'time\', ax=ax[0])\n    >>> ax[0].label_outer()\n    >>> ax[1].plot(times, oenv, label=\'Onset strength\')\n    >>> ax[1].vlines(librosa.frames_to_time(onset_raw), 0, oenv.max(), label=\'Raw onsets\')\n    >>> ax[1].vlines(librosa.frames_to_time(onset_bt), 0, oenv.max(), label=\'Backtracked\', color=\'r\')\n    >>> ax[1].legend()\n    >>> ax[1].label_outer()\n    >>> ax[2].plot(times, rms[0], label=\'RMS\')\n    >>> ax[2].vlines(librosa.frames_to_time(onset_bt_rms), 0, rms.max(), label=\'Backtracked (RMS)\', color=\'r\')\n    >>> ax[2].legend()\n    '
    minima = np.flatnonzero((energy[1:-1] <= energy[:-2]) & (energy[1:-1] < energy[2:]))
    minima = util.fix_frames(1 + minima, x_min=0)
    results: np.ndarray = minima[util.match_events(events, minima, right=False)]
    return results

@cache(level=30)
def onset_strength_multi(*, y: Optional[np.ndarray]=None, sr: float=22050, S: Optional[np.ndarray]=None, n_fft: int=2048, hop_length: int=512, lag: int=1, max_size: int=1, ref: Optional[np.ndarray]=None, detrend: bool=False, center: bool=True, feature: Optional[Callable]=None, aggregate: Optional[Union[Callable, bool]]=None, channels: Optional[Union[Sequence[int], Sequence[slice]]]=None, **kwargs: Any) -> np.ndarray:
    if False:
        return 10
    'Compute a spectral flux onset strength envelope across multiple channels.\n\n    Onset strength for channel ``i`` at time ``t`` is determined by::\n\n        mean_{f in channels[i]} max(0, S[f, t+1] - S[f, t])\n\n    Parameters\n    ----------\n    y : np.ndarray [shape=(..., n,)]\n        audio time-series. Multi-channel is supported.\n\n    sr : number > 0 [scalar]\n        sampling rate of ``y``\n\n    S : np.ndarray [shape=(..., d, m)]\n        pre-computed (log-power) spectrogram\n\n    n_fft : int > 0 [scalar]\n        FFT window size for use in ``feature()`` if ``S`` is not provided.\n\n    hop_length : int > 0 [scalar]\n        hop length for use in ``feature()`` if ``S`` is not provided.\n\n    lag : int > 0\n        time lag for computing differences\n\n    max_size : int > 0\n        size (in frequency bins) of the local max filter.\n        set to `1` to disable filtering.\n\n    ref : None or np.ndarray [shape=(d, m)]\n        An optional pre-computed reference spectrum, of the same shape as ``S``.\n        If not provided, it will be computed from ``S``.\n        If provided, it will override any local max filtering governed by ``max_size``.\n\n    detrend : bool [scalar]\n        Filter the onset strength to remove the DC component\n\n    center : bool [scalar]\n        Shift the onset function by ``n_fft // (2 * hop_length)`` frames.\n        This corresponds to using a centered frame analysis in the short-time Fourier\n        transform.\n\n    feature : function\n        Function for computing time-series features, eg, scaled spectrograms.\n        By default, uses `librosa.feature.melspectrogram` with ``fmax=sr/2``\n\n        Must support arguments: ``y, sr, n_fft, hop_length``\n\n    aggregate : function or False\n        Aggregation function to use when combining onsets\n        at different frequency bins.\n\n        If ``False``, then no aggregation is performed.\n\n        Default: `np.mean`\n\n    channels : list or None\n        Array of channel boundaries or slice objects.\n        If `None`, then a single channel is generated to span all bands.\n\n    **kwargs : additional keyword arguments\n        Additional parameters to ``feature()``, if ``S`` is not provided.\n\n    Returns\n    -------\n    onset_envelope : np.ndarray [shape=(..., n_channels, m)]\n        array containing the onset strength envelope for each specified channel\n\n    Raises\n    ------\n    ParameterError\n        if neither ``(y, sr)`` nor ``S`` are provided\n\n    See Also\n    --------\n    onset_strength\n\n    Notes\n    -----\n    This function caches at level 30.\n\n    Examples\n    --------\n    First, load some audio and plot the spectrogram\n\n    >>> import matplotlib.pyplot as plt\n    >>> y, sr = librosa.load(librosa.ex(\'choice\'), duration=5)\n    >>> D = np.abs(librosa.stft(y))\n    >>> fig, ax = plt.subplots(nrows=2, sharex=True)\n    >>> img1 = librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),\n    ...                          y_axis=\'log\', x_axis=\'time\', ax=ax[0])\n    >>> ax[0].set(title=\'Power spectrogram\')\n    >>> ax[0].label_outer()\n    >>> fig.colorbar(img1, ax=[ax[0]], format="%+2.f dB")\n\n    Construct a standard onset function over four sub-bands\n\n    >>> onset_subbands = librosa.onset.onset_strength_multi(y=y, sr=sr,\n    ...                                                     channels=[0, 32, 64, 96, 128])\n    >>> img2 = librosa.display.specshow(onset_subbands, x_axis=\'time\', ax=ax[1])\n    >>> ax[1].set(ylabel=\'Sub-bands\', title=\'Sub-band onset strength\')\n    >>> fig.colorbar(img2, ax=[ax[1]])\n    '
    if feature is None:
        feature = melspectrogram
        kwargs.setdefault('fmax', 0.5 * sr)
    if aggregate is None:
        aggregate = np.mean
    if not util.is_positive_int(lag):
        raise ParameterError(f'lag={lag} must be a positive integer')
    if not util.is_positive_int(max_size):
        raise ParameterError(f'max_size={max_size} must be a positive integer')
    if S is None:
        S = np.abs(feature(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, **kwargs))
        S = core.power_to_db(S)
    S = np.atleast_2d(S)
    if ref is None:
        if max_size == 1:
            ref = S
        else:
            ref = scipy.ndimage.maximum_filter1d(S, max_size, axis=-2)
    elif ref.shape != S.shape:
        raise ParameterError(f'Reference spectrum shape {ref.shape} must match input spectrum {S.shape}')
    onset_env = S[..., lag:] - ref[..., :-lag]
    onset_env = np.maximum(0.0, onset_env)
    pad = True
    if channels is None:
        channels = [slice(None)]
    else:
        pad = False
    if callable(aggregate):
        onset_env = util.sync(onset_env, channels, aggregate=aggregate, pad=pad, axis=-2)
    pad_width = lag
    if center:
        pad_width += n_fft // (2 * hop_length)
    padding = [(0, 0) for _ in onset_env.shape]
    padding[-1] = (int(pad_width), 0)
    onset_env = np.pad(onset_env, padding, mode='constant')
    if detrend:
        onset_env = scipy.signal.lfilter([1.0, -1.0], [1.0, -0.99], onset_env, axis=-1)
    if center:
        onset_env = onset_env[..., :S.shape[-1]]
    return onset_env