"""Harmonic calculations for frequency representations"""
import warnings
import numpy as np
import scipy.interpolate
import scipy.signal
from ..util.exceptions import ParameterError
from ..util import is_unique
from numpy.typing import ArrayLike
from typing import Callable, Optional, Sequence
__all__ = ['salience', 'interp_harmonics', 'f0_harmonics']

def salience(S: np.ndarray, *, freqs: np.ndarray, harmonics: Sequence[float], weights: Optional[ArrayLike]=None, aggregate: Optional[Callable]=None, filter_peaks: bool=True, fill_value: float=np.nan, kind: str='linear', axis: int=-2) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    'Harmonic salience function.\n\n    Parameters\n    ----------\n    S : np.ndarray [shape=(..., d, n)]\n        input time frequency magnitude representation (e.g. STFT or CQT magnitudes).\n        Must be real-valued and non-negative.\n\n    freqs : np.ndarray, shape=(S.shape[axis]) or shape=S.shape\n        The frequency values corresponding to S\'s elements along the\n        chosen axis.\n\n        Frequencies can also be time-varying, e.g. as computed by\n        `reassigned_spectrogram`, in which case the shape should\n        match ``S``.\n\n    harmonics : list-like, non-negative\n        Harmonics to include in salience computation.  The first harmonic (1)\n        corresponds to ``S`` itself. Values less than one (e.g., 1/2) correspond\n        to sub-harmonics.\n\n    weights : list-like\n        The weight to apply to each harmonic in the summation. (default:\n        uniform weights). Must be the same length as ``harmonics``.\n\n    aggregate : function\n        aggregation function (default: `np.average`)\n\n        If ``aggregate=np.average``, then a weighted average is\n        computed per-harmonic according to the specified weights.\n        For all other aggregation functions, all harmonics\n        are treated equally.\n\n    filter_peaks : bool\n        If true, returns harmonic summation only on frequencies of peak\n        magnitude. Otherwise returns harmonic summation over the full spectrum.\n        Defaults to True.\n\n    fill_value : float\n        The value to fill non-peaks in the output representation. (default:\n        `np.nan`) Only used if ``filter_peaks == True``.\n\n    kind : str\n        Interpolation type for harmonic estimation.\n        See `scipy.interpolate.interp1d`.\n\n    axis : int\n        The axis along which to compute harmonics\n\n    Returns\n    -------\n    S_sal : np.ndarray\n        ``S_sal`` will have the same shape as ``S``, and measure\n        the overall harmonic energy at each frequency.\n\n    See Also\n    --------\n    interp_harmonics\n\n    Examples\n    --------\n    >>> y, sr = librosa.load(librosa.ex(\'trumpet\'), duration=3)\n    >>> S = np.abs(librosa.stft(y))\n    >>> freqs = librosa.fft_frequencies(sr=sr)\n    >>> harms = [1, 2, 3, 4]\n    >>> weights = [1.0, 0.5, 0.33, 0.25]\n    >>> S_sal = librosa.salience(S, freqs=freqs, harmonics=harms, weights=weights, fill_value=0)\n    >>> print(S_sal.shape)\n    (1025, 115)\n    >>> import matplotlib.pyplot as plt\n    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)\n    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),\n    ...                          sr=sr, y_axis=\'log\', x_axis=\'time\', ax=ax[0])\n    >>> ax[0].set(title=\'Magnitude spectrogram\')\n    >>> ax[0].label_outer()\n    >>> img = librosa.display.specshow(librosa.amplitude_to_db(S_sal,\n    ...                                                        ref=np.max),\n    ...                                sr=sr, y_axis=\'log\', x_axis=\'time\', ax=ax[1])\n    >>> ax[1].set(title=\'Salience spectrogram\')\n    >>> fig.colorbar(img, ax=ax, format="%+2.0f dB")\n    '
    if aggregate is None:
        aggregate = np.average
    if weights is None:
        weights = np.ones((len(harmonics),))
    else:
        weights = np.array(weights, dtype=float)
    S_harm = interp_harmonics(S, freqs=freqs, harmonics=harmonics, kind=kind, axis=axis)
    S_sal: np.ndarray
    if aggregate is np.average:
        S_sal = aggregate(S_harm, axis=axis - 1, weights=weights)
    else:
        S_sal = aggregate(S_harm, axis=axis - 1)
    if filter_peaks:
        S_peaks = scipy.signal.argrelmax(S, axis=axis)
        S_out = np.empty(S.shape)
        S_out.fill(fill_value)
        S_out[S_peaks] = S_sal[S_peaks]
        S_sal = S_out
    return S_sal

def interp_harmonics(x: np.ndarray, *, freqs: np.ndarray, harmonics: ArrayLike, kind: str='linear', fill_value: float=0, axis: int=-2) -> np.ndarray:
    if False:
        print('Hello World!')
    'Compute the energy at harmonics of time-frequency representation.\n\n    Given a frequency-based energy representation such as a spectrogram\n    or tempogram, this function computes the energy at the chosen harmonics\n    of the frequency axis.  (See examples below.)\n    The resulting harmonic array can then be used as input to a salience\n    computation.\n\n    Parameters\n    ----------\n    x : np.ndarray\n        The input energy\n    freqs : np.ndarray, shape=(x.shape[axis]) or shape=x.shape\n        The frequency values corresponding to x\'s elements along the\n        chosen axis.\n        Frequencies can also be time-varying, e.g. as computed by\n        `reassigned_spectrogram`, in which case the shape should\n        match ``x``.\n    harmonics : list-like, non-negative\n        Harmonics to compute as ``harmonics[i] * freqs``.\n        The first harmonic (1) corresponds to ``freqs``.\n        Values less than one (e.g., 1/2) correspond to sub-harmonics.\n    kind : str\n        Interpolation type.  See `scipy.interpolate.interp1d`.\n    fill_value : float\n        The value to fill when extrapolating beyond the observed\n        frequency range.\n    axis : int\n        The axis along which to compute harmonics\n\n    Returns\n    -------\n    x_harm : np.ndarray\n        ``x_harm[i]`` will have the same shape as ``x``, and measure\n        the energy at the ``harmonics[i]`` harmonic of each frequency.\n        A new dimension indexing harmonics will be inserted immediately\n        before ``axis``.\n\n    See Also\n    --------\n    scipy.interpolate.interp1d\n\n    Examples\n    --------\n    Estimate the harmonics of a time-averaged tempogram\n\n    >>> y, sr = librosa.load(librosa.ex(\'sweetwaltz\'))\n    >>> # Compute the time-varying tempogram and average over time\n    >>> tempi = np.mean(librosa.feature.tempogram(y=y, sr=sr), axis=1)\n    >>> # We\'ll measure the first five harmonics\n    >>> harmonics = [1, 2, 3, 4, 5]\n    >>> f_tempo = librosa.tempo_frequencies(len(tempi), sr=sr)\n    >>> # Build the harmonic tensor; we only have one axis here (tempo)\n    >>> t_harmonics = librosa.interp_harmonics(tempi, freqs=f_tempo, harmonics=harmonics, axis=0)\n    >>> print(t_harmonics.shape)\n    (5, 384)\n\n    >>> # And plot the results\n    >>> import matplotlib.pyplot as plt\n    >>> fig, ax = plt.subplots()\n    >>> librosa.display.specshow(t_harmonics, x_axis=\'tempo\', sr=sr, ax=ax)\n    >>> ax.set(yticks=np.arange(len(harmonics)),\n    ...        yticklabels=[\'{:.3g}\'.format(_) for _ in harmonics],\n    ...        ylabel=\'Harmonic\', xlabel=\'Tempo (BPM)\')\n\n    We can also compute frequency harmonics for spectrograms.\n    To calculate sub-harmonic energy, use values < 1.\n\n    >>> y, sr = librosa.load(librosa.ex(\'trumpet\'), duration=3)\n    >>> harmonics = [1./3, 1./2, 1, 2, 3, 4]\n    >>> S = np.abs(librosa.stft(y))\n    >>> fft_freqs = librosa.fft_frequencies(sr=sr)\n    >>> S_harm = librosa.interp_harmonics(S, freqs=fft_freqs, harmonics=harmonics, axis=0)\n    >>> print(S_harm.shape)\n    (6, 1025, 646)\n\n    >>> fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True)\n    >>> for i, _sh in enumerate(S_harm):\n    ...     img = librosa.display.specshow(librosa.amplitude_to_db(_sh,\n    ...                                                      ref=S.max()),\n    ...                              sr=sr, y_axis=\'log\', x_axis=\'time\',\n    ...                              ax=ax.flat[i])\n    ...     ax.flat[i].set(title=\'h={:.3g}\'.format(harmonics[i]))\n    ...     ax.flat[i].label_outer()\n    >>> fig.colorbar(img, ax=ax, format="%+2.f dB")\n    '
    if freqs.ndim == 1 and len(freqs) == x.shape[axis]:
        if not is_unique(freqs, axis=0):
            warnings.warn('Frequencies are not unique. This may produce incorrect harmonic interpolations.', stacklevel=2)
        f_interp = scipy.interpolate.interp1d(freqs, x, axis=axis, bounds_error=False, copy=False, kind=kind, fill_value=fill_value)
        f_out = np.multiply.outer(harmonics, freqs)
        return f_interp(f_out)
    elif freqs.shape == x.shape:
        if not np.all(is_unique(freqs, axis=axis)):
            warnings.warn('Frequencies are not unique. This may produce incorrect harmonic interpolations.', stacklevel=2)

        def _f_interp(_a, _b):
            if False:
                while True:
                    i = 10
            interp = scipy.interpolate.interp1d(_a, _b, bounds_error=False, copy=False, kind=kind, fill_value=fill_value)
            return interp(np.multiply.outer(_a, harmonics))
        xfunc = np.vectorize(_f_interp, signature='(f),(f)->(f,h)')
        return xfunc(freqs.swapaxes(axis, -1), x.swapaxes(axis, -1)).swapaxes(-2, axis).swapaxes(-1, axis - 1)
    else:
        raise ParameterError(f'freqs.shape={freqs.shape} is incompatible with input shape={x.shape}')

def f0_harmonics(x: np.ndarray, *, f0: np.ndarray, freqs: np.ndarray, harmonics: ArrayLike, kind: str='linear', fill_value: float=0, axis: int=-2) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    'Compute the energy at selected harmonics of a time-varying\n    fundamental frequency.\n\n    This function can be used to reduce a `frequency * time` representation\n    to a `harmonic * time` representation, effectively normalizing out for\n    the fundamental frequency.  The result can be used as a representation\n    of timbre when f0 corresponds to pitch, or as a representation of\n    rhythm when f0 corresponds to tempo.\n\n    This function differs from `interp_harmonics`, which computes the\n    harmonics of *all* frequencies.\n\n    Parameters\n    ----------\n    x : np.ndarray [shape=(..., frequencies, n)]\n        The input array (e.g., STFT magnitudes)\n    f0 : np.ndarray [shape=(..., n)]\n        The fundamental frequency (f0) of each frame in the input\n        Shape should match ``x.shape[-1]``\n    freqs : np.ndarray, shape=(x.shape[axis]) or shape=x.shape\n        The frequency values corresponding to X\'s elements along the\n        chosen axis.\n        Frequencies can also be time-varying, e.g. as computed by\n        `reassigned_spectrogram`, in which case the shape should\n        match ``x``.\n    harmonics : list-like, non-negative\n        Harmonics to compute as ``harmonics[i] * f0``\n        Values less than one (e.g., 1/2) correspond to sub-harmonics.\n    kind : str\n        Interpolation type.  See `scipy.interpolate.interp1d`.\n    fill_value : float\n        The value to fill when extrapolating beyond the observed\n        frequency range.\n    axis : int\n        The axis corresponding to frequency in ``x``\n\n    Returns\n    -------\n    f0_harm : np.ndarray [shape=(..., len(harmonics), n)]\n        Interpolated energy at each specified harmonic of the fundamental\n        frequency for each time step.\n\n    See Also\n    --------\n    interp_harmonics\n    librosa.feature.tempogram_ratio\n\n    Examples\n    --------\n    This example estimates the fundamental (f0), and then extracts the first\n    12 harmonics\n\n    >>> y, sr = librosa.load(librosa.ex(\'trumpet\'))\n    >>> f0, voicing, voicing_p = librosa.pyin(y=y, sr=sr, fmin=200, fmax=700)\n    >>> S = np.abs(librosa.stft(y))\n    >>> freqs = librosa.fft_frequencies(sr=sr)\n    >>> harmonics = np.arange(1, 13)\n    >>> f0_harm = librosa.f0_harmonics(S, freqs=freqs, f0=f0, harmonics=harmonics)\n\n    >>> import matplotlib.pyplot as plt\n    >>> fig, ax =plt.subplots(nrows=2, sharex=True)\n    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),\n    ...                          x_axis=\'time\', y_axis=\'log\', ax=ax[0])\n    >>> times = librosa.times_like(f0)\n    >>> for h in harmonics:\n    ...     ax[0].plot(times, h * f0, label=f"{h}*f0")\n    >>> ax[0].legend(ncols=4, loc=\'lower right\')\n    >>> ax[0].label_outer()\n    >>> librosa.display.specshow(librosa.amplitude_to_db(f0_harm, ref=np.max),\n    ...                          x_axis=\'time\', ax=ax[1])\n    >>> ax[1].set_yticks(harmonics-1)\n    >>> ax[1].set_yticklabels(harmonics)\n    >>> ax[1].set(ylabel=\'Harmonics\')\n    '
    result: np.ndarray
    if freqs.ndim == 1 and len(freqs) == x.shape[axis]:
        if not is_unique(freqs, axis=0):
            warnings.warn('Frequencies are not unique. This may produce incorrect harmonic interpolations.', stacklevel=2)
        idx = np.isfinite(freqs)

        def _f_interps(data, f):
            if False:
                print('Hello World!')
            interp = scipy.interpolate.interp1d(freqs[idx], data[idx], axis=0, bounds_error=False, copy=False, assume_sorted=False, kind=kind, fill_value=fill_value)
            return interp(f)
        xfunc = np.vectorize(_f_interps, signature='(f),(h)->(h)')
        result = xfunc(x.swapaxes(axis, -1), np.multiply.outer(f0, harmonics)).swapaxes(axis, -1)
    elif freqs.shape == x.shape:
        if not np.all(is_unique(freqs, axis=axis)):
            warnings.warn('Frequencies are not unique. This may produce incorrect harmonic interpolations.', stacklevel=2)

        def _f_interpd(data, frequencies, f):
            if False:
                i = 10
                return i + 15
            idx = np.isfinite(frequencies)
            interp = scipy.interpolate.interp1d(frequencies[idx], data[idx], axis=0, bounds_error=False, copy=False, assume_sorted=False, kind=kind, fill_value=fill_value)
            return interp(f)
        xfunc = np.vectorize(_f_interpd, signature='(f),(f),(h)->(h)')
        result = xfunc(x.swapaxes(axis, -1), freqs.swapaxes(axis, -1), np.multiply.outer(f0, harmonics)).swapaxes(axis, -1)
    else:
        raise ParameterError(f'freqs.shape={freqs.shape} is incompatible with input shape={x.shape}')
    return np.nan_to_num(result, copy=False, nan=fill_value)