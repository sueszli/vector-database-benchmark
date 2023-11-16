"""Constant-Q transforms"""
import warnings
import numpy as np
from numba import jit
from . import audio
from .intervals import interval_frequencies
from .fft import get_fftlib
from .convert import cqt_frequencies, note_to_hz
from .spectrum import stft, istft
from .pitch import estimate_tuning
from .._cache import cache
from .. import filters
from .. import util
from ..util.exceptions import ParameterError
from numpy.typing import DTypeLike
from typing import Optional, Union, Collection, List
from .._typing import _WindowSpec, _PadMode, _FloatLike_co, _ensure_not_reachable
__all__ = ['cqt', 'hybrid_cqt', 'pseudo_cqt', 'icqt', 'griffinlim_cqt', 'vqt']

@cache(level=20)
def cqt(y: np.ndarray, *, sr: float=22050, hop_length: int=512, fmin: Optional[_FloatLike_co]=None, n_bins: int=84, bins_per_octave: int=12, tuning: Optional[float]=0.0, filter_scale: float=1, norm: Optional[float]=1, sparsity: float=0.01, window: _WindowSpec='hann', scale: bool=True, pad_mode: _PadMode='constant', res_type: Optional[str]='soxr_hq', dtype: Optional[DTypeLike]=None) -> np.ndarray:
    if False:
        return 10
    'Compute the constant-Q transform of an audio signal.\n\n    This implementation is based on the recursive sub-sampling method\n    described by [#]_.\n\n    .. [#] Schoerkhuber, Christian, and Anssi Klapuri.\n        "Constant-Q transform toolbox for music processing."\n        7th Sound and Music Computing Conference, Barcelona, Spain. 2010.\n\n    Parameters\n    ----------\n    y : np.ndarray [shape=(..., n)]\n        audio time series. Multi-channel is supported.\n\n    sr : number > 0 [scalar]\n        sampling rate of ``y``\n\n    hop_length : int > 0 [scalar]\n        number of samples between successive CQT columns.\n\n    fmin : float > 0 [scalar]\n        Minimum frequency. Defaults to `C1 ~= 32.70 Hz`\n\n    n_bins : int > 0 [scalar]\n        Number of frequency bins, starting at ``fmin``\n\n    bins_per_octave : int > 0 [scalar]\n        Number of bins per octave\n\n    tuning : None or float\n        Tuning offset in fractions of a bin.\n\n        If ``None``, tuning will be automatically estimated from the signal.\n\n        The minimum frequency of the resulting CQT will be modified to\n        ``fmin * 2**(tuning / bins_per_octave)``.\n\n    filter_scale : float > 0\n        Filter scale factor. Small values (<1) use shorter windows\n        for improved time resolution.\n\n    norm : {inf, -inf, 0, float > 0}\n        Type of norm to use for basis function normalization.\n        See `librosa.util.normalize`.\n\n    sparsity : float in [0, 1)\n        Sparsify the CQT basis by discarding up to ``sparsity``\n        fraction of the energy in each basis.\n\n        Set ``sparsity=0`` to disable sparsification.\n\n    window : str, tuple, number, or function\n        Window specification for the basis filters.\n        See `filters.get_window` for details.\n\n    scale : bool\n        If ``True``, scale the CQT response by square-root the length of\n        each channel\'s filter.  This is analogous to ``norm=\'ortho\'`` in FFT.\n\n        If ``False``, do not scale the CQT. This is analogous to\n        ``norm=None`` in FFT.\n\n    pad_mode : string\n        Padding mode for centered frame analysis.\n\n        See also: `librosa.stft` and `numpy.pad`.\n\n    res_type : string\n        The resampling mode for recursive downsampling.\n\n    dtype : np.dtype\n        The (complex) data type of the output array.  By default, this is inferred to match\n        the numerical precision of the input signal.\n\n    Returns\n    -------\n    CQT : np.ndarray [shape=(..., n_bins, t)]\n        Constant-Q value each frequency at each time.\n\n    See Also\n    --------\n    vqt\n    librosa.resample\n    librosa.util.normalize\n\n    Notes\n    -----\n    This function caches at level 20.\n\n    Examples\n    --------\n    Generate and plot a constant-Q power spectrum\n\n    >>> import matplotlib.pyplot as plt\n    >>> y, sr = librosa.load(librosa.ex(\'trumpet\'))\n    >>> C = np.abs(librosa.cqt(y, sr=sr))\n    >>> fig, ax = plt.subplots()\n    >>> img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),\n    ...                                sr=sr, x_axis=\'time\', y_axis=\'cqt_note\', ax=ax)\n    >>> ax.set_title(\'Constant-Q power spectrum\')\n    >>> fig.colorbar(img, ax=ax, format="%+2.0f dB")\n\n    Limit the frequency range\n\n    >>> C = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz(\'C2\'),\n    ...                 n_bins=60))\n    >>> C\n    array([[6.830e-04, 6.361e-04, ..., 7.362e-09, 9.102e-09],\n           [5.366e-04, 4.818e-04, ..., 8.953e-09, 1.067e-08],\n           ...,\n           [4.288e-02, 4.580e-01, ..., 1.529e-05, 5.572e-06],\n           [2.965e-03, 1.508e-01, ..., 8.965e-06, 1.455e-05]])\n\n    Using a higher frequency resolution\n\n    >>> C = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz(\'C2\'),\n    ...                 n_bins=60 * 2, bins_per_octave=12 * 2))\n    >>> C\n    array([[5.468e-04, 5.382e-04, ..., 5.911e-09, 6.105e-09],\n           [4.118e-04, 4.014e-04, ..., 7.788e-09, 8.160e-09],\n           ...,\n           [2.780e-03, 1.424e-01, ..., 4.225e-06, 2.388e-05],\n           [5.147e-02, 6.959e-02, ..., 1.694e-05, 5.811e-06]])\n    '
    return vqt(y=y, sr=sr, hop_length=hop_length, fmin=fmin, n_bins=n_bins, intervals='equal', gamma=0, bins_per_octave=bins_per_octave, tuning=tuning, filter_scale=filter_scale, norm=norm, sparsity=sparsity, window=window, scale=scale, pad_mode=pad_mode, res_type=res_type, dtype=dtype)

@cache(level=20)
def hybrid_cqt(y: np.ndarray, *, sr: float=22050, hop_length: int=512, fmin: Optional[_FloatLike_co]=None, n_bins: int=84, bins_per_octave: int=12, tuning: Optional[float]=0.0, filter_scale: float=1, norm: Optional[float]=1, sparsity: float=0.01, window: _WindowSpec='hann', scale: bool=True, pad_mode: _PadMode='constant', res_type: str='soxr_hq', dtype: Optional[DTypeLike]=None) -> np.ndarray:
    if False:
        return 10
    "Compute the hybrid constant-Q transform of an audio signal.\n\n    Here, the hybrid CQT uses the pseudo CQT for higher frequencies where\n    the hop_length is longer than half the filter length and the full CQT\n    for lower frequencies.\n\n    Parameters\n    ----------\n    y : np.ndarray [shape=(..., n)]\n        audio time series. Multi-channel is supported.\n\n    sr : number > 0 [scalar]\n        sampling rate of ``y``\n\n    hop_length : int > 0 [scalar]\n        number of samples between successive CQT columns.\n\n    fmin : float > 0 [scalar]\n        Minimum frequency. Defaults to `C1 ~= 32.70 Hz`\n\n    n_bins : int > 0 [scalar]\n        Number of frequency bins, starting at ``fmin``\n\n    bins_per_octave : int > 0 [scalar]\n        Number of bins per octave\n\n    tuning : None or float\n        Tuning offset in fractions of a bin.\n\n        If ``None``, tuning will be automatically estimated from the signal.\n\n        The minimum frequency of the resulting CQT will be modified to\n        ``fmin * 2**(tuning / bins_per_octave)``.\n\n    filter_scale : float > 0\n        Filter filter_scale factor. Larger values use longer windows.\n\n    norm : {inf, -inf, 0, float > 0}\n        Type of norm to use for basis function normalization.\n        See `librosa.util.normalize`.\n\n    sparsity : float in [0, 1)\n        Sparsify the CQT basis by discarding up to ``sparsity``\n        fraction of the energy in each basis.\n\n        Set ``sparsity=0`` to disable sparsification.\n\n    window : str, tuple, number, or function\n        Window specification for the basis filters.\n        See `filters.get_window` for details.\n\n    scale : bool\n        If ``True``, scale the CQT response by square-root the length of\n        each channel's filter.  This is analogous to ``norm='ortho'`` in FFT.\n\n        If ``False``, do not scale the CQT. This is analogous to\n        ``norm=None`` in FFT.\n\n    pad_mode : string\n        Padding mode for centered frame analysis.\n\n        See also: `librosa.stft` and `numpy.pad`.\n\n    res_type : string\n        Resampling mode.  See `librosa.cqt` for details.\n\n    dtype : np.dtype, optional\n        The complex dtype to use for computing the CQT.\n        By default, this is inferred to match the precision of\n        the input signal.\n\n    Returns\n    -------\n    CQT : np.ndarray [shape=(..., n_bins, t), dtype=np.float]\n        Constant-Q energy for each frequency at each time.\n\n    See Also\n    --------\n    cqt\n    pseudo_cqt\n\n    Notes\n    -----\n    This function caches at level 20.\n    "
    if fmin is None:
        fmin = note_to_hz('C1')
    if tuning is None:
        tuning = estimate_tuning(y=y, sr=sr, bins_per_octave=bins_per_octave)
    fmin = fmin * 2.0 ** (tuning / bins_per_octave)
    freqs = cqt_frequencies(n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
    if n_bins == 1:
        alpha = __et_relative_bw(bins_per_octave)
    else:
        alpha = filters._relative_bandwidth(freqs=freqs)
    (lengths, _) = filters.wavelet_lengths(freqs=freqs, sr=sr, filter_scale=filter_scale, window=window, alpha=alpha)
    pseudo_filters = 2.0 ** np.ceil(np.log2(lengths)) < 2 * hop_length
    n_bins_pseudo = int(np.sum(pseudo_filters))
    n_bins_full = n_bins - n_bins_pseudo
    cqt_resp = []
    if n_bins_pseudo > 0:
        fmin_pseudo = np.min(freqs[pseudo_filters])
        cqt_resp.append(pseudo_cqt(y, sr=sr, hop_length=hop_length, fmin=fmin_pseudo, n_bins=n_bins_pseudo, bins_per_octave=bins_per_octave, filter_scale=filter_scale, norm=norm, sparsity=sparsity, window=window, scale=scale, pad_mode=pad_mode, dtype=dtype))
    if n_bins_full > 0:
        cqt_resp.append(np.abs(cqt(y, sr=sr, hop_length=hop_length, fmin=fmin, n_bins=n_bins_full, bins_per_octave=bins_per_octave, filter_scale=filter_scale, norm=norm, sparsity=sparsity, window=window, scale=scale, pad_mode=pad_mode, res_type=res_type, dtype=dtype)))
    return __trim_stack(cqt_resp, n_bins, cqt_resp[-1].dtype)

@cache(level=20)
def pseudo_cqt(y: np.ndarray, *, sr: float=22050, hop_length: int=512, fmin: Optional[_FloatLike_co]=None, n_bins: int=84, bins_per_octave: int=12, tuning: Optional[float]=0.0, filter_scale: float=1, norm: Optional[float]=1, sparsity: float=0.01, window: _WindowSpec='hann', scale: bool=True, pad_mode: _PadMode='constant', dtype: Optional[DTypeLike]=None) -> np.ndarray:
    if False:
        return 10
    "Compute the pseudo constant-Q transform of an audio signal.\n\n    This uses a single fft size that is the smallest power of 2 that is greater\n    than or equal to the max of:\n\n        1. The longest CQT filter\n        2. 2x the hop_length\n\n    Parameters\n    ----------\n    y : np.ndarray [shape=(..., n)]\n        audio time series. Multi-channel is supported.\n\n    sr : number > 0 [scalar]\n        sampling rate of ``y``\n\n    hop_length : int > 0 [scalar]\n        number of samples between successive CQT columns.\n\n    fmin : float > 0 [scalar]\n        Minimum frequency. Defaults to `C1 ~= 32.70 Hz`\n\n    n_bins : int > 0 [scalar]\n        Number of frequency bins, starting at ``fmin``\n\n    bins_per_octave : int > 0 [scalar]\n        Number of bins per octave\n\n    tuning : None or float\n        Tuning offset in fractions of a bin.\n\n        If ``None``, tuning will be automatically estimated from the signal.\n\n        The minimum frequency of the resulting CQT will be modified to\n        ``fmin * 2**(tuning / bins_per_octave)``.\n\n    filter_scale : float > 0\n        Filter filter_scale factor. Larger values use longer windows.\n\n    norm : {inf, -inf, 0, float > 0}\n        Type of norm to use for basis function normalization.\n        See `librosa.util.normalize`.\n\n    sparsity : float in [0, 1)\n        Sparsify the CQT basis by discarding up to ``sparsity``\n        fraction of the energy in each basis.\n\n        Set ``sparsity=0`` to disable sparsification.\n\n    window : str, tuple, number, or function\n        Window specification for the basis filters.\n        See `filters.get_window` for details.\n\n    scale : bool\n        If ``True``, scale the CQT response by square-root the length of\n        each channel's filter.  This is analogous to ``norm='ortho'`` in FFT.\n\n        If ``False``, do not scale the CQT. This is analogous to\n        ``norm=None`` in FFT.\n\n    pad_mode : string\n        Padding mode for centered frame analysis.\n\n        See also: `librosa.stft` and `numpy.pad`.\n\n    dtype : np.dtype, optional\n        The complex data type for CQT calculations.\n        By default, this is inferred to match the precision of the input signal.\n\n    Returns\n    -------\n    CQT : np.ndarray [shape=(..., n_bins, t), dtype=np.float]\n        Pseudo Constant-Q energy for each frequency at each time.\n\n    Notes\n    -----\n    This function caches at level 20.\n    "
    if fmin is None:
        fmin = note_to_hz('C1')
    if tuning is None:
        tuning = estimate_tuning(y=y, sr=sr, bins_per_octave=bins_per_octave)
    if dtype is None:
        dtype = util.dtype_r2c(y.dtype)
    fmin = fmin * 2.0 ** (tuning / bins_per_octave)
    freqs = cqt_frequencies(fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
    if n_bins == 1:
        alpha = __et_relative_bw(bins_per_octave)
    else:
        alpha = filters._relative_bandwidth(freqs=freqs)
    (lengths, _) = filters.wavelet_lengths(freqs=freqs, sr=sr, window=window, filter_scale=filter_scale, alpha=alpha)
    (fft_basis, n_fft, _) = __vqt_filter_fft(sr, freqs, filter_scale, norm, sparsity, hop_length=hop_length, window=window, dtype=dtype, alpha=alpha)
    fft_basis = np.abs(fft_basis)
    C: np.ndarray = __cqt_response(y, n_fft, hop_length, fft_basis, pad_mode, window='hann', dtype=dtype, phase=False)
    if scale:
        C /= np.sqrt(n_fft)
    else:
        lengths = util.expand_to(lengths, ndim=C.ndim, axes=-2)
        C *= np.sqrt(lengths / n_fft)
    return C

@cache(level=40)
def icqt(C: np.ndarray, *, sr: float=22050, hop_length: int=512, fmin: Optional[_FloatLike_co]=None, bins_per_octave: int=12, tuning: float=0.0, filter_scale: float=1, norm: Optional[float]=1, sparsity: float=0.01, window: _WindowSpec='hann', scale: bool=True, length: Optional[int]=None, res_type: str='soxr_hq', dtype: Optional[DTypeLike]=None) -> np.ndarray:
    if False:
        while True:
            i = 10
    "Compute the inverse constant-Q transform.\n\n    Given a constant-Q transform representation ``C`` of an audio signal ``y``,\n    this function produces an approximation ``y_hat``.\n\n    Parameters\n    ----------\n    C : np.ndarray, [shape=(..., n_bins, n_frames)]\n        Constant-Q representation as produced by `cqt`\n\n    sr : number > 0 [scalar]\n        sampling rate of the signal\n\n    hop_length : int > 0 [scalar]\n        number of samples between successive frames\n\n    fmin : float > 0 [scalar]\n        Minimum frequency. Defaults to `C1 ~= 32.70 Hz`\n\n    bins_per_octave : int > 0 [scalar]\n        Number of bins per octave\n\n    tuning : float [scalar]\n        Tuning offset in fractions of a bin.\n\n        The minimum frequency of the CQT will be modified to\n        ``fmin * 2**(tuning / bins_per_octave)``.\n\n    filter_scale : float > 0 [scalar]\n        Filter scale factor. Small values (<1) use shorter windows\n        for improved time resolution.\n\n    norm : {inf, -inf, 0, float > 0}\n        Type of norm to use for basis function normalization.\n        See `librosa.util.normalize`.\n\n    sparsity : float in [0, 1)\n        Sparsify the CQT basis by discarding up to ``sparsity``\n        fraction of the energy in each basis.\n\n        Set ``sparsity=0`` to disable sparsification.\n\n    window : str, tuple, number, or function\n        Window specification for the basis filters.\n        See `filters.get_window` for details.\n\n    scale : bool\n        If ``True``, scale the CQT response by square-root the length\n        of each channel's filter. This is analogous to ``norm='ortho'`` in FFT.\n\n        If ``False``, do not scale the CQT. This is analogous to ``norm=None``\n        in FFT.\n\n    length : int > 0, optional\n        If provided, the output ``y`` is zero-padded or clipped to exactly\n        ``length`` samples.\n\n    res_type : string\n        Resampling mode.\n        See `librosa.resample` for supported modes.\n\n    dtype : numeric type\n        Real numeric type for ``y``.  Default is inferred to match the numerical\n        precision of the input CQT.\n\n    Returns\n    -------\n    y : np.ndarray, [shape=(..., n_samples), dtype=np.float]\n        Audio time-series reconstructed from the CQT representation.\n\n    See Also\n    --------\n    cqt\n    librosa.resample\n\n    Notes\n    -----\n    This function caches at level 40.\n\n    Examples\n    --------\n    Using default parameters\n\n    >>> y, sr = librosa.load(librosa.ex('trumpet'))\n    >>> C = librosa.cqt(y=y, sr=sr)\n    >>> y_hat = librosa.icqt(C=C, sr=sr)\n\n    Or with a different hop length and frequency resolution:\n\n    >>> hop_length = 256\n    >>> bins_per_octave = 12 * 3\n    >>> C = librosa.cqt(y=y, sr=sr, hop_length=256, n_bins=7*bins_per_octave,\n    ...                 bins_per_octave=bins_per_octave)\n    >>> y_hat = librosa.icqt(C=C, sr=sr, hop_length=hop_length,\n    ...                 bins_per_octave=bins_per_octave)\n    "
    if fmin is None:
        fmin = note_to_hz('C1')
    fmin = fmin * 2.0 ** (tuning / bins_per_octave)
    n_bins = C.shape[-2]
    n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
    freqs = cqt_frequencies(fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
    if n_bins == 1:
        alpha = __et_relative_bw(bins_per_octave)
    else:
        alpha = filters._relative_bandwidth(freqs=freqs)
    (lengths, f_cutoff) = filters.wavelet_lengths(freqs=freqs, sr=sr, window=window, filter_scale=filter_scale, alpha=alpha)
    if length is not None:
        n_frames = int(np.ceil((length + max(lengths)) / hop_length))
        C = C[..., :n_frames]
    C_scale = np.sqrt(lengths)
    y: Optional[np.ndarray] = None
    srs = [sr]
    hops = [hop_length]
    for i in range(n_octaves - 1):
        if hops[0] % 2 == 0:
            srs.insert(0, srs[0] * 0.5)
            hops.insert(0, hops[0] // 2)
        else:
            srs.insert(0, srs[0])
            hops.insert(0, hops[0])
    for (i, (my_sr, my_hop)) in enumerate(zip(srs, hops)):
        n_filters = min(bins_per_octave, n_bins - bins_per_octave * i)
        sl = slice(bins_per_octave * i, bins_per_octave * i + n_filters)
        (fft_basis, n_fft, _) = __vqt_filter_fft(my_sr, freqs[sl], filter_scale, norm, sparsity, window=window, dtype=dtype, alpha=alpha[sl])
        inv_basis = fft_basis.H.todense()
        freq_power = 1 / np.sum(util.abs2(np.asarray(inv_basis)), axis=0)
        freq_power *= n_fft / lengths[sl]
        if scale:
            D_oct = np.einsum('fc,c,c,...ct->...ft', inv_basis, C_scale[sl], freq_power, C[..., sl, :], optimize=True)
        else:
            D_oct = np.einsum('fc,c,...ct->...ft', inv_basis, freq_power, C[..., sl, :], optimize=True)
        y_oct = istft(D_oct, window='ones', hop_length=my_hop, dtype=dtype)
        y_oct = audio.resample(y_oct, orig_sr=1, target_sr=sr // my_sr, res_type=res_type, scale=False, fix=False)
        if y is None:
            y = y_oct
        else:
            y[..., :y_oct.shape[-1]] += y_oct
    assert y is not None
    if length:
        y = util.fix_length(y, size=length)
    return y

@cache(level=20)
def vqt(y: np.ndarray, *, sr: float=22050, hop_length: int=512, fmin: Optional[_FloatLike_co]=None, n_bins: int=84, intervals: Union[str, Collection[float]]='equal', gamma: Optional[float]=None, bins_per_octave: int=12, tuning: Optional[float]=0.0, filter_scale: float=1, norm: Optional[float]=1, sparsity: float=0.01, window: _WindowSpec='hann', scale: bool=True, pad_mode: _PadMode='constant', res_type: Optional[str]='soxr_hq', dtype: Optional[DTypeLike]=None) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Compute the variable-Q transform of an audio signal.\n\n    This implementation is based on the recursive sub-sampling method\n    described by [#]_.\n\n    .. [#] Schörkhuber, Christian, Anssi Klapuri, Nicki Holighaus, and Monika Dörfler.\n        "A Matlab toolbox for efficient perfect reconstruction time-frequency\n        transforms with log-frequency resolution."\n        In Audio Engineering Society Conference: 53rd International Conference: Semantic Audio.\n        Audio Engineering Society, 2014.\n\n    Parameters\n    ----------\n    y : np.ndarray [shape=(..., n)]\n        audio time series. Multi-channel is supported.\n\n    sr : number > 0 [scalar]\n        sampling rate of ``y``\n\n    hop_length : int > 0 [scalar]\n        number of samples between successive VQT columns.\n\n    fmin : float > 0 [scalar]\n        Minimum frequency. Defaults to `C1 ~= 32.70 Hz`\n\n    n_bins : int > 0 [scalar]\n        Number of frequency bins, starting at ``fmin``\n\n    intervals : str or array of floats in [1, 2)\n        Either a string specification for an interval set, e.g.,\n        `\'equal\'`, `\'pythagorean\'`, `\'ji3\'`, etc. or an array of\n        intervals expressed as numbers between 1 and 2.\n        .. see also:: librosa.interval_frequencies\n\n    gamma : number > 0 [scalar]\n        Bandwidth offset for determining filter lengths.\n\n        If ``gamma=0``, produces the constant-Q transform.\n\n        If \'gamma=None\', gamma will be calculated such that filter bandwidths are equal to a\n        constant fraction of the equivalent rectangular bandwidths (ERB). This is accomplished\n        by solving for the gamma which gives::\n\n            B_k = alpha * f_k + gamma = C * ERB(f_k),\n\n        where ``B_k`` is the bandwidth of filter ``k`` with center frequency ``f_k``, alpha\n        is the inverse of what would be the constant Q-factor, and ``C = alpha / 0.108`` is the\n        constant fraction across all filters.\n\n        Here we use ``ERB(f_k) = 24.7 + 0.108 * f_k``, the best-fit curve derived\n        from experimental data in [#]_.\n\n        .. [#] Glasberg, Brian R., and Brian CJ Moore.\n            "Derivation of auditory filter shapes from notched-noise data."\n            Hearing research 47.1-2 (1990): 103-138.\n\n    bins_per_octave : int > 0 [scalar]\n        Number of bins per octave\n\n    tuning : None or float\n        Tuning offset in fractions of a bin.\n\n        If ``None``, tuning will be automatically estimated from the signal.\n\n        The minimum frequency of the resulting VQT will be modified to\n        ``fmin * 2**(tuning / bins_per_octave)``.\n\n    filter_scale : float > 0\n        Filter scale factor. Small values (<1) use shorter windows\n        for improved time resolution.\n\n    norm : {inf, -inf, 0, float > 0}\n        Type of norm to use for basis function normalization.\n        See `librosa.util.normalize`.\n\n    sparsity : float in [0, 1)\n        Sparsify the VQT basis by discarding up to ``sparsity``\n        fraction of the energy in each basis.\n\n        Set ``sparsity=0`` to disable sparsification.\n\n    window : str, tuple, number, or function\n        Window specification for the basis filters.\n        See `filters.get_window` for details.\n\n    scale : bool\n        If ``True``, scale the VQT response by square-root the length of\n        each channel\'s filter.  This is analogous to ``norm=\'ortho\'`` in FFT.\n\n        If ``False``, do not scale the VQT. This is analogous to\n        ``norm=None`` in FFT.\n\n    pad_mode : string\n        Padding mode for centered frame analysis.\n\n        See also: `librosa.stft` and `numpy.pad`.\n\n    res_type : string\n        The resampling mode for recursive downsampling.\n\n    dtype : np.dtype\n        The dtype of the output array.  By default, this is inferred to match the\n        numerical precision of the input signal.\n\n    Returns\n    -------\n    VQT : np.ndarray [shape=(..., n_bins, t), dtype=np.complex]\n        Variable-Q value each frequency at each time.\n\n    See Also\n    --------\n    cqt\n\n    Notes\n    -----\n    This function caches at level 20.\n\n    Examples\n    --------\n    Generate and plot a variable-Q power spectrum\n\n    >>> import matplotlib.pyplot as plt\n    >>> y, sr = librosa.load(librosa.ex(\'choice\'), duration=5)\n    >>> C = np.abs(librosa.cqt(y, sr=sr))\n    >>> V = np.abs(librosa.vqt(y, sr=sr))\n    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)\n    >>> librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),\n    ...                          sr=sr, x_axis=\'time\', y_axis=\'cqt_note\', ax=ax[0])\n    >>> ax[0].set(title=\'Constant-Q power spectrum\', xlabel=None)\n    >>> ax[0].label_outer()\n    >>> img = librosa.display.specshow(librosa.amplitude_to_db(V, ref=np.max),\n    ...                                sr=sr, x_axis=\'time\', y_axis=\'cqt_note\', ax=ax[1])\n    >>> ax[1].set_title(\'Variable-Q power spectrum\')\n    >>> fig.colorbar(img, ax=ax, format="%+2.0f dB")\n    '
    if not isinstance(intervals, str):
        bins_per_octave = len(intervals)
    n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
    n_filters = min(bins_per_octave, n_bins)
    if fmin is None:
        fmin = note_to_hz('C1')
    if tuning is None:
        tuning = estimate_tuning(y=y, sr=sr, bins_per_octave=bins_per_octave)
    if dtype is None:
        dtype = util.dtype_r2c(y.dtype)
    fmin = fmin * 2.0 ** (tuning / bins_per_octave)
    freqs = interval_frequencies(n_bins=n_bins, fmin=fmin, intervals=intervals, bins_per_octave=bins_per_octave, sort=True)
    freqs_top = freqs[-bins_per_octave:]
    fmax_t: float = np.max(freqs_top)
    if n_bins == 1:
        alpha = __et_relative_bw(bins_per_octave)
    else:
        alpha = filters._relative_bandwidth(freqs=freqs)
    (lengths, filter_cutoff) = filters.wavelet_lengths(freqs=freqs, sr=sr, window=window, filter_scale=filter_scale, gamma=gamma, alpha=alpha)
    nyquist = sr / 2.0
    if filter_cutoff > nyquist:
        raise ParameterError(f'Wavelet basis with max frequency={fmax_t} would exceed the Nyquist frequency={nyquist}. Try reducing the number of frequency bins.')
    if res_type is None:
        warnings.warn('Support for VQT with res_type=None is deprecated in librosa 0.10\nand will be removed in version 1.0.', category=FutureWarning, stacklevel=2)
        res_type = 'soxr_hq'
    (y, sr, hop_length) = __early_downsample(y, sr, hop_length, res_type, n_octaves, nyquist, filter_cutoff, scale)
    vqt_resp = []
    (my_y, my_sr, my_hop) = (y, sr, hop_length)
    for i in range(n_octaves):
        if i == 0:
            sl = slice(-n_filters, None)
        else:
            sl = slice(-n_filters * (i + 1), -n_filters * i)
        freqs_oct = freqs[sl]
        alpha_oct = alpha[sl]
        (fft_basis, n_fft, _) = __vqt_filter_fft(my_sr, freqs_oct, filter_scale, norm, sparsity, window=window, gamma=gamma, dtype=dtype, alpha=alpha_oct)
        fft_basis[:] *= np.sqrt(sr / my_sr)
        vqt_resp.append(__cqt_response(my_y, n_fft, my_hop, fft_basis, pad_mode, dtype=dtype))
        if my_hop % 2 == 0:
            my_hop //= 2
            my_sr /= 2.0
            my_y = audio.resample(my_y, orig_sr=2, target_sr=1, res_type=res_type, scale=True)
    V = __trim_stack(vqt_resp, n_bins, dtype)
    if scale:
        (lengths, _) = filters.wavelet_lengths(freqs=freqs, sr=sr, window=window, filter_scale=filter_scale, gamma=gamma, alpha=alpha)
        lengths = util.expand_to(lengths, ndim=V.ndim, axes=-2)
        V /= np.sqrt(lengths)
    return V

@cache(level=10)
def __vqt_filter_fft(sr, freqs, filter_scale, norm, sparsity, hop_length=None, window='hann', gamma=0.0, dtype=np.complex64, alpha=None):
    if False:
        while True:
            i = 10
    'Generate the frequency domain variable-Q filter basis.'
    (basis, lengths) = filters.wavelet(freqs=freqs, sr=sr, filter_scale=filter_scale, norm=norm, pad_fft=True, window=window, gamma=gamma, alpha=alpha)
    n_fft = basis.shape[1]
    if hop_length is not None and n_fft < 2.0 ** (1 + np.ceil(np.log2(hop_length))):
        n_fft = int(2.0 ** (1 + np.ceil(np.log2(hop_length))))
    basis *= lengths[:, np.newaxis] / float(n_fft)
    fft = get_fftlib()
    fft_basis = fft.fft(basis, n=n_fft, axis=1)[:, :n_fft // 2 + 1]
    fft_basis = util.sparsify_rows(fft_basis, quantile=sparsity, dtype=dtype)
    return (fft_basis, n_fft, lengths)

def __trim_stack(cqt_resp: List[np.ndarray], n_bins: int, dtype: DTypeLike) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    'Trim and stack a collection of CQT responses'
    max_col = min((c_i.shape[-1] for c_i in cqt_resp))
    shape = list(cqt_resp[0].shape)
    shape[-2] = n_bins
    shape[-1] = max_col
    cqt_out = np.empty(shape, dtype=dtype, order='F')
    end = n_bins
    for c_i in cqt_resp:
        n_oct = c_i.shape[-2]
        if end < n_oct:
            cqt_out[..., :end, :] = c_i[..., -end:, :max_col]
        else:
            cqt_out[..., end - n_oct:end, :] = c_i[..., :max_col]
        end -= n_oct
    return cqt_out

def __cqt_response(y, n_fft, hop_length, fft_basis, mode, window='ones', phase=True, dtype=None):
    if False:
        i = 10
        return i + 15
    'Compute the filter response with a target STFT hop.'
    D = stft(y, n_fft=n_fft, hop_length=hop_length, window=window, pad_mode=mode, dtype=dtype)
    if not phase:
        D = np.abs(D)
    Dr = D.reshape((-1, D.shape[-2], D.shape[-1]))
    output_flat = np.empty((Dr.shape[0], fft_basis.shape[0], Dr.shape[-1]), dtype=D.dtype)
    for i in range(Dr.shape[0]):
        output_flat[i] = fft_basis.dot(Dr[i])
    shape = list(D.shape)
    shape[-2] = fft_basis.shape[0]
    return output_flat.reshape(shape)

def __early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves):
    if False:
        for i in range(10):
            print('nop')
    'Compute the number of early downsampling operations'
    downsample_count1 = max(0, int(np.ceil(np.log2(nyquist / filter_cutoff)) - 1) - 1)
    num_twos = __num_two_factors(hop_length)
    downsample_count2 = max(0, num_twos - n_octaves + 1)
    return min(downsample_count1, downsample_count2)

def __early_downsample(y, sr, hop_length, res_type, n_octaves, nyquist, filter_cutoff, scale):
    if False:
        while True:
            i = 10
    'Perform early downsampling on an audio signal, if it applies.'
    downsample_count = __early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves)
    if downsample_count > 0:
        downsample_factor = 2 ** downsample_count
        hop_length //= downsample_factor
        if y.shape[-1] < downsample_factor:
            raise ParameterError(f'Input signal length={len(y):d} is too short for {n_octaves:d}-octave CQT')
        new_sr = sr / float(downsample_factor)
        y = audio.resample(y, orig_sr=downsample_factor, target_sr=1, res_type=res_type, scale=True)
        if not scale:
            y *= np.sqrt(downsample_factor)
        sr = new_sr
    return (y, sr, hop_length)

@jit(nopython=True, cache=True)
def __num_two_factors(x):
    if False:
        for i in range(10):
            print('nop')
    'Return how many times integer x can be evenly divided by 2.\n\n    Returns 0 for non-positive integers.\n    '
    if x <= 0:
        return 0
    num_twos = 0
    while x % 2 == 0:
        num_twos += 1
        x //= 2
    return num_twos

def griffinlim_cqt(C: np.ndarray, *, n_iter: int=32, sr: float=22050, hop_length: int=512, fmin: Optional[_FloatLike_co]=None, bins_per_octave: int=12, tuning: float=0.0, filter_scale: float=1, norm: Optional[float]=1, sparsity: float=0.01, window: _WindowSpec='hann', scale: bool=True, pad_mode: _PadMode='constant', res_type: str='soxr_hq', dtype: Optional[DTypeLike]=None, length: Optional[int]=None, momentum: float=0.99, init: Optional[str]='random', random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]]=None) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    'Approximate constant-Q magnitude spectrogram inversion using the "fast" Griffin-Lim\n    algorithm.\n\n    Given the magnitude of a constant-Q spectrogram (``C``), the algorithm randomly initializes\n    phase estimates, and then alternates forward- and inverse-CQT operations. [#]_\n\n    This implementation is based on the (fast) Griffin-Lim method for Short-time Fourier Transforms, [#]_\n    but adapted for use with constant-Q spectrograms.\n\n    .. [#] D. W. Griffin and J. S. Lim,\n        "Signal estimation from modified short-time Fourier transform,"\n        IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.\n\n    .. [#] Perraudin, N., Balazs, P., & Søndergaard, P. L.\n        "A fast Griffin-Lim algorithm,"\n        IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (pp. 1-4),\n        Oct. 2013.\n\n    Parameters\n    ----------\n    C : np.ndarray [shape=(..., n_bins, n_frames)]\n        The constant-Q magnitude spectrogram\n\n    n_iter : int > 0\n        The number of iterations to run\n\n    sr : number > 0\n        Audio sampling rate\n\n    hop_length : int > 0\n        The hop length of the CQT\n\n    fmin : number > 0\n        Minimum frequency for the CQT.\n\n        If not provided, it defaults to `C1`.\n\n    bins_per_octave : int > 0\n        Number of bins per octave\n\n    tuning : float\n        Tuning deviation from A440, in fractions of a bin\n\n    filter_scale : float > 0\n        Filter scale factor. Small values (<1) use shorter windows\n        for improved time resolution.\n\n    norm : {inf, -inf, 0, float > 0}\n        Type of norm to use for basis function normalization.\n        See `librosa.util.normalize`.\n\n    sparsity : float in [0, 1)\n        Sparsify the CQT basis by discarding up to ``sparsity``\n        fraction of the energy in each basis.\n\n        Set ``sparsity=0`` to disable sparsification.\n\n    window : str, tuple, or function\n        Window specification for the basis filters.\n        See `filters.get_window` for details.\n\n    scale : bool\n        If ``True``, scale the CQT response by square-root the length\n        of each channel\'s filter.  This is analogous to ``norm=\'ortho\'``\n        in FFT.\n\n        If ``False``, do not scale the CQT. This is analogous to ``norm=None``\n        in FFT.\n\n    pad_mode : string\n        Padding mode for centered frame analysis.\n\n        See also: `librosa.stft` and `numpy.pad`.\n\n    res_type : string\n        The resampling mode for recursive downsampling.\n\n        See ``librosa.resample`` for a list of available options.\n\n    dtype : numeric type\n        Real numeric type for ``y``.  Default is inferred to match the precision\n        of the input CQT.\n\n    length : int > 0, optional\n        If provided, the output ``y`` is zero-padded or clipped to exactly\n        ``length`` samples.\n\n    momentum : float > 0\n        The momentum parameter for fast Griffin-Lim.\n        Setting this to 0 recovers the original Griffin-Lim method.\n        Values near 1 can lead to faster convergence, but above 1 may not converge.\n\n    init : None or \'random\' [default]\n        If \'random\' (the default), then phase values are initialized randomly\n        according to ``random_state``.  This is recommended when the input ``C`` is\n        a magnitude spectrogram with no initial phase estimates.\n\n        If ``None``, then the phase is initialized from ``C``.  This is useful when\n        an initial guess for phase can be provided, or when you want to resume\n        Griffin-Lim from a previous output.\n\n    random_state : None, int, np.random.RandomState, or np.random.Generator\n        If int, random_state is the seed used by the random number generator\n        for phase initialization.\n\n        If `np.random.RandomState` or `np.random.Generator` instance, the random number generator itself.\n\n        If ``None``, defaults to the `np.random.default_rng()` object.\n\n    Returns\n    -------\n    y : np.ndarray [shape=(..., n)]\n        time-domain signal reconstructed from ``C``\n\n    See Also\n    --------\n    cqt\n    icqt\n    griffinlim\n    filters.get_window\n    resample\n\n    Examples\n    --------\n    A basis CQT inverse example\n\n    >>> y, sr = librosa.load(librosa.ex(\'trumpet\', hq=True), sr=None)\n    >>> # Get the CQT magnitude, 7 octaves at 36 bins per octave\n    >>> C = np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=36, n_bins=7*36))\n    >>> # Invert using Griffin-Lim\n    >>> y_inv = librosa.griffinlim_cqt(C, sr=sr, bins_per_octave=36)\n    >>> # And invert without estimating phase\n    >>> y_icqt = librosa.icqt(C, sr=sr, bins_per_octave=36)\n\n    Wave-plot the results\n\n    >>> import matplotlib.pyplot as plt\n    >>> fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)\n    >>> librosa.display.waveshow(y, sr=sr, color=\'b\', ax=ax[0])\n    >>> ax[0].set(title=\'Original\', xlabel=None)\n    >>> ax[0].label_outer()\n    >>> librosa.display.waveshow(y_inv, sr=sr, color=\'g\', ax=ax[1])\n    >>> ax[1].set(title=\'Griffin-Lim reconstruction\', xlabel=None)\n    >>> ax[1].label_outer()\n    >>> librosa.display.waveshow(y_icqt, sr=sr, color=\'r\', ax=ax[2])\n    >>> ax[2].set(title=\'Magnitude-only icqt reconstruction\')\n    '
    if fmin is None:
        fmin = note_to_hz('C1')
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.RandomState(seed=random_state)
    elif isinstance(random_state, (np.random.RandomState, np.random.Generator)):
        rng = random_state
    else:
        _ensure_not_reachable(random_state)
        raise ParameterError(f'Unsupported random_state={random_state!r}')
    if momentum > 1:
        warnings.warn(f'Griffin-Lim with momentum={momentum} > 1 can be unstable. Proceed with caution!', stacklevel=2)
    elif momentum < 0:
        raise ParameterError(f'griffinlim_cqt() called with momentum={momentum} < 0')
    angles = np.empty(C.shape, dtype=np.complex64)
    eps = util.tiny(angles)
    if init == 'random':
        angles[:] = util.phasor(2 * np.pi * rng.random(size=C.shape))
    elif init is None:
        angles[:] = 1.0
    else:
        raise ParameterError(f"init={init} must either None or 'random'")
    rebuilt: np.ndarray = np.array(0.0)
    for _ in range(n_iter):
        tprev = rebuilt
        inverse = icqt(C * angles, sr=sr, hop_length=hop_length, bins_per_octave=bins_per_octave, fmin=fmin, tuning=tuning, filter_scale=filter_scale, window=window, length=length, res_type=res_type, norm=norm, scale=scale, sparsity=sparsity, dtype=dtype)
        rebuilt = cqt(inverse, sr=sr, bins_per_octave=bins_per_octave, n_bins=C.shape[-2], hop_length=hop_length, fmin=fmin, tuning=tuning, filter_scale=filter_scale, window=window, norm=norm, scale=scale, sparsity=sparsity, pad_mode=pad_mode, res_type=res_type)
        angles[:] = rebuilt - momentum / (1 + momentum) * tprev
        angles[:] /= np.abs(angles) + eps
    return icqt(C * angles, sr=sr, hop_length=hop_length, bins_per_octave=bins_per_octave, tuning=tuning, filter_scale=filter_scale, fmin=fmin, window=window, length=length, res_type=res_type, norm=norm, scale=scale, sparsity=sparsity, dtype=dtype)

def __et_relative_bw(bins_per_octave: int) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Compute the relative bandwidth coefficient for equal\n    (geometric) freuqency spacing and a give number of bins\n    per octave.\n\n    This is a special case of the more general `relative_bandwidth`\n    calculation that can be used when only a single basis frequency\n    is used.\n\n    Parameters\n    ----------\n    bins_per_octave : int\n\n    Returns\n    -------\n    alpha : np.ndarray > 0\n        Value is cast up to a 1d array to allow slicing\n    '
    r = 2 ** (1 / bins_per_octave)
    return np.atleast_1d((r ** 2 - 1) / (r ** 2 + 1))