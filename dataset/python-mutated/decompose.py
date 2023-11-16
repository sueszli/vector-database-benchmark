"""
Spectrogram decomposition
=========================
.. autosummary::
    :toctree: generated/

    decompose
    hpss
    nn_filter
"""
import numpy as np
import scipy.sparse
from scipy.ndimage import median_filter
import sklearn.decomposition
from . import core
from ._cache import cache
from . import segment
from . import util
from .util.exceptions import ParameterError
from typing import Any, Callable, List, Optional, Tuple, Union
from ._typing import _IntLike_co, _FloatLike_co
__all__ = ['decompose', 'hpss', 'nn_filter']

def decompose(S: np.ndarray, *, n_components: Optional[int]=None, transformer: Optional[object]=None, sort: bool=False, fit: bool=True, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    'Decompose a feature matrix.\n\n    Given a spectrogram ``S``, produce a decomposition into ``components``\n    and ``activations`` such that ``S ~= components.dot(activations)``.\n\n    By default, this is done with with non-negative matrix factorization (NMF),\n    but any `sklearn.decomposition`-type object will work.\n\n    Parameters\n    ----------\n    S : np.ndarray [shape=(..., n_features, n_samples), dtype=float]\n        The input feature matrix (e.g., magnitude spectrogram)\n\n        If the input has multiple channels (leading dimensions), they will be automatically\n        flattened prior to decomposition.\n\n        If the input is multi-channel, channels and features are automatically flattened into\n        a single axis before the decomposition.\n        For example, a stereo input `S` with shape `(2, n_features, n_samples)` is\n        automatically reshaped to `(2 * n_features, n_samples)`.\n\n    n_components : int > 0 [scalar] or None\n        number of desired components\n\n        if None, then ``n_features`` components are used\n\n    transformer : None or object\n        If None, use `sklearn.decomposition.NMF`\n\n        Otherwise, any object with a similar interface to NMF should work.\n        ``transformer`` must follow the scikit-learn convention, where\n        input data is ``(n_samples, n_features)``.\n\n        `transformer.fit_transform()` will be run on ``S.T`` (not ``S``),\n        the return value of which is stored (transposed) as ``activations``\n\n        The components will be retrieved as ``transformer.components_.T``::\n\n            S ~= np.dot(activations, transformer.components_).T\n\n        or equivalently::\n\n            S ~= np.dot(transformer.components_.T, activations.T)\n\n    sort : bool\n        If ``True``, components are sorted by ascending peak frequency.\n\n        .. note:: If used with ``transformer``, sorting is applied to copies\n            of the decomposition parameters, and not to ``transformer``\n            internal parameters.\n\n        .. warning:: If the input array has more than two dimensions\n            (e.g., if it\'s a multi-channel spectrogram), then axis sorting\n            is not supported and a `ParameterError` exception is raised.\n\n    fit : bool\n        If `True`, components are estimated from the input ``S``.\n\n        If `False`, components are assumed to be pre-computed and stored\n        in ``transformer``, and are not changed.\n\n    **kwargs : Additional keyword arguments to the default transformer\n        `sklearn.decomposition.NMF`\n\n    Returns\n    -------\n    components: np.ndarray [shape=(..., n_features, n_components)]\n        matrix of components (basis elements).\n    activations: np.ndarray [shape=(n_components, n_samples)]\n        transformed matrix/activation matrix\n\n    Raises\n    ------\n    ParameterError\n        if ``fit`` is False and no ``transformer`` object is provided.\n\n        if the input array is multi-channel and ``sort=True`` is specified.\n\n    See Also\n    --------\n    sklearn.decomposition : SciKit-Learn matrix decomposition modules\n\n    Examples\n    --------\n    Decompose a magnitude spectrogram into 16 components with NMF\n\n    >>> y, sr = librosa.load(librosa.ex(\'pistachio\'), duration=5)\n    >>> S = np.abs(librosa.stft(y))\n    >>> comps, acts = librosa.decompose.decompose(S, n_components=16)\n\n    Sort components by ascending peak frequency\n\n    >>> comps, acts = librosa.decompose.decompose(S, n_components=16,\n    ...                                           sort=True)\n\n    Or with sparse dictionary learning\n\n    >>> import sklearn.decomposition\n    >>> T = sklearn.decomposition.MiniBatchDictionaryLearning(n_components=16)\n    >>> scomps, sacts = librosa.decompose.decompose(S, transformer=T, sort=True)\n\n    >>> import matplotlib.pyplot as plt\n    >>> layout = [list(".AAAA"), list("BCCCC"), list(".DDDD")]\n    >>> fig, ax = plt.subplot_mosaic(layout, constrained_layout=True)\n    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),\n    ...                          y_axis=\'log\', x_axis=\'time\', ax=ax[\'A\'])\n    >>> ax[\'A\'].set(title=\'Input spectrogram\')\n    >>> ax[\'A\'].label_outer()\n    >>> librosa.display.specshow(librosa.amplitude_to_db(comps,\n    >>>                                                  ref=np.max),\n    >>>                          y_axis=\'log\', ax=ax[\'B\'])\n    >>> ax[\'B\'].set(title=\'Components\')\n    >>> ax[\'B\'].label_outer()\n    >>> ax[\'B\'].sharey(ax[\'A\'])\n    >>> librosa.display.specshow(acts, x_axis=\'time\', ax=ax[\'C\'], cmap=\'gray_r\')\n    >>> ax[\'C\'].set(ylabel=\'Components\', title=\'Activations\')\n    >>> ax[\'C\'].sharex(ax[\'A\'])\n    >>> ax[\'C\'].label_outer()\n    >>> S_approx = comps.dot(acts)\n    >>> img = librosa.display.specshow(librosa.amplitude_to_db(S_approx,\n    >>>                                                        ref=np.max),\n    >>>                                y_axis=\'log\', x_axis=\'time\', ax=ax[\'D\'])\n    >>> ax[\'D\'].set(title=\'Reconstructed spectrogram\')\n    >>> ax[\'D\'].sharex(ax[\'A\'])\n    >>> ax[\'D\'].sharey(ax[\'A\'])\n    >>> ax[\'D\'].label_outer()\n    >>> fig.colorbar(img, ax=list(ax.values()), format="%+2.f dB")\n    '
    orig_shape = list(S.shape)
    if S.ndim > 2 and sort:
        raise ParameterError('Parameter sort=True is unsupported for input with more than two dimensions')
    S = S.T.reshape((S.shape[-1], -1), order='F')
    if n_components is None:
        n_components = S.shape[-1]
    if transformer is None:
        if fit is False:
            raise ParameterError('fit must be True if transformer is None')
        transformer = sklearn.decomposition.NMF(n_components=n_components, **kwargs)
    activations: np.ndarray
    if fit:
        activations = transformer.fit_transform(S).T
    else:
        activations = transformer.transform(S).T
    components: np.ndarray = transformer.components_
    component_shape = orig_shape[:-1] + [-1]
    components = components.reshape(component_shape[::-1], order='F').T
    if sort:
        (components, idx) = util.axis_sort(components, index=True)
        activations = activations[idx]
    return (components, activations)

@cache(level=30)
def hpss(S: np.ndarray, *, kernel_size: Union[_IntLike_co, Tuple[_IntLike_co, _IntLike_co], List[_IntLike_co]]=31, power: float=2.0, mask: bool=False, margin: Union[_FloatLike_co, Tuple[_FloatLike_co, _FloatLike_co], List[_FloatLike_co]]=1.0) -> Tuple[np.ndarray, np.ndarray]:
    if False:
        print('Hello World!')
    'Median-filtering harmonic percussive source separation (HPSS).\n\n    If ``margin = 1.0``, decomposes an input spectrogram ``S = H + P``\n    where ``H`` contains the harmonic components,\n    and ``P`` contains the percussive components.\n\n    If ``margin > 1.0``, decomposes an input spectrogram ``S = H + P + R``\n    where ``R`` contains residual components not included in ``H`` or ``P``.\n\n    This implementation is based upon the algorithm described by [#]_ and [#]_.\n\n    .. [#] Fitzgerald, Derry.\n        "Harmonic/percussive separation using median filtering."\n        13th International Conference on Digital Audio Effects (DAFX10),\n        Graz, Austria, 2010.\n\n    .. [#] Driedger, Müller, Disch.\n        "Extending harmonic-percussive separation of audio."\n        15th International Society for Music Information Retrieval Conference (ISMIR 2014),\n        Taipei, Taiwan, 2014.\n\n    Parameters\n    ----------\n    S : np.ndarray [shape=(..., d, n)]\n        input spectrogram. May be real (magnitude) or complex.\n        Multi-channel is supported.\n\n    kernel_size : int or tuple (kernel_harmonic, kernel_percussive)\n        kernel size(s) for the median filters.\n\n        - If scalar, the same size is used for both harmonic and percussive.\n        - If tuple, the first value specifies the width of the\n          harmonic filter, and the second value specifies the width\n          of the percussive filter.\n\n    power : float > 0 [scalar]\n        Exponent for the Wiener filter when constructing soft mask matrices.\n\n    mask : bool\n        Return the masking matrices instead of components.\n\n        Masking matrices contain non-negative real values that\n        can be used to measure the assignment of energy from ``S``\n        into harmonic or percussive components.\n\n        Components can be recovered by multiplying ``S * mask_H``\n        or ``S * mask_P``.\n\n    margin : float or tuple (margin_harmonic, margin_percussive)\n        margin size(s) for the masks (as described in [2]_)\n\n        - If scalar, the same size is used for both harmonic and percussive.\n        - If tuple, the first value specifies the margin of the\n          harmonic mask, and the second value specifies the margin\n          of the percussive mask.\n\n    Returns\n    -------\n    harmonic : np.ndarray [shape=(..., d, n)]\n        harmonic component (or mask)\n    percussive : np.ndarray [shape=(..., d, n)]\n        percussive component (or mask)\n\n    See Also\n    --------\n    librosa.util.softmask\n\n    Notes\n    -----\n    This function caches at level 30.\n\n    Examples\n    --------\n    Separate into harmonic and percussive\n\n    >>> y, sr = librosa.load(librosa.ex(\'choice\'), duration=5)\n    >>> D = librosa.stft(y)\n    >>> H, P = librosa.decompose.hpss(D)\n\n    >>> import matplotlib.pyplot as plt\n    >>> fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)\n    >>> img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(D),\n    ...                                                        ref=np.max),\n    ...                          y_axis=\'log\', x_axis=\'time\', ax=ax[0])\n    >>> ax[0].set(title=\'Full power spectrogram\')\n    >>> ax[0].label_outer()\n    >>> librosa.display.specshow(librosa.amplitude_to_db(np.abs(H),\n    ...                                                  ref=np.max(np.abs(D))),\n    ...                          y_axis=\'log\', x_axis=\'time\', ax=ax[1])\n    >>> ax[1].set(title=\'Harmonic power spectrogram\')\n    >>> ax[1].label_outer()\n    >>> librosa.display.specshow(librosa.amplitude_to_db(np.abs(P),\n    ...                                                  ref=np.max(np.abs(D))),\n    ...                          y_axis=\'log\', x_axis=\'time\', ax=ax[2])\n    >>> ax[2].set(title=\'Percussive power spectrogram\')\n    >>> fig.colorbar(img, ax=ax, format=\'%+2.0f dB\')\n\n    Or with a narrower horizontal filter\n\n    >>> H, P = librosa.decompose.hpss(D, kernel_size=(13, 31))\n\n    Just get harmonic/percussive masks, not the spectra\n\n    >>> mask_H, mask_P = librosa.decompose.hpss(D, mask=True)\n    >>> mask_H\n    array([[1.853e-03, 1.701e-04, ..., 9.922e-01, 1.000e+00],\n           [2.316e-03, 2.127e-04, ..., 9.989e-01, 1.000e+00],\n           ...,\n           [8.195e-05, 6.939e-05, ..., 3.105e-04, 4.231e-04],\n           [3.159e-05, 4.156e-05, ..., 6.216e-04, 6.188e-04]],\n          dtype=float32)\n    >>> mask_P\n    array([[9.981e-01, 9.998e-01, ..., 7.759e-03, 3.201e-05],\n           [9.977e-01, 9.998e-01, ..., 1.122e-03, 4.451e-06],\n           ...,\n           [9.999e-01, 9.999e-01, ..., 9.997e-01, 9.996e-01],\n           [1.000e+00, 1.000e+00, ..., 9.994e-01, 9.994e-01]],\n          dtype=float32)\n\n    Separate into harmonic/percussive/residual components by using a margin > 1.0\n\n    >>> H, P = librosa.decompose.hpss(D, margin=3.0)\n    >>> R = D - (H+P)\n    >>> y_harm = librosa.istft(H)\n    >>> y_perc = librosa.istft(P)\n    >>> y_resi = librosa.istft(R)\n\n    Get a more isolated percussive component by widening its margin\n\n    >>> H, P = librosa.decompose.hpss(D, margin=(1.0,5.0))\n    '
    phase: Union[float, np.ndarray]
    if np.iscomplexobj(S):
        (S, phase) = core.magphase(S)
    else:
        phase = 1
    if isinstance(kernel_size, tuple) or isinstance(kernel_size, list):
        win_harm = kernel_size[0]
        win_perc = kernel_size[1]
    else:
        win_harm = kernel_size
        win_perc = kernel_size
    if isinstance(margin, tuple) or isinstance(margin, list):
        margin_harm = margin[0]
        margin_perc = margin[1]
    else:
        margin_harm = margin
        margin_perc = margin
    if margin_harm < 1 or margin_perc < 1:
        raise ParameterError('Margins must be >= 1.0. A typical range is between 1 and 10.')
    harm_shape: List[_IntLike_co] = [1] * S.ndim
    harm_shape[-1] = win_harm
    perc_shape: List[_IntLike_co] = [1] * S.ndim
    perc_shape[-2] = win_perc
    harm = np.empty_like(S)
    harm[:] = median_filter(S, size=harm_shape, mode='reflect')
    perc = np.empty_like(S)
    perc[:] = median_filter(S, size=perc_shape, mode='reflect')
    split_zeros = margin_harm == 1 and margin_perc == 1
    mask_harm = util.softmask(harm, perc * margin_harm, power=power, split_zeros=split_zeros)
    mask_perc = util.softmask(perc, harm * margin_perc, power=power, split_zeros=split_zeros)
    if mask:
        return (mask_harm, mask_perc)
    return (S * mask_harm * phase, S * mask_perc * phase)

@cache(level=30)
def nn_filter(S: np.ndarray, *, rec: Optional[Union[scipy.sparse.spmatrix, np.ndarray]]=None, aggregate: Optional[Callable]=None, axis: int=-1, **kwargs: Any) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    'Filter by nearest-neighbor aggregation.\n\n    Each data point (e.g, spectrogram column) is replaced\n    by aggregating its nearest neighbors in feature space.\n\n    This can be useful for de-noising a spectrogram or feature matrix.\n\n    The non-local means method [#]_ can be recovered by providing a\n    weighted recurrence matrix as input and specifying ``aggregate=np.average``.\n\n    Similarly, setting ``aggregate=np.median`` produces sparse de-noising\n    as in REPET-SIM [#]_.\n\n    .. [#] Buades, A., Coll, B., & Morel, J. M.\n        (2005, June). A non-local algorithm for image denoising.\n        In Computer Vision and Pattern Recognition, 2005.\n        CVPR 2005. IEEE Computer Society Conference on (Vol. 2, pp. 60-65). IEEE.\n\n    .. [#] Rafii, Z., & Pardo, B.\n        (2012, October).  "Music/Voice Separation Using the Similarity Matrix."\n        International Society for Music Information Retrieval Conference, 2012.\n\n    Parameters\n    ----------\n    S : np.ndarray\n        The input data (spectrogram) to filter. Multi-channel is supported.\n\n    rec : (optional) scipy.sparse.spmatrix or np.ndarray\n        Optionally, a pre-computed nearest-neighbor matrix\n        as provided by `librosa.segment.recurrence_matrix`\n\n    aggregate : function\n        aggregation function (default: `np.mean`)\n\n        If ``aggregate=np.average``, then a weighted average is\n        computed according to the (per-row) weights in ``rec``.\n\n        For all other aggregation functions, all neighbors\n        are treated equally.\n\n    axis : int\n        The axis along which to filter (by default, columns)\n\n    **kwargs\n        Additional keyword arguments provided to\n        `librosa.segment.recurrence_matrix` if ``rec`` is not provided\n\n    Returns\n    -------\n    S_filtered : np.ndarray\n        The filtered data, with shape equivalent to the input ``S``.\n\n    Raises\n    ------\n    ParameterError\n        if ``rec`` is provided and its shape is incompatible with ``S``.\n\n    See Also\n    --------\n    decompose\n    hpss\n    librosa.segment.recurrence_matrix\n\n    Notes\n    -----\n    This function caches at level 30.\n\n    Examples\n    --------\n    De-noise a chromagram by non-local median filtering.\n    By default this would use euclidean distance to select neighbors,\n    but this can be overridden directly by setting the ``metric`` parameter.\n\n    >>> y, sr = librosa.load(librosa.ex(\'brahms\'),\n    ...                      offset=30, duration=10)\n    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr)\n    >>> chroma_med = librosa.decompose.nn_filter(chroma,\n    ...                                          aggregate=np.median,\n    ...                                          metric=\'cosine\')\n\n    To use non-local means, provide an affinity matrix and ``aggregate=np.average``.\n\n    >>> rec = librosa.segment.recurrence_matrix(chroma, mode=\'affinity\',\n    ...                                         metric=\'cosine\', sparse=True)\n    >>> chroma_nlm = librosa.decompose.nn_filter(chroma, rec=rec,\n    ...                                          aggregate=np.average)\n\n    >>> import matplotlib.pyplot as plt\n    >>> fig, ax = plt.subplots(nrows=5, sharex=True, sharey=True, figsize=(10, 10))\n    >>> librosa.display.specshow(chroma, y_axis=\'chroma\', x_axis=\'time\', ax=ax[0])\n    >>> ax[0].set(title=\'Unfiltered\')\n    >>> ax[0].label_outer()\n    >>> librosa.display.specshow(chroma_med, y_axis=\'chroma\', x_axis=\'time\', ax=ax[1])\n    >>> ax[1].set(title=\'Median-filtered\')\n    >>> ax[1].label_outer()\n    >>> imgc = librosa.display.specshow(chroma_nlm, y_axis=\'chroma\', x_axis=\'time\', ax=ax[2])\n    >>> ax[2].set(title=\'Non-local means\')\n    >>> ax[2].label_outer()\n    >>> imgr1 = librosa.display.specshow(chroma - chroma_med,\n    ...                          y_axis=\'chroma\', x_axis=\'time\', ax=ax[3])\n    >>> ax[3].set(title=\'Original - median\')\n    >>> ax[3].label_outer()\n    >>> imgr2 = librosa.display.specshow(chroma - chroma_nlm,\n    ...                          y_axis=\'chroma\', x_axis=\'time\', ax=ax[4])\n    >>> ax[4].label_outer()\n    >>> ax[4].set(title=\'Original - NLM\')\n    >>> fig.colorbar(imgc, ax=ax[:3])\n    >>> fig.colorbar(imgr1, ax=[ax[3]])\n    >>> fig.colorbar(imgr2, ax=[ax[4]])\n    '
    if aggregate is None:
        aggregate = np.mean
    rec_s: scipy.sparse.spmatrix
    if rec is None:
        kwargs = dict(kwargs)
        kwargs['sparse'] = True
        rec_s = segment.recurrence_matrix(S, axis=axis, **kwargs)
    elif not scipy.sparse.issparse(rec):
        rec_s = scipy.sparse.csc_matrix(rec)
    else:
        rec_s = rec
    if rec_s.shape[0] != S.shape[axis] or rec_s.shape[0] != rec_s.shape[1]:
        raise ParameterError(f'Invalid self-similarity matrix shape rec.shape={rec_s.shape} for S.shape={S.shape}')
    return __nn_filter_helper(rec_s.data, rec_s.indices, rec_s.indptr, S.swapaxes(0, axis), aggregate).swapaxes(0, axis)

def __nn_filter_helper(R_data, R_indices, R_ptr, S: np.ndarray, aggregate: Callable) -> np.ndarray:
    if False:
        return 10
    'Nearest-neighbor filter helper function.\n\n    This is an internal function, not for use outside of the decompose module.\n\n    It applies the nearest-neighbor filter to S, assuming that the first index\n    corresponds to observations.\n\n    Parameters\n    ----------\n    R_data, R_indices, R_ptr : np.ndarrays\n        The ``data``, ``indices``, and ``indptr`` of a scipy.sparse matrix\n    S : np.ndarray\n        The observation data to filter\n    aggregate : callable\n        The aggregation operator\n\n    Returns\n    -------\n    S_out : np.ndarray like S\n        The filtered data array\n    '
    s_out = np.empty_like(S)
    for i in range(len(R_ptr) - 1):
        targets = R_indices[R_ptr[i]:R_ptr[i + 1]]
        if not len(targets):
            s_out[i] = S[i]
            continue
        neighbors = np.take(S, targets, axis=0)
        if aggregate is np.average:
            weights = R_data[R_ptr[i]:R_ptr[i + 1]]
            s_out[i] = aggregate(neighbors, axis=0, weights=weights)
        else:
            s_out[i] = aggregate(neighbors, axis=0)
    return s_out