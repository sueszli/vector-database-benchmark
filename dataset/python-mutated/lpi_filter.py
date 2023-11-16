"""
:author: Stefan van der Walt, 2008
:license: modified BSD
"""
import numpy as np
import scipy.fft as fft
from .._shared.utils import _supported_float_type, check_nD

def _min_limit(x, val=np.finfo(float).eps):
    if False:
        while True:
            i = 10
    mask = np.abs(x) < val
    x[mask] = np.sign(x[mask]) * val

def _center(x, oshape):
    if False:
        print('Hello World!')
    'Return an array of shape ``oshape`` from the center of array ``x``.'
    start = (np.array(x.shape) - np.array(oshape)) // 2
    out = x[tuple((slice(s, s + n) for (s, n) in zip(start, oshape)))]
    return out

def _pad(data, shape):
    if False:
        for i in range(10):
            print('nop')
    'Pad the data to the given shape with zeros.\n\n    Parameters\n    ----------\n    data : 2-d ndarray\n        Input data\n    shape : (2,) tuple\n\n    '
    out = np.zeros(shape, dtype=data.dtype)
    out[tuple((slice(0, n) for n in data.shape))] = data
    return out

class LPIFilter2D:
    """Linear Position-Invariant Filter (2-dimensional)"""

    def __init__(self, impulse_response, **filter_params):
        if False:
            i = 10
            return i + 15
        "\n        Parameters\n        ----------\n        impulse_response : callable `f(r, c, **filter_params)`\n            Function that yields the impulse response.  ``r`` and ``c`` are\n            1-dimensional vectors that represent row and column positions, in\n            other words coordinates are (r[0],c[0]),(r[0],c[1]) etc.\n            `**filter_params` are passed through.\n\n            In other words, ``impulse_response`` would be called like this:\n\n            >>> def impulse_response(r, c, **filter_params):\n            ...     pass\n            >>>\n            >>> r = [0,0,0,1,1,1,2,2,2]\n            >>> c = [0,1,2,0,1,2,0,1,2]\n            >>> filter_params = {'kw1': 1, 'kw2': 2, 'kw3': 3}\n            >>> impulse_response(r, c, **filter_params)\n\n\n        Examples\n        --------\n        Gaussian filter without normalization of coefficients:\n\n        >>> def filt_func(r, c, sigma=1):\n        ...     return np.exp(-(r**2 + c**2)/(2 * sigma**2))\n        >>> filter = LPIFilter2D(filt_func)\n\n        "
        if not callable(impulse_response):
            raise ValueError('Impulse response must be a callable.')
        self.impulse_response = impulse_response
        self.filter_params = filter_params
        self._cache = None

    def _prepare(self, data):
        if False:
            print('Hello World!')
        'Calculate filter and data FFT in preparation for filtering.'
        dshape = np.array(data.shape)
        even_offset = (dshape % 2 == 0).astype(int)
        dshape += even_offset
        oshape = np.array(data.shape) * 2 - 1
        float_dtype = _supported_float_type(data.dtype)
        data = data.astype(float_dtype, copy=False)
        if self._cache is None or np.any(self._cache.shape != oshape):
            coords = np.mgrid[[slice(0 + offset, float(n + offset)) for (n, offset) in zip(dshape, even_offset)]]
            for (k, coord) in enumerate(coords):
                coord -= (dshape[k] - 1) / 2.0
            coords = coords.reshape(2, -1).T
            coords = coords.astype(float_dtype, copy=False)
            f = self.impulse_response(coords[:, 0], coords[:, 1], **self.filter_params).reshape(dshape)
            f = _pad(f, oshape)
            F = fft.fftn(f)
            self._cache = F
        else:
            F = self._cache
        data = _pad(data, oshape)
        G = fft.fftn(data)
        return (F, G)

    def __call__(self, data):
        if False:
            while True:
                i = 10
        'Apply the filter to the given data.\n\n        Parameters\n        ----------\n        data : (M, N) ndarray\n\n        '
        check_nD(data, 2, 'data')
        (F, G) = self._prepare(data)
        out = fft.ifftn(F * G)
        out = np.abs(_center(out, data.shape))
        return out

def filter_forward(data, impulse_response=None, filter_params=None, predefined_filter=None):
    if False:
        print('Hello World!')
    'Apply the given filter to data.\n\n    Parameters\n    ----------\n    data : (M, N) ndarray\n        Input data.\n    impulse_response : callable `f(r, c, **filter_params)`\n        Impulse response of the filter.  See LPIFilter2D.__init__.\n    filter_params : dict, optional\n        Additional keyword parameters to the impulse_response function.\n\n    Other Parameters\n    ----------------\n    predefined_filter : LPIFilter2D\n        If you need to apply the same filter multiple times over different\n        images, construct the LPIFilter2D and specify it here.\n\n    Examples\n    --------\n\n    Gaussian filter without normalization:\n\n    >>> def filt_func(r, c, sigma=1):\n    ...     return np.exp(-(r**2 + c**2)/(2 * sigma**2))\n    >>>\n    >>> from skimage import data\n    >>> filtered = filter_forward(data.coins(), filt_func)\n\n    '
    if filter_params is None:
        filter_params = {}
    check_nD(data, 2, 'data')
    if predefined_filter is None:
        predefined_filter = LPIFilter2D(impulse_response, **filter_params)
    return predefined_filter(data)

def filter_inverse(data, impulse_response=None, filter_params=None, max_gain=2, predefined_filter=None):
    if False:
        i = 10
        return i + 15
    'Apply the filter in reverse to the given data.\n\n    Parameters\n    ----------\n    data : (M, N) ndarray\n        Input data.\n    impulse_response : callable `f(r, c, **filter_params)`\n        Impulse response of the filter.  See :class:`~.LPIFilter2D`. This is a required\n        argument unless a `predifined_filter` is provided.\n    filter_params : dict, optional\n        Additional keyword parameters to the impulse_response function.\n    max_gain : float, optional\n        Limit the filter gain.  Often, the filter contains zeros, which would\n        cause the inverse filter to have infinite gain.  High gain causes\n        amplification of artefacts, so a conservative limit is recommended.\n\n    Other Parameters\n    ----------------\n    predefined_filter : LPIFilter2D, optional\n        If you need to apply the same filter multiple times over different\n        images, construct the LPIFilter2D and specify it here.\n\n    '
    if filter_params is None:
        filter_params = {}
    check_nD(data, 2, 'data')
    if predefined_filter is None:
        filt = LPIFilter2D(impulse_response, **filter_params)
    else:
        filt = predefined_filter
    (F, G) = filt._prepare(data)
    _min_limit(F, val=np.finfo(F.real.dtype).eps)
    F = 1 / F
    mask = np.abs(F) > max_gain
    F[mask] = np.sign(F[mask]) * max_gain
    return _center(np.abs(fft.ifftshift(fft.ifftn(G * F))), data.shape)

def wiener(data, impulse_response=None, filter_params=None, K=0.25, predefined_filter=None):
    if False:
        return 10
    'Minimum Mean Square Error (Wiener) inverse filter.\n\n    Parameters\n    ----------\n    data : (M, N) ndarray\n        Input data.\n    K : float or (M, N) ndarray\n        Ratio between power spectrum of noise and undegraded\n        image.\n    impulse_response : callable `f(r, c, **filter_params)`\n        Impulse response of the filter.  See LPIFilter2D.__init__.\n    filter_params : dict, optional\n        Additional keyword parameters to the impulse_response function.\n\n    Other Parameters\n    ----------------\n    predefined_filter : LPIFilter2D\n        If you need to apply the same filter multiple times over different\n        images, construct the LPIFilter2D and specify it here.\n\n    '
    if filter_params is None:
        filter_params = {}
    check_nD(data, 2, 'data')
    if not isinstance(K, float):
        check_nD(K, 2, 'K')
    if predefined_filter is None:
        filt = LPIFilter2D(impulse_response, **filter_params)
    else:
        filt = predefined_filter
    (F, G) = filt._prepare(data)
    _min_limit(F, val=np.finfo(F.real.dtype).eps)
    H_mag_sqr = np.abs(F) ** 2
    F = 1 / F * H_mag_sqr / (H_mag_sqr + K)
    return _center(np.abs(fft.ifftshift(fft.ifftn(G * F))), data.shape)