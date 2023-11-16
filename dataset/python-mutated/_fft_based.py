import functools
import numpy as np
import scipy.fft as fft
from .._shared.utils import _supported_float_type

def _get_nd_butterworth_filter(shape, factor, order, high_pass, real, dtype=np.float64, squared_butterworth=True):
    if False:
        i = 10
        return i + 15
    'Create a N-dimensional Butterworth mask for an FFT\n\n    Parameters\n    ----------\n    shape : tuple of int\n        Shape of the n-dimensional FFT and mask.\n    factor : float\n        Fraction of mask dimensions where the cutoff should be.\n    order : float\n        Controls the slope in the cutoff region.\n    high_pass : bool\n        Whether the filter is high pass (low frequencies attenuated) or\n        low pass (high frequencies are attenuated).\n    real : bool\n        Whether the FFT is of a real (True) or complex (False) image\n    squared_butterworth : bool, optional\n        When True, the square of the Butterworth filter is used.\n\n    Returns\n    -------\n    wfilt : ndarray\n        The FFT mask.\n\n    '
    ranges = []
    for (i, d) in enumerate(shape):
        axis = np.arange(-(d - 1) // 2, (d - 1) // 2 + 1) / (d * factor)
        ranges.append(fft.ifftshift(axis ** 2))
    if real:
        limit = d // 2 + 1
        ranges[-1] = ranges[-1][:limit]
    q2 = functools.reduce(np.add, np.meshgrid(*ranges, indexing='ij', sparse=True))
    q2 = q2.astype(dtype)
    q2 = np.power(q2, order)
    wfilt = 1 / (1 + q2)
    if high_pass:
        wfilt *= q2
    if not squared_butterworth:
        np.sqrt(wfilt, out=wfilt)
    return wfilt

def butterworth(image, cutoff_frequency_ratio=0.005, high_pass=True, order=2.0, channel_axis=None, *, squared_butterworth=True, npad=0):
    if False:
        for i in range(10):
            print('nop')
    'Apply a Butterworth filter to enhance high or low frequency features.\n\n    This filter is defined in the Fourier domain.\n\n    Parameters\n    ----------\n    image : (M[, N[, ..., P]][, C]) ndarray\n        Input image.\n    cutoff_frequency_ratio : float, optional\n        Determines the position of the cut-off relative to the shape of the\n        FFT. Receives a value between [0, 0.5].\n    high_pass : bool, optional\n        Whether to perform a high pass filter. If False, a low pass filter is\n        performed.\n    order : float, optional\n        Order of the filter which affects the slope near the cut-off. Higher\n        order means steeper slope in frequency space.\n    channel_axis : int, optional\n        If there is a channel dimension, provide the index here. If None\n        (default) then all axes are assumed to be spatial dimensions.\n    squared_butterworth : bool, optional\n        When True, the square of a Butterworth filter is used. See notes below\n        for more details.\n    npad : int, optional\n        Pad each edge of the image by `npad` pixels using `numpy.pad`\'s\n        ``mode=\'edge\'`` extension.\n\n    Returns\n    -------\n    result : ndarray\n        The Butterworth-filtered image.\n\n    Notes\n    -----\n    A band-pass filter can be achieved by combining a high-pass and low-pass\n    filter. The user can increase `npad` if boundary artifacts are apparent.\n\n    The "Butterworth filter" used in image processing textbooks (e.g. [1]_,\n    [2]_) is often the square of the traditional Butterworth filters as\n    described by [3]_, [4]_. The squared version will be used here if\n    `squared_butterworth` is set to ``True``. The lowpass, squared Butterworth\n    filter is given by the following expression for the lowpass case:\n\n    .. math::\n        H_{low}(f) = \\frac{1}{1 + \\left(\\frac{f}{c f_s}\\right)^{2n}}\n\n    with the highpass case given by\n\n    .. math::\n        H_{hi}(f) = 1 - H_{low}(f)\n\n    where :math:`f=\\sqrt{\\sum_{d=0}^{\\mathrm{ndim}} f_{d}^{2}}` is the\n    absolute value of the spatial frequency, :math:`f_s` is the sampling\n    frequency, :math:`c` the ``cutoff_frequency_ratio``, and :math:`n` is the\n    filter `order` [1]_. When ``squared_butterworth=False``, the square root of\n    the above expressions are used instead.\n\n    Note that ``cutoff_frequency_ratio`` is defined in terms of the sampling\n    frequency, :math:`f_s`. The FFT spectrum covers the Nyquist range\n    (:math:`[-f_s/2, f_s/2]`) so ``cutoff_frequency_ratio`` should have a value\n    between 0 and 0.5. The frequency response (gain) at the cutoff is 0.5 when\n    ``squared_butterworth`` is true and :math:`1/\\sqrt{2}` when it is false.\n\n    Examples\n    --------\n    Apply a high-pass and low-pass Butterworth filter to a grayscale and\n    color image respectively:\n\n    >>> from skimage.data import camera, astronaut\n    >>> from skimage.filters import butterworth\n    >>> high_pass = butterworth(camera(), 0.07, True, 8)\n    >>> low_pass = butterworth(astronaut(), 0.01, False, 4, channel_axis=-1)\n\n    References\n    ----------\n    .. [1] Russ, John C., et al. The Image Processing Handbook, 3rd. Ed.\n           1999, CRC Press, LLC.\n    .. [2] Birchfield, Stan. Image Processing and Analysis. 2018. Cengage\n           Learning.\n    .. [3] Butterworth, Stephen. "On the theory of filter amplifiers."\n           Wireless Engineer 7.6 (1930): 536-541.\n    .. [4] https://en.wikipedia.org/wiki/Butterworth_filter\n\n    '
    if npad < 0:
        raise ValueError('npad must be >= 0')
    elif npad > 0:
        center_slice = tuple((slice(npad, s + npad) for s in image.shape))
        image = np.pad(image, npad, mode='edge')
    fft_shape = image.shape if channel_axis is None else np.delete(image.shape, channel_axis)
    is_real = np.isrealobj(image)
    float_dtype = _supported_float_type(image.dtype, allow_complex=True)
    if cutoff_frequency_ratio < 0 or cutoff_frequency_ratio > 0.5:
        raise ValueError('cutoff_frequency_ratio should be in the range [0, 0.5]')
    wfilt = _get_nd_butterworth_filter(fft_shape, cutoff_frequency_ratio, order, high_pass, is_real, float_dtype, squared_butterworth)
    axes = np.arange(image.ndim)
    if channel_axis is not None:
        axes = np.delete(axes, channel_axis)
        abs_channel = channel_axis % image.ndim
        post = image.ndim - abs_channel - 1
        sl = (slice(None),) * abs_channel + (np.newaxis,) + (slice(None),) * post
        wfilt = wfilt[sl]
    if is_real:
        butterfilt = fft.irfftn(wfilt * fft.rfftn(image, axes=axes), s=fft_shape, axes=axes)
    else:
        butterfilt = fft.ifftn(wfilt * fft.fftn(image, axes=axes), s=fft_shape, axes=axes)
    if npad > 0:
        butterfilt = butterfilt[center_slice]
    return butterfilt