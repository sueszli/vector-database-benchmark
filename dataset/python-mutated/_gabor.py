import math
import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import _supported_float_type, check_nD
__all__ = ['gabor_kernel', 'gabor']

def _sigma_prefactor(bandwidth):
    if False:
        while True:
            i = 10
    b = bandwidth
    return 1.0 / np.pi * math.sqrt(math.log(2) / 2.0) * (2.0 ** b + 1) / (2.0 ** b - 1)

def gabor_kernel(frequency, theta=0, bandwidth=1, sigma_x=None, sigma_y=None, n_stds=3, offset=0, dtype=np.complex128):
    if False:
        for i in range(10):
            print('nop')
    'Return complex 2D Gabor filter kernel.\n\n    Gabor kernel is a Gaussian kernel modulated by a complex harmonic function.\n    Harmonic function consists of an imaginary sine function and a real\n    cosine function. Spatial frequency is inversely proportional to the\n    wavelength of the harmonic and to the standard deviation of a Gaussian\n    kernel. The bandwidth is also inversely proportional to the standard\n    deviation.\n\n    Parameters\n    ----------\n    frequency : float\n        Spatial frequency of the harmonic function. Specified in pixels.\n    theta : float, optional\n        Orientation in radians. If 0, the harmonic is in the x-direction.\n    bandwidth : float, optional\n        The bandwidth captured by the filter. For fixed bandwidth, ``sigma_x``\n        and ``sigma_y`` will decrease with increasing frequency. This value is\n        ignored if ``sigma_x`` and ``sigma_y`` are set by the user.\n    sigma_x, sigma_y : float, optional\n        Standard deviation in x- and y-directions. These directions apply to\n        the kernel *before* rotation. If `theta = pi/2`, then the kernel is\n        rotated 90 degrees so that ``sigma_x`` controls the *vertical*\n        direction.\n    n_stds : scalar, optional\n        The linear size of the kernel is n_stds (3 by default) standard\n        deviations\n    offset : float, optional\n        Phase offset of harmonic function in radians.\n    dtype : {np.complex64, np.complex128}\n        Specifies if the filter is single or double precision complex.\n\n    Returns\n    -------\n    g : complex array\n        Complex filter kernel.\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Gabor_filter\n    .. [2] https://web.archive.org/web/20180127125930/http://mplab.ucsd.edu/tutorials/gabor.pdf\n\n    Examples\n    --------\n    >>> from skimage.filters import gabor_kernel\n    >>> from skimage import io\n    >>> from matplotlib import pyplot as plt  # doctest: +SKIP\n\n    >>> gk = gabor_kernel(frequency=0.2)\n    >>> plt.figure()        # doctest: +SKIP\n    >>> io.imshow(gk.real)  # doctest: +SKIP\n    >>> io.show()           # doctest: +SKIP\n\n    >>> # more ripples (equivalent to increasing the size of the\n    >>> # Gaussian spread)\n    >>> gk = gabor_kernel(frequency=0.2, bandwidth=0.1)\n    >>> plt.figure()        # doctest: +SKIP\n    >>> io.imshow(gk.real)  # doctest: +SKIP\n    >>> io.show()           # doctest: +SKIP\n    '
    if sigma_x is None:
        sigma_x = _sigma_prefactor(bandwidth) / frequency
    if sigma_y is None:
        sigma_y = _sigma_prefactor(bandwidth) / frequency
    if np.dtype(dtype).kind != 'c':
        raise ValueError('dtype must be complex')
    ct = math.cos(theta)
    st = math.sin(theta)
    x0 = math.ceil(max(abs(n_stds * sigma_x * ct), abs(n_stds * sigma_y * st), 1))
    y0 = math.ceil(max(abs(n_stds * sigma_y * ct), abs(n_stds * sigma_x * st), 1))
    (y, x) = np.meshgrid(np.arange(-y0, y0 + 1), np.arange(-x0, x0 + 1), indexing='ij', sparse=True)
    rotx = x * ct + y * st
    roty = -x * st + y * ct
    g = np.empty(roty.shape, dtype=dtype)
    np.exp(-0.5 * (rotx ** 2 / sigma_x ** 2 + roty ** 2 / sigma_y ** 2) + 1j * (2 * np.pi * frequency * rotx + offset), out=g)
    g *= 1 / (2 * np.pi * sigma_x * sigma_y)
    return g

def gabor(image, frequency, theta=0, bandwidth=1, sigma_x=None, sigma_y=None, n_stds=3, offset=0, mode='reflect', cval=0):
    if False:
        i = 10
        return i + 15
    "Return real and imaginary responses to Gabor filter.\n\n    The real and imaginary parts of the Gabor filter kernel are applied to the\n    image and the response is returned as a pair of arrays.\n\n    Gabor filter is a linear filter with a Gaussian kernel which is modulated\n    by a sinusoidal plane wave. Frequency and orientation representations of\n    the Gabor filter are similar to those of the human visual system.\n    Gabor filter banks are commonly used in computer vision and image\n    processing. They are especially suitable for edge detection and texture\n    classification.\n\n    Parameters\n    ----------\n    image : 2-D array\n        Input image.\n    frequency : float\n        Spatial frequency of the harmonic function. Specified in pixels.\n    theta : float, optional\n        Orientation in radians. If 0, the harmonic is in the x-direction.\n    bandwidth : float, optional\n        The bandwidth captured by the filter. For fixed bandwidth, ``sigma_x``\n        and ``sigma_y`` will decrease with increasing frequency. This value is\n        ignored if ``sigma_x`` and ``sigma_y`` are set by the user.\n    sigma_x, sigma_y : float, optional\n        Standard deviation in x- and y-directions. These directions apply to\n        the kernel *before* rotation. If `theta = pi/2`, then the kernel is\n        rotated 90 degrees so that ``sigma_x`` controls the *vertical*\n        direction.\n    n_stds : scalar, optional\n        The linear size of the kernel is n_stds (3 by default) standard\n        deviations.\n    offset : float, optional\n        Phase offset of harmonic function in radians.\n    mode : {'constant', 'nearest', 'reflect', 'mirror', 'wrap'}, optional\n        Mode used to convolve image with a kernel, passed to `ndi.convolve`\n    cval : scalar, optional\n        Value to fill past edges of input if ``mode`` of convolution is\n        'constant'. The parameter is passed to `ndi.convolve`.\n\n    Returns\n    -------\n    real, imag : arrays\n        Filtered images using the real and imaginary parts of the Gabor filter\n        kernel. Images are of the same dimensions as the input one.\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Gabor_filter\n    .. [2] https://web.archive.org/web/20180127125930/http://mplab.ucsd.edu/tutorials/gabor.pdf\n\n    Examples\n    --------\n    >>> from skimage.filters import gabor\n    >>> from skimage import data, io\n    >>> from matplotlib import pyplot as plt  # doctest: +SKIP\n\n    >>> image = data.coins()\n    >>> # detecting edges in a coin image\n    >>> filt_real, filt_imag = gabor(image, frequency=0.6)\n    >>> plt.figure()            # doctest: +SKIP\n    >>> io.imshow(filt_real)    # doctest: +SKIP\n    >>> io.show()               # doctest: +SKIP\n\n    >>> # less sensitivity to finer details with the lower frequency kernel\n    >>> filt_real, filt_imag = gabor(image, frequency=0.1)\n    >>> plt.figure()            # doctest: +SKIP\n    >>> io.imshow(filt_real)    # doctest: +SKIP\n    >>> io.show()               # doctest: +SKIP\n    "
    check_nD(image, 2)
    if image.dtype.kind == 'f':
        float_dtype = _supported_float_type(image.dtype)
        image = image.astype(float_dtype, copy=False)
        kernel_dtype = np.promote_types(image.dtype, np.complex64)
    else:
        kernel_dtype = np.complex128
    g = gabor_kernel(frequency, theta, bandwidth, sigma_x, sigma_y, n_stds, offset, dtype=kernel_dtype)
    filtered_real = ndi.convolve(image, np.real(g), mode=mode, cval=cval)
    filtered_imag = ndi.convolve(image, np.imag(g), mode=mode, cval=cval)
    return (filtered_real, filtered_imag)