import numpy as np
from .._shared.utils import _supported_float_type
from ._rolling_ball_cy import apply_kernel, apply_kernel_nan

def rolling_ball(image, *, radius=100, kernel=None, nansafe=False, num_threads=None):
    if False:
        while True:
            i = 10
    'Estimate background intensity by rolling/translating a kernel.\n\n    This rolling ball algorithm estimates background intensity for a\n    ndimage in case of uneven exposure. It is a generalization of the\n    frequently used rolling ball algorithm [1]_.\n\n    Parameters\n    ----------\n    image : ndarray\n        The image to be filtered.\n    radius : int, optional\n        Radius of a ball-shaped kernel to be rolled/translated in the image.\n        Used if ``kernel = None``.\n    kernel : ndarray, optional\n        The kernel to be rolled/translated in the image. It must have the\n        same number of dimensions as ``image``. Kernel is filled with the\n        intensity of the kernel at that position.\n    nansafe: bool, optional\n        If ``False`` (default) assumes that none of the values in ``image``\n        are ``np.nan``, and uses a faster implementation.\n    num_threads: int, optional\n        The maximum number of threads to use. If ``None`` use the OpenMP\n        default value; typically equal to the maximum number of virtual cores.\n        Note: This is an upper limit to the number of threads. The exact number\n        is determined by the system\'s OpenMP library.\n\n    Returns\n    -------\n    background : ndarray\n        The estimated background of the image.\n\n    Notes\n    -----\n    For the pixel that has its background intensity estimated (without loss\n    of generality at ``center``) the rolling ball method centers ``kernel``\n    under it and raises the kernel until the surface touches the image umbra\n    at some ``pos=(y,x)``. The background intensity is then estimated\n    using the image intensity at that position (``image[pos]``) plus the\n    difference of ``kernel[center] - kernel[pos]``.\n\n    This algorithm assumes that dark pixels correspond to the background. If\n    you have a bright background, invert the image before passing it to the\n    function, e.g., using `utils.invert`. See the gallery example for details.\n\n    This algorithm is sensitive to noise (in particular salt-and-pepper\n    noise). If this is a problem in your image, you can apply mild\n    gaussian smoothing before passing the image to this function.\n\n    References\n    ----------\n    .. [1] Sternberg, Stanley R. "Biomedical image processing." Computer 1\n           (1983): 22-34. :DOI:`10.1109/MC.1983.1654163`\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from skimage import data\n    >>> from skimage.restoration import rolling_ball\n    >>> image = data.coins()\n    >>> background = rolling_ball(data.coins())\n    >>> filtered_image = image - background\n\n\n    >>> import numpy as np\n    >>> from skimage import data\n    >>> from skimage.restoration import rolling_ball, ellipsoid_kernel\n    >>> image = data.coins()\n    >>> kernel = ellipsoid_kernel((101, 101), 75)\n    >>> background = rolling_ball(data.coins(), kernel=kernel)\n    >>> filtered_image = image - background\n    '
    image = np.asarray(image)
    float_type = _supported_float_type(image.dtype)
    img = image.astype(float_type, copy=False)
    if num_threads is None:
        num_threads = 0
    if kernel is None:
        kernel = ball_kernel(radius, image.ndim)
    kernel = kernel.astype(float_type)
    kernel_shape = np.asarray(kernel.shape)
    kernel_center = kernel_shape // 2
    center_intensity = kernel[tuple(kernel_center)]
    intensity_difference = center_intensity - kernel
    intensity_difference[kernel == np.inf] = np.inf
    intensity_difference = intensity_difference.astype(img.dtype)
    intensity_difference = intensity_difference.reshape(-1)
    img = np.pad(img, kernel_center[:, np.newaxis], constant_values=np.inf, mode='constant')
    func = apply_kernel_nan if nansafe else apply_kernel
    background = func(img.reshape(-1), intensity_difference, np.zeros_like(image, dtype=img.dtype).reshape(-1), np.array(image.shape, dtype=np.intp), np.array(img.shape, dtype=np.intp), kernel_shape.astype(np.intp), num_threads)
    background = background.astype(image.dtype, copy=False)
    return background

def ball_kernel(radius, ndim):
    if False:
        for i in range(10):
            print('nop')
    'Create a ball kernel for restoration.rolling_ball.\n\n    Parameters\n    ----------\n    radius : int\n        Radius of the ball.\n    ndim : int\n        Number of dimensions of the ball. ``ndim`` should match the\n        dimensionality of the image the kernel will be applied to.\n\n    Returns\n    -------\n    kernel : ndarray\n        The kernel containing the surface intensity of the top half\n        of the ellipsoid.\n\n    See Also\n    --------\n    rolling_ball\n    '
    kernel_coords = np.stack(np.meshgrid(*[np.arange(-x, x + 1) for x in [np.ceil(radius)] * ndim], indexing='ij'), axis=-1)
    sum_of_squares = np.sum(kernel_coords ** 2, axis=-1)
    distance_from_center = np.sqrt(sum_of_squares)
    kernel = np.sqrt(np.clip(radius ** 2 - sum_of_squares, 0, None))
    kernel[distance_from_center > radius] = np.inf
    return kernel

def ellipsoid_kernel(shape, intensity):
    if False:
        for i in range(10):
            print('nop')
    'Create an ellipoid kernel for restoration.rolling_ball.\n\n    Parameters\n    ----------\n    shape : array-like\n        Length of the principal axis of the ellipsoid (excluding\n        the intensity axis). The kernel needs to have the same\n        dimensionality as the image it will be applied to.\n    intensity : int\n        Length of the intensity axis of the ellipsoid.\n\n    Returns\n    -------\n    kernel : ndarray\n        The kernel containing the surface intensity of the top half\n        of the ellipsoid.\n\n    See Also\n    --------\n    rolling_ball\n    '
    shape = np.asarray(shape)
    semi_axis = np.clip(shape // 2, 1, None)
    kernel_coords = np.stack(np.meshgrid(*[np.arange(-x, x + 1) for x in semi_axis], indexing='ij'), axis=-1)
    intensity_scaling = 1 - np.sum((kernel_coords / semi_axis) ** 2, axis=-1)
    kernel = intensity * np.sqrt(np.clip(intensity_scaling, 0, None))
    kernel[intensity_scaling < 0] = np.inf
    return kernel