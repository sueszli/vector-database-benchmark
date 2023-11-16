import math
import numpy as np
import scipy.ndimage as ndi
from scipy import spatial
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, check_nD
from ..transform import integral_image
from ..util import img_as_float
from ._hessian_det_appx import _hessian_matrix_det
from .peak import peak_local_max

def _compute_disk_overlap(d, r1, r2):
    if False:
        while True:
            i = 10
    '\n    Compute fraction of surface overlap between two disks of radii\n    ``r1`` and ``r2``, with centers separated by a distance ``d``.\n\n    Parameters\n    ----------\n    d : float\n        Distance between centers.\n    r1 : float\n        Radius of the first disk.\n    r2 : float\n        Radius of the second disk.\n\n    Returns\n    -------\n    fraction: float\n        Fraction of area of the overlap between the two disks.\n    '
    ratio1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
    ratio1 = np.clip(ratio1, -1, 1)
    acos1 = math.acos(ratio1)
    ratio2 = (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)
    ratio2 = np.clip(ratio2, -1, 1)
    acos2 = math.acos(ratio2)
    a = -d + r2 + r1
    b = d - r2 + r1
    c = d + r2 - r1
    d = d + r2 + r1
    area = r1 ** 2 * acos1 + r2 ** 2 * acos2 - 0.5 * math.sqrt(abs(a * b * c * d))
    return area / (math.pi * min(r1, r2) ** 2)

def _compute_sphere_overlap(d, r1, r2):
    if False:
        return 10
    '\n    Compute volume overlap fraction between two spheres of radii\n    ``r1`` and ``r2``, with centers separated by a distance ``d``.\n\n    Parameters\n    ----------\n    d : float\n        Distance between centers.\n    r1 : float\n        Radius of the first sphere.\n    r2 : float\n        Radius of the second sphere.\n\n    Returns\n    -------\n    fraction: float\n        Fraction of volume of the overlap between the two spheres.\n\n    Notes\n    -----\n    See for example http://mathworld.wolfram.com/Sphere-SphereIntersection.html\n    for more details.\n    '
    vol = math.pi / (12 * d) * (r1 + r2 - d) ** 2 * (d ** 2 + 2 * d * (r1 + r2) - 3 * (r1 ** 2 + r2 ** 2) + 6 * r1 * r2)
    return vol / (4.0 / 3 * math.pi * min(r1, r2) ** 3)

def _blob_overlap(blob1, blob2, *, sigma_dim=1):
    if False:
        i = 10
        return i + 15
    'Finds the overlapping area fraction between two blobs.\n\n    Returns a float representing fraction of overlapped area. Note that 0.0\n    is *always* returned for dimension greater than 3.\n\n    Parameters\n    ----------\n    blob1 : sequence of arrays\n        A sequence of ``(row, col, sigma)`` or ``(pln, row, col, sigma)``,\n        where ``row, col`` (or ``(pln, row, col)``) are coordinates\n        of blob and ``sigma`` is the standard deviation of the Gaussian kernel\n        which detected the blob.\n    blob2 : sequence of arrays\n        A sequence of ``(row, col, sigma)`` or ``(pln, row, col, sigma)``,\n        where ``row, col`` (or ``(pln, row, col)``) are coordinates\n        of blob and ``sigma`` is the standard deviation of the Gaussian kernel\n        which detected the blob.\n    sigma_dim : int, optional\n        The dimensionality of the sigma value. Can be 1 or the same as the\n        dimensionality of the blob space (2 or 3).\n\n    Returns\n    -------\n    f : float\n        Fraction of overlapped area (or volume in 3D).\n    '
    ndim = len(blob1) - sigma_dim
    if ndim > 3:
        return 0.0
    root_ndim = math.sqrt(ndim)
    if blob1[-1] == blob2[-1] == 0:
        return 0.0
    elif blob1[-1] > blob2[-1]:
        max_sigma = blob1[-sigma_dim:]
        r1 = 1
        r2 = blob2[-1] / blob1[-1]
    else:
        max_sigma = blob2[-sigma_dim:]
        r2 = 1
        r1 = blob1[-1] / blob2[-1]
    pos1 = blob1[:ndim] / (max_sigma * root_ndim)
    pos2 = blob2[:ndim] / (max_sigma * root_ndim)
    d = np.sqrt(np.sum((pos2 - pos1) ** 2))
    if d > r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return 1.0
    if ndim == 2:
        return _compute_disk_overlap(d, r1, r2)
    else:
        return _compute_sphere_overlap(d, r1, r2)

def _prune_blobs(blobs_array, overlap, *, sigma_dim=1):
    if False:
        for i in range(10):
            print('nop')
    'Eliminated blobs with area overlap.\n\n    Parameters\n    ----------\n    blobs_array : ndarray\n        A 2d array with each row representing 3 (or 4) values,\n        ``(row, col, sigma)`` or ``(pln, row, col, sigma)`` in 3D,\n        where ``(row, col)`` (``(pln, row, col)``) are coordinates of the blob\n        and ``sigma`` is the standard deviation of the Gaussian kernel which\n        detected the blob.\n        This array must not have a dimension of size 0.\n    overlap : float\n        A value between 0 and 1. If the fraction of area overlapping for 2\n        blobs is greater than `overlap` the smaller blob is eliminated.\n    sigma_dim : int, optional\n        The number of columns in ``blobs_array`` corresponding to sigmas rather\n        than positions.\n\n    Returns\n    -------\n    A : ndarray\n        `array` with overlapping blobs removed.\n    '
    sigma = blobs_array[:, -sigma_dim:].max()
    distance = 2 * sigma * math.sqrt(blobs_array.shape[1] - sigma_dim)
    tree = spatial.cKDTree(blobs_array[:, :-sigma_dim])
    pairs = np.array(list(tree.query_pairs(distance)))
    if len(pairs) == 0:
        return blobs_array
    else:
        for (i, j) in pairs:
            (blob1, blob2) = (blobs_array[i], blobs_array[j])
            if _blob_overlap(blob1, blob2, sigma_dim=sigma_dim) > overlap:
                if blob1[-1] > blob2[-1]:
                    blob2[-1] = 0
                else:
                    blob1[-1] = 0
    return np.stack([b for b in blobs_array if b[-1] > 0])

def _format_exclude_border(img_ndim, exclude_border):
    if False:
        print('Hello World!')
    'Format an ``exclude_border`` argument as a tuple of ints for calling\n    ``peak_local_max``.\n    '
    if isinstance(exclude_border, tuple):
        if len(exclude_border) != img_ndim:
            raise ValueError('`exclude_border` should have the same length as the dimensionality of the image.')
        for exclude in exclude_border:
            if not isinstance(exclude, int):
                raise ValueError('exclude border, when expressed as a tuple, must only contain ints.')
        return exclude_border + (0,)
    elif isinstance(exclude_border, int):
        return (exclude_border,) * img_ndim + (0,)
    elif exclude_border is True:
        raise ValueError('exclude_border cannot be True')
    elif exclude_border is False:
        return (0,) * (img_ndim + 1)
    else:
        raise ValueError(f'Unsupported value ({exclude_border}) for exclude_border')

def blob_dog(image, min_sigma=1, max_sigma=50, sigma_ratio=1.6, threshold=0.5, overlap=0.5, *, threshold_rel=None, exclude_border=False):
    if False:
        return 10
    'Finds blobs in the given grayscale image.\n\n    Blobs are found using the Difference of Gaussian (DoG) method [1]_, [2]_.\n    For each blob found, the method returns its coordinates and the standard\n    deviation of the Gaussian kernel that detected the blob.\n\n    Parameters\n    ----------\n    image : ndarray\n        Input grayscale image, blobs are assumed to be light on dark\n        background (white on black).\n    min_sigma : scalar or sequence of scalars, optional\n        The minimum standard deviation for Gaussian kernel. Keep this low to\n        detect smaller blobs. The standard deviations of the Gaussian filter\n        are given for each axis as a sequence, or as a single number, in\n        which case it is equal for all axes.\n    max_sigma : scalar or sequence of scalars, optional\n        The maximum standard deviation for Gaussian kernel. Keep this high to\n        detect larger blobs. The standard deviations of the Gaussian filter\n        are given for each axis as a sequence, or as a single number, in\n        which case it is equal for all axes.\n    sigma_ratio : float, optional\n        The ratio between the standard deviation of Gaussian Kernels used for\n        computing the Difference of Gaussians\n    threshold : float or None, optional\n        The absolute lower bound for scale space maxima. Local maxima smaller\n        than `threshold` are ignored. Reduce this to detect blobs with lower\n        intensities. If `threshold_rel` is also specified, whichever threshold\n        is larger will be used. If None, `threshold_rel` is used instead.\n    overlap : float, optional\n        A value between 0 and 1. If the area of two blobs overlaps by a\n        fraction greater than `threshold`, the smaller blob is eliminated.\n    threshold_rel : float or None, optional\n        Minimum intensity of peaks, calculated as\n        ``max(dog_space) * threshold_rel``, where ``dog_space`` refers to the\n        stack of Difference-of-Gaussian (DoG) images computed internally. This\n        should have a value between 0 and 1. If None, `threshold` is used\n        instead.\n    exclude_border : tuple of ints, int, or False, optional\n        If tuple of ints, the length of the tuple must match the input array\'s\n        dimensionality.  Each element of the tuple will exclude peaks from\n        within `exclude_border`-pixels of the border of the image along that\n        dimension.\n        If nonzero int, `exclude_border` excludes peaks from within\n        `exclude_border`-pixels of the border of the image.\n        If zero or False, peaks are identified regardless of their\n        distance from the border.\n\n    Returns\n    -------\n    A : (n, image.ndim + sigma) ndarray\n        A 2d array with each row representing 2 coordinate values for a 2D\n        image, or 3 coordinate values for a 3D image, plus the sigma(s) used.\n        When a single sigma is passed, outputs are:\n        ``(r, c, sigma)`` or ``(p, r, c, sigma)`` where ``(r, c)`` or\n        ``(p, r, c)`` are coordinates of the blob and ``sigma`` is the standard\n        deviation of the Gaussian kernel which detected the blob. When an\n        anisotropic gaussian is used (sigmas per dimension), the detected sigma\n        is returned for each dimension.\n\n    See also\n    --------\n    skimage.filters.difference_of_gaussians\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Blob_detection#The_difference_of_Gaussians_approach\n    .. [2] Lowe, D. G. "Distinctive Image Features from Scale-Invariant\n        Keypoints." International Journal of Computer Vision 60, 91–110 (2004).\n        https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf\n        :DOI:`10.1023/B:VISI.0000029664.99615.94`\n\n    Examples\n    --------\n    >>> from skimage import data, feature\n    >>> coins = data.coins()\n    >>> feature.blob_dog(coins, threshold=.05, min_sigma=10, max_sigma=40)\n    array([[128., 155.,  10.],\n           [198., 155.,  10.],\n           [124., 338.,  10.],\n           [127., 102.,  10.],\n           [193., 281.,  10.],\n           [126., 208.,  10.],\n           [267., 115.,  10.],\n           [197., 102.,  10.],\n           [198., 215.,  10.],\n           [123., 279.,  10.],\n           [126.,  46.,  10.],\n           [259., 247.,  10.],\n           [196.,  43.,  10.],\n           [ 54., 276.,  10.],\n           [267., 358.,  10.],\n           [ 58., 100.,  10.],\n           [259., 305.,  10.],\n           [185., 347.,  16.],\n           [261., 174.,  16.],\n           [ 46., 336.,  16.],\n           [ 54., 217.,  10.],\n           [ 55., 157.,  10.],\n           [ 57.,  41.,  10.],\n           [260.,  47.,  16.]])\n\n    Notes\n    -----\n    The radius of each blob is approximately :math:`\\sqrt{2}\\sigma` for\n    a 2-D image and :math:`\\sqrt{3}\\sigma` for a 3-D image.\n    '
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    scalar_sigma = np.isscalar(max_sigma) and np.isscalar(min_sigma)
    if np.isscalar(max_sigma):
        max_sigma = np.full(image.ndim, max_sigma, dtype=float_dtype)
    if np.isscalar(min_sigma):
        min_sigma = np.full(image.ndim, min_sigma, dtype=float_dtype)
    min_sigma = np.asarray(min_sigma, dtype=float_dtype)
    max_sigma = np.asarray(max_sigma, dtype=float_dtype)
    if sigma_ratio <= 1.0:
        raise ValueError('sigma_ratio must be > 1.0')
    k = int(np.mean(np.log(max_sigma / min_sigma) / np.log(sigma_ratio) + 1))
    sigma_list = np.array([min_sigma * sigma_ratio ** i for i in range(k + 1)])
    dog_image_cube = np.empty(image.shape + (k,), dtype=float_dtype)
    gaussian_previous = gaussian(image, sigma_list[0], mode='reflect')
    for (i, s) in enumerate(sigma_list[1:]):
        gaussian_current = gaussian(image, s, mode='reflect')
        dog_image_cube[..., i] = gaussian_previous - gaussian_current
        gaussian_previous = gaussian_current
    sf = 1 / (sigma_ratio - 1)
    dog_image_cube *= sf
    exclude_border = _format_exclude_border(image.ndim, exclude_border)
    local_maxima = peak_local_max(dog_image_cube, threshold_abs=threshold, threshold_rel=threshold_rel, exclude_border=exclude_border, footprint=np.ones((3,) * (image.ndim + 1)))
    if local_maxima.size == 0:
        return np.empty((0, image.ndim + (1 if scalar_sigma else image.ndim)))
    lm = local_maxima.astype(float_dtype)
    sigmas_of_peaks = sigma_list[local_maxima[:, -1]]
    if scalar_sigma:
        sigmas_of_peaks = sigmas_of_peaks[:, 0:1]
    lm = np.hstack([lm[:, :-1], sigmas_of_peaks])
    sigma_dim = sigmas_of_peaks.shape[1]
    return _prune_blobs(lm, overlap, sigma_dim=sigma_dim)

def blob_log(image, min_sigma=1, max_sigma=50, num_sigma=10, threshold=0.2, overlap=0.5, log_scale=False, *, threshold_rel=None, exclude_border=False):
    if False:
        return 10
    "Finds blobs in the given grayscale image.\n\n    Blobs are found using the Laplacian of Gaussian (LoG) method [1]_.\n    For each blob found, the method returns its coordinates and the standard\n    deviation of the Gaussian kernel that detected the blob.\n\n    Parameters\n    ----------\n    image : ndarray\n        Input grayscale image, blobs are assumed to be light on dark\n        background (white on black).\n    min_sigma : scalar or sequence of scalars, optional\n        the minimum standard deviation for Gaussian kernel. Keep this low to\n        detect smaller blobs. The standard deviations of the Gaussian filter\n        are given for each axis as a sequence, or as a single number, in\n        which case it is equal for all axes.\n    max_sigma : scalar or sequence of scalars, optional\n        The maximum standard deviation for Gaussian kernel. Keep this high to\n        detect larger blobs. The standard deviations of the Gaussian filter\n        are given for each axis as a sequence, or as a single number, in\n        which case it is equal for all axes.\n    num_sigma : int, optional\n        The number of intermediate values of standard deviations to consider\n        between `min_sigma` and `max_sigma`.\n    threshold : float or None, optional\n        The absolute lower bound for scale space maxima. Local maxima smaller\n        than `threshold` are ignored. Reduce this to detect blobs with lower\n        intensities. If `threshold_rel` is also specified, whichever threshold\n        is larger will be used. If None, `threshold_rel` is used instead.\n    overlap : float, optional\n        A value between 0 and 1. If the area of two blobs overlaps by a\n        fraction greater than `threshold`, the smaller blob is eliminated.\n    log_scale : bool, optional\n        If set intermediate values of standard deviations are interpolated\n        using a logarithmic scale to the base `10`. If not, linear\n        interpolation is used.\n    threshold_rel : float or None, optional\n        Minimum intensity of peaks, calculated as\n        ``max(log_space) * threshold_rel``, where ``log_space`` refers to the\n        stack of Laplacian-of-Gaussian (LoG) images computed internally. This\n        should have a value between 0 and 1. If None, `threshold` is used\n        instead.\n    exclude_border : tuple of ints, int, or False, optional\n        If tuple of ints, the length of the tuple must match the input array's\n        dimensionality.  Each element of the tuple will exclude peaks from\n        within `exclude_border`-pixels of the border of the image along that\n        dimension.\n        If nonzero int, `exclude_border` excludes peaks from within\n        `exclude_border`-pixels of the border of the image.\n        If zero or False, peaks are identified regardless of their\n        distance from the border.\n\n    Returns\n    -------\n    A : (n, image.ndim + sigma) ndarray\n        A 2d array with each row representing 2 coordinate values for a 2D\n        image, or 3 coordinate values for a 3D image, plus the sigma(s) used.\n        When a single sigma is passed, outputs are:\n        ``(r, c, sigma)`` or ``(p, r, c, sigma)`` where ``(r, c)`` or\n        ``(p, r, c)`` are coordinates of the blob and ``sigma`` is the standard\n        deviation of the Gaussian kernel which detected the blob. When an\n        anisotropic gaussian is used (sigmas per dimension), the detected sigma\n        is returned for each dimension.\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Blob_detection#The_Laplacian_of_Gaussian\n\n    Examples\n    --------\n    >>> from skimage import data, feature, exposure\n    >>> img = data.coins()\n    >>> img = exposure.equalize_hist(img)  # improves detection\n    >>> feature.blob_log(img, threshold = .3)\n    array([[124.        , 336.        ,  11.88888889],\n           [198.        , 155.        ,  11.88888889],\n           [194.        , 213.        ,  17.33333333],\n           [121.        , 272.        ,  17.33333333],\n           [263.        , 244.        ,  17.33333333],\n           [194.        , 276.        ,  17.33333333],\n           [266.        , 115.        ,  11.88888889],\n           [128.        , 154.        ,  11.88888889],\n           [260.        , 174.        ,  17.33333333],\n           [198.        , 103.        ,  11.88888889],\n           [126.        , 208.        ,  11.88888889],\n           [127.        , 102.        ,  11.88888889],\n           [263.        , 302.        ,  17.33333333],\n           [197.        ,  44.        ,  11.88888889],\n           [185.        , 344.        ,  17.33333333],\n           [126.        ,  46.        ,  11.88888889],\n           [113.        , 323.        ,   1.        ]])\n\n    Notes\n    -----\n    The radius of each blob is approximately :math:`\\sqrt{2}\\sigma` for\n    a 2-D image and :math:`\\sqrt{3}\\sigma` for a 3-D image.\n    "
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    scalar_sigma = True if np.isscalar(max_sigma) and np.isscalar(min_sigma) else False
    if np.isscalar(max_sigma):
        max_sigma = np.full(image.ndim, max_sigma, dtype=float_dtype)
    if np.isscalar(min_sigma):
        min_sigma = np.full(image.ndim, min_sigma, dtype=float_dtype)
    min_sigma = np.asarray(min_sigma, dtype=float_dtype)
    max_sigma = np.asarray(max_sigma, dtype=float_dtype)
    if log_scale:
        start = np.log10(min_sigma)
        stop = np.log10(max_sigma)
        sigma_list = np.logspace(start, stop, num_sigma)
    else:
        sigma_list = np.linspace(min_sigma, max_sigma, num_sigma)
    image_cube = np.empty(image.shape + (len(sigma_list),), dtype=float_dtype)
    for (i, s) in enumerate(sigma_list):
        image_cube[..., i] = -ndi.gaussian_laplace(image, s) * np.mean(s) ** 2
    exclude_border = _format_exclude_border(image.ndim, exclude_border)
    local_maxima = peak_local_max(image_cube, threshold_abs=threshold, threshold_rel=threshold_rel, exclude_border=exclude_border, footprint=np.ones((3,) * (image.ndim + 1)))
    if local_maxima.size == 0:
        return np.empty((0, image.ndim + (1 if scalar_sigma else image.ndim)))
    lm = local_maxima.astype(float_dtype)
    sigmas_of_peaks = sigma_list[local_maxima[:, -1]]
    if scalar_sigma:
        sigmas_of_peaks = sigmas_of_peaks[:, 0:1]
    lm = np.hstack([lm[:, :-1], sigmas_of_peaks])
    sigma_dim = sigmas_of_peaks.shape[1]
    return _prune_blobs(lm, overlap, sigma_dim=sigma_dim)

def blob_doh(image, min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.01, overlap=0.5, log_scale=False, *, threshold_rel=None):
    if False:
        return 10
    'Finds blobs in the given grayscale image.\n\n    Blobs are found using the Determinant of Hessian method [1]_. For each blob\n    found, the method returns its coordinates and the standard deviation\n    of the Gaussian Kernel used for the Hessian matrix whose determinant\n    detected the blob. Determinant of Hessians is approximated using [2]_.\n\n    Parameters\n    ----------\n    image : 2D ndarray\n        Input grayscale image.Blobs can either be light on dark or vice versa.\n    min_sigma : float, optional\n        The minimum standard deviation for Gaussian Kernel used to compute\n        Hessian matrix. Keep this low to detect smaller blobs.\n    max_sigma : float, optional\n        The maximum standard deviation for Gaussian Kernel used to compute\n        Hessian matrix. Keep this high to detect larger blobs.\n    num_sigma : int, optional\n        The number of intermediate values of standard deviations to consider\n        between `min_sigma` and `max_sigma`.\n    threshold : float or None, optional\n        The absolute lower bound for scale space maxima. Local maxima smaller\n        than `threshold` are ignored. Reduce this to detect blobs with lower\n        intensities. If `threshold_rel` is also specified, whichever threshold\n        is larger will be used. If None, `threshold_rel` is used instead.\n    overlap : float, optional\n        A value between 0 and 1. If the area of two blobs overlaps by a\n        fraction greater than `threshold`, the smaller blob is eliminated.\n    log_scale : bool, optional\n        If set intermediate values of standard deviations are interpolated\n        using a logarithmic scale to the base `10`. If not, linear\n        interpolation is used.\n    threshold_rel : float or None, optional\n        Minimum intensity of peaks, calculated as\n        ``max(doh_space) * threshold_rel``, where ``doh_space`` refers to the\n        stack of Determinant-of-Hessian (DoH) images computed internally. This\n        should have a value between 0 and 1. If None, `threshold` is used\n        instead.\n\n    Returns\n    -------\n    A : (n, 3) ndarray\n        A 2d array with each row representing 3 values, ``(y,x,sigma)``\n        where ``(y,x)`` are coordinates of the blob and ``sigma`` is the\n        standard deviation of the Gaussian kernel of the Hessian Matrix whose\n        determinant detected the blob.\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Blob_detection#The_determinant_of_the_Hessian\n    .. [2] Herbert Bay, Andreas Ess, Tinne Tuytelaars, Luc Van Gool,\n           "SURF: Speeded Up Robust Features"\n           ftp://ftp.vision.ee.ethz.ch/publications/articles/eth_biwi_00517.pdf\n\n    Examples\n    --------\n    >>> from skimage import data, feature\n    >>> img = data.coins()\n    >>> feature.blob_doh(img)\n    array([[197.        , 153.        ,  20.33333333],\n           [124.        , 336.        ,  20.33333333],\n           [126.        , 153.        ,  20.33333333],\n           [195.        , 100.        ,  23.55555556],\n           [192.        , 212.        ,  23.55555556],\n           [121.        , 271.        ,  30.        ],\n           [126.        , 101.        ,  20.33333333],\n           [193.        , 275.        ,  23.55555556],\n           [123.        , 205.        ,  20.33333333],\n           [270.        , 363.        ,  30.        ],\n           [265.        , 113.        ,  23.55555556],\n           [262.        , 243.        ,  23.55555556],\n           [185.        , 348.        ,  30.        ],\n           [156.        , 302.        ,  30.        ],\n           [123.        ,  44.        ,  23.55555556],\n           [260.        , 173.        ,  30.        ],\n           [197.        ,  44.        ,  20.33333333]])\n\n    Notes\n    -----\n    The radius of each blob is approximately `sigma`.\n    Computation of Determinant of Hessians is independent of the standard\n    deviation. Therefore detecting larger blobs won\'t take more time. In\n    methods line :py:meth:`blob_dog` and :py:meth:`blob_log` the computation\n    of Gaussians for larger `sigma` takes more time. The downside is that\n    this method can\'t be used for detecting blobs of radius less than `3px`\n    due to the box filters used in the approximation of Hessian Determinant.\n    '
    check_nD(image, 2)
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    image = integral_image(image)
    if log_scale:
        (start, stop) = (math.log(min_sigma, 10), math.log(max_sigma, 10))
        sigma_list = np.logspace(start, stop, num_sigma)
    else:
        sigma_list = np.linspace(min_sigma, max_sigma, num_sigma)
    image_cube = np.empty(shape=image.shape + (len(sigma_list),), dtype=float_dtype)
    for (j, s) in enumerate(sigma_list):
        image_cube[..., j] = _hessian_matrix_det(image, s)
    local_maxima = peak_local_max(image_cube, threshold_abs=threshold, threshold_rel=threshold_rel, exclude_border=False, footprint=np.ones((3,) * image_cube.ndim))
    if local_maxima.size == 0:
        return np.empty((0, 3))
    lm = local_maxima.astype(np.float64)
    lm[:, -1] = sigma_list[local_maxima[:, -1]]
    return _prune_blobs(lm, overlap)