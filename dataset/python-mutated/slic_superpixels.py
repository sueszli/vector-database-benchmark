import math
from collections.abc import Iterable
from warnings import warn
import numpy as np
from numpy import random
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import pdist, squareform
from .._shared import utils
from .._shared.filters import gaussian
from ..color import rgb2lab
from ..util import img_as_float, regular_grid
from ._slic import _enforce_label_connectivity_cython, _slic_cython

def _get_mask_centroids(mask, n_centroids, multichannel):
    if False:
        return 10
    'Find regularly spaced centroids on a mask.\n\n    Parameters\n    ----------\n    mask : 3D ndarray\n        The mask within which the centroids must be positioned.\n    n_centroids : int\n        The number of centroids to be returned.\n\n    Returns\n    -------\n    centroids : 2D ndarray\n        The coordinates of the centroids with shape (n_centroids, 3).\n    steps : 1D ndarray\n        The approximate distance between two seeds in all dimensions.\n\n    '
    coord = np.array(np.nonzero(mask), dtype=float).T
    rng = random.RandomState(123)
    idx_full = np.arange(len(coord), dtype=int)
    idx = np.sort(rng.choice(idx_full, min(n_centroids, len(coord)), replace=False))
    dense_factor = 10
    ndim_spatial = mask.ndim - 1 if multichannel else mask.ndim
    n_dense = int(dense_factor ** ndim_spatial * n_centroids)
    if len(coord) > n_dense:
        idx_dense = np.sort(rng.choice(idx_full, n_dense, replace=False))
    else:
        idx_dense = Ellipsis
    (centroids, _) = kmeans2(coord[idx_dense], coord[idx], iter=5)
    dist = squareform(pdist(centroids))
    np.fill_diagonal(dist, np.inf)
    closest_pts = dist.argmin(-1)
    steps = abs(centroids - centroids[closest_pts, :]).mean(0)
    return (centroids, steps)

def _get_grid_centroids(image, n_centroids):
    if False:
        print('Hello World!')
    'Find regularly spaced centroids on the image.\n\n    Parameters\n    ----------\n    image : 2D, 3D or 4D ndarray\n        Input image, which can be 2D or 3D, and grayscale or\n        multichannel.\n    n_centroids : int\n        The (approximate) number of centroids to be returned.\n\n    Returns\n    -------\n    centroids : 2D ndarray\n        The coordinates of the centroids with shape (~n_centroids, 3).\n    steps : 1D ndarray\n        The approximate distance between two seeds in all dimensions.\n\n    '
    (d, h, w) = image.shape[:3]
    (grid_z, grid_y, grid_x) = np.mgrid[:d, :h, :w]
    slices = regular_grid(image.shape[:3], n_centroids)
    centroids_z = grid_z[slices].ravel()[..., np.newaxis]
    centroids_y = grid_y[slices].ravel()[..., np.newaxis]
    centroids_x = grid_x[slices].ravel()[..., np.newaxis]
    centroids = np.concatenate([centroids_z, centroids_y, centroids_x], axis=-1)
    steps = np.asarray([float(s.step) if s.step is not None else 1.0 for s in slices])
    return (centroids, steps)

@utils.channel_as_last_axis(multichannel_output=False)
def slic(image, n_segments=100, compactness=10.0, max_num_iter=10, sigma=0, spacing=None, convert2lab=None, enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3, slic_zero=False, start_label=1, mask=None, *, channel_axis=-1):
    if False:
        print('Hello World!')
    'Segments image using k-means clustering in Color-(x,y,z) space.\n\n    Parameters\n    ----------\n    image : (M, N[, P][, C]) ndarray\n        Input image. Can be 2D or 3D, and grayscale or multichannel\n        (see `channel_axis` parameter).\n        Input image must either be NaN-free or the NaN\'s must be masked out.\n    n_segments : int, optional\n        The (approximate) number of labels in the segmented output image.\n    compactness : float, optional\n        Balances color proximity and space proximity. Higher values give\n        more weight to space proximity, making superpixel shapes more\n        square/cubic. In SLICO mode, this is the initial compactness.\n        This parameter depends strongly on image contrast and on the\n        shapes of objects in the image. We recommend exploring possible\n        values on a log scale, e.g., 0.01, 0.1, 1, 10, 100, before\n        refining around a chosen value.\n    max_num_iter : int, optional\n        Maximum number of iterations of k-means.\n    sigma : float or array-like of floats, optional\n        Width of Gaussian smoothing kernel for pre-processing for each\n        dimension of the image. The same sigma is applied to each dimension in\n        case of a scalar value. Zero means no smoothing.\n        Note that `sigma` is automatically scaled if it is scalar and\n        if a manual voxel spacing is provided (see Notes section). If\n        sigma is array-like, its size must match ``image``\'s number\n        of spatial dimensions.\n    spacing : array-like of floats, optional\n        The voxel spacing along each spatial dimension. By default,\n        `slic` assumes uniform spacing (same voxel resolution along\n        each spatial dimension).\n        This parameter controls the weights of the distances along the\n        spatial dimensions during k-means clustering.\n    convert2lab : bool, optional\n        Whether the input should be converted to Lab colorspace prior to\n        segmentation. The input image *must* be RGB. Highly recommended.\n        This option defaults to ``True`` when ``channel_axis` is not None *and*\n        ``image.shape[-1] == 3``.\n    enforce_connectivity : bool, optional\n        Whether the generated segments are connected or not\n    min_size_factor : float, optional\n        Proportion of the minimum segment size to be removed with respect\n        to the supposed segment size ```depth*width*height/n_segments```\n    max_size_factor : float, optional\n        Proportion of the maximum connected segment size. A value of 3 works\n        in most of the cases.\n    slic_zero : bool, optional\n        Run SLIC-zero, the zero-parameter mode of SLIC. [2]_\n    start_label : int, optional\n        The labels\' index start. Should be 0 or 1.\n\n        .. versionadded:: 0.17\n           ``start_label`` was introduced in 0.17\n    mask : ndarray, optional\n        If provided, superpixels are computed only where mask is True,\n        and seed points are homogeneously distributed over the mask\n        using a k-means clustering strategy. Mask number of dimensions\n        must be equal to image number of spatial dimensions.\n\n        .. versionadded:: 0.17\n           ``mask`` was introduced in 0.17\n    channel_axis : int or None, optional\n        If None, the image is assumed to be a grayscale (single channel) image.\n        Otherwise, this parameter indicates which axis of the array corresponds\n        to channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    labels : 2D or 3D array\n        Integer mask indicating segment labels.\n\n    Raises\n    ------\n    ValueError\n        If ``convert2lab`` is set to ``True`` but the last array\n        dimension is not of length 3.\n    ValueError\n        If ``start_label`` is not 0 or 1.\n    ValueError\n        If ``image`` contains unmasked NaN values.\n    ValueError\n        If ``image`` contains unmasked infinite values.\n    ValueError\n        If ``image`` is 2D but ``channel_axis`` is -1 (the default).\n\n    Notes\n    -----\n    * If `sigma > 0`, the image is smoothed using a Gaussian kernel prior to\n      segmentation.\n\n    * If `sigma` is scalar and `spacing` is provided, the kernel width is\n      divided along each dimension by the spacing. For example, if ``sigma=1``\n      and ``spacing=[5, 1, 1]``, the effective `sigma` is ``[0.2, 1, 1]``. This\n      ensures sensible smoothing for anisotropic images.\n\n    * The image is rescaled to be in [0, 1] prior to processing (masked\n      values are ignored).\n\n    * Images of shape (M, N, 3) are interpreted as 2D RGB images by default. To\n      interpret them as 3D with the last dimension having length 3, use\n      `channel_axis=None`.\n\n    * `start_label` is introduced to handle the issue [4]_. Label indexing\n      starts at 1 by default.\n\n    References\n    ----------\n    .. [1] Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi,\n        Pascal Fua, and Sabine Süsstrunk, SLIC Superpixels Compared to\n        State-of-the-art Superpixel Methods, TPAMI, May 2012.\n        :DOI:`10.1109/TPAMI.2012.120`\n    .. [2] https://www.epfl.ch/labs/ivrl/research/slic-superpixels/#SLICO\n    .. [3] Irving, Benjamin. "maskSLIC: regional superpixel generation with\n           application to local pathology characterisation in medical images.",\n           2016, :arXiv:`1606.09518`\n    .. [4] https://github.com/scikit-image/scikit-image/issues/3722\n\n    Examples\n    --------\n    >>> from skimage.segmentation import slic\n    >>> from skimage.data import astronaut\n    >>> img = astronaut()\n    >>> segments = slic(img, n_segments=100, compactness=10)\n\n    Increasing the compactness parameter yields more square regions:\n\n    >>> segments = slic(img, n_segments=100, compactness=20)\n\n    '
    if image.ndim == 2 and channel_axis is not None:
        raise ValueError(f'channel_axis={channel_axis} indicates multichannel, which is not supported for a two-dimensional image; use channel_axis=None if the image is grayscale')
    image = img_as_float(image)
    float_dtype = utils._supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=True)
    if mask is not None:
        mask = np.ascontiguousarray(mask, dtype=bool)
        if channel_axis is not None:
            mask_ = np.expand_dims(mask, axis=channel_axis)
            mask_ = np.broadcast_to(mask_, image.shape)
        else:
            mask_ = mask
        image_values = image[mask_]
    else:
        image_values = image
    imin = image_values.min()
    imax = image_values.max()
    if np.isnan(imin):
        raise ValueError('unmasked NaN values in image are not supported')
    if np.isinf(imin) or np.isinf(imax):
        raise ValueError('unmasked infinite values in image are not supported')
    image -= imin
    if imax != imin:
        image /= imax - imin
    use_mask = mask is not None
    dtype = image.dtype
    is_2d = False
    multichannel = channel_axis is not None
    if image.ndim == 2:
        image = image[np.newaxis, ..., np.newaxis]
        is_2d = True
    elif image.ndim == 3 and multichannel:
        image = image[np.newaxis, ...]
        is_2d = True
    elif image.ndim == 3 and (not multichannel):
        image = image[..., np.newaxis]
    if multichannel and (convert2lab or convert2lab is None):
        if image.shape[channel_axis] != 3 and convert2lab:
            raise ValueError('Lab colorspace conversion requires a RGB image.')
        elif image.shape[channel_axis] == 3:
            image = rgb2lab(image)
    if start_label not in [0, 1]:
        raise ValueError('start_label should be 0 or 1.')
    update_centroids = False
    if use_mask:
        mask = mask.view('uint8')
        if mask.ndim == 2:
            mask = np.ascontiguousarray(mask[np.newaxis, ...])
        if mask.shape != image.shape[:3]:
            raise ValueError('image and mask should have the same shape.')
        (centroids, steps) = _get_mask_centroids(mask, n_segments, multichannel)
        update_centroids = True
    else:
        (centroids, steps) = _get_grid_centroids(image, n_segments)
    if spacing is None:
        spacing = np.ones(3, dtype=dtype)
    elif isinstance(spacing, Iterable):
        spacing = np.asarray(spacing, dtype=dtype)
        if is_2d:
            if spacing.size != 2:
                if spacing.size == 3:
                    warn('Input image is 2D: spacing number of elements must be 2. In the future, a ValueError will be raised.', FutureWarning, stacklevel=2)
                else:
                    raise ValueError(f'Input image is 2D, but spacing has {spacing.size} elements (expected 2).')
            else:
                spacing = np.insert(spacing, 0, 1)
        elif spacing.size != 3:
            raise ValueError(f'Input image is 3D, but spacing has {spacing.size} elements (expected 3).')
        spacing = np.ascontiguousarray(spacing, dtype=dtype)
    else:
        raise TypeError('spacing must be None or iterable.')
    if np.isscalar(sigma):
        sigma = np.array([sigma, sigma, sigma], dtype=dtype)
        sigma /= spacing
    elif isinstance(sigma, Iterable):
        sigma = np.asarray(sigma, dtype=dtype)
        if is_2d:
            if sigma.size != 2:
                if spacing.size == 3:
                    warn('Input image is 2D: sigma number of elements must be 2. In the future, a ValueError will be raised.', FutureWarning, stacklevel=2)
                else:
                    raise ValueError(f'Input image is 2D, but sigma has {sigma.size} elements (expected 2).')
            else:
                sigma = np.insert(sigma, 0, 0)
        elif sigma.size != 3:
            raise ValueError(f'Input image is 3D, but sigma has {sigma.size} elements (expected 3).')
    if (sigma > 0).any():
        sigma = list(sigma) + [0]
        image = gaussian(image, sigma, mode='reflect')
    n_centroids = centroids.shape[0]
    segments = np.ascontiguousarray(np.concatenate([centroids, np.zeros((n_centroids, image.shape[3]))], axis=-1), dtype=dtype)
    step = max(steps)
    ratio = 1.0 / compactness
    image = np.ascontiguousarray(image * ratio, dtype=dtype)
    if update_centroids:
        _slic_cython(image, mask, segments, step, max_num_iter, spacing, slic_zero, ignore_color=True, start_label=start_label)
    labels = _slic_cython(image, mask, segments, step, max_num_iter, spacing, slic_zero, ignore_color=False, start_label=start_label)
    if enforce_connectivity:
        if use_mask:
            segment_size = mask.sum() / n_centroids
        else:
            segment_size = math.prod(image.shape[:3]) / n_centroids
        min_size = int(min_size_factor * segment_size)
        max_size = int(max_size_factor * segment_size)
        labels = _enforce_label_connectivity_cython(labels, min_size, max_size, start_label=start_label)
    if is_2d:
        labels = labels[0]
    return labels