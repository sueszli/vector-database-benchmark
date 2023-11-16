"""watershed.py - watershed algorithm

This module implements a watershed algorithm that apportions pixels into
marked basins. The algorithm uses a priority queue to hold the pixels
with the metric for the priority queue being pixel value, then the time
of entry into the queue - this settles ties in favor of the closest marker.

Some ideas taken from
Soille, "Automated Basin Delineation from Digital Elevation Models Using
Mathematical Morphology", Signal Processing 20 (1990) 171-182.

The most important insight in the paper is that entry time onto the queue
solves two problems: a pixel should be assigned to the neighbor with the
largest gradient or, if there is no gradient, pixels on a plateau should
be split between markers on opposite sides.
"""
import numpy as np
from scipy import ndimage as ndi
from . import _watershed_cy
from ..morphology.extrema import local_minima
from ..morphology._util import _validate_connectivity, _offsets_to_raveled_neighbors
from ..util import crop, regular_seeds

def _validate_inputs(image, markers, mask, connectivity):
    if False:
        for i in range(10):
            print('nop')
    "Ensure that all inputs to watershed have matching shapes and types.\n\n    Parameters\n    ----------\n    image : array\n        The input image.\n    markers : int or array of int\n        The marker image.\n    mask : array, or None\n        A boolean mask, True where we want to compute the watershed.\n    connectivity : int in {1, ..., image.ndim}\n        The connectivity of the neighborhood of a pixel.\n\n    Returns\n    -------\n    image, markers, mask : arrays\n        The validated and formatted arrays. Image will have dtype float64,\n        markers int32, and mask int8. If ``None`` was given for the mask,\n        it is a volume of all 1s.\n\n    Raises\n    ------\n    ValueError\n        If the shapes of the given arrays don't match.\n    "
    n_pixels = image.size
    if mask is None:
        mask = np.ones(image.shape, bool)
    else:
        mask = np.asanyarray(mask, dtype=bool)
        n_pixels = np.sum(mask)
        if mask.shape != image.shape:
            message = f'`mask` (shape {mask.shape}) must have same shape as `image` (shape {image.shape})'
            raise ValueError(message)
    if markers is None:
        markers_bool = local_minima(image, connectivity=connectivity) * mask
        footprint = ndi.generate_binary_structure(markers_bool.ndim, connectivity)
        markers = ndi.label(markers_bool, structure=footprint)[0]
    elif not isinstance(markers, np.ndarray | list | tuple):
        markers = regular_seeds(image.shape, int(markers / (n_pixels / image.size)))
        markers *= mask
    else:
        markers = np.asanyarray(markers) * mask
        if markers.shape != image.shape:
            message = f'`markers` (shape {markers.shape}) must have same shape as `image` (shape {image.shape})'
            raise ValueError(message)
    return (image.astype(np.float64), markers, mask.astype(np.int8))

def watershed(image, markers=None, connectivity=1, offset=None, mask=None, compactness=0, watershed_line=False):
    if False:
        i = 10
        return i + 15
    'Find watershed basins in an image flooded from given markers.\n\n    Parameters\n    ----------\n    image : (M, N[, ...]) ndarray\n        Data array where the lowest value points are labeled first.\n    markers : int, or (M, N[, ...]) ndarray of int, optional\n        The desired number of basins, or an array marking the basins with the\n        values to be assigned in the label matrix. Zero means not a marker. If\n        None, the (default) markers are determined as the local minima of\n        `image`. Specifically, the computation is equivalent to applying\n        :func:`skimage.morphology.local_minima` onto `image`, followed by\n        :func:`skimage.measure.label` onto the result (with the same given\n        `connectivity`). Generally speaking, users are encouraged to pass\n        markers explicitly.\n    connectivity : ndarray, optional\n        An array with the same number of dimensions as `image` whose\n        non-zero elements indicate neighbors for connection.\n        Following the scipy convention, default is a one-connected array of\n        the dimension of the image.\n    offset : array_like of shape image.ndim, optional\n        offset of the connectivity (one offset per dimension)\n    mask : (M, N[, ...]) ndarray of bools or 0\'s and 1\'s, optional\n        Array of same shape as `image`. Only points at which mask == True\n        will be labeled.\n    compactness : float, optional\n        Use compact watershed [1]_ with given compactness parameter.\n        Higher values result in more regularly-shaped watershed basins.\n    watershed_line : bool, optional\n        If True, a one-pixel wide line separates the regions\n        obtained by the watershed algorithm. The line has the label 0.\n        Note that the method used for adding this line expects that\n        marker regions are not adjacent; the watershed line may not catch\n        borders between adjacent marker regions.\n\n    Returns\n    -------\n    out : ndarray\n        A labeled matrix of the same type and shape as `markers`.\n\n    See Also\n    --------\n    skimage.segmentation.random_walker\n        A segmentation algorithm based on anisotropic diffusion, usually\n        slower than the watershed but with good results on noisy data and\n        boundaries with holes.\n\n    Notes\n    -----\n    This function implements a watershed algorithm [2]_ [3]_ that apportions\n    pixels into marked basins. The algorithm uses a priority queue to hold\n    the pixels with the metric for the priority queue being pixel value, then\n    the time of entry into the queue -- this settles ties in favor of the\n    closest marker.\n\n    Some ideas are taken from [4]_.\n    The most important insight in the paper is that entry time onto the queue\n    solves two problems: a pixel should be assigned to the neighbor with the\n    largest gradient or, if there is no gradient, pixels on a plateau should\n    be split between markers on opposite sides.\n\n    This implementation converts all arguments to specific, lowest common\n    denominator types, then passes these to a C algorithm.\n\n    Markers can be determined manually, or automatically using for example\n    the local minima of the gradient of the image, or the local maxima of the\n    distance function to the background for separating overlapping objects\n    (see example).\n\n    References\n    ----------\n    .. [1] P. Neubert and P. Protzel, "Compact Watershed and Preemptive SLIC:\n           On Improving Trade-offs of Superpixel Segmentation Algorithms,"\n           2014 22nd International Conference on Pattern Recognition,\n           Stockholm, Sweden, 2014, pp. 996-1001, :DOI:`10.1109/ICPR.2014.181`\n           https://www.tu-chemnitz.de/etit/proaut/publications/cws_pSLIC_ICPR.pdf\n\n    .. [2] https://en.wikipedia.org/wiki/Watershed_%28image_processing%29\n\n    .. [3] http://cmm.ensmp.fr/~beucher/wtshed.html\n\n    .. [4] P. J. Soille and M. M. Ansoult, "Automated basin delineation from\n           digital elevation models using mathematical morphology," Signal\n           Processing, 20(2):171-182, :DOI:`10.1016/0165-1684(90)90127-K`\n\n    Examples\n    --------\n    The watershed algorithm is useful to separate overlapping objects.\n\n    We first generate an initial image with two overlapping circles:\n\n    >>> x, y = np.indices((80, 80))\n    >>> x1, y1, x2, y2 = 28, 28, 44, 52\n    >>> r1, r2 = 16, 20\n    >>> mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2\n    >>> mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2\n    >>> image = np.logical_or(mask_circle1, mask_circle2)\n\n    Next, we want to separate the two circles. We generate markers at the\n    maxima of the distance to the background:\n\n    >>> from scipy import ndimage as ndi\n    >>> distance = ndi.distance_transform_edt(image)\n    >>> from skimage.feature import peak_local_max\n    >>> max_coords = peak_local_max(distance, labels=image,\n    ...                             footprint=np.ones((3, 3)))\n    >>> local_maxima = np.zeros_like(image, dtype=bool)\n    >>> local_maxima[tuple(max_coords.T)] = True\n    >>> markers = ndi.label(local_maxima)[0]\n\n    Finally, we run the watershed on the image and markers:\n\n    >>> labels = watershed(-distance, markers, mask=image)\n\n    The algorithm works also for 3-D images, and can be used for example to\n    separate overlapping spheres.\n    '
    (image, markers, mask) = _validate_inputs(image, markers, mask, connectivity)
    (connectivity, offset) = _validate_connectivity(image.ndim, connectivity, offset)
    pad_width = [(p, p) for p in offset]
    image = np.pad(image, pad_width, mode='constant')
    mask = np.pad(mask, pad_width, mode='constant').ravel()
    output = np.pad(markers, pad_width, mode='constant')
    flat_neighborhood = _offsets_to_raveled_neighbors(image.shape, connectivity, center=offset)
    marker_locations = np.flatnonzero(output)
    image_strides = np.array(image.strides, dtype=np.intp) // image.itemsize
    _watershed_cy.watershed_raveled(image.ravel(), marker_locations, flat_neighborhood, mask, image_strides, compactness, output.ravel(), watershed_line)
    output = crop(output, pad_width, copy=True)
    return output