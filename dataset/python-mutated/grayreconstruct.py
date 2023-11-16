import numpy as np
from .._shared.utils import _supported_float_type
from ..filters._rank_order import rank_order
from ._grayreconstruct import reconstruction_loop

def reconstruction(seed, mask, method='dilation', footprint=None, offset=None):
    if False:
        for i in range(10):
            print('nop')
    'Perform a morphological reconstruction of an image.\n\n    Morphological reconstruction by dilation is similar to basic morphological\n    dilation: high-intensity values will replace nearby low-intensity values.\n    The basic dilation operator, however, uses a footprint to\n    determine how far a value in the input image can spread. In contrast,\n    reconstruction uses two images: a "seed" image, which specifies the values\n    that spread, and a "mask" image, which gives the maximum allowed value at\n    each pixel. The mask image, like the footprint, limits the spread\n    of high-intensity values. Reconstruction by erosion is simply the inverse:\n    low-intensity values spread from the seed image and are limited by the mask\n    image, which represents the minimum allowed value.\n\n    Alternatively, you can think of reconstruction as a way to isolate the\n    connected regions of an image. For dilation, reconstruction connects\n    regions marked by local maxima in the seed image: neighboring pixels\n    less-than-or-equal-to those seeds are connected to the seeded region.\n    Local maxima with values larger than the seed image will get truncated to\n    the seed value.\n\n    Parameters\n    ----------\n    seed : ndarray\n        The seed image (a.k.a. marker image), which specifies the values that\n        are dilated or eroded.\n    mask : ndarray\n        The maximum (dilation) / minimum (erosion) allowed value at each pixel.\n    method : {\'dilation\'|\'erosion\'}, optional\n        Perform reconstruction by dilation or erosion. In dilation (or\n        erosion), the seed image is dilated (or eroded) until limited by the\n        mask image. For dilation, each seed value must be less than or equal\n        to the corresponding mask value; for erosion, the reverse is true.\n        Default is \'dilation\'.\n    footprint : ndarray, optional\n        The neighborhood expressed as an n-D array of 1\'s and 0\'s.\n        Default is the n-D square of radius equal to 1 (i.e. a 3x3 square\n        for 2D images, a 3x3x3 cube for 3D images, etc.)\n    offset : ndarray, optional\n        The coordinates of the center of the footprint.\n        Default is located on the geometrical center of the footprint, in that\n        case footprint dimensions must be odd.\n\n    Returns\n    -------\n    reconstructed : ndarray\n        The result of morphological reconstruction.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from skimage.morphology import reconstruction\n\n    First, we create a sinusoidal mask image with peaks at middle and ends.\n\n    >>> x = np.linspace(0, 4 * np.pi)\n    >>> y_mask = np.cos(x)\n\n    Then, we create a seed image initialized to the minimum mask value (for\n    reconstruction by dilation, min-intensity values don\'t spread) and add\n    "seeds" to the left and right peak, but at a fraction of peak value (1).\n\n    >>> y_seed = y_mask.min() * np.ones_like(x)\n    >>> y_seed[0] = 0.5\n    >>> y_seed[-1] = 0\n    >>> y_rec = reconstruction(y_seed, y_mask)\n\n    The reconstructed image (or curve, in this case) is exactly the same as the\n    mask image, except that the peaks are truncated to 0.5 and 0. The middle\n    peak disappears completely: Since there were no seed values in this peak\n    region, its reconstructed value is truncated to the surrounding value (-1).\n\n    As a more practical example, we try to extract the bright features of an\n    image by subtracting a background image created by reconstruction.\n\n    >>> y, x = np.mgrid[:20:0.5, :20:0.5]\n    >>> bumps = np.sin(x) + np.sin(y)\n\n    To create the background image, set the mask image to the original image,\n    and the seed image to the original image with an intensity offset, `h`.\n\n    >>> h = 0.3\n    >>> seed = bumps - h\n    >>> background = reconstruction(seed, bumps)\n\n    The resulting reconstructed image looks exactly like the original image,\n    but with the peaks of the bumps cut off. Subtracting this reconstructed\n    image from the original image leaves just the peaks of the bumps\n\n    >>> hdome = bumps - background\n\n    This operation is known as the h-dome of the image and leaves features\n    of height `h` in the subtracted image.\n\n    Notes\n    -----\n    The algorithm is taken from [1]_. Applications for grayscale reconstruction\n    are discussed in [2]_ and [3]_.\n\n    References\n    ----------\n    .. [1] Robinson, "Efficient morphological reconstruction: a downhill\n           filter", Pattern Recognition Letters 25 (2004) 1759-1767.\n    .. [2] Vincent, L., "Morphological Grayscale Reconstruction in Image\n           Analysis: Applications and Efficient Algorithms", IEEE Transactions\n           on Image Processing (1993)\n    .. [3] Soille, P., "Morphological Image Analysis: Principles and\n           Applications", Chapter 6, 2nd edition (2003), ISBN 3540429883.\n    '
    assert tuple(seed.shape) == tuple(mask.shape)
    if method == 'dilation' and np.any(seed > mask):
        raise ValueError('Intensity of seed image must be less than that of the mask image for reconstruction by dilation.')
    elif method == 'erosion' and np.any(seed < mask):
        raise ValueError('Intensity of seed image must be greater than that of the mask image for reconstruction by erosion.')
    if footprint is None:
        footprint = np.ones([3] * seed.ndim, dtype=bool)
    else:
        footprint = footprint.astype(bool, copy=True)
    if offset is None:
        if not all([d % 2 == 1 for d in footprint.shape]):
            raise ValueError('Footprint dimensions must all be odd')
        offset = np.array([d // 2 for d in footprint.shape])
    else:
        if offset.ndim != footprint.ndim:
            raise ValueError('Offset and footprint ndims must be equal.')
        if not all([0 <= o < d for (o, d) in zip(offset, footprint.shape)]):
            raise ValueError('Offset must be included inside footprint')
    footprint[tuple((slice(d, d + 1) for d in offset))] = False
    dims = np.zeros(seed.ndim + 1, dtype=int)
    dims[1:] = np.array(seed.shape) + (np.array(footprint.shape) - 1)
    dims[0] = 2
    inside_slices = tuple((slice(o, o + s) for (o, s) in zip(offset, seed.shape)))
    if method == 'dilation':
        pad_value = np.min(seed)
    elif method == 'erosion':
        pad_value = np.max(seed)
    else:
        raise ValueError(f"Reconstruction method can be one of 'erosion' or 'dilation'. Got '{method}'.")
    float_dtype = _supported_float_type(mask.dtype)
    images = np.full(dims, pad_value, dtype=float_dtype)
    images[(0, *inside_slices)] = seed
    images[(1, *inside_slices)] = mask
    isize = images.size
    signed_int_dtype = np.result_type(np.min_scalar_type(-isize), np.int32)
    unsigned_int_dtype = np.dtype(signed_int_dtype.char.upper())
    value_stride = np.array(images.strides[1:]) // images.dtype.itemsize
    image_stride = images.strides[0] // images.dtype.itemsize
    footprint_mgrid = np.mgrid[[slice(-o, d - o) for (d, o) in zip(footprint.shape, offset)]]
    footprint_offsets = footprint_mgrid[:, footprint].transpose()
    nb_strides = np.array([np.sum(value_stride * footprint_offset) for footprint_offset in footprint_offsets], signed_int_dtype)
    images = images.reshape(-1)
    index_sorted = np.argsort(images).astype(signed_int_dtype, copy=False)
    if method == 'dilation':
        index_sorted = index_sorted[::-1]
    prev = np.full(isize, -1, signed_int_dtype)
    next = np.full(isize, -1, signed_int_dtype)
    prev[index_sorted[1:]] = index_sorted[:-1]
    next[index_sorted[:-1]] = index_sorted[1:]
    if method == 'dilation':
        (value_rank, value_map) = rank_order(images)
    elif method == 'erosion':
        (value_rank, value_map) = rank_order(-images)
        value_map = -value_map
    start = index_sorted[0]
    value_rank = value_rank.astype(unsigned_int_dtype, copy=False)
    reconstruction_loop(value_rank, prev, next, nb_strides, start, image_stride)
    rec_img = value_map[value_rank[:image_stride]]
    rec_img.shape = np.array(seed.shape) + (np.array(footprint.shape) - 1)
    return rec_img[inside_slices]