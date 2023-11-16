"""Approximate bilateral rank filter for local (custom kernel) mean.

The local histogram is computed using a sliding window similar to the method
described in [1]_.

The pixel neighborhood is defined by:

* the given footprint (structuring element)
* an interval [g-s0, g+s1] in graylevel around g the processed pixel graylevel

The kernel is flat (i.e. each pixel belonging to the neighborhood contributes
equally).

Result image is 8-/16-bit or double with respect to the input image and the
rank filter operation.

References
----------

.. [1] Huang, T. ,Yang, G. ;  Tang, G.. "A fast two-dimensional
       median filtering algorithm", IEEE Transactions on Acoustics, Speech and
       Signal Processing, Feb 1979. Volume: 27 , Issue: 1, Page(s): 13 - 18.

"""
from ..._shared.utils import check_nD
from . import bilateral_cy
from .generic import _preprocess_input
__all__ = ['mean_bilateral', 'pop_bilateral', 'sum_bilateral']

def _apply(func, image, footprint, out, mask, shift_x, shift_y, s0, s1, out_dtype=None):
    if False:
        i = 10
        return i + 15
    check_nD(image, 2)
    (image, footprint, out, mask, n_bins) = _preprocess_input(image, footprint, out, mask, out_dtype)
    func(image, footprint, shift_x=shift_x, shift_y=shift_y, mask=mask, out=out, n_bins=n_bins, s0=s0, s1=s1)
    return out.reshape(out.shape[:2])

def mean_bilateral(image, footprint, out=None, mask=None, shift_x=False, shift_y=False, s0=10, s1=10):
    if False:
        while True:
            i = 10
    "Apply a flat kernel bilateral filter.\n\n    This is an edge-preserving and noise reducing denoising filter. It averages\n    pixels based on their spatial closeness and radiometric similarity.\n\n    Spatial closeness is measured by considering only the local pixel\n    neighborhood given by a footprint (structuring element).\n\n    Radiometric similarity is defined by the graylevel interval [g-s0, g+s1]\n    where g is the current pixel graylevel.\n\n    Only pixels belonging to the footprint and having a graylevel inside this\n    interval are averaged.\n\n    Parameters\n    ----------\n    image : 2-D array (uint8, uint16)\n        Input image.\n    footprint : 2-D array\n        The neighborhood expressed as a 2-D array of 1's and 0's.\n    out : 2-D array (same dtype as input)\n        If None, a new array is allocated.\n    mask : ndarray\n        Mask array that defines (>0) area of the image included in the local\n        neighborhood. If None, the complete image is used (default).\n    shift_x, shift_y : int\n        Offset added to the footprint center point. Shift is bounded to the\n        footprint sizes (center must be inside the given footprint).\n    s0, s1 : int\n        Define the [s0, s1] interval around the grayvalue of the center pixel\n        to be considered for computing the value.\n\n    Returns\n    -------\n    out : 2-D array (same dtype as input image)\n        Output image.\n\n    See also\n    --------\n    skimage.restoration.denoise_bilateral\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from skimage import data\n    >>> from skimage.morphology import disk\n    >>> from skimage.filters.rank import mean_bilateral\n    >>> img = data.camera().astype(np.uint16)\n    >>> bilat_img = mean_bilateral(img, disk(20), s0=10,s1=10)\n\n    "
    return _apply(bilateral_cy._mean, image, footprint, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y, s0=s0, s1=s1)

def pop_bilateral(image, footprint, out=None, mask=None, shift_x=False, shift_y=False, s0=10, s1=10):
    if False:
        i = 10
        return i + 15
    "Return the local number (population) of pixels.\n\n\n    The number of pixels is defined as the number of pixels which are included\n    in the footprint and the mask. Additionally pixels must have a graylevel\n    inside the interval [g-s0, g+s1] where g is the grayvalue of the center\n    pixel.\n\n    Parameters\n    ----------\n    image : 2-D array (uint8, uint16)\n        Input image.\n    footprint : 2-D array\n        The neighborhood expressed as a 2-D array of 1's and 0's.\n    out : 2-D array (same dtype as input)\n        If None, a new array is allocated.\n    mask : ndarray\n        Mask array that defines (>0) area of the image included in the local\n        neighborhood. If None, the complete image is used (default).\n    shift_x, shift_y : int\n        Offset added to the footprint center point. Shift is bounded to the\n        footprint sizes (center must be inside the given footprint).\n    s0, s1 : int\n        Define the [s0, s1] interval around the grayvalue of the center pixel\n        to be considered for computing the value.\n\n    Returns\n    -------\n    out : 2-D array (same dtype as input image)\n        Output image.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from skimage.morphology import square\n    >>> import skimage.filters.rank as rank\n    >>> img = 255 * np.array([[0, 0, 0, 0, 0],\n    ...                       [0, 1, 1, 1, 0],\n    ...                       [0, 1, 1, 1, 0],\n    ...                       [0, 1, 1, 1, 0],\n    ...                       [0, 0, 0, 0, 0]], dtype=np.uint16)\n    >>> rank.pop_bilateral(img, square(3), s0=10, s1=10)\n    array([[3, 4, 3, 4, 3],\n           [4, 4, 6, 4, 4],\n           [3, 6, 9, 6, 3],\n           [4, 4, 6, 4, 4],\n           [3, 4, 3, 4, 3]], dtype=uint16)\n\n    "
    return _apply(bilateral_cy._pop, image, footprint, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y, s0=s0, s1=s1)

def sum_bilateral(image, footprint, out=None, mask=None, shift_x=False, shift_y=False, s0=10, s1=10):
    if False:
        print('Hello World!')
    "Apply a flat kernel bilateral filter.\n\n    This is an edge-preserving and noise reducing denoising filter. It averages\n    pixels based on their spatial closeness and radiometric similarity.\n\n    Spatial closeness is measured by considering only the local pixel\n    neighborhood given by a footprint (structuring element).\n\n    Radiometric similarity is defined by the graylevel interval [g-s0, g+s1]\n    where g is the current pixel graylevel.\n\n    Only pixels belonging to the footprint AND having a graylevel inside this\n    interval are summed.\n\n    Note that the sum may overflow depending on the data type of the input\n    array.\n\n    Parameters\n    ----------\n    image : 2-D array (uint8, uint16)\n        Input image.\n    footprint : 2-D array\n        The neighborhood expressed as a 2-D array of 1's and 0's.\n    out : 2-D array (same dtype as input)\n        If None, a new array is allocated.\n    mask : ndarray\n        Mask array that defines (>0) area of the image included in the local\n        neighborhood. If None, the complete image is used (default).\n    shift_x, shift_y : int\n        Offset added to the footprint center point. Shift is bounded to the\n        footprint sizes (center must be inside the given footprint).\n    s0, s1 : int\n        Define the [s0, s1] interval around the grayvalue of the center pixel\n        to be considered for computing the value.\n\n    Returns\n    -------\n    out : 2-D array (same dtype as input image)\n        Output image.\n\n    See also\n    --------\n    skimage.restoration.denoise_bilateral\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from skimage import data\n    >>> from skimage.morphology import disk\n    >>> from skimage.filters.rank import sum_bilateral\n    >>> img = data.camera().astype(np.uint16)\n    >>> bilat_img = sum_bilateral(img, disk(10), s0=10, s1=10)\n\n    "
    return _apply(bilateral_cy._sum, image, footprint, out=out, mask=mask, shift_x=shift_x, shift_y=shift_y, s0=s0, s1=s1)