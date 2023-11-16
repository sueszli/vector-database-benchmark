"""
The arraycrop module contains functions to crop values from the edges of an
n-dimensional array.
"""
import numpy as np
from numbers import Integral
__all__ = ['crop']

def crop(ar, crop_width, copy=False, order='K'):
    if False:
        print('Hello World!')
    "Crop array `ar` by `crop_width` along each dimension.\n\n    Parameters\n    ----------\n    ar : array-like of rank N\n        Input array.\n    crop_width : {sequence, int}\n        Number of values to remove from the edges of each axis.\n        ``((before_1, after_1),`` ... ``(before_N, after_N))`` specifies\n        unique crop widths at the start and end of each axis.\n        ``((before, after),) or (before, after)`` specifies\n        a fixed start and end crop for every axis.\n        ``(n,)`` or ``n`` for integer ``n`` is a shortcut for\n        before = after = ``n`` for all axes.\n    copy : bool, optional\n        If `True`, ensure the returned array is a contiguous copy. Normally,\n        a crop operation will return a discontiguous view of the underlying\n        input array.\n    order : {'C', 'F', 'A', 'K'}, optional\n        If ``copy==True``, control the memory layout of the copy. See\n        ``np.copy``.\n\n    Returns\n    -------\n    cropped : array\n        The cropped array. If ``copy=False`` (default), this is a sliced\n        view of the input array.\n    "
    ar = np.array(ar, copy=False)
    if isinstance(crop_width, Integral):
        crops = [[crop_width, crop_width]] * ar.ndim
    elif isinstance(crop_width[0], Integral):
        if len(crop_width) == 1:
            crops = [[crop_width[0], crop_width[0]]] * ar.ndim
        elif len(crop_width) == 2:
            crops = [crop_width] * ar.ndim
        else:
            raise ValueError(f'crop_width has an invalid length: {len(crop_width)}\ncrop_width should be a sequence of N pairs, a single pair, or a single integer')
    elif len(crop_width) == 1:
        crops = [crop_width[0]] * ar.ndim
    elif len(crop_width) == ar.ndim:
        crops = crop_width
    else:
        raise ValueError(f'crop_width has an invalid length: {len(crop_width)}\ncrop_width should be a sequence of N pairs, a single pair, or a single integer')
    slices = tuple((slice(a, ar.shape[i] - b) for (i, (a, b)) in enumerate(crops)))
    if copy:
        cropped = np.array(ar[slices], order=order, copy=True)
    else:
        cropped = ar[slices]
    return cropped