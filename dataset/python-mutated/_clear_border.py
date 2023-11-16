import numpy as np
from ..measure import label

def clear_border(labels, buffer_size=0, bgval=0, mask=None, *, out=None):
    if False:
        return 10
    'Clear objects connected to the label image border.\n\n    Parameters\n    ----------\n    labels : (M[, N[, ..., P]]) array of int or bool\n        Imaging data labels.\n    buffer_size : int, optional\n        The width of the border examined.  By default, only objects\n        that touch the outside of the image are removed.\n    bgval : float or int, optional\n        Cleared objects are set to this value.\n    mask : ndarray of bool, same shape as `image`, optional.\n        Image data mask. Objects in labels image overlapping with\n        False pixels of mask will be removed. If defined, the\n        argument buffer_size will be ignored.\n    out : ndarray\n        Array of the same shape as `labels`, into which the\n        output is placed. By default, a new array is created.\n\n    Returns\n    -------\n    out : (M[, N[, ..., P]]) array\n        Imaging data labels with cleared borders\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from skimage.segmentation import clear_border\n    >>> labels = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0],\n    ...                    [1, 1, 0, 0, 1, 0, 0, 1, 0],\n    ...                    [1, 1, 0, 1, 0, 1, 0, 0, 0],\n    ...                    [0, 0, 0, 1, 1, 1, 1, 0, 0],\n    ...                    [0, 1, 1, 1, 1, 1, 1, 1, 0],\n    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0]])\n    >>> clear_border(labels)\n    array([[0, 0, 0, 0, 0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 1, 0, 0, 0, 0],\n           [0, 0, 0, 1, 0, 1, 0, 0, 0],\n           [0, 0, 0, 1, 1, 1, 1, 0, 0],\n           [0, 1, 1, 1, 1, 1, 1, 1, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0]])\n    >>> mask = np.array([[0, 0, 1, 1, 1, 1, 1, 1, 1],\n    ...                  [0, 0, 1, 1, 1, 1, 1, 1, 1],\n    ...                  [1, 1, 1, 1, 1, 1, 1, 1, 1],\n    ...                  [1, 1, 1, 1, 1, 1, 1, 1, 1],\n    ...                  [1, 1, 1, 1, 1, 1, 1, 1, 1],\n    ...                  [1, 1, 1, 1, 1, 1, 1, 1, 1]]).astype(bool)\n    >>> clear_border(labels, mask=mask)\n    array([[0, 0, 0, 0, 0, 0, 0, 1, 0],\n           [0, 0, 0, 0, 1, 0, 0, 1, 0],\n           [0, 0, 0, 1, 0, 1, 0, 0, 0],\n           [0, 0, 0, 1, 1, 1, 1, 0, 0],\n           [0, 1, 1, 1, 1, 1, 1, 1, 0],\n           [0, 0, 0, 0, 0, 0, 0, 0, 0]])\n\n    '
    if any((buffer_size >= s for s in labels.shape)) and mask is None:
        raise ValueError('buffer size may not be greater than labels size')
    if out is None:
        out = labels.copy()
    if mask is not None:
        err_msg = f'labels and mask should have the same shape but are {out.shape} and {mask.shape}'
        if out.shape != mask.shape:
            raise (ValueError, err_msg)
        if mask.dtype != bool:
            raise TypeError('mask should be of type bool.')
        borders = ~mask
    else:
        borders = np.zeros_like(out, dtype=bool)
        ext = buffer_size + 1
        slstart = slice(ext)
        slend = slice(-ext, None)
        slices = [slice(None) for _ in out.shape]
        for d in range(out.ndim):
            slices[d] = slstart
            borders[tuple(slices)] = True
            slices[d] = slend
            borders[tuple(slices)] = True
            slices[d] = slice(None)
    (labels, number) = label(out, background=0, return_num=True)
    borders_indices = np.unique(labels[borders])
    indices = np.arange(number + 1)
    label_mask = np.isin(indices, borders_indices)
    mask = label_mask[labels.reshape(-1)].reshape(labels.shape)
    out[mask] = bgval
    return out