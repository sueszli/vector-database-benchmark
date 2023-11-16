import numpy as np
from .._shared.utils import _supported_float_type, _to_np_mode

def _validate_window_size(axis_sizes):
    if False:
        for i in range(10):
            print('nop')
    'Ensure all sizes in ``axis_sizes`` are odd.\n\n    Parameters\n    ----------\n    axis_sizes : iterable of int\n\n    Raises\n    ------\n    ValueError\n        If any given axis size is even.\n    '
    for axis_size in axis_sizes:
        if axis_size % 2 == 0:
            msg = f'Window size for `threshold_sauvola` or `threshold_niblack` must not be even on any dimension. Got {axis_sizes}'
            raise ValueError(msg)

def _get_view(padded, kernel_shape, idx, val):
    if False:
        for i in range(10):
            print('nop')
    'Get a view into `padded` that is offset by `idx` and scaled by `val`.\n\n    If `padded` was created by padding the original image by `kernel_shape` as\n    in correlate_sparse, then the view created here will match the size of the\n    original image.\n    '
    sl_shift = tuple([slice(c, s - (w_ - 1 - c)) for (c, w_, s) in zip(idx, kernel_shape, padded.shape)])
    v = padded[sl_shift]
    if val == 1:
        return v
    return val * v

def _correlate_sparse(image, kernel_shape, kernel_indices, kernel_values):
    if False:
        return 10
    "Perform correlation with a sparse kernel.\n\n    Parameters\n    ----------\n    image : ndarray\n        The (prepadded) image to be correlated.\n    kernel_shape : tuple of int\n        The shape of the sparse filter kernel.\n    kernel_indices : list of coordinate tuples\n        The indices of each non-zero kernel entry.\n    kernel_values : list of float\n        The kernel values at each location in kernel_indices.\n\n    Returns\n    -------\n    out : ndarray\n        The filtered image.\n\n    Notes\n    -----\n    This function only returns results for the 'valid' region of the\n    convolution, and thus `out` will be smaller than `image` by an amount\n    equal to the kernel size along each axis.\n    "
    (idx, val) = (kernel_indices[0], kernel_values[0])
    if tuple(idx) != (0,) * image.ndim:
        raise RuntimeError('Unexpected initial index in kernel_indices')
    out = _get_view(image, kernel_shape, idx, val).copy()
    for (idx, val) in zip(kernel_indices[1:], kernel_values[1:]):
        out += _get_view(image, kernel_shape, idx, val)
    return out

def correlate_sparse(image, kernel, mode='reflect'):
    if False:
        for i in range(10):
            print('nop')
    "Compute valid cross-correlation of `padded_array` and `kernel`.\n\n    This function is *fast* when `kernel` is large with many zeros.\n\n    See ``scipy.ndimage.correlate`` for a description of cross-correlation.\n\n    Parameters\n    ----------\n    image : ndarray, dtype float, shape (M, N[, ...], P)\n        The input array. If mode is 'valid', this array should already be\n        padded, as a margin of the same shape as kernel will be stripped\n        off.\n    kernel : ndarray, dtype float, shape (Q, R[, ...], S)\n        The kernel to be correlated. Must have the same number of\n        dimensions as `padded_array`. For high performance, it should\n        be sparse (few nonzero entries).\n    mode : string, optional\n        See `scipy.ndimage.correlate` for valid modes.\n        Additionally, mode 'valid' is accepted, in which case no padding is\n        applied and the result is the result for the smaller image for which\n        the kernel is entirely inside the original data.\n\n    Returns\n    -------\n    result : array of float, shape (M, N[, ...], P)\n        The result of cross-correlating `image` with `kernel`. If mode\n        'valid' is used, the resulting shape is (M-Q+1, N-R+1[, ...], P-S+1).\n    "
    kernel = np.asarray(kernel)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    if mode == 'valid':
        padded_image = image
    else:
        np_mode = _to_np_mode(mode)
        _validate_window_size(kernel.shape)
        padded_image = np.pad(image, [(w // 2, w // 2) for w in kernel.shape], mode=np_mode)
    indices = np.nonzero(kernel)
    values = list(kernel[indices].astype(float_dtype, copy=False))
    indices = list(zip(*indices))
    corner_index = (0,) * kernel.ndim
    if corner_index not in indices:
        indices = [corner_index] + indices
        values = [0.0] + values
    return _correlate_sparse(padded_image, kernel.shape, indices, values)