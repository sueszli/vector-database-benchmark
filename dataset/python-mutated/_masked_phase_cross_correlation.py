"""
Implementation of the masked normalized cross-correlation.

Based on the following publication:
D. Padfield. Masked object registration in the Fourier domain.
IEEE Transactions on Image Processing (2012)

and the author's original MATLAB implementation, available on this website:
http://www.dirkpadfield.com/
"""
from functools import partial
import numpy as np
import scipy.fft as fftmodule
from scipy.fft import next_fast_len
from .._shared.utils import _supported_float_type

def _masked_phase_cross_correlation(reference_image, moving_image, reference_mask, moving_mask=None, overlap_ratio=0.3):
    if False:
        i = 10
        return i + 15
    'Masked image translation registration by masked normalized\n    cross-correlation.\n\n    Parameters\n    ----------\n    reference_image : ndarray\n        Reference image.\n    moving_image : ndarray\n        Image to register. Must be same dimensionality as ``reference_image``,\n        but not necessarily the same size.\n    reference_mask : ndarray\n        Boolean mask for ``reference_image``. The mask should evaluate\n        to ``True`` (or 1) on valid pixels. ``reference_mask`` should\n        have the same shape as ``reference_image``.\n    moving_mask : ndarray or None, optional\n        Boolean mask for ``moving_image``. The mask should evaluate to ``True``\n        (or 1) on valid pixels. ``moving_mask`` should have the same shape\n        as ``moving_image``. If ``None``, ``reference_mask`` will be used.\n    overlap_ratio : float, optional\n        Minimum allowed overlap ratio between images. The correlation for\n        translations corresponding with an overlap ratio lower than this\n        threshold will be ignored. A lower `overlap_ratio` leads to smaller\n        maximum translation, while a higher `overlap_ratio` leads to greater\n        robustness against spurious matches due to small overlap between\n        masked images.\n\n    Returns\n    -------\n    shifts : ndarray\n        Shift vector (in pixels) required to register ``moving_image``\n        with ``reference_image``. Axis ordering is consistent with numpy.\n\n    References\n    ----------\n    .. [1] Dirk Padfield. Masked Object Registration in the Fourier Domain.\n           IEEE Transactions on Image Processing, vol. 21(5),\n           pp. 2706-2718 (2012). :DOI:`10.1109/TIP.2011.2181402`\n    .. [2] D. Padfield. "Masked FFT registration". In Proc. Computer Vision and\n           Pattern Recognition, pp. 2918-2925 (2010).\n           :DOI:`10.1109/CVPR.2010.5540032`\n\n    '
    if moving_mask is None:
        if reference_image.shape != moving_image.shape:
            raise ValueError('Input images have different shapes, moving_mask must be explicitly set.')
        moving_mask = reference_mask.astype(bool)
    for (im, mask) in [(reference_image, reference_mask), (moving_image, moving_mask)]:
        if im.shape != mask.shape:
            raise ValueError('Image sizes must match their respective mask sizes.')
    xcorr = cross_correlate_masked(moving_image, reference_image, moving_mask, reference_mask, axes=tuple(range(moving_image.ndim)), mode='full', overlap_ratio=overlap_ratio)
    maxima = np.stack(np.nonzero(xcorr == xcorr.max()), axis=1)
    center = np.mean(maxima, axis=0)
    shifts = center - np.array(reference_image.shape) + 1
    size_mismatch = np.array(moving_image.shape) - np.array(reference_image.shape)
    return -shifts + size_mismatch / 2

def cross_correlate_masked(arr1, arr2, m1, m2, mode='full', axes=(-2, -1), overlap_ratio=0.3):
    if False:
        print('Hello World!')
    '\n    Masked normalized cross-correlation between arrays.\n\n    Parameters\n    ----------\n    arr1 : ndarray\n        First array.\n    arr2 : ndarray\n        Seconds array. The dimensions of `arr2` along axes that are not\n        transformed should be equal to that of `arr1`.\n    m1 : ndarray\n        Mask of `arr1`. The mask should evaluate to `True`\n        (or 1) on valid pixels. `m1` should have the same shape as `arr1`.\n    m2 : ndarray\n        Mask of `arr2`. The mask should evaluate to `True`\n        (or 1) on valid pixels. `m2` should have the same shape as `arr2`.\n    mode : {\'full\', \'same\'}, optional\n        \'full\':\n            This returns the convolution at each point of overlap. At\n            the end-points of the convolution, the signals do not overlap\n            completely, and boundary effects may be seen.\n        \'same\':\n            The output is the same size as `arr1`, centered with respect\n            to the `‘full’` output. Boundary effects are less prominent.\n    axes : tuple of ints, optional\n        Axes along which to compute the cross-correlation.\n    overlap_ratio : float, optional\n        Minimum allowed overlap ratio between images. The correlation for\n        translations corresponding with an overlap ratio lower than this\n        threshold will be ignored. A lower `overlap_ratio` leads to smaller\n        maximum translation, while a higher `overlap_ratio` leads to greater\n        robustness against spurious matches due to small overlap between\n        masked images.\n\n    Returns\n    -------\n    out : ndarray\n        Masked normalized cross-correlation.\n\n    Raises\n    ------\n    ValueError : if correlation `mode` is not valid, or array dimensions along\n        non-transformation axes are not equal.\n\n    References\n    ----------\n    .. [1] Dirk Padfield. Masked Object Registration in the Fourier Domain.\n           IEEE Transactions on Image Processing, vol. 21(5),\n           pp. 2706-2718 (2012). :DOI:`10.1109/TIP.2011.2181402`\n    .. [2] D. Padfield. "Masked FFT registration". In Proc. Computer Vision and\n           Pattern Recognition, pp. 2918-2925 (2010).\n           :DOI:`10.1109/CVPR.2010.5540032`\n    '
    if mode not in {'full', 'same'}:
        raise ValueError(f"Correlation mode '{mode}' is not valid.")
    fixed_image = np.asarray(arr1)
    moving_image = np.asarray(arr2)
    float_dtype = _supported_float_type((fixed_image.dtype, moving_image.dtype))
    if float_dtype.kind == 'c':
        raise ValueError('complex-valued arr1, arr2 are not supported')
    fixed_image = fixed_image.astype(float_dtype)
    fixed_mask = np.array(m1, dtype=bool)
    moving_image = moving_image.astype(float_dtype)
    moving_mask = np.array(m2, dtype=bool)
    eps = np.finfo(float_dtype).eps
    all_axes = set(range(fixed_image.ndim))
    for axis in all_axes - set(axes):
        if fixed_image.shape[axis] != moving_image.shape[axis]:
            raise ValueError(f'Array shapes along non-transformation axes should be equal, but dimensions along axis {axis} are not.')
    final_shape = list(arr1.shape)
    for axis in axes:
        final_shape[axis] = fixed_image.shape[axis] + moving_image.shape[axis] - 1
    final_shape = tuple(final_shape)
    final_slice = tuple([slice(0, int(sz)) for sz in final_shape])
    fast_shape = tuple([next_fast_len(final_shape[ax]) for ax in axes])
    fft = partial(fftmodule.fftn, s=fast_shape, axes=axes)
    _ifft = partial(fftmodule.ifftn, s=fast_shape, axes=axes)

    def ifft(x):
        if False:
            print('Hello World!')
        return _ifft(x).real
    fixed_image[np.logical_not(fixed_mask)] = 0.0
    moving_image[np.logical_not(moving_mask)] = 0.0
    rotated_moving_image = _flip(moving_image, axes=axes)
    rotated_moving_mask = _flip(moving_mask, axes=axes)
    fixed_fft = fft(fixed_image)
    rotated_moving_fft = fft(rotated_moving_image)
    fixed_mask_fft = fft(fixed_mask.astype(float_dtype))
    rotated_moving_mask_fft = fft(rotated_moving_mask.astype(float_dtype))
    number_overlap_masked_px = ifft(rotated_moving_mask_fft * fixed_mask_fft)
    number_overlap_masked_px[:] = np.round(number_overlap_masked_px)
    number_overlap_masked_px[:] = np.fmax(number_overlap_masked_px, eps)
    masked_correlated_fixed_fft = ifft(rotated_moving_mask_fft * fixed_fft)
    masked_correlated_rotated_moving_fft = ifft(fixed_mask_fft * rotated_moving_fft)
    numerator = ifft(rotated_moving_fft * fixed_fft)
    numerator -= masked_correlated_fixed_fft * masked_correlated_rotated_moving_fft / number_overlap_masked_px
    fixed_squared_fft = fft(np.square(fixed_image))
    fixed_denom = ifft(rotated_moving_mask_fft * fixed_squared_fft)
    fixed_denom -= np.square(masked_correlated_fixed_fft) / number_overlap_masked_px
    fixed_denom[:] = np.fmax(fixed_denom, 0.0)
    rotated_moving_squared_fft = fft(np.square(rotated_moving_image))
    moving_denom = ifft(fixed_mask_fft * rotated_moving_squared_fft)
    moving_denom -= np.square(masked_correlated_rotated_moving_fft) / number_overlap_masked_px
    moving_denom[:] = np.fmax(moving_denom, 0.0)
    denom = np.sqrt(fixed_denom * moving_denom)
    numerator = numerator[final_slice]
    denom = denom[final_slice]
    number_overlap_masked_px = number_overlap_masked_px[final_slice]
    if mode == 'same':
        _centering = partial(_centered, newshape=fixed_image.shape, axes=axes)
        denom = _centering(denom)
        numerator = _centering(numerator)
        number_overlap_masked_px = _centering(number_overlap_masked_px)
    tol = 1000.0 * eps * np.max(np.abs(denom), axis=axes, keepdims=True)
    nonzero_indices = denom > tol
    out = np.zeros_like(denom, dtype=float_dtype)
    out[nonzero_indices] = numerator[nonzero_indices] / denom[nonzero_indices]
    np.clip(out, a_min=-1, a_max=1, out=out)
    number_px_threshold = overlap_ratio * np.max(number_overlap_masked_px, axis=axes, keepdims=True)
    out[number_overlap_masked_px < number_px_threshold] = 0.0
    return out

def _centered(arr, newshape, axes):
    if False:
        while True:
            i = 10
    'Return the center `newshape` portion of `arr`, leaving axes not\n    in `axes` untouched.'
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    slices = [slice(None, None)] * arr.ndim
    for ax in axes:
        startind = (currshape[ax] - newshape[ax]) // 2
        endind = startind + newshape[ax]
        slices[ax] = slice(startind, endind)
    return arr[tuple(slices)]

def _flip(arr, axes=None):
    if False:
        return 10
    'Reverse array over many axes. Generalization of arr[::-1] for many\n    dimensions. If `axes` is `None`, flip along all axes.'
    if axes is None:
        reverse = [slice(None, None, -1)] * arr.ndim
    else:
        reverse = [slice(None, None, None)] * arr.ndim
        for axis in axes:
            reverse[axis] = slice(None, None, -1)
    return arr[tuple(reverse)]