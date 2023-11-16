from warnings import warn
import numpy as np
import scipy.ndimage as ndi
from .. import measure
from .._shared.coord import ensure_spacing

def _get_high_intensity_peaks(image, mask, num_peaks, min_distance, p_norm):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the highest intensity peak coordinates.\n    '
    coord = np.nonzero(mask)
    intensities = image[coord]
    idx_maxsort = np.argsort(-intensities, kind='stable')
    coord = np.transpose(coord)[idx_maxsort]
    if np.isfinite(num_peaks):
        max_out = int(num_peaks)
    else:
        max_out = None
    coord = ensure_spacing(coord, spacing=min_distance, p_norm=p_norm, max_out=max_out)
    if len(coord) > num_peaks:
        coord = coord[:num_peaks]
    return coord

def _get_peak_mask(image, footprint, threshold, mask=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the mask containing all peak candidates above thresholds.\n    '
    if footprint.size == 1 or image.size == 1:
        return image > threshold
    image_max = ndi.maximum_filter(image, footprint=footprint, mode='nearest')
    out = image == image_max
    image_is_trivial = np.all(out) if mask is None else np.all(out[mask])
    if image_is_trivial:
        out[:] = False
        if mask is not None:
            isolated_px = np.logical_xor(mask, ndi.binary_opening(mask))
            out[isolated_px] = True
    out &= image > threshold
    return out

def _exclude_border(label, border_width):
    if False:
        while True:
            i = 10
    'Set label border values to 0.'
    for (i, width) in enumerate(border_width):
        if width == 0:
            continue
        label[(slice(None),) * i + (slice(None, width),)] = 0
        label[(slice(None),) * i + (slice(-width, None),)] = 0
    return label

def _get_threshold(image, threshold_abs, threshold_rel):
    if False:
        i = 10
        return i + 15
    'Return the threshold value according to an absolute and a relative\n    value.\n\n    '
    threshold = threshold_abs if threshold_abs is not None else image.min()
    if threshold_rel is not None:
        threshold = max(threshold, threshold_rel * image.max())
    return threshold

def _get_excluded_border_width(image, min_distance, exclude_border):
    if False:
        print('Hello World!')
    'Return border_width values relative to a min_distance if requested.'
    if isinstance(exclude_border, bool):
        border_width = (min_distance if exclude_border else 0,) * image.ndim
    elif isinstance(exclude_border, int):
        if exclude_border < 0:
            raise ValueError('`exclude_border` cannot be a negative value')
        border_width = (exclude_border,) * image.ndim
    elif isinstance(exclude_border, tuple):
        if len(exclude_border) != image.ndim:
            raise ValueError('`exclude_border` should have the same length as the dimensionality of the image.')
        for exclude in exclude_border:
            if not isinstance(exclude, int):
                raise ValueError('`exclude_border`, when expressed as a tuple, must only contain ints.')
            if exclude < 0:
                raise ValueError('`exclude_border` can not be a negative value')
        border_width = exclude_border
    else:
        raise TypeError('`exclude_border` must be bool, int, or tuple with the same length as the dimensionality of the image.')
    return border_width

def peak_local_max(image, min_distance=1, threshold_abs=None, threshold_rel=None, exclude_border=True, num_peaks=np.inf, footprint=None, labels=None, num_peaks_per_label=np.inf, p_norm=np.inf):
    if False:
        for i in range(10):
            print('nop')
    "Find peaks in an image as coordinate list.\n\n    Peaks are the local maxima in a region of `2 * min_distance + 1`\n    (i.e. peaks are separated by at least `min_distance`).\n\n    If both `threshold_abs` and `threshold_rel` are provided, the maximum\n    of the two is chosen as the minimum intensity threshold of peaks.\n\n    .. versionchanged:: 0.18\n        Prior to version 0.18, peaks of the same height within a radius of\n        `min_distance` were all returned, but this could cause unexpected\n        behaviour. From 0.18 onwards, an arbitrary peak within the region is\n        returned. See issue gh-2592.\n\n    Parameters\n    ----------\n    image : ndarray\n        Input image.\n    min_distance : int, optional\n        The minimal allowed distance separating peaks. To find the\n        maximum number of peaks, use `min_distance=1`.\n    threshold_abs : float or None, optional\n        Minimum intensity of peaks. By default, the absolute threshold is\n        the minimum intensity of the image.\n    threshold_rel : float or None, optional\n        Minimum intensity of peaks, calculated as\n        ``max(image) * threshold_rel``.\n    exclude_border : int, tuple of ints, or bool, optional\n        If positive integer, `exclude_border` excludes peaks from within\n        `exclude_border`-pixels of the border of the image.\n        If tuple of non-negative ints, the length of the tuple must match the\n        input array's dimensionality.  Each element of the tuple will exclude\n        peaks from within `exclude_border`-pixels of the border of the image\n        along that dimension.\n        If True, takes the `min_distance` parameter as value.\n        If zero or False, peaks are identified regardless of their distance\n        from the border.\n    num_peaks : int, optional\n        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,\n        return `num_peaks` peaks based on highest peak intensity.\n    footprint : ndarray of bools, optional\n        If provided, `footprint == 1` represents the local region within which\n        to search for peaks at every point in `image`.\n    labels : ndarray of ints, optional\n        If provided, each unique region `labels == value` represents a unique\n        region to search for peaks. Zero is reserved for background.\n    num_peaks_per_label : int, optional\n        Maximum number of peaks for each label.\n    p_norm : float\n        Which Minkowski p-norm to use. Should be in the range [1, inf].\n        A finite large p may cause a ValueError if overflow can occur.\n        ``inf`` corresponds to the Chebyshev distance and 2 to the\n        Euclidean distance.\n\n    Returns\n    -------\n    output : ndarray\n        The coordinates of the peaks.\n\n    Notes\n    -----\n    The peak local maximum function returns the coordinates of local peaks\n    (maxima) in an image. Internally, a maximum filter is used for finding\n    local maxima. This operation dilates the original image. After comparison\n    of the dilated and original images, this function returns the coordinates\n    of the peaks where the dilated image equals the original image.\n\n    See also\n    --------\n    skimage.feature.corner_peaks\n\n    Examples\n    --------\n    >>> img1 = np.zeros((7, 7))\n    >>> img1[3, 4] = 1\n    >>> img1[3, 2] = 1.5\n    >>> img1\n    array([[0. , 0. , 0. , 0. , 0. , 0. , 0. ],\n           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],\n           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],\n           [0. , 0. , 1.5, 0. , 1. , 0. , 0. ],\n           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],\n           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],\n           [0. , 0. , 0. , 0. , 0. , 0. , 0. ]])\n\n    >>> peak_local_max(img1, min_distance=1)\n    array([[3, 2],\n           [3, 4]])\n\n    >>> peak_local_max(img1, min_distance=2)\n    array([[3, 2]])\n\n    >>> img2 = np.zeros((20, 20, 20))\n    >>> img2[10, 10, 10] = 1\n    >>> img2[15, 15, 15] = 1\n    >>> peak_idx = peak_local_max(img2, exclude_border=0)\n    >>> peak_idx\n    array([[10, 10, 10],\n           [15, 15, 15]])\n\n    >>> peak_mask = np.zeros_like(img2, dtype=bool)\n    >>> peak_mask[tuple(peak_idx.T)] = True\n    >>> np.argwhere(peak_mask)\n    array([[10, 10, 10],\n           [15, 15, 15]])\n\n    "
    if (footprint is None or footprint.size == 1) and min_distance < 1:
        warn('When min_distance < 1, peak_local_max acts as finding image > max(threshold_abs, threshold_rel * max(image)).', RuntimeWarning, stacklevel=2)
    border_width = _get_excluded_border_width(image, min_distance, exclude_border)
    threshold = _get_threshold(image, threshold_abs, threshold_rel)
    if footprint is None:
        size = 2 * min_distance + 1
        footprint = np.ones((size,) * image.ndim, dtype=bool)
    else:
        footprint = np.asarray(footprint)
    if labels is None:
        mask = _get_peak_mask(image, footprint, threshold)
        mask = _exclude_border(mask, border_width)
        coordinates = _get_high_intensity_peaks(image, mask, num_peaks, min_distance, p_norm)
    else:
        _labels = _exclude_border(labels.astype(int, casting='safe'), border_width)
        if np.issubdtype(image.dtype, np.floating):
            bg_val = np.finfo(image.dtype).min
        else:
            bg_val = np.iinfo(image.dtype).min
        labels_peak_coord = []
        for (label_idx, roi) in enumerate(ndi.find_objects(_labels)):
            if roi is None:
                continue
            label_mask = labels[roi] == label_idx + 1
            img_object = image[roi].copy()
            img_object[np.logical_not(label_mask)] = bg_val
            mask = _get_peak_mask(img_object, footprint, threshold, label_mask)
            coordinates = _get_high_intensity_peaks(img_object, mask, num_peaks_per_label, min_distance, p_norm)
            for (idx, s) in enumerate(roi):
                coordinates[:, idx] += s.start
            labels_peak_coord.append(coordinates)
        if labels_peak_coord:
            coordinates = np.vstack(labels_peak_coord)
        else:
            coordinates = np.empty((0, 2), dtype=int)
        if len(coordinates) > num_peaks:
            out = np.zeros_like(image, dtype=bool)
            out[tuple(coordinates.T)] = True
            coordinates = _get_high_intensity_peaks(image, out, num_peaks, min_distance, p_norm)
    return coordinates

def _prominent_peaks(image, min_xdistance=1, min_ydistance=1, threshold=None, num_peaks=np.inf):
    if False:
        return 10
    'Return peaks with non-maximum suppression.\n\n    Identifies most prominent features separated by certain distances.\n    Non-maximum suppression with different sizes is applied separately\n    in the first and second dimension of the image to identify peaks.\n\n    Parameters\n    ----------\n    image : (M, N) ndarray\n        Input image.\n    min_xdistance : int\n        Minimum distance separating features in the x dimension.\n    min_ydistance : int\n        Minimum distance separating features in the y dimension.\n    threshold : float\n        Minimum intensity of peaks. Default is `0.5 * max(image)`.\n    num_peaks : int\n        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,\n        return `num_peaks` coordinates based on peak intensity.\n\n    Returns\n    -------\n    intensity, xcoords, ycoords : tuple of array\n        Peak intensity values, x and y indices.\n    '
    img = image.copy()
    (rows, cols) = img.shape
    if threshold is None:
        threshold = 0.5 * np.max(img)
    ycoords_size = 2 * min_ydistance + 1
    xcoords_size = 2 * min_xdistance + 1
    img_max = ndi.maximum_filter1d(img, size=ycoords_size, axis=0, mode='constant', cval=0)
    img_max = ndi.maximum_filter1d(img_max, size=xcoords_size, axis=1, mode='constant', cval=0)
    mask = img == img_max
    img *= mask
    img_t = img > threshold
    label_img = measure.label(img_t)
    props = measure.regionprops(label_img, img_max)
    props = sorted(props, key=lambda x: x.intensity_max)[::-1]
    coords = np.array([np.round(p.centroid) for p in props], dtype=int)
    img_peaks = []
    ycoords_peaks = []
    xcoords_peaks = []
    (ycoords_ext, xcoords_ext) = np.mgrid[-min_ydistance:min_ydistance + 1, -min_xdistance:min_xdistance + 1]
    for (ycoords_idx, xcoords_idx) in coords:
        accum = img_max[ycoords_idx, xcoords_idx]
        if accum > threshold:
            ycoords_nh = ycoords_idx + ycoords_ext
            xcoords_nh = xcoords_idx + xcoords_ext
            ycoords_in = np.logical_and(ycoords_nh > 0, ycoords_nh < rows)
            ycoords_nh = ycoords_nh[ycoords_in]
            xcoords_nh = xcoords_nh[ycoords_in]
            xcoords_low = xcoords_nh < 0
            ycoords_nh[xcoords_low] = rows - ycoords_nh[xcoords_low]
            xcoords_nh[xcoords_low] += cols
            xcoords_high = xcoords_nh >= cols
            ycoords_nh[xcoords_high] = rows - ycoords_nh[xcoords_high]
            xcoords_nh[xcoords_high] -= cols
            img_max[ycoords_nh, xcoords_nh] = 0
            img_peaks.append(accum)
            ycoords_peaks.append(ycoords_idx)
            xcoords_peaks.append(xcoords_idx)
    img_peaks = np.array(img_peaks)
    ycoords_peaks = np.array(ycoords_peaks)
    xcoords_peaks = np.array(xcoords_peaks)
    if num_peaks < len(img_peaks):
        idx_maxsort = np.argsort(img_peaks)[::-1][:num_peaks]
        img_peaks = img_peaks[idx_maxsort]
        ycoords_peaks = ycoords_peaks[idx_maxsort]
        xcoords_peaks = xcoords_peaks[idx_maxsort]
    return (img_peaks, xcoords_peaks, ycoords_peaks)