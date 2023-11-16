import numpy as np
from . import _hoghistogram
from .._shared import utils

def _hog_normalize_block(block, method, eps=1e-05):
    if False:
        print('Hello World!')
    if method == 'L1':
        out = block / (np.sum(np.abs(block)) + eps)
    elif method == 'L1-sqrt':
        out = np.sqrt(block / (np.sum(np.abs(block)) + eps))
    elif method == 'L2':
        out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
    elif method == 'L2-Hys':
        out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
        out = np.minimum(out, 0.2)
        out = out / np.sqrt(np.sum(out ** 2) + eps ** 2)
    else:
        raise ValueError('Selected block normalization method is invalid.')
    return out

def _hog_channel_gradient(channel):
    if False:
        while True:
            i = 10
    'Compute unnormalized gradient image along `row` and `col` axes.\n\n    Parameters\n    ----------\n    channel : (M, N) ndarray\n        Grayscale image or one of image channel.\n\n    Returns\n    -------\n    g_row, g_col : channel gradient along `row` and `col` axes correspondingly.\n    '
    g_row = np.empty(channel.shape, dtype=channel.dtype)
    g_row[0, :] = 0
    g_row[-1, :] = 0
    g_row[1:-1, :] = channel[2:, :] - channel[:-2, :]
    g_col = np.empty(channel.shape, dtype=channel.dtype)
    g_col[:, 0] = 0
    g_col[:, -1] = 0
    g_col[:, 1:-1] = channel[:, 2:] - channel[:, :-2]
    return (g_row, g_col)

@utils.channel_as_last_axis(multichannel_output=False)
def hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True, *, channel_axis=None):
    if False:
        i = 10
        return i + 15
    "Extract Histogram of Oriented Gradients (HOG) for a given image.\n\n    Compute a Histogram of Oriented Gradients (HOG) by\n\n        1. (optional) global image normalization\n        2. computing the gradient image in `row` and `col`\n        3. computing gradient histograms\n        4. normalizing across blocks\n        5. flattening into a feature vector\n\n    Parameters\n    ----------\n    image : (M, N[, C]) ndarray\n        Input image.\n    orientations : int, optional\n        Number of orientation bins.\n    pixels_per_cell : 2-tuple (int, int), optional\n        Size (in pixels) of a cell.\n    cells_per_block : 2-tuple (int, int), optional\n        Number of cells in each block.\n    block_norm : str {'L1', 'L1-sqrt', 'L2', 'L2-Hys'}, optional\n        Block normalization method:\n\n        ``L1``\n           Normalization using L1-norm.\n        ``L1-sqrt``\n           Normalization using L1-norm, followed by square root.\n        ``L2``\n           Normalization using L2-norm.\n        ``L2-Hys``\n           Normalization using L2-norm, followed by limiting the\n           maximum values to 0.2 (`Hys` stands for `hysteresis`) and\n           renormalization using L2-norm. (default)\n           For details, see [3]_, [4]_.\n\n    visualize : bool, optional\n        Also return an image of the HOG.  For each cell and orientation bin,\n        the image contains a line segment that is centered at the cell center,\n        is perpendicular to the midpoint of the range of angles spanned by the\n        orientation bin, and has intensity proportional to the corresponding\n        histogram value.\n    transform_sqrt : bool, optional\n        Apply power law compression to normalize the image before\n        processing. DO NOT use this if the image contains negative\n        values. Also see `notes` section below.\n    feature_vector : bool, optional\n        Return the data as a feature vector by calling .ravel() on the result\n        just before returning.\n    channel_axis : int or None, optional\n        If None, the image is assumed to be a grayscale (single channel) image.\n        Otherwise, this parameter indicates which axis of the array corresponds\n        to channels.\n\n        .. versionadded:: 0.19\n           `channel_axis` was added in 0.19.\n\n    Returns\n    -------\n    out : (n_blocks_row, n_blocks_col, n_cells_row, n_cells_col, n_orient) ndarray\n        HOG descriptor for the image. If `feature_vector` is True, a 1D\n        (flattened) array is returned.\n    hog_image : (M, N) ndarray, optional\n        A visualisation of the HOG image. Only provided if `visualize` is True.\n\n    Raises\n    ------\n    ValueError\n        If the image is too small given the values of pixels_per_cell and\n        cells_per_block.\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients\n\n    .. [2] Dalal, N and Triggs, B, Histograms of Oriented Gradients for\n           Human Detection, IEEE Computer Society Conference on Computer\n           Vision and Pattern Recognition 2005 San Diego, CA, USA,\n           https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf,\n           :DOI:`10.1109/CVPR.2005.177`\n\n    .. [3] Lowe, D.G., Distinctive image features from scale-invatiant\n           keypoints, International Journal of Computer Vision (2004) 60: 91,\n           http://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf,\n           :DOI:`10.1023/B:VISI.0000029664.99615.94`\n\n    .. [4] Dalal, N, Finding People in Images and Videos,\n           Human-Computer Interaction [cs.HC], Institut National Polytechnique\n           de Grenoble - INPG, 2006,\n           https://tel.archives-ouvertes.fr/tel-00390303/file/NavneetDalalThesis.pdf\n\n    Notes\n    -----\n    The presented code implements the HOG extraction method from [2]_ with\n    the following changes: (I) blocks of (3, 3) cells are used ((2, 2) in the\n    paper); (II) no smoothing within cells (Gaussian spatial window with sigma=8pix\n    in the paper); (III) L1 block normalization is used (L2-Hys in the paper).\n\n    Power law compression, also known as Gamma correction, is used to reduce\n    the effects of shadowing and illumination variations. The compression makes\n    the dark regions lighter. When the kwarg `transform_sqrt` is set to\n    ``True``, the function computes the square root of each color channel\n    and then applies the hog algorithm to the image.\n    "
    image = np.atleast_2d(image)
    float_dtype = utils._supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    multichannel = channel_axis is not None
    ndim_spatial = image.ndim - 1 if multichannel else image.ndim
    if ndim_spatial != 2:
        raise ValueError('Only images with two spatial dimensions are supported. If using with color/multichannel images, specify `channel_axis`.')
    '\n    The first stage applies an optional global image normalization\n    equalisation that is designed to reduce the influence of illumination\n    effects. In practice we use gamma (power law) compression, either\n    computing the square root or the log of each color channel.\n    Image texture strength is typically proportional to the local surface\n    illumination so this compression helps to reduce the effects of local\n    shadowing and illumination variations.\n    '
    if transform_sqrt:
        image = np.sqrt(image)
    '\n    The second stage computes first order image gradients. These capture\n    contour, silhouette and some texture information, while providing\n    further resistance to illumination variations. The locally dominant\n    color channel is used, which provides color invariance to a large\n    extent. Variant methods may also include second order image derivatives,\n    which act as primitive bar detectors - a useful feature for capturing,\n    e.g. bar like structures in bicycles and limbs in humans.\n    '
    if multichannel:
        g_row_by_ch = np.empty_like(image, dtype=float_dtype)
        g_col_by_ch = np.empty_like(image, dtype=float_dtype)
        g_magn = np.empty_like(image, dtype=float_dtype)
        for idx_ch in range(image.shape[2]):
            (g_row_by_ch[:, :, idx_ch], g_col_by_ch[:, :, idx_ch]) = _hog_channel_gradient(image[:, :, idx_ch])
            g_magn[:, :, idx_ch] = np.hypot(g_row_by_ch[:, :, idx_ch], g_col_by_ch[:, :, idx_ch])
        idcs_max = g_magn.argmax(axis=2)
        (rr, cc) = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij', sparse=True)
        g_row = g_row_by_ch[rr, cc, idcs_max]
        g_col = g_col_by_ch[rr, cc, idcs_max]
    else:
        (g_row, g_col) = _hog_channel_gradient(image)
    '\n    The third stage aims to produce an encoding that is sensitive to\n    local image content while remaining resistant to small changes in\n    pose or appearance. The adopted method pools gradient orientation\n    information locally in the same way as the SIFT [Lowe 2004]\n    feature. The image window is divided into small spatial regions,\n    called "cells". For each cell we accumulate a local 1-D histogram\n    of gradient or edge orientations over all the pixels in the\n    cell. This combined cell-level 1-D histogram forms the basic\n    "orientation histogram" representation. Each orientation histogram\n    divides the gradient angle range into a fixed number of\n    predetermined bins. The gradient magnitudes of the pixels in the\n    cell are used to vote into the orientation histogram.\n    '
    (s_row, s_col) = image.shape[:2]
    (c_row, c_col) = pixels_per_cell
    (b_row, b_col) = cells_per_block
    n_cells_row = int(s_row // c_row)
    n_cells_col = int(s_col // c_col)
    orientation_histogram = np.zeros((n_cells_row, n_cells_col, orientations), dtype=float)
    g_row = g_row.astype(float, copy=False)
    g_col = g_col.astype(float, copy=False)
    _hoghistogram.hog_histograms(g_col, g_row, c_col, c_row, s_col, s_row, n_cells_col, n_cells_row, orientations, orientation_histogram)
    hog_image = None
    if visualize:
        from .. import draw
        radius = min(c_row, c_col) // 2 - 1
        orientations_arr = np.arange(orientations)
        orientation_bin_midpoints = np.pi * (orientations_arr + 0.5) / orientations
        dr_arr = radius * np.sin(orientation_bin_midpoints)
        dc_arr = radius * np.cos(orientation_bin_midpoints)
        hog_image = np.zeros((s_row, s_col), dtype=float_dtype)
        for r in range(n_cells_row):
            for c in range(n_cells_col):
                for (o, dr, dc) in zip(orientations_arr, dr_arr, dc_arr):
                    centre = tuple([r * c_row + c_row // 2, c * c_col + c_col // 2])
                    (rr, cc) = draw.line(int(centre[0] - dc), int(centre[1] + dr), int(centre[0] + dc), int(centre[1] - dr))
                    hog_image[rr, cc] += orientation_histogram[r, c, o]
    '\n    The fourth stage computes normalization, which takes local groups of\n    cells and contrast normalizes their overall responses before passing\n    to next stage. Normalization introduces better invariance to illumination,\n    shadowing, and edge contrast. It is performed by accumulating a measure\n    of local histogram "energy" over local groups of cells that we call\n    "blocks". The result is used to normalize each cell in the block.\n    Typically each individual cell is shared between several blocks, but\n    its normalizations are block dependent and thus different. The cell\n    thus appears several times in the final output vector with different\n    normalizations. This may seem redundant but it improves the performance.\n    We refer to the normalized block descriptors as Histogram of Oriented\n    Gradient (HOG) descriptors.\n    '
    n_blocks_row = n_cells_row - b_row + 1
    n_blocks_col = n_cells_col - b_col + 1
    if n_blocks_col <= 0 or n_blocks_row <= 0:
        min_row = b_row * c_row
        min_col = b_col * c_col
        raise ValueError(f'The input image is too small given the values of pixels_per_cell and cells_per_block. It should have at least: {min_row} rows and {min_col} cols.')
    normalized_blocks = np.zeros((n_blocks_row, n_blocks_col, b_row, b_col, orientations), dtype=float_dtype)
    for r in range(n_blocks_row):
        for c in range(n_blocks_col):
            block = orientation_histogram[r:r + b_row, c:c + b_col, :]
            normalized_blocks[r, c, :] = _hog_normalize_block(block, method=block_norm)
    '\n    The final step collects the HOG descriptors from all blocks of a dense\n    overlapping grid of blocks covering the detection window into a combined\n    feature vector for use in the window classifier.\n    '
    if feature_vector:
        normalized_blocks = normalized_blocks.ravel()
    if visualize:
        return (normalized_blocks, hog_image)
    else:
        return normalized_blocks