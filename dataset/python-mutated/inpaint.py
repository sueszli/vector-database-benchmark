import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import scipy.ndimage as ndi
from scipy.ndimage import laplace
import skimage
from .._shared import utils
from ..measure import label
from ._inpaint import _build_matrix_inner

def _get_neighborhood(nd_idx, radius, nd_shape):
    if False:
        print('Hello World!')
    bounds_lo = np.maximum(nd_idx - radius, 0)
    bounds_hi = np.minimum(nd_idx + radius + 1, nd_shape)
    return (bounds_lo, bounds_hi)

def _get_neigh_coef(shape, center, dtype=float):
    if False:
        return 10
    neigh_coef = np.zeros(shape, dtype=dtype)
    neigh_coef[center] = 1
    neigh_coef = laplace(laplace(neigh_coef))
    coef_idx = np.where(neigh_coef)
    coef_vals = neigh_coef[coef_idx]
    coef_idx = np.stack(coef_idx, axis=0)
    return (neigh_coef, coef_idx, coef_vals)

def _inpaint_biharmonic_single_region(image, mask, out, neigh_coef_full, coef_vals, raveled_offsets):
    if False:
        return 10
    'Solve a (sparse) linear system corresponding to biharmonic inpainting.\n\n    This function creates a linear system of the form:\n\n    ``A @ u = b``\n\n    where ``A`` is a sparse matrix, ``b`` is a vector enforcing smoothness and\n    boundary constraints and ``u`` is the vector of inpainted values to be\n    (uniquely) determined by solving the linear system.\n\n    ``A`` is a sparse matrix of shape (n_mask, n_mask) where ``n_mask``\n    corresponds to the number of non-zero values in ``mask`` (i.e. the number\n    of pixels to be inpainted). Each row in A will have a number of non-zero\n    values equal to the number of non-zero values in the biharmonic kernel,\n    ``neigh_coef_full``. In practice, biharmonic kernels with reduced extent\n    are used at the image borders. This matrix, ``A`` is the same for all\n    image channels (since the same inpainting mask is currently used for all\n    channels).\n\n    ``u`` is a dense matrix of shape ``(n_mask, n_channels)`` and represents\n    the vector of unknown values for each channel.\n\n    ``b`` is a dense matrix of shape ``(n_mask, n_channels)`` and represents\n    the desired output of convolving the solution with the biharmonic kernel.\n    At mask locations where there is no overlap with known values, ``b`` will\n    have a value of 0. This enforces the biharmonic smoothness constraint in\n    the interior of inpainting regions. For regions near the boundary that\n    overlap with known values, the entries in ``b`` enforce boundary conditions\n    designed to avoid discontinuity with the known values.\n    '
    n_channels = out.shape[-1]
    radius = neigh_coef_full.shape[0] // 2
    edge_mask = np.ones(mask.shape, dtype=bool)
    edge_mask[(slice(radius, -radius),) * mask.ndim] = 0
    boundary_mask = edge_mask * mask
    center_mask = ~edge_mask * mask
    boundary_pts = np.where(boundary_mask)
    boundary_i = np.flatnonzero(boundary_mask)
    center_i = np.flatnonzero(center_mask)
    mask_i = np.concatenate((boundary_i, center_i))
    center_pts = np.where(center_mask)
    mask_pts = tuple([np.concatenate((b, c)) for (b, c) in zip(boundary_pts, center_pts)])
    structure = neigh_coef_full != 0
    tmp = ndi.convolve(mask, structure, output=np.uint8, mode='constant')
    nnz_matrix = tmp[mask].sum()
    n_mask = np.count_nonzero(mask)
    n_struct = np.count_nonzero(structure)
    nnz_rhs_vector_max = n_mask - np.count_nonzero(tmp == n_struct)
    row_idx_known = np.empty(nnz_rhs_vector_max, dtype=np.intp)
    data_known = np.zeros((nnz_rhs_vector_max, n_channels), dtype=out.dtype)
    row_idx_unknown = np.empty(nnz_matrix, dtype=np.intp)
    col_idx_unknown = np.empty(nnz_matrix, dtype=np.intp)
    data_unknown = np.empty(nnz_matrix, dtype=out.dtype)
    coef_cache = {}
    mask_flat = mask.reshape(-1)
    out_flat = np.ascontiguousarray(out.reshape((-1, n_channels)))
    idx_known = 0
    idx_unknown = 0
    mask_pt_n = -1
    boundary_pts = np.stack(boundary_pts, axis=1)
    for (mask_pt_n, nd_idx) in enumerate(boundary_pts):
        (b_lo, b_hi) = _get_neighborhood(nd_idx, radius, mask.shape)
        coef_shape = tuple(b_hi - b_lo)
        coef_center = tuple(nd_idx - b_lo)
        (coef_idx, coefs) = coef_cache.get((coef_shape, coef_center), (None, None))
        if coef_idx is None:
            (_, coef_idx, coefs) = _get_neigh_coef(coef_shape, coef_center, dtype=out.dtype)
            coef_cache[coef_shape, coef_center] = (coef_idx, coefs)
        coef_idx = coef_idx + b_lo[:, np.newaxis]
        index1d = np.ravel_multi_index(coef_idx, mask.shape)
        nvals = 0
        for (coef, i) in zip(coefs, index1d):
            if mask_flat[i]:
                row_idx_unknown[idx_unknown] = mask_pt_n
                col_idx_unknown[idx_unknown] = i
                data_unknown[idx_unknown] = coef
                idx_unknown += 1
            else:
                data_known[idx_known, :] -= coef * out_flat[i, :]
                nvals += 1
        if nvals:
            row_idx_known[idx_known] = mask_pt_n
            idx_known += 1
    row_start = mask_pt_n + 1
    known_start_idx = idx_known
    unknown_start_idx = idx_unknown
    nnz_rhs = _build_matrix_inner(row_start, known_start_idx, unknown_start_idx, center_i, raveled_offsets, coef_vals, mask_flat, out_flat, row_idx_known, data_known, row_idx_unknown, col_idx_unknown, data_unknown)
    row_idx_known = row_idx_known[:nnz_rhs]
    data_known = data_known[:nnz_rhs, :]
    sp_shape = (n_mask, out.size)
    matrix_unknown = sparse.coo_matrix((data_unknown, (row_idx_unknown, col_idx_unknown)), shape=sp_shape).tocsr()
    matrix_unknown = matrix_unknown[:, mask_i]
    rhs = np.zeros((n_mask, n_channels), dtype=out.dtype)
    rhs[row_idx_known, :] = data_known
    result = spsolve(matrix_unknown, rhs, use_umfpack=False, permc_spec='MMD_ATA')
    if result.ndim == 1:
        result = result[:, np.newaxis]
    out[mask_pts] = result
    return out

@utils.channel_as_last_axis()
def inpaint_biharmonic(image, mask, *, split_into_regions=False, channel_axis=None):
    if False:
        i = 10
        return i + 15
    'Inpaint masked points in image with biharmonic equations.\n\n    Parameters\n    ----------\n    image : (M[, N[, ..., P]][, C]) ndarray\n        Input image.\n    mask : (M[, N[, ..., P]]) ndarray\n        Array of pixels to be inpainted. Have to be the same shape as one\n        of the \'image\' channels. Unknown pixels have to be represented with 1,\n        known pixels - with 0.\n    split_into_regions : boolean, optional\n        If True, inpainting is performed on a region-by-region basis. This is\n        likely to be slower, but will have reduced memory requirements.\n    channel_axis : int or None, optional\n        If None, the image is assumed to be a grayscale (single channel) image.\n        Otherwise, this parameter indicates which axis of the array corresponds\n        to channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    out : (M[, N[, ..., P]][, C]) ndarray\n        Input image with masked pixels inpainted.\n\n    References\n    ----------\n    .. [1]  S.B.Damelin and N.S.Hoang. "On Surface Completion and Image\n            Inpainting by Biharmonic Functions: Numerical Aspects",\n            International Journal of Mathematics and Mathematical Sciences,\n            Vol. 2018, Article ID 3950312\n            :DOI:`10.1155/2018/3950312`\n    .. [2]  C. K. Chui and H. N. Mhaskar, MRA Contextual-Recovery Extension of\n            Smooth Functions on Manifolds, Appl. and Comp. Harmonic Anal.,\n            28 (2010), 104-113,\n            :DOI:`10.1016/j.acha.2009.04.004`\n\n    Examples\n    --------\n    >>> img = np.tile(np.square(np.linspace(0, 1, 5)), (5, 1))\n    >>> mask = np.zeros_like(img)\n    >>> mask[2, 2:] = 1\n    >>> mask[1, 3:] = 1\n    >>> mask[0, 4:] = 1\n    >>> out = inpaint_biharmonic(img, mask)\n    '
    if image.ndim < 1:
        raise ValueError('Input array has to be at least 1D')
    multichannel = channel_axis is not None
    img_baseshape = image.shape[:-1] if multichannel else image.shape
    if img_baseshape != mask.shape:
        raise ValueError('Input arrays have to be the same shape')
    if np.ma.isMaskedArray(image):
        raise TypeError('Masked arrays are not supported')
    image = skimage.img_as_float(image)
    float_dtype = utils._supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    mask = mask.astype(bool, copy=False)
    if not multichannel:
        image = image[..., np.newaxis]
    out = np.copy(image, order='C')
    radius = 2
    coef_shape = (2 * radius + 1,) * mask.ndim
    coef_center = (radius,) * mask.ndim
    (neigh_coef_full, coef_idx, coef_vals) = _get_neigh_coef(coef_shape, coef_center, dtype=out.dtype)
    channel_stride_bytes = out.strides[-2]
    offsets = coef_idx - radius
    known_points = image[~mask]
    limits = (known_points.min(axis=0), known_points.max(axis=0))
    if split_into_regions:
        kernel = ndi.generate_binary_structure(mask.ndim, 1)
        mask_dilated = ndi.binary_dilation(mask, structure=kernel)
        mask_labeled = label(mask_dilated)
        mask_labeled *= mask
        bbox_slices = ndi.find_objects(mask_labeled)
        for (idx_region, bb_slice) in enumerate(bbox_slices, 1):
            roi_sl = tuple((slice(max(sl.start - radius, 0), min(sl.stop + radius, size)) for (sl, size) in zip(bb_slice, mask_labeled.shape)))
            mask_region = mask_labeled[roi_sl] == idx_region
            roi_sl += (slice(None),)
            otmp = out[roi_sl].copy()
            ostrides = np.array([s // channel_stride_bytes for s in otmp[..., 0].strides])
            raveled_offsets = np.sum(offsets * ostrides[..., np.newaxis], axis=0)
            _inpaint_biharmonic_single_region(image[roi_sl], mask_region, otmp, neigh_coef_full, coef_vals, raveled_offsets)
            out[roi_sl] = otmp
    else:
        ostrides = np.array([s // channel_stride_bytes for s in out[..., 0].strides])
        raveled_offsets = np.sum(offsets * ostrides[..., np.newaxis], axis=0)
        _inpaint_biharmonic_single_region(image, mask, out, neigh_coef_full, coef_vals, raveled_offsets)
    np.clip(out, a_min=limits[0], a_max=limits[1], out=out)
    if not multichannel:
        out = out[..., 0]
    return out