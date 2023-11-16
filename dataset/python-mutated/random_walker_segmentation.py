"""
Random walker segmentation algorithm

from *Random walks for image segmentation*, Leo Grady, IEEE Trans
Pattern Anal Mach Intell. 2006 Nov;28(11):1768-83.

Installing pyamg and using the 'cg_mg' mode of random_walker improves
significantly the performance.
"""
import numpy as np
from scipy import sparse, ndimage as ndi
from .._shared import utils
from .._shared.utils import warn
try:
    from scipy.sparse.linalg.dsolve.linsolve import umfpack
    old_del = umfpack.UmfpackContext.__del__

    def new_del(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            old_del(self)
        except AttributeError:
            pass
    umfpack.UmfpackContext.__del__ = new_del
    UmfpackContext = umfpack.UmfpackContext()
except ImportError:
    UmfpackContext = None
try:
    from pyamg import ruge_stuben_solver
    amg_loaded = True
except ImportError:
    amg_loaded = False
from ..util import img_as_float
from scipy.sparse.linalg import cg, spsolve

def _make_graph_edges_3d(n_x, n_y, n_z):
    if False:
        return 10
    'Returns a list of edges for a 3D image.\n\n    Parameters\n    ----------\n    n_x : integer\n        The size of the grid in the x direction.\n    n_y : integer\n        The size of the grid in the y direction\n    n_z : integer\n        The size of the grid in the z direction\n\n    Returns\n    -------\n    edges : (2, N) ndarray\n        with the total number of edges::\n\n            N = n_x * n_y * (nz - 1) +\n                n_x * (n_y - 1) * nz +\n                (n_x - 1) * n_y * nz\n\n        Graph edges with each column describing a node-id pair.\n    '
    vertices = np.arange(n_x * n_y * n_z).reshape((n_x, n_y, n_z))
    edges_deep = np.vstack((vertices[..., :-1].ravel(), vertices[..., 1:].ravel()))
    edges_right = np.vstack((vertices[:, :-1].ravel(), vertices[:, 1:].ravel()))
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))
    edges = np.hstack((edges_deep, edges_right, edges_down))
    return edges

def _compute_weights_3d(data, spacing, beta, eps, multichannel):
    if False:
        print('Hello World!')
    gradients = np.concatenate([np.diff(data[..., 0], axis=ax).ravel() / spacing[ax] for ax in [2, 1, 0] if data.shape[ax] > 1], axis=0) ** 2
    for channel in range(1, data.shape[-1]):
        gradients += np.concatenate([np.diff(data[..., channel], axis=ax).ravel() / spacing[ax] for ax in [2, 1, 0] if data.shape[ax] > 1], axis=0) ** 2
    scale_factor = -beta / (10 * data.std())
    if multichannel:
        scale_factor /= np.sqrt(data.shape[-1])
    weights = np.exp(scale_factor * gradients)
    weights += eps
    return -weights

def _build_laplacian(data, spacing, mask, beta, multichannel):
    if False:
        return 10
    (l_x, l_y, l_z) = data.shape[:3]
    edges = _make_graph_edges_3d(l_x, l_y, l_z)
    weights = _compute_weights_3d(data, spacing, beta=beta, eps=1e-10, multichannel=multichannel)
    if mask is not None:
        mask0 = np.hstack([mask[..., :-1].ravel(), mask[:, :-1].ravel(), mask[:-1].ravel()])
        mask1 = np.hstack([mask[..., 1:].ravel(), mask[:, 1:].ravel(), mask[1:].ravel()])
        ind_mask = np.logical_and(mask0, mask1)
        (edges, weights) = (edges[:, ind_mask], weights[ind_mask])
        (_, inv_idx) = np.unique(edges, return_inverse=True)
        edges = inv_idx.reshape(edges.shape)
    pixel_nb = l_x * l_y * l_z
    i_indices = edges.ravel()
    j_indices = edges[::-1].ravel()
    data = np.hstack((weights, weights))
    lap = sparse.coo_matrix((data, (i_indices, j_indices)), shape=(pixel_nb, pixel_nb))
    lap.setdiag(-np.ravel(lap.sum(axis=0)))
    return lap.tocsr()

def _build_linear_system(data, spacing, labels, nlabels, mask, beta, multichannel):
    if False:
        i = 10
        return i + 15
    '\n    Build the matrix A and rhs B of the linear system to solve.\n    A and B are two block of the laplacian of the image graph.\n    '
    if mask is None:
        labels = labels.ravel()
    else:
        labels = labels[mask]
    indices = np.arange(labels.size)
    seeds_mask = labels > 0
    unlabeled_indices = indices[~seeds_mask]
    seeds_indices = indices[seeds_mask]
    lap_sparse = _build_laplacian(data, spacing, mask=mask, beta=beta, multichannel=multichannel)
    rows = lap_sparse[unlabeled_indices, :]
    lap_sparse = rows[:, unlabeled_indices]
    B = -rows[:, seeds_indices]
    seeds = labels[seeds_mask]
    seeds_mask = sparse.csc_matrix(np.hstack([np.atleast_2d(seeds == lab).T for lab in range(1, nlabels + 1)]))
    rhs = B.dot(seeds_mask)
    return (lap_sparse, rhs)

def _solve_linear_system(lap_sparse, B, tol, mode):
    if False:
        for i in range(10):
            print('nop')
    if mode is None:
        mode = 'cg_j'
    if mode == 'cg_mg' and (not amg_loaded):
        warn('"cg_mg" not available, it requires pyamg to be installed. The "cg_j" mode will be used instead.', stacklevel=2)
        mode = 'cg_j'
    if mode == 'bf':
        X = spsolve(lap_sparse, B.toarray()).T
    else:
        maxiter = None
        if mode == 'cg':
            if UmfpackContext is None:
                warn('"cg" mode may be slow because UMFPACK is not available. Consider building Scipy with UMFPACK or use a preconditioned version of CG ("cg_j" or "cg_mg" modes).', stacklevel=2)
            M = None
        elif mode == 'cg_j':
            M = sparse.diags(1.0 / lap_sparse.diagonal())
        else:
            lap_sparse = lap_sparse.tocsr()
            ml = ruge_stuben_solver(lap_sparse, coarse_solver='pinv')
            M = ml.aspreconditioner(cycle='V')
            maxiter = 30
        cg_out = [cg(lap_sparse, B[:, i].toarray(), tol=tol, atol=0, M=M, maxiter=maxiter) for i in range(B.shape[1])]
        if np.any([info > 0 for (_, info) in cg_out]):
            warn('Conjugate gradient convergence to tolerance not achieved. Consider decreasing beta to improve system conditionning.', stacklevel=2)
        X = np.asarray([x for (x, _) in cg_out])
    return X

def _preprocess(labels):
    if False:
        return 10
    (label_values, inv_idx) = np.unique(labels, return_inverse=True)
    if max(label_values) <= 0:
        raise ValueError('No seeds provided in label image: please ensure it contains at least one positive value')
    if not (label_values == 0).any():
        warn('Random walker only segments unlabeled areas, where labels == 0. No zero valued areas in labels were found. Returning provided labels.', stacklevel=2)
        return (labels, None, None, None, None)
    null_mask = labels == 0
    pos_mask = labels > 0
    mask = labels >= 0
    fill = ndi.binary_propagation(null_mask, mask=mask)
    isolated = np.logical_and(pos_mask, np.logical_not(fill))
    pos_mask[isolated] = False
    if label_values[0] < 0 or np.any(isolated):
        isolated = np.logical_and(np.logical_not(ndi.binary_propagation(pos_mask, mask=mask)), null_mask)
        labels[isolated] = -1
        if np.all(isolated[null_mask]):
            warn('All unlabeled pixels are isolated, they could not be determined by the random walker algorithm.', stacklevel=2)
            return (labels, None, None, None, None)
        mask[isolated] = False
        mask = np.atleast_3d(mask)
    else:
        mask = None
    zero_idx = np.searchsorted(label_values, 0)
    labels = np.atleast_3d(inv_idx.reshape(labels.shape) - zero_idx)
    nlabels = label_values[zero_idx + 1:].shape[0]
    inds_isolated_seeds = np.nonzero(isolated)
    isolated_values = labels[inds_isolated_seeds]
    return (labels, nlabels, mask, inds_isolated_seeds, isolated_values)

@utils.channel_as_last_axis(multichannel_output=False)
def random_walker(data, labels, beta=130, mode='cg_j', tol=0.001, copy=True, return_full_prob=False, spacing=None, *, prob_tol=0.001, channel_axis=None):
    if False:
        return 10
    'Random walker algorithm for segmentation from markers.\n\n    Random walker algorithm is implemented for gray-level or multichannel\n    images.\n\n    Parameters\n    ----------\n    data : (M, N[, P][, C]) ndarray\n        Image to be segmented in phases. Gray-level `data` can be two- or\n        three-dimensional; multichannel data can be three- or four-\n        dimensional with `channel_axis` specifying the dimension containing\n        channels. Data spacing is assumed isotropic unless the `spacing`\n        keyword argument is used.\n    labels : (M, N[, P]) array of ints\n        Array of seed markers labeled with different positive integers\n        for different phases. Zero-labeled pixels are unlabeled pixels.\n        Negative labels correspond to inactive pixels that are not taken\n        into account (they are removed from the graph). If labels are not\n        consecutive integers, the labels array will be transformed so that\n        labels are consecutive. In the multichannel case, `labels` should have\n        the same shape as a single channel of `data`, i.e. without the final\n        dimension denoting channels.\n    beta : float, optional\n        Penalization coefficient for the random walker motion\n        (the greater `beta`, the more difficult the diffusion).\n    mode : string, available options {\'cg\', \'cg_j\', \'cg_mg\', \'bf\'}\n        Mode for solving the linear system in the random walker algorithm.\n\n        - \'bf\' (brute force): an LU factorization of the Laplacian is\n          computed. This is fast for small images (<1024x1024), but very slow\n          and memory-intensive for large images (e.g., 3-D volumes).\n        - \'cg\' (conjugate gradient): the linear system is solved iteratively\n          using the Conjugate Gradient method from scipy.sparse.linalg. This is\n          less memory-consuming than the brute force method for large images,\n          but it is quite slow.\n        - \'cg_j\' (conjugate gradient with Jacobi preconditionner): the\n          Jacobi preconditionner is applied during the Conjugate\n          gradient method iterations. This may accelerate the\n          convergence of the \'cg\' method.\n        - \'cg_mg\' (conjugate gradient with multigrid preconditioner): a\n          preconditioner is computed using a multigrid solver, then the\n          solution is computed with the Conjugate Gradient method. This mode\n          requires that the pyamg module is installed.\n    tol : float, optional\n        Tolerance to achieve when solving the linear system using\n        the conjugate gradient based modes (\'cg\', \'cg_j\' and \'cg_mg\').\n    copy : bool, optional\n        If copy is False, the `labels` array will be overwritten with\n        the result of the segmentation. Use copy=False if you want to\n        save on memory.\n    return_full_prob : bool, optional\n        If True, the probability that a pixel belongs to each of the\n        labels will be returned, instead of only the most likely\n        label.\n    spacing : iterable of floats, optional\n        Spacing between voxels in each spatial dimension. If `None`, then\n        the spacing between pixels/voxels in each dimension is assumed 1.\n    prob_tol : float, optional\n        Tolerance on the resulting probability to be in the interval [0, 1].\n        If the tolerance is not satisfied, a warning is displayed.\n    channel_axis : int or None, optional\n        If None, the image is assumed to be a grayscale (single channel) image.\n        Otherwise, this parameter indicates which axis of the array corresponds\n        to channels.\n\n        .. versionadded:: 0.19\n           ``channel_axis`` was added in 0.19.\n\n    Returns\n    -------\n    output : ndarray\n        * If `return_full_prob` is False, array of ints of same shape\n          and data type as `labels`, in which each pixel has been\n          labeled according to the marker that reached the pixel first\n          by anisotropic diffusion.\n        * If `return_full_prob` is True, array of floats of shape\n          `(nlabels, labels.shape)`. `output[label_nb, i, j]` is the\n          probability that label `label_nb` reaches the pixel `(i, j)`\n          first.\n\n    See Also\n    --------\n    skimage.segmentation.watershed\n        A segmentation algorithm based on mathematical morphology\n        and "flooding" of regions from markers.\n\n    Notes\n    -----\n    Multichannel inputs are scaled with all channel data combined. Ensure all\n    channels are separately normalized prior to running this algorithm.\n\n    The `spacing` argument is specifically for anisotropic datasets, where\n    data points are spaced differently in one or more spatial dimensions.\n    Anisotropic data is commonly encountered in medical imaging.\n\n    The algorithm was first proposed in [1]_.\n\n    The algorithm solves the diffusion equation at infinite times for\n    sources placed on markers of each phase in turn. A pixel is labeled with\n    the phase that has the greatest probability to diffuse first to the pixel.\n\n    The diffusion equation is solved by minimizing x.T L x for each phase,\n    where L is the Laplacian of the weighted graph of the image, and x is\n    the probability that a marker of the given phase arrives first at a pixel\n    by diffusion (x=1 on markers of the phase, x=0 on the other markers, and\n    the other coefficients are looked for). Each pixel is attributed the label\n    for which it has a maximal value of x. The Laplacian L of the image\n    is defined as:\n\n       - L_ii = d_i, the number of neighbors of pixel i (the degree of i)\n       - L_ij = -w_ij if i and j are adjacent pixels\n\n    The weight w_ij is a decreasing function of the norm of the local gradient.\n    This ensures that diffusion is easier between pixels of similar values.\n\n    When the Laplacian is decomposed into blocks of marked and unmarked\n    pixels::\n\n        L = M B.T\n            B A\n\n    with first indices corresponding to marked pixels, and then to unmarked\n    pixels, minimizing x.T L x for one phase amount to solving::\n\n        A x = - B x_m\n\n    where x_m = 1 on markers of the given phase, and 0 on other markers.\n    This linear system is solved in the algorithm using a direct method for\n    small images, and an iterative method for larger images.\n\n    References\n    ----------\n    .. [1] Leo Grady, Random walks for image segmentation, IEEE Trans Pattern\n        Anal Mach Intell. 2006 Nov;28(11):1768-83.\n        :DOI:`10.1109/TPAMI.2006.233`.\n\n    Examples\n    --------\n    >>> rng = np.random.default_rng()\n    >>> a = np.zeros((10, 10)) + 0.2 * rng.random((10, 10))\n    >>> a[5:8, 5:8] += 1\n    >>> b = np.zeros_like(a, dtype=np.int32)\n    >>> b[3, 3] = 1  # Marker for first phase\n    >>> b[6, 6] = 2  # Marker for second phase\n    >>> random_walker(a, b)  # doctest: +SKIP\n    array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],\n           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],\n           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],\n           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],\n           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],\n           [1, 1, 1, 1, 1, 2, 2, 2, 1, 1],\n           [1, 1, 1, 1, 1, 2, 2, 2, 1, 1],\n           [1, 1, 1, 1, 1, 2, 2, 2, 1, 1],\n           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],\n           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=int32)\n\n    '
    if mode not in ('cg_mg', 'cg', 'bf', 'cg_j', None):
        raise ValueError(f"{mode} is not a valid mode. Valid modes are 'cg_mg', 'cg', 'cg_j', 'bf', and None")
    if spacing is None:
        spacing = np.ones(3)
    elif len(spacing) == labels.ndim:
        if len(spacing) == 2:
            spacing = np.r_[spacing, 1.0]
        spacing = np.asarray(spacing)
    else:
        raise ValueError('Input argument `spacing` incorrect, should be an iterable with one number per spatial dimension.')
    multichannel = channel_axis is not None
    if not multichannel:
        if data.ndim not in (2, 3):
            raise ValueError('For non-multichannel input, data must be of dimension 2 or 3.')
        if data.shape != labels.shape:
            raise ValueError('Incompatible data and labels shapes.')
        data = np.atleast_3d(img_as_float(data))[..., np.newaxis]
    else:
        if data.ndim not in (3, 4):
            raise ValueError('For multichannel input, data must have 3 or 4 dimensions.')
        if data.shape[:-1] != labels.shape:
            raise ValueError('Incompatible data and labels shapes.')
        data = img_as_float(data)
        if data.ndim == 3:
            data = data[:, :, np.newaxis, :]
    labels_shape = labels.shape
    labels_dtype = labels.dtype
    if copy:
        labels = np.copy(labels)
    (labels, nlabels, mask, inds_isolated_seeds, isolated_values) = _preprocess(labels)
    if isolated_values is None:
        if return_full_prob:
            return np.concatenate([np.atleast_3d(labels == lab) for lab in np.unique(labels) if lab > 0], axis=-1)
        return labels
    (lap_sparse, B) = _build_linear_system(data, spacing, labels, nlabels, mask, beta, multichannel)
    X = _solve_linear_system(lap_sparse, B, tol, mode)
    if X.min() < -prob_tol or X.max() > 1 + prob_tol:
        warn('The probability range is outside [0, 1] given the tolerance `prob_tol`. Consider decreasing `beta` and/or decreasing `tol`.')
    labels[inds_isolated_seeds] = isolated_values
    labels = labels.reshape(labels_shape)
    mask = labels == 0
    mask[inds_isolated_seeds] = False
    if return_full_prob:
        out = np.zeros((nlabels,) + labels_shape)
        for (lab, (label_prob, prob)) in enumerate(zip(out, X), start=1):
            label_prob[mask] = prob
            label_prob[labels == lab] = 1
    else:
        X = np.argmax(X, axis=0) + 1
        out = labels.astype(labels_dtype)
        out[mask] = X
    return out