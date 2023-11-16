import numpy as np
import scipy.sparse as sparse
from ._contingency_table import contingency_table
from .._shared.utils import check_shape_equality
__all__ = ['variation_of_information']

def variation_of_information(image0=None, image1=None, *, table=None, ignore_labels=()):
    if False:
        i = 10
        return i + 15
    'Return symmetric conditional entropies associated with the VI. [1]_\n\n    The variation of information is defined as VI(X,Y) = H(X|Y) + H(Y|X).\n    If X is the ground-truth segmentation, then H(X|Y) can be interpreted\n    as the amount of under-segmentation and H(Y|X) as the amount\n    of over-segmentation. In other words, a perfect over-segmentation\n    will have H(X|Y)=0 and a perfect under-segmentation will have H(Y|X)=0.\n\n    Parameters\n    ----------\n    image0, image1 : ndarray of int\n        Label images / segmentations, must have same shape.\n    table : scipy.sparse array in csr format, optional\n        A contingency table built with skimage.evaluate.contingency_table.\n        If None, it will be computed with skimage.evaluate.contingency_table.\n        If given, the entropies will be computed from this table and any images\n        will be ignored.\n    ignore_labels : sequence of int, optional\n        Labels to ignore. Any part of the true image labeled with any of these\n        values will not be counted in the score.\n\n    Returns\n    -------\n    vi : ndarray of float, shape (2,)\n        The conditional entropies of image1|image0 and image0|image1.\n\n    References\n    ----------\n    .. [1] Marina Meilă (2007), Comparing clusterings—an information based\n        distance, Journal of Multivariate Analysis, Volume 98, Issue 5,\n        Pages 873-895, ISSN 0047-259X, :DOI:`10.1016/j.jmva.2006.11.013`.\n    '
    (h0g1, h1g0) = _vi_tables(image0, image1, table=table, ignore_labels=ignore_labels)
    return np.array([h1g0.sum(), h0g1.sum()])

def _xlogx(x):
    if False:
        for i in range(10):
            print('nop')
    'Compute x * log_2(x).\n\n    We define 0 * log_2(0) = 0\n\n    Parameters\n    ----------\n    x : ndarray or scipy.sparse.csc_matrix or csr_matrix\n        The input array.\n\n    Returns\n    -------\n    y : same type as x\n        Result of x * log_2(x).\n    '
    y = x.copy()
    if isinstance(y, sparse.csc_matrix) or isinstance(y, sparse.csr_matrix):
        z = y.data
    else:
        z = np.asarray(y)
    nz = z.nonzero()
    z[nz] *= np.log2(z[nz])
    return y

def _vi_tables(im_true, im_test, table=None, ignore_labels=()):
    if False:
        print('Hello World!')
    'Compute probability tables used for calculating VI.\n\n    Parameters\n    ----------\n    im_true, im_test : ndarray of int\n        Input label images, any dimensionality.\n    table : csr matrix, optional\n        Pre-computed contingency table.\n    ignore_labels : sequence of int, optional\n        Labels to ignore when computing scores.\n\n    Returns\n    -------\n    hxgy, hygx : ndarray of float\n        Per-segment conditional entropies of ``im_true`` given ``im_test`` and\n        vice-versa.\n    '
    check_shape_equality(im_true, im_test)
    if table is None:
        pxy = contingency_table(im_true, im_test, ignore_labels=ignore_labels, normalize=True)
    else:
        pxy = table
    px = np.ravel(pxy.sum(axis=1))
    py = np.ravel(pxy.sum(axis=0))
    px_inv = sparse.diags(_invert_nonzero(px))
    py_inv = sparse.diags(_invert_nonzero(py))
    hygx = -px @ _xlogx(px_inv @ pxy).sum(axis=1)
    hxgy = -_xlogx(pxy @ py_inv).sum(axis=0) @ py
    return list(map(np.asarray, [hxgy, hygx]))

def _invert_nonzero(arr):
    if False:
        for i in range(10):
            print('nop')
    'Compute the inverse of the non-zero elements of arr, not changing 0.\n\n    Parameters\n    ----------\n    arr : ndarray\n\n    Returns\n    -------\n    arr_inv : ndarray\n        Array containing the inverse of the non-zero elements of arr, and\n        zero elsewhere.\n    '
    arr_inv = arr.copy()
    nz = np.nonzero(arr)
    arr_inv[nz] = 1 / arr[nz]
    return arr_inv