"""
This module provides functions to perform full Procrustes analysis.

This code was originally written by Justin Kucynski and ported over from
scikit-bio by Yoshiki Vazquez-Baeza.
"""
import numpy as np
from scipy.linalg import orthogonal_procrustes
__all__ = ['procrustes']

def procrustes(data1, data2):
    if False:
        print('Hello World!')
    'Procrustes analysis, a similarity test for two data sets.\n\n    Each input matrix is a set of points or vectors (the rows of the matrix).\n    The dimension of the space is the number of columns of each matrix. Given\n    two identically sized matrices, procrustes standardizes both such that:\n\n    - :math:`tr(AA^{T}) = 1`.\n\n    - Both sets of points are centered around the origin.\n\n    Procrustes ([1]_, [2]_) then applies the optimal transform to the second\n    matrix (including scaling/dilation, rotations, and reflections) to minimize\n    :math:`M^{2}=\\sum(data1-data2)^{2}`, or the sum of the squares of the\n    pointwise differences between the two input datasets.\n\n    This function was not designed to handle datasets with different numbers of\n    datapoints (rows).  If two data sets have different dimensionality\n    (different number of columns), simply add columns of zeros to the smaller\n    of the two.\n\n    Parameters\n    ----------\n    data1 : array_like\n        Matrix, n rows represent points in k (columns) space `data1` is the\n        reference data, after it is standardised, the data from `data2` will be\n        transformed to fit the pattern in `data1` (must have >1 unique points).\n    data2 : array_like\n        n rows of data in k space to be fit to `data1`.  Must be the  same\n        shape ``(numrows, numcols)`` as data1 (must have >1 unique points).\n\n    Returns\n    -------\n    mtx1 : array_like\n        A standardized version of `data1`.\n    mtx2 : array_like\n        The orientation of `data2` that best fits `data1`. Centered, but not\n        necessarily :math:`tr(AA^{T}) = 1`.\n    disparity : float\n        :math:`M^{2}` as defined above.\n\n    Raises\n    ------\n    ValueError\n        If the input arrays are not two-dimensional.\n        If the shape of the input arrays is different.\n        If the input arrays have zero columns or zero rows.\n\n    See Also\n    --------\n    scipy.linalg.orthogonal_procrustes\n    scipy.spatial.distance.directed_hausdorff : Another similarity test\n      for two data sets\n\n    Notes\n    -----\n    - The disparity should not depend on the order of the input matrices, but\n      the output matrices will, as only the first output matrix is guaranteed\n      to be scaled such that :math:`tr(AA^{T}) = 1`.\n\n    - Duplicate data points are generally ok, duplicating a data point will\n      increase its effect on the procrustes fit.\n\n    - The disparity scales as the number of points per input matrix.\n\n    References\n    ----------\n    .. [1] Krzanowski, W. J. (2000). "Principles of Multivariate analysis".\n    .. [2] Gower, J. C. (1975). "Generalized procrustes analysis".\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.spatial import procrustes\n\n    The matrix ``b`` is a rotated, shifted, scaled and mirrored version of\n    ``a`` here:\n\n    >>> a = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], \'d\')\n    >>> b = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], \'d\')\n    >>> mtx1, mtx2, disparity = procrustes(a, b)\n    >>> round(disparity)\n    0.0\n\n    '
    mtx1 = np.array(data1, dtype=np.float64, copy=True)
    mtx2 = np.array(data2, dtype=np.float64, copy=True)
    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError('Input matrices must be two-dimensional')
    if mtx1.shape != mtx2.shape:
        raise ValueError('Input matrices must be of same shape')
    if mtx1.size == 0:
        raise ValueError('Input matrices must be >0 rows and >0 cols')
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)
    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)
    if norm1 == 0 or norm2 == 0:
        raise ValueError('Input matrices must contain >1 unique points')
    mtx1 /= norm1
    mtx2 /= norm2
    (R, s) = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s
    disparity = np.sum(np.square(mtx1 - mtx2))
    return (mtx1, mtx2, disparity)