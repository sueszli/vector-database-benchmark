"""
The :mod:`sklearn.utils.sparsefuncs` module includes a collection of utilities to
work with sparse matrices and arrays.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from ..utils.fixes import _sparse_min_max, _sparse_nan_min_max
from ..utils.validation import _check_sample_weight
from .sparsefuncs_fast import csc_mean_variance_axis0 as _csc_mean_var_axis0
from .sparsefuncs_fast import csr_mean_variance_axis0 as _csr_mean_var_axis0
from .sparsefuncs_fast import incr_mean_variance_axis0 as _incr_mean_var_axis0

def _raise_typeerror(X):
    if False:
        print('Hello World!')
    'Raises a TypeError if X is not a CSR or CSC matrix'
    input_type = X.format if sp.issparse(X) else type(X)
    err = 'Expected a CSR or CSC sparse matrix, got %s.' % input_type
    raise TypeError(err)

def _raise_error_wrong_axis(axis):
    if False:
        return 10
    if axis not in (0, 1):
        raise ValueError('Unknown axis value: %d. Use 0 for rows, or 1 for columns' % axis)

def inplace_csr_column_scale(X, scale):
    if False:
        i = 10
        return i + 15
    'Inplace column scaling of a CSR matrix.\n\n    Scale each feature of the data matrix by multiplying with specific scale\n    provided by the caller assuming a (n_samples, n_features) shape.\n\n    Parameters\n    ----------\n    X : sparse matrix of shape (n_samples, n_features)\n        Matrix to normalize using the variance of the features.\n        It should be of CSR format.\n\n    scale : ndarray of shape (n_features,), dtype={np.float32, np.float64}\n        Array of precomputed feature-wise values to use for scaling.\n    '
    assert scale.shape[0] == X.shape[1]
    X.data *= scale.take(X.indices, mode='clip')

def inplace_csr_row_scale(X, scale):
    if False:
        while True:
            i = 10
    'Inplace row scaling of a CSR matrix.\n\n    Scale each sample of the data matrix by multiplying with specific scale\n    provided by the caller assuming a (n_samples, n_features) shape.\n\n    Parameters\n    ----------\n    X : sparse matrix of shape (n_samples, n_features)\n        Matrix to be scaled. It should be of CSR format.\n\n    scale : ndarray of float of shape (n_samples,)\n        Array of precomputed sample-wise values to use for scaling.\n    '
    assert scale.shape[0] == X.shape[0]
    X.data *= np.repeat(scale, np.diff(X.indptr))

def mean_variance_axis(X, axis, weights=None, return_sum_weights=False):
    if False:
        print('Hello World!')
    'Compute mean and variance along an axis on a CSR or CSC matrix.\n\n    Parameters\n    ----------\n    X : sparse matrix of shape (n_samples, n_features)\n        Input data. It can be of CSR or CSC format.\n\n    axis : {0, 1}\n        Axis along which the axis should be computed.\n\n    weights : ndarray of shape (n_samples,) or (n_features,), default=None\n        If axis is set to 0 shape is (n_samples,) or\n        if axis is set to 1 shape is (n_features,).\n        If it is set to None, then samples are equally weighted.\n\n        .. versionadded:: 0.24\n\n    return_sum_weights : bool, default=False\n        If True, returns the sum of weights seen for each feature\n        if `axis=0` or each sample if `axis=1`.\n\n        .. versionadded:: 0.24\n\n    Returns\n    -------\n\n    means : ndarray of shape (n_features,), dtype=floating\n        Feature-wise means.\n\n    variances : ndarray of shape (n_features,), dtype=floating\n        Feature-wise variances.\n\n    sum_weights : ndarray of shape (n_features,), dtype=floating\n        Returned if `return_sum_weights` is `True`.\n    '
    _raise_error_wrong_axis(axis)
    if sp.issparse(X) and X.format == 'csr':
        if axis == 0:
            return _csr_mean_var_axis0(X, weights=weights, return_sum_weights=return_sum_weights)
        else:
            return _csc_mean_var_axis0(X.T, weights=weights, return_sum_weights=return_sum_weights)
    elif sp.issparse(X) and X.format == 'csc':
        if axis == 0:
            return _csc_mean_var_axis0(X, weights=weights, return_sum_weights=return_sum_weights)
        else:
            return _csr_mean_var_axis0(X.T, weights=weights, return_sum_weights=return_sum_weights)
    else:
        _raise_typeerror(X)

def incr_mean_variance_axis(X, *, axis, last_mean, last_var, last_n, weights=None):
    if False:
        while True:
            i = 10
    'Compute incremental mean and variance along an axis on a CSR or CSC matrix.\n\n    last_mean, last_var are the statistics computed at the last step by this\n    function. Both must be initialized to 0-arrays of the proper size, i.e.\n    the number of features in X. last_n is the number of samples encountered\n    until now.\n\n    Parameters\n    ----------\n    X : CSR or CSC sparse matrix of shape (n_samples, n_features)\n        Input data.\n\n    axis : {0, 1}\n        Axis along which the axis should be computed.\n\n    last_mean : ndarray of shape (n_features,) or (n_samples,), dtype=floating\n        Array of means to update with the new data X.\n        Should be of shape (n_features,) if axis=0 or (n_samples,) if axis=1.\n\n    last_var : ndarray of shape (n_features,) or (n_samples,), dtype=floating\n        Array of variances to update with the new data X.\n        Should be of shape (n_features,) if axis=0 or (n_samples,) if axis=1.\n\n    last_n : float or ndarray of shape (n_features,) or (n_samples,),             dtype=floating\n        Sum of the weights seen so far, excluding the current weights\n        If not float, it should be of shape (n_features,) if\n        axis=0 or (n_samples,) if axis=1. If float it corresponds to\n        having same weights for all samples (or features).\n\n    weights : ndarray of shape (n_samples,) or (n_features,), default=None\n        If axis is set to 0 shape is (n_samples,) or\n        if axis is set to 1 shape is (n_features,).\n        If it is set to None, then samples are equally weighted.\n\n        .. versionadded:: 0.24\n\n    Returns\n    -------\n    means : ndarray of shape (n_features,) or (n_samples,), dtype=floating\n        Updated feature-wise means if axis = 0 or\n        sample-wise means if axis = 1.\n\n    variances : ndarray of shape (n_features,) or (n_samples,), dtype=floating\n        Updated feature-wise variances if axis = 0 or\n        sample-wise variances if axis = 1.\n\n    n : ndarray of shape (n_features,) or (n_samples,), dtype=integral\n        Updated number of seen samples per feature if axis=0\n        or number of seen features per sample if axis=1.\n\n        If weights is not None, n is a sum of the weights of the seen\n        samples or features instead of the actual number of seen\n        samples or features.\n\n    Notes\n    -----\n    NaNs are ignored in the algorithm.\n    '
    _raise_error_wrong_axis(axis)
    if not (sp.issparse(X) and X.format in ('csc', 'csr')):
        _raise_typeerror(X)
    if np.size(last_n) == 1:
        last_n = np.full(last_mean.shape, last_n, dtype=last_mean.dtype)
    if not np.size(last_mean) == np.size(last_var) == np.size(last_n):
        raise ValueError('last_mean, last_var, last_n do not have the same shapes.')
    if axis == 1:
        if np.size(last_mean) != X.shape[0]:
            raise ValueError(f'If axis=1, then last_mean, last_n, last_var should be of size n_samples {X.shape[0]} (Got {np.size(last_mean)}).')
    elif np.size(last_mean) != X.shape[1]:
        raise ValueError(f'If axis=0, then last_mean, last_n, last_var should be of size n_features {X.shape[1]} (Got {np.size(last_mean)}).')
    X = X.T if axis == 1 else X
    if weights is not None:
        weights = _check_sample_weight(weights, X, dtype=X.dtype)
    return _incr_mean_var_axis0(X, last_mean=last_mean, last_var=last_var, last_n=last_n, weights=weights)

def inplace_column_scale(X, scale):
    if False:
        print('Hello World!')
    'Inplace column scaling of a CSC/CSR matrix.\n\n    Scale each feature of the data matrix by multiplying with specific scale\n    provided by the caller assuming a (n_samples, n_features) shape.\n\n    Parameters\n    ----------\n    X : sparse matrix of shape (n_samples, n_features)\n        Matrix to normalize using the variance of the features. It should be\n        of CSC or CSR format.\n\n    scale : ndarray of shape (n_features,), dtype={np.float32, np.float64}\n        Array of precomputed feature-wise values to use for scaling.\n    '
    if sp.issparse(X) and X.format == 'csc':
        inplace_csr_row_scale(X.T, scale)
    elif sp.issparse(X) and X.format == 'csr':
        inplace_csr_column_scale(X, scale)
    else:
        _raise_typeerror(X)

def inplace_row_scale(X, scale):
    if False:
        print('Hello World!')
    'Inplace row scaling of a CSR or CSC matrix.\n\n    Scale each row of the data matrix by multiplying with specific scale\n    provided by the caller assuming a (n_samples, n_features) shape.\n\n    Parameters\n    ----------\n    X : sparse matrix of shape (n_samples, n_features)\n        Matrix to be scaled. It should be of CSR or CSC format.\n\n    scale : ndarray of shape (n_features,), dtype={np.float32, np.float64}\n        Array of precomputed sample-wise values to use for scaling.\n    '
    if sp.issparse(X) and X.format == 'csc':
        inplace_csr_column_scale(X.T, scale)
    elif sp.issparse(X) and X.format == 'csr':
        inplace_csr_row_scale(X, scale)
    else:
        _raise_typeerror(X)

def inplace_swap_row_csc(X, m, n):
    if False:
        for i in range(10):
            print('nop')
    'Swap two rows of a CSC matrix in-place.\n\n    Parameters\n    ----------\n    X : sparse matrix of shape (n_samples, n_features)\n        Matrix whose two rows are to be swapped. It should be of\n        CSC format.\n\n    m : int\n        Index of the row of X to be swapped.\n\n    n : int\n        Index of the row of X to be swapped.\n    '
    for t in [m, n]:
        if isinstance(t, np.ndarray):
            raise TypeError('m and n should be valid integers')
    if m < 0:
        m += X.shape[0]
    if n < 0:
        n += X.shape[0]
    m_mask = X.indices == m
    X.indices[X.indices == n] = m
    X.indices[m_mask] = n

def inplace_swap_row_csr(X, m, n):
    if False:
        i = 10
        return i + 15
    'Swap two rows of a CSR matrix in-place.\n\n    Parameters\n    ----------\n    X : sparse matrix of shape (n_samples, n_features)\n        Matrix whose two rows are to be swapped. It should be of\n        CSR format.\n\n    m : int\n        Index of the row of X to be swapped.\n\n    n : int\n        Index of the row of X to be swapped.\n    '
    for t in [m, n]:
        if isinstance(t, np.ndarray):
            raise TypeError('m and n should be valid integers')
    if m < 0:
        m += X.shape[0]
    if n < 0:
        n += X.shape[0]
    if m > n:
        (m, n) = (n, m)
    indptr = X.indptr
    m_start = indptr[m]
    m_stop = indptr[m + 1]
    n_start = indptr[n]
    n_stop = indptr[n + 1]
    nz_m = m_stop - m_start
    nz_n = n_stop - n_start
    if nz_m != nz_n:
        X.indptr[m + 2:n] += nz_n - nz_m
        X.indptr[m + 1] = m_start + nz_n
        X.indptr[n] = n_stop - nz_m
    X.indices = np.concatenate([X.indices[:m_start], X.indices[n_start:n_stop], X.indices[m_stop:n_start], X.indices[m_start:m_stop], X.indices[n_stop:]])
    X.data = np.concatenate([X.data[:m_start], X.data[n_start:n_stop], X.data[m_stop:n_start], X.data[m_start:m_stop], X.data[n_stop:]])

def inplace_swap_row(X, m, n):
    if False:
        print('Hello World!')
    '\n    Swap two rows of a CSC/CSR matrix in-place.\n\n    Parameters\n    ----------\n    X : sparse matrix of shape (n_samples, n_features)\n        Matrix whose two rows are to be swapped. It should be of CSR or\n        CSC format.\n\n    m : int\n        Index of the row of X to be swapped.\n\n    n : int\n        Index of the row of X to be swapped.\n    '
    if sp.issparse(X) and X.format == 'csc':
        inplace_swap_row_csc(X, m, n)
    elif sp.issparse(X) and X.format == 'csr':
        inplace_swap_row_csr(X, m, n)
    else:
        _raise_typeerror(X)

def inplace_swap_column(X, m, n):
    if False:
        while True:
            i = 10
    '\n    Swap two columns of a CSC/CSR matrix in-place.\n\n    Parameters\n    ----------\n    X : sparse matrix of shape (n_samples, n_features)\n        Matrix whose two columns are to be swapped. It should be of\n        CSR or CSC format.\n\n    m : int\n        Index of the column of X to be swapped.\n\n    n : int\n        Index of the column of X to be swapped.\n    '
    if m < 0:
        m += X.shape[1]
    if n < 0:
        n += X.shape[1]
    if sp.issparse(X) and X.format == 'csc':
        inplace_swap_row_csr(X, m, n)
    elif sp.issparse(X) and X.format == 'csr':
        inplace_swap_row_csc(X, m, n)
    else:
        _raise_typeerror(X)

def min_max_axis(X, axis, ignore_nan=False):
    if False:
        for i in range(10):
            print('nop')
    'Compute minimum and maximum along an axis on a CSR or CSC matrix.\n\n     Optionally ignore NaN values.\n\n    Parameters\n    ----------\n    X : sparse matrix of shape (n_samples, n_features)\n        Input data. It should be of CSR or CSC format.\n\n    axis : {0, 1}\n        Axis along which the axis should be computed.\n\n    ignore_nan : bool, default=False\n        Ignore or passing through NaN values.\n\n        .. versionadded:: 0.20\n\n    Returns\n    -------\n\n    mins : ndarray of shape (n_features,), dtype={np.float32, np.float64}\n        Feature-wise minima.\n\n    maxs : ndarray of shape (n_features,), dtype={np.float32, np.float64}\n        Feature-wise maxima.\n    '
    if sp.issparse(X) and X.format in ('csr', 'csc'):
        if ignore_nan:
            return _sparse_nan_min_max(X, axis=axis)
        else:
            return _sparse_min_max(X, axis=axis)
    else:
        _raise_typeerror(X)

def count_nonzero(X, axis=None, sample_weight=None):
    if False:
        i = 10
        return i + 15
    'A variant of X.getnnz() with extension to weighting on axis 0.\n\n    Useful in efficiently calculating multilabel metrics.\n\n    Parameters\n    ----------\n    X : sparse matrix of shape (n_samples, n_labels)\n        Input data. It should be of CSR format.\n\n    axis : {0, 1}, default=None\n        The axis on which the data is aggregated.\n\n    sample_weight : array-like of shape (n_samples,), default=None\n        Weight for each row of X.\n\n    Returns\n    -------\n    nnz : int, float, ndarray of shape (n_samples,) or ndarray of shape (n_features,)\n        Number of non-zero values in the array along a given axis. Otherwise,\n        the total number of non-zero values in the array is returned.\n    '
    if axis == -1:
        axis = 1
    elif axis == -2:
        axis = 0
    elif X.format != 'csr':
        raise TypeError('Expected CSR sparse format, got {0}'.format(X.format))
    if axis is None:
        if sample_weight is None:
            return X.nnz
        else:
            return np.dot(np.diff(X.indptr), sample_weight)
    elif axis == 1:
        out = np.diff(X.indptr)
        if sample_weight is None:
            return out.astype('intp')
        return out * sample_weight
    elif axis == 0:
        if sample_weight is None:
            return np.bincount(X.indices, minlength=X.shape[1])
        else:
            weights = np.repeat(sample_weight, np.diff(X.indptr))
            return np.bincount(X.indices, minlength=X.shape[1], weights=weights)
    else:
        raise ValueError('Unsupported axis: {0}'.format(axis))

def _get_median(data, n_zeros):
    if False:
        print('Hello World!')
    'Compute the median of data with n_zeros additional zeros.\n\n    This function is used to support sparse matrices; it modifies data\n    in-place.\n    '
    n_elems = len(data) + n_zeros
    if not n_elems:
        return np.nan
    n_negative = np.count_nonzero(data < 0)
    (middle, is_odd) = divmod(n_elems, 2)
    data.sort()
    if is_odd:
        return _get_elem_at_rank(middle, data, n_negative, n_zeros)
    return (_get_elem_at_rank(middle - 1, data, n_negative, n_zeros) + _get_elem_at_rank(middle, data, n_negative, n_zeros)) / 2.0

def _get_elem_at_rank(rank, data, n_negative, n_zeros):
    if False:
        i = 10
        return i + 15
    'Find the value in data augmented with n_zeros for the given rank'
    if rank < n_negative:
        return data[rank]
    if rank - n_negative < n_zeros:
        return 0
    return data[rank - n_zeros]

def csc_median_axis_0(X):
    if False:
        for i in range(10):
            print('nop')
    'Find the median across axis 0 of a CSC matrix.\n\n    It is equivalent to doing np.median(X, axis=0).\n\n    Parameters\n    ----------\n    X : sparse matrix of shape (n_samples, n_features)\n        Input data. It should be of CSC format.\n\n    Returns\n    -------\n    median : ndarray of shape (n_features,)\n        Median.\n    '
    if not (sp.issparse(X) and X.format == 'csc'):
        raise TypeError('Expected matrix of CSC format, got %s' % X.format)
    indptr = X.indptr
    (n_samples, n_features) = X.shape
    median = np.zeros(n_features)
    for (f_ind, (start, end)) in enumerate(zip(indptr[:-1], indptr[1:])):
        data = np.copy(X.data[start:end])
        nz = n_samples - data.size
        median[f_ind] = _get_median(data, nz)
    return median

def _implicit_column_offset(X, offset):
    if False:
        return 10
    'Create an implicitly offset linear operator.\n\n    This is used by PCA on sparse data to avoid densifying the whole data\n    matrix.\n\n    Params\n    ------\n        X : sparse matrix of shape (n_samples, n_features)\n        offset : ndarray of shape (n_features,)\n\n    Returns\n    -------\n    centered : LinearOperator\n    '
    offset = offset[None, :]
    XT = X.T
    return LinearOperator(matvec=lambda x: X @ x - offset @ x, matmat=lambda x: X @ x - offset @ x, rmatvec=lambda x: XT @ x - offset * x.sum(), rmatmat=lambda x: XT @ x - offset.T @ x.sum(axis=0)[None, :], dtype=X.dtype, shape=X.shape)