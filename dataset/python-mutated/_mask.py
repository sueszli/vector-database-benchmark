from contextlib import suppress
import numpy as np
from scipy import sparse as sp
from . import is_scalar_nan
from .fixes import _object_dtype_isnan

def _get_dense_mask(X, value_to_mask):
    if False:
        print('Hello World!')
    with suppress(ImportError, AttributeError):
        import pandas
        if value_to_mask is pandas.NA:
            return pandas.isna(X)
    if is_scalar_nan(value_to_mask):
        if X.dtype.kind == 'f':
            Xt = np.isnan(X)
        elif X.dtype.kind in ('i', 'u'):
            Xt = np.zeros(X.shape, dtype=bool)
        else:
            Xt = _object_dtype_isnan(X)
    else:
        Xt = X == value_to_mask
    return Xt

def _get_mask(X, value_to_mask):
    if False:
        return 10
    'Compute the boolean mask X == value_to_mask.\n\n    Parameters\n    ----------\n    X : {ndarray, sparse matrix} of shape (n_samples, n_features)\n        Input data, where ``n_samples`` is the number of samples and\n        ``n_features`` is the number of features.\n\n    value_to_mask : {int, float}\n        The value which is to be masked in X.\n\n    Returns\n    -------\n    X_mask : {ndarray, sparse matrix} of shape (n_samples, n_features)\n        Missing mask.\n    '
    if not sp.issparse(X):
        return _get_dense_mask(X, value_to_mask)
    Xt = _get_dense_mask(X.data, value_to_mask)
    sparse_constructor = sp.csr_matrix if X.format == 'csr' else sp.csc_matrix
    Xt_sparse = sparse_constructor((Xt, X.indices.copy(), X.indptr.copy()), shape=X.shape, dtype=bool)
    return Xt_sparse