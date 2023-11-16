"""Copyright (c) 2015 The auto-sklearn developers. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of the auto-sklearn Developers  nor the names of
     its contributors may be used to endorse or promote products
     derived from this software without specific prior written
     permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
SPARSE_ENCODINGS = {'OTHER': 1, 'NAN': 2}

def auto_select_categorical_features(X, threshold=10):
    if False:
        while True:
            i = 10
    'Make a feature mask of categorical features in X.\n\n    Features with less than 10 unique values are considered categorical.\n\n    Parameters\n    ----------\n    X : array-like or sparse matrix, shape=(n_samples, n_features)\n        Dense array or sparse matrix.\n\n    threshold : int\n        Maximum number of unique values per feature to consider the feature\n        to be categorical.\n\n    Returns\n    -------\n    feature_mask : array of booleans of size {n_features, }\n    '
    feature_mask = []
    for column in range(X.shape[1]):
        if sparse.issparse(X):
            indptr_start = X.indptr[column]
            indptr_end = X.indptr[column + 1]
            unique = np.unique(X.data[indptr_start:indptr_end])
        else:
            unique = np.unique(X[:, column])
        feature_mask.append(len(unique) <= threshold)
    return feature_mask

def _X_selected(X, selected):
    if False:
        for i in range(10):
            print('nop')
    'Split X into selected features and other features'
    n_features = X.shape[1]
    ind = np.arange(n_features)
    sel = np.zeros(n_features, dtype=bool)
    sel[np.asarray(selected)] = True
    non_sel = np.logical_not(sel)
    n_selected = np.sum(sel)
    X_sel = X[:, ind[sel]]
    X_not_sel = X[:, ind[non_sel]]
    return (X_sel, X_not_sel, n_selected, n_features)

def _transform_selected(X, transform, selected, copy=True):
    if False:
        print('Hello World!')
    'Apply a transform function to portion of selected features.\n\n    Parameters\n    ----------\n    X : array-like or sparse matrix, shape=(n_samples, n_features)\n        Dense array or sparse matrix.\n\n    transform : callable\n        A callable transform(X) -> X_transformed\n\n    copy : boolean, optional\n        Copy X even if it could be avoided.\n\n    selected: "all", "auto" or array of indices or mask\n        Specify which features to apply the transform to.\n\n    Returns\n    -------\n    X : array or sparse matrix, shape=(n_samples, n_features_new)\n    '
    if selected == 'all':
        return transform(X)
    if len(selected) == 0:
        return X
    X = check_array(X, accept_sparse='csc', force_all_finite=False)
    (X_sel, X_not_sel, n_selected, n_features) = _X_selected(X, selected)
    if n_selected == 0:
        return X
    elif n_selected == n_features:
        return transform(X)
    else:
        X_sel = transform(X_sel)
        if sparse.issparse(X_sel) or sparse.issparse(X_not_sel):
            return sparse.hstack((X_sel, X_not_sel), format='csr')
        else:
            return np.hstack((X_sel, X_not_sel))

class OneHotEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical integer features using a one-hot aka one-of-K scheme.

    The input to this transformer should be a matrix of integers, denoting
    the values taken on by categorical (discrete) features. The output will be
    a sparse matrix were each column corresponds to one possible value of one
    feature. It is assumed that input features take on values in the range
    [0, n_values).

    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.

    Parameters
    ----------

    categorical_features: "all" or array of indices or mask
        Specify what features are treated as categorical.

        - 'all': All features are treated as categorical.
        - 'auto' (default): Select only features that have less than 10 unique values.
        - array of indices: Array of categorical feature indices.
        - mask: Array of length n_features and with dtype=bool.

        Non-categorical features are always stacked to the right of the matrix.

    dtype : number type, default=np.float64
        Desired dtype of output.

    sparse : boolean, default=True
        Will return sparse matrix if set True else will return an array.

    threshold : int, default=10
        Maximum number of unique values per feature to consider the feature
        to be categorical when categorical_features is 'auto'.

    minimum_fraction : float, default=None
        Minimum fraction of unique values in a feature to consider the feature
        to be categorical.

    Attributes
    ----------
    `active_features_` : array
        Indices for active features, meaning values that actually occur
        in the training set. Only available when n_values is ``'auto'``.

    `feature_indices_` : array of shape (n_features,)
        Indices to feature ranges.
        Feature ``i`` in the original data is mapped to features
        from ``feature_indices_[i]`` to ``feature_indices_[i+1]``
        (and then potentially masked by `active_features_` afterwards)

    `n_values_` : array of shape (n_features,)
        Maximum number of values per feature.

    Examples
    --------
    Given a dataset with three features and two samples, we let the encoder
    find the maximum value per feature and transform the data to a binary
    one-hot encoding.

    >>> from sklearn.preprocessing import OneHotEncoder
    >>> enc = OneHotEncoder()
    >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])  # doctest: +ELLIPSIS
    OneHotEncoder(categorical_features='all', dtype=<... 'float'>,
           sparse=True, minimum_fraction=None)
    >>> enc.n_values_
    array([2, 3, 4])
    >>> enc.feature_indices_
    array([0, 2, 5, 9])
    >>> enc.transform([[0, 1, 1]]).toarray()
    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.]])

    See also
    --------
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """

    def __init__(self, categorical_features='auto', dtype=np.float64, sparse=True, minimum_fraction=None, threshold=10):
        if False:
            i = 10
            return i + 15
        self.categorical_features = categorical_features
        self.dtype = dtype
        self.sparse = sparse
        self.minimum_fraction = minimum_fraction
        self.threshold = threshold

    def fit(self, X, y=None):
        if False:
            return 10
        'Fit OneHotEncoder to X.\n\n        Parameters\n        ----------\n        X : array-like, shape=(n_samples, n_feature)\n            Input array of type int.\n\n        Returns\n        -------\n        self\n        '
        self.fit_transform(X)
        return self

    def _matrix_adjust(self, X):
        if False:
            while True:
                i = 10
        'Adjust all values in X to encode for NaNs and infinities in the data.\n\n        Parameters\n        ----------\n        X : array-like, shape=(n_samples, n_feature)\n            Input array of type int.\n\n        Returns\n        -------\n        X : array-like, shape=(n_samples, n_feature)\n            Input array without any NaNs or infinities.\n        '
        data_matrix = X.data if sparse.issparse(X) else X
        data_matrix += len(SPARSE_ENCODINGS) + 1
        data_matrix[~np.isfinite(data_matrix)] = SPARSE_ENCODINGS['NAN']
        return X

    def _fit_transform(self, X):
        if False:
            while True:
                i = 10
        'Assume X contains only categorical features.\n\n        Parameters\n        ----------\n        X : array-like or sparse matrix, shape=(n_samples, n_features)\n            Dense array or sparse matrix.\n        '
        X = self._matrix_adjust(X)
        X = check_array(X, accept_sparse='csc', force_all_finite=False, dtype=int)
        if X.min() < 0:
            raise ValueError('X needs to contain only non-negative integers.')
        (n_samples, n_features) = X.shape
        if self.minimum_fraction is not None:
            do_not_replace_by_other = []
            for column in range(X.shape[1]):
                do_not_replace_by_other.append(list())
                if sparse.issparse(X):
                    indptr_start = X.indptr[column]
                    indptr_end = X.indptr[column + 1]
                    unique = np.unique(X.data[indptr_start:indptr_end])
                    colsize = indptr_end - indptr_start
                else:
                    unique = np.unique(X[:, column])
                    colsize = X.shape[0]
                for unique_value in unique:
                    if np.isfinite(unique_value):
                        if sparse.issparse(X):
                            indptr_start = X.indptr[column]
                            indptr_end = X.indptr[column + 1]
                            count = np.nansum(unique_value == X.data[indptr_start:indptr_end])
                        else:
                            count = np.nansum(unique_value == X[:, column])
                    elif sparse.issparse(X):
                        indptr_start = X.indptr[column]
                        indptr_end = X.indptr[column + 1]
                        count = np.nansum(~np.isfinite(X.data[indptr_start:indptr_end]))
                    else:
                        count = np.nansum(~np.isfinite(X[:, column]))
                    fraction = float(count) / colsize
                    if fraction >= self.minimum_fraction:
                        do_not_replace_by_other[-1].append(unique_value)
                for unique_value in unique:
                    if unique_value not in do_not_replace_by_other[-1]:
                        if sparse.issparse(X):
                            indptr_start = X.indptr[column]
                            indptr_end = X.indptr[column + 1]
                            X.data[indptr_start:indptr_end][X.data[indptr_start:indptr_end] == unique_value] = SPARSE_ENCODINGS['OTHER']
                        else:
                            X[:, column][X[:, column] == unique_value] = SPARSE_ENCODINGS['OTHER']
            self.do_not_replace_by_other_ = do_not_replace_by_other
        if sparse.issparse(X):
            n_values = X.max(axis=0).toarray().flatten() + len(SPARSE_ENCODINGS)
        else:
            n_values = np.max(X, axis=0) + len(SPARSE_ENCODINGS)
        self.n_values_ = n_values
        n_values = np.hstack([[0], n_values])
        indices = np.cumsum(n_values)
        self.feature_indices_ = indices
        if sparse.issparse(X):
            row_indices = X.indices
            column_indices = []
            for i in range(len(X.indptr) - 1):
                nbr = X.indptr[i + 1] - X.indptr[i]
                column_indices_ = [indices[i]] * nbr
                column_indices_ += X.data[X.indptr[i]:X.indptr[i + 1]]
                column_indices.extend(column_indices_)
            data = np.ones(X.data.size)
        else:
            column_indices = (X + indices[:-1]).ravel()
            row_indices = np.repeat(np.arange(n_samples, dtype=np.int32), n_features)
            data = np.ones(n_samples * n_features)
        out = sparse.coo_matrix((data, (row_indices, column_indices)), shape=(n_samples, indices[-1]), dtype=self.dtype).tocsc()
        mask = np.array(out.sum(axis=0)).ravel() != 0
        active_features = np.where(mask)[0]
        out = out[:, active_features]
        self.active_features_ = active_features
        return out.tocsr() if self.sparse else out.toarray()

    def fit_transform(self, X, y=None):
        if False:
            i = 10
            return i + 15
        'Fit OneHotEncoder to X, then transform X.\n\n        Equivalent to self.fit(X).transform(X), but more convenient and more\n        efficient. See fit for the parameters, transform for the return value.\n\n        Parameters\n        ----------\n        X : array-like or sparse matrix, shape=(n_samples, n_features)\n            Dense array or sparse matrix.\n        y: array-like {n_samples,} (Optional, ignored)\n            Feature labels\n        '
        if self.categorical_features == 'auto':
            self.categorical_features_ = auto_select_categorical_features(X, threshold=self.threshold)
        else:
            self.categorical_features_ = self.categorical_features
        return _transform_selected(X, self._fit_transform, self.categorical_features_, copy=True)

    def _transform(self, X):
        if False:
            for i in range(10):
                print('nop')
        'Asssume X contains only categorical features.\n\n        Parameters\n        ----------\n        X : array-like or sparse matrix, shape=(n_samples, n_features)\n            Dense array or sparse matrix.\n        '
        X = self._matrix_adjust(X)
        X = check_array(X, accept_sparse='csc', force_all_finite=False, dtype=int)
        if X.min() < 0:
            raise ValueError('X needs to contain only non-negative integers.')
        (n_samples, n_features) = X.shape
        indices = self.feature_indices_
        if n_features != indices.shape[0] - 1:
            raise ValueError('X has different shape than during fitting. Expected %d, got %d.' % (indices.shape[0] - 1, n_features))
        if self.minimum_fraction is not None:
            for column in range(X.shape[1]):
                if sparse.issparse(X):
                    indptr_start = X.indptr[column]
                    indptr_end = X.indptr[column + 1]
                    unique = np.unique(X.data[indptr_start:indptr_end])
                else:
                    unique = np.unique(X[:, column])
                for unique_value in unique:
                    if unique_value not in self.do_not_replace_by_other_[column]:
                        if sparse.issparse(X):
                            indptr_start = X.indptr[column]
                            indptr_end = X.indptr[column + 1]
                            X.data[indptr_start:indptr_end][X.data[indptr_start:indptr_end] == unique_value] = SPARSE_ENCODINGS['OTHER']
                        else:
                            X[:, column][X[:, column] == unique_value] = SPARSE_ENCODINGS['OTHER']
        if sparse.issparse(X):
            n_values_check = X.max(axis=0).toarray().flatten() + 1
        else:
            n_values_check = np.max(X, axis=0) + 1
        if (n_values_check > self.n_values_).any():
            for (i, n_value_check) in enumerate(n_values_check):
                if n_value_check - 1 >= self.n_values_[i]:
                    if sparse.issparse(X):
                        indptr_start = X.indptr[i]
                        indptr_end = X.indptr[i + 1]
                        X.data[indptr_start:indptr_end][X.data[indptr_start:indptr_end] >= self.n_values_[i]] = 0
                    else:
                        X[:, i][X[:, i] >= self.n_values_[i]] = 0
        if sparse.issparse(X):
            row_indices = X.indices
            column_indices = []
            for i in range(len(X.indptr) - 1):
                nbr = X.indptr[i + 1] - X.indptr[i]
                column_indices_ = [indices[i]] * nbr
                column_indices_ += X.data[X.indptr[i]:X.indptr[i + 1]]
                column_indices.extend(column_indices_)
            data = np.ones(X.data.size)
        else:
            column_indices = (X + indices[:-1]).ravel()
            row_indices = np.repeat(np.arange(n_samples, dtype=np.int32), n_features)
            data = np.ones(n_samples * n_features)
        out = sparse.coo_matrix((data, (row_indices, column_indices)), shape=(n_samples, indices[-1]), dtype=self.dtype).tocsc()
        out = out[:, self.active_features_]
        return out.tocsr() if self.sparse else out.toarray()

    def transform(self, X):
        if False:
            i = 10
            return i + 15
        'Transform X using one-hot encoding.\n\n        Parameters\n        ----------\n        X : array-like or sparse matrix, shape=(n_samples, n_features)\n            Dense array or sparse matrix.\n\n        Returns\n        -------\n        X_out : sparse matrix if sparse=True else a 2-d array, dtype=int\n            Transformed input.\n        '
        return _transform_selected(X, self._transform, self.categorical_features_, copy=True)