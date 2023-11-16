"""
This module merges two files from Scikit-Learn 0.20 to make a few encoders
available for users using an earlier version:
    * sklearn/preprocessing/data.py (OneHotEncoder and CategoricalEncoder)
    * sklearn/compose/_column_transformer.py (ColumnTransformer)
I just copy/pasted the contents, fixed the imports and __all__, and also
copied the definitions of three pipeline functions whose signature changes
in 0.20: _fit_one_transformer, _transform_one and _fit_transform_one.
The original authors are listed below.
----
The :mod:`sklearn.compose._column_transformer` module implements utilities
to work with heterogeneous data and to apply different transformers to
different columns.
"""
from __future__ import division
import numbers
import warnings
import numpy as np
from scipy import sparse
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.externals import six
from sklearn.utils import Bunch, check_array
from sklearn.externals.joblib.parallel import delayed, Parallel
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
from sklearn.pipeline import _name_estimators
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing.label import LabelEncoder
from itertools import chain

def _fit_one_transformer(transformer, X, y, weight=None, **fit_params):
    if False:
        print('Hello World!')
    return transformer.fit(X, y)

def _transform_one(transformer, X, y, weight, **fit_params):
    if False:
        for i in range(10):
            print('nop')
    res = transformer.transform(X)
    if weight is None:
        return res
    return res * weight

def _fit_transform_one(transformer, X, y, weight, **fit_params):
    if False:
        while True:
            i = 10
    if hasattr(transformer, 'fit_transform'):
        res = transformer.fit_transform(X, y, **fit_params)
    else:
        res = transformer.fit(X, y, **fit_params).transform(X)
    if weight is None:
        return (res, transformer)
    return (res * weight, transformer)
BOUNDS_THRESHOLD = 1e-07
zip = six.moves.zip
map = six.moves.map
range = six.moves.range
__all__ = ['OneHotEncoder', 'OrdinalEncoder', 'ColumnTransformer', 'make_column_transformer']

def _argmax(arr_or_spmatrix, axis=None):
    if False:
        i = 10
        return i + 15
    return arr_or_spmatrix.argmax(axis=axis)

def _handle_zeros_in_scale(scale, copy=True):
    if False:
        print('Hello World!')
    ' Makes sure that whenever scale is zero, we handle it correctly.\n\n    This happens in most scalers when we have constant features.'
    if np.isscalar(scale):
        if scale == 0.0:
            scale = 1.0
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale

def _transform_selected(X, transform, selected='all', copy=True):
    if False:
        return 10
    'Apply a transform function to portion of selected features\n\n    Parameters\n    ----------\n    X : {array-like, sparse matrix}, shape [n_samples, n_features]\n        Dense array or sparse matrix.\n\n    transform : callable\n        A callable transform(X) -> X_transformed\n\n    copy : boolean, optional\n        Copy X even if it could be avoided.\n\n    selected: "all" or array of indices or mask\n        Specify which features to apply the transform to.\n\n    Returns\n    -------\n    X : array or sparse matrix, shape=(n_samples, n_features_new)\n    '
    X = check_array(X, accept_sparse='csc', copy=copy, dtype=FLOAT_DTYPES)
    if isinstance(selected, six.string_types) and selected == 'all':
        return transform(X)
    if len(selected) == 0:
        return X
    n_features = X.shape[1]
    ind = np.arange(n_features)
    sel = np.zeros(n_features, dtype=bool)
    sel[np.asarray(selected)] = True
    not_sel = np.logical_not(sel)
    n_selected = np.sum(sel)
    if n_selected == 0:
        return X
    elif n_selected == n_features:
        return transform(X)
    else:
        X_sel = transform(X[:, ind[sel]])
        X_not_sel = X[:, ind[not_sel]]
        if sparse.issparse(X_sel) or sparse.issparse(X_not_sel):
            return sparse.hstack((X_sel, X_not_sel))
        else:
            return np.hstack((X_sel, X_not_sel))

class _BaseEncoder(BaseEstimator, TransformerMixin):
    """
    Base class for encoders that includes the code to categorize and
    transform the input features.

    """

    def _fit(self, X, handle_unknown='error'):
        if False:
            print('Hello World!')
        X_temp = check_array(X, dtype=None)
        if not hasattr(X, 'dtype') and np.issubdtype(X_temp.dtype, np.str_):
            X = check_array(X, dtype=np.object)
        else:
            X = X_temp
        (n_samples, n_features) = X.shape
        if self.categories != 'auto':
            for cats in self.categories:
                if not np.all(np.sort(cats) == np.array(cats)):
                    raise ValueError('Unsorted categories are not yet supported')
            if len(self.categories) != n_features:
                raise ValueError('Shape mismatch: if n_values is an array, it has to be of shape (n_features,).')
        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]
        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                if handle_unknown == 'error':
                    valid_mask = np.in1d(Xi, self.categories[i])
                    if not np.all(valid_mask):
                        diff = np.unique(Xi[~valid_mask])
                        msg = 'Found unknown categories {0} in column {1} during fit'.format(diff, i)
                        raise ValueError(msg)
                le.classes_ = np.array(self.categories[i])
        self.categories_ = [le.classes_ for le in self._label_encoders_]

    def _transform(self, X, handle_unknown='error'):
        if False:
            return 10
        X_temp = check_array(X, dtype=None)
        if not hasattr(X, 'dtype') and np.issubdtype(X_temp.dtype, np.str_):
            X = check_array(X, dtype=np.object)
        else:
            X = X_temp
        (_, n_features) = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)
        for i in range(n_features):
            Xi = X[:, i]
            valid_mask = np.in1d(Xi, self.categories_[i])
            if not np.all(valid_mask):
                if handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = 'Found unknown categories {0} in column {1} during transform'.format(diff, i)
                    raise ValueError(msg)
                else:
                    X_mask[:, i] = valid_mask
                    Xi = Xi.copy()
                    Xi[~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(Xi)
        return (X_int, X_mask)
WARNING_MSG = 'The handling of integer data will change in the future. Currently, the categories are determined based on the range [0, max(values)], while in the future they will be determined based on the unique values.\nIf you want the future behaviour, you can specify "categories=\'auto\'".'

class OneHotEncoder(_BaseEncoder):
    """Encode categorical integer features as a one-hot numeric array.

    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are encoded using a one-hot (aka 'one-of-K' or 'dummy')
    encoding scheme. This creates a binary column for each category and
    returns a sparse matrix or dense array.

    By default, the encoder derives the categories based on the unique values
    in each feature. Alternatively, you can also specify the `categories`
    manually.
    The OneHotEncoder previously assumed that the input features take on
    values in the range [0, max(values)). This behaviour is deprecated.

    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.

    Note: a one-hot encoding of y labels should use a LabelBinarizer
    instead.

    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.

    Parameters
    ----------
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories must be sorted and should not mix
          strings and numeric values.

        The used categories can be found in the ``categories_`` attribute.

    sparse : boolean, default=True
        Will return sparse matrix if set True else will return an array.

    dtype : number type, default=np.float
        Desired dtype of output.

    handle_unknown : 'error' (default) or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform (default is to raise). When this parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros. In the inverse transform, an unknown category
        will be denoted as None.

    n_values : 'auto', int or array of ints
        Number of values per feature.

        - 'auto' : determine value range from training data.
        - int : number of categorical values per feature.
                Each feature value should be in ``range(n_values)``
        - array : ``n_values[i]`` is the number of categorical values in
                  ``X[:, i]``. Each feature value should be
                  in ``range(n_values[i])``

        .. deprecated:: 0.20
            The `n_values` keyword is deprecated and will be removed in 0.22.
            Use `categories` instead.

    categorical_features : "all" or array of indices or mask
        Specify what features are treated as categorical.

        - 'all' (default): All features are treated as categorical.
        - array of indices: Array of categorical feature indices.
        - mask: Array of length n_features and with dtype=bool.

        Non-categorical features are always stacked to the right of the matrix.

        .. deprecated:: 0.20
            The `categorical_features` keyword is deprecated and will be
            removed in 0.22.

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting
        (in order corresponding with output of ``transform``).

    active_features_ : array
        Indices for active features, meaning values that actually occur
        in the training set. Only available when n_values is ``'auto'``.

        .. deprecated:: 0.20

    feature_indices_ : array of shape (n_features,)
        Indices to feature ranges.
        Feature ``i`` in the original data is mapped to features
        from ``feature_indices_[i]`` to ``feature_indices_[i+1]``
        (and then potentially masked by `active_features_` afterwards)

        .. deprecated:: 0.20

    n_values_ : array of shape (n_features,)
        Maximum number of values per feature.

        .. deprecated:: 0.20

    Examples
    --------
    Given a dataset with two features, we let the encoder find the unique
    values per feature and transform the data to a binary one-hot encoding.

    >>> from sklearn.preprocessing import OneHotEncoder
    >>> enc = OneHotEncoder(handle_unknown='ignore')
    >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
    >>> enc.fit(X)
    ... # doctest: +ELLIPSIS
    OneHotEncoder(categories='auto', dtype=<... 'numpy.float64'>,
           handle_unknown='ignore', sparse=True)

    >>> enc.categories_
    [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    >>> enc.transform([['Female', 1], ['Male', 4]]).toarray()
    array([[ 1.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  0.,  0.,  0.]])
    >>> enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])
    array([['Male', 1],
           [None, 2]], dtype=object)

    See also
    --------
    sklearn.preprocessing.OrdinalEncoder : performs an ordinal (integer)
      encoding of the categorical features.
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    sklearn.preprocessing.LabelBinarizer : binarizes labels in a one-vs-all
      fashion.
    sklearn.preprocessing.MultiLabelBinarizer : transforms between iterable of
      iterables and a multilabel format, e.g. a (samples x classes) binary
      matrix indicating the presence of a class label.
    """

    def __init__(self, n_values=None, categorical_features=None, categories=None, sparse=True, dtype=np.float64, handle_unknown='error'):
        if False:
            for i in range(10):
                print('nop')
        self._categories = categories
        if categories is None:
            self.categories = 'auto'
        else:
            self.categories = categories
        self.sparse = sparse
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        if n_values is not None:
            pass
        else:
            n_values = 'auto'
        self._deprecated_n_values = n_values
        if categorical_features is not None:
            pass
        else:
            categorical_features = 'all'
        self._deprecated_categorical_features = categorical_features

    @property
    def n_values(self):
        if False:
            while True:
                i = 10
        warnings.warn("The 'n_values' parameter is deprecated.", DeprecationWarning)
        return self._deprecated_n_values

    @n_values.setter
    def n_values(self, value):
        if False:
            while True:
                i = 10
        warnings.warn("The 'n_values' parameter is deprecated.", DeprecationWarning)
        self._deprecated_n_values = value

    @property
    def categorical_features(self):
        if False:
            return 10
        warnings.warn("The 'categorical_features' parameter is deprecated.", DeprecationWarning)
        return self._deprecated_categorical_features

    @categorical_features.setter
    def categorical_features(self, value):
        if False:
            i = 10
            return i + 15
        warnings.warn("The 'categorical_features' parameter is deprecated.", DeprecationWarning)
        self._deprecated_categorical_features = value

    @property
    def active_features_(self):
        if False:
            i = 10
            return i + 15
        check_is_fitted(self, 'categories_')
        warnings.warn("The 'active_features_' attribute is deprecated.", DeprecationWarning)
        return self._active_features_

    @property
    def feature_indices_(self):
        if False:
            return 10
        check_is_fitted(self, 'categories_')
        warnings.warn("The 'feature_indices_' attribute is deprecated.", DeprecationWarning)
        return self._feature_indices_

    @property
    def n_values_(self):
        if False:
            while True:
                i = 10
        check_is_fitted(self, 'categories_')
        warnings.warn("The 'n_values_' attribute is deprecated.", DeprecationWarning)
        return self._n_values_

    def _handle_deprecations(self, X):
        if False:
            for i in range(10):
                print('nop')
        user_set_categories = False
        if self._categories is not None:
            self._legacy_mode = False
            user_set_categories = True
        elif self._deprecated_n_values != 'auto':
            msg = "Passing 'n_values' is deprecated and will be removed in a future release. You can use the 'categories' keyword instead. 'n_values=n' corresponds to 'n_values=[range(n)]'."
            warnings.warn(msg, DeprecationWarning)
            X = check_array(X, dtype=np.int)
            if isinstance(self._deprecated_n_values, numbers.Integral):
                n_features = X.shape[1]
                self.categories = [list(range(self._deprecated_n_values)) for _ in range(n_features)]
                n_values = np.empty(n_features, dtype=np.int)
                n_values.fill(self._deprecated_n_values)
            else:
                try:
                    n_values = np.asarray(self._deprecated_n_values, dtype=int)
                    self.categories = [list(range(i)) for i in self._deprecated_n_values]
                except (ValueError, TypeError):
                    raise TypeError("Wrong type for parameter `n_values`. Expected 'auto', int or array of ints, got %r".format(type(X)))
            self._n_values_ = n_values
            n_values = np.hstack([[0], n_values])
            indices = np.cumsum(n_values)
            self._feature_indices_ = indices
            self._legacy_mode = False
        elif self.handle_unknown == 'ignore':
            self._legacy_mode = False
        else:
            try:
                X = check_array(X, dtype=np.int)
            except ValueError:
                self._legacy_mode = False
            else:
                warnings.warn(WARNING_MSG, DeprecationWarning)
                self._legacy_mode = True
        if not isinstance(self._deprecated_categorical_features, six.string_types) or (isinstance(self._deprecated_categorical_features, six.string_types) and self._deprecated_categorical_features != 'all'):
            if user_set_categories:
                raise ValueError("The 'categorical_features' keyword is deprecated, and cannot be used together with specifying 'categories'.")
            warnings.warn("The 'categorical_features' keyword is deprecated.", DeprecationWarning)
            self._legacy_mode = True

    def fit(self, X, y=None):
        if False:
            for i in range(10):
                print('nop')
        'Fit OneHotEncoder to X.\n\n        Parameters\n        ----------\n        X : array-like, shape [n_samples, n_feature]\n            The data to determine the categories of each feature.\n\n        Returns\n        -------\n        self\n        '
        if self.handle_unknown not in ['error', 'ignore']:
            template = "handle_unknown should be either 'error' or 'ignore', got %s"
            raise ValueError(template % self.handle_unknown)
        self._handle_deprecations(X)
        if self._legacy_mode:
            self._legacy_fit_transform(X)
            return self
        else:
            self._fit(X, handle_unknown=self.handle_unknown)
            return self

    def _legacy_fit_transform(self, X):
        if False:
            i = 10
            return i + 15
        'Assumes X contains only categorical features.'
        self_n_values = self._deprecated_n_values
        dtype = getattr(X, 'dtype', None)
        X = check_array(X, dtype=np.int)
        if np.any(X < 0):
            raise ValueError('X needs to contain only non-negative integers.')
        (n_samples, n_features) = X.shape
        if isinstance(self_n_values, six.string_types) and self_n_values == 'auto':
            n_values = np.max(X, axis=0) + 1
        elif isinstance(self_n_values, numbers.Integral):
            if (np.max(X, axis=0) >= self_n_values).any():
                raise ValueError('Feature out of bounds for n_values=%d' % self_n_values)
            n_values = np.empty(n_features, dtype=np.int)
            n_values.fill(self_n_values)
        else:
            try:
                n_values = np.asarray(self_n_values, dtype=int)
            except (ValueError, TypeError):
                raise TypeError("Wrong type for parameter `n_values`. Expected 'auto', int or array of ints, got %r" % type(X))
            if n_values.ndim < 1 or n_values.shape[0] != X.shape[1]:
                raise ValueError('Shape mismatch: if n_values is an array, it has to be of shape (n_features,).')
        self._n_values_ = n_values
        self.categories_ = [np.arange(n_val - 1, dtype=dtype) for n_val in n_values]
        n_values = np.hstack([[0], n_values])
        indices = np.cumsum(n_values)
        self._feature_indices_ = indices
        column_indices = (X + indices[:-1]).ravel()
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32), n_features)
        data = np.ones(n_samples * n_features)
        out = sparse.coo_matrix((data, (row_indices, column_indices)), shape=(n_samples, indices[-1]), dtype=self.dtype).tocsr()
        if isinstance(self_n_values, six.string_types) and self_n_values == 'auto':
            mask = np.array(out.sum(axis=0)).ravel() != 0
            active_features = np.where(mask)[0]
            out = out[:, active_features]
            self._active_features_ = active_features
            self.categories_ = [np.unique(X[:, i]).astype(dtype) if dtype else np.unique(X[:, i]) for i in range(n_features)]
        return out if self.sparse else out.toarray()

    def fit_transform(self, X, y=None):
        if False:
            while True:
                i = 10
        'Fit OneHotEncoder to X, then transform X.\n\n        Equivalent to self.fit(X).transform(X), but more convenient and more\n        efficient. See fit for the parameters, transform for the return value.\n\n        Parameters\n        ----------\n        X : array-like, shape [n_samples, n_feature]\n            Input array of type int.\n        '
        if self.handle_unknown not in ['error', 'ignore']:
            template = "handle_unknown should be either 'error' or 'ignore', got %s"
            raise ValueError(template % self.handle_unknown)
        self._handle_deprecations(X)
        if self._legacy_mode:
            return _transform_selected(X, self._legacy_fit_transform, self._deprecated_categorical_features, copy=True)
        else:
            return self.fit(X).transform(X)

    def _legacy_transform(self, X):
        if False:
            return 10
        'Assumes X contains only categorical features.'
        self_n_values = self._deprecated_n_values
        X = check_array(X, dtype=np.int)
        if np.any(X < 0):
            raise ValueError('X needs to contain only non-negative integers.')
        (n_samples, n_features) = X.shape
        indices = self._feature_indices_
        if n_features != indices.shape[0] - 1:
            raise ValueError('X has different shape than during fitting. Expected %d, got %d.' % (indices.shape[0] - 1, n_features))
        mask = (X < self._n_values_).ravel()
        if np.any(~mask):
            if self.handle_unknown not in ['error', 'ignore']:
                raise ValueError('handle_unknown should be either error or unknown got %s' % self.handle_unknown)
            if self.handle_unknown == 'error':
                raise ValueError('unknown categorical feature present %s during transform.' % X.ravel()[~mask])
        column_indices = (X + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32), n_features)[mask]
        data = np.ones(np.sum(mask))
        out = sparse.coo_matrix((data, (row_indices, column_indices)), shape=(n_samples, indices[-1]), dtype=self.dtype).tocsr()
        if isinstance(self_n_values, six.string_types) and self_n_values == 'auto':
            out = out[:, self._active_features_]
        return out if self.sparse else out.toarray()

    def _transform_new(self, X):
        if False:
            return 10
        'New implementation assuming categorical input'
        X_temp = check_array(X, dtype=None)
        if not hasattr(X, 'dtype') and np.issubdtype(X_temp.dtype, np.str_):
            X = check_array(X, dtype=np.object)
        else:
            X = X_temp
        (n_samples, n_features) = X.shape
        (X_int, X_mask) = self._transform(X, handle_unknown=self.handle_unknown)
        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        feature_indices = np.cumsum(n_values)
        indices = (X_int + feature_indices[:-1]).ravel()[mask]
        indptr = X_mask.sum(axis=1).cumsum()
        indptr = np.insert(indptr, 0, 0)
        data = np.ones(n_samples * n_features)[mask]
        out = sparse.csr_matrix((data, indices, indptr), shape=(n_samples, feature_indices[-1]), dtype=self.dtype)
        if not self.sparse:
            return out.toarray()
        else:
            return out

    def transform(self, X):
        if False:
            print('Hello World!')
        'Transform X using one-hot encoding.\n\n        Parameters\n        ----------\n        X : array-like, shape [n_samples, n_features]\n            The data to encode.\n\n        Returns\n        -------\n        X_out : sparse matrix if sparse=True else a 2-d array\n            Transformed input.\n        '
        if not self._legacy_mode:
            return self._transform_new(X)
        else:
            return _transform_selected(X, self._legacy_transform, self._deprecated_categorical_features, copy=True)

    def inverse_transform(self, X):
        if False:
            print('Hello World!')
        "Convert back the data to the original representation.\n\n        In case unknown categories are encountered (all zero's in the\n        one-hot encoding), ``None`` is used to represent this category.\n\n        Parameters\n        ----------\n        X : array-like or sparse matrix, shape [n_samples, n_encoded_features]\n            The transformed data.\n\n        Returns\n        -------\n        X_tr : array-like, shape [n_samples, n_features]\n            Inverse transformed array.\n\n        "
        check_is_fitted(self, 'categories_')
        X = check_array(X, accept_sparse='csr')
        (n_samples, _) = X.shape
        n_features = len(self.categories_)
        n_transformed_features = sum([len(cats) for cats in self.categories_])
        msg = 'Shape of the passed X data is not correct. Expected {0} columns, got {1}.'
        if X.shape[1] != n_transformed_features:
            raise ValueError(msg.format(n_transformed_features, X.shape[1]))
        dt = np.find_common_type([cat.dtype for cat in self.categories_], [])
        X_tr = np.empty((n_samples, n_features), dtype=dt)
        j = 0
        found_unknown = {}
        for i in range(n_features):
            n_categories = len(self.categories_[i])
            sub = X[:, j:j + n_categories]
            labels = np.asarray(_argmax(sub, axis=1)).flatten()
            X_tr[:, i] = self.categories_[i][labels]
            if self.handle_unknown == 'ignore':
                unknown = np.asarray(sub.sum(axis=1) == 0).flatten()
                if unknown.any():
                    found_unknown[i] = unknown
            j += n_categories
        if found_unknown:
            if X_tr.dtype != object:
                X_tr = X_tr.astype(object)
            for (idx, mask) in found_unknown.items():
                X_tr[mask, idx] = None
        return X_tr

class OrdinalEncoder(_BaseEncoder):
    """Encode categorical features as an integer array.

    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are converted to ordinal integers. This results in
   a single column of integers (0 to n_categories - 1) per feature.

    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.

    Parameters
    ----------
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories must be sorted and should not mix
          strings and numeric values.

        The used categories can be found in the ``categories_`` attribute.

    dtype : number type, default np.float64
        Desired dtype of output.

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting
        (in order corresponding with output of ``transform``).

    Examples
    --------
    Given a dataset with two features, we let the encoder find the unique
    values per feature and transform the data to a binary one-hot encoding.

    >>> from sklearn.preprocessing import OrdinalEncoder
    >>> enc = OrdinalEncoder()
    >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
    >>> enc.fit(X)
    ... # doctest: +ELLIPSIS
    OrdinalEncoder(categories='auto', dtype=<... 'numpy.float64'>)
    >>> enc.categories_
    [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    >>> enc.transform([['Female', 3], ['Male', 1]])
    array([[ 0.,  2.],
           [ 1.,  0.]])

    >>> enc.inverse_transform([[1, 0], [0, 1]])
    array([['Male', 1],
           ['Female', 2]], dtype=object)

    See also
    --------
    sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of
      categorical features.
    sklearn.preprocessing.LabelEncoder : encodes target labels with values
      between 0 and n_classes-1.
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """

    def __init__(self, categories='auto', dtype=np.float64):
        if False:
            i = 10
            return i + 15
        self.categories = categories
        self.dtype = dtype

    def fit(self, X, y=None):
        if False:
            print('Hello World!')
        'Fit the OrdinalEncoder to X.\n\n        Parameters\n        ----------\n        X : array-like, shape [n_samples, n_features]\n            The data to determine the categories of each feature.\n\n        Returns\n        -------\n        self\n\n        '
        self._fit(X)
        return self

    def transform(self, X):
        if False:
            return 10
        'Transform X to ordinal codes.\n\n        Parameters\n        ----------\n        X : array-like, shape [n_samples, n_features]\n            The data to encode.\n\n        Returns\n        -------\n        X_out : sparse matrix or a 2-d array\n            Transformed input.\n\n        '
        (X_int, _) = self._transform(X)
        return X_int.astype(self.dtype, copy=False)

    def inverse_transform(self, X):
        if False:
            print('Hello World!')
        'Convert back the data to the original representation.\n\n        Parameters\n        ----------\n        X : array-like or sparse matrix, shape [n_samples, n_encoded_features]\n            The transformed data.\n\n        Returns\n        -------\n        X_tr : array-like, shape [n_samples, n_features]\n            Inverse transformed array.\n\n        '
        check_is_fitted(self, 'categories_')
        X = check_array(X, accept_sparse='csr')
        (n_samples, _) = X.shape
        n_features = len(self.categories_)
        msg = 'Shape of the passed X data is not correct. Expected {0} columns, got {1}.'
        if X.shape[1] != n_features:
            raise ValueError(msg.format(n_features, X.shape[1]))
        dt = np.find_common_type([cat.dtype for cat in self.categories_], [])
        X_tr = np.empty((n_samples, n_features), dtype=dt)
        for i in range(n_features):
            labels = X[:, i].astype('int64')
            X_tr[:, i] = self.categories_[i][labels]
        return X_tr
_ERR_MSG_1DCOLUMN = '1D data passed to a transformer that expects 2D data. Try to specify the column selection as a list of one item instead of a scalar.'

class ColumnTransformer(_BaseComposition, TransformerMixin):
    """Applies transformers to columns of an array or pandas DataFrame.

    EXPERIMENTAL: some behaviors may change between releases without
    deprecation.

    This estimator allows different columns or column subsets of the input
    to be transformed separately and the results combined into a single
    feature space.
    This is useful for heterogeneous or columnar data, to combine several
    feature extraction mechanisms or transformations into a single transformer.

    Read more in the :ref:`User Guide <column_transformer>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    transformers : list of tuples
        List of (name, transformer, column(s)) tuples specifying the
        transformer objects to be applied to subsets of the data.

        name : string
            Like in Pipeline and FeatureUnion, this allows the transformer and
            its parameters to be set using ``set_params`` and searched in grid
            search.
        transformer : estimator or {'passthrough', 'drop'}
            Estimator must support `fit` and `transform`. Special-cased
            strings 'drop' and 'passthrough' are accepted as well, to
            indicate to drop the columns or to pass them through untransformed,
            respectively.
        column(s) : string or int, array-like of string or int, slice, boolean mask array or callable
            Indexes the data on its second axis. Integers are interpreted as
            positional columns, while strings can reference DataFrame columns
            by name.  A scalar string or int should be used where
            ``transformer`` expects X to be a 1d array-like (vector),
            otherwise a 2d array will be passed to the transformer.
            A callable is passed the input data `X` and can return any of the
            above.

    remainder : {'drop', 'passthrough'} or estimator, default 'drop'
        By default, only the specified columns in `transformers` are
        transformed and combined in the output, and the non-specified
        columns are dropped. (default of ``'drop'``).
        By specifying ``remainder='passthrough'``, all remaining columns that
        were not specified in `transformers` will be automatically passed
        through. This subset of columns is concatenated with the output of
        the transformers.
        By setting ``remainder`` to be an estimator, the remaining
        non-specified columns will use the ``remainder`` estimator. The
        estimator must support `fit` and `transform`.

    sparse_threshold : float, default = 0.3
        If the transformed output consists of a mix of sparse and dense data,
        it will be stacked as a sparse matrix if the density is lower than this
        value. Use ``sparse_threshold=0`` to always return dense.
        When the transformed output consists of all sparse or all dense data,
        the stacked result will be sparse or dense, respectively, and this
        keyword will be ignored.

    n_jobs : int, optional
        Number of jobs to run in parallel (default 1).

    transformer_weights : dict, optional
        Multiplicative weights for features per transformer. The output of the
        transformer is multiplied by these weights. Keys are transformer names,
        values the weights.

    Attributes
    ----------
    transformers_ : list
        The collection of fitted transformers as tuples of
        (name, fitted_transformer, column). `fitted_transformer` can be an
        estimator, 'drop', or 'passthrough'. If there are remaining columns,
        the final element is a tuple of the form:
        ('remainder', transformer, remaining_columns) corresponding to the
        ``remainder`` parameter. If there are remaining columns, then
        ``len(transformers_)==len(transformers)+1``, otherwise
        ``len(transformers_)==len(transformers)``.

    named_transformers_ : Bunch object, a dictionary with attribute access
        Read-only attribute to access any transformer by given name.
        Keys are transformer names and values are the fitted transformer
        objects.

    sparse_output_ : boolean
        Boolean flag indicating wether the output of ``transform`` is a
        sparse matrix or a dense numpy array, which depends on the output
        of the individual transformers and the `sparse_threshold` keyword.

    Notes
    -----
    The order of the columns in the transformed feature matrix follows the
    order of how the columns are specified in the `transformers` list.
    Columns of the original feature matrix that are not specified are
    dropped from the resulting transformed feature matrix, unless specified
    in the `passthrough` keyword. Those columns specified with `passthrough`
    are added at the right to the output of the transformers.

    See also
    --------
    sklearn.compose.make_column_transformer : convenience function for
        combining the outputs of multiple transformer objects applied to
        column subsets of the original feature space.

    Examples
    --------
    >>> from sklearn.compose import ColumnTransformer
    >>> from sklearn.preprocessing import Normalizer
    >>> ct = ColumnTransformer(
    ...     [("norm1", Normalizer(norm='l1'), [0, 1]),
    ...      ("norm2", Normalizer(norm='l1'), slice(2, 4))])
    >>> X = np.array([[0., 1., 2., 2.],
    ...               [1., 1., 0., 1.]])
    >>> # Normalizer scales each row of X to unit norm. A separate scaling
    >>> # is applied for the two first and two last elements of each
    >>> # row independently.
    >>> ct.fit_transform(X)    # doctest: +NORMALIZE_WHITESPACE
    array([[0. , 1. , 0.5, 0.5],
           [0.5, 0.5, 0. , 1. ]])

    """

    def __init__(self, transformers, remainder='drop', sparse_threshold=0.3, n_jobs=1, transformer_weights=None):
        if False:
            while True:
                i = 10
        self.transformers = transformers
        self.remainder = remainder
        self.sparse_threshold = sparse_threshold
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights

    @property
    def _transformers(self):
        if False:
            return 10
        '\n        Internal list of transformer only containing the name and\n        transformers, dropping the columns. This is for the implementation\n        of get_params via BaseComposition._get_params which expects lists\n        of tuples of len 2.\n        '
        return [(name, trans) for (name, trans, _) in self.transformers]

    @_transformers.setter
    def _transformers(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.transformers = [(name, trans, col) for ((name, trans), (_, _, col)) in zip(value, self.transformers)]

    def get_params(self, deep=True):
        if False:
            print('Hello World!')
        'Get parameters for this estimator.\n\n        Parameters\n        ----------\n        deep : boolean, optional\n            If True, will return the parameters for this estimator and\n            contained subobjects that are estimators.\n\n        Returns\n        -------\n        params : mapping of string to any\n            Parameter names mapped to their values.\n        '
        return self._get_params('_transformers', deep=deep)

    def set_params(self, **kwargs):
        if False:
            while True:
                i = 10
        'Set the parameters of this estimator.\n\n        Valid parameter keys can be listed with ``get_params()``.\n\n        Returns\n        -------\n        self\n        '
        self._set_params('_transformers', **kwargs)
        return self

    def _iter(self, X=None, fitted=False, replace_strings=False):
        if False:
            return 10
        'Generate (name, trans, column, weight) tuples\n        '
        if fitted:
            transformers = self.transformers_
        else:
            transformers = self.transformers
            if self._remainder[2] is not None:
                transformers = chain(transformers, [self._remainder])
        get_weight = (self.transformer_weights or {}).get
        for (name, trans, column) in transformers:
            sub = None if X is None else _get_column(X, column)
            if replace_strings:
                if trans == 'passthrough':
                    trans = FunctionTransformer(validate=False, accept_sparse=True, check_inverse=False)
                elif trans == 'drop':
                    continue
            yield (name, trans, sub, get_weight(name))

    def _validate_transformers(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.transformers:
            return
        (names, transformers, _) = zip(*self.transformers)
        self._validate_names(names)
        for t in transformers:
            if t in ('drop', 'passthrough'):
                continue
            if not (hasattr(t, 'fit') or hasattr(t, 'fit_transform')) or not hasattr(t, 'transform'):
                raise TypeError("All estimators should implement fit and transform, or can be 'drop' or 'passthrough' specifiers. '%s' (type %s) doesn't." % (t, type(t)))

    def _validate_remainder(self, X):
        if False:
            for i in range(10):
                print('nop')
        '\n        Validates ``remainder`` and defines ``_remainder`` targeting\n        the remaining columns.\n        '
        is_transformer = (hasattr(self.remainder, 'fit') or hasattr(self.remainder, 'fit_transform')) and hasattr(self.remainder, 'transform')
        if self.remainder not in ('drop', 'passthrough') and (not is_transformer):
            raise ValueError("The remainder keyword needs to be one of 'drop', 'passthrough', or estimator. '%s' was passed instead" % self.remainder)
        n_columns = X.shape[1]
        cols = []
        for (_, _, columns) in self.transformers:
            cols.extend(_get_column_indices(X, columns))
        remaining_idx = sorted(list(set(range(n_columns)) - set(cols))) or None
        self._remainder = ('remainder', self.remainder, remaining_idx)

    @property
    def named_transformers_(self):
        if False:
            for i in range(10):
                print('nop')
        'Access the fitted transformer by name.\n\n        Read-only attribute to access any transformer by given name.\n        Keys are transformer names and values are the fitted transformer\n        objects.\n\n        '
        return Bunch(**dict([(name, trans) for (name, trans, _) in self.transformers_]))

    def get_feature_names(self):
        if False:
            while True:
                i = 10
        'Get feature names from all transformers.\n\n        Returns\n        -------\n        feature_names : list of strings\n            Names of the features produced by transform.\n        '
        check_is_fitted(self, 'transformers_')
        feature_names = []
        for (name, trans, _, _) in self._iter(fitted=True):
            if trans == 'drop':
                continue
            elif trans == 'passthrough':
                raise NotImplementedError("get_feature_names is not yet supported when using a 'passthrough' transformer.")
            elif not hasattr(trans, 'get_feature_names'):
                raise AttributeError('Transformer %s (type %s) does not provide get_feature_names.' % (str(name), type(trans).__name__))
            feature_names.extend([name + '__' + f for f in trans.get_feature_names()])
        return feature_names

    def _update_fitted_transformers(self, transformers):
        if False:
            i = 10
            return i + 15
        transformers = iter(transformers)
        transformers_ = []
        transformer_iter = self.transformers
        if self._remainder[2] is not None:
            transformer_iter = chain(transformer_iter, [self._remainder])
        for (name, old, column) in transformer_iter:
            if old == 'drop':
                trans = 'drop'
            elif old == 'passthrough':
                next(transformers)
                trans = 'passthrough'
            else:
                trans = next(transformers)
            transformers_.append((name, trans, column))
        assert not list(transformers)
        self.transformers_ = transformers_

    def _validate_output(self, result):
        if False:
            return 10
        '\n        Ensure that the output of each transformer is 2D. Otherwise\n        hstack can raise an error or produce incorrect results.\n        '
        names = [name for (name, _, _, _) in self._iter(replace_strings=True)]
        for (Xs, name) in zip(result, names):
            if not getattr(Xs, 'ndim', 0) == 2:
                raise ValueError("The output of the '{0}' transformer should be 2D (scipy matrix, array, or pandas DataFrame).".format(name))

    def _fit_transform(self, X, y, func, fitted=False):
        if False:
            print('Hello World!')
        '\n        Private function to fit and/or transform on demand.\n\n        Return value (transformers and/or transformed X data) depends\n        on the passed function.\n        ``fitted=True`` ensures the fitted transformers are used.\n        '
        try:
            return Parallel(n_jobs=self.n_jobs)((delayed(func)(clone(trans) if not fitted else trans, X_sel, y, weight) for (_, trans, X_sel, weight) in self._iter(X=X, fitted=fitted, replace_strings=True)))
        except ValueError as e:
            if 'Expected 2D array, got 1D array instead' in str(e):
                raise ValueError(_ERR_MSG_1DCOLUMN)
            else:
                raise

    def fit(self, X, y=None):
        if False:
            while True:
                i = 10
        'Fit all transformers using X.\n\n        Parameters\n        ----------\n        X : array-like or DataFrame of shape [n_samples, n_features]\n            Input data, of which specified subsets are used to fit the\n            transformers.\n\n        y : array-like, shape (n_samples, ...), optional\n            Targets for supervised learning.\n\n        Returns\n        -------\n        self : ColumnTransformer\n            This estimator\n\n        '
        self.fit_transform(X, y=y)
        return self

    def fit_transform(self, X, y=None):
        if False:
            print('Hello World!')
        'Fit all transformers, transform the data and concatenate results.\n\n        Parameters\n        ----------\n        X : array-like or DataFrame of shape [n_samples, n_features]\n            Input data, of which specified subsets are used to fit the\n            transformers.\n\n        y : array-like, shape (n_samples, ...), optional\n            Targets for supervised learning.\n\n        Returns\n        -------\n        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)\n            hstack of results of transformers. sum_n_components is the\n            sum of n_components (output dimension) over transformers. If\n            any result is a sparse matrix, everything will be converted to\n            sparse matrices.\n\n        '
        self._validate_remainder(X)
        self._validate_transformers()
        result = self._fit_transform(X, y, _fit_transform_one)
        if not result:
            self._update_fitted_transformers([])
            return np.zeros((X.shape[0], 0))
        (Xs, transformers) = zip(*result)
        if all((sparse.issparse(X) for X in Xs)):
            self.sparse_output_ = True
        elif any((sparse.issparse(X) for X in Xs)):
            nnz = sum((X.nnz if sparse.issparse(X) else X.size for X in Xs))
            total = sum((X.shape[0] * X.shape[1] if sparse.issparse(X) else X.size for X in Xs))
            density = nnz / total
            self.sparse_output_ = density < self.sparse_threshold
        else:
            self.sparse_output_ = False
        self._update_fitted_transformers(transformers)
        self._validate_output(Xs)
        return self._hstack(list(Xs))

    def transform(self, X):
        if False:
            print('Hello World!')
        'Transform X separately by each transformer, concatenate results.\n\n        Parameters\n        ----------\n        X : array-like or DataFrame of shape [n_samples, n_features]\n            The data to be transformed by subset.\n\n        Returns\n        -------\n        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)\n            hstack of results of transformers. sum_n_components is the\n            sum of n_components (output dimension) over transformers. If\n            any result is a sparse matrix, everything will be converted to\n            sparse matrices.\n\n        '
        check_is_fitted(self, 'transformers_')
        Xs = self._fit_transform(X, None, _transform_one, fitted=True)
        self._validate_output(Xs)
        if not Xs:
            return np.zeros((X.shape[0], 0))
        return self._hstack(list(Xs))

    def _hstack(self, Xs):
        if False:
            print('Hello World!')
        'Stacks Xs horizontally.\n\n        This allows subclasses to control the stacking behavior, while reusing\n        everything else from ColumnTransformer.\n\n        Parameters\n        ----------\n        Xs : List of numpy arrays, sparse arrays, or DataFrames\n        '
        if self.sparse_output_:
            return sparse.hstack(Xs).tocsr()
        else:
            Xs = [f.toarray() if sparse.issparse(f) else f for f in Xs]
            return np.hstack(Xs)

def _check_key_type(key, superclass):
    if False:
        i = 10
        return i + 15
    '\n    Check that scalar, list or slice is of a certain type.\n\n    This is only used in _get_column and _get_column_indices to check\n    if the `key` (column specification) is fully integer or fully string-like.\n\n    Parameters\n    ----------\n    key : scalar, list, slice, array-like\n        The column specification to check\n    superclass : int or six.string_types\n        The type for which to check the `key`\n\n    '
    if isinstance(key, superclass):
        return True
    if isinstance(key, slice):
        return isinstance(key.start, (superclass, type(None))) and isinstance(key.stop, (superclass, type(None)))
    if isinstance(key, list):
        return all((isinstance(x, superclass) for x in key))
    if hasattr(key, 'dtype'):
        if superclass is int:
            return key.dtype.kind == 'i'
        else:
            return key.dtype.kind in ('O', 'U', 'S')
    return False

def _get_column(X, key):
    if False:
        while True:
            i = 10
    '\n    Get feature column(s) from input data X.\n\n    Supported input types (X): numpy arrays, sparse arrays and DataFrames\n\n    Supported key types (key):\n    - scalar: output is 1D\n    - lists, slices, boolean masks: output is 2D\n    - callable that returns any of the above\n\n    Supported key data types:\n\n    - integer or boolean mask (positional):\n        - supported for arrays, sparse matrices and dataframes\n    - string (key-based):\n        - only supported for dataframes\n        - So no keys other than strings are allowed (while in principle you\n          can use any hashable object as key).\n\n    '
    if callable(key):
        key = key(X)
    if _check_key_type(key, int):
        column_names = False
    elif _check_key_type(key, six.string_types):
        column_names = True
    elif hasattr(key, 'dtype') and np.issubdtype(key.dtype, np.bool_):
        column_names = False
        if hasattr(X, 'loc'):
            column_names = True
    else:
        raise ValueError('No valid specification of the columns. Only a scalar, list or slice of all integers or all strings, or boolean mask is allowed')
    if column_names:
        if hasattr(X, 'loc'):
            return X.loc[:, key]
        else:
            raise ValueError('Specifying the columns using strings is only supported for pandas DataFrames')
    elif hasattr(X, 'iloc'):
        return X.iloc[:, key]
    else:
        return X[:, key]

def _get_column_indices(X, key):
    if False:
        return 10
    '\n    Get feature column indices for input data X and key.\n\n    For accepted values of `key`, see the docstring of _get_column\n\n    '
    n_columns = X.shape[1]
    if callable(key):
        key = key(X)
    if _check_key_type(key, int):
        if isinstance(key, int):
            return [key]
        elif isinstance(key, slice):
            return list(range(n_columns)[key])
        else:
            return list(key)
    elif _check_key_type(key, six.string_types):
        try:
            all_columns = list(X.columns)
        except AttributeError:
            raise ValueError('Specifying the columns using strings is only supported for pandas DataFrames')
        if isinstance(key, six.string_types):
            columns = [key]
        elif isinstance(key, slice):
            (start, stop) = (key.start, key.stop)
            if start is not None:
                start = all_columns.index(start)
            if stop is not None:
                stop = all_columns.index(stop) + 1
            else:
                stop = n_columns + 1
            return list(range(n_columns)[slice(start, stop)])
        else:
            columns = list(key)
        return [all_columns.index(col) for col in columns]
    elif hasattr(key, 'dtype') and np.issubdtype(key.dtype, np.bool_):
        return list(np.arange(n_columns)[key])
    else:
        raise ValueError('No valid specification of the columns. Only a scalar, list or slice of all integers or all strings, or boolean mask is allowed')

def _get_transformer_list(estimators):
    if False:
        for i in range(10):
            print('nop')
    '\n    Construct (name, trans, column) tuples from list\n\n    '
    transformers = [trans[1] for trans in estimators]
    columns = [trans[0] for trans in estimators]
    names = [trans[0] for trans in _name_estimators(transformers)]
    transformer_list = list(zip(names, transformers, columns))
    return transformer_list

def make_column_transformer(*transformers, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "Construct a ColumnTransformer from the given transformers.\n\n    This is a shorthand for the ColumnTransformer constructor; it does not\n    require, and does not permit, naming the transformers. Instead, they will\n    be given names automatically based on their types. It also does not allow\n    weighting.\n\n    Parameters\n    ----------\n    *transformers : tuples of column selections and transformers\n\n    remainder : {'drop', 'passthrough'} or estimator, default 'drop'\n        By default, only the specified columns in `transformers` are\n        transformed and combined in the output, and the non-specified\n        columns are dropped. (default of ``'drop'``).\n        By specifying ``remainder='passthrough'``, all remaining columns that\n        were not specified in `transformers` will be automatically passed\n        through. This subset of columns is concatenated with the output of\n        the transformers.\n        By setting ``remainder`` to be an estimator, the remaining\n        non-specified columns will use the ``remainder`` estimator. The\n        estimator must support `fit` and `transform`.\n\n    n_jobs : int, optional\n        Number of jobs to run in parallel (default 1).\n\n    Returns\n    -------\n    ct : ColumnTransformer\n\n    See also\n    --------\n    sklearn.compose.ColumnTransformer : Class that allows combining the\n        outputs of multiple transformer objects used on column subsets\n        of the data into a single feature space.\n\n    Examples\n    --------\n    >>> from sklearn.preprocessing import StandardScaler, OneHotEncoder\n    >>> from sklearn.compose import make_column_transformer\n    >>> make_column_transformer(\n    ...     (['numerical_column'], StandardScaler()),\n    ...     (['categorical_column'], OneHotEncoder()))\n    ...     # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS\n    ColumnTransformer(n_jobs=1, remainder='drop', sparse_threshold=0.3,\n             transformer_weights=None,\n             transformers=[('standardscaler',\n                            StandardScaler(...),\n                            ['numerical_column']),\n                           ('onehotencoder',\n                            OneHotEncoder(...),\n                            ['categorical_column'])])\n\n    "
    n_jobs = kwargs.pop('n_jobs', 1)
    remainder = kwargs.pop('remainder', 'drop')
    if kwargs:
        raise TypeError('Unknown keyword arguments: "{}"'.format(list(kwargs.keys())[0]))
    transformer_list = _get_transformer_list(transformers)
    return ColumnTransformer(transformer_list, n_jobs=n_jobs, remainder=remainder)