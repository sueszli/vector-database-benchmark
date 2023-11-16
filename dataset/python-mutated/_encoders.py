import numbers
import warnings
from numbers import Integral
import numpy as np
from scipy import sparse
from ..base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin, _fit_context
from ..utils import _safe_indexing, check_array, is_scalar_nan
from ..utils._encode import _check_unknown, _encode, _get_counts, _unique
from ..utils._mask import _get_mask
from ..utils._param_validation import Hidden, Interval, RealNotInt, StrOptions
from ..utils._set_output import _get_output_config
from ..utils.validation import _check_feature_names_in, check_is_fitted
__all__ = ['OneHotEncoder', 'OrdinalEncoder']

class _BaseEncoder(TransformerMixin, BaseEstimator):
    """
    Base class for encoders that includes the code to categorize and
    transform the input features.

    """

    def _check_X(self, X, force_all_finite=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform custom check_array:\n        - convert list of strings to object dtype\n        - check for missing values for object dtype data (check_array does\n          not do that)\n        - return list of features (arrays): this list of features is\n          constructed feature by feature to preserve the data types\n          of pandas DataFrame columns, as otherwise information is lost\n          and cannot be used, e.g. for the `categories_` attribute.\n\n        '
        if not (hasattr(X, 'iloc') and getattr(X, 'ndim', 0) == 2):
            X_temp = check_array(X, dtype=None, force_all_finite=force_all_finite)
            if not hasattr(X, 'dtype') and np.issubdtype(X_temp.dtype, np.str_):
                X = check_array(X, dtype=object, force_all_finite=force_all_finite)
            else:
                X = X_temp
            needs_validation = False
        else:
            needs_validation = force_all_finite
        (n_samples, n_features) = X.shape
        X_columns = []
        for i in range(n_features):
            Xi = _safe_indexing(X, indices=i, axis=1)
            Xi = check_array(Xi, ensure_2d=False, dtype=None, force_all_finite=needs_validation)
            X_columns.append(Xi)
        return (X_columns, n_samples, n_features)

    def _fit(self, X, handle_unknown='error', force_all_finite=True, return_counts=False, return_and_ignore_missing_for_infrequent=False):
        if False:
            for i in range(10):
                print('nop')
        self._check_infrequent_enabled()
        self._check_n_features(X, reset=True)
        self._check_feature_names(X, reset=True)
        (X_list, n_samples, n_features) = self._check_X(X, force_all_finite=force_all_finite)
        self.n_features_in_ = n_features
        if self.categories != 'auto':
            if len(self.categories) != n_features:
                raise ValueError('Shape mismatch: if categories is an array, it has to be of shape (n_features,).')
        self.categories_ = []
        category_counts = []
        compute_counts = return_counts or self._infrequent_enabled
        for i in range(n_features):
            Xi = X_list[i]
            if self.categories == 'auto':
                result = _unique(Xi, return_counts=compute_counts)
                if compute_counts:
                    (cats, counts) = result
                    category_counts.append(counts)
                else:
                    cats = result
            else:
                if np.issubdtype(Xi.dtype, np.str_):
                    Xi_dtype = object
                else:
                    Xi_dtype = Xi.dtype
                cats = np.array(self.categories[i], dtype=Xi_dtype)
                if cats.dtype == object and isinstance(cats[0], bytes) and (Xi.dtype.kind != 'S'):
                    msg = f"In column {i}, the predefined categories have type 'bytes' which is incompatible with values of type '{type(Xi[0]).__name__}'."
                    raise ValueError(msg)
                for category in cats[:-1]:
                    if is_scalar_nan(category):
                        raise ValueError(f'Nan should be the last element in user provided categories, see categories {cats} in column #{i}')
                if cats.size != len(_unique(cats)):
                    msg = f'In column {i}, the predefined categories contain duplicate elements.'
                    raise ValueError(msg)
                if Xi.dtype.kind not in 'OUS':
                    sorted_cats = np.sort(cats)
                    error_msg = 'Unsorted categories are not supported for numerical categories'
                    stop_idx = -1 if np.isnan(sorted_cats[-1]) else None
                    if np.any(sorted_cats[:stop_idx] != cats[:stop_idx]):
                        raise ValueError(error_msg)
                if handle_unknown == 'error':
                    diff = _check_unknown(Xi, cats)
                    if diff:
                        msg = 'Found unknown categories {0} in column {1} during fit'.format(diff, i)
                        raise ValueError(msg)
                if compute_counts:
                    category_counts.append(_get_counts(Xi, cats))
            self.categories_.append(cats)
        output = {'n_samples': n_samples}
        if return_counts:
            output['category_counts'] = category_counts
        missing_indices = {}
        if return_and_ignore_missing_for_infrequent:
            for (feature_idx, categories_for_idx) in enumerate(self.categories_):
                for (category_idx, category) in enumerate(categories_for_idx):
                    if is_scalar_nan(category):
                        missing_indices[feature_idx] = category_idx
                        break
            output['missing_indices'] = missing_indices
        if self._infrequent_enabled:
            self._fit_infrequent_category_mapping(n_samples, category_counts, missing_indices)
        return output

    def _transform(self, X, handle_unknown='error', force_all_finite=True, warn_on_unknown=False, ignore_category_indices=None):
        if False:
            return 10
        (X_list, n_samples, n_features) = self._check_X(X, force_all_finite=force_all_finite)
        self._check_feature_names(X, reset=False)
        self._check_n_features(X, reset=False)
        X_int = np.zeros((n_samples, n_features), dtype=int)
        X_mask = np.ones((n_samples, n_features), dtype=bool)
        columns_with_unknown = []
        for i in range(n_features):
            Xi = X_list[i]
            (diff, valid_mask) = _check_unknown(Xi, self.categories_[i], return_mask=True)
            if not np.all(valid_mask):
                if handle_unknown == 'error':
                    msg = 'Found unknown categories {0} in column {1} during transform'.format(diff, i)
                    raise ValueError(msg)
                else:
                    if warn_on_unknown:
                        columns_with_unknown.append(i)
                    X_mask[:, i] = valid_mask
                    if self.categories_[i].dtype.kind in ('U', 'S') and self.categories_[i].itemsize > Xi.itemsize:
                        Xi = Xi.astype(self.categories_[i].dtype)
                    elif self.categories_[i].dtype.kind == 'O' and Xi.dtype.kind == 'U':
                        Xi = Xi.astype('O')
                    else:
                        Xi = Xi.copy()
                    Xi[~valid_mask] = self.categories_[i][0]
            X_int[:, i] = _encode(Xi, uniques=self.categories_[i], check_unknown=False)
        if columns_with_unknown:
            warnings.warn(f'Found unknown categories in columns {columns_with_unknown} during transform. These unknown categories will be encoded as all zeros', UserWarning)
        self._map_infrequent_categories(X_int, X_mask, ignore_category_indices)
        return (X_int, X_mask)

    @property
    def infrequent_categories_(self):
        if False:
            return 10
        'Infrequent categories for each feature.'
        infrequent_indices = self._infrequent_indices
        return [None if indices is None else category[indices] for (category, indices) in zip(self.categories_, infrequent_indices)]

    def _check_infrequent_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This functions checks whether _infrequent_enabled is True or False.\n        This has to be called after parameter validation in the fit function.\n        '
        max_categories = getattr(self, 'max_categories', None)
        min_frequency = getattr(self, 'min_frequency', None)
        self._infrequent_enabled = max_categories is not None and max_categories >= 1 or min_frequency is not None

    def _identify_infrequent(self, category_count, n_samples, col_idx):
        if False:
            while True:
                i = 10
        'Compute the infrequent indices.\n\n        Parameters\n        ----------\n        category_count : ndarray of shape (n_cardinality,)\n            Category counts.\n\n        n_samples : int\n            Number of samples.\n\n        col_idx : int\n            Index of the current category. Only used for the error message.\n\n        Returns\n        -------\n        output : ndarray of shape (n_infrequent_categories,) or None\n            If there are infrequent categories, indices of infrequent\n            categories. Otherwise None.\n        '
        if isinstance(self.min_frequency, numbers.Integral):
            infrequent_mask = category_count < self.min_frequency
        elif isinstance(self.min_frequency, numbers.Real):
            min_frequency_abs = n_samples * self.min_frequency
            infrequent_mask = category_count < min_frequency_abs
        else:
            infrequent_mask = np.zeros(category_count.shape[0], dtype=bool)
        n_current_features = category_count.size - infrequent_mask.sum() + 1
        if self.max_categories is not None and self.max_categories < n_current_features:
            frequent_category_count = self.max_categories - 1
            if frequent_category_count == 0:
                infrequent_mask[:] = True
            else:
                smallest_levels = np.argsort(category_count, kind='mergesort')[:-frequent_category_count]
                infrequent_mask[smallest_levels] = True
        output = np.flatnonzero(infrequent_mask)
        return output if output.size > 0 else None

    def _fit_infrequent_category_mapping(self, n_samples, category_counts, missing_indices):
        if False:
            while True:
                i = 10
        'Fit infrequent categories.\n\n        Defines the private attribute: `_default_to_infrequent_mappings`. For\n        feature `i`, `_default_to_infrequent_mappings[i]` defines the mapping\n        from the integer encoding returned by `super().transform()` into\n        infrequent categories. If `_default_to_infrequent_mappings[i]` is None,\n        there were no infrequent categories in the training set.\n\n        For example if categories 0, 2 and 4 were frequent, while categories\n        1, 3, 5 were infrequent for feature 7, then these categories are mapped\n        to a single output:\n        `_default_to_infrequent_mappings[7] = array([0, 3, 1, 3, 2, 3])`\n\n        Defines private attribute: `_infrequent_indices`. `_infrequent_indices[i]`\n        is an array of indices such that\n        `categories_[i][_infrequent_indices[i]]` are all the infrequent category\n        labels. If the feature `i` has no infrequent categories\n        `_infrequent_indices[i]` is None.\n\n        .. versionadded:: 1.1\n\n        Parameters\n        ----------\n        n_samples : int\n            Number of samples in training set.\n        category_counts: list of ndarray\n            `category_counts[i]` is the category counts corresponding to\n            `self.categories_[i]`.\n        missing_indices : dict\n            Dict mapping from feature_idx to category index with a missing value.\n        '
        if missing_indices:
            category_counts_ = []
            for (feature_idx, count) in enumerate(category_counts):
                if feature_idx in missing_indices:
                    category_counts_.append(np.delete(count, missing_indices[feature_idx]))
                else:
                    category_counts_.append(count)
        else:
            category_counts_ = category_counts
        self._infrequent_indices = [self._identify_infrequent(category_count, n_samples, col_idx) for (col_idx, category_count) in enumerate(category_counts_)]
        self._default_to_infrequent_mappings = []
        for (feature_idx, infreq_idx) in enumerate(self._infrequent_indices):
            cats = self.categories_[feature_idx]
            if infreq_idx is None:
                self._default_to_infrequent_mappings.append(None)
                continue
            n_cats = len(cats)
            if feature_idx in missing_indices:
                n_cats -= 1
            mapping = np.empty(n_cats, dtype=np.int64)
            n_infrequent_cats = infreq_idx.size
            n_frequent_cats = n_cats - n_infrequent_cats
            mapping[infreq_idx] = n_frequent_cats
            frequent_indices = np.setdiff1d(np.arange(n_cats), infreq_idx)
            mapping[frequent_indices] = np.arange(n_frequent_cats)
            self._default_to_infrequent_mappings.append(mapping)

    def _map_infrequent_categories(self, X_int, X_mask, ignore_category_indices):
        if False:
            print('Hello World!')
        'Map infrequent categories to integer representing the infrequent category.\n\n        This modifies X_int in-place. Values that were invalid based on `X_mask`\n        are mapped to the infrequent category if there was an infrequent\n        category for that feature.\n\n        Parameters\n        ----------\n        X_int: ndarray of shape (n_samples, n_features)\n            Integer encoded categories.\n\n        X_mask: ndarray of shape (n_samples, n_features)\n            Bool mask for valid values in `X_int`.\n\n        ignore_category_indices : dict\n            Dictionary mapping from feature_idx to category index to ignore.\n            Ignored indexes will not be grouped and the original ordinal encoding\n            will remain.\n        '
        if not self._infrequent_enabled:
            return
        ignore_category_indices = ignore_category_indices or {}
        for col_idx in range(X_int.shape[1]):
            infrequent_idx = self._infrequent_indices[col_idx]
            if infrequent_idx is None:
                continue
            X_int[~X_mask[:, col_idx], col_idx] = infrequent_idx[0]
            if self.handle_unknown == 'infrequent_if_exist':
                X_mask[:, col_idx] = True
        for (i, mapping) in enumerate(self._default_to_infrequent_mappings):
            if mapping is None:
                continue
            if i in ignore_category_indices:
                rows_to_update = X_int[:, i] != ignore_category_indices[i]
            else:
                rows_to_update = slice(None)
            X_int[rows_to_update, i] = np.take(mapping, X_int[rows_to_update, i])

    def _more_tags(self):
        if False:
            while True:
                i = 10
        return {'X_types': ['2darray', 'categorical'], 'allow_nan': True}

class OneHotEncoder(_BaseEncoder):
    """
    Encode categorical features as a one-hot numeric array.

    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are encoded using a one-hot (aka 'one-of-K' or 'dummy')
    encoding scheme. This creates a binary column for each category and
    returns a sparse matrix or dense array (depending on the ``sparse_output``
    parameter).

    By default, the encoder derives the categories based on the unique values
    in each feature. Alternatively, you can also specify the `categories`
    manually.

    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.

    Note: a one-hot encoding of y labels should use a LabelBinarizer
    instead.

    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    For a comparison of different encoders, refer to:
    :ref:`sphx_glr_auto_examples_preprocessing_plot_target_encoder.py`.

    Parameters
    ----------
    categories : 'auto' or a list of array-like, default='auto'
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories should not mix strings and numeric
          values within a single feature, and should be sorted in case of
          numeric values.

        The used categories can be found in the ``categories_`` attribute.

        .. versionadded:: 0.20

    drop : {'first', 'if_binary'} or an array-like of shape (n_features,),             default=None
        Specifies a methodology to use to drop one of the categories per
        feature. This is useful in situations where perfectly collinear
        features cause problems, such as when feeding the resulting data
        into an unregularized linear regression model.

        However, dropping one category breaks the symmetry of the original
        representation and can therefore induce a bias in downstream models,
        for instance for penalized linear classification or regression models.

        - None : retain all features (the default).
        - 'first' : drop the first category in each feature. If only one
          category is present, the feature will be dropped entirely.
        - 'if_binary' : drop the first category in each feature with two
          categories. Features with 1 or more than 2 categories are
          left intact.
        - array : ``drop[i]`` is the category in feature ``X[:, i]`` that
          should be dropped.

        When `max_categories` or `min_frequency` is configured to group
        infrequent categories, the dropping behavior is handled after the
        grouping.

        .. versionadded:: 0.21
           The parameter `drop` was added in 0.21.

        .. versionchanged:: 0.23
           The option `drop='if_binary'` was added in 0.23.

        .. versionchanged:: 1.1
            Support for dropping infrequent categories.

    sparse : bool, default=True
        Will return sparse matrix if set True else will return an array.

        .. deprecated:: 1.2
           `sparse` is deprecated in 1.2 and will be removed in 1.4. Use
           `sparse_output` instead.

    sparse_output : bool, default=True
        When ``True``, it returns a :class:`scipy.sparse.csr_matrix`,
        i.e. a sparse matrix in "Compressed Sparse Row" (CSR) format.

        .. versionadded:: 1.2
           `sparse` was renamed to `sparse_output`

    dtype : number type, default=np.float64
        Desired dtype of output.

    handle_unknown : {'error', 'ignore', 'infrequent_if_exist'},                      default='error'
        Specifies the way unknown categories are handled during :meth:`transform`.

        - 'error' : Raise an error if an unknown category is present during transform.
        - 'ignore' : When an unknown category is encountered during
          transform, the resulting one-hot encoded columns for this feature
          will be all zeros. In the inverse transform, an unknown category
          will be denoted as None.
        - 'infrequent_if_exist' : When an unknown category is encountered
          during transform, the resulting one-hot encoded columns for this
          feature will map to the infrequent category if it exists. The
          infrequent category will be mapped to the last position in the
          encoding. During inverse transform, an unknown category will be
          mapped to the category denoted `'infrequent'` if it exists. If the
          `'infrequent'` category does not exist, then :meth:`transform` and
          :meth:`inverse_transform` will handle an unknown category as with
          `handle_unknown='ignore'`. Infrequent categories exist based on
          `min_frequency` and `max_categories`. Read more in the
          :ref:`User Guide <encoder_infrequent_categories>`.

        .. versionchanged:: 1.1
            `'infrequent_if_exist'` was added to automatically handle unknown
            categories and infrequent categories.

    min_frequency : int or float, default=None
        Specifies the minimum frequency below which a category will be
        considered infrequent.

        - If `int`, categories with a smaller cardinality will be considered
          infrequent.

        - If `float`, categories with a smaller cardinality than
          `min_frequency * n_samples`  will be considered infrequent.

        .. versionadded:: 1.1
            Read more in the :ref:`User Guide <encoder_infrequent_categories>`.

    max_categories : int, default=None
        Specifies an upper limit to the number of output features for each input
        feature when considering infrequent categories. If there are infrequent
        categories, `max_categories` includes the category representing the
        infrequent categories along with the frequent categories. If `None`,
        there is no limit to the number of output features.

        .. versionadded:: 1.1
            Read more in the :ref:`User Guide <encoder_infrequent_categories>`.

    feature_name_combiner : "concat" or callable, default="concat"
        Callable with signature `def callable(input_feature, category)` that returns a
        string. This is used to create feature names to be returned by
        :meth:`get_feature_names_out`.

        `"concat"` concatenates encoded feature name and category with
        `feature + "_" + str(category)`.E.g. feature X with values 1, 6, 7 create
        feature names `X_1, X_6, X_7`.

        .. versionadded:: 1.3

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting
        (in order of the features in X and corresponding with the output
        of ``transform``). This includes the category specified in ``drop``
        (if any).

    drop_idx_ : array of shape (n_features,)
        - ``drop_idx_[i]`` is the index in ``categories_[i]`` of the category
          to be dropped for each feature.
        - ``drop_idx_[i] = None`` if no category is to be dropped from the
          feature with index ``i``, e.g. when `drop='if_binary'` and the
          feature isn't binary.
        - ``drop_idx_ = None`` if all the transformed features will be
          retained.

        If infrequent categories are enabled by setting `min_frequency` or
        `max_categories` to a non-default value and `drop_idx[i]` corresponds
        to a infrequent category, then the entire infrequent category is
        dropped.

        .. versionchanged:: 0.23
           Added the possibility to contain `None` values.

    infrequent_categories_ : list of ndarray
        Defined only if infrequent categories are enabled by setting
        `min_frequency` or `max_categories` to a non-default value.
        `infrequent_categories_[i]` are the infrequent categories for feature
        `i`. If the feature `i` has no infrequent categories
        `infrequent_categories_[i]` is None.

        .. versionadded:: 1.1

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 1.0

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    feature_name_combiner : callable or None
        Callable with signature `def callable(input_feature, category)` that returns a
        string. This is used to create feature names to be returned by
        :meth:`get_feature_names_out`.

        .. versionadded:: 1.3

    See Also
    --------
    OrdinalEncoder : Performs an ordinal (integer)
      encoding of the categorical features.
    TargetEncoder : Encodes categorical features using the target.
    sklearn.feature_extraction.DictVectorizer : Performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : Performs an approximate one-hot
      encoding of dictionary items or strings.
    LabelBinarizer : Binarizes labels in a one-vs-all
      fashion.
    MultiLabelBinarizer : Transforms between iterable of
      iterables and a multilabel format, e.g. a (samples x classes) binary
      matrix indicating the presence of a class label.

    Examples
    --------
    Given a dataset with two features, we let the encoder find the unique
    values per feature and transform the data to a binary one-hot encoding.

    >>> from sklearn.preprocessing import OneHotEncoder

    One can discard categories not seen during `fit`:

    >>> enc = OneHotEncoder(handle_unknown='ignore')
    >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
    >>> enc.fit(X)
    OneHotEncoder(handle_unknown='ignore')
    >>> enc.categories_
    [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    >>> enc.transform([['Female', 1], ['Male', 4]]).toarray()
    array([[1., 0., 1., 0., 0.],
           [0., 1., 0., 0., 0.]])
    >>> enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])
    array([['Male', 1],
           [None, 2]], dtype=object)
    >>> enc.get_feature_names_out(['gender', 'group'])
    array(['gender_Female', 'gender_Male', 'group_1', 'group_2', 'group_3'], ...)

    One can always drop the first column for each feature:

    >>> drop_enc = OneHotEncoder(drop='first').fit(X)
    >>> drop_enc.categories_
    [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    >>> drop_enc.transform([['Female', 1], ['Male', 2]]).toarray()
    array([[0., 0., 0.],
           [1., 1., 0.]])

    Or drop a column for feature only having 2 categories:

    >>> drop_binary_enc = OneHotEncoder(drop='if_binary').fit(X)
    >>> drop_binary_enc.transform([['Female', 1], ['Male', 2]]).toarray()
    array([[0., 1., 0., 0.],
           [1., 0., 1., 0.]])

    One can change the way feature names are created.

    >>> def custom_combiner(feature, category):
    ...     return str(feature) + "_" + type(category).__name__ + "_" + str(category)
    >>> custom_fnames_enc = OneHotEncoder(feature_name_combiner=custom_combiner).fit(X)
    >>> custom_fnames_enc.get_feature_names_out()
    array(['x0_str_Female', 'x0_str_Male', 'x1_int_1', 'x1_int_2', 'x1_int_3'],
          dtype=object)

    Infrequent categories are enabled by setting `max_categories` or `min_frequency`.

    >>> import numpy as np
    >>> X = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3], dtype=object).T
    >>> ohe = OneHotEncoder(max_categories=3, sparse_output=False).fit(X)
    >>> ohe.infrequent_categories_
    [array(['a', 'd'], dtype=object)]
    >>> ohe.transform([["a"], ["b"]])
    array([[0., 0., 1.],
           [1., 0., 0.]])
    """
    _parameter_constraints: dict = {'categories': [StrOptions({'auto'}), list], 'drop': [StrOptions({'first', 'if_binary'}), 'array-like', None], 'dtype': 'no_validation', 'handle_unknown': [StrOptions({'error', 'ignore', 'infrequent_if_exist'})], 'max_categories': [Interval(Integral, 1, None, closed='left'), None], 'min_frequency': [Interval(Integral, 1, None, closed='left'), Interval(RealNotInt, 0, 1, closed='neither'), None], 'sparse': [Hidden(StrOptions({'deprecated'})), 'boolean'], 'sparse_output': ['boolean'], 'feature_name_combiner': [StrOptions({'concat'}), callable]}

    def __init__(self, *, categories='auto', drop=None, sparse='deprecated', sparse_output=True, dtype=np.float64, handle_unknown='error', min_frequency=None, max_categories=None, feature_name_combiner='concat'):
        if False:
            i = 10
            return i + 15
        self.categories = categories
        self.sparse = sparse
        self.sparse_output = sparse_output
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.drop = drop
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.feature_name_combiner = feature_name_combiner

    def _map_drop_idx_to_infrequent(self, feature_idx, drop_idx):
        if False:
            while True:
                i = 10
        'Convert `drop_idx` into the index for infrequent categories.\n\n        If there are no infrequent categories, then `drop_idx` is\n        returned. This method is called in `_set_drop_idx` when the `drop`\n        parameter is an array-like.\n        '
        if not self._infrequent_enabled:
            return drop_idx
        default_to_infrequent = self._default_to_infrequent_mappings[feature_idx]
        if default_to_infrequent is None:
            return drop_idx
        infrequent_indices = self._infrequent_indices[feature_idx]
        if infrequent_indices is not None and drop_idx in infrequent_indices:
            categories = self.categories_[feature_idx]
            raise ValueError(f'Unable to drop category {categories[drop_idx].item()!r} from feature {feature_idx} because it is infrequent')
        return default_to_infrequent[drop_idx]

    def _set_drop_idx(self):
        if False:
            return 10
        "Compute the drop indices associated with `self.categories_`.\n\n        If `self.drop` is:\n        - `None`, No categories have been dropped.\n        - `'first'`, All zeros to drop the first category.\n        - `'if_binary'`, All zeros if the category is binary and `None`\n          otherwise.\n        - array-like, The indices of the categories that match the\n          categories in `self.drop`. If the dropped category is an infrequent\n          category, then the index for the infrequent category is used. This\n          means that the entire infrequent category is dropped.\n\n        This methods defines a public `drop_idx_` and a private\n        `_drop_idx_after_grouping`.\n\n        - `drop_idx_`: Public facing API that references the drop category in\n          `self.categories_`.\n        - `_drop_idx_after_grouping`: Used internally to drop categories *after* the\n          infrequent categories are grouped together.\n\n        If there are no infrequent categories or drop is `None`, then\n        `drop_idx_=_drop_idx_after_grouping`.\n        "
        if self.drop is None:
            drop_idx_after_grouping = None
        elif isinstance(self.drop, str):
            if self.drop == 'first':
                drop_idx_after_grouping = np.zeros(len(self.categories_), dtype=object)
            elif self.drop == 'if_binary':
                n_features_out_no_drop = [len(cat) for cat in self.categories_]
                if self._infrequent_enabled:
                    for (i, infreq_idx) in enumerate(self._infrequent_indices):
                        if infreq_idx is None:
                            continue
                        n_features_out_no_drop[i] -= infreq_idx.size - 1
                drop_idx_after_grouping = np.array([0 if n_features_out == 2 else None for n_features_out in n_features_out_no_drop], dtype=object)
        else:
            drop_array = np.asarray(self.drop, dtype=object)
            droplen = len(drop_array)
            if droplen != len(self.categories_):
                msg = '`drop` should have length equal to the number of features ({}), got {}'
                raise ValueError(msg.format(len(self.categories_), droplen))
            missing_drops = []
            drop_indices = []
            for (feature_idx, (drop_val, cat_list)) in enumerate(zip(drop_array, self.categories_)):
                if not is_scalar_nan(drop_val):
                    drop_idx = np.where(cat_list == drop_val)[0]
                    if drop_idx.size:
                        drop_indices.append(self._map_drop_idx_to_infrequent(feature_idx, drop_idx[0]))
                    else:
                        missing_drops.append((feature_idx, drop_val))
                    continue
                for (cat_idx, cat) in enumerate(cat_list):
                    if is_scalar_nan(cat):
                        drop_indices.append(self._map_drop_idx_to_infrequent(feature_idx, cat_idx))
                        break
                else:
                    missing_drops.append((feature_idx, drop_val))
            if any(missing_drops):
                msg = 'The following categories were supposed to be dropped, but were not found in the training data.\n{}'.format('\n'.join(['Category: {}, Feature: {}'.format(c, v) for (c, v) in missing_drops]))
                raise ValueError(msg)
            drop_idx_after_grouping = np.array(drop_indices, dtype=object)
        self._drop_idx_after_grouping = drop_idx_after_grouping
        if not self._infrequent_enabled or drop_idx_after_grouping is None:
            self.drop_idx_ = self._drop_idx_after_grouping
        else:
            drop_idx_ = []
            for (feature_idx, drop_idx) in enumerate(drop_idx_after_grouping):
                default_to_infrequent = self._default_to_infrequent_mappings[feature_idx]
                if drop_idx is None or default_to_infrequent is None:
                    orig_drop_idx = drop_idx
                else:
                    orig_drop_idx = np.flatnonzero(default_to_infrequent == drop_idx)[0]
                drop_idx_.append(orig_drop_idx)
            self.drop_idx_ = np.asarray(drop_idx_, dtype=object)

    def _compute_transformed_categories(self, i, remove_dropped=True):
        if False:
            print('Hello World!')
        "Compute the transformed categories used for column `i`.\n\n        1. If there are infrequent categories, the category is named\n        'infrequent_sklearn'.\n        2. Dropped columns are removed when remove_dropped=True.\n        "
        cats = self.categories_[i]
        if self._infrequent_enabled:
            infreq_map = self._default_to_infrequent_mappings[i]
            if infreq_map is not None:
                frequent_mask = infreq_map < infreq_map.max()
                infrequent_cat = 'infrequent_sklearn'
                cats = np.concatenate((cats[frequent_mask], np.array([infrequent_cat], dtype=object)))
        if remove_dropped:
            cats = self._remove_dropped_categories(cats, i)
        return cats

    def _remove_dropped_categories(self, categories, i):
        if False:
            return 10
        'Remove dropped categories.'
        if self._drop_idx_after_grouping is not None and self._drop_idx_after_grouping[i] is not None:
            return np.delete(categories, self._drop_idx_after_grouping[i])
        return categories

    def _compute_n_features_outs(self):
        if False:
            return 10
        'Compute the n_features_out for each input feature.'
        output = [len(cats) for cats in self.categories_]
        if self._drop_idx_after_grouping is not None:
            for (i, drop_idx) in enumerate(self._drop_idx_after_grouping):
                if drop_idx is not None:
                    output[i] -= 1
        if not self._infrequent_enabled:
            return output
        for (i, infreq_idx) in enumerate(self._infrequent_indices):
            if infreq_idx is None:
                continue
            output[i] -= infreq_idx.size - 1
        return output

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        if False:
            while True:
                i = 10
        '\n        Fit OneHotEncoder to X.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            The data to determine the categories of each feature.\n\n        y : None\n            Ignored. This parameter exists only for compatibility with\n            :class:`~sklearn.pipeline.Pipeline`.\n\n        Returns\n        -------\n        self\n            Fitted encoder.\n        '
        if self.sparse != 'deprecated':
            warnings.warn('`sparse` was renamed to `sparse_output` in version 1.2 and will be removed in 1.4. `sparse_output` is ignored unless you leave `sparse` to its default value.', FutureWarning)
            self.sparse_output = self.sparse
        self._fit(X, handle_unknown=self.handle_unknown, force_all_finite='allow-nan')
        self._set_drop_idx()
        self._n_features_outs = self._compute_n_features_outs()
        return self

    def transform(self, X):
        if False:
            return 10
        '\n        Transform X using one-hot encoding.\n\n        If `sparse_output=True` (default), it returns an instance of\n        :class:`scipy.sparse._csr.csr_matrix` (CSR format).\n\n        If there are infrequent categories for a feature, set by specifying\n        `max_categories` or `min_frequency`, the infrequent categories are\n        grouped into a single category.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            The data to encode.\n\n        Returns\n        -------\n        X_out : {ndarray, sparse matrix} of shape                 (n_samples, n_encoded_features)\n            Transformed input. If `sparse_output=True`, a sparse matrix will be\n            returned.\n        '
        check_is_fitted(self)
        transform_output = _get_output_config('transform', estimator=self)['dense']
        if transform_output == 'pandas' and self.sparse_output:
            raise ValueError('Pandas output does not support sparse data. Set sparse_output=False to output pandas DataFrames or disable pandas output via `ohe.set_output(transform="default").')
        warn_on_unknown = self.drop is not None and self.handle_unknown in {'ignore', 'infrequent_if_exist'}
        (X_int, X_mask) = self._transform(X, handle_unknown=self.handle_unknown, force_all_finite='allow-nan', warn_on_unknown=warn_on_unknown)
        (n_samples, n_features) = X_int.shape
        if self._drop_idx_after_grouping is not None:
            to_drop = self._drop_idx_after_grouping.copy()
            keep_cells = X_int != to_drop
            for (i, cats) in enumerate(self.categories_):
                if to_drop[i] is None:
                    to_drop[i] = len(cats)
            to_drop = to_drop.reshape(1, -1)
            X_int[X_int > to_drop] -= 1
            X_mask &= keep_cells
        mask = X_mask.ravel()
        feature_indices = np.cumsum([0] + self._n_features_outs)
        indices = (X_int + feature_indices[:-1]).ravel()[mask]
        indptr = np.empty(n_samples + 1, dtype=int)
        indptr[0] = 0
        np.sum(X_mask, axis=1, out=indptr[1:], dtype=indptr.dtype)
        np.cumsum(indptr[1:], out=indptr[1:])
        data = np.ones(indptr[-1])
        out = sparse.csr_matrix((data, indices, indptr), shape=(n_samples, feature_indices[-1]), dtype=self.dtype)
        if not self.sparse_output:
            return out.toarray()
        else:
            return out

    def inverse_transform(self, X):
        if False:
            print('Hello World!')
        "\n        Convert the data back to the original representation.\n\n        When unknown categories are encountered (all zeros in the\n        one-hot encoding), ``None`` is used to represent this category. If the\n        feature with the unknown category has a dropped category, the dropped\n        category will be its inverse.\n\n        For a given input feature, if there is an infrequent category,\n        'infrequent_sklearn' will be used to represent the infrequent category.\n\n        Parameters\n        ----------\n        X : {array-like, sparse matrix} of shape                 (n_samples, n_encoded_features)\n            The transformed data.\n\n        Returns\n        -------\n        X_tr : ndarray of shape (n_samples, n_features)\n            Inverse transformed array.\n        "
        check_is_fitted(self)
        X = check_array(X, accept_sparse='csr')
        (n_samples, _) = X.shape
        n_features = len(self.categories_)
        n_features_out = np.sum(self._n_features_outs)
        msg = 'Shape of the passed X data is not correct. Expected {0} columns, got {1}.'
        if X.shape[1] != n_features_out:
            raise ValueError(msg.format(n_features_out, X.shape[1]))
        transformed_features = [self._compute_transformed_categories(i, remove_dropped=False) for (i, _) in enumerate(self.categories_)]
        dt = np.result_type(*[cat.dtype for cat in transformed_features])
        X_tr = np.empty((n_samples, n_features), dtype=dt)
        j = 0
        found_unknown = {}
        if self._infrequent_enabled:
            infrequent_indices = self._infrequent_indices
        else:
            infrequent_indices = [None] * n_features
        for i in range(n_features):
            cats_wo_dropped = self._remove_dropped_categories(transformed_features[i], i)
            n_categories = cats_wo_dropped.shape[0]
            if n_categories == 0:
                X_tr[:, i] = self.categories_[i][self._drop_idx_after_grouping[i]]
                j += n_categories
                continue
            sub = X[:, j:j + n_categories]
            labels = np.asarray(sub.argmax(axis=1)).flatten()
            X_tr[:, i] = cats_wo_dropped[labels]
            if self.handle_unknown == 'ignore' or (self.handle_unknown == 'infrequent_if_exist' and infrequent_indices[i] is None):
                unknown = np.asarray(sub.sum(axis=1) == 0).flatten()
                if unknown.any():
                    if self._drop_idx_after_grouping is None or self._drop_idx_after_grouping[i] is None:
                        found_unknown[i] = unknown
                    else:
                        X_tr[unknown, i] = self.categories_[i][self._drop_idx_after_grouping[i]]
            else:
                dropped = np.asarray(sub.sum(axis=1) == 0).flatten()
                if dropped.any():
                    if self._drop_idx_after_grouping is None:
                        all_zero_samples = np.flatnonzero(dropped)
                        raise ValueError(f"Samples {all_zero_samples} can not be inverted when drop=None and handle_unknown='error' because they contain all zeros")
                    drop_idx = self._drop_idx_after_grouping[i]
                    X_tr[dropped, i] = transformed_features[i][drop_idx]
            j += n_categories
        if found_unknown:
            if X_tr.dtype != object:
                X_tr = X_tr.astype(object)
            for (idx, mask) in found_unknown.items():
                X_tr[mask, idx] = None
        return X_tr

    def get_feature_names_out(self, input_features=None):
        if False:
            print('Hello World!')
        'Get output feature names for transformation.\n\n        Parameters\n        ----------\n        input_features : array-like of str or None, default=None\n            Input features.\n\n            - If `input_features` is `None`, then `feature_names_in_` is\n              used as feature names in. If `feature_names_in_` is not defined,\n              then the following input feature names are generated:\n              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.\n            - If `input_features` is an array-like, then `input_features` must\n              match `feature_names_in_` if `feature_names_in_` is defined.\n\n        Returns\n        -------\n        feature_names_out : ndarray of str objects\n            Transformed feature names.\n        '
        check_is_fitted(self)
        input_features = _check_feature_names_in(self, input_features)
        cats = [self._compute_transformed_categories(i) for (i, _) in enumerate(self.categories_)]
        name_combiner = self._check_get_feature_name_combiner()
        feature_names = []
        for i in range(len(cats)):
            names = [name_combiner(input_features[i], t) for t in cats[i]]
            feature_names.extend(names)
        return np.array(feature_names, dtype=object)

    def _check_get_feature_name_combiner(self):
        if False:
            for i in range(10):
                print('nop')
        if self.feature_name_combiner == 'concat':
            return lambda feature, category: feature + '_' + str(category)
        else:
            dry_run_combiner = self.feature_name_combiner('feature', 'category')
            if not isinstance(dry_run_combiner, str):
                raise TypeError(f'When `feature_name_combiner` is a callable, it should return a Python string. Got {type(dry_run_combiner)} instead.')
            return self.feature_name_combiner

class OrdinalEncoder(OneToOneFeatureMixin, _BaseEncoder):
    """
    Encode categorical features as an integer array.

    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are converted to ordinal integers. This results in
    a single column of integers (0 to n_categories - 1) per feature.

    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    For a comparison of different encoders, refer to:
    :ref:`sphx_glr_auto_examples_preprocessing_plot_target_encoder.py`.

    .. versionadded:: 0.20

    Parameters
    ----------
    categories : 'auto' or a list of array-like, default='auto'
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories should not mix strings and numeric
          values, and should be sorted in case of numeric values.

        The used categories can be found in the ``categories_`` attribute.

    dtype : number type, default=np.float64
        Desired dtype of output.

    handle_unknown : {'error', 'use_encoded_value'}, default='error'
        When set to 'error' an error will be raised in case an unknown
        categorical feature is present during transform. When set to
        'use_encoded_value', the encoded value of unknown categories will be
        set to the value given for the parameter `unknown_value`. In
        :meth:`inverse_transform`, an unknown category will be denoted as None.

        .. versionadded:: 0.24

    unknown_value : int or np.nan, default=None
        When the parameter handle_unknown is set to 'use_encoded_value', this
        parameter is required and will set the encoded value of unknown
        categories. It has to be distinct from the values used to encode any of
        the categories in `fit`. If set to np.nan, the `dtype` parameter must
        be a float dtype.

        .. versionadded:: 0.24

    encoded_missing_value : int or np.nan, default=np.nan
        Encoded value of missing categories. If set to `np.nan`, then the `dtype`
        parameter must be a float dtype.

        .. versionadded:: 1.1

    min_frequency : int or float, default=None
        Specifies the minimum frequency below which a category will be
        considered infrequent.

        - If `int`, categories with a smaller cardinality will be considered
          infrequent.

        - If `float`, categories with a smaller cardinality than
          `min_frequency * n_samples`  will be considered infrequent.

        .. versionadded:: 1.3
            Read more in the :ref:`User Guide <encoder_infrequent_categories>`.

    max_categories : int, default=None
        Specifies an upper limit to the number of output categories for each input
        feature when considering infrequent categories. If there are infrequent
        categories, `max_categories` includes the category representing the
        infrequent categories along with the frequent categories. If `None`,
        there is no limit to the number of output features.

        `max_categories` do **not** take into account missing or unknown
        categories. Setting `unknown_value` or `encoded_missing_value` to an
        integer will increase the number of unique integer codes by one each.
        This can result in up to `max_categories + 2` integer codes.

        .. versionadded:: 1.3
            Read more in the :ref:`User Guide <encoder_infrequent_categories>`.

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during ``fit`` (in order of
        the features in X and corresponding with the output of ``transform``).
        This does not include categories that weren't seen during ``fit``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 1.0

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    infrequent_categories_ : list of ndarray
        Defined only if infrequent categories are enabled by setting
        `min_frequency` or `max_categories` to a non-default value.
        `infrequent_categories_[i]` are the infrequent categories for feature
        `i`. If the feature `i` has no infrequent categories
        `infrequent_categories_[i]` is None.

        .. versionadded:: 1.3

    See Also
    --------
    OneHotEncoder : Performs a one-hot encoding of categorical features. This encoding
        is suitable for low to medium cardinality categorical variables, both in
        supervised and unsupervised settings.
    TargetEncoder : Encodes categorical features using supervised signal
        in a classification or regression pipeline. This encoding is typically
        suitable for high cardinality categorical variables.
    LabelEncoder : Encodes target labels with values between 0 and
        ``n_classes-1``.

    Notes
    -----
    With a high proportion of `nan` values, inferring categories becomes slow with
    Python versions before 3.10. The handling of `nan` values was improved
    from Python 3.10 onwards, (c.f.
    `bpo-43475 <https://github.com/python/cpython/issues/87641>`_).

    Examples
    --------
    Given a dataset with two features, we let the encoder find the unique
    values per feature and transform the data to an ordinal encoding.

    >>> from sklearn.preprocessing import OrdinalEncoder
    >>> enc = OrdinalEncoder()
    >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
    >>> enc.fit(X)
    OrdinalEncoder()
    >>> enc.categories_
    [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    >>> enc.transform([['Female', 3], ['Male', 1]])
    array([[0., 2.],
           [1., 0.]])

    >>> enc.inverse_transform([[1, 0], [0, 1]])
    array([['Male', 1],
           ['Female', 2]], dtype=object)

    By default, :class:`OrdinalEncoder` is lenient towards missing values by
    propagating them.

    >>> import numpy as np
    >>> X = [['Male', 1], ['Female', 3], ['Female', np.nan]]
    >>> enc.fit_transform(X)
    array([[ 1.,  0.],
           [ 0.,  1.],
           [ 0., nan]])

    You can use the parameter `encoded_missing_value` to encode missing values.

    >>> enc.set_params(encoded_missing_value=-1).fit_transform(X)
    array([[ 1.,  0.],
           [ 0.,  1.],
           [ 0., -1.]])

    Infrequent categories are enabled by setting `max_categories` or `min_frequency`.
    In the following example, "a" and "d" are considered infrequent and grouped
    together into a single category, "b" and "c" are their own categories, unknown
    values are encoded as 3 and missing values are encoded as 4.

    >>> X_train = np.array(
    ...     [["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3 + [np.nan]],
    ...     dtype=object).T
    >>> enc = OrdinalEncoder(
    ...     handle_unknown="use_encoded_value", unknown_value=3,
    ...     max_categories=3, encoded_missing_value=4)
    >>> _ = enc.fit(X_train)
    >>> X_test = np.array([["a"], ["b"], ["c"], ["d"], ["e"], [np.nan]], dtype=object)
    >>> enc.transform(X_test)
    array([[2.],
           [0.],
           [1.],
           [2.],
           [3.],
           [4.]])
    """
    _parameter_constraints: dict = {'categories': [StrOptions({'auto'}), list], 'dtype': 'no_validation', 'encoded_missing_value': [Integral, type(np.nan)], 'handle_unknown': [StrOptions({'error', 'use_encoded_value'})], 'unknown_value': [Integral, type(np.nan), None], 'max_categories': [Interval(Integral, 1, None, closed='left'), None], 'min_frequency': [Interval(Integral, 1, None, closed='left'), Interval(RealNotInt, 0, 1, closed='neither'), None]}

    def __init__(self, *, categories='auto', dtype=np.float64, handle_unknown='error', unknown_value=None, encoded_missing_value=np.nan, min_frequency=None, max_categories=None):
        if False:
            for i in range(10):
                print('nop')
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoded_missing_value = encoded_missing_value
        self.min_frequency = min_frequency
        self.max_categories = max_categories

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        if False:
            return 10
        '\n        Fit the OrdinalEncoder to X.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            The data to determine the categories of each feature.\n\n        y : None\n            Ignored. This parameter exists only for compatibility with\n            :class:`~sklearn.pipeline.Pipeline`.\n\n        Returns\n        -------\n        self : object\n            Fitted encoder.\n        '
        if self.handle_unknown == 'use_encoded_value':
            if is_scalar_nan(self.unknown_value):
                if np.dtype(self.dtype).kind != 'f':
                    raise ValueError(f'When unknown_value is np.nan, the dtype parameter should be a float dtype. Got {self.dtype}.')
            elif not isinstance(self.unknown_value, numbers.Integral):
                raise TypeError(f"unknown_value should be an integer or np.nan when handle_unknown is 'use_encoded_value', got {self.unknown_value}.")
        elif self.unknown_value is not None:
            raise TypeError(f"unknown_value should only be set when handle_unknown is 'use_encoded_value', got {self.unknown_value}.")
        fit_results = self._fit(X, handle_unknown=self.handle_unknown, force_all_finite='allow-nan', return_and_ignore_missing_for_infrequent=True)
        self._missing_indices = fit_results['missing_indices']
        cardinalities = [len(categories) for categories in self.categories_]
        if self._infrequent_enabled:
            for (feature_idx, infrequent) in enumerate(self.infrequent_categories_):
                if infrequent is not None:
                    cardinalities[feature_idx] -= len(infrequent)
        for (cat_idx, categories_for_idx) in enumerate(self.categories_):
            for cat in categories_for_idx:
                if is_scalar_nan(cat):
                    cardinalities[cat_idx] -= 1
                    continue
        if self.handle_unknown == 'use_encoded_value':
            for cardinality in cardinalities:
                if 0 <= self.unknown_value < cardinality:
                    raise ValueError(f'The used value for unknown_value {self.unknown_value} is one of the values already used for encoding the seen categories.')
        if self._missing_indices:
            if np.dtype(self.dtype).kind != 'f' and is_scalar_nan(self.encoded_missing_value):
                raise ValueError(f'There are missing values in features {list(self._missing_indices)}. For OrdinalEncoder to encode missing values with dtype: {self.dtype}, set encoded_missing_value to a non-nan value, or set dtype to a float')
            if not is_scalar_nan(self.encoded_missing_value):
                invalid_features = [cat_idx for (cat_idx, cardinality) in enumerate(cardinalities) if cat_idx in self._missing_indices and 0 <= self.encoded_missing_value < cardinality]
                if invalid_features:
                    if hasattr(self, 'feature_names_in_'):
                        invalid_features = self.feature_names_in_[invalid_features]
                    raise ValueError(f'encoded_missing_value ({self.encoded_missing_value}) is already used to encode a known category in features: {invalid_features}')
        return self

    def transform(self, X):
        if False:
            for i in range(10):
                print('nop')
        '\n        Transform X to ordinal codes.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            The data to encode.\n\n        Returns\n        -------\n        X_out : ndarray of shape (n_samples, n_features)\n            Transformed input.\n        '
        (X_int, X_mask) = self._transform(X, handle_unknown=self.handle_unknown, force_all_finite='allow-nan', ignore_category_indices=self._missing_indices)
        X_trans = X_int.astype(self.dtype, copy=False)
        for (cat_idx, missing_idx) in self._missing_indices.items():
            X_missing_mask = X_int[:, cat_idx] == missing_idx
            X_trans[X_missing_mask, cat_idx] = self.encoded_missing_value
        if self.handle_unknown == 'use_encoded_value':
            X_trans[~X_mask] = self.unknown_value
        return X_trans

    def inverse_transform(self, X):
        if False:
            print('Hello World!')
        '\n        Convert the data back to the original representation.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_encoded_features)\n            The transformed data.\n\n        Returns\n        -------\n        X_tr : ndarray of shape (n_samples, n_features)\n            Inverse transformed array.\n        '
        check_is_fitted(self)
        X = check_array(X, force_all_finite='allow-nan')
        (n_samples, _) = X.shape
        n_features = len(self.categories_)
        msg = 'Shape of the passed X data is not correct. Expected {0} columns, got {1}.'
        if X.shape[1] != n_features:
            raise ValueError(msg.format(n_features, X.shape[1]))
        dt = np.result_type(*[cat.dtype for cat in self.categories_])
        X_tr = np.empty((n_samples, n_features), dtype=dt)
        found_unknown = {}
        infrequent_masks = {}
        infrequent_indices = getattr(self, '_infrequent_indices', None)
        for i in range(n_features):
            labels = X[:, i]
            if i in self._missing_indices:
                X_i_mask = _get_mask(labels, self.encoded_missing_value)
                labels[X_i_mask] = self._missing_indices[i]
            rows_to_update = slice(None)
            categories = self.categories_[i]
            if infrequent_indices is not None and infrequent_indices[i] is not None:
                infrequent_encoding_value = len(categories) - len(infrequent_indices[i])
                infrequent_masks[i] = labels == infrequent_encoding_value
                rows_to_update = ~infrequent_masks[i]
                frequent_categories_mask = np.ones_like(categories, dtype=bool)
                frequent_categories_mask[infrequent_indices[i]] = False
                categories = categories[frequent_categories_mask]
            if self.handle_unknown == 'use_encoded_value':
                unknown_labels = _get_mask(labels, self.unknown_value)
                found_unknown[i] = unknown_labels
                known_labels = ~unknown_labels
                if isinstance(rows_to_update, np.ndarray):
                    rows_to_update &= known_labels
                else:
                    rows_to_update = known_labels
            labels_int = labels[rows_to_update].astype('int64', copy=False)
            X_tr[rows_to_update, i] = categories[labels_int]
        if found_unknown or infrequent_masks:
            X_tr = X_tr.astype(object, copy=False)
        if found_unknown:
            for (idx, mask) in found_unknown.items():
                X_tr[mask, idx] = None
        if infrequent_masks:
            for (idx, mask) in infrequent_masks.items():
                X_tr[mask, idx] = 'infrequent_sklearn'
        return X_tr