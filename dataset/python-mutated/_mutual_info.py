from numbers import Integral
import numpy as np
from scipy.sparse import issparse
from scipy.special import digamma
from ..metrics.cluster import mutual_info_score
from ..neighbors import KDTree, NearestNeighbors
from ..preprocessing import scale
from ..utils import check_random_state
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.multiclass import check_classification_targets
from ..utils.validation import check_array, check_X_y

def _compute_mi_cc(x, y, n_neighbors):
    if False:
        while True:
            i = 10
    'Compute mutual information between two continuous variables.\n\n    Parameters\n    ----------\n    x, y : ndarray, shape (n_samples,)\n        Samples of two continuous random variables, must have an identical\n        shape.\n\n    n_neighbors : int\n        Number of nearest neighbors to search for each point, see [1]_.\n\n    Returns\n    -------\n    mi : float\n        Estimated mutual information. If it turned out to be negative it is\n        replace by 0.\n\n    Notes\n    -----\n    True mutual information can\'t be negative. If its estimate by a numerical\n    method is negative, it means (providing the method is adequate) that the\n    mutual information is close to 0 and replacing it by 0 is a reasonable\n    strategy.\n\n    References\n    ----------\n    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual\n           information". Phys. Rev. E 69, 2004.\n    '
    n_samples = x.size
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    xy = np.hstack((x, y))
    nn = NearestNeighbors(metric='chebyshev', n_neighbors=n_neighbors)
    nn.fit(xy)
    radius = nn.kneighbors()[0]
    radius = np.nextafter(radius[:, -1], 0)
    kd = KDTree(x, metric='chebyshev')
    nx = kd.query_radius(x, radius, count_only=True, return_distance=False)
    nx = np.array(nx) - 1.0
    kd = KDTree(y, metric='chebyshev')
    ny = kd.query_radius(y, radius, count_only=True, return_distance=False)
    ny = np.array(ny) - 1.0
    mi = digamma(n_samples) + digamma(n_neighbors) - np.mean(digamma(nx + 1)) - np.mean(digamma(ny + 1))
    return max(0, mi)

def _compute_mi_cd(c, d, n_neighbors):
    if False:
        i = 10
        return i + 15
    'Compute mutual information between continuous and discrete variables.\n\n    Parameters\n    ----------\n    c : ndarray, shape (n_samples,)\n        Samples of a continuous random variable.\n\n    d : ndarray, shape (n_samples,)\n        Samples of a discrete random variable.\n\n    n_neighbors : int\n        Number of nearest neighbors to search for each point, see [1]_.\n\n    Returns\n    -------\n    mi : float\n        Estimated mutual information. If it turned out to be negative it is\n        replace by 0.\n\n    Notes\n    -----\n    True mutual information can\'t be negative. If its estimate by a numerical\n    method is negative, it means (providing the method is adequate) that the\n    mutual information is close to 0 and replacing it by 0 is a reasonable\n    strategy.\n\n    References\n    ----------\n    .. [1] B. C. Ross "Mutual Information between Discrete and Continuous\n       Data Sets". PLoS ONE 9(2), 2014.\n    '
    n_samples = c.shape[0]
    c = c.reshape((-1, 1))
    radius = np.empty(n_samples)
    label_counts = np.empty(n_samples)
    k_all = np.empty(n_samples)
    nn = NearestNeighbors()
    for label in np.unique(d):
        mask = d == label
        count = np.sum(mask)
        if count > 1:
            k = min(n_neighbors, count - 1)
            nn.set_params(n_neighbors=k)
            nn.fit(c[mask])
            r = nn.kneighbors()[0]
            radius[mask] = np.nextafter(r[:, -1], 0)
            k_all[mask] = k
        label_counts[mask] = count
    mask = label_counts > 1
    n_samples = np.sum(mask)
    label_counts = label_counts[mask]
    k_all = k_all[mask]
    c = c[mask]
    radius = radius[mask]
    kd = KDTree(c)
    m_all = kd.query_radius(c, radius, count_only=True, return_distance=False)
    m_all = np.array(m_all)
    mi = digamma(n_samples) + np.mean(digamma(k_all)) - np.mean(digamma(label_counts)) - np.mean(digamma(m_all))
    return max(0, mi)

def _compute_mi(x, y, x_discrete, y_discrete, n_neighbors=3):
    if False:
        while True:
            i = 10
    'Compute mutual information between two variables.\n\n    This is a simple wrapper which selects a proper function to call based on\n    whether `x` and `y` are discrete or not.\n    '
    if x_discrete and y_discrete:
        return mutual_info_score(x, y)
    elif x_discrete and (not y_discrete):
        return _compute_mi_cd(y, x, n_neighbors)
    elif not x_discrete and y_discrete:
        return _compute_mi_cd(x, y, n_neighbors)
    else:
        return _compute_mi_cc(x, y, n_neighbors)

def _iterate_columns(X, columns=None):
    if False:
        return 10
    'Iterate over columns of a matrix.\n\n    Parameters\n    ----------\n    X : ndarray or csc_matrix, shape (n_samples, n_features)\n        Matrix over which to iterate.\n\n    columns : iterable or None, default=None\n        Indices of columns to iterate over. If None, iterate over all columns.\n\n    Yields\n    ------\n    x : ndarray, shape (n_samples,)\n        Columns of `X` in dense format.\n    '
    if columns is None:
        columns = range(X.shape[1])
    if issparse(X):
        for i in columns:
            x = np.zeros(X.shape[0])
            (start_ptr, end_ptr) = (X.indptr[i], X.indptr[i + 1])
            x[X.indices[start_ptr:end_ptr]] = X.data[start_ptr:end_ptr]
            yield x
    else:
        for i in columns:
            yield X[:, i]

def _estimate_mi(X, y, discrete_features='auto', discrete_target=False, n_neighbors=3, copy=True, random_state=None):
    if False:
        i = 10
        return i + 15
    'Estimate mutual information between the features and the target.\n\n    Parameters\n    ----------\n    X : array-like or sparse matrix, shape (n_samples, n_features)\n        Feature matrix.\n\n    y : array-like of shape (n_samples,)\n        Target vector.\n\n    discrete_features : {\'auto\', bool, array-like}, default=\'auto\'\n        If bool, then determines whether to consider all features discrete\n        or continuous. If array, then it should be either a boolean mask\n        with shape (n_features,) or array with indices of discrete features.\n        If \'auto\', it is assigned to False for dense `X` and to True for\n        sparse `X`.\n\n    discrete_target : bool, default=False\n        Whether to consider `y` as a discrete variable.\n\n    n_neighbors : int, default=3\n        Number of neighbors to use for MI estimation for continuous variables,\n        see [1]_ and [2]_. Higher values reduce variance of the estimation, but\n        could introduce a bias.\n\n    copy : bool, default=True\n        Whether to make a copy of the given data. If set to False, the initial\n        data will be overwritten.\n\n    random_state : int, RandomState instance or None, default=None\n        Determines random number generation for adding small noise to\n        continuous variables in order to remove repeated values.\n        Pass an int for reproducible results across multiple function calls.\n        See :term:`Glossary <random_state>`.\n\n    Returns\n    -------\n    mi : ndarray, shape (n_features,)\n        Estimated mutual information between each feature and the target.\n        A negative value will be replaced by 0.\n\n    References\n    ----------\n    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual\n           information". Phys. Rev. E 69, 2004.\n    .. [2] B. C. Ross "Mutual Information between Discrete and Continuous\n           Data Sets". PLoS ONE 9(2), 2014.\n    '
    (X, y) = check_X_y(X, y, accept_sparse='csc', y_numeric=not discrete_target)
    (n_samples, n_features) = X.shape
    if isinstance(discrete_features, (str, bool)):
        if isinstance(discrete_features, str):
            if discrete_features == 'auto':
                discrete_features = issparse(X)
            else:
                raise ValueError('Invalid string value for discrete_features.')
        discrete_mask = np.empty(n_features, dtype=bool)
        discrete_mask.fill(discrete_features)
    else:
        discrete_features = check_array(discrete_features, ensure_2d=False)
        if discrete_features.dtype != 'bool':
            discrete_mask = np.zeros(n_features, dtype=bool)
            discrete_mask[discrete_features] = True
        else:
            discrete_mask = discrete_features
    continuous_mask = ~discrete_mask
    if np.any(continuous_mask) and issparse(X):
        raise ValueError("Sparse matrix `X` can't have continuous features.")
    rng = check_random_state(random_state)
    if np.any(continuous_mask):
        X = X.astype(np.float64, copy=copy)
        X[:, continuous_mask] = scale(X[:, continuous_mask], with_mean=False, copy=False)
        means = np.maximum(1, np.mean(np.abs(X[:, continuous_mask]), axis=0))
        X[:, continuous_mask] += 1e-10 * means * rng.standard_normal(size=(n_samples, np.sum(continuous_mask)))
    if not discrete_target:
        y = scale(y, with_mean=False)
        y += 1e-10 * np.maximum(1, np.mean(np.abs(y))) * rng.standard_normal(size=n_samples)
    mi = [_compute_mi(x, y, discrete_feature, discrete_target, n_neighbors) for (x, discrete_feature) in zip(_iterate_columns(X), discrete_mask)]
    return np.array(mi)

@validate_params({'X': ['array-like', 'sparse matrix'], 'y': ['array-like'], 'discrete_features': [StrOptions({'auto'}), 'boolean', 'array-like'], 'n_neighbors': [Interval(Integral, 1, None, closed='left')], 'copy': ['boolean'], 'random_state': ['random_state']}, prefer_skip_nested_validation=True)
def mutual_info_regression(X, y, *, discrete_features='auto', n_neighbors=3, copy=True, random_state=None):
    if False:
        i = 10
        return i + 15
    'Estimate mutual information for a continuous target variable.\n\n    Mutual information (MI) [1]_ between two random variables is a non-negative\n    value, which measures the dependency between the variables. It is equal\n    to zero if and only if two random variables are independent, and higher\n    values mean higher dependency.\n\n    The function relies on nonparametric methods based on entropy estimation\n    from k-nearest neighbors distances as described in [2]_ and [3]_. Both\n    methods are based on the idea originally proposed in [4]_.\n\n    It can be used for univariate features selection, read more in the\n    :ref:`User Guide <univariate_feature_selection>`.\n\n    Parameters\n    ----------\n    X : array-like or sparse matrix, shape (n_samples, n_features)\n        Feature matrix.\n\n    y : array-like of shape (n_samples,)\n        Target vector.\n\n    discrete_features : {\'auto\', bool, array-like}, default=\'auto\'\n        If bool, then determines whether to consider all features discrete\n        or continuous. If array, then it should be either a boolean mask\n        with shape (n_features,) or array with indices of discrete features.\n        If \'auto\', it is assigned to False for dense `X` and to True for\n        sparse `X`.\n\n    n_neighbors : int, default=3\n        Number of neighbors to use for MI estimation for continuous variables,\n        see [2]_ and [3]_. Higher values reduce variance of the estimation, but\n        could introduce a bias.\n\n    copy : bool, default=True\n        Whether to make a copy of the given data. If set to False, the initial\n        data will be overwritten.\n\n    random_state : int, RandomState instance or None, default=None\n        Determines random number generation for adding small noise to\n        continuous variables in order to remove repeated values.\n        Pass an int for reproducible results across multiple function calls.\n        See :term:`Glossary <random_state>`.\n\n    Returns\n    -------\n    mi : ndarray, shape (n_features,)\n        Estimated mutual information between each feature and the target.\n\n    Notes\n    -----\n    1. The term "discrete features" is used instead of naming them\n       "categorical", because it describes the essence more accurately.\n       For example, pixel intensities of an image are discrete features\n       (but hardly categorical) and you will get better results if mark them\n       as such. Also note, that treating a continuous variable as discrete and\n       vice versa will usually give incorrect results, so be attentive about\n       that.\n    2. True mutual information can\'t be negative. If its estimate turns out\n       to be negative, it is replaced by zero.\n\n    References\n    ----------\n    .. [1] `Mutual Information\n           <https://en.wikipedia.org/wiki/Mutual_information>`_\n           on Wikipedia.\n    .. [2] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual\n           information". Phys. Rev. E 69, 2004.\n    .. [3] B. C. Ross "Mutual Information between Discrete and Continuous\n           Data Sets". PLoS ONE 9(2), 2014.\n    .. [4] L. F. Kozachenko, N. N. Leonenko, "Sample Estimate of the Entropy\n           of a Random Vector", Probl. Peredachi Inf., 23:2 (1987), 9-16\n    '
    return _estimate_mi(X, y, discrete_features, False, n_neighbors, copy, random_state)

@validate_params({'X': ['array-like', 'sparse matrix'], 'y': ['array-like'], 'discrete_features': [StrOptions({'auto'}), 'boolean', 'array-like'], 'n_neighbors': [Interval(Integral, 1, None, closed='left')], 'copy': ['boolean'], 'random_state': ['random_state']}, prefer_skip_nested_validation=True)
def mutual_info_classif(X, y, *, discrete_features='auto', n_neighbors=3, copy=True, random_state=None):
    if False:
        for i in range(10):
            print('nop')
    'Estimate mutual information for a discrete target variable.\n\n    Mutual information (MI) [1]_ between two random variables is a non-negative\n    value, which measures the dependency between the variables. It is equal\n    to zero if and only if two random variables are independent, and higher\n    values mean higher dependency.\n\n    The function relies on nonparametric methods based on entropy estimation\n    from k-nearest neighbors distances as described in [2]_ and [3]_. Both\n    methods are based on the idea originally proposed in [4]_.\n\n    It can be used for univariate features selection, read more in the\n    :ref:`User Guide <univariate_feature_selection>`.\n\n    Parameters\n    ----------\n    X : {array-like, sparse matrix} of shape (n_samples, n_features)\n        Feature matrix.\n\n    y : array-like of shape (n_samples,)\n        Target vector.\n\n    discrete_features : \'auto\', bool or array-like, default=\'auto\'\n        If bool, then determines whether to consider all features discrete\n        or continuous. If array, then it should be either a boolean mask\n        with shape (n_features,) or array with indices of discrete features.\n        If \'auto\', it is assigned to False for dense `X` and to True for\n        sparse `X`.\n\n    n_neighbors : int, default=3\n        Number of neighbors to use for MI estimation for continuous variables,\n        see [2]_ and [3]_. Higher values reduce variance of the estimation, but\n        could introduce a bias.\n\n    copy : bool, default=True\n        Whether to make a copy of the given data. If set to False, the initial\n        data will be overwritten.\n\n    random_state : int, RandomState instance or None, default=None\n        Determines random number generation for adding small noise to\n        continuous variables in order to remove repeated values.\n        Pass an int for reproducible results across multiple function calls.\n        See :term:`Glossary <random_state>`.\n\n    Returns\n    -------\n    mi : ndarray, shape (n_features,)\n        Estimated mutual information between each feature and the target.\n\n    Notes\n    -----\n    1. The term "discrete features" is used instead of naming them\n       "categorical", because it describes the essence more accurately.\n       For example, pixel intensities of an image are discrete features\n       (but hardly categorical) and you will get better results if mark them\n       as such. Also note, that treating a continuous variable as discrete and\n       vice versa will usually give incorrect results, so be attentive about\n       that.\n    2. True mutual information can\'t be negative. If its estimate turns out\n       to be negative, it is replaced by zero.\n\n    References\n    ----------\n    .. [1] `Mutual Information\n           <https://en.wikipedia.org/wiki/Mutual_information>`_\n           on Wikipedia.\n    .. [2] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual\n           information". Phys. Rev. E 69, 2004.\n    .. [3] B. C. Ross "Mutual Information between Discrete and Continuous\n           Data Sets". PLoS ONE 9(2), 2014.\n    .. [4] L. F. Kozachenko, N. N. Leonenko, "Sample Estimate of the Entropy\n           of a Random Vector:, Probl. Peredachi Inf., 23:2 (1987), 9-16\n    '
    check_classification_targets(y)
    return _estimate_mi(X, y, discrete_features, True, n_neighbors, copy, random_state)