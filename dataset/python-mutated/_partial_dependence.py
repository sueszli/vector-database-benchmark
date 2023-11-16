"""Partial dependence plots for regression and classification models."""
from collections.abc import Iterable
import numpy as np
from scipy import sparse
from scipy.stats.mstats import mquantiles
from ..base import is_classifier, is_regressor
from ..ensemble import RandomForestRegressor
from ..ensemble._gb import BaseGradientBoosting
from ..ensemble._hist_gradient_boosting.gradient_boosting import BaseHistGradientBoosting
from ..exceptions import NotFittedError
from ..tree import DecisionTreeRegressor
from ..utils import Bunch, _determine_key_type, _get_column_indices, _safe_assign, _safe_indexing, check_array, check_matplotlib_support
from ..utils._param_validation import HasMethods, Integral, Interval, StrOptions, validate_params
from ..utils.extmath import cartesian
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._pd_utils import _check_feature_names, _get_feature_index
__all__ = ['partial_dependence']

def _grid_from_X(X, percentiles, is_categorical, grid_resolution):
    if False:
        print('Hello World!')
    'Generate a grid of points based on the percentiles of X.\n\n    The grid is a cartesian product between the columns of ``values``. The\n    ith column of ``values`` consists in ``grid_resolution`` equally-spaced\n    points between the percentiles of the jth column of X.\n\n    If ``grid_resolution`` is bigger than the number of unique values in the\n    j-th column of X or if the feature is a categorical feature (by inspecting\n    `is_categorical`) , then those unique values will be used instead.\n\n    Parameters\n    ----------\n    X : array-like of shape (n_samples, n_target_features)\n        The data.\n\n    percentiles : tuple of float\n        The percentiles which are used to construct the extreme values of\n        the grid. Must be in [0, 1].\n\n    is_categorical : list of bool\n        For each feature, tells whether it is categorical or not. If a feature\n        is categorical, then the values used will be the unique ones\n        (i.e. categories) instead of the percentiles.\n\n    grid_resolution : int\n        The number of equally spaced points to be placed on the grid for each\n        feature.\n\n    Returns\n    -------\n    grid : ndarray of shape (n_points, n_target_features)\n        A value for each feature at each point in the grid. ``n_points`` is\n        always ``<= grid_resolution ** X.shape[1]``.\n\n    values : list of 1d ndarrays\n        The values with which the grid has been created. The size of each\n        array ``values[j]`` is either ``grid_resolution``, or the number of\n        unique values in ``X[:, j]``, whichever is smaller.\n    '
    if not isinstance(percentiles, Iterable) or len(percentiles) != 2:
        raise ValueError("'percentiles' must be a sequence of 2 elements.")
    if not all((0 <= x <= 1 for x in percentiles)):
        raise ValueError("'percentiles' values must be in [0, 1].")
    if percentiles[0] >= percentiles[1]:
        raise ValueError('percentiles[0] must be strictly less than percentiles[1].')
    if grid_resolution <= 1:
        raise ValueError("'grid_resolution' must be strictly greater than 1.")
    values = []
    for (feature, is_cat) in enumerate(is_categorical):
        try:
            uniques = np.unique(_safe_indexing(X, feature, axis=1))
        except TypeError as exc:
            raise ValueError(f'The column #{feature} contains mixed data types. Finding unique categories fail due to sorting. It usually means that the column contains `np.nan` values together with `str` categories. Such use case is not yet supported in scikit-learn.') from exc
        if is_cat or uniques.shape[0] < grid_resolution:
            axis = uniques
        else:
            emp_percentiles = mquantiles(_safe_indexing(X, feature, axis=1), prob=percentiles, axis=0)
            if np.allclose(emp_percentiles[0], emp_percentiles[1]):
                raise ValueError('percentiles are too close to each other, unable to build the grid. Please choose percentiles that are further apart.')
            axis = np.linspace(emp_percentiles[0], emp_percentiles[1], num=grid_resolution, endpoint=True)
        values.append(axis)
    return (cartesian(values), values)

def _partial_dependence_recursion(est, grid, features):
    if False:
        i = 10
        return i + 15
    "Calculate partial dependence via the recursion method.\n\n    The recursion method is in particular enabled for tree-based estimators.\n\n    For each `grid` value, a weighted tree traversal is performed: if a split node\n    involves an input feature of interest, the corresponding left or right branch\n    is followed; otherwise both branches are followed, each branch being weighted\n    by the fraction of training samples that entered that branch. Finally, the\n    partial dependence is given by a weighted average of all the visited leaves\n    values.\n\n    This method is more efficient in terms of speed than the `'brute'` method\n    (:func:`~sklearn.inspection._partial_dependence._partial_dependence_brute`).\n    However, here, the partial dependence computation is done explicitly with the\n    `X` used during training of `est`.\n\n    Parameters\n    ----------\n    est : BaseEstimator\n        A fitted estimator object implementing :term:`predict` or\n        :term:`decision_function`. Multioutput-multiclass classifiers are not\n        supported. Note that `'recursion'` is only supported for some tree-based\n        estimators (namely\n        :class:`~sklearn.ensemble.GradientBoostingClassifier`,\n        :class:`~sklearn.ensemble.GradientBoostingRegressor`,\n        :class:`~sklearn.ensemble.HistGradientBoostingClassifier`,\n        :class:`~sklearn.ensemble.HistGradientBoostingRegressor`,\n        :class:`~sklearn.tree.DecisionTreeRegressor`,\n        :class:`~sklearn.ensemble.RandomForestRegressor`,\n        ).\n\n    grid : array-like of shape (n_points, n_target_features)\n        The grid of feature values for which the partial dependence is calculated.\n        Note that `n_points` is the number of points in the grid and `n_target_features`\n        is the number of features you are doing partial dependence at.\n\n    features : array-like of {int, str}\n        The feature (e.g. `[0]`) or pair of interacting features\n        (e.g. `[(0, 1)]`) for which the partial dependency should be computed.\n\n    Returns\n    -------\n    averaged_predictions : array-like of shape (n_targets, n_points)\n        The averaged predictions for the given `grid` of features values.\n        Note that `n_targets` is the number of targets (e.g. 1 for binary\n        classification, `n_tasks` for multi-output regression, and `n_classes` for\n        multiclass classification) and `n_points` is the number of points in the `grid`.\n    "
    averaged_predictions = est._compute_partial_dependence_recursion(grid, features)
    if averaged_predictions.ndim == 1:
        averaged_predictions = averaged_predictions.reshape(1, -1)
    return averaged_predictions

def _partial_dependence_brute(est, grid, features, X, response_method, sample_weight=None):
    if False:
        while True:
            i = 10
    "Calculate partial dependence via the brute force method.\n\n    The brute method explicitly averages the predictions of an estimator over a\n    grid of feature values.\n\n    For each `grid` value, all the samples from `X` have their variables of\n    interest replaced by that specific `grid` value. The predictions are then made\n    and averaged across the samples.\n\n    This method is slower than the `'recursion'`\n    (:func:`~sklearn.inspection._partial_dependence._partial_dependence_recursion`)\n    version for estimators with this second option. However, with the `'brute'`\n    force method, the average will be done with the given `X` and not the `X`\n    used during training, as it is done in the `'recursion'` version. Therefore\n    the average can always accept `sample_weight` (even when the estimator was\n    fitted without).\n\n    Parameters\n    ----------\n    est : BaseEstimator\n        A fitted estimator object implementing :term:`predict`,\n        :term:`predict_proba`, or :term:`decision_function`.\n        Multioutput-multiclass classifiers are not supported.\n\n    grid : array-like of shape (n_points, n_target_features)\n        The grid of feature values for which the partial dependence is calculated.\n        Note that `n_points` is the number of points in the grid and `n_target_features`\n        is the number of features you are doing partial dependence at.\n\n    features : array-like of {int, str}\n        The feature (e.g. `[0]`) or pair of interacting features\n        (e.g. `[(0, 1)]`) for which the partial dependency should be computed.\n\n    X : array-like of shape (n_samples, n_features)\n        `X` is used to generate values for the complement features. That is, for\n        each value in `grid`, the method will average the prediction of each\n        sample from `X` having that grid value for `features`.\n\n    response_method : {'auto', 'predict_proba', 'decision_function'},             default='auto'\n        Specifies whether to use :term:`predict_proba` or\n        :term:`decision_function` as the target response. For regressors\n        this parameter is ignored and the response is always the output of\n        :term:`predict`. By default, :term:`predict_proba` is tried first\n        and we revert to :term:`decision_function` if it doesn't exist.\n\n    sample_weight : array-like of shape (n_samples,), default=None\n        Sample weights are used to calculate weighted means when averaging the\n        model output. If `None`, then samples are equally weighted. Note that\n        `sample_weight` does not change the individual predictions.\n\n    Returns\n    -------\n    averaged_predictions : array-like of shape (n_targets, n_points)\n        The averaged predictions for the given `grid` of features values.\n        Note that `n_targets` is the number of targets (e.g. 1 for binary\n        classification, `n_tasks` for multi-output regression, and `n_classes` for\n        multiclass classification) and `n_points` is the number of points in the `grid`.\n\n    predictions : array-like\n        The predictions for the given `grid` of features values over the samples\n        from `X`. For non-multioutput regression and binary classification the\n        shape is `(n_instances, n_points)` and for multi-output regression and\n        multiclass classification the shape is `(n_targets, n_instances, n_points)`,\n        where `n_targets` is the number of targets (`n_tasks` for multi-output\n        regression, and `n_classes` for multiclass classification), `n_instances`\n        is the number of instances in `X`, and `n_points` is the number of points\n        in the `grid`.\n    "
    predictions = []
    averaged_predictions = []
    if is_regressor(est):
        prediction_method = est.predict
    else:
        predict_proba = getattr(est, 'predict_proba', None)
        decision_function = getattr(est, 'decision_function', None)
        if response_method == 'auto':
            prediction_method = predict_proba or decision_function
        else:
            prediction_method = predict_proba if response_method == 'predict_proba' else decision_function
        if prediction_method is None:
            if response_method == 'auto':
                raise ValueError('The estimator has no predict_proba and no decision_function method.')
            elif response_method == 'predict_proba':
                raise ValueError('The estimator has no predict_proba method.')
            else:
                raise ValueError('The estimator has no decision_function method.')
    X_eval = X.copy()
    for new_values in grid:
        for (i, variable) in enumerate(features):
            _safe_assign(X_eval, new_values[i], column_indexer=variable)
        try:
            pred = prediction_method(X_eval)
            predictions.append(pred)
            averaged_predictions.append(np.average(pred, axis=0, weights=sample_weight))
        except NotFittedError as e:
            raise ValueError("'estimator' parameter must be a fitted estimator") from e
    n_samples = X.shape[0]
    predictions = np.array(predictions).T
    if is_regressor(est) and predictions.ndim == 2:
        predictions = predictions.reshape(n_samples, -1)
    elif is_classifier(est) and predictions.shape[0] == 2:
        predictions = predictions[1]
        predictions = predictions.reshape(n_samples, -1)
    averaged_predictions = np.array(averaged_predictions).T
    if is_regressor(est) and averaged_predictions.ndim == 1:
        averaged_predictions = averaged_predictions.reshape(1, -1)
    elif is_classifier(est) and averaged_predictions.shape[0] == 2:
        averaged_predictions = averaged_predictions[1]
        averaged_predictions = averaged_predictions.reshape(1, -1)
    return (averaged_predictions, predictions)

@validate_params({'estimator': [HasMethods(['fit', 'predict']), HasMethods(['fit', 'predict_proba']), HasMethods(['fit', 'decision_function'])], 'X': ['array-like', 'sparse matrix'], 'features': ['array-like', Integral, str], 'sample_weight': ['array-like', None], 'categorical_features': ['array-like', None], 'feature_names': ['array-like', None], 'response_method': [StrOptions({'auto', 'predict_proba', 'decision_function'})], 'percentiles': [tuple], 'grid_resolution': [Interval(Integral, 1, None, closed='left')], 'method': [StrOptions({'auto', 'recursion', 'brute'})], 'kind': [StrOptions({'average', 'individual', 'both'})]}, prefer_skip_nested_validation=True)
def partial_dependence(estimator, X, features, *, sample_weight=None, categorical_features=None, feature_names=None, response_method='auto', percentiles=(0.05, 0.95), grid_resolution=100, method='auto', kind='average'):
    if False:
        return 10
    "Partial dependence of ``features``.\n\n    Partial dependence of a feature (or a set of features) corresponds to\n    the average response of an estimator for each possible value of the\n    feature.\n\n    Read more in the :ref:`User Guide <partial_dependence>`.\n\n    .. warning::\n\n        For :class:`~sklearn.ensemble.GradientBoostingClassifier` and\n        :class:`~sklearn.ensemble.GradientBoostingRegressor`, the\n        `'recursion'` method (used by default) will not account for the `init`\n        predictor of the boosting process. In practice, this will produce\n        the same values as `'brute'` up to a constant offset in the target\n        response, provided that `init` is a constant estimator (which is the\n        default). However, if `init` is not a constant estimator, the\n        partial dependence values are incorrect for `'recursion'` because the\n        offset will be sample-dependent. It is preferable to use the `'brute'`\n        method. Note that this only applies to\n        :class:`~sklearn.ensemble.GradientBoostingClassifier` and\n        :class:`~sklearn.ensemble.GradientBoostingRegressor`, not to\n        :class:`~sklearn.ensemble.HistGradientBoostingClassifier` and\n        :class:`~sklearn.ensemble.HistGradientBoostingRegressor`.\n\n    Parameters\n    ----------\n    estimator : BaseEstimator\n        A fitted estimator object implementing :term:`predict`,\n        :term:`predict_proba`, or :term:`decision_function`.\n        Multioutput-multiclass classifiers are not supported.\n\n    X : {array-like, sparse matrix or dataframe} of shape (n_samples, n_features)\n        ``X`` is used to generate a grid of values for the target\n        ``features`` (where the partial dependence will be evaluated), and\n        also to generate values for the complement features when the\n        `method` is 'brute'.\n\n    features : array-like of {int, str, bool} or int or str\n        The feature (e.g. `[0]`) or pair of interacting features\n        (e.g. `[(0, 1)]`) for which the partial dependency should be computed.\n\n    sample_weight : array-like of shape (n_samples,), default=None\n        Sample weights are used to calculate weighted means when averaging the\n        model output. If `None`, then samples are equally weighted. If\n        `sample_weight` is not `None`, then `method` will be set to `'brute'`.\n        Note that `sample_weight` is ignored for `kind='individual'`.\n\n        .. versionadded:: 1.3\n\n    categorical_features : array-like of shape (n_features,) or shape             (n_categorical_features,), dtype={bool, int, str}, default=None\n        Indicates the categorical features.\n\n        - `None`: no feature will be considered categorical;\n        - boolean array-like: boolean mask of shape `(n_features,)`\n            indicating which features are categorical. Thus, this array has\n            the same shape has `X.shape[1]`;\n        - integer or string array-like: integer indices or strings\n            indicating categorical features.\n\n        .. versionadded:: 1.2\n\n    feature_names : array-like of shape (n_features,), dtype=str, default=None\n        Name of each feature; `feature_names[i]` holds the name of the feature\n        with index `i`.\n        By default, the name of the feature corresponds to their numerical\n        index for NumPy array and their column name for pandas dataframe.\n\n        .. versionadded:: 1.2\n\n    response_method : {'auto', 'predict_proba', 'decision_function'},             default='auto'\n        Specifies whether to use :term:`predict_proba` or\n        :term:`decision_function` as the target response. For regressors\n        this parameter is ignored and the response is always the output of\n        :term:`predict`. By default, :term:`predict_proba` is tried first\n        and we revert to :term:`decision_function` if it doesn't exist. If\n        ``method`` is 'recursion', the response is always the output of\n        :term:`decision_function`.\n\n    percentiles : tuple of float, default=(0.05, 0.95)\n        The lower and upper percentile used to create the extreme values\n        for the grid. Must be in [0, 1].\n\n    grid_resolution : int, default=100\n        The number of equally spaced points on the grid, for each target\n        feature.\n\n    method : {'auto', 'recursion', 'brute'}, default='auto'\n        The method used to calculate the averaged predictions:\n\n        - `'recursion'` is only supported for some tree-based estimators\n          (namely\n          :class:`~sklearn.ensemble.GradientBoostingClassifier`,\n          :class:`~sklearn.ensemble.GradientBoostingRegressor`,\n          :class:`~sklearn.ensemble.HistGradientBoostingClassifier`,\n          :class:`~sklearn.ensemble.HistGradientBoostingRegressor`,\n          :class:`~sklearn.tree.DecisionTreeRegressor`,\n          :class:`~sklearn.ensemble.RandomForestRegressor`,\n          ) when `kind='average'`.\n          This is more efficient in terms of speed.\n          With this method, the target response of a\n          classifier is always the decision function, not the predicted\n          probabilities. Since the `'recursion'` method implicitly computes\n          the average of the Individual Conditional Expectation (ICE) by\n          design, it is not compatible with ICE and thus `kind` must be\n          `'average'`.\n\n        - `'brute'` is supported for any estimator, but is more\n          computationally intensive.\n\n        - `'auto'`: the `'recursion'` is used for estimators that support it,\n          and `'brute'` is used otherwise. If `sample_weight` is not `None`,\n          then `'brute'` is used regardless of the estimator.\n\n        Please see :ref:`this note <pdp_method_differences>` for\n        differences between the `'brute'` and `'recursion'` method.\n\n    kind : {'average', 'individual', 'both'}, default='average'\n        Whether to return the partial dependence averaged across all the\n        samples in the dataset or one value per sample or both.\n        See Returns below.\n\n        Note that the fast `method='recursion'` option is only available for\n        `kind='average'` and `sample_weights=None`. Computing individual\n        dependencies and doing weighted averages requires using the slower\n        `method='brute'`.\n\n        .. versionadded:: 0.24\n\n    Returns\n    -------\n    predictions : :class:`~sklearn.utils.Bunch`\n        Dictionary-like object, with the following attributes.\n\n        individual : ndarray of shape (n_outputs, n_instances,                 len(values[0]), len(values[1]), ...)\n            The predictions for all the points in the grid for all\n            samples in X. This is also known as Individual\n            Conditional Expectation (ICE).\n            Only available when `kind='individual'` or `kind='both'`.\n\n        average : ndarray of shape (n_outputs, len(values[0]),                 len(values[1]), ...)\n            The predictions for all the points in the grid, averaged\n            over all samples in X (or over the training data if\n            `method` is 'recursion').\n            Only available when `kind='average'` or `kind='both'`.\n\n        values : seq of 1d ndarrays\n            The values with which the grid has been created.\n\n            .. deprecated:: 1.3\n                The key `values` has been deprecated in 1.3 and will be removed\n                in 1.5 in favor of `grid_values`. See `grid_values` for details\n                about the `values` attribute.\n\n        grid_values : seq of 1d ndarrays\n            The values with which the grid has been created. The generated\n            grid is a cartesian product of the arrays in `grid_values` where\n            `len(grid_values) == len(features)`. The size of each array\n            `grid_values[j]` is either `grid_resolution`, or the number of\n            unique values in `X[:, j]`, whichever is smaller.\n\n            .. versionadded:: 1.3\n\n        `n_outputs` corresponds to the number of classes in a multi-class\n        setting, or to the number of tasks for multi-output regression.\n        For classical regression and binary classification `n_outputs==1`.\n        `n_values_feature_j` corresponds to the size `grid_values[j]`.\n\n    See Also\n    --------\n    PartialDependenceDisplay.from_estimator : Plot Partial Dependence.\n    PartialDependenceDisplay : Partial Dependence visualization.\n\n    Examples\n    --------\n    >>> X = [[0, 0, 2], [1, 0, 0]]\n    >>> y = [0, 1]\n    >>> from sklearn.ensemble import GradientBoostingClassifier\n    >>> gb = GradientBoostingClassifier(random_state=0).fit(X, y)\n    >>> partial_dependence(gb, features=[0], X=X, percentiles=(0, 1),\n    ...                    grid_resolution=2) # doctest: +SKIP\n    (array([[-4.52...,  4.52...]]), [array([ 0.,  1.])])\n    "
    check_is_fitted(estimator)
    if not (is_classifier(estimator) or is_regressor(estimator)):
        raise ValueError("'estimator' must be a fitted regressor or classifier.")
    if is_classifier(estimator) and isinstance(estimator.classes_[0], np.ndarray):
        raise ValueError('Multiclass-multioutput estimators are not supported')
    if not (hasattr(X, '__array__') or sparse.issparse(X)):
        X = check_array(X, force_all_finite='allow-nan', dtype=object)
    if is_regressor(estimator) and response_method != 'auto':
        raise ValueError("The response_method parameter is ignored for regressors and must be 'auto'.")
    if kind != 'average':
        if method == 'recursion':
            raise ValueError("The 'recursion' method only applies when 'kind' is set to 'average'")
        method = 'brute'
    if method == 'recursion' and sample_weight is not None:
        raise ValueError("The 'recursion' method can only be applied when sample_weight is None.")
    if method == 'auto':
        if sample_weight is not None:
            method = 'brute'
        elif isinstance(estimator, BaseGradientBoosting) and estimator.init is None:
            method = 'recursion'
        elif isinstance(estimator, (BaseHistGradientBoosting, DecisionTreeRegressor, RandomForestRegressor)):
            method = 'recursion'
        else:
            method = 'brute'
    if method == 'recursion':
        if not isinstance(estimator, (BaseGradientBoosting, BaseHistGradientBoosting, DecisionTreeRegressor, RandomForestRegressor)):
            supported_classes_recursion = ('GradientBoostingClassifier', 'GradientBoostingRegressor', 'HistGradientBoostingClassifier', 'HistGradientBoostingRegressor', 'HistGradientBoostingRegressor', 'DecisionTreeRegressor', 'RandomForestRegressor')
            raise ValueError("Only the following estimators support the 'recursion' method: {}. Try using method='brute'.".format(', '.join(supported_classes_recursion)))
        if response_method == 'auto':
            response_method = 'decision_function'
        if response_method != 'decision_function':
            raise ValueError("With the 'recursion' method, the response_method must be 'decision_function'. Got {}.".format(response_method))
    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X)
    if _determine_key_type(features, accept_slice=False) == 'int':
        if np.any(np.less(features, 0)):
            raise ValueError('all features must be in [0, {}]'.format(X.shape[1] - 1))
    features_indices = np.asarray(_get_column_indices(X, features), dtype=np.int32, order='C').ravel()
    feature_names = _check_feature_names(X, feature_names)
    n_features = X.shape[1]
    if categorical_features is None:
        is_categorical = [False] * len(features_indices)
    else:
        categorical_features = np.array(categorical_features, copy=False)
        if categorical_features.dtype.kind == 'b':
            if categorical_features.size != n_features:
                raise ValueError(f'When `categorical_features` is a boolean array-like, the array should be of shape (n_features,). Got {categorical_features.size} elements while `X` contains {n_features} features.')
            is_categorical = [categorical_features[idx] for idx in features_indices]
        elif categorical_features.dtype.kind in ('i', 'O', 'U'):
            categorical_features_idx = [_get_feature_index(cat, feature_names=feature_names) for cat in categorical_features]
            is_categorical = [idx in categorical_features_idx for idx in features_indices]
        else:
            raise ValueError(f'Expected `categorical_features` to be an array-like of boolean, integer, or string. Got {categorical_features.dtype} instead.')
    (grid, values) = _grid_from_X(_safe_indexing(X, features_indices, axis=1), percentiles, is_categorical, grid_resolution)
    if method == 'brute':
        (averaged_predictions, predictions) = _partial_dependence_brute(estimator, grid, features_indices, X, response_method, sample_weight)
        predictions = predictions.reshape(-1, X.shape[0], *[val.shape[0] for val in values])
    else:
        averaged_predictions = _partial_dependence_recursion(estimator, grid, features_indices)
    averaged_predictions = averaged_predictions.reshape(-1, *[val.shape[0] for val in values])
    pdp_results = Bunch()
    msg = "Key: 'values', is deprecated in 1.3 and will be removed in 1.5. Please use 'grid_values' instead."
    pdp_results._set_deprecated(values, new_key='grid_values', deprecated_key='values', warning_message=msg)
    if kind == 'average':
        pdp_results['average'] = averaged_predictions
    elif kind == 'individual':
        pdp_results['individual'] = predictions
    else:
        pdp_results['average'] = averaged_predictions
        pdp_results['individual'] = predictions
    return pdp_results