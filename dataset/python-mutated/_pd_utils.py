def _check_feature_names(X, feature_names=None):
    if False:
        return 10
    'Check feature names.\n\n    Parameters\n    ----------\n    X : array-like of shape (n_samples, n_features)\n        Input data.\n\n    feature_names : None or array-like of shape (n_names,), dtype=str\n        Feature names to check or `None`.\n\n    Returns\n    -------\n    feature_names : list of str\n        Feature names validated. If `feature_names` is `None`, then a list of\n        feature names is provided, i.e. the column names of a pandas dataframe\n        or a generic list of feature names (e.g. `["x0", "x1", ...]`) for a\n        NumPy array.\n    '
    if feature_names is None:
        if hasattr(X, 'columns') and hasattr(X.columns, 'tolist'):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f'x{i}' for i in range(X.shape[1])]
    elif hasattr(feature_names, 'tolist'):
        feature_names = feature_names.tolist()
    if len(set(feature_names)) != len(feature_names):
        raise ValueError('feature_names should not contain duplicates.')
    return feature_names

def _get_feature_index(fx, feature_names=None):
    if False:
        print('Hello World!')
    'Get feature index.\n\n    Parameters\n    ----------\n    fx : int or str\n        Feature index or name.\n\n    feature_names : list of str, default=None\n        All feature names from which to search the indices.\n\n    Returns\n    -------\n    idx : int\n        Feature index.\n    '
    if isinstance(fx, str):
        if feature_names is None:
            raise ValueError(f'Cannot plot partial dependence for feature {fx!r} since the list of feature names was not provided, neither as column names of a pandas data-frame nor via the feature_names parameter.')
        try:
            return feature_names.index(fx)
        except ValueError as e:
            raise ValueError(f'Feature {fx!r} not in feature_names') from e
    return fx