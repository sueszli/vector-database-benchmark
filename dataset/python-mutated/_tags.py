import numpy as np
_DEFAULT_TAGS = {'array_api_support': False, 'non_deterministic': False, 'requires_positive_X': False, 'requires_positive_y': False, 'X_types': ['2darray'], 'poor_score': False, 'no_validation': False, 'multioutput': False, 'allow_nan': False, 'stateless': False, 'multilabel': False, '_skip_test': False, '_xfail_checks': False, 'multioutput_only': False, 'binary_only': False, 'requires_fit': True, 'preserves_dtype': [np.float64], 'requires_y': False, 'pairwise': False}

def _safe_tags(estimator, key=None):
    if False:
        i = 10
        return i + 15
    'Safely get estimator tags.\n\n    :class:`~sklearn.BaseEstimator` provides the estimator tags machinery.\n    However, if an estimator does not inherit from this base class, we should\n    fall-back to the default tags.\n\n    For scikit-learn built-in estimators, we should still rely on\n    `self._get_tags()`. `_safe_tags(est)` should be used when we are not sure\n    where `est` comes from: typically `_safe_tags(self.base_estimator)` where\n    `self` is a meta-estimator, or in the common checks.\n\n    Parameters\n    ----------\n    estimator : estimator object\n        The estimator from which to get the tag.\n\n    key : str, default=None\n        Tag name to get. By default (`None`), all tags are returned.\n\n    Returns\n    -------\n    tags : dict or tag value\n        The estimator tags. A single value is returned if `key` is not None.\n    '
    if hasattr(estimator, '_get_tags'):
        tags_provider = '_get_tags()'
        tags = estimator._get_tags()
    elif hasattr(estimator, '_more_tags'):
        tags_provider = '_more_tags()'
        tags = {**_DEFAULT_TAGS, **estimator._more_tags()}
    else:
        tags_provider = '_DEFAULT_TAGS'
        tags = _DEFAULT_TAGS
    if key is not None:
        if key not in tags:
            raise ValueError(f'The key {key} is not defined in {tags_provider} for the class {estimator.__class__.__name__}.')
        return tags[key]
    return tags