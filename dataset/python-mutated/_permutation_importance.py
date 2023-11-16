"""Permutation importance for estimators."""
import numbers
import numpy as np
from ..ensemble._bagging import _generate_indices
from ..metrics import check_scoring, get_scorer_names
from ..metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
from ..model_selection._validation import _aggregate_score_dicts
from ..utils import Bunch, _safe_indexing, check_array, check_random_state
from ..utils._param_validation import HasMethods, Integral, Interval, RealNotInt, StrOptions, validate_params
from ..utils.parallel import Parallel, delayed

def _weights_scorer(scorer, estimator, X, y, sample_weight):
    if False:
        for i in range(10):
            print('nop')
    if sample_weight is not None:
        return scorer(estimator, X, y, sample_weight=sample_weight)
    return scorer(estimator, X, y)

def _calculate_permutation_scores(estimator, X, y, sample_weight, col_idx, random_state, n_repeats, scorer, max_samples):
    if False:
        print('Hello World!')
    'Calculate score when `col_idx` is permuted.'
    random_state = check_random_state(random_state)
    if max_samples < X.shape[0]:
        row_indices = _generate_indices(random_state=random_state, bootstrap=False, n_population=X.shape[0], n_samples=max_samples)
        X_permuted = _safe_indexing(X, row_indices, axis=0)
        y = _safe_indexing(y, row_indices, axis=0)
    else:
        X_permuted = X.copy()
    scores = []
    shuffling_idx = np.arange(X_permuted.shape[0])
    for _ in range(n_repeats):
        random_state.shuffle(shuffling_idx)
        if hasattr(X_permuted, 'iloc'):
            col = X_permuted.iloc[shuffling_idx, col_idx]
            col.index = X_permuted.index
            X_permuted[X_permuted.columns[col_idx]] = col
        else:
            X_permuted[:, col_idx] = X_permuted[shuffling_idx, col_idx]
        scores.append(_weights_scorer(scorer, estimator, X_permuted, y, sample_weight))
    if isinstance(scores[0], dict):
        scores = _aggregate_score_dicts(scores)
    else:
        scores = np.array(scores)
    return scores

def _create_importances_bunch(baseline_score, permuted_score):
    if False:
        return 10
    'Compute the importances as the decrease in score.\n\n    Parameters\n    ----------\n    baseline_score : ndarray of shape (n_features,)\n        The baseline score without permutation.\n    permuted_score : ndarray of shape (n_features, n_repeats)\n        The permuted scores for the `n` repetitions.\n\n    Returns\n    -------\n    importances : :class:`~sklearn.utils.Bunch`\n        Dictionary-like object, with the following attributes.\n        importances_mean : ndarray, shape (n_features, )\n            Mean of feature importance over `n_repeats`.\n        importances_std : ndarray, shape (n_features, )\n            Standard deviation over `n_repeats`.\n        importances : ndarray, shape (n_features, n_repeats)\n            Raw permutation importance scores.\n    '
    importances = baseline_score - permuted_score
    return Bunch(importances_mean=np.mean(importances, axis=1), importances_std=np.std(importances, axis=1), importances=importances)

@validate_params({'estimator': [HasMethods(['fit'])], 'X': ['array-like'], 'y': ['array-like', None], 'scoring': [StrOptions(set(get_scorer_names())), callable, list, tuple, dict, None], 'n_repeats': [Interval(Integral, 1, None, closed='left')], 'n_jobs': [Integral, None], 'random_state': ['random_state'], 'sample_weight': ['array-like', None], 'max_samples': [Interval(Integral, 1, None, closed='left'), Interval(RealNotInt, 0, 1, closed='right')]}, prefer_skip_nested_validation=True)
def permutation_importance(estimator, X, y, *, scoring=None, n_repeats=5, n_jobs=None, random_state=None, sample_weight=None, max_samples=1.0):
    if False:
        print('Hello World!')
    'Permutation importance for feature evaluation [BRE]_.\n\n    The :term:`estimator` is required to be a fitted estimator. `X` can be the\n    data set used to train the estimator or a hold-out set. The permutation\n    importance of a feature is calculated as follows. First, a baseline metric,\n    defined by :term:`scoring`, is evaluated on a (potentially different)\n    dataset defined by the `X`. Next, a feature column from the validation set\n    is permuted and the metric is evaluated again. The permutation importance\n    is defined to be the difference between the baseline metric and metric from\n    permutating the feature column.\n\n    Read more in the :ref:`User Guide <permutation_importance>`.\n\n    Parameters\n    ----------\n    estimator : object\n        An estimator that has already been :term:`fitted` and is compatible\n        with :term:`scorer`.\n\n    X : ndarray or DataFrame, shape (n_samples, n_features)\n        Data on which permutation importance will be computed.\n\n    y : array-like or None, shape (n_samples, ) or (n_samples, n_classes)\n        Targets for supervised or `None` for unsupervised.\n\n    scoring : str, callable, list, tuple, or dict, default=None\n        Scorer to use.\n        If `scoring` represents a single score, one can use:\n\n        - a single string (see :ref:`scoring_parameter`);\n        - a callable (see :ref:`scoring`) that returns a single value.\n\n        If `scoring` represents multiple scores, one can use:\n\n        - a list or tuple of unique strings;\n        - a callable returning a dictionary where the keys are the metric\n          names and the values are the metric scores;\n        - a dictionary with metric names as keys and callables a values.\n\n        Passing multiple scores to `scoring` is more efficient than calling\n        `permutation_importance` for each of the scores as it reuses\n        predictions to avoid redundant computation.\n\n        If None, the estimator\'s default scorer is used.\n\n    n_repeats : int, default=5\n        Number of times to permute a feature.\n\n    n_jobs : int or None, default=None\n        Number of jobs to run in parallel. The computation is done by computing\n        permutation score for each columns and parallelized over the columns.\n        `None` means 1 unless in a :obj:`joblib.parallel_backend` context.\n        `-1` means using all processors. See :term:`Glossary <n_jobs>`\n        for more details.\n\n    random_state : int, RandomState instance, default=None\n        Pseudo-random number generator to control the permutations of each\n        feature.\n        Pass an int to get reproducible results across function calls.\n        See :term:`Glossary <random_state>`.\n\n    sample_weight : array-like of shape (n_samples,), default=None\n        Sample weights used in scoring.\n\n        .. versionadded:: 0.24\n\n    max_samples : int or float, default=1.0\n        The number of samples to draw from X to compute feature importance\n        in each repeat (without replacement).\n\n        - If int, then draw `max_samples` samples.\n        - If float, then draw `max_samples * X.shape[0]` samples.\n        - If `max_samples` is equal to `1.0` or `X.shape[0]`, all samples\n          will be used.\n\n        While using this option may provide less accurate importance estimates,\n        it keeps the method tractable when evaluating feature importance on\n        large datasets. In combination with `n_repeats`, this allows to control\n        the computational speed vs statistical accuracy trade-off of this method.\n\n        .. versionadded:: 1.0\n\n    Returns\n    -------\n    result : :class:`~sklearn.utils.Bunch` or dict of such instances\n        Dictionary-like object, with the following attributes.\n\n        importances_mean : ndarray of shape (n_features, )\n            Mean of feature importance over `n_repeats`.\n        importances_std : ndarray of shape (n_features, )\n            Standard deviation over `n_repeats`.\n        importances : ndarray of shape (n_features, n_repeats)\n            Raw permutation importance scores.\n\n        If there are multiple scoring metrics in the scoring parameter\n        `result` is a dict with scorer names as keys (e.g. \'roc_auc\') and\n        `Bunch` objects like above as values.\n\n    References\n    ----------\n    .. [BRE] :doi:`L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32,\n             2001. <10.1023/A:1010933404324>`\n\n    Examples\n    --------\n    >>> from sklearn.linear_model import LogisticRegression\n    >>> from sklearn.inspection import permutation_importance\n    >>> X = [[1, 9, 9],[1, 9, 9],[1, 9, 9],\n    ...      [0, 9, 9],[0, 9, 9],[0, 9, 9]]\n    >>> y = [1, 1, 1, 0, 0, 0]\n    >>> clf = LogisticRegression().fit(X, y)\n    >>> result = permutation_importance(clf, X, y, n_repeats=10,\n    ...                                 random_state=0)\n    >>> result.importances_mean\n    array([0.4666..., 0.       , 0.       ])\n    >>> result.importances_std\n    array([0.2211..., 0.       , 0.       ])\n    '
    if not hasattr(X, 'iloc'):
        X = check_array(X, force_all_finite='allow-nan', dtype=None)
    random_state = check_random_state(random_state)
    random_seed = random_state.randint(np.iinfo(np.int32).max + 1)
    if not isinstance(max_samples, numbers.Integral):
        max_samples = int(max_samples * X.shape[0])
    elif max_samples > X.shape[0]:
        raise ValueError('max_samples must be <= n_samples')
    if callable(scoring):
        scorer = scoring
    elif scoring is None or isinstance(scoring, str):
        scorer = check_scoring(estimator, scoring=scoring)
    else:
        scorers_dict = _check_multimetric_scoring(estimator, scoring)
        scorer = _MultimetricScorer(scorers=scorers_dict)
    baseline_score = _weights_scorer(scorer, estimator, X, y, sample_weight)
    scores = Parallel(n_jobs=n_jobs)((delayed(_calculate_permutation_scores)(estimator, X, y, sample_weight, col_idx, random_seed, n_repeats, scorer, max_samples) for col_idx in range(X.shape[1])))
    if isinstance(baseline_score, dict):
        return {name: _create_importances_bunch(baseline_score[name], np.array([scores[col_idx][name] for col_idx in range(X.shape[1])])) for name in baseline_score}
    else:
        return _create_importances_bunch(baseline_score, np.array(scores))