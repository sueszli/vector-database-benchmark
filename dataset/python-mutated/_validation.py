"""
The :mod:`sklearn.model_selection._validation` module includes classes and
functions to validate the model.
"""
import numbers
import time
import warnings
from collections import Counter
from contextlib import suppress
from functools import partial
from numbers import Real
from traceback import format_exc
import numpy as np
import scipy.sparse as sp
from joblib import logger
from ..base import clone, is_classifier
from ..exceptions import FitFailedWarning, UnsetMetadataPassedError
from ..metrics import check_scoring, get_scorer_names
from ..metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
from ..preprocessing import LabelEncoder
from ..utils import Bunch, _safe_indexing, check_random_state, indexable
from ..utils._param_validation import HasMethods, Integral, Interval, StrOptions, validate_params
from ..utils.metadata_routing import MetadataRouter, MethodMapping, _routing_enabled, process_routing
from ..utils.metaestimators import _safe_split
from ..utils.parallel import Parallel, delayed
from ..utils.validation import _check_method_params, _num_samples
from ._split import check_cv
__all__ = ['cross_validate', 'cross_val_score', 'cross_val_predict', 'permutation_test_score', 'learning_curve', 'validation_curve']

def _check_params_groups_deprecation(fit_params, params, groups):
    if False:
        for i in range(10):
            print('nop')
    'A helper function to check deprecations on `groups` and `fit_params`.\n\n    To be removed when set_config(enable_metadata_routing=False) is not possible.\n    '
    if params is not None and fit_params is not None:
        raise ValueError('`params` and `fit_params` cannot both be provided. Pass parameters via `params`. `fit_params` is deprecated and will be removed in version 1.6.')
    elif fit_params is not None:
        warnings.warn('`fit_params` is deprecated and will be removed in version 1.6. Pass parameters via `params` instead.', FutureWarning)
        params = fit_params
    params = {} if params is None else params
    if groups is not None and _routing_enabled():
        raise ValueError('`groups` can only be passed if metadata routing is not enabled via `sklearn.set_config(enable_metadata_routing=True)`. When routing is enabled, pass `groups` alongside other metadata via the `params` argument instead.')
    return params

@validate_params({'estimator': [HasMethods('fit')], 'X': ['array-like', 'sparse matrix'], 'y': ['array-like', None], 'groups': ['array-like', None], 'scoring': [StrOptions(set(get_scorer_names())), callable, list, tuple, dict, None], 'cv': ['cv_object'], 'n_jobs': [Integral, None], 'verbose': ['verbose'], 'fit_params': [dict, None], 'params': [dict, None], 'pre_dispatch': [Integral, str], 'return_train_score': ['boolean'], 'return_estimator': ['boolean'], 'return_indices': ['boolean'], 'error_score': [StrOptions({'raise'}), Real]}, prefer_skip_nested_validation=False)
def cross_validate(estimator, X, y=None, *, groups=None, scoring=None, cv=None, n_jobs=None, verbose=0, fit_params=None, params=None, pre_dispatch='2*n_jobs', return_train_score=False, return_estimator=False, return_indices=False, error_score=np.nan):
    if False:
        for i in range(10):
            print('nop')
    'Evaluate metric(s) by cross-validation and also record fit/score times.\n\n    Read more in the :ref:`User Guide <multimetric_cross_validation>`.\n\n    Parameters\n    ----------\n    estimator : estimator object implementing \'fit\'\n        The object to use to fit the data.\n\n    X : {array-like, sparse matrix} of shape (n_samples, n_features)\n        The data to fit. Can be for example a list, or an array.\n\n    y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None\n        The target variable to try to predict in the case of\n        supervised learning.\n\n    groups : array-like of shape (n_samples,), default=None\n        Group labels for the samples used while splitting the dataset into\n        train/test set. Only used in conjunction with a "Group" :term:`cv`\n        instance (e.g., :class:`GroupKFold`).\n\n        .. versionchanged:: 1.4\n            ``groups`` can only be passed if metadata routing is not enabled\n            via ``sklearn.set_config(enable_metadata_routing=True)``. When routing\n            is enabled, pass ``groups`` alongside other metadata via the ``params``\n            argument instead. E.g.:\n            ``cross_validate(..., params={\'groups\': groups})``.\n\n    scoring : str, callable, list, tuple, or dict, default=None\n        Strategy to evaluate the performance of the cross-validated model on\n        the test set.\n\n        If `scoring` represents a single score, one can use:\n\n        - a single string (see :ref:`scoring_parameter`);\n        - a callable (see :ref:`scoring`) that returns a single value.\n\n        If `scoring` represents multiple scores, one can use:\n\n        - a list or tuple of unique strings;\n        - a callable returning a dictionary where the keys are the metric\n          names and the values are the metric scores;\n        - a dictionary with metric names as keys and callables a values.\n\n        See :ref:`multimetric_grid_search` for an example.\n\n    cv : int, cross-validation generator or an iterable, default=None\n        Determines the cross-validation splitting strategy.\n        Possible inputs for cv are:\n\n        - None, to use the default 5-fold cross validation,\n        - int, to specify the number of folds in a `(Stratified)KFold`,\n        - :term:`CV splitter`,\n        - An iterable yielding (train, test) splits as arrays of indices.\n\n        For int/None inputs, if the estimator is a classifier and ``y`` is\n        either binary or multiclass, :class:`StratifiedKFold` is used. In all\n        other cases, :class:`KFold` is used. These splitters are instantiated\n        with `shuffle=False` so the splits will be the same across calls.\n\n        Refer :ref:`User Guide <cross_validation>` for the various\n        cross-validation strategies that can be used here.\n\n        .. versionchanged:: 0.22\n            ``cv`` default value if None changed from 3-fold to 5-fold.\n\n    n_jobs : int, default=None\n        Number of jobs to run in parallel. Training the estimator and computing\n        the score are parallelized over the cross-validation splits.\n        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.\n        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`\n        for more details.\n\n    verbose : int, default=0\n        The verbosity level.\n\n    fit_params : dict, default=None\n        Parameters to pass to the fit method of the estimator.\n\n        .. deprecated:: 1.4\n            This parameter is deprecated and will be removed in version 1.6. Use\n            ``params`` instead.\n\n    params : dict, default=None\n        Parameters to pass to the underlying estimator\'s ``fit``, the scorer,\n        and the CV splitter.\n\n        .. versionadded:: 1.4\n\n    pre_dispatch : int or str, default=\'2*n_jobs\'\n        Controls the number of jobs that get dispatched during parallel\n        execution. Reducing this number can be useful to avoid an\n        explosion of memory consumption when more jobs get dispatched\n        than CPUs can process. This parameter can be:\n\n            - An int, giving the exact number of total jobs that are\n              spawned\n\n            - A str, giving an expression as a function of n_jobs,\n              as in \'2*n_jobs\'\n\n    return_train_score : bool, default=False\n        Whether to include train scores.\n        Computing training scores is used to get insights on how different\n        parameter settings impact the overfitting/underfitting trade-off.\n        However computing the scores on the training set can be computationally\n        expensive and is not strictly required to select the parameters that\n        yield the best generalization performance.\n\n        .. versionadded:: 0.19\n\n        .. versionchanged:: 0.21\n            Default value was changed from ``True`` to ``False``\n\n    return_estimator : bool, default=False\n        Whether to return the estimators fitted on each split.\n\n        .. versionadded:: 0.20\n\n    return_indices : bool, default=False\n        Whether to return the train-test indices selected for each split.\n\n        .. versionadded:: 1.3\n\n    error_score : \'raise\' or numeric, default=np.nan\n        Value to assign to the score if an error occurs in estimator fitting.\n        If set to \'raise\', the error is raised.\n        If a numeric value is given, FitFailedWarning is raised.\n\n        .. versionadded:: 0.20\n\n    Returns\n    -------\n    scores : dict of float arrays of shape (n_splits,)\n        Array of scores of the estimator for each run of the cross validation.\n\n        A dict of arrays containing the score/time arrays for each scorer is\n        returned. The possible keys for this ``dict`` are:\n\n            ``test_score``\n                The score array for test scores on each cv split.\n                Suffix ``_score`` in ``test_score`` changes to a specific\n                metric like ``test_r2`` or ``test_auc`` if there are\n                multiple scoring metrics in the scoring parameter.\n            ``train_score``\n                The score array for train scores on each cv split.\n                Suffix ``_score`` in ``train_score`` changes to a specific\n                metric like ``train_r2`` or ``train_auc`` if there are\n                multiple scoring metrics in the scoring parameter.\n                This is available only if ``return_train_score`` parameter\n                is ``True``.\n            ``fit_time``\n                The time for fitting the estimator on the train\n                set for each cv split.\n            ``score_time``\n                The time for scoring the estimator on the test set for each\n                cv split. (Note time for scoring on the train set is not\n                included even if ``return_train_score`` is set to ``True``\n            ``estimator``\n                The estimator objects for each cv split.\n                This is available only if ``return_estimator`` parameter\n                is set to ``True``.\n            ``indices``\n                The train/test positional indices for each cv split. A dictionary\n                is returned where the keys are either `"train"` or `"test"`\n                and the associated values are a list of integer-dtyped NumPy\n                arrays with the indices. Available only if `return_indices=True`.\n\n    See Also\n    --------\n    cross_val_score : Run cross-validation for single metric evaluation.\n\n    cross_val_predict : Get predictions from each split of cross-validation for\n        diagnostic purposes.\n\n    sklearn.metrics.make_scorer : Make a scorer from a performance metric or\n        loss function.\n\n    Examples\n    --------\n    >>> from sklearn import datasets, linear_model\n    >>> from sklearn.model_selection import cross_validate\n    >>> from sklearn.metrics import make_scorer\n    >>> from sklearn.metrics import confusion_matrix\n    >>> from sklearn.svm import LinearSVC\n    >>> diabetes = datasets.load_diabetes()\n    >>> X = diabetes.data[:150]\n    >>> y = diabetes.target[:150]\n    >>> lasso = linear_model.Lasso()\n\n    Single metric evaluation using ``cross_validate``\n\n    >>> cv_results = cross_validate(lasso, X, y, cv=3)\n    >>> sorted(cv_results.keys())\n    [\'fit_time\', \'score_time\', \'test_score\']\n    >>> cv_results[\'test_score\']\n    array([0.3315057 , 0.08022103, 0.03531816])\n\n    Multiple metric evaluation using ``cross_validate``\n    (please refer the ``scoring`` parameter doc for more information)\n\n    >>> scores = cross_validate(lasso, X, y, cv=3,\n    ...                         scoring=(\'r2\', \'neg_mean_squared_error\'),\n    ...                         return_train_score=True)\n    >>> print(scores[\'test_neg_mean_squared_error\'])\n    [-3635.5... -3573.3... -6114.7...]\n    >>> print(scores[\'train_r2\'])\n    [0.28009951 0.3908844  0.22784907]\n    '
    params = _check_params_groups_deprecation(fit_params, params, groups)
    (X, y) = indexable(X, y)
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    if callable(scoring):
        scorers = scoring
    elif scoring is None or isinstance(scoring, str):
        scorers = check_scoring(estimator, scoring)
    else:
        scorers = _check_multimetric_scoring(estimator, scoring)
    if _routing_enabled():
        if isinstance(scorers, dict):
            _scorer = _MultimetricScorer(scorers=scorers, raise_exc=error_score == 'raise')
        else:
            _scorer = scorers
        router = MetadataRouter(owner='cross_validate').add(splitter=cv, method_mapping=MethodMapping().add(caller='fit', callee='split')).add(estimator=estimator, method_mapping=MethodMapping().add(caller='fit', callee='fit')).add(scorer=_scorer, method_mapping=MethodMapping().add(caller='fit', callee='score'))
        try:
            routed_params = process_routing(router, 'fit', **params)
        except UnsetMetadataPassedError as e:
            raise UnsetMetadataPassedError(message=f'{sorted(e.unrequested_params.keys())} are passed to cross validation but are not explicitly requested or unrequested. See the Metadata Routing User guide <https://scikit-learn.org/stable/metadata_routing.html> for more information.', unrequested_params=e.unrequested_params, routed_params=e.routed_params)
    else:
        routed_params = Bunch()
        routed_params.splitter = Bunch(split={'groups': groups})
        routed_params.estimator = Bunch(fit=params)
        routed_params.scorer = Bunch(score={})
    indices = cv.split(X, y, **routed_params.splitter.split)
    if return_indices:
        indices = list(indices)
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    results = parallel((delayed(_fit_and_score)(clone(estimator), X, y, scorer=scorers, train=train, test=test, verbose=verbose, parameters=None, fit_params=routed_params.estimator.fit, score_params=routed_params.scorer.score, return_train_score=return_train_score, return_times=True, return_estimator=return_estimator, error_score=error_score) for (train, test) in indices))
    _warn_or_raise_about_fit_failures(results, error_score)
    if callable(scoring):
        _insert_error_scores(results, error_score)
    results = _aggregate_score_dicts(results)
    ret = {}
    ret['fit_time'] = results['fit_time']
    ret['score_time'] = results['score_time']
    if return_estimator:
        ret['estimator'] = results['estimator']
    if return_indices:
        ret['indices'] = {}
        (ret['indices']['train'], ret['indices']['test']) = zip(*indices)
    test_scores_dict = _normalize_score_results(results['test_scores'])
    if return_train_score:
        train_scores_dict = _normalize_score_results(results['train_scores'])
    for name in test_scores_dict:
        ret['test_%s' % name] = test_scores_dict[name]
        if return_train_score:
            key = 'train_%s' % name
            ret[key] = train_scores_dict[name]
    return ret

def _insert_error_scores(results, error_score):
    if False:
        return 10
    'Insert error in `results` by replacing them inplace with `error_score`.\n\n    This only applies to multimetric scores because `_fit_and_score` will\n    handle the single metric case.\n    '
    successful_score = None
    failed_indices = []
    for (i, result) in enumerate(results):
        if result['fit_error'] is not None:
            failed_indices.append(i)
        elif successful_score is None:
            successful_score = result['test_scores']
    if isinstance(successful_score, dict):
        formatted_error = {name: error_score for name in successful_score}
        for i in failed_indices:
            results[i]['test_scores'] = formatted_error.copy()
            if 'train_scores' in results[i]:
                results[i]['train_scores'] = formatted_error.copy()

def _normalize_score_results(scores, scaler_score_key='score'):
    if False:
        i = 10
        return i + 15
    'Creates a scoring dictionary based on the type of `scores`'
    if isinstance(scores[0], dict):
        return _aggregate_score_dicts(scores)
    return {scaler_score_key: scores}

def _warn_or_raise_about_fit_failures(results, error_score):
    if False:
        for i in range(10):
            print('nop')
    fit_errors = [result['fit_error'] for result in results if result['fit_error'] is not None]
    if fit_errors:
        num_failed_fits = len(fit_errors)
        num_fits = len(results)
        fit_errors_counter = Counter(fit_errors)
        delimiter = '-' * 80 + '\n'
        fit_errors_summary = '\n'.join((f'{delimiter}{n} fits failed with the following error:\n{error}' for (error, n) in fit_errors_counter.items()))
        if num_failed_fits == num_fits:
            all_fits_failed_message = f"\nAll the {num_fits} fits failed.\nIt is very likely that your model is misconfigured.\nYou can try to debug the error by setting error_score='raise'.\n\nBelow are more details about the failures:\n{fit_errors_summary}"
            raise ValueError(all_fits_failed_message)
        else:
            some_fits_failed_message = f"\n{num_failed_fits} fits failed out of a total of {num_fits}.\nThe score on these train-test partitions for these parameters will be set to {error_score}.\nIf these failures are not expected, you can try to debug them by setting error_score='raise'.\n\nBelow are more details about the failures:\n{fit_errors_summary}"
            warnings.warn(some_fits_failed_message, FitFailedWarning)

@validate_params({'estimator': [HasMethods('fit')], 'X': ['array-like', 'sparse matrix'], 'y': ['array-like', None], 'groups': ['array-like', None], 'scoring': [StrOptions(set(get_scorer_names())), callable, None], 'cv': ['cv_object'], 'n_jobs': [Integral, None], 'verbose': ['verbose'], 'fit_params': [dict, None], 'params': [dict, None], 'pre_dispatch': [Integral, str, None], 'error_score': [StrOptions({'raise'}), Real]}, prefer_skip_nested_validation=False)
def cross_val_score(estimator, X, y=None, *, groups=None, scoring=None, cv=None, n_jobs=None, verbose=0, fit_params=None, params=None, pre_dispatch='2*n_jobs', error_score=np.nan):
    if False:
        i = 10
        return i + 15
    'Evaluate a score by cross-validation.\n\n    Read more in the :ref:`User Guide <cross_validation>`.\n\n    Parameters\n    ----------\n    estimator : estimator object implementing \'fit\'\n        The object to use to fit the data.\n\n    X : {array-like, sparse matrix} of shape (n_samples, n_features)\n        The data to fit. Can be for example a list, or an array.\n\n    y : array-like of shape (n_samples,) or (n_samples, n_outputs),             default=None\n        The target variable to try to predict in the case of\n        supervised learning.\n\n    groups : array-like of shape (n_samples,), default=None\n        Group labels for the samples used while splitting the dataset into\n        train/test set. Only used in conjunction with a "Group" :term:`cv`\n        instance (e.g., :class:`GroupKFold`).\n\n        .. versionchanged:: 1.4\n            ``groups`` can only be passed if metadata routing is not enabled\n            via ``sklearn.set_config(enable_metadata_routing=True)``. When routing\n            is enabled, pass ``groups`` alongside other metadata via the ``params``\n            argument instead. E.g.:\n            ``cross_val_score(..., params={\'groups\': groups})``.\n\n    scoring : str or callable, default=None\n        A str (see model evaluation documentation) or\n        a scorer callable object / function with signature\n        ``scorer(estimator, X, y)`` which should return only\n        a single value.\n\n        Similar to :func:`cross_validate`\n        but only a single metric is permitted.\n\n        If `None`, the estimator\'s default scorer (if available) is used.\n\n    cv : int, cross-validation generator or an iterable, default=None\n        Determines the cross-validation splitting strategy.\n        Possible inputs for cv are:\n\n        - `None`, to use the default 5-fold cross validation,\n        - int, to specify the number of folds in a `(Stratified)KFold`,\n        - :term:`CV splitter`,\n        - An iterable that generates (train, test) splits as arrays of indices.\n\n        For `int`/`None` inputs, if the estimator is a classifier and `y` is\n        either binary or multiclass, :class:`StratifiedKFold` is used. In all\n        other cases, :class:`KFold` is used. These splitters are instantiated\n        with `shuffle=False` so the splits will be the same across calls.\n\n        Refer :ref:`User Guide <cross_validation>` for the various\n        cross-validation strategies that can be used here.\n\n        .. versionchanged:: 0.22\n            `cv` default value if `None` changed from 3-fold to 5-fold.\n\n    n_jobs : int, default=None\n        Number of jobs to run in parallel. Training the estimator and computing\n        the score are parallelized over the cross-validation splits.\n        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.\n        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`\n        for more details.\n\n    verbose : int, default=0\n        The verbosity level.\n\n    fit_params : dict, default=None\n        Parameters to pass to the fit method of the estimator.\n\n        .. deprecated:: 1.4\n            This parameter is deprecated and will be removed in version 1.6. Use\n            ``params`` instead.\n\n    params : dict, default=None\n        Parameters to pass to the underlying estimator\'s ``fit``, the scorer,\n        and the CV splitter.\n\n        .. versionadded:: 1.4\n\n    pre_dispatch : int or str, default=\'2*n_jobs\'\n        Controls the number of jobs that get dispatched during parallel\n        execution. Reducing this number can be useful to avoid an\n        explosion of memory consumption when more jobs get dispatched\n        than CPUs can process. This parameter can be:\n\n            - ``None``, in which case all the jobs are immediately\n              created and spawned. Use this for lightweight and\n              fast-running jobs, to avoid delays due to on-demand\n              spawning of the jobs\n\n            - An int, giving the exact number of total jobs that are\n              spawned\n\n            - A str, giving an expression as a function of n_jobs,\n              as in \'2*n_jobs\'\n\n    error_score : \'raise\' or numeric, default=np.nan\n        Value to assign to the score if an error occurs in estimator fitting.\n        If set to \'raise\', the error is raised.\n        If a numeric value is given, FitFailedWarning is raised.\n\n        .. versionadded:: 0.20\n\n    Returns\n    -------\n    scores : ndarray of float of shape=(len(list(cv)),)\n        Array of scores of the estimator for each run of the cross validation.\n\n    See Also\n    --------\n    cross_validate : To run cross-validation on multiple metrics and also to\n        return train scores, fit times and score times.\n\n    cross_val_predict : Get predictions from each split of cross-validation for\n        diagnostic purposes.\n\n    sklearn.metrics.make_scorer : Make a scorer from a performance metric or\n        loss function.\n\n    Examples\n    --------\n    >>> from sklearn import datasets, linear_model\n    >>> from sklearn.model_selection import cross_val_score\n    >>> diabetes = datasets.load_diabetes()\n    >>> X = diabetes.data[:150]\n    >>> y = diabetes.target[:150]\n    >>> lasso = linear_model.Lasso()\n    >>> print(cross_val_score(lasso, X, y, cv=3))\n    [0.3315057  0.08022103 0.03531816]\n    '
    scorer = check_scoring(estimator, scoring=scoring)
    cv_results = cross_validate(estimator=estimator, X=X, y=y, groups=groups, scoring={'score': scorer}, cv=cv, n_jobs=n_jobs, verbose=verbose, fit_params=fit_params, params=params, pre_dispatch=pre_dispatch, error_score=error_score)
    return cv_results['test_score']

def _fit_and_score(estimator, X, y, *, scorer, train, test, verbose, parameters, fit_params, score_params, return_train_score=False, return_parameters=False, return_n_test_samples=False, return_times=False, return_estimator=False, split_progress=None, candidate_progress=None, error_score=np.nan):
    if False:
        while True:
            i = 10
    "Fit estimator and compute scores for a given dataset split.\n\n    Parameters\n    ----------\n    estimator : estimator object implementing 'fit'\n        The object to use to fit the data.\n\n    X : array-like of shape (n_samples, n_features)\n        The data to fit.\n\n    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None\n        The target variable to try to predict in the case of\n        supervised learning.\n\n    scorer : A single callable or dict mapping scorer name to the callable\n        If it is a single callable, the return value for ``train_scores`` and\n        ``test_scores`` is a single float.\n\n        For a dict, it should be one mapping the scorer name to the scorer\n        callable object / function.\n\n        The callable object / fn should have signature\n        ``scorer(estimator, X, y)``.\n\n    train : array-like of shape (n_train_samples,)\n        Indices of training samples.\n\n    test : array-like of shape (n_test_samples,)\n        Indices of test samples.\n\n    verbose : int\n        The verbosity level.\n\n    error_score : 'raise' or numeric, default=np.nan\n        Value to assign to the score if an error occurs in estimator fitting.\n        If set to 'raise', the error is raised.\n        If a numeric value is given, FitFailedWarning is raised.\n\n    parameters : dict or None\n        Parameters to be set on the estimator.\n\n    fit_params : dict or None\n        Parameters that will be passed to ``estimator.fit``.\n\n    score_params : dict or None\n        Parameters that will be passed to the scorer.\n\n    return_train_score : bool, default=False\n        Compute and return score on training set.\n\n    return_parameters : bool, default=False\n        Return parameters that has been used for the estimator.\n\n    split_progress : {list, tuple} of int, default=None\n        A list or tuple of format (<current_split_id>, <total_num_of_splits>).\n\n    candidate_progress : {list, tuple} of int, default=None\n        A list or tuple of format\n        (<current_candidate_id>, <total_number_of_candidates>).\n\n    return_n_test_samples : bool, default=False\n        Whether to return the ``n_test_samples``.\n\n    return_times : bool, default=False\n        Whether to return the fit/score times.\n\n    return_estimator : bool, default=False\n        Whether to return the fitted estimator.\n\n    Returns\n    -------\n    result : dict with the following attributes\n        train_scores : dict of scorer name -> float\n            Score on training set (for all the scorers),\n            returned only if `return_train_score` is `True`.\n        test_scores : dict of scorer name -> float\n            Score on testing set (for all the scorers).\n        n_test_samples : int\n            Number of test samples.\n        fit_time : float\n            Time spent for fitting in seconds.\n        score_time : float\n            Time spent for scoring in seconds.\n        parameters : dict or None\n            The parameters that have been evaluated.\n        estimator : estimator object\n            The fitted estimator.\n        fit_error : str or None\n            Traceback str if the fit failed, None if the fit succeeded.\n    "
    if not isinstance(error_score, numbers.Number) and error_score != 'raise':
        raise ValueError("error_score must be the string 'raise' or a numeric value. (Hint: if using 'raise', please make sure that it has been spelled correctly.)")
    progress_msg = ''
    if verbose > 2:
        if split_progress is not None:
            progress_msg = f' {split_progress[0] + 1}/{split_progress[1]}'
        if candidate_progress and verbose > 9:
            progress_msg += f'; {candidate_progress[0] + 1}/{candidate_progress[1]}'
    if verbose > 1:
        if parameters is None:
            params_msg = ''
        else:
            sorted_keys = sorted(parameters)
            params_msg = ', '.join((f'{k}={parameters[k]}' for k in sorted_keys))
    if verbose > 9:
        start_msg = f'[CV{progress_msg}] START {params_msg}'
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_method_params(X, params=fit_params, indices=train)
    score_params = score_params if score_params is not None else {}
    score_params_train = _check_method_params(X, params=score_params, indices=train)
    score_params_test = _check_method_params(X, params=score_params, indices=test)
    if parameters is not None:
        estimator = estimator.set_params(**clone(parameters, safe=False))
    start_time = time.time()
    (X_train, y_train) = _safe_split(estimator, X, y, train)
    (X_test, y_test) = _safe_split(estimator, X, y, test, train)
    result = {}
    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)
    except Exception:
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                test_scores = {name: error_score for name in scorer}
                if return_train_score:
                    train_scores = test_scores.copy()
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
        result['fit_error'] = format_exc()
    else:
        result['fit_error'] = None
        fit_time = time.time() - start_time
        test_scores = _score(estimator, X_test, y_test, scorer, score_params_test, error_score)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = _score(estimator, X_train, y_train, scorer, score_params_train, error_score)
    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f'[CV{progress_msg}] END '
        result_msg = params_msg + (';' if params_msg else '')
        if verbose > 2:
            if isinstance(test_scores, dict):
                for scorer_name in sorted(test_scores):
                    result_msg += f' {scorer_name}: ('
                    if return_train_score:
                        scorer_scores = train_scores[scorer_name]
                        result_msg += f'train={scorer_scores:.3f}, '
                    result_msg += f'test={test_scores[scorer_name]:.3f})'
            else:
                result_msg += ', score='
                if return_train_score:
                    result_msg += f'(train={train_scores:.3f}, test={test_scores:.3f})'
                else:
                    result_msg += f'{test_scores:.3f}'
        result_msg += f' total time={logger.short_format_time(total_time)}'
        end_msg += '.' * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)
    result['test_scores'] = test_scores
    if return_train_score:
        result['train_scores'] = train_scores
    if return_n_test_samples:
        result['n_test_samples'] = _num_samples(X_test)
    if return_times:
        result['fit_time'] = fit_time
        result['score_time'] = score_time
    if return_parameters:
        result['parameters'] = parameters
    if return_estimator:
        result['estimator'] = estimator
    return result

def _score(estimator, X_test, y_test, scorer, score_params, error_score='raise'):
    if False:
        while True:
            i = 10
    'Compute the score(s) of an estimator on a given test set.\n\n    Will return a dict of floats if `scorer` is a dict, otherwise a single\n    float is returned.\n    '
    if isinstance(scorer, dict):
        scorer = _MultimetricScorer(scorers=scorer, raise_exc=error_score == 'raise')
    score_params = {} if score_params is None else score_params
    try:
        if y_test is None:
            scores = scorer(estimator, X_test, **score_params)
        else:
            scores = scorer(estimator, X_test, y_test, **score_params)
    except Exception:
        if isinstance(scorer, _MultimetricScorer):
            raise
        elif error_score == 'raise':
            raise
        else:
            scores = error_score
            warnings.warn(f'Scoring failed. The score on this train-test partition for these parameters will be set to {error_score}. Details: \n{format_exc()}', UserWarning)
    if isinstance(scorer, _MultimetricScorer):
        exception_messages = [(name, str_e) for (name, str_e) in scores.items() if isinstance(str_e, str)]
        if exception_messages:
            for (name, str_e) in exception_messages:
                scores[name] = error_score
                warnings.warn(f'Scoring failed. The score on this train-test partition for these parameters will be set to {error_score}. Details: \n{str_e}', UserWarning)
    error_msg = 'scoring must return a number, got %s (%s) instead. (scorer=%s)'
    if isinstance(scores, dict):
        for (name, score) in scores.items():
            if hasattr(score, 'item'):
                with suppress(ValueError):
                    score = score.item()
            if not isinstance(score, numbers.Number):
                raise ValueError(error_msg % (score, type(score), name))
            scores[name] = score
    else:
        if hasattr(scores, 'item'):
            with suppress(ValueError):
                scores = scores.item()
        if not isinstance(scores, numbers.Number):
            raise ValueError(error_msg % (scores, type(scores), scorer))
    return scores

@validate_params({'estimator': [HasMethods(['fit', 'predict'])], 'X': ['array-like', 'sparse matrix'], 'y': ['array-like', None], 'groups': ['array-like', None], 'cv': ['cv_object'], 'n_jobs': [Integral, None], 'verbose': ['verbose'], 'fit_params': [dict, None], 'params': [dict, None], 'pre_dispatch': [Integral, str, None], 'method': [StrOptions({'predict', 'predict_proba', 'predict_log_proba', 'decision_function'})]}, prefer_skip_nested_validation=False)
def cross_val_predict(estimator, X, y=None, *, groups=None, cv=None, n_jobs=None, verbose=0, fit_params=None, params=None, pre_dispatch='2*n_jobs', method='predict'):
    if False:
        return 10
    'Generate cross-validated estimates for each input data point.\n\n    The data is split according to the cv parameter. Each sample belongs\n    to exactly one test set, and its prediction is computed with an\n    estimator fitted on the corresponding training set.\n\n    Passing these predictions into an evaluation metric may not be a valid\n    way to measure generalization performance. Results can differ from\n    :func:`cross_validate` and :func:`cross_val_score` unless all tests sets\n    have equal size and the metric decomposes over samples.\n\n    Read more in the :ref:`User Guide <cross_validation>`.\n\n    Parameters\n    ----------\n    estimator : estimator\n        The estimator instance to use to fit the data. It must implement a `fit`\n        method and the method given by the `method` parameter.\n\n    X : {array-like, sparse matrix} of shape (n_samples, n_features)\n        The data to fit. Can be, for example a list, or an array at least 2d.\n\n    y : array-like of shape (n_samples,) or (n_samples, n_outputs),             default=None\n        The target variable to try to predict in the case of\n        supervised learning.\n\n    groups : array-like of shape (n_samples,), default=None\n        Group labels for the samples used while splitting the dataset into\n        train/test set. Only used in conjunction with a "Group" :term:`cv`\n        instance (e.g., :class:`GroupKFold`).\n\n        .. versionchanged:: 1.4\n            ``groups`` can only be passed if metadata routing is not enabled\n            via ``sklearn.set_config(enable_metadata_routing=True)``. When routing\n            is enabled, pass ``groups`` alongside other metadata via the ``params``\n            argument instead. E.g.:\n            ``cross_val_predict(..., params={\'groups\': groups})``.\n\n    cv : int, cross-validation generator or an iterable, default=None\n        Determines the cross-validation splitting strategy.\n        Possible inputs for cv are:\n\n        - None, to use the default 5-fold cross validation,\n        - int, to specify the number of folds in a `(Stratified)KFold`,\n        - :term:`CV splitter`,\n        - An iterable that generates (train, test) splits as arrays of indices.\n\n        For int/None inputs, if the estimator is a classifier and ``y`` is\n        either binary or multiclass, :class:`StratifiedKFold` is used. In all\n        other cases, :class:`KFold` is used. These splitters are instantiated\n        with `shuffle=False` so the splits will be the same across calls.\n\n        Refer :ref:`User Guide <cross_validation>` for the various\n        cross-validation strategies that can be used here.\n\n        .. versionchanged:: 0.22\n            ``cv`` default value if None changed from 3-fold to 5-fold.\n\n    n_jobs : int, default=None\n        Number of jobs to run in parallel. Training the estimator and\n        predicting are parallelized over the cross-validation splits.\n        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.\n        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`\n        for more details.\n\n    verbose : int, default=0\n        The verbosity level.\n\n    fit_params : dict, default=None\n        Parameters to pass to the fit method of the estimator.\n\n        .. deprecated:: 1.4\n            This parameter is deprecated and will be removed in version 1.6. Use\n            ``params`` instead.\n\n    params : dict, default=None\n        Parameters to pass to the underlying estimator\'s ``fit`` and the CV\n        splitter.\n\n        .. versionadded:: 1.4\n\n    pre_dispatch : int or str, default=\'2*n_jobs\'\n        Controls the number of jobs that get dispatched during parallel\n        execution. Reducing this number can be useful to avoid an\n        explosion of memory consumption when more jobs get dispatched\n        than CPUs can process. This parameter can be:\n\n            - None, in which case all the jobs are immediately\n              created and spawned. Use this for lightweight and\n              fast-running jobs, to avoid delays due to on-demand\n              spawning of the jobs\n\n            - An int, giving the exact number of total jobs that are\n              spawned\n\n            - A str, giving an expression as a function of n_jobs,\n              as in \'2*n_jobs\'\n\n    method : {\'predict\', \'predict_proba\', \'predict_log_proba\',               \'decision_function\'}, default=\'predict\'\n        The method to be invoked by `estimator`.\n\n    Returns\n    -------\n    predictions : ndarray\n        This is the result of calling `method`. Shape:\n\n            - When `method` is \'predict\' and in special case where `method` is\n              \'decision_function\' and the target is binary: (n_samples,)\n            - When `method` is one of {\'predict_proba\', \'predict_log_proba\',\n              \'decision_function\'} (unless special case above):\n              (n_samples, n_classes)\n            - If `estimator` is :term:`multioutput`, an extra dimension\n              \'n_outputs\' is added to the end of each shape above.\n\n    See Also\n    --------\n    cross_val_score : Calculate score for each CV split.\n    cross_validate : Calculate one or more scores and timings for each CV\n        split.\n\n    Notes\n    -----\n    In the case that one or more classes are absent in a training portion, a\n    default score needs to be assigned to all instances for that class if\n    ``method`` produces columns per class, as in {\'decision_function\',\n    \'predict_proba\', \'predict_log_proba\'}.  For ``predict_proba`` this value is\n    0.  In order to ensure finite output, we approximate negative infinity by\n    the minimum finite float value for the dtype in other cases.\n\n    Examples\n    --------\n    >>> from sklearn import datasets, linear_model\n    >>> from sklearn.model_selection import cross_val_predict\n    >>> diabetes = datasets.load_diabetes()\n    >>> X = diabetes.data[:150]\n    >>> y = diabetes.target[:150]\n    >>> lasso = linear_model.Lasso()\n    >>> y_pred = cross_val_predict(lasso, X, y, cv=3)\n    '
    params = _check_params_groups_deprecation(fit_params, params, groups)
    (X, y) = indexable(X, y)
    if _routing_enabled():
        router = MetadataRouter(owner='cross_validate').add(splitter=cv, method_mapping=MethodMapping().add(caller='fit', callee='split')).add(estimator=estimator, method_mapping=MethodMapping().add(caller='fit', callee='fit'))
        try:
            routed_params = process_routing(router, 'fit', **params)
        except UnsetMetadataPassedError as e:
            raise UnsetMetadataPassedError(message=f'{sorted(e.unrequested_params.keys())} are passed to cross validation but are not explicitly requested or unrequested. See the Metadata Routing User guide <https://scikit-learn.org/stable/metadata_routing.html> for more information.', unrequested_params=e.unrequested_params, routed_params=e.routed_params)
    else:
        routed_params = Bunch()
        routed_params.splitter = Bunch(split={'groups': groups})
        routed_params.estimator = Bunch(fit=params)
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    splits = list(cv.split(X, y, **routed_params.splitter.split))
    test_indices = np.concatenate([test for (_, test) in splits])
    if not _check_is_permutation(test_indices, _num_samples(X)):
        raise ValueError('cross_val_predict only works for partitions')
    encode = method in ['decision_function', 'predict_proba', 'predict_log_proba'] and y is not None
    if encode:
        y = np.asarray(y)
        if y.ndim == 1:
            le = LabelEncoder()
            y = le.fit_transform(y)
        elif y.ndim == 2:
            y_enc = np.zeros_like(y, dtype=int)
            for i_label in range(y.shape[1]):
                y_enc[:, i_label] = LabelEncoder().fit_transform(y[:, i_label])
            y = y_enc
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    predictions = parallel((delayed(_fit_and_predict)(clone(estimator), X, y, train, test, routed_params.estimator.fit, method) for (train, test) in splits))
    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))
    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)
    elif encode and isinstance(predictions[0], list):
        n_labels = y.shape[1]
        concat_pred = []
        for i_label in range(n_labels):
            label_preds = np.concatenate([p[i_label] for p in predictions])
            concat_pred.append(label_preds)
        predictions = concat_pred
    else:
        predictions = np.concatenate(predictions)
    if isinstance(predictions, list):
        return [p[inv_test_indices] for p in predictions]
    else:
        return predictions[inv_test_indices]

def _fit_and_predict(estimator, X, y, train, test, fit_params, method):
    if False:
        i = 10
        return i + 15
    "Fit estimator and predict values for a given dataset split.\n\n    Read more in the :ref:`User Guide <cross_validation>`.\n\n    Parameters\n    ----------\n    estimator : estimator object implementing 'fit' and 'predict'\n        The object to use to fit the data.\n\n    X : array-like of shape (n_samples, n_features)\n        The data to fit.\n\n        .. versionchanged:: 0.20\n            X is only required to be an object with finite length or shape now\n\n    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None\n        The target variable to try to predict in the case of\n        supervised learning.\n\n    train : array-like of shape (n_train_samples,)\n        Indices of training samples.\n\n    test : array-like of shape (n_test_samples,)\n        Indices of test samples.\n\n    fit_params : dict or None\n        Parameters that will be passed to ``estimator.fit``.\n\n    method : str\n        Invokes the passed method name of the passed estimator.\n\n    Returns\n    -------\n    predictions : sequence\n        Result of calling 'estimator.method'\n    "
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_method_params(X, params=fit_params, indices=train)
    (X_train, y_train) = _safe_split(estimator, X, y, train)
    (X_test, _) = _safe_split(estimator, X, y, test, train)
    if y_train is None:
        estimator.fit(X_train, **fit_params)
    else:
        estimator.fit(X_train, y_train, **fit_params)
    func = getattr(estimator, method)
    predictions = func(X_test)
    encode = method in ['decision_function', 'predict_proba', 'predict_log_proba'] and y is not None
    if encode:
        if isinstance(predictions, list):
            predictions = [_enforce_prediction_order(estimator.classes_[i_label], predictions[i_label], n_classes=len(set(y[:, i_label])), method=method) for i_label in range(len(predictions))]
        else:
            n_classes = len(set(y)) if y.ndim == 1 else y.shape[1]
            predictions = _enforce_prediction_order(estimator.classes_, predictions, n_classes, method)
    return predictions

def _enforce_prediction_order(classes, predictions, n_classes, method):
    if False:
        print('Hello World!')
    'Ensure that prediction arrays have correct column order\n\n    When doing cross-validation, if one or more classes are\n    not present in the subset of data used for training,\n    then the output prediction array might not have the same\n    columns as other folds. Use the list of class names\n    (assumed to be ints) to enforce the correct column order.\n\n    Note that `classes` is the list of classes in this fold\n    (a subset of the classes in the full training set)\n    and `n_classes` is the number of classes in the full training set.\n    '
    if n_classes != len(classes):
        recommendation = 'To fix this, use a cross-validation technique resulting in properly stratified folds'
        warnings.warn('Number of classes in training fold ({}) does not match total number of classes ({}). Results may not be appropriate for your use case. {}'.format(len(classes), n_classes, recommendation), RuntimeWarning)
        if method == 'decision_function':
            if predictions.ndim == 2 and predictions.shape[1] != len(classes):
                raise ValueError('Output shape {} of {} does not match number of classes ({}) in fold. Irregular decision_function outputs are not currently supported by cross_val_predict'.format(predictions.shape, method, len(classes)))
            if len(classes) <= 2:
                raise ValueError('Only {} class/es in training fold, but {} in overall dataset. This is not supported for decision_function with imbalanced folds. {}'.format(len(classes), n_classes, recommendation))
        float_min = np.finfo(predictions.dtype).min
        default_values = {'decision_function': float_min, 'predict_log_proba': float_min, 'predict_proba': 0}
        predictions_for_all_classes = np.full((_num_samples(predictions), n_classes), default_values[method], dtype=predictions.dtype)
        predictions_for_all_classes[:, classes] = predictions
        predictions = predictions_for_all_classes
    return predictions

def _check_is_permutation(indices, n_samples):
    if False:
        print('Hello World!')
    'Check whether indices is a reordering of the array np.arange(n_samples)\n\n    Parameters\n    ----------\n    indices : ndarray\n        int array to test\n    n_samples : int\n        number of expected elements\n\n    Returns\n    -------\n    is_partition : bool\n        True iff sorted(indices) is np.arange(n)\n    '
    if len(indices) != n_samples:
        return False
    hit = np.zeros(n_samples, dtype=bool)
    hit[indices] = True
    if not np.all(hit):
        return False
    return True

@validate_params({'estimator': [HasMethods('fit')], 'X': ['array-like', 'sparse matrix'], 'y': ['array-like', None], 'groups': ['array-like', None], 'cv': ['cv_object'], 'n_permutations': [Interval(Integral, 1, None, closed='left')], 'n_jobs': [Integral, None], 'random_state': ['random_state'], 'verbose': ['verbose'], 'scoring': [StrOptions(set(get_scorer_names())), callable, None], 'fit_params': [dict, None]}, prefer_skip_nested_validation=False)
def permutation_test_score(estimator, X, y, *, groups=None, cv=None, n_permutations=100, n_jobs=None, random_state=0, verbose=0, scoring=None, fit_params=None):
    if False:
        i = 10
        return i + 15
    "Evaluate the significance of a cross-validated score with permutations.\n\n    Permutes targets to generate 'randomized data' and compute the empirical\n    p-value against the null hypothesis that features and targets are\n    independent.\n\n    The p-value represents the fraction of randomized data sets where the\n    estimator performed as well or better than in the original data. A small\n    p-value suggests that there is a real dependency between features and\n    targets which has been used by the estimator to give good predictions.\n    A large p-value may be due to lack of real dependency between features\n    and targets or the estimator was not able to use the dependency to\n    give good predictions.\n\n    Read more in the :ref:`User Guide <permutation_test_score>`.\n\n    Parameters\n    ----------\n    estimator : estimator object implementing 'fit'\n        The object to use to fit the data.\n\n    X : array-like of shape at least 2D\n        The data to fit.\n\n    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None\n        The target variable to try to predict in the case of\n        supervised learning.\n\n    groups : array-like of shape (n_samples,), default=None\n        Labels to constrain permutation within groups, i.e. ``y`` values\n        are permuted among samples with the same group identifier.\n        When not specified, ``y`` values are permuted among all samples.\n\n        When a grouped cross-validator is used, the group labels are\n        also passed on to the ``split`` method of the cross-validator. The\n        cross-validator uses them for grouping the samples  while splitting\n        the dataset into train/test set.\n\n    cv : int, cross-validation generator or an iterable, default=None\n        Determines the cross-validation splitting strategy.\n        Possible inputs for cv are:\n\n        - `None`, to use the default 5-fold cross validation,\n        - int, to specify the number of folds in a `(Stratified)KFold`,\n        - :term:`CV splitter`,\n        - An iterable yielding (train, test) splits as arrays of indices.\n\n        For `int`/`None` inputs, if the estimator is a classifier and `y` is\n        either binary or multiclass, :class:`StratifiedKFold` is used. In all\n        other cases, :class:`KFold` is used. These splitters are instantiated\n        with `shuffle=False` so the splits will be the same across calls.\n\n        Refer :ref:`User Guide <cross_validation>` for the various\n        cross-validation strategies that can be used here.\n\n        .. versionchanged:: 0.22\n            `cv` default value if `None` changed from 3-fold to 5-fold.\n\n    n_permutations : int, default=100\n        Number of times to permute ``y``.\n\n    n_jobs : int, default=None\n        Number of jobs to run in parallel. Training the estimator and computing\n        the cross-validated score are parallelized over the permutations.\n        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.\n        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`\n        for more details.\n\n    random_state : int, RandomState instance or None, default=0\n        Pass an int for reproducible output for permutation of\n        ``y`` values among samples. See :term:`Glossary <random_state>`.\n\n    verbose : int, default=0\n        The verbosity level.\n\n    scoring : str or callable, default=None\n        A single str (see :ref:`scoring_parameter`) or a callable\n        (see :ref:`scoring`) to evaluate the predictions on the test set.\n\n        If `None` the estimator's score method is used.\n\n    fit_params : dict, default=None\n        Parameters to pass to the fit method of the estimator.\n\n        .. versionadded:: 0.24\n\n    Returns\n    -------\n    score : float\n        The true score without permuting targets.\n\n    permutation_scores : array of shape (n_permutations,)\n        The scores obtained for each permutations.\n\n    pvalue : float\n        The p-value, which approximates the probability that the score would\n        be obtained by chance. This is calculated as:\n\n        `(C + 1) / (n_permutations + 1)`\n\n        Where C is the number of permutations whose score >= the true score.\n\n        The best possible p-value is 1/(n_permutations + 1), the worst is 1.0.\n\n    Notes\n    -----\n    This function implements Test 1 in:\n\n        Ojala and Garriga. `Permutation Tests for Studying Classifier\n        Performance\n        <http://www.jmlr.org/papers/volume11/ojala10a/ojala10a.pdf>`_. The\n        Journal of Machine Learning Research (2010) vol. 11\n    "
    (X, y, groups) = indexable(X, y, groups)
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorer = check_scoring(estimator, scoring=scoring)
    random_state = check_random_state(random_state)
    score = _permutation_test_score(clone(estimator), X, y, groups, cv, scorer, fit_params=fit_params)
    permutation_scores = Parallel(n_jobs=n_jobs, verbose=verbose)((delayed(_permutation_test_score)(clone(estimator), X, _shuffle(y, groups, random_state), groups, cv, scorer, fit_params=fit_params) for _ in range(n_permutations)))
    permutation_scores = np.array(permutation_scores)
    pvalue = (np.sum(permutation_scores >= score) + 1.0) / (n_permutations + 1)
    return (score, permutation_scores, pvalue)

def _permutation_test_score(estimator, X, y, groups, cv, scorer, fit_params):
    if False:
        return 10
    'Auxiliary function for permutation_test_score'
    fit_params = fit_params if fit_params is not None else {}
    avg_score = []
    for (train, test) in cv.split(X, y, groups):
        (X_train, y_train) = _safe_split(estimator, X, y, train)
        (X_test, y_test) = _safe_split(estimator, X, y, test, train)
        fit_params = _check_method_params(X, params=fit_params, indices=train)
        estimator.fit(X_train, y_train, **fit_params)
        avg_score.append(scorer(estimator, X_test, y_test))
    return np.mean(avg_score)

def _shuffle(y, groups, random_state):
    if False:
        i = 10
        return i + 15
    'Return a shuffled copy of y eventually shuffle among same groups.'
    if groups is None:
        indices = random_state.permutation(len(y))
    else:
        indices = np.arange(len(groups))
        for group in np.unique(groups):
            this_mask = groups == group
            indices[this_mask] = random_state.permutation(indices[this_mask])
    return _safe_indexing(y, indices)

@validate_params({'estimator': [HasMethods(['fit'])], 'X': ['array-like', 'sparse matrix'], 'y': ['array-like', None], 'groups': ['array-like', None], 'train_sizes': ['array-like'], 'cv': ['cv_object'], 'scoring': [StrOptions(set(get_scorer_names())), callable, None], 'exploit_incremental_learning': ['boolean'], 'n_jobs': [Integral, None], 'pre_dispatch': [Integral, str], 'verbose': ['verbose'], 'shuffle': ['boolean'], 'random_state': ['random_state'], 'error_score': [StrOptions({'raise'}), Real], 'return_times': ['boolean'], 'fit_params': [dict, None]}, prefer_skip_nested_validation=False)
def learning_curve(estimator, X, y, *, groups=None, train_sizes=np.linspace(0.1, 1.0, 5), cv=None, scoring=None, exploit_incremental_learning=False, n_jobs=None, pre_dispatch='all', verbose=0, shuffle=False, random_state=None, error_score=np.nan, return_times=False, fit_params=None):
    if False:
        print('Hello World!')
    'Learning curve.\n\n    Determines cross-validated training and test scores for different training\n    set sizes.\n\n    A cross-validation generator splits the whole dataset k times in training\n    and test data. Subsets of the training set with varying sizes will be used\n    to train the estimator and a score for each training subset size and the\n    test set will be computed. Afterwards, the scores will be averaged over\n    all k runs for each training subset size.\n\n    Read more in the :ref:`User Guide <learning_curve>`.\n\n    Parameters\n    ----------\n    estimator : object type that implements the "fit" method\n        An object of that type which is cloned for each validation. It must\n        also implement "predict" unless `scoring` is a callable that doesn\'t\n        rely on "predict" to compute a score.\n\n    X : {array-like, sparse matrix} of shape (n_samples, n_features)\n        Training vector, where `n_samples` is the number of samples and\n        `n_features` is the number of features.\n\n    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None\n        Target relative to X for classification or regression;\n        None for unsupervised learning.\n\n    groups : array-like of shape (n_samples,), default=None\n        Group labels for the samples used while splitting the dataset into\n        train/test set. Only used in conjunction with a "Group" :term:`cv`\n        instance (e.g., :class:`GroupKFold`).\n\n    train_sizes : array-like of shape (n_ticks,),             default=np.linspace(0.1, 1.0, 5)\n        Relative or absolute numbers of training examples that will be used to\n        generate the learning curve. If the dtype is float, it is regarded as a\n        fraction of the maximum size of the training set (that is determined\n        by the selected validation method), i.e. it has to be within (0, 1].\n        Otherwise it is interpreted as absolute sizes of the training sets.\n        Note that for classification the number of samples usually have to\n        be big enough to contain at least one sample from each class.\n\n    cv : int, cross-validation generator or an iterable, default=None\n        Determines the cross-validation splitting strategy.\n        Possible inputs for cv are:\n\n        - None, to use the default 5-fold cross validation,\n        - int, to specify the number of folds in a `(Stratified)KFold`,\n        - :term:`CV splitter`,\n        - An iterable yielding (train, test) splits as arrays of indices.\n\n        For int/None inputs, if the estimator is a classifier and ``y`` is\n        either binary or multiclass, :class:`StratifiedKFold` is used. In all\n        other cases, :class:`KFold` is used. These splitters are instantiated\n        with `shuffle=False` so the splits will be the same across calls.\n\n        Refer :ref:`User Guide <cross_validation>` for the various\n        cross-validation strategies that can be used here.\n\n        .. versionchanged:: 0.22\n            ``cv`` default value if None changed from 3-fold to 5-fold.\n\n    scoring : str or callable, default=None\n        A str (see model evaluation documentation) or\n        a scorer callable object / function with signature\n        ``scorer(estimator, X, y)``.\n\n    exploit_incremental_learning : bool, default=False\n        If the estimator supports incremental learning, this will be\n        used to speed up fitting for different training set sizes.\n\n    n_jobs : int, default=None\n        Number of jobs to run in parallel. Training the estimator and computing\n        the score are parallelized over the different training and test sets.\n        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.\n        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`\n        for more details.\n\n    pre_dispatch : int or str, default=\'all\'\n        Number of predispatched jobs for parallel execution (default is\n        all). The option can reduce the allocated memory. The str can\n        be an expression like \'2*n_jobs\'.\n\n    verbose : int, default=0\n        Controls the verbosity: the higher, the more messages.\n\n    shuffle : bool, default=False\n        Whether to shuffle training data before taking prefixes of it\n        based on``train_sizes``.\n\n    random_state : int, RandomState instance or None, default=None\n        Used when ``shuffle`` is True. Pass an int for reproducible\n        output across multiple function calls.\n        See :term:`Glossary <random_state>`.\n\n    error_score : \'raise\' or numeric, default=np.nan\n        Value to assign to the score if an error occurs in estimator fitting.\n        If set to \'raise\', the error is raised.\n        If a numeric value is given, FitFailedWarning is raised.\n\n        .. versionadded:: 0.20\n\n    return_times : bool, default=False\n        Whether to return the fit and score times.\n\n    fit_params : dict, default=None\n        Parameters to pass to the fit method of the estimator.\n\n        .. versionadded:: 0.24\n\n    Returns\n    -------\n    train_sizes_abs : array of shape (n_unique_ticks,)\n        Numbers of training examples that has been used to generate the\n        learning curve. Note that the number of ticks might be less\n        than n_ticks because duplicate entries will be removed.\n\n    train_scores : array of shape (n_ticks, n_cv_folds)\n        Scores on training sets.\n\n    test_scores : array of shape (n_ticks, n_cv_folds)\n        Scores on test set.\n\n    fit_times : array of shape (n_ticks, n_cv_folds)\n        Times spent for fitting in seconds. Only present if ``return_times``\n        is True.\n\n    score_times : array of shape (n_ticks, n_cv_folds)\n        Times spent for scoring in seconds. Only present if ``return_times``\n        is True.\n\n    Examples\n    --------\n    >>> from sklearn.datasets import make_classification\n    >>> from sklearn.tree import DecisionTreeClassifier\n    >>> from sklearn.model_selection import learning_curve\n    >>> X, y = make_classification(n_samples=100, n_features=10, random_state=42)\n    >>> tree = DecisionTreeClassifier(max_depth=4, random_state=42)\n    >>> train_size_abs, train_scores, test_scores = learning_curve(\n    ...     tree, X, y, train_sizes=[0.3, 0.6, 0.9]\n    ... )\n    >>> for train_size, cv_train_scores, cv_test_scores in zip(\n    ...     train_size_abs, train_scores, test_scores\n    ... ):\n    ...     print(f"{train_size} samples were used to train the model")\n    ...     print(f"The average train accuracy is {cv_train_scores.mean():.2f}")\n    ...     print(f"The average test accuracy is {cv_test_scores.mean():.2f}")\n    24 samples were used to train the model\n    The average train accuracy is 1.00\n    The average test accuracy is 0.85\n    48 samples were used to train the model\n    The average train accuracy is 1.00\n    The average test accuracy is 0.90\n    72 samples were used to train the model\n    The average train accuracy is 1.00\n    The average test accuracy is 0.93\n    '
    if exploit_incremental_learning and (not hasattr(estimator, 'partial_fit')):
        raise ValueError('An estimator must support the partial_fit interface to exploit incremental learning')
    (X, y, groups) = indexable(X, y, groups)
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    cv_iter = list(cv.split(X, y, groups))
    scorer = check_scoring(estimator, scoring=scoring)
    n_max_training_samples = len(cv_iter[0][0])
    train_sizes_abs = _translate_train_sizes(train_sizes, n_max_training_samples)
    n_unique_ticks = train_sizes_abs.shape[0]
    if verbose > 0:
        print('[learning_curve] Training set sizes: ' + str(train_sizes_abs))
    parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch, verbose=verbose)
    if shuffle:
        rng = check_random_state(random_state)
        cv_iter = ((rng.permutation(train), test) for (train, test) in cv_iter)
    if exploit_incremental_learning:
        classes = np.unique(y) if is_classifier(estimator) else None
        out = parallel((delayed(_incremental_fit_estimator)(clone(estimator), X, y, classes, train, test, train_sizes_abs, scorer, return_times, error_score=error_score, fit_params=fit_params) for (train, test) in cv_iter))
        out = np.asarray(out).transpose((2, 1, 0))
    else:
        train_test_proportions = []
        for (train, test) in cv_iter:
            for n_train_samples in train_sizes_abs:
                train_test_proportions.append((train[:n_train_samples], test))
        results = parallel((delayed(_fit_and_score)(clone(estimator), X, y, scorer=scorer, train=train, test=test, verbose=verbose, parameters=None, fit_params=fit_params, score_params=None, return_train_score=True, error_score=error_score, return_times=return_times) for (train, test) in train_test_proportions))
        results = _aggregate_score_dicts(results)
        train_scores = results['train_scores'].reshape(-1, n_unique_ticks).T
        test_scores = results['test_scores'].reshape(-1, n_unique_ticks).T
        out = [train_scores, test_scores]
        if return_times:
            fit_times = results['fit_time'].reshape(-1, n_unique_ticks).T
            score_times = results['score_time'].reshape(-1, n_unique_ticks).T
            out.extend([fit_times, score_times])
    ret = (train_sizes_abs, out[0], out[1])
    if return_times:
        ret = ret + (out[2], out[3])
    return ret

def _translate_train_sizes(train_sizes, n_max_training_samples):
    if False:
        i = 10
        return i + 15
    "Determine absolute sizes of training subsets and validate 'train_sizes'.\n\n    Examples:\n        _translate_train_sizes([0.5, 1.0], 10) -> [5, 10]\n        _translate_train_sizes([5, 10], 10) -> [5, 10]\n\n    Parameters\n    ----------\n    train_sizes : array-like of shape (n_ticks,)\n        Numbers of training examples that will be used to generate the\n        learning curve. If the dtype is float, it is regarded as a\n        fraction of 'n_max_training_samples', i.e. it has to be within (0, 1].\n\n    n_max_training_samples : int\n        Maximum number of training samples (upper bound of 'train_sizes').\n\n    Returns\n    -------\n    train_sizes_abs : array of shape (n_unique_ticks,)\n        Numbers of training examples that will be used to generate the\n        learning curve. Note that the number of ticks might be less\n        than n_ticks because duplicate entries will be removed.\n    "
    train_sizes_abs = np.asarray(train_sizes)
    n_ticks = train_sizes_abs.shape[0]
    n_min_required_samples = np.min(train_sizes_abs)
    n_max_required_samples = np.max(train_sizes_abs)
    if np.issubdtype(train_sizes_abs.dtype, np.floating):
        if n_min_required_samples <= 0.0 or n_max_required_samples > 1.0:
            raise ValueError('train_sizes has been interpreted as fractions of the maximum number of training samples and must be within (0, 1], but is within [%f, %f].' % (n_min_required_samples, n_max_required_samples))
        train_sizes_abs = (train_sizes_abs * n_max_training_samples).astype(dtype=int, copy=False)
        train_sizes_abs = np.clip(train_sizes_abs, 1, n_max_training_samples)
    elif n_min_required_samples <= 0 or n_max_required_samples > n_max_training_samples:
        raise ValueError('train_sizes has been interpreted as absolute numbers of training samples and must be within (0, %d], but is within [%d, %d].' % (n_max_training_samples, n_min_required_samples, n_max_required_samples))
    train_sizes_abs = np.unique(train_sizes_abs)
    if n_ticks > train_sizes_abs.shape[0]:
        warnings.warn("Removed duplicate entries from 'train_sizes'. Number of ticks will be less than the size of 'train_sizes': %d instead of %d." % (train_sizes_abs.shape[0], n_ticks), RuntimeWarning)
    return train_sizes_abs

def _incremental_fit_estimator(estimator, X, y, classes, train, test, train_sizes, scorer, return_times, error_score, fit_params):
    if False:
        for i in range(10):
            print('nop')
    'Train estimator on training subsets incrementally and compute scores.'
    (train_scores, test_scores, fit_times, score_times) = ([], [], [], [])
    partitions = zip(train_sizes, np.split(train, train_sizes)[:-1])
    if fit_params is None:
        fit_params = {}
    if classes is None:
        partial_fit_func = partial(estimator.partial_fit, **fit_params)
    else:
        partial_fit_func = partial(estimator.partial_fit, classes=classes, **fit_params)
    for (n_train_samples, partial_train) in partitions:
        train_subset = train[:n_train_samples]
        (X_train, y_train) = _safe_split(estimator, X, y, train_subset)
        (X_partial_train, y_partial_train) = _safe_split(estimator, X, y, partial_train)
        (X_test, y_test) = _safe_split(estimator, X, y, test, train_subset)
        start_fit = time.time()
        if y_partial_train is None:
            partial_fit_func(X_partial_train)
        else:
            partial_fit_func(X_partial_train, y_partial_train)
        fit_time = time.time() - start_fit
        fit_times.append(fit_time)
        start_score = time.time()
        test_scores.append(_score(estimator, X_test, y_test, scorer, score_params=None, error_score=error_score))
        train_scores.append(_score(estimator, X_train, y_train, scorer, score_params=None, error_score=error_score))
        score_time = time.time() - start_score
        score_times.append(score_time)
    ret = (train_scores, test_scores, fit_times, score_times) if return_times else (train_scores, test_scores)
    return np.array(ret).T

@validate_params({'estimator': [HasMethods(['fit'])], 'X': ['array-like', 'sparse matrix'], 'y': ['array-like', None], 'param_name': [str], 'param_range': ['array-like'], 'groups': ['array-like', None], 'cv': ['cv_object'], 'scoring': [StrOptions(set(get_scorer_names())), callable, None], 'n_jobs': [Integral, None], 'pre_dispatch': [Integral, str], 'verbose': ['verbose'], 'error_score': [StrOptions({'raise'}), Real], 'fit_params': [dict, None]}, prefer_skip_nested_validation=False)
def validation_curve(estimator, X, y, *, param_name, param_range, groups=None, cv=None, scoring=None, n_jobs=None, pre_dispatch='all', verbose=0, error_score=np.nan, fit_params=None):
    if False:
        while True:
            i = 10
    'Validation curve.\n\n    Determine training and test scores for varying parameter values.\n\n    Compute scores for an estimator with different values of a specified\n    parameter. This is similar to grid search with one parameter. However, this\n    will also compute training scores and is merely a utility for plotting the\n    results.\n\n    Read more in the :ref:`User Guide <validation_curve>`.\n\n    Parameters\n    ----------\n    estimator : object type that implements the "fit" method\n        An object of that type which is cloned for each validation. It must\n        also implement "predict" unless `scoring` is a callable that doesn\'t\n        rely on "predict" to compute a score.\n\n    X : {array-like, sparse matrix} of shape (n_samples, n_features)\n        Training vector, where `n_samples` is the number of samples and\n        `n_features` is the number of features.\n\n    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None\n        Target relative to X for classification or regression;\n        None for unsupervised learning.\n\n    param_name : str\n        Name of the parameter that will be varied.\n\n    param_range : array-like of shape (n_values,)\n        The values of the parameter that will be evaluated.\n\n    groups : array-like of shape (n_samples,), default=None\n        Group labels for the samples used while splitting the dataset into\n        train/test set. Only used in conjunction with a "Group" :term:`cv`\n        instance (e.g., :class:`GroupKFold`).\n\n    cv : int, cross-validation generator or an iterable, default=None\n        Determines the cross-validation splitting strategy.\n        Possible inputs for cv are:\n\n        - None, to use the default 5-fold cross validation,\n        - int, to specify the number of folds in a `(Stratified)KFold`,\n        - :term:`CV splitter`,\n        - An iterable yielding (train, test) splits as arrays of indices.\n\n        For int/None inputs, if the estimator is a classifier and ``y`` is\n        either binary or multiclass, :class:`StratifiedKFold` is used. In all\n        other cases, :class:`KFold` is used. These splitters are instantiated\n        with `shuffle=False` so the splits will be the same across calls.\n\n        Refer :ref:`User Guide <cross_validation>` for the various\n        cross-validation strategies that can be used here.\n\n        .. versionchanged:: 0.22\n            ``cv`` default value if None changed from 3-fold to 5-fold.\n\n    scoring : str or callable, default=None\n        A str (see model evaluation documentation) or\n        a scorer callable object / function with signature\n        ``scorer(estimator, X, y)``.\n\n    n_jobs : int, default=None\n        Number of jobs to run in parallel. Training the estimator and computing\n        the score are parallelized over the combinations of each parameter\n        value and each cross-validation split.\n        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.\n        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`\n        for more details.\n\n    pre_dispatch : int or str, default=\'all\'\n        Number of predispatched jobs for parallel execution (default is\n        all). The option can reduce the allocated memory. The str can\n        be an expression like \'2*n_jobs\'.\n\n    verbose : int, default=0\n        Controls the verbosity: the higher, the more messages.\n\n    error_score : \'raise\' or numeric, default=np.nan\n        Value to assign to the score if an error occurs in estimator fitting.\n        If set to \'raise\', the error is raised.\n        If a numeric value is given, FitFailedWarning is raised.\n\n        .. versionadded:: 0.20\n\n    fit_params : dict, default=None\n        Parameters to pass to the fit method of the estimator.\n\n        .. versionadded:: 0.24\n\n    Returns\n    -------\n    train_scores : array of shape (n_ticks, n_cv_folds)\n        Scores on training sets.\n\n    test_scores : array of shape (n_ticks, n_cv_folds)\n        Scores on test set.\n\n    Notes\n    -----\n    See :ref:`sphx_glr_auto_examples_model_selection_plot_validation_curve.py`\n    '
    (X, y, groups) = indexable(X, y, groups)
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorer = check_scoring(estimator, scoring=scoring)
    parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch, verbose=verbose)
    results = parallel((delayed(_fit_and_score)(clone(estimator), X, y, scorer=scorer, train=train, test=test, verbose=verbose, parameters={param_name: v}, fit_params=fit_params, score_params=None, return_train_score=True, error_score=error_score) for (train, test) in cv.split(X, y, groups) for v in param_range))
    n_params = len(param_range)
    results = _aggregate_score_dicts(results)
    train_scores = results['train_scores'].reshape(-1, n_params).T
    test_scores = results['test_scores'].reshape(-1, n_params).T
    return (train_scores, test_scores)

def _aggregate_score_dicts(scores):
    if False:
        i = 10
        return i + 15
    "Aggregate the list of dict to dict of np ndarray\n\n    The aggregated output of _aggregate_score_dicts will be a list of dict\n    of form [{'prec': 0.1, 'acc':1.0}, {'prec': 0.1, 'acc':1.0}, ...]\n    Convert it to a dict of array {'prec': np.array([0.1 ...]), ...}\n\n    Parameters\n    ----------\n\n    scores : list of dict\n        List of dicts of the scores for all scorers. This is a flat list,\n        assumed originally to be of row major order.\n\n    Example\n    -------\n\n    >>> scores = [{'a': 1, 'b':10}, {'a': 2, 'b':2}, {'a': 3, 'b':3},\n    ...           {'a': 10, 'b': 10}]                         # doctest: +SKIP\n    >>> _aggregate_score_dicts(scores)                        # doctest: +SKIP\n    {'a': array([1, 2, 3, 10]),\n     'b': array([10, 2, 3, 10])}\n    "
    return {key: np.asarray([score[key] for score in scores]) if isinstance(scores[0][key], numbers.Number) else [score[key] for score in scores] for key in scores[0]}