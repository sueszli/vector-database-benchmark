"""
The :mod:`sklearn.model_selection._search` includes utilities to fine-tune the
parameters of an estimator.
"""
import numbers
import operator
import time
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from functools import partial, reduce
from itertools import product
import numpy as np
from numpy.ma import MaskedArray
from scipy.stats import rankdata
from ..base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone, is_classifier
from ..exceptions import NotFittedError
from ..metrics import check_scoring
from ..metrics._scorer import _check_multimetric_scoring, _MultimetricScorer, get_scorer_names
from ..utils import Bunch, check_random_state
from ..utils._param_validation import HasMethods, Interval, StrOptions
from ..utils._tags import _safe_tags
from ..utils.metadata_routing import MetadataRouter, MethodMapping, _raise_for_params, _routing_enabled, process_routing
from ..utils.metaestimators import available_if
from ..utils.parallel import Parallel, delayed
from ..utils.random import sample_without_replacement
from ..utils.validation import _check_method_params, check_is_fitted, indexable
from ._split import check_cv
from ._validation import _aggregate_score_dicts, _fit_and_score, _insert_error_scores, _normalize_score_results, _warn_or_raise_about_fit_failures
__all__ = ['GridSearchCV', 'ParameterGrid', 'ParameterSampler', 'RandomizedSearchCV']

class ParameterGrid:
    """Grid of parameters with a discrete number of values for each.

    Can be used to iterate over parameter value combinations with the
    Python built-in function iter.
    The order of the generated parameter combinations is deterministic.

    Read more in the :ref:`User Guide <grid_search>`.

    Parameters
    ----------
    param_grid : dict of str to sequence, or sequence of such
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.

        An empty dict signifies default parameters.

        A sequence of dicts signifies a sequence of grids to search, and is
        useful to avoid exploring parameter combinations that make no sense
        or have no effect. See the examples below.

    Examples
    --------
    >>> from sklearn.model_selection import ParameterGrid
    >>> param_grid = {'a': [1, 2], 'b': [True, False]}
    >>> list(ParameterGrid(param_grid)) == (
    ...    [{'a': 1, 'b': True}, {'a': 1, 'b': False},
    ...     {'a': 2, 'b': True}, {'a': 2, 'b': False}])
    True

    >>> grid = [{'kernel': ['linear']}, {'kernel': ['rbf'], 'gamma': [1, 10]}]
    >>> list(ParameterGrid(grid)) == [{'kernel': 'linear'},
    ...                               {'kernel': 'rbf', 'gamma': 1},
    ...                               {'kernel': 'rbf', 'gamma': 10}]
    True
    >>> ParameterGrid(grid)[1] == {'kernel': 'rbf', 'gamma': 1}
    True

    See Also
    --------
    GridSearchCV : Uses :class:`ParameterGrid` to perform a full parallelized
        parameter search.
    """

    def __init__(self, param_grid):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(param_grid, (Mapping, Iterable)):
            raise TypeError(f'Parameter grid should be a dict or a list, got: {param_grid!r} of type {type(param_grid).__name__}')
        if isinstance(param_grid, Mapping):
            param_grid = [param_grid]
        for grid in param_grid:
            if not isinstance(grid, dict):
                raise TypeError(f'Parameter grid is not a dict ({grid!r})')
            for (key, value) in grid.items():
                if isinstance(value, np.ndarray) and value.ndim > 1:
                    raise ValueError(f'Parameter array for {key!r} should be one-dimensional, got: {value!r} with shape {value.shape}')
                if isinstance(value, str) or not isinstance(value, (np.ndarray, Sequence)):
                    raise TypeError(f'Parameter grid for parameter {key!r} needs to be a list or a numpy array, but got {value!r} (of type {type(value).__name__}) instead. Single values need to be wrapped in a list with one element.')
                if len(value) == 0:
                    raise ValueError(f'Parameter grid for parameter {key!r} need to be a non-empty sequence, got: {value!r}')
        self.param_grid = param_grid

    def __iter__(self):
        if False:
            return 10
        'Iterate over the points in the grid.\n\n        Returns\n        -------\n        params : iterator over dict of str to any\n            Yields dictionaries mapping each estimator parameter to one of its\n            allowed values.\n        '
        for p in self.param_grid:
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                (keys, values) = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

    def __len__(self):
        if False:
            return 10
        'Number of points on the grid.'
        product = partial(reduce, operator.mul)
        return sum((product((len(v) for v in p.values())) if p else 1 for p in self.param_grid))

    def __getitem__(self, ind):
        if False:
            print('Hello World!')
        'Get the parameters that would be ``ind``th in iteration\n\n        Parameters\n        ----------\n        ind : int\n            The iteration index\n\n        Returns\n        -------\n        params : dict of str to any\n            Equal to list(self)[ind]\n        '
        for sub_grid in self.param_grid:
            if not sub_grid:
                if ind == 0:
                    return {}
                else:
                    ind -= 1
                    continue
            (keys, values_lists) = zip(*sorted(sub_grid.items())[::-1])
            sizes = [len(v_list) for v_list in values_lists]
            total = np.prod(sizes)
            if ind >= total:
                ind -= total
            else:
                out = {}
                for (key, v_list, n) in zip(keys, values_lists, sizes):
                    (ind, offset) = divmod(ind, n)
                    out[key] = v_list[offset]
                return out
        raise IndexError('ParameterGrid index out of range')

class ParameterSampler:
    """Generator on parameters sampled from given distributions.

    Non-deterministic iterable over random candidate combinations for hyper-
    parameter search. If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Read more in the :ref:`User Guide <grid_search>`.

    Parameters
    ----------
    param_distributions : dict
        Dictionary with parameters names (`str`) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.
        If a list of dicts is given, first a dict is sampled uniformly, and
        then a parameter is sampled using that dict as above.

    n_iter : int
        Number of parameter settings that are produced.

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    params : dict of str to any
        **Yields** dictionaries mapping each estimator parameter to
        as sampled value.

    Examples
    --------
    >>> from sklearn.model_selection import ParameterSampler
    >>> from scipy.stats.distributions import expon
    >>> import numpy as np
    >>> rng = np.random.RandomState(0)
    >>> param_grid = {'a':[1, 2], 'b': expon()}
    >>> param_list = list(ParameterSampler(param_grid, n_iter=4,
    ...                                    random_state=rng))
    >>> rounded_list = [dict((k, round(v, 6)) for (k, v) in d.items())
    ...                 for d in param_list]
    >>> rounded_list == [{'b': 0.89856, 'a': 1},
    ...                  {'b': 0.923223, 'a': 1},
    ...                  {'b': 1.878964, 'a': 2},
    ...                  {'b': 1.038159, 'a': 2}]
    True
    """

    def __init__(self, param_distributions, n_iter, *, random_state=None):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(param_distributions, (Mapping, Iterable)):
            raise TypeError(f'Parameter distribution is not a dict or a list, got: {param_distributions!r} of type {type(param_distributions).__name__}')
        if isinstance(param_distributions, Mapping):
            param_distributions = [param_distributions]
        for dist in param_distributions:
            if not isinstance(dist, dict):
                raise TypeError('Parameter distribution is not a dict ({!r})'.format(dist))
            for key in dist:
                if not isinstance(dist[key], Iterable) and (not hasattr(dist[key], 'rvs')):
                    raise TypeError(f'Parameter grid for parameter {key!r} is not iterable or a distribution (value={dist[key]})')
        self.n_iter = n_iter
        self.random_state = random_state
        self.param_distributions = param_distributions

    def _is_all_lists(self):
        if False:
            while True:
                i = 10
        return all((all((not hasattr(v, 'rvs') for v in dist.values())) for dist in self.param_distributions))

    def __iter__(self):
        if False:
            while True:
                i = 10
        rng = check_random_state(self.random_state)
        if self._is_all_lists():
            param_grid = ParameterGrid(self.param_distributions)
            grid_size = len(param_grid)
            n_iter = self.n_iter
            if grid_size < n_iter:
                warnings.warn('The total space of parameters %d is smaller than n_iter=%d. Running %d iterations. For exhaustive searches, use GridSearchCV.' % (grid_size, self.n_iter, grid_size), UserWarning)
                n_iter = grid_size
            for i in sample_without_replacement(grid_size, n_iter, random_state=rng):
                yield param_grid[i]
        else:
            for _ in range(self.n_iter):
                dist = rng.choice(self.param_distributions)
                items = sorted(dist.items())
                params = dict()
                for (k, v) in items:
                    if hasattr(v, 'rvs'):
                        params[k] = v.rvs(random_state=rng)
                    else:
                        params[k] = v[rng.randint(len(v))]
                yield params

    def __len__(self):
        if False:
            while True:
                i = 10
        'Number of points that will be sampled.'
        if self._is_all_lists():
            grid_size = len(ParameterGrid(self.param_distributions))
            return min(self.n_iter, grid_size)
        else:
            return self.n_iter

def _check_refit(search_cv, attr):
    if False:
        print('Hello World!')
    if not search_cv.refit:
        raise AttributeError(f'This {type(search_cv).__name__} instance was initialized with `refit=False`. {attr} is available only after refitting on the best parameters. You can refit an estimator manually using the `best_params_` attribute')

def _estimator_has(attr):
    if False:
        return 10
    'Check if we can delegate a method to the underlying estimator.\n\n    Calling a prediction method will only be available if `refit=True`. In\n    such case, we check first the fitted best estimator. If it is not\n    fitted, we check the unfitted estimator.\n\n    Checking the unfitted estimator allows to use `hasattr` on the `SearchCV`\n    instance even before calling `fit`.\n    '

    def check(self):
        if False:
            return 10
        _check_refit(self, attr)
        if hasattr(self, 'best_estimator_'):
            getattr(self.best_estimator_, attr)
            return True
        getattr(self.estimator, attr)
        return True
    return check

class BaseSearchCV(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    """Abstract base class for hyper parameter search with cross-validation."""
    _parameter_constraints: dict = {'estimator': [HasMethods(['fit'])], 'scoring': [StrOptions(set(get_scorer_names())), callable, list, tuple, dict, None], 'n_jobs': [numbers.Integral, None], 'refit': ['boolean', str, callable], 'cv': ['cv_object'], 'verbose': ['verbose'], 'pre_dispatch': [numbers.Integral, str], 'error_score': [StrOptions({'raise'}), numbers.Real], 'return_train_score': ['boolean']}

    @abstractmethod
    def __init__(self, estimator, *, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=np.nan, return_train_score=True):
        if False:
            i = 10
            return i + 15
        self.scoring = scoring
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score

    @property
    def _estimator_type(self):
        if False:
            print('Hello World!')
        return self.estimator._estimator_type

    def _more_tags(self):
        if False:
            i = 10
            return i + 15
        return {'pairwise': _safe_tags(self.estimator, 'pairwise'), '_xfail_checks': {'check_supervised_y_2d': 'DataConversionWarning not caught'}}

    def score(self, X, y=None, **params):
        if False:
            for i in range(10):
                print('nop')
        'Return the score on the given data, if the estimator has been refit.\n\n        This uses the score defined by ``scoring`` where provided, and the\n        ``best_estimator_.score`` method otherwise.\n\n        Parameters\n        ----------\n        X : array-like of shape (n_samples, n_features)\n            Input data, where `n_samples` is the number of samples and\n            `n_features` is the number of features.\n\n        y : array-like of shape (n_samples, n_output)             or (n_samples,), default=None\n            Target relative to X for classification or regression;\n            None for unsupervised learning.\n\n        **params : dict\n            Parameters to be passed to the underlying scorer(s).\n\n            ..versionadded:: 1.4\n                Only available if `enable_metadata_routing=True`. See\n                :ref:`Metadata Routing User Guide <metadata_routing>` for more\n                details.\n\n        Returns\n        -------\n        score : float\n            The score defined by ``scoring`` if provided, and the\n            ``best_estimator_.score`` method otherwise.\n        '
        _check_refit(self, 'score')
        check_is_fitted(self)
        _raise_for_params(params, self, 'score')
        if _routing_enabled():
            score_params = process_routing(self, 'score', **params).scorer['score']
        else:
            score_params = dict()
        if self.scorer_ is None:
            raise ValueError("No score function explicitly defined, and the estimator doesn't provide one %s" % self.best_estimator_)
        if isinstance(self.scorer_, dict):
            if self.multimetric_:
                scorer = self.scorer_[self.refit]
            else:
                scorer = self.scorer_
            return scorer(self.best_estimator_, X, y, **score_params)
        score = self.scorer_(self.best_estimator_, X, y, **score_params)
        if self.multimetric_:
            score = score[self.refit]
        return score

    @available_if(_estimator_has('score_samples'))
    def score_samples(self, X):
        if False:
            for i in range(10):
                print('nop')
        'Call score_samples on the estimator with the best found parameters.\n\n        Only available if ``refit=True`` and the underlying estimator supports\n        ``score_samples``.\n\n        .. versionadded:: 0.24\n\n        Parameters\n        ----------\n        X : iterable\n            Data to predict on. Must fulfill input requirements\n            of the underlying estimator.\n\n        Returns\n        -------\n        y_score : ndarray of shape (n_samples,)\n            The ``best_estimator_.score_samples`` method.\n        '
        check_is_fitted(self)
        return self.best_estimator_.score_samples(X)

    @available_if(_estimator_has('predict'))
    def predict(self, X):
        if False:
            while True:
                i = 10
        'Call predict on the estimator with the best found parameters.\n\n        Only available if ``refit=True`` and the underlying estimator supports\n        ``predict``.\n\n        Parameters\n        ----------\n        X : indexable, length n_samples\n            Must fulfill the input assumptions of the\n            underlying estimator.\n\n        Returns\n        -------\n        y_pred : ndarray of shape (n_samples,)\n            The predicted labels or values for `X` based on the estimator with\n            the best found parameters.\n        '
        check_is_fitted(self)
        return self.best_estimator_.predict(X)

    @available_if(_estimator_has('predict_proba'))
    def predict_proba(self, X):
        if False:
            while True:
                i = 10
        'Call predict_proba on the estimator with the best found parameters.\n\n        Only available if ``refit=True`` and the underlying estimator supports\n        ``predict_proba``.\n\n        Parameters\n        ----------\n        X : indexable, length n_samples\n            Must fulfill the input assumptions of the\n            underlying estimator.\n\n        Returns\n        -------\n        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)\n            Predicted class probabilities for `X` based on the estimator with\n            the best found parameters. The order of the classes corresponds\n            to that in the fitted attribute :term:`classes_`.\n        '
        check_is_fitted(self)
        return self.best_estimator_.predict_proba(X)

    @available_if(_estimator_has('predict_log_proba'))
    def predict_log_proba(self, X):
        if False:
            return 10
        'Call predict_log_proba on the estimator with the best found parameters.\n\n        Only available if ``refit=True`` and the underlying estimator supports\n        ``predict_log_proba``.\n\n        Parameters\n        ----------\n        X : indexable, length n_samples\n            Must fulfill the input assumptions of the\n            underlying estimator.\n\n        Returns\n        -------\n        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)\n            Predicted class log-probabilities for `X` based on the estimator\n            with the best found parameters. The order of the classes\n            corresponds to that in the fitted attribute :term:`classes_`.\n        '
        check_is_fitted(self)
        return self.best_estimator_.predict_log_proba(X)

    @available_if(_estimator_has('decision_function'))
    def decision_function(self, X):
        if False:
            while True:
                i = 10
        'Call decision_function on the estimator with the best found parameters.\n\n        Only available if ``refit=True`` and the underlying estimator supports\n        ``decision_function``.\n\n        Parameters\n        ----------\n        X : indexable, length n_samples\n            Must fulfill the input assumptions of the\n            underlying estimator.\n\n        Returns\n        -------\n        y_score : ndarray of shape (n_samples,) or (n_samples, n_classes)                 or (n_samples, n_classes * (n_classes-1) / 2)\n            Result of the decision function for `X` based on the estimator with\n            the best found parameters.\n        '
        check_is_fitted(self)
        return self.best_estimator_.decision_function(X)

    @available_if(_estimator_has('transform'))
    def transform(self, X):
        if False:
            print('Hello World!')
        'Call transform on the estimator with the best found parameters.\n\n        Only available if the underlying estimator supports ``transform`` and\n        ``refit=True``.\n\n        Parameters\n        ----------\n        X : indexable, length n_samples\n            Must fulfill the input assumptions of the\n            underlying estimator.\n\n        Returns\n        -------\n        Xt : {ndarray, sparse matrix} of shape (n_samples, n_features)\n            `X` transformed in the new space based on the estimator with\n            the best found parameters.\n        '
        check_is_fitted(self)
        return self.best_estimator_.transform(X)

    @available_if(_estimator_has('inverse_transform'))
    def inverse_transform(self, Xt):
        if False:
            while True:
                i = 10
        'Call inverse_transform on the estimator with the best found params.\n\n        Only available if the underlying estimator implements\n        ``inverse_transform`` and ``refit=True``.\n\n        Parameters\n        ----------\n        Xt : indexable, length n_samples\n            Must fulfill the input assumptions of the\n            underlying estimator.\n\n        Returns\n        -------\n        X : {ndarray, sparse matrix} of shape (n_samples, n_features)\n            Result of the `inverse_transform` function for `Xt` based on the\n            estimator with the best found parameters.\n        '
        check_is_fitted(self)
        return self.best_estimator_.inverse_transform(Xt)

    @property
    def n_features_in_(self):
        if False:
            for i in range(10):
                print('nop')
        'Number of features seen during :term:`fit`.\n\n        Only available when `refit=True`.\n        '
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError('{} object has no n_features_in_ attribute.'.format(self.__class__.__name__)) from nfe
        return self.best_estimator_.n_features_in_

    @property
    def classes_(self):
        if False:
            i = 10
            return i + 15
        'Class labels.\n\n        Only available when `refit=True` and the estimator is a classifier.\n        '
        _estimator_has('classes_')(self)
        return self.best_estimator_.classes_

    def _run_search(self, evaluate_candidates):
        if False:
            for i in range(10):
                print('nop')
        "Repeatedly calls `evaluate_candidates` to conduct a search.\n\n        This method, implemented in sub-classes, makes it possible to\n        customize the scheduling of evaluations: GridSearchCV and\n        RandomizedSearchCV schedule evaluations for their whole parameter\n        search space at once but other more sequential approaches are also\n        possible: for instance is possible to iteratively schedule evaluations\n        for new regions of the parameter search space based on previously\n        collected evaluation results. This makes it possible to implement\n        Bayesian optimization or more generally sequential model-based\n        optimization by deriving from the BaseSearchCV abstract base class.\n        For example, Successive Halving is implemented by calling\n        `evaluate_candidates` multiples times (once per iteration of the SH\n        process), each time passing a different set of candidates with `X`\n        and `y` of increasing sizes.\n\n        Parameters\n        ----------\n        evaluate_candidates : callable\n            This callback accepts:\n                - a list of candidates, where each candidate is a dict of\n                  parameter settings.\n                - an optional `cv` parameter which can be used to e.g.\n                  evaluate candidates on different dataset splits, or\n                  evaluate candidates on subsampled data (as done in the\n                  SucessiveHaling estimators). By default, the original `cv`\n                  parameter is used, and it is available as a private\n                  `_checked_cv_orig` attribute.\n                - an optional `more_results` dict. Each key will be added to\n                  the `cv_results_` attribute. Values should be lists of\n                  length `n_candidates`\n\n            It returns a dict of all results so far, formatted like\n            ``cv_results_``.\n\n            Important note (relevant whether the default cv is used or not):\n            in randomized splitters, and unless the random_state parameter of\n            cv was set to an int, calling cv.split() multiple times will\n            yield different splits. Since cv.split() is called in\n            evaluate_candidates, this means that candidates will be evaluated\n            on different splits each time evaluate_candidates is called. This\n            might be a methodological issue depending on the search strategy\n            that you're implementing. To prevent randomized splitters from\n            being used, you may use _split._yields_constant_splits()\n\n        Examples\n        --------\n\n        ::\n\n            def _run_search(self, evaluate_candidates):\n                'Try C=0.1 only if C=1 is better than C=10'\n                all_results = evaluate_candidates([{'C': 1}, {'C': 10}])\n                score = all_results['mean_test_score']\n                if score[0] < score[1]:\n                    evaluate_candidates([{'C': 0.1}])\n        "
        raise NotImplementedError('_run_search not implemented.')

    def _check_refit_for_multimetric(self, scores):
        if False:
            print('Hello World!')
        'Check `refit` is compatible with `scores` is valid'
        multimetric_refit_msg = f'For multi-metric scoring, the parameter refit must be set to a scorer key or a callable to refit an estimator with the best parameter setting on the whole data and make the best_* attributes available for that metric. If this is not needed, refit should be set to False explicitly. {self.refit!r} was passed.'
        valid_refit_dict = isinstance(self.refit, str) and self.refit in scores
        if self.refit is not False and (not valid_refit_dict) and (not callable(self.refit)):
            raise ValueError(multimetric_refit_msg)

    @staticmethod
    def _select_best_index(refit, refit_metric, results):
        if False:
            print('Hello World!')
        'Select index of the best combination of hyperparemeters.'
        if callable(refit):
            best_index = refit(results)
            if not isinstance(best_index, numbers.Integral):
                raise TypeError('best_index_ returned is not an integer')
            if best_index < 0 or best_index >= len(results['params']):
                raise IndexError('best_index_ index out of range')
        else:
            best_index = results[f'rank_test_{refit_metric}'].argmin()
        return best_index

    def _get_scorers(self, convert_multimetric):
        if False:
            return 10
        'Get the scorer(s) to be used.\n\n        This is used in ``fit`` and ``get_metadata_routing``.\n\n        Parameters\n        ----------\n        convert_multimetric : bool\n            Whether to convert a dict of scorers to a _MultimetricScorer. This\n            is used in ``get_metadata_routing`` to include the routing info for\n            multiple scorers.\n\n        Returns\n        -------\n        scorers, refit_metric\n        '
        refit_metric = 'score'
        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(self.estimator, self.scoring)
        else:
            scorers = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(scorers)
            refit_metric = self.refit
            if convert_multimetric and isinstance(scorers, dict):
                scorers = _MultimetricScorer(scorers=scorers, raise_exc=self.error_score == 'raise')
        return (scorers, refit_metric)

    def _get_routed_params_for_fit(self, params):
        if False:
            i = 10
            return i + 15
        "Get the parameters to be used for routing.\n\n        This is a method instead of a snippet in ``fit`` since it's used twice,\n        here in ``fit``, and in ``HalvingRandomSearchCV.fit``.\n        "
        if _routing_enabled():
            routed_params = process_routing(self, 'fit', **params)
        else:
            params = params.copy()
            groups = params.pop('groups', None)
            routed_params = Bunch(estimator=Bunch(fit=params), splitter=Bunch(split={'groups': groups}), scorer=Bunch(score={}))
        return routed_params

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y=None, **params):
        if False:
            return 10
        'Run fit with all sets of parameters.\n\n        Parameters\n        ----------\n\n        X : array-like of shape (n_samples, n_features)\n            Training vector, where `n_samples` is the number of samples and\n            `n_features` is the number of features.\n\n        y : array-like of shape (n_samples, n_output)             or (n_samples,), default=None\n            Target relative to X for classification or regression;\n            None for unsupervised learning.\n\n        **params : dict of str -> object\n            Parameters passed to the ``fit`` method of the estimator, the scorer,\n            and the CV splitter.\n\n            If a fit parameter is an array-like whose length is equal to\n            `num_samples` then it will be split across CV groups along with `X`\n            and `y`. For example, the :term:`sample_weight` parameter is split\n            because `len(sample_weights) = len(X)`.\n\n        Returns\n        -------\n        self : object\n            Instance of fitted estimator.\n        '
        estimator = self.estimator
        (scorers, refit_metric) = self._get_scorers(convert_multimetric=False)
        (X, y) = indexable(X, y)
        params = _check_method_params(X, params=params)
        routed_params = self._get_routed_params_for_fit(params)
        cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv_orig.get_n_splits(X, y, **routed_params.splitter.split)
        base_estimator = clone(self.estimator)
        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)
        fit_and_score_kwargs = dict(scorer=scorers, fit_params=routed_params.estimator.fit, score_params=routed_params.scorer.score, return_train_score=self.return_train_score, return_n_test_samples=True, return_times=True, return_parameters=False, error_score=self.error_score, verbose=self.verbose)
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, cv=None, more_results=None):
                if False:
                    while True:
                        i = 10
                cv = cv or cv_orig
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)
                if self.verbose > 0:
                    print('Fitting {0} folds for each of {1} candidates, totalling {2} fits'.format(n_splits, n_candidates, n_candidates * n_splits))
                out = parallel((delayed(_fit_and_score)(clone(base_estimator), X, y, train=train, test=test, parameters=parameters, split_progress=(split_idx, n_splits), candidate_progress=(cand_idx, n_candidates), **fit_and_score_kwargs) for ((cand_idx, parameters), (split_idx, (train, test))) in product(enumerate(candidate_params), enumerate(cv.split(X, y, **routed_params.splitter.split)))))
                if len(out) < 1:
                    raise ValueError('No fits were performed. Was the CV iterator empty? Were there no candidates?')
                elif len(out) != n_candidates * n_splits:
                    raise ValueError('cv.split and cv.get_n_splits returned inconsistent results. Expected {} splits, got {}'.format(n_splits, len(out) // n_candidates))
                _warn_or_raise_about_fit_failures(out, self.error_score)
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)
                all_candidate_params.extend(candidate_params)
                all_out.extend(out)
                if more_results is not None:
                    for (key, value) in more_results.items():
                        all_more_results[key].extend(value)
                nonlocal results
                results = self._format_results(all_candidate_params, n_splits, all_out, all_more_results)
                return results
            self._run_search(evaluate_candidates)
            first_test_score = all_out[0]['test_scores']
            self.multimetric_ = isinstance(first_test_score, dict)
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit
        if self.refit or not self.multimetric_:
            self.best_index_ = self._select_best_index(self.refit, refit_metric, results)
            if not callable(self.refit):
                self.best_score_ = results[f'mean_test_{refit_metric}'][self.best_index_]
            self.best_params_ = results['params'][self.best_index_]
        if self.refit:
            self.best_estimator_ = clone(base_estimator).set_params(**clone(self.best_params_, safe=False))
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **routed_params.estimator.fit)
            else:
                self.best_estimator_.fit(X, **routed_params.estimator.fit)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time
            if hasattr(self.best_estimator_, 'feature_names_in_'):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_
        self.scorer_ = scorers
        self.cv_results_ = results
        self.n_splits_ = n_splits
        return self

    def _format_results(self, candidate_params, n_splits, out, more_results=None):
        if False:
            print('Hello World!')
        n_candidates = len(candidate_params)
        out = _aggregate_score_dicts(out)
        results = dict(more_results or {})
        for (key, val) in results.items():
            results[key] = np.asarray(val)

        def _store(key_name, array, weights=None, splits=False, rank=False):
            if False:
                print('Hello World!')
            'A small helper to store the scores/times to the cv_results_'
            array = np.array(array, dtype=np.float64).reshape(n_candidates, n_splits)
            if splits:
                for split_idx in range(n_splits):
                    results['split%d_%s' % (split_idx, key_name)] = array[:, split_idx]
            array_means = np.average(array, axis=1, weights=weights)
            results['mean_%s' % key_name] = array_means
            if key_name.startswith(('train_', 'test_')) and np.any(~np.isfinite(array_means)):
                warnings.warn(f"One or more of the {key_name.split('_')[0]} scores are non-finite: {array_means}", category=UserWarning)
            array_stds = np.sqrt(np.average((array - array_means[:, np.newaxis]) ** 2, axis=1, weights=weights))
            results['std_%s' % key_name] = array_stds
            if rank:
                if np.isnan(array_means).all():
                    rank_result = np.ones_like(array_means, dtype=np.int32)
                else:
                    min_array_means = np.nanmin(array_means) - 1
                    array_means = np.nan_to_num(array_means, nan=min_array_means)
                    rank_result = rankdata(-array_means, method='min').astype(np.int32, copy=False)
                results['rank_%s' % key_name] = rank_result
        _store('fit_time', out['fit_time'])
        _store('score_time', out['score_time'])
        param_results = defaultdict(partial(MaskedArray, np.empty(n_candidates), mask=True, dtype=object))
        for (cand_idx, params) in enumerate(candidate_params):
            for (name, value) in params.items():
                param_results['param_%s' % name][cand_idx] = value
        results.update(param_results)
        results['params'] = candidate_params
        test_scores_dict = _normalize_score_results(out['test_scores'])
        if self.return_train_score:
            train_scores_dict = _normalize_score_results(out['train_scores'])
        for scorer_name in test_scores_dict:
            _store('test_%s' % scorer_name, test_scores_dict[scorer_name], splits=True, rank=True, weights=None)
            if self.return_train_score:
                _store('train_%s' % scorer_name, train_scores_dict[scorer_name], splits=True)
        return results

    def get_metadata_routing(self):
        if False:
            i = 10
            return i + 15
        'Get metadata routing of this object.\n\n        Please check :ref:`User Guide <metadata_routing>` on how the routing\n        mechanism works.\n\n        .. versionadded:: 1.4\n\n        Returns\n        -------\n        routing : MetadataRouter\n            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating\n            routing information.\n        '
        router = MetadataRouter(owner=self.__class__.__name__)
        router.add(estimator=self.estimator, method_mapping=MethodMapping().add(caller='fit', callee='fit'))
        (scorer, _) = self._get_scorers(convert_multimetric=True)
        router.add(scorer=scorer, method_mapping=MethodMapping().add(caller='score', callee='score').add(caller='fit', callee='score'))
        router.add(splitter=self.cv, method_mapping=MethodMapping().add(caller='fit', callee='split'))
        return router

class GridSearchCV(BaseSearchCV):
    """Exhaustive search over specified parameter values for an estimator.

    Important members are fit, predict.

    GridSearchCV implements a "fit" and a "score" method.
    It also implements "score_samples", "predict", "predict_proba",
    "decision_function", "transform" and "inverse_transform" if they are
    implemented in the estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated grid-search over a parameter grid.

    Read more in the :ref:`User Guide <grid_search>`.

    Parameters
    ----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (`str`) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    scoring : str, callable, list, tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

        If `scoring` represents a single score, one can use:

        - a single string (see :ref:`scoring_parameter`);
        - a callable (see :ref:`scoring`) that returns a single value.

        If `scoring` represents multiple scores, one can use:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.

        See :ref:`multimetric_grid_search` for an example.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionchanged:: v0.20
           `n_jobs` default changed from 1 to None

    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a `str` denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given ``cv_results_``. In that
        case, the ``best_estimator_`` and ``best_params_`` will be set
        according to the returned ``best_index_`` while the ``best_score_``
        attribute will not be available.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``GridSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

        See :ref:`sphx_glr_auto_examples_model_selection_plot_grid_search_digits.py`
        to see how to design a custom selection strategy using a callable
        via `refit`.

        .. versionchanged:: 0.20
            Support for callable added.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    verbose : int
        Controls the verbosity: the higher, the more messages.

        - >1 : the computation time for each fold and parameter candidate is
          displayed;
        - >2 : the score is also displayed;
        - >3 : the fold and candidate parameter indexes are also displayed
          together with the starting time of the computation.

    pre_dispatch : int, or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

        .. versionadded:: 0.19

        .. versionchanged:: 0.21
            Default value was changed from ``True`` to ``False``

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +------------+-----------+------------+-----------------+---+---------+
        |param_kernel|param_gamma|param_degree|split0_test_score|...|rank_t...|
        +============+===========+============+=================+===+=========+
        |  'poly'    |     --    |      2     |       0.80      |...|    2    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'poly'    |     --    |      3     |       0.70      |...|    4    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.1   |     --     |       0.80      |...|    3    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.2   |     --     |       0.93      |...|    1    |
        +------------+-----------+------------+-----------------+---+---------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
                                         mask = [False False False False]...)
            'param_gamma': masked_array(data = [-- -- 0.1 0.2],
                                        mask = [ True  True False False]...),
            'param_degree': masked_array(data = [2.0 3.0 -- --],
                                         mask = [False False  True  True]...),
            'split0_test_score'  : [0.80, 0.70, 0.80, 0.93],
            'split1_test_score'  : [0.82, 0.50, 0.70, 0.78],
            'mean_test_score'    : [0.81, 0.60, 0.75, 0.85],
            'std_test_score'     : [0.01, 0.10, 0.05, 0.08],
            'rank_test_score'    : [2, 4, 3, 1],
            'split0_train_score' : [0.80, 0.92, 0.70, 0.93],
            'split1_train_score' : [0.82, 0.55, 0.70, 0.87],
            'mean_train_score'   : [0.81, 0.74, 0.70, 0.90],
            'std_train_score'    : [0.01, 0.19, 0.00, 0.03],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00, 0.01],
            'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
            }

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

        This attribute is not available if ``refit`` is a function.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

        This is present only if ``refit`` is not False.

        .. versionadded:: 0.20

    multimetric_ : bool
        Whether or not the scorers compute several metrics.

    classes_ : ndarray of shape (n_classes,)
        The classes labels. This is present only if ``refit`` is specified and
        the underlying estimator is a classifier.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `n_features_in_` when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `feature_names_in_` when fit.

        .. versionadded:: 1.0

    See Also
    --------
    ParameterGrid : Generates all the combinations of a hyperparameter grid.
    train_test_split : Utility function to split the data into a development
        set usable for fitting a GridSearchCV instance and an evaluation set
        for its final evaluation.
    sklearn.metrics.make_scorer : Make a scorer from a performance metric or
        loss function.

    Notes
    -----
    The parameters selected are those that maximize the score of the left out
    data, unless an explicit score is passed in which case it is used instead.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    point in the grid (and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    Examples
    --------
    >>> from sklearn import svm, datasets
    >>> from sklearn.model_selection import GridSearchCV
    >>> iris = datasets.load_iris()
    >>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    >>> svc = svm.SVC()
    >>> clf = GridSearchCV(svc, parameters)
    >>> clf.fit(iris.data, iris.target)
    GridSearchCV(estimator=SVC(),
                 param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')})
    >>> sorted(clf.cv_results_.keys())
    ['mean_fit_time', 'mean_score_time', 'mean_test_score',...
     'param_C', 'param_kernel', 'params',...
     'rank_test_score', 'split0_test_score',...
     'split2_test_score', ...
     'std_fit_time', 'std_score_time', 'std_test_score']
    """
    _required_parameters = ['estimator', 'param_grid']
    _parameter_constraints: dict = {**BaseSearchCV._parameter_constraints, 'param_grid': [dict, list]}

    def __init__(self, estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=np.nan, return_train_score=False):
        if False:
            return 10
        super().__init__(estimator=estimator, scoring=scoring, n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose, pre_dispatch=pre_dispatch, error_score=error_score, return_train_score=return_train_score)
        self.param_grid = param_grid

    def _run_search(self, evaluate_candidates):
        if False:
            return 10
        'Search all candidates in param_grid'
        evaluate_candidates(ParameterGrid(self.param_grid))

class RandomizedSearchCV(BaseSearchCV):
    """Randomized search on hyper parameters.

    RandomizedSearchCV implements a "fit" and a "score" method.
    It also implements "score_samples", "predict", "predict_proba",
    "decision_function", "transform" and "inverse_transform" if they are
    implemented in the estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.

    In contrast to GridSearchCV, not all parameter values are tried out, but
    rather a fixed number of parameter settings is sampled from the specified
    distributions. The number of parameter settings that are tried is
    given by n_iter.

    If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous
    parameters.

    Read more in the :ref:`User Guide <randomized_parameter_search>`.

    .. versionadded:: 0.14

    Parameters
    ----------
    estimator : estimator object
        An object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions : dict or list of dicts
        Dictionary with parameters names (`str`) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.
        If a list of dicts is given, first a dict is sampled uniformly, and
        then a parameter is sampled using that dict as above.

    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.

    scoring : str, callable, list, tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

        If `scoring` represents a single score, one can use:

        - a single string (see :ref:`scoring_parameter`);
        - a callable (see :ref:`scoring`) that returns a single value.

        If `scoring` represents multiple scores, one can use:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.

        See :ref:`multimetric_grid_search` for an example.

        If None, the estimator's score method is used.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionchanged:: v0.20
           `n_jobs` default changed from 1 to None

    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a `str` denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given the ``cv_results``. In that
        case, the ``best_estimator_`` and ``best_params_`` will be set
        according to the returned ``best_index_`` while the ``best_score_``
        attribute will not be available.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``RandomizedSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

        .. versionchanged:: 0.20
            Support for callable added.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    verbose : int
        Controls the verbosity: the higher, the more messages.

        - >1 : the computation time for each fold and parameter candidate is
          displayed;
        - >2 : the score is also displayed;
        - >3 : the fold and candidate parameter indexes are also displayed
          together with the starting time of the computation.

    pre_dispatch : int, or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.
        See :term:`Glossary <random_state>`.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

        .. versionadded:: 0.19

        .. versionchanged:: 0.21
            Default value was changed from ``True`` to ``False``

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +--------------+-------------+-------------------+---+---------------+
        | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
        +==============+=============+===================+===+===============+
        |    'rbf'     |     0.1     |       0.80        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.2     |       0.84        |...|       3       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.3     |       0.70        |...|       2       |
        +--------------+-------------+-------------------+---+---------------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.80, 0.84, 0.70],
            'split1_test_score'  : [0.82, 0.50, 0.70],
            'mean_test_score'    : [0.81, 0.67, 0.70],
            'std_test_score'     : [0.01, 0.24, 0.00],
            'rank_test_score'    : [1, 3, 2],
            'split0_train_score' : [0.80, 0.92, 0.70],
            'split1_train_score' : [0.82, 0.55, 0.70],
            'mean_train_score'   : [0.81, 0.74, 0.70],
            'std_train_score'    : [0.01, 0.19, 0.00],
            'mean_fit_time'      : [0.73, 0.63, 0.43],
            'std_fit_time'       : [0.01, 0.02, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00],
            'params'             : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
            }

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        For multi-metric evaluation, this attribute is present only if
        ``refit`` is specified.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

        This attribute is not available if ``refit`` is a function.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

        This is present only if ``refit`` is not False.

        .. versionadded:: 0.20

    multimetric_ : bool
        Whether or not the scorers compute several metrics.

    classes_ : ndarray of shape (n_classes,)
        The classes labels. This is present only if ``refit`` is specified and
        the underlying estimator is a classifier.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `n_features_in_` when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `feature_names_in_` when fit.

        .. versionadded:: 1.0

    See Also
    --------
    GridSearchCV : Does exhaustive search over a grid of parameters.
    ParameterSampler : A generator over parameter settings, constructed from
        param_distributions.

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    parameter setting(and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import RandomizedSearchCV
    >>> from scipy.stats import uniform
    >>> iris = load_iris()
    >>> logistic = LogisticRegression(solver='saga', tol=1e-2, max_iter=200,
    ...                               random_state=0)
    >>> distributions = dict(C=uniform(loc=0, scale=4),
    ...                      penalty=['l2', 'l1'])
    >>> clf = RandomizedSearchCV(logistic, distributions, random_state=0)
    >>> search = clf.fit(iris.data, iris.target)
    >>> search.best_params_
    {'C': 2..., 'penalty': 'l1'}
    """
    _required_parameters = ['estimator', 'param_distributions']
    _parameter_constraints: dict = {**BaseSearchCV._parameter_constraints, 'param_distributions': [dict, list], 'n_iter': [Interval(numbers.Integral, 1, None, closed='left')], 'random_state': ['random_state']}

    def __init__(self, estimator, param_distributions, *, n_iter=10, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', random_state=None, error_score=np.nan, return_train_score=False):
        if False:
            while True:
                i = 10
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        super().__init__(estimator=estimator, scoring=scoring, n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose, pre_dispatch=pre_dispatch, error_score=error_score, return_train_score=return_train_score)

    def _run_search(self, evaluate_candidates):
        if False:
            while True:
                i = 10
        'Search n_iter candidates from param_distributions'
        evaluate_candidates(ParameterSampler(self.param_distributions, self.n_iter, random_state=self.random_state))