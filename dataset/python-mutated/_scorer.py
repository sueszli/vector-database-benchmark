"""
The :mod:`sklearn.metrics.scorer` submodule implements a flexible
interface for model selection and evaluation using
arbitrary score functions.

A scorer object is a callable that can be passed to
:class:`~sklearn.model_selection.GridSearchCV` or
:func:`sklearn.model_selection.cross_val_score` as the ``scoring``
parameter, to specify how a model should be evaluated.

The signature of the call is ``(estimator, X, y)`` where ``estimator``
is the model to be evaluated, ``X`` is the test data and ``y`` is the
ground truth labeling (or ``None`` in the case of unsupervised models).
"""
import copy
import warnings
from collections import Counter
from functools import partial
from inspect import signature
from traceback import format_exc
from ..base import is_regressor
from ..utils import Bunch
from ..utils._param_validation import HasMethods, Hidden, StrOptions, validate_params
from ..utils._response import _get_response_values
from ..utils.metadata_routing import MetadataRequest, MetadataRouter, _MetadataRequester, _raise_for_params, _routing_enabled, get_routing_for_object, process_routing
from ..utils.validation import _check_response_method
from . import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, class_likelihood_ratios, explained_variance_score, f1_score, jaccard_score, log_loss, matthews_corrcoef, max_error, mean_absolute_error, mean_absolute_percentage_error, mean_gamma_deviance, mean_poisson_deviance, mean_squared_error, mean_squared_log_error, median_absolute_error, precision_score, r2_score, recall_score, roc_auc_score, root_mean_squared_error, root_mean_squared_log_error, top_k_accuracy_score
from .cluster import adjusted_mutual_info_score, adjusted_rand_score, completeness_score, fowlkes_mallows_score, homogeneity_score, mutual_info_score, normalized_mutual_info_score, rand_score, v_measure_score

def _cached_call(cache, estimator, response_method, *args, **kwargs):
    if False:
        print('Hello World!')
    'Call estimator with method and args and kwargs.'
    if cache is not None and response_method in cache:
        return cache[response_method]
    (result, _) = _get_response_values(estimator, *args, response_method=response_method, **kwargs)
    if cache is not None:
        cache[response_method] = result
    return result

class _MultimetricScorer:
    """Callable for multimetric scoring used to avoid repeated calls
    to `predict_proba`, `predict`, and `decision_function`.

    `_MultimetricScorer` will return a dictionary of scores corresponding to
    the scorers in the dictionary. Note that `_MultimetricScorer` can be
    created with a dictionary with one key  (i.e. only one actual scorer).

    Parameters
    ----------
    scorers : dict
        Dictionary mapping names to callable scorers.

    raise_exc : bool, default=True
        Whether to raise the exception in `__call__` or not. If set to `False`
        a formatted string of the exception details is passed as result of
        the failing scorer.
    """

    def __init__(self, *, scorers, raise_exc=True):
        if False:
            for i in range(10):
                print('nop')
        self._scorers = scorers
        self._raise_exc = raise_exc

    def __call__(self, estimator, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Evaluate predicted target values.'
        scores = {}
        cache = {} if self._use_cache(estimator) else None
        cached_call = partial(_cached_call, cache)
        if _routing_enabled():
            routed_params = process_routing(self, 'score', **kwargs)
        else:
            routed_params = Bunch(**{name: Bunch(score=kwargs) for name in self._scorers})
        for (name, scorer) in self._scorers.items():
            try:
                if isinstance(scorer, _BaseScorer):
                    score = scorer._score(cached_call, estimator, *args, **routed_params.get(name).score)
                else:
                    score = scorer(estimator, *args, **routed_params.get(name).score)
                scores[name] = score
            except Exception as e:
                if self._raise_exc:
                    raise e
                else:
                    scores[name] = format_exc()
        return scores

    def _use_cache(self, estimator):
        if False:
            for i in range(10):
                print('nop')
        'Return True if using a cache is beneficial, thus when a response method will\n        be called several time.\n        '
        if len(self._scorers) == 1:
            return False
        counter = Counter([_check_response_method(estimator, scorer._response_method).__name__ for scorer in self._scorers.values() if isinstance(scorer, _BaseScorer)])
        if any((val > 1 for val in counter.values())):
            return True
        return False

    def get_metadata_routing(self):
        if False:
            for i in range(10):
                print('nop')
        'Get metadata routing of this object.\n\n        Please check :ref:`User Guide <metadata_routing>` on how the routing\n        mechanism works.\n\n        .. versionadded:: 1.3\n\n        Returns\n        -------\n        routing : MetadataRouter\n            A :class:`~utils.metadata_routing.MetadataRouter` encapsulating\n            routing information.\n        '
        return MetadataRouter(owner=self.__class__.__name__).add(**self._scorers, method_mapping='score')

class _BaseScorer(_MetadataRequester):

    def __init__(self, score_func, sign, kwargs, response_method='predict'):
        if False:
            return 10
        self._score_func = score_func
        self._sign = sign
        self._kwargs = kwargs
        self._response_method = response_method

    def _get_pos_label(self):
        if False:
            print('Hello World!')
        if 'pos_label' in self._kwargs:
            return self._kwargs['pos_label']
        score_func_params = signature(self._score_func).parameters
        if 'pos_label' in score_func_params:
            return score_func_params['pos_label'].default
        return None

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        sign_string = '' if self._sign > 0 else ', greater_is_better=False'
        response_method_string = f', response_method={self._response_method!r}'
        kwargs_string = ''.join([f', {k}={v}' for (k, v) in self._kwargs.items()])
        return f'make_scorer({self._score_func.__name__}{sign_string}{response_method_string}{kwargs_string})'

    def __call__(self, estimator, X, y_true, sample_weight=None, **kwargs):
        if False:
            return 10
        'Evaluate predicted target values for X relative to y_true.\n\n        Parameters\n        ----------\n        estimator : object\n            Trained estimator to use for scoring. Must have a predict_proba\n            method; the output of that is used to compute the score.\n\n        X : {array-like, sparse matrix}\n            Test data that will be fed to estimator.predict.\n\n        y_true : array-like\n            Gold standard target values for X.\n\n        sample_weight : array-like of shape (n_samples,), default=None\n            Sample weights.\n\n        **kwargs : dict\n            Other parameters passed to the scorer. Refer to\n            :func:`set_score_request` for more details.\n\n            Only available if `enable_metadata_routing=True`. See the\n            :ref:`User Guide <metadata_routing>`.\n\n            .. versionadded:: 1.3\n\n        Returns\n        -------\n        score : float\n            Score function applied to prediction of estimator on X.\n        '
        _raise_for_params(kwargs, self, None)
        _kwargs = copy.deepcopy(kwargs)
        if sample_weight is not None:
            _kwargs['sample_weight'] = sample_weight
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)

    def _warn_overlap(self, message, kwargs):
        if False:
            return 10
        'Warn if there is any overlap between ``self._kwargs`` and ``kwargs``.\n\n        This method is intended to be used to check for overlap between\n        ``self._kwargs`` and ``kwargs`` passed as metadata.\n        '
        _kwargs = set() if self._kwargs is None else set(self._kwargs.keys())
        overlap = _kwargs.intersection(kwargs.keys())
        if overlap:
            warnings.warn(f'{message} Overlapping parameters are: {overlap}', UserWarning)

    def set_score_request(self, **kwargs):
        if False:
            i = 10
            return i + 15
        'Set requested parameters by the scorer.\n\n        Please see :ref:`User Guide <metadata_routing>` on how the routing\n        mechanism works.\n\n        .. versionadded:: 1.3\n\n        Parameters\n        ----------\n        kwargs : dict\n            Arguments should be of the form ``param_name=alias``, and `alias`\n            can be one of ``{True, False, None, str}``.\n        '
        if not _routing_enabled():
            raise RuntimeError('This method is only available when metadata routing is enabled. You can enable it using sklearn.set_config(enable_metadata_routing=True).')
        self._warn_overlap(message='You are setting metadata request for parameters which are already set as kwargs for this metric. These set values will be overridden by passed metadata if provided. Please pass them either as metadata or kwargs to `make_scorer`.', kwargs=kwargs)
        self._metadata_request = MetadataRequest(owner=self.__class__.__name__)
        for (param, alias) in kwargs.items():
            self._metadata_request.score.add_request(param=param, alias=alias)
        return self

class _Scorer(_BaseScorer):

    def _score(self, method_caller, estimator, X, y_true, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Evaluate the response method of `estimator` on `X` and `y_true`.\n\n        Parameters\n        ----------\n        method_caller : callable\n            Returns predictions given an estimator, method name, and other\n            arguments, potentially caching results.\n\n        estimator : object\n            Trained estimator to use for scoring.\n\n        X : {array-like, sparse matrix}\n            Test data that will be fed to clf.decision_function or\n            clf.predict_proba.\n\n        y_true : array-like\n            Gold standard target values for X. These must be class labels,\n            not decision function values.\n\n        **kwargs : dict\n            Other parameters passed to the scorer. Refer to\n            :func:`set_score_request` for more details.\n\n        Returns\n        -------\n        score : float\n            Score function applied to prediction of estimator on X.\n        '
        self._warn_overlap(message='There is an overlap between set kwargs of this scorer instance and passed metadata. Please pass them either as kwargs to `make_scorer` or metadata, but not both.', kwargs=kwargs)
        pos_label = None if is_regressor(estimator) else self._get_pos_label()
        response_method = _check_response_method(estimator, self._response_method)
        y_pred = method_caller(estimator, response_method.__name__, X, pos_label=pos_label)
        scoring_kwargs = {**self._kwargs, **kwargs}
        return self._sign * self._score_func(y_true, y_pred, **scoring_kwargs)

@validate_params({'scoring': [str, callable, None]}, prefer_skip_nested_validation=True)
def get_scorer(scoring):
    if False:
        while True:
            i = 10
    'Get a scorer from string.\n\n    Read more in the :ref:`User Guide <scoring_parameter>`.\n    :func:`~sklearn.metrics.get_scorer_names` can be used to retrieve the names\n    of all available scorers.\n\n    Parameters\n    ----------\n    scoring : str, callable or None\n        Scoring method as string. If callable it is returned as is.\n        If None, returns None.\n\n    Returns\n    -------\n    scorer : callable\n        The scorer.\n\n    Notes\n    -----\n    When passed a string, this function always returns a copy of the scorer\n    object. Calling `get_scorer` twice for the same scorer results in two\n    separate scorer objects.\n    '
    if isinstance(scoring, str):
        try:
            scorer = copy.deepcopy(_SCORERS[scoring])
        except KeyError:
            raise ValueError('%r is not a valid scoring value. Use sklearn.metrics.get_scorer_names() to get valid options.' % scoring)
    else:
        scorer = scoring
    return scorer

class _PassthroughScorer:

    def __init__(self, estimator):
        if False:
            return 10
        self._estimator = estimator

    def __call__(self, estimator, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Method that wraps estimator.score'
        return estimator.score(*args, **kwargs)

    def get_metadata_routing(self):
        if False:
            return 10
        'Get requested data properties.\n\n        Please check :ref:`User Guide <metadata_routing>` on how the routing\n        mechanism works.\n\n        .. versionadded:: 1.3\n\n        Returns\n        -------\n        routing : MetadataRouter\n            A :class:`~utils.metadata_routing.MetadataRouter` encapsulating\n            routing information.\n        '
        return get_routing_for_object(self._estimator)

def _check_multimetric_scoring(estimator, scoring):
    if False:
        for i in range(10):
            print('nop')
    'Check the scoring parameter in cases when multiple metrics are allowed.\n\n    In addition, multimetric scoring leverages a caching mechanism to not call the same\n    estimator response method multiple times. Hence, the scorer is modified to only use\n    a single response method given a list of response methods and the estimator.\n\n    Parameters\n    ----------\n    estimator : sklearn estimator instance\n        The estimator for which the scoring will be applied.\n\n    scoring : list, tuple or dict\n        Strategy to evaluate the performance of the cross-validated model on\n        the test set.\n\n        The possibilities are:\n\n        - a list or tuple of unique strings;\n        - a callable returning a dictionary where they keys are the metric\n          names and the values are the metric scores;\n        - a dictionary with metric names as keys and callables a values.\n\n        See :ref:`multimetric_grid_search` for an example.\n\n    Returns\n    -------\n    scorers_dict : dict\n        A dict mapping each scorer name to its validated scorer.\n    '
    err_msg_generic = f'scoring is invalid (got {scoring!r}). Refer to the scoring glossary for details: https://scikit-learn.org/stable/glossary.html#term-scoring'
    if isinstance(scoring, (list, tuple, set)):
        err_msg = 'The list/tuple elements must be unique strings of predefined scorers. '
        try:
            keys = set(scoring)
        except TypeError as e:
            raise ValueError(err_msg) from e
        if len(keys) != len(scoring):
            raise ValueError(f'{err_msg} Duplicate elements were found in the given list. {scoring!r}')
        elif len(keys) > 0:
            if not all((isinstance(k, str) for k in keys)):
                if any((callable(k) for k in keys)):
                    raise ValueError(f'{err_msg} One or more of the elements were callables. Use a dict of score name mapped to the scorer callable. Got {scoring!r}')
                else:
                    raise ValueError(f'{err_msg} Non-string types were found in the given list. Got {scoring!r}')
            scorers = {scorer: check_scoring(estimator, scoring=scorer) for scorer in scoring}
        else:
            raise ValueError(f'{err_msg} Empty list was given. {scoring!r}')
    elif isinstance(scoring, dict):
        keys = set(scoring)
        if not all((isinstance(k, str) for k in keys)):
            raise ValueError(f'Non-string types were found in the keys of the given dict. scoring={scoring!r}')
        if len(keys) == 0:
            raise ValueError(f'An empty dict was passed. {scoring!r}')
        scorers = {key: check_scoring(estimator, scoring=scorer) for (key, scorer) in scoring.items()}
    else:
        raise ValueError(err_msg_generic)
    return scorers

def _get_response_method(response_method, needs_threshold, needs_proba):
    if False:
        for i in range(10):
            print('nop')
    'Handles deprecation of `needs_threshold` and `needs_proba` parameters in\n    favor of `response_method`.\n    '
    needs_threshold_provided = needs_threshold != 'deprecated'
    needs_proba_provided = needs_proba != 'deprecated'
    response_method_provided = response_method is not None
    needs_threshold = False if needs_threshold == 'deprecated' else needs_threshold
    needs_proba = False if needs_proba == 'deprecated' else needs_proba
    if response_method_provided and (needs_proba_provided or needs_threshold_provided):
        raise ValueError('You cannot set both `response_method` and `needs_proba` or `needs_threshold` at the same time. Only use `response_method` since the other two are deprecated in version 1.4 and will be removed in 1.6.')
    if needs_proba_provided or needs_threshold_provided:
        warnings.warn('The `needs_threshold` and `needs_proba` parameter are deprecated in version 1.4 and will be removed in 1.6. You can either let `response_method` be `None` or set it to `predict` to preserve the same behaviour.', FutureWarning)
    if response_method_provided:
        return response_method
    if needs_proba is True and needs_threshold is True:
        raise ValueError('You cannot set both `needs_proba` and `needs_threshold` at the same time. Use `response_method` instead since the other two are deprecated in version 1.4 and will be removed in 1.6.')
    if needs_proba is True:
        response_method = 'predict_proba'
    elif needs_threshold is True:
        response_method = ('decision_function', 'predict_proba')
    else:
        response_method = 'predict'
    return response_method

@validate_params({'score_func': [callable], 'response_method': [None, list, tuple, StrOptions({'predict', 'predict_proba', 'decision_function'})], 'greater_is_better': ['boolean'], 'needs_proba': ['boolean', Hidden(StrOptions({'deprecated'}))], 'needs_threshold': ['boolean', Hidden(StrOptions({'deprecated'}))]}, prefer_skip_nested_validation=True)
def make_scorer(score_func, *, response_method=None, greater_is_better=True, needs_proba='deprecated', needs_threshold='deprecated', **kwargs):
    if False:
        return 10
    'Make a scorer from a performance metric or loss function.\n\n    A scorer is a wrapper around an arbitrary metric or loss function that is called\n    with the signature `scorer(estimator, X, y_true, **kwargs)`.\n\n    It is accepted in all scikit-learn estimators or functions allowing a `scoring`\n    parameter.\n\n    The parameter `response_method` allows to specify which method of the estimator\n    should be used to feed the scoring/loss function.\n\n    Read more in the :ref:`User Guide <scoring>`.\n\n    Parameters\n    ----------\n    score_func : callable\n        Score function (or loss function) with signature\n        ``score_func(y, y_pred, **kwargs)``.\n\n    response_method : {"predict_proba", "decision_function", "predict"} or             list/tuple of such str, default=None\n\n        Specifies the response method to use get prediction from an estimator\n        (i.e. :term:`predict_proba`, :term:`decision_function` or\n        :term:`predict`). Possible choices are:\n\n        - if `str`, it corresponds to the name to the method to return;\n        - if a list or tuple of `str`, it provides the method names in order of\n          preference. The method returned corresponds to the first method in\n          the list and which is implemented by `estimator`.\n        - if `None`, it is equivalent to `"predict"`.\n\n        .. versionadded:: 1.4\n\n    greater_is_better : bool, default=True\n        Whether `score_func` is a score function (default), meaning high is\n        good, or a loss function, meaning low is good. In the latter case, the\n        scorer object will sign-flip the outcome of the `score_func`.\n\n    needs_proba : bool, default=False\n        Whether `score_func` requires `predict_proba` to get probability\n        estimates out of a classifier.\n\n        If True, for binary `y_true`, the score function is supposed to accept\n        a 1D `y_pred` (i.e., probability of the positive class, shape\n        `(n_samples,)`).\n\n        .. deprecated:: 1.4\n           `needs_proba` is deprecated in version 1.4 and will be removed in\n           1.6. Use `response_method="predict_proba"` instead.\n\n    needs_threshold : bool, default=False\n        Whether `score_func` takes a continuous decision certainty.\n        This only works for binary classification using estimators that\n        have either a `decision_function` or `predict_proba` method.\n\n        If True, for binary `y_true`, the score function is supposed to accept\n        a 1D `y_pred` (i.e., probability of the positive class or the decision\n        function, shape `(n_samples,)`).\n\n        For example `average_precision` or the area under the roc curve\n        can not be computed using discrete predictions alone.\n\n        .. deprecated:: 1.4\n           `needs_threshold` is deprecated in version 1.4 and will be removed\n           in 1.6. Use `response_method=("decision_function", "predict_proba")`\n           instead to preserve the same behaviour.\n\n    **kwargs : additional arguments\n        Additional parameters to be passed to `score_func`.\n\n    Returns\n    -------\n    scorer : callable\n        Callable object that returns a scalar score; greater is better.\n\n    Examples\n    --------\n    >>> from sklearn.metrics import fbeta_score, make_scorer\n    >>> ftwo_scorer = make_scorer(fbeta_score, beta=2)\n    >>> ftwo_scorer\n    make_scorer(fbeta_score, response_method=\'predict\', beta=2)\n    >>> from sklearn.model_selection import GridSearchCV\n    >>> from sklearn.svm import LinearSVC\n    >>> grid = GridSearchCV(LinearSVC(), param_grid={\'C\': [1, 10]},\n    ...                     scoring=ftwo_scorer)\n    '
    response_method = _get_response_method(response_method, needs_threshold, needs_proba)
    sign = 1 if greater_is_better else -1
    return _Scorer(score_func, sign, kwargs, response_method)
explained_variance_scorer = make_scorer(explained_variance_score)
r2_scorer = make_scorer(r2_score)
max_error_scorer = make_scorer(max_error, greater_is_better=False)
neg_mean_squared_error_scorer = make_scorer(mean_squared_error, greater_is_better=False)
neg_mean_squared_log_error_scorer = make_scorer(mean_squared_log_error, greater_is_better=False)
neg_mean_absolute_error_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
neg_mean_absolute_percentage_error_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
neg_median_absolute_error_scorer = make_scorer(median_absolute_error, greater_is_better=False)
neg_root_mean_squared_error_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)
neg_root_mean_squared_log_error_scorer = make_scorer(root_mean_squared_log_error, greater_is_better=False)
neg_mean_poisson_deviance_scorer = make_scorer(mean_poisson_deviance, greater_is_better=False)
neg_mean_gamma_deviance_scorer = make_scorer(mean_gamma_deviance, greater_is_better=False)
accuracy_scorer = make_scorer(accuracy_score)
balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)
matthews_corrcoef_scorer = make_scorer(matthews_corrcoef)

def positive_likelihood_ratio(y_true, y_pred):
    if False:
        print('Hello World!')
    return class_likelihood_ratios(y_true, y_pred)[0]

def negative_likelihood_ratio(y_true, y_pred):
    if False:
        return 10
    return class_likelihood_ratios(y_true, y_pred)[1]
positive_likelihood_ratio_scorer = make_scorer(positive_likelihood_ratio)
neg_negative_likelihood_ratio_scorer = make_scorer(negative_likelihood_ratio, greater_is_better=False)
top_k_accuracy_scorer = make_scorer(top_k_accuracy_score, greater_is_better=True, response_method=('decision_function', 'predict_proba'))
roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, response_method=('decision_function', 'predict_proba'))
average_precision_scorer = make_scorer(average_precision_score, response_method=('decision_function', 'predict_proba'))
roc_auc_ovo_scorer = make_scorer(roc_auc_score, response_method='predict_proba', multi_class='ovo')
roc_auc_ovo_weighted_scorer = make_scorer(roc_auc_score, response_method='predict_proba', multi_class='ovo', average='weighted')
roc_auc_ovr_scorer = make_scorer(roc_auc_score, response_method='predict_proba', multi_class='ovr')
roc_auc_ovr_weighted_scorer = make_scorer(roc_auc_score, response_method='predict_proba', multi_class='ovr', average='weighted')
neg_log_loss_scorer = make_scorer(log_loss, greater_is_better=False, response_method='predict_proba')
neg_brier_score_scorer = make_scorer(brier_score_loss, greater_is_better=False, response_method='predict_proba')
brier_score_loss_scorer = make_scorer(brier_score_loss, greater_is_better=False, response_method='predict_proba')
adjusted_rand_scorer = make_scorer(adjusted_rand_score)
rand_scorer = make_scorer(rand_score)
homogeneity_scorer = make_scorer(homogeneity_score)
completeness_scorer = make_scorer(completeness_score)
v_measure_scorer = make_scorer(v_measure_score)
mutual_info_scorer = make_scorer(mutual_info_score)
adjusted_mutual_info_scorer = make_scorer(adjusted_mutual_info_score)
normalized_mutual_info_scorer = make_scorer(normalized_mutual_info_score)
fowlkes_mallows_scorer = make_scorer(fowlkes_mallows_score)
_SCORERS = dict(explained_variance=explained_variance_scorer, r2=r2_scorer, max_error=max_error_scorer, matthews_corrcoef=matthews_corrcoef_scorer, neg_median_absolute_error=neg_median_absolute_error_scorer, neg_mean_absolute_error=neg_mean_absolute_error_scorer, neg_mean_absolute_percentage_error=neg_mean_absolute_percentage_error_scorer, neg_mean_squared_error=neg_mean_squared_error_scorer, neg_mean_squared_log_error=neg_mean_squared_log_error_scorer, neg_root_mean_squared_error=neg_root_mean_squared_error_scorer, neg_root_mean_squared_log_error=neg_root_mean_squared_log_error_scorer, neg_mean_poisson_deviance=neg_mean_poisson_deviance_scorer, neg_mean_gamma_deviance=neg_mean_gamma_deviance_scorer, accuracy=accuracy_scorer, top_k_accuracy=top_k_accuracy_scorer, roc_auc=roc_auc_scorer, roc_auc_ovr=roc_auc_ovr_scorer, roc_auc_ovo=roc_auc_ovo_scorer, roc_auc_ovr_weighted=roc_auc_ovr_weighted_scorer, roc_auc_ovo_weighted=roc_auc_ovo_weighted_scorer, balanced_accuracy=balanced_accuracy_scorer, average_precision=average_precision_scorer, neg_log_loss=neg_log_loss_scorer, neg_brier_score=neg_brier_score_scorer, positive_likelihood_ratio=positive_likelihood_ratio_scorer, neg_negative_likelihood_ratio=neg_negative_likelihood_ratio_scorer, adjusted_rand_score=adjusted_rand_scorer, rand_score=rand_scorer, homogeneity_score=homogeneity_scorer, completeness_score=completeness_scorer, v_measure_score=v_measure_scorer, mutual_info_score=mutual_info_scorer, adjusted_mutual_info_score=adjusted_mutual_info_scorer, normalized_mutual_info_score=normalized_mutual_info_scorer, fowlkes_mallows_score=fowlkes_mallows_scorer)

def get_scorer_names():
    if False:
        return 10
    'Get the names of all available scorers.\n\n    These names can be passed to :func:`~sklearn.metrics.get_scorer` to\n    retrieve the scorer object.\n\n    Returns\n    -------\n    list of str\n        Names of all available scorers.\n    '
    return sorted(_SCORERS.keys())
for (name, metric) in [('precision', precision_score), ('recall', recall_score), ('f1', f1_score), ('jaccard', jaccard_score)]:
    _SCORERS[name] = make_scorer(metric, average='binary')
    for average in ['macro', 'micro', 'samples', 'weighted']:
        qualified_name = '{0}_{1}'.format(name, average)
        _SCORERS[qualified_name] = make_scorer(metric, pos_label=None, average=average)

@validate_params({'estimator': [HasMethods('fit')], 'scoring': [StrOptions(set(get_scorer_names())), callable, None], 'allow_none': ['boolean']}, prefer_skip_nested_validation=True)
def check_scoring(estimator, scoring=None, *, allow_none=False):
    if False:
        i = 10
        return i + 15
    "Determine scorer from user options.\n\n    A TypeError will be thrown if the estimator cannot be scored.\n\n    Parameters\n    ----------\n    estimator : estimator object implementing 'fit'\n        The object to use to fit the data.\n\n    scoring : str or callable, default=None\n        A string (see model evaluation documentation) or\n        a scorer callable object / function with signature\n        ``scorer(estimator, X, y)``.\n        If None, the provided estimator object's `score` method is used.\n\n    allow_none : bool, default=False\n        If no scoring is specified and the estimator has no score function, we\n        can either return None or raise an exception.\n\n    Returns\n    -------\n    scoring : callable\n        A scorer callable object / function with signature\n        ``scorer(estimator, X, y)``.\n    "
    if isinstance(scoring, str):
        return get_scorer(scoring)
    if callable(scoring):
        module = getattr(scoring, '__module__', None)
        if hasattr(module, 'startswith') and module.startswith('sklearn.metrics.') and (not module.startswith('sklearn.metrics._scorer')) and (not module.startswith('sklearn.metrics.tests.')):
            raise ValueError('scoring value %r looks like it is a metric function rather than a scorer. A scorer should require an estimator as its first parameter. Please use `make_scorer` to convert a metric to a scorer.' % scoring)
        return get_scorer(scoring)
    if scoring is None:
        if hasattr(estimator, 'score'):
            return _PassthroughScorer(estimator)
        elif allow_none:
            return None
        else:
            raise TypeError("If no scoring is specified, the estimator passed should have a 'score' method. The estimator %r does not." % estimator)