"""
Helper classes and functions used internally to compute label quality scores in multi-label classification.
"""
from enum import Enum
from typing import Callable, Dict, Optional, Union
import numpy as np
from sklearn.model_selection import cross_val_predict
from cleanlab.internal.multilabel_utils import _is_multilabel, stack_complement
from cleanlab.internal.label_quality_utils import _subtract_confident_thresholds
from cleanlab.internal.numerics import softmax
from cleanlab.rank import get_self_confidence_for_each_label, get_normalized_margin_for_each_label, get_confidence_weighted_entropy_for_each_label

class _Wrapper:
    """Helper class for wrapping callable functions as attributes of an Enum instead of
    setting them as methods of the Enum class.


    This class is only intended to be used internally for the ClassLabelScorer or
    other cases where functions are used for enumeration values.
    """

    def __init__(self, f: Callable) -> None:
        if False:
            print('Hello World!')
        self.f = f

    def __call__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self.f(*args, **kwargs)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return self.f.__name__

class ClassLabelScorer(Enum):
    """Enum for the different methods to compute label quality scores."""
    SELF_CONFIDENCE = _Wrapper(get_self_confidence_for_each_label)
    'Returns the self-confidence label-quality score for each datapoint.\n\n    See also\n    --------\n    cleanlab.rank.get_self_confidence_for_each_label\n    '
    NORMALIZED_MARGIN = _Wrapper(get_normalized_margin_for_each_label)
    'Returns the "normalized margin" label-quality score for each datapoint.\n\n    See also\n    --------\n    cleanlab.rank.get_normalized_margin_for_each_label\n    '
    CONFIDENCE_WEIGHTED_ENTROPY = _Wrapper(get_confidence_weighted_entropy_for_each_label)
    'Returns the "confidence weighted entropy" label-quality score for each datapoint.\n\n    See also\n    --------\n    cleanlab.rank.get_confidence_weighted_entropy_for_each_label\n    '

    def __call__(self, labels: np.ndarray, pred_probs: np.ndarray, **kwargs) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        'Returns the label-quality scores for each datapoint based on the given labels and predicted probabilities.\n\n        See the documentation for each method for more details.\n\n        Example\n        -------\n        >>> import numpy as np\n        >>> from cleanlab.internal.multilabel_scorer import ClassLabelScorer\n        >>> labels = np.array([0, 0, 0, 1, 1, 1])\n        >>> pred_probs = np.array([\n        ...     [0.9, 0.1],\n        ...     [0.8, 0.2],\n        ...     [0.7, 0.3],\n        ...     [0.2, 0.8],\n        ...     [0.75, 0.25],\n        ...     [0.1, 0.9],\n        ... ])\n        >>> ClassLabelScorer.SELF_CONFIDENCE(labels, pred_probs)\n        array([0.9 , 0.8 , 0.7 , 0.8 , 0.25, 0.9 ])\n        '
        pred_probs = self._adjust_pred_probs(labels, pred_probs, **kwargs)
        return self.value(labels, pred_probs)

    def _adjust_pred_probs(self, labels: np.ndarray, pred_probs: np.ndarray, **kwargs) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        'Returns adjusted predicted probabilities by subtracting the class confident thresholds and renormalizing.\n\n        This is used to adjust the predicted probabilities for the SELF_CONFIDENCE and NORMALIZED_MARGIN methods.\n        '
        if kwargs.get('adjust_pred_probs', False) is True:
            if self == ClassLabelScorer.CONFIDENCE_WEIGHTED_ENTROPY:
                raise ValueError(f'adjust_pred_probs is not currently supported for {self}.')
            pred_probs = _subtract_confident_thresholds(labels, pred_probs)
        return pred_probs

    @classmethod
    def from_str(cls, method: str) -> 'ClassLabelScorer':
        if False:
            print('Hello World!')
        'Constructs an instance of the ClassLabelScorer enum based on the given method name.\n\n        Parameters\n        ----------\n        method:\n            The name of the scoring method to use.\n\n        Returns\n        -------\n        scorer:\n            An instance of the ClassLabelScorer enum.\n\n        Raises\n        ------\n        ValueError:\n            If the given method name is not a valid method name.\n            It must be one of the following: "self_confidence", "normalized_margin", or "confidence_weighted_entropy".\n\n        Example\n        -------\n        >>> from cleanlab.internal.multilabel_scorer import ClassLabelScorer\n        >>> ClassLabelScorer.from_str("self_confidence")\n        <ClassLabelScorer.SELF_CONFIDENCE: get_self_confidence_for_each_label>\n        '
        try:
            return cls[method.upper()]
        except KeyError:
            raise ValueError(f'Invalid method name: {method}')

def exponential_moving_average(s: np.ndarray, *, alpha: Optional[float]=None, axis: int=1, **_) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    'Exponential moving average (EMA) score aggregation function.\n\n    For a score vector s = (s_1, ..., s_K) with K scores, the values\n    are sorted in *descending* order and the exponential moving average\n    of the last score is calculated, denoted as EMA_K according to the\n    note below.\n\n    Note\n    ----\n\n    The recursive formula for the EMA at step :math:`t = 2, ..., K` is:\n\n    .. math::\n\n        \\text{EMA}_t = \\alpha \\cdot s_t + (1 - \\alpha) \\cdot \\text{EMA}_{t-1}, \\qquad 0 \\leq \\alpha \\leq 1\n\n    We set :math:`\\text{EMA}_1 = s_1` as the largest score in the sorted vector s.\n\n    :math:`\\alpha` is the "forgetting factor" that gives more weight to the\n    most recent scores, and successively less weight to the previous scores.\n\n    Parameters\n    ----------\n    s :\n        Scores to be transformed.\n\n    alpha :\n        Discount factor that determines the weight of the previous EMA score.\n        Higher alpha means that the previous EMA score has a lower weight while\n        the current score has a higher weight.\n\n        Its value must be in the interval [0, 1].\n\n        If alpha is None, it is set to 2 / (K + 1) where K is the number of scores.\n\n    axis :\n        Axis along which the scores are sorted.\n\n    Returns\n    -------\n    s_ema :\n        Exponential moving average score.\n\n    Examples\n    --------\n    >>> from cleanlab.internal.multilabel_scorer import exponential_moving_average\n    >>> import numpy as np\n    >>> s = np.array([[0.1, 0.2, 0.3]])\n    >>> exponential_moving_average(s, alpha=0.5)\n    np.array([0.175])\n    '
    K = s.shape[1]
    s_sorted = np.fliplr(np.sort(s, axis=axis))
    if alpha is None:
        alpha = float(2 / (K + 1))
    if not 0 <= alpha <= 1:
        raise ValueError(f'alpha must be in the interval [0, 1], got {alpha}')
    s_T = s_sorted.T
    (s_ema, s_next) = (s_T[0], s_T[1:])
    for s_i in s_next:
        s_ema = alpha * s_i + (1 - alpha) * s_ema
    return s_ema

def softmin(s: np.ndarray, *, temperature: float=0.1, axis: int=1, **_) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Softmin score aggregation function.\n\n    Parameters\n    ----------\n    s :\n        Input array.\n\n    temperature :\n        Temperature parameter. Too small values may cause numerical underflow and NaN scores.\n\n    axis :\n        Axis along which to apply the function.\n\n    Returns\n    -------\n        Softmin score.\n    '
    return np.einsum('ij,ij->i', s, softmax(x=1 - s, temperature=temperature, axis=axis, shift=True))

class Aggregator:
    """Helper class for aggregating the label quality scores for each class into a single score for each datapoint.

    Parameters
    ----------
    method:
        The method to compute the label quality scores for each class.
        If passed as a callable, your function should take in a 1D array of K scores and return a single aggregated score.
        See :py:func:`exponential_moving_average <cleanlab.internal.multilabel_scorer.exponential_moving_average>` for an example of such a function.
        Alternatively, this can be a str value to specify a built-in function, possible values are the keys of the ``Aggregator``'s `possible_methods` attribute.

    kwargs:
        Additional keyword arguments to pass to the aggregation function when it is called.
    """
    possible_methods: Dict[str, Callable[..., np.ndarray]] = {'exponential_moving_average': exponential_moving_average, 'softmin': softmin}

    def __init__(self, method: Union[str, Callable], **kwargs):
        if False:
            print('Hello World!')
        if isinstance(method, str):
            if method in self.possible_methods:
                method = self.possible_methods[method]
            else:
                raise ValueError(f"Invalid aggregation method specified: '{method}', must be one of the following: {list(self.possible_methods.keys())}")
        self._validate_method(method)
        self.method = method
        self.kwargs = kwargs

    @staticmethod
    def _validate_method(method) -> None:
        if False:
            print('Hello World!')
        if not callable(method):
            raise TypeError(f'Expected callable method, got {type(method)}')

    @staticmethod
    def _validate_scores(scores: np.ndarray) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not (isinstance(scores, np.ndarray) and scores.ndim == 2):
            raise ValueError(f'Expected 2D array for scores, got {type(scores)} with shape {scores.shape}')

    def __call__(self, scores: np.ndarray, **kwargs) -> np.ndarray:
        if False:
            print('Hello World!')
        'Returns the label quality scores for each datapoint based on the given label quality scores for each class.\n\n        Parameters\n        ----------\n        scores:\n            The label quality scores for each class.\n\n        Returns\n        -------\n        aggregated_scores:\n            A single label quality score for each datapoint.\n        '
        self._validate_scores(scores)
        kwargs['axis'] = 1
        updated_kwargs = {**self.kwargs, **kwargs}
        return self.method(scores, **updated_kwargs)

    def __repr__(self):
        if False:
            return 10
        return f'Aggregator(method={self.method.__name__}, kwargs={self.kwargs})'

class MultilabelScorer:
    """Aggregates label quality scores across different classes to produce one score per example in multi-label classification tasks.

    Parameters
    ----------
    base_scorer:
        The method to compute the label quality scores for each class.

        See the documentation for the ClassLabelScorer enum for more details.

    aggregator:
        The method to aggregate the label quality scores for each class into a single score for each datapoint.

        Defaults to the EMA (exponential moving average) aggregator with forgetting factor ``alpha=0.8``.

        See the documentation for the Aggregator class for more details.

        See also
        --------
        exponential_moving_average

    strict:
        Flag for performing strict validation of the input data.
    """

    def __init__(self, base_scorer: ClassLabelScorer=ClassLabelScorer.SELF_CONFIDENCE, aggregator: Union[Aggregator, Callable]=Aggregator(exponential_moving_average, alpha=0.8), *, strict: bool=True):
        if False:
            for i in range(10):
                print('nop')
        self.base_scorer = base_scorer
        if not isinstance(aggregator, Aggregator):
            self.aggregator = Aggregator(aggregator)
        else:
            self.aggregator = aggregator
        self.strict = strict

    def __call__(self, labels: np.ndarray, pred_probs: np.ndarray, base_scorer_kwargs: Optional[dict]=None, **aggregator_kwargs) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Computes a quality score for each label in a multi-label classification problem\n        based on out-of-sample predicted probabilities.\n        For each example, the label quality scores for each class are aggregated into a single overall label quality score.\n\n        Parameters\n        ----------\n        labels:\n            A 2D array of shape (n_samples, n_labels) with binary labels.\n\n        pred_probs:\n            A 2D array of shape (n_samples, n_labels) with predicted probabilities.\n\n        kwargs:\n            Additional keyword arguments to pass to the base_scorer and the aggregator.\n\n        base_scorer_kwargs:\n             Keyword arguments to pass to the base_scorer\n\n         aggregator_kwargs:\n             Additional keyword arguments to pass to the aggregator.\n\n        Returns\n        -------\n        scores:\n            A 1D array of shape (n_samples,) with the quality scores for each datapoint.\n\n        Examples\n        --------\n        >>> from cleanlab.internal.multilabel_scorer import MultilabelScorer, ClassLabelScorer\n        >>> import numpy as np\n        >>> labels = np.array([[0, 1, 0], [1, 0, 1]])\n        >>> pred_probs = np.array([[0.1, 0.9, 0.1], [0.4, 0.1, 0.9]])\n        >>> scorer = MultilabelScorer()\n        >>> scores = scorer(labels, pred_probs)\n        >>> scores\n        array([0.9, 0.5])\n\n        >>> scorer = MultilabelScorer(\n        ...     base_scorer = ClassLabelScorer.NORMALIZED_MARGIN,\n        ...     aggregator = np.min,  # Use the "worst" label quality score for each example.\n        ... )\n        >>> scores = scorer(labels, pred_probs)\n        >>> scores\n        array([0.9, 0.4])\n        '
        if self.strict:
            self._validate_labels_and_pred_probs(labels, pred_probs)
        scores = self.get_class_label_quality_scores(labels, pred_probs, base_scorer_kwargs)
        return self.aggregate(scores, **aggregator_kwargs)

    def aggregate(self, class_label_quality_scores: np.ndarray, **kwargs) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        'Aggregates the label quality scores for each class into a single overall label quality score for each example.\n\n        Parameters\n        ----------\n        class_label_quality_scores:\n            A 2D array of shape (n_samples, n_labels) with the label quality scores for each class.\n\n            See also\n            --------\n            get_class_label_quality_scores\n\n        kwargs:\n            Additional keyword arguments to pass to the aggregator.\n\n        Returns\n        -------\n        scores:\n            A 1D array of shape (n_samples,) with the quality scores for each datapoint.\n\n        Examples\n        --------\n        >>> from cleanlab.internal.multilabel_scorer import MultilabelScorer\n        >>> import numpy as np\n        >>> class_label_quality_scores = np.array([[0.9, 0.9, 0.3],[0.4, 0.9, 0.6]])\n        >>> scorer = MultilabelScorer() # Use the default aggregator (exponential moving average) with default parameters.\n        >>> scores = scorer.aggregate(class_label_quality_scores)\n        >>> scores\n        array([0.42, 0.452])\n        >>> new_scores = scorer.aggregate(class_label_quality_scores, alpha=0.5) # Use the default aggregator with custom parameters.\n        >>> new_scores\n        array([0.6, 0.575])\n\n        Warning\n        -------\n        Make sure that keyword arguments correspond to the aggregation function used.\n        I.e. the ``exponential_moving_average`` function supports an ``alpha`` keyword argument, but ``np.min`` does not.\n        '
        return self.aggregator(class_label_quality_scores, **kwargs)

    def get_class_label_quality_scores(self, labels: np.ndarray, pred_probs: np.ndarray, base_scorer_kwargs: Optional[dict]=None) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        'Computes separate label quality scores for each class.\n\n        Parameters\n        ----------\n        labels:\n            A 2D array of shape (n_samples, n_labels) with binary labels.\n\n        pred_probs:\n            A 2D array of shape (n_samples, n_labels) with predicted probabilities.\n\n        base_scorer_kwargs:\n            Keyword arguments to pass to the base scoring-function.\n\n        Returns\n        -------\n        class_label_quality_scores:\n            A 2D array of shape (n_samples, n_labels) with the quality scores for each label.\n\n        Examples\n        --------\n        >>> from cleanlab.internal.multilabel_scorer import MultilabelScorer\n        >>> import numpy as np\n        >>> labels = np.array([[0, 1, 0], [1, 0, 1]])\n        >>> pred_probs = np.array([[0.1, 0.9, 0.7], [0.4, 0.1, 0.6]])\n        >>> scorer = MultilabelScorer() # Use the default base scorer (SELF_CONFIDENCE)\n        >>> class_label_quality_scores = scorer.get_label_quality_scores_per_class(labels, pred_probs)\n        >>> class_label_quality_scores\n        array([[0.9, 0.9, 0.3],\n               [0.4, 0.9, 0.6]])\n        '
        class_label_quality_scores = np.zeros(shape=labels.shape)
        if base_scorer_kwargs is None:
            base_scorer_kwargs = {}
        for (i, (label_i, pred_prob_i)) in enumerate(zip(labels.T, pred_probs.T)):
            pred_prob_i_two_columns = stack_complement(pred_prob_i)
            class_label_quality_scores[:, i] = self.base_scorer(label_i, pred_prob_i_two_columns, **base_scorer_kwargs)
        return class_label_quality_scores

    @staticmethod
    def _validate_labels_and_pred_probs(labels: np.ndarray, pred_probs: np.ndarray) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Checks that (multi-)labels are in the proper binary indicator format and that\n        they are compatible with the predicted probabilities.\n        '
        if not isinstance(labels, np.ndarray):
            raise TypeError('Labels must be a numpy array.')
        if not _is_multilabel(labels):
            raise ValueError('Labels must be in multi-label format.')
        if labels.shape != pred_probs.shape:
            raise ValueError('Labels and predicted probabilities must have the same shape.')

def get_label_quality_scores(labels, pred_probs, *, method: MultilabelScorer=MultilabelScorer(), base_scorer_kwargs: Optional[dict]=None, **aggregator_kwargs) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Computes a quality score for each label in a multi-label classification problem\n    based on out-of-sample predicted probabilities.\n\n    Parameters\n    ----------\n    labels:\n        A 2D array of shape (N, K) with binary labels.\n\n    pred_probs:\n        A 2D array of shape (N, K) with predicted probabilities.\n\n    method:\n        A scoring+aggregation method for computing the label quality scores of examples in a multi-label classification setting.\n\n    base_scorer_kwargs:\n        Keyword arguments to pass to the class-label scorer.\n\n    aggregator_kwargs:\n        Additional keyword arguments to pass to the aggregator.\n\n    Returns\n    -------\n    scores:\n        A 1D array of shape (N,) with the quality scores for each datapoint.\n\n    Examples\n    --------\n    >>> import cleanlab.internal.multilabel_scorer as ml_scorer\n    >>> import numpy as np\n    >>> labels = np.array([[0, 1, 0], [1, 0, 1]])\n    >>> pred_probs = np.array([[0.1, 0.9, 0.1], [0.4, 0.1, 0.9]])\n    >>> scores = ml_scorer.get_label_quality_scores(labels, pred_probs, method=ml_scorer.MultilabelScorer())\n    >>> scores\n    array([0.9, 0.5])\n\n    See also\n    --------\n    MultilabelScorer:\n        See the documentation for the MultilabelScorer class for more examples of scoring methods and aggregation methods.\n    '
    return method(labels, pred_probs, base_scorer_kwargs=base_scorer_kwargs, **aggregator_kwargs)

def multilabel_py(y: np.ndarray) -> np.ndarray:
    if False:
        return 10
    'Compute the prior probability of each label in a multi-label classification problem.\n\n    Parameters\n    ----------\n    y :\n        A 2d array of binarized multi-labels of shape (N, K) where N is the number of samples and K is the number of classes.\n\n    Returns\n    -------\n    py :\n        A 2d array of prior probabilities of shape (K,2) where the first column is the probability of the label being 0\n        and the second column is the probability of the label being 1 for each class.\n\n    Examples\n    --------\n    >>> from cleanlab.internal.multilabel_scorer import multilabel_py\n    >>> import numpy as np\n    >>> y = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])\n    >>> multilabel_py(y)\n    array([[0.5, 0.5],\n           [0.5, 0.5]])\n    >>> y = np.array([[0, 0], [0, 1], [1, 0], [1, 0], [1, 0]])\n    >>> multilabel_py(y)\n    array([[0.4, 0.6],\n           [0.8, 0.2]])\n    '

    def compute_class_py(y_slice: np.ndarray) -> np.ndarray:
        if False:
            print('Hello World!')
        assert y_slice.ndim == 1
        (unique_values, counts) = np.unique(y_slice, axis=0, return_counts=True)
        N = y_slice.shape[0]
        if len(unique_values) == 1:
            counts = np.array([0, N] if unique_values[0] == 1 else [N, 0])
        return counts / N
    py = np.apply_along_axis(compute_class_py, axis=1, arr=y.T)
    return py

def _get_split_generator(labels, cv):
    if False:
        for i in range(10):
            print('nop')
    unique_labels = np.unique(labels, axis=0)
    label_to_index = {tuple(label): i for (i, label) in enumerate(unique_labels)}
    multilabel_ids = np.array([label_to_index[tuple(label)] for label in labels])
    split_generator = cv.split(X=multilabel_ids, y=multilabel_ids)
    return split_generator

def get_cross_validated_multilabel_pred_probs(X, labels: np.ndarray, *, clf, cv) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    'Get predicted probabilities for a multi-label classifier via cross-validation.\n\n    Note\n    ----\n    The labels are reformatted to a "multi-class" format internally to support a wider range of cross-validation strategies.\n    If you have a multi-label dataset with `K` classes, the labels are reformatted to a "multi-class" format with up to `2**K` classes\n    (i.e. the number of possible class-assignment configurations).\n    It is unlikely that you\'ll all `2**K` configurations in your dataset.\n\n    Parameters\n    ----------\n    X :\n        A 2d array of features of shape (N, M) where N is the number of samples and M is the number of features.\n\n    labels :\n        A 2d array of binarized multi-labels of shape (N, K) where N is the number of samples and K is the number of classes.\n\n    clf :\n        A multi-label classifier with a ``predict_proba`` method.\n\n    cv :\n        A cross-validation splitter with a ``split`` method that returns a generator of train/test indices.\n\n    Returns\n    -------\n    pred_probs :\n        A 2d array of predicted probabilities of shape (N, K) where N is the number of samples and K is the number of classes.\n\n        Note\n        ----\n        The predicted probabilities are not expected to sum to 1 for each sample in the case of multi-label classification.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from sklearn.model_selection import KFold\n    >>> from sklearn.multiclass import OneVsRestClassifier\n    >>> from sklearn.ensemble import RandomForestClassifier\n    >>> from cleanlab.internal.multilabel_scorer import get_cross_validated_multilabel_pred_probs\n    >>> np.random.seed(0)\n    >>> X = np.random.rand(16, 2)\n    >>> labels = np.random.randint(0, 2, size=(16, 2))\n    >>> clf = OneVsRestClassifier(RandomForestClassifier())\n    >>> cv = KFold(n_splits=2)\n    >>> get_cross_validated_multilabel_pred_probs(X, labels, clf=clf, cv=cv)\n    '
    split_generator = _get_split_generator(labels, cv)
    pred_probs = cross_val_predict(clf, X, labels, cv=split_generator, method='predict_proba')
    return pred_probs