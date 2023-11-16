"""
Methods for finding out-of-distribution examples in a dataset via scores that quantify how atypical each example is compared to the others.

The underlying algorithms are described in `this paper <https://arxiv.org/abs/2207.03061>`_.
"""
import warnings
import numpy as np
from cleanlab.count import get_confident_thresholds
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import NotFittedError
from typing import Optional, Union, Tuple, Dict, cast
from cleanlab.internal.label_quality_utils import _subtract_confident_thresholds, get_normalized_entropy
from cleanlab.internal.numerics import softmax
from cleanlab.internal.outlier import transform_distances_to_scores
from cleanlab.internal.validation import assert_valid_inputs, labels_to_array
from cleanlab.typing import LabelLike

class OutOfDistribution:
    """
    Provides scores to detect Out Of Distribution (OOD) examples that are outliers in a dataset.

    Each example's OOD score lies in [0,1] with smaller values indicating examples that are less typical under the data distribution.
    OOD scores may be estimated from either: numeric feature embeddings or predicted probabilities from a trained classifier.

    To get indices of examples that are the most severe outliers, call `~cleanlab.rank.find_top_issues` function on the returned OOD scores.

    Parameters
    ----------
    params : dict, default = {}
     Optional keyword arguments to control how this estimator is fit. Effect of arguments passed in depends on if
     `OutOfDistribution` estimator will rely on `features` or `pred_probs`. These are stored as an instance attribute `self.params`.

     If `features` is passed in during ``fit()``, `params` could contain following keys:
       *  knn: sklearn.neighbors.NearestNeighbors, default = None
             Instantiated ``NearestNeighbors`` object that's been fitted on a dataset in the same feature space.
             Note that the distance metric and `n_neighbors` is specified when instantiating this class.
             You can also pass in a subclass of ``sklearn.neighbors.NearestNeighbors`` which allows you to use faster
             approximate neighbor libraries as long as you wrap them behind the same sklearn API.
             If you specify ``knn`` here, there is no need to later call ``fit()`` before calling ``score()``.
             If ``knn = None``, then by default: ``knn = sklearn.neighbors.NearestNeighbors(n_neighbors=k, metric=dist_metric).fit(features)``
             where ``dist_metric == "cosine"`` if ``dim(features) > 3`` or ``dist_metric == "euclidean"`` otherwise.
             See: https://scikit-learn.org/stable/modules/neighbors.html
       *  k : int, default=None
             Optional number of neighbors to use when calculating outlier score (average distance to neighbors).
             If `k` is not provided, then by default ``k = knn.n_neighbors`` or ``k = 10`` if ``knn is None``.
             If an existing ``knn`` object is provided, you can still specify that outlier scores should use
             a different value of `k` than originally used in the ``knn``,
             as long as your specified value of `k` is smaller than the value originally used in ``knn``.
       *  t : int, default=1
             Optional hyperparameter only for advanced users.
             Controls transformation of distances between examples into similarity scores that lie in [0,1].
             The transformation applied to distances `x` is ``exp(-x*t)``.
             If you find your scores are all too close to 1, consider increasing `t`,
             although the relative scores of examples will still have the same ranking across the dataset.

     If `pred_probs` is passed in during ``fit()``, `params` could contain following keys:
       *  confident_thresholds: np.ndarray, default = None
             An array of shape ``(K, )`` where K is the number of classes.
             Confident threshold for a class j is the expected (average) "self-confidence" for that class.
             If you specify `confident_thresholds` here, there is no need to later call ``fit()`` before calling ``score()``.
       *  adjust_pred_probs : bool, True
             If True, account for class imbalance by adjusting predicted probabilities
             via subtraction of class confident thresholds and renormalization.
             If False, you do not have to pass in `labels` later to fit this OOD estimator.
             See `Northcutt et al., 2021 <https://jair.org/index.php/jair/article/view/12125>`_.
       *  method : {"entropy", "least_confidence"}, default="entropy"
             Method to use when computing outlier scores based on `pred_probs`.
             Letting length-K vector ``P = pred_probs[i]`` denote the given predicted class-probabilities
             for the i-th example in dataset, its outlier score can either be:

             - ``'entropy'``: ``1 - sum_{j} P[j] * log(P[j]) / log(K)``
             - ``'least_confidence'``: ``max(P)`` (equivalent to Maximum Softmax Probability method from the OOD detection literature)
             - ``gen``: Generalized ENtropy score from the paper of Liu, Lochman, and Zach (https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_GEN_Pushing_the_Limits_of_Softmax-Based_Out-of-Distribution_Detection_CVPR_2023_paper.pdf)

    """
    OUTLIER_PARAMS = {'k', 't', 'knn'}
    OOD_PARAMS = {'confident_thresholds', 'adjust_pred_probs', 'method', 'M', 'gamma'}
    DEFAULT_PARAM_DICT: Dict[str, Union[str, int, float, None, np.ndarray]] = {'k': None, 't': 1, 'knn': None, 'method': 'entropy', 'adjust_pred_probs': True, 'confident_thresholds': None, 'M': 100, 'gamma': 0.1}

    def __init__(self, params: Optional[dict]=None) -> None:
        if False:
            while True:
                i = 10
        self._assert_valid_params(params, self.DEFAULT_PARAM_DICT)
        self.params = self.DEFAULT_PARAM_DICT.copy()
        if params is not None:
            self.params.update(params)
        if self.params['adjust_pred_probs'] and self.params['method'] == 'gen':
            print("CAUTION: GEN method is not recommended for use with adjusted pred_probs. To use GEN, we recommend setting: params['adjust_pred_probs'] = False")

    def fit_score(self, *, features: Optional[np.ndarray]=None, pred_probs: Optional[np.ndarray]=None, labels: Optional[np.ndarray]=None, verbose: bool=True) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        "\n        Fits this estimator to a given dataset and returns out-of-distribution scores for the same dataset.\n\n        Scores lie in [0,1] with smaller values indicating examples that are less typical under the dataset\n        distribution (values near 0 indicate outliers). Exactly one of `features` or `pred_probs` needs to be passed\n        in to calculate scores.\n\n        If `features` are passed in a ``NearestNeighbors`` object is fit. If `pred_probs` and 'labels' are passed in a\n        `confident_thresholds` ``np.ndarray`` is fit. For details see `~cleanlab.outlier.OutOfDistribution.fit`.\n\n        Parameters\n        ----------\n        features : np.ndarray, optional\n          Feature array of shape ``(N, M)``, where N is the number of examples and M is the number of features used to represent each example.\n          For details, `features` in the same format expected by the `~cleanlab.outlier.OutOfDistribution.fit` function.\n\n        pred_probs : np.ndarray, optional\n          An array of shape ``(N, K)`` of predicted class probabilities output by a trained classifier.\n          For details, `pred_probs` in the same format expected by the `~cleanlab.outlier.OutOfDistribution.fit` function.\n\n        labels : array_like, optional\n          A discrete array of given class labels for the data of shape ``(N,)``.\n          For details, `labels` in the same format expected by the `~cleanlab.outlier.OutOfDistribution.fit` function.\n\n        verbose : bool, default = True\n          Set to ``False`` to suppress all print statements.\n\n        Returns\n        -------\n        scores : np.ndarray\n          If `features` are passed in, `ood_features_scores` are returned.\n          If `pred_probs` are passed in, `ood_predictions_scores` are returned.\n          For details see return of `~cleanlab.outlier.OutOfDistribution.scores` function.\n\n        "
        scores = self._shared_fit(features=features, pred_probs=pred_probs, labels=labels, verbose=verbose)
        if scores is None:
            scores = self.score(features=features, pred_probs=pred_probs)
        return scores

    def fit(self, *, features: Optional[np.ndarray]=None, pred_probs: Optional[np.ndarray]=None, labels: Optional[LabelLike]=None, verbose: bool=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fits this estimator to a given dataset.\n\n        One of `features` or `pred_probs` must be specified.\n\n        If `features` are passed in, a ``NearestNeighbors`` object is fit.\n        If `pred_probs` and \'labels\' are passed in, a `confident_thresholds` ``np.ndarray`` is fit.\n        For details see `~cleanlab.outlier.OutOfDistribution` documentation.\n\n        Parameters\n        ----------\n        features : np.ndarray, optional\n          Feature array of shape ``(N, M)``, where N is the number of examples and M is the number of features used to represent each example.\n          All features should be **numeric**. For less structured data (e.g. images, text, categorical values, ...), you should provide\n          vector embeddings to represent each example (e.g. extracted from some pretrained neural network).\n\n        pred_probs : np.ndarray, optional\n           An array of shape ``(N, K)`` of model-predicted probabilities,\n          ``P(label=k|x)``. Each row of this matrix corresponds\n          to an example `x` and contains the model-predicted probabilities that\n          `x` belongs to each possible class, for each of the K classes. The\n          columns must be ordered such that these probabilities correspond to\n          class 0, 1, ..., K-1.\n\n        labels : array_like, optional\n          A discrete vector of given labels for the data of shape ``(N,)``. Supported `array_like` types include: ``np.ndarray`` or ``list``.\n          *Format requirements*: for dataset with K classes, labels must be in 0, 1, ..., K-1.\n          All the classes (0, 1, ..., and K-1) MUST be present in ``labels``, such that: ``len(set(labels)) == pred_probs.shape[1]``\n          If ``params["adjust_confident_thresholds"]`` was previously set to ``False``, you do not have to pass in `labels`.\n          Note: multi-label classification is not supported by this method, each example must belong to a single class, e.g. ``labels = np.ndarray([1,0,2,1,1,0...])``.\n\n        verbose : bool, default = True\n          Set to ``False`` to suppress all print statements.\n\n        '
        _ = self._shared_fit(features=features, pred_probs=pred_probs, labels=labels, verbose=verbose)

    def score(self, *, features: Optional[np.ndarray]=None, pred_probs: Optional[np.ndarray]=None) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        "\n        Use fitted estimator and passed in `features` or `pred_probs` to calculate out-of-distribution scores for a dataset.\n\n        Score for each example corresponds to the likelihood this example stems from the same distribution as the dataset previously specified in ``fit()`` (i.e. is not an outlier).\n\n        If `features` are passed, returns OOD score for each example based on its feature values.\n        If `pred_probs` are passed, returns OOD score for each example based on classifier's probabilistic predictions.\n        You may have to previously call ``fit()`` or call ``fit_score()`` instead.\n\n        Parameters\n        ----------\n        features : np.ndarray, optional\n          Feature array of shape ``(N, M)``, where N is the number of examples and M is the number of features used to represent each example.\n          For details, see `features` in `~cleanlab.outlier.OutOfDistribution.fit` function.\n\n        pred_probs : np.ndarray, optional\n          An array of shape ``(N, K)``  of predicted class probabilities output by a trained classifier.\n          For details, see `pred_probs` in `~cleanlab.outlier.OutOfDistribution.fit` function.\n\n        Returns\n        -------\n        scores : np.ndarray\n          Scores lie in [0,1] with smaller values indicating examples that are less typical under the dataset distribution\n          (values near 0 indicate outliers).\n\n          If `features` are passed, `ood_features_scores` are returned.\n          The score is based on the average distance between the example and its K nearest neighbors in the dataset\n          (in feature space).\n\n          If `pred_probs` are passed, `ood_predictions_scores` are returned.\n          The score is based on the uncertainty in the classifier's predicted probabilities.\n        "
        self._assert_valid_inputs(features, pred_probs)
        if features is not None:
            if self.params['knn'] is None:
                raise ValueError('OOD estimator needs to be fit on features first. Call `fit()` or `fit_scores()` before this function.')
            (scores, _) = _get_ood_features_scores(features, **self._get_params(self.OUTLIER_PARAMS))
        if pred_probs is not None:
            if self.params['confident_thresholds'] is None and self.params['adjust_pred_probs']:
                raise ValueError("OOD estimator needs to be fit on pred_probs first since params['adjust_pred_probs']=True. Call `fit()` or `fit_scores()` before this function.")
            (scores, _) = _get_ood_predictions_scores(pred_probs, **self._get_params(self.OOD_PARAMS))
        return scores

    def _get_params(self, param_keys) -> dict:
        if False:
            i = 10
            return i + 15
        'Get function specific dictionary of parameters (i.e. only those in param_keys).'
        return {k: v for (k, v) in self.params.items() if k in param_keys}

    @staticmethod
    def _assert_valid_params(params, param_keys):
        if False:
            i = 10
            return i + 15
        'Validate passed in params and get list of parameters in param that are not in param_keys.'
        if params is not None:
            wrong_params = list(set(params.keys()).difference(set(param_keys)))
            if len(wrong_params) > 0:
                raise ValueError(f'Passed in params dict can only contain {param_keys}. Remove {wrong_params} from params dict.')

    @staticmethod
    def _assert_valid_inputs(features, pred_probs):
        if False:
            return 10
        'Check whether features and pred_prob inputs are valid, throw error if not.'
        if features is None and pred_probs is None:
            raise ValueError('Not enough information to compute scores. Pass in either features or pred_probs.')
        if features is not None and pred_probs is not None:
            raise ValueError('Cannot fit to OOD Estimator to both features and pred_probs. Pass in either one or the other.')
        if features is not None and len(features.shape) != 2:
            raise ValueError('Feature array needs to be of shape (N, M), where N is the number of examples and M is the number of features used to represent each example. ')

    def _shared_fit(self, *, features: Optional[np.ndarray]=None, pred_probs: Optional[np.ndarray]=None, labels: Optional[LabelLike]=None, verbose: bool=True) -> Optional[np.ndarray]:
        if False:
            print('Hello World!')
        '\n        Shared fit functionality between ``fit()`` and ``fit_score()``.\n\n        For details, refer to `~cleanlab.outlier.OutOfDistribution.fit`\n        or `~cleanlab.outlier.OutOfDistribution.fit_score`.\n        '
        self._assert_valid_inputs(features, pred_probs)
        scores = None
        if features is not None:
            if self.params['knn'] is not None:
                warnings.warn('A KNN estimator has previously already been fit, call score() to apply it to data, or create a new OutOfDistribution object to fit a different estimator.', UserWarning)
            else:
                if verbose:
                    print('Fitting OOD estimator based on provided features ...')
                (scores, knn) = _get_ood_features_scores(features, **self._get_params(self.OUTLIER_PARAMS))
                self.params['knn'] = knn
        if pred_probs is not None:
            if self.params['confident_thresholds'] is not None:
                warnings.warn('Confident thresholds have previously already been fit, call score() to apply them to data, or create a new OutOfDistribution object to fit a different estimator.', UserWarning)
            else:
                if verbose:
                    print('Fitting OOD estimator based on provided pred_probs ...')
                (scores, confident_thresholds) = _get_ood_predictions_scores(pred_probs, labels=labels, **self._get_params(self.OOD_PARAMS))
                if confident_thresholds is None:
                    warnings.warn('No estimates need to be be fit under the provided params, so you could directly call score() as an alternative.', UserWarning)
                else:
                    self.params['confident_thresholds'] = confident_thresholds
        return scores

def _get_ood_features_scores(features: Optional[np.ndarray]=None, knn: Optional[NearestNeighbors]=None, k: Optional[int]=None, t: int=1) -> Tuple[np.ndarray, Optional[NearestNeighbors]]:
    if False:
        return 10
    '\n    Return outlier score based on feature values using `k` nearest neighbors.\n\n    The outlier score for each example is computed inversely proportional to\n    the average distance between this example and its K nearest neighbors (in feature space).\n\n    Parameters\n    ----------\n    features : np.ndarray\n      Feature array of shape ``(N, M)``, where N is the number of examples and M is the number of features used to represent each example.\n      For details, `features` in the same format expected by the `~cleanlab.outlier.OutOfDistribution.fit` function.\n\n    knn : sklearn.neighbors.NearestNeighbors, default = None\n      For details, see key `knn` in the params dict arg of `~cleanlab.outlier.OutOfDistribution`.\n\n    k : int, default=None\n      Optional number of neighbors to use when calculating outlier score (average distance to neighbors).\n      For details, see key `k` in the params dict arg of `~cleanlab.outlier.OutOfDistribution`.\n\n    t : int, default=1\n      Controls transformation of distances between examples into similarity scores that lie in [0,1].\n      For details, see key `t` in the params dict arg of `~cleanlab.outlier.OutOfDistribution`.\n\n    Returns\n    -------\n    ood_features_scores : Tuple[np.ndarray, Optional[NearestNeighbors]]\n      Return a tuple whose first element is array of `ood_features_scores` and second is a `knn` Estimator object.\n    '
    DEFAULT_K = 10
    if knn is None:
        if features is None:
            raise ValueError('Both knn and features arguments cannot be None at the same time. Not enough information to compute outlier scores.')
        if k is None:
            k = DEFAULT_K
        if k > len(features):
            raise ValueError(f'Number of nearest neighbors k={k} cannot exceed the number of examples N={len(features)} passed into the estimator (knn).')
        if features.shape[1] > 3:
            metric = 'cosine'
        else:
            metric = 'euclidean'
        knn = NearestNeighbors(n_neighbors=k, metric=metric).fit(features)
        features = None
    elif k is None:
        k = knn.n_neighbors
    max_k = knn.n_neighbors
    if k > max_k:
        warnings.warn(f'Chosen k={k} cannot be greater than n_neighbors={max_k} which was used when fitting NearestNeighbors object! Value of k changed to k={max_k}.', UserWarning)
        k = max_k
    try:
        knn.kneighbors(features)
    except NotFittedError:
        knn.fit(features)
    (distances, _) = knn.kneighbors(features)
    ood_features_scores = transform_distances_to_scores(distances, cast(int, k), t)
    return (ood_features_scores, knn)

def _get_ood_predictions_scores(pred_probs: np.ndarray, *, labels: Optional[LabelLike]=None, confident_thresholds: Optional[np.ndarray]=None, adjust_pred_probs: bool=True, method: str='entropy', M: int=100, gamma: float=0.1) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if False:
        for i in range(10):
            print('nop')
    'Return an OOD (out of distribution) score for each example based on it pred_prob values.\n\n    Parameters\n    ----------\n    pred_probs : np.ndarray\n      An array of shape ``(N, K)`` of model-predicted probabilities,\n      `pred_probs` in the same format expected by the `~cleanlab.outlier.OutOfDistribution.fit` function.\n\n    confident_thresholds : np.ndarray, default = None\n      For details, see key `confident_thresholds` in the params dict arg of `~cleanlab.outlier.OutOfDistribution`.\n\n    labels : array_like, optional\n      `labels` in the same format expected by the `~cleanlab.outlier.OutOfDistribution.fit` function.\n\n    adjust_pred_probs : bool, True\n      Account for class imbalance in the label-quality scoring.\n      For details, see key `adjust_pred_probs` in the params dict arg of `~cleanlab.outlier.OutOfDistribution`.\n\n    method : {"entropy", "least_confidence", "gen"}, default="entropy"\n      Which method to use for computing outlier scores based on pred_probs.\n      For details see key `method` in the params dict arg of `~cleanlab.outlier.OutOfDistribution`.\n\n    M : int, default=100\n      For GEN method only. Hyperparameter that controls the number of top classes to consider when calculating OOD scores.\n\n    gamma : float, default=0.1\n      For GEN method only. Hyperparameter that controls the weight of the second term in the GEN score.\n\n\n    Returns\n    -------\n    ood_predictions_scores : Tuple[np.ndarray, Optional[np.ndarray]]\n      Returns a tuple. First element is array of `ood_predictions_scores` and second is an np.ndarray of `confident_thresholds` or None is \'confident_thresholds\' is not calculated.\n    '
    valid_methods = ('entropy', 'least_confidence', 'gen')
    if (confident_thresholds is not None or labels is not None) and (not adjust_pred_probs):
        warnings.warn("OOD scores are not adjusted with confident thresholds. If scores need to be adjusted set params['adjusted_pred_probs'] = True. Otherwise passing in confident_thresholds and/or labels does not change score calculation.", UserWarning)
    if adjust_pred_probs:
        if confident_thresholds is None:
            if labels is None:
                raise ValueError("Cannot calculate adjust_pred_probs without labels. Either pass in labels parameter or set params['adjusted_pred_probs'] = False. ")
            labels = labels_to_array(labels)
            assert_valid_inputs(X=None, y=labels, pred_probs=pred_probs, multi_label=False)
            confident_thresholds = get_confident_thresholds(labels, pred_probs, multi_label=False)
        pred_probs = _subtract_confident_thresholds(None, pred_probs, multi_label=False, confident_thresholds=confident_thresholds)
    if method == 'entropy':
        ood_predictions_scores = 1.0 - get_normalized_entropy(pred_probs)
    elif method == 'least_confidence':
        ood_predictions_scores = pred_probs.max(axis=1)
    elif method == 'gen':
        if pred_probs.shape[1] < M:
            warnings.warn(f"GEN with the default hyperparameter settings is intended for datasets with at least {M} classes. You can adjust params['M'] according to the number of classes in your dataset.", UserWarning)
        probs = softmax(pred_probs, axis=1)
        probs_sorted = np.sort(probs, axis=1)[:, -M:]
        ood_predictions_scores = 1 - np.sum(probs_sorted ** gamma * (1 - probs_sorted) ** gamma, axis=1) / M
    else:
        raise ValueError(f'\n            {method} is not a valid OOD scoring method!\n            Please choose a valid scoring_method: {valid_methods}\n            ')
    return (ood_predictions_scores, confident_thresholds)