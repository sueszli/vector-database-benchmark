"""Helper methods used internally for computing label quality scores."""
import warnings
import numpy as np
from typing import Optional
from scipy.special import xlogy
from cleanlab.count import get_confident_thresholds

def _subtract_confident_thresholds(labels: Optional[np.ndarray], pred_probs: np.ndarray, multi_label: bool=False, confident_thresholds: Optional[np.ndarray]=None) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    '\n    Return adjusted predicted probabilities by subtracting the class confident thresholds and renormalizing.\n\n    The confident class threshold for a class j is the expected (average) "self-confidence" for class j.\n    The purpose of this adjustment is to handle class imbalance.\n\n    Parameters\n    ----------\n    labels : np.ndarray\n      Labels in the same format expected by the `cleanlab.count.get_confident_thresholds()` method.\n      If labels is None, confident_thresholds needs to be passed in as it will not be calculated.\n    pred_probs : np.ndarray (shape (N, K))\n      Predicted-probabilities in the same format expected by the `cleanlab.count.get_confident_thresholds()` method.\n    confident_thresholds : np.ndarray (shape (K,))\n      Pre-calculated confident thresholds. If passed in, function will subtract these thresholds instead of calculating\n      confident_thresholds from the given labels and pred_probs.\n    multi_label : bool, optional\n      If ``True``, labels should be an iterable (e.g. list) of iterables, containing a\n      list of labels for each example, instead of just a single label.\n      The multi-label setting supports classification tasks where an example has 1 or more labels.\n      Example of a multi-labeled `labels` input: ``[[0,1], [1], [0,2], [0,1,2], [0], [1], ...]``.\n      The major difference in how this is calibrated versus single-label is that\n      the total number of errors considered is based on the number of labels,\n      not the number of examples. So, the calibrated `confident_joint` will sum\n      to the number of total labels.\n\n    Returns\n    -------\n    pred_probs_adj : np.ndarray (float)\n      Adjusted pred_probs.\n    '
    if confident_thresholds is None:
        if labels is None:
            raise ValueError('Cannot calculate confident_thresholds without labels. Pass in either labels or already calculated confident_thresholds parameter. ')
        confident_thresholds = get_confident_thresholds(labels, pred_probs, multi_label=multi_label)
    pred_probs_adj = pred_probs - confident_thresholds
    pred_probs_adj += confident_thresholds.max()
    pred_probs_adj /= pred_probs_adj.sum(axis=1, keepdims=True)
    return pred_probs_adj

def get_normalized_entropy(pred_probs: np.ndarray, min_allowed_prob: Optional[float]=None) -> np.ndarray:
    if False:
        print('Hello World!')
    "Return the normalized entropy of pred_probs.\n\n    Normalized entropy is between 0 and 1. Higher values of entropy indicate higher uncertainty in the model's prediction of the correct label.\n\n    Read more about normalized entropy `on Wikipedia <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_.\n\n    Normalized entropy is used in active learning for uncertainty sampling: https://towardsdatascience.com/uncertainty-sampling-cheatsheet-ec57bc067c0b\n\n    Unlike label-quality scores, entropy only depends on the model's predictions, not the given label.\n\n    Parameters\n    ----------\n    pred_probs : np.ndarray (shape (N, K))\n      Each row of this matrix corresponds to an example x and contains the model-predicted\n      probabilities that x belongs to each possible class: P(label=k|x)\n\n    min_allowed_prob : float, default: None, deprecated\n      Minimum allowed probability value. If not `None` (default),\n      entries of `pred_probs` below this value will be clipped to this value.\n\n      .. deprecated:: 2.5.0\n         This keyword is deprecated and should be left to the default.\n         The entropy is well-behaved even if `pred_probs` contains zeros,\n         clipping is unnecessary and (slightly) changes the results.\n\n    Returns\n    -------\n    entropy : np.ndarray (shape (N, ))\n      Each element is the normalized entropy of the corresponding row of ``pred_probs``.\n\n    Raises\n    ------\n    ValueError\n        An error is raised if any of the probabilities is not in the interval [0, 1].\n    "
    if np.any(pred_probs < 0) or np.any(pred_probs > 1):
        raise ValueError('All probabilities are required to be in the interval [0, 1].')
    num_classes = pred_probs.shape[1]
    if min_allowed_prob is not None:
        warnings.warn('Using `min_allowed_prob` is not necessary anymore and will be removed.', DeprecationWarning)
        pred_probs = np.clip(pred_probs, a_min=min_allowed_prob, a_max=None)
    return -np.sum(xlogy(pred_probs, pred_probs), axis=1) / np.log(num_classes)