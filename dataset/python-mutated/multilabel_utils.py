"""
Helper functions used internally for multi-label classification tasks.
"""
from typing import Tuple, Optional, List
import numpy as np
from cleanlab.internal.util import get_num_classes

def _is_multilabel(y: np.ndarray) -> bool:
    if False:
        while True:
            i = 10
    'Checks whether `y` is in a multi-label indicator matrix format.\n\n    Sparse matrices are not supported.\n    '
    if not (isinstance(y, np.ndarray) and y.ndim == 2 and (y.shape[1] > 1)):
        return False
    return np.array_equal(np.unique(y), [0, 1])

def stack_complement(pred_prob_slice: np.ndarray) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    '\n    Extends predicted probabilities of a single class to two columns.\n\n    Parameters\n    ----------\n    pred_prob_slice:\n        A 1D array with predicted probabilities for a single class.\n\n    Example\n    -------\n    >>> pred_prob_slice = np.array([0.1, 0.9, 0.3, 0.8])\n    >>> stack_complement(pred_prob_slice)\n    array([[0.9, 0.1],\n            [0.1, 0.9],\n            [0.7, 0.3],\n            [0.2, 0.8]])\n    '
    return np.vstack((1 - pred_prob_slice, pred_prob_slice)).T

def get_onehot_num_classes(labels: list, pred_probs: Optional[np.ndarray]=None) -> Tuple[np.ndarray, int]:
    if False:
        i = 10
        return i + 15
    'Returns OneHot encoding of MultiLabel Data, and number of classes'
    num_classes = get_num_classes(labels=labels, pred_probs=pred_probs)
    try:
        y_one = int2onehot(labels, K=num_classes)
    except TypeError:
        raise ValueError('wrong format for labels, should be a list of list[indices], please check the documentation in find_label_issues for further information')
    return (y_one, num_classes)

def int2onehot(labels: list, K: int) -> np.ndarray:
    if False:
        return 10
    'Convert multi-label classification `labels` from a ``List[List[int]]`` format to a onehot matrix.\n    This returns a binarized format of the labels as a multi-hot vector for each example, where the entries in this vector are 1 for each class that applies to this example and 0 otherwise.\n\n    Parameters\n    ----------\n    labels: list of lists of integers\n      e.g. [[0,1], [3], [1,2,3], [1], [2]]\n      All integers from 0,1,...,K-1 must be represented.\n    K: int\n      The number of classes.'
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=range(K))
    return mlb.fit_transform(labels)

def onehot2int(onehot_matrix: np.ndarray) -> List[List[int]]:
    if False:
        i = 10
        return i + 15
    'Convert multi-label classification `labels` from a onehot matrix format to a ``List[List[int]]`` format that can be used with other cleanlab functions.\n\n    Parameters\n    ----------\n    onehot_matrix: 2D np.ndarray of 0s and 1s\n      A matrix representation of multi-label classification labels in a binarized format as a multi-hot vector for each example.\n      The entries in this vector are 1 for each class that applies to this example and 0 otherwise.\n\n    Returns\n    -------\n    labels: list of lists of integers\n      e.g. [[0,1], [3], [1,2,3], [1], [2]]\n      All integers from 0,1,...,K-1 must be represented.'
    return [list(np.where(row == 1)[0]) for row in onehot_matrix]