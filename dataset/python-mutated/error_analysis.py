import logging
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple
import numpy as np
from snorkel.utils import to_int_label_array

def get_label_buckets(*y: np.ndarray) -> Dict[Tuple[int, ...], np.ndarray]:
    if False:
        i = 10
        return i + 15
    'Return data point indices bucketed by label combinations.\n\n    Parameters\n    ----------\n    *y\n        A list of np.ndarray of (int) labels\n\n    Returns\n    -------\n    Dict[Tuple[int, ...], np.ndarray]\n        A mapping of each label bucket to a NumPy array of its corresponding indices\n\n    Example\n    -------\n    A common use case is calling ``buckets = label_buckets(Y_gold, Y_pred)`` where\n    ``Y_gold`` is a set of gold (i.e. ground truth) labels and ``Y_pred`` is a\n    corresponding set of predicted labels.\n\n    >>> Y_gold = np.array([1, 1, 1, 0])\n    >>> Y_pred = np.array([1, 1, -1, -1])\n    >>> buckets = get_label_buckets(Y_gold, Y_pred)\n\n    The returned ``buckets[(i, j)]`` is a NumPy array of data point indices with\n    true label i and predicted label j.\n\n    More generally, the returned indices within each bucket refer to the order of the\n    labels that were passed in as function arguments.\n\n    >>> buckets[(1, 1)]  # true positives\n    array([0, 1])\n    >>> (1, 0) in buckets  # false positives\n    False\n    >>> (0, 1) in buckets  # false negatives\n    False\n    >>> (0, 0) in buckets  # true negatives\n    False\n    >>> buckets[(1, -1)]  # abstained positives\n    array([2])\n    >>> buckets[(0, -1)]  # abstained negatives\n    array([3])\n    '
    buckets: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)
    y_flat = list(map(lambda x: to_int_label_array(x, flatten_vector=True), y))
    if len(set(map(len, y_flat))) != 1:
        raise ValueError('Arrays must all have the same number of elements')
    for (i, labels) in enumerate(zip(*y_flat)):
        buckets[labels].append(i)
    return {k: np.array(v) for (k, v) in buckets.items()}

def get_label_instances(bucket: Tuple[int, ...], x: np.ndarray, *y: np.ndarray) -> np.ndarray:
    if False:
        return 10
    'Return instances in x with the specified combination of labels.\n\n    Parameters\n    ----------\n    bucket\n        A tuple of label values corresponding to which instances from x are returned\n    x\n        NumPy array of data instances to be returned\n    *y\n        A list of np.ndarray of (int) labels\n\n    Returns\n    -------\n    np.ndarray\n        NumPy array of instances from x with the specified combination of labels\n\n    Example\n    -------\n    A common use case is calling ``get_label_instances(bucket, x.to_numpy(), Y_gold, Y_pred)``\n    where ``x`` is a NumPy array of data instances that the labels correspond to,\n    ``Y_gold`` is a list of gold (i.e. ground truth) labels, and\n    ``Y_pred`` is a corresponding list of predicted labels.\n\n    >>> import pandas as pd\n    >>> x = pd.DataFrame(data={\'col1\': ["this is a string", "a second string", "a third string"], \'col2\': ["1", "2", "3"]})\n    >>> Y_gold = np.array([1, 1, 1])\n    >>> Y_pred = np.array([1, 0, 0])\n    >>> bucket = (1, 0)\n\n    The returned NumPy array of data instances from ``x`` will correspond to\n    the rows where the first list had a 1 and the second list had a 0.\n    >>> get_label_instances(bucket, x.to_numpy(), Y_gold, Y_pred)\n    array([[\'a second string\', \'2\'],\n           [\'a third string\', \'3\']], dtype=object)\n\n    More generally, given bucket ``(i, j, ...)`` and lists ``y1, y2, ...``\n    the returned data instances from ``x`` will correspond to the rows where\n    y1 had label i, y2 had label j, and so on. Note that ``x`` and ``y``\n    must all be the same length.\n    '
    if len(y) != len(bucket):
        raise ValueError('Number of lists must match the amount of labels in bucket')
    if x.shape[0] != len(y[0]):
        raise ValueError('Number of rows in x does not match number of elements in at least one label list')
    buckets = get_label_buckets(*y)
    try:
        indices = buckets[bucket]
    except KeyError:
        logging.warning('Bucket' + str(bucket) + ' does not exist.')
        return np.array([])
    instances = x[indices]
    return instances