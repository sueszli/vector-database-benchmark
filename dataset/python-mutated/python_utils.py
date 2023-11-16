import logging
import numpy as np
from scipy import sparse
logger = logging.getLogger()

def exponential_decay(value, max_val, half_life):
    if False:
        return 10
    'Compute decay factor for a given value based on an exponential decay.\n\n    Values greater than `max_val` will be set to 1.\n\n    Args:\n        value (numeric): Value to calculate decay factor\n        max_val (numeric): Value at which decay factor will be 1\n        half_life (numeric): Value at which decay factor will be 0.5\n\n    Returns:\n        float: Decay factor\n    '
    return np.minimum(1.0, np.power(0.5, (max_val - value) / half_life))

def _get_row_and_column_matrix(array):
    if False:
        print('Hello World!')
    'Helper method to get the row and column matrix from an array.\n\n    Args:\n        array (numpy.ndarray): the array from which to get the row and column matrix.\n\n    Returns:\n        (numpy.ndarray, numpy.ndarray): (row matrix, column matrix)\n    '
    row_matrix = np.expand_dims(array, axis=0)
    column_matrix = np.expand_dims(array, axis=1)
    return (row_matrix, column_matrix)

def jaccard(cooccurrence):
    if False:
        i = 10
        return i + 15
    'Helper method to calculate the Jaccard similarity of a matrix of\n    co-occurrences.  When comparing Jaccard with count co-occurrence\n    and lift similarity, count favours predictability, meaning that\n    the most popular items will be recommended most of the time. Lift,\n    by contrast, favours discoverability/serendipity, meaning that an\n    item that is less popular overall but highly favoured by a small\n    subset of users is more likely to be recommended. Jaccard is a\n    compromise between the two.\n\n    Args:\n        cooccurrence (numpy.ndarray): the symmetric matrix of co-occurrences of items.\n\n    Returns:\n        numpy.ndarray: The matrix of Jaccard similarities between any two items.\n\n    '
    (diag_rows, diag_cols) = _get_row_and_column_matrix(cooccurrence.diagonal())
    with np.errstate(invalid='ignore', divide='ignore'):
        result = cooccurrence / (diag_rows + diag_cols - cooccurrence)
    return np.array(result)

def lift(cooccurrence):
    if False:
        return 10
    'Helper method to calculate the Lift of a matrix of\n    co-occurrences. In comparison with basic co-occurrence and Jaccard\n    similarity, lift favours discoverability and serendipity, as\n    opposed to co-occurrence that favours the most popular items, and\n    Jaccard that is a compromise between the two.\n\n    Args:\n        cooccurrence (numpy.ndarray): The symmetric matrix of co-occurrences of items.\n\n    Returns:\n        numpy.ndarray: The matrix of Lifts between any two items.\n\n    '
    (diag_rows, diag_cols) = _get_row_and_column_matrix(cooccurrence.diagonal())
    with np.errstate(invalid='ignore', divide='ignore'):
        result = cooccurrence / (diag_rows * diag_cols)
    return np.array(result)

def mutual_information(cooccurrence):
    if False:
        i = 10
        return i + 15
    'Helper method to calculate the Mutual Information of a matrix of\n    co-occurrences.\n\n    Mutual information is a measurement of the amount of information\n    explained by the i-th j-th item column vector.\n\n    Args:\n        cooccurrence (numpy.ndarray): The symmetric matrix of co-occurrences of items.\n\n    Returns:\n        numpy.ndarray: The matrix of mutual information between any two items.\n\n    '
    with np.errstate(invalid='ignore', divide='ignore'):
        result = np.log2(cooccurrence.shape[0] * lift(cooccurrence))
    return np.array(result)

def lexicographers_mutual_information(cooccurrence):
    if False:
        return 10
    'Helper method to calculate the Lexicographers Mutual Information of\n    a matrix of co-occurrences.\n\n    Due to the bias of mutual information for low frequency items,\n    lexicographers mutual information corrects the formula by\n    multiplying it by the co-occurrence frequency.\n\n    Args:\n        cooccurrence (numpy.ndarray): The symmetric matrix of co-occurrences of items.\n\n    Returns:\n        numpy.ndarray: The matrix of lexicographers mutual information between any two items.\n\n    '
    with np.errstate(invalid='ignore', divide='ignore'):
        result = cooccurrence * mutual_information(cooccurrence)
    return np.array(result)

def cosine_similarity(cooccurrence):
    if False:
        for i in range(10):
            print('nop')
    'Helper method to calculate the Cosine similarity of a matrix of\n    co-occurrences.\n\n    Cosine similarity can be interpreted as the angle between the i-th\n    and j-th item.\n\n    Args:\n        cooccurrence (numpy.ndarray): The symmetric matrix of co-occurrences of items.\n\n    Returns:\n        numpy.ndarray: The matrix of cosine similarity between any two items.\n\n    '
    (diag_rows, diag_cols) = _get_row_and_column_matrix(cooccurrence.diagonal())
    with np.errstate(invalid='ignore', divide='ignore'):
        result = cooccurrence / np.sqrt(diag_rows * diag_cols)
    return np.array(result)

def inclusion_index(cooccurrence):
    if False:
        while True:
            i = 10
    'Helper method to calculate the Inclusion Index of a matrix of\n    co-occurrences.\n\n    Inclusion index measures the overlap between items.\n\n    Args:\n        cooccurrence (numpy.ndarray): The symmetric matrix of co-occurrences of items.\n\n    Returns:\n        numpy.ndarray: The matrix of inclusion index between any two items.\n\n    '
    (diag_rows, diag_cols) = _get_row_and_column_matrix(cooccurrence.diagonal())
    with np.errstate(invalid='ignore', divide='ignore'):
        result = cooccurrence / np.minimum(diag_rows, diag_cols)
    return np.array(result)

def get_top_k_scored_items(scores, top_k, sort_top_k=False):
    if False:
        while True:
            i = 10
    "Extract top K items from a matrix of scores for each user-item pair, optionally sort results per user.\n\n    Args:\n        scores (numpy.ndarray): Score matrix (users x items).\n        top_k (int): Number of top items to recommend.\n        sort_top_k (bool): Flag to sort top k results.\n\n    Returns:\n        numpy.ndarray, numpy.ndarray:\n        - Indices into score matrix for each user's top items.\n        - Scores corresponding to top items.\n\n    "
    if isinstance(scores, sparse.spmatrix):
        scores = scores.todense()
    if scores.shape[1] < top_k:
        logger.warning('Number of items is less than top_k, limiting top_k to number of items')
    k = min(top_k, scores.shape[1])
    test_user_idx = np.arange(scores.shape[0])[:, None]
    top_items = np.argpartition(scores, -k, axis=1)[:, -k:]
    top_scores = scores[test_user_idx, top_items]
    if sort_top_k:
        sort_ind = np.argsort(-top_scores)
        top_items = top_items[test_user_idx, sort_ind]
        top_scores = top_scores[test_user_idx, sort_ind]
    return (np.array(top_items), np.array(top_scores))

def binarize(a, threshold):
    if False:
        while True:
            i = 10
    'Binarize the values.\n\n    Args:\n        a (numpy.ndarray): Input array that needs to be binarized.\n        threshold (float): Threshold below which all values are set to 0, else 1.\n\n    Returns:\n        numpy.ndarray: Binarized array.\n    '
    return np.where(a > threshold, 1.0, 0.0)

def rescale(data, new_min=0, new_max=1, data_min=None, data_max=None):
    if False:
        return 10
    'Rescale/normalize the data to be within the range `[new_min, new_max]`\n    If data_min and data_max are explicitly provided, they will be used\n    as the old min/max values instead of taken from the data.\n\n    .. note::\n        This is same as the `scipy.MinMaxScaler` with the exception that we can override\n        the min/max of the old scale.\n\n    Args:\n        data (numpy.ndarray): 1d scores vector or 2d score matrix (users x items).\n        new_min (int|float): The minimum of the newly scaled data.\n        new_max (int|float): The maximum of the newly scaled data.\n        data_min (None|number): The minimum of the passed data [if omitted it will be inferred].\n        data_max (None|number): The maximum of the passed data [if omitted it will be inferred].\n\n    Returns:\n        numpy.ndarray: The newly scaled/normalized data.\n    '
    data_min = data.min() if data_min is None else data_min
    data_max = data.max() if data_max is None else data_max
    return (data - data_min) / (data_max - data_min) * (new_max - new_min) + new_min