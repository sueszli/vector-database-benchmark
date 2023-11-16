""" A collection of statistical models
"""
from __future__ import division
from __future__ import print_function
import numpy as np
from numba import njit
from scipy.stats import pearsonr
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_consistent_length

def pairwise_distances_no_broadcast(X, Y):
    if False:
        while True:
            i = 10
    'Utility function to calculate row-wise euclidean distance of two matrix.\n    Different from pair-wise calculation, this function would not broadcast.\n\n    For instance, X and Y are both (4,3) matrices, the function would return\n    a distance vector with shape (4,), instead of (4,4).\n\n    Parameters\n    ----------\n    X : array of shape (n_samples, n_features)\n        First input samples\n\n    Y : array of shape (n_samples, n_features)\n        Second input samples\n\n    Returns\n    -------\n    distance : array of shape (n_samples,)\n        Row-wise euclidean distance of X and Y\n    '
    X = check_array(X)
    Y = check_array(Y)
    if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
        raise ValueError('pairwise_distances_no_broadcast function receivematrix with different shapes {0} and {1}'.format(X.shape, Y.shape))
    return _pairwise_distances_no_broadcast_helper(X, Y)

@njit
def _pairwise_distances_no_broadcast_helper(X, Y):
    if False:
        for i in range(10):
            print('nop')
    'Internal function for calculating the distance with numba. Do not use.\n\n    Parameters\n    ----------\n    X : array of shape (n_samples, n_features)\n        First input samples\n\n    Y : array of shape (n_samples, n_features)\n        Second input samples\n\n    Returns\n    -------\n    distance : array of shape (n_samples,)\n        Intermediate results. Do not use.\n\n    '
    euclidean_sq = np.square(Y - X)
    return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()

def wpearsonr(x, y, w=None):
    if False:
        return 10
    'Utility function to calculate the weighted Pearson correlation of two\n    samples.\n\n    See https://stats.stackexchange.com/questions/221246/such-thing-as-a-weighted-correlation\n    for more information\n\n    Parameters\n    ----------\n    x : array, shape (n,)\n        Input x.\n\n    y : array, shape (n,)\n        Input y.\n\n    w : array, shape (n,)\n        Weights w.\n\n    Returns\n    -------\n    scores : float in range of [-1,1]\n        Weighted Pearson Correlation between x and y.\n\n    '
    if w is None:
        return pearsonr(x, y)
    x = np.asarray(x)
    y = np.asarray(y)
    w = np.asarray(w)
    check_consistent_length([x, y, w])
    w_sum = w.sum()
    mx = np.sum(x * w) / w_sum
    my = np.sum(y * w) / w_sum
    (xm, ym) = (x - mx, y - my)
    r_num = np.sum(xm * ym * w) / w_sum
    xm2 = np.sum(xm * xm * w) / w_sum
    ym2 = np.sum(ym * ym * w) / w_sum
    r_den = np.sqrt(xm2 * ym2)
    r = r_num / r_den
    r = max(min(r, 1.0), -1.0)
    return r

def pearsonr_mat(mat, w=None):
    if False:
        return 10
    'Utility function to calculate pearson matrix (row-wise).\n\n    Parameters\n    ----------\n    mat : numpy array of shape (n_samples, n_features)\n        Input matrix.\n\n    w : numpy array of shape (n_features,)\n        Weights.\n\n    Returns\n    -------\n    pear_mat : numpy array of shape (n_samples, n_samples)\n        Row-wise pearson score matrix.\n\n    '
    mat = check_array(mat)
    n_row = mat.shape[0]
    n_col = mat.shape[1]
    pear_mat = np.full([n_row, n_row], 1).astype(float)
    if w is not None:
        for cx in range(n_row):
            for cy in range(cx + 1, n_row):
                curr_pear = wpearsonr(mat[cx, :], mat[cy, :], w)
                pear_mat[cx, cy] = curr_pear
                pear_mat[cy, cx] = curr_pear
    else:
        for cx in range(n_col):
            for cy in range(cx + 1, n_row):
                curr_pear = pearsonr(mat[cx, :], mat[cy, :])[0]
                pear_mat[cx, cy] = curr_pear
                pear_mat[cy, cx] = curr_pear
    return pear_mat

def column_ecdf(matrix: np.ndarray) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    '\n    Utility function to compute the column wise empirical cumulative distribution of a 2D feature matrix,\n    where the rows are samples and the columns are features per sample. The accumulation is done in the positive\n    direction of the sample axis.\n\n    E.G.\n    p(1) = 0.2, p(0) = 0.3, p(2) = 0.1, p(6) = 0.4\n    ECDF E(5) = p(x <= 5)\n    ECDF E would be E(-1) = 0, E(0) = 0.3, E(1) = 0.5, E(2) = 0.6, E(3) = 0.6, E(4) = 0.6, E(5) = 0.6, E(6) = 1\n\n    Similar to and tested against:\n    https://www.statsmodels.org/stable/generated/statsmodels.distributions.empirical_distribution.ECDF.html\n\n    Returns\n    -------\n\n    '
    assert len(matrix.shape) == 2, 'Matrix needs to be two dimensional for the ECDF computation.'
    probabilities = np.linspace(np.ones(matrix.shape[1]) / matrix.shape[0], np.ones(matrix.shape[1]), matrix.shape[0])
    sort_idx = np.argsort(matrix, axis=0)
    matrix = np.take_along_axis(matrix, sort_idx, axis=0)
    ecdf_terminate_equals_inplace(matrix, probabilities)
    reordered_probabilities = np.ones_like(probabilities)
    np.put_along_axis(reordered_probabilities, sort_idx, probabilities, axis=0)
    return reordered_probabilities

@njit
def ecdf_terminate_equals_inplace(matrix: np.ndarray, probabilities: np.ndarray):
    if False:
        while True:
            i = 10
    "\n    This is a helper function for computing the ecdf of an array. It has been outsourced from the original\n    function in order to be able to use the njit compiler of numpy for increased speeds, as it unfortunately\n    needs a loop over all rows and columns of a matrix. It acts in place on the probabilities' matrix.\n\n    Parameters\n    ----------\n    matrix : a feature matrix where the rows are samples and each column is a feature !(expected to be sorted)!\n\n    probabilities : a probability matrix that will be used building the ecdf. It has values between 0 and 1 and\n                    is also sorted.\n\n    Returns\n    -------\n\n    "
    for cx in range(probabilities.shape[1]):
        for rx in range(probabilities.shape[0] - 2, -1, -1):
            if matrix[rx, cx] == matrix[rx + 1, cx]:
                probabilities[rx, cx] = probabilities[rx + 1, cx]