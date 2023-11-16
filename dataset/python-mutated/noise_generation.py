"""
Helper methods that are useful for benchmarking cleanlab’s core algorithms.
These methods introduce synthetic noise into the labels of a classification dataset.
Specifically, this module provides methods for generating valid noise matrices (for which learning with noise is possible),
generating noisy labels given a noise matrix, generating valid noise matrices with a specific trace value, and more.
"""
from typing import Optional
import numpy as np
from cleanlab.internal.util import value_counts
from cleanlab.internal.constants import FLOATING_POINT_COMPARISON

def noise_matrix_is_valid(noise_matrix, py, *, verbose=False) -> bool:
    if False:
        i = 10
        return i + 15
    'Given a prior `py` representing ``p(true_label=k)``, checks if the given `noise_matrix` is a\n    learnable matrix. Learnability means that it is possible to achieve\n    better than random performance, on average, for the amount of noise in\n    `noise_matrix`.\n\n    Parameters\n    ----------\n    noise_matrix : np.ndarray\n      An array of shape ``(K, K)`` representing the conditional probability\n      matrix ``P(label=k_s|true_label=k_y)`` containing the fraction of\n      examples in every class, labeled as every other class. Assumes columns of\n      `noise_matrix` sum to 1.\n\n    py : np.ndarray\n      An array of shape ``(K,)`` representing the fraction (prior probability)\n      of each true class label, ``P(true_label = k)``.\n\n    Returns\n    -------\n    is_valid : bool\n      Whether the noise matrix is a learnable matrix.\n    '
    K = len(py)
    N = float(10000)
    ps = np.dot(noise_matrix, py)
    joint_noise = np.multiply(noise_matrix, py)
    if not abs(joint_noise.sum() - 1.0) < FLOATING_POINT_COMPARISON:
        return False
    for i in range(K):
        C = N * joint_noise[i][i]
        E1 = N * joint_noise[i].sum() - C
        E2 = N * joint_noise.T[i].sum() - C
        O = N - E1 - E2 - C
        if verbose:
            print('E1E2/C', round(E1 * E2 / C), 'E1', round(E1), 'E2', round(E2), 'C', round(C), '|', round(E1 * E2 / C + E1 + E2 + C), '|', round(E1 * E2 / C), '<', round(O))
            print(round(ps[i] * py[i]), '<', round(joint_noise[i][i]), ':', ps[i] * py[i] < joint_noise[i][i])
        if not ps[i] * py[i] < joint_noise[i][i]:
            return False
    return True

def generate_noisy_labels(true_labels, noise_matrix) -> np.ndarray:
    if False:
        print('Hello World!')
    'Generates noisy `labels` from perfect labels `true_labels`,\n    "exactly" yielding the provided `noise_matrix` between `labels` and `true_labels`.\n\n    Below we provide a for loop implementation of what this function does.\n    We do not use this implementation as it is not a fast algorithm, but\n    it explains as Python pseudocode what is happening in this function.\n\n    Parameters\n    ----------\n    true_labels : np.ndarray\n      An array of shape ``(N,)`` representing perfect labels, without any\n      noise. Contains K distinct natural number classes, 0, 1, ..., K-1.\n\n    noise_matrix : np.ndarray\n      An array of shape ``(K, K)`` representing the conditional probability\n      matrix ``P(label=k_s|true_label=k_y)`` containing the fraction of\n      examples in every class, labeled as every other class. Assumes columns of\n      `noise_matrix` sum to 1.\n\n    Returns\n    -------\n    labels : np.ndarray\n      An array of shape ``(N,)`` of noisy labels.\n\n    Examples\n    --------\n\n    .. code:: python\n\n        # Generate labels\n        count_joint = (noise_matrix * py * len(y)).round().astype(int)\n        labels = np.ndarray(y)\n        for k_s in range(K):\n            for k_y in range(K):\n                if k_s != k_y:\n                    idx_flip = np.where((labels==k_y)&(true_label==k_y))[0]\n                    if len(idx_flip): # pragma: no cover\n                        labels[np.random.choice(\n                            idx_flip,\n                            count_joint[k_s][k_y],\n                            replace=False,\n                        )] = k_s\n    '
    true_labels = np.asarray(true_labels)
    K = len(noise_matrix)
    py = value_counts(true_labels) / float(len(true_labels))
    count_joint = (noise_matrix * py * len(true_labels)).astype(int)
    np.fill_diagonal(count_joint, 0)
    labels = np.array(true_labels)
    for k in range(K):
        labels_per_class = np.where(count_joint[:, k] != 0)[0]
        label_counts = count_joint[labels_per_class, k]
        noise = [labels_per_class[i] for (i, c) in enumerate(label_counts) for z in range(c)]
        idx_flip = np.where((labels == k) & (true_labels == k))[0]
        if len(idx_flip) and len(noise) and (len(idx_flip) >= len(noise)):
            labels[np.random.choice(idx_flip, len(noise), replace=False)] = noise
    return labels

def generate_noise_matrix_from_trace(K, trace, *, max_trace_prob=1.0, min_trace_prob=1e-05, max_noise_rate=1 - 1e-05, min_noise_rate=0.0, valid_noise_matrix=True, py=None, frac_zero_noise_rates=0.0, seed=0, max_iter=10000) -> Optional[np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    'Generates a ``K x K`` noise matrix ``P(label=k_s|true_label=k_y)`` with\n    ``np.sum(np.diagonal(noise_matrix))`` equal to the given `trace`.\n\n    Parameters\n    ----------\n    K : int\n      Creates a noise matrix of shape ``(K, K)``. Implies there are\n      K classes for learning with noisy labels.\n\n    trace : float\n      Sum of diagonal entries of array of random probabilities returned.\n\n    max_trace_prob : float\n      Maximum probability of any entry in the trace of the return matrix.\n\n    min_trace_prob : float\n      Minimum probability of any entry in the trace of the return matrix.\n\n    max_noise_rate : float\n      Maximum noise_rate (non-diagonal entry) in the returned np.ndarray.\n\n    min_noise_rate : float\n      Minimum noise_rate (non-diagonal entry) in the returned np.ndarray.\n\n    valid_noise_matrix : bool, default=True\n      If ``True``, returns a matrix having all necessary conditions for\n      learning with noisy labels. In particular, ``p(true_label=k)p(label=k) < p(true_label=k,label=k)``\n      is satisfied. This requires that ``trace > 1``.\n\n    py : np.ndarray\n      An array of shape ``(K,)`` representing the fraction (prior probability) of each true class label, ``P(true_label = k)``.\n      This argument is **required** when ``valid_noise_matrix=True``.\n\n    frac_zero_noise_rates : float\n      The fraction of the ``n*(n-1)`` noise rates\n      that will be set to 0. Note that if you set a high trace, it may be\n      impossible to also have a low fraction of zero noise rates without\n      forcing all non-1 diagonal values. Instead, when this happens we only\n      guarantee to produce a noise matrix with `frac_zero_noise_rates` *or\n      higher*. The opposite occurs with a small trace.\n\n    seed : int\n      Seeds the random number generator for numpy.\n\n    max_iter : int, default=10000\n      The max number of tries to produce a valid matrix before returning ``None``.\n\n    Returns\n    -------\n    noise_matrix : np.ndarray or None\n      An array of shape ``(K, K)`` representing the noise matrix ``P(label=k_s|true_label=k_y)`` with `trace`\n      equal to ``np.sum(np.diagonal(noise_matrix))``. This a conditional probability matrix and a\n      left stochastic matrix. Returns ``None`` if `max_iter` is exceeded.\n    '
    if valid_noise_matrix and trace <= 1:
        raise ValueError('trace = {}. trace > 1 is necessary for a'.format(trace) + ' valid noise matrix to be returned (valid_noise_matrix == True)')
    if valid_noise_matrix and py is None and (K > 2):
        raise ValueError('py must be provided (not None) if the input parameter' + ' valid_noise_matrix == True')
    if K <= 1:
        raise ValueError('K must be >= 2, but K = {}.'.format(K))
    if max_iter < 1:
        return None
    np.random.seed(seed)
    if K == 2:
        if frac_zero_noise_rates >= 0.5:
            noise_mat = np.array([[1.0, 1 - (trace - 1.0)], [0.0, trace - 1.0]])
            return noise_mat if np.random.rand() > 0.5 else np.rot90(noise_mat, k=2)
        else:
            diag = generate_n_rand_probabilities_that_sum_to_m(2, trace)
            noise_matrix = np.array([[diag[0], 1 - diag[1]], [1 - diag[0], diag[1]]])
            return noise_matrix
    for z in range(max_iter):
        noise_matrix = np.zeros(shape=(K, K))
        nm_diagonal = generate_n_rand_probabilities_that_sum_to_m(n=K, m=trace, max_prob=max_trace_prob, min_prob=min_trace_prob)
        np.fill_diagonal(noise_matrix, nm_diagonal)
        num_col_with_noise = K - np.count_nonzero(1 == nm_diagonal)
        num_zero_noise_rates = int(K * (K - 1) * frac_zero_noise_rates)
        num_zero_noise_rates -= (K - num_col_with_noise) * (K - 1)
        num_zero_noise_rates = np.maximum(num_zero_noise_rates, 0)
        num_zero_noise_rates_per_col = randomly_distribute_N_balls_into_K_bins(N=num_zero_noise_rates, K=num_col_with_noise, max_balls_per_bin=K - 2, min_balls_per_bin=0) if K > 2 else np.array([0, 0])
        stack_nonzero_noise_rates_per_col = list(K - 1 - num_zero_noise_rates_per_col)[::-1]
        for col in np.arange(K)[nm_diagonal != 1]:
            num_noise = stack_nonzero_noise_rates_per_col.pop()
            noise_rates_col = list(generate_n_rand_probabilities_that_sum_to_m(n=num_noise, m=1 - nm_diagonal[col], max_prob=max_noise_rate, min_prob=min_noise_rate))
            rows = np.random.choice([row for row in range(K) if row != col], num_noise, replace=False)
            for row in rows:
                noise_matrix[row][col] = noise_rates_col.pop()
        if not valid_noise_matrix or noise_matrix_is_valid(noise_matrix, py):
            return noise_matrix
    return None

def generate_n_rand_probabilities_that_sum_to_m(n, m, *, max_prob=1.0, min_prob=0.0) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    '\n    Generates `n` random probabilities that sum to `m`.\n\n    When ``min_prob=0`` and ``max_prob = 1.0``, use\n    ``np.random.dirichlet(np.ones(n))*m`` instead.\n\n    Parameters\n    ----------\n    n : int\n      Length of array of random probabilities to be returned.\n\n    m : float\n      Sum of array of random probabilities that is returned.\n\n    max_prob : float, default=1.0\n      Maximum probability of any entry in the returned array. Must be between 0 and 1.\n\n    min_prob : float, default=0.0\n      Minimum probability of any entry in the returned array. Must be between 0 and 1.\n\n    Returns\n    -------\n    probabilities : np.ndarray\n      An array of probabilities.\n    '
    if n == 0:
        return np.array([])
    if max_prob + FLOATING_POINT_COMPARISON < m / float(n):
        raise ValueError('max_prob must be greater or equal to m / n, but ' + 'max_prob = ' + str(max_prob) + ', m = ' + str(m) + ', n = ' + str(n) + ', m / n = ' + str(m / float(n)))
    if min_prob > (m + FLOATING_POINT_COMPARISON) / float(n):
        raise ValueError('min_prob must be less or equal to m / n, but ' + 'max_prob = ' + str(max_prob) + ', m = ' + str(m) + ', n = ' + str(n) + ', m / n = ' + str(m / float(n)))
    result = np.random.dirichlet(np.ones(n)) * m
    min_val = min(result)
    max_val = max(result)
    while max_val > max_prob + FLOATING_POINT_COMPARISON:
        new_min = min_val + (max_val - max_prob)
        adjustment = (max_prob - new_min) * np.random.rand()
        result[np.argmin(result)] = new_min + adjustment
        result[np.argmax(result)] = max_prob - adjustment
        min_val = min(result)
        max_val = max(result)
    min_val = min(result)
    max_val = max(result)
    while min_val < min_prob - FLOATING_POINT_COMPARISON:
        min_val = min(result)
        max_val = max(result)
        new_max = max_val - (min_prob - min_val)
        adjustment = (new_max - min_prob) * np.random.rand()
        result[np.argmax(result)] = new_max - adjustment
        result[np.argmin(result)] = min_prob + adjustment
        min_val = min(result)
        max_val = max(result)
    return result

def randomly_distribute_N_balls_into_K_bins(N, K, *, max_balls_per_bin=None, min_balls_per_bin=None) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Returns a uniformly random numpy integer array of length `N` that sums\n    to `K`.\n\n    Parameters\n    ----------\n    N : int\n      Number of balls.\n    K : int\n      Number of bins.\n    max_balls_per_bin : int\n      Ensure that each bin contains at most `max_balls_per_bin` balls.\n    min_balls_per_bin : int\n      Ensure that each bin contains at least `min_balls_per_bin` balls.\n\n    Returns\n    -------\n    int_array : np.array\n      Length `N` array that sums to `K`.\n    '
    if N == 0:
        return np.zeros(K, dtype=int)
    if max_balls_per_bin is None:
        max_balls_per_bin = N
    else:
        max_balls_per_bin = min(max_balls_per_bin, N)
    if min_balls_per_bin is None:
        min_balls_per_bin = 0
    else:
        min_balls_per_bin = min(min_balls_per_bin, N / K)
    if N / float(K) > max_balls_per_bin:
        N = max_balls_per_bin * K
    arr = np.round(generate_n_rand_probabilities_that_sum_to_m(n=K, m=1, max_prob=max_balls_per_bin / float(N), min_prob=min_balls_per_bin / float(N)) * N)
    while sum(arr) != N:
        while sum(arr) > N:
            arr[np.argmax(arr)] -= 1
        while sum(arr) < N:
            arr[np.argmin(arr)] += 1
    return arr.astype(int)