"""
Contains mathematical functions relating the latent terms,
``P(given_label)``, ``P(given_label | true_label)``, ``P(true_label | given_label)``, ``P(true_label)``, etc. together.
For every function here, if the inputs are exact, the output is guaranteed to be exact.
Every function herein is the computational equivalent of a mathematical equation having a closed, exact form.
If the inputs are inexact, the error will of course propagate.
Throughout `K` denotes the number of classes in the classification task.
"""
import warnings
import numpy as np
from typing import Tuple
from cleanlab.internal.util import value_counts, clip_values, clip_noise_rates
from cleanlab.internal.constants import TINY_VALUE, CLIPPING_LOWER_BOUND

def compute_ps_py_inv_noise_matrix(labels, noise_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if False:
        i = 10
        return i + 15
    'Compute ``ps := P(labels=k), py := P(true_labels=k)``, and the inverse noise matrix.\n\n    Parameters\n    ----------\n    labels : np.ndarray\n          A discrete vector of noisy labels, i.e. some labels may be erroneous.\n          *Format requirements*: for dataset with `K` classes, labels must be in ``{0,1,...,K-1}``.\n\n    noise_matrix : np.ndarray\n        A conditional probability matrix (of shape ``(K, K)``) of the form ``P(label=k_s|true_label=k_y)`` containing\n        the fraction of examples in every class, labeled as every other class.\n        Assumes columns of noise_matrix sum to 1.'
    ps = value_counts(labels) / float(len(labels))
    (py, inverse_noise_matrix) = compute_py_inv_noise_matrix(ps, noise_matrix)
    return (ps, py, inverse_noise_matrix)

def compute_py_inv_noise_matrix(ps, noise_matrix) -> Tuple[np.ndarray, np.ndarray]:
    if False:
        print('Hello World!')
    'Compute py := P(true_label=k), and the inverse noise matrix.\n\n    Parameters\n    ----------\n    ps : np.ndarray\n        Array of shape ``(K, )`` or ``(1, K)``.\n        The fraction (prior probability) of each observed, NOISY class ``P(labels = k)``.\n\n    noise_matrix : np.ndarray\n        A conditional probability matrix (of shape ``(K, K)``) of the form ``P(label=k_s|true_label=k_y)`` containing\n        the fraction of examples in every class, labeled as every other class.\n        Assumes columns of noise_matrix sum to 1.'
    py = np.linalg.inv(noise_matrix).dot(ps)
    py = clip_values(py, low=CLIPPING_LOWER_BOUND, high=1.0, new_sum=1.0)
    return (py, compute_inv_noise_matrix(py=py, noise_matrix=noise_matrix, ps=ps))

def compute_inv_noise_matrix(py, noise_matrix, *, ps=None) -> np.ndarray:
    if False:
        while True:
            i = 10
    "Compute the inverse noise matrix if py := P(true_label=k) is given.\n\n    Parameters\n    ----------\n    py : np.ndarray (shape (K, 1))\n        The fraction (prior probability) of each TRUE class label, P(true_label = k)\n\n    noise_matrix : np.ndarray\n        A conditional probability matrix (of shape ``(K, K)``) of the form ``P(label=k_s|true_label=k_y)`` containing\n        the fraction of examples in every class, labeled as every other class.\n        Assumes columns of noise_matrix sum to 1.\n\n    ps : np.ndarray\n        Array of shape ``(K, 1)`` containing the fraction (prior probability) of each NOISY given label, ``P(labels = k)``.\n        `ps` is easily computable from py and should only be provided if it has already been precomputed, to increase code efficiency.\n\n    Examples\n    --------\n    For loop based implementation:\n\n    .. code:: python\n\n        # Number of classes\n        K = len(py)\n\n        # 'ps' is p(labels=k) = noise_matrix * p(true_labels=k)\n        # because in *vector computation*: P(label=k|true_label=k) * p(true_label=k) = P(label=k)\n        if ps is None:\n            ps = noise_matrix.dot(py)\n\n        # Estimate the (K, K) inverse noise matrix P(true_label = k_y | label = k_s)\n        inverse_noise_matrix = np.empty(shape=(K,K))\n        # k_s is the class value k of noisy label `label == k`\n        for k_s in range(K):\n            # k_y is the (guessed) class value k of true label y\n            for k_y in range(K):\n                # P(true_label|label) = P(label|y) * P(true_label) / P(labels)\n                inverse_noise_matrix[k_y][k_s] = noise_matrix[k_s][k_y] *                                                  py[k_y] / ps[k_s]\n    "
    joint = noise_matrix * py
    ps = joint.sum(axis=1) if ps is None else ps
    inverse_noise_matrix = joint.T / np.clip(ps, a_min=TINY_VALUE, a_max=None)
    return clip_noise_rates(inverse_noise_matrix)

def compute_noise_matrix_from_inverse(ps, inverse_noise_matrix, *, py=None) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    "Compute the noise matrix ``P(label=k_s|true_label=k_y)``.\n\n    Parameters\n    ----------\n    py : np.ndarray\n        Array of shape ``(K, 1)`` containing the fraction (prior probability) of each TRUE class label, ``P(true_label = k)``.\n\n    inverse_noise_matrix : np.ndarray\n        A conditional probability matrix (of shape ``(K, K)``) of the form P(true_label=k_y|label=k_s) representing\n        the estimated fraction observed examples in each class k_s, that are\n        mislabeled examples from every other class k_y. If None, the\n        inverse_noise_matrix will be computed from pred_probs and labels.\n        Assumes columns of inverse_noise_matrix sum to 1.\n\n    ps : np.ndarray\n        Array of shape ``(K, 1)`` containing the fraction (prior probability) of each observed NOISY label, P(labels = k).\n        `ps` is easily computable from `py` and should only be provided if it has already been precomputed, to increase code efficiency.\n\n    Returns\n    -------\n    noise_matrix : np.ndarray\n        Array of shape ``(K, K)``, where `K` = number of classes, whose columns sum to 1.\n        A conditional probability matrix of the form ``P(label=k_s|true_label=k_y)`` containing\n        the fraction of examples in every class, labeled as every other class.\n\n    Examples\n    --------\n    For loop based implementation:\n\n    .. code:: python\n\n        # Number of classes labels\n        K = len(ps)\n\n        # 'py' is p(true_label=k) = inverse_noise_matrix * p(true_label=k)\n        # because in *vector computation*: P(true_label=k|label=k) * p(label=k) = P(true_label=k)\n        if py is None:\n            py = inverse_noise_matrix.dot(ps)\n\n        # Estimate the (K, K) noise matrix P(labels = k_s | true_labels = k_y)\n        noise_matrix = np.empty(shape=(K,K))\n        # k_s is the class value k of noisy label `labels == k`\n        for k_s in range(K):\n            # k_y is the (guessed) class value k of true label y\n            for k_y in range(K):\n                # P(labels|y) = P(true_label|labels) * P(labels) / P(true_label)\n                noise_matrix[k_s][k_y] = inverse_noise_matrix[k_y][k_s] *                                          ps[k_s] / py[k_y]\n\n    "
    joint = (inverse_noise_matrix * ps).T
    py = joint.sum(axis=0) if py is None else py
    noise_matrix = joint / np.clip(py, a_min=TINY_VALUE, a_max=None)
    return clip_noise_rates(noise_matrix)

def compute_py(ps, noise_matrix, inverse_noise_matrix, *, py_method='cnt', true_labels_class_counts=None) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Compute ``py := P(true_labels=k)`` from ``ps := P(labels=k)``, `noise_matrix`, and\n    `inverse_noise_matrix`.\n\n    This method is ** ROBUST ** when ``py_method = \'cnt\'``\n    It may work well even when the noise matrices are estimated\n    poorly by using the diagonals of the matrices\n    instead of all the probabilities in the entire matrix.\n\n    Parameters\n    ----------\n    ps : np.ndarray\n        Array of shape ``(K, )`` or ``(1, K)`` containing the fraction (prior probability) of each observed, noisy label, P(labels = k)\n\n    noise_matrix : np.ndarray\n        A conditional probability matrix ( of shape ``(K, K)``) of the form ``P(label=k_s|true_label=k_y)`` containing\n        the fraction of examples in every class, labeled as every other class.\n        Assumes columns of noise_matrix sum to 1.\n\n    inverse_noise_matrix : np.ndarray of shape (K, K), K = number of classes\n        A conditional probability matrix ( of shape ``(K, K)``) of the form ``P(true_label=k_y|label=k_s)`` representing\n        the estimated fraction observed examples in each class `k_s`, that are\n        mislabeled examples from every other class `k_y`. If ``None``, the\n        inverse_noise_matrix will be computed from `pred_probs` and `labels`.\n        Assumes columns of `inverse_noise_matrix` sum to 1.\n\n    py_method : str (Options: ["cnt", "eqn", "marginal", "marginal_ps"])\n        How to compute the latent prior ``p(true_label=k)``. Default is "cnt" as it often\n        works well even when the noise matrices are estimated poorly by using\n        the matrix diagonals instead of all the probabilities.\n\n    true_labels_class_counts : np.ndarray\n        Array of shape ``(K, )`` or ``(1, K)`` containing the marginal counts of the confident joint\n        (like ``cj.sum(axis = 0)``).\n\n    Returns\n    -------\n    py : np.ndarray\n        Array of shape ``(K, )`` or ``(1, K)``.\n        The fraction (prior probability) of each TRUE class label, ``P(true_label = k)``.'
    if len(np.shape(ps)) > 2 or (len(np.shape(ps)) == 2 and np.shape(ps)[0] != 1):
        w = 'Input parameter np.ndarray ps has shape ' + str(np.shape(ps))
        w += ', but shape should be (K, ) or (1, K)'
        warnings.warn(w)
    if py_method == 'marginal' and true_labels_class_counts is None:
        msg = 'py_method == "marginal" requires true_labels_class_counts, but true_labels_class_counts is None. '
        msg += ' Provide parameter true_labels_class_counts.'
        raise ValueError(msg)
    if py_method == 'cnt':
        py = inverse_noise_matrix.diagonal() / np.clip(noise_matrix.diagonal(), a_min=TINY_VALUE, a_max=None) * ps
    elif py_method == 'eqn':
        py = np.linalg.inv(noise_matrix).dot(ps)
    elif py_method == 'marginal':
        py = true_labels_class_counts / np.clip(float(sum(true_labels_class_counts)), a_min=TINY_VALUE, a_max=None)
    elif py_method == 'marginal_ps':
        py = np.dot(inverse_noise_matrix, ps)
    else:
        err = 'py_method {}'.format(py_method)
        err += ' should be in [cnt, eqn, marginal, marginal_ps]'
        raise ValueError(err)
    py = clip_values(py, low=CLIPPING_LOWER_BOUND, high=1.0, new_sum=1.0)
    return py

def compute_pyx(pred_probs, noise_matrix, inverse_noise_matrix):
    if False:
        print('Hello World!')
    'Compute ``pyx := P(true_label=k|x)`` from ``pred_probs := P(label=k|x)``, `noise_matrix` and\n    `inverse_noise_matrix`.\n\n    This method is ROBUST - meaning it works well even when the\n    noise matrices are estimated poorly by only using the diagonals of the\n    matrices which tend to be easy to estimate correctly.\n\n    Parameters\n    ----------\n    pred_probs : np.ndarray\n        ``P(label=k|x)`` is a ``(N x K)`` matrix with K model-predicted probabilities.\n        Each row of this matrix corresponds to an example `x` and contains the model-predicted\n        probabilities that `x` belongs to each possible class.\n        The columns must be ordered such that these probabilities correspond to class 0,1,2,...\n        `pred_probs` should have been computed using 3 (or higher) fold cross-validation.\n\n    noise_matrix : np.ndarray\n        A conditional probability matrix (of shape ``(K, K)``) of the form ``P(label=k_s|true_label=k_y)`` containing\n        the fraction of examples in every class, labeled as every other class.\n        Assumes columns of `noise_matrix` sum to 1.\n\n    inverse_noise_matrix : np.ndarray\n        A conditional probability matrix (of shape ``(K, K)``)  of the form ``P(true_label=k_y|label=k_s)`` representing\n        the estimated fraction observed examples in each class `k_s`, that are\n        mislabeled examples from every other class `k_y`. If None, the\n        inverse_noise_matrix will be computed from `pred_probs` and `labels`.\n        Assumes columns of `inverse_noise_matrix` sum to 1.\n\n    Returns\n    -------\n    pyx : np.ndarray\n        ``P(true_label=k|x)`` is a  ``(N, K)`` matrix of model-predicted probabilities.\n        Each row of this matrix corresponds to an example `x` and contains the model-predicted\n        probabilities that `x` belongs to each possible class.\n        The columns must be ordered such that these probabilities correspond to class 0,1,2,...\n        `pred_probs` should have been computed using 3 (or higher) fold cross-validation.'
    if len(np.shape(pred_probs)) != 2:
        raise ValueError("Input parameter np.ndarray 'pred_probs' has shape " + str(np.shape(pred_probs)) + ', but shape should be (N, K)')
    pyx = pred_probs * inverse_noise_matrix.diagonal() / np.clip(noise_matrix.diagonal(), a_min=TINY_VALUE, a_max=None)
    return np.apply_along_axis(func1d=clip_values, axis=1, arr=pyx, **{'low': 0.0, 'high': 1.0, 'new_sum': 1.0})