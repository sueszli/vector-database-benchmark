from typing import Tuple
import numpy as np

def generate_simple_label_matrix(n: int, m: int, cardinality: int, abstain_multiplier: float=1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    'Generate a synthetic label matrix with true parameters and labels.\n\n    This function generates a set of labeling function conditional probability tables,\n    P(LF=l | Y=y), stored as a matrix P, and true labels Y, and then generates the\n    resulting label matrix L.\n\n    Parameters\n    ----------\n    n\n        Number of data points\n    m\n        Number of labeling functions\n    cardinality\n        Cardinality of true labels (i.e. not including abstains)\n    abstain_multiplier\n        Factor to multiply the probability of abstaining by\n\n    Returns\n    -------\n    Tuple[np.ndarray, np.ndarray, np.ndarray]\n        A tuple containing the LF conditional probabilities P,\n        the true labels Y, and the output label matrix L\n    '
    P = np.empty((m, cardinality + 1, cardinality))
    for i in range(m):
        p = np.random.rand(cardinality + 1, cardinality)
        p[1:, :] += (cardinality - 1) * np.eye(cardinality)
        p[0, :] *= abstain_multiplier
        P[i] = p @ np.diag(1 / p.sum(axis=0))
    Y = np.random.choice(cardinality, n)
    L: np.ndarray = np.empty((n, m), dtype=int)
    for i in range(n):
        for j in range(m):
            L[i, j] = np.random.choice(cardinality + 1, p=P[j, :, Y[i]]) - 1
    return (P, Y, L)