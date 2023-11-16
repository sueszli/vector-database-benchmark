"""Code for doing logistic regressions (DEPRECATED).

Classes:
 - LogisticRegression    Holds information for a LogisticRegression classifier.

Functions:
 - train        Train a new classifier.
 - calculate    Calculate the probabilities of each class, given an observation.
 - classify     Classify an observation into a class.

This module has been deprecated, please consider an alternative like scikit-learn
insead.
"""
import warnings
from Bio import BiopythonDeprecationWarning
warnings.warn("The 'Bio.LogisticRegression' module is deprecated and will be removed in a future release of Biopython. Consider using scikit-learn instead.", BiopythonDeprecationWarning)
try:
    import numpy as np
    import numpy.linalg
except ImportError:
    from Bio import MissingPythonDependencyError
    raise MissingPythonDependencyError('Please install NumPy if you want to use Bio.LogisticRegression. See http://www.numpy.org/') from None

class LogisticRegression:
    """Holds information necessary to do logistic regression classification.

    Attributes:
     - beta - List of the weights for each dimension.

    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        'Initialize the class.'
        self.beta = []

def train(xs, ys, update_fn=None, typecode=None):
    if False:
        i = 10
        return i + 15
    'Train a logistic regression classifier on a training set.\n\n    Argument xs is a list of observations and ys is a list of the class\n    assignments, which should be 0 or 1.  xs and ys should contain the\n    same number of elements.  update_fn is an optional callback function\n    that takes as parameters that iteration number and log likelihood.\n    '
    if len(xs) != len(ys):
        raise ValueError('xs and ys should be the same length.')
    classes = set(ys)
    if classes != {0, 1}:
        raise ValueError("Classes should be 0's and 1's")
    if typecode is None:
        typecode = 'd'
    (N, ndims) = (len(xs), len(xs[0]) + 1)
    if N == 0 or ndims == 1:
        raise ValueError('No observations or observation of 0 dimension.')
    X = np.ones((N, ndims), typecode)
    X[:, 1:] = xs
    Xt = np.transpose(X)
    y = np.asarray(ys, typecode)
    beta = np.zeros(ndims, typecode)
    MAX_ITERATIONS = 500
    CONVERGE_THRESHOLD = 0.01
    stepsize = 1.0
    i = 0
    old_beta = old_llik = None
    while i < MAX_ITERATIONS:
        ebetaX = np.exp(np.dot(beta, Xt))
        p = ebetaX / (1 + ebetaX)
        logp = y * np.log(p) + (1 - y) * np.log(1 - p)
        llik = sum(logp)
        if update_fn is not None:
            update_fn(iter, llik)
        if old_llik is not None:
            if llik < old_llik:
                stepsize /= 2.0
                beta = old_beta
            if np.fabs(llik - old_llik) <= CONVERGE_THRESHOLD:
                break
        (old_llik, old_beta) = (llik, beta)
        i += 1
        W = np.identity(N) * p
        Xtyp = np.dot(Xt, y - p)
        XtWX = np.dot(np.dot(Xt, W), X)
        delta = numpy.linalg.solve(XtWX, Xtyp)
        if np.fabs(stepsize - 1.0) > 0.001:
            delta *= stepsize
        beta += delta
    else:
        raise RuntimeError("Didn't converge.")
    lr = LogisticRegression()
    lr.beta = list(beta)
    return lr

def calculate(lr, x):
    if False:
        for i in range(10):
            print('nop')
    'Calculate the probability for each class.\n\n    Arguments:\n     - lr is a LogisticRegression object.\n     - x is the observed data.\n\n    Returns a list of the probability that it fits each class.\n    '
    x = np.asarray([1.0] + x)
    ebetaX = np.exp(np.dot(lr.beta, x))
    p = ebetaX / (1 + ebetaX)
    return [1 - p, p]

def classify(lr, x):
    if False:
        print('Hello World!')
    'Classify an observation into a class.'
    probs = calculate(lr, x)
    if probs[0] > probs[1]:
        return 0
    return 1