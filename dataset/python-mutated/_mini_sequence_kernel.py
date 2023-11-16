import numpy as np
from sklearn.base import clone
from sklearn.gaussian_process.kernels import GenericKernelMixin, Hyperparameter, Kernel, StationaryKernelMixin

class MiniSeqKernel(GenericKernelMixin, StationaryKernelMixin, Kernel):
    """
    A minimal (but valid) convolutional kernel for sequences of variable
    length.
    """

    def __init__(self, baseline_similarity=0.5, baseline_similarity_bounds=(1e-05, 1)):
        if False:
            i = 10
            return i + 15
        self.baseline_similarity = baseline_similarity
        self.baseline_similarity_bounds = baseline_similarity_bounds

    @property
    def hyperparameter_baseline_similarity(self):
        if False:
            i = 10
            return i + 15
        return Hyperparameter('baseline_similarity', 'numeric', self.baseline_similarity_bounds)

    def _f(self, s1, s2):
        if False:
            return 10
        return sum([1.0 if c1 == c2 else self.baseline_similarity for c1 in s1 for c2 in s2])

    def _g(self, s1, s2):
        if False:
            i = 10
            return i + 15
        return sum([0.0 if c1 == c2 else 1.0 for c1 in s1 for c2 in s2])

    def __call__(self, X, Y=None, eval_gradient=False):
        if False:
            return 10
        if Y is None:
            Y = X
        if eval_gradient:
            return (np.array([[self._f(x, y) for y in Y] for x in X]), np.array([[[self._g(x, y)] for y in Y] for x in X]))
        else:
            return np.array([[self._f(x, y) for y in Y] for x in X])

    def diag(self, X):
        if False:
            i = 10
            return i + 15
        return np.array([self._f(x, x) for x in X])

    def clone_with_theta(self, theta):
        if False:
            i = 10
            return i + 15
        cloned = clone(self)
        cloned.theta = theta
        return cloned