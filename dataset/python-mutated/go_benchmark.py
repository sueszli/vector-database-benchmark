import numpy as np
from numpy import abs, asarray
from ..common import safe_import

class Benchmark:
    """
    Defines a global optimization benchmark problem.

    This abstract class defines the basic structure of a global
    optimization problem. Subclasses should implement the ``fun`` method
    for a particular optimization problem.

    Attributes
    ----------
    N : int
        The dimensionality of the problem.
    bounds : sequence
        The lower/upper bounds to be used for minimizing the problem.
        This a list of (lower, upper) tuples that contain the lower and upper
        bounds for the problem.  The problem should not be asked for evaluation
        outside these bounds. ``len(bounds) == N``.
    xmin : sequence
        The lower bounds for the problem
    xmax : sequence
        The upper bounds for the problem
    fglob : float
        The global minimum of the evaluated function.
    global_optimum : sequence
        A list of vectors that provide the locations of the global minimum.
        Note that some problems have multiple global minima, not all of which
        may be listed.
    nfev : int
        the number of function evaluations that the object has been asked to
        calculate.
    change_dimensionality : bool
        Whether we can change the benchmark function `x` variable length (i.e.,
        the dimensionality of the problem)
    custom_bounds : sequence
        a list of tuples that contain lower/upper bounds for use in plotting.
    """

    def __init__(self, dimensions):
        if False:
            return 10
        '\n        Initialises the problem\n\n        Parameters\n        ----------\n\n        dimensions : int\n            The dimensionality of the problem\n        '
        self._dimensions = dimensions
        self.nfev = 0
        self.fglob = np.nan
        self.global_optimum = None
        self.change_dimensionality = False
        self.custom_bounds = None

    def __str__(self):
        if False:
            return 10
        return '{0} ({1} dimensions)'.format(self.__class__.__name__, self.N)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return self.__class__.__name__

    def initial_vector(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Random initialisation for the benchmark problem.\n\n        Returns\n        -------\n        x : sequence\n            a vector of length ``N`` that contains random floating point\n            numbers that lie between the lower and upper bounds for a given\n            parameter.\n        '
        return asarray([np.random.uniform(l, u) for (l, u) in self.bounds])

    def success(self, x, tol=1e-05):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests if a candidate solution at the global minimum.\n        The default test is\n\n        Parameters\n        ----------\n        x : sequence\n            The candidate vector for testing if the global minimum has been\n            reached. Must have ``len(x) == self.N``\n        tol : float\n            The evaluated function and known global minimum must differ by less\n            than this amount to be at a global minimum.\n\n        Returns\n        -------\n        bool : is the candidate vector at the global minimum?\n        '
        val = self.fun(asarray(x))
        if abs(val - self.fglob) < tol:
            return True
        bounds = np.asarray(self.bounds, dtype=np.float64)
        if np.any(x > bounds[:, 1]):
            return False
        if np.any(x < bounds[:, 0]):
            return False
        if val < self.fglob:
            raise ValueError('Found a lower global minimum', x, val, self.fglob)
        return False

    def fun(self, x):
        if False:
            return 10
        '\n        Evaluation of the benchmark function.\n\n        Parameters\n        ----------\n        x : sequence\n            The candidate vector for evaluating the benchmark problem. Must\n            have ``len(x) == self.N``.\n\n        Returns\n        -------\n        val : float\n              the evaluated benchmark function\n        '
        raise NotImplementedError

    def change_dimensions(self, ndim):
        if False:
            return 10
        '\n        Changes the dimensionality of the benchmark problem\n\n        The dimensionality will only be changed if the problem is suitable\n\n        Parameters\n        ----------\n        ndim : int\n               The new dimensionality for the problem.\n        '
        if self.change_dimensionality:
            self._dimensions = ndim
        else:
            raise ValueError('dimensionality cannot be changed for thisproblem')

    @property
    def bounds(self):
        if False:
            print('Hello World!')
        '\n        The lower/upper bounds to be used for minimizing the problem.\n        This a list of (lower, upper) tuples that contain the lower and upper\n        bounds for the problem.  The problem should not be asked for evaluation\n        outside these bounds. ``len(bounds) == N``.\n        '
        if self.change_dimensionality:
            return [self._bounds[0]] * self.N
        else:
            return self._bounds

    @property
    def N(self):
        if False:
            while True:
                i = 10
        '\n        The dimensionality of the problem.\n\n        Returns\n        -------\n        N : int\n            The dimensionality of the problem\n        '
        return self._dimensions

    @property
    def xmin(self):
        if False:
            while True:
                i = 10
        '\n        The lower bounds for the problem\n\n        Returns\n        -------\n        xmin : sequence\n            The lower bounds for the problem\n        '
        return asarray([b[0] for b in self.bounds])

    @property
    def xmax(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The upper bounds for the problem\n\n        Returns\n        -------\n        xmax : sequence\n            The upper bounds for the problem\n        '
        return asarray([b[1] for b in self.bounds])