import warnings
warnings.warn('Importing from numpy.matlib is deprecated since 1.19.0. The matrix subclass is not the recommended way to represent matrices or deal with linear algebra (see https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html). Please adjust your code to use regular ndarray. ', PendingDeprecationWarning, stacklevel=2)
import numpy as np
from numpy.matrixlib.defmatrix import matrix, asmatrix
from numpy import *
__version__ = np.__version__
__all__ = np.__all__[:]
__all__ += ['rand', 'randn', 'repmat']

def empty(shape, dtype=None, order='C'):
    if False:
        print('Hello World!')
    "Return a new matrix of given shape and type, without initializing entries.\n\n    Parameters\n    ----------\n    shape : int or tuple of int\n        Shape of the empty matrix.\n    dtype : data-type, optional\n        Desired output data-type.\n    order : {'C', 'F'}, optional\n        Whether to store multi-dimensional data in row-major\n        (C-style) or column-major (Fortran-style) order in\n        memory.\n\n    See Also\n    --------\n    empty_like, zeros\n\n    Notes\n    -----\n    `empty`, unlike `zeros`, does not set the matrix values to zero,\n    and may therefore be marginally faster.  On the other hand, it requires\n    the user to manually set all the values in the array, and should be\n    used with caution.\n\n    Examples\n    --------\n    >>> import numpy.matlib\n    >>> np.matlib.empty((2, 2))    # filled with random data\n    matrix([[  6.76425276e-320,   9.79033856e-307], # random\n            [  7.39337286e-309,   3.22135945e-309]])\n    >>> np.matlib.empty((2, 2), dtype=int)\n    matrix([[ 6600475,        0], # random\n            [ 6586976, 22740995]])\n\n    "
    return ndarray.__new__(matrix, shape, dtype, order=order)

def ones(shape, dtype=None, order='C'):
    if False:
        while True:
            i = 10
    "\n    Matrix of ones.\n\n    Return a matrix of given shape and type, filled with ones.\n\n    Parameters\n    ----------\n    shape : {sequence of ints, int}\n        Shape of the matrix\n    dtype : data-type, optional\n        The desired data-type for the matrix, default is np.float64.\n    order : {'C', 'F'}, optional\n        Whether to store matrix in C- or Fortran-contiguous order,\n        default is 'C'.\n\n    Returns\n    -------\n    out : matrix\n        Matrix of ones of given shape, dtype, and order.\n\n    See Also\n    --------\n    ones : Array of ones.\n    matlib.zeros : Zero matrix.\n\n    Notes\n    -----\n    If `shape` has length one i.e. ``(N,)``, or is a scalar ``N``,\n    `out` becomes a single row matrix of shape ``(1,N)``.\n\n    Examples\n    --------\n    >>> np.matlib.ones((2,3))\n    matrix([[1.,  1.,  1.],\n            [1.,  1.,  1.]])\n\n    >>> np.matlib.ones(2)\n    matrix([[1.,  1.]])\n\n    "
    a = ndarray.__new__(matrix, shape, dtype, order=order)
    a.fill(1)
    return a

def zeros(shape, dtype=None, order='C'):
    if False:
        while True:
            i = 10
    "\n    Return a matrix of given shape and type, filled with zeros.\n\n    Parameters\n    ----------\n    shape : int or sequence of ints\n        Shape of the matrix\n    dtype : data-type, optional\n        The desired data-type for the matrix, default is float.\n    order : {'C', 'F'}, optional\n        Whether to store the result in C- or Fortran-contiguous order,\n        default is 'C'.\n\n    Returns\n    -------\n    out : matrix\n        Zero matrix of given shape, dtype, and order.\n\n    See Also\n    --------\n    numpy.zeros : Equivalent array function.\n    matlib.ones : Return a matrix of ones.\n\n    Notes\n    -----\n    If `shape` has length one i.e. ``(N,)``, or is a scalar ``N``,\n    `out` becomes a single row matrix of shape ``(1,N)``.\n\n    Examples\n    --------\n    >>> import numpy.matlib\n    >>> np.matlib.zeros((2, 3))\n    matrix([[0.,  0.,  0.],\n            [0.,  0.,  0.]])\n\n    >>> np.matlib.zeros(2)\n    matrix([[0.,  0.]])\n\n    "
    a = ndarray.__new__(matrix, shape, dtype, order=order)
    a.fill(0)
    return a

def identity(n, dtype=None):
    if False:
        print('Hello World!')
    '\n    Returns the square identity matrix of given size.\n\n    Parameters\n    ----------\n    n : int\n        Size of the returned identity matrix.\n    dtype : data-type, optional\n        Data-type of the output. Defaults to ``float``.\n\n    Returns\n    -------\n    out : matrix\n        `n` x `n` matrix with its main diagonal set to one,\n        and all other elements zero.\n\n    See Also\n    --------\n    numpy.identity : Equivalent array function.\n    matlib.eye : More general matrix identity function.\n\n    Examples\n    --------\n    >>> import numpy.matlib\n    >>> np.matlib.identity(3, dtype=int)\n    matrix([[1, 0, 0],\n            [0, 1, 0],\n            [0, 0, 1]])\n\n    '
    a = array([1] + n * [0], dtype=dtype)
    b = empty((n, n), dtype=dtype)
    b.flat = a
    return b

def eye(n, M=None, k=0, dtype=float, order='C'):
    if False:
        while True:
            i = 10
    "\n    Return a matrix with ones on the diagonal and zeros elsewhere.\n\n    Parameters\n    ----------\n    n : int\n        Number of rows in the output.\n    M : int, optional\n        Number of columns in the output, defaults to `n`.\n    k : int, optional\n        Index of the diagonal: 0 refers to the main diagonal,\n        a positive value refers to an upper diagonal,\n        and a negative value to a lower diagonal.\n    dtype : dtype, optional\n        Data-type of the returned matrix.\n    order : {'C', 'F'}, optional\n        Whether the output should be stored in row-major (C-style) or\n        column-major (Fortran-style) order in memory.\n\n        .. versionadded:: 1.14.0\n\n    Returns\n    -------\n    I : matrix\n        A `n` x `M` matrix where all elements are equal to zero,\n        except for the `k`-th diagonal, whose values are equal to one.\n\n    See Also\n    --------\n    numpy.eye : Equivalent array function.\n    identity : Square identity matrix.\n\n    Examples\n    --------\n    >>> import numpy.matlib\n    >>> np.matlib.eye(3, k=1, dtype=float)\n    matrix([[0.,  1.,  0.],\n            [0.,  0.,  1.],\n            [0.,  0.,  0.]])\n\n    "
    return asmatrix(np.eye(n, M=M, k=k, dtype=dtype, order=order))

def rand(*args):
    if False:
        print('Hello World!')
    '\n    Return a matrix of random values with given shape.\n\n    Create a matrix of the given shape and propagate it with\n    random samples from a uniform distribution over ``[0, 1)``.\n\n    Parameters\n    ----------\n    \\*args : Arguments\n        Shape of the output.\n        If given as N integers, each integer specifies the size of one\n        dimension.\n        If given as a tuple, this tuple gives the complete shape.\n\n    Returns\n    -------\n    out : ndarray\n        The matrix of random values with shape given by `\\*args`.\n\n    See Also\n    --------\n    randn, numpy.random.RandomState.rand\n\n    Examples\n    --------\n    >>> np.random.seed(123)\n    >>> import numpy.matlib\n    >>> np.matlib.rand(2, 3)\n    matrix([[0.69646919, 0.28613933, 0.22685145],\n            [0.55131477, 0.71946897, 0.42310646]])\n    >>> np.matlib.rand((2, 3))\n    matrix([[0.9807642 , 0.68482974, 0.4809319 ],\n            [0.39211752, 0.34317802, 0.72904971]])\n\n    If the first argument is a tuple, other arguments are ignored:\n\n    >>> np.matlib.rand((2, 3), 4)\n    matrix([[0.43857224, 0.0596779 , 0.39804426],\n            [0.73799541, 0.18249173, 0.17545176]])\n\n    '
    if isinstance(args[0], tuple):
        args = args[0]
    return asmatrix(np.random.rand(*args))

def randn(*args):
    if False:
        while True:
            i = 10
    '\n    Return a random matrix with data from the "standard normal" distribution.\n\n    `randn` generates a matrix filled with random floats sampled from a\n    univariate "normal" (Gaussian) distribution of mean 0 and variance 1.\n\n    Parameters\n    ----------\n    \\*args : Arguments\n        Shape of the output.\n        If given as N integers, each integer specifies the size of one\n        dimension. If given as a tuple, this tuple gives the complete shape.\n\n    Returns\n    -------\n    Z : matrix of floats\n        A matrix of floating-point samples drawn from the standard normal\n        distribution.\n\n    See Also\n    --------\n    rand, numpy.random.RandomState.randn\n\n    Notes\n    -----\n    For random samples from the normal distribution with mean ``mu`` and\n    standard deviation ``sigma``, use::\n\n        sigma * np.matlib.randn(...) + mu\n\n    Examples\n    --------\n    >>> np.random.seed(123)\n    >>> import numpy.matlib\n    >>> np.matlib.randn(1)\n    matrix([[-1.0856306]])\n    >>> np.matlib.randn(1, 2, 3)\n    matrix([[ 0.99734545,  0.2829785 , -1.50629471],\n            [-0.57860025,  1.65143654, -2.42667924]])\n\n    Two-by-four matrix of samples from the normal distribution with\n    mean 3 and standard deviation 2.5:\n\n    >>> 2.5 * np.matlib.randn((2, 4)) + 3\n    matrix([[1.92771843, 6.16484065, 0.83314899, 1.30278462],\n            [2.76322758, 6.72847407, 1.40274501, 1.8900451 ]])\n\n    '
    if isinstance(args[0], tuple):
        args = args[0]
    return asmatrix(np.random.randn(*args))

def repmat(a, m, n):
    if False:
        print('Hello World!')
    '\n    Repeat a 0-D to 2-D array or matrix MxN times.\n\n    Parameters\n    ----------\n    a : array_like\n        The array or matrix to be repeated.\n    m, n : int\n        The number of times `a` is repeated along the first and second axes.\n\n    Returns\n    -------\n    out : ndarray\n        The result of repeating `a`.\n\n    Examples\n    --------\n    >>> import numpy.matlib\n    >>> a0 = np.array(1)\n    >>> np.matlib.repmat(a0, 2, 3)\n    array([[1, 1, 1],\n           [1, 1, 1]])\n\n    >>> a1 = np.arange(4)\n    >>> np.matlib.repmat(a1, 2, 2)\n    array([[0, 1, 2, 3, 0, 1, 2, 3],\n           [0, 1, 2, 3, 0, 1, 2, 3]])\n\n    >>> a2 = np.asmatrix(np.arange(6).reshape(2, 3))\n    >>> np.matlib.repmat(a2, 2, 3)\n    matrix([[0, 1, 2, 0, 1, 2, 0, 1, 2],\n            [3, 4, 5, 3, 4, 5, 3, 4, 5],\n            [0, 1, 2, 0, 1, 2, 0, 1, 2],\n            [3, 4, 5, 3, 4, 5, 3, 4, 5]])\n\n    '
    a = asanyarray(a)
    ndim = a.ndim
    if ndim == 0:
        (origrows, origcols) = (1, 1)
    elif ndim == 1:
        (origrows, origcols) = (1, a.shape[0])
    else:
        (origrows, origcols) = a.shape
    rows = origrows * m
    cols = origcols * n
    c = a.reshape(1, a.size).repeat(m, 0).reshape(rows, origcols).repeat(n, 0)
    return c.reshape(rows, cols)