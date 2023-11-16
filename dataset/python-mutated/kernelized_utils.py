"""Utility methods related to kernelized layers."""
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

def _to_matrix(u):
    if False:
        return 10
    'If input tensor is a vector (i.e., has rank 1), converts it to matrix.'
    u_rank = len(u.shape)
    if u_rank not in [1, 2]:
        raise ValueError('The input tensor should have rank 1 or 2. Given rank: {}'.format(u_rank))
    if u_rank == 1:
        return array_ops.expand_dims(u, 0)
    return u

def _align_matrices(x, y):
    if False:
        while True:
            i = 10
    'Aligns x and y tensors to allow computations over pairs of their rows.'
    x_matrix = _to_matrix(x)
    y_matrix = _to_matrix(y)
    x_shape = x_matrix.shape
    y_shape = y_matrix.shape
    if y_shape[1] != x_shape[1]:
        raise ValueError('The outermost dimensions of the input tensors should match. Given: {} vs {}.'.format(y_shape[1], x_shape[1]))
    x_tile = array_ops.tile(array_ops.expand_dims(x_matrix, 1), [1, y_shape[0], 1])
    y_tile = array_ops.tile(array_ops.expand_dims(y_matrix, 0), [x_shape[0], 1, 1])
    return (x_tile, y_tile)

def inner_product(u, v):
    if False:
        return 10
    u = _to_matrix(u)
    v = _to_matrix(v)
    return math_ops.matmul(u, v, transpose_b=True)

def exact_gaussian_kernel(x, y, stddev):
    if False:
        i = 10
        return i + 15
    "Computes exact Gaussian kernel value(s) for tensors x and y and stddev.\n\n  The Gaussian kernel for vectors u, v is defined as follows:\n       K(u, v) = exp(-||u-v||^2 / (2* stddev^2))\n  where the norm is the l2-norm. x, y can be either vectors or matrices. If they\n  are vectors, they must have the same dimension. If they are matrices, they\n  must have the same number of columns. In the latter case, the method returns\n  (as a matrix) K(u, v) values for all pairs (u, v) where u is a row from x and\n  v is a row from y.\n\n  Args:\n    x: a tensor of rank 1 or 2. It's shape should be either [dim] or [m, dim].\n    y: a tensor of rank 1 or 2. It's shape should be either [dim] or [n, dim].\n    stddev: The width of the Gaussian kernel.\n\n  Returns:\n    A single value (scalar) with shape (1, 1) (if x, y are vectors) or a matrix\n      of shape (m, n) with entries K(u, v) (where K is the Gaussian kernel) for\n      all (u,v) pairs where u, v are rows from x and y respectively.\n\n  Raises:\n    ValueError: if the shapes of x, y are not compatible.\n  "
    (x_aligned, y_aligned) = _align_matrices(x, y)
    diff_squared_l2_norm = math_ops.reduce_sum(math_ops.squared_difference(x_aligned, y_aligned), 2)
    return math_ops.exp(-diff_squared_l2_norm / (2 * stddev * stddev))

def exact_laplacian_kernel(x, y, stddev):
    if False:
        i = 10
        return i + 15
    "Computes exact Laplacian kernel value(s) for tensors x and y using stddev.\n\n  The Laplacian kernel for vectors u, v is defined as follows:\n       K(u, v) = exp(-||u-v|| / stddev)\n  where the norm is the l1-norm. x, y can be either vectors or matrices. If they\n  are vectors, they must have the same dimension. If they are matrices, they\n  must have the same number of columns. In the latter case, the method returns\n  (as a matrix) K(u, v) values for all pairs (u, v) where u is a row from x and\n  v is a row from y.\n\n  Args:\n    x: a tensor of rank 1 or 2. It's shape should be either [dim] or [m, dim].\n    y: a tensor of rank 1 or 2. It's shape should be either [dim] or [n, dim].\n    stddev: The width of the Gaussian kernel.\n\n  Returns:\n    A single value (scalar) with shape (1, 1)  if x, y are vectors or a matrix\n    of shape (m, n) with entries K(u, v) (where K is the Laplacian kernel) for\n    all (u,v) pairs where u, v are rows from x and y respectively.\n\n  Raises:\n    ValueError: if the shapes of x, y are not compatible.\n  "
    (x_aligned, y_aligned) = _align_matrices(x, y)
    diff_l1_norm = math_ops.reduce_sum(math_ops.abs(math_ops.subtract(x_aligned, y_aligned)), 2)
    return math_ops.exp(-diff_l1_norm / stddev)