"""A cache for storing small matrices in multiple formats."""
from sympy.core.numbers import I, Rational, pi
from sympy.core.power import Pow
from sympy.functions.elementary.exponential import exp
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.matrixutils import to_sympy, to_numpy, to_scipy_sparse

class MatrixCache:
    """A cache for small matrices in different formats.

    This class takes small matrices in the standard ``sympy.Matrix`` format,
    and then converts these to both ``numpy.matrix`` and
    ``scipy.sparse.csr_matrix`` matrices. These matrices are then stored for
    future recovery.
    """

    def __init__(self, dtype='complex'):
        if False:
            print('Hello World!')
        self._cache = {}
        self.dtype = dtype

    def cache_matrix(self, name, m):
        if False:
            i = 10
            return i + 15
        'Cache a matrix by its name.\n\n        Parameters\n        ----------\n        name : str\n            A descriptive name for the matrix, like "identity2".\n        m : list of lists\n            The raw matrix data as a SymPy Matrix.\n        '
        try:
            self._sympy_matrix(name, m)
        except ImportError:
            pass
        try:
            self._numpy_matrix(name, m)
        except ImportError:
            pass
        try:
            self._scipy_sparse_matrix(name, m)
        except ImportError:
            pass

    def get_matrix(self, name, format):
        if False:
            for i in range(10):
                print('nop')
        'Get a cached matrix by name and format.\n\n        Parameters\n        ----------\n        name : str\n            A descriptive name for the matrix, like "identity2".\n        format : str\n            The format desired (\'sympy\', \'numpy\', \'scipy.sparse\')\n        '
        m = self._cache.get((name, format))
        if m is not None:
            return m
        raise NotImplementedError('Matrix with name %s and format %s is not available.' % (name, format))

    def _store_matrix(self, name, format, m):
        if False:
            i = 10
            return i + 15
        self._cache[name, format] = m

    def _sympy_matrix(self, name, m):
        if False:
            for i in range(10):
                print('nop')
        self._store_matrix(name, 'sympy', to_sympy(m))

    def _numpy_matrix(self, name, m):
        if False:
            print('Hello World!')
        m = to_numpy(m, dtype=self.dtype)
        self._store_matrix(name, 'numpy', m)

    def _scipy_sparse_matrix(self, name, m):
        if False:
            i = 10
            return i + 15
        m = to_scipy_sparse(m, dtype=self.dtype)
        self._store_matrix(name, 'scipy.sparse', m)
sqrt2_inv = Pow(2, Rational(-1, 2), evaluate=False)
matrix_cache = MatrixCache()
matrix_cache.cache_matrix('eye2', Matrix([[1, 0], [0, 1]]))
matrix_cache.cache_matrix('op11', Matrix([[0, 0], [0, 1]]))
matrix_cache.cache_matrix('op00', Matrix([[1, 0], [0, 0]]))
matrix_cache.cache_matrix('op10', Matrix([[0, 0], [1, 0]]))
matrix_cache.cache_matrix('op01', Matrix([[0, 1], [0, 0]]))
matrix_cache.cache_matrix('X', Matrix([[0, 1], [1, 0]]))
matrix_cache.cache_matrix('Y', Matrix([[0, -I], [I, 0]]))
matrix_cache.cache_matrix('Z', Matrix([[1, 0], [0, -1]]))
matrix_cache.cache_matrix('S', Matrix([[1, 0], [0, I]]))
matrix_cache.cache_matrix('T', Matrix([[1, 0], [0, exp(I * pi / 4)]]))
matrix_cache.cache_matrix('H', sqrt2_inv * Matrix([[1, 1], [1, -1]]))
matrix_cache.cache_matrix('Hsqrt2', Matrix([[1, 1], [1, -1]]))
matrix_cache.cache_matrix('SWAP', Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))
matrix_cache.cache_matrix('ZX', sqrt2_inv * Matrix([[1, 1], [1, -1]]))
matrix_cache.cache_matrix('ZY', Matrix([[I, 0], [0, -I]]))