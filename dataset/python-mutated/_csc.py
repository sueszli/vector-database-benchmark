try:
    import scipy.sparse
    _scipy_available = True
except ImportError:
    _scipy_available = False
import cupy
from cupy_backends.cuda.api import driver
from cupy_backends.cuda.api import runtime
import cupyx.scipy.sparse
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _compressed

class csc_matrix(_compressed._compressed_sparse_matrix):
    """Compressed Sparse Column matrix.

    This can be instantiated in several ways.

    ``csc_matrix(D)``
        ``D`` is a rank-2 :class:`cupy.ndarray`.
    ``csc_matrix(S)``
        ``S`` is another sparse matrix. It is equivalent to ``S.tocsc()``.
    ``csc_matrix((M, N), [dtype])``
        It constructs an empty matrix whose shape is ``(M, N)``. Default dtype
        is float64.
    ``csc_matrix((data, (row, col)))``
        All ``data``, ``row`` and ``col`` are one-dimenaional
        :class:`cupy.ndarray`.
    ``csc_matrix((data, indices, indptr))``
        All ``data``, ``indices`` and ``indptr`` are one-dimenaional
        :class:`cupy.ndarray`.

    Args:
        arg1: Arguments for the initializer.
        shape (tuple): Shape of a matrix. Its length must be two.
        dtype: Data type. It must be an argument of :class:`numpy.dtype`.
        copy (bool): If ``True``, copies of given arrays are always used.

    .. seealso::
        :class:`scipy.sparse.csc_matrix`

    """
    format = 'csc'

    def get(self, stream=None):
        if False:
            i = 10
            return i + 15
        'Returns a copy of the array on host memory.\n\n        .. warning::\n           You need to install SciPy to use this method.\n\n        Args:\n            stream (cupy.cuda.Stream): CUDA stream object. If it is given, the\n                copy runs asynchronously. Otherwise, the copy is synchronous.\n\n        Returns:\n            scipy.sparse.csc_matrix: Copy of the array on host memory.\n\n        '
        if not _scipy_available:
            raise RuntimeError('scipy is not available')
        data = self.data.get(stream)
        indices = self.indices.get(stream)
        indptr = self.indptr.get(stream)
        return scipy.sparse.csc_matrix((data, indices, indptr), shape=self._shape)

    def _convert_dense(self, x):
        if False:
            return 10
        from cupyx import cusparse
        if cusparse.check_availability('denseToSparse'):
            m = cusparse.denseToSparse(x, format='csc')
        else:
            m = cusparse.dense2csc(x)
        return (m.data, m.indices, m.indptr)

    def _swap(self, x, y):
        if False:
            print('Hello World!')
        return (y, x)

    def __mul__(self, other):
        if False:
            return 10
        from cupyx import cusparse
        if cupy.isscalar(other):
            self.sum_duplicates()
            return self._with_data(self.data * other)
        elif cupyx.scipy.sparse.isspmatrix_csr(other):
            self.sum_duplicates()
            other.sum_duplicates()
            if cusparse.check_availability('spgemm'):
                a = self.tocsr()
                a.sum_duplicates()
                return cusparse.spgemm(a, other)
            elif cusparse.check_availability('csrgemm') and (not runtime.is_hip):
                a = self.T
                return cusparse.csrgemm(a, other, transa=True)
            elif cusparse.check_availability('csrgemm2'):
                a = self.tocsr()
                a.sum_duplicates()
                return cusparse.csrgemm2(a, other)
            else:
                raise AssertionError
        elif isspmatrix_csc(other):
            self.sum_duplicates()
            other.sum_duplicates()
            if cusparse.check_availability('csrgemm') and (not runtime.is_hip):
                a = self.T
                b = other.T
                return cusparse.csrgemm(a, b, transa=True, transb=True)
            elif cusparse.check_availability('csrgemm2'):
                a = self.tocsr()
                b = other.tocsr()
                a.sum_duplicates()
                b.sum_duplicates()
                return cusparse.csrgemm2(a, b)
            elif cusparse.check_availability('spgemm'):
                a = self.tocsr()
                b = other.tocsr()
                a.sum_duplicates()
                b.sum_duplicates()
                return cusparse.spgemm(a, b)
            else:
                raise AssertionError
        elif cupyx.scipy.sparse.isspmatrix(other):
            return self * other.tocsr()
        elif _base.isdense(other):
            if other.ndim == 0:
                self.sum_duplicates()
                return self._with_data(self.data * other)
            elif other.ndim == 1:
                self.sum_duplicates()
                if cusparse.check_availability('csrmv') and (not runtime.is_hip or driver.get_build_version() >= 50000000):
                    csrmv = cusparse.csrmv
                elif cusparse.check_availability('spmv') and (not runtime.is_hip):
                    csrmv = cusparse.spmv
                else:
                    raise AssertionError
                return csrmv(self.T, cupy.asfortranarray(other), transa=True)
            elif other.ndim == 2:
                self.sum_duplicates()
                if cusparse.check_availability('csrmm2') and (not runtime.is_hip or driver.get_build_version() >= 50000000):
                    csrmm = cusparse.csrmm2
                elif cusparse.check_availability('spmm'):
                    csrmm = cusparse.spmm
                else:
                    raise AssertionError
                return csrmm(self.T, cupy.asfortranarray(other), transa=True)
            else:
                raise ValueError('could not interpret dimensions')
        else:
            return NotImplemented

    def eliminate_zeros(self):
        if False:
            i = 10
            return i + 15
        'Removes zero entories in place.'
        t = self.T
        t.eliminate_zeros()
        compress = t.T
        self.data = compress.data
        self.indices = compress.indices
        self.indptr = compress.indptr

    def sort_indices(self):
        if False:
            i = 10
            return i + 15
        'Sorts the indices of this matrix *in place*.\n\n        .. warning::\n            Calling this function might synchronize the device.\n\n        '
        from cupyx import cusparse
        if not self.has_sorted_indices:
            cusparse.cscsort(self)
            self.has_sorted_indices = True

    def toarray(self, order=None, out=None):
        if False:
            for i in range(10):
                print('nop')
        "Returns a dense matrix representing the same value.\n\n        Args:\n            order ({'C', 'F', None}): Whether to store data in C (row-major)\n                order or F (column-major) order. Default is C-order.\n            out: Not supported.\n\n        Returns:\n            cupy.ndarray: Dense array representing the same matrix.\n\n        .. seealso:: :meth:`scipy.sparse.csc_matrix.toarray`\n\n        "
        from cupyx import cusparse
        if order is None:
            order = 'C'
        order = order.upper()
        if self.nnz == 0:
            return cupy.zeros(shape=self.shape, dtype=self.dtype, order=order)
        x = self.copy()
        x.has_canonical_format = False
        x.sum_duplicates()
        if cusparse.check_availability('sparseToDense') and (not runtime.is_hip or x.nnz > 0):
            y = cusparse.sparseToDense(x)
            if order == 'F':
                return y
            elif order == 'C':
                return cupy.ascontiguousarray(y)
            else:
                raise ValueError('order not understood')
        elif order == 'C':
            return cusparse.csr2dense(x.T).T
        elif order == 'F':
            return cusparse.csc2dense(x)
        else:
            raise ValueError('order not understood')

    def _add_sparse(self, other, alpha, beta):
        if False:
            for i in range(10):
                print('nop')
        from cupyx import cusparse
        self.sum_duplicates()
        other = other.tocsc().T
        other.sum_duplicates()
        if cusparse.check_availability('csrgeam2'):
            csrgeam = cusparse.csrgeam2
        elif cusparse.check_availability('csrgeam'):
            csrgeam = cusparse.csrgeam
        else:
            raise NotImplementedError
        return csrgeam(self.T, other, alpha, beta).T

    def tocoo(self, copy=False):
        if False:
            return 10
        'Converts the matrix to COOdinate format.\n\n        Args:\n            copy (bool): If ``False``, it shares data arrays as much as\n                possible.\n\n        Returns:\n            cupyx.scipy.sparse.coo_matrix: Converted matrix.\n\n        '
        from cupyx import cusparse
        if copy:
            data = self.data.copy()
            indices = self.indices.copy()
        else:
            data = self.data
            indices = self.indices
        return cusparse.csc2coo(self, data, indices)

    def tocsc(self, copy=None):
        if False:
            for i in range(10):
                print('nop')
        'Converts the matrix to Compressed Sparse Column format.\n\n        Args:\n            copy (bool): If ``False``, the method returns itself.\n                Otherwise it makes a copy of the matrix.\n\n        Returns:\n            cupyx.scipy.sparse.csc_matrix: Converted matrix.\n\n        '
        if copy:
            return self.copy()
        else:
            return self

    def tocsr(self, copy=False):
        if False:
            return 10
        'Converts the matrix to Compressed Sparse Row format.\n\n        Args:\n            copy (bool): If ``False``, it shares data arrays as much as\n                possible. Actually this option is ignored because all\n                arrays in a matrix cannot be shared in csr to csc conversion.\n\n        Returns:\n            cupyx.scipy.sparse.csr_matrix: Converted matrix.\n\n        '
        from cupyx import cusparse
        if cusparse.check_availability('csc2csr'):
            csc2csr = cusparse.csc2csr
        elif cusparse.check_availability('csc2csrEx2'):
            csc2csr = cusparse.csc2csrEx2
        else:
            raise NotImplementedError
        return csc2csr(self)

    def _tocsx(self):
        if False:
            i = 10
            return i + 15
        'Inverts the format.\n        '
        return self.tocsr()

    def transpose(self, axes=None, copy=False):
        if False:
            while True:
                i = 10
        'Returns a transpose matrix.\n\n        Args:\n            axes: This option is not supported.\n            copy (bool): If ``True``, a returned matrix shares no data.\n                Otherwise, it shared data arrays as much as possible.\n\n        Returns:\n            cupyx.scipy.sparse.csr_matrix: `self` with the dimensions reversed.\n\n        '
        if axes is not None:
            raise ValueError("Sparse matrices do not support an 'axes' parameter because swapping dimensions is the only logical permutation.")
        shape = (self.shape[1], self.shape[0])
        trans = cupyx.scipy.sparse.csr_matrix((self.data, self.indices, self.indptr), shape=shape, copy=copy)
        trans.has_canonical_format = self.has_canonical_format
        return trans

    def getrow(self, i):
        if False:
            return 10
        'Returns a copy of row i of the matrix, as a (1 x n)\n        CSR matrix (row vector).\n\n        Args:\n            i (integer): Row\n\n        Returns:\n            cupyx.scipy.sparse.csc_matrix: Sparse matrix with single row\n        '
        return self._minor_slice(slice(i, i + 1), copy=True).tocsr()

    def getcol(self, i):
        if False:
            return 10
        'Returns a copy of column i of the matrix, as a (m x 1)\n        CSC matrix (column vector).\n\n        Args:\n            i (integer): Column\n\n        Returns:\n            cupyx.scipy.sparse.csc_matrix: Sparse matrix with single column\n        '
        return self._major_slice(slice(i, i + 1), copy=True)

    def _get_intXarray(self, row, col):
        if False:
            while True:
                i = 10
        row = slice(row, row + 1)
        return self._major_index_fancy(col)._minor_slice(row)

    def _get_intXslice(self, row, col):
        if False:
            for i in range(10):
                print('nop')
        row = slice(row, row + 1)
        copy = col.step in (1, None)
        return self._major_slice(col)._minor_slice(row, copy=copy)

    def _get_sliceXint(self, row, col):
        if False:
            print('Hello World!')
        col = slice(col, col + 1)
        return self._major_slice(col)._minor_slice(row, copy=True)

    def _get_sliceXarray(self, row, col):
        if False:
            print('Hello World!')
        return self._major_index_fancy(col)._minor_slice(row)

    def _get_arrayXint(self, row, col):
        if False:
            return 10
        col = slice(col, col + 1)
        return self._major_slice(col)._minor_index_fancy(row)

    def _get_arrayXslice(self, row, col):
        if False:
            print('Hello World!')
        return self._major_slice(col)._minor_index_fancy(row)

def isspmatrix_csc(x):
    if False:
        i = 10
        return i + 15
    'Checks if a given matrix is of CSC format.\n\n    Returns:\n        bool: Returns if ``x`` is :class:`cupyx.scipy.sparse.csc_matrix`.\n\n    '
    return isinstance(x, csc_matrix)