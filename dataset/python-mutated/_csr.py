import operator
import warnings
import numpy
try:
    import scipy.sparse
    _scipy_available = True
except ImportError:
    _scipy_available = False
import cupy
from cupy._core import _accelerator
from cupy.cuda import cub
from cupy.cuda import runtime
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _compressed
from cupyx.scipy.sparse import _csc
from cupyx.scipy.sparse import SparseEfficiencyWarning
from cupyx.scipy.sparse import _util

class csr_matrix(_compressed._compressed_sparse_matrix):
    """Compressed Sparse Row matrix.

    This can be instantiated in several ways.

    ``csr_matrix(D)``
        ``D`` is a rank-2 :class:`cupy.ndarray`.
    ``csr_matrix(S)``
        ``S`` is another sparse matrix. It is equivalent to ``S.tocsr()``.
    ``csr_matrix((M, N), [dtype])``
        It constructs an empty matrix whose shape is ``(M, N)``. Default dtype
        is float64.
    ``csr_matrix((data, (row, col)))``
        All ``data``, ``row`` and ``col`` are one-dimenaional
        :class:`cupy.ndarray`.
    ``csr_matrix((data, indices, indptr))``
        All ``data``, ``indices`` and ``indptr`` are one-dimenaional
        :class:`cupy.ndarray`.

    Args:
        arg1: Arguments for the initializer.
        shape (tuple): Shape of a matrix. Its length must be two.
        dtype: Data type. It must be an argument of :class:`numpy.dtype`.
        copy (bool): If ``True``, copies of given arrays are always used.

    .. seealso::
        :class:`scipy.sparse.csr_matrix`

    """
    format = 'csr'

    def get(self, stream=None):
        if False:
            i = 10
            return i + 15
        'Returns a copy of the array on host memory.\n\n        Args:\n            stream (cupy.cuda.Stream): CUDA stream object. If it is given, the\n                copy runs asynchronously. Otherwise, the copy is synchronous.\n\n        Returns:\n            scipy.sparse.csr_matrix: Copy of the array on host memory.\n\n        '
        if not _scipy_available:
            raise RuntimeError('scipy is not available')
        data = self.data.get(stream)
        indices = self.indices.get(stream)
        indptr = self.indptr.get(stream)
        return scipy.sparse.csr_matrix((data, indices, indptr), shape=self._shape)

    def _convert_dense(self, x):
        if False:
            return 10
        m = dense2csr(x)
        return (m.data, m.indices, m.indptr)

    def _swap(self, x, y):
        if False:
            i = 10
            return i + 15
        return (x, y)

    def _add_sparse(self, other, alpha, beta):
        if False:
            i = 10
            return i + 15
        from cupyx import cusparse
        self.sum_duplicates()
        other = other.tocsr()
        other.sum_duplicates()
        if cusparse.check_availability('csrgeam2'):
            csrgeam = cusparse.csrgeam2
        elif cusparse.check_availability('csrgeam'):
            csrgeam = cusparse.csrgeam
        else:
            raise NotImplementedError
        return csrgeam(self, other, alpha, beta)

    def _comparison(self, other, op, op_name):
        if False:
            i = 10
            return i + 15
        if _util.isscalarlike(other):
            data = cupy.asarray(other, dtype=self.dtype).reshape(1)
            if numpy.isnan(data[0]):
                if op_name == '_ne_':
                    return csr_matrix(cupy.ones(self.shape, dtype=numpy.bool_))
                else:
                    return csr_matrix(self.shape, dtype=numpy.bool_)
            indices = cupy.zeros((1,), dtype=numpy.int32)
            indptr = cupy.arange(2, dtype=numpy.int32)
            other = csr_matrix((data, indices, indptr), shape=(1, 1))
            return binopt_csr(self, other, op_name)
        elif _util.isdense(other):
            return op(self.todense(), other)
        elif isspmatrix_csr(other):
            self.sum_duplicates()
            other.sum_duplicates()
            if op_name in ('_ne_', '_lt_', '_gt_'):
                return binopt_csr(self, other, op_name)
            warnings.warn('Comparing sparse matrices using ==, <=, and >= is inefficient, try using !=, <, or > instead.', SparseEfficiencyWarning)
            if op_name == '_eq_':
                opposite_op_name = '_ne_'
            elif op_name == '_le_':
                opposite_op_name = '_gt_'
            elif op_name == '_ge_':
                opposite_op_name = '_lt_'
            res = binopt_csr(self, other, opposite_op_name)
            out = cupy.logical_not(res.toarray())
            return csr_matrix(out)
        raise NotImplementedError

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._comparison(other, operator.eq, '_eq_')

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        return self._comparison(other, operator.ne, '_ne_')

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        return self._comparison(other, operator.lt, '_lt_')

    def __gt__(self, other):
        if False:
            return 10
        return self._comparison(other, operator.gt, '_gt_')

    def __le__(self, other):
        if False:
            print('Hello World!')
        return self._comparison(other, operator.le, '_le_')

    def __ge__(self, other):
        if False:
            i = 10
            return i + 15
        return self._comparison(other, operator.ge, '_ge_')

    def __mul__(self, other):
        if False:
            return 10
        from cupyx import cusparse
        if cupy.isscalar(other):
            self.sum_duplicates()
            return self._with_data(self.data * other)
        elif isspmatrix_csr(other):
            self.sum_duplicates()
            other.sum_duplicates()
            if cusparse.check_availability('spgemm'):
                return cusparse.spgemm(self, other)
            elif cusparse.check_availability('csrgemm2'):
                return cusparse.csrgemm2(self, other)
            elif cusparse.check_availability('csrgemm'):
                return cusparse.csrgemm(self, other)
            else:
                raise AssertionError
        elif _csc.isspmatrix_csc(other):
            self.sum_duplicates()
            other.sum_duplicates()
            if cusparse.check_availability('csrgemm') and (not runtime.is_hip):
                return cusparse.csrgemm(self, other.T, transb=True)
            elif cusparse.check_availability('spgemm'):
                b = other.tocsr()
                b.sum_duplicates()
                return cusparse.spgemm(self, b)
            elif cusparse.check_availability('csrgemm2'):
                b = other.tocsr()
                b.sum_duplicates()
                return cusparse.csrgemm2(self, b)
            else:
                raise AssertionError
        elif _base.isspmatrix(other):
            return self * other.tocsr()
        elif _base.isdense(other):
            if other.ndim == 0:
                self.sum_duplicates()
                return self._with_data(self.data * other)
            elif other.ndim == 1:
                self.sum_duplicates()
                other = cupy.asfortranarray(other)
                is_cub_safe = self.indptr.data.mem.size > self.indptr.size * self.indptr.dtype.itemsize
                is_cub_safe &= cub._get_cuda_build_version() < 11000
                for accelerator in _accelerator.get_routine_accelerators():
                    if accelerator == _accelerator.ACCELERATOR_CUB and (not runtime.is_hip) and is_cub_safe and other.flags.c_contiguous:
                        return cub.device_csrmv(self.shape[0], self.shape[1], self.nnz, self.data, self.indptr, self.indices, other)
                if cusparse.check_availability('csrmvEx') and self.nnz > 0 and cusparse.csrmvExIsAligned(self, other):
                    csrmv = cusparse.csrmvEx
                elif cusparse.check_availability('csrmv'):
                    csrmv = cusparse.csrmv
                elif cusparse.check_availability('spmv'):
                    csrmv = cusparse.spmv
                else:
                    raise AssertionError
                return csrmv(self, other)
            elif other.ndim == 2:
                self.sum_duplicates()
                if cusparse.check_availability('csrmm2'):
                    csrmm = cusparse.csrmm2
                elif cusparse.check_availability('spmm'):
                    csrmm = cusparse.spmm
                else:
                    raise AssertionError
                return csrmm(self, cupy.asfortranarray(other))
            else:
                raise ValueError('could not interpret dimensions')
        else:
            return NotImplemented

    def __div__(self, other):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def __rdiv__(self, other):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def __truediv__(self, other):
        if False:
            return 10
        'Point-wise division by another matrix, vector or scalar'
        if _util.isscalarlike(other):
            dtype = self.dtype
            if dtype == numpy.float32:
                dtype = numpy.float64
            dtype = cupy.result_type(dtype, other)
            d = cupy.reciprocal(other, dtype=dtype)
            return multiply_by_scalar(self, d)
        elif _util.isdense(other):
            other = cupy.atleast_2d(other)
            other = cupy.broadcast_to(other, self.shape)
            check_shape_for_pointwise_op(self.shape, other.shape)
            ret = self.tocoo()
            ret.data = _cupy_divide_by_dense()(ret.data, ret.row, ret.col, ret.shape[1], other)
            return ret
        elif _base.isspmatrix(other):
            check_shape_for_pointwise_op(self.shape, other.shape, allow_broadcasting=False)
            dtype = numpy.promote_types(self.dtype, other.dtype)
            if dtype.char not in 'FD':
                dtype = numpy.promote_types(numpy.float64, dtype)
            self_dense = self.todense().astype(dtype, copy=False)
            return self_dense / other.todense()
        return NotImplemented

    def __rtruediv__(self, other):
        if False:
            i = 10
            return i + 15
        return NotImplemented

    def diagonal(self, k=0):
        if False:
            while True:
                i = 10
        (rows, cols) = self.shape
        ylen = min(rows + min(k, 0), cols - max(k, 0))
        if ylen <= 0:
            return cupy.empty(0, dtype=self.dtype)
        self.sum_duplicates()
        y = cupy.empty(ylen, dtype=self.dtype)
        _cupy_csr_diagonal()(k, rows, cols, self.data, self.indptr, self.indices, y)
        return y

    def eliminate_zeros(self):
        if False:
            print('Hello World!')
        'Removes zero entories in place.'
        from cupyx import cusparse
        compress = cusparse.csr2csr_compress(self, 0)
        self.data = compress.data
        self.indices = compress.indices
        self.indptr = compress.indptr

    def _maximum_minimum(self, other, cupy_op, op_name, dense_check):
        if False:
            return 10
        if _util.isscalarlike(other):
            other = cupy.asarray(other, dtype=self.dtype)
            if dense_check(other):
                dtype = self.dtype
                if dtype == numpy.float32:
                    dtype = numpy.float64
                elif dtype == numpy.complex64:
                    dtype = numpy.complex128
                dtype = cupy.result_type(dtype, other)
                other = other.astype(dtype, copy=False)
                new_array = cupy_op(self.todense(), other)
                return csr_matrix(new_array)
            else:
                self.sum_duplicates()
                new_data = cupy_op(self.data, other)
                return csr_matrix((new_data, self.indices, self.indptr), shape=self.shape, dtype=self.dtype)
        elif _util.isdense(other):
            self.sum_duplicates()
            other = cupy.atleast_2d(other)
            return cupy_op(self.todense(), other)
        elif isspmatrix_csr(other):
            self.sum_duplicates()
            other.sum_duplicates()
            return binopt_csr(self, other, op_name)
        raise NotImplementedError

    def maximum(self, other):
        if False:
            i = 10
            return i + 15
        return self._maximum_minimum(other, cupy.maximum, '_maximum_', lambda x: x > 0)

    def minimum(self, other):
        if False:
            i = 10
            return i + 15
        return self._maximum_minimum(other, cupy.minimum, '_minimum_', lambda x: x < 0)

    def multiply(self, other):
        if False:
            while True:
                i = 10
        'Point-wise multiplication by another matrix, vector or scalar'
        if cupy.isscalar(other):
            return multiply_by_scalar(self, other)
        elif _util.isdense(other):
            self.sum_duplicates()
            other = cupy.atleast_2d(other)
            return multiply_by_dense(self, other)
        elif isspmatrix_csr(other):
            self.sum_duplicates()
            other.sum_duplicates()
            return multiply_by_csr(self, other)
        else:
            msg = 'expected scalar, dense matrix/vector or csr matrix'
            raise TypeError(msg)

    def setdiag(self, values, k=0):
        if False:
            return 10
        'Set diagonal or off-diagonal elements of the array.'
        (rows, cols) = self.shape
        (row_st, col_st) = (max(0, -k), max(0, k))
        x_len = min(rows - row_st, cols - col_st)
        if x_len <= 0:
            raise ValueError('k exceeds matrix dimensions')
        values = values.astype(self.dtype)
        if values.ndim == 0:
            x_data = cupy.full((x_len,), values, dtype=self.dtype)
        else:
            x_len = min(x_len, values.size)
            x_data = values[:x_len]
        x_indices = cupy.arange(col_st, col_st + x_len, dtype='i')
        x_indptr = cupy.zeros((rows + 1,), dtype='i')
        x_indptr[row_st:row_st + x_len + 1] = cupy.arange(x_len + 1, dtype='i')
        x_indptr[row_st + x_len + 1:] = x_len
        x_data -= self.diagonal(k=k)[:x_len]
        y = self + csr_matrix((x_data, x_indices, x_indptr), shape=self.shape)
        self.data = y.data
        self.indices = y.indices
        self.indptr = y.indptr

    def sort_indices(self):
        if False:
            return 10
        'Sorts the indices of this matrix *in place*.\n\n        .. warning::\n            Calling this function might synchronize the device.\n\n        '
        from cupyx import cusparse
        if not self.has_sorted_indices:
            cusparse.csrsort(self)
            self.has_sorted_indices = True

    def toarray(self, order=None, out=None):
        if False:
            print('Hello World!')
        "Returns a dense matrix representing the same value.\n\n        Args:\n            order ({'C', 'F', None}): Whether to store data in C (row-major)\n                order or F (column-major) order. Default is C-order.\n            out: Not supported.\n\n        Returns:\n            cupy.ndarray: Dense array representing the same matrix.\n\n        .. seealso:: :meth:`scipy.sparse.csr_matrix.toarray`\n\n        "
        from cupyx import cusparse
        order = 'C' if order is None else order.upper()
        if self.nnz == 0:
            return cupy.zeros(shape=self.shape, dtype=self.dtype, order=order)
        if self.dtype.char not in 'fdFD':
            return csr2dense(self, order)
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
            return cusparse.csc2dense(x.T).T
        elif order == 'F':
            return cusparse.csr2dense(x)
        else:
            raise ValueError('order not understood')

    def tobsr(self, blocksize=None, copy=False):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def tocoo(self, copy=False):
        if False:
            i = 10
            return i + 15
        'Converts the matrix to COOdinate format.\n\n        Args:\n            copy (bool): If ``False``, it shares data arrays as much as\n                possible.\n\n        Returns:\n            cupyx.scipy.sparse.coo_matrix: Converted matrix.\n\n        '
        from cupyx import cusparse
        if copy:
            data = self.data.copy()
            indices = self.indices.copy()
        else:
            data = self.data
            indices = self.indices
        return cusparse.csr2coo(self, data, indices)

    def tocsc(self, copy=False):
        if False:
            i = 10
            return i + 15
        'Converts the matrix to Compressed Sparse Column format.\n\n        Args:\n            copy (bool): If ``False``, it shares data arrays as much as\n                possible. Actually this option is ignored because all\n                arrays in a matrix cannot be shared in csr to csc conversion.\n\n        Returns:\n            cupyx.scipy.sparse.csc_matrix: Converted matrix.\n\n        '
        from cupyx import cusparse
        if cusparse.check_availability('csr2csc'):
            csr2csc = cusparse.csr2csc
        elif cusparse.check_availability('csr2cscEx2'):
            csr2csc = cusparse.csr2cscEx2
        else:
            raise NotImplementedError
        return csr2csc(self)

    def tocsr(self, copy=False):
        if False:
            for i in range(10):
                print('nop')
        'Converts the matrix to Compressed Sparse Row format.\n\n        Args:\n            copy (bool): If ``False``, the method returns itself.\n                Otherwise it makes a copy of the matrix.\n\n        Returns:\n            cupyx.scipy.sparse.csr_matrix: Converted matrix.\n\n        '
        if copy:
            return self.copy()
        else:
            return self

    def _tocsx(self):
        if False:
            print('Hello World!')
        'Inverts the format.\n        '
        return self.tocsc()

    def todia(self, copy=False):
        if False:
            return 10
        raise NotImplementedError

    def todok(self, copy=False):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def tolil(self, copy=False):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def transpose(self, axes=None, copy=False):
        if False:
            i = 10
            return i + 15
        'Returns a transpose matrix.\n\n        Args:\n            axes: This option is not supported.\n            copy (bool): If ``True``, a returned matrix shares no data.\n                Otherwise, it shared data arrays as much as possible.\n\n        Returns:\n            cupyx.scipy.sparse.csc_matrix: `self` with the dimensions reversed.\n\n        '
        if axes is not None:
            raise ValueError("Sparse matrices do not support an 'axes' parameter because swapping dimensions is the only logical permutation.")
        shape = (self.shape[1], self.shape[0])
        trans = _csc.csc_matrix((self.data, self.indices, self.indptr), shape=shape, copy=copy)
        trans.has_canonical_format = self.has_canonical_format
        return trans

    def getrow(self, i):
        if False:
            print('Hello World!')
        'Returns a copy of row i of the matrix, as a (1 x n)\n        CSR matrix (row vector).\n\n        Args:\n            i (integer): Row\n\n        Returns:\n            cupyx.scipy.sparse.csr_matrix: Sparse matrix with single row\n        '
        return self._major_slice(slice(i, i + 1), copy=True)

    def getcol(self, i):
        if False:
            while True:
                i = 10
        'Returns a copy of column i of the matrix, as a (m x 1)\n        CSR matrix (column vector).\n\n        Args:\n            i (integer): Column\n\n        Returns:\n            cupyx.scipy.sparse.csr_matrix: Sparse matrix with single column\n        '
        return self._minor_slice(slice(i, i + 1), copy=True)

    def _get_intXarray(self, row, col):
        if False:
            i = 10
            return i + 15
        row = slice(row, row + 1)
        return self._major_slice(row)._minor_index_fancy(col)

    def _get_intXslice(self, row, col):
        if False:
            for i in range(10):
                print('nop')
        row = slice(row, row + 1)
        return self._major_slice(row)._minor_slice(col, copy=True)

    def _get_sliceXint(self, row, col):
        if False:
            for i in range(10):
                print('nop')
        col = slice(col, col + 1)
        copy = row.step in (1, None)
        return self._major_slice(row)._minor_slice(col, copy=copy)

    def _get_sliceXarray(self, row, col):
        if False:
            while True:
                i = 10
        return self._major_slice(row)._minor_index_fancy(col)

    def _get_arrayXint(self, row, col):
        if False:
            i = 10
            return i + 15
        col = slice(col, col + 1)
        return self._major_index_fancy(row)._minor_slice(col)

    def _get_arrayXslice(self, row, col):
        if False:
            print('Hello World!')
        if col.step not in (1, None):
            (start, stop, step) = col.indices(self.shape[1])
            cols = cupy.arange(start, stop, step, self.indices.dtype)
            return self._get_arrayXarray(row, cols)
        return self._major_index_fancy(row)._minor_slice(col)

def isspmatrix_csr(x):
    if False:
        print('Hello World!')
    'Checks if a given matrix is of CSR format.\n\n    Returns:\n        bool: Returns if ``x`` is :class:`cupyx.scipy.sparse.csr_matrix`.\n\n    '
    return isinstance(x, csr_matrix)

def check_shape_for_pointwise_op(a_shape, b_shape, allow_broadcasting=True):
    if False:
        while True:
            i = 10
    if allow_broadcasting:
        (a_m, a_n) = a_shape
        (b_m, b_n) = b_shape
        if not (a_m == b_m or a_m == 1 or b_m == 1):
            raise ValueError('inconsistent shape')
        if not (a_n == b_n or a_n == 1 or b_n == 1):
            raise ValueError('inconsistent shape')
    elif a_shape != b_shape:
        raise ValueError('inconsistent shape')

def multiply_by_scalar(sp, a):
    if False:
        i = 10
        return i + 15
    data = sp.data * a
    indices = sp.indices.copy()
    indptr = sp.indptr.copy()
    return csr_matrix((data, indices, indptr), shape=sp.shape)

def multiply_by_dense(sp, dn):
    if False:
        return 10
    check_shape_for_pointwise_op(sp.shape, dn.shape)
    (sp_m, sp_n) = sp.shape
    (dn_m, dn_n) = dn.shape
    (m, n) = (max(sp_m, dn_m), max(sp_n, dn_n))
    nnz = sp.nnz * (m // sp_m) * (n // sp_n)
    dtype = numpy.promote_types(sp.dtype, dn.dtype)
    data = cupy.empty(nnz, dtype=dtype)
    indices = cupy.empty(nnz, dtype=sp.indices.dtype)
    if m > sp_m:
        if n > sp_n:
            indptr = cupy.arange(0, nnz + 1, n, dtype=sp.indptr.dtype)
        else:
            indptr = cupy.arange(0, nnz + 1, sp.nnz, dtype=sp.indptr.dtype)
    else:
        indptr = sp.indptr.copy()
        if n > sp_n:
            indptr *= n
    cupy_multiply_by_dense()(sp.data, sp.indptr, sp.indices, sp_m, sp_n, dn, dn_m, dn_n, indptr, m, n, data, indices)
    return csr_matrix((data, indices, indptr), shape=(m, n))
_GET_ROW_ID_ = '\n__device__ inline int get_row_id(int i, int min, int max, const int *indptr) {\n    int row = (min + max) / 2;\n    while (min < max) {\n        if (i < indptr[row]) {\n            max = row - 1;\n        } else if (i >= indptr[row + 1]) {\n            min = row + 1;\n        } else {\n            break;\n        }\n        row = (min + max) / 2;\n    }\n    return row;\n}\n'
_FIND_INDEX_HOLDING_COL_IN_ROW_ = '\n__device__ inline int find_index_holding_col_in_row(\n        int row, int col, const int *indptr, const int *indices) {\n    int j_min = indptr[row];\n    int j_max = indptr[row+1] - 1;\n    while (j_min <= j_max) {\n        int j = (j_min + j_max) / 2;\n        int j_col = indices[j];\n        if (j_col == col) {\n            return j;\n        } else if (j_col < col) {\n            j_min = j + 1;\n        } else {\n            j_max = j - 1;\n        }\n    }\n    return -1;\n}\n'

@cupy._util.memoize(for_each_device=True)
def cupy_multiply_by_dense():
    if False:
        print('Hello World!')
    return cupy.ElementwiseKernel('\n        raw S SP_DATA, raw I SP_INDPTR, raw I SP_INDICES,\n        int32 SP_M, int32 SP_N,\n        raw D DN_DATA, int32 DN_M, int32 DN_N,\n        raw I OUT_INDPTR, int32 OUT_M, int32 OUT_N\n        ', 'O OUT_DATA, I OUT_INDICES', '\n        int i_out = i;\n        int m_out = get_row_id(i_out, 0, OUT_M - 1, &(OUT_INDPTR[0]));\n        int i_sp = i_out;\n        if (OUT_M > SP_M && SP_M == 1) {\n            i_sp -= OUT_INDPTR[m_out];\n        }\n        if (OUT_N > SP_N && SP_N == 1) {\n            i_sp /= OUT_N;\n        }\n        int n_out = SP_INDICES[i_sp];\n        if (OUT_N > SP_N && SP_N == 1) {\n            n_out = i_out - OUT_INDPTR[m_out];\n        }\n        int m_dn = m_out;\n        if (OUT_M > DN_M && DN_M == 1) {\n            m_dn = 0;\n        }\n        int n_dn = n_out;\n        if (OUT_N > DN_N && DN_N == 1) {\n            n_dn = 0;\n        }\n        OUT_DATA = (O)(SP_DATA[i_sp] * DN_DATA[n_dn + (DN_N * m_dn)]);\n        OUT_INDICES = n_out;\n        ', 'cupyx_scipy_sparse_csr_multiply_by_dense', preamble=_GET_ROW_ID_)

@cupy._util.memoize(for_each_device=True)
def _cupy_divide_by_dense():
    if False:
        while True:
            i = 10
    return cupy.ElementwiseKernel('T data, I row, I col, I width, raw T other', 'T res', '\n        res = data / other[row * width + col]\n        ', 'cupyx_scipy_sparse_coo_divide_dense')

def multiply_by_csr(a, b):
    if False:
        print('Hello World!')
    check_shape_for_pointwise_op(a.shape, b.shape)
    (a_m, a_n) = a.shape
    (b_m, b_n) = b.shape
    (m, n) = (max(a_m, b_m), max(a_n, b_n))
    a_nnz = a.nnz * (m // a_m) * (n // a_n)
    b_nnz = b.nnz * (m // b_m) * (n // b_n)
    if a_nnz > b_nnz:
        return multiply_by_csr(b, a)
    c_nnz = a_nnz
    dtype = numpy.promote_types(a.dtype, b.dtype)
    c_data = cupy.empty(c_nnz, dtype=dtype)
    c_indices = cupy.empty(c_nnz, dtype=a.indices.dtype)
    if m > a_m:
        if n > a_n:
            c_indptr = cupy.arange(0, c_nnz + 1, n, dtype=a.indptr.dtype)
        else:
            c_indptr = cupy.arange(0, c_nnz + 1, a.nnz, dtype=a.indptr.dtype)
    else:
        c_indptr = a.indptr.copy()
        if n > a_n:
            c_indptr *= n
    flags = cupy.zeros(c_nnz + 1, dtype=a.indices.dtype)
    nnz_each_row = cupy.zeros(m + 1, dtype=a.indptr.dtype)
    cupy_multiply_by_csr_step1()(a.data, a.indptr, a.indices, a_m, a_n, b.data, b.indptr, b.indices, b_m, b_n, c_indptr, m, n, c_data, c_indices, flags, nnz_each_row)
    flags = cupy.cumsum(flags, dtype=a.indptr.dtype)
    d_indptr = cupy.cumsum(nnz_each_row, dtype=a.indptr.dtype)
    d_nnz = int(d_indptr[-1])
    d_data = cupy.empty(d_nnz, dtype=dtype)
    d_indices = cupy.empty(d_nnz, dtype=a.indices.dtype)
    cupy_multiply_by_csr_step2()(c_data, c_indices, flags, d_data, d_indices)
    return csr_matrix((d_data, d_indices, d_indptr), shape=(m, n))

@cupy._util.memoize(for_each_device=True)
def cupy_multiply_by_csr_step1():
    if False:
        return 10
    return cupy.ElementwiseKernel('\n        raw A A_DATA, raw I A_INDPTR, raw I A_INDICES, int32 A_M, int32 A_N,\n        raw B B_DATA, raw I B_INDPTR, raw I B_INDICES, int32 B_M, int32 B_N,\n        raw I C_INDPTR, int32 C_M, int32 C_N\n        ', 'C C_DATA, I C_INDICES, raw I FLAGS, raw I NNZ_EACH_ROW', '\n        int i_c = i;\n        int m_c = get_row_id(i_c, 0, C_M - 1, &(C_INDPTR[0]));\n\n        int i_a = i;\n        if (C_M > A_M && A_M == 1) {\n            i_a -= C_INDPTR[m_c];\n        }\n        if (C_N > A_N && A_N == 1) {\n            i_a /= C_N;\n        }\n        int n_c = A_INDICES[i_a];\n        if (C_N > A_N && A_N == 1) {\n            n_c = i % C_N;\n        }\n        int m_b = m_c;\n        if (C_M > B_M && B_M == 1) {\n            m_b = 0;\n        }\n        int n_b = n_c;\n        if (C_N > B_N && B_N == 1) {\n            n_b = 0;\n        }\n        int i_b = find_index_holding_col_in_row(m_b, n_b,\n            &(B_INDPTR[0]), &(B_INDICES[0]));\n        if (i_b >= 0) {\n            atomicAdd(&(NNZ_EACH_ROW[m_c+1]), 1);\n            FLAGS[i+1] = 1;\n            C_DATA = (C)(A_DATA[i_a] * B_DATA[i_b]);\n            C_INDICES = n_c;\n        }\n        ', 'cupyx_scipy_sparse_csr_multiply_by_csr_step1', preamble=_GET_ROW_ID_ + _FIND_INDEX_HOLDING_COL_IN_ROW_)

@cupy._util.memoize(for_each_device=True)
def cupy_multiply_by_csr_step2():
    if False:
        return 10
    return cupy.ElementwiseKernel('T C_DATA, I C_INDICES, raw I FLAGS', 'raw D D_DATA, raw I D_INDICES', '\n        int j = FLAGS[i];\n        if (j < FLAGS[i+1]) {\n            D_DATA[j] = (D)(C_DATA);\n            D_INDICES[j] = C_INDICES;\n        }\n        ', 'cupyx_scipy_sparse_csr_multiply_by_csr_step2')
_BINOPT_MAX_ = '\n__device__ inline O binopt(T in1, T in2) {\n    return max(in1, in2);\n}\n'
_BINOPT_MIN_ = '\n__device__ inline O binopt(T in1, T in2) {\n    return min(in1, in2);\n}\n'
_BINOPT_EQ_ = '\n__device__ inline O binopt(T in1, T in2) {\n    return (in1 == in2);\n}\n'
_BINOPT_NE_ = '\n__device__ inline O binopt(T in1, T in2) {\n    return (in1 != in2);\n}\n'
_BINOPT_LT_ = '\n__device__ inline O binopt(T in1, T in2) {\n    return (in1 < in2);\n}\n'
_BINOPT_GT_ = '\n__device__ inline O binopt(T in1, T in2) {\n    return (in1 > in2);\n}\n'
_BINOPT_LE_ = '\n__device__ inline O binopt(T in1, T in2) {\n    return (in1 <= in2);\n}\n'
_BINOPT_GE_ = '\n__device__ inline O binopt(T in1, T in2) {\n    return (in1 >= in2);\n}\n'

def binopt_csr(a, b, op_name):
    if False:
        i = 10
        return i + 15
    check_shape_for_pointwise_op(a.shape, b.shape)
    (a_m, a_n) = a.shape
    (b_m, b_n) = b.shape
    (m, n) = (max(a_m, b_m), max(a_n, b_n))
    a_nnz = a.nnz * (m // a_m) * (n // a_n)
    b_nnz = b.nnz * (m // b_m) * (n // b_n)
    a_info = cupy.zeros(a_nnz + 1, dtype=a.indices.dtype)
    b_info = cupy.zeros(b_nnz + 1, dtype=b.indices.dtype)
    a_valid = cupy.zeros(a_nnz, dtype=numpy.int8)
    b_valid = cupy.zeros(b_nnz, dtype=numpy.int8)
    c_indptr = cupy.zeros(m + 1, dtype=a.indptr.dtype)
    in_dtype = numpy.promote_types(a.dtype, b.dtype)
    a_data = a.data.astype(in_dtype, copy=False)
    b_data = b.data.astype(in_dtype, copy=False)
    funcs = _GET_ROW_ID_
    if op_name == '_maximum_':
        funcs += _BINOPT_MAX_
        out_dtype = in_dtype
    elif op_name == '_minimum_':
        funcs += _BINOPT_MIN_
        out_dtype = in_dtype
    elif op_name == '_eq_':
        funcs += _BINOPT_EQ_
        out_dtype = numpy.bool_
    elif op_name == '_ne_':
        funcs += _BINOPT_NE_
        out_dtype = numpy.bool_
    elif op_name == '_lt_':
        funcs += _BINOPT_LT_
        out_dtype = numpy.bool_
    elif op_name == '_gt_':
        funcs += _BINOPT_GT_
        out_dtype = numpy.bool_
    elif op_name == '_le_':
        funcs += _BINOPT_LE_
        out_dtype = numpy.bool_
    elif op_name == '_ge_':
        funcs += _BINOPT_GE_
        out_dtype = numpy.bool_
    else:
        raise ValueError('invalid op_name: {}'.format(op_name))
    a_tmp_data = cupy.empty(a_nnz, dtype=out_dtype)
    b_tmp_data = cupy.empty(b_nnz, dtype=out_dtype)
    a_tmp_indices = cupy.empty(a_nnz, dtype=a.indices.dtype)
    b_tmp_indices = cupy.empty(b_nnz, dtype=b.indices.dtype)
    _size = a_nnz + b_nnz
    cupy_binopt_csr_step1(op_name, preamble=funcs)(m, n, a.indptr, a.indices, a_data, a_m, a_n, a.nnz, a_nnz, b.indptr, b.indices, b_data, b_m, b_n, b.nnz, b_nnz, a_info, a_valid, a_tmp_indices, a_tmp_data, b_info, b_valid, b_tmp_indices, b_tmp_data, c_indptr, size=_size)
    a_info = cupy.cumsum(a_info, dtype=a_info.dtype)
    b_info = cupy.cumsum(b_info, dtype=b_info.dtype)
    c_indptr = cupy.cumsum(c_indptr, dtype=c_indptr.dtype)
    c_nnz = int(c_indptr[-1])
    c_indices = cupy.empty(c_nnz, dtype=a.indices.dtype)
    c_data = cupy.empty(c_nnz, dtype=out_dtype)
    cupy_binopt_csr_step2(op_name)(a_info, a_valid, a_tmp_indices, a_tmp_data, a_nnz, b_info, b_valid, b_tmp_indices, b_tmp_data, b_nnz, c_indices, c_data, size=_size)
    return csr_matrix((c_data, c_indices, c_indptr), shape=(m, n))

@cupy._util.memoize(for_each_device=True)
def cupy_binopt_csr_step1(op_name, preamble=''):
    if False:
        while True:
            i = 10
    name = 'cupyx_scipy_sparse_csr_binopt_' + op_name + 'step1'
    return cupy.ElementwiseKernel('\n        int32 M, int32 N,\n        raw I A_INDPTR, raw I A_INDICES, raw T A_DATA,\n        int32 A_M, int32 A_N, int32 A_NNZ_ACT, int32 A_NNZ,\n        raw I B_INDPTR, raw I B_INDICES, raw T B_DATA,\n        int32 B_M, int32 B_N, int32 B_NNZ_ACT, int32 B_NNZ\n        ', '\n        raw I A_INFO, raw B A_VALID, raw I A_TMP_INDICES, raw O A_TMP_DATA,\n        raw I B_INFO, raw B B_VALID, raw I B_TMP_INDICES, raw O B_TMP_DATA,\n        raw I C_INFO\n        ', '\n        if (i >= A_NNZ + B_NNZ) return;\n\n        const int *MY_INDPTR, *MY_INDICES;  int *MY_INFO;  const T *MY_DATA;\n        const int *OP_INDPTR, *OP_INDICES;  int *OP_INFO;  const T *OP_DATA;\n        int MY_M, MY_N, MY_NNZ_ACT, MY_NNZ;\n        int OP_M, OP_N, OP_NNZ_ACT, OP_NNZ;\n        signed char *MY_VALID;  I *MY_TMP_INDICES;  O *MY_TMP_DATA;\n\n        int my_j;\n        if (i < A_NNZ) {\n            // in charge of one of non-zero element of sparse matrix A\n            my_j = i;\n            MY_INDPTR  = &(A_INDPTR[0]);   OP_INDPTR  = &(B_INDPTR[0]);\n            MY_INDICES = &(A_INDICES[0]);  OP_INDICES = &(B_INDICES[0]);\n            MY_INFO    = &(A_INFO[0]);     OP_INFO    = &(B_INFO[0]);\n            MY_DATA    = &(A_DATA[0]);     OP_DATA    = &(B_DATA[0]);\n            MY_M       = A_M;              OP_M       = B_M;\n            MY_N       = A_N;              OP_N       = B_N;\n            MY_NNZ_ACT = A_NNZ_ACT;        OP_NNZ_ACT = B_NNZ_ACT;\n            MY_NNZ     = A_NNZ;            OP_NNZ     = B_NNZ;\n            MY_VALID   = &(A_VALID[0]);\n            MY_TMP_DATA= &(A_TMP_DATA[0]);\n            MY_TMP_INDICES = &(A_TMP_INDICES[0]);\n        } else {\n            // in charge of one of non-zero element of sparse matrix B\n            my_j = i - A_NNZ;\n            MY_INDPTR  = &(B_INDPTR[0]);   OP_INDPTR  = &(A_INDPTR[0]);\n            MY_INDICES = &(B_INDICES[0]);  OP_INDICES = &(A_INDICES[0]);\n            MY_INFO    = &(B_INFO[0]);     OP_INFO    = &(A_INFO[0]);\n            MY_DATA    = &(B_DATA[0]);     OP_DATA    = &(A_DATA[0]);\n            MY_M       = B_M;              OP_M       = A_M;\n            MY_N       = B_N;              OP_N       = A_N;\n            MY_NNZ_ACT = B_NNZ_ACT;        OP_NNZ_ACT = A_NNZ_ACT;\n            MY_NNZ     = B_NNZ;            OP_NNZ     = A_NNZ;\n            MY_VALID   = &(B_VALID[0]);\n            MY_TMP_DATA= &(B_TMP_DATA[0]);\n            MY_TMP_INDICES = &(B_TMP_INDICES[0]);\n        }\n        int _min, _max, _mid;\n\n        // get column location\n        int my_col;\n        int my_j_act = my_j;\n        if (MY_M == 1 && MY_M < M) {\n            if (MY_N == 1 && MY_N < N) my_j_act = 0;\n            else                       my_j_act = my_j % MY_NNZ_ACT;\n        } else {\n            if (MY_N == 1 && MY_N < N) my_j_act = my_j / N;\n        }\n        my_col = MY_INDICES[my_j_act];\n        if (MY_N == 1 && MY_N < N) {\n            my_col = my_j % N;\n        }\n\n        // get row location\n        int my_row = get_row_id(my_j_act, 0, MY_M - 1, &(MY_INDPTR[0]));\n        if (MY_M == 1 && MY_M < M) {\n            if (MY_N == 1 && MY_N < N) my_row = my_j / N;\n            else                       my_row = my_j / MY_NNZ_ACT;\n        }\n\n        int op_row = my_row;\n        int op_row_act = op_row;\n        if (OP_M == 1 && OP_M < M) {\n            op_row_act = 0;\n        }\n\n        int op_col = 0;\n        _min = OP_INDPTR[op_row_act];\n        _max = OP_INDPTR[op_row_act + 1] - 1;\n        int op_j_act = _min;\n        bool op_nz = false;\n        if (_min <= _max) {\n            if (OP_N == 1 && OP_N < N) {\n                op_col = my_col;\n                op_nz = true;\n            }\n            else {\n                _mid = (_min + _max) / 2;\n                op_col = OP_INDICES[_mid];\n                while (_min < _max) {\n                    if (op_col < my_col) {\n                        _min = _mid + 1;\n                    } else if (op_col > my_col) {\n                        _max = _mid;\n                    } else {\n                        break;\n                    }\n                    _mid = (_min + _max) / 2;\n                    op_col = OP_INDICES[_mid];\n                }\n                op_j_act = _mid;\n                if (op_col == my_col) {\n                    op_nz = true;\n                } else if (op_col < my_col) {\n                    op_col = N;\n                    op_j_act += 1;\n                }\n            }\n        }\n\n        int op_j = op_j_act;\n        if (OP_M == 1 && OP_M < M) {\n            if (OP_N == 1 && OP_N < N) {\n                op_j = (op_col + N * op_row) * OP_NNZ_ACT;\n            } else {\n                op_j = op_j_act + OP_NNZ_ACT * op_row;\n            }\n        } else {\n            if (OP_N == 1 && OP_N < N) {\n                op_j = op_col + N * op_j_act;\n            }\n        }\n\n        if (i < A_NNZ || !op_nz) {\n            T my_data = MY_DATA[my_j_act];\n            T op_data = 0;\n            if (op_nz) op_data = OP_DATA[op_j_act];\n            O out;\n            if (i < A_NNZ) out = binopt(my_data, op_data);\n            else           out = binopt(op_data, my_data);\n            if (out != static_cast<O>(0)) {\n                MY_VALID[my_j] = 1;\n                MY_TMP_DATA[my_j] = out;\n                MY_TMP_INDICES[my_j] = my_col;\n                atomicAdd( &(C_INFO[my_row + 1]), 1 );\n                atomicAdd( &(MY_INFO[my_j + 1]), 1 );\n                atomicAdd( &(OP_INFO[op_j]), 1 );\n            }\n        }\n        ', name, preamble=preamble)

@cupy._util.memoize(for_each_device=True)
def cupy_binopt_csr_step2(op_name):
    if False:
        i = 10
        return i + 15
    name = 'cupyx_scipy_sparse_csr_binopt' + op_name + 'step2'
    return cupy.ElementwiseKernel('\n        raw I A_INFO, raw B A_VALID, raw I A_TMP_INDICES, raw O A_TMP_DATA,\n        int32 A_NNZ,\n        raw I B_INFO, raw B B_VALID, raw I B_TMP_INDICES, raw O B_TMP_DATA,\n        int32 B_NNZ\n        ', 'raw I C_INDICES, raw O C_DATA', '\n        if (i < A_NNZ) {\n            int j = i;\n            if (A_VALID[j]) {\n                C_INDICES[A_INFO[j]] = A_TMP_INDICES[j];\n                C_DATA[A_INFO[j]]    = A_TMP_DATA[j];\n            }\n        } else if (i < A_NNZ + B_NNZ) {\n            int j = i - A_NNZ;\n            if (B_VALID[j]) {\n                C_INDICES[B_INFO[j]] = B_TMP_INDICES[j];\n                C_DATA[B_INFO[j]]    = B_TMP_DATA[j];\n            }\n        }\n        ', name)

def csr2dense(a, order):
    if False:
        return 10
    out = cupy.zeros(a.shape, dtype=a.dtype, order=order)
    (m, n) = a.shape
    kern = _cupy_csr2dense(a.dtype)
    kern(m, n, a.indptr, a.indices, a.data, order == 'C', out)
    return out

@cupy._util.memoize(for_each_device=True)
def _cupy_csr2dense(dtype):
    if False:
        i = 10
        return i + 15
    if dtype == '?':
        op = 'if (DATA) OUT[index] = true;'
    else:
        op = 'atomicAdd(&OUT[index], DATA);'
    return cupy.ElementwiseKernel('int32 M, int32 N, raw I INDPTR, I INDICES, T DATA, bool C_ORDER', 'raw T OUT', '\n        int row = get_row_id(i, 0, M - 1, &(INDPTR[0]));\n        int col = INDICES;\n        int index = C_ORDER ? col + N * row : row + M * col;\n        ' + op, 'cupyx_scipy_sparse_csr2dense', preamble=_GET_ROW_ID_)

def dense2csr(a):
    if False:
        return 10
    from cupyx import cusparse
    if a.dtype.char in 'fdFD':
        if cusparse.check_availability('denseToSparse'):
            return cusparse.denseToSparse(a, format='csr')
        else:
            return cusparse.dense2csr(a)
    (m, n) = a.shape
    a = cupy.ascontiguousarray(a)
    indptr = cupy.zeros(m + 1, dtype=numpy.int32)
    info = cupy.zeros(m * n + 1, dtype=numpy.int32)
    cupy_dense2csr_step1()(m, n, a, indptr, info)
    indptr = cupy.cumsum(indptr, dtype=numpy.int32)
    info = cupy.cumsum(info, dtype=numpy.int32)
    nnz = int(indptr[-1])
    indices = cupy.empty(nnz, dtype=numpy.int32)
    data = cupy.empty(nnz, dtype=a.dtype)
    cupy_dense2csr_step2()(m, n, a, info, indices, data)
    return csr_matrix((data, indices, indptr), shape=(m, n))

@cupy._util.memoize(for_each_device=True)
def cupy_dense2csr_step1():
    if False:
        while True:
            i = 10
    return cupy.ElementwiseKernel('int32 M, int32 N, T A', 'raw I INDPTR, raw I INFO', '\n        int row = i / N;\n        int col = i % N;\n        if (A != static_cast<T>(0)) {\n            atomicAdd( &(INDPTR[row + 1]), 1 );\n            INFO[i + 1] = 1;\n        }\n        ', 'cupyx_scipy_sparse_dense2csr_step1')

@cupy._util.memoize(for_each_device=True)
def cupy_dense2csr_step2():
    if False:
        while True:
            i = 10
    return cupy.ElementwiseKernel('int32 M, int32 N, T A, raw I INFO', 'raw I INDICES, raw T DATA', '\n        int row = i / N;\n        int col = i % N;\n        if (A != static_cast<T>(0)) {\n            int idx = INFO[i];\n            INDICES[idx] = col;\n            DATA[idx] = A;\n        }\n        ', 'cupyx_scipy_sparse_dense2csr_step2')

@cupy._util.memoize(for_each_device=True)
def _cupy_csr_diagonal():
    if False:
        for i in range(10):
            print('nop')
    return cupy.ElementwiseKernel('int32 k, int32 rows, int32 cols, raw T data, raw I indptr, raw I indices', 'T y', '\n        int row = i;\n        int col = i;\n        if (k < 0) row -= k;\n        if (k > 0) col += k;\n        if (row >= rows || col >= cols) return;\n        int j = find_index_holding_col_in_row(row, col,\n            &(indptr[0]), &(indices[0]));\n        if (j >= 0) {\n            y = data[j];\n        } else {\n            y = static_cast<T>(0);\n        }\n        ', 'cupyx_scipy_sparse_csr_diagonal', preamble=_FIND_INDEX_HOLDING_COL_IN_ROW_)