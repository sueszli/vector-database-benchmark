import string
import warnings
import numpy
try:
    import scipy.sparse
    scipy_available = True
except ImportError:
    scipy_available = False
import cupy
import cupyx
from cupy import _core
from cupy._core import _scalar
from cupy._creation import basic
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _coo
from cupyx.scipy.sparse import _data as sparse_data
from cupyx.scipy.sparse import _sputils
from cupyx.scipy.sparse import _util
from cupyx.scipy.sparse import _index

class _compressed_sparse_matrix(sparse_data._data_matrix, sparse_data._minmax_mixin, _index.IndexMixin):
    _max_min_reduction_code = '\n        extern "C" __global__\n        void ${func}(double* data, int* x, int* y, int length,\n                           double* z) {\n            // Get the index of the block\n            int tid = blockIdx.x * blockDim.x + threadIdx.x;\n\n            // Calculate the block length\n            int block_length = y[tid] - x[tid];\n\n            // Select initial value based on the block density\n            double running_value = 0;\n            if (${cond}){\n                running_value = data[x[tid]];\n            } else {\n                running_value = 0;\n            }\n\n            // Iterate over the block and update\n            for (int entry = x[tid]; entry < y[tid]; entry++){\n                if (data[entry] != data[entry]){\n                    // Check for NaN\n                    running_value = nan("");\n                    break;\n                } else {\n                    // Check for a value update\n                    if (data[entry] ${op} running_value){\n                        running_value = data[entry];\n                    }\n                }\n            }\n\n            // Store in the return function\n            z[tid] = running_value;\n        }'
    _max_reduction_kern = _core.RawKernel(string.Template(_max_min_reduction_code).substitute(func='max_reduction', op='>', cond='block_length == length'), 'max_reduction')
    _max_nonzero_reduction_kern = _core.RawKernel(string.Template(_max_min_reduction_code).substitute(func='max_nonzero_reduction', op='>', cond='block_length > 0'), 'max_nonzero_reduction')
    _min_reduction_kern = _core.RawKernel(string.Template(_max_min_reduction_code).substitute(func='min_reduction', op='<', cond='block_length == length'), 'min_reduction')
    _min_nonzero_reduction_kern = _core.RawKernel(string.Template(_max_min_reduction_code).substitute(func='min_nonzero_reduction', op='<', cond='block_length > 0'), 'min_nonzero_reduction')
    _argmax_argmin_code = '\n        template<typename T1, typename T2> __global__ void\n        ${func}_arg_reduction(T1* data, int* indices, int* x, int* y,\n                              int length, T2* z) {\n            // Get the index of the block\n            int tid = blockIdx.x * blockDim.x + threadIdx.x;\n\n            // Calculate the block length\n            int block_length = y[tid] - x[tid];\n\n            // Select initial value based on the block density\n            int data_index = 0;\n            double data_value = 0;\n\n            if (block_length == length){\n                // Block is dense. Fill the first value\n                data_value = data[x[tid]];\n                data_index = indices[x[tid]];\n            } else if (block_length > 0)  {\n                // Block has at least one zero. Assign first occurrence as the\n                // starting reference\n                data_value = 0;\n                for (data_index = 0; data_index < length; data_index++){\n                    if (data_index != indices[x[tid] + data_index] ||\n                        x[tid] + data_index >= y[tid]){\n                        break;\n                    }\n                }\n            } else {\n                // Zero valued array\n                data_value = 0;\n                data_index = 0;\n            }\n\n            // Iterate over the section of the sparse matrix\n            for (int entry = x[tid]; entry < y[tid]; entry++){\n                if (data[entry] != data[entry]){\n                    // Check for NaN\n                    data_value = nan("");\n                    data_index = 0;\n                    break;\n                } else {\n                    // Check for a value update\n                    if (data[entry] ${op} data_value){\n                        data_index = indices[entry];\n                        data_value = data[entry];\n                    }\n                }\n            }\n\n            // Store in the return function\n            z[tid] = data_index;\n        }'
    _max_arg_reduction_mod = _core.RawModule(code=string.Template(_argmax_argmin_code).substitute(func='max', op='>'), options=('-std=c++11',), name_expressions=['max_arg_reduction<float, int>', 'max_arg_reduction<float, long long>', 'max_arg_reduction<double, int>', 'max_arg_reduction<double, long long>'])
    _min_arg_reduction_mod = _core.RawModule(code=string.Template(_argmax_argmin_code).substitute(func='min', op='<'), options=('-std=c++11',), name_expressions=['min_arg_reduction<float, int>', 'min_arg_reduction<float, long long>', 'min_arg_reduction<double, int>', 'min_arg_reduction<double, long long>'])
    _has_sorted_indices_kern = _core.ElementwiseKernel('raw T indptr, raw T indices', 'bool diff', '\n        bool diff_out = true;\n        for (T jj = indptr[i]; jj < indptr[i+1] - 1; jj++) {\n            if (indices[jj] > indices[jj+1]){\n                diff_out = false;\n            }\n        }\n        diff = diff_out;\n        ', 'cupyx_scipy_sparse_has_sorted_indices')
    _has_canonical_format_kern = _core.ElementwiseKernel('raw T indptr, raw T indices', 'bool diff', '\n        bool diff_out = true;\n        if (indptr[i] > indptr[i+1]) {\n            diff = false;\n            return;\n        }\n        for (T jj = indptr[i]; jj < indptr[i+1] - 1; jj++) {\n            if (indices[jj] >= indices[jj+1]) {\n                diff_out = false;\n            }\n        }\n        diff = diff_out;\n        ', 'cupyx_scipy_sparse_has_canonical_format')

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        if False:
            print('Hello World!')
        from cupyx import cusparse
        if shape is not None:
            if not _util.isshape(shape):
                raise ValueError('invalid shape (must be a 2-tuple of int)')
            shape = (int(shape[0]), int(shape[1]))
        if _base.issparse(arg1):
            x = arg1.asformat(self.format)
            data = x.data
            indices = x.indices
            indptr = x.indptr
            if arg1.format != self.format:
                copy = False
            if shape is None:
                shape = arg1.shape
        elif _util.isshape(arg1):
            (m, n) = arg1
            (m, n) = (int(m), int(n))
            data = basic.zeros(0, dtype if dtype else 'd')
            indices = basic.zeros(0, 'i')
            indptr = basic.zeros(self._swap(m, n)[0] + 1, dtype='i')
            shape = (m, n)
            copy = False
        elif scipy_available and scipy.sparse.issparse(arg1):
            x = arg1.asformat(self.format)
            data = cupy.array(x.data)
            indices = cupy.array(x.indices, dtype='i')
            indptr = cupy.array(x.indptr, dtype='i')
            copy = False
            if shape is None:
                shape = arg1.shape
        elif isinstance(arg1, tuple) and len(arg1) == 2:
            sp_coo = _coo.coo_matrix(arg1, shape=shape, dtype=dtype, copy=copy)
            sp_compressed = sp_coo.asformat(self.format)
            data = sp_compressed.data
            indices = sp_compressed.indices
            indptr = sp_compressed.indptr
        elif isinstance(arg1, tuple) and len(arg1) == 3:
            (data, indices, indptr) = arg1
            if not (_base.isdense(data) and data.ndim == 1 and _base.isdense(indices) and (indices.ndim == 1) and _base.isdense(indptr) and (indptr.ndim == 1)):
                raise ValueError('data, indices, and indptr should be 1-D')
            if len(data) != len(indices):
                raise ValueError('indices and data should have the same size')
        elif _base.isdense(arg1):
            if arg1.ndim > 2:
                raise TypeError('expected dimension <= 2 array or matrix')
            elif arg1.ndim == 1:
                arg1 = arg1[None]
            elif arg1.ndim == 0:
                arg1 = arg1[None, None]
            (data, indices, indptr) = self._convert_dense(arg1)
            copy = False
            if shape is None:
                shape = arg1.shape
        else:
            raise ValueError('Unsupported initializer format')
        if dtype is None:
            dtype = data.dtype
        else:
            dtype = numpy.dtype(dtype)
        if dtype.char not in '?fdFD':
            raise ValueError('Only bool, float32, float64, complex64 and complex128 are supported')
        data = data.astype(dtype, copy=copy)
        sparse_data._data_matrix.__init__(self, data)
        self.indices = indices.astype('i', copy=copy)
        self.indptr = indptr.astype('i', copy=copy)
        if shape is None:
            shape = self._swap(len(indptr) - 1, int(indices.max()) + 1)
        (major, minor) = self._swap(*shape)
        if len(indptr) != major + 1:
            raise ValueError('index pointer size (%d) should be (%d)' % (len(indptr), major + 1))
        self._descr = cusparse.MatDescriptor.create()
        self._shape = shape

    def _with_data(self, data, copy=True):
        if False:
            return 10
        if copy:
            return self.__class__((data, self.indices.copy(), self.indptr.copy()), shape=self.shape, dtype=data.dtype)
        else:
            return self.__class__((data, self.indices, self.indptr), shape=self.shape, dtype=data.dtype)

    def _convert_dense(self, x):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def _swap(self, x, y):
        if False:
            return 10
        raise NotImplementedError

    def _add_sparse(self, other, alpha, beta):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def _add(self, other, lhs_negative, rhs_negative):
        if False:
            return 10
        if cupy.isscalar(other):
            if other == 0:
                if lhs_negative:
                    return -self
                else:
                    return self.copy()
            else:
                raise NotImplementedError('adding a nonzero scalar to a sparse matrix is not supported')
        elif _base.isspmatrix(other):
            alpha = -1 if lhs_negative else 1
            beta = -1 if rhs_negative else 1
            return self._add_sparse(other, alpha, beta)
        elif _base.isdense(other):
            if lhs_negative:
                if rhs_negative:
                    return -self.todense() - other
                else:
                    return other - self.todense()
            elif rhs_negative:
                return self.todense() - other
            else:
                return self.todense() + other
        else:
            return NotImplemented

    def __add__(self, other):
        if False:
            return 10
        return self._add(other, False, False)

    def __radd__(self, other):
        if False:
            i = 10
            return i + 15
        return self._add(other, False, False)

    def __sub__(self, other):
        if False:
            return 10
        return self._add(other, False, True)

    def __rsub__(self, other):
        if False:
            while True:
                i = 10
        return self._add(other, True, False)

    def _get_intXint(self, row, col):
        if False:
            while True:
                i = 10
        (major, minor) = self._swap(row, col)
        (data, indices, _) = _index._get_csr_submatrix_major_axis(self.data, self.indices, self.indptr, major, major + 1)
        dtype = data.dtype
        res = cupy.zeros((), dtype=dtype)
        if dtype.kind == 'c':
            _index._compress_getitem_complex_kern(data.real, data.imag, indices, minor, res.real, res.imag)
        else:
            _index._compress_getitem_kern(data, indices, minor, res)
        return res

    def _get_sliceXslice(self, row, col):
        if False:
            return 10
        (major, minor) = self._swap(row, col)
        copy = major.step in (1, None)
        return self._major_slice(major)._minor_slice(minor, copy=copy)

    def _get_arrayXarray(self, row, col, not_found_val=0):
        if False:
            return 10
        idx_dtype = self.indices.dtype
        (M, N) = self._swap(*self.shape)
        (major, minor) = self._swap(row, col)
        major = major.astype(idx_dtype, copy=False)
        minor = minor.astype(idx_dtype, copy=False)
        val = _index._csr_sample_values(M, N, self.indptr, self.indices, self.data, major.ravel(), minor.ravel(), not_found_val)
        if major.ndim == 1:
            return cupy.expand_dims(val, 0)
        return self.__class__(val.reshape(major.shape))

    def _get_columnXarray(self, row, col):
        if False:
            while True:
                i = 10
        (major, minor) = self._swap(row, col)
        return self._major_index_fancy(major)._minor_index_fancy(minor)

    def _major_index_fancy(self, idx):
        if False:
            return 10
        'Index along the major axis where idx is an array of ints.\n        '
        (_, N) = self._swap(*self.shape)
        M = idx.size
        new_shape = self._swap(M, N)
        if self.nnz == 0 or M == 0:
            return self.__class__(new_shape, dtype=self.dtype)
        return self.__class__(_index._csr_row_index(self.data, self.indices, self.indptr, idx), shape=new_shape, copy=False)

    def _minor_index_fancy(self, idx):
        if False:
            while True:
                i = 10
        'Index along the minor axis where idx is an array of ints.\n        '
        (M, _) = self._swap(*self.shape)
        N = idx.size
        new_shape = self._swap(M, N)
        if self.nnz == 0 or N == 0:
            return self.__class__(new_shape, dtype=self.dtype)
        if idx.size * M < self.nnz:
            pass
        return self._tocsx()._major_index_fancy(idx)._tocsx()

    def _major_slice(self, idx, copy=False):
        if False:
            i = 10
            return i + 15
        'Index along the major axis where idx is a slice object.\n        '
        (M, N) = self._swap(*self.shape)
        (start, stop, step) = idx.indices(M)
        if start == 0 and stop == M and (step == 1):
            return self.copy() if copy else self
        M = len(range(start, stop, step))
        new_shape = self._swap(M, N)
        if step == 1:
            if M == 0 or self.nnz == 0:
                return self.__class__(new_shape, dtype=self.dtype)
            return self.__class__(_index._get_csr_submatrix_major_axis(self.data, self.indices, self.indptr, start, stop), shape=new_shape, copy=copy)
        rows = cupy.arange(start, stop, step, dtype=self.indptr.dtype)
        return self._major_index_fancy(rows)

    def _minor_slice(self, idx, copy=False):
        if False:
            print('Hello World!')
        'Index along the minor axis where idx is a slice object.\n        '
        (M, N) = self._swap(*self.shape)
        (start, stop, step) = idx.indices(N)
        if start == 0 and stop == N and (step == 1):
            return self.copy() if copy else self
        N = len(range(start, stop, step))
        new_shape = self._swap(M, N)
        if N == 0 or self.nnz == 0:
            return self.__class__(new_shape, dtype=self.dtype)
        if step == 1:
            return self.__class__(_index._get_csr_submatrix_minor_axis(self.data, self.indices, self.indptr, start, stop), shape=new_shape, copy=False)
        cols = cupy.arange(start, stop, step, dtype=self.indices.dtype)
        return self._minor_index_fancy(cols)

    def _set_intXint(self, row, col, x):
        if False:
            for i in range(10):
                print('nop')
        (i, j) = self._swap(row, col)
        self._set_many(i, j, x)

    def _set_arrayXarray(self, row, col, x):
        if False:
            return 10
        (i, j) = self._swap(row, col)
        self._set_many(i, j, x)

    def _set_arrayXarray_sparse(self, row, col, x):
        if False:
            for i in range(10):
                print('nop')
        self._zero_many(*self._swap(row, col))
        (M, N) = row.shape
        broadcast_row = M != 1 and x.shape[0] == 1
        broadcast_col = N != 1 and x.shape[1] == 1
        (r, c) = (x.row, x.col)
        x = cupy.asarray(x.data, dtype=self.dtype)
        if broadcast_row:
            r = cupy.repeat(cupy.arange(M), r.size)
            c = cupy.tile(c, M)
            x = cupy.tile(x, M)
        if broadcast_col:
            r = cupy.repeat(r, N)
            c = cupy.tile(cupy.arange(N), c.size)
            x = cupy.repeat(x, N)
        (i, j) = self._swap(row[r, c], col[r, c])
        self._set_many(i, j, x)

    def _prepare_indices(self, i, j):
        if False:
            print('Hello World!')
        (M, N) = self._swap(*self.shape)

        def check_bounds(indices, bound):
            if False:
                print('Hello World!')
            idx = indices.max()
            if idx >= bound:
                raise IndexError('index (%d) out of range (>= %d)' % (idx, bound))
            idx = indices.min()
            if idx < -bound:
                raise IndexError('index (%d) out of range (< -%d)' % (idx, bound))
        i = cupy.array(i, dtype=self.indptr.dtype, copy=True, ndmin=1).ravel()
        j = cupy.array(j, dtype=self.indices.dtype, copy=True, ndmin=1).ravel()
        check_bounds(i, M)
        check_bounds(j, N)
        return (i, j, M, N)

    def _set_many(self, i, j, x):
        if False:
            print('Hello World!')
        'Sets value at each (i, j) to x\n        Here (i,j) index major and minor respectively, and must not contain\n        duplicate entries.\n        '
        (i, j, M, N) = self._prepare_indices(i, j)
        x = cupy.array(x, dtype=self.dtype, copy=True, ndmin=1).ravel()
        new_sp = cupyx.scipy.sparse.csr_matrix((cupy.arange(self.nnz, dtype=cupy.float32), self.indices, self.indptr), shape=(M, N))
        offsets = new_sp._get_arrayXarray(i, j, not_found_val=-1).astype(cupy.int32).ravel()
        mask = offsets > -1
        self.data[offsets[mask]] = x[mask]
        if mask.all():
            return
        warnings.warn('Changing the sparsity structure of a {}_matrix is expensive.'.format(self.format), _base.SparseEfficiencyWarning)
        mask = ~mask
        i = i[mask]
        i[i < 0] += M
        j = j[mask]
        j[j < 0] += N
        self._insert_many(i, j, x[mask])

    def _zero_many(self, i, j):
        if False:
            return 10
        'Sets value at each (i, j) to zero, preserving sparsity structure.\n        Here (i,j) index major and minor respectively.\n        '
        (i, j, M, N) = self._prepare_indices(i, j)
        new_sp = cupyx.scipy.sparse.csr_matrix((cupy.arange(self.nnz, dtype=cupy.float32), self.indices, self.indptr), shape=(M, N))
        offsets = new_sp._get_arrayXarray(i, j, not_found_val=-1).astype(cupy.int32).ravel()
        self.data[offsets[offsets > -1]] = 0

    def _perform_insert(self, indices_inserts, data_inserts, rows, row_counts, idx_dtype):
        if False:
            print('Hello World!')
        'Insert new elements into current sparse matrix in sorted order'
        indptr_diff = cupy.diff(self.indptr)
        indptr_diff[rows] += row_counts
        new_indptr = cupy.empty(self.indptr.shape, dtype=idx_dtype)
        new_indptr[0] = idx_dtype(0)
        new_indptr[1:] = indptr_diff
        cupy.cumsum(new_indptr, out=new_indptr)
        out_nnz = int(new_indptr[-1])
        new_indices = cupy.empty(out_nnz, dtype=idx_dtype)
        new_data = cupy.empty(out_nnz, dtype=self.data.dtype)
        new_indptr_lookup = cupy.zeros(new_indptr.size, dtype=idx_dtype)
        new_indptr_lookup[1:][rows] = row_counts
        cupy.cumsum(new_indptr_lookup, out=new_indptr_lookup)
        _index._insert_many_populate_arrays(indices_inserts, data_inserts, new_indptr_lookup, self.indptr, self.indices, self.data, new_indptr, new_indices, new_data, size=self.indptr.size - 1)
        self.indptr = new_indptr
        self.indices = new_indices
        self.data = new_data

    def _insert_many(self, i, j, x):
        if False:
            i = 10
            return i + 15
        'Inserts new nonzero at each (i, j) with value x\n        Here (i,j) index major and minor respectively.\n        i, j and x must be non-empty, 1d arrays.\n        Inserts each major group (e.g. all entries per row) at a time.\n        Maintains has_sorted_indices property.\n        Modifies i, j, x in place.\n        '
        order = cupy.argsort(i)
        i = i.take(order)
        j = j.take(order)
        x = x.take(order)
        idx_dtype = _sputils.get_index_dtype((self.indices, self.indptr), maxval=self.nnz + x.size)
        self.indptr = self.indptr.astype(idx_dtype)
        self.indices = self.indices.astype(idx_dtype)
        self.data = self.data.astype(self.dtype)
        (indptr_inserts, indices_inserts, data_inserts) = _index._select_last_indices(i, j, x, idx_dtype)
        (rows, ui_indptr) = cupy.unique(indptr_inserts, return_index=True)
        to_add = cupy.empty(ui_indptr.size + 1, ui_indptr.dtype)
        to_add[-1] = j.size
        to_add[:-1] = ui_indptr
        ui_indptr = to_add
        row_counts = cupy.zeros(ui_indptr.size - 1, dtype=idx_dtype)
        cupy.add.at(row_counts, cupy.searchsorted(rows, indptr_inserts), 1)
        self._perform_insert(indices_inserts, data_inserts, rows, row_counts, idx_dtype)

    def __get_has_canonical_format(self):
        if False:
            return 10
        'Determine whether the matrix has sorted indices and no duplicates.\n\n        Returns\n            bool: ``True`` if the above applies, otherwise ``False``.\n\n        .. note::\n            :attr:`has_canonical_format` implies :attr:`has_sorted_indices`, so\n            if the latter flag is ``False``, so will the former be; if the\n            former is found ``True``, the latter flag is also set.\n\n        .. warning::\n            Getting this property might synchronize the device.\n\n        '
        if self.data.size == 0:
            self._has_canonical_format = True
        elif not getattr(self, '_has_sorted_indices', True):
            self._has_canonical_format = False
        elif not hasattr(self, '_has_canonical_format'):
            is_canonical = self._has_canonical_format_kern(self.indptr, self.indices, size=self.indptr.size - 1)
            self._has_canonical_format = bool(is_canonical.all())
        return self._has_canonical_format

    def __set_has_canonical_format(self, val):
        if False:
            while True:
                i = 10
        'Taken from SciPy as is.'
        self._has_canonical_format = bool(val)
        if val:
            self.has_sorted_indices = True
    has_canonical_format = property(fget=__get_has_canonical_format, fset=__set_has_canonical_format)

    def __get_sorted(self):
        if False:
            i = 10
            return i + 15
        'Determine whether the matrix has sorted indices.\n\n        Returns\n            bool:\n                ``True`` if the indices of the matrix are in sorted order,\n                otherwise ``False``.\n\n        .. warning::\n            Getting this property might synchronize the device.\n\n        '
        if self.data.size == 0:
            self._has_sorted_indices = True
        elif not hasattr(self, '_has_sorted_indices'):
            is_sorted = self._has_sorted_indices_kern(self.indptr, self.indices, size=self.indptr.size - 1)
            self._has_sorted_indices = bool(is_sorted.all())
        return self._has_sorted_indices

    def __set_sorted(self, val):
        if False:
            for i in range(10):
                print('nop')
        self._has_sorted_indices = bool(val)
    has_sorted_indices = property(fget=__get_sorted, fset=__set_sorted)

    def get_shape(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the shape of the matrix.\n\n        Returns:\n            tuple: Shape of the matrix.\n\n        '
        return self._shape

    def getnnz(self, axis=None):
        if False:
            i = 10
            return i + 15
        'Returns the number of stored values, including explicit zeros.\n\n        Args:\n            axis: Not supported yet.\n\n        Returns:\n            int: The number of stored values.\n\n        '
        if axis is None:
            return self.data.size
        else:
            raise ValueError

    def sorted_indices(self):
        if False:
            i = 10
            return i + 15
        'Return a copy of this matrix with sorted indices\n\n        .. warning::\n            Calling this function might synchronize the device.\n        '
        A = self.copy()
        A.sort_indices()
        return A

    def sort_indices(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def sum_duplicates(self):
        if False:
            i = 10
            return i + 15
        'Eliminate duplicate matrix entries by adding them together.\n\n        .. note::\n            This is an *in place* operation.\n\n        .. warning::\n            Calling this function might synchronize the device.\n\n        .. seealso::\n           :meth:`scipy.sparse.csr_matrix.sum_duplicates`,\n           :meth:`scipy.sparse.csc_matrix.sum_duplicates`\n        '
        if self.has_canonical_format:
            return
        coo = self.tocoo()
        coo.sum_duplicates()
        self.__init__(coo.asformat(self.format))
        self.has_canonical_format = True

    def _minor_reduce(self, ufunc, axis, nonzero):
        if False:
            for i in range(10):
                print('nop')
        'Reduce nonzeros with a ufunc over the minor axis when non-empty\n\n        Can be applied to a function of self.data by supplying data parameter.\n        Warning: this does not call sum_duplicates()\n\n        Args:\n            ufunc (object): Function handle giving the operation to be\n                conducted.\n            axis (int): Matrix over which the reduction should be\n                conducted.\n\n        Returns:\n            (cupy.ndarray): Reduce result for nonzeros in each\n            major_index.\n\n        '
        out_shape = self.shape[1 - axis]
        out = cupy.zeros(out_shape).astype(cupy.float64)
        if nonzero:
            kerns = {cupy.amax: self._max_nonzero_reduction_kern, cupy.amin: self._min_nonzero_reduction_kern}
        else:
            kerns = {cupy.amax: self._max_reduction_kern, cupy.amin: self._min_reduction_kern}
        kerns[ufunc]((out_shape,), (1,), (self.data.astype(cupy.float64), self.indptr[:len(self.indptr) - 1], self.indptr[1:], cupy.int64(self.shape[axis]), out))
        return out

    def _arg_minor_reduce(self, ufunc, axis):
        if False:
            for i in range(10):
                print('nop')
        'Reduce nonzeros with a ufunc over the minor axis when non-empty\n\n        Can be applied to a function of self.data by supplying data parameter.\n        Warning: this does not call sum_duplicates()\n\n        Args:\n            ufunc (object): Function handle giving the operation to be\n                conducted.\n            axis (int): Maxtrix over which the reduction should be conducted\n\n        Returns:\n            (cupy.ndarray): Reduce result for nonzeros in each\n            major_index\n\n        '
        out_shape = self.shape[1 - axis]
        out = cupy.zeros(out_shape, dtype=int)
        ker_name = '_arg_reduction<{}, {}>'.format(_scalar.get_typename(self.data.dtype), _scalar.get_typename(out.dtype))
        if ufunc == cupy.argmax:
            ker = self._max_arg_reduction_mod.get_function('max' + ker_name)
        elif ufunc == cupy.argmin:
            ker = self._min_arg_reduction_mod.get_function('min' + ker_name)
        ker((out_shape,), (1,), (self.data, self.indices, self.indptr[:len(self.indptr) - 1], self.indptr[1:], cupy.int64(self.shape[axis]), out))
        return out