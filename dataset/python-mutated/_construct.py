import numpy
import cupy
from cupyx.scipy.sparse import _coo
from cupyx.scipy.sparse import _csc
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse import _dia
from cupyx.scipy.sparse import _sputils

def eye(m, n=None, k=0, dtype='d', format=None):
    if False:
        while True:
            i = 10
    'Creates a sparse matrix with ones on diagonal.\n\n    Args:\n        m (int): Number of rows.\n        n (int or None): Number of columns. If it is ``None``,\n            it makes a square matrix.\n        k (int): Diagonal to place ones on.\n        dtype: Type of a matrix to create.\n        format (str or None): Format of the result, e.g. ``format="csr"``.\n\n    Returns:\n        cupyx.scipy.sparse.spmatrix: Created sparse matrix.\n\n    .. seealso:: :func:`scipy.sparse.eye`\n\n    '
    if n is None:
        n = m
    (m, n) = (int(m), int(n))
    if m == n and k == 0:
        if format in ['csr', 'csc']:
            indptr = cupy.arange(n + 1, dtype='i')
            indices = cupy.arange(n, dtype='i')
            data = cupy.ones(n, dtype=dtype)
            if format == 'csr':
                cls = _csr.csr_matrix
            else:
                cls = _csc.csc_matrix
            return cls((data, indices, indptr), (n, n))
        elif format == 'coo':
            row = cupy.arange(n, dtype='i')
            col = cupy.arange(n, dtype='i')
            data = cupy.ones(n, dtype=dtype)
            return _coo.coo_matrix((data, (row, col)), (n, n))
    diags = cupy.ones((1, max(0, min(m + k, n))), dtype=dtype)
    return spdiags(diags, k, m, n).asformat(format)

def identity(n, dtype='d', format=None):
    if False:
        print('Hello World!')
    'Creates an identity matrix in sparse format.\n\n    .. note::\n       Currently it only supports csr, csc and coo formats.\n\n    Args:\n        n (int): Number of rows and columns.\n        dtype: Type of a matrix to create.\n        format (str or None): Format of the result, e.g. ``format="csr"``.\n\n    Returns:\n        cupyx.scipy.sparse.spmatrix: Created identity matrix.\n\n    .. seealso:: :func:`scipy.sparse.identity`\n\n    '
    return eye(n, n, dtype=dtype, format=format)

def spdiags(data, diags, m, n, format=None):
    if False:
        return 10
    'Creates a sparse matrix from diagonals.\n\n    Args:\n        data (cupy.ndarray): Matrix diagonals stored row-wise.\n        diags (cupy.ndarray): Diagonals to set.\n        m (int): Number of rows.\n        n (int): Number of cols.\n        format (str or None): Sparse format, e.g. ``format="csr"``.\n\n    Returns:\n        cupyx.scipy.sparse.spmatrix: Created sparse matrix.\n\n    .. seealso:: :func:`scipy.sparse.spdiags`\n\n    '
    return _dia.dia_matrix((data, diags), shape=(m, n)).asformat(format)

def _compressed_sparse_stack(blocks, axis):
    if False:
        while True:
            i = 10
    'Fast path for stacking CSR/CSC matrices\n    (i) vstack for CSR, (ii) hstack for CSC.\n    '
    other_axis = 1 if axis == 0 else 0
    data = cupy.concatenate([b.data for b in blocks])
    constant_dim = blocks[0].shape[other_axis]
    idx_dtype = _sputils.get_index_dtype(arrays=[b.indptr for b in blocks], maxval=max(data.size, constant_dim))
    indices = cupy.empty(data.size, dtype=idx_dtype)
    indptr = cupy.empty(sum((b.shape[axis] for b in blocks)) + 1, dtype=idx_dtype)
    last_indptr = idx_dtype(0)
    sum_dim = 0
    sum_indices = 0
    for b in blocks:
        if b.shape[other_axis] != constant_dim:
            raise ValueError('incompatible dimensions for axis %d' % other_axis)
        indices[sum_indices:sum_indices + b.indices.size] = b.indices
        sum_indices += b.indices.size
        idxs = slice(sum_dim, sum_dim + b.shape[axis])
        indptr[idxs] = b.indptr[:-1]
        indptr[idxs] += last_indptr
        sum_dim += b.shape[axis]
        last_indptr += b.indptr[-1]
    indptr[-1] = last_indptr
    if axis == 0:
        return _csr.csr_matrix((data, indices, indptr), shape=(sum_dim, constant_dim))
    else:
        return _csc.csc_matrix((data, indices, indptr), shape=(constant_dim, sum_dim))

def hstack(blocks, format=None, dtype=None):
    if False:
        i = 10
        return i + 15
    'Stacks sparse matrices horizontally (column wise)\n\n    Args:\n        blocks (sequence of cupyx.scipy.sparse.spmatrix):\n            sparse matrices to stack\n\n        format (str):\n            sparse format of the result (e.g. "csr")\n            by default an appropriate sparse matrix format is returned.\n            This choice is subject to change.\n        dtype (dtype, optional):\n            The data-type of the output matrix.  If not given, the dtype is\n            determined from that of ``blocks``.\n\n    Returns:\n        cupyx.scipy.sparse.spmatrix: the stacked sparse matrix\n\n    .. seealso:: :func:`scipy.sparse.hstack`\n\n    Examples:\n        >>> from cupy import array\n        >>> from cupyx.scipy.sparse import csr_matrix, hstack\n        >>> A = csr_matrix(array([[1., 2.], [3., 4.]]))\n        >>> B = csr_matrix(array([[5.], [6.]]))\n        >>> hstack([A, B]).toarray()\n        array([[1., 2., 5.],\n               [3., 4., 6.]])\n    '
    return bmat([blocks], format=format, dtype=dtype)

def vstack(blocks, format=None, dtype=None):
    if False:
        while True:
            i = 10
    'Stacks sparse matrices vertically (row wise)\n\n    Args:\n        blocks (sequence of cupyx.scipy.sparse.spmatrix)\n            sparse matrices to stack\n        format (str, optional):\n            sparse format of the result (e.g. "csr")\n            by default an appropriate sparse matrix format is returned.\n            This choice is subject to change.\n        dtype (dtype, optional):\n            The data-type of the output matrix.  If not given, the dtype is\n            determined from that of `blocks`.\n\n    Returns:\n        cupyx.scipy.sparse.spmatrix: the stacked sparse matrix\n\n    .. seealso:: :func:`scipy.sparse.vstack`\n\n    Examples:\n        >>> from cupy import array\n        >>> from cupyx.scipy.sparse import csr_matrix, vstack\n        >>> A = csr_matrix(array([[1., 2.], [3., 4.]]))\n        >>> B = csr_matrix(array([[5., 6.]]))\n        >>> vstack([A, B]).toarray()\n        array([[1., 2.],\n               [3., 4.],\n               [5., 6.]])\n    '
    return bmat([[b] for b in blocks], format=format, dtype=dtype)

def bmat(blocks, format=None, dtype=None):
    if False:
        print('Hello World!')
    'Builds a sparse matrix from sparse sub-blocks\n\n    Args:\n        blocks (array_like):\n            Grid of sparse matrices with compatible shapes.\n            An entry of None implies an all-zero matrix.\n        format ({\'bsr\', \'coo\', \'csc\', \'csr\', \'dia\', \'dok\', \'lil\'}, optional):\n            The sparse format of the result (e.g. "csr").  By default an\n            appropriate sparse matrix format is returned.\n            This choice is subject to change.\n        dtype (dtype, optional):\n            The data-type of the output matrix.  If not given, the dtype is\n            determined from that of `blocks`.\n    Returns:\n        bmat (sparse matrix)\n\n    .. seealso:: :func:`scipy.sparse.bmat`\n\n    Examples:\n        >>> from cupy import array\n        >>> from cupyx.scipy.sparse import csr_matrix, bmat\n        >>> A = csr_matrix(array([[1., 2.], [3., 4.]]))\n        >>> B = csr_matrix(array([[5.], [6.]]))\n        >>> C = csr_matrix(array([[7.]]))\n        >>> bmat([[A, B], [None, C]]).toarray()\n        array([[1., 2., 5.],\n               [3., 4., 6.],\n               [0., 0., 7.]])\n        >>> bmat([[A, None], [None, C]]).toarray()\n        array([[1., 2., 0.],\n               [3., 4., 0.],\n               [0., 0., 7.]])\n\n    '
    M = len(blocks)
    N = len(blocks[0])
    blocks_flat = []
    for m in range(M):
        for n in range(N):
            if blocks[m][n] is not None:
                blocks_flat.append(blocks[m][n])
    if len(blocks_flat) == 0:
        return _coo.coo_matrix((0, 0), dtype=dtype)
    if N == 1 and format in (None, 'csr') and all((isinstance(b, _csr.csr_matrix) for b in blocks_flat)):
        A = _compressed_sparse_stack(blocks_flat, 0)
        if dtype is not None:
            A = A.astype(dtype)
        return A
    elif M == 1 and format in (None, 'csc') and all((isinstance(b, _csc.csc_matrix) for b in blocks_flat)):
        A = _compressed_sparse_stack(blocks_flat, 1)
        if dtype is not None:
            A = A.astype(dtype)
        return A
    block_mask = numpy.zeros((M, N), dtype=bool)
    brow_lengths = numpy.zeros(M + 1, dtype=numpy.int64)
    bcol_lengths = numpy.zeros(N + 1, dtype=numpy.int64)
    for i in range(M):
        for j in range(N):
            if blocks[i][j] is not None:
                A = _coo.coo_matrix(blocks[i][j])
                blocks[i][j] = A
                block_mask[i][j] = True
                if brow_lengths[i + 1] == 0:
                    brow_lengths[i + 1] = A.shape[0]
                elif brow_lengths[i + 1] != A.shape[0]:
                    msg = 'blocks[{i},:] has incompatible row dimensions. Got blocks[{i},{j}].shape[0] == {got}, expected {exp}.'.format(i=i, j=j, exp=brow_lengths[i + 1], got=A.shape[0])
                    raise ValueError(msg)
                if bcol_lengths[j + 1] == 0:
                    bcol_lengths[j + 1] = A.shape[1]
                elif bcol_lengths[j + 1] != A.shape[1]:
                    msg = 'blocks[:,{j}] has incompatible row dimensions. Got blocks[{i},{j}].shape[1] == {got}, expected {exp}.'.format(i=i, j=j, exp=bcol_lengths[j + 1], got=A.shape[1])
                    raise ValueError(msg)
    nnz = sum((block.nnz for block in blocks_flat))
    if dtype is None:
        all_dtypes = [blk.dtype for blk in blocks_flat]
        dtype = _sputils.upcast(*all_dtypes) if all_dtypes else None
    row_offsets = numpy.cumsum(brow_lengths)
    col_offsets = numpy.cumsum(bcol_lengths)
    shape = (row_offsets[-1], col_offsets[-1])
    data = cupy.empty(nnz, dtype=dtype)
    idx_dtype = _sputils.get_index_dtype(maxval=max(shape))
    row = cupy.empty(nnz, dtype=idx_dtype)
    col = cupy.empty(nnz, dtype=idx_dtype)
    nnz = 0
    (ii, jj) = numpy.nonzero(block_mask)
    for (i, j) in zip(ii, jj):
        B = blocks[int(i)][int(j)]
        idx = slice(nnz, nnz + B.nnz)
        data[idx] = B.data
        row[idx] = B.row + row_offsets[i]
        col[idx] = B.col + col_offsets[j]
        nnz += B.nnz
    return _coo.coo_matrix((data, (row, col)), shape=shape).asformat(format)

def random(m, n, density=0.01, format='coo', dtype=None, random_state=None, data_rvs=None):
    if False:
        return 10
    'Generates a random sparse matrix.\n\n    This function generates a random sparse matrix. First it selects non-zero\n    elements with given density ``density`` from ``(m, n)`` elements.\n    So the number of non-zero elements ``k`` is ``k = m * n * density``.\n    Value of each element is selected with ``data_rvs`` function.\n\n    Args:\n        m (int): Number of rows.\n        n (int): Number of cols.\n        density (float): Ratio of non-zero entries.\n        format (str): Matrix format.\n        dtype (~cupy.dtype): Type of the returned matrix values.\n        random_state (cupy.random.RandomState or int):\n            State of random number generator.\n            If an integer is given, the method makes a new state for random\n            number generator and uses it.\n            If it is not given, the default state is used.\n            This state is used to generate random indexes for nonzero entries.\n        data_rvs (callable): A function to generate data for a random matrix.\n            If it is not given, `random_state.rand` is used.\n\n    Returns:\n        cupyx.scipy.sparse.spmatrix: Generated matrix.\n\n    .. seealso:: :func:`scipy.sparse.random`\n\n    '
    if density < 0 or density > 1:
        raise ValueError('density expected to be 0 <= density <= 1')
    dtype = cupy.dtype(dtype)
    if dtype.char not in 'fd':
        raise NotImplementedError('type %s not supported' % dtype)
    mn = m * n
    k = int(density * m * n)
    if random_state is None:
        random_state = cupy.random
    elif isinstance(random_state, (int, cupy.integer)):
        random_state = cupy.random.RandomState(random_state)
    if data_rvs is None:
        data_rvs = random_state.rand
    ind = random_state.choice(mn, size=k, replace=False)
    j = ind // m
    i = ind - j * m
    vals = data_rvs(k).astype(dtype)
    return _coo.coo_matrix((vals, (i, j)), shape=(m, n)).asformat(format)

def rand(m, n, density=0.01, format='coo', dtype=None, random_state=None):
    if False:
        i = 10
        return i + 15
    'Generates a random sparse matrix.\n\n    See :func:`cupyx.scipy.sparse.random` for detail.\n\n    Args:\n        m (int): Number of rows.\n        n (int): Number of cols.\n        density (float): Ratio of non-zero entries.\n        format (str): Matrix format.\n        dtype (~cupy.dtype): Type of the returned matrix values.\n        random_state (cupy.random.RandomState or int):\n            State of random number generator.\n            If an integer is given, the method makes a new state for random\n            number generator and uses it.\n            If it is not given, the default state is used.\n            This state is used to generate random indexes for nonzero entries.\n\n    Returns:\n        cupyx.scipy.sparse.spmatrix: Generated matrix.\n\n    .. seealso:: :func:`scipy.sparse.rand`\n    .. seealso:: :func:`cupyx.scipy.sparse.random`\n\n    '
    return random(m, n, density, format, dtype, random_state)

def diags(diagonals, offsets=0, shape=None, format=None, dtype=None):
    if False:
        while True:
            i = 10
    'Construct a sparse matrix from diagonals.\n\n    Args:\n        diagonals (sequence of array_like):\n            Sequence of arrays containing the matrix diagonals, corresponding\n            to `offsets`.\n        offsets (sequence of int or an int):\n            Diagonals to set:\n                - k = 0  the main diagonal (default)\n                - k > 0  the k-th upper diagonal\n                - k < 0  the k-th lower diagonal\n        shape (tuple of int):\n            Shape of the result. If omitted, a square matrix large enough\n            to contain the diagonals is returned.\n        format ({"dia", "csr", "csc", "lil", ...}):\n            Matrix format of the result.  By default (format=None) an\n            appropriate sparse matrix format is returned.  This choice is\n            subject to change.\n        dtype (dtype): Data type of the matrix.\n\n    Returns:\n        cupyx.scipy.sparse.spmatrix: Generated matrix.\n\n    Notes:\n        This function differs from `spdiags` in the way it handles\n        off-diagonals.\n\n        The result from `diags` is the sparse equivalent of::\n\n            cupy.diag(diagonals[0], offsets[0])\n            + ...\n            + cupy.diag(diagonals[k], offsets[k])\n\n        Repeated diagonal offsets are disallowed.\n    '
    if _sputils.isscalarlike(offsets):
        if len(diagonals) == 0 or _sputils.isscalarlike(diagonals[0]):
            diagonals = [cupy.atleast_1d(diagonals)]
        else:
            raise ValueError('Different number of diagonals and offsets.')
    else:
        diagonals = list(map(cupy.atleast_1d, diagonals))
    if isinstance(offsets, cupy.ndarray):
        offsets = offsets.get()
    offsets = numpy.atleast_1d(offsets)
    if len(diagonals) != len(offsets):
        raise ValueError('Different number of diagonals and offsets.')
    if shape is None:
        m = len(diagonals[0]) + abs(int(offsets[0]))
        shape = (m, m)
    if dtype is None:
        dtype = cupy.common_type(*diagonals)
    (m, n) = shape
    M = max([min(m + offset, n - offset) + max(0, offset) for offset in offsets])
    M = max(0, M)
    data_arr = cupy.zeros((len(offsets), M), dtype=dtype)
    K = min(m, n)
    for (j, diagonal) in enumerate(diagonals):
        offset = offsets[j]
        k = max(0, offset)
        length = min(m + offset, n - offset, K)
        if length < 0:
            raise ValueError('Offset %d (index %d) out of bounds' % (offset, j))
        try:
            data_arr[j, k:k + length] = diagonal[..., :length]
        except ValueError:
            if len(diagonal) != length and len(diagonal) != 1:
                raise ValueError('Diagonal length (index %d: %d at offset %d) does not agree with matrix size (%d, %d).' % (j, len(diagonal), offset, m, n))
            raise
    return _dia.dia_matrix((data_arr, offsets), shape=(m, n)).asformat(format)

def kron(A, B, format=None):
    if False:
        while True:
            i = 10
    'Kronecker product of sparse matrices A and B.\n\n    Args:\n        A (cupyx.scipy.sparse.spmatrix): a sparse matrix.\n        B (cupyx.scipy.sparse.spmatrix): a sparse matrix.\n        format (str): the format of the returned sparse matrix.\n\n    Returns:\n        cupyx.scipy.sparse.spmatrix:\n            Generated sparse matrix with the specified ``format``.\n\n    .. seealso:: :func:`scipy.sparse.kron`\n\n    '
    A = _coo.coo_matrix(A)
    B = _coo.coo_matrix(B)
    out_shape = (A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])
    if A.nnz == 0 or B.nnz == 0:
        return _coo.coo_matrix(out_shape).asformat(format)
    if max(out_shape[0], out_shape[1]) > cupy.iinfo('int32').max:
        dtype = cupy.int64
    else:
        dtype = cupy.int32
    row = A.row.astype(dtype, copy=True) * B.shape[0]
    row = row.repeat(B.nnz)
    col = A.col.astype(dtype, copy=True) * B.shape[1]
    col = col.repeat(B.nnz)
    data = A.data.repeat(B.nnz)
    (row, col) = (row.reshape(-1, B.nnz), col.reshape(-1, B.nnz))
    row += B.row
    col += B.col
    (row, col) = (row.ravel(), col.ravel())
    data = data.reshape(-1, B.nnz) * B.data
    data = data.ravel()
    return _coo.coo_matrix((data, (row, col)), shape=out_shape).asformat(format)

def kronsum(A, B, format=None):
    if False:
        i = 10
        return i + 15
    'Kronecker sum of sparse matrices A and B.\n\n    Kronecker sum is the sum of two Kronecker products\n    ``kron(I_n, A) + kron(B, I_m)``, where ``I_n`` and ``I_m`` are identity\n    matrices.\n\n    Args:\n        A (cupyx.scipy.sparse.spmatrix): a sparse matrix.\n        B (cupyx.scipy.sparse.spmatrix): a sparse matrix.\n        format (str): the format of the returned sparse matrix.\n\n    Returns:\n        cupyx.scipy.sparse.spmatrix:\n            Generated sparse matrix with the specified ``format``.\n\n    .. seealso:: :func:`scipy.sparse.kronsum`\n\n    '
    A = _coo.coo_matrix(A)
    B = _coo.coo_matrix(B)
    if A.shape[0] != A.shape[1]:
        raise ValueError('A is not square matrix')
    if B.shape[0] != B.shape[1]:
        raise ValueError('B is not square matrix')
    dtype = _sputils.upcast(A.dtype, B.dtype)
    L = kron(eye(B.shape[0], dtype=dtype), A, format=format)
    R = kron(B, eye(A.shape[0], dtype=dtype), format=format)
    return (L + R).asformat(format)