import numpy
import cupy
from cupy.cuda import device
from cupy.linalg import _util
from cupyx.scipy import sparse

def lschol(A, b):
    if False:
        while True:
            i = 10
    'Solves linear system with cholesky decomposition.\n\n    Find the solution to a large, sparse, linear system of equations.\n    The function solves ``Ax = b``. Given two-dimensional matrix ``A`` is\n    decomposed into ``L * L^*``.\n\n    Args:\n        A (cupy.ndarray or cupyx.scipy.sparse.csr_matrix): The input matrix\n            with dimension ``(N, N)``. Must be positive-definite input matrix.\n            Only symmetric real matrix is supported currently.\n        b (cupy.ndarray): Right-hand side vector.\n\n    Returns:\n        ret (cupy.ndarray): The solution vector ``x``.\n\n    '
    from cupy_backends.cuda.libs import cusolver
    if not sparse.isspmatrix_csr(A):
        A = sparse.csr_matrix(A)
    _util._assert_stacked_square(A)
    _util._assert_cupy_array(b)
    m = A.shape[0]
    if b.ndim != 1 or len(b) != m:
        raise ValueError('b must be 1-d array whose size is same as A')
    if A.dtype == 'f' or A.dtype == 'd':
        dtype = A.dtype
    else:
        dtype = numpy.promote_types(A.dtype, 'f')
    handle = device.get_cusolver_sp_handle()
    nnz = A.nnz
    tol = 1.0
    reorder = 1
    x = cupy.empty(m, dtype=dtype)
    singularity = numpy.empty(1, numpy.int32)
    if dtype == 'f':
        csrlsvchol = cusolver.scsrlsvchol
    else:
        csrlsvchol = cusolver.dcsrlsvchol
    csrlsvchol(handle, m, nnz, A._descr.descriptor, A.data.data.ptr, A.indptr.data.ptr, A.indices.data.ptr, b.data.ptr, tol, reorder, x.data.ptr, singularity.ctypes.data)
    x = x.astype(numpy.float64)
    return x