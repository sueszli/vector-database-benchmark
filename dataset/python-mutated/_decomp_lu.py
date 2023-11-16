from warnings import warn
import numpy
import cupy
from cupy.cuda import cublas
from cupy.cuda import device
from cupy.cuda import runtime
from cupy.linalg import _util
from cupyx.scipy.linalg import _uarray

@_uarray.implements('lu_factor')
def lu_factor(a, overwrite_a=False, check_finite=True):
    if False:
        while True:
            i = 10
    'LU decomposition.\n\n    Decompose a given two-dimensional square matrix into ``P * L * U``,\n    where ``P`` is a permutation matrix,  ``L`` lower-triangular with\n    unit diagonal elements, and ``U`` upper-triangular matrix.\n\n    Args:\n        a (cupy.ndarray): The input matrix with dimension ``(M, N)``\n        overwrite_a (bool): Allow overwriting data in ``a`` (may enhance\n            performance)\n        check_finite (bool): Whether to check that the input matrices contain\n            only finite numbers. Disabling may give a performance gain, but may\n            result in problems (crashes, non-termination) if the inputs do\n            contain infinities or NaNs.\n\n    Returns:\n        tuple:\n            ``(lu, piv)`` where ``lu`` is a :class:`cupy.ndarray`\n            storing ``U`` in its upper triangle, and ``L`` without\n            unit diagonal elements in its lower triangle, and ``piv`` is\n            a :class:`cupy.ndarray` storing pivot indices representing\n            permutation matrix ``P``. For ``0 <= i < min(M,N)``, row\n            ``i`` of the matrix was interchanged with row ``piv[i]``\n\n    .. seealso:: :func:`scipy.linalg.lu_factor`\n    '
    return _lu_factor(a, overwrite_a, check_finite)

@_uarray.implements('lu')
def lu(a, permute_l=False, overwrite_a=False, check_finite=True):
    if False:
        i = 10
        return i + 15
    'LU decomposition.\n\n    Decomposes a given two-dimensional matrix into ``P @ L @ U``, where ``P``\n    is a permutation matrix, ``L`` is a lower triangular or trapezoidal matrix\n    with unit diagonal, and ``U`` is a upper triangular or trapezoidal matrix.\n\n    Args:\n        a (cupy.ndarray): The input matrix with dimension ``(M, N)``.\n        permute_l (bool): If ``True``, perform the multiplication ``P @ L``.\n        overwrite_a (bool): Allow overwriting data in ``a`` (may enhance\n            performance)\n        check_finite (bool): Whether to check that the input matrices contain\n            only finite numbers. Disabling may give a performance gain, but may\n            result in problems (crashes, non-termination) if the inputs do\n            contain infinities or NaNs.\n\n    Returns:\n        tuple:\n            ``(P, L, U)`` if ``permute_l == False``, otherwise ``(PL, U)``.\n            ``P`` is a :class:`cupy.ndarray` storing permutation matrix with\n            dimension ``(M, M)``. ``L`` is a :class:`cupy.ndarray` storing\n            lower triangular or trapezoidal matrix with unit diagonal with\n            dimension ``(M, K)`` where ``K = min(M, N)``. ``U`` is a\n            :class:`cupy.ndarray` storing upper triangular or trapezoidal\n            matrix with dimension ``(K, N)``. ``PL`` is a :class:`cupy.ndarray`\n            storing permuted ``L`` matrix with dimension ``(M, K)``.\n\n    .. seealso:: :func:`scipy.linalg.lu`\n    '
    (lu, piv) = _lu_factor(a, overwrite_a, check_finite)
    (m, n) = lu.shape
    k = min(m, n)
    (L, U) = _cupy_split_lu(lu)
    if permute_l:
        _cupy_laswp(L, 0, k - 1, piv, -1)
        return (L, U)
    else:
        r_dtype = numpy.float32 if lu.dtype.char in 'fF' else numpy.float64
        P = cupy.diag(cupy.ones((m,), dtype=r_dtype))
        _cupy_laswp(P, 0, k - 1, piv, -1)
        return (P, L, U)

def _lu_factor(a, overwrite_a=False, check_finite=True):
    if False:
        return 10
    from cupy_backends.cuda.libs import cusolver
    a = cupy.asarray(a)
    _util._assert_2d(a)
    dtype = a.dtype
    if dtype.char == 'f':
        getrf = cusolver.sgetrf
        getrf_bufferSize = cusolver.sgetrf_bufferSize
    elif dtype.char == 'd':
        getrf = cusolver.dgetrf
        getrf_bufferSize = cusolver.dgetrf_bufferSize
    elif dtype.char == 'F':
        getrf = cusolver.cgetrf
        getrf_bufferSize = cusolver.cgetrf_bufferSize
    elif dtype.char == 'D':
        getrf = cusolver.zgetrf
        getrf_bufferSize = cusolver.zgetrf_bufferSize
    else:
        msg = 'Only float32, float64, complex64 and complex128 are supported.'
        raise NotImplementedError(msg)
    a = a.astype(dtype, order='F', copy=not overwrite_a)
    if check_finite:
        if a.dtype.kind == 'f' and (not cupy.isfinite(a).all()):
            raise ValueError('array must not contain infs or NaNs')
    cusolver_handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype=numpy.int32)
    (m, n) = a.shape
    ipiv = cupy.empty((min(m, n),), dtype=numpy.intc)
    buffersize = getrf_bufferSize(cusolver_handle, m, n, a.data.ptr, m)
    workspace = cupy.empty(buffersize, dtype=dtype)
    getrf(cusolver_handle, m, n, a.data.ptr, m, workspace.data.ptr, ipiv.data.ptr, dev_info.data.ptr)
    if not runtime.is_hip and dev_info[0] < 0:
        raise ValueError('illegal value in %d-th argument of internal getrf (lu_factor)' % -dev_info[0])
    elif dev_info[0] > 0:
        warn('Diagonal number %d is exactly zero. Singular matrix.' % dev_info[0], RuntimeWarning, stacklevel=2)
    ipiv -= 1
    return (a, ipiv)

def _cupy_split_lu(LU, order='C'):
    if False:
        return 10
    assert LU._f_contiguous
    (m, n) = LU.shape
    k = min(m, n)
    order = 'F' if order == 'F' else 'C'
    L = cupy.empty((m, k), order=order, dtype=LU.dtype)
    U = cupy.empty((k, n), order=order, dtype=LU.dtype)
    size = m * n
    _kernel_cupy_split_lu(LU, m, n, k, L._c_contiguous, L, U, size=size)
    return (L, U)
_device_get_index = '\n__device__ inline int get_index(int row, int col, int num_rows, int num_cols,\n                                bool c_contiguous)\n{\n    if (c_contiguous) {\n        return col + num_cols * row;\n    } else {\n        return row + num_rows * col;\n    }\n}\n'
_kernel_cupy_split_lu = cupy.ElementwiseKernel('raw T LU, int32 M, int32 N, int32 K, bool C_CONTIGUOUS', 'raw T L, raw T U', '\n    // LU: shape: (M, N)\n    // L: shape: (M, K)\n    // U: shape: (K, N)\n    const T* ptr_LU = &(LU[0]);\n    T* ptr_L = &(L[0]);\n    T* ptr_U = &(U[0]);\n    int row, col;\n    if (C_CONTIGUOUS) {\n        row = i / N;\n        col = i % N;\n    } else {\n        row = i % M;\n        col = i / M;\n    }\n    T lu_val = ptr_LU[get_index(row, col, M, N, false)];\n    T l_val, u_val;\n    if (row > col) {\n        l_val = lu_val;\n        u_val = static_cast<T>(0);\n    } else if (row == col) {\n        l_val = static_cast<T>(1);\n        u_val = lu_val;\n    } else {\n        l_val = static_cast<T>(0);\n        u_val = lu_val;\n    }\n    if (col < K) {\n        ptr_L[get_index(row, col, M, K, C_CONTIGUOUS)] = l_val;\n    }\n    if (row < K) {\n        ptr_U[get_index(row, col, K, N, C_CONTIGUOUS)] = u_val;\n    }\n    ', 'cupyx_scipy_linalg_split_lu', preamble=_device_get_index)

def _cupy_laswp(A, k1, k2, ipiv, incx):
    if False:
        i = 10
        return i + 15
    (m, n) = A.shape
    k = ipiv.shape[0]
    assert 0 <= k1 and k1 <= k2 and (k2 < k)
    assert A._c_contiguous or A._f_contiguous
    _kernel_cupy_laswp(m, n, k1, k2, ipiv, incx, A._c_contiguous, A, size=n)
_kernel_cupy_laswp = cupy.ElementwiseKernel('int32 M, int32 N, int32 K1, int32 K2, raw I IPIV, int32 INCX, bool C_CONTIGUOUS', 'raw T A', '\n    // IPIV: 0-based pivot indices. shape: (K,)  (*) K > K2\n    // A: shape: (M, N)\n    T* ptr_A = &(A[0]);\n    if (K1 > K2) return;\n    int row_start, row_end, row_inc;\n    if (INCX > 0) {\n        row_start = K1; row_end = K2; row_inc = 1;\n    } else if (INCX < 0) {\n        row_start = K2; row_end = K1; row_inc = -1;\n    } else {\n        return;\n    }\n    int col = i;\n    int row1 = row_start;\n    while (1) {\n        int row2 = IPIV[row1];\n        if (row1 != row2) {\n            int idx1 = get_index(row1, col, M, N, C_CONTIGUOUS);\n            int idx2 = get_index(row2, col, M, N, C_CONTIGUOUS);\n            T tmp       = ptr_A[idx1];\n            ptr_A[idx1] = ptr_A[idx2];\n            ptr_A[idx2] = tmp;\n        }\n        if (row1 == row_end) break;\n        row1 += row_inc;\n    }\n    ', 'cupyx_scipy_linalg_laswp', preamble=_device_get_index)

@_uarray.implements('lu_solve')
def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
    if False:
        for i in range(10):
            print('nop')
    'Solve an equation system, ``a * x = b``, given the LU factorization of ``a``\n\n    Args:\n        lu_and_piv (tuple): LU factorization of matrix ``a`` (``(M, M)``)\n            together with pivot indices.\n        b (cupy.ndarray): The matrix with dimension ``(M,)`` or\n            ``(M, N)``.\n        trans ({0, 1, 2}): Type of system to solve:\n\n            ========  =========\n            trans     system\n            ========  =========\n            0         a x  = b\n            1         a^T x = b\n            2         a^H x = b\n            ========  =========\n        overwrite_b (bool): Allow overwriting data in b (may enhance\n            performance)\n        check_finite (bool): Whether to check that the input matrices contain\n            only finite numbers. Disabling may give a performance gain, but may\n            result in problems (crashes, non-termination) if the inputs do\n            contain infinities or NaNs.\n\n    Returns:\n        cupy.ndarray:\n            The matrix with dimension ``(M,)`` or ``(M, N)``.\n\n    .. seealso:: :func:`scipy.linalg.lu_solve`\n    '
    from cupy_backends.cuda.libs import cusolver
    (lu, ipiv) = lu_and_piv
    _util._assert_cupy_array(lu)
    _util._assert_2d(lu)
    _util._assert_stacked_square(lu)
    m = lu.shape[0]
    if m != b.shape[0]:
        raise ValueError('incompatible dimensions.')
    dtype = lu.dtype
    if dtype.char == 'f':
        getrs = cusolver.sgetrs
    elif dtype.char == 'd':
        getrs = cusolver.dgetrs
    elif dtype.char == 'F':
        getrs = cusolver.cgetrs
    elif dtype.char == 'D':
        getrs = cusolver.zgetrs
    else:
        msg = 'Only float32, float64, complex64 and complex128 are supported.'
        raise NotImplementedError(msg)
    if trans == 0:
        trans = cublas.CUBLAS_OP_N
    elif trans == 1:
        trans = cublas.CUBLAS_OP_T
    elif trans == 2:
        trans = cublas.CUBLAS_OP_C
    else:
        raise ValueError('unknown trans')
    lu = lu.astype(dtype, order='F', copy=False)
    ipiv = ipiv.astype(ipiv.dtype, order='F', copy=True)
    ipiv += 1
    b = b.astype(dtype, order='F', copy=not overwrite_b)
    if check_finite:
        if lu.dtype.kind == 'f' and (not cupy.isfinite(lu).all()):
            raise ValueError('array must not contain infs or NaNs.\nNote that when a singular matrix is given, unlike scipy.linalg.lu_factor, cupyx.scipy.linalg.lu_factor returns an array containing NaN.')
        if b.dtype.kind == 'f' and (not cupy.isfinite(b).all()):
            raise ValueError('array must not contain infs or NaNs')
    n = 1 if b.ndim == 1 else b.shape[1]
    cusolver_handle = device.get_cusolver_handle()
    dev_info = cupy.empty(1, dtype=numpy.int32)
    getrs(cusolver_handle, trans, m, n, lu.data.ptr, m, ipiv.data.ptr, b.data.ptr, m, dev_info.data.ptr)
    if not runtime.is_hip and dev_info[0] < 0:
        raise ValueError('illegal value in %d-th argument of internal getrs (lu_solve)' % -dev_info[0])
    return b