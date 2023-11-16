import numpy
import cupy
from cupy import cublas
from cupy._core import _dtype
from cupy.cuda import device
from cupy_backends.cuda.libs import cublas as _cublas
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse.linalg import _interface

def eigsh(a, k=6, *, which='LM', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=True):
    if False:
        i = 10
        return i + 15
    "\n    Find ``k`` eigenvalues and eigenvectors of the real symmetric square\n    matrix or complex Hermitian matrix ``A``.\n\n    Solves ``Ax = wx``, the standard eigenvalue problem for ``w`` eigenvalues\n    with corresponding eigenvectors ``x``.\n\n    Args:\n        a (ndarray, spmatrix or LinearOperator): A symmetric square matrix with\n            dimension ``(n, n)``. ``a`` must :class:`cupy.ndarray`,\n            :class:`cupyx.scipy.sparse.spmatrix` or\n            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.\n        k (int): The number of eigenvalues and eigenvectors to compute. Must be\n            ``1 <= k < n``.\n        which (str): 'LM' or 'LA'. 'LM': finds ``k`` largest (in magnitude)\n            eigenvalues. 'LA': finds ``k`` largest (algebraic) eigenvalues.\n            'SA': finds ``k`` smallest (algebraic) eigenvalues.\n\n        v0 (ndarray): Starting vector for iteration. If ``None``, a random\n            unit vector is used.\n        ncv (int): The number of Lanczos vectors generated. Must be\n            ``k + 1 < ncv < n``. If ``None``, default value is used.\n        maxiter (int): Maximum number of Lanczos update iterations.\n            If ``None``, default value is used.\n        tol (float): Tolerance for residuals ``||Ax - wx||``. If ``0``, machine\n            precision is used.\n        return_eigenvectors (bool): If ``True``, returns eigenvectors in\n            addition to eigenvalues.\n\n    Returns:\n        tuple:\n            If ``return_eigenvectors is True``, it returns ``w`` and ``x``\n            where ``w`` is eigenvalues and ``x`` is eigenvectors. Otherwise,\n            it returns only ``w``.\n\n    .. seealso:: :func:`scipy.sparse.linalg.eigsh`\n\n    .. note::\n        This function uses the thick-restart Lanczos methods\n        (https://sdm.lbl.gov/~kewu/ps/trlan.html).\n\n    "
    n = a.shape[0]
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError('expected square matrix (shape: {})'.format(a.shape))
    if a.dtype.char not in 'fdFD':
        raise TypeError('unsupprted dtype (actual: {})'.format(a.dtype))
    if k <= 0:
        raise ValueError('k must be greater than 0 (actual: {})'.format(k))
    if k >= n:
        raise ValueError('k must be smaller than n (actual: {})'.format(k))
    if which not in ('LM', 'LA', 'SA'):
        raise ValueError("which must be 'LM','LA'or'SA' (actual: {})".format(which))
    if ncv is None:
        ncv = min(max(2 * k, k + 32), n - 1)
    else:
        ncv = min(max(ncv, k + 2), n - 1)
    if maxiter is None:
        maxiter = 10 * n
    if tol == 0:
        tol = numpy.finfo(a.dtype).eps
    alpha = cupy.zeros((ncv,), dtype=a.dtype)
    beta = cupy.zeros((ncv,), dtype=a.dtype.char.lower())
    V = cupy.empty((ncv, n), dtype=a.dtype)
    if v0 is None:
        u = cupy.random.random((n,)).astype(a.dtype)
        V[0] = u / cublas.nrm2(u)
    else:
        u = v0
        V[0] = v0 / cublas.nrm2(v0)
    upadte_impl = 'fast'
    if upadte_impl == 'fast':
        lanczos = _lanczos_fast(a, n, ncv)
    else:
        lanczos = _lanczos_asis
    lanczos(a, V, u, alpha, beta, 0, ncv)
    iter = ncv
    (w, s) = _eigsh_solve_ritz(alpha, beta, None, k, which)
    x = V.T @ s
    beta_k = beta[-1] * s[-1, :]
    res = cublas.nrm2(beta_k)
    uu = cupy.empty((k,), dtype=a.dtype)
    while res > tol and iter < maxiter:
        beta[:k] = 0
        alpha[:k] = w
        V[:k] = x.T
        cublas.gemv(_cublas.CUBLAS_OP_C, 1, V[:k].T, u, 0, uu)
        cublas.gemv(_cublas.CUBLAS_OP_N, -1, V[:k].T, uu, 1, u)
        V[k] = u / cublas.nrm2(u)
        u[...] = a @ V[k]
        cublas.dotc(V[k], u, out=alpha[k])
        u -= alpha[k] * V[k]
        u -= V[:k].T @ beta_k
        cublas.nrm2(u, out=beta[k])
        V[k + 1] = u / beta[k]
        lanczos(a, V, u, alpha, beta, k + 1, ncv)
        iter += ncv - k
        (w, s) = _eigsh_solve_ritz(alpha, beta, beta_k, k, which)
        x = V.T @ s
        beta_k = beta[-1] * s[-1, :]
        res = cublas.nrm2(beta_k)
    if return_eigenvectors:
        idx = cupy.argsort(w)
        return (w[idx], x[:, idx])
    else:
        return cupy.sort(w)

def _lanczos_asis(a, V, u, alpha, beta, i_start, i_end):
    if False:
        i = 10
        return i + 15
    for i in range(i_start, i_end):
        u[...] = a @ V[i]
        cublas.dotc(V[i], u, out=alpha[i])
        u -= u.T @ V[:i + 1].conj().T @ V[:i + 1]
        cublas.nrm2(u, out=beta[i])
        if i >= i_end - 1:
            break
        V[i + 1] = u / beta[i]

def _lanczos_fast(A, n, ncv):
    if False:
        while True:
            i = 10
    from cupy_backends.cuda.libs import cusparse as _cusparse
    from cupyx import cusparse
    cublas_handle = device.get_cublas_handle()
    cublas_pointer_mode = _cublas.getPointerMode(cublas_handle)
    if A.dtype.char == 'f':
        dotc = _cublas.sdot
        nrm2 = _cublas.snrm2
        gemv = _cublas.sgemv
        axpy = _cublas.saxpy
    elif A.dtype.char == 'd':
        dotc = _cublas.ddot
        nrm2 = _cublas.dnrm2
        gemv = _cublas.dgemv
        axpy = _cublas.daxpy
    elif A.dtype.char == 'F':
        dotc = _cublas.cdotc
        nrm2 = _cublas.scnrm2
        gemv = _cublas.cgemv
        axpy = _cublas.caxpy
    elif A.dtype.char == 'D':
        dotc = _cublas.zdotc
        nrm2 = _cublas.dznrm2
        gemv = _cublas.zgemv
        axpy = _cublas.zaxpy
    else:
        raise TypeError('invalid dtype ({})'.format(A.dtype))
    cusparse_handle = None
    if _csr.isspmatrix_csr(A) and cusparse.check_availability('spmv'):
        cusparse_handle = device.get_cusparse_handle()
        spmv_op_a = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        spmv_alpha = numpy.array(1.0, A.dtype)
        spmv_beta = numpy.array(0.0, A.dtype)
        spmv_cuda_dtype = _dtype.to_cuda_dtype(A.dtype)
        spmv_alg = _cusparse.CUSPARSE_MV_ALG_DEFAULT
    v = cupy.empty((n,), dtype=A.dtype)
    uu = cupy.empty((ncv,), dtype=A.dtype)
    vv = cupy.empty((n,), dtype=A.dtype)
    b = cupy.empty((), dtype=A.dtype)
    one = numpy.array(1.0, dtype=A.dtype)
    zero = numpy.array(0.0, dtype=A.dtype)
    mone = numpy.array(-1.0, dtype=A.dtype)
    outer_A = A

    def aux(A, V, u, alpha, beta, i_start, i_end):
        if False:
            return 10
        assert A is outer_A
        if cusparse_handle is not None:
            spmv_desc_A = cusparse.SpMatDescriptor.create(A)
            spmv_desc_v = cusparse.DnVecDescriptor.create(v)
            spmv_desc_u = cusparse.DnVecDescriptor.create(u)
            buff_size = _cusparse.spMV_bufferSize(cusparse_handle, spmv_op_a, spmv_alpha.ctypes.data, spmv_desc_A.desc, spmv_desc_v.desc, spmv_beta.ctypes.data, spmv_desc_u.desc, spmv_cuda_dtype, spmv_alg)
            spmv_buff = cupy.empty(buff_size, cupy.int8)
        v[...] = V[i_start]
        for i in range(i_start, i_end):
            if cusparse_handle is None:
                u[...] = A @ v
            else:
                _cusparse.spMV(cusparse_handle, spmv_op_a, spmv_alpha.ctypes.data, spmv_desc_A.desc, spmv_desc_v.desc, spmv_beta.ctypes.data, spmv_desc_u.desc, spmv_cuda_dtype, spmv_alg, spmv_buff.data.ptr)
            _cublas.setPointerMode(cublas_handle, _cublas.CUBLAS_POINTER_MODE_DEVICE)
            try:
                dotc(cublas_handle, n, v.data.ptr, 1, u.data.ptr, 1, alpha.data.ptr + i * alpha.itemsize)
            finally:
                _cublas.setPointerMode(cublas_handle, cublas_pointer_mode)
            vv.fill(0)
            b[...] = beta[i - 1]
            _cublas.setPointerMode(cublas_handle, _cublas.CUBLAS_POINTER_MODE_DEVICE)
            try:
                axpy(cublas_handle, n, alpha.data.ptr + i * alpha.itemsize, v.data.ptr, 1, vv.data.ptr, 1)
                axpy(cublas_handle, n, b.data.ptr, V[i - 1].data.ptr, 1, vv.data.ptr, 1)
            finally:
                _cublas.setPointerMode(cublas_handle, cublas_pointer_mode)
            axpy(cublas_handle, n, mone.ctypes.data, vv.data.ptr, 1, u.data.ptr, 1)
            gemv(cublas_handle, _cublas.CUBLAS_OP_C, n, i + 1, one.ctypes.data, V.data.ptr, n, u.data.ptr, 1, zero.ctypes.data, uu.data.ptr, 1)
            gemv(cublas_handle, _cublas.CUBLAS_OP_N, n, i + 1, mone.ctypes.data, V.data.ptr, n, uu.data.ptr, 1, one.ctypes.data, u.data.ptr, 1)
            alpha[i] += uu[i]
            _cublas.setPointerMode(cublas_handle, _cublas.CUBLAS_POINTER_MODE_DEVICE)
            try:
                nrm2(cublas_handle, n, u.data.ptr, 1, beta.data.ptr + i * beta.itemsize)
            finally:
                _cublas.setPointerMode(cublas_handle, cublas_pointer_mode)
            if i >= i_end - 1:
                break
            _kernel_normalize(u, beta, i, n, v, V)
    return aux
_kernel_normalize = cupy.ElementwiseKernel('T u, raw S beta, int32 j, int32 n', 'T v, raw T V', 'v = u / beta[j]; V[i + (j+1) * n] = v;', 'cupy_eigsh_normalize')

def _eigsh_solve_ritz(alpha, beta, beta_k, k, which):
    if False:
        i = 10
        return i + 15
    alpha = cupy.asnumpy(alpha)
    beta = cupy.asnumpy(beta)
    t = numpy.diag(alpha)
    t = t + numpy.diag(beta[:-1], k=1)
    t = t + numpy.diag(beta[:-1], k=-1)
    if beta_k is not None:
        beta_k = cupy.asnumpy(beta_k)
        t[k, :k] = beta_k
        t[:k, k] = beta_k
    (w, s) = numpy.linalg.eigh(t)
    if which == 'LA':
        idx = numpy.argsort(w)
        wk = w[idx[-k:]]
        sk = s[:, idx[-k:]]
    elif which == 'LM':
        idx = numpy.argsort(numpy.absolute(w))
        wk = w[idx[-k:]]
        sk = s[:, idx[-k:]]
    elif which == 'SA':
        idx = numpy.argsort(w)
        wk = w[idx[:k]]
        sk = s[:, idx[:k]]
    return (cupy.array(wk), cupy.array(sk))

def svds(a, k=6, *, ncv=None, tol=0, which='LM', maxiter=None, return_singular_vectors=True):
    if False:
        for i in range(10):
            print('nop')
    "Finds the largest ``k`` singular values/vectors for a sparse matrix.\n\n    Args:\n        a (ndarray, spmatrix or LinearOperator): A real or complex array with\n            dimension ``(m, n)``. ``a`` must :class:`cupy.ndarray`,\n            :class:`cupyx.scipy.sparse.spmatrix` or\n            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.\n        k (int): The number of singular values/vectors to compute. Must be\n            ``1 <= k < min(m, n)``.\n        ncv (int): The number of Lanczos vectors generated. Must be\n            ``k + 1 < ncv < min(m, n)``. If ``None``, default value is used.\n        tol (float): Tolerance for singular values. If ``0``, machine precision\n            is used.\n        which (str): Only 'LM' is supported. 'LM': finds ``k`` largest singular\n            values.\n        maxiter (int): Maximum number of Lanczos update iterations.\n            If ``None``, default value is used.\n        return_singular_vectors (bool): If ``True``, returns singular vectors\n            in addition to singular values.\n\n    Returns:\n        tuple:\n            If ``return_singular_vectors`` is ``True``, it returns ``u``, ``s``\n            and ``vt`` where ``u`` is left singular vectors, ``s`` is singular\n            values and ``vt`` is right singular vectors. Otherwise, it returns\n            only ``s``.\n\n    .. seealso:: :func:`scipy.sparse.linalg.svds`\n\n    .. note::\n        This is a naive implementation using cupyx.scipy.sparse.linalg.eigsh as\n        an eigensolver on ``a.H @ a`` or ``a @ a.H``.\n\n    "
    if a.ndim != 2:
        raise ValueError('expected 2D (shape: {})'.format(a.shape))
    if a.dtype.char not in 'fdFD':
        raise TypeError('unsupprted dtype (actual: {})'.format(a.dtype))
    (m, n) = a.shape
    if k <= 0:
        raise ValueError('k must be greater than 0 (actual: {})'.format(k))
    if k >= min(m, n):
        raise ValueError('k must be smaller than min(m, n) (actual: {})'.format(k))
    a = _interface.aslinearoperator(a)
    if m >= n:
        (aH, a) = (a.H, a)
    else:
        (aH, a) = (a, a.H)
    if return_singular_vectors:
        (w, x) = eigsh(aH @ a, k=k, which=which, ncv=ncv, maxiter=maxiter, tol=tol, return_eigenvectors=True)
    else:
        w = eigsh(aH @ a, k=k, which=which, ncv=ncv, maxiter=maxiter, tol=tol, return_eigenvectors=False)
    w = cupy.maximum(w, 0)
    t = w.dtype.char.lower()
    factor = {'f': 1000.0, 'd': 1000000.0}
    cond = factor[t] * numpy.finfo(t).eps
    cutoff = cond * cupy.max(w)
    above_cutoff = w > cutoff
    n_large = above_cutoff.sum().item()
    s = cupy.zeros_like(w)
    s[:n_large] = cupy.sqrt(w[above_cutoff])
    if not return_singular_vectors:
        return s
    x = x[:, above_cutoff]
    if m >= n:
        v = x
        u = a @ v / s[:n_large]
    else:
        u = x
        v = a @ u / s[:n_large]
    u = _augmented_orthnormal_cols(u, k - n_large)
    v = _augmented_orthnormal_cols(v, k - n_large)
    return (u, s, v.conj().T)

def _augmented_orthnormal_cols(x, n_aug):
    if False:
        for i in range(10):
            print('nop')
    if n_aug <= 0:
        return x
    (m, n) = x.shape
    y = cupy.empty((m, n + n_aug), dtype=x.dtype)
    y[:, :n] = x
    for i in range(n, n + n_aug):
        v = cupy.random.random((m,)).astype(x.dtype)
        v -= v @ y[:, :i].conj() @ y[:, :i].T
        y[:, i] = v / cupy.linalg.norm(v)
    return y