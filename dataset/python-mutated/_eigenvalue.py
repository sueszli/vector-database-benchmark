import numpy
import cupy
from cupy.cuda import device
from cupy.cuda import runtime
from cupy.linalg import _util
from cupy._core import _dtype
_cuda_runtime_version = -1

def _syevd(a, UPLO, with_eigen_vector, overwrite_a=False):
    if False:
        i = 10
        return i + 15
    from cupy_backends.cuda.libs import cublas
    from cupy_backends.cuda.libs import cusolver
    if UPLO not in ('L', 'U'):
        raise ValueError("UPLO argument must be 'L' or 'U'")
    (dtype, v_dtype) = _util.linalg_common_type(a, reject_float16=False)
    real_dtype = dtype.char.lower()
    w_dtype = v_dtype.char.lower()
    v = a.astype(dtype, order='F', copy=not overwrite_a)
    (m, lda) = a.shape
    w = cupy.empty(m, real_dtype)
    dev_info = cupy.empty((), numpy.int32)
    handle = device.Device().cusolver_handle
    if with_eigen_vector:
        jobz = cusolver.CUSOLVER_EIG_MODE_VECTOR
    else:
        jobz = cusolver.CUSOLVER_EIG_MODE_NOVECTOR
    if UPLO == 'L':
        uplo = cublas.CUBLAS_FILL_MODE_LOWER
    else:
        uplo = cublas.CUBLAS_FILL_MODE_UPPER
    global _cuda_runtime_version
    if _cuda_runtime_version < 0:
        _cuda_runtime_version = runtime.runtimeGetVersion()
    if not runtime.is_hip and _cuda_runtime_version >= 11010:
        if dtype.char not in 'fdFD':
            raise RuntimeError('Only float32, float64, complex64, and complex128 are supported')
        type_v = _dtype.to_cuda_dtype(dtype)
        type_w = _dtype.to_cuda_dtype(real_dtype)
        params = cusolver.createParams()
        try:
            (work_device_size, work_host_sizse) = cusolver.xsyevd_bufferSize(handle, params, jobz, uplo, m, type_v, v.data.ptr, lda, type_w, w.data.ptr, type_v)
            work_device = cupy.empty(work_device_size, 'b')
            work_host = numpy.empty(work_host_sizse, 'b')
            cusolver.xsyevd(handle, params, jobz, uplo, m, type_v, v.data.ptr, lda, type_w, w.data.ptr, type_v, work_device.data.ptr, work_device_size, work_host.ctypes.data, work_host_sizse, dev_info.data.ptr)
        finally:
            cusolver.destroyParams(params)
        cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(cusolver.xsyevd, dev_info)
    else:
        if dtype == 'f':
            buffer_size = cusolver.ssyevd_bufferSize
            syevd = cusolver.ssyevd
        elif dtype == 'd':
            buffer_size = cusolver.dsyevd_bufferSize
            syevd = cusolver.dsyevd
        elif dtype == 'F':
            buffer_size = cusolver.cheevd_bufferSize
            syevd = cusolver.cheevd
        elif dtype == 'D':
            buffer_size = cusolver.zheevd_bufferSize
            syevd = cusolver.zheevd
        else:
            raise RuntimeError('Only float32, float64, complex64, and complex128 are supported')
        work_size = buffer_size(handle, jobz, uplo, m, v.data.ptr, lda, w.data.ptr)
        work = cupy.empty(work_size, dtype)
        syevd(handle, jobz, uplo, m, v.data.ptr, lda, w.data.ptr, work.data.ptr, work_size, dev_info.data.ptr)
        cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(syevd, dev_info)
    return (w.astype(w_dtype, copy=False), v.astype(v_dtype, copy=False))

def eigh(a, UPLO='L'):
    if False:
        i = 10
        return i + 15
    "\n    Return the eigenvalues and eigenvectors of a complex Hermitian\n    (conjugate symmetric) or a real symmetric matrix.\n\n    Returns two objects, a 1-D array containing the eigenvalues of `a`, and\n    a 2-D square array or matrix (depending on the input type) of the\n    corresponding eigenvectors (in columns).\n\n    Args:\n        a (cupy.ndarray): A symmetric 2-D square matrix ``(M, M)`` or a batch\n            of symmetric 2-D square matrices ``(..., M, M)``.\n        UPLO (str): Select from ``'L'`` or ``'U'``. It specifies which\n            part of ``a`` is used. ``'L'`` uses the lower triangular part of\n            ``a``, and ``'U'`` uses the upper triangular part of ``a``.\n    Returns:\n        tuple of :class:`~cupy.ndarray`:\n            Returns a tuple ``(w, v)``. ``w`` contains eigenvalues and\n            ``v`` contains eigenvectors. ``v[:, i]`` is an eigenvector\n            corresponding to an eigenvalue ``w[i]``. For batch input,\n            ``v[k, :, i]`` is an eigenvector corresponding to an eigenvalue\n            ``w[k, i]`` of ``a[k]``.\n\n    .. warning::\n        This function calls one or more cuSOLVER routine(s) which may yield\n        invalid results if input conditions are not met.\n        To detect these invalid results, you can set the `linalg`\n        configuration to a value that is not `ignore` in\n        :func:`cupyx.errstate` or :func:`cupyx.seterr`.\n\n    .. seealso:: :func:`numpy.linalg.eigh`\n    "
    import cupyx.cusolver
    _util._assert_stacked_2d(a)
    _util._assert_stacked_square(a)
    if a.size == 0:
        (_, v_dtype) = _util.linalg_common_type(a)
        w_dtype = v_dtype.char.lower()
        w = cupy.empty(a.shape[:-1], w_dtype)
        v = cupy.empty(a.shape, v_dtype)
        return (w, v)
    if a.ndim > 2 or runtime.is_hip:
        (w, v) = cupyx.cusolver.syevj(a, UPLO, True)
        return (w, v)
    else:
        return _syevd(a, UPLO, True)

def eigvalsh(a, UPLO='L'):
    if False:
        i = 10
        return i + 15
    "\n    Compute the eigenvalues of a complex Hermitian or real symmetric matrix.\n\n    Main difference from eigh: the eigenvectors are not computed.\n\n    Args:\n        a (cupy.ndarray): A symmetric 2-D square matrix ``(M, M)`` or a batch\n            of symmetric 2-D square matrices ``(..., M, M)``.\n        UPLO (str): Select from ``'L'`` or ``'U'``. It specifies which\n            part of ``a`` is used. ``'L'`` uses the lower triangular part of\n            ``a``, and ``'U'`` uses the upper triangular part of ``a``.\n    Returns:\n        cupy.ndarray:\n            Returns eigenvalues as a vector ``w``. For batch input,\n            ``w[k]`` is a vector of eigenvalues of matrix ``a[k]``.\n\n    .. warning::\n        This function calls one or more cuSOLVER routine(s) which may yield\n        invalid results if input conditions are not met.\n        To detect these invalid results, you can set the `linalg`\n        configuration to a value that is not `ignore` in\n        :func:`cupyx.errstate` or :func:`cupyx.seterr`.\n\n    .. seealso:: :func:`numpy.linalg.eigvalsh`\n    "
    import cupyx.cusolver
    _util._assert_stacked_2d(a)
    _util._assert_stacked_square(a)
    if a.size == 0:
        (_, v_dtype) = _util.linalg_common_type(a)
        w_dtype = v_dtype.char.lower()
        return cupy.empty(a.shape[:-1], w_dtype)
    if a.ndim > 2 or runtime.is_hip:
        return cupyx.cusolver.syevj(a, UPLO, False)
    else:
        return _syevd(a, UPLO, False)[0]