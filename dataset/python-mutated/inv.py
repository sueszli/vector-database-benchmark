import numpy.linalg
import chainer
from chainer.backends import cuda
from chainer import function_node
import chainer.functions
from chainer.functions.math import matmul
from chainer import utils
from chainer.utils import precision
from chainer.utils import type_check

def _inv_gpu(b):
    if False:
        i = 10
        return i + 15
    a = matmul._as_batch_mat(b).copy()
    n = a.shape[1]
    n_matrices = len(a)
    p = cuda.cupy.empty((n, n_matrices), dtype=numpy.int32)
    c = cuda.cupy.empty_like(a)
    info = cuda.cupy.empty(n_matrices, dtype=numpy.int32)
    ap = matmul._mat_ptrs(a)
    cp = matmul._mat_ptrs(c)
    (_, lda) = matmul._get_ld(a)
    (_, ldc) = matmul._get_ld(c)
    handle = cuda.Device().cublas_handle
    if b.dtype == numpy.float32:
        cuda.cublas.sgetrfBatched(handle, n, ap.data.ptr, lda, p.data.ptr, info.data.ptr, n_matrices)
        cuda.cublas.sgetriBatched(handle, n, ap.data.ptr, lda, p.data.ptr, cp.data.ptr, ldc, info.data.ptr, n_matrices)
    elif b.dtype == numpy.float64:
        cuda.cublas.dgetrfBatched(handle, n, ap.data.ptr, lda, p.data.ptr, info.data.ptr, n_matrices)
        cuda.cublas.dgetriBatched(handle, n, ap.data.ptr, lda, p.data.ptr, cp.data.ptr, ldc, info.data.ptr, n_matrices)
    else:
        assert False
    return (c, info)

class Inv(function_node.FunctionNode):

    def check_type_forward(self, in_types):
        if False:
            for i in range(10):
                print('nop')
        type_check._argname(in_types, ('a',))
        (a_type,) = in_types
        type_check.expect(a_type.dtype.kind == 'f')
        type_check.expect(a_type.ndim == 2)
        type_check.expect(a_type.shape[0] == a_type.shape[1])

    @precision._fp16_mixed_precision_helper
    def forward_cpu(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.retain_outputs((0,))
        try:
            invx = utils.force_array(numpy.linalg.inv(x[0]))
        except numpy.linalg.LinAlgError:
            raise ValueError('Input has singular matrices.')
        return (invx,)

    @precision._fp16_mixed_precision_helper
    def forward_gpu(self, x):
        if False:
            return 10
        self.retain_outputs((0,))
        shape = x[0].shape
        (invx, info) = _inv_gpu(x[0].reshape(1, *shape))
        if chainer.is_debug():
            if cuda.cupy.any(info != 0):
                raise ValueError('Input has singular matrices.')
        invx = invx.reshape(shape)
        return (invx,)

    def backward(self, x, gy):
        if False:
            for i in range(10):
                print('nop')
        (invx,) = self.get_retained_outputs()
        invxT = chainer.functions.transpose(invx)
        gx = chainer.functions.matmul(chainer.functions.matmul(-invxT, gy[0]), invxT)
        return (gx,)

class BatchInv(function_node.FunctionNode):

    def check_type_forward(self, in_types):
        if False:
            return 10
        type_check._argname(in_types, ('a',))
        (a_type,) = in_types
        type_check.expect(a_type.dtype.kind == 'f')
        type_check.expect(a_type.ndim == 3)
        type_check.expect(a_type.shape[-1] == a_type.shape[-2])

    @precision._fp16_mixed_precision_helper
    def forward_cpu(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.retain_outputs((0,))
        try:
            invx = utils.force_array(numpy.linalg.inv(x[0]))
        except numpy.linalg.LinAlgError:
            raise ValueError('Input has singular matrices.')
        return (invx,)

    @precision._fp16_mixed_precision_helper
    def forward_gpu(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.retain_outputs((0,))
        (invx, info) = _inv_gpu(x[0])
        if chainer.is_debug():
            if cuda.cupy.any(info != 0):
                raise ValueError('Input has singular matrices.')
        return (invx,)

    def backward(self, x, gy):
        if False:
            for i in range(10):
                print('nop')
        (invx,) = self.get_retained_outputs()
        (gy,) = gy
        ret = chainer.functions.matmul(-invx, gy, transa=True)
        ret2 = chainer.functions.matmul(ret, invx, transb=True)
        return (ret2,)

def inv(a):
    if False:
        i = 10
        return i + 15
    'Computes the inverse of square matrix.\n\n        a (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Input array to compute the inverse for. Shape of\n            the array should be ``(n, n)`` where ``n`` is the dimensionality of\n            a square matrix.\n\n    Returns:\n        ~chainer.Variable: Matrix inverse of ``a``.\n    '
    return Inv().apply((a,))[0]

def batch_inv(a):
    if False:
        return 10
    'Computes the inverse of a batch of square matrices.\n\n    Args:\n        a (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Input array to compute the inverse for. Shape of\n            the array should be ``(m, n, n)`` where ``m`` is the number of\n            matrices in the batch, and ``n`` is the dimensionality of a square\n            matrix.\n\n    Returns:\n        ~chainer.Variable: Inverse of every matrix in the batch of matrices.\n    '
    return BatchInv().apply((a,))[0]