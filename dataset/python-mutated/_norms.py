import numpy
import cupy
from cupy import _core
from cupy.linalg import _decomposition
from cupy.linalg import _util
import functools

def _multi_svd_norm(x, row_axis, col_axis, op):
    if False:
        for i in range(10):
            print('nop')
    y = cupy.moveaxis(x, (row_axis, col_axis), (-2, -1))
    result = op(_decomposition.svd(y, compute_uv=False), axis=-1)
    return result
_norm_ord2 = _core.create_reduction_func('_norm_ord2', ('?->l', 'b->l', 'B->L', 'h->l', 'H->L', 'i->l', 'I->L', 'l->l', 'L->L', 'q->q', 'Q->Q', ('e->e', (None, None, None, 'float')), 'f->f', 'd->d'), ('in0 * in0', 'a + b', 'out0 = sqrt(type_out0_raw(a))', None), 0)
_norm_ord2_complex = _core.create_reduction_func('_norm_ord2_complex', ('F->f', 'D->d'), ('in0.real() * in0.real() + in0.imag() * in0.imag()', 'a + b', 'out0 = sqrt(type_out0_raw(a))', None), 0)

def norm(x, ord=None, axis=None, keepdims=False):
    if False:
        return 10
    "Returns one of matrix norms specified by ``ord`` parameter.\n\n    See numpy.linalg.norm for more detail.\n\n    Args:\n        x (cupy.ndarray): Array to take norm. If ``axis`` is None,\n            ``x`` must be 1-D or 2-D.\n        ord (non-zero int, inf, -inf, 'fro'): Norm type.\n        axis (int, 2-tuple of ints, None): 1-D or 2-D norm is cumputed over\n            ``axis``.\n        keepdims (bool): If this is set ``True``, the axes which are normed\n            over are left.\n\n    Returns:\n        cupy.ndarray\n\n    "
    if not issubclass(x.dtype.type, numpy.inexact):
        x = x.astype(float)
    if axis is None:
        ndim = x.ndim
        if ord is None or (ndim == 1 and ord == 2) or (ndim == 2 and ord in ('f', 'fro')):
            if x.dtype.kind == 'c':
                s = abs(x.ravel())
                s *= s
                ret = cupy.sqrt(s.sum())
            else:
                ret = cupy.sqrt((x * x).sum())
            if keepdims:
                ret = ret.reshape((1,) * ndim)
            return ret
    nd = x.ndim
    if axis is None:
        axis = tuple(range(nd))
    elif not isinstance(axis, tuple):
        try:
            axis = int(axis)
        except Exception:
            raise TypeError("'axis' must be None, an integer or a tuple of integers")
        axis = (axis,)
    if len(axis) == 1:
        if ord == numpy.inf:
            return abs(x).max(axis=axis, keepdims=keepdims)
        elif ord == -numpy.inf:
            return abs(x).min(axis=axis, keepdims=keepdims)
        elif ord == 0:
            return (x != 0).astype(x.real.dtype).sum(axis=axis, keepdims=keepdims)
        elif ord == 1:
            return abs(x).sum(axis=axis, keepdims=keepdims)
        elif ord is None or ord == 2:
            if x.dtype.kind == 'c':
                return _norm_ord2_complex(x, axis=axis, keepdims=keepdims)
            return _norm_ord2(x, axis=axis, keepdims=keepdims)
        else:
            try:
                float(ord)
            except TypeError:
                raise ValueError('Invalid norm order for vectors.')
            absx = abs(x)
            absx **= ord
            ret = absx.sum(axis=axis, keepdims=keepdims)
            ret **= cupy.reciprocal(ord, dtype=ret.dtype)
            return ret
    elif len(axis) == 2:
        (row_axis, col_axis) = axis
        if row_axis < 0:
            row_axis += nd
        if col_axis < 0:
            col_axis += nd
        if not (0 <= row_axis < nd and 0 <= col_axis < nd):
            raise ValueError('Invalid axis %r for an array with shape %r' % (axis, x.shape))
        if row_axis == col_axis:
            raise ValueError('Duplicate axes given.')
        if ord == 2:
            op_max = functools.partial(cupy.take, indices=0)
            ret = _multi_svd_norm(x, row_axis, col_axis, op_max)
        elif ord == -2:
            op_min = functools.partial(cupy.take, indices=-1)
            ret = _multi_svd_norm(x, row_axis, col_axis, op_min)
        elif ord == 1:
            if col_axis > row_axis:
                col_axis -= 1
            ret = abs(x).sum(axis=row_axis).max(axis=col_axis)
        elif ord == numpy.inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = abs(x).sum(axis=col_axis).max(axis=row_axis)
        elif ord == -1:
            if col_axis > row_axis:
                col_axis -= 1
            ret = abs(x).sum(axis=row_axis).min(axis=col_axis)
        elif ord == -numpy.inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = abs(x).sum(axis=col_axis).min(axis=row_axis)
        elif ord in [None, 'fro', 'f']:
            if x.dtype.kind == 'c':
                ret = _norm_ord2_complex(x, axis=axis)
            else:
                ret = _norm_ord2(x, axis=axis)
        elif ord == 'nuc':
            ret = _multi_svd_norm(x, row_axis, col_axis, cupy.sum)
        else:
            raise ValueError('Invalid norm order for matrices.')
        if keepdims:
            ret_shape = list(x.shape)
            ret_shape[axis[0]] = 1
            ret_shape[axis[1]] = 1
            ret = ret.reshape(ret_shape)
        return ret
    else:
        raise ValueError('Improper number of dimensions to norm.')

def det(a):
    if False:
        return 10
    'Returns the determinant of an array.\n\n    Args:\n        a (cupy.ndarray): The input matrix with dimension ``(..., N, N)``.\n\n    Returns:\n        cupy.ndarray: Determinant of ``a``. Its shape is ``a.shape[:-2]``.\n\n    .. seealso:: :func:`numpy.linalg.det`\n    '
    (sign, logdet) = slogdet(a)
    return sign * cupy.exp(logdet)

def matrix_rank(M, tol=None):
    if False:
        i = 10
        return i + 15
    'Return matrix rank of array using SVD method\n\n    Args:\n        M (cupy.ndarray): Input array. Its `ndim` must be less than or equal to\n            2.\n        tol (None or float): Threshold of singular value of `M`.\n            When `tol` is `None`, and `eps` is the epsilon value for datatype\n            of `M`, then `tol` is set to `S.max() * max(M.shape) * eps`,\n            where `S` is the singular value of `M`.\n            It obeys :func:`numpy.linalg.matrix_rank`.\n\n    Returns:\n        cupy.ndarray: Rank of `M`.\n\n    .. seealso:: :func:`numpy.linalg.matrix_rank`\n    '
    if M.ndim < 2:
        return (M != 0).any().astype(int)
    S = _decomposition.svd(M, compute_uv=False)
    if tol is None:
        tol = S.max(axis=-1, keepdims=True) * max(M.shape[-2:]) * numpy.finfo(S.dtype).eps
    return (S > tol).sum(axis=-1, dtype=numpy.intp)

def slogdet(a):
    if False:
        i = 10
        return i + 15
    "Returns sign and logarithm of the determinant of an array.\n\n    It calculates the natural logarithm of the determinant of a given value.\n\n    Args:\n        a (cupy.ndarray): The input matrix with dimension ``(..., N, N)``.\n\n    Returns:\n        tuple of :class:`~cupy.ndarray`:\n            It returns a tuple ``(sign, logdet)``. ``sign`` represents each\n            sign of the determinant as a real number ``0``, ``1`` or ``-1``.\n            'logdet' represents the natural logarithm of the absolute of the\n            determinant.\n            If the determinant is zero, ``sign`` will be ``0`` and ``logdet``\n            will be ``-inf``.\n            The shapes of both ``sign`` and ``logdet`` are equal to\n            ``a.shape[:-2]``.\n\n    .. warning::\n        This function calls one or more cuSOLVER routine(s) which may yield\n        invalid results if input conditions are not met.\n        To detect these invalid results, you can set the `linalg`\n        configuration to a value that is not `ignore` in\n        :func:`cupyx.errstate` or :func:`cupyx.seterr`.\n\n    .. warning::\n        To produce the same results as :func:`numpy.linalg.slogdet` for\n        singular inputs, set the `linalg` configuration to `raise`.\n\n    .. seealso:: :func:`numpy.linalg.slogdet`\n    "
    _util._assert_stacked_2d(a)
    _util._assert_stacked_square(a)
    (dtype, sign_dtype) = _util.linalg_common_type(a)
    logdet_dtype = numpy.dtype(sign_dtype.char.lower())
    a_shape = a.shape
    shape = a_shape[:-2]
    n = a_shape[-2]
    if a.size == 0:
        sign = cupy.ones(shape, sign_dtype)
        logdet = cupy.zeros(shape, logdet_dtype)
        return (sign, logdet)
    (lu, ipiv, dev_info) = _decomposition._lu_factor(a, dtype)
    diag = cupy.diagonal(lu, axis1=-2, axis2=-1)
    logdet = cupy.log(cupy.abs(diag)).sum(axis=-1)
    non_zero = cupy.count_nonzero(ipiv != cupy.arange(1, n + 1), axis=-1)
    if dtype.kind == 'f':
        non_zero += cupy.count_nonzero(diag < 0, axis=-1)
    sign = non_zero % 2 * -2 + 1
    if dtype.kind == 'c':
        sign = sign * cupy.prod(diag / cupy.abs(diag), axis=-1)
    sign = sign.astype(dtype)
    logdet = logdet.astype(logdet_dtype, copy=False)
    singular = dev_info > 0
    return (cupy.where(singular, sign_dtype.type(0), sign).reshape(shape), cupy.where(singular, logdet_dtype.type('-inf'), logdet).reshape(shape))

def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    if False:
        print('Hello World!')
    'Returns the sum along the diagonals of an array.\n\n    It computes the sum along the diagonals at ``axis1`` and ``axis2``.\n\n    Args:\n        a (cupy.ndarray): Array to take trace.\n        offset (int): Index of diagonals. Zero indicates the main diagonal, a\n            positive value an upper diagonal, and a negative value a lower\n            diagonal.\n        axis1 (int): The first axis along which the trace is taken.\n        axis2 (int): The second axis along which the trace is taken.\n        dtype: Data type specifier of the output.\n        out (cupy.ndarray): Output array.\n\n    Returns:\n        cupy.ndarray: The trace of ``a`` along axes ``(axis1, axis2)``.\n\n    .. seealso:: :func:`numpy.trace`\n\n    '
    return a.trace(offset, axis1, axis2, dtype, out)