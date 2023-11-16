import collections.abc
import numbers
import numpy
import cupy
from cupy import _core
from cupy._core import internal
from cupy._core._gufuncs import _GUFunc
from cupy.linalg import _solve
from cupy.linalg import _util
matmul = _GUFunc(_core.matmul, '(n?,k),(k,m?)->(n?,m?)', supports_batched=True, supports_out=True, doc='matmul(x1, x2, /, out=None, \\*\\*kwargs)\n\n    Matrix product of two arrays.\n\n    Returns the matrix product of two arrays and is the implementation of\n    the `@` operator introduced in Python 3.5 following PEP465.\n\n    The main difference against cupy.dot are the handling of arrays with more\n    than 2 dimensions. For more information see :func:`numpy.matmul`.\n\n    Args:\n        x1 (cupy.ndarray): The left argument.\n        x2 (cupy.ndarray): The right argument.\n        out (cupy.ndarray, optional): Output array.\n        \\*\\*kwargs: ufunc keyword arguments.\n\n    Returns:\n        cupy.ndarray: Output array.\n\n    .. seealso:: :func:`numpy.matmul`\n    ')

def dot(a, b, out=None):
    if False:
        print('Hello World!')
    'Returns a dot product of two arrays.\n\n    For arrays with more than one axis, it computes the dot product along the\n    last axis of ``a`` and the second-to-last axis of ``b``. This is just a\n    matrix product if the both arrays are 2-D. For 1-D arrays, it uses their\n    unique axis as an axis to take dot product over.\n\n    Args:\n        a (cupy.ndarray): The left argument.\n        b (cupy.ndarray): The right argument.\n        out (cupy.ndarray): Output array.\n\n    Returns:\n        cupy.ndarray: The dot product of ``a`` and ``b``.\n\n    .. seealso:: :func:`numpy.dot`\n\n    '
    return a.dot(b, out)

def vdot(a, b):
    if False:
        return 10
    'Returns the dot product of two vectors.\n\n    The input arrays are flattened into 1-D vectors and then it performs inner\n    product of these vectors.\n\n    Args:\n        a (cupy.ndarray): The first argument.\n        b (cupy.ndarray): The second argument.\n\n    Returns:\n        cupy.ndarray: Zero-dimensional array of the dot product result.\n\n    .. seealso:: :func:`numpy.vdot`\n\n    '
    if a.size != b.size:
        raise ValueError('Axis dimension mismatch')
    if a.dtype.kind == 'c':
        a = a.conj()
    return _core.tensordot_core(a, b, None, 1, 1, a.size, ())

def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    if False:
        print('Hello World!')
    'Returns the cross product of two vectors.\n\n    The cross product of ``a`` and ``b`` in :math:`R^3` is a vector\n    perpendicular to both ``a`` and ``b``.  If ``a`` and ``b`` are arrays\n    of vectors, the vectors are defined by the last axis of ``a`` and ``b``\n    by default, and these axes can have dimensions 2 or 3.  Where the\n    dimension of either ``a`` or ``b`` is 2, the third component of the input\n    vector is assumed to be zero and the cross product calculated accordingly.\n    In cases where both input vectors have dimension 2, the z-component of\n    the cross product is returned.\n\n    Args:\n        a (cupy.ndarray): Components of the first vector(s).\n        b (cupy.ndarray): Components of the second vector(s).\n        axisa (int, optional):\n            Axis of ``a`` that defines the vector(s).\n            By default, the last axis.\n        axisb (int, optional):\n            Axis of ``b`` that defines the vector(s).\n            By default, the last axis.\n        axisc (int, optional):\n            Axis of ``c`` containing the cross product vector(s).  Ignored if\n            both input vectors have dimension 2, as the return is scalar.\n            By default, the last axis.\n        axis (int, optional):\n            If defined, the axis of ``a``, ``b`` and ``c``\n            that defines the vector(s) and cross product(s).\n            Overrides ``axisa``, ``axisb`` and ``axisc``.\n\n    Returns:\n        cupy.ndarray :\n            Vector cross product(s).\n\n    .. seealso:: :func:`numpy.cross`\n\n    '
    if axis is not None:
        (axisa, axisb, axisc) = (axis,) * 3
    a = cupy.asarray(a)
    b = cupy.asarray(b)
    axisa = internal._normalize_axis_index(axisa, a.ndim)
    axisb = internal._normalize_axis_index(axisb, b.ndim)
    a = cupy.moveaxis(a, axisa, -1)
    b = cupy.moveaxis(b, axisb, -1)
    if a.shape[-1] not in (2, 3) or b.shape[-1] not in (2, 3):
        msg = 'incompatible dimensions for cross product\n(dimension must be 2 or 3)'
        raise ValueError(msg)
    shape = cupy.broadcast(a[..., 0], b[..., 0]).shape
    if a.shape[-1] == 3 or b.shape[-1] == 3:
        shape += (3,)
        axisc = internal._normalize_axis_index(axisc, len(shape))
    dtype = cupy.promote_types(a.dtype, b.dtype)
    cp = cupy.empty(shape, dtype)
    a0 = a[..., 0]
    a1 = a[..., 1]
    if a.shape[-1] == 3:
        a2 = a[..., 2]
    b0 = b[..., 0]
    b1 = b[..., 1]
    if b.shape[-1] == 3:
        b2 = b[..., 2]
    if cp.ndim != 0 and cp.shape[-1] == 3:
        cp0 = cp[..., 0]
        cp1 = cp[..., 1]
        cp2 = cp[..., 2]
    if a.shape[-1] == 2:
        if b.shape[-1] == 2:
            cupy.multiply(a0, b1, out=cp)
            cp -= a1 * b0
            return cp
        else:
            assert b.shape[-1] == 3
            cupy.multiply(a1, b2, out=cp0)
            cupy.multiply(a0, b2, out=cp1)
            cupy.negative(cp1, out=cp1)
            cupy.multiply(a0, b1, out=cp2)
            cp2 -= a1 * b0
    else:
        assert a.shape[-1] == 3
        if b.shape[-1] == 3:
            cupy.multiply(a1, b2, out=cp0)
            tmp = a2 * b1
            cp0 -= tmp
            cupy.multiply(a2, b0, out=cp1)
            cupy.multiply(a0, b2, out=tmp)
            cp1 -= tmp
            cupy.multiply(a0, b1, out=cp2)
            cupy.multiply(a1, b0, out=tmp)
            cp2 -= tmp
        else:
            assert b.shape[-1] == 2
            cupy.multiply(a2, b1, out=cp0)
            cupy.negative(cp0, out=cp0)
            cupy.multiply(a2, b0, out=cp1)
            cupy.multiply(a0, b1, out=cp2)
            cp2 -= a1 * b0
    return cupy.moveaxis(cp, -1, axisc)

def inner(a, b):
    if False:
        print('Hello World!')
    'Returns the inner product of two arrays.\n\n    It uses the last axis of each argument to take sum product.\n\n    Args:\n        a (cupy.ndarray): The first argument.\n        b (cupy.ndarray): The second argument.\n\n    Returns:\n        cupy.ndarray: The inner product of ``a`` and ``b``.\n\n    .. seealso:: :func:`numpy.inner`\n\n    '
    a_ndim = a.ndim
    b_ndim = b.ndim
    if a_ndim == 0 or b_ndim == 0:
        return cupy.multiply(a, b)
    a_axis = a_ndim - 1
    b_axis = b_ndim - 1
    if a.shape[-1] != b.shape[-1]:
        raise ValueError('Axis dimension mismatch')
    if a_axis:
        a = cupy.rollaxis(a, a_axis, 0)
    if b_axis:
        b = cupy.rollaxis(b, b_axis, 0)
    ret_shape = a.shape[1:] + b.shape[1:]
    k = a.shape[0]
    n = a.size // k
    m = b.size // k
    return _core.tensordot_core(a, b, None, n, m, k, ret_shape)

def outer(a, b, out=None):
    if False:
        print('Hello World!')
    'Returns the outer product of two vectors.\n\n    The input arrays are flattened into 1-D vectors and then it performs outer\n    product of these vectors.\n\n    Args:\n        a (cupy.ndarray): The first argument.\n        b (cupy.ndarray): The second argument.\n        out (cupy.ndarray): Output array.\n\n    Returns:\n        cupy.ndarray: 2-D array of the outer product of ``a`` and ``b``.\n\n    .. seealso:: :func:`numpy.outer`\n\n    '
    return cupy.multiply(a.ravel()[:, None], b.ravel()[None, :], out=out)

def tensordot(a, b, axes=2):
    if False:
        print('Hello World!')
    'Returns the tensor dot product of two arrays along specified axes.\n\n    This is equivalent to compute dot product along the specified axes which\n    are treated as one axis by reshaping.\n\n    Args:\n        a (cupy.ndarray): The first argument.\n        b (cupy.ndarray): The second argument.\n        axes:\n            - If it is an integer, then ``axes`` axes at the last of ``a`` and\n              the first of ``b`` are used.\n            - If it is a pair of sequences of integers, then these two\n              sequences specify the list of axes for ``a`` and ``b``. The\n              corresponding axes are paired for sum-product.\n\n    Returns:\n        cupy.ndarray: The tensor dot product of ``a`` and ``b`` along the\n        axes specified by ``axes``.\n\n    .. seealso:: :func:`numpy.tensordot`\n\n    '
    a_ndim = a.ndim
    b_ndim = b.ndim
    if a_ndim == 0 or b_ndim == 0:
        if axes != 0 and axes != ((), ()):
            raise ValueError('An input is zero-dim while axes has dimensions')
        return cupy.multiply(a, b)
    if isinstance(axes, collections.abc.Sequence):
        if len(axes) != 2:
            raise ValueError('Axes must consist of two arrays.')
        (a_axes, b_axes) = axes
        if numpy.isscalar(a_axes):
            a_axes = (a_axes,)
        if numpy.isscalar(b_axes):
            b_axes = (b_axes,)
    else:
        a_axes = tuple(range(a_ndim - axes, a_ndim))
        b_axes = tuple(range(axes))
    sum_ndim = len(a_axes)
    if sum_ndim != len(b_axes):
        raise ValueError('Axes length mismatch')
    for (a_axis, b_axis) in zip(a_axes, b_axes):
        if a.shape[a_axis] != b.shape[b_axis]:
            raise ValueError('Axis dimension mismatch')
    a = _move_axes_to_head(a, [axis % a_ndim for axis in a_axes])
    b = _move_axes_to_head(b, [axis % b_ndim for axis in b_axes])
    ret_shape = a.shape[sum_ndim:] + b.shape[sum_ndim:]
    k = internal.prod(a.shape[:sum_ndim])
    n = a.size // k if k != 0 else 0
    m = b.size // k if k != 0 else 0
    return _core.tensordot_core(a, b, None, n, m, k, ret_shape)

def matrix_power(M, n):
    if False:
        while True:
            i = 10
    'Raise a square matrix to the (integer) power `n`.\n\n    Args:\n        M (~cupy.ndarray): Matrix to raise by power n.\n        n (~int): Power to raise matrix to.\n\n    Returns:\n        ~cupy.ndarray: Output array.\n\n    ..seealso:: :func:`numpy.linalg.matrix_power`\n    '
    _util._assert_cupy_array(M)
    _util._assert_stacked_2d(M)
    _util._assert_stacked_square(M)
    if not isinstance(n, int):
        raise TypeError('exponent must be an integer')
    if n == 0:
        return _util.stacked_identity_like(M)
    elif n < 0:
        M = _solve.inv(M)
        n *= -1
    if n <= 3:
        if n == 1:
            return M
        elif n == 2:
            return cupy.matmul(M, M)
        else:
            return cupy.matmul(cupy.matmul(M, M), M)
    (result, Z) = (None, None)
    for b in cupy.binary_repr(n)[::-1]:
        Z = M if Z is None else cupy.matmul(Z, Z)
        if b == '1':
            result = Z if result is None else cupy.matmul(result, Z)
    return result

def kron(a, b):
    if False:
        i = 10
        return i + 15
    'Returns the kronecker product of two arrays.\n\n    Args:\n        a (~cupy.ndarray): The first argument.\n        b (~cupy.ndarray): The second argument.\n\n    Returns:\n        ~cupy.ndarray: Output array.\n\n    .. seealso:: :func:`numpy.kron`\n\n    '
    a_isnumber = isinstance(a, numbers.Number)
    b_isnumber = isinstance(b, numbers.Number)
    if a_isnumber and b_isnumber:
        return a * b
    if a_isnumber or b_isnumber:
        return cupy.multiply(a, b)
    a_ndim = a.ndim
    b_ndim = b.ndim
    if a_ndim == 0 or b_ndim == 0:
        return cupy.multiply(a, b)
    ndim = b_ndim
    a_shape = a.shape
    b_shape = b.shape
    if a_ndim != b_ndim:
        if b_ndim > a_ndim:
            a_shape = (1,) * (b_ndim - a_ndim) + a_shape
        else:
            b_shape = (1,) * (a_ndim - b_ndim) + b_shape
            ndim = a_ndim
    axis = ndim - 1
    out = _core.tensordot_core(a, b, None, a.size, b.size, 1, a_shape + b_shape)
    for _ in range(ndim):
        out = _core.concatenate_method(out, axis=axis)
    return out

def _move_axes_to_head(a, axes):
    if False:
        return 10
    for (idx, axis) in enumerate(axes):
        if idx != axis:
            break
    else:
        return a
    return a.transpose(axes + [i for i in range(a.ndim) if i not in axes])