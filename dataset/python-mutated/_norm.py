import numpy
import cupy
import cupyx.scipy.sparse

def _sparse_frobenius_norm(x):
    if False:
        return 10
    if cupy.issubdtype(x.dtype, cupy.complexfloating):
        sqnorm = abs(x).power(2).sum()
    else:
        sqnorm = x.power(2).sum()
    return cupy.sqrt(sqnorm)

def norm(x, ord=None, axis=None):
    if False:
        for i in range(10):
            print('nop')
    "Norm of a cupy.scipy.spmatrix\n\n    This function is able to return one of seven different sparse matrix norms,\n    depending on the value of the ``ord`` parameter.\n\n    Args:\n        x (sparse matrix) : Input sparse matrix.\n        ord (non-zero int, inf, -inf, 'fro', optional) : Order of the norm (see\n            table under ``Notes``). inf means numpy's `inf` object.\n        axis : (int, 2-tuple of ints, None, optional): If `axis` is an\n            integer, it specifies the axis of `x` along which to\n            compute the vector norms.  If `axis` is a 2-tuple, it specifies the\n            axes that hold 2-D matrices, and the matrix norms of these matrices\n            are computed.  If `axis` is None then either a vector norm\n            (when `x` is 1-D) or a matrix norm (when `x` is 2-D) is returned.\n    Returns:\n        ndarray : 0-D or 1-D array or norm(s).\n\n    .. seealso:: :func:`scipy.sparse.linalg.norm`\n    "
    if not cupyx.scipy.sparse.issparse(x):
        raise TypeError('input is not sparse. use cupy.linalg.norm')
    if axis is None and ord in (None, 'fro', 'f'):
        return _sparse_frobenius_norm(x)
    x = x.tocsr()
    if axis is None:
        axis = (0, 1)
    elif not isinstance(axis, tuple):
        msg = "'axis' must be None, an integer or a tuple of integers"
        try:
            int_axis = int(axis)
        except TypeError:
            raise TypeError(msg)
        if axis != int_axis:
            raise TypeError(msg)
        axis = (int_axis,)
    nd = 2
    if len(axis) == 2:
        (row_axis, col_axis) = axis
        if not (-nd <= row_axis < nd and -nd <= col_axis < nd):
            raise ValueError('Invalid axis %r for an array with shape %r' % (axis, x.shape))
        if row_axis % nd == col_axis % nd:
            raise ValueError('Duplicate axes given.')
        if ord == 2:
            raise NotImplementedError
        elif ord == -2:
            raise NotImplementedError
        elif ord == 1:
            return abs(x).sum(axis=row_axis).max()
        elif ord == numpy.inf:
            return abs(x).sum(axis=col_axis).max()
        elif ord == -1:
            return abs(x).sum(axis=row_axis).min()
        elif ord == -numpy.inf:
            return abs(x).sum(axis=col_axis).min()
        elif ord in (None, 'f', 'fro'):
            return _sparse_frobenius_norm(x)
        else:
            raise ValueError('Invalid norm order for matrices.')
    elif len(axis) == 1:
        (a,) = axis
        if not -nd <= a < nd:
            raise ValueError('Invalid axis %r for an array with shape %r' % (axis, x.shape))
        if ord == numpy.inf:
            return abs(x).max(axis=a).A.ravel()
        elif ord == -numpy.inf:
            return abs(x).min(axis=a).A.ravel()
        elif ord == 0:
            return (x != 0).astype(numpy.float32).sum(axis=a).ravel().astype(numpy.int_)
        elif ord == 1:
            return abs(x).sum(axis=a).ravel()
        elif ord in (2, None):
            return cupy.sqrt(abs(x).power(2).sum(axis=a)).ravel()
        else:
            try:
                ord + 1
            except TypeError:
                raise ValueError('Invalid norm order for vectors.')
            return cupy.power(abs(x).power(ord).sum(axis=a), 1 / ord).ravel()
    else:
        raise ValueError('Improper number of dimensions to norm.')