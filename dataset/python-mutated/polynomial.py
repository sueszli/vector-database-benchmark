import cupy

def polyvander(x, deg):
    if False:
        for i in range(10):
            print('nop')
    'Computes the Vandermonde matrix of given degree.\n\n    Args:\n        x (cupy.ndarray): array of points\n        deg (int): degree of the resulting matrix.\n\n    Returns:\n        cupy.ndarray: The Vandermonde matrix\n\n    .. seealso:: :func:`numpy.polynomial.polynomial.polyvander`\n\n    '
    deg = cupy.polynomial.polyutils._deprecate_as_int(deg, 'deg')
    if deg < 0:
        raise ValueError('degree must be non-negative')
    if x.ndim == 0:
        x = x.ravel()
    dtype = cupy.float64 if x.dtype.kind in 'biu' else x.dtype
    out = x ** cupy.arange(deg + 1, dtype=dtype).reshape((-1,) + (1,) * x.ndim)
    return cupy.moveaxis(out, 0, -1)

def polycompanion(c):
    if False:
        print('Hello World!')
    'Computes the companion matrix of c.\n\n    Args:\n        c (cupy.ndarray): 1-D array of polynomial coefficients\n            ordered from low to high degree.\n\n    Returns:\n        cupy.ndarray: Companion matrix of dimensions (deg, deg).\n\n    .. seealso:: :func:`numpy.polynomial.polynomial.polycompanion`\n\n    '
    [c] = cupy.polynomial.polyutils.as_series([c])
    deg = c.size - 1
    if deg == 0:
        raise ValueError('Series must have maximum degree of at least 1.')
    matrix = cupy.eye(deg, k=-1, dtype=c.dtype)
    matrix[:, -1] -= c[:-1] / c[-1]
    return matrix

def polyval(x, c, tensor=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Evaluate a polynomial at points x.\n\n    If `c` is of length `n + 1`, this function returns the value\n\n    .. math:: p(x) = c_0 + c_1 * x + ... + c_n * x^n\n\n    The parameter `x` is converted to an array only if it is a tuple or a\n    list, otherwise it is treated as a scalar. In either case, either `x`\n    or its elements must support multiplication and addition both with\n    themselves and with the elements of `c`.\n\n    If `c` is a 1-D array, then `p(x)` will have the same shape as `x`.  If\n    `c` is multidimensional, then the shape of the result depends on the\n    value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +\n    x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that\n    scalars have shape (,).\n\n    Trailing zeros in the coefficients will be used in the evaluation, so\n    they should be avoided if efficiency is a concern.\n\n    Parameters\n    ----------\n    x : array_like, compatible object\n        If `x` is a list or tuple, it is converted to an ndarray, otherwise\n        it is left unchanged and treated as a scalar. In either case, `x`\n        or its elements must support addition and multiplication with\n        with themselves and with the elements of `c`.\n    c : array_like\n        Array of coefficients ordered so that the coefficients for terms of\n        degree n are contained in c[n]. If `c` is multidimensional the\n        remaining indices enumerate multiple polynomials. In the two\n        dimensional case the coefficients may be thought of as stored in\n        the columns of `c`.\n    tensor : boolean, optional\n        If True, the shape of the coefficient array is extended with ones\n        on the right, one for each dimension of `x`. Scalars have dimension 0\n        for this action. The result is that every column of coefficients in\n        `c` is evaluated for every element of `x`. If False, `x` is broadcast\n        over the columns of `c` for the evaluation.  This keyword is useful\n        when `c` is multidimensional. The default value is True.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The shape of the returned array is described above.\n\n    See Also\n    --------\n    numpy.polynomial.polynomial.polyval\n\n    Notes\n    -----\n    The evaluation uses Horner's method.\n\n    "
    c = cupy.array(c, ndmin=1, copy=False)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c + 0.0
    if isinstance(x, (tuple, list)):
        x = cupy.asarray(x)
    if isinstance(x, cupy.ndarray) and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)
    c0 = c[-1] + x * 0
    for i in range(2, len(c) + 1):
        c0 = c[-i] + c0 * x
    return c0

def polyvalfromroots(x, r, tensor=True):
    if False:
        i = 10
        return i + 15
    '\n    Evaluate a polynomial specified by its roots at points x.\n\n    If `r` is of length `N`, this function returns the value\n\n    .. math:: p(x) = \\prod_{n=1}^{N} (x - r_n)\n\n    The parameter `x` is converted to an array only if it is a tuple or a\n    list, otherwise it is treated as a scalar. In either case, either `x`\n    or its elements must support multiplication and addition both with\n    themselves and with the elements of `r`.\n\n    If `r` is a 1-D array, then `p(x)` will have the same shape as `x`.  If `r`\n    is multidimensional, then the shape of the result depends on the value of\n    `tensor`. If `tensor` is ``True`` the shape will be r.shape[1:] + x.shape;\n    that is, each polynomial is evaluated at every value of `x`. If `tensor` is\n    ``False``, the shape will be r.shape[1:]; that is, each polynomial is\n    evaluated only for the corresponding broadcast value of `x`. Note that\n    scalars have shape (,).\n\n    Parameters\n    ----------\n    x : array_like, compatible object\n        If `x` is a list or tuple, it is converted to an ndarray, otherwise\n        it is left unchanged and treated as a scalar. In either case, `x`\n        or its elements must support addition and multiplication with\n        with themselves and with the elements of `r`.\n    r : array_like\n        Array of roots. If `r` is multidimensional the first index is the\n        root index, while the remaining indices enumerate multiple\n        polynomials. For instance, in the two dimensional case the roots\n        of each polynomial may be thought of as stored in the columns of `r`.\n    tensor : boolean, optional\n        If True, the shape of the roots array is extended with ones on the\n        right, one for each dimension of `x`. Scalars have dimension 0 for this\n        action. The result is that every column of coefficients in `r` is\n        evaluated for every element of `x`. If False, `x` is broadcast over the\n        columns of `r` for the evaluation.  This keyword is useful when `r` is\n        multidimensional. The default value is True.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The shape of the returned array is described above.\n\n    See Also\n    --------\n    numpy.polynomial.polynomial.polyvalfroomroots\n    '
    r = cupy.array(r, ndmin=1, copy=False)
    if r.dtype.char in '?bBhHiIlLqQpP':
        r = r.astype(cupy.double)
    if isinstance(x, (tuple, list)):
        x = cupy.asarray(x)
    if isinstance(x, cupy.ndarray):
        if tensor:
            r = r.reshape(r.shape + (1,) * x.ndim)
        elif x.ndim >= r.ndim:
            raise ValueError('x.ndim must be < r.ndim when tensor == False')
    return cupy.prod(x - r, axis=0)