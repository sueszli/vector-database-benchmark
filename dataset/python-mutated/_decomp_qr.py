"""QR decomposition functions."""
import numpy
from .lapack import get_lapack_funcs
from ._misc import _datacopied
__all__ = ['qr', 'qr_multiply', 'rq']

def safecall(f, name, *args, **kwargs):
    if False:
        return 10
    'Call a LAPACK routine, determining lwork automatically and handling\n    error return values'
    lwork = kwargs.get('lwork', None)
    if lwork in (None, -1):
        kwargs['lwork'] = -1
        ret = f(*args, **kwargs)
        kwargs['lwork'] = ret[-2][0].real.astype(numpy.int_)
    ret = f(*args, **kwargs)
    if ret[-1] < 0:
        raise ValueError('illegal value in %dth argument of internal %s' % (-ret[-1], name))
    return ret[:-2]

def qr(a, overwrite_a=False, lwork=None, mode='full', pivoting=False, check_finite=True):
    if False:
        while True:
            i = 10
    "\n    Compute QR decomposition of a matrix.\n\n    Calculate the decomposition ``A = Q R`` where Q is unitary/orthogonal\n    and R upper triangular.\n\n    Parameters\n    ----------\n    a : (M, N) array_like\n        Matrix to be decomposed\n    overwrite_a : bool, optional\n        Whether data in `a` is overwritten (may improve performance if\n        `overwrite_a` is set to True by reusing the existing input data\n        structure rather than creating a new one.)\n    lwork : int, optional\n        Work array size, lwork >= a.shape[1]. If None or -1, an optimal size\n        is computed.\n    mode : {'full', 'r', 'economic', 'raw'}, optional\n        Determines what information is to be returned: either both Q and R\n        ('full', default), only R ('r') or both Q and R but computed in\n        economy-size ('economic', see Notes). The final option 'raw'\n        (added in SciPy 0.11) makes the function return two matrices\n        (Q, TAU) in the internal format used by LAPACK.\n    pivoting : bool, optional\n        Whether or not factorization should include pivoting for rank-revealing\n        qr decomposition. If pivoting, compute the decomposition\n        ``A P = Q R`` as above, but where P is chosen such that the diagonal\n        of R is non-increasing.\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    Q : float or complex ndarray\n        Of shape (M, M), or (M, K) for ``mode='economic'``. Not returned\n        if ``mode='r'``. Replaced by tuple ``(Q, TAU)`` if ``mode='raw'``.\n    R : float or complex ndarray\n        Of shape (M, N), or (K, N) for ``mode in ['economic', 'raw']``.\n        ``K = min(M, N)``.\n    P : int ndarray\n        Of shape (N,) for ``pivoting=True``. Not returned if\n        ``pivoting=False``.\n\n    Raises\n    ------\n    LinAlgError\n        Raised if decomposition fails\n\n    Notes\n    -----\n    This is an interface to the LAPACK routines dgeqrf, zgeqrf,\n    dorgqr, zungqr, dgeqp3, and zgeqp3.\n\n    If ``mode=economic``, the shapes of Q and R are (M, K) and (K, N) instead\n    of (M,M) and (M,N), with ``K=min(M,N)``.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy import linalg\n    >>> rng = np.random.default_rng()\n    >>> a = rng.standard_normal((9, 6))\n\n    >>> q, r = linalg.qr(a)\n    >>> np.allclose(a, np.dot(q, r))\n    True\n    >>> q.shape, r.shape\n    ((9, 9), (9, 6))\n\n    >>> r2 = linalg.qr(a, mode='r')\n    >>> np.allclose(r, r2)\n    True\n\n    >>> q3, r3 = linalg.qr(a, mode='economic')\n    >>> q3.shape, r3.shape\n    ((9, 6), (6, 6))\n\n    >>> q4, r4, p4 = linalg.qr(a, pivoting=True)\n    >>> d = np.abs(np.diag(r4))\n    >>> np.all(d[1:] <= d[:-1])\n    True\n    >>> np.allclose(a[:, p4], np.dot(q4, r4))\n    True\n    >>> q4.shape, r4.shape, p4.shape\n    ((9, 9), (9, 6), (6,))\n\n    >>> q5, r5, p5 = linalg.qr(a, mode='economic', pivoting=True)\n    >>> q5.shape, r5.shape, p5.shape\n    ((9, 6), (6, 6), (6,))\n\n    "
    if mode not in ['full', 'qr', 'r', 'economic', 'raw']:
        raise ValueError("Mode argument should be one of ['full', 'r','economic', 'raw']")
    if check_finite:
        a1 = numpy.asarray_chkfinite(a)
    else:
        a1 = numpy.asarray(a)
    if len(a1.shape) != 2:
        raise ValueError('expected a 2-D array')
    (M, N) = a1.shape
    overwrite_a = overwrite_a or _datacopied(a1, a)
    if pivoting:
        (geqp3,) = get_lapack_funcs(('geqp3',), (a1,))
        (qr, jpvt, tau) = safecall(geqp3, 'geqp3', a1, overwrite_a=overwrite_a)
        jpvt -= 1
    else:
        (geqrf,) = get_lapack_funcs(('geqrf',), (a1,))
        (qr, tau) = safecall(geqrf, 'geqrf', a1, lwork=lwork, overwrite_a=overwrite_a)
    if mode not in ['economic', 'raw'] or M < N:
        R = numpy.triu(qr)
    else:
        R = numpy.triu(qr[:N, :])
    if pivoting:
        Rj = (R, jpvt)
    else:
        Rj = (R,)
    if mode == 'r':
        return Rj
    elif mode == 'raw':
        return ((qr, tau),) + Rj
    (gor_un_gqr,) = get_lapack_funcs(('orgqr',), (qr,))
    if M < N:
        (Q,) = safecall(gor_un_gqr, 'gorgqr/gungqr', qr[:, :M], tau, lwork=lwork, overwrite_a=1)
    elif mode == 'economic':
        (Q,) = safecall(gor_un_gqr, 'gorgqr/gungqr', qr, tau, lwork=lwork, overwrite_a=1)
    else:
        t = qr.dtype.char
        qqr = numpy.empty((M, M), dtype=t)
        qqr[:, :N] = qr
        (Q,) = safecall(gor_un_gqr, 'gorgqr/gungqr', qqr, tau, lwork=lwork, overwrite_a=1)
    return (Q,) + Rj

def qr_multiply(a, c, mode='right', pivoting=False, conjugate=False, overwrite_a=False, overwrite_c=False):
    if False:
        print('Hello World!')
    "\n    Calculate the QR decomposition and multiply Q with a matrix.\n\n    Calculate the decomposition ``A = Q R`` where Q is unitary/orthogonal\n    and R upper triangular. Multiply Q with a vector or a matrix c.\n\n    Parameters\n    ----------\n    a : (M, N), array_like\n        Input array\n    c : array_like\n        Input array to be multiplied by ``q``.\n    mode : {'left', 'right'}, optional\n        ``Q @ c`` is returned if mode is 'left', ``c @ Q`` is returned if\n        mode is 'right'.\n        The shape of c must be appropriate for the matrix multiplications,\n        if mode is 'left', ``min(a.shape) == c.shape[0]``,\n        if mode is 'right', ``a.shape[0] == c.shape[1]``.\n    pivoting : bool, optional\n        Whether or not factorization should include pivoting for rank-revealing\n        qr decomposition, see the documentation of qr.\n    conjugate : bool, optional\n        Whether Q should be complex-conjugated. This might be faster\n        than explicit conjugation.\n    overwrite_a : bool, optional\n        Whether data in a is overwritten (may improve performance)\n    overwrite_c : bool, optional\n        Whether data in c is overwritten (may improve performance).\n        If this is used, c must be big enough to keep the result,\n        i.e. ``c.shape[0]`` = ``a.shape[0]`` if mode is 'left'.\n\n    Returns\n    -------\n    CQ : ndarray\n        The product of ``Q`` and ``c``.\n    R : (K, N), ndarray\n        R array of the resulting QR factorization where ``K = min(M, N)``.\n    P : (N,) ndarray\n        Integer pivot array. Only returned when ``pivoting=True``.\n\n    Raises\n    ------\n    LinAlgError\n        Raised if QR decomposition fails.\n\n    Notes\n    -----\n    This is an interface to the LAPACK routines ``?GEQRF``, ``?ORMQR``,\n    ``?UNMQR``, and ``?GEQP3``.\n\n    .. versionadded:: 0.11.0\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.linalg import qr_multiply, qr\n    >>> A = np.array([[1, 3, 3], [2, 3, 2], [2, 3, 3], [1, 3, 2]])\n    >>> qc, r1, piv1 = qr_multiply(A, 2*np.eye(4), pivoting=1)\n    >>> qc\n    array([[-1.,  1., -1.],\n           [-1., -1.,  1.],\n           [-1., -1., -1.],\n           [-1.,  1.,  1.]])\n    >>> r1\n    array([[-6., -3., -5.            ],\n           [ 0., -1., -1.11022302e-16],\n           [ 0.,  0., -1.            ]])\n    >>> piv1\n    array([1, 0, 2], dtype=int32)\n    >>> q2, r2, piv2 = qr(A, mode='economic', pivoting=1)\n    >>> np.allclose(2*q2 - qc, np.zeros((4, 3)))\n    True\n\n    "
    if mode not in ['left', 'right']:
        raise ValueError("Mode argument can only be 'left' or 'right' but not '{}'".format(mode))
    c = numpy.asarray_chkfinite(c)
    if c.ndim < 2:
        onedim = True
        c = numpy.atleast_2d(c)
        if mode == 'left':
            c = c.T
    else:
        onedim = False
    a = numpy.atleast_2d(numpy.asarray(a))
    (M, N) = a.shape
    if mode == 'left':
        if c.shape[0] != min(M, N + overwrite_c * (M - N)):
            raise ValueError('Array shapes are not compatible for Q @ c operation: {} vs {}'.format(a.shape, c.shape))
    elif M != c.shape[1]:
        raise ValueError('Array shapes are not compatible for c @ Q operation: {} vs {}'.format(c.shape, a.shape))
    raw = qr(a, overwrite_a, None, 'raw', pivoting)
    (Q, tau) = raw[0]
    (gor_un_mqr,) = get_lapack_funcs(('ormqr',), (Q,))
    if gor_un_mqr.typecode in ('s', 'd'):
        trans = 'T'
    else:
        trans = 'C'
    Q = Q[:, :min(M, N)]
    if M > N and mode == 'left' and (not overwrite_c):
        if conjugate:
            cc = numpy.zeros((c.shape[1], M), dtype=c.dtype, order='F')
            cc[:, :N] = c.T
        else:
            cc = numpy.zeros((M, c.shape[1]), dtype=c.dtype, order='F')
            cc[:N, :] = c
            trans = 'N'
        if conjugate:
            lr = 'R'
        else:
            lr = 'L'
        overwrite_c = True
    elif c.flags['C_CONTIGUOUS'] and trans == 'T' or conjugate:
        cc = c.T
        if mode == 'left':
            lr = 'R'
        else:
            lr = 'L'
    else:
        trans = 'N'
        cc = c
        if mode == 'left':
            lr = 'L'
        else:
            lr = 'R'
    (cQ,) = safecall(gor_un_mqr, 'gormqr/gunmqr', lr, trans, Q, tau, cc, overwrite_c=overwrite_c)
    if trans != 'N':
        cQ = cQ.T
    if mode == 'right':
        cQ = cQ[:, :min(M, N)]
    if onedim:
        cQ = cQ.ravel()
    return (cQ,) + raw[1:]

def rq(a, overwrite_a=False, lwork=None, mode='full', check_finite=True):
    if False:
        return 10
    "\n    Compute RQ decomposition of a matrix.\n\n    Calculate the decomposition ``A = R Q`` where Q is unitary/orthogonal\n    and R upper triangular.\n\n    Parameters\n    ----------\n    a : (M, N) array_like\n        Matrix to be decomposed\n    overwrite_a : bool, optional\n        Whether data in a is overwritten (may improve performance)\n    lwork : int, optional\n        Work array size, lwork >= a.shape[1]. If None or -1, an optimal size\n        is computed.\n    mode : {'full', 'r', 'economic'}, optional\n        Determines what information is to be returned: either both Q and R\n        ('full', default), only R ('r') or both Q and R but computed in\n        economy-size ('economic', see Notes).\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    R : float or complex ndarray\n        Of shape (M, N) or (M, K) for ``mode='economic'``. ``K = min(M, N)``.\n    Q : float or complex ndarray\n        Of shape (N, N) or (K, N) for ``mode='economic'``. Not returned\n        if ``mode='r'``.\n\n    Raises\n    ------\n    LinAlgError\n        If decomposition fails.\n\n    Notes\n    -----\n    This is an interface to the LAPACK routines sgerqf, dgerqf, cgerqf, zgerqf,\n    sorgrq, dorgrq, cungrq and zungrq.\n\n    If ``mode=economic``, the shapes of Q and R are (K, N) and (M, K) instead\n    of (N,N) and (M,N), with ``K=min(M,N)``.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy import linalg\n    >>> rng = np.random.default_rng()\n    >>> a = rng.standard_normal((6, 9))\n    >>> r, q = linalg.rq(a)\n    >>> np.allclose(a, r @ q)\n    True\n    >>> r.shape, q.shape\n    ((6, 9), (9, 9))\n    >>> r2 = linalg.rq(a, mode='r')\n    >>> np.allclose(r, r2)\n    True\n    >>> r3, q3 = linalg.rq(a, mode='economic')\n    >>> r3.shape, q3.shape\n    ((6, 6), (6, 9))\n\n    "
    if mode not in ['full', 'r', 'economic']:
        raise ValueError("Mode argument should be one of ['full', 'r', 'economic']")
    if check_finite:
        a1 = numpy.asarray_chkfinite(a)
    else:
        a1 = numpy.asarray(a)
    if len(a1.shape) != 2:
        raise ValueError('expected matrix')
    (M, N) = a1.shape
    overwrite_a = overwrite_a or _datacopied(a1, a)
    (gerqf,) = get_lapack_funcs(('gerqf',), (a1,))
    (rq, tau) = safecall(gerqf, 'gerqf', a1, lwork=lwork, overwrite_a=overwrite_a)
    if not mode == 'economic' or N < M:
        R = numpy.triu(rq, N - M)
    else:
        R = numpy.triu(rq[-M:, -M:])
    if mode == 'r':
        return R
    (gor_un_grq,) = get_lapack_funcs(('orgrq',), (rq,))
    if N < M:
        (Q,) = safecall(gor_un_grq, 'gorgrq/gungrq', rq[-N:], tau, lwork=lwork, overwrite_a=1)
    elif mode == 'economic':
        (Q,) = safecall(gor_un_grq, 'gorgrq/gungrq', rq, tau, lwork=lwork, overwrite_a=1)
    else:
        rq1 = numpy.empty((N, N), dtype=rq.dtype)
        rq1[-M:] = rq
        (Q,) = safecall(gor_un_grq, 'gorgrq/gungrq', rq1, tau, lwork=lwork, overwrite_a=1)
    return (R, Q)