"""
Direct wrappers for Fortran `id_dist` backend.
"""
import scipy.linalg._interpolative as _id
import numpy as np
_RETCODE_ERROR = RuntimeError('nonzero return code')

def _asfortranarray_copy(A):
    if False:
        while True:
            i = 10
    '\n    Same as np.asfortranarray, but ensure a copy\n    '
    A = np.asarray(A)
    if A.flags.f_contiguous:
        A = A.copy(order='F')
    else:
        A = np.asfortranarray(A)
    return A

def id_srand(n):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generate standard uniform pseudorandom numbers via a very efficient lagged\n    Fibonacci method.\n\n    :param n:\n        Number of pseudorandom numbers to generate.\n    :type n: int\n\n    :return:\n        Pseudorandom numbers.\n    :rtype: :class:`numpy.ndarray`\n    '
    return _id.id_srand(n)

def id_srandi(t):
    if False:
        i = 10
        return i + 15
    '\n    Initialize seed values for :func:`id_srand` (any appropriately random\n    numbers will do).\n\n    :param t:\n        Array of 55 seed values.\n    :type t: :class:`numpy.ndarray`\n    '
    t = np.asfortranarray(t)
    _id.id_srandi(t)

def id_srando():
    if False:
        print('Hello World!')
    '\n    Reset seed values to their original values.\n    '
    _id.id_srando()

def idd_frm(n, w, x):
    if False:
        i = 10
        return i + 15
    "\n    Transform real vector via a composition of Rokhlin's random transform,\n    random subselection, and an FFT.\n\n    In contrast to :func:`idd_sfrm`, this routine works best when the length of\n    the transformed vector is the power-of-two integer output by\n    :func:`idd_frmi`, or when the length is not specified but instead\n    determined a posteriori from the output. The returned transformed vector is\n    randomly permuted.\n\n    :param n:\n        Greatest power-of-two integer satisfying `n <= x.size` as obtained from\n        :func:`idd_frmi`; `n` is also the length of the output vector.\n    :type n: int\n    :param w:\n        Initialization array constructed by :func:`idd_frmi`.\n    :type w: :class:`numpy.ndarray`\n    :param x:\n        Vector to be transformed.\n    :type x: :class:`numpy.ndarray`\n\n    :return:\n        Transformed vector.\n    :rtype: :class:`numpy.ndarray`\n    "
    return _id.idd_frm(n, w, x)

def idd_sfrm(l, n, w, x):
    if False:
        for i in range(10):
            print('nop')
    "\n    Transform real vector via a composition of Rokhlin's random transform,\n    random subselection, and an FFT.\n\n    In contrast to :func:`idd_frm`, this routine works best when the length of\n    the transformed vector is known a priori.\n\n    :param l:\n        Length of transformed vector, satisfying `l <= n`.\n    :type l: int\n    :param n:\n        Greatest power-of-two integer satisfying `n <= x.size` as obtained from\n        :func:`idd_sfrmi`.\n    :type n: int\n    :param w:\n        Initialization array constructed by :func:`idd_sfrmi`.\n    :type w: :class:`numpy.ndarray`\n    :param x:\n        Vector to be transformed.\n    :type x: :class:`numpy.ndarray`\n\n    :return:\n        Transformed vector.\n    :rtype: :class:`numpy.ndarray`\n    "
    return _id.idd_sfrm(l, n, w, x)

def idd_frmi(m):
    if False:
        while True:
            i = 10
    '\n    Initialize data for :func:`idd_frm`.\n\n    :param m:\n        Length of vector to be transformed.\n    :type m: int\n\n    :return:\n        Greatest power-of-two integer `n` satisfying `n <= m`.\n    :rtype: int\n    :return:\n        Initialization array to be used by :func:`idd_frm`.\n    :rtype: :class:`numpy.ndarray`\n    '
    return _id.idd_frmi(m)

def idd_sfrmi(l, m):
    if False:
        print('Hello World!')
    '\n    Initialize data for :func:`idd_sfrm`.\n\n    :param l:\n        Length of output transformed vector.\n    :type l: int\n    :param m:\n        Length of the vector to be transformed.\n    :type m: int\n\n    :return:\n        Greatest power-of-two integer `n` satisfying `n <= m`.\n    :rtype: int\n    :return:\n        Initialization array to be used by :func:`idd_sfrm`.\n    :rtype: :class:`numpy.ndarray`\n    '
    return _id.idd_sfrmi(l, m)

def iddp_id(eps, A):
    if False:
        return 10
    '\n    Compute ID of a real matrix to a specified relative precision.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n\n    :return:\n        Rank of ID.\n    :rtype: int\n    :return:\n        Column index array.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Interpolation coefficients.\n    :rtype: :class:`numpy.ndarray`\n    '
    A = _asfortranarray_copy(A)
    (k, idx, rnorms) = _id.iddp_id(eps, A)
    n = A.shape[1]
    proj = A.T.ravel()[:k * (n - k)].reshape((k, n - k), order='F')
    return (k, idx, proj)

def iddr_id(A, k):
    if False:
        i = 10
        return i + 15
    '\n    Compute ID of a real matrix to a specified rank.\n\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n    :param k:\n        Rank of ID.\n    :type k: int\n\n    :return:\n        Column index array.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Interpolation coefficients.\n    :rtype: :class:`numpy.ndarray`\n    '
    A = _asfortranarray_copy(A)
    (idx, rnorms) = _id.iddr_id(A, k)
    n = A.shape[1]
    proj = A.T.ravel()[:k * (n - k)].reshape((k, n - k), order='F')
    return (idx, proj)

def idd_reconid(B, idx, proj):
    if False:
        for i in range(10):
            print('nop')
    '\n    Reconstruct matrix from real ID.\n\n    :param B:\n        Skeleton matrix.\n    :type B: :class:`numpy.ndarray`\n    :param idx:\n        Column index array.\n    :type idx: :class:`numpy.ndarray`\n    :param proj:\n        Interpolation coefficients.\n    :type proj: :class:`numpy.ndarray`\n\n    :return:\n        Reconstructed matrix.\n    :rtype: :class:`numpy.ndarray`\n    '
    B = np.asfortranarray(B)
    if proj.size > 0:
        return _id.idd_reconid(B, idx, proj)
    else:
        return B[:, np.argsort(idx)]

def idd_reconint(idx, proj):
    if False:
        print('Hello World!')
    '\n    Reconstruct interpolation matrix from real ID.\n\n    :param idx:\n        Column index array.\n    :type idx: :class:`numpy.ndarray`\n    :param proj:\n        Interpolation coefficients.\n    :type proj: :class:`numpy.ndarray`\n\n    :return:\n        Interpolation matrix.\n    :rtype: :class:`numpy.ndarray`\n    '
    return _id.idd_reconint(idx, proj)

def idd_copycols(A, k, idx):
    if False:
        print('Hello World!')
    '\n    Reconstruct skeleton matrix from real ID.\n\n    :param A:\n        Original matrix.\n    :type A: :class:`numpy.ndarray`\n    :param k:\n        Rank of ID.\n    :type k: int\n    :param idx:\n        Column index array.\n    :type idx: :class:`numpy.ndarray`\n\n    :return:\n        Skeleton matrix.\n    :rtype: :class:`numpy.ndarray`\n    '
    A = np.asfortranarray(A)
    return _id.idd_copycols(A, k, idx)

def idd_id2svd(B, idx, proj):
    if False:
        while True:
            i = 10
    '\n    Convert real ID to SVD.\n\n    :param B:\n        Skeleton matrix.\n    :type B: :class:`numpy.ndarray`\n    :param idx:\n        Column index array.\n    :type idx: :class:`numpy.ndarray`\n    :param proj:\n        Interpolation coefficients.\n    :type proj: :class:`numpy.ndarray`\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    '
    B = np.asfortranarray(B)
    (U, V, S, ier) = _id.idd_id2svd(B, idx, proj)
    if ier:
        raise _RETCODE_ERROR
    return (U, V, S)

def idd_snorm(m, n, matvect, matvec, its=20):
    if False:
        i = 10
        return i + 15
    '\n    Estimate spectral norm of a real matrix by the randomized power method.\n\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matvect:\n        Function to apply the matrix transpose to a vector, with call signature\n        `y = matvect(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvect: function\n    :param matvec:\n        Function to apply the matrix to a vector, with call signature\n        `y = matvec(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvec: function\n    :param its:\n        Number of power method iterations.\n    :type its: int\n\n    :return:\n        Spectral norm estimate.\n    :rtype: float\n    '
    (snorm, v) = _id.idd_snorm(m, n, matvect, matvec, its)
    return snorm

def idd_diffsnorm(m, n, matvect, matvect2, matvec, matvec2, its=20):
    if False:
        i = 10
        return i + 15
    '\n    Estimate spectral norm of the difference of two real matrices by the\n    randomized power method.\n\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matvect:\n        Function to apply the transpose of the first matrix to a vector, with\n        call signature `y = matvect(x)`, where `x` and `y` are the input and\n        output vectors, respectively.\n    :type matvect: function\n    :param matvect2:\n        Function to apply the transpose of the second matrix to a vector, with\n        call signature `y = matvect2(x)`, where `x` and `y` are the input and\n        output vectors, respectively.\n    :type matvect2: function\n    :param matvec:\n        Function to apply the first matrix to a vector, with call signature\n        `y = matvec(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvec: function\n    :param matvec2:\n        Function to apply the second matrix to a vector, with call signature\n        `y = matvec2(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvec2: function\n    :param its:\n        Number of power method iterations.\n    :type its: int\n\n    :return:\n        Spectral norm estimate of matrix difference.\n    :rtype: float\n    '
    return _id.idd_diffsnorm(m, n, matvect, matvect2, matvec, matvec2, its)

def iddr_svd(A, k):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute SVD of a real matrix to a specified rank.\n\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n    :param k:\n        Rank of SVD.\n    :type k: int\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    '
    A = np.asfortranarray(A)
    (U, V, S, ier) = _id.iddr_svd(A, k)
    if ier:
        raise _RETCODE_ERROR
    return (U, V, S)

def iddp_svd(eps, A):
    if False:
        return 10
    '\n    Compute SVD of a real matrix to a specified relative precision.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    '
    A = np.asfortranarray(A)
    (m, n) = A.shape
    (k, iU, iV, iS, w, ier) = _id.iddp_svd(eps, A)
    if ier:
        raise _RETCODE_ERROR
    U = w[iU - 1:iU + m * k - 1].reshape((m, k), order='F')
    V = w[iV - 1:iV + n * k - 1].reshape((n, k), order='F')
    S = w[iS - 1:iS + k - 1]
    return (U, V, S)

def iddp_aid(eps, A):
    if False:
        i = 10
        return i + 15
    '\n    Compute ID of a real matrix to a specified relative precision using random\n    sampling.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n\n    :return:\n        Rank of ID.\n    :rtype: int\n    :return:\n        Column index array.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Interpolation coefficients.\n    :rtype: :class:`numpy.ndarray`\n    '
    A = np.asfortranarray(A)
    (m, n) = A.shape
    (n2, w) = idd_frmi(m)
    proj = np.empty(n * (2 * n2 + 1) + n2 + 1, order='F')
    (k, idx, proj) = _id.iddp_aid(eps, A, w, proj)
    proj = proj[:k * (n - k)].reshape((k, n - k), order='F')
    return (k, idx, proj)

def idd_estrank(eps, A):
    if False:
        while True:
            i = 10
    '\n    Estimate rank of a real matrix to a specified relative precision using\n    random sampling.\n\n    The output rank is typically about 8 higher than the actual rank.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n\n    :return:\n        Rank estimate.\n    :rtype: int\n    '
    A = np.asfortranarray(A)
    (m, n) = A.shape
    (n2, w) = idd_frmi(m)
    ra = np.empty(n * n2 + (n + 1) * (n2 + 1), order='F')
    (k, ra) = _id.idd_estrank(eps, A, w, ra)
    return k

def iddp_asvd(eps, A):
    if False:
        print('Hello World!')
    '\n    Compute SVD of a real matrix to a specified relative precision using random\n    sampling.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    '
    A = np.asfortranarray(A)
    (m, n) = A.shape
    (n2, winit) = _id.idd_frmi(m)
    w = np.empty(max((min(m, n) + 1) * (3 * m + 5 * n + 1) + 25 * min(m, n) ** 2, (2 * n + 1) * (n2 + 1)), order='F')
    (k, iU, iV, iS, w, ier) = _id.iddp_asvd(eps, A, winit, w)
    if ier:
        raise _RETCODE_ERROR
    U = w[iU - 1:iU + m * k - 1].reshape((m, k), order='F')
    V = w[iV - 1:iV + n * k - 1].reshape((n, k), order='F')
    S = w[iS - 1:iS + k - 1]
    return (U, V, S)

def iddp_rid(eps, m, n, matvect):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute ID of a real matrix to a specified relative precision using random\n    matrix-vector multiplication.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matvect:\n        Function to apply the matrix transpose to a vector, with call signature\n        `y = matvect(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvect: function\n\n    :return:\n        Rank of ID.\n    :rtype: int\n    :return:\n        Column index array.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Interpolation coefficients.\n    :rtype: :class:`numpy.ndarray`\n    '
    proj = np.empty(m + 1 + 2 * n * (min(m, n) + 1), order='F')
    (k, idx, proj, ier) = _id.iddp_rid(eps, m, n, matvect, proj)
    if ier != 0:
        raise _RETCODE_ERROR
    proj = proj[:k * (n - k)].reshape((k, n - k), order='F')
    return (k, idx, proj)

def idd_findrank(eps, m, n, matvect):
    if False:
        return 10
    '\n    Estimate rank of a real matrix to a specified relative precision using\n    random matrix-vector multiplication.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matvect:\n        Function to apply the matrix transpose to a vector, with call signature\n        `y = matvect(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvect: function\n\n    :return:\n        Rank estimate.\n    :rtype: int\n    '
    (k, ra, ier) = _id.idd_findrank(eps, m, n, matvect)
    if ier:
        raise _RETCODE_ERROR
    return k

def iddp_rsvd(eps, m, n, matvect, matvec):
    if False:
        print('Hello World!')
    '\n    Compute SVD of a real matrix to a specified relative precision using random\n    matrix-vector multiplication.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matvect:\n        Function to apply the matrix transpose to a vector, with call signature\n        `y = matvect(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvect: function\n    :param matvec:\n        Function to apply the matrix to a vector, with call signature\n        `y = matvec(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvec: function\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    '
    (k, iU, iV, iS, w, ier) = _id.iddp_rsvd(eps, m, n, matvect, matvec)
    if ier:
        raise _RETCODE_ERROR
    U = w[iU - 1:iU + m * k - 1].reshape((m, k), order='F')
    V = w[iV - 1:iV + n * k - 1].reshape((n, k), order='F')
    S = w[iS - 1:iS + k - 1]
    return (U, V, S)

def iddr_aid(A, k):
    if False:
        while True:
            i = 10
    '\n    Compute ID of a real matrix to a specified rank using random sampling.\n\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n    :param k:\n        Rank of ID.\n    :type k: int\n\n    :return:\n        Column index array.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Interpolation coefficients.\n    :rtype: :class:`numpy.ndarray`\n    '
    A = np.asfortranarray(A)
    (m, n) = A.shape
    w = iddr_aidi(m, n, k)
    (idx, proj) = _id.iddr_aid(A, k, w)
    if k == n:
        proj = np.empty((k, n - k), dtype='float64', order='F')
    else:
        proj = proj.reshape((k, n - k), order='F')
    return (idx, proj)

def iddr_aidi(m, n, k):
    if False:
        for i in range(10):
            print('nop')
    '\n    Initialize array for :func:`iddr_aid`.\n\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param k:\n        Rank of ID.\n    :type k: int\n\n    :return:\n        Initialization array to be used by :func:`iddr_aid`.\n    :rtype: :class:`numpy.ndarray`\n    '
    return _id.iddr_aidi(m, n, k)

def iddr_asvd(A, k):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute SVD of a real matrix to a specified rank using random sampling.\n\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n    :param k:\n        Rank of SVD.\n    :type k: int\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    '
    A = np.asfortranarray(A)
    (m, n) = A.shape
    w = np.empty((2 * k + 28) * m + (6 * k + 21) * n + 25 * k ** 2 + 100, order='F')
    w_ = iddr_aidi(m, n, k)
    w[:w_.size] = w_
    (U, V, S, ier) = _id.iddr_asvd(A, k, w)
    if ier != 0:
        raise _RETCODE_ERROR
    return (U, V, S)

def iddr_rid(m, n, matvect, k):
    if False:
        while True:
            i = 10
    '\n    Compute ID of a real matrix to a specified rank using random matrix-vector\n    multiplication.\n\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matvect:\n        Function to apply the matrix transpose to a vector, with call signature\n        `y = matvect(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvect: function\n    :param k:\n        Rank of ID.\n    :type k: int\n\n    :return:\n        Column index array.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Interpolation coefficients.\n    :rtype: :class:`numpy.ndarray`\n    '
    (idx, proj) = _id.iddr_rid(m, n, matvect, k)
    proj = proj[:k * (n - k)].reshape((k, n - k), order='F')
    return (idx, proj)

def iddr_rsvd(m, n, matvect, matvec, k):
    if False:
        print('Hello World!')
    '\n    Compute SVD of a real matrix to a specified rank using random matrix-vector\n    multiplication.\n\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matvect:\n        Function to apply the matrix transpose to a vector, with call signature\n        `y = matvect(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvect: function\n    :param matvec:\n        Function to apply the matrix to a vector, with call signature\n        `y = matvec(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvec: function\n    :param k:\n        Rank of SVD.\n    :type k: int\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    '
    (U, V, S, ier) = _id.iddr_rsvd(m, n, matvect, matvec, k)
    if ier != 0:
        raise _RETCODE_ERROR
    return (U, V, S)

def idz_frm(n, w, x):
    if False:
        i = 10
        return i + 15
    "\n    Transform complex vector via a composition of Rokhlin's random transform,\n    random subselection, and an FFT.\n\n    In contrast to :func:`idz_sfrm`, this routine works best when the length of\n    the transformed vector is the power-of-two integer output by\n    :func:`idz_frmi`, or when the length is not specified but instead\n    determined a posteriori from the output. The returned transformed vector is\n    randomly permuted.\n\n    :param n:\n        Greatest power-of-two integer satisfying `n <= x.size` as obtained from\n        :func:`idz_frmi`; `n` is also the length of the output vector.\n    :type n: int\n    :param w:\n        Initialization array constructed by :func:`idz_frmi`.\n    :type w: :class:`numpy.ndarray`\n    :param x:\n        Vector to be transformed.\n    :type x: :class:`numpy.ndarray`\n\n    :return:\n        Transformed vector.\n    :rtype: :class:`numpy.ndarray`\n    "
    return _id.idz_frm(n, w, x)

def idz_sfrm(l, n, w, x):
    if False:
        for i in range(10):
            print('nop')
    "\n    Transform complex vector via a composition of Rokhlin's random transform,\n    random subselection, and an FFT.\n\n    In contrast to :func:`idz_frm`, this routine works best when the length of\n    the transformed vector is known a priori.\n\n    :param l:\n        Length of transformed vector, satisfying `l <= n`.\n    :type l: int\n    :param n:\n        Greatest power-of-two integer satisfying `n <= x.size` as obtained from\n        :func:`idz_sfrmi`.\n    :type n: int\n    :param w:\n        Initialization array constructed by :func:`idd_sfrmi`.\n    :type w: :class:`numpy.ndarray`\n    :param x:\n        Vector to be transformed.\n    :type x: :class:`numpy.ndarray`\n\n    :return:\n        Transformed vector.\n    :rtype: :class:`numpy.ndarray`\n    "
    return _id.idz_sfrm(l, n, w, x)

def idz_frmi(m):
    if False:
        return 10
    '\n    Initialize data for :func:`idz_frm`.\n\n    :param m:\n        Length of vector to be transformed.\n    :type m: int\n\n    :return:\n        Greatest power-of-two integer `n` satisfying `n <= m`.\n    :rtype: int\n    :return:\n        Initialization array to be used by :func:`idz_frm`.\n    :rtype: :class:`numpy.ndarray`\n    '
    return _id.idz_frmi(m)

def idz_sfrmi(l, m):
    if False:
        for i in range(10):
            print('nop')
    '\n    Initialize data for :func:`idz_sfrm`.\n\n    :param l:\n        Length of output transformed vector.\n    :type l: int\n    :param m:\n        Length of the vector to be transformed.\n    :type m: int\n\n    :return:\n        Greatest power-of-two integer `n` satisfying `n <= m`.\n    :rtype: int\n    :return:\n        Initialization array to be used by :func:`idz_sfrm`.\n    :rtype: :class:`numpy.ndarray`\n    '
    return _id.idz_sfrmi(l, m)

def idzp_id(eps, A):
    if False:
        print('Hello World!')
    '\n    Compute ID of a complex matrix to a specified relative precision.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n\n    :return:\n        Rank of ID.\n    :rtype: int\n    :return:\n        Column index array.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Interpolation coefficients.\n    :rtype: :class:`numpy.ndarray`\n    '
    A = _asfortranarray_copy(A)
    (k, idx, rnorms) = _id.idzp_id(eps, A)
    n = A.shape[1]
    proj = A.T.ravel()[:k * (n - k)].reshape((k, n - k), order='F')
    return (k, idx, proj)

def idzr_id(A, k):
    if False:
        return 10
    '\n    Compute ID of a complex matrix to a specified rank.\n\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n    :param k:\n        Rank of ID.\n    :type k: int\n\n    :return:\n        Column index array.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Interpolation coefficients.\n    :rtype: :class:`numpy.ndarray`\n    '
    A = _asfortranarray_copy(A)
    (idx, rnorms) = _id.idzr_id(A, k)
    n = A.shape[1]
    proj = A.T.ravel()[:k * (n - k)].reshape((k, n - k), order='F')
    return (idx, proj)

def idz_reconid(B, idx, proj):
    if False:
        i = 10
        return i + 15
    '\n    Reconstruct matrix from complex ID.\n\n    :param B:\n        Skeleton matrix.\n    :type B: :class:`numpy.ndarray`\n    :param idx:\n        Column index array.\n    :type idx: :class:`numpy.ndarray`\n    :param proj:\n        Interpolation coefficients.\n    :type proj: :class:`numpy.ndarray`\n\n    :return:\n        Reconstructed matrix.\n    :rtype: :class:`numpy.ndarray`\n    '
    B = np.asfortranarray(B)
    if proj.size > 0:
        return _id.idz_reconid(B, idx, proj)
    else:
        return B[:, np.argsort(idx)]

def idz_reconint(idx, proj):
    if False:
        return 10
    '\n    Reconstruct interpolation matrix from complex ID.\n\n    :param idx:\n        Column index array.\n    :type idx: :class:`numpy.ndarray`\n    :param proj:\n        Interpolation coefficients.\n    :type proj: :class:`numpy.ndarray`\n\n    :return:\n        Interpolation matrix.\n    :rtype: :class:`numpy.ndarray`\n    '
    return _id.idz_reconint(idx, proj)

def idz_copycols(A, k, idx):
    if False:
        return 10
    '\n    Reconstruct skeleton matrix from complex ID.\n\n    :param A:\n        Original matrix.\n    :type A: :class:`numpy.ndarray`\n    :param k:\n        Rank of ID.\n    :type k: int\n    :param idx:\n        Column index array.\n    :type idx: :class:`numpy.ndarray`\n\n    :return:\n        Skeleton matrix.\n    :rtype: :class:`numpy.ndarray`\n    '
    A = np.asfortranarray(A)
    return _id.idz_copycols(A, k, idx)

def idz_id2svd(B, idx, proj):
    if False:
        return 10
    '\n    Convert complex ID to SVD.\n\n    :param B:\n        Skeleton matrix.\n    :type B: :class:`numpy.ndarray`\n    :param idx:\n        Column index array.\n    :type idx: :class:`numpy.ndarray`\n    :param proj:\n        Interpolation coefficients.\n    :type proj: :class:`numpy.ndarray`\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    '
    B = np.asfortranarray(B)
    (U, V, S, ier) = _id.idz_id2svd(B, idx, proj)
    if ier:
        raise _RETCODE_ERROR
    return (U, V, S)

def idz_snorm(m, n, matveca, matvec, its=20):
    if False:
        i = 10
        return i + 15
    '\n    Estimate spectral norm of a complex matrix by the randomized power method.\n\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matveca:\n        Function to apply the matrix adjoint to a vector, with call signature\n        `y = matveca(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matveca: function\n    :param matvec:\n        Function to apply the matrix to a vector, with call signature\n        `y = matvec(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvec: function\n    :param its:\n        Number of power method iterations.\n    :type its: int\n\n    :return:\n        Spectral norm estimate.\n    :rtype: float\n    '
    (snorm, v) = _id.idz_snorm(m, n, matveca, matvec, its)
    return snorm

def idz_diffsnorm(m, n, matveca, matveca2, matvec, matvec2, its=20):
    if False:
        for i in range(10):
            print('nop')
    '\n    Estimate spectral norm of the difference of two complex matrices by the\n    randomized power method.\n\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matveca:\n        Function to apply the adjoint of the first matrix to a vector, with\n        call signature `y = matveca(x)`, where `x` and `y` are the input and\n        output vectors, respectively.\n    :type matveca: function\n    :param matveca2:\n        Function to apply the adjoint of the second matrix to a vector, with\n        call signature `y = matveca2(x)`, where `x` and `y` are the input and\n        output vectors, respectively.\n    :type matveca2: function\n    :param matvec:\n        Function to apply the first matrix to a vector, with call signature\n        `y = matvec(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvec: function\n    :param matvec2:\n        Function to apply the second matrix to a vector, with call signature\n        `y = matvec2(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvec2: function\n    :param its:\n        Number of power method iterations.\n    :type its: int\n\n    :return:\n        Spectral norm estimate of matrix difference.\n    :rtype: float\n    '
    return _id.idz_diffsnorm(m, n, matveca, matveca2, matvec, matvec2, its)

def idzr_svd(A, k):
    if False:
        return 10
    '\n    Compute SVD of a complex matrix to a specified rank.\n\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n    :param k:\n        Rank of SVD.\n    :type k: int\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    '
    A = np.asfortranarray(A)
    (U, V, S, ier) = _id.idzr_svd(A, k)
    if ier:
        raise _RETCODE_ERROR
    return (U, V, S)

def idzp_svd(eps, A):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute SVD of a complex matrix to a specified relative precision.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    '
    A = np.asfortranarray(A)
    (m, n) = A.shape
    (k, iU, iV, iS, w, ier) = _id.idzp_svd(eps, A)
    if ier:
        raise _RETCODE_ERROR
    U = w[iU - 1:iU + m * k - 1].reshape((m, k), order='F')
    V = w[iV - 1:iV + n * k - 1].reshape((n, k), order='F')
    S = w[iS - 1:iS + k - 1]
    return (U, V, S)

def idzp_aid(eps, A):
    if False:
        while True:
            i = 10
    '\n    Compute ID of a complex matrix to a specified relative precision using\n    random sampling.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n\n    :return:\n        Rank of ID.\n    :rtype: int\n    :return:\n        Column index array.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Interpolation coefficients.\n    :rtype: :class:`numpy.ndarray`\n    '
    A = np.asfortranarray(A)
    (m, n) = A.shape
    (n2, w) = idz_frmi(m)
    proj = np.empty(n * (2 * n2 + 1) + n2 + 1, dtype='complex128', order='F')
    (k, idx, proj) = _id.idzp_aid(eps, A, w, proj)
    proj = proj[:k * (n - k)].reshape((k, n - k), order='F')
    return (k, idx, proj)

def idz_estrank(eps, A):
    if False:
        print('Hello World!')
    '\n    Estimate rank of a complex matrix to a specified relative precision using\n    random sampling.\n\n    The output rank is typically about 8 higher than the actual rank.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n\n    :return:\n        Rank estimate.\n    :rtype: int\n    '
    A = np.asfortranarray(A)
    (m, n) = A.shape
    (n2, w) = idz_frmi(m)
    ra = np.empty(n * n2 + (n + 1) * (n2 + 1), dtype='complex128', order='F')
    (k, ra) = _id.idz_estrank(eps, A, w, ra)
    return k

def idzp_asvd(eps, A):
    if False:
        return 10
    '\n    Compute SVD of a complex matrix to a specified relative precision using\n    random sampling.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    '
    A = np.asfortranarray(A)
    (m, n) = A.shape
    (n2, winit) = _id.idz_frmi(m)
    w = np.empty(max((min(m, n) + 1) * (3 * m + 5 * n + 11) + 8 * min(m, n) ** 2, (2 * n + 1) * (n2 + 1)), dtype=np.complex128, order='F')
    (k, iU, iV, iS, w, ier) = _id.idzp_asvd(eps, A, winit, w)
    if ier:
        raise _RETCODE_ERROR
    U = w[iU - 1:iU + m * k - 1].reshape((m, k), order='F')
    V = w[iV - 1:iV + n * k - 1].reshape((n, k), order='F')
    S = w[iS - 1:iS + k - 1]
    return (U, V, S)

def idzp_rid(eps, m, n, matveca):
    if False:
        return 10
    '\n    Compute ID of a complex matrix to a specified relative precision using\n    random matrix-vector multiplication.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matveca:\n        Function to apply the matrix adjoint to a vector, with call signature\n        `y = matveca(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matveca: function\n\n    :return:\n        Rank of ID.\n    :rtype: int\n    :return:\n        Column index array.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Interpolation coefficients.\n    :rtype: :class:`numpy.ndarray`\n    '
    proj = np.empty(m + 1 + 2 * n * (min(m, n) + 1), dtype=np.complex128, order='F')
    (k, idx, proj, ier) = _id.idzp_rid(eps, m, n, matveca, proj)
    if ier:
        raise _RETCODE_ERROR
    proj = proj[:k * (n - k)].reshape((k, n - k), order='F')
    return (k, idx, proj)

def idz_findrank(eps, m, n, matveca):
    if False:
        while True:
            i = 10
    '\n    Estimate rank of a complex matrix to a specified relative precision using\n    random matrix-vector multiplication.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matveca:\n        Function to apply the matrix adjoint to a vector, with call signature\n        `y = matveca(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matveca: function\n\n    :return:\n        Rank estimate.\n    :rtype: int\n    '
    (k, ra, ier) = _id.idz_findrank(eps, m, n, matveca)
    if ier:
        raise _RETCODE_ERROR
    return k

def idzp_rsvd(eps, m, n, matveca, matvec):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute SVD of a complex matrix to a specified relative precision using\n    random matrix-vector multiplication.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matveca:\n        Function to apply the matrix adjoint to a vector, with call signature\n        `y = matveca(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matveca: function\n    :param matvec:\n        Function to apply the matrix to a vector, with call signature\n        `y = matvec(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvec: function\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    '
    (k, iU, iV, iS, w, ier) = _id.idzp_rsvd(eps, m, n, matveca, matvec)
    if ier:
        raise _RETCODE_ERROR
    U = w[iU - 1:iU + m * k - 1].reshape((m, k), order='F')
    V = w[iV - 1:iV + n * k - 1].reshape((n, k), order='F')
    S = w[iS - 1:iS + k - 1]
    return (U, V, S)

def idzr_aid(A, k):
    if False:
        i = 10
        return i + 15
    '\n    Compute ID of a complex matrix to a specified rank using random sampling.\n\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n    :param k:\n        Rank of ID.\n    :type k: int\n\n    :return:\n        Column index array.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Interpolation coefficients.\n    :rtype: :class:`numpy.ndarray`\n    '
    A = np.asfortranarray(A)
    (m, n) = A.shape
    w = idzr_aidi(m, n, k)
    (idx, proj) = _id.idzr_aid(A, k, w)
    if k == n:
        proj = np.empty((k, n - k), dtype='complex128', order='F')
    else:
        proj = proj.reshape((k, n - k), order='F')
    return (idx, proj)

def idzr_aidi(m, n, k):
    if False:
        for i in range(10):
            print('nop')
    '\n    Initialize array for :func:`idzr_aid`.\n\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param k:\n        Rank of ID.\n    :type k: int\n\n    :return:\n        Initialization array to be used by :func:`idzr_aid`.\n    :rtype: :class:`numpy.ndarray`\n    '
    return _id.idzr_aidi(m, n, k)

def idzr_asvd(A, k):
    if False:
        i = 10
        return i + 15
    '\n    Compute SVD of a complex matrix to a specified rank using random sampling.\n\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n    :param k:\n        Rank of SVD.\n    :type k: int\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    '
    A = np.asfortranarray(A)
    (m, n) = A.shape
    w = np.empty((2 * k + 22) * m + (6 * k + 21) * n + 8 * k ** 2 + 10 * k + 90, dtype='complex128', order='F')
    w_ = idzr_aidi(m, n, k)
    w[:w_.size] = w_
    (U, V, S, ier) = _id.idzr_asvd(A, k, w)
    if ier:
        raise _RETCODE_ERROR
    return (U, V, S)

def idzr_rid(m, n, matveca, k):
    if False:
        return 10
    '\n    Compute ID of a complex matrix to a specified rank using random\n    matrix-vector multiplication.\n\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matveca:\n        Function to apply the matrix adjoint to a vector, with call signature\n        `y = matveca(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matveca: function\n    :param k:\n        Rank of ID.\n    :type k: int\n\n    :return:\n        Column index array.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Interpolation coefficients.\n    :rtype: :class:`numpy.ndarray`\n    '
    (idx, proj) = _id.idzr_rid(m, n, matveca, k)
    proj = proj[:k * (n - k)].reshape((k, n - k), order='F')
    return (idx, proj)

def idzr_rsvd(m, n, matveca, matvec, k):
    if False:
        while True:
            i = 10
    '\n    Compute SVD of a complex matrix to a specified rank using random\n    matrix-vector multiplication.\n\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matveca:\n        Function to apply the matrix adjoint to a vector, with call signature\n        `y = matveca(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matveca: function\n    :param matvec:\n        Function to apply the matrix to a vector, with call signature\n        `y = matvec(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvec: function\n    :param k:\n        Rank of SVD.\n    :type k: int\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    '
    (U, V, S, ier) = _id.idzr_rsvd(m, n, matveca, matvec, k)
    if ier:
        raise _RETCODE_ERROR
    return (U, V, S)