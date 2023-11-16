__all__ = ['eig', 'eigvals', 'eigh', 'eigvalsh', 'eig_banded', 'eigvals_banded', 'eigh_tridiagonal', 'eigvalsh_tridiagonal', 'hessenberg', 'cdf2rdf']
import warnings
import numpy
from numpy import array, isfinite, inexact, nonzero, iscomplexobj, flatnonzero, conj, asarray, argsort, empty, iscomplex, zeros, einsum, eye, inf
from scipy._lib._util import _asarray_validated
from ._misc import LinAlgError, _datacopied, norm
from .lapack import get_lapack_funcs, _compute_lwork
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
_I = numpy.array(1j, dtype='F')

def _make_complex_eigvecs(w, vin, dtype):
    if False:
        print('Hello World!')
    '\n    Produce complex-valued eigenvectors from LAPACK DGGEV real-valued output\n    '
    v = numpy.array(vin, dtype=dtype)
    m = w.imag > 0
    m[:-1] |= w.imag[1:] < 0
    for i in flatnonzero(m):
        v.imag[:, i] = vin[:, i + 1]
        conj(v[:, i], v[:, i + 1])
    return v

def _make_eigvals(alpha, beta, homogeneous_eigvals):
    if False:
        return 10
    if homogeneous_eigvals:
        if beta is None:
            return numpy.vstack((alpha, numpy.ones_like(alpha)))
        else:
            return numpy.vstack((alpha, beta))
    elif beta is None:
        return alpha
    else:
        w = numpy.empty_like(alpha)
        alpha_zero = alpha == 0
        beta_zero = beta == 0
        beta_nonzero = ~beta_zero
        w[beta_nonzero] = alpha[beta_nonzero] / beta[beta_nonzero]
        w[~alpha_zero & beta_zero] = numpy.inf
        if numpy.all(alpha.imag == 0):
            w[alpha_zero & beta_zero] = numpy.nan
        else:
            w[alpha_zero & beta_zero] = complex(numpy.nan, numpy.nan)
        return w

def _geneig(a1, b1, left, right, overwrite_a, overwrite_b, homogeneous_eigvals):
    if False:
        print('Hello World!')
    (ggev,) = get_lapack_funcs(('ggev',), (a1, b1))
    (cvl, cvr) = (left, right)
    res = ggev(a1, b1, lwork=-1)
    lwork = res[-2][0].real.astype(numpy.int_)
    if ggev.typecode in 'cz':
        (alpha, beta, vl, vr, work, info) = ggev(a1, b1, cvl, cvr, lwork, overwrite_a, overwrite_b)
        w = _make_eigvals(alpha, beta, homogeneous_eigvals)
    else:
        (alphar, alphai, beta, vl, vr, work, info) = ggev(a1, b1, cvl, cvr, lwork, overwrite_a, overwrite_b)
        alpha = alphar + _I * alphai
        w = _make_eigvals(alpha, beta, homogeneous_eigvals)
    _check_info(info, 'generalized eig algorithm (ggev)')
    only_real = numpy.all(w.imag == 0.0)
    if not (ggev.typecode in 'cz' or only_real):
        t = w.dtype.char
        if left:
            vl = _make_complex_eigvecs(w, vl, t)
        if right:
            vr = _make_complex_eigvecs(w, vr, t)
    for i in range(vr.shape[0]):
        if right:
            vr[:, i] /= norm(vr[:, i])
        if left:
            vl[:, i] /= norm(vl[:, i])
    if not (left or right):
        return w
    if left:
        if right:
            return (w, vl, vr)
        return (w, vl)
    return (w, vr)

def eig(a, b=None, left=False, right=True, overwrite_a=False, overwrite_b=False, check_finite=True, homogeneous_eigvals=False):
    if False:
        while True:
            i = 10
    '\n    Solve an ordinary or generalized eigenvalue problem of a square matrix.\n\n    Find eigenvalues w and right or left eigenvectors of a general matrix::\n\n        a   vr[:,i] = w[i]        b   vr[:,i]\n        a.H vl[:,i] = w[i].conj() b.H vl[:,i]\n\n    where ``.H`` is the Hermitian conjugation.\n\n    Parameters\n    ----------\n    a : (M, M) array_like\n        A complex or real matrix whose eigenvalues and eigenvectors\n        will be computed.\n    b : (M, M) array_like, optional\n        Right-hand side matrix in a generalized eigenvalue problem.\n        Default is None, identity matrix is assumed.\n    left : bool, optional\n        Whether to calculate and return left eigenvectors.  Default is False.\n    right : bool, optional\n        Whether to calculate and return right eigenvectors.  Default is True.\n    overwrite_a : bool, optional\n        Whether to overwrite `a`; may improve performance.  Default is False.\n    overwrite_b : bool, optional\n        Whether to overwrite `b`; may improve performance.  Default is False.\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n    homogeneous_eigvals : bool, optional\n        If True, return the eigenvalues in homogeneous coordinates.\n        In this case ``w`` is a (2, M) array so that::\n\n            w[1,i] a vr[:,i] = w[0,i] b vr[:,i]\n\n        Default is False.\n\n    Returns\n    -------\n    w : (M,) or (2, M) double or complex ndarray\n        The eigenvalues, each repeated according to its\n        multiplicity. The shape is (M,) unless\n        ``homogeneous_eigvals=True``.\n    vl : (M, M) double or complex ndarray\n        The normalized left eigenvector corresponding to the eigenvalue\n        ``w[i]`` is the column ``vl[:,i]``. Only returned if ``left=True``.\n    vr : (M, M) double or complex ndarray\n        The normalized right eigenvector corresponding to the eigenvalue\n        ``w[i]`` is the column ``vr[:,i]``.  Only returned if ``right=True``.\n\n    Raises\n    ------\n    LinAlgError\n        If eigenvalue computation does not converge.\n\n    See Also\n    --------\n    eigvals : eigenvalues of general arrays\n    eigh : Eigenvalues and right eigenvectors for symmetric/Hermitian arrays.\n    eig_banded : eigenvalues and right eigenvectors for symmetric/Hermitian\n        band matrices\n    eigh_tridiagonal : eigenvalues and right eiegenvectors for\n        symmetric/Hermitian tridiagonal matrices\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy import linalg\n    >>> a = np.array([[0., -1.], [1., 0.]])\n    >>> linalg.eigvals(a)\n    array([0.+1.j, 0.-1.j])\n\n    >>> b = np.array([[0., 1.], [1., 1.]])\n    >>> linalg.eigvals(a, b)\n    array([ 1.+0.j, -1.+0.j])\n\n    >>> a = np.array([[3., 0., 0.], [0., 8., 0.], [0., 0., 7.]])\n    >>> linalg.eigvals(a, homogeneous_eigvals=True)\n    array([[3.+0.j, 8.+0.j, 7.+0.j],\n           [1.+0.j, 1.+0.j, 1.+0.j]])\n\n    >>> a = np.array([[0., -1.], [1., 0.]])\n    >>> linalg.eigvals(a) == linalg.eig(a)[0]\n    array([ True,  True])\n    >>> linalg.eig(a, left=True, right=False)[1] # normalized left eigenvector\n    array([[-0.70710678+0.j        , -0.70710678-0.j        ],\n           [-0.        +0.70710678j, -0.        -0.70710678j]])\n    >>> linalg.eig(a, left=False, right=True)[1] # normalized right eigenvector\n    array([[0.70710678+0.j        , 0.70710678-0.j        ],\n           [0.        -0.70710678j, 0.        +0.70710678j]])\n\n\n\n    '
    a1 = _asarray_validated(a, check_finite=check_finite)
    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square matrix')
    overwrite_a = overwrite_a or _datacopied(a1, a)
    if b is not None:
        b1 = _asarray_validated(b, check_finite=check_finite)
        overwrite_b = overwrite_b or _datacopied(b1, b)
        if len(b1.shape) != 2 or b1.shape[0] != b1.shape[1]:
            raise ValueError('expected square matrix')
        if b1.shape != a1.shape:
            raise ValueError('a and b must have the same shape')
        return _geneig(a1, b1, left, right, overwrite_a, overwrite_b, homogeneous_eigvals)
    (geev, geev_lwork) = get_lapack_funcs(('geev', 'geev_lwork'), (a1,))
    (compute_vl, compute_vr) = (left, right)
    lwork = _compute_lwork(geev_lwork, a1.shape[0], compute_vl=compute_vl, compute_vr=compute_vr)
    if geev.typecode in 'cz':
        (w, vl, vr, info) = geev(a1, lwork=lwork, compute_vl=compute_vl, compute_vr=compute_vr, overwrite_a=overwrite_a)
        w = _make_eigvals(w, None, homogeneous_eigvals)
    else:
        (wr, wi, vl, vr, info) = geev(a1, lwork=lwork, compute_vl=compute_vl, compute_vr=compute_vr, overwrite_a=overwrite_a)
        t = {'f': 'F', 'd': 'D'}[wr.dtype.char]
        w = wr + _I * wi
        w = _make_eigvals(w, None, homogeneous_eigvals)
    _check_info(info, 'eig algorithm (geev)', positive='did not converge (only eigenvalues with order >= %d have converged)')
    only_real = numpy.all(w.imag == 0.0)
    if not (geev.typecode in 'cz' or only_real):
        t = w.dtype.char
        if left:
            vl = _make_complex_eigvecs(w, vl, t)
        if right:
            vr = _make_complex_eigvecs(w, vr, t)
    if not (left or right):
        return w
    if left:
        if right:
            return (w, vl, vr)
        return (w, vl)
    return (w, vr)

@_deprecate_positional_args(version='1.14.0')
def eigh(a, b=None, *, lower=True, eigvals_only=False, overwrite_a=False, overwrite_b=False, turbo=_NoValue, eigvals=_NoValue, type=1, check_finite=True, subset_by_index=None, subset_by_value=None, driver=None):
    if False:
        return 10
    '\n    Solve a standard or generalized eigenvalue problem for a complex\n    Hermitian or real symmetric matrix.\n\n    Find eigenvalues array ``w`` and optionally eigenvectors array ``v`` of\n    array ``a``, where ``b`` is positive definite such that for every\n    eigenvalue λ (i-th entry of w) and its eigenvector ``vi`` (i-th column of\n    ``v``) satisfies::\n\n                      a @ vi = λ * b @ vi\n        vi.conj().T @ a @ vi = λ\n        vi.conj().T @ b @ vi = 1\n\n    In the standard problem, ``b`` is assumed to be the identity matrix.\n\n    Parameters\n    ----------\n    a : (M, M) array_like\n        A complex Hermitian or real symmetric matrix whose eigenvalues and\n        eigenvectors will be computed.\n    b : (M, M) array_like, optional\n        A complex Hermitian or real symmetric definite positive matrix in.\n        If omitted, identity matrix is assumed.\n    lower : bool, optional\n        Whether the pertinent array data is taken from the lower or upper\n        triangle of ``a`` and, if applicable, ``b``. (Default: lower)\n    eigvals_only : bool, optional\n        Whether to calculate only eigenvalues and no eigenvectors.\n        (Default: both are calculated)\n    subset_by_index : iterable, optional\n        If provided, this two-element iterable defines the start and the end\n        indices of the desired eigenvalues (ascending order and 0-indexed).\n        To return only the second smallest to fifth smallest eigenvalues,\n        ``[1, 4]`` is used. ``[n-3, n-1]`` returns the largest three. Only\n        available with "evr", "evx", and "gvx" drivers. The entries are\n        directly converted to integers via ``int()``.\n    subset_by_value : iterable, optional\n        If provided, this two-element iterable defines the half-open interval\n        ``(a, b]`` that, if any, only the eigenvalues between these values\n        are returned. Only available with "evr", "evx", and "gvx" drivers. Use\n        ``np.inf`` for the unconstrained ends.\n    driver : str, optional\n        Defines which LAPACK driver should be used. Valid options are "ev",\n        "evd", "evr", "evx" for standard problems and "gv", "gvd", "gvx" for\n        generalized (where b is not None) problems. See the Notes section.\n        The default for standard problems is "evr". For generalized problems,\n        "gvd" is used for full set, and "gvx" for subset requested cases.\n    type : int, optional\n        For the generalized problems, this keyword specifies the problem type\n        to be solved for ``w`` and ``v`` (only takes 1, 2, 3 as possible\n        inputs)::\n\n            1 =>     a @ v = w @ b @ v\n            2 => a @ b @ v = w @ v\n            3 => b @ a @ v = w @ v\n\n        This keyword is ignored for standard problems.\n    overwrite_a : bool, optional\n        Whether to overwrite data in ``a`` (may improve performance). Default\n        is False.\n    overwrite_b : bool, optional\n        Whether to overwrite data in ``b`` (may improve performance). Default\n        is False.\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n    turbo : bool, optional, deprecated\n            .. deprecated:: 1.5.0\n                `eigh` keyword argument `turbo` is deprecated in favour of\n                ``driver=gvd`` keyword instead and will be removed in SciPy\n                1.14.0.\n    eigvals : tuple (lo, hi), optional, deprecated\n            .. deprecated:: 1.5.0\n                `eigh` keyword argument `eigvals` is deprecated in favour of\n                `subset_by_index` keyword instead and will be removed in SciPy\n                1.14.0.\n\n    Returns\n    -------\n    w : (N,) ndarray\n        The N (N<=M) selected eigenvalues, in ascending order, each\n        repeated according to its multiplicity.\n    v : (M, N) ndarray\n        The normalized eigenvector corresponding to the eigenvalue ``w[i]`` is\n        the column ``v[:,i]``. Only returned if ``eigvals_only=False``.\n\n    Raises\n    ------\n    LinAlgError\n        If eigenvalue computation does not converge, an error occurred, or\n        b matrix is not definite positive. Note that if input matrices are\n        not symmetric or Hermitian, no error will be reported but results will\n        be wrong.\n\n    See Also\n    --------\n    eigvalsh : eigenvalues of symmetric or Hermitian arrays\n    eig : eigenvalues and right eigenvectors for non-symmetric arrays\n    eigh_tridiagonal : eigenvalues and right eiegenvectors for\n        symmetric/Hermitian tridiagonal matrices\n\n    Notes\n    -----\n    This function does not check the input array for being Hermitian/symmetric\n    in order to allow for representing arrays with only their upper/lower\n    triangular parts. Also, note that even though not taken into account,\n    finiteness check applies to the whole array and unaffected by "lower"\n    keyword.\n\n    This function uses LAPACK drivers for computations in all possible keyword\n    combinations, prefixed with ``sy`` if arrays are real and ``he`` if\n    complex, e.g., a float array with "evr" driver is solved via\n    "syevr", complex arrays with "gvx" driver problem is solved via "hegvx"\n    etc.\n\n    As a brief summary, the slowest and the most robust driver is the\n    classical ``<sy/he>ev`` which uses symmetric QR. ``<sy/he>evr`` is seen as\n    the optimal choice for the most general cases. However, there are certain\n    occasions that ``<sy/he>evd`` computes faster at the expense of more\n    memory usage. ``<sy/he>evx``, while still being faster than ``<sy/he>ev``,\n    often performs worse than the rest except when very few eigenvalues are\n    requested for large arrays though there is still no performance guarantee.\n\n\n    For the generalized problem, normalization with respect to the given\n    type argument::\n\n            type 1 and 3 :      v.conj().T @ a @ v = w\n            type 2       : inv(v).conj().T @ a @ inv(v) = w\n\n            type 1 or 2  :      v.conj().T @ b @ v  = I\n            type 3       : v.conj().T @ inv(b) @ v  = I\n\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.linalg import eigh\n    >>> A = np.array([[6, 3, 1, 5], [3, 0, 5, 1], [1, 5, 6, 2], [5, 1, 2, 2]])\n    >>> w, v = eigh(A)\n    >>> np.allclose(A @ v - v @ np.diag(w), np.zeros((4, 4)))\n    True\n\n    Request only the eigenvalues\n\n    >>> w = eigh(A, eigvals_only=True)\n\n    Request eigenvalues that are less than 10.\n\n    >>> A = np.array([[34, -4, -10, -7, 2],\n    ...               [-4, 7, 2, 12, 0],\n    ...               [-10, 2, 44, 2, -19],\n    ...               [-7, 12, 2, 79, -34],\n    ...               [2, 0, -19, -34, 29]])\n    >>> eigh(A, eigvals_only=True, subset_by_value=[-np.inf, 10])\n    array([6.69199443e-07, 9.11938152e+00])\n\n    Request the second smallest eigenvalue and its eigenvector\n\n    >>> w, v = eigh(A, subset_by_index=[1, 1])\n    >>> w\n    array([9.11938152])\n    >>> v.shape  # only a single column is returned\n    (5, 1)\n\n    '
    if turbo is not _NoValue:
        warnings.warn("Keyword argument 'turbo' is deprecated in favour of 'driver=gvd' keyword instead and will be removed in SciPy 1.14.0.", DeprecationWarning, stacklevel=2)
    if eigvals is not _NoValue:
        warnings.warn("Keyword argument 'eigvals' is deprecated in favour of 'subset_by_index' keyword instead and will be removed in SciPy 1.14.0.", DeprecationWarning, stacklevel=2)
    uplo = 'L' if lower else 'U'
    _job = 'N' if eigvals_only else 'V'
    drv_str = [None, 'ev', 'evd', 'evr', 'evx', 'gv', 'gvd', 'gvx']
    if driver not in drv_str:
        raise ValueError('"{}" is unknown. Possible values are "None", "{}".'.format(driver, '", "'.join(drv_str[1:])))
    a1 = _asarray_validated(a, check_finite=check_finite)
    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square "a" matrix')
    overwrite_a = overwrite_a or _datacopied(a1, a)
    cplx = True if iscomplexobj(a1) else False
    n = a1.shape[0]
    drv_args = {'overwrite_a': overwrite_a}
    if b is not None:
        b1 = _asarray_validated(b, check_finite=check_finite)
        overwrite_b = overwrite_b or _datacopied(b1, b)
        if len(b1.shape) != 2 or b1.shape[0] != b1.shape[1]:
            raise ValueError('expected square "b" matrix')
        if b1.shape != a1.shape:
            raise ValueError('wrong b dimensions {}, should be {}'.format(b1.shape, a1.shape))
        if type not in [1, 2, 3]:
            raise ValueError('"type" keyword only accepts 1, 2, and 3.')
        cplx = True if iscomplexobj(b1) else cplx or False
        drv_args.update({'overwrite_b': overwrite_b, 'itype': type})
    subset_by_index = subset_by_index if eigvals in (None, _NoValue) else eigvals
    subset = subset_by_index is not None or subset_by_value is not None
    if subset_by_index and subset_by_value:
        raise ValueError('Either index or value subset can be requested.')
    if turbo not in (None, _NoValue) and b is not None:
        driver = 'gvx' if subset else 'gvd'
    if subset_by_index:
        (lo, hi) = (int(x) for x in subset_by_index)
        if not 0 <= lo <= hi < n:
            raise ValueError('Requested eigenvalue indices are not valid. Valid range is [0, {}] and start <= end, but start={}, end={} is given'.format(n - 1, lo, hi))
        drv_args.update({'range': 'I', 'il': lo + 1, 'iu': hi + 1})
    if subset_by_value:
        (lo, hi) = subset_by_value
        if not -inf <= lo < hi <= inf:
            raise ValueError('Requested eigenvalue bounds are not valid. Valid range is (-inf, inf) and low < high, but low={}, high={} is given'.format(lo, hi))
        drv_args.update({'range': 'V', 'vl': lo, 'vu': hi})
    pfx = 'he' if cplx else 'sy'
    if driver:
        if b is None and driver in ['gv', 'gvd', 'gvx']:
            raise ValueError('{} requires input b array to be supplied for generalized eigenvalue problems.'.format(driver))
        if b is not None and driver in ['ev', 'evd', 'evr', 'evx']:
            raise ValueError('"{}" does not accept input b array for standard eigenvalue problems.'.format(driver))
        if subset and driver in ['ev', 'evd', 'gv', 'gvd']:
            raise ValueError('"{}" cannot compute subsets of eigenvalues'.format(driver))
    else:
        driver = 'evr' if b is None else 'gvx' if subset else 'gvd'
    lwork_spec = {'syevd': ['lwork', 'liwork'], 'syevr': ['lwork', 'liwork'], 'heevd': ['lwork', 'liwork', 'lrwork'], 'heevr': ['lwork', 'lrwork', 'liwork']}
    if b is None:
        (drv, drvlw) = get_lapack_funcs((pfx + driver, pfx + driver + '_lwork'), [a1])
        clw_args = {'n': n, 'lower': lower}
        if driver == 'evd':
            clw_args.update({'compute_v': 0 if _job == 'N' else 1})
        lw = _compute_lwork(drvlw, **clw_args)
        if isinstance(lw, tuple):
            lwork_args = dict(zip(lwork_spec[pfx + driver], lw))
        else:
            lwork_args = {'lwork': lw}
        drv_args.update({'lower': lower, 'compute_v': 0 if _job == 'N' else 1})
        (w, v, *other_args, info) = drv(a=a1, **drv_args, **lwork_args)
    else:
        if driver == 'gvd':
            drv = get_lapack_funcs(pfx + 'gvd', [a1, b1])
            lwork_args = {}
        else:
            (drv, drvlw) = get_lapack_funcs((pfx + driver, pfx + driver + '_lwork'), [a1, b1])
            lw = _compute_lwork(drvlw, n, uplo=uplo)
            lwork_args = {'lwork': lw}
        drv_args.update({'uplo': uplo, 'jobz': _job})
        (w, v, *other_args, info) = drv(a=a1, b=b1, **drv_args, **lwork_args)
    w = w[:other_args[0]] if subset else w
    v = v[:, :other_args[0]] if subset and (not eigvals_only) else v
    if info == 0:
        if eigvals_only:
            return w
        else:
            return (w, v)
    elif info < -1:
        raise LinAlgError('Illegal value in argument {} of internal {}'.format(-info, drv.typecode + pfx + driver))
    elif info > n:
        raise LinAlgError('The leading minor of order {} of B is not positive definite. The factorization of B could not be completed and no eigenvalues or eigenvectors were computed.'.format(info - n))
    else:
        drv_err = {'ev': 'The algorithm failed to converge; {} off-diagonal elements of an intermediate tridiagonal form did not converge to zero.', 'evx': '{} eigenvectors failed to converge.', 'evd': 'The algorithm failed to compute an eigenvalue while working on the submatrix lying in rows and columns {0}/{1} through mod({0},{1}).', 'evr': 'Internal Error.'}
        if driver in ['ev', 'gv']:
            msg = drv_err['ev'].format(info)
        elif driver in ['evx', 'gvx']:
            msg = drv_err['evx'].format(info)
        elif driver in ['evd', 'gvd']:
            if eigvals_only:
                msg = drv_err['ev'].format(info)
            else:
                msg = drv_err['evd'].format(info, n + 1)
        else:
            msg = drv_err['evr']
        raise LinAlgError(msg)
_conv_dict = {0: 0, 1: 1, 2: 2, 'all': 0, 'value': 1, 'index': 2, 'a': 0, 'v': 1, 'i': 2}

def _check_select(select, select_range, max_ev, max_len):
    if False:
        print('Hello World!')
    'Check that select is valid, convert to Fortran style.'
    if isinstance(select, str):
        select = select.lower()
    try:
        select = _conv_dict[select]
    except KeyError as e:
        raise ValueError('invalid argument for select') from e
    (vl, vu) = (0.0, 1.0)
    il = iu = 1
    if select != 0:
        sr = asarray(select_range)
        if sr.ndim != 1 or sr.size != 2 or sr[1] < sr[0]:
            raise ValueError('select_range must be a 2-element array-like in nondecreasing order')
        if select == 1:
            (vl, vu) = sr
            if max_ev == 0:
                max_ev = max_len
        else:
            if sr.dtype.char.lower() not in 'hilqp':
                raise ValueError('when using select="i", select_range must contain integers, got dtype %s (%s)' % (sr.dtype, sr.dtype.char))
            (il, iu) = sr + 1
            if min(il, iu) < 1 or max(il, iu) > max_len:
                raise ValueError('select_range out of bounds')
            max_ev = iu - il + 1
    return (select, vl, vu, il, iu, max_ev)

def eig_banded(a_band, lower=False, eigvals_only=False, overwrite_a_band=False, select='a', select_range=None, max_ev=0, check_finite=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Solve real symmetric or complex Hermitian band matrix eigenvalue problem.\n\n    Find eigenvalues w and optionally right eigenvectors v of a::\n\n        a v[:,i] = w[i] v[:,i]\n        v.H v    = identity\n\n    The matrix a is stored in a_band either in lower diagonal or upper\n    diagonal ordered form:\n\n        a_band[u + i - j, j] == a[i,j]        (if upper form; i <= j)\n        a_band[    i - j, j] == a[i,j]        (if lower form; i >= j)\n\n    where u is the number of bands above the diagonal.\n\n    Example of a_band (shape of a is (6,6), u=2)::\n\n        upper form:\n        *   *   a02 a13 a24 a35\n        *   a01 a12 a23 a34 a45\n        a00 a11 a22 a33 a44 a55\n\n        lower form:\n        a00 a11 a22 a33 a44 a55\n        a10 a21 a32 a43 a54 *\n        a20 a31 a42 a53 *   *\n\n    Cells marked with * are not used.\n\n    Parameters\n    ----------\n    a_band : (u+1, M) array_like\n        The bands of the M by M matrix a.\n    lower : bool, optional\n        Is the matrix in the lower form. (Default is upper form)\n    eigvals_only : bool, optional\n        Compute only the eigenvalues and no eigenvectors.\n        (Default: calculate also eigenvectors)\n    overwrite_a_band : bool, optional\n        Discard data in a_band (may enhance performance)\n    select : {'a', 'v', 'i'}, optional\n        Which eigenvalues to calculate\n\n        ======  ========================================\n        select  calculated\n        ======  ========================================\n        'a'     All eigenvalues\n        'v'     Eigenvalues in the interval (min, max]\n        'i'     Eigenvalues with indices min <= i <= max\n        ======  ========================================\n    select_range : (min, max), optional\n        Range of selected eigenvalues\n    max_ev : int, optional\n        For select=='v', maximum number of eigenvalues expected.\n        For other values of select, has no meaning.\n\n        In doubt, leave this parameter untouched.\n\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    w : (M,) ndarray\n        The eigenvalues, in ascending order, each repeated according to its\n        multiplicity.\n    v : (M, M) float or complex ndarray\n        The normalized eigenvector corresponding to the eigenvalue w[i] is\n        the column v[:,i]. Only returned if ``eigvals_only=False``.\n\n    Raises\n    ------\n    LinAlgError\n        If eigenvalue computation does not converge.\n\n    See Also\n    --------\n    eigvals_banded : eigenvalues for symmetric/Hermitian band matrices\n    eig : eigenvalues and right eigenvectors of general arrays.\n    eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays\n    eigh_tridiagonal : eigenvalues and right eigenvectors for\n        symmetric/Hermitian tridiagonal matrices\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.linalg import eig_banded\n    >>> A = np.array([[1, 5, 2, 0], [5, 2, 5, 2], [2, 5, 3, 5], [0, 2, 5, 4]])\n    >>> Ab = np.array([[1, 2, 3, 4], [5, 5, 5, 0], [2, 2, 0, 0]])\n    >>> w, v = eig_banded(Ab, lower=True)\n    >>> np.allclose(A @ v - v @ np.diag(w), np.zeros((4, 4)))\n    True\n    >>> w = eig_banded(Ab, lower=True, eigvals_only=True)\n    >>> w\n    array([-4.26200532, -2.22987175,  3.95222349, 12.53965359])\n\n    Request only the eigenvalues between ``[-3, 4]``\n\n    >>> w, v = eig_banded(Ab, lower=True, select='v', select_range=[-3, 4])\n    >>> w\n    array([-2.22987175,  3.95222349])\n\n    "
    if eigvals_only or overwrite_a_band:
        a1 = _asarray_validated(a_band, check_finite=check_finite)
        overwrite_a_band = overwrite_a_band or _datacopied(a1, a_band)
    else:
        a1 = array(a_band)
        if issubclass(a1.dtype.type, inexact) and (not isfinite(a1).all()):
            raise ValueError('array must not contain infs or NaNs')
        overwrite_a_band = 1
    if len(a1.shape) != 2:
        raise ValueError('expected a 2-D array')
    (select, vl, vu, il, iu, max_ev) = _check_select(select, select_range, max_ev, a1.shape[1])
    del select_range
    if select == 0:
        if a1.dtype.char in 'GFD':
            internal_name = 'hbevd'
        else:
            internal_name = 'sbevd'
        (bevd,) = get_lapack_funcs((internal_name,), (a1,))
        (w, v, info) = bevd(a1, compute_v=not eigvals_only, lower=lower, overwrite_ab=overwrite_a_band)
    else:
        if eigvals_only:
            max_ev = 1
        if a1.dtype.char in 'fF':
            (lamch,) = get_lapack_funcs(('lamch',), (array(0, dtype='f'),))
        else:
            (lamch,) = get_lapack_funcs(('lamch',), (array(0, dtype='d'),))
        abstol = 2 * lamch('s')
        if a1.dtype.char in 'GFD':
            internal_name = 'hbevx'
        else:
            internal_name = 'sbevx'
        (bevx,) = get_lapack_funcs((internal_name,), (a1,))
        (w, v, m, ifail, info) = bevx(a1, vl, vu, il, iu, compute_v=not eigvals_only, mmax=max_ev, range=select, lower=lower, overwrite_ab=overwrite_a_band, abstol=abstol)
        w = w[:m]
        if not eigvals_only:
            v = v[:, :m]
    _check_info(info, internal_name)
    if eigvals_only:
        return w
    return (w, v)

def eigvals(a, b=None, overwrite_a=False, check_finite=True, homogeneous_eigvals=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute eigenvalues from an ordinary or generalized eigenvalue problem.\n\n    Find eigenvalues of a general matrix::\n\n        a   vr[:,i] = w[i]        b   vr[:,i]\n\n    Parameters\n    ----------\n    a : (M, M) array_like\n        A complex or real matrix whose eigenvalues and eigenvectors\n        will be computed.\n    b : (M, M) array_like, optional\n        Right-hand side matrix in a generalized eigenvalue problem.\n        If omitted, identity matrix is assumed.\n    overwrite_a : bool, optional\n        Whether to overwrite data in a (may improve performance)\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities\n        or NaNs.\n    homogeneous_eigvals : bool, optional\n        If True, return the eigenvalues in homogeneous coordinates.\n        In this case ``w`` is a (2, M) array so that::\n\n            w[1,i] a vr[:,i] = w[0,i] b vr[:,i]\n\n        Default is False.\n\n    Returns\n    -------\n    w : (M,) or (2, M) double or complex ndarray\n        The eigenvalues, each repeated according to its multiplicity\n        but not in any specific order. The shape is (M,) unless\n        ``homogeneous_eigvals=True``.\n\n    Raises\n    ------\n    LinAlgError\n        If eigenvalue computation does not converge\n\n    See Also\n    --------\n    eig : eigenvalues and right eigenvectors of general arrays.\n    eigvalsh : eigenvalues of symmetric or Hermitian arrays\n    eigvals_banded : eigenvalues for symmetric/Hermitian band matrices\n    eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal\n        matrices\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy import linalg\n    >>> a = np.array([[0., -1.], [1., 0.]])\n    >>> linalg.eigvals(a)\n    array([0.+1.j, 0.-1.j])\n\n    >>> b = np.array([[0., 1.], [1., 1.]])\n    >>> linalg.eigvals(a, b)\n    array([ 1.+0.j, -1.+0.j])\n\n    >>> a = np.array([[3., 0., 0.], [0., 8., 0.], [0., 0., 7.]])\n    >>> linalg.eigvals(a, homogeneous_eigvals=True)\n    array([[3.+0.j, 8.+0.j, 7.+0.j],\n           [1.+0.j, 1.+0.j, 1.+0.j]])\n\n    '
    return eig(a, b=b, left=0, right=0, overwrite_a=overwrite_a, check_finite=check_finite, homogeneous_eigvals=homogeneous_eigvals)

@_deprecate_positional_args(version='1.14.0')
def eigvalsh(a, b=None, *, lower=True, overwrite_a=False, overwrite_b=False, turbo=_NoValue, eigvals=_NoValue, type=1, check_finite=True, subset_by_index=None, subset_by_value=None, driver=None):
    if False:
        while True:
            i = 10
    '\n    Solves a standard or generalized eigenvalue problem for a complex\n    Hermitian or real symmetric matrix.\n\n    Find eigenvalues array ``w`` of array ``a``, where ``b`` is positive\n    definite such that for every eigenvalue λ (i-th entry of w) and its\n    eigenvector vi (i-th column of v) satisfies::\n\n                      a @ vi = λ * b @ vi\n        vi.conj().T @ a @ vi = λ\n        vi.conj().T @ b @ vi = 1\n\n    In the standard problem, b is assumed to be the identity matrix.\n\n    Parameters\n    ----------\n    a : (M, M) array_like\n        A complex Hermitian or real symmetric matrix whose eigenvalues will\n        be computed.\n    b : (M, M) array_like, optional\n        A complex Hermitian or real symmetric definite positive matrix in.\n        If omitted, identity matrix is assumed.\n    lower : bool, optional\n        Whether the pertinent array data is taken from the lower or upper\n        triangle of ``a`` and, if applicable, ``b``. (Default: lower)\n    overwrite_a : bool, optional\n        Whether to overwrite data in ``a`` (may improve performance). Default\n        is False.\n    overwrite_b : bool, optional\n        Whether to overwrite data in ``b`` (may improve performance). Default\n        is False.\n    type : int, optional\n        For the generalized problems, this keyword specifies the problem type\n        to be solved for ``w`` and ``v`` (only takes 1, 2, 3 as possible\n        inputs)::\n\n            1 =>     a @ v = w @ b @ v\n            2 => a @ b @ v = w @ v\n            3 => b @ a @ v = w @ v\n\n        This keyword is ignored for standard problems.\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n    subset_by_index : iterable, optional\n        If provided, this two-element iterable defines the start and the end\n        indices of the desired eigenvalues (ascending order and 0-indexed).\n        To return only the second smallest to fifth smallest eigenvalues,\n        ``[1, 4]`` is used. ``[n-3, n-1]`` returns the largest three. Only\n        available with "evr", "evx", and "gvx" drivers. The entries are\n        directly converted to integers via ``int()``.\n    subset_by_value : iterable, optional\n        If provided, this two-element iterable defines the half-open interval\n        ``(a, b]`` that, if any, only the eigenvalues between these values\n        are returned. Only available with "evr", "evx", and "gvx" drivers. Use\n        ``np.inf`` for the unconstrained ends.\n    driver : str, optional\n        Defines which LAPACK driver should be used. Valid options are "ev",\n        "evd", "evr", "evx" for standard problems and "gv", "gvd", "gvx" for\n        generalized (where b is not None) problems. See the Notes section of\n        `scipy.linalg.eigh`.\n    turbo : bool, optional, deprecated\n        .. deprecated:: 1.5.0\n            \'eigvalsh\' keyword argument `turbo` is deprecated in favor of\n            ``driver=gvd`` option and will be removed in SciPy 1.14.0.\n\n    eigvals : tuple (lo, hi), optional\n        .. deprecated:: 1.5.0\n            \'eigvalsh\' keyword argument `eigvals` is deprecated in favor of\n            `subset_by_index` option and will be removed in SciPy 1.14.0.\n\n    Returns\n    -------\n    w : (N,) ndarray\n        The N (N<=M) selected eigenvalues, in ascending order, each\n        repeated according to its multiplicity.\n\n    Raises\n    ------\n    LinAlgError\n        If eigenvalue computation does not converge, an error occurred, or\n        b matrix is not definite positive. Note that if input matrices are\n        not symmetric or Hermitian, no error will be reported but results will\n        be wrong.\n\n    See Also\n    --------\n    eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays\n    eigvals : eigenvalues of general arrays\n    eigvals_banded : eigenvalues for symmetric/Hermitian band matrices\n    eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal\n        matrices\n\n    Notes\n    -----\n    This function does not check the input array for being Hermitian/symmetric\n    in order to allow for representing arrays with only their upper/lower\n    triangular parts.\n\n    This function serves as a one-liner shorthand for `scipy.linalg.eigh` with\n    the option ``eigvals_only=True`` to get the eigenvalues and not the\n    eigenvectors. Here it is kept as a legacy convenience. It might be\n    beneficial to use the main function to have full control and to be a bit\n    more pythonic.\n\n    Examples\n    --------\n    For more examples see `scipy.linalg.eigh`.\n\n    >>> import numpy as np\n    >>> from scipy.linalg import eigvalsh\n    >>> A = np.array([[6, 3, 1, 5], [3, 0, 5, 1], [1, 5, 6, 2], [5, 1, 2, 2]])\n    >>> w = eigvalsh(A)\n    >>> w\n    array([-3.74637491, -0.76263923,  6.08502336, 12.42399079])\n\n    '
    return eigh(a, b=b, lower=lower, eigvals_only=True, overwrite_a=overwrite_a, overwrite_b=overwrite_b, turbo=turbo, eigvals=eigvals, type=type, check_finite=check_finite, subset_by_index=subset_by_index, subset_by_value=subset_by_value, driver=driver)

def eigvals_banded(a_band, lower=False, overwrite_a_band=False, select='a', select_range=None, check_finite=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Solve real symmetric or complex Hermitian band matrix eigenvalue problem.\n\n    Find eigenvalues w of a::\n\n        a v[:,i] = w[i] v[:,i]\n        v.H v    = identity\n\n    The matrix a is stored in a_band either in lower diagonal or upper\n    diagonal ordered form:\n\n        a_band[u + i - j, j] == a[i,j]        (if upper form; i <= j)\n        a_band[    i - j, j] == a[i,j]        (if lower form; i >= j)\n\n    where u is the number of bands above the diagonal.\n\n    Example of a_band (shape of a is (6,6), u=2)::\n\n        upper form:\n        *   *   a02 a13 a24 a35\n        *   a01 a12 a23 a34 a45\n        a00 a11 a22 a33 a44 a55\n\n        lower form:\n        a00 a11 a22 a33 a44 a55\n        a10 a21 a32 a43 a54 *\n        a20 a31 a42 a53 *   *\n\n    Cells marked with * are not used.\n\n    Parameters\n    ----------\n    a_band : (u+1, M) array_like\n        The bands of the M by M matrix a.\n    lower : bool, optional\n        Is the matrix in the lower form. (Default is upper form)\n    overwrite_a_band : bool, optional\n        Discard data in a_band (may enhance performance)\n    select : {'a', 'v', 'i'}, optional\n        Which eigenvalues to calculate\n\n        ======  ========================================\n        select  calculated\n        ======  ========================================\n        'a'     All eigenvalues\n        'v'     Eigenvalues in the interval (min, max]\n        'i'     Eigenvalues with indices min <= i <= max\n        ======  ========================================\n    select_range : (min, max), optional\n        Range of selected eigenvalues\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    w : (M,) ndarray\n        The eigenvalues, in ascending order, each repeated according to its\n        multiplicity.\n\n    Raises\n    ------\n    LinAlgError\n        If eigenvalue computation does not converge.\n\n    See Also\n    --------\n    eig_banded : eigenvalues and right eigenvectors for symmetric/Hermitian\n        band matrices\n    eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal\n        matrices\n    eigvals : eigenvalues of general arrays\n    eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays\n    eig : eigenvalues and right eigenvectors for non-symmetric arrays\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.linalg import eigvals_banded\n    >>> A = np.array([[1, 5, 2, 0], [5, 2, 5, 2], [2, 5, 3, 5], [0, 2, 5, 4]])\n    >>> Ab = np.array([[1, 2, 3, 4], [5, 5, 5, 0], [2, 2, 0, 0]])\n    >>> w = eigvals_banded(Ab, lower=True)\n    >>> w\n    array([-4.26200532, -2.22987175,  3.95222349, 12.53965359])\n    "
    return eig_banded(a_band, lower=lower, eigvals_only=1, overwrite_a_band=overwrite_a_band, select=select, select_range=select_range, check_finite=check_finite)

def eigvalsh_tridiagonal(d, e, select='a', select_range=None, check_finite=True, tol=0.0, lapack_driver='auto'):
    if False:
        while True:
            i = 10
    "\n    Solve eigenvalue problem for a real symmetric tridiagonal matrix.\n\n    Find eigenvalues `w` of ``a``::\n\n        a v[:,i] = w[i] v[:,i]\n        v.H v    = identity\n\n    For a real symmetric matrix ``a`` with diagonal elements `d` and\n    off-diagonal elements `e`.\n\n    Parameters\n    ----------\n    d : ndarray, shape (ndim,)\n        The diagonal elements of the array.\n    e : ndarray, shape (ndim-1,)\n        The off-diagonal elements of the array.\n    select : {'a', 'v', 'i'}, optional\n        Which eigenvalues to calculate\n\n        ======  ========================================\n        select  calculated\n        ======  ========================================\n        'a'     All eigenvalues\n        'v'     Eigenvalues in the interval (min, max]\n        'i'     Eigenvalues with indices min <= i <= max\n        ======  ========================================\n    select_range : (min, max), optional\n        Range of selected eigenvalues\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n    tol : float\n        The absolute tolerance to which each eigenvalue is required\n        (only used when ``lapack_driver='stebz'``).\n        An eigenvalue (or cluster) is considered to have converged if it\n        lies in an interval of this width. If <= 0. (default),\n        the value ``eps*|a|`` is used where eps is the machine precision,\n        and ``|a|`` is the 1-norm of the matrix ``a``.\n    lapack_driver : str\n        LAPACK function to use, can be 'auto', 'stemr', 'stebz',  'sterf',\n        or 'stev'. When 'auto' (default), it will use 'stemr' if ``select='a'``\n        and 'stebz' otherwise. 'sterf' and 'stev' can only be used when\n        ``select='a'``.\n\n    Returns\n    -------\n    w : (M,) ndarray\n        The eigenvalues, in ascending order, each repeated according to its\n        multiplicity.\n\n    Raises\n    ------\n    LinAlgError\n        If eigenvalue computation does not converge.\n\n    See Also\n    --------\n    eigh_tridiagonal : eigenvalues and right eiegenvectors for\n        symmetric/Hermitian tridiagonal matrices\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.linalg import eigvalsh_tridiagonal, eigvalsh\n    >>> d = 3*np.ones(4)\n    >>> e = -1*np.ones(3)\n    >>> w = eigvalsh_tridiagonal(d, e)\n    >>> A = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)\n    >>> w2 = eigvalsh(A)  # Verify with other eigenvalue routines\n    >>> np.allclose(w - w2, np.zeros(4))\n    True\n    "
    return eigh_tridiagonal(d, e, eigvals_only=True, select=select, select_range=select_range, check_finite=check_finite, tol=tol, lapack_driver=lapack_driver)

def eigh_tridiagonal(d, e, eigvals_only=False, select='a', select_range=None, check_finite=True, tol=0.0, lapack_driver='auto'):
    if False:
        while True:
            i = 10
    "\n    Solve eigenvalue problem for a real symmetric tridiagonal matrix.\n\n    Find eigenvalues `w` and optionally right eigenvectors `v` of ``a``::\n\n        a v[:,i] = w[i] v[:,i]\n        v.H v    = identity\n\n    For a real symmetric matrix ``a`` with diagonal elements `d` and\n    off-diagonal elements `e`.\n\n    Parameters\n    ----------\n    d : ndarray, shape (ndim,)\n        The diagonal elements of the array.\n    e : ndarray, shape (ndim-1,)\n        The off-diagonal elements of the array.\n    eigvals_only : bool, optional\n        Compute only the eigenvalues and no eigenvectors.\n        (Default: calculate also eigenvectors)\n    select : {'a', 'v', 'i'}, optional\n        Which eigenvalues to calculate\n\n        ======  ========================================\n        select  calculated\n        ======  ========================================\n        'a'     All eigenvalues\n        'v'     Eigenvalues in the interval (min, max]\n        'i'     Eigenvalues with indices min <= i <= max\n        ======  ========================================\n    select_range : (min, max), optional\n        Range of selected eigenvalues\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n    tol : float\n        The absolute tolerance to which each eigenvalue is required\n        (only used when 'stebz' is the `lapack_driver`).\n        An eigenvalue (or cluster) is considered to have converged if it\n        lies in an interval of this width. If <= 0. (default),\n        the value ``eps*|a|`` is used where eps is the machine precision,\n        and ``|a|`` is the 1-norm of the matrix ``a``.\n    lapack_driver : str\n        LAPACK function to use, can be 'auto', 'stemr', 'stebz', 'sterf',\n        or 'stev'. When 'auto' (default), it will use 'stemr' if ``select='a'``\n        and 'stebz' otherwise. When 'stebz' is used to find the eigenvalues and\n        ``eigvals_only=False``, then a second LAPACK call (to ``?STEIN``) is\n        used to find the corresponding eigenvectors. 'sterf' can only be\n        used when ``eigvals_only=True`` and ``select='a'``. 'stev' can only\n        be used when ``select='a'``.\n\n    Returns\n    -------\n    w : (M,) ndarray\n        The eigenvalues, in ascending order, each repeated according to its\n        multiplicity.\n    v : (M, M) ndarray\n        The normalized eigenvector corresponding to the eigenvalue ``w[i]`` is\n        the column ``v[:,i]``. Only returned if ``eigvals_only=False``.\n\n    Raises\n    ------\n    LinAlgError\n        If eigenvalue computation does not converge.\n\n    See Also\n    --------\n    eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal\n        matrices\n    eig : eigenvalues and right eigenvectors for non-symmetric arrays\n    eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays\n    eig_banded : eigenvalues and right eigenvectors for symmetric/Hermitian\n        band matrices\n\n    Notes\n    -----\n    This function makes use of LAPACK ``S/DSTEMR`` routines.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.linalg import eigh_tridiagonal\n    >>> d = 3*np.ones(4)\n    >>> e = -1*np.ones(3)\n    >>> w, v = eigh_tridiagonal(d, e)\n    >>> A = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)\n    >>> np.allclose(A @ v - v @ np.diag(w), np.zeros((4, 4)))\n    True\n    "
    d = _asarray_validated(d, check_finite=check_finite)
    e = _asarray_validated(e, check_finite=check_finite)
    for check in (d, e):
        if check.ndim != 1:
            raise ValueError('expected a 1-D array')
        if check.dtype.char in 'GFD':
            raise TypeError('Only real arrays currently supported')
    if d.size != e.size + 1:
        raise ValueError('d (%s) must have one more element than e (%s)' % (d.size, e.size))
    (select, vl, vu, il, iu, _) = _check_select(select, select_range, 0, d.size)
    if not isinstance(lapack_driver, str):
        raise TypeError('lapack_driver must be str')
    drivers = ('auto', 'stemr', 'sterf', 'stebz', 'stev')
    if lapack_driver not in drivers:
        raise ValueError('lapack_driver must be one of %s, got %s' % (drivers, lapack_driver))
    if lapack_driver == 'auto':
        lapack_driver = 'stemr' if select == 0 else 'stebz'
    (func,) = get_lapack_funcs((lapack_driver,), (d, e))
    compute_v = not eigvals_only
    if lapack_driver == 'sterf':
        if select != 0:
            raise ValueError('sterf can only be used when select == "a"')
        if not eigvals_only:
            raise ValueError('sterf can only be used when eigvals_only is True')
        (w, info) = func(d, e)
        m = len(w)
    elif lapack_driver == 'stev':
        if select != 0:
            raise ValueError('stev can only be used when select == "a"')
        (w, v, info) = func(d, e, compute_v=compute_v)
        m = len(w)
    elif lapack_driver == 'stebz':
        tol = float(tol)
        internal_name = 'stebz'
        (stebz,) = get_lapack_funcs((internal_name,), (d, e))
        order = 'E' if eigvals_only else 'B'
        (m, w, iblock, isplit, info) = stebz(d, e, select, vl, vu, il, iu, tol, order)
    else:
        e_ = empty(e.size + 1, e.dtype)
        e_[:-1] = e
        (stemr_lwork,) = get_lapack_funcs(('stemr_lwork',), (d, e))
        (lwork, liwork, info) = stemr_lwork(d, e_, select, vl, vu, il, iu, compute_v=compute_v)
        _check_info(info, 'stemr_lwork')
        (m, w, v, info) = func(d, e_, select, vl, vu, il, iu, compute_v=compute_v, lwork=lwork, liwork=liwork)
    _check_info(info, lapack_driver + ' (eigh_tridiagonal)')
    w = w[:m]
    if eigvals_only:
        return w
    else:
        if lapack_driver == 'stebz':
            (func,) = get_lapack_funcs(('stein',), (d, e))
            (v, info) = func(d, e, w, iblock, isplit)
            _check_info(info, 'stein (eigh_tridiagonal)', positive='%d eigenvectors failed to converge')
            order = argsort(w)
            (w, v) = (w[order], v[:, order])
        else:
            v = v[:, :m]
        return (w, v)

def _check_info(info, driver, positive='did not converge (LAPACK info=%d)'):
    if False:
        print('Hello World!')
    'Check info return value.'
    if info < 0:
        raise ValueError('illegal value in argument %d of internal %s' % (-info, driver))
    if info > 0 and positive:
        raise LinAlgError(('%s ' + positive) % (driver, info))

def hessenberg(a, calc_q=False, overwrite_a=False, check_finite=True):
    if False:
        print('Hello World!')
    '\n    Compute Hessenberg form of a matrix.\n\n    The Hessenberg decomposition is::\n\n        A = Q H Q^H\n\n    where `Q` is unitary/orthogonal and `H` has only zero elements below\n    the first sub-diagonal.\n\n    Parameters\n    ----------\n    a : (M, M) array_like\n        Matrix to bring into Hessenberg form.\n    calc_q : bool, optional\n        Whether to compute the transformation matrix.  Default is False.\n    overwrite_a : bool, optional\n        Whether to overwrite `a`; may improve performance.\n        Default is False.\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    H : (M, M) ndarray\n        Hessenberg form of `a`.\n    Q : (M, M) ndarray\n        Unitary/orthogonal similarity transformation matrix ``A = Q H Q^H``.\n        Only returned if ``calc_q=True``.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.linalg import hessenberg\n    >>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])\n    >>> H, Q = hessenberg(A, calc_q=True)\n    >>> H\n    array([[  2.        , -11.65843866,   1.42005301,   0.25349066],\n           [ -9.94987437,  14.53535354,  -5.31022304,   2.43081618],\n           [  0.        ,  -1.83299243,   0.38969961,  -0.51527034],\n           [  0.        ,   0.        ,  -3.83189513,   1.07494686]])\n    >>> np.allclose(Q @ H @ Q.conj().T - A, np.zeros((4, 4)))\n    True\n    '
    a1 = _asarray_validated(a, check_finite=check_finite)
    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square matrix')
    overwrite_a = overwrite_a or _datacopied(a1, a)
    if a1.shape[0] <= 2:
        if calc_q:
            return (a1, eye(a1.shape[0]))
        return a1
    (gehrd, gebal, gehrd_lwork) = get_lapack_funcs(('gehrd', 'gebal', 'gehrd_lwork'), (a1,))
    (ba, lo, hi, pivscale, info) = gebal(a1, permute=0, overwrite_a=overwrite_a)
    _check_info(info, 'gebal (hessenberg)', positive=False)
    n = len(a1)
    lwork = _compute_lwork(gehrd_lwork, ba.shape[0], lo=lo, hi=hi)
    (hq, tau, info) = gehrd(ba, lo=lo, hi=hi, lwork=lwork, overwrite_a=1)
    _check_info(info, 'gehrd (hessenberg)', positive=False)
    h = numpy.triu(hq, -1)
    if not calc_q:
        return h
    (orghr, orghr_lwork) = get_lapack_funcs(('orghr', 'orghr_lwork'), (a1,))
    lwork = _compute_lwork(orghr_lwork, n, lo=lo, hi=hi)
    (q, info) = orghr(a=hq, tau=tau, lo=lo, hi=hi, lwork=lwork, overwrite_a=1)
    _check_info(info, 'orghr (hessenberg)', positive=False)
    return (h, q)

def cdf2rdf(w, v):
    if False:
        return 10
    '\n    Converts complex eigenvalues ``w`` and eigenvectors ``v`` to real\n    eigenvalues in a block diagonal form ``wr`` and the associated real\n    eigenvectors ``vr``, such that::\n\n        vr @ wr = X @ vr\n\n    continues to hold, where ``X`` is the original array for which ``w`` and\n    ``v`` are the eigenvalues and eigenvectors.\n\n    .. versionadded:: 1.1.0\n\n    Parameters\n    ----------\n    w : (..., M) array_like\n        Complex or real eigenvalues, an array or stack of arrays\n\n        Conjugate pairs must not be interleaved, else the wrong result\n        will be produced. So ``[1+1j, 1, 1-1j]`` will give a correct result,\n        but ``[1+1j, 2+1j, 1-1j, 2-1j]`` will not.\n\n    v : (..., M, M) array_like\n        Complex or real eigenvectors, a square array or stack of square arrays.\n\n    Returns\n    -------\n    wr : (..., M, M) ndarray\n        Real diagonal block form of eigenvalues\n    vr : (..., M, M) ndarray\n        Real eigenvectors associated with ``wr``\n\n    See Also\n    --------\n    eig : Eigenvalues and right eigenvectors for non-symmetric arrays\n    rsf2csf : Convert real Schur form to complex Schur form\n\n    Notes\n    -----\n    ``w``, ``v`` must be the eigenstructure for some *real* matrix ``X``.\n    For example, obtained by ``w, v = scipy.linalg.eig(X)`` or\n    ``w, v = numpy.linalg.eig(X)`` in which case ``X`` can also represent\n    stacked arrays.\n\n    .. versionadded:: 1.1.0\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> X = np.array([[1, 2, 3], [0, 4, 5], [0, -5, 4]])\n    >>> X\n    array([[ 1,  2,  3],\n           [ 0,  4,  5],\n           [ 0, -5,  4]])\n\n    >>> from scipy import linalg\n    >>> w, v = linalg.eig(X)\n    >>> w\n    array([ 1.+0.j,  4.+5.j,  4.-5.j])\n    >>> v\n    array([[ 1.00000+0.j     , -0.01906-0.40016j, -0.01906+0.40016j],\n           [ 0.00000+0.j     ,  0.00000-0.64788j,  0.00000+0.64788j],\n           [ 0.00000+0.j     ,  0.64788+0.j     ,  0.64788-0.j     ]])\n\n    >>> wr, vr = linalg.cdf2rdf(w, v)\n    >>> wr\n    array([[ 1.,  0.,  0.],\n           [ 0.,  4.,  5.],\n           [ 0., -5.,  4.]])\n    >>> vr\n    array([[ 1.     ,  0.40016, -0.01906],\n           [ 0.     ,  0.64788,  0.     ],\n           [ 0.     ,  0.     ,  0.64788]])\n\n    >>> vr @ wr\n    array([[ 1.     ,  1.69593,  1.9246 ],\n           [ 0.     ,  2.59153,  3.23942],\n           [ 0.     , -3.23942,  2.59153]])\n    >>> X @ vr\n    array([[ 1.     ,  1.69593,  1.9246 ],\n           [ 0.     ,  2.59153,  3.23942],\n           [ 0.     , -3.23942,  2.59153]])\n    '
    (w, v) = (_asarray_validated(w), _asarray_validated(v))
    if w.ndim < 1:
        raise ValueError('expected w to be at least 1D')
    if v.ndim < 2:
        raise ValueError('expected v to be at least 2D')
    if v.ndim != w.ndim + 1:
        raise ValueError('expected eigenvectors array to have exactly one dimension more than eigenvalues array')
    n = w.shape[-1]
    M = w.shape[:-1]
    if v.shape[-2] != v.shape[-1]:
        raise ValueError('expected v to be a square matrix or stacked square matrices: v.shape[-2] = v.shape[-1]')
    if v.shape[-1] != n:
        raise ValueError('expected the same number of eigenvalues as eigenvectors')
    complex_mask = iscomplex(w)
    n_complex = complex_mask.sum(axis=-1)
    if not (n_complex % 2 == 0).all():
        raise ValueError('expected complex-conjugate pairs of eigenvalues')
    idx = nonzero(complex_mask)
    idx_stack = idx[:-1]
    idx_elem = idx[-1]
    j = idx_elem[0::2]
    k = idx_elem[1::2]
    stack_ind = ()
    for i in idx_stack:
        assert (i[0::2] == i[1::2]).all(), 'Conjugate pair spanned different arrays!'
        stack_ind += (i[0::2],)
    wr = zeros(M + (n, n), dtype=w.real.dtype)
    di = range(n)
    wr[..., di, di] = w.real
    wr[stack_ind + (j, k)] = w[stack_ind + (j,)].imag
    wr[stack_ind + (k, j)] = w[stack_ind + (k,)].imag
    u = zeros(M + (n, n), dtype=numpy.cdouble)
    u[..., di, di] = 1.0
    u[stack_ind + (j, j)] = 0.5j
    u[stack_ind + (j, k)] = 0.5
    u[stack_ind + (k, j)] = -0.5j
    u[stack_ind + (k, k)] = 0.5
    vr = einsum('...ij,...jk->...ik', v, u).real
    return (wr, vr)