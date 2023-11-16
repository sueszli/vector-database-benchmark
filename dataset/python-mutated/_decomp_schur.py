"""Schur decomposition functions."""
import numpy
from numpy import asarray_chkfinite, single, asarray, array
from numpy.linalg import norm
from ._misc import LinAlgError, _datacopied
from .lapack import get_lapack_funcs
from ._decomp import eigvals
__all__ = ['schur', 'rsf2csf']
_double_precision = ['i', 'l', 'd']

def schur(a, output='real', lwork=None, overwrite_a=False, sort=None, check_finite=True):
    if False:
        i = 10
        return i + 15
    "\n    Compute Schur decomposition of a matrix.\n\n    The Schur decomposition is::\n\n        A = Z T Z^H\n\n    where Z is unitary and T is either upper-triangular, or for real\n    Schur decomposition (output='real'), quasi-upper triangular. In\n    the quasi-triangular form, 2x2 blocks describing complex-valued\n    eigenvalue pairs may extrude from the diagonal.\n\n    Parameters\n    ----------\n    a : (M, M) array_like\n        Matrix to decompose\n    output : {'real', 'complex'}, optional\n        Construct the real or complex Schur decomposition (for real matrices).\n    lwork : int, optional\n        Work array size. If None or -1, it is automatically computed.\n    overwrite_a : bool, optional\n        Whether to overwrite data in a (may improve performance).\n    sort : {None, callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional\n        Specifies whether the upper eigenvalues should be sorted. A callable\n        may be passed that, given a eigenvalue, returns a boolean denoting\n        whether the eigenvalue should be sorted to the top-left (True).\n        Alternatively, string parameters may be used::\n\n            'lhp'   Left-hand plane (x.real < 0.0)\n            'rhp'   Right-hand plane (x.real > 0.0)\n            'iuc'   Inside the unit circle (x*x.conjugate() <= 1.0)\n            'ouc'   Outside the unit circle (x*x.conjugate() > 1.0)\n\n        Defaults to None (no sorting).\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    T : (M, M) ndarray\n        Schur form of A. It is real-valued for the real Schur decomposition.\n    Z : (M, M) ndarray\n        An unitary Schur transformation matrix for A.\n        It is real-valued for the real Schur decomposition.\n    sdim : int\n        If and only if sorting was requested, a third return value will\n        contain the number of eigenvalues satisfying the sort condition.\n\n    Raises\n    ------\n    LinAlgError\n        Error raised under three conditions:\n\n        1. The algorithm failed due to a failure of the QR algorithm to\n           compute all eigenvalues.\n        2. If eigenvalue sorting was requested, the eigenvalues could not be\n           reordered due to a failure to separate eigenvalues, usually because\n           of poor conditioning.\n        3. If eigenvalue sorting was requested, roundoff errors caused the\n           leading eigenvalues to no longer satisfy the sorting condition.\n\n    See Also\n    --------\n    rsf2csf : Convert real Schur form to complex Schur form\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.linalg import schur, eigvals\n    >>> A = np.array([[0, 2, 2], [0, 1, 2], [1, 0, 1]])\n    >>> T, Z = schur(A)\n    >>> T\n    array([[ 2.65896708,  1.42440458, -1.92933439],\n           [ 0.        , -0.32948354, -0.49063704],\n           [ 0.        ,  1.31178921, -0.32948354]])\n    >>> Z\n    array([[0.72711591, -0.60156188, 0.33079564],\n           [0.52839428, 0.79801892, 0.28976765],\n           [0.43829436, 0.03590414, -0.89811411]])\n\n    >>> T2, Z2 = schur(A, output='complex')\n    >>> T2\n    array([[ 2.65896708, -1.22839825+1.32378589j,  0.42590089+1.51937378j],\n           [ 0.        , -0.32948354+0.80225456j, -0.59877807+0.56192146j],\n           [ 0.        ,  0.                    , -0.32948354-0.80225456j]])\n    >>> eigvals(T2)\n    array([2.65896708, -0.32948354+0.80225456j, -0.32948354-0.80225456j])\n\n    An arbitrary custom eig-sorting condition, having positive imaginary part,\n    which is satisfied by only one eigenvalue\n\n    >>> T3, Z3, sdim = schur(A, output='complex', sort=lambda x: x.imag > 0)\n    >>> sdim\n    1\n\n    "
    if output not in ['real', 'complex', 'r', 'c']:
        raise ValueError("argument must be 'real', or 'complex'")
    if check_finite:
        a1 = asarray_chkfinite(a)
    else:
        a1 = asarray(a)
    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square matrix')
    typ = a1.dtype.char
    if output in ['complex', 'c'] and typ not in ['F', 'D']:
        if typ in _double_precision:
            a1 = a1.astype('D')
            typ = 'D'
        else:
            a1 = a1.astype('F')
            typ = 'F'
    overwrite_a = overwrite_a or _datacopied(a1, a)
    (gees,) = get_lapack_funcs(('gees',), (a1,))
    if lwork is None or lwork == -1:
        result = gees(lambda x: None, a1, lwork=-1)
        lwork = result[-2][0].real.astype(numpy.int_)
    if sort is None:
        sort_t = 0

        def sfunction(x):
            if False:
                while True:
                    i = 10
            return None
    else:
        sort_t = 1
        if callable(sort):
            sfunction = sort
        elif sort == 'lhp':

            def sfunction(x):
                if False:
                    while True:
                        i = 10
                return x.real < 0.0
        elif sort == 'rhp':

            def sfunction(x):
                if False:
                    i = 10
                    return i + 15
                return x.real >= 0.0
        elif sort == 'iuc':

            def sfunction(x):
                if False:
                    while True:
                        i = 10
                return abs(x) <= 1.0
        elif sort == 'ouc':

            def sfunction(x):
                if False:
                    while True:
                        i = 10
                return abs(x) > 1.0
        else:
            raise ValueError("'sort' parameter must either be 'None', or a callable, or one of ('lhp','rhp','iuc','ouc')")
    result = gees(sfunction, a1, lwork=lwork, overwrite_a=overwrite_a, sort_t=sort_t)
    info = result[-1]
    if info < 0:
        raise ValueError('illegal value in {}-th argument of internal gees'.format(-info))
    elif info == a1.shape[0] + 1:
        raise LinAlgError('Eigenvalues could not be separated for reordering.')
    elif info == a1.shape[0] + 2:
        raise LinAlgError('Leading eigenvalues do not satisfy sort condition.')
    elif info > 0:
        raise LinAlgError('Schur form not found. Possibly ill-conditioned.')
    if sort_t == 0:
        return (result[0], result[-3])
    else:
        return (result[0], result[-3], result[1])
eps = numpy.finfo(float).eps
feps = numpy.finfo(single).eps
_array_kind = {'b': 0, 'h': 0, 'B': 0, 'i': 0, 'l': 0, 'f': 0, 'd': 0, 'F': 1, 'D': 1}
_array_precision = {'i': 1, 'l': 1, 'f': 0, 'd': 1, 'F': 0, 'D': 1}
_array_type = [['f', 'd'], ['F', 'D']]

def _commonType(*arrays):
    if False:
        for i in range(10):
            print('nop')
    kind = 0
    precision = 0
    for a in arrays:
        t = a.dtype.char
        kind = max(kind, _array_kind[t])
        precision = max(precision, _array_precision[t])
    return _array_type[kind][precision]

def _castCopy(type, *arrays):
    if False:
        print('Hello World!')
    cast_arrays = ()
    for a in arrays:
        if a.dtype.char == type:
            cast_arrays = cast_arrays + (a.copy(),)
        else:
            cast_arrays = cast_arrays + (a.astype(type),)
    if len(cast_arrays) == 1:
        return cast_arrays[0]
    else:
        return cast_arrays

def rsf2csf(T, Z, check_finite=True):
    if False:
        print('Hello World!')
    '\n    Convert real Schur form to complex Schur form.\n\n    Convert a quasi-diagonal real-valued Schur form to the upper-triangular\n    complex-valued Schur form.\n\n    Parameters\n    ----------\n    T : (M, M) array_like\n        Real Schur form of the original array\n    Z : (M, M) array_like\n        Schur transformation matrix\n    check_finite : bool, optional\n        Whether to check that the input arrays contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    T : (M, M) ndarray\n        Complex Schur form of the original array\n    Z : (M, M) ndarray\n        Schur transformation matrix corresponding to the complex form\n\n    See Also\n    --------\n    schur : Schur decomposition of an array\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.linalg import schur, rsf2csf\n    >>> A = np.array([[0, 2, 2], [0, 1, 2], [1, 0, 1]])\n    >>> T, Z = schur(A)\n    >>> T\n    array([[ 2.65896708,  1.42440458, -1.92933439],\n           [ 0.        , -0.32948354, -0.49063704],\n           [ 0.        ,  1.31178921, -0.32948354]])\n    >>> Z\n    array([[0.72711591, -0.60156188, 0.33079564],\n           [0.52839428, 0.79801892, 0.28976765],\n           [0.43829436, 0.03590414, -0.89811411]])\n    >>> T2 , Z2 = rsf2csf(T, Z)\n    >>> T2\n    array([[2.65896708+0.j, -1.64592781+0.743164187j, -1.21516887+1.00660462j],\n           [0.+0.j , -0.32948354+8.02254558e-01j, -0.82115218-2.77555756e-17j],\n           [0.+0.j , 0.+0.j, -0.32948354-0.802254558j]])\n    >>> Z2\n    array([[0.72711591+0.j,  0.28220393-0.31385693j,  0.51319638-0.17258824j],\n           [0.52839428+0.j,  0.24720268+0.41635578j, -0.68079517-0.15118243j],\n           [0.43829436+0.j, -0.76618703+0.01873251j, -0.03063006+0.46857912j]])\n\n    '
    if check_finite:
        (Z, T) = map(asarray_chkfinite, (Z, T))
    else:
        (Z, T) = map(asarray, (Z, T))
    for (ind, X) in enumerate([Z, T]):
        if X.ndim != 2 or X.shape[0] != X.shape[1]:
            raise ValueError("Input '{}' must be square.".format('ZT'[ind]))
    if T.shape[0] != Z.shape[0]:
        raise ValueError('Input array shapes must match: Z: {} vs. T: {}'.format(Z.shape, T.shape))
    N = T.shape[0]
    t = _commonType(Z, T, array([3.0], 'F'))
    (Z, T) = _castCopy(t, Z, T)
    for m in range(N - 1, 0, -1):
        if abs(T[m, m - 1]) > eps * (abs(T[m - 1, m - 1]) + abs(T[m, m])):
            mu = eigvals(T[m - 1:m + 1, m - 1:m + 1]) - T[m, m]
            r = norm([mu[0], T[m, m - 1]])
            c = mu[0] / r
            s = T[m, m - 1] / r
            G = array([[c.conj(), s], [-s, c]], dtype=t)
            T[m - 1:m + 1, m - 1:] = G.dot(T[m - 1:m + 1, m - 1:])
            T[:m + 1, m - 1:m + 1] = T[:m + 1, m - 1:m + 1].dot(G.conj().T)
            Z[:, m - 1:m + 1] = Z[:, m - 1:m + 1].dot(G.conj().T)
        T[m, m - 1] = 0.0
    return (T, Z)