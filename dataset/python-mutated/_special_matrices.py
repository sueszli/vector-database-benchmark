import math
import warnings
import cupy
from cupy import _core
from cupyx.scipy.linalg import _uarray

@_uarray.implements('tri')
def tri(N, M=None, k=0, dtype=None):
    if False:
        return 10
    ' Construct (``N``, ``M``) matrix filled with ones at and below the\n    ``k``-th diagonal. The matrix has ``A[i,j] == 1`` for ``i <= j + k``.\n\n    Args:\n        N (int): The size of the first dimension of the matrix.\n        M (int, optional): The size of the second dimension of the matrix. If\n            ``M`` is None, ``M = N`` is assumed.\n        k (int, optional):  Number of subdiagonal below which matrix is filled\n            with ones. ``k = 0`` is the main diagonal, ``k < 0`` subdiagonal\n            and ``k > 0`` superdiagonal.\n        dtype (dtype, optional): Data type of the matrix.\n\n    Returns:\n        cupy.ndarray: Tri matrix.\n\n    .. seealso:: :func:`scipy.linalg.tri`\n    '
    warnings.warn("'tri'/'tril/'triu' are deprecated", DeprecationWarning)
    if M is None:
        M = N
    elif isinstance(M, str):
        (M, dtype) = (N, M)
    return cupy.tri(N, M, k, bool if dtype is None else dtype)

@_uarray.implements('tril')
def tril(m, k=0):
    if False:
        return 10
    'Make a copy of a matrix with elements above the ``k``-th diagonal\n    zeroed.\n\n    Args:\n        m (cupy.ndarray): Matrix whose elements to return\n        k (int, optional): Diagonal above which to zero elements.\n            ``k == 0`` is the main diagonal, ``k < 0`` subdiagonal and\n            ``k > 0`` superdiagonal.\n\n    Returns:\n        (cupy.ndarray): Return is the same shape and type as ``m``.\n\n    .. seealso:: :func:`scipy.linalg.tril`\n    '
    warnings.warn("'tri'/'tril/'triu' are deprecated", DeprecationWarning)
    t = tri(m.shape[0], m.shape[1], k=k, dtype=m.dtype.char)
    t *= m
    return t

@_uarray.implements('triu')
def triu(m, k=0):
    if False:
        return 10
    'Make a copy of a matrix with elements below the ``k``-th diagonal\n    zeroed.\n\n    Args:\n        m (cupy.ndarray): Matrix whose elements to return\n        k (int, optional): Diagonal above which to zero elements.\n            ``k == 0`` is the main diagonal, ``k < 0`` subdiagonal and\n            ``k > 0`` superdiagonal.\n\n    Returns:\n        (cupy.ndarray): Return matrix with zeroed elements below the kth\n        diagonal and has same shape and type as ``m``.\n\n    .. seealso:: :func:`scipy.linalg.triu`\n    '
    warnings.warn("'tri'/'tril/'triu' are deprecated", DeprecationWarning)
    t = tri(m.shape[0], m.shape[1], k - 1, m.dtype.char)
    cupy.subtract(1, t, out=t)
    t *= m
    return t

@_uarray.implements('toeplitz')
def toeplitz(c, r=None):
    if False:
        return 10
    'Construct a Toeplitz matrix.\n\n    The Toeplitz matrix has constant diagonals, with ``c`` as its first column\n    and ``r`` as its first row. If ``r`` is not given, ``r == conjugate(c)`` is\n    assumed.\n\n    Args:\n        c (cupy.ndarray): First column of the matrix. Whatever the actual shape\n            of ``c``, it will be converted to a 1-D array.\n        r (cupy.ndarray, optional): First row of the matrix. If None,\n            ``r = conjugate(c)`` is assumed; in this case, if ``c[0]`` is real,\n            the result is a Hermitian matrix. r[0] is ignored; the first row of\n            the returned matrix is ``[c[0], r[1:]]``. Whatever the actual shape\n            of ``r``, it will be converted to a 1-D array.\n\n    Returns:\n        cupy.ndarray: The Toeplitz matrix. Dtype is the same as\n        ``(c[0] + r[0]).dtype``.\n\n    .. seealso:: :func:`cupyx.scipy.linalg.circulant`\n    .. seealso:: :func:`cupyx.scipy.linalg.hankel`\n    .. seealso:: :func:`cupyx.scipy.linalg.solve_toeplitz`\n    .. seealso:: :func:`cupyx.scipy.linalg.fiedler`\n    .. seealso:: :func:`scipy.linalg.toeplitz`\n    '
    c = c.ravel()
    r = c.conjugate() if r is None else r.ravel()
    return _create_toeplitz_matrix(c[::-1], r[1:])

@_uarray.implements('circulant')
def circulant(c):
    if False:
        i = 10
        return i + 15
    'Construct a circulant matrix.\n\n    Args:\n        c (cupy.ndarray): 1-D array, the first column of the matrix.\n\n    Returns:\n        cupy.ndarray: A circulant matrix whose first column is ``c``.\n\n    .. seealso:: :func:`cupyx.scipy.linalg.toeplitz`\n    .. seealso:: :func:`cupyx.scipy.linalg.hankel`\n    .. seealso:: :func:`cupyx.scipy.linalg.solve_circulant`\n    .. seealso:: :func:`cupyx.scipy.linalg.fiedler`\n    .. seealso:: :func:`scipy.linalg.circulant`\n    '
    c = c.ravel()
    return _create_toeplitz_matrix(c[::-1], c[:0:-1])

@_uarray.implements('hankel')
def hankel(c, r=None):
    if False:
        return 10
    'Construct a Hankel matrix.\n\n    The Hankel matrix has constant anti-diagonals, with ``c`` as its first\n    column and ``r`` as its last row. If ``r`` is not given, then\n    ``r = zeros_like(c)`` is assumed.\n\n    Args:\n        c (cupy.ndarray): First column of the matrix. Whatever the actual shape\n            of ``c``, it will be converted to a 1-D array.\n        r (cupy.ndarray, optionnal): Last row of the matrix. If None,\n            ``r = zeros_like(c)`` is assumed. ``r[0]`` is ignored; the last row\n            of the returned matrix is ``[c[-1], r[1:]]``. Whatever the actual\n            shape of ``r``, it will be converted to a 1-D array.\n\n    Returns:\n        cupy.ndarray: The Hankel matrix. Dtype is the same as\n        ``(c[0] + r[0]).dtype``.\n\n    .. seealso:: :func:`cupyx.scipy.linalg.toeplitz`\n    .. seealso:: :func:`cupyx.scipy.linalg.circulant`\n    .. seealso:: :func:`scipy.linalg.hankel`\n    '
    c = c.ravel()
    r = cupy.zeros_like(c) if r is None else r.ravel()
    return _create_toeplitz_matrix(c, r[1:], True)

def _create_toeplitz_matrix(c, r, hankel=False):
    if False:
        return 10
    vals = cupy.concatenate((c, r))
    n = vals.strides[0]
    return cupy.lib.stride_tricks.as_strided(vals if hankel else vals[c.size - 1:], shape=(c.size, r.size + 1), strides=(n if hankel else -n, n)).copy()

@_uarray.implements('hadamard')
def hadamard(n, dtype=int):
    if False:
        i = 10
        return i + 15
    "Construct an Hadamard matrix.\n\n    Constructs an n-by-n Hadamard matrix, using Sylvester's construction. ``n``\n    must be a power of 2.\n\n    Args:\n        n (int): The order of the matrix. ``n`` must be a power of 2.\n        dtype (dtype, optional): The data type of the array to be constructed.\n\n    Returns:\n        (cupy.ndarray): The Hadamard matrix.\n\n    .. seealso:: :func:`scipy.linalg.hadamard`\n    "
    lg2 = 0 if n < 1 else int(n).bit_length() - 1
    if 2 ** lg2 != n:
        raise ValueError('n must be an positive a power of 2 integer')
    H = cupy.empty((n, n), dtype)
    return _hadamard_kernel(H, H)
_hadamard_kernel = _core.ElementwiseKernel('T in', 'T out', 'out = (__popc(_ind.get()[0] & _ind.get()[1]) & 1) ? -1 : 1;', 'cupyx_scipy_linalg_hadamard', reduce_dims=False)

@_uarray.implements('leslie')
def leslie(f, s):
    if False:
        i = 10
        return i + 15
    'Create a Leslie matrix.\n\n    Given the length n array of fecundity coefficients ``f`` and the length n-1\n    array of survival coefficients ``s``, return the associated Leslie matrix.\n\n    Args:\n        f (cupy.ndarray): The "fecundity" coefficients.\n        s (cupy.ndarray): The "survival" coefficients, has to be 1-D.  The\n            length of ``s`` must be one less than the length of ``f``, and it\n            must be at least 1.\n\n    Returns:\n        cupy.ndarray: The array is zero except for the first row, which is\n        ``f``, and the first sub-diagonal, which is ``s``. The data-type of\n        the array will be the data-type of ``f[0]+s[0]``.\n\n    .. seealso:: :func:`scipy.linalg.leslie`\n    '
    if f.ndim != 1:
        raise ValueError('Incorrect shape for f. f must be 1D')
    if s.ndim != 1:
        raise ValueError('Incorrect shape for s. s must be 1D')
    n = f.size
    if n != s.size + 1:
        raise ValueError('Length of s must be one less than length of f')
    if s.size == 0:
        raise ValueError('The length of s must be at least 1.')
    a = cupy.zeros((n, n), dtype=cupy.result_type(f, s))
    a[0] = f
    cupy.fill_diagonal(a[1:], s)
    return a

@_uarray.implements('kron')
def kron(a, b):
    if False:
        i = 10
        return i + 15
    'Kronecker product.\n\n    The result is the block matrix::\n        a[0,0]*b    a[0,1]*b  ... a[0,-1]*b\n        a[1,0]*b    a[1,1]*b  ... a[1,-1]*b\n        ...\n        a[-1,0]*b   a[-1,1]*b ... a[-1,-1]*b\n\n    Args:\n        a (cupy.ndarray): Input array\n        b (cupy.ndarray): Input array\n\n    Returns:\n        cupy.ndarray: Kronecker product of ``a`` and ``b``.\n\n    .. seealso:: :func:`scipy.linalg.kron`\n    '
    o = cupy.outer(a, b)
    o = o.reshape(a.shape + b.shape)
    return cupy.concatenate(cupy.concatenate(o, axis=1), axis=1)

@_uarray.implements('block_diag')
def block_diag(*arrs):
    if False:
        i = 10
        return i + 15
    'Create a block diagonal matrix from provided arrays.\n\n    Given the inputs ``A``, ``B``, and ``C``, the output will have these\n    arrays arranged on the diagonal::\n\n        [A, 0, 0]\n        [0, B, 0]\n        [0, 0, C]\n\n    Args:\n        A, B, C, ... (cupy.ndarray): Input arrays. A 1-D array of length ``n``\n            is treated as a 2-D array with shape ``(1,n)``.\n\n    Returns:\n        (cupy.ndarray): Array with ``A``, ``B``, ``C``, ... on the diagonal.\n        Output has the same dtype as ``A``.\n\n    .. seealso:: :func:`scipy.linalg.block_diag`\n    '
    if not arrs:
        return cupy.empty((1, 0))
    if len(arrs) == 1:
        arrs = (cupy.atleast_2d(*arrs),)
    else:
        arrs = cupy.atleast_2d(*arrs)
    if any((a.ndim != 2 for a in arrs)):
        bad = [k for k in range(len(arrs)) if arrs[k].ndim != 2]
        raise ValueError('arguments in the following positions have dimension greater than 2: {}'.format(bad))
    shapes = tuple((a.shape for a in arrs))
    shape = tuple((sum(x) for x in zip(*shapes)))
    out = cupy.zeros(shape, dtype=cupy.result_type(*arrs))
    (r, c) = (0, 0)
    for arr in arrs:
        (rr, cc) = arr.shape
        out[r:r + rr, c:c + cc] = arr
        r += rr
        c += cc
    return out

@_uarray.implements('companion')
def companion(a):
    if False:
        while True:
            i = 10
    'Create a companion matrix.\n\n    Create the companion matrix associated with the polynomial whose\n    coefficients are given in ``a``.\n\n    Args:\n        a (cupy.ndarray): 1-D array of polynomial coefficients. The length of\n            ``a`` must be at least two, and ``a[0]`` must not be zero.\n\n    Returns:\n        (cupy.ndarray): The first row of the output is ``-a[1:]/a[0]``, and the\n        first sub-diagonal is all ones. The data-type of the array is the\n        same as the data-type of ``-a[1:]/a[0]``.\n\n    .. seealso:: :func:`cupyx.scipy.linalg.fiedler_companion`\n    .. seealso:: :func:`scipy.linalg.companion`\n    '
    n = a.size
    if a.ndim != 1:
        raise ValueError('`a` must be one-dimensional.')
    if n < 2:
        raise ValueError('The length of `a` must be at least 2.')
    first_row = -a[1:] / a[0]
    c = cupy.zeros((n - 1, n - 1), dtype=first_row.dtype)
    c[0] = first_row
    cupy.fill_diagonal(c[1:], 1)
    return c

@_uarray.implements('helmert')
def helmert(n, full=False):
    if False:
        i = 10
        return i + 15
    'Create an Helmert matrix of order ``n``.\n\n    This has applications in statistics, compositional or simplicial analysis,\n    and in Aitchison geometry.\n\n    Args:\n        n (int): The size of the array to create.\n        full (bool, optional): If True the (n, n) ndarray will be returned.\n            Otherwise, the default, the submatrix that does not include the\n            first row will be returned.\n\n    Returns:\n        cupy.ndarray: The Helmert matrix. The shape is (n, n) or (n-1, n)\n        depending on the ``full`` argument.\n\n    .. seealso:: :func:`scipy.linalg.helmert`\n    '
    d = cupy.arange(n)
    H = cupy.tri(n, n, -1)
    H.diagonal()[:] -= d
    d *= cupy.arange(1, n + 1)
    H[0] = 1
    d[0] = n
    H /= cupy.sqrt(d)[:, None]
    return H if full else H[1:]

@_uarray.implements('hilbert')
def hilbert(n):
    if False:
        print('Hello World!')
    'Create a Hilbert matrix of order ``n``.\n\n    Returns the ``n`` by ``n`` array with entries ``h[i,j] = 1 / (i + j + 1)``.\n\n    Args:\n        n (int): The size of the array to create.\n\n    Returns:\n        cupy.ndarray: The Hilbert matrix.\n\n    .. seealso:: :func:`scipy.linalg.hilbert`\n    '
    values = cupy.arange(1, 2 * n, dtype=cupy.float64)
    cupy.reciprocal(values, values)
    return hankel(values[:n], r=values[n - 1:])

@_uarray.implements('dft')
def dft(n, scale=None):
    if False:
        i = 10
        return i + 15
    "Discrete Fourier transform matrix.\n\n    Create the matrix that computes the discrete Fourier transform of a\n    sequence. The nth primitive root of unity used to generate the matrix is\n    exp(-2*pi*i/n), where i = sqrt(-1).\n\n    Args:\n        n (int): Size the matrix to create.\n        scale (str, optional): Must be None, 'sqrtn', or 'n'.\n            If ``scale`` is 'sqrtn', the matrix is divided by ``sqrt(n)``.\n            If ``scale`` is 'n', the matrix is divided by ``n``.\n            If ``scale`` is None (default), the matrix is not normalized, and\n            the return value is simply the Vandermonde matrix of the roots of\n            unity.\n\n    Returns:\n        (cupy.ndarray): The DFT matrix.\n\n    Notes:\n        When ``scale`` is None, multiplying a vector by the matrix returned by\n        ``dft`` is mathematically equivalent to (but much less efficient than)\n        the calculation performed by ``scipy.fft.fft``.\n\n    .. seealso:: :func:`scipy.linalg.dft`\n    "
    if scale not in (None, 'sqrtn', 'n'):
        raise ValueError("scale must be None, 'sqrtn', or 'n'; %r is not valid." % (scale,))
    r = cupy.arange(n, dtype='complex128')
    r *= -2j * cupy.pi / n
    omegas = cupy.exp(r, out=r)[:, None]
    m = omegas ** cupy.arange(n)
    if scale is not None:
        m *= 1 / math.sqrt(n) if scale == 'sqrtn' else 1 / n
    return m

@_uarray.implements('fiedler')
def fiedler(a):
    if False:
        print('Hello World!')
    'Returns a symmetric Fiedler matrix\n\n    Given an sequence of numbers ``a``, Fiedler matrices have the structure\n    ``F[i, j] = np.abs(a[i] - a[j])``, and hence zero diagonals and nonnegative\n    entries. A Fiedler matrix has a dominant positive eigenvalue and other\n    eigenvalues are negative. Although not valid generally, for certain inputs,\n    the inverse and the determinant can be derived explicitly.\n\n    Args:\n        a (cupy.ndarray): coefficient array\n\n    Returns:\n        cupy.ndarray: the symmetric Fiedler matrix\n\n    .. seealso:: :func:`cupyx.scipy.linalg.circulant`\n    .. seealso:: :func:`cupyx.scipy.linalg.toeplitz`\n    .. seealso:: :func:`scipy.linalg.fiedler`\n    '
    if a.ndim != 1:
        raise ValueError('Input `a` must be a 1D array.')
    if a.size == 0:
        return cupy.zeros(0)
    if a.size == 1:
        return cupy.zeros((1, 1))
    a = a[:, None] - a
    return cupy.abs(a, out=a)

@_uarray.implements('fiedler_companion')
def fiedler_companion(a):
    if False:
        while True:
            i = 10
    'Returns a Fiedler companion matrix\n\n    Given a polynomial coefficient array ``a``, this function forms a\n    pentadiagonal matrix with a special structure whose eigenvalues coincides\n    with the roots of ``a``.\n\n    Args:\n        a (cupy.ndarray): 1-D array of polynomial coefficients in descending\n            order with a nonzero leading coefficient. For ``N < 2``, an empty\n            array is returned.\n\n    Returns:\n        cupy.ndarray: Resulting companion matrix\n\n    Notes:\n        Similar to ``companion`` the leading coefficient should be nonzero. In\n        the case the leading coefficient is not 1, other coefficients are\n        rescaled before the array generation. To avoid numerical issues, it is\n        best to provide a monic polynomial.\n\n    .. seealso:: :func:`cupyx.scipy.linalg.companion`\n    .. seealso:: :func:`scipy.linalg.fiedler_companion`\n    '
    if a.ndim != 1:
        raise ValueError('Input `a` must be a 1-D array.')
    if a.size < 2:
        return cupy.zeros((0,), a.dtype)
    if a.size == 2:
        return (-a[1] / a[0])[None, None]
    a = a / a[0]
    n = a.size - 1
    c = cupy.zeros((n, n), dtype=a.dtype)
    cupy.fill_diagonal(c[3::2, 1::2], 1)
    cupy.fill_diagonal(c[2::2, 1::2], -a[3::2])
    cupy.fill_diagonal(c[::2, 2::2], 1)
    cupy.fill_diagonal(c[::2, 1::2], -a[2::2])
    c[0, 0] = -a[1]
    c[1, 0] = 1
    return c

@_uarray.implements('convolution_matrix')
def convolution_matrix(a, n, mode='full'):
    if False:
        i = 10
        return i + 15
    "Construct a convolution matrix.\n\n    Constructs the Toeplitz matrix representing one-dimensional convolution.\n\n    Args:\n        a (cupy.ndarray): The 1-D array to convolve.\n        n (int): The number of columns in the resulting matrix. It gives the\n            length of the input to be convolved with ``a``. This is analogous\n            to the length of ``v`` in ``numpy.convolve(a, v)``.\n        mode (str): This must be one of (``'full'``, ``'valid'``, ``'same'``).\n            This is analogous to ``mode`` in ``numpy.convolve(v, a, mode)``.\n\n    Returns:\n        cupy.ndarray: The convolution matrix whose row count k depends on\n        ``mode``:\n\n        =========== =========================\n        ``mode``    k\n        =========== =========================\n        ``'full'``  m + n - 1\n        ``'same'``  max(m, n)\n        ``'valid'`` max(m, n) - min(m, n) + 1\n        =========== =========================\n\n    .. seealso:: :func:`cupyx.scipy.linalg.toeplitz`\n    .. seealso:: :func:`scipy.linalg.convolution_matrix`\n    "
    if n <= 0:
        raise ValueError('n must be a positive integer.')
    if a.ndim != 1:
        raise ValueError('convolution_matrix expects a one-dimensional array as input')
    if a.size == 0:
        raise ValueError('len(a) must be at least 1.')
    if mode not in ('full', 'valid', 'same'):
        raise ValueError("`mode` argument must be one of ('full', 'valid', 'same')")
    az = cupy.pad(a, (0, n - 1), 'constant')
    raz = cupy.pad(a[::-1], (0, n - 1), 'constant')
    if mode == 'same':
        trim = min(n, a.size) - 1
        tb = trim // 2
        te = trim - tb
        col0 = az[tb:az.size - te]
        row0 = raz[-n - tb:raz.size - tb]
    elif mode == 'valid':
        tb = min(n, a.size) - 1
        te = tb
        col0 = az[tb:az.size - te]
        row0 = raz[-n - tb:raz.size - tb]
    else:
        col0 = az
        row0 = raz[-n:]
    return toeplitz(col0, row0)