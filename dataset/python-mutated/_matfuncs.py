import math
import cupy
from cupy.linalg import _util

def khatri_rao(a, b):
    if False:
        return 10
    '\n    Khatri-rao product\n\n    A column-wise Kronecker product of two matrices\n\n    Parameters\n    ----------\n    a : (n, k) array_like\n        Input array\n    b : (m, k) array_like\n        Input array\n\n    Returns\n    -------\n    c:  (n*m, k) ndarray\n        Khatri-rao product of `a` and `b`.\n\n    See Also\n    --------\n    .. seealso:: :func:`scipy.linalg.khatri_rao`\n\n    '
    _util._assert_2d(a)
    _util._assert_2d(b)
    if a.shape[1] != b.shape[1]:
        raise ValueError('The number of columns for both arrays should be equal.')
    c = a[..., :, cupy.newaxis, :] * b[..., cupy.newaxis, :, :]
    return c.reshape((-1,) + c.shape[2:])
b = [6.476475253248e+16, 3.238237626624e+16, 7771770303897600.0, 1187353796428800.0, 129060195264000.0, 10559470521600.0, 670442572800.0, 33522128640.0, 1323241920.0, 40840800.0, 960960.0, 16380.0, 182.0, 1.0]
th13 = 5.37

def expm(a):
    if False:
        print('Hello World!')
    'Compute the matrix exponential.\n\n    Parameters\n    ----------\n    a : ndarray, 2D\n\n    Returns\n    -------\n    matrix exponential of `a`\n\n    Notes\n    -----\n    Uses (a simplified) version of Algorithm 2.3 of [1]_:\n    a [13 / 13] Pade approximant with scaling and squaring.\n\n    Simplifications:\n\n        * we always use a [13/13] approximate\n        * no matrix balancing\n\n    References\n    ----------\n    .. [1] N. Higham, SIAM J. MATRIX ANAL. APPL. Vol. 26(4), p. 1179 (2005)\n       https://doi.org/10.1137/04061101X\n\n    '
    if a.size == 0:
        return cupy.zeros((0, 0), dtype=a.dtype)
    n = a.shape[0]
    mu = cupy.diag(a).sum() / n
    A = a - cupy.eye(n) * mu
    nrmA = cupy.linalg.norm(A, ord=1).item()
    scale = nrmA > th13
    if scale:
        s = int(math.ceil(math.log2(float(nrmA) / th13))) + 1
    else:
        s = 1
    A /= 2 ** s
    A2 = A @ A
    A4 = A2 @ A2
    A6 = A2 @ A4
    E = cupy.eye(A.shape[0])
    (u1, u2, v1, v2) = _expm_inner(E, A, A2, A4, A6, cupy.asarray(b))
    u = A @ (A6 @ u1 + u2)
    v = A6 @ v1 + v2
    r13 = cupy.linalg.solve(-u + v, u + v)
    x = r13
    for _ in range(s):
        x = x @ x
    x *= math.exp(mu)
    return x

@cupy.fuse
def _expm_inner(E, A, A2, A4, A6, b):
    if False:
        i = 10
        return i + 15
    u1 = b[13] * A6 + b[11] * A4 + b[9] * A2
    u2 = b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * E
    v1 = b[12] * A6 + b[10] * A4 + b[8] * A
    v2 = b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * E
    return (u1, u2, v1, v2)