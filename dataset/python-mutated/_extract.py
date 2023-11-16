import cupy
import cupyx
from cupyx.scipy import sparse

def find(A):
    if False:
        i = 10
        return i + 15
    'Returns the indices and values of the nonzero elements of a matrix\n\n    Args:\n        A (cupy.ndarray or cupyx.scipy.sparse.spmatrix): Matrix whose nonzero\n            elements are desired.\n\n    Returns:\n        tuple of cupy.ndarray:\n            It returns (``I``, ``J``, ``V``). ``I``, ``J``, and ``V`` contain\n            respectively the row indices, column indices, and values of the\n            nonzero matrix entries.\n\n    .. seealso:: :func:`scipy.sparse.find`\n    '
    _check_A_type(A)
    A = sparse.coo_matrix(A, copy=True)
    A.sum_duplicates()
    nz_mask = A.data != 0
    return (A.row[nz_mask], A.col[nz_mask], A.data[nz_mask])

def tril(A, k=0, format=None):
    if False:
        while True:
            i = 10
    "Returns the lower triangular portion of a matrix in sparse format\n\n    Args:\n        A (cupy.ndarray or cupyx.scipy.sparse.spmatrix): Matrix whose lower\n            triangular portion is desired.\n        k (integer): The top-most diagonal of the lower triangle.\n        format (string): Sparse format of the result, e.g. 'csr', 'csc', etc.\n\n    Returns:\n        cupyx.scipy.sparse.spmatrix:\n            Lower triangular portion of A in sparse format.\n\n    .. seealso:: :func:`scipy.sparse.tril`\n    "
    _check_A_type(A)
    A = sparse.coo_matrix(A, copy=False)
    mask = A.row + k >= A.col
    return _masked_coo(A, mask).asformat(format)

def triu(A, k=0, format=None):
    if False:
        print('Hello World!')
    "Returns the upper triangular portion of a matrix in sparse format\n\n    Args:\n        A (cupy.ndarray or cupyx.scipy.sparse.spmatrix): Matrix whose upper\n            triangular portion is desired.\n        k (integer): The bottom-most diagonal of the upper triangle.\n        format (string): Sparse format of the result, e.g. 'csr', 'csc', etc.\n\n    Returns:\n        cupyx.scipy.sparse.spmatrix:\n            Upper triangular portion of A in sparse format.\n\n    .. seealso:: :func:`scipy.sparse.triu`\n    "
    _check_A_type(A)
    A = sparse.coo_matrix(A, copy=False)
    mask = A.row + k <= A.col
    return _masked_coo(A, mask).asformat(format)

def _check_A_type(A):
    if False:
        return 10
    if not (isinstance(A, cupy.ndarray) or cupyx.scipy.sparse.isspmatrix(A)):
        msg = 'A must be cupy.ndarray or cupyx.scipy.sparse.spmatrix'
        raise TypeError(msg)

def _masked_coo(A, mask):
    if False:
        for i in range(10):
            print('nop')
    row = A.row[mask]
    col = A.col[mask]
    data = A.data[mask]
    return sparse.coo_matrix((data, (row, col)), shape=A.shape, dtype=A.dtype)