"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numbers
import numpy as np
import scipy.sparse as sp
from cvxpy.interface import numpy_interface as np_intf
INTERFACES = {np.ndarray: np_intf.NDArrayInterface(), np.matrix: np_intf.MatrixInterface(), sp.csc_matrix: np_intf.SparseMatrixInterface()}
DEFAULT_NP_INTF = INTERFACES[np.ndarray]
DEFAULT_INTF = INTERFACES[np.ndarray]
DEFAULT_SPARSE_INTF = INTERFACES[sp.csc_matrix]

def get_matrix_interface(target_class):
    if False:
        return 10
    return INTERFACES[target_class]

def get_cvxopt_dense_intf():
    if False:
        for i in range(10):
            print('nop')
    'Dynamic import of CVXOPT dense interface.\n    '
    import cvxpy.interface.cvxopt_interface.valuerix_interface as dmi
    return dmi.DenseMatrixInterface()

def get_cvxopt_sparse_intf():
    if False:
        i = 10
        return i + 15
    'Dynamic import of CVXOPT sparse interface.\n    '
    import cvxpy.interface.cvxopt_interface.sparse_matrix_interface as smi
    return smi.SparseMatrixInterface()

def sparse2cvxopt(value):
    if False:
        while True:
            i = 10
    'Converts a SciPy sparse matrix to a CVXOPT sparse matrix.\n\n    Parameters\n    ----------\n    sparse_mat : SciPy sparse matrix\n        The matrix to convert.\n\n    Returns\n    -------\n    CVXOPT spmatrix\n        The converted matrix.\n    '
    import cvxopt
    if isinstance(value, (np.ndarray, np.matrix)):
        return cvxopt.sparse(cvxopt.matrix(value.astype('float64')), tc='d')
    elif sp.issparse(value):
        value = value.tocoo()
        return cvxopt.spmatrix(value.data.tolist(), value.row.tolist(), value.col.tolist(), size=value.shape, tc='d')

def dense2cvxopt(value):
    if False:
        print('Hello World!')
    'Converts a NumPy matrix to a CVXOPT matrix.\n\n    Parameters\n    ----------\n    value : NumPy matrix/ndarray\n        The matrix to convert.\n\n    Returns\n    -------\n    CVXOPT matrix\n        The converted matrix.\n    '
    import cvxopt
    return cvxopt.matrix(value, tc='d')

def cvxopt2dense(value):
    if False:
        while True:
            i = 10
    'Converts a CVXOPT matrix to a NumPy ndarray.\n\n    Parameters\n    ----------\n    value : CVXOPT matrix\n        The matrix to convert.\n\n    Returns\n    -------\n    NumPy ndarray\n        The converted matrix.\n    '
    return np.array(value)

def is_sparse(constant) -> bool:
    if False:
        return 10
    'Is the constant a sparse matrix?\n    '
    return sp.issparse(constant)

def shape(constant):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(constant, numbers.Number) or np.isscalar(constant):
        return tuple()
    elif isinstance(constant, list):
        if len(constant) == 0:
            return (0,)
        elif isinstance(constant[0], numbers.Number):
            return (len(constant),)
        else:
            return (len(constant[0]), len(constant))
    elif constant.__class__ in INTERFACES:
        return INTERFACES[constant.__class__].shape(constant)
    elif is_sparse(constant):
        return INTERFACES[sp.csc_matrix].shape(constant)
    else:
        raise TypeError('%s is not a valid type for a Constant value.' % type(constant))

def is_vector(constant) -> bool:
    if False:
        print('Hello World!')
    return shape(constant)[1] == 1

def is_scalar(constant) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return shape(constant) == (1, 1)

def from_2D_to_1D(constant):
    if False:
        for i in range(10):
            print('nop')
    'Convert 2D Numpy matrices or arrays to 1D.\n    '
    if isinstance(constant, np.ndarray) and constant.ndim == 2:
        return np.asarray(constant)[:, 0]
    else:
        return constant

def from_1D_to_2D(constant):
    if False:
        i = 10
        return i + 15
    'Convert 1D Numpy arrays to matrices.\n    '
    if isinstance(constant, np.ndarray) and constant.ndim == 1:
        return np.mat(constant).T
    else:
        return constant

def convert(constant, sparse: bool=False, convert_scalars: bool=False):
    if False:
        print('Hello World!')
    'Convert to appropriate type.\n    '
    if isinstance(constant, (list, np.matrix)):
        return DEFAULT_INTF.const_to_matrix(constant, convert_scalars=convert_scalars)
    elif sparse:
        return DEFAULT_SPARSE_INTF.const_to_matrix(constant, convert_scalars=convert_scalars)
    else:
        return constant

def scalar_value(constant):
    if False:
        i = 10
        return i + 15
    if isinstance(constant, numbers.Number) or np.isscalar(constant):
        return constant
    elif isinstance(constant, list):
        return constant[0]
    elif constant.__class__ in INTERFACES:
        return INTERFACES[constant.__class__].scalar_value(constant)
    elif is_sparse(constant):
        return INTERFACES[sp.csc_matrix].scalar_value(constant.tocsc())
    else:
        raise TypeError('%s is not a valid type for a Constant value.' % type(constant))

def sign(constant):
    if False:
        for i in range(10):
            print('nop')
    'Return (is positive, is negative).\n\n    Parameters\n    ----------\n    constant : numeric type\n        The numeric value to evaluate the sign of.\n\n    Returns\n    -------\n    tuple\n        The sign of the constant.\n    '
    if isinstance(constant, numbers.Number):
        max_val = constant
        min_val = constant
    elif sp.issparse(constant):
        max_val = constant.max()
        min_val = constant.min()
    else:
        mat = INTERFACES[np.ndarray].const_to_matrix(constant)
        max_val = mat.max()
        min_val = mat.min()
    return (min_val >= 0, max_val <= 0)

def is_complex(constant, tol: float=1e-05) -> bool:
    if False:
        return 10
    'Return (is real, is imaginary).\n\n    Parameters\n    ----------\n    constant : numeric type\n        The numeric value to evaluate the sign of.\n    tol : float, optional\n        The largest magnitude considered nonzero.\n\n    Returns\n    -------\n    tuple\n        The sign of the constant.\n    '
    complex_type = np.iscomplexobj(constant)
    if not complex_type:
        return (True, False)
    if isinstance(constant, numbers.Number):
        real_max = np.abs(np.real(constant))
        imag_max = np.abs(np.imag(constant))
    elif sp.issparse(constant):
        real_max = np.abs(constant.real).max()
        imag_max = np.abs(constant.imag).max()
    else:
        constant = INTERFACES[np.ndarray].const_to_matrix(constant)
        real_max = np.abs(constant.real).max()
        imag_max = np.abs(constant.imag).max()
    return (real_max >= tol, imag_max >= tol)

def index(constant, key):
    if False:
        print('Hello World!')
    if is_scalar(constant):
        return constant
    elif constant.__class__ in INTERFACES:
        return INTERFACES[constant.__class__].index(constant, key)
    elif is_sparse(constant):
        interface = INTERFACES[sp.csc_matrix]
        constant = interface.const_to_matrix(constant)
        return interface.index(constant, key)

def is_hermitian(constant) -> bool:
    if False:
        i = 10
        return i + 15
    'Check if a matrix is Hermitian and/or symmetric.\n    '
    complex_type = np.iscomplexobj(constant)
    if complex_type:
        is_symm = False
        if sp.issparse(constant):
            is_herm = is_sparse_symmetric(constant, complex=True)
        else:
            is_herm = np.allclose(constant, np.conj(constant.T))
    else:
        if sp.issparse(constant):
            is_symm = is_sparse_symmetric(constant, complex=False)
        else:
            is_symm = np.allclose(constant, constant.T)
        is_herm = is_symm
    return (is_symm, is_herm)

def is_skew_symmetric(constant) -> bool:
    if False:
        i = 10
        return i + 15
    'Is the '
    complex_type = np.iscomplexobj(constant)
    if complex_type:
        return False
    else:
        if sp.issparse(constant):
            is_skew_symm = is_sparse_skew_symmetric(constant)
        else:
            is_skew_symm = np.allclose(constant + constant.T, 0.0)
        return is_skew_symm

def is_sparse_symmetric(m, complex: bool=False) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Check if a sparse matrix is symmetric\n\n    Parameters\n    ----------\n    m : array or sparse matrix\n        A square matrix.\n\n    Returns\n    -------\n    check : bool\n        The check result.\n\n    '
    if m.shape[0] != m.shape[1]:
        raise ValueError('m must be a square matrix')
    if not isinstance(m, sp.coo_matrix):
        m = sp.coo_matrix(m)
    (r, c, v) = (m.row, m.col, m.data)
    tril_no_diag = r > c
    triu_no_diag = c > r
    if triu_no_diag.sum() != tril_no_diag.sum():
        return False
    rl = r[tril_no_diag]
    cl = c[tril_no_diag]
    vl = v[tril_no_diag]
    ru = r[triu_no_diag]
    cu = c[triu_no_diag]
    vu = v[triu_no_diag]
    sortl = np.lexsort((cl, rl))
    sortu = np.lexsort((ru, cu))
    vl = vl[sortl]
    vu = vu[sortu]
    if complex:
        check = np.allclose(vl, np.conj(vu))
    else:
        check = np.allclose(vl, vu)
    return check

def is_sparse_skew_symmetric(A) -> bool:
    if False:
        print('Hello World!')
    'Check if a real sparse matrix A satisfies A + A.T == 0.\n\n    Parameters\n    ----------\n    A : array or sparse matrix\n        A square matrix.\n\n    Returns\n    -------\n    check : bool\n        The check result.\n    '
    if A.shape[0] != A.shape[1]:
        raise ValueError('m must be a square matrix')
    if not isinstance(A, sp.coo_matrix):
        A = sp.coo_matrix(A)
    (r, c, v) = (A.row, A.col, A.data)
    tril = r >= c
    triu = c >= r
    if triu.sum() != tril.sum():
        return False
    rl = r[tril]
    cl = c[tril]
    vl = v[tril]
    ru = r[triu]
    cu = c[triu]
    vu = v[triu]
    sortl = np.lexsort((cl, rl))
    sortu = np.lexsort((ru, cu))
    vl = vl[sortl]
    vu = vu[sortu]
    check = np.allclose(vl + vu, 0)
    return check