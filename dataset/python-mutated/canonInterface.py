"""
Copyright 2017 Steven Diamond

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
from __future__ import annotations
import numbers
import os
import numpy as np
import scipy.sparse as sp
import cvxpy.cvxcore.python.cvxcore as cvxcore
import cvxpy.settings as s
from cvxpy.lin_ops import lin_op as lo
from cvxpy.lin_ops.canon_backend import CanonBackend

def get_parameter_vector(param_size, param_id_to_col, param_id_to_size, param_id_to_value_fn, zero_offset: bool=False):
    if False:
        print('Hello World!')
    'Returns a flattened parameter vector\n\n    The flattened vector includes a constant offset (i.e, a 1).\n\n    Parameters\n    ----------\n        param_size: The number of parameters\n        param_id_to_col: A dict from parameter id to column offset\n        param_id_to_size: A dict from parameter id to parameter size\n        param_id_to_value_fn: A callable that returns a value for a parameter id\n        zero_offset: (optional) if True, zero out the constant offset in the\n                     parameter vector\n\n    Returns\n    -------\n        A flattened NumPy array of parameter values, of length param_size + 1\n    '
    if param_size == 0:
        return None
    param_vec = np.zeros(param_size + 1)
    for (param_id, col) in param_id_to_col.items():
        if param_id == lo.CONSTANT_ID:
            if not zero_offset:
                param_vec[col] = 1
        else:
            value = param_id_to_value_fn(param_id).flatten(order='F')
            size = param_id_to_size[param_id]
            param_vec[col:col + size] = value
    return param_vec

def reduce_problem_data_tensor(A, var_length, quad_form: bool=False):
    if False:
        while True:
            i = 10
    "Reduce a problem data tensor, for efficient construction of the problem data\n\n    If quad_form=False, the problem data tensor A is a matrix of shape (m, p), where p is the\n    length of the parameter vector. The product A@param_vec gives the\n    entries of the problem data matrix for a solver;\n    the solver's problem data matrix has dimensions(n_constr, n_var + 1),\n    and n_constr*(n_var + 1) = m. In other words, each row in A corresponds\n    to an entry in the solver's data matrix.\n\n    If quad_form=True, the problem data tensor A is a matrix of shape (m, p), where p is the\n    length of the parameter vector. The product A@param_vec gives the\n    entries of the quadratic form matrix P for a QP solver;\n    the solver's quadratic matrix P has dimensions(n_var, n_var),\n    and n_var*n_var = m. In other words, each row in A corresponds\n    to an entry in the solver's quadratic form matrix P.\n\n    This function removes the rows in A that are identically zero, since these\n    rows correspond to zeros in the problem data. It also returns the indices\n    and indptr to construct the problem data matrix from the reduced\n    representation of A, and the shape of the problem data matrix.\n\n    Let reduced_A be the sparse matrix returned by this function. Then the\n    problem data can be computed using\n\n       data : = reduced_A @ param_vec\n\n    and the problem data matrix can be constructed with\n\n        problem_data : = sp.csc_matrix(\n            (data, indices, indptr), shape = shape)\n\n    Parameters\n    ----------\n        A : A sparse matrix, the problem data tensor; must not have a 0 in its\n            shape\n        var_length: number of variables in the problem\n        quad_form: (optional) if True, consider quadratic form matrix P\n\n    Returns\n    -------\n        reduced_A: A CSR sparse matrix with redundant rows removed\n        indices: CSC indices for the problem data matrix\n        indptr: CSC indptr for the problem data matrix\n        shape: the shape of the problem data matrix\n    "
    A.eliminate_zeros()
    A_coo = A.tocoo()
    (unique_old_row, reduced_row) = np.unique(A_coo.row, return_inverse=True)
    reduced_A_shape = (unique_old_row.size, A_coo.shape[1])
    reduced_A = sp.coo_matrix((A_coo.data, (reduced_row, A_coo.col)), shape=reduced_A_shape)
    reduced_A = reduced_A.tocsr()
    nonzero_rows = unique_old_row
    n_cols = var_length
    if not quad_form:
        n_cols += 1
    n_constr = A.shape[0] // n_cols
    shape = (n_constr, n_cols)
    indices = nonzero_rows % n_constr
    cols = nonzero_rows // n_constr
    indptr = np.zeros(n_cols + 1, dtype=np.int32)
    (positions, counts) = np.unique(cols, return_counts=True)
    indptr[positions + 1] = counts
    indptr = np.cumsum(indptr)
    return (reduced_A, indices, indptr, shape)

def nonzero_csc_matrix(A):
    if False:
        i = 10
        return i + 15
    assert not np.isnan(A.data).any()
    zero_indices = A.data == 0
    A.data[zero_indices] = np.nan
    (A_rows, A_cols) = A.nonzero()
    ind = np.argsort(A_cols, kind='mergesort')
    A_rows = A_rows[ind]
    A_cols = A_cols[ind]
    A.data[zero_indices] = 0
    return (A_rows, A_cols)

def A_mapping_nonzero_rows(problem_data_tensor, var_length):
    if False:
        for i in range(10):
            print('nop')
    problem_data_tensor_csc = problem_data_tensor.tocsc()
    A_nrows = problem_data_tensor.shape[0] // (var_length + 1)
    A_ncols = var_length
    A_mapping = problem_data_tensor_csc[:A_nrows * A_ncols, :-1]
    (A_mapping_nonzero_rows, _) = A_mapping.nonzero()
    return np.unique(A_mapping_nonzero_rows)

def get_matrix_from_tensor(problem_data_tensor, param_vec, var_length, nonzero_rows=None, with_offset=True, problem_data_index=None):
    if False:
        for i in range(10):
            print('nop')
    'Applies problem_data_tensor to param_vec to obtain matrix and (optionally)\n    the offset.\n\n    This function applies problem_data_tensor to param_vec to obtain\n    a matrix representation of the corresponding affine map.\n\n    Parameters\n    ----------\n        problem_data_tensor: tensor returned from get_problem_matrix,\n            representing a parameterized affine map\n        param_vec: flattened parameter vector\n        var_length: the number of variables\n        nonzero_rows: (optional) rows in the part of problem_data_tensor\n            corresponding to A that have nonzeros in them (i.e., rows that\n            are affected by parameters); if not None, then the corresponding\n            entries in A will have explicit zeros.\n        with_offset: (optional) return offset. Defaults to True.\n        problem_data_index: (optional) a tuple (indices, indptr, shape) for\n            construction of the CSC matrix holding the problem data and offset\n\n    Returns\n    -------\n        A tuple (A, b), where A is a matrix with `var_length` columns\n        and b is a flattened NumPy array representing the constant offset.\n        If with_offset=False, returned b is None.\n    '
    if param_vec is None:
        flat_problem_data = problem_data_tensor
        if problem_data_index is not None:
            flat_problem_data = flat_problem_data.toarray().flatten()
    elif problem_data_index is not None:
        flat_problem_data = problem_data_tensor @ param_vec
    else:
        param_vec = sp.csc_matrix(param_vec[:, None])
        flat_problem_data = problem_data_tensor @ param_vec
    if problem_data_index is not None:
        (indices, indptr, shape) = problem_data_index
        M = sp.csc_matrix((flat_problem_data, indices, indptr), shape=shape)
    else:
        n_cols = var_length
        if with_offset:
            n_cols += 1
        M = flat_problem_data.reshape((-1, n_cols), order='F').tocsc()
    if with_offset:
        A = M[:, :-1].tocsc()
        b = np.squeeze(M[:, -1].toarray().flatten())
    else:
        A = M.tocsc()
        b = None
    if nonzero_rows is not None and nonzero_rows.size > 0:
        (A_nrows, _) = A.shape
        (A_rows, A_cols) = nonzero_csc_matrix(A)
        A_vals = np.append(A.data, np.zeros(nonzero_rows.size))
        A_rows = np.append(A_rows, nonzero_rows % A_nrows)
        A_cols = np.append(A_cols, nonzero_rows // A_nrows)
        A = sp.csc_matrix((A_vals, (A_rows, A_cols)), shape=A.shape)
    return (A, b)

def get_matrix_and_offset_from_unparameterized_tensor(problem_data_tensor, var_length):
    if False:
        i = 10
        return i + 15
    'Converts unparameterized tensor to matrix offset representation\n\n    problem_data_tensor _must_ have been obtained from calling\n    get_problem_matrix on a problem with 0 parameters.\n\n    Parameters\n    ----------\n        problem_data_tensor: tensor returned from get_problem_matrix,\n            representing an affine map\n        var_length: the number of variables\n\n    Returns\n    -------\n        A tuple (A, b), where A is a matrix with `var_length` columns\n        and b is a flattened NumPy array representing the constant offset.\n    '
    assert problem_data_tensor.shape[1] == 1
    return get_matrix_and_offset_from_tensor(problem_data_tensor, None, var_length)

def get_default_canon_backend() -> str:
    if False:
        i = 10
        return i + 15
    '\n    Returns the default canonicalization backend, which can be set globally using an\n    environment variable.\n    '
    return os.environ.get('CVXPY_DEFAULT_CANON_BACKEND', s.DEFAULT_CANON_BACKEND)

def get_problem_matrix(linOps, var_length, id_to_col, param_to_size, param_to_col, constr_length, canon_backend: str | None=None):
    if False:
        while True:
            i = 10
    "\n    Builds a sparse representation of the problem data.\n\n    Parameters\n    ----------\n        linOps: A list of python linOp trees representing an affine expression\n        var_length: The total length of the variables.\n        id_to_col: A map from variable id to column offset.\n        param_to_size: A map from parameter id to parameter size.\n        param_to_col: A map from parameter id to column in tensor.\n        constr_length: Summed sizes of constraints input.\n        canon_backend :\n            'CPP' (default) | 'SCIPY'\n            Specifies which backend to use for canonicalization, which can affect\n            compilation time. Defaults to None, i.e., selecting the default backend.\n\n    Returns\n    -------\n        A sparse (CSC) matrix with constr_length * (var_length + 1) rows and\n        param_size + 1 columns (where param_size is the length of the\n        parameter vector).\n    "
    default_canon_backend = get_default_canon_backend()
    canon_backend = default_canon_backend if not canon_backend else canon_backend
    if canon_backend == s.CPP_CANON_BACKEND:
        lin_vec = cvxcore.ConstLinOpVector()
        id_to_col_C = cvxcore.IntIntMap()
        for (id, col) in id_to_col.items():
            id_to_col_C[int(id)] = int(col)
        param_to_size_C = cvxcore.IntIntMap()
        for (id, size) in param_to_size.items():
            param_to_size_C[int(id)] = int(size)
        linPy_to_linC = {}
        for lin in linOps:
            build_lin_op_tree(lin, linPy_to_linC)
            tree = linPy_to_linC[lin]
            lin_vec.push_back(tree)
        problemData = cvxcore.build_matrix(lin_vec, int(var_length), id_to_col_C, param_to_size_C, s.get_num_threads())
        tensor_V = {}
        tensor_I = {}
        tensor_J = {}
        for (param_id, size) in param_to_size.items():
            tensor_V[param_id] = []
            tensor_I[param_id] = []
            tensor_J[param_id] = []
            problemData.param_id = param_id
            for i in range(size):
                problemData.vec_idx = i
                prob_len = problemData.getLen()
                tensor_V[param_id].append(problemData.getV(prob_len))
                tensor_I[param_id].append(problemData.getI(prob_len))
                tensor_J[param_id].append(problemData.getJ(prob_len))
        V = []
        I = []
        J = []
        param_size_plus_one = 0
        for (param_id, col) in param_to_col.items():
            size = param_to_size[param_id]
            param_size_plus_one += size
            for i in range(size):
                V.append(tensor_V[param_id][i])
                I.append(tensor_I[param_id][i] + tensor_J[param_id][i] * constr_length)
                J.append(tensor_J[param_id][i] * 0 + (i + col))
        V = np.concatenate(V)
        I = np.concatenate(I)
        J = np.concatenate(J)
        output_shape = (np.int64(constr_length) * np.int64(var_length + 1), param_size_plus_one)
        A = sp.csc_matrix((V, (I, J)), shape=output_shape)
        return A
    elif canon_backend in {s.SCIPY_CANON_BACKEND, s.RUST_CANON_BACKEND, s.NUMPY_CANON_BACKEND}:
        param_size_plus_one = sum(param_to_size.values())
        output_shape = (np.int64(constr_length) * np.int64(var_length + 1), param_size_plus_one)
        if len(linOps) > 0:
            backend = CanonBackend.get_backend(canon_backend, id_to_col, param_to_size, param_to_col, param_size_plus_one, var_length)
            A_py = backend.build_matrix(linOps)
        else:
            A_py = sp.csc_matrix(((), ((), ())), output_shape)
        assert A_py.shape == output_shape
        return A_py
    else:
        raise ValueError(f'Unknown backend: {canon_backend}')

def format_matrix(matrix, shape=None, format='dense'):
    if False:
        return 10
    'Returns the matrix in the appropriate form for SWIG wrapper'
    if format == 'dense':
        if len(shape) == 0:
            shape = (1, 1)
        elif len(shape) == 1:
            shape = shape + (1,)
        return np.reshape(matrix, shape, order='F')
    elif format == 'sparse':
        return sp.coo_matrix(matrix)
    elif format == 'scalar':
        return np.asfortranarray([[matrix]])
    else:
        raise NotImplementedError()
TYPE_MAP = {'VARIABLE': cvxcore.VARIABLE, 'PARAM': cvxcore.PARAM, 'PROMOTE': cvxcore.PROMOTE, 'MUL': cvxcore.MUL, 'RMUL': cvxcore.RMUL, 'MUL_ELEM': cvxcore.MUL_ELEM, 'DIV': cvxcore.DIV, 'SUM': cvxcore.SUM, 'NEG': cvxcore.NEG, 'INDEX': cvxcore.INDEX, 'TRANSPOSE': cvxcore.TRANSPOSE, 'SUM_ENTRIES': cvxcore.SUM_ENTRIES, 'TRACE': cvxcore.TRACE, 'RESHAPE': cvxcore.RESHAPE, 'DIAG_VEC': cvxcore.DIAG_VEC, 'DIAG_MAT': cvxcore.DIAG_MAT, 'UPPER_TRI': cvxcore.UPPER_TRI, 'CONV': cvxcore.CONV, 'HSTACK': cvxcore.HSTACK, 'VSTACK': cvxcore.VSTACK, 'SCALAR_CONST': cvxcore.SCALAR_CONST, 'DENSE_CONST': cvxcore.DENSE_CONST, 'SPARSE_CONST': cvxcore.SPARSE_CONST, 'NO_OP': cvxcore.NO_OP, 'KRON_R': cvxcore.KRON_R, 'KRON_L': cvxcore.KRON_L}

def get_type(linPy):
    if False:
        return 10
    ty = linPy.type.upper()
    if ty in TYPE_MAP:
        return TYPE_MAP[ty]
    else:
        raise NotImplementedError('Type %s is not supported.' % ty)

def set_matrix_data(linC, linPy) -> None:
    if False:
        return 10
    'Calls the appropriate cvxcore function to set the matrix data field of\n       our C++ linOp.\n    '
    if get_type(linPy) == cvxcore.SPARSE_CONST:
        coo = format_matrix(linPy.data, format='sparse')
        linC.set_sparse_data(coo.data, coo.row.astype(float), coo.col.astype(float), coo.shape[0], coo.shape[1])
    else:
        linC.set_dense_data(format_matrix(linPy.data, shape=linPy.shape))
        linC.set_data_ndim(len(linPy.data.shape))

def set_slice_data(linC, linPy) -> None:
    if False:
        return 10
    "\n    Loads the slice data, start, stop, and step into our C++ linOp.\n    The semantics of the slice operator is treated exactly the same as in\n    Python.  Note that the 'None' cases had to be handled at the wrapper level,\n    since we must load integers into our vector.\n    "
    for (i, sl) in enumerate(linPy.data):
        slice_vec = cvxcore.IntVector()
        for var in [sl.start, sl.stop, sl.step]:
            slice_vec.push_back(int(var))
        linC.push_back_slice_vec(slice_vec)

def set_linC_data(linC, linPy) -> None:
    if False:
        print('Hello World!')
    'Sets numerical data fields in linC.'
    assert linPy.data is not None
    if isinstance(linPy.data, tuple) and isinstance(linPy.data[0], slice):
        set_slice_data(linC, linPy)
    elif isinstance(linPy.data, float) or isinstance(linPy.data, numbers.Integral):
        linC.set_dense_data(format_matrix(linPy.data, format='scalar'))
        linC.set_data_ndim(0)
    else:
        set_matrix_data(linC, linPy)

def make_linC_from_linPy(linPy, linPy_to_linC) -> None:
    if False:
        print('Hello World!')
    'Construct a C++ LinOp corresponding to LinPy.\n\n    Children of linPy are retrieved from linPy_to_linC.\n    '
    if linPy in linPy_to_linC:
        return
    typ = get_type(linPy)
    shape = cvxcore.IntVector()
    lin_args_vec = cvxcore.ConstLinOpVector()
    for dim in linPy.shape:
        shape.push_back(int(dim))
    for argPy in linPy.args:
        lin_args_vec.push_back(linPy_to_linC[argPy])
    linC = cvxcore.LinOp(typ, shape, lin_args_vec)
    linPy_to_linC[linPy] = linC
    if linPy.data is not None:
        if isinstance(linPy.data, lo.LinOp):
            linC_data = linPy_to_linC[linPy.data]
            linC.set_linOp_data(linC_data)
            linC.set_data_ndim(len(linPy.data.shape))
        else:
            set_linC_data(linC, linPy)

def build_lin_op_tree(root_linPy, linPy_to_linC) -> None:
    if False:
        i = 10
        return i + 15
    'Construct C++ LinOp tree from Python LinOp tree.\n\n    Constructed C++ linOps are stored in the linPy_to_linC dict,\n    which maps Python linOps to their corresponding C++ linOps.\n\n    Parameters\n    ----------\n        linPy_to_linC: a dict for memoizing construction and storing\n            the C++ LinOps\n    '
    bfs_stack = [root_linPy]
    post_order_stack = []
    while bfs_stack:
        linPy = bfs_stack.pop()
        if linPy not in linPy_to_linC:
            post_order_stack.append(linPy)
            for arg in linPy.args:
                bfs_stack.append(arg)
            if isinstance(linPy.data, lo.LinOp):
                bfs_stack.append(linPy.data)
    while post_order_stack:
        linPy = post_order_stack.pop()
        make_linC_from_linPy(linPy, linPy_to_linC)