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

THIS FILE IS DEPRECATED AND MAY BE REMOVED WITHOUT WARNING!
DO NOT CALL THESE FUNCTIONS IN YOUR CODE!
"""
import numpy as np
import scipy.sparse as sp

def get_row_nnz(mat, row):
    if False:
        while True:
            i = 10
    'Return the number of nonzeros in row.\n    '
    return mat.indptr[row + 1] - mat.indptr[row]

def compress_matrix(A, b, equil_eps: float=1e-10):
    if False:
        print('Hello World!')
    'Compresses A and b by eliminating redundant rows.\n\n    Identifies rows that are multiples of another row.\n    Reduces A and b to C = PA, d = Pb, where P has one\n    nonzero per row.\n\n    Parameters\n    ----------\n    A : SciPy CSR matrix\n        The constraints matrix to compress.\n    b : NumPy 1D array\n        The vector associated with the constraints matrix.\n    equil_eps : float, optional\n        Standard for considering two numbers equivalent.\n\n    Returns\n    -------\n    tuple\n        The tuple (A, b, P) where A and b are compressed according to P.\n    '
    P_V = []
    P_I = []
    P_J = []
    row_to_keep = []
    sparsity_to_row = {}
    prev_ptr = A.indptr[0]
    for row_num in range(A.shape[0]):
        keep_row = True
        ptr = A.indptr[row_num + 1]
        pattern = tuple(A.indices[prev_ptr:ptr])
        nnz = ptr - prev_ptr
        if nnz == 0 or np.linalg.norm(A.data[prev_ptr:ptr]) < equil_eps:
            keep_row = False
            P_V.append(0.0)
            P_I.append(row_num)
            P_J.append(0)
        elif pattern in sparsity_to_row and nnz == get_row_nnz(A, sparsity_to_row[pattern][0]):
            row_matches = sparsity_to_row[pattern]
            for row_match in row_matches:
                cur_vals = A.data[prev_ptr:ptr]
                prev_match_ptr = A.indptr[row_match]
                match_ptr = A.indptr[row_match + 1]
                match_vals = A.data[prev_match_ptr:match_ptr]
                ratio = cur_vals / match_vals
                if np.ptp(ratio) < equil_eps and abs(ratio[0] - b[row_num] / b[row_match]) < equil_eps:
                    keep_row = False
                    P_V.append(ratio[0])
                    P_I.append(row_num)
                    P_J.append(row_match)
            if keep_row:
                sparsity_to_row[pattern].append(row_num)
        else:
            sparsity_to_row[pattern] = [row_num]
        if keep_row:
            row_to_keep.append(row_num)
            P_V.append(1.0)
            P_I.append(row_num)
            P_J.append(len(row_to_keep) - 1)
    cols = max(len(row_to_keep), 1)
    P = sp.coo_matrix((P_V, (P_I, P_J)), (A.shape[0], cols))
    A_compr = A[row_to_keep, :]
    b_compr = b[row_to_keep]
    return (A_compr, b_compr, P)