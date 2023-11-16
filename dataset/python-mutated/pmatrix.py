"""
Matrix designed for fast multiplication by permutation and block-diagonal ones.
"""
import numpy as np
from .layer import Layer1Q, Layer2Q

class PMatrix:
    """
    Wrapper around a matrix that enables fast multiplication by permutation
    matrices and block-diagonal ones.
    """

    def __init__(self, num_qubits: int):
        if False:
            return 10
        '\n        Initializes the internal structures of this object but does not set\n        the matrix yet.\n\n        Args:\n            num_qubits: number of qubits.\n        '
        dim = 2 ** num_qubits
        self._mat = np.empty(0)
        self._dim = dim
        self._temp_g2x2 = np.zeros((2, 2), dtype=np.complex128)
        self._temp_g4x4 = np.zeros((4, 4), dtype=np.complex128)
        self._temp_2x2 = self._temp_g2x2.copy()
        self._temp_4x4 = self._temp_g4x4.copy()
        self._identity_perm = np.arange(dim, dtype=np.int64)
        self._left_perm = self._identity_perm.copy()
        self._right_perm = self._identity_perm.copy()
        self._temp_perm = self._identity_perm.copy()
        self._temp_slice_dim_x_2 = np.zeros((dim, 2), dtype=np.complex128)
        self._temp_slice_dim_x_4 = np.zeros((dim, 4), dtype=np.complex128)
        self._idx_mat = self._init_index_matrix(dim)
        self._temp_block_diag = np.zeros(self._idx_mat.shape, dtype=np.complex128)

    def set_matrix(self, mat: np.ndarray):
        if False:
            print('Hello World!')
        '\n        Copies specified matrix to internal storage. Once the matrix\n        is set, the object is ready for use.\n\n        **Note**, the matrix will be copied, mind the size issues.\n\n        Args:\n            mat: matrix we want to multiply on the left and on the right by\n                 layer matrices.\n        '
        if self._mat.size == 0:
            self._mat = mat.copy()
        else:
            np.copyto(self._mat, mat)

    @staticmethod
    def _init_index_matrix(dim: int) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Fast multiplication can be implemented by picking up a subset of\n        entries in a sparse matrix.\n\n        Args:\n            dim: problem dimensionality.\n\n        Returns:\n            2d-array of indices for the fast multiplication.\n        '
        all_idx = np.arange(dim * dim, dtype=np.int64).reshape(dim, dim)
        idx = np.full((dim // 4, 4 * 4), fill_value=0, dtype=np.int64)
        b = np.full((4, 4), fill_value=0, dtype=np.int64)
        for i in range(0, dim, 4):
            b[:, :] = all_idx[i:i + 4, i:i + 4]
            idx[i // 4, :] = b.T.ravel()
        return idx

    def mul_right_q1(self, layer: Layer1Q, temp_mat: np.ndarray, dagger: bool):
        if False:
            print('Hello World!')
        '\n        Multiplies ``NxN`` matrix, wrapped by this object, by a 1-qubit layer\n        matrix of the right, where ``N`` is the actual size of matrices involved,\n        ``N = 2^{num. of qubits}``.\n\n        Args:\n            layer: 1-qubit layer, i.e. the layer with just one non-trivial\n                   1-qubit gate and other gates are just identity operators.\n            temp_mat: a temporary NxN matrix used as a workspace.\n            dagger: if true, the right-hand side matrix will be taken as\n                    conjugate transposed.\n        '
        (gmat, perm, inv_perm) = layer.get_attr()
        mat = self._mat
        dim = perm.size
        np.take(mat, np.take(self._right_perm, perm, out=self._temp_perm), axis=1, out=temp_mat)
        gmat_right = np.conj(gmat, out=self._temp_g2x2).T if dagger else gmat
        for i in range(0, dim, 2):
            mat[:, i:i + 2] = np.dot(temp_mat[:, i:i + 2], gmat_right, out=self._temp_slice_dim_x_2)
        self._right_perm[:] = inv_perm

    def mul_right_q2(self, layer: Layer2Q, temp_mat: np.ndarray, dagger: bool=True):
        if False:
            print('Hello World!')
        '\n        Multiplies ``NxN`` matrix, wrapped by this object, by a 2-qubit layer\n        matrix on the right, where ``N`` is the actual size of matrices involved,\n        ``N = 2^{num. of qubits}``.\n\n        Args:\n            layer: 2-qubit layer, i.e. the layer with just one non-trivial\n                   2-qubit gate and other gates are just identity operators.\n            temp_mat: a temporary NxN matrix used as a workspace.\n            dagger: if true, the right-hand side matrix will be taken as\n                    conjugate transposed.\n        '
        (gmat, perm, inv_perm) = layer.get_attr()
        mat = self._mat
        dim = perm.size
        np.take(mat, np.take(self._right_perm, perm, out=self._temp_perm), axis=1, out=temp_mat)
        gmat_right = np.conj(gmat, out=self._temp_g4x4).T if dagger else gmat
        for i in range(0, dim, 4):
            mat[:, i:i + 4] = np.dot(temp_mat[:, i:i + 4], gmat_right, out=self._temp_slice_dim_x_4)
        self._right_perm[:] = inv_perm

    def mul_left_q1(self, layer: Layer1Q, temp_mat: np.ndarray):
        if False:
            for i in range(10):
                print('nop')
        '\n        Multiplies ``NxN`` matrix, wrapped by this object, by a 1-qubit layer\n        matrix of the left, where ``dim`` is the actual size of matrices involved,\n        ``dim = 2^{num. of qubits}``.\n\n        Args:\n            layer: 1-qubit layer, i.e. the layer with just one non-trivial\n                   1-qubit gate and other gates are just identity operators.\n            temp_mat: a temporary NxN matrix used as a workspace.\n        '
        mat = self._mat
        (gmat, perm, inv_perm) = layer.get_attr()
        dim = perm.size
        np.take(mat, np.take(self._left_perm, perm, out=self._temp_perm), axis=0, out=temp_mat)
        if dim > 512:
            for i in range(0, dim, 2):
                np.dot(gmat, temp_mat[i:i + 2, :], out=mat[i:i + 2, :])
        else:
            half = dim // 2
            np.copyto(mat.reshape((2, half, dim)), np.swapaxes(temp_mat.reshape((half, 2, dim)), 0, 1))
            np.dot(gmat, mat.reshape(2, -1), out=temp_mat.reshape(2, -1))
            np.copyto(mat.reshape((half, 2, dim)), np.swapaxes(temp_mat.reshape((2, half, dim)), 0, 1))
        self._left_perm[:] = inv_perm

    def mul_left_q2(self, layer: Layer2Q, temp_mat: np.ndarray):
        if False:
            print('Hello World!')
        '\n        Multiplies ``NxN`` matrix, wrapped by this object, by a 2-qubit layer\n        matrix on the left, where ``dim`` is the actual size of matrices involved,\n        ``dim = 2^{num. of qubits}``.\n\n        Args:\n            layer: 2-qubit layer, i.e. the layer with just one non-trivial\n                   2-qubit gate and other gates are just identity operators.\n            temp_mat: a temporary NxN matrix used as a workspace.\n        '
        mat = self._mat
        (gmat, perm, inv_perm) = layer.get_attr()
        dim = perm.size
        np.take(mat, np.take(self._left_perm, perm, out=self._temp_perm), axis=0, out=temp_mat)
        if dim > 512:
            for i in range(0, dim, 4):
                np.dot(gmat, temp_mat[i:i + 4, :], out=mat[i:i + 4, :])
        else:
            half = dim // 4
            np.copyto(mat.reshape((4, half, dim)), np.swapaxes(temp_mat.reshape((half, 4, dim)), 0, 1))
            np.dot(gmat, mat.reshape(4, -1), out=temp_mat.reshape(4, -1))
            np.copyto(mat.reshape((half, 4, dim)), np.swapaxes(temp_mat.reshape((4, half, dim)), 0, 1))
        self._left_perm[:] = inv_perm

    def product_q1(self, layer: Layer1Q, tmp1: np.ndarray, tmp2: np.ndarray) -> np.complex128:
        if False:
            print('Hello World!')
        '\n        Computes and returns: ``Trace(mat @ C) = Trace(mat @ P^T @ gmat @ P) =\n        Trace((P @ mat @ P^T) @ gmat) = Trace(C @ (P @ mat @ P^T)) =\n        vec(gmat^T)^T @ vec(P @ mat @ P^T)``, where mat is ``NxN`` matrix wrapped\n        by this object, ``C`` is matrix representation of the layer ``L``, and gmat\n        is 2x2 matrix of underlying 1-qubit gate.\n\n        **Note**: matrix of this class must be finalized beforehand.\n\n        Args:\n            layer: 1-qubit layer.\n            tmp1: temporary, external matrix used as a workspace.\n            tmp2: temporary, external matrix used as a workspace.\n\n        Returns:\n            trace of the matrix product.\n        '
        mat = self._mat
        (gmat, perm, _) = layer.get_attr()
        np.take(np.take(mat, perm, axis=0, out=tmp1), perm, axis=1, out=tmp2)
        (gmat_t, tmp3) = (self._temp_g2x2, self._temp_2x2)
        np.copyto(gmat_t, gmat.T)
        _sum = 0.0
        for i in range(0, mat.shape[0], 2):
            tmp3[:, :] = tmp2[i:i + 2, i:i + 2]
            _sum += np.dot(gmat_t.ravel(), tmp3.ravel())
        return np.complex128(_sum)

    def product_q2(self, layer: Layer2Q, tmp1: np.ndarray, tmp2: np.ndarray) -> np.complex128:
        if False:
            i = 10
            return i + 15
        '\n        Computes and returns: ``Trace(mat @ C) = Trace(mat @ P^T @ gmat @ P) =\n        Trace((P @ mat @ P^T) @ gmat) = Trace(C @ (P @ mat @ P^T)) =\n        vec(gmat^T)^T @ vec(P @ mat @ P^T)``, where mat is ``NxN`` matrix wrapped\n        by this object, ``C`` is matrix representation of the layer ``L``, and gmat\n        is 4x4 matrix of underlying 2-qubit gate.\n\n        **Note**: matrix of this class must be finalized beforehand.\n\n        Args:\n            layer: 2-qubit layer.\n            tmp1: temporary, external matrix used as a workspace.\n            tmp2: temporary, external matrix used as a workspace.\n\n        Returns:\n            trace of the matrix product.\n        '
        mat = self._mat
        (gmat, perm, _) = layer.get_attr()
        np.take(np.take(mat, perm, axis=0, out=tmp1), perm, axis=1, out=tmp2)
        bldia = self._temp_block_diag
        np.take(tmp2.ravel(), self._idx_mat.ravel(), axis=0, out=bldia.ravel())
        bldia *= gmat.reshape(-1, gmat.size)
        return np.complex128(np.sum(bldia))

    def finalize(self, temp_mat: np.ndarray) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Applies the left (row) and right (column) permutations to the matrix.\n        at the end of computation process.\n\n        Args:\n            temp_mat: temporary, external matrix.\n\n        Returns:\n            finalized matrix with all transformations applied.\n        '
        mat = self._mat
        np.take(mat, self._left_perm, axis=0, out=temp_mat)
        np.take(temp_mat, self._right_perm, axis=1, out=mat)
        self._left_perm[:] = self._identity_perm
        self._right_perm[:] = self._identity_perm
        return self._mat