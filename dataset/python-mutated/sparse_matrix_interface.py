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
import numpy as np
import scipy.sparse as sp
from cvxpy.interface.numpy_interface.ndarray_interface import NDArrayInterface

class SparseMatrixInterface(NDArrayInterface):
    """
    An interface to convert constant values to the scipy sparse CSC class.
    """
    TARGET_MATRIX = sp.csc_matrix

    @NDArrayInterface.scalar_const
    def const_to_matrix(self, value, convert_scalars: bool=False):
        if False:
            print('Hello World!')
        'Convert an arbitrary value into a matrix of type self.target_matrix.\n\n        Args:\n            value: The constant to be converted.\n            convert_scalars: Should scalars be converted?\n\n        Returns:\n            A matrix of type self.target_matrix or a scalar.\n        '
        if isinstance(value, list):
            return sp.csc_matrix(value, dtype=np.double).T
        if value.dtype in [np.double, complex]:
            dtype = value.dtype
        else:
            dtype = np.double
        return sp.csc_matrix(value, dtype=dtype)

    def identity(self, size):
        if False:
            return 10
        'Return an identity matrix.\n        '
        return sp.eye(size, size, format='csc')

    def size(self, matrix):
        if False:
            while True:
                i = 10
        'Return the dimensions of the matrix.\n        '
        return matrix.shape

    def scalar_value(self, matrix):
        if False:
            for i in range(10):
                print('nop')
        'Get the value of the passed matrix, interpreted as a scalar.\n        '
        return matrix[0, 0]

    def zeros(self, rows, cols):
        if False:
            print('Hello World!')
        "Return a matrix with all 0's.\n        "
        return sp.csc_matrix((rows, cols), dtype='float64')

    def reshape(self, matrix, size):
        if False:
            while True:
                i = 10
        'Change the shape of the matrix.\n        '
        matrix = matrix.todense()
        matrix = super(SparseMatrixInterface, self).reshape(matrix, size)
        return self.const_to_matrix(matrix, convert_scalars=True)

    def block_add(self, matrix, block, vert_offset, horiz_offset, rows, cols, vert_step: int=1, horiz_step: int=1) -> None:
        if False:
            while True:
                i = 10
        'Add the block to a slice of the matrix.\n\n        Args:\n            matrix: The matrix the block will be added to.\n            block: The matrix/scalar to be added.\n            vert_offset: The starting row for the matrix slice.\n            horiz_offset: The starting column for the matrix slice.\n            rows: The height of the block.\n            cols: The width of the block.\n            vert_step: The row step size for the matrix slice.\n            horiz_step: The column step size for the matrix slice.\n        '
        block = self._format_block(matrix, block, rows, cols)
        slice_ = [slice(vert_offset, rows + vert_offset, vert_step), slice(horiz_offset, horiz_offset + cols, horiz_step)]
        matrix[slice_[0], slice_[1]] = matrix[slice_[0], slice_[1]] + block