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
from typing import Tuple
import numpy as np
import scipy.sparse as sp
from cvxpy.interface.numpy_interface.ndarray_interface import NDArrayInterface

class MatrixInterface(NDArrayInterface):
    """
    An interface to convert constant values to the numpy matrix class.
    """
    TARGET_MATRIX = np.matrix

    @NDArrayInterface.scalar_const
    def const_to_matrix(self, value, convert_scalars: bool=False):
        if False:
            for i in range(10):
                print('nop')
        'Convert an arbitrary value into a matrix of type self.target_matrix.\n\n        Args:\n            value: The constant to be converted.\n            convert_scalars: Should scalars be converted?\n\n        Returns:\n            A matrix of type self.target_matrix or a scalar.\n        '
        if isinstance(value, list) or (isinstance(value, np.ndarray) and value.ndim == 1):
            value = np.asmatrix(value, dtype='float64').T
        elif sp.issparse(value):
            value = value.todense()
        return np.asmatrix(value, dtype='float64')

    def identity(self, size):
        if False:
            return 10
        return np.asmatrix(np.eye(size))

    def scalar_matrix(self, value, shape: Tuple[int, ...]):
        if False:
            i = 10
            return i + 15
        mat = np.zeros(shape, dtype='float64') + value
        return np.asmatrix(mat)

    def reshape(self, matrix, size):
        if False:
            for i in range(10):
                print('nop')
        return np.reshape(matrix, size, order='F')