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
import numpy
import scipy.sparse
from .. import base_matrix_interface as base
COMPLEX_TYPES = [complex, numpy.complex64, numpy.complex128]

class NDArrayInterface(base.BaseMatrixInterface):
    """
    An interface to convert constant values to the numpy ndarray class.
    """
    TARGET_MATRIX = numpy.ndarray

    def const_to_matrix(self, value, convert_scalars: bool=False):
        if False:
            i = 10
            return i + 15
        'Convert an arbitrary value into a matrix of type self.target_matrix.\n\n        Args:\n            value: The constant to be converted.\n            convert_scalars: Should scalars be converted?\n\n        Returns:\n            A matrix of type self.target_matrix or a scalar.\n        '
        if scipy.sparse.issparse(value):
            result = value.A
        elif isinstance(value, numpy.matrix):
            result = value.A
        elif isinstance(value, list):
            result = numpy.asarray(value).T
        else:
            result = numpy.asarray(value)
        if result.dtype in [numpy.float64] + COMPLEX_TYPES:
            return result
        else:
            return result.astype(numpy.float64)

    def identity(self, size):
        if False:
            return 10
        return numpy.eye(size)

    def shape(self, matrix) -> Tuple[int, ...]:
        if False:
            for i in range(10):
                print('nop')
        return tuple((int(d) for d in matrix.shape))

    def size(self, matrix):
        if False:
            i = 10
            return i + 15
        'Returns the number of elements in the matrix.\n        '
        return numpy.prod(self.shape(matrix))

    def scalar_value(self, matrix):
        if False:
            i = 10
            return i + 15
        return matrix.item()

    def scalar_matrix(self, value, shape: Tuple[int, ...]):
        if False:
            print('Hello World!')
        return numpy.zeros(shape, dtype='float64') + value

    def reshape(self, matrix, size):
        if False:
            while True:
                i = 10
        return numpy.reshape(matrix, size, order='F')