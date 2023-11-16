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
import abc
from typing import Tuple
import numpy as np
import cvxpy.interface.matrix_utilities

class BaseMatrixInterface:
    """
    An interface between constants' internal values
    and the target matrix used internally.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def const_to_matrix(self, value, convert_scalars: bool=False):
        if False:
            i = 10
            return i + 15
        'Convert an arbitrary value into a matrix of type self.target_matrix.\n\n        Args:\n            value: The constant to be converted.\n            convert_scalars: Should scalars be converted?\n\n        Returns:\n            A matrix of type self.target_matrix or a scalar.\n        '
        raise NotImplementedError()

    @staticmethod
    def scalar_const(converter):
        if False:
            while True:
                i = 10

        def new_converter(self, value, convert_scalars: bool=False):
            if False:
                for i in range(10):
                    print('nop')
            if not convert_scalars and cvxpy.interface.matrix_utilities.is_scalar(value):
                return cvxpy.interface.matrix_utilities.scalar_value(value)
            else:
                return converter(self, value)
        return new_converter

    @abc.abstractmethod
    def identity(self, size):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def size(self, matrix):
        if False:
            return 10
        return np.prod(self.shape(matrix), dtype=int)

    @abc.abstractmethod
    def shape(self, matrix):
        if False:
            return 10
        raise NotImplementedError()

    @abc.abstractmethod
    def scalar_value(self, matrix):
        if False:
            return 10
        raise NotImplementedError()

    def zeros(self, shape: Tuple[int, ...]):
        if False:
            while True:
                i = 10
        return self.scalar_matrix(0, shape)

    def ones(self, shape: Tuple[int, ...]):
        if False:
            return 10
        return self.scalar_matrix(1, shape)

    @abc.abstractmethod
    def scalar_matrix(self, value, shape: Tuple[int, ...]):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def index(self, matrix, key):
        if False:
            for i in range(10):
                print('nop')
        value = matrix[key]
        if cvxpy.interface.matrix_utilities.shape(value) == (1, 1):
            return cvxpy.interface.matrix_utilities.scalar_value(value)
        else:
            return value

    @abc.abstractmethod
    def reshape(self, matrix, shape: Tuple[int, ...]):
        if False:
            return 10
        raise NotImplementedError()

    def block_add(self, matrix, block, vert_offset, horiz_offset, rows, cols, vert_step: int=1, horiz_step: int=1) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Add the block to a slice of the matrix.\n\n        Args:\n            matrix: The matrix the block will be added to.\n            block: The matrix/scalar to be added.\n            vert_offset: The starting row for the matrix slice.\n            horiz_offset: The starting column for the matrix slice.\n            rows: The height of the block.\n            cols: The width of the block.\n            vert_step: The row step size for the matrix slice.\n            horiz_step: The column step size for the matrix slice.\n        '
        block = self._format_block(matrix, block, rows, cols)
        matrix[vert_offset:rows + vert_offset:vert_step, horiz_offset:horiz_offset + cols:horiz_step] += block

    def _format_block(self, matrix, block, rows, cols):
        if False:
            print('Hello World!')
        'Formats the block for block_add.\n\n        Args:\n            matrix: The matrix the block will be added to.\n            block: The matrix/scalar to be added.\n            rows: The height of the block.\n            cols: The width of the block.\n        '
        if cvxpy.interface.matrix_utilities.is_scalar(block):
            block = self.scalar_matrix(cvxpy.interface.matrix_utilities.scalar_value(block), rows, cols)
        elif cvxpy.interface.matrix_utilities.is_vector(block) and cols > 1:
            block = self.reshape(block, (rows, cols))
        elif not cvxpy.interface.matrix_utilities.is_vector(block) and cols == 1:
            block = self.reshape(block, (rows, cols))
        elif type(block) != type(matrix):
            block = self.const_to_matrix(block)
        return block