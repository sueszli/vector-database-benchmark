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
from munkres import Munkres
import cvxpy.lin_ops.lin_utils as lu
from .boolean import Boolean

class Assign(Boolean):
    """ An assignment matrix. """

    def __init__(self, rows, cols, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        assert rows >= cols
        super(Assign, self).__init__(*args, rows=rows, cols=cols, **kwargs)

    def init_z(self):
        if False:
            i = 10
            return i + 15
        self.z.value = np.ones(self.size) / self.size[1]

    def _round(self, matrix):
        if False:
            for i in range(10):
                print('nop')
        m = Munkres()
        lists = self.matrix_to_lists(matrix)
        indexes = m.compute(lists)
        matrix *= 0
        for (row, column) in indexes:
            matrix[row, column] = 1
        return matrix

    def matrix_to_lists(self, matrix):
        if False:
            while True:
                i = 10
        'Convert a matrix to a list of lists.\n        '
        (rows, cols) = matrix.shape
        lists = []
        for i in range(rows):
            lists.append(matrix[i, :].tolist()[0])
        return lists

    def _fix(self, matrix):
        if False:
            for i in range(10):
                print('nop')
        return [self == matrix]

    def canonicalize(self):
        if False:
            i = 10
            return i + 15
        (obj, constraints) = super(Assign, self).canonicalize()
        shape = (self.size[1], 1)
        one_row_vec = lu.create_const(np.ones(shape), shape)
        shape = (1, self.size[0])
        one_col_vec = lu.create_const(np.ones(shape), shape)
        row_sum = lu.rmul_expr(obj, one_row_vec, (self.size[0], 1))
        constraints += [lu.create_leq(row_sum, lu.transpose(one_col_vec))]
        col_sum = lu.mul_expr(one_col_vec, obj, (1, self.size[1]))
        constraints += [lu.create_eq(col_sum, lu.transpose(one_row_vec))]
        return (obj, constraints)