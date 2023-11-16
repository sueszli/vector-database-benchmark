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
from __future__ import annotations
from typing import List, Tuple, Union
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.vec import vec
from cvxpy.constraints.constraint import Constraint

def diag(expr, k: int=0) -> Union['diag_mat', 'diag_vec']:
    if False:
        i = 10
        return i + 15
    'Extracts the diagonal from a matrix or makes a vector a diagonal matrix.\n\n    Parameters\n    ----------\n    expr : Expression or numeric constant\n        A vector or square matrix.\n\n    k : int\n        Diagonal in question. The default is 0.\n        Use k>0 for diagonals above the main diagonal,\n        and k<0 for diagonals below the main diagonal.\n\n    Returns\n    -------\n    Expression\n        An Expression representing the diagonal vector/matrix.\n    '
    expr = AffAtom.cast_to_const(expr)
    if expr.is_vector():
        return diag_vec(vec(expr), k)
    elif expr.ndim == 2 and expr.shape[0] == expr.shape[1]:
        assert abs(k) < expr.shape[0], 'Offset out of bounds.'
        return diag_mat(expr, k)
    else:
        raise ValueError('Argument to diag must be a vector or square matrix.')

class diag_vec(AffAtom):
    """Converts a vector into a diagonal matrix.
    """

    def __init__(self, expr, k: int=0) -> None:
        if False:
            while True:
                i = 10
        self.k = k
        super(diag_vec, self).__init__(expr)

    def get_data(self) -> list[int]:
        if False:
            print('Hello World!')
        return [self.k]

    def is_atom_log_log_convex(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the atom log-log convex?\n        '
        return True

    def is_atom_log_log_concave(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the atom log-log concave?\n        '
        return True

    def numeric(self, values):
        if False:
            while True:
                i = 10
        'Convert the vector constant into a diagonal matrix.\n        '
        return np.diag(values[0], k=self.k)

    def shape_from_args(self) -> Tuple[int, int]:
        if False:
            for i in range(10):
                print('nop')
        'A square matrix.\n        '
        rows = self.args[0].shape[0] + abs(self.k)
        return (rows, rows)

    def is_symmetric(self) -> bool:
        if False:
            print('Hello World!')
        'Is the expression symmetric?\n        '
        return self.k == 0

    def is_hermitian(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the expression hermitian?\n        '
        return self.k == 0

    def is_psd(self) -> bool:
        if False:
            print('Hello World!')
        'Is the expression a positive semidefinite matrix?\n        '
        return self.is_nonneg() and self.k == 0

    def is_nsd(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the expression a negative semidefinite matrix?\n        '
        return self.is_nonpos() and self.k == 0

    def graph_implementation(self, arg_objs, shape: Tuple[int, ...], data=None) -> Tuple[lo.LinOp, List[Constraint]]:
        if False:
            return 10
        'Convolve two vectors.\n\n        Parameters\n        ----------\n        arg_objs : list\n            LinExpr for each argument.\n        shape : tuple\n            The shape of the resulting expression.\n        data :\n            Additional data required by the atom.\n\n        Returns\n        -------\n        tuple\n            (LinOp for objective, list of constraints)\n        '
        return (lu.diag_vec(arg_objs[0], self.k), [])

class diag_mat(AffAtom):
    """Extracts the diagonal from a square matrix.
    """

    def __init__(self, expr, k: int=0) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.k = k
        super(diag_mat, self).__init__(expr)

    def get_data(self) -> list[int]:
        if False:
            print('Hello World!')
        return [self.k]

    def is_atom_log_log_convex(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the atom log-log convex?\n        '
        return True

    def is_atom_log_log_concave(self) -> bool:
        if False:
            return 10
        'Is the atom log-log concave?\n        '
        return True

    @AffAtom.numpy_numeric
    def numeric(self, values):
        if False:
            return 10
        'Extract the diagonal from a square matrix constant.\n        '
        return np.diag(values[0], k=self.k)

    def shape_from_args(self) -> Tuple[int]:
        if False:
            i = 10
            return i + 15
        'A column vector.\n        '
        (rows, _) = self.args[0].shape
        rows -= abs(self.k)
        return (rows,)

    def is_nonneg(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the expression nonnegative?\n        '
        return (self.args[0].is_nonneg() or self.args[0].is_psd()) and self.k == 0

    def graph_implementation(self, arg_objs, shape: Tuple[int, ...], data=None) -> Tuple[lo.LinOp, List[Constraint]]:
        if False:
            while True:
                i = 10
        'Extracts the diagonal of a matrix.\n\n        Parameters\n        ----------\n        arg_objs : list\n            LinExpr for each argument.\n        shape : tuple\n            The shape of the resulting expression.\n        data :\n            Additional data required by the atom.\n\n        Returns\n        -------\n        tuple\n            (LinOp for objective, list of constraints)\n        '
        return (lu.diag_mat(arg_objs[0], self.k), [])