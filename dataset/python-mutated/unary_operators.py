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
import operator as op
from typing import List, Tuple
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint

class UnaryOperator(AffAtom):
    """
    Base class for expressions involving unary operators.
    """

    def __init__(self, expr) -> None:
        if False:
            while True:
                i = 10
        super(UnaryOperator, self).__init__(expr)

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return self.OP_NAME + self.args[0].name()

    def numeric(self, values):
        if False:
            for i in range(10):
                print('nop')
        return self.OP_FUNC(values[0])

class NegExpression(UnaryOperator):
    """Negation of an expression.
    """
    OP_NAME = '-'
    OP_FUNC = op.neg

    def shape_from_args(self) -> Tuple[int, ...]:
        if False:
            print('Hello World!')
        'Returns the (row, col) shape of the expression.\n        '
        return self.args[0].shape

    def sign_from_args(self) -> Tuple[bool, bool]:
        if False:
            while True:
                i = 10
        'Returns sign (is positive, is negative) of the expression.\n        '
        return (self.args[0].is_nonpos(), self.args[0].is_nonneg())

    def is_incr(self, idx) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the composition non-decreasing in argument idx?\n        '
        return False

    def is_decr(self, idx) -> bool:
        if False:
            return 10
        'Is the composition non-increasing in argument idx?\n        '
        return True

    def is_symmetric(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the expression symmetric?\n        '
        return self.args[0].is_symmetric()

    def is_hermitian(self) -> bool:
        if False:
            return 10
        'Is the expression Hermitian?\n        '
        return self.args[0].is_hermitian()

    def graph_implementation(self, arg_objs, shape: Tuple[int, ...], data=None) -> Tuple[lo.LinOp, List[Constraint]]:
        if False:
            return 10
        'Negate the affine objective.\n\n        Parameters\n        ----------\n        arg_objs : list\n            LinExpr for each argument.\n        shape : tuple\n            The shape of the resulting expression.\n        data :\n            Additional data required by the atom.\n\n        Returns\n        -------\n        tuple\n            (LinOp for objective, list of constraints)\n        '
        return (lu.neg_expr(arg_objs[0]), [])