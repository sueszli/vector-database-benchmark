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
from typing import List, Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint

class conj(AffAtom):
    """Complex conjugate.
    """

    def __init__(self, expr) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(conj, self).__init__(expr)

    def numeric(self, values):
        if False:
            return 10
        'Convert the vector constant into a diagonal matrix.\n        '
        return np.conj(values[0])

    def shape_from_args(self) -> Tuple[int, ...]:
        if False:
            print('Hello World!')
        'Returns the shape of the expression.\n        '
        return self.args[0].shape

    def is_incr(self, idx) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the composition non-decreasing in argument idx?\n        '
        return False

    def is_decr(self, idx) -> bool:
        if False:
            print('Hello World!')
        'Is the composition non-increasing in argument idx?\n        '
        return False

    def is_symmetric(self) -> bool:
        if False:
            print('Hello World!')
        'Is the expression symmetric?\n        '
        return self.args[0].is_symmetric()

    def is_hermitian(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the expression Hermitian?\n        '
        return self.args[0].is_hermitian()

    def graph_implementation(self, arg_objs, shape: Tuple[int, ...], data=None) -> Tuple[lo.LinOp, List[Constraint]]:
        if False:
            for i in range(10):
                print('nop')
        'Multiply the linear expressions.\n\n        Parameters\n        ----------\n        arg_objs : list\n            LinExpr for each argument.\n        shape : tuple\n            The shape of the resulting expression.\n        data :\n            Additional data required by the atom.\n\n        Returns\n        -------\n        tuple\n            (LinOp for objective, list of constraints)\n        '
        return (arg_objs[0], [])