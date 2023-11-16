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
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.constants.parameter import is_param_free

class kron(AffAtom):
    """Kronecker product.
    """

    def __init__(self, lh_expr, rh_expr) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(kron, self).__init__(lh_expr, rh_expr)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        if False:
            while True:
                i = 10
        'Kronecker product of the two values.\n        '
        return np.kron(values[0], values[1])

    def validate_arguments(self) -> None:
        if False:
            print('Hello World!')
        'Checks that both arguments are vectors, and the first is constant.\n        '
        if not (self.args[0].is_constant() or self.args[1].is_constant()):
            raise ValueError('At least one argument to kron must be constant.')
        elif self.args[0].ndim != 2 or self.args[1].ndim != 2:
            raise ValueError('kron requires matrix arguments.')

    def shape_from_args(self) -> Tuple[int, int]:
        if False:
            return 10
        rows = self.args[0].shape[0] * self.args[1].shape[0]
        cols = self.args[0].shape[1] * self.args[1].shape[1]
        return (rows, cols)

    def is_atom_convex(self) -> bool:
        if False:
            print('Hello World!')
        'Is the atom convex?\n        '
        if u.scopes.dpp_scope_active():
            x = self.args[0]
            y = self.args[1]
            return (x.is_constant() or y.is_constant()) and (is_param_free(x) and is_param_free(y))
        else:
            return self.args[0].is_constant() or self.args[1].is_constant()

    def is_atom_concave(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the atom concave?\n        '
        return self.is_atom_convex()

    def sign_from_args(self) -> Tuple[bool, bool]:
        if False:
            print('Hello World!')
        'Same as times.\n        '
        return u.sign.mul_sign(self.args[0], self.args[1])

    def is_incr(self, idx) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the composition non-decreasing in argument idx?\n        '
        cst_loc = 0 if self.args[0].is_constant() else 1
        return self.args[cst_loc].is_nonneg()

    def is_decr(self, idx) -> bool:
        if False:
            while True:
                i = 10
        'Is the composition non-increasing in argument idx?\n        '
        cst_loc = 0 if self.args[0].is_constant() else 1
        return self.args[cst_loc].is_nonpos()

    def is_psd(self):
        if False:
            return 10
        'Check a *sufficient condition* that the expression is PSD,\n        by checking if both arguments are PSD or both are NSD.\n        '
        case1 = self.args[0].is_psd() and self.args[1].is_psd()
        case2 = self.args[0].is_nsd() and self.args[1].is_nsd()
        return case1 or case2

    def is_nsd(self):
        if False:
            return 10
        'Check a *sufficient condition* that the expression is NSD,\n        by checking if one argument is PSD and the other is NSD.\n        '
        case1 = self.args[0].is_psd() and self.args[1].is_nsd()
        case2 = self.args[0].is_nsd() and self.args[1].is_psd()
        return case1 or case2

    def graph_implementation(self, arg_objs, shape: Tuple[int, ...], data=None) -> Tuple[lo.LinOp, List[Constraint]]:
        if False:
            print('Hello World!')
        'Kronecker product of two matrices.\n\n        Parameters\n        ----------\n        arg_objs : list\n            LinOp for each argument.\n        shape : tuple\n            The shape of the resulting expression.\n        data :\n            Additional data required by the atom.\n\n        Returns\n        -------\n        tuple\n            (LinOp for objective, list of constraints)\n        '
        if self.args[0].is_constant():
            return (lu.kron_r(arg_objs[0], arg_objs[1], shape), [])
        else:
            return (lu.kron_l(arg_objs[0], arg_objs[1], shape), [])