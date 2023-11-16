"""
Copyright, the CVXPY authors

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
import cvxpy.utilities as u
from cvxpy.atoms.atom import Atom
from cvxpy.expressions import cvxtypes

class gmatmul(Atom):
    """Geometric matrix multiplication; :math:`A \\mathbin{\\diamond} X`.

    For :math:`A \\in \\mathbf{R}^{m \\times n}` and
    :math:`X \\in \\mathbf{R}^{n \\times p}_{++}`, this atom represents

    .. math::

        \\left[\\begin{array}{ccc}
         \\prod_{j=1}^n X_{j1}^{A_{1j}} & \\cdots & \\prod_{j=1}^n X_{pj}^{A_{1j}} \\\\
         \\vdots &  & \\vdots \\\\
         \\prod_{j=1}^n X_{j1}^{A_{mj}} & \\cdots & \\prod_{j=1}^n X_{pj}^{A_{mj}}
        \\end{array}\\right]

    This atom is log-log affine (in :math:`X`).

    Parameters
    ----------
    A : cvxpy.Expression
        A constant matrix.
    X : cvxpy.Expression
        A positive matrix.
    """

    def __init__(self, A, X) -> None:
        if False:
            i = 10
            return i + 15
        self.A = Atom.cast_to_const(A)
        super(gmatmul, self).__init__(X)

    def numeric(self, values):
        if False:
            i = 10
            return i + 15
        'Geometric matrix multiplication.\n        '
        logX = np.log(values[0])
        return np.exp(self.A.value @ logX)

    def name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return '%s(%s, %s)' % (self.__class__.__name__, self.A, self.args[0])

    def validate_arguments(self) -> None:
        if False:
            while True:
                i = 10
        'Raises an error if the arguments are invalid.\n        '
        super(gmatmul, self).validate_arguments()
        if not self.A.is_constant():
            raise ValueError('gmatmul(A, X) requires that A be constant.')
        if self.A.parameters() and (not isinstance(self.A, cvxtypes.parameter())):
            raise ValueError('gmatmul(A, X) requires that A be a Constant or a Parameter.')
        if not self.args[0].is_pos():
            raise ValueError('gmatmul(A, X) requires that X be positive.')

    def shape_from_args(self) -> Tuple[int, ...]:
        if False:
            while True:
                i = 10
        'Returns the (row, col) shape of the expression.\n        '
        return u.shape.mul_shapes(self.A.shape, self.args[0].shape)

    def get_data(self):
        if False:
            i = 10
            return i + 15
        'Returns info needed to reconstruct the expression besides the args.\n        '
        return [self.A]

    def sign_from_args(self) -> Tuple[bool, bool]:
        if False:
            while True:
                i = 10
        'Returns sign (is positive, is negative) of the expression.\n        '
        return (True, False)

    def is_atom_convex(self) -> bool:
        if False:
            return 10
        'Is the atom convex?\n        '
        return False

    def is_atom_concave(self) -> bool:
        if False:
            return 10
        'Is the atom concave?\n        '
        return False

    def parameters(self):
        if False:
            while True:
                i = 10
        return self.args[0].parameters() + self.A.parameters()

    def is_atom_log_log_convex(self) -> bool:
        if False:
            return 10
        'Is the atom log-log convex?\n        '
        if u.scopes.dpp_scope_active():
            X = self.args[0]
            A = self.A
            return not (X.parameters() and A.parameters())
        else:
            return True

    def is_atom_log_log_concave(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the atom log-log concave?\n        '
        return self.is_atom_log_log_convex()

    def is_incr(self, idx) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the composition non-decreasing in argument idx?\n        '
        return self.A.is_nonneg()

    def is_decr(self, idx) -> bool:
        if False:
            return 10
        'Is the composition non-increasing in argument idx?\n        '
        return self.A.is_nonpos()

    def _grad(self, values) -> None:
        if False:
            while True:
                i = 10
        return None