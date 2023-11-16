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
import scipy as scipy
import scipy.sparse as sp
from cvxpy.atoms.atom import Atom
from cvxpy.constraints.constraint import Constraint

class quad_over_lin(Atom):
    """ :math:`(sum_{ij}X^2_{ij})/y`

    """
    _allow_complex = True

    def __init__(self, x, y) -> None:
        if False:
            i = 10
            return i + 15
        super(quad_over_lin, self).__init__(x, y)

    @Atom.numpy_numeric
    def numeric(self, values):
        if False:
            for i in range(10):
                print('nop')
        'Returns the sum of the entries of x squared over y.\n        '
        if self.args[0].is_complex():
            return (np.square(values[0].imag) + np.square(values[0].real)).sum() / values[1]
        return np.square(values[0]).sum() / values[1]

    def _domain(self) -> List[Constraint]:
        if False:
            for i in range(10):
                print('nop')
        'Returns constraints describing the domain of the node.\n        '
        return [self.args[1] >= 0]

    def _grad(self, values):
        if False:
            for i in range(10):
                print('nop')
        'Gives the (sub/super)gradient of the atom w.r.t. each argument.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Args:\n            values: A list of numeric values for the arguments.\n\n        Returns:\n            A list of SciPy CSC sparse matrices or None.\n        '
        X = values[0]
        y = values[1]
        if y <= 0:
            return [None, None]
        else:
            if self.args[0].is_complex():
                Dy = -(np.square(X.real) + np.square(X.imag)).sum() / np.square(y)
            else:
                Dy = -np.square(X).sum() / np.square(y)
            Dy = sp.csc_matrix(Dy)
            DX = 2.0 * X / y
            DX = np.reshape(DX, (self.args[0].size, 1))
            DX = scipy.sparse.csc_matrix(DX)
            return [DX, Dy]

    def shape_from_args(self) -> Tuple[int, ...]:
        if False:
            return 10
        'Returns the (row, col) shape of the expression.\n        '
        return tuple()

    def sign_from_args(self) -> Tuple[bool, bool]:
        if False:
            print('Hello World!')
        'Returns sign (is positive, is negative) of the expression.\n        '
        return (True, False)

    def is_atom_convex(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the atom convex?\n        '
        return True

    def is_atom_concave(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the atom concave?\n        '
        return False

    def is_atom_log_log_convex(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the atom log-log convex?\n        '
        return True

    def is_atom_log_log_concave(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the atom log-log concave?\n        '
        return False

    def is_incr(self, idx) -> bool:
        if False:
            return 10
        'Is the composition non-decreasing in argument idx?\n        '
        return idx == 0 and self.args[idx].is_nonneg()

    def is_decr(self, idx) -> bool:
        if False:
            print('Hello World!')
        'Is the composition non-increasing in argument idx?\n        '
        return idx == 0 and self.args[idx].is_nonpos() or idx == 1

    def validate_arguments(self) -> None:
        if False:
            return 10
        'Check dimensions of arguments.\n        '
        if not self.args[1].is_scalar():
            raise ValueError('The second argument to quad_over_lin must be a scalar.')
        if self.args[1].is_complex():
            raise ValueError('The second argument to quad_over_lin cannot be complex.')
        super(quad_over_lin, self).validate_arguments()

    def is_quadratic(self) -> bool:
        if False:
            return 10
        'Quadratic if x is affine and y is constant.\n        '
        return self.args[0].is_affine() and self.args[1].is_constant()

    def has_quadratic_term(self) -> bool:
        if False:
            print('Hello World!')
        'A quadratic term if y is constant.\n        '
        return self.args[1].is_constant()

    def is_qpwa(self) -> bool:
        if False:
            while True:
                i = 10
        'Quadratic of piecewise affine if x is PWL and y is constant.\n        '
        return self.args[0].is_pwl() and self.args[1].is_constant()