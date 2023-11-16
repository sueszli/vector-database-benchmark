"""
Copyright 2018 Akshay Agrawal

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
from cvxpy.atoms.atom import Atom

def resolvent(X, s: float):
    if False:
        for i in range(10):
            print('nop')
    'The resolvent of a positive matrix, :math:`(sI - X)^{-1}`.\n\n    For an elementwise positive matrix :math:`X` and a positive scalar\n    :math:`s`, this atom computes\n\n    .. math::\n\n        (sI - X)^{-1},\n\n    and it enforces the constraint that the spectral radius of :math:`X/s`\n    is at most :math:`1`.\n\n    This atom is log-log convex.\n\n    Parameters\n    ----------\n    X : cvxpy.Expression\n        A positive square matrix.\n    s : cvxpy.Expression or numeric\n        A positive scalar.\n    '
    return 1.0 / s * eye_minus_inv(X / s)

class eye_minus_inv(Atom):
    """The unity resolvent of a positive matrix, :math:`(I - X)^{-1}`.

    For an elementwise positive matrix :math:`X`, this atom represents

    .. math::

        (I - X)^{-1},

    and it enforces the constraint that the spectral radius of :math:`X`
    is at most :math:`1`.

    This atom is log-log convex.

    Parameters
    ----------
    X : cvxpy.Expression
        A positive square matrix.
    """

    def __init__(self, X) -> None:
        if False:
            while True:
                i = 10
        super(eye_minus_inv, self).__init__(X)
        if len(X.shape) != 2 or X.shape[0] != X.shape[1]:
            raise ValueError('The argument to `eye_minus_inv` must be a square matrix, received ', X)
        self.args[0] = X

    def numeric(self, values):
        if False:
            i = 10
            return i + 15
        return np.linalg.inv(np.eye(self.args[0].shape[0]) - values[0])

    def name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return '%s(%s)' % (self.__class__.__name__, self.args[0])

    def shape_from_args(self) -> Tuple[int, ...]:
        if False:
            while True:
                i = 10
        'Returns the (row, col) shape of the expression.\n        '
        return self.args[0].shape

    def sign_from_args(self) -> Tuple[bool, bool]:
        if False:
            while True:
                i = 10
        'Returns sign (is positive, is negative) of the expression.\n        '
        return (True, False)

    def is_atom_convex(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the atom convex?\n        '
        return False

    def is_atom_concave(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the atom concave?\n        '
        return False

    def is_atom_log_log_convex(self) -> bool:
        if False:
            return 10
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
            for i in range(10):
                print('nop')
        'Is the composition non-decreasing in argument idx?\n        '
        return False

    def is_decr(self, idx) -> bool:
        if False:
            return 10
        'Is the composition non-increasing in argument idx?\n        '
        return False

    def _grad(self, values) -> None:
        if False:
            return 10
        return None