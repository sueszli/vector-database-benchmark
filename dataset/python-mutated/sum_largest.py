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
import numpy as np
import scipy.sparse as sp
import cvxpy.interface as intf
from cvxpy.atoms.atom import Atom

class sum_largest(Atom):
    """Sum of the largest k values in the matrix X.
    """

    def __init__(self, x, k) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.k = k
        super(sum_largest, self).__init__(x)

    def validate_arguments(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Verify that k is a positive integer.\n        '
        if int(self.k) != self.k or self.k <= 0:
            raise ValueError('Second argument must be a positive integer.')
        super(sum_largest, self).validate_arguments()

    def numeric(self, values):
        if False:
            while True:
                i = 10
        'Returns the sum of the k largest entries of the matrix.\n        '
        value = values[0].flatten()
        indices = np.argsort(-value)[:int(self.k)]
        return value[indices].sum()

    def _grad(self, values):
        if False:
            for i in range(10):
                print('nop')
        'Gives the (sub/super)gradient of the atom w.r.t. each argument.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Args:\n            values: A list of numeric values for the arguments.\n\n        Returns:\n            A list of SciPy CSC sparse matrices or None.\n        '
        value = intf.from_2D_to_1D(values[0].flatten().T)
        indices = np.argsort(-value)[:int(self.k)]
        D = np.zeros((self.args[0].shape[0] * self.args[0].shape[1], 1))
        D[indices] = 1
        return [sp.csc_matrix(D)]

    def shape_from_args(self) -> Tuple[int, ...]:
        if False:
            while True:
                i = 10
        'Returns the (row, col) shape of the expression.\n        '
        return tuple()

    def sign_from_args(self) -> Tuple[bool, bool]:
        if False:
            while True:
                i = 10
        'Returns sign (is positive, is negative) of the expression.\n        '
        return (self.args[0].is_nonneg(), self.args[0].is_nonpos())

    def is_atom_convex(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the atom convex?\n        '
        return True

    def is_atom_concave(self) -> bool:
        if False:
            return 10
        'Is the atom concave?\n        '
        return False

    def is_incr(self, idx) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the composition non-decreasing in argument idx?\n        '
        return True

    def is_decr(self, idx) -> bool:
        if False:
            return 10
        'Is the composition non-increasing in argument idx?\n        '
        return False

    def is_pwl(self) -> bool:
        if False:
            return 10
        'Is the atom piecewise linear?\n        '
        return all((arg.is_pwl() for arg in self.args))

    def get_data(self):
        if False:
            i = 10
            return i + 15
        'Returns the parameter k.\n        '
        return [self.k]