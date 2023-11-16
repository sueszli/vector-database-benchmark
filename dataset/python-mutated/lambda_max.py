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
import scipy.sparse as sp
from scipy import linalg as LA
from cvxpy.atoms.atom import Atom
from cvxpy.constraints.constraint import Constraint

class lambda_max(Atom):
    """ Maximum eigenvalue; :math:`\\lambda_{\\max}(A)`.
    """

    def __init__(self, A) -> None:
        if False:
            i = 10
            return i + 15
        super(lambda_max, self).__init__(A)

    def numeric(self, values):
        if False:
            i = 10
            return i + 15
        'Returns the largest eigenvalue of A.\n\n        Requires that A be symmetric.\n        '
        lo = hi = self.args[0].shape[0] - 1
        return LA.eigvalsh(values[0], eigvals=(lo, hi))[0]

    def _domain(self) -> List[Constraint]:
        if False:
            for i in range(10):
                print('nop')
        'Returns constraints describing the domain of the node.\n        '
        return [self.args[0].H == self.args[0]]

    def _grad(self, values):
        if False:
            return 10
        'Gives the (sub/super)gradient of the atom w.r.t. each argument.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Args:\n            values: A list of numeric values for the arguments.\n\n        Returns:\n            A list of SciPy CSC sparse matrices or None.\n        '
        (w, v) = LA.eigh(values[0])
        d = np.zeros(w.shape)
        d[-1] = 1
        d = np.diag(d)
        D = v.dot(d).dot(v.T)
        return [sp.csc_matrix(D.ravel(order='F')).T]

    def validate_arguments(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Verify that the argument A is square.\n        '
        if not self.args[0].ndim == 2 or self.args[0].shape[0] != self.args[0].shape[1]:
            raise ValueError("The argument '%s' to lambda_max must resolve to a square matrix." % self.args[0].name())

    def shape_from_args(self) -> Tuple[int, ...]:
        if False:
            return 10
        'Returns the (row, col) shape of the expression.\n        '
        return tuple()

    def sign_from_args(self) -> Tuple[bool, bool]:
        if False:
            i = 10
            return i + 15
        'Returns sign (is positive, is negative) of the expression.\n        '
        return (False, False)

    def is_atom_convex(self) -> bool:
        if False:
            return 10
        'Is the atom convex?\n        '
        return True

    def is_atom_concave(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the atom concave?\n        '
        return False

    def is_incr(self, idx) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the composition non-decreasing in argument idx?\n        '
        return False

    def is_decr(self, idx) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the composition non-increasing in argument idx?\n        '
        return False

    @property
    def value(self):
        if False:
            print('Hello World!')
        if not np.allclose(self.args[0].value, self.args[0].value.T.conj()):
            raise ValueError('Input matrix was not Hermitian/symmetric.')
        if any([p.value is None for p in self.parameters()]):
            return None
        return self._value_impl()