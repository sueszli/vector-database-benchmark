"""
Copyright 2022, the CVXPY authors

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
from numpy import linalg as LA
import cvxpy.settings as s
from cvxpy.atoms.atom import Atom
from cvxpy.constraints.constraint import Constraint

class tr_inv(Atom):
    """
    :math:`\\mathrm{tr}\\left(X^{-1} \\right),`
    where :math:`X` is positive definite.
    """

    def __init__(self, X) -> None:
        if False:
            print('Hello World!')
        super(tr_inv, self).__init__(X)

    def numeric(self, values):
        if False:
            return 10
        'Returns the trinv of positive definite matrix X.\n\n        For positive definite matrix X, this is the trace of inverse of X.\n        '
        if LA.norm(values[0] - values[0].T.conj()) >= 1e-08:
            return np.inf
        symm = (values[0] + values[0].T) / 2
        eigVal = LA.eigvalsh(symm)
        if min(eigVal) <= 0:
            return np.inf
        return np.sum(eigVal ** (-1))

    def validate_arguments(self) -> None:
        if False:
            return 10
        X = self.args[0]
        if len(X.shape) == 1 or X.shape[0] != X.shape[1]:
            raise TypeError('The argument to tr_inv must be a square matrix.')

    def shape_from_args(self) -> Tuple[int, ...]:
        if False:
            while True:
                i = 10
        'Returns the (row, col) shape of the expression.\n        '
        return tuple()

    def sign_from_args(self) -> Tuple[bool, bool]:
        if False:
            print('Hello World!')
        'Returns sign (is positive, is negative) of the expression.\n        '
        return (True, False)

    def is_atom_convex(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the atom convex?\n        '
        return True

    def is_atom_concave(self) -> bool:
        if False:
            print('Hello World!')
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
            return 10
        'Is the composition non-increasing in argument idx?\n        '
        return False

    def _grad(self, values):
        if False:
            print('Hello World!')
        'Gives the (sub/super)gradient of the atom w.r.t. each argument.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Args:\n            values: A list of numeric values for the arguments.\n\n        Returns:\n            A list of SciPy CSC sparse matrices or None.\n        '
        X = values[0]
        eigen_val = LA.eigvalsh(X)
        if np.min(eigen_val) > 0:
            D = np.linalg.inv(X).T
            D = -D @ D
            return [sp.csc_matrix(D.ravel(order='F')).T]
        else:
            return [None]

    def _domain(self) -> List[Constraint]:
        if False:
            for i in range(10):
                print('nop')
        'Returns constraints describing the domain of the node.\n        '
        return [self.args[0] >> 0]

    @property
    def value(self) -> float:
        if False:
            i = 10
            return i + 15
        if not np.allclose(self.args[0].value, self.args[0].value.T.conj(), rtol=s.ATOM_EVAL_TOL, atol=s.ATOM_EVAL_TOL):
            raise ValueError('Input matrix was not Hermitian/symmetric.')
        if any([p.value is None for p in self.parameters()]):
            return None
        return self._value_impl()