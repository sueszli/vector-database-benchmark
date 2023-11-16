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
from numpy import linalg as LA
from cvxpy.atoms.atom import Atom

class sigma_max(Atom):
    """ Maximum singular value. """
    _allow_complex = True

    def __init__(self, A) -> None:
        if False:
            i = 10
            return i + 15
        super(sigma_max, self).__init__(A)

    @Atom.numpy_numeric
    def numeric(self, values):
        if False:
            print('Hello World!')
        'Returns the largest singular value of A.\n        '
        return LA.norm(values[0], 2)

    def _grad(self, values):
        if False:
            return 10
        'Gives the (sub/super)gradient of the atom w.r.t. each argument.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Args:\n            values: A list of numeric values for the arguments.\n\n        Returns:\n            A list of SciPy CSC sparse matrices or None.\n        '
        (U, s, V) = LA.svd(values[0])
        ds = np.zeros(len(s))
        ds[0] = 1
        D = U.dot(np.diag(ds)).dot(V)
        return [sp.csc_matrix(D.ravel(order='F')).T]

    def shape_from_args(self) -> Tuple[int, ...]:
        if False:
            i = 10
            return i + 15
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
            i = 10
            return i + 15
        'Is the atom concave?\n        '
        return False

    def is_incr(self, idx) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the composition non-decreasing in argument idx?\n        '
        return False

    def is_decr(self, idx) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the composition non-increasing in argument idx?\n        '
        return False