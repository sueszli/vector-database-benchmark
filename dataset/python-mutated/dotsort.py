"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from typing import List, Tuple
import numpy as np
import scipy.sparse as sp
import cvxpy.utilities as u
from cvxpy.atoms.atom import Atom
from cvxpy.expressions.constants.parameter import is_param_affine

class dotsort(Atom):
    """ Computes :math:`\\langle sort\\left(vec(X)\\right), sort\\left(vec(W)\\right) \\rangle`,
    where :math:`X` is an expression and :math:`W` is constant.

    | Both arguments are flattened, i.e., we define :math:`x=vec(X)`, :math:`w=vec(W)`.
    | If the length of :math:`w` is less than the length of :math:`x`,
     it is conceptually padded with zeros.
    | When the length of :math:`w` is larger than the length of :math:`x`, an exception is raised.

    `dotsort` is a generalization of `sum_largest` and `sum_smallest`:

    | `sum_largest(x, 3)` is equivalent to `dotsort(x,[1,1,1])`
    | `sum_largest(x, 3.5)` is equivalent to `dotsort(x,[1,1,1,0.5])`
    | `sum_smallest(x,3)` is equivalent to `-dotsort(x, [-1,-1,-1])`

    When the constant argument is not a boolean vector, `dotsort` can be considered as a
    weighted sum of :math:`x`, where the largest weight is
    assigned to the largest entry in :math:`x`, etc..
    """

    def __init__(self, X, W) -> None:
        if False:
            return 10
        super(dotsort, self).__init__(X, W)

    def validate_arguments(self) -> None:
        if False:
            return 10
        if not self.args[1].is_constant():
            raise ValueError('The W argument must be constant.')
        if self.args[0].size < self.args[1].size:
            raise ValueError('The size of of W must be less or equal to the size of X.')
        super(dotsort, self).validate_arguments()

    def numeric(self, values):
        if False:
            print('Hello World!')
        '\n        Returns the inner product of the sorted values of vec(X) and the sorted\n        (and potentially padded) values of vec(W).\n        '
        (x, w_padded) = self._get_args_from_values(values)
        return np.sort(x) @ np.sort(w_padded)

    def _grad(self, values):
        if False:
            while True:
                i = 10
        'Gives the (sub/super)gradient of the atom w.r.t. each argument.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Args:\n            values: A list of numeric values for the arguments.\n\n        Returns:\n            A list of SciPy CSC sparse matrices or None.\n        '
        (x, w_padded) = self._get_args_from_values(values)
        indices = np.argsort(x)
        n = len(x)
        sorted_w = np.sort(w_padded)
        return [sp.csc_matrix((sorted_w, (indices, np.zeros(n))), shape=(n, 1))]

    def shape_from_args(self) -> Tuple[int, ...]:
        if False:
            while True:
                i = 10
        'Returns the (row, col) shape of the expression.\n        '
        return tuple()

    def sign_from_args(self) -> Tuple[bool, bool]:
        if False:
            return 10
        'Returns sign (is positive, is negative) of the expression.\n        '
        x_pos = self.args[0].is_nonneg()
        x_neg = self.args[0].is_nonpos()
        w_pos = self.args[1].is_nonneg()
        w_neg = self.args[1].is_nonpos()
        is_positive = x_pos and w_pos or (x_neg and w_neg)
        is_negative = x_neg and w_pos or (x_pos and w_neg)
        return (is_positive, is_negative)

    def is_atom_convex(self) -> bool:
        if False:
            print('Hello World!')
        'Is the atom convex?\n        '
        if u.scopes.dpp_scope_active():
            X = self.args[0]
            W = self.args[1]
            return X.is_constant() or is_param_affine(W)
        else:
            return True

    def is_atom_concave(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the atom concave?\n        '
        return False

    def is_incr(self, idx) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the composition non-decreasing in argument idx?\n        '
        return self.args[1].is_nonneg()

    def is_decr(self, idx) -> bool:
        if False:
            print('Hello World!')
        'Is the composition non-increasing in argument idx?\n        '
        return self.args[1].is_nonpos()

    def get_data(self):
        if False:
            i = 10
            return i + 15
        'Returns None, W is stored as an argument.\n        '
        return None

    @staticmethod
    def _get_args_from_values(values: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            return 10
        x = values[0].flatten()
        w = values[1].flatten()
        w_padded = np.zeros_like(x)
        w_padded[:len(w)] = w
        return (x, w_padded)