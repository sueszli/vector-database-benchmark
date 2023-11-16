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
from cvxpy.atoms.axis_atom import AxisAtom
from cvxpy.constraints.constraint import Constraint

class norm_inf(AxisAtom):
    _allow_complex = True

    def numeric(self, values):
        if False:
            print('Hello World!')
        'Returns the inf norm of x.\n        '
        if self.axis is None:
            if sp.issparse(values[0]):
                values = values[0].todense().A.flatten()
            else:
                values = np.array(values[0]).flatten()
        else:
            values = np.array(values[0])
        return np.linalg.norm(values, np.inf, axis=self.axis, keepdims=self.keepdims)

    def sign_from_args(self) -> Tuple[bool, bool]:
        if False:
            return 10
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
            return 10
        'Is the atom concave?\n        '
        return False

    def is_atom_log_log_convex(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the atom log-log convex?\n        '
        return True

    def is_atom_log_log_concave(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the atom log-log concave?\n        '
        return False

    def is_incr(self, idx) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the composition non-decreasing in argument idx?\n        '
        return self.args[0].is_nonneg()

    def is_decr(self, idx) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the composition non-increasing in argument idx?\n        '
        return self.args[0].is_nonpos()

    def is_pwl(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the atom piecewise linear?\n        '
        return self.args[0].is_pwl()

    def get_data(self):
        if False:
            return 10
        return [self.axis]

    def name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return '%s(%s)' % (self.__class__.__name__, self.args[0].name())

    def _domain(self) -> List[Constraint]:
        if False:
            i = 10
            return i + 15
        'Returns constraints describing the domain of the node.\n        '
        return []

    def _grad(self, values):
        if False:
            while True:
                i = 10
        'Gives the (sub/super)gradient of the atom w.r.t. each argument.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Args:\n            values: A list of numeric values for the arguments.\n\n        Returns:\n            A list of SciPy CSC sparse matrices or None.\n        '
        return self._axis_grad(values)

    def _column_grad(self, value):
        if False:
            i = 10
            return i + 15
        'Gives the (sub/super)gradient of the atom w.r.t. a column argument.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Args:\n            value: A numeric value for a column.\n\n        Returns:\n            A NumPy ndarray matrix or None.\n        '
        raise NotImplementedError