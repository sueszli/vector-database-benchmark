"""
Copyright 2021 The CVXPY Developers

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
from __future__ import division
from typing import List, Optional, Tuple
import numpy as np
from scipy.sparse import csc_matrix
from scipy.special import rel_entr as rel_entr_scipy
from cvxpy.atoms.elementwise.elementwise import Elementwise

class rel_entr(Elementwise):
    """:math:`x\\log(x/y)`

    For disambiguation between rel_entr and kl_div, see https://github.com/cvxpy/cvxpy/issues/733
    """

    def __init__(self, x, y) -> None:
        if False:
            while True:
                i = 10
        super(rel_entr, self).__init__(x, y)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        if False:
            print('Hello World!')
        x = values[0]
        y = values[1]
        return rel_entr_scipy(x, y)

    def sign_from_args(self) -> Tuple[bool, bool]:
        if False:
            return 10
        'Returns sign (is positive, is negative) of the expression.\n        '
        return (False, False)

    def is_atom_convex(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
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
            i = 10
            return i + 15
        'Is the composition non-decreasing in argument idx?\n        '
        return False

    def is_decr(self, idx) -> bool:
        if False:
            while True:
                i = 10
        'Is the composition non-increasing in argument idx?\n        '
        if idx == 0:
            return False
        else:
            return True

    def _grad(self, values) -> List[Optional[csc_matrix]]:
        if False:
            i = 10
            return i + 15
        'Gives the (sub/super)gradient of the atom w.r.t. each argument.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n\n        Args:\n            values: A list of numeric values for the arguments.\n\n        Returns:\n            A list of SciPy CSC sparse matrices or None.\n        '
        if np.min(values[0]) <= 0 or np.min(values[1]) <= 0:
            return [None, None]
        else:
            div = values[0] / values[1]
            grad_vals = [np.log(div) + 1, -div]
            grad_list = []
            for idx in range(len(values)):
                rows = self.args[idx].size
                cols = self.size
                grad_list += [rel_entr.elemwise_grad_to_diag(grad_vals[idx], rows, cols)]
            return grad_list

    def _domain(self):
        if False:
            while True:
                i = 10
        'Returns constraints describing the domain of the node.\n        '
        return [self.args[0] >= 0, self.args[1] >= 0]