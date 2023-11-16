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
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint

def vstack(arg_list) -> 'Vstack':
    if False:
        for i in range(10):
            print('nop')
    'Wrapper on vstack to ensure list argument.\n    '
    return Vstack(*arg_list)

class Vstack(AffAtom):
    """ Vertical concatenation """

    def is_atom_log_log_convex(self) -> bool:
        if False:
            print('Hello World!')
        'Is the atom log-log convex?\n        '
        return True

    def is_atom_log_log_concave(self) -> bool:
        if False:
            print('Hello World!')
        'Is the atom log-log concave?\n        '
        return True

    @AffAtom.numpy_numeric
    def numeric(self, values):
        if False:
            for i in range(10):
                print('nop')
        return np.vstack(values)

    def shape_from_args(self) -> Tuple[int, ...]:
        if False:
            i = 10
            return i + 15
        self.args[0].shape
        if self.args[0].ndim == 0:
            return (len(self.args), 1)
        elif self.args[0].ndim == 1:
            return (len(self.args), self.args[0].shape[0])
        else:
            rows = sum((arg.shape[0] for arg in self.args))
            return (rows,) + self.args[0].shape[1:]

    def validate_arguments(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        model = self.args[0].shape
        if model == ():
            model = (1,)
        for arg in self.args[1:]:
            arg_shape = arg.shape
            if arg_shape == ():
                arg_shape = (1,)
            if len(arg_shape) != len(model) or (len(model) > 1 and model[1:] != arg_shape[1:]) or (len(model) <= 1 and model != arg_shape):
                raise ValueError('All the input dimensions except for axis 0 must match exactly.')

    def graph_implementation(self, arg_objs, shape: Tuple[int, ...], data=None) -> Tuple[lo.LinOp, List[Constraint]]:
        if False:
            while True:
                i = 10
        'Stack the expressions vertically.\n\n        Parameters\n        ----------\n        arg_objs : list\n            LinExpr for each argument.\n        shape : tuple\n            The shape of the resulting expression.\n        data :\n            Additional data required by the atom.\n\n        Returns\n        -------\n        tuple\n            (LinOp for objective, list of constraints)\n        '
        return (lu.vstack(arg_objs, shape), [])