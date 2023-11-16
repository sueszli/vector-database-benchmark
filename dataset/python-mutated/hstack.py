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

def hstack(arg_list) -> 'Hstack':
    if False:
        i = 10
        return i + 15
    'Horizontal concatenation of an arbitrary number of Expressions.\n\n    Parameters\n    ----------\n    arg_list : list of Expression\n        The Expressions to concatenate.\n    '
    arg_list = [AffAtom.cast_to_const(arg) for arg in arg_list]
    for (idx, arg) in enumerate(arg_list):
        if arg.ndim == 0:
            arg_list[idx] = arg.flatten()
    return Hstack(*arg_list)

class Hstack(AffAtom):
    """ Horizontal concatenation """

    def is_atom_log_log_convex(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def is_atom_log_log_concave(self) -> bool:
        if False:
            print('Hello World!')
        return True

    def numeric(self, values):
        if False:
            while True:
                i = 10
        return np.hstack(values)

    def shape_from_args(self) -> Tuple[int, ...]:
        if False:
            while True:
                i = 10
        if self.args[0].ndim == 1:
            return (sum((arg.size for arg in self.args)),)
        else:
            cols = sum((arg.shape[1] for arg in self.args))
            return (self.args[0].shape[0], cols) + self.args[0].shape[2:]

    def validate_arguments(self) -> None:
        if False:
            i = 10
            return i + 15
        model = self.args[0].shape
        error = ValueError('All the input dimensions except for axis 1 must match exactly.')
        for arg in self.args[1:]:
            if len(arg.shape) != len(model):
                raise error
            elif len(model) > 1:
                for i in range(len(model)):
                    if i != 1 and arg.shape[i] != model[i]:
                        raise error

    def graph_implementation(self, arg_objs, shape: Tuple[int, ...], data=None) -> Tuple[lo.LinOp, List[Constraint]]:
        if False:
            print('Hello World!')
        'Stack the expressions horizontally.\n\n        Parameters\n        ----------\n        arg_objs : list\n            LinExpr for each argument.\n        shape : tuple\n            The shape of the resulting expression.\n        data :\n            Additional data required by the atom.\n\n        Returns\n        -------\n        tuple\n            (LinOp for objective, list of constraints)\n        '
        return (lu.hstack(arg_objs, shape), [])