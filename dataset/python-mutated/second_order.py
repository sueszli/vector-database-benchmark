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
from typing import List, Optional
import numpy as np
from cvxpy.constraints.cones import Cone
from cvxpy.expressions import cvxtypes
from cvxpy.utilities import scopes

class SOC(Cone):
    """A second-order cone constraint for each row/column.

    Assumes ``t`` is a vector the same length as ``X``'s columns (rows) for
    ``axis == 0`` (``1``).

    Attributes:
        t: The scalar part of the second-order constraint.
        X: A matrix whose rows/columns are each a cone.
        axis: Slice by column 0 or row 1.
    """

    def __init__(self, t, X, axis: int=0, constr_id=None) -> None:
        if False:
            return 10
        t = cvxtypes.expression().cast_to_const(t)
        if len(t.shape) >= 2 or not t.is_real():
            raise ValueError('Invalid first argument.')
        if len(X.shape) <= 1 and t.size > 1 or (len(X.shape) == 2 and t.size != X.shape[1 - axis]) or (len(X.shape) == 1 and axis == 1):
            raise ValueError('Argument dimensions %s and %s, with axis=%i, are incompatible.' % (t.shape, X.shape, axis))
        self.axis = axis
        if len(t.shape) == 0:
            t = t.flatten()
        super(SOC, self).__init__([t, X], constr_id)

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        return 'SOC(%s, %s)' % (self.args[0], self.args[1])

    @property
    def residual(self) -> Optional[np.ndarray]:
        if False:
            while True:
                i = 10
        '\n        For each cone, returns:\n\n        ||(t,X) - proj(t,X)||\n        with\n        proj(t,X) = (t,X)                       if t >= ||x||\n                    0.5*(t/||x|| + 1)(||x||,x)  if -||x|| < t < ||x||\n                    0                           if t <= -||x||\n\n        References:\n             https://docs.mosek.com/modeling-cookbook/practical.html#distance-to-a-cone\n             https://math.stackexchange.com/questions/2509986/projection-onto-the-second-order-cone\n        '
        t = self.args[0].value
        X = self.args[1].value
        if t is None or X is None:
            return None
        if self.axis == 0:
            X = X.T
        promoted = X.ndim == 1
        X = np.atleast_2d(X)
        t_proj = np.zeros(t.shape)
        X_proj = np.zeros(X.shape)
        norms = np.linalg.norm(X, ord=2, axis=1)
        t_geq_x_norm = t >= norms
        t_proj[t_geq_x_norm] = t[t_geq_x_norm]
        X_proj[t_geq_x_norm] = X[t_geq_x_norm]
        abs_t_less_x_norm = np.abs(t) < norms
        avg_coeff = 0.5 * (1 + t / norms)
        X_proj[abs_t_less_x_norm] = avg_coeff[abs_t_less_x_norm, None] * X[abs_t_less_x_norm]
        t_proj[abs_t_less_x_norm] = avg_coeff[abs_t_less_x_norm] * norms[abs_t_less_x_norm]
        Xt = np.concatenate([X, t[:, None]], axis=1)
        Xt_proj = np.concatenate([X_proj, t_proj[:, None]], axis=1)
        resid = np.linalg.norm(Xt - Xt_proj, ord=2, axis=1)
        if promoted:
            return resid[0]
        else:
            return resid

    def get_data(self):
        if False:
            return 10
        'Returns info needed to reconstruct the object besides the args.\n\n        Returns\n        -------\n        list\n        '
        return [self.axis, self.id]

    def num_cones(self):
        if False:
            i = 10
            return i + 15
        'The number of elementwise cones.\n        '
        return self.args[0].size

    @property
    def size(self) -> int:
        if False:
            print('Hello World!')
        'The number of entries in the combined cones.\n        '
        cone_size = 1 + self.args[1].shape[self.axis]
        return cone_size * self.num_cones()

    def cone_sizes(self) -> List[int]:
        if False:
            print('Hello World!')
        'The dimensions of the second-order cones.\n\n        Returns\n        -------\n        list\n            A list of the sizes of the elementwise cones.\n        '
        cone_size = 1 + self.args[1].shape[self.axis]
        return [cone_size] * self.num_cones()

    def is_dcp(self, dpp: bool=False) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'An SOC constraint is DCP if each of its arguments is affine.\n        '
        if dpp:
            with scopes.dpp_scope():
                return all((arg.is_affine() for arg in self.args))
        return all((arg.is_affine() for arg in self.args))

    def is_dgp(self, dpp: bool=False) -> bool:
        if False:
            while True:
                i = 10
        return False

    def is_dqcp(self) -> bool:
        if False:
            return 10
        return self.is_dcp()

    def save_dual_value(self, value) -> None:
        if False:
            print('Hello World!')
        cone_size = 1 + self.args[1].shape[self.axis]
        value = np.reshape(value, newshape=(-1, cone_size))
        t = value[:, 0]
        X = value[:, 1:]
        if self.axis == 0:
            X = X.T
        self.dual_variables[0].save_value(t)
        self.dual_variables[1].save_value(X)

    def _dual_cone(self, *args):
        if False:
            while True:
                i = 10
        'Implements the dual cone of the second-order cone\n        See Pg 85 of the MOSEK modelling cookbook for more information'
        if args is None:
            return SOC(self.dual_variables[0], self.dual_variables[1], self.axis)
        else:
            args_shapes = [arg.shape for arg in args]
            instance_args_shapes = [arg.shape for arg in self.args]
            assert len(args) == len(self.args)
            assert args_shapes == instance_args_shapes
            return SOC(args[0], args[1], self.axis)