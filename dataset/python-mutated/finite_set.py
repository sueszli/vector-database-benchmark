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
import numpy as np
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions import cvxtypes
from cvxpy.utilities import scopes

class FiniteSet(Constraint):
    """
    Constrain each entry of an Expression to take a value in a given set of real numbers.

    Parameters
    ----------
    expre : Expression
        The given expression to be constrained. This Expression must be affine.
        If ``expre`` has multiple elements, then the constraint is applied separately to
        each element. I.e., after solving a problem with this constraint, we should have:

        .. code-block:: python

            for e in expre.flatten():
                print(e.value in vec) # => True

    vec : Union[Expression, np.ndarray, set]
        The finite collection of values to which each entry of ``expre``
        is to be constrained.

    ineq_form : bool
        Controls how this constraint is canonicalized into mixed integer linear
        constraints.

        If True, then we use a formulation with ``vec.size - 1`` inequality constraints,
        one equality constraint, and ``vec.size - 1`` binary variables for each element
        of ``expre``.

        If False, then we use a formulation with ``vec.size`` binary variables and two
        equality constraints for each element of ``expre``.

        Defaults to False. The case ``ineq_form=True`` may speed up some mixed-integer
        solvers that use simple branch and bound methods.
    """

    def __init__(self, expre, vec, ineq_form: bool=False, constr_id=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        Expression = cvxtypes.expression()
        if isinstance(vec, set):
            vec = list(vec)
        vec = Expression.cast_to_const(vec).flatten()
        if not expre.is_affine() and (not expre.is_log_log_affine()):
            msg = '\n            Provided Expression must be affine or log-log affine, but had curvature %s.\n            ' % expre.curvature
            raise ValueError(msg)
        self.expre = expre
        self.vec = vec
        self._ineq_form = ineq_form
        super(FiniteSet, self).__init__([expre, vec], constr_id)

    def name(self) -> str:
        if False:
            print('Hello World!')
        return 'FiniteSet(%s, %s)' % (self.args[0], self.args[1])

    def get_data(self):
        if False:
            while True:
                i = 10
        return [self._ineq_form, self.id]

    def is_dcp(self, dpp: bool=False) -> bool:
        if False:
            return 10
        '\n        A FiniteSet constraint is DCP if the constrained expression is affine.\n        '
        if dpp:
            with scopes.dpp_scope():
                return self.args[0].is_affine()
        return self.args[0].is_affine()

    def is_dgp(self, dpp: bool=False) -> bool:
        if False:
            return 10
        if self.vec.parameters():
            return False
        vec_val = self.vec.value
        if dpp:
            with scopes.dpp_scope():
                return self.expre.is_log_log_affine() and np.all(vec_val > 0)
        else:
            return self.expre.is_log_log_affine() and np.all(vec_val > 0)

    def is_dqcp(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.is_dcp()

    @property
    def size(self):
        if False:
            while True:
                i = 10
        return self.expre.size

    @property
    def ineq_form(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Choose between two constraining methodologies, use ``ineq_form=False`` while\n        working with ``Parameter`` types.\n        '
        return self._ineq_form

    @property
    def shape(self):
        if False:
            i = 10
            return i + 15
        return self.expre.shape

    @property
    def residual(self):
        if False:
            print('Hello World!')
        '\n        The residual of the constraint.\n\n        Returns\n        -------\n        float\n        '
        expr_val = np.array(self.expre.value).flatten()
        vec_val = self.vec.value
        resids = [np.min(np.abs(val - vec_val)) for val in expr_val]
        res = max(resids)
        return res