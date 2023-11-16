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
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
from cvxpy.error import DCPError
from cvxpy.expressions.expression import Expression
from cvxpy.interface.matrix_utilities import scalar_value
from cvxpy.utilities import scopes

class Objective(u.Canonical):
    """An optimization objective.

    Parameters
    ----------
    expr : Expression
        The expression to act upon. Must be a scalar.

    Raises
    ------
    ValueError
        If expr is not a scalar.
    """
    NAME = 'objective'

    def __init__(self, expr) -> None:
        if False:
            i = 10
            return i + 15
        self.args = [Expression.cast_to_const(expr)]
        if not self.args[0].is_scalar():
            raise ValueError("The '%s' objective must resolve to a scalar." % self.NAME)
        if not self.args[0].is_real():
            raise ValueError("The '%s' objective must be real valued." % self.NAME)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return '%s(%s)' % (self.__class__.__name__, repr(self.args[0]))

    def __str__(self) -> str:
        if False:
            return 10
        return ' '.join([self.NAME, self.args[0].name()])

    def __radd__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if other == 0:
            return self
        else:
            raise NotImplementedError()

    def __sub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, (Minimize, Maximize)):
            raise NotImplementedError()
        return self + -other

    def __rsub__(self, other):
        if False:
            print('Hello World!')
        if other == 0:
            return -self
        else:
            raise NotImplementedError()

    def __mul__(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, (int, float)):
            raise NotImplementedError()
        if (type(self) == Maximize) == (other < 0.0):
            return Minimize(self.args[0] * other)
        else:
            return Maximize(self.args[0] * other)
    __rmul__ = __mul__

    def __div__(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, (int, float)):
            raise NotImplementedError()
        return self * (1.0 / other)
    __truediv__ = __div__

    @property
    def value(self):
        if False:
            return 10
        'The value of the objective expression.\n        '
        v = self.args[0].value
        if v is None:
            return None
        else:
            return scalar_value(v)

    def is_quadratic(self) -> bool:
        if False:
            return 10
        'Returns if the objective is a quadratic function.\n        '
        return self.args[0].is_quadratic()

    def is_qpwa(self) -> bool:
        if False:
            return 10
        'Returns if the objective is a quadratic of piecewise affine.\n        '
        return self.args[0].is_qpwa()

class Minimize(Objective):
    """An optimization objective for minimization.

    Parameters
    ----------
    expr : Expression
        The expression to minimize. Must be a scalar.

    Raises
    ------
    ValueError
        If expr is not a scalar.
    """
    NAME = 'minimize'

    def __neg__(self) -> 'Maximize':
        if False:
            for i in range(10):
                print('nop')
        return Maximize(-self.args[0])

    def __add__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, (Minimize, Maximize)):
            raise NotImplementedError()
        if type(other) is Minimize:
            return Minimize(self.args[0] + other.args[0])
        else:
            raise DCPError('Problem does not follow DCP rules.')

    def canonicalize(self):
        if False:
            i = 10
            return i + 15
        "Pass on the target expression's objective and constraints.\n        "
        return self.args[0].canonical_form

    def is_dcp(self, dpp: bool=False) -> bool:
        if False:
            while True:
                i = 10
        'The objective must be convex.\n        '
        if dpp:
            with scopes.dpp_scope():
                return self.args[0].is_convex()
        return self.args[0].is_convex()

    def is_dgp(self, dpp: bool=False) -> bool:
        if False:
            i = 10
            return i + 15
        'The objective must be log-log convex.\n        '
        if dpp:
            with scopes.dpp_scope():
                return self.args[0].is_log_log_convex()
        return self.args[0].is_log_log_convex()

    def is_dpp(self, context='dcp') -> bool:
        if False:
            i = 10
            return i + 15
        with scopes.dpp_scope():
            if context.lower() == 'dcp':
                return self.is_dcp(dpp=True)
            elif context.lower() == 'dgp':
                return self.is_dgp(dpp=True)
            else:
                raise ValueError('Unsupported context ', context)

    def is_dqcp(self) -> bool:
        if False:
            while True:
                i = 10
        'The objective must be quasiconvex.\n        '
        return self.args[0].is_quasiconvex()

    @staticmethod
    def primal_to_result(result):
        if False:
            for i in range(10):
                print('nop')
        'The value of the objective given the solver primal value.\n        '
        return result

class Maximize(Objective):
    """An optimization objective for maximization.

    Parameters
    ----------
    expr : Expression
        The expression to maximize. Must be a scalar.

    Raises
    ------
    ValueError
        If expr is not a scalar.
    """
    NAME = 'maximize'

    def __neg__(self) -> Minimize:
        if False:
            while True:
                i = 10
        return Minimize(-self.args[0])

    def __add__(self, other):
        if False:
            print('Hello World!')
        if not isinstance(other, (Minimize, Maximize)):
            raise NotImplementedError()
        if type(other) is Maximize:
            return Maximize(self.args[0] + other.args[0])
        else:
            raise Exception('Problem does not follow DCP rules.')

    def canonicalize(self):
        if False:
            print('Hello World!')
        "Negates the target expression's objective.\n        "
        (obj, constraints) = self.args[0].canonical_form
        return (lu.neg_expr(obj), constraints)

    def is_dcp(self, dpp: bool=False) -> bool:
        if False:
            print('Hello World!')
        'The objective must be concave.\n        '
        if dpp:
            with scopes.dpp_scope():
                return self.args[0].is_concave()
        return self.args[0].is_concave()

    def is_dgp(self, dpp: bool=False) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'The objective must be log-log concave.\n        '
        if dpp:
            with scopes.dpp_scope():
                return self.args[0].is_log_log_concave()
        return self.args[0].is_log_log_concave()

    def is_dpp(self, context='dcp') -> bool:
        if False:
            for i in range(10):
                print('nop')
        with scopes.dpp_scope():
            if context.lower() == 'dcp':
                return self.is_dcp(dpp=True)
            elif context.lower() == 'dgp':
                return self.is_dgp(dpp=True)
            else:
                raise ValueError('Unsupported context ', context)

    def is_dqcp(self) -> bool:
        if False:
            while True:
                i = 10
        'The objective must be quasiconcave.\n        '
        return self.args[0].is_quasiconcave()

    @staticmethod
    def primal_to_result(result):
        if False:
            while True:
                i = 10
        'The value of the objective given the solver primal value.\n        '
        return -result