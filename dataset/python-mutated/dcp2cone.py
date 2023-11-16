"""
Copyright 2013 Steven Diamond, 2017 Akshay Agrawal, 2017 Robin Verschueren

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
from cvxpy import problems
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.expression import Expression
from cvxpy.problems.objective import Minimize
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.dcp2cone.canonicalizers import CANON_METHODS as cone_canon_methods
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.qp2quad_form.canonicalizers import QUAD_CANON_METHODS as quad_canon_methods

class Dcp2Cone(Canonicalization):
    """Reduce DCP problems to a conic form.

    This reduction takes as input (minimization) DCP problems and converts
    them into problems with affine or quadratic objectives and conic
    constraints whose arguments are affine.
    """

    def __init__(self, problem=None, quad_obj: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(Canonicalization, self).__init__(problem=problem)
        self.cone_canon_methods = cone_canon_methods
        self.quad_canon_methods = quad_canon_methods
        self.quad_obj = quad_obj

    def accepts(self, problem):
        if False:
            i = 10
            return i + 15
        'A problem is accepted if it is a minimization and is DCP.\n        '
        return type(problem.objective) == Minimize and problem.is_dcp()

    def apply(self, problem):
        if False:
            for i in range(10):
                print('nop')
        'Converts a DCP problem to a conic form.\n        '
        if not self.accepts(problem):
            raise ValueError('Cannot reduce problem to cone program')
        inverse_data = InverseData(problem)
        (canon_objective, canon_constraints) = self.canonicalize_tree(problem.objective, True)
        for constraint in problem.constraints:
            (canon_constr, aux_constr) = self.canonicalize_tree(constraint, False)
            canon_constraints += aux_constr + [canon_constr]
            inverse_data.cons_id_map.update({constraint.id: canon_constr.id})
        new_problem = problems.problem.Problem(canon_objective, canon_constraints)
        return (new_problem, inverse_data)

    def canonicalize_tree(self, expr, affine_above: bool) -> Tuple[Expression, list]:
        if False:
            for i in range(10):
                print('nop')
        'Recursively canonicalize an Expression.\n\n        Parameters\n        ----------\n        expr : The expression tree to canonicalize.\n        affine_above : The path up to the root node is all affine atoms.\n\n        Returns\n        -------\n        A tuple of the canonicalized expression and generated constraints.\n        '
        if type(expr) == cvxtypes.partial_problem():
            (canon_expr, constrs) = self.canonicalize_tree(expr.args[0].objective.expr, False)
            for constr in expr.args[0].constraints:
                (canon_constr, aux_constr) = self.canonicalize_tree(constr, False)
                constrs += [canon_constr] + aux_constr
        else:
            affine_atom = type(expr) not in self.cone_canon_methods
            canon_args = []
            constrs = []
            for arg in expr.args:
                (canon_arg, c) = self.canonicalize_tree(arg, affine_atom and affine_above)
                canon_args += [canon_arg]
                constrs += c
            (canon_expr, c) = self.canonicalize_expr(expr, canon_args, affine_above)
            constrs += c
        return (canon_expr, constrs)

    def canonicalize_expr(self, expr, args, affine_above: bool) -> Tuple[Expression, list]:
        if False:
            for i in range(10):
                print('nop')
        'Canonicalize an expression, w.r.t. canonicalized arguments.\n\n        Parameters\n        ----------\n        expr : The expression tree to canonicalize.\n        args : The canonicalized arguments of expr.\n        affine_above : The path up to the root node is all affine atoms.\n\n        Returns\n        -------\n        A tuple of the canonicalized expression and generated constraints.\n        '
        if isinstance(expr, Expression) and (expr.is_constant() and (not expr.parameters())):
            return (expr, [])
        if self.quad_obj and affine_above and (type(expr) in self.quad_canon_methods):
            if type(expr) == cvxtypes.power() and (not expr._quadratic_power()):
                return self.cone_canon_methods[type(expr)](expr, args)
            else:
                return self.quad_canon_methods[type(expr)](expr, args)
        if type(expr) in self.cone_canon_methods:
            return self.cone_canon_methods[type(expr)](expr, args)
        return (expr.copy(args), [])