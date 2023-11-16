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
from typing import List, Optional, Tuple
import cvxpy.settings as s
import cvxpy.utilities as u
from cvxpy.atoms import sum, trace
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.problems.problem import Problem

def partial_optimize(prob: Problem, opt_vars: Optional[List[Variable]]=None, dont_opt_vars: Optional[List[Variable]]=None, solver=None, **kwargs) -> 'PartialProblem':
    if False:
        i = 10
        return i + 15
    'Partially optimizes the given problem over the specified variables.\n\n    Either opt_vars or dont_opt_vars must be given.\n    If both are given, they must contain all the variables in the problem.\n\n    Partial optimize is useful for two-stage optimization and graph implementations.\n    For example, we can write\n\n    .. code :: python\n\n        x = Variable(n)\n        t = Variable(n)\n        abs_x = partial_optimize(Problem(Minimize(sum(t)),\n                  [-t <= x, x <= t]), opt_vars=[t])\n\n    to define the entrywise absolute value of x.\n\n    Parameters\n    ----------\n    prob : Problem\n        The problem to partially optimize.\n    opt_vars : list, optional\n        The variables to optimize over.\n    dont_opt_vars : list, optional\n        The variables to not optimize over.\n    solver : str, optional\n        The default solver to use for value and grad.\n    kwargs : keywords, optional\n        Additional solver specific keyword arguments.\n\n    Returns\n    -------\n    Expression\n        An expression representing the partial optimization.\n        Convex for minimization objectives and concave for maximization objectives.\n    '
    if opt_vars is None and dont_opt_vars is None:
        raise ValueError('partial_optimize called with neither opt_vars nor dont_opt_vars.')
    elif opt_vars is None:
        ids = [id(var) for var in dont_opt_vars]
        opt_vars = [var for var in prob.variables() if id(var) not in ids]
    elif dont_opt_vars is None:
        ids = [id(var) for var in opt_vars]
        dont_opt_vars = [var for var in prob.variables() if id(var) not in ids]
    elif opt_vars is not None and dont_opt_vars is not None:
        ids = [id(var) for var in opt_vars + dont_opt_vars]
        for var in prob.variables():
            if id(var) not in ids:
                raise ValueError('If opt_vars and new_opt_vars are both specified, they must contain all variables in the problem.')
    id_to_new_var = {id(var): Variable(var.shape, **var.attributes) for var in opt_vars}
    new_obj = prob.objective.tree_copy(id_to_new_var)
    new_constrs = [con.tree_copy(id_to_new_var) for con in prob.constraints]
    new_var_prob = Problem(new_obj, new_constrs)
    return PartialProblem(new_var_prob, opt_vars, dont_opt_vars, solver, **kwargs)

class PartialProblem(Expression):
    """A partial optimization problem.

    Attributes
    ----------
    opt_vars : list
        The variables to optimize over.
    dont_opt_vars : list
        The variables to not optimize over.
    """

    def __init__(self, prob: Problem, opt_vars: List[Variable], dont_opt_vars: List[Variable], solver, **kwargs) -> None:
        if False:
            return 10
        self.opt_vars = opt_vars
        self.dont_opt_vars = dont_opt_vars
        self.solver = solver
        self.args = [prob]
        self._solve_kwargs = kwargs
        super(PartialProblem, self).__init__()

    def get_data(self):
        if False:
            return 10
        'Returns info needed to reconstruct the expression besides the args.\n        '
        return [self.opt_vars, self.dont_opt_vars, self.solver]

    def is_constant(self) -> bool:
        if False:
            while True:
                i = 10
        return len(self.args[0].variables()) == 0

    def is_convex(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the expression convex?\n        '
        return self.args[0].is_dcp() and type(self.args[0].objective) == Minimize

    def is_concave(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the expression concave?\n        '
        return self.args[0].is_dcp() and type(self.args[0].objective) == Maximize

    def is_dpp(self, context: str='dcp') -> bool:
        if False:
            i = 10
            return i + 15
        'The expression is a disciplined parameterized expression.\n        '
        if context.lower() in ['dcp', 'dgp']:
            return self.args[0].is_dpp(context)
        else:
            raise ValueError('Unsupported context', context)

    def is_log_log_convex(self) -> bool:
        if False:
            return 10
        'Is the expression convex?\n        '
        return self.args[0].is_dgp() and type(self.args[0].objective) == Minimize

    def is_log_log_concave(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the expression convex?\n        '
        return self.args[0].is_dgp() and type(self.args[0].objective) == Maximize

    def is_nonneg(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is the expression nonnegative?\n        '
        return self.args[0].objective.args[0].is_nonneg()

    def is_nonpos(self) -> bool:
        if False:
            print('Hello World!')
        'Is the expression nonpositive?\n        '
        return self.args[0].objective.args[0].is_nonpos()

    def is_imag(self) -> bool:
        if False:
            while True:
                i = 10
        'Is the Leaf imaginary?\n        '
        return False

    def is_complex(self) -> bool:
        if False:
            return 10
        'Is the Leaf complex valued?\n        '
        return False

    @property
    def shape(self) -> Tuple[int, ...]:
        if False:
            i = 10
            return i + 15
        'Returns the (row, col) dimensions of the expression.\n        '
        return tuple()

    def name(self) -> str:
        if False:
            return 10
        'Returns the string representation of the expression.\n        '
        return f'PartialProblem({self.args[0]})'

    def variables(self) -> List[Variable]:
        if False:
            for i in range(10):
                print('nop')
        'Returns the variables in the problem.\n        '
        return self.args[0].variables()

    def parameters(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the parameters in the problem.\n        '
        return self.args[0].parameters()

    def constants(self) -> List[Constant]:
        if False:
            for i in range(10):
                print('nop')
        'Returns the constants in the problem.\n        '
        return self.args[0].constants()

    @property
    def grad(self):
        if False:
            while True:
                i = 10
        'Gives the (sub/super)gradient of the expression w.r.t. each variable.\n\n        Matrix expressions are vectorized, so the gradient is a matrix.\n        None indicates variable values unknown or outside domain.\n\n        Returns:\n            A map of variable to SciPy CSC sparse matrix or None.\n        '
        if self.is_constant():
            return u.grad.constant_grad(self)
        old_vals = {var.id: var.value for var in self.variables()}
        fix_vars = []
        for var in self.dont_opt_vars:
            if var.value is None:
                return u.grad.error_grad(self)
            else:
                fix_vars += [var == var.value]
        prob = Problem(self.args[0].objective, fix_vars + self.args[0].constraints)
        prob.solve(solver=self.solver, **self._solve_kwargs)
        if prob.status in s.SOLUTION_PRESENT:
            sign = self.is_convex() - self.is_concave()
            lagr = self.args[0].objective.args[0]
            for constr in self.args[0].constraints:
                lagr_multiplier = self.cast_to_const(sign * constr.dual_value)
                prod = lagr_multiplier.T @ constr.expr
                if prod.is_scalar():
                    lagr += sum(prod)
                else:
                    lagr += trace(prod)
            grad_map = lagr.grad
            result = {var: grad_map[var] for var in self.dont_opt_vars}
        else:
            result = u.grad.error_grad(self)
        for var in self.variables():
            var.value = old_vals[var.id]
        return result

    @property
    def domain(self):
        if False:
            while True:
                i = 10
        'A list of constraints describing the closure of the region\n           where the expression is finite.\n        '
        obj_expr = self.args[0].objective.args[0]
        return self.args[0].constraints + obj_expr.domain

    @property
    def value(self):
        if False:
            while True:
                i = 10
        'Returns the numeric value of the expression.\n\n        Returns:\n            A numpy matrix or a scalar.\n        '
        old_vals = {var.id: var.value for var in self.variables()}
        fix_vars = []
        for var in self.dont_opt_vars:
            if var.value is None:
                return None
            else:
                fix_vars += [var == var.value]
        prob = Problem(self.args[0].objective, fix_vars + self.args[0].constraints)
        prob.solve(solver=self.solver, **self._solve_kwargs)
        for var in self.variables():
            var.value = old_vals[var.id]
        return prob._solution.opt_val

    def canonicalize(self):
        if False:
            while True:
                i = 10
        'Returns the graph implementation of the object.\n\n        Change the ids of all the opt_vars.\n\n        Returns\n        -------\n            A tuple of (affine expression, [constraints]).\n        '
        (obj, constrs) = self.args[0].objective.args[0].canonical_form
        for cons in self.args[0].constraints:
            constrs += cons.canonical_form[1]
        return (obj, constrs)