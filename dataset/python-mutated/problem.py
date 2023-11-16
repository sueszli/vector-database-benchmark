"""
Copyright 2013 Steven Diamond, 2017 Akshay Agrawal

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
from __future__ import annotations
import time
import warnings
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
import cvxpy.utilities as u
import cvxpy.utilities.performance_utils as perf
from cvxpy import Constant, error
from cvxpy import settings as s
from cvxpy.atoms.atom import Atom
from cvxpy.constraints import Equality, Inequality, NonNeg, NonPos, Zero
from cvxpy.constraints.constraint import Constraint
from cvxpy.error import DPPError
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.variable import Variable
from cvxpy.interface.matrix_utilities import scalar_value
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.reductions import InverseData
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.dgp2dcp.dgp2dcp import Dgp2Dcp
from cvxpy.reductions.dqcp2dcp import dqcp2dcp
from cvxpy.reductions.eval_params import EvalParams
from cvxpy.reductions.flip_objective import FlipObjective
from cvxpy.reductions.solution import INF_OR_UNB_MESSAGE
from cvxpy.reductions.solvers import bisection
from cvxpy.reductions.solvers import defines as slv_def
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.defines import SOLVER_MAP_CONIC, SOLVER_MAP_QP
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.reductions.solvers.solving_chain import SolvingChain, construct_solving_chain
from cvxpy.settings import SOLVERS
from cvxpy.utilities import debug_tools
from cvxpy.utilities.deterministic import unique_list
SolveResult = namedtuple('SolveResult', ['opt_value', 'status', 'primal_values', 'dual_values'])
_COL_WIDTH = 79
_HEADER = '=' * _COL_WIDTH + '\n' + 'CVXPY'.center(_COL_WIDTH) + '\n' + ('v' + cvxtypes.version()).center(_COL_WIDTH) + '\n' + '=' * _COL_WIDTH
_COMPILATION_STR = '-' * _COL_WIDTH + '\n' + 'Compilation'.center(_COL_WIDTH) + '\n' + '-' * _COL_WIDTH
_NUM_SOLVER_STR = '-' * _COL_WIDTH + '\n' + 'Numerical solver'.center(_COL_WIDTH) + '\n' + '-' * _COL_WIDTH
_FOOTER = '-' * _COL_WIDTH + '\n' + 'Summary'.center(_COL_WIDTH) + '\n' + '-' * _COL_WIDTH

class Cache:

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.key = None
        self.solving_chain: Optional[SolvingChain] = None
        self.param_prog = None
        self.inverse_data: Optional[InverseData] = None

    def invalidate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.key = None
        self.solving_chain = None
        self.param_prog = None
        self.inverse_data = None

    def make_key(self, solver, gp, ignore_dpp, use_quad_obj):
        if False:
            for i in range(10):
                print('nop')
        return (solver, gp, ignore_dpp, use_quad_obj)

    def gp(self):
        if False:
            for i in range(10):
                print('nop')
        return self.key is not None and self.key[1]

def _validate_constraint(constraint):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(constraint, Constraint):
        return constraint
    elif isinstance(constraint, bool):
        return Constant(0) <= Constant(1) if constraint else Constant(1) <= Constant(0)
    else:
        raise ValueError('Problem has an invalid constraint of type %s' % type(constraint))

class Problem(u.Canonical):
    """A convex optimization problem.

    Problems are immutable, save for modification through the specification
    of :class:`~cvxpy.expressions.constants.parameters.Parameter`

    Arguments
    ---------
    objective : Minimize or Maximize
        The problem's objective.
    constraints : list
        The constraints on the problem variables.
    """
    REGISTERED_SOLVE_METHODS = {}

    def __init__(self, objective: Union[Minimize, Maximize], constraints: Optional[List[Constraint]]=None) -> None:
        if False:
            i = 10
            return i + 15
        if constraints is None:
            constraints = []
        if not isinstance(objective, (Minimize, Maximize)):
            raise error.DCPError('Problem objective must be Minimize or Maximize.')
        self._objective = objective
        if debug_tools.node_count(self._objective) >= debug_tools.MAX_NODES:
            warnings.warn('Objective contains too many subexpressions. Consider vectorizing your CVXPY code to speed up compilation.')
        self._constraints = [_validate_constraint(c) for c in constraints]
        for (i, constraint) in enumerate(self._constraints):
            if debug_tools.node_count(constraint) >= debug_tools.MAX_NODES:
                warnings.warn(f'Constraint #{i} contains too many subexpressions. Consider vectorizing your CVXPY code to speed up compilation.')
        self._value = None
        self._status: Optional[str] = None
        self._solution = None
        self._cache = Cache()
        self._solver_cache = {}
        self._size_metrics: Optional['SizeMetrics'] = None
        self._solver_stats: Optional['SolverStats'] = None
        self._compilation_time: Optional[float] = None
        self._solve_time: Optional[float] = None
        self.args = [self._objective, self._constraints]

    @property
    def value(self):
        if False:
            i = 10
            return i + 15
        'float : The value from the last time the problem was solved\n                   (or None if not solved).\n        '
        if self._value is None:
            return None
        else:
            return scalar_value(self._value)

    @property
    def status(self) -> str:
        if False:
            print('Hello World!')
        'str : The status from the last time the problem was solved; one\n                 of optimal, infeasible, or unbounded (with or without\n                 suffix inaccurate).\n        '
        return self._status

    @property
    def solution(self):
        if False:
            while True:
                i = 10
        'Solution : The solution from the last time the problem was solved.\n        '
        return self._solution

    @property
    def objective(self) -> Union[Minimize, Maximize]:
        if False:
            while True:
                i = 10
        "Minimize or Maximize : The problem's objective.\n\n        Note that the objective cannot be reassigned after creation,\n        and modifying the objective after creation will result in\n        undefined behavior.\n        "
        return self._objective

    @property
    def constraints(self) -> List[Constraint]:
        if False:
            while True:
                i = 10
        "A shallow copy of the problem's constraints.\n\n        Note that constraints cannot be reassigned, appended to, or otherwise\n        modified after creation, except through parameters.\n        "
        return self._constraints[:]

    @property
    def param_dict(self):
        if False:
            return 10
        '\n        Expose all parameters as a dictionary\n        '
        return {parameters.name(): parameters for parameters in self.parameters()}

    @property
    def var_dict(self) -> Dict[str, Variable]:
        if False:
            i = 10
            return i + 15
        '\n        Expose all variables as a dictionary\n        '
        return {variable.name(): variable for variable in self.variables()}

    @perf.compute_once
    def is_dcp(self, dpp: bool=False) -> bool:
        if False:
            return 10
        'Does the problem satisfy DCP rules?\n\n        Arguments\n        ---------\n        dpp : bool, optional\n            If True, enforce the disciplined parametrized programming (DPP)\n            ruleset; only relevant when the problem involves Parameters.\n            DPP is a mild restriction of DCP. When a problem involving\n            Parameters is DPP, subsequent solves can be much faster than\n            the first one. For more information, consult the documentation at\n\n            https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming\n\n        Returns\n        -------\n        bool\n            True if the Expression is DCP, False otherwise.\n        '
        return all((expr.is_dcp(dpp) for expr in self.constraints + [self.objective]))

    @perf.compute_once
    def is_dgp(self, dpp: bool=False) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Does the problem satisfy DGP rules?\n\n        Arguments\n        ---------\n        dpp : bool, optional\n            If True, enforce the disciplined parametrized programming (DPP)\n            ruleset; only relevant when the problem involves Parameters.\n            DPP is a mild restriction of DGP. When a problem involving\n            Parameters is DPP, subsequent solves can be much faster than\n            the first one. For more information, consult the documentation at\n\n            https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming\n\n        Returns\n        -------\n        bool\n            True if the Expression is DGP, False otherwise.\n        '
        return all((expr.is_dgp(dpp) for expr in self.constraints + [self.objective]))

    @perf.compute_once
    def is_dqcp(self) -> bool:
        if False:
            return 10
        'Does the problem satisfy the DQCP rules?\n        '
        return all((expr.is_dqcp() for expr in self.constraints + [self.objective]))

    @perf.compute_once
    def is_dpp(self, context: str='dcp') -> bool:
        if False:
            i = 10
            return i + 15
        "Does the problem satisfy DPP rules?\n\n        DPP is a mild restriction of DGP. When a problem involving\n        Parameters is DPP, subsequent solves can be much faster than\n        the first one. For more information, consult the documentation at\n\n        https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming\n\n        Arguments\n        ---------\n        context : str\n            Whether to check DPP-compliance for DCP or DGP; ``context`` should\n            be either ``'dcp'`` or ``'dgp'``. Calling ``problem.is_dpp('dcp')``\n            is equivalent to ``problem.is_dcp(dpp=True)``, and\n            `problem.is_dpp('dgp')`` is equivalent to\n            `problem.is_dgp(dpp=True)`.\n\n        Returns\n        -------\n        bool\n            Whether the problem satisfies the DPP rules.\n        "
        if context.lower() == 'dcp':
            return self.is_dcp(dpp=True)
        elif context.lower() == 'dgp':
            return self.is_dgp(dpp=True)
        else:
            raise ValueError('Unsupported context ', context)

    @perf.compute_once
    def is_qp(self) -> bool:
        if False:
            print('Hello World!')
        'Is problem a quadratic program?\n        '
        for c in self.constraints:
            if not (isinstance(c, (Equality, Zero)) or c.args[0].is_pwl()):
                return False
        for var in self.variables():
            if var.is_psd() or var.is_nsd():
                return False
        return self.is_dcp() and self.objective.args[0].is_qpwa()

    @perf.compute_once
    def is_mixed_integer(self) -> bool:
        if False:
            while True:
                i = 10
        return any((v.attributes['boolean'] or v.attributes['integer'] for v in self.variables()))

    @perf.compute_once
    def variables(self) -> List[Variable]:
        if False:
            i = 10
            return i + 15
        'Accessor method for variables.\n\n        Returns\n        -------\n        list of :class:`~cvxpy.expressions.variable.Variable`\n            A list of the variables in the problem.\n        '
        vars_ = self.objective.variables()
        for constr in self.constraints:
            vars_ += constr.variables()
        return unique_list(vars_)

    @perf.compute_once
    def parameters(self):
        if False:
            while True:
                i = 10
        'Accessor method for parameters.\n\n        Returns\n        -------\n        list of :class:`~cvxpy.expressions.constants.parameter.Parameter`\n            A list of the parameters in the problem.\n        '
        params = self.objective.parameters()
        for constr in self.constraints:
            params += constr.parameters()
        return unique_list(params)

    @perf.compute_once
    def constants(self) -> List[Constant]:
        if False:
            return 10
        'Accessor method for constants.\n\n        Returns\n        -------\n        list of :class:`~cvxpy.expressions.constants.constant.Constant`\n            A list of the constants in the problem.\n        '
        const_dict = {}
        constants_ = self.objective.constants()
        for constr in self.constraints:
            constants_ += constr.constants()
        const_dict = {id(constant): constant for constant in constants_}
        return list(const_dict.values())

    def atoms(self) -> List[Atom]:
        if False:
            while True:
                i = 10
        'Accessor method for atoms.\n\n        Returns\n        -------\n        list of :class:`~cvxpy.atoms.Atom`\n            A list of the atom types in the problem; note that this list\n            contains classes, not instances.\n        '
        atoms = self.objective.atoms()
        for constr in self.constraints:
            atoms += constr.atoms()
        return unique_list(atoms)

    @property
    def size_metrics(self) -> 'SizeMetrics':
        if False:
            while True:
                i = 10
        ":class:`~cvxpy.problems.problem.SizeMetrics` : Information about the problem's size.\n        "
        if self._size_metrics is None:
            self._size_metrics = SizeMetrics(self)
        return self._size_metrics

    @property
    def solver_stats(self) -> 'SolverStats':
        if False:
            i = 10
            return i + 15
        ':class:`~cvxpy.problems.problem.SolverStats` : Information returned by the solver.\n        '
        return self._solver_stats

    @property
    def compilation_time(self) -> float | None:
        if False:
            print('Hello World!')
        'float : The number of seconds it took to compile the problem the\n                   last time it was compiled.\n        '
        return self._compilation_time

    def solve(self, *args, **kwargs):
        if False:
            print('Hello World!')
        'Compiles and solves the problem using the specified method.\n\n        Populates the :code:`status` and :code:`value` attributes on the\n        problem object as a side-effect.\n\n        Arguments\n        ---------\n        solver : str, optional\n            The solver to use. For example, \'ECOS\', \'SCS\', or \'OSQP\'.\n        verbose : bool, optional\n            Overrides the default of hiding solver output, and prints\n            logging information describing CVXPY\'s compilation process.\n        gp : bool, optional\n            If True, parses the problem as a disciplined geometric program\n            instead of a disciplined convex program.\n        qcp : bool, optional\n            If True, parses the problem as a disciplined quasiconvex program\n            instead of a disciplined convex program.\n        requires_grad : bool, optional\n            Makes it possible to compute gradients of a solution with respect to\n            Parameters by calling ``problem.backward()`` after solving, or to\n            compute perturbations to the variables given perturbations to Parameters by\n            calling ``problem.derivative()``.\n\n            Gradients are only supported for DCP and DGP problems, not\n            quasiconvex problems. When computing gradients (i.e., when\n            this argument is True), the problem must satisfy the DPP rules.\n        enforce_dpp : bool, optional\n            When True, a DPPError will be thrown when trying to solve a non-DPP\n            problem (instead of just a warning). Only relevant for problems\n            involving Parameters. Defaults to False.\n        ignore_dpp : bool, optional\n            When True, DPP problems will be treated as non-DPP,\n            which may speed up compilation. Defaults to False.\n        method : function, optional\n            A custom solve method to use.\n        kwargs : keywords, optional\n            Additional solver specific arguments. See Notes below.\n\n        Notes\n        ------\n        CVXPY interfaces with a wide range of solvers; the algorithms used by these solvers\n        have arguments relating to stopping criteria, and strategies to improve solution quality.\n\n        There is no one choice of arguments which is perfect for every problem. If you are not\n        getting satisfactory results from a solver, you can try changing its arguments. The\n        exact way this is done depends on the specific solver. Here are some examples:\n\n        ::\n\n            prob.solve(solver=\'ECOS\', abstol=1e-6)\n            prob.solve(solver=\'OSQP\', max_iter=10000).\n            mydict = {"MSK_DPAR_INTPNT_CO_TOL_NEAR_REL":  10}\n            prob.solve(solver=\'MOSEK\', mosek_params=mydict).\n\n        You should refer to CVXPY\'s web documentation for details on how to pass solver\n        solver arguments, available at\n\n        https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options\n\n        Returns\n        -------\n        float\n            The optimal value for the problem, or a string indicating\n            why the problem could not be solved.\n\n        Raises\n        ------\n        cvxpy.error.DCPError\n            Raised if the problem is not DCP and `gp` is False.\n        cvxpy.error.DGPError\n            Raised if the problem is not DGP and `gp` is True.\n        cvxpy.error.DPPError\n            Raised if DPP settings are invalid.\n        cvxpy.error.SolverError\n            Raised if no suitable solver exists among the installed solvers,\n            or if an unanticipated error is encountered.\n        '
        func_name = kwargs.pop('method', None)
        if func_name is not None:
            solve_func = Problem.REGISTERED_SOLVE_METHODS[func_name]
        else:
            solve_func = Problem._solve
        return solve_func(self, *args, **kwargs)

    @classmethod
    def register_solve(cls, name: str, func) -> None:
        if False:
            return 10
        'Adds a solve method to the Problem class.\n\n        Arguments\n        ---------\n        name : str\n            The keyword for the method.\n        func : function\n            The function that executes the solve method. This function must\n            take as its first argument the problem instance to solve.\n        '
        cls.REGISTERED_SOLVE_METHODS[name] = func

    def get_problem_data(self, solver, gp: bool=False, enforce_dpp: bool=False, ignore_dpp: bool=False, verbose: bool=False, canon_backend: str | None=None, solver_opts: Optional[dict]=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns the problem data used in the call to the solver.\n\n        When a problem is solved, CVXPY creates a chain of reductions enclosed\n        in a :class:`~cvxpy.reductions.solvers.solving_chain.SolvingChain`,\n        and compiles it to some low-level representation that is\n        compatible with the targeted solver. This method returns that low-level\n        representation.\n\n        For some solving chains, this low-level representation is a dictionary\n        that contains exactly those arguments that were supplied to the solver;\n        however, for other solving chains, the data is an intermediate\n        representation that is compiled even further by the solver interfaces.\n\n        A solution to the equivalent low-level problem can be obtained via the\n        data by invoking the `solve_via_data` method of the returned solving\n        chain, a thin wrapper around the code external to CVXPY that further\n        processes and solves the problem. Invoke the unpack_results method\n        to recover a solution to the original problem.\n\n        For example:\n\n        ::\n\n            objective = ...\n            constraints = ...\n            problem = cp.Problem(objective, constraints)\n            data, chain, inverse_data = problem.get_problem_data(cp.SCS)\n            # calls SCS using `data`\n            soln = chain.solve_via_data(problem, data)\n            # unpacks the solution returned by SCS into `problem`\n            problem.unpack_results(soln, chain, inverse_data)\n\n        Alternatively, the `data` dictionary returned by this method\n        contains enough information to bypass CVXPY and call the solver\n        directly.\n\n        For example:\n\n        ::\n\n            problem = cp.Problem(objective, constraints)\n            data, _, _ = problem.get_problem_data(cp.SCS)\n\n            import scs\n            probdata = {\n              \'A\': data[\'A\'],\n              \'b\': data[\'b\'],\n              \'c\': data[\'c\'],\n            }\n            cone_dims = data[\'dims\']\n            cones = {\n                "f": cone_dims.zero,\n                "l": cone_dims.nonneg,\n                "q": cone_dims.soc,\n                "ep": cone_dims.exp,\n                "s": cone_dims.psd,\n            }\n            soln = scs.solve(data, cones)\n\n        The structure of the data dict that CVXPY returns depends on the\n        solver. For details, consult the solver interfaces in\n        `cvxpy/reductions/solvers`.\n\n        Arguments\n        ---------\n        solver : str\n            The solver the problem data is for.\n        gp : bool, optional\n            If True, then parses the problem as a disciplined geometric program\n            instead of a disciplined convex program.\n        enforce_dpp : bool, optional\n            When True, a DPPError will be thrown when trying to parse a non-DPP\n            problem (instead of just a warning). Defaults to False.\n        ignore_dpp : bool, optional\n            When True, DPP problems will be treated as non-DPP,\n            which may speed up compilation. Defaults to False.\n        canon_backend : str, optional\n            \'CPP\' (default) | \'SCIPY\'\n            Specifies which backend to use for canonicalization, which can affect\n            compilation time. Defaults to None, i.e., selecting the default\n            backend.\n        verbose : bool, optional\n            If True, print verbose output related to problem compilation.\n        solver_opts : dict, optional\n            A dict of options that will be passed to the specific solver.\n            In general, these options will override any default settings\n            imposed by cvxpy.\n\n        Returns\n        -------\n        dict or object\n            lowest level representation of problem\n        SolvingChain\n            The solving chain that created the data.\n        list\n            The inverse data generated by the chain.\n\n        Raises\n        ------\n        cvxpy.error.DPPError\n            Raised if DPP settings are invalid.\n        '
        if enforce_dpp and ignore_dpp:
            raise DPPError('Cannot set enforce_dpp = True and ignore_dpp = True.')
        start = time.time()
        if solver_opts is None:
            use_quad_obj = None
        else:
            use_quad_obj = solver_opts.get('use_quad_obj', None)
        key = self._cache.make_key(solver, gp, ignore_dpp, use_quad_obj)
        if key != self._cache.key:
            self._cache.invalidate()
            solving_chain = self._construct_chain(solver=solver, gp=gp, enforce_dpp=enforce_dpp, ignore_dpp=ignore_dpp, canon_backend=canon_backend, solver_opts=solver_opts)
            self._cache.key = key
            self._cache.solving_chain = solving_chain
            self._solver_cache = {}
        else:
            solving_chain = self._cache.solving_chain
        if verbose:
            print(_COMPILATION_STR)
        if self._cache.param_prog is not None:
            if verbose:
                s.LOGGER.info('Using cached ASA map, for faster compilation (bypassing reduction chain).')
            if gp:
                dgp2dcp = self._cache.solving_chain.get(Dgp2Dcp)
                old_params_to_new_params = dgp2dcp.canon_methods._parameters
                for param in self.parameters():
                    if param in old_params_to_new_params:
                        old_params_to_new_params[param].value = np.log(param.value)
            (data, solver_inverse_data) = solving_chain.solver.apply(self._cache.param_prog)
            inverse_data = self._cache.inverse_data + [solver_inverse_data]
            self._compilation_time = time.time() - start
            if verbose:
                s.LOGGER.info('Finished problem compilation (took %.3e seconds).', self._compilation_time)
        else:
            if verbose:
                solver_name = solving_chain.reductions[-1].name()
                reduction_chain_str = ' -> '.join((type(r).__name__ for r in solving_chain.reductions))
                s.LOGGER.info('Compiling problem (target solver=%s).', solver_name)
                s.LOGGER.info('Reduction chain: %s', reduction_chain_str)
            (data, inverse_data) = solving_chain.apply(self, verbose)
            safe_to_cache = isinstance(data, dict) and s.PARAM_PROB in data and (not any((isinstance(reduction, EvalParams) for reduction in solving_chain.reductions)))
            self._compilation_time = time.time() - start
            if verbose:
                s.LOGGER.info('Finished problem compilation (took %.3e seconds).', self._compilation_time)
            if safe_to_cache:
                if verbose and self.parameters():
                    s.LOGGER.info('(Subsequent compilations of this problem, using the same arguments, should take less time.)')
                self._cache.param_prog = data[s.PARAM_PROB]
                self._cache.inverse_data = inverse_data[:-1]
        return (data, solving_chain, inverse_data)

    def _find_candidate_solvers(self, solver=None, gp: bool=False):
        if False:
            print('Hello World!')
        '\n        Find candidate solvers for the current problem. If solver\n        is not None, it checks if the specified solver is compatible\n        with the problem passed.\n\n        Arguments\n        ---------\n        solver : Union[string, Solver, None]\n            The name of the solver with which to solve the problem or an\n            instance of a custom solver. If no solver is supplied\n            (i.e., if solver is None), then the targeted solver may be any\n            of those that are installed. If the problem is variable-free,\n            then this parameter is ignored.\n        gp : bool\n            If True, the problem is parsed as a Disciplined Geometric Program\n            instead of as a Disciplined Convex Program.\n\n        Returns\n        -------\n        dict\n            A dictionary of compatible solvers divided in `qp_solvers`\n            and `conic_solvers`.\n\n        Raises\n        ------\n        cvxpy.error.SolverError\n            Raised if the problem is not DCP and `gp` is False.\n        cvxpy.error.DGPError\n            Raised if the problem is not DGP and `gp` is True.\n        '
        candidates = {'qp_solvers': [], 'conic_solvers': []}
        if isinstance(solver, Solver):
            return self._add_custom_solver_candidates(solver)
        if solver is not None:
            if solver not in slv_def.INSTALLED_SOLVERS:
                raise error.SolverError('The solver %s is not installed.' % solver)
            if solver in slv_def.CONIC_SOLVERS:
                candidates['conic_solvers'] += [solver]
            if solver in slv_def.QP_SOLVERS:
                candidates['qp_solvers'] += [solver]
        else:
            candidates['qp_solvers'] = [s for s in slv_def.INSTALLED_SOLVERS if s in slv_def.QP_SOLVERS]
            candidates['conic_solvers'] = []
            for slv in slv_def.INSTALLED_SOLVERS:
                if slv in slv_def.CONIC_SOLVERS and slv != s.ECOS_BB:
                    candidates['conic_solvers'].append(slv)
        if gp:
            if solver is not None and solver not in slv_def.CONIC_SOLVERS:
                raise error.SolverError("When `gp=True`, `solver` must be a conic solver (received '%s'); try calling " % solver + ' `solve()` with `solver=cvxpy.ECOS`.')
            elif solver is None:
                candidates['qp_solvers'] = []
        if self.is_mixed_integer():
            if slv_def.INSTALLED_MI_SOLVERS == [s.ECOS_BB] and solver != s.ECOS_BB:
                msg = "\n\n                    You need a mixed-integer solver for this model. Refer to the documentation\n                        https://www.cvxpy.org/tutorial/advanced/index.html#mixed-integer-programs\n                    for discussion on this topic.\n\n                    Quick fix 1: if you install the python package CVXOPT (pip install cvxopt),\n                    then CVXPY can use the open-source mixed-integer linear programming\n                    solver `GLPK`. If your problem is nonlinear then you can install SCIP\n                    (pip install pyscipopt).\n\n                    Quick fix 2: you can explicitly specify solver='ECOS_BB'. This may result\n                    in incorrect solutions and is not recommended.\n                "
                raise error.SolverError(msg)
            candidates['qp_solvers'] = [s for s in candidates['qp_solvers'] if slv_def.SOLVER_MAP_QP[s].MIP_CAPABLE]
            candidates['conic_solvers'] = [s for s in candidates['conic_solvers'] if slv_def.SOLVER_MAP_CONIC[s].MIP_CAPABLE]
            if not candidates['conic_solvers'] and (not candidates['qp_solvers']):
                raise error.SolverError('Problem is mixed-integer, but candidate QP/Conic solvers (%s) are not MIP-capable.' % (candidates['qp_solvers'] + candidates['conic_solvers']))
        return candidates

    def _add_custom_solver_candidates(self, custom_solver: Solver):
        if False:
            return 10
        '\n        Returns a list of candidate solvers where custom_solver is the only potential option.\n\n        Arguments\n        ---------\n        custom_solver : Solver\n\n        Returns\n        -------\n        dict\n            A dictionary of compatible solvers divided in `qp_solvers`\n            and `conic_solvers`.\n\n        Raises\n        ------\n        cvxpy.error.SolverError\n            Raised if the name of the custom solver conflicts with the name of some officially\n            supported solver\n        '
        if custom_solver.name() in SOLVERS:
            message = 'Custom solvers must have a different name than the officially supported ones'
            raise error.SolverError(message)
        candidates = {'qp_solvers': [], 'conic_solvers': []}
        if not self.is_mixed_integer() or custom_solver.MIP_CAPABLE:
            if isinstance(custom_solver, QpSolver):
                SOLVER_MAP_QP[custom_solver.name()] = custom_solver
                candidates['qp_solvers'] = [custom_solver.name()]
            elif isinstance(custom_solver, ConicSolver):
                SOLVER_MAP_CONIC[custom_solver.name()] = custom_solver
                candidates['conic_solvers'] = [custom_solver.name()]
        return candidates

    def _construct_chain(self, solver: Optional[str]=None, gp: bool=False, enforce_dpp: bool=False, ignore_dpp: bool=False, canon_backend: str | None=None, solver_opts: Optional[dict]=None) -> SolvingChain:
        if False:
            while True:
                i = 10
        "\n        Construct the chains required to reformulate and solve the problem.\n\n        In particular, this function\n\n        # finds the candidate solvers\n        # constructs the solving chain that performs the\n           numeric reductions and solves the problem.\n\n        Arguments\n        ---------\n        solver : str, optional\n            The solver to use. Defaults to ECOS.\n        gp : bool, optional\n            If True, the problem is parsed as a Disciplined Geometric Program\n            instead of as a Disciplined Convex Program.\n        enforce_dpp : bool, optional\n            Whether to error on DPP violations.\n        ignore_dpp : bool, optional\n            When True, DPP problems will be treated as non-DPP,\n            which may speed up compilation. Defaults to False.\n        canon_backend : str, optional\n            'CPP' (default) | 'SCIPY'\n            Specifies which backend to use for canonicalization, which can affect\n            compilation time. Defaults to None, i.e., selecting the default\n            backend.\n        solver_opts: dict, optional\n            Additional arguments to pass to the solver.\n\n        Returns\n        -------\n        A solving chain\n        "
        candidate_solvers = self._find_candidate_solvers(solver=solver, gp=gp)
        self._sort_candidate_solvers(candidate_solvers)
        return construct_solving_chain(self, candidate_solvers, gp=gp, enforce_dpp=enforce_dpp, ignore_dpp=ignore_dpp, canon_backend=canon_backend, solver_opts=solver_opts, specified_solver=solver)

    @staticmethod
    def _sort_candidate_solvers(solvers) -> None:
        if False:
            while True:
                i = 10
        'Sorts candidate solvers lists according to slv_def.CONIC_SOLVERS/QP_SOLVERS\n\n        Arguments\n        ---------\n        candidates : dict\n            Dictionary of candidate solvers divided in qp_solvers\n            and conic_solvers\n        Returns\n        -------\n        None\n        '
        if len(solvers['conic_solvers']) > 1:
            solvers['conic_solvers'] = sorted(solvers['conic_solvers'], key=lambda s: slv_def.CONIC_SOLVERS.index(s))
        if len(solvers['qp_solvers']) > 1:
            solvers['qp_solvers'] = sorted(solvers['qp_solvers'], key=lambda s: slv_def.QP_SOLVERS.index(s))

    def _invalidate_cache(self) -> None:
        if False:
            i = 10
            return i + 15
        self._cache_key = None
        self._solving_chain = None
        self._param_prog = None
        self._inverse_data = None

    def _solve(self, solver: str=None, warm_start: bool=True, verbose: bool=False, gp: bool=False, qcp: bool=False, requires_grad: bool=False, enforce_dpp: bool=False, ignore_dpp: bool=False, canon_backend: str | None=None, **kwargs):
        if False:
            return 10
        "Solves a DCP compliant optimization problem.\n\n        Saves the values of primal and dual variables in the variable\n        and constraint objects, respectively.\n\n        Arguments\n        ---------\n        solver : str, optional\n            The solver to use. Defaults to ECOS.\n        warm_start : bool, optional\n            Should the previous solver result be used to warm start?\n        verbose : bool, optional\n            Overrides the default of hiding solver output.\n        gp : bool, optional\n            If True, parses the problem as a disciplined geometric program.\n        qcp : bool, optional\n            If True, parses the problem as a disciplined quasiconvex program.\n        requires_grad : bool, optional\n            Makes it possible to compute gradients with respect to\n            parameters by calling `backward()` after solving, or to compute\n            perturbations to the variables by calling `derivative()`. When\n            True, the solver must be SCS, and dqcp must be False.\n            A DPPError is thrown when problem is not DPP.\n        enforce_dpp : bool, optional\n            When True, a DPPError will be thrown when trying to solve a non-DPP\n            problem (instead of just a warning). Defaults to False.\n        ignore_dpp : bool, optional\n            When True, DPP problems will be treated as non-DPP,\n            which may speed up compilation. Defaults to False.\n        canon_backend : str, optional\n            'CPP' (default) | 'SCIPY'\n            Specifies which backend to use for canonicalization, which can affect\n            compilation time. Defaults to None, i.e., selecting the default\n            backend.\n        kwargs : dict, optional\n            A dict of options that will be passed to the specific solver.\n            In general, these options will override any default settings\n            imposed by cvxpy.\n\n        Returns\n        -------\n        float\n            The optimal value for the problem, or a string indicating\n            why the problem could not be solved.\n        "
        if verbose:
            print(_HEADER)
        for parameter in self.parameters():
            if parameter.value is None:
                raise error.ParameterError("A Parameter (whose name is '%s') does not have a value associated with it; all Parameter objects must have values before solving a problem." % parameter.name())
        if verbose:
            n_variables = sum((np.prod(v.shape) for v in self.variables()))
            n_parameters = sum((np.prod(p.shape) for p in self.parameters()))
            s.LOGGER.info('Your problem has %d variables, %d constraints, and %d parameters.', n_variables, len(self.constraints), n_parameters)
            curvatures = []
            if self.is_dcp():
                curvatures.append('DCP')
            if self.is_dgp():
                curvatures.append('DGP')
            if self.is_dqcp():
                curvatures.append('DQCP')
            s.LOGGER.info('It is compliant with the following grammars: %s', ', '.join(curvatures))
            if n_parameters == 0:
                s.LOGGER.info('(If you need to solve this problem multiple times, but with different data, consider using parameters.)')
            s.LOGGER.info('CVXPY will first compile your problem; then, it will invoke a numerical solver to obtain a solution.')
            s.LOGGER.info('Your problem is compiled with the %s canonicalization backend.', s.DEFAULT_CANON_BACKEND if canon_backend is None else canon_backend)
        if requires_grad:
            dpp_context = 'dgp' if gp else 'dcp'
            if qcp:
                raise ValueError('Cannot compute gradients of DQCP problems.')
            elif not self.is_dpp(dpp_context):
                raise error.DPPError('Problem is not DPP (when requires_grad is True, problem must be DPP).')
            elif solver is not None and solver not in [s.SCS, s.DIFFCP]:
                raise ValueError('When requires_grad is True, the only supported solver is SCS (received %s).' % solver)
            elif s.DIFFCP not in slv_def.INSTALLED_SOLVERS:
                raise ModuleNotFoundError('The Python package diffcp must be installed to differentiate through problems. Please follow the installation instructions at https://github.com/cvxgrp/diffcp')
            else:
                solver = s.DIFFCP
        else:
            if gp and qcp:
                raise ValueError('At most one of `gp` and `qcp` can be True.')
            if qcp and (not self.is_dcp()):
                if not self.is_dqcp():
                    raise error.DQCPError('The problem is not DQCP.')
                if verbose:
                    s.LOGGER.info('Reducing DQCP problem to a one-parameter family of DCP problems, for bisection.')
                reductions = [dqcp2dcp.Dqcp2Dcp()]
                start = time.time()
                if type(self.objective) == Maximize:
                    reductions = [FlipObjective()] + reductions
                    (low, high) = (kwargs.get('low'), kwargs.get('high'))
                    if high is not None:
                        kwargs['low'] = high * -1
                    if low is not None:
                        kwargs['high'] = low * -1
                chain = Chain(problem=self, reductions=reductions)
                soln = bisection.bisect(chain.reduce(), solver=solver, verbose=verbose, **kwargs)
                self.unpack(chain.retrieve(soln))
                return self.value
        (data, solving_chain, inverse_data) = self.get_problem_data(solver, gp, enforce_dpp, ignore_dpp, verbose, canon_backend, kwargs)
        if verbose:
            print(_NUM_SOLVER_STR)
            s.LOGGER.info('Invoking solver %s  to obtain a solution.', solving_chain.reductions[-1].name())
        start = time.time()
        solution = solving_chain.solve_via_data(self, data, warm_start, verbose, kwargs)
        end = time.time()
        self._solve_time = end - start
        self.unpack_results(solution, solving_chain, inverse_data)
        if verbose:
            print(_FOOTER)
            s.LOGGER.info('Problem status: %s', self.status)
            val = self.value if self.value is not None else np.NaN
            s.LOGGER.info('Optimal value: %.3e', val)
            s.LOGGER.info('Compilation took %.3e seconds', self._compilation_time)
            s.LOGGER.info('Solver (including time spent in interface) took %.3e seconds', self._solve_time)
        return self.value

    def backward(self) -> None:
        if False:
            while True:
                i = 10
        'Compute the gradient of a solution with respect to Parameters.\n\n        This method differentiates through the solution map of the problem,\n        obtaining the gradient of a solution with respect to the Parameters.\n        In other words, it calculates the sensitivities of the Parameters\n        with respect to perturbations in the optimal Variable values. This\n        can be useful for integrating CVXPY into automatic differentiation\n        toolkits.\n\n        ``backward()`` populates the ``gradient`` attribute of each Parameter\n        in the problem as a side-effect. It can only be called after calling\n        ``solve()`` with ``requires_grad=True``.\n\n        Below is a simple example:\n\n        ::\n\n            import cvxpy as cp\n            import numpy as np\n\n            p = cp.Parameter()\n            x = cp.Variable()\n            quadratic = cp.square(x - 2 * p)\n            problem = cp.Problem(cp.Minimize(quadratic), [x >= 0])\n            p.value = 3.0\n            problem.solve(requires_grad=True, eps=1e-10)\n            # backward() populates the gradient attribute of the parameters\n            problem.backward()\n            # Because x* = 2 * p, dx*/dp = 2\n            np.testing.assert_allclose(p.gradient, 2.0)\n\n        In the above example, the gradient could easily be computed by hand.\n        The ``backward()`` is useful because for almost all problems, the\n        gradient cannot be computed analytically.\n\n        This method can be used to differentiate through any DCP or DGP\n        problem, as long as the problem is DPP compliant (i.e.,\n        ``problem.is_dcp(dpp=True)`` or ``problem.is_dgp(dpp=True)`` evaluates to\n        ``True``).\n\n        This method uses the chain rule to evaluate the gradients of a\n        scalar-valued function of the Variables with respect to the Parameters.\n        For example, let x be a variable and p a Parameter; x and p might be\n        scalars, vectors, or matrices. Let f be a scalar-valued function, with\n        z = f(x). Then this method computes dz/dp = (dz/dx) (dx/p). dz/dx\n        is chosen as the all-ones vector by default, corresponding to\n        choosing f to be the sum function. You can specify a custom value for\n        dz/dx by setting the ``gradient`` attribute on your variables. For example,\n\n        ::\n\n            import cvxpy as cp\n            import numpy as np\n\n\n            b = cp.Parameter()\n            x = cp.Variable()\n            quadratic = cp.square(x - 2 * b)\n            problem = cp.Problem(cp.Minimize(quadratic), [x >= 0])\n            b.value = 3.\n            problem.solve(requires_grad=True, eps=1e-10)\n            x.gradient = 4.\n            problem.backward()\n            # dz/dp = dz/dx dx/dp = 4. * 2. == 8.\n            np.testing.assert_allclose(b.gradient, 8.)\n\n        The ``gradient`` attribute on a variable can also be interpreted as a\n        perturbation to its optimal value.\n\n        Raises\n        ------\n            ValueError\n                if solve was not called with ``requires_grad=True``\n            SolverError\n                if the problem is infeasible or unbounded\n        '
        if s.DIFFCP not in self._solver_cache:
            raise ValueError('backward can only be called after calling solve with `requires_grad=True`')
        elif self.status not in s.SOLUTION_PRESENT:
            raise error.SolverError('Backpropagating through infeasible/unbounded problems is not yet supported. Please file an issue on Github if you need this feature.')
        backward_cache = self._solver_cache[s.DIFFCP]
        DT = backward_cache['DT']
        zeros = np.zeros(backward_cache['s'].shape)
        del_vars = {}
        gp = self._cache.gp()
        for variable in self.variables():
            if variable.gradient is None:
                del_vars[variable.id] = np.ones(variable.shape)
            else:
                del_vars[variable.id] = np.asarray(variable.gradient, dtype=np.float64)
            if gp:
                del_vars[variable.id] *= variable.value
        dx = self._cache.param_prog.split_adjoint(del_vars)
        start = time.time()
        (dA, db, dc) = DT(dx, zeros, zeros)
        end = time.time()
        backward_cache['DT_TIME'] = end - start
        dparams = self._cache.param_prog.apply_param_jac(dc, -dA, db)
        if not gp:
            for param in self.parameters():
                param.gradient = dparams[param.id]
        else:
            dgp2dcp = self._cache.solving_chain.get(Dgp2Dcp)
            old_params_to_new_params = dgp2dcp.canon_methods._parameters
            for param in self.parameters():
                grad = 0.0 if param.id not in dparams else dparams[param.id]
                if param in old_params_to_new_params:
                    new_param = old_params_to_new_params[param]
                    grad += 1.0 / param.value * dparams[new_param.id]
                param.gradient = grad

    def derivative(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Apply the derivative of the solution map to perturbations in the Parameters\n\n        This method applies the derivative of the solution map to perturbations\n        in the Parameters to obtain perturbations in the optimal values of the\n        Variables. In other words, it tells you how the optimal values of the\n        Variables would be changed by small changes to the Parameters.\n\n        You can specify perturbations in a Parameter by setting its ``delta``\n        attribute (if unspecified, the perturbation defaults to 0).\n\n        This method populates the ``delta`` attribute of the Variables as a\n        side-effect.\n\n        This method can only be called after calling ``solve()`` with\n        ``requires_grad=True``. It is compatible with both DCP and DGP\n        problems (that are also DPP-compliant).\n\n        Below is a simple example:\n\n        ::\n\n            import cvxpy as cp\n            import numpy as np\n\n            p = cp.Parameter()\n            x = cp.Variable()\n            quadratic = cp.square(x - 2 * p)\n            problem = cp.Problem(cp.Minimize(quadratic), [x >= 0])\n            p.value = 3.0\n            problem.solve(requires_grad=True, eps=1e-10)\n            # derivative() populates the delta attribute of the variables\n            p.delta = 1e-3\n            problem.derivative()\n            # Because x* = 2 * p, dx*/dp = 2, so (dx*/dp)(p.delta) == 2e-3\n            np.testing.assert_allclose(x.delta, 2e-3)\n\n        Raises\n        ------\n            ValueError\n                if solve was not called with ``requires_grad=True``\n            SolverError\n                if the problem is infeasible or unbounded\n        '
        if s.DIFFCP not in self._solver_cache:
            raise ValueError('derivative can only be called after calling solve with `requires_grad=True`')
        elif self.status not in s.SOLUTION_PRESENT:
            raise ValueError('Differentiating through infeasible/unbounded problems is not yet supported. Please file an issue on Github if you need this feature.')
        backward_cache = self._solver_cache[s.DIFFCP]
        param_prog = self._cache.param_prog
        D = backward_cache['D']
        param_deltas = {}
        gp = self._cache.gp()
        if gp:
            dgp2dcp = self._cache.solving_chain.get(Dgp2Dcp)
        if not self.parameters():
            for variable in self.variables():
                variable.delta = np.zeros(variable.shape)
            return
        for param in self.parameters():
            delta = param.delta if param.delta is not None else np.zeros(param.shape)
            if gp:
                if param in dgp2dcp.canon_methods._parameters:
                    new_param_id = dgp2dcp.canon_methods._parameters[param].id
                else:
                    new_param_id = param.id
                param_deltas[new_param_id] = 1.0 / param.value * np.asarray(delta, dtype=np.float64)
                if param.id in param_prog.param_id_to_col:
                    param_deltas[param.id] = np.asarray(delta, dtype=np.float64)
            else:
                param_deltas[param.id] = np.asarray(delta, dtype=np.float64)
        (dc, _, dA, db) = param_prog.apply_parameters(param_deltas, zero_offset=True)
        start = time.time()
        (dx, _, _) = D(-dA, db, dc)
        end = time.time()
        backward_cache['D_TIME'] = end - start
        dvars = param_prog.split_solution(dx, [v.id for v in self.variables()])
        for variable in self.variables():
            variable.delta = dvars[variable.id]
            if gp:
                variable.delta *= variable.value

    def _clear_solution(self) -> None:
        if False:
            return 10
        for v in self.variables():
            v.save_value(None)
        for c in self.constraints:
            for dv in c.dual_variables:
                dv.save_value(None)
        self._value = None
        self._status = None
        self._solution = None

    def unpack(self, solution) -> None:
        if False:
            i = 10
            return i + 15
        'Updates the problem state given a Solution.\n\n        Updates problem.status, problem.value and value of primal and dual\n        variables. If solution.status is in cvxpy.settins.ERROR, this method\n        is a no-op.\n\n        Arguments\n        _________\n        solution : cvxpy.Solution\n            A Solution object.\n\n        Raises\n        ------\n        ValueError\n            If the solution object has an invalid status\n        '
        if solution.status in s.SOLUTION_PRESENT:
            for v in self.variables():
                v.save_value(solution.primal_vars[v.id])
            for c in self.constraints:
                if c.id in solution.dual_vars:
                    c.save_dual_value(solution.dual_vars[c.id])
            self._value = self.objective.value
        elif solution.status in s.INF_OR_UNB:
            for v in self.variables():
                v.save_value(None)
            for constr in self.constraints:
                for dv in constr.dual_variables:
                    dv.save_value(None)
            self._value = solution.opt_val
        else:
            raise ValueError('Cannot unpack invalid solution: %s' % solution)
        self._status = solution.status
        self._solution = solution

    def unpack_results(self, solution, chain: SolvingChain, inverse_data) -> None:
        if False:
            return 10
        'Updates the problem state given the solver results.\n\n        Updates problem.status, problem.value and value of\n        primal and dual variables.\n\n        Arguments\n        _________\n        solution : object\n            The solution returned by applying the chain to the problem\n            and invoking the solver on the resulting data.\n        chain : SolvingChain\n            A solving chain that was used to solve the problem.\n        inverse_data : list\n            The inverse data returned by applying the chain to the problem.\n\n        Raises\n        ------\n        cvxpy.error.SolverError\n            If the solver failed\n        '
        solution = chain.invert(solution, inverse_data)
        if solution.status in s.INACCURATE:
            warnings.warn('Solution may be inaccurate. Try another solver, adjusting the solver settings, or solve with verbose=True for more information.')
        if solution.status == s.INFEASIBLE_OR_UNBOUNDED:
            warnings.warn(INF_OR_UNB_MESSAGE)
        if solution.status in s.ERROR:
            raise error.SolverError("Solver '%s' failed. " % chain.solver.name() + 'Try another solver, or solve with verbose=True for more information.')
        self.unpack(solution)
        self._solver_stats = SolverStats.from_dict(self._solution.attr, chain.solver.name())

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        if len(self.constraints) == 0:
            return str(self.objective)
        else:
            subject_to = 'subject to '
            lines = [str(self.objective), subject_to + str(self.constraints[0])]
            for constr in self.constraints[1:]:
                lines += [len(subject_to) * ' ' + str(constr)]
            return '\n'.join(lines)

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return 'Problem(%s, %s)' % (repr(self.objective), repr(self.constraints))

    def __neg__(self) -> 'Problem':
        if False:
            i = 10
            return i + 15
        return Problem(-self.objective, self.constraints)

    def __add__(self, other) -> 'Problem':
        if False:
            return 10
        if other == 0:
            return self
        elif not isinstance(other, Problem):
            raise NotImplementedError()
        return Problem(self.objective + other.objective, unique_list(self.constraints + other.constraints))

    def __radd__(self, other) -> 'Problem':
        if False:
            i = 10
            return i + 15
        if other == 0:
            return self
        else:
            raise NotImplementedError()

    def __sub__(self, other) -> 'Problem':
        if False:
            while True:
                i = 10
        if not isinstance(other, Problem):
            raise NotImplementedError()
        return Problem(self.objective - other.objective, unique_list(self.constraints + other.constraints))

    def __rsub__(self, other) -> 'Problem':
        if False:
            for i in range(10):
                print('nop')
        if other == 0:
            return -self
        else:
            raise NotImplementedError()

    def __mul__(self, other) -> 'Problem':
        if False:
            i = 10
            return i + 15
        if not isinstance(other, (int, float)):
            raise NotImplementedError()
        return Problem(self.objective * other, self.constraints)
    __rmul__ = __mul__

    def __div__(self, other) -> 'Problem':
        if False:
            while True:
                i = 10
        if not isinstance(other, (int, float)):
            raise NotImplementedError()
        return Problem(self.objective * (1.0 / other), self.constraints)

    def is_constant(self) -> bool:
        if False:
            print('Hello World!')
        return False
    __truediv__ = __div__

@dataclass
class SolverStats:
    """Reports some of the miscellaneous information that is returned
    by the solver after solving but that is not captured directly by
    the Problem instance.

    Attributes
    ----------
    solver_name : str
        The name of the solver.
    solve_time : double
        The time (in seconds) it took for the solver to solve the problem.
    setup_time : double
        The time (in seconds) it took for the solver to setup the problem.
    num_iters : int
        The number of iterations the solver had to go through to find a solution.
    extra_stats : object
        Extra statistics specific to the solver; these statistics are typically
        returned directly from the solver, without modification by CVXPY.
        This object may be a dict, or a custom Python object.
    """
    solver_name: str
    solve_time: Optional[float] = None
    setup_time: Optional[float] = None
    num_iters: Optional[int] = None
    extra_stats: Optional[dict] = None

    @classmethod
    def from_dict(cls, attr: dict, solver_name: str) -> 'SolverStats':
        if False:
            while True:
                i = 10
        'Construct a SolverStats object from a dictionary of attributes.\n\n        Parameters\n        ----------\n        attr : dict\n            A dictionary of attributes returned by the solver.\n        solver_name : str\n            The name of the solver.\n\n        Returns\n        -------\n        SolverStats\n            A SolverStats object.\n        '
        return cls(solver_name, solve_time=attr.get(s.SOLVE_TIME), setup_time=attr.get(s.SETUP_TIME), num_iters=attr.get(s.NUM_ITERS), extra_stats=attr.get(s.EXTRA_STATS))

class SizeMetrics:
    """Reports various metrics regarding the problem.

    Attributes
    ----------

    num_scalar_variables : integer
        The number of scalar variables in the problem.
    num_scalar_data : integer
        The number of scalar constants and parameters in the problem. The number of
        constants used across all matrices, vectors, in the problem.
        Some constants are not apparent when the problem is constructed: for example,
        The sum_squares expression is a wrapper for a quad_over_lin expression with a
        constant 1 in the denominator.
    num_scalar_eq_constr : integer
        The number of scalar equality constraints in the problem.
    num_scalar_leq_constr : integer
        The number of scalar inequality constraints in the problem.

    max_data_dimension : integer
        The longest dimension of any data block constraint or parameter.
    max_big_small_squared : integer
        The maximum value of (big)(small)^2 over all data blocks of the problem, where
        (big) is the larger dimension and (small) is the smaller dimension
        for each data block.
    """

    def __init__(self, problem: Problem) -> None:
        if False:
            while True:
                i = 10
        self.num_scalar_variables = 0
        for var in problem.variables():
            self.num_scalar_variables += var.size
        self.max_data_dimension = 0
        self.num_scalar_data = 0
        self.max_big_small_squared = 0
        for const in problem.constants() + problem.parameters():
            big = 0
            self.num_scalar_data += const.size
            big = 1 if len(const.shape) == 0 else max(const.shape)
            small = 1 if len(const.shape) == 0 else min(const.shape)
            if self.max_data_dimension < big:
                self.max_data_dimension = big
            max_big_small_squared = float(big) * float(small) ** 2
            if self.max_big_small_squared < max_big_small_squared:
                self.max_big_small_squared = max_big_small_squared
        self.num_scalar_eq_constr = 0
        for constraint in problem.constraints:
            if isinstance(constraint, (Equality, Zero)):
                self.num_scalar_eq_constr += constraint.expr.size
        self.num_scalar_leq_constr = 0
        for constraint in problem.constraints:
            if isinstance(constraint, (Inequality, NonPos, NonNeg)):
                self.num_scalar_leq_constr += constraint.expr.size