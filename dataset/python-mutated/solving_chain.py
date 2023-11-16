from __future__ import annotations
import warnings
import numpy as np
from cvxpy.atoms import EXP_ATOMS, NONPOS_ATOMS, PSD_ATOMS, SOC_ATOMS
from cvxpy.constraints import PSD, SOC, Equality, ExpCone, FiniteSet, Inequality, NonNeg, NonPos, PowCone3D, Zero
from cvxpy.constraints.exponential import OpRelEntrConeQuad, RelEntrConeQuad
from cvxpy.error import DCPError, DGPError, DPPError, SolverError
from cvxpy.problems.objective import Maximize
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.complex2real import complex2real
from cvxpy.reductions.cone2cone.approximations import APPROX_CONES, QuadApprox
from cvxpy.reductions.cone2cone.exotic2common import EXOTIC_CONES, Exotic2Common
from cvxpy.reductions.cone2cone.soc2psd import SOC2PSD
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.dgp2dcp.dgp2dcp import Dgp2Dcp
from cvxpy.reductions.discrete2mixedint.valinvec2mixedint import Valinvec2mixedint
from cvxpy.reductions.eval_params import EvalParams
from cvxpy.reductions.flip_objective import FlipObjective
from cvxpy.reductions.qp2quad_form import qp2symbolic_qp
from cvxpy.reductions.qp2quad_form.qp_matrix_stuffing import QpMatrixStuffing
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solvers import defines as slv_def
from cvxpy.reductions.solvers.constant_solver import ConstantSolver
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.settings import ECOS, PARAM_THRESHOLD
from cvxpy.utilities.debug_tools import build_non_disciplined_error_msg
DPP_ERROR_MSG = 'You are solving a parameterized problem that is not DPP. Because the problem is not DPP, subsequent solves will not be faster than the first one. For more information, see the documentation on Discplined Parametrized Programming, at\n\thttps://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming'
ECOS_DEPRECATION_MSG = '\n    Your problem is being solved with the ECOS solver by default. Starting in \n    CVXPY 1.5.0, Clarabel will be used as the default solver instead. To continue \n    using ECOS, specify the ECOS solver explicitly using the ``solver=cp.ECOS`` \n    argument to the ``problem.solve`` method.\n    '

def _is_lp(self):
    if False:
        for i in range(10):
            print('nop')
    'Is problem a linear program?\n    '
    for c in self.constraints:
        if not (isinstance(c, (Equality, Zero)) or c.args[0].is_pwl()):
            return False
    for var in self.variables():
        if var.is_psd() or var.is_nsd():
            return False
    return self.is_dcp() and self.objective.args[0].is_pwl()

def _solve_as_qp(problem, candidates):
    if False:
        for i in range(10):
            print('nop')
    if _is_lp(problem) and [s for s in candidates['conic_solvers'] if s not in candidates['qp_solvers']]:
        return False
    return candidates['qp_solvers'] and qp2symbolic_qp.accepts(problem)

def _reductions_for_problem_class(problem, candidates, gp: bool=False, solver_opts=None) -> list[Reduction]:
    if False:
        return 10
    '\n    Builds a chain that rewrites a problem into an intermediate\n    representation suitable for numeric reductions.\n\n    Parameters\n    ----------\n    problem : Problem\n        The problem for which to build a chain.\n    candidates : dict\n        Dictionary of candidate solvers divided in qp_solvers\n        and conic_solvers.\n    gp : bool\n        If True, the problem is parsed as a Disciplined Geometric Program\n        instead of as a Disciplined Convex Program.\n    Returns\n    -------\n    list of Reduction objects\n        A list of reductions that can be used to convert the problem to an\n        intermediate form.\n    Raises\n    ------\n    DCPError\n        Raised if the problem is not DCP and `gp` is False.\n    DGPError\n        Raised if the problem is not DGP and `gp` is True.\n    '
    reductions = []
    if complex2real.accepts(problem):
        reductions += [complex2real.Complex2Real()]
    if gp:
        reductions += [Dgp2Dcp()]
    if not gp and (not problem.is_dcp()):
        append = build_non_disciplined_error_msg(problem, 'DCP')
        if problem.is_dgp():
            append += '\nHowever, the problem does follow DGP rules. Consider calling solve() with `gp=True`.'
        elif problem.is_dqcp():
            append += '\nHowever, the problem does follow DQCP rules. Consider calling solve() with `qcp=True`.'
        raise DCPError('Problem does not follow DCP rules. Specifically:\n' + append)
    elif gp and (not problem.is_dgp()):
        append = build_non_disciplined_error_msg(problem, 'DGP')
        if problem.is_dcp():
            append += '\nHowever, the problem does follow DCP rules. Consider calling solve() with `gp=False`.'
        elif problem.is_dqcp():
            append += '\nHowever, the problem does follow DQCP rules. Consider calling solve() with `qcp=True`.'
        raise DGPError('Problem does not follow DGP rules.' + append)
    if type(problem.objective) == Maximize:
        reductions += [FlipObjective()]
    use_quad = True if solver_opts is None else solver_opts.get('use_quad_obj', True)
    if _solve_as_qp(problem, candidates) and use_quad:
        reductions += [CvxAttr2Constr(), qp2symbolic_qp.Qp2SymbolicQp()]
    elif not candidates['conic_solvers']:
        raise SolverError('Problem could not be reduced to a QP, and no conic solvers exist among candidate solvers (%s).' % candidates)
    constr_types = {type(c) for c in problem.constraints}
    if FiniteSet in constr_types:
        reductions += [Valinvec2mixedint()]
    return reductions

def construct_solving_chain(problem, candidates, gp: bool=False, enforce_dpp: bool=False, ignore_dpp: bool=False, canon_backend: str | None=None, solver_opts: dict | None=None, specified_solver: str | None=None) -> 'SolvingChain':
    if False:
        while True:
            i = 10
    "Build a reduction chain from a problem to an installed solver.\n\n    Note that if the supplied problem has 0 variables, then the solver\n    parameter will be ignored.\n\n    Parameters\n    ----------\n    problem : Problem\n        The problem for which to build a chain.\n    candidates : dict\n        Dictionary of candidate solvers divided in qp_solvers\n        and conic_solvers.\n    gp : bool\n        If True, the problem is parsed as a Disciplined Geometric Program\n        instead of as a Disciplined Convex Program.\n    enforce_dpp : bool, optional\n        When True, a DPPError will be thrown when trying to parse a non-DPP\n        problem (instead of just a warning). Defaults to False.\n    ignore_dpp : bool, optional\n        When True, DPP problems will be treated as non-DPP,\n        which may speed up compilation. Defaults to False.\n    canon_backend : str, optional\n        'CPP' (default) | 'SCIPY'\n        Specifies which backend to use for canonicalization, which can affect\n        compilation time. Defaults to None, i.e., selecting the default\n        backend.\n    solver_opts : dict, optional\n        Additional arguments to pass to the solver.\n    specified_solver: str, optional\n        A solver specified by the user.\n\n    Returns\n    -------\n    SolvingChain\n        A SolvingChain that can be used to solve the problem.\n\n    Raises\n    ------\n    SolverError\n        Raised if no suitable solver exists among the installed solvers, or\n        if the target solver is not installed.\n    "
    if len(problem.variables()) == 0:
        return SolvingChain(reductions=[ConstantSolver()])
    reductions = _reductions_for_problem_class(problem, candidates, gp, solver_opts)
    dpp_context = 'dcp' if not gp else 'dgp'
    if ignore_dpp or not problem.is_dpp(dpp_context):
        if ignore_dpp:
            reductions = [EvalParams()] + reductions
        elif not enforce_dpp:
            warnings.warn(DPP_ERROR_MSG)
            reductions = [EvalParams()] + reductions
        else:
            raise DPPError(DPP_ERROR_MSG)
    elif any((param.is_complex() for param in problem.parameters())):
        reductions = [EvalParams()] + reductions
    else:
        n_parameters = sum((np.prod(param.shape) for param in problem.parameters()))
        if n_parameters >= PARAM_THRESHOLD:
            warnings.warn("Your problem has too many parameters for efficient DPP compilation. We suggest setting 'ignore_dpp = True'.")
    use_quad = True if solver_opts is None else solver_opts.get('use_quad_obj', True)
    if _solve_as_qp(problem, candidates) and use_quad:
        solver = candidates['qp_solvers'][0]
        solver_instance = slv_def.SOLVER_MAP_QP[solver]
        reductions += [QpMatrixStuffing(canon_backend=canon_backend), solver_instance]
        return SolvingChain(reductions=reductions)
    if not candidates['conic_solvers']:
        raise SolverError('Problem could not be reduced to a QP, and no conic solvers exist among candidate solvers (%s).' % candidates)
    constr_types = set()
    for c in problem.constraints:
        constr_types.add(type(c))
    ex_cos = [ct for ct in constr_types if ct in EXOTIC_CONES]
    approx_cos = [ct for ct in constr_types if ct in APPROX_CONES]
    for co in ex_cos:
        sim_cos = EXOTIC_CONES[co]
        constr_types.update(sim_cos)
        constr_types.remove(co)
    for co in approx_cos:
        app_cos = APPROX_CONES[co]
        constr_types.update(app_cos)
        constr_types.remove(co)
    cones = []
    atoms = problem.atoms()
    if SOC in constr_types or any((atom in SOC_ATOMS for atom in atoms)):
        cones.append(SOC)
    if ExpCone in constr_types or any((atom in EXP_ATOMS for atom in atoms)):
        cones.append(ExpCone)
    if any((t in constr_types for t in [Inequality, NonPos, NonNeg])) or any((atom in NONPOS_ATOMS for atom in atoms)):
        cones.append(NonNeg)
    if Equality in constr_types or Zero in constr_types:
        cones.append(Zero)
    if PSD in constr_types or any((atom in PSD_ATOMS for atom in atoms)) or any((v.is_psd() or v.is_nsd() for v in problem.variables())):
        cones.append(PSD)
    if PowCone3D in constr_types:
        cones.append(PowCone3D)
    has_constr = len(cones) > 0 or len(problem.constraints) > 0
    for solver in candidates['conic_solvers']:
        solver_instance = slv_def.SOLVER_MAP_CONIC[solver]
        if problem.is_mixed_integer():
            supported_constraints = solver_instance.MI_SUPPORTED_CONSTRAINTS
        else:
            supported_constraints = solver_instance.SUPPORTED_CONSTRAINTS
        unsupported_constraints = [cone for cone in cones if cone not in supported_constraints]
        if has_constr or not solver_instance.REQUIRES_CONSTR:
            if ex_cos:
                reductions.append(Exotic2Common())
            if RelEntrConeQuad in approx_cos or OpRelEntrConeQuad in approx_cos:
                reductions.append(QuadApprox())
            if solver_opts is None:
                use_quad_obj = True
            else:
                use_quad_obj = solver_opts.get('use_quad_obj', True)
            quad_obj = use_quad_obj and solver_instance.supports_quad_obj() and problem.objective.expr.has_quadratic_term()
            reductions += [Dcp2Cone(quad_obj=quad_obj), CvxAttr2Constr()]
            if all((c in supported_constraints for c in cones)):
                if solver == ECOS and specified_solver is None:
                    warnings.warn(ECOS_DEPRECATION_MSG, FutureWarning)
                reductions += [ConeMatrixStuffing(quad_obj=quad_obj, canon_backend=canon_backend), solver_instance]
                return SolvingChain(reductions=reductions)
            elif all((c == SOC for c in unsupported_constraints)) and PSD in supported_constraints:
                reductions += [SOC2PSD(), ConeMatrixStuffing(quad_obj=quad_obj, canon_backend=canon_backend), solver_instance]
                return SolvingChain(reductions=reductions)
    raise SolverError('Either candidate conic solvers (%s) do not support the cones output by the problem (%s), or there are not enough constraints in the problem.' % (candidates['conic_solvers'], ', '.join([cone.__name__ for cone in cones])))

class SolvingChain(Chain):
    """A reduction chain that ends with a solver.

    Parameters
    ----------
    reductions : list[Reduction]
        A list of reductions. The last reduction in the list must be a solver
        instance.

    Attributes
    ----------
    reductions : list[Reduction]
        A list of reductions.
    solver : Solver
        The solver, i.e., reductions[-1].
    """

    def __init__(self, problem=None, reductions=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(SolvingChain, self).__init__(problem=problem, reductions=reductions)
        if not isinstance(self.reductions[-1], Solver):
            raise ValueError('Solving chains must terminate with a Solver.')
        self.solver = self.reductions[-1]

    def prepend(self, chain) -> 'SolvingChain':
        if False:
            print('Hello World!')
        '\n        Create and return a new SolvingChain by concatenating\n        chain with this instance.\n        '
        return SolvingChain(reductions=chain.reductions + self.reductions)

    def solve(self, problem, warm_start: bool, verbose: bool, solver_opts):
        if False:
            i = 10
            return i + 15
        'Solves the problem by applying the chain.\n\n        Applies each reduction in the chain to the problem, solves it,\n        and then inverts the chain to return a solution of the supplied\n        problem.\n\n        Parameters\n        ----------\n        problem : Problem\n            The problem to solve.\n        warm_start : bool\n            Whether to warm start the solver.\n        verbose : bool\n            Whether to enable solver verbosity.\n        solver_opts : dict\n            Solver specific options.\n\n        Returns\n        -------\n        solution : Solution\n            A solution to the problem.\n        '
        (data, inverse_data) = self.apply(problem)
        solution = self.solver.solve_via_data(data, warm_start, verbose, solver_opts)
        return self.invert(solution, inverse_data)

    def solve_via_data(self, problem, data, warm_start: bool=False, verbose: bool=False, solver_opts={}):
        if False:
            while True:
                i = 10
        'Solves the problem using the data output by the an apply invocation.\n\n        The semantics are:\n\n        .. code :: python\n\n            data, inverse_data = solving_chain.apply(problem)\n            solution = solving_chain.invert(solver_chain.solve_via_data(data, ...))\n\n        which is equivalent to writing\n\n        .. code :: python\n\n            solution = solving_chain.solve(problem, ...)\n\n        Parameters\n        ----------\n        problem : Problem\n            The problem to solve.\n        data : map\n            Data for the solver.\n        warm_start : bool\n            Whether to warm start the solver.\n        verbose : bool\n            Whether to enable solver verbosity.\n        solver_opts : dict\n            Solver specific options.\n\n        Returns\n        -------\n        raw solver solution\n            The information returned by the solver; this is not necessarily\n            a Solution object.\n        '
        return self.solver.solve_via_data(data, warm_start, verbose, solver_opts, problem._solver_cache)