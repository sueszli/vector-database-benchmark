import numpy as np
import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.conic_solvers.cplex_conif import get_status, hide_solver_output, set_parameters
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver

def constrain_cplex_infty(v) -> None:
    if False:
        while True:
            i = 10
    '\n    Limit values of vector v between +/- infinity as\n    defined in the CPLEX package\n    '
    import cplex as cpx
    n = len(v)
    for i in range(n):
        if v[i] >= cpx.infinity:
            v[i] = cpx.infinity
        if v[i] <= -cpx.infinity:
            v[i] = -cpx.infinity

class CPLEX(QpSolver):
    """QP interface for the CPLEX solver"""
    MIP_CAPABLE = True

    def name(self):
        if False:
            i = 10
            return i + 15
        return s.CPLEX

    def import_solver(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        import cplex
        cplex

    def invert(self, results, inverse_data):
        if False:
            for i in range(10):
                print('nop')
        model = results['model']
        attr = {}
        if 'cputime' in results:
            attr[s.SOLVE_TIME] = results['cputime']
        attr[s.NUM_ITERS] = int(model.solution.progress.get_num_barrier_iterations()) if not inverse_data[CPLEX.IS_MIP] else 0
        status = get_status(model)
        if status in s.SOLUTION_PRESENT:
            opt_val = model.solution.get_objective_value() + inverse_data[s.OFFSET]
            x = np.array(model.solution.get_values())
            primal_vars = {CPLEX.VAR_ID: intf.DEFAULT_INTF.const_to_matrix(np.array(x))}
            dual_vars = None
            if not inverse_data[CPLEX.IS_MIP]:
                y = -np.array(model.solution.get_dual_values())
                dual_vars = {CPLEX.DUAL_VAR_ID: y}
            sol = Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            sol = failure_solution(status, attr)
        return sol

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        if False:
            for i in range(10):
                print('nop')
        import cplex as cpx
        P = data[s.P].tocsr()
        q = data[s.Q]
        A = data[s.A].tocsr()
        b = data[s.B]
        F = data[s.F].tocsr()
        g = data[s.G]
        n_var = data['n_var']
        n_eq = data['n_eq']
        n_ineq = data['n_ineq']
        constrain_cplex_infty(b)
        constrain_cplex_infty(g)
        model = cpx.Cplex()
        model.objective.set_sense(model.objective.sense.minimize)
        var_idx = list(model.variables.add(obj=q, lb=-cpx.infinity * np.ones(n_var), ub=cpx.infinity * np.ones(n_var)))
        for i in data[s.BOOL_IDX]:
            model.variables.set_types(var_idx[i], model.variables.type.binary)
        for i in data[s.INT_IDX]:
            model.variables.set_types(var_idx[i], model.variables.type.integer)
        (lin_expr, rhs) = ([], [])
        for i in range(n_eq):
            start = A.indptr[i]
            end = A.indptr[i + 1]
            lin_expr.append([A.indices[start:end].tolist(), A.data[start:end].tolist()])
            rhs.append(b[i])
        if lin_expr:
            model.linear_constraints.add(lin_expr=lin_expr, senses=['E'] * len(lin_expr), rhs=rhs)
        (lin_expr, rhs) = ([], [])
        for i in range(n_ineq):
            start = F.indptr[i]
            end = F.indptr[i + 1]
            lin_expr.append([F.indices[start:end].tolist(), F.data[start:end].tolist()])
            rhs.append(g[i])
        if lin_expr:
            model.linear_constraints.add(lin_expr=lin_expr, senses=['L'] * len(lin_expr), rhs=rhs)
        if P.count_nonzero():
            qmat = []
            for i in range(n_var):
                start = P.indptr[i]
                end = P.indptr[i + 1]
                qmat.append([P.indices[start:end].tolist(), P.data[start:end].tolist()])
            model.objective.set_quadratic(qmat)
        if not verbose:
            hide_solver_output(model)
        reoptimize = solver_opts.pop('reoptimize', False)
        set_parameters(model, solver_opts)
        results_dict = {}
        try:
            start = model.get_time()
            model.solve()
            end = model.get_time()
            results_dict['cputime'] = end - start
            ambiguous_status = get_status(model) == s.INFEASIBLE_OR_UNBOUNDED
            if ambiguous_status and reoptimize:
                model.parameters.preprocessing.presolve.set(0)
                start_time = model.get_time()
                model.solve()
                results_dict['cputime'] += model.get_time() - start_time
        except Exception:
            results_dict['status'] = s.SOLVER_ERROR
        results_dict['model'] = model
        return results_dict