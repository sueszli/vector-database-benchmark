import numpy as np
import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver

def constrain_gurobi_infty(v) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Limit values of vector v between +/- infinity as\n    defined in the Gurobi package\n    '
    import gurobipy as grb
    n = len(v)
    for i in range(n):
        if v[i] >= 1e+20:
            v[i] = grb.GRB.INFINITY
        if v[i] <= -1e+20:
            v[i] = -grb.GRB.INFINITY

class GUROBI(QpSolver):
    """QP interface for the Gurobi solver"""
    MIP_CAPABLE = True
    STATUS_MAP = {2: s.OPTIMAL, 3: s.INFEASIBLE, 5: s.UNBOUNDED, 4: s.INFEASIBLE_OR_UNBOUNDED, 6: s.INFEASIBLE, 7: s.SOLVER_ERROR, 8: s.SOLVER_ERROR, 9: s.USER_LIMIT, 10: s.SOLVER_ERROR, 11: s.SOLVER_ERROR, 12: s.SOLVER_ERROR, 13: s.OPTIMAL_INACCURATE}

    def name(self):
        if False:
            while True:
                i = 10
        return s.GUROBI

    def import_solver(self) -> None:
        if False:
            print('Hello World!')
        import gurobipy
        gurobipy

    def apply(self, problem):
        if False:
            return 10
        "\n        Construct QP problem data stored in a dictionary.\n        The QP has the following form\n\n            minimize      1/2 x' P x + q' x\n            subject to    A x =  b\n                          F x <= g\n\n        "
        import gurobipy as grb
        (data, inv_data) = super(GUROBI, self).apply(problem)
        data['init_value'] = utilities.stack_vals(problem.variables, grb.GRB.UNDEFINED)
        return (data, inv_data)

    def invert(self, results, inverse_data):
        if False:
            i = 10
            return i + 15
        model = results['model']
        x_grb = model.getVars()
        n = len(x_grb)
        constraints_grb = model.getConstrs()
        m = len(constraints_grb)
        try:
            bar_iter_count = model.BarIterCount
        except AttributeError:
            bar_iter_count = 0
        try:
            simplex_iter_count = model.IterCount
        except AttributeError:
            simplex_iter_count = 0
        iter_count = bar_iter_count + simplex_iter_count
        attr = {s.SOLVE_TIME: model.Runtime, s.NUM_ITERS: iter_count, s.EXTRA_STATS: model}
        status = self.STATUS_MAP.get(model.Status, s.SOLVER_ERROR)
        if status == s.USER_LIMIT and (not model.SolCount):
            status = s.INFEASIBLE_INACCURATE
        if status in s.SOLUTION_PRESENT or model.solCount > 0:
            opt_val = model.objVal + inverse_data[s.OFFSET]
            x = np.array([x_grb[i].X for i in range(n)])
            primal_vars = {GUROBI.VAR_ID: intf.DEFAULT_INTF.const_to_matrix(np.array(x))}
            dual_vars = None
            if not inverse_data[GUROBI.IS_MIP]:
                y = -np.array([constraints_grb[i].Pi for i in range(m)])
                dual_vars = {GUROBI.DUAL_VAR_ID: y}
            sol = Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            sol = failure_solution(status, attr)
        return sol

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        if False:
            return 10
        import gurobipy as grb
        P = data[s.P]
        q = data[s.Q]
        A = data[s.A].tocsr()
        b = data[s.B]
        F = data[s.F].tocsr()
        g = data[s.G]
        n = data['n_var']
        constrain_gurobi_infty(b)
        constrain_gurobi_infty(g)
        if 'env' in solver_opts:
            default_env = solver_opts['env']
            del solver_opts['env']
            model = grb.Model(env=default_env)
        else:
            model = grb.Model()
        model.setParam('OutputFlag', verbose)
        vtypes = {}
        for ind in data[s.BOOL_IDX]:
            vtypes[ind] = grb.GRB.BINARY
        for ind in data[s.INT_IDX]:
            vtypes[ind] = grb.GRB.INTEGER
        for i in range(n):
            if i not in vtypes:
                vtypes[i] = grb.GRB.CONTINUOUS
        x_grb = model.addVars(int(n), ub={i: grb.GRB.INFINITY for i in range(n)}, lb={i: -grb.GRB.INFINITY for i in range(n)}, vtype=vtypes)
        if warm_start and solver_cache is not None and (self.name() in solver_cache):
            old_model = solver_cache[self.name()]
            old_status = self.STATUS_MAP.get(old_model.Status, s.SOLVER_ERROR)
            if old_status in s.SOLUTION_PRESENT or old_model.solCount > 0:
                old_x_grb = old_model.getVars()
                for idx in range(len(x_grb)):
                    x_grb[idx].start = old_x_grb[idx].X
        elif warm_start:
            for idx in range(len(x_grb)):
                x_grb[idx].start = data['init_value'][idx]
        model.update()
        x = np.array(model.getVars(), copy=False)
        if A.shape[0] > 0:
            if hasattr(model, 'addMConstr'):
                model.addMConstr(A, None, grb.GRB.EQUAL, b)
            elif hasattr(model, 'addMConstrs'):
                model.addMConstrs(A, None, grb.GRB.EQUAL, b)
            else:
                for i in range(A.shape[0]):
                    start = A.indptr[i]
                    end = A.indptr[i + 1]
                    variables = x[A.indices[start:end]]
                    coeff = A.data[start:end]
                    expr = grb.LinExpr(coeff, variables)
                    model.addConstr(expr, grb.GRB.EQUAL, b[i])
        model.update()
        if F.shape[0] > 0:
            if hasattr(model, 'addMConstr'):
                model.addMConstr(F, None, grb.GRB.LESS_EQUAL, g)
            elif hasattr(model, 'addMConstrs'):
                model.addMConstrs(F, None, grb.GRB.LESS_EQUAL, g)
            else:
                for i in range(F.shape[0]):
                    start = F.indptr[i]
                    end = F.indptr[i + 1]
                    variables = x[F.indices[start:end]]
                    coeff = F.data[start:end]
                    expr = grb.LinExpr(coeff, variables)
                    model.addConstr(expr, grb.GRB.LESS_EQUAL, g[i])
        model.update()
        if hasattr(model, 'setMObjective'):
            P = P.tocoo()
            model.setMObjective(0.5 * P, q, 0.0)
        elif hasattr(model, '_v811_setMObjective'):
            P = P.tocoo()
            model._v811_setMObjective(0.5 * P, q)
        else:
            obj = grb.QuadExpr()
            if P.count_nonzero():
                P = P.tocoo()
                obj.addTerms(0.5 * P.data, vars=list(x[P.row]), vars2=list(x[P.col]))
            obj.add(grb.LinExpr(q, x))
            model.setObjective(obj)
        model.update()
        model.setParam('QCPDual', True)
        for (key, value) in solver_opts.items():
            model.setParam(key, value)
        model.update()
        if 'save_file' in solver_opts:
            model.write(solver_opts['save_file'])
        results_dict = {}
        try:
            model.optimize()
            if model.Status == 4 and solver_opts.get('reoptimize', False):
                model.setParam('DualReductions', 0)
                model.optimize()
        except Exception:
            results_dict['status'] = s.SOLVER_ERROR
        results_dict['model'] = model
        if solver_cache is not None:
            solver_cache[self.name()] = model
        return results_dict