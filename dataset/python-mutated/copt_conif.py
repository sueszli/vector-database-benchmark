"""
This file is the CVXPY conic extension of the Cardinal Optimizer
"""
import numpy as np
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver, dims_to_solver_dict

def tri_to_full(lower_tri, n):
    if False:
        while True:
            i = 10
    '\n    Expands n*(n+1)//2 lower triangular to full matrix\n\n    Parameters\n    ----------\n    lower_tri : numpy.ndarray\n        A NumPy array representing the lower triangular part of the\n        matrix, stacked in column-major order.\n    n : int\n        The number of rows (columns) in the full square matrix.\n\n    Returns\n    -------\n    numpy.ndarray\n        A 2-dimensional ndarray that is the scaled expansion of the lower\n        triangular array.\n    '
    full = np.zeros((n, n))
    full[np.triu_indices(n)] = lower_tri
    full += full.T
    full[np.diag_indices(n)] /= 2.0
    return np.reshape(full, n * n, order='F')

class COPT(ConicSolver):
    """
    An interface for the COPT solver.
    """
    MIP_CAPABLE = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC, PSD]
    REQUIRES_CONSTR = True
    MI_SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS
    STATUS_MAP = {1: s.OPTIMAL, 2: s.INFEASIBLE, 3: s.UNBOUNDED, 4: s.INF_OR_UNB, 5: s.SOLVER_ERROR, 6: s.USER_LIMIT, 7: s.OPTIMAL_INACCURATE, 8: s.USER_LIMIT, 9: s.SOLVER_ERROR, 10: s.USER_LIMIT}

    def name(self):
        if False:
            while True:
                i = 10
        '\n        The name of solver.\n        '
        return 'COPT'

    def import_solver(self):
        if False:
            print('Hello World!')
        '\n        Imports the solver.\n        '
        import coptpy

    def accepts(self, problem):
        if False:
            print('Hello World!')
        '\n        Can COPT solve the problem?\n        '
        if not problem.objective.args[0].is_affine():
            return False
        for constr in problem.constraints:
            if type(constr) not in self.SUPPORTED_CONSTRAINTS:
                return False
            for arg in constr.args:
                if not arg.is_affine():
                    return False
        return True

    @staticmethod
    def psd_format_mat(constr):
        if False:
            i = 10
            return i + 15
        '\n        Return a linear operator to multiply by PSD constraint coefficients.\n\n        Special cases PSD constraints, as COPT expects constraints to be\n        imposed on solely the lower triangular part of the variable matrix.\n        '
        rows = cols = constr.expr.shape[0]
        entries = rows * (cols + 1) // 2
        row_arr = np.arange(0, entries)
        lower_diag_indices = np.tril_indices(rows)
        col_arr = np.sort(np.ravel_multi_index(lower_diag_indices, (rows, cols), order='F'))
        val_arr = np.zeros((rows, cols))
        val_arr[lower_diag_indices] = 1.0
        np.fill_diagonal(val_arr, 1.0)
        val_arr = np.ravel(val_arr, order='F')
        val_arr = val_arr[np.nonzero(val_arr)]
        shape = (entries, rows * cols)
        scaled_lower_tri = sp.csc_matrix((val_arr, (row_arr, col_arr)), shape)
        idx = np.arange(rows * cols)
        val_symm = 0.5 * np.ones(2 * rows * cols)
        K = idx.reshape((rows, cols))
        row_symm = np.append(idx, np.ravel(K, order='F'))
        col_symm = np.append(idx, np.ravel(K.T, order='F'))
        symm_matrix = sp.csc_matrix((val_symm, (row_symm, col_symm)))
        return scaled_lower_tri @ symm_matrix

    @staticmethod
    def extract_dual_value(result_vec, offset, constraint):
        if False:
            print('Hello World!')
        '\n        Extracts the dual value for constraint starting at offset.\n\n        Special cases PSD constraints, as per the COPT specification.\n        '
        if isinstance(constraint, PSD):
            dim = constraint.shape[0]
            lower_tri_dim = dim * (dim + 1) // 2
            new_offset = offset + lower_tri_dim
            lower_tri = result_vec[offset:new_offset]
            full = tri_to_full(lower_tri, dim)
            return (full, new_offset)
        else:
            return utilities.extract_dual_value(result_vec, offset, constraint)

    def apply(self, problem):
        if False:
            print('Hello World!')
        '\n        Returns a new problem and data for inverting the new solution.\n\n        Returns\n        -------\n        tuple\n            (dict of arguments needed for the solver, inverse data)\n        '
        (data, inv_data) = super(COPT, self).apply(problem)
        variables = problem.x
        data[s.BOOL_IDX] = [int(t[0]) for t in variables.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in variables.integer_idx]
        inv_data['is_mip'] = data[s.BOOL_IDX] or data[s.INT_IDX]
        return (data, inv_data)

    def invert(self, solution, inverse_data):
        if False:
            print('Hello World!')
        '\n        Returns the solution to the original problem given the inverse_data.\n        '
        status = solution[s.STATUS]
        attr = {s.SOLVE_TIME: solution[s.SOLVE_TIME], s.NUM_ITERS: solution[s.NUM_ITERS], s.EXTRA_STATS: solution['model']}
        primal_vars = None
        dual_vars = None
        if status in s.SOLUTION_PRESENT:
            opt_val = solution[s.VALUE] + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[COPT.VAR_ID]: solution[s.PRIMAL]}
            if not inverse_data['is_mip']:
                eq_dual = utilities.get_dual_values(solution[s.EQ_DUAL], self.extract_dual_value, inverse_data[COPT.EQ_CONSTR])
                leq_dual = utilities.get_dual_values(solution[s.INEQ_DUAL], self.extract_dual_value, inverse_data[COPT.NEQ_CONSTR])
                eq_dual.update(leq_dual)
                dual_vars = eq_dual
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        if False:
            print('Hello World!')
        '\n        Returns the result of the call to the solver.\n\n        Parameters\n        ----------\n        data : dict\n            Data used by the solver.\n        warm_start : bool\n            Not used.\n        verbose : bool\n            Should the solver print output?\n        solver_opts : dict\n            Additional arguments for the solver.\n        solver_cache: None\n            None\n\n        Returns\n        -------\n        tuple\n            (status, optimal value, primal, equality dual, inequality dual)\n        '
        import coptpy as copt
        envconfig = copt.EnvrConfig()
        if not verbose:
            envconfig.set('nobanner', '1')
        env = copt.Envr(envconfig)
        model = env.createModel()
        model.setParam(copt.COPT.Param.Logging, verbose)
        dims = dims_to_solver_dict(data[s.DIMS])
        rowmap = None
        if dims[s.PSD_DIM]:
            c = data[s.C]
            A = data[s.A]
            b = data[s.B]
            rowmap = model.loadConeMatrix(-b, A.transpose().tocsc(), -c, dims)
            model.objsense = copt.COPT.MAXIMIZE
        else:
            n = data[s.C].shape[0]
            c = data[s.C]
            A = data[s.A]
            lhs = np.copy(data[s.B])
            lhs[range(dims[s.EQ_DIM], dims[s.EQ_DIM] + dims[s.LEQ_DIM])] = -copt.COPT.INFINITY
            rhs = np.copy(data[s.B])
            lb = np.full(n, -copt.COPT.INFINITY)
            ub = np.full(n, +copt.COPT.INFINITY)
            vtype = None
            if data[s.BOOL_IDX] or data[s.INT_IDX]:
                vtype = np.array([copt.COPT.CONTINUOUS] * n)
                if data[s.BOOL_IDX]:
                    vtype[data[s.BOOL_IDX]] = copt.COPT.BINARY
                    lb[data[s.BOOL_IDX]] = 0
                    ub[data[s.BOOL_IDX]] = 1
                if data[s.INT_IDX]:
                    vtype[data[s.INT_IDX]] = copt.COPT.INTEGER
            ncone = 0
            nconedim = 0
            if dims[s.SOC_DIM]:
                ncone = len(dims[s.SOC_DIM])
                nconedim = sum(dims[s.SOC_DIM])
                nlinrow = dims[s.EQ_DIM] + dims[s.LEQ_DIM]
                nlincol = A.shape[1]
                diag = sp.spdiags(np.ones(nconedim), -nlinrow, A.shape[0], nconedim)
                A = sp.csc_matrix(sp.hstack([A, diag]))
                c = np.append(c, np.zeros(nconedim))
                lb = np.append(lb, -copt.COPT.INFINITY * np.ones(nconedim))
                ub = np.append(ub, +copt.COPT.INFINITY * np.ones(nconedim))
                lb[nlincol] = 0.0
                if len(dims[s.SOC_DIM]) > 1:
                    for dim in dims[s.SOC_DIM][:-1]:
                        nlincol += dim
                        lb[nlincol] = 0.0
                if data[s.BOOL_IDX] or data[s.INT_IDX]:
                    vtype = np.append(vtype, [copt.COPT.CONTINUOUS] * nconedim)
            model.loadMatrix(c, A, lhs, rhs, lb, ub, vtype)
            if dims[s.SOC_DIM]:
                model.loadCone(ncone, None, dims[s.SOC_DIM], range(A.shape[1] - nconedim, A.shape[1]))
        for (key, value) in solver_opts.items():
            model.setParam(key, value)
        solution = {}
        try:
            model.solve()
            if model.status == copt.COPT.INF_OR_UNB and solver_opts.get('reoptimize', True):
                model.setParam(copt.COPT.Param.Presolve, 0)
                model.solve()
            if dims[s.PSD_DIM]:
                if model.haslpsol:
                    solution[s.VALUE] = model.objval
                    nrow = len(c)
                    duals = model.getDuals()
                    psdduals = model.getPsdDuals()
                    y = np.zeros(nrow)
                    for i in range(nrow):
                        if rowmap[i] < 0:
                            y[i] = -psdduals[-rowmap[i] - 1]
                        else:
                            y[i] = -duals[rowmap[i] - 1]
                    solution[s.PRIMAL] = y
                    solution['y'] = np.hstack((model.getValues(), model.getPsdValues()))
                    solution[s.EQ_DUAL] = solution['y'][0:dims[s.EQ_DIM]]
                    solution[s.INEQ_DUAL] = solution['y'][dims[s.EQ_DIM]:]
            else:
                if model.haslpsol or model.hasmipsol:
                    solution[s.VALUE] = model.objval
                    solution[s.PRIMAL] = np.array(model.getValues())
                if not (data[s.BOOL_IDX] or data[s.INT_IDX]) and model.haslpsol:
                    solution['y'] = -np.array(model.getDuals())
                    solution[s.EQ_DUAL] = solution['y'][0:dims[s.EQ_DIM]]
                    solution[s.INEQ_DUAL] = solution['y'][dims[s.EQ_DIM]:]
        except Exception:
            pass
        solution[s.SOLVE_TIME] = model.solvingtime
        solution[s.NUM_ITERS] = model.barrieriter + model.simplexiter
        if dims[s.PSD_DIM]:
            if model.status == copt.COPT.INFEASIBLE:
                solution[s.STATUS] = s.UNBOUNDED
            elif model.status == copt.COPT.UNBOUNDED:
                solution[s.STATUS] = s.INFEASIBLE
            else:
                solution[s.STATUS] = self.STATUS_MAP.get(model.status, s.SOLVER_ERROR)
        else:
            solution[s.STATUS] = self.STATUS_MAP.get(model.status, s.SOLVER_ERROR)
        if solution[s.STATUS] == s.USER_LIMIT and model.hasmipsol:
            solution[s.STATUS] = s.OPTIMAL_INACCURATE
        if solution[s.STATUS] == s.USER_LIMIT and (not model.hasmipsol):
            solution[s.STATUS] = s.INFEASIBLE_INACCURATE
        solution['model'] = model
        return solution