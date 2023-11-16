"""
Copyright 2013 Steven Diamond, 2017 Robin Verschueren

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
import numpy as np
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.constraints import SOC
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver, dims_to_solver_dict

class GUROBI(ConicSolver):
    """
    An interface for the Gurobi solver.
    """
    MIP_CAPABLE = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC]
    MI_SUPPORTED_CONSTRAINTS = SUPPORTED_CONSTRAINTS
    STATUS_MAP = {2: s.OPTIMAL, 3: s.INFEASIBLE, 4: s.INFEASIBLE_OR_UNBOUNDED, 5: s.UNBOUNDED, 6: s.SOLVER_ERROR, 7: s.SOLVER_ERROR, 8: s.SOLVER_ERROR, 9: s.USER_LIMIT, 10: s.SOLVER_ERROR, 11: s.SOLVER_ERROR, 12: s.SOLVER_ERROR, 13: s.SOLVER_ERROR}

    def name(self):
        if False:
            i = 10
            return i + 15
        'The name of the solver.\n        '
        return s.GUROBI

    def import_solver(self) -> None:
        if False:
            print('Hello World!')
        'Imports the solver.\n        '
        import gurobipy

    def accepts(self, problem) -> bool:
        if False:
            print('Hello World!')
        'Can Gurobi solve the problem?\n        '
        if not problem.objective.args[0].is_affine():
            return False
        for constr in problem.constraints:
            if type(constr) not in self.SUPPORTED_CONSTRAINTS:
                return False
            for arg in constr.args:
                if not arg.is_affine():
                    return False
        return True

    def apply(self, problem):
        if False:
            print('Hello World!')
        'Returns a new problem and data for inverting the new solution.\n\n        Returns\n        -------\n        tuple\n            (dict of arguments needed for the solver, inverse data)\n        '
        import gurobipy as grb
        (data, inv_data) = super(GUROBI, self).apply(problem)
        variables = problem.x
        data[s.BOOL_IDX] = [int(t[0]) for t in variables.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in variables.integer_idx]
        inv_data['is_mip'] = data[s.BOOL_IDX] or data[s.INT_IDX]
        data['init_value'] = utilities.stack_vals(problem.variables, grb.GRB.UNDEFINED)
        return (data, inv_data)

    def invert(self, solution, inverse_data):
        if False:
            return 10
        'Returns the solution to the original problem given the inverse_data.\n        '
        status = solution['status']
        attr = {s.EXTRA_STATS: solution['model'], s.SOLVE_TIME: solution[s.SOLVE_TIME]}
        primal_vars = None
        dual_vars = None
        if status in s.SOLUTION_PRESENT:
            opt_val = solution['value'] + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[GUROBI.VAR_ID]: solution['primal']}
            if 'eq_dual' in solution and (not inverse_data['is_mip']):
                eq_dual = utilities.get_dual_values(solution['eq_dual'], utilities.extract_dual_value, inverse_data[GUROBI.EQ_CONSTR])
                leq_dual = utilities.get_dual_values(solution['ineq_dual'], utilities.extract_dual_value, inverse_data[GUROBI.NEQ_CONSTR])
                eq_dual.update(leq_dual)
                dual_vars = eq_dual
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns the result of the call to the solver.\n\n        Parameters\n        ----------\n        data : dict\n            Data used by the solver.\n        warm_start : bool\n            Not used.\n        verbose : bool\n            Should the solver print output?\n        solver_opts : dict\n            Additional arguments for the solver.\n\n        Returns\n        -------\n        tuple\n            (status, optimal value, primal, equality dual, inequality dual)\n        '
        import gurobipy
        c = data[s.C]
        b = data[s.B]
        A = sp.csr_matrix(data[s.A])
        dims = dims_to_solver_dict(data[s.DIMS])
        n = c.shape[0]
        if 'env' in solver_opts:
            default_env = solver_opts['env']
            del solver_opts['env']
            model = gurobipy.Model(env=default_env)
        else:
            model = gurobipy.Model()
        model.setParam('OutputFlag', verbose)
        variables = []
        for i in range(n):
            if i in data[s.BOOL_IDX]:
                vtype = gurobipy.GRB.BINARY
            elif i in data[s.INT_IDX]:
                vtype = gurobipy.GRB.INTEGER
            else:
                vtype = gurobipy.GRB.CONTINUOUS
            variables.append(model.addVar(obj=c[i], name='x_%d' % i, vtype=vtype, lb=-gurobipy.GRB.INFINITY, ub=gurobipy.GRB.INFINITY))
        model.update()
        x = model.getVars()
        if warm_start and solver_cache is not None and (self.name() in solver_cache):
            old_model = solver_cache[self.name()]
            old_status = self.STATUS_MAP.get(old_model.Status, s.SOLVER_ERROR)
            if old_status in s.SOLUTION_PRESENT or old_model.solCount > 0:
                old_x = old_model.getVars()
                for idx in range(len(x)):
                    x[idx].start = old_x[idx].X
        elif warm_start:
            for i in range(len(x)):
                x[i].start = data['init_value'][i]
        leq_start = dims[s.EQ_DIM]
        leq_end = dims[s.EQ_DIM] + dims[s.LEQ_DIM]
        if hasattr(model, 'addMConstr'):
            eq_constrs = model.addMConstr(A[:leq_start, :], None, gurobipy.GRB.EQUAL, b[:leq_start]).tolist()
            ineq_constrs = model.addMConstr(A[leq_start:leq_end, :], None, gurobipy.GRB.LESS_EQUAL, b[leq_start:leq_end]).tolist()
        elif hasattr(model, 'addMConstrs'):
            eq_constrs = model.addMConstrs(A[:leq_start, :], None, gurobipy.GRB.EQUAL, b[:leq_start])
            ineq_constrs = model.addMConstrs(A[leq_start:leq_end, :], None, gurobipy.GRB.LESS_EQUAL, b[leq_start:leq_end])
        else:
            eq_constrs = self.add_model_lin_constr(model, variables, range(dims[s.EQ_DIM]), gurobipy.GRB.EQUAL, A, b)
            ineq_constrs = self.add_model_lin_constr(model, variables, range(leq_start, leq_end), gurobipy.GRB.LESS_EQUAL, A, b)
        soc_start = leq_end
        soc_constrs = []
        new_leq_constrs = []
        for constr_len in dims[s.SOC_DIM]:
            soc_end = soc_start + constr_len
            (soc_constr, new_leq, new_vars) = self.add_model_soc_constr(model, variables, range(soc_start, soc_end), A, b)
            soc_constrs.append(soc_constr)
            new_leq_constrs += new_leq
            variables += new_vars
            soc_start += constr_len
        if 'save_file' in solver_opts:
            model.write(solver_opts['save_file'])
        model.setParam('QCPDual', True)
        for (key, value) in solver_opts.items():
            model.setParam(key, value)
        solution = {}
        try:
            model.optimize()
            if model.Status == 4 and solver_opts.get('reoptimize', False):
                model.setParam('DualReductions', 0)
                model.optimize()
            solution['value'] = model.ObjVal
            solution['primal'] = np.array([v.X for v in variables])
            vals = []
            if not (data[s.BOOL_IDX] or data[s.INT_IDX]):
                lin_constrs = eq_constrs + ineq_constrs + new_leq_constrs
                vals += model.getAttr('Pi', lin_constrs)
                vals += model.getAttr('QCPi', soc_constrs)
                solution['y'] = -np.array(vals)
                solution[s.EQ_DUAL] = solution['y'][0:dims[s.EQ_DIM]]
                solution[s.INEQ_DUAL] = solution['y'][dims[s.EQ_DIM]:]
        except Exception:
            pass
        solution[s.SOLVE_TIME] = model.Runtime
        solution['status'] = self.STATUS_MAP.get(model.Status, s.SOLVER_ERROR)
        if solution['status'] == s.SOLVER_ERROR and model.SolCount:
            solution['status'] = s.OPTIMAL_INACCURATE
        if solution['status'] == s.USER_LIMIT and (not model.SolCount):
            solution['status'] = s.INFEASIBLE_INACCURATE
        solution['model'] = model
        if solver_cache is not None:
            solver_cache[self.name()] = model
        return solution

    def add_model_lin_constr(self, model, variables, rows, ctype, mat, vec):
        if False:
            while True:
                i = 10
        'Adds EQ/LEQ constraints to the model using the data from mat and vec.\n\n        Parameters\n        ----------\n        model : GUROBI model\n            The problem model.\n        variables : list\n            The problem variables.\n        rows : range\n            The rows to be constrained.\n        ctype : GUROBI constraint type\n            The type of constraint.\n        mat : SciPy COO matrix\n            The matrix representing the constraints.\n        vec : NDArray\n            The constant part of the constraints.\n\n        Returns\n        -------\n        list\n            A list of constraints.\n        '
        import gurobipy as gp
        constr = []
        for i in rows:
            start = mat.indptr[i]
            end = mat.indptr[i + 1]
            x = [variables[j] for j in mat.indices[start:end]]
            coeff = mat.data[start:end]
            expr = gp.LinExpr(coeff, x)
            constr.append(model.addLConstr(expr, ctype, vec[i]))
        return constr

    def add_model_soc_constr(self, model, variables, rows, mat, vec):
        if False:
            for i in range(10):
                print('nop')
        'Adds SOC constraint to the model using the data from mat and vec.\n\n        Parameters\n        ----------\n        model : GUROBI model\n            The problem model.\n        variables : list\n            The problem variables.\n        rows : range\n            The rows to be constrained.\n        mat : SciPy COO matrix\n            The matrix representing the constraints.\n        vec : NDArray\n            The constant part of the constraints.\n\n        Returns\n        -------\n        tuple\n            A tuple of (QConstr, list of Constr, and list of variables).\n        '
        import gurobipy as gp
        soc_vars = [model.addVar(obj=0, name='soc_t_%d' % rows[0], vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY)]
        for i in rows[1:]:
            soc_vars += [model.addVar(obj=0, name='soc_x_%d' % i, vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)]
        new_lin_constrs = []
        for (i, row) in enumerate(rows):
            start = mat.indptr[row]
            end = mat.indptr[row + 1]
            x = [variables[j] for j in mat.indices[start:end]]
            coeff = -mat.data[start:end]
            expr = gp.LinExpr(coeff, x)
            expr.addConstant(vec[row])
            new_lin_constrs.append(model.addLConstr(soc_vars[i], gp.GRB.EQUAL, expr))
        t_term = soc_vars[0] * soc_vars[0]
        x_term = gp.QuadExpr()
        x_term.addTerms(np.ones(len(rows) - 1), soc_vars[1:], soc_vars[1:])
        return (model.addQConstr(x_term <= t_term), new_lin_constrs, soc_vars)