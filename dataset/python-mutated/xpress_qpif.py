import numpy as np
import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.conic_solvers.xpress_conif import get_status_maps, makeMstart
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver

class XPRESS(QpSolver):
    """Quadratic interface for the FICO Xpress solver"""
    MIP_CAPABLE = True

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self.prob_ = None

    def name(self):
        if False:
            return 10
        return s.XPRESS

    def import_solver(self) -> None:
        if False:
            print('Hello World!')
        import xpress

    def apply(self, problem):
        if False:
            i = 10
            return i + 15
        'Returns a new problem and data for inverting the new solution.\n\n        Returns\n        -------\n        tuple\n            (dict of arguments needed for the solver, inverse data)\n        '
        'Returns a new problem and data for inverting the new solution.\n\n        Returns\n        -------\n        tuple\n            (dict of arguments needed for the solver, inverse data)\n        '
        (data, inv_data) = super(XPRESS, self).apply(problem)
        return (data, inv_data)

    def invert(self, results, inverse_data):
        if False:
            while True:
                i = 10
        attr = {}
        if s.SOLVE_TIME in results:
            attr[s.SOLVE_TIME] = results[s.SOLVE_TIME]
        attr[s.NUM_ITERS] = int(results['bariter']) if not inverse_data[XPRESS.IS_MIP] else 0
        (status_map_lp, status_map_mip) = get_status_maps()
        if results['status'] == 'solver_error':
            status = 'solver_error'
        elif 'mip_' in results['getProbStatusString']:
            status = status_map_mip[results['status']]
        else:
            status = status_map_lp[results['status']]
        if status in s.SOLUTION_PRESENT:
            opt_val = results['getObjVal'] + inverse_data[s.OFFSET]
            x = np.array(results['getSolution'])
            primal_vars = {XPRESS.VAR_ID: intf.DEFAULT_INTF.const_to_matrix(np.array(x))}
            dual_vars = None
            if not inverse_data[XPRESS.IS_MIP]:
                y = -np.array(results['getDual'])
                dual_vars = {XPRESS.DUAL_VAR_ID: y}
            sol = Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            sol = failure_solution(status, attr)
        return sol

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        if False:
            return 10
        import xpress as xp
        Q = data[s.P]
        q = data[s.Q]
        A = data[s.A]
        b = data[s.B]
        n_var = data['n_var']
        n_eq = data['n_eq']
        self.prob_ = xp.problem()
        mstart = makeMstart(A, n_var, 1)
        if len(Q.data) != 0:
            Q += Q.transpose()
            Q /= 2
            Q = Q.tocoo()
            mqcol1 = Q.row[Q.row <= Q.col]
            mqcol2 = Q.col[Q.row <= Q.col]
            dqe = Q.data[Q.row <= Q.col]
        else:
            (mqcol1, mqcol2, dqe) = ([], [], [])
        colnames = ['x_{0:09d}'.format(i) for i in range(n_var)]
        rownames = ['eq_{0:09d}'.format(i) for i in range(n_eq)]
        if verbose:
            self.prob_.controls.miplog = 2
            self.prob_.controls.lplog = 1
            self.prob_.controls.outputlog = 1
        else:
            self.prob_.controls.miplog = 0
            self.prob_.controls.lplog = 0
            self.prob_.controls.outputlog = 0
            self.prob_.controls.xslp_log = -1
        self.prob_.loadproblem(probname='CVX_xpress_qp', qrtypes=['E'] * n_eq, rhs=b, range=None, obj=q, mstart=mstart, mnel=None, mrwind=A.indices[A.data != 0], dmatval=A.data[A.data != 0], dlb=[-xp.infinity] * len(q), dub=[xp.infinity] * len(q), mqcol1=mqcol1, mqcol2=mqcol2, dqe=dqe, qgtype=['B'] * len(data[s.BOOL_IDX]) + ['I'] * len(data[s.INT_IDX]), mgcols=data[s.BOOL_IDX] + data[s.INT_IDX], colnames=colnames, rownames=rownames)
        n_ineq = data['n_ineq']
        if n_ineq > 0:
            F = data[s.F].tocsr()
            g = data[s.G]
            mstartIneq = makeMstart(F, n_ineq, 0)
            rownames_ineq = ['ineq_{0:09d}'.format(i) for i in range(n_ineq)]
            self.prob_.addrows(qrtype=['L'] * n_ineq, rhs=g, mstart=mstartIneq, mclind=F.indices[F.data != 0], dmatval=F.data[F.data != 0], names=rownames_ineq)
        self.prob_.setControl({i: solver_opts[i] for i in solver_opts if i in xp.controls.__dict__})
        if 'bargaptarget' not in solver_opts.keys():
            self.prob_.controls.bargaptarget = 1e-30
        if 'feastol' not in solver_opts.keys():
            self.prob_.controls.feastol = 1e-09
        results_dict = {'model': self.prob_}
        try:
            if 'write_mps' in solver_opts.keys():
                self.prob_.write(solver_opts['write_mps'])
            self.prob_.solve()
            results_dict[s.SOLVE_TIME] = self.prob_.attributes.time
        except xp.SolverError:
            results_dict['status'] = s.SOLVER_ERROR
        else:
            results_dict['status'] = self.prob_.getProbStatus()
            results_dict['getProbStatusString'] = self.prob_.getProbStatusString()
            results_dict['obj_value'] = self.prob_.getObjVal()
            try:
                results_dict[s.PRIMAL] = np.array(self.prob_.getSolution())
            except xp.SolverError:
                results_dict[s.PRIMAL] = np.zeros(self.prob_.attributes.ncol)
            (status_map_lp, status_map_mip) = get_status_maps()
            if results_dict['status'] == 'solver_error':
                status = 'solver_error'
            elif 'mip_' in results_dict['getProbStatusString']:
                status = status_map_mip[results_dict['status']]
            else:
                status = status_map_lp[results_dict['status']]
            results_dict['bariter'] = self.prob_.attributes.bariter
            results_dict['getProbStatusString'] = self.prob_.getProbStatusString()
            if status in s.SOLUTION_PRESENT:
                results_dict['getObjVal'] = self.prob_.getObjVal()
                results_dict['getSolution'] = self.prob_.getSolution()
                if not (data[s.BOOL_IDX] or data[s.INT_IDX]):
                    results_dict['getDual'] = self.prob_.getDual()
        del self.prob_
        return results_dict