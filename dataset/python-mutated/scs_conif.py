"""
Copyright 2013 Steven Diamond, 2017 Robin Verschueren, 2017 Akshay Agrawal

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
from cvxpy.constraints import PSD, SOC, ExpCone, PowCone3D
from cvxpy.expressions.expression import Expression
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.conic_solvers.conic_solver import dims_to_solver_dict as dims_to_solver_dict_default
from cvxpy.utilities.versioning import Version

def dims_to_solver_dict(cone_dims):
    if False:
        while True:
            i = 10
    cones = dims_to_solver_dict_default(cone_dims)
    import scs
    if Version(scs.__version__) >= Version('3.0.0'):
        cones['z'] = cones.pop('f')
    return cones

def tri_to_full(lower_tri, n):
    if False:
        return 10
    'Expands n*(n+1)//2 lower triangular to full matrix\n\n    Scales off-diagonal by 1/sqrt(2), as per the SCS specification.\n\n    Parameters\n    ----------\n    lower_tri : numpy.ndarray\n        A NumPy array representing the lower triangular part of the\n        matrix, stacked in column-major order.\n    n : int\n        The number of rows (columns) in the full square matrix.\n\n    Returns\n    -------\n    numpy.ndarray\n        A 2-dimensional ndarray that is the scaled expansion of the lower\n        triangular array.\n\n    Notes\n    -----\n    SCS tracks "lower triangular" indices in a way that corresponds to numpy\'s\n    "upper triangular" indices. So the function call below uses ``np.triu_indices``\n    in a way that looks weird, but is nevertheless correct.\n    '
    full = np.zeros((n, n))
    full[np.triu_indices(n)] = lower_tri
    full += full.T
    full[np.diag_indices(n)] /= 2
    full[np.tril_indices(n, k=-1)] /= np.sqrt(2)
    full[np.triu_indices(n, k=1)] /= np.sqrt(2)
    return np.reshape(full, n * n, order='F')

def scs_psdvec_to_psdmat(vec: Expression, indices: np.ndarray) -> Expression:
    if False:
        i = 10
        return i + 15
    '\n    Return "V" so that "vec[indices] belongs to the SCS-standard PSD cone"\n    can be written in natural cvxpy syntax as "V >> 0".\n\n    Parameters\n    ----------\n    vec : cvxpy.expressions.expression.Expression\n        Must have ``vec.is_affine() == True``.\n    indices : ndarray\n        Contains nonnegative integers, which can index into ``vec``.\n\n    Notes\n    -----\n    This function is similar to ``tri_to_full``, which is also found\n    in this file. The difference is that this function works without\n    indexed assignment ``mat[i,j] = expr``. Such indexed assignment\n    cannot be used, because this function builds a cvxpy Expression,\n    rather than a numpy ndarray.\n    '
    n = int(np.sqrt(indices.size * 2))
    (rows, cols) = np.triu_indices(n)
    mats = []
    for (i, idx) in enumerate(indices):
        (r, c) = (rows[i], cols[i])
        mat = np.zeros(shape=(n, n))
        if r == c:
            mat[r, r] = 1
        else:
            mat[r, c] = 1 / np.sqrt(2)
            mat[c, r] = 1 / np.sqrt(2)
        mat = vec[idx] * mat
        mats.append(mat)
    V = sum(mats)
    return V

class SCS(ConicSolver):
    """An interface for the SCS solver.
    """
    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC, ExpCone, PSD, PowCone3D]
    REQUIRES_CONSTR = True
    STATUS_MAP = {1: s.OPTIMAL, 2: s.OPTIMAL_INACCURATE, -1: s.UNBOUNDED, -6: s.UNBOUNDED_INACCURATE, -2: s.INFEASIBLE, -7: s.INFEASIBLE_INACCURATE, -4: s.SOLVER_ERROR, -3: s.SOLVER_ERROR, -5: s.SOLVER_ERROR}
    EXP_CONE_ORDER = [0, 1, 2]
    ACCELERATION_RETRY_MESSAGE = '\n    CVXPY has just called the numerical solver SCS (version %s),\n    which could not accurately solve the problem with the provided solver\n    options. No value was specified for the SCS option called\n    "acceleration_lookback". That option often has a major impact on\n    whether this version of SCS converges to an accurate solution.\n\n    We will try to solve the problem again by setting acceleration_lookback = 0.\n    To avoid this error in the future we recommend installing SCS version 3.0\n    or higher.\n\n    More information on SCS options can be found at the following URL:\n    https://www.cvxgrp.org/scs/api/settings.html\n    '

    def name(self):
        if False:
            i = 10
            return i + 15
        'The name of the solver.\n        '
        return s.SCS

    def import_solver(self) -> None:
        if False:
            print('Hello World!')
        'Imports the solver.\n        '
        import scs

    def supports_quad_obj(self) -> bool:
        if False:
            print('Hello World!')
        'SCS >= 3.0.0 supports a quadratic objective.\n        '
        import scs
        return Version(scs.__version__) >= Version('3.0.0')

    @staticmethod
    def psd_format_mat(constr):
        if False:
            return 10
        'Return a linear operator to multiply by PSD constraint coefficients.\n\n        Special cases PSD constraints, as SCS expects constraints to be\n        imposed on solely the lower triangular part of the variable matrix.\n        Moreover, it requires the off-diagonal coefficients to be scaled by\n        sqrt(2), and applies to the symmetric part of the constrained expression.\n        '
        rows = cols = constr.expr.shape[0]
        entries = rows * (cols + 1) // 2
        row_arr = np.arange(0, entries)
        lower_diag_indices = np.tril_indices(rows)
        col_arr = np.sort(np.ravel_multi_index(lower_diag_indices, (rows, cols), order='F'))
        val_arr = np.zeros((rows, cols))
        val_arr[lower_diag_indices] = np.sqrt(2)
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

    def apply(self, problem):
        if False:
            return 10
        'Returns a new problem and data for inverting the new solution.\n\n        Returns\n        -------\n        tuple\n            (dict of arguments needed for the solver, inverse data)\n        '
        return super(SCS, self).apply(problem)

    @staticmethod
    def extract_dual_value(result_vec, offset, constraint):
        if False:
            return 10
        'Extracts the dual value for constraint starting at offset.\n\n        Special cases PSD constraints, as per the SCS specification.\n        '
        if isinstance(constraint, PSD):
            dim = constraint.shape[0]
            lower_tri_dim = dim * (dim + 1) // 2
            new_offset = offset + lower_tri_dim
            lower_tri = result_vec[offset:new_offset]
            full = tri_to_full(lower_tri, dim)
            return (full, new_offset)
        else:
            return utilities.extract_dual_value(result_vec, offset, constraint)

    def invert(self, solution, inverse_data):
        if False:
            print('Hello World!')
        'Returns the solution to the original problem given the inverse_data.\n        '
        import scs
        attr = {}
        if Version(scs.__version__) < Version('3.0.0'):
            status = self.STATUS_MAP[solution['info']['statusVal']]
            attr[s.SOLVE_TIME] = solution['info']['solveTime'] / 1000
            attr[s.SETUP_TIME] = solution['info']['setupTime'] / 1000
        else:
            status = self.STATUS_MAP[solution['info']['status_val']]
            attr[s.SOLVE_TIME] = solution['info']['solve_time'] / 1000
            attr[s.SETUP_TIME] = solution['info']['setup_time'] / 1000
        attr[s.NUM_ITERS] = solution['info']['iter']
        attr[s.EXTRA_STATS] = solution
        if status in s.SOLUTION_PRESENT:
            primal_val = solution['info']['pobj']
            opt_val = primal_val + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[SCS.VAR_ID]: solution['x']}
            eq_dual_vars = utilities.get_dual_values(solution['y'][:inverse_data[ConicSolver.DIMS].zero], self.extract_dual_value, inverse_data[SCS.EQ_CONSTR])
            ineq_dual_vars = utilities.get_dual_values(solution['y'][inverse_data[ConicSolver.DIMS].zero:], self.extract_dual_value, inverse_data[SCS.NEQ_CONSTR])
            dual_vars = {}
            dual_vars.update(eq_dual_vars)
            dual_vars.update(ineq_dual_vars)
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)

    @staticmethod
    def parse_solver_options(solver_opts):
        if False:
            i = 10
            return i + 15
        import scs
        if Version(scs.__version__) < Version('3.0.0'):
            if 'eps_abs' in solver_opts or 'eps_rel' in solver_opts:
                solver_opts['eps'] = min(solver_opts.get('eps_abs', 1), solver_opts.get('eps_rel', 1))
            else:
                solver_opts['eps'] = solver_opts.get('eps', 0.0001)
        elif 'eps' in solver_opts:
            solver_opts['eps_abs'] = solver_opts['eps']
            solver_opts['eps_rel'] = solver_opts['eps']
            del solver_opts['eps']
        else:
            solver_opts['eps_abs'] = solver_opts.get('eps_abs', 1e-05)
            solver_opts['eps_rel'] = solver_opts.get('eps_rel', 1e-05)
        if 'use_quad_obj' in solver_opts:
            del solver_opts['use_quad_obj']
        return solver_opts

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        if False:
            print('Hello World!')
        'Returns the result of the call to the solver.\n\n        Parameters\n        ----------\n        data : dict\n            Data generated via an apply call.\n        warm_start : Bool\n            Whether to warm_start SCS.\n        verbose : Bool\n            Control the verbosity.\n        solver_opts : dict\n            SCS-specific solver options.\n\n        Returns\n        -------\n        The result returned by a call to scs.solve().\n        '
        import scs
        scs_version = Version(scs.__version__)
        args = {'A': data[s.A], 'b': data[s.B], 'c': data[s.C]}
        if s.P in data:
            args['P'] = data[s.P]
        if warm_start and solver_cache is not None and (self.name() in solver_cache):
            args['x'] = solver_cache[self.name()]['x']
            args['y'] = solver_cache[self.name()]['y']
            args['s'] = solver_cache[self.name()]['s']
        cones = dims_to_solver_dict(data[ConicSolver.DIMS])

        def solve(_solver_opts):
            if False:
                return 10
            if scs_version.major < 3:
                _results = scs.solve(args, cones, verbose=verbose, **_solver_opts)
                _status = self.STATUS_MAP[_results['info']['statusVal']]
            else:
                _results = scs.solve(args, cones, verbose=verbose, **_solver_opts)
                _status = self.STATUS_MAP[_results['info']['status_val']]
            return (_results, _status)
        solver_opts = SCS.parse_solver_options(solver_opts)
        (results, status) = solve(solver_opts)
        if status in s.INACCURATE and scs_version.major == 2 and ('acceleration_lookback' not in solver_opts):
            import warnings
            warnings.warn(SCS.ACCELERATION_RETRY_MESSAGE % str(scs_version))
            retry_opts = solver_opts.copy()
            retry_opts['acceleration_lookback'] = 0
            (results, status) = solve(retry_opts)
        if solver_cache is not None and status == s.OPTIMAL:
            solver_cache[self.name()] = results
        return results