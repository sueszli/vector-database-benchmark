"""
Copyright, the CVXPY authors

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
import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS, QP_SOLVERS
from cvxpy.tests.base_test import BaseTest

class TestParamQuadProg(BaseTest):

    def setUp(self) -> None:
        if False:
            return 10
        self.solvers = [x for x in QP_SOLVERS if x in INSTALLED_SOLVERS]

    def assertItemsAlmostEqual(self, a, b, places: int=2) -> None:
        if False:
            print('Hello World!')
        super(TestParamQuadProg, self).assertItemsAlmostEqual(a, b, places=places)

    def assertAlmostEqual(self, a, b, places: int=2) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(TestParamQuadProg, self).assertAlmostEqual(a, b, places=places)

    def test_param_data(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        for solver in self.solvers:
            np.random.seed(0)
            m = 30
            n = 20
            A = np.random.randn(m, n)
            b = np.random.randn(m)
            x = cp.Variable(n)
            gamma = cp.Parameter(nonneg=True)
            gamma_val = 0.5
            gamma_val_new = 0.1
            objective = cp.Minimize(gamma * cp.sum_squares(A @ x - b) + cp.norm(x, 1))
            constraints = [1 <= x, x <= 2]
            prob = cp.Problem(objective, constraints)
            self.assertTrue(prob.is_dpp())
            gamma.value = gamma_val_new
            (data_scratch, _, _) = prob.get_problem_data(solver)
            prob.solve(solver=solver)
            x_scratch = np.copy(x.value)
            prob = cp.Problem(objective, constraints)
            gamma.value = gamma_val
            (data_param, _, _) = prob.get_problem_data(solver)
            prob.solve(solver=solver)
            gamma.value = gamma_val_new
            (data_param_new, _, _) = prob.get_problem_data(solver)
            prob.solve(solver=solver)
            x_gamma_new = np.copy(x.value)
            np.testing.assert_allclose(data_param_new['P'].todense(), data_scratch['P'].todense())
            np.testing.assert_allclose(x_gamma_new, x_scratch, rtol=0.01, atol=0.01)

    def test_qp_problem(self) -> None:
        if False:
            print('Hello World!')
        for solver in self.solvers:
            m = 30
            n = 20
            A = np.random.randn(m, n)
            b = np.random.randn(m)
            x = cp.Variable(n)
            gamma = cp.Parameter(nonneg=True)
            gamma.value = 0.5
            objective = cp.Minimize(cp.sum_squares(A @ x - b) + gamma * cp.norm(x, 1))
            constraints = [0 <= x, x <= 1]
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=solver)
            x_full = np.copy(x.value)
            solving_chain = problem._cache.solving_chain
            solver = problem._cache.solving_chain.solver
            inverse_data = problem._cache.inverse_data
            param_prog = problem._cache.param_prog
            (data, solver_inverse_data) = solving_chain.solver.apply(param_prog)
            inverse_data = inverse_data + [solver_inverse_data]
            raw_solution = solver.solve_via_data(data, warm_start=False, verbose=False, solver_opts={})
            problem.unpack_results(raw_solution, solving_chain, inverse_data)
            x_param = np.copy(x.value)
            np.testing.assert_allclose(x_param, x_full, rtol=0.01, atol=0.01)