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
from scipy.linalg import lstsq
import cvxpy as cp
from cvxpy import Maximize, Minimize, Parameter, Problem
from cvxpy.atoms import QuadForm, abs, huber, matrix_frac, norm, power, quad_over_lin, sum, sum_squares
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS, QP_SOLVERS
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import StandardTestLPs

class TestQp(BaseTest):
    """ Unit tests for the domain module. """

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        self.a = Variable(name='a')
        self.b = Variable(name='b')
        self.c = Variable(name='c')
        self.x = Variable(2, name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(2, name='z')
        self.w = Variable(5, name='w')
        self.A = Variable((2, 2), name='A')
        self.B = Variable((2, 2), name='B')
        self.C = Variable((3, 2), name='C')
        self.slope = Variable(1, name='slope')
        self.offset = Variable(1, name='offset')
        self.quadratic_coeff = Variable(1, name='quadratic_coeff')
        T = 30
        self.position = Variable((2, T), name='position')
        self.velocity = Variable((2, T), name='velocity')
        self.force = Variable((2, T - 1), name='force')
        self.xs = Variable(80, name='xs')
        self.xsr = Variable(50, name='xsr')
        self.xef = Variable(80, name='xef')
        self.solvers = [x for x in QP_SOLVERS if x in INSTALLED_SOLVERS]
        if 'MOSEK' in INSTALLED_SOLVERS:
            self.solvers.append('MOSEK')

    def solve_QP(self, problem, solver_name):
        if False:
            print('Hello World!')
        return problem.solve(solver=solver_name, verbose=False)

    def test_all_solvers(self) -> None:
        if False:
            print('Hello World!')
        for solver in self.solvers:
            self.quad_over_lin(solver)
            self.power(solver)
            self.power_matrix(solver)
            self.square_affine(solver)
            self.quad_form(solver)
            self.affine_problem(solver)
            self.maximize_problem(solver)
            self.abs(solver)
            self.quad_form_coeff(solver)
            self.quad_form_bound(solver)
            self.regression_1(solver)
            self.regression_2(solver)
            self.rep_quad_form(solver)
            self.control(solver)
            self.sparse_system(solver)
            self.smooth_ridge(solver)
            self.huber_small(solver)
            self.huber(solver)
            self.equivalent_forms_1(solver)
            self.equivalent_forms_2(solver)
            self.equivalent_forms_3(solver)

    def quad_over_lin(self, solver) -> None:
        if False:
            return 10
        p = Problem(Minimize(0.5 * quad_over_lin(abs(self.x - 1), 1)), [self.x <= -1])
        self.solve_QP(p, solver)
        for var in p.variables():
            self.assertItemsAlmostEqual(np.array([-1.0, -1.0]), var.value, places=4)
        for con in p.constraints:
            self.assertItemsAlmostEqual(np.array([2.0, 2.0]), con.dual_value, places=4)

    def abs(self, solver) -> None:
        if False:
            return 10
        u = Variable(2)
        constr = []
        constr += [abs(u[1] - u[0]) <= 100]
        prob = Problem(Minimize(sum_squares(u)), constr)
        print('The problem is QP: ', prob.is_qp())
        self.assertEqual(prob.is_qp(), True)
        result = prob.solve(solver=solver)
        self.assertAlmostEqual(result, 0)

    def power(self, solver) -> None:
        if False:
            print('Hello World!')
        p = Problem(Minimize(sum(power(self.x, 2))), [])
        self.solve_QP(p, solver)
        for var in p.variables():
            self.assertItemsAlmostEqual([0.0, 0.0], var.value, places=4)

    def power_matrix(self, solver) -> None:
        if False:
            return 10
        p = Problem(Minimize(sum(power(self.A - 3.0, 2))), [])
        self.solve_QP(p, solver)
        for var in p.variables():
            self.assertItemsAlmostEqual([3.0, 3.0, 3.0, 3.0], var.value, places=4)

    def square_affine(self, solver) -> None:
        if False:
            print('Hello World!')
        A = np.random.randn(10, 2)
        b = np.random.randn(10)
        p = Problem(Minimize(sum_squares(A @ self.x - b)))
        self.solve_QP(p, solver)
        for var in p.variables():
            self.assertItemsAlmostEqual(lstsq(A, b)[0].flatten(), var.value, places=1)

    def quad_form(self, solver) -> None:
        if False:
            while True:
                i = 10
        np.random.seed(0)
        A = np.random.randn(5, 5)
        z = np.random.randn(5)
        P = A.T.dot(A)
        q = -2 * P.dot(z)
        p = Problem(Minimize(QuadForm(self.w, P) + q.T @ self.w))
        self.solve_QP(p, solver)
        for var in p.variables():
            self.assertItemsAlmostEqual(z, var.value, places=4)

    def rep_quad_form(self, solver) -> None:
        if False:
            for i in range(10):
                print('nop')
        'A problem where the quad_form term is used multiple times.\n        '
        np.random.seed(0)
        A = np.random.randn(5, 5)
        z = np.random.randn(5)
        P = A.T.dot(A)
        q = -2 * P.dot(z)
        qf = QuadForm(self.w, P)
        p = Problem(Minimize(0.5 * qf + 0.5 * qf + q.T @ self.w))
        self.solve_QP(p, solver)
        for var in p.variables():
            self.assertItemsAlmostEqual(z, var.value, places=4)

    def affine_problem(self, solver) -> None:
        if False:
            while True:
                i = 10
        A = np.random.randn(5, 2)
        A = np.maximum(A, 0)
        b = np.random.randn(5)
        b = np.maximum(b, 0)
        p = Problem(Minimize(sum(self.x)), [self.x >= 0, A @ self.x <= b])
        self.solve_QP(p, solver)
        for var in p.variables():
            self.assertItemsAlmostEqual([0.0, 0.0], var.value, places=3)

    def maximize_problem(self, solver) -> None:
        if False:
            for i in range(10):
                print('nop')
        A = np.random.randn(5, 2)
        A = np.maximum(A, 0)
        b = np.random.randn(5)
        b = np.maximum(b, 0)
        p = Problem(Maximize(-sum(self.x)), [self.x >= 0, A @ self.x <= b])
        self.solve_QP(p, solver)
        for var in p.variables():
            self.assertItemsAlmostEqual([0.0, 0.0], var.value, places=3)

    def norm_2(self, solver) -> None:
        if False:
            return 10
        A = np.random.randn(10, 5)
        b = np.random.randn(10)
        p = Problem(Minimize(norm(A @ self.w - b, 2)))
        self.solve_QP(p, solver)
        for var in p.variables():
            self.assertItemsAlmostEqual(lstsq(A, b)[0].flatten(), var.value, places=1)

    def mat_norm_2(self, solver) -> None:
        if False:
            while True:
                i = 10
        A = np.random.randn(5, 3)
        B = np.random.randn(5, 2)
        p = Problem(Minimize(norm(A @ self.C - B, 2)))
        s = self.solve_QP(p, solver)
        for var in p.variables():
            self.assertItemsAlmostEqual(lstsq(A, B)[0], s.primal_vars[var.id], places=1)

    def quad_form_coeff(self, solver) -> None:
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(0)
        A = np.random.randn(5, 5)
        z = np.random.randn(5)
        P = A.T.dot(A)
        q = -2 * P.dot(z)
        p = Problem(Minimize(QuadForm(self.w, P) + q.T @ self.w))
        self.solve_QP(p, solver)
        for var in p.variables():
            self.assertItemsAlmostEqual(z, var.value, places=4)

    def quad_form_bound(self, solver) -> None:
        if False:
            for i in range(10):
                print('nop')
        P = np.array([[13, 12, -2], [12, 17, 6], [-2, 6, 12]])
        q = np.array([[-22], [-14.5], [13]])
        r = 1
        y_star = np.array([[1], [0.5], [-1]])
        p = Problem(Minimize(0.5 * QuadForm(self.y, P) + q.T @ self.y + r), [self.y >= -1, self.y <= 1])
        self.solve_QP(p, solver)
        for var in p.variables():
            self.assertItemsAlmostEqual(y_star, var.value, places=4)

    def regression_1(self, solver) -> None:
        if False:
            print('Hello World!')
        np.random.seed(1)
        n = 100
        true_coeffs = np.array([[2, -2, 0.5]]).T
        x_data = np.random.rand(n) * 5
        x_data = np.atleast_2d(x_data)
        x_data_expanded = np.vstack([np.power(x_data, i) for i in range(1, 4)])
        x_data_expanded = np.atleast_2d(x_data_expanded)
        y_data = x_data_expanded.T.dot(true_coeffs) + 0.5 * np.random.rand(n, 1)
        y_data = np.atleast_2d(y_data)
        line = self.offset + x_data * self.slope
        residuals = line.T - y_data
        fit_error = sum_squares(residuals)
        p = Problem(Minimize(fit_error), [])
        self.solve_QP(p, solver)
        self.assertAlmostEqual(1171.60037715, p.value, places=4)

    def regression_2(self, solver) -> None:
        if False:
            print('Hello World!')
        np.random.seed(1)
        n = 100
        true_coeffs = np.array([2, -2, 0.5])
        x_data = np.random.rand(n) * 5
        x_data_expanded = np.vstack([np.power(x_data, i) for i in range(1, 4)])
        print(x_data_expanded.shape, true_coeffs.shape)
        y_data = x_data_expanded.T.dot(true_coeffs) + 0.5 * np.random.rand(n)
        quadratic = self.offset + x_data * self.slope + self.quadratic_coeff * np.power(x_data, 2)
        residuals = quadratic.T - y_data
        fit_error = sum_squares(residuals)
        p = Problem(Minimize(fit_error), [])
        self.solve_QP(p, solver)
        self.assertAlmostEqual(139.225660756, p.value, places=4)

    def control(self, solver) -> None:
        if False:
            i = 10
            return i + 15
        initial_velocity = np.array([-20, 100])
        final_position = np.array([100, 100])
        T = 30
        h = 0.1
        mass = 1
        drag = 0.1
        g = np.array([0, -9.8])
        constraints = []
        for i in range(T - 1):
            constraints += [self.position[:, i + 1] == self.position[:, i] + h * self.velocity[:, i]]
            acceleration = self.force[:, i] / mass + g - drag * self.velocity[:, i]
            constraints += [self.velocity[:, i + 1] == self.velocity[:, i] + h * acceleration]
        constraints += [self.position[:, 0] == 0]
        constraints += [self.position[:, -1] == final_position]
        constraints += [self.velocity[:, 0] == initial_velocity]
        constraints += [self.velocity[:, -1] == 0]
        p = Problem(Minimize(0.01 * sum_squares(self.force)), constraints)
        self.solve_QP(p, solver)
        self.assertAlmostEqual(1059.616, p.value, places=1)

    def sparse_system(self, solver) -> None:
        if False:
            for i in range(10):
                print('nop')
        m = 100
        n = 80
        np.random.seed(1)
        density = 0.4
        A = sp.rand(m, n, density)
        b = np.random.randn(m)
        p = Problem(Minimize(sum_squares(A @ self.xs - b)), [self.xs == 0])
        self.solve_QP(p, solver)
        self.assertAlmostEqual(b.T.dot(b), p.value, places=4)

    def smooth_ridge(self, solver) -> None:
        if False:
            i = 10
            return i + 15
        np.random.seed(1)
        n = 50
        k = 20
        eta = 1
        A = np.ones((k, n))
        b = np.ones(k)
        obj = sum_squares(A @ self.xsr - b) + eta * sum_squares(self.xsr[:-1] - self.xsr[1:])
        p = Problem(Minimize(obj), [])
        self.solve_QP(p, solver)
        self.assertAlmostEqual(0, p.value, places=4)

    def huber_small(self, solver) -> None:
        if False:
            print('Hello World!')
        x = Variable(3)
        objective = sum(huber(x))
        p = Problem(Minimize(objective), [x[2] >= 3])
        self.solve_QP(p, solver)
        self.assertAlmostEqual(3, x.value[2], places=4)
        self.assertAlmostEqual(5, objective.value, places=4)

    def huber(self, solver) -> None:
        if False:
            return 10
        np.random.seed(2)
        n = 3
        m = 5
        A = sp.random(m, n, density=0.8, format='csc')
        x_true = np.random.randn(n) / np.sqrt(n)
        ind95 = (np.random.rand(m) < 0.95).astype(float)
        b = A.dot(x_true) + np.multiply(0.5 * np.random.randn(m), ind95) + np.multiply(10.0 * np.random.rand(m), 1.0 - ind95)
        x = Variable(n)
        objective = sum(huber(A @ x - b))
        p = Problem(Minimize(objective))
        self.solve_QP(p, solver)
        self.assertAlmostEqual(1.327429461061672, objective.value, places=3)
        self.assertItemsAlmostEqual(x.value, [-1.03751745, 0.86657204, -0.9649172], places=3)

    def equivalent_forms_1(self, solver) -> None:
        if False:
            return 10
        m = 100
        n = 80
        r = 70
        np.random.seed(1)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        G = np.random.randn(r, n)
        h = np.random.randn(r)
        obj1 = 0.1 * sum((A @ self.xef - b) ** 2)
        cons = [G @ self.xef == h]
        p1 = Problem(Minimize(obj1), cons)
        self.solve_QP(p1, solver)
        self.assertAlmostEqual(p1.value, 68.1119420108, places=4)

    def equivalent_forms_2(self, solver) -> None:
        if False:
            i = 10
            return i + 15
        m = 100
        n = 80
        r = 70
        np.random.seed(1)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        G = np.random.randn(r, n)
        h = np.random.randn(r)
        P = np.dot(A.T, A)
        q = -2 * np.dot(A.T, b)
        r = np.dot(b.T, b)
        obj2 = 0.1 * (QuadForm(self.xef, P) + q.T @ self.xef + r)
        cons = [G @ self.xef == h]
        p2 = Problem(Minimize(obj2), cons)
        self.solve_QP(p2, solver)
        self.assertAlmostEqual(p2.value, 68.1119420108, places=4)

    def equivalent_forms_3(self, solver) -> None:
        if False:
            i = 10
            return i + 15
        m = 100
        n = 80
        r = 70
        np.random.seed(1)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        G = np.random.randn(r, n)
        h = np.random.randn(r)
        P = np.dot(A.T, A)
        q = -2 * np.dot(A.T, b)
        r = np.dot(b.T, b)
        Pinv = np.linalg.inv(P)
        obj3 = 0.1 * (matrix_frac(self.xef, Pinv) + q.T @ self.xef + r)
        cons = [G @ self.xef == h]
        p3 = Problem(Minimize(obj3), cons)
        self.solve_QP(p3, solver)
        self.assertAlmostEqual(p3.value, 68.1119420108, places=4)

    def test_warm_start(self) -> None:
        if False:
            while True:
                i = 10
        'Test warm start.\n        '
        m = 200
        n = 100
        np.random.seed(1)
        A = np.random.randn(m, n)
        b = Parameter(m)
        x = Variable(n)
        prob = Problem(Minimize(sum_squares(A @ x - b)))
        b.value = np.random.randn(m)
        result = prob.solve(solver='OSQP', warm_start=False)
        result2 = prob.solve(solver='OSQP', warm_start=True)
        self.assertAlmostEqual(result, result2)
        b.value = np.random.randn(m)
        result = prob.solve(solver='OSQP', warm_start=True)
        result2 = prob.solve(solver='OSQP', warm_start=False)
        self.assertAlmostEqual(result, result2)

    def test_gurobi_warmstart(self) -> None:
        if False:
            while True:
                i = 10
        'Test Gurobi warm start with a user provided point.\n        '
        if cp.GUROBI in INSTALLED_SOLVERS:
            import gurobipy
            m = 4
            n = 3
            y = Variable(nonneg=True)
            X = Variable((m, n))
            X_vals = np.reshape(np.arange(m * n), (m, n))
            prob = Problem(Minimize(y ** 2 + cp.sum(X)), [X == X_vals])
            X.value = X_vals + 1
            prob.solve(solver=cp.GUROBI, warm_start=True)
            model = prob.solver_stats.extra_stats
            model_x = model.getVars()
            assert gurobipy.GRB.UNDEFINED == model_x[0].start
            assert np.isclose(0, model_x[0].x)
            for i in range(1, X.size + 1):
                row = (i - 1) % X.shape[0]
                col = (i - 1) // X.shape[0]
                assert X_vals[row, col] + 1 == model_x[i].start
                assert np.isclose(X.value[row, col], model_x[i].x)

    def test_parametric(self) -> None:
        if False:
            return 10
        'Test solve parametric problem vs full problem'
        x = Variable()
        a = 10
        b_vec = [-10, -2.0]
        for solver in self.solvers:
            print(solver)
            x_full = []
            obj_full = []
            for b in b_vec:
                obj = Minimize(a * x ** 2 + b * x)
                constraints = [0 <= x, x <= 1]
                prob = Problem(obj, constraints)
                prob.solve(solver=solver)
                x_full += [x.value]
                obj_full += [prob.value]
            x_param = []
            obj_param = []
            b = Parameter()
            obj = Minimize(a * x ** 2 + b * x)
            constraints = [0 <= x, x <= 1]
            prob = Problem(obj, constraints)
            for b_value in b_vec:
                b.value = b_value
                prob.solve(solver=solver)
                x_param += [x.value]
                obj_param += [prob.value]
            print(x_full)
            print(x_param)
            for i in range(len(b_vec)):
                self.assertItemsAlmostEqual(x_full[i], x_param[i], places=3)
                self.assertAlmostEqual(obj_full[i], obj_param[i])

    def test_square_param(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test issue arising with square plus parameter.\n        '
        a = Parameter(value=1)
        b = Variable()
        obj = Minimize(b ** 2 + abs(a))
        prob = Problem(obj)
        prob.solve(solver='SCS')
        self.assertAlmostEqual(obj.value, 1.0)

    def test_gurobi_time_limit_no_solution(self) -> None:
        if False:
            i = 10
            return i + 15
        "Make sure that if Gurobi terminates due to a time limit before finding a solution:\n            1) no error is raised,\n            2) solver stats are returned.\n            The test is skipped if something changes on Gurobi's side so that:\n            - a solution is found despite a time limit of zero,\n            - a different termination criteria is hit first.\n        "
        from cvxpy import GUROBI
        if GUROBI in INSTALLED_SOLVERS:
            import gurobipy
            objective = Minimize(self.x[0])
            constraints = [self.x[0] >= 1]
            prob = Problem(objective, constraints)
            try:
                prob.solve(solver=GUROBI, TimeLimit=0.0)
            except Exception as e:
                self.fail('An exception %s is raised instead of returning a result.' % e)
            extra_stats = None
            solver_stats = getattr(prob, 'solver_stats', None)
            if solver_stats:
                extra_stats = getattr(solver_stats, 'extra_stats', None)
            self.assertTrue(extra_stats, 'Solver stats have not been returned.')
            nb_solutions = getattr(extra_stats, 'SolCount', None)
            if nb_solutions:
                self.skipTest('Gurobi has found a solution, the test is not relevant anymore.')
            solver_status = getattr(extra_stats, 'Status', None)
            if solver_status != gurobipy.StatusConstClass.TIME_LIMIT:
                self.skipTest('Gurobi terminated for a different reason than reaching time limit, the test is not relevant anymore.')
        else:
            with self.assertRaises(Exception) as cm:
                prob = Problem(Minimize(norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=GUROBI, TimeLimit=0)
            self.assertEqual(str(cm.exception), 'The solver %s is not installed.' % GUROBI)

    def test_gurobi_environment(self) -> None:
        if False:
            print('Hello World!')
        'Tests that Gurobi environments can be passed to Model.\n        Gurobi environments can include licensing and model parameter data.\n        '
        from cvxpy import GUROBI
        if GUROBI in INSTALLED_SOLVERS:
            import gurobipy
            params = {'MIPGap': np.random.random(), 'AggFill': np.random.randint(10), 'PerturbValue': np.random.random()}
            custom_env = gurobipy.Env()
            for (k, v) in params.items():
                custom_env.setParam(k, v)
            sth = StandardTestLPs.test_lp_0(solver='GUROBI', env=custom_env)
            model = sth.prob.solver_stats.extra_stats
            for (k, v) in params.items():
                (name, p_type, p_val, p_min, p_max, p_def) = model.getParamInfo(k)
                self.assertEqual(v, p_val)
        else:
            with self.assertRaises(Exception) as cm:
                prob = Problem(Minimize(norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=GUROBI, TimeLimit=0)
            self.assertEqual(str(cm.exception), 'The solver %s is not installed.' % GUROBI)