"""
Copyright 2017 Steven Diamond

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
import builtins
import pickle
import sys
import warnings
from fractions import Fraction
from io import StringIO
import ecos
import numpy
import numpy as np
import scipy.sparse as sp
import scs
from numpy import linalg as LA
import cvxpy as cp
import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.constraints import PSD, ExpCone, NonNeg, Zero
from cvxpy.error import DCPError, ParameterError, SolverError
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.problems.problem import Problem
from cvxpy.reductions.solvers.conic_solvers import ecos_conif, scs_conif
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS, SOLVER_MAP_CONIC
from cvxpy.reductions.solvers.solving_chain import ECOS_DEPRECATION_MSG
from cvxpy.tests.base_test import BaseTest

class TestProblem(BaseTest):
    """Unit tests for the expression/expression module.
    """

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        self.a = Variable(name='a')
        self.b = Variable(name='b')
        self.c = Variable(name='c')
        self.x = Variable(2, name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(2, name='z')
        self.A = Variable((2, 2), name='A')
        self.B = Variable((2, 2), name='B')
        self.C = Variable((3, 2), name='C')

    def test_to_str(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test string representations.\n        '
        obj = cp.Minimize(self.a)
        prob = Problem(obj)
        self.assertEqual(repr(prob), 'Problem(%s, %s)' % (repr(obj), repr([])))
        constraints = [self.x * 2 == self.x, self.x == 0]
        prob = Problem(obj, constraints)
        self.assertEqual(repr(prob), 'Problem(%s, %s)' % (repr(obj), repr(constraints)))
        result = 'minimize %(name)s\nsubject to %(name)s == 0\n           %(name)s >= 0' % {'name': self.a.name()}
        prob = Problem(cp.Minimize(self.a), [Zero(self.a), NonNeg(self.a)])
        self.assertEqual(str(prob), result)

    def test_variables(self) -> None:
        if False:
            print('Hello World!')
        'Test the variables method.\n        '
        p = Problem(cp.Minimize(self.a), [self.a <= self.x, self.b <= self.A + 2])
        vars_ = p.variables()
        ref = [self.a, self.x, self.b, self.A]
        self.assertCountEqual(vars_, ref)

    def test_var_dict(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        p = Problem(cp.Minimize(self.a), [self.a <= self.x, self.b <= self.A + 2])
        assert p.var_dict == {'a': self.a, 'x': self.x, 'b': self.b, 'A': self.A}

    def test_parameters(self) -> None:
        if False:
            while True:
                i = 10
        'Test the parameters method.\n        '
        p1 = Parameter()
        p2 = Parameter(3, nonpos=True)
        p3 = Parameter((4, 4), nonneg=True)
        p = Problem(cp.Minimize(p1), [self.a + p1 <= p2, self.b <= p3 + p3 + 2])
        params = p.parameters()
        ref = [p1, p2, p3]
        self.assertCountEqual(params, ref)

    def test_param_dict(self) -> None:
        if False:
            print('Hello World!')
        p1 = Parameter(name='p1')
        p2 = Parameter(3, nonpos=True, name='p2')
        p3 = Parameter((4, 4), nonneg=True, name='p3')
        p = Problem(cp.Minimize(p1), [self.a + p1 <= p2, self.b <= p3 + p3 + 2])
        assert p.param_dict == {'p1': p1, 'p2': p2, 'p3': p3}

    def test_solving_a_problem_with_unspecified_parameters(self) -> None:
        if False:
            print('Hello World!')
        param = cp.Parameter(name='lambda')
        problem = cp.Problem(cp.Minimize(param), [])
        with self.assertRaises(ParameterError, msg="A Parameter (whose name is 'lambda').*"):
            problem.solve(solver=cp.SCS)

    def test_constants(self) -> None:
        if False:
            print('Hello World!')
        'Test the constants method.\n        '
        c1 = numpy.random.randn(1, 2)
        c2 = numpy.random.randn(2)
        p = Problem(cp.Minimize(c1 @ self.x), [self.x >= c2])
        constants_ = p.constants()
        ref = [c1, c2]
        self.assertEqual(len(ref), len(constants_))
        for (c, r) in zip(constants_, ref):
            self.assertTupleEqual(c.shape, r.shape)
            self.assertTrue((c.value == r).all())
        p = Problem(cp.Minimize(self.a), [self.x >= 1])
        constants_ = p.constants()
        ref = [numpy.array(1)]
        self.assertEqual(len(ref), len(constants_))
        for (c, r) in zip(constants_, ref):
            self.assertEqual(c.shape, r.shape) and self.assertTrue((c.value == r).all())

    def test_size_metrics(self) -> None:
        if False:
            return 10
        'Test the size_metrics method.\n        '
        p1 = Parameter()
        p2 = Parameter(3, nonpos=True)
        p3 = Parameter((4, 4), nonneg=True)
        c1 = numpy.random.randn(2, 1)
        c2 = numpy.random.randn(1, 2)
        constants = [2, c2.dot(c1)]
        p = Problem(cp.Minimize(p1), [self.a + p1 <= p2, self.b <= p3 + p3 + constants[0], self.c == constants[1]])
        n_variables = p.size_metrics.num_scalar_variables
        ref = self.a.size + self.b.size + self.c.size
        self.assertEqual(n_variables, ref)
        n_data = p.size_metrics.num_scalar_data
        ref = numpy.prod(p1.size) + numpy.prod(p2.size) + numpy.prod(p3.size) + len(constants)
        self.assertEqual(n_data, ref)
        n_eq_constr = p.size_metrics.num_scalar_eq_constr
        ref = c2.dot(c1).size
        self.assertEqual(n_eq_constr, ref)
        n_leq_constr = p.size_metrics.num_scalar_leq_constr
        ref = numpy.prod(p3.size) + numpy.prod(p2.size)
        self.assertEqual(n_leq_constr, ref)
        max_data_dim = p.size_metrics.max_data_dimension
        ref = max(p3.shape)
        self.assertEqual(max_data_dim, ref)

    def test_solver_stats(self) -> None:
        if False:
            return 10
        'Test the solver_stats method.\n        '
        prob = Problem(cp.Minimize(cp.norm(self.x)), [self.x == 0])
        prob.solve(solver=s.ECOS)
        stats = prob.solver_stats
        self.assertGreater(stats.solve_time, 0)
        self.assertGreater(stats.setup_time, 0)
        self.assertGreater(stats.num_iters, 0)
        self.assertIn('info', stats.extra_stats)
        prob = Problem(cp.Minimize(cp.norm(self.x)), [self.x == 0])
        prob.solve(solver=s.SCS)
        stats = prob.solver_stats
        self.assertGreater(stats.solve_time, 0)
        self.assertGreater(stats.setup_time, 0)
        self.assertGreater(stats.num_iters, 0)
        self.assertIn('info', stats.extra_stats)
        prob = Problem(cp.Minimize(cp.sum(self.x)), [self.x == 0])
        prob.solve(solver=s.OSQP)
        stats = prob.solver_stats
        self.assertGreater(stats.solve_time, 0)
        self.assertGreater(stats.num_iters, 0)
        self.assertTrue(hasattr(stats.extra_stats, 'info'))
        self.assertTrue(str(stats).startswith('SolverStats(solver_name='))

    def test_compilation_time(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        prob = Problem(cp.Minimize(cp.norm(self.x)), [self.x == 0])
        prob.solve()
        assert isinstance(prob.compilation_time, float)

    def test_get_problem_data(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test get_problem_data method.\n        '
        (data, _, _) = Problem(cp.Minimize(cp.exp(self.a) + 2)).get_problem_data(s.SCS)
        dims = data[ConicSolver.DIMS]
        self.assertEqual(dims.exp, 1)
        self.assertEqual(data['c'].shape, (2,))
        self.assertEqual(data['A'].shape, (3, 2))
        (data, _, _) = Problem(cp.Minimize(cp.norm(self.x) + 3)).get_problem_data(s.ECOS)
        dims = data[ConicSolver.DIMS]
        self.assertEqual(dims.soc, [3])
        self.assertEqual(data['c'].shape, (3,))
        self.assertIsNone(data['A'])
        self.assertEqual(data['G'].shape, (3, 3))
        p = Problem(cp.Minimize(cp.sum_squares(self.x) + 2))
        (data, _, _) = p.get_problem_data(s.SCS, solver_opts={'use_quad_obj': False})
        dims = data[ConicSolver.DIMS]
        self.assertEqual(dims.soc, [4])
        self.assertEqual(data['c'].shape, (3,))
        self.assertEqual(data['A'].shape, (4, 3))
        (data, _, _) = p.get_problem_data(s.SCS, solver_opts={'use_quad_obj': True})
        dims = data[ConicSolver.DIMS]
        self.assertEqual(dims.soc, [])
        self.assertEqual(data['P'].shape, (2, 2))
        self.assertEqual(data['c'].shape, (2,))
        self.assertEqual(data['A'].shape, (0, 2))
        if s.CVXOPT in INSTALLED_SOLVERS:
            (data, _, _) = Problem(cp.Minimize(cp.norm(self.x) + 3)).get_problem_data(s.CVXOPT)
            dims = data[ConicSolver.DIMS]
            self.assertEqual(dims.soc, [3])

    def test_unpack_results(self) -> None:
        if False:
            while True:
                i = 10
        'Test unpack results method.\n        '
        prob = Problem(cp.Minimize(cp.exp(self.a)), [self.a == 0])
        (args, chain, inv) = prob.get_problem_data(s.SCS)
        data = {'c': args['c'], 'A': args['A'], 'b': args['b']}
        cones = scs_conif.dims_to_solver_dict(args[ConicSolver.DIMS])
        solution = scs.solve(data, cones)
        prob = Problem(cp.Minimize(cp.exp(self.a)), [self.a == 0])
        prob.unpack_results(solution, chain, inv)
        self.assertAlmostEqual(self.a.value, 0, places=3)
        self.assertAlmostEqual(prob.value, 1, places=3)
        self.assertAlmostEqual(prob.status, s.OPTIMAL)
        prob = Problem(cp.Minimize(cp.norm(self.x)), [self.x == 0])
        (args, chain, inv) = prob.get_problem_data(s.ECOS)
        cones = ecos_conif.dims_to_solver_dict(args[ConicSolver.DIMS])
        solution = ecos.solve(args['c'], args['G'], args['h'], cones, args['A'], args['b'])
        prob = Problem(cp.Minimize(cp.norm(self.x)), [self.x == 0])
        prob.unpack_results(solution, chain, inv)
        self.assertItemsAlmostEqual(self.x.value, [0, 0])
        self.assertAlmostEqual(prob.value, 0)
        self.assertAlmostEqual(prob.status, s.OPTIMAL)

    def test_verbose(self) -> None:
        if False:
            return 10
        'Test silencing and enabling solver messages.\n        '
        outputs = {True: [], False: []}
        backup = sys.stdout
        for solver in INSTALLED_SOLVERS:
            for verbose in [True, False]:
                if solver in [cp.GLPK, cp.GLPK_MI, cp.MOSEK, cp.CBC, cp.SCIPY, cp.COPT]:
                    continue
                sys.stdout = StringIO()
                p = Problem(cp.Minimize(self.a + self.x[0]), [self.a >= 2, self.x >= 2])
                p.solve(verbose=verbose, solver=solver)
                if solver in SOLVER_MAP_CONIC:
                    if SOLVER_MAP_CONIC[solver].MIP_CAPABLE:
                        p.constraints.append(Variable(boolean=True) == 0)
                        p.solve(verbose=verbose, solver=solver)
                    if ExpCone in SOLVER_MAP_CONIC[solver].SUPPORTED_CONSTRAINTS:
                        p = Problem(cp.Minimize(self.a), [cp.log(self.a) >= 2])
                        p.solve(verbose=verbose, solver=solver)
                    if PSD in SOLVER_MAP_CONIC[solver].SUPPORTED_CONSTRAINTS:
                        a_mat = cp.reshape(self.a, shape=(1, 1))
                        p = Problem(cp.Minimize(self.a), [cp.lambda_min(a_mat) >= 2])
                        p.solve(verbose=verbose, solver=solver)
                out = sys.stdout.getvalue()
                sys.stdout.close()
                sys.stdout = backup
                outputs[verbose].append((out, solver))
        for (output, solver) in outputs[True]:
            print(solver)
            assert len(output) > 0
        for (output, solver) in outputs[False]:
            print(solver)
            assert len(output) == 0

    def test_register_solve(self) -> None:
        if False:
            return 10
        Problem.register_solve('test', lambda self: 1)
        p = Problem(cp.Minimize(1))
        result = p.solve(method='test')
        self.assertEqual(result, 1)

        def test(self, a, b: int=2):
            if False:
                for i in range(10):
                    print('nop')
            return (a, b)
        Problem.register_solve('test', test)
        p = Problem(cp.Minimize(0))
        result = p.solve(1, b=3, method='test')
        self.assertEqual(result, (1, 3))
        result = p.solve(1, method='test')
        self.assertEqual(result, (1, 2))
        result = p.solve(1, method='test', b=4)
        self.assertEqual(result, (1, 4))

    def test_is_dcp(self) -> None:
        if False:
            i = 10
            return i + 15
        p = Problem(cp.Minimize(cp.norm_inf(self.a)))
        self.assertEqual(p.is_dcp(), True)
        p = Problem(cp.Maximize(cp.norm_inf(self.a)))
        self.assertEqual(p.is_dcp(), False)
        with self.assertRaises(DCPError):
            p.solve(solver=cp.SCS)

    def test_is_qp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        A = numpy.random.randn(4, 3)
        b = numpy.random.randn(4)
        Aeq = numpy.random.randn(2, 3)
        beq = numpy.random.randn(2)
        F = numpy.random.randn(2, 3)
        g = numpy.random.randn(2)
        obj = cp.sum_squares(A @ self.y - b)
        qpwa_obj = 3 * cp.sum_squares(-cp.abs(A @ self.y)) + cp.quad_over_lin(cp.maximum(cp.abs(A @ self.y), [3.0, 3.0, 3.0, 3.0]), 2.0)
        not_qpwa_obj = 3 * cp.sum_squares(cp.abs(A @ self.y)) + cp.quad_over_lin(cp.minimum(cp.abs(A @ self.y), [3.0, 3.0, 3.0, 3.0]), 2.0)
        p = Problem(cp.Minimize(obj), [])
        self.assertEqual(p.is_qp(), True)
        p = Problem(cp.Minimize(qpwa_obj), [])
        self.assertEqual(p.is_qp(), True)
        p = Problem(cp.Minimize(not_qpwa_obj), [])
        self.assertEqual(p.is_qp(), False)
        p = Problem(cp.Minimize(obj), [Aeq @ self.y == beq, F @ self.y <= g])
        self.assertEqual(p.is_qp(), True)
        p = Problem(cp.Minimize(qpwa_obj), [Aeq @ self.y == beq, F @ self.y <= g])
        self.assertEqual(p.is_qp(), True)
        p = Problem(cp.Minimize(obj), [cp.maximum(1, 3 * self.y) <= 200, cp.abs(2 * self.y) <= 100, cp.norm(2 * self.y, 1) <= 1000, Aeq @ self.y == beq])
        self.assertEqual(p.is_qp(), True)
        p = Problem(cp.Minimize(qpwa_obj), [cp.maximum(1, 3 * self.y) <= 200, cp.abs(2 * self.y) <= 100, cp.norm(2 * self.y, 1) <= 1000, Aeq @ self.y == beq])
        self.assertEqual(p.is_qp(), True)
        p = Problem(cp.Minimize(obj), [cp.maximum(1, 3 * self.y ** 2) <= 200])
        self.assertEqual(p.is_qp(), False)
        p = Problem(cp.Minimize(qpwa_obj), [cp.maximum(1, 3 * self.y ** 2) <= 200])
        self.assertEqual(p.is_qp(), False)

    def test_variable_name_conflict(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        var = Variable(name='a')
        p = Problem(cp.Maximize(self.a + var), [var == 2 + self.a, var <= 3])
        result = p.solve(solver=cp.SCS, eps=1e-05)
        self.assertAlmostEqual(result, 4.0)
        self.assertAlmostEqual(self.a.value, 1)
        self.assertAlmostEqual(var.value, 3)

    def test_add_problems(self) -> None:
        if False:
            i = 10
            return i + 15
        prob1 = Problem(cp.Minimize(self.a), [self.a >= self.b])
        prob2 = Problem(cp.Minimize(2 * self.b), [self.a >= 1, self.b >= 2])
        prob_minimize = prob1 + prob2
        self.assertEqual(len(prob_minimize.constraints), 3)
        self.assertAlmostEqual(prob_minimize.solve(solver=cp.SCS, eps=1e-06), 6)
        prob3 = Problem(cp.Maximize(self.a), [self.b <= 1])
        prob4 = Problem(cp.Maximize(2 * self.b), [self.a <= 2])
        prob_maximize = prob3 + prob4
        self.assertEqual(len(prob_maximize.constraints), 2)
        self.assertAlmostEqual(prob_maximize.solve(solver=cp.SCS, eps=1e-06), 4)
        prob5 = Problem(cp.Minimize(3 * self.a))
        prob_sum = sum([prob1, prob2, prob5])
        self.assertEqual(len(prob_sum.constraints), 3)
        self.assertAlmostEqual(prob_sum.solve(solver=cp.SCS, eps=1e-06), 12)
        prob_sum = sum([prob1])
        self.assertEqual(len(prob_sum.constraints), 1)
        with self.assertRaises(DCPError) as cm:
            prob1 + prob3
        self.assertEqual(str(cm.exception), 'Problem does not follow DCP rules.')

    def test_mul_problems(self) -> None:
        if False:
            i = 10
            return i + 15
        prob1 = Problem(cp.Minimize(pow(self.a, 2)), [self.a >= 2])
        answer = prob1.solve(solver=cp.SCS)
        factors = [0, 1, 2.3, -4.321]
        for f in factors:
            self.assertAlmostEqual((f * prob1).solve(solver=cp.SCS), f * answer, places=3)
            self.assertAlmostEqual((prob1 * f).solve(solver=cp.SCS), f * answer, places=3)

    def test_lin_combination_problems(self) -> None:
        if False:
            i = 10
            return i + 15
        prob1 = Problem(cp.Minimize(self.a), [self.a >= self.b])
        prob2 = Problem(cp.Minimize(2 * self.b), [self.a >= 1, self.b >= 2])
        prob3 = Problem(cp.Maximize(-pow(self.b + self.a, 2)), [self.b >= 3])
        combo1 = prob1 + 2 * prob2
        combo1_ref = Problem(cp.Minimize(self.a + 4 * self.b), [self.a >= self.b, self.a >= 1, self.b >= 2])
        self.assertAlmostEqual(combo1.solve(solver=cp.ECOS), combo1_ref.solve(solver=cp.ECOS))
        combo2 = prob1 - prob3 / 2
        combo2_ref = Problem(cp.Minimize(self.a + pow(self.b + self.a, 2) / 2), [self.b >= 3, self.a >= self.b])
        self.assertAlmostEqual(combo2.solve(solver=cp.ECOS), combo2_ref.solve(solver=cp.ECOS))
        combo3 = prob1 + 0 * prob2 - 3 * prob3
        combo3_ref = Problem(cp.Minimize(self.a + 3 * pow(self.b + self.a, 2)), [self.a >= self.b, self.a >= 1, self.b >= 3])
        self.assertAlmostEqual(combo3.solve(solver=cp.ECOS), combo3_ref.solve(solver=cp.ECOS))

    def test_scalar_lp(self) -> None:
        if False:
            return 10
        p = Problem(cp.Minimize(3 * self.a), [self.a >= 2])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 6)
        self.assertAlmostEqual(self.a.value, 2)
        p = Problem(cp.Maximize(3 * self.a - self.b), [self.a <= 2, self.b == self.a, self.b <= 5])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 4.0)
        self.assertAlmostEqual(self.a.value, 2)
        self.assertAlmostEqual(self.b.value, 2)
        p = Problem(cp.Minimize(3 * self.a - self.b + 100), [self.a >= 2, self.b + 5 * self.c - 2 == self.a, self.b <= 5 + self.c])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 101 + 1.0 / 6)
        self.assertAlmostEqual(self.a.value, 2)
        self.assertAlmostEqual(self.b.value, 5 - 1.0 / 6)
        self.assertAlmostEqual(self.c.value, -1.0 / 6)
        exp = cp.Maximize(self.a)
        p = Problem(exp, [self.a <= 2])
        result = p.solve(solver=s.ECOS)
        self.assertEqual(result, p.value)
        self.assertEqual(p.status, s.OPTIMAL)
        assert self.a.value is not None
        assert p.constraints[0].dual_value is not None
        p = Problem(cp.Maximize(self.a), [self.a >= 2])
        p.solve(solver=s.ECOS)
        self.assertEqual(p.status, s.UNBOUNDED)
        assert numpy.isinf(p.value)
        assert p.value > 0
        assert self.a.value is None
        assert p.constraints[0].dual_value is None
        if s.CVXOPT in INSTALLED_SOLVERS:
            p = Problem(cp.Minimize(-self.a), [self.a >= 2])
            result = p.solve(solver=s.CVXOPT)
            self.assertEqual(result, p.value)
            self.assertEqual(p.status, s.UNBOUNDED)
            assert numpy.isinf(p.value)
            assert p.value < 0
        p = Problem(cp.Maximize(self.a), [self.a >= 2, self.a <= 1])
        self.a.save_value(2)
        p.constraints[0].save_dual_value(2)
        result = p.solve(solver=s.ECOS)
        self.assertEqual(result, p.value)
        self.assertEqual(p.status, s.INFEASIBLE)
        assert numpy.isinf(p.value)
        assert p.value < 0
        assert self.a.value is None
        assert p.constraints[0].dual_value is None
        p = Problem(cp.Minimize(-self.a), [self.a >= 2, self.a <= 1])
        result = p.solve(solver=s.ECOS)
        self.assertEqual(result, p.value)
        self.assertEqual(p.status, s.INFEASIBLE)
        assert numpy.isinf(p.value)
        assert p.value > 0

    def test_vector_lp(self) -> None:
        if False:
            while True:
                i = 10
        c = Constant(numpy.array([[1, 2]]).T).value
        p = Problem(cp.Minimize(c.T @ self.x), [self.x[:, None] >= c])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 5)
        self.assertItemsAlmostEqual(self.x.value, [1, 2])
        A = Constant(numpy.array([[3, 5], [1, 2]]).T).value
        Imat = Constant([[1, 0], [0, 1]])
        p = Problem(cp.Minimize(c.T @ self.x + self.a), [A @ self.x >= [-1, 1], 4 * Imat @ self.z == self.x, self.z >= [2, 2], self.a >= 2])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 26, places=3)
        obj = (c.T @ self.x + self.a).value[0]
        self.assertAlmostEqual(obj, result)
        self.assertItemsAlmostEqual(self.x.value, [8, 8], places=3)
        self.assertItemsAlmostEqual(self.z.value, [2, 2], places=3)

    def test_ecos_noineq(self) -> None:
        if False:
            print('Hello World!')
        'Test ECOS with no inequality constraints.\n        '
        T = Constant(numpy.ones((2, 2))).value
        p = Problem(cp.Minimize(1), [self.A == T])
        result = p.solve(solver=s.ECOS)
        self.assertAlmostEqual(result, 1)
        self.assertItemsAlmostEqual(self.A.value, T)

    def test_matrix_lp(self) -> None:
        if False:
            return 10
        T = Constant(numpy.ones((2, 2))).value
        p = Problem(cp.Minimize(1), [self.A == T])
        result = p.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, 1)
        self.assertItemsAlmostEqual(self.A.value, T)
        T = Constant(numpy.ones((2, 3)) * 2).value
        p = Problem(cp.Minimize(1), [self.A >= T @ self.C, self.A == self.B, self.C == T.T])
        result = p.solve(solver=cp.ECOS)
        self.assertAlmostEqual(result, 1)
        self.assertItemsAlmostEqual(self.A.value, self.B.value)
        self.assertItemsAlmostEqual(self.C.value, T)
        assert (self.A.value >= (T @ self.C).value).all()
        self.assertEqual(type(self.A.value), intf.DEFAULT_INTF.TARGET_MATRIX)

    def test_variable_promotion(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        p = Problem(cp.Minimize(self.a), [self.x <= self.a, self.x == [1, 2]])
        result = p.solve(solver=cp.ECOS)
        self.assertAlmostEqual(result, 2)
        self.assertAlmostEqual(self.a.value, 2)
        p = Problem(cp.Minimize(self.a), [self.A <= self.a, self.A == [[1, 2], [3, 4]]])
        result = p.solve(solver=cp.ECOS)
        self.assertAlmostEqual(result, 4)
        self.assertAlmostEqual(self.a.value, 4)
        p = Problem(cp.Minimize([[1], [1]] @ (self.x + self.a + 1)), [self.a + self.x >= [1, 2]])
        result = p.solve(solver=cp.ECOS)
        self.assertAlmostEqual(result, 5)

    def test_parameter_promotion(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        a = Parameter()
        exp = [[1, 2], [3, 4]] * a
        a.value = 2
        assert not (exp.value - 2 * numpy.array([[1, 2], [3, 4]]).T).any()

    def test_parameter_problems(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test problems with parameters.\n        '
        p1 = Parameter()
        p2 = Parameter(3, nonpos=True)
        p3 = Parameter((4, 4), nonneg=True)
        p = Problem(cp.Maximize(p1 * self.a), [self.a + p1 <= p2, self.b <= p3 + p3 + 2])
        p1.value = 2
        p2.value = -numpy.ones((3,))
        p3.value = numpy.ones((4, 4))
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, -6)
        p1.value = None
        with self.assertRaises(ParameterError):
            p.solve(solver=cp.SCS, eps=1e-06)

    def test_norm_inf(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        p = Problem(cp.Minimize(cp.norm_inf(-2)))
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 2)
        p = Problem(cp.Minimize(cp.norm_inf(self.a)), [self.a >= 2])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 2)
        self.assertAlmostEqual(self.a.value, 2)
        p = Problem(cp.Minimize(3 * cp.norm_inf(self.a + 2 * self.b) + self.c), [self.a >= 2, self.b <= -1, self.c == 3])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 3)
        self.assertAlmostEqual(self.a.value + 2 * self.b.value, 0)
        self.assertAlmostEqual(self.c.value, 3)
        p = Problem(cp.Maximize(-cp.norm_inf(self.a)), [self.a <= -2])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, -2)
        self.assertAlmostEqual(self.a.value, -2)
        p = Problem(cp.Minimize(cp.norm_inf(self.x - self.z) + 5), [self.x >= [2, 3], self.z <= [-1, -4]])
        result = p.solve(solver=cp.ECOS)
        self.assertAlmostEqual(float(result), 12)
        self.assertAlmostEqual(float(list(self.x.value)[1] - list(self.z.value)[1]), 7)

    def test_norm1(self) -> None:
        if False:
            i = 10
            return i + 15
        p = Problem(cp.Minimize(cp.norm1(-2)))
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 2)
        p = Problem(cp.Minimize(cp.norm1(self.a)), [self.a <= -2])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 2)
        self.assertAlmostEqual(self.a.value, -2)
        p = Problem(cp.Maximize(-cp.norm1(self.a)), [self.a <= -2])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, -2)
        self.assertAlmostEqual(self.a.value, -2)
        p = Problem(cp.Minimize(cp.norm1(self.x - self.z) + 5), [self.x >= [2, 3], self.z <= [-1, -4]])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(float(result), 15)
        self.assertAlmostEqual(float(list(self.x.value)[1] - list(self.z.value)[1]), 7)

    def test_norm2(self) -> None:
        if False:
            i = 10
            return i + 15
        p = Problem(cp.Minimize(cp.pnorm(-2, p=2)))
        result = p.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, 2)
        p = Problem(cp.Minimize(cp.pnorm(self.a, p=2)), [self.a <= -2])
        result = p.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, 2)
        self.assertAlmostEqual(self.a.value, -2)
        p = Problem(cp.Maximize(-cp.pnorm(self.a, p=2)), [self.a <= -2])
        result = p.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, -2)
        self.assertAlmostEqual(self.a.value, -2)
        p = Problem(cp.Minimize(cp.pnorm(self.x - self.z, p=2) + 5), [self.x >= [2, 3], self.z <= [-1, -4]])
        result = p.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, 12.61577)
        self.assertItemsAlmostEqual(self.x.value, [2, 3])
        self.assertItemsAlmostEqual(self.z.value, [-1, -4])
        p = Problem(cp.Minimize(cp.pnorm((self.x - self.z).T, p=2) + 5), [self.x >= [2, 3], self.z <= [-1, -4]])
        result = p.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, 12.61577)
        self.assertItemsAlmostEqual(self.x.value, [2, 3])
        self.assertItemsAlmostEqual(self.z.value, [-1, -4])

    def test_abs(self) -> None:
        if False:
            print('Hello World!')
        p = Problem(cp.Minimize(cp.sum(cp.abs(self.A))), [-2 >= self.A])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 8)
        self.assertItemsAlmostEqual(self.A.value, [-2, -2, -2, -2])

    def test_quad_form(self) -> None:
        if False:
            i = 10
            return i + 15
        with self.assertRaises(Exception) as cm:
            Problem(cp.Minimize(cp.quad_form(self.x, self.A))).solve(solver=cp.SCS, eps=1e-06)
        self.assertEqual(str(cm.exception), 'At least one argument to quad_form must be non-variable.')
        with self.assertRaises(Exception) as cm:
            Problem(cp.Minimize(cp.quad_form(1, self.A))).solve(solver=cp.SCS, eps=1e-06)
        self.assertEqual(str(cm.exception), 'Invalid dimensions for arguments.')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with self.assertRaises(Exception) as cm:
                objective = cp.Minimize(cp.quad_form(self.x, [[-1, 0], [0, 9]]))
                Problem(objective).solve(solver=cp.SCS, eps=1e-06)
            self.assertTrue('Problem does not follow DCP rules.' in str(cm.exception))
        P = [[4, 0], [0, 9]]
        p = Problem(cp.Minimize(cp.quad_form(self.x, P)), [self.x >= 1])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 13, places=3)
        c = [1, 2]
        p = Problem(cp.Minimize(cp.quad_form(c, self.A)), [self.A >= 1])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 9)
        c = [1, 2]
        P = [[4, 0], [0, 9]]
        p = Problem(cp.Minimize(cp.quad_form(c, P)))
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 40)

    def test_mixed_atoms(self) -> None:
        if False:
            print('Hello World!')
        p = Problem(cp.Minimize(cp.pnorm(5 + cp.norm1(self.z) + cp.norm1(self.x) + cp.norm_inf(self.x - self.z), p=2)), [self.x >= [2, 3], self.z <= [-1, -4], cp.pnorm(self.x + self.z, p=2) <= 2])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 22)
        self.assertItemsAlmostEqual(self.x.value, [2, 3])
        self.assertItemsAlmostEqual(self.z.value, [-1, -4])

    def test_mult_constant_atoms(self) -> None:
        if False:
            return 10
        p = Problem(cp.Minimize(cp.pnorm([3, 4], p=2) * self.a), [self.a >= 2])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 10)
        self.assertAlmostEqual(self.a.value, 2)

    def test_dual_variables(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test recovery of dual variables.\n        '
        for solver in [s.ECOS, s.SCS, s.CVXOPT]:
            if solver in INSTALLED_SOLVERS:
                if solver == s.SCS:
                    acc = 1
                else:
                    acc = 5
                p = Problem(cp.Minimize(cp.norm1(self.x + self.z)), [self.x >= [2, 3], [[1, 2], [3, 4]] @ self.z == [-1, -4], cp.pnorm(self.x + self.z, p=2) <= 100])
                result = p.solve(solver=solver)
                self.assertAlmostEqual(result, 4, places=acc)
                self.assertItemsAlmostEqual(self.x.value, [4, 3], places=acc)
                self.assertItemsAlmostEqual(self.z.value, [-4, 1], places=acc)
                self.assertItemsAlmostEqual(p.constraints[0].dual_value, [0, 1], places=acc)
                self.assertItemsAlmostEqual(p.constraints[1].dual_value, [-1, 0.5], places=acc)
                self.assertAlmostEqual(p.constraints[2].dual_value, 0, places=acc)
                T = numpy.ones((2, 3)) * 2
                p = Problem(cp.Minimize(1), [self.A >= T @ self.C, self.A == self.B, self.C == T.T])
                result = p.solve(solver=solver)
                self.assertItemsAlmostEqual(p.constraints[0].dual_value, 4 * [0], places=acc)
                self.assertItemsAlmostEqual(p.constraints[1].dual_value, 4 * [0], places=acc)
                self.assertItemsAlmostEqual(p.constraints[2].dual_value, 6 * [0], places=acc)

    def test_indexing(self) -> None:
        if False:
            i = 10
            return i + 15
        p = Problem(cp.Maximize(self.x[0]), [self.x[0] <= 2, self.x[1] == 3])
        result = p.solve(solver=cp.SCS, eps=1e-08)
        self.assertAlmostEqual(result, 2)
        self.assertItemsAlmostEqual(self.x.value, [2, 3])
        n = 10
        A = numpy.arange(n * n)
        A = numpy.reshape(A, (n, n))
        x = Variable((n, n))
        p = Problem(cp.Minimize(cp.sum(x)), [x == A])
        result = p.solve(solver=cp.SCS, eps=1e-08)
        answer = n * n * (n * n + 1) / 2 - n * n
        self.assertAlmostEqual(result, answer)
        p = Problem(cp.Maximize(sum((self.A[i, i] + self.A[i, 1 - i] for i in range(2)))), [self.A <= [[1, -2], [-3, 4]]])
        result = p.solve(solver=cp.SCS, eps=1e-08)
        self.assertAlmostEqual(result, 0)
        self.assertItemsAlmostEqual(self.A.value, [1, -2, -3, 4])
        expr = [[1, 2], [3, 4]] @ self.z + self.x
        p = Problem(cp.Minimize(expr[1]), [self.x == self.z, self.z == [1, 2]])
        result = p.solve(solver=cp.SCS, eps=1e-08)
        self.assertAlmostEqual(result, 12)
        self.assertItemsAlmostEqual(self.x.value, self.z.value)

    def test_non_python_int_index(self) -> None:
        if False:
            return 10
        'Test problems that have special types as indices.\n        '
        import sys
        if sys.version_info > (3,):
            my_long = int
        else:
            my_long = long
        cost = self.x[0:my_long(2)][0]
        p = Problem(cp.Minimize(cost), [self.x == 1])
        result = p.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, 1)
        self.assertItemsAlmostEqual(self.x.value, [1, 1])
        cost = self.x[0:numpy.int64(2)][0]
        p = Problem(cp.Minimize(cost), [self.x == 1])
        result = p.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, 1)
        self.assertItemsAlmostEqual(self.x.value, [1, 1])

    def test_slicing(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        p = Problem(cp.Maximize(cp.sum(self.C)), [self.C[1:3, :] <= 2, self.C[0, :] == 1])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 10)
        self.assertItemsAlmostEqual(self.C.value, 2 * [1, 2, 2])
        p = Problem(cp.Maximize(cp.sum(self.C[0:3:2, 1])), [self.C[1:3, :] <= 2, self.C[0, :] == 1])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 3)
        self.assertItemsAlmostEqual(self.C.value[0:3:2, 1], [1, 2])
        p = Problem(cp.Maximize(cp.sum((self.C[0:2, :] + self.A)[:, 0:2])), [self.C[1:3, :] <= 2, self.C[0, :] == 1, (self.A + self.B)[:, 0] == 3, (self.A + self.B)[:, 1] == 2, self.B == 1])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 12)
        self.assertItemsAlmostEqual(self.C.value[0:2, :], [1, 2, 1, 2])
        self.assertItemsAlmostEqual(self.A.value, [2, 2, 1, 1])
        p = Problem(cp.Maximize([[3], [4]] @ (self.C[0:2, :] + self.A)[:, 0]), [self.C[1:3, :] <= 2, self.C[0, :] == 1, [[1], [2]] @ (self.A + self.B)[:, 0] == 3, (self.A + self.B)[:, 1] == 2, self.B == 1, 3 * self.A[:, 0] <= 3])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 12)
        self.assertItemsAlmostEqual(self.C.value[0:2, 0], [1, 2])
        self.assertItemsAlmostEqual(self.A.value, [1, -0.5, 1, 1])
        p = Problem(cp.Minimize(cp.pnorm((self.C[0:2, :] + self.A)[:, 0], p=2)), [self.C[1:3, :] <= 2, self.C[0, :] == 1, (self.A + self.B)[:, 0] == 3, (self.A + self.B)[:, 1] == 2, self.B == 1])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 3)
        self.assertItemsAlmostEqual(self.C.value[0:2, 0], [1, -2], places=3)
        self.assertItemsAlmostEqual(self.A.value, [2, 2, 1, 1])
        p = Problem(cp.Maximize(cp.sum(self.C)), [self.C[1:3, :].T <= 2, self.C[0, :].T == 1])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 10)
        self.assertItemsAlmostEqual(self.C.value, 2 * [1, 2, 2])

    def test_vstack(self) -> None:
        if False:
            i = 10
            return i + 15
        a = Variable((1, 1), name='a')
        b = Variable((1, 1), name='b')
        x = Variable((2, 1), name='x')
        y = Variable((3, 1), name='y')
        c = numpy.ones((1, 5))
        problem = Problem(cp.Minimize(c @ cp.vstack([x, y])), [x == [[1, 2]], y == [[3, 4, 5]]])
        result = problem.solve(solver=cp.SCS, eps=1e-05)
        self.assertAlmostEqual(result, 15)
        c = numpy.ones((1, 4))
        problem = Problem(cp.Minimize(c @ cp.vstack([x, x])), [x == [[1, 2]]])
        result = problem.solve(solver=cp.SCS, eps=1e-08)
        self.assertAlmostEqual(result, 6)
        c = numpy.ones((2, 2))
        problem = Problem(cp.Minimize(cp.sum(cp.vstack([self.A, self.C]))), [self.A >= 2 * c, self.C == -2])
        result = problem.solve(solver=cp.SCS, eps=1e-08)
        self.assertAlmostEqual(result, -4)
        c = numpy.ones((1, 2))
        problem = Problem(cp.Minimize(cp.sum(cp.vstack([c @ self.A, c @ self.B]))), [self.A >= 2, self.B == -2])
        result = problem.solve(solver=cp.SCS, eps=1e-08)
        self.assertAlmostEqual(result, 0)
        c = numpy.array([[1, -1]]).T
        problem = Problem(cp.Minimize(c.T @ cp.vstack([cp.square(a), cp.sqrt(b)])), [a == 2, b == 16])
        with self.assertRaises(Exception) as cm:
            problem.solve(solver=cp.SCS, eps=1e-05)
        self.assertTrue('Problem does not follow DCP rules.' in str(cm.exception))
        p = Parameter((2, 1), value=np.array([[3], [3]]))
        q = Parameter((2, 1), value=np.array([[-8], [-8]]))
        vars_arg = cp.vstack([cp.vstack([a, a]), cp.vstack([b, b])])
        problem = Problem(cp.Minimize(cp.vstack([p, q]).T @ vars_arg), [a == 1, b == 2])
        problem.solve()
        self.assertAlmostEqual(problem.value, -26)

    def test_hstack(self) -> None:
        if False:
            print('Hello World!')
        a = Variable((1, 1), name='a')
        b = Variable((1, 1), name='b')
        x = Variable((2, 1), name='x')
        y = Variable((3, 1), name='y')
        c = numpy.ones((1, 5))
        problem = Problem(cp.Minimize(c @ cp.hstack([x.T, y.T]).T), [x == [[1, 2]], y == [[3, 4, 5]]])
        result = problem.solve(solver=cp.SCS, eps=1e-08)
        self.assertAlmostEqual(result, 15)
        c = numpy.ones((1, 4))
        problem = Problem(cp.Minimize(c @ cp.hstack([x.T, x.T]).T), [x == [[1, 2]]])
        result = problem.solve(solver=cp.SCS, eps=1e-08)
        self.assertAlmostEqual(result, 6)
        c = numpy.ones((2, 2))
        problem = Problem(cp.Minimize(cp.sum(cp.hstack([self.A.T, self.C.T]))), [self.A >= 2 * c, self.C == -2])
        result = problem.solve(solver=cp.SCS, eps=1e-08)
        self.assertAlmostEqual(result, -4)
        D = Variable((3, 3))
        expr = cp.hstack([self.C, D])
        problem = Problem(cp.Minimize(expr[0, 1] + cp.sum(cp.hstack([expr, expr]))), [self.C >= 0, D >= 0, D[0, 0] == 2, self.C[0, 1] == 3])
        result = problem.solve(solver=cp.SCS, eps=1e-08)
        self.assertAlmostEqual(result, 13)
        c = numpy.array([[1, -1]]).T
        problem = Problem(cp.Minimize(c.T @ cp.hstack([cp.square(a).T, cp.sqrt(b).T]).T), [a == 2, b == 16])
        with self.assertRaises(Exception) as cm:
            problem.solve(solver=cp.SCS, eps=1e-05)
        self.assertTrue('Problem does not follow DCP rules.' in str(cm.exception))
        (p, q) = (Parameter(2, value=[3, 3]), Parameter(2, value=[-8, -8]))
        vars_arg = cp.hstack([cp.hstack([a[0], a[0]]), cp.hstack([b[0], b[0]])])
        problem = Problem(cp.Minimize(cp.hstack([p, q]).T @ vars_arg), [a == 1, b == 2])
        problem.solve()
        self.assertAlmostEqual(problem.value, -26)

    def test_bad_objective(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test using a cvxpy expression as an objective.\n        '
        with self.assertRaises(Exception) as cm:
            Problem(self.x + 2)
        self.assertEqual(str(cm.exception), 'Problem objective must be Minimize or Maximize.')

    def test_transpose(self) -> None:
        if False:
            return 10
        p = Problem(cp.Minimize(cp.sum(self.x)), [self.x[None, :] >= numpy.array([[1, 2]])])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 3)
        self.assertItemsAlmostEqual(self.x.value, [1, 2])
        p = Problem(cp.Minimize(cp.sum(self.C)), [numpy.array([[1, 1]]) @ self.C.T >= numpy.array([[0, 1, 2]])])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        value = self.C.value
        constraints = [1 * self.C[i, 0] + 1 * self.C[i, 1] >= i for i in range(3)]
        p = Problem(cp.Minimize(cp.sum(self.C)), constraints)
        result2 = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, result2)
        self.assertItemsAlmostEqual(self.C.value, value)
        p = Problem(cp.Minimize(self.A[0, 1] - self.A.T[1, 0]), [self.A == [[1, 2], [3, 4]]])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 0)
        p = Problem(cp.Minimize(cp.sum(self.x)), [(-self.x).T <= 1])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, -2)
        c = numpy.array([[1, -1]]).T
        p = Problem(cp.Minimize(cp.maximum(c.T, 2, 2 + c.T)[0, 1]))
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 2)
        c = numpy.array([[1, -1, 2], [1, -1, 2]]).T
        p = Problem(cp.Minimize(cp.sum(cp.maximum(c, 2, 2 + c).T[:, 0])))
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 6)
        c = numpy.array([[1, -1, 2], [1, -1, 2]]).T
        p = Problem(cp.Minimize(cp.sum(cp.square(c.T).T[:, 0])))
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 6)
        p = Problem(cp.Maximize(cp.sum(self.C)), [self.C.T[:, 1:3] <= 2, self.C.T[:, 0] == 1])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 10)
        self.assertItemsAlmostEqual(self.C.value, 2 * [1, 2, 2])

    def test_multiplication_on_left(self) -> None:
        if False:
            while True:
                i = 10
        'Test multiplication on the left by a non-constant.\n        '
        c = numpy.array([[1, 2]]).T
        p = Problem(cp.Minimize(c.T @ self.A @ c), [self.A >= 2])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 18)
        p = Problem(cp.Minimize(self.a * 2), [self.a >= 2])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 4)
        p = Problem(cp.Minimize(self.x.T @ c), [self.x >= 2])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 6)
        p = Problem(cp.Minimize((self.x.T + self.z.T) @ c), [self.x >= 2, self.z >= 1])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 9)
        A = numpy.ones((5, 10))
        x = Variable(5)
        p = cp.Problem(cp.Minimize(cp.sum(x @ A)), [x >= 0])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 0)

    def test_redundant_constraints(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        obj = cp.Minimize(cp.sum(self.x))
        constraints = [self.x == 2, self.x == 2, self.x.T == 2, self.x[0] == 2]
        p = Problem(obj, constraints)
        result = p.solve(solver=s.SCS)
        self.assertAlmostEqual(result, 4)
        obj = cp.Minimize(cp.sum(cp.square(self.x)))
        constraints = [self.x == self.x]
        p = Problem(obj, constraints)
        result = p.solve(solver=s.SCS)
        self.assertAlmostEqual(result, 0)
        with self.assertRaises(ValueError) as cm:
            obj = cp.Minimize(cp.sum(cp.square(self.x)))
            constraints = [self.x == self.x]
            problem = Problem(obj, constraints)
            problem.solve(solver=s.ECOS)
        self.assertEqual(str(cm.exception), 'ECOS cannot handle sparse data with nnz == 0; this is a bug in ECOS, and it indicates that your problem might have redundant constraints.')

    def test_sdp_symmetry(self) -> None:
        if False:
            return 10
        p = Problem(cp.Minimize(cp.lambda_max(self.A)), [self.A >= 2])
        p.solve(solver=cp.SCS)
        self.assertItemsAlmostEqual(self.A.value, self.A.value.T, places=3)
        p = Problem(cp.Minimize(cp.lambda_max(self.A)), [self.A == [[1, 2], [3, 4]]])
        p.solve(solver=cp.SCS)
        self.assertEqual(p.status, s.INFEASIBLE)

    def test_sdp(self) -> None:
        if False:
            return 10
        obj = cp.Maximize(self.A[1, 0] - self.A[0, 1])
        p = Problem(obj, [cp.lambda_max(self.A) <= 100, self.A[0, 0] == 2, self.A[1, 1] == 2, self.A[1, 0] == 2])
        result = p.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, 0, places=3)

    def test_expression_values(self) -> None:
        if False:
            print('Hello World!')
        diff_exp = self.x - self.z
        inf_exp = cp.norm_inf(diff_exp)
        sum_exp = 5 + cp.norm1(self.z) + cp.norm1(self.x) + inf_exp
        constr_exp = cp.pnorm(self.x + self.z, p=2)
        obj = cp.pnorm(sum_exp, p=2)
        p = Problem(cp.Minimize(obj), [self.x >= [2, 3], self.z <= [-1, -4], constr_exp <= 2])
        result = p.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 22)
        self.assertItemsAlmostEqual(self.x.value, [2, 3])
        self.assertItemsAlmostEqual(self.z.value, [-1, -4])
        self.assertItemsAlmostEqual(diff_exp.value, self.x.value - self.z.value)
        self.assertAlmostEqual(inf_exp.value, LA.norm(self.x.value - self.z.value, numpy.inf))
        self.assertAlmostEqual(sum_exp.value, 5 + LA.norm(self.z.value, 1) + LA.norm(self.x.value, 1) + LA.norm(self.x.value - self.z.value, numpy.inf))
        self.assertAlmostEqual(constr_exp.value, LA.norm(self.x.value + self.z.value, 2))
        self.assertAlmostEqual(obj.value, result)

    def test_mult_by_zero(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test multiplication by zero.\n        '
        self.a.value = 1
        exp = 0 * self.a
        self.assertEqual(exp.value, 0)
        obj = cp.Minimize(exp)
        p = Problem(obj)
        result = p.solve(solver=cp.ECOS)
        self.assertAlmostEqual(result, 0)
        assert self.a.value is not None

    def test_div(self) -> None:
        if False:
            print('Hello World!')
        'Tests a problem with division.\n        '
        obj = cp.Minimize(cp.norm_inf(self.A / 5))
        p = Problem(obj, [self.A >= 5])
        result = p.solve(solver=cp.ECOS)
        self.assertAlmostEqual(result, 1)
        c = cp.Constant([[1.0, -1], [2, -2]])
        expr = self.A / (1.0 / c)
        obj = cp.Minimize(cp.norm_inf(expr))
        p = Problem(obj, [self.A == 5])
        result = p.solve(solver=cp.ECOS)
        self.assertAlmostEqual(result, 10)
        self.assertItemsAlmostEqual(expr.value, [5, -5] + [10, -10])
        import scipy.sparse as sp
        interface = intf.get_matrix_interface(sp.csc_matrix)
        c = interface.const_to_matrix([1, 2])
        c = cp.Constant(c)
        expr = self.x[:, None] / (1 / c)
        obj = cp.Minimize(cp.norm_inf(expr))
        p = Problem(obj, [self.x == 5])
        result = p.solve(solver=cp.ECOS)
        self.assertAlmostEqual(result, 10)
        self.assertItemsAlmostEqual(expr.value, [5, 10])
        c = [[1, -1], [2, -2]]
        c = cp.Constant(c)
        expr = self.a / (1 / c)
        obj = cp.Minimize(cp.norm_inf(expr))
        p = Problem(obj, [self.a == 5])
        result = p.solve(solver=cp.ECOS)
        self.assertAlmostEqual(result, 10)
        self.assertItemsAlmostEqual(expr.value, [5, -5] + [10, -10])

    def test_multiply(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Tests problems with multiply.\n        '
        c = [[1, -1], [2, -2]]
        expr = cp.multiply(c, self.A)
        obj = cp.Minimize(cp.norm_inf(expr))
        p = Problem(obj, [self.A == 5])
        result = p.solve(solver=cp.ECOS)
        self.assertAlmostEqual(result, 10)
        self.assertItemsAlmostEqual(expr.value, [5, -5] + [10, -10])
        import scipy.sparse as sp
        interface = intf.get_matrix_interface(sp.csc_matrix)
        c = interface.const_to_matrix([1, 2])
        expr = cp.multiply(c, self.x[:, None])
        obj = cp.Minimize(cp.norm_inf(expr))
        p = Problem(obj, [self.x == 5])
        result = p.solve(solver=cp.ECOS)
        self.assertAlmostEqual(result, 10)
        self.assertItemsAlmostEqual(expr.value.toarray(), [5, 10])
        c = [[1, -1], [2, -2]]
        expr = cp.multiply(c, self.a)
        obj = cp.Minimize(cp.norm_inf(expr))
        p = Problem(obj, [self.a == 5])
        result = p.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, 10)
        self.assertItemsAlmostEqual(expr.value, [5, -5] + [10, -10])

    def test_invalid_solvers(self) -> None:
        if False:
            while True:
                i = 10
        'Tests that errors occur when you use an invalid solver.\n        '
        with self.assertRaises(SolverError):
            Problem(cp.Minimize(Variable(boolean=True))).solve(solver=s.ECOS)
        with self.assertRaises(SolverError):
            Problem(cp.Minimize(cp.lambda_max(self.A))).solve(solver=s.ECOS)
        with self.assertRaises(SolverError):
            Problem(cp.Minimize(self.a)).solve(solver=s.SCS)

    def test_solver_error_raised_on_failure(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that a SolverError is raised when a solver fails.\n        '
        A = numpy.random.randn(40, 40)
        b = cp.matmul(A, numpy.random.randn(40))
        with self.assertRaises(SolverError):
            Problem(cp.Minimize(cp.sum_squares(cp.matmul(A, cp.Variable(40)) - b))).solve(solver=s.OSQP, max_iter=1)

    def test_reshape(self) -> None:
        if False:
            return 10
        'Tests problems with reshape.\n        '
        self.assertEqual(cp.reshape(1, (1, 1)).value, 1)
        x = Variable(4)
        mat = numpy.array([[1, -1], [2, -2]]).T
        vec = numpy.array([[1, 2, 3, 4]]).T
        vec_mat = numpy.array([[1, 2], [3, 4]]).T
        expr = cp.reshape(x, (2, 2))
        obj = cp.Minimize(cp.sum(mat @ expr))
        prob = Problem(obj, [x[:, None] == vec])
        result = prob.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, numpy.sum(mat.dot(vec_mat)))
        c = [1, 2, 3, 4]
        expr = cp.reshape(self.A, (4, 1))
        obj = cp.Minimize(expr.T @ c)
        constraints = [self.A == [[-1, -2], [3, 4]]]
        prob = Problem(obj, constraints)
        result = prob.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, 20)
        self.assertItemsAlmostEqual(expr.value, [-1, -2, 3, 4])
        self.assertItemsAlmostEqual(cp.reshape(expr, (2, 2)).value, [-1, -2, 3, 4])
        expr = cp.reshape(self.C, (2, 3))
        mat = numpy.array([[1, -1], [2, -2]])
        C_mat = numpy.array([[1, 4], [2, 5], [3, 6]])
        obj = cp.Minimize(cp.sum(mat @ expr))
        prob = Problem(obj, [self.C == C_mat])
        result = prob.solve(solver=cp.SCS)
        reshaped = numpy.reshape(C_mat, (2, 3), 'F')
        self.assertAlmostEqual(result, mat.dot(reshaped).sum())
        self.assertItemsAlmostEqual(expr.value, C_mat)
        c = numpy.array([[1, -1], [2, -2]]).T
        expr = cp.reshape(c * self.a, (1, 4))
        obj = cp.Minimize(expr @ [1, 2, 3, 4])
        prob = Problem(obj, [self.a == 2])
        result = prob.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, -6)
        self.assertItemsAlmostEqual(expr.value, 2 * c)
        expr = cp.reshape(c * self.a, (4, 1))
        obj = cp.Minimize(expr.T @ [1, 2, 3, 4])
        prob = Problem(obj, [self.a == 2])
        result = prob.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, -6)
        self.assertItemsAlmostEqual(expr.value, 2 * c)

    def test_cumsum(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test problems with cumsum.\n        '
        tt = cp.Variable(5)
        prob = cp.Problem(cp.Minimize(cp.sum(tt)), [cp.cumsum(tt, 0) >= -0.0001])
        result = prob.solve(solver=cp.SCS, eps=1e-08)
        self.assertAlmostEqual(result, -0.0001)

    def test_cummax(self) -> None:
        if False:
            print('Hello World!')
        'Test problems with cummax.\n        '
        tt = cp.Variable(5)
        prob = cp.Problem(cp.Maximize(cp.sum(tt)), [cp.cummax(tt, 0) <= numpy.array([1, 2, 3, 4, 5])])
        result = prob.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 15)

    def test_vec(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests problems with vec.\n        '
        c = [1, 2, 3, 4]
        expr = cp.vec(self.A)
        obj = cp.Minimize(expr.T @ c)
        constraints = [self.A == [[-1, -2], [3, 4]]]
        prob = Problem(obj, constraints)
        result = prob.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, 20)
        self.assertItemsAlmostEqual(expr.value, [-1, -2, 3, 4])

    def test_diag_prob(self) -> None:
        if False:
            return 10
        'Test a problem with diag.\n        '
        C = Variable((3, 3))
        obj = cp.Maximize(C[0, 2])
        constraints = [cp.diag(C) == 1, C[0, 1] == 0.6, C[1, 2] == -0.3, C == Variable((3, 3), PSD=True)]
        prob = Problem(obj, constraints)
        result = prob.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, 0.583151, places=2)

    def test_diag_offset_problem(self) -> None:
        if False:
            print('Hello World!')
        n = 4
        A = np.arange(int(n ** 2)).reshape((n, n))
        for k in range(-n + 1, n):
            x = cp.Variable(n - abs(k))
            obj = cp.Minimize(cp.sum(x))
            constraints = [cp.diag(x, k) == np.diag(np.diag(A, k), k)]
            prob = cp.Problem(obj, constraints)
            result = prob.solve(solver=cp.SCS, eps=1e-06)
            self.assertAlmostEqual(result, np.sum(np.diag(A, k)))
            assert np.allclose(x.value, np.diag(A, k), atol=0.0001)
            X = cp.Variable((n, n), nonneg=True)
            obj = cp.Minimize(cp.sum(X))
            constraints = [cp.diag(X, k) == np.diag(A, k)]
            prob = cp.Problem(obj, constraints)
            result = prob.solve(solver=cp.SCS, eps=1e-06)
            self.assertAlmostEqual(result, np.sum(np.diag(A, k)))
            assert np.allclose(X.value, np.diag(np.diag(A, k), k), atol=0.0001)

    def test_presolve_parameters(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test presolve with parameters.\n        '
        gamma = Parameter(nonneg=True)
        x = Variable()
        obj = cp.Minimize(x)
        prob = Problem(obj, [gamma == 1, x >= 0])
        gamma.value = 0
        prob.solve(solver=s.SCS)
        self.assertEqual(prob.status, s.INFEASIBLE)
        gamma.value = 1
        prob.solve(solver=s.SCS)
        self.assertEqual(prob.status, s.OPTIMAL)

    def test_parameter_expressions(self) -> None:
        if False:
            while True:
                i = 10
        'Test that expressions with parameters are updated properly.\n        '
        x = Variable()
        y = Variable()
        x0 = Parameter()
        xSquared = x0 * x0 + 2 * x0 * (x - x0)
        x0.value = 2
        g = xSquared - y
        obj = cp.abs(x - 1)
        prob = Problem(cp.Minimize(obj), [g == 0])
        self.assertFalse(prob.is_dpp())
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prob.solve(cp.SCS)
        x0.value = 1
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prob.solve(solver=cp.SCS)
        self.assertAlmostEqual(g.value, 0)
        prob = Problem(cp.Minimize(x0 * x), [x == 1])
        x0.value = 2
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prob.solve(solver=cp.SCS)
        x0.value = 1
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prob.solve(solver=cp.SCS)
        self.assertAlmostEqual(prob.value, 1, places=2)

    def test_psd_constraints(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test positive definite constraints.\n        '
        C = Variable((3, 3))
        obj = cp.Maximize(C[0, 2])
        constraints = [cp.diag(C) == 1, C[0, 1] == 0.6, C[1, 2] == -0.3, C == C.T, C >> 0]
        prob = Problem(obj, constraints)
        result = prob.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, 0.583151, places=2)
        C = Variable((2, 2))
        obj = cp.Maximize(C[0, 1])
        constraints = [C == 1, C >> [[2, 0], [0, 2]]]
        prob = Problem(obj, constraints)
        result = prob.solve(solver=cp.SCS)
        self.assertEqual(prob.status, s.INFEASIBLE)
        C = Variable((2, 2), symmetric=True)
        obj = cp.Minimize(C[0, 0])
        constraints = [C << [[2, 0], [0, 2]]]
        prob = Problem(obj, constraints)
        result = prob.solve(solver=cp.SCS)
        self.assertEqual(prob.status, s.UNBOUNDED)

    def test_psd_duals(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test the duals of PSD constraints.\n        '
        if s.CVXOPT in INSTALLED_SOLVERS:
            C = Variable((2, 2), symmetric=True, name='C')
            obj = cp.Maximize(C[0, 0])
            constraints = [C << [[2, 0], [0, 2]]]
            prob = Problem(obj, constraints)
            result = prob.solve(solver=s.CVXOPT)
            self.assertAlmostEqual(result, 2)
            psd_constr_dual = constraints[0].dual_value.copy()
            C = Variable((2, 2), symmetric=True, name='C')
            X = Variable((2, 2), PSD=True)
            obj = cp.Maximize(C[0, 0])
            constraints = [X == [[2, 0], [0, 2]] - C]
            prob = Problem(obj, constraints)
            result = prob.solve(solver=s.CVXOPT)
            new_constr_dual = (constraints[0].dual_value + constraints[0].dual_value.T) / 2
            self.assertItemsAlmostEqual(new_constr_dual, psd_constr_dual)
        C = Variable((2, 2), symmetric=True)
        obj = cp.Maximize(C[0, 0])
        constraints = [C << [[2, 0], [0, 2]]]
        prob = Problem(obj, constraints)
        result = prob.solve(solver=s.SCS)
        self.assertAlmostEqual(result, 2, places=4)
        psd_constr_dual = constraints[0].dual_value
        C = Variable((2, 2), symmetric=True)
        X = Variable((2, 2), PSD=True)
        obj = cp.Maximize(C[0, 0])
        constraints = [X == [[2, 0], [0, 2]] - C]
        prob = Problem(obj, constraints)
        result = prob.solve(solver=s.SCS)
        self.assertItemsAlmostEqual(constraints[0].dual_value, psd_constr_dual)
        C = Variable((2, 2), symmetric=True)
        obj = cp.Maximize(C[0, 1] + C[1, 0])
        constraints = [C << [[2, 0], [0, 2]], C >= 0]
        prob = Problem(obj, constraints)
        result = prob.solve(solver=s.SCS)
        self.assertAlmostEqual(result, 4, places=3)
        psd_constr_dual = constraints[0].dual_value
        C = Variable((2, 2), symmetric=True)
        X = Variable((2, 2), PSD=True)
        obj = cp.Maximize(C[0, 1] + C[1, 0])
        constraints = [X == [[2, 0], [0, 2]] - C, C >= 0]
        prob = Problem(obj, constraints)
        result = prob.solve(solver=s.SCS)
        self.assertItemsAlmostEqual(constraints[0].dual_value, psd_constr_dual, places=3)

    def test_geo_mean(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        import numpy as np
        x = Variable(2)
        cost = cp.geo_mean(x)
        prob = Problem(cp.Maximize(cost), [x <= 1])
        prob.solve(solver=cp.SCS, eps=1e-05)
        self.assertAlmostEqual(prob.value, 1)
        prob = Problem(cp.Maximize(cost), [cp.sum(x) <= 1])
        prob.solve(solver=cp.SCS, eps=1e-05)
        self.assertItemsAlmostEqual(x.value, [0.5, 0.5])
        x = Variable((3, 3))
        self.assertRaises(ValueError, cp.geo_mean, x)
        x = Variable((3, 1))
        g = cp.geo_mean(x)
        self.assertSequenceEqual(g.w, [Fraction(1, 3)] * 3)
        x = Variable((1, 5))
        g = cp.geo_mean(x)
        self.assertSequenceEqual(g.w, [Fraction(1, 5)] * 5)
        p = np.array([0.07, 0.12, 0.23, 0.19, 0.39])

        def short_geo_mean(x, p):
            if False:
                for i in range(10):
                    print('nop')
            p = np.array(p) / sum(p)
            x = np.array(x)
            return np.prod(x ** p)
        x = Variable(5)
        prob = Problem(cp.Maximize(cp.geo_mean(x, p)), [cp.sum(x) <= 1])
        prob.solve(solver=cp.SCS, eps=1e-05)
        x = np.array(x.value).flatten()
        x_true = p / sum(p)
        self.assertTrue(np.allclose(prob.value, cp.geo_mean(list(x), p).value))
        self.assertTrue(np.allclose(prob.value, short_geo_mean(x, p)))
        self.assertTrue(np.allclose(x, x_true, 0.001))
        x = Variable(5)
        prob = Problem(cp.Maximize(cp.geo_mean(x, p)), [cp.norm(x) <= 1])
        prob.solve(solver=cp.SCS, eps=1e-05)
        x = np.array(x.value).flatten()
        x_true = np.sqrt(p / sum(p))
        self.assertTrue(np.allclose(prob.value, cp.geo_mean(list(x), p).value))
        self.assertTrue(np.allclose(prob.value, short_geo_mean(x, p)))
        self.assertTrue(np.allclose(x, x_true, 0.001))
        n = 5
        x_true = np.ones(n)
        x = Variable(n)
        Problem(cp.Maximize(cp.geo_mean(x)), [x <= 1]).solve(solver=cp.SCS)
        xval = np.array(x.value).flatten()
        self.assertTrue(np.allclose(xval, x_true, 0.001))
        y = cp.vstack([x[i] for i in range(n)])
        Problem(cp.Maximize(cp.geo_mean(y)), [x <= 1]).solve(solver=cp.SCS)
        xval = np.array(x.value).flatten()
        self.assertTrue(np.allclose(xval, x_true, 0.001))
        y = cp.hstack([x[i] for i in range(n)])
        Problem(cp.Maximize(cp.geo_mean(y)), [x <= 1]).solve(solver=cp.SCS)
        xval = np.array(x.value).flatten()
        self.assertTrue(np.allclose(xval, x_true, 0.001))

    def test_pnorm(self) -> None:
        if False:
            while True:
                i = 10
        import numpy as np
        x = Variable(3, name='x')
        a = np.array([1.0, 2, 3])
        for p in (1, 1.6, 1.3, 2, 1.99, 3, 3.7, np.inf):
            prob = Problem(cp.Minimize(cp.pnorm(x, p=p)), [x.T @ a >= 1])
            prob.solve(solver=cp.ECOS, verbose=True)
            if p == np.inf:
                x_true = np.ones_like(a) / sum(a)
            elif p == 1:
                x_true = np.array([0, 0, 1.0 / 3])
            else:
                x_true = a ** (1.0 / (p - 1)) / a.dot(a ** (1.0 / (p - 1)))
            x_alg = np.array(x.value).flatten()
            self.assertTrue(np.allclose(x_alg, x_true, 0.01), 'p = {}'.format(p))
            self.assertTrue(np.allclose(prob.value, np.linalg.norm(x_alg, p)))
            self.assertTrue(np.allclose(np.linalg.norm(x_alg, p), cp.pnorm(x_alg, p).value))

    def test_pnorm_concave(self) -> None:
        if False:
            return 10
        import numpy as np
        x = Variable(3, name='x')
        a = np.array([-1.0, 2, 3])
        for p in (-1, 0.5, 0.3, -2.3):
            prob = Problem(cp.Minimize(cp.sum(cp.abs(x - a))), [cp.pnorm(x, p) >= 0])
            prob.solve(solver=cp.ECOS)
            self.assertTrue(np.allclose(prob.value, 1))
        a = np.array([1.0, 2, 3])
        for p in (-1, 0.5, 0.3, -2.3):
            prob = Problem(cp.Minimize(cp.sum(cp.abs(x - a))), [cp.pnorm(x, p) >= 0])
            prob.solve(solver=cp.ECOS)
            self.assertAlmostEqual(prob.value, 0, places=6)

    def test_power(self) -> None:
        if False:
            return 10
        x = Variable()
        prob = Problem(cp.Minimize(cp.power(x, 1.7) + cp.power(x, -2.3) - cp.power(x, 0.45)))
        prob.solve(solver=cp.SCS, eps=1e-05)
        x = x.value
        self.assertTrue(builtins.abs(1.7 * x ** 0.7 - 2.3 * x ** (-3.3) - 0.45 * x ** (-0.55)) <= 0.001)

    def test_multiply_by_scalar(self) -> None:
        if False:
            while True:
                i = 10
        'Test a problem with multiply by a scalar.\n        '
        import numpy as np
        T = 10
        J = 20
        rvec = np.random.randn(T, J)
        dy = np.random.randn(2 * T)
        theta = Variable(J)
        delta = 0.001
        loglambda = rvec @ theta
        a = cp.multiply(dy[0:T], loglambda)
        b1 = cp.exp(loglambda)
        b2 = cp.multiply(delta, b1)
        cost = -a + b1
        cost = -a + b2
        prob = Problem(cp.Minimize(cp.sum(cost)))
        prob.solve(solver=s.SCS)
        obj = cp.Minimize(cp.sum(cp.multiply(2, self.x)))
        prob = Problem(obj, [self.x == 2])
        result = prob.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, 8)

    def test_int64(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test bug with 64 bit integers.\n        '
        q = cp.Variable(numpy.int64(2))
        objective = cp.Minimize(cp.norm(q, 1))
        problem = cp.Problem(objective)
        problem.solve(solver=cp.SCS)
        print(q.value)

    def test_neg_slice(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test bug with negative slice.\n        '
        x = cp.Variable(2)
        objective = cp.Minimize(x[0] + x[1])
        constraints = [x[-2:] >= 1]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS, eps=1e-06)
        self.assertItemsAlmostEqual(x.value, [1, 1])

    def test_pnorm_axis(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test pnorm with axis != 0.\n        '
        b = numpy.arange(2)
        X = cp.Variable(shape=(2, 10))
        expr = cp.pnorm(X, p=2, axis=1) - b
        con = [expr <= 0]
        obj = cp.Maximize(cp.sum(X))
        prob = cp.Problem(obj, con)
        prob.solve(solver=cp.ECOS)
        self.assertItemsAlmostEqual(expr.value, numpy.zeros(2))
        b = numpy.arange(10)
        X = cp.Variable(shape=(10, 2))
        expr = cp.pnorm(X, p=2, axis=1) - b
        con = [expr <= 0]
        obj = cp.Maximize(cp.sum(X))
        prob = cp.Problem(obj, con)
        prob.solve(solver=cp.ECOS)
        self.assertItemsAlmostEqual(expr.value, numpy.zeros(10))

    def test_bool_constr(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test constraints that evaluate to booleans.\n        '
        x = cp.Variable(pos=True)
        prob = cp.Problem(cp.Minimize(x), [True])
        prob.solve(solver=cp.ECOS)
        self.assertAlmostEqual(x.value, 0)
        x = cp.Variable(pos=True)
        prob = cp.Problem(cp.Minimize(x), [True] * 10)
        prob.solve(solver=cp.ECOS)
        self.assertAlmostEqual(x.value, 0)
        prob = cp.Problem(cp.Minimize(x), [42 <= x] + [True] * 10)
        prob.solve(solver=cp.ECOS)
        self.assertAlmostEqual(x.value, 42)
        prob = cp.Problem(cp.Minimize(x), [True] + [42 <= x] + [True] * 10)
        prob.solve(solver=cp.ECOS)
        self.assertAlmostEqual(x.value, 42)
        prob = cp.Problem(cp.Minimize(x), [False])
        prob.solve(solver=cp.ECOS)
        self.assertEqual(prob.status, s.INFEASIBLE)
        prob = cp.Problem(cp.Minimize(x), [False] * 10)
        prob.solve(solver=cp.ECOS)
        self.assertEqual(prob.status, s.INFEASIBLE)
        prob = cp.Problem(cp.Minimize(x), [True] * 10 + [False] + [True] * 10)
        prob.solve(solver=cp.ECOS)
        self.assertEqual(prob.status, s.INFEASIBLE)
        prob = cp.Problem(cp.Minimize(x), [42 <= x] + [True] * 10 + [False])
        prob.solve(solver=cp.ECOS)
        self.assertEqual(prob.status, s.INFEASIBLE)
        prob = cp.Problem(cp.Minimize(x), [True] + [x <= -42] + [True] * 10)
        prob.solve(solver=cp.ECOS)
        self.assertEqual(prob.status, s.INFEASIBLE)

    def test_invalid_constr(self) -> None:
        if False:
            while True:
                i = 10
        'Test a problem with an invalid constraint.\n        '
        x = cp.Variable()
        with self.assertRaisesRegex(ValueError, 'Problem has an invalid constraint.*'):
            cp.Problem(cp.Minimize(x), [cp.sum(x)])

    def test_pos(self) -> None:
        if False:
            while True:
                i = 10
        'Test the pos and neg attributes.\n        '
        x = cp.Variable(pos=True)
        prob = cp.Problem(cp.Minimize(x))
        prob.solve(solver=cp.ECOS)
        self.assertAlmostEqual(x.value, 0)
        x = cp.Variable(neg=True)
        prob = cp.Problem(cp.Maximize(x))
        prob.solve(solver=cp.ECOS)
        self.assertAlmostEqual(x.value, 0)

    def test_pickle(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test pickling and unpickling problems.\n        '
        prob = cp.Problem(cp.Minimize(2 * self.a + 3), [self.a >= 1])
        prob_str = pickle.dumps(prob)
        new_prob = pickle.loads(prob_str)
        result = new_prob.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, 5.0)
        self.assertAlmostEqual(new_prob.variables()[0].value, 1.0)

    def test_spare_int8_matrix(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test problem with sparse int8 matrix.\n           issue #809.\n        '
        a = Variable(shape=(3, 1))
        q = np.array([1.88922129, 0.06938685, 0.91948919])
        P = np.array([[280.64, -49.84, -80.0], [-49.84, 196.04, 139.0], [-80.0, 139.0, 106.0]])
        D_dense = np.array([[-1, 1, 0, 0, 0, 0], [0, -1, 1, 0, 0, 0], [0, 0, 0, -1, 1, 0]], dtype=np.int8)
        D_sparse = sp.coo_matrix(D_dense)

        def make_problem(D):
            if False:
                i = 10
                return i + 15
            obj = cp.Minimize(0.5 * cp.quad_form(a, P) - a.T @ q)
            assert obj.is_dcp()
            alpha = cp.Parameter(nonneg=True, value=2)
            constraints = [a >= 0.0, -alpha <= D.T @ a, D.T @ a <= alpha]
            prob = cp.Problem(obj, constraints)
            prob.solve(solver=cp.settings.ECOS)
            assert prob.status == 'optimal'
            return prob
        expected_coef = np.array([[-0.011728003147, 0.011728002895, 2.52e-10, -0.017524801335, 0.017524801335, 0.0]])
        make_problem(D_dense)
        coef_dense = a.value.T.dot(D_dense)
        np.testing.assert_almost_equal(expected_coef, coef_dense)
        make_problem(D_sparse)
        coef_sparse = a.value.T @ D_sparse
        np.testing.assert_almost_equal(expected_coef, coef_sparse)

    def test_special_index(self) -> None:
        if False:
            while True:
                i = 10
        'Test QP code path with special indexing.\n        '
        x = cp.Variable((1, 3))
        y = cp.sum(x[:, 0:2], axis=1)
        cost = cp.QuadForm(y, np.diag([1]))
        prob = cp.Problem(cp.Minimize(cost))
        result1 = prob.solve(solver=cp.SCS)
        x = cp.Variable((1, 3))
        y = cp.sum(x[:, [0, 1]], axis=1)
        cost = cp.QuadForm(y, np.diag([1]))
        prob = cp.Problem(cp.Minimize(cost))
        result2 = prob.solve(solver=cp.SCS)
        self.assertAlmostEqual(result1, result2)

    def test_indicator(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test a problem with indicators.\n        '
        n = 5
        m = 2
        q = np.arange(n)
        a = np.ones((m, n))
        b = np.ones((m, 1))
        x = cp.Variable((n, 1), name='x')
        constraints = [a @ x == b]
        objective = cp.Minimize(1 / 2 * cp.square(q.T @ x) + cp.transforms.indicator(constraints))
        problem = cp.Problem(objective)
        solution1 = problem.solve(solver=cp.SCS, eps=1e-05)
        objective = cp.Minimize(1 / 2 * cp.square(q.T @ x))
        problem = cp.Problem(objective, constraints)
        solution2 = problem.solve(solver=cp.SCS, eps=1e-05)
        self.assertAlmostEqual(solution1, solution2)

    def test_rmul_scalar_mats(self) -> None:
        if False:
            print('Hello World!')
        'Test that rmul works with 1x1 matrices.\n        '
        x = [[4144.30127531]]
        y = [[7202.52114311]]
        z = cp.Variable(shape=(1, 1))
        objective = cp.Minimize(cp.quad_form(z, x) - 2 * z.T @ y)
        prob = cp.Problem(objective)
        prob.solve(cp.OSQP, verbose=True)
        result1 = prob.value
        x = 4144.30127531
        y = 7202.52114311
        z = cp.Variable()
        objective = cp.Minimize(x * z ** 2 - 2 * z * y)
        prob = cp.Problem(objective)
        prob.solve(cp.OSQP, verbose=True)
        self.assertAlmostEqual(prob.value, result1)

    def test_min_with_axis(self) -> None:
        if False:
            while True:
                i = 10
        'Test reshape of a min with axis=0.\n        '
        x = cp.Variable((5, 2))
        y = cp.Variable((5, 2))
        stacked_flattened = cp.vstack([cp.vec(x), cp.vec(y)])
        minimum = cp.min(stacked_flattened, axis=0)
        reshaped_minimum = cp.reshape(minimum, (5, 2))
        obj = cp.sum(reshaped_minimum)
        problem = cp.Problem(cp.Maximize(obj), [x == 1, y == 2])
        result = problem.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, 10)

    def test_constant_infeasible(self) -> None:
        if False:
            while True:
                i = 10
        'Test a problem with constant values only that is infeasible.\n        '
        p = cp.Problem(cp.Maximize(0), [cp.Constant(0) == 1])
        p.solve(solver=cp.SCS)
        self.assertEqual(p.status, cp.INFEASIBLE)

    def test_huber_scs(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test that huber regression works with SCS.\n           See issue #1370.\n        '
        np.random.seed(1)
        m = 5
        n = 2
        x0 = np.random.randn(n)
        A = np.random.randn(m, n)
        b = A.dot(x0) + 0.01 * np.random.randn(m)
        k = int(0.02 * m)
        idx = np.random.randint(m, size=k)
        b[idx] += 10 * np.random.randn(k)
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.huber(A @ x - b))))
        prob.solve(solver=cp.SCS)

    def test_rmul_param(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test a complex rmul expression with a parameter.\n           See issue #1555.\n        '
        b = cp.Variable((1,))
        param = cp.Parameter(1)
        constraints = []
        objective = cp.Minimize(2 * b @ param)
        prob = cp.Problem(objective, constraints)
        param.value = np.array([1])
        prob.solve()
        assert prob.value == -np.inf

    def test_cumsum_axis(self) -> None:
        if False:
            return 10
        'Test the cumsum axis bug with row or column matrix\n           See issue #1678\n        '
        n = 5
        x1 = cp.Variable((1, n))
        expr1 = cp.cumsum(x1, axis=0)
        prob1 = cp.Problem(cp.Minimize(0), [expr1 == 1])
        prob1.solve()
        expect = np.ones((1, n))
        self.assertItemsAlmostEqual(expr1.value, expect)
        x2 = cp.Variable((n, 1))
        expr2 = cp.cumsum(x2, axis=1)
        prob2 = cp.Problem(cp.Minimize(0), [expr2 == 1])
        prob2.solve()
        expect = np.ones((n, 1))
        self.assertItemsAlmostEqual(expr2.value, expect)

    def test_cummax_axis(self) -> None:
        if False:
            return 10
        'Test the cumsum axis bug with row or column matrix\n           See issue #1678\n        '
        n = 5
        x1 = cp.Variable((1, n))
        expr1 = cp.cummax(x1, axis=0)
        prob1 = cp.Problem(cp.Maximize(cp.sum(x1)), [expr1 <= 1])
        prob1.solve()
        expect = np.ones((1, n))
        self.assertItemsAlmostEqual(expr1.value, expect)
        x2 = cp.Variable((n, 1))
        expr2 = cp.cummax(x2, axis=1)
        prob2 = cp.Problem(cp.Maximize(cp.sum(x2)), [expr2 <= 1])
        prob2.solve()
        expect = np.ones((n, 1))
        self.assertItemsAlmostEqual(expr2.value, expect)

    def test_cp_node_count_warn(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that a warning is raised for high node count.'
        with warnings.catch_warnings(record=True) as w:
            a = cp.Variable(shape=(100, 100))
            b = sum((sum(x) for x in a))
            cp.Problem(cp.Maximize(0), [b >= 0])
            assert len(w) == 1
            assert 'vectorizing' in str(w[-1].message)
            assert 'Constraint #0' in str(w[-1].message)
        with warnings.catch_warnings(record=True) as w:
            a = cp.Variable(shape=(100, 100))
            b = sum((sum(x) for x in a))
            cp.Problem(cp.Maximize(b))
            assert len(w) == 1
            assert 'vectorizing' in str(w[-1].message)
            assert 'Objective' in str(w[-1].message)
        with warnings.catch_warnings(record=True) as w:
            a = cp.Variable(shape=(100, 100))
            c = cp.sum(a)
            cp.Problem(cp.Maximize(0), [c >= 0])
            assert len(w) == 0

    def test_ecos_warning(self) -> None:
        if False:
            while True:
                i = 10
        'Test that a warning is raised when ECOS\n           is called by default.\n        '
        x = cp.Variable()
        prob = cp.Problem(cp.Maximize(x), [x ** 2 <= 1])
        candidate_solvers = prob._find_candidate_solvers(solver=None, gp=False)
        prob._sort_candidate_solvers(candidate_solvers)
        if candidate_solvers['conic_solvers'][0] == cp.ECOS:
            with warnings.catch_warnings(record=True) as w:
                prob.solve()
                assert isinstance(w[0].message, FutureWarning)
                assert str(w[0].message) == ECOS_DEPRECATION_MSG
            with warnings.catch_warnings(record=True) as w:
                prob.solve(solver=cp.ECOS)
                assert len(w) == 0