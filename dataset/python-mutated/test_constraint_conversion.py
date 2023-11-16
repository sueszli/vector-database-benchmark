"""
Unit test for constraint conversion
"""
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose, assert_warns, suppress_warnings
import pytest
from scipy.optimize import NonlinearConstraint, LinearConstraint, OptimizeWarning, minimize, BFGS
from .test_minimize_constrained import Maratos, HyperbolicIneq, Rosenbrock, IneqRosenbrock, EqIneqRosenbrock, BoundedRosenbrock, Elec

class TestOldToNew:
    x0 = (2, 0)
    bnds = ((0, None), (0, None))
    method = 'trust-constr'

    def test_constraint_dictionary_1(self):
        if False:
            i = 10
            return i + 15

        def fun(x):
            if False:
                return 10
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2
        cons = ({'type': 'ineq', 'fun': lambda x: x[0] - 2 * x[1] + 2}, {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6}, {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'delta_grad == 0.0')
            res = minimize(fun, self.x0, method=self.method, bounds=self.bnds, constraints=cons)
        assert_allclose(res.x, [1.4, 1.7], rtol=0.0001)
        assert_allclose(res.fun, 0.8, rtol=0.0001)

    def test_constraint_dictionary_2(self):
        if False:
            for i in range(10):
                print('nop')

        def fun(x):
            if False:
                return 10
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2
        cons = {'type': 'eq', 'fun': lambda x, p1, p2: p1 * x[0] - p2 * x[1], 'args': (1, 1.1), 'jac': lambda x, p1, p2: np.array([[p1, -p2]])}
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'delta_grad == 0.0')
            res = minimize(fun, self.x0, method=self.method, bounds=self.bnds, constraints=cons)
        assert_allclose(res.x, [1.7918552, 1.62895927])
        assert_allclose(res.fun, 1.3857466063348418)

    def test_constraint_dictionary_3(self):
        if False:
            while True:
                i = 10

        def fun(x):
            if False:
                for i in range(10):
                    print('nop')
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2
        cons = [{'type': 'ineq', 'fun': lambda x: x[0] - 2 * x[1] + 2}, NonlinearConstraint(lambda x: x[0] - x[1], 0, 0)]
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'delta_grad == 0.0')
            res = minimize(fun, self.x0, method=self.method, bounds=self.bnds, constraints=cons)
        assert_allclose(res.x, [1.75, 1.75], rtol=0.0001)
        assert_allclose(res.fun, 1.125, rtol=0.0001)

class TestNewToOld:

    def test_multiple_constraint_objects(self):
        if False:
            for i in range(10):
                print('nop')

        def fun(x):
            if False:
                return 10
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2 + (x[2] - 0.75) ** 2
        x0 = [2, 0, 1]
        coni = []
        methods = ['slsqp', 'cobyla', 'trust-constr']
        coni.append([{'type': 'ineq', 'fun': lambda x: x[0] - 2 * x[1] + 2}, NonlinearConstraint(lambda x: x[0] - x[1], -1, 1)])
        coni.append([LinearConstraint([1, -2, 0], -2, np.inf), NonlinearConstraint(lambda x: x[0] - x[1], -1, 1)])
        coni.append([NonlinearConstraint(lambda x: x[0] - 2 * x[1] + 2, 0, np.inf), NonlinearConstraint(lambda x: x[0] - x[1], -1, 1)])
        for con in coni:
            funs = {}
            for method in methods:
                with suppress_warnings() as sup:
                    sup.filter(UserWarning)
                    result = minimize(fun, x0, method=method, constraints=con)
                    funs[method] = result.fun
            assert_allclose(funs['slsqp'], funs['trust-constr'], rtol=0.0001)
            assert_allclose(funs['cobyla'], funs['trust-constr'], rtol=0.0001)

    def test_individual_constraint_objects(self):
        if False:
            for i in range(10):
                print('nop')

        def fun(x):
            if False:
                i = 10
                return i + 15
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2 + (x[2] - 0.75) ** 2
        x0 = [2, 0, 1]
        cone = []
        coni = []
        methods = ['slsqp', 'cobyla', 'trust-constr']
        cone.append(NonlinearConstraint(lambda x: x[0] - x[1], 1, 1))
        cone.append(NonlinearConstraint(lambda x: x[0] - x[1], [1.21], [1.21]))
        cone.append(NonlinearConstraint(lambda x: x[0] - x[1], 1.21, np.array([1.21])))
        cone.append(NonlinearConstraint(lambda x: [x[0] - x[1], x[1] - x[2]], 1.21, 1.21))
        cone.append(NonlinearConstraint(lambda x: [x[0] - x[1], x[1] - x[2]], [1.21, 1.4], [1.21, 1.4]))
        cone.append(NonlinearConstraint(lambda x: [x[0] - x[1], x[1] - x[2]], [1.21, 1.21], 1.21))
        cone.append(NonlinearConstraint(lambda x: [x[0] - x[1], x[1] - x[2]], [1.21, -np.inf], [1.21, np.inf]))
        coni.append(NonlinearConstraint(lambda x: x[0] - x[1], 1.21, np.inf))
        coni.append(NonlinearConstraint(lambda x: x[0] - x[1], [1.21], np.inf))
        coni.append(NonlinearConstraint(lambda x: x[0] - x[1], 1.21, np.array([np.inf])))
        coni.append(NonlinearConstraint(lambda x: x[0] - x[1], -np.inf, -3))
        coni.append(NonlinearConstraint(lambda x: x[0] - x[1], np.array(-np.inf), -3))
        coni.append(NonlinearConstraint(lambda x: [x[0] - x[1], x[1] - x[2]], 1.21, np.inf))
        cone.append(NonlinearConstraint(lambda x: [x[0] - x[1], x[1] - x[2]], [1.21, -np.inf], [1.21, 1.4]))
        coni.append(NonlinearConstraint(lambda x: [x[0] - x[1], x[1] - x[2]], [1.1, 0.8], [1.2, 1.4]))
        coni.append(NonlinearConstraint(lambda x: [x[0] - x[1], x[1] - x[2]], [-1.2, -1.4], [-1.1, -0.8]))
        cone.append(LinearConstraint([1, -1, 0], 1.21, 1.21))
        cone.append(LinearConstraint([[1, -1, 0], [0, 1, -1]], 1.21, 1.21))
        cone.append(LinearConstraint([[1, -1, 0], [0, 1, -1]], [1.21, -np.inf], [1.21, 1.4]))
        for con in coni:
            funs = {}
            for method in methods:
                with suppress_warnings() as sup:
                    sup.filter(UserWarning)
                    result = minimize(fun, x0, method=method, constraints=con)
                    funs[method] = result.fun
            assert_allclose(funs['slsqp'], funs['trust-constr'], rtol=0.001)
            assert_allclose(funs['cobyla'], funs['trust-constr'], rtol=0.001)
        for con in cone:
            funs = {}
            for method in methods[::2]:
                with suppress_warnings() as sup:
                    sup.filter(UserWarning)
                    result = minimize(fun, x0, method=method, constraints=con)
                    funs[method] = result.fun
            assert_allclose(funs['slsqp'], funs['trust-constr'], rtol=0.001)

class TestNewToOldSLSQP:
    method = 'slsqp'
    elec = Elec(n_electrons=2)
    elec.x_opt = np.array([-0.58438468, 0.58438466, 0.73597047, -0.73597044, 0.34180668, -0.34180667])
    brock = BoundedRosenbrock()
    brock.x_opt = [0, 0]
    list_of_problems = [Maratos(), HyperbolicIneq(), Rosenbrock(), IneqRosenbrock(), EqIneqRosenbrock(), elec, brock]

    def test_list_of_problems(self):
        if False:
            i = 10
            return i + 15
        for prob in self.list_of_problems:
            with suppress_warnings() as sup:
                sup.filter(UserWarning)
                result = minimize(prob.fun, prob.x0, method=self.method, bounds=prob.bounds, constraints=prob.constr)
            assert_array_almost_equal(result.x, prob.x_opt, decimal=3)

    def test_warn_mixed_constraints(self):
        if False:
            print('Hello World!')

        def fun(x):
            if False:
                print('Hello World!')
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2 + (x[2] - 0.75) ** 2
        cons = NonlinearConstraint(lambda x: [x[0] ** 2 - x[1], x[1] - x[2]], [1.1, 0.8], [1.1, 1.4])
        bnds = ((0, None), (0, None), (0, None))
        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'delta_grad == 0.0')
            assert_warns(OptimizeWarning, minimize, fun, (2, 0, 1), method=self.method, bounds=bnds, constraints=cons)

    def test_warn_ignored_options(self):
        if False:
            while True:
                i = 10

        def fun(x):
            if False:
                i = 10
                return i + 15
            return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2 + (x[2] - 0.75) ** 2
        x0 = (2, 0, 1)
        if self.method == 'slsqp':
            bnds = ((0, None), (0, None), (0, None))
        else:
            bnds = None
        cons = NonlinearConstraint(lambda x: x[0], 2, np.inf)
        res = minimize(fun, x0, method=self.method, bounds=bnds, constraints=cons)
        assert_allclose(res.fun, 1)
        cons = LinearConstraint([1, 0, 0], 2, np.inf)
        res = minimize(fun, x0, method=self.method, bounds=bnds, constraints=cons)
        assert_allclose(res.fun, 1)
        cons = []
        cons.append(NonlinearConstraint(lambda x: x[0] ** 2, 2, np.inf, keep_feasible=True))
        cons.append(NonlinearConstraint(lambda x: x[0] ** 2, 2, np.inf, hess=BFGS()))
        cons.append(NonlinearConstraint(lambda x: x[0] ** 2, 2, np.inf, finite_diff_jac_sparsity=42))
        cons.append(NonlinearConstraint(lambda x: x[0] ** 2, 2, np.inf, finite_diff_rel_step=42))
        cons.append(LinearConstraint([1, 0, 0], 2, np.inf, keep_feasible=True))
        for con in cons:
            assert_warns(OptimizeWarning, minimize, fun, x0, method=self.method, bounds=bnds, constraints=cons)

class TestNewToOldCobyla:
    method = 'cobyla'
    list_of_problems = [Elec(n_electrons=2), Elec(n_electrons=4)]

    @pytest.mark.slow
    def test_list_of_problems(self):
        if False:
            while True:
                i = 10
        for prob in self.list_of_problems:
            with suppress_warnings() as sup:
                sup.filter(UserWarning)
                truth = minimize(prob.fun, prob.x0, method='trust-constr', bounds=prob.bounds, constraints=prob.constr)
                result = minimize(prob.fun, prob.x0, method=self.method, bounds=prob.bounds, constraints=prob.constr)
            assert_allclose(result.fun, truth.fun, rtol=0.001)