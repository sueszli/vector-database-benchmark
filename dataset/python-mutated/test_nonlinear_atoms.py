"""
Copyright 2013 Steven Diamond, Eric Chu

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
import math
import numpy as np
import cvxpy as cvx
from cvxpy.tests.base_test import BaseTest

class TestNonlinearAtoms(BaseTest):
    """ Unit tests for the nonlinear atoms module. """

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        self.x = cvx.Variable(2, name='x')
        self.y = cvx.Variable(2, name='y')
        self.A = cvx.Variable((2, 2), name='A')
        self.B = cvx.Variable((2, 2), name='B')
        self.C = cvx.Variable((3, 2), name='C')

    def test_log_problem(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        obj = cvx.Maximize(cvx.sum(cvx.log(self.x)))
        constr = [self.x <= [1, math.e]]
        p = cvx.Problem(obj, constr)
        result = p.solve(solver=cvx.ECOS)
        self.assertAlmostEqual(result, 1)
        self.assertItemsAlmostEqual(self.x.value, [1, math.e])
        obj = cvx.Minimize(cvx.sum(self.x))
        constr = [cvx.log(self.x) >= 0, self.x <= [1, 1]]
        p = cvx.Problem(obj, constr)
        result = p.solve(solver=cvx.ECOS)
        self.assertAlmostEqual(result, 2)
        self.assertItemsAlmostEqual(self.x.value, [1, 1])
        obj = cvx.Maximize(cvx.log(self.x)[1])
        constr = [self.x <= [1, math.e]]
        p = cvx.Problem(obj, constr)
        result = p.solve(solver=cvx.ECOS)
        self.assertAlmostEqual(result, 1)
        obj = cvx.Maximize(cvx.log(self.x[1]))
        constr = [self.x <= [1, math.e]]
        p = cvx.Problem(obj, constr)
        result = p.solve(solver=cvx.ECOS)
        self.assertAlmostEqual(result, 1)

    def test_entr(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test the entr atom.\n        '
        self.assertEqual(cvx.entr(0).value, 0)
        assert np.isneginf(cvx.entr(-1).value)

    def test_kl_div(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test a problem with kl_div.\n        '
        kK = 50
        kSeed = 10
        prng = np.random.RandomState(kSeed)
        npSPriors = prng.uniform(0.0, 1.0, kK)
        npSPriors = npSPriors / sum(npSPriors)
        p_refProb = cvx.Parameter(kK, nonneg=True)
        v_prob = cvx.Variable(kK)
        objkl = cvx.sum(cvx.kl_div(v_prob, p_refProb))
        constrs = [cvx.sum(v_prob) == 1]
        klprob = cvx.Problem(cvx.Minimize(objkl), constrs)
        p_refProb.value = npSPriors
        klprob.solve(solver=cvx.SCS, verbose=True)
        self.assertItemsAlmostEqual(v_prob.value, npSPriors, places=3)
        klprob.solve(solver=cvx.ECOS, verbose=True)
        self.assertItemsAlmostEqual(v_prob.value, npSPriors)

    def test_rel_entr(self) -> None:
        if False:
            while True:
                i = 10
        'Test a problem with rel_entr.\n        '
        kK = 50
        kSeed = 10
        prng = np.random.RandomState(kSeed)
        npSPriors = prng.uniform(0.0, 1.0, kK)
        npSPriors = npSPriors / sum(npSPriors)
        p_refProb = cvx.Parameter(kK, nonneg=True)
        v_prob = cvx.Variable(kK)
        obj_rel_entr = cvx.sum(cvx.rel_entr(v_prob, p_refProb))
        constrs = [cvx.sum(v_prob) == 1]
        rel_entr_prob = cvx.Problem(cvx.Minimize(obj_rel_entr), constrs)
        p_refProb.value = npSPriors
        rel_entr_prob.solve(solver=cvx.SCS, verbose=True)
        self.assertItemsAlmostEqual(v_prob.value, npSPriors, places=3)
        rel_entr_prob.solve(solver=cvx.ECOS, verbose=True)
        self.assertItemsAlmostEqual(v_prob.value, npSPriors)

    def test_difference_kl_div_rel_entr(self) -> None:
        if False:
            while True:
                i = 10
        'A test showing the difference between kl_div and rel_entr\n        '
        x = cvx.Variable()
        y = cvx.Variable()
        kl_div_prob = cvx.Problem(cvx.Minimize(cvx.kl_div(x, y)), constraints=[x + y <= 1])
        kl_div_prob.solve(solver=cvx.ECOS)
        self.assertItemsAlmostEqual(x.value, y.value)
        self.assertItemsAlmostEqual(kl_div_prob.value, 0)
        rel_entr_prob = cvx.Problem(cvx.Minimize(cvx.rel_entr(x, y)), constraints=[x + y <= 1])
        rel_entr_prob.solve(solver=cvx.ECOS)
        '\n        Reference solution computed by passing the following command to Wolfram Alpha:\n        minimize x*log(x/y) subject to {x + y <= 1, 0 <= x, 0 <= y}\n        '
        self.assertItemsAlmostEqual(x.value, 0.2178117, places=4)
        self.assertItemsAlmostEqual(y.value, 0.7821882, places=4)
        self.assertItemsAlmostEqual(rel_entr_prob.value, -0.278464)

    def test_entr_prob(self) -> None:
        if False:
            print('Hello World!')
        'Test a problem with entr.\n        '
        for n in [5, 10, 25]:
            print(n)
            x = cvx.Variable(n)
            obj = cvx.Maximize(cvx.sum(cvx.entr(x)))
            p = cvx.Problem(obj, [cvx.sum(x) == 1])
            p.solve(solver=cvx.ECOS, verbose=True)
            self.assertItemsAlmostEqual(x.value, n * [1.0 / n])
            p.solve(solver=cvx.SCS, verbose=True)
            self.assertItemsAlmostEqual(x.value, n * [1.0 / n], places=3)

    def test_exp(self) -> None:
        if False:
            while True:
                i = 10
        'Test a problem with exp.\n        '
        for n in [5, 10, 25]:
            print(n)
            x = cvx.Variable(n)
            obj = cvx.Minimize(cvx.sum(cvx.exp(x)))
            p = cvx.Problem(obj, [cvx.sum(x) == 1])
            p.solve(solver=cvx.SCS, verbose=True)
            self.assertItemsAlmostEqual(x.value, n * [1.0 / n], places=3)
            p.solve(solver=cvx.ECOS, verbose=True)
            self.assertItemsAlmostEqual(x.value, n * [1.0 / n])

    def test_log(self) -> None:
        if False:
            print('Hello World!')
        'Test a problem with log.\n        '
        for n in [5, 10, 25]:
            print(n)
            x = cvx.Variable(n)
            obj = cvx.Maximize(cvx.sum(cvx.log(x)))
            p = cvx.Problem(obj, [cvx.sum(x) == 1])
            p.solve(solver=cvx.ECOS, verbose=True)
            self.assertItemsAlmostEqual(x.value, n * [1.0 / n])
            p.solve(solver=cvx.SCS, verbose=True)
            self.assertItemsAlmostEqual(x.value, n * [1.0 / n], places=2)