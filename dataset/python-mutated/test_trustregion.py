"""
Unit tests for trust-region optimization routines.

To run it in its simplest form::
  nosetests test_optimize.py

"""
import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_allclose
from scipy.optimize import minimize, rosen, rosen_der, rosen_hess, rosen_hess_prod

class Accumulator:
    """ This is for testing callbacks."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.count = 0
        self.accum = None

    def __call__(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.count += 1
        if self.accum is None:
            self.accum = np.array(x)
        else:
            self.accum += x

class TestTrustRegionSolvers:

    def setup_method(self):
        if False:
            print('Hello World!')
        self.x_opt = [1.0, 1.0]
        self.easy_guess = [2.0, 2.0]
        self.hard_guess = [-1.2, 1.0]

    def test_dogleg_accuracy(self):
        if False:
            return 10
        x0 = self.hard_guess
        r = minimize(rosen, x0, jac=rosen_der, hess=rosen_hess, tol=1e-08, method='dogleg', options={'return_all': True})
        assert_allclose(x0, r['allvecs'][0])
        assert_allclose(r['x'], r['allvecs'][-1])
        assert_allclose(r['x'], self.x_opt)

    def test_dogleg_callback(self):
        if False:
            while True:
                i = 10
        accumulator = Accumulator()
        maxiter = 5
        r = minimize(rosen, self.hard_guess, jac=rosen_der, hess=rosen_hess, callback=accumulator, method='dogleg', options={'return_all': True, 'maxiter': maxiter})
        assert_equal(accumulator.count, maxiter)
        assert_equal(len(r['allvecs']), maxiter + 1)
        assert_allclose(r['x'], r['allvecs'][-1])
        assert_allclose(sum(r['allvecs'][1:]), accumulator.accum)

    def test_dogleg_user_warning(self):
        if False:
            i = 10
            return i + 15
        with pytest.warns(RuntimeWarning, match='Maximum number of iterations'):
            minimize(rosen, self.hard_guess, jac=rosen_der, hess=rosen_hess, method='dogleg', options={'disp': True, 'maxiter': 1})

    def test_solver_concordance(self):
        if False:
            while True:
                i = 10
        f = rosen
        g = rosen_der
        h = rosen_hess
        for x0 in (self.easy_guess, self.hard_guess):
            r_dogleg = minimize(f, x0, jac=g, hess=h, tol=1e-08, method='dogleg', options={'return_all': True})
            r_trust_ncg = minimize(f, x0, jac=g, hess=h, tol=1e-08, method='trust-ncg', options={'return_all': True})
            r_trust_krylov = minimize(f, x0, jac=g, hess=h, tol=1e-08, method='trust-krylov', options={'return_all': True})
            r_ncg = minimize(f, x0, jac=g, hess=h, tol=1e-08, method='newton-cg', options={'return_all': True})
            r_iterative = minimize(f, x0, jac=g, hess=h, tol=1e-08, method='trust-exact', options={'return_all': True})
            assert_allclose(self.x_opt, r_dogleg['x'])
            assert_allclose(self.x_opt, r_trust_ncg['x'])
            assert_allclose(self.x_opt, r_trust_krylov['x'])
            assert_allclose(self.x_opt, r_ncg['x'])
            assert_allclose(self.x_opt, r_iterative['x'])
            assert_(len(r_dogleg['allvecs']) < len(r_ncg['allvecs']))

    def test_trust_ncg_hessp(self):
        if False:
            return 10
        for x0 in (self.easy_guess, self.hard_guess, self.x_opt):
            r = minimize(rosen, x0, jac=rosen_der, hessp=rosen_hess_prod, tol=1e-08, method='trust-ncg')
            assert_allclose(self.x_opt, r['x'])

    def test_trust_ncg_start_in_optimum(self):
        if False:
            while True:
                i = 10
        r = minimize(rosen, x0=self.x_opt, jac=rosen_der, hess=rosen_hess, tol=1e-08, method='trust-ncg')
        assert_allclose(self.x_opt, r['x'])

    def test_trust_krylov_start_in_optimum(self):
        if False:
            for i in range(10):
                print('nop')
        r = minimize(rosen, x0=self.x_opt, jac=rosen_der, hess=rosen_hess, tol=1e-08, method='trust-krylov')
        assert_allclose(self.x_opt, r['x'])

    def test_trust_exact_start_in_optimum(self):
        if False:
            return 10
        r = minimize(rosen, x0=self.x_opt, jac=rosen_der, hess=rosen_hess, tol=1e-08, method='trust-exact')
        assert_allclose(self.x_opt, r['x'])