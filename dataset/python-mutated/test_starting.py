import re
import numpy as np
import pytest
from numpy.testing import assert_allclose
import pymc as pm
from pymc.exceptions import ImputationWarning
from pymc.step_methods.metropolis import tune
from pymc.testing import select_by_precision
from pymc.tuning import find_MAP
from tests import models
from tests.models import non_normal, simple_arbitrary_det, simple_model

@pytest.mark.parametrize('bounded', [False, True])
def test_mle_jacobian(bounded):
    if False:
        return 10
    'Test MAP / MLE estimation for distributions with flat priors.'
    truth = 10.0
    rtol = 0.0001
    (start, model, _) = models.simple_normal(bounded_prior=bounded)
    with model:
        map_estimate = find_MAP(method='BFGS', model=model)
    assert_allclose(map_estimate['mu_i'], truth, rtol=rtol)

def test_tune_not_inplace():
    if False:
        i = 10
        return i + 15
    orig_scaling = np.array([0.001, 0.1])
    returned_scaling = tune(orig_scaling, acc_rate=0.6)
    assert returned_scaling is not orig_scaling
    assert np.all(orig_scaling == np.array([0.001, 0.1]))

def test_accuracy_normal():
    if False:
        i = 10
        return i + 15
    (_, model, (mu, _)) = simple_model()
    with model:
        newstart = find_MAP(pm.Point(x=[-10.5, 100.5]))
        assert_allclose(newstart['x'], [mu, mu], atol=select_by_precision(float64=1e-05, float32=0.0001))

def test_accuracy_non_normal():
    if False:
        i = 10
        return i + 15
    (_, model, (mu, _)) = non_normal(4)
    with model:
        newstart = find_MAP(pm.Point(x=[0.5, 0.01, 0.95, 0.99]))
        assert_allclose(newstart['x'], mu, atol=select_by_precision(float64=1e-05, float32=0.0001))

def test_find_MAP_discrete():
    if False:
        print('Hello World!')
    tol1 = 2.0 ** (-11)
    tol2 = 2.0 ** (-6)
    alpha = 4
    beta = 4
    n = 20
    yes = 15
    with pm.Model() as model:
        p = pm.Beta('p', alpha, beta)
        pm.Binomial('ss', n=n, p=p)
        pm.Binomial('s', n=n, p=p, observed=yes)
        map_est1 = find_MAP()
        map_est2 = find_MAP(vars=model.value_vars)
    assert_allclose(map_est1['p'], 0.6086956533498806, atol=tol1, rtol=0)
    assert_allclose(map_est2['p'], 0.695642178810167, atol=tol2, rtol=0)
    assert map_est2['ss'] == 14

def test_find_MAP_no_gradient():
    if False:
        i = 10
        return i + 15
    (_, model) = simple_arbitrary_det()
    with model:
        find_MAP()

def test_find_MAP():
    if False:
        return 10
    tol = 2.0 ** (-11)
    data = np.random.randn(100)
    data = (data - np.mean(data)) / np.std(data)
    with pm.Model():
        mu = pm.Uniform('mu', -1, 1)
        sigma = pm.Uniform('sigma', 0.5, 1.5)
        pm.Normal('y', mu=mu, tau=sigma ** (-2), observed=data)
        map_est1 = find_MAP(progressbar=False)
        map_est2 = find_MAP(progressbar=False, method='Powell')
    assert_allclose(map_est1['mu'], 0, atol=tol)
    assert_allclose(map_est1['sigma'], 1, atol=tol)
    assert_allclose(map_est2['mu'], 0, atol=tol)
    assert_allclose(map_est2['sigma'], 1, atol=tol)

def test_find_MAP_issue_5923():
    if False:
        i = 10
        return i + 15
    tol = 2.0 ** (-11)
    data = np.random.randn(100)
    data = (data - np.mean(data)) / np.std(data)
    with pm.Model():
        mu = pm.Uniform('mu', -1, 1)
        sigma = pm.Uniform('sigma', 0.5, 1.5)
        pm.Normal('y', mu=mu, tau=sigma ** (-2), observed=data)
        start = {'mu': -0.5, 'sigma': 1.25}
        map_est1 = find_MAP(progressbar=False, vars=[mu, sigma], start=start)
        map_est2 = find_MAP(progressbar=False, vars=[sigma, mu], start=start)
    assert_allclose(map_est1['mu'], 0, atol=tol)
    assert_allclose(map_est1['sigma'], 1, atol=tol)
    assert_allclose(map_est2['mu'], 0, atol=tol)
    assert_allclose(map_est2['sigma'], 1, atol=tol)

def test_find_MAP_issue_4488():
    if False:
        for i in range(10):
            print('nop')
    with pm.Model() as m:
        with pytest.warns(ImputationWarning):
            x = pm.Gamma('x', alpha=3, beta=10, observed=np.array([1, np.nan]))
        y = pm.Deterministic('y', x + 1)
        map_estimate = find_MAP()
    assert not set.difference({'x_unobserved', 'x_unobserved_log__', 'y'}, set(map_estimate.keys()))
    assert_allclose(map_estimate['x_unobserved'], 0.2, rtol=0.0001, atol=0.0001)
    assert_allclose(map_estimate['y'], [2.0, map_estimate['x_unobserved'][0] + 1])

def test_find_MAP_warning_non_free_RVs():
    if False:
        while True:
            i = 10
    with pm.Model() as m:
        x = pm.Normal('x')
        y = pm.Normal('y')
        det = pm.Deterministic('det', x + y)
        pm.Normal('z', det, 1e-05, observed=100)
        msg = 'Intermediate variables (such as Deterministic or Potential) were passed'
        with pytest.warns(UserWarning, match=re.escape(msg)):
            r = pm.find_MAP(vars=[det])
        assert_allclose([r['x'], r['y'], r['det']], [50, 50, 100])