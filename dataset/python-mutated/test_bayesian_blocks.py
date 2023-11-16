import numpy as np
import pytest
from numpy.testing import assert_allclose
from astropy.stats import RegularEvents, bayesian_blocks

def test_single_change_point(rseed=0):
    if False:
        return 10
    rng = np.random.default_rng(rseed)
    x = np.concatenate([rng.random(100), 1 + rng.random(200)])
    bins = bayesian_blocks(x)
    assert len(bins) == 3
    assert_allclose(bins[1], 0.927289, rtol=0.02)

def test_duplicate_events(rseed=0):
    if False:
        while True:
            i = 10
    rng = np.random.default_rng(rseed)
    t = rng.random(100)
    t[80:] = t[:20]
    x = np.ones(t.shape, dtype=int)
    x[:20] += 1
    bins1 = bayesian_blocks(t)
    bins2 = bayesian_blocks(t[:80], x[:80])
    assert_allclose(bins1, bins2)

def test_measures_fitness_homoscedastic(rseed=0):
    if False:
        print('Hello World!')
    rng = np.random.default_rng(rseed)
    t = np.linspace(0, 1, 11)
    x = np.exp(-0.5 * (t - 0.5) ** 2 / 0.01 ** 2)
    sigma = 0.05
    x = x + sigma * rng.standard_normal(len(x))
    bins = bayesian_blocks(t, x, sigma, fitness='measures')
    assert_allclose(bins, [0, 0.45, 0.55, 1])

def test_measures_fitness_heteroscedastic():
    if False:
        print('Hello World!')
    rng = np.random.default_rng(1)
    t = np.linspace(0, 1, 11)
    x = np.exp(-0.5 * (t - 0.5) ** 2 / 0.01 ** 2)
    sigma = 0.02 + 0.02 * rng.random(len(x))
    x = x + sigma * rng.standard_normal(len(x))
    bins = bayesian_blocks(t, x, sigma, fitness='measures')
    assert_allclose(bins, [0, 0.45, 0.55, 1])

def test_regular_events():
    if False:
        return 10
    rng = np.random.default_rng(1234)
    dt = 0.01
    steps = np.concatenate([np.unique(rng.integers(0, 500, 100)), np.unique(rng.integers(500, 1000, 200))])
    t = dt * steps
    bins1 = bayesian_blocks(t, fitness='regular_events', dt=dt)
    assert len(bins1) == 3
    assert_allclose(bins1[1], 5, rtol=0.05)
    bins2 = bayesian_blocks(t, fitness=RegularEvents, dt=dt)
    assert_allclose(bins1, bins2)
    bins3 = bayesian_blocks(t, fitness=RegularEvents(dt=dt))
    assert_allclose(bins1, bins3)

def test_errors():
    if False:
        return 10
    rng = np.random.default_rng(0)
    t = rng.random(100)
    with pytest.raises(ValueError):
        bayesian_blocks(t, fitness='events', x=t)
    with pytest.raises(ValueError):
        bayesian_blocks(t, fitness='regular_events', x=10 * t, dt=1)
    with pytest.raises(ValueError):
        bayesian_blocks(t, fitness='measures')
    with pytest.raises(ValueError):
        bayesian_blocks(t, fitness='events', sigma=0.5)
    with pytest.raises(ValueError):
        bayesian_blocks(t, fitness='measures', x=t[:-1])
    t2 = t.copy()
    t2[1] = t2[0]
    with pytest.raises(ValueError):
        bayesian_blocks(t2, fitness='measures', x=t)
    with pytest.raises(ValueError):
        bayesian_blocks(t, fitness='measures', x=t, sigma=t[:-1])

def test_fitness_function_results():
    if False:
        while True:
            i = 10
    'Test results for several fitness functions'
    rng = np.random.default_rng(42)
    t = rng.standard_normal(100)
    edges = bayesian_blocks(t, fitness='events')
    assert_allclose(edges, [-1.95103519, -1.01861547, 0.95442154, 2.1416476])
    t[80:] = t[:20]
    edges = bayesian_blocks(t, fitness='events', p0=0.01)
    assert_allclose(edges, [-1.95103519, -1.08663566, 1.17575682, 2.1416476])
    dt = 0.01
    t = dt * np.arange(1000)
    x = np.zeros(len(t))
    N = len(t) // 10
    x[rng.integers(0, len(t), N)] = 1
    x[rng.integers(0, len(t) // 2, N)] = 1
    edges = bayesian_blocks(t, x, fitness='regular_events', dt=dt)
    assert_allclose(edges, [0, 4.365, 4.995, 9.99])
    t = 100 * rng.random(20)
    x = np.exp(-0.5 * (t - 50) ** 2)
    sigma = 0.1
    x_obs = x + sigma * rng.standard_normal(len(x))
    edges = bayesian_blocks(t, x_obs, sigma, fitness='measures')
    expected = [1.39362877, 44.30811196, 49.46626158, 54.37232704, 92.7562551]
    assert_allclose(edges, expected)
    p0_sel = 0.05
    edges = bayesian_blocks(t, x_obs, sigma, fitness='measures', p0=p0_sel)
    assert_allclose(edges, expected)
    ncp_prior_sel = 4 - np.log(73.53 * p0_sel * len(t) ** (-0.478))
    edges = bayesian_blocks(t, x_obs, sigma, fitness='measures', ncp_prior=ncp_prior_sel)
    assert_allclose(edges, expected)
    gamma_sel = np.exp(-ncp_prior_sel)
    edges = bayesian_blocks(t, x_obs, sigma, fitness='measures', gamma=gamma_sel)
    assert_allclose(edges, expected)

def test_zero_change_points(rseed=0):
    if False:
        while True:
            i = 10
    '\n    Ensure that edges contains both endpoints when there are no change points\n    '
    np.random.seed(rseed)
    values = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2])
    bins = bayesian_blocks(values)
    assert values.min() == bins[0]
    assert values.max() == bins[-1]