import warnings
import pytest
import torch
from pytest import approx
from torch.distributions import constraints
from pyro.generic import distributions as dist
from pyro.generic import infer, ops, optim, pyro, pyro_backend
from tests.common import assert_close, xfail_param

def build_svi(model, guide, elbo):
    if False:
        while True:
            i = 10
    pyro.get_param_store().clear()
    adam = optim.Adam({'lr': 1e-06})
    return infer.SVI(model, guide, adam, elbo)

def assert_ok(model, guide, elbo, steps=2, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Assert that inference works without warnings or errors.\n    '
    inference = build_svi(model, guide, elbo)
    for i in range(steps):
        inference.step(*args, **kwargs)

def assert_error(model, guide, elbo, match=None):
    if False:
        print('Hello World!')
    '\n    Assert that inference fails with an error.\n    '
    inference = build_svi(model, guide, elbo)
    with pytest.raises((NotImplementedError, UserWarning, KeyError, ValueError, RuntimeError), match=match):
        inference.step()

def assert_warning(model, guide, elbo):
    if False:
        print('Hello World!')
    '\n    Assert that inference works but with a warning.\n    '
    inference = build_svi(model, guide, elbo)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        inference.step()
        assert len(w), 'No warnings were raised'
        for warning in w:
            print(warning)

def constrained_model(data):
    if False:
        i = 10
        return i + 15
    locs = pyro.param('locs', torch.randn(3), constraint=constraints.real)
    scales = pyro.param('scales', ops.exp(torch.randn(3)), constraint=constraints.positive)
    p = torch.tensor([0.5, 0.3, 0.2])
    x = pyro.sample('x', dist.Categorical(p))
    pyro.sample('obs', dist.Normal(locs[x], scales[x]), obs=data)

def guide_constrained_model(data):
    if False:
        return 10
    q = pyro.param('q', ops.exp(torch.randn(3)), constraint=constraints.simplex)
    pyro.sample('x', dist.Categorical(q))

@pytest.mark.parametrize('backend', ['pyro', 'minipyro'])
def test_generate_data(backend):
    if False:
        while True:
            i = 10

    def model(data=None):
        if False:
            return 10
        loc = pyro.param('loc', torch.tensor(2.0))
        scale = pyro.param('scale', torch.tensor(1.0))
        x = pyro.sample('x', dist.Normal(loc, scale), obs=data)
        return x
    with pyro_backend(backend):
        data = model().data
        assert data.shape == ()

@pytest.mark.parametrize('backend', ['pyro', 'minipyro'])
def test_generate_data_plate(backend):
    if False:
        return 10
    num_points = 1000
    expected_mean = 2.0

    def model(data=None):
        if False:
            while True:
                i = 10
        loc = pyro.param('loc', torch.tensor(expected_mean))
        scale = pyro.param('scale', torch.tensor(1.0))
        with pyro.plate('data', 1000, dim=-1):
            x = pyro.sample('x', dist.Normal(loc, scale), obs=data)
        return x
    with pyro_backend(backend):
        data = model().data
        assert data.shape == (num_points,)
        mean = float(ops.sum(data)) / num_points
        assert mean == approx(expected_mean, abs=0.1)

@pytest.mark.parametrize('jit', [False, True], ids=['py', 'jit'])
@pytest.mark.parametrize('backend', ['pyro', 'minipyro'])
def test_nonempty_model_empty_guide_ok(backend, jit):
    if False:
        while True:
            i = 10

    def model(data):
        if False:
            print('Hello World!')
        loc = pyro.param('loc', torch.tensor(0.0))
        pyro.sample('x', dist.Normal(loc, 1.0), obs=data)

    def guide(data):
        if False:
            for i in range(10):
                print('nop')
        pass
    data = torch.tensor(2.0)
    with pyro_backend(backend):
        Elbo = infer.JitTrace_ELBO if jit else infer.Trace_ELBO
        elbo = Elbo(ignore_jit_warnings=True)
        steps = 2
        assert_ok(model, guide, elbo, steps, data)

@pytest.mark.parametrize('jit', [False, True], ids=['py', 'jit'])
@pytest.mark.parametrize('backend', ['pyro', 'minipyro'])
def test_plate_ok(backend, jit):
    if False:
        print('Hello World!')
    data = torch.randn(10)

    def model():
        if False:
            return 10
        locs = pyro.param('locs', torch.tensor([0.2, 0.3, 0.5]))
        p = torch.tensor([0.2, 0.3, 0.5])
        with pyro.plate('plate', len(data), dim=-1):
            x = pyro.sample('x', dist.Categorical(p))
            pyro.sample('obs', dist.Normal(locs[x], 1.0), obs=data)

    def guide():
        if False:
            i = 10
            return i + 15
        p = pyro.param('p', torch.tensor([0.5, 0.3, 0.2]))
        with pyro.plate('plate', len(data), dim=-1):
            pyro.sample('x', dist.Categorical(p))
    with pyro_backend(backend):
        elbo_factory = infer.JitTrace_ELBO if jit else infer.Trace_ELBO
        elbo = elbo_factory(ignore_jit_warnings=True)
        assert_ok(model, guide, elbo)

@pytest.mark.parametrize('jit', [False, True], ids=['py', 'jit'])
@pytest.mark.parametrize('backend', ['pyro', 'minipyro'])
def test_nested_plate_plate_ok(backend, jit):
    if False:
        for i in range(10):
            print('nop')
    data = torch.randn(2, 3)

    def model():
        if False:
            print('Hello World!')
        loc = torch.tensor(3.0)
        with pyro.plate('plate_outer', data.size(-1), dim=-1):
            x = pyro.sample('x', dist.Normal(loc, 1.0))
            with pyro.plate('plate_inner', data.size(-2), dim=-2):
                pyro.sample('y', dist.Normal(x, 1.0), obs=data)

    def guide():
        if False:
            return 10
        loc = pyro.param('loc', torch.tensor(0.0))
        scale = pyro.param('scale', torch.tensor(1.0))
        with pyro.plate('plate_outer', data.size(-1), dim=-1):
            pyro.sample('x', dist.Normal(loc, scale))
    with pyro_backend(backend):
        Elbo = infer.JitTrace_ELBO if jit else infer.Trace_ELBO
        elbo = Elbo(ignore_jit_warnings=True)
        assert_ok(model, guide, elbo)

@pytest.mark.parametrize('jit', [False, True], ids=['py', 'jit'])
@pytest.mark.parametrize('backend', ['pyro', xfail_param('minipyro', reason='not implemented')])
def test_local_param_ok(backend, jit):
    if False:
        while True:
            i = 10
    data = torch.randn(10)

    def model():
        if False:
            print('Hello World!')
        locs = pyro.param('locs', torch.tensor([-1.0, 0.0, 1.0]))
        with pyro.plate('plate', len(data), dim=-1):
            x = pyro.sample('x', dist.Categorical(torch.ones(3) / 3))
            pyro.sample('obs', dist.Normal(locs[x], 1.0), obs=data)

    def guide():
        if False:
            while True:
                i = 10
        with pyro.plate('plate', len(data), dim=-1):
            p = pyro.param('p', torch.ones(len(data), 3) / 3, event_dim=1)
            pyro.sample('x', dist.Categorical(p))
        return p
    with pyro_backend(backend):
        Elbo = infer.JitTrace_ELBO if jit else infer.Trace_ELBO
        elbo = Elbo(ignore_jit_warnings=True)
        assert_ok(model, guide, elbo)
        expected = guide()
        actual = pyro.param('p')
        assert_close(actual, expected)

@pytest.mark.parametrize('jit', [False, True], ids=['py', 'jit'])
@pytest.mark.parametrize('backend', ['pyro', 'minipyro'])
def test_constraints(backend, jit):
    if False:
        i = 10
        return i + 15
    data = torch.tensor(0.5)
    with pyro_backend(backend):
        Elbo = infer.JitTrace_ELBO if jit else infer.Trace_ELBO
        elbo = Elbo(ignore_jit_warnings=True)
        steps = 2
        assert_ok(constrained_model, guide_constrained_model, elbo, steps, data)

@pytest.mark.parametrize('backend', ['pyro', 'minipyro'])
def test_elbo_jit(backend):
    if False:
        i = 10
        return i + 15
    '\n    Given model and guide\n    After 15 iterations of\n    SVI, JitTrace_ELBO yield\n    similar results for both pyro and minipyro.\n    This is a relaxed assertion over\n    the one reached in test_elbo_equivalence\n    Note: 15 is the minimun number for iterations for this\n    to happen.\n    '
    pyro.set_rng_seed(0)
    data = torch.tensor(0.5)
    elbo_test_case(backend, jit=True, expected_elbo=0.478, data=data, steps=50)

@pytest.mark.parametrize(['backend', 'jit'], [('pyro', True), ('pyro', False), ('minipyro', False)])
def test_elbo_equivalence(backend, jit):
    if False:
        return 10
    '\n    Given model and guide\n    Calculating the elbo yields equivalent results\n    '
    pyro.set_rng_seed(0)
    data = torch.tensor(0.5)
    elbo_test_case(backend, jit, expected_elbo=0.478, data=data)

def elbo_test_case(backend, jit, expected_elbo, data, steps=None):
    if False:
        i = 10
        return i + 15
    if torch.__version__.startswith('2'):
        pytest.xfail(reason='https://github.com/pyro-ppl/pyro/issues/3221')
    with pyro_backend(backend):
        Elbo = infer.JitTrace_ELBO if jit else infer.Trace_ELBO
        elbo = Elbo(ignore_jit_warnings=True)
        if steps:
            assert_ok(constrained_model, guide_constrained_model, elbo, steps, data)
        if backend == 'pyro':
            elbo = elbo.loss
        with torch.no_grad():
            actual = elbo(constrained_model, guide_constrained_model, data)
        assert actual == approx(expected_elbo, rel=0.1)