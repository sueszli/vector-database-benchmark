import logging
import warnings
from collections import defaultdict
import pytest
import torch
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.testing import fakes
from pyro.infer import SVI, EnergyDistance, Trace_ELBO, TraceEnum_ELBO, TraceGraph_ELBO, TraceMeanField_ELBO, TraceTailAdaptive_ELBO, config_enumerate
from pyro.infer.reparam import LatentStableReparam
from pyro.infer.tracetmc_elbo import TraceTMC_ELBO
from pyro.infer.util import torch_item
from pyro.ops.indexing import Vindex
from pyro.optim import Adam
from pyro.poutine.plate_messenger import block_plate
from tests.common import assert_close
logger = logging.getLogger(__name__)

def EnergyDistance_prior(**kwargs):
    if False:
        print('Hello World!')
    kwargs['prior_scale'] = 0.0
    kwargs.pop('strict_enumeration_warning', None)
    return EnergyDistance(**kwargs)

def EnergyDistance_noprior(**kwargs):
    if False:
        i = 10
        return i + 15
    kwargs['prior_scale'] = 1.0
    kwargs.pop('strict_enumeration_warning', None)
    return EnergyDistance(**kwargs)

def assert_ok(model, guide, elbo, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Assert that inference works without warnings or errors.\n    '
    pyro.clear_param_store()
    inference = SVI(model, guide, Adam({'lr': 1e-06}), elbo)
    inference.step(**kwargs)
    try:
        pyro.set_rng_seed(0)
        loss = elbo.loss(model, guide, **kwargs)
        if hasattr(elbo, 'differentiable_loss'):
            try:
                pyro.set_rng_seed(0)
                differentiable_loss = torch_item(elbo.differentiable_loss(model, guide, **kwargs))
            except ValueError:
                pass
            else:
                assert_close(differentiable_loss, loss, atol=0.01)
        if hasattr(elbo, 'loss_and_grads'):
            pyro.set_rng_seed(0)
            loss_and_grads = elbo.loss_and_grads(model, guide, **kwargs)
            assert_close(loss_and_grads, loss, atol=0.01)
    except NotImplementedError:
        pass

def assert_error(model, guide, elbo, match=None):
    if False:
        while True:
            i = 10
    '\n    Assert that inference fails with an error.\n    '
    pyro.clear_param_store()
    inference = SVI(model, guide, Adam({'lr': 1e-06}), elbo)
    with pytest.raises((NotImplementedError, UserWarning, KeyError, ValueError, RuntimeError), match=match):
        inference.step()

def assert_warning(model, guide, elbo):
    if False:
        i = 10
        return i + 15
    '\n    Assert that inference works but with a warning.\n    '
    pyro.clear_param_store()
    inference = SVI(model, guide, Adam({'lr': 1e-06}), elbo)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        inference.step()
        assert len(w), 'No warnings were raised'
        for warning in w:
            logger.info(warning)

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO, EnergyDistance_prior, EnergyDistance_noprior])
@pytest.mark.parametrize('strict_enumeration_warning', [True, False])
def test_nonempty_model_empty_guide_ok(Elbo, strict_enumeration_warning):
    if False:
        return 10

    def model():
        if False:
            print('Hello World!')
        loc = torch.tensor([0.0, 0.0])
        scale = torch.tensor([1.0, 1.0])
        pyro.sample('x', dist.Normal(loc, scale).to_event(1), obs=loc)

    def guide():
        if False:
            return 10
        pass
    elbo = Elbo(strict_enumeration_warning=strict_enumeration_warning)
    if strict_enumeration_warning and Elbo in (TraceEnum_ELBO, TraceTMC_ELBO):
        assert_warning(model, guide, elbo)
    else:
        assert_ok(model, guide, elbo)

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO, EnergyDistance_prior, EnergyDistance_noprior])
@pytest.mark.parametrize('strict_enumeration_warning', [True, False])
def test_nonempty_model_empty_guide_error(Elbo, strict_enumeration_warning):
    if False:
        i = 10
        return i + 15

    def model():
        if False:
            i = 10
            return i + 15
        pyro.sample('x', dist.Normal(0, 1))

    def guide():
        if False:
            for i in range(10):
                print('nop')
        pass
    elbo = Elbo(strict_enumeration_warning=strict_enumeration_warning)
    assert_error(model, guide, elbo)

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
@pytest.mark.parametrize('strict_enumeration_warning', [True, False])
def test_empty_model_empty_guide_ok(Elbo, strict_enumeration_warning):
    if False:
        i = 10
        return i + 15

    def model():
        if False:
            while True:
                i = 10
        pass

    def guide():
        if False:
            print('Hello World!')
        pass
    elbo = Elbo(strict_enumeration_warning=strict_enumeration_warning)
    if strict_enumeration_warning and Elbo in (TraceEnum_ELBO, TraceTMC_ELBO):
        assert_warning(model, guide, elbo)
    else:
        assert_ok(model, guide, elbo)

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_variable_clash_in_model_error(Elbo):
    if False:
        for i in range(10):
            print('nop')

    def model():
        if False:
            while True:
                i = 10
        p = torch.tensor(0.5)
        pyro.sample('x', dist.Bernoulli(p))
        pyro.sample('x', dist.Bernoulli(p))

    def guide():
        if False:
            while True:
                i = 10
        p = pyro.param('p', torch.tensor(0.5, requires_grad=True))
        pyro.sample('x', dist.Bernoulli(p))
    assert_error(model, guide, Elbo(), match='Multiple sample sites named')

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_model_guide_dim_mismatch_error(Elbo):
    if False:
        for i in range(10):
            print('nop')

    def model():
        if False:
            i = 10
            return i + 15
        loc = torch.zeros(2)
        scale = torch.ones(2)
        pyro.sample('x', dist.Normal(loc, scale).to_event(1))

    def guide():
        if False:
            return 10
        loc = pyro.param('loc', torch.zeros(2, 1, requires_grad=True))
        scale = pyro.param('scale', torch.ones(2, 1, requires_grad=True))
        pyro.sample('x', dist.Normal(loc, scale).to_event(2))
    assert_error(model, guide, Elbo(strict_enumeration_warning=False), match='invalid log_prob shape|Model and guide event_dims disagree')

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_model_guide_shape_mismatch_error(Elbo):
    if False:
        for i in range(10):
            print('nop')

    def model():
        if False:
            while True:
                i = 10
        loc = torch.zeros(1, 2)
        scale = torch.ones(1, 2)
        pyro.sample('x', dist.Normal(loc, scale).to_event(2))

    def guide():
        if False:
            i = 10
            return i + 15
        loc = pyro.param('loc', torch.zeros(2, 1, requires_grad=True))
        scale = pyro.param('scale', torch.ones(2, 1, requires_grad=True))
        pyro.sample('x', dist.Normal(loc, scale).to_event(2))
    assert_error(model, guide, Elbo(strict_enumeration_warning=False), match='Model and guide shapes disagree')

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_variable_clash_in_guide_error(Elbo):
    if False:
        print('Hello World!')

    def model():
        if False:
            while True:
                i = 10
        p = torch.tensor(0.5)
        pyro.sample('x', dist.Bernoulli(p))

    def guide():
        if False:
            print('Hello World!')
        p = pyro.param('p', torch.tensor(0.5, requires_grad=True))
        pyro.sample('x', dist.Bernoulli(p))
        pyro.sample('x', dist.Bernoulli(p))
    assert_error(model, guide, Elbo(), match='Multiple sample sites named')

@pytest.mark.parametrize('has_rsample', [False, True])
@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_set_has_rsample_ok(has_rsample, Elbo):
    if False:
        return 10

    def model():
        if False:
            i = 10
            return i + 15
        z = pyro.sample('z', dist.Normal(0, 1))
        loc = (z * 100).clamp(min=0, max=1)
        pyro.sample('x', dist.Normal(loc, 1), obs=torch.tensor(0.0))

    def guide():
        if False:
            return 10
        loc = pyro.param('loc', torch.tensor(0.0))
        pyro.sample('z', dist.Normal(loc, 1).has_rsample_(has_rsample))
    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)
    assert_ok(model, guide, Elbo(strict_enumeration_warning=False))

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_not_has_rsample_ok(Elbo):
    if False:
        print('Hello World!')

    def model():
        if False:
            i = 10
            return i + 15
        z = pyro.sample('z', dist.Normal(0, 1))
        p = z.round().clamp(min=0.2, max=0.8)
        pyro.sample('x', dist.Bernoulli(p), obs=torch.tensor(0.0))

    def guide():
        if False:
            print('Hello World!')
        loc = pyro.param('loc', torch.tensor(0.0))
        pyro.sample('z', dist.Normal(loc, 1).has_rsample_(False))
    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)
    assert_ok(model, guide, Elbo(strict_enumeration_warning=False))

@pytest.mark.parametrize('subsample_size', [None, 2], ids=['full', 'subsample'])
@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_iplate_ok(subsample_size, Elbo):
    if False:
        print('Hello World!')

    def model():
        if False:
            for i in range(10):
                print('nop')
        p = torch.tensor(0.5)
        for i in pyro.plate('plate', 4, subsample_size):
            pyro.sample('x_{}'.format(i), dist.Bernoulli(p))

    def guide():
        if False:
            while True:
                i = 10
        p = pyro.param('p', torch.tensor(0.5, requires_grad=True))
        for i in pyro.plate('plate', 4, subsample_size):
            pyro.sample('x_{}'.format(i), dist.Bernoulli(p))
    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)
    assert_ok(model, guide, Elbo())

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_iplate_variable_clash_error(Elbo):
    if False:
        i = 10
        return i + 15

    def model():
        if False:
            for i in range(10):
                print('nop')
        p = torch.tensor(0.5)
        for i in pyro.plate('plate', 2):
            pyro.sample('x', dist.Bernoulli(p))

    def guide():
        if False:
            return 10
        p = pyro.param('p', torch.tensor(0.5, requires_grad=True))
        for i in pyro.plate('plate', 2):
            pyro.sample('x', dist.Bernoulli(p))
    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)
    assert_error(model, guide, Elbo(), match='Multiple sample sites named')

@pytest.mark.parametrize('subsample_size', [None, 5], ids=['full', 'subsample'])
@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_plate_ok(subsample_size, Elbo):
    if False:
        return 10

    def model():
        if False:
            return 10
        p = torch.tensor(0.5)
        with pyro.plate('plate', 10, subsample_size) as ind:
            pyro.sample('x', dist.Bernoulli(p).expand_by([len(ind)]))

    def guide():
        if False:
            return 10
        p = pyro.param('p', torch.tensor(0.5, requires_grad=True))
        with pyro.plate('plate', 10, subsample_size) as ind:
            pyro.sample('x', dist.Bernoulli(p).expand_by([len(ind)]))
    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)
    assert_ok(model, guide, Elbo())

@pytest.mark.parametrize('subsample_size', [None, 5], ids=['full', 'subsample'])
@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_plate_subsample_param_ok(subsample_size, Elbo):
    if False:
        return 10

    def model():
        if False:
            i = 10
            return i + 15
        p = torch.tensor(0.5)
        with pyro.plate('plate', 10, subsample_size):
            pyro.sample('x', dist.Bernoulli(p))

    def guide():
        if False:
            return 10
        with pyro.plate('plate', 10, subsample_size) as ind:
            p0 = pyro.param('p0', torch.tensor(0.0), event_dim=0)
            assert p0.shape == ()
            p = pyro.param('p', 0.5 * torch.ones(10), event_dim=0)
            assert len(p) == len(ind)
            pyro.sample('x', dist.Bernoulli(p))
    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)
    assert_ok(model, guide, Elbo())

@pytest.mark.parametrize('subsample_size', [None, 5], ids=['full', 'subsample'])
@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_plate_subsample_primitive_ok(subsample_size, Elbo):
    if False:
        i = 10
        return i + 15

    def model():
        if False:
            return 10
        p = torch.tensor(0.5)
        with pyro.plate('plate', 10, subsample_size):
            pyro.sample('x', dist.Bernoulli(p))

    def guide():
        if False:
            return 10
        with pyro.plate('plate', 10, subsample_size) as ind:
            p0 = torch.tensor(0.0)
            p0 = pyro.subsample(p0, event_dim=0)
            assert p0.shape == ()
            p = 0.5 * torch.ones(10)
            p = pyro.subsample(p, event_dim=0)
            assert len(p) == len(ind)
            pyro.sample('x', dist.Bernoulli(p))
    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)
    assert_ok(model, guide, Elbo())

@pytest.mark.parametrize('subsample_size', [None, 5], ids=['full', 'subsample'])
@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
@pytest.mark.parametrize('shape,ok', [((), True), ((1,), True), ((10,), True), ((3, 1), True), ((3, 10), True), (5, False), ((3, 5), False)])
def test_plate_param_size_mismatch_error(subsample_size, Elbo, shape, ok):
    if False:
        print('Hello World!')

    def model():
        if False:
            while True:
                i = 10
        p = torch.tensor(0.5)
        with pyro.plate('plate', 10, subsample_size):
            pyro.sample('x', dist.Bernoulli(p))

    def guide():
        if False:
            while True:
                i = 10
        with pyro.plate('plate', 10, subsample_size):
            pyro.param('p0', torch.ones(shape), event_dim=0)
            p = pyro.param('p', torch.ones(10), event_dim=0)
            pyro.sample('x', dist.Bernoulli(p))
    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)
    if ok:
        assert_ok(model, guide, Elbo())
    else:
        assert_error(model, guide, Elbo(), match='invalid shape of pyro.param')

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_plate_no_size_ok(Elbo):
    if False:
        return 10

    def model():
        if False:
            i = 10
            return i + 15
        p = torch.tensor(0.5)
        with pyro.plate('plate'):
            pyro.sample('x', dist.Bernoulli(p).expand_by([10]))

    def guide():
        if False:
            print('Hello World!')
        p = pyro.param('p', torch.tensor(0.5, requires_grad=True))
        with pyro.plate('plate'):
            pyro.sample('x', dist.Bernoulli(p).expand_by([10]))
    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, default='parallel', num_samples=2)
    assert_ok(model, guide, Elbo())

@pytest.mark.parametrize('max_plate_nesting', [0, float('inf')])
@pytest.mark.parametrize('subsample_size', [None, 2], ids=['full', 'subsample'])
@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_iplate_iplate_ok(subsample_size, Elbo, max_plate_nesting):
    if False:
        for i in range(10):
            print('nop')

    def model():
        if False:
            return 10
        p = torch.tensor(0.5)
        outer_iplate = pyro.plate('plate_0', 3, subsample_size)
        inner_iplate = pyro.plate('plate_1', 3, subsample_size)
        for i in outer_iplate:
            for j in inner_iplate:
                pyro.sample('x_{}_{}'.format(i, j), dist.Bernoulli(p))

    def guide():
        if False:
            i = 10
            return i + 15
        p = pyro.param('p', torch.tensor(0.5, requires_grad=True))
        outer_iplate = pyro.plate('plate_0', 3, subsample_size)
        inner_iplate = pyro.plate('plate_1', 3, subsample_size)
        for i in outer_iplate:
            for j in inner_iplate:
                pyro.sample('x_{}_{}'.format(i, j), dist.Bernoulli(p))
    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide, 'parallel')
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)
    assert_ok(model, guide, Elbo(max_plate_nesting=max_plate_nesting))

@pytest.mark.parametrize('max_plate_nesting', [0, float('inf')])
@pytest.mark.parametrize('subsample_size', [None, 2], ids=['full', 'subsample'])
@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_iplate_iplate_swap_ok(subsample_size, Elbo, max_plate_nesting):
    if False:
        while True:
            i = 10

    def model():
        if False:
            print('Hello World!')
        p = torch.tensor(0.5)
        outer_iplate = pyro.plate('plate_0', 3, subsample_size)
        inner_iplate = pyro.plate('plate_1', 3, subsample_size)
        for i in outer_iplate:
            for j in inner_iplate:
                pyro.sample('x_{}_{}'.format(i, j), dist.Bernoulli(p))

    def guide():
        if False:
            i = 10
            return i + 15
        p = pyro.param('p', torch.tensor(0.5, requires_grad=True))
        outer_iplate = pyro.plate('plate_0', 3, subsample_size)
        inner_iplate = pyro.plate('plate_1', 3, subsample_size)
        for j in inner_iplate:
            for i in outer_iplate:
                pyro.sample('x_{}_{}'.format(i, j), dist.Bernoulli(p))
    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide, 'parallel')
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, default='parallel', num_samples=2)
    assert_ok(model, guide, Elbo(max_plate_nesting=max_plate_nesting))

@pytest.mark.parametrize('subsample_size', [None, 5], ids=['full', 'subsample'])
@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_iplate_in_model_not_guide_ok(subsample_size, Elbo):
    if False:
        i = 10
        return i + 15

    def model():
        if False:
            while True:
                i = 10
        p = torch.tensor(0.5)
        for i in pyro.plate('plate', 10, subsample_size):
            pass
        pyro.sample('x', dist.Bernoulli(p))

    def guide():
        if False:
            for i in range(10):
                print('nop')
        p = pyro.param('p', torch.tensor(0.5, requires_grad=True))
        pyro.sample('x', dist.Bernoulli(p))
    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)
    assert_ok(model, guide, Elbo())

@pytest.mark.parametrize('subsample_size', [None, 5], ids=['full', 'subsample'])
@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
@pytest.mark.parametrize('is_validate', [True, False])
def test_iplate_in_guide_not_model_error(subsample_size, Elbo, is_validate):
    if False:
        print('Hello World!')

    def model():
        if False:
            for i in range(10):
                print('nop')
        p = torch.tensor(0.5)
        pyro.sample('x', dist.Bernoulli(p))

    def guide():
        if False:
            i = 10
            return i + 15
        p = pyro.param('p', torch.tensor(0.5, requires_grad=True))
        for i in pyro.plate('plate', 10, subsample_size):
            pass
        pyro.sample('x', dist.Bernoulli(p))
    with pyro.validation_enabled(is_validate):
        if is_validate:
            assert_error(model, guide, Elbo(), match='Found plate statements in guide but not model')
        else:
            assert_ok(model, guide, Elbo())

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_plate_broadcast_error(Elbo):
    if False:
        for i in range(10):
            print('nop')

    def model():
        if False:
            for i in range(10):
                print('nop')
        p = torch.tensor(0.5, requires_grad=True)
        with pyro.plate('plate', 10, 5):
            pyro.sample('x', dist.Bernoulli(p).expand_by([2]))
    assert_error(model, model, Elbo(), match='Shape mismatch inside plate')

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_plate_iplate_ok(Elbo):
    if False:
        while True:
            i = 10

    def model():
        if False:
            for i in range(10):
                print('nop')
        p = torch.tensor(0.5)
        with pyro.plate('plate', 3, 2) as ind:
            for i in pyro.plate('iplate', 3, 2):
                pyro.sample('x_{}'.format(i), dist.Bernoulli(p).expand_by([len(ind)]))

    def guide():
        if False:
            for i in range(10):
                print('nop')
        p = pyro.param('p', torch.tensor(0.5, requires_grad=True))
        with pyro.plate('plate', 3, 2) as ind:
            for i in pyro.plate('iplate', 3, 2):
                pyro.sample('x_{}'.format(i), dist.Bernoulli(p).expand_by([len(ind)]))
    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)
    assert_ok(model, guide, Elbo())

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_iplate_plate_ok(Elbo):
    if False:
        i = 10
        return i + 15

    def model():
        if False:
            return 10
        p = torch.tensor(0.5)
        inner_plate = pyro.plate('plate', 3, 2)
        for i in pyro.plate('iplate', 3, 2):
            with inner_plate as ind:
                pyro.sample('x_{}'.format(i), dist.Bernoulli(p).expand_by([len(ind)]))

    def guide():
        if False:
            print('Hello World!')
        p = pyro.param('p', torch.tensor(0.5, requires_grad=True))
        inner_plate = pyro.plate('plate', 3, 2)
        for i in pyro.plate('iplate', 3, 2):
            with inner_plate as ind:
                pyro.sample('x_{}'.format(i), dist.Bernoulli(p).expand_by([len(ind)]))
    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)
    assert_ok(model, guide, Elbo())

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
@pytest.mark.parametrize('sizes', [(3,), (3, 4), (3, 4, 5)])
def test_plate_stack_ok(Elbo, sizes):
    if False:
        print('Hello World!')

    def model():
        if False:
            return 10
        p = torch.tensor(0.5)
        with pyro.plate_stack('plate_stack', sizes):
            pyro.sample('x', dist.Bernoulli(p))

    def guide():
        if False:
            i = 10
            return i + 15
        p = pyro.param('p', torch.tensor(0.5, requires_grad=True))
        with pyro.plate_stack('plate_stack', sizes):
            pyro.sample('x', dist.Bernoulli(p))
    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)
    assert_ok(model, guide, Elbo())

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
@pytest.mark.parametrize('sizes', [(3,), (3, 4), (3, 4, 5)])
def test_plate_stack_and_plate_ok(Elbo, sizes):
    if False:
        print('Hello World!')

    def model():
        if False:
            i = 10
            return i + 15
        p = torch.tensor(0.5)
        with pyro.plate_stack('plate_stack', sizes):
            with pyro.plate('plate', 7):
                pyro.sample('x', dist.Bernoulli(p))

    def guide():
        if False:
            while True:
                i = 10
        p = pyro.param('p', torch.tensor(0.5, requires_grad=True))
        with pyro.plate_stack('plate_stack', sizes):
            with pyro.plate('plate', 7):
                pyro.sample('x', dist.Bernoulli(p))
    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)
    assert_ok(model, guide, Elbo())

@pytest.mark.parametrize('sizes', [(3,), (3, 4), (3, 4, 5)])
def test_plate_stack_sizes(sizes):
    if False:
        while True:
            i = 10

    def model():
        if False:
            print('Hello World!')
        p = 0.5 * torch.ones(3)
        with pyro.plate_stack('plate_stack', sizes):
            x = pyro.sample('x', dist.Bernoulli(p).to_event(1))
            assert x.shape == sizes + (3,)
    model()

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_nested_plate_plate_ok(Elbo):
    if False:
        return 10

    def model():
        if False:
            for i in range(10):
                print('nop')
        p = torch.tensor(0.5, requires_grad=True)
        with pyro.plate('plate_outer', 10, 5) as ind_outer:
            pyro.sample('x', dist.Bernoulli(p).expand_by([len(ind_outer)]))
            with pyro.plate('plate_inner', 11, 6) as ind_inner:
                pyro.sample('y', dist.Bernoulli(p).expand_by([len(ind_inner), len(ind_outer)]))
    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(model)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(model, num_samples=2)
    else:
        guide = model
    assert_ok(model, guide, Elbo())

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_plate_reuse_ok(Elbo):
    if False:
        while True:
            i = 10

    def model():
        if False:
            for i in range(10):
                print('nop')
        p = torch.tensor(0.5, requires_grad=True)
        plate_outer = pyro.plate('plate_outer', 10, 5, dim=-1)
        plate_inner = pyro.plate('plate_inner', 11, 6, dim=-2)
        with plate_outer as ind_outer:
            pyro.sample('x', dist.Bernoulli(p).expand_by([len(ind_outer)]))
        with plate_inner as ind_inner:
            pyro.sample('y', dist.Bernoulli(p).expand_by([len(ind_inner), 1]))
        with plate_outer as ind_outer, plate_inner as ind_inner:
            pyro.sample('z', dist.Bernoulli(p).expand_by([len(ind_inner), len(ind_outer)]))
    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(model)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(model, num_samples=2)
    else:
        guide = model
    assert_ok(model, guide, Elbo())

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_nested_plate_plate_dim_error_1(Elbo):
    if False:
        for i in range(10):
            print('nop')

    def model():
        if False:
            return 10
        p = torch.tensor([0.5], requires_grad=True)
        with pyro.plate('plate_outer', 10, 5) as ind_outer:
            pyro.sample('x', dist.Bernoulli(p).expand_by([len(ind_outer)]))
            with pyro.plate('plate_inner', 11, 6) as ind_inner:
                pyro.sample('y', dist.Bernoulli(p).expand_by([len(ind_inner)]))
                pyro.sample('z', dist.Bernoulli(p).expand_by([len(ind_outer), len(ind_inner)]))
    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(model)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(model, num_samples=2)
    else:
        guide = model
    assert_error(model, guide, Elbo(), match='invalid log_prob shape')

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_nested_plate_plate_dim_error_2(Elbo):
    if False:
        i = 10
        return i + 15

    def model():
        if False:
            for i in range(10):
                print('nop')
        p = torch.tensor([0.5], requires_grad=True)
        with pyro.plate('plate_outer', 10, 5) as ind_outer:
            pyro.sample('x', dist.Bernoulli(p).expand_by([len(ind_outer), 1]))
            with pyro.plate('plate_inner', 11, 6) as ind_inner:
                pyro.sample('y', dist.Bernoulli(p).expand_by([len(ind_outer)]))
                pyro.sample('z', dist.Bernoulli(p).expand_by([len(ind_outer), len(ind_inner)]))
    guide = config_enumerate(model) if Elbo is TraceEnum_ELBO else model
    assert_error(model, guide, Elbo(), match='Shape mismatch inside plate')

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_nested_plate_plate_dim_error_3(Elbo):
    if False:
        for i in range(10):
            print('nop')

    def model():
        if False:
            i = 10
            return i + 15
        p = torch.tensor([0.5], requires_grad=True)
        with pyro.plate('plate_outer', 10, 5) as ind_outer:
            pyro.sample('x', dist.Bernoulli(p).expand_by([len(ind_outer), 1]))
            with pyro.plate('plate_inner', 11, 6) as ind_inner:
                pyro.sample('y', dist.Bernoulli(p).expand_by([len(ind_inner)]))
                pyro.sample('z', dist.Bernoulli(p).expand_by([len(ind_inner), 1]))
    guide = config_enumerate(model) if Elbo is TraceEnum_ELBO else model
    assert_error(model, guide, Elbo(), match='invalid log_prob shape|shape mismatch')

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_nested_plate_plate_dim_error_4(Elbo):
    if False:
        while True:
            i = 10

    def model():
        if False:
            for i in range(10):
                print('nop')
        p = torch.tensor([0.5], requires_grad=True)
        with pyro.plate('plate_outer', 10, 5) as ind_outer:
            pyro.sample('x', dist.Bernoulli(p).expand_by([len(ind_outer), 1]))
            with pyro.plate('plate_inner', 11, 6) as ind_inner:
                pyro.sample('y', dist.Bernoulli(p).expand_by([len(ind_inner)]))
                pyro.sample('z', dist.Bernoulli(p).expand_by([len(ind_outer), len(ind_outer)]))
    guide = config_enumerate(model) if Elbo is TraceEnum_ELBO else model
    assert_error(model, guide, Elbo(), match='hape mismatch inside plate')

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_nested_plate_plate_subsample_param_ok(Elbo):
    if False:
        print('Hello World!')

    def model():
        if False:
            return 10
        with pyro.plate('plate_outer', 10, 5):
            pyro.sample('x', dist.Bernoulli(0.2))
            with pyro.plate('plate_inner', 11, 6):
                pyro.sample('y', dist.Bernoulli(0.2))

    def guide():
        if False:
            return 10
        p0 = pyro.param('p0', 0.5 * torch.ones(4, 5), event_dim=2)
        assert p0.shape == (4, 5)
        with pyro.plate('plate_outer', 10, 5):
            p1 = pyro.param('p1', 0.5 * torch.ones(10, 3), event_dim=1)
            assert p1.shape == (5, 3)
            px = pyro.param('px', 0.5 * torch.ones(10), event_dim=0)
            assert px.shape == (5,)
            pyro.sample('x', dist.Bernoulli(px))
            with pyro.plate('plate_inner', 11, 6):
                py = pyro.param('py', 0.5 * torch.ones(11, 10), event_dim=0)
                assert py.shape == (6, 5)
                pyro.sample('y', dist.Bernoulli(py))
    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    assert_ok(model, guide, Elbo())

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_nonnested_plate_plate_ok(Elbo):
    if False:
        i = 10
        return i + 15

    def model():
        if False:
            print('Hello World!')
        p = torch.tensor(0.5, requires_grad=True)
        with pyro.plate('plate_0', 10, 5) as ind1:
            pyro.sample('x0', dist.Bernoulli(p).expand_by([len(ind1)]))
        with pyro.plate('plate_1', 11, 6) as ind2:
            pyro.sample('x1', dist.Bernoulli(p).expand_by([len(ind2)]))
    guide = config_enumerate(model) if Elbo is TraceEnum_ELBO else model
    assert_ok(model, guide, Elbo())

def test_three_indep_plate_at_different_depths_ok():
    if False:
        while True:
            i = 10
    '\n      /\\\n     /\\ ia\n    ia ia\n    '

    def model():
        if False:
            while True:
                i = 10
        p = torch.tensor(0.5)
        inner_plate = pyro.plate('plate2', 10, 5)
        for i in pyro.plate('plate0', 2):
            pyro.sample('x_%d' % i, dist.Bernoulli(p))
            if i == 0:
                for j in pyro.plate('plate1', 2):
                    with inner_plate as ind:
                        pyro.sample('y_%d' % j, dist.Bernoulli(p).expand_by([len(ind)]))
            elif i == 1:
                with inner_plate as ind:
                    pyro.sample('z', dist.Bernoulli(p).expand_by([len(ind)]))

    def guide():
        if False:
            i = 10
            return i + 15
        p = pyro.param('p', torch.tensor(0.5, requires_grad=True))
        inner_plate = pyro.plate('plate2', 10, 5)
        for i in pyro.plate('plate0', 2):
            pyro.sample('x_%d' % i, dist.Bernoulli(p))
            if i == 0:
                for j in pyro.plate('plate1', 2):
                    with inner_plate as ind:
                        pyro.sample('y_%d' % j, dist.Bernoulli(p).expand_by([len(ind)]))
            elif i == 1:
                with inner_plate as ind:
                    pyro.sample('z', dist.Bernoulli(p).expand_by([len(ind)]))
    assert_ok(model, guide, TraceGraph_ELBO())

def test_plate_wrong_size_error():
    if False:
        print('Hello World!')

    def model():
        if False:
            for i in range(10):
                print('nop')
        p = torch.tensor(0.5)
        with pyro.plate('plate', 10, 5) as ind:
            pyro.sample('x', dist.Bernoulli(p).expand_by([1 + len(ind)]))

    def guide():
        if False:
            for i in range(10):
                print('nop')
        p = pyro.param('p', torch.tensor(0.5, requires_grad=True))
        with pyro.plate('plate', 10, 5) as ind:
            pyro.sample('x', dist.Bernoulli(p).expand_by([1 + len(ind)]))
    assert_error(model, guide, TraceGraph_ELBO(), match='Shape mismatch inside plate')

def test_block_plate_name_ok():
    if False:
        while True:
            i = 10

    def model():
        if False:
            print('Hello World!')
        a = pyro.sample('a', dist.Normal(0, 1))
        assert a.shape == ()
        with pyro.plate('plate', 2):
            b = pyro.sample('b', dist.Normal(0, 1))
            assert b.shape == (2,)
            with block_plate('plate'):
                c = pyro.sample('c', dist.Normal(0, 1))
                assert c.shape == ()

    def guide():
        if False:
            return 10
        c = pyro.sample('c', dist.Normal(0, 1))
        assert c.shape == ()
        with pyro.plate('plate', 2):
            b = pyro.sample('b', dist.Normal(0, 1))
            assert b.shape == (2,)
            with block_plate('plate'):
                a = pyro.sample('a', dist.Normal(0, 1))
                assert a.shape == ()
    assert_ok(model, guide, Trace_ELBO())

def test_block_plate_dim_ok():
    if False:
        return 10

    def model():
        if False:
            for i in range(10):
                print('nop')
        a = pyro.sample('a', dist.Normal(0, 1))
        assert a.shape == ()
        with pyro.plate('plate', 2):
            b = pyro.sample('b', dist.Normal(0, 1))
            assert b.shape == (2,)
            with block_plate(dim=-1):
                c = pyro.sample('c', dist.Normal(0, 1))
                assert c.shape == ()

    def guide():
        if False:
            i = 10
            return i + 15
        c = pyro.sample('c', dist.Normal(0, 1))
        assert c.shape == ()
        with pyro.plate('plate', 2):
            b = pyro.sample('b', dist.Normal(0, 1))
            assert b.shape == (2,)
            with block_plate(dim=-1):
                a = pyro.sample('a', dist.Normal(0, 1))
                assert a.shape == ()
    assert_ok(model, guide, Trace_ELBO())

def test_block_plate_missing_error():
    if False:
        print('Hello World!')

    def model():
        if False:
            print('Hello World!')
        with block_plate('plate'):
            pyro.sample('a', dist.Normal(0, 1))

    def guide():
        if False:
            i = 10
            return i + 15
        pyro.sample('a', dist.Normal(0, 1))
    assert_error(model, guide, Trace_ELBO(), match='block_plate matched 0 messengers')

@pytest.mark.parametrize('enumerate_', [None, 'sequential', 'parallel'])
@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_enum_discrete_misuse_warning(Elbo, enumerate_):
    if False:
        for i in range(10):
            print('nop')

    def model():
        if False:
            while True:
                i = 10
        p = torch.tensor(0.5)
        pyro.sample('x', dist.Bernoulli(p))

    def guide():
        if False:
            while True:
                i = 10
        p = pyro.param('p', torch.tensor(0.5, requires_grad=True))
        pyro.sample('x', dist.Bernoulli(p), infer={'enumerate': enumerate_})
    if (enumerate_ is None) == (Elbo is TraceEnum_ELBO):
        assert_warning(model, guide, Elbo(max_plate_nesting=0))
    else:
        assert_ok(model, guide, Elbo(max_plate_nesting=0))

def test_enum_discrete_single_ok():
    if False:
        while True:
            i = 10

    def model():
        if False:
            for i in range(10):
                print('nop')
        p = torch.tensor(0.5)
        pyro.sample('x', dist.Bernoulli(p))

    def guide():
        if False:
            return 10
        p = pyro.param('p', torch.tensor(0.5, requires_grad=True))
        pyro.sample('x', dist.Bernoulli(p))
    assert_ok(model, config_enumerate(guide), TraceEnum_ELBO())

@pytest.mark.parametrize('strict_enumeration_warning', [False, True])
def test_enum_discrete_missing_config_warning(strict_enumeration_warning):
    if False:
        while True:
            i = 10

    def model():
        if False:
            i = 10
            return i + 15
        p = torch.tensor(0.5)
        pyro.sample('x', dist.Bernoulli(p))

    def guide():
        if False:
            return 10
        p = pyro.param('p', torch.tensor(0.5, requires_grad=True))
        pyro.sample('x', dist.Bernoulli(p))
    elbo = TraceEnum_ELBO(strict_enumeration_warning=strict_enumeration_warning)
    if strict_enumeration_warning:
        assert_warning(model, guide, elbo)
    else:
        assert_ok(model, guide, elbo)

def test_enum_discrete_single_single_ok():
    if False:
        for i in range(10):
            print('nop')

    def model():
        if False:
            i = 10
            return i + 15
        p = torch.tensor(0.5)
        pyro.sample('x', dist.Bernoulli(p))
        pyro.sample('y', dist.Bernoulli(p))

    def guide():
        if False:
            return 10
        p = pyro.param('p', torch.tensor(0.5, requires_grad=True))
        pyro.sample('x', dist.Bernoulli(p))
        pyro.sample('y', dist.Bernoulli(p))
    assert_ok(model, config_enumerate(guide), TraceEnum_ELBO())

def test_enum_discrete_iplate_single_ok():
    if False:
        for i in range(10):
            print('nop')

    def model():
        if False:
            while True:
                i = 10
        p = torch.tensor(0.5)
        for i in pyro.plate('plate', 10, 5):
            pyro.sample('x_{}'.format(i), dist.Bernoulli(p))

    def guide():
        if False:
            while True:
                i = 10
        p = pyro.param('p', torch.tensor(0.5, requires_grad=True))
        for i in pyro.plate('plate', 10, 5):
            pyro.sample('x_{}'.format(i), dist.Bernoulli(p))
    assert_ok(model, config_enumerate(guide), TraceEnum_ELBO())

def test_plate_enum_discrete_batch_ok():
    if False:
        print('Hello World!')

    def model():
        if False:
            i = 10
            return i + 15
        p = torch.tensor(0.5)
        with pyro.plate('plate', 10, 5) as ind:
            pyro.sample('x', dist.Bernoulli(p).expand_by([len(ind)]))

    def guide():
        if False:
            i = 10
            return i + 15
        p = pyro.param('p', torch.tensor(0.5, requires_grad=True))
        with pyro.plate('plate', 10, 5) as ind:
            pyro.sample('x', dist.Bernoulli(p).expand_by([len(ind)]))
    assert_ok(model, config_enumerate(guide), TraceEnum_ELBO())

@pytest.mark.parametrize('strict_enumeration_warning', [False, True])
def test_plate_enum_discrete_no_discrete_vars_warning(strict_enumeration_warning):
    if False:
        return 10

    def model():
        if False:
            print('Hello World!')
        loc = torch.tensor(0.0)
        scale = torch.tensor(1.0)
        with pyro.plate('plate', 10, 5) as ind:
            pyro.sample('x', dist.Normal(loc, scale).expand_by([len(ind)]))

    @config_enumerate(default='sequential')
    def guide():
        if False:
            i = 10
            return i + 15
        loc = pyro.param('loc', torch.tensor(1.0, requires_grad=True))
        scale = pyro.param('scale', torch.tensor(2.0, requires_grad=True))
        with pyro.plate('plate', 10, 5) as ind:
            pyro.sample('x', dist.Normal(loc, scale).expand_by([len(ind)]))
    elbo = TraceEnum_ELBO(strict_enumeration_warning=strict_enumeration_warning)
    if strict_enumeration_warning:
        assert_warning(model, guide, elbo)
    else:
        assert_ok(model, guide, elbo)

def test_no_plate_enum_discrete_batch_error():
    if False:
        while True:
            i = 10

    def model():
        if False:
            while True:
                i = 10
        p = torch.tensor(0.5)
        pyro.sample('x', dist.Bernoulli(p).expand_by([5]))

    def guide():
        if False:
            while True:
                i = 10
        p = pyro.param('p', torch.tensor(0.5, requires_grad=True))
        pyro.sample('x', dist.Bernoulli(p).expand_by([5]))
    assert_error(model, config_enumerate(guide), TraceEnum_ELBO(), match='invalid log_prob shape')

@pytest.mark.parametrize('max_plate_nesting', [0, 1, 2, float('inf')])
def test_enum_discrete_parallel_ok(max_plate_nesting):
    if False:
        while True:
            i = 10
    guessed_nesting = 0 if max_plate_nesting == float('inf') else max_plate_nesting
    plate_shape = torch.Size([1] * guessed_nesting)

    def model():
        if False:
            while True:
                i = 10
        p = torch.tensor(0.5)
        x = pyro.sample('x', dist.Bernoulli(p))
        if max_plate_nesting != float('inf'):
            assert x.shape == torch.Size([2]) + plate_shape

    def guide():
        if False:
            print('Hello World!')
        p = pyro.param('p', torch.tensor(0.5, requires_grad=True))
        x = pyro.sample('x', dist.Bernoulli(p))
        if max_plate_nesting != float('inf'):
            assert x.shape == torch.Size([2]) + plate_shape
    assert_ok(model, config_enumerate(guide, 'parallel'), TraceEnum_ELBO(max_plate_nesting=max_plate_nesting))

@pytest.mark.parametrize('max_plate_nesting', [0, 1, 2, float('inf')])
def test_enum_discrete_parallel_nested_ok(max_plate_nesting):
    if False:
        print('Hello World!')
    guessed_nesting = 0 if max_plate_nesting == float('inf') else max_plate_nesting
    plate_shape = torch.Size([1] * guessed_nesting)

    def model():
        if False:
            return 10
        p2 = torch.ones(2) / 2
        p3 = torch.ones(3) / 3
        x2 = pyro.sample('x2', dist.OneHotCategorical(p2))
        x3 = pyro.sample('x3', dist.OneHotCategorical(p3))
        if max_plate_nesting != float('inf'):
            assert x2.shape == torch.Size([2]) + plate_shape + p2.shape
            assert x3.shape == torch.Size([3, 1]) + plate_shape + p3.shape
    assert_ok(model, config_enumerate(model, 'parallel'), TraceEnum_ELBO(max_plate_nesting=max_plate_nesting))

@pytest.mark.parametrize('enumerate_,expand,num_samples', [(None, False, None), ('sequential', False, None), ('sequential', True, None), ('parallel', False, None), ('parallel', True, None), ('parallel', True, 3)])
def test_enumerate_parallel_plate_ok(enumerate_, expand, num_samples):
    if False:
        i = 10
        return i + 15

    def model():
        if False:
            while True:
                i = 10
        p2 = torch.ones(2) / 2
        p34 = torch.ones(3, 4) / 4
        p536 = torch.ones(5, 3, 6) / 6
        x2 = pyro.sample('x2', dist.Categorical(p2))
        with pyro.plate('outer', 3):
            x34 = pyro.sample('x34', dist.Categorical(p34))
            with pyro.plate('inner', 5):
                x536 = pyro.sample('x536', dist.Categorical(p536))
        if enumerate_ == 'parallel':
            if num_samples:
                n = num_samples
                assert x2.shape == torch.Size([n, 1, 1])
                assert x34.shape == torch.Size([n, 1, 1, 3])
                assert x536.shape == torch.Size([n, 1, 1, 5, 3])
            elif expand:
                assert x2.shape == torch.Size([2, 1, 1])
                assert x34.shape == torch.Size([4, 1, 1, 3])
                assert x536.shape == torch.Size([6, 1, 1, 5, 3])
            else:
                assert x2.shape == torch.Size([2, 1, 1])
                assert x34.shape == torch.Size([4, 1, 1, 1])
                assert x536.shape == torch.Size([6, 1, 1, 1, 1])
        elif enumerate_ == 'sequential':
            if expand:
                assert x2.shape == torch.Size([])
                assert x34.shape == torch.Size([3])
                assert x536.shape == torch.Size([5, 3])
            else:
                assert x2.shape == torch.Size([])
                assert x34.shape == torch.Size([1])
                assert x536.shape == torch.Size([1, 1])
        else:
            assert x2.shape == torch.Size([])
            assert x34.shape == torch.Size([3])
            assert x536.shape == torch.Size([5, 3])
    elbo = TraceEnum_ELBO(max_plate_nesting=2, strict_enumeration_warning=enumerate_)
    guide = config_enumerate(model, enumerate_, expand, num_samples)
    assert_ok(model, guide, elbo)

@pytest.mark.parametrize('max_plate_nesting', [1, float('inf')])
@pytest.mark.parametrize('enumerate_', [None, 'sequential', 'parallel'])
@pytest.mark.parametrize('is_validate', [True, False])
def test_enum_discrete_plate_dependency_warning(enumerate_, is_validate, max_plate_nesting):
    if False:
        i = 10
        return i + 15

    def model():
        if False:
            i = 10
            return i + 15
        pyro.sample('w', dist.Bernoulli(0.5), infer={'enumerate': 'parallel'})
        with pyro.plate('plate', 10, 5):
            x = pyro.sample('x', dist.Bernoulli(0.5).expand_by([5]), infer={'enumerate': enumerate_})
        pyro.sample('y', dist.Bernoulli(x.mean()))
    with pyro.validation_enabled(is_validate):
        elbo = TraceEnum_ELBO(max_plate_nesting=max_plate_nesting)
        if enumerate_ and is_validate:
            assert_warning(model, model, elbo)
        else:
            assert_ok(model, model, elbo)

@pytest.mark.parametrize('max_plate_nesting', [1, float('inf')])
@pytest.mark.parametrize('enumerate_', [None, 'sequential', 'parallel'])
def test_enum_discrete_iplate_plate_dependency_ok(enumerate_, max_plate_nesting):
    if False:
        return 10

    def model():
        if False:
            print('Hello World!')
        pyro.sample('w', dist.Bernoulli(0.5), infer={'enumerate': 'parallel'})
        inner_plate = pyro.plate('plate', 10, 5)
        for i in pyro.plate('iplate', 3):
            pyro.sample('y_{}'.format(i), dist.Bernoulli(0.5))
            with inner_plate:
                pyro.sample('x_{}'.format(i), dist.Bernoulli(0.5).expand_by([5]), infer={'enumerate': enumerate_})
    assert_ok(model, model, TraceEnum_ELBO(max_plate_nesting=max_plate_nesting))

@pytest.mark.parametrize('max_plate_nesting', [1, float('inf')])
@pytest.mark.parametrize('enumerate_', [None, 'sequential', 'parallel'])
@pytest.mark.parametrize('is_validate', [True, False])
def test_enum_discrete_iplates_plate_dependency_warning(enumerate_, is_validate, max_plate_nesting):
    if False:
        i = 10
        return i + 15

    def model():
        if False:
            return 10
        pyro.sample('w', dist.Bernoulli(0.5), infer={'enumerate': 'parallel'})
        inner_plate = pyro.plate('plate', 10, 5)
        for i in pyro.plate('iplate1', 2):
            with inner_plate:
                pyro.sample('x_{}'.format(i), dist.Bernoulli(0.5).expand_by([5]), infer={'enumerate': enumerate_})
        for i in pyro.plate('iplate2', 2):
            pyro.sample('y_{}'.format(i), dist.Bernoulli(0.5))
    with pyro.validation_enabled(is_validate):
        elbo = TraceEnum_ELBO(max_plate_nesting=max_plate_nesting)
        if enumerate_ and is_validate:
            assert_warning(model, model, elbo)
        else:
            assert_ok(model, model, elbo)

@pytest.mark.parametrize('enumerate_', [None, 'sequential', 'parallel'])
def test_enum_discrete_plates_dependency_ok(enumerate_):
    if False:
        i = 10
        return i + 15

    def model():
        if False:
            while True:
                i = 10
        pyro.sample('w', dist.Bernoulli(0.5), infer={'enumerate': 'parallel'})
        x_plate = pyro.plate('x_plate', 10, 5, dim=-1)
        y_plate = pyro.plate('y_plate', 11, 6, dim=-2)
        pyro.sample('a', dist.Bernoulli(0.5))
        with x_plate:
            pyro.sample('b', dist.Bernoulli(0.5).expand_by([5]))
        with y_plate:
            pyro.sample('c', dist.Bernoulli(0.5).expand_by([6, 1]))
        with x_plate, y_plate:
            pyro.sample('d', dist.Bernoulli(0.5).expand_by([6, 5]))
    assert_ok(model, model, TraceEnum_ELBO(max_plate_nesting=2))

@pytest.mark.parametrize('enumerate_', [None, 'sequential', 'parallel'])
def test_enum_discrete_non_enumerated_plate_ok(enumerate_):
    if False:
        print('Hello World!')

    def model():
        if False:
            print('Hello World!')
        pyro.sample('w', dist.Bernoulli(0.5), infer={'enumerate': 'parallel'})
        with pyro.plate('non_enum', 2):
            a = pyro.sample('a', dist.Bernoulli(0.5).expand_by([2]), infer={'enumerate': None})
        p = (1.0 + a.sum(-1)) / (2.0 + a.size(0))
        with pyro.plate('enum_1', 3):
            pyro.sample('b', dist.Bernoulli(p).expand_by([3]), infer={'enumerate': enumerate_})
    with pyro.validation_enabled():
        assert_ok(model, model, TraceEnum_ELBO(max_plate_nesting=1))

def test_plate_shape_broadcasting():
    if False:
        while True:
            i = 10
    data = torch.ones(1000, 2)

    def model():
        if False:
            for i in range(10):
                print('nop')
        with pyro.plate('num_particles', 10, dim=-3):
            with pyro.plate('components', 2, dim=-1):
                p = pyro.sample('p', dist.Beta(torch.tensor(1.1), torch.tensor(1.1)))
                assert p.shape == torch.Size((10, 1, 2))
            with pyro.plate('data', data.shape[0], dim=-2):
                pyro.sample('obs', dist.Bernoulli(p), obs=data)

    def guide():
        if False:
            print('Hello World!')
        with pyro.plate('num_particles', 10, dim=-3):
            with pyro.plate('components', 2, dim=-1):
                pyro.sample('p', dist.Beta(torch.tensor(1.1), torch.tensor(1.1)))
    assert_ok(model, guide, Trace_ELBO())

@pytest.mark.parametrize('enumerate_,expand,num_samples', [(None, True, None), ('sequential', True, None), ('sequential', False, None), ('parallel', True, None), ('parallel', False, None), ('parallel', True, 3)])
def test_enum_discrete_plate_shape_broadcasting_ok(enumerate_, expand, num_samples):
    if False:
        return 10

    def model():
        if False:
            for i in range(10):
                print('nop')
        x_plate = pyro.plate('x_plate', 10, 5, dim=-1)
        y_plate = pyro.plate('y_plate', 11, 6, dim=-2)
        with pyro.plate('num_particles', 50, dim=-3):
            with x_plate:
                b = pyro.sample('b', dist.Beta(torch.tensor(1.1), torch.tensor(1.1)))
            with y_plate:
                c = pyro.sample('c', dist.Bernoulli(0.5))
            with x_plate, y_plate:
                d = pyro.sample('d', dist.Bernoulli(b))
        if enumerate_ == 'parallel':
            if num_samples and expand:
                assert b.shape == (num_samples, 50, 1, 5)
                assert c.shape == (num_samples, 1, 50, 6, 1)
                assert d.shape == (num_samples, 1, num_samples, 50, 6, 5)
            elif num_samples and (not expand):
                assert b.shape == (num_samples, 50, 1, 5)
                assert c.shape == (num_samples, 1, 50, 6, 1)
                assert d.shape == (num_samples, 1, 1, 50, 6, 5)
            elif expand:
                assert b.shape == (50, 1, 5)
                assert c.shape == (2, 50, 6, 1)
                assert d.shape == (2, 1, 50, 6, 5)
            else:
                assert b.shape == (50, 1, 5)
                assert c.shape == (2, 1, 1, 1)
                assert d.shape == (2, 1, 1, 1, 1)
        elif enumerate_ == 'sequential':
            if expand:
                assert b.shape == (50, 1, 5)
                assert c.shape == (50, 6, 1)
                assert d.shape == (50, 6, 5)
            else:
                assert b.shape == (50, 1, 5)
                assert c.shape == (1, 1, 1)
                assert d.shape == (1, 1, 1)
        else:
            assert b.shape == (50, 1, 5)
            assert c.shape == (50, 6, 1)
            assert d.shape == (50, 6, 5)
    guide = config_enumerate(model, default=enumerate_, expand=expand, num_samples=num_samples)
    elbo = TraceEnum_ELBO(max_plate_nesting=3, strict_enumeration_warning=enumerate_ == 'parallel')
    assert_ok(model, guide, elbo)

@pytest.mark.parametrize('Elbo,expand', [(Trace_ELBO, False), (TraceGraph_ELBO, False), (TraceEnum_ELBO, False), (TraceEnum_ELBO, True)])
def test_dim_allocation_ok(Elbo, expand):
    if False:
        print('Hello World!')
    enumerate_ = Elbo is TraceEnum_ELBO

    def model():
        if False:
            while True:
                i = 10
        p = torch.tensor(0.5, requires_grad=True)
        with pyro.plate('plate_outer', 10, 5, dim=-3):
            x = pyro.sample('x', dist.Bernoulli(p))
            with pyro.plate('plate_inner_1', 11, 6):
                y = pyro.sample('y', dist.Bernoulli(p))
                with pyro.plate('plate_inner_2', 12, 7):
                    z = pyro.sample('z', dist.Bernoulli(p))
                    with pyro.plate('plate_inner_3', 13, 8):
                        q = pyro.sample('q', dist.Bernoulli(p))
        if enumerate_ and (not expand):
            assert x.shape == (1, 1, 1)
            assert y.shape == (1, 1, 1)
            assert z.shape == (1, 1, 1)
            assert q.shape == (1, 1, 1, 1)
        else:
            assert x.shape == (5, 1, 1)
            assert y.shape == (5, 1, 6)
            assert z.shape == (5, 7, 6)
            assert q.shape == (8, 5, 7, 6)
    guide = config_enumerate(model, 'sequential', expand=expand) if enumerate_ else model
    assert_ok(model, guide, Elbo(max_plate_nesting=4))

@pytest.mark.parametrize('Elbo,expand', [(Trace_ELBO, False), (TraceGraph_ELBO, False), (TraceEnum_ELBO, False), (TraceEnum_ELBO, True)])
def test_dim_allocation_error(Elbo, expand):
    if False:
        print('Hello World!')
    enumerate_ = Elbo is TraceEnum_ELBO

    def model():
        if False:
            print('Hello World!')
        p = torch.tensor(0.5, requires_grad=True)
        with pyro.plate('plate_outer', 10, 5, dim=-2):
            x = pyro.sample('x', dist.Bernoulli(p))
            with pyro.plate('plate_inner_1', 11, 6):
                y = pyro.sample('y', dist.Bernoulli(p))
                with pyro.plate('plate_inner_2', 12, 7, dim=-1):
                    pyro.sample('z', dist.Bernoulli(p))
        if enumerate_ and (not expand):
            assert x.shape == (1, 1)
            assert y.shape == (1, 1)
        else:
            assert x.shape == (5, 1)
            assert y.shape == (5, 6)
    guide = config_enumerate(model, expand=expand) if Elbo is TraceEnum_ELBO else model
    assert_error(model, guide, Elbo(), match='collide at dim=')

def test_enum_in_model_ok():
    if False:
        for i in range(10):
            print('nop')
    infer = {'enumerate': 'parallel'}

    def model():
        if False:
            for i in range(10):
                print('nop')
        p = pyro.param('p', torch.tensor(0.25))
        a = pyro.sample('a', dist.Bernoulli(p))
        b = pyro.sample('b', dist.Bernoulli(p + a / 2))
        c = pyro.sample('c', dist.Bernoulli(p + b / 2), infer=infer)
        d = pyro.sample('d', dist.Bernoulli(p + c / 2))
        e = pyro.sample('e', dist.Bernoulli(p + d / 2))
        f = pyro.sample('f', dist.Bernoulli(p + e / 2), infer=infer)
        g = pyro.sample('g', dist.Bernoulli(p + f / 2), obs=torch.tensor(0.0))
        assert a.shape == ()
        assert b.shape == (2,)
        assert c.shape == (2, 1, 1)
        assert d.shape == (2,)
        assert e.shape == (2, 1)
        assert f.shape == (2, 1, 1, 1)
        assert g.shape == ()

    def guide():
        if False:
            return 10
        p = pyro.param('p', torch.tensor(0.25))
        a = pyro.sample('a', dist.Bernoulli(p))
        b = pyro.sample('b', dist.Bernoulli(p + a / 2), infer=infer)
        d = pyro.sample('d', dist.Bernoulli(p + b / 2))
        e = pyro.sample('e', dist.Bernoulli(p + d / 2), infer=infer)
        assert a.shape == ()
        assert b.shape == (2,)
        assert d.shape == (2,)
        assert e.shape == (2, 1)
    assert_ok(model, guide, TraceEnum_ELBO(max_plate_nesting=0))

def test_enum_in_model_plate_ok():
    if False:
        return 10
    infer = {'enumerate': 'parallel'}

    def model():
        if False:
            i = 10
            return i + 15
        p = pyro.param('p', torch.tensor(0.25))
        a = pyro.sample('a', dist.Bernoulli(p))
        b = pyro.sample('b', dist.Bernoulli(p + a / 2))
        with pyro.plate('data', 3):
            c = pyro.sample('c', dist.Bernoulli(p + b / 2), infer=infer)
            d = pyro.sample('d', dist.Bernoulli(p + c / 2))
            e = pyro.sample('e', dist.Bernoulli(p + d / 2))
            f = pyro.sample('f', dist.Bernoulli(p + e / 2), infer=infer)
            g = pyro.sample('g', dist.Bernoulli(p + f / 2), obs=torch.zeros(3))
        assert a.shape == ()
        assert b.shape == (2, 1)
        assert c.shape == (2, 1, 1, 1)
        assert d.shape == (2, 3)
        assert e.shape == (2, 1, 1)
        assert f.shape == (2, 1, 1, 1, 1)
        assert g.shape == (3,)

    def guide():
        if False:
            while True:
                i = 10
        p = pyro.param('p', torch.tensor(0.25))
        a = pyro.sample('a', dist.Bernoulli(p))
        b = pyro.sample('b', dist.Bernoulli(p + a / 2), infer=infer)
        with pyro.plate('data', 3):
            d = pyro.sample('d', dist.Bernoulli(p + b / 2))
            e = pyro.sample('e', dist.Bernoulli(p + d / 2), infer=infer)
        assert a.shape == ()
        assert b.shape == (2, 1)
        assert d.shape == (2, 3)
        assert e.shape == (2, 1, 1)
    assert_ok(model, guide, TraceEnum_ELBO(max_plate_nesting=1))

def test_enum_sequential_in_model_error():
    if False:
        for i in range(10):
            print('nop')

    def model():
        if False:
            i = 10
            return i + 15
        p = pyro.param('p', torch.tensor(0.25))
        pyro.sample('a', dist.Bernoulli(p), infer={'enumerate': 'sequential'})

    def guide():
        if False:
            i = 10
            return i + 15
        pass
    assert_error(model, guide, TraceEnum_ELBO(max_plate_nesting=0), match='At site .*, model-side sequential enumeration is not implemented')

def test_enum_in_model_plate_reuse_ok():
    if False:
        i = 10
        return i + 15

    @config_enumerate
    def model():
        if False:
            for i in range(10):
                print('nop')
        p = pyro.param('p', torch.tensor([0.2, 0.8]))
        a = pyro.sample('a', dist.Bernoulli(0.3)).long()
        with pyro.plate('b_axis', 2):
            pyro.sample('b', dist.Bernoulli(p[a]), obs=torch.tensor([0.0, 1.0]))
        c = pyro.sample('c', dist.Bernoulli(0.3)).long()
        with pyro.plate('c_axis', 2):
            pyro.sample('d', dist.Bernoulli(p[c]), obs=torch.tensor([0.0, 0.0]))

    def guide():
        if False:
            print('Hello World!')
        pass
    assert_ok(model, guide, TraceEnum_ELBO(max_plate_nesting=1))

def test_enum_in_model_multi_scale_error():
    if False:
        for i in range(10):
            print('nop')

    @config_enumerate
    def model():
        if False:
            print('Hello World!')
        p = pyro.param('p', torch.tensor([0.2, 0.8]))
        x = pyro.sample('x', dist.Bernoulli(0.3)).long()
        with poutine.scale(scale=2.0):
            pyro.sample('y', dist.Bernoulli(p[x]), obs=torch.tensor(0.0))

    def guide():
        if False:
            for i in range(10):
                print('nop')
        pass
    assert_error(model, guide, TraceEnum_ELBO(max_plate_nesting=0), match='Expected all enumerated sample sites to share a common poutine.scale')

@pytest.mark.parametrize('use_vindex', [False, True])
def test_enum_in_model_diamond_error(use_vindex):
    if False:
        i = 10
        return i + 15
    data = torch.tensor([[0, 1], [0, 0]])

    @config_enumerate
    def model():
        if False:
            for i in range(10):
                print('nop')
        pyro.param('probs_a', torch.tensor([0.45, 0.55]))
        pyro.param('probs_b', torch.tensor([[0.6, 0.4], [0.4, 0.6]]))
        pyro.param('probs_c', torch.tensor([[0.75, 0.25], [0.55, 0.45]]))
        pyro.param('probs_d', torch.tensor([[[0.4, 0.6], [0.3, 0.7]], [[0.3, 0.7], [0.2, 0.8]]]))
        probs_a = pyro.param('probs_a')
        probs_b = pyro.param('probs_b')
        probs_c = pyro.param('probs_c')
        probs_d = pyro.param('probs_d')
        b_axis = pyro.plate('b_axis', 2, dim=-1)
        c_axis = pyro.plate('c_axis', 2, dim=-2)
        a = pyro.sample('a', dist.Categorical(probs_a))
        with b_axis:
            b = pyro.sample('b', dist.Categorical(probs_b[a]))
        with c_axis:
            c = pyro.sample('c', dist.Categorical(probs_c[a]))
        with b_axis, c_axis:
            if use_vindex:
                probs = Vindex(probs_d)[b, c]
            else:
                d_ind = torch.arange(2, dtype=torch.long)
                probs = probs_d[b.unsqueeze(-1), c.unsqueeze(-1), d_ind]
            pyro.sample('d', dist.Categorical(probs), obs=data)

    def guide():
        if False:
            return 10
        pass
    assert_error(model, guide, TraceEnum_ELBO(max_plate_nesting=2), match='Expected tree-structured plate nesting')

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_vectorized_num_particles(Elbo):
    if False:
        print('Hello World!')
    data = torch.ones(1000, 2)

    def model():
        if False:
            i = 10
            return i + 15
        with pyro.plate('components', 2):
            p = pyro.sample('p', dist.Beta(torch.tensor(1.1), torch.tensor(1.1)))
            assert p.shape == torch.Size((10, 1, 2))
            with pyro.plate('data', data.shape[0]):
                pyro.sample('obs', dist.Bernoulli(p), obs=data)

    def guide():
        if False:
            while True:
                i = 10
        with pyro.plate('components', 2):
            pyro.sample('p', dist.Beta(torch.tensor(1.1), torch.tensor(1.1)))
    pyro.clear_param_store()
    guide = config_enumerate(guide) if Elbo is TraceEnum_ELBO else guide
    assert_ok(model, guide, Elbo(num_particles=10, vectorize_particles=True, max_plate_nesting=2, strict_enumeration_warning=False))

@pytest.mark.parametrize('enumerate_,expand,num_samples', [(None, False, None), ('sequential', False, None), ('sequential', True, None), ('parallel', False, None), ('parallel', True, None), ('parallel', True, 3)])
@pytest.mark.parametrize('num_particles', [1, 50])
def test_enum_discrete_vectorized_num_particles(enumerate_, expand, num_samples, num_particles):
    if False:
        i = 10
        return i + 15

    @config_enumerate(default=enumerate_, expand=expand, num_samples=num_samples)
    def model():
        if False:
            while True:
                i = 10
        x_plate = pyro.plate('x_plate', 10, 5, dim=-1)
        y_plate = pyro.plate('y_plate', 11, 6, dim=-2)
        with x_plate:
            b = pyro.sample('b', dist.Beta(torch.tensor(1.1), torch.tensor(1.1)))
        with y_plate:
            c = pyro.sample('c', dist.Bernoulli(0.5))
        with x_plate, y_plate:
            d = pyro.sample('d', dist.Bernoulli(b))
        if num_particles > 1:
            if enumerate_ == 'parallel':
                if num_samples and expand:
                    assert b.shape == (num_samples, num_particles, 1, 5)
                    assert c.shape == (num_samples, 1, num_particles, 6, 1)
                    assert d.shape == (num_samples, 1, num_samples, num_particles, 6, 5)
                elif num_samples and (not expand):
                    assert b.shape == (num_samples, num_particles, 1, 5)
                    assert c.shape == (num_samples, 1, num_particles, 6, 1)
                    assert d.shape == (num_samples, 1, 1, num_particles, 6, 5)
                elif expand:
                    assert b.shape == (num_particles, 1, 5)
                    assert c.shape == (2, num_particles, 6, 1)
                    assert d.shape == (2, 1, num_particles, 6, 5)
                else:
                    assert b.shape == (num_particles, 1, 5)
                    assert c.shape == (2, 1, 1, 1)
                    assert d.shape == (2, 1, 1, 1, 1)
            elif enumerate_ == 'sequential':
                if expand:
                    assert b.shape == (num_particles, 1, 5)
                    assert c.shape == (num_particles, 6, 1)
                    assert d.shape == (num_particles, 6, 5)
                else:
                    assert b.shape == (num_particles, 1, 5)
                    assert c.shape == (1, 1, 1)
                    assert d.shape == (1, 1, 1)
            else:
                assert b.shape == (num_particles, 1, 5)
                assert c.shape == (num_particles, 6, 1)
                assert d.shape == (num_particles, 6, 5)
        elif enumerate_ == 'parallel':
            if num_samples and expand:
                assert b.shape == (num_samples, 1, 5)
                assert c.shape == (num_samples, 1, 6, 1)
                assert d.shape == (num_samples, 1, num_samples, 6, 5)
            elif num_samples and (not expand):
                assert b.shape == (num_samples, 1, 5)
                assert c.shape == (num_samples, 1, 6, 1)
                assert d.shape == (num_samples, 1, 1, 6, 5)
            elif expand:
                assert b.shape == (5,)
                assert c.shape == (2, 6, 1)
                assert d.shape == (2, 1, 6, 5)
            else:
                assert b.shape == (5,)
                assert c.shape == (2, 1, 1)
                assert d.shape == (2, 1, 1, 1)
        elif enumerate_ == 'sequential':
            if expand:
                assert b.shape == (5,)
                assert c.shape == (6, 1)
                assert d.shape == (6, 5)
            else:
                assert b.shape == (5,)
                assert c.shape == (1, 1)
                assert d.shape == (1, 1)
        else:
            assert b.shape == (5,)
            assert c.shape == (6, 1)
            assert d.shape == (6, 5)
    assert_ok(model, model, TraceEnum_ELBO(max_plate_nesting=2, num_particles=num_particles, vectorize_particles=True, strict_enumeration_warning=enumerate_ == 'parallel'))

def test_enum_recycling_chain():
    if False:
        return 10

    @config_enumerate
    def model():
        if False:
            return 10
        p = pyro.param('p', torch.tensor([[0.2, 0.8], [0.1, 0.9]]))
        x = 0
        for t in pyro.markov(range(100)):
            x = pyro.sample('x_{}'.format(t), dist.Categorical(p[x]))
            assert x.dim() <= 2

    def guide():
        if False:
            while True:
                i = 10
        pass
    assert_ok(model, guide, TraceEnum_ELBO(max_plate_nesting=0))

@pytest.mark.parametrize('use_vindex', [False, True])
@pytest.mark.parametrize('markov', [False, True])
def test_enum_recycling_dbn(markov, use_vindex):
    if False:
        for i in range(10):
            print('nop')

    @config_enumerate
    def model():
        if False:
            i = 10
            return i + 15
        p = pyro.param('p', torch.ones(3, 3))
        q = pyro.param('q', torch.ones(2))
        r = pyro.param('r', torch.ones(3, 2, 4))
        x = 0
        times = pyro.markov(range(100)) if markov else range(11)
        for t in times:
            x = pyro.sample('x_{}'.format(t), dist.Categorical(p[x]))
            y = pyro.sample('y_{}'.format(t), dist.Categorical(q))
            if use_vindex:
                probs = Vindex(r)[x, y]
            else:
                z_ind = torch.arange(4, dtype=torch.long)
                probs = r[x.unsqueeze(-1), y.unsqueeze(-1), z_ind]
            pyro.sample('z_{}'.format(t), dist.Categorical(probs), obs=torch.tensor(0.0))

    def guide():
        if False:
            print('Hello World!')
        pass
    assert_ok(model, guide, TraceEnum_ELBO(max_plate_nesting=0))

def test_enum_recycling_nested():
    if False:
        while True:
            i = 10

    @config_enumerate
    def model():
        if False:
            i = 10
            return i + 15
        p = pyro.param('p', torch.ones(3, 3))
        x = pyro.sample('x', dist.Categorical(p[0]))
        y = x
        for i in pyro.markov(range(10)):
            y = pyro.sample('y_{}'.format(i), dist.Categorical(p[y]))
            z = y
            for j in pyro.markov(range(10)):
                z = pyro.sample('z_{}_{}'.format(i, j), dist.Categorical(p[z]))

    def guide():
        if False:
            i = 10
            return i + 15
        pass
    assert_ok(model, guide, TraceEnum_ELBO(max_plate_nesting=0))

@pytest.mark.parametrize('use_vindex', [False, True])
def test_enum_recycling_grid(use_vindex):
    if False:
        i = 10
        return i + 15

    @config_enumerate
    def model():
        if False:
            for i in range(10):
                print('nop')
        p = pyro.param('p_leaf', torch.ones(2, 2, 2))
        x = defaultdict(lambda : torch.tensor(0))
        y_axis = pyro.markov(range(4), keep=True)
        for i in pyro.markov(range(4)):
            for j in y_axis:
                if use_vindex:
                    probs = Vindex(p)[x[i - 1, j], x[i, j - 1]]
                else:
                    ind = torch.arange(2, dtype=torch.long)
                    probs = p[x[i - 1, j].unsqueeze(-1), x[i, j - 1].unsqueeze(-1), ind]
                x[i, j] = pyro.sample('x_{}_{}'.format(i, j), dist.Categorical(probs))

    def guide():
        if False:
            return 10
        pass
    assert_ok(model, guide, TraceEnum_ELBO(max_plate_nesting=0))

def test_enum_recycling_reentrant():
    if False:
        return 10
    data = (True, False)
    for i in range(5):
        data = (data, data, False)

    @pyro.markov
    def model(data, state=0, address=''):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(data, bool):
            p = pyro.param('p_leaf', torch.ones(10))
            pyro.sample('leaf_{}'.format(address), dist.Bernoulli(p[state]), obs=torch.tensor(1.0 if data else 0.0))
        else:
            p = pyro.param('p_branch', torch.ones(10, 10))
            for (branch, letter) in zip(data, 'abcdefg'):
                next_state = pyro.sample('branch_{}'.format(address + letter), dist.Categorical(p[state]), infer={'enumerate': 'parallel'})
                model(branch, next_state, address + letter)

    def guide(data):
        if False:
            i = 10
            return i + 15
        pass
    assert_ok(model, guide, TraceEnum_ELBO(max_plate_nesting=0), data=data)

@pytest.mark.parametrize('history', [1, 2])
def test_enum_recycling_reentrant_history(history):
    if False:
        while True:
            i = 10
    data = (True, False)
    for i in range(5):
        data = (data, data, False)

    @pyro.markov(history=history)
    def model(data, state=0, address=''):
        if False:
            while True:
                i = 10
        if isinstance(data, bool):
            p = pyro.param('p_leaf', torch.ones(10))
            pyro.sample('leaf_{}'.format(address), dist.Bernoulli(p[state]), obs=torch.tensor(1.0 if data else 0.0))
        else:
            assert isinstance(data, tuple)
            p = pyro.param('p_branch', torch.ones(10, 10))
            for (branch, letter) in zip(data, 'abcdefg'):
                next_state = pyro.sample('branch_{}'.format(address + letter), dist.Categorical(p[state]), infer={'enumerate': 'parallel'})
                model(branch, next_state, address + letter)

    def guide(data):
        if False:
            while True:
                i = 10
        pass
    assert_ok(model, guide, TraceEnum_ELBO(max_plate_nesting=0), data=data)

def test_enum_recycling_mutual_recursion():
    if False:
        for i in range(10):
            print('nop')
    data = (True, False)
    for i in range(5):
        data = (data, data, False)

    def model_leaf(data, state=0, address=''):
        if False:
            i = 10
            return i + 15
        p = pyro.param('p_leaf', torch.ones(10))
        pyro.sample('leaf_{}'.format(address), dist.Bernoulli(p[state]), obs=torch.tensor(1.0 if data else 0.0))

    @pyro.markov
    def model1(data, state=0, address=''):
        if False:
            i = 10
            return i + 15
        if isinstance(data, bool):
            model_leaf(data, state, address)
        else:
            p = pyro.param('p_branch', torch.ones(10, 10))
            for (branch, letter) in zip(data, 'abcdefg'):
                next_state = pyro.sample('branch_{}'.format(address + letter), dist.Categorical(p[state]), infer={'enumerate': 'parallel'})
                model2(branch, next_state, address + letter)

    @pyro.markov
    def model2(data, state=0, address=''):
        if False:
            while True:
                i = 10
        if isinstance(data, bool):
            model_leaf(data, state, address)
        else:
            p = pyro.param('p_branch', torch.ones(10, 10))
            for (branch, letter) in zip(data, 'abcdefg'):
                next_state = pyro.sample('branch_{}'.format(address + letter), dist.Categorical(p[state]), infer={'enumerate': 'parallel'})
                model1(branch, next_state, address + letter)

    def guide(data):
        if False:
            while True:
                i = 10
        pass
    assert_ok(model1, guide, TraceEnum_ELBO(max_plate_nesting=0), data=data)

def test_enum_recycling_interleave():
    if False:
        return 10

    def model():
        if False:
            print('Hello World!')
        with pyro.markov() as m:
            with pyro.markov():
                with m:
                    pyro.sample('x', dist.Categorical(torch.ones(4)), infer={'enumerate': 'parallel'})

    def guide():
        if False:
            while True:
                i = 10
        pass
    assert_ok(model, guide, TraceEnum_ELBO(max_plate_nesting=0, strict_enumeration_warning=False))

def test_enum_recycling_plate():
    if False:
        while True:
            i = 10

    @config_enumerate
    def model():
        if False:
            print('Hello World!')
        p = pyro.param('p', torch.ones(3, 3))
        q = pyro.param('q', torch.tensor([0.5, 0.5]))
        plate_x = pyro.plate('plate_x', 2, dim=-1)
        plate_y = pyro.plate('plate_y', 3, dim=-1)
        plate_z = pyro.plate('plate_z', 4, dim=-2)
        a = pyro.sample('a', dist.Bernoulli(q[0])).long()
        w = 0
        for i in pyro.markov(range(5)):
            w = pyro.sample('w_{}'.format(i), dist.Categorical(p[w]))
        with plate_x:
            b = pyro.sample('b', dist.Bernoulli(q[a])).long()
            x = 0
            for i in pyro.markov(range(6)):
                x = pyro.sample('x_{}'.format(i), dist.Categorical(p[x]))
        with plate_y:
            c = pyro.sample('c', dist.Bernoulli(q[a])).long()
            y = 0
            for i in pyro.markov(range(7)):
                y = pyro.sample('y_{}'.format(i), dist.Categorical(p[y]))
        with plate_z:
            d = pyro.sample('d', dist.Bernoulli(q[a])).long()
            z = 0
            for i in pyro.markov(range(8)):
                z = pyro.sample('z_{}'.format(i), dist.Categorical(p[z]))
        with plate_x, plate_z:
            e = pyro.sample('e', dist.Bernoulli(q[b])).long()
            xz = 0
            for i in pyro.markov(range(9)):
                xz = pyro.sample('xz_{}'.format(i), dist.Categorical(p[xz]))
        return (a, b, c, d, e)

    def guide():
        if False:
            for i in range(10):
                print('nop')
        pass
    assert_ok(model, guide, TraceEnum_ELBO(max_plate_nesting=2))

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_factor_in_model_ok(Elbo):
    if False:
        while True:
            i = 10

    def model():
        if False:
            for i in range(10):
                print('nop')
        pyro.factor('f', torch.tensor(0.0))

    def guide():
        if False:
            for i in range(10):
                print('nop')
        pass
    elbo = Elbo(strict_enumeration_warning=False)
    assert_ok(model, guide, elbo)

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_factor_in_guide_error(Elbo):
    if False:
        print('Hello World!')

    def model():
        if False:
            i = 10
            return i + 15
        pass

    def guide():
        if False:
            while True:
                i = 10
        pyro.factor('f', torch.tensor(0.0))
    elbo = Elbo(strict_enumeration_warning=False)
    assert_error(model, guide, elbo, match='.*missing specification of has_rsample.*')

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
@pytest.mark.parametrize('has_rsample', [False, True])
def test_factor_in_guide_ok(Elbo, has_rsample):
    if False:
        return 10

    def model():
        if False:
            return 10
        pass

    def guide():
        if False:
            for i in range(10):
                print('nop')
        pyro.factor('f', torch.tensor(0.0), has_rsample=has_rsample)
    elbo = Elbo(strict_enumeration_warning=False)
    assert_ok(model, guide, elbo)

@pytest.mark.parametrize('history', [0, 1, 2, 3])
def test_markov_history(history):
    if False:
        for i in range(10):
            print('nop')

    @config_enumerate
    def model():
        if False:
            for i in range(10):
                print('nop')
        p = pyro.param('p', 0.25 * torch.ones(2, 2))
        q = pyro.param('q', 0.25 * torch.ones(2))
        x_prev = torch.tensor(0)
        x_curr = torch.tensor(0)
        for t in pyro.markov(range(10), history=history):
            probs = p[x_prev, x_curr]
            (x_prev, x_curr) = (x_curr, pyro.sample('x_{}'.format(t), dist.Bernoulli(probs)).long())
            pyro.sample('y_{}'.format(t), dist.Bernoulli(q[x_curr]), obs=torch.tensor(0.0))

    def guide():
        if False:
            return 10
        pass
    if history < 2:
        assert_error(model, guide, TraceEnum_ELBO(max_plate_nesting=0), match='Enumeration dim conflict')
    else:
        assert_ok(model, guide, TraceEnum_ELBO(max_plate_nesting=0))

def test_mean_field_ok():
    if False:
        print('Hello World!')

    def model():
        if False:
            for i in range(10):
                print('nop')
        x = pyro.sample('x', dist.Normal(0.0, 1.0))
        pyro.sample('y', dist.Normal(x, 1.0))

    def guide():
        if False:
            return 10
        loc = pyro.param('loc', torch.tensor(0.0))
        x = pyro.sample('x', dist.Normal(loc, 1.0))
        pyro.sample('y', dist.Normal(x, 1.0))
    assert_ok(model, guide, TraceMeanField_ELBO())

@pytest.mark.parametrize('mask', [True, False])
def test_mean_field_mask_ok(mask):
    if False:
        while True:
            i = 10

    def model():
        if False:
            print('Hello World!')
        x = pyro.sample('x', dist.Normal(0.0, 1.0).mask(mask))
        pyro.sample('y', dist.Normal(x, 1.0))

    def guide():
        if False:
            return 10
        loc = pyro.param('loc', torch.tensor(0.0))
        x = pyro.sample('x', dist.Normal(loc, 1.0).mask(mask))
        pyro.sample('y', dist.Normal(x, 1.0))
    assert_ok(model, guide, TraceMeanField_ELBO())

def test_mean_field_warn():
    if False:
        for i in range(10):
            print('nop')

    def model():
        if False:
            return 10
        x = pyro.sample('x', dist.Normal(0.0, 1.0))
        pyro.sample('y', dist.Normal(x, 1.0))

    def guide():
        if False:
            for i in range(10):
                print('nop')
        loc = pyro.param('loc', torch.tensor(0.0))
        y = pyro.sample('y', dist.Normal(loc, 1.0))
        pyro.sample('x', dist.Normal(y, 1.0))
    assert_warning(model, guide, TraceMeanField_ELBO())

def test_tail_adaptive_ok():
    if False:
        return 10

    def plateless_model():
        if False:
            for i in range(10):
                print('nop')
        pyro.sample('x', dist.Normal(0.0, 1.0))

    def plate_model():
        if False:
            while True:
                i = 10
        x = pyro.sample('x', dist.Normal(0.0, 1.0))
        with pyro.plate('observe_data'):
            pyro.sample('obs', dist.Normal(x, 1.0), obs=torch.arange(5).type_as(x))

    def rep_guide():
        if False:
            while True:
                i = 10
        pyro.sample('x', dist.Normal(0.0, 2.0))
    assert_ok(plateless_model, rep_guide, TraceTailAdaptive_ELBO(vectorize_particles=True, num_particles=2))
    assert_ok(plate_model, rep_guide, TraceTailAdaptive_ELBO(vectorize_particles=True, num_particles=2))

def test_tail_adaptive_error():
    if False:
        print('Hello World!')

    def plateless_model():
        if False:
            print('Hello World!')
        pyro.sample('x', dist.Normal(0.0, 1.0))

    def rep_guide():
        if False:
            for i in range(10):
                print('nop')
        pyro.sample('x', dist.Normal(0.0, 2.0))

    def nonrep_guide():
        if False:
            for i in range(10):
                print('nop')
        pyro.sample('x', fakes.NonreparameterizedNormal(0.0, 2.0))
    assert_error(plateless_model, rep_guide, TraceTailAdaptive_ELBO(vectorize_particles=False, num_particles=2))
    assert_error(plateless_model, nonrep_guide, TraceTailAdaptive_ELBO(vectorize_particles=True, num_particles=2))

def test_tail_adaptive_warning():
    if False:
        i = 10
        return i + 15

    def plateless_model():
        if False:
            while True:
                i = 10
        pyro.sample('x', dist.Normal(0.0, 1.0))

    def rep_guide():
        if False:
            return 10
        pyro.sample('x', dist.Normal(0.0, 2.0))
    assert_warning(plateless_model, rep_guide, TraceTailAdaptive_ELBO(vectorize_particles=True, num_particles=1))

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceMeanField_ELBO, EnergyDistance_prior, EnergyDistance_noprior])
def test_reparam_ok(Elbo):
    if False:
        while True:
            i = 10

    def model():
        if False:
            print('Hello World!')
        x = pyro.sample('x', dist.Normal(0.0, 1.0))
        pyro.sample('y', dist.Normal(x, 1.0), obs=torch.tensor(0.0))

    def guide():
        if False:
            print('Hello World!')
        loc = pyro.param('loc', torch.tensor(0.0))
        pyro.sample('x', dist.Normal(loc, 1.0))
    assert_ok(model, guide, Elbo())

@pytest.mark.parametrize('mask', [True, False, torch.tensor(True), torch.tensor(False)])
@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceMeanField_ELBO, EnergyDistance_prior, EnergyDistance_noprior])
def test_reparam_mask_ok(Elbo, mask):
    if False:
        return 10

    def model():
        if False:
            for i in range(10):
                print('nop')
        x = pyro.sample('x', dist.Normal(0.0, 1.0))
        with poutine.mask(mask=mask):
            pyro.sample('y', dist.Normal(x, 1.0), obs=torch.tensor(0.0))

    def guide():
        if False:
            for i in range(10):
                print('nop')
        loc = pyro.param('loc', torch.tensor(0.0))
        pyro.sample('x', dist.Normal(loc, 1.0))
    assert_ok(model, guide, Elbo())

@pytest.mark.parametrize('mask', [True, False, torch.tensor(True), torch.tensor(False), torch.tensor([False, True])])
@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceMeanField_ELBO, EnergyDistance_prior, EnergyDistance_noprior])
def test_reparam_mask_plate_ok(Elbo, mask):
    if False:
        for i in range(10):
            print('nop')
    data = torch.randn(2, 3).exp()
    data /= data.sum(-1, keepdim=True)

    def model():
        if False:
            while True:
                i = 10
        c = pyro.sample('c', dist.LogNormal(0.0, 1.0).expand([3]).to_event(1))
        with pyro.plate('data', len(data)), poutine.mask(mask=mask):
            pyro.sample('obs', dist.Dirichlet(c), obs=data)

    def guide():
        if False:
            for i in range(10):
                print('nop')
        loc = pyro.param('loc', torch.zeros(3))
        scale = pyro.param('scale', torch.ones(3), constraint=constraints.positive)
        pyro.sample('c', dist.LogNormal(loc, scale).to_event(1))
    assert_ok(model, guide, Elbo())

@pytest.mark.parametrize('num_particles', [1, 2])
@pytest.mark.parametrize('mask', [torch.tensor(True), torch.tensor(False), torch.tensor([True]), torch.tensor([False]), torch.tensor([False, True, False])])
@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceEnum_ELBO, TraceGraph_ELBO, TraceMeanField_ELBO])
def test_obs_mask_ok(Elbo, mask, num_particles):
    if False:
        i = 10
        return i + 15
    data = torch.tensor([7.0, 7.0, 7.0])

    def model():
        if False:
            while True:
                i = 10
        x = pyro.sample('x', dist.Normal(0.0, 1.0))
        with pyro.plate('plate', len(data)):
            y = pyro.sample('y', dist.Normal(x, 1.0), obs=data, obs_mask=mask)
            assert ((y == data) == mask).all()

    def guide():
        if False:
            for i in range(10):
                print('nop')
        loc = pyro.param('loc', torch.zeros(()))
        scale = pyro.param('scale', torch.ones(()), constraint=constraints.positive)
        x = pyro.sample('x', dist.Normal(loc, scale))
        with pyro.plate('plate', len(data)):
            with poutine.mask(mask=~mask):
                pyro.sample('y_unobserved', dist.Normal(x, 1.0))
    elbo = Elbo(num_particles=num_particles, vectorize_particles=True, strict_enumeration_warning=False)
    assert_ok(model, guide, elbo)

@pytest.mark.parametrize('num_particles', [1, 2])
@pytest.mark.parametrize('mask', [torch.tensor(True), torch.tensor(False), torch.tensor([True]), torch.tensor([False]), torch.tensor([False, True, True, False])])
@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceEnum_ELBO, TraceGraph_ELBO, TraceMeanField_ELBO])
def test_obs_mask_multivariate_ok(Elbo, mask, num_particles):
    if False:
        while True:
            i = 10
    data = torch.full((4, 3), 7.0)

    def model():
        if False:
            return 10
        x = pyro.sample('x', dist.MultivariateNormal(torch.zeros(3), torch.eye(3)))
        with pyro.plate('plate', len(data)):
            y = pyro.sample('y', dist.MultivariateNormal(x, torch.eye(3)), obs=data, obs_mask=mask)
            assert ((y == data).all(-1) == mask).all()

    def guide():
        if False:
            i = 10
            return i + 15
        loc = pyro.param('loc', torch.zeros(3))
        cov = pyro.param('cov', torch.eye(3), constraint=constraints.positive_definite)
        x = pyro.sample('x', dist.MultivariateNormal(loc, cov))
        with pyro.plate('plate', len(data)):
            with poutine.mask(mask=~mask):
                pyro.sample('y_unobserved', dist.MultivariateNormal(x, torch.eye(3)))
    elbo = Elbo(num_particles=num_particles, vectorize_particles=True, strict_enumeration_warning=False)
    assert_ok(model, guide, elbo)

@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceEnum_ELBO, TraceGraph_ELBO, TraceMeanField_ELBO])
def test_obs_mask_multivariate_error(Elbo):
    if False:
        while True:
            i = 10
    data = torch.full((3, 2), 7.0)
    mask = torch.tensor([[False, False], [False, True], [True, False]])

    def model():
        if False:
            i = 10
            return i + 15
        x = pyro.sample('x', dist.MultivariateNormal(torch.zeros(2), torch.eye(2)))
        with pyro.plate('plate', len(data)):
            pyro.sample('y', dist.MultivariateNormal(x, torch.eye(2)), obs=data, obs_mask=mask)

    def guide():
        if False:
            while True:
                i = 10
        loc = pyro.param('loc', torch.zeros(2))
        x = pyro.sample('x', dist.MultivariateNormal(loc, torch.eye(2)))
        with pyro.plate('plate', len(data)):
            with poutine.mask(mask=~mask):
                pyro.sample('y_unobserved', dist.MultivariateNormal(x, torch.eye(2)))
    elbo = Elbo(strict_enumeration_warning=False)
    assert_error(model, guide, elbo, match='Invalid obs_mask shape')

@pytest.mark.parametrize('scale', [1, 0.1, torch.tensor(0.5)])
@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceMeanField_ELBO, EnergyDistance_prior, EnergyDistance_noprior])
def test_reparam_scale_ok(Elbo, scale):
    if False:
        print('Hello World!')

    def model():
        if False:
            while True:
                i = 10
        x = pyro.sample('x', dist.Normal(0.0, 1.0))
        with poutine.scale(scale=scale):
            pyro.sample('y', dist.Normal(x, 1.0), obs=torch.tensor(0.0))

    def guide():
        if False:
            return 10
        loc = pyro.param('loc', torch.tensor(0.0))
        pyro.sample('x', dist.Normal(loc, 1.0))
    assert_ok(model, guide, Elbo())

@pytest.mark.parametrize('scale', [1, 0.1, torch.tensor(0.5), torch.tensor([0.1, 0.9])])
@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceMeanField_ELBO, EnergyDistance_prior, EnergyDistance_noprior])
def test_reparam_scale_plate_ok(Elbo, scale):
    if False:
        i = 10
        return i + 15
    data = torch.randn(2, 3).exp()
    data /= data.sum(-1, keepdim=True)

    def model():
        if False:
            i = 10
            return i + 15
        c = pyro.sample('c', dist.LogNormal(0.0, 1.0).expand([3]).to_event(1))
        with pyro.plate('data', len(data)), poutine.scale(scale=scale):
            pyro.sample('obs', dist.Dirichlet(c), obs=data)

    def guide():
        if False:
            i = 10
            return i + 15
        loc = pyro.param('loc', torch.zeros(3))
        scale = pyro.param('scale', torch.ones(3), constraint=constraints.positive)
        pyro.sample('c', dist.LogNormal(loc, scale).to_event(1))
    assert_ok(model, guide, Elbo())

@pytest.mark.parametrize('Elbo', [EnergyDistance_prior, EnergyDistance_noprior])
def test_no_log_prob_ok(Elbo):
    if False:
        for i in range(10):
            print('nop')

    def model(data):
        if False:
            for i in range(10):
                print('nop')
        loc = pyro.sample('loc', dist.Normal(0, 1))
        scale = pyro.sample('scale', dist.LogNormal(0, 1))
        with pyro.plate('data', len(data)):
            pyro.sample('obs', dist.Stable(1.5, 0.5, scale, loc), obs=data)

    def guide(data):
        if False:
            i = 10
            return i + 15
        map_loc = pyro.param('map_loc', torch.tensor(0.0))
        map_scale = pyro.param('map_scale', torch.tensor(1.0), constraint=constraints.positive)
        pyro.sample('loc', dist.Delta(map_loc))
        pyro.sample('scale', dist.Delta(map_scale))
    data = torch.randn(10)
    assert_ok(model, guide, Elbo(), data=data)

def test_reparam_stable():
    if False:
        i = 10
        return i + 15

    @poutine.reparam(config={'z': LatentStableReparam()})
    def model():
        if False:
            print('Hello World!')
        stability = pyro.sample('stability', dist.Uniform(0.0, 2.0))
        skew = pyro.sample('skew', dist.Uniform(-1.0, 1.0))
        y = pyro.sample('z', dist.Stable(stability, skew))
        pyro.sample('x', dist.Poisson(y.abs()), obs=torch.tensor(1.0))

    def guide():
        if False:
            i = 10
            return i + 15
        pyro.sample('stability', dist.Delta(torch.tensor(1.5)))
        pyro.sample('skew', dist.Delta(torch.tensor(0.0)))
        pyro.sample('z_uniform', dist.Delta(torch.tensor(0.1)))
        pyro.sample('z_exponential', dist.Delta(torch.tensor(1.0)))
    assert_ok(model, guide, Trace_ELBO())

@pytest.mark.stage('funsor')
@pytest.mark.parametrize('num_particles', [1, 2])
def test_collapse_normal_normal(num_particles):
    if False:
        print('Hello World!')
    pytest.importorskip('funsor')
    data = torch.tensor(0.0)

    def model():
        if False:
            print('Hello World!')
        x = pyro.sample('x', dist.Normal(0.0, 1.0))
        with poutine.collapse():
            y = pyro.sample('y', dist.Normal(x, 1.0))
            pyro.sample('z', dist.Normal(y, 1.0), obs=data)

    def guide():
        if False:
            for i in range(10):
                print('nop')
        loc = pyro.param('loc', torch.tensor(0.0))
        scale = pyro.param('scale', torch.tensor(1.0), constraint=constraints.positive)
        pyro.sample('x', dist.Normal(loc, scale))
    elbo = Trace_ELBO(num_particles=num_particles, vectorize_particles=True)
    assert_ok(model, guide, elbo)

@pytest.mark.stage('funsor')
@pytest.mark.parametrize('num_particles', [1, 2])
def test_collapse_normal_normal_plate(num_particles):
    if False:
        while True:
            i = 10
    pytest.importorskip('funsor')
    data = torch.randn(5)

    def model():
        if False:
            while True:
                i = 10
        x = pyro.sample('x', dist.Normal(0.0, 1.0))
        with poutine.collapse():
            y = pyro.sample('y', dist.Normal(x, 1.0))
            with pyro.plate('data', len(data), dim=-1):
                pyro.sample('z', dist.Normal(y, 1.0), obs=data)

    def guide():
        if False:
            while True:
                i = 10
        loc = pyro.param('loc', torch.tensor(0.0))
        scale = pyro.param('scale', torch.tensor(1.0), constraint=constraints.positive)
        pyro.sample('x', dist.Normal(loc, scale))
    elbo = Trace_ELBO(num_particles=num_particles, vectorize_particles=True, max_plate_nesting=1)
    assert_ok(model, guide, elbo)

@pytest.mark.stage('funsor')
@pytest.mark.parametrize('num_particles', [1, 2])
def test_collapse_normal_plate_normal(num_particles):
    if False:
        while True:
            i = 10
    pytest.importorskip('funsor')
    data = torch.randn(5)

    def model():
        if False:
            while True:
                i = 10
        x = pyro.sample('x', dist.Normal(0.0, 1.0))
        with poutine.collapse():
            with pyro.plate('data', len(data), dim=-1):
                y = pyro.sample('y', dist.Normal(x, 1.0))
                pyro.sample('z', dist.Normal(y, 1.0), obs=data)

    def guide():
        if False:
            i = 10
            return i + 15
        loc = pyro.param('loc', torch.tensor(0.0))
        scale = pyro.param('scale', torch.tensor(1.0), constraint=constraints.positive)
        pyro.sample('x', dist.Normal(loc, scale))
    elbo = Trace_ELBO(num_particles=num_particles, vectorize_particles=True, max_plate_nesting=1)
    assert_ok(model, guide, elbo)

@pytest.mark.stage('funsor')
@pytest.mark.parametrize('num_particles', [1, 2])
def test_collapse_beta_bernoulli(num_particles):
    if False:
        print('Hello World!')
    pytest.importorskip('funsor')
    data = torch.tensor(0.0)

    def model():
        if False:
            i = 10
            return i + 15
        c = pyro.sample('c', dist.Gamma(1, 1))
        with poutine.collapse():
            probs = pyro.sample('probs', dist.Beta(c, 2))
            pyro.sample('obs', dist.Bernoulli(probs), obs=data)

    def guide():
        if False:
            i = 10
            return i + 15
        a = pyro.param('a', torch.tensor(1.0), constraint=constraints.positive)
        b = pyro.param('b', torch.tensor(1.0), constraint=constraints.positive)
        pyro.sample('c', dist.Gamma(a, b))
    elbo = Trace_ELBO(num_particles=num_particles, vectorize_particles=True)
    assert_ok(model, guide, elbo)

@pytest.mark.stage('funsor')
@pytest.mark.parametrize('num_particles', [1, 2])
def test_collapse_beta_binomial(num_particles):
    if False:
        print('Hello World!')
    pytest.importorskip('funsor')
    data = torch.tensor(5.0)

    def model():
        if False:
            print('Hello World!')
        c = pyro.sample('c', dist.Gamma(1, 1))
        with poutine.collapse():
            probs = pyro.sample('probs', dist.Beta(c, 2))
            pyro.sample('obs', dist.Binomial(10, probs), obs=data)

    def guide():
        if False:
            print('Hello World!')
        a = pyro.param('a', torch.tensor(1.0), constraint=constraints.positive)
        b = pyro.param('b', torch.tensor(1.0), constraint=constraints.positive)
        pyro.sample('c', dist.Gamma(a, b))
    elbo = Trace_ELBO(num_particles=num_particles, vectorize_particles=True)
    assert_ok(model, guide, elbo)

@pytest.mark.stage('funsor')
@pytest.mark.parametrize('num_particles', [1, 2])
def test_collapse_beta_binomial_plate(num_particles):
    if False:
        for i in range(10):
            print('nop')
    pytest.importorskip('funsor')
    data = torch.tensor([0.0, 1.0, 5.0, 5.0])

    def model():
        if False:
            return 10
        c = pyro.sample('c', dist.Gamma(1, 1))
        with poutine.collapse():
            probs = pyro.sample('probs', dist.Beta(c, 2))
            with pyro.plate('plate', len(data)):
                pyro.sample('obs', dist.Binomial(10, probs), obs=data)

    def guide():
        if False:
            return 10
        a = pyro.param('a', torch.tensor(1.0), constraint=constraints.positive)
        b = pyro.param('b', torch.tensor(1.0), constraint=constraints.positive)
        pyro.sample('c', dist.Gamma(a, b))
    elbo = Trace_ELBO(num_particles=num_particles, vectorize_particles=True, max_plate_nesting=1)
    assert_ok(model, guide, elbo)

@pytest.mark.stage('funsor')
@pytest.mark.parametrize('num_particles', [1, 2])
def test_collapse_barrier(num_particles):
    if False:
        for i in range(10):
            print('nop')
    pytest.importorskip('funsor')
    data = torch.tensor([0.0, 1.0, 5.0, 5.0])

    def model():
        if False:
            for i in range(10):
                print('nop')
        with poutine.collapse():
            z = pyro.sample('z_init', dist.Normal(0, 1))
            for (t, x) in enumerate(data):
                z = pyro.sample('z_{}'.format(t), dist.Normal(z, 1))
                pyro.sample('x_t{}'.format(t), dist.Normal(z, 1), obs=x)
                z = pyro.barrier(z)
                z = torch.sigmoid(z)
        return z

    def guide():
        if False:
            return 10
        pass
    elbo = Trace_ELBO(num_particles=num_particles, vectorize_particles=True)
    assert_ok(model, guide, elbo)

def test_ordered_logistic_plate():
    if False:
        i = 10
        return i + 15
    N = 5
    K = 4
    data = (K * torch.rand(N)).long().float()

    def model():
        if False:
            return 10
        predictor = pyro.sample('predictor', dist.Normal(0.0, 1.0).expand([N]).to_event(1))
        cutpoints = pyro.sample('cutpoints', dist.Normal(0.0, 1.0).expand([K - 1]).to_event(1))
        cutpoints = torch.sort(cutpoints, dim=-1).values
        with pyro.plate('obs_plate', N):
            pyro.sample('obs', dist.OrderedLogistic(predictor, cutpoints), obs=data)

    def guide():
        if False:
            for i in range(10):
                print('nop')
        pred_mu = pyro.param('pred_mu', torch.zeros(N))
        pred_std = pyro.param('pred_std', torch.ones(N))
        cp_mu = pyro.param('cp_mu', torch.zeros(K - 1))
        cp_std = pyro.param('cp_std', torch.ones(K - 1))
        pyro.sample('predictor', dist.Normal(pred_mu, pred_std).to_event(1))
        pyro.sample('cutpoints', dist.Normal(cp_mu, cp_std).to_event(1))
    assert_ok(model, guide, Trace_ELBO())