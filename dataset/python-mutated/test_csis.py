import pytest
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
from tests.common import assert_equal, assert_not_equal

def model(observations={'y1': 0, 'y2': 0}):
    if False:
        i = 10
        return i + 15
    x = pyro.sample('x', dist.Normal(torch.tensor(0.0), torch.tensor(5 ** 0.5)))
    pyro.sample('y1', dist.Normal(x, torch.tensor(2 ** 0.5)), obs=observations['y1'])
    pyro.sample('y2', dist.Normal(x, torch.tensor(2 ** 0.5)), obs=observations['y2'])
    return x

class Guide(nn.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)
        self.std = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, observations={'y1': 0, 'y2': 0}):
        if False:
            for i in range(10):
                print('nop')
        pyro.module('guide', self)
        summed_obs = observations['y1'] + observations['y2']
        mean = self.linear(summed_obs.view(1, 1))[0, 0]
        pyro.sample('x', dist.Normal(mean, self.std))

@pytest.mark.init(rng_seed=7)
def test_csis_sampling():
    if False:
        return 10
    pyro.clear_param_store()
    guide = Guide()
    csis = pyro.infer.CSIS(model, guide, pyro.optim.Adam({}), num_inference_samples=500)
    posterior = csis.run({'y1': torch.tensor(-1.0), 'y2': torch.tensor(1.0)})
    assert_equal(len(posterior.exec_traces), 500)
    marginal = pyro.infer.EmpiricalMarginal(posterior, 'x')
    assert_equal(marginal.mean, torch.tensor(0.0), prec=0.1)

@pytest.mark.init(rng_seed=7)
def test_csis_parameter_update():
    if False:
        i = 10
        return i + 15
    pyro.clear_param_store()
    guide = Guide()
    initial_parameters = {k: v.item() for (k, v) in guide.named_parameters()}
    csis = pyro.infer.CSIS(model, guide, pyro.optim.Adam({'lr': 0.01}))
    csis.step()
    updated_parameters = {k: v.item() for (k, v) in guide.named_parameters()}
    for (k, init_v) in initial_parameters.items():
        assert_not_equal(init_v, updated_parameters[k])

@pytest.mark.init(rng_seed=7)
def test_csis_validation_batch():
    if False:
        for i in range(10):
            print('nop')
    pyro.clear_param_store()
    guide = Guide()
    csis = pyro.infer.CSIS(model, guide, pyro.optim.Adam({}), validation_batch_size=5)
    init_loss_1 = csis.validation_loss()
    init_loss_2 = csis.validation_loss()
    csis.step()
    next_loss = csis.validation_loss()
    assert_equal(init_loss_1, init_loss_2)
    assert_not_equal(init_loss_1, next_loss)