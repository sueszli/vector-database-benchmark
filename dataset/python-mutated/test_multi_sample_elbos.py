import pytest
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import RenyiELBO
from tests.common import assert_close

def check_elbo(model, guide, Elbo):
    if False:
        for i in range(10):
            print('nop')
    elbo = Elbo(num_particles=2, vectorize_particles=False)
    pyro.set_rng_seed(123)
    loss1 = elbo.loss(model, guide)
    pyro.set_rng_seed(123)
    loss2 = elbo.loss_and_grads(model, guide)
    assert_close(loss1, loss2)
    elbo = Elbo(num_particles=10000, vectorize_particles=True)
    loss1 = elbo.loss(model, guide)
    loss2 = elbo.loss_and_grads(model, guide)
    assert_close(loss1, loss2, atol=0.1)

@pytest.mark.parametrize('Elbo', [RenyiELBO])
def test_inner_outer(Elbo):
    if False:
        return 10
    data = torch.randn(2, 3)

    def model():
        if False:
            return 10
        with pyro.plate('outer', 3, dim=-1):
            x = pyro.sample('x', dist.Normal(0, 1))
            with pyro.plate('inner', 2, dim=-2):
                pyro.sample('y', dist.Normal(x, 1), obs=data)

    def guide():
        if False:
            print('Hello World!')
        with pyro.plate('outer', 3, dim=-1):
            pyro.sample('x', dist.Normal(1, 1))
    check_elbo(model, guide, Elbo)

@pytest.mark.parametrize('Elbo', [RenyiELBO])
def test_outer_inner(Elbo):
    if False:
        print('Hello World!')
    data = torch.randn(2, 3)

    def model():
        if False:
            i = 10
            return i + 15
        with pyro.plate('outer', 2, dim=-2):
            x = pyro.sample('x', dist.Normal(0, 1))
            with pyro.plate('inner', 3, dim=-1):
                pyro.sample('y', dist.Normal(x, 1), obs=data)

    def guide():
        if False:
            for i in range(10):
                print('nop')
        with pyro.plate('outer', 2, dim=-2):
            pyro.sample('x', dist.Normal(1, 1))
    check_elbo(model, guide, Elbo)