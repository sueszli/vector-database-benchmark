import pytest
import torch
from pyro.distributions import TruncatedPolyaGamma
from tests.common import assert_close

@pytest.mark.parametrize('batch_shape', [(), (3,), (2, 1)])
def test_polya_gamma(batch_shape, num_points=20000):
    if False:
        while True:
            i = 10
    d = TruncatedPolyaGamma(prototype=torch.ones(1)).expand(batch_shape)
    x = torch.linspace(1e-06, d.truncation_point, num_points).expand(batch_shape + (num_points,))
    prob = d.truncation_point / num_points * torch.logsumexp(d.log_prob(x), dim=-1).exp()
    assert_close(prob, torch.tensor(1.0).expand(batch_shape), rtol=0.0001)
    z = d.sample(sample_shape=(3000,))
    mean = z.mean(-1)
    assert_close(mean, torch.tensor(0.25).expand(batch_shape), rtol=0.07)