import torch
from torch import nn
from torch.distributions import Normal
from kornia.augmentation.random_generator import DistributionWithMapper
from kornia.testing import assert_close

class TestDistMapper:

    def test_mapper(self):
        if False:
            while True:
                i = 10
        _ = torch.manual_seed(0)
        dist = DistributionWithMapper(Normal(0.0, 1.0), map_fn=nn.Sigmoid())
        out = dist.rsample((8,))
        exp = torch.tensor([0.8236, 0.4272, 0.1017, 0.6384, 0.2527, 0.198, 0.5995, 0.698])
        assert_close(out, exp, rtol=0.0001, atol=0.0001)