import pytest
import torch
from pyro.ops.ssm_gp import MaternKernel
from tests.common import assert_equal

@pytest.mark.parametrize('num_gps', [1, 2, 3])
@pytest.mark.parametrize('nu', [0.5, 1.5, 2.5])
def test_matern_kernel(num_gps, nu):
    if False:
        return 10
    mk = MaternKernel(nu=nu, num_gps=num_gps, length_scale_init=0.1 + torch.rand(num_gps))
    dt = torch.rand(1).item()
    forward = mk.transition_matrix(dt)
    backward = mk.transition_matrix(-dt)
    forward_backward = torch.matmul(forward, backward)
    eye = torch.eye(mk.state_dim).unsqueeze(0).expand(num_gps, mk.state_dim, mk.state_dim)
    assert_equal(forward_backward, eye)
    torch.linalg.cholesky(mk.stationary_covariance())
    torch.linalg.cholesky(mk.process_covariance(forward))
    nudge = mk.transition_matrix(torch.tensor([1e-09]))
    assert_equal(nudge, eye)