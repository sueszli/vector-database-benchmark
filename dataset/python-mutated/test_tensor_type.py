import pytest
import scipy.stats as sp
import torch
import pyro.distributions as dist
from tests.common import assert_equal

@pytest.fixture()
def test_data():
    if False:
        for i in range(10):
            print('nop')
    return torch.DoubleTensor([0.4])

@pytest.fixture()
def alpha():
    if False:
        i = 10
        return i + 15
    '\n    alpha parameter for the Beta distribution.\n    '
    return torch.DoubleTensor([2.4])

@pytest.fixture()
def beta():
    if False:
        return 10
    '\n    beta parameter for the Beta distribution.\n    '
    return torch.DoubleTensor([3.7])

@pytest.fixture()
def float_test_data(test_data):
    if False:
        while True:
            i = 10
    return torch.FloatTensor(test_data.detach().cpu().numpy())

@pytest.fixture()
def float_alpha(alpha):
    if False:
        return 10
    return torch.FloatTensor(alpha.detach().cpu().numpy())

@pytest.fixture()
def float_beta(beta):
    if False:
        for i in range(10):
            print('nop')
    return torch.FloatTensor(beta.detach().cpu().numpy())

def test_double_type(test_data, alpha, beta):
    if False:
        while True:
            i = 10
    log_px_torch = dist.Beta(alpha, beta).log_prob(test_data).data
    assert isinstance(log_px_torch, torch.DoubleTensor)
    log_px_val = log_px_torch.numpy()
    log_px_np = sp.beta.logpdf(test_data.detach().cpu().numpy(), alpha.detach().cpu().numpy(), beta.detach().cpu().numpy())
    assert_equal(log_px_val, log_px_np, prec=0.0001)

def test_float_type(float_test_data, float_alpha, float_beta, test_data, alpha, beta):
    if False:
        while True:
            i = 10
    log_px_torch = dist.Beta(float_alpha, float_beta).log_prob(float_test_data).data
    assert isinstance(log_px_torch, torch.FloatTensor)
    log_px_val = log_px_torch.numpy()
    log_px_np = sp.beta.logpdf(test_data.detach().cpu().numpy(), alpha.detach().cpu().numpy(), beta.detach().cpu().numpy())
    assert_equal(log_px_val, log_px_np, prec=0.0001)

@pytest.mark.xfail(reason='https://github.com/pytorch/pytorch/issues/43138#issuecomment-677804776')
def test_conflicting_types(test_data, float_alpha, beta):
    if False:
        return 10
    with pytest.raises((TypeError, RuntimeError)):
        dist.Beta(float_alpha, beta).log_prob(test_data)