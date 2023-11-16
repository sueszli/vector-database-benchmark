import math
import os
import pytest
import torch
from torch import optim
from pyro.distributions import VonMises, VonMises3D
from pyro.distributions.testing.gof import auto_goodness_of_fit
from tests.common import TEST_FAILURE_RATE, skipif_param, xfail_if_not_implemented

def _eval_poly(y, coef):
    if False:
        print('Hello World!')
    coef = list(coef)
    result = coef.pop()
    while coef:
        result = coef.pop() + y * result
    return result
_I0_COEF_SMALL = [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.0360768, 0.0045813]
_I0_COEF_LARGE = [0.39894228, 0.01328592, 0.00225319, -0.00157565, 0.00916281, -0.02057706, 0.02635537, -0.01647633, 0.00392377]
_I1_COEF_SMALL = [0.5, 0.87890594, 0.51498869, 0.15084934, 0.02658733, 0.00301532, 0.00032411]
_I1_COEF_LARGE = [0.39894228, -0.03988024, -0.00362018, 0.00163801, -0.01031555, 0.02282967, -0.02895312, 0.01787654, -0.00420059]
_COEF_SMALL = [_I0_COEF_SMALL, _I1_COEF_SMALL]
_COEF_LARGE = [_I0_COEF_LARGE, _I1_COEF_LARGE]

def _log_modified_bessel_fn(x, order=0):
    if False:
        print('Hello World!')
    '\n    Returns ``log(I_order(x))`` for ``x > 0``,\n    where `order` is either 0 or 1.\n    '
    assert order == 0 or order == 1
    y = (x / 3.75).pow(2)
    small = _eval_poly(y, _COEF_SMALL[order])
    if order == 1:
        small = x.abs() * small
    small = small.log()
    y = 3.75 / x
    large = x - 0.5 * x.log() + _eval_poly(y, _COEF_LARGE[order]).log()
    mask = x < 3.75
    result = large
    if mask.any():
        result[mask] = small[mask]
    return result

def _fit_params_from_samples(samples, n_iter):
    if False:
        print('Hello World!')
    assert samples.dim() == 1
    samples_count = samples.size(0)
    samples_cs = samples.cos().sum()
    samples_ss = samples.sin().sum()
    mu = torch.atan2(samples_ss / samples_count, samples_cs / samples_count)
    samples_r = (samples_cs ** 2 + samples_ss ** 2).sqrt() / samples_count
    kappa = (samples_r * 2 - samples_r ** 3) / (1 - samples_r ** 2)
    lr = 0.01
    kappa.requires_grad = True
    bfgs = optim.LBFGS([kappa], lr=lr)

    def bfgs_closure():
        if False:
            print('Hello World!')
        bfgs.zero_grad()
        obj = _log_modified_bessel_fn(kappa, order=1) - _log_modified_bessel_fn(kappa, order=0)
        obj = (obj - samples_r.log()).abs()
        obj.backward()
        return obj
    for i in range(n_iter):
        bfgs.step(bfgs_closure)
    return (mu, kappa.detach())

@pytest.mark.parametrize('loc', [-math.pi / 2.0, 0.0, math.pi / 2.0])
@pytest.mark.parametrize('concentration', [skipif_param(0.01, condition='CUDA_TEST' in os.environ, reason='low precision.'), 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
def test_sample(loc, concentration, n_samples=int(1000000.0), n_iter=50):
    if False:
        print('Hello World!')
    prob = VonMises(loc, concentration)
    samples = prob.sample((n_samples,))
    (mu, kappa) = _fit_params_from_samples(samples, n_iter=n_iter)
    assert abs(loc - mu) < 0.1
    assert abs(concentration - kappa) < concentration * 0.1

@pytest.mark.parametrize('concentration', [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
def test_log_prob_normalized(concentration):
    if False:
        return 10
    grid = torch.arange(0.0, 2 * math.pi, 0.0001)
    prob = VonMises(0.0, concentration).log_prob(grid).exp()
    norm = prob.mean().item() * 2 * math.pi
    assert abs(norm - 1) < 0.001, norm

@pytest.mark.parametrize('loc', [-math.pi / 2.0, 0.0, math.pi / 2.0])
@pytest.mark.parametrize('concentration', [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
def test_von_mises_gof(loc, concentration):
    if False:
        return 10
    d = VonMises(loc, concentration)
    samples = d.sample(torch.Size([100000]))
    probs = d.log_prob(samples).exp()
    gof = auto_goodness_of_fit(samples, probs, dim=1)
    assert gof > TEST_FAILURE_RATE

@pytest.mark.parametrize('scale', [0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0])
def test_von_mises_3d(scale):
    if False:
        return 10
    concentration = torch.randn(3)
    concentration = concentration * (scale / concentration.norm(2))
    num_samples = 100000
    samples = torch.randn(num_samples, 3)
    samples = samples / samples.norm(2, dim=-1, keepdim=True)
    d = VonMises3D(concentration, validate_args=True)
    actual_total = d.log_prob(samples).exp().mean()
    expected_total = 1 / (4 * math.pi)
    ratio = actual_total / expected_total
    assert torch.abs(ratio - 1) < 0.01, ratio

@pytest.mark.parametrize('scale', [0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0])
def test_von_mises_3d_gof(scale):
    if False:
        for i in range(10):
            print('nop')
    concentration = torch.randn(3)
    concentration = concentration * (scale / concentration.norm(2))
    d = VonMises3D(concentration, validate_args=True)
    with xfail_if_not_implemented():
        samples = d.sample(torch.Size([2000]))
    probs = d.log_prob(samples).exp()
    gof = auto_goodness_of_fit(samples, probs, dim=2)
    assert gof > TEST_FAILURE_RATE