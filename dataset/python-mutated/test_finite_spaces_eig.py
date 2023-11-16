from contextlib import ExitStack
import pytest
import torch
import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.contrib.oed.eig import donsker_varadhan_eig, lfire_eig, marginal_eig, marginal_likelihood_eig, nmc_eig, posterior_eig, vnmc_eig
from pyro.contrib.util import iter_plates_to_shape
from tests.common import assert_equal

@pytest.fixture
def finite_space_model():
    if False:
        for i in range(10):
            print('nop')

    def model(design):
        if False:
            print('Hello World!')
        batch_shape = design.shape
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)
            theta = pyro.sample('theta', dist.Bernoulli(0.4).expand(batch_shape))
            y = pyro.sample('y', dist.Bernoulli((design + theta) / 2.0))
            return y
    return model

@pytest.fixture
def one_point_design():
    if False:
        return 10
    return torch.tensor(0.5)

@pytest.fixture
def true_eig():
    if False:
        for i in range(10):
            print('nop')
    return torch.tensor(0.12580366909478014)

def posterior_guide(y_dict, design, observation_labels, target_labels):
    if False:
        for i in range(10):
            print('nop')
    y = torch.cat(list(y_dict.values()), dim=-1)
    (a, b) = (pyro.param('a', torch.tensor(0.0)), pyro.param('b', torch.tensor(0.0)))
    pyro.sample('theta', dist.Bernoulli(logits=a + b * y))

def marginal_guide(design, observation_labels, target_labels):
    if False:
        for i in range(10):
            print('nop')
    logit_p = pyro.param('logit_p', torch.tensor(0.0))
    pyro.sample('y', dist.Bernoulli(logits=logit_p))

def likelihood_guide(theta_dict, design, observation_labels, target_labels):
    if False:
        while True:
            i = 10
    theta = torch.cat(list(theta_dict.values()), dim=-1)
    (a, b) = (pyro.param('a', torch.tensor(0.0)), pyro.param('b', torch.tensor(0.0)))
    pyro.sample('y', dist.Bernoulli(logits=a + b * theta))

def make_lfire_classifier(n_theta_samples):
    if False:
        for i in range(10):
            print('nop')

    def lfire_classifier(design, trace, observation_labels, target_labels):
        if False:
            while True:
                i = 10
        y_dict = {l: trace.nodes[l]['value'] for l in observation_labels}
        y = torch.cat(list(y_dict.values()), dim=-1)
        (a, b) = (pyro.param('a', torch.zeros(n_theta_samples)), pyro.param('b', torch.zeros(n_theta_samples)))
        return a + b * y
    return lfire_classifier

def dv_critic(design, trace, observation_labels, target_labels):
    if False:
        print('Hello World!')
    y_dict = {l: trace.nodes[l]['value'] for l in observation_labels}
    y = torch.cat(list(y_dict.values()), dim=-1)
    theta_dict = {l: trace.nodes[l]['value'] for l in target_labels}
    theta = torch.cat(list(theta_dict.values()), dim=-1)
    w_y = pyro.param('w_y', torch.tensor(0.0))
    w_theta = pyro.param('w_theta', torch.tensor(0.0))
    w_ytheta = pyro.param('w_ytheta', torch.tensor(0.0))
    return y * w_y + theta * w_theta + y * theta * w_ytheta

def test_posterior_finite_space_model(finite_space_model, one_point_design, true_eig):
    if False:
        while True:
            i = 10
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    posterior_eig(finite_space_model, one_point_design, 'y', 'theta', num_samples=10, num_steps=250, guide=posterior_guide, optim=optim.Adam({'lr': 0.1}))
    estimated_eig = posterior_eig(finite_space_model, one_point_design, 'y', 'theta', num_samples=10, num_steps=250, guide=posterior_guide, optim=optim.Adam({'lr': 0.01}), final_num_samples=1000)
    assert_equal(estimated_eig, true_eig, prec=0.01)

def test_marginal_finite_space_model(finite_space_model, one_point_design, true_eig):
    if False:
        while True:
            i = 10
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    marginal_eig(finite_space_model, one_point_design, 'y', 'theta', num_samples=10, num_steps=250, guide=marginal_guide, optim=optim.Adam({'lr': 0.1}))
    estimated_eig = marginal_eig(finite_space_model, one_point_design, 'y', 'theta', num_samples=10, num_steps=250, guide=marginal_guide, optim=optim.Adam({'lr': 0.01}), final_num_samples=1000)
    assert_equal(estimated_eig, true_eig, prec=0.01)

def test_marginal_likelihood_finite_space_model(finite_space_model, one_point_design, true_eig):
    if False:
        i = 10
        return i + 15
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    marginal_likelihood_eig(finite_space_model, one_point_design, 'y', 'theta', num_samples=10, num_steps=250, marginal_guide=marginal_guide, cond_guide=likelihood_guide, optim=optim.Adam({'lr': 0.1}))
    estimated_eig = marginal_likelihood_eig(finite_space_model, one_point_design, 'y', 'theta', num_samples=10, num_steps=250, marginal_guide=marginal_guide, cond_guide=likelihood_guide, optim=optim.Adam({'lr': 0.01}), final_num_samples=1000)
    assert_equal(estimated_eig, true_eig, prec=0.01)

@pytest.mark.xfail(reason='Bernoullis are not reparametrizable and current VNMC implementation assumes reparametrization')
def test_vnmc_finite_space_model(finite_space_model, one_point_design, true_eig):
    if False:
        print('Hello World!')
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    vnmc_eig(finite_space_model, one_point_design, 'y', 'theta', num_samples=[9, 3], num_steps=250, guide=posterior_guide, optim=optim.Adam({'lr': 0.1}))
    estimated_eig = vnmc_eig(finite_space_model, one_point_design, 'y', 'theta', num_samples=[9, 3], num_steps=250, guide=posterior_guide, optim=optim.Adam({'lr': 0.01}), final_num_samples=[1000, 100])
    assert_equal(estimated_eig, true_eig, prec=0.01)

def test_nmc_eig_finite_space_model(finite_space_model, one_point_design, true_eig):
    if False:
        i = 10
        return i + 15
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    estimated_eig = nmc_eig(finite_space_model, one_point_design, 'y', 'theta', M=40, N=40 * 40)
    assert_equal(estimated_eig, true_eig, prec=0.01)

def test_lfire_finite_space_model(finite_space_model, one_point_design, true_eig):
    if False:
        i = 10
        return i + 15
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    estimated_eig = lfire_eig(finite_space_model, one_point_design, 'y', 'theta', num_y_samples=5, num_theta_samples=50, num_steps=1000, classifier=make_lfire_classifier(50), optim=optim.Adam({'lr': 0.0025}), final_num_samples=500)
    assert_equal(estimated_eig, true_eig, prec=0.01)

def test_dv_finite_space_model(finite_space_model, one_point_design, true_eig):
    if False:
        print('Hello World!')
    pyro.set_rng_seed(42)
    pyro.clear_param_store()
    donsker_varadhan_eig(finite_space_model, one_point_design, 'y', 'theta', num_samples=100, num_steps=250, T=dv_critic, optim=optim.Adam({'lr': 0.1}))
    estimated_eig = donsker_varadhan_eig(finite_space_model, one_point_design, 'y', 'theta', num_samples=100, num_steps=250, T=dv_critic, optim=optim.Adam({'lr': 0.01}), final_num_samples=2000)
    assert_equal(estimated_eig, true_eig, prec=0.01)