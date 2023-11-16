import numpy as np
import pytest
import pymc as pm

@pytest.mark.parametrize('distribution, lower, upper, init_guess, fixed_params, mass_below_lower', [(pm.Gamma, 0.1, 0.4, {'alpha': 1, 'beta': 10}, {}, None), (pm.Normal, 155, 180, {'mu': 170, 'sigma': 3}, {}, None), (pm.StudentT, 0.1, 0.4, {'mu': 10, 'sigma': 3}, {'nu': 7}, None), (pm.StudentT, 0, 1, {'mu': 5, 'sigma': 2, 'nu': 7}, {}, None), (pm.Exponential, 0, 1, {'lam': 1}, {}, 0), (pm.HalfNormal, 0, 1, {'sigma': 1}, {}, 0), (pm.Binomial, 0, 8, {'p': 0.5}, {'n': 10}, None), (pm.Poisson, 1, 15, {'mu': 10}, {}, None), (pm.Poisson, 19, 41, {'mu': 30}, {}, None)])
@pytest.mark.parametrize('mass', [0.5, 0.75, 0.95])
def test_find_constrained_prior(distribution, lower, upper, init_guess, fixed_params, mass, mass_below_lower):
    if False:
        return 10
    opt_params = pm.find_constrained_prior(distribution, lower=lower, upper=upper, mass=mass, init_guess=init_guess, fixed_params=fixed_params, mass_below_lower=mass_below_lower)
    opt_distribution = distribution.dist(**opt_params)
    mass_in_interval = (pm.math.exp(pm.logcdf(opt_distribution, upper)) - pm.math.exp(pm.logcdf(opt_distribution, lower))).eval()
    assert np.abs(mass_in_interval - mass) <= 1e-05

@pytest.mark.parametrize('distribution, lower, upper, init_guess, fixed_params', [(pm.Gamma, 0.1, 0.4, {'alpha': 1}, {'beta': 10}), (pm.Exponential, 0.1, 1, {'lam': 1}, {}), (pm.Binomial, 0, 2, {'p': 0.8}, {'n': 10})])
def test_find_constrained_prior_error_too_large(distribution, lower, upper, init_guess, fixed_params):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError, match='Optimization of parameters failed.\nOptimization termination details:\n'):
        pm.find_constrained_prior(distribution, lower=lower, upper=upper, mass=0.95, init_guess=init_guess, fixed_params=fixed_params)

def test_find_constrained_prior_input_errors():
    if False:
        i = 10
        return i + 15
    with pytest.raises(TypeError, match='required positional argument'):
        pm.find_constrained_prior(pm.StudentT, lower=0.1, upper=0.4, mass=0.95, init_guess={'mu': 170, 'sigma': 3})
    with pytest.raises(AssertionError, match='has to be between 0.01 and 0.99'):
        pm.find_constrained_prior(pm.StudentT, lower=0.1, upper=0.4, mass=0.995, init_guess={'mu': 170, 'sigma': 3}, fixed_params={'nu': 7})
    with pytest.raises(AssertionError, match='has to be between 0.01 and 0.99'):
        pm.find_constrained_prior(pm.StudentT, lower=0.1, upper=0.4, mass=0.005, init_guess={'mu': 170, 'sigma': 3}, fixed_params={'nu': 7})
    with pytest.raises(NotImplementedError, match='does not work with non-scalar parameters yet'):
        pm.find_constrained_prior(pm.MvNormal, lower=0, upper=1, mass=0.95, init_guess={'mu': 5, 'cov': np.asarray([[1, 0.2], [0.2, 1]])})