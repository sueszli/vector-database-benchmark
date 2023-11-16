import numpy as np
import pytest
import pymc as pm
from pymc.distributions.shape_utils import change_dist_size

class TestCensored:

    @pytest.mark.parametrize('censored', (False, True))
    def test_censored_workflow(self, censored):
        if False:
            while True:
                i = 10
        rng = np.random.default_rng(1234)
        size = 500
        true_mu = 13.0
        true_sigma = 5.0
        low = 3.0
        high = 16.0
        data = rng.normal(true_mu, true_sigma, size)
        data[data <= low] = low
        data[data >= high] = high
        rng = 17092021
        with pm.Model() as m:
            mu = pm.Normal('mu', mu=(high - low) / 2 + low, sigma=(high - low) / 2.0, initval='moment')
            sigma = pm.HalfNormal('sigma', sigma=(high - low) / 2.0, initval='moment')
            observed = pm.Censored('observed', pm.Normal.dist(mu=mu, sigma=sigma), lower=low if censored else None, upper=high if censored else None, observed=data)
            prior_pred = pm.sample_prior_predictive(random_seed=rng)
            posterior = pm.sample(tune=500, draws=500, random_seed=rng)
            posterior_pred = pm.sample_posterior_predictive(posterior, random_seed=rng)
        expected = True if censored else False
        assert (9 < prior_pred.prior_predictive.mean() < 10) == expected
        assert (13 < posterior.posterior['mu'].mean() < 14) == expected
        assert (4.5 < posterior.posterior['sigma'].mean() < 5.5) == expected
        assert (12 < posterior_pred.posterior_predictive.mean() < 13) == expected

    def test_censored_invalid_dist(self):
        if False:
            print('Hello World!')
        with pm.Model():
            invalid_dist = pm.Normal
            with pytest.raises(ValueError, match='Censoring dist must be a distribution created via the'):
                x = pm.Censored('x', invalid_dist, lower=None, upper=None)
        with pm.Model():
            mv_dist = pm.Dirichlet.dist(a=[1, 1, 1])
            with pytest.raises(NotImplementedError, match='Censoring of multivariate distributions has not been implemented yet'):
                x = pm.Censored('x', mv_dist, lower=None, upper=None)
        with pm.Model():
            registered_dist = pm.Normal('dist')
            with pytest.raises(ValueError, match='The dist dist was already registered in the current model'):
                x = pm.Censored('x', registered_dist, lower=None, upper=None)

    def test_change_dist_size(self):
        if False:
            while True:
                i = 10
        base_dist = pm.Censored.dist(pm.Normal.dist(), -1, 1, size=(3, 2))
        new_dist = change_dist_size(base_dist, (4,))
        assert new_dist.eval().shape == (4,)
        new_dist = change_dist_size(base_dist, (4,), expand=True)
        assert new_dist.eval().shape == (4, 3, 2)

    def test_dist_broadcasted_by_lower_upper(self):
        if False:
            for i in range(10):
                print('nop')
        x = pm.Censored.dist(pm.Normal.dist(), lower=np.zeros((2,)), upper=None)
        assert tuple(x.owner.inputs[0].shape.eval()) == (2,)
        x = pm.Censored.dist(pm.Normal.dist(), lower=np.zeros((2,)), upper=np.zeros((4, 2)))
        assert tuple(x.owner.inputs[0].shape.eval()) == (4, 2)
        x = pm.Censored.dist(pm.Normal.dist(size=(3, 4, 2)), lower=np.zeros((2,)), upper=np.zeros((4, 2)))
        assert tuple(x.owner.inputs[0].shape.eval()) == (3, 4, 2)