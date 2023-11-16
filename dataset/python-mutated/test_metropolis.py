import warnings
import arviz as az
import numpy as np
import numpy.testing as npt
import pytensor
import pytest
import pymc as pm
from pymc.step_methods.metropolis import BinaryGibbsMetropolis, CategoricalGibbsMetropolis, DEMetropolis, DEMetropolisZ, Metropolis, MultivariateNormalProposal, NormalProposal
from pymc.testing import fast_unstable_sampling_mode
from tests import sampler_fixtures as sf
from tests.helpers import RVsAssignmentStepsTester, StepMethodTester
from tests.models import mv_simple, mv_simple_discrete, simple_categorical

class TestMetropolisUniform(sf.MetropolisFixture, sf.UniformFixture):
    n_samples = 50000
    tune = 10000
    burn = 0
    chains = 4
    min_n_eff = 10000
    rtol = 0.1
    atol = 0.05

class TestMetropolis:

    def test_proposal_choice(self):
        if False:
            while True:
                i = 10
        with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
            (_, model, _) = mv_simple()
            with model:
                initial_point = model.initial_point()
                initial_point_size = sum((initial_point[n.name].size for n in model.value_vars))
                s = np.ones(initial_point_size)
                sampler = Metropolis(S=s)
                assert isinstance(sampler.proposal_dist, NormalProposal)
                s = np.diag(s)
                sampler = Metropolis(S=s)
                assert isinstance(sampler.proposal_dist, MultivariateNormalProposal)
                s[0, 0] = -s[0, 0]
                with pytest.raises(np.linalg.LinAlgError):
                    sampler = Metropolis(S=s)

    def test_mv_proposal(self):
        if False:
            while True:
                i = 10
        np.random.seed(42)
        cov = np.random.randn(5, 5)
        cov = cov.dot(cov.T)
        prop = MultivariateNormalProposal(cov)
        samples = np.array([prop() for _ in range(10000)])
        npt.assert_allclose(np.cov(samples.T), cov, rtol=0.2)

    def test_tuning_reset(self):
        if False:
            while True:
                i = 10
        'Re-use of the step method instance with cores=1 must not leak tuning information between chains.'
        with pm.Model() as pmodel:
            D = 3
            pm.Normal('n', 0, 2, size=(D,))
            idata = pm.sample(tune=600, draws=500, step=Metropolis(tune=True, scaling=0.1), cores=1, chains=3, discard_tuned_samples=False)
        for c in idata.posterior.chain:
            assert idata.warmup_sample_stats['scaling'].sel(chain=c).values[0] == 0.1
            tuned = idata.warmup_sample_stats['scaling'].sel(chain=c).values[-1]
            assert tuned != 0.1
            np.testing.assert_array_equal(idata.sample_stats['scaling'].sel(chain=c).values, tuned)

    @pytest.mark.parametrize('batched_dist', (pm.Binomial.dist(n=5, p=0.9), pm.Binomial.dist(n=np.arange(40) + 1, p=np.linspace(0.1, 0.9, 40), shape=(40,)), pm.Binomial.dist(n=(np.arange(20) + 1)[::-1], p=np.linspace(0.1, 0.9, 20), shape=(2, 20)), pm.Dirichlet.dist(a=np.ones(3) * (np.arange(40) + 1)[:, None], shape=(40, 3)), pm.Dirichlet.dist(a=np.ones(3) * (np.arange(20) + 1)[:, None], shape=(2, 20, 3))))
    def test_elemwise_update(self, batched_dist):
        if False:
            while True:
                i = 10
        with pm.Model() as m:
            m.register_rv(batched_dist, name='batched_dist')
            step = pm.Metropolis([batched_dist])
            assert step.elemwise_update == (batched_dist.ndim > 0)
            trace = pm.sample(draws=1000, chains=2, step=step, random_seed=428)
        assert az.rhat(trace).max()['batched_dist'].values < 1.1
        assert az.ess(trace).min()['batched_dist'].values > 50

    def test_elemwise_update_different_scales(self):
        if False:
            return 10
        mu = [1, 2, 3, 4, 5, 100, 1000, 10000]
        with pm.Model() as m:
            x = pm.Poisson('x', mu=mu)
            step = pm.Metropolis([x])
            trace = pm.sample(draws=1000, chains=2, step=step, random_seed=128).posterior
        np.testing.assert_allclose(trace['x'].mean(('draw', 'chain')), mu, rtol=0.1)
        np.testing.assert_allclose(trace['x'].var(('draw', 'chain')), mu, rtol=0.2)

    def test_multinomial_no_elemwise_update(self):
        if False:
            for i in range(10):
                print('nop')
        with pm.Model() as m:
            batched_dist = pm.Multinomial('batched_dist', n=5, p=np.ones(4) / 4, shape=(10, 4))
            with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
                step = pm.Metropolis([batched_dist])
                assert not step.elemwise_update

class TestDEMetropolis:

    def test_demcmc_tune_parameter(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests that validity of the tune setting is checked'
        with pm.Model() as model:
            pm.Normal('n', mu=0, sigma=1, size=(2, 3))
            step = DEMetropolis()
            assert step.tune == 'scaling'
            step = DEMetropolis(tune=None)
            assert step.tune is None
            step = DEMetropolis(tune='scaling')
            assert step.tune == 'scaling'
            step = DEMetropolis(tune='lambda')
            assert step.tune == 'lambda'
            with pytest.raises(ValueError):
                DEMetropolis(tune='foo')

class TestDEMetropolisZ:

    def test_tuning_lambda_sequential(self):
        if False:
            while True:
                i = 10
        with pm.Model() as pmodel:
            pm.Normal('n', 0, 2, size=(3,))
            idata = pm.sample(tune=1000, draws=500, step=DEMetropolisZ(tune='lambda', lamb=0.92), cores=1, chains=3, discard_tuned_samples=False)
        for c in idata.posterior.chain:
            assert idata.warmup_sample_stats['lambda'].sel(chain=c).values[0] == 0.92
            tuned = idata.warmup_sample_stats['lambda'].sel(chain=c).values[-1]
            assert tuned != 0.92
            np.testing.assert_array_equal(idata.sample_stats['lambda'].sel(chain=c).values, tuned)

    def test_tuning_epsilon_parallel(self):
        if False:
            i = 10
            return i + 15
        with pm.Model() as pmodel:
            pm.Normal('n', 0, 2, size=(3,))
            idata = pm.sample(tune=1000, draws=500, step=DEMetropolisZ(tune='scaling', scaling=0.002), cores=2, chains=2, discard_tuned_samples=False)
        for c in idata.posterior.chain:
            assert idata.warmup_sample_stats['scaling'].sel(chain=c).values[0] == 0.002
            tuned = idata.warmup_sample_stats['scaling'].sel(chain=c).values[-1]
            assert tuned != 0.002
            np.testing.assert_array_equal(idata.sample_stats['scaling'].sel(chain=c).values, tuned)

    def test_tuning_none(self):
        if False:
            i = 10
            return i + 15
        with pm.Model() as pmodel:
            pm.Normal('n', 0, 2, size=(3,))
            idata = pm.sample(tune=1000, draws=500, step=DEMetropolisZ(tune=None), cores=1, chains=2, discard_tuned_samples=False)
        for c in idata.posterior.chain:
            assert len(set(idata.warmup_sample_stats['lambda'].sel(chain=c).values)) == 1
            assert len(set(idata.warmup_sample_stats['scaling'].sel(chain=c).values)) == 1

    def test_tuning_reset(self):
        if False:
            print('Hello World!')
        'Re-use of the step method instance with cores=1 must not leak tuning information between chains.'
        with pm.Model() as pmodel:
            D = 3
            pm.Normal('n', 0, 2, size=(D,))
            idata = pm.sample(tune=1000, draws=500, step=DEMetropolisZ(tune='scaling', scaling=0.002), cores=1, chains=3, discard_tuned_samples=False, random_seed=1)
        for c in idata.posterior.chain:
            warmup = idata.warmup_sample_stats['scaling'].sel(chain=c).values
            assert warmup[0] == 0.002
            assert warmup[-1] != 0.002
            samples = idata.warmup_posterior['n'].sel(chain=c).values
            for d in range(D):
                var_start = np.var(samples[:50, d])
                var_end = np.var(samples[-100:, d])
                assert var_start < 0.1 * var_end

    def test_tune_drop_fraction(self):
        if False:
            for i in range(10):
                print('nop')
        tune = 300
        tune_drop_fraction = 0.85
        draws = 200
        with pm.Model() as pmodel:
            pm.Normal('n', 0, 2, size=(3,))
            step = DEMetropolisZ(tune_drop_fraction=tune_drop_fraction)
            idata = pm.sample(tune=tune, draws=draws, step=step, cores=1, chains=1, discard_tuned_samples=False)
            assert len(idata.warmup_posterior.draw) == tune
            assert len(idata.posterior.draw) == draws
            assert len(step._history) == tune - tune * tune_drop_fraction + draws

    @pytest.mark.parametrize('variable,has_grad,outcome', [('n', True, 1), ('n', False, 1), ('b', True, 0), ('b', False, 0)])
    def test_competence(self, variable, has_grad, outcome):
        if False:
            i = 10
            return i + 15
        with pm.Model() as pmodel:
            pm.Normal('n', 0, 2, size=(3,))
            pm.Binomial('b', n=2, p=0.3)
        assert DEMetropolisZ.competence(pmodel[variable], has_grad=has_grad) == outcome

    @pytest.mark.parametrize('tune_setting', ['foo', True, False])
    def test_invalid_tune(self, tune_setting):
        if False:
            print('Hello World!')
        with pm.Model() as pmodel:
            pm.Normal('n', 0, 2, size=(3,))
            with pytest.raises(ValueError):
                DEMetropolisZ(tune=tune_setting)

    def test_custom_proposal_dist(self):
        if False:
            return 10
        with pm.Model() as pmodel:
            D = 3
            pm.Normal('n', 0, 2, size=(D,))
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', '.*number of samples.*', UserWarning)
                trace = pm.sample(tune=100, draws=50, step=DEMetropolisZ(proposal_dist=NormalProposal), cores=1, chains=3, discard_tuned_samples=False)

class TestStepMetropolis(StepMethodTester):

    def test_step_discrete(self):
        if False:
            return 10
        (start, model, (mu, C)) = mv_simple_discrete()
        unc = np.diag(C) ** 0.5
        check = (('x', np.mean, mu, unc / 10.0), ('x', np.std, unc, unc / 10.0))
        with model:
            step = Metropolis(S=C, proposal_dist=MultivariateNormalProposal)
            idata = pm.sample(tune=1000, draws=2000, chains=1, step=step, initvals=start, model=model, random_seed=1)
            self.check_stat(check, idata)
            self.check_stat_dtype(idata, step)

    @pytest.mark.parametrize('proposal', ['uniform', 'proportional'])
    def test_step_categorical(self, proposal):
        if False:
            print('Hello World!')
        (start, model, (mu, C)) = simple_categorical()
        unc = C ** 0.5
        check = (('x', np.mean, mu, unc / 10.0), ('x', np.std, unc, unc / 10.0))
        with model:
            step = CategoricalGibbsMetropolis([model.x], proposal=proposal)
            idata = pm.sample(tune=1000, draws=2000, chains=1, step=step, initvals=start, model=model, random_seed=1)
            self.check_stat(check, idata)
            self.check_stat_dtype(idata, step)

    @pytest.mark.parametrize('step_fn, draws', [(lambda C, _: Metropolis(S=C, proposal_dist=MultivariateNormalProposal, blocked=True), 4000)], ids=str)
    def test_step_continuous(self, step_fn, draws):
        if False:
            i = 10
            return i + 15
        self.step_continuous(step_fn, draws)

class TestRVsAssignmentMetropolis(RVsAssignmentStepsTester):

    @pytest.mark.parametrize('step, step_kwargs', [(BinaryGibbsMetropolis, {}), (CategoricalGibbsMetropolis, {})])
    def test_discrete_steps(self, step, step_kwargs):
        if False:
            print('Hello World!')
        with pm.Model() as m:
            d1 = pm.Bernoulli('d1', p=0.5)
            d2 = pm.Bernoulli('d2', p=0.5)
            with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
                assert [m.rvs_to_values[d1]] == step([d1], **step_kwargs).vars
            assert {m.rvs_to_values[d1], m.rvs_to_values[d2]} == set(step([d1, d2], **step_kwargs).vars)

    @pytest.mark.parametrize('step, step_kwargs', [(Metropolis, {}), (DEMetropolis, {}), (DEMetropolisZ, {})])
    def test_continuous_steps(self, step, step_kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.continuous_steps(step, step_kwargs)