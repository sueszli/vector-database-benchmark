import numpy as np
import numpy.testing as npt
import pytest
from pymc import ConstantData, Model, Normal, sample

@pytest.mark.parametrize('nuts_sampler', ['pymc', 'nutpie', 'blackjax', 'numpyro'])
def test_external_nuts_sampler(recwarn, nuts_sampler):
    if False:
        for i in range(10):
            print('nop')
    if nuts_sampler != 'pymc':
        pytest.importorskip(nuts_sampler)
    with Model():
        x = Normal('x', 100, 5)
        y = ConstantData('y', [1, 2, 3, 4])
        ConstantData('z', [100, 190, 310, 405])
        Normal('L', mu=x, sigma=0.1, observed=y)
        kwargs = dict(nuts_sampler=nuts_sampler, random_seed=123, chains=2, tune=500, draws=500, progressbar=False, initvals={'x': 0.0})
        idata1 = sample(**kwargs)
        idata2 = sample(**kwargs)
    warns = {(warn.category, warn.message.args[0]) for warn in recwarn if warn.category is not FutureWarning}
    expected = set()
    if nuts_sampler == 'nutpie':
        expected.add((UserWarning, '`initvals` are currently not passed to nutpie sampler. Use `init_mean` kwarg following nutpie specification instead.'))
    assert warns == expected
    assert 'y' in idata1.constant_data
    assert 'z' in idata1.constant_data
    assert 'L' in idata1.observed_data
    assert idata1.posterior.chain.size == 2
    assert idata1.posterior.draw.size == 500
    np.testing.assert_array_equal(idata1.posterior.x, idata2.posterior.x)

def test_step_args():
    if False:
        for i in range(10):
            print('nop')
    with Model() as model:
        a = Normal('a')
        idata = sample(nuts_sampler='numpyro', target_accept=0.5, nuts={'max_treedepth': 10}, random_seed=1410)
    npt.assert_almost_equal(idata.sample_stats.acceptance_rate.mean(), 0.5, decimal=1)