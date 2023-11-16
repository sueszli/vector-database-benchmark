import functools as ft
import numpy as np
import pytensor.tensor as pt
import pytest
import pymc as pm
from pymc.variational import opvi
from pymc.variational.approximations import Empirical, EmpiricalGroup, FullRank, FullRankGroup, MeanField, MeanFieldGroup
from tests.helpers import not_raises

def test_discrete_not_allowed():
    if False:
        i = 10
        return i + 15
    mu_true = np.array([-2, 0, 2])
    z_true = np.random.randint(len(mu_true), size=100)
    y = np.random.normal(mu_true[z_true], np.ones_like(z_true))
    with pm.Model():
        mu = pm.Normal('mu', mu=0, sigma=10, size=3)
        z = pm.Categorical('z', p=pt.ones(3) / 3, size=len(y))
        pm.Normal('y_obs', mu=mu[z], sigma=1.0, observed=y)
        with pytest.raises(opvi.ParametrizationError, match='Discrete variables'):
            pm.fit(n=1)

@pytest.fixture(scope='module')
def three_var_model():
    if False:
        for i in range(10):
            print('nop')
    with pm.Model() as model:
        pm.HalfNormal('one', size=(10, 2))
        pm.Normal('two', size=(10,))
        pm.Normal('three', size=(10, 1, 2))
    return model

@pytest.mark.parametrize('raises, vfam, type_, kw', [(not_raises(), 'mean_field', MeanFieldGroup, {}), (not_raises(), 'mf', MeanFieldGroup, {}), (not_raises(), 'full_rank', FullRankGroup, {}), (not_raises(), 'fr', FullRankGroup, {}), (not_raises(), 'FR', FullRankGroup, {}), (pytest.raises(ValueError, match='Need `trace` or `size`'), 'empirical', EmpiricalGroup, {}), (not_raises(), 'empirical', EmpiricalGroup, {'size': 100})])
def test_group_api_vfam(three_var_model, raises, vfam, type_, kw):
    if False:
        for i in range(10):
            print('nop')
    with three_var_model, raises:
        g = opvi.Group([three_var_model.one], vfam, **kw)
        assert isinstance(g, type_)
        assert not hasattr(g, '_kwargs')

@pytest.mark.parametrize('raises, params, type_, kw, formula', [(not_raises(), dict(mu=np.ones((10, 2), 'float32'), rho=np.ones((10, 2), 'float32')), MeanFieldGroup, {}, None), (not_raises(), dict(mu=np.ones((10, 2), 'float32'), L_tril=np.ones(FullRankGroup.get_param_spec_for(d=np.prod((10, 2)))['L_tril'], 'float32')), FullRankGroup, {}, None)])
def test_group_api_params(three_var_model, raises, params, type_, kw, formula):
    if False:
        i = 10
        return i + 15
    with three_var_model, raises:
        g = opvi.Group([three_var_model.one], params=params, **kw)
        assert isinstance(g, type_)
        if g.has_logq:
            logq = g.logq
            logq = g.set_size_and_deterministic(logq, 1, 0)
            logq.eval()

@pytest.mark.parametrize('gcls, approx, kw', [(MeanFieldGroup, MeanField, {}), (FullRankGroup, FullRank, {}), (EmpiricalGroup, Empirical, {'size': 100})])
def test_single_group_shortcuts(three_var_model, approx, kw, gcls):
    if False:
        for i in range(10):
            print('nop')
    with three_var_model:
        a = approx(**kw)
    assert isinstance(a, opvi.Approximation)
    assert len(a.groups) == 1
    assert isinstance(a.groups[0], gcls)

@pytest.mark.parametrize(['raises', 'grouping'], [(not_raises(), [(MeanFieldGroup, None)]), (not_raises(), [(FullRankGroup, None), (MeanFieldGroup, ['one'])]), (pytest.raises(TypeError, match='No approximation is specified'), [(MeanFieldGroup, ['one', 'two'])]), (not_raises(), [(MeanFieldGroup, ['one']), (FullRankGroup, ['two', 'three'])]), (not_raises(), [(MeanFieldGroup, ['one']), (FullRankGroup, ['two']), (MeanFieldGroup, ['three'])]), (pytest.raises(TypeError, match='Found duplicates'), [(MeanFieldGroup, ['one']), (FullRankGroup, ['two', 'one']), (MeanFieldGroup, ['three'])])])
def test_init_groups(three_var_model, raises, grouping):
    if False:
        return 10
    with raises, three_var_model:
        (approxes, groups) = zip(*grouping)
        groups = [list(map(ft.partial(getattr, three_var_model), g)) if g is not None else None for g in groups]
        inited_groups = [a(group=g) for (a, g) in zip(approxes, groups)]
        approx = opvi.Approximation(inited_groups)
        for (ig, g) in zip(inited_groups, groups):
            if g is None:
                pass
            else:
                assert {pm.util.get_transformed(z) for z in g} == set(ig.group)
        else:
            model_dim = sum((v.size for v in three_var_model.initial_point(0).values()))
            assert approx.ndim == model_dim
        trace = approx.sample(100)

@pytest.fixture(params=[({}, {MeanFieldGroup: (None, {})}), ({}, {FullRankGroup: (None, {}), MeanFieldGroup: (['one'], {})}), ({}, {MeanFieldGroup: (['one'], {}), FullRankGroup: (['two', 'three'], {})}), ({}, {MeanFieldGroup: (['one'], {}), EmpiricalGroup: (['two', 'three'], {'size': 100})})], ids=lambda t: ', '.join((f'{k.__name__}: {v[0]}' for (k, v) in t[1].items())))
def three_var_groups(request, three_var_model):
    if False:
        for i in range(10):
            print('nop')
    (kw, grouping) = request.param
    (approxes, groups) = zip(*grouping.items())
    (groups, gkwargs) = zip(*groups)
    groups = [list(map(ft.partial(getattr, three_var_model), g)) if g is not None else None for g in groups]
    inited_groups = [a(group=g, model=three_var_model, **gk) for (a, g, gk) in zip(approxes, groups, gkwargs)]
    return inited_groups

@pytest.fixture
def three_var_approx(three_var_model, three_var_groups):
    if False:
        while True:
            i = 10
    approx = opvi.Approximation(three_var_groups, model=three_var_model)
    return approx

@pytest.fixture
def three_var_approx_single_group_mf(three_var_model):
    if False:
        return 10
    return MeanField(model=three_var_model)

def test_pickle_approx(three_var_approx):
    if False:
        return 10
    import cloudpickle
    dump = cloudpickle.dumps(three_var_approx)
    new = cloudpickle.loads(dump)
    assert new.sample(1)

def test_pickle_single_group(three_var_approx_single_group_mf):
    if False:
        return 10
    import cloudpickle
    dump = cloudpickle.dumps(three_var_approx_single_group_mf)
    new = cloudpickle.loads(dump)
    assert new.sample(1)

def test_sample_simple(three_var_approx):
    if False:
        for i in range(10):
            print('nop')
    trace = three_var_approx.sample(100, return_inferencedata=False)
    assert set(trace.varnames) == {'one', 'one_log__', 'three', 'two'}
    assert len(trace) == 100
    assert trace[0]['one'].shape == (10, 2)
    assert trace[0]['two'].shape == (10,)
    assert trace[0]['three'].shape == (10, 1, 2)

@pytest.fixture(params=[(MeanFieldGroup, {}), (FullRankGroup, {})], ids=lambda t: f'{t[0].__name__}: {t[1]}')
def parametric_grouped_approxes(request):
    if False:
        print('Hello World!')
    return request.param

def test_logq_mini_1_sample_1_var(parametric_grouped_approxes, three_var_model):
    if False:
        while True:
            i = 10
    (cls, kw) = parametric_grouped_approxes
    approx = cls([three_var_model.one], model=three_var_model, **kw)
    logq = approx.logq
    logq = approx.set_size_and_deterministic(logq, 1, 0)
    logq.eval()

def test_logq_mini_2_sample_2_var(parametric_grouped_approxes, three_var_model):
    if False:
        for i in range(10):
            print('nop')
    (cls, kw) = parametric_grouped_approxes
    approx = cls([three_var_model.one, three_var_model.two], model=three_var_model, **kw)
    logq = approx.logq
    logq = approx.set_size_and_deterministic(logq, 2, 0)
    logq.eval()

def test_logq_globals(three_var_approx):
    if False:
        for i in range(10):
            print('nop')
    if not three_var_approx.has_logq:
        pytest.skip('%s does not implement logq' % three_var_approx)
    approx = three_var_approx
    (logq, symbolic_logq) = approx.set_size_and_deterministic([approx.logq, approx.symbolic_logq], 1, 0)
    e = logq.eval()
    es = symbolic_logq.eval()
    assert e.shape == ()
    assert es.shape == (1,)
    (logq, symbolic_logq) = approx.set_size_and_deterministic([approx.logq, approx.symbolic_logq], 2, 0)
    e = logq.eval()
    es = symbolic_logq.eval()
    assert e.shape == ()
    assert es.shape == (2,)