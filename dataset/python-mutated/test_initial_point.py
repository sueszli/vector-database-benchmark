import cloudpickle
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from pytensor.tensor.random.op import RandomVariable
import pymc as pm
from pymc.distributions.distribution import moment
from pymc.initial_point import make_initial_point_fn, make_initial_point_fns_per_chain

def transform_fwd(rv, expected_untransformed, model):
    if False:
        print('Hello World!')
    return model.rvs_to_transforms[rv].forward(expected_untransformed, *rv.owner.inputs).eval()

def transform_back(rv, transformed, model) -> np.ndarray:
    if False:
        while True:
            i = 10
    return model.rvs_to_transforms[rv].backward(transformed, *rv.owner.inputs).eval()

class TestInitvalAssignment:

    def test_dist_warnings_and_errors(self):
        if False:
            return 10
        with pytest.warns(FutureWarning, match='argument is deprecated and has no effect'):
            rv = pm.Exponential.dist(lam=1, testval=0.5)
        assert not hasattr(rv.tag, 'test_value')
        with pytest.raises(TypeError, match='Unexpected keyword argument `initval`.'):
            pm.Normal.dist(1, 2, initval=None)
        pass

    def test_new_warnings(self):
        if False:
            while True:
                i = 10
        with pm.Model() as pmodel:
            with pytest.warns(FutureWarning, match='`testval` argument is deprecated'):
                rv = pm.Uniform('u', 0, 1, testval=0.75)
                initial_point = pmodel.initial_point(random_seed=0)
                assert initial_point['u_interval__'] == transform_fwd(rv, 0.75, model=pmodel)
                assert not hasattr(rv.tag, 'test_value')
        pass

    def test_valid_string_strategy(self):
        if False:
            print('Hello World!')
        with pm.Model() as pmodel:
            pm.Uniform('x', 0, 1, size=2, initval='unknown')
            with pytest.raises(ValueError, match='Invalid string strategy: unknown'):
                pmodel.initial_point(random_seed=0)

class TestInitvalEvaluation:

    def test_make_initial_point_fns_per_chain_checks_kwargs(self):
        if False:
            print('Hello World!')
        with pm.Model() as pmodel:
            A = pm.Uniform('A', 0, 1, initval=0.5)
            B = pm.Uniform('B', lower=A, upper=1.5, transform=None, initval='moment')
        with pytest.raises(ValueError, match='Number of initval dicts'):
            make_initial_point_fns_per_chain(model=pmodel, overrides=[{}, None], jitter_rvs={}, chains=1)
        pass

    def test_dependent_initvals(self):
        if False:
            i = 10
            return i + 15
        with pm.Model() as pmodel:
            L = pm.Uniform('L', 0, 1, initval=0.5)
            U = pm.Uniform('U', lower=9, upper=10, initval=9.5)
            B1 = pm.Uniform('B1', lower=L, upper=U, initval=5)
            B2 = pm.Uniform('B2', lower=L, upper=U, initval=(L + U) / 2)
            ip = pmodel.initial_point(random_seed=0)
            assert ip['L_interval__'] == 0
            assert ip['U_interval__'] == 0
            assert ip['B1_interval__'] == 0
            assert ip['B2_interval__'] == 0
            pmodel.rvs_to_initial_values[U] = 9.9
            ip = pmodel.initial_point(random_seed=0)
            assert ip['B1_interval__'] < 0
            assert ip['B2_interval__'] == 0
        pass

    def test_nested_initvals(self):
        if False:
            while True:
                i = 10
        with pm.Model() as pmodel:
            one = pm.LogNormal('one', mu=np.log(1), sigma=1e-05, initval='prior')
            two = pm.Lognormal('two', mu=np.log(one * 2), sigma=1e-05, initval='prior')
            three = pm.LogNormal('three', mu=np.log(two * 2), sigma=1e-05, initval='prior')
            four = pm.LogNormal('four', mu=np.log(three * 2), sigma=1e-05, initval='prior')
            five = pm.LogNormal('five', mu=np.log(four * 2), sigma=1e-05, initval='prior')
            six = pm.LogNormal('six', mu=np.log(five * 2), sigma=1e-05, initval='prior')
        ip_vals = list(make_initial_point_fn(model=pmodel, return_transformed=True)(0).values())
        assert np.allclose(np.exp(ip_vals), [1, 2, 4, 8, 16, 32], rtol=0.001)
        ip_vals = list(make_initial_point_fn(model=pmodel, return_transformed=False)(0).values())
        assert np.allclose(ip_vals, [1, 2, 4, 8, 16, 32], rtol=0.001)
        pmodel.rvs_to_initial_values[four] = 1
        ip_vals = list(make_initial_point_fn(model=pmodel, return_transformed=True)(0).values())
        assert np.allclose(np.exp(ip_vals), [1, 2, 4, 1, 2, 4], rtol=0.001)
        ip_vals = list(make_initial_point_fn(model=pmodel, return_transformed=False)(0).values())
        assert np.allclose(ip_vals, [1, 2, 4, 1, 2, 4], rtol=0.001)

    def test_initval_resizing(self):
        if False:
            for i in range(10):
                print('nop')
        with pm.Model() as pmodel:
            data = pytensor.shared(np.arange(4))
            rv = pm.Uniform('u', lower=data, upper=10, initval='prior')
            ip = pmodel.initial_point(random_seed=0)
            assert np.shape(ip['u_interval__']) == (4,)
            data.set_value(np.arange(5))
            ip = pmodel.initial_point(random_seed=0)
            assert np.shape(ip['u_interval__']) == (5,)
        pass

    def test_seeding(self):
        if False:
            while True:
                i = 10
        with pm.Model() as pmodel:
            pm.Normal('A', initval='prior')
            pm.Uniform('B', initval='prior')
            pm.Normal('C', initval='moment')
            ip1 = pmodel.initial_point(random_seed=42)
            ip2 = pmodel.initial_point(random_seed=42)
            ip3 = pmodel.initial_point(random_seed=15)
            assert ip1 == ip2
            assert ip3 != ip2
        pass

    def test_untransformed_initial_point(self):
        if False:
            for i in range(10):
                print('nop')
        with pm.Model() as pmodel:
            pm.Flat('A', initval='moment')
            pm.HalfFlat('B', initval='moment')
        fn = make_initial_point_fn(model=pmodel, jitter_rvs={}, return_transformed=False)
        iv = fn(0)
        assert iv['A'] == 0
        assert iv['B'] == 1
        pass

    def test_adds_jitter(self):
        if False:
            while True:
                i = 10
        with pm.Model() as pmodel:
            A = pm.Flat('A', initval='moment')
            B = pm.HalfFlat('B', initval='moment')
            C = pm.Normal('C', mu=A + B, initval='moment')
        fn = make_initial_point_fn(model=pmodel, jitter_rvs={B}, return_transformed=True)
        iv = fn(0)
        assert iv['A'] == 0
        b_transformed = iv['B_log__']
        b_untransformed = transform_back(B, b_transformed, model=pmodel)
        assert b_transformed != 0
        assert -1 < b_transformed < 1
        assert np.isclose(iv['C'], np.array(0 + b_untransformed, dtype=pytensor.config.floatX))
        assert fn(0) == fn(0)
        assert fn(0) != fn(1)

    def test_respects_overrides(self):
        if False:
            i = 10
            return i + 15
        with pm.Model() as pmodel:
            A = pm.Flat('A', initval='moment')
            B = pm.HalfFlat('B', initval=4)
            C = pm.Normal('C', mu=A + B, initval='moment')
        fn = make_initial_point_fn(model=pmodel, jitter_rvs={}, return_transformed=True, overrides={A: pt.as_tensor(2, dtype=int), B: 3, C: 5})
        iv = fn(0)
        assert iv['A'] == 2
        assert np.isclose(iv['B_log__'], np.log(3))
        assert iv['C'] == 5

    def test_string_overrides_work(self):
        if False:
            for i in range(10):
                print('nop')
        with pm.Model() as pmodel:
            A = pm.Flat('A', initval=10)
            B = pm.HalfFlat('B', initval=10)
            C = pm.HalfFlat('C', initval=10)
        fn = make_initial_point_fn(model=pmodel, jitter_rvs={}, return_transformed=True, overrides={'A': 1, 'B': 1, 'C_log__': 0})
        iv = fn(0)
        assert iv['A'] == 1
        assert np.isclose(iv['B_log__'], 0)
        assert iv['C_log__'] == 0

class TestMoment:

    def test_basic(self):
        if False:
            for i in range(10):
                print('nop')
        rv = pm.Normal.dist(mu=2.3)
        np.testing.assert_allclose(moment(rv).eval(), 2.3)
        rv = pm.Flat.dist()
        assert moment(rv).eval() == np.zeros(())
        rv = pm.HalfFlat.dist()
        assert moment(rv).eval() == np.ones(())
        rv = pm.Flat.dist(size=(2, 4))
        assert np.all(moment(rv).eval() == np.zeros((2, 4)))
        rv = pm.HalfFlat.dist(size=(2, 4))
        assert np.all(moment(rv).eval() == np.ones((2, 4)))

    @pytest.mark.parametrize('rv_cls', [pm.Flat, pm.HalfFlat])
    def test_numeric_moment_shape(self, rv_cls):
        if False:
            while True:
                i = 10
        rv = rv_cls.dist(shape=(2,))
        assert not hasattr(rv.tag, 'test_value')
        assert tuple(moment(rv).shape.eval()) == (2,)

    @pytest.mark.parametrize('rv_cls', [pm.Flat, pm.HalfFlat])
    def test_symbolic_moment_shape(self, rv_cls):
        if False:
            return 10
        s = pt.scalar(dtype='int64')
        rv = rv_cls.dist(shape=(s,))
        assert not hasattr(rv.tag, 'test_value')
        assert tuple(moment(rv).shape.eval({s: 4})) == (4,)
        pass

    @pytest.mark.parametrize('rv_cls', [pm.Flat, pm.HalfFlat])
    def test_moment_from_dims(self, rv_cls):
        if False:
            for i in range(10):
                print('nop')
        with pm.Model(coords={'year': [2019, 2020, 2021, 2022], 'city': ['Bonn', 'Paris', 'Lisbon']}):
            rv = rv_cls('rv', dims=('year', 'city'))
            assert not hasattr(rv.tag, 'test_value')
            assert tuple(moment(rv).shape.eval()) == (4, 3)
        pass

    def test_moment_not_implemented_fallback(self):
        if False:
            print('Hello World!')

        class MyNormalRV(RandomVariable):
            name = 'my_normal'
            ndim_supp = 0
            ndims_params = [0, 0]
            dtype = 'floatX'

            @classmethod
            def rng_fn(cls, rng, mu, sigma, size):
                if False:
                    for i in range(10):
                        print('nop')
                return np.pi

        class MyNormalDistribution(pm.Normal):
            rv_op = MyNormalRV()
        with pm.Model() as m:
            x = MyNormalDistribution('x', 0, 1, initval='moment')
        with pytest.warns(UserWarning, match='Moment not defined for variable x of type MyNormalRV'):
            res = m.initial_point()
        assert np.isclose(res['x'], np.pi)

def test_pickling_issue_5090():
    if False:
        return 10
    with pm.Model() as model:
        pm.Normal('x', initval='prior')
    ip_before = model.initial_point(random_seed=5090)
    model = cloudpickle.loads(cloudpickle.dumps(model))
    ip_after = model.initial_point(random_seed=5090)
    assert ip_before['x'] == ip_after['x']