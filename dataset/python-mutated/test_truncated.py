import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import scipy
from pytensor.tensor.random.basic import GeometricRV, NormalRV
from pymc import Censored, Model, draw, find_MAP
from pymc.distributions.continuous import Exponential, Gamma, TruncatedNormal, TruncatedNormalRV
from pymc.distributions.shape_utils import change_dist_size
from pymc.distributions.transforms import _default_transform
from pymc.distributions.truncated import Truncated, TruncatedRV, _truncated
from pymc.exceptions import TruncationError
from pymc.logprob.abstract import _icdf
from pymc.logprob.basic import logcdf, logp
from pymc.logprob.transforms import IntervalTransform
from pymc.logprob.utils import ParameterValueError
from pymc.testing import assert_moment_is_expected

class IcdfNormalRV(NormalRV):
    """Normal RV that has icdf but not truncated dispatching"""

class RejectionNormalRV(NormalRV):
    """Normal RV that has neither icdf nor truncated dispatching."""

class IcdfGeometricRV(GeometricRV):
    """Geometric RV that has icdf but not truncated dispatching."""

class RejectionGeometricRV(GeometricRV):
    """Geometric RV that has neither icdf nor truncated dispatching."""
icdf_normal = no_moment_normal = IcdfNormalRV()
rejection_normal = RejectionNormalRV()
icdf_geometric = IcdfGeometricRV()
rejection_geometric = RejectionGeometricRV()

@_truncated.register(IcdfNormalRV)
@_truncated.register(RejectionNormalRV)
@_truncated.register(IcdfGeometricRV)
@_truncated.register(RejectionGeometricRV)
def _truncated_not_implemented(*args, **kwargs):
    if False:
        return 10
    raise NotImplementedError()

@_icdf.register(RejectionNormalRV)
@_icdf.register(RejectionGeometricRV)
def _icdf_not_implemented(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    raise NotImplementedError()

@pytest.mark.parametrize('shape_info', ('shape', 'dims', 'observed'))
def test_truncation_specialized_op(shape_info):
    if False:
        return 10
    rng = pytensor.shared(np.random.default_rng())
    x = pt.random.normal(0, 10, rng=rng, name='x')
    with Model(coords={'dim': range(100)}) as m:
        if shape_info == 'shape':
            xt = Truncated('xt', dist=x, lower=5, upper=15, shape=(100,))
        elif shape_info == 'dims':
            xt = Truncated('xt', dist=x, lower=5, upper=15, dims=('dim',))
        elif shape_info == 'observed':
            xt = Truncated('xt', dist=x, lower=5, upper=15, observed=np.zeros(100))
        else:
            raise ValueError(f'Not a valid shape_info parametrization: {shape_info}')
    assert isinstance(xt.owner.op, TruncatedNormalRV)
    assert xt.shape.eval() == (100,)
    assert xt.owner.inputs[0] is not rng
    lower_upper = pt.stack(xt.owner.inputs[5:])
    assert np.all(lower_upper.eval() == [5, 15])

@pytest.mark.parametrize('lower, upper', [(-1, np.inf), (-1, 1.5), (-np.inf, 1.5)])
@pytest.mark.parametrize('op_type', ['icdf', 'rejection'])
@pytest.mark.parametrize('scalar', [True, False])
def test_truncation_continuous_random(op_type, lower, upper, scalar):
    if False:
        for i in range(10):
            print('nop')
    loc = 0.15
    scale = 10
    normal_op = icdf_normal if op_type == 'icdf' else rejection_normal
    x = normal_op(loc, scale, name='x', size=() if scalar else (100,))
    xt = Truncated.dist(x, lower=lower, upper=upper)
    assert isinstance(xt.owner.op, TruncatedRV)
    assert xt.type.dtype == x.type.dtype
    xt_draws = draw(xt, draws=5)
    assert np.all(xt_draws >= lower)
    assert np.all(xt_draws <= upper)
    assert np.unique(xt_draws).size == xt_draws.size
    ref_xt = scipy.stats.truncnorm((lower - loc) / scale, (upper - loc) / scale, loc, scale)
    assert scipy.stats.cramervonmises(xt_draws.ravel(), ref_xt.cdf).pvalue > 0.001
    xt = Truncated.dist(x, lower=lower, upper=upper, max_n_steps=1)
    if op_type == 'icdf':
        xt_draws = draw(xt)
        assert np.all(xt_draws >= lower)
        assert np.all(xt_draws <= upper)
        assert np.unique(xt_draws).size == xt_draws.size
    else:
        with pytest.raises(TruncationError, match='^Truncation did not converge'):
            draw(xt, draws=100 if scalar else 1)

@pytest.mark.parametrize('lower, upper', [(-1, np.inf), (-1, 1.5), (-np.inf, 1.5)])
@pytest.mark.parametrize('op_type', ['icdf', 'rejection'])
def test_truncation_continuous_logp(op_type, lower, upper):
    if False:
        while True:
            i = 10
    loc = 0.15
    scale = 10
    op = icdf_normal if op_type == 'icdf' else rejection_normal
    x = op(loc, scale, name='x')
    xt = Truncated.dist(x, lower=lower, upper=upper)
    assert isinstance(xt.owner.op, TruncatedRV)
    xt_vv = xt.clone()
    xt_logp_fn = pytensor.function([xt_vv], logp(xt, xt_vv))
    ref_xt = scipy.stats.truncnorm((lower - loc) / scale, (upper - loc) / scale, loc, scale)
    for bound in (lower, upper):
        if np.isinf(bound):
            return
        for offset in (-1, 0, 1):
            test_xt_v = bound + offset
            assert np.isclose(xt_logp_fn(test_xt_v), ref_xt.logpdf(test_xt_v))

@pytest.mark.parametrize('lower, upper', [(-1, np.inf), (-1, 1.5), (-np.inf, 1.5)])
@pytest.mark.parametrize('op_type', ['icdf', 'rejection'])
def test_truncation_continuous_logcdf(op_type, lower, upper):
    if False:
        i = 10
        return i + 15
    loc = 0.15
    scale = 10
    op = icdf_normal if op_type == 'icdf' else rejection_normal
    x = op(loc, scale, name='x')
    xt = Truncated.dist(x, lower=lower, upper=upper)
    assert isinstance(xt.owner.op, TruncatedRV)
    xt_vv = xt.clone()
    xt_logcdf_fn = pytensor.function([xt_vv], logcdf(xt, xt_vv))
    ref_xt = scipy.stats.truncnorm((lower - loc) / scale, (upper - loc) / scale, loc, scale)
    for bound in (lower, upper):
        if np.isinf(bound):
            return
        for offset in (-1, 0, 1):
            test_xt_v = bound + offset
            assert np.isclose(xt_logcdf_fn(test_xt_v), ref_xt.logcdf(test_xt_v))

@pytest.mark.parametrize('lower, upper', [(2, np.inf), (2, 5), (-np.inf, 5)])
@pytest.mark.parametrize('op_type', ['icdf', 'rejection'])
def test_truncation_discrete_random(op_type, lower, upper):
    if False:
        i = 10
        return i + 15
    p = 0.2
    geometric_op = icdf_geometric if op_type == 'icdf' else rejection_geometric
    x = geometric_op(p, name='x', size=500)
    xt = Truncated.dist(x, lower=lower, upper=upper)
    assert isinstance(xt.owner.op, TruncatedRV)
    xt_draws = draw(xt)
    assert np.all(xt_draws >= lower)
    assert np.all(xt_draws <= upper)
    assert np.any(xt_draws == max(1, lower))
    if upper != np.inf:
        assert np.any(xt_draws == upper)
    xt = Truncated.dist(x, lower=lower, upper=upper, max_n_steps=3)
    if op_type == 'icdf':
        xt_draws = draw(xt)
        assert np.all(xt_draws >= lower)
        assert np.all(xt_draws <= upper)
        assert np.any(xt_draws == max(1, lower))
        if upper != np.inf:
            assert np.any(xt_draws == upper)
    else:
        with pytest.raises(TruncationError, match='^Truncation did not converge'):
            draw(xt, random_seed=2297228)

@pytest.mark.parametrize('lower, upper', [(2, np.inf), (2, 5), (-np.inf, 5)])
@pytest.mark.parametrize('op_type', ['icdf', 'rejection'])
def test_truncation_discrete_logp(op_type, lower, upper):
    if False:
        print('Hello World!')
    p = 0.7
    op = icdf_geometric if op_type == 'icdf' else rejection_geometric
    x = op(p, name='x')
    xt = Truncated.dist(x, lower=lower, upper=upper)
    assert isinstance(xt.owner.op, TruncatedRV)
    xt_vv = xt.clone()
    xt_logp_fn = pytensor.function([xt_vv], logp(xt, xt_vv))
    ref_xt = scipy.stats.geom(p)
    log_norm = np.log(ref_xt.cdf(upper) - ref_xt.cdf(lower - 1))

    def ref_xt_logpmf(value):
        if False:
            return 10
        if value < lower or value > upper:
            return -np.inf
        return ref_xt.logpmf(value) - log_norm
    for bound in (lower, upper):
        if np.isinf(bound):
            continue
        for offset in (-1, 0, 1):
            test_xt_v = bound + offset
            assert np.isclose(xt_logp_fn(test_xt_v), ref_xt_logpmf(test_xt_v))
    log_integral = scipy.special.logsumexp([xt_logp_fn(v) for v in range(min(upper + 1, 20))])
    assert np.isclose(log_integral, 0.0, atol=1e-05)

@pytest.mark.parametrize('lower, upper', [(2, np.inf), (2, 5), (-np.inf, 5)])
@pytest.mark.parametrize('op_type', ['icdf', 'rejection'])
def test_truncation_discrete_logcdf(op_type, lower, upper):
    if False:
        print('Hello World!')
    p = 0.7
    op = icdf_geometric if op_type == 'icdf' else rejection_geometric
    x = op(p, name='x')
    xt = Truncated.dist(x, lower=lower, upper=upper)
    assert isinstance(xt.owner.op, TruncatedRV)
    xt_vv = xt.clone()
    xt_logcdf_fn = pytensor.function([xt_vv], logcdf(xt, xt_vv))
    ref_xt = scipy.stats.geom(p)
    log_norm = np.log(ref_xt.cdf(upper) - ref_xt.cdf(lower - 1))

    def ref_xt_logcdf(value):
        if False:
            for i in range(10):
                print('nop')
        if value < lower:
            return -np.inf
        elif value > upper:
            return 0.0
        return np.log(ref_xt.cdf(value) - ref_xt.cdf(lower - 1)) - log_norm
    for bound in (lower, upper):
        if np.isinf(bound):
            continue
        for offset in (-1, 0, 1):
            test_xt_v = bound + offset
            assert np.isclose(xt_logcdf_fn(test_xt_v), ref_xt_logcdf(test_xt_v))

def test_truncation_exceptions():
    if False:
        print('Hello World!')
    with pytest.raises(ValueError, match='lower and upper cannot both be None'):
        Truncated.dist(pt.random.normal())
    with pytest.raises(NotImplementedError, match='Truncation not implemented for SymbolicRandomVariable CensoredRV'):
        Truncated.dist(Censored.dist(pt.random.normal(), lower=-1, upper=1), -1, 1)
    with pytest.raises(NotImplementedError, match='Truncation not implemented for multivariate distributions'):
        Truncated.dist(pt.random.dirichlet([1, 1, 1]), -1, 1)

def test_truncation_logprob_bound_check():
    if False:
        while True:
            i = 10
    x = pt.random.normal(name='x')
    xt = Truncated.dist(x, lower=5, upper=-5)
    with pytest.raises(ParameterValueError):
        logp(xt, 0).eval()

def test_change_truncated_size():
    if False:
        print('Hello World!')
    x = Truncated.dist(icdf_normal(0, [1, 2, 3]), lower=-1, size=(2, 3))
    x.eval().shape == (2, 3)
    new_x = change_dist_size(x, (4, 3))
    assert isinstance(new_x.owner.op, TruncatedRV)
    new_x.eval().shape == (4, 3)
    new_x = change_dist_size(x, (4, 3), expand=True)
    assert isinstance(new_x.owner.op, TruncatedRV)
    new_x.eval().shape == (4, 3, 2, 3)

def test_truncated_default_transform():
    if False:
        return 10
    base_dist = rejection_geometric(1)
    x = Truncated.dist(base_dist, lower=None, upper=5)
    assert _default_transform(x.owner.op, x) is None
    base_dist = rejection_normal(0, 1)
    x = Truncated.dist(base_dist, lower=None, upper=5)
    assert isinstance(_default_transform(x.owner.op, x), IntervalTransform)

def test_truncated_transform_logp():
    if False:
        i = 10
        return i + 15
    with Model() as m:
        base_dist = rejection_normal(0, 1)
        x = Truncated('x', base_dist, lower=0, upper=None, transform=None)
        y = Truncated('y', base_dist, lower=0, upper=None)
        logp_eval = m.compile_logp(sum=False)({'x': -1, 'y_interval__': -1})
    assert logp_eval[0] == -np.inf
    assert np.isfinite(logp_eval[1])

@pytest.mark.parametrize('truncated_dist, lower, upper, shape, expected', [(icdf_normal(0, 1), -1, 2, None, 0), (icdf_normal(3, 1), -1, 2, (2,), np.full((2,), 3 / 2)), (icdf_normal(-3, 1), -1, None, (2, 3), np.full((2, 3), 0)), (icdf_normal([0, 3, 3], 1), None, [2, 2, 4], (4, 3), np.full((4, 3), [0, 1, 3]))])
def test_truncated_moment(truncated_dist, lower, upper, shape, expected):
    if False:
        return 10
    with Model() as model:
        Truncated('x', dist=truncated_dist, lower=lower, upper=upper, shape=shape)
    assert_moment_is_expected(model, expected)

def test_truncated_inference():
    if False:
        for i in range(10):
            print('nop')
    lam_true = 3
    lower = 0
    upper = 5
    rng = np.random.default_rng(260)
    x = rng.exponential(lam_true, size=5000)
    obs = x[np.where(~((x < lower) | (x > upper)))]
    with Model() as m:
        lam = Exponential('lam', lam=1 / 5)
        Truncated('x', dist=Exponential.dist(lam=1 / lam), lower=lower, upper=upper, observed=obs)
        map = find_MAP(progressbar=False)
    assert np.isclose(map['lam'], lam_true, atol=0.1)

def test_truncated_gamma():
    if False:
        for i in range(10):
            print('nop')
    alpha = 3.0
    beta = 3.0
    upper = 2.5
    x = np.linspace(0.0, upper + 0.5, 100)
    gamma_scipy = scipy.stats.gamma(a=alpha, scale=1.0 / beta)
    logp_scipy = gamma_scipy.logpdf(x) - gamma_scipy.logcdf(upper)
    logp_scipy[x > upper] = -np.inf
    gamma_trunc_pymc = Truncated.dist(Gamma.dist(alpha=alpha, beta=beta), upper=upper)
    logp_pymc = logp(gamma_trunc_pymc, x).eval()
    np.testing.assert_allclose(logp_pymc, logp_scipy)
    resized_gamma_trunc_pymc = change_dist_size(gamma_trunc_pymc, new_size=x.shape)
    logp_resized_pymc = logp(resized_gamma_trunc_pymc, x).eval()
    np.testing.assert_allclose(logp_resized_pymc, logp_scipy)

def test_vectorized_bounds():
    if False:
        print('Hello World!')
    with Model() as m:
        x1 = TruncatedNormal('x1', lower=None, upper=0, initval=-1)
        x2 = TruncatedNormal('x2', lower=0, upper=None, initval=1)
        x3 = TruncatedNormal('x3', lower=-np.pi, upper=np.e, initval=-1)
        x4 = TruncatedNormal('x4', lower=None, upper=None, initval=1)
        xs = TruncatedNormal('xs', lower=[-np.inf, 0, -np.pi, -np.inf], upper=[0, np.inf, np.e, np.inf], initval=[-1, 1, -1, 1])
        xs_sym = Truncated('xs_sym', dist=rejection_normal(), lower=[-np.inf, 0, -np.pi, -np.inf], upper=[0, np.inf, np.e, np.inf], initval=[-1, 1, -1, 1])
    ip = m.initial_point()
    np.testing.assert_allclose(np.stack([ip[f'x{i + 1}_interval__'] for i in range(4)]), ip['xs_interval__'])
    np.testing.assert_allclose(ip['xs_interval__'], ip['xs_sym_interval__'])
    np.testing.assert_allclose(m.rvs_to_transforms[xs].backward(ip['xs_interval__'], *xs.owner.inputs).eval(), [-1, 1, -1, 1])
    np.testing.assert_allclose(m.rvs_to_transforms[xs_sym].backward(ip['xs_sym_interval__'], *xs_sym.owner.inputs).eval(), [-1, 1, -1, 1])
    (*x_logp, xs_logp, xs_sym_logp) = m.compile_logp(sum=False)(ip)
    assert np.all(np.isfinite(xs_logp))
    np.testing.assert_allclose(np.stack(x_logp), xs_logp)
    np.testing.assert_allclose(xs_logp, xs_sym_logp)