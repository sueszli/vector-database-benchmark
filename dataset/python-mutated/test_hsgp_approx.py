import arviz as az
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import scipy as sp
from scipy.spatial import distance
import pymc as pm

def build_mmd_func(sample1, sample2):
    if False:
        for i in range(10):
            print('nop')
    'Build a PyTensor function that calculates the minimum mean discrepancy (MMD) statistic.'
    assert sample1.shape[1] == sample2.shape[1]
    s1 = pt.matrix(name='s1', shape=sample1.shape)
    s2 = pt.matrix(name='s2', shape=sample2.shape)
    X = np.concatenate((sample1, sample2), axis=0)
    test_ell = np.median(distance.pdist(X)) / 2
    K = pm.gp.cov.ExpQuad(sample1.shape[1], ls=test_ell)
    Kxx = K(s1)
    Kyy = K(s2)
    Kxy = K(s1, s2)
    (n_x, n_y) = (s1.shape[0], s2.shape[0])
    mmd = pt.sum(Kxx) / (n_x * (n_x - 1)) + pt.sum(Kyy) / (n_y * (n_y - 1)) - 2 * pt.sum(Kxy) / (n_x * n_y)
    calc_mmd = pytensor.function(inputs=[s1, s2], outputs=mmd)
    return calc_mmd

def two_sample_test(sample1, sample2, n_sims=1000, alpha=0.05):
    if False:
        return 10
    'Calculate test whose null hypothesis is that two sets of samples were drawn from\n    the same distribution.\n\n    Largely taken from https://torchdrift.org/notebooks/note_on_mmd.html\n    '
    calc_mmd = build_mmd_func(sample1, sample2)
    X = np.concatenate((sample1, sample2), axis=0)
    half_N = int(X.shape[0] // 2)
    ix = np.arange(half_N * 2)
    h0 = []
    for i in range(n_sims):
        np.random.shuffle(ix)
        X = X[ix, :]
        h0.append(calc_mmd(X[:half_N, :], X[half_N:, :]))
    h0 = np.asarray(h0)
    critical_value = np.percentile(h0, 100 * (1 - alpha))
    mmd = calc_mmd(sample1, sample2)
    return (h0, mmd, critical_value, mmd > critical_value)

class TestHSGP:

    @pytest.fixture
    def rng(self):
        if False:
            print('Hello World!')
        return np.random.RandomState(10)

    @pytest.fixture
    def data(self, rng):
        if False:
            return 10
        X1 = np.linspace(-5, 5, 100)[:, None]
        (x1, x2, x3) = np.meshgrid(np.linspace(0, 10, 5), np.linspace(20, 30, 5), np.linspace(10, 20, 5))
        X2 = np.vstack([x1.flatten(), x2.flatten(), x3.flatten()]).T
        return (X1, X2)

    @pytest.fixture
    def X1(self, data):
        if False:
            print('Hello World!')
        return data[0]

    @pytest.fixture
    def X2(self, data):
        if False:
            print('Hello World!')
        return data[1]

    @pytest.fixture
    def model(self):
        if False:
            i = 10
            return i + 15
        return pm.Model()

    @pytest.fixture
    def cov_func(self):
        if False:
            for i in range(10):
                print('nop')
        return pm.gp.cov.ExpQuad(1, ls=1)

    @pytest.fixture
    def gp(self, cov_func):
        if False:
            i = 10
            return i + 15
        gp = pm.gp.Latent(cov_func=cov_func)
        return gp

    def test_set_boundaries_1d(self, X1):
        if False:
            print('Hello World!')
        X1s = X1 - np.mean(X1, axis=0)
        L = pm.gp.hsgp_approx.set_boundary(X1s, c=2).eval()
        assert np.all(L == 10)

    def test_set_boundaries_3d(self, X2):
        if False:
            i = 10
            return i + 15
        X2s = X2 - np.mean(X2, axis=0)
        L = pm.gp.hsgp_approx.set_boundary(X2s, c=2).eval()
        assert np.all(L == 10)

    def test_parametrization(self):
        if False:
            i = 10
            return i + 15
        err_msg = '`m` and L, if provided, must be sequences with one element per active dimension'
        with pytest.raises(ValueError, match=err_msg):
            cov_func = pm.gp.cov.ExpQuad(1, ls=0.1)
            pm.gp.HSGP(m=500, c=2, cov_func=cov_func)
        with pytest.raises(ValueError, match=err_msg):
            cov_func = pm.gp.cov.ExpQuad(2, ls=[1, 2])
            pm.gp.HSGP(m=[500], L=[12, 12], cov_func=cov_func)
        with pytest.raises(ValueError, match=err_msg):
            cov_func = pm.gp.cov.ExpQuad(1, ls=0.1)
            pm.gp.HSGP(m=[500], L=[12, 12], cov_func=cov_func)
        cov_func = pm.gp.cov.ExpQuad(3, ls=[1, 2], active_dims=[0, 2])
        pm.gp.HSGP(m=[50, 50], c=2, cov_func=cov_func)
        cov_func = pm.gp.cov.ExpQuad(2, ls=[1, 2])
        pm.gp.HSGP(m=[50, 50], L=[12, 12], cov_func=cov_func)

    @pytest.mark.parametrize('drop_first', [True, False])
    def test_parametrization_drop_first(self, model, cov_func, X1, drop_first):
        if False:
            return 10
        n_basis = 100
        with model:
            gp = pm.gp.HSGP(m=[n_basis], c=4.0, cov_func=cov_func, drop_first=drop_first)
            gp.prior('f1', X1)
            n_coeffs = model.f1_hsgp_coeffs_.type.shape[0]
            if drop_first:
                assert n_coeffs == n_basis - 1, f'one basis vector should have been dropped, {n_coeffs}'
            else:
                assert n_coeffs == n_basis, "one was dropped when it shouldn't have been"

    @pytest.mark.parametrize('parameterization', ['centered', 'noncentered'])
    def test_prior(self, model, cov_func, X1, parameterization, rng):
        if False:
            for i in range(10):
                print('nop')
        'Compare HSGP prior to unapproximated GP prior, pm.gp.Latent.  Draw samples from the\n        prior and compare them using MMD two sample test.  Tests both centered and non-centered\n        parameterizations.\n        '
        with model:
            hsgp = pm.gp.HSGP(m=[200], c=2.0, parameterization=parameterization, cov_func=cov_func)
            f1 = hsgp.prior('f1', X=X1)
            gp = pm.gp.Latent(cov_func=cov_func)
            f2 = gp.prior('f2', X=X1)
            idata = pm.sample_prior_predictive(samples=1000, random_seed=rng)
        samples1 = az.extract(idata.prior['f1'])['f1'].values.T
        samples2 = az.extract(idata.prior['f2'])['f2'].values.T
        (h0, mmd, critical_value, reject) = two_sample_test(samples1, samples2, n_sims=500, alpha=0.01)
        assert not reject, 'H0 was rejected, even though HSGP and GP priors should match.'

    @pytest.mark.parametrize('parameterization', ['centered', 'noncentered'])
    def test_conditional(self, model, cov_func, X1, parameterization):
        if False:
            for i in range(10):
                print('nop')
        'Compare HSGP conditional to unapproximated GP prior, pm.gp.Latent.  Draw samples from the\n        prior and compare them using MMD two sample test.  Tests both centered and non-centered\n        parameterizations.  The conditional should match the prior when no data is observed.\n        '
        with model:
            hsgp = pm.gp.HSGP(m=[100], c=2.0, parameterization=parameterization, cov_func=cov_func)
            f = hsgp.prior('f', X=X1)
            fc = hsgp.conditional('fc', Xnew=X1)
            idata = pm.sample_prior_predictive(samples=1000)
        samples1 = az.extract(idata.prior['f'])['f'].values.T
        samples2 = az.extract(idata.prior['fc'])['fc'].values.T
        (h0, mmd, critical_value, reject) = two_sample_test(samples1, samples2, n_sims=500, alpha=0.01)
        assert not reject, 'H0 was rejected, even though HSGP prior and conditional should match.'