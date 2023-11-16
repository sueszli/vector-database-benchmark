"""
Test the fastica algorithm.
"""
import itertools
import os
import warnings
import numpy as np
import pytest
from scipy import stats
from sklearn.decomposition import PCA, FastICA, fastica
from sklearn.decomposition._fastica import _gs_decorrelation
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import assert_allclose

def center_and_norm(x, axis=-1):
    if False:
        i = 10
        return i + 15
    'Centers and norms x **in place**\n\n    Parameters\n    -----------\n    x: ndarray\n        Array with an axis of observations (statistical units) measured on\n        random variables.\n    axis: int, optional\n        Axis along which the mean and variance are calculated.\n    '
    x = np.rollaxis(x, axis)
    x -= x.mean(axis=0)
    x /= x.std(axis=0)

def test_gs():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    (W, _, _) = np.linalg.svd(rng.randn(10, 10))
    w = rng.randn(10)
    _gs_decorrelation(w, W, 10)
    assert (w ** 2).sum() < 1e-10
    w = rng.randn(10)
    u = _gs_decorrelation(w, W, 5)
    tmp = np.dot(u, W.T)
    assert (tmp[:5] ** 2).sum() < 1e-10

def test_fastica_attributes_dtypes(global_dtype):
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    X = rng.random_sample((100, 10)).astype(global_dtype, copy=False)
    fica = FastICA(n_components=5, max_iter=1000, whiten='unit-variance', random_state=0).fit(X)
    assert fica.components_.dtype == global_dtype
    assert fica.mixing_.dtype == global_dtype
    assert fica.mean_.dtype == global_dtype
    assert fica.whitening_.dtype == global_dtype

def test_fastica_return_dtypes(global_dtype):
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.RandomState(0)
    X = rng.random_sample((100, 10)).astype(global_dtype, copy=False)
    (k_, mixing_, s_) = fastica(X, max_iter=1000, whiten='unit-variance', random_state=rng)
    assert k_.dtype == global_dtype
    assert mixing_.dtype == global_dtype
    assert s_.dtype == global_dtype

@pytest.mark.parametrize('add_noise', [True, False])
def test_fastica_simple(add_noise, global_random_seed, global_dtype):
    if False:
        return 10
    if global_random_seed == 20 and global_dtype == np.float32 and (not add_noise) and (os.getenv('DISTRIB') == 'ubuntu'):
        pytest.xfail('FastICA instability with Ubuntu Atlas build with float32 global_dtype. For more details, see https://github.com/scikit-learn/scikit-learn/issues/24131#issuecomment-1208091119')
    rng = np.random.RandomState(global_random_seed)
    n_samples = 1000
    s1 = (2 * np.sin(np.linspace(0, 100, n_samples)) > 0) - 1
    s2 = stats.t.rvs(1, size=n_samples, random_state=global_random_seed)
    s = np.c_[s1, s2].T
    center_and_norm(s)
    s = s.astype(global_dtype)
    (s1, s2) = s
    phi = 0.6
    mixing = np.array([[np.cos(phi), np.sin(phi)], [np.sin(phi), -np.cos(phi)]])
    mixing = mixing.astype(global_dtype)
    m = np.dot(mixing, s)
    if add_noise:
        m += 0.1 * rng.randn(2, 1000)
    center_and_norm(m)

    def g_test(x):
        if False:
            for i in range(10):
                print('nop')
        return (x ** 3, (3 * x ** 2).mean(axis=-1))
    algos = ['parallel', 'deflation']
    nls = ['logcosh', 'exp', 'cube', g_test]
    whitening = ['arbitrary-variance', 'unit-variance', False]
    for (algo, nl, whiten) in itertools.product(algos, nls, whitening):
        if whiten:
            (k_, mixing_, s_) = fastica(m.T, fun=nl, whiten=whiten, algorithm=algo, random_state=rng)
            with pytest.raises(ValueError):
                fastica(m.T, fun=np.tanh, whiten=whiten, algorithm=algo)
        else:
            pca = PCA(n_components=2, whiten=True, random_state=rng)
            X = pca.fit_transform(m.T)
            (k_, mixing_, s_) = fastica(X, fun=nl, algorithm=algo, whiten=False, random_state=rng)
            with pytest.raises(ValueError):
                fastica(X, fun=np.tanh, algorithm=algo)
        s_ = s_.T
        if whiten:
            atol = 1e-05 if global_dtype == np.float32 else 0
            assert_allclose(np.dot(np.dot(mixing_, k_), m), s_, atol=atol)
        center_and_norm(s_)
        (s1_, s2_) = s_
        if abs(np.dot(s1_, s2)) > abs(np.dot(s1_, s1)):
            (s2_, s1_) = s_
        s1_ *= np.sign(np.dot(s1_, s1))
        s2_ *= np.sign(np.dot(s2_, s2))
        if not add_noise:
            assert_allclose(np.dot(s1_, s1) / n_samples, 1, atol=0.01)
            assert_allclose(np.dot(s2_, s2) / n_samples, 1, atol=0.01)
        else:
            assert_allclose(np.dot(s1_, s1) / n_samples, 1, atol=0.1)
            assert_allclose(np.dot(s2_, s2) / n_samples, 1, atol=0.1)
    (_, _, sources_fun) = fastica(m.T, fun=nl, algorithm=algo, random_state=global_random_seed)
    ica = FastICA(fun=nl, algorithm=algo, random_state=global_random_seed)
    sources = ica.fit_transform(m.T)
    assert ica.components_.shape == (2, 2)
    assert sources.shape == (1000, 2)
    assert_allclose(sources_fun, sources)
    atol = np.max(np.abs(sources)) * (1e-05 if global_dtype == np.float32 else 1e-07)
    assert_allclose(sources, ica.transform(m.T), atol=atol)
    assert ica.mixing_.shape == (2, 2)
    ica = FastICA(fun=np.tanh, algorithm=algo)
    with pytest.raises(ValueError):
        ica.fit(m.T)

def test_fastica_nowhiten():
    if False:
        i = 10
        return i + 15
    m = [[0, 1], [1, 0]]
    ica = FastICA(n_components=1, whiten=False, random_state=0)
    warn_msg = 'Ignoring n_components with whiten=False.'
    with pytest.warns(UserWarning, match=warn_msg):
        ica.fit(m)
    assert hasattr(ica, 'mixing_')

def test_fastica_convergence_fail():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    n_samples = 1000
    t = np.linspace(0, 100, n_samples)
    s1 = np.sin(t)
    s2 = np.ceil(np.sin(np.pi * t))
    s = np.c_[s1, s2].T
    center_and_norm(s)
    mixing = rng.randn(6, 2)
    m = np.dot(mixing, s)
    warn_msg = 'FastICA did not converge. Consider increasing tolerance or the maximum number of iterations.'
    with pytest.warns(ConvergenceWarning, match=warn_msg):
        ica = FastICA(algorithm='parallel', n_components=2, random_state=rng, max_iter=2, tol=0.0)
        ica.fit(m.T)

@pytest.mark.parametrize('add_noise', [True, False])
def test_non_square_fastica(add_noise):
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    n_samples = 1000
    t = np.linspace(0, 100, n_samples)
    s1 = np.sin(t)
    s2 = np.ceil(np.sin(np.pi * t))
    s = np.c_[s1, s2].T
    center_and_norm(s)
    (s1, s2) = s
    mixing = rng.randn(6, 2)
    m = np.dot(mixing, s)
    if add_noise:
        m += 0.1 * rng.randn(6, n_samples)
    center_and_norm(m)
    (k_, mixing_, s_) = fastica(m.T, n_components=2, whiten='unit-variance', random_state=rng)
    s_ = s_.T
    assert_allclose(s_, np.dot(np.dot(mixing_, k_), m))
    center_and_norm(s_)
    (s1_, s2_) = s_
    if abs(np.dot(s1_, s2)) > abs(np.dot(s1_, s1)):
        (s2_, s1_) = s_
    s1_ *= np.sign(np.dot(s1_, s1))
    s2_ *= np.sign(np.dot(s2_, s2))
    if not add_noise:
        assert_allclose(np.dot(s1_, s1) / n_samples, 1, atol=0.001)
        assert_allclose(np.dot(s2_, s2) / n_samples, 1, atol=0.001)

def test_fit_transform(global_random_seed, global_dtype):
    if False:
        return 10
    'Test unit variance of transformed data using FastICA algorithm.\n\n    Check that `fit_transform` gives the same result as applying\n    `fit` and then `transform`.\n\n    Bug #13056\n    '
    rng = np.random.RandomState(global_random_seed)
    X = rng.random_sample((100, 10)).astype(global_dtype)
    max_iter = 300
    for (whiten, n_components) in [['unit-variance', 5], [False, None]]:
        n_components_ = n_components if n_components is not None else X.shape[1]
        ica = FastICA(n_components=n_components, max_iter=max_iter, whiten=whiten, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            warnings.simplefilter('ignore', ConvergenceWarning)
            Xt = ica.fit_transform(X)
        assert ica.components_.shape == (n_components_, 10)
        assert Xt.shape == (X.shape[0], n_components_)
        ica2 = FastICA(n_components=n_components, max_iter=max_iter, whiten=whiten, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            warnings.simplefilter('ignore', ConvergenceWarning)
            ica2.fit(X)
        assert ica2.components_.shape == (n_components_, 10)
        Xt2 = ica2.transform(X)
        if global_dtype:
            atol = np.abs(Xt2).mean() / 1000000.0
        else:
            atol = 0.0
        assert_allclose(Xt, Xt2, atol=atol)

@pytest.mark.filterwarnings('ignore:Ignoring n_components with whiten=False.')
@pytest.mark.parametrize('whiten, n_components, expected_mixing_shape', [('arbitrary-variance', 5, (10, 5)), ('arbitrary-variance', 10, (10, 10)), ('unit-variance', 5, (10, 5)), ('unit-variance', 10, (10, 10)), (False, 5, (10, 10)), (False, 10, (10, 10))])
def test_inverse_transform(whiten, n_components, expected_mixing_shape, global_random_seed, global_dtype):
    if False:
        i = 10
        return i + 15
    n_samples = 100
    rng = np.random.RandomState(global_random_seed)
    X = rng.random_sample((n_samples, 10)).astype(global_dtype)
    ica = FastICA(n_components=n_components, random_state=rng, whiten=whiten)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        Xt = ica.fit_transform(X)
    assert ica.mixing_.shape == expected_mixing_shape
    X2 = ica.inverse_transform(Xt)
    assert X.shape == X2.shape
    if n_components == X.shape[1]:
        if global_dtype:
            atol = np.abs(X2).mean() / 100000.0
        else:
            atol = 0.0
        assert_allclose(X, X2, atol=atol)

def test_fastica_errors():
    if False:
        for i in range(10):
            print('nop')
    n_features = 3
    n_samples = 10
    rng = np.random.RandomState(0)
    X = rng.random_sample((n_samples, n_features))
    w_init = rng.randn(n_features + 1, n_features + 1)
    with pytest.raises(ValueError, match='alpha must be in \\[1,2\\]'):
        fastica(X, fun_args={'alpha': 0})
    with pytest.raises(ValueError, match='w_init has invalid shape.+should be \\(3L?, 3L?\\)'):
        fastica(X, w_init=w_init)

def test_fastica_whiten_unit_variance():
    if False:
        return 10
    'Test unit variance of transformed data using FastICA algorithm.\n\n    Bug #13056\n    '
    rng = np.random.RandomState(0)
    X = rng.random_sample((100, 10))
    n_components = X.shape[1]
    ica = FastICA(n_components=n_components, whiten='unit-variance', random_state=0)
    Xt = ica.fit_transform(X)
    assert np.var(Xt) == pytest.approx(1.0)

@pytest.mark.parametrize('whiten', ['arbitrary-variance', 'unit-variance', False])
@pytest.mark.parametrize('return_X_mean', [True, False])
@pytest.mark.parametrize('return_n_iter', [True, False])
def test_fastica_output_shape(whiten, return_X_mean, return_n_iter):
    if False:
        return 10
    n_features = 3
    n_samples = 10
    rng = np.random.RandomState(0)
    X = rng.random_sample((n_samples, n_features))
    expected_len = 3 + return_X_mean + return_n_iter
    out = fastica(X, whiten=whiten, return_n_iter=return_n_iter, return_X_mean=return_X_mean)
    assert len(out) == expected_len
    if not whiten:
        assert out[0] is None

@pytest.mark.parametrize('add_noise', [True, False])
def test_fastica_simple_different_solvers(add_noise, global_random_seed):
    if False:
        return 10
    'Test FastICA is consistent between whiten_solvers.'
    rng = np.random.RandomState(global_random_seed)
    n_samples = 1000
    s1 = (2 * np.sin(np.linspace(0, 100, n_samples)) > 0) - 1
    s2 = stats.t.rvs(1, size=n_samples, random_state=rng)
    s = np.c_[s1, s2].T
    center_and_norm(s)
    (s1, s2) = s
    phi = rng.rand() * 2 * np.pi
    mixing = np.array([[np.cos(phi), np.sin(phi)], [np.sin(phi), -np.cos(phi)]])
    m = np.dot(mixing, s)
    if add_noise:
        m += 0.1 * rng.randn(2, 1000)
    center_and_norm(m)
    outs = {}
    for solver in ('svd', 'eigh'):
        ica = FastICA(random_state=0, whiten='unit-variance', whiten_solver=solver)
        sources = ica.fit_transform(m.T)
        outs[solver] = sources
        assert ica.components_.shape == (2, 2)
        assert sources.shape == (1000, 2)
    assert_allclose(outs['eigh'], outs['svd'], atol=1e-12)

def test_fastica_eigh_low_rank_warning(global_random_seed):
    if False:
        for i in range(10):
            print('nop')
    'Test FastICA eigh solver raises warning for low-rank data.'
    rng = np.random.RandomState(global_random_seed)
    A = rng.randn(10, 2)
    X = A @ A.T
    ica = FastICA(random_state=0, whiten='unit-variance', whiten_solver='eigh')
    msg = 'There are some small singular values'
    with pytest.warns(UserWarning, match=msg):
        ica.fit(X)