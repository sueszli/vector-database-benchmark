import re
import sys
from io import StringIO
import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.neural_network import BernoulliRBM
from sklearn.utils._testing import assert_allclose, assert_almost_equal, assert_array_equal
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
from sklearn.utils.validation import assert_all_finite
(Xdigits, _) = load_digits(return_X_y=True)
Xdigits -= Xdigits.min()
Xdigits /= Xdigits.max()

def test_fit():
    if False:
        while True:
            i = 10
    X = Xdigits.copy()
    rbm = BernoulliRBM(n_components=64, learning_rate=0.1, batch_size=10, n_iter=7, random_state=9)
    rbm.fit(X)
    assert_almost_equal(rbm.score_samples(X).mean(), -21.0, decimal=0)
    assert_array_equal(X, Xdigits)

def test_partial_fit():
    if False:
        for i in range(10):
            print('nop')
    X = Xdigits.copy()
    rbm = BernoulliRBM(n_components=64, learning_rate=0.1, batch_size=20, random_state=9)
    n_samples = X.shape[0]
    n_batches = int(np.ceil(float(n_samples) / rbm.batch_size))
    batch_slices = np.array_split(X, n_batches)
    for i in range(7):
        for batch in batch_slices:
            rbm.partial_fit(batch)
    assert_almost_equal(rbm.score_samples(X).mean(), -21.0, decimal=0)
    assert_array_equal(X, Xdigits)

def test_transform():
    if False:
        print('Hello World!')
    X = Xdigits[:100]
    rbm1 = BernoulliRBM(n_components=16, batch_size=5, n_iter=5, random_state=42)
    rbm1.fit(X)
    Xt1 = rbm1.transform(X)
    Xt2 = rbm1._mean_hiddens(X)
    assert_array_equal(Xt1, Xt2)

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_small_sparse(csr_container):
    if False:
        i = 10
        return i + 15
    X = csr_container(Xdigits[:4])
    BernoulliRBM().fit(X)

@pytest.mark.parametrize('sparse_container', CSC_CONTAINERS + CSR_CONTAINERS)
def test_small_sparse_partial_fit(sparse_container):
    if False:
        return 10
    X_sparse = sparse_container(Xdigits[:100])
    X = Xdigits[:100].copy()
    rbm1 = BernoulliRBM(n_components=64, learning_rate=0.1, batch_size=10, random_state=9)
    rbm2 = BernoulliRBM(n_components=64, learning_rate=0.1, batch_size=10, random_state=9)
    rbm1.partial_fit(X_sparse)
    rbm2.partial_fit(X)
    assert_almost_equal(rbm1.score_samples(X).mean(), rbm2.score_samples(X).mean(), decimal=0)

def test_sample_hiddens():
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.RandomState(0)
    X = Xdigits[:100]
    rbm1 = BernoulliRBM(n_components=2, batch_size=5, n_iter=5, random_state=42)
    rbm1.fit(X)
    h = rbm1._mean_hiddens(X[0])
    hs = np.mean([rbm1._sample_hiddens(X[0], rng) for i in range(100)], 0)
    assert_almost_equal(h, hs, decimal=1)

@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_fit_gibbs(csc_container):
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(42)
    X = np.array([[0.0], [1.0]])
    rbm1 = BernoulliRBM(n_components=2, batch_size=2, n_iter=42, random_state=rng)
    rbm1.fit(X)
    assert_almost_equal(rbm1.components_, np.array([[0.02649814], [0.02009084]]), decimal=4)
    assert_almost_equal(rbm1.gibbs(X), X)
    rng = np.random.RandomState(42)
    X = csc_container([[0.0], [1.0]])
    rbm2 = BernoulliRBM(n_components=2, batch_size=2, n_iter=42, random_state=rng)
    rbm2.fit(X)
    assert_almost_equal(rbm2.components_, np.array([[0.02649814], [0.02009084]]), decimal=4)
    assert_almost_equal(rbm2.gibbs(X), X.toarray())
    assert_almost_equal(rbm1.components_, rbm2.components_)

def test_gibbs_smoke():
    if False:
        while True:
            i = 10
    X = Xdigits
    rbm1 = BernoulliRBM(n_components=42, batch_size=40, n_iter=20, random_state=42)
    rbm1.fit(X)
    X_sampled = rbm1.gibbs(X)
    assert_all_finite(X_sampled)
    X_sampled2 = rbm1.gibbs(X)
    assert np.all((X_sampled != X_sampled2).max(axis=1))

@pytest.mark.parametrize('lil_containers', LIL_CONTAINERS)
def test_score_samples(lil_containers):
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.RandomState(42)
    X = np.vstack([np.zeros(1000), np.ones(1000)])
    rbm1 = BernoulliRBM(n_components=10, batch_size=2, n_iter=10, random_state=rng)
    rbm1.fit(X)
    assert (rbm1.score_samples(X) < -300).all()
    rbm1.random_state = 42
    d_score = rbm1.score_samples(X)
    rbm1.random_state = 42
    s_score = rbm1.score_samples(lil_containers(X))
    assert_almost_equal(d_score, s_score)
    with np.errstate(under='ignore'):
        rbm1.score_samples([np.arange(1000) * 100])

def test_rbm_verbose():
    if False:
        print('Hello World!')
    rbm = BernoulliRBM(n_iter=2, verbose=10)
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        rbm.fit(Xdigits)
    finally:
        sys.stdout = old_stdout

@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_sparse_and_verbose(csc_container):
    if False:
        i = 10
        return i + 15
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    X = csc_container([[0.0], [1.0]])
    rbm = BernoulliRBM(n_components=2, batch_size=2, n_iter=1, random_state=42, verbose=True)
    try:
        rbm.fit(X)
        s = sys.stdout.getvalue()
        assert re.match('\\[BernoulliRBM\\] Iteration 1, pseudo-likelihood = -?(\\d)+(\\.\\d+)?, time = (\\d|\\.)+s', s)
    finally:
        sys.stdout = old_stdout

@pytest.mark.parametrize('dtype_in, dtype_out', [(np.float32, np.float32), (np.float64, np.float64), (int, np.float64)])
def test_transformer_dtypes_casting(dtype_in, dtype_out):
    if False:
        i = 10
        return i + 15
    X = Xdigits[:100].astype(dtype_in)
    rbm = BernoulliRBM(n_components=16, batch_size=5, n_iter=5, random_state=42)
    Xt = rbm.fit_transform(X)
    assert Xt.dtype == dtype_out, 'transform dtype: {} - original dtype: {}'.format(Xt.dtype, X.dtype)

def test_convergence_dtype_consistency():
    if False:
        i = 10
        return i + 15
    X_64 = Xdigits[:100].astype(np.float64)
    rbm_64 = BernoulliRBM(n_components=16, batch_size=5, n_iter=5, random_state=42)
    Xt_64 = rbm_64.fit_transform(X_64)
    X_32 = Xdigits[:100].astype(np.float32)
    rbm_32 = BernoulliRBM(n_components=16, batch_size=5, n_iter=5, random_state=42)
    Xt_32 = rbm_32.fit_transform(X_32)
    assert_allclose(Xt_64, Xt_32, rtol=1e-06, atol=0)
    assert_allclose(rbm_64.intercept_hidden_, rbm_32.intercept_hidden_, rtol=1e-06, atol=0)
    assert_allclose(rbm_64.intercept_visible_, rbm_32.intercept_visible_, rtol=1e-05, atol=0)
    assert_allclose(rbm_64.components_, rbm_32.components_, rtol=0.001, atol=0)
    assert_allclose(rbm_64.h_samples_, rbm_32.h_samples_)

@pytest.mark.parametrize('method', ['fit', 'partial_fit'])
def test_feature_names_out(method):
    if False:
        return 10
    'Check `get_feature_names_out` for `BernoulliRBM`.'
    n_components = 10
    rbm = BernoulliRBM(n_components=n_components)
    getattr(rbm, method)(Xdigits)
    names = rbm.get_feature_names_out()
    expected_names = [f'bernoullirbm{i}' for i in range(n_components)]
    assert_array_equal(expected_names, names)