import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except:
    pass
import numpy as np
import scipy.sparse
import librosa
import sklearn.decomposition
import pytest
from test_core import srand

def test_default_decompose():
    if False:
        print('Hello World!')
    X = np.array([[1, 2, 3, 4, 5, 6], [1, 1, 1.2, 1, 0.8, 1]])
    (W, H) = librosa.decompose.decompose(X, random_state=0)
    assert np.allclose(X, W.dot(H), rtol=0.01, atol=0.01)

def test_given_decompose():
    if False:
        for i in range(10):
            print('nop')
    D = sklearn.decomposition.NMF(random_state=0)
    X = np.array([[1, 2, 3, 4, 5, 6], [1, 1, 1.2, 1, 0.8, 1]])
    (W, H) = librosa.decompose.decompose(X, transformer=D)
    assert np.allclose(X, W.dot(H), rtol=0.01, atol=0.01)

def test_decompose_fit():
    if False:
        i = 10
        return i + 15
    srand()
    D = sklearn.decomposition.NMF(random_state=0)
    X = np.array([[1, 2, 3, 4, 5, 6], [1, 1, 1.2, 1, 0.8, 1]])
    (W, H) = librosa.decompose.decompose(X, transformer=D, fit=True)
    X = np.asarray(np.random.randn(*X.shape) ** 2)
    (W2, H2) = librosa.decompose.decompose(X, transformer=D, fit=False)
    assert np.allclose(W, W2)

@pytest.mark.xfail(raises=librosa.ParameterError)
def test_decompose_multi_sort():
    if False:
        print('Hello World!')
    librosa.decompose.decompose(np.zeros((3, 3, 3)), sort=True)

def test_decompose_multi():
    if False:
        print('Hello World!')
    srand()
    X = np.random.random_sample(size=(2, 20, 100))
    (components, activations) = librosa.decompose.decompose(X, n_components=20, random_state=0)
    Xflat = np.vstack([X[0], X[1]])
    (c_flat, a_flat) = librosa.decompose.decompose(Xflat, n_components=20, random_state=0)
    assert np.allclose(c_flat[:X.shape[1]], components[0])
    assert np.allclose(c_flat[X.shape[1]:], components[1])
    assert np.allclose(activations, a_flat)

@pytest.mark.xfail(raises=librosa.ParameterError)
def test_decompose_fit_false():
    if False:
        print('Hello World!')
    X = np.array([[1, 2, 3, 4, 5, 6], [1, 1, 1.2, 1, 0.8, 1]])
    (W, H) = librosa.decompose.decompose(X, fit=False)

def test_sorted_decompose():
    if False:
        for i in range(10):
            print('nop')
    X = np.array([[1, 2, 3, 4, 5, 6], [1, 1, 1.2, 1, 0.8, 1]])
    (W, H) = librosa.decompose.decompose(X, sort=True, random_state=0)
    assert np.allclose(X, W.dot(H), rtol=0.01, atol=0.01)

@pytest.fixture
def y22050():
    if False:
        while True:
            i = 10
    (y, _) = librosa.load(os.path.join('tests', 'data', 'test1_22050.wav'))
    return y

@pytest.fixture
def D22050(y22050):
    if False:
        while True:
            i = 10
    return librosa.stft(y22050)

@pytest.fixture
def S22050(D22050):
    if False:
        for i in range(10):
            print('nop')
    return np.abs(D22050)

@pytest.mark.parametrize('window', [31, (5, 5)])
@pytest.mark.parametrize('power', [1, 2, 10])
@pytest.mark.parametrize('mask', [False, True])
@pytest.mark.parametrize('margin', [1.0, 3.0, (1.0, 1.0), (9.0, 10.0)])
def test_real_hpss(S22050, window, power, mask, margin):
    if False:
        return 10
    (H, P) = librosa.decompose.hpss(S22050, kernel_size=window, power=power, mask=mask, margin=margin)
    if margin == 1.0 or margin == (1.0, 1.0):
        if mask:
            assert np.allclose(H + P, np.ones_like(S22050))
        else:
            assert np.allclose(H + P, S22050)
    elif mask:
        assert np.all(H + P <= np.ones_like(S22050))
    else:
        assert np.all(H + P <= S22050)

@pytest.mark.xfail(raises=librosa.ParameterError)
def test_hpss_margin_error(S22050):
    if False:
        return 10
    (H, P) = librosa.decompose.hpss(S22050, margin=0.9)

def test_complex_hpss(D22050):
    if False:
        print('Hello World!')
    (H, P) = librosa.decompose.hpss(D22050)
    assert np.allclose(H + P, D22050)

def test_nn_filter_mean():
    if False:
        i = 10
        return i + 15
    srand()
    X = np.random.randn(10, 100)
    rec = librosa.segment.recurrence_matrix(X)
    X_filtered = librosa.decompose.nn_filter(X)
    rec = librosa.util.normalize(rec.astype(float), axis=0, norm=1)
    assert np.allclose(X_filtered, X.dot(rec))

def test_nn_filter_mean_rec():
    if False:
        i = 10
        return i + 15
    srand()
    X = np.random.randn(10, 100)
    rec = librosa.segment.recurrence_matrix(X)
    rec[:, :3] = False
    X_filtered = librosa.decompose.nn_filter(X, rec=rec)
    for i in range(3):
        assert np.allclose(X_filtered[:, i], X[:, i])
    rec = librosa.util.normalize(rec.astype(float), axis=0, norm=1)
    assert np.allclose(X_filtered[:, 3:], X.dot(rec)[:, 3:])

def test_nn_filter_mean_rec_sparse():
    if False:
        for i in range(10):
            print('nop')
    srand()
    X = np.random.randn(10, 100)
    rec = librosa.segment.recurrence_matrix(X, sparse=True)
    X_filtered = librosa.decompose.nn_filter(X, rec=rec)
    rec = librosa.util.normalize(rec.toarray().astype(float), axis=0, norm=1)
    assert np.allclose(X_filtered, X.dot(rec))

@pytest.fixture(scope='module')
def s_multi():
    if False:
        i = 10
        return i + 15
    (y, sr) = librosa.load(os.path.join('tests', 'data', 'test1_44100.wav'), sr=None, mono=False)
    return np.abs(librosa.stft(y))

@pytest.mark.parametrize('useR,sparse', [(False, False), (True, False), (True, True)])
def test_nn_filter_multi(s_multi, useR, sparse):
    if False:
        return 10
    R = librosa.segment.recurrence_matrix(s_multi, mode='affinity', sparse=sparse)
    if useR:
        R_multi = R
    else:
        R_multi = None
    s_filt = librosa.decompose.nn_filter(s_multi, rec=R_multi, mode='affinity', sparse=sparse)
    s_filt0 = librosa.decompose.nn_filter(s_multi[0], rec=R)
    s_filt1 = librosa.decompose.nn_filter(s_multi[1], rec=R)
    assert np.allclose(s_filt[0], s_filt0)
    assert np.allclose(s_filt[1], s_filt1)
    assert not np.allclose(s_filt0, s_filt1)

def test_nn_filter_avg():
    if False:
        return 10
    srand()
    X = np.random.randn(10, 100)
    rec = librosa.segment.recurrence_matrix(X, mode='affinity')
    X_filtered = librosa.decompose.nn_filter(X, rec=rec, aggregate=np.average)
    rec = librosa.util.normalize(rec, axis=0, norm=1)
    assert np.allclose(X_filtered, X.dot(rec))

@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize('x,y', [(10, 10), (100, 20), (20, 100), (100, 101), (101, 101)])
@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('data', [np.zeros((10, 100))])
def test_nn_filter_badselfsim(data, x, y, sparse):
    if False:
        while True:
            i = 10
    srand()
    rec = np.random.randn(x, y)
    if sparse:
        rec = scipy.sparse.csr_matrix(rec)
    librosa.decompose.nn_filter(data, rec=rec)