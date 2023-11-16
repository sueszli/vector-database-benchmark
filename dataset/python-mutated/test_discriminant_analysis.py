import numpy as np
import pytest
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf, ShrunkCovariance, ledoit_wolf
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis, _cov
from sklearn.preprocessing import StandardScaler
from sklearn.utils import _IS_WASM, check_random_state
from sklearn.utils._testing import _convert_container, assert_allclose, assert_almost_equal, assert_array_almost_equal, assert_array_equal
X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]], dtype='f')
y = np.array([1, 1, 1, 2, 2, 2])
y3 = np.array([1, 1, 2, 2, 3, 3])
X1 = np.array([[-2], [-1], [-1], [1], [1], [2]], dtype='f')
X6 = np.array([[0, 0], [-2, -2], [-2, -1], [-1, -1], [-1, -2], [1, 3], [1, 2], [2, 1], [2, 2]])
y6 = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2])
y7 = np.array([1, 2, 3, 2, 3, 1, 2, 3, 1])
X7 = np.array([[-3], [-2], [-1], [-1], [0], [1], [1], [2], [3]])
X2 = np.array([[-3, 0], [-2, 0], [-1, 0], [-1, 0], [0, 0], [1, 0], [1, 0], [2, 0], [3, 0]])
y4 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2])
X5 = np.c_[np.arange(8), np.zeros((8, 3))]
y5 = np.array([0, 0, 0, 0, 0, 1, 1, 1])
solver_shrinkage = [('svd', None), ('lsqr', None), ('eigen', None), ('lsqr', 'auto'), ('lsqr', 0), ('lsqr', 0.43), ('eigen', 'auto'), ('eigen', 0), ('eigen', 0.43)]

def test_lda_predict():
    if False:
        i = 10
        return i + 15
    for test_case in solver_shrinkage:
        (solver, shrinkage) = test_case
        clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
        y_pred = clf.fit(X, y).predict(X)
        assert_array_equal(y_pred, y, 'solver %s' % solver)
        y_pred1 = clf.fit(X1, y).predict(X1)
        assert_array_equal(y_pred1, y, 'solver %s' % solver)
        y_proba_pred1 = clf.predict_proba(X1)
        assert_array_equal((y_proba_pred1[:, 1] > 0.5) + 1, y, 'solver %s' % solver)
        y_log_proba_pred1 = clf.predict_log_proba(X1)
        assert_allclose(np.exp(y_log_proba_pred1), y_proba_pred1, rtol=1e-06, atol=1e-06, err_msg='solver %s' % solver)
        y_pred3 = clf.fit(X, y3).predict(X)
        assert np.any(y_pred3 != y3), 'solver %s' % solver
    clf = LinearDiscriminantAnalysis(solver='svd', shrinkage='auto')
    with pytest.raises(NotImplementedError):
        clf.fit(X, y)
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.1, covariance_estimator=ShrunkCovariance())
    with pytest.raises(ValueError, match='covariance_estimator and shrinkage parameters are not None. Only one of the two can be set.'):
        clf.fit(X, y)
    clf = LinearDiscriminantAnalysis(solver='svd', covariance_estimator=LedoitWolf())
    with pytest.raises(ValueError, match='covariance estimator is not supported with svd'):
        clf.fit(X, y)
    clf = LinearDiscriminantAnalysis(solver='lsqr', covariance_estimator=KMeans(n_clusters=2, n_init='auto'))
    with pytest.raises(ValueError):
        clf.fit(X, y)

@pytest.mark.parametrize('n_classes', [2, 3])
@pytest.mark.parametrize('solver', ['svd', 'lsqr', 'eigen'])
def test_lda_predict_proba(solver, n_classes):
    if False:
        while True:
            i = 10

    def generate_dataset(n_samples, centers, covariances, random_state=None):
        if False:
            return 10
        'Generate a multivariate normal data given some centers and\n        covariances'
        rng = check_random_state(random_state)
        X = np.vstack([rng.multivariate_normal(mean, cov, size=n_samples // len(centers)) for (mean, cov) in zip(centers, covariances)])
        y = np.hstack([[clazz] * (n_samples // len(centers)) for clazz in range(len(centers))])
        return (X, y)
    blob_centers = np.array([[0, 0], [-10, 40], [-30, 30]])[:n_classes]
    blob_stds = np.array([[[10, 10], [10, 100]]] * len(blob_centers))
    (X, y) = generate_dataset(n_samples=90000, centers=blob_centers, covariances=blob_stds, random_state=42)
    lda = LinearDiscriminantAnalysis(solver=solver, store_covariance=True, shrinkage=None).fit(X, y)
    assert_allclose(lda.means_, blob_centers, atol=0.1)
    assert_allclose(lda.covariance_, blob_stds[0], atol=1)
    precision = linalg.inv(blob_stds[0])
    alpha_k = []
    alpha_k_0 = []
    for clazz in range(len(blob_centers) - 1):
        alpha_k.append(np.dot(precision, (blob_centers[clazz] - blob_centers[-1])[:, np.newaxis]))
        alpha_k_0.append(np.dot(-0.5 * (blob_centers[clazz] + blob_centers[-1])[np.newaxis, :], alpha_k[-1]))
    sample = np.array([[-22, 22]])

    def discriminant_func(sample, coef, intercept, clazz):
        if False:
            i = 10
            return i + 15
        return np.exp(intercept[clazz] + np.dot(sample, coef[clazz])).item()
    prob = np.array([float(discriminant_func(sample, alpha_k, alpha_k_0, clazz) / (1 + sum([discriminant_func(sample, alpha_k, alpha_k_0, clazz) for clazz in range(n_classes - 1)]))) for clazz in range(n_classes - 1)])
    prob_ref = 1 - np.sum(prob)
    prob_ref_2 = float(1 / (1 + sum([discriminant_func(sample, alpha_k, alpha_k_0, clazz) for clazz in range(n_classes - 1)])))
    assert prob_ref == pytest.approx(prob_ref_2)
    assert_allclose(lda.predict_proba(sample), np.hstack([prob, prob_ref])[np.newaxis], atol=0.01)

def test_lda_priors():
    if False:
        i = 10
        return i + 15
    priors = np.array([0.5, -0.5])
    clf = LinearDiscriminantAnalysis(priors=priors)
    msg = 'priors must be non-negative'
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)
    clf = LinearDiscriminantAnalysis(priors=[0.5, 0.5])
    clf.fit(X, y)
    priors = np.array([0.5, 0.6])
    prior_norm = np.array([0.45, 0.55])
    clf = LinearDiscriminantAnalysis(priors=priors)
    with pytest.warns(UserWarning):
        clf.fit(X, y)
    assert_array_almost_equal(clf.priors_, prior_norm, 2)

def test_lda_coefs():
    if False:
        i = 10
        return i + 15
    n_features = 2
    n_classes = 2
    n_samples = 1000
    (X, y) = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_classes, random_state=11)
    clf_lda_svd = LinearDiscriminantAnalysis(solver='svd')
    clf_lda_lsqr = LinearDiscriminantAnalysis(solver='lsqr')
    clf_lda_eigen = LinearDiscriminantAnalysis(solver='eigen')
    clf_lda_svd.fit(X, y)
    clf_lda_lsqr.fit(X, y)
    clf_lda_eigen.fit(X, y)
    assert_array_almost_equal(clf_lda_svd.coef_, clf_lda_lsqr.coef_, 1)
    assert_array_almost_equal(clf_lda_svd.coef_, clf_lda_eigen.coef_, 1)
    assert_array_almost_equal(clf_lda_eigen.coef_, clf_lda_lsqr.coef_, 1)

def test_lda_transform():
    if False:
        i = 10
        return i + 15
    clf = LinearDiscriminantAnalysis(solver='svd', n_components=1)
    X_transformed = clf.fit(X, y).transform(X)
    assert X_transformed.shape[1] == 1
    clf = LinearDiscriminantAnalysis(solver='eigen', n_components=1)
    X_transformed = clf.fit(X, y).transform(X)
    assert X_transformed.shape[1] == 1
    clf = LinearDiscriminantAnalysis(solver='lsqr', n_components=1)
    clf.fit(X, y)
    msg = "transform not implemented for 'lsqr'"
    with pytest.raises(NotImplementedError, match=msg):
        clf.transform(X)

def test_lda_explained_variance_ratio():
    if False:
        print('Hello World!')
    state = np.random.RandomState(0)
    X = state.normal(loc=0, scale=100, size=(40, 20))
    y = state.randint(0, 3, size=(40,))
    clf_lda_eigen = LinearDiscriminantAnalysis(solver='eigen')
    clf_lda_eigen.fit(X, y)
    assert_almost_equal(clf_lda_eigen.explained_variance_ratio_.sum(), 1.0, 3)
    assert clf_lda_eigen.explained_variance_ratio_.shape == (2,), 'Unexpected length for explained_variance_ratio_'
    clf_lda_svd = LinearDiscriminantAnalysis(solver='svd')
    clf_lda_svd.fit(X, y)
    assert_almost_equal(clf_lda_svd.explained_variance_ratio_.sum(), 1.0, 3)
    assert clf_lda_svd.explained_variance_ratio_.shape == (2,), 'Unexpected length for explained_variance_ratio_'
    assert_array_almost_equal(clf_lda_svd.explained_variance_ratio_, clf_lda_eigen.explained_variance_ratio_)

def test_lda_orthogonality():
    if False:
        while True:
            i = 10
    means = np.array([[0, 0, -1], [0, 2, 0], [0, -2, 0], [0, 0, 5]])
    scatter = np.array([[0.1, 0, 0], [-0.1, 0, 0], [0, 0.1, 0], [0, -0.1, 0], [0, 0, 0.1], [0, 0, -0.1]])
    X = (means[:, np.newaxis, :] + scatter[np.newaxis, :, :]).reshape((-1, 3))
    y = np.repeat(np.arange(means.shape[0]), scatter.shape[0])
    clf = LinearDiscriminantAnalysis(solver='svd').fit(X, y)
    means_transformed = clf.transform(means)
    d1 = means_transformed[3] - means_transformed[0]
    d2 = means_transformed[2] - means_transformed[1]
    d1 /= np.sqrt(np.sum(d1 ** 2))
    d2 /= np.sqrt(np.sum(d2 ** 2))
    assert_almost_equal(np.cov(clf.transform(scatter).T), np.eye(2))
    assert_almost_equal(np.abs(np.dot(d1[:2], [1, 0])), 1.0)
    assert_almost_equal(np.abs(np.dot(d2[:2], [0, 1])), 1.0)

def test_lda_scaling():
    if False:
        while True:
            i = 10
    n = 100
    rng = np.random.RandomState(1234)
    x1 = rng.uniform(-1, 1, (n, 3)) + [-10, 0, 0]
    x2 = rng.uniform(-1, 1, (n, 3)) + [10, 0, 0]
    x = np.vstack((x1, x2)) * [1, 100, 10000]
    y = [-1] * n + [1] * n
    for solver in ('svd', 'lsqr', 'eigen'):
        clf = LinearDiscriminantAnalysis(solver=solver)
        assert clf.fit(x, y).score(x, y) == 1.0, 'using covariance: %s' % solver

def test_lda_store_covariance():
    if False:
        print('Hello World!')
    for solver in ('lsqr', 'eigen'):
        clf = LinearDiscriminantAnalysis(solver=solver).fit(X6, y6)
        assert hasattr(clf, 'covariance_')
        clf = LinearDiscriminantAnalysis(solver=solver, store_covariance=True).fit(X6, y6)
        assert hasattr(clf, 'covariance_')
        assert_array_almost_equal(clf.covariance_, np.array([[0.422222, 0.088889], [0.088889, 0.533333]]))
    clf = LinearDiscriminantAnalysis(solver='svd').fit(X6, y6)
    assert not hasattr(clf, 'covariance_')
    clf = LinearDiscriminantAnalysis(solver=solver, store_covariance=True).fit(X6, y6)
    assert hasattr(clf, 'covariance_')
    assert_array_almost_equal(clf.covariance_, np.array([[0.422222, 0.088889], [0.088889, 0.533333]]))

@pytest.mark.parametrize('seed', range(10))
def test_lda_shrinkage(seed):
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.RandomState(seed)
    X = rng.rand(100, 10)
    y = rng.randint(3, size=100)
    c1 = LinearDiscriminantAnalysis(store_covariance=True, shrinkage=0.5, solver='lsqr')
    c2 = LinearDiscriminantAnalysis(store_covariance=True, covariance_estimator=ShrunkCovariance(shrinkage=0.5), solver='lsqr')
    c1.fit(X, y)
    c2.fit(X, y)
    assert_allclose(c1.means_, c2.means_)
    assert_allclose(c1.covariance_, c2.covariance_)

def test_lda_ledoitwolf():
    if False:
        print('Hello World!')

    class StandardizedLedoitWolf:

        def fit(self, X):
            if False:
                for i in range(10):
                    print('nop')
            sc = StandardScaler()
            X_sc = sc.fit_transform(X)
            s = ledoit_wolf(X_sc)[0]
            s = sc.scale_[:, np.newaxis] * s * sc.scale_[np.newaxis, :]
            self.covariance_ = s
    rng = np.random.RandomState(0)
    X = rng.rand(100, 10)
    y = rng.randint(3, size=(100,))
    c1 = LinearDiscriminantAnalysis(store_covariance=True, shrinkage='auto', solver='lsqr')
    c2 = LinearDiscriminantAnalysis(store_covariance=True, covariance_estimator=StandardizedLedoitWolf(), solver='lsqr')
    c1.fit(X, y)
    c2.fit(X, y)
    assert_allclose(c1.means_, c2.means_)
    assert_allclose(c1.covariance_, c2.covariance_)

@pytest.mark.parametrize('n_features', [3, 5])
@pytest.mark.parametrize('n_classes', [5, 3])
def test_lda_dimension_warning(n_classes, n_features):
    if False:
        while True:
            i = 10
    rng = check_random_state(0)
    n_samples = 10
    X = rng.randn(n_samples, n_features)
    y = np.tile(range(n_classes), n_samples // n_classes + 1)[:n_samples]
    max_components = min(n_features, n_classes - 1)
    for n_components in [max_components - 1, None, max_components]:
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        lda.fit(X, y)
    for n_components in [max_components + 1, max(n_features, n_classes - 1) + 1]:
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        msg = 'n_components cannot be larger than '
        with pytest.raises(ValueError, match=msg):
            lda.fit(X, y)

@pytest.mark.parametrize('data_type, expected_type', [(np.float32, np.float32), (np.float64, np.float64), (np.int32, np.float64), (np.int64, np.float64)])
def test_lda_dtype_match(data_type, expected_type):
    if False:
        print('Hello World!')
    for (solver, shrinkage) in solver_shrinkage:
        clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
        clf.fit(X.astype(data_type), y.astype(data_type))
        assert clf.coef_.dtype == expected_type

def test_lda_numeric_consistency_float32_float64():
    if False:
        for i in range(10):
            print('nop')
    for (solver, shrinkage) in solver_shrinkage:
        clf_32 = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
        clf_32.fit(X.astype(np.float32), y.astype(np.float32))
        clf_64 = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
        clf_64.fit(X.astype(np.float64), y.astype(np.float64))
        rtol = 1e-06
        assert_allclose(clf_32.coef_, clf_64.coef_, rtol=rtol)

def test_qda():
    if False:
        while True:
            i = 10
    clf = QuadraticDiscriminantAnalysis()
    y_pred = clf.fit(X6, y6).predict(X6)
    assert_array_equal(y_pred, y6)
    y_pred1 = clf.fit(X7, y6).predict(X7)
    assert_array_equal(y_pred1, y6)
    y_proba_pred1 = clf.predict_proba(X7)
    assert_array_equal((y_proba_pred1[:, 1] > 0.5) + 1, y6)
    y_log_proba_pred1 = clf.predict_log_proba(X7)
    assert_array_almost_equal(np.exp(y_log_proba_pred1), y_proba_pred1, 8)
    y_pred3 = clf.fit(X6, y7).predict(X6)
    assert np.any(y_pred3 != y7)
    with pytest.raises(ValueError):
        clf.fit(X6, y4)

def test_qda_priors():
    if False:
        print('Hello World!')
    clf = QuadraticDiscriminantAnalysis()
    y_pred = clf.fit(X6, y6).predict(X6)
    n_pos = np.sum(y_pred == 2)
    neg = 1e-10
    clf = QuadraticDiscriminantAnalysis(priors=np.array([neg, 1 - neg]))
    y_pred = clf.fit(X6, y6).predict(X6)
    n_pos2 = np.sum(y_pred == 2)
    assert n_pos2 > n_pos

@pytest.mark.parametrize('priors_type', ['list', 'tuple', 'array'])
def test_qda_prior_type(priors_type):
    if False:
        for i in range(10):
            print('nop')
    'Check that priors accept array-like.'
    priors = [0.5, 0.5]
    clf = QuadraticDiscriminantAnalysis(priors=_convert_container([0.5, 0.5], priors_type)).fit(X6, y6)
    assert isinstance(clf.priors_, np.ndarray)
    assert_array_equal(clf.priors_, priors)

def test_qda_prior_copy():
    if False:
        print('Hello World!')
    "Check that altering `priors` without `fit` doesn't change `priors_`"
    priors = np.array([0.5, 0.5])
    qda = QuadraticDiscriminantAnalysis(priors=priors).fit(X, y)
    assert_array_equal(qda.priors_, qda.priors)
    priors[0] = 0.2
    assert qda.priors_[0] != qda.priors[0]

def test_qda_store_covariance():
    if False:
        print('Hello World!')
    clf = QuadraticDiscriminantAnalysis().fit(X6, y6)
    assert not hasattr(clf, 'covariance_')
    clf = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X6, y6)
    assert hasattr(clf, 'covariance_')
    assert_array_almost_equal(clf.covariance_[0], np.array([[0.7, 0.45], [0.45, 0.7]]))
    assert_array_almost_equal(clf.covariance_[1], np.array([[0.33333333, -0.33333333], [-0.33333333, 0.66666667]]))

@pytest.mark.xfail(_IS_WASM, reason='no floating point exceptions, see https://github.com/numpy/numpy/pull/21895#issuecomment-1311525881')
def test_qda_regularization():
    if False:
        for i in range(10):
            print('nop')
    collinear_msg = 'Variables are collinear'
    clf = QuadraticDiscriminantAnalysis()
    with pytest.warns(UserWarning, match=collinear_msg):
        y_pred = clf.fit(X2, y6)
    with pytest.warns(RuntimeWarning, match='divide by zero'):
        y_pred = clf.predict(X2)
    assert np.any(y_pred != y6)
    clf = QuadraticDiscriminantAnalysis(reg_param=0.01)
    with pytest.warns(UserWarning, match=collinear_msg):
        clf.fit(X2, y6)
    y_pred = clf.predict(X2)
    assert_array_equal(y_pred, y6)
    clf = QuadraticDiscriminantAnalysis(reg_param=0.1)
    with pytest.warns(UserWarning, match=collinear_msg):
        clf.fit(X5, y5)
    y_pred5 = clf.predict(X5)
    assert_array_equal(y_pred5, y5)

def test_covariance():
    if False:
        return 10
    (x, y) = make_blobs(n_samples=100, n_features=5, centers=1, random_state=42)
    x = np.dot(x, np.arange(x.shape[1] ** 2).reshape(x.shape[1], x.shape[1]))
    c_e = _cov(x, 'empirical')
    assert_almost_equal(c_e, c_e.T)
    c_s = _cov(x, 'auto')
    assert_almost_equal(c_s, c_s.T)

@pytest.mark.parametrize('solver', ['svd', 'lsqr', 'eigen'])
def test_raises_value_error_on_same_number_of_classes_and_samples(solver):
    if False:
        return 10
    '\n    Tests that if the number of samples equals the number\n    of classes, a ValueError is raised.\n    '
    X = np.array([[0.5, 0.6], [0.6, 0.5]])
    y = np.array(['a', 'b'])
    clf = LinearDiscriminantAnalysis(solver=solver)
    with pytest.raises(ValueError, match='The number of samples must be more'):
        clf.fit(X, y)

def test_get_feature_names_out():
    if False:
        for i in range(10):
            print('nop')
    'Check get_feature_names_out uses class name as prefix.'
    est = LinearDiscriminantAnalysis().fit(X, y)
    names_out = est.get_feature_names_out()
    class_name_lower = 'LinearDiscriminantAnalysis'.lower()
    expected_names_out = np.array([f'{class_name_lower}{i}' for i in range(est.explained_variance_ratio_.shape[0])], dtype=object)
    assert_array_equal(names_out, expected_names_out)