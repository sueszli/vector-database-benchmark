"""
Testing for Neighborhood Component Analysis module (sklearn.neighbors.nca)
"""
import re
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.optimize import check_grad
from sklearn import clone
from sklearn.datasets import load_iris, make_blobs, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
rng = check_random_state(0)
iris = load_iris()
perm = rng.permutation(iris.target.size)
iris_data = iris.data[perm]
iris_target = iris.target[perm]
EPS = np.finfo(float).eps

def test_simple_example():
    if False:
        i = 10
        return i + 15
    'Test on a simple example.\n\n    Puts four points in the input space where the opposite labels points are\n    next to each other. After transform the samples from the same class\n    should be next to each other.\n\n    '
    X = np.array([[0, 0], [0, 1], [2, 0], [2, 1]])
    y = np.array([1, 0, 1, 0])
    nca = NeighborhoodComponentsAnalysis(n_components=2, init='identity', random_state=42)
    nca.fit(X, y)
    X_t = nca.transform(X)
    assert_array_equal(pairwise_distances(X_t).argsort()[:, 1], np.array([2, 3, 0, 1]))

def test_toy_example_collapse_points():
    if False:
        print('Hello World!')
    'Test on a toy example of three points that should collapse\n\n    We build a simple example: two points from the same class and a point from\n    a different class in the middle of them. On this simple example, the new\n    (transformed) points should all collapse into one single point. Indeed, the\n    objective is 2/(1 + exp(d/2)), with d the euclidean distance between the\n    two samples from the same class. This is maximized for d=0 (because d>=0),\n    with an objective equal to 1 (loss=-1.).\n\n    '
    rng = np.random.RandomState(42)
    input_dim = 5
    two_points = rng.randn(2, input_dim)
    X = np.vstack([two_points, two_points.mean(axis=0)[np.newaxis, :]])
    y = [0, 0, 1]

    class LossStorer:

        def __init__(self, X, y):
            if False:
                while True:
                    i = 10
            self.loss = np.inf
            self.fake_nca = NeighborhoodComponentsAnalysis()
            self.fake_nca.n_iter_ = np.inf
            (self.X, y) = self.fake_nca._validate_data(X, y, ensure_min_samples=2)
            y = LabelEncoder().fit_transform(y)
            self.same_class_mask = y[:, np.newaxis] == y[np.newaxis, :]

        def callback(self, transformation, n_iter):
            if False:
                print('Hello World!')
            'Stores the last value of the loss function'
            (self.loss, _) = self.fake_nca._loss_grad_lbfgs(transformation, self.X, self.same_class_mask, -1.0)
    loss_storer = LossStorer(X, y)
    nca = NeighborhoodComponentsAnalysis(random_state=42, callback=loss_storer.callback)
    X_t = nca.fit_transform(X, y)
    print(X_t)
    assert_array_almost_equal(X_t - X_t[0], 0.0)
    assert abs(loss_storer.loss + 1) < 1e-10

def test_finite_differences(global_random_seed):
    if False:
        while True:
            i = 10
    'Test gradient of loss function\n\n    Assert that the gradient is almost equal to its finite differences\n    approximation.\n    '
    rng = np.random.RandomState(global_random_seed)
    (X, y) = make_classification(random_state=global_random_seed)
    M = rng.randn(rng.randint(1, X.shape[1] + 1), X.shape[1])
    nca = NeighborhoodComponentsAnalysis()
    nca.n_iter_ = 0
    mask = y[:, np.newaxis] == y[np.newaxis, :]

    def fun(M):
        if False:
            i = 10
            return i + 15
        return nca._loss_grad_lbfgs(M, X, mask)[0]

    def grad(M):
        if False:
            while True:
                i = 10
        return nca._loss_grad_lbfgs(M, X, mask)[1]
    diff = check_grad(fun, grad, M.ravel())
    assert diff == pytest.approx(0.0, abs=0.0001)

def test_params_validation():
    if False:
        while True:
            i = 10
    X = np.arange(12).reshape(4, 3)
    y = [1, 1, 2, 2]
    NCA = NeighborhoodComponentsAnalysis
    rng = np.random.RandomState(42)
    init = rng.rand(5, 3)
    msg = f'The output dimensionality ({init.shape[0]}) of the given linear transformation `init` cannot be greater than its input dimensionality ({init.shape[1]}).'
    with pytest.raises(ValueError, match=re.escape(msg)):
        NCA(init=init).fit(X, y)
    n_components = 10
    msg = f'The preferred dimensionality of the projected space `n_components` ({n_components}) cannot be greater than the given data dimensionality ({X.shape[1]})!'
    with pytest.raises(ValueError, match=re.escape(msg)):
        NCA(n_components=n_components).fit(X, y)

def test_transformation_dimensions():
    if False:
        return 10
    X = np.arange(12).reshape(4, 3)
    y = [1, 1, 2, 2]
    transformation = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        NeighborhoodComponentsAnalysis(init=transformation).fit(X, y)
    transformation = np.array([[1, 2], [3, 4], [5, 6]])
    with pytest.raises(ValueError):
        NeighborhoodComponentsAnalysis(init=transformation).fit(X, y)
    transformation = np.arange(9).reshape(3, 3)
    NeighborhoodComponentsAnalysis(init=transformation).fit(X, y)

def test_n_components():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(42)
    X = np.arange(12).reshape(4, 3)
    y = [1, 1, 2, 2]
    init = rng.rand(X.shape[1] - 1, 3)
    n_components = X.shape[1]
    nca = NeighborhoodComponentsAnalysis(init=init, n_components=n_components)
    msg = f'The preferred dimensionality of the projected space `n_components` ({n_components}) does not match the output dimensionality of the given linear transformation `init` ({init.shape[0]})!'
    with pytest.raises(ValueError, match=re.escape(msg)):
        nca.fit(X, y)
    n_components = X.shape[1] + 2
    nca = NeighborhoodComponentsAnalysis(init=init, n_components=n_components)
    msg = f'The preferred dimensionality of the projected space `n_components` ({n_components}) cannot be greater than the given data dimensionality ({X.shape[1]})!'
    with pytest.raises(ValueError, match=re.escape(msg)):
        nca.fit(X, y)
    nca = NeighborhoodComponentsAnalysis(n_components=2, init='identity')
    nca.fit(X, y)

def test_init_transformation():
    if False:
        return 10
    rng = np.random.RandomState(42)
    (X, y) = make_blobs(n_samples=30, centers=6, n_features=5, random_state=0)
    nca = NeighborhoodComponentsAnalysis(init='identity')
    nca.fit(X, y)
    nca_random = NeighborhoodComponentsAnalysis(init='random')
    nca_random.fit(X, y)
    nca_auto = NeighborhoodComponentsAnalysis(init='auto')
    nca_auto.fit(X, y)
    nca_pca = NeighborhoodComponentsAnalysis(init='pca')
    nca_pca.fit(X, y)
    nca_lda = NeighborhoodComponentsAnalysis(init='lda')
    nca_lda.fit(X, y)
    init = rng.rand(X.shape[1], X.shape[1])
    nca = NeighborhoodComponentsAnalysis(init=init)
    nca.fit(X, y)
    init = rng.rand(X.shape[1], X.shape[1] + 1)
    nca = NeighborhoodComponentsAnalysis(init=init)
    msg = f'The input dimensionality ({init.shape[1]}) of the given linear transformation `init` must match the dimensionality of the given inputs `X` ({X.shape[1]}).'
    with pytest.raises(ValueError, match=re.escape(msg)):
        nca.fit(X, y)
    init = rng.rand(X.shape[1] + 1, X.shape[1])
    nca = NeighborhoodComponentsAnalysis(init=init)
    msg = f'The output dimensionality ({init.shape[0]}) of the given linear transformation `init` cannot be greater than its input dimensionality ({init.shape[1]}).'
    with pytest.raises(ValueError, match=re.escape(msg)):
        nca.fit(X, y)
    init = rng.rand(X.shape[1], X.shape[1])
    n_components = X.shape[1] - 2
    nca = NeighborhoodComponentsAnalysis(init=init, n_components=n_components)
    msg = f'The preferred dimensionality of the projected space `n_components` ({n_components}) does not match the output dimensionality of the given linear transformation `init` ({init.shape[0]})!'
    with pytest.raises(ValueError, match=re.escape(msg)):
        nca.fit(X, y)

@pytest.mark.parametrize('n_samples', [3, 5, 7, 11])
@pytest.mark.parametrize('n_features', [3, 5, 7, 11])
@pytest.mark.parametrize('n_classes', [5, 7, 11])
@pytest.mark.parametrize('n_components', [3, 5, 7, 11])
def test_auto_init(n_samples, n_features, n_classes, n_components):
    if False:
        print('Hello World!')
    rng = np.random.RandomState(42)
    nca_base = NeighborhoodComponentsAnalysis(init='auto', n_components=n_components, max_iter=1, random_state=rng)
    if n_classes >= n_samples:
        pass
    else:
        X = rng.randn(n_samples, n_features)
        y = np.tile(range(n_classes), n_samples // n_classes + 1)[:n_samples]
        if n_components > n_features:
            pass
        else:
            nca = clone(nca_base)
            nca.fit(X, y)
            if n_components <= min(n_classes - 1, n_features):
                nca_other = clone(nca_base).set_params(init='lda')
            elif n_components < min(n_features, n_samples):
                nca_other = clone(nca_base).set_params(init='pca')
            else:
                nca_other = clone(nca_base).set_params(init='identity')
            nca_other.fit(X, y)
            assert_array_almost_equal(nca.components_, nca_other.components_)

def test_warm_start_validation():
    if False:
        print('Hello World!')
    (X, y) = make_classification(n_samples=30, n_features=5, n_classes=4, n_redundant=0, n_informative=5, random_state=0)
    nca = NeighborhoodComponentsAnalysis(warm_start=True, max_iter=5)
    nca.fit(X, y)
    (X_less_features, y) = make_classification(n_samples=30, n_features=4, n_classes=4, n_redundant=0, n_informative=4, random_state=0)
    msg = f'The new inputs dimensionality ({X_less_features.shape[1]}) does not match the input dimensionality of the previously learned transformation ({nca.components_.shape[1]}).'
    with pytest.raises(ValueError, match=re.escape(msg)):
        nca.fit(X_less_features, y)

def test_warm_start_effectiveness():
    if False:
        while True:
            i = 10
    nca_warm = NeighborhoodComponentsAnalysis(warm_start=True, random_state=0)
    nca_warm.fit(iris_data, iris_target)
    transformation_warm = nca_warm.components_
    nca_warm.max_iter = 1
    nca_warm.fit(iris_data, iris_target)
    transformation_warm_plus_one = nca_warm.components_
    nca_cold = NeighborhoodComponentsAnalysis(warm_start=False, random_state=0)
    nca_cold.fit(iris_data, iris_target)
    transformation_cold = nca_cold.components_
    nca_cold.max_iter = 1
    nca_cold.fit(iris_data, iris_target)
    transformation_cold_plus_one = nca_cold.components_
    diff_warm = np.sum(np.abs(transformation_warm_plus_one - transformation_warm))
    diff_cold = np.sum(np.abs(transformation_cold_plus_one - transformation_cold))
    assert diff_warm < 3.0, 'Transformer changed significantly after one iteration even though it was warm-started.'
    assert diff_cold > diff_warm, 'Cold-started transformer changed less significantly than warm-started transformer after one iteration.'

@pytest.mark.parametrize('init_name', ['pca', 'lda', 'identity', 'random', 'precomputed'])
def test_verbose(init_name, capsys):
    if False:
        print('Hello World!')
    rng = np.random.RandomState(42)
    (X, y) = make_blobs(n_samples=30, centers=6, n_features=5, random_state=0)
    regexp_init = '... done in \\ *\\d+\\.\\d{2}s'
    msgs = {'pca': 'Finding principal components' + regexp_init, 'lda': 'Finding most discriminative components' + regexp_init}
    if init_name == 'precomputed':
        init = rng.randn(X.shape[1], X.shape[1])
    else:
        init = init_name
    nca = NeighborhoodComponentsAnalysis(verbose=1, init=init)
    nca.fit(X, y)
    (out, _) = capsys.readouterr()
    lines = re.split('\n+', out)
    if init_name in ['pca', 'lda']:
        assert re.match(msgs[init_name], lines[0])
        lines = lines[1:]
    assert lines[0] == '[NeighborhoodComponentsAnalysis]'
    header = '{:>10} {:>20} {:>10}'.format('Iteration', 'Objective Value', 'Time(s)')
    assert lines[1] == '[NeighborhoodComponentsAnalysis] {}'.format(header)
    assert lines[2] == '[NeighborhoodComponentsAnalysis] {}'.format('-' * len(header))
    for line in lines[3:-2]:
        assert re.match('\\[NeighborhoodComponentsAnalysis\\] *\\d+ *\\d\\.\\d{6}e[+|-]\\d+\\ *\\d+\\.\\d{2}', line)
    assert re.match('\\[NeighborhoodComponentsAnalysis\\] Training took\\ *\\d+\\.\\d{2}s\\.', lines[-2])
    assert lines[-1] == ''

def test_no_verbose(capsys):
    if False:
        i = 10
        return i + 15
    nca = NeighborhoodComponentsAnalysis()
    nca.fit(iris_data, iris_target)
    (out, _) = capsys.readouterr()
    assert out == ''

def test_singleton_class():
    if False:
        return 10
    X = iris_data
    y = iris_target
    singleton_class = 1
    (ind_singleton,) = np.where(y == singleton_class)
    y[ind_singleton] = 2
    y[ind_singleton[0]] = singleton_class
    nca = NeighborhoodComponentsAnalysis(max_iter=30)
    nca.fit(X, y)
    (ind_1,) = np.where(y == 1)
    (ind_2,) = np.where(y == 2)
    y[ind_1] = 0
    y[ind_1[0]] = 1
    y[ind_2] = 0
    y[ind_2[0]] = 2
    nca = NeighborhoodComponentsAnalysis(max_iter=30)
    nca.fit(X, y)
    (ind_0,) = np.where(y == 0)
    (ind_1,) = np.where(y == 1)
    (ind_2,) = np.where(y == 2)
    X = X[[ind_0[0], ind_1[0], ind_2[0]]]
    y = y[[ind_0[0], ind_1[0], ind_2[0]]]
    nca = NeighborhoodComponentsAnalysis(init='identity', max_iter=30)
    nca.fit(X, y)
    assert_array_equal(X, nca.transform(X))

def test_one_class():
    if False:
        i = 10
        return i + 15
    X = iris_data[iris_target == 0]
    y = iris_target[iris_target == 0]
    nca = NeighborhoodComponentsAnalysis(max_iter=30, n_components=X.shape[1], init='identity')
    nca.fit(X, y)
    assert_array_equal(X, nca.transform(X))

def test_callback(capsys):
    if False:
        while True:
            i = 10
    max_iter = 10

    def my_cb(transformation, n_iter):
        if False:
            while True:
                i = 10
        assert transformation.shape == (iris_data.shape[1] ** 2,)
        rem_iter = max_iter - n_iter
        print('{} iterations remaining...'.format(rem_iter))
    nca = NeighborhoodComponentsAnalysis(max_iter=max_iter, callback=my_cb, verbose=1)
    nca.fit(iris_data, iris_target)
    (out, _) = capsys.readouterr()
    assert '{} iterations remaining...'.format(max_iter - 1) in out

def test_expected_transformation_shape():
    if False:
        print('Hello World!')
    'Test that the transformation has the expected shape.'
    X = iris_data
    y = iris_target

    class TransformationStorer:

        def __init__(self, X, y):
            if False:
                i = 10
                return i + 15
            self.fake_nca = NeighborhoodComponentsAnalysis()
            self.fake_nca.n_iter_ = np.inf
            (self.X, y) = self.fake_nca._validate_data(X, y, ensure_min_samples=2)
            y = LabelEncoder().fit_transform(y)
            self.same_class_mask = y[:, np.newaxis] == y[np.newaxis, :]

        def callback(self, transformation, n_iter):
            if False:
                while True:
                    i = 10
            'Stores the last value of the transformation taken as input by\n            the optimizer'
            self.transformation = transformation
    transformation_storer = TransformationStorer(X, y)
    cb = transformation_storer.callback
    nca = NeighborhoodComponentsAnalysis(max_iter=5, callback=cb)
    nca.fit(X, y)
    assert transformation_storer.transformation.size == X.shape[1] ** 2

def test_convergence_warning():
    if False:
        for i in range(10):
            print('nop')
    nca = NeighborhoodComponentsAnalysis(max_iter=2, verbose=1)
    cls_name = nca.__class__.__name__
    msg = '[{}] NCA did not converge'.format(cls_name)
    with pytest.warns(ConvergenceWarning, match=re.escape(msg)):
        nca.fit(iris_data, iris_target)

@pytest.mark.parametrize('param, value', [('n_components', np.int32(3)), ('max_iter', np.int32(100)), ('tol', np.float32(0.0001))])
def test_parameters_valid_types(param, value):
    if False:
        return 10
    nca = NeighborhoodComponentsAnalysis(**{param: value})
    X = iris_data
    y = iris_target
    nca.fit(X, y)

def test_nca_feature_names_out():
    if False:
        while True:
            i = 10
    'Check `get_feature_names_out` for `NeighborhoodComponentsAnalysis`.'
    X = iris_data
    y = iris_target
    est = NeighborhoodComponentsAnalysis().fit(X, y)
    names_out = est.get_feature_names_out()
    class_name_lower = est.__class__.__name__.lower()
    expected_names_out = np.array([f'{class_name_lower}{i}' for i in range(est.components_.shape[1])], dtype=object)
    assert_array_equal(names_out, expected_names_out)