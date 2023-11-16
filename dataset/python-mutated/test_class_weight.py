import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._testing import assert_almost_equal, assert_array_almost_equal
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.utils.fixes import CSC_CONTAINERS

def test_compute_class_weight():
    if False:
        i = 10
        return i + 15
    y = np.asarray([2, 2, 2, 3, 3, 4])
    classes = np.unique(y)
    cw = compute_class_weight('balanced', classes=classes, y=y)
    class_counts = np.bincount(y)[2:]
    assert_almost_equal(np.dot(cw, class_counts), y.shape[0])
    assert cw[0] < cw[1] < cw[2]

@pytest.mark.parametrize('y_type, class_weight, classes, err_msg', [('numeric', 'balanced', np.arange(4), 'classes should have valid labels that are in y'), ('numeric', {'label_not_present': 1.0}, np.arange(4), 'The classes, \\[0, 1, 2, 3\\], are not in class_weight'), ('numeric', 'balanced', np.arange(2), 'classes should include all valid labels'), ('numeric', {0: 1.0, 1: 2.0}, np.arange(2), 'classes should include all valid labels'), ('string', {'dogs': 3, 'cat': 2}, np.array(['dog', 'cat']), "The classes, \\['dog'\\], are not in class_weight")])
def test_compute_class_weight_not_present(y_type, class_weight, classes, err_msg):
    if False:
        i = 10
        return i + 15
    y = np.asarray([0, 0, 0, 1, 1, 2]) if y_type == 'numeric' else np.asarray(['dog', 'cat', 'dog'])
    print(y)
    with pytest.raises(ValueError, match=err_msg):
        compute_class_weight(class_weight, classes=classes, y=y)

def test_compute_class_weight_dict():
    if False:
        for i in range(10):
            print('nop')
    classes = np.arange(3)
    class_weights = {0: 1.0, 1: 2.0, 2: 3.0}
    y = np.asarray([0, 0, 1, 2])
    cw = compute_class_weight(class_weights, classes=classes, y=y)
    assert_array_almost_equal(np.asarray([1.0, 2.0, 3.0]), cw)
    class_weights = {0: 1.0, 1: 2.0, 2: 3.0, 4: 1.5}
    cw = compute_class_weight(class_weights, classes=classes, y=y)
    assert_allclose([1.0, 2.0, 3.0], cw)
    class_weights = {-1: 5.0, 0: 4.0, 1: 2.0, 2: 3.0}
    cw = compute_class_weight(class_weights, classes=classes, y=y)
    assert_allclose([4.0, 2.0, 3.0], cw)

def test_compute_class_weight_invariance():
    if False:
        for i in range(10):
            print('nop')
    (X, y) = make_blobs(centers=2, random_state=0)
    X_1 = np.vstack([X] + [X[y == 1]] * 2)
    y_1 = np.hstack([y] + [y[y == 1]] * 2)
    X_0 = np.vstack([X] + [X[y == 0]] * 2)
    y_0 = np.hstack([y] + [y[y == 0]] * 2)
    X_ = np.vstack([X] * 2)
    y_ = np.hstack([y] * 2)
    logreg1 = LogisticRegression(class_weight='balanced').fit(X_1, y_1)
    logreg0 = LogisticRegression(class_weight='balanced').fit(X_0, y_0)
    logreg = LogisticRegression(class_weight='balanced').fit(X_, y_)
    assert_array_almost_equal(logreg1.coef_, logreg0.coef_)
    assert_array_almost_equal(logreg.coef_, logreg0.coef_)

def test_compute_class_weight_balanced_negative():
    if False:
        i = 10
        return i + 15
    classes = np.array([-2, -1, 0])
    y = np.asarray([-1, -1, 0, 0, -2, -2])
    cw = compute_class_weight('balanced', classes=classes, y=y)
    assert len(cw) == len(classes)
    assert_array_almost_equal(cw, np.array([1.0, 1.0, 1.0]))
    y = np.asarray([-1, 0, 0, -2, -2, -2])
    cw = compute_class_weight('balanced', classes=classes, y=y)
    assert len(cw) == len(classes)
    class_counts = np.bincount(y + 2)
    assert_almost_equal(np.dot(cw, class_counts), y.shape[0])
    assert_array_almost_equal(cw, [2.0 / 3, 2.0, 1.0])

def test_compute_class_weight_balanced_unordered():
    if False:
        for i in range(10):
            print('nop')
    classes = np.array([1, 0, 3])
    y = np.asarray([1, 0, 0, 3, 3, 3])
    cw = compute_class_weight('balanced', classes=classes, y=y)
    class_counts = np.bincount(y)[classes]
    assert_almost_equal(np.dot(cw, class_counts), y.shape[0])
    assert_array_almost_equal(cw, [2.0, 1.0, 2.0 / 3])

def test_compute_class_weight_default():
    if False:
        for i in range(10):
            print('nop')
    y = np.asarray([2, 2, 2, 3, 3, 4])
    classes = np.unique(y)
    classes_len = len(classes)
    cw = compute_class_weight(None, classes=classes, y=y)
    assert len(cw) == classes_len
    assert_array_almost_equal(cw, np.ones(3))
    cw = compute_class_weight({2: 1.5}, classes=classes, y=y)
    assert len(cw) == classes_len
    assert_array_almost_equal(cw, [1.5, 1.0, 1.0])
    cw = compute_class_weight({2: 1.5, 4: 0.5}, classes=classes, y=y)
    assert len(cw) == classes_len
    assert_array_almost_equal(cw, [1.5, 1.0, 0.5])

def test_compute_sample_weight():
    if False:
        while True:
            i = 10
    y = np.asarray([1, 1, 1, 2, 2, 2])
    sample_weight = compute_sample_weight('balanced', y)
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    sample_weight = compute_sample_weight({1: 2, 2: 1}, y)
    assert_array_almost_equal(sample_weight, [2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
    y = np.asarray([[1], [1], [1], [2], [2], [2]])
    sample_weight = compute_sample_weight('balanced', y)
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    y = np.asarray([1, 1, 1, 2, 2, 2, 3])
    sample_weight = compute_sample_weight('balanced', y)
    expected_balanced = np.array([0.7777, 0.7777, 0.7777, 0.7777, 0.7777, 0.7777, 2.3333])
    assert_array_almost_equal(sample_weight, expected_balanced, decimal=4)
    sample_weight = compute_sample_weight(None, y)
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    y = np.asarray([[1, 0], [1, 0], [1, 0], [2, 1], [2, 1], [2, 1]])
    sample_weight = compute_sample_weight('balanced', y)
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    y = np.asarray([[1, 0], [1, 0], [1, 0], [2, 1], [2, 1], [2, 1]])
    sample_weight = compute_sample_weight([{1: 2, 2: 1}, {0: 1, 1: 2}], y)
    assert_array_almost_equal(sample_weight, [2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    y = np.asarray([[1, 0], [1, 0], [1, 0], [2, 1], [2, 1], [2, 1], [3, -1]])
    sample_weight = compute_sample_weight('balanced', y)
    assert_array_almost_equal(sample_weight, expected_balanced ** 2, decimal=3)

def test_compute_sample_weight_with_subsample():
    if False:
        while True:
            i = 10
    y = np.asarray([1, 1, 1, 2, 2, 2])
    sample_weight = compute_sample_weight('balanced', y, indices=range(6))
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    y = np.asarray([[1], [1], [1], [2], [2], [2]])
    sample_weight = compute_sample_weight('balanced', y, indices=range(6))
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    y = np.asarray([1, 1, 1, 2, 2, 2])
    sample_weight = compute_sample_weight('balanced', y, indices=range(4))
    assert_array_almost_equal(sample_weight, [2.0 / 3, 2.0 / 3, 2.0 / 3, 2.0, 2.0, 2.0])
    y = np.asarray([1, 1, 1, 2, 2, 2])
    sample_weight = compute_sample_weight('balanced', y, indices=[0, 1, 1, 2, 2, 3])
    expected_balanced = np.asarray([0.6, 0.6, 0.6, 3.0, 3.0, 3.0])
    assert_array_almost_equal(sample_weight, expected_balanced)
    y = np.asarray([[1, 0], [1, 0], [1, 0], [2, 1], [2, 1], [2, 1]])
    sample_weight = compute_sample_weight('balanced', y, indices=[0, 1, 1, 2, 2, 3])
    assert_array_almost_equal(sample_weight, expected_balanced ** 2)
    y = np.asarray([1, 1, 1, 2, 2, 2, 3])
    sample_weight = compute_sample_weight('balanced', y, indices=range(6))
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    y = np.asarray([[1, 0], [1, 0], [1, 0], [2, 1], [2, 1], [2, 1], [2, 2]])
    sample_weight = compute_sample_weight('balanced', y, indices=range(6))
    assert_array_almost_equal(sample_weight, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])

@pytest.mark.parametrize('y_type, class_weight, indices, err_msg', [('single-output', {1: 2, 2: 1}, range(4), "The only valid class_weight for subsampling is 'balanced'."), ('multi-output', {1: 2, 2: 1}, None, 'For multi-output, class_weight should be a list of dicts, or the string'), ('multi-output', [{1: 2, 2: 1}], None, 'Got 1 element\\(s\\) while having 2 outputs')])
def test_compute_sample_weight_errors(y_type, class_weight, indices, err_msg):
    if False:
        i = 10
        return i + 15
    y_single_output = np.asarray([1, 1, 1, 2, 2, 2])
    y_multi_output = np.asarray([[1, 0], [1, 0], [1, 0], [2, 1], [2, 1], [2, 1]])
    y = y_single_output if y_type == 'single-output' else y_multi_output
    with pytest.raises(ValueError, match=err_msg):
        compute_sample_weight(class_weight, y, indices=indices)

def test_compute_sample_weight_more_than_32():
    if False:
        return 10
    y = np.arange(50)
    indices = np.arange(50)
    weight = compute_sample_weight('balanced', y, indices=indices)
    assert_array_almost_equal(weight, np.ones(y.shape[0]))

def test_class_weight_does_not_contains_more_classes():
    if False:
        while True:
            i = 10
    'Check that class_weight can contain more labels than in y.\n\n    Non-regression test for #22413\n    '
    tree = DecisionTreeClassifier(class_weight={0: 1, 1: 10, 2: 20})
    tree.fit([[0, 0, 1], [1, 0, 1], [1, 2, 0]], [0, 0, 1])

@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_compute_sample_weight_sparse(csc_container):
    if False:
        print('Hello World!')
    'Check that we can compute weight for sparse `y`.'
    y = csc_container(np.asarray([0, 1, 1])).T
    sample_weight = compute_sample_weight('balanced', y)
    assert_allclose(sample_weight, [1.5, 0.75, 0.75])