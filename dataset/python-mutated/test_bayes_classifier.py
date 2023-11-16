import os
import numpy
import torch
import pytest
from pomegranate.bayes_classifier import BayesClassifier
from pomegranate.distributions import Exponential
from .distributions._utils import _test_initialization_raises_one_parameter
from .distributions._utils import _test_initialization
from .distributions._utils import _test_predictions
from .distributions._utils import _test_efd_from_summaries
from .distributions._utils import _test_raises
from nose.tools import assert_raises
from numpy.testing import assert_array_almost_equal
MIN_VALUE = 0
MAX_VALUE = None
VALID_VALUE = 1.2

@pytest.fixture
def X():
    if False:
        for i in range(10):
            print('nop')
    return [[1, 2, 0], [0, 0, 1], [1, 1, 2], [2, 2, 2], [3, 1, 0], [5, 1, 4], [2, 1, 0], [1, 0, 2], [1, 1, 0], [0, 2, 1], [0, 0, 0]]

@pytest.fixture
def X_masked(X):
    if False:
        return 10
    mask = torch.tensor(numpy.array([[False, True, True], [True, True, False], [False, False, False], [True, True, True], [False, True, False], [True, True, True], [False, False, False], [True, False, True], [True, True, True], [True, True, True], [True, False, True]]))
    X = torch.tensor(numpy.array(X))
    return torch.masked.MaskedTensor(X, mask=mask)

@pytest.fixture
def w():
    if False:
        for i in range(10):
            print('nop')
    return [[1], [2], [0], [0], [5], [1], [2], [1], [1], [2], [0]]

@pytest.fixture
def y():
    if False:
        print('Hello World!')
    return [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1]

@pytest.fixture
def model():
    if False:
        for i in range(10):
            print('nop')
    d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
    return BayesClassifier(d, priors=[0.7, 0.3])

def test_initialization():
    if False:
        while True:
            i = 10
    d = [Exponential(), Exponential()]
    model = BayesClassifier(d)
    _test_initialization(model, None, 'priors', 0.0, False, None)
    assert_raises(AttributeError, getattr, model, '_w_sum')
    assert_raises(AttributeError, getattr, model, '_log_priors')

def test_initialization_raises():
    if False:
        print('Hello World!')
    d = [Exponential(), Exponential()]
    assert_raises(TypeError, BayesClassifier)
    assert_raises(ValueError, BayesClassifier, d, [0.2, 0.2, 0.6])
    assert_raises(ValueError, BayesClassifier, d, [0.2, 1.0])
    assert_raises(ValueError, BayesClassifier, d, [-0.2, 1.2])
    assert_raises(ValueError, BayesClassifier, Exponential)
    assert_raises(ValueError, BayesClassifier, d, inertia=-0.4)
    assert_raises(ValueError, BayesClassifier, d, inertia=1.2)
    assert_raises(ValueError, BayesClassifier, d, inertia=1.2, frozen='true')
    assert_raises(ValueError, BayesClassifier, d, inertia=1.2, frozen=3)

def test_reset_cache(X, y):
    if False:
        for i in range(10):
            print('nop')
    d = [Exponential(), Exponential()]
    model = BayesClassifier(d)
    model.summarize(X, y)
    assert_array_almost_equal(model._w_sum, [6.0, 5.0])
    assert_array_almost_equal(model._log_priors, [-0.693147, -0.693147])
    model._reset_cache()
    assert_array_almost_equal(model._w_sum, [0.0, 0.0])
    assert_array_almost_equal(model._log_priors, [-0.693147, -0.693147])

def test_initialize(X):
    if False:
        while True:
            i = 10
    d = [Exponential(), Exponential()]
    model = BayesClassifier(d)
    assert model.d is None
    assert model.k == 2
    assert model._initialized == False
    assert_raises(AttributeError, getattr, model, '_w_sum')
    assert_raises(AttributeError, getattr, model, '_log_priors')
    model._initialize(3)
    assert model._initialized == True
    assert model.priors.shape[0] == 2
    assert model.d == 3
    assert model.k == 2
    assert_array_almost_equal(model.priors, [0.5, 0.5])
    assert_array_almost_equal(model._w_sum, [0.0, 0.0])
    model._initialize(2)
    assert model._initialized == True
    assert model.priors.shape[0] == 2
    assert model.d == 2
    assert model.k == 2
    assert_array_almost_equal(model.priors, [0.5, 0.5])
    assert_array_almost_equal(model._w_sum, [0.0, 0.0])
    d = [Exponential([0.4, 2.1]), Exponential([3, 1]), Exponential([0.2, 1])]
    model = BayesClassifier(d)
    assert model._initialized == True
    assert model.d == 2
    assert model.k == 3
    model._initialize(3)
    assert model._initialized == True
    assert model.priors.shape[0] == 3
    assert model.d == 3
    assert model.k == 3
    assert_array_almost_equal(model.priors, [1.0 / 3, 1.0 / 3, 1.0 / 3])
    assert_array_almost_equal(model._w_sum, [0.0, 0.0, 0.0])

def test_emission_matrix(model, X):
    if False:
        for i in range(10):
            print('nop')
    e = model._emission_matrix(X)
    assert_array_almost_equal(e, [[-4.7349, -4.8411], [-7.5921, -3.9838], [-21.4016, -5.4276], [-25.2111, -6.4169], [-2.354, -5.8519], [-43.3063, -9.0034], [-1.8778, -5.1852], [-18.0682, -5.1051], [-1.4016, -4.5185], [-14.2587, -4.629], [2.4079, -3.5293]], 4)
    assert_array_almost_equal(e[:, 0], model.distributions[0].log_probability(X) - 0.3567, 4)
    assert_array_almost_equal(e[:, 1], model.distributions[1].log_probability(X) - 1.204, 4)

def test_emission_matrix_raises(model, X):
    if False:
        return 10
    _test_raises(model, '_emission_matrix', X, min_value=MIN_VALUE)
    d = [Exponential(), Exponential()]
    model = BayesClassifier(d)
    _test_raises(model, '_emission_matrix', X, min_value=MIN_VALUE)

def test_log_probability(model, X):
    if False:
        return 10
    logp = model.log_probability(X)
    assert_array_almost_equal(logp, [-4.0935, -3.9571, -5.4276, -6.4169, -2.3241, -9.0034, -1.8418, -5.1051, -1.3582, -4.6289, 2.4106], 4)

def test_log_probability_raises(model, X):
    if False:
        print('Hello World!')
    _test_raises(model, 'log_probability', X, min_value=MIN_VALUE)
    d = [Exponential(), Exponential()]
    model = BayesClassifier(d)
    _test_raises(model, 'log_probability', X, min_value=MIN_VALUE)

def test_predict(model, X):
    if False:
        while True:
            i = 10
    y_hat = model.predict(X)
    assert_array_almost_equal(y_hat, [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0], 4)

def test_predict_raises(model, X):
    if False:
        i = 10
        return i + 15
    _test_raises(model, 'predict', X, min_value=MIN_VALUE)
    d = [Exponential(), Exponential()]
    model = BayesClassifier(d)
    _test_raises(model, 'predict', X, min_value=MIN_VALUE)

def test_predict_proba(model, X):
    if False:
        return 10
    y_hat = model.predict_proba(X)
    assert_array_almost_equal(y_hat, [[0.52653, 0.47347], [0.026385, 0.97361], [1.1551e-07, 1.0], [6.883e-09, 1.0], [0.97063, 0.029372], [1.266e-15, 1.0], [0.96468, 0.035317], [2.3451e-06, 1.0], [0.95759, 0.042413], [6.5741e-05, 0.99993], [0.99737, 0.0026323]], 4)
    model2 = BayesClassifier(model.distributions)
    y_hat2 = model2.predict_proba(X)
    assert_array_almost_equal(y_hat2, [[0.32277, 0.67723], [0.011481, 0.98852], [4.9503e-08, 1.0], [2.9498e-09, 1.0], [0.93405, 0.065951], [5.4255e-16, 1.0], [0.9213, 0.0787], [1.005e-06, 1.0], [0.90633, 0.093666], [2.8176e-05, 0.99997], [0.99388, 0.0061207]], 4)

def test_predict_proba_raises(model, X):
    if False:
        i = 10
        return i + 15
    _test_raises(model, 'predict_proba', X, min_value=MIN_VALUE)
    d = [Exponential(), Exponential()]
    model = BayesClassifier(d)
    _test_raises(model, 'predict_proba', X, min_value=MIN_VALUE)

def test_predict_log_proba(model, X):
    if False:
        print('Hello World!')
    y_hat = model.predict_log_proba(X)
    assert_array_almost_equal(y_hat, [[-0.64145, -0.74766], [-3.635, -0.02674], [-15.974, 0.0], [-18.794, 0.0], [-0.029812, -3.5277], [-34.303, 0.0], [-0.035955, -3.3434], [-12.963, -2.3842e-06], [-0.043338, -3.1603], [-9.6298, -6.5804e-05], [-0.0026357, -5.9399]], 3)
    model2 = BayesClassifier(model.distributions)
    y_hat2 = model2.predict_log_proba(X)
    assert_array_almost_equal(y_hat2, [[-1.1308, -0.38974], [-4.4671, -0.011548], [-16.821, 0.0], [-19.642, 0.0], [-0.068226, -2.7188], [-35.15, 0.0], [-0.081969, -2.5421], [-13.81, -9.5367e-07], [-0.098348, -2.368], [-10.477, -2.8133e-05], [-0.0061395, -5.0961]], 3)

def test_predict_log_proba_raises(model, X):
    if False:
        i = 10
        return i + 15
    _test_raises(model, 'predict_log_proba', X, min_value=MIN_VALUE)
    d = [Exponential(), Exponential()]
    model = BayesClassifier(d)
    _test_raises(model, 'predict_log_proba', X, min_value=MIN_VALUE)

def test_partial_summarize(model, X, y):
    if False:
        i = 10
        return i + 15
    model.summarize(X[:4], y[:4])
    assert_array_almost_equal(model._w_sum, [2.0, 2.0])
    model.summarize(X[4:], y[4:])
    assert_array_almost_equal(model._w_sum, [6.0, 5.0])
    model = BayesClassifier([Exponential(), Exponential()])
    model.summarize(X[:4], y[:4])
    assert_array_almost_equal(model._w_sum, [2.0, 2.0])
    model.summarize(X[4:], y[4:])
    assert_array_almost_equal(model._w_sum, [6.0, 5.0])

def test_full_summarize(model, X, y):
    if False:
        print('Hello World!')
    model.summarize(X, y)
    assert_array_almost_equal(model._w_sum, [6.0, 5.0])
    model = BayesClassifier([Exponential(), Exponential()])
    model.summarize(X, y)
    assert_array_almost_equal(model._w_sum, [6.0, 5.0])

def test_summarize_weighted(model, X, y, w):
    if False:
        i = 10
        return i + 15
    model.summarize(X, y, sample_weight=w)
    assert_array_almost_equal(model._w_sum, [7.0, 8.0])
    model = BayesClassifier([Exponential(), Exponential()])
    model.summarize(X, y, sample_weight=w)
    assert_array_almost_equal(model._w_sum, [7.0, 8.0])

def test_summarize_weighted_flat(model, X, y, w):
    if False:
        print('Hello World!')
    w = numpy.array(w)[:, 0]
    model.summarize(X, y, sample_weight=w)
    assert_array_almost_equal(model._w_sum, [7.0, 8.0])
    model = BayesClassifier([Exponential(), Exponential()])
    model.summarize(X, y, sample_weight=w)
    assert_array_almost_equal(model._w_sum, [7.0, 8.0])

def test_summarize_weighted_2d(model, X, y):
    if False:
        while True:
            i = 10
    model.summarize(X, y, sample_weight=X)
    assert_array_almost_equal(model._w_sum, [7.666667, 5.333333])
    model = BayesClassifier([Exponential(), Exponential()])
    model.summarize(X, y, sample_weight=X)
    assert_array_almost_equal(model._w_sum, [7.666667, 5.333333])

def test_summarize_raises(model, X, y, w):
    if False:
        print('Hello World!')
    assert_raises(ValueError, model.summarize, [X], y)
    assert_raises(ValueError, model.summarize, X[0], y)
    assert_raises((ValueError, TypeError), model.summarize, X[0][0], y)
    assert_raises(ValueError, model.summarize, [x[:-1] for x in X], y)
    assert_raises(ValueError, model.summarize, [[-0.1 for i in range(3)] for x in X], y)
    assert_raises(ValueError, model.summarize, [X], y, w)
    assert_raises(ValueError, model.summarize, X, [y], w)
    assert_raises(ValueError, model.summarize, X, y, [w])
    assert_raises(ValueError, model.summarize, [X], y, [w])
    assert_raises(ValueError, model.summarize, X[:len(X) - 1], y, w)
    assert_raises(ValueError, model.summarize, X, y[:len(y) - 1], w)
    assert_raises(ValueError, model.summarize, X, y, w[:len(w) - 1])

def test_from_summaries(model, X, y):
    if False:
        while True:
            i = 10
    model.summarize(X, y)
    model.from_summaries()
    assert_array_almost_equal(model._w_sum, [0.0, 0.0])
    assert_array_almost_equal(model.priors, [6.0 / 11, 5.0 / 11])
    assert_array_almost_equal(model._log_priors, numpy.log([6.0 / 11, 5.0 / 11]))
    X_ = numpy.array(X)[numpy.array(y) == 0]
    d = Exponential().fit(X_)
    assert_array_almost_equal(d.scales, model.distributions[0].scales)
    model = BayesClassifier([Exponential(), Exponential()])
    model.summarize(X, y)
    model.from_summaries()
    assert_array_almost_equal(model._w_sum, [0.0, 0.0])
    assert_array_almost_equal(model.priors, [6.0 / 11, 5.0 / 11])
    assert_array_almost_equal(model._log_priors, numpy.log([6.0 / 11, 5.0 / 11]))
    assert_array_almost_equal(d.scales, model.distributions[0].scales)

def test_from_summaries_weighted(model, X, y, w):
    if False:
        for i in range(10):
            print('nop')
    model.summarize(X, y, sample_weight=w)
    model.from_summaries()
    assert_array_almost_equal(model._w_sum, [0.0, 0.0])
    assert_array_almost_equal(model.priors, [0.466667, 0.533333])
    assert_array_almost_equal(model._log_priors, numpy.log([0.466667, 0.533333]))
    idxs = numpy.array(y) == 0
    X_ = numpy.array(X)[idxs]
    w_ = numpy.array(w)[idxs]
    d = Exponential().fit(X_, sample_weight=w_)
    assert_array_almost_equal(d.scales, model.distributions[0].scales)
    model = BayesClassifier([Exponential(), Exponential()])
    model.summarize(X, y, sample_weight=w)
    model.from_summaries()
    assert_array_almost_equal(model._w_sum, [0.0, 0.0])
    assert_array_almost_equal(model.priors, [0.466667, 0.533333])
    assert_array_almost_equal(model._log_priors, numpy.log([0.466667, 0.533333]))
    assert_array_almost_equal(d.scales, model.distributions[0].scales)

def test_from_summaries_null(model):
    if False:
        print('Hello World!')
    model.from_summaries()
    assert model.distributions[0].scales[0] != 2.1
    assert model.distributions[1].scales[0] != 1.5
    assert_array_almost_equal(model._w_sum, [0.0, 0.0])

def test_from_summaries_inertia(X, y):
    if False:
        return 10
    d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
    model = BayesClassifier(d, priors=[0.2, 0.8], inertia=0.3)
    model.summarize(X, y)
    model.from_summaries()
    assert_array_almost_equal(model._w_sum, [0.0, 0.0])
    assert_array_almost_equal(model.priors, [0.441818, 0.558182])
    assert_array_almost_equal(model._log_priors, numpy.log([0.441818, 0.558182]))
    d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
    model = BayesClassifier(d, priors=[0.2, 0.8], inertia=1.0)
    model.summarize(X, y)
    model.from_summaries()
    assert_array_almost_equal(model._w_sum, [0.0, 0.0])
    assert_array_almost_equal(model.priors, [0.2, 0.8])
    assert_array_almost_equal(model._log_priors, numpy.log([0.2, 0.8]))
    (s1, s2) = ([2.1, 0.3, 0.1], [1.5, 3.1, 2.2])
    d = [Exponential(s1, inertia=1.0), Exponential(s2, inertia=1.0)]
    model = BayesClassifier(d, priors=[0.2, 0.8])
    model.summarize(X, y)
    model.from_summaries()
    assert_array_almost_equal(model.distributions[0].scales, s1)
    assert_array_almost_equal(model.distributions[1].scales, s2)
    assert_array_almost_equal(model._w_sum, [0.0, 0.0])
    assert_array_almost_equal(model.priors, [0.545455, 0.454545])
    assert_array_almost_equal(model._log_priors, numpy.log([0.545455, 0.454545]))
    d = [Exponential(s1, inertia=1.0), Exponential(s2)]
    model = BayesClassifier(d, priors=[0.2, 0.8], inertia=1.0)
    model.summarize(X, y)
    model.from_summaries()
    assert_array_almost_equal(model.distributions[0].scales, s1)
    assert_array_almost_equal(model.distributions[1].scales, [1.2, 1.4, 0.6])
    assert_array_almost_equal(model._w_sum, [0.0, 0.0])
    assert_array_almost_equal(model.priors, [0.2, 0.8])
    assert_array_almost_equal(model._log_priors, numpy.log([0.2, 0.8]))

def test_from_summaries_weighted_inertia(X, y, w):
    if False:
        for i in range(10):
            print('nop')
    d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
    model = BayesClassifier(d, priors=[0.2, 0.8], inertia=0.3)
    model.summarize(X, y, sample_weight=w)
    model.from_summaries()
    assert_array_almost_equal(model._w_sum, [0.0, 0.0])
    assert_array_almost_equal(model.priors, [0.386667, 0.613333])
    assert_array_almost_equal(model._log_priors, numpy.log([0.386667, 0.613333]))
    d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
    model = BayesClassifier(d, priors=[0.2, 0.8], inertia=1.0)
    model.summarize(X, y, sample_weight=w)
    model.from_summaries()
    assert_array_almost_equal(model._w_sum, [0.0, 0.0])
    assert_array_almost_equal(model.priors, [0.2, 0.8])
    assert_array_almost_equal(model._log_priors, numpy.log([0.2, 0.8]))

def test_from_summaries_frozen(model, X, y):
    if False:
        print('Hello World!')
    d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
    model = BayesClassifier(d, priors=[0.2, 0.8], frozen=True)
    model.summarize(X, y)
    model.from_summaries()
    assert_array_almost_equal(model._w_sum, [0.0, 0.0])
    assert_array_almost_equal(model.priors, [0.2, 0.8])
    assert_array_almost_equal(model._log_priors, numpy.log([0.2, 0.8]))
    (s1, s2) = ([2.1, 0.3, 0.1], [1.5, 3.1, 2.2])
    d = [Exponential(s1, frozen=True), Exponential(s2, frozen=True)]
    model = BayesClassifier(d, priors=[0.2, 0.8])
    model.summarize(X, y)
    model.from_summaries()
    assert_array_almost_equal(model.distributions[0].scales, s1)
    assert_array_almost_equal(model.distributions[1].scales, s2)
    assert_array_almost_equal(model._w_sum, [0.0, 0.0])
    assert_array_almost_equal(model.priors, [0.545455, 0.454545])
    assert_array_almost_equal(model._log_priors, numpy.log([0.545455, 0.454545]))
    d = [Exponential(s1, frozen=True), Exponential(s2)]
    model = BayesClassifier(d, priors=[0.2, 0.8], frozen=True)
    model.summarize(X, y)
    model.from_summaries()
    assert_array_almost_equal(model.distributions[0].scales, s1)
    assert_array_almost_equal(model.distributions[1].scales, [1.2, 1.4, 0.6])
    assert_array_almost_equal(model._w_sum, [0.0, 0.0])
    assert_array_almost_equal(model.priors, [0.2, 0.8])
    assert_array_almost_equal(model._log_priors, numpy.log([0.2, 0.8]))

def test_fit(model, X, y):
    if False:
        while True:
            i = 10
    model.fit(X, y)
    assert_array_almost_equal(model._w_sum, [0.0, 0.0])
    assert_array_almost_equal(model.priors, [6.0 / 11, 5.0 / 11])
    assert_array_almost_equal(model._log_priors, numpy.log([6.0 / 11, 5.0 / 11]))
    X_ = numpy.array(X)[numpy.array(y) == 0]
    d = Exponential().fit(X_)
    assert_array_almost_equal(d.scales, model.distributions[0].scales)
    model = BayesClassifier([Exponential(), Exponential()])
    model.fit(X, y)
    assert_array_almost_equal(model._w_sum, [0.0, 0.0])
    assert_array_almost_equal(model.priors, [6.0 / 11, 5.0 / 11])
    assert_array_almost_equal(model._log_priors, numpy.log([6.0 / 11, 5.0 / 11]))
    assert_array_almost_equal(d.scales, model.distributions[0].scales)

def test_fit_weighted(model, X, y, w):
    if False:
        print('Hello World!')
    model.fit(X, y, sample_weight=w)
    assert_array_almost_equal(model._w_sum, [0.0, 0.0])
    assert_array_almost_equal(model.priors, [0.466667, 0.533333])
    assert_array_almost_equal(model._log_priors, numpy.log([0.466667, 0.533333]))
    idxs = numpy.array(y) == 0
    X_ = numpy.array(X)[idxs]
    w_ = numpy.array(w)[idxs]
    d = Exponential().fit(X_, sample_weight=w_)
    assert_array_almost_equal(d.scales, model.distributions[0].scales)
    model = BayesClassifier([Exponential(), Exponential()])
    model.fit(X, y, sample_weight=w)
    assert_array_almost_equal(model._w_sum, [0.0, 0.0])
    assert_array_almost_equal(model.priors, [0.466667, 0.533333])
    assert_array_almost_equal(model._log_priors, numpy.log([0.466667, 0.533333]))
    assert_array_almost_equal(d.scales, model.distributions[0].scales)

def test_fit_chain(X, y):
    if False:
        i = 10
        return i + 15
    model = BayesClassifier([Exponential(), Exponential()]).fit(X, y)
    assert_array_almost_equal(model._w_sum, [0.0, 0.0])
    assert_array_almost_equal(model.priors, [6.0 / 11, 5.0 / 11])
    assert_array_almost_equal(model._log_priors, numpy.log([6.0 / 11, 5.0 / 11]))

def test_fit_raises(model, X, w, y):
    if False:
        i = 10
        return i + 15
    assert_raises(ValueError, model.fit, [X], y)
    assert_raises(ValueError, model.fit, X[0], y)
    assert_raises((ValueError, TypeError), model.fit, X[0][0], y)
    assert_raises(ValueError, model.fit, [x[:-1] for x in X], y)
    assert_raises(ValueError, model.fit, [[-0.1 for i in range(3)] for x in X], y)
    assert_raises(ValueError, model.fit, [X], y, w)
    assert_raises(ValueError, model.fit, X, [y], w)
    assert_raises(ValueError, model.fit, X, y, [w])
    assert_raises(ValueError, model.fit, [X], y, [w])
    assert_raises(ValueError, model.fit, X[:len(X) - 1], y, w)
    assert_raises(ValueError, model.fit, X, y[:len(y) - 1], w)
    assert_raises(ValueError, model.fit, X, y, w[:len(w) - 1])

def test_serialization(X, model):
    if False:
        while True:
            i = 10
    torch.save(model, '.pytest.torch')
    model2 = torch.load('.pytest.torch')
    os.system('rm .pytest.torch')
    assert_array_almost_equal(model2.priors, model.priors)
    assert_array_almost_equal(model2._log_priors, model._log_priors)
    assert_array_almost_equal(model2.predict_proba(X), model.predict_proba(X))
    (m1d1, m1d2) = model.distributions
    (m2d1, m2d2) = model2.distributions
    assert m1d1 is not m2d1
    assert m1d2 is not m2d2
    assert_array_almost_equal(m1d1.scales, m2d1.scales)
    assert_array_almost_equal(m1d2.scales, m2d2.scales)

def test_masked_probability(model, X, X_masked):
    if False:
        i = 10
        return i + 15
    X = torch.tensor(numpy.array(X))
    y = [0.01668138, 0.01911842, 0.004393471, 0.001633741, 0.09786682, 0.0001229918, 0.1585297, 0.006066021, 0.257113, 0.009765117, 11.14044]
    mask = torch.ones_like(X).type(torch.bool)
    X_ = torch.masked.MaskedTensor(X, mask=mask)
    assert_array_almost_equal(y, model.probability(X_), 5)
    y = [0.05277007, 1.175627, 1.0, 0.001633741, 0.1533307, 0.0001229918, 1.0, 0.01880462, 0.257113, 0.009765117, 3.424242]
    assert_array_almost_equal(y, model.probability(X_masked), 5)

def test_masked_log_probability(model, X, X_masked):
    if False:
        while True:
            i = 10
    X = torch.tensor(numpy.array(X))
    y = [-4.09346, -3.9571, -5.42764, -6.41688, -2.32415, -9.00339, -1.84181, -5.10505, -1.35824, -4.62894, 2.41058]
    mask = torch.ones_like(X).type(torch.bool)
    X_ = torch.masked.MaskedTensor(X, mask=mask)
    assert_array_almost_equal(y, model.log_probability(X_), 5)
    y = [-2.94181, 0.1618, 0.0, -6.41688, -1.87516, -9.00339, 0.0, -3.97365, -1.35824, -4.62894, 1.23088]
    assert_array_almost_equal(y, model.log_probability(X_masked), 5)

def test_masked_emission_matrix(model, X, X_masked):
    if False:
        while True:
            i = 10
    X = torch.tensor(numpy.array(X))
    mask = torch.ones_like(X).type(torch.bool)
    X_ = torch.masked.MaskedTensor(X, mask=mask)
    e = model._emission_matrix(X_)
    assert_array_almost_equal(e, [[-4.7349, -4.8411], [-7.5921, -3.9838], [-21.4016, -5.4276], [-25.2111, -6.4169], [-2.354, -5.8519], [-43.3063, -9.0034], [-1.8778, -5.1852], [-18.0682, -5.1051], [-1.4016, -4.5185], [-14.2587, -4.629], [2.4079, -3.5293]], 4)
    d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
    model = BayesClassifier(d)
    e = model._emission_matrix(X_masked)
    assert_array_almost_equal(e, [[-3.8533, -3.2582], [-0.2311, -2.23], [-0.6931, -0.6931], [-25.5476, -5.9061], [-2.8225, -2.1471], [-43.6428, -8.4926], [-0.6931, -0.6931], [-19.6087, -3.4628], [-1.7381, -4.0077], [-14.5952, -4.1182], [0.8675, -1.8871]], 4)

def test_masked_summarize(model, X, X_masked, w, y):
    if False:
        print('Hello World!')
    X = torch.tensor(numpy.array(X))
    mask = torch.ones_like(X).type(torch.bool)
    X_ = torch.masked.MaskedTensor(X, mask=mask)
    d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
    model = BayesClassifier(d, priors=[0.7, 0.3])
    model.summarize(X_, y, sample_weight=w)
    assert_array_almost_equal(model._w_sum, [7.0, 8.0])
    d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
    model = BayesClassifier(d, priors=[0.7, 0.3])
    model.summarize(X_masked, y, sample_weight=w)
    assert_array_almost_equal(model._w_sum, [5.0, 8.0])

def test_masked_from_summaries(model, X, X_masked, y):
    if False:
        while True:
            i = 10
    X = torch.tensor(numpy.array(X))
    mask = torch.ones_like(X).type(torch.bool)
    X_ = torch.masked.MaskedTensor(X, mask=mask)
    model.summarize(X_, y)
    model.from_summaries()
    assert_array_almost_equal(model._w_sum, [0.0, 0.0])
    assert_array_almost_equal(model.priors, [0.545455, 0.454545])
    assert_array_almost_equal(model._log_priors, numpy.log([0.545455, 0.454545]))
    d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
    model = BayesClassifier(d, priors=[0.7, 0.3])
    model.summarize(X_masked, y, sample_weight=w)
    model.from_summaries()
    assert_array_almost_equal(model._w_sum, [0.0, 0.0])
    assert_array_almost_equal(model.priors, [0.444444, 0.555556])
    assert_array_almost_equal(model._log_priors, numpy.log([0.444444, 0.555556]))

def test_masked_fit(X, X_masked, y):
    if False:
        print('Hello World!')
    X = torch.tensor(numpy.array(X))
    mask = torch.ones_like(X).type(torch.bool)
    X_ = torch.masked.MaskedTensor(X, mask=mask)
    d = [Exponential([2.1, 0.3, 1.1]), Exponential([1.5, 3.1, 2.2])]
    model = BayesClassifier(d)
    model.fit(X_, y)
    assert_array_almost_equal(model._w_sum, [0.0, 0.0])
    assert_array_almost_equal(model.priors, [0.545455, 0.454545])
    assert_array_almost_equal(model._log_priors, numpy.log([0.545455, 0.454545]))
    d = [Exponential([2.1, 0.3, 0.1]), Exponential([1.5, 3.1, 2.2])]
    model = BayesClassifier(d)
    model.fit(X_masked, y)
    assert_array_almost_equal(model._w_sum, [0.0, 0.0])
    assert_array_almost_equal(model.priors, [0.444444, 0.555556])
    assert_array_almost_equal(model._log_priors, numpy.log([0.444444, 0.555556]))