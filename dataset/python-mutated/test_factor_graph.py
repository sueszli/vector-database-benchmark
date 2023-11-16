import numpy
import torch
import pytest
from pomegranate.factor_graph import FactorGraph
from pomegranate.distributions import Exponential
from pomegranate.distributions import Categorical
from pomegranate.distributions import JointCategorical
from nose.tools import assert_raises
from numpy.testing import assert_array_almost_equal

@pytest.fixture
def X():
    if False:
        while True:
            i = 10
    return [[1, 2, 0, 0], [0, 0, 1, 0], [1, 1, 1, 0], [1, 2, 1, 1], [1, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 2, 1, 0], [0, 0, 0, 1]]

@pytest.fixture
def w():
    if False:
        print('Hello World!')
    return [0, 1.3, 0, 1.2, 2.5, 6.1, 0, 0, 2.1, 0.3, 3.3]

@pytest.fixture
def X_masked(X):
    if False:
        while True:
            i = 10
    mask = torch.tensor(numpy.array([[False, True, True, True], [True, True, False, False], [False, False, False, True], [True, True, True, True], [False, True, False, False], [True, True, True, False], [False, False, False, True], [True, False, True, True], [True, True, True, False], [True, True, True, True], [True, False, True, False]]))
    X = torch.tensor(numpy.array(X))
    return torch.masked.MaskedTensor(X, mask=mask)

@pytest.fixture
def model():
    if False:
        i = 10
        return i + 15
    f1 = Categorical([[0.23, 0.77]])
    f2 = JointCategorical([[[0.1, 0.05, 0.05], [0.15, 0.05, 0.04]], [[0.2, 0.1, 0.05], [0.05, 0.1, 0.06]]])
    f3 = Categorical([[0.61, 0.39]])
    f4 = JointCategorical([[0.17, 0.15], [0.4, 0.28]])
    m1 = Categorical([[0.5, 0.5]])
    m2 = Categorical([[1.0 / 3, 1.0 / 3, 1.0 / 3]])
    m3 = Categorical([[0.5, 0.5]])
    m4 = Categorical([[0.5, 0.5]])
    model = FactorGraph()
    model.add_factor(f1)
    model.add_factor(f2)
    model.add_factor(f3)
    model.add_factor(f4)
    model.add_marginal(m1)
    model.add_marginal(m2)
    model.add_marginal(m3)
    model.add_marginal(m4)
    model.add_edge(m1, f1)
    model.add_edge(m1, f2)
    model.add_edge(m3, f2)
    model.add_edge(m2, f2)
    model.add_edge(m3, f3)
    model.add_edge(m3, f4)
    model.add_edge(m4, f4)
    return model

@pytest.fixture
def model2():
    if False:
        while True:
            i = 10
    f1 = Categorical([[0.23, 0.77]])
    f3 = JointCategorical([[0.17, 0.15], [0.4, 0.28]])
    f4 = JointCategorical([[0.32, 0.12], [0.08, 0.48]])
    f2 = JointCategorical([[[0.1, 0.05, 0.05], [0.15, 0.05, 0.04]], [[0.2, 0.1, 0.05], [0.05, 0.1, 0.06]]])
    m1 = Categorical([[0.5, 0.5]])
    m2 = Categorical([[1.0 / 3, 1.0 / 3, 1.0 / 3]])
    m3 = Categorical([[0.5, 0.5]])
    m4 = Categorical([[0.5, 0.5]])
    model = FactorGraph()
    model.add_factor(f1)
    model.add_factor(f2)
    model.add_factor(f3)
    model.add_factor(f4)
    model.add_marginal(m1)
    model.add_marginal(m2)
    model.add_marginal(m3)
    model.add_marginal(m4)
    model.add_edge(m1, f1)
    model.add_edge(m1, f3)
    model.add_edge(m1, f4)
    model.add_edge(m3, f3)
    model.add_edge(m4, f4)
    model.add_edge(m3, f2)
    model.add_edge(m4, f2)
    model.add_edge(m2, f2)
    return model

def test_initialization():
    if False:
        return 10
    model = FactorGraph()
    assert len(model.factors) == 0
    assert len(model.marginals) == 0
    assert len(model._factor_idxs) == 0
    assert len(model._marginal_idxs) == 0
    m = Categorical([[0.5, 0.5]])
    f = Categorical([[0.8, 0.1]])
    model = FactorGraph([f], [m])
    assert len(model.factors) == 1
    assert len(model.marginals) == 1
    assert len(model._factor_idxs) == 1
    assert len(model._marginal_idxs) == 1
    assert sum(map(len, model._factor_edges)) == 0
    assert sum(map(len, model._marginal_edges)) == 0
    model = FactorGraph([f], [m], [(m, f)])
    assert len(model.factors) == 1
    assert len(model.marginals) == 1
    assert len(model._factor_idxs) == 1
    assert len(model._marginal_idxs) == 1
    assert sum(map(len, model._factor_edges)) == 1
    assert sum(map(len, model._marginal_edges)) == 1

def test_initialization_raises():
    if False:
        for i in range(10):
            print('nop')
    m1 = Categorical([[0.5, 0.5]])
    m2 = Categorical([[0.5, 0.5]])
    f1 = Categorical([[0.8, 0.1]])
    f2 = JointCategorical([[[0.2, 0.1], [0.1, 0.6]]])
    assert_raises(ValueError, FactorGraph, [m1, m2], [f1, f2])
    assert_raises(ValueError, FactorGraph, [f1, f2], [m1, m2], [(f1, f2)])
    assert_raises(ValueError, FactorGraph, [f1, f2], [m1, m2], [(m1, m2)])
    assert_raises(ValueError, FactorGraph, [f1, f2], [m1, m2], [(f1, m1)])
    assert_raises(ValueError, FactorGraph, None, None, [(f1, m1)])

def test_add_factor():
    if False:
        while True:
            i = 10
    f1 = Categorical([[0.8, 0.1]])
    f2 = JointCategorical([[[0.2, 0.1], [0.1, 0.6]]])
    model = FactorGraph()
    assert len(model.factors) == 0
    assert model._factor_idxs == {}
    assert model._factor_edges == []
    model.add_factor(f1)
    assert len(model.factors) == 1
    assert model._factor_idxs == {f1: 0}
    assert model._factor_edges == [[]]
    model.add_factor(f2)
    assert len(model.factors) == 2
    assert model._factor_idxs == {f1: 0, f2: 1}
    assert model._factor_edges == [[], []]

def test_add_factor_raises():
    if False:
        i = 10
        return i + 15
    f1 = Categorical([[0.8, 0.1]])
    f2 = JointCategorical([[[0.2, 0.1], [0.1, 0.6]]])
    model = FactorGraph()
    assert_raises(ValueError, model.add_factor, [f1])
    model = FactorGraph()
    assert_raises(ValueError, model.add_factor, None)
    model = FactorGraph()
    assert_raises(ValueError, model.add_factor, Exponential())

def test_add_marginal():
    if False:
        for i in range(10):
            print('nop')
    m1 = Categorical([[0.5, 0.5]])
    m2 = Categorical([[0.5, 0.5]])
    model = FactorGraph()
    assert len(model.marginals) == 0
    assert model._marginal_idxs == {}
    assert model._marginal_edges == []
    model.add_marginal(m1)
    assert len(model.marginals) == 1
    assert model._marginal_idxs == {m1: 0}
    assert model._marginal_edges == [[]]
    model.add_marginal(m2)
    assert len(model.marginals) == 2
    assert model._marginal_idxs == {m1: 0, m2: 1}
    assert model._marginal_edges == [[], []]

def test_add_marginal_raises():
    if False:
        while True:
            i = 10
    m = Categorical([[0.5, 0.5]])
    f = JointCategorical([[[0.2, 0.1], [0.1, 0.6]]])
    model = FactorGraph()
    assert_raises(ValueError, model.add_marginal, [m])
    model = FactorGraph()
    assert_raises(ValueError, model.add_marginal, None)
    model = FactorGraph()
    assert_raises(ValueError, model.add_marginal, Exponential())
    model = FactorGraph()
    assert_raises(ValueError, model.add_marginal, f)

def test_add_edge():
    if False:
        while True:
            i = 10
    m1 = Categorical([[0.5, 0.5]])
    m2 = Categorical([[0.5, 0.5]])
    f1 = Categorical([[0.8, 0.1]])
    f2 = JointCategorical([[[0.2, 0.1], [0.1, 0.6]]])
    model = FactorGraph([f1, f2], [m1, m2])
    assert model._factor_edges == [[], []]
    assert model._marginal_edges == [[], []]
    model.add_edge(m1, f1)
    assert model._factor_edges == [[0], []]
    assert model._marginal_edges == [[0], []]
    model.add_edge(m1, f2)
    assert model._factor_edges == [[0], [0]]
    assert model._marginal_edges == [[0, 1], []]
    model.add_edge(m2, f2)
    assert model._factor_edges == [[0], [0, 1]]
    assert model._marginal_edges == [[0, 1], [1]]

def test_add_edge_raises():
    if False:
        print('Hello World!')
    m1 = Categorical([[0.5, 0.5]])
    m2 = Categorical([[0.5, 0.5]])
    m3 = Categorical([[0.5, 0.5]])
    f1 = Categorical([[0.8, 0.1]])
    f2 = JointCategorical([[[0.2, 0.1], [0.1, 0.6]]])
    model = FactorGraph([f1, f2], [m1, m2])
    assert_raises(ValueError, model.add_edge, f2, f1)
    assert_raises(ValueError, model.add_edge, f2, m1)
    assert_raises(ValueError, model.add_edge, f1, m1)
    assert_raises(ValueError, model.add_edge, m1, m2)
    assert_raises(ValueError, model.add_edge, m3, f1)
    assert_raises(ValueError, model.add_edge, None, f1)
    assert_raises(ValueError, model.add_edge, m1, None)
    assert_raises(ValueError, model.add_edge, m1, m1)

def test_probability(X, model, model2):
    if False:
        i = 10
        return i + 15
    prob = model.probability(X)
    assert_array_almost_equal(prob, [0.000166352, 0.00022425, 0.0005004999, 0.00021021, 0.0002935625, 4.384369e-05, 4.968956e-05, 0.000175175, 0.0002935625, 5.979991e-05, 8.768743e-05])
    prob = model2.probability(X)
    assert_array_almost_equal(prob, [5.133332e-05, 9.199999e-05, 7.18666e-05, 0.0002587201, 0.0003079999, 9.774998e-06, 2.606666e-05, 0.0002156001, 0.0003079999, 2.299997e-05, 2.932498e-05])

def test_log_probability(X, model, model2):
    if False:
        i = 10
        return i + 15
    prob = model.log_probability(X)
    assert_array_almost_equal(prob, [-8.701405, -8.402749, -7.599903, -8.467403, -8.13342, -10.03488, -9.909716, -8.649725, -8.13342, -9.724506, -9.341732])
    prob = model2.log_probability(X)
    assert_array_almost_equal(prob, [-9.877171, -9.293722, -9.540699, -8.259764, -8.085411, -11.535683, -10.554853, -8.442085, -8.085411, -10.680017, -10.437071])

def test_predict(X_masked, model, model2):
    if False:
        i = 10
        return i + 15
    y_hat = model.predict(X_masked)
    assert_array_almost_equal(y_hat, [[1, 2, 0, 0], [0, 0, 1, 0], [1, 0, 1, 0], [1, 2, 1, 1], [1, 1, 1, 0], [0, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0], [0, 2, 1, 0], [0, 0, 0, 0]])
    y_hat = model2.predict(X_masked)
    assert_array_almost_equal(y_hat, [[1, 2, 0, 0], [0, 0, 1, 0], [1, 0, 1, 0], [1, 2, 1, 1], [1, 1, 1, 1], [0, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1], [1, 1, 0, 1], [0, 2, 1, 0], [0, 0, 0, 0]])

def test_predict_proba_one_edge():
    if False:
        print('Hello World!')
    m = Categorical([[1.0 / 3, 1.0 / 3, 1.0 / 3]])
    f = Categorical([[0.23, 0.17, 0.6]])
    X = torch.tensor([[0], [0], [1], [2]])
    mask = torch.tensor([[False], [True], [True], [True]])
    X_masked = torch.masked.MaskedTensor(X, mask=mask)
    model = FactorGraph([f], [m], [(m, f)])
    y_hat = model.predict_proba(X_masked)
    assert_array_almost_equal(y_hat[0], [[0.23, 0.17, 0.6], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

def test_predict_proba_one_edge_raises():
    if False:
        i = 10
        return i + 15
    m = Categorical([[1.0 / 3, 1.0 / 3, 1.0 / 3]])
    f = Categorical([[0.23, 0.17, 0.6]])
    model = FactorGraph([f], [m], [(m, f)])
    X = torch.tensor([[0, 0]])
    mask = torch.tensor([[False, True]])
    X_masked = torch.masked.MaskedTensor(X, mask=mask)
    assert_raises(ValueError, model.predict_proba, X_masked)
    X = torch.tensor([[3]])
    mask = torch.tensor([[True]])
    X_masked = torch.masked.MaskedTensor(X, mask=mask)
    assert_raises(IndexError, model.predict_proba, X_masked)
    mask = torch.tensor([[False]])
    X_masked = torch.masked.MaskedTensor(X, mask=mask)
    model.predict_proba(X_masked)

def test_predict_proba_one_edge_nonuniform():
    if False:
        for i in range(10):
            print('nop')
    m = Categorical([[0.4, 0.11, 0.49]])
    f = Categorical([[0.23, 0.17, 0.6]])
    X = torch.tensor([[0], [0]])
    mask = torch.tensor([[False], [True]])
    X_masked = torch.masked.MaskedTensor(X, mask=mask)
    model = FactorGraph([f], [m], [(m, f)])
    y_hat = model.predict_proba(X_masked)
    z = 0.23 * 0.4 + 0.17 * 0.11 + 0.6 * 0.49
    assert_array_almost_equal(y_hat[0], [[0.23 * 0.4 / z, 0.17 * 0.11 / z, 0.6 * 0.49 / z], [1.0, 0.0, 0.0]])

def test_predict_proba_monty_hall():
    if False:
        print('Hello World!')
    m1 = Categorical([[1.0 / 3, 1.0 / 3, 1.0 / 3]])
    m2 = Categorical([[1.0 / 3, 1.0 / 3, 1.0 / 3]])
    m3 = Categorical([[1.0 / 3, 1.0 / 3, 1.0 / 3]])
    p = numpy.array([[[0.0, 0.5, 0.5], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0], [0.5, 0.0, 0.5], [1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]])
    f1 = Categorical([[1.0 / 3, 1.0 / 3, 1.0 / 3]])
    f2 = Categorical([[1.0 / 3, 1.0 / 3, 1.0 / 3]])
    f3 = JointCategorical(p / 9.0)
    X = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0]])
    mask = torch.tensor([[False, False, False], [False, True, False], [False, True, False], [True, False, False], [True, False, False]])
    X_masked = torch.masked.MaskedTensor(X, mask=mask)
    model = FactorGraph([f1, f2, f3], [m1, m2, m3], [(m1, f1), (m1, f3), (m2, f3), (m3, f3), (m2, f2)])
    y_hat = model.predict_proba(X_masked)
    assert_array_almost_equal(y_hat[0], [[1.0 / 3, 1.0 / 3, 1.0 / 3], [1.0 / 3, 1.0 / 3, 1.0 / 3], [1.0 / 3, 1.0 / 3, 1.0 / 3], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert_array_almost_equal(y_hat[1], [[1.0 / 3, 1.0 / 3, 1.0 / 3], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0 / 3, 1.0 / 3, 1.0 / 3], [1.0 / 3, 1.0 / 3, 1.0 / 3]])
    assert_array_almost_equal(y_hat[2], [[1.0 / 3, 1.0 / 3, 1.0 / 3], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    X = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [0, 0, 1], [0, 1, 0], [2, 1, 0]])
    mask = torch.tensor([[True, True, False], [True, True, False], [True, True, False], [True, True, False], [True, True, False], [False, True, True], [False, False, True], [True, False, True]])
    X_masked = torch.masked.MaskedTensor(X, mask=mask)
    y_hat = model.predict_proba(X_masked)
    assert_array_almost_equal(y_hat[0], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0 / 3, 0.0, 2.0 / 3], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])
    assert_array_almost_equal(y_hat[1], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.0, 2.0 / 3, 1.0 / 3]])
    assert_array_almost_equal(y_hat[2], [[0.0, 0.5, 0.5], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

def test_predict_proba_simple(model, X_masked):
    if False:
        for i in range(10):
            print('nop')
    y_hat = model.predict_proba(X_masked)
    assert_array_almost_equal(y_hat[0], [[0.23, 0.77], [1.0, 0.0], [0.2011, 0.7989], [0.0, 1.0], [0.1299, 0.8701], [1.0, 0.0], [0.2011, 0.7989], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]], 4)
    assert_array_almost_equal(y_hat[1], [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.4469, 0.3453, 0.2078], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.4469, 0.3453, 0.2078], [0.2381, 0.4762, 0.2857], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.25, 0.25]], 4)
    assert_array_almost_equal(y_hat[2], [[1.0, 0.0], [0.3292, 0.6708], [0.4916, 0.5084], [0.0, 1.0], [0.424, 0.576], [1.0, 0.0], [0.4916, 0.5084], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], 4)
    assert_array_almost_equal(y_hat[3], [[1.0, 0.0], [0.5695, 0.4305], [1.0, 0.0], [0.0, 1.0], [0.5641, 0.4359], [0.5312, 0.4688], [1.0, 0.0], [0.0, 1.0], [0.5312, 0.4688], [1.0, 0.0], [0.5312, 0.4688]], 4)

def test_predict_proba_cycle(model2, X_masked):
    if False:
        return 10
    y_hat = model2.predict_proba(X_masked)
    assert_array_almost_equal(y_hat[0], [[0.3368, 0.6632], [1.0, 0.0], [0.3673, 0.6327], [0.0, 1.0], [0.1031, 0.8969], [1.0, 0.0], [0.3673, 0.6327], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]], 4)
    assert_array_almost_equal(y_hat[1], [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.5408, 0.2704, 0.1888], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.5408, 0.2704, 0.1888], [0.2381, 0.4762, 0.2857], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5388, 0.2371, 0.2241]], 4)
    assert_array_almost_equal(y_hat[2], [[1.0, 0.0], [0.4474, 0.5526], [0.4287, 0.5713], [0.0, 1.0], [0.411, 0.589], [1.0, 0.0], [0.4287, 0.5713], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], 4)
    assert_array_almost_equal(y_hat[3], [[1.0, 0.0], [0.7916, 0.2084], [1.0, 0.0], [0.0, 1.0], [0.2031, 0.7969], [0.7273, 0.2727], [1.0, 0.0], [0.0, 1.0], [0.1429, 0.8571], [1.0, 0.0], [0.6897, 0.3103]], 4)

def test_fit(X, model, model2):
    if False:
        return 10
    model.fit(X)
    assert_array_almost_equal(model.factors[0].probs, [[0.4545, 0.5455]], 4)
    assert_array_almost_equal(model.factors[1].probs, [[[0.0909, 0.1818, 0.0], [0.0909, 0.0, 0.0909]], [[0.0, 0.1818, 0.0909], [0.0909, 0.0909, 0.0909]]], 4)
    assert_array_almost_equal(model.factors[2].probs, [[0.5455, 0.4545]], 4)
    assert_array_almost_equal(model.factors[3].probs, [[0.1818, 0.3636], [0.2727, 0.1818]], 4)
    assert_array_almost_equal(model.marginals[0].probs, [[0.5, 0.5]])
    assert_array_almost_equal(model.marginals[1].probs, [[1.0 / 3, 1.0 / 3, 1.0 / 3]])
    assert_array_almost_equal(model.marginals[2].probs, [[0.5, 0.5]])
    assert_array_almost_equal(model.marginals[3].probs, [[0.5, 0.5]])
    model2.fit(X)
    assert_array_almost_equal(model2.factors[0].probs, [[0.4545, 0.5455]], 4)
    assert_array_almost_equal(model2.factors[1].probs, [[[0.0, 0.0909, 0.0909], [0.0909, 0.2727, 0.0]], [[0.0909, 0.0909, 0.0909], [0.0909, 0.0, 0.0909]]], 4)
    assert_array_almost_equal(model2.factors[2].probs, [[0.2727, 0.1818], [0.2727, 0.2727]], 4)
    assert_array_almost_equal(model2.factors[3].probs, [[0.2727, 0.1818], [0.1818, 0.3636]], 4)
    assert_array_almost_equal(model2.marginals[0].probs, [[0.5, 0.5]])
    assert_array_almost_equal(model2.marginals[1].probs, [[1.0 / 3, 1.0 / 3, 1.0 / 3]])
    assert_array_almost_equal(model2.marginals[2].probs, [[0.5, 0.5]])
    assert_array_almost_equal(model2.marginals[3].probs, [[0.5, 0.5]])

def test_fit_weighted(X, w, model, model2):
    if False:
        print('Hello World!')
    model.fit(X, sample_weight=w)
    assert_array_almost_equal(model.factors[0].probs, [[0.6548, 0.3452]], 4)
    assert_array_almost_equal(model.factors[1].probs, [[[0.1964, 0.3631, 0.0], [0.0774, 0.0, 0.0179]], [[0.0, 0.2738, 0.0], [0.0, 0.0, 0.0714]]], 4)
    assert_array_almost_equal(model.factors[2].probs, [[0.8333, 0.1667]], 4)
    assert_array_almost_equal(model.factors[3].probs, [[0.0, 0.8333], [0.0952, 0.0714]], 4)
    assert_array_almost_equal(model.marginals[0].probs, [[0.5, 0.5]])
    assert_array_almost_equal(model.marginals[1].probs, [[1.0 / 3, 1.0 / 3, 1.0 / 3]])
    assert_array_almost_equal(model.marginals[2].probs, [[0.5, 0.5]])
    assert_array_almost_equal(model.marginals[3].probs, [[0.5, 0.5]])
    model2.fit(X, sample_weight=w)
    assert_array_almost_equal(model2.factors[0].probs, [[0.6548, 0.3452]], 4)
    assert_array_almost_equal(model2.factors[1].probs, [[[0.0, 0.0, 0.0], [0.1964, 0.6369, 0.0]], [[0.0774, 0.0, 0.0179], [0.0, 0.0, 0.0714]]], 4)
    assert_array_almost_equal(model2.factors[2].probs, [[0.5595, 0.0952], [0.2738, 0.0714]], 4)
    assert_array_almost_equal(model2.factors[3].probs, [[0.0952, 0.5595], [0.0, 0.3452]], 4)
    assert_array_almost_equal(model2.marginals[0].probs, [[0.5, 0.5]])
    assert_array_almost_equal(model2.marginals[1].probs, [[1.0 / 3, 1.0 / 3, 1.0 / 3]])
    assert_array_almost_equal(model2.marginals[2].probs, [[0.5, 0.5]])
    assert_array_almost_equal(model2.marginals[3].probs, [[0.5, 0.5]])

def test_summarize(X, model, model2):
    if False:
        return 10
    model.summarize(X)
    assert_array_almost_equal(model.factors[0].probs, [[0.23, 0.77]])
    assert_array_almost_equal(model.factors[1].probs, [[[0.1, 0.05, 0.05], [0.15, 0.05, 0.04]], [[0.2, 0.1, 0.05], [0.05, 0.1, 0.06]]])
    assert_array_almost_equal(model.factors[2].probs, [[0.61, 0.39]])
    assert_array_almost_equal(model.factors[3].probs, [[0.17, 0.15], [0.4, 0.28]])
    assert_array_almost_equal(model.marginals[0].probs, [[0.5, 0.5]])
    assert_array_almost_equal(model.marginals[1].probs, [[1.0 / 3, 1.0 / 3, 1.0 / 3]])
    assert_array_almost_equal(model.marginals[2].probs, [[0.5, 0.5]])
    assert_array_almost_equal(model.marginals[3].probs, [[0.5, 0.5]])
    model2.summarize(X)
    assert_array_almost_equal(model2.factors[0].probs, [[0.23, 0.77]], 4)
    assert_array_almost_equal(model2.factors[1].probs, [[[0.1, 0.05, 0.05], [0.15, 0.05, 0.04]], [[0.2, 0.1, 0.05], [0.05, 0.1, 0.06]]])
    assert_array_almost_equal(model2.factors[2].probs, [[0.17, 0.15], [0.4, 0.28]])
    assert_array_almost_equal(model2.factors[3].probs, [[0.32, 0.12], [0.08, 0.48]])
    assert_array_almost_equal(model2.marginals[0].probs, [[0.5, 0.5]])
    assert_array_almost_equal(model2.marginals[1].probs, [[1.0 / 3, 1.0 / 3, 1.0 / 3]])
    assert_array_almost_equal(model2.marginals[2].probs, [[0.5, 0.5]])
    assert_array_almost_equal(model2.marginals[3].probs, [[0.5, 0.5]])

def test_from_summarize(X, model, model2):
    if False:
        print('Hello World!')
    model.summarize(X)
    model.from_summaries()
    assert_array_almost_equal(model.factors[0].probs, [[0.4545, 0.5455]], 4)
    assert_array_almost_equal(model.factors[1].probs, [[[0.0909, 0.1818, 0.0], [0.0909, 0.0, 0.0909]], [[0.0, 0.1818, 0.0909], [0.0909, 0.0909, 0.0909]]], 4)
    assert_array_almost_equal(model.factors[2].probs, [[0.5455, 0.4545]], 4)
    assert_array_almost_equal(model.factors[3].probs, [[0.1818, 0.3636], [0.2727, 0.1818]], 4)
    assert_array_almost_equal(model.marginals[0].probs, [[0.5, 0.5]])
    assert_array_almost_equal(model.marginals[1].probs, [[1.0 / 3, 1.0 / 3, 1.0 / 3]])
    assert_array_almost_equal(model.marginals[2].probs, [[0.5, 0.5]])
    assert_array_almost_equal(model.marginals[3].probs, [[0.5, 0.5]])
    model2.summarize(X)
    model2.from_summaries()
    assert_array_almost_equal(model2.factors[0].probs, [[0.4545, 0.5455]], 4)
    assert_array_almost_equal(model2.factors[1].probs, [[[0.0, 0.0909, 0.0909], [0.0909, 0.2727, 0.0]], [[0.0909, 0.0909, 0.0909], [0.0909, 0.0, 0.0909]]], 4)
    assert_array_almost_equal(model2.factors[2].probs, [[0.2727, 0.1818], [0.2727, 0.2727]], 4)
    assert_array_almost_equal(model2.factors[3].probs, [[0.2727, 0.1818], [0.1818, 0.3636]], 4)
    assert_array_almost_equal(model2.marginals[0].probs, [[0.5, 0.5]])
    assert_array_almost_equal(model2.marginals[1].probs, [[1.0 / 3, 1.0 / 3, 1.0 / 3]])
    assert_array_almost_equal(model2.marginals[2].probs, [[0.5, 0.5]])
    assert_array_almost_equal(model2.marginals[3].probs, [[0.5, 0.5]])