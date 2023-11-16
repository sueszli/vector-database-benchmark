import numpy
import torch
import pytest
from pomegranate.bayesian_network import BayesianNetwork
from pomegranate.distributions import Exponential
from pomegranate.distributions import Categorical
from pomegranate.distributions import ConditionalCategorical
from nose.tools import assert_raises
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

def assert_tuple_equal(x, y):
    if False:
        while True:
            i = 10
    assert len(x) == len(y)
    for i in range(len(x)):
        assert len(x[i]) == len(y[i])
        if len(x[i]) > 0:
            for j in range(len(x[i])):
                assert x[i][j] == y[i][j]

@pytest.fixture
def X():
    if False:
        i = 10
        return i + 15
    return [[1, 2, 0, 0], [0, 0, 1, 0], [1, 1, 1, 0], [1, 2, 1, 1], [1, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 2, 1, 0], [0, 0, 0, 1]]

@pytest.fixture
def w():
    if False:
        for i in range(10):
            print('nop')
    return [0, 1.3, 0, 1.2, 2.5, 6.1, 0, 0, 2.1, 0.3, 3.3]

@pytest.fixture
def X_masked(X):
    if False:
        print('Hello World!')
    mask = torch.tensor(numpy.array([[False, True, True, True], [True, True, False, False], [False, False, False, True], [True, True, True, True], [False, True, False, False], [True, True, True, False], [False, False, False, True], [True, False, True, True], [True, True, True, False], [True, True, True, True], [True, False, True, False]]))
    X = torch.tensor(numpy.array(X))
    return torch.masked.MaskedTensor(X, mask=mask)

@pytest.fixture
def distributions():
    if False:
        print('Hello World!')
    d1 = Categorical([[0.8, 0.1]])
    d2 = Categorical([[0.3, 0.6, 0.1]])
    d3 = Categorical([[0.9, 0.1]])
    d4 = Categorical([[0.3, 0.7]])
    d12 = ConditionalCategorical([[[0.4, 0.6], [0.7, 0.3]]])
    d22 = ConditionalCategorical([[[0.7, 0.1, 0.2], [0.5, 0.4, 0.1]]])
    d32 = ConditionalCategorical([[[0.5, 0.5], [0.1, 0.9]]])
    d13 = ConditionalCategorical([[[[0.6, 0.4], [0.3, 0.7]], [[0.5, 0.5], [0.2, 0.8]], [[0.3, 0.7], [0.9, 0.1]]]])
    return (d1, d2, d3, d4, d12, d22, d32, d13)

def test_initialization(distributions):
    if False:
        print('Hello World!')
    (d1, d2, d3, d4, d12, d22, d32, d13) = distributions
    model = BayesianNetwork()
    assert len(model.distributions) == 0
    assert len(model.edges) == 0
    assert len(model._marginal_mapping) == 0
    assert len(model._factor_mapping) == 0
    assert len(model._factor_graph.factors) == 0
    assert len(model._factor_graph.marginals) == 0
    assert len(model._factor_graph._factor_idxs) == 0
    assert len(model._factor_graph._marginal_idxs) == 0
    assert sum(map(len, model._factor_graph._factor_edges)) == 0
    assert sum(map(len, model._factor_graph._marginal_edges)) == 0
    model = BayesianNetwork([d1])
    assert len(model.distributions) == 1
    assert len(model.edges) == 0
    assert len(model._marginal_mapping) == 1
    assert len(model._factor_mapping) == 1
    assert len(model._factor_graph.factors) == 1
    assert len(model._factor_graph.marginals) == 1
    assert len(model._factor_graph._factor_idxs) == 1
    assert len(model._factor_graph._marginal_idxs) == 1
    assert sum(map(len, model._factor_graph._factor_edges)) == 1
    assert sum(map(len, model._factor_graph._marginal_edges)) == 1
    model = BayesianNetwork([d1, d2])
    assert len(model.distributions) == 2
    assert len(model.edges) == 0
    assert len(model._marginal_mapping) == 2
    assert len(model._factor_mapping) == 2
    assert len(model._factor_graph.factors) == 2
    assert len(model._factor_graph.marginals) == 2
    assert len(model._factor_graph._factor_idxs) == 2
    assert len(model._factor_graph._marginal_idxs) == 2
    assert sum(map(len, model._factor_graph._factor_edges)) == 2
    assert sum(map(len, model._factor_graph._marginal_edges)) == 2
    model = BayesianNetwork([d1, d12], [(d1, d12)])
    assert len(model.distributions) == 2
    assert len(model.edges) == 1
    assert len(model._marginal_mapping) == 2
    assert len(model._factor_mapping) == 2
    assert len(model._factor_graph.factors) == 2
    assert len(model._factor_graph.marginals) == 2
    assert len(model._factor_graph._factor_idxs) == 2
    assert len(model._factor_graph._marginal_idxs) == 2
    assert sum(map(len, model._factor_graph._factor_edges)) == 3
    assert sum(map(len, model._factor_graph._marginal_edges)) == 3

def test_initialization_raises(distributions):
    if False:
        return 10
    (d1, d2, d3, d4, d12, d22, d32, d13) = distributions
    assert_raises(ValueError, BayesianNetwork, [d1, d12], [(d12, d1)])
    assert_raises(ValueError, BayesianNetwork, None, [(d1, d12)])
    assert_raises(ValueError, BayesianNetwork, [d1], [(d1, d12)])
    assert_raises(ValueError, BayesianNetwork, [d12], [(d1, d12)])
    assert_raises(ValueError, BayesianNetwork, [d1, d12], [(d1, d1)])
    assert_raises(ValueError, BayesianNetwork, [d1, d12], [(d12, d12)])

def test_add_distribution(distributions):
    if False:
        return 10
    (d1, d2, d3, d4, d12, d22, d32, d13) = distributions
    model = BayesianNetwork()
    assert len(model.distributions) == 0
    assert len(model.edges) == 0
    assert model._marginal_mapping == {}
    assert model._factor_mapping == {}
    assert len(model._factor_graph.factors) == 0
    assert len(model._factor_graph.marginals) == 0
    assert model._factor_graph._factor_idxs == {}
    assert model._factor_graph._factor_edges == []
    assert model._factor_graph._marginal_idxs == {}
    assert model._factor_graph._marginal_edges == []
    model.add_distribution(d1)
    assert len(model.distributions) == 1
    assert len(model.edges) == 0
    assert len(model._marginal_mapping) == 1
    assert len(model._factor_mapping) == 1
    assert len(model._factor_graph.factors) == 1
    assert len(model._factor_graph.marginals) == 1
    assert len(model._factor_graph._factor_idxs) == 1
    assert model._factor_graph._factor_edges == [[0]]
    assert len(model._factor_graph._marginal_idxs) == 1
    assert model._factor_graph._marginal_edges == [[0]]
    model.add_distribution(d12)
    assert len(model.distributions) == 2
    assert len(model.edges) == 0
    assert len(model._marginal_mapping) == 2
    assert len(model._factor_mapping) == 2
    assert len(model._factor_graph.factors) == 2
    assert len(model._factor_graph.marginals) == 2
    assert len(model._factor_graph._factor_idxs) == 2
    assert model._factor_graph._factor_edges == [[0], [1]]
    assert len(model._factor_graph._marginal_idxs) == 2
    assert model._factor_graph._marginal_edges == [[0], [1]]

def test_add_distribution_raises(distributions):
    if False:
        for i in range(10):
            print('nop')
    (d1, d2, d3, d4, d12, d22, d32, d13) = distributions
    model = BayesianNetwork()
    assert_raises(ValueError, model.add_distribution, [d1])
    assert_raises(ValueError, model.add_distribution, None)
    assert_raises(ValueError, model.add_distribution, Exponential())

def test_add_edge(distributions):
    if False:
        while True:
            i = 10
    (d1, d2, d3, d4, d12, d22, d32, d13) = distributions
    model = BayesianNetwork([d1, d2, d12, d22, d13])
    assert len(model.distributions) == 5
    assert len(model.edges) == 0
    assert model._factor_graph._factor_edges == [[0], [1], [2], [3], [4]]
    assert model._factor_graph._marginal_edges == [[0], [1], [2], [3], [4]]
    model.add_edge(d1, d12)
    assert len(model.distributions) == 5
    assert len(model.edges) == 1
    assert model._factor_graph._factor_edges == [[0], [1], [0, 2], [3], [4]]
    assert model._factor_graph._marginal_edges == [[2, 0], [1], [2], [3], [4]]
    model.add_edge(d1, d22)
    assert len(model.distributions) == 5
    assert len(model.edges) == 2
    assert model._factor_graph._factor_edges == [[0], [1], [0, 2], [0, 3], [4]]
    assert model._factor_graph._marginal_edges == [[2, 3, 0], [1], [2], [3], [4]]
    model.add_edge(d2, d13)
    model.add_edge(d1, d13)
    assert len(model.distributions) == 5
    assert len(model.edges) == 4
    assert model._factor_graph._factor_edges == [[0], [1], [0, 2], [0, 3], [1, 0, 4]]
    assert model._factor_graph._marginal_edges == [[2, 3, 4, 0], [4, 1], [2], [3], [4]]

def test_add_edge_raises(distributions):
    if False:
        return 10
    (d1, d2, d3, d4, d12, d22, d32, d13) = distributions
    model = BayesianNetwork([d1, d2, d12, d22, d13])
    assert_raises(ValueError, model.add_edge, d1, d1)
    assert_raises(ValueError, model.add_edge, d2, d1)
    assert_raises(ValueError, model.add_edge, d12, d1)
    assert_raises(ValueError, model.add_edge, None, d1)
    assert_raises(ValueError, model.add_edge, d1, None)
    assert_raises(ValueError, model.add_edge, d3, d1)
    assert_raises(ValueError, model.add_edge, d1, d3)

@pytest.mark.sample
def test_sample(distributions):
    if False:
        print('Hello World!')
    torch.manual_seed(0)
    (d1, d2, d3, d4, d12, d22, d32, d13) = distributions
    model = BayesianNetwork([d1, d22, d13, d4], [(d1, d22), (d22, d13), (d4, d13)])
    X = model.sample(1)
    assert_array_equal(X, [[1, 1, 1, 1]])
    X = model.sample(5)
    assert_array_equal(X, [[0, 0, 1, 1], [0, 0, 0, 1], [0, 2, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]])

def test_log_probability(X, distributions):
    if False:
        for i in range(10):
            print('nop')
    (d1, d2, d3, d4, d12, d22, d32, d13) = distributions
    model = BayesianNetwork([d1, d22, d13, d4], [(d1, d22), (d22, d13), (d4, d13)])
    logps = model.log_probability(X)
    assert_array_almost_equal(logps, [-7.013116, -2.700082, -5.115996, -7.264431, -5.184989, -4.491842, -4.422849, -3.709082, -5.184989, -3.393229, -2.140466])

def test_probability(X, distributions):
    if False:
        while True:
            i = 10
    (d1, d2, d3, d4, d12, d22, d32, d13) = distributions
    model = BayesianNetwork([d1, d22, d13, d4], [(d1, d22), (d22, d13), (d4, d13)])
    ps = model.probability(X)
    assert_array_almost_equal(ps, numpy.exp([-7.013116, -2.700082, -5.115996, -7.264431, -5.184989, -4.491842, -4.422849, -3.709082, -5.184989, -3.393229, -2.140466]), 4)

def test_predict_proba_one_node():
    if False:
        print('Hello World!')
    d = Categorical([[0.23, 0.17, 0.6]])
    X = torch.tensor([[0], [0], [1], [2]])
    mask = torch.tensor([[False], [True], [True], [True]])
    X_masked = torch.masked.MaskedTensor(X, mask=mask)
    model = BayesianNetwork([d])
    y_hat = model.predict_proba(X_masked)
    assert_array_almost_equal(y_hat[0], [[0.23, 0.17, 0.6], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

def test_predict_proba_one_node_raises():
    if False:
        while True:
            i = 10
    d = Categorical([[0.23, 0.17, 0.6]])
    model = BayesianNetwork([d])
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

def test_predict_proba_one_edge(distributions):
    if False:
        return 10
    (d1, d2, d3, d4, d12, d22, d32, d13) = distributions
    model = BayesianNetwork([d1, d22], [(d1, d22)])
    X = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 2]])
    mask = torch.tensor([[False, False], [True, False], [True, True], [False, True]])
    X_masked = torch.masked.MaskedTensor(X, mask=mask)
    y_hat = model.predict_proba(X_masked)
    assert_array_almost_equal(y_hat[0], [[0.8889, 0.1111], [1.0, 0.0], [0.0, 1.0], [0.9412, 0.0588]], 4)
    assert_array_almost_equal(y_hat[1], [[0.6778, 0.1333, 0.1889], [0.7, 0.1, 0.2], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], 4)

def test_predict_proba_monty_hall():
    if False:
        i = 10
        return i + 15
    p = numpy.array([[[[0.0, 0.5, 0.5], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0], [0.5, 0.0, 0.5], [1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]]])
    d1 = Categorical([[1.0 / 3, 1.0 / 3, 1.0 / 3]])
    d2 = Categorical([[1.0 / 3, 1.0 / 3, 1.0 / 3]])
    d3 = ConditionalCategorical(p)
    X = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0]])
    mask = torch.tensor([[False, False, False], [False, True, False], [False, True, False], [True, False, False], [True, False, False]])
    X_masked = torch.masked.MaskedTensor(X, mask=mask)
    model = BayesianNetwork([d1, d2, d3], [(d1, d3), (d2, d3)])
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

def test_predict_proba_simple(X_masked, distributions):
    if False:
        i = 10
        return i + 15
    (d1, d2, d3, d4, d12, d22, d32, d13) = distributions
    model = BayesianNetwork([d1, d22, d13, d4], [(d1, d22), (d22, d13), (d4, d13)])
    y_hat = model.predict_proba(X_masked)
    assert_array_almost_equal(y_hat[0], [[0.9412, 0.0588], [1.0, 0.0], [0.8889, 0.1111], [0.0, 1.0], [0.6667, 0.3333], [1.0, 0.0], [0.8889, 0.1111], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]], 4)
    assert_array_almost_equal(y_hat[1], [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.6778, 0.1333, 0.1889], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.6778, 0.1333, 0.1889], [0.5147, 0.4706, 0.0147], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.6121, 0.065, 0.3229]], 4)
    assert_array_almost_equal(y_hat[2], [[1.0, 0.0], [0.39, 0.61], [0.53, 0.47], [0.0, 1.0], [0.29, 0.71], [1.0, 0.0], [0.53, 0.47], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], 4)
    assert_array_almost_equal(y_hat[3], [[1.0, 0.0], [0.3, 0.7], [1.0, 0.0], [0.0, 1.0], [0.3, 0.7], [0.5172, 0.4828], [1.0, 0.0], [0.0, 1.0], [0.5172, 0.4828], [1.0, 0.0], [0.3565, 0.6435]], 4)

def test_predict_proba_diamond(X_masked, distributions):
    if False:
        for i in range(10):
            print('nop')
    (d1, d2, d3, d4, d12, d22, d32, d13) = distributions
    model = BayesianNetwork([d13, d22, d32, d4], [(d4, d22), (d4, d32), (d22, d13), (d32, d13)])
    y_hat = model.predict_proba(X_masked)
    assert_array_almost_equal(y_hat[0], [[0.3, 0.7], [1.0, 0.0], [0.47, 0.53], [0.0, 1.0], [0.2416, 0.7584], [1.0, 0.0], [0.47, 0.53], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]], 4)
    assert_array_almost_equal(y_hat[1], [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.7, 0.1, 0.2], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.1, 0.2], [0.5147, 0.4706, 0.0147], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.7204, 0.1844, 0.0952]], 4)
    assert_array_almost_equal(y_hat[2], [[1.0, 0.0], [0.4, 0.6], [0.5, 0.5], [0.0, 1.0], [0.1387, 0.8613], [1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], 4)
    assert_array_almost_equal(y_hat[3], [[1.0, 0.0], [0.45, 0.55], [1.0, 0.0], [0.0, 1.0], [0.0968, 0.9032], [0.3488, 0.6512], [1.0, 0.0], [0.0, 1.0], [0.3488, 0.6512], [1.0, 0.0], [0.6818, 0.3182]], 4)

def test_predict_proba_cycle(X, X_masked, distributions):
    if False:
        while True:
            i = 10
    model = BayesianNetwork(structure=((1,), (2,), (3,), (0,))).fit(X)
    y_hat = model.predict_proba(X_masked)
    assert_array_almost_equal(y_hat[0], [[0.4737, 0.5263], [1.0, 0.0], [0.6065, 0.3935], [0.0, 1.0], [0.3825, 0.6175], [1.0, 0.0], [0.6065, 0.3935], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]], 4)
    assert_array_almost_equal(y_hat[1], [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.3435, 0.3728, 0.2837], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.3435, 0.3728, 0.2837], [0.2564, 0.2308, 0.5128], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.2564, 0.6154, 0.1282]], 4)
    assert_array_almost_equal(y_hat[2], [[1.0, 0.0], [0.2997, 0.7003], [0.3935, 0.6065], [0.0, 1.0], [0.8025, 0.1975], [1.0, 0.0], [0.3935, 0.6065], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], 4)
    assert_array_almost_equal(y_hat[3], [[1.0, 0.0], [0.653, 0.347], [1.0, 0.0], [0.0, 1.0], [0.3728, 0.6272], [0.4737, 0.5263], [1.0, 0.0], [0.0, 1.0], [0.2308, 0.7692], [1.0, 0.0], [0.4737, 0.5263]], 4)

def test_predict_log_proba_diamond(X_masked, distributions):
    if False:
        for i in range(10):
            print('nop')
    (d1, d2, d3, d4, d12, d22, d32, d13) = distributions
    model = BayesianNetwork([d13, d22, d32, d4], [(d4, d22), (d4, d32), (d22, d13), (d32, d13)])
    y_hat = model.predict_log_proba(X_masked)
    assert_array_almost_equal(y_hat[0], numpy.log([[0.3, 0.7], [1.0, 0.0], [0.47, 0.53], [0.0, 1.0], [0.2416, 0.7584], [1.0, 0.0], [0.47, 0.53], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]), 3)
    assert_array_almost_equal(y_hat[1], numpy.log([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.7, 0.1, 0.2], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.1, 0.2], [0.5147, 0.4706, 0.0147], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.7204, 0.1844, 0.0952]]), 3)
    assert_array_almost_equal(y_hat[2], numpy.log([[1.0, 0.0], [0.4, 0.6], [0.5, 0.5], [0.0, 1.0], [0.1387, 0.8613], [1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]), 3)
    assert_array_almost_equal(y_hat[3], numpy.log([[1.0, 0.0], [0.45, 0.55], [1.0, 0.0], [0.0, 1.0], [0.0968, 0.9032], [0.3488, 0.6512], [1.0, 0.0], [0.0, 1.0], [0.3488, 0.6512], [1.0, 0.0], [0.6818, 0.3182]]), 3)

def test_predict_cycle(X, X_masked, distributions):
    if False:
        i = 10
        return i + 15
    model = BayesianNetwork(structure=((1,), (2,), (3,), (0,))).fit(X)
    y_hat = model.predict(X_masked)
    assert_array_equal(y_hat, [[1, 2, 0, 0], [0, 0, 1, 0], [0, 1, 1, 0], [1, 2, 1, 1], [1, 1, 0, 1], [0, 1, 0, 1], [0, 1, 1, 0], [1, 2, 1, 1], [1, 1, 0, 1], [0, 2, 1, 0], [0, 1, 0, 1]])

def test_predict_diamond(X_masked, distributions):
    if False:
        for i in range(10):
            print('nop')
    (d1, d2, d3, d4, d12, d22, d32, d13) = distributions
    model = BayesianNetwork([d13, d22, d32, d4], [(d4, d22), (d4, d32), (d22, d13), (d32, d13)])
    y_hat = model.predict(X_masked)
    assert_array_equal(y_hat, [[1, 2, 0, 0], [0, 0, 1, 1], [1, 0, 0, 0], [1, 2, 1, 1], [1, 1, 1, 1], [0, 1, 0, 1], [1, 0, 0, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 2, 1, 0], [0, 0, 0, 0]])

def test_fit(X, distributions):
    if False:
        i = 10
        return i + 15
    (d1, d2, d3, d4, d12, d22, d32, d13) = distributions
    model = BayesianNetwork([d13, d22, d32, d4], [(d4, d22), (d4, d32), (d22, d13), (d32, d13)])
    model.fit(X)
    assert_array_almost_equal(model.distributions[0].probs[0], [[[1.0, 0.0], [0.5, 0.5]], [[0.5, 0.5], [0.0, 1.0]], [[0.0, 1.0], [0.5, 0.5]]])
    assert_array_almost_equal(model.distributions[1].probs[0], [[0.2, 0.4, 0.4], [0.3333, 0.5, 0.1667]], 4)
    assert_array_almost_equal(model.distributions[2].probs[0], [[0.4, 0.6], [0.6667, 0.3333]], 4)
    assert_array_almost_equal(model.distributions[3].probs, [[0.4545, 0.5455]], 4)

def test_fit_weighted(X, w, distributions):
    if False:
        for i in range(10):
            print('nop')
    (d1, d2, d3, d4, d12, d22, d32, d13) = distributions
    model = BayesianNetwork([d13, d22, d32, d4], [(d4, d22), (d4, d32), (d22, d13), (d32, d13)])
    model.fit(X, sample_weight=w)
    assert_array_almost_equal(model.distributions[0].probs[0], [[[1.0, 0.0], [1.0, 0.0]], [[0.5701, 0.4299], [0.5, 0.5]], [[0.5, 0.5], [0.2, 0.8]]], 4)
    assert_array_almost_equal(model.distributions[1].probs[0], [[0.8125, 0.0, 0.1875], [0.2171, 0.7039, 0.0789]], 4)
    assert_array_almost_equal(model.distributions[2].probs[0], [[0.0, 1.0], [0.9211, 0.0789]], 4)
    assert_array_almost_equal(model.distributions[3].probs, [[0.0952, 0.9048]], 4)

def test_fit_structure(X):
    if False:
        return 10
    model = BayesianNetwork(structure=((1, 2), (3,), (3,), ()))
    model.fit(X)
    assert_array_almost_equal(model.distributions[0].probs[0], [[[1.0, 0.0], [0.5, 0.5]], [[0.5, 0.5], [0.0, 1.0]], [[0.0, 1.0], [0.5, 0.5]]])
    assert_array_almost_equal(model.distributions[1].probs[0], [[0.2, 0.4, 0.4], [0.3333, 0.5, 0.1667]], 4)
    assert_array_almost_equal(model.distributions[2].probs[0], [[0.4, 0.6], [0.6667, 0.3333]], 4)
    assert_array_almost_equal(model.distributions[3].probs, [[0.4545, 0.5455]], 4)

def test_fit_fg(X, distributions):
    if False:
        return 10
    (d1, d2, d3, d4, d12, d22, d32, d13) = distributions
    model = BayesianNetwork([d13, d22, d32, d4], [(d4, d22), (d4, d32), (d22, d13), (d32, d13)])
    model.fit(X)
    assert_array_almost_equal(model._factor_graph.factors[0].probs, [[[0.1667, 0.0], [0.0833, 0.0833]], [[0.0833, 0.0833], [0.0, 0.1667]], [[0.0, 0.1667], [0.0833, 0.0833]]], 4)
    assert_array_almost_equal(model._factor_graph.factors[1].probs, [[0.1, 0.2, 0.2], [0.1667, 0.25, 0.0833]], 4)
    assert_array_almost_equal(model._factor_graph.factors[2].probs, [[0.2, 0.3], [0.3333, 0.1667]], 4)
    assert_array_almost_equal(model._factor_graph.factors[3].probs, [[0.4545, 0.5455]], 4)

def test_fit_weighted_fg(X, w, distributions):
    if False:
        i = 10
        return i + 15
    (d1, d2, d3, d4, d12, d22, d32, d13) = distributions
    model = BayesianNetwork([d13, d22, d32, d4], [(d4, d22), (d4, d32), (d22, d13), (d32, d13)])
    model.fit(X, sample_weight=w)
    assert_array_almost_equal(model._factor_graph.factors[0].probs, [[[0.1667, 0.0], [0.1667, 0.0]], [[0.095, 0.0717], [0.0833, 0.0833]], [[0.0833, 0.0833], [0.0333, 0.1333]]], 4)
    assert_array_almost_equal(model._factor_graph.factors[1].probs, [[0.4062, 0.0, 0.0938], [0.1086, 0.352, 0.0395]], 4)
    assert_array_almost_equal(model._factor_graph.factors[2].probs, [[0.0, 0.5], [0.4605, 0.0395]], 4)
    assert_array_almost_equal(model._factor_graph.factors[3].probs, [[0.0952, 0.9048]], 4)

def test_fit_predict_proba(X, X_masked, distributions):
    if False:
        i = 10
        return i + 15
    (d1, d2, d3, d4, d12, d22, d32, d13) = distributions
    model = BayesianNetwork([d13, d22, d32, d4], [(d4, d22), (d4, d32), (d22, d13), (d32, d13)])
    y_hat = model.predict_proba(X_masked)
    assert_array_almost_equal(y_hat[0], [[0.3, 0.7], [1.0, 0.0], [0.47, 0.53], [0.0, 1.0], [0.2416, 0.7584], [1.0, 0.0], [0.47, 0.53], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]], 4)
    assert_array_almost_equal(y_hat[1], [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.7, 0.1, 0.2], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.1, 0.2], [0.5147, 0.4706, 0.0147], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.7204, 0.1844, 0.0952]], 4)
    assert_array_almost_equal(y_hat[2], [[1.0, 0.0], [0.4, 0.6], [0.5, 0.5], [0.0, 1.0], [0.1387, 0.8613], [1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], 4)
    assert_array_almost_equal(y_hat[3], [[1.0, 0.0], [0.45, 0.55], [1.0, 0.0], [0.0, 1.0], [0.0968, 0.9032], [0.3488, 0.6512], [1.0, 0.0], [0.0, 1.0], [0.3488, 0.6512], [1.0, 0.0], [0.6818, 0.3182]], 4)
    model.fit(X)
    y_hat = model.predict_proba(X_masked)
    assert_array_almost_equal(y_hat[0], [[0.0, 1.0], [1.0, 0.0], [0.34, 0.66], [0.0, 1.0], [0.28, 0.72], [1.0, 0.0], [0.34, 0.66], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]], 4)
    assert_array_almost_equal(y_hat[1], [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.2, 0.4, 0.4], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.2, 0.4, 0.4], [0.2222, 0.6667, 0.1111], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5532, 0.4468, 0.0]], 4)
    assert_array_almost_equal(y_hat[2], [[1.0, 0.0], [0.7324, 0.2676], [0.4, 0.6], [0.0, 1.0], [0.56, 0.44], [1.0, 0.0], [0.4, 0.6], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], 4)
    assert_array_almost_equal(y_hat[3], [[1.0, 0.0], [0.2958, 0.7042], [1.0, 0.0], [0.0, 1.0], [0.4, 0.6], [0.2857, 0.7143], [1.0, 0.0], [0.0, 1.0], [0.2857, 0.7143], [1.0, 0.0], [0.2553, 0.7447]], 4)

def test_learn_structure_chow_liu(X):
    if False:
        for i in range(10):
            print('nop')
    X = numpy.array(X)
    X[:, 1] = X[:, 3]
    model = BayesianNetwork(algorithm='chow-liu')
    model.fit(X)
    assert_tuple_equal(model._parents, ((), (0,), (1,), (1,)))
    assert_array_almost_equal(model.distributions[0].probs, [[0.4545, 0.5455]], 4)
    assert_array_almost_equal(model.distributions[1].probs[0], [[0.6, 0.4], [0.3333, 0.6667]], 4)
    assert_array_almost_equal(model.distributions[2].probs[0], [[0.4, 0.6], [0.6667, 0.3333]], 4)
    assert_array_almost_equal(model.distributions[3].probs[0], [[1.0, 0.0], [0.0, 1.0]], 4)
    assert_array_almost_equal(model._factor_graph.factors[0].probs, [[0.4545, 0.5455]], 4)
    assert_array_almost_equal(model._factor_graph.factors[1].probs, [[0.3, 0.2], [0.1667, 0.3333]], 4)
    assert_array_almost_equal(model._factor_graph.factors[2].probs, [[0.2, 0.3], [0.3333, 0.1667]], 4)
    assert_array_almost_equal(model._factor_graph.factors[3].probs, [[0.5, 0.0], [0.0, 0.5]], 4)

def test_learn_structure_exact(X):
    if False:
        return 10
    X = numpy.array(X)
    X[:, 1] = X[:, 3]
    model = BayesianNetwork(algorithm='exact')
    model.fit(X)
    assert_tuple_equal(model._parents, ((), (), (0, 1), ()))
    assert_array_almost_equal(model.distributions[0].probs, [[0.4545, 0.5455]], 4)
    assert_array_almost_equal(model.distributions[1].probs, [[0.4545, 0.5455]], 4)
    assert_array_almost_equal(model.distributions[3].probs, [[0.4545, 0.5455]], 4)
    assert_array_almost_equal(model.distributions[2].probs[0], [[[0.3333, 0.6667], [1.0, 0.0]], [[0.5, 0.5], [0.5, 0.5]]], 4)
    assert_array_almost_equal(model._factor_graph.factors[0].probs, [[0.4545, 0.5455]], 4)
    assert_array_almost_equal(model._factor_graph.factors[1].probs, [[0.4545, 0.5455]], 4)
    assert_array_almost_equal(model._factor_graph.factors[3].probs, [[0.4545, 0.5455]], 4)
    assert_array_almost_equal(model._factor_graph.factors[2].probs, [[[0.0833, 0.1667], [0.25, 0.0]], [[0.125, 0.125], [0.125, 0.125]]], 4)

def test_summarize(X, distributions):
    if False:
        i = 10
        return i + 15
    (d1, d2, d3, d4, d12, d22, d32, d13) = distributions
    model = BayesianNetwork([d13, d22, d32, d4], [(d4, d22), (d4, d32), (d22, d13), (d32, d13)])
    model.summarize(X[:4])
    model.summarize(X[4:])
    model.from_summaries()
    assert_array_almost_equal(model.distributions[0].probs[0], [[[1.0, 0.0], [0.5, 0.5]], [[0.5, 0.5], [0.0, 1.0]], [[0.0, 1.0], [0.5, 0.5]]])
    assert_array_almost_equal(model.distributions[1].probs[0], [[0.2, 0.4, 0.4], [0.3333, 0.5, 0.1667]], 4)
    assert_array_almost_equal(model.distributions[2].probs[0], [[0.4, 0.6], [0.6667, 0.3333]], 4)
    assert_array_almost_equal(model.distributions[3].probs, [[0.4545, 0.5455]], 4)

def test_summarize_weighted(X, w, distributions):
    if False:
        i = 10
        return i + 15
    (d1, d2, d3, d4, d12, d22, d32, d13) = distributions
    model = BayesianNetwork([d13, d22, d32, d4], [(d4, d22), (d4, d32), (d22, d13), (d32, d13)])
    model.summarize(X[:4], sample_weight=w[:4])
    model.summarize(X[4:], sample_weight=w[4:])
    model.from_summaries()
    assert_array_almost_equal(model.distributions[0].probs[0], [[[1.0, 0.0], [1.0, 0.0]], [[0.5701, 0.4299], [0.5, 0.5]], [[0.5, 0.5], [0.2, 0.8]]], 4)
    assert_array_almost_equal(model.distributions[1].probs[0], [[0.8125, 0.0, 0.1875], [0.2171, 0.7039, 0.0789]], 4)
    assert_array_almost_equal(model.distributions[2].probs[0], [[0.0, 1.0], [0.9211, 0.0789]], 4)
    assert_array_almost_equal(model.distributions[3].probs, [[0.0952, 0.9048]], 4)