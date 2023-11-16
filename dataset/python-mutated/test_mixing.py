import pytest
np = pytest.importorskip('numpy')
import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing

class TestDegreeMixingDict(BaseTestDegreeMixing):

    def test_degree_mixing_dict_undirected(self):
        if False:
            print('Hello World!')
        d = nx.degree_mixing_dict(self.P4)
        d_result = {1: {2: 2}, 2: {1: 2, 2: 2}}
        assert d == d_result

    def test_degree_mixing_dict_undirected_normalized(self):
        if False:
            return 10
        d = nx.degree_mixing_dict(self.P4, normalized=True)
        d_result = {1: {2: 1.0 / 3}, 2: {1: 1.0 / 3, 2: 1.0 / 3}}
        assert d == d_result

    def test_degree_mixing_dict_directed(self):
        if False:
            i = 10
            return i + 15
        d = nx.degree_mixing_dict(self.D)
        print(d)
        d_result = {1: {3: 2}, 2: {1: 1, 3: 1}, 3: {}}
        assert d == d_result

    def test_degree_mixing_dict_multigraph(self):
        if False:
            return 10
        d = nx.degree_mixing_dict(self.M)
        d_result = {1: {2: 1}, 2: {1: 1, 3: 3}, 3: {2: 3}}
        assert d == d_result

    def test_degree_mixing_dict_weighted(self):
        if False:
            i = 10
            return i + 15
        d = nx.degree_mixing_dict(self.W, weight='weight')
        d_result = {0.5: {1.5: 1}, 1.5: {1.5: 6, 0.5: 1}}
        assert d == d_result

class TestDegreeMixingMatrix(BaseTestDegreeMixing):

    def test_degree_mixing_matrix_undirected(self):
        if False:
            print('Hello World!')
        a_result = np.array([[0, 2], [2, 2]])
        a = nx.degree_mixing_matrix(self.P4, normalized=False)
        np.testing.assert_equal(a, a_result)
        a = nx.degree_mixing_matrix(self.P4)
        np.testing.assert_equal(a, a_result / a_result.sum())

    def test_degree_mixing_matrix_directed(self):
        if False:
            print('Hello World!')
        a_result = np.array([[0, 0, 2], [1, 0, 1], [0, 0, 0]])
        a = nx.degree_mixing_matrix(self.D, normalized=False)
        np.testing.assert_equal(a, a_result)
        a = nx.degree_mixing_matrix(self.D)
        np.testing.assert_equal(a, a_result / a_result.sum())

    def test_degree_mixing_matrix_multigraph(self):
        if False:
            return 10
        a_result = np.array([[0, 1, 0], [1, 0, 3], [0, 3, 0]])
        a = nx.degree_mixing_matrix(self.M, normalized=False)
        np.testing.assert_equal(a, a_result)
        a = nx.degree_mixing_matrix(self.M)
        np.testing.assert_equal(a, a_result / a_result.sum())

    def test_degree_mixing_matrix_selfloop(self):
        if False:
            for i in range(10):
                print('nop')
        a_result = np.array([[2]])
        a = nx.degree_mixing_matrix(self.S, normalized=False)
        np.testing.assert_equal(a, a_result)
        a = nx.degree_mixing_matrix(self.S)
        np.testing.assert_equal(a, a_result / a_result.sum())

    def test_degree_mixing_matrix_weighted(self):
        if False:
            i = 10
            return i + 15
        a_result = np.array([[0.0, 1.0], [1.0, 6.0]])
        a = nx.degree_mixing_matrix(self.W, weight='weight', normalized=False)
        np.testing.assert_equal(a, a_result)
        a = nx.degree_mixing_matrix(self.W, weight='weight')
        np.testing.assert_equal(a, a_result / float(a_result.sum()))

    def test_degree_mixing_matrix_mapping(self):
        if False:
            i = 10
            return i + 15
        a_result = np.array([[6.0, 1.0], [1.0, 0.0]])
        mapping = {0.5: 1, 1.5: 0}
        a = nx.degree_mixing_matrix(self.W, weight='weight', normalized=False, mapping=mapping)
        np.testing.assert_equal(a, a_result)

class TestAttributeMixingDict(BaseTestAttributeMixing):

    def test_attribute_mixing_dict_undirected(self):
        if False:
            while True:
                i = 10
        d = nx.attribute_mixing_dict(self.G, 'fish')
        d_result = {'one': {'one': 2, 'red': 1}, 'two': {'two': 2, 'blue': 1}, 'red': {'one': 1}, 'blue': {'two': 1}}
        assert d == d_result

    def test_attribute_mixing_dict_directed(self):
        if False:
            print('Hello World!')
        d = nx.attribute_mixing_dict(self.D, 'fish')
        d_result = {'one': {'one': 1, 'red': 1}, 'two': {'two': 1, 'blue': 1}, 'red': {}, 'blue': {}}
        assert d == d_result

    def test_attribute_mixing_dict_multigraph(self):
        if False:
            i = 10
            return i + 15
        d = nx.attribute_mixing_dict(self.M, 'fish')
        d_result = {'one': {'one': 4}, 'two': {'two': 2}}
        assert d == d_result

class TestAttributeMixingMatrix(BaseTestAttributeMixing):

    def test_attribute_mixing_matrix_undirected(self):
        if False:
            while True:
                i = 10
        mapping = {'one': 0, 'two': 1, 'red': 2, 'blue': 3}
        a_result = np.array([[2, 0, 1, 0], [0, 2, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
        a = nx.attribute_mixing_matrix(self.G, 'fish', mapping=mapping, normalized=False)
        np.testing.assert_equal(a, a_result)
        a = nx.attribute_mixing_matrix(self.G, 'fish', mapping=mapping)
        np.testing.assert_equal(a, a_result / a_result.sum())

    def test_attribute_mixing_matrix_directed(self):
        if False:
            print('Hello World!')
        mapping = {'one': 0, 'two': 1, 'red': 2, 'blue': 3}
        a_result = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        a = nx.attribute_mixing_matrix(self.D, 'fish', mapping=mapping, normalized=False)
        np.testing.assert_equal(a, a_result)
        a = nx.attribute_mixing_matrix(self.D, 'fish', mapping=mapping)
        np.testing.assert_equal(a, a_result / a_result.sum())

    def test_attribute_mixing_matrix_multigraph(self):
        if False:
            while True:
                i = 10
        mapping = {'one': 0, 'two': 1, 'red': 2, 'blue': 3}
        a_result = np.array([[4, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        a = nx.attribute_mixing_matrix(self.M, 'fish', mapping=mapping, normalized=False)
        np.testing.assert_equal(a, a_result)
        a = nx.attribute_mixing_matrix(self.M, 'fish', mapping=mapping)
        np.testing.assert_equal(a, a_result / a_result.sum())

    def test_attribute_mixing_matrix_negative(self):
        if False:
            for i in range(10):
                print('nop')
        mapping = {-2: 0, -3: 1, -4: 2}
        a_result = np.array([[4.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        a = nx.attribute_mixing_matrix(self.N, 'margin', mapping=mapping, normalized=False)
        np.testing.assert_equal(a, a_result)
        a = nx.attribute_mixing_matrix(self.N, 'margin', mapping=mapping)
        np.testing.assert_equal(a, a_result / float(a_result.sum()))

    def test_attribute_mixing_matrix_float(self):
        if False:
            while True:
                i = 10
        mapping = {0.5: 1, 1.5: 0}
        a_result = np.array([[6.0, 1.0], [1.0, 0.0]])
        a = nx.attribute_mixing_matrix(self.F, 'margin', mapping=mapping, normalized=False)
        np.testing.assert_equal(a, a_result)
        a = nx.attribute_mixing_matrix(self.F, 'margin', mapping=mapping)
        np.testing.assert_equal(a, a_result / a_result.sum())