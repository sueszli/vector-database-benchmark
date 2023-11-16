import unittest
import numpy as np
import networkx
from scipy.sparse import csc_matrix, csr_matrix
from Orange.clustering.clustering import ClusteringModel
from Orange.clustering.louvain import matrix_to_knn_graph
from Orange.data import Table
from Orange.clustering.louvain import Louvain

class TestLouvain(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.iris = Table('iris')
        self.louvain = Louvain()

    def test_louvain(self):
        if False:
            while True:
                i = 10
        c = self.louvain(self.iris)
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))
        self.assertEqual(1, len(set(c[:20].ravel())))

    def test_louvain_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        louvain = Louvain(k_neighbors=3, resolution=1.2, random_state=42, metric='l2')
        c = louvain(self.iris)
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))

    def test_predict_table(self):
        if False:
            i = 10
            return i + 15
        c = self.louvain(self.iris)
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))

    def test_predict_numpy(self):
        if False:
            while True:
                i = 10
        c = self.louvain.fit(self.iris.X)
        self.assertEqual(ClusteringModel, type(c))
        self.assertEqual(np.ndarray, type(c.labels))
        self.assertEqual(len(self.iris), len(c.labels))

    def test_predict_sparse_csc(self):
        if False:
            i = 10
            return i + 15
        with self.iris.unlocked():
            self.iris.X = csc_matrix(self.iris.X[::5])
        c = self.louvain(self.iris)
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))

    def test_predict_sparse_csr(self):
        if False:
            print('Hello World!')
        with self.iris.unlocked():
            self.iris.X = csr_matrix(self.iris.X[::5])
        c = self.louvain(self.iris)
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))

    def test_model(self):
        if False:
            for i in range(10):
                print('nop')
        c = self.louvain.get_model(self.iris)
        self.assertEqual(ClusteringModel, type(c))
        self.assertEqual(len(self.iris), len(c.labels))
        self.assertRaises(NotImplementedError, c, self.iris)

    def test_model_np(self):
        if False:
            i = 10
            return i + 15
        '\n        Test with numpy array as an input in model.\n        '
        c = self.louvain.get_model(self.iris)
        self.assertRaises(NotImplementedError, c, self.iris.X)

    def test_model_sparse(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test with sparse array as an input in model.\n        '
        c = self.louvain.get_model(self.iris)
        self.assertRaises(NotImplementedError, c, csr_matrix(self.iris.X))

    def test_model_instance(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test with instance as an input in model.\n        '
        c = self.louvain.get_model(self.iris)
        self.assertRaises(NotImplementedError, c, self.iris[0])

    def test_model_list(self):
        if False:
            while True:
                i = 10
        '\n        Test with list as an input in model.\n        '
        c = self.louvain.get_model(self.iris)
        self.assertRaises(NotImplementedError, c, self.iris.X.tolist())

    def test_graph(self):
        if False:
            return 10
        '\n        Louvain accepts graphs too.\n        :return:\n        '
        graph = matrix_to_knn_graph(self.iris.X, 30, 'l2')
        self.assertIsNotNone(graph)
        self.assertEqual(networkx.Graph, type(graph), 1)
        c = self.louvain(graph)
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))
        self.assertEqual(1, len(set(c[:20].ravel())))
        c = self.louvain.get_model(graph)
        self.assertEqual(ClusteringModel, type(c))
        self.assertEqual(len(self.iris), len(c.labels))

    def test_model_bad_datatype(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check model with data-type that is not supported.\n        '
        c = self.louvain.get_model(self.iris)
        self.assertRaises(TypeError, c, 10)