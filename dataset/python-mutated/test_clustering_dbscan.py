import unittest
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from Orange.clustering.clustering import ClusteringModel
from Orange.data import Table
from Orange.clustering.dbscan import DBSCAN

class TestDBSCAN(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.iris = Table('iris')
        self.dbscan = DBSCAN()

    def test_dbscan(self):
        if False:
            for i in range(10):
                print('nop')
        c = self.dbscan(self.iris)
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))
        self.assertEqual(1, len(set(c[:20].ravel())))

    def test_dbscan_parameters(self):
        if False:
            i = 10
            return i + 15
        dbscan = DBSCAN(eps=0.1, min_samples=7, metric='euclidean', algorithm='auto', leaf_size=12, p=None)
        c = dbscan(self.iris)
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))

    def test_predict_table(self):
        if False:
            for i in range(10):
                print('nop')
        pred = self.dbscan(self.iris)
        self.assertEqual(np.ndarray, type(pred))
        self.assertEqual(len(self.iris), len(pred))

    def test_predict_numpy(self):
        if False:
            print('Hello World!')
        model = self.dbscan.fit(self.iris.X)
        self.assertEqual(ClusteringModel, type(model))
        self.assertEqual(np.ndarray, type(model.labels))
        self.assertEqual(len(self.iris), len(model.labels))

    def test_predict_sparse_csc(self):
        if False:
            print('Hello World!')
        with self.iris.unlocked():
            self.iris.X = csc_matrix(self.iris.X[::20])
        c = self.dbscan(self.iris)
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))

    def test_predict_spares_csr(self):
        if False:
            for i in range(10):
                print('nop')
        with self.iris.unlocked():
            self.iris.X = csr_matrix(self.iris.X[::20])
        c = self.dbscan(self.iris)
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))

    def test_model(self):
        if False:
            print('Hello World!')
        c = self.dbscan.get_model(self.iris)
        self.assertEqual(ClusteringModel, type(c))
        self.assertEqual(len(self.iris), len(c.labels))
        self.assertRaises(NotImplementedError, c, self.iris)

    def test_model_np(self):
        if False:
            while True:
                i = 10
        '\n        Test with numpy array as an input in model.\n        '
        c = self.dbscan.get_model(self.iris)
        self.assertRaises(NotImplementedError, c, self.iris.X)

    def test_model_sparse(self):
        if False:
            while True:
                i = 10
        '\n        Test with sparse array as an input in model.\n        '
        c = self.dbscan.get_model(self.iris)
        self.assertRaises(NotImplementedError, c, csr_matrix(self.iris.X))

    def test_model_instance(self):
        if False:
            while True:
                i = 10
        '\n        Test with instance as an input in model.\n        '
        c = self.dbscan.get_model(self.iris)
        self.assertRaises(NotImplementedError, c, self.iris[0])

    def test_model_list(self):
        if False:
            return 10
        '\n        Test with list as an input in model.\n        '
        c = self.dbscan.get_model(self.iris)
        self.assertRaises(NotImplementedError, c, self.iris.X.tolist())

    def test_model_bad_datatype(self):
        if False:
            print('Hello World!')
        '\n        Check model with data-type that is not supported.\n        '
        c = self.dbscan.get_model(self.iris)
        self.assertRaises(TypeError, c, 10)