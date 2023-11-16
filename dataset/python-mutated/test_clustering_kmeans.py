import unittest
import warnings
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
import Orange
from Orange.clustering.kmeans import KMeans, KMeansModel
from Orange.data import Table, Domain, ContinuousVariable
from Orange.data.table import DomainTransformationError

class TestKMeans(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.kmeans = KMeans(n_clusters=2)
        self.iris = Orange.data.Table('iris')

    def test_kmeans(self):
        if False:
            while True:
                i = 10
        c = self.kmeans(self.iris)
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))
        self.assertEqual(1, len(set(c[:20].ravel())))

    def test_kmeans_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        kmeans = KMeans(n_clusters=10, max_iter=10, random_state=42, tol=0.001, init='random')
        c = kmeans(self.iris)
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))

    def test_predict_table(self):
        if False:
            print('Hello World!')
        c = self.kmeans(self.iris)
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))

    def test_predict_numpy(self):
        if False:
            return 10
        c = self.kmeans.fit(self.iris.X)
        self.assertEqual(KMeansModel, type(c))
        self.assertEqual(np.ndarray, type(c.labels))
        self.assertEqual(len(self.iris), len(c.labels))

    def test_predict_sparse_csc(self):
        if False:
            i = 10
            return i + 15
        with self.iris.unlocked():
            self.iris.X = csc_matrix(self.iris.X[::20])
        c = self.kmeans(self.iris)
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))

    def test_predict_spares_csr(self):
        if False:
            print('Hello World!')
        with self.iris.unlocked():
            self.iris.X = csr_matrix(self.iris.X[::20])
        c = self.kmeans(self.iris)
        self.assertEqual(np.ndarray, type(c))
        self.assertEqual(len(self.iris), len(c))

    def test_model(self):
        if False:
            print('Hello World!')
        c = self.kmeans.get_model(self.iris)
        self.assertEqual(KMeansModel, type(c))
        self.assertEqual(len(self.iris), len(c.labels))
        c1 = c(self.iris)
        np.testing.assert_array_almost_equal(c.labels, c1)

    def test_model_np(self):
        if False:
            while True:
                i = 10
        '\n        Test with numpy array as an input in model.\n        '
        c = self.kmeans.get_model(self.iris)
        c1 = c(self.iris.X)
        np.testing.assert_array_almost_equal(c.labels, c1)

    def test_model_sparse_csc(self):
        if False:
            print('Hello World!')
        '\n        Test with sparse array as an input in model.\n        '
        c = self.kmeans.get_model(self.iris)
        c1 = c(csc_matrix(self.iris.X))
        np.testing.assert_array_almost_equal(c.labels, c1)

    def test_model_sparse_csr(self):
        if False:
            return 10
        '\n        Test with sparse array as an input in model.\n        '
        c = self.kmeans.get_model(self.iris)
        c1 = c(csr_matrix(self.iris.X))
        np.testing.assert_array_almost_equal(c.labels, c1)

    def test_model_instance(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test with instance as an input in model.\n        '
        c = self.kmeans.get_model(self.iris)
        c1 = c(self.iris[0])
        self.assertEqual(c1, c.labels[0])

    def test_model_list(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test with list as an input in model.\n        '
        c = self.kmeans.get_model(self.iris)
        c1 = c(self.iris.X.tolist())
        np.testing.assert_array_almost_equal(c.labels, c1)
        c1 = c(self.iris.X.tolist()[0])
        np.testing.assert_array_almost_equal(c.labels[0], c1)

    def test_model_bad_datatype(self):
        if False:
            return 10
        '\n        Check model with data-type that is not supported.\n        '
        c = self.kmeans.get_model(self.iris)
        self.assertRaises(TypeError, c, 10)

    def test_model_data_table_domain(self):
        if False:
            i = 10
            return i + 15
        '\n        Check model with data-type that is not supported.\n        '
        data = Table(Domain(list(self.iris.domain.attributes) + [ContinuousVariable('a')]), np.concatenate((self.iris.X, np.ones((len(self.iris), 1))), axis=1))
        c = self.kmeans.get_model(self.iris)
        res = c(data)
        np.testing.assert_array_almost_equal(c.labels, res)
        self.assertRaises(DomainTransformationError, c, Table('housing'))

    def test_deprecated_silhouette(self):
        if False:
            while True:
                i = 10
        with warnings.catch_warnings(record=True) as w:
            KMeans(compute_silhouette_score=True)
            assert len(w) == 1
            assert issubclass(w[-1].category, DeprecationWarning)
        with warnings.catch_warnings(record=True) as w:
            KMeans(compute_silhouette_score=False)
            assert len(w) == 1
            assert issubclass(w[-1].category, DeprecationWarning)