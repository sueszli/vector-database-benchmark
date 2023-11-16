from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import unittest
import copy
import numpy as np
from numpy.testing import assert_allclose
import turicreate as tc
from . import util as test_util
from turicreate.util import _assert_sframe_equal as assert_sframe_equal
from turicreate.toolkits._main import ToolkitError
import sys
if sys.version_info.major == 3:
    unittest.TestCase.assertItemsEqual = unittest.TestCase.assertCountEqual

def make_clustering_data(n, d, seed=None):
    if False:
        while True:
            i = 10
    '\n    Construct a synthetic dataset with a variety of data types for testing\n    clustering models.\n    '
    if seed:
        np.random.seed(seed)
    sf = tc.SFrame()
    for i in range(d):
        sf['int{}'.format(i)] = np.random.randint(low=-10, high=10, size=n)
    for i in range(d):
        v = np.random.rand(n)
        sf['float{}'.format(i)] = v * 20 - 10
    string_col = test_util.uniform_string_column(n, word_length=5, alphabet_size=5, missingness=0.0)
    sf['dict0'] = tc.text_analytics.count_ngrams(string_col, n=3, method='character', to_lower=False)
    return sf

class KmeansModelTest(unittest.TestCase):
    """
    Unit test class for an already trained Kmeans model.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.n = 100
        self.dim = 3
        self.K = 10
        self.max_iter = 10
        self.sf = make_clustering_data(n=self.n, d=self.dim, seed=8)
        self.model = tc.kmeans.create(self.sf, num_clusters=self.K, max_iterations=self.max_iter, batch_size=None, verbose=False)

    def test__list_fields(self):
        if False:
            print('Hello World!')
        '\n        Check the model list fields method.\n        '
        correct_fields = ['batch_size', 'row_label_name', 'cluster_id', 'cluster_info', 'features', 'max_iterations', 'method', 'num_clusters', 'num_examples', 'num_features', 'num_unpacked_features', 'training_iterations', 'training_time', 'unpacked_features']
        self.assertItemsEqual(self.model._list_fields(), correct_fields)

    def test_get(self):
        if False:
            print('Hello World!')
        "\n        Check the various 'get' methods against known answers for each field.\n        "
        correct_fields = {'max_iterations': self.max_iter, 'row_label_name': 'row_id', 'num_clusters': self.K, 'num_examples': self.n, 'method': 'elkan', 'batch_size': self.n, 'num_features': 2 * self.dim + 1}
        print(self.model)
        for (field, ans) in correct_fields.items():
            self.assertEqual(self.model._get(field), ans, '{} failed'.format(field))
        self.assertGreaterEqual(self.model.training_time, 0)
        self.assertGreater(self.model.num_unpacked_features, self.n)
        self.assertItemsEqual(self.model.features, ['int0', 'int1', 'int2', 'float0', 'float1', 'float2', 'dict0'])

    def test_summaries(self):
        if False:
            i = 10
            return i + 15
        '\n        Unit test for __repr__, __str__, and model summary methods.\n        '
        try:
            ans = str(self.model)
        except:
            assert False, 'Model __repr__ failed.'
        try:
            print(self.model)
        except:
            assert False, 'Model print failed.'
        try:
            self.model.summary()
        except:
            assert False, 'Model summary failed.'

    def test_save_and_load(self):
        if False:
            i = 10
            return i + 15
        '\n        Ensure that model saving and loading retains all model information.\n        '
        with test_util.TempDirectory() as f:
            self.model.save(f)
            self.model = tc.load_model(f)
            try:
                self.test__list_fields()
                self.test_get()
            except:
                assert False, 'List fields or get failed after save and load.'
            try:
                self.test_summaries()
            except:
                assert False, 'Model summaries failed after save and load.'
            del self.model

    def test_predict_params(self):
        if False:
            return 10
        "\n        Make sure the parameters for `predict` work correctly. Don't worry\n        about accuracy of the results - see the KmeansResultsTest class for\n        that.\n        "
        with self.assertRaises(ToolkitError):
            ans = self.model.predict(tc.SFrame(), verbose=False)
        with self.assertRaises(ToolkitError):
            ans = self.model.predict(self.sf['int0'], verbose=False)
        with self.assertRaises(ToolkitError):
            ans = self.model.predict(self.sf[['int0']], verbose=False)
        with self.assertRaises(TypeError):
            ans = self.model.predict(self.sf, output_type=1, verbose=False)
        ans = self.model.predict(self.sf, output_type='cluster_id', verbose=False)
        self.assertIsInstance(ans, tc.SArray)
        self.assertTrue(ans.dtype == int)
        ans = self.model.predict(self.sf, output_type='distance', verbose=False)
        self.assertIsInstance(ans, tc.SArray)
        self.assertTrue(ans.dtype == float)

class KmeansCreateTest(unittest.TestCase):
    """
    Test creation of the Kmeans model with various parameter and dataset
    configurations.
    """

    @classmethod
    def setUpClass(self):
        if False:
            i = 10
            return i + 15
        self.n = 100
        self.dim = 3
        self.K = 10
        self.max_iter = 10
        self.verbose = False
        self.sf = make_clustering_data(n=self.n, d=self.dim, seed=8)

    def test_input_mutations(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Make sure inputs to the create() method are not mutated. Note that\n        'batch_size' may be mutated by the model, by design. The input data\n        does have integer types, which are cast internally to floats. The\n        user's data should not be changed at all.\n        "
        sf = copy.copy(self.sf)
        verbose = copy.copy(self.verbose)
        K = copy.copy(self.K)
        max_iter = copy.copy(self.max_iter)
        features = copy.copy(self.sf.column_names())
        m = tc.kmeans.create(sf, features=features, num_clusters=K, max_iterations=max_iter, verbose=verbose)
        assert_sframe_equal(sf, self.sf)
        self.assertEqual(verbose, self.verbose)
        self.assertEqual(K, self.K)
        self.assertEqual(max_iter, self.max_iter)
        self.assertEqual(features, self.sf.column_names())

    def test_bad_data(self):
        if False:
            while True:
                i = 10
        '\n        Test error trapping and handling for inappropriate input datasets, both\n        the main dataset, and initial centers.\n        '
        with self.assertRaises(ValueError):
            m = tc.kmeans.create(dataset=tc.SFrame(), num_clusters=self.K, max_iterations=self.max_iter, verbose=False)
        with self.assertRaises(TypeError):
            m = tc.kmeans.create(dataset=self.sf['int0'], num_clusters=self.K, max_iterations=self.max_iter, verbose=False)

    def test_bogus_parameters(self):
        if False:
            while True:
                i = 10
        '\n        Ensure error trapping works correctly for unacceptable parameter values\n        and types.\n        '
        for k in [0, -1, 'fossa', 3.5]:
            with self.assertRaises(ToolkitError):
                m = tc.kmeans.create(dataset=self.sf, num_clusters=k, verbose=False)
        with self.assertRaises(ValueError):
            m = tc.kmeans.create(dataset=self.sf, num_clusters=self.n + 1, verbose=False)
        for max_iter in [-1, 'fossa', 3.5]:
            with self.assertRaises(ToolkitError):
                m = tc.kmeans.create(dataset=self.sf, num_clusters=self.K, max_iterations=max_iter, verbose=False)
        for batch_size in [-1, 0, 'fossa', 3.5]:
            with self.assertRaises(ToolkitError):
                m = tc.kmeans.create(dataset=self.sf, num_clusters=self.K, batch_size=batch_size, verbose=False)

    def test_default_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that model creation works correctly with only default inputs.\n        '
        with self.assertRaises(ValueError):
            m = tc.kmeans.create(self.sf)
        m = tc.kmeans.create(self.sf, self.K)
        correct_fields = {'max_iterations': 10, 'num_features': 2 * self.dim + 1, 'method': 'elkan', 'batch_size': self.n}
        for (field, ans) in correct_fields.items():
            self.assertEqual(m._get(field), ans, '{} failed'.format(field))

    def test_features_param(self):
        if False:
            while True:
                i = 10
        '\n        Test that the features are selected and screened correctly.\n        '
        m = tc.kmeans.create(self.sf, num_clusters=self.K, verbose=False)
        self.assertItemsEqual(m.features, ['int0', 'int1', 'int2', 'float0', 'float1', 'float2', 'dict0'])
        test_ftrs = ['int0', 'int1', 'int2']
        m = tc.kmeans.create(self.sf, num_clusters=self.K, features=test_ftrs, verbose=False)
        self.assertItemsEqual(m.features, test_ftrs)
        m = tc.kmeans.create(self.sf, num_clusters=self.K, features=['int0', 'int1', 'fossa'], verbose=False)
        self.assertItemsEqual(m.features, ['int0', 'int1'])
        test_ftrs = ['int0', 'int0', 'int1', 'int2']
        m = tc.kmeans.create(self.sf, num_clusters=self.K, features=test_ftrs, verbose=False)
        self.assertItemsEqual(m.features, ['int0', 'int1', 'int2'])
        test_ftrs = [2.71, 'int0', 'int1', 'int2']
        m = tc.kmeans.create(self.sf, num_clusters=self.K, features=test_ftrs, verbose=False)
        self.assertItemsEqual(m.features, ['int0', 'int1', 'int2'])
        sf = copy.copy(self.sf)
        sf['list0'] = sf['dict0'].dict_keys()
        test_ftrs = ['list0', 'int0', 'int1', 'int2']
        m = tc.kmeans.create(sf, num_clusters=self.K, features=test_ftrs, verbose=False)
        self.assertItemsEqual(m.features, ['int0', 'int1', 'int2'])
        sf = sf.add_row_number('row_id')
        test_ftrs = ['row_id', 'int0', 'int1', 'int2']
        m = tc.kmeans.create(sf, label='row_id', num_clusters=self.K, features=test_ftrs, verbose=False)
        self.assertItemsEqual(m.features, ['int0', 'int1', 'int2'])
        with self.assertRaises(ValueError):
            m = tc.kmeans.create(self.sf, features=[], num_clusters=self.K, verbose=False)
        test_ftrs = ['row_id', 'list0']
        with self.assertRaises(ToolkitError):
            m = tc.kmeans.create(sf, features=test_ftrs, label='row_id', num_clusters=self.K, verbose=False)

    def test_label_param(self):
        if False:
            return 10
        '\n        Make sure the `label` parameter works correctly.\n        '
        m = tc.kmeans.create(self.sf, num_clusters=self.K, verbose=False)
        self.assertItemsEqual(m.cluster_id.column_names(), ['row_id', 'cluster_id', 'distance'])
        self.assertEqual(m.cluster_id['row_id'].dtype, int)
        label_name = 'row_labels'
        sf = self.sf.add_row_number(label_name)
        sf[label_name] = sf[label_name].astype(str) + 'a'
        m = tc.kmeans.create(sf, label=label_name, num_clusters=self.K, verbose=False)
        self.assertItemsEqual(m.cluster_id.column_names(), [label_name, 'cluster_id', 'distance'])
        self.assertEqual(m.cluster_id[label_name].dtype, str)
        label_name = 'row_id'
        sf = self.sf.add_row_number(label_name)
        sf[label_name] = sf[label_name].astype(str) + 'a'
        m = tc.kmeans.create(sf, label=label_name, num_clusters=self.K, verbose=False)
        self.assertItemsEqual(m.cluster_id.column_names(), [label_name, 'cluster_id', 'distance'])
        self.assertEqual(m.cluster_id[label_name].dtype, str)

    def test_batch_size(self):
        if False:
            i = 10
            return i + 15
        "\n        Test that the batch size parameter is dealt with correctly, including\n        the choice of training method, re-sizing of 'batch_size', and existence\n        of complete results.\n        "
        m = tc.kmeans.create(self.sf, num_clusters=self.K, batch_size=self.n / 5, max_iterations=10, verbose=False)
        self.assertEqual(m.method, 'minibatch')
        self.assertEqual(m.batch_size, self.n / 5)
        self.assertEqual(m.max_iterations, 10)
        self.assertEqual(m.cluster_id.num_rows(), self.n)
        self.assertEqual(m.cluster_info.num_rows(), self.K)
        m = tc.kmeans.create(self.sf, num_clusters=self.K, batch_size=2 * self.n, max_iterations=10, verbose=False)
        self.assertEqual(m.method, 'elkan')
        self.assertEqual(m.batch_size, self.n)
        self.assertEqual(m.max_iterations, 10)
        self.assertEqual(m.cluster_id.num_rows(), self.n)
        self.assertEqual(m.cluster_info.num_rows(), self.K)
        (n, K) = (6, 2)
        sf = make_clustering_data(n=n, d=self.dim, seed=11)
        m = tc.kmeans.create(sf, num_clusters=2, batch_size=10, verbose=False)
        self.assertEqual(m.method, 'elkan')
        self.assertEqual(m.batch_size, n)
        self.assertEqual(m.max_iterations, 10)
        self.assertEqual(m.cluster_id.num_rows(), n)
        self.assertEqual(m.cluster_info.num_rows(), K)

    def test_custom_initial_centers(self):
        if False:
            print('Hello World!')
        '\n        Test that the user can pass hard-coded initial cluster centers, and\n        that these are actually used to initialize the clusters.\n        '
        with self.assertRaises(ValueError):
            m = tc.kmeans.create(dataset=self.sf, initial_centers=tc.SFrame(), max_iterations=self.max_iter, verbose=False)
        with self.assertRaises(TypeError):
            m = tc.kmeans.create(dataset=self.sf, initial_centers=tc.SArray([1, 2, 3]), max_iterations=self.max_iter, verbose=False)
        sf_init = make_clustering_data(n=10, d=self.dim - 1, seed=43)
        with self.assertRaises(ValueError):
            m = tc.kmeans.create(dataset=self.sf, initial_centers=sf_init, max_iterations=self.max_iter, verbose=False)
        sf_init = make_clustering_data(n=10, d=self.dim, seed=43)
        ftrs = ['float0', 'float1', 'dict0']
        m = tc.kmeans.create(self.sf, features=ftrs, initial_centers=sf_init, max_iterations=0, verbose=False)
        model_init_centers = m.cluster_info
        assert_sframe_equal(sf_init[ftrs], model_init_centers[ftrs])

    def test_random_initial_centers(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make sure randomly initialized cluster centers work correctly.\n        '
        m = tc.kmeans.create(self.sf, num_clusters=self.K, max_iterations=0, verbose=False)
        self.assertEqual(m.cluster_id.num_rows(), self.n)
        self.assertItemsEqual(m.cluster_id.column_names(), ['row_id', 'cluster_id', 'distance'])
        self.assertItemsEqual(m.cluster_id['cluster_id'].unique(), range(10))
        self.assertItemsEqual(m.cluster_info['cluster_id'], range(10))
        self.assertTrue((m.cluster_info['size'] > 0).all())
        self.assertEqual(m.cluster_info.num_rows(), self.K)
        self.assertItemsEqual(m.cluster_info.column_names(), self.sf.column_names() + ['cluster_id', 'size', 'sum_squared_distance'])
        self.assertEqual(m.training_iterations, 0)
        self.assertGreaterEqual(m.training_time, 0.0)
        self.assertEqual(m.num_clusters, self.K)

class KmeansClusterTest(unittest.TestCase):
    """
    Unit test class for the correctness or plausibility of Kmeans clustering
    results.
    """

    @classmethod
    def setUpClass(self):
        if False:
            return 10
        self.n = 100
        self.sf = make_clustering_data(n=self.n, d=2, seed=43)

    def test_extreme_cluster_numbers(self):
        if False:
            i = 10
            return i + 15
        '\n        Test results for one cluster and for one cluster per point.\n        '
        m = tc.kmeans.create(self.sf, num_clusters=1, verbose=False)
        self.assertEqual(m.cluster_info.num_rows(), 1)
        self.assertEqual(m.cluster_info['cluster_id'][0], 0)
        self.assertEqual(m.cluster_info['size'][0], self.n)
        self.assertTrue(all(m.cluster_id['cluster_id'] == 0))
        m = tc.kmeans.create(self.sf, num_clusters=self.n, verbose=False)
        self.assertItemsEqual(m.cluster_id['cluster_id'], range(self.n))
        self.assertTrue(all(m.cluster_id['distance'] < 1e-12))
        self.assertItemsEqual(m.cluster_info['cluster_id'], range(self.n))
        self.assertTrue(all(m.cluster_info['size'] == 1))
        self.assertTrue(all(m.cluster_info['sum_squared_distance'] < 1e-12))

    def test_distance_accuracy(self):
        if False:
            return 10
        '\n        Check that Kmeans distances match nearest neighbors distances. This was\n        a problem in early versions of the tool due to integer casting in the\n        cluster centers.\n        '
        ftrs = ['int0', 'int1', 'float0']
        kmeans = tc.kmeans.create(self.sf, features=ftrs, num_clusters=3, verbose=False)
        knn = tc.nearest_neighbors.create(kmeans.cluster_info, features=ftrs, method='ball_tree', distance='euclidean', verbose=False)
        coltype_map = {k: v for (k, v) in zip(self.sf.column_names(), self.sf.column_types())}
        sf_float = tc.SFrame()
        for ftr in ftrs:
            if coltype_map[ftr] is int:
                sf_float[ftr] = self.sf[ftr].astype(float)
            else:
                sf_float[ftr] = self.sf[ftr]
        knn_dists = knn.query(sf_float, k=1, radius=None, verbose=False)
        self.assertTrue((kmeans.cluster_id['row_id'] == knn_dists['query_label']).all())
        self.assertTrue((kmeans.cluster_id['cluster_id'] == knn_dists['reference_label']).all())
        assert_allclose(kmeans.cluster_id['distance'], knn_dists['distance'])

    def test_predictions(self):
        if False:
            i = 10
            return i + 15
        '\n        Test correctness of predictions on new data, by comparing to nearest\n        neighbors search results. Note that this implicitly checks that integer\n        features are correctly cast as floats in the predict method.\n        '
        sf_train = self.sf[:-10]
        sf_predict = self.sf[-10:]
        kmeans = tc.kmeans.create(sf_train, num_clusters=3, verbose=False)
        sf_train_copy = copy.copy(sf_train)
        yhat = kmeans.predict(sf_train)
        assert_sframe_equal(sf_train, sf_train_copy)
        self.assertTrue((yhat == kmeans.cluster_id['cluster_id']).all())
        yhat_dists = kmeans.predict(sf_train, output_type='distance')
        assert_allclose(yhat_dists, kmeans.cluster_id['distance'], rtol=1e-06)
        ystar_labels = kmeans.predict(sf_predict, output_type='cluster_id')
        ystar_dists = kmeans.predict(sf_predict, output_type='distance')
        ystar = tc.SFrame({'cluster_id': ystar_labels, 'distance': ystar_dists})
        ystar = ystar.add_row_number('row_id')
        coltype_map = {k: v for (k, v) in zip(sf_predict.column_names(), sf_predict.column_types())}
        for ftr in coltype_map.keys():
            if coltype_map[ftr] is int:
                sf_predict[ftr] = sf_predict[ftr].astype(float)
        knn_model = tc.nearest_neighbors.create(kmeans.cluster_info, features=kmeans.features, distance='euclidean', method='ball_tree')
        knn_dists = knn_model.query(sf_predict, k=1, radius=None)
        assert_sframe_equal(ystar[['row_id', 'cluster_id']], knn_dists[['query_label', 'reference_label']], check_column_names=False)
        assert_allclose(ystar['distance'], knn_dists['distance'], rtol=1e-06)