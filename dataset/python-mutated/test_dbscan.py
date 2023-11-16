from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import copy
import numpy as np
import unittest
import turicreate as tc
from . import util as test_util
from turicreate.util import _assert_sframe_equal as assert_sframe_equal
from turicreate.toolkits._main import ToolkitError
from .test_knn_classifier import make_classifier_data
import sys
if sys.version_info.major == 3:
    unittest.TestCase.assertItemsEqual = unittest.TestCase.assertCountEqual
import os as _os

class CreateTest(unittest.TestCase):
    """
    Test the create method for DBSCAN clustering.
    """

    @classmethod
    def setUpClass(self):
        if False:
            for i in range(10):
                print('nop')
        self.sf = tc.SFrame({'X1': [[0.286053821661, 0.958105566519], [0.770312932219, 0.986870003092], [0.208165461905, 0.136917048844], [0.90837380229, 0.0686385179771], [0.0753327223397, 0.543534689487], [0.0893997165181, 0.382393267526], [0.668560439681, 0.429169022562], [0.0439563074864, 0.194285988749], [0.446659483973, 0.062573278102], [0.297567282015, 0.943630899918], [0.282811075761, 0.267693546553], [0.407219004134, 0.825990402953], [0.506700663192, 0.269475381046], [0.340188287419, 0.97447185149], [0.18430457912, 0.242272172626], [0.6904593137, 0.383935276414], [0.461442452896, 0.675224987045], [0.0857306038525, 0.234016647286], [0.522458878224, 0.0691166755345], [0.0902366982884, 0.0839678579833], [0.3228005527, 0.910903399861], [0.831990012991, 0.75008026969], [0.469253814747, 0.867324370425], [0.279287904686, 0.0816360972888], [0.14921147693, 0.494767544759], [0.303711931037, 0.301766873086], [0.007386922447, 0.579463366777], [0.718318063984, 0.407263481941], [0.162964200289, 0.210306678644], [0.760123026079, 0.357788149323]]})
        self.min_core_neighbors = 3
        self.radius = 0.3
        self.distance = [[['X1'], 'euclidean', 1]]
        self.model = tc.dbscan.create(self.sf, distance=self.distance, radius=self.radius, min_core_neighbors=self.min_core_neighbors, verbose=False)

    def test_input_mutations(self):
        if False:
            print('Hello World!')
        '\n        Make sure inputs to the create() method are not mutated.\n        '
        local_sf = copy.copy(self.sf)
        local_dist = copy.deepcopy(self.distance)
        local_radius = copy.deepcopy(self.radius)
        local_min_core_neighbors = copy.deepcopy(self.min_core_neighbors)
        local_model = tc.dbscan.create(self.sf, distance=self.distance, radius=self.radius, min_core_neighbors=self.min_core_neighbors, verbose=False)
        assert_sframe_equal(self.sf, local_sf)
        self.assertEqual(self.distance, local_dist)
        self.assertEqual(self.radius, local_radius)
        self.assertEqual(self.min_core_neighbors, local_min_core_neighbors)

    def test_bogus_inputs(self):
        if False:
            return 10
        '\n        Check that bad inputs are handled appropriately.\n        '
        with self.assertRaises(ToolkitError):
            tc.dbscan.create(dataset=tc.SFrame(), radius=1.0, min_core_neighbors=5, verbose=False)
        with self.assertRaises(ToolkitError):
            tc.dbscan.create(dataset=self.sf.to_dataframe(), radius=1.0, min_core_neighbors=5, verbose=False)
        for val in [-1, 'fossa', [1.0, 2.0, 3.0]]:
            with self.assertRaises(ValueError):
                tc.dbscan.create(self.sf, distance='euclidean', radius=val, min_core_neighbors=self.min_core_neighbors, verbose=False)
            with self.assertRaises(ValueError):
                tc.dbscan.create(self.sf, distance='euclidean', radius=self.radius, min_core_neighbors=val, verbose=False)
        with self.assertRaises(TypeError):
            tc.dbscan.create(self.sf, distance=3)
        with self.assertRaises(ValueError):
            tc.dbscan.create(self.sf, distance='fossa')

    def test_create_features(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make sure the features parameter works properly, particularly with\n        respect to the distance parameter, which can be composite. These\n        parameters get passed directly to a nearest neighbors model, so they\n        would work in the same way.\n        '
        self.assertItemsEqual(self.model.features, ['X1'])
        self.assertItemsEqual(self.model.unpacked_features, ['X1[0]', 'X1[1]'])
        sf = self.sf.unpack('X1')
        m = tc.dbscan.create(sf, distance='euclidean', radius=self.radius, min_core_neighbors=self.min_core_neighbors, verbose=False)
        self.assertItemsEqual(m.features, ['X1.0', 'X1.1'])
        m = tc.dbscan.create(sf, features=['X1.0'], distance='euclidean', radius=self.radius, min_core_neighbors=self.min_core_neighbors, verbose=False)
        self.assertItemsEqual(m.features, ['X1.0'])
        test_dist = [[['X1.0'], 'euclidean', 1], [['X1.1'], 'manhattan', 1]]
        m = tc.dbscan.create(sf, distance=test_dist, radius=self.radius, min_core_neighbors=self.min_core_neighbors, verbose=False)
        self.assertItemsEqual(m.features, ['X1.0', 'X1.1'])
        m = tc.dbscan.create(sf, features=['X1.0'], distance=test_dist, radius=self.radius, min_core_neighbors=self.min_core_neighbors, verbose=False)
        self.assertItemsEqual(m.features, ['X1.0', 'X1.1'])

    def test_distances(self):
        if False:
            i = 10
            return i + 15
        "\n        Check error trapping and processing of the 'distance' parameter, including\n        construction of an automatic composite distance if no distance is specified.\n        DBSCAN *should* rely entirely on the nearest neighbors toolkit for this.\n        "
        sf = make_classifier_data(n=10, d=2, seed=37)
        sf.remove_column('class', inplace=True)
        numeric_features = ['int0', 'int1', 'float0', 'float1']
        array_features = ['array0']
        string_features = ['str0']
        dict_features = ['dict0']
        for d in ['euclidean', 'squared_euclidean', 'manhattan', 'cosine', 'transformed_dot_product']:
            try:
                m = tc.dbscan.create(sf, features=numeric_features, distance=d, radius=1, min_core_neighbors=3, verbose=False)
            except:
                assert False, 'Standard distance {} failed.'.format(d)
        for d in ['euclidean', 'squared_euclidean', 'manhattan', 'cosine', 'transformed_dot_product']:
            try:
                m = tc.dbscan.create(sf, features=array_features, distance=d, radius=1, min_core_neighbors=3, verbose=False)
            except:
                assert False, 'Standard distance {} failed.'.format(d)
        for d in ['levenshtein']:
            try:
                m = tc.dbscan.create(sf, features=string_features, distance=d, radius=1, min_core_neighbors=3, verbose=False)
            except:
                assert False, 'Standard distance {} failed.'.format(d)
        for d in ['jaccard', 'weighted_jaccard', 'cosine', 'transformed_dot_product']:
            try:
                m = tc.dbscan.create(sf, features=dict_features, distance=d, radius=1, min_core_neighbors=3, verbose=False)
            except:
                assert False, 'Standard distance {} failed.'.format(d)
        with self.assertRaises(ValueError):
            m = tc.dbscan.create(sf, features=numeric_features, distance='levenshtein', radius=1, min_core_neighbors=3, verbose=False)
        with self.assertRaises(ToolkitError):
            m = tc.dbscan.create(sf, features=dict_features, distance='levenshtein', radius=1, min_core_neighbors=3, verbose=False)
        with self.assertRaises(ToolkitError):
            m = tc.dbscan.create(sf, features=string_features, distance='euclidean', radius=1, min_core_neighbors=3, verbose=False)
        correct_dist = [[['str0'], 'levenshtein', 1], [['str1'], 'levenshtein', 1], [['dict0'], 'jaccard', 1], [['int0', 'int1', 'float0', 'float1'], 'euclidean', 1], [['array0'], 'euclidean', 1]]
        m = tc.dbscan.create(sf, radius=1, distance=None, min_core_neighbors=3, verbose=False)
        self.assertItemsEqual(m.distance, correct_dist)
        m = tc.dbscan.create(sf, radius=1, distance='auto', min_core_neighbors=3, verbose=False)
        self.assertItemsEqual(m.distance, correct_dist)

class ModelMethodsTest(unittest.TestCase):
    """
    Check API functionality for a DBSCAN model that has already been created.
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        np.random.seed(29)
        sf = tc.SFrame(np.random.rand(30, 2))
        self.min_core_neighbors = 3
        self.radius = 0.3
        self.distance = [[['X1'], 'euclidean', 1]]
        self.model = tc.dbscan.create(sf, distance=self.distance, radius=self.radius, min_core_neighbors=self.min_core_neighbors, verbose=False)

    def test__list_fields(self):
        if False:
            while True:
                i = 10
        '\n        Check the model list fields method.\n        '
        correct_fields = ['distance', 'verbose', 'min_core_neighbors', 'num_features', 'unpacked_features', 'num_distance_components', 'training_time', 'radius', 'num_unpacked_features', 'num_examples', 'cluster_id', 'num_clusters', 'features']
        self.assertItemsEqual(self.model._list_fields(), correct_fields)

    def test_get(self):
        if False:
            return 10
        "\n        Check the various 'get' methods against known answers for each field.\n        "
        simple_fields = {'verbose': False, 'min_core_neighbors': self.min_core_neighbors, 'num_features': 1, 'num_unpacked_features': 2, 'num_distance_components': 1, 'radius': self.radius, 'num_examples': 30}
        for (field, ans) in simple_fields.items():
            self.assertEqual(self.model._get(field), ans, '{} failed'.format(field))
        _list_fields = {'distance': self.distance, 'unpacked_features': ['X1[0]', 'X1[1]'], 'features': ['X1']}
        for (field, ans) in _list_fields.items():
            self.assertItemsEqual(self.model._get(field), ans, '{} failed'.format(field))
        self.assertGreaterEqual(self.model.training_time, 0)
        self.assertGreaterEqual(self.model.num_clusters, 0)
        self.assertEqual(self.model.cluster_id.num_rows(), 30)

    def test_summaries(self):
        if False:
            print('Hello World!')
        '\n        Check that something comes out for __repr__, __str__, and model summary\n        methods.\n        '
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
            print('Hello World!')
        '\n        Ensure that model saving and loading retains all model information.\n        '
        with test_util.TempDirectory() as f:
            self.model.save(f)
            self.model = tc.load_model(f)
            try:
                self.test__list_fields()
                print('Saved model list fields passed')
                self.test_get()
                print('Saved model get passed')
                self.test_summaries()
                print('Saved model summaries passed')
            except:
                assert False, 'Failed during save and load tests.'
            del self.model

class ResultsTest(unittest.TestCase):
    """
    Test the quality of DBSCAN clustering.
    """

    @classmethod
    def setUpClass(self):
        if False:
            while True:
                i = 10
        np.random.seed(37)
        self.n = 30
        self.sf = tc.SFrame(np.random.rand(self.n, 2))

    def test_extreme_neighborhoods(self):
        if False:
            print('Hello World!')
        '\n        Test what happens when there are no core points, boundary points, and\n        noise points, respectively.\n        '
        m = tc.dbscan.create(self.sf, distance='euclidean', radius=0.0, min_core_neighbors=3, verbose=False)
        self.assertEqual(m.num_clusters, 0)
        self.assertEqual(sum(m.cluster_id['type'] == 'noise'), self.n)
        m = tc.dbscan.create(self.sf, distance='euclidean', radius=0.0, min_core_neighbors=31, verbose=False)
        self.assertEqual(m.num_clusters, 0)
        self.assertEqual(sum(m.cluster_id['type'] == 'noise'), self.n)
        m = tc.dbscan.create(self.sf, distance='euclidean', radius=100.0, min_core_neighbors=3, verbose=False)
        self.assertEqual(m.num_clusters, 1)
        self.assertEqual(sum(m.cluster_id['type'] == 'core'), self.n)
        m = tc.dbscan.create(self.sf, distance='euclidean', radius=0.5, min_core_neighbors=0, verbose=False)
        self.assertEqual(m.num_clusters, 1)
        self.assertEqual(sum(m.cluster_id['type'] == 'core'), self.n)