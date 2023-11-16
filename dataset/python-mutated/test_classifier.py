from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import unittest
import turicreate as tc
import numpy as np

class ClassifierCreateTest(unittest.TestCase):
    """
    Unit test class for testing a classifier model.
    """

    def _test_create(self, n, d, validation_set='auto'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creation test helper function.\n        '
        np.random.seed(42)
        sf = tc.SFrame()
        for i in range(d):
            sf.add_column(tc.SArray(np.random.rand(n)), inplace=True)
        target = np.random.randn(n)
        sf['target'] = target
        sf['target'] = sf['target'] > 0
        model = tc.classifier.create(sf, 'target', features=None, validation_set=validation_set)
        self.assertTrue(model is not None, 'Model is None.')
        features = sf.column_names()
        features.remove('target')
        model = tc.classifier.create(sf, 'target', features=features, validation_set=validation_set)
        self.assertTrue(model is not None, 'Model is None.')
        self.assertTrue(isinstance(model, tc.toolkits._supervised_learning.SupervisedLearningModel))

    def test_multi_class_create(self):
        if False:
            for i in range(10):
                print('nop')
        d = 10
        n = 100
        np.random.seed(42)
        sf = tc.SFrame()
        for i in range(d):
            sf.add_column(tc.SArray(np.random.rand(n)), inplace=True)
        sf['target'] = [1, 2, 3, 4] * 25
        model = tc.classifier.create(sf, 'target')
        self.assertTrue(isinstance(model, tc.toolkits._supervised_learning.SupervisedLearningModel))

    def test_create(self):
        if False:
            i = 10
            return i + 15
        self._test_create(99, 10)
        self._test_create(100, 100)
        self._test_create(20000, 10)
        self._test_create(99, 10, validation_set=None)
        self._test_create(100, 100, validation_set=None)
        self._test_create(20000, 10, validation_set=None)