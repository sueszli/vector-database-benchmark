from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import unittest
import turicreate as tc
import numpy as np

class RegressionCreateTest(unittest.TestCase):
    """
    Unit test class for testing a regression model.
    """
    '\n       Creation test helper function.\n    '

    def _test_create(self, n, d, validation_set='auto'):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(42)
        sf = tc.SFrame()
        for i in range(d):
            sf.add_column(tc.SArray(np.random.rand(n)), inplace=True)
        target = np.random.rand(n)
        sf['target'] = target
        model = tc.regression.create(sf, target='target', features=None, validation_set=validation_set)
        self.assertTrue(model is not None)
        features = sf.column_names()
        features.remove('target')
        model = tc.regression.create(sf, target='target', features=features, validation_set=validation_set)
        self.assertTrue(model is not None)
    '\n       Test create.\n    '

    def test_create(self):
        if False:
            print('Hello World!')
        self._test_create(99, 10)
        self._test_create(100, 100)
        self._test_create(20000, 10)
        self._test_create(99, 10, validation_set=None)
        self._test_create(100, 100, validation_set=None)
        self._test_create(20000, 10, validation_set=None)