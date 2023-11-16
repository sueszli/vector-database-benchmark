from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from art.estimators.regression.scikitlearn import ScikitlearnDecisionTreeRegressor
from art.estimators.regression.scikitlearn import ScikitlearnRegressor
from tests.utils import TestBase, master_seed
logger = logging.getLogger(__name__)

class TestScikitlearnDecisionTreeRegressor(TestBase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        master_seed(seed=1234)
        super().setUpClass()
        cls.sklearn_model = DecisionTreeRegressor()
        cls.classifier = ScikitlearnDecisionTreeRegressor(model=cls.sklearn_model)
        cls.classifier.fit(x=cls.x_train_diabetes, y=cls.y_train_diabetes)

    def test_type(self):
        if False:
            i = 10
            return i + 15
        self.assertIsInstance(self.classifier, type(ScikitlearnRegressor(model=self.sklearn_model)))
        with self.assertRaises(TypeError):
            ScikitlearnDecisionTreeRegressor(model='sklearn_model')

    def test_predict(self):
        if False:
            while True:
                i = 10
        y_predicted = self.classifier.predict(self.x_test_diabetes[:4])
        y_expected = np.asarray([69.0, 81.0, 68.0, 68.0])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=1)

    def test_save(self):
        if False:
            i = 10
            return i + 15
        self.classifier.save(filename='test.file', path=None)
        self.classifier.save(filename='test.file', path='./')

    def test_clone_for_refitting(self):
        if False:
            while True:
                i = 10
        _ = self.classifier.clone_for_refitting()
if __name__ == '__main__':
    unittest.main()