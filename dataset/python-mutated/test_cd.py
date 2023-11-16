from __future__ import division
from __future__ import print_function
import os
import sys
import unittest
from numpy.testing import assert_equal
from numpy.testing import assert_raises
from sklearn.base import clone
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('__file__'), '..')))
from pyod.models.cd import CD
from pyod.utils.data import generate_data

class TestCD(unittest.TestCase):
    """
    Notes: GAN may yield unstable results, so the test is design for running
    models only, without any performance check.
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.n_train = 1000
        self.n_test = 200
        self.n_features = 2
        self.contamination = 0.1
        (self.X_train, self.X_test, self.y_train, self.y_test) = generate_data(n_train=self.n_train, n_test=self.n_test, n_features=self.n_features, contamination=self.contamination, random_state=42)
        self.clf = CD(contamination=self.contamination)
        self.clf.fit(self.X_train, self.y_train)

    def test_parameters(self):
        if False:
            print('Hello World!')
        assert hasattr(self.clf, 'decision_scores_') and self.clf.decision_scores_ is not None
        assert hasattr(self.clf, 'labels_') and self.clf.labels_ is not None
        assert hasattr(self.clf, 'threshold_') and self.clf.threshold_ is not None

    def test_train_scores(self):
        if False:
            for i in range(10):
                print('nop')
        assert_equal(len(self.clf.decision_scores_), self.X_train.shape[0])

    def test_prediction_scores(self):
        if False:
            i = 10
            return i + 15
        pred_scores = self.clf.decision_function(self.X_test)
        assert_equal(pred_scores.shape[0], self.X_test.shape[0])

    def test_prediction_labels(self):
        if False:
            return 10
        pred_labels = self.clf.predict(self.X_test)
        assert_equal(pred_labels.shape, self.y_test.shape)

    def test_prediction_proba(self):
        if False:
            i = 10
            return i + 15
        pred_proba = self.clf.predict_proba(self.X_test)
        assert pred_proba.min() >= 0
        assert pred_proba.max() <= 1

    def test_prediction_proba_linear(self):
        if False:
            while True:
                i = 10
        pred_proba = self.clf.predict_proba(self.X_test, method='linear')
        assert pred_proba.min() >= 0
        assert pred_proba.max() <= 1

    def test_prediction_proba_unify(self):
        if False:
            return 10
        pred_proba = self.clf.predict_proba(self.X_test, method='unify')
        assert pred_proba.min() >= 0
        assert pred_proba.max() <= 1

    def test_prediction_proba_parameter(self):
        if False:
            while True:
                i = 10
        with assert_raises(ValueError):
            self.clf.predict_proba(self.X_test, method='something')

    def test_prediction_labels_confidence(self):
        if False:
            i = 10
            return i + 15
        (pred_labels, confidence) = self.clf.predict(self.X_test, return_confidence=True)
        assert_equal(pred_labels.shape, self.y_test.shape)
        assert_equal(confidence.shape, self.y_test.shape)
        assert confidence.min() >= 0
        assert confidence.max() <= 1

    def test_prediction_proba_linear_confidence(self):
        if False:
            print('Hello World!')
        (pred_proba, confidence) = self.clf.predict_proba(self.X_test, method='linear', return_confidence=True)
        assert pred_proba.min() >= 0
        assert pred_proba.max() <= 1
        assert_equal(confidence.shape, self.y_test.shape)
        assert confidence.min() >= 0
        assert confidence.max() <= 1

    def test_model_clone(self):
        if False:
            i = 10
            return i + 15
        clone_clf = clone(self.clf)

    def tearDown(self):
        if False:
            return 10
        pass
if __name__ == '__main__':
    unittest.main()