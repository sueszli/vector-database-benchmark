from __future__ import division
from __future__ import print_function
import os
import sys
import unittest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_less
from numpy.testing import assert_equal
from numpy.testing import assert_raises
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyod.models.mad import MAD
from pyod.utils.data import generate_data

class TestMAD(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.n_train = 100
        self.n_test = 50
        self.contamination = 0.1
        self.roc_floor = 0.8
        (self.X_train, self.X_test, self.y_train, self.y_test) = generate_data(n_train=self.n_train, n_test=self.n_test, n_features=1, contamination=self.contamination, random_state=42)
        self.clf = MAD()
        self.clf.fit(self.X_train)
        (self.X_train_nan, self.X_test_nan, self.y_train_nan, self.y_test_nan) = generate_data(n_train=self.n_train, n_test=self.n_test, n_features=1, contamination=self.contamination, random_state=42, n_nan=1)
        self.clf_nan = MAD()
        self.clf_nan.fit(self.X_train_nan)
        (self.X_train_inf, self.X_test_inf, self.y_train_inf, self.y_test_inf) = generate_data(n_train=self.n_train, n_test=self.n_test, n_features=1, contamination=self.contamination, random_state=42, n_inf=1)
        self.clf_inf = MAD()
        self.clf_inf.fit(self.X_train_inf)

    def test_parameters(self):
        if False:
            print('Hello World!')
        assert hasattr(self.clf, 'decision_scores_') and self.clf.decision_scores_ is not None
        assert hasattr(self.clf, 'labels_') and self.clf.labels_ is not None
        assert hasattr(self.clf, 'threshold_') and self.clf.threshold_ is not None
        with assert_raises(TypeError):
            MAD(threshold='str')

    def test_train_scores(self):
        if False:
            return 10
        assert_equal(len(self.clf.decision_scores_), self.X_train.shape[0])

    def test_prediction_scores(self):
        if False:
            i = 10
            return i + 15
        pred_scores = self.clf.decision_function(self.X_test)
        assert_equal(pred_scores.shape[0], self.X_test.shape[0])
        assert roc_auc_score(self.y_test, pred_scores) >= self.roc_floor

    def test_prediction_labels(self):
        if False:
            while True:
                i = 10
        pred_labels = self.clf.predict(self.X_test)
        assert_equal(pred_labels.shape, self.y_test.shape)

    def test_prediction_proba(self):
        if False:
            return 10
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
            i = 10
            return i + 15
        pred_proba = self.clf.predict_proba(self.X_test, method='unify')
        assert pred_proba.min() >= 0
        assert pred_proba.max() <= 1

    def test_prediction_proba_parameter(self):
        if False:
            for i in range(10):
                print('nop')
        with assert_raises(ValueError):
            self.clf.predict_proba(self.X_test, method='something')

    def test_prediction_labels_confidence(self):
        if False:
            while True:
                i = 10
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

    def test_fit_predict(self):
        if False:
            while True:
                i = 10
        pred_labels = self.clf.fit_predict(self.X_train)
        assert_equal(pred_labels.shape, self.y_train.shape)

    def test_fit_predict_with_nan(self):
        if False:
            while True:
                i = 10
        pred_labels = self.clf_nan.fit_predict(self.X_train_nan)
        assert_equal(pred_labels.shape, self.y_train_nan.shape)

    def test_fit_predict_with_inf(self):
        if False:
            print('Hello World!')
        pred_labels = self.clf_inf.fit_predict(self.X_train_inf)
        assert_equal(pred_labels.shape, self.y_train_inf.shape)

    def test_fit_predict_score(self):
        if False:
            return 10
        self.clf.fit_predict_score(self.X_test, self.y_test)
        self.clf.fit_predict_score(self.X_test, self.y_test, scoring='roc_auc_score')
        self.clf.fit_predict_score(self.X_test, self.y_test, scoring='prc_n_score')
        with assert_raises(NotImplementedError):
            self.clf.fit_predict_score(self.X_test, self.y_test, scoring='something')

    def test_predict_rank(self):
        if False:
            i = 10
            return i + 15
        pred_scores = self.clf.decision_function(self.X_test)
        pred_ranks = self.clf._predict_rank(self.X_test)
        print(pred_ranks)
        assert_allclose(rankdata(pred_ranks), rankdata(pred_scores), atol=2)
        assert_array_less(pred_ranks, self.X_train.shape[0] + 1)
        assert_array_less(-0.1, pred_ranks)

    def test_predict_rank_with_nan(self):
        if False:
            for i in range(10):
                print('nop')
        pred_scores = self.clf_nan.decision_function(self.X_test_nan)
        pred_ranks = self.clf_nan._predict_rank(self.X_test_nan)
        print(pred_ranks)
        assert_allclose(rankdata(pred_ranks), rankdata(pred_scores), atol=2)
        assert_array_less(pred_ranks, self.X_train_nan.shape[0] + 1)
        assert_array_less(-0.1, pred_ranks)

    def test_predict_rank_with_inf(self):
        if False:
            while True:
                i = 10
        pred_scores = self.clf_inf.decision_function(self.X_test_inf)
        pred_ranks = self.clf_inf._predict_rank(self.X_test_inf)
        print(pred_ranks)
        assert_allclose(rankdata(pred_ranks), rankdata(pred_scores), atol=2)
        assert_array_less(pred_ranks, self.X_train_inf.shape[0] + 1)
        assert_array_less(-0.1, pred_ranks)

    def test_predict_rank_normalized(self):
        if False:
            return 10
        pred_scores = self.clf.decision_function(self.X_test)
        pred_ranks = self.clf._predict_rank(self.X_test, normalized=True)
        assert_allclose(rankdata(pred_ranks), rankdata(pred_scores), atol=2)
        assert_array_less(pred_ranks, 1.01)
        assert_array_less(-0.1, pred_ranks)

    def test_predict_rank_normalized_with_nan(self):
        if False:
            while True:
                i = 10
        pred_scores = self.clf_nan.decision_function(self.X_test_nan)
        pred_ranks = self.clf_nan._predict_rank(self.X_test_nan, normalized=True)
        assert_allclose(rankdata(pred_ranks), rankdata(pred_scores), atol=2)
        assert_array_less(pred_ranks, 1.01)
        assert_array_less(-0.1, pred_ranks)

    def test_predict_rank_normalized_with_inf(self):
        if False:
            return 10
        pred_scores = self.clf_inf.decision_function(self.X_test_inf)
        pred_ranks = self.clf_inf._predict_rank(self.X_test_inf, normalized=True)
        assert_allclose(rankdata(pred_ranks), rankdata(pred_scores), atol=2)
        assert_array_less(pred_ranks, 1.01)
        assert_array_less(-0.1, pred_ranks)

    def test_check_univariate(self):
        if False:
            print('Hello World!')
        with assert_raises(ValueError):
            MAD().fit(X=[[0.0, 0.0], [0.0, 0.0]])
        with assert_raises(ValueError):
            MAD().decision_function(X=[[0.0, 0.0], [0.0, 0.0]])

    def test_detect_anomaly(self):
        if False:
            i = 10
            return i + 15
        X_test = [[10000]]
        score = self.clf.decision_function(X_test)
        anomaly = self.clf.predict(X_test)
        self.assertGreaterEqual(score[0], self.clf.threshold_)
        self.assertEqual(anomaly[0], 1)

    def test_detect_anomaly_with_nan(self):
        if False:
            print('Hello World!')
        X_test = [[10000]]
        score = self.clf_nan.decision_function(X_test)
        anomaly = self.clf_nan.predict(X_test)
        self.assertGreaterEqual(score[0], self.clf_nan.threshold_)
        self.assertEqual(anomaly[0], 1)

    def test_detect_anomaly_with_inf(self):
        if False:
            return 10
        X_test = [[10000]]
        score = self.clf_inf.decision_function(X_test)
        anomaly = self.clf_inf.predict(X_test)
        self.assertGreaterEqual(score[0], self.clf_inf.threshold_)
        self.assertEqual(anomaly[0], 1)

    def test_model_clone(self):
        if False:
            i = 10
            return i + 15
        clone_clf = clone(self.clf)

    def tearDown(self):
        if False:
            while True:
                i = 10
        pass
if __name__ == '__main__':
    unittest.main()