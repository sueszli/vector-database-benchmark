from __future__ import division
from __future__ import print_function
import os
import sys
import unittest
from os import path
from numpy.testing import assert_allclose
from numpy.testing import assert_array_less
from numpy.testing import assert_equal
from numpy.testing import assert_raises
from scipy.io import loadmat
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyod.models.lscp import LSCP
from pyod.models.lof import LOF
from pyod.utils.data import generate_data

class TestLSCP(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        this_directory = path.abspath(path.dirname(__file__))
        mat_file = 'cardio.mat'
        try:
            mat = loadmat(path.join(*[this_directory, 'data', mat_file]))
        except TypeError:
            print('{data_file} does not exist. Use generated data'.format(data_file=mat_file))
            (X, y) = generate_data(train_only=True)
        except IOError:
            print('{data_file} does not exist. Use generated data'.format(data_file=mat_file))
            (X, y) = generate_data(train_only=True)
        else:
            X = mat['X']
            y = mat['y'].ravel()
            (X, y) = check_X_y(X, y)
        (self.X_train, self.X_test, self.y_train, self.y_test) = train_test_split(X, y, test_size=0.4, random_state=42)
        self.detector_list = [LOF(), LOF()]
        self.clf = LSCP(self.detector_list)
        self.clf.fit(self.X_train)
        self.roc_floor = 0.6

    def test_parameters(self):
        if False:
            print('Hello World!')
        assert hasattr(self.clf, 'decision_scores_') and self.clf.decision_scores_ is not None
        assert hasattr(self.clf, 'labels_') and self.clf.labels_ is not None
        assert hasattr(self.clf, 'threshold_') and self.clf.threshold_ is not None
        assert hasattr(self.clf, '_mu') and self.clf._mu is not None
        assert hasattr(self.clf, '_sigma') and self.clf._sigma is not None
        assert hasattr(self.clf, 'detector_list') and self.clf.detector_list is not None

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
        assert roc_auc_score(self.y_test, pred_scores) >= self.roc_floor

    def test_prediction_labels(self):
        if False:
            for i in range(10):
                print('nop')
        pred_labels = self.clf.predict(self.X_test)
        assert_equal(pred_labels.shape, self.y_test.shape)

    def test_prediction_proba(self):
        if False:
            while True:
                i = 10
        pred_proba = self.clf.predict_proba(self.X_test)
        assert pred_proba.min() >= 0
        assert pred_proba.max() <= 1

    def test_prediction_proba_linear(self):
        if False:
            return 10
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
            while True:
                i = 10
        with assert_raises(ValueError):
            self.clf.predict_proba(self.X_test, method='something')

    def test_prediction_labels_confidence(self):
        if False:
            print('Hello World!')
        (pred_labels, confidence) = self.clf.predict(self.X_test, return_confidence=True)
        assert_equal(pred_labels.shape, self.y_test.shape)
        assert_equal(confidence.shape, self.y_test.shape)
        assert confidence.min() >= 0
        assert confidence.max() <= 1

    def test_prediction_proba_linear_confidence(self):
        if False:
            i = 10
            return i + 15
        (pred_proba, confidence) = self.clf.predict_proba(self.X_test, method='linear', return_confidence=True)
        assert pred_proba.min() >= 0
        assert pred_proba.max() <= 1
        assert_equal(confidence.shape, self.y_test.shape)
        assert confidence.min() >= 0
        assert confidence.max() <= 1

    def test_fit_predict(self):
        if False:
            return 10
        pred_labels = self.clf.fit_predict(self.X_train)
        assert_equal(pred_labels.shape, self.y_train.shape)

    def test_fit_predict_score(self):
        if False:
            for i in range(10):
                print('nop')
        self.clf.fit_predict_score(self.X_test, self.y_test)
        self.clf.fit_predict_score(self.X_test, self.y_test, scoring='roc_auc_score')
        self.clf.fit_predict_score(self.X_test, self.y_test, scoring='prc_n_score')
        with assert_raises(NotImplementedError):
            self.clf.fit_predict_score(self.X_test, self.y_test, scoring='something')

    def test_predict_rank(self):
        if False:
            while True:
                i = 10
        pred_socres = self.clf.decision_function(self.X_test)
        pred_ranks = self.clf._predict_rank(self.X_test)
        assert_allclose(rankdata(pred_ranks), rankdata(pred_socres), atol=3)
        assert_array_less(pred_ranks, self.X_train.shape[0] + 1)
        assert_array_less(-0.1, pred_ranks)

    def test_predict_rank_normalized(self):
        if False:
            return 10
        pred_socres = self.clf.decision_function(self.X_test)
        pred_ranks = self.clf._predict_rank(self.X_test, normalized=True)
        assert_allclose(rankdata(pred_ranks), rankdata(pred_socres), atol=3)
        assert_array_less(pred_ranks, 1.01)
        assert_array_less(-0.1, pred_ranks)

    def test_model_clone(self):
        if False:
            i = 10
            return i + 15
        clone_clf = clone(self.clf)

    def tearDown(self):
        if False:
            print('Hello World!')
        pass
if __name__ == '__main__':
    unittest.main()