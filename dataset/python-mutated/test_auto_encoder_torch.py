from __future__ import division
from __future__ import print_function
import os
import sys
import unittest
import numpy as np
import torch
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises
from sklearn.metrics import roc_auc_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyod.models.auto_encoder_torch import AutoEncoder
from pyod.utils.data import generate_data
from pyod.models.auto_encoder_torch import PyODDataset

class TestPyODDataset(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.n_train = 3000
        self.n_test = 1000
        self.n_features = 200
        self.contamination = 0.1
        self.batch_size = 1000
        (self.X_train, self.X_test, self.y_train, self.y_test) = generate_data(n_train=self.n_train, n_test=self.n_test, n_features=self.n_features, contamination=self.contamination, random_state=42)

    def test_no_preprocessing(self):
        if False:
            return 10
        train_set = PyODDataset(X=self.X_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        for (data, data_idx) in train_loader:
            assert data.shape[0] == self.batch_size
            assert data.shape[1] == self.n_features

    def test_preprocessing(self):
        if False:
            for i in range(10):
                print('nop')
        (self.mean, self.std) = (np.mean(self.X_train, axis=0), np.std(self.X_train, axis=0))
        train_set = PyODDataset(X=self.X_train, mean=self.mean, std=self.std)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        for (data, data_idx) in train_loader:
            assert data.shape[0] == self.batch_size
            assert data.shape[1] == self.n_features
            assert_almost_equal(data.mean(), 0, decimal=1)

class TestAutoEncoder(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.n_train = 3000
        self.n_test = 1000
        self.n_features = 200
        self.contamination = 0.1
        self.roc_floor = 0.8
        (self.X_train, self.X_test, self.y_train, self.y_test) = generate_data(n_train=self.n_train, n_test=self.n_test, n_features=self.n_features, contamination=self.contamination, random_state=42)
        self.clf = AutoEncoder(epochs=3, contamination=self.contamination)
        self.clf.fit(self.X_train)

    def test_parameters(self):
        if False:
            print('Hello World!')
        assert hasattr(self.clf, 'decision_scores_') and self.clf.decision_scores_ is not None
        assert hasattr(self.clf, 'labels_') and self.clf.labels_ is not None
        assert hasattr(self.clf, 'threshold_') and self.clf.threshold_ is not None
        assert hasattr(self.clf, '_mu') and self.clf._mu is not None
        assert hasattr(self.clf, '_sigma') and self.clf._sigma is not None
        assert hasattr(self.clf, 'model') and self.clf.model is not None

    def test_train_scores(self):
        if False:
            print('Hello World!')
        assert_equal(len(self.clf.decision_scores_), self.X_train.shape[0])

    def test_prediction_scores(self):
        if False:
            while True:
                i = 10
        pred_scores = self.clf.decision_function(self.X_test)
        assert_equal(pred_scores.shape[0], self.X_test.shape[0])
        assert roc_auc_score(self.y_test, pred_scores) >= self.roc_floor

    def test_prediction_labels(self):
        if False:
            i = 10
            return i + 15
        pred_labels = self.clf.predict(self.X_test)
        assert_equal(pred_labels.shape, self.y_test.shape)

    def test_prediction_proba(self):
        if False:
            for i in range(10):
                print('nop')
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
            return 10
        (pred_labels, confidence) = self.clf.predict(self.X_test, return_confidence=True)
        assert_equal(pred_labels.shape, self.y_test.shape)
        assert_equal(confidence.shape, self.y_test.shape)
        assert confidence.min() >= 0
        assert confidence.max() <= 1

    def test_prediction_proba_linear_confidence(self):
        if False:
            while True:
                i = 10
        (pred_proba, confidence) = self.clf.predict_proba(self.X_test, method='linear', return_confidence=True)
        assert pred_proba.min() >= 0
        assert pred_proba.max() <= 1
        assert_equal(confidence.shape, self.y_test.shape)
        assert confidence.min() >= 0
        assert confidence.max() <= 1

    def test_fit_predict(self):
        if False:
            i = 10
            return i + 15
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

    def test_model_clone(self):
        if False:
            print('Hello World!')
        pass

    def tearDown(self):
        if False:
            return 10
        pass
if __name__ == '__main__':
    unittest.main()