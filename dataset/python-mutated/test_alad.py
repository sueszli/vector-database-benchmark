from __future__ import division
from __future__ import print_function
import os
import sys
import unittest
from numpy.testing import assert_equal
from numpy.testing import assert_raises
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyod.models.alad import ALAD
from pyod.utils.data import generate_data

class TestALAD(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.n_train = 500
        self.n_test = 200
        self.n_features = 2
        self.contamination = 0.1
        self.roc_floor = 0.8
        (self.X_train, self.X_test, self.y_train, self.y_test) = generate_data(n_train=self.n_train, n_test=self.n_test, n_features=self.n_features, contamination=self.contamination, random_state=42)
        self.clf = ALAD(epochs=100, latent_dim=2, learning_rate_disc=0.0001, learning_rate_gen=0.0001, dropout_rate=0.2, add_recon_loss=False, lambda_recon_loss=0.05, add_disc_zz_loss=True, dec_layers=[75, 100], enc_layers=[100, 75], disc_xx_layers=[100, 75], disc_zz_layers=[25, 25], disc_xz_layers=[100, 75], spectral_normalization=False, activation_hidden_disc='tanh', activation_hidden_gen='tanh', preprocessing=True, batch_size=200, contamination=self.contamination)
        self.clf.fit(self.X_train)

    def test_parameters(self):
        if False:
            return 10
        assert hasattr(self.clf, 'decision_scores_') and self.clf.decision_scores_ is not None
        assert hasattr(self.clf, 'labels_') and self.clf.labels_ is not None
        assert hasattr(self.clf, 'threshold_') and self.clf.threshold_ is not None
        assert hasattr(self.clf, '_mu') and self.clf._mu is not None
        assert hasattr(self.clf, '_sigma') and self.clf._sigma is not None

    def test_fit_predict(self):
        if False:
            return 10
        pred_labels = self.clf.fit_predict(self.X_train)
        assert_equal(pred_labels.shape, self.y_train.shape)

    def test_model_clone(self):
        if False:
            print('Hello World!')
        clone_clf = clone(self.clf)

    def tearDown(self):
        if False:
            print('Hello World!')
        pass
if __name__ == '__main__':
    unittest.main()