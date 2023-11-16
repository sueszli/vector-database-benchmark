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
from pyod.models.anogan import AnoGAN
from pyod.utils.data import generate_data

class TestAnoGAN(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.n_train = 500
        self.n_test = 200
        self.n_features = 2
        self.contamination = 0.1
        self.roc_floor = 0.8
        (self.X_train, self.X_test, self.y_train, self.y_test) = generate_data(n_train=self.n_train, n_test=self.n_test, n_features=self.n_features, contamination=self.contamination, random_state=42)
        self.clf = AnoGAN(epochs=3, contamination=self.contamination)
if __name__ == '__main__':
    unittest.main()