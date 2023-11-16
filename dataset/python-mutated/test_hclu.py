from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
import numpy as np
import GPy
from art.attacks.evasion.hclu import HighConfidenceLowUncertainty
from art.estimators.classification.GPy import GPyGaussianProcessClassifier
from tests.utils import TestBase
from tests.attacks.utils import backend_test_classifier_type_check_fail
logger = logging.getLogger(__name__)

class TestHCLU(TestBase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        super().setUpClass()
        cls.x_train = cls.x_train_iris[0:35]
        cls.y_train = cls.y_train_iris[0:35, 1]
        cls.x_test = cls.x_train
        cls.y_test = cls.y_train

    def test_GPy(self):
        if False:
            i = 10
            return i + 15
        x_test_original = self.x_test.copy()
        gpkern = GPy.kern.RBF(np.shape(self.x_train)[1])
        m = GPy.models.GPClassification(self.x_train, self.y_train.reshape(-1, 1), kernel=gpkern)
        m.inference_method = GPy.inference.latent_function_inference.laplace.Laplace()
        m.optimize(messages=True, optimizer='lbfgs')
        m_art = GPyGaussianProcessClassifier(m)
        clean_acc = np.mean(np.argmin(m_art.predict(self.x_test), axis=1) == self.y_test)
        attack = HighConfidenceLowUncertainty(m_art, conf=0.9, min_val=-0.0, max_val=1.0, verbose=False)
        adv = attack.generate(self.x_test)
        adv_acc = np.mean(np.argmin(m_art.predict(adv), axis=1) == self.y_test)
        unc_f = m_art.predict_uncertainty(adv)
        self.assertGreater(clean_acc, adv_acc)
        attack = HighConfidenceLowUncertainty(m_art, unc_increase=0.9, conf=0.9, min_val=0.0, max_val=1.0, verbose=False)
        adv = attack.generate(self.x_test)
        adv_acc = np.mean(np.argmin(m_art.predict(adv), axis=1) == self.y_test)
        unc_o = m_art.predict_uncertainty(adv)
        self.assertGreater(clean_acc, adv_acc)
        self.assertGreater(np.mean(unc_f > unc_o), 0.6)
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test))), 0.0, delta=1e-05)

    def test_check_params(self):
        if False:
            return 10
        gpkern = GPy.kern.RBF(np.shape(self.x_train)[1])
        m = GPy.models.GPClassification(self.x_train, self.y_train.reshape(-1, 1), kernel=gpkern)
        m_art = GPyGaussianProcessClassifier(m)
        with self.assertRaises(ValueError):
            _ = HighConfidenceLowUncertainty(m_art, conf=0.1, unc_increase=100.0, min_val=0.0, max_val=1.0, verbose=False)
        with self.assertRaises(ValueError):
            _ = HighConfidenceLowUncertainty(m_art, conf=0.75, unc_increase=-100.0, min_val=0.0, max_val=1.0, verbose=False)
        with self.assertRaises(ValueError):
            _ = HighConfidenceLowUncertainty(m_art, conf=0.75, unc_increase=100.0, min_val=1.0, max_val=0.0, verbose=False)
        with self.assertRaises(ValueError):
            _ = HighConfidenceLowUncertainty(m_art, conf=0.75, unc_increase=100.0, min_val=0.0, max_val=1.0, verbose='False')

    def test_classifier_type_check_fail(self):
        if False:
            i = 10
            return i + 15
        backend_test_classifier_type_check_fail(HighConfidenceLowUncertainty, [GPyGaussianProcessClassifier])
if __name__ == '__main__':
    unittest.main()