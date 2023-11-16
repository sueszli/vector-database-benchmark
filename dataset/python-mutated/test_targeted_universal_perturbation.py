from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
import numpy as np
from art.attacks.evasion.targeted_universal_perturbation import TargetedUniversalPerturbation
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.estimator import BaseEstimator
from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import TestBase, get_image_classifier_kr, get_image_classifier_pt, get_image_classifier_tf
logger = logging.getLogger(__name__)

class TestTargetedUniversalPerturbation(TestBase):
    """
    A unittest class for testing the TargetedUniversalPerturbation attack.

    This module tests the Targeted Universal Perturbation.

    | Paper link: https://arxiv.org/abs/1911.06502)
    """

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        super().setUpClass()
        cls.n_train = 500
        cls.n_test = 10
        cls.x_train_mnist = cls.x_train_mnist[0:cls.n_train]
        cls.y_train_mnist = cls.y_train_mnist[0:cls.n_train]
        cls.x_test_mnist = cls.x_test_mnist[0:cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0:cls.n_test]

    def test_2_tensorflow_mnist(self):
        if False:
            i = 10
            return i + 15
        '\n        First test with the TensorFlowClassifier.\n        :return:\n        '
        x_test_original = self.x_test_mnist.copy()
        (tfc, sess) = get_image_classifier_tf()
        target = 0
        y_target = np.zeros([len(self.x_train_mnist), 10])
        for i in range(len(self.x_train_mnist)):
            y_target[i, target] = 1.0
        up = TargetedUniversalPerturbation(tfc, max_iter=1, attacker='fgsm', attacker_params={'eps': 0.3, 'targeted': True})
        x_train_adv = up.generate(self.x_train_mnist, y=y_target)
        self.assertTrue(up.fooling_rate >= 0.2 or not up.converged)
        x_test_adv = self.x_test_mnist + up.noise
        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        train_y_pred = np.argmax(tfc.predict(x_train_adv), axis=1)
        test_y_pred = np.argmax(tfc.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_mnist, axis=1) == test_y_pred).all())
        self.assertFalse((np.argmax(self.y_train_mnist, axis=1) == train_y_pred).all())
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=1e-05)

    def test_4_keras_mnist(self):
        if False:
            print('Hello World!')
        '\n        Second test with the KerasClassifier.\n        :return:\n        '
        x_test_original = self.x_test_mnist.copy()
        krc = get_image_classifier_kr()
        target = 0
        y_target = np.zeros([len(self.x_train_mnist), 10])
        for i in range(len(self.x_train_mnist)):
            y_target[i, target] = 1.0
        up = TargetedUniversalPerturbation(krc, max_iter=1, attacker='fgsm', attacker_params={'eps': 0.3, 'targeted': True})
        x_train_adv = up.generate(self.x_train_mnist, y=y_target)
        self.assertTrue(up.fooling_rate >= 0.2 or not up.converged)
        x_test_adv = self.x_test_mnist + up.noise
        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        train_y_pred = np.argmax(krc.predict(x_train_adv), axis=1)
        test_y_pred = np.argmax(krc.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_mnist, axis=1) == test_y_pred).all())
        self.assertFalse((np.argmax(self.y_train_mnist, axis=1) == train_y_pred).all())
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=1e-05)

    def test_3_pytorch_mnist(self):
        if False:
            return 10
        '\n        Third test with the PyTorchClassifier.\n        :return:\n        '
        x_train_mnist = np.swapaxes(self.x_train_mnist, 1, 3).astype(np.float32)
        x_test_mnist = np.swapaxes(self.x_test_mnist, 1, 3).astype(np.float32)
        x_test_original = x_test_mnist.copy()
        ptc = get_image_classifier_pt()
        target = 0
        y_target = np.zeros([len(self.x_train_mnist), 10])
        for i in range(len(self.x_train_mnist)):
            y_target[i, target] = 1.0
        up = TargetedUniversalPerturbation(ptc, max_iter=1, attacker='fgsm', attacker_params={'eps': 0.3, 'targeted': True})
        x_train_mnist_adv = up.generate(x_train_mnist, y=y_target)
        self.assertTrue(up.fooling_rate >= 0.2 or not up.converged)
        x_test_mnist_adv = x_test_mnist + up.noise
        self.assertFalse((x_test_mnist == x_test_mnist_adv).all())
        train_y_pred = np.argmax(ptc.predict(x_train_mnist_adv), axis=1)
        test_y_pred = np.argmax(ptc.predict(x_test_mnist_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_mnist, axis=1) == test_y_pred).all())
        self.assertFalse((np.argmax(self.y_train_mnist, axis=1) == train_y_pred).all())
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test_mnist))), 0.0, delta=1e-05)

    def test_check_params(self):
        if False:
            for i in range(10):
                print('nop')
        ptc = get_image_classifier_pt(from_logits=True)
        with self.assertRaises(ValueError):
            _ = TargetedUniversalPerturbation(ptc, delta=-1)
        with self.assertRaises(ValueError):
            _ = TargetedUniversalPerturbation(ptc, max_iter=-1)
        with self.assertRaises(ValueError):
            _ = TargetedUniversalPerturbation(ptc, eps=-1)

    def test_1_classifier_type_check_fail(self):
        if False:
            while True:
                i = 10
        backend_test_classifier_type_check_fail(TargetedUniversalPerturbation, (BaseEstimator, ClassifierMixin))
if __name__ == '__main__':
    unittest.main()