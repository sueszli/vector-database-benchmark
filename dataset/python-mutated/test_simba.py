from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
import numpy as np
from art.attacks.evasion.simba import SimBA
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import get_labels_np_array
from tests.utils import TestBase
from tests.utils import get_image_classifier_tf, get_image_classifier_kr, get_image_classifier_pt
from tests.attacks.utils import backend_test_classifier_type_check_fail
logger = logging.getLogger(__name__)

class TestSimBA(TestBase):
    """
    A unittest class for testing the Simple Black-box Adversarial Attacks (SimBA).

    This module tests SimBA.

    | Paper link: https://arxiv.org/abs/1905.07121
    """

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        super().setUpClass()
        cls.n_test = 2
        cls.x_test_mnist = cls.x_test_mnist[0:cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0:cls.n_test]

    def test_5_keras_mnist(self):
        if False:
            while True:
                i = 10
        '\n        Test with the KerasClassifier. (Untargeted Attack)\n        :return:\n        '
        classifier = get_image_classifier_kr()
        self._test_attack(classifier, self.x_test_mnist, self.y_test_mnist, False)

    def test_2_tensorflow_mnist(self):
        if False:
            print('Hello World!')
        '\n        Test with the TensorFlowClassifier. (Untargeted Attack)\n        :return:\n        '
        (classifier, sess) = get_image_classifier_tf()
        self._test_attack(classifier, self.x_test_mnist, self.y_test_mnist, False)

    def test_3_pytorch_mnist(self):
        if False:
            i = 10
            return i + 15
        '\n        Test with the PyTorchClassifier. (Untargeted Attack)\n        :return:\n        '
        x_test = np.reshape(self.x_test_mnist, (self.x_test_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        classifier = get_image_classifier_pt()
        self._test_attack(classifier, x_test, self.y_test_mnist, False)

    def test_6_keras_mnist_targeted(self):
        if False:
            return 10
        '\n        Test with the KerasClassifier. (Targeted Attack)\n        :return:\n        '
        classifier = get_image_classifier_kr()
        self._test_attack(classifier, self.x_test_mnist, self.y_test_mnist, True)

    def test_2_tensorflow_mnist_targeted(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test with the TensorFlowClassifier. (Targeted Attack)\n        :return:\n        '
        (classifier, sess) = get_image_classifier_tf()
        self._test_attack(classifier, self.x_test_mnist, self.y_test_mnist, True)

    def test_4_pytorch_mnist_targeted(self):
        if False:
            print('Hello World!')
        '\n        Test with the PyTorchClassifier. (Targeted Attack)\n        :return:\n        '
        x_test = np.reshape(self.x_test_mnist, (self.x_test_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        classifier = get_image_classifier_pt()
        self._test_attack(classifier, x_test, self.y_test_mnist, True)

    def _test_attack(self, classifier, x_test, y_test, targeted):
        if False:
            print('Hello World!')
        '\n        Test with SimBA\n        :return:\n        '
        x_test_original = x_test.copy()
        if targeted:
            y_target = np.zeros(10)
            y_target[8] = 1.0
        df = SimBA(classifier, attack='dct', targeted=targeted)
        x_i = x_test_original[0][None, ...]
        if targeted:
            x_test_adv = df.generate(x_i, y=y_target.reshape(1, 10))
        else:
            x_test_adv = df.generate(x_i)
        for i in range(1, len(x_test_original)):
            x_i = x_test_original[i][None, ...]
            if targeted:
                tmp_x_test_adv = df.generate(x_i, y=y_target.reshape(1, 10))
                x_test_adv = np.concatenate([x_test_adv, tmp_x_test_adv])
            else:
                tmp_x_test_adv = df.generate(x_i)
                x_test_adv = np.concatenate([x_test_adv, tmp_x_test_adv])
        self.assertFalse((x_test == x_test_adv).all())
        self.assertFalse((0.0 == x_test_adv).all())
        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((y_test == y_pred).all())
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=1e-05)
        df_px = SimBA(classifier, attack='px', targeted=targeted)
        x_i = x_test_original[0][None, ...]
        if targeted:
            x_test_adv = df_px.generate(x_i, y=y_target.reshape(1, 10))
        else:
            x_test_adv = df_px.generate(x_i)
        for i in range(1, len(x_test_original)):
            x_i = x_test_original[i][None, ...]
            if targeted:
                tmp_x_test_adv = df_px.generate(x_i, y=y_target.reshape(1, 10))
                x_test_adv = np.concatenate([x_test_adv, tmp_x_test_adv])
            else:
                tmp_x_test_adv = df_px.generate(x_i)
                x_test_adv = np.concatenate([x_test_adv, tmp_x_test_adv])
        self.assertFalse((x_test == x_test_adv).all())
        self.assertFalse((0.0 == x_test_adv).all())
        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((y_test == y_pred).all())
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=1e-05)
        df_px = SimBA(classifier, attack='px', targeted=targeted, order='diag')
        x_i = x_test_original[0][None, ...]
        if targeted:
            x_test_adv = df_px.generate(x_i, y=y_target.reshape(1, 10))
        else:
            x_test_adv = df_px.generate(x_i)
        for i in range(1, len(x_test_original)):
            x_i = x_test_original[i][None, ...]
            if targeted:
                tmp_x_test_adv = df_px.generate(x_i, y=y_target.reshape(1, 10))
                x_test_adv = np.concatenate([x_test_adv, tmp_x_test_adv])
            else:
                tmp_x_test_adv = df_px.generate(x_i)
                x_test_adv = np.concatenate([x_test_adv, tmp_x_test_adv])
        self.assertFalse((x_test == x_test_adv).all())
        self.assertFalse((0.0 == x_test_adv).all())
        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((y_test == y_pred).all())
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=1e-05)

    def test_check_params(self):
        if False:
            for i in range(10):
                print('nop')
        ptc = get_image_classifier_pt(from_logits=True)
        with self.assertRaises(ValueError):
            _ = SimBA(ptc, max_iter=1.0)
        with self.assertRaises(ValueError):
            _ = SimBA(ptc, max_iter=-1)
        with self.assertRaises(ValueError):
            _ = SimBA(ptc, epsilon=-1)
        with self.assertRaises(ValueError):
            _ = SimBA(ptc, batch_size=2)
        with self.assertRaises(ValueError):
            _ = SimBA(ptc, stride=1.0)
        with self.assertRaises(ValueError):
            _ = SimBA(ptc, stride=-1)
        with self.assertRaises(ValueError):
            _ = SimBA(ptc, freq_dim=1.0)
        with self.assertRaises(ValueError):
            _ = SimBA(ptc, freq_dim=-1)
        with self.assertRaises(ValueError):
            _ = SimBA(ptc, order='test')
        with self.assertRaises(ValueError):
            _ = SimBA(ptc, attack='test')
        with self.assertRaises(ValueError):
            _ = SimBA(ptc, targeted='test')

    def test_1_classifier_type_check_fail(self):
        if False:
            i = 10
            return i + 15
        backend_test_classifier_type_check_fail(SimBA, (BaseEstimator, ClassifierMixin, NeuralNetworkMixin))
if __name__ == '__main__':
    unittest.main()