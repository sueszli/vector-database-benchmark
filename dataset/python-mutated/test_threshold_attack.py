"""
This module tests the Threshold Attack.

| Paper link:
    https://arxiv.org/abs/1906.06026
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
import numpy as np
from art.attacks.evasion.pixel_threshold import ThresholdAttack
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import get_labels_np_array
from tests.utils import TestBase
from tests.utils import get_image_classifier_tf, get_image_classifier_kr, get_image_classifier_pt
from tests.attacks.utils import backend_test_classifier_type_check_fail
logger = logging.getLogger(__name__)

class TestThresholdAttack(TestBase):
    """
    A unittest class for testing the Threshold Attack.

    This module tests the Threshold Attack.

    | Paper link:
        https://arxiv.org/abs/1906.06026
    """

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        super().setUpClass()
        cls.n_test = 2
        cls.x_test_mnist = cls.x_test_mnist[0:cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0:cls.n_test]

    def test_6_keras_mnist(self):
        if False:
            while True:
                i = 10
        '\n        Test with the KerasClassifier. (Untargeted Attack)\n        :return:\n        '
        classifier = get_image_classifier_kr()
        self._test_attack(classifier, self.x_test_mnist, self.y_test_mnist, False)

    def test_2_tensorflow_mnist(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test with the TensorFlowClassifier. (Untargeted Attack)\n        :return:\n        '
        (classifier, sess) = get_image_classifier_tf()
        self._test_attack(classifier, self.x_test_mnist, self.y_test_mnist, False)

    def test_4_pytorch_mnist(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test with the PyTorchClassifier. (Untargeted Attack)\n        :return:\n        '
        x_test = np.reshape(self.x_test_mnist, (self.x_test_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        classifier = get_image_classifier_pt()
        self._test_attack(classifier, x_test, self.y_test_mnist, False)

    def test_7_keras_mnist_targeted(self):
        if False:
            return 10
        '\n        Test with the KerasClassifier. (Targeted Attack)\n        :return:\n        '
        classifier = get_image_classifier_kr()
        self._test_attack(classifier, self.x_test_mnist, self.y_test_mnist, True)

    def test_3_tensorflow_mnist_targeted(self):
        if False:
            return 10
        '\n        Test with the TensorFlowClassifier. (Targeted Attack)\n        :return:\n        '
        (classifier, sess) = get_image_classifier_tf()
        self._test_attack(classifier, self.x_test_mnist, self.y_test_mnist, True)

    def test_5_pytorch_mnist_targeted(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test with the PyTorchClassifier. (Targeted Attack)\n        :return:\n        '
        x_test = np.reshape(self.x_test_mnist, (self.x_test_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        classifier = get_image_classifier_pt()
        self._test_attack(classifier, x_test, self.y_test_mnist, True)

    def _test_attack(self, classifier, x_test, y_test, targeted):
        if False:
            print('Hello World!')
        '\n        Test with the Threshold Attack\n        :return:\n        '
        x_test_original = x_test.copy()
        if targeted:
            class_y_test = np.argmax(y_test, axis=1)
            nb_classes = np.unique(class_y_test).shape[0]
            np.random.seed(seed=487)
            targets = np.random.randint(nb_classes, size=self.n_test)
            for i in range(self.n_test):
                if class_y_test[i] == targets[i]:
                    targets[i] -= 1
        else:
            targets = y_test
        for es in [1]:
            df = ThresholdAttack(classifier, th=128, es=es, max_iter=10, targeted=targeted, verbose=False)
            x_test_adv = df.generate(x_test_original, targets)
            np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, x_test, x_test_adv)
            self.assertFalse((0.0 == x_test_adv).all())
            y_pred = get_labels_np_array(classifier.predict(x_test_adv))
            accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.n_test
            logger.info('Accuracy on adversarial examples: %.2f%%', accuracy * 100)
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=1e-05)

    def test_1_classifier_type_check_fail(self):
        if False:
            for i in range(10):
                print('nop')
        backend_test_classifier_type_check_fail(ThresholdAttack, [BaseEstimator, NeuralNetworkMixin, ClassifierMixin])
if __name__ == '__main__':
    unittest.main()