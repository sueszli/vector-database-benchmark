from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
import numpy as np
from art.attacks.evasion.universal_perturbation import UniversalPerturbation
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.classification.keras import KerasClassifier
from art.estimators.estimator import BaseEstimator
from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import TestBase, get_image_classifier_kr, get_image_classifier_pt, get_image_classifier_tf, get_tabular_classifier_kr, get_tabular_classifier_pt, get_tabular_classifier_tf
logger = logging.getLogger(__name__)

class TestUniversalPerturbation(TestBase):
    """
    A unittest class for testing the UniversalPerturbation attack.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        super().setUpClass()
        cls.n_train = 500
        cls.n_test = 10
        cls.x_train_mnist = cls.x_train_mnist[0:cls.n_train]
        cls.y_train_mnist = cls.y_train_mnist[0:cls.n_train]
        cls.x_test_mnist = cls.x_test_mnist[0:cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0:cls.n_test]

    def test_3_tensorflow_mnist(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        First test with the TensorFlowClassifier.\n        :return:\n        '
        x_test_original = self.x_test_mnist.copy()
        (tfc, sess) = get_image_classifier_tf()
        up = UniversalPerturbation(tfc, max_iter=1, attacker='newtonfool', attacker_params={'max_iter': 5, 'verbose': False}, verbose=False)
        x_train_adv = up.generate(self.x_train_mnist)
        self.assertTrue(up.fooling_rate >= 0.2 or not up.converged)
        x_test_adv = self.x_test_mnist + up.noise
        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        train_y_pred = np.argmax(tfc.predict(x_train_adv), axis=1)
        test_y_pred = np.argmax(tfc.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_mnist, axis=1) == test_y_pred).all())
        self.assertFalse((np.argmax(self.y_train_mnist, axis=1) == train_y_pred).all())
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=1e-05)

    def test_8_keras_mnist(self):
        if False:
            return 10
        '\n        Second test with the KerasClassifier.\n        :return:\n        '
        x_test_original = self.x_test_mnist.copy()
        krc = get_image_classifier_kr()
        up = UniversalPerturbation(krc, max_iter=1, attacker='ead', attacker_params={'max_iter': 2, 'targeted': False, 'verbose': False}, verbose=False)
        x_train_adv = up.generate(self.x_train_mnist)
        self.assertTrue(up.fooling_rate >= 0.2 or not up.converged)
        x_test_adv = self.x_test_mnist + up.noise
        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        train_y_pred = np.argmax(krc.predict(x_train_adv), axis=1)
        test_y_pred = np.argmax(krc.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_mnist, axis=1) == test_y_pred).all())
        self.assertFalse((np.argmax(self.y_train_mnist, axis=1) == train_y_pred).all())
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=1e-05)

    def test_5_pytorch_mnist(self):
        if False:
            return 10
        '\n        Third test with the PyTorchClassifier.\n        :return:\n        '
        x_train_mnist = np.swapaxes(self.x_train_mnist, 1, 3).astype(np.float32)
        x_test_mnist = np.swapaxes(self.x_test_mnist, 1, 3).astype(np.float32)
        x_test_original = x_test_mnist.copy()
        ptc = get_image_classifier_pt()
        up = UniversalPerturbation(ptc, max_iter=1, attacker='newtonfool', attacker_params={'max_iter': 5, 'verbose': False}, verbose=False)
        x_train_mnist_adv = up.generate(x_train_mnist)
        self.assertTrue(up.fooling_rate >= 0.2 or not up.converged)
        x_test_mnist_adv = x_test_mnist + up.noise
        self.assertFalse((x_test_mnist == x_test_mnist_adv).all())
        train_y_pred = np.argmax(ptc.predict(x_train_mnist_adv), axis=1)
        test_y_pred = np.argmax(ptc.predict(x_test_mnist_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_mnist, axis=1) == test_y_pred).all())
        self.assertFalse((np.argmax(self.y_train_mnist, axis=1) == train_y_pred).all())
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test_mnist))), 0.0, delta=1e-05)

    def test_6_keras_iris_clipped(self):
        if False:
            i = 10
            return i + 15
        classifier = get_tabular_classifier_kr()
        attack_params = {'max_iter': 1, 'attacker': 'newtonfool', 'attacker_params': {'max_iter': 5, 'verbose': False}}
        attack = UniversalPerturbation(classifier, verbose=False)
        attack.set_params(**attack_params)
        x_test_iris_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_iris_adv).all())
        self.assertTrue((x_test_iris_adv <= 1).all())
        self.assertTrue((x_test_iris_adv >= 0).all())
        preds_adv = np.argmax(classifier.predict(x_test_iris_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info('Accuracy on Iris with universal adversarial examples: %.2f%%', acc * 100)

    def test_7_keras_iris_unbounded(self):
        if False:
            print('Hello World!')
        classifier = get_tabular_classifier_kr()
        classifier = KerasClassifier(model=classifier._model, use_logits=False, channels_first=True)
        attack_params = {'max_iter': 1, 'attacker': 'newtonfool', 'attacker_params': {'max_iter': 5, 'verbose': False}}
        attack = UniversalPerturbation(classifier, verbose=False)
        attack.set_params(**attack_params)
        x_test_iris_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_iris_adv).all())
        preds_adv = np.argmax(classifier.predict(x_test_iris_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info('Accuracy on Iris with universal adversarial examples: %.2f%%', acc * 100)

    def test_2_tensorflow_iris(self):
        if False:
            print('Hello World!')
        (classifier, _) = get_tabular_classifier_tf()
        attack_params = {'max_iter': 1, 'attacker': 'ead', 'attacker_params': {'max_iter': 5, 'targeted': False, 'verbose': False}}
        attack = UniversalPerturbation(classifier, verbose=False)
        attack.set_params(**attack_params)
        x_test_iris_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_iris_adv).all())
        self.assertTrue((x_test_iris_adv <= 1).all())
        self.assertTrue((x_test_iris_adv >= 0).all())
        preds_adv = np.argmax(classifier.predict(x_test_iris_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info('Accuracy on Iris with universal adversarial examples: %.2f%%', acc * 100)

    def test_4_pytorch_iris(self):
        if False:
            print('Hello World!')
        classifier = get_tabular_classifier_pt()
        attack_params = {'max_iter': 1, 'attacker': 'ead', 'attacker_params': {'max_iter': 5, 'targeted': False, 'verbose': False}}
        attack = UniversalPerturbation(classifier, verbose=False)
        attack.set_params(**attack_params)
        x_test_iris_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_iris_adv).all())
        self.assertTrue((x_test_iris_adv <= 1).all())
        self.assertTrue((x_test_iris_adv >= 0).all())
        preds_adv = np.argmax(classifier.predict(x_test_iris_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info('Accuracy on Iris with universal adversarial examples: %.2f%%', acc * 100)

    def test_check_params(self):
        if False:
            print('Hello World!')
        ptc = get_image_classifier_pt(from_logits=True)
        with self.assertRaises(ValueError):
            _ = UniversalPerturbation(ptc, delta=-1)
        with self.assertRaises(ValueError):
            _ = UniversalPerturbation(ptc, max_iter=-1)
        with self.assertRaises(ValueError):
            _ = UniversalPerturbation(ptc, eps=-1)
        with self.assertRaises(ValueError):
            _ = UniversalPerturbation(ptc, batch_size=-1)
        with self.assertRaises(ValueError):
            _ = UniversalPerturbation(ptc, verbose='False')

    def test_1_classifier_type_check_fail(self):
        if False:
            i = 10
            return i + 15
        backend_test_classifier_type_check_fail(UniversalPerturbation, [BaseEstimator, ClassifierMixin])
if __name__ == '__main__':
    unittest.main()