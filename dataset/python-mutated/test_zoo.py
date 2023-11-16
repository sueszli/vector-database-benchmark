from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
import keras.backend as k
import numpy as np
from art.attacks.evasion.zoo import ZooAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import random_targets
from tests.utils import TestBase, get_image_classifier_kr, get_image_classifier_pt
from tests.utils import get_image_classifier_tf, master_seed
from tests.attacks.utils import backend_test_classifier_type_check_fail
logger = logging.getLogger(__name__)

class TestZooAttack(TestBase):
    """
    A unittest class for testing the ZOO attack.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        master_seed(seed=1234)
        super().setUpClass()
        cls.n_train = 1
        cls.n_test = 1
        cls.x_train_mnist = cls.x_train_mnist[0:cls.n_train]
        cls.y_train_mnist = cls.y_train_mnist[0:cls.n_train]
        cls.x_test_mnist = cls.x_test_mnist[0:cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0:cls.n_test]

    def test_2_tensorflow_failure_attack(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test the corner case when attack fails.\n        :return:\n        '
        x_test_original = self.x_test_mnist.copy()
        (tfc, sess) = get_image_classifier_tf()
        zoo = ZooAttack(classifier=tfc, max_iter=0, binary_search_steps=0, learning_rate=0, verbose=False)
        x_test_mnist_adv = zoo.generate(self.x_test_mnist)
        self.assertLessEqual(np.amax(x_test_mnist_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_mnist_adv), 0.0)
        np.testing.assert_almost_equal(self.x_test_mnist, x_test_mnist_adv, 3)
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=1e-05)
        if sess is not None:
            sess.close()

    def test_3_tensorflow_mnist(self):
        if False:
            return 10
        '\n        First test with the TensorFlowClassifier.\n        :return:\n        '
        x_test_original = self.x_test_mnist.copy()
        (tfc, sess) = get_image_classifier_tf()
        zoo = ZooAttack(classifier=tfc, targeted=True, max_iter=30, binary_search_steps=8, batch_size=128, verbose=False)
        params = {'y': random_targets(self.y_test_mnist, tfc.nb_classes)}
        x_test_mnist_adv = zoo.generate(self.x_test_mnist, **params)
        self.assertFalse((self.x_test_mnist == x_test_mnist_adv).all())
        self.assertLessEqual(np.amax(x_test_mnist_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_mnist_adv), 0.0)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_mnist_adv), axis=1)
        logger.debug('ZOO target: %s', target)
        logger.debug('ZOO actual: %s', y_pred_adv)
        logger.info('ZOO success rate on MNIST: %.2f', sum(target == y_pred_adv) / float(len(target)))
        zoo = ZooAttack(classifier=tfc, targeted=False, max_iter=10, binary_search_steps=3, verbose=False)
        x_test_mnist_adv = zoo.generate(self.x_test_mnist)
        self.assertLessEqual(np.amax(x_test_mnist_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_mnist_adv), 0.0)
        y_pred = np.argmax(tfc.predict(self.x_test_mnist), axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_mnist_adv), axis=1)
        logger.debug('ZOO actual: %s', y_pred_adv)
        logger.info('ZOO success rate on MNIST: %.2f', sum(y_pred != y_pred_adv) / float(len(y_pred)))
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=1e-05)
        x_test_resized = zoo._resize_image(self.x_test_mnist, 64, 64)
        self.assertEqual(x_test_resized.shape, (1, 64, 64, 1))
        if sess is not None:
            sess.close()

    def test_5_keras_mnist(self):
        if False:
            print('Hello World!')
        '\n        Second test with the KerasClassifier.\n        :return:\n        '
        x_test_original = self.x_test_mnist.copy()
        krc = get_image_classifier_kr()
        zoo = ZooAttack(classifier=krc, targeted=False, batch_size=5, max_iter=10, binary_search_steps=3, verbose=False)
        params = {'y': random_targets(self.y_test_mnist, krc.nb_classes)}
        x_test_mnist_adv = zoo.generate(self.x_test_mnist, **params)
        self.assertLessEqual(np.amax(x_test_mnist_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_mnist_adv), 0.0)
        y_pred_adv = np.argmax(krc.predict(x_test_mnist_adv), axis=1)
        y_pred = np.argmax(krc.predict(self.x_test_mnist), axis=1)
        logger.debug('ZOO actual: %s', y_pred_adv)
        logger.info('ZOO success rate on MNIST: %.2f', sum(y_pred != y_pred_adv) / float(len(y_pred)))
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=1e-05)
        k.clear_session()

    def test_4_pytorch_mnist(self):
        if False:
            i = 10
            return i + 15
        '\n        Third test with the PyTorchClassifier.\n        :return:\n        '
        ptc = get_image_classifier_pt()
        x_test_mnist = np.swapaxes(self.x_test_mnist, 1, 3).astype(np.float32)
        x_test_original = x_test_mnist.copy()
        zoo = ZooAttack(classifier=ptc, targeted=False, learning_rate=0.01, max_iter=10, binary_search_steps=3, abort_early=False, use_resize=False, use_importance=False, verbose=False)
        x_test_mnist_adv = zoo.generate(x_test_mnist)
        self.assertLessEqual(np.amax(x_test_mnist_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_mnist_adv), 0.0)
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test_mnist))), 0.0, delta=1e-05)

    def test_check_params(self):
        if False:
            return 10
        ptc = get_image_classifier_pt(from_logits=True)
        with self.assertRaises(ValueError):
            _ = ZooAttack(ptc, binary_search_steps=1.0)
        with self.assertRaises(ValueError):
            _ = ZooAttack(ptc, binary_search_steps=-1)
        with self.assertRaises(ValueError):
            _ = ZooAttack(ptc, max_iter=1.0)
        with self.assertRaises(ValueError):
            _ = ZooAttack(ptc, max_iter=-1)
        with self.assertRaises(ValueError):
            _ = ZooAttack(ptc, nb_parallel=1.0)
        with self.assertRaises(ValueError):
            _ = ZooAttack(ptc, nb_parallel=-1)
        with self.assertRaises(ValueError):
            _ = ZooAttack(ptc, batch_size=1.0)
        with self.assertRaises(ValueError):
            _ = ZooAttack(ptc, batch_size=-1)
        with self.assertRaises(ValueError):
            _ = ZooAttack(ptc, verbose='true')

    def test_1_classifier_type_check_fail(self):
        if False:
            while True:
                i = 10
        backend_test_classifier_type_check_fail(ZooAttack, [BaseEstimator, ClassifierMixin])
if __name__ == '__main__':
    unittest.main()