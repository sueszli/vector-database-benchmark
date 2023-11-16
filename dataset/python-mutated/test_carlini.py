from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
import numpy as np
from art.attacks.evasion.carlini import CarliniL0Method, CarliniL2Method, CarliniLInfMethod
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassGradientsMixin
from art.utils import random_targets
from tests.utils import TestBase, master_seed
from tests.utils import get_image_classifier_tf
from tests.utils import get_image_classifier_pt
from tests.attacks.utils import backend_test_classifier_type_check_fail
logger = logging.getLogger(__name__)

class TestCarlini(TestBase):
    """
    A unittest class for testing the Carlini L2 attack.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        super().setUpClass()
        cls.n_train = 10
        cls.n_test = 10
        cls.x_train_mnist = cls.x_train_mnist[0:cls.n_train]
        cls.y_train_mnist = cls.y_train_mnist[0:cls.n_train]
        cls.x_test_mnist = cls.x_test_mnist[0:cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0:cls.n_test]

    def setUp(self):
        if False:
            print('Hello World!')
        master_seed(seed=1234)
        super().setUp()

    def test_tensorflow_failure_attack_L2(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test the corner case when attack is failed.\n        :return:\n        '
        x_test_original = self.x_test_mnist.copy()
        (tfc, sess) = get_image_classifier_tf(from_logits=True)
        cl2m = CarliniL2Method(classifier=tfc, targeted=True, max_iter=0, binary_search_steps=0, learning_rate=0, initial_const=1, verbose=False)
        params = {'y': random_targets(self.y_test_mnist, tfc.nb_classes)}
        x_test_adv = cl2m.generate(self.x_test_mnist, **params)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        np.testing.assert_array_almost_equal(self.x_test_mnist, x_test_adv, decimal=3)
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=1e-05)
        if sess is not None:
            sess.close()

    def test_tensorflow_mnist_L2(self):
        if False:
            print('Hello World!')
        '\n        First test with the TensorFlowClassifier.\n        :return:\n        '
        x_test_original = self.x_test_mnist.copy()
        (tfc, sess) = get_image_classifier_tf(from_logits=True)
        cl2m = CarliniL2Method(classifier=tfc, targeted=True, max_iter=10, verbose=False)
        params = {'y': random_targets(self.y_test_mnist, tfc.nb_classes)}
        x_test_adv = cl2m.generate(self.x_test_mnist, **params)
        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('CW2 Target: %s', target)
        logger.debug('CW2 Actual: %s', y_pred_adv)
        logger.info('CW2 Success Rate: %.2f', np.sum(target == y_pred_adv) / float(len(target)))
        self.assertTrue((target == y_pred_adv).any())
        cl2m = CarliniL2Method(classifier=tfc, targeted=False, max_iter=10, batch_size=1, verbose=False)
        x_test_adv = cl2m.generate(self.x_test_mnist)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('CW2 Target: %s', target)
        logger.debug('CW2 Actual: %s', y_pred_adv)
        logger.info('CW2 Success Rate: %.2f', np.sum(target == y_pred_adv) / float(len(target)))
        self.assertTrue((target != y_pred_adv).any())
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=1e-05)
        if sess is not None:
            sess.close()

    def test_classifier_type_check_fail_L2(self):
        if False:
            i = 10
            return i + 15
        backend_test_classifier_type_check_fail(CarliniL2Method, [BaseEstimator, ClassGradientsMixin])

    def test_check_params_L2(self):
        if False:
            print('Hello World!')
        ptc = get_image_classifier_pt(from_logits=True)
        with self.assertRaises(ValueError):
            _ = CarliniL2Method(ptc, binary_search_steps='1.0')
        with self.assertRaises(ValueError):
            _ = CarliniL2Method(ptc, binary_search_steps=-1)
        with self.assertRaises(ValueError):
            _ = CarliniL2Method(ptc, max_iter='1.0')
        with self.assertRaises(ValueError):
            _ = CarliniL2Method(ptc, max_iter=-1)
        with self.assertRaises(ValueError):
            _ = CarliniL2Method(ptc, max_halving='1.0')
        with self.assertRaises(ValueError):
            _ = CarliniL2Method(ptc, max_halving=-1)
        with self.assertRaises(ValueError):
            _ = CarliniL2Method(ptc, max_doubling='1.0')
        with self.assertRaises(ValueError):
            _ = CarliniL2Method(ptc, max_doubling=-1)
        with self.assertRaises(ValueError):
            _ = CarliniL2Method(ptc, batch_size='1.0')
        with self.assertRaises(ValueError):
            _ = CarliniL2Method(ptc, batch_size=-1)
    '\n    A unittest class for testing the Carlini LInf attack.\n    '

    def test_tensorflow_failure_attack_LInf(self):
        if False:
            print('Hello World!')
        '\n        Test the corner case when attack is failed.\n        :return:\n        '
        (tfc, sess) = get_image_classifier_tf(from_logits=True)
        clinfm = CarliniLInfMethod(classifier=tfc, targeted=True, max_iter=0, learning_rate=0, verbose=False)
        params = {'y': random_targets(self.y_test_mnist, tfc.nb_classes)}
        x_test_adv = clinfm.generate(self.x_test_mnist, **params)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        self.assertTrue(np.allclose(self.x_test_mnist, x_test_adv, atol=0.001))
        if sess is not None:
            sess.close()

    def test_tensorflow_mnist_LInf(self):
        if False:
            print('Hello World!')
        '\n        First test with the TensorFlowClassifier.\n        :return:\n        '
        (tfc, sess) = get_image_classifier_tf(from_logits=True)
        clinfm = CarliniLInfMethod(classifier=tfc, targeted=True, max_iter=10, initial_const=1, largest_const=1.1, verbose=False)
        params = {'y': random_targets(self.y_test_mnist, tfc.nb_classes)}
        x_test_adv = clinfm.generate(self.x_test_mnist, **params)
        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('CW0 Target: %s', target)
        logger.debug('CW0 Actual: %s', y_pred_adv)
        logger.info('CW0 Success Rate: %.2f', np.sum(target == y_pred_adv) / float(len(target)))
        self.assertTrue((target == y_pred_adv).any())
        clinfm = CarliniLInfMethod(classifier=tfc, targeted=False, max_iter=10, initial_const=1, largest_const=1.1, verbose=False)
        x_test_adv = clinfm.generate(self.x_test_mnist)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), -1e-06)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('CW0 Target: %s', target)
        logger.debug('CW0 Actual: %s', y_pred_adv)
        logger.info('CW0 Success Rate: %.2f', np.sum(target != y_pred_adv) / float(len(target)))
        self.assertTrue((target != y_pred_adv).any())
        if sess is not None:
            sess.close()

    def test_classifier_type_check_fail_LInf(self):
        if False:
            return 10
        backend_test_classifier_type_check_fail(CarliniLInfMethod, [BaseEstimator, ClassGradientsMixin])

    def test_check_params_LInf(self):
        if False:
            print('Hello World!')
        ptc = get_image_classifier_pt(from_logits=True)
        with self.assertRaises(ValueError):
            _ = CarliniLInfMethod(ptc, max_iter='1.0')
        with self.assertRaises(ValueError):
            _ = CarliniLInfMethod(ptc, max_iter=-1)
        with self.assertRaises(ValueError):
            _ = CarliniLInfMethod(ptc, decrease_factor='1.0')
        with self.assertRaises(ValueError):
            _ = CarliniLInfMethod(ptc, decrease_factor=-1)
        with self.assertRaises(ValueError):
            _ = CarliniLInfMethod(ptc, initial_const='1.0')
        with self.assertRaises(ValueError):
            _ = CarliniLInfMethod(ptc, initial_const=-1)
        with self.assertRaises(ValueError):
            _ = CarliniLInfMethod(ptc, largest_const='1.0')
        with self.assertRaises(ValueError):
            _ = CarliniLInfMethod(ptc, largest_const=-1)
        with self.assertRaises(ValueError):
            _ = CarliniLInfMethod(ptc, const_factor='1.0')
        with self.assertRaises(ValueError):
            _ = CarliniLInfMethod(ptc, const_factor=-1)
    '\n    A unittest class for testing the Carlini L0 attack.\n    '

    def test_tensorflow_failure_attack_L0(self):
        if False:
            return 10
        '\n        Test the corner case when attack is failed.\n        :return:\n        '
        (tfc, sess) = get_image_classifier_tf(from_logits=True)
        cl0m = CarliniL0Method(classifier=tfc, targeted=False, max_iter=1, batch_size=10, learning_rate=0.01, binary_search_steps=1, warm_start=True, verbose=False)
        params = {'y': random_targets(self.y_test_mnist, tfc.nb_classes)}
        _ = cl0m.generate(self.x_test_mnist, **params)
        if sess is not None:
            sess.close()

    def test_tensorflow_mnist_L0(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        First test with the TensorFlowClassifier.\n        :return:\n        '
        (tfc, sess) = get_image_classifier_tf(from_logits=True)
        cl0m = CarliniL0Method(classifier=tfc, targeted=True, max_iter=1, batch_size=10, binary_search_steps=1, verbose=False)
        params = {'y': random_targets(self.y_test_mnist, tfc.nb_classes)}
        x_test_adv = cl0m.generate(self.x_test_mnist, **params)
        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        target = np.argmax(params['y'], axis=1)
        cl0m = CarliniL0Method(classifier=tfc, targeted=False, max_iter=1, batch_size=10, binary_search_steps=1, verbose=False)
        x_test_adv = cl0m.generate(self.x_test_mnist)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), -1e-06)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('CW0 Target: %s', target)
        logger.debug('CW0 Actual: %s', y_pred_adv)
        logger.info('CW0 Success Rate: %.2f', np.sum(target != y_pred_adv) / float(len(target)))
        self.assertTrue((target != y_pred_adv).any())
        if sess is not None:
            sess.close()

    def test_classifier_type_check_fail_L0(self):
        if False:
            print('Hello World!')
        backend_test_classifier_type_check_fail(CarliniL0Method, [BaseEstimator, ClassGradientsMixin])

    def test_check_params_L0(self):
        if False:
            for i in range(10):
                print('nop')
        ptc = get_image_classifier_pt(from_logits=True)
        with self.assertRaises(ValueError):
            _ = CarliniL0Method(ptc, binary_search_steps='1.0')
        with self.assertRaises(ValueError):
            _ = CarliniL0Method(ptc, binary_search_steps=-1)
        with self.assertRaises(ValueError):
            _ = CarliniL0Method(ptc, max_iter='1.0')
        with self.assertRaises(ValueError):
            _ = CarliniL0Method(ptc, max_iter=-1)
        with self.assertRaises(ValueError):
            _ = CarliniL0Method(ptc, max_halving='1.0')
        with self.assertRaises(ValueError):
            _ = CarliniL0Method(ptc, max_halving=-1)
        with self.assertRaises(ValueError):
            _ = CarliniL0Method(ptc, max_doubling='1.0')
        with self.assertRaises(ValueError):
            _ = CarliniL0Method(ptc, max_doubling=-1)
        with self.assertRaises(ValueError):
            _ = CarliniL0Method(ptc, batch_size='1.0')
        with self.assertRaises(ValueError):
            _ = CarliniL0Method(ptc, batch_size=-1)
if __name__ == '__main__':
    unittest.main()