from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
import numpy as np
from art.attacks.evasion.virtual_adversarial import VirtualAdversarialMethod
from art.estimators.classification.keras import KerasClassifier
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import get_labels_np_array
from tests.utils import TestBase
from tests.utils import get_image_classifier_tf, get_image_classifier_kr, get_image_classifier_pt
from tests.utils import get_tabular_classifier_tf, get_tabular_classifier_kr, get_tabular_classifier_pt
from tests.attacks.utils import backend_test_classifier_type_check_fail
logger = logging.getLogger(__name__)

class TestVirtualAdversarial(TestBase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        super().setUpClass()
        cls.n_train = 100
        cls.n_test = 10
        cls.x_train_mnist = cls.x_train_mnist[0:cls.n_train]
        cls.y_train_mnist = cls.y_train_mnist[0:cls.n_train]
        cls.x_test_mnist = cls.x_test_mnist[0:cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0:cls.n_test]

    def test_8_keras_mnist(self):
        if False:
            return 10
        classifier = get_image_classifier_kr()
        scores = classifier._model.evaluate(self.x_train_mnist, self.y_train_mnist)
        logging.info('[Keras, MNIST] Accuracy on training set: %.2f%%', scores[1] * 100)
        scores = classifier._model.evaluate(self.x_test_mnist, self.y_test_mnist)
        logging.info('[Keras, MNIST] Accuracy on test set: %.2f%%', scores[1] * 100)
        self._test_backend_mnist(classifier, self.x_test_mnist, self.y_test_mnist)

    def test_3_tensorflow_mnist(self):
        if False:
            for i in range(10):
                print('nop')
        (classifier, sess) = get_image_classifier_tf(from_logits=False)
        scores = get_labels_np_array(classifier.predict(self.x_train_mnist))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_train_mnist, axis=1)) / self.y_train_mnist.shape[0]
        logger.info('[TF, MNIST] Accuracy on training set: %.2f%%', acc * 100)
        scores = get_labels_np_array(classifier.predict(self.x_test_mnist))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.y_test_mnist.shape[0]
        logger.info('[TF, MNIST] Accuracy on test set: %.2f%%', acc * 100)
        self._test_backend_mnist(classifier, self.x_test_mnist, self.y_test_mnist)

    def test_5_pytorch_mnist(self):
        if False:
            for i in range(10):
                print('nop')
        x_train_mnist = np.swapaxes(self.x_train_mnist, 1, 3).astype(np.float32)
        x_test_mnist = np.swapaxes(self.x_test_mnist, 1, 3).astype(np.float32)
        classifier = get_image_classifier_pt()
        scores = get_labels_np_array(classifier.predict(x_train_mnist))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_train_mnist, axis=1)) / self.y_train_mnist.shape[0]
        logger.info('[PyTorch, MNIST] Accuracy on training set: %.2f%%', acc * 100)
        scores = get_labels_np_array(classifier.predict(x_test_mnist))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.y_test_mnist.shape[0]
        logger.info('[PyTorch, MNIST] Accuracy on test set: %.2f%%', acc * 100)
        self._test_backend_mnist(classifier, x_test_mnist, self.y_test_mnist)

    def _test_backend_mnist(self, classifier, x_test, y_test):
        if False:
            while True:
                i = 10
        x_test_original = x_test.copy()
        df = VirtualAdversarialMethod(classifier, batch_size=100, max_iter=2, verbose=False)
        x_test_adv = df.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((y_test == y_pred).all())
        acc = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on adversarial examples: %.2f%%', acc * 100)
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=1e-05)

    def test_6_keras_iris_clipped(self):
        if False:
            print('Hello World!')
        classifier = get_tabular_classifier_kr()
        attack = VirtualAdversarialMethod(classifier, eps=0.1, verbose=False)
        x_test_iris_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_iris_adv).all())
        self.assertTrue((x_test_iris_adv <= 1).all())
        self.assertTrue((x_test_iris_adv >= 0).all())
        preds_adv = np.argmax(classifier.predict(x_test_iris_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info('Accuracy on Iris with VAT adversarial examples: %.2f%%', acc * 100)

    def test_7_keras_iris_unbounded(self):
        if False:
            i = 10
            return i + 15
        classifier = get_tabular_classifier_kr()
        classifier = KerasClassifier(model=classifier._model, use_logits=False, channels_first=True)
        attack = VirtualAdversarialMethod(classifier, eps=1, verbose=False)
        x_test_iris_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_iris_adv).all())
        self.assertTrue((x_test_iris_adv > 1).any())
        self.assertTrue((x_test_iris_adv < 0).any())
        preds_adv = np.argmax(classifier.predict(x_test_iris_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info('Accuracy on Iris with VAT adversarial examples: %.2f%%', acc * 100)

    def test_2_tensorflow_iris(self):
        if False:
            while True:
                i = 10
        (classifier, _) = get_tabular_classifier_tf()
        attack = VirtualAdversarialMethod(classifier, eps=0.1, verbose=False)
        with self.assertRaises(TypeError) as context:
            _ = attack.generate(self.x_test_iris)
        self.assertIn('This attack requires a classifier predicting probabilities in the range [0, 1] as output.Values smaller than 0.0 or larger than 1.0 have been detected.', str(context.exception))

    def test_4_pytorch_iris(self):
        if False:
            return 10
        classifier = get_tabular_classifier_pt()
        attack = VirtualAdversarialMethod(classifier, eps=0.1, verbose=False)
        with self.assertRaises(TypeError) as context:
            _ = attack.generate(self.x_test_iris.astype(np.float32))
        self.assertIn('This attack requires a classifier predicting probabilities in the range [0, 1] as output.Values smaller than 0.0 or larger than 1.0 have been detected.', str(context.exception))

    def test_check_params(self):
        if False:
            return 10
        ptc = get_image_classifier_pt(from_logits=True)
        with self.assertRaises(ValueError):
            _ = VirtualAdversarialMethod(ptc, max_iter=1.0)
        with self.assertRaises(ValueError):
            _ = VirtualAdversarialMethod(ptc, max_iter=-1)
        with self.assertRaises(ValueError):
            _ = VirtualAdversarialMethod(ptc, eps=-1)
        with self.assertRaises(ValueError):
            _ = VirtualAdversarialMethod(ptc, finite_diff=1)
        with self.assertRaises(ValueError):
            _ = VirtualAdversarialMethod(ptc, finite_diff=-1.0)
        with self.assertRaises(ValueError):
            _ = VirtualAdversarialMethod(ptc, batch_size=-1)
        with self.assertRaises(ValueError):
            _ = VirtualAdversarialMethod(ptc, verbose='true')

    def test_1_classifier_type_check_fail(self):
        if False:
            while True:
                i = 10
        backend_test_classifier_type_check_fail(VirtualAdversarialMethod, [BaseEstimator, ClassifierMixin])
if __name__ == '__main__':
    unittest.main()