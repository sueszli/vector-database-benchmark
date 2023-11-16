from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
import numpy as np
from art.attacks.evasion.newtonfool import NewtonFool
from art.estimators.classification.classifier import ClassGradientsMixin
from art.estimators.classification.keras import KerasClassifier
from art.estimators.estimator import BaseEstimator
from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import TestBase, get_image_classifier_kr, get_image_classifier_pt, get_image_classifier_tf, get_tabular_classifier_kr, get_tabular_classifier_pt, get_tabular_classifier_tf
logger = logging.getLogger(__name__)

class TestNewtonFool(TestBase):
    """
    A unittest class for testing the NewtonFool attack.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        super().setUpClass()

    def test_3_tensorflow_mnist(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        First test with the TensorFlowClassifier.\n        :return:\n        '
        x_test_original = self.x_test_mnist.copy()
        (tfc, sess) = get_image_classifier_tf()
        nf = NewtonFool(tfc, max_iter=5, batch_size=100, verbose=False)
        x_test_adv = nf.generate(self.x_test_mnist)
        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        y_pred = tfc.predict(self.x_test_mnist)
        y_pred_adv = tfc.predict(x_test_adv)
        y_pred_bool = y_pred.max(axis=1, keepdims=1) == y_pred
        y_pred_max = y_pred.max(axis=1)
        y_pred_adv_max = y_pred_adv[y_pred_bool]
        self.assertTrue((y_pred_max >= 0.9 * y_pred_adv_max).all())
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=1e-05)

    def test_9_keras_mnist(self):
        if False:
            i = 10
            return i + 15
        '\n        Second test with the KerasClassifier.\n        :return:\n        '
        x_test_original = self.x_test_mnist.copy()
        krc = get_image_classifier_kr()
        nf = NewtonFool(krc, max_iter=5, batch_size=100, verbose=False)
        x_test_adv = nf.generate(self.x_test_mnist)
        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        y_pred = krc.predict(self.x_test_mnist)
        y_pred_adv = krc.predict(x_test_adv)
        y_pred_bool = y_pred.max(axis=1, keepdims=1) == y_pred
        y_pred_max = y_pred.max(axis=1)
        y_pred_adv_max = y_pred_adv[y_pred_bool]
        self.assertTrue((y_pred_max >= 0.9 * y_pred_adv_max).all())
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=1e-05)

    def test_5_pytorch_mnist(self):
        if False:
            return 10
        '\n        Third test with the PyTorchClassifier.\n        :return:\n        '
        x_test = np.swapaxes(self.x_test_mnist, 1, 3).astype(np.float32)
        x_test_original = x_test.copy()
        ptc = get_image_classifier_pt()
        nf = NewtonFool(ptc, max_iter=5, batch_size=100, verbose=False)
        x_test_adv = nf.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        y_pred = ptc.predict(x_test)
        y_pred_adv = ptc.predict(x_test_adv)
        y_pred_bool = y_pred.max(axis=1, keepdims=1) == y_pred
        y_pred_max = y_pred.max(axis=1)
        y_pred_adv_max = y_pred_adv[y_pred_bool]
        self.assertTrue((y_pred_max >= 0.9 * y_pred_adv_max).all())
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=1e-05)

    def test_7_keras_iris_clipped(self):
        if False:
            return 10
        classifier = get_tabular_classifier_kr()
        attack = NewtonFool(classifier, max_iter=5, verbose=False)
        x_test_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())
        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info('Accuracy on Iris with NewtonFool adversarial examples: %.2f%%', acc * 100)

    def test_8_keras_iris_unbounded(self):
        if False:
            for i in range(10):
                print('nop')
        classifier = get_tabular_classifier_kr()
        classifier = KerasClassifier(model=classifier._model, use_logits=False, channels_first=True)
        attack = NewtonFool(classifier, max_iter=5, batch_size=128, verbose=False)
        x_test_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info('Accuracy on Iris with NewtonFool adversarial examples: %.2f%%', acc * 100)

    def test_2_tensorflow_iris(self):
        if False:
            while True:
                i = 10
        (classifier, _) = get_tabular_classifier_tf()
        attack = NewtonFool(classifier, max_iter=5, batch_size=128, verbose=False)
        x_test_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())
        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info('Accuracy on Iris with NewtonFool adversarial examples: %.2f%%', acc * 100)

    def test_4_pytorch_iris(self):
        if False:
            while True:
                i = 10
        classifier = get_tabular_classifier_pt()
        attack = NewtonFool(classifier, max_iter=5, batch_size=128, verbose=False)
        x_test_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())
        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info('Accuracy on Iris with NewtonFool adversarial examples: %.2f%%', acc * 100)

    def test_6_scikitlearn(self):
        if False:
            return 10
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC, LinearSVC
        from art.estimators.classification.scikitlearn import SklearnClassifier
        scikitlearn_test_cases = [LogisticRegression(solver='lbfgs', multi_class='auto'), SVC(gamma='auto'), LinearSVC()]
        x_test_original = self.x_test_iris.copy()
        for model in scikitlearn_test_cases:
            classifier = SklearnClassifier(model=model, clip_values=(0, 1))
            classifier.fit(x=self.x_test_iris, y=self.y_test_iris)
            attack = NewtonFool(classifier, max_iter=5, batch_size=128, verbose=False)
            x_test_adv = attack.generate(self.x_test_iris)
            self.assertFalse((self.x_test_iris == x_test_adv).all())
            self.assertTrue((x_test_adv <= 1).all())
            self.assertTrue((x_test_adv >= 0).all())
            preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
            acc = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
            logger.info('Accuracy of ' + classifier.__class__.__name__ + ' on Iris with NewtonFool adversarial examples: %.2f%%', acc * 100)
            self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_iris))), 0.0, delta=1e-05)

    def test_check_params(self):
        if False:
            for i in range(10):
                print('nop')
        ptc = get_image_classifier_pt(from_logits=True)
        with self.assertRaises(ValueError):
            _ = NewtonFool(ptc, max_iter=-1)
        with self.assertRaises(ValueError):
            _ = NewtonFool(ptc, eta=-1)
        with self.assertRaises(ValueError):
            _ = NewtonFool(ptc, batch_size=-1)
        with self.assertRaises(ValueError):
            _ = NewtonFool(ptc, verbose='False')

    def test_1_classifier_type_check_fail(self):
        if False:
            print('Hello World!')
        backend_test_classifier_type_check_fail(NewtonFool, [BaseEstimator, ClassGradientsMixin])
if __name__ == '__main__':
    unittest.main()