from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
import numpy as np
from art.attacks.evasion.saliency_map import SaliencyMapMethod
from art.estimators.classification.classifier import ClassGradientsMixin
from art.estimators.classification.keras import KerasClassifier
from art.estimators.estimator import BaseEstimator
from art.utils import get_labels_np_array, to_categorical
from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import TestBase, get_image_classifier_kr, get_image_classifier_pt, get_image_classifier_tf, get_tabular_classifier_kr, get_tabular_classifier_pt, get_tabular_classifier_tf
logger = logging.getLogger(__name__)

class TestSaliencyMap(TestBase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        super().setUpClass()
        cls.n_train = 100
        cls.n_test = 2
        cls.x_train_mnist = cls.x_train_mnist[0:cls.n_train]
        cls.y_train_mnist = cls.y_train_mnist[0:cls.n_train]
        cls.x_test_mnist = cls.x_test_mnist[0:cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0:cls.n_test]

    def test_9_keras_mnist(self):
        if False:
            while True:
                i = 10
        x_test_original = self.x_test_mnist.copy()
        classifier = get_image_classifier_kr()
        scores = classifier._model.evaluate(self.x_train_mnist, self.y_train_mnist)
        logger.info('[Keras, MNIST] Accuracy on training set: %.2f%%', scores[1] * 100)
        scores = classifier._model.evaluate(self.x_test_mnist, self.y_test_mnist)
        logger.info('[Keras, MNIST] Accuracy on test set: %.2f%%', scores[1] * 100)
        nb_classes = np.unique(np.argmax(self.y_test_mnist, axis=1)).shape[0]
        targets = np.random.randint(nb_classes, size=self.n_test)
        while (targets == np.argmax(self.y_test_mnist, axis=1)).any():
            targets = np.random.randint(nb_classes, size=self.n_test)
        df = SaliencyMapMethod(classifier, theta=1, batch_size=100, verbose=False)
        x_test_adv = df.generate(self.x_test_mnist, y=to_categorical(targets, nb_classes))
        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertFalse((0.0 == x_test_adv).all())
        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((self.y_test_mnist == y_pred).all())
        accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.n_test
        logger.info('Accuracy on adversarial examples: %.2f%%', accuracy * 100)
        df = SaliencyMapMethod(classifier, theta=1, batch_size=100, verbose=False)
        x_test_adv = df.generate(self.x_test_mnist)
        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertFalse((0.0 == x_test_adv).all())
        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((self.y_test_mnist == y_pred).all())
        accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.n_test
        logger.info('Accuracy on adversarial examples: %.2f%%', accuracy * 100)
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=1e-05)

    def test_3_tensorflow_mnist(self):
        if False:
            i = 10
            return i + 15
        x_test_original = self.x_test_mnist.copy()
        (classifier, sess) = get_image_classifier_tf()
        scores = get_labels_np_array(classifier.predict(self.x_train_mnist))
        accuracy = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_train_mnist, axis=1)) / self.n_train
        logger.info('[TF, MNIST] Accuracy on training set: %.2f%%', accuracy * 100)
        scores = get_labels_np_array(classifier.predict(self.x_test_mnist))
        accuracy = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.n_train
        logger.info('[TF, MNIST] Accuracy on test set: %.2f%%', accuracy * 100)
        nb_classes = np.unique(np.argmax(self.y_test_mnist, axis=1)).shape[0]
        targets = np.random.randint(nb_classes, size=self.n_test)
        while (targets == np.argmax(self.y_test_mnist, axis=1)).any():
            targets = np.random.randint(nb_classes, size=self.n_test)
        df = SaliencyMapMethod(classifier, theta=1, batch_size=100, verbose=False)
        x_test_adv = df.generate(self.x_test_mnist, y=to_categorical(targets, nb_classes))
        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertFalse((0.0 == x_test_adv).all())
        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((self.y_test_mnist == y_pred).all())
        accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.n_test
        logger.info('Accuracy on adversarial examples: %.2f%%', accuracy * 100)
        df = SaliencyMapMethod(classifier, theta=1, batch_size=100, verbose=False)
        x_test_adv = df.generate(self.x_test_mnist)
        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertFalse((0.0 == x_test_adv).all())
        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((self.y_test_mnist == y_pred).all())
        accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.n_test
        logger.info('Accuracy on adversarial examples: %.2f%%', accuracy * 100)
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=1e-05)

    def test_5_pytorch_mnist(self):
        if False:
            return 10
        x_train_mnist = np.swapaxes(self.x_train_mnist, 1, 3).astype(np.float32)
        x_test_mnist = np.swapaxes(self.x_test_mnist, 1, 3).astype(np.float32)
        x_test_original = x_test_mnist.copy()
        classifier = get_image_classifier_pt()
        scores = get_labels_np_array(classifier.predict(x_train_mnist))
        accuracy = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_train_mnist, axis=1)) / self.n_train
        logger.info('[PyTorch, MNIST] Accuracy on training set: %.2f%%', accuracy * 100)
        scores = get_labels_np_array(classifier.predict(x_test_mnist))
        accuracy = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.n_test
        logger.info('\n[PyTorch, MNIST] Accuracy on test set: %.2f%%', accuracy * 100)
        nb_classes = np.unique(np.argmax(self.y_test_mnist, axis=1)).shape[0]
        targets = np.random.randint(nb_classes, size=self.n_test)
        while (targets == np.argmax(self.y_test_mnist, axis=1)).any():
            targets = np.random.randint(nb_classes, size=self.n_test)
        df = SaliencyMapMethod(classifier, theta=1, batch_size=100, verbose=False)
        x_test_mnist_adv = df.generate(x_test_mnist, y=to_categorical(targets, nb_classes))
        self.assertFalse((x_test_mnist == x_test_mnist_adv).all())
        self.assertFalse((0.0 == x_test_mnist_adv).all())
        y_pred = get_labels_np_array(classifier.predict(x_test_mnist_adv))
        self.assertFalse((self.y_test_mnist == y_pred).all())
        accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.n_test
        logger.info('Accuracy on adversarial examples: %.2f%%', accuracy * 100)
        df = SaliencyMapMethod(classifier, theta=1, batch_size=100, verbose=False)
        x_test_mnist_adv = df.generate(x_test_mnist)
        self.assertFalse((x_test_mnist == x_test_mnist_adv).all())
        self.assertFalse((0.0 == x_test_mnist_adv).all())
        y_pred = get_labels_np_array(classifier.predict(x_test_mnist_adv))
        self.assertFalse((self.y_test_mnist == y_pred).all())
        accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.n_test
        logger.info('Accuracy on adversarial examples: %.2f%%', accuracy * 100)
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test_mnist))), 0.0, delta=1e-05)

    def test_7_keras_iris_vector_clipped(self):
        if False:
            for i in range(10):
                print('nop')
        classifier = get_tabular_classifier_kr()
        attack = SaliencyMapMethod(classifier, theta=1, verbose=False)
        x_test_iris_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_iris_adv).all())
        self.assertTrue((x_test_iris_adv <= 1).all())
        self.assertTrue((x_test_iris_adv >= 0).all())
        preds_adv = np.argmax(classifier.predict(x_test_iris_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        accuracy = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info('Accuracy on Iris with JSMA adversarial examples: %.2f%%', accuracy * 100)

    def test_8_keras_iris_vector_unbounded(self):
        if False:
            print('Hello World!')
        classifier = get_tabular_classifier_kr()
        classifier = KerasClassifier(model=classifier._model, use_logits=False, channels_first=True)
        attack = SaliencyMapMethod(classifier, theta=1, verbose=False)
        x_test_iris_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_iris_adv).all())

    def test_2_tensorflow_iris_vector(self):
        if False:
            for i in range(10):
                print('nop')
        (classifier, _) = get_tabular_classifier_tf()
        attack = SaliencyMapMethod(classifier, theta=1, verbose=False)
        x_test_iris_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_iris_adv).all())
        self.assertTrue((x_test_iris_adv <= 1).all())
        self.assertTrue((x_test_iris_adv >= 0).all())
        preds_adv = np.argmax(classifier.predict(x_test_iris_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        accuracy = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info('Accuracy on Iris with JSMA adversarial examples: %.2f%%', accuracy * 100)

    def test_4_pytorch_iris_vector(self):
        if False:
            while True:
                i = 10
        classifier = get_tabular_classifier_pt()
        attack = SaliencyMapMethod(classifier, theta=1, verbose=False)
        x_test_iris_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_iris_adv).all())
        self.assertTrue((x_test_iris_adv <= 1).all())
        self.assertTrue((x_test_iris_adv >= 0).all())
        preds_adv = np.argmax(classifier.predict(x_test_iris_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
        accuracy = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info('Accuracy on Iris with JSMA adversarial examples: %.2f%%', accuracy * 100)

    def test_6_scikitlearn(self):
        if False:
            i = 10
            return i + 15
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC, LinearSVC
        from art.estimators.classification.scikitlearn import SklearnClassifier
        scikitlearn_test_cases = [LogisticRegression(solver='lbfgs', multi_class='auto'), SVC(gamma='auto'), LinearSVC()]
        x_test_original = self.x_test_iris.copy()
        for model in scikitlearn_test_cases:
            classifier = SklearnClassifier(model=model, clip_values=(0, 1))
            classifier.fit(x=self.x_test_iris, y=self.y_test_iris)
            attack = SaliencyMapMethod(classifier, theta=1, batch_size=128, verbose=False)
            x_test_iris_adv = attack.generate(self.x_test_iris)
            self.assertFalse((self.x_test_iris == x_test_iris_adv).all())
            self.assertTrue((x_test_iris_adv <= 1).all())
            self.assertTrue((x_test_iris_adv >= 0).all())
            preds_adv = np.argmax(classifier.predict(x_test_iris_adv), axis=1)
            self.assertFalse((np.argmax(self.y_test_iris, axis=1) == preds_adv).all())
            accuracy = np.sum(preds_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
            logger.info('Accuracy of ' + classifier.__class__.__name__ + ' on Iris with JSMA adversarial examples: %.2f%%', accuracy * 100)
            self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_iris))), 0.0, delta=1e-05)

    def test_check_params(self):
        if False:
            i = 10
            return i + 15
        ptc = get_image_classifier_pt(from_logits=True)
        with self.assertRaises(ValueError):
            _ = SaliencyMapMethod(ptc, gamma=-1)
        with self.assertRaises(ValueError):
            _ = SaliencyMapMethod(ptc, batch_size=-1)
        with self.assertRaises(ValueError):
            _ = SaliencyMapMethod(ptc, verbose='False')

    def test_1_classifier_type_check_fail(self):
        if False:
            while True:
                i = 10
        backend_test_classifier_type_check_fail(SaliencyMapMethod, [BaseEstimator, ClassGradientsMixin])
if __name__ == '__main__':
    unittest.main()