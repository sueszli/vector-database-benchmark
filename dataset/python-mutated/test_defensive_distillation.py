from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
import numpy as np
from art.defences.transformer.evasion import DefensiveDistillation
from tests.utils import master_seed, TestBase
from tests.utils import get_image_classifier_tf, get_image_classifier_pt, get_image_classifier_kr
from tests.utils import get_tabular_classifier_tf, get_tabular_classifier_kr, get_tabular_classifier_pt
logger = logging.getLogger(__name__)
BATCH_SIZE = 10
NB_EPOCHS = 30

def cross_entropy(prob1, prob2, eps=1e-10):
    if False:
        i = 10
        return i + 15
    '\n    Compute cross-entropy between two probability distributions.\n\n    :param prob1: First probability distribution.\n    :type prob1: `np.ndarray`\n    :param prob2: Second probability distribution.\n    :type prob2: `np.ndarray`\n    :param eps: A small amount to avoid the possibility of having a log of zero.\n    :type eps: `float`\n    :return: Cross entropy.\n    :rtype: `float`\n    '
    prob1 = np.clip(prob1, eps, 1.0 - eps)
    size = prob1.shape[0]
    result = -np.sum(prob2 * np.log(prob1 + eps)) / size
    return result

class TestDefensiveDistillation(TestBase):
    """
    A unittest class for testing the DefensiveDistillation transformer on image data.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        master_seed(seed=1234, set_tensorflow=True)
        super().setUpClass()

    def setUp(self):
        if False:
            return 10
        super().setUp()

    def test_1_tensorflow_classifier(self):
        if False:
            while True:
                i = 10
        '\n        First test with the TensorFlowClassifier.\n        :return:\n        '
        (trained_classifier, sess) = get_image_classifier_tf()
        (transformed_classifier, _) = get_image_classifier_tf(load_init=False, sess=sess)
        transformer = DefensiveDistillation(classifier=trained_classifier, batch_size=BATCH_SIZE, nb_epochs=NB_EPOCHS)
        transformed_classifier = transformer(x=self.x_train_mnist, transformed_classifier=transformed_classifier)
        preds1 = trained_classifier.predict(x=self.x_train_mnist, batch_size=BATCH_SIZE)
        preds2 = transformed_classifier.predict(x=self.x_train_mnist, batch_size=BATCH_SIZE)
        preds1 = np.argmax(preds1, axis=1)
        preds2 = np.argmax(preds2, axis=1)
        acc = np.sum(preds1 == preds2) / len(preds1)
        self.assertGreater(acc, 0.5)
        ce = cross_entropy(preds1, preds2)
        self.assertLess(ce, 10)
        self.assertGreaterEqual(ce, 0)
        if sess is not None:
            sess.close()

    def test_3_pytorch_classifier(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Second test with the PyTorchClassifier.\n        :return:\n        '
        self.x_train_mnist = np.reshape(self.x_train_mnist, (self.x_train_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        trained_classifier = get_image_classifier_pt()
        transformed_classifier = get_image_classifier_pt(load_init=False)
        transformer = DefensiveDistillation(classifier=trained_classifier, batch_size=BATCH_SIZE, nb_epochs=NB_EPOCHS)
        transformed_classifier = transformer(x=self.x_train_mnist, transformed_classifier=transformed_classifier)
        preds1 = trained_classifier.predict(x=self.x_train_mnist, batch_size=BATCH_SIZE)
        preds2 = transformed_classifier.predict(x=self.x_train_mnist, batch_size=BATCH_SIZE)
        preds1 = np.argmax(preds1, axis=1)
        preds2 = np.argmax(preds2, axis=1)
        acc = np.sum(preds1 == preds2) / len(preds1)
        self.assertGreater(acc, 0.5)
        ce = cross_entropy(preds1, preds2)
        self.assertLess(ce, 10)
        self.assertGreaterEqual(ce, 0)
        self.x_train_mnist = np.reshape(self.x_train_mnist, (self.x_train_mnist.shape[0], 28, 28, 1)).astype(np.float32)

    def test_5_keras_classifier(self):
        if False:
            i = 10
            return i + 15
        '\n        Third test with the KerasClassifier.\n        :return:\n        '
        trained_classifier = get_image_classifier_kr()
        transformed_classifier = get_image_classifier_kr(load_init=False)
        transformer = DefensiveDistillation(classifier=trained_classifier, batch_size=BATCH_SIZE, nb_epochs=NB_EPOCHS)
        transformed_classifier = transformer(x=self.x_train_mnist, transformed_classifier=transformed_classifier)
        preds1 = trained_classifier.predict(x=self.x_train_mnist, batch_size=BATCH_SIZE)
        preds2 = transformed_classifier.predict(x=self.x_train_mnist, batch_size=BATCH_SIZE)
        preds1 = np.argmax(preds1, axis=1)
        preds2 = np.argmax(preds2, axis=1)
        acc = np.sum(preds1 == preds2) / len(preds1)
        self.assertGreater(acc, 0.5)
        ce = cross_entropy(preds1, preds2)
        self.assertLess(ce, 10)
        self.assertGreaterEqual(ce, 0)

    def test_2_tensorflow_iris(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        First test for TensorFlow.\n        :return:\n        '
        (trained_classifier, sess) = get_tabular_classifier_tf()
        (transformed_classifier, _) = get_tabular_classifier_tf(load_init=False, sess=sess)
        transformer = DefensiveDistillation(classifier=trained_classifier, batch_size=BATCH_SIZE, nb_epochs=NB_EPOCHS)
        with self.assertRaises(ValueError) as context:
            _ = transformer(x=self.x_train_iris, transformed_classifier=transformed_classifier)
        self.assertIn('The input trained classifier do not produce probability outputs.', str(context.exception))
        if sess is not None:
            sess.close()

    def test_6_keras_iris(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Second test for Keras.\n        :return:\n        '
        trained_classifier = get_tabular_classifier_kr()
        transformed_classifier = get_tabular_classifier_kr(load_init=False)
        transformer = DefensiveDistillation(classifier=trained_classifier, batch_size=BATCH_SIZE, nb_epochs=NB_EPOCHS)
        transformed_classifier = transformer(x=self.x_train_iris, transformed_classifier=transformed_classifier)
        preds1 = trained_classifier.predict(x=self.x_train_iris, batch_size=BATCH_SIZE)
        preds2 = transformed_classifier.predict(x=self.x_train_iris, batch_size=BATCH_SIZE)
        preds1 = np.argmax(preds1, axis=1)
        preds2 = np.argmax(preds2, axis=1)
        acc = np.sum(preds1 == preds2) / len(preds1)
        self.assertGreater(acc, 0.2)
        ce = cross_entropy(preds1, preds2)
        self.assertLess(ce, 20)
        self.assertGreaterEqual(ce, 0)

    def test_4_pytorch_iris(self):
        if False:
            return 10
        '\n        Third test for PyTorch.\n        :return:\n        '
        trained_classifier = get_tabular_classifier_pt()
        transformed_classifier = get_tabular_classifier_pt(load_init=False)
        transformer = DefensiveDistillation(classifier=trained_classifier, batch_size=BATCH_SIZE, nb_epochs=NB_EPOCHS)
        with self.assertRaises(ValueError) as context:
            _ = transformer(x=self.x_train_iris, transformed_classifier=transformed_classifier)
        self.assertIn('The input trained classifier do not produce probability outputs.', str(context.exception))

    def test_check_params_pt(self):
        if False:
            for i in range(10):
                print('nop')
        ptc = get_image_classifier_pt(from_logits=True)
        with self.assertRaises(ValueError):
            _ = DefensiveDistillation(ptc, batch_size=1.0)
        with self.assertRaises(ValueError):
            _ = DefensiveDistillation(ptc, batch_size=-1)
        with self.assertRaises(ValueError):
            _ = DefensiveDistillation(ptc, nb_epochs=1.0)
        with self.assertRaises(ValueError):
            _ = DefensiveDistillation(ptc, nb_epochs=-1)
if __name__ == '__main__':
    unittest.main()