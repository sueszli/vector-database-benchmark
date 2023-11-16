from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
import numpy as np
from art.metrics.gradient_check import loss_gradient_check
from art.utils import load_mnist
from tests.utils import TestBase, master_seed, get_image_classifier_kr
logger = logging.getLogger(__name__)
BATCH_SIZE = 10
NB_TRAIN = 100
NB_TEST = 100

class TestGradientCheck(TestBase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        master_seed(seed=42)
        super().setUp()

    def test_loss_gradient_check(self):
        if False:
            print('Hello World!')
        ((x_train, y_train), (x_test, y_test), _, _) = load_mnist()
        (x_train, y_train) = (x_train[:NB_TRAIN], y_train[:NB_TRAIN])
        (x_test, y_test) = (x_test[:NB_TEST], y_test[:NB_TEST])
        classifier = get_image_classifier_kr()
        classifier.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epochs=2)
        weights = classifier._model.layers[0].get_weights()
        new_weights = [np.zeros(w.shape) for w in weights]
        classifier._model.layers[0].set_weights(new_weights)
        is_bad = loss_gradient_check(classifier, x_test, y_test, verbose=False)
        self.assertTrue(np.all(np.any(is_bad, 1)))
        weights = classifier._model.layers[0].get_weights()
        new_weights = [np.empty(w.shape) for w in weights]
        for i in range(len(new_weights)):
            new_weights[i][:] = np.nan
        classifier._model.layers[0].set_weights(new_weights)
        is_bad = loss_gradient_check(classifier, x_test, y_test, verbose=False)
        self.assertTrue(np.all(np.any(is_bad, 1)))
if __name__ == '__main__':
    unittest.main()