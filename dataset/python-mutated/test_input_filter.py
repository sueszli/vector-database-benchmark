import logging
import unittest
import numpy as np
from tests.utils import TestBase, master_seed, get_image_classifier_kr_tf
logger = logging.getLogger(__name__)
BATCH_SIZE = 10
NB_TRAIN = 500
NB_TEST = 100

class TestInputFilter(TestBase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        master_seed(seed=1234)
        super().setUpClass()

    def setUp(self):
        if False:
            print('Hello World!')
        master_seed(1234)
        super().setUp()

    def test_fit(self):
        if False:
            for i in range(10):
                print('nop')
        labels = np.argmax(self.y_test_mnist, axis=1)
        classifier = get_image_classifier_kr_tf()
        acc = np.sum(np.argmax(classifier.predict(self.x_test_mnist), axis=1) == labels) / NB_TEST
        logger.info('Accuracy: %.2f%%', acc * 100)
        classifier.fit(self.x_train_mnist, self.y_train_mnist, batch_size=BATCH_SIZE, nb_epochs=2)
        acc2 = np.sum(np.argmax(classifier.predict(self.x_test_mnist), axis=1) == labels) / NB_TEST
        logger.info('Accuracy: %.2f%%', acc2 * 100)
        self.assertEqual(acc, 0.32)
        self.assertEqual(acc2, 0.77)
        classifier.fit(self.x_train_mnist, y=self.y_train_mnist, batch_size=BATCH_SIZE, nb_epochs=2)
        classifier.fit(x=self.x_train_mnist, y=self.y_train_mnist, batch_size=BATCH_SIZE, nb_epochs=2)

    def test_layers(self):
        if False:
            return 10
        classifier = get_image_classifier_kr_tf()
        self.assertEqual(len(classifier.layer_names), 3)
        layer_names = classifier.layer_names
        for (i, name) in enumerate(layer_names):
            act_i = classifier.get_activations(self.x_test_mnist, i, batch_size=128)
            act_name = classifier.get_activations(self.x_test_mnist, name, batch_size=128)
            np.testing.assert_array_equal(act_name, act_i)
if __name__ == '__main__':
    unittest.main()