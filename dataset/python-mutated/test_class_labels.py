import logging
import unittest
import numpy as np
from art.utils import load_dataset
from art.defences.postprocessor import ClassLabels
from tests.utils import master_seed, get_image_classifier_kr_tf, get_image_classifier_kr_tf_binary
logger = logging.getLogger(__name__)

class TestClassLabels(unittest.TestCase):
    """
    A unittest class for testing the ClassLabels postprocessor.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        ((x_train, y_train), (x_test, y_test), _, _) = load_dataset('mnist')
        cls.mnist = ((x_train, y_train), (x_test, y_test))

    def setUp(self):
        if False:
            print('Hello World!')
        master_seed(seed=1234)

    def test_class_labels(self):
        if False:
            print('Hello World!')
        '\n        Test class labels.\n        '
        ((_, _), (x_test, _)) = self.mnist
        classifier = get_image_classifier_kr_tf()
        preds = classifier.predict(x_test[0:1])
        postprocessor = ClassLabels()
        post_preds = postprocessor(preds=preds)
        classifier_prediction_expected = np.asarray([[0.12109935, 0.0498215, 0.0993958, 0.06410096, 0.11366928, 0.04645343, 0.06419807, 0.30685693, 0.07616714, 0.05823757]], dtype=np.float32)
        post_classifier_prediction_expected = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(preds, classifier_prediction_expected, decimal=4)
        np.testing.assert_array_almost_equal(post_preds, post_classifier_prediction_expected, decimal=4)

    def test_class_labels_binary(self):
        if False:
            return 10
        '\n        Test class labels for binary classifier.\n        '
        ((_, _), (x_test, _)) = self.mnist
        classifier = get_image_classifier_kr_tf_binary()
        preds = classifier.predict(x_test[0:1])
        postprocessor = ClassLabels()
        post_preds = postprocessor(preds=preds)
        classifier_prediction_expected = np.asarray([[0.5301345]], dtype=np.float32)
        post_classifier_prediction_expected = np.asarray([[1.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(preds, classifier_prediction_expected, decimal=4)
        np.testing.assert_array_almost_equal(post_preds, post_classifier_prediction_expected, decimal=4)
if __name__ == '__main__':
    unittest.main()