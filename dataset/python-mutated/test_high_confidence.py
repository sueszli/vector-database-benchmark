import logging
import unittest
import numpy as np
from art.defences.postprocessor import HighConfidence
from art.utils import load_dataset
from tests.utils import master_seed, get_image_classifier_kr_tf, get_image_classifier_kr_tf_binary
logger = logging.getLogger(__name__)

class TestHighConfidence(unittest.TestCase):
    """
    A unittest class for testing the HighConfidence postprocessor.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        ((x_train, y_train), (x_test, y_test), _, _) = load_dataset('mnist')
        cls.mnist = ((x_train, y_train), (x_test, y_test))

    def setUp(self):
        if False:
            i = 10
            return i + 15
        master_seed(seed=1234)

    def test_decimals_0_1(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test with cutoff of 0.1.\n        '
        ((_, _), (x_test, _)) = self.mnist
        classifier = get_image_classifier_kr_tf()
        preds = classifier.predict(x_test[0:1])
        postprocessor = HighConfidence(cutoff=0.1)
        post_preds = postprocessor(preds=preds)
        classifier_prediction_expected = np.asarray([[0.12109935, 0.0498215, 0.0993958, 0.06410096, 0.11366928, 0.04645343, 0.06419807, 0.30685693, 0.07616714, 0.05823757]], dtype=np.float32)
        post_classifier_prediction_expected = np.asarray([[0.12109935, 0.0, 0.0, 0.0, 0.11366928, 0.0, 0.0, 0.30685693, 0.0, 0.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(preds, classifier_prediction_expected, decimal=4)
        np.testing.assert_array_almost_equal(post_preds, post_classifier_prediction_expected, decimal=4)

    def test_decimals_0_2(self):
        if False:
            i = 10
            return i + 15
        '\n        Test with cutoff of 0.2.\n        '
        ((_, _), (x_test, _)) = self.mnist
        classifier = get_image_classifier_kr_tf()
        preds = classifier.predict(x_test[0:1])
        postprocessor = HighConfidence(cutoff=0.2)
        post_preds = postprocessor(preds=preds)
        classifier_prediction_expected = np.asarray([[0.12109935, 0.0498215, 0.0993958, 0.06410096, 0.11366928, 0.04645343, 0.06419807, 0.30685693, 0.07616714, 0.05823757]], dtype=np.float32)
        post_classifier_prediction_expected = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.30685693, 0.0, 0.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(preds, classifier_prediction_expected, decimal=4)
        np.testing.assert_array_almost_equal(post_preds, post_classifier_prediction_expected, decimal=4)

    def test_binary_decimals_0_5(self):
        if False:
            while True:
                i = 10
        '\n        Test with cutoff of 0.5 for binary classifier.\n        '
        ((_, _), (x_test, _)) = self.mnist
        classifier = get_image_classifier_kr_tf_binary()
        preds = classifier.predict(x_test[0:1])
        postprocessor = HighConfidence(cutoff=0.5)
        post_preds = postprocessor(preds=preds)
        classifier_prediction_expected = np.asarray([[0.5301345]], dtype=np.float32)
        post_classifier_prediction_expected = np.asarray([[0.5301345]], dtype=np.float32)
        np.testing.assert_array_almost_equal(preds, classifier_prediction_expected, decimal=4)
        np.testing.assert_array_almost_equal(post_preds, post_classifier_prediction_expected, decimal=4)

    def test_binary_decimals_0_6(self):
        if False:
            return 10
        '\n        Test with cutoff of 0.6 for binary classifier.\n        '
        ((_, _), (x_test, _)) = self.mnist
        classifier = get_image_classifier_kr_tf_binary()
        preds = classifier.predict(x_test[0:1])
        postprocessor = HighConfidence(cutoff=0.6)
        post_preds = postprocessor(preds=preds)
        classifier_prediction_expected = np.asarray([[0.5301345]], dtype=np.float32)
        post_classifier_prediction_expected = np.asarray([[0.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(preds, classifier_prediction_expected, decimal=4)
        np.testing.assert_array_almost_equal(post_preds, post_classifier_prediction_expected, decimal=4)

    def test_check_params(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError):
            _ = HighConfidence(cutoff=-0.6)
if __name__ == '__main__':
    unittest.main()