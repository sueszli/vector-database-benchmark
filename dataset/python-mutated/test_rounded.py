import logging
import unittest
import numpy as np
from art.defences.postprocessor import Rounded
from art.utils import load_dataset
from tests.utils import master_seed, get_image_classifier_kr
logger = logging.getLogger(__name__)

class TestRounded(unittest.TestCase):
    """
    A unittest class for testing the Rounded postprocessor.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        ((x_train, y_train), (x_test, y_test), _, _) = load_dataset('mnist')
        cls.mnist = ((x_train, y_train), (x_test, y_test))
        cls.classifier = get_image_classifier_kr()

    def setUp(self):
        if False:
            while True:
                i = 10
        master_seed(seed=1234)

    def test_decimals_2(self):
        if False:
            i = 10
            return i + 15
        '\n        Test with 2 decimal places.\n        '
        ((_, _), (x_test, _)) = self.mnist
        preds = self.classifier.predict(x_test[0:1])
        postprocessor = Rounded(decimals=2)
        post_preds = postprocessor(preds=preds)
        expected_predictions = np.asarray([[0.12, 0.05, 0.1, 0.06, 0.11, 0.05, 0.06, 0.31, 0.08, 0.06]], dtype=np.float32)
        np.testing.assert_array_equal(post_preds, expected_predictions)

    def test_decimals_3(self):
        if False:
            print('Hello World!')
        '\n        Test with 3 decimal places.\n        '
        ((_, _), (x_test, _)) = self.mnist
        preds = self.classifier.predict(x_test[0:1])
        postprocessor = Rounded(decimals=3)
        post_preds = postprocessor(preds=preds)
        expected_predictions = np.asarray([[0.121, 0.05, 0.099, 0.064, 0.114, 0.046, 0.064, 0.307, 0.076, 0.058]], dtype=np.float32)
        np.testing.assert_array_equal(post_preds, expected_predictions)

    def test_check_params(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            _ = Rounded(decimals=-3)
if __name__ == '__main__':
    unittest.main()