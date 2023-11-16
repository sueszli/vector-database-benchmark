import logging
import unittest
import numpy as np
from art.defences.postprocessor import GaussianNoise
from art.utils import load_dataset
from tests.utils import master_seed, get_image_classifier_kr_tf, get_image_classifier_kr_tf_binary
logger = logging.getLogger(__name__)

class TestGaussianNoise(unittest.TestCase):
    """
    A unittest class for testing the GaussianNoise postprocessor.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        ((x_train, y_train), (x_test, y_test), _, _) = load_dataset('mnist')
        cls.mnist = ((x_train, y_train), (x_test, y_test))

    def setUp(self):
        if False:
            while True:
                i = 10
        master_seed(seed=1234)

    def test_gaussian_noise(self):
        if False:
            print('Hello World!')
        '\n        Test Gaussian noise.\n        '
        ((_, _), (x_test, _)) = self.mnist
        classifier = get_image_classifier_kr_tf()
        preds = classifier.predict(x_test[0:1])
        postprocessor = GaussianNoise(scale=0.1)
        post_preds = postprocessor(preds=preds)
        classifier_prediction_expected = np.asarray([[0.12109935, 0.0498215, 0.0993958, 0.06410096, 0.11366928, 0.04645343, 0.06419807, 0.30685693, 0.07616714, 0.05823757]], dtype=np.float32)
        post_classifier_prediction_expected = np.asarray([[0.15412168, 0.0, 0.2222987, 0.03007976, 0.0381179, 0.12382449, 0.13755375, 0.22279163, 0.07121207, 0.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(preds, classifier_prediction_expected, decimal=4)
        np.testing.assert_array_almost_equal(post_preds, post_classifier_prediction_expected, decimal=4)

    def test_gaussian_noise_binary(self):
        if False:
            print('Hello World!')
        '\n        Test Gaussian noise for binary classifier.\n        '
        ((_, _), (x_test, _)) = self.mnist
        classifier = get_image_classifier_kr_tf_binary()
        preds = classifier.predict(x_test[0:1])
        postprocessor = GaussianNoise(scale=0.1)
        post_preds = postprocessor(preds=preds)
        classifier_prediction_expected = np.asarray([[0.5301345]], dtype=np.float32)
        post_classifier_prediction_expected = np.asarray([[0.577278]], dtype=np.float32)
        np.testing.assert_array_almost_equal(preds, classifier_prediction_expected, decimal=4)
        np.testing.assert_array_almost_equal(post_preds, post_classifier_prediction_expected, decimal=4)

    def test_check_params(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            _ = GaussianNoise(scale=-0.1)

    def test_set_params(self):
        if False:
            while True:
                i = 10
        gan = GaussianNoise(scale=0.1)
        gan.set_params(scale=0.2)
        self.assertEqual(gan.scale, 0.2)

    def test_super(self):
        if False:
            for i in range(10):
                print('nop')
        gan = GaussianNoise(scale=0.1)
        self.assertTrue(gan.is_fitted)
        self.assertFalse(gan._apply_fit)
        self.assertTrue(gan._apply_predict)
        gan.fit(preds=np.array([0.1, 0.2, 0.3]))
if __name__ == '__main__':
    unittest.main()