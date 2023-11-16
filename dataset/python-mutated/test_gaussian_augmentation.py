from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
import numpy as np
from art.defences.preprocessor import GaussianAugmentation
from tests.utils import master_seed
logger = logging.getLogger(__name__)

class TestGaussianAugmentation(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        master_seed(seed=1234)

    def test_small_size(self):
        if False:
            i = 10
            return i + 15
        x = np.arange(15).reshape((5, 3))
        ga = GaussianAugmentation(ratio=0.4, clip_values=(0, 15))
        (x_new, _) = ga(x)
        self.assertEqual(x_new.shape, (7, 3))

    def test_double_size(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.arange(12).reshape((4, 3))
        x_original = x.copy()
        ga = GaussianAugmentation()
        (x_new, _) = ga(x)
        self.assertEqual(x_new.shape[0], 2 * x.shape[0])
        self.assertAlmostEqual(float(np.max(np.abs(x_original - x))), 0.0, delta=1e-05)

    def test_multiple_size(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.arange(12).reshape((4, 3))
        x_original = x.copy()
        ga = GaussianAugmentation(ratio=3.5)
        (x_new, _) = ga(x)
        self.assertEqual(int(4.5 * x.shape[0]), x_new.shape[0])
        self.assertAlmostEqual(float(np.max(np.abs(x_original - x))), 0.0, delta=1e-05)

    def test_labels(self):
        if False:
            return 10
        x = np.arange(12).reshape((4, 3))
        y = np.arange(8).reshape((4, 2))
        ga = GaussianAugmentation()
        (x_new, new_y) = ga(x, y)
        self.assertTrue(x_new.shape[0] == new_y.shape[0] == 8)
        self.assertEqual(x_new.shape[1:], x.shape[1:])
        self.assertEqual(new_y.shape[1:], y.shape[1:])

    def test_no_augmentation(self):
        if False:
            return 10
        x = np.arange(12).reshape((4, 3))
        ga = GaussianAugmentation(augmentation=False)
        (x_new, _) = ga(x)
        self.assertEqual(x.shape, x_new.shape)
        self.assertFalse((x == x_new).all())

    def test_failure_augmentation_fit_predict(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError) as context:
            _ = GaussianAugmentation(augmentation=True, apply_fit=False, apply_predict=True)
        self.assertTrue('If `augmentation` is `True`, then `apply_fit` must be `True` and `apply_predict` must be `False`.' in str(context.exception))
        with self.assertRaises(ValueError) as context:
            _ = GaussianAugmentation(augmentation=True, apply_fit=False, apply_predict=False)
        self.assertIn("If `augmentation` is `True`, then `apply_fit` and `apply_predict` can't be both `False`.", str(context.exception))

    def test_check_params(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            _ = GaussianAugmentation(augmentation=True, ratio=-1)
        with self.assertRaises(ValueError):
            _ = GaussianAugmentation(clip_values=(0, 1, 2))
        with self.assertRaises(ValueError):
            _ = GaussianAugmentation(clip_values=(1, 0))
if __name__ == '__main__':
    unittest.main()