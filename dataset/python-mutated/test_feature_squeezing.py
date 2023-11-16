from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
import numpy as np
from art.defences.preprocessor import FeatureSqueezing
from tests.utils import master_seed
logger = logging.getLogger(__name__)

class TestFeatureSqueezing(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        master_seed(seed=1234)

    def test_ones(self):
        if False:
            i = 10
            return i + 15
        (m, n) = (10, 2)
        x = np.ones((m, n))
        for depth in range(1, 50):
            preproc = FeatureSqueezing(clip_values=(0, 1), bit_depth=depth)
            (x_squeezed, _) = preproc(x)
            self.assertTrue((x_squeezed == 1).all())

    def test_random(self):
        if False:
            while True:
                i = 10
        (m, n) = (1000, 20)
        x = np.random.rand(m, n)
        x_original = x.copy()
        x_zero = np.where(x < 0.5)
        x_one = np.where(x >= 0.5)
        preproc = FeatureSqueezing(clip_values=(0, 1), bit_depth=1)
        (x_squeezed, _) = preproc(x)
        self.assertTrue((x_squeezed[x_zero] == 0.0).all())
        self.assertTrue((x_squeezed[x_one] == 1.0).all())
        preproc = FeatureSqueezing(clip_values=(0, 1), bit_depth=2)
        (x_squeezed, _) = preproc(x)
        self.assertFalse(np.logical_and(0.0 < x_squeezed, x_squeezed < 0.33).any())
        self.assertFalse(np.logical_and(0.34 < x_squeezed, x_squeezed < 0.66).any())
        self.assertFalse(np.logical_and(0.67 < x_squeezed, x_squeezed < 1.0).any())
        self.assertAlmostEqual(float(np.max(np.abs(x_original - x))), 0.0, delta=1e-05)

    def test_data_range(self):
        if False:
            i = 10
            return i + 15
        x = np.arange(5)
        preproc = FeatureSqueezing(clip_values=(0, 4), bit_depth=2)
        (x_squeezed, _) = preproc(x)
        self.assertTrue(np.array_equal(x, np.arange(5)))
        self.assertTrue(np.allclose(x_squeezed, [0, 1.33, 2.67, 2.67, 4], atol=0.1))

    def test_check_params(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            _ = FeatureSqueezing(clip_values=(0, 4), bit_depth=-1)
        with self.assertRaises(ValueError):
            _ = FeatureSqueezing(clip_values=(0, 4, 8))
        with self.assertRaises(ValueError):
            _ = FeatureSqueezing(clip_values=(4, 0))
if __name__ == '__main__':
    unittest.main()