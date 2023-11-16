from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
import numpy as np
from art.defences.preprocessor import TotalVarMin
from tests.utils import master_seed
logger = logging.getLogger(__name__)

class TestTotalVarMin(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        master_seed(seed=1234)

    def test_one_channel(self):
        if False:
            return 10
        clip_values = (0, 1)
        x = np.random.rand(2, 28, 28, 1)
        preprocess = TotalVarMin(clip_values=(0, 1))
        (x_preprocessed, _) = preprocess(x)
        self.assertEqual(x_preprocessed.shape, x.shape)
        self.assertTrue((x_preprocessed >= clip_values[0]).all())
        self.assertTrue((x_preprocessed <= clip_values[1]).all())
        self.assertFalse((x_preprocessed == x).all())

    def test_three_channels(self):
        if False:
            while True:
                i = 10
        clip_values = (0, 1)
        x = np.random.rand(2, 32, 32, 3)
        x_original = x.copy()
        preprocess = TotalVarMin(clip_values=clip_values)
        (x_preprocessed, _) = preprocess(x)
        self.assertEqual(x_preprocessed.shape, x.shape)
        self.assertTrue((x_preprocessed >= clip_values[0]).all())
        self.assertTrue((x_preprocessed <= clip_values[1]).all())
        self.assertFalse((x_preprocessed == x).all())
        self.assertAlmostEqual(float(np.max(np.abs(x_original - x))), 0.0, delta=1e-05)

    def test_failure_feature_vectors(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.random.rand(10, 3)
        preprocess = TotalVarMin()
        with self.assertRaises(ValueError) as context:
            preprocess(x)
        self.assertIn('Feature vectors detected.', str(context.exception))

    def test_check_params(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            _ = TotalVarMin(prob=-1)
        with self.assertRaises(ValueError):
            _ = TotalVarMin(norm=-1)
        with self.assertRaises(ValueError):
            _ = TotalVarMin(solver='solver')
        with self.assertRaises(ValueError):
            _ = TotalVarMin(max_iter=-1)
        with self.assertRaises(ValueError):
            _ = TotalVarMin(clip_values=(0, 1, 2))
        with self.assertRaises(ValueError):
            _ = TotalVarMin(clip_values=(1, 0))
        with self.assertRaises(ValueError):
            _ = TotalVarMin(verbose='False')
if __name__ == '__main__':
    unittest.main()