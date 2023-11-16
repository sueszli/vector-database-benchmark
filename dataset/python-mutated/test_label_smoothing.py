from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import unittest
import numpy as np
from art.defences.preprocessor import LabelSmoothing
from tests.utils import master_seed
logger = logging.getLogger(__name__)

class TestLabelSmoothing(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        master_seed(seed=1234)

    def test_default(self):
        if False:
            while True:
                i = 10
        (m, n) = (1000, 20)
        y = np.zeros((m, n))
        y[range(m), np.random.choice(range(n), m)] = 1.0
        ls = LabelSmoothing()
        (_, y_smooth) = ls(None, y)
        self.assertTrue(np.isclose(np.sum(y_smooth, axis=1), np.ones(m)).all())
        self.assertTrue((np.max(y_smooth, axis=1) == np.ones(m) * 0.9).all())

    def test_customizing(self):
        if False:
            for i in range(10):
                print('nop')
        (m, n) = (1000, 20)
        y = np.zeros((m, n))
        y[range(m), np.random.choice(range(n), m)] = 1.0
        ls = LabelSmoothing(max_value=1.0 / n)
        (_, y_smooth) = ls(None, y)
        self.assertTrue(np.isclose(np.sum(y_smooth, axis=1), np.ones(m)).all())
        self.assertTrue((np.max(y_smooth, axis=1) == np.ones(m) / n).all())
        self.assertTrue(np.isclose(y_smooth, np.ones((m, n)) / n).all())

    def test_check_params(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            _ = LabelSmoothing(max_value=-1)
if __name__ == '__main__':
    unittest.main()