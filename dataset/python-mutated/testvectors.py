"""
Vectors module tests
"""
import unittest
import numpy as np
from txtai.vectors import Vectors

class TestVectors(unittest.TestCase):
    """
    Vectors tests.
    """

    def testNotImplemented(self):
        if False:
            i = 10
            return i + 15
        '\n        Test exceptions for non-implemented methods\n        '
        vectors = Vectors(None, None)
        self.assertRaises(NotImplementedError, vectors.load, None)
        self.assertRaises(NotImplementedError, vectors.encode, None)

    def testNormalize(self):
        if False:
            print('Hello World!')
        '\n        Test batch normalize and single input normalize are equal\n        '
        vectors = Vectors(None, None)
        data1 = np.random.rand(5, 5).astype(np.float32)
        data2 = data1.copy()
        original = data1.copy()
        vectors.normalize(data1)
        for x in data2:
            vectors.normalize(x)
        self.assertTrue(np.allclose(data1, data2))
        self.assertFalse(np.allclose(data1, original))