"""
External module tests
"""
import os
import pickle
import unittest
import numpy as np
from txtai.vectors import ExternalVectors, VectorsFactory

class TestExternalVectors(unittest.TestCase):
    """
    ExternalVectors tests
    """

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create single ExternalVectors instance.\n        '
        cls.model = VectorsFactory.create({'method': 'external'}, None)

    def testIndex(self):
        if False:
            while True:
                i = 10
        '\n        Test indexing with external vectors\n        '
        data = np.random.rand(1000, 768).astype(np.float32)
        documents = [(x, data[x], None) for x in range(1000)]
        (ids, dimension, batches, stream) = self.model.index(documents)
        self.assertEqual(len(ids), 1000)
        self.assertEqual(dimension, 768)
        self.assertEqual(batches, 2)
        self.assertIsNotNone(os.path.exists(stream))
        with open(stream, 'rb') as queue:
            self.assertEqual(pickle.load(queue).shape, (500, 768))

    def testMethod(self):
        if False:
            while True:
                i = 10
        '\n        Test method is derived when transform function passed\n        '
        model = VectorsFactory.create({'transform': lambda x: x}, None)
        self.assertTrue(isinstance(model, ExternalVectors))