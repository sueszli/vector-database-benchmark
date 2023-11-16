"""
Vectors module tests
"""
import os
import pickle
import unittest
import numpy as np
from txtai.vectors import VectorsFactory

class TestTransformersVectors(unittest.TestCase):
    """
    TransformerVectors tests
    """

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        '\n        Create single TransformersVectors instance.\n        '
        cls.model = VectorsFactory.create({'path': 'sentence-transformers/nli-mpnet-base-v2'}, None)

    def testIndex(self):
        if False:
            print('Hello World!')
        '\n        Test transformers indexing\n        '
        documents = [(x, 'This is a test', None) for x in range(1000)]
        (ids, dimension, batches, stream) = self.model.index(documents)
        self.assertEqual(len(ids), 1000)
        self.assertEqual(dimension, 768)
        self.assertEqual(batches, 2)
        self.assertIsNotNone(os.path.exists(stream))
        with open(stream, 'rb') as queue:
            self.assertEqual(pickle.load(queue).shape, (500, 768))

    def testSentenceTransformers(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test creating a model with sentence transformers\n        '
        model = VectorsFactory.create({'method': 'sentence-transformers', 'path': 'paraphrase-MiniLM-L3-v2'}, None)
        self.assertEqual(model.transform((0, 'This is a test', None)).shape, (384,))

    def testText(self):
        if False:
            while True:
                i = 10
        '\n        Test transformers text conversion\n        '
        self.model.tokenize = True
        self.assertEqual(self.model.prepare('Y 123 This is a test!'), 'test')
        self.assertEqual(self.model.prepare(['This', 'is', 'a', 'test']), 'This is a test')
        self.model.tokenize = False
        self.assertEqual(self.model.prepare('Y 123 This is a test!'), 'Y 123 This is a test!')
        self.assertEqual(self.model.prepare(['This', 'is', 'a', 'test']), 'This is a test')

    def testTransform(self):
        if False:
            i = 10
            return i + 15
        '\n        Test transformers transform\n        '
        documents = [(0, 'This is a test and has no tokenization', None), (1, 'test tokenization', None)]
        self.model.tokenize = True
        embeddings1 = [self.model.transform(d) for d in documents]
        self.model.tokenize = False
        embeddings2 = [self.model.transform(d) for d in documents]
        self.assertFalse(np.array_equal(embeddings1[0], embeddings2[0]))
        self.assertTrue(np.array_equal(embeddings1[1], embeddings2[1]))

    def testTransformArray(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test transformers skips transforming NumPy arrays\n        '
        data1 = np.random.rand(5, 5).astype(np.float32)
        data2 = self.model.transform((0, data1, None))
        self.assertTrue(np.array_equal(data1, data2))

    def testTransformLong(self):
        if False:
            return 10
        '\n        Test transformers transform on long text\n        '
        documents = [(0, 'This is long text ' * 512, None), (1, 'This is short text', None)]
        embeddings = [self.model.transform(d) for d in documents]
        self.assertIsNotNone(embeddings)