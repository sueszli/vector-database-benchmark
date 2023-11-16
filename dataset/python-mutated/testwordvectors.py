"""
Vectors module tests
"""
import os
import pickle
import tempfile
import unittest
from unittest.mock import patch
from txtai.vectors import WordVectors, VectorsFactory

class TestWordVectors(unittest.TestCase):
    """
    Vectors tests.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        '\n        Test a WordVectors build.\n        '
        path = os.path.join(tempfile.gettempdir(), 'vectors')
        cls.path = path + '.magnitude'
        WordVectors.build('README.md', 10, 3, path)

    def testBlocking(self):
        if False:
            i = 10
            return i + 15
        '\n        Test blocking load of vector model\n        '
        config = {'path': self.path}
        model = VectorsFactory.create(config, None)
        self.assertFalse(model.initialized)
        config['ids'] = ['0', '1']
        config['dimensions'] = 10
        model = VectorsFactory.create(config, None)
        self.assertTrue(model.initialized)

    @patch('os.cpu_count')
    def testIndex(self, cpucount):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test word vectors indexing\n        '
        cpucount.return_value = 1
        documents = [(x, 'This is a test', None) for x in range(1000)]
        model = VectorsFactory.create({'path': self.path, 'parallel': True}, None)
        (ids, dimension, batches, stream) = model.index(documents)
        self.assertEqual(len(ids), 1000)
        self.assertEqual(dimension, 10)
        self.assertEqual(batches, 1000)
        self.assertIsNotNone(os.path.exists(stream))
        with open(stream, 'rb') as queue:
            self.assertEqual(pickle.load(queue).shape, (1, 10))

    @patch('os.cpu_count')
    def testIndexBatch(self, cpucount):
        if False:
            while True:
                i = 10
        '\n        Test word vectors indexing with batch size set\n        '
        cpucount.return_value = 1
        documents = [(x, 'This is a test', None) for x in range(1000)]
        model = VectorsFactory.create({'path': self.path, 'parallel': True}, None)
        (ids, dimension, batches, stream) = model.index(documents, 512)
        self.assertEqual(len(ids), 1000)
        self.assertEqual(dimension, 10)
        self.assertEqual(batches, 2)
        self.assertIsNotNone(os.path.exists(stream))
        with open(stream, 'rb') as queue:
            self.assertEqual(pickle.load(queue).shape, (512, 10))
            self.assertEqual(pickle.load(queue).shape, (488, 10))

    def testIndexSerial(self):
        if False:
            while True:
                i = 10
        '\n        Test word vector indexing in single process mode\n        '
        documents = [(x, 'This is a test', None) for x in range(1000)]
        model = VectorsFactory.create({'path': self.path, 'parallel': False}, None)
        (ids, dimension, batches, stream) = model.index(documents)
        self.assertEqual(len(ids), 1000)
        self.assertEqual(dimension, 10)
        self.assertEqual(batches, 1000)
        self.assertIsNotNone(os.path.exists(stream))
        with open(stream, 'rb') as queue:
            self.assertEqual(pickle.load(queue).shape, (1, 10))

    def testIndexSerialBatch(self):
        if False:
            return 10
        '\n        Test word vector indexing in single process mode with batch size set\n        '
        documents = [(x, 'This is a test', None) for x in range(1000)]
        model = VectorsFactory.create({'path': self.path, 'parallel': False}, None)
        (ids, dimension, batches, stream) = model.index(documents, 512)
        self.assertEqual(len(ids), 1000)
        self.assertEqual(dimension, 10)
        self.assertEqual(batches, 2)
        self.assertIsNotNone(os.path.exists(stream))
        with open(stream, 'rb') as queue:
            self.assertEqual(pickle.load(queue).shape, (512, 10))
            self.assertEqual(pickle.load(queue).shape, (488, 10))

    def testLookup(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test word vector lookup\n        '
        model = VectorsFactory.create({'path': self.path}, None)
        self.assertEqual(model.lookup(['txtai', 'embeddings', 'sentence']).shape, (3, 10))

    def testNoExist(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test loading model that doesn't exist\n        "
        with self.assertRaises(IOError):
            VectorsFactory.create({'method': 'words', 'path': os.path.join(tempfile.gettempdir(), 'noexist')}, None)

    def testTransform(self):
        if False:
            print('Hello World!')
        '\n        Test word vector transform\n        '
        model = VectorsFactory.create({'path': self.path}, None)
        self.assertEqual(len(model.transform((None, ['txtai'], None))), 10)