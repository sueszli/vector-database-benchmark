"""
Test encoding/decoding database objects
"""
import glob
import os
import unittest
import tempfile
from io import BytesIO
from PIL import Image
from txtai.embeddings import Embeddings
from utils import Utils

class TestEncoder(unittest.TestCase):
    """
    Encoder tests.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize test data.\n        '
        cls.data = []
        for path in glob.glob(Utils.PATH + '/*jpg'):
            cls.data.append((path, {'object': Image.open(path)}, None))
        cls.embeddings = Embeddings({'method': 'sentence-transformers', 'path': 'sentence-transformers/clip-ViT-B-32', 'content': True, 'objects': 'image'})

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        '\n        Cleanup data.\n        '
        if cls.embeddings:
            cls.embeddings.close()

    def testDefault(self):
        if False:
            while True:
                i = 10
        '\n        Test an index with default encoder\n        '
        try:
            self.embeddings.config['objects'] = True
            for content in ['duckdb', 'sqlite']:
                self.embeddings.config['content'] = content
                data = [(0, {'object': bytearray([1, 2, 3]), 'text': 'default test'}, None)]
                self.embeddings.index(data)
                result = self.embeddings.search('select object from txtai limit 1')[0]
                self.assertEqual(result['object'].getvalue(), bytearray([1, 2, 3]))
        finally:
            self.embeddings.config['objects'] = 'image'
            self.embeddings.config['content'] = True

    def testImages(self):
        if False:
            i = 10
            return i + 15
        '\n        Test an index with image encoder\n        '
        self.embeddings.index(self.data)
        result = self.embeddings.search("select id, object from txtai where similar('universe') limit 1")[0]
        self.assertTrue(result['id'].endswith('stars.jpg'))
        self.assertTrue(isinstance(result['object'], Image.Image))

    def testPickle(self):
        if False:
            return 10
        '\n        Test an index with pickle encoder\n        '
        try:
            self.embeddings.config['objects'] = 'pickle'
            data = [(0, {'object': [1, 2, 3, 4, 5], 'text': 'default test'}, None)]
            self.embeddings.index(data)
            result = self.embeddings.search('select object from txtai limit 1')[0]
            self.assertEqual(result['object'], [1, 2, 3, 4, 5])
        finally:
            self.embeddings.config['objects'] = 'image'

    def testReindex(self):
        if False:
            i = 10
            return i + 15
        '\n        Test reindex with objects\n        '
        self.embeddings.index(self.data)
        self.embeddings.reindex({'method': 'sentence-transformers', 'path': 'sentence-transformers/clip-ViT-B-32'})
        result = self.embeddings.search("select id, object from txtai where similar('universe') limit 1")[0]
        self.assertTrue(result['id'].endswith('stars.jpg'))
        self.assertTrue(isinstance(result['object'], Image.Image))

    def testReindexFunction(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test reindex with objects and a function\n        '
        try:

            def prepare(documents):
                if False:
                    while True:
                        i = 10
                for (uid, data, tags) in documents:
                    yield (uid, Image.open(data), tags)
            self.embeddings.index(self.data)
            self.embeddings.config['objects'] = True
            index = os.path.join(tempfile.gettempdir(), 'objects')
            self.embeddings.save(index)
            self.embeddings.load(index)
            self.embeddings.reindex({'method': 'sentence-transformers', 'path': 'sentence-transformers/clip-ViT-B-32'}, function=prepare)
            result = self.embeddings.search("select id, object from txtai where similar('universe') limit 1")[0]
            self.assertTrue(result['id'].endswith('stars.jpg'))
            self.assertTrue(isinstance(result['object'], BytesIO))
        finally:
            self.embeddings.config['objects'] = 'image'