"""
Models module tests
"""
import unittest
from unittest.mock import patch
import torch
from txtai.models import Models, ClsPooling, MeanPooling, PoolingFactory

class TestModels(unittest.TestCase):
    """
    Models tests.
    """

    @patch('torch.cuda.is_available')
    def testDeviceid(self, cuda):
        if False:
            return 10
        '\n        Test the deviceid method\n        '
        cuda.return_value = True
        self.assertEqual(Models.deviceid(True), 0)
        self.assertEqual(Models.deviceid(False), -1)
        self.assertEqual(Models.deviceid(0), 0)
        self.assertEqual(Models.deviceid(1), 1)
        self.assertEqual(Models.deviceid(torch.device('cpu')), torch.device('cpu'))
        cuda.return_value = False
        self.assertEqual(Models.deviceid(True), -1)
        self.assertEqual(Models.deviceid(False), -1)
        self.assertEqual(Models.deviceid(0), -1)
        self.assertEqual(Models.deviceid(1), -1)

    def testDevice(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests the device method\n        '
        self.assertEqual(Models.device('cpu'), torch.device('cpu'))
        self.assertEqual(Models.device(torch.device('cpu')), torch.device('cpu'))

    def testPooling(self):
        if False:
            while True:
                i = 10
        '\n        Tests pooling methods\n        '
        device = Models.deviceid(True)
        pooling = PoolingFactory.create({'path': 'sentence-transformers/nli-mpnet-base-v2', 'device': device})
        self.assertEqual(type(pooling), MeanPooling)
        pooling = PoolingFactory.create({'method': 'meanpooling', 'path': 'flax-sentence-embeddings/multi-qa_v1-MiniLM-L6-cls_dot', 'device': device})
        self.assertEqual(type(pooling), MeanPooling)
        pooling = PoolingFactory.create({'path': 'flax-sentence-embeddings/multi-qa_v1-MiniLM-L6-cls_dot', 'device': device})
        self.assertEqual(type(pooling), ClsPooling)
        pooling = PoolingFactory.create({'method': 'clspooling', 'path': 'sentence-transformers/nli-mpnet-base-v2', 'device': device})
        self.assertEqual(type(pooling), ClsPooling)
        self.assertEqual(pooling.encode(['test'])[0].shape, (768,))