"""
ANN module tests
"""
import os
import tempfile
import unittest
import numpy as np
from txtai.ann import ANNFactory, ANN

class TestANN(unittest.TestCase):
    """
    ANN tests.
    """

    def testAnnoy(self):
        if False:
            i = 10
            return i + 15
        '\n        Test Annoy backend\n        '
        self.runTests('annoy', None, False)

    def testAnnoyCustom(self):
        if False:
            i = 10
            return i + 15
        '\n        Test Annoy backend with custom settings\n        '
        self.runTests('annoy', {'annoy': {'ntrees': 2, 'searchk': 1}}, False)

    def testCustomBackend(self):
        if False:
            while True:
                i = 10
        '\n        Test resolving a custom backend\n        '
        self.runTests('txtai.ann.Faiss')

    def testCustomBackendNotFound(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test resolving an unresolvable backend\n        '
        with self.assertRaises(ImportError):
            ANNFactory.create({'backend': 'notfound.ann'})

    def testFaiss(self):
        if False:
            i = 10
            return i + 15
        '\n        Test Faiss backend\n        '
        self.runTests('faiss')

    def testFaissCustom(self):
        if False:
            return 10
        '\n        Test Faiss backend with custom settings\n        '
        self.runTests('faiss', {'faiss': {'nprobe': 2, 'components': 'PCA16,IDMap,SQ8', 'sample': 1.0}}, False)
        self.runTests('faiss', {'faiss': {'components': 'IVF,SQ8'}}, False)

    @unittest.skipIf(os.name == 'nt', 'mmap not supported on Windows')
    def testFaissMmap(self):
        if False:
            print('Hello World!')
        '\n        Test Faiss backend with mmap enabled\n        '
        self.runTests('faiss', {'faiss': {'mmap': True}}, False)

    def testHnsw(self):
        if False:
            return 10
        '\n        Test Hnswlib backend\n        '
        self.runTests('hnsw')

    def testHnswCustom(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test Hnswlib backend with custom settings\n        '
        self.runTests('hnsw', {'hnsw': {'efconstruction': 100, 'm': 4, 'randomseed': 0, 'efsearch': 5}})

    def testNotImplemented(self):
        if False:
            i = 10
            return i + 15
        '\n        Test exceptions for non-implemented methods\n        '
        ann = ANN({})
        self.assertRaises(NotImplementedError, ann.load, None)
        self.assertRaises(NotImplementedError, ann.index, None)
        self.assertRaises(NotImplementedError, ann.append, None)
        self.assertRaises(NotImplementedError, ann.delete, None)
        self.assertRaises(NotImplementedError, ann.search, None, None)
        self.assertRaises(NotImplementedError, ann.count)
        self.assertRaises(NotImplementedError, ann.save, None)

    def testNumPy(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test NumPy backend\n        '
        self.runTests('numpy')

    def testTorch(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test Torch backend\n        '
        self.runTests('torch')

    def runTests(self, name, params=None, update=True):
        if False:
            while True:
                i = 10
        '\n        Runs a series of standard backend tests.\n\n        Args:\n            name: backend name\n            params: additional config parameters\n            update: If append/delete options should be tested\n        '
        self.assertEqual(self.backend(name, params).config['backend'], name)
        self.assertEqual(self.save(name, params).count(), 10000)
        if update:
            self.assertEqual(self.append(name, params, 500).count(), 10500)
            self.assertEqual(self.delete(name, params, [0, 1]).count(), 9998)
            self.assertEqual(self.delete(name, params, [100000]).count(), 10000)
        self.assertGreater(self.search(name, params), 0)

    def backend(self, name, params=None, length=10000):
        if False:
            i = 10
            return i + 15
        '\n        Test a backend.\n\n        Args:\n            name: backend name\n            params: additional config parameters\n            length: number of rows to generate\n\n        Returns:\n            ANN model\n        '
        data = np.random.rand(length, 300).astype(np.float32)
        self.normalize(data)
        config = {'backend': name, 'dimensions': data.shape[1]}
        if params:
            config.update(params)
        model = ANNFactory.create(config)
        model.index(data)
        return model

    def append(self, name, params=None, length=500):
        if False:
            print('Hello World!')
        '\n        Appends new data to index.\n\n        Args:\n            name: backend name\n            params: additional config parameters\n            length: number of rows to generate\n\n        Returns:\n            ANN model\n        '
        model = self.backend(name, params)
        data = np.random.rand(length, 300).astype(np.float32)
        self.normalize(data)
        model.append(data)
        return model

    def delete(self, name, params=None, ids=None):
        if False:
            while True:
                i = 10
        '\n        Deletes data from index.\n\n        Args:\n            name: backend name\n            params: additional config parameters\n            ids: ids to delete\n\n        Returns:\n            ANN model\n        '
        model = self.backend(name, params)
        model.delete(ids)
        return model

    def save(self, name, params=None):
        if False:
            while True:
                i = 10
        '\n        Test save/load.\n\n        Args:\n            name: backend name\n            params: additional config parameters\n\n        Returns:\n            ANN model\n        '
        model = self.backend(name, params)
        index = os.path.join(tempfile.gettempdir(), 'ann')
        model.save(index)
        model.load(index)
        return model

    def search(self, name, params=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test ANN search.\n\n        Args:\n            name: backend name\n            params: additional config parameters\n\n        Returns:\n            search results\n        '
        model = self.backend(name, params)
        query = np.random.rand(300).astype(np.float32)
        self.normalize(query)
        return model.search(np.array([query]), 1)[0][0][1]

    def normalize(self, embeddings):
        if False:
            print('Hello World!')
        '\n        Normalizes embeddings using L2 normalization. Operation applied directly on array.\n\n        Args:\n            embeddings: input embeddings matrix\n        '
        if len(embeddings.shape) > 1:
            embeddings /= np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        else:
            embeddings /= np.linalg.norm(embeddings)