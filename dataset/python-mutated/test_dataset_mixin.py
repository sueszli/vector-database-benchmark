import unittest
import numpy
from chainer import dataset
from chainer import testing

class SimpleDataset(dataset.DatasetMixin):

    def __init__(self, values):
        if False:
            return 10
        self.values = values

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.values)

    def get_example(self, i):
        if False:
            print('Hello World!')
        return self.values[i]

class TestDatasetMixin(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.ds = SimpleDataset([1, 2, 3, 4, 5])

    def test_getitem(self):
        if False:
            print('Hello World!')
        for i in range(len(self.ds.values)):
            self.assertEqual(self.ds[i], self.ds.values[i])

    def test_slice(self):
        if False:
            while True:
                i = 10
        ds = self.ds
        self.assertEqual(ds[:], ds.values)
        self.assertEqual(ds[1:], ds.values[1:])
        self.assertEqual(ds[2:], ds.values[2:])
        self.assertEqual(ds[1:4], ds.values[1:4])
        self.assertEqual(ds[0:4], ds.values[0:4])
        self.assertEqual(ds[1:5], ds.values[1:5])
        self.assertEqual(ds[:-1], ds.values[:-1])
        self.assertEqual(ds[1:-2], ds.values[1:-2])
        self.assertEqual(ds[-4:-1], ds.values[-4:-1])
        self.assertEqual(ds[::-1], ds.values[::-1])
        self.assertEqual(ds[4::-1], ds.values[4::-1])
        self.assertEqual(ds[:2:-1], ds.values[:2:-1])
        self.assertEqual(ds[-1::-1], ds.values[-1::-1])
        self.assertEqual(ds[:-3:-1], ds.values[:-3:-1])
        self.assertEqual(ds[-1:-3:-1], ds.values[-1:-3:-1])
        self.assertEqual(ds[4:1:-1], ds.values[4:1:-1])
        self.assertEqual(ds[-1:1:-1], ds.values[-1:1:-1])
        self.assertEqual(ds[4:-3:-1], ds.values[4:-3:-1])
        self.assertEqual(ds[-2:-4:-1], ds.values[-2:-4:-1])
        self.assertEqual(ds[::2], ds.values[::2])
        self.assertEqual(ds[1::2], ds.values[1::2])
        self.assertEqual(ds[:3:2], ds.values[:3:2])
        self.assertEqual(ds[1:4:2], ds.values[1:4:2])
        self.assertEqual(ds[::-2], ds.values[::-2])
        self.assertEqual(ds[:10], ds.values[:10])

    def test_advanced_indexing(self):
        if False:
            i = 10
            return i + 15
        ds = self.ds
        self.assertEqual(ds[[1, 2]], [ds.values[1], ds.values[2]])
        self.assertEqual(ds[[1, 2]], ds[1:3])
        self.assertEqual(ds[[4, 0]], [ds.values[4], ds.values[0]])
        self.assertEqual(ds[[4]], [ds.values[4]])
        self.assertEqual(ds[[4, 1, 3, 2, 2, 1]], [ds.values[4], ds.values[1], ds.values[3], ds.values[2], ds.values[2], ds.values[1]])
        self.assertEqual(ds[[-2, -1]], [ds.values[-2], ds.values[-1]])
        self.assertEqual(ds[numpy.asarray([1, 2, 3])], ds[1:4])

    def test_large_dataset(self):
        if False:
            for i in range(10):
                print('nop')
        ds = SimpleDataset(list(numpy.arange(1000000)))
        self.assertEqual(ds[3453], ds.values[3453])
        self.assertEqual(ds[:], ds.values)
        self.assertEqual(ds[2:987654:7], ds.values[2:987654:7])
        self.assertEqual(ds[::-3], ds.values[::-3])
        for i in range(100):
            self.assertEqual(ds[i * 4096:(i + 1) * 4096], ds.values[i * 4096:(i + 1) * 4096])
testing.run_module(__name__, __file__)