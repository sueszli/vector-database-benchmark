import unittest
import numpy
from chainer.backends import cuda
from chainer import datasets
from chainer import testing
from chainer.testing import attr

class TestTupleDataset(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.x0 = numpy.random.rand(3, 4)
        self.x1 = numpy.random.rand(3, 5)
        self.z0 = numpy.random.rand(4, 4)

    def check_tuple_dataset(self, x0, x1):
        if False:
            i = 10
            return i + 15
        td = datasets.TupleDataset(x0, x1)
        self.assertEqual(len(td), len(x0))
        for i in range(len(x0)):
            example = td[i]
            self.assertEqual(len(example), 2)
            numpy.testing.assert_array_equal(cuda.to_cpu(example[0]), cuda.to_cpu(x0[i]))
            numpy.testing.assert_array_equal(cuda.to_cpu(example[1]), cuda.to_cpu(x1[i]))
        example_range = td[0:len(x0)]
        for i in range(len(x0)):
            example = example_range[i]
            self.assertEqual(len(example), 2)
            numpy.testing.assert_array_equal(cuda.to_cpu(example[0]), cuda.to_cpu(x0[i]))
            numpy.testing.assert_array_equal(cuda.to_cpu(example[1]), cuda.to_cpu(x1[i]))

    def test_tuple_dataset_cpu(self):
        if False:
            while True:
                i = 10
        self.check_tuple_dataset(self.x0, self.x1)

    @attr.gpu
    def test_tuple_dataset_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_tuple_dataset(cuda.to_gpu(self.x0), cuda.to_gpu(self.x1))

    def test_tuple_dataset_len_mismatch(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError):
            datasets.TupleDataset(self.x0, self.z0)

    def test_tuple_dataset_overrun(self):
        if False:
            while True:
                i = 10
        td = datasets.TupleDataset(self.x0, self.x1)
        with self.assertRaises(IndexError):
            td[3]
testing.run_module(__name__, __file__)