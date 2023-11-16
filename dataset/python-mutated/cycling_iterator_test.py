import unittest
from torch.distributed.elastic.utils.data import CyclingIterator

class CyclingIteratorTest(unittest.TestCase):

    def generator(self, epoch, stride, max_epochs):
        if False:
            while True:
                i = 10
        return iter([stride * epoch + i for i in range(0, stride)])

    def test_cycling_iterator(self):
        if False:
            for i in range(10):
                print('nop')
        stride = 3
        max_epochs = 90

        def generator_fn(epoch):
            if False:
                while True:
                    i = 10
            return self.generator(epoch, stride, max_epochs)
        it = CyclingIterator(n=max_epochs, generator_fn=generator_fn)
        for i in range(0, stride * max_epochs):
            self.assertEqual(i, next(it))
        with self.assertRaises(StopIteration):
            next(it)

    def test_cycling_iterator_start_epoch(self):
        if False:
            i = 10
            return i + 15
        stride = 3
        max_epochs = 2
        start_epoch = 1

        def generator_fn(epoch):
            if False:
                print('Hello World!')
            return self.generator(epoch, stride, max_epochs)
        it = CyclingIterator(max_epochs, generator_fn, start_epoch)
        for i in range(stride * start_epoch, stride * max_epochs):
            self.assertEqual(i, next(it))
        with self.assertRaises(StopIteration):
            next(it)