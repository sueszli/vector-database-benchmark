import unittest
import numpy as np
import paddle.nn.functional as F
from paddle import base
from paddle.base.reader import use_pinned_memory

def get_random_images_and_labels(image_shape, label_shape):
    if False:
        print('Hello World!')
    image = np.random.random(size=image_shape).astype('float32')
    label = np.random.random(size=label_shape).astype('int64')
    return (image, label)

def sample_generator_creator(batch_size, batch_num):
    if False:
        i = 10
        return i + 15

    def __reader__():
        if False:
            i = 10
            return i + 15
        for _ in range(batch_num * batch_size):
            (image, label) = get_random_images_and_labels([784], [1])
            yield (image, label)
    return __reader__

class TestDygraphDataLoader(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.batch_size = 8
        self.batch_num = 4
        self.epoch_num = 1
        self.capacity = 5

    def iter_loader_data(self, loader):
        if False:
            i = 10
            return i + 15
        for _ in range(self.epoch_num):
            for (image, label) in loader():
                relu = F.relu(image)
                self.assertEqual(image.shape, [self.batch_size, 784])
                self.assertEqual(label.shape, [self.batch_size, 1])
                self.assertEqual(relu.shape, [self.batch_size, 784])

    def test_single_process_loader(self):
        if False:
            for i in range(10):
                print('nop')
        with base.dygraph.guard():
            loader = base.io.DataLoader.from_generator(capacity=self.capacity, iterable=False, use_multiprocess=False)
            loader.set_sample_generator(sample_generator_creator(self.batch_size, self.batch_num), batch_size=self.batch_size, places=base.CPUPlace())
            self.iter_loader_data(loader)

    def test_multi_process_loader(self):
        if False:
            i = 10
            return i + 15
        with base.dygraph.guard():
            loader = base.io.DataLoader.from_generator(capacity=self.capacity, use_multiprocess=True)
            loader.set_sample_generator(sample_generator_creator(self.batch_size, self.batch_num), batch_size=self.batch_size, places=base.CPUPlace())
            self.iter_loader_data(loader)

    def test_generator_no_places(self):
        if False:
            for i in range(10):
                print('nop')
        with base.dygraph.guard():
            loader = base.io.DataLoader.from_generator(capacity=self.capacity)
            loader.set_sample_generator(sample_generator_creator(self.batch_size, self.batch_num), batch_size=self.batch_size)
            self.iter_loader_data(loader)

    def test_set_pin_memory(self):
        if False:
            return 10
        with base.dygraph.guard():
            use_pinned_memory(False)
            loader = base.io.DataLoader.from_generator(capacity=self.capacity, iterable=False, use_multiprocess=False)
            loader.set_sample_generator(sample_generator_creator(self.batch_size, self.batch_num), batch_size=self.batch_size, places=base.CPUPlace())
            self.iter_loader_data(loader)
            use_pinned_memory(True)
if __name__ == '__main__':
    unittest.main()