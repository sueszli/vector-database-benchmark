import unittest
import numpy as np
import paddle.nn.functional as F
from paddle import base
from paddle.io import DataLoader, Dataset

def get_random_images_and_labels(image_shape, label_shape):
    if False:
        i = 10
        return i + 15
    image = np.random.random(size=image_shape).astype('float32')
    label = np.random.random(size=label_shape).astype('int64')
    return (image, label)

def batch_generator_creator(batch_size, batch_num):
    if False:
        print('Hello World!')

    def __reader__():
        if False:
            i = 10
            return i + 15
        for _ in range(batch_num):
            (batch_image, batch_label) = get_random_images_and_labels([batch_size, 784], [batch_size, 1])
            yield (batch_image, batch_label)
    return __reader__

class RandomDataset(Dataset):

    def __init__(self, sample_num):
        if False:
            while True:
                i = 10
        self.sample_num = sample_num

    def __getitem__(self, idx):
        if False:
            while True:
                i = 10
        np.random.seed(idx)
        image = np.random.random([784]).astype('float32')
        label = np.random.randint(0, 9, (1,)).astype('int64')
        return (image, label)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.sample_num

class TestDygraphDataLoaderMmapFdsClear(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.batch_size = 8
        self.batch_num = 100
        self.epoch_num = 2
        self.capacity = 50

    def prepare_data_loader(self):
        if False:
            while True:
                i = 10
        loader = base.io.DataLoader.from_generator(capacity=self.capacity, use_multiprocess=True)
        loader.set_batch_generator(batch_generator_creator(self.batch_size, self.batch_num), places=base.CPUPlace())
        return loader

    def run_one_epoch_with_break(self, loader):
        if False:
            print('Hello World!')
        for (step_id, data) in enumerate(loader()):
            (image, label) = data
            relu = F.relu(image)
            self.assertEqual(image.shape, [self.batch_size, 784])
            self.assertEqual(label.shape, [self.batch_size, 1])
            self.assertEqual(relu.shape, [self.batch_size, 784])
            if step_id == 30:
                break

    def test_data_loader_break(self):
        if False:
            print('Hello World!')
        with base.dygraph.guard():
            loader = self.prepare_data_loader()
            for _ in range(self.epoch_num):
                self.run_one_epoch_with_break(loader)
                break

    def test_data_loader_continue_break(self):
        if False:
            i = 10
            return i + 15
        with base.dygraph.guard():
            loader = self.prepare_data_loader()
            for _ in range(self.epoch_num):
                self.run_one_epoch_with_break(loader)

class TestMultiProcessDataLoaderMmapFdsClear(TestDygraphDataLoaderMmapFdsClear):

    def prepare_data_loader(self):
        if False:
            print('Hello World!')
        place = base.CPUPlace()
        with base.dygraph.guard(place):
            dataset = RandomDataset(self.batch_size * self.batch_num)
            loader = DataLoader(dataset, places=place, batch_size=self.batch_size, drop_last=True, num_workers=2)
            return loader
if __name__ == '__main__':
    unittest.main()