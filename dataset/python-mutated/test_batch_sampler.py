import random
import unittest
import numpy as np
from paddle.io import BatchSampler, Dataset, RandomSampler, Sampler, SequenceSampler, SubsetRandomSampler, WeightedRandomSampler
IMAGE_SIZE = 32

class RandomDataset(Dataset):

    def __init__(self, sample_num, class_num):
        if False:
            for i in range(10):
                print('nop')
        self.sample_num = sample_num
        self.class_num = class_num

    def __getitem__(self, idx):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(idx)
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, self.class_num - 1, (1,)).astype('int64')
        return (image, label)

    def __len__(self):
        if False:
            print('Hello World!')
        return self.sample_num

class TestSampler(unittest.TestCase):

    def test_main(self):
        if False:
            i = 10
            return i + 15
        dataset = RandomDataset(100, 10)
        sampler = Sampler(dataset)
        try:
            iter(sampler)
            self.assertTrue(False)
        except NotImplementedError:
            pass

class TestSequenceSampler(unittest.TestCase):

    def test_main(self):
        if False:
            i = 10
            return i + 15
        dataset = RandomDataset(100, 10)
        sampler = SequenceSampler(dataset)
        assert len(sampler) == 100
        for (i, index) in enumerate(iter(sampler)):
            assert i == index

class TestRandomSampler(unittest.TestCase):

    def test_main(self):
        if False:
            i = 10
            return i + 15
        dataset = RandomDataset(100, 10)
        sampler = RandomSampler(dataset)
        assert len(sampler) == 100
        rets = []
        for i in iter(sampler):
            rets.append(i)
        assert tuple(sorted(rets)) == tuple(range(0, 100))

    def test_with_num_samples(self):
        if False:
            print('Hello World!')
        dataset = RandomDataset(100, 10)
        sampler = RandomSampler(dataset, num_samples=50, replacement=True)
        assert len(sampler) == 50
        rets = []
        for i in iter(sampler):
            rets.append(i)
            assert i >= 0 and i < 100

    def test_with_generator(self):
        if False:
            while True:
                i = 10
        dataset = RandomDataset(100, 10)
        generator = iter(range(0, 60))
        sampler = RandomSampler(dataset, generator=generator)
        assert len(sampler) == 100
        rets = []
        for i in iter(sampler):
            rets.append(i)
        assert tuple(sorted(rets)) == tuple(range(0, 60))

    def test_with_generator_num_samples(self):
        if False:
            return 10
        dataset = RandomDataset(100, 10)
        generator = iter(range(0, 60))
        sampler = RandomSampler(dataset, generator=generator, num_samples=50, replacement=True)
        assert len(sampler) == 50
        rets = []
        for i in iter(sampler):
            rets.append(i)
        assert tuple(sorted(rets)) == tuple(range(0, 50))

class TestSubsetRandomSampler(unittest.TestCase):

    def test_main(self):
        if False:
            print('Hello World!')
        indices = list(range(100))
        random.shuffle(indices)
        indices = indices[:30]
        sampler = SubsetRandomSampler(indices)
        assert len(sampler) == len(indices)
        hints = {i: 0 for i in indices}
        for index in iter(sampler):
            hints[index] += 1
        for h in hints.values():
            assert h == 1

    def test_raise(self):
        if False:
            print('Hello World!')
        try:
            sampler = SubsetRandomSampler([])
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

class TestBatchSampler(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.num_samples = 1000
        self.num_classes = 10
        self.batch_size = 32
        self.shuffle = False
        self.drop_last = False

    def init_batch_sampler(self):
        if False:
            while True:
                i = 10
        dataset = RandomDataset(self.num_samples, self.num_classes)
        bs = BatchSampler(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last)
        return bs

    def test_main(self):
        if False:
            for i in range(10):
                print('nop')
        bs = self.init_batch_sampler()
        bs_len = (self.num_samples + int(not self.drop_last) * (self.batch_size - 1)) // self.batch_size
        self.assertTrue(bs_len == len(bs))
        if not self.shuffle:
            index = 0
            for indices in bs:
                for idx in indices:
                    self.assertTrue(index == idx)
                    index += 1

class TestBatchSamplerDropLast(TestBatchSampler):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.num_samples = 1000
        self.num_classes = 10
        self.batch_size = 32
        self.shuffle = False
        self.drop_last = True

class TestBatchSamplerShuffle(TestBatchSampler):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.num_samples = 1000
        self.num_classes = 10
        self.batch_size = 32
        self.shuffle = True
        self.drop_last = True

class TestBatchSamplerWithSampler(TestBatchSampler):

    def init_batch_sampler(self):
        if False:
            return 10
        dataset = RandomDataset(1000, 10)
        sampler = SequenceSampler(dataset)
        bs = BatchSampler(sampler=sampler, batch_size=self.batch_size, drop_last=self.drop_last)
        return bs

class TestBatchSamplerWithSamplerDropLast(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.num_samples = 1000
        self.num_classes = 10
        self.batch_size = 32
        self.shuffle = False
        self.drop_last = True

class TestBatchSamplerWithSamplerShuffle(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.num_samples = 1000
        self.num_classes = 10
        self.batch_size = 32
        self.shuffle = True
        self.drop_last = True

    def test_main(self):
        if False:
            while True:
                i = 10
        try:
            dataset = RandomDataset(self.num_samples, self.num_classes)
            sampler = RandomSampler(dataset)
            bs = BatchSampler(sampler=sampler, shuffle=self.shuffle, batch_size=self.batch_size, drop_last=self.drop_last)
            self.assertTrue(False)
        except AssertionError:
            pass

class TestWeightedRandomSampler(unittest.TestCase):

    def init_probs(self, total, pos):
        if False:
            i = 10
            return i + 15
        pos_probs = np.random.random((pos,)).astype('float32')
        probs = np.zeros((total,)).astype('float32')
        probs[:pos] = pos_probs
        np.random.shuffle(probs)
        return probs

    def test_replacement(self):
        if False:
            for i in range(10):
                print('nop')
        probs = self.init_probs(20, 10)
        sampler = WeightedRandomSampler(probs, 30, True)
        assert len(sampler) == 30
        for idx in iter(sampler):
            assert probs[idx] > 0.0

    def test_no_replacement(self):
        if False:
            print('Hello World!')
        probs = self.init_probs(20, 10)
        sampler = WeightedRandomSampler(probs, 10, False)
        assert len(sampler) == 10
        idxs = []
        for idx in iter(sampler):
            assert probs[idx] > 0.0
            idxs.append(idx)
        assert len(set(idxs)) == len(idxs)

    def test_assert(self):
        if False:
            return 10
        probs = np.zeros((10,)).astype('float32')
        sampler = WeightedRandomSampler(probs, 10, True)
        try:
            for idx in iter(sampler):
                pass
            self.assertTrue(False)
        except AssertionError:
            self.assertTrue(True)
        probs = self.init_probs(10, 5)
        sampler = WeightedRandomSampler(probs, 10, False)
        try:
            for idx in iter(sampler):
                pass
            self.assertTrue(False)
        except AssertionError:
            self.assertTrue(True)
        probs = -1.0 * np.ones((10,)).astype('float32')
        sampler = WeightedRandomSampler(probs, 10, True)
        try:
            for idx in iter(sampler):
                pass
            self.assertTrue(False)
        except AssertionError:
            self.assertTrue(True)

    def test_raise(self):
        if False:
            while True:
                i = 10
        probs = self.init_probs(10, 5)
        try:
            sampler = WeightedRandomSampler(probs, 2.3, True)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
        probs = self.init_probs(10, 5)
        try:
            sampler = WeightedRandomSampler(probs, -1, True)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
        probs = self.init_probs(10, 5)
        try:
            sampler = WeightedRandomSampler(probs, 5, 5)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
if __name__ == '__main__':
    unittest.main()