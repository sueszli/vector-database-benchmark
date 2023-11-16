"""Tests for random_crop."""
import numpy as np
from tensorflow.python.framework import test_util
from tensorflow.python.ops import random_crop_ops
from tensorflow.python.platform import test

class RandomCropTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testNoOp(self):
        if False:
            print('Hello World!')
        for shape in ((2, 1, 1), (2, 1, 3), (4, 5, 3)):
            value = np.arange(0, np.prod(shape), dtype=np.int32).reshape(shape)
            with self.cached_session():
                crop = random_crop_ops.random_crop(value, shape).eval()
                self.assertAllEqual(crop, value)

    def testContains(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            shape = (3, 5, 7)
            target = (2, 3, 4)
            value = np.random.randint(1000000, size=shape)
            value_set = set((tuple(value[i:i + 2, j:j + 3, k:k + 4].ravel()) for i in range(2) for j in range(3) for k in range(4)))
            crop = random_crop_ops.random_crop(value, size=target)
            for _ in range(20):
                y = self.evaluate(crop)
                self.assertAllEqual(y.shape, target)
                self.assertTrue(tuple(y.ravel()) in value_set)

    @test_util.run_deprecated_v1
    def testRandomization(self):
        if False:
            print('Hello World!')
        num_samples = 1000
        shape = [5, 4, 1]
        size = np.prod(shape)
        single = [1, 1, 1]
        value = np.arange(size).reshape(shape)
        with self.cached_session():
            crop = random_crop_ops.random_crop(value, single, seed=7)
            counts = np.zeros(size, dtype=np.int32)
            for _ in range(num_samples):
                y = self.evaluate(crop)
                self.assertAllEqual(y.shape, single)
                counts[y] += 1
        mean = np.repeat(num_samples / size, size)
        four_stddev = 4.0 * np.sqrt(mean)
        self.assertAllClose(counts, mean, atol=four_stddev)

class StatelessRandomCropTest(test.TestCase):

    def testNoOp(self):
        if False:
            print('Hello World!')
        for shape in ((2, 1, 1), (2, 1, 3), (4, 5, 3)):
            value = np.arange(0, np.prod(shape), dtype=np.int32).reshape(shape)
            crop = random_crop_ops.stateless_random_crop(value, shape, seed=(1, 2))
            self.evaluate(crop)
            self.assertAllEqual(crop, value)

    def testContains(self):
        if False:
            i = 10
            return i + 15
        with test_util.use_gpu():
            shape = (3, 5, 7)
            target = (2, 3, 4)
            value = np.random.randint(1000000, size=shape)
            iterations = 10
            value_set = set((tuple(value[i:i + 2, j:j + 3, k:k + 4].ravel()) for i in range(2) for j in range(3) for k in range(4)))
            test_seeds = [tuple(map(lambda x, i=i: x + 1 * i, t)) for (i, t) in enumerate(((1, 2) for _ in range(iterations)))]
            for seed in test_seeds:
                crop = random_crop_ops.stateless_random_crop(value, size=target, seed=seed)
                y = self.evaluate(crop)
                self.assertAllEqual(y.shape, target)
                self.assertIn(tuple(y.ravel()), value_set)

    def testRandomization(self):
        if False:
            i = 10
            return i + 15
        with test_util.use_gpu():
            shape = [5, 4, 1]
            size = np.prod(shape)
            single = [1, 1, 1]
            value = np.arange(size).reshape(shape)
            iterations = 5
            num_samples = 5
            test_seed = (1, 2)
            observations = [[] for _ in range(iterations)]
            for observation in observations:
                crop = random_crop_ops.stateless_random_crop(value, single, seed=test_seed)
                counts = np.zeros(size, dtype=np.int32)
                for _ in range(num_samples):
                    y = self.evaluate(crop)
                    self.assertAllEqual(y.shape, single)
                    counts[y] += 1
                observation.append(counts)
            for i in range(1, iterations):
                self.assertAllEqual(observations[0], observations[i])
            test_seeds = [tuple(map(lambda x, i=i: x + 1 * i, t)) for (i, t) in enumerate(((1, 2) for _ in range(iterations)))]
            observations = [[] for _ in range(iterations)]
            for observation in observations:
                counts = np.zeros(size, dtype=np.int32)
                for seed in test_seeds:
                    crop = random_crop_ops.stateless_random_crop(value, single, seed=seed)
                    y = self.evaluate(crop)
                    self.assertAllEqual(y.shape, single)
                    counts[y] += 1
                observation.append(counts)
            for i in range(1, iterations):
                self.assertAllEqual(observations[0], observations[i])
if __name__ == '__main__':
    test.main()