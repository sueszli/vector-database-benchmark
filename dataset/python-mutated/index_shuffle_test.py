"""Tests for `tf.data.experimental.index_shuffle()`."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.data.experimental.ops import shuffle_ops
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class IndexShuffleTest(test_base.DatasetTestBase, parameterized.TestCase):

    def _build_dataset(self, seed=None, reshuffle_each_iteration=None, num_elements=10):
        if False:
            print('Hello World!')
        file_infos = []
        for _ in range(5):
            file_infos.append({'path': 'unused', 'num_elements': num_elements})

        def reader_factory(files):
            if False:
                print('Hello World!')
            return dataset_ops.Dataset.range(num_elements * array_ops.shape(files, out_type=dtypes.int64)[0])
        return shuffle_ops.index_shuffle(file_infos, reader_factory, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)

    def testProcessFileInfos(self):
        if False:
            i = 10
            return i + 15
        file_infos = []
        file_infos.append({'path': 'take_50', 'num_elements': 100, 'skip': 25, 'take': 50})
        file_infos.append({'path': 'skip_all', 'num_elements': 100, 'skip': -1})
        file_infos.append({'path': 'take_all', 'num_elements': 100, 'take': -1})
        file_infos.append({'path': 'take_10', 'num_elements': 100, 'skip': 90, 'take': 20})
        result = shuffle_ops._process_file_infos(file_infos)
        self.assertEqual(result['files'], ['take_50', 'skip_all', 'take_all', 'take_10'])
        self.assertEqual(result['num_elements'], 160)
        inputs = [0, 49, 50, 51, 149, 150, 151, 159]
        expected = [25, 74, 200, 201, 299, 390, 391, 399]
        for (i, expected) in enumerate(expected):
            self.assertEqual(self.evaluate(shuffle_ops._adjust_index([inputs[i]], result['thresholds'], result['offsets'])), expected)

    @combinations.generate(test_base.default_test_combinations())
    def testUnseeded(self):
        if False:
            while True:
                i = 10
        unshuffled_elements = np.arange(50)
        shuffled_elements_1 = self.getDatasetOutput(self._build_dataset(), requires_initialization=True)
        shuffled_elements_2 = self.getDatasetOutput(self._build_dataset(), requires_initialization=True)
        self.assertAllEqual(sorted(unshuffled_elements), sorted(shuffled_elements_1))
        self.assertAllEqual(sorted(unshuffled_elements), sorted(shuffled_elements_2))
        self.assertNotEqual(shuffled_elements_1, shuffled_elements_2)

    @combinations.generate(test_base.default_test_combinations())
    def testSameSeed(self):
        if False:
            i = 10
            return i + 15
        shuffled_elements_1 = self.getDatasetOutput(self._build_dataset(seed=42), requires_initialization=True)
        shuffled_elements_2 = self.getDatasetOutput(self._build_dataset(seed=42), requires_initialization=True)
        self.assertEqual(shuffled_elements_1, shuffled_elements_2)

    @combinations.generate(test_base.default_test_combinations())
    def testLargeDataSet(self):
        if False:
            print('Hello World!')
        self._build_dataset(seed=42, num_elements=128 * 1024 * 1024)

    @combinations.generate(test_base.default_test_combinations())
    def testDifferentSeed(self):
        if False:
            while True:
                i = 10
        shuffled_elements_1 = self.getDatasetOutput(self._build_dataset(seed=42), requires_initialization=True)
        shuffled_elements_2 = self.getDatasetOutput(self._build_dataset(seed=24), requires_initialization=True)
        self.assertNotEqual(shuffled_elements_1, shuffled_elements_2)
        self.assertAllEqual(sorted(shuffled_elements_1), sorted(shuffled_elements_2))

    @combinations.generate(combinations.times(test_base.v2_eager_only_combinations(), combinations.combine(reshuffle_each_iteration=[True, False])))
    def testReshuffleEachIteration(self, reshuffle_each_iteration):
        if False:
            while True:
                i = 10
        dataset = self._build_dataset(seed=42, reshuffle_each_iteration=reshuffle_each_iteration)
        shuffled_elements_1 = self.getDatasetOutput(dataset)
        shuffled_elements_2 = self.getDatasetOutput(dataset)
        if reshuffle_each_iteration:
            self.assertNotEqual(shuffled_elements_1, shuffled_elements_2)
            self.assertAllEqual(sorted(shuffled_elements_1), sorted(shuffled_elements_2))
        else:
            self.assertAllEqual(shuffled_elements_1, shuffled_elements_2)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(skip=[8, 2, 0, -1], take=[8, 2, 0, -1])))
    def testSkipAndTake(self, skip, take):
        if False:
            print('Hello World!')
        num_elements = 10
        file_infos = []
        file_infos.append({'path': 'unused', 'num_elements': num_elements, 'skip': skip if skip >= 0 else num_elements, 'take': take if take >= 0 else num_elements})
        start = skip if skip >= 0 else num_elements
        stop = min(num_elements, skip + take if take >= 0 else num_elements)
        expected = np.arange(start, stop)

        def reader_factory(_):
            if False:
                while True:
                    i = 10
            return dataset_ops.Dataset.range(10)
        dataset = shuffle_ops.index_shuffle(file_infos, reader_factory)
        actual = self.getDatasetOutput(dataset, requires_initialization=True)
        self.assertAllEqual(sorted(expected), sorted(actual))

class IndexShuffleCheckpointTest(checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):

    def _build_dataset(self, num_elements_per_file, num_files, num_epochs, seed=None, reshuffle_each_iteration=None, symbolic_checkpoint=None):
        if False:
            i = 10
            return i + 15
        file_infos = []
        for _ in range(num_files):
            file_infos.append({'path': 'unused', 'num_elements': num_elements_per_file})

        def reader_factory(files):
            if False:
                return 10
            return dataset_ops.Dataset.range(num_elements_per_file * array_ops.shape(files, out_type=dtypes.int64)[0])
        dataset = shuffle_ops.index_shuffle(file_infos, reader_factory, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)
        dataset = dataset.repeat(num_epochs)
        if symbolic_checkpoint:
            options = options_lib.Options()
            options.experimental_symbolic_checkpoint = symbolic_checkpoint
            dataset = dataset.with_options(options)
        return dataset

    @combinations.generate(combinations.times(test_base.default_test_combinations(), checkpoint_test_base.default_test_combinations(), combinations.combine(symbolic_checkpoint=[False, True], reshuffle_each_iteration=[False, True])))
    def test(self, verify_fn, symbolic_checkpoint, reshuffle_each_iteration):
        if False:
            for i in range(10):
                print('nop')
        seed = 42
        num_elements_per_file = 8
        num_files = 3
        num_epochs = 2
        num_outputs = num_elements_per_file * num_files * num_epochs
        verify_fn(self, lambda : self._build_dataset(num_elements_per_file=num_elements_per_file, num_files=num_files, num_epochs=num_epochs, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration, symbolic_checkpoint=symbolic_checkpoint), num_outputs)
if __name__ == '__main__':
    test.main()