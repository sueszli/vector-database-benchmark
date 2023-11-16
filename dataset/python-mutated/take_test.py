"""Tests for `tf.data.Dataset.take()`."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test

class TakeTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(count=[-1, 0, 4, 10, 25])))
    def testBasic(self, count):
        if False:
            while True:
                i = 10
        components = (np.arange(10),)
        dataset = dataset_ops.Dataset.from_tensor_slices(components).take(count)
        self.assertEqual([c.shape[1:] for c in components], [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])
        num_output = min(count, 10) if count != -1 else 10
        self.assertDatasetProduces(dataset, [tuple(components[0][i:i + 1]) for i in range(num_output)])

    @combinations.generate(test_base.default_test_combinations())
    def testName(self):
        if False:
            return 10
        dataset = dataset_ops.Dataset.from_tensors(42).take(1, name='take')
        self.assertDatasetProduces(dataset, [42])

class TakeDatasetCheckpointTest(checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):

    def _build_take_dataset(self, count, options=None):
        if False:
            return 10
        dataset = dataset_ops.Dataset.range(100).take(count)
        if options:
            dataset = dataset.with_options(options)
        return dataset

    @combinations.generate(combinations.times(test_base.default_test_combinations(), checkpoint_test_base.default_test_combinations(), combinations.combine(symbolic_checkpoint=[False, True]), combinations.combine(count=[50], num_outputs=[50]) + combinations.combine(count=[200, 100, -1], num_outputs=[100]) + combinations.combine(count=[0], num_outputs=[0])))
    def test(self, verify_fn, symbolic_checkpoint, count, num_outputs):
        if False:
            while True:
                i = 10
        options = options_lib.Options()
        options.experimental_symbolic_checkpoint = symbolic_checkpoint
        verify_fn(self, lambda : self._build_take_dataset(count, options), num_outputs)

class TakeRandomAccessTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(index=[-1, 3, 4])))
    def testInvalidIndex(self, index):
        if False:
            return 10
        dataset = dataset_ops.Dataset.range(10).take(3)
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(random_access.at(dataset, index=index))

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(index=[-2, 0, 1])))
    def testEmptyDataset(self, index):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.from_tensor_slices([]).take(5)
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(random_access.at(dataset, index=index))

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(count=[-1, 0, 4, 10, 25])))
    def testBasic(self, count):
        if False:
            i = 10
            return i + 15
        dataset = dataset_ops.Dataset.range(10).take(count)
        num_output = min(count, 10) if count != -1 else 10
        for i in range(num_output):
            self.assertEqual(self.evaluate(random_access.at(dataset, index=i)), i)
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(random_access.at(dataset, index=num_output))
if __name__ == '__main__':
    test.main()