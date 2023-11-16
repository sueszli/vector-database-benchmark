"""Tests for `tf.data.Dataset.random()`."""
import warnings
from absl.testing import parameterized
from tensorflow.python import tf2
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import test

class RandomTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(global_seed=[None, 10], local_seed=[None, 20])))
    def testDeterminism(self, global_seed, local_seed):
        if False:
            i = 10
            return i + 15
        expect_determinism = global_seed is not None or local_seed is not None
        random_seed.set_random_seed(global_seed)
        ds = dataset_ops.Dataset.random(seed=local_seed).take(10)
        output_1 = self.getDatasetOutput(ds, requires_initialization=True)
        ds = self.graphRoundTrip(ds)
        output_2 = self.getDatasetOutput(ds, requires_initialization=True)
        if expect_determinism:
            self.assertEqual(output_1, output_2)
        else:
            self.assertNotEqual(output_1, output_2)

    @combinations.generate(combinations.times(test_base.graph_only_combinations(), combinations.combine(rerandomize=[None, True, False])))
    def testRerandomizeEachIterationEpochsIgnored(self, rerandomize):
        if False:
            for i in range(10):
                print('nop')
        with warnings.catch_warnings(record=True) as w:
            dataset = dataset_ops.Dataset.random(seed=42, rerandomize_each_iteration=rerandomize, name='random').take(10)
        first_epoch = self.getDatasetOutput(dataset, requires_initialization=True)
        second_epoch = self.getDatasetOutput(dataset, requires_initialization=True)
        if rerandomize:
            if not tf2.enabled() and rerandomize:
                found_warning = False
                for warning in w:
                    if 'In TF 1, the `rerandomize_each_iteration=True` option' in str(warning):
                        found_warning = True
                        break
                self.assertTrue(found_warning)
        self.assertEqual(first_epoch, second_epoch)

    @combinations.generate(combinations.times(test_base.eager_only_combinations(), combinations.combine(rerandomize=[None, True, False])))
    def testRerandomizeEachIterationEpochs(self, rerandomize):
        if False:
            return 10
        dataset = dataset_ops.Dataset.random(seed=42, rerandomize_each_iteration=rerandomize, name='random').take(10)
        first_epoch = self.getDatasetOutput(dataset)
        second_epoch = self.getDatasetOutput(dataset)
        if rerandomize:
            self.assertEqual(first_epoch == second_epoch, not rerandomize or rerandomize is None)
        else:
            self.assertEqual(first_epoch, second_epoch)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(rerandomize=[None, True, False])))
    def testRerandomizeRepeatEpochs(self, rerandomize):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.random(seed=42, rerandomize_each_iteration=rerandomize, name='random').take(10)
        dataset = dataset.repeat(2)
        next_element = self.getNext(dataset, requires_initialization=True)
        first_epoch = []
        for _ in range(10):
            first_epoch.append(self.evaluate(next_element()))
        second_epoch = []
        for _ in range(10):
            second_epoch.append(self.evaluate(next_element()))
        if rerandomize:
            self.assertEqual(first_epoch == second_epoch, not rerandomize or rerandomize is None)
        else:
            self.assertEqual(first_epoch, second_epoch)

    @combinations.generate(combinations.times(test_base.v2_eager_only_combinations(), combinations.combine(rerandomize=[None, True, False])))
    def testRerandomizeInsideFunction(self, rerandomize):
        if False:
            for i in range(10):
                print('nop')

        @def_function.function
        def make_dataset():
            if False:
                while True:
                    i = 10
            dataset = dataset_ops.Dataset.random(seed=42, rerandomize_each_iteration=rerandomize, name='random').take(10)
            return dataset
        dataset = make_dataset()
        first_epoch = self.getDatasetOutput(dataset)
        second_epoch = self.getDatasetOutput(dataset)
        if rerandomize:
            self.assertEqual(first_epoch == second_epoch, not rerandomize or rerandomize is None)
        else:
            self.assertEqual(first_epoch, second_epoch)

    @combinations.generate(test_base.default_test_combinations())
    def testName(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.random(seed=42, name='random').take(1).map(lambda _: 42)
        self.assertDatasetProduces(dataset, expected_output=[42], requires_initialization=True)

class RandomCheckpointTest(checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):

    def _build_random_dataset(self, num_elements=10, seed=None, rerandomize_each_iteration=None):
        if False:
            while True:
                i = 10
        dataset = dataset_ops.Dataset.random(seed=seed, rerandomize_each_iteration=rerandomize_each_iteration)
        return dataset.take(num_elements)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), checkpoint_test_base.default_test_combinations(), combinations.combine(rerandomize_each_iteration=[True, False])))
    def test(self, verify_fn, rerandomize_each_iteration):
        if False:
            i = 10
            return i + 15
        seed = 55
        num_elements = 10
        verify_fn(self, lambda : self._build_random_dataset(seed=seed, num_elements=num_elements, rerandomize_each_iteration=rerandomize_each_iteration), num_elements)
if __name__ == '__main__':
    test.main()