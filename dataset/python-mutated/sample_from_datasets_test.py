"""Tests for `tf.data.Dataset.sample_from_dataset()`."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.compat import compat as tf_compat
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import test

def _weights_type_combinations():
    if False:
        i = 10
        return i + 15
    return combinations.combine(weights_type=['list', 'tensor', 'dataset'])

def _get_weights_of_type(weights_list, weights_type):
    if False:
        while True:
            i = 10
    if weights_type == 'list':
        return weights_list
    if weights_type == 'tensor':
        return ops.convert_to_tensor(weights_list, name='weights')
    return dataset_ops.Dataset.from_tensors(weights_list).repeat()

class SampleFromDatasetsTest(test_base.DatasetTestBase, parameterized.TestCase):

    def _normalize(self, vec):
        if False:
            print('Hello World!')
        return vec / vec.sum()

    def _chi2(self, expected, actual):
        if False:
            print('Hello World!')
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        diff = actual - expected
        chi2 = np.sum(diff * diff / expected, axis=0)
        return chi2

    @combinations.generate(combinations.times(test_base.default_test_combinations(), _weights_type_combinations()))
    def testSampleFromDatasets(self, weights_type):
        if False:
            print('Hello World!')
        random_seed.set_random_seed(1619)
        num_samples = 5000
        rand_probs = self._normalize(np.random.random_sample((5,)))
        for probs in [[0.85, 0.05, 0.1], rand_probs, [1.0]]:
            weights = _get_weights_of_type(np.asarray(probs), weights_type)
            classes = len(probs)
            dataset = dataset_ops.Dataset.sample_from_datasets([dataset_ops.Dataset.from_tensors(i).repeat() for i in range(classes)], weights)
            dataset = dataset.take(num_samples)
            next_element = self.getNext(dataset, requires_initialization=True)
            freqs = np.zeros([classes])
            for _ in range(num_samples):
                freqs[self.evaluate(next_element())] += 1
            with self.assertRaises(errors.OutOfRangeError):
                self.evaluate(next_element())
            self.assertLess(self._chi2(probs, freqs / num_samples), 0.01)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), _weights_type_combinations()))
    def testSampleFromDatasetsStoppingOnEmptyDataset(self, weights_type):
        if False:
            i = 10
            return i + 15
        weights = _get_weights_of_type(np.asarray([0.5, 0.1, 0.4]), weights_type)
        datasets = [dataset_ops.Dataset.from_tensors(np.int64(-1)), dataset_ops.Dataset.from_tensors(np.int64(1)).repeat(), dataset_ops.Dataset.range(10).repeat()]
        sample_dataset = dataset_ops.Dataset.sample_from_datasets(datasets, weights=weights, stop_on_empty_dataset=True)
        samples_list = self.getIteratorOutput(self.getNext(sample_dataset, requires_initialization=True))
        self.assertEqual(samples_list.count(-1), 1)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), _weights_type_combinations()))
    def testSampleFromDatasetsSkippingEmptyDataset(self, weights_type):
        if False:
            while True:
                i = 10
        weights = _get_weights_of_type(np.asarray([0.5, 0.1, 0.4]), weights_type)
        datasets = [dataset_ops.Dataset.from_tensors(np.int64(-1)), dataset_ops.Dataset.from_tensors(np.int64(1)).repeat(), dataset_ops.Dataset.range(10).repeat()]
        sample_dataset = dataset_ops.Dataset.sample_from_datasets(datasets, weights=weights, stop_on_empty_dataset=False).take(100)
        samples_list = self.getIteratorOutput(self.getNext(sample_dataset, requires_initialization=True))
        self.assertLen(samples_list, 100)
        self.assertEqual(samples_list.count(-1), 1)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), _weights_type_combinations()))
    def testSampleFromDatasetsWithZeroWeight(self, weights_type):
        if False:
            return 10
        weights = _get_weights_of_type(np.asarray([0.0, 1.0]), weights_type)
        datasets = [dataset_ops.Dataset.from_tensors(-1).repeat(2), dataset_ops.Dataset.from_tensors(1).repeat(2)]
        sample_dataset = dataset_ops.Dataset.sample_from_datasets(datasets, weights=weights, stop_on_empty_dataset=True)
        self.assertDatasetProduces(sample_dataset, [1, 1], requires_initialization=True)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), _weights_type_combinations()))
    def testSampleFromEmptyDataset(self, weights_type):
        if False:
            return 10
        weights = _get_weights_of_type(np.asarray([1.0, 0.0]), weights_type)
        datasets = [dataset_ops.Dataset.range(0), dataset_ops.Dataset.range(1).repeat()]
        sample_dataset = dataset_ops.Dataset.sample_from_datasets(datasets, weights=weights, stop_on_empty_dataset=True)
        self.assertDatasetProduces(sample_dataset, [], requires_initialization=True)

    @combinations.generate(test_base.default_test_combinations())
    def testSampleFromDatasetsSkippingDatasetsWithZeroWeight(self):
        if False:
            return 10
        weights = np.asarray([0.0, 1.0])
        datasets = [dataset_ops.Dataset.from_tensors(-1).repeat(), dataset_ops.Dataset.from_tensors(1)]
        sample_dataset = dataset_ops.Dataset.sample_from_datasets(datasets, weights=weights, stop_on_empty_dataset=False)
        self.assertDatasetProduces(sample_dataset, [1])

    @combinations.generate(test_base.default_test_combinations())
    def testSampleFromDatasetsAllWeightsAreZero(self):
        if False:
            for i in range(10):
                print('nop')
        weights = np.asarray([0.0, 0.0])
        datasets = [dataset_ops.Dataset.from_tensors(-1).repeat(), dataset_ops.Dataset.from_tensors(1).repeat()]
        sample_dataset = dataset_ops.Dataset.sample_from_datasets(datasets, weights=weights, stop_on_empty_dataset=False)
        self.assertDatasetProduces(sample_dataset, [])

    @combinations.generate(test_base.default_test_combinations())
    def testSampleFromDatasetsCardinality(self):
        if False:
            return 10
        ds1 = dataset_ops.Dataset.from_tensors([1.0]).repeat()
        ds2 = dataset_ops.Dataset.from_tensors([2.0]).repeat()
        ds = dataset_ops.Dataset.sample_from_datasets([ds1, ds2])
        self.assertEqual(self.evaluate(ds.cardinality()), dataset_ops.INFINITE)

    @combinations.generate(test_base.default_test_combinations())
    def testSampleFromDatasetsNested(self):
        if False:
            return 10
        ds1 = dataset_ops.Dataset.range(10).window(2)
        ds2 = dataset_ops.Dataset.range(10, 20).window(2)
        ds = dataset_ops.Dataset.sample_from_datasets([ds1, ds2], weights=[0.3, 0.7])
        ds = ds.flat_map(lambda x: x)
        next_element = self.getNext(ds, requires_initialization=True)
        self.evaluate(next_element())

    @combinations.generate(combinations.times(test_base.eager_only_combinations(), combinations.combine(rerandomize=[None, True, False])))
    def testSampleFromDatasetsRerandomizeEachIterationEpochs(self, rerandomize):
        if False:
            i = 10
            return i + 15
        if rerandomize is not None and (not tf_compat.forward_compatible(2022, 12, 17)):
            self.skipTest('target functionality not available due to forward compatibility')
        dataset1 = dataset_ops.Dataset.range(0, 10)
        dataset2 = dataset_ops.Dataset.range(100, 110)
        sample_dataset = dataset_ops.Dataset.sample_from_datasets([dataset1, dataset2], seed=42, weights=[0.5, 0.5], stop_on_empty_dataset=True, rerandomize_each_iteration=rerandomize)
        first_epoch = self.getDatasetOutput(sample_dataset)
        second_epoch = self.getDatasetOutput(sample_dataset)
        if rerandomize:
            self.assertNotEqual(first_epoch, second_epoch)
        else:
            self.assertEqual(first_epoch, second_epoch)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(rerandomize=[None, True, False])))
    def testSampleFromDatasetsRerandomizeRepeatEpochs(self, rerandomize):
        if False:
            for i in range(10):
                print('nop')
        if rerandomize is not None and (not tf_compat.forward_compatible(2022, 12, 17)):
            self.skipTest('target functionality not available due to forward compatibility')
        dataset1 = dataset_ops.Dataset.range(0, 10)
        dataset2 = dataset_ops.Dataset.range(100, 110)
        sample_dataset = dataset_ops.Dataset.sample_from_datasets([dataset1, dataset2], seed=42, weights=[0.5, 0.5], stop_on_empty_dataset=True, rerandomize_each_iteration=rerandomize)
        sample_dataset = sample_dataset.repeat(2)
        epochs = self.getDatasetOutput(sample_dataset, requires_initialization=True)
        first_epoch = epochs[:len(epochs) // 2]
        second_epoch = epochs[len(epochs) // 2:]
        if rerandomize:
            self.assertNotEqual(first_epoch, second_epoch)
        else:
            self.assertEqual(first_epoch, second_epoch)

    @combinations.generate(combinations.times(test_base.v2_eager_only_combinations(), combinations.combine(rerandomize=[None, True, False])))
    def testSampleFromDatasetsRerandomizeInsideFunction(self, rerandomize):
        if False:
            print('Hello World!')
        if rerandomize is not None and (not tf_compat.forward_compatible(2022, 12, 17)):
            self.skipTest('target functionality not available due to forward compatibility')

        @def_function.function
        def make_dataset():
            if False:
                for i in range(10):
                    print('nop')
            dataset1 = dataset_ops.Dataset.range(0, 10)
            dataset2 = dataset_ops.Dataset.range(100, 110)
            sample_dataset = dataset_ops.Dataset.sample_from_datasets([dataset1, dataset2], seed=42, weights=[0.5, 0.5], stop_on_empty_dataset=True, rerandomize_each_iteration=rerandomize)
            return sample_dataset
        sample_dataset = make_dataset()
        first_epoch = self.getDatasetOutput(sample_dataset)
        second_epoch = self.getDatasetOutput(sample_dataset)
        if rerandomize:
            self.assertNotEqual(first_epoch, second_epoch)
        else:
            self.assertEqual(first_epoch, second_epoch)

    @combinations.generate(test_base.default_test_combinations())
    def testErrors(self):
        if False:
            return 10
        with self.assertRaisesRegex(ValueError, 'should have the same length'):
            dataset_ops.Dataset.sample_from_datasets([dataset_ops.Dataset.range(10), dataset_ops.Dataset.range(20)], weights=[0.25, 0.25, 0.25, 0.25])
        with self.assertRaisesRegex(TypeError, '`tf.float32` or `tf.float64`'):
            dataset_ops.Dataset.sample_from_datasets([dataset_ops.Dataset.range(10), dataset_ops.Dataset.range(20)], weights=[1, 1])
        with self.assertRaisesRegex(TypeError, 'must have compatible'):
            dataset_ops.Dataset.sample_from_datasets([dataset_ops.Dataset.from_tensors(0), dataset_ops.Dataset.from_tensors(0.0)])
        with self.assertRaisesRegex(ValueError, 'Invalid `datasets`. `datasets` should not be empty.'):
            dataset_ops.Dataset.sample_from_datasets(datasets=[], weights=[])

class SampleFromDatasetsCheckpointTest(checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):

    def _build_dataset(self, probs, num_samples, options=None):
        if False:
            i = 10
            return i + 15
        datasets = [dataset_ops.Dataset.from_tensors(i).repeat(None) for i in range(len(probs))]
        dataset = dataset_ops.Dataset.sample_from_datasets(datasets, probs, seed=1813)
        dataset = dataset.take(num_samples)
        if options:
            dataset = dataset.with_options(options)
        return dataset

    @combinations.generate(combinations.times(test_base.default_test_combinations(), checkpoint_test_base.default_test_combinations(), combinations.combine(symbolic_checkpoint=[False, True])))
    def test(self, verify_fn, symbolic_checkpoint):
        if False:
            while True:
                i = 10
        options = options_lib.Options()
        options.experimental_symbolic_checkpoint = symbolic_checkpoint
        verify_fn(self, lambda : self._build_dataset([0.5, 0.5], 100, options), num_outputs=100)
if __name__ == '__main__':
    test.main()