"""Tests for `tf.data.experimental.pad_to_cardinality()."""
from absl.testing import parameterized
from tensorflow.python.data.experimental.ops import pad_to_cardinality
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test
pad_to_cardinality = pad_to_cardinality.pad_to_cardinality

class PadToCardinalityTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(test_base.default_test_combinations())
    def testBasic(self):
        if False:
            print('Hello World!')
        data = [1, 2, 3, 4, 5]
        target = 12
        ds = dataset_ops.Dataset.from_tensor_slices({'a': data})
        ds = ds.apply(pad_to_cardinality(target))
        expected_data = [{'a': data[i], 'valid': True} for i in range(len(data))]
        expected_padding = [{'a': 0, 'valid': False} for _ in range(target - len(data))]
        self.assertAllEqual(self.getDatasetOutput(ds), expected_data + expected_padding)

    @combinations.generate(test_base.default_test_combinations())
    def testNoPadding(self):
        if False:
            while True:
                i = 10
        data = [1, 2, 3, 4, 5]
        target = 5
        ds = dataset_ops.Dataset.from_tensor_slices({'a': data})
        ds = ds.apply(pad_to_cardinality(target))
        expected_data = [{'a': data[i], 'valid': True} for i in range(len(data))]
        self.assertAllEqual(self.getDatasetOutput(ds), expected_data)

    @combinations.generate(test_base.default_test_combinations())
    def testStructuredData(self):
        if False:
            while True:
                i = 10
        data = {'a': [1, 2, 3, 4, 5], 'b': ([b'a', b'b', b'c', b'd', b'e'], [-1, -2, -3, -4, -5])}
        data_len = len(data['a'])
        target = 12
        ds = dataset_ops.Dataset.from_tensor_slices(data)
        ds = ds.apply(pad_to_cardinality(target))
        expected_data = [{'a': data['a'][i], 'b': (data['b'][0][i], data['b'][1][i]), 'valid': True} for i in range(data_len)]
        expected_padding = [{'a': 0, 'b': (b'', 0), 'valid': False} for _ in range(target - data_len)]
        self.assertAllEqual(self.getDatasetOutput(ds), expected_data + expected_padding)

    @combinations.generate(test_base.v2_eager_only_combinations())
    def testUnknownCardinality(self):
        if False:
            for i in range(10):
                print('nop')
        ds = dataset_ops.Dataset.from_tensors({'a': 1}).filter(lambda x: True)
        with self.assertRaisesRegex(ValueError, 'The dataset passed into `pad_to_cardinality` must have a known cardinalty, but has cardinality -2'):
            ds = ds.apply(pad_to_cardinality(5))
            self.getDatasetOutput(ds)

    @combinations.generate(test_base.v2_eager_only_combinations())
    def testInfiniteCardinality(self):
        if False:
            i = 10
            return i + 15
        ds = dataset_ops.Dataset.from_tensors({'a': 1}).repeat()
        with self.assertRaisesRegex(ValueError, 'The dataset passed into `pad_to_cardinality` must have a known cardinalty, but has cardinality -1'):
            ds = ds.apply(pad_to_cardinality(5))
            self.getDatasetOutput(ds)

    @combinations.generate(test_base.v2_only_combinations())
    def testNonMapping(self):
        if False:
            return 10
        ds = dataset_ops.Dataset.from_tensors(1)
        with self.assertRaisesRegex(ValueError, '`pad_to_cardinality` requires its input dataset to be a dictionary'):
            ds = ds.apply(pad_to_cardinality(5))
            self.getDatasetOutput(ds)

    @combinations.generate(test_base.v2_eager_only_combinations())
    def testRequestedCardinalityTooShort(self):
        if False:
            return 10
        ds = dataset_ops.Dataset.from_tensors({'a': 1}).repeat(5)
        with self.assertRaisesRegex(ValueError, 'The dataset passed into `pad_to_cardinality` must have a cardinalty less than the target cardinality \\(3\\), but has cardinality 5'):
            ds = ds.apply(pad_to_cardinality(3))
            self.getDatasetOutput(ds)
if __name__ == '__main__':
    test.main()