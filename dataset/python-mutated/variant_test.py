"""Tests for `tf.data.experimental.{from,to}_variant()`."""
from absl.testing import parameterized
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test

class VariantTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(test_base.default_test_combinations())
    def testRoundtripRange(self):
        if False:
            return 10
        dataset = dataset_ops.Dataset.range(10)
        variant = dataset_ops.to_variant(dataset)
        dataset = dataset_ops.from_variant(variant, dataset_ops.get_structure(dataset))
        self.assertDatasetProduces(dataset, range(10))
        self.assertEqual(self.evaluate(dataset.cardinality()), 10)

    @combinations.generate(combinations.combine(tf_api_version=[2], mode=['eager', 'graph']))
    def testRoundtripMap(self):
        if False:
            while True:
                i = 10
        dataset = dataset_ops.Dataset.range(10).map(lambda x: x * x)
        variant = dataset_ops.to_variant(dataset)
        dataset = dataset_ops.from_variant(variant, dataset_ops.get_structure(dataset))
        self.assertDatasetProduces(dataset, [x * x for x in range(10)])
        self.assertEqual(self.evaluate(dataset.cardinality()), 10)
if __name__ == '__main__':
    test.main()