"""Tests for `tf.data.Dataset.__len__()`."""
from absl.testing import parameterized
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test

class LenTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(test_base.eager_only_combinations())
    def testKnown(self):
        if False:
            for i in range(10):
                print('nop')
        num_elements = 10
        ds = dataset_ops.Dataset.range(num_elements)
        self.assertLen(ds, 10)

    @combinations.generate(test_base.eager_only_combinations())
    def testInfinite(self):
        if False:
            while True:
                i = 10
        num_elements = 10
        ds = dataset_ops.Dataset.range(num_elements).repeat()
        with self.assertRaisesRegex(TypeError, 'infinite'):
            len(ds)

    @combinations.generate(test_base.eager_only_combinations())
    def testUnknown(self):
        if False:
            while True:
                i = 10
        num_elements = 10
        ds = dataset_ops.Dataset.range(num_elements).filter(lambda x: True)
        with self.assertRaisesRegex(TypeError, 'unknown'):
            len(ds)

    @combinations.generate(test_base.graph_only_combinations())
    def testGraphMode(self):
        if False:
            print('Hello World!')
        num_elements = 10
        ds = dataset_ops.Dataset.range(num_elements)
        with self.assertRaisesRegex(TypeError, '`tf.data.Dataset` only supports `len` in eager mode. Use `tf.data.Dataset.cardinality\\(\\)` instead.'):
            len(ds)
if __name__ == '__main__':
    test.main()