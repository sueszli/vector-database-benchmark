"""Tests for the `MapAndBatchFusion` optimization."""
from absl.testing import parameterized
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test

class MapAndBatchFusionTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(test_base.default_test_combinations())
    def testMapAndBatchFusion(self):
        if False:
            return 10
        dataset = dataset_ops.Dataset.range(10).apply(testing.assert_next(['MapAndBatch'])).map(lambda x: x * x).batch(10)
        options = options_lib.Options()
        options.experimental_optimization.apply_default_optimizations = False
        options.experimental_optimization.map_and_batch_fusion = True
        dataset = dataset.with_options(options)
        self.assertDatasetProduces(dataset, expected_output=[[x * x for x in range(10)]])
if __name__ == '__main__':
    test.main()