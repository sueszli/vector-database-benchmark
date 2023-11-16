"""Tests for wrapping / unwrapping dataset variants."""
from absl.testing import parameterized
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.platform import test

class WrapUnwrapTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(test_base.default_test_combinations())
    def DISABLED_testBasic(self):
        if False:
            i = 10
            return i + 15
        ds = dataset_ops.Dataset.range(100)
        ds_variant = ds._variant_tensor
        wrapped_variant = gen_dataset_ops.wrap_dataset_variant(ds_variant)
        unwrapped_variant = gen_dataset_ops.unwrap_dataset_variant(wrapped_variant)
        variant_ds = dataset_ops._VariantDataset(unwrapped_variant, ds.element_spec)
        get_next = self.getNext(variant_ds, requires_initialization=True)
        for i in range(100):
            self.assertEqual(i, self.evaluate(get_next()))

    @combinations.generate(test_base.graph_only_combinations())
    def testGPU(self):
        if False:
            return 10
        ds = dataset_ops.Dataset.range(100)
        ds_variant = ds._variant_tensor
        wrapped_variant = gen_dataset_ops.wrap_dataset_variant(ds_variant)
        with ops.device('/gpu:0'):
            gpu_wrapped_variant = array_ops.identity(wrapped_variant)
        unwrapped_variant = gen_dataset_ops.unwrap_dataset_variant(gpu_wrapped_variant)
        variant_ds = dataset_ops._VariantDataset(unwrapped_variant, ds.element_spec)
        iterator = dataset_ops.make_initializable_iterator(variant_ds)
        get_next = iterator.get_next()
        with self.cached_session():
            self.evaluate(iterator.initializer)
            for i in range(100):
                self.assertEqual(i, self.evaluate(get_next))
if __name__ == '__main__':
    test.main()