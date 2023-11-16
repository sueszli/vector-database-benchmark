"""Tests for compression ops."""
from collections import namedtuple
from absl.testing import parameterized
from tensorflow.python.data.experimental.ops import compression_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test

def _test_objects():
    if False:
        return 10
    Item = namedtuple('Item', 'id name')
    return [combinations.NamedObject('int', 1), combinations.NamedObject('string', 'dog'), combinations.NamedObject('tuple', (1, 1)), combinations.NamedObject('nested_tuple', ((1, 1), (2, 2))), combinations.NamedObject('named_tuple', Item(id=1, name='item1')), combinations.NamedObject('unicode', 'アヒル'), combinations.NamedObject('nested_named_tuple', (Item(id=1, name='item1'), Item(id=2, name='item2'))), combinations.NamedObject('int_string_tuple', (1, 'dog')), combinations.NamedObject('sparse', sparse_tensor.SparseTensorValue(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])), combinations.NamedObject('sparse_structured', {'a': sparse_tensor.SparseTensorValue(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]), 'b': (1, 2, 'dog')})]

def _test_v2_eager_only_objects():
    if False:
        for i in range(10):
            print('nop')
    return [combinations.NamedObject('ragged', ragged_factory_ops.constant([[0, 1, 2, 3], [4, 5], [6, 7, 8], [9]])), combinations.NamedObject('sparse_ragged_structured', {'sparse': sparse_tensor.SparseTensorValue(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]), 'ragged': ragged_factory_ops.constant([[0, 1, 2, 3], [9]])})]

class CompressionOpsTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(element=_test_objects())) + combinations.times(test_base.v2_eager_only_combinations(), combinations.combine(element=_test_v2_eager_only_objects())))
    def testCompression(self, element):
        if False:
            i = 10
            return i + 15
        element = element._obj
        compressed = compression_ops.compress(element)
        uncompressed = compression_ops.uncompress(compressed, structure.type_spec_from_value(element))
        self.assertValuesEqual(element, self.evaluate(uncompressed))

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(element=_test_objects())) + combinations.times(test_base.v2_eager_only_combinations(), combinations.combine(element=_test_v2_eager_only_objects())))
    def testDatasetCompression(self, element):
        if False:
            while True:
                i = 10
        element = element._obj
        dataset = dataset_ops.Dataset.from_tensors(element)
        element_spec = dataset.element_spec
        dataset = dataset.map(lambda *x: compression_ops.compress(x))
        dataset = dataset.map(lambda x: compression_ops.uncompress(x, element_spec))
        self.assertDatasetProduces(dataset, [element])

    @combinations.generate(combinations.times(test_base.default_test_combinations()))
    def testCompressionOutputDTypeMismatch(self):
        if False:
            for i in range(10):
                print('nop')
        element = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        compressed = compression_ops.compress(element)
        with self.assertRaisesRegex(errors.FailedPreconditionError, 'but got a tensor of type string'):
            uncompressed = compression_ops.uncompress(compressed, structure.type_spec_from_value(0))
            self.evaluate(uncompressed)

    @combinations.generate(combinations.times(test_base.default_test_combinations()))
    def testCompressionInputShapeMismatch(self):
        if False:
            while True:
                i = 10
        element = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        compressed = compression_ops.compress(element)
        compressed = [compressed, compressed]
        error = errors.InvalidArgumentError if context.executing_eagerly() else ValueError
        with self.assertRaises(error):
            uncompressed = compression_ops.uncompress(compressed, structure.type_spec_from_value(0))
            self.evaluate(uncompressed)

    @combinations.generate(combinations.times(test_base.default_test_combinations()))
    def testCompressionInputDTypeMismatch(self):
        if False:
            i = 10
            return i + 15
        uncompressed = list(range(10))
        with self.assertRaises(TypeError):
            uncompressed = compression_ops.uncompress(uncompressed, structure.type_spec_from_value(uncompressed))
            self.evaluate(uncompressed)

    @combinations.generate(combinations.times(test_base.default_test_combinations()))
    def testCompressionVariantMismatch(self):
        if False:
            i = 10
            return i + 15
        dataset = dataset_ops.Dataset.range(10)
        variant = dataset._variant_tensor
        with self.assertRaises(errors.InvalidArgumentError):
            uncompressed = compression_ops.uncompress(variant, dataset.element_spec)
            self.evaluate(uncompressed)

    @combinations.generate(combinations.times(test_base.eager_only_combinations()))
    def testDatasetVariantMismatch(self):
        if False:
            i = 10
            return i + 15
        dataset = dataset_ops.Dataset.from_tensors(dataset_ops.Dataset.range(10))
        with self.assertRaises(TypeError):
            dataset = dataset.map(lambda x: compression_ops.uncompress(x, dataset.element_spec))
            self.getDatasetOutput(dataset)
if __name__ == '__main__':
    test.main()