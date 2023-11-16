"""Tests for cross_device_utils."""
from absl.testing import parameterized
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

class IndexedSlicesUtilsTest(test.TestCase, parameterized.TestCase):

    def _assert_values_equal(self, left, right):
        if False:
            while True:
                i = 10
        self.assertAllEqual(self.evaluate(ops.convert_to_tensor(left)), self.evaluate(ops.convert_to_tensor(right)))

    @test_util.run_in_graph_and_eager_modes
    def testAggregateTensors(self):
        if False:
            print('Hello World!')
        t0 = constant_op.constant([[1.0, 2.0], [0, 0], [3.0, 4.0]])
        t1 = constant_op.constant([[0.0, 0.0], [5, 6], [7.0, 8.0]])
        total = constant_op.constant([[1.0, 2.0], [5, 6], [10.0, 12.0]])
        result = cross_device_utils.aggregate_tensors_or_indexed_slices([t0, t1])
        self._assert_values_equal(total, result)

    @test_util.run_in_graph_and_eager_modes
    def testAggregateIndexedSlices(self):
        if False:
            print('Hello World!')
        t0 = math_ops._as_indexed_slices(constant_op.constant([[1.0, 2.0], [0, 0], [3.0, 4.0]]))
        t1 = math_ops._as_indexed_slices(constant_op.constant([[0.0, 0.0], [5, 6], [7.0, 8.0]]))
        total = constant_op.constant([[1.0, 2.0], [5, 6], [10.0, 12.0]])
        result = cross_device_utils.aggregate_tensors_or_indexed_slices([t0, t1])
        self.assertIsInstance(result, indexed_slices.IndexedSlices)
        self._assert_values_equal(total, result)

    @test_util.run_in_graph_and_eager_modes
    def testDivideTensor(self):
        if False:
            while True:
                i = 10
        t = constant_op.constant([[1.0, 2.0], [0, 0], [3.0, 4.0]])
        n = 2
        expected = constant_op.constant([[0.5, 1.0], [0, 0], [1.5, 2.0]])
        result = cross_device_utils.divide_by_n_tensors_or_indexed_slices(t, n)
        self._assert_values_equal(expected, result)

    @test_util.run_in_graph_and_eager_modes
    def testDivideIndexedSlices(self):
        if False:
            for i in range(10):
                print('nop')
        t = math_ops._as_indexed_slices(constant_op.constant([[1.0, 2.0], [0, 0], [3.0, 4.0]]))
        n = 2
        expected = constant_op.constant([[0.5, 1.0], [0, 0], [1.5, 2.0]])
        result = cross_device_utils.divide_by_n_tensors_or_indexed_slices(t, n)
        self.assertIsInstance(result, indexed_slices.IndexedSlices)
        self._assert_values_equal(expected, result)

    @test_util.run_in_graph_and_eager_modes
    def testIsIndexedSlices(self):
        if False:
            print('Hello World!')
        t = math_ops._as_indexed_slices(constant_op.constant([[1.0, 2.0], [0, 0], [3.0, 4.0]]))
        self.assertTrue(cross_device_utils.is_indexed_slices(t))

    @combinations.generate(combinations.combine(mode=['graph', 'eager'], required_gpus=1))
    def testCopyTensor(self):
        if False:
            print('Hello World!')
        with ops.device('/cpu:0'):
            t = constant_op.constant([[1.0, 2.0], [0, 0], [3.0, 4.0]])
        destination = '/gpu:0'
        result = cross_device_utils.copy_tensor_or_indexed_slices_to_device(t, destination)
        self._assert_values_equal(t, result)
        self.assertEqual(device_util.resolve(destination), device_util.resolve(result.device))

    @combinations.generate(combinations.combine(mode=['graph', 'eager'], required_gpus=1))
    def testCopyIndexedSlices(self):
        if False:
            while True:
                i = 10
        with ops.device('/cpu:0'):
            t = math_ops._as_indexed_slices(constant_op.constant([[1.0, 2.0], [0, 0], [3.0, 4.0]]))
        destination = '/gpu:0'
        result = cross_device_utils.copy_tensor_or_indexed_slices_to_device(t, destination)
        self.assertIsInstance(result, indexed_slices.IndexedSlices)
        self._assert_values_equal(t, result)
        self.assertEqual(device_util.resolve(destination), device_util.resolve(result.device))

    @combinations.generate(combinations.combine(mode=['graph', 'eager'], required_gpus=1))
    def testCopyIndexedSlicesNoDenseShape(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.device('/cpu:0'):
            t = indexed_slices.IndexedSlices(indices=array_ops.identity([0]), values=array_ops.identity([1.0]))
        destination = '/gpu:0'
        result = cross_device_utils.copy_tensor_or_indexed_slices_to_device(t, destination)
        self.assertIsInstance(result, indexed_slices.IndexedSlices)
        self.assertAllEqual(t.indices, result.indices)
        self.assertAllEqual(t.values, result.values)
        self.assertEqual(device_util.resolve(destination), device_util.resolve(result.device))

class GroupBySizeTest(test.TestCase):

    def testPreferLargerPack(self):
        if False:
            print('Hello World!')
        values = [array_ops.ones([2, 4, 4], dtype=dtypes.float32), array_ops.ones([8], dtype=dtypes.int32), array_ops.ones([10, 10], dtype=dtypes.int64), array_ops.ones([1], dtype=dtypes.int32)]
        packs = cross_device_utils.group_by_size(values, bytes_per_pack=200)
        self.assertLen(packs, 2)
        self.assertLen(packs[0], 3)
        self.assertEqual(packs[0][0].shape, [2, 4, 4])
        self.assertEqual(packs[0][1].shape, [8])
        self.assertEqual(packs[0][2].shape, [10, 10])
        self.assertLen(packs[1], 1)
        self.assertEqual(packs[1][0].shape, [1])

    def testZeroBytesPerPack(self):
        if False:
            while True:
                i = 10
        values = [array_ops.ones([1], dtype=dtypes.float32), array_ops.ones([2], dtype=dtypes.float32)]
        packs = cross_device_utils.group_by_size(values, bytes_per_pack=0)
        self.assertLen(packs, 1)
        self.assertLen(packs[0], 2)
        self.assertEqual(packs[0][0].shape, [1])
        self.assertEqual(packs[0][1].shape, [2])

    def testUnknownShape(self):
        if False:
            for i in range(10):
                print('nop')

        def create_placeholder(shape, dtype):
            if False:
                while True:
                    i = 10
            with ops.Graph().as_default():
                return array_ops.placeholder(dtype=dtype, shape=shape)
        values = [array_ops.ones([10, 10], dtype=dtypes.float32), create_placeholder([None, 10], dtype=dtypes.float32)]
        packs = cross_device_utils.group_by_size(values, bytes_per_pack=1)
        self.assertLen(packs, 1)
        self.assertEqual(packs[0], values)
if __name__ == '__main__':
    test.main()