"""Tests for `tf.data.experimental.group_by_window()`."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test

class GroupByWindowTest(test_base.DatasetTestBase, parameterized.TestCase):

    def _dynamicPad(self, bucket, window, window_size):
        if False:
            for i in range(10):
                print('nop')
        return dataset_ops.Dataset.zip((dataset_ops.Dataset.from_tensors(bucket), window.padded_batch(32, (tensor_shape.TensorShape([]), tensor_shape.TensorShape([None]), tensor_shape.TensorShape([3])))))

    @combinations.generate(test_base.default_test_combinations())
    def testSingleBucket(self):
        if False:
            print('Hello World!')

        def _map_fn(v):
            if False:
                for i in range(10):
                    print('nop')
            return (v, array_ops.fill([v], v), array_ops.fill([3], string_ops.as_string(v)))
        input_dataset = dataset_ops.Dataset.from_tensor_slices(math_ops.range(32)).map(_map_fn)
        bucketed_dataset = input_dataset.group_by_window(key_func=lambda x, y, z: 0, reduce_func=lambda k, bucket: self._dynamicPad(k, bucket, 32), window_size=32)
        get_next = self.getNext(bucketed_dataset)
        (which_bucket, bucketed_values) = self.evaluate(get_next())
        self.assertEqual(0, which_bucket)
        expected_scalar_int = np.arange(32, dtype=np.int64)
        expected_unk_int64 = np.zeros((32, 31)).astype(np.int64)
        for i in range(32):
            expected_unk_int64[i, :i] = i
        expected_vec3_str = np.vstack(3 * [np.arange(32).astype(bytes)]).T
        self.assertAllEqual(expected_scalar_int, bucketed_values[0])
        self.assertAllEqual(expected_unk_int64, bucketed_values[1])
        self.assertAllEqual(expected_vec3_str, bucketed_values[2])

    @combinations.generate(test_base.default_test_combinations())
    def testEvenOddBuckets(self):
        if False:
            print('Hello World!')

        def _map_fn(v):
            if False:
                for i in range(10):
                    print('nop')
            return (v, array_ops.fill([v], v), array_ops.fill([3], string_ops.as_string(v)))
        input_dataset = dataset_ops.Dataset.from_tensor_slices(math_ops.range(64)).map(_map_fn)
        bucketed_dataset = input_dataset.group_by_window(key_func=lambda x, y, z: math_ops.cast(x % 2, dtypes.int64), reduce_func=lambda k, bucket: self._dynamicPad(k, bucket, 32), window_size=32)
        get_next = self.getNext(bucketed_dataset)
        (which_bucket_even, bucketed_values_even) = self.evaluate(get_next())
        (which_bucket_odd, bucketed_values_odd) = self.evaluate(get_next())
        self.assertEqual(3, len(bucketed_values_even))
        self.assertEqual(3, len(bucketed_values_odd))
        self.assertAllEqual(0, which_bucket_even)
        self.assertAllEqual(1, which_bucket_odd)
        expected_scalar_int = np.arange(0, 32 * 2, 2, dtype=np.int64)
        expected_unk_int64 = np.zeros((32, 31 * 2)).astype(np.int64)
        for i in range(0, 32):
            expected_unk_int64[i, :2 * i] = 2 * i
            expected_vec3_str = np.vstack(3 * [np.arange(0, 32 * 2, 2).astype(bytes)]).T
        self.assertAllEqual(expected_scalar_int, bucketed_values_even[0])
        self.assertAllEqual(expected_unk_int64, bucketed_values_even[1])
        self.assertAllEqual(expected_vec3_str, bucketed_values_even[2])
        expected_scalar_int = np.arange(1, 32 * 2 + 1, 2, dtype=np.int64)
        expected_unk_int64 = np.zeros((32, 31 * 2 + 1)).astype(np.int64)
        for i in range(0, 32):
            expected_unk_int64[i, :2 * i + 1] = 2 * i + 1
            expected_vec3_str = np.vstack(3 * [np.arange(1, 32 * 2 + 1, 2).astype(bytes)]).T
        self.assertAllEqual(expected_scalar_int, bucketed_values_odd[0])
        self.assertAllEqual(expected_unk_int64, bucketed_values_odd[1])
        self.assertAllEqual(expected_vec3_str, bucketed_values_odd[2])

    @combinations.generate(test_base.default_test_combinations())
    def testEvenOddBucketsFilterOutAllOdd(self):
        if False:
            i = 10
            return i + 15

        def _map_fn(v):
            if False:
                while True:
                    i = 10
            return {'x': v, 'y': array_ops.fill([v], v), 'z': array_ops.fill([3], string_ops.as_string(v))}

        def _dynamic_pad_fn(bucket, window, _):
            if False:
                return 10
            return dataset_ops.Dataset.zip((dataset_ops.Dataset.from_tensors(bucket), window.padded_batch(32, {'x': tensor_shape.TensorShape([]), 'y': tensor_shape.TensorShape([None]), 'z': tensor_shape.TensorShape([3])})))
        input_dataset = dataset_ops.Dataset.from_tensor_slices(math_ops.range(128)).map(_map_fn).filter(lambda d: math_ops.equal(d['x'] % 2, 0))
        bucketed_dataset = input_dataset.group_by_window(key_func=lambda d: math_ops.cast(d['x'] % 2, dtypes.int64), reduce_func=lambda k, bucket: _dynamic_pad_fn(k, bucket, 32), window_size=32)
        get_next = self.getNext(bucketed_dataset)
        (which_bucket0, bucketed_values_even0) = self.evaluate(get_next())
        (which_bucket1, bucketed_values_even1) = self.evaluate(get_next())
        self.assertAllEqual(0, which_bucket0)
        self.assertAllEqual(0, which_bucket1)
        self.assertAllEqual(np.arange(0, 64, 2, dtype=np.int64), bucketed_values_even0['x'])
        self.assertAllEqual(np.arange(64, 128, 2, dtype=np.int64), bucketed_values_even1['x'])

    @combinations.generate(test_base.default_test_combinations())
    def testDynamicWindowSize(self):
        if False:
            while True:
                i = 10
        components = np.arange(100).astype(np.int64)

        def window_size_func(key):
            if False:
                print('Hello World!')
            window_sizes = constant_op.constant([5, 10], dtype=dtypes.int64)
            return window_sizes[key]
        dataset = dataset_ops.Dataset.from_tensor_slices(components)
        dataset = dataset.group_by_window(key_func=lambda x: x % 2, reduce_func=lambda _, xs: xs.batch(20), window_size=None, window_size_func=window_size_func)
        get_next = self.getNext(dataset)
        with self.assertRaises(errors.OutOfRangeError):
            batches = 0
            while True:
                result = self.evaluate(get_next())
                is_even = all((x % 2 == 0 for x in result))
                is_odd = all((x % 2 == 1 for x in result))
                self.assertTrue(is_even or is_odd)
                expected_batch_size = 5 if is_even else 10
                self.assertEqual(expected_batch_size, result.shape[0])
                batches += 1
        self.assertEqual(batches, 15)

    @combinations.generate(test_base.default_test_combinations())
    def testSimple(self):
        if False:
            i = 10
            return i + 15
        components = np.random.randint(100, size=(200,)).astype(np.int64)
        dataset = dataset_ops.Dataset.from_tensor_slices(components).map(lambda x: x * x)
        dataset = dataset.group_by_window(key_func=lambda x: x % 2, reduce_func=lambda _, xs: xs.batch(4), window_size=4)
        get_next = self.getNext(dataset)
        counts = []
        with self.assertRaises(errors.OutOfRangeError):
            while True:
                result = self.evaluate(get_next())
                self.assertTrue((all((x % 2 == 0 for x in result)) or all(x % 2 == 1) for x in result))
                counts.append(result.shape[0])
        self.assertEqual(len(components), sum(counts))
        num_full_batches = len([c for c in counts if c == 4])
        self.assertGreaterEqual(num_full_batches, 24)
        self.assertTrue(all((c == 4 for c in counts[:num_full_batches])))

    @combinations.generate(test_base.default_test_combinations())
    def testImmediateOutput(self):
        if False:
            while True:
                i = 10
        components = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 0, 0, 2, 2, 0, 0], dtype=np.int64)
        dataset = dataset_ops.Dataset.from_tensor_slices(components)
        dataset = dataset.repeat(-1)
        dataset = dataset.group_by_window(key_func=lambda x: x % 3, reduce_func=lambda _, xs: xs.batch(4), window_size=4)
        get_next = self.getNext(dataset)
        for _ in range(3):
            self.assertAllEqual([0, 0, 0, 0], self.evaluate(get_next()))
            self.assertAllEqual([1, 1, 1, 1], self.evaluate(get_next()))
            self.assertAllEqual([2, 2, 2, 2], self.evaluate(get_next()))
            self.assertAllEqual([0, 0, 0, 0], self.evaluate(get_next()))

    @combinations.generate(test_base.default_test_combinations())
    def testSmallGroups(self):
        if False:
            print('Hello World!')
        components = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0], dtype=np.int64)
        dataset = dataset_ops.Dataset.from_tensor_slices(components)
        dataset = dataset.group_by_window(key_func=lambda x: x % 2, reduce_func=lambda _, xs: xs.batch(4), window_size=4)
        get_next = self.getNext(dataset)
        self.assertAllEqual([0, 0, 0, 0], self.evaluate(get_next()))
        self.assertAllEqual([1, 1, 1, 1], self.evaluate(get_next()))
        self.assertAllEqual([0, 0, 0], self.evaluate(get_next()))
        self.assertAllEqual([1], self.evaluate(get_next()))

    @combinations.generate(test_base.default_test_combinations())
    def testEmpty(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.range(4).group_by_window(key_func=lambda _: 0, reduce_func=lambda _, xs: xs, window_size=0)
        get_next = self.getNext(dataset)
        with self.assertRaisesRegex(errors.InvalidArgumentError, 'Window size must be greater than zero, but got 0.'):
            print(self.evaluate(get_next()))

    @combinations.generate(test_base.default_test_combinations())
    def testReduceFuncError(self):
        if False:
            while True:
                i = 10
        components = np.random.randint(100, size=(200,)).astype(np.int64)

        def reduce_func(_, xs):
            if False:
                for i in range(10):
                    print('nop')
            return xs.padded_batch(4, padded_shapes=(tensor_shape.TensorShape([]), constant_op.constant([5], dtype=dtypes.int64) * -1))
        dataset = dataset_ops.Dataset.from_tensor_slices(components)
        dataset = dataset.map(lambda x: (x, ops.convert_to_tensor([x * x])))
        dataset = dataset.group_by_window(key_func=lambda x, _: x % 2, reduce_func=reduce_func, window_size=32)
        get_next = self.getNext(dataset)
        with self.assertRaises(errors.InvalidArgumentError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testConsumeWindowDatasetMoreThanOnce(self):
        if False:
            for i in range(10):
                print('nop')
        components = np.random.randint(50, size=(200,)).astype(np.int64)

        def reduce_func(key, window):
            if False:
                for i in range(10):
                    print('nop')
            return dataset_ops.Dataset.zip((window.padded_batch(4, padded_shapes=tensor_shape.TensorShape([None])), window.padded_batch(4, padded_shapes=ops.convert_to_tensor([(key + 1) * 10]))))
        dataset = dataset_ops.Dataset.from_tensor_slices(components)
        dataset = dataset.map(lambda x: array_ops.fill([math_ops.cast(x, dtypes.int32)], x))
        dataset = dataset.group_by_window(key_func=lambda x: math_ops.cast(array_ops.shape(x)[0] // 10, dtypes.int64), reduce_func=reduce_func, window_size=4)
        get_next = self.getNext(dataset)
        counts = []
        with self.assertRaises(errors.OutOfRangeError):
            while True:
                (tight_result, multiple_of_10_result) = self.evaluate(get_next())
                self.assertEqual(0, multiple_of_10_result.shape[1] % 10)
                self.assertAllEqual(tight_result, multiple_of_10_result[:, :tight_result.shape[1]])
                counts.append(tight_result.shape[0])
        self.assertEqual(len(components), sum(counts))

    @combinations.generate(test_base.default_test_combinations())
    def testShortCircuit(self):
        if False:
            i = 10
            return i + 15
        dataset = dataset_ops.Dataset.range(10).group_by_window(key_func=lambda x: x, reduce_func=lambda _, window: window.batch(1), window_size=1)
        self.assertDatasetProduces(dataset, expected_output=[[i] for i in range(10)])

    @combinations.generate(test_base.default_test_combinations())
    def testGroupByWindowWithAutotune(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.range(1000).group_by_window(key_func=lambda x: x // 10, reduce_func=lambda key, window: dataset_ops.Dataset.from_tensors(key), window_size=4)
        dataset = dataset.map(lambda x: x + 1, num_parallel_calls=-1)
        get_next = self.getNext(dataset)
        self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testGroupByWindowCardinality(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.range(1).repeat().group_by_window(key_func=lambda x: x % 2, reduce_func=lambda key, window: dataset_ops.Dataset.from_tensors(key), window_size=4)
        self.assertEqual(self.evaluate(dataset.cardinality()), dataset_ops.INFINITE)

    @combinations.generate(test_base.default_test_combinations())
    def testName(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.from_tensors(np.int64(42)).group_by_window(key_func=lambda x: x, reduce_func=lambda key, window: window.batch(4), window_size=4, name='group_by_window')
        self.assertDatasetProduces(dataset, [[42]])

class GroupByWindowCheckpointTest(checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):

    def _build_dataset(self, components):
        if False:
            return 10
        dataset = dataset_ops.Dataset.from_tensor_slices(components).repeat(-1)
        dataset = dataset.group_by_window(key_func=lambda x: x % 3, reduce_func=lambda _, xs: xs.batch(4), window_size=4)
        return dataset

    @combinations.generate(test_base.default_test_combinations())
    def test(self):
        if False:
            for i in range(10):
                print('nop')
        components = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 0, 0, 2, 2, 0, 0], dtype=np.int64)
        self.verify_unused_iterator(lambda : self._build_dataset(components), num_outputs=12, verify_exhausted=False)
        self.verify_multiple_breaks(lambda : self._build_dataset(components), num_outputs=12, verify_exhausted=False)
        self.verify_reset_restored_iterator(lambda : self._build_dataset(components), num_outputs=12, verify_exhausted=False)
if __name__ == '__main__':
    test.main()