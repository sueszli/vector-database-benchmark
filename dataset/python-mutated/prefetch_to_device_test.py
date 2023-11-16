"""Tests for `tf.data.experimental.prefetch_to_device()`."""
from absl.testing import parameterized
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.experimental.ops import prefetching_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import structure
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

class PrefetchToDeviceTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(test_base.graph_only_combinations())
    def testPrefetchToDevice(self):
        if False:
            for i in range(10):
                print('nop')
        host_dataset = dataset_ops.Dataset.range(10)
        device_dataset = host_dataset.apply(prefetching_ops.prefetch_to_device('/cpu:1'))
        with ops.device('/cpu:1'):
            iterator = dataset_ops.make_one_shot_iterator(device_dataset)
            next_element = iterator.get_next()
        self.assertTrue(structure.are_compatible(dataset_ops.get_structure(host_dataset), dataset_ops.get_structure(device_dataset)))
        self.assertEqual(dtypes.int64, next_element.dtype)
        self.assertEqual([], next_element.shape)
        worker_config = config_pb2.ConfigProto(device_count={'CPU': 2})
        with self.test_session(config=worker_config):
            for i in range(10):
                self.assertEqual(i, self.evaluate(next_element))
            with self.assertRaises(errors.OutOfRangeError):
                self.evaluate(next_element)

    @combinations.generate(test_base.graph_only_combinations())
    def testPrefetchToSameDevice(self):
        if False:
            print('Hello World!')
        host_dataset = dataset_ops.Dataset.range(10)
        device_dataset = host_dataset.apply(prefetching_ops.prefetch_to_device('/job:localhost/replica:0/task:0/device:CPU:0'))
        with ops.device('/cpu:1'):
            iterator = dataset_ops.make_one_shot_iterator(device_dataset)
            next_element = iterator.get_next()
        self.assertTrue(structure.are_compatible(dataset_ops.get_structure(host_dataset), dataset_ops.get_structure(device_dataset)))
        self.assertEqual(dtypes.int64, next_element.dtype)
        self.assertEqual([], next_element.shape)
        worker_config = config_pb2.ConfigProto(device_count={'CPU': 2})
        with self.test_session(config=worker_config):
            for i in range(10):
                self.assertEqual(i, self.evaluate(next_element))
            with self.assertRaises(errors.OutOfRangeError):
                self.evaluate(next_element)

    @combinations.generate(test_base.graph_only_combinations())
    def testPrefetchDictToDevice(self):
        if False:
            while True:
                i = 10
        host_dataset = dataset_ops.Dataset.range(10).map(lambda x: {'a': x})
        device_dataset = host_dataset.apply(prefetching_ops.prefetch_to_device('/cpu:1'))
        with ops.device('/cpu:1'):
            iterator = dataset_ops.make_one_shot_iterator(device_dataset)
            next_element = iterator.get_next()
        self.assertTrue(structure.are_compatible(dataset_ops.get_structure(host_dataset), dataset_ops.get_structure(device_dataset)))
        self.assertEqual(dtypes.int64, next_element['a'].dtype)
        self.assertEqual([], next_element['a'].shape)
        worker_config = config_pb2.ConfigProto(device_count={'CPU': 2})
        with self.test_session(config=worker_config):
            for i in range(10):
                self.assertEqual({'a': i}, self.evaluate(next_element))
            with self.assertRaises(errors.OutOfRangeError):
                self.evaluate(next_element)

    @combinations.generate(test_base.graph_only_combinations())
    def testPrefetchSparseTensorsToDevice(self):
        if False:
            return 10

        def make_tensor(i):
            if False:
                while True:
                    i = 10
            return sparse_tensor.SparseTensorValue(indices=[[0, 0]], values=i * [1], dense_shape=[2, 2])
        host_dataset = dataset_ops.Dataset.range(10).map(make_tensor)
        device_dataset = host_dataset.apply(prefetching_ops.prefetch_to_device('/cpu:1'))
        with ops.device('/cpu:1'):
            iterator = dataset_ops.make_one_shot_iterator(device_dataset)
            next_element = iterator.get_next()
        self.assertTrue(structure.are_compatible(dataset_ops.get_structure(host_dataset), dataset_ops.get_structure(device_dataset)))
        self.assertEqual(dtypes.int64, next_element.dtype)
        worker_config = config_pb2.ConfigProto(device_count={'CPU': 2})
        with self.test_session(config=worker_config):
            for i in range(10):
                actual = self.evaluate(next_element)
                self.assertAllEqual([i], actual.values)
                self.assertAllEqual([[0, 0]], actual.indices)
                self.assertAllEqual([2, 2], actual.dense_shape)
            with self.assertRaises(errors.OutOfRangeError):
                self.evaluate(next_element)

    @combinations.generate(test_base.default_test_combinations())
    def testPrefetchToDeviceGpu(self):
        if False:
            return 10
        if not test_util.is_gpu_available():
            self.skipTest('No GPU available')
        host_dataset = dataset_ops.Dataset.range(10)
        device_dataset = host_dataset.apply(prefetching_ops.prefetch_to_device('/gpu:0'))
        self.assertDatasetProduces(device_dataset, list(range(10)))

    @combinations.generate(test_base.default_test_combinations())
    def testPrefetchToDeviceCorrectPlacement(self):
        if False:
            i = 10
            return i + 15
        if not test_util.is_gpu_available():
            self.skipTest('No GPU available')
        dataset = dataset_ops.Dataset.range(10)
        dataset = dataset.apply(prefetching_ops.prefetch_to_device('/gpu:0'))
        self.assertIn('gpu:0', dataset._variant_tensor.device.lower())

    @combinations.generate(test_base.graph_only_combinations())
    def testPrefetchToDeviceWithReInit(self):
        if False:
            i = 10
            return i + 15
        host_dataset = dataset_ops.Dataset.range(10)
        device_dataset = host_dataset.apply(prefetching_ops.prefetch_to_device('/cpu:1'))
        with ops.device('/cpu:1'):
            iterator = dataset_ops.make_initializable_iterator(device_dataset)
            next_element = iterator.get_next()
        self.assertTrue(structure.are_compatible(dataset_ops.get_structure(host_dataset), dataset_ops.get_structure(device_dataset)))
        self.assertEqual(dtypes.int64, next_element.dtype)
        self.assertEqual([], next_element.shape)
        worker_config = config_pb2.ConfigProto(device_count={'CPU': 2})
        with self.test_session(config=worker_config):
            self.evaluate(iterator.initializer)
            for i in range(5):
                self.assertEqual(i, self.evaluate(next_element))
            self.evaluate(iterator.initializer)
            for i in range(10):
                self.assertEqual(i, self.evaluate(next_element))
            with self.assertRaises(errors.OutOfRangeError):
                self.evaluate(next_element)

    @combinations.generate(test_base.graph_only_combinations())
    def testPrefetchToDeviceGpuWithReInit(self):
        if False:
            for i in range(10):
                print('nop')
        if not test_util.is_gpu_available():
            self.skipTest('No GPU available')
        host_dataset = dataset_ops.Dataset.range(10)
        device_dataset = host_dataset.apply(prefetching_ops.prefetch_to_device('/gpu:0'))
        iterator = dataset_ops.make_initializable_iterator(device_dataset)
        next_element = iterator.get_next()
        with self.cached_session(config=config_pb2.ConfigProto(allow_soft_placement=False)):
            self.evaluate(iterator.initializer)
            for i in range(5):
                self.assertEqual(i, self.evaluate(next_element))
            self.evaluate(iterator.initializer)
            for i in range(10):
                self.assertEqual(i, self.evaluate(next_element))
            with self.assertRaises(errors.OutOfRangeError):
                self.evaluate(next_element)

    @combinations.generate(test_base.eager_only_combinations())
    def testPrefetchToDevicePlacement(self):
        if False:
            print('Hello World!')
        if not test_util.is_gpu_available():
            self.skipTest('No GPU available')
        host_dataset = dataset_ops.Dataset.range(10)
        device_dataset = host_dataset.apply(prefetching_ops.prefetch_to_device('/gpu:0'))
        self.assertEqual(device_dataset._variant_tensor.device, '/job:localhost/replica:0/task:0/device:GPU:0')
if __name__ == '__main__':
    test.main()