"""Tests for `tf.data.experimental.copy_to_device()`."""
from absl.testing import parameterized
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.experimental.ops import prefetching_ops
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.util import structure
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat as util_compat

class CopyToDeviceTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(test_base.graph_only_combinations())
    def testCopyToDevice(self):
        if False:
            for i in range(10):
                print('nop')
        host_dataset = dataset_ops.Dataset.range(10)
        device_dataset = host_dataset.apply(prefetching_ops.copy_to_device('/cpu:1'))
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
    def testCopyToDeviceHostOptimizations(self):
        if False:
            return 10
        host_dataset = dataset_ops.Dataset.range(10)
        host_dataset = host_dataset.apply(testing.assert_next(['MapAndBatch']))
        host_dataset = host_dataset.map(lambda x: x * x).batch(10)
        device_dataset = host_dataset.apply(prefetching_ops.copy_to_device('/cpu:1'))
        with ops.device('/cpu:1'):
            iterator = dataset_ops.make_one_shot_iterator(device_dataset)
            next_element = iterator.get_next()
        worker_config = config_pb2.ConfigProto(device_count={'CPU': 2})
        with self.test_session(config=worker_config):
            self.assertAllEqual([x * x for x in range(10)], self.evaluate(next_element))
            with self.assertRaises(errors.OutOfRangeError):
                self.evaluate(next_element)

    @combinations.generate(test_base.graph_only_combinations())
    def testCopyToDeviceInt32(self):
        if False:
            for i in range(10):
                print('nop')
        host_dataset = dataset_ops.Dataset.from_tensors([0, 1, 2, 3])
        device_dataset = host_dataset.apply(prefetching_ops.copy_to_device('/cpu:1'))
        with ops.device('/cpu:1'):
            iterator = dataset_ops.make_one_shot_iterator(device_dataset)
            next_element = iterator.get_next()
        self.assertTrue(structure.are_compatible(dataset_ops.get_structure(host_dataset), dataset_ops.get_structure(device_dataset)))
        self.assertEqual(dtypes.int32, next_element.dtype)
        self.assertEqual((4,), next_element.shape)
        worker_config = config_pb2.ConfigProto(device_count={'CPU': 2})
        with self.test_session(config=worker_config):
            self.assertAllEqual([0, 1, 2, 3], self.evaluate(next_element))
            with self.assertRaises(errors.OutOfRangeError):
                self.evaluate(next_element)

    @combinations.generate(test_base.graph_only_combinations())
    def testCopyToSameDevice(self):
        if False:
            while True:
                i = 10
        host_dataset = dataset_ops.Dataset.range(10)
        device_dataset = host_dataset.apply(prefetching_ops.copy_to_device('/cpu:0'))
        with ops.device('/cpu:0'):
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
    def testCopyToDeviceWithPrefetch(self):
        if False:
            for i in range(10):
                print('nop')
        host_dataset = dataset_ops.Dataset.range(10)
        device_dataset = host_dataset.apply(prefetching_ops.copy_to_device('/cpu:1')).prefetch(1)
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
    def testCopyDictToDevice(self):
        if False:
            for i in range(10):
                print('nop')
        host_dataset = dataset_ops.Dataset.range(10).map(lambda x: {'a': x})
        device_dataset = host_dataset.apply(prefetching_ops.copy_to_device('/cpu:1'))
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
    def testCopyDictToDeviceWithPrefetch(self):
        if False:
            i = 10
            return i + 15
        host_dataset = dataset_ops.Dataset.range(10).map(lambda x: {'a': x})
        device_dataset = host_dataset.apply(prefetching_ops.copy_to_device('/cpu:1')).prefetch(1)
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
    def testCopySparseTensorsToDevice(self):
        if False:
            i = 10
            return i + 15

        def make_tensor(i):
            if False:
                while True:
                    i = 10
            return sparse_tensor.SparseTensorValue(indices=[[0, 0]], values=i * [1], dense_shape=[2, 2])
        host_dataset = dataset_ops.Dataset.range(10).map(make_tensor)
        device_dataset = host_dataset.apply(prefetching_ops.copy_to_device('/cpu:1'))
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

    @combinations.generate(test_base.graph_only_combinations())
    def testCopySparseTensorsToDeviceWithPrefetch(self):
        if False:
            print('Hello World!')

        def make_tensor(i):
            if False:
                for i in range(10):
                    print('nop')
            return sparse_tensor.SparseTensorValue(indices=[[0, 0]], values=i * [1], dense_shape=[2, 2])
        host_dataset = dataset_ops.Dataset.range(10).map(make_tensor)
        device_dataset = host_dataset.apply(prefetching_ops.copy_to_device('/cpu:1')).prefetch(1)
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
    def testCopyToDeviceGpu(self):
        if False:
            for i in range(10):
                print('nop')
        if not test_util.is_gpu_available():
            self.skipTest('No GPU available')
        host_dataset = dataset_ops.Dataset.range(10)
        device_dataset = host_dataset.apply(prefetching_ops.copy_to_device('/gpu:0'))
        with ops.device('/gpu:0'):
            self.assertDatasetProduces(device_dataset, list(range(10)))

    @combinations.generate(test_base.default_test_combinations())
    def testCopyToDeviceGpuWithPrefetch(self):
        if False:
            while True:
                i = 10
        if not test_util.is_gpu_available():
            self.skipTest('No GPU available')
        host_dataset = dataset_ops.Dataset.range(10)
        device_dataset = host_dataset.apply(prefetching_ops.copy_to_device('/gpu:0')).prefetch(1)
        with ops.device('/gpu:0'):
            self.assertDatasetProduces(device_dataset, list(range(10)))

    @combinations.generate(test_base.graph_only_combinations())
    def testCopyToDeviceGpuWithMap(self):
        if False:
            for i in range(10):
                print('nop')
        if not test_util.is_gpu_available():
            self.skipTest('No GPU available')

        def generator():
            if False:
                for i in range(10):
                    print('nop')
            for i in range(10):
                yield (i, float(i), str(i))
        host_dataset = dataset_ops.Dataset.from_generator(generator, output_types=(dtypes.int32, dtypes.float32, dtypes.string))
        device_dataset = host_dataset.apply(prefetching_ops.copy_to_device('/gpu:0'))

        def gpu_map_func(x, y, z):
            if False:
                for i in range(10):
                    print('nop')
            return (math_ops.square(x), math_ops.square(y), z)
        device_dataset = device_dataset.apply(prefetching_ops.map_on_gpu(gpu_map_func))
        options = options_lib.Options()
        options.autotune.enabled = False
        device_dataset = device_dataset.with_options(options)
        with ops.device('/gpu:0'):
            iterator = dataset_ops.make_initializable_iterator(device_dataset)
            next_element = iterator.get_next()
        with self.cached_session(config=config_pb2.ConfigProto(allow_soft_placement=False)):
            self.evaluate(iterator.initializer)
            for i in range(10):
                (x, y, z) = self.evaluate(next_element)
                self.assertEqual(i ** 2, x)
                self.assertEqual(float(i ** 2), y)
                self.assertEqual(util_compat.as_bytes(str(i)), z)
            with self.assertRaises(errors.OutOfRangeError):
                self.evaluate(next_element)

    @combinations.generate(test_base.graph_only_combinations())
    def testCopyToDeviceGpuInt32(self):
        if False:
            for i in range(10):
                print('nop')
        if not test_util.is_gpu_available():
            self.skipTest('No GPU available')
        host_dataset = dataset_ops.Dataset.from_tensors([0, 1, 2, 3])
        device_dataset = host_dataset.apply(prefetching_ops.copy_to_device('/gpu:0'))
        with ops.device('/gpu:0'):
            iterator = dataset_ops.make_initializable_iterator(device_dataset)
            next_element = iterator.get_next()
        with self.cached_session(config=config_pb2.ConfigProto(allow_soft_placement=False)):
            self.evaluate(iterator.initializer)
            self.assertAllEqual([0, 1, 2, 3], self.evaluate(next_element))
            with self.assertRaises(errors.OutOfRangeError):
                self.evaluate(next_element)

    @combinations.generate(test_base.graph_only_combinations())
    def testCopyToDeviceGpuInt32AndPrefetch(self):
        if False:
            while True:
                i = 10
        if not test_util.is_gpu_available():
            self.skipTest('No GPU available')
        host_dataset = dataset_ops.Dataset.from_tensors([0, 1, 2, 3])
        device_dataset = host_dataset.apply(prefetching_ops.copy_to_device('/gpu:0')).prefetch(1)
        with ops.device('/gpu:0'):
            iterator = dataset_ops.make_initializable_iterator(device_dataset)
            next_element = iterator.get_next()
        with self.cached_session(config=config_pb2.ConfigProto(allow_soft_placement=False)):
            self.evaluate(iterator.initializer)
            self.assertAllEqual([0, 1, 2, 3], self.evaluate(next_element))
            with self.assertRaises(errors.OutOfRangeError):
                self.evaluate(next_element)

    @combinations.generate(test_base.graph_only_combinations())
    def testCopyToDeviceGpuStrings(self):
        if False:
            print('Hello World!')
        if not test_util.is_gpu_available():
            self.skipTest('No GPU available')
        host_dataset = dataset_ops.Dataset.from_tensors(['a', 'b', 'c'])
        device_dataset = host_dataset.apply(prefetching_ops.copy_to_device('/gpu:0'))
        with ops.device('/gpu:0'):
            iterator = dataset_ops.make_initializable_iterator(device_dataset)
            next_element = iterator.get_next()
        with self.cached_session(config=config_pb2.ConfigProto(allow_soft_placement=False)):
            self.evaluate(iterator.initializer)
            self.assertAllEqual([b'a', b'b', b'c'], self.evaluate(next_element))
            with self.assertRaises(errors.OutOfRangeError):
                self.evaluate(next_element)

    @combinations.generate(test_base.graph_only_combinations())
    def testCopyToDeviceGpuStringsAndPrefetch(self):
        if False:
            print('Hello World!')
        if not test_util.is_gpu_available():
            self.skipTest('No GPU available')
        host_dataset = dataset_ops.Dataset.from_tensors(['a', 'b', 'c'])
        device_dataset = host_dataset.apply(prefetching_ops.copy_to_device('/gpu:0'))
        with ops.device('/gpu:0'):
            iterator = dataset_ops.make_initializable_iterator(device_dataset)
            next_element = iterator.get_next()
        with self.cached_session(config=config_pb2.ConfigProto(allow_soft_placement=False)):
            self.evaluate(iterator.initializer)
            self.assertAllEqual([b'a', b'b', b'c'], self.evaluate(next_element))
            with self.assertRaises(errors.OutOfRangeError):
                self.evaluate(next_element)

    @combinations.generate(test_base.graph_only_combinations())
    def testCopyToDevicePingPongCPUGPU(self):
        if False:
            print('Hello World!')
        if not test_util.is_gpu_available():
            self.skipTest('No GPU available')
        host_dataset = dataset_ops.Dataset.range(10)
        device_dataset = host_dataset.apply(prefetching_ops.copy_to_device('/gpu:0', source_device='/cpu:0'))
        back_to_cpu_dataset = device_dataset.apply(prefetching_ops.copy_to_device('/cpu:0', source_device='/gpu:0'))
        with ops.device('/cpu:0'):
            iterator = dataset_ops.make_initializable_iterator(back_to_cpu_dataset)
            next_element = iterator.get_next()
        with self.cached_session(config=config_pb2.ConfigProto(allow_soft_placement=False)):
            self.evaluate(iterator.initializer)
            for i in range(10):
                self.assertEqual(i, self.evaluate(next_element))
            with self.assertRaises(errors.OutOfRangeError):
                self.evaluate(next_element)

    @combinations.generate(test_base.graph_only_combinations())
    def testCopyToDeviceWithReInit(self):
        if False:
            print('Hello World!')
        host_dataset = dataset_ops.Dataset.range(10)
        device_dataset = host_dataset.apply(prefetching_ops.copy_to_device('/cpu:1'))
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
    def testCopyToDeviceWithReInitAndPrefetch(self):
        if False:
            return 10
        host_dataset = dataset_ops.Dataset.range(10)
        device_dataset = host_dataset.apply(prefetching_ops.copy_to_device('/cpu:1')).prefetch(1)
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
    def testCopyToDeviceGpuWithReInit(self):
        if False:
            while True:
                i = 10
        if not test_util.is_gpu_available():
            self.skipTest('No GPU available')
        host_dataset = dataset_ops.Dataset.range(10)
        device_dataset = host_dataset.apply(prefetching_ops.copy_to_device('/gpu:0'))
        with ops.device('/gpu:0'):
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

    @combinations.generate(test_base.graph_only_combinations())
    def testCopyToDeviceGpuWithReInitAndPrefetch(self):
        if False:
            for i in range(10):
                print('nop')
        if not test_util.is_gpu_available():
            self.skipTest('No GPU available')
        host_dataset = dataset_ops.Dataset.range(10)
        device_dataset = host_dataset.apply(prefetching_ops.copy_to_device('/gpu:0')).prefetch(1)
        with ops.device('/gpu:0'):
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

    @combinations.generate(test_base.graph_only_combinations())
    def testIteratorGetNextAsOptionalOnGPU(self):
        if False:
            i = 10
            return i + 15
        if not test_util.is_gpu_available():
            self.skipTest('No GPU available')
        host_dataset = dataset_ops.Dataset.range(3)
        device_dataset = host_dataset.apply(prefetching_ops.copy_to_device('/gpu:0'))
        with ops.device('/gpu:0'):
            iterator = dataset_ops.make_initializable_iterator(device_dataset)
            next_elem = iterator_ops.get_next_as_optional(iterator)
            elem_has_value_t = next_elem.has_value()
            elem_value_t = next_elem.get_value()
        with self.cached_session(config=config_pb2.ConfigProto(allow_soft_placement=False)):
            with self.assertRaises(errors.FailedPreconditionError):
                self.evaluate(elem_has_value_t)
            with self.assertRaises(errors.FailedPreconditionError):
                self.evaluate(elem_value_t)
            self.evaluate(iterator.initializer)
            for i in range(3):
                (elem_has_value, elem_value) = self.evaluate([elem_has_value_t, elem_value_t])
                self.assertTrue(elem_has_value)
                self.assertEqual(i, elem_value)
            for _ in range(2):
                self.assertFalse(self.evaluate(elem_has_value_t))
                with self.assertRaises(errors.InvalidArgumentError):
                    self.evaluate(elem_value_t)
if __name__ == '__main__':
    test.main()