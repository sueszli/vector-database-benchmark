"""Verify that memory usage is minimal in eager mode."""
import gc
import time
import weakref
from absl.testing import parameterized
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import internal
try:
    import memory_profiler
except ImportError:
    memory_profiler = None

class MemoryCleanupTest(test_base.DatasetTestBase, parameterized.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(MemoryCleanupTest, self).setUp()
        self._devices = self.configureDevicesForMultiDeviceTest(3)

    def assertMemoryNotIncreasing(self, f, num_iters, max_increase_mb):
        if False:
            return 10
        "Assert memory usage doesn't increase beyond given threshold for f."
        f()
        time.sleep(4)
        initial = memory_profiler.memory_usage(-1)[0]
        for _ in range(num_iters):
            f()
        increase = memory_profiler.memory_usage(-1)[0] - initial
        logging.info('Memory increase observed: %f MB' % increase)
        assert increase < max_increase_mb, 'Increase is too high. Initial memory usage: %f MB. Increase: %f MB. Maximum allowed increase: %f' % (initial, increase, max_increase_mb)

    def assertNoMemoryLeak(self, dataset_fn):
        if False:
            while True:
                i = 10
        'Assert consuming elements from the dataset does not leak memory.'

        def run():
            if False:
                print('Hello World!')
            get_next = self.getNext(dataset_fn())
            for _ in range(100):
                self.evaluate(get_next())
        for _ in range(10):
            run()
        gc.collect()

        def is_native_object(o):
            if False:
                return 10
            if isinstance(o, weakref.ProxyTypes):
                return False
            return isinstance(o, internal.NativeObject)
        tensors = [o for o in gc.get_objects() if is_native_object(o)]
        self.assertEmpty(tensors, '%d Tensors are still alive.' % len(tensors))

    @combinations.generate(test_base.eager_only_combinations())
    def testEagerMemoryUsageWithReset(self):
        if False:
            for i in range(10):
                print('nop')
        if memory_profiler is None:
            self.skipTest('memory_profiler required to run this test')
        dataset = dataset_ops.Dataset.range(10)
        multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(dataset, [self._devices[1], self._devices[2]])

        def f():
            if False:
                for i in range(10):
                    print('nop')
            self.evaluate(multi_device_iterator.get_next())
            multi_device_iterator._eager_reset()
        self.assertMemoryNotIncreasing(f, num_iters=50, max_increase_mb=250)

    @combinations.generate(test_base.eager_only_combinations())
    def testEagerMemoryUsageWithRecreation(self):
        if False:
            return 10
        if memory_profiler is None:
            self.skipTest('memory_profiler required to run this test')
        dataset = dataset_ops.Dataset.range(10)

        def f():
            if False:
                print('Hello World!')
            multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(dataset, [self._devices[1], self._devices[2]])
            self.evaluate(multi_device_iterator.get_next())
            del multi_device_iterator
        self.assertMemoryNotIncreasing(f, num_iters=50, max_increase_mb=250)

    @combinations.generate(test_base.eager_only_combinations())
    def testFilter(self):
        if False:
            i = 10
            return i + 15

        def get_dataset():
            if False:
                i = 10
                return i + 15

            def fn(_):
                if False:
                    while True:
                        i = 10
                return True
            return dataset_ops.Dataset.range(0, 100).filter(fn)
        self.assertNoMemoryLeak(get_dataset)

    @combinations.generate(combinations.combine(tf_api_version=1, mode='eager'))
    def testFilterLegacy(self):
        if False:
            print('Hello World!')

        def get_dataset():
            if False:
                for i in range(10):
                    print('nop')

            def fn(_):
                if False:
                    print('Hello World!')
                return True
            return dataset_ops.Dataset.range(0, 100).filter_with_legacy_function(fn)
        self.assertNoMemoryLeak(get_dataset)

    @combinations.generate(test_base.eager_only_combinations())
    def testFlatMap(self):
        if False:
            for i in range(10):
                print('nop')

        def get_dataset():
            if False:
                for i in range(10):
                    print('nop')

            def fn(x):
                if False:
                    while True:
                        i = 10
                return dataset_ops.Dataset.from_tensors(x * x)
            return dataset_ops.Dataset.range(0, 100).flat_map(fn)
        self.assertNoMemoryLeak(get_dataset)

    @combinations.generate(test_base.eager_only_combinations())
    def testFromGenerator(self):
        if False:
            return 10

        def get_dataset():
            if False:
                while True:
                    i = 10

            def fn():
                if False:
                    print('Hello World!')
                return range(100)
            return dataset_ops.Dataset.from_generator(fn, output_types=dtypes.float32)
        self.assertNoMemoryLeak(get_dataset)

    @combinations.generate(combinations.times(test_base.eager_only_combinations(), combinations.combine(num_parallel_calls=[None, 10])))
    def testMap(self, num_parallel_calls):
        if False:
            print('Hello World!')

        def get_dataset():
            if False:
                while True:
                    i = 10

            def fn(x):
                if False:
                    i = 10
                    return i + 15
                return x * x
            return dataset_ops.Dataset.range(0, 100).map(fn, num_parallel_calls=num_parallel_calls)
        self.assertNoMemoryLeak(get_dataset)

    @combinations.generate(combinations.combine(tf_api_version=1, mode='eager', num_parallel_calls=[None, 10]))
    def testMapLegacy(self, num_parallel_calls):
        if False:
            while True:
                i = 10

        def get_dataset():
            if False:
                i = 10
                return i + 15

            def fn(x):
                if False:
                    for i in range(10):
                        print('nop')
                return x * x
            return dataset_ops.Dataset.range(0, 100).map_with_legacy_function(fn, num_parallel_calls=num_parallel_calls)
        self.assertNoMemoryLeak(get_dataset)

    @combinations.generate(combinations.times(test_base.eager_only_combinations(), combinations.combine(num_parallel_calls=[None, 10])))
    def testInterleave(self, num_parallel_calls):
        if False:
            for i in range(10):
                print('nop')

        def get_dataset():
            if False:
                for i in range(10):
                    print('nop')

            def fn(x):
                if False:
                    return 10
                return dataset_ops.Dataset.from_tensors(x * x)
            return dataset_ops.Dataset.range(0, 100).interleave(fn, num_parallel_calls=num_parallel_calls, cycle_length=10)
        self.assertNoMemoryLeak(get_dataset)
if __name__ == '__main__':
    test.main()