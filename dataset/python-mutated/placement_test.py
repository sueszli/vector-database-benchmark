"""Tests for tf.data placement within tf.functions."""
from absl.testing import parameterized
from tensorflow.python.data.experimental.ops import prefetching_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class PlacementTest(test_base.DatasetTestBase, parameterized.TestCase):
    """Tests for tf.data placement within tf.functions.

  Specifically, tf.data dataset tensors cannot be copied between devices. These
  tests verify the ops are placed in a way that avoids this.
  """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(PlacementTest, self).setUp()
        config.set_optimizer_experimental_options({'disable_meta_optimizer': True})

    @combinations.generate(test_base.eager_only_combinations())
    def testWhileWithCapturedDataset(self):
        if False:
            return 10
        dataset = dataset_ops.Dataset.range(10)

        @def_function.function
        def f():
            if False:
                while True:
                    i = 10
            total = constant_op.constant(0, dtypes.int64)
            for _ in math_ops.range(1):
                for elem in dataset:
                    total += elem
            return total
        self.assertEqual(f().numpy(), 45)

    @combinations.generate(test_base.eager_only_combinations())
    def testWhile(self):
        if False:
            print('Hello World!')

        @def_function.function
        def f():
            if False:
                print('Hello World!')
            dataset = dataset_ops.Dataset.range(10)
            total = constant_op.constant(0, dtypes.int64)
            for _ in math_ops.range(1):
                for elem in dataset:
                    total += elem
            return total
        self.assertEqual(f().numpy(), 45)

    @combinations.generate(test_base.eager_only_combinations())
    def testCondWithPlacement(self):
        if False:
            return 10

        @def_function.function
        def f():
            if False:
                while True:
                    i = 10
            dataset = dataset_ops.Dataset.range(10)

            def fn():
                if False:
                    for i in range(10):
                        print('nop')
                return dataset.map(lambda x: x + 1)
            c = constant_op.constant(2)
            with ops.device('/cpu:0'):
                a = cond.cond(math_ops.equal(c, 2), fn, fn)
                iterator = iter(a)
                nxt = next(iterator)
            return nxt
        self.assertEqual(f().numpy(), 1)

    @combinations.generate(test_base.eager_only_combinations())
    def testCondWithColocation(self):
        if False:
            return 10

        @def_function.function
        def f():
            if False:
                print('Hello World!')
            dataset = dataset_ops.Dataset.range(8)

            def fn():
                if False:
                    i = 10
                    return i + 15
                return dataset.map(lambda x: x + 1)
            c = constant_op.constant(2)
            with ops.colocate_with(dataset._variant_tensor):
                a = cond.cond(math_ops.equal(c, 2), fn, fn)
                iterator = iter(a)
                nxt = next(iterator)
            return nxt
        self.assertEqual(f().numpy(), 1)

    @combinations.generate(test_base.eager_only_combinations())
    def testCond(self):
        if False:
            return 10

        @def_function.function
        def f():
            if False:
                print('Hello World!')
            dataset = dataset_ops.Dataset.range(8)
            c = constant_op.constant(2)
            a = cond.cond(math_ops.equal(c, 2), lambda : dataset.map(lambda x: x + 1), lambda : dataset.map(lambda x: x + 2))
            return next(iter(a))
        self.assertEqual(f().numpy(), 1)

    @combinations.generate(test_base.eager_only_combinations())
    def testId(self):
        if False:
            while True:
                i = 10

        @def_function.function
        def f():
            if False:
                while True:
                    i = 10
            dataset = dataset_ops.Dataset.range(10)
            dataset = array_ops.identity(dataset)
            return dataset
        f()

    @combinations.generate(test_base.default_test_combinations())
    @test_util.run_gpu_only
    def testFunctionCall(self):
        if False:
            i = 10
            return i + 15

        @def_function.function
        def test_call(dataset):
            if False:
                for i in range(10):
                    print('nop')
            return dataset.reduce(0, lambda s, _: s + 1)

        @def_function.function
        def f():
            if False:
                while True:
                    i = 10
            dataset = dataset_ops.Dataset.range(10)
            return test_call(dataset)
        self.assertEqual(self.evaluate(f()), 10)

    @combinations.generate(test_base.eager_only_combinations())
    @test_util.run_gpu_only
    def testIteratorOnDeviceEagerMode(self):
        if False:
            i = 10
            return i + 15
        dataset = dataset_ops.Dataset.range(10)
        dataset = dataset.apply(prefetching_ops.prefetch_to_device('/gpu:0'))
        iterator = iter(dataset)
        data = next(iterator)
        optional_data = iterator.get_next_as_optional()
        self.assertIn('gpu:0', dataset._variant_tensor.device.lower())
        self.assertIn('gpu:0', iterator._iterator_resource.device.lower())
        self.assertIn('gpu:0', data.device.lower())
        self.assertIn('gpu:0', optional_data.get_value().device.lower())
        self.assertIn('gpu:0', optional_data.has_value().device.lower())

    @combinations.generate(test_base.eager_only_combinations())
    @test_util.run_gpu_only
    def testCreateIteratorInFuncOnGpu(self):
        if False:
            return 10

        @def_function.function
        def create_iter():
            if False:
                print('Hello World!')
            return gen_dataset_ops.anonymous_iterator_v2(output_types=[dtypes.float32], output_shapes=[[]])
        create_iter()

    @combinations.generate(test_base.graph_only_combinations())
    @test_util.run_gpu_only()
    def testIteratorOnDeviceGraphModeOneShotIterator(self):
        if False:
            print('Hello World!')
        self.skipTest('TODO(b/169429285): tf.data.Dataset.make_one_shot_iterator does not support GPU placement.')
        dataset = dataset_ops.Dataset.range(10)
        dataset = dataset.apply(prefetching_ops.prefetch_to_device('/gpu:0'))
        iterator = dataset_ops.make_one_shot_iterator(dataset)
        data = iterator.get_next()
        optional_data = iterator.get_next_as_optional()
        with ops.colocate_with(dataset._variant_tensor):
            dataset_device = test_ops.device_placement_op()
        self.assertIn(b'GPU:0', self.evaluate(dataset_device))
        with ops.colocate_with(iterator._iterator_resource):
            iterator_device = test_ops.device_placement_op()
        self.assertIn(b'GPU:0', self.evaluate(iterator_device))
        with ops.colocate_with(data):
            data_device = test_ops.device_placement_op()
        self.assertIn(b'GPU:0', self.evaluate(data_device))
        with ops.colocate_with(optional_data.get_value()):
            get_value_device = test_ops.device_placement_op()
        self.assertIn(b'GPU:0', self.evaluate(get_value_device))
        with ops.colocate_with(optional_data.has_value()):
            has_value_device = test_ops.device_placement_op()
        self.assertIn(b'GPU:0', self.evaluate(has_value_device))

    @combinations.generate(test_base.graph_only_combinations())
    @test_util.run_gpu_only()
    def testIteratorOnDeviceGraphModeInitializableIterator(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.range(10)
        dataset = dataset.apply(prefetching_ops.prefetch_to_device('/gpu:0'))
        iterator = dataset_ops.make_initializable_iterator(dataset)
        data = iterator.get_next()
        optional_data = iterator.get_next_as_optional()
        with ops.colocate_with(dataset._variant_tensor):
            dataset_device = test_ops.device_placement_op()
        self.assertIn(b'GPU:0', self.evaluate(dataset_device))
        with ops.colocate_with(iterator._iterator_resource):
            iterator_device = test_ops.device_placement_op()
        self.assertIn(b'GPU:0', self.evaluate(iterator_device))
        with ops.colocate_with(data):
            data_device = test_ops.device_placement_op()
        self.assertIn(b'GPU:0', self.evaluate(data_device))
        with ops.colocate_with(optional_data.get_value()):
            get_value_device = test_ops.device_placement_op()
        self.assertIn(b'GPU:0', self.evaluate(get_value_device))
        with ops.colocate_with(optional_data.has_value()):
            has_value_device = test_ops.device_placement_op()
        self.assertIn(b'GPU:0', self.evaluate(has_value_device))

    @combinations.generate(test_base.eager_only_combinations())
    @test_util.run_gpu_only()
    def testIterDatasetEagerModeWithExplicitDevice(self):
        if False:
            i = 10
            return i + 15

        @def_function.function
        def comp():
            if False:
                while True:
                    i = 10
            value = constant_op.constant(0, dtype=dtypes.int64)
            for d in iter(dataset_ops.Dataset.range(10)):
                value += d
            return value
        with ops.device('/gpu:0'):
            result = comp()
        self.assertEqual(result.numpy(), 45)

    @combinations.generate(test_base.eager_only_combinations())
    @test_util.run_gpu_only()
    def testFunctionInliningColocation(self):
        if False:
            return 10

        @def_function.function
        def f(ds):
            if False:
                while True:
                    i = 10
            return next(iter(ds))

        @def_function.function
        def g():
            if False:
                for i in range(10):
                    print('nop')
            dataset = dataset_ops.Dataset.range(10)
            return f(dataset)
        with ops.device('/gpu:0'):
            self.assertEqual(self.evaluate(g()), 0)
if __name__ == '__main__':
    test.main()