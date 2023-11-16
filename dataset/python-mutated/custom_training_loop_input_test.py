"""Tests for custom training loops."""
from absl.testing import parameterized
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import test_util
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.tpu import tpu
from tensorflow.python.util import nest

def get_dataset_from_tensor_slices(inp_array):
    if False:
        for i in range(10):
            print('nop')
    dataset = dataset_ops.DatasetV2.from_tensor_slices(inp_array)
    if not tf2.enabled():
        dataset = dataset_ops.Dataset.from_tensor_slices(inp_array)
    return dataset

class AssertFlattenedMixin(object):
    """Mixin for specialized asserts."""

    def assert_equal_flattened(self, expected_results, actual_results):
        if False:
            return 10
        'Asserts that flattened results are equal.\n\n    Due to the number of replicas in the strategy, the output may have a\n    different structure and needs to be flattened for comparison.\n\n    Args:\n      expected_results: The results expected as a result of a computation.\n      actual_results: The actual results of a computation.\n    '
        self.assertEqual(len(expected_results), len(actual_results))
        for (i, expected_result) in enumerate(expected_results):
            final_result = []
            actual_result = actual_results[i]
            for val in actual_result:
                final_result.extend(val.numpy())
            self.assertAllEqual(expected_result, final_result)

class InputIterationTest(test.TestCase, parameterized.TestCase, AssertFlattenedMixin):

    @combinations.generate(combinations.combine(distribution=strategy_combinations.all_strategies, mode=['eager']))
    def testConstantNumpyInput(self, distribution):
        if False:
            for i in range(10):
                print('nop')

        @def_function.function
        def run(x):
            if False:
                print('Hello World!')

            def computation(x):
                if False:
                    while True:
                        i = 10
                return math_ops.square(x)
            outputs = distribution.experimental_local_results(distribution.run(computation, args=(x,)))
            return outputs
        self.assertAllEqual(constant_op.constant(4.0, shape=distribution.num_replicas_in_sync), run(2.0))

    @combinations.generate(combinations.combine(distribution=strategy_combinations.all_strategies, mode=['eager']))
    def testStatefulExperimentalRunAlwaysExecute(self, distribution):
        if False:
            while True:
                i = 10
        with distribution.scope():
            v = variables.Variable(0.0, aggregation=variables.VariableAggregation.MEAN)

        @def_function.function
        def train_step():
            if False:
                while True:
                    i = 10

            def assign_add():
                if False:
                    i = 10
                    return i + 15
                v.assign_add(1.0)
            distribution.run(assign_add)
            return array_ops.zeros([])
        train_step()
        self.assertAllEqual(1.0, v.numpy())

    @combinations.generate(combinations.combine(distribution=strategy_combinations.strategies_minus_tpu, mode=['eager']))
    def testFullEager(self, distribution):
        if False:
            i = 10
            return i + 15
        dataset = get_dataset_from_tensor_slices([5.0, 6.0, 7.0, 8.0]).batch(2)

        def train_step(data):
            if False:
                i = 10
                return i + 15
            return math_ops.square(data)
        dist_dataset = distribution.experimental_distribute_dataset(dataset)
        results = []
        for x in dist_dataset:
            output = distribution.experimental_local_results(distribution.run(train_step, args=(x,)))
            results.append(output)
        self.assert_equal_flattened([[25.0, 36.0], [49.0, 64.0]], results)

    @combinations.generate(combinations.combine(distribution=strategy_combinations.all_strategies, mode=['eager']))
    def testGetNextAsOptional(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        data = [5.0, 6.0, 7.0, 8.0]
        dataset = get_dataset_from_tensor_slices(data).batch(2)
        dist_dataset = distribution.experimental_distribute_dataset(dataset)
        iterator = iter(dist_dataset)

        def train_step(data):
            if False:
                for i in range(10):
                    print('nop')
            return math_ops.square(data)

        @def_function.function
        def run(iterator):
            if False:
                return 10
            return distribution.experimental_local_results(distribution.run(train_step, args=(iterator.get_next_as_optional().get_value(),)))
        self.assert_equal_flattened([[25.0, 36.0]], [run(iterator)])

    @combinations.generate(combinations.combine(distribution=strategy_combinations.all_strategies, mode=['eager']))
    def testGetNextAsOptionalExampleUsage(self, distribution):
        if False:
            while True:
                i = 10
        global_batch_size = 2
        steps_per_loop = 6
        dataset = dataset_ops.Dataset.range(8, output_type=dtypes.int32).batch(global_batch_size)
        distributed_iterator = iter(distribution.experimental_distribute_dataset(dataset))

        @def_function.function
        def train_fn(distributed_iterator):
            if False:
                while True:
                    i = 10

            def step_fn(x):
                if False:
                    while True:
                        i = 10
                return x
            for _ in math_ops.range(steps_per_loop):
                optional_data = distributed_iterator.get_next_as_optional()
                if not optional_data.has_value():
                    break
                distribution.run(step_fn, args=(optional_data.get_value(),))
        train_fn(distributed_iterator)

    @combinations.generate(combinations.combine(distribution=strategy_combinations.tpu_strategies, mode=['eager']))
    def testFullEagerTPU(self, distribution):
        if False:
            return 10
        dataset = get_dataset_from_tensor_slices([5.0, 6.0, 7.0, 8.0]).batch(2)

        def train_step(data):
            if False:
                for i in range(10):
                    print('nop')
            return math_ops.square(data)
        input_iterator = iter(distribution.experimental_distribute_dataset(dataset))
        with self.assertRaisesRegex(NotImplementedError, 'does not support pure eager execution'):
            distribution.run(train_step, args=(next(input_iterator),))

    @combinations.generate(combinations.combine(distribution=strategy_combinations.all_strategies, mode=['eager']))
    def testStepInFunction(self, distribution):
        if False:
            while True:
                i = 10
        dataset = get_dataset_from_tensor_slices([5.0, 6.0, 7.0, 8.0]).batch(2)

        @def_function.function
        def train_step(data):
            if False:
                i = 10
                return i + 15
            return math_ops.square(data)
        dist_dataset = distribution.experimental_distribute_dataset(dataset)
        results = []
        for x in dist_dataset:
            output = distribution.experimental_local_results(distribution.run(train_step, args=(x,)))
            results.append(output)
        self.assert_equal_flattened([[25.0, 36.0], [49.0, 64.0]], results)

    @combinations.generate(combinations.combine(distribution=strategy_combinations.all_strategies, mode=['eager']))
    def testRunInFunction(self, distribution):
        if False:
            return 10
        dataset = get_dataset_from_tensor_slices([5.0, 6.0, 7.0, 8.0]).batch(2)

        def train_step(data):
            if False:
                print('Hello World!')
            return math_ops.square(data)

        @def_function.function
        def f_train_step(input_data):
            if False:
                while True:
                    i = 10
            return distribution.experimental_local_results(distribution.run(train_step, args=(input_data,)))
        dist_dataset = distribution.experimental_distribute_dataset(dataset)
        results = []
        for x in dist_dataset:
            output = f_train_step(x)
            results.append(output)
        self.assert_equal_flattened([[25.0, 36.0], [49.0, 64.0]], results)

    @combinations.generate(combinations.combine(distribution=[strategy_combinations.mirrored_strategy_with_gpu_and_cpu, strategy_combinations.tpu_strategy, strategy_combinations.tpu_strategy_packed_var], mode=['eager']))
    def testNestedOutput(self, distribution):
        if False:
            print('Hello World!')
        dataset = get_dataset_from_tensor_slices([0, 1, 2, 3]).batch(2)
        input_iterator = iter(distribution.experimental_distribute_dataset(dataset))

        @def_function.function
        def run(iterator):
            if False:
                print('Hello World!')

            def computation(x):
                if False:
                    while True:
                        i = 10
                return [{'a': x - 1, 'b': x + 1}]
            inputs = next(iterator)
            outputs = distribution.run(computation, args=(inputs,))
            return nest.map_structure(distribution.experimental_local_results, outputs)
        results = run(input_iterator)
        for replica in range(distribution.num_replicas_in_sync):
            self.assertAllEqual(results[0]['a'][replica], [replica - 1])
            self.assertAllEqual(results[0]['b'][replica], [replica + 1])

    @combinations.generate(combinations.combine(distribution=strategy_combinations.all_strategies, mode=['eager']))
    def testRunInFunctionAutoGraphApplication(self, distribution):
        if False:
            return 10
        dataset = get_dataset_from_tensor_slices([5.0, 6.0, 7.0, 8.0]).batch(2)

        def train_step(data):
            if False:
                i = 10
                return i + 15
            return math_ops.square(data)

        @def_function.function
        def f_train_step(input_data):
            if False:
                print('Hello World!')
            return distribution.experimental_local_results(distribution.run(train_step, args=(input_data,)))
        dist_dataset = distribution.experimental_distribute_dataset(dataset)
        results = []
        for x in dist_dataset:
            output = f_train_step(x)
            results.append(output)
        self.assert_equal_flattened([[25.0, 36.0], [49.0, 64.0]], results)

    @combinations.generate(combinations.combine(distribution=strategy_combinations.all_strategies, mode=['eager']))
    def testDatasetIterationInFunction(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        with distribution.scope():
            a = variables.Variable(1.0, aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA)

        def train_step(_):
            if False:
                i = 10
                return i + 15
            a.assign_add(1.0)

        @def_function.function
        def f_train_step(dist_dataset):
            if False:
                while True:
                    i = 10
            number_of_steps = constant_op.constant(0.0)
            product_of_means = constant_op.constant(2.0)
            for x in dist_dataset:
                number_of_steps += 1
                product_of_means *= math_ops.cast(distribution.reduce('MEAN', x, axis=0), product_of_means.dtype)
            for y in dist_dataset:
                distribution.run(train_step, args=(y,))
            return (number_of_steps, product_of_means)
        dataset = get_dataset_from_tensor_slices([5.0, 6.0, 7.0, 8.0]).batch(2)
        dist_dataset = distribution.experimental_distribute_dataset(dataset)
        (number_of_steps, product_of_means) = f_train_step(dist_dataset)
        self.assertEqual(2, number_of_steps.numpy())
        self.assertNear(2 * (5 + 6) / 2 * (7 + 8) / 2, product_of_means.numpy(), 0.001)
        self.assertEqual(3.0, a.numpy())

    @combinations.generate(combinations.combine(distribution=strategy_combinations.all_strategies, mode=['eager']))
    def testDatasetAssertWithDynamicBatch(self, distribution):
        if False:
            i = 10
            return i + 15

        def step_fn(data):
            if False:
                for i in range(10):
                    print('nop')
            assert_op = control_flow_assert.Assert(math_ops.less_equal(math_ops.reduce_max(data), 100.0), [data])
            with ops.control_dependencies([assert_op]):
                return math_ops.square(data)

        @def_function.function
        def train(dataset):
            if False:
                i = 10
                return i + 15
            results = []
            iterator = iter(dataset)
            for _ in range(2):
                elem = next(iterator)
                output = distribution.experimental_local_results(distribution.run(step_fn, args=(elem,)))
                results.append(output)
            return results
        dataset = dataset_ops.DatasetV2.from_tensor_slices([5.0, 6.0, 7.0]).batch(2)
        if not tf2.enabled():
            dataset = dataset_ops.Dataset.from_tensor_slices([5.0, 6.0, 7.0]).batch(2)
        dist_dataset = distribution.experimental_distribute_dataset(dataset)
        results = train(dist_dataset)
        expected_results = [[25.0, 36.0], [49.0]]
        self.assertEqual(len(expected_results), len(results))
        for (i, expected_result) in enumerate(expected_results):
            final_result = []
            actual_result = results[i]
            for val in actual_result:
                final_result.extend(val.numpy())
            self.assertAllEqual(expected_result, final_result)

    @combinations.generate(combinations.combine(distribution=strategy_combinations.all_strategies, mode=['eager']))
    def testDistributeDatasetIteratorWithoutFunction(self, distribution):
        if False:
            i = 10
            return i + 15
        data = [5.0, 6.0, 7.0, 8.0]
        input_iterator = iter(distribution.distribute_datasets_from_function(lambda _: get_dataset_from_tensor_slices(data)))
        self.assertAllEqual(distribution.experimental_local_results(input_iterator.get_next()), data[0:distribution.num_replicas_in_sync])

    @combinations.generate(combinations.combine(distribution=strategy_combinations.multidevice_strategies, mode=['eager']))
    def testDistributeDatasetIteratorWithFunction(self, distribution):
        if False:
            i = 10
            return i + 15
        data = [5.0, 6.0, 7.0, 8.0]
        input_iterator = iter(distribution.distribute_datasets_from_function(lambda _: get_dataset_from_tensor_slices(data)))

        @def_function.function
        def run(iterator):
            if False:
                i = 10
                return i + 15
            return distribution.experimental_local_results(iterator.get_next())
        local_results = run(input_iterator)
        self.assertAllEqual(local_results, data[0:distribution.num_replicas_in_sync])
        backing_devices = [result.backing_device for result in local_results]
        self.assertAllEqual(backing_devices, distribution.extended.worker_devices)

    @combinations.generate(combinations.combine(distribution=strategy_combinations.multidevice_strategies, mode=['eager']))
    def testDistributeDatasetPrefetch(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        data = [5.0, 6.0, 7.0, 8.0]
        input_iterator = iter(distribution.experimental_distribute_dataset(get_dataset_from_tensor_slices(data).batch(2)))
        local_results = distribution.experimental_local_results(input_iterator.get_next())
        backing_devices = [result.backing_device for result in local_results]
        self.assertAllEqual(backing_devices, distribution.extended.worker_devices)

    @combinations.generate(combinations.combine(distribution=strategy_combinations.multidevice_strategies, mode=['eager']))
    def testDistributeDatasetFunctionPrefetch(self, distribution):
        if False:
            return 10
        data = [5.0, 6.0, 7.0, 8.0]
        input_iterator = iter(distribution.distribute_datasets_from_function(lambda _: get_dataset_from_tensor_slices(data)))
        local_results = distribution.experimental_local_results(input_iterator.get_next())
        backing_devices = [result.backing_device for result in local_results]
        self.assertAllEqual(backing_devices, distribution.extended.worker_devices)

    @combinations.generate(combinations.combine(distribution=strategy_combinations.tpu_strategies, mode=['eager']))
    def testDistributeDatasetHostPrefetch(self, distribution):
        if False:
            i = 10
            return i + 15
        data = [5.0, 6.0, 7.0, 8.0]
        input_iterator = iter(distribution.experimental_distribute_dataset(get_dataset_from_tensor_slices(data).batch(2), distribute_lib.InputOptions(experimental_fetch_to_device=False)))
        local_results = distribution.experimental_local_results(input_iterator.get_next())
        for result in local_results:
            self.assertEqual(result.backing_device, device_util.resolve('/device:CPU:0'))

    @combinations.generate(combinations.combine(distribution=strategy_combinations.tpu_strategies, mode=['eager']))
    def testDistributeDatasetFunctionHostPrefetch(self, distribution):
        if False:
            return 10
        data = [5.0, 6.0, 7.0, 8.0]
        input_iterator = iter(distribution.distribute_datasets_from_function(lambda _: get_dataset_from_tensor_slices(data), distribute_lib.InputOptions(experimental_fetch_to_device=False)))
        local_results = distribution.experimental_local_results(input_iterator.get_next())
        for result in local_results:
            self.assertEqual(result.backing_device, device_util.resolve('/device:CPU:0'))

    @combinations.generate(combinations.combine(distribution=strategy_combinations.multidevice_strategies, mode=['eager']))
    def testDynamicShapes(self, distribution):
        if False:
            print('Hello World!')
        dataset = get_dataset_from_tensor_slices([5.0, 6.0, 7.0]).batch(4)
        input_iterator = iter(distribution.experimental_distribute_dataset(dataset))

        @def_function.function
        def run(iterator):
            if False:
                return 10

            def computation(x):
                if False:
                    print('Hello World!')
                return math_ops.reduce_mean(x)
            inputs = next(iterator)
            outputs = distribution.experimental_local_results(distribution.run(computation, args=(inputs,)))
            return outputs
        self.assertAllEqual([5.5, 7.0], run(input_iterator))

    @combinations.generate(combinations.combine(distribution=strategy_combinations.tpu_strategy, mode=['eager']))
    def testDynamicShapesWithRunOptionsBucketizing(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        dataset = get_dataset_from_tensor_slices([5.0, 6.0, 7.0]).batch(4)
        input_iterator = iter(distribution.experimental_distribute_dataset(dataset))
        options = distribute_lib.RunOptions(experimental_bucketizing_dynamic_shape=True)

        @def_function.function
        def run(iterator):
            if False:
                while True:
                    i = 10

            def computation(x):
                if False:
                    while True:
                        i = 10
                return math_ops.reduce_mean(x)
            inputs = next(iterator)
            outputs = distribution.experimental_local_results(distribution.run(computation, args=(inputs,), options=options))
            return outputs
        self.assertAllEqual([5.5, 7.0], run(input_iterator))

    @combinations.generate(combinations.combine(distribution=strategy_combinations.tpu_strategy, mode=['eager']))
    def testDynamicShapesWithRunOptionsDisableDynamicPadder(self, distribution):
        if False:
            i = 10
            return i + 15
        dataset = get_dataset_from_tensor_slices([5, 6, 7]).batch(4)
        mask_dataset = get_dataset_from_tensor_slices([1, 0, 1]).batch(4)
        dataset = dataset_ops.DatasetV2.zip((dataset, mask_dataset))
        input_iterator = iter(distribution.experimental_distribute_dataset(dataset))
        options = distribute_lib.RunOptions(experimental_xla_options=tpu.XLAOptions(enable_xla_dynamic_padder=False))

        @def_function.function
        def run(iterator):
            if False:
                i = 10
                return i + 15

            def computation(inputs):
                if False:
                    i = 10
                    return i + 15
                (x, mask) = inputs
                y = x * mask
                return math_ops.reduce_sum(y)
            inputs = next(iterator)
            outputs = distribution.experimental_local_results(distribution.run(computation, args=(inputs,), options=options))
            return outputs
        self.assertAllEqual([5, 7], run(input_iterator))

    @combinations.generate(combinations.combine(distribution=strategy_combinations.multidevice_strategies, mode=['eager']))
    def testDynamicOutputsWithX64(self, distribution):
        if False:
            while True:
                i = 10
        dataset = get_dataset_from_tensor_slices([5]).map(lambda x: math_ops.cast(x, dtypes.int64)).batch(2)
        input_iterator = iter(distribution.experimental_distribute_dataset(dataset))

        @def_function.function
        def run(iterator):
            if False:
                print('Hello World!')

            def computation(x):
                if False:
                    print('Hello World!')
                return math_ops.add(x, x)
            inputs = next(iterator)
            outputs = distribution.experimental_local_results(distribution.run(computation, args=(inputs,)))
            return outputs
        result = run(input_iterator)
        self.assertAllEqual([10], result[0])
        self.assertAllEqual([], result[1])

    @combinations.generate(combinations.combine(distribution=strategy_combinations.multidevice_strategies, mode=['eager']))
    def testDynamicShapesWithGetNextOutsideFunction(self, distribution):
        if False:
            while True:
                i = 10
        dataset = get_dataset_from_tensor_slices([5.0, 6.0, 7.0]).batch(4)
        input_iterator = iter(distribution.experimental_distribute_dataset(dataset))

        @def_function.function
        def run(inputs):
            if False:
                return 10

            def computation(x):
                if False:
                    i = 10
                    return i + 15
                return math_ops.reduce_mean(x)
            outputs = distribution.experimental_local_results(distribution.run(computation, args=(inputs,)))
            return outputs
        self.assertAllEqual([5.5, 7.0], run(next(input_iterator)))

    @combinations.generate(combinations.combine(distribution=strategy_combinations.multidevice_strategies, mode=['eager']))
    def testStrategyReduceWithDynamicShapes(self, distribution):
        if False:
            print('Hello World!')
        dataset = get_dataset_from_tensor_slices([5.0, 6.0, 7.0]).batch(4)
        input_iterator = iter(distribution.experimental_distribute_dataset(dataset))

        @def_function.function
        def run(iterator):
            if False:
                return 10
            inputs = next(iterator)
            return distribution.reduce(reduce_util.ReduceOp.MEAN, inputs, axis=0)
        self.assertAllEqual(6.0, run(input_iterator))

    @combinations.generate(combinations.combine(distribution=strategy_combinations.multidevice_strategies, mode=['eager']))
    def testStrategyReduceWithDynamicShapesRank2(self, distribution):
        if False:
            while True:
                i = 10
        dataset = get_dataset_from_tensor_slices([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]).batch(4)
        input_iterator = iter(distribution.experimental_distribute_dataset(dataset))

        @def_function.function
        def run(iterator):
            if False:
                i = 10
                return i + 15
            inputs = next(iterator)
            return distribution.reduce(reduce_util.ReduceOp.MEAN, inputs, axis=0)
        self.assertAllEqual([1.0, 1.0], run(input_iterator))

    @combinations.generate(combinations.combine(distribution=strategy_combinations.multidevice_strategies, mode=['eager']))
    def testDynamicShapesWithSizeOp(self, distribution):
        if False:
            return 10
        dataset = get_dataset_from_tensor_slices([5.0, 6.0, 7.0]).batch(4)
        input_iterator = iter(distribution.experimental_distribute_dataset(dataset))

        @def_function.function
        def run(inputs):
            if False:
                i = 10
                return i + 15

            def computation(x):
                if False:
                    for i in range(10):
                        print('nop')
                return array_ops.size_v2(x)
            outputs = distribution.experimental_local_results(distribution.run(computation, args=(inputs,)))
            return outputs
        self.assertAllEqual([2, 1], run(next(input_iterator)))

    @combinations.generate(combinations.combine(distribution=strategy_combinations.multidevice_strategies, mode=['eager']))
    def testSegmentSumWithDynamicNumberOfSegments(self, distribution):
        if False:
            return 10

        def dataset_fn(_):
            if False:
                i = 10
                return i + 15
            data = array_ops.zeros(5, dtype=dtypes.int32)
            dataset = get_dataset_from_tensor_slices(data)
            dataset = dataset.batch(3)
            return dataset
        input_iterator = iter(distribution.distribute_datasets_from_function(dataset_fn))

        @def_function.function
        def step_fn(example):
            if False:
                i = 10
                return i + 15
            segment_ids = array_ops.zeros_like_v2(example)
            num_segment = array_ops.shape(example)[0]
            return math_ops.unsorted_segment_sum(example, segment_ids, num_segment)
        outputs = distribution.experimental_local_results(distribution.run(step_fn, args=(next(input_iterator),)))
        self.assertAllEqual((3,), outputs[0].shape)
        self.assertAllEqual((2,), outputs[1].shape)

    @combinations.generate(combinations.combine(distribution=strategy_combinations.multidevice_strategies, mode=['eager']))
    def testReshapeWithDynamicInputs(self, distribution):
        if False:
            return 10

        def dataset_fn(_):
            if False:
                while True:
                    i = 10
            data = array_ops.zeros((5, 1, 2), dtype=dtypes.int32)
            dataset = get_dataset_from_tensor_slices(data)
            dataset = dataset.batch(3)
            return dataset
        input_iterator = iter(distribution.distribute_datasets_from_function(dataset_fn))

        @def_function.function
        def step_fn(example):
            if False:
                print('Hello World!')
            tile = array_ops.tile(example, [1, array_ops.shape(example)[0], 1])
            reshape1 = array_ops.reshape(tile, [-1, 2])
            reshape2 = array_ops.reshape(reshape1, [array_ops.shape(example)[0], array_ops.shape(example)[0], 2])
            reshape3 = array_ops.reshape(reshape1, [array_ops.shape(example)[0], -1, 2])
            reshape4 = array_ops.reshape(reshape1, [-1, array_ops.shape(example)[0], 2])
            return [reshape1, reshape2, reshape3, reshape4, reshape1]
        outputs = distribution.experimental_local_results(distribution.run(step_fn, args=(next(input_iterator),)))
        self.assertAllEqual((9, 2), outputs[0][0].shape)
        self.assertAllEqual((3, 3, 2), outputs[0][1].shape)
        self.assertAllEqual((3, 3, 2), outputs[0][2].shape)
        self.assertAllEqual((3, 3, 2), outputs[0][3].shape)
        self.assertAllEqual((9, 2), outputs[0][4].shape)
        self.assertAllEqual((4, 2), outputs[1][0].shape)
        self.assertAllEqual((2, 2, 2), outputs[1][1].shape)
        self.assertAllEqual((2, 2, 2), outputs[1][2].shape)
        self.assertAllEqual((2, 2, 2), outputs[1][3].shape)
        self.assertAllEqual((4, 2), outputs[1][4].shape)

    @combinations.generate(combinations.combine(distribution=strategy_combinations.multidevice_strategies, mode=['eager']))
    def testDynamicShapesWithFirstReplicaNotMaximumShape(self, distribution):
        if False:
            print('Hello World!')

        def dataset_fn(_):
            if False:
                return 10
            dataset1 = get_dataset_from_tensor_slices([[1.0, 2.0], [1.0, 2.0]])
            dataset2 = get_dataset_from_tensor_slices([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
            dataset = dataset1.concatenate(dataset2)
            dataset = dataset.batch(2, drop_remainder=True)
            return dataset
        input_iterator = iter(distribution.distribute_datasets_from_function(dataset_fn))

        @def_function.function
        def run(inputs):
            if False:
                return 10

            def computation(x):
                if False:
                    while True:
                        i = 10
                return math_ops.reduce_mean(x)
            outputs = distribution.experimental_local_results(distribution.run(computation, args=(inputs,)))
            return outputs
        self.assertAllEqual([1.5, 2.0], run(next(input_iterator)))

    @combinations.generate(combinations.combine(distribution=strategy_combinations.multidevice_strategies, mode=['eager']))
    def testMapFnWithDynamicInputs(self, distribution):
        if False:
            i = 10
            return i + 15

        def dataset_fn(_):
            if False:
                print('Hello World!')
            data = array_ops.zeros((20, 300, 32), dtype=dtypes.int32)
            dataset = get_dataset_from_tensor_slices(data)
            dataset = dataset.batch(16)
            return dataset
        input_iterator = iter(distribution.distribute_datasets_from_function(dataset_fn))

        def embedding_lookup(inputs):
            if False:
                while True:
                    i = 10
            embedding_weights = array_ops.zeros((1, 128))
            flat_inputs = array_ops.reshape(inputs, [-1])
            embeddings = array_ops.gather(embedding_weights, flat_inputs)
            embeddings = array_ops.reshape(embeddings, inputs.shape.as_list() + [128])
            return embeddings

        @def_function.function
        def step_fn(example):
            if False:
                for i in range(10):
                    print('nop')
            return map_fn.map_fn(embedding_lookup, example, fn_output_signature=dtypes.float32)
        outputs = distribution.experimental_local_results(distribution.run(step_fn, args=(next(input_iterator),)))
        self.assertAllEqual((16, 300, 32, 128), outputs[0].shape)
        self.assertAllEqual((4, 300, 32, 128), outputs[1].shape)

    @combinations.generate(combinations.combine(distribution=strategy_combinations.all_strategies, mode=['eager']))
    def testDatasetDistributeEvenlyDivisibleDrop(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        dataset = get_dataset_from_tensor_slices([5.0, 6.0]).batch(2, drop_remainder=True)
        input_iterator = iter(distribution.experimental_distribute_dataset(dataset))
        data = next(input_iterator)
        expected_result = [5.0, 6.0]
        final_result = []
        actual_result = distribution.experimental_local_results(data)
        for val in actual_result:
            final_result.extend(val)
        self.assertAllEqual(expected_result, final_result)

    @combinations.generate(combinations.combine(distribution=strategy_combinations.all_strategies, mode=['eager']))
    def testDatasetDistributeNotDivisibleDrop(self, distribution):
        if False:
            print('Hello World!')
        dataset = get_dataset_from_tensor_slices([5.0, 6.0]).batch(1, drop_remainder=True)
        input_iterator = iter(distribution.experimental_distribute_dataset(dataset))
        data = next(input_iterator)
        expected_result = [5.0]
        final_result = []
        actual_result = distribution.experimental_local_results(data)
        for val in actual_result:
            final_result.extend(val)
        self.assertAllEqual(expected_result, final_result)

    @combinations.generate(combinations.combine(distribution=strategy_combinations.all_strategies, mode=['eager']))
    def testDatasetDistributeEvenlyDivisibleNoDrop(self, distribution):
        if False:
            return 10
        dataset = get_dataset_from_tensor_slices([5.0, 6.0]).batch(2, drop_remainder=False)
        input_iterator = iter(distribution.experimental_distribute_dataset(dataset))
        data = next(input_iterator)
        expected_result = [5.0, 6.0]
        final_result = []
        actual_result = distribution.experimental_local_results(data)
        for val in actual_result:
            final_result.extend(val)
        self.assertAllEqual(expected_result, final_result)

    @combinations.generate(combinations.combine(distribution=strategy_combinations.all_strategies, mode=['eager']))
    def testDatasetPartialBatchWithMixedOutputs(self, distribution):
        if False:
            while True:
                i = 10
        dataset = get_dataset_from_tensor_slices([5.0]).batch(2)
        input_iterator = iter(distribution.experimental_distribute_dataset(dataset))

        @def_function.function
        def run(iterator):
            if False:
                while True:
                    i = 10

            def computation(x):
                if False:
                    print('Hello World!')
                return (array_ops.zeros([3]), math_ops.square(x))
            return distribution.run(computation, args=(next(iterator),))
        results = run(input_iterator)
        for replica_id in range(distribution.num_replicas_in_sync):
            self.assertAllEqual([0.0, 0.0, 0.0], distribution.experimental_local_results(results[0])[replica_id])
        self.assertAllEqual([25.0], distribution.experimental_local_results(results[1])[0])
        for replica_id in range(1, distribution.num_replicas_in_sync):
            self.assertAllEqual([], distribution.experimental_local_results(results[1])[replica_id])

    @combinations.generate(combinations.combine(distribution=strategy_combinations.all_strategies, mode=['eager']))
    def testIterationInsideFunction(self, distribution):
        if False:
            while True:
                i = 10

        def step_fn(data):
            if False:
                print('Hello World!')
            return math_ops.square(data)

        @def_function.function
        def train(dataset):
            if False:
                print('Hello World!')
            results = []
            iterator = iter(dataset)
            for _ in range(2):
                elem = next(iterator)
                output = distribution.experimental_local_results(distribution.run(step_fn, args=(elem,)))
                results.append(output)
            return results
        dataset = get_dataset_from_tensor_slices([5.0, 6.0, 7.0, 8.0]).batch(2)
        dist_dataset = distribution.experimental_distribute_dataset(dataset)
        results = train(dist_dataset)
        self.assert_equal_flattened([[25.0, 36.0], [49.0, 64.0]], results)

    @combinations.generate(combinations.combine(distribution=strategy_combinations.all_strategies, mode=['eager']))
    def testIterationOutsideFunction(self, distribution):
        if False:
            i = 10
            return i + 15

        def train_step(data):
            if False:
                return 10
            return math_ops.square(data)

        @def_function.function
        def f_train_step(input_data):
            if False:
                i = 10
                return i + 15
            return distribution.experimental_local_results(distribution.run(train_step, args=(input_data,)))
        dataset = get_dataset_from_tensor_slices([5.0, 6.0, 7.0, 8.0]).batch(2)
        dist_dataset = distribution.experimental_distribute_dataset(dataset)
        iterator = iter(dist_dataset)
        results = []
        for _ in range(2):
            output = f_train_step(next(iterator))
            results.append(output)
        self.assert_equal_flattened([[25.0, 36.0], [49.0, 64.0]], results)

    @combinations.generate(combinations.combine(distribution=strategy_combinations.all_strategies, mode=['eager']))
    def testMultiDeviceDataCapturedFunction(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        inputs = constant_op.constant([2.0, 3.0])
        dataset = lambda _: dataset_ops.Dataset.from_tensor_slices(inputs).repeat(5)
        input_iterator = iter(distribution.distribute_datasets_from_function(dataset))
        with distribution.scope():
            var = variables.Variable(1.0)

        @def_function.function
        def train_step(input_iterator):
            if False:
                print('Hello World!')

            def func(inputs):
                if False:
                    i = 10
                    return i + 15
                return math_ops.square(inputs) + var
            per_replica_outputs = distribution.run(func, (next(input_iterator),))
            mean = distribution.reduce(reduce_util.ReduceOp.MEAN, per_replica_outputs, axis=None)
            for _ in dataset_ops.Dataset.range(1):
                per_replica_outputs = distribution.run(func, (next(input_iterator),))
                mean = distribution.reduce(reduce_util.ReduceOp.MEAN, per_replica_outputs, axis=None)
            return mean
        with distribution.scope():
            if distribution.num_replicas_in_sync == 1:
                self.assertAlmostEqual(10.0, self.evaluate(train_step(input_iterator)))
            else:
                self.assertAlmostEqual(7.5, self.evaluate(train_step(input_iterator)))

    @combinations.generate(combinations.combine(distribution=strategy_combinations.all_strategies, mode=['eager']))
    def testDatasetOutOfRange(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        with distribution.scope():
            a = variables.Variable(0.0, aggregation=variables.VariableAggregation.SUM)

        def train_step(val):
            if False:
                while True:
                    i = 10
            a.assign_add(math_ops.reduce_sum(val))

        @def_function.function
        def f_train_step(iterator):
            if False:
                return 10
            distribution.run(train_step, args=(next(iterator),))
            return a
        dataset = get_dataset_from_tensor_slices([5.0, 6.0, 7.0, 8.0]).batch(2)
        dist_dataset = distribution.experimental_distribute_dataset(dataset)
        iterator = iter(dist_dataset)
        with self.assertRaises(errors.OutOfRangeError):
            for _ in range(100):
                f_train_step(iterator)
        self.assertAlmostEqual(26.0, a.numpy())

    @combinations.generate(combinations.combine(distribution=strategy_combinations.multidevice_strategies, mode=['eager']))
    def testComputeLossWithDynamicShapes(self, distribution):
        if False:
            while True:
                i = 10
        dataset = get_dataset_from_tensor_slices([5.0, 6.0, 7.0]).batch(4)
        input_iterator = iter(distribution.experimental_distribute_dataset(dataset))

        @def_function.function
        def run(iterator):
            if False:
                while True:
                    i = 10

            def computation(x):
                if False:
                    print('Hello World!')
                return losses.compute_weighted_loss(x, weights=array_ops.ones_like(x))
            inputs = next(iterator)
            outputs = distribution.experimental_local_results(distribution.run(computation, args=(inputs,)))
            return outputs
        self.assertAllEqual([5.5, 7.0], run(input_iterator))
if __name__ == '__main__':
    test_util.main()