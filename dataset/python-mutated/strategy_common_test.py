"""Tests for common methods in strategy classes."""
from absl.testing import parameterized
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute.collective_all_reduce_strategy import CollectiveAllReduceStrategy
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.util import nest

@combinations.generate(combinations.combine(strategy=[strategy_combinations.multi_worker_mirrored_2x1_cpu, strategy_combinations.multi_worker_mirrored_2x1_gpu] + strategy_combinations.all_strategies, mode=['eager']))
class StrategyTest(test.TestCase, parameterized.TestCase):

    def testCaptureReplicaId(self, strategy):
        if False:
            return 10
        m = {}

        @def_function.function
        def f():
            if False:
                i = 10
                return i + 15
            return distribute_lib.get_replica_context().replica_id_in_sync_group

        @def_function.function
        def g():
            if False:
                return 10
            if m.get('v', None) is None:
                m['v'] = variables.Variable(0.0)
            return strategy.run(f)
        g()

    def testMergeCallInitScope(self, strategy):
        if False:
            return 10
        with strategy.scope():

            @def_function.function
            def fn():
                if False:
                    return 10

                def merge_fn(unused_strat):
                    if False:
                        return 10
                    y = constant_op.constant(11)
                    return y

                def replica_fn():
                    if False:
                        for i in range(10):
                            print('nop')
                    with ops.init_scope():
                        y = distribute_lib.get_replica_context().merge_call(merge_fn)
                        z = y + 1
                        return z
                return strategy.run(replica_fn)
            result = strategy.experimental_local_results(fn())
            self.assertAllClose(result, [12] * _get_num_replicas_per_client(strategy))

@combinations.generate(combinations.combine(distribution=[strategy_combinations.mirrored_strategy_with_cpu_1_and_2, strategy_combinations.multi_worker_mirrored_2x2_gpu, strategy_combinations.tpu_strategy], mode=['graph', 'eager']))
class StrategyLocalResultTest(test.TestCase):

    def testLocalResultForDictionary(self, distribution):
        if False:
            i = 10
            return i + 15

        @def_function.function
        def model_fn():
            if False:
                i = 10
                return i + 15
            return {'a': constant_op.constant(1.0), 'b': constant_op.constant(2.0)}
        with distribution.scope():
            result = distribution.run(model_fn)
            got = self.evaluate(distribution.experimental_local_results(result))
            self.assertEqual(got, ({'a': 1.0, 'b': 2.0}, {'a': 1.0, 'b': 2.0}))

    def testLocalResultForList(self, distribution):
        if False:
            i = 10
            return i + 15

        @def_function.function
        def model_fn():
            if False:
                for i in range(10):
                    print('nop')
            return [constant_op.constant(1.0), constant_op.constant(2.0)]
        with distribution.scope():
            result = distribution.run(model_fn)
            got = self.evaluate(distribution.experimental_local_results(result))
            self.assertEqual(got, ([1.0, 2.0], [1.0, 2.0]))

    def testLocalResultForTuple(self, distribution):
        if False:
            i = 10
            return i + 15

        @def_function.function
        def model_fn():
            if False:
                while True:
                    i = 10
            return (constant_op.constant(1.0), constant_op.constant(2.0), constant_op.constant(3.0))
        with distribution.scope():
            result = distribution.run(model_fn)
            got = self.evaluate(distribution.experimental_local_results(result))
            self.assertEqual(got, ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)))

    def testLocalResultForNestedStruct(self, distribution):
        if False:
            while True:
                i = 10

        @def_function.function
        def model_fn():
            if False:
                for i in range(10):
                    print('nop')
            return ({'a': constant_op.constant(1.0), 'b': constant_op.constant(2.0)}, {'a': constant_op.constant(4.0), 'b': constant_op.constant(6.0)})
        with distribution.scope():
            result = distribution.run(model_fn)
            got = self.evaluate(distribution.experimental_local_results(result))
            self.assertEqual(got, (({'a': 1.0, 'b': 2.0}, {'a': 4.0, 'b': 6.0}), ({'a': 1.0, 'b': 2.0}, {'a': 4.0, 'b': 6.0})))

    def testLocalResultForNestedStructWithoutTensor(self, distribution):
        if False:
            print('Hello World!')

        @def_function.function
        def model_fn():
            if False:
                for i in range(10):
                    print('nop')
            return {'a': 1.0, 'b': 2.0}
        with distribution.scope():
            result = distribution.run(model_fn)
            v = self.evaluate(distribution.experimental_local_results(result))
            self.assertIsInstance(v, tuple)
            self.assertAllEqual(v, ({'a': 1.0, 'b': 2.0}, {'a': 1.0, 'b': 2.0}))

    def testLocalResultForScalarValue(self, distribution):
        if False:
            while True:
                i = 10

        @def_function.function
        def model_fn():
            if False:
                while True:
                    i = 10
            return distribution.extended._get_local_replica_id(distribute_lib.get_replica_context().replica_id_in_sync_group)
        with distribution.scope():
            result = distribution.run(model_fn)
            v = self.evaluate(distribution.experimental_local_results(result))
            self.assertIsInstance(v, tuple)
            self.assertEqual(v, (0, 1))

    def testLocalResultForDictionaryDifferentReplicas(self, distribution):
        if False:
            print('Hello World!')

        @def_function.function
        def model_fn():
            if False:
                print('Hello World!')
            replica_id = distribution.extended._get_local_replica_id(distribute_lib.get_replica_context().replica_id_in_sync_group)
            return {'a': math_ops.cast(replica_id + 1, dtype=float), 'b': math_ops.cast(replica_id + 2, dtype=float)}
        with distribution.scope():
            result = distribution.run(model_fn)
            got = self.evaluate(distribution.experimental_local_results(result))
            self.assertAllEqual(got, ({'a': 1.0, 'b': 2.0}, {'a': 2.0, 'b': 3.0}))

    def testLocalResultForTensor(self, distribution):
        if False:
            return 10

        @def_function.function
        def model_fn():
            if False:
                while True:
                    i = 10
            return constant_op.constant([2.0, 3.0])
        with distribution.scope():
            result = distribution.run(model_fn)
            v = self.evaluate(distribution.experimental_local_results(result))
            self.assertAllEqual(v, ([2.0, 3.0], [2.0, 3.0]))

@combinations.generate(combinations.combine(strategy=[strategy_combinations.multi_worker_mirrored_2x1_cpu, strategy_combinations.multi_worker_mirrored_2x1_gpu] + strategy_combinations.all_strategies, mode=['eager']))
class ReduceTest(test.TestCase, parameterized.TestCase):

    def testBasic(self, strategy):
        if False:
            i = 10
            return i + 15
        per_replica_value = strategy.experimental_distribute_values_from_function(lambda _: array_ops.ones((), dtypes.float32))

        def fn_eager():
            if False:
                return 10
            return strategy.reduce(reduce_util.ReduceOp.SUM, value=per_replica_value, axis=None)
        fn_graph = def_function.function(fn_eager)
        with strategy.scope():
            self.assertEqual(fn_eager().numpy(), 1.0 * strategy.num_replicas_in_sync)
            self.assertEqual(fn_graph().numpy(), 1.0 * strategy.num_replicas_in_sync)
        self.assertEqual(fn_eager().numpy(), 1.0 * strategy.num_replicas_in_sync)
        self.assertEqual(fn_graph().numpy(), 1.0 * strategy.num_replicas_in_sync)

    def testAxis(self, strategy):
        if False:
            while True:
                i = 10

        @def_function.function
        def fn():
            if False:
                i = 10
                return i + 15
            return constant_op.constant([1.0, 2.0])
        x = strategy.run(fn)
        x_m = strategy.reduce(reduce_util.ReduceOp.MEAN, x, axis=0)
        self.assertEqual(1.5, x_m)
        x_s = strategy.reduce(reduce_util.ReduceOp.SUM, x, axis=0)
        self.assertEqual(3 * strategy.num_replicas_in_sync, x_s)

@combinations.generate(combinations.combine(strategy=[strategy_combinations.default_strategy, strategy_combinations.mirrored_strategy_with_gpu_and_cpu, strategy_combinations.mirrored_strategy_with_two_gpus_no_merge_call, strategy_combinations.tpu_strategy, strategy_combinations.tpu_strategy_packed_var, strategy_combinations.multi_worker_mirrored_2x1_cpu, strategy_combinations.multi_worker_mirrored_2x2_gpu, strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call], update_fn=['assign', 'assign_add', 'assign_sub'], tf_function=[True, False], mode=['eager']))
class ReplicaCtxUpdateTest(test.TestCase, parameterized.TestCase):

    def testDenseUpdate(self, strategy, tf_function, update_fn):
        if False:
            return 10
        if strategy_test_lib.is_tpu_strategy(strategy) and (not tf_function):
            self.skipTest('Skip TPUStrategy + eager combination.')
        with strategy.scope():
            distributed_variable1 = variables.Variable(5.0)

        def replica_fn():
            if False:
                for i in range(10):
                    print('nop')
            value = array_ops.constant(2.0)
            python_literal = 1.0
            replica_context = distribute_lib.get_replica_context()
            fn_sets = {'assign': lambda var, value: var.assign(value), 'assign_add': lambda var, value: var.assign_add(value), 'assign_sub': lambda var, value: var.assign_sub(value)}
            replica_context._update(distributed_variable1, fn_sets[update_fn], args=(value,))
            replica_context._update(distributed_variable1, fn_sets[update_fn], args=(python_literal,))
        if tf_function:
            replica_fn = def_function.function(replica_fn)
        strategy.run(replica_fn)
        expected_result = {'assign': 1.0, 'assign_add': 8.0, 'assign_sub': 2.0}
        self.assertAllEqual(strategy.experimental_local_results(distributed_variable1), [expected_result[update_fn]] * _get_num_replicas_per_client(strategy))

@combinations.generate(combinations.combine(strategy=[strategy_combinations.multi_worker_mirrored_2x1_cpu, strategy_combinations.multi_worker_mirrored_2x1_gpu, strategy_combinations.multi_worker_mirrored_2x2_gpu, strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call, strategy_combinations.tpu_strategy] + strategy_combinations.strategies_minus_tpu, tf_function=[combinations.tf_function, combinations.no_tf_function], mode=['eager']))
class ReplicaCtxAllReduceTest(test.TestCase, parameterized.TestCase):

    def testDense(self, strategy, tf_function):
        if False:
            print('Hello World!')
        if strategy_test_lib.is_tpu_strategy(strategy) and tf_function is combinations.no_tf_function:
            self.skipTest('Skip TPUStrategy + eager combination.')

        @tf_function
        def fn():
            if False:
                while True:
                    i = 10

            def replica_fn():
                if False:
                    return 10
                value = array_ops.identity(1.0)
                reduced = strategy.extended._replica_ctx_all_reduce(reduce_util.ReduceOp.SUM, value)
                return reduced
            return strategy.experimental_local_results(strategy.run(replica_fn))
        got = fn()[0]
        self.assertEqual(got, 1.0 * strategy.num_replicas_in_sync)

    def testSparse(self, strategy, tf_function):
        if False:
            for i in range(10):
                print('nop')
        if tf_function is combinations.no_tf_function:
            self.skipTest('Skip IndexedSlices + eager combination.')

        @tf_function
        def fn():
            if False:
                for i in range(10):
                    print('nop')

            def replica_fn():
                if False:
                    print('Hello World!')
                value = indexed_slices.IndexedSlices(values=array_ops.identity([[1.0]]), indices=array_ops.identity([0]), dense_shape=array_ops.identity([5, 1]))
                reduced = strategy.extended._replica_ctx_all_reduce(reduce_util.ReduceOp.SUM, value)
                return reduced
            return strategy.experimental_local_results(strategy.run(replica_fn))
        got = fn()[0]
        expect = indexed_slices.IndexedSlices(values=array_ops.identity([[1.0 * strategy.num_replicas_in_sync]]), indices=array_ops.identity([0]), dense_shape=array_ops.identity([5, 1]))
        self.assertAllEqual(ops.convert_to_tensor(got), ops.convert_to_tensor(expect))

    def testNestedInput(self, strategy, tf_function):
        if False:
            while True:
                i = 10
        if tf_function is combinations.no_tf_function:
            self.skipTest('Skip IndexedSlices + eager combination.')

        @tf_function
        def fn():
            if False:
                print('Hello World!')

            def replica_fn():
                if False:
                    return 10
                value = (array_ops.identity(1.0), indexed_slices.IndexedSlices(values=array_ops.identity([[1.0]]), indices=array_ops.identity([0]), dense_shape=array_ops.identity([5, 1])), array_ops.identity(2.0), indexed_slices.IndexedSlices(values=array_ops.identity([[2.0]]), indices=array_ops.identity([1]), dense_shape=array_ops.identity([5, 1])))
                reduced = strategy.extended._replica_ctx_all_reduce(reduce_util.ReduceOp.SUM, value)
                return reduced
            return strategy.experimental_local_results(strategy.run(replica_fn))
        got = fn()[0]
        expect = (1.0 * strategy.num_replicas_in_sync, indexed_slices.IndexedSlices(values=array_ops.identity([[1.0 * strategy.num_replicas_in_sync]]), indices=array_ops.identity([0]), dense_shape=array_ops.identity([5, 1])), 2.0 * strategy.num_replicas_in_sync, indexed_slices.IndexedSlices(values=array_ops.identity([[2.0 * strategy.num_replicas_in_sync]]), indices=array_ops.identity([1]), dense_shape=array_ops.identity([5, 1])))
        self.assertAllClose(nest.map_structure(ops.convert_to_tensor, got), nest.map_structure(ops.convert_to_tensor, expect))

    def testSyncOnReadVariableInput(self, strategy, tf_function):
        if False:
            print('Hello World!')
        if not strategy_test_lib.is_mirrored_strategy(strategy) and (not strategy_test_lib.is_multi_worker_mirrored_strategy(strategy)) and (not strategy_test_lib.is_tpu_strategy(strategy)):
            self.skipTest('Skip strategies not using SyncOnReadVariables.')
        if strategy_test_lib.is_tpu_strategy(strategy) and tf_function is combinations.no_tf_function:
            self.skipTest('Skip TPUStrategy + eager combination.')
        if strategy_test_lib.is_multi_worker_mirrored_strategy(strategy) and tf_function is combinations.tf_function:
            self.skipTest('Skip MWMS + graph combination until b/228512201 is fixed.')
        with strategy.scope():
            var = variables.Variable(0.0, synchronization=variables.VariableSynchronization.ON_READ, aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA)

        @tf_function
        def replica_fn():
            if False:
                while True:
                    i = 10
            replica_context = distribute_lib.get_replica_context()
            replica_id = replica_context.replica_id_in_sync_group
            var.assign(math_ops.cast(replica_id, dtype=float) * 3.0)
            return replica_context.all_reduce(reduce_util.ReduceOp.SUM, var)
        if strategy_test_lib.is_multi_worker_mirrored_strategy(strategy):
            client_local_replica_num = strategy.extended._num_devices_per_worker
        else:
            client_local_replica_num = strategy.num_replicas_in_sync
        workers_num = strategy.num_replicas_in_sync
        expected_sum = sum(range(workers_num)) * 3.0
        result = strategy.run(replica_fn)
        if hasattr(result, 'values'):
            result = result.values
        result = nest.flatten(result)
        for i in range(client_local_replica_num):
            self.assertEqual(result[i].numpy(), expected_sum)

@combinations.generate(combinations.combine(strategy=[strategy_combinations.multi_worker_mirrored_2x1_cpu, strategy_combinations.multi_worker_mirrored_2x1_gpu, strategy_combinations.multi_worker_mirrored_2x2_gpu, strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call, strategy_combinations.tpu_strategy] + strategy_combinations.strategies_minus_tpu, tf_function=[combinations.tf_function, combinations.no_tf_function], mode=['eager']))
class AllReduceTest(test.TestCase, parameterized.TestCase):

    def testDense(self, strategy, tf_function):
        if False:
            print('Hello World!')
        if strategy_test_lib.is_tpu_strategy(strategy) and tf_function is combinations.no_tf_function:
            self.skipTest('Skip TPUStrategy + eager combination.')

        @tf_function
        def fn():
            if False:
                for i in range(10):
                    print('nop')

            def replica_fn():
                if False:
                    for i in range(10):
                        print('nop')
                value = array_ops.identity(1.0)
                rep_ctx = distribute_lib.get_replica_context()
                reduced = rep_ctx.all_reduce(reduce_util.ReduceOp.SUM, value)
                return reduced
            return strategy.experimental_local_results(strategy.run(replica_fn))
        got = fn()[0]
        self.assertEqual(got, 1.0 * strategy.num_replicas_in_sync)

    def testSparse(self, strategy, tf_function):
        if False:
            return 10
        if tf_function is combinations.no_tf_function:
            self.skipTest('Skip IndexedSlices + eager combination.')

        @tf_function
        def fn():
            if False:
                while True:
                    i = 10

            def replica_fn():
                if False:
                    for i in range(10):
                        print('nop')
                value = indexed_slices.IndexedSlices(values=array_ops.identity([[1.0]]), indices=array_ops.identity([0]), dense_shape=array_ops.identity([5, 1]))
                rep_ctx = distribute_lib.get_replica_context()
                reduced = rep_ctx.all_reduce(reduce_util.ReduceOp.MEAN, value)
                return reduced
            return strategy.experimental_local_results(strategy.run(replica_fn))
        got = fn()[0]
        if not strategy_test_lib.is_tpu_strategy(strategy):
            self.assertIsInstance(got, indexed_slices.IndexedSlices)
        expect = indexed_slices.IndexedSlices(values=array_ops.identity([[1.0]]), indices=array_ops.identity([0]), dense_shape=array_ops.identity([5, 1]))
        self.assertAllEqual(ops.convert_to_tensor(got), ops.convert_to_tensor(expect))

    def testSparseTuple(self, strategy, tf_function):
        if False:
            while True:
                i = 10
        if tf_function is combinations.no_tf_function:
            self.skipTest('Skip IndexedSlices + eager combination.')

        @tf_function
        def fn():
            if False:
                return 10

            def replica_fn():
                if False:
                    for i in range(10):
                        print('nop')
                value1 = indexed_slices.IndexedSlices(values=array_ops.identity([[1.0]]), indices=array_ops.identity([0]), dense_shape=array_ops.identity([5, 1]))
                value2 = indexed_slices.IndexedSlices(values=array_ops.identity([[2.0]]), indices=array_ops.identity([0]), dense_shape=array_ops.identity([5, 1]))
                rep_ctx = distribute_lib.get_replica_context()
                reduced = rep_ctx.all_reduce(reduce_util.ReduceOp.SUM, [value1, value2])
                return reduced
            return strategy.experimental_local_results(strategy.run(replica_fn))
        got = fn()[0]
        if not strategy_test_lib.is_tpu_strategy(strategy):
            for g in got:
                self.assertIsInstance(g, indexed_slices.IndexedSlices)
        expect = [indexed_slices.IndexedSlices(values=array_ops.identity([[1.0 * strategy.num_replicas_in_sync]]), indices=array_ops.identity([0]), dense_shape=array_ops.identity([5, 1])), indexed_slices.IndexedSlices(values=array_ops.identity([[2.0 * strategy.num_replicas_in_sync]]), indices=array_ops.identity([0]), dense_shape=array_ops.identity([5, 1]))]
        self.assertAllEqual(nest.map_structure(ops.convert_to_tensor, got), nest.map_structure(ops.convert_to_tensor, expect))

    def testNestedInput(self, strategy, tf_function):
        if False:
            for i in range(10):
                print('nop')
        if tf_function is combinations.no_tf_function:
            self.skipTest('Skip IndexedSlices + eager combination.')

        @tf_function
        def fn():
            if False:
                i = 10
                return i + 15

            def replica_fn():
                if False:
                    for i in range(10):
                        print('nop')
                value = (array_ops.identity(1.0), indexed_slices.IndexedSlices(values=array_ops.identity([[1.0]]), indices=array_ops.identity([0]), dense_shape=array_ops.identity([5, 1])), array_ops.identity(2.0), indexed_slices.IndexedSlices(values=array_ops.identity([[2.0]]), indices=array_ops.identity([1]), dense_shape=array_ops.identity([5, 1])))
                rep_ctx = distribute_lib.get_replica_context()
                reduced = rep_ctx.all_reduce(reduce_util.ReduceOp.SUM, value)
                return reduced
            return strategy.experimental_local_results(strategy.run(replica_fn))
        got = fn()[0]
        expect = (1.0 * strategy.num_replicas_in_sync, indexed_slices.IndexedSlices(values=array_ops.identity([[1.0 * strategy.num_replicas_in_sync]]), indices=array_ops.identity([0]), dense_shape=array_ops.identity([5, 1])), 2.0 * strategy.num_replicas_in_sync, indexed_slices.IndexedSlices(values=array_ops.identity([[2.0 * strategy.num_replicas_in_sync]]), indices=array_ops.identity([1]), dense_shape=array_ops.identity([5, 1])))
        self.assertAllClose(nest.map_structure(ops.convert_to_tensor, got), nest.map_structure(ops.convert_to_tensor, expect))

def _make_indexed_slices(values, indices, dense_shape):
    if False:
        print('Hello World!')
    tensor = indexed_slices.IndexedSlices(values=constant_op.constant(values), indices=constant_op.constant(indices), dense_shape=constant_op.constant(dense_shape))
    return tensor

def _get_num_replicas_per_client(strategy):
    if False:
        i = 10
        return i + 15
    if isinstance(strategy, CollectiveAllReduceStrategy):
        resolver = strategy.cluster_resolver
        return max(nest.flatten(resolver.num_accelerators())[0], 1)
    else:
        return strategy.num_replicas_in_sync

@combinations.generate(combinations.combine(strategy=[strategy_combinations.multi_worker_mirrored_2x1_cpu, strategy_combinations.multi_worker_mirrored_2x1_gpu], mode=['eager']))
class DistributedCollectiveAllReduceStrategyTest(strategy_test_lib.DistributionTestBase, parameterized.TestCase):

    def testDatasetFromFunction(self, strategy):
        if False:
            return 10

        def dataset_fn(input_context):
            if False:
                print('Hello World!')
            global_batch_size = 10
            batch_size = input_context.get_per_replica_batch_size(global_batch_size)
            d = dataset_ops.DatasetV2.range(100).repeat().batch(batch_size)
            return d.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
        expected_sum_on_workers = {'chief': 10, 'worker': 35}
        input_iterator = iter(strategy.distribute_datasets_from_function(dataset_fn))

        @def_function.function
        def run(iterator):
            if False:
                while True:
                    i = 10
            return strategy.experimental_local_results(iterator.get_next())
        result = run(input_iterator)
        sum_value = math_ops.reduce_sum(result)
        self.assertEqual(sum_value.numpy(), expected_sum_on_workers[multi_worker_test_base.get_task_type()])

    def testSimpleInputFromDatasetLastPartialBatch(self, strategy):
        if False:
            while True:
                i = 10
        global_batch_size = 8
        dataset = dataset_ops.DatasetV2.range(14).batch(global_batch_size, drop_remainder=False)
        input_iterator = iter(strategy.experimental_distribute_dataset(dataset))

        @def_function.function
        def run(input_iterator):
            if False:
                return 10
            return strategy.run(lambda x: x, args=(next(input_iterator),))
        run(input_iterator)
        result = run(input_iterator)
        expected_data_on_workers = {'chief': [8, 9, 10], 'worker': [11, 12, 13]}
        self.assertAllEqual(expected_data_on_workers[multi_worker_test_base.get_task_type()], result.numpy())

    def testSimpleInputFromFnLastPartialBatch(self, strategy):
        if False:
            while True:
                i = 10

        def dataset_fn(input_context):
            if False:
                i = 10
                return i + 15
            global_batch_size = 8
            batch_size = input_context.get_per_replica_batch_size(global_batch_size)
            dataset = dataset_ops.DatasetV2.range(14).batch(batch_size, drop_remainder=False)
            return dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
        input_iterator = iter(strategy.distribute_datasets_from_function(dataset_fn))

        @def_function.function
        def run(input_iterator):
            if False:
                i = 10
                return i + 15
            return strategy.run(lambda x: x, args=(next(input_iterator),))
        run(input_iterator)
        result = run(input_iterator)
        expected_data_on_worker = {'chief': [8, 9, 10, 11], 'worker': [12, 13]}
        self.assertAllEqual(expected_data_on_worker[multi_worker_test_base.get_task_type()], result.numpy())

    def testReduceHostTensor(self, strategy):
        if False:
            print('Hello World!')
        reduced = strategy.reduce(reduce_util.ReduceOp.SUM, array_ops.identity(1.0), axis=None)
        self.assertEqual(reduced.numpy(), 2.0)

    def testReduceToHostTensor(self, strategy):
        if False:
            i = 10
            return i + 15
        value = array_ops.identity(1.0)
        reduced = strategy.extended.reduce_to(reduce_util.ReduceOp.SUM, value, value)
        self.assertEqual(reduced.numpy(), 2.0)

    def testBatchReduceToHostTensor(self, strategy):
        if False:
            return 10
        value = array_ops.identity(1.0)
        reduced = strategy.extended.batch_reduce_to(reduce_util.ReduceOp.SUM, [(value, value), (value, value)])
        self.assertAllEqual([2.0, 2.0], reduced)

    def testReduceDeviceTensors(self, strategy):
        if False:
            print('Hello World!')
        value = strategy.run(lambda : array_ops.identity(1.0))
        reduced = strategy.reduce(reduce_util.ReduceOp.SUM, value, axis=None)
        self.assertEqual(reduced.numpy(), 2.0)

    def testReduceToDeviceTensors(self, strategy):
        if False:
            while True:
                i = 10
        value = strategy.run(lambda : array_ops.identity(1.0))
        reduced = strategy.extended.reduce_to(reduce_util.ReduceOp.SUM, value, value)
        self.assertEqual(reduced.numpy(), 2.0)

    def testBatchReduceToDeviceTensors(self, strategy):
        if False:
            for i in range(10):
                print('nop')
        value = strategy.run(lambda : array_ops.identity(1.0))
        reduced = strategy.extended.batch_reduce_to(reduce_util.ReduceOp.SUM, [(value, value), (value, value)])
        self.assertAllEqual([2.0, 2.0], reduced)

class StrategyClusterResolverTest(test.TestCase, parameterized.TestCase):

    @combinations.generate(combinations.combine(strategy=[strategy_combinations.multi_worker_mirrored_2x1_cpu] + strategy_combinations.all_strategies, mode=['eager']))
    def testClusterResolverProperty(self, strategy):
        if False:
            while True:
                i = 10
        resolver = strategy.cluster_resolver
        if not isinstance(strategy, CollectiveAllReduceStrategy) and (not strategy_test_lib.is_tpu_strategy(strategy)):
            self.assertIsNone(resolver)
            return
        with strategy.scope():
            self.assertIs(strategy.cluster_resolver, resolver)
        self.assertTrue(hasattr(resolver, 'cluster_spec'))
        self.assertTrue(hasattr(resolver, 'master'))
        self.assertTrue(hasattr(resolver, 'num_accelerators'))
        self.assertTrue(hasattr(resolver, 'task_id'))
        self.assertTrue(hasattr(resolver, 'task_type'))
        if isinstance(strategy, CollectiveAllReduceStrategy):
            self.assertEqual(resolver.task_id, 0)
            self.assertAllInSet(resolver.task_type, ['chief', 'worker'])
if __name__ == '__main__':
    test_util.main()