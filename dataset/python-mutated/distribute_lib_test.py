"""Test DistributionStrategy, ReplicaContext, and supporting APIs."""
from absl.testing import parameterized
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib
from tensorflow.python.util import nest

class _TestReplicaContext(distribute_lib.ReplicaContext):

    def merge_call(self, fn, *args, **kwargs):
        if False:
            while True:
                i = 10
        return kwargs['test_arg']

def _get_test_variable(name, synchronization, aggregation):
    if False:
        return 10
    return {'name': name, 'synchronization': synchronization, 'aggregation': aggregation}

def _test_input_fn(input_context):
    if False:
        i = 10
        return i + 15
    del input_context
    return dataset_ops.DatasetV2.from_tensors(1.0).repeat()

class _TestStrategy(distribute_lib.Strategy):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(_TestStrategy, self).__init__(_TestExtended(self))

class _TestExtended(distribute_lib.StrategyExtendedV1):

    def __init__(self, distribute):
        if False:
            return 10
        super(_TestExtended, self).__init__(distribute)
        worker_device_pairs = [('', ['/device:CPU:0'])]
        self._input_workers = input_lib.InputWorkers(worker_device_pairs)

    def _call_for_each_replica(self, fn, args, kwargs):
        if False:
            return 10
        with _TestReplicaContext(self._container_strategy(), replica_id_in_sync_group=0):
            return fn(*args, **kwargs)

    def _create_variable(self, next_creator, **kwargs):
        if False:
            print('Hello World!')
        return _get_test_variable(kwargs['name'], kwargs['synchronization'], kwargs['aggregation'])

    def _make_input_fn_iterator(self, input_fn, replication_mode=distribute_lib.InputReplicationMode.PER_WORKER):
        if False:
            for i in range(10):
                print('nop')
        return input_lib_v1.InputFunctionIterator(input_fn, self._input_workers, [distribute_lib.InputContext()], self._container_strategy())

    def _distribute_datasets_from_function(self, dataset_fn, options):
        if False:
            print('Hello World!')
        return dataset_fn(distribute_lib.InputContext())

    def _local_results(self, value):
        if False:
            i = 10
            return i + 15
        return (value,)

    def _reduce_to(self, reduce_op, value, destinations, options):
        if False:
            print('Hello World!')
        del reduce_op, destinations, options
        return value

    def _experimental_run_steps_on_iterator(self, fn, iterator, iterations, initial_loop_values=None):
        if False:
            return 10
        ctx = input_lib.MultiStepContext()
        for _ in range(iterations):
            fn(ctx, iterator.get_next())
        return ctx

    def _update(self, var, fn, args, kwargs, group):
        if False:
            i = 10
            return i + 15
        return self._update_non_slot(var, fn, (var,) + tuple(args), kwargs, group)

    def _update_non_slot(self, colocate_with, fn, args, kwargs, group):
        if False:
            print('Hello World!')
        del colocate_with
        result = fn(*args, **kwargs)
        if group:
            return result
        else:
            return nest.map_structure(self._unwrap, result)

    def _get_local_replica_id(self, replica_id_in_sync_group):
        if False:
            return 10
        return replica_id_in_sync_group

def _assert_in_default_state(t):
    if False:
        i = 10
        return i + 15
    t.assertIs(distribute_lib._get_default_replica_context(), distribute_lib.get_replica_context())
    t.assertIs(None, distribute_lib.get_cross_replica_context())
    t.assertFalse(distribute_lib.in_cross_replica_context())
    t.assertIs(distribute_lib._get_default_strategy(), distribute_lib.get_strategy())
    t.assertFalse(distribute_lib.has_strategy())

def _run_in_and_out_of_scope(unbound_test_method):
    if False:
        while True:
            i = 10

    def wrapper(test_case):
        if False:
            while True:
                i = 10
        dist = _TestStrategy()
        _assert_in_default_state(test_case)
        unbound_test_method(test_case, dist)
        with dist.scope():
            unbound_test_method(test_case, dist)
        _assert_in_default_state(test_case)
        another_strategy = _TestStrategy()
        msg = 'Mixing different .*Strategy objects'
        with test_case.assertRaisesRegex(RuntimeError, msg):
            with another_strategy.scope():
                unbound_test_method(test_case, dist)
    return wrapper

class TestStrategyTest(test.TestCase):

    def testCallForEachReplica(self):
        if False:
            while True:
                i = 10
        _assert_in_default_state(self)
        dist = _TestStrategy()

        def run_fn():
            if False:
                while True:
                    i = 10
            replica_context = distribute_lib.get_replica_context()
            self.assertIsNotNone(replica_context)
            self.assertIs(None, distribute_lib.get_cross_replica_context())
            self.assertFalse(distribute_lib.in_cross_replica_context())
            self.assertTrue(distribute_lib.has_strategy())
            self.assertIs(dist, distribute_lib.get_strategy())
            self.assertEqual('foo', replica_context.merge_call(None, test_arg='foo'))
            expected_value = _get_test_variable('bar', variable_scope.VariableSynchronization.AUTO, variable_scope.VariableAggregation.NONE)
            self.assertDictEqual(expected_value, variable_v1.VariableV1(1.0, name='bar'))
        dist.extended.call_for_each_replica(run_fn)
        with dist.scope():
            dist.extended.call_for_each_replica(run_fn)
        _assert_in_default_state(self)

    def testScope(self):
        if False:
            return 10
        _assert_in_default_state(self)
        dist = _TestStrategy()
        with dist.scope():
            self.assertIs(None, distribute_lib.get_replica_context())
            self.assertIs(dist, distribute_lib.get_cross_replica_context())
            self.assertTrue(distribute_lib.in_cross_replica_context())
            self.assertTrue(distribute_lib.has_strategy())
            self.assertIs(dist, distribute_lib.get_strategy())
            expected_value = _get_test_variable('baz', variable_scope.VariableSynchronization.AUTO, variable_scope.VariableAggregation.NONE)
            self.assertDictEqual(expected_value, variable_v1.VariableV1(1.0, name='baz'))
        _assert_in_default_state(self)

    def testScopeDeviceNestingError(self):
        if False:
            for i in range(10):
                print('nop')
        _assert_in_default_state(self)
        dist = _TestStrategy()
        dist.extended._default_device = '/device:GPU:0'
        scope = dist.scope()
        scope.__enter__()
        self.assertIs(dist, distribute_lib.get_strategy())
        with ops.device('/device:CPU:0'):
            with self.assertRaisesRegex(RuntimeError, 'Device scope nesting error'):
                scope.__exit__(None, None, None)
        scope.__exit__(None, None, None)
        _assert_in_default_state(self)

    def testScopeVarCreatorNestingError(self):
        if False:
            while True:
                i = 10

        def creator(next_creator, **kwargs):
            if False:
                i = 10
                return i + 15
            return next_creator(**kwargs)
        _assert_in_default_state(self)
        dist = _TestStrategy()
        scope = dist.scope()
        scope.__enter__()
        self.assertIs(dist, distribute_lib.get_strategy())
        with variable_scope.variable_creator_scope(creator):
            with self.assertRaisesRegex(RuntimeError, 'Variable creator scope nesting error'):
                scope.__exit__(None, None, None)
        scope.__exit__(None, None, None)
        _assert_in_default_state(self)

    def testScopeVarScopeNestingError(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            _assert_in_default_state(self)
            dist = _TestStrategy()
            scope = dist.scope()
            scope.__enter__()
            self.assertIs(dist, distribute_lib.get_strategy())
            with variable_scope.variable_scope('AA'):
                with self.assertRaisesRegex(RuntimeError, 'Variable scope nesting error'):
                    scope.__exit__(None, None, None)
        _assert_in_default_state(self)

    def testSettingSynchronizationAndAggregation(self):
        if False:
            print('Hello World!')
        _assert_in_default_state(self)
        dist = _TestStrategy()
        with dist.scope():
            expected_value = _get_test_variable('baz', variable_scope.VariableSynchronization.ON_WRITE, variable_scope.VariableAggregation.MEAN)
            self.assertDictEqual(expected_value, variable_v1.VariableV1(1.0, name='baz', synchronization=variable_scope.VariableSynchronization.ON_WRITE, aggregation=variable_scope.VariableAggregation.MEAN))
        _assert_in_default_state(self)

    def testSetStrategy(self):
        if False:
            print('Hello World!')
        _assert_in_default_state(self)
        dist = _TestStrategy()
        dist2 = _TestStrategy()
        distribute_lib.experimental_set_strategy(dist)
        self.assertIs(None, distribute_lib.get_replica_context())
        self.assertIs(dist, distribute_lib.get_cross_replica_context())
        self.assertTrue(distribute_lib.in_cross_replica_context())
        self.assertTrue(distribute_lib.has_strategy())
        self.assertIs(dist, distribute_lib.get_strategy())
        expected_value = _get_test_variable('baz', variable_scope.VariableSynchronization.AUTO, variable_scope.VariableAggregation.NONE)
        self.assertDictEqual(expected_value, variable_v1.VariableV1(1.0, name='baz'))
        distribute_lib.experimental_set_strategy(dist2)
        self.assertIs(dist2, distribute_lib.get_strategy())
        distribute_lib.experimental_set_strategy(None)
        _assert_in_default_state(self)

    def testSetStrategyInScope(self):
        if False:
            while True:
                i = 10
        _assert_in_default_state(self)
        dist = _TestStrategy()
        with dist.scope():
            with self.assertRaisesRegex(RuntimeError, 'Must not be called inside a `tf.distribute.Strategy` scope'):
                distribute_lib.experimental_set_strategy(_TestStrategy())
            with self.assertRaisesRegex(RuntimeError, 'Must not be called inside a `tf.distribute.Strategy` scope'):
                distribute_lib.experimental_set_strategy(dist)
            with self.assertRaisesRegex(RuntimeError, 'Must not be called inside a `tf.distribute.Strategy` scope'):
                distribute_lib.experimental_set_strategy(None)
        _assert_in_default_state(self)

    def testSameScopeNesting(self):
        if False:
            return 10
        _assert_in_default_state(self)
        dist = _TestStrategy()
        scope_a = dist.scope()
        with scope_a:
            self.assertIs(dist, distribute_lib.get_strategy())
            scope_b = dist.scope()
            with scope_b:
                self.assertIs(dist, distribute_lib.get_strategy())
                with scope_a:
                    self.assertIs(dist, distribute_lib.get_strategy())
                self.assertIs(dist, distribute_lib.get_strategy())
            self.assertIs(dist, distribute_lib.get_strategy())
            dist2 = _TestStrategy()
            scope2 = dist2.scope()
            with self.assertRaisesRegex(RuntimeError, 'Mixing different tf.distribute.Strategy objects'):
                with scope2:
                    pass
        _assert_in_default_state(self)
        with scope_b:
            self.assertIs(dist, distribute_lib.get_strategy())
        _assert_in_default_state(self)

    @_run_in_and_out_of_scope
    def testMakeInputFnIterator(self, dist):
        if False:
            print('Hello World!')
        self.assertIsNotNone(dist.make_input_fn_iterator(_test_input_fn))

    @_run_in_and_out_of_scope
    def testReduce(self, dist):
        if False:
            for i in range(10):
                print('nop')
        x = constant_op.constant(1.0)
        x_r = dist.reduce(reduce_util.ReduceOp.MEAN, x, axis=None)
        self.assertEqual(self.evaluate(x), self.evaluate(x_r))

    def testReductions_acceptStringOps(self):
        if False:
            i = 10
            return i + 15
        dist = _TestStrategy()
        for op in ('mean', 'MEAN', 'sum', 'SUM'):
            x = constant_op.constant(1.0)
            y = constant_op.constant(1.0)
            x_r = dist.reduce(op, x, axis=None)
            self.assertEqual(self.evaluate(x), self.evaluate(x_r))
            x_r = dist.extended.reduce_to(op, x, '/CPU:0')
            self.assertEqual(self.evaluate(x), self.evaluate(x_r))
            (x_r, y_r) = dist.extended.batch_reduce_to(op, ((x, '/CPU:0'), (y, '/CPU:0')))
            self.assertEqual(self.evaluate(x), self.evaluate(x_r))
            self.assertEqual(self.evaluate(y), self.evaluate(y_r))

    @_run_in_and_out_of_scope
    def testReduceMeanAxis(self, dist):
        if False:
            return 10
        x = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
        x_r = dist.reduce(reduce_util.ReduceOp.MEAN, x, axis=None)
        self.assertAllEqual(self.evaluate(x), self.evaluate(x_r))
        x_r = dist.reduce(reduce_util.ReduceOp.MEAN, x, axis=0)
        self.assertAllEqual([2.0, 3.0], self.evaluate(x_r))
        x_r = dist.reduce(reduce_util.ReduceOp.MEAN, x, axis=(0, 1))
        self.assertEqual(2.5, self.evaluate(x_r))

    @_run_in_and_out_of_scope
    def testReduceSumAxis(self, dist):
        if False:
            while True:
                i = 10
        x = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
        x_r = dist.reduce(reduce_util.ReduceOp.SUM, x, axis=None)
        self.assertAllEqual(self.evaluate(x), self.evaluate(x_r))
        x_r = dist.reduce(reduce_util.ReduceOp.SUM, x, axis=0)
        self.assertAllEqual([4.0, 6.0], self.evaluate(x_r))
        x_r = dist.reduce(reduce_util.ReduceOp.SUM, x, axis=(0, 1))
        self.assertEqual(10.0, self.evaluate(x_r))

    @_run_in_and_out_of_scope
    def testExperimentalRunStepsOnIterator(self, dist):
        if False:
            i = 10
            return i + 15
        all_inputs = []
        dataset = dataset_ops.Dataset.from_tensors(1.0).repeat()
        dist.extended.experimental_run_steps_on_iterator(lambda _, inputs: all_inputs.append(self.evaluate(inputs)), dataset_ops.make_one_shot_iterator(dataset))
        self.assertEqual(all_inputs, [1.0])

    @_run_in_and_out_of_scope
    def testReduceTo(self, dist):
        if False:
            while True:
                i = 10
        x = constant_op.constant(1.0)
        x_r = dist.extended.reduce_to(reduce_util.ReduceOp.MEAN, x, '/CPU:0')
        self.assertEqual(self.evaluate(x), self.evaluate(x_r))

    @_run_in_and_out_of_scope
    def testBatchReduceTo(self, dist):
        if False:
            return 10
        x = constant_op.constant(1.0)
        y = constant_op.constant(1.0)
        (x_r, y_r) = dist.extended.batch_reduce_to(reduce_util.ReduceOp.MEAN, ((x, '/CPU:0'), (y, '/CPU:0')))
        self.assertEqual(self.evaluate(x), self.evaluate(x_r))
        self.assertEqual(self.evaluate(y), self.evaluate(y_r))

    @_run_in_and_out_of_scope
    def testUpdate(self, dist):
        if False:
            i = 10
            return i + 15
        with dist.scope():
            v = variables.Variable(1.0)
        t = constant_op.constant(2.0)

        def assign_fn(vv, tt):
            if False:
                while True:
                    i = 10
            self.assertIs(vv, v)
            self.assertIs(tt, t)
        dist.extended.update(v, assign_fn, (t,))

    @_run_in_and_out_of_scope
    def testUpdateAutoGraph(self, dist):
        if False:
            while True:
                i = 10
        with dist.scope():
            v = variables.Variable(1.0)
        t = constant_op.constant(2.0)

        def assign_fn(unused_vv, unused_tt):
            if False:
                for i in range(10):
                    print('nop')
            self.assertTrue(converter_testing.is_inside_generated_code())

        @def_function.function
        def test_fn():
            if False:
                print('Hello World!')
            dist.extended.update(v, assign_fn, (t,))
        test_fn()

    @_run_in_and_out_of_scope
    def testUpdateNonSlot(self, dist):
        if False:
            for i in range(10):
                print('nop')
        t = constant_op.constant(2.0)
        update_calls = []
        dist.extended.update_non_slot(t, lambda : update_calls.append(1))
        self.assertEqual(len(update_calls), 1)

    @_run_in_and_out_of_scope
    def testUpdateNonSlotAutoGraph(self, dist):
        if False:
            for i in range(10):
                print('nop')
        t = constant_op.constant(2.0)

        def update_fn():
            if False:
                print('Hello World!')
            self.assertTrue(converter_testing.is_inside_generated_code())

        @def_function.function
        def test_fn():
            if False:
                print('Hello World!')
            dist.extended.update_non_slot(t, update_fn)
        test_fn()

    def testClusterResolverDefaultNotImplemented(self):
        if False:
            print('Hello World!')
        dist = _TestStrategy()
        self.assertIsNone(dist.cluster_resolver)
        base_cluster_spec = server_lib.ClusterSpec({'ps': ['ps0:2222', 'ps1:2222'], 'worker': ['worker0:2222', 'worker1:2222', 'worker2:2222']})
        cluster_resolver = cluster_resolver_lib.SimpleClusterResolver(base_cluster_spec)
        dist.extended._cluster_resolver = cluster_resolver
        self.assertIs(dist.cluster_resolver, cluster_resolver)

class _TestStrategy2(distribute_lib.Strategy):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(_TestStrategy2, self).__init__(_TestExtended2(self))

class _TestExtended2(_TestExtended):

    def _create_variable(self, next_creator, **kwargs):
        if False:
            while True:
                i = 10
        return next_creator(**kwargs)

class DefaultDistributionStrategyTest(test.TestCase, parameterized.TestCase):

    def testMergeCall(self):
        if False:
            return 10
        _assert_in_default_state(self)

        def merge_fn(dist, s):
            if False:
                while True:
                    i = 10
            self.assertIs(distribute_lib._get_default_strategy(), dist)
            self.assertIs(None, distribute_lib.get_replica_context())
            self.assertIs(dist, distribute_lib.get_cross_replica_context())
            self.assertTrue(distribute_lib.in_cross_replica_context())
            self.assertIs(dist, distribute_lib.get_strategy())
            self.assertFalse(distribute_lib.has_strategy())
            return 'foo_' + s
        replica_ctx = distribute_lib.get_replica_context()
        self.assertIs(distribute_lib._get_default_replica_context(), replica_ctx)
        self.assertEqual('foo_bar', replica_ctx.merge_call(merge_fn, args=('bar',)))
        _assert_in_default_state(self)

    def testMergeCallAutoGraph(self):
        if False:
            print('Hello World!')
        _assert_in_default_state(self)

        def merge_fn(_, s):
            if False:
                while True:
                    i = 10
            self.assertTrue(converter_testing.is_inside_generated_code())
            return s

        @def_function.function
        def test_fn():
            if False:
                i = 10
                return i + 15
            replica_ctx = distribute_lib.get_replica_context()
            replica_ctx.merge_call(merge_fn, args=('bar',))
        test_fn()

    def testScopeMostlyNoOp(self):
        if False:
            return 10
        _assert_in_default_state(self)
        test_strategy = _TestStrategy2()
        with test_strategy.scope():
            variable_v1.VariableV1(1.0, name='before')
        default_strategy = distribute_lib._get_default_strategy()
        scope = default_strategy.scope()
        with scope:
            _assert_in_default_state(self)
            with test_strategy.scope():
                with self.assertRaisesRegex(RuntimeError, 'Mixing different tf.distribute.Strategy objects'):
                    variable_v1.VariableV1(1.0, name='error')
            with scope:
                _assert_in_default_state(self)
                with test_strategy.scope():
                    with self.assertRaisesRegex(RuntimeError, 'Mixing different tf.distribute.Strategy objects'):
                        variable_v1.VariableV1(1.0, name='also_error')
            _assert_in_default_state(self)
        _assert_in_default_state(self)
        with test_strategy.scope():
            variable_v1.VariableV1(1.0, name='after')

    def testExperimentalRunV2(self):
        if False:
            i = 10
            return i + 15
        default_strategy = distribute_lib._get_default_strategy()
        dataset = dataset_ops.Dataset.range(10).batch(2)
        iterator = default_strategy.extended._make_dataset_iterator(dataset)
        next_val = iterator.get_next()

        def train_step(input_data):
            if False:
                return 10
            return input_data
        for _ in range(2):
            default_strategy.run(train_step, args=(next_val,))

    @combinations.generate(combinations.combine(mode=['graph', 'eager']))
    def testDistributedDatasets(self):
        if False:
            for i in range(10):
                print('nop')
        default_strategy = distribute_lib._get_default_strategy()
        if context.executing_eagerly():
            dataset_fn = lambda _: dataset_ops.DatasetV2.range(10).batch(2)
            dist_dataset = default_strategy.experimental_distribute_dataset(dataset_fn(distribute_lib.InputContext()))
            next_val = next(iter(dist_dataset))
        else:
            dataset_fn = lambda _: dataset_ops.DatasetV1.range(10).batch(2)
            dist_dataset = default_strategy.experimental_distribute_dataset(dataset_fn(distribute_lib.InputContext()))
            iterator = dist_dataset.make_initializable_iterator()
            self.evaluate(iterator.initializer)
            next_val = iterator.get_next()
        self.assertAllEqual([0, 1], self.evaluate(next_val))

    @combinations.generate(combinations.combine(mode=['graph', 'eager']))
    def testDistributedDatasetsFromFunction(self):
        if False:
            i = 10
            return i + 15
        default_strategy = distribute_lib._get_default_strategy()
        if context.executing_eagerly():
            dataset_fn = lambda _: dataset_ops.DatasetV2.range(10).batch(2)
            dist_dataset_from_func = default_strategy.distribute_datasets_from_function(dataset_fn)
            next_val = next(iter(dist_dataset_from_func))
            self.assertAllEqual([0, 1], self.evaluate(next_val))
        else:
            dataset_fn = lambda _: dataset_ops.DatasetV2.range(10).batch(2)
            dist_dataset_from_func = default_strategy.distribute_datasets_from_function(dataset_fn)
            dataset_ops.make_initializable_iterator(dist_dataset_from_func)

    @combinations.generate(combinations.combine(tf_api_version=1))
    def testV1(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIsInstance(distribute_lib.get_strategy(), distribute_lib.StrategyV1)

    @combinations.generate(combinations.combine(tf_api_version=2))
    def testV2(self):
        if False:
            return 10
        self.assertIsInstance(distribute_lib.get_strategy(), distribute_lib.Strategy)

class InputContextTest(test.TestCase):

    def testProperties(self):
        if False:
            while True:
                i = 10
        input_context = distribute_lib.InputContext(num_input_pipelines=2, input_pipeline_id=1, num_replicas_in_sync=6)
        self.assertEqual(6, input_context.num_replicas_in_sync)
        self.assertEqual(1, input_context.input_pipeline_id)
        self.assertEqual(2, input_context.num_input_pipelines)

    def testPerReplicaBatchSize(self):
        if False:
            return 10
        input_context = distribute_lib.InputContext(num_input_pipelines=2, input_pipeline_id=1, num_replicas_in_sync=6)
        self.assertEqual(2, input_context.get_per_replica_batch_size(12))
        with self.assertRaises(ValueError):
            input_context.get_per_replica_batch_size(13)

    def testStr(self):
        if False:
            i = 10
            return i + 15
        input_context = distribute_lib.InputContext(num_input_pipelines=1, input_pipeline_id=0, num_replicas_in_sync=42)
        self.assertEqual('tf.distribute.InputContext(input pipeline id 0, total: 1)', str(input_context))
        input_context = distribute_lib.InputContext(num_input_pipelines=3, input_pipeline_id=1, num_replicas_in_sync=42)
        self.assertEqual('tf.distribute.InputContext(input pipeline id 1, total: 3)', str(input_context))
if __name__ == '__main__':
    test.main()