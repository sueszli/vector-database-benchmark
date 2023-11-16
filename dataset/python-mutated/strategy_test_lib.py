"""Library for testing DistributionStrategy descendants."""
import functools
import os
import tempfile
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_all_reduce_strategy as mwms_lib
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import mirrored_strategy as mirrored_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2 as summary_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_util
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect

class _TestException(Exception):
    pass

def _maybe_run_in_function(fn, run_in_function=False):
    if False:
        i = 10
        return i + 15
    if not run_in_function or not context.executing_eagerly():
        return fn
    else:
        return def_function.function()(fn)

def _raise_exception_fn(_=None):
    if False:
        return 10
    raise _TestException()

def _merge_raises_fn():
    if False:
        i = 10
        return i + 15
    distribute_lib.get_replica_context().merge_call(_raise_exception_fn)

def _call_raises_fn(dist):
    if False:
        while True:
            i = 10
    dist.extended.call_for_each_replica(_raise_exception_fn)

def _merge_call_raises_fn():
    if False:
        while True:
            i = 10
    distribute_lib.get_replica_context().merge_call(_call_raises_fn)

def _call_merge_raises_fn(dist):
    if False:
        print('Hello World!')
    dist.extended.call_for_each_replica(_merge_raises_fn)

def _merge_call_merge_raises_fn():
    if False:
        return 10
    distribute_lib.get_replica_context().merge_call(_call_merge_raises_fn)

def _events_from_logdir(test_case, logdir):
    if False:
        print('Hello World!')
    'Reads summary events from log directory.'
    test_case.assertTrue(gfile.Exists(logdir))
    files = gfile.ListDirectory(logdir)
    test_case.assertLen(files, 1)
    records = list(tf_record.tf_record_iterator(os.path.join(logdir, files[0])))
    result = []
    for r in records:
        event = event_pb2.Event()
        event.ParseFromString(r)
        result.append(event)
    return result

def create_variable_like_keras_layer(name, shape, dtype):
    if False:
        print('Hello World!')
    'Utitlity for create variables that works like variable in keras layer.'
    initializer = functools.partial(init_ops_v2.GlorotUniform(), shape, dtype=dtype)
    return variables.Variable(initial_value=initializer, name=name, trainable=True)

def is_optimizer_v2_instance(optimizer_obj):
    if False:
        return 10
    arg_spec = tf_inspect.getfullargspec(optimizer_obj.minimize)
    return 'var_list' in arg_spec.args[:-len(arg_spec.defaults)]

def is_mirrored_strategy(strategy: distribute_lib.Strategy) -> bool:
    if False:
        return 10
    return isinstance(strategy, (mirrored_lib.MirroredStrategy, mirrored_lib.MirroredStrategyV1))

def is_multi_worker_mirrored_strategy(strategy: distribute_lib.Strategy) -> bool:
    if False:
        i = 10
        return i + 15
    return isinstance(strategy, (mwms_lib.CollectiveAllReduceStrategy, mwms_lib.CollectiveAllReduceStrategyV1))

def is_tpu_strategy(strategy: distribute_lib.Strategy) -> bool:
    if False:
        return 10
    return isinstance(strategy, (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV1, tpu_strategy.TPUStrategyV2))

class DistributionTestBase(test.TestCase):
    """Some tests that should work with any DistributionStrategy."""

    def _test_minimize_loss_eager(self, d):
        if False:
            return 10
        with d.scope():
            kernel = create_variable_like_keras_layer(name='kernel', shape=(1, 1), dtype=dtypes.float32)

            def loss(x):
                if False:
                    i = 10
                    return i + 15
                y = array_ops.reshape(math_ops.mat_mul(x, kernel), []) - array_ops.identity(1.0)
                return y * y
            grad_fn = backprop.implicit_grad(loss)
            grad_fn = optimizer.get_filtered_grad_fn(grad_fn)

            def update(v, g):
                if False:
                    for i in range(10):
                        print('nop')
                return v.assign_sub(0.2 * g)
            one = array_ops.identity([[1.0]])

            def step():
                if False:
                    i = 10
                    return i + 15
                'Perform one optimization step.'
                g_v = d.extended.call_for_each_replica(grad_fn, args=(one,))
                before_list = []
                after_list = []
                for (g, v) in g_v:
                    fetched = d.extended.read_var(v)
                    before_list.append(fetched)
                    with ops.control_dependencies([fetched]):
                        g = d.extended.reduce_to(reduce_util.ReduceOp.SUM, g, destinations=v)
                        with ops.control_dependencies(d.extended.update(v, update, args=(g,), group=False)):
                            after_list.append(d.extended.read_var(v))
                return (before_list, after_list)
            for i in range(10):
                (b, a) = step()
                if i == 0:
                    (before,) = b
                (after,) = a
            error_before = abs(before.numpy() - 1)
            error_after = abs(after.numpy() - 1)
            self.assertLess(error_after, error_before)

    def _test_minimize_loss_graph(self, d, soft_placement=False, learning_rate=0.2):
        if False:
            while True:
                i = 10
        config = config_pb2.ConfigProto()
        config.allow_soft_placement = soft_placement
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        with context.graph_mode(), ops.Graph().as_default(), self.cached_session(config=config) as sess, d.scope():
            kernel = create_variable_like_keras_layer(name='kernel', shape=(1, 1), dtype=dtypes.float32)

            def loss(x):
                if False:
                    print('Hello World!')
                y = array_ops.reshape(math_ops.mat_mul(x, kernel), []) - array_ops.identity(1.0)
                return y * y
            grad_fn = backprop.implicit_grad(loss)

            def update(v, g):
                if False:
                    while True:
                        i = 10
                return v.assign_sub(learning_rate * g)
            one = array_ops.identity([[1.0]])

            def step():
                if False:
                    return 10
                'Perform one optimization step.'
                g_v = d.extended.call_for_each_replica(grad_fn, args=(one,))
                before_list = []
                after_list = []
                for (g, v) in g_v:
                    fetched = d.extended.read_var(v)
                    before_list.append(fetched)
                    with ops.control_dependencies([fetched]):
                        g = d.extended.reduce_to(reduce_util.ReduceOp.SUM, g, destinations=v)
                        with ops.control_dependencies(d.extended.update(v, update, args=(g,), group=False)):
                            after_list.append(d.extended.read_var(v))
                return (before_list, after_list)
            (before_out, after_out) = step()
            variables.global_variables_initializer().run()
            for i in range(10):
                (b, a) = sess.run((before_out, after_out))
                if i == 0:
                    (before,) = b
                (after,) = a
            error_before = abs(before - 1)
            error_after = abs(after - 1)
            self.assertLess(error_after, error_before)

    def _test_summary_for_replica_zero_only(self, d):
        if False:
            while True:
                i = 10
        logdir = tempfile.mkdtemp()

        def run_fn():
            if False:
                i = 10
                return i + 15
            'Function executed for each replica.'
            with summary_writer.as_default():
                replica_id = distribute_lib.get_replica_context().replica_id_in_sync_group
                return summary_ops.write('a', replica_id)
        with self.cached_session() as sess, d.scope(), summary_ops.always_record_summaries():
            global_step = training_util.get_or_create_global_step()
            if not context.executing_eagerly():
                global_step.initializer.run()
            summary_ops.set_step(0)
            summary_writer = summary_ops.create_file_writer(logdir)
            output = d.extended.call_for_each_replica(run_fn)
            unwrapped = d.unwrap(output)
            if not context.executing_eagerly():
                sess.run(summary_writer.init())
                sess.run(unwrapped)
                sess.run(summary_writer.close())
            events = _events_from_logdir(self, logdir)
            self.assertLen(events, 2)
            self.assertEqual(events[1].summary.value[0].tag, 'a')
            self.assertEqual(events[1].summary.value[0].simple_value, 0.0)

    def _test_replica_id(self, d):
        if False:
            return 10
        with d.scope():
            expected_devices = [False] * len(d.extended.worker_devices)

            def mark_devices_fn():
                if False:
                    print('Hello World!')
                replica_id = self.evaluate(distribute_lib.get_replica_context().replica_id_in_sync_group)
                self.assertLess(replica_id, len(d.extended.worker_devices))
                self.assertFalse(expected_devices[replica_id])
                expected_devices[replica_id] = True
            d.extended.call_for_each_replica(mark_devices_fn)
            self.assertAllEqual(expected_devices, [True] * len(d.extended.worker_devices))

    def _test_call_and_merge_exceptions(self, dist):
        if False:
            print('Hello World!')
        with dist.scope():
            with self.assertRaises(_TestException):
                dist.extended.call_for_each_replica(_raise_exception_fn)
            with self.assertRaises(_TestException):
                dist.extended.call_for_each_replica(_merge_raises_fn)
            with self.assertRaises(_TestException):
                dist.extended.call_for_each_replica(_merge_call_raises_fn)
            with self.assertRaises(_TestException):
                dist.extended.call_for_each_replica(_merge_call_merge_raises_fn)

    def _input_fn_to_test_input_context(self, dataset_or_callable_fn, expected_num_replicas_in_sync, expected_num_input_pipelines, expected_input_pipeline_id):
        if False:
            while True:
                i = 10
        worker_id_counter = [0]

        def _input_fn(input_context):
            if False:
                print('Hello World!')
            'Input fn for testing.'
            self.assertIsNotNone(input_context)
            self.assertEqual(expected_num_replicas_in_sync, input_context.num_replicas_in_sync)
            self.assertEqual(expected_num_input_pipelines, input_context.num_input_pipelines)
            if expected_input_pipeline_id is not None:
                self.assertEqual(expected_input_pipeline_id, input_context.input_pipeline_id)
            else:
                self.assertEqual(worker_id_counter[0], input_context.input_pipeline_id)
                worker_id_counter[0] += 1
            return dataset_or_callable_fn()
        return _input_fn

    def _test_input_fn_iterable(self, strategy, input_fn, expected_values, ignore_order=False):
        if False:
            return 10
        assert_same = self.assertCountEqual if ignore_order else self.assertEqual
        iterable = strategy.distribute_datasets_from_function(input_fn)
        if context.executing_eagerly():
            iterator = iter(iterable)
            for expected_value in expected_values:
                computed_value = self.evaluate(list(strategy.experimental_local_results(next(iterator))))
                assert_same(expected_value, computed_value)
            with self.assertRaises(StopIteration):
                self.evaluate(strategy.experimental_local_results(next(iterator)))
            iterator = iter(iterable)
            for expected_value in expected_values:
                computed_value = self.evaluate(list(strategy.experimental_local_results(next(iterator))))
                assert_same(expected_value, computed_value)
        else:
            iterator = dataset_ops.make_initializable_iterator(iterable)
            self._test_input_fn_iterator(iterator, strategy.extended.worker_devices, expected_values, test_reinitialize=True, ignore_order=ignore_order)

    def _test_input_fn_iterator(self, iterator, devices, expected_values, sess=None, test_reinitialize=True, ignore_order=False):
        if False:
            for i in range(10):
                print('nop')
        evaluate = lambda x: sess.run(x) if sess else self.evaluate(x)
        evaluate(iterator.initializer)
        for expected_value in expected_values:
            next_element = iterator.get_next()
            computed_value = evaluate([distribute_utils.select_replica(r, next_element) for r in range(len(devices))])
            if ignore_order:
                self.assertCountEqual(expected_value, computed_value)
            else:
                self.assertEqual(expected_value, computed_value)
        with self.assertRaises(errors.OutOfRangeError):
            next_element = iterator.get_next()
            evaluate([distribute_utils.select_replica(r, next_element) for r in range(len(devices))])
        if test_reinitialize:
            evaluate(iterator.initializer)
            for expected_value in expected_values:
                next_element = iterator.get_next()
                computed_value = evaluate([distribute_utils.select_replica(r, next_element) for r in range(len(devices))])
                if ignore_order:
                    self.assertCountEqual(expected_value, computed_value)
                else:
                    self.assertEqual(expected_value, computed_value)

    def _test_global_step_update(self, strategy):
        if False:
            for i in range(10):
                print('nop')
        with strategy.scope():
            global_step = variable_scope.get_variable('global_step', shape=[], dtype=dtypes.int64, initializer=init_ops.zeros_initializer(), trainable=False, aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA)
            self.evaluate(variables.global_variables_initializer())

            def model_fn():
                if False:
                    while True:
                        i = 10
                train_op = global_step.assign_add(1)
                value = global_step.read_value()
                return (train_op, value)
            (train_ops, value) = strategy.extended.call_for_each_replica(model_fn)
            self.evaluate(strategy.group(train_ops))
            global_step_tensors = strategy.experimental_local_results(value)
            global_step_values = self.evaluate(global_step_tensors)
            self.assertEqual((1,) * len(global_step_tensors), global_step_values)

    def _test_numpy_dataset(self, strategy, session=None, run_in_function=False):
        if False:
            while True:
                i = 10
        if not isinstance(strategy, distribute_lib.StrategyV1):
            self.skipTest('n/a: V1 only')
        cached_session = session or self.cached_session()
        with strategy.scope(), cached_session as sess:
            x = np.asarray([[1, 2], [6, 12], [2, 4], [5, 10], [3, 6], [4, 8]])
            y = np.asarray([5, 4, 3, 2, 1, 0])
            batch_size = 6
            if not strategy.extended._global_batch_size:
                batch_size = batch_size // strategy.num_replicas_in_sync
            ds = strategy.extended.experimental_make_numpy_dataset((x, y), session=sess or self.cached_session())
            ds = ds.repeat(2)
            drop_remainder = strategy.extended.experimental_require_static_shapes
            ds = ds.batch(batch_size, drop_remainder=drop_remainder)
            i = strategy.make_dataset_iterator(ds)
            self.evaluate(i.initializer)

            def run_and_concatenate(strategy, i):
                if False:
                    while True:
                        i = 10
                (x, y) = strategy.experimental_run(_maybe_run_in_function(lambda z: z, run_in_function), i)
                (x, y) = self.evaluate((strategy.experimental_local_results(x), strategy.experimental_local_results(y)))
                return (np.concatenate(x), np.concatenate(y))
            (x_1, y_1) = run_and_concatenate(strategy, i)
            self.assertAllEqual(x, x_1)
            self.assertAllEqual(y, y_1)
            (x_2, y_2) = run_and_concatenate(strategy, i)
            self.assertAllEqual(x, x_2)
            self.assertAllEqual(y, y_2)
            with self.assertRaises(errors.OutOfRangeError):
                run_and_concatenate(strategy, i)

    def _test_trainable_variable(self, strategy):
        if False:
            i = 10
            return i + 15
        for cls in [variable_v1.VariableV1, variables.Variable]:
            with strategy.scope():
                v1 = cls(1.0)
                self.assertEqual(True, v1.trainable)
                v2 = cls(1.0, synchronization=variables.VariableSynchronization.ON_READ)
                self.assertEqual(False, v2.trainable)
                v3 = cls(1.0, synchronization=variables.VariableSynchronization.ON_READ, trainable=True)
                self.assertEqual(True, v3.trainable)
                v4 = cls(1.0, synchronization=variables.VariableSynchronization.ON_READ, trainable=False)
                self.assertEqual(False, v4.trainable)

class OneDeviceDistributionTestBase(test.TestCase):
    """Some tests that should work with any one-device DistributionStrategy."""

    def _test_run(self, strategy):
        if False:
            for i in range(10):
                print('nop')
        out1 = strategy.run(lambda : array_ops.identity(4.0))
        self.assertAllEqual([4.0], self.evaluate(strategy.unwrap(out1)))
        out2 = strategy.run(lambda x: {'a': x * 2, 'b': x * x}, args=(out1,))
        out2_vals = self.evaluate(nest.map_structure(strategy.unwrap, out2))
        self.assertAllEqual([8.0], out2_vals['a'])
        self.assertAllEqual([16.0], out2_vals['b'])
        out3 = strategy.run(lambda b, a: a + 2 * b + 2, kwargs=out2)
        self.assertAllEqual([42.0], self.evaluate(strategy.unwrap(out3)))

    def _test_all_reduce_sum(self, strategy):
        if False:
            i = 10
            return i + 15
        self._test_collective_comms(strategy, _all_sum, inputs=(4.0, [42.0, 43.0]), expected=(4.0, [42.0, 43.0]))

    def _test_all_reduce_sum_gradients(self, strategy):
        if False:
            i = 10
            return i + 15
        self._test_collective_comms_gradients(strategy, _all_sum, inputs=[4.0], expected_grads=[4.0])

    def _test_all_reduce_sum_gradient_tape(self, strategy):
        if False:
            for i in range(10):
                print('nop')
        self._test_collective_comms_gradient_tape(strategy, _all_sum, inputs=[4.0], expected_grads=[4.0])

    def _test_all_reduce_mean(self, strategy):
        if False:
            while True:
                i = 10
        self._test_collective_comms(strategy, _all_mean, inputs=(2.0, [21.0, 22.0]), expected=(2.0, [21.0, 22.0]))

    def _test_all_reduce_mean_gradients(self, strategy):
        if False:
            print('Hello World!')
        self._test_collective_comms_gradients(strategy, _all_mean, inputs=[5.0], expected_grads=[5.0])

    def _test_all_reduce_mean_gradient_tape(self, strategy):
        if False:
            while True:
                i = 10
        self._test_collective_comms_gradient_tape(strategy, _all_mean, inputs=[5.0], expected_grads=[5.0])

    def _test_collective_comms(self, strategy, comm_fn, inputs, expected):
        if False:
            print('Hello World!')
        inputs = strategy.make_input_fn_iterator(lambda _: dataset_ops.Dataset.from_tensors(inputs))
        self.evaluate(inputs.initialize())
        outputs = self.evaluate(list(map(strategy.experimental_local_results, strategy.experimental_run(comm_fn, inputs))))
        self.assertAllEqual([expected[0]], outputs[0])
        self.assertAllEqual([expected[1]], outputs[1])

    def _test_collective_comms_gradients(self, strategy, comm_fn, inputs, expected_grads):
        if False:
            for i in range(10):
                print('nop')
        if context.executing_eagerly():
            self.skipTest('`tf.gradients` is not supported with eager execution.')

        def step(c):
            if False:
                return 10
            x = array_ops.identity(42.0)
            y = comm_fn(x) * c
            return gradients_impl.gradients(y, [x])[0]
        inputs = strategy.make_input_fn_iterator(lambda _: dataset_ops.Dataset.from_tensors(inputs))
        self.evaluate(inputs.initialize())
        self.assertAllEqual(expected_grads, self.evaluate(strategy.experimental_local_results(strategy.experimental_run(step, inputs))))

    def _test_collective_comms_gradient_tape(self, strategy, comm_fn, inputs, expected_grads):
        if False:
            return 10

        def step(c):
            if False:
                print('Hello World!')
            x = array_ops.identity(42.0)
            with backprop.GradientTape() as tape:
                tape.watch(x)
                y = comm_fn(x) * c
            return tape.gradient(y, x)
        inputs = strategy.make_input_fn_iterator(lambda _: dataset_ops.Dataset.from_tensors(inputs))
        self.evaluate(inputs.initialize())
        self.assertAllEqual(expected_grads, self.evaluate(strategy.experimental_local_results(strategy.experimental_run(step, inputs))))

    def _test_device_and_input_device_are_colocated(self, strategy):
        if False:
            i = 10
            return i + 15
        if context.executing_eagerly():
            self.skipTest('cross-device tests are not supported with eager execution.')
        (workers, _) = test_util.create_local_cluster(2, 0)
        inputs = strategy.make_input_fn_iterator(lambda _: dataset_ops.Dataset.range(5))
        comm_fn = lambda x: x + 1
        run_op = strategy.experimental_run(comm_fn, inputs)
        with session_lib.Session(target=workers[1].target) as sess:
            sess.run(inputs.initialize())
            sess.run(run_op)

    def _test_device_and_input_device_are_colocated_with_function(self, strategy):
        if False:
            for i in range(10):
                print('nop')
        if context.executing_eagerly():
            self.skipTest('cross-device tests are not supported with eager execution.')
        (workers, _) = test_util.create_local_cluster(2, 0)
        inputs = strategy.make_input_fn_iterator(lambda _: dataset_ops.Dataset.range(5))
        comm_fn = lambda x: x + 1
        experimental_run = def_function.function()(strategy.experimental_run)
        with ops.device('/job:worker/replica:0/task:1/device:CPU:0'):
            run_op = experimental_run(comm_fn, inputs)
        with session_lib.Session(target=workers[1].target) as sess:
            sess.run(inputs.initialize())
            sess.run(run_op)

class TwoDeviceDistributionTestBase(test.TestCase):
    """Some tests that should work with any two-device DistributionStrategy."""

    def _test_run(self, strategy, run_in_function=False):
        if False:
            return 10
        out1 = strategy.run(_maybe_run_in_function(lambda : distribute_lib.get_replica_context().replica_id_in_sync_group + 1, run_in_function))
        self.assertAllEqual([1, 2], self.evaluate(strategy.unwrap(out1)))
        out2 = strategy.run(_maybe_run_in_function(lambda x: {'a': x * 2, 'b': x * x}, run_in_function), args=(out1,))
        out2_vals = self.evaluate(nest.map_structure(strategy.unwrap, out2))
        self.assertAllEqual([2, 4], out2_vals['a'])
        self.assertAllEqual([1, 4], out2_vals['b'])
        out3 = strategy.run(_maybe_run_in_function(lambda b, a: a + 2 * b + 2, run_in_function), kwargs=out2)
        self.assertAllEqual([6, 14], self.evaluate(strategy.unwrap(out3)))

    def _test_all_reduce_sum(self, strategy, run_in_function=False):
        if False:
            print('Hello World!')
        self._test_collective_comms(strategy, _all_sum, inputs=([1.0, 3.0], [[39.0, 2.0], [3.0, 41.0]]), expected=(4.0, [42.0, 43.0]), run_in_function=run_in_function)

    def _test_all_reduce_sum_gradients(self, strategy, run_in_function=False):
        if False:
            return 10
        self._test_collective_comms_gradients(strategy, _all_sum, inputs=[1.0, 3.0], expected_grads=[4.0, 4.0], run_in_function=run_in_function)

    def _test_all_reduce_sum_gradient_tape(self, strategy, run_in_function=False):
        if False:
            i = 10
            return i + 15
        self._test_collective_comms_gradient_tape(strategy, _all_sum, inputs=[1.0, 3.0], expected_grads=[4.0, 4.0], run_in_function=run_in_function)

    def _test_all_reduce_mean(self, strategy, run_in_function=False):
        if False:
            i = 10
            return i + 15
        self._test_collective_comms(strategy, _all_mean, inputs=([1.0, 3.0], [[39.0, 2.0], [3.0, 41.0]]), expected=(2.0, [21.0, 21.5]), run_in_function=run_in_function)

    def _test_all_reduce_mean_gradients(self, strategy, run_in_function=False):
        if False:
            while True:
                i = 10
        self._test_collective_comms_gradients(strategy, _all_mean, inputs=[1.0, 3.0], expected_grads=[2.0, 2.0], run_in_function=run_in_function)

    def _test_all_reduce_mean_gradient_tape(self, strategy, run_in_function=False):
        if False:
            print('Hello World!')
        self._test_collective_comms_gradient_tape(strategy, _all_mean, inputs=[1.0, 3.0], expected_grads=[2.0, 2.0], run_in_function=run_in_function)

    def _test_collective_comms(self, strategy, comm_fn, inputs, expected, run_in_function=False):
        if False:
            return 10
        inputs = strategy.make_input_fn_iterator(lambda _: dataset_ops.Dataset.from_tensor_slices(inputs))
        self.evaluate(inputs.initialize())
        outputs = self.evaluate(list(map(strategy.experimental_local_results, strategy.experimental_run(_maybe_run_in_function(comm_fn, run_in_function), inputs))))
        self.assertAllEqual([expected[0], expected[0]], outputs[0])
        self.assertAllEqual([expected[1], expected[1]], outputs[1])

    def _test_collective_comms_gradients(self, strategy, comm_fn, inputs, expected_grads, run_in_function=False):
        if False:
            print('Hello World!')
        if context.executing_eagerly() and (not run_in_function):
            self.skipTest('`tf.gradients` is not supported with eager execution without using tf.functions.')

        def step(c):
            if False:
                while True:
                    i = 10
            x = array_ops.identity(42.0)
            y = comm_fn(x) * c
            return gradients_impl.gradients(y, [x])[0]
        inputs = strategy.make_input_fn_iterator(lambda _: dataset_ops.Dataset.from_tensor_slices(inputs))
        self.evaluate(inputs.initialize())
        self.assertAllEqual(expected_grads, self.evaluate(strategy.experimental_local_results(strategy.experimental_run(_maybe_run_in_function(step, run_in_function), inputs))))

    def _test_collective_comms_gradient_tape(self, strategy, comm_fn, inputs, expected_grads, run_in_function=False):
        if False:
            for i in range(10):
                print('nop')

        def step(c):
            if False:
                return 10
            x = array_ops.identity(42.0)
            with backprop.GradientTape() as tape:
                tape.watch(x)
                y = comm_fn(x) * c
            return tape.gradient(y, x)
        inputs = strategy.make_input_fn_iterator(lambda _: dataset_ops.Dataset.from_tensor_slices(inputs))
        self.evaluate(inputs.initialize())
        self.assertAllEqual(expected_grads, self.evaluate(strategy.experimental_local_results(strategy.experimental_run(_maybe_run_in_function(step, run_in_function), inputs))))

class RemoteSingleWorkerMirroredStrategyBase(DistributionTestBase):
    """Tests for a Remote single worker."""

    def _get_num_gpus(self):
        if False:
            return 10
        pass

    def _testNumReplicasInSync(self, distribution):
        if False:
            return 10
        self.assertEqual(self._get_num_gpus(), distribution.num_replicas_in_sync)

    def _testMinimizeLoss(self, distribution):
        if False:
            i = 10
            return i + 15
        if context.executing_eagerly():
            self._test_minimize_loss_eager(distribution)
        else:
            self._test_minimize_loss_graph(distribution, learning_rate=0.05)

    def _testDeviceScope(self, distribution):
        if False:
            i = 10
            return i + 15
        with distribution.scope():
            a = array_ops.identity(1.0)
            with ops.device('/cpu:0'):
                b = array_ops.identity(1.0)
            if context.executing_eagerly():
                device = '/job:worker/replica:0/task:0/device:CPU:0'
            else:
                device = '/job:worker/replica:0/task:0'
            self.assertEqual(a.device, device)
            self.assertEqual(b.device, '/job:worker/replica:0/task:0/device:CPU:0')

    def _testMakeInputFnIteratorWithDataset(self, distribution):
        if False:
            return 10
        dataset_fn = lambda : dataset_ops.Dataset.range(100)
        num_gpus = self._get_num_gpus()
        num_workers = 1
        expected_values = [[i + j for j in range(num_gpus)] * num_workers for i in range(0, 100, num_gpus)]
        with self.cached_session() as sess:
            input_fn = self._input_fn_to_test_input_context(dataset_fn, expected_num_replicas_in_sync=num_workers * num_gpus, expected_num_input_pipelines=num_workers, expected_input_pipeline_id=None)
            iterator = distribution.make_input_fn_iterator(input_fn)
            self._test_input_fn_iterator(iterator, distribution.extended.worker_devices, expected_values, sess)

    def _testMakeInputFnIteratorWithCallable(self, distribution):
        if False:
            for i in range(10):
                print('nop')

        def fn():
            if False:
                i = 10
                return i + 15
            dataset = dataset_ops.Dataset.range(100)
            it = dataset_ops.make_one_shot_iterator(dataset)
            return it.get_next
        num_gpus = self._get_num_gpus()
        num_workers = 1
        expected_values = []
        for i in range(0, 100, num_gpus):
            expected_values.append([i + j for j in range(num_gpus)] * num_workers)
        with self.cached_session() as sess:
            input_fn = self._input_fn_to_test_input_context(fn, expected_num_replicas_in_sync=num_workers * num_gpus, expected_num_input_pipelines=num_workers, expected_input_pipeline_id=None)
            iterator = distribution.make_input_fn_iterator(input_fn)
            self._test_input_fn_iterator(iterator, distribution.extended.worker_devices, expected_values, sess, test_reinitialize=False, ignore_order=True)

def _all_sum(value):
    if False:
        return 10
    ctx = distribute_lib.get_replica_context()
    return ctx.all_reduce(reduce_util.ReduceOp.SUM, value)

def _all_mean(value):
    if False:
        print('Hello World!')
    ctx = distribute_lib.get_replica_context()
    return ctx.all_reduce(reduce_util.ReduceOp.MEAN, value)