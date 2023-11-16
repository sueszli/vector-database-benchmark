"""Tests for TPUStrategy."""
from absl import logging
from absl.testing import parameterized
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.distribute import tpu_strategy as tpu_lib
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_switch_case
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import flags
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu import tpu_hardware_feature
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu import tpu_strategy_util
from tensorflow.python.training import server_lib
from tensorflow.python.util import nest
FLAGS = flags.FLAGS
flags.DEFINE_string('tpu', '', 'Name of TPU to connect to.')
flags.DEFINE_string('project', None, 'Name of GCP project with TPU.')
flags.DEFINE_string('zone', None, 'Name of GCP zone with TPU.')

def get_tpu_cluster_resolver():
    if False:
        while True:
            i = 10
    resolver = tpu_cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu, zone=FLAGS.zone, project=FLAGS.project)
    return resolver

def get_tpu_strategy(enable_packed_var=False):
    if False:
        for i in range(10):
            print('nop')
    resolver = get_tpu_cluster_resolver()
    remote.connect_to_cluster(resolver)
    tpu_cluster_resolver.initialize_tpu_system(resolver)
    strategy = tpu_lib.TPUStrategyV2(resolver)
    strategy._enable_packed_variable_in_eager_mode = enable_packed_var
    return strategy

@test_util.with_eager_op_as_function
class TPUTest(test.TestCase):

    def test_single_tpu_jit_compile(self):
        if False:
            print('Hello World!')
        with ops.device('/device:TPU:0'):
            a = variables.Variable(1)

        def get_a_plus_one():
            if False:
                i = 10
                return i + 15
            return a + 1

        @def_function.function(input_signature=[tensor_spec.TensorSpec([], dtypes.int32)])
        def foo(x):
            if False:
                i = 10
                return i + 15
            b = x + get_a_plus_one()
            b = b + get_a_plus_one()
            return b + 1
        with ops.device('/device:TPU:0'):
            result = foo(a)
        self.assertAllEqual(6, result)

    def test_single_tpu_jit_compile_with_outside_compilation(self):
        if False:
            while True:
                i = 10
        context.enable_jit_compile_rewrite()
        get_tpu_strategy(True)
        config.set_soft_device_placement(True)
        with ops.device('/device:TPU:1'):
            a = variables.Variable(1)

        def get_a_plus_one():
            if False:
                return 10
            return a + 1

        @def_function.function(input_signature=[tensor_spec.TensorSpec([], dtypes.int32)])
        def foo(x):
            if False:
                i = 10
                return i + 15
            b = x + get_a_plus_one()
            my_str = string_ops.as_string(b)
            new_str = my_str + '0'
            c = string_ops.string_to_number(new_str, out_type=dtypes.int32)
            logging_ops.print_v2(c)
            b = c + get_a_plus_one()
            return b + 1
        with ops.device('/device:TPU:1'):
            result = foo(a)
        self.assertAllEqual(33, result)

    def test_single_tpu_on_demand(self):
        if False:
            i = 10
            return i + 15
        with ops.device('/device:TPU:0'):
            a = variables.Variable(1)

        def get_a_plus_one():
            if False:
                i = 10
                return i + 15
            return a + 1
        x = 1
        with ops.device('/device:TPU:0'):
            b = x + get_a_plus_one()
            b = b + get_a_plus_one()
        result = b + 1
        self.assertAllEqual(6, result)

    def test_single_tpu_on_demand_tf_function(self):
        if False:
            i = 10
            return i + 15
        with ops.device('/device:TPU:0'):
            a = variables.Variable(1)

        def get_a_plus_one():
            if False:
                i = 10
                return i + 15
            return a + 1

        @def_function.function(input_signature=[tensor_spec.TensorSpec([], dtypes.int32)])
        def foo(x):
            if False:
                while True:
                    i = 10
            with ops.device('/device:TPU:0'):
                b = x + get_a_plus_one()
                b = b + get_a_plus_one()
            return b + 1
        result = foo(a)
        self.assertAllEqual(6, result)

    def test_multiple_initialize_system(self):
        if False:
            for i in range(10):
                print('nop')
        resolver = get_tpu_cluster_resolver()
        remote.connect_to_cluster(resolver)
        tpu_cluster_resolver.initialize_tpu_system(resolver)
        with test.mock.patch.object(logging, 'warning') as mock_log:
            tpu_cluster_resolver.initialize_tpu_system(resolver)
            self.assertRegex(str(mock_log.call_args), 'already been initialized')

    def test_initialize_tpu_system_impl_input(self):
        if False:
            i = 10
            return i + 15
        resolver = get_tpu_cluster_resolver()
        with self.assertRaisesRegex(TypeError, 'tpu_cluster_resolver_cls is not tf.distribute.cluster_resolver.TPUClusterResolver.'):
            tpu_strategy_util.initialize_tpu_system_impl(resolver, tpu_cluster_resolver_cls=None)

    def test_shutdown_tpu_system_impl_input(self):
        if False:
            i = 10
            return i + 15
        resolver = get_tpu_cluster_resolver()
        with self.assertRaisesRegex(TypeError, 'tpu_cluster_resolver_cls is not tf.distribute.cluster_resolver.TPUClusterResolver.'):
            tpu_strategy_util.shutdown_tpu_system_impl(resolver, tpu_cluster_resolver_cls=None)

    def test_tpu_tf_function_same_device(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.device('/device:TPU:0'):
            a = variables.Variable(1)

        @def_function.function(experimental_attributes={'_noinline': True})
        def get_a_plus_one():
            if False:
                print('Hello World!')
            return a + 1

        @def_function.function(input_signature=[tensor_spec.TensorSpec([], dtypes.int32)])
        def foo(x):
            if False:
                print('Hello World!')
            with ops.device('/device:TPU:0'):
                b = x + get_a_plus_one()
            return b + 1
        result = foo(a)
        self.assertAllEqual(4, result)

    def test_tpu_return_int32(self):
        if False:
            i = 10
            return i + 15
        with ops.device('/device:TPU:0'):
            a = variables.Variable(0)

        @def_function.function
        def foo():
            if False:
                print('Hello World!')
            return a + 1

        @def_function.function
        def bar():
            if False:
                print('Hello World!')
            with ops.device('/device:TPU:1'):
                return foo()
        with ops.device('/device:CPU:0'):
            result = bar() + 1
            self.assertAllEqual(result, 2)

    def test_tpu_output_device(self):
        if False:
            i = 10
            return i + 15

        def foo():
            if False:
                return 10
            return 1 + 1
        func1 = def_function.function(foo, jit_compile=False)
        func2 = def_function.function(foo, jit_compile=False, experimental_attributes={'_OutputsOnOpDevice': True})
        with ops.device('/device:TPU:0'):
            ret1 = func1()
            ret2 = func2()
        self.assertAllEqual(ret1.backing_device, '/job:localhost/replica:0/task:0/device:CPU:0')
        self.assertAllEqual(ret2.backing_device, '/job:localhost/replica:0/task:0/device:TPU:0')

    def test_on_demand_op_with_dynamic_output(self):
        if False:
            while True:
                i = 10
        with ops.device('/device:TPU:0'):
            where_output = array_ops.where([True, False, True])
        self.assertAllEqual(where_output, [[0], [2]])
        with ops.device('/device:TPU:0'):
            repeat_output = array_ops.repeat(math_ops.range(2), [1, 4])
        self.assertAllEqual(repeat_output, [0, 1, 1, 1, 1])

@parameterized.named_parameters([('PackedVar', True), ('', False)])
@test_util.with_eager_op_as_function
class TPUStrategyTest(test.TestCase, parameterized.TestCase):

    def test_handle_in_cross_replica_context(self, enable_packed_var):
        if False:
            print('Hello World!')
        strategy = get_tpu_strategy(enable_packed_var)
        with strategy.scope():
            v = variables.Variable(1.0)

        @def_function.function
        def func():
            if False:
                for i in range(10):
                    print('nop')
            self.assertEndsWith(v.handle.device, 'device:TPU:0')
            return v + 1.0
        ret = func()
        self.assertAllEqual(ret, 2.0)

    def testStaticHashTableDatasetFnHostTrainingLoop(self, enable_packed_var):
        if False:
            for i in range(10):
                print('nop')
        self._dataset_fn_tracing_count = 0
        strategy = get_tpu_strategy(enable_packed_var)
        with strategy.scope():
            vals = [0, 1, 2]
            keys_tensor = constant_op.constant(list(range(len(vals))), dtype=dtypes.int64)
            vals_tensor = constant_op.constant(vals)
            initializer = lookup_ops.KeyValueTensorInitializer(keys_tensor, vals_tensor)
            per_worker_table = lookup_ops.StaticHashTable(initializer, default_value=-1)

        @def_function.function
        def dataset_fn(input_context):
            if False:
                i = 10
                return i + 15
            tensor = constant_op.constant([0, 1, 3], dtype=dtypes.int64)
            global_batch_size = 2
            batch_size = input_context.get_per_replica_batch_size(global_batch_size)
            dataset = dataset_ops.Dataset.from_tensors(tensor).repeat().batch(batch_size, drop_remainder=True)
            dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
            dataset = dataset.prefetch(2)
            dataset = dataset.map(per_worker_table.lookup)
            self._dataset_fn_tracing_count += 1
            return dataset
        dist_iterator = iter(strategy.experimental_distribute_datasets_from_function(dataset_fn))

        @def_function.function
        def step_fn(inputs):
            if False:
                print('Hello World!')
            return math_ops.reduce_sum(inputs)

        def train_steps(iterator, steps):
            if False:
                while True:
                    i = 10
            for _ in math_ops.range(steps):
                strategy.run(step_fn, args=(next(iterator),))
        train_steps(dist_iterator, steps=5)
        self.assertEqual(self._dataset_fn_tracing_count, 1)

    def test_function_compile_with_xla(self, enable_packed_var):
        if False:
            i = 10
            return i + 15
        if FLAGS.tpu_use_tfrt:
            self.skipTest('This test triggers _XlaCompile and XlaLaunch which are not supported in tfrt yet. We should avoid using these kernels on TPU. However, it is a workaround to support b/129842431. We need more discussion about how to support it in the long term.')
        strategy = get_tpu_strategy(enable_packed_var)
        with strategy.scope():
            v = variables.Variable(1.0)

        @def_function.function
        def func():
            if False:
                for i in range(10):
                    print('nop')
            return v.read_value() + 1.0
        with ops.device('/device:TPU:0'):
            self.assertAllEqual(func(), 2.0)

    def test_sequential_runs(self, enable_packed_var):
        if False:
            while True:
                i = 10
        resolver = get_tpu_cluster_resolver()
        remote.connect_to_cluster(resolver)
        topology = tpu_cluster_resolver.initialize_tpu_system(resolver)
        device_assignment = device_assignment_lib.DeviceAssignment.build(topology, num_replicas=2)
        strategy = tpu_lib.TPUStrategyV2(resolver, experimental_device_assignment=device_assignment)
        strategy._enable_packed_variable_in_eager_mode = enable_packed_var
        device_assignment2 = device_assignment_lib.DeviceAssignment.build(topology, num_replicas=1)
        strategy2 = tpu_lib.TPUStrategyV2(resolver, experimental_device_assignment=device_assignment2)

        def computation(x):
            if False:
                while True:
                    i = 10
            return math_ops.square(x)

        @def_function.function
        def train_step():
            if False:
                while True:
                    i = 10
            outputs = strategy.experimental_local_results(strategy.run(computation, args=([2.0, 2.0],)))
            outputs2 = strategy2.run(computation, args=([outputs[0]],))
            return outputs2
        self.assertAllEqual([[16.0, 16.0]], train_step())

    def test_device_switch_case(self, enable_packed_var):
        if False:
            while True:
                i = 10
        strategy = get_tpu_strategy(enable_packed_var)
        with strategy.scope():
            a = variables.Variable(1)
        inference_iteration = variables.Variable(-1)

        def inference_fn(x, i):
            if False:
                i = 10
                return i + 15
            return a + x + i

        @def_function.function
        def run_inference(x):
            if False:
                return 10

            def do_inference(device, inference_fn, i):
                if False:
                    print('Hello World!')
                with ops.device(device):
                    return inference_fn(x, i)
            branch_fns = {0: lambda : do_inference('/device:TPU:0', inference_fn, 0), 1: lambda : do_inference('/device:TPU:1', inference_fn, 1)}
            branch_index = inference_iteration.assign_add(1, use_locking=True) % 2
            return control_flow_switch_case.switch_case(branch_index, branch_fns)
        self.assertAllEqual(2.0, run_inference(1))
        self.assertAllEqual(3.0, run_inference(1))

    def test_recover_from_compilation_failures(self, enable_packed_var):
        if False:
            while True:
                i = 10
        if FLAGS.tpu:
            self.skipTest('Recovery fails for non-local TPU, see b/148150981')
        config.set_soft_device_placement(False)
        strategy = get_tpu_strategy(enable_packed_var)

        @def_function.function
        def compilation_failure_run():
            if False:
                i = 10
                return i + 15

            def computation():
                if False:
                    while True:
                        i = 10
                return random_ops.random_gamma([10], [0.5, 1.5])
            return strategy.run(computation)
        with self.assertRaises(errors.OpError):
            compilation_failure_run()

        @def_function.function
        def good_run():
            if False:
                i = 10
                return i + 15

            def computation():
                if False:
                    i = 10
                    return i + 15
                return random_ops.random_normal([10])
            return strategy.run(computation)
        good_run()

    def test_dynamic_shape_with_outside_compilation_failure(self, enable_packed_var):
        if False:
            return 10
        config.set_soft_device_placement(True)
        strategy = get_tpu_strategy(enable_packed_var)
        dataset = dataset_ops.Dataset.from_tensors(('string', 1.0)).repeat().batch(2, drop_remainder=False)
        dataset = strategy.experimental_distribute_dataset(dataset)
        iterator = iter(dataset)

        @def_function.function
        def train_fn(iterator):
            if False:
                while True:
                    i = 10

            def step_fn(inputs):
                if False:
                    i = 10
                    return i + 15
                (input0, input1) = inputs
                return (array_ops.size(input0), math_ops.reduce_sum(input1))
            return strategy.experimental_local_results(strategy.run(step_fn, args=(next(iterator),)))
        with self.assertRaises(errors.InvalidArgumentError):
            logging.info(train_fn(iterator))

    def test_computation_on_subset_cores(self, enable_packed_var):
        if False:
            i = 10
            return i + 15
        resolver = get_tpu_cluster_resolver()
        remote.connect_to_cluster(resolver)
        topology = tpu_cluster_resolver.initialize_tpu_system(resolver)
        all_core_strategy = tpu_lib.TPUStrategyV2(resolver)
        all_core_strategy._enable_packed_variable_in_eager_mode = enable_packed_var
        with all_core_strategy.scope():
            v = variables.Variable(0.0, aggregation=variables.VariableAggregation.MEAN)
        device_assignment = device_assignment_lib.DeviceAssignment.build(topology, num_replicas=1)
        first_core_strategy = tpu_lib.TPUStrategyV2(resolver, experimental_device_assignment=device_assignment)
        first_core_strategy._enable_packed_variable_in_eager_mode = enable_packed_var
        device_assignment2 = device_assignment_lib.DeviceAssignment(topology, [[[0, 0, 0, 1]]])
        second_core_strategy = tpu_lib.TPUStrategyV2(resolver, experimental_device_assignment=device_assignment2)
        second_core_strategy._enable_packed_variable_in_eager_mode = enable_packed_var

        @def_function.function
        def train_step():
            if False:
                i = 10
                return i + 15

            def step_fn():
                if False:
                    for i in range(10):
                        print('nop')
                return v + 1.0
            all_core_strategy.run(step_fn)
            r1 = first_core_strategy.run(step_fn)
            r2 = second_core_strategy.run(step_fn)
            return r1 + r2
        train_step()
        self.assertAllEqual(2.0, train_step())

    def test_worker_devices_on_subset_cores(self, enable_packed_var):
        if False:
            print('Hello World!')
        resolver = get_tpu_cluster_resolver()
        remote.connect_to_cluster(resolver)
        topology = tpu_cluster_resolver.initialize_tpu_system(resolver)
        device_assignment = device_assignment_lib.DeviceAssignment.build(topology, num_replicas=1)
        first_core_strategy = tpu_lib.TPUStrategyV2(resolver, experimental_device_assignment=device_assignment)
        first_core_strategy._enable_packed_variable_in_eager_mode = enable_packed_var
        device_assignment2 = device_assignment_lib.DeviceAssignment(topology, [[[0, 0, 0, 1]]])
        second_core_strategy = tpu_lib.TPUStrategyV2(resolver, experimental_device_assignment=device_assignment2)
        second_core_strategy._enable_packed_variable_in_eager_mode = enable_packed_var
        self.assertLen(first_core_strategy.extended.worker_devices, 1)
        self.assertEndsWith(first_core_strategy.extended.worker_devices[0], 'device:TPU:0')
        self.assertLen(second_core_strategy.extended.worker_devices, 1)
        self.assertEndsWith(second_core_strategy.extended.worker_devices[0], 'device:TPU:1')

    def test_control_output_in_while_body_fn(self, enable_packed_var):
        if False:
            while True:
                i = 10
        strategy = get_tpu_strategy(enable_packed_var)
        with strategy.scope():
            v = variables.Variable(0.0, aggregation=variables.VariableAggregation.MEAN)

        @def_function.function
        def train_step():
            if False:
                while True:
                    i = 10

            def step_fn():
                if False:
                    print('Hello World!')
                v.assign_add(1)
            for _ in math_ops.range(2):
                strategy.run(step_fn)
        train_step()
        self.assertEqual(2.0, v.numpy())

    def test_cluster_conditional_with_dynamic_shape(self, enable_packed_var):
        if False:
            return 10
        strategy = get_tpu_strategy(enable_packed_var)

        @def_function.function
        def train_step():
            if False:
                i = 10
                return i + 15

            def shape_list(tensor):
                if False:
                    print('Hello World!')
                shape = tensor.shape.as_list()
                non_static_indexes = []
                for (index, dim) in enumerate(shape):
                    if dim is None:
                        non_static_indexes.append(index)
                if not non_static_indexes:
                    return shape
                dynamic_shape = array_ops.shape(input=tensor)
                for index in non_static_indexes:
                    shape[index] = dynamic_shape[index]
                return shape

            def step_fn(condition):
                if False:
                    return 10
                where = array_ops.where(condition)
                if array_ops.shape(where)[0] > 0:
                    tensor_shape = shape_list(where)
                    d1 = tensor_shape[0]
                    d2 = tensor_shape[1]
                    where = array_ops.reshape(where, [d1, d2])
                return where
            return strategy.run(step_fn, args=([True, False, True],))
        outputs = strategy.experimental_local_results(train_step())
        self.assertAllEqual(outputs[0].numpy(), [[0], [2]])

    def test_cluster_in_graph_and_while_body_fn(self, enable_packed_var):
        if False:
            while True:
                i = 10
        strategy = get_tpu_strategy(enable_packed_var)

        @def_function.function
        def train_step():
            if False:
                i = 10
                return i + 15

            def step_fn(prev):
                if False:
                    print('Hello World!')
                s = prev + 1
                return s

            def init_fn():
                if False:
                    i = 10
                    return i + 15
                return array_ops.zeros(shape=())
            prev = strategy.run(init_fn)
            for _ in math_ops.range(10):
                prev = strategy.run(step_fn, args=(prev,))
            return strategy.reduce(reduce_util.ReduceOp.SUM, prev, axis=None)
        sum_val = train_step().numpy().astype(float)
        self.assertEqual(sum_val, strategy.num_replicas_in_sync * 10)

    def test_two_clusters_with_same_fn(self, enable_packed_var):
        if False:
            i = 10
            return i + 15
        strategy = get_tpu_strategy(enable_packed_var)

        @def_function.function
        def foo(x):
            if False:
                return 10
            return strategy.run(lambda x: x + 1, (x,))

        @def_function.function
        def bar(x):
            if False:
                for i in range(10):
                    print('nop')
            foo(x)
            return foo(x)
        bar(1)

    def test_tpu_variable_run_argument(self, enable_packed_var):
        if False:
            while True:
                i = 10
        strategy = get_tpu_strategy(enable_packed_var)
        with strategy.scope():
            tpu_variable = variables.Variable(1)

        def replica_step(first_arg, variable):
            if False:
                i = 10
                return i + 15
            del first_arg
            if variable is not None:
                self.assertIsInstance(variable, tpu_values.TPUDistributedVariable)

        @def_function.function
        def step():
            if False:
                for i in range(10):
                    print('nop')
            strategy.run(replica_step, args=(2, tpu_variable))
        step()

    def test_tpu_run_arg_parsing(self, enable_packed_var):
        if False:
            while True:
                i = 10
        strategy = get_tpu_strategy(enable_packed_var)
        with strategy.scope():
            tpu_vars = [variables.Variable(1)]

        def only_star_args(*args):
            if False:
                for i in range(10):
                    print('nop')
            del args

        def pos_and_star_args(first_arg, *args):
            if False:
                return 10
            del first_arg
            del args

        def named_args(first_arg, second_arg):
            if False:
                while True:
                    i = 10
            del first_arg
            del second_arg

        def star_args_and_kw_only(*args, kw):
            if False:
                print('Hello World!')
            del args
            del kw

        @def_function.function
        def step():
            if False:
                while True:
                    i = 10
            strategy.run(only_star_args, args=(2,))
        step()

        @def_function.function
        def step():
            if False:
                print('Hello World!')
            strategy.run(named_args, kwargs={'first_arg': 2, 'second_arg': 3})
        step()
        with self.assertRaisesRegex(TypeError, 'got multiple values for argument'):

            @def_function.function
            def step():
                if False:
                    print('Hello World!')
                strategy.run(named_args, args=(1,), kwargs={'first_arg': 2, 'second_arg': 3})
            step()
        with self.assertRaisesRegex(ValueError, 'cannot handle Variables passed to \\*args'):

            @def_function.function
            def step():
                if False:
                    print('Hello World!')
                strategy.run(only_star_args, args=(2, tpu_vars))
            step()

        @def_function.function
        def step():
            if False:
                i = 10
                return i + 15
            strategy.run(pos_and_star_args, args=(2, 3, 4))
        step()

        @def_function.function
        def step():
            if False:
                while True:
                    i = 10
            strategy.run(star_args_and_kw_only, args=(2, 3), kwargs={'kw': tpu_vars})
        step()
        with self.assertRaisesRegex(ValueError, 'mix of positional args and \\*args'):

            @def_function.function
            def step():
                if False:
                    return 10
                strategy.run(pos_and_star_args, args=(tpu_vars, 3, 4))
            step()
        with self.assertRaisesRegex(ValueError, 'Too many positional arguments'):

            @def_function.function
            def step():
                if False:
                    while True:
                        i = 10
                strategy.run(named_args, args=(2, 3, 4))
            step()

        class DummyClass:

            @def_function.function
            def method(self, arg_1):
                if False:
                    print('Hello World!')
                del arg_1

            def step(self):
                if False:
                    i = 10
                    return i + 15
                strategy.run(self.method, args=(tpu_vars,))
        DummyClass().step()

    def test_using_external_variable_inside_tf_function(self, enable_packed_var):
        if False:
            while True:
                i = 10
        strategy = get_tpu_strategy(enable_packed_var)
        dataset = dataset_ops.Dataset.range(strategy.num_replicas_in_sync * 2, output_type=dtypes.float32).batch(strategy.num_replicas_in_sync)
        input_iterator = iter(strategy.experimental_distribute_dataset(dataset))
        v = variables.Variable(2.0)

        @def_function.function
        def train_step(data):
            if False:
                print('Hello World!')

            def computation(inputs):
                if False:
                    for i in range(10):
                        print('nop')
                return inputs + v
            return strategy.run(computation, args=(data,))
        expected_result = [[x + 2.0] for x in range(0, strategy.num_replicas_in_sync)]
        self.assertAllEqual(expected_result, strategy.experimental_local_results(train_step(next(input_iterator))))

    def test_all_reduce_on_sync_on_read_variable(self, enable_packed_var):
        if False:
            print('Hello World!')
        strategy = get_tpu_strategy(enable_packed_var)
        dataset = dataset_ops.Dataset.range(strategy.num_replicas_in_sync, output_type=dtypes.float32).batch(strategy.num_replicas_in_sync, drop_remainder=True)
        input_iterator = iter(strategy.experimental_distribute_dataset(dataset))
        with strategy.scope():
            w = variables.Variable((0.0,), shape=(1,), trainable=False, synchronization=variables.VariableSynchronization.ON_READ, aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA)
        self.assertFalse(w._is_mirrored())

        @def_function.function
        def run(iterator):
            if False:
                for i in range(10):
                    print('nop')

            def computation(x):
                if False:
                    return 10
                w.assign(x + w)
                return w

            def all_reduce(x):
                if False:
                    return 10
                ctx = distribute_lib.get_replica_context()
                return ctx.all_reduce('SUM', w) + x
            outputs = strategy.run(computation, args=(next(iterator),))
            outputs2 = strategy.experimental_local_results(strategy.run(all_reduce, args=(outputs,)))
            return outputs2
        data = range(0, strategy.num_replicas_in_sync)
        data_sum = sum(data)
        expected_result = [[x + data_sum] for x in range(0, strategy.num_replicas_in_sync)]
        self.assertAllEqual(expected_result, run(input_iterator))
        self.assertAllEqual((0.0,), w.read_value())

    def test_run_output_on_device(self, enable_packed_var):
        if False:
            i = 10
            return i + 15
        strategy = get_tpu_strategy(enable_packed_var)

        def computation(x):
            if False:
                while True:
                    i = 10
            return math_ops.square(x)

        @def_function.function
        def train_step():
            if False:
                print('Hello World!')
            outputs = strategy.experimental_local_results(strategy.run(computation, args=(2,)))
            return outputs
        results = train_step()
        self.assertAllEqual([4.0, 4.0], results)
        self.assertAllEqual('/job:localhost/replica:0/task:0/device:TPU:0', results[0].backing_device)
        self.assertAllEqual('/job:localhost/replica:0/task:0/device:TPU:1', results[1].backing_device)

    def test_run_passing_and_returning_nones(self, enable_packed_var):
        if False:
            i = 10
            return i + 15
        strategy = get_tpu_strategy(enable_packed_var)

        @def_function.function
        def train_step():
            if False:
                while True:
                    i = 10

            def computation(x):
                if False:
                    for i in range(10):
                        print('nop')
                return x
            outputs = strategy.experimental_local_results(strategy.run(computation, args=([1, [2, None]],)))
            return outputs
        results = train_step()
        self.assertAllEqual(1, results[0][0])
        self.assertAllEqual(2, results[0][1][0])
        self.assertIsNone(results[0][1][1])

    def test_run_passing_and_returning_empty_list(self, enable_packed_var):
        if False:
            i = 10
            return i + 15
        strategy = get_tpu_strategy(enable_packed_var)

        @def_function.function
        def train_step():
            if False:
                i = 10
                return i + 15

            def computation(x):
                if False:
                    i = 10
                    return i + 15
                return x
            outputs = strategy.experimental_local_results(strategy.run(computation, args=([],)))
            return outputs
        self.assertEqual([], train_step()[0])

    def test_run_passing_and_returning_empty_dict(self, enable_packed_var):
        if False:
            while True:
                i = 10
        strategy = get_tpu_strategy(enable_packed_var)

        @def_function.function
        def train_step():
            if False:
                while True:
                    i = 10

            def computation(x):
                if False:
                    return 10
                return x
            outputs = strategy.experimental_local_results(strategy.run(computation, args=({},)))
            return outputs
        self.assertEqual({}, train_step()[0])

    def test_composite_input_output(self, enable_packed_var):
        if False:
            i = 10
            return i + 15
        strategy = get_tpu_strategy(enable_packed_var)
        if strategy.num_replicas_in_sync != 2:
            self.skipTest('Test assumes two replicas.')
        with strategy.scope():
            table = variables.Variable(initial_value=[[0.0, 1.0], [3.0, 7.0]], dtype=dtypes.float32)

        @def_function.function
        def sparse_lookup(iterator):
            if False:
                while True:
                    i = 10

            def tpu_function(sparse):
                if False:
                    while True:
                        i = 10
                looked_up = array_ops.gather(table, sparse.values)
                segment_sum = math_ops.unsorted_segment_sum(looked_up, sparse.indices[:, 0], 2)
                return (sparse, segment_sum)
            return nest.map_structure(strategy.experimental_local_results, strategy.run(tpu_function, args=(next(iterator),)))

        def dataset_fn(_):
            if False:
                i = 10
                return i + 15
            dataset = dataset_ops.Dataset.range(2)

            def make_sparse(_):
                if False:
                    for i in range(10):
                        print('nop')
                return sparse_tensor.SparseTensor(indices=array_ops.constant([[0, 0], [1, 0], [1, 1]], dtype=dtypes.int64), values=array_ops.constant([0, 0, 1], dtype=dtypes.int32), dense_shape=array_ops.constant([2, 2], dtype=dtypes.int64))
            return dataset.map(make_sparse)
        dataset = iter(strategy.distribute_datasets_from_function(dataset_fn, distribute_lib.InputOptions(experimental_fetch_to_device=False)))
        (sparse, result) = sparse_lookup(dataset)
        for replica in range(strategy.num_replicas_in_sync):
            self.assertIsInstance(sparse[replica], sparse_tensor.SparseTensor)
            self.assertAllEqual(sparse[replica].indices, [[0, 0], [1, 0], [1, 1]])
            self.assertAllEqual(sparse[replica].values, [0, 0, 1])
            self.assertAllEqual(sparse[replica].dense_shape, [2, 2])
            self.assertAllEqual(result[replica], [[0.0, 1.0], [3.0, 8.0]])

    def test_composite_input_non_flat_output(self, enable_packed_var):
        if False:
            i = 10
            return i + 15
        strategy = get_tpu_strategy(enable_packed_var)
        if strategy.num_replicas_in_sync != 2:
            self.skipTest('Test assumes two replicas.')
        with strategy.scope():
            table = variables.Variable(initial_value=[[0.0, 1.0], [3.0, 7.0]], dtype=dtypes.float32)

        @def_function.function
        def sparse_lookup(iterator):
            if False:
                while True:
                    i = 10

            def tpu_function(sparse):
                if False:
                    return 10
                looked_up = array_ops.gather(table, sparse.values)
                segment_sum = math_ops.unsorted_segment_sum(looked_up, sparse.indices[:, 0], 2)
                return {'sparse': sparse, 'segment_sum': segment_sum}
            return nest.map_structure(strategy.experimental_local_results, strategy.run(tpu_function, args=(next(iterator),)))

        def dataset_fn(_):
            if False:
                print('Hello World!')
            dataset = dataset_ops.Dataset.range(2)

            def make_sparse(_):
                if False:
                    print('Hello World!')
                return sparse_tensor.SparseTensor(indices=array_ops.constant([[0, 0], [1, 0], [1, 1]], dtype=dtypes.int64), values=array_ops.constant([0, 0, 1], dtype=dtypes.int32), dense_shape=array_ops.constant([2, 2], dtype=dtypes.int64))
            return dataset.map(make_sparse)
        dataset = iter(strategy.distribute_datasets_from_function(dataset_fn, distribute_lib.InputOptions(experimental_fetch_to_device=False)))
        output = sparse_lookup(dataset)
        for replica in range(strategy.num_replicas_in_sync):
            self.assertIsInstance(output['sparse'][replica], sparse_tensor.SparseTensor)
            self.assertAllEqual(output['sparse'][replica].indices, [[0, 0], [1, 0], [1, 1]])
            self.assertAllEqual(output['sparse'][replica].values, [0, 0, 1])
            self.assertAllEqual(output['sparse'][replica].dense_shape, [2, 2])
            self.assertAllEqual(output['segment_sum'][replica], [[0.0, 1.0], [3.0, 8.0]])

    def test_composite_input_dynamic_shapes_outside_compilation(self, enable_packed_var):
        if False:
            for i in range(10):
                print('nop')
        strategy = get_tpu_strategy(enable_packed_var)
        if strategy.num_replicas_in_sync != 2:
            self.skipTest('Test assumes two replicas.')
        table = variables.Variable(initial_value=[[0.0, 1.0], [3.0, 7.0]], dtype=dtypes.float32)

        @def_function.function
        def sparse_lookup(iterator):
            if False:
                print('Hello World!')

            def tpu_function(sparse):
                if False:
                    print('Hello World!')
                lookup = tpu_replication.outside_compilation(embedding_ops.safe_embedding_lookup_sparse, table, sparse)
                return math_ops.reduce_sum(lookup, axis=0)
            return strategy.experimental_local_results(strategy.run(tpu_function, args=(next(iterator),)))

        def dataset_fn(_):
            if False:
                for i in range(10):
                    print('nop')
            dataset = dataset_ops.Dataset.range(2)

            def make_sparse(i):
                if False:
                    return 10
                indices = array_ops.constant([[0, 0], [1, 0], [1, 1]], dtype=dtypes.int64)[0:2 + i]
                values = array_ops.constant([0, 0, 1], dtype=dtypes.int32)[0:2 + i]
                shape = [array_ops.constant([2], dtype=dtypes.int64), array_ops.expand_dims(1 + i, axis=0)]
                dense_shape = array_ops.concat(shape, axis=0)
                return sparse_tensor.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
            return dataset.map(make_sparse)
        dataset = iter(strategy.distribute_datasets_from_function(dataset_fn, options=distribute_lib.InputOptions(experimental_fetch_to_device=False)))
        result = sparse_lookup(dataset)
        self.assertAllEqual(result, [[0.0, 2.0], [1.5, 5.0]])

    def test_composite_input_with_non_flat_components(self, enable_packed_var):
        if False:
            return 10
        strategy = get_tpu_strategy(enable_packed_var)

        class TestCompositeTypeSpec(type_spec.TypeSpec):

            def __init__(self, component_type_spec):
                if False:
                    for i in range(10):
                        print('nop')
                self._component_type_spec = component_type_spec

            @property
            def value_type(self):
                if False:
                    i = 10
                    return i + 15
                return TestComposite

            def _to_components(self, value):
                if False:
                    return 10
                return value.values

            def _from_components(self, components):
                if False:
                    while True:
                        i = 10
                return TestComposite(components[0], components[1][0], components[1][1])

            @property
            def _component_specs(self):
                if False:
                    return 10
                return [self._component_type_spec, [self._component_type_spec, self._component_type_spec]]

            def _serialize(self):
                if False:
                    return 10
                return (self._component_type_spec,)

        class TestComposite(composite_tensor.CompositeTensor):

            def __init__(self, value1, value2, value3):
                if False:
                    for i in range(10):
                        print('nop')
                self.values = [value1, [value2, value3]]

            @property
            def _type_spec(self):
                if False:
                    while True:
                        i = 10
                return TestCompositeTypeSpec(tensor_spec.TensorSpec.from_tensor(self.values[0]))

            def _shape_invariant_to_type_spec(self, shape):
                if False:
                    return 10
                return [shape, [shape, shape]]

        @def_function.function
        def test_fn(test_composite):
            if False:
                for i in range(10):
                    print('nop')

            def tpu_function(composite):
                if False:
                    while True:
                        i = 10
                return (composite, composite.values[0] + (composite.values[1][0] + composite.values[1][1]) / 2)
            return nest.map_structure(strategy.experimental_local_results, strategy.run(tpu_function, args=(test_composite,)))
        a = array_ops.constant([0.1])
        b = array_ops.constant([1.2])
        c = array_ops.constant([-0.4])
        test_composite = TestComposite(a, b, c)
        (composite, result) = test_fn(test_composite)
        for replica in range(strategy.num_replicas_in_sync):
            self.assertIsInstance(composite[replica], TestComposite)
            self.assertAllEqual(composite[replica].values[0], a)
            self.assertAllEqual(composite[replica].values[1][0], b)
            self.assertAllEqual(composite[replica].values[1][1], c)
            self.assertAllEqual(result[replica], array_ops.constant([0.50000006]))

    def test_per_device_tracing_of_mirrored_variables(self, enable_packed_var):
        if False:
            i = 10
            return i + 15
        trace_count = [0]
        strategy = get_tpu_strategy(enable_packed_var)
        with strategy.scope():
            variable = variables.Variable(0.0)

        @def_function.function
        def add_one():
            if False:
                i = 10
                return i + 15
            trace_count[0] = trace_count[0] + 1
            return math_ops.add(variable, constant_op.constant(1.0))

        @def_function.function
        def update_variable():
            if False:
                while True:
                    i = 10
            for device in set(strategy.extended.worker_devices):
                with ops.device(device):
                    add_one()
        with strategy.scope():
            update_variable.get_concrete_function()
            self.assertLen(strategy.extended.worker_devices, trace_count[0])

    def test_tpu_cancellation_does_not_close_chips(self, enable_packed_var):
        if False:
            for i in range(10):
                print('nop')
        if tpu_lib.enable_batch_variable_initialization():
            self.skipTest('b/271767559')
        if not FLAGS.tpu_use_tfrt:
            self.skipTest('`tpu_cancellation_closes_chip only applies to TFRT TPU Runtime.')
        strategy = get_tpu_strategy(enable_packed_var)
        num_replicas = strategy.num_replicas_in_sync
        with strategy.scope():
            x = random_ops.random_normal((10240, 10240))
            y = random_ops.random_normal((10240, 10240))
            v = variables.Variable(array_ops.identity(x))
            dist_dataset = strategy.experimental_distribute_dataset(dataset_ops.Dataset.from_tensors(y).repeat(num_replicas).batch(num_replicas))
            dist_iterator = iter(dist_dataset)

            @def_function.function
            def train_steps(v, iterator, steps):
                if False:
                    return 10

                def step_fn(inputs):
                    if False:
                        while True:
                            i = 10
                    for val in inputs:
                        v.assign(math_ops.matmul(v, val))
                for _ in math_ops.range(steps):
                    strategy.run(step_fn, args=(next(iterator),))
            with self.assertRaises(errors.OutOfRangeError):
                train_steps(v, dist_iterator, 2)
            w = variables.Variable(array_ops.identity(x))
            dist_dataset = strategy.experimental_distribute_dataset(dataset_ops.Dataset.from_tensors(y).repeat(num_replicas).batch(num_replicas))
            dist_iterator = iter(dist_dataset)
            train_steps(w, dist_iterator, 1)

    def test_tpu_hardware_feature(self, enable_packed_var):
        if False:
            print('Hello World!')
        strategy = get_tpu_strategy(enable_packed_var)
        self.assertIsInstance(strategy.extended.tpu_hardware_feature.embedding_feature, tpu_hardware_feature.HardwareFeature.EmbeddingFeature)

    def test_get_tpu_cluster_resolver(self, enable_packed_var):
        if False:
            print('Hello World!')
        strategy = get_tpu_strategy(enable_packed_var)
        self.assertIsNotNone(strategy.cluster_resolver)

    def test_replica_order_for_distribute_datasets_from_function(self, enable_packed_var):
        if False:
            while True:
                i = 10

        def _create_dataset(strategy):
            if False:
                for i in range(10):
                    print('nop')

            def dataset_fn(ctx):
                if False:
                    for i in range(10):
                        print('nop')
                del ctx
                return dataset_ops.Dataset.range(2)
            return strategy.distribute_datasets_from_function(dataset_fn)
        values = self._test_replica_order(_create_dataset).values
        self.assertLen(values, 2)
        self.assertEqual(1, values[0].numpy())
        self.assertEqual(0, values[1].numpy())

    def test_replica_order_for_experimental_distribute_dataset(self, enable_packed_var):
        if False:
            i = 10
            return i + 15

        def _create_dataset(strategy):
            if False:
                for i in range(10):
                    print('nop')
            dataset = dataset_ops.Dataset.range(2).batch(2)
            return strategy.experimental_distribute_dataset(dataset)
        values = self._test_replica_order(_create_dataset).values
        self.assertLen(values, 2)
        self.assertEqual(1, values[0].numpy())
        self.assertEqual(0, values[1].numpy())

    def _test_replica_order(self, create_dist_dataset_fn):
        if False:
            return 10
        tf2.enable()
        resolver = get_tpu_cluster_resolver()
        remote.connect_to_cluster(resolver)
        topology = tpu_cluster_resolver.initialize_tpu_system(resolver)
        device_assignment = device_assignment_lib.DeviceAssignment(topology, core_assignment=[[[0, 0, 0, 1]], [[0, 0, 0, 0]]])
        strategy = tpu_lib.TPUStrategyV2(resolver, experimental_device_assignment=device_assignment)
        strategy.extended._enable_data_reorder = True
        dist_dataset = create_dist_dataset_fn(strategy)
        iterator = iter(dist_dataset)

        @def_function.function
        def test_iterators_order(iterator):
            if False:
                for i in range(10):
                    print('nop')
            return next(iterator)
        return test_iterators_order(iterator)

@test_util.with_eager_op_as_function
class TPUStrategyDataPrefetchTest(test.TestCase):

    def test_prefetch_to_device_default(self):
        if False:
            return 10
        strategy = get_tpu_strategy()
        dataset = dataset_ops.Dataset.range(strategy.num_replicas_in_sync * 2, output_type=dtypes.float32).batch(strategy.num_replicas_in_sync)
        dataset_item = next(iter(strategy.experimental_distribute_dataset(dataset)))
        dataset_location = tf_device.DeviceSpec.from_string(dataset_item.values[0].device)
        self.assertEqual(dataset_location.device_type, 'TPU')

    def test_prefetch_to_device_tpu(self):
        if False:
            return 10
        strategy = get_tpu_strategy()
        dataset = dataset_ops.Dataset.range(strategy.num_replicas_in_sync * 2, output_type=dtypes.float32).batch(strategy.num_replicas_in_sync)
        input_options = distribute_lib.InputOptions(experimental_fetch_to_device=True)
        dataset_item = next(iter(strategy.experimental_distribute_dataset(dataset, options=input_options)))
        dataset_location = tf_device.DeviceSpec.from_string(dataset_item.values[0].device)
        self.assertEqual(dataset_location.device_type, 'TPU')

    def test_prefetch_to_device_cpu(self):
        if False:
            print('Hello World!')
        strategy = get_tpu_strategy()
        dataset = dataset_ops.Dataset.range(strategy.num_replicas_in_sync * 2, output_type=dtypes.float32).batch(strategy.num_replicas_in_sync)
        input_options = distribute_lib.InputOptions(experimental_fetch_to_device=False)
        dataset_item = next(iter(strategy.experimental_distribute_dataset(dataset, options=input_options)))
        dataset_location = tf_device.DeviceSpec.from_string(dataset_item.values[0].device)
        self.assertEqual(dataset_location.device_type, 'CPU')

    def test_prefetch_to_device_sparse_dataset(self):
        if False:
            return 10
        strategy = get_tpu_strategy()
        dataset = dataset_ops.Dataset.from_tensors(sparse_tensor.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1, 2, 3], dense_shape=[2, 2]))
        dataset = dataset.repeat()
        dataset = dataset.batch(strategy.num_replicas_in_sync)
        with self.assertRaisesRegex(ValueError, 'TPUStrategy does not support'):
            iter(strategy.experimental_distribute_dataset(dataset))

    def test_prefetch_to_device_ragged_dataset(self):
        if False:
            i = 10
            return i + 15
        strategy = get_tpu_strategy()
        dataset = dataset_ops.Dataset.from_tensors(ragged_tensor.RaggedTensor.from_row_splits(values=[1, 2, 3], row_splits=[0, 2, 3]))
        dataset = dataset.repeat()
        dataset = dataset.batch(strategy.num_replicas_in_sync)
        with self.assertRaisesRegex(ValueError, 'TPUStrategy does not support'):
            iter(strategy.experimental_distribute_dataset(dataset))

    def test_prefetch_to_device_sparse_dataset_fn(self):
        if False:
            for i in range(10):
                print('nop')
        strategy = get_tpu_strategy()

        def dataset_fn(ctx):
            if False:
                return 10
            del ctx
            dataset = dataset_ops.Dataset.from_tensors(sparse_tensor.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1, 2, 3], dense_shape=[2, 2]))
            dataset = dataset.repeat()
            return dataset.batch(strategy.num_replicas_in_sync)
        with self.assertRaisesRegex(ValueError, 'TPUStrategy does not support'):
            iter(strategy.distribute_datasets_from_function(dataset_fn))

    def test_prefetch_to_device_ragged_dataset_fn(self):
        if False:
            i = 10
            return i + 15
        strategy = get_tpu_strategy()

        def dataset_fn(ctx):
            if False:
                for i in range(10):
                    print('nop')
            del ctx
            dataset = dataset_ops.Dataset.from_tensors(ragged_tensor.RaggedTensor.from_row_splits(values=[1, 2, 3], row_splits=[0, 2, 3]))
            dataset = dataset.repeat()
            return dataset.batch(strategy.num_replicas_in_sync)
        with self.assertRaisesRegex(ValueError, 'TPUStrategy does not support'):
            iter(strategy.distribute_datasets_from_function(dataset_fn))

    def test_create_iterator_on_device(self):
        if False:
            print('Hello World!')

        @def_function.function
        def create_iter():
            if False:
                return 10
            with ops.device('/device:TPU:0'):
                return gen_dataset_ops.anonymous_iterator_v3(output_types=[dtypes.float32], output_shapes=[[]])
        create_iter()

@test_util.with_eager_op_as_function
class TPUStrategyDistributionTest(strategy_test_lib.DistributionTestBase, strategy_test_lib.TwoDeviceDistributionTestBase):

    def test_update_config_proto(self):
        if False:
            return 10
        resolver = get_tpu_cluster_resolver()
        remote.connect_to_cluster(resolver)
        tpu_cluster_resolver.initialize_tpu_system(resolver)
        strategy = tpu_lib.TPUStrategyV2(resolver)
        config_proto = config_pb2.ConfigProto()
        cluster_spec = server_lib.ClusterSpec({'worker': ['fake1', 'fake2']})
        with test.mock.patch.object(resolver, 'cluster_spec', return_value=cluster_spec):
            new_config = strategy.update_config_proto(config_proto)
        self.assertProtoEquals(cluster_spec.as_cluster_def(), new_config.cluster_def)
        self.assertTrue(new_config.isolate_session_state)

    def test_make_input_fn_iterable(self):
        if False:
            return 10
        dataset_fn = lambda : dataset_ops.Dataset.range(10)
        expected_values = [[i, i + 1] for i in range(0, 10, 2)]
        distribution = get_tpu_strategy()
        input_fn = self._input_fn_to_test_input_context(dataset_fn, expected_num_replicas_in_sync=2, expected_num_input_pipelines=1, expected_input_pipeline_id=0)
        self._test_input_fn_iterable(distribution, input_fn, expected_values)

    def test_make_input_fn_iterator(self):
        if False:
            print('Hello World!')
        dataset_fn = lambda : dataset_ops.Dataset.range(10)
        expected_values = [[i, i + 1] for i in range(0, 10, 2)]
        distribution = get_tpu_strategy()
        input_fn = self._input_fn_to_test_input_context(dataset_fn, expected_num_replicas_in_sync=2, expected_num_input_pipelines=1, expected_input_pipeline_id=0)
        iterator = distribution.make_input_fn_iterator(input_fn)
        self._test_input_fn_iterator(iterator, distribution.extended.worker_devices, expected_values)

    def test_num_replicas_in_sync(self):
        if False:
            for i in range(10):
                print('nop')
        strategy = get_tpu_strategy()
        self.assertEqual(2, strategy.num_replicas_in_sync)

    def test_call_and_merge_exceptions(self):
        if False:
            print('Hello World!')
        strategy = get_tpu_strategy()
        self._test_call_and_merge_exceptions(strategy)

    def test_numpy_dataset(self):
        if False:
            i = 10
            return i + 15
        strategy = get_tpu_strategy()
        self._test_numpy_dataset(strategy, run_in_function=True)

    def test_global_step_update(self):
        if False:
            for i in range(10):
                print('nop')
        strategy = get_tpu_strategy()
        self._test_global_step_update(strategy)

    def test_run(self):
        if False:
            i = 10
            return i + 15
        strategy = get_tpu_strategy()
        self._test_run(strategy, run_in_function=True)

    def test_summary_for_replica_zero_only(self):
        if False:
            for i in range(10):
                print('nop')
        strategy = get_tpu_strategy()
        self._test_summary_for_replica_zero_only(strategy)

    def test_all_reduce_sum(self):
        if False:
            return 10
        strategy = get_tpu_strategy()
        self._test_all_reduce_sum(strategy, run_in_function=True)

    def test_all_reduce_sum_gradients(self):
        if False:
            i = 10
            return i + 15
        strategy = get_tpu_strategy()
        self._test_all_reduce_sum_gradients(strategy, run_in_function=True)

    def test_all_reduce_sum_gradient_tape(self):
        if False:
            while True:
                i = 10
        strategy = get_tpu_strategy()
        self._test_all_reduce_sum_gradient_tape(strategy, run_in_function=True)

    def test_all_reduce_mean(self):
        if False:
            print('Hello World!')
        strategy = get_tpu_strategy()
        self._test_all_reduce_mean(strategy, run_in_function=True)

    def test_all_reduce_mean_gradients(self):
        if False:
            i = 10
            return i + 15
        strategy = get_tpu_strategy()
        self._test_all_reduce_mean_gradients(strategy, run_in_function=True)

    def test_all_reduce_mean_gradient_tape(self):
        if False:
            for i in range(10):
                print('nop')
        strategy = get_tpu_strategy()
        self._test_all_reduce_mean_gradient_tape(strategy, run_in_function=True)

    def test_reduce(self):
        if False:
            print('Hello World!')
        strategy = get_tpu_strategy()
        inputs = strategy.make_input_fn_iterator(lambda _: dataset_ops.Dataset.from_tensor_slices([2.0, 3.0]))
        self.evaluate(inputs.initialize())
        per_replica_outputs = strategy.run(def_function.function(math_ops.square), args=(next(inputs),))
        with strategy.scope():
            mean = strategy.reduce(reduce_util.ReduceOp.MEAN, per_replica_outputs, axis=None)
            self.assertEqual(6.5, self.evaluate(mean))

    def test_constraint(self):
        if False:
            while True:
                i = 10
        strategy = get_tpu_strategy()
        with strategy.scope():
            variable = variables.Variable(initial_value=2.0, constraint=lambda x: 0.0 * x + 1.0)
        self.assertEqual(variable.value().numpy(), 2)

        @def_function.function
        def update_variable():
            if False:
                i = 10
                return i + 15
            variable.assign_add(1)
            variable.assign(variable.constraint(variable))
        update_variable()
        self.assertEqual(variable.value().numpy(), 1)

    def test_trainable_variables(self):
        if False:
            for i in range(10):
                print('nop')
        strategy = get_tpu_strategy()
        self._test_trainable_variable(strategy)

@test_util.with_eager_op_as_function
class DeviceAssignmentTest(test.TestCase):

    def test_core_assignment(self):
        if False:
            i = 10
            return i + 15
        resolver = get_tpu_cluster_resolver()
        remote.connect_to_cluster(resolver)
        topology = tpu_cluster_resolver.initialize_tpu_system(resolver)
        device_assignment = device_assignment_lib.DeviceAssignment(topology, core_assignment=[[[0, 0, 0, 0]]])
        self.assertAllEqual([[[0, 0, 0, 0]]], device_assignment.core_assignment)
        self.assertEqual(1, device_assignment.num_cores_per_replica)
        self.assertEqual(1, device_assignment.num_replicas)
        self.assertEqual('/task:0/device:TPU:0', device_assignment.tpu_device())
        self.assertEqual('/task:0/device:CPU:0', device_assignment.host_device())

    def test_device_assignment_strategy_properties(self):
        if False:
            print('Hello World!')
        resolver = get_tpu_cluster_resolver()
        remote.connect_to_cluster(resolver)
        topology = tpu_cluster_resolver.initialize_tpu_system(resolver)
        device_assignment = device_assignment_lib.DeviceAssignment(topology, core_assignment=[[[0, 0, 0, 0]]])
        strategy = tpu_lib.TPUStrategyV2(resolver, experimental_device_assignment=device_assignment)
        self.assertEqual(strategy.extended.num_hosts, 1)
        self.assertEqual(strategy.num_replicas_in_sync, 1)
        self.assertEqual(strategy.extended.num_replicas_per_host, 1)

    def test_device_assignment_constants(self):
        if False:
            while True:
                i = 10
        resolver = get_tpu_cluster_resolver()
        remote.connect_to_cluster(resolver)
        topology = tpu_cluster_resolver.initialize_tpu_system(resolver)
        device_assignment = device_assignment_lib.DeviceAssignment(topology, core_assignment=device_assignment_lib.SINGLE_CORE_ASSIGNMENT)
        self.assertAllEqual([[[0, 0, 0, 0]]], device_assignment.core_assignment)
        self.assertEqual(1, device_assignment.num_cores_per_replica)
        self.assertEqual(1, device_assignment.num_replicas)
        self.assertEqual('/task:0/device:TPU:0', device_assignment.tpu_device())
        self.assertEqual('/task:0/device:CPU:0', device_assignment.host_device())

    def test_variables_mismatched_device_assignment(self):
        if False:
            return 10
        resolver = get_tpu_cluster_resolver()
        remote.connect_to_cluster(resolver)
        topology = tpu_cluster_resolver.initialize_tpu_system(resolver)
        strategy0 = tpu_lib.TPUStrategyV2(resolver)
        self.assertEqual(('/job:localhost/replica:0/task:0/device:TPU:0', '/job:localhost/replica:0/task:0/device:TPU:1'), strategy0.extended.worker_devices)
        with strategy0.scope():
            v = variables.Variable(1.0)
        v1_assign_op = strategy0.experimental_local_results(v)[1].assign(42.0)
        with self.cached_session():
            self.evaluate(variables.global_variables_initializer())
            self.evaluate(v1_assign_op)
            self.assertAllEqual([1.0, 42.0], self.evaluate(strategy0.experimental_local_results(v)))
        device_assignment = device_assignment_lib.DeviceAssignment(topology, core_assignment=[[[0, 0, 0, 1]], [[0, 0, 0, 0]]])
        strategy1 = tpu_lib.TPUStrategyV2(resolver, experimental_device_assignment=device_assignment)
        self.assertEqual(('/job:localhost/replica:0/task:0/device:TPU:1', '/job:localhost/replica:0/task:0/device:TPU:0'), strategy1.extended.worker_devices)
        v_read = strategy1.run(def_function.function(v.read_value))
        with self.cached_session():
            self.assertAllEqual([42.0, 1.0], self.evaluate(strategy0.experimental_local_results(v_read)))

class VariableCreationTest(test.TestCase):

    def test_custom_tpu_variable_creator(self):
        if False:
            while True:
                i = 10
        strategy = get_tpu_strategy()

        def variable_creator(next_creator, **kwargs):
            if False:
                for i in range(10):
                    print('nop')

            def custom_tpu_variable_creator(next_creator, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                return next_creator(**kwargs)
            kwargs['custom_tpu_variable_creator'] = custom_tpu_variable_creator
            return next_creator(**kwargs)
        with strategy.scope():
            tpu_variable = variables.Variable(1.0)
            self.assertIsInstance(tpu_variable, tpu_values.TPUDistributedVariable)
            with variable_scope.variable_creator_scope(variable_creator):
                non_tpu_variable = variables.Variable(1.0)
                self.assertIsInstance(non_tpu_variable, variables.Variable)
if __name__ == '__main__':
    test.main()