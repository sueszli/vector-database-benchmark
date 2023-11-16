"""Tests for MirroredStrategy."""
import json
import sys
from absl.testing import parameterized
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import tf2
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util as util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.training import server_lib
from tensorflow.python.util import traceback_utils
GPU_TEST = 'test_gpu' in sys.argv[0]

@combinations.generate(combinations.combine(distribution=[strategy_combinations.mirrored_strategy_with_gpu_and_cpu, strategy_combinations.mirrored_strategy_with_two_gpus, strategy_combinations.mirrored_strategy_with_two_gpus_no_merge_call], mode=['graph', 'eager']))
class MirroredTwoDeviceDistributionTest(strategy_test_lib.DistributionTestBase, strategy_test_lib.TwoDeviceDistributionTestBase, parameterized.TestCase):

    def testMinimizeLoss(self, distribution):
        if False:
            while True:
                i = 10
        if context.executing_eagerly():
            self._test_minimize_loss_eager(distribution)
        else:
            self._test_minimize_loss_graph(distribution)

    def testReplicaId(self, distribution):
        if False:
            print('Hello World!')
        self._test_replica_id(distribution)

    def testNumReplicasInSync(self, distribution):
        if False:
            i = 10
            return i + 15
        self.assertEqual(2, distribution.num_replicas_in_sync)

    def testCallAndMergeExceptions(self, distribution):
        if False:
            return 10
        self._test_call_and_merge_exceptions(distribution)

    def testRunRegroupError(self, distribution):
        if False:
            i = 10
            return i + 15
        if not distribution.extended._use_merge_call():
            self.skipTest('Collective all-reduce does not support int32 on GPU.')

        def run_fn():
            if False:
                print('Hello World!')
            replica_id = int(self.evaluate(_replica_id()))
            return list(range(replica_id))
        with distribution.scope(), self.assertRaises(AssertionError):
            distribution.extended.call_for_each_replica(run_fn)

    def testReduceToCpu(self, distribution):
        if False:
            while True:
                i = 10
        if not distribution.extended._use_merge_call():
            self.skipTest('Collective all-reduce does not support int32 on GPU.')
        with distribution.scope():
            result = distribution.extended.call_for_each_replica(_replica_id)
            reduced = distribution.reduce(reduce_util.ReduceOp.SUM, result, axis=None)
            expected = sum(range(distribution.num_replicas_in_sync))
            self.assertEqual(expected, self.evaluate(reduced))

    def testReduceToCpuNested(self, distribution):
        if False:
            return 10
        if not distribution.extended._use_merge_call():
            self.skipTest('Collective all-reduce does not support int32 on GPU.')
        with distribution.scope():

            def replica_fn(input_tensor):
                if False:
                    while True:
                        i = 10
                return (input_tensor + constant_op.constant(1.0), input_tensor - constant_op.constant(1.0))
            input_tensor = constant_op.constant(3.0)
            run_result = distribution.run(replica_fn, args=(input_tensor,))
            reduced_result = distribution.reduce('SUM', run_result, axis=None)
            expected_result = (4 * distribution.num_replicas_in_sync, 2 * distribution.num_replicas_in_sync)
            self.assertEqual(expected_result, self.evaluate(reduced_result))

    def reduce_axis_helper(self, distribution, replica_squared_fn):
        if False:
            i = 10
            return i + 15
        with distribution.scope():
            num_replicas = distribution.num_replicas_in_sync
            result = distribution.extended.call_for_each_replica(replica_squared_fn)
            reduced = distribution.reduce(reduce_util.ReduceOp.SUM, result, axis=0)
            expected = sum((x * (x + 1) for x in range(num_replicas)))
            self.assertNear(expected, self.evaluate(reduced), 1e-05)
            reduced = distribution.reduce(reduce_util.ReduceOp.MEAN, result, axis=0)
            expected /= sum((x + 1 for x in range(num_replicas)))
            self.assertNear(expected, self.evaluate(reduced), 1e-05)

    def testReduceAxisToCpu(self, distribution):
        if False:
            while True:
                i = 10
        if not distribution.extended._use_merge_call():
            self.skipTest('Collective all-reduce does not support int32 on GPU.')
        for dtype in (dtypes.float32, dtypes.int32):

            def replica_squared_fn(dtype=dtype):
                if False:
                    return 10
                replica_id = _replica_id_as_int()
                return array_ops.identity(math_ops.cast([replica_id] * (replica_id + 1), dtype))
            self.reduce_axis_helper(distribution, replica_squared_fn)

    def set_v2_tensorshape(self, v2):
        if False:
            while True:
                i = 10
        if v2:
            tensor_shape.enable_v2_tensorshape()
        else:
            tensor_shape.disable_v2_tensorshape()

    def testReduceAxisToCpuUnknownShape(self, distribution):
        if False:
            print('Hello World!')
        if not distribution.extended._use_merge_call():
            self.skipTest('Collective all-reduce does not support int32 on GPU.')
        original_v2 = tensor_shape._TENSORSHAPE_V2_OVERRIDE
        try:
            for v2 in (False, True):
                self.set_v2_tensorshape(v2)
                for dtype in (dtypes.float32, dtypes.int32):
                    for shape in ((None,), None):

                        def replica_squared_fn(dtype=dtype, shape=shape):
                            if False:
                                for i in range(10):
                                    print('nop')
                            replica_id = _replica_id_as_int()
                            tensor = math_ops.cast([replica_id] * (replica_id + 1), dtype)
                            return array_ops.placeholder_with_default(tensor, shape=shape)
                        self.reduce_axis_helper(distribution, replica_squared_fn)
        finally:
            self.set_v2_tensorshape(original_v2)

    def testReplicateDataset(self, distribution):
        if False:
            i = 10
            return i + 15
        if tf2.enabled() and (not context.executing_eagerly()):
            self.skipTest('Skipping test since we do not support graph mode in TF 2')
        dataset_fn = lambda : dataset_ops.Dataset.range(10)
        expected_values = [[i, i + 1] for i in range(0, 10, 2)]
        input_fn = self._input_fn_to_test_input_context(dataset_fn, expected_num_replicas_in_sync=2, expected_num_input_pipelines=1, expected_input_pipeline_id=0)
        self._test_input_fn_iterable(distribution, input_fn, expected_values)

    def testMakeInputFnIteratorWithDataset(self, distribution):
        if False:
            return 10
        dataset_fn = lambda : dataset_ops.Dataset.range(10)
        expected_values = [[i, i + 1] for i in range(0, 10, 2)]
        input_fn = self._input_fn_to_test_input_context(dataset_fn, expected_num_replicas_in_sync=2, expected_num_input_pipelines=1, expected_input_pipeline_id=0)
        iterator = distribution.make_input_fn_iterator(input_fn)
        self._test_input_fn_iterator(iterator, distribution.extended.worker_devices, expected_values)

    def testMakeInputFnIteratorWithCallable(self, distribution):
        if False:
            for i in range(10):
                print('nop')

        def fn():
            if False:
                for i in range(10):
                    print('nop')
            dataset = dataset_ops.Dataset.range(2).interleave(lambda _: dataset_ops.Dataset.range(10), cycle_length=2)
            it = dataset_ops.make_one_shot_iterator(dataset)
            return it.get_next
        expected_values = [[i, i] for i in range(0, 10)]
        input_fn = self._input_fn_to_test_input_context(fn, expected_num_replicas_in_sync=2, expected_num_input_pipelines=1, expected_input_pipeline_id=0)
        iterator = distribution.make_input_fn_iterator(input_fn)
        self._test_input_fn_iterator(iterator, distribution.extended.worker_devices, expected_values, test_reinitialize=False, ignore_order=True)

    def testNumpyDataset(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        self._test_numpy_dataset(distribution)

    def testGlobalStepUpdate(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        self._test_global_step_update(distribution)

    def testRun(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        self._test_run(distribution)

    def testAllReduceSum(self, distribution):
        if False:
            i = 10
            return i + 15
        self._test_all_reduce_sum(distribution)

    def testAllReduceSumGradients(self, distribution):
        if False:
            print('Hello World!')
        self._test_all_reduce_sum_gradients(distribution)

    def testAllReduceSumGradientTape(self, distribution):
        if False:
            print('Hello World!')
        self._test_all_reduce_sum_gradient_tape(distribution)

    def testAllReduceMean(self, distribution):
        if False:
            return 10
        self._test_all_reduce_mean(distribution)

    def testAllReduceMeanGradients(self, distribution):
        if False:
            i = 10
            return i + 15
        self._test_all_reduce_mean_gradients(distribution)

    def testAllReduceMeanGradientTape(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        self._test_all_reduce_mean_gradient_tape(distribution)

    def testSummaryForReplicaZeroOnly(self, distribution):
        if False:
            i = 10
            return i + 15
        self._test_summary_for_replica_zero_only(distribution)

    def testTrainableVariables(self, distribution):
        if False:
            i = 10
            return i + 15
        self._test_trainable_variable(distribution)

    def test_prefetch_to_device_dataset(self, distribution):
        if False:
            print('Hello World!')
        input_options = distribute_lib.InputOptions(experimental_fetch_to_device=True)
        dataset = dataset_ops.Dataset.range(100)
        dataset = dataset.batch(distribution.num_replicas_in_sync)
        dataset = distribution.experimental_distribute_dataset(dataset, options=input_options)
        if context.executing_eagerly():
            item = next(iter(dataset))
        elif isinstance(dataset, input_lib_v1.DistributedDatasetV1):
            item = dataset.make_initializable_iterator().get_next()
        else:
            self.skipTest('unsupported test combination')
        device_types = [tf_device.DeviceSpec.from_string(tensor.device).device_type for tensor in item.values]
        expected_device_types = [tf_device.DeviceSpec.from_string(device).device_type for device in distribution.extended.worker_devices]
        self.assertAllEqual(device_types, expected_device_types)

    def test_prefetch_to_host_dataset(self, distribution):
        if False:
            print('Hello World!')
        input_options = distribute_lib.InputOptions(experimental_fetch_to_device=False)
        dataset = dataset_ops.Dataset.range(100)
        dataset = dataset.batch(distribution.num_replicas_in_sync)
        dataset = distribution.experimental_distribute_dataset(dataset, options=input_options)
        if context.executing_eagerly():
            item = next(iter(dataset))
        elif isinstance(dataset, input_lib_v1.DistributedDatasetV1):
            item = dataset.make_initializable_iterator().get_next()
        else:
            self.skipTest('unsupported test combination')
        device_types = {tf_device.DeviceSpec.from_string(tensor.device).device_type for tensor in item.values}
        self.assertAllEqual(list(device_types), ['CPU'])

@combinations.generate(combinations.combine(mode=['eager', 'graph'], required_gpus=[2]))
class MirroredCollectiveOpTest(strategy_test_lib.DistributionTestBase, strategy_test_lib.TwoDeviceDistributionTestBase, parameterized.TestCase):

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        super(MirroredCollectiveOpTest, self).tearDown()
        context._reset_context()

    def testAllCpu(self):
        if False:
            print('Hello World!')

        @def_function.function
        def fn():
            if False:
                i = 10
                return i + 15
            strategy = mirrored_strategy.MirroredStrategy(['CPU:0', 'CPU:1'])
            if ops.executing_eagerly_outside_functions():
                self.assertIsInstance(strategy.extended._collective_ops, cross_device_ops_lib.CollectiveAllReduce)
                self.assertEqual(strategy.extended._collective_ops._options.implementation, collective_util.CommunicationImplementation.RING)
            else:
                self.assertIsInstance(strategy.extended._collective_ops, cross_device_ops_lib.ReductionToOneDevice)
        fn()

    def testMixedDevices(self):
        if False:
            i = 10
            return i + 15

        @def_function.function
        def fn():
            if False:
                for i in range(10):
                    print('nop')
            strategy = mirrored_strategy.MirroredStrategy(['CPU:0', 'GPU:0'])
            self.assertIsInstance(strategy.extended._collective_ops, cross_device_ops_lib.ReductionToOneDevice)
        fn()

    def testAllPhysicalGpu(self):
        if False:
            print('Hello World!')

        @def_function.function
        def fn():
            if False:
                print('Hello World!')
            strategy = mirrored_strategy.MirroredStrategy(['GPU:0', 'GPU:1'])
            self.assertIsInstance(strategy.extended._collective_ops, cross_device_ops_lib.CollectiveAllReduce)
            self.assertEqual(strategy.extended._collective_ops._options.implementation, collective_util.CommunicationImplementation.NCCL)
        fn()

    def testVirtualGpu(self):
        if False:
            print('Hello World!')
        context._reset_context()
        physical_gpus = context.context().list_physical_devices(device_type='GPU')
        context.context().set_logical_device_configuration(physical_gpus[1], [context.LogicalDeviceConfiguration(memory_limit=1024), context.LogicalDeviceConfiguration(memory_limit=1024)])

        @def_function.function
        def fn():
            if False:
                i = 10
                return i + 15
            strategy = mirrored_strategy.MirroredStrategy(['GPU:0', 'GPU:1', 'GPU:2'])
            if ops.executing_eagerly_outside_functions():
                self.assertIsInstance(strategy.extended._collective_ops, cross_device_ops_lib.CollectiveAllReduce)
                self.assertEqual(strategy.extended._collective_ops._options.implementation, collective_util.CommunicationImplementation.RING)
            else:
                self.assertEqual(strategy.extended._collective_ops, cross_device_ops_lib.ReductionToOneDevice)
        fn()

@combinations.generate(combinations.combine(mode=['graph', 'eager'], required_gpus=[2], use_default=[True, False]))
class MirroredGetCrossDeviceOpTest(strategy_test_lib.DistributionTestBase, strategy_test_lib.TwoDeviceDistributionTestBase, parameterized.TestCase):

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        super().tearDown()
        context._reset_context()

    def testGpusCollectiveOp(self, use_default):
        if False:
            i = 10
            return i + 15

        @def_function.function(jit_compile=util.is_xla_enabled())
        def fn(var, use_default):
            if False:
                for i in range(10):
                    print('nop')
            if use_default or util.is_xla_enabled():
                self.assertIsInstance(strategy.extended._get_cross_device_ops(var), cross_device_ops_lib.CollectiveAllReduce)
            else:
                self.assertIsInstance(strategy.extended._get_cross_device_ops(var), cross_device_ops_lib.NcclAllReduce)
        strategy = mirrored_strategy.MirroredStrategy(['GPU:0', 'GPU:1'], cross_device_ops=None if use_default else cross_device_ops_lib.NcclAllReduce())
        with strategy.scope():
            var = variables.Variable(1.0)
        fn(var, use_default)

    def testVirtualGpusCollectiveOp(self, use_default):
        if False:
            i = 10
            return i + 15
        context._reset_context()
        physical_gpus = context.context().list_physical_devices(device_type='GPU')
        context.context().set_logical_device_configuration(physical_gpus[1], [context.LogicalDeviceConfiguration(memory_limit=1024), context.LogicalDeviceConfiguration(memory_limit=1024)])

        @def_function.function(jit_compile=util.is_xla_enabled())
        def fn(var, use_default):
            if False:
                return 10
            if use_default or util.is_xla_enabled():
                self.assertIsInstance(strategy.extended._get_cross_device_ops(var), cross_device_ops_lib.CollectiveAllReduce)
                self.assertEqual(strategy.extended._get_cross_device_ops(var)._options.implementation, collective_util.CommunicationImplementation.RING)
            else:
                self.assertIsInstance(strategy.extended._get_cross_device_ops(var), cross_device_ops_lib.NcclAllReduce)
        strategy = mirrored_strategy.MirroredStrategy(['GPU:0', 'GPU:1', 'GPU:2'], cross_device_ops=None if use_default else cross_device_ops_lib.NcclAllReduce())
        with strategy.scope():
            var = variables.Variable(1.0)
        fn(var, use_default)

    def testCpusCollectiveOp(self, use_default):
        if False:
            print('Hello World!')
        del use_default
        if util.is_xla_enabled():
            self.skipTest('Only expected to run under non-XLA context.')

        @def_function.function(jit_compile=True)
        def fn(var):
            if False:
                for i in range(10):
                    print('nop')
            if not ops.executing_eagerly_outside_functions():
                self.assertIsInstance(strategy.extended._get_cross_device_ops(var), cross_device_ops_lib.ReductionToOneDevice)
            else:
                self.assertIsInstance(strategy.extended._get_cross_device_ops(var), cross_device_ops_lib.CollectiveAllReduce)
        strategy = mirrored_strategy.MirroredStrategy(['CPU:0', 'CPU:1'])
        with strategy.scope():
            var = variables.Variable(1.0)
        fn(var)

    def testMixedDevicesCollectiveOp(self, use_default):
        if False:
            print('Hello World!')
        del use_default
        if util.is_xla_enabled():
            self.skipTest('All devices should be identical in XLA context.')
        strategy = mirrored_strategy.MirroredStrategy(['CPU:0', 'GPU:0'])
        with strategy.scope():
            var = variables.Variable(1.0)
        self.assertIsInstance(strategy.extended._get_cross_device_ops(var), cross_device_ops_lib.ReductionToOneDevice)

    def testMirroredStrategyInt32VariableCollectiveOp(self, use_default):
        if False:
            i = 10
            return i + 15
        if util.is_xla_enabled():
            self.skipTest('Only expected to run under non-XLA context.')
        strategy = mirrored_strategy.MirroredStrategy(['GPU:0', 'GPU:1'], cross_device_ops=None if use_default else cross_device_ops_lib.NcclAllReduce())
        with strategy.scope():
            var = variables.Variable(1)
        self.assertIsInstance(strategy.extended._get_cross_device_ops(var), cross_device_ops_lib.ReductionToOneDevice)

def one_device_combinations():
    if False:
        for i in range(10):
            print('nop')
    return combinations.combine(distribution=[strategy_combinations.mirrored_strategy_with_one_cpu, strategy_combinations.mirrored_strategy_with_one_gpu], mode=['graph', 'eager'])

@combinations.generate(one_device_combinations())
class MirroredOneDeviceDistributionTest(strategy_test_lib.DistributionTestBase, strategy_test_lib.OneDeviceDistributionTestBase, parameterized.TestCase):

    def testMinimizeLoss(self, distribution):
        if False:
            print('Hello World!')
        if context.executing_eagerly():
            self._test_minimize_loss_eager(distribution)
        else:
            self._test_minimize_loss_graph(distribution)

    def testReplicaId(self, distribution):
        if False:
            return 10
        self._test_replica_id(distribution)

    def testCallAndMergeExceptions(self, distribution):
        if False:
            while True:
                i = 10
        self._test_call_and_merge_exceptions(distribution)

    def testRun(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        self._test_run(distribution)

    def testAllReduceSum(self, distribution):
        if False:
            while True:
                i = 10
        self._test_all_reduce_sum(distribution)

    def testAllReduceSumGradients(self, distribution):
        if False:
            while True:
                i = 10
        self._test_all_reduce_sum_gradients(distribution)

    def testAllReduceSumGradientTape(self, distribution):
        if False:
            i = 10
            return i + 15
        self._test_all_reduce_sum_gradient_tape(distribution)

    def testAllReduceMean(self, distribution):
        if False:
            print('Hello World!')
        self._test_all_reduce_mean(distribution)

    def testAllReduceMeanGradients(self, distribution):
        if False:
            return 10
        self._test_all_reduce_mean_gradients(distribution)

    def testAllReduceMeanGradientTape(self, distribution):
        if False:
            while True:
                i = 10
        self._test_all_reduce_mean_gradient_tape(distribution)

class MirroredStrategyVariableCreatorStackTest(test.TestCase, parameterized.TestCase):

    @combinations.generate(combinations.combine(distribution=[strategy_combinations.mirrored_strategy_with_gpu_and_cpu], mode=['graph']))
    def testCreatorStacksAreThreadLocal(self, distribution):
        if False:
            print('Hello World!')

        def model_fn():
            if False:
                while True:
                    i = 10
            replica_id_str = str(self.evaluate(_replica_id()))

            def thread_creator_fn(next_creator, **kwargs):
                if False:
                    print('Hello World!')
                return next_creator(**kwargs) + ':thread_' + replica_id_str
            with variable_scope.variable_creator_scope(thread_creator_fn):
                v = variable_v1.VariableV1(1.0)
                distribute_lib.get_replica_context().merge_call(lambda _: _)
            return v

        def main_thread_creator(next_creator, **kwargs):
            if False:
                i = 10
                return i + 15
            del next_creator, kwargs
            return 'main_thread'
        with context.graph_mode(), distribution.scope(), variable_scope.variable_creator_scope(main_thread_creator):
            result = distribution.extended.call_for_each_replica(model_fn)
            result = distribution.experimental_local_results(result)
            expected = ('main_thread:thread_0', 'main_thread:thread_1')
            self.assertEqual(expected, result)

@combinations.generate(combinations.combine(distribution=[strategy_combinations.mirrored_strategy_with_gpu_and_cpu], mode=['graph', 'eager']))
class MirroredStrategyCallForEachReplicaTest(test.TestCase):

    def testExecutingEagerlyOutsideFunction(self, distribution):
        if False:
            i = 10
            return i + 15
        'Verify we preserve the value of executing_eagerly_outside_functions().'

        def model_fn():
            if False:
                print('Hello World!')
            return ops.executing_eagerly_outside_functions()
        originally = ops.executing_eagerly_outside_functions()
        with distribution.scope():
            in_scope = ops.executing_eagerly_outside_functions()
            in_model_fn = distribution.extended.call_for_each_replica(model_fn)
            unwrapped = distribution.experimental_local_results(in_model_fn)
            self.assertEqual(in_scope, unwrapped[0])
            self.assertEqual(in_scope, originally)
        with func_graph.FuncGraph('fg').as_default(), distribution.scope():
            in_scope = ops.executing_eagerly_outside_functions()
            in_model_fn = distribution.extended.call_for_each_replica(model_fn)
            unwrapped = distribution.experimental_local_results(in_model_fn)
            self.assertEqual(in_scope, unwrapped[0])
            self.assertEqual(in_scope, originally)

    def testFunctionInCallForEachReplica(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        traces = []

        @def_function.function
        def model_fn():
            if False:
                print('Hello World!')
            traces.append(1)
            return distribute_lib.get_replica_context().replica_id_in_sync_group
        with distribution.scope():
            result = distribution.extended.call_for_each_replica(model_fn)
            self.assertEqual((0, 1), self.evaluate(distribution.experimental_local_results(result)))
            self.assertLen(traces, distribution.num_replicas_in_sync)

    def testFunctionInCallForEachReplicaInsideAnotherFunction(self, distribution):
        if False:
            print('Hello World!')
        traces = []

        @def_function.function
        def model_fn():
            if False:
                while True:
                    i = 10
            traces.append(1)
            return distribute_lib.get_replica_context().replica_id_in_sync_group

        @def_function.function
        def step():
            if False:
                while True:
                    i = 10
            return distribution.extended.call_for_each_replica(model_fn)
        with distribution.scope():
            result = step()
            self.assertEqual((0, 1), self.evaluate(distribution.experimental_local_results(result)))
            self.assertLen(traces, distribution.num_replicas_in_sync)

    def testControlFlowFunctionInCallForEachReplicaWithMergeCall(self, distribution):
        if False:
            print('Hello World!')

        def merge_fn(strategy, value):
            if False:
                i = 10
                return i + 15
            return strategy.reduce(reduce_util.ReduceOp.SUM, value, axis=None)

        @def_function.function
        def model_fn():
            if False:
                for i in range(10):
                    print('nop')

            def body_fn(i):
                if False:
                    print('Hello World!')
                return distribute_lib.get_replica_context().merge_call(merge_fn, args=(i,))
            return while_loop.while_loop_v2(lambda i: i < 2, body_fn, [0])
        with distribution.scope():
            with self.assertRaisesRegex(RuntimeError, '`merge_call` called while defining a new graph.'):
                distribution.extended.call_for_each_replica(model_fn)

    def testNestedFunctionInCallForEachReplicaWithMergeCall(self, distribution):
        if False:
            i = 10
            return i + 15

        def merge_fn(strategy, value):
            if False:
                for i in range(10):
                    print('nop')
            return strategy.reduce(reduce_util.ReduceOp.SUM, value, axis=None)

        def model_fn():
            if False:
                return 10

            @def_function.function
            def model_fn_nested():
                if False:
                    i = 10
                    return i + 15
                t = constant_op.constant(1)
                return distribute_lib.get_replica_context().merge_call(merge_fn, args=(t,))
            return model_fn_nested()
        with distribution.scope():
            with self.assertRaisesRegex(RuntimeError, '`merge_call` called while defining a new graph.'):
                distribution.extended.call_for_each_replica(model_fn)

    def testFunctionInCallForEachReplicaWithMergeCall(self, distribution):
        if False:
            print('Hello World!')

        def merge_fn(_):
            if False:
                i = 10
                return i + 15
            pass

        @def_function.function
        def model_fn():
            if False:
                print('Hello World!')
            distribute_lib.get_replica_context().merge_call(merge_fn)
            return 0.0
        with distribution.scope():
            self.assertEqual(self.evaluate(distribution.extended.call_for_each_replica(model_fn)), 0.0)

    def testFunctionInCallForEachReplicaCached(self, distribution):
        if False:
            print('Hello World!')
        traces = []

        @def_function.function
        def model_fn():
            if False:
                i = 10
                return i + 15
            traces.append(None)
        self.assertEmpty(traces)
        for i in range(10):
            distribution.extended.call_for_each_replica(model_fn)
            if i == 0:
                num_devices = len(traces)
                self.assertGreater(num_devices, 0)
            else:
                self.assertLen(traces, num_devices)

@combinations.generate(combinations.combine(distribution=[strategy_combinations.mirrored_strategy_with_gpu_and_cpu], mode=['graph']))
class MirroredStrategyNameScopeTest(test.TestCase):

    def testNameScope(self, distribution):
        if False:
            return 10

        def model_fn():
            if False:
                return 10
            with ops.name_scope('foo'):
                a = constant_op.constant(1.0, name='a')
                distribute_lib.get_replica_context().merge_call(lambda _: _)
                b = constant_op.constant(1.0, name='b')
            return (a, b)
        with context.graph_mode(), distribution.scope():
            with ops.name_scope('main'):
                result = distribution.extended.call_for_each_replica(model_fn)
                self.assertEqual(2, len(result))
                for (v, name) in zip(result, ['a', 'b']):
                    self.assertIsInstance(v, values.DistributedValues)
                    (v0, v1) = distribution.experimental_local_results(v)
                    self.assertEqual('main/foo/' + name + ':0', v0.name)
                    self.assertEqual('main/replica_1/foo/' + name + ':0', v1.name)

    def testWithDefaultName(self, distribution):
        if False:
            return 10

        def model_fn():
            if False:
                i = 10
                return i + 15
            with ops.name_scope(None, 'foo'):
                a = constant_op.constant(1.0, name='a')
                distribute_lib.get_replica_context().merge_call(lambda _: _)
                b = constant_op.constant(2.0, name='b')
            return (a, b)
        with context.graph_mode(), distribution.scope():
            result = distribution.extended.call_for_each_replica(model_fn)
            self.assertEqual(2, len(result))
            for (v, name) in zip(result, ['a', 'b']):
                self.assertIsInstance(v, values.DistributedValues)
                (v0, v1) = distribution.experimental_local_results(v)
                self.assertEqual('foo/' + name + ':0', v0.name)
                self.assertEqual('replica_1/foo/' + name + ':0', v1.name)

    def testNameScopeWithVariable(self, distribution):
        if False:
            for i in range(10):
                print('nop')

        def in_cross_replica(_):
            if False:
                i = 10
                return i + 15
            c = variable_v1.VariableV1(1.0, name='c')
            return c

        def model_fn():
            if False:
                while True:
                    i = 10
            b = variable_v1.VariableV1(1.0, name='b')
            with ops.name_scope('foo'):
                c = distribute_lib.get_replica_context().merge_call(in_cross_replica)
            return (b, c)
        with context.graph_mode(), distribution.scope():
            with ops.name_scope('main'):
                a = variable_v1.VariableV1(1.0, name='a')
                result = distribution.extended.call_for_each_replica(model_fn)
            result_b = result[0]
            result_c = result[1]
            self.assertIsInstance(result_b, values.DistributedValues)
            self.assertIsInstance(result_c, values.DistributedValues)
            (a0, a1) = distribution.experimental_local_results(a)
            (b0, b1) = distribution.experimental_local_results(result_b)
            (c0, c1) = distribution.experimental_local_results(result_c)
            self.assertEqual('main/a:0', a0.name)
            self.assertEqual('main/a/replica_1:0', a1.name)
            self.assertEqual('main/b:0', b0.name)
            self.assertEqual('main/b/replica_1:0', b1.name)
            self.assertEqual('main/foo/c:0', c0.name)
            self.assertEqual('main/foo/c/replica_1:0', c1.name)

    def testNameScopeWithGetVariable(self, distribution):
        if False:
            print('Hello World!')

        def in_cross_replica(_):
            if False:
                while True:
                    i = 10
            c = variable_scope.get_variable('c', [1])
            return c

        def model_fn():
            if False:
                print('Hello World!')
            b = variable_scope.get_variable('b', [1])
            with ops.name_scope('foo'):
                c = distribute_lib.get_replica_context().merge_call(in_cross_replica)
            return (b, c)
        with context.graph_mode(), distribution.scope():
            with ops.name_scope('main'):
                a = variable_scope.get_variable('a', [1])
                result = distribution.extended.call_for_each_replica(model_fn)
            result_b = result[0]
            result_c = result[1]
            self.assertIsInstance(result_b, values.DistributedValues)
            self.assertIsInstance(result_c, values.DistributedValues)
            (a0, a1) = distribution.experimental_local_results(a)
            (b0, b1) = distribution.experimental_local_results(result_b)
            (c0, c1) = distribution.experimental_local_results(result_c)
            self.assertEqual('a:0', a0.name)
            self.assertEqual('a/replica_1:0', a1.name)
            self.assertEqual('b:0', b0.name)
            self.assertEqual('b/replica_1:0', b1.name)
            self.assertEqual('c:0', c0.name)
            self.assertEqual('c/replica_1:0', c1.name)

    def testVariableScopeWithGetVariable(self, distribution):
        if False:
            for i in range(10):
                print('nop')

        def in_cross_replica(_):
            if False:
                while True:
                    i = 10
            c = variable_scope.get_variable('c', [1])
            return c

        def model_fn():
            if False:
                for i in range(10):
                    print('nop')
            b = variable_scope.get_variable('b', [1])
            with variable_scope.variable_scope('foo'):
                c = distribute_lib.get_replica_context().merge_call(in_cross_replica)
            return (b, c)
        with context.graph_mode(), distribution.scope():
            with variable_scope.variable_scope('main'):
                a = variable_scope.get_variable('a', [1])
                result = distribution.extended.call_for_each_replica(model_fn)
            result_b = result[0]
            result_c = result[1]
            self.assertIsInstance(result_b, values.DistributedValues)
            self.assertIsInstance(result_c, values.DistributedValues)
            (a0, a1) = distribution.experimental_local_results(a)
            (b0, b1) = distribution.experimental_local_results(result_b)
            (c0, c1) = distribution.experimental_local_results(result_c)
            self.assertEqual('main/a:0', a0.name)
            self.assertEqual('main/a/replica_1:0', a1.name)
            self.assertEqual('main/b:0', b0.name)
            self.assertEqual('main/b/replica_1:0', b1.name)
            self.assertEqual('main/foo/c:0', c0.name)
            self.assertEqual('main/foo/c/replica_1:0', c1.name)

@combinations.generate(combinations.combine(distribution=[combinations.NamedDistribution('Mirrored3Devices', lambda : mirrored_strategy.MirroredStrategy(['/device:GPU:0', '/device:GPU:1', '/device:CPU:0']), required_gpus=2)], mode=['graph', 'eager']))
class MirroredThreeDeviceDistributionTest(strategy_test_lib.DistributionTestBase, parameterized.TestCase):

    def testThreeDevices(self, distribution):
        if False:
            print('Hello World!')

        def model_fn():
            if False:
                while True:
                    i = 10
            v = variable_v1.VariableV1(1.0, name='foo')
            distribute_lib.get_replica_context().merge_call(lambda _: _)
            return v
        with distribution.scope():
            result = distribution.extended.call_for_each_replica(model_fn)
            self.assertTrue(distribute_utils.is_mirrored(result))
            self.assertEqual('foo:0', result.name)

@combinations.generate(combinations.combine(distribution=[strategy_combinations.mirrored_strategy_with_gpu_and_cpu], mode=['graph', 'eager']))
class MirroredVariableUpdateTest(test.TestCase):

    def testAssignMirroredVarReplicaContextWithoutAggregationType(self, distribution):
        if False:
            while True:
                i = 10

        def var_fn():
            if False:
                return 10
            v = variable_v1.VariableV1(1.0, name='foo')
            return v
        with distribution.scope():
            mirrored_var = distribution.extended.call_for_each_replica(var_fn)
            self.assertTrue(distribute_utils.is_mirrored(mirrored_var))
            self.evaluate(variables.global_variables_initializer())

            def model_fn():
                if False:
                    return 10
                return mirrored_var.assign(5.0)
            self.evaluate(distribution.experimental_local_results(distribution.extended.call_for_each_replica(model_fn)))
            self.assertEqual(5.0, self.evaluate(mirrored_var))

    def testAssignMirroredVarReplicaContextWithSum(self, distribution):
        if False:
            return 10

        def var_fn():
            if False:
                for i in range(10):
                    print('nop')
            v = variable_v1.VariableV1(1.0, name='foo', aggregation=variable_scope.VariableAggregation.SUM)
            return v
        with distribution.scope():
            mirrored_var = distribution.extended.call_for_each_replica(var_fn)
            self.assertTrue(distribute_utils.is_mirrored(mirrored_var))
            self.evaluate(variables.global_variables_initializer())

            def model_fn():
                if False:
                    while True:
                        i = 10
                return mirrored_var.assign(5.0)
            if distribution.extended._use_merge_call():
                with self.assertRaisesRegex(ValueError, 'A non-DistributedValues value 5.0 cannot be reduced with the given reduce op ReduceOp.SUM.'):
                    self.evaluate(distribution.experimental_local_results(distribution.extended.call_for_each_replica(model_fn)))
            else:
                result = self.evaluate(distribution.experimental_local_results(distribution.extended.call_for_each_replica(model_fn)))
                self.assertAllEqual(result[0], 5.0)

    def testAssignMirroredVarCrossDeviceContext(self, distribution):
        if False:
            for i in range(10):
                print('nop')

        def var_fn():
            if False:
                while True:
                    i = 10
            return variable_v1.VariableV1(1.0, name='foo')
        with distribution.scope():
            mirrored_var = distribution.extended.call_for_each_replica(var_fn)
            self.assertTrue(distribute_utils.is_mirrored(mirrored_var))
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual(1.0, self.evaluate(mirrored_var))
            mirrored_var_result = self.evaluate(mirrored_var.assign(6.0))
            self.assertEqual(6.0, mirrored_var_result)

    def testAssignMirroredVarReplicaContext(self, distribution):
        if False:
            for i in range(10):
                print('nop')

        def var_fn():
            if False:
                i = 10
                return i + 15
            return variable_v1.VariableV1(1.0, name='foo', aggregation=variable_scope.VariableAggregation.MEAN)
        with distribution.scope():
            mirrored_var = distribution.extended.call_for_each_replica(var_fn)
            self.assertTrue(distribute_utils.is_mirrored(mirrored_var))
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual(1.0, self.evaluate(mirrored_var))

            def model_fn():
                if False:
                    for i in range(10):
                        print('nop')
                value = math_ops.cast(distribute_lib.get_replica_context().replica_id_in_sync_group, mirrored_var.dtype)
                return mirrored_var.assign(value)
            self.evaluate(distribution.experimental_local_results(distribution.extended.call_for_each_replica(model_fn)))
            self.assertEqual(0.5, self.evaluate(mirrored_var))

    def testAssignMirroredVarReplicaContextWithSingleValue(self, distribution):
        if False:
            while True:
                i = 10

        def var_fn():
            if False:
                for i in range(10):
                    print('nop')
            return variable_v1.VariableV1(1.0, name='foo', aggregation=variable_scope.VariableAggregation.MEAN)
        with distribution.scope():
            mirrored_var = distribution.extended.call_for_each_replica(var_fn)
            self.assertTrue(distribute_utils.is_mirrored(mirrored_var))
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual(1.0, self.evaluate(mirrored_var))

            def model_fn():
                if False:
                    i = 10
                    return i + 15
                return mirrored_var.assign(5.0)
            self.evaluate(distribution.experimental_local_results(distribution.extended.call_for_each_replica(model_fn)))
            self.assertEqual(5.0, self.evaluate(mirrored_var))

    def testAssignAddMirroredVarCrossDeviceContext(self, distribution):
        if False:
            while True:
                i = 10

        def var_fn():
            if False:
                print('Hello World!')
            return variable_v1.VariableV1(1.0, name='foo')
        with distribution.scope():
            mirrored_var = distribution.extended.call_for_each_replica(var_fn)
            self.assertTrue(distribute_utils.is_mirrored(mirrored_var))
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual(1.0, self.evaluate(mirrored_var))
            mirrored_var_result = self.evaluate(mirrored_var.assign_add(6.0, read_value=True))
            self.assertEqual(7.0, mirrored_var_result)
            self.assertEqual(7.0, self.evaluate(distribution.experimental_local_results(mirrored_var)[0]))
            self.assertEqual(7.0, self.evaluate(distribution.experimental_local_results(mirrored_var)[1]))
            self.assertEqual(distribution.extended.worker_devices[0], mirrored_var._devices[0])
            self.assertEqual(distribution.extended.worker_devices[1], mirrored_var._devices[1])
            self.evaluate(mirrored_var.assign_add(2.0, read_value=False))
            self.assertEqual(9.0, self.evaluate(distribution.experimental_local_results(mirrored_var)[0]))
            self.assertEqual(9.0, self.evaluate(distribution.experimental_local_results(mirrored_var)[1]))
            self.assertEqual(distribution.extended.worker_devices[0], mirrored_var._devices[0])
            self.assertEqual(distribution.extended.worker_devices[1], mirrored_var._devices[1])

    def testAssignAddMirroredVarReplicaContext(self, distribution):
        if False:
            i = 10
            return i + 15

        def var_fn():
            if False:
                print('Hello World!')
            return variable_v1.VariableV1(1.0, name='foo', aggregation=variable_scope.VariableAggregation.MEAN)
        with distribution.scope():
            mirrored_var = distribution.extended.call_for_each_replica(var_fn)
            self.assertTrue(distribute_utils.is_mirrored(mirrored_var))
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual(1.0, self.evaluate(mirrored_var))

            def model_fn():
                if False:
                    i = 10
                    return i + 15
                value = math_ops.cast(distribute_lib.get_replica_context().replica_id_in_sync_group, mirrored_var.dtype)
                return mirrored_var.assign_add(value)
            self.evaluate(distribution.experimental_local_results(distribution.extended.call_for_each_replica(model_fn)))
            self.assertEqual(1.5, self.evaluate(mirrored_var))

    def testAssignAddMirroredVarReplicaContextWithSingleValue(self, distribution):
        if False:
            print('Hello World!')

        def var_fn():
            if False:
                return 10
            return variable_v1.VariableV1(1.0, name='foo', aggregation=variable_scope.VariableAggregation.MEAN)
        with distribution.scope():
            mirrored_var = distribution.extended.call_for_each_replica(var_fn)
            self.assertTrue(distribute_utils.is_mirrored(mirrored_var))
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual(1.0, self.evaluate(mirrored_var))

            def model_fn():
                if False:
                    while True:
                        i = 10
                return mirrored_var.assign_add(5.0)
            self.evaluate(distribution.experimental_local_results(distribution.extended.call_for_each_replica(model_fn)))
            self.assertEqual(6.0, self.evaluate(mirrored_var))

    def testAssignSubMirroredVarCrossDeviceContext(self, distribution):
        if False:
            print('Hello World!')

        def var_fn():
            if False:
                while True:
                    i = 10
            return variable_v1.VariableV1(5.0, name='foo')
        with distribution.scope():
            mirrored_var = distribution.extended.call_for_each_replica(var_fn)
            self.assertTrue(distribute_utils.is_mirrored(mirrored_var))
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual(5.0, self.evaluate(mirrored_var))
            mirrored_var_result = self.evaluate(mirrored_var.assign_sub(2.0))
            self.assertEqual(3.0, mirrored_var_result)
            self.assertEqual(3.0, self.evaluate(distribution.experimental_local_results(mirrored_var)[0]))
            self.assertEqual(3.0, self.evaluate(distribution.experimental_local_results(mirrored_var)[1]))
            self.assertEqual(distribution.extended.worker_devices[0], mirrored_var._devices[0])
            self.assertEqual(distribution.extended.worker_devices[1], mirrored_var._devices[1])

    def testAssignSubMirroredVarReplicaContext(self, distribution):
        if False:
            print('Hello World!')

        def var_fn():
            if False:
                print('Hello World!')
            return variable_v1.VariableV1(5.0, name='foo', aggregation=variable_scope.VariableAggregation.MEAN)
        with distribution.scope():
            mirrored_var = distribution.extended.call_for_each_replica(var_fn)
            self.assertTrue(distribute_utils.is_mirrored(mirrored_var))
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual(5.0, self.evaluate(mirrored_var))

            def model_fn():
                if False:
                    while True:
                        i = 10
                value = math_ops.cast(distribute_lib.get_replica_context().replica_id_in_sync_group, mirrored_var.dtype)
                return mirrored_var.assign_sub(value)
            self.evaluate(distribution.experimental_local_results(distribution.extended.call_for_each_replica(model_fn)))
            self.assertEqual(4.5, self.evaluate(mirrored_var))

    def testAssignSubMirroredVarReplicaContextWithSingleValue(self, distribution):
        if False:
            while True:
                i = 10

        def var_fn():
            if False:
                for i in range(10):
                    print('nop')
            return variable_v1.VariableV1(5.0, name='foo', aggregation=variable_scope.VariableAggregation.MEAN)
        with distribution.scope():
            mirrored_var = distribution.extended.call_for_each_replica(var_fn)
            self.assertTrue(distribute_utils.is_mirrored(mirrored_var))
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual(5.0, self.evaluate(mirrored_var))

            def model_fn():
                if False:
                    print('Hello World!')
                return mirrored_var.assign_sub(1.0)
            self.evaluate(distribution.experimental_local_results(distribution.extended.call_for_each_replica(model_fn)))
            self.assertEqual(4.0, self.evaluate(mirrored_var))

@combinations.generate(combinations.combine(distribution=[strategy_combinations.mirrored_strategy_with_gpu_and_cpu], mode=['graph', 'eager']))
class MirroredAndSyncOnReadVariableInitializerTest(test.TestCase):

    def testAssignMirroredVarInitializer(self, distribution):
        if False:
            return 10
        with context.graph_mode():

            def var_fn():
                if False:
                    for i in range(10):
                        print('nop')
                v = variable_v1.VariableV1(1.0, name='foo')
                return v
            with distribution.scope():
                mirrored_var = distribution.extended.call_for_each_replica(var_fn)
                self.assertTrue(distribute_utils.is_mirrored(mirrored_var))
                self.assertFalse(self.evaluate(mirrored_var.is_initialized()))
                self.evaluate(mirrored_var.initializer)
                self.assertTrue(self.evaluate(mirrored_var.is_initialized()))

    def testAssignReplicaLocalVarInitializer(self, distribution):
        if False:
            while True:
                i = 10
        with context.graph_mode():

            def model_fn():
                if False:
                    for i in range(10):
                        print('nop')
                v_sum = variable_v1.VariableV1(1.0, synchronization=variable_scope.VariableSynchronization.ON_READ, aggregation=variable_scope.VariableAggregation.SUM)
                self.assertTrue(distribute_utils.is_sync_on_read(v_sum))
                return v_sum
            with distribution.scope():
                sync_on_read_var = distribution.extended.call_for_each_replica(model_fn)
                self.assertTrue(distribute_utils.is_sync_on_read(sync_on_read_var))
                self.assertFalse(self.evaluate(sync_on_read_var.is_initialized()))
                self.evaluate(sync_on_read_var.initializer)
                self.assertTrue(self.evaluate(sync_on_read_var.is_initialized()))

@combinations.generate(combinations.combine(distribution=[strategy_combinations.mirrored_strategy_with_gpu_and_cpu], mode=['graph', 'eager']))
class SyncOnReadVariableAssignTest(test.TestCase):

    def testAssignReplicaLocalVarSumAggregation(self, distribution):
        if False:
            for i in range(10):
                print('nop')

        def model_fn():
            if False:
                for i in range(10):
                    print('nop')
            v_sum = variable_v1.VariableV1(1.0, synchronization=variable_scope.VariableSynchronization.ON_READ, aggregation=variable_scope.VariableAggregation.SUM)
            return v_sum
        with distribution.scope():
            sync_on_read_var = distribution.extended.call_for_each_replica(model_fn)
            self.assertTrue(distribute_utils.is_sync_on_read(sync_on_read_var))
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual(2.0, self.evaluate(distribution.extended.read_var(sync_on_read_var)))
            tlv_ops = sync_on_read_var.assign(6.0)
            self.evaluate(tlv_ops)
            self.assertEqual(6.0, self.evaluate(distribution.extended.read_var(sync_on_read_var)))

    def testAssignReplicaLocalVarMeanAggregation(self, distribution):
        if False:
            print('Hello World!')

        def model_fn():
            if False:
                while True:
                    i = 10
            v_sum = variable_v1.VariableV1(1.0, synchronization=variable_scope.VariableSynchronization.ON_READ, aggregation=variable_scope.VariableAggregation.MEAN)
            return v_sum
        with distribution.scope():
            sync_on_read_var = distribution.extended.call_for_each_replica(model_fn)
            self.assertTrue(distribute_utils.is_sync_on_read(sync_on_read_var))
            self.evaluate(variables.global_variables_initializer())
            self.assertEqual(1.0, self.evaluate(distribution.extended.read_var(sync_on_read_var)))
            tlv_ops = sync_on_read_var.assign(6.0)
            self.evaluate(tlv_ops)
            self.assertEqual(6.0, self.evaluate(distribution.extended.read_var(sync_on_read_var)))

class MockModel(object):

    def __init__(self, two_variables=False):
        if False:
            while True:
                i = 10
        self.variables = []
        self.variables.append(variable_v1.VariableV1(1.25, name='dummy_var1'))
        if two_variables:
            self.variables.append(variable_v1.VariableV1(2.0, name='dummy_var2'))

    def __call__(self, factor=2):
        if False:
            while True:
                i = 10
        x = factor * self.variables[0]
        if len(self.variables) > 1:
            x += self.variables[1]
        return x

@combinations.generate(combinations.combine(distribution=[combinations.NamedDistribution('Mirrored', lambda : mirrored_strategy.MirroredStrategy(devices=mirrored_strategy.all_local_devices(), cross_device_ops=cross_device_ops_lib.ReductionToOneDevice()), required_gpus=1)], mode=['graph']))
class MultiWorkerMirroredStrategyTest(multi_worker_test_base.MultiWorkerTestBase, strategy_test_lib.DistributionTestBase):

    def _configure_distribution_strategy(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        cluster_spec = server_lib.ClusterSpec({'worker': ['/job:worker/task:0', '/job:worker/task:1']})
        distribution.configure(cluster_spec=cluster_spec)

    def test_num_replicas_in_sync(self, distribution):
        if False:
            while True:
                i = 10
        self._configure_distribution_strategy(distribution)
        self.assertEqual(context.num_gpus() * 2, distribution.num_replicas_in_sync)

    def testMinimizeLossGraph(self, distribution):
        if False:
            while True:
                i = 10
        self._configure_distribution_strategy(distribution)
        self._test_minimize_loss_graph(distribution, learning_rate=0.05)

    def testDeviceScope(self, distribution):
        if False:
            return 10
        'Test the device scope of multi-worker MirroredStrategy.'
        self._configure_distribution_strategy(distribution)
        with distribution.scope():
            a = constant_op.constant(1.0)
            with ops.device('/cpu:0'):
                b = constant_op.constant(1.0)
            self.assertEqual(a.device, '/job:worker/task:0')
            self.assertEqual(b.device, '/job:worker/task:0/device:CPU:0')

    def testMakeInputFnIteratorWithDataset(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        self._configure_distribution_strategy(distribution)
        dataset_fn = lambda : dataset_ops.Dataset.range(100)
        num_gpus = context.num_gpus()
        num_workers = 2
        expected_values = [[i + j for j in range(num_gpus)] * num_workers for i in range(0, 100, num_gpus)]
        with context.graph_mode(), self.cached_session() as sess:
            input_fn = self._input_fn_to_test_input_context(dataset_fn, expected_num_replicas_in_sync=num_workers * num_gpus, expected_num_input_pipelines=num_workers, expected_input_pipeline_id=None)
            iterator = distribution.make_input_fn_iterator(input_fn)
            self._test_input_fn_iterator(iterator, distribution.extended.worker_devices, expected_values, sess)

    def testMakeInputFnIteratorWithCallable(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        self._configure_distribution_strategy(distribution)

        def fn():
            if False:
                i = 10
                return i + 15
            dataset = dataset_ops.Dataset.range(100)
            it = dataset_ops.make_one_shot_iterator(dataset)
            return it.get_next
        num_gpus = context.num_gpus()
        num_workers = 2
        expected_values = []
        for i in range(0, 100, num_gpus):
            expected_values.append([i + j for j in range(num_gpus)] * num_workers)
        with context.graph_mode(), self.cached_session() as sess:
            input_fn = self._input_fn_to_test_input_context(fn, expected_num_replicas_in_sync=num_workers * num_gpus, expected_num_input_pipelines=num_workers, expected_input_pipeline_id=None)
            iterator = distribution.make_input_fn_iterator(input_fn)
            self._test_input_fn_iterator(iterator, distribution.extended.worker_devices, expected_values, sess, test_reinitialize=False, ignore_order=True)

    def testUpdateConfigProto(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        distribution.configure(cluster_spec={'worker': ['fake1', 'fake2']})
        config_proto = config_pb2.ConfigProto()
        new_config = distribution.update_config_proto(config_proto)
        self.assertTrue(new_config.isolate_session_state)

@combinations.generate(combinations.combine(distribution=[combinations.NamedDistribution('Mirrored', lambda : mirrored_strategy.MirroredStrategy(devices=['/job:worker/task:0/gpu:{}'.format(i) for i in range(context.num_gpus())]), required_gpus=1)], mode=['graph']))
class RemoteSingleWorkerMirroredStrategyGraph(multi_worker_test_base.SingleWorkerTestBaseGraph, strategy_test_lib.RemoteSingleWorkerMirroredStrategyBase):

    def _get_num_gpus(self):
        if False:
            print('Hello World!')
        return context.num_gpus()

    def testNumReplicasInSync(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        self._testNumReplicasInSync(distribution)

    def testMinimizeLoss(self, distribution):
        if False:
            print('Hello World!')
        self._testMinimizeLoss(distribution)

    def testDeviceScope(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        self._testDeviceScope(distribution)

    def testMakeInputFnIteratorWithDataset(self, distribution):
        if False:
            i = 10
            return i + 15
        self._testMakeInputFnIteratorWithDataset(distribution)

    def testMakeInputFnIteratorWithCallable(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        self._testMakeInputFnIteratorWithCallable(distribution)

class MultiWorkerMirroredStrategyTestWithChief(multi_worker_test_base.MultiWorkerTestBase, strategy_test_lib.DistributionTestBase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        'Create a local cluster with 2 workers and 1 chief.'
        cls._cluster_spec = multi_worker_test_base.create_in_process_cluster(num_workers=2, num_ps=0, has_chief=True)
        cls._default_target = 'grpc://' + cls._cluster_spec['chief'][0]

    def _make_cross_device_ops(self):
        if False:
            i = 10
            return i + 15
        return cross_device_ops_lib.ReductionToOneDevice()

    def testMinimizeLossGraph(self):
        if False:
            for i in range(10):
                print('nop')
        with context.graph_mode():
            strategy = mirrored_strategy.MirroredStrategy(cross_device_ops=self._make_cross_device_ops())
            strategy.configure(cluster_spec=self._cluster_spec)
            self._test_minimize_loss_graph(strategy, learning_rate=0.05)

    def testMinimizeLossGraphMirroredStrategy(self):
        if False:
            while True:
                i = 10
        with context.graph_mode():
            strategy = mirrored_strategy.MirroredStrategy(mirrored_strategy.all_local_devices(), cross_device_ops=self._make_cross_device_ops())
            strategy.configure(cluster_spec=self._cluster_spec)
            self._test_minimize_loss_graph(strategy, learning_rate=0.05)

    def testMinimizeLossGraphMirroredStrategyWithOneNode(self):
        if False:
            i = 10
            return i + 15
        with context.graph_mode():
            cluster_spec = {}
            cluster_spec['chief'] = self._cluster_spec['chief']
            tf_config = {'cluster': cluster_spec}
            with test.mock.patch.dict('os.environ', {'TF_CONFIG': json.dumps(tf_config)}):
                strategy = mirrored_strategy.MirroredStrategy()
                if context.num_gpus() == 0:
                    self.assertIsInstance(strategy.extended._cross_device_ops, cross_device_ops_lib.ReductionToOneDevice)
            self.skipTest('b/130551176, run the following once fixed.')
            self._test_minimize_loss_graph(strategy, learning_rate=0.05)

    def testInitializeFromTFConfig(self):
        if False:
            for i in range(10):
                print('nop')
        with context.graph_mode():
            tf_config = {'cluster': self._cluster_spec}
            with test.mock.patch.dict('os.environ', {'TF_CONFIG': json.dumps(tf_config)}):
                strategy = mirrored_strategy.MirroredStrategy(cross_device_ops=self._make_cross_device_ops())
                self.assertEqual(max(context.num_gpus(), 1) * 3, strategy.num_replicas_in_sync)

    def testSummaryForReplicaZeroOnly(self):
        if False:
            return 10
        with context.graph_mode():
            strategy = mirrored_strategy.MirroredStrategy(mirrored_strategy.all_local_devices(), cross_device_ops=self._make_cross_device_ops())
            strategy.configure(cluster_spec=self._cluster_spec)
            self._test_summary_for_replica_zero_only(strategy)

class MirroredVariableStopGradientTest(test.TestCase, parameterized.TestCase):

    @combinations.generate(combinations.combine(distribution=[strategy_combinations.mirrored_strategy_with_one_cpu, strategy_combinations.mirrored_strategy_with_one_gpu], mode=['graph']))
    def testMirroredVariableAsStopGradient(self, distribution):
        if False:
            print('Hello World!')
        with distribution.scope():
            inp = constant_op.constant(1.0)
            x = variables.Variable(1.0)
            y = inp * x
            grads = gradients.gradients(x, y, stop_gradients=x)
            self.assertIsNone(grads[0])

@combinations.generate(combinations.combine(distribution=[strategy_combinations.mirrored_strategy_with_gpu_and_cpu], mode=['eager']))
class FunctionTest(test.TestCase, parameterized.TestCase):

    def testBackwardFunctionDevicePlacement(self, distribution):
        if False:
            i = 10
            return i + 15
        with distribution.scope():
            w = variable_v1.VariableV1([1.5], name='w')
            b = variable_v1.VariableV1([0.5], name='b')

        @def_function.function
        def forward(x, w, b):
            if False:
                print('Hello World!')
            return x * w + b
        x = array_ops.identity([1.0], name='x_useless')
        concrete_forward = forward.get_concrete_function(x, w._primary, b._primary)
        with distribution.scope():

            def replica_fn():
                if False:
                    for i in range(10):
                        print('nop')
                with backprop.GradientTape() as t:
                    x = array_ops.identity([1.0], name='x')
                    loss = concrete_forward(x, w._get(), b._get()) - [1.0]
                    return t.gradient(loss, [w, b])

            def step_fn():
                if False:
                    print('Hello World!')
                return distribution.run(replica_fn)
            context.enable_run_metadata()
            (g1, g2) = step_fn()
            run_metadata = context.export_run_metadata()
            context.disable_run_metadata()
            self.assertEqual(self.evaluate(g1._primary), 1.0)
            self.assertEqual(self.evaluate(g2._primary), 1.0)
            node_name = 'gradients_mul_grad_mul_1_x'
            devices_for_this_node = set()
            for partition_graph in run_metadata.partition_graphs:
                for node in partition_graph.node:
                    if node.name == node_name:
                        devices_for_this_node.add(node.device)
            devices = [device_util.resolve('/device:GPU:0'), device_util.resolve('/device:CPU:0')]
            self.assertSetEqual(devices_for_this_node, set(devices))

    def testFuctionPreservesAutoGraph(self, distribution):
        if False:
            i = 10
            return i + 15

        def f():
            if False:
                i = 10
                return i + 15
            self.assertTrue(converter_testing.is_inside_generated_code())
            return 1
        with distribution.scope():

            @def_function.function
            def replica_fn():
                if False:
                    while True:
                        i = 10
                return f()
            distribution.run(replica_fn)

    def testPreserveTracebackFiltering(self, distribution):
        if False:
            while True:
                i = 10
        traceback_utils.disable_traceback_filtering()
        self.assertFalse(traceback_utils.is_traceback_filtering_enabled())

        def f():
            if False:
                return 10
            self.assertFalse(traceback_utils.is_traceback_filtering_enabled())
        distribution.run(f)

def _replica_id():
    if False:
        return 10
    replica_id = distribute_lib.get_replica_context().replica_id_in_sync_group
    if not isinstance(replica_id, tensor_lib.Tensor):
        replica_id = constant_op.constant(replica_id)
    return array_ops.identity(replica_id)

def _replica_id_as_int():
    if False:
        i = 10
        return i + 15
    replica_id = distribute_lib.get_replica_context().replica_id_in_sync_group
    if isinstance(replica_id, tensor_lib.Tensor):
        replica_id = tensor_util.constant_value(replica_id)
    return replica_id
if __name__ == '__main__':
    test_util.main(config_logical_devices=False)