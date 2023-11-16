"""Tests for class OneDeviceStrategy."""
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import device as tf_device

@combinations.generate(combinations.combine(distribution=[strategy_combinations.one_device_strategy, strategy_combinations.one_device_strategy_gpu], mode=['eager', 'graph']))
class OneDeviceStrategyTest(strategy_test_lib.DistributionTestBase, strategy_test_lib.OneDeviceDistributionTestBase):

    def testMinimizeLoss(self, distribution):
        if False:
            for i in range(10):
                print('nop')
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
            print('Hello World!')
        self._test_call_and_merge_exceptions(distribution)

    def testReplicateDataset(self, distribution):
        if False:
            i = 10
            return i + 15
        if tf2.enabled() and (not context.executing_eagerly()):
            self.skipTest('Skipping test since we do not support graph mode in TF 2')
        dataset_fn = lambda : dataset_ops.Dataset.range(10)
        expected_values = [[i] for i in range(10)]
        input_fn = self._input_fn_to_test_input_context(dataset_fn, expected_num_replicas_in_sync=1, expected_num_input_pipelines=1, expected_input_pipeline_id=0)
        self._test_input_fn_iterable(distribution, input_fn, expected_values)

    def testMakeInputFnIteratorWithDataset(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        dataset_fn = lambda : dataset_ops.Dataset.range(10)
        expected_values = [[i] for i in range(10)]
        input_fn = self._input_fn_to_test_input_context(dataset_fn, expected_num_replicas_in_sync=1, expected_num_input_pipelines=1, expected_input_pipeline_id=0)
        iterator = distribution.make_input_fn_iterator(input_fn)
        self._test_input_fn_iterator(iterator, distribution.extended.worker_devices, expected_values)

    def testMakeInputFnIteratorWithCallable(self, distribution):
        if False:
            print('Hello World!')

        def fn():
            if False:
                for i in range(10):
                    print('nop')
            dataset = dataset_ops.Dataset.range(10)
            it = dataset_ops.make_one_shot_iterator(dataset)
            return it.get_next
        expected_values = [[i] for i in range(10)]
        input_fn = self._input_fn_to_test_input_context(fn, expected_num_replicas_in_sync=1, expected_num_input_pipelines=1, expected_input_pipeline_id=0)
        iterator = distribution.make_input_fn_iterator(input_fn)
        self._test_input_fn_iterator(iterator, distribution.extended.worker_devices, expected_values, test_reinitialize=False, ignore_order=True)

    def testNumpyDataset(self, distribution):
        if False:
            while True:
                i = 10
        self._test_numpy_dataset(distribution)

    def testRun(self, distribution):
        if False:
            while True:
                i = 10
        self._test_run(distribution)

    def testAllReduceSum(self, distribution):
        if False:
            while True:
                i = 10
        self._test_all_reduce_sum(distribution)

    def testAllReduceSumGradients(self, distribution):
        if False:
            print('Hello World!')
        self._test_all_reduce_sum_gradients(distribution)

    def testAllReduceSumGradientTape(self, distribution):
        if False:
            for i in range(10):
                print('nop')
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
            i = 10
            return i + 15
        self._test_all_reduce_mean_gradient_tape(distribution)

    def testTrainableVariables(self, distribution):
        if False:
            return 10
        self._test_trainable_variable(distribution)

    def test_prefetch_to_device_dataset(self, distribution):
        if False:
            i = 10
            return i + 15
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
        device_types = tf_device.DeviceSpec.from_string(item.device).device_type
        expected_device_types = tf_device.DeviceSpec.from_string(distribution.extended.worker_devices[0]).device_type
        self.assertAllEqual(device_types, expected_device_types)

    def test_prefetch_to_host_dataset(self, distribution):
        if False:
            return 10
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
        self.assertAllEqual(tf_device.DeviceSpec.from_string(item.device).device_type, 'CPU')

@combinations.generate(combinations.combine(distribution=[strategy_combinations.one_device_strategy_on_worker_1, strategy_combinations.one_device_strategy_gpu_on_worker_1], mode=['eager', 'graph']))
class OneDeviceStrategyOnRemoteWorkerTest(strategy_test_lib.DistributionTestBase, strategy_test_lib.OneDeviceDistributionTestBase):

    def testDeviceAndInputDeviceAreColocated(self, distribution):
        if False:
            i = 10
            return i + 15
        self._test_device_and_input_device_are_colocated(distribution)

    def testDeviceAndInputDeviceAreColocatedWithFunction(self, distribution):
        if False:
            return 10
        self._test_device_and_input_device_are_colocated_with_function(distribution)
if __name__ == '__main__':
    test.main()