"""Tests for a little bit of strategy_combinations."""
from absl.testing import parameterized
from tensorflow.python import tf2
from tensorflow.python.distribute import central_storage_strategy
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import one_device_strategy
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class StrategyCombinationsTest(test.TestCase, parameterized.TestCase):

    @combinations.generate(combinations.combine(strategy=strategy_combinations.two_replica_strategies, mode=['graph', 'eager']))
    def testTwoReplicaStrategy(self, strategy):
        if False:
            print('Hello World!')
        with strategy.scope():

            @def_function.function
            def one():
                if False:
                    for i in range(10):
                        print('nop')
                return array_ops.identity(1.0)
            one_per_replica = strategy.run(one)
            num_replicas = strategy.reduce(reduce_util.ReduceOp.SUM, one_per_replica, axis=None)
            self.assertEqual(self.evaluate(num_replicas), 2.0)

    @combinations.generate(combinations.combine(strategy=strategy_combinations.four_replica_strategies, mode=['graph', 'eager']))
    def testFourReplicaStrategy(self, strategy):
        if False:
            i = 10
            return i + 15
        with strategy.scope():

            @def_function.function
            def one():
                if False:
                    return 10
                return array_ops.identity(1.0)
            one_per_replica = strategy.run(one)
            num_replicas = strategy.reduce(reduce_util.ReduceOp.SUM, one_per_replica, axis=None)
            self.assertEqual(self.evaluate(num_replicas), 4.0)

    @combinations.generate(combinations.combine(distribution=[strategy_combinations.mirrored_strategy_with_cpu_1_and_2], mode=['graph', 'eager']))
    def testMirrored2CPUs(self, distribution):
        if False:
            while True:
                i = 10
        with distribution.scope():
            one_per_replica = distribution.run(lambda : constant_op.constant(1))
            num_replicas = distribution.reduce(reduce_util.ReduceOp.SUM, one_per_replica, axis=None)
            self.assertEqual(2, self.evaluate(num_replicas))

class V1StrategyTest(test.TestCase, parameterized.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        tf2.disable()

    @combinations.generate(combinations.combine(strategy=[strategy_combinations.one_device_strategy, strategy_combinations.one_device_strategy_gpu, strategy_combinations.one_device_strategy_gpu_on_worker_1, strategy_combinations.one_device_strategy_on_worker_1]))
    def testOneDevice(self, strategy):
        if False:
            i = 10
            return i + 15
        self.assertIsInstance(strategy, one_device_strategy.OneDeviceStrategyV1)

    @combinations.generate(combinations.combine(strategy=[strategy_combinations.mirrored_strategy_with_cpu_1_and_2, strategy_combinations.mirrored_strategy_with_gpu_and_cpu, strategy_combinations.mirrored_strategy_with_one_cpu, strategy_combinations.mirrored_strategy_with_one_gpu, strategy_combinations.mirrored_strategy_with_two_gpus]))
    def testMirrored(self, strategy):
        if False:
            for i in range(10):
                print('nop')
        self.assertIsInstance(strategy, mirrored_strategy.MirroredStrategyV1)

    @combinations.generate(combinations.combine(strategy=[strategy_combinations.multi_worker_mirrored_2x1_cpu, strategy_combinations.multi_worker_mirrored_2x1_gpu, strategy_combinations.multi_worker_mirrored_2x2_gpu, strategy_combinations.multi_worker_mirrored_4x1_cpu]))
    def testMultiWorkerMirrored(self, strategy):
        if False:
            for i in range(10):
                print('nop')
        self.assertIsInstance(strategy, collective_all_reduce_strategy.CollectiveAllReduceStrategy)

    @combinations.generate(combinations.combine(strategy=[strategy_combinations.central_storage_strategy_with_gpu_and_cpu, strategy_combinations.central_storage_strategy_with_two_gpus]))
    def testCentralStorage(self, strategy):
        if False:
            return 10
        self.assertIsInstance(strategy, central_storage_strategy.CentralStorageStrategyV1)

    @combinations.generate(combinations.combine(strategy=strategy_combinations.tpu_strategies))
    def testTPU(self, strategy):
        if False:
            return 10
        self.assertIsInstance(strategy, tpu_strategy.TPUStrategyV1)

class V2StrategyTest(test.TestCase, parameterized.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        tf2.enable()

    @combinations.generate(combinations.combine(strategy=[strategy_combinations.one_device_strategy, strategy_combinations.one_device_strategy_gpu, strategy_combinations.one_device_strategy_gpu_on_worker_1, strategy_combinations.one_device_strategy_on_worker_1]))
    def testOneDevice(self, strategy):
        if False:
            return 10
        self.assertIsInstance(strategy, one_device_strategy.OneDeviceStrategy)

    @combinations.generate(combinations.combine(strategy=[strategy_combinations.mirrored_strategy_with_cpu_1_and_2, strategy_combinations.mirrored_strategy_with_gpu_and_cpu, strategy_combinations.mirrored_strategy_with_one_cpu, strategy_combinations.mirrored_strategy_with_one_gpu, strategy_combinations.mirrored_strategy_with_two_gpus]))
    def testMirrored(self, strategy):
        if False:
            print('Hello World!')
        self.assertIsInstance(strategy, mirrored_strategy.MirroredStrategy)

    @combinations.generate(combinations.combine(strategy=[strategy_combinations.multi_worker_mirrored_2x1_cpu, strategy_combinations.multi_worker_mirrored_2x1_gpu, strategy_combinations.multi_worker_mirrored_2x2_gpu, strategy_combinations.multi_worker_mirrored_4x1_cpu]))
    def testMultiWorkerMirrored(self, strategy):
        if False:
            for i in range(10):
                print('nop')
        self.assertIsInstance(strategy, collective_all_reduce_strategy.CollectiveAllReduceStrategy)

    @combinations.generate(combinations.combine(strategy=[strategy_combinations.central_storage_strategy_with_gpu_and_cpu, strategy_combinations.central_storage_strategy_with_two_gpus]))
    def testCentralStorage(self, strategy):
        if False:
            print('Hello World!')
        self.assertIsInstance(strategy, central_storage_strategy.CentralStorageStrategy)

    @combinations.generate(combinations.combine(strategy=strategy_combinations.tpu_strategies))
    def testTPU(self, strategy):
        if False:
            for i in range(10):
                print('nop')
        self.assertIsInstance(strategy, (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV2))

    @combinations.generate(combinations.combine(strategy=[strategy_combinations.parameter_server_strategy_3worker_2ps_cpu, strategy_combinations.parameter_server_strategy_1worker_2ps_cpu, strategy_combinations.parameter_server_strategy_3worker_2ps_1gpu, strategy_combinations.parameter_server_strategy_1worker_2ps_1gpu]))
    def testParameterServer(self, strategy):
        if False:
            i = 10
            return i + 15
        self.assertIsInstance(strategy, parameter_server_strategy_v2.ParameterServerStrategyV2)
if __name__ == '__main__':
    test_util.main()