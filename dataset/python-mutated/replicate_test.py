"""Tests for the private `replicate()` transformation."""
from absl.testing import parameterized
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib

class LocalReplicateTest(test_base.DatasetTestBase, parameterized.TestCase):

    def __init__(self, methodName='runTest'):
        if False:
            for i in range(10):
                print('nop')
        super(LocalReplicateTest, self).__init__(methodName)
        cpus = config.list_physical_devices('CPU')
        config.set_logical_device_configuration(cpus[0], [context.LogicalDeviceConfiguration(), context.LogicalDeviceConfiguration(), context.LogicalDeviceConfiguration()])
        self._device0 = '/device:CPU:0'
        self._device1 = '/device:CPU:1'
        self._device2 = '/device:CPU:2'

    def tearDown(self):
        if False:
            while True:
                i = 10
        super().tearDown()
        ops.device(None).__enter__()
        context._reset_context()

    @combinations.generate(test_base.default_test_combinations())
    def testBasic(self):
        if False:
            i = 10
            return i + 15
        with ops.device(self._device0):
            dataset0 = dataset_ops.Dataset.range(100)
        replicated_ds = distribute.replicate(dataset0, [self._device1, self._device2])
        dataset1 = replicated_ds[self._device1]
        dataset2 = replicated_ds[self._device2]
        with ops.device(self._device0):
            self.assertDatasetProduces(dataset0, range(100))
        with ops.device(self._device1):
            self.assertDatasetProduces(dataset1, range(100))
        with ops.device(self._device2):
            self.assertDatasetProduces(dataset2, range(100))

    @combinations.generate(test_base.default_test_combinations())
    def testFromTensorsWithDataset(self):
        if False:
            while True:
                i = 10
        with ops.device(self._device0):
            dataset0 = dataset_ops.Dataset.range(100)
            dataset0 = dataset_ops.Dataset.from_tensors(dataset0)
            dataset0 = dataset0.flat_map(lambda x: x)
        replicated_ds = distribute.replicate(dataset0, [self._device1, self._device2])
        dataset1 = replicated_ds[self._device1]
        dataset2 = replicated_ds[self._device2]
        with ops.device(self._device0):
            self.assertDatasetProduces(dataset0, range(100))
        with ops.device(self._device1):
            self.assertDatasetProduces(dataset1, range(100))
        with ops.device(self._device2):
            self.assertDatasetProduces(dataset2, range(100))

    @combinations.generate(test_base.default_test_combinations())
    def testFromTensorSlicesWithDataset(self):
        if False:
            i = 10
            return i + 15
        with ops.device(self._device0):
            dataset0 = dataset_ops.Dataset.range(100)
            dataset0 = dataset_ops.Dataset.from_tensor_slices([dataset0])
            dataset0 = dataset0.flat_map(lambda x: x)
        replicated_ds = distribute.replicate(dataset0, [self._device1, self._device2])
        dataset1 = replicated_ds[self._device1]
        dataset2 = replicated_ds[self._device2]
        with ops.device(self._device0):
            self.assertDatasetProduces(dataset0, range(100))
        with ops.device(self._device1):
            self.assertDatasetProduces(dataset1, range(100))
        with ops.device(self._device2):
            self.assertDatasetProduces(dataset2, range(100))

    @combinations.generate(test_base.default_test_combinations())
    def testVariableInput(self):
        if False:
            return 10
        with ops.device(self._device0):
            counter_var = variable_scope.get_variable('counter', (), dtypes.int32, use_resource=True)
            dataset0 = dataset_ops.Dataset.range(100).map(lambda _: counter_var.assign_add(1))
        replicated_ds = distribute.replicate(dataset0, [self._device1, self._device2])
        dataset1 = replicated_ds[self._device1]
        dataset2 = replicated_ds[self._device2]
        self.evaluate(counter_var.initializer)
        with ops.device(self._device1):
            self.assertDatasetProduces(dataset1, range(1, 101), requires_initialization=True)
        with ops.device(self._device2):
            self.assertDatasetProduces(dataset2, range(1, 101), requires_initialization=True)
        with ops.device(self._device0):
            self.assertDatasetProduces(dataset0, range(1, 101), requires_initialization=True)

    @combinations.generate(test_base.default_test_combinations())
    def testExternalStatePolicyIgnore(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.device(self._device0):
            dataset0 = dataset_ops.Dataset.range(100).map(lambda _: random_ops.random_uniform([], minval=1, maxval=10, dtype=dtypes.float32))
            opt = options_lib.Options()
            opt.experimental_external_state_policy = options_lib.ExternalStatePolicy.IGNORE
            dataset0 = dataset0.with_options(opt)
        replicated_ds = distribute.replicate(dataset0, [self._device1, self._device2])
        dataset1 = replicated_ds[self._device1]
        dataset2 = replicated_ds[self._device2]
        with ops.device(self._device0):
            get_next0 = self.getNext(dataset0)
        with ops.device(self._device1):
            get_next1 = self.getNext(dataset1)
        with ops.device(self._device2):
            get_next2 = self.getNext(dataset2)
        for _ in range(100):
            self.evaluate(get_next0())
            self.evaluate(get_next1())
            self.evaluate(get_next2())

    @combinations.generate(test_base.default_test_combinations())
    def testExternalStatePolicyWarn(self):
        if False:
            i = 10
            return i + 15
        with ops.device(self._device0):
            dataset0 = dataset_ops.Dataset.range(100).map(lambda _: random_ops.random_uniform([], minval=1, maxval=10, dtype=dtypes.float32))
            opt = options_lib.Options()
            opt.experimental_external_state_policy = options_lib.ExternalStatePolicy.WARN
            dataset0 = dataset0.with_options(opt)
        replicated_ds = distribute.replicate(dataset0, [self._device1, self._device2])
        dataset1 = replicated_ds[self._device1]
        dataset2 = replicated_ds[self._device2]
        with ops.device(self._device0):
            get_next0 = self.getNext(dataset0)
        with ops.device(self._device1):
            get_next1 = self.getNext(dataset1)
        with ops.device(self._device2):
            get_next2 = self.getNext(dataset2)
        for _ in range(100):
            self.evaluate(get_next0())
            self.evaluate(get_next1())
            self.evaluate(get_next2())

    @combinations.generate(test_base.default_test_combinations())
    def testExternalStatePolicyFail(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.device(self._device0):
            dataset0 = dataset_ops.Dataset.range(100).map(lambda _: random_ops.random_uniform([], minval=1, maxval=10, dtype=dtypes.float32))
            opt = options_lib.Options()
            opt.experimental_external_state_policy = options_lib.ExternalStatePolicy.FAIL
            dataset0 = dataset0.with_options(opt)
        with self.assertRaises(errors.FailedPreconditionError):
            replicated_ds = distribute.replicate(dataset0, [self._device1, self._device2])
            dataset1 = replicated_ds[self._device1]
            dataset2 = replicated_ds[self._device2]
            with ops.device(self._device0):
                get_next0 = self.getNext(dataset0)
            with ops.device(self._device1):
                get_next1 = self.getNext(dataset1)
            with ops.device(self._device2):
                get_next2 = self.getNext(dataset2)
            for _ in range(100):
                self.evaluate(get_next0())
                self.evaluate(get_next1())
                self.evaluate(get_next2())

def _get_server_def(job_name, local_server_port, remote_server_addresses, task_index):
    if False:
        for i in range(10):
            print('nop')
    'Returns a server def with a single job + multiple tasks.'
    cluster_def = cluster_pb2.ClusterDef()
    job_def = cluster_def.job.add()
    job_def.name = job_name
    job_def.tasks[0] = 'localhost:%d' % local_server_port
    for (i, remote_server_address) in enumerate(remote_server_addresses, start=1):
        job_def.tasks[i] = remote_server_address
    server_def = tensorflow_server_pb2.ServerDef(cluster=cluster_def, job_name=job_name, task_index=task_index, protocol='grpc')
    return server_def

class EagerClusterReplicateTest(test_base.DatasetTestBase, parameterized.TestCase):

    def __init__(self, methodName='runTest'):
        if False:
            return 10
        super(EagerClusterReplicateTest, self).__init__(methodName)
        self._job_name = 'remove_device'
        self._device0 = '/job:%s/replica:0/task:0/device:CPU:0' % self._job_name
        self._device1 = '/job:%s/replica:0/task:1/device:CPU:0' % self._job_name
        self._device2 = '/job:%s/replica:0/task:2/device:CPU:0' % self._job_name

    def setUp(self):
        if False:
            return 10
        super(EagerClusterReplicateTest, self).setUp()
        self._cached_server1 = server_lib.Server.create_local_server()
        self._cached_server2 = server_lib.Server.create_local_server()
        self._cached_server1_target = self._cached_server1.target[len('grpc://'):]
        self._cached_server2_target = self._cached_server2.target[len('grpc://'):]
        local_port = pywrap_tfe.TF_PickUnusedPortOrDie()
        context.set_server_def(server_def=_get_server_def(self._job_name, local_server_port=local_port, remote_server_addresses=[self._cached_server1_target, self._cached_server2_target], task_index=0))

    def tearDown(self):
        if False:
            print('Hello World!')
        super().tearDown()
        ops.device(None).__enter__()
        context._reset_context()

    @combinations.generate(combinations.combine(tf_api_version=[2], mode=['eager']))
    def testBasic(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.device(self._device0):
            dataset0 = dataset_ops.Dataset.range(100)
        replicated_ds = distribute.replicate(dataset0, [self._device1, self._device2])
        dataset1 = replicated_ds[self._device1]
        dataset2 = replicated_ds[self._device2]
        with ops.device(self._device0):
            self.assertDatasetProduces(dataset0, range(100))
        with ops.device(self._device1):
            self.assertDatasetProduces(dataset1, range(100))
        with ops.device(self._device2):
            self.assertDatasetProduces(dataset2, range(100))

    @combinations.generate(combinations.combine(tf_api_version=[2], mode=['eager']))
    def testMap(self):
        if False:
            print('Hello World!')
        with ops.device(self._device0):
            dataset0 = dataset_ops.Dataset.range(100).map(lambda x: x * 2)
        replicated_ds = distribute.replicate(dataset0, [self._device1, self._device2])
        dataset1 = replicated_ds[self._device1]
        dataset2 = replicated_ds[self._device2]
        with ops.device(self._device0):
            self.assertDatasetProduces(dataset0, range(0, 200, 2))
        with ops.device(self._device1):
            self.assertDatasetProduces(dataset1, range(0, 200, 2))
        with ops.device(self._device2):
            self.assertDatasetProduces(dataset2, range(0, 200, 2))

    @combinations.generate(combinations.combine(tf_api_version=[2], mode=['eager']))
    def testVariableInput(self):
        if False:
            print('Hello World!')
        with ops.device(self._device0):
            counter_var = variable_scope.get_variable('counter', (), dtypes.int32, use_resource=True)
            dataset0 = dataset_ops.Dataset.range(100).map(lambda _: counter_var.assign_add(1))
        replicated_ds = distribute.replicate(dataset0, [self._device1, self._device2])
        dataset1 = replicated_ds[self._device1]
        dataset2 = replicated_ds[self._device2]
        with ops.device(self._device0):
            self.assertDatasetProduces(dataset0, range(1, 101), requires_initialization=True)
        with ops.device(self._device1):
            self.assertDatasetProduces(dataset1, range(1, 101), requires_initialization=True)
        with ops.device(self._device2):
            self.assertDatasetProduces(dataset2, range(1, 101), requires_initialization=True)

class GraphClusterReplicateTest(test_base.DatasetTestBase, parameterized.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(GraphClusterReplicateTest, self).setUp()
        worker_config = config_pb2.ConfigProto()
        worker_config.device_count['CPU'] = 2
        (worker, _) = test_util.create_local_cluster(3, 0, worker_config=worker_config)
        self._device0 = '/job:worker/replica:0/task:0/device:CPU:0'
        self._device1 = '/job:worker/replica:0/task:1/device:CPU:0'
        self._device2 = '/job:worker/replica:0/task:2/device:CPU:0'
        self._target = worker[0].target

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        super().tearDown()
        ops.device(None).__enter__()
        context._reset_context()

    @combinations.generate(combinations.combine(tf_api_version=[1], mode=['graph']))
    def testBasic(self):
        if False:
            while True:
                i = 10
        with ops.device(self._device0):
            dataset0 = dataset_ops.Dataset.range(100)
        replicated_ds = distribute.replicate(dataset0, [self._device1, self._device2])
        dataset1 = replicated_ds[self._device1]
        dataset2 = replicated_ds[self._device2]
        with ops.device(self._device0):
            get_next = self.getNext(dataset0)
        with ops.device(self._device1):
            get_next1 = self.getNext(dataset1)
        with ops.device(self._device2):
            get_next2 = self.getNext(dataset2)
        with session.Session(self._target) as sess:
            for i in range(100):
                self.assertEqual(i, sess.run(get_next()))
                self.assertEqual(i, sess.run(get_next1()))
                self.assertEqual(i, sess.run(get_next2()))

    @combinations.generate(combinations.combine(tf_api_version=[1], mode=['graph']))
    def testMap(self):
        if False:
            print('Hello World!')
        with ops.device(self._device0):
            dataset0 = dataset_ops.Dataset.range(100).map(lambda x: x * 2)
        replicated_ds = distribute.replicate(dataset0, [self._device1, self._device2])
        dataset1 = replicated_ds[self._device1]
        dataset2 = replicated_ds[self._device2]
        with ops.device(self._device0):
            get_next = self.getNext(dataset0)
        with ops.device(self._device1):
            get_next1 = self.getNext(dataset1)
        with ops.device(self._device2):
            get_next2 = self.getNext(dataset2)
        with session.Session(self._target) as sess:
            for i in range(100):
                self.assertEqual(i * 2, sess.run(get_next()))
                self.assertEqual(i * 2, sess.run(get_next1()))
                self.assertEqual(i * 2, sess.run(get_next2()))

    @combinations.generate(combinations.combine(tf_api_version=[1], mode=['graph']))
    def testVariableInput(self):
        if False:
            while True:
                i = 10
        with ops.device(self._device0):
            counter_var = variable_scope.get_variable('counter', (), dtypes.int32, use_resource=True)
            dataset0 = dataset_ops.Dataset.range(100).map(lambda _: counter_var.assign_add(1))
        replicated_ds = distribute.replicate(dataset0, [self._device1, self._device2])
        dataset1 = replicated_ds[self._device1]
        with ops.device(self._device1):
            it1 = dataset_ops.make_initializable_iterator(dataset1)
        with session.Session(self._target) as sess:
            with self.assertRaises(errors.OpError):
                sess.run(it1.initializer)
if __name__ == '__main__':
    test.main()