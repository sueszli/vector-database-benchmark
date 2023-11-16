"""Tests for device utilities."""
from absl.testing import parameterized
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib

class DeviceUtilTest(test.TestCase, parameterized.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(DeviceUtilTest, self).setUp()
        context._reset_context()

    @combinations.generate(combinations.combine(mode='graph'))
    def testCurrentDeviceWithGlobalGraph(self):
        if False:
            return 10
        with ops.device('/cpu:0'):
            self.assertEqual(device_util.current(), '/device:CPU:0')
        with ops.device('/job:worker'):
            with ops.device('/cpu:0'):
                self.assertEqual(device_util.current(), '/job:worker/device:CPU:0')
        with ops.device('/cpu:0'):
            with ops.device('/gpu:0'):
                self.assertEqual(device_util.current(), '/device:GPU:0')

    def testCurrentDeviceWithNonGlobalGraph(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            with ops.device('/cpu:0'):
                self.assertEqual(device_util.current(), '/device:CPU:0')

    def testCurrentDeviceWithEager(self):
        if False:
            for i in range(10):
                print('nop')
        with context.eager_mode():
            with ops.device('/cpu:0'):
                self.assertEqual(device_util.current(), '/job:localhost/replica:0/task:0/device:CPU:0')

    @combinations.generate(combinations.combine(mode=['graph', 'eager']))
    def testCanonicalizeWithoutDefaultDevice(self, mode):
        if False:
            while True:
                i = 10
        if mode == 'graph':
            self.assertEqual(device_util.canonicalize('/cpu:0'), '/replica:0/task:0/device:CPU:0')
        else:
            self.assertEqual(device_util.canonicalize('/cpu:0'), '/job:localhost/replica:0/task:0/device:CPU:0')
        self.assertEqual(device_util.canonicalize('/job:worker/cpu:0'), '/job:worker/replica:0/task:0/device:CPU:0')
        self.assertEqual(device_util.canonicalize('/job:worker/task:1/cpu:0'), '/job:worker/replica:0/task:1/device:CPU:0')

    @combinations.generate(combinations.combine(mode=['eager']))
    def testCanonicalizeWithoutDefaultDeviceCollectiveEnabled(self):
        if False:
            for i in range(10):
                print('nop')
        cluster_spec = server_lib.ClusterSpec(multi_worker_test_base.create_cluster_spec(has_chief=False, num_workers=1, num_ps=0, has_eval=False))
        server_def = tensorflow_server_pb2.ServerDef(cluster=cluster_spec.as_cluster_def(), job_name='worker', task_index=0, protocol='grpc', port=0)
        context.context().enable_collective_ops(server_def)
        self.assertEqual(device_util.canonicalize('/cpu:0'), '/job:worker/replica:0/task:0/device:CPU:0')

    def testCanonicalizeWithDefaultDevice(self):
        if False:
            print('Hello World!')
        self.assertEqual(device_util.canonicalize('/job:worker/task:1/cpu:0', default='/gpu:0'), '/job:worker/replica:0/task:1/device:CPU:0')
        self.assertEqual(device_util.canonicalize('/job:worker/task:1', default='/gpu:0'), '/job:worker/replica:0/task:1/device:GPU:0')
        self.assertEqual(device_util.canonicalize('/cpu:0', default='/job:worker'), '/job:worker/replica:0/task:0/device:CPU:0')
        self.assertEqual(device_util.canonicalize('/job:worker/replica:0/task:1/device:CPU:0', default='/job:chief/replica:0/task:1/device:CPU:0'), '/job:worker/replica:0/task:1/device:CPU:0')

    def testResolveWithDeviceScope(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.device('/gpu:0'):
            self.assertEqual(device_util.resolve('/job:worker/task:1/cpu:0'), '/job:worker/replica:0/task:1/device:CPU:0')
            self.assertEqual(device_util.resolve('/job:worker/task:1'), '/job:worker/replica:0/task:1/device:GPU:0')
        with ops.device('/job:worker'):
            self.assertEqual(device_util.resolve('/cpu:0'), '/job:worker/replica:0/task:0/device:CPU:0')
if __name__ == '__main__':
    test.main()