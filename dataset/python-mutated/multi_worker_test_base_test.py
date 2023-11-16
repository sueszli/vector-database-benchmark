"""Tests for multi-process clusters."""
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.eager import context
from tensorflow.python.eager import remote
from tensorflow.python.eager import test

class MultiProcessClusterTest(test.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(MultiProcessClusterTest, self).setUp()
        self._cluster = multi_worker_test_base.create_multi_process_cluster(num_workers=2, num_ps=1, has_chief=True, rpc_layer='grpc')
        remote.connect_to_cluster(self._cluster.cluster_resolver.cluster_spec(), protocol='grpc')
        context.ensure_initialized()

    def testClusterIsAlive(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(context.check_alive('/job:worker/replica:0/task:0'))
        self.assertTrue(context.check_alive('/job:worker/replica:0/task:1'))
        self.assertTrue(context.check_alive('/job:ps/replica:0/task:0'))
        self.assertTrue(context.check_alive('/job:chief/replica:0/task:0'))

    def testKillAndStartTask(self):
        if False:
            print('Hello World!')
        self.assertTrue(context.check_alive('/job:worker/replica:0/task:0'))
        with self.assertRaises(ValueError):
            self._cluster.start_task('worker', 0)
        self._cluster.kill_task('worker', 0)
        self.assertFalse(context.check_alive('/job:worker/replica:0/task:0'))
        with self.assertRaises(ValueError):
            self._cluster.kill_task('worker', 0)
        self._cluster.start_task('worker', 0)
        context.context().update_server_def(context.get_server_def())
        self.assertTrue(context.check_alive('/job:worker/replica:0/task:0'))

    def testStop(self):
        if False:
            return 10
        self._cluster.stop()
        self.assertFalse(context.check_alive('/job:worker/replica:0/task:0'))
        self.assertFalse(context.check_alive('/job:worker/replica:0/task:1'))
        self.assertFalse(context.check_alive('/job:ps/replica:0/task:0'))
        self.assertFalse(context.check_alive('/job:chief/replica:0/task:0'))

    def testClusterResolverProperty(self):
        if False:
            return 10
        cluster_spec = self._cluster.cluster_resolver.cluster_spec().as_dict()
        self.assertEqual(len(cluster_spec['worker']), 2)
        self.assertEqual(len(cluster_spec['ps']), 1)
        self.assertEqual(len(cluster_spec['chief']), 1)
if __name__ == '__main__':
    multi_process_runner.test_main()