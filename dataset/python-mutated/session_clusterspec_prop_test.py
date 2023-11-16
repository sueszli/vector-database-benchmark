"""Tests for tensorflow.python.client.session.Session's ClusterSpec Propagation.

These tests exercise the ClusterSpec Propagation capabilities of distributed
Sessions.
"""
import numpy as np
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib

class SessionClusterSpecPropagationTest(test_util.TensorFlowTestCase):

    def testClusterSpecPropagationSimple(self):
        if False:
            print('Hello World!')
        server1 = server_lib.Server.create_local_server()
        server2 = server_lib.Server.create_local_server()
        cluster_def = cluster_pb2.ClusterDef()
        job = cluster_def.job.add()
        job.name = 'worker'
        job.tasks[0] = server1.target[len('grpc://'):]
        job.tasks[1] = server2.target[len('grpc://'):]
        config = config_pb2.ConfigProto(cluster_def=cluster_def)
        const = constant_op.constant(17)
        sess = session.Session(server1.target, config=config)
        output = self.evaluate(const)
        self.assertEqual(17, output)

    def testClusterSpecPropagationWorker2Placement(self):
        if False:
            return 10
        server1 = server_lib.Server.create_local_server()
        server2 = server_lib.Server.create_local_server()
        cluster_def = cluster_pb2.ClusterDef()
        job = cluster_def.job.add()
        job.name = 'worker'
        job.tasks[0] = server1.target[len('grpc://'):]
        job.tasks[1] = server2.target[len('grpc://'):]
        config = config_pb2.ConfigProto(cluster_def=cluster_def)
        with ops.Graph().as_default() as g, ops.device('/job:worker/task:1'):
            with ops.device('/cpu:0'):
                const = constant_op.constant(17)
        sess = session.Session(server1.target, config=config, graph=g)
        run_options = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
        run_metadata = config_pb2.RunMetadata()
        output = sess.run(const, options=run_options, run_metadata=run_metadata)
        self.assertEqual(17, output)
        self.assertEqual(1, len([node_stats for dev_stats in run_metadata.step_stats.dev_stats for node_stats in dev_stats.node_stats if '/job:worker/replica:0/task:1/device:CPU:0' == dev_stats.device and 'Const' == node_stats.node_name]))

    def testClusterSpecPropagationWorker1Placement(self):
        if False:
            print('Hello World!')
        server1 = server_lib.Server.create_local_server()
        server2 = server_lib.Server.create_local_server()
        cluster_def = cluster_pb2.ClusterDef()
        job = cluster_def.job.add()
        job.name = 'worker'
        job.tasks[0] = server1.target[len('grpc://'):]
        job.tasks[1] = server2.target[len('grpc://'):]
        config = config_pb2.ConfigProto(cluster_def=cluster_def)
        with ops.Graph().as_default() as g, ops.device('/job:worker/task:0'):
            const = constant_op.constant(17)
        with session.Session(server1.target, config=config, graph=g):
            output = self.evaluate(const)
        self.assertEqual(17, output)

    def testCanonicalDeviceNames(self):
        if False:
            return 10
        server1 = server_lib.Server.create_local_server()
        server2 = server_lib.Server.create_local_server()
        cluster_def = cluster_pb2.ClusterDef()
        job = cluster_def.job.add()
        job.name = 'worker'
        job.tasks[0] = server1.target[len('grpc://'):]
        job.tasks[1] = server2.target[len('grpc://'):]
        config = config_pb2.ConfigProto(cluster_def=cluster_def)
        with ops.Graph().as_default() as g, ops.device('/job:worker/task:1/device:CPU:0'):
            const = constant_op.constant(17)
        sess = session.Session(server1.target, config=config, graph=g)
        run_options = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
        run_metadata = config_pb2.RunMetadata()
        output = sess.run(const, options=run_options, run_metadata=run_metadata)
        self.assertEqual(17, output)
        self.assertEqual(1, len([node_stats for dev_stats in run_metadata.step_stats.dev_stats for node_stats in dev_stats.node_stats if '/job:worker/replica:0/task:1/device:CPU:0' == dev_stats.device and 'Const' == node_stats.node_name]))

    def testFullDeviceNames(self):
        if False:
            while True:
                i = 10
        server1 = server_lib.Server.create_local_server()
        server2 = server_lib.Server.create_local_server()
        cluster_def = cluster_pb2.ClusterDef()
        job = cluster_def.job.add()
        job.name = 'renamed_worker'
        job.tasks[0] = server1.target[len('grpc://'):]
        job.tasks[1] = server2.target[len('grpc://'):]
        config = config_pb2.ConfigProto(cluster_def=cluster_def)
        with ops.Graph().as_default() as g, ops.device('/job:renamed_worker/replica:0/task:1/device:CPU:0'):
            const = constant_op.constant(17)
        sess = session.Session(server1.target, config=config, graph=g)
        run_options = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
        run_metadata = config_pb2.RunMetadata()
        output = sess.run(const, options=run_options, run_metadata=run_metadata)
        self.assertEqual(17, output)
        self.assertEqual(1, len([node_stats for dev_stats in run_metadata.step_stats.dev_stats for node_stats in dev_stats.node_stats if '/job:renamed_worker/replica:0/task:1/device:CPU:0' == dev_stats.device and 'Const' == node_stats.node_name]))

    def testMultipleLocalDevices(self):
        if False:
            print('Hello World!')
        server_config = config_pb2.ConfigProto(device_count={'CPU': 2})
        server1 = server_lib.Server.create_local_server(config=server_config)
        server2 = server_lib.Server.create_local_server(config=server_config)
        cluster_def = cluster_pb2.ClusterDef()
        job = cluster_def.job.add()
        job.name = 'worker'
        job.tasks[0] = server1.target[len('grpc://'):]
        job.tasks[1] = server2.target[len('grpc://'):]
        config = config_pb2.ConfigProto(cluster_def=cluster_def)
        with ops.Graph().as_default() as g:
            with ops.device('/job:worker/task:1/cpu:1'):
                input1 = constant_op.constant(17, dtypes.float32)
            with ops.device('/job:worker/task:0/cpu:1'):
                input2 = constant_op.constant(3, dtypes.float32)
            with ops.device('/job:worker/task:1/cpu:0'):
                sum1 = input1 + input2
            if test.is_gpu_available():
                device_str = '/job:worker/task:0/device:GPU:0'
            else:
                device_str = '/job:worker/task:0/cpu:1'
            with ops.device(device_str):
                sum2 = input2 + input1
            with ops.device('/job:worker/task:0/cpu:0'):
                sum3 = sum1 + sum2
        with session.Session(server1.target, config=config, graph=g):
            output = self.evaluate(sum3)
        self.assertEqual(40, output)

    def testLegacyDeviceNames(self):
        if False:
            print('Hello World!')
        server1 = server_lib.Server.create_local_server()
        server2 = server_lib.Server.create_local_server()
        cluster_def = cluster_pb2.ClusterDef()
        job = cluster_def.job.add()
        job.name = 'worker'
        job.tasks[0] = server1.target[len('grpc://'):]
        job.tasks[1] = server2.target[len('grpc://'):]
        config = config_pb2.ConfigProto(cluster_def=cluster_def)
        with ops.Graph().as_default() as g, ops.device('/job:worker/task:1/cpu:0'):
            const = constant_op.constant(17)
        sess = session.Session(server1.target, config=config, graph=g)
        run_options = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
        run_metadata = config_pb2.RunMetadata()
        output = sess.run(const, options=run_options, run_metadata=run_metadata)
        self.assertEqual(17, output)
        self.assertEqual(1, len([node_stats for dev_stats in run_metadata.step_stats.dev_stats for node_stats in dev_stats.node_stats if '/job:worker/replica:0/task:1/device:CPU:0' == dev_stats.device and 'Const' == node_stats.node_name]))

    def testClusterSpecPropagationThreeServers2Graphs(self):
        if False:
            i = 10
            return i + 15
        'Boots 3 servers, creates 2 sessions, ensures appropriate operations.\n\n    We create 2 clusterspecs:\n     1. server2 as the master, server1 as a worker\n     2. server2 as the master, server3 as a worker\n\n    We ensure that variables on the workers are independent.\n    '
        server1 = server_lib.Server.create_local_server()
        server2 = server_lib.Server.create_local_server()
        server3 = server_lib.Server.create_local_server()
        cluster_def1 = cluster_pb2.ClusterDef()
        job1 = cluster_def1.job.add()
        job1.name = 'worker1'
        job1.tasks[0] = server2.target[len('grpc://'):]
        job1.tasks[1] = server1.target[len('grpc://'):]
        cluster_def2 = cluster_pb2.ClusterDef()
        job2 = cluster_def2.job.add()
        job2.name = 'worker2'
        job2.tasks[0] = server2.target[len('grpc://'):]
        job2.tasks[1] = server3.target[len('grpc://'):]
        config1 = config_pb2.ConfigProto(cluster_def=cluster_def1)
        config2 = config_pb2.ConfigProto(cluster_def=cluster_def2)
        with ops.Graph().as_default() as g1:
            with ops.device('/job:worker1/task:1'):
                var1 = variables.Variable(array_ops.zeros([2]), name='var1')
                update_op1 = state_ops.assign_add(var1, array_ops.ones([2]), name='var1_assign_add')
                init1 = variables.global_variables_initializer()
        with ops.Graph().as_default() as g2:
            with ops.device('/job:worker2/task:1'):
                var2 = variables.Variable(array_ops.zeros([2]), name='var2')
                update_op2 = state_ops.assign_add(var2, array_ops.ones([2]), name='var2_assign_add')
                init2 = variables.global_variables_initializer()
        sess1 = session.Session(server2.target, graph=g1, config=config1)
        sess2 = session.Session(server2.target, graph=g2, config=config2)
        init1.run(session=sess1)
        init2.run(session=sess2)
        expected_zeros = np.zeros([2])
        expected_ones = np.ones([2])
        self.assertAllEqual(expected_zeros, sess1.run(var1))
        self.assertAllEqual(expected_zeros, sess2.run(var2))
        self.assertAllEqual(expected_ones, sess1.run(update_op1))
        self.assertAllEqual(expected_ones, sess1.run(var1))
        self.assertAllEqual(expected_zeros, sess2.run(var2))
        self.assertAllEqual(expected_ones, sess2.run(update_op2))
        self.assertAllEqual(expected_ones + expected_ones, sess1.run(update_op1))
        self.assertAllEqual(expected_ones, sess2.run(var2))
        self.assertAllEqual(expected_ones + expected_ones, sess1.run(var1))

    def testClusterSpecPropagationThreeServers(self):
        if False:
            print('Hello World!')
        'Boots 3 servers, creates 2 sessions, ensures appropriate operations.\n\n    We create 2 clusterspecs:\n     1. server2 as the master, server1 as a worker\n     2. server2 as the master, server3 as a worker\n\n    We ensure that variables on the workers are independent.\n    '
        server1 = server_lib.Server.create_local_server()
        server2 = server_lib.Server.create_local_server()
        server3 = server_lib.Server.create_local_server()
        cluster_def1 = cluster_pb2.ClusterDef()
        job1 = cluster_def1.job.add()
        job1.name = 'worker'
        job1.tasks[0] = server2.target[len('grpc://'):]
        job1.tasks[1] = server1.target[len('grpc://'):]
        cluster_def2 = cluster_pb2.ClusterDef()
        job2 = cluster_def2.job.add()
        job2.name = 'worker'
        job2.tasks[0] = server2.target[len('grpc://'):]
        job2.tasks[1] = server3.target[len('grpc://'):]
        config1 = config_pb2.ConfigProto(cluster_def=cluster_def1)
        config2 = config_pb2.ConfigProto(cluster_def=cluster_def2)
        with ops.device('/job:worker/task:1'):
            var = variables.Variable(array_ops.zeros([2]), name='var')
            feed = array_ops.placeholder(dtypes.float32, shape=2)
            update_op = var.assign_add(feed)
        sess1 = session.Session(server2.target, config=config1)
        sess2 = session.Session(server2.target, config=config2)
        variables.global_variables_initializer().run(session=sess1)
        variables.global_variables_initializer().run(session=sess2)
        expected_zeros = np.zeros([2])
        expected_ones = np.ones([2])
        self.assertAllEqual(expected_zeros, sess1.run(var))
        self.assertAllEqual(expected_zeros, sess2.run(var))
        self.assertAllEqual(expected_ones, sess1.run(update_op, feed_dict={feed: expected_ones}))
        self.assertAllEqual(expected_ones, sess1.run(var))
        self.assertAllEqual(expected_zeros, sess2.run(var))
        self.assertAllEqual(expected_ones, sess2.run(update_op, feed_dict={feed: expected_ones}))
        self.assertAllEqual(expected_ones + expected_ones, sess1.run(update_op, feed_dict={feed: expected_ones}))
        self.assertAllEqual(expected_ones, sess2.run(var))
        self.assertAllEqual(expected_ones + expected_ones, sess1.run(var))

    def testClusterSpecPropagationThreeServersOneCluster(self):
        if False:
            for i in range(10):
                print('nop')
        'Boots 3 servers, ensures appropriate communication across workers.\n\n    Additionally, in this cluster, we ensure the master is not the 0-th worker.\n\n    Note: this test only uses one session.\n    '
        server1 = server_lib.Server.create_local_server()
        server2 = server_lib.Server.create_local_server()
        server3 = server_lib.Server.create_local_server()
        cluster_def = cluster_pb2.ClusterDef()
        job = cluster_def.job.add()
        job.name = 'worker'
        job.tasks[0] = server3.target[len('grpc://'):]
        job.tasks[1] = server2.target[len('grpc://'):]
        job.tasks[2] = server1.target[len('grpc://'):]
        config = config_pb2.ConfigProto(cluster_def=cluster_def)
        with ops.device('/job:worker/task:1'):
            feed1 = array_ops.placeholder(dtypes.float32, shape=2)
            const1 = constant_op.constant(2.0)
            mul1 = const1 * feed1
        with ops.device('/job:worker/task:2'):
            feed2 = array_ops.placeholder(dtypes.float32, shape=2)
            const2 = constant_op.constant(2.0)
            mul2 = const2 * feed2
        with ops.device('/job:worker/task:0'):
            feed0 = array_ops.placeholder(dtypes.float32, shape=2)
            const0 = constant_op.constant(2.0)
            mul0 = const0 * feed0
        sum_op = mul0 + mul1 + mul2
        ones = np.ones([2])
        run_options = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
        run_metadata = config_pb2.RunMetadata()
        with session.Session(server1.target, config=config) as sess:
            output = sess.run(sum_op, options=run_options, run_metadata=run_metadata, feed_dict={feed1: ones, feed2: ones, feed0: ones})
            self.assertAllEqual(6 * ones, output)
            self.assertEqual(3, len([dev_stats.device for dev_stats in run_metadata.step_stats.dev_stats for node_stats in dev_stats.node_stats if '/job:worker/replica:0/task:' in dev_stats.device and node_stats.node_name.startswith('Const')]), run_metadata)

    def testClusterSpecPropagationIsolation(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that two sessions using ClusterSpec propagation are isolated.'
        server = server_lib.Server.create_local_server()
        init_value = array_ops.placeholder(dtypes.int32, shape=[])
        v = variables.Variable(init_value)
        cluster_def = cluster_pb2.ClusterDef()
        job = cluster_def.job.add()
        job.name = 'worker'
        job.tasks[0] = server.target[len('grpc://'):]
        config = config_pb2.ConfigProto(cluster_def=cluster_def)
        sess1 = session.Session(server.target, config=config)
        sess2 = session.Session(server.target, config=config)
        with self.assertRaises(errors.FailedPreconditionError):
            sess1.run(v)
        with self.assertRaises(errors.FailedPreconditionError):
            sess2.run(v)
        sess1.run(v.initializer, feed_dict={init_value: 37})
        self.assertEqual(37, sess1.run(v))
        with self.assertRaises(errors.FailedPreconditionError):
            sess2.run(v)
        sess2.run(v.initializer, feed_dict={init_value: 86})
        self.assertEqual(37, sess1.run(v))
        self.assertEqual(86, sess2.run(v))
        sess2.close()
        self.assertEqual(37, sess1.run(v))
        sess3 = session.Session(server.target, config=config)
        self.assertEqual(37, sess1.run(v))
        with self.assertRaises(errors.FailedPreconditionError):
            sess3.run(v)

    def testClusterSpecPropagationNonIsolation(self):
        if False:
            print('Hello World!')
        'Test that two sessions using ClusterSpec propagation shares state.\n\n    For example, the updated Variable value are visible among all worker\n    sessions registered in the same server.\n    '
        server = server_lib.Server.create_local_server()
        init_value = array_ops.placeholder(dtypes.int32, shape=[])
        v = variables.Variable(init_value)
        cluster_def = cluster_pb2.ClusterDef()
        job = cluster_def.job.add()
        job.name = 'worker'
        job.tasks[0] = server.target[len('grpc://'):]
        config = config_pb2.ConfigProto(cluster_def=cluster_def)
        config.experimental.share_session_state_in_clusterspec_propagation = True
        sess1 = session.Session(server.target, config=config)
        sess2 = session.Session(server.target, config=config)
        with self.assertRaises(errors.FailedPreconditionError):
            sess1.run(v)
        with self.assertRaises(errors.FailedPreconditionError):
            sess2.run(v)
        sess1.run(v.initializer, feed_dict={init_value: 37})
        self.assertEqual(37, sess1.run(v))
        self.assertEqual(37, sess2.run(v))
        sess2.close()
        self.assertEqual(37, sess1.run(v))
        sess3 = session.Session(server.target, config=config)
        self.assertEqual(37, sess1.run(v))
        self.assertEqual(37, sess3.run(v))

    def testClusterSpecPropagationNonIsolation2Graphs(self):
        if False:
            while True:
                i = 10
        'Creates 2 sessions with each own graph, ensures appropriate operations.\n\n    We ensure that variables on the workers shares state.\n    '
        server = server_lib.Server.create_local_server()
        cluster_def = cluster_pb2.ClusterDef()
        job = cluster_def.job.add()
        job.name = 'worker'
        job.tasks[0] = server.target[len('grpc://'):]
        config = config_pb2.ConfigProto(cluster_def=cluster_def)
        config.experimental.share_session_state_in_clusterspec_propagation = True
        with ops.Graph().as_default() as g1:
            var1 = variables.Variable(array_ops.zeros([2]), name='var')
            update_op1 = state_ops.assign_add(var1, array_ops.ones([2]), name='var1_assign_add')
            init1 = variables.global_variables_initializer()
        with ops.Graph().as_default() as g2:
            var2 = variables.Variable(array_ops.zeros([2]), name='var')
            update_op2 = state_ops.assign_add(var2, array_ops.ones([2]), name='var2_assign_add')
        sess1 = session.Session(server.target, graph=g1, config=config)
        sess2 = session.Session(server.target, graph=g2, config=config)
        expected_zeros = np.zeros([2])
        expected_ones = np.ones([2])
        init1.run(session=sess1)
        self.assertAllEqual(expected_zeros, sess1.run(var1))
        self.assertAllEqual(expected_zeros, sess2.run(var2))
        self.assertAllEqual(expected_ones, sess1.run(update_op1))
        self.assertAllEqual(expected_ones, sess1.run(var1))
        self.assertAllEqual(expected_ones, sess2.run(var2))
        self.assertAllEqual(expected_ones + expected_ones, sess2.run(update_op2))
        self.assertAllEqual(expected_ones + expected_ones, sess2.run(var2))
        self.assertAllEqual(expected_ones + expected_ones, sess1.run(var1))

    def testClusterSpecPropagationPartialRun(self):
        if False:
            for i in range(10):
                print('nop')
        'Test successful partial run with ClusterSpec propagation.'
        server1 = server_lib.Server.create_local_server()
        server2 = server_lib.Server.create_local_server()
        cluster_def = cluster_pb2.ClusterDef()
        job = cluster_def.job.add()
        job.name = 'worker'
        job.tasks[0] = server1.target[len('grpc://'):]
        job.tasks[1] = server2.target[len('grpc://'):]
        config = config_pb2.ConfigProto(cluster_def=cluster_def)
        with ops.device('/job:worker/task:0'):
            a = array_ops.placeholder(dtypes.float32, shape=[])
        with ops.device('/job:worker/task:1'):
            b = array_ops.placeholder(dtypes.float32, shape=[])
            c = array_ops.placeholder(dtypes.float32, shape=[])
            r1 = math_ops.add(a, b)
        with ops.device('/job:worker/task:0'):
            r2 = math_ops.multiply(r1, c)
        with session.Session(server1.target, config=config) as sess:
            h = sess.partial_run_setup([r1, r2], [a, b, c])
            res = sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
            self.assertEqual(3, res)
            res = sess.partial_run(h, r2, feed_dict={c: 3})
            self.assertEqual(9, res)
if __name__ == '__main__':
    googletest.main()