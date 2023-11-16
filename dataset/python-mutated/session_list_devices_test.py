"""Tests for tensorflow.python.client.session.Session's list_devices API."""
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import pywrap_tf_session as tf_session
from tensorflow.python.client import session
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.training import server_lib

class SessionListDevicesTest(test_util.TensorFlowTestCase):

    def testListDevices(self):
        if False:
            i = 10
            return i + 15
        with session.Session() as sess:
            devices = sess.list_devices()
            self.assertTrue('/job:localhost/replica:0/task:0/device:CPU:0' in set([d.name for d in devices]), devices)
            self.assertTrue(all((d.incarnation != 0 for d in devices)))

    def testInvalidDeviceNumber(self):
        if False:
            return 10
        opts = tf_session.TF_NewSessionOptions()
        with ops.get_default_graph()._c_graph.get() as c_graph:
            c_session = tf_session.TF_NewSession(c_graph, opts)
        raw_device_list = tf_session.TF_SessionListDevices(c_session)
        size = tf_session.TF_DeviceListCount(raw_device_list)
        with self.assertRaises(errors.InvalidArgumentError):
            tf_session.TF_DeviceListMemoryBytes(raw_device_list, size)
        tf_session.TF_DeleteDeviceList(raw_device_list)
        tf_session.TF_CloseSession(c_session)

    def testListDevicesGrpcSession(self):
        if False:
            return 10
        server = server_lib.Server.create_local_server()
        with session.Session(server.target) as sess:
            devices = sess.list_devices()
            self.assertTrue('/job:localhost/replica:0/task:0/device:CPU:0' in set([d.name for d in devices]), devices)
            self.assertTrue(all((d.incarnation != 0 for d in devices)))

    def testListDevicesClusterSpecPropagation(self):
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
        with session.Session(server1.target, config=config) as sess:
            devices = sess.list_devices()
            device_names = set((d.name for d in devices))
            self.assertTrue('/job:worker/replica:0/task:0/device:CPU:0' in device_names)
            self.assertTrue('/job:worker/replica:0/task:1/device:CPU:0' in device_names)
            self.assertTrue(all((d.incarnation != 0 for d in devices)))
if __name__ == '__main__':
    googletest.main()