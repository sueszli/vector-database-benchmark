"""Tests for TFCONFIGClusterResolver."""
import os
from tensorflow.python.client import session
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib
mock = test.mock

@test_util.run_all_in_graph_and_eager_modes
class TFConfigClusterResolverTest(test.TestCase):

    def _verifyClusterSpecEquality(self, cluster_spec, expected_proto):
        if False:
            i = 10
            return i + 15
        self.assertProtoEquals(expected_proto, cluster_spec.as_cluster_def())
        self.assertProtoEquals(expected_proto, server_lib.ClusterSpec(cluster_spec).as_cluster_def())
        self.assertProtoEquals(expected_proto, server_lib.ClusterSpec(cluster_spec.as_cluster_def()).as_cluster_def())
        self.assertProtoEquals(expected_proto, server_lib.ClusterSpec(cluster_spec.as_dict()).as_cluster_def())

    def testNormalClusterSpecRead(self):
        if False:
            while True:
                i = 10
        os.environ['TF_CONFIG'] = '\n    {\n      "cluster": {\n        "ps": ["ps0:2222", "ps1:2222"],\n        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]\n      },\n      "task": {\n        "type": "ps",\n        "index": 0\n      }\n    }\n    '
        cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
        expected_proto = "\n    job { name: 'ps' tasks { key: 0 value: 'ps0:2222' }\n                     tasks { key: 1 value: 'ps1:2222' } }\n    job { name: 'worker' tasks { key: 0 value: 'worker0:2222' }\n                         tasks { key: 1 value: 'worker1:2222' }\n                         tasks { key: 2 value: 'worker2:2222' } }\n    "
        actual_cluster_spec = cluster_resolver.cluster_spec()
        self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

    def testSparseClusterSpecRead(self):
        if False:
            i = 10
            return i + 15
        os.environ['TF_CONFIG'] = '\n    {\n      "cluster": {\n        "ps": ["ps0:2222", "ps1:2222"],\n        "worker": {"1": "worker1:2222"}\n      },\n      "task": {\n        "type": "worker",\n        "index": 1\n      }\n    }\n    '
        cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
        expected_proto = "\n    job { name: 'ps' tasks { key: 0 value: 'ps0:2222' }\n                     tasks { key: 1 value: 'ps1:2222' } }\n    job { name: 'worker' tasks { key: 1 value: 'worker1:2222' } }\n    "
        actual_cluster_spec = cluster_resolver.cluster_spec()
        self._verifyClusterSpecEquality(actual_cluster_spec, expected_proto)

    def testAutomaticMasterRead(self):
        if False:
            while True:
                i = 10
        os.environ['TF_CONFIG'] = '\n    {\n      "cluster": {\n        "ps": ["ps0:2222", "ps1:2222"],\n        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]\n      },\n      "task": {\n        "type": "ps",\n        "index": 0\n      }\n    }\n    '
        cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
        self.assertEqual('ps0:2222', cluster_resolver.master())

    def testSpecifiedTaskTypeAndIndexMasterRead(self):
        if False:
            i = 10
            return i + 15
        os.environ['TF_CONFIG'] = '\n    {\n      "cluster": {\n        "ps": ["ps0:2222", "ps1:2222"],\n        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]\n      },\n      "task": {\n        "type": "ps",\n        "index": 0\n      }\n    }\n    '
        cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
        self.assertEqual('worker1:2222', cluster_resolver.master('worker', 1))

    def testSessionMasterRead(self):
        if False:
            print('Hello World!')
        os.environ['TF_CONFIG'] = '\n    {\n      "cluster": {\n        "ps": ["ps0:2222", "ps1:2222"],\n        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]\n      },\n      "session_master": "sessionmaster:2222",\n      "task": {\n        "type": "ps",\n        "index": 0\n      }\n    }\n    '
        cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
        self.assertEqual('sessionmaster:2222', cluster_resolver.master())

    def testRpcLayerRead(self):
        if False:
            while True:
                i = 10
        os.environ['TF_CONFIG'] = '\n    {\n      "cluster": {\n        "ps": ["ps0:2222", "ps1:2222"],\n        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]\n      },\n      "rpc_layer": "grpc",\n      "task": {\n        "type": "ps",\n        "index": 0\n      }\n    }\n    '
        cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
        self.assertEqual('grpc://ps0:2222', cluster_resolver.master())

    def testTaskTypeIndexRpcRead(self):
        if False:
            while True:
                i = 10
        os.environ['TF_CONFIG'] = '\n    {\n      "cluster": {\n        "ps": ["ps0:2222", "ps1:2222"],\n        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]\n      },\n      "rpc_layer": "grpc",\n      "task": {\n        "type": "ps",\n        "index": 0\n      }\n    }\n    '
        cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
        self.assertEqual('ps', cluster_resolver.task_type)
        self.assertEqual(0, cluster_resolver.task_id)
        self.assertEqual('grpc', cluster_resolver.rpc_layer)

    def testParameterOverrides(self):
        if False:
            i = 10
            return i + 15
        os.environ['TF_CONFIG'] = '\n    {\n      "cluster": {\n        "ps": ["ps0:2222", "ps1:2222"],\n        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]\n      },\n      "rpc_layer": "grpc",\n      "task": {\n        "type": "ps",\n        "index": 1\n      }\n    }\n    '
        cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver(task_type='ps', task_id=0)
        self.assertEqual('grpc://ps0:2222', cluster_resolver.master())
        self.assertEqual('ps', cluster_resolver.task_type)
        self.assertEqual(0, cluster_resolver.task_id)
        cluster_resolver.task_type = 'worker'
        cluster_resolver.task_id = 1
        cluster_resolver.rpc_layer = 'test'
        self.assertEqual('test://worker1:2222', cluster_resolver.master())
        self.assertEqual('worker', cluster_resolver.task_type)
        self.assertEqual(1, cluster_resolver.task_id)
        self.assertEqual('test', cluster_resolver.rpc_layer)

    def testTaskTypeCastToString(self):
        if False:
            print('Hello World!')
        os.environ['TF_CONFIG'] = '\n    {\n      "cluster": {\n        "123456": ["ps0:2222", "ps1:2222"],\n        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]\n      },\n      "rpc_layer": "grpc",\n      "task": {\n        "type": 123456,\n        "index": 0\n      }\n    }\n    '
        cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
        self.assertEqual('123456', cluster_resolver.task_type)

    def testTaskIndexCastToInteger(self):
        if False:
            while True:
                i = 10
        os.environ['TF_CONFIG'] = '\n    {\n      "cluster": {\n        "ps": ["ps0:2222", "ps1:2222"],\n        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]\n      },\n      "rpc_layer": "grpc",\n      "task": {\n        "type": "ps",\n        "index": "1"\n      }\n    }\n    '
        cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
        self.assertEqual(1, cluster_resolver.task_id)

    def testTaskIndexOverride(self):
        if False:
            print('Hello World!')
        os.environ['TF_CONFIG'] = '\n    {\n      "cluster": {\n        "worker": ["worker0:2222", "worker1:2222"]\n      },\n      "task": {\n        "type": "worker",\n        "index": "0"\n      }\n    }\n    '
        cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver(task_id=1)
        self.assertEqual(1, cluster_resolver.task_id)

    def testZeroItemsInClusterSpecMasterRead(self):
        if False:
            print('Hello World!')
        os.environ['TF_CONFIG'] = '\n    {}\n    '
        cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
        self.assertEqual('', cluster_resolver.master())

    def testOneItemInClusterSpecMasterRead(self):
        if False:
            return 10
        os.environ['TF_CONFIG'] = '\n    {\n      "cluster": {\n        "worker": ["worker0:2222"]\n      }\n    }\n    '
        cluster_resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
        self.assertEqual('', cluster_resolver.master())

    @mock.patch.object(config, 'list_logical_devices')
    @mock.patch.object(session.BaseSession, 'list_devices')
    def testNumAcceleratorsFilterTasksByEnvVar(self, mock_list_devices, mock_eager_list_devices):
        if False:
            i = 10
            return i + 15
        os.environ['TF_CONFIG'] = '\n    {\n      "cluster": {\n        "worker1": ["w10:2222"],\n        "worker2": ["w21:2222", "w22:2222", "w23:2222", "w24:2222"]\n      },\n      "rpc_layer": "grpc",\n      "task": {\n        "type": "worker1",\n        "index": "0"\n      }\n    }\n    '
        devices = [context.LogicalDevice('/job:worker1/task:0/device:TPU:0', 'TPU'), context.LogicalDevice('/job:worker1/task:0/device:TPU:1', 'TPU'), context.LogicalDevice('/job:worker1/task:0/device:GPU:0', 'GPU'), context.LogicalDevice('/job:worker1/task:0/device:GPU:1', 'GPU'), context.LogicalDevice('/job:worker2/task:1/device:TPU:2', 'TPU'), context.LogicalDevice('/job:worker2/task:2/device:TPU:3', 'TPU'), context.LogicalDevice('/job:worker2/task:3/device:GPU:2', 'GPU'), context.LogicalDevice('/job:worker2/task:4/device:GPU:3', 'GPU')]
        device_list = [session._DeviceAttributes(d.name, d.device_type, 1024, 0) for d in devices]
        mock_eager_list_devices.return_value = devices
        mock_list_devices.return_value = device_list
        resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
        self.assertEqual(resolver.num_accelerators(), {'TPU': 2, 'GPU': 2})
        self.assertEqual(resolver.num_accelerators(task_type='worker2', task_id=3), {'GPU': 1})
if __name__ == '__main__':
    test.main()