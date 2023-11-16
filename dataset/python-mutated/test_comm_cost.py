import json
import os
import tempfile
import unittest
from test_cluster import cluster_json, multi_cluster_json
import paddle
from paddle.distributed.auto_parallel.static.cluster import Cluster
from paddle.distributed.auto_parallel.static.cost import AllgatherOpCost, AllreduceSumOpCost, BroadcastOpCost, CommContext, IdentityOpCost, RecvOpCost, SendOpCost, build_comm_desc

class TestCommOpCost(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.temp_dir.cleanup()

    def test_comm_cost(self):
        if False:
            while True:
                i = 10
        cluster_json_path = os.path.join(self.temp_dir.name, 'auto_parallel_cluster0.json')
        cluster_json_object = json.loads(cluster_json)
        with open(cluster_json_path, 'w') as cluster_json_file:
            json.dump(cluster_json_object, cluster_json_file)
        cluster = Cluster()
        cluster.build_from_file(cluster_json_path)
        CommContext._has_instance = None
        CommContext._instance = None
        comm_context = CommContext(cluster)
        allreduce_sum_op_desc = build_comm_desc('c_allreduce_sum', [0, 1, 2, 3, 4, 5, 6, 7], paddle.float32, [1, 32 * 10 ** 6])
        allreduce_sum_op_cost = AllreduceSumOpCost(op_desc=allreduce_sum_op_desc, comm_context=comm_context)
        allgather_op_desc = build_comm_desc('c_allgather', [0, 1, 2, 3, 4, 5, 6, 7], paddle.float32, [1, 32 * 10 ** 6])
        allgather_op_cost = AllgatherOpCost(op_desc=allgather_op_desc, comm_context=comm_context)
        self.assertTrue(allgather_op_cost.time > 0)
        broadcast_op_desc = build_comm_desc('c_broadcast', [0, 1, 2, 3, 4, 5, 6, 7], paddle.float32, [1, 32 * 10 ** 6])
        broadcast_op_cost = BroadcastOpCost(op_desc=broadcast_op_desc, comm_context=comm_context)
        self.assertTrue(broadcast_op_cost.time > 0)
        send_op_desc = build_comm_desc('send_v2', [0, 1], paddle.float32, [1, 32 * 10 ** 6])
        send_op_cost = SendOpCost(op_desc=send_op_desc, comm_context=comm_context)
        self.assertTrue(send_op_cost.time > 0)
        recv_op_desc = build_comm_desc('recv_v2', [0, 1], paddle.float32, [1, 32 * 10 ** 6])
        recv_op_cost = RecvOpCost(op_desc=recv_op_desc, comm_context=comm_context)
        self.assertTrue(recv_op_cost.time > 0)
        identity_op_desc = build_comm_desc('c_identity', [0, 1], paddle.float32, [1, 32 * 10 ** 6])
        identity_op_cost = IdentityOpCost(op_desc=identity_op_desc, comm_context=comm_context)
        self.assertTrue(identity_op_cost.time >= 0)

    def test_cross_machine_comm_cost(self):
        if False:
            return 10
        cluster_json_path = os.path.join(self.temp_dir.name, 'auto_parallel_cluster1.json')
        cluster_json_object = json.loads(multi_cluster_json)
        with open(cluster_json_path, 'w') as cluster_json_file:
            json.dump(cluster_json_object, cluster_json_file)
        cluster = Cluster()
        cluster.build_from_file(cluster_json_path)
        CommContext._has_instance = None
        CommContext._instance = None
        comm_context = CommContext(cluster)
        allreduce_sum_op_desc = build_comm_desc('c_allreduce_sum', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], paddle.float32, [1, 32 * 10 ** 6])
        allreduce_sum_op_cost = AllreduceSumOpCost(op_desc=allreduce_sum_op_desc, comm_context=comm_context)
        allgather_op_desc = build_comm_desc('c_allgather', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], paddle.float32, [1, 32 * 10 ** 6])
        allgather_op_cost = AllgatherOpCost(op_desc=allgather_op_desc, comm_context=comm_context)
        self.assertTrue(allgather_op_cost.time > 0)
        broadcast_op_desc = build_comm_desc('c_broadcast', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], paddle.float32, [1, 32 * 10 ** 6])
        broadcast_op_cost = BroadcastOpCost(op_desc=broadcast_op_desc, comm_context=comm_context)
        self.assertTrue(broadcast_op_cost.time > 0)
        send_op_desc = build_comm_desc('send_v2', [0, 1], paddle.float32, [1, 32 * 10 ** 6])
        send_op_cost = SendOpCost(op_desc=send_op_desc, comm_context=comm_context)
        self.assertTrue(send_op_cost.time > 0)
        recv_op_desc = build_comm_desc('recv_v2', [0, 1], paddle.float32, [1, 32 * 10 ** 6])
        recv_op_cost = RecvOpCost(op_desc=recv_op_desc, comm_context=comm_context)
        self.assertTrue(recv_op_cost.time > 0)
        if os.path.exists(cluster_json_path):
            os.remove(cluster_json_path)
if __name__ == '__main__':
    unittest.main()