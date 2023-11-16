import json
import os
import tempfile
import unittest
from test_cluster import cluster_json
import paddle
import paddle.distributed.auto_parallel.static.cost as cost_model
from paddle.distributed.auto_parallel.static.cluster import Cluster
from paddle.distributed.auto_parallel.static.cost import CommContext
from paddle.distributed.auto_parallel.static.cost.base_cost import build_comp_desc_from_op, build_comp_desc_str_for_predict, calc_time_by_modeling
paddle.enable_static()

def check_cost(cost):
    if False:
        i = 10
        return i + 15
    if cost.memory >= 0 and cost.flops >= 0 and (cost.time >= 0):
        return True
    return False

class TestCost(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.temp_dir.cleanup()

    def test_base_cost(self):
        if False:
            i = 10
            return i + 15
        cost = cost_model.Cost(memory=100, flops=200, time=0.5)
        self.assertTrue(check_cost(cost))

    def test_comp_cost(self):
        if False:
            while True:
                i = 10
        x = paddle.static.data(name='x', shape=[20, 20], dtype='float32')
        y = paddle.static.data(name='y', shape=[20, 20], dtype='float32')
        z = paddle.matmul(x, y)
        matmul_v2_op = None
        ops = paddle.static.default_main_program().global_block().ops
        for op in ops:
            if op.type == 'matmul_v2':
                matmul_v2_op = op
                break
        matmul_v2_cost = cost_model._g_op_cost_factory['matmul_v2'](op=matmul_v2_op)
        desc = build_comp_desc_from_op(op=matmul_v2_op)
        desc_str = build_comp_desc_str_for_predict(desc)
        self.assertIsNotNone(desc_str)
        self.assertTrue(check_cost(matmul_v2_cost.cost))
        time = calc_time_by_modeling(op=matmul_v2_op)
        self.assertEqual(time, matmul_v2_cost.cost.time)
        tensor_cost = cost_model.TensorCost(tensor=x)
        self.assertEqual(tensor_cost.cost.memory, 1600)

    def test_comm_cost(self):
        if False:
            print('Hello World!')
        cluster_json_path = os.path.join(self.temp_dir.name, 'auto_parallel_cluster.json')
        cluster_json_object = json.loads(cluster_json)
        with open(cluster_json_path, 'w') as cluster_json_file:
            json.dump(cluster_json_object, cluster_json_file)
        cluster = Cluster()
        cluster.build_from_file(cluster_json_path)
        CommContext._has_instance = None
        CommContext._instance = None
        comm_context = CommContext(cluster)
        desc = {}
        desc['op'] = 'c_allreduce_sum'
        desc['inputs'] = {'X': [(paddle.float32, [100, 200])]}
        desc['group_ranks'] = [0, 1]
        allreduce_cost = cost_model._g_op_cost_factory['c_allreduce_sum'](op_desc=desc, comm_context=CommContext(cluster))
        self.assertTrue(check_cost(allreduce_cost.cost))
        if os.path.exists(cluster_json_path):
            os.remove(cluster_json_path)

    def test_cost_estimator(self):
        if False:
            i = 10
            return i + 15
        cluster_json_path = os.path.join(self.temp_dir.name, 'auto_parallel_cluster.json')
        cluster_json_object = json.loads(cluster_json)
        with open(cluster_json_path, 'w') as cluster_json_file:
            json.dump(cluster_json_object, cluster_json_file)
        cluster = Cluster()
        cluster.build_from_file(cluster_json_path)
        train_program = paddle.static.Program()
        cost_estimator = cost_model.CostEstimator(train_program, cluster=cluster)
        self.assertIsNotNone(cost_estimator)
        if os.path.exists(cluster_json_path):
            os.remove(cluster_json_path)
if __name__ == '__main__':
    unittest.main()