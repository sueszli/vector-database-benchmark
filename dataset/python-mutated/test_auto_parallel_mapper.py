import json
import os
import tempfile
import unittest
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import base, nn, static, utils
from paddle.base import core
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.static.cluster import Cluster
from paddle.distributed.auto_parallel.static.completion import Completer
from paddle.distributed.auto_parallel.static.dist_context import DistributedContext
from paddle.distributed.auto_parallel.static.mapper import get_comm_volume, get_dtype_bytes, mapping
from paddle.distributed.auto_parallel.static.parallelizer import AutoParallelizer
from paddle.distributed.auto_parallel.static.partitioner import Partitioner
from paddle.distributed.auto_parallel.static.reshard import Resharder
from paddle.distributed.fleet import auto
if os.getenv('CUDA_VISIBLE_DEVICES') is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
paddle.enable_static()
_global_parallel_strategy = None
_global_process_mesh = None
_global_num_stages = None
cluster_json = '\n{\n  "machines": [\n    {\n      "hostname": "machine0",\n      "addr": "0.0.0.1",\n      "port": "768",\n      "devices": [\n        {\n          "global_id": 0,\n          "local_id": 0,\n          "type": "GPU",\n          "model": "A100-SXM4-40GB",\n          "sp_gflops": 19500,\n          "dp_gflops": 9700,\n          "memory": 40\n        },\n        {\n          "global_id": 1,\n          "local_id": 1,\n          "type": "GPU",\n          "model": "A100-SXM4-40GB",\n          "sp_gflops": 19500,\n          "dp_gflops": 9700,\n          "memory": 40\n        },\n        {\n          "global_id": 2,\n          "local_id": 2,\n          "type": "GPU",\n          "model": "A100-SXM4-40GB",\n          "sp_gflops": 19500,\n          "dp_gflops": 9700,\n          "memory": 40\n        },\n        {\n          "global_id": 3,\n          "local_id": 3,\n          "type": "GPU",\n          "model": "A100-SXM4-40GB",\n          "sp_gflops": 19500,\n          "dp_gflops": 9700,\n          "memory": 40\n        },\n        {\n          "global_id": 4,\n          "local_id": 0,\n          "type": "NIC"\n        }\n      ],\n      "links": [\n        {\n          "source_global_id": 0,\n          "target_global_id": 1,\n          "type": "NVL",\n          "bandwidth": 42\n        },\n        {\n          "source_global_id": 0,\n          "target_global_id": 2,\n          "type": "NVL",\n          "bandwidth": 42\n        },\n        {\n          "source_global_id": 0,\n          "target_global_id": 3,\n          "type": "NVL",\n          "bandwidth": 42\n        },\n        {\n          "source_global_id": 0,\n          "target_global_id": 4,\n          "type": "PHB",\n          "bandwidth": 12\n        },\n        {\n          "source_global_id": 1,\n          "target_global_id": 0,\n          "type": "NVL",\n          "bandwidth": 42\n        },\n        {\n          "source_global_id": 1,\n          "target_global_id": 2,\n          "type": "NVL",\n          "bandwidth": 42\n        },\n        {\n          "source_global_id": 1,\n          "target_global_id": 3,\n          "type": "NVL",\n          "bandwidth": 42\n        },\n        {\n          "source_global_id": 1,\n          "target_global_id": 4,\n          "type": "PHB",\n          "bandwidth": 12\n        },\n        {\n          "source_global_id": 2,\n          "target_global_id": 0,\n          "type": "NVL",\n          "bandwidth": 42\n        },\n        {\n          "source_global_id": 2,\n          "target_global_id": 1,\n          "type": "NVL",\n          "bandwidth": 42\n        },\n        {\n          "source_global_id": 2,\n          "target_global_id": 3,\n          "type": "NVL",\n          "bandwidth": 42\n        },\n        {\n          "source_global_id": 2,\n          "target_global_id": 4,\n          "type": "PHB",\n          "bandwidth": 12\n        },\n        {\n          "source_global_id": 3,\n          "target_global_id": 0,\n          "type": "NVL",\n          "bandwidth": 42\n        },\n        {\n          "source_global_id": 3,\n          "target_global_id": 1,\n          "type": "NVL",\n          "bandwidth": 42\n        },\n        {\n          "source_global_id": 3,\n          "target_global_id": 2,\n          "type": "NVL",\n          "bandwidth": 42\n        },\n        {\n          "source_global_id": 3,\n          "target_global_id": 4,\n          "type": "PHB",\n          "bandwidth": 12\n        },\n        {\n          "source_global_id": 4,\n          "target_global_id": 9,\n          "type": "NET",\n          "bandwidth": 1\n        }\n      ]\n    },\n    {\n      "hostname": "machine1",\n      "addr": "0.0.0.2",\n      "port": "768",\n      "devices": [\n        {\n          "global_id": 5,\n          "local_id": 0,\n          "type": "GPU",\n          "model": "Tesla V100-SXM2-32GB",\n          "sp_gflops": 15700,\n          "dp_gflops": 7800,\n          "memory": 32\n        },\n        {\n          "global_id": 6,\n          "local_id": 1,\n          "type": "GPU",\n          "model": "Tesla V100-SXM2-32GB",\n          "sp_gflops": 15700,\n          "dp_gflops": 7800,\n          "memory": 32\n        },\n        {\n          "global_id": 7,\n          "local_id": 2,\n          "type": "GPU",\n          "model": "Tesla V100-SXM2-32GB",\n          "sp_gflops": 15700,\n          "dp_gflops": 7800,\n          "memory": 32\n        },\n        {\n          "global_id": 8,\n          "local_id": 3,\n          "type": "GPU",\n          "model": "Tesla V100-SXM2-32GB",\n          "sp_gflops": 15700,\n          "dp_gflops": 7800,\n          "memory": 32\n        },\n        {\n          "global_id": 9,\n          "local_id": 0,\n          "type": "NIC"\n        }\n      ],\n      "links": [\n        {\n          "source_global_id": 5,\n          "target_global_id": 6,\n          "type": "NVL",\n          "bandwidth": 42\n        },\n        {\n          "source_global_id": 5,\n          "target_global_id": 7,\n          "type": "NVL",\n          "bandwidth": 42\n        },\n        {\n          "source_global_id": 5,\n          "target_global_id": 8,\n          "type": "NVL",\n          "bandwidth": 42\n        },\n        {\n          "source_global_id": 5,\n          "target_global_id": 9,\n          "type": "PHB",\n          "bandwidth": 12\n        },\n        {\n          "source_global_id": 6,\n          "target_global_id": 5,\n          "type": "NVL",\n          "bandwidth": 42\n        },\n        {\n          "source_global_id": 6,\n          "target_global_id": 7,\n          "type": "NVL",\n          "bandwidth": 42\n        },\n        {\n          "source_global_id": 6,\n          "target_global_id": 8,\n          "type": "NVL",\n          "bandwidth": 42\n        },\n        {\n          "source_global_id": 6,\n          "target_global_id": 9,\n          "type": "PHB",\n          "bandwidth": 12\n        },\n        {\n          "source_global_id": 7,\n          "target_global_id": 5,\n          "type": "NVL",\n          "bandwidth": 42\n        },\n        {\n          "source_global_id": 7,\n          "target_global_id": 6,\n          "type": "NVL",\n          "bandwidth": 42\n        },\n        {\n          "source_global_id": 7,\n          "target_global_id": 8,\n          "type": "NVL",\n          "bandwidth": 42\n        },\n        {\n          "source_global_id": 7,\n          "target_global_id": 9,\n          "type": "PHB",\n          "bandwidth": 12\n        },\n        {\n          "source_global_id": 8,\n          "target_global_id": 5,\n          "type": "NVL",\n          "bandwidth": 42\n        },\n        {\n          "source_global_id": 8,\n          "target_global_id": 6,\n          "type": "NVL",\n          "bandwidth": 42\n        },\n        {\n          "source_global_id": 8,\n          "target_global_id": 7,\n          "type": "NVL",\n          "bandwidth": 42\n        },\n        {\n          "source_global_id": 8,\n          "target_global_id": 9,\n          "type": "PHB",\n          "bandwidth": 12\n        },\n        {\n          "source_global_id": 9,\n          "target_global_id": 4,\n          "type": "NET",\n          "bandwidth": 1\n        }\n      ]\n    }\n  ]\n}\n'

class MLPLayer(nn.Layer):

    def __init__(self, hidden_size=64, intermediate_size=4 * 64, initializer_range=0.02):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        np.random.seed(2021)
        arr0 = np.random.normal(0, 0.02, size=(d_model, dim_feedforward))
        arr1 = np.random.normal(0, 0.02, size=(dim_feedforward, d_model))
        arr2 = np.random.normal(0, 0.02, size=(d_model, dim_feedforward))
        arr3 = np.random.normal(0, 0.02, size=(dim_feedforward, d_model))
        weight_attr0 = paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(arr0))
        weight_attr1 = paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(arr1))
        weight_attr2 = paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(arr2))
        weight_attr3 = paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(arr3))
        bias_attr = None
        self.linear0 = nn.Linear(d_model, dim_feedforward, weight_attr0, bias_attr=bias_attr)
        self.linear1 = nn.Linear(dim_feedforward, d_model, weight_attr1, bias_attr=bias_attr)
        self.norm = nn.LayerNorm(d_model, epsilon=1e-05)
        self.linear2 = nn.Linear(d_model, dim_feedforward, weight_attr2, bias_attr=bias_attr)
        self.linear3 = nn.Linear(dim_feedforward, d_model, weight_attr3, bias_attr=bias_attr)

    def forward(self, input):
        if False:
            print('Hello World!')
        if _global_parallel_strategy == 'dp_mp_pp':
            auto.shard_tensor(self.linear0.weight, _global_process_mesh[0], [None, 'y'])
            auto.shard_tensor(self.linear1.weight, _global_process_mesh[0], ['y', None])
            auto.shard_tensor(self.linear2.weight, _global_process_mesh[1], [None, 'y'])
            auto.shard_tensor(self.linear3.weight, _global_process_mesh[1], ['y', None])
        out = self.norm(input)
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)
        auto.shard_tensor(out, _global_process_mesh[1], ['x', None])
        out = self.linear2(out)
        out = F.gelu(out, approximate=True)
        out = self.linear3(out)
        return out

def mlp_forward(train_program, start_program):
    if False:
        i = 10
        return i + 15
    with static.program_guard(train_program, start_program), utils.unique_name.guard():
        batch_size = 4
        hidden_size = 64
        input = static.data(name='input', shape=[batch_size, hidden_size], dtype='float32')
        label = static.data(name='label', shape=[batch_size, 1], dtype='float32')
        if _global_parallel_strategy == 'dp_mp_pp':
            auto.shard_tensor(input, _global_process_mesh[0], ['x', None])
        mlp = MLPLayer(hidden_size=hidden_size, intermediate_size=4 * hidden_size, initializer_range=0.02)
        predict = mlp(input)
        error_cost = paddle.nn.functional.square_error_cost(predict, label)
        loss = paddle.mean(error_cost)
    return (loss, train_program, start_program)

def get_dist_prog(train_program, startup_program, dist_context, rank_id):
    if False:
        return 10
    (loss, train_program, startup_program) = mlp_forward(train_program, startup_program)
    fleet._user_defined_strategy = fleet.DistributedStrategy()
    fleet.user_defined_optimizer = paddle.optimizer.Adam()
    parallelizer = AutoParallelizer(fleet)
    parallelizer._dist_context = dist_context
    completer = Completer(dist_context)
    complete_train_program = completer.complete_forward_annotation(train_program)
    dist_context.block_state.parse_forward_blocks(complete_train_program)
    params_grads = parallelizer._generate_backward(complete_train_program, startup_program, loss, parameter_list=None, no_grad_set=None, callbacks=None)
    partitioner = Partitioner(dist_context, rank_id)
    (dist_train_program, dist_startup_prog, dist_params_grads) = partitioner.partition(complete_train_program, startup_program, params_grads)
    partitioned_optimize_ops = parallelizer._apply_optimize(dist_train_program, dist_startup_prog, dist_params_grads)
    resharder = Resharder(dist_train_program, dist_startup_prog, rank_id, dist_context, dist_params_grads)
    resharder.reshard()
    return (dist_train_program, dist_startup_prog)

def is_in_machine(device_local_id, machine):
    if False:
        i = 10
        return i + 15
    for device in machine.devices.values():
        if device_local_id == device.local_id:
            return True
    return False

def get_device_local_ids(machine):
    if False:
        return 10
    local_ids = []
    for device in machine.devices.values():
        local_ids.append[device.local_id]
    return local_ids

class TestAutoParallelMapper(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.temp_dir.cleanup()

    def test_mapper_dp_mp_pp(self):
        if False:
            i = 10
            return i + 15
        cluster_json_path = os.path.join(self.temp_dir.name, 'auto_parallel_cluster.json')
        cluster_json_object = json.loads(cluster_json)
        with open(cluster_json_path, 'w') as cluster_json_file:
            json.dump(cluster_json_object, cluster_json_file)
        cluster = Cluster()
        cluster.build_from_file(cluster_json_path)
        global _global_parallel_strategy
        _global_parallel_strategy = 'dp_mp_pp'
        global _global_num_stages
        _global_num_stages = 2
        global _global_process_mesh
        _global_process_mesh = [auto.ProcessMesh([[0, 1], [2, 3]], dim_names=['x', 'y']), auto.ProcessMesh([[4, 5], [6, 7]], dim_names=['x', 'y'])]
        processes = [0, 1, 2, 3, 4, 5, 6, 7]
        dist_programs = {}
        for rank_id in processes:
            train_program = static.Program()
            startup_program = static.Program()
            dist_context = DistributedContext()
            (dist_train_program, dist_startup_prog) = get_dist_prog(train_program, startup_program, dist_context, rank_id)
            dist_programs[rank_id] = [dist_train_program, None]
        rank_mapping = mapping(dist_programs, cluster)
        all_mapped_ranks = set()
        for (machine_id, machine_mapping) in rank_mapping.items():
            machine = cluster.machines[machine_id]
            machine_mapped_ranks = set()
            machine_mapped_device_local_ids = set()
            for (rank, device_ids) in machine_mapping['ranks'].items():
                self.assertEqual(len(device_ids), 1)
                self.assertTrue(is_in_machine(device_ids[0], machine))
                machine_mapped_ranks.add(rank)
                machine_mapped_device_local_ids.add(device_ids[0])
            self.assertEqual(len(machine_mapped_ranks), len(machine_mapped_device_local_ids))
            all_mapped_ranks.update(machine_mapped_ranks)
        self.assertEqual(set(processes), all_mapped_ranks)

    def test_mapper_misc(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(get_dtype_bytes(paddle.float64), 8)
        self.assertEqual(get_dtype_bytes(paddle.float32), 4)
        self.assertEqual(get_dtype_bytes(paddle.float16), 2)
        self.assertEqual(get_dtype_bytes(paddle.bfloat16), 2)
        self.assertEqual(get_dtype_bytes(paddle.int64), 8)
        self.assertEqual(get_dtype_bytes(paddle.int32), 4)
        self.assertEqual(get_dtype_bytes(paddle.int16), 2)
        self.assertEqual(get_dtype_bytes(paddle.int8), 1)
        self.assertEqual(get_dtype_bytes(paddle.uint8), 1)
        self.assertRaises(ValueError, get_dtype_bytes, 'unknown type')
        train_program = static.Program()
        startup_program = static.Program()
        ring_id = 0
        root_id = 0
        nranks = 2
        with base.program_guard(train_program, startup_program):
            input = paddle.static.data(name='input', shape=[-1, 10, 10], dtype='float32')
            output = train_program.current_block().create_var(name='outofbroadcast', dtype='float32', type=core.VarDesc.VarType.LOD_TENSOR, persistable=False, stop_gradient=False)
            broadcast_op = train_program.global_block().append_op(type='c_broadcast', inputs={'X': input}, attrs={'ring_id': ring_id, 'root': root_id}, outputs={'Out': output})
            self.assertEqual(get_comm_volume(broadcast_op, 0, 1), 400)
            self.assertIsNone(get_comm_volume(broadcast_op, 1, 0))
            allgather_op = train_program.global_block().append_op(type='c_allgather', inputs={'X': input}, attrs={'ring_id': ring_id, 'nranks': nranks}, outputs={'Out': output})
            self.assertEqual(get_comm_volume(allgather_op, 0, 1), 400)
            self.assertIsNone(get_comm_volume(allgather_op, 0, 0))
            reduce_op = train_program.global_block().append_op(type='c_reduce_sum', inputs={'X': input}, attrs={'ring_id': ring_id, 'root_id': root_id}, outputs={'Out': output})
            self.assertIsNone(get_comm_volume(reduce_op, 0, 1))
            self.assertEqual(get_comm_volume(reduce_op, 1, 0), 400)
            cast_op = train_program.global_block().append_op(type='cast', inputs={'X': input}, outputs={'Out': output}, attrs={'in_dtype': base.core.VarDesc.VarType.FP32, 'out_dtype': base.core.VarDesc.VarType.FP32})
            self.assertRaises(ValueError, get_comm_volume, cast_op, 0, 1)
if __name__ == '__main__':
    unittest.main()