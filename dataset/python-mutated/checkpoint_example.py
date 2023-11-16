"""
The following example contains a simple MLP model that uses
different DTensor layouts, and use the checkpointing API to
checkpoint save/load the model.
"""
import os
from typing import cast, List
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._tensor import DeviceMesh, distribute_module, distribute_tensor, DTensor, Replicate, Shard
from torch.distributed._tensor.placement_types import Placement
from torch.distributed.tensor.parallel import PairwiseParallel, parallelize_module

class SimpleMLP(torch.nn.Module):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.net1 = torch.nn.Linear(5, 128)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(128, 12)

    def forward(self, x):
        if False:
            print('Hello World!')
        return self.net2(F.relu(self.net1(x)))

def gen_tensor_parallel_model(model: nn.Module, mesh: DeviceMesh) -> nn.Module:
    if False:
        return 10
    '\n    generates a nn.Module where parameters are sharded in the tensor-parallel\n    fashion.\n    '
    return parallelize_module(model, mesh, PairwiseParallel())

def gen_partial_replicate_2d(model: nn.Module, mesh: DeviceMesh) -> nn.Module:
    if False:
        for i in range(10):
            print('nop')
    '\n    generates a nn.Module where parameters are replicated in the first mesh\n    dimension, and sharded in the second mesh dimension.\n    '

    def parallel_fn(name, module, device_mesh):
        if False:
            print('Hello World!')
        assert device_mesh.ndim == 2
        if isinstance(module, torch.nn.Linear) and name == 'net1':
            for (name, param) in module.named_parameters():
                dist_param = torch.nn.Parameter(distribute_tensor(param, device_mesh, [Replicate(), Shard(0)]))
                module.register_parameter(name, dist_param)
        elif isinstance(module, torch.nn.Linear) and name == 'net2':
            for (name, param) in module.named_parameters():
                dist_spec = [Replicate(), Shard(1)] if name == 'weight' else [Replicate(), Replicate()]
                dist_param = torch.nn.Parameter(distribute_tensor(param, device_mesh, dist_spec))
                module.register_parameter(name, dist_param)

    def input_fn(inputs, device_mesh):
        if False:
            for i in range(10):
                print('nop')
        return DTensor.from_local(inputs[0], device_mesh, [Replicate(), Replicate()])

    def output_fn(outputs, device_mesh):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(outputs, DTensor)
        return outputs.to_local()
    return distribute_module(model, mesh, partition_fn=parallel_fn, input_fn=input_fn, output_fn=output_fn)

def gen_model_param_in_submesh(model: nn.Module, sub_mesh: DeviceMesh) -> nn.Module:
    if False:
        while True:
            i = 10
    '\n    generates a nn.Module where parameters are sharded/replicated only on a\n    sub-mesh (i.e. mesh(0, 2) in a world size of 4)\n    '

    def parallel_fn(name, module, device_mesh):
        if False:
            while True:
                i = 10
        assert device_mesh.ndim == 1
        if isinstance(module, torch.nn.Linear) and name == 'net1':
            for (name, param) in module.named_parameters():
                dist_param = torch.nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
                module.register_parameter(name, dist_param)
        elif isinstance(module, torch.nn.Linear) and name == 'net2':
            for (name, param) in module.named_parameters():
                dist_spec = cast(List[Placement], [Shard(1)] if name == 'weight' else [Replicate()])
                dist_param = torch.nn.Parameter(distribute_tensor(param, device_mesh, dist_spec))
                module.register_parameter(name, dist_param)

    def input_fn(inputs, device_mesh):
        if False:
            while True:
                i = 10
        return DTensor.from_local(inputs[0], device_mesh, [Replicate()])

    def output_fn(outputs, device_mesh):
        if False:
            i = 10
            return i + 15
        assert isinstance(outputs, DTensor)
        return outputs.to_local()
    return distribute_module(model, sub_mesh, partition_fn=parallel_fn, input_fn=input_fn, output_fn=output_fn)

def checkpoint(model: nn.Module, mesh: DeviceMesh) -> nn.Module:
    if False:
        while True:
            i = 10
    '\n    checkpoint save/load models with DTensor parameters\n    '
    pass

def run_checkpoint_example(rank, world_size):
    if False:
        for i in range(10):
            print('nop')
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    mesh = DeviceMesh('cpu', torch.arange(world_size))
    model_tp = gen_tensor_parallel_model(SimpleMLP(), mesh)
    model_tp(torch.rand(5, 5))
    mesh_2d = DeviceMesh('cpu', torch.arange(world_size).reshape(2, 2))
    model_2d = gen_partial_replicate_2d(SimpleMLP(), mesh_2d)
    model_2d(torch.rand(5, 5))
    submesh = DeviceMesh('cpu', [0, 2])
    model_submesh = gen_model_param_in_submesh(SimpleMLP(), submesh)
    model_submesh(torch.rand(5, 5))
    print(f'partial replicate model state_dict: {model_submesh.state_dict()}')
    model = checkpoint(model_2d, mesh)
    dist.destroy_process_group()
if __name__ == '__main__':
    world_size = 4
    mp.spawn(run_checkpoint_example, args=(world_size,), nprocs=world_size, join=True)