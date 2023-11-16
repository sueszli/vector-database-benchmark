from copy import deepcopy
import torch
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, distribute_module, distribute_tensor, DTensor, Replicate, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, MLPModule, with_comms

def shard_fn(name, module, device_mesh):
    if False:
        print('Hello World!')
    if isinstance(module, nn.Linear):
        for (name, param) in module.named_parameters():
            dist_param = torch.nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            dist_param.register_hook(lambda grad: grad.redistribute(placements=[Shard(0)]))
            module.register_parameter(name, dist_param)

def input_fn(inputs, device_mesh):
    if False:
        while True:
            i = 10
    dist_inp = distribute_tensor(inputs[0], device_mesh, [Shard(0)])
    return dist_inp

def output_fn(outputs, device_mesh):
    if False:
        while True:
            i = 10
    assert isinstance(outputs, DTensor)
    return outputs.redistribute(placements=[Replicate()] * device_mesh.ndim).to_local()

class TestDTensorOptimizer(DTensorTestBase):

    def _assert_optimizer(self, mesh, model, optim, dist_model, dist_optim, inputs):
        if False:
            for i in range(10):
                print('nop')
        optim.zero_grad()
        out = model(inputs)
        loss = out.sum()
        loss.backward()
        optim.step()
        dist_optim.zero_grad()
        dist_out = dist_model(inputs)
        dist_loss = dist_out.sum()
        dist_loss.backward()
        dist_optim.step()
        for (p1, p2) in zip(model.parameters(), dist_model.parameters()):
            p2 = p2.redistribute(placements=[Replicate()] * mesh.ndim)
            p2 = p2.to_local()
            self.assertEqual(p1, p2)

    @with_comms
    def test_adam_1d_sharding(self):
        if False:
            print('Hello World!')
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        adam_configs = [{'lr': 0.1}, {'lr': 0.1, 'weight_decay': 0.05}, {'lr': 0.1, 'foreach': True}, {'lr': 0.1, 'weight_decay': 0.05, 'foreach': True}, {'lr': 0.1, 'weight_decay': 0.05, 'amsgrad': True, 'foreach': True}, {'lr': 0.1, 'weight_decay': 0.05, 'maximize': True, 'amsgrad': True, 'foreach': True}]
        for config in adam_configs:
            mod = MLPModule(self.device_type)
            opt = torch.optim.Adam(mod.parameters(), **config)
            dist_mod = distribute_module(deepcopy(mod), mesh, shard_fn, input_fn, output_fn)
            dist_opt = torch.optim.Adam(dist_mod.parameters(), **config)
            inp = torch.ones(8, 10, device=self.device_type)
            self._assert_optimizer(mesh, mod, opt, dist_mod, dist_opt, inp)