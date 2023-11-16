import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate
from torch.distributed.tensor.parallel import PairwiseParallel, parallelize_module
from torch.distributed.tensor.parallel.ddp import _pre_dp_module_transform
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, MLPModule, with_comms
TP_DEGREE = 2
LR = 3e-05

def init_model(device_type, model_parallel_size=TP_DEGREE):
    if False:
        for i in range(10):
            print('nop')
    torch.manual_seed(0)
    model = MLPModule(device_type)
    torch.manual_seed(0)
    twod_model = MLPModule(device_type)
    model = DDP(model)
    world_size = dist.get_world_size()
    twod_mesh = DeviceMesh(device_type=device_type, mesh=torch.arange(0, world_size).view(-1, model_parallel_size))
    dp_pg = twod_mesh.get_dim_groups()[0]
    twod_model = parallelize_module(twod_model, twod_mesh, PairwiseParallel(), tp_mesh_dim=1)
    _pre_dp_module_transform(twod_model)
    twod_model = DDP(twod_model, process_group=dp_pg)
    return (model, twod_model, dp_pg)

class Test2dParallelIntegration(DTensorTestBase):

    def _check_module(self, m1, m2, check_grad=False):
        if False:
            while True:
                i = 10
        named_parameters = dict(m1.named_parameters())
        for (name, param_m2) in m2.named_parameters():
            if name not in named_parameters:
                print(name, named_parameters.keys())
            self.assertTrue(name in named_parameters)
            param_m1 = named_parameters[name]
            if check_grad:
                param_m2 = param_m2.grad
                param_m1 = param_m1.grad
            if isinstance(param_m2, DTensor):
                replicate = [Replicate()]
                param_m2 = param_m2.redistribute(device_mesh=param_m2.device_mesh, placements=replicate).to_local()
            self.assertEqual(param_m2, param_m1)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_ddp_integration_functionality(self) -> None:
        if False:
            return 10
        (model, twod_model, dp_pg) = init_model(self.device_type)
        optim = torch.optim.Adam(model.parameters(), lr=LR)
        twod_optim = torch.optim.Adam(twod_model.parameters(), lr=LR)
        input_seed = dist.get_rank(dp_pg)
        torch.manual_seed(input_seed + 1)
        input = torch.rand(4, 10, device=self.device_type)
        output = model(input)
        twod_output = twod_model(input)
        self.assertEqual(output, twod_output)
        output.sum().backward()
        twod_output.sum().backward()
        self._check_module(model, twod_model, check_grad=True)
        optim.step()
        twod_optim.step()
        self._check_module(model, twod_model)
        torch.manual_seed(input_seed + 1004)
        input = torch.rand(16, 10, device=self.device_type)
        output = model(input)
        twod_output = twod_model(input)
        self.assertEqual(output, twod_output)
if __name__ == '__main__':
    run_tests()