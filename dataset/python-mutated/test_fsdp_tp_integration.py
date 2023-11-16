import copy
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import distributed as dist
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._tensor import DeviceMesh, DTensor as DT, init_device_mesh
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, FullyShardedDataParallel as FSDP
from torch.distributed.tensor.parallel import parallelize_module, SequenceParallel
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TEST_WITH_DEV_DBG_ASAN
if not dist.is_available():
    print('Distributed not available, skipping tests', file=sys.stderr)
    sys.exit(0)
if TEST_WITH_DEV_DBG_ASAN:
    print('Skip dev-asan as torch + multiprocessing spawn have known issues', file=sys.stderr)
    sys.exit(0)

def _is_nested_tensor(val: Any) -> bool:
    if False:
        for i in range(10):
            print('nop')
    if type(val) is ShardedTensor:
        if len(val.local_shards()) == 0:
            return False
        if type(val.local_shards()[0].tensor) is ShardedTensor:
            return True
    return False

class SimpleModel(torch.nn.Module):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.net1 = torch.nn.Linear(5, 8)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(8, 4)
        self.net3 = torch.nn.Linear(4, 12)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        return self.net3(self.net2(self.relu(self.net1(x))))

    @staticmethod
    def get_sharded_param_names() -> List[str]:
        if False:
            return 10
        return ['net1.weight', 'net1.bias', 'net2.weight']

    @staticmethod
    def get_non_sharded_param_names() -> List[str]:
        if False:
            print('Hello World!')
        return ['net3.weight', 'net3.bias']

class TestTPFSDPIntegration(FSDPTest):

    def _get_params_and_sharding_info(self, model: SimpleModel, sharded_param_names: List[str], tensor_parallel_size: int) -> Tuple[Dict[str, int], Dict[str, Tuple[torch.Size, int]]]:
        if False:
            print('Hello World!')
        ' '
        assert type(model) is SimpleModel, 'Expects a `SimpleModel` since the sharding cases on the model definition'
        param_name_to_numel = OrderedDict()
        param_name_to_sharding_info = OrderedDict()
        for (param_name, param) in model.named_parameters():
            if param_name not in sharded_param_names:
                param_name_to_numel[param_name] = param.numel()
            else:
                param_name_to_numel[param_name] = param.numel() // tensor_parallel_size
                param_name_to_sharding_info[param_name] = (param.size(), 0 if 'net1' in param_name else 1)
        return (param_name_to_numel, param_name_to_sharding_info)

    def _get_sub_pgs(self, tensor_parallel_size: int):
        if False:
            print('Hello World!')
        '\n        Generates TP and FSDP subprocess groups. ``tensor_parallel_size`` gives\n        the TP process group size.\n\n        For example, if the global world size is 8 and the tensor parallel size\n        is 2, then this creates:\n        - 4 TP subprocess groups: [0, 1], [2, 3], [4, 5], [6, 7]\n        - 2 FSDP subprocess groups: [0, 2, 4, 6], [1, 3, 5, 7]\n        '
        twod_mesh = DeviceMesh(device_type='cuda', mesh=torch.arange(0, self.world_size).view(-1, tensor_parallel_size))
        fsdp_pg = twod_mesh.get_dim_groups()[0]
        tp_pg = twod_mesh.get_dim_groups()[1]
        return (twod_mesh, fsdp_pg, tp_pg)

    def _get_chunk_sharding_spec(self, tp_world_size: int, tp_pg: dist.ProcessGroup):
        if False:
            for i in range(10):
                print('nop')
        placements = [f'rank:{idx}/cuda:{dist.distributed_c10d.get_global_rank(tp_pg, idx) % torch.cuda.device_count()}' for idx in range(tp_world_size)]
        colwise_spec = ChunkShardingSpec(dim=0, placements=placements)
        rowwise_spec = ChunkShardingSpec(dim=1, placements=placements)
        return (colwise_spec, rowwise_spec)

    def _sync_tp_grads(self, tp_fsdp_model: FSDP, tp_pg: dist.ProcessGroup, param_name_to_numel: Dict[str, int], non_sharded_param_names: List[str]) -> None:
        if False:
            while True:
                i = 10
        "\n        Syncs the tensor parallel parameters' gradients following the data\n        parallel paradigm where gradients are averaged over ranks (in this\n        case, the ones in the tensor parallel process group).\n        "
        tp_world_size = tp_pg.size()
        fsdp_world_size = self.world_size // tp_world_size
        assert type(tp_fsdp_model) is FSDP and len(list(tp_fsdp_model.parameters())) == 1, 'The following logic assumes a single top-level-only FSDP wrapping the model with TP already applied'
        flat_param = tp_fsdp_model.params[0]
        splits = tuple(param_name_to_numel.values())
        unsharded_size = torch.Size([flat_param.numel() * fsdp_world_size])
        unsharded_zeros = torch.zeros(unsharded_size, device=flat_param.device)
        per_param_masks = unsharded_zeros.split(splits)
        for (param_idx, param_name) in enumerate(param_name_to_numel.keys()):
            if param_name not in non_sharded_param_names:
                per_param_masks[param_idx][:] = 1
        unsharded_mask = torch.cat(per_param_masks).contiguous().type(torch.BoolTensor)
        sharded_mask = unsharded_mask.chunk(fsdp_world_size)[self.rank // tp_world_size]
        grad_device = flat_param.grad.device
        grad = flat_param.grad.detach().clone().cuda(self.rank)
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=tp_pg)
        grad = grad.to(grad_device)
        flat_param.grad[~sharded_mask] = grad[~sharded_mask]
        flat_param.grad /= tp_world_size

    def _get_grads_as_flattened(self, model: FSDP, uses_tp: bool, param_name_to_numel: Dict[str, int], param_name_to_sharding_info: Dict[str, Tuple[torch.Size, int]], tp_pg: Optional[dist.ProcessGroup], fsdp_pg: Optional[dist.ProcessGroup], sharded_param_names: Optional[List[str]]) -> torch.Tensor:
        if False:
            return 10
        '\n        Returns all unsharded gradients as a single flattened tensor. This\n        returns the same value on all ranks.\n        '
        local_grads_as_flattened = torch.cat([torch.flatten(param.grad) for param in model.parameters()]).contiguous().cuda(self.rank)
        all_grads_as_flattened = torch.cat([torch.empty_like(local_grads_as_flattened) for _ in range(fsdp_pg.size())]).contiguous()
        dist._all_gather_base(all_grads_as_flattened, local_grads_as_flattened, group=fsdp_pg)
        if not uses_tp:
            return all_grads_as_flattened
        splits = tuple(param_name_to_numel.values())
        all_grads_per_param = list(all_grads_as_flattened.split(splits))
        for (param_idx, param_name) in enumerate(param_name_to_numel.keys()):
            if param_name in sharded_param_names:
                local_tensor_size = list(param_name_to_sharding_info[param_name][0])
                sharding_dim = param_name_to_sharding_info[param_name][1]
                local_tensor_size[sharding_dim] //= tp_pg.size()
                local_tensor = all_grads_per_param[param_idx].view(*local_tensor_size)
                local_tensors = [torch.empty_like(local_tensor) for _ in range(tp_pg.size())]
                dist.all_gather(local_tensors, local_tensor, group=tp_pg)
                all_grads_per_param[param_idx] = torch.cat(local_tensors, dim=sharding_dim).reshape(-1)
        return torch.cat(all_grads_per_param).contiguous()

    @skip_if_lt_x_gpu(4)
    @parametrize('tensor_parallel_size', [2, 4])
    @parametrize('cpu_offload', [CPUOffload(offload_params=False), CPUOffload(offload_params=True)])
    def test_fsdp_tp_integration(self, tensor_parallel_size, cpu_offload):
        if False:
            print('Hello World!')
        '\n        Tests training for TP + FSDP integration by comparing an FSDP-only\n        model with a TP + FSDP model.\n        '
        LR = 3e-05
        torch.manual_seed(0)
        model = SimpleModel().cuda(self.rank)
        tp_fsdp_model = copy.deepcopy(model)
        sharded_param_names = SimpleModel.get_sharded_param_names()
        non_sharded_param_names = SimpleModel.get_non_sharded_param_names()
        (param_name_to_numel, param_name_to_sharding_info) = self._get_params_and_sharding_info(model, sharded_param_names, tensor_parallel_size)
        input_seed = self.rank
        torch.manual_seed(input_seed + 1)
        inp_size = [2, 3, 5]
        inp = torch.rand(*inp_size).cuda(self.rank)
        self.assertEqual(model(inp), tp_fsdp_model(inp))
        mesh_1d = init_device_mesh('cuda', (self.world_size,))
        fsdp_model = FSDP(model, cpu_offload=cpu_offload, device_mesh=mesh_1d)
        mesh_2d = init_device_mesh('cuda', (self.world_size // tensor_parallel_size, tensor_parallel_size), mesh_dim_names=['dp', 'tp'])
        tp_fsdp_model = parallelize_module(tp_fsdp_model, mesh_2d['tp'], SequenceParallel())
        tp_pg = mesh_2d['tp'].get_dim_groups(mesh_dim=0)
        assert isinstance(tp_fsdp_model.net1.weight, DT)
        assert isinstance(tp_fsdp_model.net2.weight, DT)
        tp_fsdp_model = FSDP(tp_fsdp_model, cpu_offload=cpu_offload, device_mesh=mesh_2d['dp'])
        fsdp_pg = mesh_2d['dp'].get_dim_groups(mesh_dim=0)
        fsdp_out = fsdp_model(inp)
        tp_fsdp_out = tp_fsdp_model(inp)
        self.assertEqual(fsdp_out, tp_fsdp_out)
        fsdp_out.sum().backward()
        tp_fsdp_out.sum().backward()
        self._sync_tp_grads(tp_fsdp_model, tp_pg, param_name_to_numel, non_sharded_param_names)
        model_grads = self._get_grads_as_flattened(fsdp_model, False, param_name_to_numel, param_name_to_sharding_info, None, self.process_group, None)
        model_tp_grads = self._get_grads_as_flattened(tp_fsdp_model, True, param_name_to_numel, param_name_to_sharding_info, tp_pg, fsdp_pg, sharded_param_names)
        self.assertEqual(model_grads, model_tp_grads)
        fsdp_optim = torch.optim.SGD(fsdp_model.parameters(), lr=LR)
        tp_fsdp_optim = torch.optim.SGD(tp_fsdp_model.parameters(), lr=LR)
        fsdp_optim.step()
        tp_fsdp_optim.step()
        torch.manual_seed(input_seed + 16)
        inp = torch.rand(*inp_size).cuda(self.rank)
        fsdp_out = fsdp_model(inp)
        tp_fsdp_out = tp_fsdp_model(inp)
        self.assertEqual(fsdp_out, tp_fsdp_out)
instantiate_parametrized_tests(TestTPFSDPIntegration)
if __name__ == '__main__':
    run_tests()