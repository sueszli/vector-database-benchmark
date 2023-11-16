import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor, init_device_mesh, Replicate
from torch.distributed.checkpoint._state_dict_utils import _all_gather_sharded_tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.tensor.parallel import PairwiseParallel, parallelize_module
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, MLPModule, skip_if_lt_x_gpu, with_comms
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir

class TestFsdpTpCheckpointConversion(DTensorTestBase):

    @with_comms
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_fsdp_to_tp(self):
        if False:
            while True:
                i = 10
        CHECKPOINT_DIR = self.temp_dir
        model = MLPModule(self.device_type).cuda(self.rank)
        fsdp_model = FSDP(model, use_orig_params=True)
        FSDP.set_state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT)
        fsdp_state_dict = fsdp_model.state_dict()
        dist_cp.save_state_dict(state_dict=fsdp_state_dict, storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR))
        mesh_shape = (self.world_size,)
        device_mesh = init_device_mesh(self.device_type, mesh_shape)
        model = MLPModule(self.device_type).cuda(self.rank)
        tp_model = parallelize_module(model, device_mesh, PairwiseParallel())
        optimizer = torch.optim.SGD(tp_model.parameters(), lr=0.25)
        torch.manual_seed(0)
        inp = torch.rand(20, 10).cuda(self.rank)
        output = tp_model(inp)
        output.sum().backward()
        optimizer.step()
        tp_state_dict = tp_model.state_dict()
        for (fsdp_item, tp_item) in zip(fsdp_state_dict.items(), tp_state_dict.items()):
            (fsdp_k, fsdp_v) = fsdp_item
            (tp_k, tp_v) = tp_item
            self.assertEqual(fsdp_k, tp_k)
            if isinstance(fsdp_v, ShardedTensor) and isinstance(tp_v, DTensor):
                fsdp_redistributed = _all_gather_sharded_tensor(fsdp_v)
                tp_redistributed = tp_v.redistribute(device_mesh, placements=[Replicate()]).to_local()
                self.assertNotEqual(fsdp_redistributed, tp_redistributed)
        dist_cp.load_state_dict(state_dict=tp_state_dict, storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR))
        tp_model.load_state_dict(tp_state_dict)
        tp_state_dict_after_load = tp_model.state_dict()
        for (fsdp_item, tp_item) in zip(fsdp_state_dict.items(), tp_state_dict.items()):
            (fsdp_k, fsdp_v) = fsdp_item
            (tp_k, tp_v) = tp_item
            self.assertEqual(fsdp_k, tp_k)
            if isinstance(fsdp_v, ShardedTensor) and isinstance(tp_v, DTensor):
                fsdp_redistributed = _all_gather_sharded_tensor(fsdp_v)
                tp_redistributed = tp_v.redistribute(device_mesh, placements=[Replicate()]).to_local()
                self.assertEqual(fsdp_redistributed, tp_redistributed)
if __name__ == '__main__':
    run_tests()