from copy import deepcopy
import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed._tensor import init_device_mesh
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner, DefaultSavePlanner
from torch.distributed.tensor.parallel import PairwiseParallel, parallelize_module
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, MLPModule, skip_if_lt_x_gpu, with_comms
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir

class TestTpCheckpoint(DTensorTestBase):

    @with_comms
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_tp_checkpoint(self):
        if False:
            while True:
                i = 10
        CHECKPOINT_DIR = self.temp_dir
        mesh_shpe = (self.world_size,)
        tp_mesh = init_device_mesh(self.device_type, mesh_shpe)
        model = MLPModule(self.device_type).cuda(self.rank)
        model = parallelize_module(model, tp_mesh, PairwiseParallel())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
        original_state_dict = deepcopy(model.state_dict())
        dist_cp.save_state_dict(state_dict=original_state_dict, storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR), planner=DefaultSavePlanner())
        torch.manual_seed(0)
        inp = torch.rand(20, 10).cuda(self.rank)
        output = model(inp)
        output.sum().backward()
        optimizer.step()
        state_dict = model.state_dict()
        for (param1, param2) in zip(original_state_dict.values(), state_dict.values()):
            self.assertNotEqual(param1.to_local(), param2.to_local())
        dist_cp.load_state_dict(state_dict=state_dict, storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR), planner=DefaultLoadPlanner())
        for (param1, param2) in zip(original_state_dict.values(), state_dict.values()):
            self.assertEqual(param1.to_local(), param2.to_local())
if __name__ == '__main__':
    run_tests()