import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
import torch.distributed.checkpoint as dist_cp
import torch.distributed as dist
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner, DefaultLoadPlanner
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, with_comms
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir

class FsdpModelStateCheckpoint(DTensorTestBase):

    @property
    def backend(self):
        if False:
            for i in range(10):
                print('nop')
        return 'cpu:gloo,cuda:nccl'

    def _test_fsdp_model_state(self, process_group) -> None:
        if False:
            while True:
                i = 10
        CHECKPOINT_DIR = self.temp_dir
        model = FSDP(torch.nn.Linear(8, 8, device='meta'))
        model(torch.rand(8, 8, device=dist.get_rank())).sum().backward()
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            state_dict = {'model': model.state_dict()}
            dist_cp.save_state_dict(state_dict=state_dict, storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR), planner=DefaultSavePlanner())
        model_2 = FSDP(torch.nn.Linear(8, 8, device='meta'), process_group=process_group)
        with FSDP.summon_full_params(model):
            with FSDP.summon_full_params(model_2):
                self.assertNotEqual(model.weight, model_2.weight)
                self.assertNotEqual(model.bias, model_2.bias)
        with FSDP.state_dict_type(model_2, StateDictType.SHARDED_STATE_DICT):
            state_dict = {'model': model_2.state_dict()}
            dist_cp.load_state_dict(state_dict=state_dict, storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR), planner=DefaultLoadPlanner())
            model_2.load_state_dict(state_dict['model'])
        with FSDP.summon_full_params(model):
            with FSDP.summon_full_params(model_2):
                self.assertEqual(model.weight, model_2.weight)
                self.assertEqual(model.bias, model_2.bias)

    @with_comms
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_fsdp_model_state_no_resharding(self):
        if False:
            i = 10
            return i + 15
        self._test_fsdp_model_state(process_group=None)

    def _create_new_dist_group(self):
        if False:
            i = 10
            return i + 15
        world_size = dist.get_world_size()
        group1 = [i for i in range(world_size) if i % 2 == 0]
        group2 = [i for i in range(world_size) if i % 2 != 0]
        fsdp_0 = dist.new_group(ranks=group1)
        fsdp_1 = dist.new_group(ranks=group2)
        if dist.get_rank() % 2 == 0:
            my_fsdp = fsdp_0
        else:
            my_fsdp = fsdp_1
        return my_fsdp

    @with_comms
    @skip_if_lt_x_gpu(4)
    @with_temp_dir
    def test_fsdp_model_state_with_resharding(self):
        if False:
            i = 10
            return i + 15
        self._test_fsdp_model_state(process_group=self._create_new_dist_group())
if __name__ == '__main__':
    run_tests()