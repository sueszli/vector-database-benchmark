import torch
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.fsdp._shard_utils import _create_chunk_dtensor, _create_chunk_sharded_tensor
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, skip_if_lt_x_gpu, with_comms

class TestShardUtilsDistributed(FSDPTest):

    @property
    def world_size(self):
        if False:
            print('Hello World!')
        return 2

    def _create_tensor(self, *size):
        if False:
            return 10
        torch.manual_seed(0)
        return torch.rand(*size).cuda()

    @skip_if_lt_x_gpu(2)
    def test_create_chunk_sharded_tensor(self):
        if False:
            return 10
        for size in ((1,), (1, 6), (12,), (12, 6), (25,), (25, 6)):
            tensor = self._create_tensor(*size)
            sharded_tensor = _create_chunk_sharded_tensor(tensor, self.rank, self.world_size, torch.cuda.device_count(), _get_default_group())
            output = torch.empty(*size).cuda() if self.rank == 0 else None
            sharded_tensor.gather(0, output)
            if self.rank == 0:
                self.assertEqual(tensor, output)

class TestShardUtilsDistributedDTensor(DTensorTestBase):

    @property
    def world_size(self):
        if False:
            i = 10
            return i + 15
        return 2

    def _create_tensor(self, *size):
        if False:
            print('Hello World!')
        torch.manual_seed(0)
        return torch.rand(*size).cuda()

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_create_chunk_dtensor(self):
        if False:
            for i in range(10):
                print('nop')
        device_mesh = self.build_device_mesh()
        for size in ((1,), (1, 6), (12,), (12, 6), (25,), (25, 6)):
            tensor = self._create_tensor(*size)
            tensor_chunks = torch.chunk(tensor, self.world_size, dim=0)
            dtensor = _create_chunk_dtensor(tensor, self.rank, device_mesh)
            local_tensor = dtensor.to_local()
            if local_tensor.numel() != 0:
                self.assertEqual(local_tensor, tensor_chunks[self.rank])
            else:
                self.assertEqual(self.rank >= len(tensor_chunks), True)
if __name__ == '__main__':
    run_tests()