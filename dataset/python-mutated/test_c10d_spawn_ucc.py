import sys
import test_c10d_spawn
import torch
import torch.distributed as c10d
from test_c10d_spawn import _torch_dist_nn_available, TestDistributedNNFunctions
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_distributed import requires_ucc, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import TestCase, run_tests, skip_but_pass_in_sandcastle, skip_but_pass_in_sandcastle_if, TEST_WITH_DEV_DBG_ASAN
NO_UCC = not hasattr(c10d, 'ProcessGroupUCC')
if sys.version_info < (3, 9):

    class ProcessGroupShareTensorTest(test_c10d_spawn.AbstractProcessGroupShareTensorTest, TestCase):

        @classmethod
        def _init_pg_ucc(cls, rank, filename, world_size):
            if False:
                i = 10
                return i + 15
            store = c10d.FileStore(filename, world_size)
            c10d.init_process_group(backend='ucc', store=store, rank=rank, world_size=world_size)
            return c10d.distributed_c10d._get_default_group()

        @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, 'At least 2 CUDA GPUS needed')
        @skip_but_pass_in_sandcastle_if(NO_UCC, 'UCC needed')
        def test_shared_broadcast_ucc(self):
            if False:
                for i in range(10):
                    print('nop')
            self._test_multiprocess(ProcessGroupShareTensorTest._test_broadcast_process, [torch.ones(2, 2).to(i) * i for i in range(self.world_size)], ProcessGroupShareTensorTest._init_pg_ucc, 1)

        @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, 'At least 2 CUDA GPUS needed')
        @skip_but_pass_in_sandcastle_if(NO_UCC, 'UCC needed')
        def test_shared_allreduce_ucc(self):
            if False:
                for i in range(10):
                    print('nop')
            self._test_multiprocess(ProcessGroupShareTensorTest._test_allreduce_process, [torch.ones(2, 2).to(i) for i in range(self.world_size)], ProcessGroupShareTensorTest._init_pg_ucc, 1)

        @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, 'At least 2 CUDA GPUS needed')
        @skip_but_pass_in_sandcastle_if(NO_UCC, 'UCC needed')
        def test_shared_allgather_ucc(self):
            if False:
                while True:
                    i = 10
            self._test_multiprocess(ProcessGroupShareTensorTest._test_allgather_process, [torch.ones(2, 2).to(i) * i for i in range(self.world_size)], ProcessGroupShareTensorTest._init_pg_ucc, self.world_size)
if not TEST_WITH_DEV_DBG_ASAN:

    class TestDistributedNNFunctionsUcc(TestDistributedNNFunctions):

        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(not _torch_dist_nn_available, 'torch.distributed.nn is not available')
        def test_broadcast(self):
            if False:
                for i in range(10):
                    print('nop')
            self._test_broadcast('ucc')

        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(not _torch_dist_nn_available, 'torch.distributed.nn is not available')
        def test_reduce(self):
            if False:
                print('Hello World!')
            self._test_reduce('ucc')

        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(not _torch_dist_nn_available, 'torch.distributed.nn is not available')
        def test_allreduce(self):
            if False:
                while True:
                    i = 10
            self._test_allreduce('ucc')

        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(not _torch_dist_nn_available, 'torch.distributed.nn is not available')
        @skip_but_pass_in_sandcastle('runs into illegal memory access on first assertEqual check when run locally')
        def test_all_gather(self):
            if False:
                while True:
                    i = 10
            self._test_all_gather('ucc')

        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(not _torch_dist_nn_available, 'torch.distributed.nn is not available')
        def test_all_to_all(self):
            if False:
                return 10
            self._test_all_to_all('ucc')

        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(not _torch_dist_nn_available, 'torch.distributed.nn is not available')
        def test_all_to_all_single(self):
            if False:
                print('Hello World!')
            self._test_all_to_all_single('ucc')
if __name__ == '__main__':
    run_tests()