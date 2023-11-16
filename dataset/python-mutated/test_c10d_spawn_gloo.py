import copy
import os
import sys
import tempfile
import test_c10d_spawn
import torch
import torch.distributed as c10d
import torch.nn as nn
from test_c10d_spawn import _torch_dist_nn_available, TestDistributedNNFunctions
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU
from torch.testing._internal.common_distributed import requires_gloo, create_device, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import TestCase, run_tests, skip_but_pass_in_sandcastle_if, TEST_WITH_DEV_DBG_ASAN
if sys.version_info < (3, 9):

    class ProcessGroupShareTensorTest(test_c10d_spawn.AbstractProcessGroupShareTensorTest, TestCase):

        @classmethod
        def opts(cls, threads=2):
            if False:
                return 10
            opts = c10d.ProcessGroupGloo._Options()
            opts._timeout = 5.0
            opts._devices = [create_device(interface='lo')]
            opts._threads = threads
            return opts

        @classmethod
        def _init_pg_gloo(cls, rank, filename, world_size):
            if False:
                while True:
                    i = 10
            store = c10d.FileStore(filename, world_size)
            backend = c10d.ProcessGroupGloo(store, rank, world_size, ProcessGroupShareTensorTest.opts())
            c10d.init_process_group(backend='gloo', store=store, rank=rank, world_size=world_size)
            pg = c10d.distributed_c10d._get_default_group()
            pg._register_backend(torch.device('cpu'), c10d.ProcessGroup.BackendType.GLOO, backend)
            pg._register_backend(torch.device('cuda'), c10d.ProcessGroup.BackendType.GLOO, backend)
            return pg

        @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, 'At least 2 CUDA GPUS needed')
        def test_shared_broadcast_gloo(self):
            if False:
                print('Hello World!')
            self._test_multiprocess(ProcessGroupShareTensorTest._test_broadcast_process, [torch.ones(2, 2).to(i) * i for i in range(self.world_size)], ProcessGroupShareTensorTest._init_pg_gloo, 1)

        @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, 'At least 2 CUDA GPUS needed')
        def test_shared_allreduce_gloo(self):
            if False:
                while True:
                    i = 10
            self._test_multiprocess(ProcessGroupShareTensorTest._test_allreduce_process, [torch.ones(2, 2).to(i) for i in range(self.world_size)], ProcessGroupShareTensorTest._init_pg_gloo, 1)

        @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, 'At least 2 CUDA GPUS needed')
        def test_shared_allgather_gloo(self):
            if False:
                print('Hello World!')
            self._test_multiprocess(ProcessGroupShareTensorTest._test_allgather_process, [torch.ones(2, 2).to(i) * i for i in range(self.world_size)], ProcessGroupShareTensorTest._init_pg_gloo, self.world_size)

        @classmethod
        def _test_allgather_chunk_process(cls, rank, filename, shared_tensor, world_size, init_pg, c2p, p2c):
            if False:
                return 10
            pg = init_pg(rank, filename, world_size)
            chunks = torch.chunk(shared_tensor, world_size, dim=0)
            x = chunks[rank]
            ys = [torch.zeros_like(x) for _ in range(world_size)]
            pg.allgather(ys, x).wait()
            c2p.put((rank, chunks[0].to('cpu'), ys[0].to('cpu')))
            c2p.put((rank, chunks[1].to('cpu'), ys[1].to('cpu')))
            p2c.get()

        @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, 'At least 2 CUDA GPUS needed')
        def test_shared_allgather_chunk_gloo(self):
            if False:
                for i in range(10):
                    print('nop')
            self._test_multiprocess(ProcessGroupShareTensorTest._test_allgather_chunk_process, torch.tensor(range(4)).reshape(2, 2), ProcessGroupShareTensorTest._init_pg_gloo, self.world_size)

class DistributedDataParallelSingleProcessTest(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.rank = 0
        self.world_size = 1
        self.file = tempfile.NamedTemporaryFile(delete=False)

    def tearDown(self):
        if False:
            return 10
        try:
            os.remove(self.file.name)
        except OSError:
            pass

    def _test_base(self, net, inp, check_allclose=True):
        if False:
            i = 10
            return i + 15
        store = c10d.FileStore(self.file.name, self.world_size)
        c10d.init_process_group(backend='gloo', store=store, rank=self.rank, world_size=self.world_size)
        process_group = c10d.distributed_c10d._get_default_group()
        if inp[0].is_cuda:
            device_ids = [torch.cuda.current_device()]
        else:
            device_ids = None
        ddp = nn.parallel.DistributedDataParallel(copy.deepcopy(net), device_ids=device_ids, process_group=process_group)
        net_opt = torch.optim.Adam(net.parameters(), lr=0.001)
        ddp_opt = torch.optim.Adam(ddp.parameters(), lr=0.001)
        for (i, j) in zip(ddp.parameters(), net.parameters()):
            self.assertTrue(i.allclose(j))
        for _ in range(10):
            net_out = net(*inp)
            ddp_out = ddp(*inp)
            net_out.sum().backward()
            ddp_out.sum().backward()
            net_opt.step()
            ddp_opt.step()
        if check_allclose:
            for (i, j) in zip(ddp.parameters(), net.parameters()):
                self.assertTrue(i.allclose(j))

    @requires_gloo()
    def test_cpu(self):
        if False:
            i = 10
            return i + 15
        self._test_base(nn.Linear(2, 2), [torch.randn(30, 2)])

    @requires_gloo()
    @skip_but_pass_in_sandcastle_if(not TEST_CUDA, 'At least 1 CUDA GPUS needed')
    def test_cuda(self):
        if False:
            print('Hello World!')
        self._test_base(nn.Linear(2, 2).to(0), [torch.randn(30, 2).to(0)])

    @requires_gloo()
    @skip_but_pass_in_sandcastle_if(not TEST_CUDA, 'At least 1 CUDA GPUS needed')
    def test_rnn(self):
        if False:
            i = 10
            return i + 15
        BATCH_SIZE = 12
        INPUT_DIM = 256
        OUTPUT_DIM = 256
        HIDDEN_DIM = 256
        N_LAYERS = 3
        SEQ_LEN = 100

        class Net(nn.Module):

            def __init__(self, input_dim, hidden_dim, output_dim, hidden_layers):
                if False:
                    print('Hello World!')
                super().__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.output_dim = output_dim
                self.hidden_layers = hidden_layers
                self.lstm = nn.LSTM(input_dim, hidden_dim, hidden_layers, batch_first=True)
                self.h2o = nn.Linear(hidden_dim, output_dim)

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                self.lstm.flatten_parameters()
                (h_t, _) = self.lstm(x)
                output = self.h2o(h_t)
                loss = nn.functional.mse_loss(output, y)
                return loss
        net = Net(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS).to(0)
        inp = [torch.randn((BATCH_SIZE, SEQ_LEN, INPUT_DIM)).to(0), torch.rand((BATCH_SIZE, SEQ_LEN, OUTPUT_DIM)).to(0)]
        self._test_base(net, inp, check_allclose=False)
if not TEST_WITH_DEV_DBG_ASAN:

    class TestDistributedNNFunctionsGloo(TestDistributedNNFunctions):

        @requires_gloo()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(not _torch_dist_nn_available, 'torch.distributed.nn is not available')
        def test_broadcast(self):
            if False:
                i = 10
                return i + 15
            self._test_broadcast('gloo')

        @requires_gloo()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(not _torch_dist_nn_available, 'torch.distributed.nn is not available')
        def test_reduce(self):
            if False:
                for i in range(10):
                    print('nop')
            self._test_reduce('gloo')

        @requires_gloo()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(not _torch_dist_nn_available, 'torch.distributed.nn is not available')
        def test_allreduce(self):
            if False:
                return 10
            self._test_allreduce('gloo')

        @requires_gloo()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(not _torch_dist_nn_available, 'torch.distributed.nn is not available')
        def test_all_gather(self):
            if False:
                for i in range(10):
                    print('nop')
            self._test_all_gather('gloo')

        @requires_gloo()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(not _torch_dist_nn_available, 'torch.distributed.nn is not available')
        def test_all_to_all(self):
            if False:
                return 10
            self._test_all_to_all('gloo')

        @requires_gloo()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(not _torch_dist_nn_available, 'torch.distributed.nn is not available')
        def test_all_to_all_single(self):
            if False:
                while True:
                    i = 10
            self._test_all_to_all_single('gloo')

        @requires_gloo()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(not _torch_dist_nn_available, 'torch.distributed.nn is not available')
        def test_gather(self):
            if False:
                print('Hello World!')
            store = c10d.FileStore(self.file_name, self.world_size)
            c10d.init_process_group(store=store, rank=self.rank, world_size=self.world_size, backend='gloo')
            device = torch.device(f'cuda:{self.rank}')
            x = torch.ones(5, 5, device=device) + self.rank
            x.requires_grad = True
            tensors = torch.distributed.nn.gather(x, 1)
            if self.rank == 1:
                for (i, t) in enumerate(tensors):
                    self.assertEqual(t, torch.ones(5, 5, device=device) + i)
            elif self.rank == 0:
                for (i, t) in enumerate(tensors):
                    zeros = torch.zeros(5, 5, device=device)
                    self.assertEqual(t, zeros)
            y = torch.sum(torch.stack(tensors), axis=0)
            z = y.sin().sum()
            z.backward()
            x_s = 3 * torch.ones(5, 5, device=device)
            self.assertEqual(x.grad, x_s.cos())

        @requires_gloo()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(not _torch_dist_nn_available, 'torch.distributed.nn is not available')
        def test_scatter(self):
            if False:
                while True:
                    i = 10
            store = c10d.FileStore(self.file_name, self.world_size)
            c10d.init_process_group(store=store, rank=self.rank, world_size=self.world_size, backend='gloo')
            device = torch.device(f'cuda:{self.rank}')
            x0 = torch.ones(5, 5, device=device)
            x1 = torch.ones(5, 5, device=device) + 1
            x0.requires_grad = True
            x1.requires_grad = True
            y = torch.distributed.nn.scatter([x0, x1], 1)
            if self.rank == 1:
                self.assertEqual(y, 1 + torch.ones(5, 5, device=device))
            elif self.rank == 0:
                self.assertEqual(y, torch.ones(5, 5, device=device))
            z = y.sin().sum()
            z.backward()
            if self.rank == 1:
                x0_s = torch.ones(5, 5, device=device).cos()
                x1_s = (2 * torch.ones(5, 5, device=device)).cos()
                self.assertEqual(x0.grad, x0_s)
                self.assertEqual(x1.grad, x1_s)
            if self.rank == 0:
                self.assertEqual(x0.grad, torch.zeros(5, 5, device=device))
if __name__ == '__main__':
    run_tests()