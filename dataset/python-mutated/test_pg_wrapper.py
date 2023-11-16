import os
import sys
from datetime import timedelta
import torch
import torch.distributed as c10d
if not c10d.is_available():
    print('c10d not available, skipping tests', file=sys.stderr)
    sys.exit(0)
from test_c10d_common import LOOPBACK
from torch.testing._internal.common_distributed import create_device, MultiProcessTestCase, requires_gloo, requires_nccl, skip_if_lt_x_gpu, with_dist_debug_levels
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN

class AbstractProcessGroupWrapperTest(MultiProcessTestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self._spawn_processes()

    def _validate_error(self, exception, op_type, rank, tensor, verify_diff=True):
        if False:
            while True:
                i = 10
        err = str(exception)
        self.assertTrue(op_type in err, f'Got {err} but expected {op_type} to be in error.')
        if op_type != 'BARRIER':
            self.assertTrue(f'{list(tensor.shape)}' in err, f'Did not find shapes {list(tensor.shape)} in error {err}')
            if 'cuda' in str(tensor.device):
                self.assertTrue('cuda' in err, f'Did not find cuda device in error {err}')
            else:
                self.assertTrue(str(tensor.device) in err, f'Did not find tensor device {str(tensor.device)} in error {err}')
            if 'float' in str(tensor.dtype):
                self.assertTrue('Float' in err, 'Expected Float type')
            elif 'int' in str(tensor.dtype):
                self.assertTrue('Long' in err, 'Expected Long type')
            else:
                self.fail(f'Unexpected dtype {str(tensor.dtype)} for error {err}')
            self.assertTrue('SequenceNumber' in err)
            if verify_diff:
                self.assertTrue('Collectives differ in the following' in err, f'Got error {err}')

    def _test_collective_hang(self, wrapper_pg, use_cuda=False):
        if False:
            print('Hello World!')
        faulty_rank = 1
        if self.rank != faulty_rank:
            tensor = torch.randn(20, 10)
            if use_cuda:
                tensor = tensor.to(self.rank)
            if self.rank == 0:
                err = f'Ranks {faulty_rank} failed to pass monitoredBarrier'
            else:
                err = 'Please check rank 0 logs for faulty rank'
            err += '|Connection closed by peer|Connection reset by peer'
            with self.assertRaisesRegex(RuntimeError, err):
                wrapper_pg.allreduce([tensor])

    def _test_collectives_op_mismatch(self, wrapper_pg, use_cuda=False):
        if False:
            i = 10
            return i + 15
        tensor = torch.randn(20, 10)
        if use_cuda:
            tensor = tensor.to(self.rank)
        works = []
        for _ in range(500):
            work = wrapper_pg.allreduce([tensor])
            works.append(work)
        for w in works:
            w.wait()
        with self.assertRaisesRegex(RuntimeError, '.*') as cm:
            if self.rank == 0:
                wrapper_pg.allreduce([tensor])
            else:
                wrapper_pg.reduce([tensor])
        self._validate_error(exception=cm.exception, op_type='ALLREDUCE' if self.rank == 0 else 'REDUCE', rank=self.rank, tensor=tensor)
        with self.assertRaisesRegex(RuntimeError, '.*') as cm:
            if self.rank == 0:
                wrapper_pg.reduce([tensor])
            else:
                wrapper_pg.barrier()
        self._validate_error(exception=cm.exception, op_type='REDUCE' if self.rank == 0 else 'BARRIER', rank=self.rank, tensor=tensor)
        with self.assertRaisesRegex(RuntimeError, '.*') as cm:
            if self.rank == 0:
                wrapper_pg.broadcast(tensor, 0)
            else:
                output_tensors = [torch.zeros_like(tensor) for _ in range(self.world_size)]
                wrapper_pg.allgather([output_tensors], [tensor])
        self._validate_error(exception=cm.exception, op_type='BROADCAST' if self.rank == 0 else 'ALLGATHER', rank=self.rank, tensor=tensor)

    def _test_collective_shape_mismatch(self, wrapper_pg, use_cuda=False):
        if False:
            i = 10
            return i + 15
        wrapper_pg.barrier()
        dim = 2 if self.rank == 0 else 10
        tensor = torch.randn(20, dim)
        if use_cuda:
            tensor = tensor.to(self.rank)
        with self.assertRaisesRegex(RuntimeError, '.*') as cm:
            wrapper_pg.allreduce([tensor])
        self._validate_error(exception=cm.exception, op_type='ALLREDUCE', rank=self.rank, tensor=tensor)
        tensor = torch.randn(20, 10, 2) if self.rank == 0 else torch.randn(20, 10)
        if use_cuda:
            tensor = tensor.to(self.rank)
        with self.assertRaisesRegex(RuntimeError, '.*') as cm:
            wrapper_pg.allreduce([tensor])
        self._validate_error(exception=cm.exception, op_type='ALLREDUCE', rank=self.rank, tensor=tensor)
        input = [torch.tensor([self.rank] if self.rank == 0 else [self.rank, self.rank], device=self.rank if use_cuda else 'cpu') for _ in range(self.world_size)]
        outputs = [torch.tensor([-1] if self.rank == 0 else [-1, -1], device=self.rank if use_cuda else 'cpu') for _ in range(self.world_size)]
        root_rank = 0
        opts = c10d.ScatterOptions()
        opts.rootRank = root_rank
        with self.assertRaisesRegex(RuntimeError, '.*') as cm:
            if self.rank == root_rank:
                wrapper_pg.scatter([outputs[self.rank]], [input], opts).wait()
            else:
                wrapper_pg.scatter([outputs[self.rank]], [], opts).wait()
        self._validate_error(exception=cm.exception, op_type='SCATTER', rank=self.rank, tensor=outputs[self.rank])
if not TEST_WITH_DEV_DBG_ASAN:

    @requires_gloo()
    @requires_nccl()
    class ProcessGroupNCCLWrapperTest(AbstractProcessGroupWrapperTest):

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            super(AbstractProcessGroupWrapperTest, self).setUp()
            self._spawn_processes()
            os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'

        @property
        def world_size(self) -> int:
            if False:
                for i in range(10):
                    print('nop')
            return 2

        def _create_wrapper_pg(self, with_new_group=False, timeout=10.0):
            if False:
                i = 10
                return i + 15
            store = c10d.FileStore(self.file_name, self.world_size)
            c10d.init_process_group(backend='nccl', rank=self.rank, world_size=self.world_size, store=store, timeout=timedelta(seconds=timeout))
            if with_new_group:
                pg = c10d.new_group(backend='nccl', timeout=timedelta(seconds=timeout))
            else:
                _pg = c10d.ProcessGroupNCCL(store, self.rank, self.world_size, timeout=timedelta(seconds=timeout))
                pg = c10d._create_process_group_wrapper(_pg, 'unused', store, self.rank, self.world_size, timeout=timeout)
            return pg

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        def test_collective_hang(self):
            if False:
                print('Hello World!')
            pg = self._create_wrapper_pg(timeout=2.0)
            self._test_collective_hang(pg)

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @with_dist_debug_levels(levels=['DETAIL'])
        def test_collectives_op_mismatch_debug_mode(self):
            if False:
                print('Hello World!')
            pg = self._create_wrapper_pg(with_new_group=True)
            self._test_collectives_op_mismatch(pg, use_cuda=True)
            self._test_nccl_only_op_mismatch(pg)

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @with_dist_debug_levels(levels=['OFF'])
        def test_collectives_op_mismatch(self):
            if False:
                i = 10
                return i + 15
            pg = self._create_wrapper_pg(with_new_group=False)
            self._test_collectives_op_mismatch(pg, use_cuda=True)
            self._test_nccl_only_op_mismatch(pg)

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @with_dist_debug_levels(levels=['DETAIL'])
        def test_collective_shape_mismatch_debug_mode_detail(self):
            if False:
                return 10
            pg = self._create_wrapper_pg(with_new_group=True)
            self._test_collective_shape_mismatch(pg, use_cuda=True)
            self._test_nccl_only_shape_mismatch(pg)

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @with_dist_debug_levels(levels=['OFF'])
        def test_collective_shape_mismatch_debug_mode_off(self):
            if False:
                for i in range(10):
                    print('nop')
            pg = self._create_wrapper_pg(with_new_group=False)
            self._test_collective_shape_mismatch(pg, use_cuda=True)
            self._test_nccl_only_shape_mismatch(pg)

        def _test_nccl_only_op_mismatch(self, wrapper_pg):
            if False:
                for i in range(10):
                    print('nop')
            device = f'cuda:{self.rank}'
            with self.assertRaisesRegex(RuntimeError, '.*') as cm:
                output = torch.zeros(4 + self.rank, device=device)
                input = torch.ones(4 * self.world_size, device=device)
                if self.rank == 0:
                    wrapper_pg._allgather_base(output, input).wait()
                else:
                    wrapper_pg._reduce_scatter_base(output, input).wait()
            op_type = 'ALLGATHER_BASE' if self.rank == 0 else 'REDUCE_SCATTER_BASE'
            self._validate_error(exception=cm.exception, op_type=op_type, rank=self.rank, tensor=input)

        def _test_nccl_only_shape_mismatch(self, wrapper_pg):
            if False:
                print('Hello World!')
            device = f'cuda:{self.rank}'
            with self.assertRaisesRegex(RuntimeError, '.*') as cm:
                output = torch.zeros(4 + self.rank, device=device)
                input = torch.ones(4 * (self.world_size + 1), device=device)
                wrapper_pg._reduce_scatter_base(output, input).wait()
            self._validate_error(exception=cm.exception, op_type='REDUCE_SCATTER_BASE', rank=self.rank, tensor=input, verify_diff=False)
            with self.assertRaisesRegex(RuntimeError, '.*') as cm:
                output = torch.zeros(4, device=device)
                input = torch.ones((4 + self.rank) * self.world_size, device=device)
                wrapper_pg._reduce_scatter_base(output, input).wait()
            self._validate_error(exception=cm.exception, op_type='REDUCE_SCATTER_BASE', rank=self.rank, tensor=input, verify_diff=False)

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @with_dist_debug_levels(levels=['DETAIL'])
        def test_coalescing_manager_debug_mode_detail(self):
            if False:
                print('Hello World!')
            '\n            Tests that coalescing manager w/TORCH_DISTRIBUTED_DEBUG\n            does not crash: https://github.com/pytorch/pytorch/issues/109520\n            '
            torch.cuda.set_device(self.rank)
            pg = self._create_wrapper_pg(with_new_group=True)
            dev = torch.cuda.current_device()
            pg._start_coalescing(torch.device(dev))
            pg.allreduce([torch.ones(1, device=dev)])
            pg._end_coalescing(torch.device(dev))

@requires_gloo()
class ProcessGroupGlooWrapperTest(AbstractProcessGroupWrapperTest):

    def opts(self, threads=2, timeout=10.0):
        if False:
            while True:
                i = 10
        opts = c10d.ProcessGroupGloo._Options()
        opts._timeout = timeout
        opts._devices = [create_device(interface=LOOPBACK)]
        opts._threads = threads
        return opts

    def _create_wrapper_pg(self, with_new_group=False, timeout=10.0):
        if False:
            while True:
                i = 10
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(backend='gloo', rank=self.rank, world_size=self.world_size, store=store)
        if with_new_group:
            pg = c10d.new_group(backend='gloo')
        else:
            _pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts(timeout=timeout))
            pg = c10d._create_process_group_wrapper(_pg, 'unused', store, self.rank, self.world_size, timeout=timeout)
        return pg

    def test_collective_hang(self):
        if False:
            i = 10
            return i + 15
        pg = self._create_wrapper_pg(timeout=2.0)
        self._test_collective_hang(pg)

    @with_dist_debug_levels(levels=['DETAIL'])
    def test_collectives_op_mismatch_debug_mode(self):
        if False:
            print('Hello World!')
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_collectives_op_mismatch(pg)

    @with_dist_debug_levels(levels=['OFF'])
    def test_collectives_op_mismatch(self):
        if False:
            print('Hello World!')
        pg = self._create_wrapper_pg(with_new_group=False)
        self._test_collectives_op_mismatch(pg)

    @with_dist_debug_levels(levels=['DETAIL'])
    def test_collective_shape_mismatch_debug_mode(self):
        if False:
            i = 10
            return i + 15
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_collective_shape_mismatch(pg)

    @with_dist_debug_levels(levels=['OFF'])
    def test_collective_shape_mismatch_debug_mode_off(self):
        if False:
            while True:
                i = 10
        pg = self._create_wrapper_pg(with_new_group=False)
        self._test_collective_shape_mismatch(pg)

    @skip_if_lt_x_gpu(4)
    @with_dist_debug_levels(levels=['DETAIL'])
    def test_collectives_op_mismatch_cuda_debug_mode(self):
        if False:
            for i in range(10):
                print('nop')
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_collectives_op_mismatch(pg, use_cuda=True)

    @skip_if_lt_x_gpu(4)
    @with_dist_debug_levels(levels=['OFF'])
    def test_collectives_op_mismatch_cuda(self):
        if False:
            return 10
        pg = self._create_wrapper_pg(with_new_group=False)
        self._test_collectives_op_mismatch(pg, use_cuda=True)

    @skip_if_lt_x_gpu(4)
    @with_dist_debug_levels(levels=['DETAIL'])
    def test_collective_shape_mismatch_cuda_debug_mode(self):
        if False:
            print('Hello World!')
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_collective_shape_mismatch(pg, use_cuda=True)

    @skip_if_lt_x_gpu(4)
    @with_dist_debug_levels(levels=['OFF'])
    def test_collective_shape_mismatch_cuda(self):
        if False:
            i = 10
            return i + 15
        pg = self._create_wrapper_pg(with_new_group=False)
        self._test_collective_shape_mismatch(pg, use_cuda=True)
if __name__ == '__main__':
    assert not torch.cuda._initialized, 'test_pg_wrapper must not have initialized CUDA context on main process'
    run_tests()