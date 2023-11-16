import sys
from contextlib import nullcontext
from enum import auto, Enum
from typing import Optional
from unittest.mock import patch
import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import CUDAInitMode, FSDPInitMode, FSDPTest, NestedWrappedModule, TransformerWithSharedParams
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TEST_WITH_DEV_DBG_ASAN
if not dist.is_available():
    print('Distributed not available, skipping tests', file=sys.stderr)
    sys.exit(0)
if TEST_WITH_DEV_DBG_ASAN:
    print('Skip dev-asan as torch + multiprocessing spawn have known issues', file=sys.stderr)
    sys.exit(0)

class PassType(Enum):
    __order__ = 'FWD BWD'
    FWD = auto()
    BWD = auto()

class TestCommunication(FSDPTest):
    """Tests ``FullyShardedDataParallel``'s collective communication usage."""

    def _init_model(self, nested_model: bool, sharding_strategy: ShardingStrategy, device: torch.device):
        if False:
            while True:
                i = 10
        fsdp_kwargs = {'sharding_strategy': sharding_strategy}
        if nested_model:
            model = NestedWrappedModule.init(self.process_group, FSDPInitMode.RECURSIVE, CUDAInitMode.CUDA_AFTER, fsdp_kwargs)
            fsdp_model: FSDP = FSDP(model, self.process_group, **fsdp_kwargs).to(device)
        else:
            fsdp_model: FSDP = TransformerWithSharedParams.init(self.process_group, FSDPInitMode.RECURSIVE, CUDAInitMode.CUDA_BEFORE, fsdp_kwargs)
        return fsdp_model

    def _run_iter(self, fsdp_model, batch, use_no_sync: bool):
        if False:
            print('Hello World!')
        'Runs an iteration inside or outside the ``no_sync()`` context.'
        context = fsdp_model.no_sync() if use_no_sync else nullcontext()
        with context:
            output = fsdp_model(*batch)
            loss = fsdp_model.module.get_loss(batch, output)
            loss.backward()

    def _get_ref_num_reduce_scatters(self, num_fsdp: int, in_no_sync: bool) -> int:
        if False:
            i = 10
            return i + 15
        'Returns the reference number of reduce-scatters for an iteration\n        in the ``no_sync()`` context.'
        return num_fsdp if not in_no_sync else 0

    def _get_ref_num_all_gathers(self, num_fsdp: int, sharding_strategy: Optional[ShardingStrategy], is_first_iter: bool, is_last_iter_no_sync: bool) -> int:
        if False:
            return 10
        'Returns the reference number of all-gathers in an iteration, summing\n        over the forward and backward passes.'
        return sum((self._get_ref_num_all_gathers_in_pass(num_fsdp, sharding_strategy, pass_type, is_first_iter, is_last_iter_no_sync) for pass_type in PassType))

    def _get_ref_num_all_gathers_in_pass(self, num_fsdp: int, sharding_strategy: Optional[ShardingStrategy], pass_type: PassType, is_first_iter: bool, is_last_iter_no_sync: bool):
        if False:
            print('Hello World!')
        'Returns the reference number of all-gathers for a given setting.'
        if sharding_strategy is None:
            sharding_strategy = ShardingStrategy.FULL_SHARD
        if pass_type == PassType.FWD and sharding_strategy == ShardingStrategy.SHARD_GRAD_OP and is_last_iter_no_sync:
            num_all_gathers = 0
        elif pass_type == PassType.FWD:
            num_all_gathers = num_fsdp
        elif pass_type == PassType.BWD and sharding_strategy == ShardingStrategy.FULL_SHARD:
            num_all_gathers = num_fsdp - 1
        elif pass_type == PassType.BWD and sharding_strategy == ShardingStrategy.SHARD_GRAD_OP:
            num_all_gathers = 0
        else:
            assert 0, f'Unsupported: add a branch for pass_type={pass_type} is_first_iter={is_first_iter} is_last_iter_no_sync={is_last_iter_no_sync} sharding_strategy={sharding_strategy}'
        if is_first_iter and pass_type == PassType.FWD:
            num_all_gathers *= 3
        return num_all_gathers

    def _print_ref_num_all_gathers_in_pass(self, num_fsdp: int, sharding_strategy: ShardingStrategy, pass_type: PassType, is_first_iter: bool, is_last_iter_no_sync: bool):
        if False:
            return 10
        'Helper method for printing the number of all-gathers for a specific\n        setting. This may be helpful since the branching is complex.'
        if self.rank != 0:
            return
        num_all_gathers = self._get_ref_num_all_gathers_in_pass(num_fsdp, sharding_strategy, pass_type, is_first_iter, is_last_iter_no_sync)
        print(f'Pass: {pass_type}\nIs First Iteration: {is_first_iter}\nSharding Strategy: {sharding_strategy}\nLast iteration in `no_sync()`: {is_last_iter_no_sync}\nNumber of all-gathers: {num_all_gathers}')

    @skip_if_lt_x_gpu(2)
    @parametrize('nested_model', [False, True])
    @parametrize('use_no_sync', [False, True])
    @parametrize('sharding_strategy', [ShardingStrategy.SHARD_GRAD_OP, None])
    def test_communication(self, nested_model: bool, use_no_sync: bool, sharding_strategy: Optional[ShardingStrategy]):
        if False:
            while True:
                i = 10
        "\n        Tests FSDP's communication cost in terms of calls to collective\n        communication primitives (i.e. all-gather and reduce-scatter).\n\n        Arguments:\n            nested_model (bool): If ``True``, uses ``NestedWrappedModule``,\n                which has nested FSDP instances; if ``False``, uses the default\n                model, which does not have nested FSDP instances.\n            use_no_sync (bool): If ``True``, runs some iterations inside the\n                ``no_sync()`` context manager to accumulate gradients, followed\n                by some iterations outside the context manager; if ``False``,\n                only runs some iterations outside the context manager.\n            sharding_strategy (Optional[ShardingStrategy]): Configures the\n                FSDP algorithm.\n        "
        dist.set_debug_level(dist.DebugLevel.DETAIL)
        device = torch.device('cuda')
        fsdp_model = self._init_model(nested_model, sharding_strategy, device)
        batch = fsdp_model.module.get_input(device)
        num_fsdp = sum((isinstance(m, FSDP) and len(m.params) > 0 for m in fsdp_model.modules()))
        num_iters = 3
        with patch('torch.distributed.all_gather_into_tensor') as mock_all_gather, patch('torch.distributed.reduce_scatter_tensor') as mock_reduce_scatter:

            def reset_mocks():
                if False:
                    print('Hello World!')
                mock_all_gather.reset_mock()
                mock_reduce_scatter.reset_mock()
            if use_no_sync:
                for i in range(num_iters):
                    reset_mocks()
                    self._run_iter(fsdp_model, batch, use_no_sync=True)
                    num_all_gathers = mock_all_gather.call_count
                    num_reduce_scatters = mock_reduce_scatter.call_count
                    ref_num_all_gathers = self._get_ref_num_all_gathers(num_fsdp, sharding_strategy, is_first_iter=i == 0, is_last_iter_no_sync=i > 0)
                    ref_num_reduce_scatters = self._get_ref_num_reduce_scatters(num_fsdp, in_no_sync=True)
                    self.assertEqual(num_all_gathers, ref_num_all_gathers)
                    self.assertEqual(num_reduce_scatters, ref_num_reduce_scatters)
            for i in range(num_iters):
                reset_mocks()
                self._run_iter(fsdp_model, batch, use_no_sync=False)
                num_all_gathers = mock_all_gather.call_count
                num_reduce_scatters = mock_reduce_scatter.call_count
                ref_num_all_gathers = self._get_ref_num_all_gathers(num_fsdp, sharding_strategy, is_first_iter=not use_no_sync and i == 0, is_last_iter_no_sync=use_no_sync and i == 0)
                ref_num_reduce_scatters = self._get_ref_num_reduce_scatters(num_fsdp, in_no_sync=False)
                self.assertEqual(num_all_gathers, ref_num_all_gathers)
                self.assertEqual(num_reduce_scatters, ref_num_reduce_scatters)
instantiate_parametrized_tests(TestCommunication)
if __name__ == '__main__':
    run_tests()