import copy
import sys
from unittest import mock
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
if not dist.is_available():
    print('Distributed not available, skipping tests', file=sys.stderr)
    sys.exit(0)
if TEST_WITH_DEV_DBG_ASAN:
    print('Skip dev-asan as torch + multiprocessing spawn have known issues', file=sys.stderr)
    sys.exit(0)

class TestFSDPFineTune(FSDPTest):
    """Tests fine-tuning cases where some parameters are frozen."""
    NUM_LINEARS = 6

    @property
    def world_size(self) -> int:
        if False:
            while True:
                i = 10
        return 2

    def _init_seq_module(self) -> nn.Module:
        if False:
            for i in range(10):
                print('nop')
        torch.manual_seed(42)
        modules = []
        for _ in range(self.NUM_LINEARS):
            modules += [nn.Linear(5, 5, device='cuda'), nn.ReLU()]
        seq = nn.Sequential(*modules)
        self._set_seq_module_requires_grad(seq, False)
        return seq

    def _set_seq_module_requires_grad(self, seq: nn.Module, requires_grad: bool):
        if False:
            return 10
        for i in range(self.NUM_LINEARS):
            if i % 2 == 0:
                for param in seq[i * 2].parameters(recurse=True):
                    param.requires_grad = requires_grad

    @skip_if_lt_x_gpu(2)
    def test_backward_reshard_hooks(self):
        if False:
            return 10
        '\n        Tests that the post-backward reshard happens even for flat parameters\n        that do not require gradients.\n        '
        self.run_subtests({'sharding_strategy': [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP, ShardingStrategy.NO_SHARD], 'use_orig_params': [False, True], 'inp_requires_grad': [False, True], 'unfreeze_params': [False, True]}, self._test_backward_reshard_hooks)

    def _test_backward_reshard_hooks(self, sharding_strategy: ShardingStrategy, use_orig_params: bool, inp_requires_grad: bool, unfreeze_params: bool):
        if False:
            print('Hello World!')
        seq = self._init_seq_module()
        policy = ModuleWrapPolicy({nn.Linear})
        seq = FSDP(seq, auto_wrap_policy=policy, sharding_strategy=sharding_strategy, use_orig_params=use_orig_params)
        orig_post_backward_reshard = torch.distributed.fsdp._runtime_utils._post_backward_reshard
        post_backward_reshard_count = 0

        def _post_backward_reshard_with_count(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            nonlocal post_backward_reshard_count
            post_backward_reshard_count += 1
            return orig_post_backward_reshard(*args, **kwargs)

        def _assert_post_backward_requires_grad(seq):
            if False:
                print('Hello World!')
            if step_idx == num_steps - 1 and unfreeze_params:
                self.assertTrue(all((p.requires_grad for p in seq.parameters())), msg='Expected all parameters to require grad but some did not!')

        def _assert_post_backward_reshard_count(step_idx, num_steps):
            if False:
                print('Hello World!')
            if step_idx < num_steps - 1 or not unfreeze_params:
                expected_post_backward_reshard_count = self.NUM_LINEARS if inp_requires_grad else self.NUM_LINEARS - 1
            else:
                expected_post_backward_reshard_count = self.NUM_LINEARS
            self.assertEqual(post_backward_reshard_count, expected_post_backward_reshard_count)
        with mock.patch('torch.distributed.fsdp._runtime_utils._post_backward_reshard', _post_backward_reshard_with_count):
            num_steps = 3
            nograd_step_idx = 1
            for step_idx in range(num_steps):
                if unfreeze_params and step_idx == num_steps - 1:
                    self._set_seq_module_requires_grad(seq, True)
                inp = torch.randn((8, 5), device='cuda', requires_grad=inp_requires_grad)
                if step_idx == nograd_step_idx:
                    with torch.no_grad():
                        output = seq(inp)
                else:
                    output = seq(inp)
                if step_idx != nograd_step_idx:
                    output.sum().backward()
                    _assert_post_backward_requires_grad(seq)
                    _assert_post_backward_reshard_count(step_idx, num_steps)
                    post_backward_reshard_count = 0

    def _init_multi_traversal_module(self) -> nn.Module:
        if False:
            while True:
                i = 10
        torch.manual_seed(42)

        class TestModule(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.layer_0 = nn.Linear(5, 5, device='cuda')
                self.layer_no_grad = nn.Linear(5, 5, device='cuda')
                self.layer_with_grad = nn.Linear(5, 5, device='cuda')
                self.layer_no_grad.requires_grad_(False)

            def forward(self, x):
                if False:
                    return 10
                x = self.layer_0(x)
                for _ in range(10):
                    x = self.layer_no_grad(self.layer_with_grad(x))
                    with torch.no_grad():
                        x += self.layer_with_grad(x)
                return x
        return TestModule()

    @skip_if_lt_x_gpu(2)
    def test_hooks_multi_traversal(self):
        if False:
            return 10
        '\n        Tests that the hooks do reshard / unshard correctly in the case of same\n        parameters being used multiple times during forward pass.\n        '
        self.run_subtests({'sharding_strategy': [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP, ShardingStrategy.NO_SHARD], 'use_orig_params': [False, True], 'inp_requires_grad': [False, True], 'forward_prefetch': [False, True]}, self._test_hooks_multi_traversal)

    def _test_hooks_multi_traversal(self, sharding_strategy: ShardingStrategy, use_orig_params: bool, inp_requires_grad: bool, forward_prefetch: bool):
        if False:
            while True:
                i = 10
        seq = self._init_multi_traversal_module()
        policy = ModuleWrapPolicy({nn.Linear})
        fsdp_seq = FSDP(copy.deepcopy(seq), auto_wrap_policy=policy, sharding_strategy=sharding_strategy, use_orig_params=use_orig_params, forward_prefetch=forward_prefetch)
        ddp_seq = DDP(copy.deepcopy(seq), device_ids=[self.rank])
        fsdp_optim = torch.optim.Adam(fsdp_seq.parameters(), lr=0.01)
        ddp_optim = torch.optim.Adam(ddp_seq.parameters(), lr=0.01)
        torch.manual_seed(self.rank + 1)
        losses = []
        for _ in range(6):
            inp = torch.randn((8, 5), device='cuda', requires_grad=inp_requires_grad)
            for (seq, optim) in ((fsdp_seq, fsdp_optim), (ddp_seq, ddp_optim)):
                loss = seq(inp).sum()
                losses.append(loss)
                loss.backward()
                optim.step()
                optim.zero_grad()
            torch.testing.assert_close(losses[0], losses[1])
            losses.clear()

    @skip_if_lt_x_gpu(2)
    def test_parity_with_ddp(self):
        if False:
            print('Hello World!')
        '\n        Tests parity with DDP when mixing flat parameters that require and do\n        not require gradients.\n        '
        self.run_subtests({'sharding_strategy': [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP, ShardingStrategy.NO_SHARD], 'use_orig_params': [False, True]}, self._test_parity_with_ddp)

    def _test_parity_with_ddp(self, sharding_strategy: ShardingStrategy, use_orig_params: bool):
        if False:
            while True:
                i = 10
        seq = self._init_seq_module()
        policy = ModuleWrapPolicy({nn.Linear})
        fsdp_seq = FSDP(copy.deepcopy(seq), auto_wrap_policy=policy, sharding_strategy=sharding_strategy, use_orig_params=use_orig_params)
        ddp_seq = DDP(copy.deepcopy(seq), device_ids=[self.rank])
        fsdp_optim = torch.optim.Adam(fsdp_seq.parameters(), lr=0.01)
        ddp_optim = torch.optim.Adam(ddp_seq.parameters(), lr=0.01)
        torch.manual_seed(self.rank + 1)
        losses = []
        for _ in range(6):
            inp = torch.randn((8, 5), device='cuda')
            for (seq, optim) in ((fsdp_seq, fsdp_optim), (ddp_seq, ddp_optim)):
                loss = seq(inp).sum()
                losses.append(loss)
                loss.backward()
                optim.step()
                optim.zero_grad()
            torch.testing.assert_close(losses[0], losses[1])
            losses.clear()
if __name__ == '__main__':
    run_tests()