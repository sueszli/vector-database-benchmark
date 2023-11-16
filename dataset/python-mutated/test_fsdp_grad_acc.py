import contextlib
import itertools
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import distributed as dist
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch, ShardingStrategy
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import CUDAInitMode, FSDPInitMode, FSDPTest, TransformerWithSharedParams
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TEST_WITH_DEV_DBG_ASAN
if not dist.is_available():
    print('Distributed not available, skipping tests', file=sys.stderr)
    sys.exit(0)
if TEST_WITH_DEV_DBG_ASAN:
    print('Skip dev-asan as torch + multiprocessing spawn have known issues', file=sys.stderr)
    sys.exit(0)

@dataclass
class _GradAccConfig:
    """
    This configures how gradients are accumulated in :meth:`_test_grad_acc`.
    Each instance of this class represents ``num_iters``-many consecutive
    iterations, where the ``no_sync()`` context manager is used or not as given
    by ``use_no_sync``.

    Attributes:
        use_no_sync (bool): Indicates whether to use the ``no_sync()`` context
            manager as the way to accumulate gradients.
        num_iters (int): Number of iterations to accumulate gradients.
    """
    use_no_sync: bool
    num_iters: int

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'(use_no_sync={self.use_no_sync},num_iters={self.num_iters})'

@dataclass
class _GradAccConfigs:
    """
    This wraps a :class:`list` of :class:`_GradAccConfig` instances with the
    sole purpose of overriding :meth:`__repr__` to remove spaces.
    """
    configs: List[_GradAccConfig]

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return '[' + ','.join((config.__repr__() for config in self.configs)) + ']'

class TestGradAcc(FSDPTest):
    """Tests ``FullyShardedDataParallel``'s gradient accumulation via both its
    ``no_sync()`` context manager and without the context manager."""

    @property
    def world_size(self) -> int:
        if False:
            print('Hello World!')
        return 2

    def _test_grad_acc(self, batch_dim: int, configs: List[_GradAccConfig], cpu_offload: CPUOffload, backward_prefetch: Optional[BackwardPrefetch], sharding_strategy: ShardingStrategy, use_orig_params: bool):
        if False:
            i = 10
            return i + 15
        "\n        Tests gradient accumulation by comparing a run that trains sequentially\n        through some batches while accumulating gradients with a run that\n        trains on the concatenation of those batches in a single iteration.\n\n        The last iteration always synchronizes gradients regardless of what is\n        specified by the last element of ``configs``.\n\n        Arguments:\n            batch_dim (int): Batch dimension in the input tensor to be passed\n                into the model for the forward pass.\n            configs (List[_GradAccConfig]): :class:`list` of configurations\n                specifying how gradients are accumulated; for example, a list\n                corresponding to [(False, 2), (True, 2), (False, 2)] indicates\n                to accumulate over 2 + 2 + 2 = 6 total iterations, where the\n                first two do not use ``no_sync()``, the middle two do use\n                ``no_sync()``, and the final two again do not use\n                ``no_sync()``.\n            cpu_offload (CPUOffload): Configures CPU offloading.\n            backward_prefetch (Optional[BackwardPrefetch]): Specifies at which\n                point to prefetch the next layer's full parameters during the\n                backward pass, if at all.\n        "
        fsdp_kwargs = {'cpu_offload': cpu_offload, 'backward_prefetch': backward_prefetch, 'sharding_strategy': sharding_strategy, 'use_orig_params': use_orig_params}
        fsdp_model: FSDP = TransformerWithSharedParams.init(self.process_group, FSDPInitMode.RECURSIVE, CUDAInitMode.CUDA_BEFORE, fsdp_kwargs, deterministic=True, add_bn=False)
        device = torch.device('cuda')
        optim = torch.optim.SGD(fsdp_model.parameters(), lr=0.01, momentum=0.9)

        def permute_tensor(x: torch.Tensor):
            if False:
                print('Hello World!')
            return x.view(-1)[torch.randperm(x.numel())].view_as(x)
        batch: Tuple[torch.Tensor, ...] = fsdp_model.module.get_input(device)
        batches: List[Tuple[torch.Tensor, ...]] = [batch]
        num_iters_to_acc = sum((config.num_iters for config in configs))
        for _ in range(num_iters_to_acc - 1):
            batches.append(tuple((permute_tensor(t) for t in batch)))
        for (batch1, batch2) in itertools.combinations(batches, r=2):
            for (t1, t2) in zip(batch1, batch2):
                assert not torch.all(t1 == t2), 'Check the test to make sure that batches are distinct'
        concat_batch: Tuple[torch.Tensor, ...] = tuple((torch.cat(ts, dim=batch_dim) for ts in zip(*batches)))
        fsdp_model.zero_grad()
        output = fsdp_model(*concat_batch)
        ref_loss = fsdp_model.module.get_loss(concat_batch, output)
        ref_loss.backward()
        ref_grads = [p.grad.detach().clone() for p in fsdp_model.parameters() if p.grad is not None]
        fsdp_model.zero_grad()
        losses = []
        batch_idx = 0
        for config in configs:
            sync_context = fsdp_model.no_sync() if config.use_no_sync else contextlib.nullcontext()
            with sync_context:
                for _ in range(config.num_iters):
                    if batch_idx == num_iters_to_acc - 1:
                        break
                    batch = batches[batch_idx]
                    batch_idx += 1
                    output = fsdp_model(*batch)
                    loss = fsdp_model.module.get_loss(batch, output)
                    loss.backward()
                    losses.append(loss)
        output = fsdp_model(*batches[-1])
        loss = fsdp_model.module.get_loss(batches[-1], output)
        loss.backward()
        losses.append(loss)
        acc_loss = sum(losses)
        acc_grads = [p.grad.detach().clone() for p in fsdp_model.parameters() if p.grad is not None]
        torch.testing.assert_close(ref_loss, acc_loss)
        self.assertEqual(len(ref_grads), len(acc_grads))
        for (ref_grad, acc_grad) in zip(ref_grads, acc_grads):
            self.assertEqual(ref_grad.device, acc_grad.device)
            self.assertEqual(ref_grad.size(), acc_grad.size())
            self.assertEqual(ref_grad.dtype, acc_grad.dtype)
            torch.testing.assert_close(ref_grad, acc_grad)
        optim.step()

    def _get_subtest_config(self) -> Dict[str, List[Any]]:
        if False:
            return 10
        'Returns a subtest configuration that subtests prefetching.'
        return {'backward_prefetch': [None, BackwardPrefetch.BACKWARD_PRE, BackwardPrefetch.BACKWARD_POST], 'sharding_strategy': [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP, ShardingStrategy.NO_SHARD]}

    @skip_if_lt_x_gpu(2)
    @parametrize('configs', [_GradAccConfigs([_GradAccConfig(use_no_sync=True, num_iters=3), _GradAccConfig(use_no_sync=False, num_iters=3), _GradAccConfig(use_no_sync=True, num_iters=3)]), _GradAccConfigs([_GradAccConfig(use_no_sync=False, num_iters=3), _GradAccConfig(use_no_sync=True, num_iters=3), _GradAccConfig(use_no_sync=False, num_iters=3)])])
    @parametrize('use_orig_params', [False, True])
    def test_grad_acc(self, configs: _GradAccConfigs, use_orig_params: bool):
        if False:
            print('Hello World!')
        '\n        Tests gradient accumulation without parameter CPU offloading.\n\n        This exercises gradient accumulation inside and outside the\n        ``no_sync()`` context manager, in particular by interleaving the two.\n        It tests both interleaving starting with (and ending with, resp.)\n        inside versus outside ``no_sync()`` to ensure that initial conditions\n        (and final conditions, resp.) do not affect the correctness.\n        '
        subtest_config = self._get_subtest_config()
        subtest_config['cpu_offload'] = [CPUOffload(offload_params=False)]
        self.run_subtests(subtest_config, self._test_grad_acc, batch_dim=1, configs=configs.configs, use_orig_params=use_orig_params)

    @skip_if_lt_x_gpu(2)
    @parametrize('use_orig_params', [False, True])
    def test_grad_acc_cpu_offload(self, use_orig_params: bool):
        if False:
            i = 10
            return i + 15
        '\n        Tests gradient accumulation with parameter CPU offloading.\n\n        NOTE: Gradient accumulation without using the ``no_sync()`` context\n        manager is not currently compatible with CPU offloading.\n        '
        configs = _GradAccConfigs([_GradAccConfig(use_no_sync=True, num_iters=3)])
        subtest_config = self._get_subtest_config()
        subtest_config['cpu_offload'] = [CPUOffload(offload_params=True)]
        self.run_subtests(subtest_config, self._test_grad_acc, batch_dim=1, configs=configs.configs, use_orig_params=use_orig_params)
instantiate_parametrized_tests(TestGradAcc)
if __name__ == '__main__':
    run_tests()