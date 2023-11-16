import copy
import functools
import itertools
import os
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple, Type
import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.fsdp import BackwardPrefetch, CPUOffload, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp._common_utils import clean_tensor_name
from torch.distributed.fsdp._flat_param import _FSDP_SKIP_WRITEBACK_CHECK, _FSDP_USE_FULL_PREC_IN_EVAL
from torch.distributed.fsdp._init_utils import NO_RESHARD_AFTER_FORWARD_STRATEGIES
from torch.distributed.fsdp.wrap import always_wrap_policy, ModuleWrapPolicy
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import CUDAInitMode, FSDPInitMode, FSDPTest, TransformerWithSharedParams
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TEST_WITH_DEV_DBG_ASAN, TestCase
if not dist.is_available():
    print('Distributed not available, skipping tests', file=sys.stderr)
    sys.exit(0)
if TEST_WITH_DEV_DBG_ASAN:
    print('Skip dev-asan as torch + multiprocessing spawn have known issues', file=sys.stderr)
    sys.exit(0)

class TestFSDPUseOrigParamsMultipleParamGroups(FSDPTest):
    """Tests multiple parameter groups."""

    @property
    def world_size(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 2

    def _get_param_groups(self, model: nn.Module) -> List[Dict[str, Any]]:
        if False:
            return 10
        '\n        Constructs separate parameter groups for weights, biases, and other\n        parameters.\n        '
        param_groups = [{'params': [], 'weight_decay': 0.1, 'lr': 0.01}, {'params': [], 'weight_decay': 0.01, 'lr': 0.001}, {'params': []}]
        for (param_name, param) in model.named_parameters():
            if 'weight' in param_name:
                param_groups[0]['params'].append(param)
            elif 'bias' in param_name:
                param_groups[1]['params'].append(param)
            else:
                param_groups[2]['params'].append(param)
        return param_groups

    def _get_optim(self, model: nn.Module, optim_class: Type[torch.optim.Optimizer], multi_tensor: bool) -> torch.optim.Optimizer:
        if False:
            for i in range(10):
                print('nop')
        '\n        Constructs an Adam optimizer with three parameter groups, one for\n        weights, one for biases, and one for everything else, each with\n        different weight decay and learning rates.\n        '
        param_groups = self._get_param_groups(model)
        return optim_class(param_groups, lr=0.005, foreach=multi_tensor)

    def _get_ddp_transformer(self, find_unused_params: bool) -> DDP:
        if False:
            while True:
                i = 10
        'Returns a transformer with shared parameters wrapped with DDP.'
        model = TransformerWithSharedParams.init(self.process_group, FSDPInitMode.NO_FSDP, CUDAInitMode.CUDA_BEFORE, deterministic=True)
        ddp_model = DDP(model, device_ids=[self.rank], find_unused_parameters=find_unused_params)
        return ddp_model

    def _get_fsdp_transformer_and_optim(self, cuda_init_mode: CUDAInitMode, init_optim_before_wrap: bool, optim_class: Type[torch.optim.Optimizer], multi_tensor: bool, sharding_strategy: ShardingStrategy, backward_prefetch: Optional[BackwardPrefetch], cpu_offload: CPUOffload) -> Tuple[FSDP, torch.optim.Optimizer]:
        if False:
            print('Hello World!')
        '\n        Returns a transformer with shared parameters wrapped with FSDP and a\n        corresponding optimizer.\n        '
        fsdp_kwargs = {'auto_wrap_policy': ModuleWrapPolicy({TransformerEncoderLayer, TransformerDecoderLayer}), 'use_orig_params': True, 'sharding_strategy': sharding_strategy, 'backward_prefetch': backward_prefetch, 'cpu_offload': cpu_offload}
        model = TransformerWithSharedParams.init(self.process_group, FSDPInitMode.NO_FSDP, cuda_init_mode, deterministic=True)
        if init_optim_before_wrap:
            fsdp_optim = self._get_optim(model, optim_class, multi_tensor)
            fsdp_model = FSDP(model, self.process_group, **fsdp_kwargs)
        else:
            fsdp_model = FSDP(model, self.process_group, **fsdp_kwargs)
            fsdp_optim = self._get_optim(fsdp_model, optim_class, multi_tensor)
        if cuda_init_mode == CUDAInitMode.CUDA_AFTER and (not fsdp_model.cpu_offload.offload_params):
            fsdp_model = fsdp_model.cuda()
        return (fsdp_model, fsdp_optim)

    def _check_train_parity(self, ddp_model: DDP, ddp_optim: torch.optim.Optimizer, fsdp_model: FSDP, fsdp_optim: torch.optim.Optimizer, set_to_none: bool, num_iters: int=10):
        if False:
            while True:
                i = 10
        'Checks training parity between DDP and FSDP.'
        device = torch.device('cuda')
        for i in range(num_iters):
            iter_losses = []
            for (model, optim) in ((ddp_model, ddp_optim), (fsdp_model, fsdp_optim)):
                module = model.module
                if i % 2 == 0:
                    optim.zero_grad(set_to_none=set_to_none)
                inp = module.get_input(device)
                output = model(*inp)
                loss = module.get_loss(inp, output).to(device)
                iter_losses.append(loss)
                if i % 2 == 1:
                    optim.zero_grad(set_to_none=set_to_none)
                module.run_backward(loss)
                if model is ddp_model and fsdp_model.cpu_offload.offload_params:
                    model.to(torch.device('cpu'))
                optim.step()
                if model is ddp_model and fsdp_model.cpu_offload.offload_params:
                    model.to(device)
            torch.testing.assert_close(iter_losses[0], iter_losses[1])
            iter_losses.clear()
        self._check_ddp_fsdp_param_parity(ddp_model, fsdp_model)

    def _check_ddp_fsdp_param_parity(self, ddp_model: DDP, fsdp_model: FSDP):
        if False:
            i = 10
            return i + 15
        with FSDP.summon_full_params(fsdp_model):
            for ((n1, p1), (n2, p2)) in zip(ddp_model.module.named_parameters(), fsdp_model.named_parameters()):
                self.assertEqual(n1, clean_tensor_name(n2))
                torch.testing.assert_close(p1, p2)

    def _get_sharding_strategy_from_str(self, sharding_strategy_str: str) -> ShardingStrategy:
        if False:
            for i in range(10):
                print('nop')
        if sharding_strategy_str == 'no_shard':
            sharding_strategy = ShardingStrategy.NO_SHARD
        elif sharding_strategy_str == 'shard_grad_op':
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        elif sharding_strategy_str == 'full_shard':
            sharding_strategy = ShardingStrategy.FULL_SHARD
        else:
            raise ValueError(f'Invalid string: {sharding_strategy_str}')
        return sharding_strategy

    @skip_if_lt_x_gpu(2)
    def test_fsdp_compile(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_subtests({'sharding_strategy': [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP, ShardingStrategy.NO_SHARD], 'skip_fsdp_guards': [True, False]}, self._test_fsdp_compile)

    def _test_fsdp_compile(self, sharding_strategy: ShardingStrategy, skip_fsdp_guards: bool):
        if False:
            i = 10
            return i + 15
        torch._dynamo.config.skip_fsdp_guards = skip_fsdp_guards
        fsdp_kwargs = {'auto_wrap_policy': ModuleWrapPolicy({TransformerEncoderLayer, TransformerDecoderLayer}), 'use_orig_params': True, 'sharding_strategy': sharding_strategy, 'backward_prefetch': BackwardPrefetch.BACKWARD_PRE, 'cpu_offload': CPUOffload(False)}
        base_model = TransformerWithSharedParams.init(self.process_group, FSDPInitMode.NO_FSDP, CUDAInitMode.CUDA_BEFORE, deterministic=True)
        ref_model = FSDP(copy.deepcopy(base_model), self.process_group, **fsdp_kwargs)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=0.01)
        model = FSDP(copy.deepcopy(base_model), self.process_group, **fsdp_kwargs)
        model = torch.compile(model)
        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        for i in range(10):
            losses = []
            inp = ref_model.get_input(torch.device('cuda'))
            for (_model, _optim) in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad()
                loss = _model(*inp).sum()
                losses.append(loss)
                loss.backward()
                _optim.step()
            self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(2)
    @parametrize('sharding_strategy_str', ['no_shard', 'shard_grad_op', 'full_shard'])
    def test_diff_hyperparams(self, sharding_strategy_str: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests FSDP parity with DDP when using multiple parameter groups with\n        different hyperparameter settings.\n        '
        sharding_strategy = self._get_sharding_strategy_from_str(sharding_strategy_str)
        self.run_subtests({'cuda_init_mode': [CUDAInitMode.CUDA_BEFORE, CUDAInitMode.CUDA_AFTER], 'init_optim_before_wrap': [False, True], 'optim_class': [torch.optim.AdamW], 'multi_tensor': [False, True], 'set_to_none': [False, True], 'backward_prefetch': [None, BackwardPrefetch.BACKWARD_PRE, BackwardPrefetch.BACKWARD_POST], 'skip_writeback_check': [False, True]}, self._test_diff_hyperparams, cpu_offload=CPUOffload(offload_params=False), sharding_strategy=sharding_strategy)

    @skip_if_lt_x_gpu(2)
    @parametrize('sharding_strategy_str', ['no_shard', 'shard_grad_op', 'full_shard'])
    def test_diff_hyperparams_cpu_offload(self, sharding_strategy_str: str):
        if False:
            return 10
        '\n        Tests FSDP parity with DDP when using multiple parameter groups with\n        different hyperparameter settings with CPU offloading enabled. This is\n        separate from :meth:`test_diff_hyperparams` because CPU offloading has\n        some issues with subtesting for some specific subtesting configs (e.g.,\n        with ``offload_params=False`` followed by ``True`` but not vice versa).\n        '
        sharding_strategy = self._get_sharding_strategy_from_str(sharding_strategy_str)
        for skip_writeback_check in (False, True):
            self._test_diff_hyperparams(cuda_init_mode=CUDAInitMode.CUDA_BEFORE, init_optim_before_wrap=False, optim_class=torch.optim.Adam, multi_tensor=False, set_to_none=False, backward_prefetch=BackwardPrefetch.BACKWARD_PRE, cpu_offload=CPUOffload(offload_params=True), sharding_strategy=sharding_strategy, skip_writeback_check=skip_writeback_check)

    def _test_diff_hyperparams(self, cuda_init_mode: CUDAInitMode, init_optim_before_wrap: bool, optim_class: Type[torch.optim.Optimizer], multi_tensor: bool, set_to_none: bool, backward_prefetch: Optional[BackwardPrefetch], cpu_offload: CPUOffload, sharding_strategy: ShardingStrategy, skip_writeback_check: bool):
        if False:
            while True:
                i = 10
        '\n        Args:\n            init_optim_before_wrap (bool): If ``True``, initializes the\n                FSDP optimizer before wrapping the model with FSDP; otherwise,\n                initializes the FSDP optimizer after wrapping the model with\n                FSDP. We permit both forms of initialization to give users\n                flexibility.\n        '
        if cuda_init_mode == CUDAInitMode.CUDA_AFTER and cpu_offload.offload_params:
            return
        if skip_writeback_check:
            os.environ[_FSDP_SKIP_WRITEBACK_CHECK] = '1'
        ddp_model = self._get_ddp_transformer(find_unused_params=False)
        ddp_optim = self._get_optim(ddp_model, optim_class, multi_tensor)
        (fsdp_model, fsdp_optim) = self._get_fsdp_transformer_and_optim(cuda_init_mode=cuda_init_mode, init_optim_before_wrap=init_optim_before_wrap, optim_class=optim_class, multi_tensor=multi_tensor, sharding_strategy=sharding_strategy, backward_prefetch=backward_prefetch, cpu_offload=cpu_offload)
        self._check_train_parity(ddp_model, ddp_optim, fsdp_model, fsdp_optim, set_to_none)

    @skip_if_lt_x_gpu(2)
    def test_diff_trainability(self):
        if False:
            print('Hello World!')
        '\n        Tests FSDP parity with DDP when using multiple parameter groups and\n        freezing the parameters in one parameter group.\n        '
        self.run_subtests({'multi_tensor': [False, True], 'sharding_strategy': [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP, ShardingStrategy.NO_SHARD]}, self._test_diff_trainability)

    def _test_diff_trainability(self, multi_tensor: bool, sharding_strategy: ShardingStrategy):
        if False:
            return 10
        optim_class = torch.optim.Adam
        ddp_model = self._get_ddp_transformer(find_unused_params=True)
        ddp_optim = self._get_optim(ddp_model, optim_class, multi_tensor)
        (fsdp_model, fsdp_optim) = self._get_fsdp_transformer_and_optim(cuda_init_mode=CUDAInitMode.CUDA_BEFORE, init_optim_before_wrap=False, optim_class=optim_class, multi_tensor=multi_tensor, sharding_strategy=sharding_strategy, backward_prefetch=BackwardPrefetch.BACKWARD_PRE, cpu_offload=None)
        for (param_name, param) in ddp_model.named_parameters():
            if 'bias' in param_name:
                param.requires_grad_(False)
        for (param_name, param) in fsdp_model.named_parameters():
            if 'bias' in param_name:
                param.requires_grad_(False)
        self._check_train_parity(ddp_model, ddp_optim, fsdp_model, fsdp_optim, False)

    @skip_if_lt_x_gpu(2)
    def test_multiple_optimizers(self):
        if False:
            while True:
                i = 10
        '\n        Tests using two optimizers where only one sets gradients to ``None``.\n        '
        self.run_subtests({'sharding_strategy': [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP]}, self._test_multiple_optimizers)

    def _test_multiple_optimizers(self, sharding_strategy: ShardingStrategy):
        if False:
            i = 10
            return i + 15
        ddp_model = self._get_ddp_transformer(find_unused_params=True)
        ddp_param_groups = self._get_param_groups(ddp_model)
        assert len(ddp_param_groups) == 3, f'{len(ddp_param_groups)}'
        (fsdp_model, _) = self._get_fsdp_transformer_and_optim(cuda_init_mode=CUDAInitMode.CUDA_BEFORE, init_optim_before_wrap=False, optim_class=torch.optim.Adam, multi_tensor=False, sharding_strategy=sharding_strategy, backward_prefetch=BackwardPrefetch.BACKWARD_PRE, cpu_offload=None)
        fsdp_param_groups = self._get_param_groups(fsdp_model)
        assert len(fsdp_param_groups) == 3, f'{len(fsdp_param_groups)}'
        ddp_optims = []
        fsdp_optims = []
        optim_ctors = [functools.partial(torch.optim.Adam, lr=0.005), functools.partial(torch.optim.AdamW, lr=0.01)]
        for (optim_ctor, ddp_param_group, fsdp_param_group) in zip(optim_ctors, ddp_param_groups[:2], fsdp_param_groups[:2]):
            ddp_optims.append(optim_ctor(ddp_param_group['params']))
            fsdp_optims.append(optim_ctor(fsdp_param_group['params']))
        device = torch.device('cuda')
        has_both = False
        for fsdp_module in FSDP.fsdp_modules(fsdp_model):
            handle = fsdp_module._handle
            if not handle:
                continue
            flat_param = handle.flat_param
            assert flat_param._params is not None
            has_weight = False
            has_bias = False
            for (param, fqn) in zip(flat_param._params, flat_param._fqns):
                if 'weight' in fqn and param.numel() > 0:
                    has_weight = True
                elif 'bias' in fqn and param.numel() > 0:
                    has_bias = True
            has_both |= has_weight and has_bias
        assert has_both, f'Rank {self.rank} does not have a `FlatParameter` with both a weight and a bias in its shard, meaning that this test is vacuous'

        def run_iter():
            if False:
                print('Hello World!')
            iter_losses = []
            for (model, optims) in ((ddp_model, ddp_optims), (fsdp_model, fsdp_optims)):
                module = model.module
                inp = module.get_input(device)
                output = model(*inp)
                loss = module.get_loss(inp, output).to(device)
                iter_losses.append(loss)
                module.run_backward(loss)
                for optim in optims:
                    optim.step()
            torch.testing.assert_close(iter_losses[0], iter_losses[1])
            iter_losses.clear()
            self._check_ddp_fsdp_param_parity(ddp_model, fsdp_model)
        run_iter()
        ddp_optims[0].zero_grad(set_to_none=True)
        fsdp_optims[0].zero_grad(set_to_none=True)
        inp = ddp_model.module.get_input(device)
        ddp_output = ddp_model(*inp)
        fsdp_output = fsdp_model(*inp)
        if sharding_strategy in NO_RESHARD_AFTER_FORWARD_STRATEGIES:
            return
        for ((ddp_n, ddp_p), (fsdp_n, fsdp_p)) in zip(ddp_model.module.named_parameters(), fsdp_model.named_parameters()):
            self.assertEqual(ddp_n, clean_tensor_name(fsdp_n))
            if fsdp_p.numel() == 0:
                self.assertTrue(fsdp_p.grad is None)
                continue
            if ddp_p.grad is None:
                self.assertTrue(fsdp_p.grad is None)
            else:
                self.assertEqual(ddp_p.flatten(), fsdp_p.flatten())
                self.assertEqual(ddp_p.grad.flatten(), fsdp_p.grad.flatten())
        self._check_ddp_fsdp_param_parity(ddp_model, fsdp_model)
        ddp_loss = ddp_model.module.get_loss(inp, ddp_output).to(device)
        fsdp_loss = fsdp_model.module.get_loss(inp, fsdp_output).to(device)
        ddp_model.module.run_backward(ddp_loss)
        fsdp_model.module.run_backward(fsdp_loss)
        for optim in itertools.chain(ddp_optims, fsdp_optims):
            optim.step()
        self._check_ddp_fsdp_param_parity(ddp_model, fsdp_model)
        run_iter()
        self._check_ddp_fsdp_param_parity(ddp_model, fsdp_model)

class TestFSDPUseOrigParamsUnshardReshard(FSDPTest):
    """Tests the unshard/reshard flow."""

    @property
    def world_size(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 2

    def _get_fsdp_models_and_optims(self, sharding_strategy: ShardingStrategy, cpu_offload: CPUOffload) -> Tuple[FSDP, torch.optim.Optimizer, FSDP, torch.optim.Optimizer]:
        if False:
            return 10
        '\n        Returns a pair of (FSDP model, optimizer) for ``use_orig_params=False``\n        and ``True``, respectively.\n        '
        LR = 0.01
        fsdp_kwargs = {'sharding_strategy': sharding_strategy, 'cpu_offload': cpu_offload, 'use_orig_params': False}
        fsdp_model = TransformerWithSharedParams.init(self.process_group, FSDPInitMode.RECURSIVE, CUDAInitMode.CUDA_BEFORE, fsdp_kwargs=fsdp_kwargs, deterministic=True)
        optim = torch.optim.Adam(fsdp_model.parameters(), foreach=False, lr=LR)
        fsdp_kwargs['use_orig_params'] = True
        fsdp_model_orig_params = TransformerWithSharedParams.init(self.process_group, FSDPInitMode.RECURSIVE, CUDAInitMode.CUDA_BEFORE, fsdp_kwargs=fsdp_kwargs, deterministic=True)
        optim_orig_params = torch.optim.Adam(fsdp_model_orig_params.parameters(), foreach=False, lr=LR)
        return (fsdp_model, optim, fsdp_model_orig_params, optim_orig_params)

    def _check_fsdp_parameter_parity(self, fsdp1: FSDP, fsdp2: FSDP) -> None:
        if False:
            return 10
        'Checks that two FSDP instances have the same model parameters.'
        with FSDP.summon_full_params(fsdp1), FSDP.summon_full_params(fsdp2):
            for ((n1, p1), (n2, p2)) in zip(fsdp1.named_parameters(), fsdp2.named_parameters()):
                self.assertEqual(n1, n2)
                torch.testing.assert_close(p1, p2)

    def _get_fsdp_parity_subtest_config(self):
        if False:
            while True:
                i = 10
        return {'sharding_strategy': [ShardingStrategy.NO_SHARD, ShardingStrategy.SHARD_GRAD_OP, ShardingStrategy.FULL_SHARD]}

    @skip_if_lt_x_gpu(2)
    @parametrize('offload_params', [False, True])
    def test_multiple_forward(self, offload_params: bool):
        if False:
            return 10
        '\n        Tests that ``use_orig_params=True`` has parity with ``False`` when\n        running multiple forward passes before a backward pass.\n        '
        cpu_offload = CPUOffload(offload_params=offload_params)
        self.run_subtests(self._get_fsdp_parity_subtest_config(), self._test_multiple_forward, cpu_offload=cpu_offload)

    @skip_if_lt_x_gpu(2)
    def _test_multiple_forward(self, sharding_strategy: ShardingStrategy, cpu_offload: CPUOffload):
        if False:
            while True:
                i = 10
        (fsdp_model, optim, fsdp_model_orig_params, optim_orig_params) = self._get_fsdp_models_and_optims(sharding_strategy, cpu_offload)
        device = torch.device('cuda')
        for _ in range(3):
            inp1 = fsdp_model.get_input(device)
            _inp2 = fsdp_model.get_input(device)
            inp2 = tuple((t + torch.ones_like(t) for t in _inp2))
            losses1 = []
            losses2 = []
            losses = []
            for (_model, _optim) in ((fsdp_model, optim), (fsdp_model_orig_params, optim_orig_params)):
                _optim.zero_grad()
                loss1 = _model(*inp1)
                losses1.append(loss1)
                loss2 = _model(*inp2)
                losses2.append(loss2)
                loss = (loss1 + loss2).sum()
                losses.append(loss)
                _model.run_backward(loss)
                _optim.step()
            self.assertEqual(losses1[0], losses1[1])
            self.assertEqual(losses2[0], losses2[1])
            self.assertEqual(losses[0], losses[1])
        self._check_fsdp_parameter_parity(fsdp_model, fsdp_model_orig_params)

    @skip_if_lt_x_gpu(2)
    @parametrize('offload_params', [False, True])
    def test_summon_between_two_forwards(self, offload_params: bool):
        if False:
            print('Hello World!')
        '\n        Tests that ``use_orig_params=True`` has parity with ``False`` when\n        running a forward pass, :meth:`summon_full_params()`, and another\n        forward pass before a backward pass.\n        '
        cpu_offload = CPUOffload(offload_params=offload_params)
        self.run_subtests(self._get_fsdp_parity_subtest_config(), self._test_summon_between_two_forwards, cpu_offload=cpu_offload)

    def _test_summon_between_two_forwards(self, sharding_strategy: ShardingStrategy, cpu_offload: CPUOffload):
        if False:
            return 10
        (fsdp_model, optim, fsdp_model_orig_params, optim_orig_params) = self._get_fsdp_models_and_optims(sharding_strategy, cpu_offload)
        device = torch.device('cuda')
        for _ in range(3):
            optim.zero_grad()
            optim_orig_params.zero_grad()
            inp1 = fsdp_model.get_input(device)
            loss1 = fsdp_model(*inp1)
            loss_orig_params1 = fsdp_model_orig_params(*inp1)
            self.assertEqual(loss1, loss_orig_params1)
            self._check_fsdp_parameter_parity(fsdp_model, fsdp_model_orig_params)
            inp2 = fsdp_model.get_input(device)
            loss2 = fsdp_model(*inp2)
            loss_orig_params2 = fsdp_model_orig_params(*inp2)
            self.assertEqual(loss2, loss_orig_params2)
            loss = (loss1 + loss2).sum()
            loss_orig_params = (loss_orig_params1 + loss_orig_params2).sum()
            fsdp_model.run_backward(loss)
            fsdp_model_orig_params.run_backward(loss_orig_params)
            optim.step()
            optim_orig_params.step()
        self._check_fsdp_parameter_parity(fsdp_model, fsdp_model_orig_params)

class TestFSDPUseOrigParamsParamAccess(FSDPTest):
    """Tests original parameter access."""

    @property
    def world_size(self):
        if False:
            print('Hello World!')
        return 2

    @skip_if_lt_x_gpu(2)
    def test_access_params_after_forward(self):
        if False:
            print('Hello World!')
        '\n        Tests that accessing the original parameters after the forward but\n        before the backward. Notably, this is not supported when\n        ``use_orig_params=False``. However, for ``True``, FSDP exposes the\n        (flattened) sharded original parameters, making it possible.\n        '
        self.run_subtests({'sharding_strategy': [ShardingStrategy.NO_SHARD, ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP]}, self._test_access_params_after_forward)

    def _test_access_params_after_forward(self, sharding_strategy: ShardingStrategy):
        if False:
            i = 10
            return i + 15

        class Model(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                torch.manual_seed(42)
                self.lin1 = nn.Linear(5, 5, bias=False)
                self.lin2 = nn.Linear(5, 7)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if False:
                    i = 10
                    return i + 15
                z = self.lin1(x)
                z = nn.functional.relu(z)
                z = self.lin2(z)
                return z

            def get_input(self, device: torch.device) -> Tuple[torch.Tensor, ...]:
                if False:
                    for i in range(10):
                        print('nop')
                return (torch.randn((2, 5)).to(device),)

            def get_loss(self, inp, out):
                if False:
                    while True:
                        i = 10
                return out.sum()

        def check_parameter_parity(ddp_model: DDP, fsdp_model: FSDP, between_fwd_and_bwd: bool):
            if False:
                i = 10
                return i + 15
            assert self.rank in (0, 1), f'Expects world size of 2 but got {self.world_size}'
            for ((n1, p1), (n2, p2)) in zip(ddp_model.module.named_parameters(), fsdp_model.named_parameters()):
                self.assertEqual(n1, clean_tensor_name(n2))
                if sharding_strategy == ShardingStrategy.NO_SHARD:
                    pass
                elif between_fwd_and_bwd and sharding_strategy in NO_RESHARD_AFTER_FORWARD_STRATEGIES:
                    pass
                elif n1 == 'lin1.weight':
                    if self.rank == 0:
                        p1 = p1.flatten()[:13]
                    elif self.rank == 1:
                        p1 = p1.flatten()[13:]
                elif n1 == 'lin2.weight':
                    if self.rank == 0:
                        p1 = p1.flatten()[:22]
                    elif self.rank == 1:
                        p1 = p1.flatten()[22:]
                elif n1 == 'lin2.bias':
                    if self.rank == 0:
                        p1 = torch.empty(0, device=p1.device)
                    elif self.rank == 1:
                        p1 = p1.flatten()
                torch.testing.assert_close(p1, p2)
        ddp_model = DDP(Model().cuda(), device_ids=[self.rank])
        fsdp_model = FSDP(Model().cuda(), sharding_strategy=sharding_strategy, auto_wrap_policy=always_wrap_policy, use_orig_params=True)
        LR = 0.01
        ddp_optim = torch.optim.Adam(ddp_model.parameters(), lr=LR)
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters(), lr=LR)
        device = torch.device('cuda')
        inp = fsdp_model.get_input(device)
        ddp_out = ddp_model(*inp)
        fsdp_out = fsdp_model(*inp)
        check_parameter_parity(ddp_model, fsdp_model, True)
        ddp_loss = ddp_model.module.get_loss(inp, ddp_out)
        fsdp_loss = fsdp_model.get_loss(inp, fsdp_out)
        ddp_loss.backward()
        fsdp_loss.backward()
        ddp_optim.step()
        fsdp_optim.step()
        check_parameter_parity(ddp_model, fsdp_model, False)
        inp = fsdp_model.get_input(device)
        ddp_out = ddp_model(*inp)
        fsdp_out = fsdp_model(*inp)
        check_parameter_parity(ddp_model, fsdp_model, True)

class TestFSDPUseOrigParamsWriteback(FSDPTest):
    """Tests parameter and gradient writeback."""

    class Model(nn.Module):

        def __init__(self, device: torch.device):
            if False:
                return 10
            super().__init__()
            torch.manual_seed(42)
            self.lin1 = nn.Linear(5, 5, bias=True, device=device)
            self.lin2 = nn.Linear(5, 7, bias=True, device=device)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if False:
                return 10
            z = self.lin1(x)
            z = nn.functional.relu(z)
            z = self.lin2(z)
            return z

        def get_input(self, device: torch.device) -> Tuple[torch.Tensor, ...]:
            if False:
                for i in range(10):
                    print('nop')
            return (torch.randn((2, 5)).to(device),)

        def get_loss(self, inp, out):
            if False:
                for i in range(10):
                    print('nop')
            return out.sum()

    @property
    def world_size(self):
        if False:
            while True:
                i = 10
        return 2

    def _check_param_parity(self, ddp_model: DDP, fsdp_model: FSDP):
        if False:
            while True:
                i = 10
        with FSDP.summon_full_params(fsdp_model):
            for ((n1, p1), (n2, p2)) in zip(ddp_model.module.named_parameters(), fsdp_model.named_parameters()):
                self.assertEqual(n1, n2)
                torch.testing.assert_close(p1, p2)

    @skip_if_lt_x_gpu(2)
    def test_param_writeback(self):
        if False:
            i = 10
            return i + 15
        'Tests that changes to the original parameters are written back.'
        self.run_subtests({'change_first_weight': [True, False], 'change_data': [True, False]}, self._test_param_writeback)

    def _test_param_writeback(self, change_first_weight: bool, change_data: bool):
        if False:
            i = 10
            return i + 15

        def transform_param(param: nn.Parameter) -> nn.Parameter:
            if False:
                i = 10
                return i + 15
            return nn.Parameter(torch.ones_like(param) * 2)
        ddp_model = DDP(TestFSDPUseOrigParamsWriteback.Model(torch.device('cuda')), device_ids=[self.rank])
        fsdp_model = FSDP(TestFSDPUseOrigParamsWriteback.Model(torch.device('cuda')), use_orig_params=True)
        ddp = ddp_model.module
        fsdp = fsdp_model.module
        if change_first_weight:
            if change_data:
                ddp.lin1.weight.data = transform_param(ddp.lin1.weight)
                fsdp.lin1.weight.data = transform_param(fsdp.lin1.weight)
            else:
                ddp.lin1.weight = transform_param(ddp.lin1.weight)
                fsdp.lin1.weight = transform_param(fsdp.lin1.weight)
        elif change_data:
            ddp.lin2.weight.data = transform_param(ddp.lin2.weight)
            fsdp.lin2.weight.data = transform_param(fsdp.lin2.weight)
        else:
            ddp.lin2.weight = transform_param(ddp.lin2.weight)
            fsdp.lin2.weight = transform_param(fsdp.lin2.weight)
        self._check_param_parity(ddp_model, fsdp_model)

    @skip_if_lt_x_gpu(2)
    def test_grad_writeback(self):
        if False:
            i = 10
            return i + 15
        "\n        Tests that changes to the original parameters' gradients are written\n        back.\n        "
        self.run_subtests({'change_first_weight_grad': [False, True], 'change_data': [False, True], 'set_to_none': [False, True]}, self._test_grad_writeback)

    def _test_grad_writeback(self, change_first_weight_grad: bool, change_data: bool, set_to_none: bool):
        if False:
            print('Hello World!')
        if change_data and set_to_none:
            return

        def transform_grad(param: nn.Parameter) -> nn.Parameter:
            if False:
                i = 10
                return i + 15
            return None if set_to_none else torch.ones_like(param) * 2
        ddp_model = DDP(TestFSDPUseOrigParamsWriteback.Model(torch.device('cuda')), device_ids=[self.rank])
        fsdp_model = FSDP(TestFSDPUseOrigParamsWriteback.Model(torch.device('cuda')), use_orig_params=True)
        LR = 0.01
        ddp_optim = torch.optim.Adam(ddp_model.parameters(), lr=LR)
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters(), lr=LR)
        inp = fsdp_model.get_input(torch.device('cuda'))
        ddp_out = ddp_model(*inp)
        fsdp_out = fsdp_model(*inp)
        ddp_out.sum().backward()
        fsdp_out.sum().backward()
        ddp = ddp_model.module
        fsdp = fsdp_model.module
        if change_first_weight_grad:
            if change_data:
                ddp.lin1.weight.grad.data = transform_grad(ddp.lin1.weight)
                if fsdp.lin1.weight.grad is not None:
                    fsdp.lin1.weight.grad.data = transform_grad(fsdp.lin1.weight)
            else:
                ddp.lin1.weight.grad = transform_grad(ddp.lin1.weight)
                fsdp.lin1.weight.grad = transform_grad(fsdp.lin1.weight)
        elif change_data:
            ddp.lin2.weight.grad.data = transform_grad(ddp.lin2.weight)
            if fsdp.lin2.weight.grad is not None:
                fsdp.lin2.weight.grad.data = transform_grad(fsdp.lin2.weight)
        else:
            ddp.lin2.weight.grad = transform_grad(ddp.lin2.weight)
            fsdp.lin2.weight.grad = transform_grad(fsdp.lin2.weight)
        ddp_optim.step()
        fsdp_optim.step()
        self._check_param_parity(ddp_model, fsdp_model)
        inp = fsdp_model.get_input(torch.device('cuda'))
        ddp_out = ddp_model(*inp)
        fsdp_out = fsdp_model(*inp)
        ddp_out.sum().backward()
        fsdp_out.sum().backward()
        ddp_optim.step()
        fsdp_optim.step()
        self._check_param_parity(ddp_model, fsdp_model)

    @skip_if_lt_x_gpu(2)
    def test_writeback_shape_mismatch(self):
        if False:
            while True:
                i = 10
        fsdp_model = FSDP(TestFSDPUseOrigParamsWriteback.Model(torch.device('cuda')), use_orig_params=True)
        fsdp = fsdp_model.module
        assert self.rank in (0, 1), f'Expects world size of 2 but got {self.world_size}'
        with self.assertRaisesRegex(RuntimeError, 'Cannot writeback'):
            if self.rank == 0:
                lin1_weight_shape = list(fsdp.lin1.weight.shape)
                for dim_index in range(len(lin1_weight_shape)):
                    lin1_weight_shape[dim_index] += 1
                fsdp.lin1.weight = nn.Parameter(torch.randn(torch.Size(lin1_weight_shape), device=fsdp.lin1.weight.device))
                fsdp.lin1.weight.grad = torch.randn(torch.Size(lin1_weight_shape), device=fsdp.lin1.weight.device)
            elif self.rank == 1:
                lin2_weight_shape = list(fsdp.lin2.weight.shape)
                for dim_index in range(len(lin2_weight_shape)):
                    lin2_weight_shape[dim_index] += 1
                fsdp.lin2.weight = nn.Parameter(torch.randn(torch.Size(lin2_weight_shape), device=fsdp.lin2.weight.device))
                fsdp.lin2.weight.grad = torch.randn(torch.Size(lin2_weight_shape), device=fsdp.lin2.weight.device)
            with FSDP.summon_full_params(fsdp_model):
                ...

    @skip_if_lt_x_gpu(2)
    def test_writeback_between_fwd_and_bwd_for_no_reshard_raises(self):
        if False:
            print('Hello World!')
        fsdp_kwargs = {'sharding_strategy': ShardingStrategy.SHARD_GRAD_OP, 'auto_wrap_policy': ModuleWrapPolicy({nn.Linear}), 'use_orig_params': True}
        fsdp_wrapper = functools.partial(FSDP, **fsdp_kwargs)
        fsdp_model = fsdp_wrapper(TestFSDPUseOrigParamsWriteback.Model(torch.device('cuda')))
        inp = fsdp_model.get_input(torch.device('cuda'))
        loss = fsdp_model(*inp).sum()
        fsdp_model.lin1.weight.data = fsdp_model.lin1.weight.clone()
        assert_msg = 'FSDP does not support changing the parameters between forward and backward'
        with self.assertRaisesRegex(AssertionError, assert_msg):
            loss.backward()
        fsdp_model = fsdp_wrapper(TestFSDPUseOrigParamsWriteback.Model(torch.device('cuda')))
        inp = fsdp_model.get_input(torch.device('cuda'))
        loss = fsdp_model(*inp).sum()
        fsdp_model.lin1._fsdp_wrapped_module.weight = nn.Parameter(fsdp_model.lin1.weight.clone())
        with self.assertRaisesRegex(AssertionError, assert_msg):
            loss.backward()

    @skip_if_lt_x_gpu(2)
    def test_no_reshard_and_mixed_precision(self):
        if False:
            while True:
                i = 10
        '\n        Tests that writeback does not falsely get triggered for a few\n        configurations (exercising the sharded view skipping logic):\n        - Train forward -> full-precision unshard -> train forward\n        - Train forward -> eval forward\n        - Train forward/backward -> eval forward -> model checkpoint\n        '
        self.run_subtests({'use_full_prec_in_eval': [False, True]}, self._test_no_reshard_and_mixed_precision)

    def _test_no_reshard_and_mixed_precision(self, use_full_prec_in_eval: bool):
        if False:
            i = 10
            return i + 15
        if use_full_prec_in_eval:
            os.environ[_FSDP_USE_FULL_PREC_IN_EVAL] = '1'
        fsdp_kwargs = {'sharding_strategy': ShardingStrategy.SHARD_GRAD_OP, 'auto_wrap_policy': ModuleWrapPolicy({nn.Linear}), 'mixed_precision': MixedPrecision(param_dtype=torch.float16), 'use_orig_params': True}
        fsdp_model = FSDP(TestFSDPUseOrigParamsWriteback.Model(torch.device('cuda')), **fsdp_kwargs)
        inp = fsdp_model.get_input(torch.device('cuda'))
        fsdp_model(*inp)
        with FSDP.summon_full_params(fsdp_model):
            ...
        fsdp_model(*inp).sum()
        fsdp_model.train()
        fsdp_model(*inp)
        fsdp_model.eval()
        fsdp_model(*inp)
        fsdp_model.train()
        fsdp_model(*inp).sum().backward()
        fsdp_model.eval()
        fsdp_model(*inp)
        with FSDP.state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT):
            sd = fsdp_model.state_dict()
            fsdp_model.load_state_dict(sd)
        fsdp_model(*inp).sum().backward()

class TestFSDPUseOrigParamsFQNs(FSDPTest):

    @skip_if_lt_x_gpu(2)
    def test_named_parameters_in_forward(self):
        if False:
            return 10
        '\n        Tests that calling ``named_parameters()`` during forward returns FQNs\n        and ``Tensor`` s corresponding to the original parameters.\n        '
        param_shapes = [None, None]
        assert_equal_fn = self.assertEqual

        class Model(nn.Module):

            def __init__(self) -> None:
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.lin = nn.Linear(5, 5)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if False:
                    while True:
                        i = 10
                nonlocal param_shapes
                param_names = [clean_tensor_name(tup[0]) for tup in self.named_parameters()]
                params = [tup[1] for tup in self.named_parameters()]
                assert param_shapes[0] is not None and param_shapes[1] is not None, '`param_sizes` should be set'
                assert_equal_fn(param_names, ['lin.weight', 'lin.bias'])
                assert_equal_fn(params[0].shape, param_shapes[0])
                assert_equal_fn(params[1].shape, param_shapes[1])
                return self.lin(x)
        model = Model().cuda()
        param_shapes[0] = model.lin.weight.shape
        param_shapes[1] = model.lin.bias.shape
        fsdp_model = FSDP(model, use_orig_params=True)
        inp = torch.randn((2, 5), device=torch.device('cuda'))
        fsdp_model(inp)

class TestFSDPUseOrigParamsNoSync(FSDPTest):

    @property
    def world_size(self) -> int:
        if False:
            while True:
                i = 10
        return 2

    @skip_if_lt_x_gpu(2)
    def test_no_sync_correctness(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests a basic ``no_sync()`` setup by comparing ``use_orig_params=True``\n        against ``use_orig_params=False``.\n        '
        self.run_subtests({'sharding_strategy': [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP, ShardingStrategy.NO_SHARD]}, self._test_no_sync_correctness)

    def _test_no_sync_correctness(self, sharding_strategy: ShardingStrategy):
        if False:
            for i in range(10):
                print('nop')
        model = nn.Linear(7, 1, bias=False, device='cuda')
        fsdp_kwargs = {'sharding_strategy': sharding_strategy}
        model_use_flat_params = FSDP(copy.deepcopy(model), use_orig_params=False, **fsdp_kwargs)
        model_use_orig_params = FSDP(model, use_orig_params=True, **fsdp_kwargs)
        optim_use_flat_params = torch.optim.AdamW(model_use_flat_params.parameters(), foreach=True)
        optim_use_orig_params = torch.optim.AdamW(model_use_orig_params.parameters(), foreach=True)

        def _check_param_grad_parity(_baseline_model: nn.Module, _test_model: nn.Module):
            if False:
                for i in range(10):
                    print('nop')
            '\n            This assumes that the model is ``nn.Linear(7, 1, bias=False)``\n            (i.e. with a single 1D weight parameter) to be able to directly\n            compare the baseline and test models. On rank 1, the baseline\n            includes 1 element of padding.\n            '
            self.assertEqual(len(list(_baseline_model.parameters())), 1)
            self.assertEqual(len(list(_test_model.parameters())), 1)
            for (flat_param, orig_param) in zip(_baseline_model.parameters(), _test_model.parameters()):
                self.assertGreaterEqual(flat_param.numel(), orig_param.numel())
                unpadded_param_numel = orig_param.numel()
                torch.testing.assert_close(flat_param[:unpadded_param_numel], orig_param.flatten())
                unpadded_grad_numel = orig_param.grad.numel()
                torch.testing.assert_close(flat_param.grad[:unpadded_grad_numel].reshape(orig_param.grad.shape), orig_param.grad)
        inp = torch.randn((2, 7), device='cuda')
        grad = torch.randn((2, 1), device='cuda')
        out_use_flat_params = model_use_flat_params(inp)
        out_use_orig_params = model_use_orig_params(inp)
        torch.testing.assert_close(out_use_flat_params, out_use_orig_params)
        out_use_flat_params.backward(grad)
        out_use_orig_params.backward(grad)
        _check_param_grad_parity(model_use_flat_params, model_use_orig_params)
        ref_grads_use_flat_params = [param.grad.detach().clone() for param in model_use_flat_params.parameters()]
        ref_grads_use_orig_params = [param.grad.detach().clone() for param in model_use_orig_params.parameters() if param.grad is not None]
        optim_use_flat_params.zero_grad(set_to_none=True)
        optim_use_orig_params.zero_grad(set_to_none=True)
        for model in (model_use_flat_params, model_use_orig_params):
            with model.no_sync():
                out = model(inp)
                out.backward(grad)
        _check_param_grad_parity(model_use_flat_params, model_use_orig_params)
        for model in (model_use_flat_params, model_use_orig_params):
            out = model(inp)
            out.backward(grad)
        _check_param_grad_parity(model_use_flat_params, model_use_orig_params)
        grads_use_flat_params = [param.grad.detach().clone() for param in model_use_flat_params.parameters()]
        grads_use_orig_params = [param.grad.detach().clone() for param in model_use_orig_params.parameters() if param.grad is not None]
        for (grad, ref_grad) in zip(grads_use_flat_params, ref_grads_use_flat_params):
            torch.testing.assert_close(grad, 2 * ref_grad)
        for (grad, ref_grad) in zip(grads_use_orig_params, ref_grads_use_orig_params):
            torch.testing.assert_close(grad, 2 * ref_grad)

    @skip_if_lt_x_gpu(2)
    def test_no_sync_mixed_precision(self):
        if False:
            return 10
        '\n        Tests that dtypes are as expected when using ``no_sync()`` with\n        ``use_orig_params=True`` and parameter mixed precision.\n        '
        self.run_subtests({'sharding_strategy': [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP, ShardingStrategy.NO_SHARD]}, self._test_no_sync_mixed_precision)

    def _test_no_sync_mixed_precision(self, sharding_strategy: ShardingStrategy):
        if False:
            i = 10
            return i + 15
        model = nn.Linear(3, 3, device='cuda')
        mixed_precision = MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float32)
        fsdp_kwargs = {'sharding_strategy': sharding_strategy, 'mixed_precision': mixed_precision, 'use_orig_params': True}
        fsdp_model = FSDP(model, **fsdp_kwargs)
        inp = torch.randn((2, 3), device='cuda')
        with fsdp_model.no_sync():
            fsdp_model(inp).sum().backward()
            for param in fsdp_model.parameters():
                if param.grad is not None:
                    self.assertEqual(param.grad.dtype, torch.float16)
            fsdp_model(inp).sum().backward()
            for param in fsdp_model.parameters():
                if param.grad is not None:
                    self.assertEqual(param.grad.dtype, torch.float16)
        fsdp_model(inp).sum().backward()
        for param in fsdp_model.parameters():
            if param.grad is not None:
                self.assertEqual(param.grad.dtype, torch.float32)

class TestFSDPUseOrigParamsInit(FSDPTest):

    @skip_if_lt_x_gpu(2)
    def test_non_uniform_requires_grad(self):
        if False:
            for i in range(10):
                print('nop')
        model = nn.Sequential(nn.Linear(3, 3, device='cuda'), nn.Linear(3, 3, device='cuda'))
        model[0].bias.requires_grad = False
        model[1].bias.requires_grad = False
        fsdp_model = FSDP(model, use_orig_params=True)
        self.assertTrue(fsdp_model[0].weight.requires_grad)
        self.assertFalse(fsdp_model[0].bias.requires_grad)
        self.assertTrue(fsdp_model[1].weight.requires_grad)
        self.assertFalse(fsdp_model[1].bias.requires_grad)
NUM_SIZE0_TENSORS = 1000

class TestMultiTensorApply(TestCase):

    def test_multi_tensor_apply_size0_tensors_cpu(self):
        if False:
            print('Hello World!')
        size0_tensors = [torch.empty(0, device='cpu') for _ in range(NUM_SIZE0_TENSORS)]
        torch._foreach_mul_(size0_tensors, 0.1)

    @unittest.skipIf(not TEST_CUDA, 'no cuda')
    def test_multi_tensor_apply_size0_tensors_cuda(self):
        if False:
            for i in range(10):
                print('nop')
        size0_tensors = [torch.empty(0, device='cuda') for _ in range(NUM_SIZE0_TENSORS)]
        torch._foreach_mul_(size0_tensors, 0.1)
instantiate_parametrized_tests(TestFSDPUseOrigParamsMultipleParamGroups)
instantiate_parametrized_tests(TestFSDPUseOrigParamsUnshardReshard)
instantiate_parametrized_tests(TestFSDPUseOrigParamsParamAccess)
instantiate_parametrized_tests(TestFSDPUseOrigParamsFQNs)
instantiate_parametrized_tests(TestFSDPUseOrigParamsNoSync)
if __name__ == '__main__':
    run_tests()