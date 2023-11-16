import itertools
import os
import re
import sys
from abc import ABC, abstractmethod
from contextlib import nullcontext
from copy import deepcopy
from enum import auto, Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from unittest import mock
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import TrainingState
from torch.distributed.fsdp._init_utils import NO_RESHARD_AFTER_FORWARD_STRATEGIES
from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import always_wrap_policy, ModuleWrapPolicy, wrap
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import MultiProcessTestCase, MultiThreadedTestCase, TEST_SKIPS
from torch.testing._internal.common_utils import FILE_SCHEMA, get_cycles_per_ms

class FSDPInitMode(Enum):
    NO_FSDP = auto()
    RECURSIVE = auto()

class CUDAInitMode(Enum):
    CUDA_BEFORE = auto()
    CUDA_AFTER = auto()
    CUDA_NEVER = auto()

class FSDPTestModel(nn.Module, ABC):
    """This defines the interface expected from all models used commonly for
    FSDP unit tests."""

    @abstractmethod
    def get_input(self, device) -> Tuple[torch.Tensor, ...]:
        if False:
            for i in range(10):
                print('nop')
        'Returns an input for the model as as tuple.'
        ...

    @abstractmethod
    def get_loss(self, input, output) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        'Returns the loss given the input and output.'
        ...

    @abstractmethod
    def run_backward(self, loss) -> None:
        if False:
            while True:
                i = 10
        'Runs the backward pass (e.g. including ``loss.backward()``).'
        ...

    @staticmethod
    @abstractmethod
    def init(group: dist.ProcessGroup, fsdp_init_mode: FSDPInitMode, *init_args: Any, cuda_init_mode: CUDAInitMode, fsdp_kwargs: Optional[Dict[str, Any]]=None, deterministic: bool=False, **init_kwargs: Any) -> nn.Module:
        if False:
            while True:
                i = 10
        'Initializes an instance of this model.'
        ...

def _assert_module_states(model: nn.Module, process_group: dist.ProcessGroup, assert_fn: Callable):
    if False:
        return 10
    '\n    All-gathers module states across ranks and calls ``assert_fn`` on each pair\n    of corresponding states from rank 0 and a nonzero rank. For example, if\n    ``assert_fn`` is ``self.assertEqual()``, then this checks that all module\n    states are equal across ranks.\n    '
    named_module_states = [(param_name, param.detach().cpu()) for (param_name, param) in model.named_parameters()]
    named_module_states += [(buffer_name, buffer.detach().cpu()) for (buffer_name, buffer) in model.named_buffers()]
    world_size = dist.get_world_size(process_group)
    olist = [None for _ in range(world_size)]
    dist.all_gather_object(olist, named_module_states, group=process_group)
    rank0_states = olist[0]
    for state in olist[1:]:
        for ((_, p1), (_, p2)) in zip(rank0_states, state):
            assert_fn(p1, p2)

def _zero_model(model: nn.Module, zero_buffers: bool=False, summon_full=True):
    if False:
        while True:
            i = 10
    'Zeros the parameters and optionally buffers of ``model`` in place.'
    ctx = FSDP.summon_full_params(model) if summon_full else nullcontext()
    with ctx:
        for param in model.parameters():
            with torch.no_grad():
                param.zero_()
        if zero_buffers:
            for buffer in model.buffers():
                with torch.no_grad():
                    buffer.zero_()

def _get_state_dict(model, cpu_offload=False, half=False):
    if False:
        return 10
    if not cpu_offload:
        model = model.cuda()
    if half:
        model.half()
    return model.state_dict()

def subtest_name(test_name_mapping, *args):
    if False:
        while True:
            i = 10
    return '_'.join([test_name_mapping[str(s)] if s is not None else 'none' for s in args])

def _broadcast_state_dict(rank, state_dict):
    if False:
        i = 10
        return i + 15
    for (param_name, param) in state_dict.items():
        if param.device != torch.device('cpu'):
            state_dict[param_name] = param.cpu()
    olist = [state_dict if rank == 0 else None]
    dist.broadcast_object_list(olist)
    state_dict = olist[0]
    for param_name in state_dict.keys():
        state_dict[param_name] = state_dict[param_name].cuda()
    return state_dict

def get_full_params(model: nn.Module, recurse: bool=True):
    if False:
        print('Hello World!')
    '\n    Returns the full unsharded parameters of ``model``. Any FSDP-managed\n    parameters offloaded to CPU are moved to GPU in the returned list.\n\n    Args:\n        recurse (bool): If ``False``, only unshards the parameters immediate to\n            ``model``; if ``True``, recurses through the module hierarchy\n            rooted at ``model``.\n    '
    with FSDP.summon_full_params(model, recurse=recurse):
        return deepcopy(list(model.parameters()))

def _maybe_cuda(model: nn.Module, move_to_cuda: bool):
    if False:
        print('Hello World!')
    return model.cuda() if move_to_cuda else model

def _maybe_wrap_fsdp(model: nn.Module, wrap_fsdp: bool, *args, **kwargs):
    if False:
        return 10
    return model if not wrap_fsdp else FSDP(model, *args, **kwargs)

class DummyProcessGroup:

    def __init__(self, rank: int, size: int):
        if False:
            for i in range(10):
                print('nop')
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        if False:
            i = 10
            return i + 15
        return self._rank

    def size(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self._size

    def allreduce(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        dist_wait = mock.Mock()

        def get_future():
            if False:
                while True:
                    i = 10
            future = torch.futures.Future()
            future.set_result(1)
            return future
        dist_wait.get_future = get_future
        return dist_wait

class TransformerWithSharedParams(FSDPTestModel):

    def __init__(self, group: dist.ProcessGroup, cuda_init_mode: CUDAInitMode, add_bn: bool, deterministic: bool):
        if False:
            print('Hello World!')
        super().__init__()
        self.rank = group.rank()
        self.world_size = group.size()
        if deterministic:
            torch.manual_seed(0)
        d_vocab = 23
        d_model = 16
        self.embed_tokens = nn.Embedding(d_vocab, d_model)
        self.transformer = nn.Transformer(d_model=d_model, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=8, dropout=0.1)
        self.output_proj = nn.Linear(d_model, d_vocab)
        self.output_proj.weight = self.embed_tokens.weight
        self.register_buffer('vocab_bias', self.embed_tokens.weight.new_ones((d_model,)))
        self.register_buffer('long_buffer', torch.zeros_like(self.vocab_bias, dtype=torch.long))
        self.bs = 2
        self.bn = torch.nn.BatchNorm1d(self.bs) if add_bn else torch.nn.Identity()
        if cuda_init_mode == CUDAInitMode.CUDA_BEFORE:
            self = self.cuda()
        if deterministic:
            self.eval()

    def get_input(self, device):
        if False:
            return 10
        torch.manual_seed(1 + self.rank)
        src = torch.arange(12, device=device).view(6, self.bs)
        tgt = torch.arange(self.bs * 4, device=device).view(4, self.bs)
        return (src, tgt)

    def forward(self, src_ids, tgt_ids):
        if False:
            return 10
        src = self.embed_tokens(src_ids)
        src = src + self.vocab_bias + self.long_buffer.type_as(src)
        tgt = self.embed_tokens(tgt_ids)
        tgt = self.bn(tgt)
        x = self.transformer(src, tgt)
        return self.output_proj(x)

    def get_loss(self, input, output):
        if False:
            i = 10
            return i + 15
        (_, tgt) = input
        return nn.functional.cross_entropy(output.view(-1, output.size(-1)), tgt.view(-1), reduction='sum')

    def run_backward(self, loss):
        if False:
            while True:
                i = 10
        loss.backward()

    @staticmethod
    def init(group: dist.ProcessGroup, fsdp_init_mode: FSDPInitMode, cuda_init_mode: CUDAInitMode, fsdp_kwargs: Optional[Dict[str, Any]]=None, deterministic: bool=False, add_bn: bool=True) -> Union[nn.Module, FSDP]:
        if False:
            while True:
                i = 10
        '\n        Initializes a :class:`TransformerWithSharedParams` instance.\n\n        Args:\n            fsdp_init_mode (FSDPInitMode): If ``NO_FSDP``, then does not wrap\n                any modules with FSDP. If ``RECURSIVE``, then wraps with\n                top-level FSDP. By default, the top-level FSDP uses the\n                ``ModuleWrapPolicy`` for encoder and decoder layers, but a\n                different auto wrap policy may be specified via\n                ``fsdp_kwargs``.\n            cuda_init_mode (CUDAInitMode): Determines model movement to CUDA.\n            fsdp_kwargs (Optional[Dict[str, Any]]): Optional keyword arguments\n                forwarded to the FSDP constructor.\n            deterministic (bool): Whether to make the model deterministic\n                across constructions.\n            add_bn (bool): Whether to include batch norm in the model.\n        '
        if fsdp_kwargs is None:
            fsdp_kwargs = {}
        if fsdp_init_mode == FSDPInitMode.NO_FSDP:
            if isinstance(group, tuple):
                pg = group[0]
            else:
                pg = group
            return TransformerWithSharedParams(pg, cuda_init_mode, add_bn, deterministic)
        elif fsdp_init_mode == FSDPInitMode.RECURSIVE:
            if 'auto_wrap_policy' not in fsdp_kwargs:
                auto_wrap_policy = ModuleWrapPolicy({TransformerEncoderLayer, TransformerDecoderLayer})
            else:
                auto_wrap_policy = fsdp_kwargs.pop('auto_wrap_policy')
            if 'sharding_strategy' in fsdp_kwargs and fsdp_kwargs['sharding_strategy'] in {ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2} and (not isinstance(group, tuple)):
                fsdp_pg = None
            else:
                fsdp_pg = group
            if isinstance(group, tuple):
                tformer_pg = group[0]
            else:
                tformer_pg = group
            m = TransformerWithSharedParams(tformer_pg, cuda_init_mode, add_bn, deterministic)
            fsdp_model = FSDP(m, fsdp_pg, auto_wrap_policy=auto_wrap_policy, **fsdp_kwargs)
            if cuda_init_mode == CUDAInitMode.CUDA_AFTER:
                fsdp_model = fsdp_model.cuda()
            return fsdp_model
        raise ValueError(f'Unsupported FSDP init mode: {fsdp_init_mode}')

    def get_ignored_modules(self):
        if False:
            print('Hello World!')
        return [self.transformer]

class NestedWrappedModule(FSDPTestModel):

    def __init__(self, group: dist.ProcessGroup, wrap_fsdp: bool, cuda_init_mode: CUDAInitMode, deterministic: bool, **fsdp_kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.rank = group.rank()
        self.world_size = group.size()
        move_to_cuda = cuda_init_mode == CUDAInitMode.CUDA_BEFORE

        def _maybe_wrap(layer):
            if False:
                while True:
                    i = 10
            if wrap_fsdp:
                return FSDP(layer, group, **fsdp_kwargs)
            return layer
        if deterministic:
            torch.manual_seed(0)
        self.module = nn.Sequential(_maybe_cuda(nn.Linear(8, 4), move_to_cuda), _maybe_wrap(nn.Sequential(_maybe_wrap(_maybe_cuda(nn.Linear(4, 16), move_to_cuda)), _maybe_cuda(nn.Linear(16, 16), move_to_cuda))), _maybe_wrap(_maybe_cuda(nn.Linear(16, 4), move_to_cuda)), _maybe_cuda(nn.Linear(4, 8), move_to_cuda))

    def get_input(self, device):
        if False:
            return 10
        torch.manual_seed(1 + self.rank)
        return (torch.rand(4, 8, device=device),)

    def forward(self, x):
        if False:
            print('Hello World!')
        return self.module(x)

    def get_loss(self, input, output):
        if False:
            while True:
                i = 10
        loss = output.sum()
        return loss

    def run_backward(self, loss):
        if False:
            i = 10
            return i + 15
        loss.backward()

    @staticmethod
    def init(group: dist.ProcessGroup, fsdp_init_mode: FSDPInitMode, cuda_init_mode: CUDAInitMode, fsdp_kwargs: Optional[Dict[str, Any]]=None, deterministic: bool=False) -> nn.Module:
        if False:
            i = 10
            return i + 15
        '\n        Initializes a :class:`NestedWrappedModule` instance.\n\n        Args:\n            fsdp_init_mode (FSDPInitMode): If ``NO_FSDP``, then does not wrap\n                any modules with FSDP. If ``RECURSIVE``, then wraps some nested\n                modules with FSDP but not the top-level module. The model may\n                later be wrapped with a top-level FSDP external to this method\n                if desired.\n            cuda_init_mode (CUDAInitMode): Determines model movement to CUDA.\n            fsdp_kwargs (Optional[Dict[str, Any]]): Optional keyword arguments\n                forwarded to the FSDP constructor.\n            deterministic (bool): Whether to make the model deterministic\n                across constructions.\n        '
        if fsdp_kwargs is None:
            fsdp_kwargs = {}
        if fsdp_init_mode == FSDPInitMode.NO_FSDP:
            return NestedWrappedModule(group, wrap_fsdp=False, cuda_init_mode=cuda_init_mode, deterministic=deterministic)
        elif fsdp_init_mode == FSDPInitMode.RECURSIVE:
            fsdp_model = NestedWrappedModule(group, wrap_fsdp=True, cuda_init_mode=cuda_init_mode, deterministic=deterministic, **fsdp_kwargs)
            if cuda_init_mode == CUDAInitMode.CUDA_AFTER:
                fsdp_model = fsdp_model.cuda()
            return fsdp_model
        raise ValueError(f'Unsupported FSDP init mode: {fsdp_init_mode}')

class AlwaysWrapNestedWrappedModule(NestedWrappedModule):

    @staticmethod
    def init(group: dist.ProcessGroup, fsdp_init_mode: FSDPInitMode, cuda_init_mode: CUDAInitMode, fsdp_kwargs: Optional[Dict[str, Any]]=None, deterministic: bool=False):
        if False:
            return 10
        '\n        Initializes a :class:`NestedWrappedModule` instance, but unlike\n        :meth:`NestedWrappedModule.init`, for the ``RECURSIVE`` init mode, this\n        wraps with top-level FSDP and the ``always_wrap_policy()`` auto wrap\n        policy.\n        '
        super_ = super(AlwaysWrapNestedWrappedModule, AlwaysWrapNestedWrappedModule)
        model = super_.init(group=group, fsdp_init_mode=FSDPInitMode.NO_FSDP, cuda_init_mode=cuda_init_mode, fsdp_kwargs=fsdp_kwargs, deterministic=deterministic)
        if fsdp_init_mode == FSDPInitMode.NO_FSDP:
            return model
        elif fsdp_init_mode == FSDPInitMode.RECURSIVE:
            fsdp_model = FSDP(model, auto_wrap_policy=always_wrap_policy, **fsdp_kwargs)
            if cuda_init_mode == CUDAInitMode.CUDA_AFTER:
                fsdp_model = fsdp_model.cuda()
            return fsdp_model

class NonUniformReqGradNWM(NestedWrappedModule):

    def __init__(self, group: dist.ProcessGroup, wrap_fsdp: bool, cuda_init_mode: CUDAInitMode, deterministic: bool, **fsdp_kwargs):
        if False:
            return 10
        super(NestedWrappedModule, self).__init__()
        self.rank = group.rank()
        self.world_size = group.size()
        move_to_cuda = cuda_init_mode == CUDAInitMode.CUDA_BEFORE

        def _maybe_wrap(layer):
            if False:
                for i in range(10):
                    print('nop')
            if wrap_fsdp:
                return FSDP(layer, group, **fsdp_kwargs)
            return layer
        if deterministic:
            torch.manual_seed(0)
        self.module = nn.Sequential(_maybe_cuda(nn.Linear(8, 4), move_to_cuda), _maybe_wrap(nn.Sequential(_maybe_wrap(_maybe_cuda(nn.Linear(4, 16), move_to_cuda)), _maybe_cuda(nn.Linear(16, 16), move_to_cuda))), _maybe_wrap(nn.Sequential(_maybe_cuda(nn.Linear(16, 4), move_to_cuda), _maybe_cuda(nn.Linear(4, 8), move_to_cuda))))

    @staticmethod
    def _set_nonuniform_req_grad(model, req_grad_mask) -> None:
        if False:
            for i in range(10):
                print('nop')
        for (n, p) in model.named_parameters():
            if not re.match(req_grad_mask, n):
                p.requires_grad_(False)

    @staticmethod
    def init(group: dist.ProcessGroup, fsdp_init_mode: FSDPInitMode, cuda_init_mode: CUDAInitMode, fsdp_kwargs: Optional[Dict[str, Any]]=None, deterministic: bool=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initializes a :class:`NestedWrappedModule` instance, but unlike\n        :meth:`NestedWrappedModule.init`, it wraps a second :class:`torch.nn.Sequential`\n        container to enable the desired non-uniform ``requires_grad``\n        ``use_orig_params=True`` tests. For both ``RECURSIVE`` and ``NO_FSDP``\n        init modes, freezes all parameters except the last two to validate\n        ``ShardedGradScaler`` support for ranks with no (non-zero sized) local shards in\n        FSDP ``use_orig_params=True`` mode.\n        '
        req_grad_pattern = re.compile('module\\.2.*\\.1.*')
        if fsdp_init_mode == FSDPInitMode.NO_FSDP:
            ddp_model = NonUniformReqGradNWM(group, wrap_fsdp=False, cuda_init_mode=cuda_init_mode, deterministic=deterministic)
            NonUniformReqGradNWM._set_nonuniform_req_grad(ddp_model, req_grad_pattern)
            return ddp_model
        elif fsdp_init_mode == FSDPInitMode.RECURSIVE:
            if fsdp_kwargs is None:
                fsdp_kwargs = {}
            fsdp_model = NonUniformReqGradNWM(group, wrap_fsdp=True, cuda_init_mode=cuda_init_mode, deterministic=deterministic, **fsdp_kwargs)
            if cuda_init_mode == CUDAInitMode.CUDA_AFTER:
                fsdp_model = fsdp_model.cuda()
            NonUniformReqGradNWM._set_nonuniform_req_grad(fsdp_model, req_grad_pattern)
            return fsdp_model
        raise ValueError(f'Unsupported FSDP init mode: {fsdp_init_mode}')

class ModuleWithDelay(FSDPTestModel):
    """This class wraps a :class:`FSDPTestModel` to optionally add a delay
    after computing the loss and/or before the gradient reduction."""

    def __init__(self, module: nn.Module, delay_after_loss_ms: int, delay_before_reduction_ms: int):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.delay_after_loss_ms = delay_after_loss_ms
        self.delay_before_reduction_ms = delay_before_reduction_ms
        self.module = module

    def get_input(self, device):
        if False:
            i = 10
            return i + 15
        return self.module.get_input(device)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        return self.module(x)

    def get_loss(self, input, output):
        if False:
            while True:
                i = 10
        loss = self.module.get_loss(input, output)
        if self.delay_after_loss_ms > 0:
            torch.cuda._sleep(int(self.delay_after_loss_ms * get_cycles_per_ms()))
        return loss

    def run_backward(self, loss):
        if False:
            return 10
        orig_reduce_scatter = torch.distributed.reduce_scatter_tensor

        def _delayed_reduce_scatter(*args, **kwargs):
            if False:
                print('Hello World!')
            if self.delay_before_reduction_ms > 0:
                torch.cuda._sleep(int(self.delay_before_reduction_ms * get_cycles_per_ms()))
            return orig_reduce_scatter(*args, **kwargs)
        with mock.patch('torch.distributed.reduce_scatter_tensor', _delayed_reduce_scatter):
            self.module.run_backward(loss)

    @staticmethod
    def init(module_class: Type[FSDPTestModel], *model_args: Any, delay_after_loss_ms: int, delay_before_reduction_ms: int, **model_kwargs: Any):
        if False:
            while True:
                i = 10
        '\n        Args:\n            module_class (Type[FSDPTestModel]): Wrapped module class to which\n                to add delays.\n            model_args: Positional arguments forwarded to the ``module_class``\n                ``init()``.\n            delay_after_loss_ms (int): Delay after computing the loss/before\n                the optimizer step (in ms).\n            delay_before_reduction_ms (int): Delay before reduce-scattering\n                gradients (in ms).\n            model_kwargs: Keyword arguments forwarded to the ``module_class``\n                ``init()``.\n        '
        return ModuleWithDelay(module_class.init(*model_args, **model_kwargs), delay_after_loss_ms, delay_before_reduction_ms)

class NestedWrappedModuleWithDelay(ModuleWithDelay):

    @staticmethod
    def init(group: dist.ProcessGroup, fsdp_init_mode: FSDPInitMode, cuda_init_mode: CUDAInitMode=CUDAInitMode.CUDA_AFTER, fsdp_kwargs: Optional[Dict[str, Any]]=None, deterministic: bool=False, delay_after_loss_ms: int=0, delay_before_reduction_ms: int=0):
        if False:
            while True:
                i = 10
        return super(NestedWrappedModuleWithDelay, NestedWrappedModuleWithDelay).init(NestedWrappedModule, group=group, fsdp_init_mode=fsdp_init_mode, cuda_init_mode=cuda_init_mode, fsdp_kwargs=fsdp_kwargs, deterministic=deterministic, delay_after_loss_ms=delay_after_loss_ms, delay_before_reduction_ms=delay_before_reduction_ms)

class DummyDDP(nn.Module):

    def __init__(self, module):
        if False:
            while True:
                i = 10
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return self.module(*args, **kwargs)

class MixtureOfExperts(NestedWrappedModule):

    def __init__(self, group: dist.ProcessGroup, wrap_fsdp: bool, cuda_init_mode: CUDAInitMode, delay_before_free_ms: int, deterministic: bool, **fsdp_kwargs):
        if False:
            print('Hello World!')
        super().__init__(group=group, wrap_fsdp=wrap_fsdp, cuda_init_mode=cuda_init_mode, deterministic=deterministic)
        self.group = group
        self.delay_before_free_ms = delay_before_free_ms
        self.wrap_fsdp = wrap_fsdp
        self.move_to_cuda = cuda_init_mode == CUDAInitMode.CUDA_BEFORE
        if deterministic:
            torch.manual_seed(42 + self.rank)
        d_expert = 23
        d_shared = 12
        d_input = 8
        expert = _maybe_cuda(nn.Linear(d_expert, d_shared), self.move_to_cuda)
        self.num_expert_params = sum([p.numel() for p in expert.parameters()])
        for p in expert.parameters():
            p.expert = True
        if deterministic:
            torch.manual_seed(0)
        shared = _maybe_cuda(nn.Linear(d_shared, d_expert), self.move_to_cuda)
        if wrap_fsdp:
            expert_group = torch.distributed.new_group([group.rank()])
            expert = FSDP(expert, expert_group, **fsdp_kwargs)
            shared = FSDP(shared, group, **fsdp_kwargs)
        self.module = nn.Sequential(_maybe_cuda(nn.Linear(d_input, d_shared), self.move_to_cuda), shared, expert, _maybe_cuda(nn.Linear(d_shared, d_input), self.move_to_cuda))

    def forward(self, x):
        if False:
            print('Hello World!')
        if self.delay_before_free_ms > 0:
            expert = self.module[2]
            if isinstance(expert, FSDP):
                orig_reshard = torch.distributed.fsdp._runtime_utils._reshard

                def _delayed_reshard(*args, **kwargs):
                    if False:
                        while True:
                            i = 10
                    torch.cuda._sleep(int(self.delay_before_free_ms * get_cycles_per_ms()))
                    return orig_reshard(*args, **kwargs)
                with mock.patch('torch.distributed.fsdp._runtime_utils._reshard', _delayed_reshard):
                    return self.module(x)
        return self.module(x)

    def run_backward(self, loss):
        if False:
            i = 10
            return i + 15
        loss.backward()
        if not self.wrap_fsdp:
            with torch.no_grad():
                for p in self.parameters():
                    if hasattr(p, 'expert'):
                        continue
                    p.grad.div_(self.world_size)
                    torch.distributed.all_reduce(p.grad, group=self.group)

    @staticmethod
    def init(group: dist.ProcessGroup, fsdp_init_mode: FSDPInitMode, cuda_init_mode: CUDAInitMode, fsdp_kwargs: Optional[Dict[str, Any]]=None, deterministic: bool=False, delay_before_free_ms: int=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initializes a :class:`MixtureOfExperts` instance.\n\n        Args:\n            fsdp_init_mode (FSDPInitMode): If ``NO_FSDP``, then does not wrap\n                any modules with FSDP. If ``RECURSIVE``, then wraps some nested\n                modules with FSDP, including the expert and shared layers, but\n                not the top-level module. The model may later be wrapped with a\n                top-level FSDP external to this method if desired.\n            cuda_init_mode (CUDAInitMode): Determines model movement to CUDA.\n            fsdp_kwargs (Optional[Dict[str, Any]]): Optional keyword arguments\n                forwarded to the FSDP constructor.\n            deterministic (bool): Whether to make the model deterministic\n                across constructions.\n            delay_before_free_ms (int): Delay before resharding expert\n                parameters in the forward pass (in ms).\n        '
        if fsdp_kwargs is None:
            fsdp_kwargs = {}
        if fsdp_init_mode == FSDPInitMode.NO_FSDP:
            return MixtureOfExperts(group, wrap_fsdp=False, cuda_init_mode=cuda_init_mode, delay_before_free_ms=delay_before_free_ms, deterministic=deterministic)
        elif fsdp_init_mode == FSDPInitMode.RECURSIVE:
            fsdp_model = MixtureOfExperts(group, wrap_fsdp=True, cuda_init_mode=cuda_init_mode, delay_before_free_ms=delay_before_free_ms, deterministic=deterministic, **fsdp_kwargs)
            if cuda_init_mode == CUDAInitMode.CUDA_AFTER:
                fsdp_model = fsdp_model.cuda()
            return fsdp_model
        raise ValueError(f'Unsupported FSDP init mode: {fsdp_init_mode}')

def run_subtests(cls_inst, subtest_config: Dict[str, List[Any]], test_fn: Callable, *test_args, **test_kwargs: Any):
    if False:
        print('Hello World!')
    '\n    Runs a test function given by ``test_fn`` as a subtest according to the\n    configurations specified by ``subtest_config``. This amortizes the\n    costly setup overhead (including process spawn and initializing the\n    process group) over the subtests.\n\n    Args:\n        subtest_config (Dict[str, List[Any]]): A mapping from subtest\n            keyword argument name to a list of its possible values.\n        test_fn (Callable): A callable that runs the actual test.\n        test_args: Positional arguments to pass to ``test_fn``.\n        test_kwargs: Keyword arguments to pass to ``test_fn``.\n    '
    subtest_config_items: List[Tuple[str, List[Any]]] = list(subtest_config.items())
    subtest_config_keys: List[str] = [item[0] for item in subtest_config_items]
    subtest_config_values: List[List[Any]] = [item[1] for item in subtest_config_items]
    for values in itertools.product(*subtest_config_values):
        subtest_kwargs = dict(zip(subtest_config_keys, values))
        with cls_inst.subTest(**subtest_kwargs):
            test_fn(*test_args, **test_kwargs, **subtest_kwargs)
        dist.barrier()

class FSDPTestMultiThread(MultiThreadedTestCase):

    @property
    def world_size(self):
        if False:
            i = 10
            return i + 15
        return torch.cuda.device_count() if torch.cuda.is_available() else 4

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self._spawn_threads()

    def run_subtests(self, *args, **kwargs):
        if False:
            return 10
        return run_subtests(self, *args, **kwargs)

class FSDPTest(MultiProcessTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        os.environ['NCCL_DESYNC_DEBUG'] = '0'
        self._spawn_processes()

    @property
    def world_size(self):
        if False:
            while True:
                i = 10
        return min(torch.cuda.device_count(), 8) if torch.cuda.is_available() else 4

    @property
    def process_group(self):
        if False:
            return 10
        return dist.distributed_c10d._get_default_group()

    @property
    def init_method(self):
        if False:
            i = 10
            return i + 15
        return f'{FILE_SCHEMA}{self.file_name}'

    def _check_cpu_offload(self, fsdp_model, cpu_offload):
        if False:
            return 10
        self.assertEqual(cpu_offload, fsdp_model.cpu_offload)

    def _check_backward_prefetch(self, fsdp_model, backward_prefetch):
        if False:
            i = 10
            return i + 15
        self.assertEqual(backward_prefetch, fsdp_model.backward_prefetch)

    def _check_forward_prefetch(self, fsdp_model, forward_prefetch):
        if False:
            print('Hello World!')
        self.assertEqual(forward_prefetch, fsdp_model.forward_prefetch)

    def run_subtests(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return run_subtests(self, *args, **kwargs)

    @classmethod
    def _run(cls, rank, test_name, file_name, pipe):
        if False:
            for i in range(10):
                print('nop')
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name
        print(f'dist init r={self.rank}, world={self.world_size}')
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        try:
            dist.init_process_group(init_method=self.init_method, backend=backend, world_size=int(self.world_size), rank=self.rank)
        except RuntimeError as e:
            if 'recompile' in e.args[0]:
                sys.exit(TEST_SKIPS['backend_unavailable'].exit_code)
            raise
        if torch.cuda.is_available() and torch.cuda.device_count():
            torch.cuda.set_device(self.rank % torch.cuda.device_count())
        dist.barrier()
        self.run_test(test_name, pipe)
        dist.barrier()
        dist.destroy_process_group()

    def _train_for_several_steps(self, model: nn.Module, num_steps: int, autocast: bool, lr: float=0.01, fsdp_cpu_offload: Optional[CPUOffload]=None, save_model: bool=False, mixed_precision: Optional[MixedPrecision]=None, enable_sharded_grad_scaler: bool=False, use_pure_fp16: bool=False, sharded_grad_scaler_kwargs: Optional[Dict[str, Any]]=None):
        if False:
            for i in range(10):
                print('nop')
        cpu_offload_params = fsdp_cpu_offload and fsdp_cpu_offload.offload_params
        model_device = next(model.parameters()).device
        if sharded_grad_scaler_kwargs is None:
            sharded_grad_scaler_kwargs = {}
        sharded_grad_scaler = ShardedGradScaler(enabled=enable_sharded_grad_scaler, **sharded_grad_scaler_kwargs)
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        for _ in range(num_steps):
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=autocast):
                input = model.module.get_input(torch.device('cuda'))
                if use_pure_fp16 or (mixed_precision and (not isinstance(model, FSDP))):
                    if isinstance(input, torch.Tensor):
                        input = input.half()
                    else:
                        input = tuple((x.half() for x in input))
                output = model(*input)
                if cpu_offload_params and isinstance(model, FSDP) and (model.sharding_strategy not in NO_RESHARD_AFTER_FORWARD_STRATEGIES):
                    for p in model.parameters():
                        self.assertEqual(p.device, torch.device('cpu'))
                loss = model.module.get_loss(input, output).to(model_device)
            loss = sharded_grad_scaler.scale(loss)
            if not mixed_precision and (not use_pure_fp16):
                assert loss.dtype == torch.float32, 'loss data type should be float32, as the original                     parameter data type is float32.'
            elif use_pure_fp16:
                self.assertEqual(loss.dtype, torch.float16)
            elif isinstance(model, FSDP):
                self.assertEqual(loss.dtype, mixed_precision.param_dtype)
            else:
                self.assertEqual(loss.dtype, torch.float32)
            model.module.run_backward(loss)
            if cpu_offload_params and isinstance(model, FSDP):
                for p in model.parameters():
                    self.assertEqual(p.device, torch.device('cpu'))
            sharded_grad_scaler.step(optim)
            sharded_grad_scaler.update()
            if save_model:
                state_dict = {k: v.clone() for (k, v) in model.state_dict().items()}
                _zero_model(model)
                model.load_state_dict(state_dict)
        if isinstance(model, FSDP):
            model._assert_state(TrainingState.IDLE)
        return loss.detach()

    def _test_fsdp_parity(self, model_class: Type[FSDPTestModel], fsdp_init_mode: FSDPInitMode, cuda_init_mode: CUDAInitMode, ref_init_fn: Optional[Callable]=None, num_iters: int=2, save_model: bool=True, cpu_offload: CPUOffload=CPUOffload(), backward_prefetch: Optional[BackwardPrefetch]=None, sharding_strategy: Optional[ShardingStrategy]=None, mixed_precision: Optional[MixedPrecision]=None, forward_prefetch: bool=False, use_orig_params: bool=False, enable_sharded_grad_scaler: bool=False, use_pure_fp16: bool=False, init_kwargs: Optional[Dict[str, Any]]=None, sharded_grad_scaler_kwargs: Optional[Dict[str, Any]]=None, **fsdp_kwargs):
        if False:
            return 10
        '\n        Tests FSDP training against a reference, which defaults to DDP but\n        may be customized with ``ref_init_fn``.\n\n        Args:\n            model_class (Type[FSDPTestModel]): A model class that inherits from\n                ``FSDPTestModel``, which defines the expected interface.\n            fsdp_init_mode (FSDPInitMode): The mode to initialize the\n                FSDP-wrapped model. This should not be ``NO_FSDP``.\n            ref_init_fn (Optional[Callable]): A callable to invoke that wraps a\n                non-wrapped model to construct the reference model, where this\n                wrapper should provide data parallel semantics. If ``None``,\n                then the callable defaults to the DDP constructor.\n        '
        assert fsdp_init_mode != FSDPInitMode.NO_FSDP, 'Expects an FSDP init mode that wraps with FSDP'
        if init_kwargs is None:
            init_kwargs = {}
        lr = 0.01
        rank = self.process_group.rank()
        model = model_class.init(self.process_group, FSDPInitMode.NO_FSDP, CUDAInitMode.CUDA_BEFORE, deterministic=True, **init_kwargs)
        if ref_init_fn is None:
            ref_model = DDP(model, device_ids=[rank], output_device=rank)
        else:
            ref_model = ref_init_fn(model)
        if use_pure_fp16:
            ref_model = ref_model.half()
        ref_loss = self._train_for_several_steps(ref_model, num_iters, autocast=mixed_precision is not None, lr=lr, fsdp_cpu_offload=cpu_offload, mixed_precision=mixed_precision, enable_sharded_grad_scaler=enable_sharded_grad_scaler, use_pure_fp16=use_pure_fp16, sharded_grad_scaler_kwargs=sharded_grad_scaler_kwargs)
        ddp_params = list(ref_model.parameters())
        fsdp_kwargs.update({'cpu_offload': cpu_offload, 'backward_prefetch': backward_prefetch, 'sharding_strategy': sharding_strategy, 'mixed_precision': mixed_precision, 'forward_prefetch': forward_prefetch, 'use_orig_params': use_orig_params})
        try:
            fsdp_model = model_class.init(self.process_group, fsdp_init_mode, cuda_init_mode, fsdp_kwargs, deterministic=True, **init_kwargs)
        except Exception as e:
            raise ValueError(f'Initializing {model_class} raised error {str(e)}') from e
        if not isinstance(fsdp_model, FSDP):
            fsdp_model = FSDP(fsdp_model, self.process_group, **fsdp_kwargs)
        if use_pure_fp16:
            fsdp_model = fsdp_model.half()
        if cuda_init_mode == CUDAInitMode.CUDA_AFTER:
            fsdp_model = fsdp_model.cuda()
        offload_params = cpu_offload is not None and cpu_offload.offload_params
        expects_device_error = offload_params and cuda_init_mode == CUDAInitMode.CUDA_AFTER
        expects_cpu_device = offload_params and cuda_init_mode != CUDAInitMode.CUDA_AFTER
        if expects_cpu_device:
            cpu_device = torch.device('cpu')
            for param in fsdp_model.parameters():
                self.assertEqual(param.device, cpu_device)
        context = self.assertRaisesRegex(RuntimeError, 'An FSDP-managed module with parameter CPU offloading enabled has parameters on cuda') if expects_device_error else nullcontext()
        with context:
            fsdp_loss = self._train_for_several_steps(fsdp_model, num_iters, autocast=False, lr=lr, fsdp_cpu_offload=cpu_offload, save_model=save_model, mixed_precision=mixed_precision, enable_sharded_grad_scaler=enable_sharded_grad_scaler, use_pure_fp16=use_pure_fp16, sharded_grad_scaler_kwargs=sharded_grad_scaler_kwargs)
        if expects_device_error:
            return
        if offload_params:
            for param in fsdp_model.parameters():
                self.assertEqual(param.device, cpu_device)
            fsdp_loss = fsdp_loss.cuda()
        fsdp_unsharded_params = get_full_params(fsdp_model)
        torch.testing.assert_close(ref_loss, fsdp_loss, check_dtype=False)
        if mixed_precision is None and (not use_pure_fp16):
            self.assertEqual(ddp_params, fsdp_unsharded_params, exact_device=True, msg='FSDP did not match DDP')

class SkipModule(nn.Module):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.lin = nn.Linear(10, 10, bias=False)

    def forward(self, x):
        if False:
            print('Hello World!')
        return self.lin(x)

class NestedLinear(nn.Module):

    def __init__(self, fsdp_wrap):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        if fsdp_wrap:
            self.nested_linear = wrap(nn.Linear(10, 10, bias=False).cuda())
        else:
            self.nested_linear = nn.Linear(10, 10, bias=False).cuda()

    def forward(self, x):
        if False:
            return 10
        return self.nested_linear(x)

class SkipModel(nn.Module):

    def __init__(self, double_nest):
        if False:
            while True:
                i = 10
        super().__init__()
        self.linear = nn.Linear(10, 10, bias=False).cuda()
        self.linear_skip = SkipModule().cuda()
        self.nested_linear = wrap(NestedLinear(fsdp_wrap=double_nest))

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        x = self.linear(x)
        x = self.linear_skip(x)
        x = self.nested_linear(x)
        return x