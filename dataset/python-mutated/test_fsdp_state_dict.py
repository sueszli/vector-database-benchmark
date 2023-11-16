import io
import itertools
import sys
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from typing import Any, Dict
import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed._shard.sharded_tensor import init_from_local_shards, Shard, ShardedTensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl
from torch.distributed.checkpoint._state_dict_utils import _all_gather_sharded_tensor, _gather_state_dict
from torch.distributed.fsdp import CPUOffload, FullStateDictConfig, FullyShardedDataParallel as FSDP, LocalStateDictConfig, MixedPrecision, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp._unshard_param_utils import FLAT_PARAM
from torch.distributed.fsdp.wrap import enable_wrap, ModuleWrapPolicy, wrap
from torch.nn import Linear, Module, TransformerDecoderLayer, TransformerEncoderLayer
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import _assert_module_states, _broadcast_state_dict, _get_state_dict, _zero_model, CUDAInitMode, FSDPInitMode, FSDPTest, get_full_params, SkipModel, TransformerWithSharedParams
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TEST_WITH_DEV_DBG_ASAN
if not dist.is_available():
    print('Distributed not available, skipping tests', file=sys.stderr)
    sys.exit(0)
if TEST_WITH_DEV_DBG_ASAN:
    print('Skip dev-asan as torch + multiprocessing spawn have known issues', file=sys.stderr)
    sys.exit(0)
INNER_SHAPE = [4, 4]
OUTER_SHAPE = [4, 5]
BUFFER_SHAPE = [5, 5]
NON_ROOT_FSDP_PREFIX = 'non_fsdp_lin'
_UNFLATTENED_STATE_DICT_IMPLS = ['state_dict', 'sharded_state_dict']
_FLATTENED_STATE_DICT_IMPLS = ['local_state_dict']
_SUPPORTED_STATE_DICT_IMPLS = _UNFLATTENED_STATE_DICT_IMPLS + _FLATTENED_STATE_DICT_IMPLS
STATE_DICT_MAPPING = {'state_dict': StateDictType.FULL_STATE_DICT, 'local_state_dict': StateDictType.LOCAL_STATE_DICT, 'sharded_state_dict': StateDictType.SHARDED_STATE_DICT}

class Model(Module):

    def __init__(self, wrap_fsdp, register_buffers=False, ignore_inner=False, mixed_precision=False, process_group=None):
        if False:
            print('Hello World!')
        super().__init__()
        self.inner = Linear(*INNER_SHAPE)
        if register_buffers:
            self.inner.register_buffer('buffer', torch.randn(BUFFER_SHAPE))
            self.inner.register_buffer('non_persistent_buffer', torch.randn(BUFFER_SHAPE), persistent=False)
        if wrap_fsdp:
            self.inner = FSDP(self.inner, ignored_modules=[self.inner] if ignore_inner else [], mixed_precision=MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16) if mixed_precision else None, process_group=process_group)
        self.outer = Linear(*OUTER_SHAPE)
        if register_buffers:
            self.outer.register_buffer('buffer', torch.randn(BUFFER_SHAPE))
            self.outer.register_buffer('non_persistent_buffer', torch.randn(BUFFER_SHAPE), persistent=False)

    def forward(self, x):
        if False:
            return 10
        i = self.inner(x)
        j = self.inner(x)
        return self.outer(i + j)

class TestDummyModel(torch.nn.Module):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
        self.net3 = self.net2
        self.random_parameter = nn.Parameter(torch.Tensor(10))
        self.shared_parameter = self.random_parameter

    def forward(self, x):
        if False:
            return 10
        return self.net3(self.net2(self.net1(x)))

    def get_input(self):
        if False:
            print('Hello World!')
        return torch.rand(8, 8, device='cuda')

class TestFSDPStateDict(FSDPTest):

    @property
    def world_size(self):
        if False:
            while True:
                i = 10
        return 2

    def _broadcast_state_dict(self, model, state_dict):
        if False:
            for i in range(10):
                print('nop')
        return _broadcast_state_dict(self.rank, state_dict)

    def _state_compare(self, model, model_new, assert_fn, state_generator='parameters'):
        if False:
            return 10
        state_base = list(getattr(model, state_generator)())
        state_new = list(getattr(model_new, state_generator)())
        self.assertEqual(len(state_base), len(state_new))
        assert_fn(state_base, state_new)

    def _compare_models(self, model, model_new, assert_fn, check_fp16=False, check_buffers=True):
        if False:
            i = 10
            return i + 15
        assert assert_fn in (self.assertEqual, self.assertNotEqual)
        with FSDP.summon_full_params(model):
            with FSDP.summon_full_params(model_new):
                self._state_compare(model, model_new, assert_fn)
                if check_buffers:
                    has_buffers = any((len(list(m.buffers())) for m in (model, model_new)))
                    if has_buffers:
                        self._state_compare(model, model_new, assert_fn, state_generator='buffers')
                if check_fp16:
                    for tensor in model_new.parameters():
                        self.assertEqual(tensor.dtype, torch.float16)

    def _get_simple_nested_model(self, *fsdp_args, wrap=True, checkpoint_wrap=False, **fsdp_kwargs):
        if False:
            i = 10
            return i + 15
        if wrap:
            lin1 = nn.Linear(10, 10, bias=False).cuda()
            lin2 = nn.Linear(10, 10, bias=False).cuda()
            if checkpoint_wrap:
                lin1 = checkpoint_wrapper(lin1)
                lin2 = checkpoint_wrapper(lin2)
            seq = nn.Sequential(FSDP(lin1, *fsdp_args, **fsdp_kwargs), lin2)
            if checkpoint_wrap:
                seq = checkpoint_wrapper(seq)
            model = FSDP(seq, *fsdp_args, **fsdp_kwargs)
        else:
            model = nn.Sequential(nn.Linear(10, 10, bias=False).cuda(), nn.Linear(10, 10, bias=False).cuda())
        return model

    def _get_simple_model(self, *fsdp_args, checkpoint_wrap=False, **fsdp_kwargs):
        if False:
            print('Hello World!')
        lin = nn.Linear(10, 10, bias=False).cuda()
        if checkpoint_wrap:
            lin = checkpoint_wrapper(lin)
        model = FSDP(lin, *fsdp_args, **fsdp_kwargs)
        return model

    def _get_multibuffer_nested_model(self, *fsdp_args, wrap=True, checkpoint_wrap=False, **fsdp_kwargs):
        if False:
            while True:
                i = 10
        full_p = torch.float32
        lin_mp = fsdp_kwargs.pop('mixed_precision', None)
        bn_mp = MixedPrecision(param_dtype=full_p, reduce_dtype=full_p, buffer_dtype=full_p) if lin_mp else None
        if wrap:
            lin1 = nn.Linear(10, 10, bias=False).cuda()
            bn1 = nn.BatchNorm1d(10).cuda()
            lin2 = nn.Linear(10, 10, bias=False).cuda()
            if checkpoint_wrap:
                lin1 = checkpoint_wrapper(lin1)
                bn1 = checkpoint_wrapper(bn1)
                lin2 = checkpoint_wrapper(lin2)
            seq = nn.Sequential(FSDP(lin1, *fsdp_args, mixed_precision=lin_mp, **fsdp_kwargs), FSDP(bn1, *fsdp_args, mixed_precision=bn_mp, **fsdp_kwargs), lin2)
            if checkpoint_wrap:
                seq = checkpoint_wrapper(seq)
            model = FSDP(seq, *fsdp_args, **fsdp_kwargs)
        else:
            model = nn.Sequential(nn.Linear(10, 10, bias=False).cuda(), nn.BatchNorm1d(10).cuda(), nn.Linear(10, 10, bias=False).cuda())
        return model

    def _get_non_fsdp_root_module(self, *fsdp_args, wrap=True, **fsdp_kwargs):
        if False:
            for i in range(10):
                print('nop')

        class FSDPContainer(nn.Module):

            def __init__(self, fsdp_1, fsdp_2):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.non_fsdp_lin = nn.Linear(10, 10, bias=False).cuda()
                self.fsdp_1 = fsdp_1
                self.fsdp_2 = fsdp_2

            def forward(self, x):
                if False:
                    return 10
                x = self.non_fsdp_lin(x)
                x = self.fsdp_1(x)
                x = self.fsdp_2(x)
                return x
        return FSDPContainer(self._get_simple_nested_model(*fsdp_args, wrap=wrap, **fsdp_kwargs), self._get_simple_nested_model(*fsdp_args, wrap=wrap, **fsdp_kwargs))

    def _get_state_dict_mgr(self, model: nn.Module, state_dict_type: str, state_dict_rank0_and_offload: bool):
        if False:
            for i in range(10):
                print('nop')
        _state_dict_type = STATE_DICT_MAPPING[state_dict_type]
        if state_dict_type == 'state_dict':
            config = FullStateDictConfig(rank0_only=state_dict_rank0_and_offload, offload_to_cpu=state_dict_rank0_and_offload)
        elif state_dict_type == 'local_state_dict':
            config = LocalStateDictConfig(offload_to_cpu=state_dict_rank0_and_offload)
        elif state_dict_type == 'sharded_state_dict':
            config = ShardedStateDictConfig(offload_to_cpu=state_dict_rank0_and_offload)
        else:
            raise ValueError('Unsupported state_dict_type')
        return FSDP.state_dict_type(model, _state_dict_type, config)

    def _validate_state_dict_contents(self, model, fsdp_state_dict, state_dict_rank0_and_offload, ignore_keys=None):
        if False:
            for i in range(10):
                print('nop')
        if state_dict_rank0_and_offload:
            if self.rank == 0:
                self.assertNotEqual(fsdp_state_dict, {})
                for (key, tensor) in fsdp_state_dict.items():
                    if ignore_keys and key in ignore_keys:
                        continue
                    self.assertEqual(tensor.device, torch.device('cpu'), f'{key} is unexpectedly on device {tensor.device}')
            elif isinstance(model, FSDP):
                self.assertEqual(fsdp_state_dict, {}, f'Expected empty state_dict but got {fsdp_state_dict} on rank {dist.get_rank()}')

    @skip_if_lt_x_gpu(2)
    @parametrize('state_dict_type', _UNFLATTENED_STATE_DICT_IMPLS)
    @parametrize('checkpoint_wrap', ['source', 'dest', 'both', 'source_after_wrap', 'both_after_wrap'])
    @parametrize('rank0_only_and_offload', [False, True])
    def test_fsdp_state_dict_with_activation_checkpoint(self, state_dict_type, checkpoint_wrap, rank0_only_and_offload):
        if False:
            return 10
        "Tests saving the state dict, zeroing a target model's parameters, and\n        loading the state dict, where the source and target models may have a\n        checkpoint wrapper."

        def apply_ac_to_linears(model) -> None:
            if False:
                while True:
                    i = 10
            non_reentrant_wrapper = partial(checkpoint_wrapper, offload_to_cpu=False, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
            apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=lambda submodule: isinstance(submodule, nn.Linear))
        for model_call in [partial(self._get_simple_model), partial(self._get_simple_nested_model)]:
            model = model_call(checkpoint_wrap=checkpoint_wrap in ('source', 'both'))
            if checkpoint_wrap in ('source_after_wrap', 'both_after_wrap'):
                apply_ac_to_linears(model)
            with self._get_state_dict_mgr(model, state_dict_type, rank0_only_and_offload):
                state_dict = _gather_state_dict(_get_state_dict(model, False, False))
                model_new = model_call(checkpoint_wrap=checkpoint_wrap in ('dest', 'both'))
                if checkpoint_wrap == 'both_after_wrap':
                    apply_ac_to_linears(model_new)
                _zero_model(model_new)
                self._compare_models(model, model_new, self.assertNotEqual)
                if rank0_only_and_offload:
                    state_dict = self._broadcast_state_dict(model, state_dict)
                model_new.load_state_dict(state_dict, strict=True)
                self._compare_models(model, model_new, self.assertEqual)

    @skip_if_lt_x_gpu(2)
    @parametrize('state_dict_type', _UNFLATTENED_STATE_DICT_IMPLS)
    @parametrize('rank0_only_and_offload', [False, True])
    def test_state_dict_with_manual_ac_wrapper(self, state_dict_type: str, rank0_only_and_offload: bool):
        if False:
            return 10
        '\n        Tests saving and loading a state dict for a model manually wrapped with\n        ``FSDP(CheckpointWrapper(module))``, where the ``CheckpointWrapper`` is\n        wrapped before FSDP.\n\n        TODO: Investigate why the test above does not cover everything in this\n        test and de-duplicate afterwards.\n        '
        if state_dict_type == 'sharded_state_dict' and rank0_only_and_offload:
            return
        model_ac = TransformerWithSharedParams.init(self.process_group, FSDPInitMode.NO_FSDP, CUDAInitMode.CUDA_BEFORE)
        model_no_ac = deepcopy(model_ac)
        for (i, layer) in enumerate(model_no_ac.transformer.encoder.layers):
            model_no_ac.transformer.encoder.layers[i] = FSDP(layer)
        for (i, layer) in enumerate(model_no_ac.transformer.decoder.layers):
            model_no_ac.transformer.decoder.layers[i] = FSDP(layer)
        model_no_ac.transformer = FSDP(model_no_ac.transformer)
        for (i, layer) in enumerate(model_ac.transformer.encoder.layers):
            layer = checkpoint_wrapper(layer)
            model_ac.transformer.encoder.layers[i] = FSDP(layer)
        for (i, layer) in enumerate(model_ac.transformer.decoder.layers):
            layer = checkpoint_wrapper(layer)
            model_ac.transformer.decoder.layers[i] = FSDP(layer)
        model_ac.transformer = FSDP(model_ac.transformer)
        with self._get_state_dict_mgr(model_no_ac, state_dict_type, rank0_only_and_offload):
            state_dict_no_ac = model_no_ac.state_dict()
        with self._get_state_dict_mgr(model_ac, state_dict_type, rank0_only_and_offload):
            state_dict_ac = model_ac.state_dict()
        self.assertEqual(state_dict_ac.keys(), state_dict_no_ac.keys())
        if rank0_only_and_offload:
            state_dict_no_ac = self._broadcast_state_dict(model_no_ac, state_dict_no_ac)
            state_dict_ac = self._broadcast_state_dict(model_ac, state_dict_ac)
        with self._get_state_dict_mgr(model_no_ac, state_dict_type, rank0_only_and_offload):
            model_no_ac.load_state_dict(state_dict_no_ac)
        with self._get_state_dict_mgr(model_ac, state_dict_type, rank0_only_and_offload):
            model_ac.load_state_dict(state_dict_ac)
        self._compare_models(model_ac, model_no_ac, self.assertEqual)

    @skip_if_lt_x_gpu(2)
    @parametrize('state_dict_type', _SUPPORTED_STATE_DICT_IMPLS)
    def test_state_dict_with_shared_parameters(self, state_dict_type):
        if False:
            while True:
                i = 10
        auto_wrap_policy = ModuleWrapPolicy({TransformerEncoderLayer, TransformerDecoderLayer})
        model_creator = partial(TransformerWithSharedParams.init, self.process_group, FSDPInitMode.RECURSIVE, CUDAInitMode.CUDA_BEFORE, {'auto_wrap_policy': auto_wrap_policy})
        fsdp_model = model_creator()
        with self._get_state_dict_mgr(fsdp_model, state_dict_type, False):
            state_dict = fsdp_model.state_dict()
        new_model = model_creator()
        _zero_model(new_model, zero_buffers=True)
        with self._get_state_dict_mgr(new_model, state_dict_type, False):
            new_model.load_state_dict(state_dict)

    @skip_if_lt_x_gpu(2)
    @parametrize('use_orig_params', [False, True])
    def test_state_dict_rank0_offload_save_load_flow(self, use_orig_params: bool):
        if False:
            i = 10
            return i + 15
        'Tests saving a model checkpoint only on rank 0 and loading it only\n        on rank 0 with ``sync_module_states=True`` to emulate the workflow to\n        avoid redundant CPU memory usage.'
        auto_wrap_policy = ModuleWrapPolicy({TransformerEncoderLayer, TransformerDecoderLayer})
        fsdp_kwargs = {'auto_wrap_policy': auto_wrap_policy, 'use_orig_params': use_orig_params}
        fsdp_model = TransformerWithSharedParams.init(self.process_group, FSDPInitMode.RECURSIVE, CUDAInitMode.CUDA_BEFORE, fsdp_kwargs)
        with FSDP.summon_full_params(fsdp_model):
            for tensor in itertools.chain(fsdp_model.parameters(), fsdp_model.buffers()):
                if torch.count_nonzero(tensor) == 0:
                    with torch.no_grad():
                        tensor.add_(torch.ones_like(tensor))
        with self._get_state_dict_mgr(fsdp_model, 'state_dict', True):
            state_dict = deepcopy(_get_state_dict(fsdp_model))
        new_model = TransformerWithSharedParams.init(self.process_group, FSDPInitMode.NO_FSDP, CUDAInitMode.CUDA_BEFORE)
        _zero_model(new_model, zero_buffers=True)
        if self.rank == 0:
            new_model.load_state_dict(state_dict, strict=True)
        _assert_module_states(new_model, process_group=self.process_group, assert_fn=self.assertNotEqual)
        new_fsdp_model = FSDP(new_model, device_id=torch.cuda.current_device(), auto_wrap_policy=auto_wrap_policy, sync_module_states=True)
        with FSDP.summon_full_params(new_fsdp_model):
            _assert_module_states(new_fsdp_model, process_group=self.process_group, assert_fn=self.assertEqual)
        with FSDP.summon_full_params(fsdp_model):
            with FSDP.summon_full_params(new_fsdp_model):
                params = list(fsdp_model.parameters())
                params_new = list(new_fsdp_model.parameters())
                self.assertEqual(params, params_new)

    @skip_if_lt_x_gpu(2)
    @parametrize('state_dict_type', _SUPPORTED_STATE_DICT_IMPLS)
    @parametrize('cpu_offload', [CPUOffload(offload_params=True), CPUOffload(offload_params=False)])
    @parametrize('fp16', [True, False])
    @parametrize('state_dict_rank0_and_offload', [True, False])
    @parametrize('use_orig_params', [True, False])
    def test_basic_save_and_load_state_dict(self, state_dict_type: str, cpu_offload: bool, fp16: bool, state_dict_rank0_and_offload: bool, use_orig_params: bool):
        if False:
            print('Hello World!')
        '\n        Tests that we can save a state_dict and load it into a blank model\n        with various configs such as fp16 and cpu offload and parameters\n        match as expected.\n        '
        if state_dict_rank0_and_offload and state_dict_type != 'state_dict' or (use_orig_params and state_dict_type not in _UNFLATTENED_STATE_DICT_IMPLS):
            return
        device = torch.device(self.rank)
        for model_call in [partial(self._get_non_fsdp_root_module, cpu_offload=cpu_offload, use_orig_params=use_orig_params), partial(self._get_simple_nested_model, cpu_offload=cpu_offload, use_orig_params=use_orig_params), partial(self._get_simple_model, cpu_offload=cpu_offload, use_orig_params=use_orig_params)]:
            model = model_call()
            if fp16:
                model.half()
            inp = torch.randn((3, 10), device=device)
            if fp16:
                inp = inp.half()
            model(inp).sum().backward()
            ctx = self._get_state_dict_mgr(model, state_dict_type, state_dict_rank0_and_offload)
            with ctx:
                fsdp_state_dict = _get_state_dict(model, cpu_offload.offload_params, fp16)
            ignore_keys = [k for k in fsdp_state_dict.keys() if NON_ROOT_FSDP_PREFIX in k]
            self._validate_state_dict_contents(model, fsdp_state_dict, state_dict_rank0_and_offload, ignore_keys=ignore_keys)
            if fp16:
                for tensor in fsdp_state_dict.values():
                    self.assertEqual(tensor.dtype, torch.float16)
            model_new = model_call()
            if not cpu_offload.offload_params:
                model_new = model_new.cuda()
            if fp16:
                model_new.half()
            inp = torch.randn((3, 10), device=device)
            if fp16:
                inp = inp.half()
            model_new(inp).sum().backward()
            _zero_model(model_new, zero_buffers=True)
            self._compare_models(model, model_new, self.assertNotEqual)
            if state_dict_rank0_and_offload:
                fsdp_state_dict = self._broadcast_state_dict(model, fsdp_state_dict)
            with FSDP.state_dict_type(model_new, STATE_DICT_MAPPING[state_dict_type]):
                model_new.load_state_dict(fsdp_state_dict, strict=True)
            self._compare_models(model, model_new, self.assertEqual, check_fp16=fp16)

    @skip_if_lt_x_gpu(2)
    @parametrize('state_dict_type', _SUPPORTED_STATE_DICT_IMPLS)
    @parametrize('cpu_offload', [CPUOffload(offload_params=True), CPUOffload(offload_params=False)])
    @parametrize('mixed_precision', [True, False])
    @parametrize('state_dict_rank0_and_offload', [True, False])
    @parametrize('use_orig_params', [True, False])
    def test_buffers_save_and_load_state_dict(self, state_dict_type: str, cpu_offload: bool, mixed_precision: bool, state_dict_rank0_and_offload: bool, use_orig_params: bool):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests that we can save a state_dict and load it for modules with persistent buffers, including\n        in the context of non-default mixed precision, different ``state_dict_type`` s and CPU offloading.\n        '
        if state_dict_rank0_and_offload and state_dict_type != 'state_dict' or (use_orig_params and state_dict_type not in _UNFLATTENED_STATE_DICT_IMPLS):
            return
        mixed_precision = MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16) if mixed_precision else None
        model_call = partial(self._get_multibuffer_nested_model, cpu_offload=cpu_offload, use_orig_params=use_orig_params, mixed_precision=mixed_precision)
        model = model_call()
        ctx = self._get_state_dict_mgr(model, state_dict_type, state_dict_rank0_and_offload)
        with ctx:
            fsdp_state_dict = _get_state_dict(model, cpu_offload.offload_params, False)
        self._validate_state_dict_contents(model, fsdp_state_dict, state_dict_rank0_and_offload)
        model_new = model_call()
        if not cpu_offload.offload_params:
            model_new = model_new.cuda()
        _zero_model(model_new, zero_buffers=True)
        self._compare_models(model, model_new, self.assertNotEqual)
        if state_dict_rank0_and_offload:
            fsdp_state_dict = self._broadcast_state_dict(model, fsdp_state_dict)
        with FSDP.state_dict_type(model_new, STATE_DICT_MAPPING[state_dict_type]):
            model_new.load_state_dict(fsdp_state_dict, strict=True)
        self._compare_models(model, model_new, self.assertEqual)

    @skip_if_lt_x_gpu(2)
    @parametrize('state_dict_type', _SUPPORTED_STATE_DICT_IMPLS)
    @parametrize('mixed_precision', [True, False])
    @parametrize('state_dict_rank0_and_offload', [True, False])
    def test_save_and_load_after_forward_state_dict(self, state_dict_type, mixed_precision, state_dict_rank0_and_offload):
        if False:
            i = 10
            return i + 15
        '\n        Test that saving after some training results in params being updated as\n        expected.\n        '
        if state_dict_rank0_and_offload and state_dict_type != 'state_dict':
            return
        torch.cuda.set_device(self.rank)
        mixed_precision = MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16) if mixed_precision else None
        model = self._get_simple_nested_model(mixed_precision=mixed_precision)
        optim = torch.optim.SGD(model.parameters(), lr=0.1)
        initial_params = get_full_params(model)
        for _ in range(6):
            inp = torch.randn(1, 10, device=torch.cuda.current_device())
            output = model(*inp)
            loss = output.sum()
            expected_dtype = torch.float32 if mixed_precision is None else torch.float16
            self.assertEqual(expected_dtype, loss.dtype)
            loss.backward()
            optim.step()
        trained_params = get_full_params(model)
        self.assertNotEqual(initial_params, trained_params)
        fsd_mgr = self._get_state_dict_mgr(model, state_dict_type, state_dict_rank0_and_offload)
        with fsd_mgr:
            state_dict = model.state_dict()
            if state_dict_type == 'state_dict':
                state_dict = {k: v.clone() for (k, v) in state_dict.items()}
            else:
                for sharded_tensor in state_dict.values():
                    shard = sharded_tensor._local_shards[0]
                    shard.tensor = shard.tensor.clone().detach_()
        self._validate_state_dict_contents(model, state_dict, state_dict_rank0_and_offload)
        _zero_model(model)
        for tensor in state_dict.values():
            self.assertEqual(tensor.dtype, torch.float32)
        if state_dict_rank0_and_offload:
            state_dict = self._broadcast_state_dict(model, state_dict)
        with FSDP.state_dict_type(model, STATE_DICT_MAPPING[state_dict_type]):
            model.load_state_dict(state_dict, strict=True)
        loaded_params = get_full_params(model)
        self.assertEqual(loaded_params, trained_params)

    def _initialize_model(self, wrap_fsdp: bool, wrap_ddp: bool=True, register_buffers: bool=False):
        if False:
            i = 10
            return i + 15
        torch.manual_seed(0)
        model = Model(wrap_fsdp, register_buffers=register_buffers).cuda()
        if wrap_fsdp:
            model = FSDP(model)
        elif wrap_ddp:
            model = DistributedDataParallel(model, device_ids=[self.rank])
        return model

    @staticmethod
    def _state_dict(model: Module, state_dict_type: str):
        if False:
            for i in range(10):
                print('nop')
        try:
            enum_val = STATE_DICT_MAPPING[state_dict_type]
        except KeyError as e:
            raise ValueError(f'No state_dict type for {state_dict_type}') from e
        with FSDP.state_dict_type(model, enum_val):
            return model.state_dict()

    @staticmethod
    def _load_state_dict(model: Module, state_dict_type: str, state_dict: Dict[str, Any]):
        if False:
            while True:
                i = 10
        try:
            enum_val = STATE_DICT_MAPPING[state_dict_type]
        except KeyError as e:
            raise ValueError(f'No state_dict for {state_dict_type}') from e
        with FSDP.state_dict_type(model, enum_val):
            return model.load_state_dict(state_dict, strict=True)

    def _dist_train(self, wrap_fsdp: bool, state_dict_type: str='', move_to_cpu: bool=False):
        if False:
            i = 10
            return i + 15
        model = self._initialize_model(wrap_fsdp)
        optim = SGD(model.parameters(), lr=0.1)
        in_data = torch.rand(64, 4, requires_grad=True, device=torch.device('cuda'))
        for _ in range(3):
            out = model(in_data)
            out.sum().backward()
            optim.step()
            optim.zero_grad()
        if wrap_fsdp:
            blank_model = FSDP(Model(True).cuda())
            _zero_model(blank_model)
            state_dict = self._state_dict(model, state_dict_type)
            if move_to_cpu:
                for key in list(state_dict.keys()):
                    tensor = state_dict[key]
                    if isinstance(tensor, torch.Tensor):
                        state_dict[key] = tensor.cpu()
                    else:
                        shards = tensor.local_shards()
                        if shards:
                            shards[0].tensor = shards[0].tensor.cpu()
            self._load_state_dict(blank_model, state_dict_type, state_dict)
            return get_full_params(blank_model)
        else:
            return list(model.parameters())

    @skip_if_lt_x_gpu(2)
    @parametrize('state_dict_type', _SUPPORTED_STATE_DICT_IMPLS)
    def test_state_dict_save_load_flow(self, state_dict_type):
        if False:
            while True:
                i = 10
        self.run_subtests({'move_to_cpu': [True, False]}, self._test_state_dict_save_load_flow, state_dict_type=state_dict_type)

    def _test_state_dict_save_load_flow(self, state_dict_type, move_to_cpu):
        if False:
            while True:
                i = 10
        fsdp_params = self._dist_train(wrap_fsdp=True, state_dict_type=state_dict_type, move_to_cpu=move_to_cpu)
        ddp_params = self._dist_train(wrap_fsdp=False)
        self.assertEqual(ddp_params, fsdp_params)

    @skip_if_lt_x_gpu(2)
    @parametrize('state_dict_type', _SUPPORTED_STATE_DICT_IMPLS)
    def test_fsdp_state_dict_keys(self, state_dict_type):
        if False:
            i = 10
            return i + 15
        state_dict = self._state_dict(self._initialize_model(True), state_dict_type)
        if state_dict_type == 'local_state_dict':
            self.assertEqual({FLAT_PARAM, f'inner.{FLAT_PARAM}'}, state_dict.keys())
        elif state_dict_type in ('state_dict', 'sharded_state_dict'):
            local_model = self._initialize_model(wrap_fsdp=False, wrap_ddp=False)
            local_keys = local_model.state_dict().keys()
            self.assertEqual(state_dict.keys(), local_keys)
        else:
            raise NotImplementedError(f'No test for {state_dict_type}!')

    @skip_if_lt_x_gpu(2)
    @parametrize('state_dict_type', _UNFLATTENED_STATE_DICT_IMPLS)
    @parametrize('state_dict_rank0_and_offload', [True, False])
    @parametrize('fsdp_root', [True, False])
    def test_state_dict_load_into_local_module(self, state_dict_type, state_dict_rank0_and_offload, fsdp_root):
        if False:
            print('Hello World!')
        "\n        Tests that FSDP's state_dict can be loaded into a local model.\n        "
        if state_dict_rank0_and_offload and state_dict_type != 'state_dict':
            return
        if not fsdp_root:
            model = self._get_non_fsdp_root_module()
        else:
            model = self._initialize_model(wrap_fsdp=True, register_buffers=True)
        optim = SGD(model.parameters(), lr=0.1)
        if not fsdp_root:
            in_data = torch.randn(1, 10, requires_grad=True, device=torch.device('cuda'))
        else:
            in_data = torch.rand(64, 4, requires_grad=True, device=torch.device('cuda'))
        for _ in range(3):
            out = model(in_data)
            out.sum().backward()
            optim.step()
            optim.zero_grad()
        with FSDP.summon_full_params(model):
            fsdp_params = deepcopy(list(model.parameters()))
        sd_mgr = self._get_state_dict_mgr(model, state_dict_type, state_dict_rank0_and_offload)
        with sd_mgr:
            fsdp_state_dict = model.state_dict()
        ignore_keys = [k for k in fsdp_state_dict.keys() if NON_ROOT_FSDP_PREFIX in k]
        self._validate_state_dict_contents(model, fsdp_state_dict, state_dict_rank0_and_offload, ignore_keys=ignore_keys)
        if not fsdp_root:
            blank_local_model = self._get_non_fsdp_root_module(wrap=False)
        else:
            blank_local_model = self._initialize_model(wrap_fsdp=False, wrap_ddp=False, register_buffers=True)
        for mod in blank_local_model.modules():
            self.assertFalse(isinstance(mod, FSDP))
        for param in blank_local_model.parameters():
            with torch.no_grad():
                param.zero_()
        fsdp_state_dict = _gather_state_dict(fsdp_state_dict)
        if state_dict_rank0_and_offload:
            fsdp_state_dict = self._broadcast_state_dict(model, fsdp_state_dict)
        blank_local_model.load_state_dict(fsdp_state_dict, strict=True)
        local_params = list(blank_local_model.parameters())
        for (fsdp_param, local_param) in zip(fsdp_params, local_params):
            self.assertEqual(fsdp_param, local_param)

    @skip_if_lt_x_gpu(2)
    @parametrize('state_dict_type', _SUPPORTED_STATE_DICT_IMPLS)
    @parametrize('double_nest', [True])
    def test_state_dict_skip_module(self, state_dict_type, double_nest):
        if False:
            for i in range(10):
                print('nop')
        torch.cuda.set_device(self.rank)

        def _create_module(wrap_fsdp=True):
            if False:
                print('Hello World!')
            LINEAR_SKIP = 'linear_skip'
            ctx = enable_wrap(wrapper_cls=FSDP) if wrap_fsdp else nullcontext()
            with ctx:
                module = SkipModel(double_nest=double_nest)
                linear_skip_tensor_names = [k for k in dict(module.named_parameters()).keys() if LINEAR_SKIP in k]
                linear_skip = getattr(module, LINEAR_SKIP)
                delattr(module, LINEAR_SKIP)
                fsdp = wrap(module)
                setattr(module, LINEAR_SKIP, linear_skip)
                return (fsdp, linear_skip_tensor_names)
        (fsdp, linear_skip_tensor_names) = _create_module()
        inp = torch.randn((1, 10), device=torch.cuda.current_device())
        loss = fsdp(inp)
        loss.sum().backward()
        with FSDP.state_dict_type(fsdp, STATE_DICT_MAPPING[state_dict_type]):
            state_dict = fsdp.state_dict()
        if self.rank == 0 and state_dict_type != 'local_state_dict':
            sd_keys = list(state_dict.keys())
            expected = list(SkipModel(double_nest=False).state_dict().keys())
            self.assertEqual(sorted(sd_keys), sorted(expected))
        (new_fsdp, _) = _create_module()
        _zero_model(new_fsdp)
        for (p1, p2) in zip(fsdp.parameters(), new_fsdp.parameters()):
            self.assertNotEqual(p1, p2)
        with FSDP.state_dict_type(new_fsdp, STATE_DICT_MAPPING[state_dict_type]):
            if state_dict_type != 'local_state_dict':
                state_dict = deepcopy(state_dict)
            new_fsdp.load_state_dict(state_dict, strict=True)
        for (p1, p2) in zip(fsdp.parameters(), new_fsdp.parameters()):
            self.assertEqual(p1, p2)
        (local, _) = _create_module(wrap_fsdp=False)
        for param in local.parameters():
            with torch.no_grad():
                param.zero_()
        with fsdp.summon_full_params(fsdp):
            for (p1, p2) in zip(fsdp.parameters(), local.parameters()):
                self.assertNotEqual(p1, p2)
        if state_dict_type == 'local_state_dict':
            return
        state_dict = _gather_state_dict(state_dict)
        with fsdp.summon_full_params(fsdp):
            if self.rank == 0:
                local.load_state_dict(state_dict, strict=True)
                for (p1, p2) in zip(fsdp.parameters(), local.parameters()):
                    self.assertEqual(p1, p2)

    @skip_if_lt_x_gpu(2)
    def test_wrong_state_dict_config(self):
        if False:
            i = 10
            return i + 15
        model = FSDP(Model(wrap_fsdp=True).cuda())
        with self.assertRaisesRegex(RuntimeError, 'Expected state_dict_config of type'):
            with model.state_dict_type(model, StateDictType.FULL_STATE_DICT, LocalStateDictConfig()):
                pass

    @skip_if_lt_x_gpu(2)
    @parametrize('state_dict_type', _UNFLATTENED_STATE_DICT_IMPLS)
    @parametrize('prefix', [True, False])
    @parametrize('ignore_inner', [True, False])
    @parametrize('mixed_precision', [True, False])
    def test_state_dict_with_ignored_modules(self, state_dict_type, prefix, ignore_inner, mixed_precision):
        if False:
            for i in range(10):
                print('nop')
        model = Model(wrap_fsdp=True, register_buffers=True, ignore_inner=ignore_inner, mixed_precision=mixed_precision).cuda()
        ignored_modules = [model.outer]
        ignored_tensor_to_tensor_name = {model.outer.bias: 'outer.bias', model.outer.weight: 'outer.weight'}
        if ignore_inner:
            ignored_tensor_to_tensor_name = {**ignored_tensor_to_tensor_name, model.inner.bias: 'inner.bias', model.inner.weight: 'inner.weight'}
        buffer_to_buffer_name = {model.inner.buffer: 'inner.buffer', model.outer.buffer: 'outer.buffer'}
        if mixed_precision and (not ignore_inner):
            buffer_to_buffer_name.pop(model.inner.buffer)
        fsdp_model = FSDP(model, ignored_modules=ignored_modules, mixed_precision=MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16) if mixed_precision else None)
        prefix_str = 'foo.' if prefix else ''
        with FSDP.state_dict_type(fsdp_model, STATE_DICT_MAPPING[state_dict_type]):
            sd1 = _gather_state_dict(fsdp_model.state_dict(prefix=prefix_str))
        with FSDP.summon_full_params(fsdp_model):
            fsdp_params = deepcopy(list(fsdp_model.parameters()))
        for (tensor, tensor_name) in {**ignored_tensor_to_tensor_name, **buffer_to_buffer_name}.items():
            prefixed_tensor_name = f'{prefix_str}{tensor_name}'
            self.assertTrue(prefixed_tensor_name in sd1)
            self.assertEqual(tensor.data_ptr(), sd1[prefixed_tensor_name].data_ptr(), f'{prefixed_tensor_name}')
        for buffer_name in buffer_to_buffer_name.values():
            prefixed_buffer_name = f'{prefix_str}{buffer_name}'
            self.assertTrue(prefixed_buffer_name in sd1)
            self.assertEqual(sd1[prefixed_buffer_name].dtype, torch.float32)
        nonwrapped_model = Model(wrap_fsdp=False, register_buffers=True).cuda()
        for param in nonwrapped_model.parameters():
            with torch.no_grad():
                param.zero_()
        to_load = {k[len(prefix_str):]: v for (k, v) in sd1.items()}
        nonwrapped_model.load_state_dict(to_load, strict=True)
        local_params = list(nonwrapped_model.parameters())
        for (fsdp_param, local_param) in zip(fsdp_params, local_params):
            self.assertEqual(fsdp_param, local_param)
        with FSDP.state_dict_type(fsdp_model, STATE_DICT_MAPPING[state_dict_type]):
            sd2 = fsdp_model.state_dict(prefix=prefix_str)
        for (tensor, tensor_name) in {**ignored_tensor_to_tensor_name, **buffer_to_buffer_name}.items():
            prefixed_tensor_name = f'{prefix_str}{tensor_name}'
            self.assertTrue(prefixed_tensor_name in sd2)
            self.assertEqual(tensor.data_ptr(), sd2[prefixed_tensor_name].data_ptr())
            self.assertEqual(sd1[prefixed_tensor_name].data_ptr(), sd2[prefixed_tensor_name].data_ptr())

    @skip_if_lt_x_gpu(2)
    def test_state_dict_type(self):
        if False:
            for i in range(10):
                print('nop')
        module = SkipModel(double_nest=True)
        with enable_wrap(wrapper_cls=FSDP):
            fsdp = wrap(module)
        with FSDP.state_dict_type(fsdp, StateDictType.LOCAL_STATE_DICT):
            pass
        for module in FSDP.fsdp_modules(fsdp):
            self.assertEqual(module._state_dict_type, StateDictType.FULL_STATE_DICT)

    @skip_if_lt_x_gpu(2)
    def test_local_state_dict_with_empty_ranks(self):
        if False:
            i = 10
            return i + 15

        class Model(Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.my_tensor = torch.full((1,), 3.1415926)
                self.my_parameter = nn.Parameter(self.my_tensor)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.my_parameter
        model = FSDP(Model().cuda())
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            out = model(None)
            out.backward()
            state_dict = deepcopy(model.state_dict())
            with torch.no_grad():
                with FSDP.summon_full_params(model):
                    self.assertEqual(model.my_parameter.item(), 3.1415926)
                    model.my_parameter.copy_(torch.full((1,), 1.75).cuda())
                    self.assertEqual(model.my_parameter.item(), 1.75)
            model.load_state_dict(state_dict)
            with FSDP.summon_full_params(model):
                self.assertEqual(model.my_parameter.item(), 3.1415926)

    @skip_if_lt_x_gpu(2)
    def test_torch_save_load(self):
        if False:
            for i in range(10):
                print('nop')
        model = Model(wrap_fsdp=True).cuda()
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            state_dict = model.state_dict()
            checkpoint = io.BytesIO()
            torch.save(state_dict, checkpoint)
            checkpoint.seek(0)
            state_dict_saved = torch.load(checkpoint)
            for (k, v) in state_dict_saved.items():
                if isinstance(v, ShardedTensor):
                    self.assertEqual(v._local_shards[0].tensor, state_dict[k]._local_shards[0].tensor)
                else:
                    self.assertEqual(v, state_dict[k])

    @skip_if_lt_x_gpu(2)
    def test_shared_module_and_shared_parameter(self):
        if False:
            for i in range(10):
                print('nop')
        model = FSDP(TestDummyModel().cuda())
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            state_dict = model.state_dict()
            self.assertEqual(state_dict['random_parameter'], state_dict['shared_parameter'])
            self.assertEqual(state_dict['net2.0.bias'], state_dict['net3.0.bias'])
            self.assertEqual(state_dict['net2.0.weight'], state_dict['net3.0.weight'])

    @skip_if_lt_x_gpu(2)
    def test_sharded_load_multi_backend_pg(self):
        if False:
            return 10
        auto_wrap_policy = ModuleWrapPolicy({TransformerEncoderLayer, TransformerDecoderLayer})
        fsdp_kwargs = {'auto_wrap_policy': auto_wrap_policy, 'use_orig_params': True}
        for load_cpu in [True, False]:
            with self.subTest(load_cpu=load_cpu):
                pg = dist.new_group(backend='cpu:gloo,cuda:nccl')
                fsdp_model = TransformerWithSharedParams.init(pg, FSDPInitMode.RECURSIVE, CUDAInitMode.CUDA_BEFORE, fsdp_kwargs)
                FSDP.set_state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT)
                sharded = fsdp_model.state_dict()
                param_copy = [t.clone().detach_() for t in fsdp_model.parameters()]
                with torch.no_grad():
                    for p in fsdp_model.parameters():
                        p.zero_()
                if load_cpu:
                    for (k, v) in sharded.items():
                        sharded[k] = v.cpu()
                fsdp_model.load_state_dict(sharded)
                for (p1, p2) in zip(param_copy, fsdp_model.parameters()):
                    self.assertEqual(p1, p2, f'not equal: {p1.sum()} vs {p2.sum()}')

    @skip_if_lt_x_gpu(2)
    def test_world_size_one(self):
        if False:
            i = 10
            return i + 15
        my_pg = None
        for i in range(self.world_size):
            pg = dist.new_group(ranks=[i])
            if i == self.rank:
                my_pg = pg
        model = TransformerWithSharedParams.init(my_pg, FSDPInitMode.RECURSIVE, CUDAInitMode.CUDA_BEFORE)
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            state_dict = model.state_dict()
            model.load_state_dict(state_dict)
        dist.barrier()

class TestFSDPStateDict4GPUs(FSDPTest):

    @property
    def world_size(self):
        if False:
            return 10
        return max(torch.cuda.device_count(), 2)

    @skip_if_lt_x_gpu(4)
    def test_local_state_dict_reshard(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This test demonstrates the ability to do resharding when using\n        local_state_dict. Although we do not recommend users to use\n        local_state_dict, there are still some corner cases that\n        using local_state_dict is a better solution.\n        '
        model = FSDP(Model(wrap_fsdp=True)).cuda()
        optim = torch.optim.SGD(model.parameters(), lr=0.1)
        batch = torch.randn(4, 4, device=torch.cuda.current_device())
        output = model(batch)
        loss = output.sum()
        loss.backward()
        optim.step()
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            state_dict = model.state_dict()
        rank = dist.get_rank()
        new_pg = dist.new_group(ranks=[0, 1])
        resharded_state_dict = {}
        for (key, value) in state_dict.items():
            if isinstance(value, ShardedTensor):
                full_flat_param = _all_gather_sharded_tensor(value)
                if rank < 2:
                    full_numel = full_flat_param.size()
                    chunks = full_flat_param.chunk(2)
                    flat_param = chunks[rank]
                    shard_offset = 0 if rank == 0 else chunks[0].numel()
                    local_shards = [Shard.from_tensor_and_offsets(flat_param, [shard_offset], rank)]
                    sharded_tensor = init_from_local_shards(local_shards, full_numel, process_group=new_pg)
                    resharded_state_dict[key] = sharded_tensor
            elif rank < 2:
                resharded_state_dict[key] = value
        if rank < 2:
            model2 = FSDP(Model(wrap_fsdp=True, process_group=new_pg), process_group=new_pg).cuda()
            with FSDP.state_dict_type(model2, StateDictType.LOCAL_STATE_DICT):
                model2.load_state_dict(resharded_state_dict)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            full_state_dict1 = model.state_dict()
        if rank < 2:
            with FSDP.state_dict_type(model2, StateDictType.FULL_STATE_DICT):
                full_state_dict2 = model2.state_dict()
            self.assertEqual(full_state_dict1, full_state_dict2)
instantiate_parametrized_tests(TestFSDPStateDict)
if __name__ == '__main__':
    run_tests()