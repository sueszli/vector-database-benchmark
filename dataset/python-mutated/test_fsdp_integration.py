import os
from copy import deepcopy
from pathlib import Path
from unittest import mock
import pytest
import torch
from lightning.fabric import Fabric
from lightning.fabric.plugins import FSDPPrecision
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0, _TORCH_GREATER_EQUAL_2_1
from lightning.fabric.wrappers import _FabricOptimizer
from torch.distributed.fsdp import FlatParameter, FullyShardedDataParallel, OptimStateKeyType
from torch.distributed.fsdp.wrap import always_wrap_policy, wrap
from torch.nn import Parameter
from tests_fabric.helpers.models import BoringFabric
from tests_fabric.helpers.runif import RunIf
from tests_fabric.test_fabric import BoringModel

class _MyFabric(BoringFabric):

    def get_model(self):
        if False:
            return 10
        model = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
        self.num_wrapped = 4
        return model

    def step(self, model, batch):
        if False:
            return 10
        wrapped_layers = [m for m in model.modules() if isinstance(m, FullyShardedDataParallel)]
        assert len(wrapped_layers) == self.num_wrapped
        assert (self.num_wrapped == 4) == isinstance(model._forward_module, FullyShardedDataParallel)
        precision = self._precision
        assert isinstance(precision, FSDPPrecision)
        if precision.precision == '16-mixed':
            param_dtype = torch.float32
            reduce_dtype = buffer_dtype = torch.float16
        elif precision.precision == 'bf16-mixed':
            param_dtype = torch.float32
            reduce_dtype = buffer_dtype = torch.bfloat16
        elif precision.precision == '16-true':
            param_dtype = reduce_dtype = buffer_dtype = torch.float16
        elif precision.precision == 'bf16-true':
            param_dtype = reduce_dtype = buffer_dtype = torch.bfloat16
        else:
            raise ValueError(f'Unknown precision {precision.precision}')
        for layer in wrapped_layers:
            assert layer.mixed_precision.param_dtype == param_dtype
            assert layer.mixed_precision.reduce_dtype == reduce_dtype
            assert layer.mixed_precision.buffer_dtype == buffer_dtype
        output = model(batch)
        return torch.nn.functional.mse_loss(output, torch.ones_like(output))

class _MyFabricManualWrapping(_MyFabric):

    def get_model(self):
        if False:
            while True:
                i = 10
        model = super().get_model()
        for (i, layer) in enumerate(model):
            if i % 2 == 0:
                model[i] = wrap(layer)
        self.num_wrapped = 2
        return model

@RunIf(min_cuda_gpus=2, standalone=True, min_torch='2.0.0')
@pytest.mark.parametrize('precision', ['16-mixed', pytest.param('bf16-mixed', marks=RunIf(bf16_cuda=True))])
@pytest.mark.parametrize('manual_wrapping', [True, False])
def test_fsdp_train_save_load(tmp_path, manual_wrapping, precision):
    if False:
        print('Hello World!')
    'Test FSDP training, saving and loading with different wrapping and precision settings.'
    fabric_cls = _MyFabricManualWrapping if manual_wrapping else _MyFabric
    fabric = fabric_cls(accelerator='cuda', strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy), devices=2, precision=precision)
    fabric.run()
    checkpoint_path = fabric.broadcast(str(tmp_path / 'fsdp-checkpoint'))
    params_before = deepcopy(list(fabric.model.parameters()))
    state = {'model': fabric.model, 'optimizer': fabric.optimizer, 'steps': 1}
    fabric.save(checkpoint_path, state)
    assert set(os.listdir(checkpoint_path)) == {'meta.pt', '.metadata', '__0_0.distcp', '__1_0.distcp'}
    fabric = fabric_cls(accelerator='cuda', strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy), devices=2, precision=precision)
    fabric.run()
    state = {'model': fabric.model, 'optimizer': fabric.optimizer, 'steps': 0}
    metadata = fabric.load(checkpoint_path, state)
    for (p0, p1) in zip(params_before, fabric.model.parameters()):
        torch.testing.assert_close(p0, p1, atol=0, rtol=0, equal_nan=True)
    assert state['steps'] == 1
    assert not metadata
    state = {'model': fabric.model, 'coconut': 11}
    with pytest.raises(KeyError, match="The requested state contains a key 'coconut' that does not exist"):
        fabric.load(checkpoint_path, state)
    state = {'model': fabric.model, 'coconut': 11}
    fabric.load(checkpoint_path, state, strict=False)
    assert state['coconut'] == 11

@RunIf(min_cuda_gpus=2, standalone=True, min_torch='2.0.0')
def test_fsdp_save_full_state_dict(tmp_path):
    if False:
        i = 10
        return i + 15
    'Test that FSDP saves the full state into a single file with `state_dict_type="full"`.'
    fabric = BoringFabric(accelerator='cuda', strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy, state_dict_type='full'), devices=2)
    fabric.run()
    checkpoint_path = Path(fabric.broadcast(str(tmp_path / 'fsdp-checkpoint.pt')))
    state = {'model': fabric.model, 'optimizer': fabric.optimizer, 'steps': 1}
    fabric.save(checkpoint_path, state)
    checkpoint = torch.load(checkpoint_path)
    assert checkpoint['steps'] == 1
    loaded_state_dict = checkpoint['model']
    with FullyShardedDataParallel.summon_full_params(fabric.model):
        state_dict = fabric.model.state_dict()
        assert set(loaded_state_dict.keys()) == set(state_dict.keys())
        for param_name in state_dict:
            assert torch.equal(loaded_state_dict[param_name], state_dict[param_name].cpu())
        params_before = [p.cpu() for p in fabric.model.parameters()]
    optimizer_state_before = FullyShardedDataParallel.full_optim_state_dict(fabric.model, fabric.optimizer, rank0_only=False)
    assert set(checkpoint['optimizer'].keys()) == set(optimizer_state_before.keys()) == {'state', 'param_groups'}
    fabric = BoringFabric(accelerator='cuda', strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy), devices=2)
    fabric.run()
    metadata = fabric.load(checkpoint_path, {'model': fabric.model, 'optimizer': fabric.optimizer})
    assert metadata == {'steps': 1}
    with FullyShardedDataParallel.summon_full_params(fabric.model):
        params_after = list(fabric.model.parameters())
        assert all((torch.equal(p0.cpu(), p1.cpu()) for (p0, p1) in zip(params_before, params_after)))
    optimizer_state_after = FullyShardedDataParallel.full_optim_state_dict(fabric.model, fabric.optimizer, rank0_only=False)
    assert set(optimizer_state_after.keys()) == set(optimizer_state_before.keys()) == {'state', 'param_groups'}
    torch.testing.assert_close(optimizer_state_after['state'], optimizer_state_before['state'], atol=0, rtol=0)
    assert optimizer_state_after['param_groups'] == optimizer_state_before['param_groups']
    fabric.run()
    fabric = BoringFabric(accelerator='cpu', devices=1)
    fabric.run()
    metadata = fabric.load(checkpoint_path, {'model': fabric.model, 'optimizer': fabric.optimizer})
    assert metadata == {'steps': 1}
    params_after = list(fabric.model.parameters())
    assert all((torch.equal(p0, p1) for (p0, p1) in zip(params_before, params_after)))
    normal_checkpoint_path = Path(fabric.broadcast(str(tmp_path / 'normal-checkpoint.pt')))
    fabric.save(normal_checkpoint_path, {'model': fabric.model, 'optimizer': fabric.optimizer, 'steps': 2})
    optimizer_state_after = torch.load(normal_checkpoint_path)['optimizer']
    optimizer_state_after = FullyShardedDataParallel.rekey_optim_state_dict(optimizer_state_after, optim_state_key_type=OptimStateKeyType.PARAM_NAME, model=fabric.model)
    assert set(optimizer_state_after.keys()) == set(optimizer_state_before.keys()) == {'state', 'param_groups'}
    torch.testing.assert_close(optimizer_state_after['state'], optimizer_state_before['state'], atol=0, rtol=0)
    fabric.run()
    fabric = BoringFabric(accelerator='cuda', strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy), devices=2)
    fabric.run()
    metadata = fabric.load(normal_checkpoint_path, {'model': fabric.model, 'optimizer': fabric.optimizer})
    assert metadata == {'steps': 2}
    with FullyShardedDataParallel.summon_full_params(fabric.model):
        params_after = list(fabric.model.parameters())
        assert all((torch.equal(p0.cpu(), p1.cpu()) for (p0, p1) in zip(params_before, params_after)))
    optimizer_state_after = FullyShardedDataParallel.full_optim_state_dict(fabric.model, fabric.optimizer, rank0_only=False)
    assert set(optimizer_state_after.keys()) == set(optimizer_state_before.keys()) == {'state', 'param_groups'}
    torch.testing.assert_close(optimizer_state_after['state'], optimizer_state_before['state'], atol=0, rtol=0)
    assert optimizer_state_after['param_groups'] == optimizer_state_before['param_groups']
    fabric.run()

@RunIf(min_cuda_gpus=2, standalone=True, min_torch='2.0.0')
def test_fsdp_load_full_state_dict_into_sharded_model(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    'Test that the strategy can load a full-state checkpoint into a FSDP sharded model.'
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    fabric = BoringFabric(accelerator='cuda', devices=1)
    fabric.seed_everything(0)
    fabric.run()
    checkpoint_path = Path(fabric.broadcast(str(tmp_path / 'full-checkpoint.pt')))
    state = {'model': fabric.model, 'optimizer': fabric.optimizer, 'steps': 1}
    fabric.save(checkpoint_path, state)
    with FSDP.summon_full_params(fabric.model, writeback=False, rank0_only=False):
        params_before = torch.cat([p.cpu().view(-1) for p in fabric.model.parameters()])
    fabric = BoringFabric(accelerator='cuda', strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy), devices=2)
    fabric.run()
    state = {'model': fabric.model, 'optimizer': fabric.optimizer, 'steps': 44}
    fabric.load(checkpoint_path, state)
    assert state['steps'] == 1
    with FSDP.summon_full_params(fabric.model, writeback=False, rank0_only=False):
        params_after = torch.cat([p.cpu().view(-1) for p in fabric.model.parameters()])
    assert torch.equal(params_before, params_after)
    raw_checkpoint_path = checkpoint_path.with_name('model-state-dict')
    if fabric.global_rank == 0:
        checkpoint = torch.load(checkpoint_path)
        torch.save(checkpoint['model'], raw_checkpoint_path)
    fabric.barrier()
    fabric.run()
    fabric.load_raw(raw_checkpoint_path, fabric.model)
    with FSDP.summon_full_params(fabric.model, writeback=False, rank0_only=False):
        params_after = torch.cat([p.cpu().view(-1) for p in fabric.model.parameters()])
    assert torch.equal(params_before, params_after)

@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
@pytest.mark.parametrize('move_to_device', [True, False])
@mock.patch('lightning.fabric.wrappers._FabricModule')
def test_setup_module_move_to_device(fabric_module_mock, move_to_device):
    if False:
        return 10
    'Test that `move_to_device` does nothing, FSDP decides which device parameters get moved to which device\n    (sharding).'
    strategy = FSDPStrategy(auto_wrap_policy=always_wrap_policy)
    fabric = Fabric(accelerator='cuda', devices=2, strategy=strategy)
    fabric.launch()
    model = torch.nn.Linear(10, 10, bias=False)
    fabric_model = fabric.setup_module(model, move_to_device=move_to_device)
    fabric_module_mock.assert_not_called()
    assert len(list(fabric_model.parameters())) == 1
    assert next(fabric_model.parameters()).device == torch.device('cuda', fabric.local_rank)
    assert next(fabric_model.parameters()).numel() == 50
    if _TORCH_GREATER_EQUAL_2_0:
        assert isinstance(next(fabric_model.parameters()), Parameter)
    else:
        assert isinstance(next(fabric_model.parameters()), FlatParameter)
    assert fabric_model.device == torch.device('cpu')
    assert fabric.device == torch.device('cuda', fabric.local_rank)

@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, min_torch='2.0.0')
def test_setup_with_orig_params_and_multiple_param_groups():
    if False:
        while True:
            i = 10
    'Test that Fabric sets `use_orig_params` for the user when jointly setting up model and optimizer.'
    strategy = FSDPStrategy(auto_wrap_policy=always_wrap_policy)
    fabric = Fabric(accelerator='cuda', devices=2, strategy=strategy)
    fabric.launch()
    model = torch.nn.Sequential(torch.nn.Linear(10, 10, bias=False), torch.nn.Linear(5, 2, bias=False))
    optimizer = torch.optim.Adam([{'params': model[0].parameters(), 'lr': 0.01}, {'params': model[1].parameters(), 'lr': 1e-06}])
    (wrapped_model, wrapped_optimizer) = fabric.setup(model, optimizer)
    assert fabric.strategy._fsdp_kwargs['use_orig_params']
    assert isinstance(wrapped_optimizer, _FabricOptimizer)
    assert len(wrapped_optimizer.param_groups) == 2
    for i in range(2):
        layer = wrapped_model._forward_module.module[i]
        assert isinstance(layer, FullyShardedDataParallel)
        assert torch.equal(wrapped_optimizer.param_groups[i]['params'][0], layer.weight)
        assert isinstance(layer.weight, torch.nn.Parameter)
        assert not isinstance(layer.weight, FlatParameter)

@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, dynamo=True)
@mock.patch.dict(os.environ, {})
@pytest.mark.parametrize('compile_after_setup', [False, pytest.param(True, marks=RunIf(min_python='3.9'))])
def test_compile(compile_after_setup):
    if False:
        for i in range(10):
            print('nop')
    'Test that the model can be compiled before and after the model is wrapped in FSDP.'
    model = BoringModel()
    strategy = FSDPStrategy(auto_wrap_policy=always_wrap_policy)
    fabric = Fabric(accelerator='cuda', devices=2, strategy=strategy)
    fabric.launch()
    if not compile_after_setup:
        model = torch.compile(model)
    model = fabric.setup(model)
    if compile_after_setup:
        model = torch.compile(model)
    for _ in range(3):
        model(torch.rand(2, 32, device=fabric.device)).sum().backward()

@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
@pytest.mark.parametrize(('precision', 'expected_dtype'), [('32-true', torch.float32), ('16-true', torch.float16), pytest.param('bf16-true', torch.bfloat16, marks=RunIf(bf16_cuda=True))])
def test_module_init_context(precision, expected_dtype):
    if False:
        for i in range(10):
            print('nop')
    'Test that the module under the init-context gets moved to the right device and dtype.'
    fabric = Fabric(accelerator='cuda', devices=2, strategy=FSDPStrategy(auto_wrap_policy=always_wrap_policy), precision=precision)
    fabric.launch()

    def _run_setup_assertions(empty_init, expected_device):
        if False:
            while True:
                i = 10
        with fabric.init_module(empty_init=empty_init):
            model = torch.nn.Linear(100, 100, bias=False)
        assert model.weight.device == expected_device
        assert model.weight.dtype == expected_dtype
        model = fabric.setup(model)
        assert model.weight.device == torch.device('cuda', fabric.local_rank)
        assert model.weight.dtype == expected_dtype
    _run_setup_assertions(empty_init=False, expected_device=torch.device('cpu'))
    if _TORCH_GREATER_EQUAL_2_1:
        _run_setup_assertions(empty_init=True, expected_device=torch.device('meta'))
    else:
        _run_setup_assertions(empty_init=True, expected_device=torch.device('cpu'))

@RunIf(min_cuda_gpus=2, standalone=True, min_torch='2.0.0')
def test_fsdp_save_filter(tmp_path):
    if False:
        i = 10
        return i + 15
    fabric = BoringFabric(accelerator='cuda', strategy=FSDPStrategy(state_dict_type='full'), devices=2)
    fabric.launch()
    model = fabric.get_model()
    model = fabric.setup_module(model)
    tmp_path = Path(fabric.broadcast(str(tmp_path)))
    state = {'model': model}
    filter = {'model': lambda k, v: 'bias' in k}
    checkpoint_path = tmp_path / 'full.pth'
    fabric.save(checkpoint_path, state, filter=filter)
    checkpoint = torch.load(checkpoint_path)['model']
    assert set(checkpoint) == {'bias'}
    assert isinstance(checkpoint['bias'], torch.Tensor)
    fabric.strategy._state_dict_type = 'sharded'
    checkpoint_path = tmp_path / 'sharded'
    with pytest.raises(NotImplementedError, match="doesn't support loading sharded filtered"):
        fabric.save(checkpoint_path, state, filter=filter)

@RunIf(min_torch='1.13', min_cuda_gpus=1)
def test_fsdp_manual_activation_checkpointing():
    if False:
        for i in range(10):
            print('nop')
    model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Linear(1, 1))
    strategy = FSDPStrategy(activation_checkpointing_policy={torch.nn.Linear})
    fabric = Fabric(devices=1, accelerator='cuda', strategy=strategy)
    fabric.launch()
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper, apply_activation_checkpointing
    apply_activation_checkpointing(model)
    wrappers = {name for (name, mod) in model.named_modules() if isinstance(mod, CheckpointWrapper)}
    assert wrappers == {'0', '1'}
    with pytest.warns(match='is configured, but the model already contains checkpointed'):
        model = fabric.setup(model)
    wrappers = {name for (name, mod) in model._forward_module.named_modules() if isinstance(mod, CheckpointWrapper)}
    assert wrappers == {'_fsdp_wrapped_module.0', '_fsdp_wrapped_module.1'}

@RunIf(min_cuda_gpus=1)
def test_rewrap_warnings():
    if False:
        while True:
            i = 10
    from torch.distributed.fsdp import FullyShardedDataParallel
    from torch.distributed.fsdp.wrap import wrap
    strategy = FSDPStrategy(auto_wrap_policy={torch.nn.Linear})
    fabric = Fabric(devices=1, accelerator='cuda', strategy=strategy)
    fabric.launch()
    with fabric.init_module():
        model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU(), wrap(torch.nn.Linear(1, 1)))
    with pytest.warns(match='the model is already wrapped'):
        model = fabric.setup(model)
    assert not isinstance(model._forward_module, FullyShardedDataParallel)
    assert isinstance(model._forward_module[2], FullyShardedDataParallel)
    if not _TORCH_GREATER_EQUAL_2_1:
        return
    with fabric.init_module(empty_init=True):
        model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU(), wrap(torch.nn.Linear(1, 1)))
    assert model[0].weight.is_meta
    with pytest.warns(match='there are still parameters on the meta device'):
        fabric_model = fabric.setup(model)
    assert next(fabric_model.parameters()).is_meta