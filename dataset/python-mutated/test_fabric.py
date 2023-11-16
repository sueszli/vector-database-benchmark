import os
import sys
from re import escape
from unittest import mock
from unittest.mock import ANY, MagicMock, Mock, PropertyMock, call
import pytest
import torch
import torch.distributed
import torch.nn.functional
from lightning.fabric.accelerators.xla import _using_pjrt
from lightning.fabric.fabric import Fabric
from lightning.fabric.plugins import Precision
from lightning.fabric.strategies import DataParallelStrategy, DDPStrategy, DeepSpeedStrategy, ParallelStrategy, SingleDeviceStrategy, Strategy, XLAStrategy
from lightning.fabric.strategies.strategy import _Sharded
from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.fabric.utilities.seed import pl_worker_init_function, seed_everything
from lightning.fabric.utilities.warnings import PossibleUserWarning
from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule, _FabricOptimizer
from lightning_utilities.test.warning import no_warning_call
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Sampler, SequentialSampler, TensorDataset
from tests_fabric.helpers.runif import RunIf

class BoringModel(nn.Module):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.layer = torch.nn.Linear(32, 2, bias=False)

    def forward(self, x):
        if False:
            print('Hello World!')
        x = self.layer(x)
        return torch.nn.functional.mse_loss(x, torch.ones_like(x))

def test_run_input_output():
    if False:
        return 10
    'Test that the dynamically patched run() method receives the input arguments and returns the result.'

    class RunFabric(Fabric):
        run_args = ()
        run_kwargs = {}

        def run(self, *args, **kwargs):
            if False:
                return 10
            self.run_args = args
            self.run_kwargs = kwargs
            return 'result'
    fabric = RunFabric()
    result = fabric.run(1, 2, three=3)
    assert result == 'result'
    assert fabric.run_args == (1, 2)
    assert fabric.run_kwargs == {'three': 3}

@mock.patch('lightning.fabric.strategies.ddp.DistributedDataParallel')
@pytest.mark.parametrize('setup_method', ['setup', 'setup_module'])
def test_setup_module(ddp_mock, setup_method):
    if False:
        for i in range(10):
            print('nop')
    'Test that the setup method lets the strategy wrap the model, but keeps a reference to the original model.'
    fabric = Fabric(accelerator='cpu', strategy='ddp', devices=2)
    fabric._launched = True
    model = nn.Linear(1, 2)
    setup_method = getattr(fabric, setup_method)
    fabric_model = setup_method(model)
    ddp_mock.assert_called_with(module=model, device_ids=ANY)
    assert fabric_model.module == model
    assert fabric_model.weight is model.weight
    assert fabric_model.forward != model.forward

@RunIf(skip_windows=True, dynamo=True)
@pytest.mark.parametrize('setup_method', ['setup', 'setup_module'])
def test_setup_compiled_module(setup_method):
    if False:
        i = 10
        return i + 15
    'Test that an `OptimizedModule` can be passed to the setup method.'
    from torch._dynamo.eval_frame import OptimizedModule
    fabric = Fabric(devices=1)
    model = nn.Linear(1, 2)
    compiled_model = torch.compile(model)
    assert isinstance(compiled_model, OptimizedModule)
    setup_method = getattr(fabric, setup_method)
    fabric_model = setup_method(compiled_model)
    assert fabric_model.module == compiled_model
    assert fabric_model.weight is model.weight

@pytest.mark.parametrize(('accelerator', 'initial_device', 'target_device'), [('cpu', 'cpu', 'cpu'), pytest.param('cpu', 'cuda:0', 'cpu', marks=RunIf(min_cuda_gpus=1)), pytest.param('cpu', 'mps:0', 'cpu', marks=RunIf(mps=True)), pytest.param('cuda', 'cpu', 'cuda:0', marks=RunIf(min_cuda_gpus=1)), pytest.param('cuda', 'cuda:1', 'cuda:0', marks=RunIf(min_cuda_gpus=2)), pytest.param('mps', 'cpu', 'mps:0', marks=RunIf(mps=True))])
@pytest.mark.parametrize('move_to_device', [True, False])
@pytest.mark.parametrize('setup_method', ['setup', 'setup_module'])
def test_setup_module_move_to_device(setup_method, move_to_device, accelerator, initial_device, target_device):
    if False:
        for i in range(10):
            print('nop')
    'Test that `move_to_device` leads to parameters being moved to the correct device and that the device attributes\n    on the wrapper are updated.'
    initial_device = torch.device(initial_device)
    target_device = torch.device(target_device)
    expected_device = target_device if move_to_device else initial_device
    fabric = Fabric(accelerator=accelerator, devices=1)
    model = nn.Linear(1, 2)
    model.to(initial_device)
    setup_method = getattr(fabric, setup_method)
    fabric_model = setup_method(model, move_to_device=move_to_device)
    assert all((param.device == expected_device for param in model.parameters()))
    assert all((param.device == expected_device for param in fabric_model.parameters()))
    assert fabric_model.device == expected_device
    assert fabric.device == target_device
    model = nn.Sequential()
    fabric_model = setup_method(model, move_to_device=move_to_device)
    assert fabric_model.device == target_device if move_to_device else torch.device('cpu')

@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize('move_to_device', [True, False])
@pytest.mark.parametrize('setup_method', ['setup', 'setup_module'])
def test_setup_module_parameters_on_different_devices(setup_method, move_to_device):
    if False:
        for i in range(10):
            print('nop')
    'Test that a warning is emitted when model parameters are on a different device prior to calling `setup()`.'
    device0 = torch.device('cpu')
    device1 = torch.device('cuda', 0)
    fabric = Fabric(accelerator='cuda', devices=1)
    module0 = nn.Linear(1, 2).to(device0)
    module1 = nn.Linear(1, 2).to(device1)
    model = nn.Sequential(module0, module1)
    setup_method = getattr(fabric, setup_method)
    match = "has 2 parameters on different devices \\(for example '1.weight' on cuda:0 and '0.weight' on cpu\\)"
    if move_to_device:
        with pytest.warns(PossibleUserWarning, match=match):
            fabric_model = setup_method(model, move_to_device=move_to_device)
        assert fabric_model.device == device1
        assert module0.weight.device == module0.bias.device == device1
        assert module1.weight.device == module1.bias.device == device1
    else:
        with no_warning_call(expected_warning=PossibleUserWarning, match=match):
            setup_method(model, move_to_device=move_to_device)

def test_setup_module_and_optimizers():
    if False:
        for i in range(10):
            print('nop')
    'Test that `setup()` can handle no optimizers, one optimizer, or multiple optimizers.'
    fabric = Fabric(devices=1)
    model = nn.Linear(1, 2)
    optimizer0 = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer1 = torch.optim.Adam(model.parameters(), lr=0.1)
    fabric_model = fabric.setup(model)
    assert isinstance(fabric_model, _FabricModule)
    assert fabric_model.module is model
    (fabric_model, fabric_optimizer) = fabric.setup(model, optimizer0)
    assert isinstance(fabric_model, _FabricModule)
    assert isinstance(fabric_optimizer, _FabricOptimizer)
    assert fabric_model.module is model
    assert fabric_optimizer.optimizer is optimizer0
    (fabric_model, fabric_optimizer0, fabric_optimizer1) = fabric.setup(model, optimizer0, optimizer1)
    assert isinstance(fabric_model, _FabricModule)
    assert isinstance(fabric_optimizer0, _FabricOptimizer)
    assert isinstance(fabric_optimizer1, _FabricOptimizer)
    assert fabric_model.module is model
    assert fabric_optimizer0.optimizer is optimizer0
    assert fabric_optimizer1.optimizer is optimizer1

def test_setup_optimizers():
    if False:
        print('Hello World!')
    'Test that `setup_optimizers()` can handle one or more optimizers.'
    fabric = Fabric()
    model = nn.Linear(1, 2)
    optimizer0 = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer1 = torch.optim.Adam(model.parameters(), lr=0.1)
    fabric_optimizer = fabric.setup_optimizers(optimizer0)
    assert isinstance(fabric_optimizer, _FabricOptimizer)
    assert fabric_optimizer.optimizer is optimizer0
    (fabric_optimizer0, fabric_optimizer1) = fabric.setup_optimizers(optimizer0, optimizer1)
    assert isinstance(fabric_optimizer0, _FabricOptimizer)
    assert isinstance(fabric_optimizer1, _FabricOptimizer)
    assert fabric_optimizer0.optimizer is optimizer0
    assert fabric_optimizer1.optimizer is optimizer1

def test_setup_twice_fails():
    if False:
        i = 10
        return i + 15
    'Test that calling `setup` with a model or optimizer that is already wrapped fails.'
    fabric = Fabric(devices=1)
    model = nn.Linear(1, 2)
    optimizer = torch.optim.Adam(model.parameters())
    (fabric_model, fabric_optimizer) = fabric.setup(model, optimizer)
    with pytest.raises(ValueError, match='A model should be passed only once to the'):
        fabric.setup(fabric_model, optimizer)
    (fabric_model, fabric_optimizer) = fabric.setup(model, optimizer)
    with pytest.raises(ValueError, match='An optimizer should be passed only once to the'):
        fabric.setup(model, fabric_optimizer)

def test_setup_module_twice_fails():
    if False:
        print('Hello World!')
    'Test that calling `setup_module` with a model that is already wrapped fails.'
    fabric = Fabric(devices=1)
    model = nn.Linear(1, 2)
    fabric_model = fabric.setup_module(model)
    with pytest.raises(ValueError, match='A model should be passed only once to the'):
        fabric.setup_module(fabric_model)

def test_setup_optimizers_twice_fails():
    if False:
        while True:
            i = 10
    'Test that calling `setup_module` with a model that is already wrapped fails.'
    fabric = Fabric()
    model = nn.Linear(1, 2)
    optimizer = torch.optim.Adam(model.parameters())
    fabric_optimizer = fabric.setup_optimizers(optimizer)
    with pytest.raises(ValueError, match='An optimizer should be passed only once to'):
        fabric.setup_optimizers(fabric_optimizer)

@pytest.mark.parametrize('strategy_cls', [DeepSpeedStrategy, XLAStrategy])
def test_setup_optimizers_not_supported(strategy_cls):
    if False:
        return 10
    'Test that `setup_optimizers` validates the strategy supports setting up model and optimizers independently.'
    fabric = Fabric()
    fabric._launched = True
    model = nn.Linear(1, 2)
    optimizer = torch.optim.Adam(model.parameters())
    fabric._strategy = Mock(spec=strategy_cls)
    with pytest.raises(RuntimeError, match=escape('requires the model and optimizer(s) to be set up jointly through')):
        fabric.setup_optimizers(optimizer)

@RunIf(min_cuda_gpus=1, min_torch='2.1')
def test_setup_optimizer_on_meta_device():
    if False:
        i = 10
        return i + 15
    "Test that the setup-methods validate that the optimizer doesn't have references to meta-device parameters."
    fabric = Fabric(strategy='fsdp', devices=1)
    fabric._launched = True
    with fabric.init_module(empty_init=True):
        model = nn.Linear(1, 2)
    assert model.weight.is_meta
    optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(RuntimeError, match="The optimizer has references to the model's meta-device parameters"):
        fabric.setup(model, optimizer)
    with pytest.raises(RuntimeError, match="The optimizer has references to the model's meta-device parameters"):
        fabric.setup_optimizers(optimizer)

def test_setup_tracks_num_models():
    if False:
        for i in range(10):
            print('nop')
    'Test that setup() tracks how many times it has setup a model.'
    fabric = Fabric(devices=1)
    model = nn.Linear(1, 2)
    optimizer = torch.optim.Adam(model.parameters())
    assert fabric._models_setup == 0
    fabric.setup(model, optimizer)
    assert fabric._models_setup == 1
    fabric.setup(model, optimizer)
    assert fabric._models_setup == 2
    fabric.setup_module(model)
    assert fabric._models_setup == 3

def test_setup_dataloaders_unsupported_input():
    if False:
        while True:
            i = 10
    'Test that the setup_dataloaders method fails when provided with non-DataLoader objects.'
    fabric = Fabric()
    with pytest.raises(ValueError, match='`setup_dataloaders` requires at least one dataloader'):
        fabric.setup_dataloaders()
    with pytest.raises(TypeError, match='Only PyTorch DataLoader are currently supported'):
        fabric.setup_dataloaders(range(2))

def test_setup_dataloaders_return_type():
    if False:
        i = 10
        return i + 15
    'Test that the setup method returns the dataloaders wrapped as FabricDataLoader and in the right order.'
    fabric = Fabric(devices=1)
    fabric_dataloader = fabric.setup_dataloaders(DataLoader(range(2)))
    assert isinstance(fabric_dataloader, _FabricDataLoader)
    dataset0 = Mock()
    dataset1 = Mock()
    dataloader0 = DataLoader(dataset0)
    dataloader1 = DataLoader(dataset1)
    (fabric_dataloader0, fabric_dataloader1) = fabric.setup_dataloaders(dataloader0, dataloader1)
    assert isinstance(fabric_dataloader0, _FabricDataLoader)
    assert isinstance(fabric_dataloader1, _FabricDataLoader)
    assert fabric_dataloader0.dataset is dataset0
    assert fabric_dataloader1.dataset is dataset1

@mock.patch('lightning.fabric.fabric._replace_dunder_methods')
def test_setup_dataloaders_captures_dataloader_arguments(ctx_manager):
    if False:
        print('Hello World!')
    'Test that Fabric intercepts the DataLoader constructor arguments with a context manager when launching a\n    function.'

    def run(_):
        if False:
            return 10
        assert ctx_manager().__enter__.call_count == 2
    fabric = Fabric()
    fabric.launch(run)
    assert ctx_manager().__exit__.call_count == 2

def test_setup_dataloaders_raises_for_unknown_custom_args():
    if False:
        while True:
            i = 10
    "Test that an error raises when custom dataloaders with unknown arguments are created from outside Fabric's run\n    method."

    class CustomDataLoader(DataLoader):

        def __init__(self, new_arg, *args, **kwargs):
            if False:
                return 10
            super().__init__(range(5), *args, **kwargs)
    dataloader = CustomDataLoader(2, batch_size=2)
    fabric = Fabric(devices=1)
    fabric_dataloader = fabric.setup_dataloaders(dataloader)
    assert fabric_dataloader._dataloader is dataloader
    fabric = Fabric(devices=2, accelerator='cpu')
    fabric._launched = True
    with pytest.raises(MisconfigurationException, match="Trying to inject custom `Sampler` into the `CustomDataLoader` instance.*The missing attributes are \\['new_arg'\\]"):
        fabric.setup_dataloaders(dataloader)

def test_setup_dataloaders_twice_fails():
    if False:
        print('Hello World!')
    'Test that calling setup_dataloaders with a dataloader that is already wrapped fails.'
    fabric = Fabric()
    dataloader = DataLoader(range(2))
    fabric_dataloader = fabric.setup_dataloaders(dataloader)
    with pytest.raises(ValueError, match='A dataloader should be passed only once to the'):
        fabric.setup_dataloaders(fabric_dataloader)

@mock.patch('lightning.fabric.fabric.Fabric.device', new_callable=PropertyMock, return_value=torch.device('cuda', 1))
def test_setup_dataloaders_move_to_device(fabric_device_mock):
    if False:
        for i in range(10):
            print('nop')
    'Test that the setup configures FabricDataLoader to move the data to the device automatically.'
    fabric = Fabric(devices=1)
    fabric_dataloaders = fabric.setup_dataloaders(DataLoader(Mock()), DataLoader(Mock()), move_to_device=False)
    assert all((dl.device is None for dl in fabric_dataloaders))
    fabric_device_mock.assert_not_called()
    fabric = Fabric(devices=1)
    fabric_dataloaders = fabric.setup_dataloaders(DataLoader(Mock()), DataLoader(Mock()), move_to_device=True)
    assert all((dl.device == torch.device('cuda', 1) for dl in fabric_dataloaders))
    fabric_device_mock.assert_called()

def test_setup_dataloaders_distributed_sampler_not_needed():
    if False:
        return 10
    'Test that `use_distributed_sampler` option has no effect when no distributed sampler is needed.'
    custom_sampler = Mock(spec=Sampler)
    dataloader = DataLoader(Mock(), sampler=custom_sampler)
    fabric = Fabric(devices=1)
    fabric_dataloader = fabric.setup_dataloaders(dataloader, use_distributed_sampler=True)
    assert fabric_dataloader._dataloader is dataloader
    assert fabric_dataloader.sampler is custom_sampler

def test_setup_dataloaders_distributed_sampler_shuffle():
    if False:
        for i in range(10):
            print('nop')
    'Test that the DataLoader(shuffle=True|False) setting gets carried over correctly into the distributed\n    sampler.'
    fabric = Fabric(accelerator='cpu', strategy='ddp_spawn', devices=2)
    fabric._launched = True
    dataset = TensorDataset(torch.arange(8))
    no_shuffle_dataloaders = [DataLoader(dataset), DataLoader(dataset, shuffle=False), DataLoader(dataset, sampler=SequentialSampler(dataset))]
    for dataloader in no_shuffle_dataloaders:
        dataloader = fabric.setup_dataloaders(dataloader)
        assert [t[0].item() for t in iter(dataloader)] == [0, 2, 4, 6]
    shuffle_dataloaders = [DataLoader(dataset, shuffle=True), DataLoader(dataset, sampler=RandomSampler(dataset))]
    for dataloader in shuffle_dataloaders:
        seed_everything(1)
        dataloader = fabric.setup_dataloaders(dataloader)
        assert [t[0].item() for t in iter(dataloader)] == [5, 2, 7, 1]

@pytest.mark.parametrize('shuffle', [True, False])
@pytest.mark.parametrize('batch_size', [1, 2, 3])
def test_setup_dataloaders_distributed_sampler_parity(shuffle, batch_size):
    if False:
        print('Hello World!')
    'Test that the distributed sampler setup in Fabric leads to the same sequence of data as in raw PyTorch.'
    torch.manual_seed(1)
    fabric = Fabric(accelerator='cpu', strategy='ddp', devices=2)
    fabric._launched = True
    dataset = torch.arange(10)
    torch_dataloader = DataLoader(dataset, sampler=DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=shuffle), batch_size=batch_size)
    fabric_dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    fabric_dataloader = fabric.setup_dataloaders(fabric_dataloader)

    def fetch_epoch(loader):
        if False:
            print('Hello World!')
        iterator = iter(loader)
        return torch.cat((next(iterator), next(iterator)))
    torch_dataloader.sampler.set_epoch(0)
    torch_data = fetch_epoch(torch_dataloader)
    fabric_data = fetch_epoch(fabric_dataloader)
    assert torch.equal(torch_data, fabric_data)
    torch_dataloader.sampler.set_epoch(1)
    torch_data = fetch_epoch(torch_dataloader)
    fabric_data = fetch_epoch(fabric_dataloader)
    assert torch.equal(torch_data, fabric_data)
    assert torch_dataloader.sampler.epoch == 1
    assert fabric_dataloader._dataloader.sampler.epoch == 1

@mock.patch.dict(os.environ, {}, clear=True)
def test_seed_everything():
    if False:
        print('Hello World!')
    'Test that seed everything is static and sets the worker init function on the dataloader.'
    Fabric.seed_everything(3)
    fabric = Fabric(devices=1)
    fabric_dataloader = fabric.setup_dataloaders(DataLoader(Mock()))
    assert fabric_dataloader.worker_init_fn.func is pl_worker_init_function
    assert os.environ == {'PL_GLOBAL_SEED': '3', 'PL_SEED_WORKERS': '1'}

@pytest.mark.parametrize('strategy', ['dp', 'ddp', 'ddp_spawn', pytest.param('ddp_fork', marks=RunIf(skip_windows=True)), pytest.param('deepspeed', marks=RunIf(deepspeed=True))])
def test_setup_dataloaders_replace_custom_sampler(strategy):
    if False:
        print('Hello World!')
    'Test that asking to replace a custom sampler results in an error when a distributed sampler would be needed.'
    custom_sampler = Mock(spec=Sampler)
    dataloader = DataLoader(Mock(), sampler=custom_sampler)
    fabric = Fabric(accelerator='cpu', strategy=strategy, devices=2)
    fabric._launched = True
    if hasattr(fabric.strategy, 'distributed_sampler_kwargs'):
        with pytest.raises(TypeError, match='You seem to have configured a sampler in your DataLoader'):
            fabric.setup_dataloaders(dataloader, use_distributed_sampler=True)
    fabric_dataloader = fabric.setup_dataloaders(dataloader, use_distributed_sampler=False)
    assert fabric_dataloader.sampler is custom_sampler

@pytest.mark.parametrize('strategy', ['dp', 'ddp', 'ddp_spawn', pytest.param('ddp_fork', marks=RunIf(skip_windows=True)), pytest.param('deepspeed', marks=RunIf(deepspeed=True))])
@pytest.mark.parametrize('shuffle', [True, False])
def test_setup_dataloaders_replace_standard_sampler(shuffle, strategy):
    if False:
        for i in range(10):
            print('nop')
    'Test that Fabric replaces the default samplers with DistributedSampler automatically.'
    fabric = Fabric(accelerator='cpu', strategy=strategy, devices=2)
    fabric._launched = True
    is_distributed = hasattr(fabric.strategy, 'distributed_sampler_kwargs')
    fabric_dataloader = fabric.setup_dataloaders(DataLoader(range(3), shuffle=shuffle))
    assert not is_distributed or isinstance(fabric_dataloader.sampler, DistributedSampler)

@pytest.mark.parametrize(('accelerator', 'expected'), [('cpu', 'cpu'), pytest.param('cuda', 'cuda:0', marks=RunIf(min_cuda_gpus=1)), pytest.param('gpu', 'cuda:0', marks=RunIf(min_cuda_gpus=1)), pytest.param('tpu', 'xla:0', marks=RunIf(tpu=True, standalone=True)), pytest.param('mps', 'mps:0', marks=RunIf(mps=True)), pytest.param('gpu', 'mps:0', marks=RunIf(mps=True))])
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_to_device(accelerator, expected):
    if False:
        i = 10
        return i + 15
    'Test that the to_device method can move various objects to the device determined by the accelerator.'
    if accelerator == 'tpu' and (not _using_pjrt()):
        expected = 'xla:1'
    fabric = Fabric(accelerator=accelerator, devices=1)
    fabric.launch()
    expected_device = torch.device(expected)
    module = torch.nn.Linear(2, 3)
    module = fabric.to_device(module)
    assert all((param.device == expected_device for param in module.parameters()))
    tensor = torch.rand(2, 2)
    tensor = fabric.to_device(tensor)
    assert tensor.device == expected_device
    collection = {'data': torch.rand(2, 2), 'int': 1}
    collection = fabric.to_device(collection)
    assert collection['data'].device == expected_device

def test_rank_properties():
    if False:
        for i in range(10):
            print('nop')
    'Test that the rank properties are determined by the strategy.'
    fabric = Fabric()
    fabric._strategy = Mock(spec=Strategy)
    fabric._strategy.world_size = 1000
    assert fabric.world_size == 1000
    fabric._strategy.global_rank = 100
    assert fabric.global_rank == 100
    fabric._strategy.local_rank = 10
    assert fabric.local_rank == 10
    fabric._strategy.node_rank = 1
    assert fabric.node_rank == 1

def test_backward():
    if False:
        i = 10
        return i + 15
    'Test that backward() calls into the precision plugin.'
    fabric = Fabric()
    fabric._strategy = Mock(spec=Precision)
    loss = Mock()
    fabric.backward(loss, 'arg', keyword='kwarg')
    fabric._strategy.backward.assert_called_with(loss, None, 'arg', keyword='kwarg')

@RunIf(deepspeed=True, mps=False)
def test_backward_model_input_required():
    if False:
        return 10
    'Test that when using deepspeed and multiple models, backward() requires the model as input.'
    fabric = Fabric(strategy='deepspeed', devices=1)
    fabric._launched = True
    model0 = nn.Linear(1, 2)
    model1 = nn.Linear(1, 2)
    optimizer0 = torch.optim.Adam(model0.parameters())
    optimizer1 = torch.optim.Adam(model1.parameters())
    fabric._strategy.setup_module_and_optimizers = lambda *args: args
    fabric.setup(model0, optimizer0)
    fabric.setup(model1, optimizer1)
    loss = model0(torch.randn(1, 1, device=fabric.device)).sum()
    with pytest.raises(ValueError, match='please provide the model used to perform'):
        fabric.backward(loss)

def test_autocast():
    if False:
        return 10
    'Test that the Fabric autocast context manager lets the precision plugin handle casting.'
    fabric = Fabric()
    fabric._precision.forward_context = MagicMock()
    fabric._precision.forward_context().__enter__.assert_not_called()
    with fabric.autocast():
        fabric._precision.forward_context().__enter__.assert_called()
    fabric._precision.forward_context().__exit__.assert_called()

def test_no_backward_sync():
    if False:
        return 10
    'Test that `Fabric.no_backward_sync()` validates the strategy and model is compatible.'
    fabric = Fabric(devices=1)
    model = nn.Linear(3, 3)
    with pytest.raises(TypeError, match='You need to set up the model first'), fabric.no_backward_sync(model):
        pass
    model = fabric.setup(model)
    fabric._strategy = Mock(spec=ParallelStrategy, _backward_sync_control=None)
    with pytest.warns(PossibleUserWarning, match='The `ParallelStrategy` does not support skipping the'), fabric.no_backward_sync(model):
        pass
    fabric._strategy = Mock(spec=SingleDeviceStrategy, _backward_sync_control=MagicMock())
    with fabric.no_backward_sync(model):
        pass
    fabric._strategy._backward_sync_control.no_backward_sync.assert_not_called()
    fabric._strategy = Mock(spec=XLAStrategy, _backward_sync_control=MagicMock())
    with fabric.no_backward_sync(model):
        pass
    fabric._strategy._backward_sync_control.no_backward_sync.assert_not_called()
    fabric._strategy = Mock(_backward_sync_control=MagicMock())
    with fabric.no_backward_sync(model, enabled=False):
        pass
    fabric._strategy._backward_sync_control.no_backward_sync.assert_not_called()
    with fabric.no_backward_sync(model):
        pass
    fabric._strategy._backward_sync_control.no_backward_sync.assert_called_once_with(model._forward_module)

def test_launch_without_function():
    if False:
        for i in range(10):
            print('nop')
    'Test the various ways `Fabric.launch()` can be called.'
    fabric = Fabric()
    nothing = Mock()
    fabric.launch(nothing)
    nothing.assert_called()
    fabric = Fabric()
    fabric._strategy._launcher = Mock()
    fabric.launch()
    fabric._strategy._launcher.launch.assert_called()

def test_launch_with_function():
    if False:
        print('Hello World!')
    'Test the various ways `Fabric.launch(function)` can be called.'

    def fn_without_args():
        if False:
            for i in range(10):
                print('nop')
        pass
    fabric = Fabric()
    with pytest.raises(TypeError, match='needs to take at least one argument'):
        fabric.launch(fn_without_args)

    def fn_with_one_arg(arg):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(arg, Fabric)
        fn_with_one_arg.called = True
    fabric = Fabric()
    fabric.launch(fn_with_one_arg)
    assert fn_with_one_arg.called
    fabric = Fabric()
    with pytest.raises(TypeError, match='needs to be a callable'):
        fabric.launch(fn_with_one_arg(fabric))

@mock.patch.dict(os.environ, {'LT_CLI_USED': '1'})
def test_launch_and_cli_not_allowed():
    if False:
        for i in range(10):
            print('nop')
    fabric = Fabric(devices=1)
    with pytest.raises(RuntimeError, match=escape('Calling  `.launch()` again is not allowed')):
        fabric.launch()

@RunIf(mps=False)
@pytest.mark.parametrize('strategy', ['xla', 'ddp_spawn'])
def test_launch_and_strategies_unsupported_combinations(strategy, xla_available):
    if False:
        print('Hello World!')
    fabric = Fabric(strategy=strategy)
    with pytest.raises(TypeError, match='launch\\(\\)` needs to be called with a function'):
        fabric.launch()

@mock.patch.dict(os.environ, {'LT_CLI_USED': '1'})
def test_overridden_run_and_cli_not_allowed():
    if False:
        print('Hello World!')

    class FabricWithRun(Fabric):

        def run(self):
            if False:
                while True:
                    i = 10
            pass
    with pytest.raises(TypeError, match=escape('Overriding `Fabric.run()` and launching from the CLI is not allowed')):
        FabricWithRun()

def test_module_sharding_context():
    if False:
        for i in range(10):
            print('nop')
    'Test that the sharding context manager gets applied when the strategy supports it and is a no-op otherwise.'
    fabric = Fabric()
    fabric._launched = True
    fabric._strategy = MagicMock(spec=DDPStrategy, module_sharded_context=Mock())
    with pytest.warns(DeprecationWarning, match='sharded_model'), fabric.sharded_model():
        pass
    fabric._strategy.module_sharded_context.assert_not_called()
    fabric._strategy = MagicMock(spec=_Sharded)
    with pytest.warns(DeprecationWarning, match='sharded_model'), fabric.sharded_model():
        pass
    fabric._strategy.module_sharded_context.assert_called_once()

def test_init_module_context(monkeypatch):
    if False:
        return 10
    'Test that the strategy returns the context manager for initializing the module.'
    import lightning.fabric
    fabric = Fabric(accelerator='cpu')
    strategy = SingleDeviceStrategy(device=torch.device('cuda'))
    strategy.module_init_context = Mock(wraps=strategy.module_init_context)
    fabric._strategy = strategy
    with fabric.init_module():
        pass
    strategy.module_init_context.assert_called_once_with(empty_init=None)
    strategy.module_init_context.reset_mock()
    monkeypatch.setattr(lightning.fabric.fabric, '_TORCH_GREATER_EQUAL_2_0', False)
    with pytest.warns(PossibleUserWarning, match="can't place the model parameters on the device"):
        with fabric.init_module():
            pass
    strategy.module_init_context.assert_called_once()

def test_init_tensor_context(monkeypatch):
    if False:
        while True:
            i = 10
    'Test that `.init_tensor()` warns if using PyTorch < 2.0.'
    import lightning.fabric
    fabric = Fabric(accelerator='cpu')
    strategy = SingleDeviceStrategy(device=torch.device('cuda'))
    strategy.tensor_init_context = Mock(wraps=strategy.tensor_init_context)
    fabric._strategy = strategy
    with fabric.init_tensor():
        pass
    strategy.tensor_init_context.assert_called_once()
    strategy.tensor_init_context.reset_mock()
    monkeypatch.setattr(lightning.fabric.fabric, '_TORCH_GREATER_EQUAL_2_0', False)
    with pytest.warns(PossibleUserWarning, match="can't place tensors on the device directly"):
        with fabric.init_tensor():
            pass
    strategy.tensor_init_context.assert_called_once()

def test_callbacks_input():
    if False:
        return 10
    'Test the various ways in which callbacks can be registered with Fabric.'
    callback0 = Mock()
    callback1 = Mock()
    fabric = Fabric(callbacks=callback0)
    assert fabric._callbacks == [callback0]
    fabric = Fabric(callbacks=[callback0, callback1])
    assert fabric._callbacks == [callback0, callback1]

def test_call():
    if False:
        while True:
            i = 10
    'Test that `fabric.call` triggers the callback implementations.'
    callback0 = Mock()
    callback1 = Mock()
    fabric = Fabric(callbacks=[callback0, callback1])
    fabric.call('on_train_end')
    callback0.on_train_end.assert_called_once()
    callback1.on_train_end.assert_called_once()
    fabric.call('on_train_end', 'positional', keyword='keyword')
    callback0.on_train_end.assert_called_with('positional', keyword='keyword')
    callback1.on_train_end.assert_called_with('positional', keyword='keyword')
    callback0 = Mock()
    callback1 = Mock(spec_set={})
    fabric = Fabric(callbacks=[callback0, callback1])
    fabric.call('on_train_end')
    callback0.on_train_end.assert_called_once()
    assert not callback1.mock_calls
    callback = Mock(not_a_method=1)
    fabric = Fabric(callbacks=[callback])
    with pytest.warns(UserWarning, match='Skipping the callback `Mock.not_a_method`'):
        fabric.call('not_a_method')
    assert not callback1.mock_calls

def test_special_callbacks():
    if False:
        i = 10
        return i + 15
    'Tests special callbacks that have hooks for internal Fabric events.'

    class SpecialCallback:

        def on_after_optimizer_step(self, strategy, optimizer):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def on_after_setup(self, fabric, module):
            if False:
                i = 10
                return i + 15
            pass
    callback = Mock(wraps=SpecialCallback())
    fabric = Fabric(accelerator='cpu', callbacks=[callback])
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    (fabric_model, fabric_optimizer) = fabric.setup(model, optimizer)
    callback.on_after_setup.assert_called_once_with(fabric=fabric, module=fabric_model)
    model(torch.randn(2, 2)).sum().backward()
    fabric_optimizer.step()
    callback.on_after_optimizer_step.assert_called_once_with(strategy=fabric._strategy, optimizer=optimizer)

def test_loggers_input():
    if False:
        while True:
            i = 10
    'Test the various ways in which loggers can be registered with Fabric.'
    logger0 = Mock()
    logger1 = Mock()
    fabric = Fabric(loggers=None)
    assert fabric._loggers == []
    fabric = Fabric(loggers=[])
    assert fabric._loggers == []
    fabric = Fabric(loggers=logger0)
    assert fabric._loggers == [logger0]
    fabric = Fabric(loggers=[logger0, logger1])
    assert fabric._loggers == [logger0, logger1]

def test_log():
    if False:
        i = 10
        return i + 15
    'Test that `fabric.log` sends the metrics to each logger.'
    logger0 = Mock()
    logger1 = Mock()
    fabric = Fabric(loggers=[logger0, logger1])
    fabric.log('test', 1)
    logger0.log_metrics.assert_called_with(metrics={'test': 1}, step=None)
    logger1.log_metrics.assert_called_with(metrics={'test': 1}, step=None)
    fabric.log('test', 2, step=15)
    logger0.log_metrics.assert_called_with(metrics={'test': 2}, step=15)
    logger1.log_metrics.assert_called_with(metrics={'test': 2}, step=15)

def test_log_dict():
    if False:
        while True:
            i = 10
    'Test that `fabric.log_dict` sends the metrics dict to each logger.'
    logger0 = Mock()
    logger1 = Mock()
    fabric = Fabric(loggers=[logger0, logger1])
    fabric.log_dict({'foo': 1, 'bar': 2}, step=None)
    logger0.log_metrics.assert_called_with(metrics={'foo': 1, 'bar': 2}, step=None)
    logger1.log_metrics.assert_called_with(metrics={'foo': 1, 'bar': 2}, step=None)
    fabric.log_dict({'foo': 3, 'bar': 4}, step=15)
    logger0.log_metrics.assert_called_with(metrics={'foo': 3, 'bar': 4}, step=15)
    logger1.log_metrics.assert_called_with(metrics={'foo': 3, 'bar': 4}, step=15)

def test_log_dict_input_parsing():
    if False:
        for i in range(10):
            print('nop')
    'Test validation of input data types and preprocessing.'
    logger = Mock()
    fabric = Fabric(loggers=[logger])
    fabric.log('log', torch.tensor(1))
    logger.log_metrics.assert_called_with(metrics={'log': 1}, step=None)
    fabric.log_dict({'log_dict': torch.tensor(1)})
    logger.log_metrics.assert_called_with(metrics={'log_dict': 1}, step=None)
    fabric.log('log', torch.tensor([2]))
    logger.log_metrics.assert_called_with(metrics={'log': 2}, step=None)
    fabric.log_dict({'log_dict': torch.tensor([2])})
    logger.log_metrics.assert_called_with(metrics={'log_dict': 2}, step=None)
    with pytest.raises(ValueError, match='it cannot be converted to a scalar.'):
        fabric.log('log', torch.tensor([3, 4]))
    with pytest.raises(ValueError, match='it cannot be converted to a scalar.'):
        fabric.log_dict({'log_dict': torch.tensor([3, 4])})

@pytest.mark.parametrize('setup', [True, False])
def test_save_wrapped_objects(setup, tmp_path):
    if False:
        while True:
            i = 10
    'Test that when modules and optimizers are in the state, they get unwrapped properly.'
    fabric = Fabric(devices=1)
    save_checkpoint_mock = Mock()
    fabric.strategy.save_checkpoint = save_checkpoint_mock
    unwrapped_model = BoringModel()
    unwrapped_optimizer = torch.optim.Adam(unwrapped_model.parameters())
    if setup:
        (model, optimizer) = fabric.setup(unwrapped_model, unwrapped_optimizer)
        assert isinstance(model, _FabricModule)
        assert isinstance(optimizer, _FabricOptimizer)
    else:
        (model, optimizer) = (unwrapped_model, unwrapped_optimizer)
    anything = {'cocofruit': 1}
    state = {'model': model, 'optimizer': optimizer, 'anything': anything}
    expected = {'model': unwrapped_model, 'optimizer': unwrapped_optimizer, 'anything': anything}
    fabric.save(tmp_path, state)
    save_checkpoint_mock.assert_called_with(state=expected, path=tmp_path, filter=None)

def test_save_filter(tmp_path):
    if False:
        print('Hello World!')
    fabric = Fabric(devices=1)
    checkpoint_io_mock = Mock()
    fabric.strategy.checkpoint_io = checkpoint_io_mock
    model = BoringModel()
    optimizer = torch.optim.Adam(model.parameters())
    anything = {'cocofruit': 1}
    state = {'model': model, 'optimizer': optimizer, 'anything': anything, 'foo': 1}
    save_path = tmp_path / 'foo.pth'
    filter = {k: lambda k, v: False for k in state}
    fabric.save(save_path, state, filter=filter)
    checkpoint_io_mock.save_checkpoint.assert_called_with(checkpoint={'foo': 1}, path=save_path, storage_options=None)
    with pytest.raises(TypeError, match='should be a dict'):
        fabric.save(save_path, state, filter='foo')
    with pytest.raises(TypeError, match="callable, given 'foo"):
        fabric.save(save_path, state, filter={'model': 'foo'})
    with pytest.raises(ValueError, match="keys {'asd'} are not present in the state keys"):
        fabric.save(save_path, state, filter={'asd': lambda k, v: True})
    checkpoint_io_mock.reset_mock()
    filter = {'model': lambda k, v: 'weight' in k, 'anything': lambda k, v: isinstance(v, int), 'optimizer': lambda k, v: 'param_groups' in k}
    fabric.save(save_path, state, filter=filter)
    checkpoint_io_mock.save_checkpoint.assert_called_with(checkpoint={'model': {'layer.weight': ANY}, 'optimizer': {'param_groups': ANY}, 'anything': anything, 'foo': 1}, path=save_path, storage_options=None)

@pytest.mark.parametrize('setup', [True, False])
def test_load_wrapped_objects(setup, tmp_path):
    if False:
        i = 10
        return i + 15
    'Test that loading happens in-place for model, optimizer, and other user data.'
    fabric = Fabric(accelerator='cpu')
    expected_remainder = {'extra': 'data'}

    def mocked_load_checkpoint(path, state, strict):
        if False:
            for i in range(10):
                print('nop')
        assert not isinstance(state['model'], _FabricModule)
        assert not isinstance(state['optimizer'], _FabricOptimizer)
        state.update({'int': 5, 'dict': {'x': 1}})
        return expected_remainder
    fabric.strategy.load_checkpoint = mocked_load_checkpoint
    unwrapped_model = BoringModel()
    unwrapped_optimizer = torch.optim.Adam(unwrapped_model.parameters())
    if setup:
        (model, optimizer) = fabric.setup(unwrapped_model, unwrapped_optimizer)
        assert isinstance(model, _FabricModule)
        assert isinstance(optimizer, _FabricOptimizer)
    else:
        (model, optimizer) = (unwrapped_model, unwrapped_optimizer)
    state = {'model': model, 'optimizer': optimizer, 'int': 0, 'dict': {'x': 0}}
    expected = {'model': model, 'optimizer': optimizer, 'int': 5, 'dict': {'x': 1}}
    remainder = fabric.load(tmp_path, state)
    assert state == expected
    assert remainder == expected_remainder

def test_load_raw():
    if False:
        return 10
    'Test that `Fabric.load_raw()` unwraps the object to load and calls into the strategy.'
    fabric = Fabric(accelerator='cpu')
    fabric.strategy.load_checkpoint = Mock()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    (wrapped_model, wrapped_optimizer) = fabric.setup(model, optimizer)
    fabric.load_raw(path='path0', obj=model)
    fabric.strategy.load_checkpoint.assert_called_with(path='path0', state=model, strict=True)
    fabric.load_raw(path='path1', obj=wrapped_model, strict=False)
    fabric.strategy.load_checkpoint.assert_called_with(path='path1', state=model, strict=False)
    fabric.load_raw(path='path2', obj=wrapped_optimizer)
    fabric.strategy.load_checkpoint.assert_called_with(path='path2', state=optimizer, strict=True)

def test_barrier():
    if False:
        while True:
            i = 10
    'Test that `Fabric.barrier()` calls into the strategy.'
    fabric = Fabric()
    fabric._strategy = Mock()
    fabric._launched = True
    fabric.barrier('test')
    fabric._strategy.barrier.assert_called_once_with(name='test')

def test_broadcast():
    if False:
        return 10
    'Test that `Fabric.broadcast()` calls into the strategy.'
    fabric = Fabric()
    fabric._strategy = Mock()
    fabric._launched = True
    fabric.broadcast(torch.tensor(1), src=2)
    fabric._strategy.broadcast.assert_called_once_with(torch.tensor(1), src=2)

def test_all_gather():
    if False:
        i = 10
        return i + 15
    'Test that `Fabric.all_gather()` applies itself to collections and calls into the strategy.'
    fabric = Fabric()
    fabric._strategy = Mock(root_device=torch.device('cpu'))
    fabric._launched = True
    defaults = {'group': None, 'sync_grads': False}
    fabric.all_gather(torch.tensor(1))
    fabric._strategy.all_gather.assert_called_once_with(torch.tensor(1), **defaults)
    fabric._strategy.reset_mock()
    fabric.all_gather([torch.tensor(2), torch.tensor(3), 'string'])
    fabric._strategy.all_gather.assert_has_calls([call(torch.tensor(2), **defaults), call(torch.tensor(3), **defaults)])
    fabric._strategy.reset_mock()
    fabric.all_gather({'a': torch.tensor(4), 'b': [torch.tensor(5)], 'c': 'string'})
    fabric._strategy.all_gather.assert_has_calls([call(torch.tensor(4), **defaults), call(torch.tensor(5), **defaults)])

def test_all_reduce():
    if False:
        while True:
            i = 10
    'Test that `Fabric.all_reduce()` applies itself to collections and calls into the strategy.'
    fabric = Fabric()
    fabric._strategy = Mock(root_device=torch.device('cpu'))
    fabric._launched = True
    defaults = {'group': None, 'reduce_op': 'mean'}
    fabric.all_reduce(torch.tensor(1))
    fabric._strategy.all_reduce.assert_called_once_with(torch.tensor(1), **defaults)
    fabric._strategy.reset_mock()
    fabric.all_reduce([torch.tensor(2), torch.tensor(3), 'string'])
    fabric._strategy.all_reduce.assert_has_calls([call(torch.tensor(2), **defaults), call(torch.tensor(3), **defaults)])
    fabric._strategy.reset_mock()
    fabric.all_reduce({'a': torch.tensor(4), 'b': [torch.tensor(5)], 'c': 'string'})
    fabric._strategy.all_reduce.assert_has_calls([call(torch.tensor(4), **defaults), call(torch.tensor(5), **defaults)])

def test_rank_zero_first():
    if False:
        print('Hello World!')
    'Test that rank 0 completes first before all other processes can execute under `.rank_zero_first()`.'

    def record_calls_for_rank(rank):
        if False:
            i = 10
            return i + 15
        call_order = []
        fabric = Fabric()
        fabric._strategy = Mock(global_rank=rank)
        fabric.barrier = Mock(side_effect=lambda *_: call_order.append('barrier'))
        target = Mock(run=Mock(side_effect=lambda *_: call_order.append('run')))
        with fabric.rank_zero_first():
            target.run()
        return call_order
    assert record_calls_for_rank(0) == ['run', 'barrier', 'barrier']
    assert record_calls_for_rank(1) == ['barrier', 'run', 'barrier']

@pytest.mark.parametrize(('clip_val', 'max_norm'), [(0.001, None), (None, 1)])
def test_grad_clipping(clip_val, max_norm):
    if False:
        while True:
            i = 10
    fabric = Fabric(devices=1)
    fabric.strategy.clip_gradients_norm = Mock()
    fabric.strategy.clip_gradients_value = Mock()
    torch_model = nn.Linear(1, 1)
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.001)
    (model, optimizer) = fabric.setup(torch_model, torch_optimizer)
    loss = model(torch.rand(1, 1).to(fabric.device))
    fabric.backward(loss)
    fabric.strategy.clip_gradients_value.assert_not_called()
    fabric.strategy.clip_gradients_norm.assert_not_called()
    fabric.clip_gradients(model, optimizer, max_norm=max_norm, clip_val=clip_val)
    if clip_val is not None:
        fabric.strategy.clip_gradients_value.assert_called_once_with(torch_model, torch_optimizer, clip_val=clip_val)
        fabric.strategy.clip_gradients_norm.assert_not_called()
    else:
        fabric.strategy.clip_gradients_value.assert_not_called()
        fabric.strategy.clip_gradients_norm.assert_called_once_with(torch_model, torch_optimizer, max_norm=max_norm, norm_type=2.0, error_if_nonfinite=True)

def test_verify_launch_called():
    if False:
        i = 10
        return i + 15
    'Test that the user gets an error message if they forgot to call `.launch()`.'
    fabric = Fabric(accelerator='cpu')
    assert not fabric._launched
    fabric._strategy = Mock(spec=SingleDeviceStrategy)
    fabric._validate_launched()
    fabric._strategy = Mock(spec=DataParallelStrategy)
    fabric._validate_launched()
    fabric._strategy = Mock(spec=DDPStrategy)
    with pytest.raises(RuntimeError, match='you must call `.launch\\(\\)`'):
        fabric._validate_launched()
    method_names = ('setup', 'setup_module', 'setup_dataloaders', 'broadcast', 'barrier', 'all_reduce', 'all_gather')
    for method_name in method_names:
        method = getattr(fabric, method_name)
        with pytest.raises(RuntimeError, match='you must call `.launch\\(\\)`'):
            method(Mock())
    ctx_manager_names = ('init_module',)
    for ctx_manager_name in ctx_manager_names:
        ctx_manager = getattr(fabric, ctx_manager_name)
        with pytest.raises(RuntimeError, match='you must call `.launch\\(\\)`'), ctx_manager():
            pass
    fabric.launch()
    assert fabric._launched
    fabric._validate_launched()

@pytest.mark.skipif(sys.platform == 'darwin', reason='https://github.com/pytorch/pytorch/issues/95708')
@RunIf(dynamo=True)
@pytest.mark.parametrize('kwargs', [{}, pytest.param({'precision': '16-true'}, marks=pytest.mark.xfail(raises=RuntimeError, match='Unsupported')), pytest.param({'precision': '64-true'}, marks=pytest.mark.xfail(raises=RuntimeError, match='Unsupported'))])
def test_fabric_with_torchdynamo_fullgraph(kwargs):
    if False:
        while True:
            i = 10

    class MyModel(torch.nn.Module):

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self.l = torch.nn.Linear(10, 10)

        def forward(self, x):
            if False:
                for i in range(10):
                    print('nop')
            assert torch._dynamo.is_compiling()
            return self.l(x)

    def fn(model, x):
        if False:
            for i in range(10):
                print('nop')
        assert torch._dynamo.is_compiling()
        a = x * 10
        return model(a)
    fabric = Fabric(devices=1, **kwargs)
    model = MyModel()
    fmodel = fabric.setup(model)
    cfn = torch.compile(fn, fullgraph=True)
    x = torch.randn(10, 10, device=fabric.device)
    out = cfn(fmodel, x)
    assert isinstance(out, torch.Tensor)