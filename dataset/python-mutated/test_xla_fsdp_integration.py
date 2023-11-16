import os
import re
from pathlib import Path
from unittest import mock
import pytest
import torch
from lightning.fabric import Fabric
from lightning.fabric.strategies import XLAFSDPStrategy
from torch.utils.data import DataLoader
from tests_fabric.helpers.models import RandomDataset
from tests_fabric.helpers.runif import RunIf

def _xla_fsdp_rewrap_warning(fabric: Fabric):
    if False:
        while True:
            i = 10
    'Fabric launch function for test_xla_fsdp_rewrap_warning.'
    from torch_xla.distributed.fsdp.xla_fully_sharded_data_parallel import XlaFullyShardedDataParallel
    with fabric.init_module():
        model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU(), XlaFullyShardedDataParallel(torch.nn.Linear(1, 1)))
    if fabric.node_rank:
        with pytest.warns(match='submodule is already wrapped'):
            model = fabric.setup_module(model)
    else:
        model = fabric.setup_module(model)
    fabric.barrier('warning_check')
    assert not isinstance(model._forward_module[0], XlaFullyShardedDataParallel)
    assert not isinstance(model._forward_module[1], XlaFullyShardedDataParallel)
    assert isinstance(model._forward_module[2], XlaFullyShardedDataParallel)

@RunIf(min_torch='2.0', tpu=True, standalone=True)
def test_xla_fsdp_rewrap_warning():
    if False:
        print('Hello World!')
    'Test that XLAFSDP warns about rewrapping the modules.'
    from torch_xla.distributed.fsdp.wrap import always_wrap_policy
    strategy = XLAFSDPStrategy(auto_wrap_policy=always_wrap_policy)
    fabric = Fabric(accelerator='tpu', strategy=strategy)
    fabric.launch(_xla_fsdp_rewrap_warning)

def xla_fsdp_train_save_load(fabric: Fabric, tmp_path, state_dict_type):
    if False:
        return 10
    'Fabric launch function for test_xla_fsdp_train_save_load.'
    tmp_path = Path(fabric.broadcast(tmp_path))
    with fabric.init_module():
        model_1 = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
    model_1 = fabric.setup_module(model_1)
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=0.1)
    optimizer_1 = fabric.setup_optimizers(optimizer_1)
    dataloader = DataLoader(RandomDataset(32, 64))
    dataloader = fabric.setup_dataloaders(dataloader)

    def step(model, batch):
        if False:
            while True:
                i = 10
        output = model(batch)
        return torch.nn.functional.mse_loss(output, torch.ones_like(output))
    model_1.train()
    data_iter = iter(dataloader)
    batch = next(data_iter)
    loss = step(model_1, batch)
    fabric.backward(loss)
    optimizer_1.step()
    optimizer_1.zero_grad()
    state = {'model': model_1, 'optimizer': optimizer_1, 'step_count': 1}
    checkpoint_path = tmp_path / 'foo.pth'
    world_size = fabric.world_size
    local_process_count = len(fabric.strategy.parallel_devices)
    is_multihost = local_process_count < world_size
    if state_dict_type == 'full' and is_multihost:
        with pytest.raises(OSError, match='Multihost setups do not have a shared filesystem'):
            fabric.save(checkpoint_path, state)
        return
    fabric.save(checkpoint_path, state)
    if state_dict_type == 'sharded':
        pattern = f'checkpoint_rank-0000000\\d-of-{world_size:08d}\\.pth'
        shards = os.listdir(checkpoint_path)
        assert len(shards) == local_process_count
        for name in shards:
            assert re.match(pattern, name)
        with fabric.init_module():
            model_2 = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
        model_2 = fabric.setup_module(model_2)
        optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=0.1)
        optimizer_2 = fabric.setup_optimizers(optimizer_2)
        state = {'model': model_2, 'optimizer': optimizer_2, 'step_count': 0}
        metadata = fabric.load(checkpoint_path, state)
        assert not metadata
        assert state['step_count'] == 1
        for (p0, p1) in zip(model_1._forward_module.parameters(), model_2.parameters()):
            torch.testing.assert_close(p0, p1, atol=0, rtol=0, equal_nan=True)
        state = {'model': model_2, 'coconut': 11}
        with pytest.raises(KeyError, match="The requested state contains a key 'coconut' that does not exist"):
            fabric.load(checkpoint_path, state)
        state = {'model': model_2, 'coconut': 11}
        fabric.load(checkpoint_path, state, strict=False)
        assert state['coconut'] == 11
    if state_dict_type == 'full':
        assert set(os.listdir(tmp_path)) == {'foo.pth'}
        with fabric.init_module():
            model_2 = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        model_2.to(device)
        state = {'model': model_2}
        fabric.load(checkpoint_path, state)
        with pytest.raises(AssertionError, match='do not match'):
            for (p0, p1) in zip(model_1.parameters(), model_2.parameters()):
                torch.testing.assert_close(p0, p1, atol=0, rtol=0, equal_nan=True)

@RunIf(min_torch='2.0', tpu=True, standalone=True)
@pytest.mark.parametrize(('use_auto_wrap_policy', 'state_dict_type', 'sequential_save'), [(False, 'sharded', False), (False, 'full', False), (False, 'full', True), (True, 'sharded', False), (True, 'full', False)])
def test_xla_fsdp_train_save_load(tmp_path, use_auto_wrap_policy, state_dict_type, sequential_save):
    if False:
        for i in range(10):
            print('nop')
    'Test XLAFSDP training, saving and loading checkpoint (both full and sharded).'
    from torch_xla.distributed.fsdp.wrap import always_wrap_policy
    policy = always_wrap_policy if use_auto_wrap_policy else None
    strategy = XLAFSDPStrategy(auto_wrap_policy=policy, state_dict_type=state_dict_type, sequential_save=sequential_save)
    fabric = Fabric(accelerator='tpu', strategy=strategy)
    fabric.launch(xla_fsdp_train_save_load, tmp_path, state_dict_type)

def _test_setup_module_move_to_device(fabric, move_to_device):
    if False:
        print('Hello World!')
    model = torch.nn.Linear(10, 10, bias=False)
    with mock.patch('lightning.fabric.wrappers._FabricModule') as fabric_module_mock:
        fabric_model = fabric.setup_module(model, move_to_device=move_to_device)
    fabric_module_mock.assert_not_called()
    assert fabric_model.device == torch.device('cpu')
    assert fabric.device.type == 'xla'

@RunIf(min_torch='2.0', tpu=True, standalone=True)
@pytest.mark.parametrize('move_to_device', [True, False])
def test_setup_module_move_to_device(move_to_device):
    if False:
        return 10
    'Test that `move_to_device` does nothing, FSDP decides which device parameters get moved to which device\n    (sharding).'
    from torch_xla.distributed.fsdp.wrap import always_wrap_policy
    strategy = XLAFSDPStrategy(auto_wrap_policy=always_wrap_policy)
    fabric = Fabric(accelerator='tpu', strategy=strategy)
    fabric.launch(_test_setup_module_move_to_device, move_to_device=move_to_device)