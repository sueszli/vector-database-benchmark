import os
from copy import deepcopy
from unittest import mock
from unittest.mock import ANY
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.plugins import DeepSpeedPrecision
from lightning.fabric.strategies import DeepSpeedStrategy
from torch.utils.data import DataLoader
from tests_fabric.helpers.models import RandomDataset, RandomIterableDataset
from tests_fabric.helpers.runif import RunIf
from tests_fabric.test_fabric import BoringModel

@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True)
def test_deepspeed_multiple_models():
    if False:
        return 10
    fabric = Fabric(strategy=DeepSpeedStrategy(stage=3, logging_batch_size_per_gpu=1), devices=2, accelerator='gpu')
    fabric.launch()
    with fabric.init_module():
        model = BoringModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    (model, optimizer) = fabric.setup(model, optimizer)
    for i in range(2):
        optimizer.zero_grad()
        x = model(torch.randn(1, 32).to(fabric.device))
        loss = x.sum()
        if i == 0:
            assert all((w.nelement() == 0 for w in model.state_dict().values()))
        fabric.backward(loss, model=model)
        if i == 0:
            state_dict = deepcopy(model.state_dict())
        optimizer.step()
    for (mw_b, mw_a) in zip(state_dict.values(), model.state_dict().values()):
        assert not torch.allclose(mw_b, mw_a)
    fabric.seed_everything(42)
    model_1 = BoringModel()
    optimizer_1 = torch.optim.SGD(model_1.parameters(), lr=0.0001)
    fabric.seed_everything(42)
    model_2 = BoringModel()
    optimizer_2 = torch.optim.SGD(model_2.parameters(), lr=0.0001)
    for (mw_1, mw_2) in zip(model_1.state_dict().values(), model_2.state_dict().values()):
        assert torch.allclose(mw_1, mw_2)
    (model_1, optimizer_1) = fabric.setup(model_1, optimizer_1)
    (model_2, optimizer_2) = fabric.setup(model_2, optimizer_2)
    fabric.seed_everything(42)
    data_list = []
    for _ in range(2):
        optimizer_1.zero_grad()
        data = torch.randn(1, 32).to(fabric.device)
        data_list.append(data)
        x = model_1(data)
        loss = x.sum()
        fabric.backward(loss, model=model_1)
        optimizer_1.step()
    assert all((w.nelement() > 1 for w in model_1.state_dict().values()))
    assert all((w.nelement() == 0 for w in model_2.state_dict().values()))
    for data in data_list:
        optimizer_2.zero_grad()
        x = model_2(data)
        loss = x.sum()
        fabric.backward(loss, model=model_2)
        optimizer_2.step()
    for (mw_1, mw_2) in zip(model_1.state_dict().values(), model_2.state_dict().values()):
        assert torch.allclose(mw_1, mw_2)
    ranks = fabric.all_gather(torch.tensor([fabric.local_rank]).to(fabric.device))
    assert torch.allclose(ranks.cpu(), torch.tensor([[0], [1]]))
    assert fabric.broadcast(True)
    assert fabric.is_global_zero == (fabric.local_rank == 0)

@RunIf(min_cuda_gpus=1, deepspeed=True)
@pytest.mark.parametrize(('dataset_cls', 'logging_batch_size_per_gpu', 'expected_batch_size'), [(RandomDataset, None, 1), (RandomDataset, 10, 10), (RandomIterableDataset, None, 1), (RandomIterableDataset, 10, 10)])
def test_deepspeed_auto_batch_size_config_select(dataset_cls, logging_batch_size_per_gpu, expected_batch_size):
    if False:
        i = 10
        return i + 15
    'Test to ensure that the batch size is correctly set as expected for deepspeed logging purposes.'
    fabric = Fabric(accelerator='cuda', devices=1, strategy=DeepSpeedStrategy(logging_batch_size_per_gpu=logging_batch_size_per_gpu, zero_optimization=False))
    fabric.launch()
    assert isinstance(fabric._strategy, DeepSpeedStrategy)
    _ = fabric.setup_dataloaders(DataLoader(dataset_cls(32, 64)))
    config = fabric._strategy.config
    assert config['train_micro_batch_size_per_gpu'] == expected_batch_size

@RunIf(min_cuda_gpus=1, standalone=True, deepspeed=True)
def test_deepspeed_configure_optimizers():
    if False:
        i = 10
        return i + 15
    'Test that the deepspeed strategy with default initialization wraps the optimizer correctly.'
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    fabric = Fabric(strategy=DeepSpeedStrategy(), accelerator='cuda', devices=1, precision='16-mixed')
    fabric.launch()
    model = nn.Linear(3, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    (model, optimizer) = fabric.setup(model, optimizer)
    assert isinstance(optimizer.optimizer, DeepSpeedZeroOptimizer)
    assert isinstance(optimizer.optimizer.optimizer, torch.optim.SGD)

@RunIf(min_cuda_gpus=1, deepspeed=True)
def test_deepspeed_custom_precision_params():
    if False:
        i = 10
        return i + 15
    'Test that if the FP16 parameters are set via the DeepSpeedStrategy, the deepspeed config contains these\n    changes.'
    strategy = DeepSpeedStrategy(loss_scale=10, initial_scale_power=11, loss_scale_window=12, hysteresis=13, min_loss_scale=14)
    fabric = Fabric(strategy=strategy, precision='16-mixed', accelerator='cuda', devices=1)
    fabric.launch()
    assert fabric._strategy._config_initialized
    assert fabric._strategy.config['fp16']['loss_scale'] == 10
    assert fabric._strategy.config['fp16']['initial_scale_power'] == 11
    assert fabric._strategy.config['fp16']['loss_scale_window'] == 12
    assert fabric._strategy.config['fp16']['hysteresis'] == 13
    assert fabric._strategy.config['fp16']['min_loss_scale'] == 14

@RunIf(min_cuda_gpus=1, standalone=True, deepspeed=True)
def test_deepspeed_custom_activation_checkpointing_params_forwarded():
    if False:
        print('Hello World!')
    'Test that the activation checkpointing parameters get passed to `deepspeed.checkpointing.configure`\n    correctly.'
    import deepspeed
    strategy = DeepSpeedStrategy(partition_activations=True, cpu_checkpointing=True, contiguous_memory_optimization=True, synchronize_checkpoint_boundary=True)
    fabric = Fabric(strategy=strategy, precision='16-mixed', accelerator='cuda', devices=1)
    fabric.launch()
    model = nn.Linear(3, 3)
    optimizer = torch.optim.Adam(model.parameters())
    with mock.patch('deepspeed.checkpointing.configure', wraps=deepspeed.checkpointing.configure) as configure:
        fabric.setup(model, optimizer)
    configure.assert_called_with(mpu_=None, partition_activations=True, contiguous_checkpointing=True, checkpoint_in_cpu=True, profile=None)

@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True)
def test_deepspeed_multigpu_stage_3():
    if False:
        for i in range(10):
            print('nop')
    'Test to ensure ZeRO Stage 3 works with a parallel model.'
    fabric = Fabric(strategy=DeepSpeedStrategy(stage=3), accelerator='cuda', devices=2, precision='16-mixed')
    fabric.launch()

    def _make_block():
        if False:
            for i in range(10):
                print('nop')
        return nn.Sequential(nn.Linear(32, 32, bias=False), nn.ReLU())
    with fabric.init_module():
        model = nn.Sequential(*(_make_block() for _ in range(5)), nn.Linear(32, 3))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    (model, optimizer) = fabric.setup(model, optimizer)
    x = torch.rand(2, 32, device=fabric.device)
    y = torch.ones(x.size(0), device=x.device, dtype=torch.long)
    x = model(x)
    x = x.float()
    logits = F.softmax(x, dim=1)
    loss = F.cross_entropy(logits, y)
    fabric.backward(loss)
    optimizer.step()
    optimizer.zero_grad()

@RunIf(deepspeed=True)
@mock.patch('deepspeed.init_distributed', autospec=True)
@mock.patch('lightning.fabric.accelerators.mps.MPSAccelerator.is_available', return_value=False)
@pytest.mark.parametrize('platform', ['Linux', 'Windows'])
def test_deepspeed_env_variables_on_platforms(_, deepspeed_dist_mock, platform):
    if False:
        i = 10
        return i + 15
    'Test to ensure that we set up distributed communication correctly.\n\n    When using Windows, ranks environment variables should not be set, and DeepSpeed should handle this.\n\n    '
    fabric = Fabric(strategy=DeepSpeedStrategy(stage=3))
    strategy = fabric._strategy
    assert isinstance(strategy, DeepSpeedStrategy)
    with mock.patch('platform.system', return_value=platform) as platform_mock:
        strategy._init_deepspeed_distributed()
    deepspeed_dist_mock.assert_called()
    platform_mock.assert_called()
    if platform == 'Windows':
        assert all((k not in os.environ for k in ('MASTER_PORT', 'MASTER_ADDR', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK')))
    else:
        assert os.environ['MASTER_ADDR'] == str(strategy.cluster_environment.main_address)
        assert os.environ['MASTER_PORT'] == str(strategy.cluster_environment.main_port)
        assert os.environ['RANK'] == str(strategy.global_rank)
        assert os.environ['WORLD_SIZE'] == str(strategy.world_size)
        assert os.environ['LOCAL_RANK'] == str(strategy.local_rank)

@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True, bf16_cuda=True)
def test_deepspeed_with_bfloat16_precision():
    if False:
        while True:
            i = 10
    'Test that the DeepSpeed strategy works with bfloat16 precision.'

    class Model(nn.Module):

        def __init__(self):
            if False:
                while True:
                    i = 10
            super().__init__()
            self.layer = nn.Linear(32, 2)

        def forward(self, x):
            if False:
                while True:
                    i = 10
            assert x.dtype == torch.bfloat16
            return self.layer(x)
    fabric = Fabric(accelerator='cuda', devices=2, strategy='deepspeed_stage_3', precision='bf16-mixed')
    assert isinstance(fabric._strategy.precision, DeepSpeedPrecision)
    assert fabric._strategy.precision.precision == 'bf16-mixed'
    assert fabric._strategy.config['zero_optimization']['stage'] == 3
    fabric.launch()
    with fabric.init_module():
        model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    (model, optimizer) = fabric.setup(model, optimizer)
    assert fabric._strategy.config['bf16']['enabled']
    assert model.layer.weight.dtype == torch.bfloat16
    batch = torch.rand(2, 32, device=fabric.device)
    assert batch.dtype == torch.float32
    loss = model(batch).sum()
    fabric.backward(loss)
    optimizer.step()
    optimizer.zero_grad()

def _assert_saved_model_is_equal(fabric, model, checkpoint_path):
    if False:
        print('Hello World!')
    'Convert the saved checkpoint to a single file with the model weights consolidated to easily verify the full\n    weights in float32 precision.'
    from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
    assert isinstance(fabric.strategy, DeepSpeedStrategy)
    if fabric.is_global_zero:
        if fabric.strategy.config['zero_optimization']['stage'] in (2, 3):
            single_ckpt_path = checkpoint_path / 'single_model.pt'
            convert_zero_checkpoint_to_fp32_state_dict(checkpoint_path, single_ckpt_path, tag='checkpoint')
            state_dict = torch.load(single_ckpt_path)
        else:
            single_ckpt_path = checkpoint_path / 'checkpoint' / 'mp_rank_00_model_states.pt'
            state_dict = torch.load(single_ckpt_path)['module']
        model = model.cpu()
        for (orig_param, saved_model_param) in zip(model.parameters(), state_dict.values()):
            saved_model_param = saved_model_param.cpu().to(orig_param.dtype)
            assert torch.equal(orig_param, saved_model_param)
    fabric.barrier()

@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True, bf16_cuda=True)
@pytest.mark.parametrize('stage', [1, 2, 3])
def test_deepspeed_save_load_checkpoint_zero_3(stage, tmp_path):
    if False:
        return 10
    'Test that DeepSpeed stage 1, 2, and 3 model checkpoints can be saved and loaded successfully.'
    from deepspeed import DeepSpeedEngine
    fabric = Fabric(accelerator='cuda', devices=2, strategy=DeepSpeedStrategy(stage=stage), precision='bf16-mixed')
    fabric.launch()
    checkpoint_path = fabric.broadcast(tmp_path / 'deepspeed-checkpoint')
    with fabric.init_module():
        model = BoringModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    (model, optimizer) = fabric.setup(model, optimizer)
    assert isinstance(model._forward_module, DeepSpeedEngine)
    assert model.dtype == torch.float32
    assert next(model.parameters()).dtype == torch.bfloat16
    output = model(torch.randn(1, 32).to(fabric.device))
    loss = output.sum()
    fabric.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    state = {'model': model, 'optimizer': optimizer, 'steps': 1}
    fabric.save(checkpoint_path, state)
    fabric = Fabric(accelerator='cuda', devices=2, strategy=DeepSpeedStrategy(stage=stage), precision='bf16')
    fabric.launch()
    with fabric.init_module():
        model = BoringModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    (model, optimizer) = fabric.setup(model, optimizer)
    state = {'model': model, 'optimizer': optimizer, 'steps': 0}
    metadata = fabric.load(checkpoint_path, state)
    assert state['steps'] == 1
    assert 'ds_version' in metadata
    _assert_saved_model_is_equal(fabric, model, checkpoint_path)

@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True, bf16_cuda=True)
@pytest.mark.parametrize('empty_init', [None, True])
def test_deepspeed_init_module_with_stage_3(empty_init):
    if False:
        return 10
    'Tests how `.init_module()` behaves with ZeRO stage 3.'
    strategy = DeepSpeedStrategy(stage=3)
    fabric = Fabric(accelerator='cuda', devices=2, strategy=strategy, precision='bf16-true')
    fabric.launch()
    with mock.patch('deepspeed.zero.Init') as zero_init_mock, fabric.init_module(empty_init=empty_init):
        BoringModel()
    fabric.barrier()
    zero_init_mock.assert_called_once_with(enabled=True, remote_device=None, config_dict_or_path=ANY)

@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True, bf16_cuda=True)
@pytest.mark.parametrize('stage', [1, 2])
@pytest.mark.parametrize('empty_init', [None, False, True])
def test_deepspeed_init_module_with_stages_1_2(stage, empty_init):
    if False:
        while True:
            i = 10
    'Tests how `.init_module()` behaves with ZeRO stages 1 and 2.'
    strategy = DeepSpeedStrategy(stage=stage)
    fabric = Fabric(accelerator='cuda', devices=2, strategy=strategy, precision='bf16-true')
    fabric.launch()
    with mock.patch('deepspeed.zero.Init') as zero_init_mock, mock.patch('torch.Tensor.uniform_') as init_mock, fabric.init_module(empty_init=empty_init):
        model = BoringModel()
    zero_init_mock.assert_called_with(enabled=False, remote_device=None, config_dict_or_path=ANY)
    assert init_mock.call_count == int(not empty_init)
    assert model.layer.weight.dtype == torch.bfloat16