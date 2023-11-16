import inspect
import os
import sys
from typing import Any, Dict
from unittest import mock
from unittest.mock import Mock
import lightning.pytorch
import pytest
import torch
import torch.distributed
from lightning.fabric.plugins.environments import KubeflowEnvironment, LightningEnvironment, SLURMEnvironment, TorchElasticEnvironment, XLAEnvironment
from lightning.fabric.utilities.imports import _IS_WINDOWS
from lightning.pytorch import Trainer
from lightning.pytorch.accelerators import Accelerator, CPUAccelerator, CUDAAccelerator, MPSAccelerator, XLAAccelerator
from lightning.pytorch.plugins.io import TorchCheckpointIO
from lightning.pytorch.plugins.layer_sync import LayerSync, TorchSyncBatchNorm
from lightning.pytorch.plugins.precision import DeepSpeedPrecision, DoublePrecision, FSDPPrecision, HalfPrecision, MixedPrecision, Precision
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy, FSDPStrategy, SingleDeviceStrategy, SingleDeviceXLAStrategy, XLAStrategy
from lightning.pytorch.strategies.ddp import _DDP_FORK_ALIASES
from lightning.pytorch.strategies.launchers import _SubprocessScriptLauncher
from lightning.pytorch.trainer.connectors.accelerator_connector import _AcceleratorConnector, _set_torch_flags
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.imports import _LIGHTNING_HABANA_AVAILABLE, _graphcore_available_and_importable
from lightning_utilities.core.imports import package_available
from tests_pytorch.conftest import mock_cuda_count, mock_mps_count, mock_tpu_available, mock_xla_available
from tests_pytorch.helpers.runif import RunIf

@pytest.mark.parametrize(('accelerator', 'devices'), [('tpu', 'auto'), ('tpu', 1), ('tpu', [1]), ('tpu', 8), ('auto', 1), ('auto', 8)])
@RunIf(min_python='3.9')
def test_accelerator_choice_tpu(accelerator, devices, tpu_available, monkeypatch):
    if False:
        print('Hello World!')
    monkeypatch.setattr(torch, 'device', DeviceMock())
    if _IS_WINDOWS:
        monkeypatch.setattr(torch.multiprocessing, 'get_all_start_methods', lambda : ['fork', 'spawn'])
    connector = _AcceleratorConnector(accelerator=accelerator, devices=devices)
    assert isinstance(connector.accelerator, XLAAccelerator)
    if devices == 'auto' or (isinstance(devices, int) and devices > 1):
        assert isinstance(connector.strategy, XLAStrategy)
        assert isinstance(connector.strategy.cluster_environment, XLAEnvironment)
        assert isinstance(connector.cluster_environment, XLAEnvironment)
    else:
        assert isinstance(connector.strategy, SingleDeviceXLAStrategy)

def test_accelerator_invalid_choice():
    if False:
        return 10
    with pytest.raises(ValueError, match="You selected an invalid accelerator name: `accelerator='invalid'`"):
        Trainer(accelerator='invalid')

@pytest.mark.parametrize('invalid_strategy', ['cocofruit', object()])
def test_invalid_strategy_choice(invalid_strategy):
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError, match='You selected an invalid strategy name:'):
        _AcceleratorConnector(strategy=invalid_strategy)

def test_precision_and_precision_plugin_raises():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError, match='both `precision=16-true` and `plugins'):
        _AcceleratorConnector(precision='16-true', plugins=Precision())

@RunIf(skip_windows=True, standalone=True)
def test_strategy_choice_ddp_on_cpu(tmpdir):
    if False:
        while True:
            i = 10
    'Test that selecting DDPStrategy on CPU works.'
    _test_strategy_choice_ddp_and_cpu(tmpdir, ddp_strategy_class=DDPStrategy)

def _test_strategy_choice_ddp_and_cpu(tmpdir, ddp_strategy_class):
    if False:
        for i in range(10):
            print('nop')
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, strategy=ddp_strategy_class(find_unused_parameters=True), accelerator='cpu', devices=2)
    assert isinstance(trainer.strategy, ddp_strategy_class)
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert trainer.strategy.num_processes == 2
    assert trainer.strategy.parallel_devices == [torch.device('cpu')] * 2

@mock.patch.dict(os.environ, {'SLURM_NTASKS': '2', 'SLURM_NTASKS_PER_NODE': '1', 'SLURM_JOB_NAME': 'SOME_NAME', 'SLURM_NODEID': '0', 'LOCAL_RANK': '0', 'SLURM_PROCID': '0', 'SLURM_LOCALID': '0'})
def test_custom_cluster_environment_in_slurm_environment(cuda_count_0, tmpdir):
    if False:
        while True:
            i = 10
    'Test that we choose the custom cluster even when SLURM or TE flags are around.'

    class CustomCluster(LightningEnvironment):

        @property
        def main_address(self):
            if False:
                print('Hello World!')
            return 'asdf'

        @property
        def creates_processes_externally(self) -> bool:
            if False:
                print('Hello World!')
            return True
    trainer = Trainer(default_root_dir=tmpdir, plugins=[CustomCluster()], fast_dev_run=True, accelerator='cpu', strategy='ddp', devices=2)
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert isinstance(trainer.strategy, DDPStrategy)
    assert isinstance(trainer.strategy.cluster_environment, CustomCluster)

@RunIf(mps=False)
@mock.patch.dict(os.environ, {'SLURM_NTASKS': '2', 'SLURM_NTASKS_PER_NODE': '1', 'SLURM_JOB_NAME': 'SOME_NAME', 'SLURM_NODEID': '0', 'LOCAL_RANK': '0', 'SLURM_PROCID': '0', 'SLURM_LOCALID': '0'})
@mock.patch('lightning.pytorch.strategies.DDPStrategy.setup_distributed', autospec=True)
def test_custom_accelerator(cuda_count_0):
    if False:
        print('Hello World!')

    class Accel(Accelerator):

        def setup_device(self, device: torch.device) -> None:
            if False:
                i = 10
                return i + 15
            pass

        def get_device_stats(self, device: torch.device) -> Dict[str, Any]:
            if False:
                i = 10
                return i + 15
            pass

        def teardown(self) -> None:
            if False:
                i = 10
                return i + 15
            pass

        @staticmethod
        def parse_devices(devices):
            if False:
                i = 10
                return i + 15
            return devices

        @staticmethod
        def get_parallel_devices(devices):
            if False:
                return 10
            return [torch.device('cpu')] * devices

        @staticmethod
        def auto_device_count() -> int:
            if False:
                i = 10
                return i + 15
            return 1

        @staticmethod
        def is_available() -> bool:
            if False:
                print('Hello World!')
            return True

        @staticmethod
        def name() -> str:
            if False:
                while True:
                    i = 10
            return 'custom_acc_name'

    class Prec(Precision):
        pass

    class Strat(SingleDeviceStrategy):
        pass
    strategy = Strat(device=torch.device('cpu'), accelerator=Accel(), precision_plugin=Prec())
    trainer = Trainer(strategy=strategy, fast_dev_run=True, devices=2)
    assert isinstance(trainer.accelerator, Accel)
    assert isinstance(trainer.strategy, Strat)
    assert isinstance(trainer.precision_plugin, Prec)
    assert trainer._accelerator_connector.strategy is strategy

    class Strat(DDPStrategy):
        pass
    strategy = Strat(accelerator=Accel(), precision_plugin=Prec())
    trainer = Trainer(strategy=strategy, fast_dev_run=True, devices=2)
    assert isinstance(trainer.accelerator, Accel)
    assert isinstance(trainer.strategy, Strat)
    assert isinstance(trainer.precision_plugin, Prec)
    assert trainer._accelerator_connector.strategy is strategy

@RunIf(mps=False)
def test_interactive_incompatible_backend_error(cuda_count_2, monkeypatch):
    if False:
        i = 10
        return i + 15
    monkeypatch.setattr(lightning.pytorch.trainer.connectors.accelerator_connector, '_IS_INTERACTIVE', True)
    with pytest.raises(MisconfigurationException, match="strategy='ddp'\\)`.*is not compatible"):
        Trainer(strategy='ddp', accelerator='gpu', devices=2)
    with pytest.raises(MisconfigurationException, match="strategy='ddp_spawn'\\)`.*is not compatible"):
        Trainer(strategy='ddp_spawn', accelerator='gpu', devices=2)

@RunIf(skip_windows=True)
def test_interactive_compatible_strategy_ddp_fork(monkeypatch):
    if False:
        i = 10
        return i + 15
    monkeypatch.setattr(lightning.pytorch.trainer.connectors.accelerator_connector, '_IS_INTERACTIVE', True)
    trainer = Trainer(strategy='ddp_fork', accelerator='cpu')
    assert trainer.strategy.launcher.is_interactive_compatible

@RunIf(mps=False)
@pytest.mark.parametrize(('strategy', 'strategy_class'), [('ddp', DDPStrategy), ('ddp_spawn', DDPStrategy), pytest.param('deepspeed', DeepSpeedStrategy, marks=RunIf(deepspeed=True))])
@pytest.mark.parametrize('devices', [1, 2])
def test_accelerator_choice_multi_node_gpu(cuda_count_2, tmpdir, strategy, strategy_class, devices):
    if False:
        print('Hello World!')
    trainer = Trainer(default_root_dir=tmpdir, num_nodes=2, accelerator='gpu', strategy=strategy, devices=devices)
    assert isinstance(trainer.strategy, strategy_class)

def test_accelerator_cpu(cuda_count_0, mps_count_0):
    if False:
        while True:
            i = 10
    trainer = Trainer(accelerator='cpu')
    assert isinstance(trainer.accelerator, CPUAccelerator)
    trainer = Trainer(devices=1)
    assert isinstance(trainer.accelerator, CPUAccelerator)
    with pytest.raises(MisconfigurationException, match='CUDAAccelerator` can not run on your system since the accelerator is not available.'):
        Trainer(accelerator='cuda')

@pytest.mark.parametrize('device_count', [['0'], [0, '1'], ['GPU'], [['0', '1'], [0, 1]], [False]])
def test_accelerator_invalid_type_devices(cuda_count_2, device_count):
    if False:
        return 10
    with pytest.raises(TypeError, match='must be an int, a string, a sequence of ints, but you'):
        _ = Trainer(accelerator='gpu', devices=device_count)

@RunIf(min_cuda_gpus=1)
def test_accelerator_gpu():
    if False:
        while True:
            i = 10
    trainer = Trainer(accelerator='gpu', devices=1)
    assert isinstance(trainer.accelerator, CUDAAccelerator)
    trainer = Trainer(accelerator='gpu')
    assert isinstance(trainer.accelerator, CUDAAccelerator)
    trainer = Trainer(accelerator='auto', devices=1)
    assert isinstance(trainer.accelerator, CUDAAccelerator)

@pytest.mark.parametrize(('devices', 'strategy_class'), [(1, SingleDeviceStrategy), (5, DDPStrategy)])
def test_accelerator_cpu_with_devices(devices, strategy_class):
    if False:
        print('Hello World!')
    trainer = Trainer(accelerator='cpu', devices=devices)
    assert trainer.num_devices == devices
    assert isinstance(trainer.strategy, strategy_class)
    assert isinstance(trainer.accelerator, CPUAccelerator)

@RunIf(min_cuda_gpus=2)
@pytest.mark.parametrize(('devices', 'strategy_class'), [(1, SingleDeviceStrategy), ([1], SingleDeviceStrategy), (2, DDPStrategy)])
def test_accelerator_gpu_with_devices(devices, strategy_class):
    if False:
        for i in range(10):
            print('nop')
    trainer = Trainer(accelerator='gpu', devices=devices)
    assert trainer.num_devices == len(devices) if isinstance(devices, list) else devices
    assert isinstance(trainer.strategy, strategy_class)
    assert isinstance(trainer.accelerator, CUDAAccelerator)

@RunIf(min_cuda_gpus=1)
def test_accelerator_auto_with_devices_gpu():
    if False:
        for i in range(10):
            print('nop')
    trainer = Trainer(accelerator='auto', devices=1)
    assert isinstance(trainer.accelerator, CUDAAccelerator)
    assert trainer.num_devices == 1

def test_set_devices_if_none_cpu():
    if False:
        return 10
    trainer = Trainer(accelerator='cpu', devices=3)
    assert trainer.num_devices == 3

@pytest.mark.parametrize(('strategy', 'strategy_class'), [('ddp_spawn', DDPStrategy), ('ddp_spawn_find_unused_parameters_false', DDPStrategy), ('ddp_spawn_find_unused_parameters_true', DDPStrategy), ('ddp', DDPStrategy), ('ddp_find_unused_parameters_false', DDPStrategy), ('ddp_find_unused_parameters_true', DDPStrategy), pytest.param('deepspeed', DeepSpeedStrategy, marks=RunIf(deepspeed=True))])
@pytest.mark.parametrize('accelerator', ['mps', 'auto', 'gpu', MPSAccelerator()])
def test_invalid_ddp_strategy_with_mps(accelerator, strategy, strategy_class, mps_count_1, cuda_count_0):
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError, match='strategies from the DDP family are not supported'):
        Trainer(accelerator=accelerator, strategy=strategy)
    with pytest.raises(ValueError, match='strategies from the DDP family are not supported'):
        Trainer(accelerator='mps', strategy=strategy_class())

@pytest.mark.parametrize(('strategy', 'strategy_class'), [('ddp_spawn', DDPStrategy), ('ddp_spawn_find_unused_parameters_false', DDPStrategy), ('ddp', DDPStrategy), ('ddp_find_unused_parameters_false', DDPStrategy)])
def test_strategy_choice_cpu_str(strategy, strategy_class):
    if False:
        while True:
            i = 10
    trainer = Trainer(strategy=strategy, accelerator='cpu', devices=2)
    assert isinstance(trainer.strategy, strategy_class)

def test_strategy_choice_cpu_instance():
    if False:
        for i in range(10):
            print('nop')
    trainer = Trainer(strategy=DDPStrategy(), accelerator='cpu', devices=2)
    assert isinstance(trainer.strategy, DDPStrategy)

@pytest.mark.parametrize(('strategy', 'strategy_class'), [('ddp_spawn', DDPStrategy), ('ddp_spawn_find_unused_parameters_false', DDPStrategy), ('ddp', DDPStrategy), ('ddp_find_unused_parameters_false', DDPStrategy), pytest.param('deepspeed', DeepSpeedStrategy, marks=RunIf(deepspeed=True))])
def test_strategy_choice_gpu_str(strategy, strategy_class, cuda_count_2, mps_count_0):
    if False:
        return 10
    trainer = Trainer(strategy=strategy, accelerator='gpu', devices=2)
    assert isinstance(trainer.strategy, strategy_class)

def test_strategy_choice_gpu_instance(cuda_count_2, mps_count_0):
    if False:
        print('Hello World!')
    trainer = Trainer(strategy=DDPStrategy(), accelerator='gpu', devices=2)
    assert isinstance(trainer.strategy, DDPStrategy)

def test_device_type_when_strategy_instance_gpu_passed(cuda_count_2, mps_count_0):
    if False:
        return 10
    trainer = Trainer(strategy=DDPStrategy(), accelerator='gpu', devices=2)
    assert isinstance(trainer.strategy, DDPStrategy)
    assert isinstance(trainer.accelerator, CUDAAccelerator)

@pytest.mark.parametrize('precision', [1, 12, 'invalid'])
def test_validate_precision_type(precision):
    if False:
        return 10
    with pytest.raises(ValueError, match=f'Precision {repr(precision)} is invalid'):
        Trainer(precision=precision)

def test_strategy_choice_ddp_spawn_cpu():
    if False:
        print('Hello World!')
    trainer = Trainer(strategy='ddp_spawn', accelerator='cpu', devices=2)
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert isinstance(trainer.strategy, DDPStrategy)
    assert isinstance(trainer.strategy.cluster_environment, LightningEnvironment)
    assert trainer.strategy.launcher._start_method == 'spawn'

@RunIf(skip_windows=True)
@mock.patch('lightning.pytorch.trainer.connectors.accelerator_connector._IS_INTERACTIVE', True)
def test_strategy_choice_ddp_fork_in_interactive():
    if False:
        while True:
            i = 10
    'Test that when strategy is unspecified, the connector chooses DDP Fork in interactive environments by\n    default.'
    trainer = Trainer(accelerator='cpu', devices=2)
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert isinstance(trainer.strategy, DDPStrategy)
    assert isinstance(trainer.strategy.cluster_environment, LightningEnvironment)
    assert trainer.strategy.launcher._start_method == 'fork'

@RunIf(skip_windows=True)
def test_strategy_choice_ddp_fork_cpu():
    if False:
        return 10
    trainer = Trainer(strategy='ddp_fork', accelerator='cpu', devices=2)
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert isinstance(trainer.strategy, DDPStrategy)
    assert isinstance(trainer.strategy.cluster_environment, LightningEnvironment)
    assert trainer.strategy.launcher._start_method == 'fork'

@pytest.mark.parametrize(('strategy', 'expected_cls'), [('ddp', DDPStrategy), ('ddp_spawn', DDPStrategy)])
def test_strategy_choice_ddp_cuda(strategy, expected_cls, mps_count_0, cuda_count_2):
    if False:
        for i in range(10):
            print('nop')
    trainer = Trainer(fast_dev_run=True, strategy=strategy, accelerator='gpu', devices=1)
    assert isinstance(trainer.accelerator, CUDAAccelerator)
    assert isinstance(trainer.strategy, expected_cls)
    assert isinstance(trainer.strategy.cluster_environment, LightningEnvironment)

@pytest.mark.parametrize(('job_name', 'expected_env'), [('some_name', SLURMEnvironment), ('bash', LightningEnvironment)])
@pytest.mark.parametrize('strategy', ['auto', 'ddp', DDPStrategy])
def test_strategy_choice_ddp_slurm(cuda_count_2, strategy, job_name, expected_env):
    if False:
        for i in range(10):
            print('nop')
    if strategy and (not isinstance(strategy, str)):
        strategy = strategy()
    with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1', 'SLURM_NTASKS': '2', 'SLURM_NTASKS_PER_NODE': '1', 'SLURM_JOB_NAME': job_name, 'SLURM_NODEID': '0', 'SLURM_PROCID': '1', 'SLURM_LOCALID': '1'}):
        trainer = Trainer(fast_dev_run=True, strategy=strategy, accelerator='cuda', devices=2)
        assert isinstance(trainer.accelerator, CUDAAccelerator)
        assert isinstance(trainer.strategy, DDPStrategy)
        assert isinstance(trainer.strategy.cluster_environment, expected_env)

@mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1', 'WORLD_SIZE': '2', 'LOCAL_WORLD_SIZE': '2', 'RANK': '1', 'LOCAL_RANK': '1', 'GROUP_RANK': '0', 'TORCHELASTIC_RUN_ID': '1'})
@mock.patch('torch.cuda.set_device')
@mock.patch('lightning.pytorch.strategies.DDPStrategy.setup_distributed', autospec=True)
def test_strategy_choice_ddp_torchelastic(_, __, mps_count_0, cuda_count_2):
    if False:
        i = 10
        return i + 15
    trainer = Trainer(fast_dev_run=True, accelerator='gpu', devices=2)
    assert isinstance(trainer.accelerator, CUDAAccelerator)
    assert isinstance(trainer.strategy, DDPStrategy)
    assert isinstance(trainer.strategy.cluster_environment, TorchElasticEnvironment)
    assert trainer.strategy.cluster_environment.local_rank() == 1
    assert trainer.strategy.local_rank == 1

@mock.patch.dict(os.environ, {'TORCHELASTIC_RUN_ID': '1', 'SLURM_NTASKS': '2', 'WORLD_SIZE': '2', 'RANK': '1', 'LOCAL_RANK': '1'})
@mock.patch('lightning.fabric.accelerators.cuda.num_cuda_devices', return_value=2)
@mock.patch('lightning.fabric.accelerators.mps.MPSAccelerator.is_available', return_value=False)
def test_torchelastic_priority_over_slurm(*_):
    if False:
        i = 10
        return i + 15
    'Test that the TorchElastic cluster environment is chosen over SLURM when both are detected.'
    assert TorchElasticEnvironment.detect()
    assert SLURMEnvironment.detect()
    connector = _AcceleratorConnector(strategy='ddp')
    assert isinstance(connector.strategy.cluster_environment, TorchElasticEnvironment)

@mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0', 'KUBERNETES_PORT': 'tcp://127.0.0.1:443', 'MASTER_ADDR': '1.2.3.4', 'MASTER_PORT': '500', 'WORLD_SIZE': '20', 'RANK': '1'})
@mock.patch('torch.cuda.set_device')
@mock.patch('lightning.pytorch.strategies.DDPStrategy.setup_distributed', autospec=True)
def test_strategy_choice_ddp_kubeflow(_, __, mps_count_0, cuda_count_2):
    if False:
        for i in range(10):
            print('nop')
    trainer = Trainer(fast_dev_run=True, accelerator='gpu', devices=2, plugins=KubeflowEnvironment())
    assert isinstance(trainer.accelerator, CUDAAccelerator)
    assert isinstance(trainer.strategy, DDPStrategy)
    assert isinstance(trainer.strategy.cluster_environment, KubeflowEnvironment)
    assert trainer.strategy.cluster_environment.local_rank() == 0
    assert trainer.strategy.local_rank == 0

@mock.patch.dict(os.environ, {'KUBERNETES_PORT': 'tcp://127.0.0.1:443', 'MASTER_ADDR': '1.2.3.4', 'MASTER_PORT': '500', 'WORLD_SIZE': '20', 'RANK': '1'})
@mock.patch('lightning.pytorch.strategies.DDPStrategy.setup_distributed', autospec=True)
def test_strategy_choice_ddp_cpu_kubeflow(cuda_count_0):
    if False:
        while True:
            i = 10
    trainer = Trainer(fast_dev_run=True, accelerator='cpu', devices=2, plugins=KubeflowEnvironment())
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert isinstance(trainer.strategy, DDPStrategy)
    assert isinstance(trainer.strategy.cluster_environment, KubeflowEnvironment)
    assert trainer.strategy.cluster_environment.local_rank() == 0
    assert trainer.strategy.local_rank == 0

@mock.patch.dict(os.environ, {'SLURM_NTASKS': '2', 'SLURM_NTASKS_PER_NODE': '1', 'SLURM_JOB_NAME': 'SOME_NAME', 'SLURM_NODEID': '0', 'LOCAL_RANK': '0', 'SLURM_PROCID': '0', 'SLURM_LOCALID': '0'})
@mock.patch('lightning.pytorch.strategies.DDPStrategy.setup_distributed', autospec=True)
@pytest.mark.parametrize('strategy', ['auto', 'ddp', DDPStrategy()])
def test_strategy_choice_ddp_cpu_slurm(cuda_count_0, strategy):
    if False:
        return 10
    trainer = Trainer(fast_dev_run=True, strategy=strategy, accelerator='cpu', devices=2)
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert isinstance(trainer.strategy, DDPStrategy)
    assert isinstance(trainer.strategy.cluster_environment, SLURMEnvironment)
    assert trainer.strategy.local_rank == 0

def test_check_fsdp_strategy_and_fallback():
    if False:
        return 10
    with pytest.raises(MisconfigurationException, match=f'You selected strategy to be `{FSDPStrategy.strategy_name}`, but GPU accelerator is not used.'):
        Trainer(accelerator='cpu', strategy='fsdp')

@mock.patch.dict(os.environ, {}, clear=True)
def test_unsupported_tpu_choice(xla_available, tpu_available):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError, match='XLAAccelerator` can only be used with a `SingleDeviceXLAStrategy`'):
        Trainer(accelerator='tpu', precision='16-true', strategy='ddp')

def mock_ipu_available(monkeypatch, value=True):
    if False:
        print('Hello World!')
    try:
        import lightning_graphcore
    except ModuleNotFoundError:
        return
    monkeypatch.setattr(lightning_graphcore.accelerator, '_IPU_AVAILABLE', value)
    monkeypatch.setattr(lightning_graphcore.strategy, '_IPU_AVAILABLE', value)
if _LIGHTNING_HABANA_AVAILABLE:
    from lightning_habana import HPUAccelerator, HPUParallelStrategy, SingleHPUStrategy
else:

    class HPUAccelerator(Mock):

        @staticmethod
        def is_available():
            if False:
                print('Hello World!')
            return True

        @classmethod
        def register_accelerators(cls, registry):
            if False:
                return 10
            registry.register('hpu', cls)

        @staticmethod
        def parse_devices(devices):
            if False:
                while True:
                    i = 10
            return int(devices)

        @staticmethod
        def get_parallel_devices(devices):
            if False:
                print('Hello World!')
            return [torch.device('hpu')] * devices

    class SingleHPUStrategy(SingleDeviceStrategy):
        strategy_name = 'hpu_single'

        @classmethod
        def register_strategies(cls, registry):
            if False:
                while True:
                    i = 10
            registry.register(cls.strategy_name, cls)

    class HPUParallelStrategy(SingleDeviceStrategy):
        strategy_name = 'hpu_parallel'

        @classmethod
        def register_strategies(cls, registry):
            if False:
                for i in range(10):
                    print('nop')
            registry.register(cls.strategy_name, cls)

class MockHPUPrecisionPlugin(Mock):
    pass

def mock_hpu_count(monkeypatch, n=1):
    if False:
        for i in range(10):
            print('nop')
    if _LIGHTNING_HABANA_AVAILABLE:
        import lightning_habana
        from lightning_habana.pytorch.accelerator import HPUAccelerator
        monkeypatch.setattr(lightning_habana.HPUAccelerator, 'auto_device_count', lambda *_: n)
        monkeypatch.setattr(lightning_habana.HPUAccelerator, 'is_available', lambda *_: n > 0)
        monkeypatch.setattr(lightning_habana, 'HPUPrecisionPlugin', MockHPUPrecisionPlugin)
    else:
        monkeypatch.setattr('lightning.pytorch.trainer.connectors.accelerator_connector._habana_available_and_importable', lambda : n > 0)
        if n < 1:
            return
        habana_mock = Mock()
        global HPUAccelerator
        HPUAccelerator.auto_device_count = lambda *_: n
        habana_mock.HPUAccelerator = HPUAccelerator
        habana_mock.SingleHPUStrategy = SingleHPUStrategy
        habana_mock.HPUParallelStrategy = HPUParallelStrategy
        habana_mock.HPUPrecisionPlugin = MockHPUPrecisionPlugin
        monkeypatch.setitem(sys.modules, 'lightning_habana', habana_mock)

def test_devices_auto_choice_cpu(monkeypatch, cuda_count_0):
    if False:
        for i in range(10):
            print('nop')
    mock_hpu_count(monkeypatch, 0)
    mock_ipu_available(monkeypatch, False)
    mock_xla_available(monkeypatch, False)
    trainer = Trainer(accelerator='auto', devices='auto')
    assert trainer.num_devices == 1

@pytest.mark.parametrize(('parallel_devices', 'accelerator'), [([torch.device('cpu')], 'cuda'), ([torch.device('cuda', i) for i in range(8)], 'tpu')])
def test_parallel_devices_in_strategy_confilict_with_accelerator(parallel_devices, accelerator):
    if False:
        print('Hello World!')
    with pytest.raises(MisconfigurationException, match='parallel_devices set through'):
        Trainer(strategy=DDPStrategy(parallel_devices=parallel_devices), accelerator=accelerator)

@pytest.mark.parametrize('deterministic', [None, True, False, 'warn'])
@mock.patch.dict(os.environ, {}, clear=True)
def test_deterministic_init(deterministic):
    if False:
        while True:
            i = 10
    with mock.patch('torch.use_deterministic_algorithms') as use_deterministic_patch:
        _set_torch_flags(deterministic=deterministic)
    if deterministic == 'warn':
        use_deterministic_patch.assert_called_once_with(True, warn_only=True)
    elif deterministic is None:
        use_deterministic_patch.assert_not_called()
    else:
        use_deterministic_patch.assert_called_once_with(deterministic)
    if deterministic:
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'

@pytest.mark.parametrize('cudnn_benchmark', [False, True])
@pytest.mark.parametrize(('benchmark_', 'deterministic', 'expected'), [(None, False, None), (None, True, False), (None, None, None), (True, False, True), (True, True, True), (True, None, True), (False, False, False), (False, True, False), (False, None, False)])
def test_benchmark_option(cudnn_benchmark, benchmark_, deterministic, expected):
    if False:
        while True:
            i = 10
    'Verify benchmark option.'
    original_val = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = cudnn_benchmark
    if benchmark_ and deterministic:
        with pytest.warns(UserWarning, match='You passed `deterministic=True` and `benchmark=True`'):
            _AcceleratorConnector(benchmark=benchmark_, deterministic=deterministic)
    else:
        _AcceleratorConnector(benchmark=benchmark_, deterministic=deterministic)
    expected = cudnn_benchmark if expected is None else expected
    assert torch.backends.cudnn.benchmark == expected
    torch.backends.cudnn.benchmark = original_val

@pytest.mark.parametrize(('sync_batchnorm', 'plugins', 'expected'), [(False, [], type(None)), (True, [], TorchSyncBatchNorm), (False, [TorchSyncBatchNorm()], TorchSyncBatchNorm), (True, [TorchSyncBatchNorm()], TorchSyncBatchNorm), (False, [Mock(spec=LayerSync)], LayerSync)])
def test_sync_batchnorm_set(sync_batchnorm, plugins, expected):
    if False:
        while True:
            i = 10
    'Test valid combinations of the sync_batchnorm Trainer flag and the plugins list of layer-sync plugins.'
    trainer = Trainer(accelerator='cpu', sync_batchnorm=sync_batchnorm, plugins=plugins, strategy='ddp')
    assert isinstance(trainer._accelerator_connector._layer_sync, expected)
    assert isinstance(trainer.strategy._layer_sync, expected)

def test_sync_batchnorm_invalid_choice():
    if False:
        print('Hello World!')
    'Test that a conflicting specification of enabled sync batchnorm and a custom plugin leads to an error.'
    custom = Mock(spec=LayerSync)
    with pytest.raises(MisconfigurationException, match='You set `Trainer\\(sync_batchnorm=True\\)` and provided a `LayerSync` plugin, but this is not allowed'):
        Trainer(sync_batchnorm=True, plugins=[custom])

@RunIf(skip_windows=True)
def test_sync_batchnorm_set_in_custom_strategy():
    if False:
        return 10
    'Tests if layer_sync is automatically set for custom strategy.'

    class CustomParallelStrategy(DDPStrategy):

        def __init__(self, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__(**kwargs)
            self._layer_sync = None
    strategy = CustomParallelStrategy()
    assert strategy._layer_sync is None
    Trainer(accelerator='cpu', strategy=strategy, sync_batchnorm=True)
    assert isinstance(strategy._layer_sync, TorchSyncBatchNorm)

@pytest.mark.parametrize(('plugins', 'expected'), [([LightningEnvironment(), SLURMEnvironment()], 'ClusterEnvironment'), ([TorchCheckpointIO(), TorchCheckpointIO()], 'CheckpointIO'), ([Precision(), DoublePrecision(), LightningEnvironment(), SLURMEnvironment()], 'Precision, ClusterEnvironment')])
def test_plugin_only_one_instance_for_one_type(plugins, expected):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(MisconfigurationException, match=f'Received multiple values for {expected}'):
        Trainer(plugins=plugins)

@pytest.mark.parametrize('accelerator', ['cpu', 'cuda', 'mps', 'tpu'])
@pytest.mark.parametrize('devices', ['0', 0, []])
def test_passing_zero_and_empty_list_to_devices_flag(accelerator, devices):
    if False:
        print('Hello World!')
    with pytest.raises(MisconfigurationException, match='value is not a valid input using'):
        Trainer(accelerator=accelerator, devices=devices)

@pytest.mark.parametrize(('expected_accelerator_flag', 'expected_accelerator_class'), [pytest.param('cuda', CUDAAccelerator, marks=RunIf(min_cuda_gpus=1)), pytest.param('mps', MPSAccelerator, marks=RunIf(mps=True))])
def test_gpu_accelerator_backend_choice(expected_accelerator_flag, expected_accelerator_class):
    if False:
        return 10
    trainer = Trainer(accelerator='gpu')
    assert trainer._accelerator_connector._accelerator_flag == expected_accelerator_flag
    assert isinstance(trainer.accelerator, expected_accelerator_class)

@RunIf(mps=False)
def test_gpu_accelerator_backend_choice_cuda(cuda_count_1):
    if False:
        print('Hello World!')
    trainer = Trainer(accelerator='gpu')
    assert trainer._accelerator_connector._accelerator_flag == 'cuda'
    assert isinstance(trainer.accelerator, CUDAAccelerator)

@RunIf(min_python='3.9')
def test_gpu_accelerator_backend_choice_mps(mps_count_1, cuda_count_0):
    if False:
        print('Hello World!')
    trainer = Trainer(accelerator='gpu')
    assert trainer._accelerator_connector._accelerator_flag == 'mps'
    assert isinstance(trainer.accelerator, MPSAccelerator)

@mock.patch('lightning.pytorch.accelerators.mps.MPSAccelerator.is_available', return_value=False)
@mock.patch('lightning.pytorch.accelerators.cuda.CUDAAccelerator.is_available', return_value=False)
def test_gpu_accelerator_misconfiguration_exception(*_):
    if False:
        while True:
            i = 10
    with pytest.raises(MisconfigurationException, match='No supported gpu backend found!'):
        Trainer(accelerator='gpu')

def test_accelerator_specific_checkpoint_io():
    if False:
        while True:
            i = 10
    ckpt_plugin = TorchCheckpointIO()
    trainer = Trainer(accelerator='cpu', strategy=DDPStrategy(), plugins=[ckpt_plugin])
    assert trainer.strategy.checkpoint_io is ckpt_plugin

@pytest.mark.parametrize('strategy', _DDP_FORK_ALIASES)
@mock.patch('lightning.pytorch.trainer.connectors.accelerator_connector.torch.multiprocessing.get_all_start_methods', return_value=[])
def test_ddp_fork_on_unsupported_platform(_, strategy):
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError, match='process forking is not supported on this platform'):
        Trainer(accelerator='cpu', strategy=strategy)

@pytest.mark.parametrize(('strategy', 'strategy_cls'), [('DDP', DDPStrategy), ('DDP_FIND_UNUSED_PARAMETERS_FALSE', DDPStrategy)])
def test_strategy_str_passed_being_case_insensitive(strategy, strategy_cls):
    if False:
        i = 10
        return i + 15
    trainer = Trainer(accelerator='cpu', strategy=strategy)
    assert isinstance(trainer.strategy, strategy_cls)

def test_connector_defaults_match_trainer_defaults():
    if False:
        return 10
    'Test that the default values for the init arguments of AcceleratorConnector match the ones in Trainer.'

    def get_defaults(cls):
        if False:
            for i in range(10):
                print('nop')
        init_signature = inspect.signature(cls)
        return {k: v.default for (k, v) in init_signature.parameters.items()}
    trainer_defaults = get_defaults(Trainer)
    connector_defaults = get_defaults(_AcceleratorConnector)
    for (name, connector_default) in connector_defaults.items():
        assert connector_default == trainer_defaults[name]

@RunIf(min_cuda_gpus=1)
@pytest.mark.xfail(raises=ImportError, reason='Not updated to latest API')
@pytest.mark.skipif(not package_available('lightning_colossalai'), reason='Requires Colossal AI Strategy')
def test_colossalai_external_strategy(monkeypatch):
    if False:
        return 10
    with mock.patch('lightning.pytorch.trainer.connectors.accelerator_connector._LIGHTNING_COLOSSALAI_AVAILABLE', False), pytest.raises(ModuleNotFoundError):
        Trainer(strategy='colossalai')
    from lightning_colossalai import ColossalAIStrategy
    trainer = Trainer(strategy='colossalai', precision='16-mixed')
    assert isinstance(trainer.strategy, ColossalAIStrategy)

@RunIf(min_cuda_gpus=1)
@pytest.mark.xfail(raises=ImportError, reason='Not updated to latest API')
@pytest.mark.skipif(not package_available('lightning_bagua'), reason='Requires Bagua Strategy')
def test_bagua_external_strategy(monkeypatch):
    if False:
        i = 10
        return i + 15
    with mock.patch('lightning.pytorch.trainer.connectors.accelerator_connector._LIGHTNING_BAGUA_AVAILABLE', False), pytest.raises(ModuleNotFoundError):
        Trainer(strategy='bagua')
    from lightning_bagua import BaguaStrategy
    trainer = Trainer(strategy='bagua')
    assert isinstance(trainer.strategy, BaguaStrategy)

class DeviceMock(Mock):

    def __instancecheck__(self, instance):
        if False:
            while True:
                i = 10
        return True

@RunIf(skip_windows=True)
def test_connector_with_tpu_accelerator_instance(tpu_available, monkeypatch):
    if False:
        while True:
            i = 10
    monkeypatch.setattr(torch, 'device', DeviceMock())
    accelerator = XLAAccelerator()
    trainer = Trainer(accelerator=accelerator, devices=1)
    assert trainer.accelerator is accelerator
    assert isinstance(trainer.strategy, SingleDeviceXLAStrategy)
    trainer = Trainer(accelerator=accelerator)
    assert trainer.accelerator is accelerator
    assert isinstance(trainer.strategy, XLAStrategy)

@pytest.mark.parametrize('is_interactive', [False, True])
@RunIf(min_python='3.9')
def test_connector_auto_selection(monkeypatch, is_interactive):
    if False:
        for i in range(10):
            print('nop')
    import lightning.fabric

    def _mock_interactive():
        if False:
            i = 10
            return i + 15
        monkeypatch.setattr(lightning.pytorch.trainer.connectors.accelerator_connector, '_IS_INTERACTIVE', is_interactive)
        if _IS_WINDOWS:
            monkeypatch.setattr(torch.multiprocessing, 'get_all_start_methods', lambda : ['fork', 'spawn'])
    _mock_interactive()

    def _mock_tpu_available(value):
        if False:
            i = 10
            return i + 15
        mock_tpu_available(monkeypatch, value)
        monkeypatch.setattr(lightning.fabric.plugins.environments.XLAEnvironment, 'node_rank', lambda *_: 0)
    with monkeypatch.context():
        mock_cuda_count(monkeypatch, 0)
        mock_mps_count(monkeypatch, 0)
        mock_tpu_available(monkeypatch, False)
        mock_ipu_available(monkeypatch, False)
        trainer = Trainer()
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert isinstance(trainer.strategy, SingleDeviceStrategy)
    assert trainer._accelerator_connector._devices_flag == 1
    assert trainer.num_devices == 1
    with monkeypatch.context():
        mock_cuda_count(monkeypatch, 1)
        mock_mps_count(monkeypatch, 0)
        mock_tpu_available(monkeypatch, False)
        mock_ipu_available(monkeypatch, False)
        trainer = Trainer()
    assert isinstance(trainer.accelerator, CUDAAccelerator)
    assert isinstance(trainer.strategy, SingleDeviceStrategy)
    assert trainer._accelerator_connector._devices_flag == [0]
    assert trainer.num_devices == 1
    with monkeypatch.context():
        mock_cuda_count(monkeypatch, 4)
        mock_mps_count(monkeypatch, 0)
        mock_tpu_available(monkeypatch, False)
        mock_ipu_available(monkeypatch, False)
        trainer = Trainer()
    assert isinstance(trainer.accelerator, CUDAAccelerator)
    assert isinstance(trainer.strategy, SingleDeviceStrategy if is_interactive else DDPStrategy)
    assert trainer._accelerator_connector._devices_flag == [0] if is_interactive else list(range(4))
    assert trainer.num_devices == 1 if is_interactive else 4
    if not is_interactive:
        assert isinstance(trainer.strategy.cluster_environment, LightningEnvironment)
        assert trainer.strategy._start_method == ('fork' if is_interactive else 'popen')
        assert trainer.strategy.launcher.is_interactive_compatible == is_interactive
    with monkeypatch.context():
        mock_cuda_count(monkeypatch, 0)
        mock_mps_count(monkeypatch, 1)
        mock_tpu_available(monkeypatch, False)
        mock_ipu_available(monkeypatch, False)
        connector = _AcceleratorConnector()
    assert isinstance(connector.accelerator, MPSAccelerator)
    assert isinstance(connector.strategy, SingleDeviceStrategy)
    assert connector._devices_flag == [0]
    with monkeypatch.context():
        mock_cuda_count(monkeypatch, 0)
        mock_mps_count(monkeypatch, 0)
        mock_ipu_available(monkeypatch, False)
        _mock_tpu_available(True)
        monkeypatch.setattr(lightning.pytorch.accelerators.XLAAccelerator, 'auto_device_count', lambda *_: 1)
        monkeypatch.setattr(torch, 'device', DeviceMock())
        connector = _AcceleratorConnector()
    assert isinstance(connector.accelerator, XLAAccelerator)
    assert isinstance(connector.strategy, SingleDeviceXLAStrategy)
    assert connector._devices_flag == 1
    monkeypatch.undo()
    _mock_interactive()
    with monkeypatch.context():
        mock_cuda_count(monkeypatch, 0)
        mock_mps_count(monkeypatch, 0)
        _mock_tpu_available(True)
        mock_ipu_available(monkeypatch, False)
        connector = _AcceleratorConnector()
    assert isinstance(connector.accelerator, XLAAccelerator)
    assert isinstance(connector.strategy, XLAStrategy)
    assert connector._devices_flag == 8
    assert isinstance(connector.strategy.cluster_environment, XLAEnvironment)
    assert connector.strategy._start_method == 'fork'
    assert connector.strategy.launcher.is_interactive_compatible
    if _graphcore_available_and_importable():
        with monkeypatch.context():
            mock_cuda_count(monkeypatch, 0)
            mock_mps_count(monkeypatch, 0)
            mock_tpu_available(monkeypatch, False)
            mock_ipu_available(monkeypatch, True)
            from lightning_graphcore import IPUAccelerator, IPUStrategy
            connector = _AcceleratorConnector()
        assert isinstance(connector.accelerator, IPUAccelerator)
        assert isinstance(connector.strategy, IPUStrategy)
        assert connector._devices_flag == 4
        assert isinstance(connector.strategy.cluster_environment, LightningEnvironment)
        assert connector.strategy.launcher is None
    with monkeypatch.context():
        mock_cuda_count(monkeypatch, 0)
        mock_mps_count(monkeypatch, 0)
        mock_tpu_available(monkeypatch, False)
        mock_ipu_available(monkeypatch, False)
        mock_hpu_count(monkeypatch, 1)
        connector = _AcceleratorConnector()
    assert isinstance(connector.accelerator, HPUAccelerator)
    assert isinstance(connector.strategy, SingleHPUStrategy)
    assert isinstance(connector.precision_plugin, MockHPUPrecisionPlugin)
    assert connector._devices_flag == 1
    monkeypatch.undo()
    _mock_interactive()
    if not is_interactive:
        with monkeypatch.context():
            mock_cuda_count(monkeypatch, 0)
            mock_mps_count(monkeypatch, 0)
            mock_tpu_available(monkeypatch, False)
            mock_ipu_available(monkeypatch, False)
            mock_hpu_count(monkeypatch, 8)
            connector = _AcceleratorConnector()
        assert isinstance(connector.accelerator, HPUAccelerator)
        assert isinstance(connector.strategy, HPUParallelStrategy)
        assert connector._devices_flag == 8
        if _LIGHTNING_HABANA_AVAILABLE:
            assert isinstance(connector.strategy.cluster_environment, LightningEnvironment)
            assert isinstance(connector.strategy.launcher, _SubprocessScriptLauncher)
            assert not connector.strategy.launcher.is_interactive_compatible
    with monkeypatch.context():
        mock_cuda_count(monkeypatch, 2)
        mock_mps_count(monkeypatch, 0)
        _mock_tpu_available(True)
        mock_ipu_available(monkeypatch, False)
        connector = _AcceleratorConnector()
    assert isinstance(connector.accelerator, XLAAccelerator)
    assert isinstance(connector.strategy, XLAStrategy)
    assert connector._devices_flag == 8
    assert isinstance(connector.strategy.cluster_environment, XLAEnvironment)
    assert connector.strategy._start_method == 'fork'
    assert connector.strategy.launcher.is_interactive_compatible

@pytest.mark.parametrize('strategy', ['ddp', 'ddp_spawn', pytest.param('deepspeed', marks=RunIf(deepspeed=True)), 'fsdp'])
def test_connector_sets_num_nodes(strategy, cuda_count_2):
    if False:
        while True:
            i = 10
    trainer = Trainer(accelerator='cuda', strategy=strategy, devices=2, num_nodes=2)
    assert trainer.strategy.num_nodes == 2

def test_connector_num_nodes_input_validation():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError, match='`num_nodes` must be a positive integer'):
        _AcceleratorConnector(num_nodes=0)
    with pytest.raises(ValueError, match='`num_nodes` must be a positive integer'):
        _AcceleratorConnector(num_nodes=-1)

@pytest.mark.parametrize(('precision_str', 'strategy_str', 'expected_precision_cls'), [('64-true', 'auto', DoublePrecision), ('32-true', 'auto', Precision), ('16-true', 'auto', HalfPrecision), ('bf16-true', 'auto', HalfPrecision), ('16-mixed', 'auto', MixedPrecision), ('bf16-mixed', 'auto', MixedPrecision), pytest.param('32-true', 'fsdp', FSDPPrecision, marks=RunIf(min_cuda_gpus=1)), pytest.param('16-true', 'fsdp', FSDPPrecision, marks=RunIf(min_cuda_gpus=1)), pytest.param('bf16-true', 'fsdp', FSDPPrecision, marks=RunIf(min_cuda_gpus=1)), pytest.param('16-mixed', 'fsdp', FSDPPrecision, marks=RunIf(min_cuda_gpus=1)), pytest.param('bf16-mixed', 'fsdp', FSDPPrecision, marks=RunIf(min_cuda_gpus=1)), pytest.param('32-true', 'deepspeed', DeepSpeedPrecision, marks=RunIf(deepspeed=True, mps=False)), pytest.param('16-true', 'deepspeed', DeepSpeedPrecision, marks=RunIf(deepspeed=True, mps=False)), pytest.param('bf16-true', 'deepspeed', DeepSpeedPrecision, marks=RunIf(deepspeed=True, mps=False)), pytest.param('16-mixed', 'deepspeed', DeepSpeedPrecision, marks=RunIf(deepspeed=True, mps=False)), pytest.param('bf16-mixed', 'deepspeed', DeepSpeedPrecision, marks=RunIf(deepspeed=True, mps=False))])
def test_precision_selection(precision_str, strategy_str, expected_precision_cls):
    if False:
        i = 10
        return i + 15
    connector = _AcceleratorConnector(precision=precision_str, strategy=strategy_str)
    assert isinstance(connector.precision_plugin, expected_precision_cls)