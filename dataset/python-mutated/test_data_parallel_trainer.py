import os
import time
from unittest.mock import patch
import pytest
import ray
from ray import train, tune
from ray.train import RunConfig, ScalingConfig
from ray.train._internal.backend_executor import BackendExecutor
from ray.train._internal.worker_group import WorkerGroup
from ray.train.backend import Backend, BackendConfig
from ray.train.data_parallel_trainer import DataParallelTrainer
from ray.train.tests.util import create_dict_checkpoint, load_dict_checkpoint
from ray.tune.callback import Callback
from ray.tune.tune_config import TuneConfig
from ray.tune.tuner import Tuner

@pytest.fixture
def ray_start_4_cpus():
    if False:
        return 10
    address_info = ray.init(num_cpus=4)
    yield address_info
    ray.shutdown()

@pytest.fixture
def ray_start_4_cpus_4_gpus_4_extra():
    if False:
        return 10
    address_info = ray.init(num_cpus=4, num_gpus=4, resources={'extra': 4})
    yield address_info
    ray.shutdown()

def gen_execute_single_async_special(special_f):
    if False:
        i = 10
        return i + 15

    def execute_single_async_special(self, i, f, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        assert len(self.workers) == 2
        if i == 0 and hasattr(self, 'should_fail') and self.should_fail:
            kwargs['train_func'] = special_f
        return self.workers[i].actor._RayTrainWorker__execute.options(name=f.__name__).remote(f, *args, **kwargs)
    return execute_single_async_special

def gen_new_backend_executor(special_f):
    if False:
        for i in range(10):
            print('nop')
    'Returns a BackendExecutor that runs special_f on worker 0 once.'

    class TestBackendExecutor(BackendExecutor):

        def __init__(self, *args, **kwargs):
            if False:
                print('Hello World!')
            super().__init__(*args, **kwargs)
            self._has_failed = False

        def start_training(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            special_execute = gen_execute_single_async_special(special_f)
            if not self._has_failed:
                self.worker_group.should_fail = True
                self._has_failed = True
            else:
                self.worker_group.should_fail = False
            with patch.object(WorkerGroup, 'execute_single_async', special_execute):
                super().start_training(*args, **kwargs)
    return TestBackendExecutor

class CaptureReportCallback(Callback):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.result_list = []

    def on_trial_result(self, iteration, trials, trial, result, **info):
        if False:
            i = 10
            return i + 15
        self.result_list.append(result)
scale_config = ScalingConfig(num_workers=2)

def test_fit_train(ray_start_4_cpus):
    if False:
        i = 10
        return i + 15

    def train_func():
        if False:
            while True:
                i = 10
        train.report({'loss': 1})
    trainer = DataParallelTrainer(train_loop_per_worker=train_func, scaling_config=scale_config)
    assert trainer.fit().metrics['loss'] == 1

def test_scaling_config(ray_start_4_cpus):
    if False:
        while True:
            i = 10

    def train_func():
        if False:
            i = 10
            return i + 15
        assert ray.available_resources()['CPU'] == 1
        train.report({'loss': 1})
    assert ray.available_resources()['CPU'] == 4
    trainer = DataParallelTrainer(train_loop_per_worker=train_func, scaling_config=ScalingConfig(num_workers=2))
    trainer.fit()

def test_fit_train_config(ray_start_4_cpus):
    if False:
        return 10

    def train_func(config):
        if False:
            while True:
                i = 10
        train.report({'loss': config['x']})
    trainer = DataParallelTrainer(train_loop_per_worker=train_func, scaling_config=scale_config, train_loop_config={'x': 100})
    assert trainer.fit().metrics['loss'] == 100

def test_datasets(ray_start_4_cpus):
    if False:
        return 10
    num_train_data = 10
    num_val_data = 6
    train_dataset = ray.data.range(num_train_data)
    val_dataset = ray.data.range(num_val_data)

    def get_dataset():
        if False:
            i = 10
            return i + 15
        train_dataset = train.get_dataset_shard('train')
        train_ds_count = len(list(train_dataset.iter_rows()))
        assert train_ds_count == num_train_data / scale_config.num_workers
        val_dataset = train.get_dataset_shard('val')
        val_ds_count = len(list(val_dataset.iter_rows()))
        assert val_ds_count == num_val_data / scale_config.num_workers
    trainer = DataParallelTrainer(train_loop_per_worker=get_dataset, scaling_config=scale_config, datasets={'train': train_dataset, 'val': val_dataset})
    trainer.fit()

def test_invalid_train_loop(ray_start_4_cpus):
    if False:
        return 10

    def train_loop(config, extra_arg):
        if False:
            while True:
                i = 10
        pass
    with pytest.raises(ValueError):
        DataParallelTrainer(train_loop_per_worker=train_loop)

def test_bad_return_in_train_loop(ray_start_4_cpus):
    if False:
        while True:
            i = 10
    'Test to check if returns from train loop are discarded.'

    class FailOnUnpickle:

        def __reduce__(self):
            if False:
                while True:
                    i = 10
            raise RuntimeError('Failing')

    def train_loop(config):
        if False:
            print('Hello World!')
        train.report({'loss': 1})
        return FailOnUnpickle()
    trainer = DataParallelTrainer(train_loop_per_worker=train_loop, scaling_config=scale_config)
    trainer.fit()

def test_tune(ray_start_4_cpus):
    if False:
        i = 10
        return i + 15

    def train_func(config):
        if False:
            i = 10
            return i + 15
        train.report({'loss': config['x']})
    trainer = DataParallelTrainer(train_loop_per_worker=train_func, train_loop_config={'x': 100}, scaling_config=scale_config)
    tuner = Tuner(trainer, param_space={'train_loop_config': {'x': tune.choice([200, 300])}}, tune_config=TuneConfig(num_samples=2))
    result_grid = tuner.fit()
    assert result_grid[0].metrics['loss'] in [200, 300]
    assert trainer._train_loop_config['x'] == 100

def test_scaling_config_validation(ray_start_4_cpus):
    if False:
        return 10

    def train_func(config):
        if False:
            while True:
                i = 10
        train.report({'loss': config['x']})
    trainer = DataParallelTrainer(train_loop_per_worker=train_func, train_loop_config={'x': 100})
    with pytest.raises(ValueError):
        trainer.fit()
    tuner = Tuner(trainer)
    with pytest.raises(ValueError):
        tuner.fit()
    tuner = Tuner(trainer, param_space={'scaling_config': ScalingConfig(num_workers=1)})
    results = tuner.fit()
    assert not results.errors

def test_fast_slow(ray_start_4_cpus):
    if False:
        return 10

    def train_func():
        if False:
            return 10
        for i in range(2):
            with create_dict_checkpoint({'epoch': i}) as checkpoint:
                train.report(dict(index=i), checkpoint=checkpoint)

    def train_slow():
        if False:
            while True:
                i = 10
        for i in range(2):
            with create_dict_checkpoint({'epoch': i}) as checkpoint:
                train.report(dict(index=i), checkpoint=checkpoint)
            time.sleep(5)
    new_backend_executor_cls = gen_new_backend_executor(train_slow)
    callback = CaptureReportCallback()

    class DataParallelTrainerPatched(DataParallelTrainer):
        _backend_executor_cls = new_backend_executor_cls
    trainer = DataParallelTrainerPatched(train_func, scaling_config=scale_config, run_config=RunConfig(callbacks=[callback]))
    results = trainer.fit()
    assert load_dict_checkpoint(results.checkpoint)['epoch'] == 1
    result_list = callback.result_list
    assert len(result_list) == 2

def test_mismatch_report(ray_start_4_cpus):
    if False:
        return 10

    def train_func():
        if False:
            return 10
        for _ in range(2):
            train.report(dict(loss=1))

    def train_mismatch():
        if False:
            for i in range(10):
                print('nop')
        train.report(dict(loss=1))
    new_backend_executor_cls = gen_new_backend_executor(train_mismatch)

    class DataParallelTrainerPatched(DataParallelTrainer):
        _backend_executor_cls = new_backend_executor_cls
    trainer = DataParallelTrainerPatched(train_func, scaling_config=scale_config)
    with pytest.raises(RuntimeError):
        trainer.fit()

def test_world_rank(ray_start_4_cpus, tmp_path):
    if False:
        while True:
            i = 10

    def train_func():
        if False:
            while True:
                i = 10
        world_rank = train.get_context().get_world_rank()
        (tmp_path / f'{world_rank}').touch()
        train.report(dict(world_rank=world_rank))
    trainer = DataParallelTrainer(train_func, scaling_config=scale_config)
    trainer.fit()
    created_files = list(tmp_path.glob('*'))
    assert len(created_files) == 2
    assert {int(file.name) for file in created_files} == {0, 1}

def test_gpu_requests(ray_start_4_cpus_4_gpus_4_extra, tmp_path):
    if False:
        for i in range(10):
            print('nop')

    def get_visible_devices_for_workers():
        if False:
            while True:
                i = 10
        return [file.read_text() for file in tmp_path.glob('*')]

    class CudaTestBackend(Backend):
        share_cuda_visible_devices = True

    class CudaTestConfig(BackendConfig):

        @property
        def backend_cls(self):
            if False:
                i = 10
                return i + 15
            return CudaTestBackend

    def get_resources():
        if False:
            i = 10
            return i + 15
        cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
        world_rank = train.get_context().get_world_rank()
        (tmp_path / f'{world_rank}').write_text(cuda_visible_devices)
        train.report(dict(devices=cuda_visible_devices))
    trainer = DataParallelTrainer(get_resources, backend_config=CudaTestConfig(), scaling_config=ScalingConfig(num_workers=2, use_gpu=False))
    trainer.fit()
    assert get_visible_devices_for_workers() == ['', '']
    trainer = DataParallelTrainer(get_resources, backend_config=CudaTestConfig(), scaling_config=ScalingConfig(num_workers=2, use_gpu=True))
    trainer.fit()
    visible_devices = get_visible_devices_for_workers()
    visible_devices = [','.join(sorted(r.split(','))) for r in visible_devices]
    assert visible_devices == ['0,1', '0,1']
    trainer = DataParallelTrainer(get_resources, backend_config=CudaTestConfig(), scaling_config=ScalingConfig(num_workers=2, use_gpu=True, resources_per_worker={'GPU': 0.1}))
    trainer.fit()
    visible_devices = get_visible_devices_for_workers()
    assert visible_devices == ['0', '0']
    trainer = DataParallelTrainer(get_resources, backend_config=CudaTestConfig(), scaling_config=ScalingConfig(num_workers=2, use_gpu=True, resources_per_worker={'GPU': 2}))
    trainer.fit()
    visible_devices = get_visible_devices_for_workers()
    visible_devices = [','.join(sorted(r.split(','))) for r in visible_devices]
    assert visible_devices == ['0,1,2,3', '0,1,2,3']
if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main(['-v', '-x', __file__]))