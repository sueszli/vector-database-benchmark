import contextlib
import os
import tempfile
import time
import uuid
from unittest.mock import patch
import pytest
import torch
import ray
import ray.train as train
from ray.cluster_utils import Cluster
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.examples.pytorch.torch_linear_example import train_func as linear_train_func
from ray.train.torch import TorchCheckpoint, TorchConfig, TorchPredictor, TorchTrainer
from ray.train.trainer import TrainingFailedError

@pytest.fixture
def ray_start_4_cpus():
    if False:
        i = 10
        return i + 15
    address_info = ray.init(num_cpus=4)
    yield address_info
    ray.shutdown()

@contextlib.contextmanager
def ray_start_2_node_cluster(num_cpus_per_node: int, num_gpus_per_node: int):
    if False:
        i = 10
        return i + 15
    cluster = Cluster()
    for _ in range(2):
        cluster.add_node(num_cpus=num_cpus_per_node, num_gpus=num_gpus_per_node)
    ray.init(address=cluster.address)
    yield
    ray.shutdown()
    cluster.shutdown()

@pytest.mark.parametrize('num_workers', [1, 2])
def test_torch_linear(ray_start_4_cpus, num_workers):
    if False:
        for i in range(10):
            print('nop')

    def train_func(config):
        if False:
            return 10
        result = linear_train_func(config)
        assert len(result) == epochs
        assert result[-1]['loss'] < result[0]['loss']
    num_workers = num_workers
    epochs = 3
    scaling_config = ScalingConfig(num_workers=num_workers)
    config = {'lr': 0.01, 'hidden_size': 1, 'batch_size': 4, 'epochs': epochs}
    trainer = TorchTrainer(train_loop_per_worker=train_func, train_loop_config=config, scaling_config=scaling_config)
    trainer.fit()

@pytest.mark.parametrize('prepare_model', (True, False))
def test_torch_e2e(ray_start_4_cpus, prepare_model):
    if False:
        print('Hello World!')

    def train_func():
        if False:
            print('Hello World!')
        model = torch.nn.Linear(3, 1)
        if prepare_model:
            model = train.torch.prepare_model(model)
        train.report({}, checkpoint=TorchCheckpoint.from_model(model))
    scaling_config = ScalingConfig(num_workers=2)
    trainer = TorchTrainer(train_loop_per_worker=train_func, scaling_config=scaling_config)
    trainer.fit()

@pytest.mark.parametrize('prepare_model', (True, False))
def test_torch_e2e_state_dict(ray_start_4_cpus, prepare_model):
    if False:
        i = 10
        return i + 15

    def train_func():
        if False:
            for i in range(10):
                print('nop')
        model = torch.nn.Linear(3, 1)
        if prepare_model:
            model = train.torch.prepare_model(model)
        train.report({}, checkpoint=TorchCheckpoint.from_state_dict(model.state_dict()))
    scaling_config = ScalingConfig(num_workers=2)
    trainer = TorchTrainer(train_loop_per_worker=train_func, scaling_config=scaling_config)
    result = trainer.fit()
    with pytest.raises(ValueError):
        torch_checkpoint = TorchCheckpoint(path=result.checkpoint.path, filesystem=result.checkpoint.filesystem)
        torch_checkpoint.get_model()

def test_torch_e2e_dir(ray_start_4_cpus):
    if False:
        i = 10
        return i + 15

    def train_func():
        if False:
            print('Hello World!')
        model = torch.nn.Linear(3, 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(model.state_dict(), os.path.join(tmpdir, 'model.pt'))
            train.report({}, checkpoint=Checkpoint.from_directory(tmpdir))
    scaling_config = ScalingConfig(num_workers=2)
    trainer = TorchTrainer(train_loop_per_worker=train_func, scaling_config=scaling_config)
    result = trainer.fit()

    class TorchScorer:

        def __init__(self, checkpoint: Checkpoint):
            if False:
                while True:
                    i = 10
            model = torch.nn.Linear(3, 1)
            with checkpoint.as_directory() as checkpoint_dir:
                state_dict = torch.load(os.path.join(checkpoint_dir, 'model.pt'))
                model.load_state_dict(state_dict)
            self.pred = TorchPredictor(model)

        def __call__(self, x):
            if False:
                return 10
            return self.pred.predict(x, dtype=torch.float)
    predict_dataset = ray.data.range(9)
    predictions = predict_dataset.map_batches(TorchScorer, batch_size=3, batch_format='pandas', compute=ray.data.ActorPoolStrategy(), fn_constructor_args=(result.checkpoint,))
    assert predictions.count() == 3

def test_checkpoint_freq(ray_start_4_cpus):
    if False:
        print('Hello World!')
    trainer = TorchTrainer(train_loop_per_worker=lambda config: None, scaling_config=train.ScalingConfig(num_workers=1), run_config=train.RunConfig(checkpoint_config=train.CheckpointConfig(checkpoint_frequency=2)))
    with pytest.raises(ValueError):
        trainer.fit()

def test_torch_session_errors(ray_start_4_cpus):
    if False:
        for i in range(10):
            print('nop')
    'Test fail-fast behavior when reporting dicts with Torch tensors'

    def train_func():
        if False:
            while True:
                i = 10
        model = torch.nn.Linear(1, 1).state_dict()
        with pytest.raises(ValueError):
            train.report(model)
    scaling_config = ScalingConfig(num_workers=2)
    trainer = TorchTrainer(train_loop_per_worker=train_func, scaling_config=scaling_config)
    trainer.fit()

def test_single_worker_failure(ray_start_4_cpus):
    if False:
        print('Hello World!')
    'Tests if training fails upon any worker failure.'

    def single_worker_fail():
        if False:
            print('Hello World!')
        if train.get_context().get_world_rank() == 0:
            raise RuntimeError
        else:
            time.sleep(1000000)
    scaling_config = ScalingConfig(num_workers=2)
    trainer = TorchTrainer(train_loop_per_worker=single_worker_fail, scaling_config=scaling_config)
    with pytest.raises(TrainingFailedError) as exc_info:
        trainer.fit()
    assert isinstance(exc_info.value.__cause__, RuntimeError)

@pytest.mark.parametrize('num_gpus_per_worker', [0.5, 1, 2])
def test_tune_torch_get_device_gpu(num_gpus_per_worker):
    if False:
        return 10
    'Tests if GPU ids are set correctly when running train concurrently in nested actors\n    (for example when used with Tune).\n    '
    from ray.train import ScalingConfig
    num_samples = 2
    num_workers = 2
    total_gpus_required = num_workers * num_gpus_per_worker * num_samples
    gpus_per_node = total_gpus_required // 2
    exception = None
    with ray_start_2_node_cluster(num_cpus_per_node=gpus_per_node, num_gpus_per_node=gpus_per_node):

        @patch('torch.cuda.is_available', lambda : True)
        def train_fn():
            if False:
                while True:
                    i = 10
            devices = train.torch.get_device()
            if isinstance(devices, list):
                assert sorted([device.index for device in devices]) == [0, 1]
            else:
                assert train.torch.get_device().index == 0

        @ray.remote(num_cpus=0)
        class TrialActor:

            def __init__(self, warmup_steps):
                if False:
                    while True:
                        i = 10
                self.trainer = TorchTrainer(train_fn, torch_config=TorchConfig(backend='gloo'), run_config=RunConfig(name=f'test_tune_torch_get_device_gpu_{uuid.uuid4()}'), scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=True, resources_per_worker={'CPU': 1, 'GPU': num_gpus_per_worker}, trainer_resources={'CPU': 0}, placement_strategy='STRICT_SPREAD'))

            def run(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.trainer.fit()
        try:
            actors = [TrialActor.remote(1) for _ in range(num_samples)]
            ray.get([actor.run.remote() for actor in actors])
        except Exception as exc:
            exception = exc
    if exception:
        raise exception

def test_torch_amp(ray_start_4_cpus):
    if False:
        i = 10
        return i + 15

    def train_fn():
        if False:
            print('Hello World!')
        train.torch.accelerate(amp=True)
        model = torch.nn.Linear(1, 1)
        model = train.torch.prepare_model(model)
        train.report({}, checkpoint=TorchCheckpoint.from_model(model))
    trainer = TorchTrainer(train_fn, scaling_config=ScalingConfig(num_workers=2))
    results = trainer.fit()
    assert results.checkpoint

def test_torch_amp_with_custom_get_state(ray_start_4_cpus):
    if False:
        i = 10
        return i + 15
    'Tests amp with a model that has a custom __getstate__ method defined.\n\n    See https://discuss.ray.io/t/ray-train-hangs-for-long-time/6333/7\n    '

    def train_fn():
        if False:
            print('Hello World!')
        train.torch.accelerate(amp=True)

        class CustomLinear(torch.nn.Linear):

            def __getstate__(self):
                if False:
                    i = 10
                    return i + 15
                return self.__dict__.copy()
        model = CustomLinear(1, 1)
        model = train.torch.prepare_model(model)
        train.report({}, checkpoint=TorchCheckpoint.from_state_dict(model.module.state_dict()))
    trainer = TorchTrainer(train_fn, scaling_config=ScalingConfig(num_workers=2))
    results = trainer.fit()
    assert results.checkpoint

def test_torch_env_vars(ray_start_4_cpus):
    if False:
        while True:
            i = 10
    'Check that env vars are set as expected.'

    def train_func(config):
        if False:
            return 10
        context = train.get_context()
        assert os.environ['LOCAL_RANK'] == str(context.get_local_rank())
        assert os.environ['RANK'] == str(context.get_world_rank())
        assert os.environ['LOCAL_WORLD_SIZE'] == str(context.get_local_world_size())
        assert os.environ['WORLD_SIZE'] == str(context.get_world_size())
        assert os.environ['NODE_RANK'] == str(context.get_node_rank())
        assert os.environ['ACCELERATE_TORCH_DEVICE'] == str(train.torch.get_device())
    num_workers = 1
    scaling_config = ScalingConfig(num_workers=num_workers)
    trainer = TorchTrainer(train_loop_per_worker=train_func, scaling_config=scaling_config)
    trainer.fit()

def test_nonserializable_train_function(ray_start_4_cpus):
    if False:
        i = 10
        return i + 15
    import threading
    lock = threading.Lock()

    def train_func():
        if False:
            print('Hello World!')
        print(lock)
    trainer = TorchTrainer(train_func)
    with pytest.raises(TypeError, match='.*was found to be non-serializable.*'):
        trainer.fit()
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', '-x', __file__]))