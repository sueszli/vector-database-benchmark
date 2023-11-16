import os
from pathlib import Path
from unittest.mock import patch
import pytest
import unittest
from typing import Optional
from sklearn.datasets import load_breast_cancer
from sklearn.utils import shuffle
import ray
from ray import train, tune
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.constants import RAY_CHDIR_TO_TRIAL_DIR
from ray.train.examples.pytorch.torch_linear_example import train_func as linear_train_func
from ray.data import Dataset, Datasource, ReadTask, from_pandas, read_datasource
from ray.data.block import BlockMetadata
from ray.train.data_parallel_trainer import DataParallelTrainer
from ray.train.torch import TorchTrainer
from ray.train.trainer import BaseTrainer
from ray.train.xgboost import XGBoostTrainer
from ray.tune import Callback, CLIReporter
from ray.tune.tune_config import TuneConfig
from ray.tune.tuner import Tuner

@pytest.fixture
def shutdown_only():
    if False:
        return 10
    yield None
    ray.shutdown()

@pytest.fixture
def chdir_tmpdir(tmpdir):
    if False:
        print('Hello World!')
    old_cwd = os.getcwd()
    os.chdir(tmpdir)
    yield tmpdir
    os.chdir(old_cwd)

class DummyTrainer(BaseTrainer):
    _scaling_config_allowed_keys = BaseTrainer._scaling_config_allowed_keys + ['num_workers', 'use_gpu', 'resources_per_worker', 'placement_strategy']

    def training_loop(self) -> None:
        if False:
            return 10
        for i in range(5):
            train.report({'step': i})

class FailingTrainer(DummyTrainer):

    def training_loop(self) -> None:
        if False:
            i = 10
            return i + 15
        raise RuntimeError('There is an error in trainer!')

class TestDatasource(Datasource):

    def __init__(self, do_shuffle: bool):
        if False:
            return 10
        self._shuffle = do_shuffle

    def prepare_read(self, parallelism: int, **read_args):
        if False:
            while True:
                i = 10
        import pyarrow as pa

        def load_data():
            if False:
                for i in range(10):
                    print('nop')
            data_raw = load_breast_cancer(as_frame=True)
            dataset_df = data_raw['data']
            dataset_df['target'] = data_raw['target']
            if self._shuffle:
                dataset_df = shuffle(dataset_df)
            return [pa.Table.from_pandas(dataset_df)]
        meta = BlockMetadata(num_rows=None, size_bytes=None, schema=None, input_files=None, exec_stats=None)
        return [ReadTask(load_data, meta)]

def gen_dataset_func(do_shuffle: Optional[bool]=False) -> Dataset:
    if False:
        return 10
    test_datasource = TestDatasource(do_shuffle)
    return read_datasource(test_datasource, parallelism=1)

def gen_dataset_func_eager():
    if False:
        i = 10
        return i + 15
    data_raw = load_breast_cancer(as_frame=True)
    dataset_df = data_raw['data']
    dataset_df['target'] = data_raw['target']
    dataset = from_pandas(dataset_df)
    return dataset

class TunerTest(unittest.TestCase):
    """The e2e test for hparam tuning using Tuner API."""

    @pytest.fixture(autouse=True)
    def local_dir(self, tmp_path, monkeypatch):
        if False:
            i = 10
            return i + 15
        monkeypatch.setenv('RAY_AIR_LOCAL_CACHE_DIR', str(tmp_path / 'ray_results'))
        self.local_dir = str(tmp_path / 'ray_results')
        yield self.local_dir

    def setUp(self):
        if False:
            while True:
                i = 10
        ray.init()

    def tearDown(self):
        if False:
            return 10
        ray.shutdown()

    def test_tuner_with_xgboost_trainer(self):
        if False:
            i = 10
            return i + 15
        'Test a successful run.'
        trainer = XGBoostTrainer(label_column='target', params={}, datasets={'train': gen_dataset_func_eager()})
        param_space = {'scaling_config': ScalingConfig(num_workers=tune.grid_search([1, 2])), 'datasets': {'train': tune.grid_search([gen_dataset_func(), gen_dataset_func(do_shuffle=True)])}, 'params': {'objective': 'binary:logistic', 'tree_method': 'approx', 'eval_metric': ['logloss', 'error'], 'eta': tune.loguniform(0.0001, 0.1), 'subsample': tune.uniform(0.5, 1.0), 'max_depth': tune.randint(1, 9)}}
        tuner = Tuner(trainable=trainer, run_config=RunConfig(name='test_tuner'), param_space=param_space, tune_config=TuneConfig(mode='min', metric='train-error'), _tuner_kwargs={'max_concurrent_trials': 1})
        results = tuner.fit()
        assert len(results) == 4

    def test_tuner_with_xgboost_trainer_driver_fail_and_resume(self):
        if False:
            print('Hello World!')
        os.environ['TUNE_GLOBAL_CHECKPOINT_S'] = '1'
        trainer = XGBoostTrainer(label_column='target', params={}, datasets={'train': gen_dataset_func_eager()})
        param_space = {'scaling_config': ScalingConfig(num_workers=tune.grid_search([1, 2])), 'datasets': {'train': tune.grid_search([gen_dataset_func(), gen_dataset_func(do_shuffle=True)])}, 'params': {'objective': 'binary:logistic', 'tree_method': 'approx', 'eval_metric': ['logloss', 'error'], 'eta': tune.loguniform(0.0001, 0.1), 'subsample': tune.uniform(0.5, 1.0), 'max_depth': tune.randint(1, 9)}}

        class FailureInjectionCallback(Callback):
            """Inject failure at the configured iteration number."""

            def __init__(self, num_iters=10):
                if False:
                    print('Hello World!')
                self.num_iters = num_iters

            def on_step_end(self, iteration, trials, **kwargs):
                if False:
                    i = 10
                    return i + 15
                if iteration == self.num_iters:
                    print(f'Failing after {self.num_iters} iters.')
                    raise RuntimeError
        tuner = Tuner(trainable=trainer, run_config=RunConfig(name='test_tuner_driver_fail', callbacks=[FailureInjectionCallback()]), param_space=param_space, tune_config=TuneConfig(mode='min', metric='train-error'), _tuner_kwargs={'max_concurrent_trials': 1})
        with self.assertRaises(RuntimeError):
            tuner.fit()
        restore_path = os.path.join(self.local_dir, 'test_tuner_driver_fail')
        tuner = Tuner.restore(restore_path, trainable=trainer, param_space=param_space)
        tuner._local_tuner._run_config.callbacks = None
        results = tuner.fit()
        assert len(results) == 4
        assert not results.errors

    def test_tuner_with_torch_trainer(self):
        if False:
            print('Hello World!')
        'Test a successful run using torch trainer.'
        config = {'lr': 0.01, 'hidden_size': 1, 'batch_size': 4, 'epochs': 10}
        scaling_config = ScalingConfig(num_workers=1, use_gpu=False)
        trainer = TorchTrainer(train_loop_per_worker=linear_train_func, train_loop_config=config, scaling_config=scaling_config)
        param_space = {'scaling_config': ScalingConfig(num_workers=tune.grid_search([1, 2])), 'train_loop_config': {'batch_size': tune.grid_search([4, 8]), 'epochs': tune.grid_search([5, 10])}}
        tuner = Tuner(trainable=trainer, run_config=RunConfig(name='test_tuner'), param_space=param_space, tune_config=TuneConfig(mode='min', metric='loss'))
        results = tuner.fit()
        assert len(results) == 8

    def test_tuner_run_config_override(self):
        if False:
            i = 10
            return i + 15
        trainer = DummyTrainer(run_config=RunConfig(stop={'metric': 4}))
        tuner = Tuner(trainer)
        assert tuner._local_tuner._run_config.stop == {'metric': 4}

@pytest.mark.parametrize('params_expected', [({'run_config': RunConfig(progress_reporter=CLIReporter())}, lambda kw: isinstance(kw['progress_reporter'], CLIReporter)), ({'tune_config': TuneConfig(reuse_actors=True)}, lambda kw: kw['reuse_actors'] is True), ({'run_config': RunConfig(log_to_file='some_file')}, lambda kw: kw['log_to_file'] == 'some_file'), ({'tune_config': TuneConfig(max_concurrent_trials=3)}, lambda kw: kw['max_concurrent_trials'] == 3), ({'tune_config': TuneConfig(time_budget_s=60)}, lambda kw: kw['time_budget_s'] == 60)])
def test_tuner_api_kwargs(shutdown_only, params_expected):
    if False:
        for i in range(10):
            print('nop')
    (tuner_params, assertion) = params_expected
    tuner = Tuner(lambda config: 1, **tuner_params)
    caught_kwargs = {}

    class MockExperimentAnalysis:
        trials = []

    def catch_kwargs(**kwargs):
        if False:
            return 10
        caught_kwargs.update(kwargs)
        return MockExperimentAnalysis()
    with patch('ray.tune.impl.tuner_internal.run', catch_kwargs):
        tuner.fit()
    assert assertion(caught_kwargs)

def test_tuner_fn_trainable_invalid_checkpoint_config(shutdown_only):
    if False:
        return 10
    tuner = Tuner(lambda config: 1, run_config=RunConfig(checkpoint_config=CheckpointConfig(checkpoint_at_end=True)))
    with pytest.raises(ValueError):
        tuner.fit()
    tuner = Tuner(lambda config: 1, run_config=RunConfig(checkpoint_config=CheckpointConfig(checkpoint_frequency=1)))
    with pytest.raises(ValueError):
        tuner.fit()

def test_tuner_trainer_checkpoint_config(shutdown_only):
    if False:
        i = 10
        return i + 15
    custom_training_loop_trainer = DataParallelTrainer(train_loop_per_worker=lambda config: 1)
    tuner = Tuner(custom_training_loop_trainer, run_config=RunConfig(checkpoint_config=CheckpointConfig(checkpoint_at_end=True)))
    with pytest.raises(ValueError):
        tuner.fit()
    tuner = Tuner(custom_training_loop_trainer, run_config=RunConfig(checkpoint_config=CheckpointConfig(checkpoint_frequency=1)))
    with pytest.raises(ValueError):
        tuner.fit()
    handles_checkpoints_trainer = XGBoostTrainer(label_column='target', params={}, datasets={'train': ray.data.from_items(list(range(5)))})
    tuner = Tuner(handles_checkpoints_trainer, run_config=RunConfig(checkpoint_config=CheckpointConfig(checkpoint_at_end=True, checkpoint_frequency=1)))._local_tuner
    tuner._get_tune_run_arguments(tuner.converted_trainable)

def test_tuner_fn_trainable_checkpoint_at_end_false(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    tuner = Tuner(lambda config: 1, run_config=RunConfig(checkpoint_config=CheckpointConfig(checkpoint_at_end=False)))
    tuner.fit()

def test_tuner_fn_trainable_checkpoint_at_end_none(shutdown_only):
    if False:
        while True:
            i = 10
    tuner = Tuner(lambda config: 1, run_config=RunConfig(checkpoint_config=CheckpointConfig(checkpoint_at_end=None)))
    tuner.fit()

def test_nonserializable_trainable():
    if False:
        return 10
    import threading
    lock = threading.Lock()
    with pytest.raises(TypeError, match='.*was found to be non-serializable.*'):
        Tuner(lambda config: print(lock))

def _test_no_chdir(runner_type, runtime_env, use_deprecated_config=False):
    if False:
        return 10
    with open('./read.txt', 'w') as f:
        f.write('data')
    ray.init(num_cpus=4, runtime_env=runtime_env)

    def train_func(config):
        if False:
            while True:
                i = 10
        assert os.path.exists('./read.txt') and open('./read.txt', 'r').read() == 'data'
        trial_dir = Path(train.get_context().get_trial_dir())
        trial_dir.joinpath('write.txt').touch()
    if runner_type == 'trainer':
        trainer = DataParallelTrainer(train_func, scaling_config=train.ScalingConfig(num_workers=2))
        result = trainer.fit()
        results = [result]
    elif runner_type == 'tuner':
        tuner = Tuner(train_func, param_space={'id': tune.grid_search(list(range(4)))}, tune_config=TuneConfig(chdir_to_trial_dir=False) if use_deprecated_config else None)
        results = tuner.fit()
        assert not results.errors
    else:
        raise NotImplementedError(f'Invalid: {runner_type}')
    for result in results:
        assert os.path.exists(os.path.join(result.path, 'write.txt'))

def test_tuner_no_chdir_to_trial_dir_deprecated(shutdown_only, chdir_tmpdir):
    if False:
        return 10
    'Test the deprecated `chdir_to_trial_dir` config.'
    _test_no_chdir('tuner', {}, use_deprecated_config=True)
    os.environ.pop(RAY_CHDIR_TO_TRIAL_DIR, None)

@pytest.mark.parametrize('runtime_env', [{}, {'working_dir': '.'}])
def test_tuner_no_chdir_to_trial_dir(shutdown_only, chdir_tmpdir, monkeypatch, runtime_env):
    if False:
        while True:
            i = 10
    'Tests that disabling the env var to keep the working directory the same\n    works for a Tuner run.'
    from ray.train.constants import RAY_CHDIR_TO_TRIAL_DIR
    monkeypatch.setenv(RAY_CHDIR_TO_TRIAL_DIR, '0')
    _test_no_chdir('tuner', runtime_env)

@pytest.mark.parametrize('runtime_env', [{}, {'working_dir': '.'}])
def test_trainer_no_chdir_to_trial_dir(shutdown_only, chdir_tmpdir, monkeypatch, runtime_env):
    if False:
        print('Hello World!')
    'Tests that disabling the env var to keep the working directory the same\n    works for a Trainer run.'
    from ray.train.constants import RAY_CHDIR_TO_TRIAL_DIR
    monkeypatch.setenv(RAY_CHDIR_TO_TRIAL_DIR, '0')
    _test_no_chdir('trainer', runtime_env)

@pytest.mark.parametrize('runtime_env', [{}, {'working_dir': '.'}])
def test_tuner_relative_pathing_with_env_vars(shutdown_only, chdir_tmpdir, runtime_env):
    if False:
        return 10
    'Tests that `TUNE_ORIG_WORKING_DIR` environment variable can be used to access\n    relative paths to the original working directory.\n    '
    with open('./read.txt', 'w') as f:
        f.write('data')
    ray.init(num_cpus=1, runtime_env=runtime_env)

    def train_func(config):
        if False:
            print('Hello World!')
        orig_working_dir = Path(os.environ['TUNE_ORIG_WORKING_DIR'])
        assert str(orig_working_dir) != os.getcwd(), f'Working directory should have changed from {orig_working_dir}'
        data_path = orig_working_dir / 'read.txt'
        assert os.path.exists(data_path) and open(data_path, 'r').read() == 'data'
        trial_dir = Path(train.get_context().get_trial_dir())
        assert str(trial_dir) == os.getcwd()
        with open(trial_dir / 'write.txt', 'w') as f:
            f.write(f"{config['id']}")
    tuner = Tuner(train_func, param_space={'id': tune.grid_search(list(range(4)))})
    results = tuner.fit()
    assert not results.errors
    for result in results:
        artifact_data = open(os.path.join(result.path, 'write.txt'), 'r').read()
        assert artifact_data == f"{result.config['id']}"

def test_invalid_param_space(shutdown_only):
    if False:
        i = 10
        return i + 15
    'Check that Tune raises an error on invalid param_space types.'

    def trainable(config):
        if False:
            for i in range(10):
                print('nop')
        return {'metric': 1}
    with pytest.raises(ValueError):
        Tuner(trainable, param_space='not allowed')
    from ray.tune.tune import _Config

    class CustomConfig(_Config):

        def to_dict(self) -> dict:
            if False:
                return 10
            return {'hparam': 1}
    with pytest.raises(ValueError):
        Tuner(trainable, param_space='not allowed').fit()
    with pytest.raises(ValueError):
        tune.run(trainable, config='not allowed')
    Tuner(trainable, param_space={}).fit()
    Tuner(trainable, param_space=CustomConfig()).fit()
    tune.run(trainable, config=CustomConfig())

def test_tuner_restore_classmethod():
    if False:
        while True:
            i = 10
    tuner = Tuner('PPO')
    with pytest.raises(AttributeError):
        tuner.restore('/', 'PPO')
    with pytest.raises(FileNotFoundError):
        tuner = Tuner.restore('/invalid', 'PPO')
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__] + sys.argv[1:]))