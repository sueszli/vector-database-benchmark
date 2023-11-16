import json
import logging
import os
import tempfile
from unittest import mock
import pytest
import sys
import time
from functools import partial
import ray
from freezegun import freeze_time
from ray.train import CheckpointConfig
from ray.air.execution import FixedResourceManager, PlacementGroupResourceManager
from ray.air.constants import TRAINING_ITERATION
from ray.train import Checkpoint
from ray.train._internal.session import _TrainingResult
from ray.train._internal.storage import StorageContext
from ray.tune import PlacementGroupFactory
from ray.tune.execution.tune_controller import TuneController
from ray.tune.experiment import Trial
from ray.tune.result import DONE
from ray.tune.schedulers import FIFOScheduler
from ray.tune.search import BasicVariantGenerator
from ray.train.tests.util import mock_storage_context
from ray.tune.tests.tune_test_util import TrialResultObserver
STORAGE = mock_storage_context()

@pytest.fixture(scope='function')
def ray_start_4_cpus_2_gpus_extra():
    if False:
        print('Hello World!')
    address_info = ray.init(num_cpus=4, num_gpus=2, resources={'a': 2})
    yield address_info
    ray.shutdown()

def create_mock_components():
    if False:
        for i in range(10):
            print('nop')

    class _MockScheduler(FIFOScheduler):
        errored_trials = []

        def on_trial_error(self, tune_controller, trial):
            if False:
                print('Hello World!')
            self.errored_trials += [trial]

    class _MockSearchAlg(BasicVariantGenerator):
        errored_trials = []

        def on_trial_complete(self, trial_id, error=False, **kwargs):
            if False:
                print('Hello World!')
            if error:
                self.errored_trials += [trial_id]
    searchalg = _MockSearchAlg()
    scheduler = _MockScheduler()
    return (searchalg, scheduler)

@pytest.mark.parametrize('resource_manager_cls', [FixedResourceManager, PlacementGroupResourceManager])
def test_checkpoint_save_restore(ray_start_4_cpus_2_gpus_extra, resource_manager_cls, tmpdir):
    if False:
        while True:
            i = 10
    'Test that a checkpoint is saved and can be used to restore a trainable.\n\n    The trainable saves a checkpoint and terminates. We then start another trial\n    that should restore from the saved checkpoint and assert that it picks up\n    the state and continues to run to termination.\n\n    Legacy test: test_trial_runner_2.py::TrialRunnerTest::testCheckpointing\n    Legacy test: test_trial_runner_2.py::TrialRunnerTest::testRestoreMetricsAfterCheckpointing  # noqa\n    '
    runner = TuneController(resource_manager_factory=lambda : resource_manager_cls(), storage=STORAGE)
    kwargs = {'stopping_criterion': {'training_iteration': 1}, 'placement_group_factory': PlacementGroupFactory([{'CPU': 1, 'GPU': 1}]), 'checkpoint_config': CheckpointConfig(checkpoint_frequency=1), 'storage': STORAGE}
    runner.add_trial(Trial('__fake', **kwargs))
    trials = runner.get_trials()
    runner.step()
    while trials[0].status != Trial.RUNNING:
        runner.step()
    assert ray.get(trials[0].temporary_state.ray_actor.set_info.remote(1)) == 1
    while trials[0].status != Trial.TERMINATED:
        runner.step()
    assert trials[0].latest_checkpoint_result.metrics[TRAINING_ITERATION] == 1
    assert trials[0].last_result[TRAINING_ITERATION] == 1
    assert trials[0].last_result['iterations_since_restore'] == 1
    kwargs['restore_path'] = trials[0].checkpoint.path
    new_trial = Trial('__fake', **kwargs)
    runner.add_trial(new_trial)
    trials = runner.get_trials()
    assert trials[1].status == Trial.PENDING
    while trials[1].status != Trial.RUNNING:
        runner.step()
    runner.step()
    assert ray.get(trials[1].temporary_state.ray_actor.get_info.remote()) == 1
    while trials[1].status != Trial.TERMINATED:
        runner.step()
    assert trials[0].latest_checkpoint_result.metrics[TRAINING_ITERATION] == 1
    assert trials[1].last_result[TRAINING_ITERATION] == 1
    assert trials[1].last_result['iterations_since_restore'] == 1

@pytest.mark.parametrize('resource_manager_cls', [FixedResourceManager, PlacementGroupResourceManager])
def test_checkpoint_at_end(ray_start_4_cpus_2_gpus_extra, resource_manager_cls, tmpdir):
    if False:
        i = 10
        return i + 15
    'Test that a checkpoint is saved at end for class trainables with that config.\n\n    Legacy test: test_trial_runner_2.py::TrialRunnerTest::testCheckpointingAtEnd\n    Legacy test: test_trial_runner_2.py::TrialRunnerTest::testResultDone\n    '
    runner = TuneController(resource_manager_factory=lambda : resource_manager_cls(), storage=STORAGE)
    kwargs = {'stopping_criterion': {'training_iteration': 2}, 'checkpoint_config': CheckpointConfig(checkpoint_at_end=True), 'placement_group_factory': PlacementGroupFactory([{'CPU': 1, 'GPU': 1}]), 'storage': STORAGE}
    runner.add_trial(Trial('__fake', **kwargs))
    trials = runner.get_trials()
    while not runner.is_finished():
        runner.step()
    assert trials[0].has_checkpoint()
    assert trials[0].last_result[DONE]

@pytest.mark.parametrize('resource_manager_cls', [FixedResourceManager, PlacementGroupResourceManager])
def test_pause_resume_trial(ray_start_4_cpus_2_gpus_extra, resource_manager_cls, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Test that trial that is paused and resumed picks up its last checkpoint.\n\n    Legacy test: test_trial_runner_2.py::TrialRunnerTest::testPauseThenResume\n    '
    runner = TuneController(resource_manager_factory=lambda : resource_manager_cls(), storage=STORAGE)
    kwargs = {'stopping_criterion': {'training_iteration': 2}, 'placement_group_factory': PlacementGroupFactory([{'CPU': 1, 'GPU': 1}]), 'checkpoint_config': CheckpointConfig(checkpoint_frequency=1), 'storage': STORAGE}
    runner.add_trial(Trial('__fake', **kwargs))
    trials = runner.get_trials()
    while trials[0].status != Trial.RUNNING:
        runner.step()
    assert ray.get(trials[0].temporary_state.ray_actor.get_info.remote()) is None
    assert ray.get(trials[0].temporary_state.ray_actor.set_info.remote(1)) == 1
    runner._schedule_trial_pause(trials[0], should_checkpoint=True)
    while trials[0].status != Trial.PAUSED:
        runner.step()
    assert trials[0].has_checkpoint()
    assert not trials[0].last_result.get(DONE), trials[0].last_result
    runner._set_trial_status(trials[0], Trial.PENDING)
    while trials[0].status != Trial.RUNNING:
        runner.step()
    assert ray.get(trials[0].temporary_state.ray_actor.get_info.remote()) == 1
    while trials[0].status != Trial.TERMINATED:
        runner.step()
    assert trials[0].checkpoint
    assert trials[0].last_result[TRAINING_ITERATION] == 2
    assert trials[0].last_result['iterations_since_restore'] == 1
    assert trials[0].last_result['time_since_restore'] > 0

@pytest.mark.parametrize('resource_manager_cls', [FixedResourceManager, PlacementGroupResourceManager])
def test_checkpoint_num_to_keep(ray_start_4_cpus_2_gpus_extra, resource_manager_cls, tmp_path):
    if False:
        return 10
    'Test that only num_to_keep checkpoints are kept.\n\n    This should also hold true when the experiment is resumed.\n\n    Legacy test: test_trial_runner_2.py::TrialRunnerTest::testPauseResumeCheckpointCount\n    '
    trial = Trial('__fake', checkpoint_config=CheckpointConfig(num_to_keep=2), storage=STORAGE)
    trial.init_local_path()

    def write_checkpoint(trial: Trial, index: int):
        if False:
            i = 10
            return i + 15
        checkpoint_dir = tmp_path / StorageContext._make_checkpoint_dir_name(index)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        result = {'training_iteration': index}
        with open(os.path.join(checkpoint_dir, 'cp.json'), 'w') as f:
            json.dump(result, f)
        checkpoint = Checkpoint.from_directory(checkpoint_dir)
        return _TrainingResult(checkpoint=checkpoint, metrics=result)

    def get_checkpoint_dirs(trial: Trial):
        if False:
            return 10
        return [d for d in os.listdir(tmp_path) if d.startswith('checkpoint_')]
    runner = TuneController(resource_manager_factory=lambda : resource_manager_cls(), storage=STORAGE)
    runner.add_trial(trial)
    result = write_checkpoint(trial, 1)
    runner._on_saving_result(trial, result)
    cp_dirs = get_checkpoint_dirs(trial)
    assert len(cp_dirs) == 1, f'Checkpoint dirs: {cp_dirs}'
    result = write_checkpoint(trial, 2)
    runner._on_saving_result(trial, result)
    cp_dirs = get_checkpoint_dirs(trial)
    assert len(cp_dirs) == 2, f'Checkpoint dirs: {cp_dirs}'
    result = write_checkpoint(trial, 3)
    runner._on_saving_result(trial, result)
    cp_dirs = get_checkpoint_dirs(trial)
    assert len(cp_dirs) == 2, f'Checkpoint dirs: {cp_dirs}'
    runner.checkpoint(force=True)
    runner = TuneController(resource_manager_factory=lambda : resource_manager_cls(), storage=STORAGE)
    runner.resume()
    trial = runner.get_trials()[0]
    result = write_checkpoint(trial, 4)
    runner._on_saving_result(trial, result)
    cp_dirs = get_checkpoint_dirs(trial)
    assert len(cp_dirs) == 2, f'Checkpoint dirs: {cp_dirs}'
    result = write_checkpoint(trial, 5)
    runner._on_saving_result(trial, result)
    cp_dirs = get_checkpoint_dirs(trial)
    assert len(cp_dirs) == 2, f'Checkpoint dirs: {cp_dirs}'
    assert 'checkpoint_000004' in cp_dirs
    assert 'checkpoint_000005' in cp_dirs
    assert 'checkpoint_000002' not in cp_dirs
    assert 'checkpoint_000003' not in cp_dirs

@pytest.mark.parametrize('resource_manager_cls', [FixedResourceManager, PlacementGroupResourceManager])
def test_checkpoint_freq_buffered(ray_start_4_cpus_2_gpus_extra, resource_manager_cls, tmp_path):
    if False:
        while True:
            i = 10
    'Test that trial checkpoints are a lower bound for buffered training iterations.\n\n    Legacy test: test_trial_runner_3.py::TrialRunnerTest::testCheckpointFreqBuffered\n    '
    with mock.patch.dict(os.environ, {'TUNE_RESULT_BUFFER_LENGTH': '7', 'TUNE_RESULT_BUFFER_MIN_TIME_S': '1'}):

        def num_checkpoints(trial):
            if False:
                i = 10
                return i + 15
            return sum((item.startswith('checkpoint_') for item in os.listdir(trial.local_path)))
        trial = Trial('__fake', checkpoint_config=CheckpointConfig(checkpoint_frequency=3), storage=STORAGE)
        runner = TuneController(resource_manager_factory=lambda : resource_manager_cls(), storage=STORAGE, checkpoint_period=0)
        runner.add_trial(trial)
        while not trial.is_saving:
            runner.step()
        runner.step()
        assert trial.last_result[TRAINING_ITERATION] == 3
        assert num_checkpoints(trial) == 1
        while not trial.is_saving:
            runner.step()
        runner.step()
        assert trial.last_result[TRAINING_ITERATION] == 6
        assert num_checkpoints(trial) == 2
        while not trial.is_saving:
            runner.step()
        runner.step()
        assert trial.last_result[TRAINING_ITERATION] == 9
        assert num_checkpoints(trial) == 3

@pytest.mark.parametrize('resource_manager_cls', [FixedResourceManager, PlacementGroupResourceManager])
def test_checkpoint_at_end_not_buffered(ray_start_4_cpus_2_gpus_extra, resource_manager_cls, tmp_path):
    if False:
        i = 10
        return i + 15
    'Test that trials with `checkpoint_at_end=True` are never buffered.\n\n    Legacy test: test_trial_runner_3.py::TrialRunnerTest::testCheckpointAtEndNotBuffered\n    '
    with mock.patch.dict(os.environ, {'TUNE_RESULT_BUFFER_LENGTH': '7', 'TUNE_RESULT_BUFFER_MIN_TIME_S': '0.5'}):

        def num_checkpoints(trial):
            if False:
                print('Hello World!')
            return sum((item.startswith('checkpoint_') for item in os.listdir(trial.local_path)))
        trial = Trial('__fake', checkpoint_config=CheckpointConfig(checkpoint_at_end=True), stopping_criterion={'training_iteration': 4}, storage=STORAGE)
        observer = TrialResultObserver()
        runner = TuneController(resource_manager_factory=lambda : resource_manager_cls(), storage=STORAGE, callbacks=[observer])
        runner.add_trial(trial)
        while not observer.just_received_a_result():
            runner.step()
        assert trial.last_result[TRAINING_ITERATION] == 1
        assert num_checkpoints(trial) == 0
        while True:
            runner.step()
            if observer.just_received_a_result():
                break
        assert trial.last_result[TRAINING_ITERATION] == 2
        assert num_checkpoints(trial) == 0
        while True:
            runner.step()
            if observer.just_received_a_result():
                break
        assert trial.last_result[TRAINING_ITERATION] == 3
        assert num_checkpoints(trial) == 0
        while True:
            runner.step()
            if observer.just_received_a_result():
                break
        assert trial.last_result[TRAINING_ITERATION] == 4
        while not runner.is_finished():
            runner.step()
        assert num_checkpoints(trial) == 1

@pytest.mark.parametrize('resource_manager_cls', [FixedResourceManager, PlacementGroupResourceManager])
def test_checkpoint_user_checkpoint(ray_start_4_cpus_2_gpus_extra, resource_manager_cls, tmp_path):
    if False:
        i = 10
        return i + 15
    'Test that user checkpoint freq is respected.\n\n    Legacy test: test_trial_runner_3.py::TrialRunnerTest::testUserCheckpoint\n    '
    with mock.patch.dict(os.environ, {'TUNE_RESULT_BUFFER_LENGTH': '1', 'TUNE_MAX_PENDING_TRIALS_PG': '1'}):
        runner = TuneController(resource_manager_factory=lambda : resource_manager_cls(), storage=STORAGE, checkpoint_period=0)
        runner.add_trial(Trial('__fake', config={'user_checkpoint_freq': 2}, storage=STORAGE))
        trials = runner.get_trials()
        while not trials[0].status == Trial.RUNNING:
            runner.step()
        assert ray.get(trials[0].temporary_state.ray_actor.set_info.remote(1)) == 1
        while trials[0].last_result.get(TRAINING_ITERATION, 0) < 1:
            runner.step()
        assert not trials[0].has_checkpoint()
        while trials[0].last_result.get(TRAINING_ITERATION, 99) < 2:
            runner.step()
        assert not trials[0].has_checkpoint()
        while trials[0].last_result.get(TRAINING_ITERATION, 99) < 3:
            runner.step()
        runner.step()
        assert trials[0].has_checkpoint()
        runner2 = TuneController(resource_manager_factory=lambda : resource_manager_cls(), storage=STORAGE, resume='LOCAL')
        trials2 = runner2.get_trials()
        while not trials2[0].status == Trial.RUNNING:
            runner2.step()
        assert ray.get(trials2[0].temporary_state.ray_actor.get_info.remote()) == 1

@pytest.mark.parametrize('resource_manager_cls', [FixedResourceManager, PlacementGroupResourceManager])
def test_checkpoint_user_checkpoint_buffered(ray_start_4_cpus_2_gpus_extra, resource_manager_cls, tmp_path):
    if False:
        print('Hello World!')
    'Test that user checkpoint freq is respected with buffered training.\n\n    Legacy test: test_trial_runner_3.py::TrialRunnerTest::testUserCheckpointBuffered\n    '

    def num_checkpoints(trial):
        if False:
            print('Hello World!')
        return sum((item.startswith('checkpoint_') for item in os.listdir(trial.local_path)))
    with mock.patch.dict(os.environ, {'TUNE_RESULT_BUFFER_LENGTH': '8', 'TUNE_RESULT_BUFFER_MIN_TIME_S': '1'}):
        runner = TuneController(resource_manager_factory=lambda : resource_manager_cls(), storage=STORAGE, checkpoint_period=0)
        runner.add_trial(Trial('__fake', config={'user_checkpoint_freq': 10}, storage=STORAGE))
        trials = runner.get_trials()
        while trials[0].status != Trial.RUNNING:
            runner.step()
        assert ray.get(trials[0].temporary_state.ray_actor.set_info.remote(1)) == 1
        assert num_checkpoints(trials[0]) == 0
        while trials[0].last_result.get(TRAINING_ITERATION, 0) < 8:
            runner.step()
        assert not trials[0].has_checkpoint()
        assert num_checkpoints(trials[0]) == 0
        while trials[0].last_result.get(TRAINING_ITERATION) < 11:
            runner.step()
        runner.step()
        assert trials[0].has_checkpoint()
        assert num_checkpoints(trials[0]) == 1
        while trials[0].last_result.get(TRAINING_ITERATION) < 19:
            runner.step()
        runner.step()
        assert trials[0].has_checkpoint()
        assert num_checkpoints(trials[0]) == 1
        while trials[0].last_result.get(TRAINING_ITERATION) < 21:
            runner.step()
        runner.step()
        assert trials[0].has_checkpoint()
        assert num_checkpoints(trials[0]) == 2
        while trials[0].last_result.get(TRAINING_ITERATION) < 29:
            runner.step()
        runner.step()
        assert trials[0].has_checkpoint()
        assert num_checkpoints(trials[0]) == 2

@pytest.mark.parametrize('resource_manager_cls', [FixedResourceManager, PlacementGroupResourceManager])
def test_checkpoint_auto_period(ray_start_4_cpus_2_gpus_extra, resource_manager_cls, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    'Test that the checkpoint auto period is adjusted when syncing takes a long time.\n\n    Legacy test: test_trial_runner_3.py::TrialRunnerTest::testCheckpointAutoPeriod\n    '
    storage = mock_storage_context(delete_syncer=False)
    with mock.patch.object(storage.syncer, 'sync_up') as sync_up, tempfile.TemporaryDirectory() as local_dir:
        storage.storage_local_path = local_dir
        sync_up.side_effect = lambda *a, **kw: time.sleep(2)
        runner = TuneController(resource_manager_factory=lambda : resource_manager_cls(), storage=storage, checkpoint_period='auto')
        runner.add_trial(Trial('__fake', config={'user_checkpoint_freq': 1}, storage=storage))
        runner.step()
        assert runner._checkpoint_manager._checkpoint_period > 38.0

@pytest.mark.parametrize('resource_manager_cls', [FixedResourceManager, PlacementGroupResourceManager])
def test_checkpoint_force_with_num_to_keep(ray_start_4_cpus_2_gpus_extra, resource_manager_cls, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    'Test that cloud syncing is forced if one of the trials has made more\n    than num_to_keep checkpoints since last sync.\n\n    Legacy test: test_trial_runner_3.py::TrialRunnerTest::\n        testCloudCheckpointForceWithNumToKeep\n    '
    storage = mock_storage_context(delete_syncer=False)
    storage.syncer.__getstate__ = lambda *a, **kw: {}
    with mock.patch.dict(os.environ, {'TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S': '2'}), mock.patch.object(storage.syncer, 'sync_up') as sync_up:
        num_to_keep = 2
        checkpoint_config = CheckpointConfig(num_to_keep=num_to_keep, checkpoint_frequency=1)
        runner = TuneController(resource_manager_factory=lambda : resource_manager_cls(), storage=storage, checkpoint_period=100, trial_checkpoint_config=checkpoint_config)

        class CheckpointingTrial(Trial):

            def should_checkpoint(self):
                if False:
                    while True:
                        i = 10
                return True

            def get_json_state(self):
                if False:
                    i = 10
                    return i + 15
                return ('', '')
        trial = CheckpointingTrial('__fake', checkpoint_config=checkpoint_config, stopping_criterion={'training_iteration': 10}, storage=storage)
        runner.add_trial(trial)
        buffer = []
        from ray.tune.execution.experiment_state import logger
        with mock.patch.object(logger, 'warning', lambda x: buffer.append(x)):
            while not runner.is_finished():
                runner.step()
        assert any(('syncing has been triggered multiple' in x for x in buffer))
        assert sync_up.call_count == 6

@pytest.mark.parametrize('resource_manager_cls', [FixedResourceManager, PlacementGroupResourceManager])
def test_checkpoint_forced_cloud_sync_timeout(ray_start_4_cpus_2_gpus_extra, resource_manager_cls, tmp_path):
    if False:
        while True:
            i = 10
    'Test that trial runner experiment checkpointing with forced cloud syncing\n    times out correctly when the sync process hangs.\n\n    Legacy test: test_trial_runner_3.py::TrialRunnerTest::\n        testForcedCloudCheckpointSyncTimeout\n    '
    storage = mock_storage_context(delete_syncer=False)
    storage.syncer.sync_period = 60
    storage.syncer.sync_timeout = 0.001

    def _hanging_sync_up_command(*args, **kwargs):
        if False:
            print('Hello World!')
        time.sleep(200)

    def _sync_up_command(self, local_path: str, uri: str, exclude=None):
        if False:
            while True:
                i = 10
        return (_hanging_sync_up_command, {})
    with mock.patch.object(storage.syncer, '_sync_up_command') as sync_up_cmd:
        sync_up_cmd.side_effect = partial(_sync_up_command, storage.syncer)
        runner = TuneController(resource_manager_factory=lambda : resource_manager_cls(), storage=storage)
        runner.checkpoint(force=True)
        assert sync_up_cmd.call_count == 1
        buffer = []
        logger = logging.getLogger('ray.tune.execution.experiment_state')
        with mock.patch.object(logger, 'warning', lambda x: buffer.append(x)):
            runner.checkpoint(force=True)
        assert any(('timed out' in x for x in buffer))
        assert sync_up_cmd.call_count == 2

@pytest.mark.parametrize('resource_manager_cls', [FixedResourceManager, PlacementGroupResourceManager])
def test_checkpoint_periodic_cloud_sync_timeout(ray_start_4_cpus_2_gpus_extra, resource_manager_cls, tmp_path):
    if False:
        while True:
            i = 10
    'Test that trial runner experiment checkpointing with the default periodic\n    cloud syncing times out and retries correctly when the sync process hangs.\n\n    Legacy test: test_trial_runner_3.py::TrialRunnerTest::\n        testPeriodicCloudCheckpointSyncTimeout\n    '
    storage = mock_storage_context(delete_syncer=False)
    storage.syncer.sync_period = 60
    storage.syncer.sync_timeout = 0.5

    def _hanging_sync_up_command(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        time.sleep(200)

    def _sync_up_command(self, local_path: str, uri: str, exclude=None):
        if False:
            i = 10
            return i + 15
        return (_hanging_sync_up_command, {})
    with mock.patch.object(storage.syncer, '_sync_up_command') as sync_up_cmd, freeze_time() as frozen:
        sync_up_cmd.side_effect = partial(_sync_up_command, storage.syncer)
        runner = TuneController(resource_manager_factory=lambda : resource_manager_cls(), storage=storage)
        runner.checkpoint()
        assert sync_up_cmd.call_count == 1
        frozen.tick(storage.syncer.sync_period / 2)
        runner.checkpoint()
        assert sync_up_cmd.call_count == 1
        frozen.tick(storage.syncer.sync_period / 2)
        buffer = []
        logger = logging.getLogger('ray.train._internal.syncer')
        with mock.patch.object(logger, 'warning', lambda x: buffer.append(x)):
            runner.checkpoint()
        assert any(('did not finish running within the timeout' in x for x in buffer)), buffer
        assert sync_up_cmd.call_count == 2
if __name__ == '__main__':
    sys.exit(pytest.main(['-v', __file__]))