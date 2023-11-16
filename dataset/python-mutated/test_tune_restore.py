import signal
import subprocess
from collections import Counter
import multiprocessing
import os
from pathlib import Path
import pytest
import shutil
import tempfile
import threading
import time
from typing import List
import unittest
from unittest import mock
import ray
import ray.train
from ray import tune
from ray._private.test_utils import recursive_fnmatch, run_string_as_driver
from ray.train import CheckpointConfig, Checkpoint
from ray.exceptions import RayTaskError
from ray.rllib import _register_all
from ray.train._internal.session import _TrainingResult
from ray.tune import TuneError
from ray.tune.callback import Callback
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search import Searcher
from ray.tune.experiment import Trial
from ray.tune.execution.tune_controller import TuneController
from ray.tune.utils import validate_save_restore
from ray.tune.utils.mock_trainable import MyTrainableClass

class TuneRestoreTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        ray.init(num_cpus=1, num_gpus=0, local_mode=True)
        tmpdir = tempfile.mkdtemp()
        test_name = 'TuneRestoreTest'
        tune.run('PPO', name=test_name, stop={'training_iteration': 1}, checkpoint_config=CheckpointConfig(checkpoint_frequency=1), storage_path=tmpdir, config={'env': 'CartPole-v0', 'framework': 'tf'})
        logdir = os.path.expanduser(os.path.join(tmpdir, test_name))
        self.logdir = logdir
        self.checkpoint_path = recursive_fnmatch(logdir, 'algorithm_state.pkl')[0]
        self.checkpoint_parent = Path(self.checkpoint_path).parent

    def tearDown(self):
        if False:
            while True:
                i = 10
        shutil.rmtree(self.logdir)
        ray.shutdown()
        _register_all()

    def testTuneRestore(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(os.path.isfile(self.checkpoint_path))
        tune.run('PPO', name='TuneRestoreTest', stop={'training_iteration': 2}, checkpoint_config=CheckpointConfig(checkpoint_frequency=1), restore=self.checkpoint_parent, config={'env': 'CartPole-v0', 'framework': 'tf'})

    def testPostRestoreCheckpointExistence(self):
        if False:
            print('Hello World!')
        'Tests that checkpoint restored from is not deleted post-restore.'
        self.assertTrue(os.path.isfile(self.checkpoint_path))
        tune.run('PPO', name='TuneRestoreTest', stop={'training_iteration': 2}, checkpoint_config=CheckpointConfig(num_to_keep=1, checkpoint_frequency=1), restore=self.checkpoint_parent, config={'env': 'CartPole-v0', 'framework': 'tf'})
        self.assertTrue(os.path.isfile(self.checkpoint_path))

class SteppingCallback(Callback):

    def __init__(self, driver_semaphore, trainer_semaphore):
        if False:
            for i in range(10):
                print('nop')
        self.driver_semaphore = driver_semaphore
        self.trainer_semaphore = trainer_semaphore

    def on_step_end(self, iteration, trials, **info):
        if False:
            return 10
        self.driver_semaphore.release()
        self.trainer_semaphore.acquire()

def _run(local_dir, driver_semaphore, trainer_semaphore):
    if False:
        i = 10
        return i + 15

    def _train(config):
        if False:
            i = 10
            return i + 15
        for i in range(7):
            ray.train.report(dict(val=i))
    tune.run(_train, storage_path=local_dir, name='interrupt', callbacks=[SteppingCallback(driver_semaphore, trainer_semaphore)])

class TuneInterruptionTest(unittest.TestCase):

    @unittest.skip('Spawn seems to have a malfunction on Python 3.8 CI')
    def testExperimentInterrupted(self):
        if False:
            for i in range(10):
                print('nop')
        local_dir = tempfile.mkdtemp()
        mp_ctx = multiprocessing.get_context('spawn')
        driver_semaphore = mp_ctx.Semaphore()
        trainer_semaphore = mp_ctx.Semaphore()
        process = mp_ctx.Process(target=_run, args=(local_dir, driver_semaphore, trainer_semaphore), name='tune_interrupt')
        process.daemon = False
        process.start()
        exp_dir = os.path.join(local_dir, 'interrupt')
        for i in range(5):
            driver_semaphore.acquire()
            trainer_semaphore.release()
        driver_semaphore.acquire()
        experiment_state_file = None
        for file in os.listdir(exp_dir):
            if file.startswith('experiment_state'):
                experiment_state_file = os.path.join(exp_dir, file)
                break
        self.assertTrue(experiment_state_file)
        last_mtime = os.path.getmtime(experiment_state_file)
        os.kill(process.pid, signal.SIGINT)
        trainer_semaphore.release()
        time.sleep(2)
        new_mtime = os.path.getmtime(experiment_state_file)
        self.assertNotEqual(last_mtime, new_mtime)
        shutil.rmtree(local_dir)

    def testInterruptDisabledInWorkerThread(self):
        if False:
            i = 10
            return i + 15
        event = threading.Event()

        def run_in_thread():
            if False:
                print('Hello World!')

            def _train(config):
                if False:
                    return 10
                for i in range(7):
                    ray.train.report(dict(val=i))
            tune.run(_train)
            event.set()
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        event.wait()
        thread.join()
        ray.shutdown()
        os.environ.pop('TUNE_DISABLE_SIGINT_HANDLER', None)

class TuneFailResumeGridTest(unittest.TestCase):

    class FailureInjectorCallback(Callback):
        """Adds random failure injection to the TrialExecutor."""

        def __init__(self, num_trials=20):
            if False:
                while True:
                    i = 10
            self.num_trials = num_trials

        def on_step_end(self, trials, **kwargs):
            if False:
                i = 10
                return i + 15
            if len(trials) == self.num_trials:
                print(f'Failing after {self.num_trials} trials.')
                raise RuntimeError

    class CheckStateCallback(Callback):
        """Checks state for the experiment initialization."""

        def __init__(self, expected_trials=20):
            if False:
                while True:
                    i = 10
            self.expected_trials = expected_trials
            self._checked = False

        def on_step_begin(self, iteration, trials, **kwargs):
            if False:
                i = 10
                return i + 15
            if not self._checked:
                assert len(trials) == self.expected_trials
                self._checked = True

    class CheckTrialResourcesCallback(Callback):
        """Checks if pending trials are requesting the right amount of
        resources.

        The check happens exactly once after `check_after` number of calls
        to on_step_begin(). Note, we deliberately delay the check to after
        `check_after` number of steps. This is because when we start a
        tuning job from fresh (rather than restored), trial list is still
        empty - any check now would be trivial and thus wasted.
        """

        def __init__(self, expected_cpu: int, check_after: int=1):
            if False:
                return 10
            self._expected_cpu = expected_cpu
            self._checked = False
            self._check_after = check_after

        def on_step_begin(self, iteration: int, trials: List['Trial'], **info):
            if False:
                while True:
                    i = 10
            if not self._checked and iteration >= self._check_after:
                for trial in trials:
                    if trial.status == Trial.PENDING:
                        assert trial.placement_group_factory.required_resources.get('CPU', 0) == self._expected_cpu
                self._checked = True

    def setUp(self):
        if False:
            return 10
        self.logdir = tempfile.mkdtemp()
        os.environ['TUNE_GLOBAL_CHECKPOINT_S'] = '0'
        ray.init(local_mode=False, num_cpus=2)
        from ray.tune import register_trainable
        register_trainable('trainable', MyTrainableClass)

    def tearDown(self):
        if False:
            return 10
        os.environ.pop('TUNE_GLOBAL_CHECKPOINT_S')
        shutil.rmtree(self.logdir)
        ray.shutdown()

    def testFailResumeGridSearch(self):
        if False:
            i = 10
            return i + 15
        os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = '1'
        config = dict(num_samples=3, fail_fast=True, config={'test': tune.grid_search([1, 2, 3]), 'test2': tune.grid_search([1, 2, 3])}, stop={'training_iteration': 2}, storage_path=self.logdir, name='testFailResumeGridSearch', verbose=1)
        with self.assertRaises(RuntimeError):
            tune.run('trainable', callbacks=[self.FailureInjectorCallback()], **config)
        analysis = tune.run('trainable', resume=True, callbacks=[self.CheckStateCallback()], **config)
        assert len(analysis.trials) == 27
        test_counter = Counter([t.config['test'] for t in analysis.trials])
        assert all((v == 9 for v in test_counter.values()))
        test2_counter = Counter([t.config['test2'] for t in analysis.trials])
        assert all((v == 9 for v in test2_counter.values()))

    def testResourceUpdateInResume(self):
        if False:
            while True:
                i = 10
        os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = '1'
        config = dict(num_samples=3, fail_fast=True, config={'test': tune.grid_search([1, 2, 3]), 'test2': tune.grid_search([1, 2, 3])}, stop={'training_iteration': 2}, storage_path=self.logdir, name='testResourceUpdateInResume', verbose=1)
        with self.assertRaises(RuntimeError):
            tune.run('trainable', callbacks=[self.FailureInjectorCallback(), self.CheckTrialResourcesCallback(1)], **config)
        analysis = tune.run('trainable', resume=True, resources_per_trial={'cpu': 2}, callbacks=[self.CheckTrialResourcesCallback(2)], **config)
        assert len(analysis.trials) == 27

    @mock.patch.dict(os.environ, {'TUNE_MAX_PENDING_TRIALS_PG': '1'})
    def testConfigUpdateInResume(self):
        if False:
            for i in range(10):
                print('nop')

        class FakeDataset:

            def __init__(self, name):
                if False:
                    i = 10
                    return i + 15
                self.name = name
        config = dict(num_samples=1, fail_fast=True, config={'test': tune.grid_search([FakeDataset('1'), FakeDataset('2'), FakeDataset('3')]), 'test2': tune.grid_search([FakeDataset('4'), FakeDataset('5'), FakeDataset('6'), FakeDataset('7')])}, stop={'training_iteration': 2}, storage_path=self.logdir, name='testConfigUpdateInResume', verbose=1)
        with self.assertRaises(RuntimeError):
            tune.run('trainable', callbacks=[self.FailureInjectorCallback(num_trials=1), self.CheckTrialResourcesCallback(1)], **config)
        config['config'] = {'test': tune.grid_search([FakeDataset('8'), FakeDataset('9'), FakeDataset('10')]), 'test2': tune.grid_search([FakeDataset('11'), FakeDataset('12'), FakeDataset('13'), FakeDataset('14')])}
        analysis = tune.run('trainable', resume=True, **config)
        assert len(analysis.trials) == 12
        for t in analysis.trials:
            assert t.config['test'].name in ['8', '9', '10']
            assert t.config['test2'].name in ['11', '12', '13', '14']

    def testFailResumeWithPreset(self):
        if False:
            i = 10
            return i + 15
        os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = '1'
        search_alg = BasicVariantGenerator(points_to_evaluate=[{'test': -1, 'test2': -1}, {'test': -1}, {'test2': -1}])
        config = dict(num_samples=3 + 3, fail_fast=True, config={'test': tune.grid_search([1, 2, 3]), 'test2': tune.grid_search([1, 2, 3])}, stop={'training_iteration': 2}, storage_path=self.logdir, name='testFailResumeWithPreset', verbose=1)
        with self.assertRaises(RuntimeError):
            tune.run('trainable', callbacks=[self.FailureInjectorCallback(5)], search_alg=search_alg, **config)
        print('---- RESTARTING RUN ----')
        analysis = tune.run('trainable', resume=True, callbacks=[self.CheckStateCallback(expected_trials=5)], search_alg=search_alg, **config)
        assert len(analysis.trials) == 34
        test_counter = Counter([t.config['test'] for t in analysis.trials])
        assert test_counter.pop(-1) == 4
        assert all((v == 10 for v in test_counter.values()))
        test2_counter = Counter([t.config['test2'] for t in analysis.trials])
        assert test2_counter.pop(-1) == 4
        assert all((v == 10 for v in test2_counter.values()))

    def testFailResumeAfterPreset(self):
        if False:
            return 10
        os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = '1'
        search_alg = BasicVariantGenerator(points_to_evaluate=[{'test': -1, 'test2': -1}, {'test': -1}, {'test2': -1}])
        config = dict(num_samples=3 + 3, fail_fast=True, config={'test': tune.grid_search([1, 2, 3]), 'test2': tune.grid_search([1, 2, 3])}, stop={'training_iteration': 2}, storage_path=self.logdir, name='testFailResumeAfterPreset', verbose=1)
        with self.assertRaises(RuntimeError):
            tune.run('trainable', callbacks=[self.FailureInjectorCallback(15)], search_alg=search_alg, **config)
        print('---- RESTARTING RUN ----')
        analysis = tune.run('trainable', resume=True, callbacks=[self.CheckStateCallback(expected_trials=15)], search_alg=search_alg, **config)
        assert len(analysis.trials) == 34
        test_counter = Counter([t.config['test'] for t in analysis.trials])
        assert test_counter.pop(-1) == 4
        assert all((v == 10 for v in test_counter.values()))
        test2_counter = Counter([t.config['test2'] for t in analysis.trials])
        assert test2_counter.pop(-1) == 4
        assert all((v == 10 for v in test2_counter.values()))

    def testMultiExperimentFail(self):
        if False:
            print('Hello World!')
        os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = '1'
        experiments = []
        for i in range(3):
            experiments.append(tune.Experiment(run=MyTrainableClass, name='testMultiExperimentFail', num_samples=2, config={'test': tune.grid_search([1, 2, 3])}, stop={'training_iteration': 1}, storage_path=self.logdir))
        with self.assertRaises(RuntimeError):
            tune.run(experiments, callbacks=[self.FailureInjectorCallback(10)], fail_fast=True)
        analysis = tune.run(experiments, resume=True, callbacks=[self.CheckStateCallback(expected_trials=10)], fail_fast=True)
        assert len(analysis.trials) == 18

    def testWarningLargeGrid(self):
        if False:
            while True:
                i = 10
        config = dict(num_samples=3, fail_fast=True, config={'test': tune.grid_search(list(range(20))), 'test2': tune.grid_search(list(range(20))), 'test3': tune.grid_search(list(range(20))), 'test4': tune.grid_search(list(range(20))), 'test5': tune.grid_search(list(range(20)))}, stop={'training_iteration': 2}, storage_path=self.logdir, name='testWarningLargeGrid', verbose=1)
        with self.assertWarnsRegex(UserWarning, 'exceeds the serialization threshold'):
            with self.assertRaises(RuntimeError):
                tune.run('trainable', callbacks=[self.FailureInjectorCallback(10)], **config)

class TuneExampleTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        ray.init(num_cpus=2)

    def tearDown(self):
        if False:
            return 10
        ray.shutdown()
        _register_all()

    def testPBTKeras(self):
        if False:
            return 10
        from ray.tune.examples.pbt_tune_cifar10_with_keras import Cifar10Model
        from tensorflow.keras.datasets import cifar10
        cifar10.load_data()
        validate_save_restore(Cifar10Model)

    def testPyTorchMNIST(self):
        if False:
            return 10
        from ray.tune.examples.mnist_pytorch_trainable import TrainMNIST
        from torchvision import datasets
        datasets.MNIST('~/data', train=True, download=True)
        validate_save_restore(TrainMNIST)

    def testHyperbandExample(self):
        if False:
            for i in range(10):
                print('nop')
        validate_save_restore(MyTrainableClass)

    def testAsyncHyperbandExample(self):
        if False:
            for i in range(10):
                print('nop')
        validate_save_restore(MyTrainableClass)

class AutoInitTest(unittest.TestCase):

    def testTuneRestore(self):
        if False:
            while True:
                i = 10
        self.assertFalse(ray.is_initialized())
        tune.run('__fake', name='TestAutoInit', stop={'training_iteration': 1})
        self.assertTrue(ray.is_initialized())

    def tearDown(self):
        if False:
            print('Hello World!')
        ray.shutdown()
        _register_all()

class SearcherTest(unittest.TestCase):

    class MockSearcher(Searcher):

        def __init__(self, data):
            if False:
                return 10
            self.data = data

        def save(self, path):
            if False:
                while True:
                    i = 10
            with open(path, 'w') as f:
                f.write(self.data)

        def restore(self, path):
            if False:
                print('Hello World!')
            with open(path, 'r') as f:
                self.data = f.read()

    def testSaveRestoreDir(self):
        if False:
            return 10
        tmpdir = tempfile.mkdtemp()
        original_data = 'hello-its-me'
        searcher = self.MockSearcher(original_data)
        searcher.save_to_dir(tmpdir)
        searcher_2 = self.MockSearcher('no-its-not-me')
        searcher_2.restore_from_dir(tmpdir)
        assert searcher_2.data == original_data

class WorkingDirectoryTest(unittest.TestCase):

    def testWorkingDir(self):
        if False:
            while True:
                i = 10
        'Trainables should know the original working dir through env variable.'
        os.environ.pop('TUNE_ORIG_WORKING_DIR', None)
        working_dir = os.getcwd()

        def f(config):
            if False:
                for i in range(10):
                    print('nop')
            assert os.environ.get('TUNE_ORIG_WORKING_DIR') == working_dir
        ray.init(num_cpus=1)
        tune.run(f)
        ray.shutdown()

class TrainableCrashWithFailFast(unittest.TestCase):

    def test(self):
        if False:
            i = 10
            return i + 15
        'Trainable crashes with fail_fast flag and the original crash message\n        should bubble up.'

        def f(config):
            if False:
                print('Hello World!')
            ray.train.report({'a': 1})
            time.sleep(0.1)
            raise RuntimeError('Error happens in trainable!!')
        with self.assertRaisesRegex(RayTaskError, 'Error happens in trainable!!'):
            tune.run(f, fail_fast=TuneController.RAISE)

class ResourceExhaustedTest(unittest.TestCase):

    def test_resource_exhausted_info(self):
        if False:
            i = 10
            return i + 15
        'This is to test if helpful information is displayed when\n        the objects captured in trainable/training function are too\n        large and RESOURCES_EXHAUSTED error of gRPC is triggered.'
        from sklearn.datasets import fetch_olivetti_faces
        a_large_array = []
        for i in range(50):
            a_large_array.append(fetch_olivetti_faces())

        def training_func(config):
            if False:
                print('Hello World!')
            for item in a_large_array:
                assert item
        with self.assertRaisesRegex(TuneError, 'The Trainable/training function is too large for grpc resource limit.'):
            tune.run(training_func)

@pytest.mark.parametrize('trial_config', [{}, {'attr': 4}, {'nested': {'key': 'value'}}])
def test_trial_last_result_restore(trial_config):
    if False:
        while True:
            i = 10
    metrics = {'metric1': 4, 'nested2': {'metric3': 6}}
    metrics['config'] = trial_config
    trial = Trial(trainable_name='stub', config=trial_config, stub=True)
    trial.update_last_result(metrics)
    result = _TrainingResult(checkpoint=Checkpoint(path='file:///tmp/no_data'), metrics=metrics)
    trial.temporary_state.restoring_from = result
    trial.on_restore()
    assert trial.run_metadata.last_result == metrics

def test_stacktrace():
    if False:
        print('Hello World!')
    'Test proper stacktrace is printed for RayTaskError.'
    CMD = '\nfrom ray import tune\n\ndef train_fn(config):\n    raise Exception("Inducing exception for testing purposes.")\n\ntune.run(train_fn, num_samples=1)\n    '
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_string_as_driver(CMD)
    assert 'Inducing exception for testing purposes.' in exc_info.value.output.decode()
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__] + sys.argv[1:]))