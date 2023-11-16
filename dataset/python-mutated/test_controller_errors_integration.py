import os
from collections import Counter
import pytest
import sys
import ray
from ray.train import CheckpointConfig
from ray.air.execution import FixedResourceManager, PlacementGroupResourceManager
from ray.tune import PlacementGroupFactory, TuneError
from ray.tune.execution.tune_controller import TuneController
from ray.tune.experiment import Trial
from ray.tune.registry import TRAINABLE_CLASS, _global_registry
from ray.tune.schedulers import FIFOScheduler
from ray.tune.search import BasicVariantGenerator
from ray.train.tests.util import mock_storage_context
from ray.tune.tests.execution.utils import BudgetResourceManager
STORAGE = mock_storage_context()

@pytest.fixture(scope='function')
def ray_start_4_cpus_2_gpus_extra():
    if False:
        for i in range(10):
            print('nop')
    address_info = ray.init(num_cpus=4, num_gpus=2, resources={'a': 2})
    yield address_info
    ray.shutdown()

def create_mock_components():
    if False:
        print('Hello World!')

    class _MockScheduler(FIFOScheduler):
        errored_trials = []

        def on_trial_error(self, tune_controller, trial):
            if False:
                return 10
            self.errored_trials += [trial]

    class _MockSearchAlg(BasicVariantGenerator):
        errored_trials = []

        def on_trial_complete(self, trial_id, error=False, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            if error:
                self.errored_trials += [trial_id]
    searchalg = _MockSearchAlg()
    scheduler = _MockScheduler()
    return (searchalg, scheduler)

@pytest.mark.parametrize('resource_manager_cls', [FixedResourceManager, PlacementGroupResourceManager])
def test_invalid_trainable(ray_start_4_cpus_2_gpus_extra, resource_manager_cls):
    if False:
        print('Hello World!')
    'An invalid trainable should make the trial fail on startup.\n\n    The controller itself should continue. Other trials should run.\n\n    Legacy test: test_trial_runner_2.py::TrialRunnerTest::testErrorHandling\n    '
    runner = TuneController(resource_manager_factory=lambda : resource_manager_cls(), storage=STORAGE)
    kwargs = {'stopping_criterion': {'training_iteration': 1}, 'placement_group_factory': PlacementGroupFactory([{'CPU': 1, 'GPU': 1}]), 'storage': STORAGE}
    _global_registry.register(TRAINABLE_CLASS, 'asdf', None)
    trials = [Trial('asdf', **kwargs), Trial('__fake', **kwargs)]
    for t in trials:
        runner.add_trial(t)
    while not trials[1].status == Trial.RUNNING:
        runner.step()
    assert trials[0].status == Trial.ERROR
    assert trials[1].status == Trial.RUNNING

def test_overstep(ray_start_4_cpus_2_gpus_extra):
    if False:
        return 10
    'Stepping when trials are finished should raise a TuneError.\n\n    Legacy test: test_trial_runner_2.py::TrialRunnerTest::testThrowOnOverstep\n    '
    os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = '1'
    runner = TuneController(resource_manager_factory=lambda : BudgetResourceManager({'CPU': 4}), storage=STORAGE)
    runner.step()
    with pytest.raises(TuneError):
        runner.step()

@pytest.mark.parametrize('resource_manager_cls', [FixedResourceManager, PlacementGroupResourceManager])
@pytest.mark.parametrize('max_failures_persistent', [(0, False), (1, False), (2, True)])
def test_failure_recovery(ray_start_4_cpus_2_gpus_extra, resource_manager_cls, max_failures_persistent):
    if False:
        while True:
            i = 10
    'Test failure recover with `max_failures`.\n\n    Trials should be retried up to `max_failures` times.\n\n    Legacy test: test_trial_runner_2.py::TrialRunnerTest::testFailureRecoveryDisabled\n    Legacy test: test_trial_runner_2.py::TrialRunnerTest::testFailureRecoveryEnabled\n    Legacy test: test_trial_runner_2.py::TrialRunnerTest::testFailureRecoveryMaxFailures\n    '
    (max_failures, persistent_error) = max_failures_persistent
    (searchalg, scheduler) = create_mock_components()
    runner = TuneController(search_alg=searchalg, scheduler=scheduler, resource_manager_factory=lambda : resource_manager_cls(), storage=STORAGE)
    kwargs = {'placement_group_factory': PlacementGroupFactory([{'CPU': 1, 'GPU': 1}]), 'stopping_criterion': {'training_iteration': 2}, 'checkpoint_config': CheckpointConfig(checkpoint_frequency=1), 'max_failures': max_failures, 'config': {'mock_error': True, 'persistent_error': persistent_error}, 'storage': STORAGE}
    runner.add_trial(Trial('__fake', **kwargs))
    trials = runner.get_trials()
    while not runner.is_finished():
        runner.step()
    if persistent_error or not max_failures:
        assert trials[0].status == Trial.ERROR
        num_failures = max_failures + 1
        assert trials[0].num_failures == num_failures
        assert len(searchalg.errored_trials) == 1
        assert len(scheduler.errored_trials) == num_failures
    else:
        assert trials[0].status == Trial.TERMINATED
        assert trials[0].num_failures == 1
        assert len(searchalg.errored_trials) == 0
        assert len(scheduler.errored_trials) == 1

@pytest.mark.parametrize('resource_manager_cls', [FixedResourceManager, PlacementGroupResourceManager])
@pytest.mark.parametrize('fail_fast', [True, TuneController.RAISE])
def test_fail_fast(ray_start_4_cpus_2_gpus_extra, resource_manager_cls, fail_fast):
    if False:
        print('Hello World!')
    'Test fail_fast feature.\n\n    If fail_fast=True, after the first failure, all other trials should be terminated\n    (because we end the experiment).\n\n    If fail_fast=RAISE, after the first failure, we should raise an error.\n\n    Legacy test: test_trial_runner_2.py::TrialRunnerTest::testFailFast\n    Legacy test: test_trial_runner_2.py::TrialRunnerTest::testFailFastRaise\n    '
    runner = TuneController(resource_manager_factory=lambda : resource_manager_cls(), fail_fast=fail_fast, storage=STORAGE)
    kwargs = {'placement_group_factory': PlacementGroupFactory([{'CPU': 1, 'GPU': 1}]), 'checkpoint_config': CheckpointConfig(checkpoint_frequency=1), 'max_failures': 0, 'config': {'mock_error': True, 'persistent_error': True}, 'storage': STORAGE}
    runner.add_trial(Trial('__fake', **kwargs))
    runner.add_trial(Trial('__fake', **kwargs))
    trials = runner.get_trials()
    if fail_fast == TuneController.RAISE:
        with pytest.raises(Exception):
            while not runner.is_finished():
                runner.step()
        runner.cleanup()
        return
    else:
        while not runner.is_finished():
            runner.step()
    status_count = Counter((t.status for t in trials))
    assert status_count.get(Trial.ERROR) == 1
    assert status_count.get(Trial.TERMINATED) == 1
    with pytest.raises(TuneError):
        runner.step()
if __name__ == '__main__':
    sys.exit(pytest.main(['-v', __file__]))