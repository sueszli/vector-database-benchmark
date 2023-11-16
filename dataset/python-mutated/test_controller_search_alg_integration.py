import os
import pickle
from collections import Counter
import pytest
import sys
import ray
from ray.air.constants import TRAINING_ITERATION
from ray.air.execution import FixedResourceManager, PlacementGroupResourceManager
from ray.tune import Experiment, PlacementGroupFactory
from ray.tune.execution.tune_controller import TuneController
from ray.tune.experiment import Trial
from ray.tune.schedulers import FIFOScheduler, TrialScheduler
from ray.tune.search import Searcher, ConcurrencyLimiter, Repeater, SearchGenerator
from ray.tune.search._mock import _MockSuggestionAlgorithm
from ray.train.tests.util import mock_storage_context

class TestTuneController(TuneController):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        kwargs.update(dict(storage=mock_storage_context()))
        super().__init__(*args, **kwargs)

@pytest.fixture(scope='function')
def ray_start_8_cpus():
    if False:
        return 10
    address_info = ray.init(num_cpus=8, num_gpus=0)
    yield address_info
    ray.shutdown()

@pytest.fixture(scope='function')
def ray_start_4_cpus_2_gpus_extra():
    if False:
        print('Hello World!')
    address_info = ray.init(num_cpus=4, num_gpus=2, resources={'a': 2})
    yield address_info
    ray.shutdown()

@pytest.mark.parametrize('resource_manager_cls', [FixedResourceManager, PlacementGroupResourceManager])
def test_search_alg_notification(ray_start_4_cpus_2_gpus_extra, resource_manager_cls):
    if False:
        while True:
            i = 10
    'Check that the searchers gets notified of trial results + completions.\n\n    Also check that the searcher is "finished" before the runner, i.e. the runner\n    continues processing trials when the searcher finished.\n\n    Legacy test: test_trial_runner_3.py::TrialRunnerTest::testSearchAlgNotification\n    Legacy test: test_trial_runner_3.py::TrialRunnerTest::testSearchAlgFinished\n    '
    experiment_spec = {'run': '__fake', 'stop': {'training_iteration': 2}}
    experiments = [Experiment.from_json('test', experiment_spec)]
    search_alg = _MockSuggestionAlgorithm()
    searcher = search_alg.searcher
    search_alg.add_configurations(experiments)
    runner = TestTuneController(resource_manager_factory=lambda : resource_manager_cls(), search_alg=search_alg)
    while not search_alg.is_finished():
        runner.step()
    trials = runner.get_trials()
    while trials[0].status != Trial.RUNNING:
        runner.step()
    assert trials[0].status == Trial.RUNNING
    assert search_alg.is_finished()
    assert not runner.is_finished()
    while not runner.is_finished():
        runner.step()
    assert trials[0].status == Trial.TERMINATED
    assert search_alg.is_finished()
    assert runner.is_finished()
    assert searcher.counter['result'] == 1
    assert searcher.counter['complete'] == 1

@pytest.mark.parametrize('resource_manager_cls', [FixedResourceManager, PlacementGroupResourceManager])
def test_search_alg_scheduler_stop(ray_start_4_cpus_2_gpus_extra, resource_manager_cls):
    if False:
        return 10
    'Check that a scheduler-issued stop also notifies the search algorithm.\n\n    Legacy test: test_trial_runner_3.py::TrialRunnerTest::testSearchAlgSchedulerInteraction  # noqa\n    '

    class _MockScheduler(FIFOScheduler):

        def on_trial_result(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            return TrialScheduler.STOP
    experiment_spec = {'run': '__fake', 'stop': {'training_iteration': 5}}
    experiments = [Experiment.from_json('test', experiment_spec)]
    search_alg = _MockSuggestionAlgorithm()
    searcher = search_alg.searcher
    search_alg.add_configurations(experiments)
    runner = TestTuneController(resource_manager_factory=lambda : resource_manager_cls(), search_alg=search_alg, scheduler=_MockScheduler())
    trials = runner.get_trials()
    while not runner.is_finished():
        runner.step()
    assert searcher.counter['result'] == 0
    assert searcher.counter['complete'] == 1
    assert trials[0].last_result[TRAINING_ITERATION] == 1

@pytest.mark.parametrize('resource_manager_cls', [FixedResourceManager, PlacementGroupResourceManager])
def test_search_alg_stalled(ray_start_4_cpus_2_gpus_extra, resource_manager_cls):
    if False:
        print('Hello World!')
    'Checks that runner and searcher state is maintained when stalled.\n\n    We use a concurrency limit of 1, meaning each trial is added one-by-one\n    from the searchers.\n\n    We then run three samples. During the second trial, we stall the searcher,\n    which means we don\'t suggest new trials after it finished.\n\n    In this case, the runner should still be considered "running". Once we unstall,\n    the experiment finishes regularly.\n\n    Legacy test: test_trial_runner_3.py::TrialRunnerTest::testSearchAlgStalled\n    '
    experiment_spec = {'run': '__fake', 'num_samples': 3, 'stop': {'training_iteration': 1}}
    experiments = [Experiment.from_json('test', experiment_spec)]
    search_alg = _MockSuggestionAlgorithm(max_concurrent=1)
    search_alg.add_configurations(experiments)
    searcher = search_alg.searcher
    runner = TestTuneController(resource_manager_factory=lambda : resource_manager_cls(), search_alg=search_alg)
    runner.step()
    trials = runner.get_trials()
    while trials[0].status != Trial.TERMINATED:
        runner.step()
    runner.step()
    trials = runner.get_trials()
    while trials[1].status != Trial.RUNNING:
        runner.step()
    assert trials[1].status == Trial.RUNNING
    assert len(searcher.live_trials) == 1
    searcher.stall = True
    while trials[1].status != Trial.TERMINATED:
        runner.step()
    assert trials[1].status == Trial.TERMINATED
    assert len(searcher.live_trials) == 0
    assert all((trial.is_finished() for trial in trials))
    assert not search_alg.is_finished()
    assert not runner.is_finished()
    searcher.stall = False
    runner.step()
    trials = runner.get_trials()
    while trials[2].status != Trial.RUNNING:
        runner.step()
    assert trials[2].status == Trial.RUNNING
    assert len(searcher.live_trials) == 1
    while trials[2].status != Trial.TERMINATED:
        runner.step()
    assert len(searcher.live_trials) == 0
    assert search_alg.is_finished()
    assert runner.is_finished()

@pytest.mark.parametrize('resource_manager_cls', [FixedResourceManager, PlacementGroupResourceManager])
def test_search_alg_finishes(ray_start_4_cpus_2_gpus_extra, resource_manager_cls):
    if False:
        return 10
    'Empty SearchAlg changing state in `next_trials` does not crash.\n\n    The search algorithm changes to ``finished`` mid-run. This should not\n    affect processing of the experiment.\n\n    Legacy test: test_trial_runner_3.py::TrialRunnerTest::testSearchAlgFinishes\n    '
    os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = '1'

    class FinishFastAlg(_MockSuggestionAlgorithm):
        _index = 0

        def next_trial(self):
            if False:
                i = 10
                return i + 15
            spec = self._experiment.spec
            trial = None
            if self._index < spec['num_samples']:
                trial = Trial(spec.get('run'), stopping_criterion=spec.get('stop'), storage=spec.get('storage'))
            self._index += 1
            if self._index > 4:
                self.set_finished()
            return trial

        def suggest(self, trial_id):
            if False:
                return 10
            return {}
    experiment_spec = {'run': '__fake', 'num_samples': 2, 'stop': {'training_iteration': 1}}
    searcher = FinishFastAlg()
    experiments = [Experiment.from_json('test', experiment_spec)]
    searcher.add_configurations(experiments)
    runner = TestTuneController(resource_manager_factory=lambda : resource_manager_cls(), search_alg=searcher)
    assert not runner.is_finished()
    while len(runner.get_trials()) < 2:
        runner.step()
    assert not searcher.is_finished()
    assert not runner.is_finished()
    searcher_finished_before = False
    while not runner.is_finished():
        runner.step()
        searcher_finished_before = searcher.is_finished()
    assert searcher_finished_before

@pytest.mark.skip('This test is currently flaky as it can fail due to timing issues.')
@pytest.mark.parametrize('resource_manager_cls', [FixedResourceManager, PlacementGroupResourceManager])
def test_searcher_save_restore(ray_start_8_cpus, resource_manager_cls, tmpdir):
    if False:
        i = 10
        return i + 15
    'Searchers state should be saved and restored in the experiment checkpoint.\n\n    Legacy test: test_trial_runner_3.py::TrialRunnerTest::testSearcherSaveRestore\n    '

    def create_searcher():
        if False:
            while True:
                i = 10

        class TestSuggestion(Searcher):

            def __init__(self, index):
                if False:
                    print('Hello World!')
                self.index = index
                self.returned_result = []
                super().__init__(metric='episode_reward_mean', mode='max')

            def suggest(self, trial_id):
                if False:
                    print('Hello World!')
                self.index += 1
                return {'test_variable': self.index}

            def on_trial_complete(self, trial_id, result=None, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                self.returned_result.append(result)

            def save(self, checkpoint_path):
                if False:
                    i = 10
                    return i + 15
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(self.__dict__, f)

            def restore(self, checkpoint_path):
                if False:
                    while True:
                        i = 10
                with open(checkpoint_path, 'rb') as f:
                    self.__dict__.update(pickle.load(f))
        searcher = TestSuggestion(0)
        searcher = ConcurrencyLimiter(searcher, max_concurrent=2)
        searcher = Repeater(searcher, repeat=3, set_index=False)
        search_alg = SearchGenerator(searcher)
        experiment_spec = {'run': '__fake', 'num_samples': 20, 'config': {'sleep': 10}, 'stop': {'training_iteration': 2}, 'resources_per_trial': PlacementGroupFactory([{'CPU': 1}])}
        experiments = [Experiment.from_json('test', experiment_spec)]
        search_alg.add_configurations(experiments)
        return search_alg
    searcher = create_searcher()
    runner = TestTuneController(resource_manager_factory=lambda : resource_manager_cls(), search_alg=searcher, checkpoint_period=-1, experiment_path=str(tmpdir))
    while len(runner.get_trials()) < 6:
        runner.step()
    assert len(runner.get_trials()) == 6, [t.config for t in runner.get_trials()]
    runner.checkpoint()
    trials = runner.get_trials()
    [runner._schedule_trial_stop(t) for t in trials if t.status is not Trial.ERROR]
    runner.cleanup()
    del runner
    searcher = create_searcher()
    runner2 = TestTuneController(resource_manager_factory=lambda : resource_manager_cls(), search_alg=searcher, experiment_path=str(tmpdir), resume='LOCAL')
    assert len(runner2.get_trials()) == 6, [t.config for t in runner2.get_trials()]

    def trial_statuses():
        if False:
            return 10
        return [t.status for t in runner2.get_trials()]

    def num_running_trials():
        if False:
            return 10
        return sum((t.status == Trial.RUNNING for t in runner2.get_trials()))
    while num_running_trials() < 6:
        runner2.step()
    assert len(set(trial_statuses())) == 1
    assert Trial.RUNNING in trial_statuses()
    for i in range(20):
        runner2.step()
        assert 1 <= num_running_trials() <= 6
    evaluated = [t.evaluated_params['test_variable'] for t in runner2.get_trials()]
    count = Counter(evaluated)
    assert all((v <= 3 for v in count.values()))
if __name__ == '__main__':
    sys.exit(pytest.main(['-v', __file__]))