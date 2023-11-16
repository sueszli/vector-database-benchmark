import pytest
import sys
from ray.tune import PlacementGroupFactory
from ray.tune.tests.execution.utils import create_execution_test_objects, TestingTrial

def test_actor_cached(tmpdir):
    if False:
        print('Hello World!')
    (tune_controller, actor_manger, resource_manager) = create_execution_test_objects(max_pending_trials=8)
    assert not actor_manger.added_actors
    tune_controller.add_trial(TestingTrial('trainable1', stub=True, trial_id='trial1'))
    tune_controller.step()
    (tracked_actor, cls_name, kwargs) = actor_manger.added_actors[0]
    assert cls_name == 'trainable1'

def test_actor_reuse_unstaged(tmpdir):
    if False:
        return 10
    "A trial that hasn't been staged can re-use an actor.\n\n    In specific circumstances, this can lead to errors. Notably, when an\n    external source (e.g. a scheduler) directly calls TuneController APIs,\n    we can be in a situation where a trial has not been staged, but there is\n    still an actor available for it to use (because it hasn't been evicted from\n    the cache, yet).\n\n    This test constructs such a situation an asserts that actor re-use does not\n    lead to errors in those cases.\n    "
    (tune_controller, actor_manger, resource_manager) = create_execution_test_objects(max_pending_trials=1)
    tune_controller._reuse_actors = True
    assert not actor_manger.added_actors
    trialA1 = TestingTrial('trainable1', stub=True, trial_id='trialA1', placement_group_factory=PlacementGroupFactory([{'CPU': 1}]))
    tune_controller.add_trial(trialA1)
    trialB1 = TestingTrial('trainable1', stub=True, trial_id='trialB1', placement_group_factory=PlacementGroupFactory([{'CPU': 5}]))
    tune_controller.add_trial(trialB1)
    trialA2 = TestingTrial('trainable1', stub=True, trial_id='trialA2', placement_group_factory=PlacementGroupFactory([{'CPU': 1}]))
    tune_controller.add_trial(trialA2)
    tune_controller.step()
    actor_manger.set_num_pending(2)
    trialA3 = TestingTrial('trainable1', stub=True, trial_id='trialA3', placement_group_factory=PlacementGroupFactory([{'CPU': 1}]))
    tune_controller.add_trial(trialA3)
    tune_controller.step()
    (tracked_actorA1, _, _) = actor_manger.added_actors[0]
    (tracked_actorB1, _, _) = actor_manger.added_actors[1]
    (tracked_actorA2, _, _) = actor_manger.added_actors[2]
    tune_controller._actor_started(tracked_actorA1)
    tune_controller._on_training_result(trialA1, {'done': True})
    assert trialA2 in tune_controller._staged_trials
    assert trialA3 not in tune_controller._staged_trials
    assert tune_controller._actor_cache.num_cached_objects == 1
    tune_controller._actor_started(tracked_actorA2)
    tune_controller._schedule_trial_stop(trialA2)
    assert tune_controller._actor_cache.num_cached_objects == 1
    tune_controller.step()
    assert actor_manger.scheduled_futures[-1][2] == 'reset'
    tune_controller._on_trial_reset(trialA3, True)
    tune_controller._actor_stopped(tracked_actorA1)
    tune_controller.step()
if __name__ == '__main__':
    sys.exit(pytest.main(['-v', __file__]))