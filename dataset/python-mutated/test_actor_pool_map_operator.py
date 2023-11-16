import collections
import time
import pytest
import ray
from ray.data._internal.compute import ActorPoolStrategy
from ray.data._internal.execution.operators.actor_pool_map_operator import AutoscalingConfig, AutoscalingPolicy, _ActorPool
from ray.data._internal.execution.util import make_ref_bundles
from ray.tests.conftest import *

@ray.remote
class PoolWorker:

    def __init__(self, node_id: str='node1'):
        if False:
            return 10
        self.node_id = node_id

    def get_location(self) -> str:
        if False:
            while True:
                i = 10
        return self.node_id

class TestActorPool:

    def _add_ready_worker(self, pool: _ActorPool) -> ray.actor.ActorHandle:
        if False:
            return 10
        actor = PoolWorker.remote()
        ready_ref = actor.get_location.remote()
        pool.add_pending_actor(actor, ready_ref)
        ray.get(ready_ref)
        has_actor = pool.pending_to_running(ready_ref)
        assert has_actor
        return actor

    def test_add_pending(self, ray_start_regular_shared):
        if False:
            return 10
        pool = _ActorPool()
        actor = PoolWorker.remote()
        ready_ref = actor.get_location.remote()
        pool.add_pending_actor(actor, ready_ref)
        assert pool.pick_actor() is None
        assert pool.num_total_actors() == 1
        assert pool.num_pending_actors() == 1
        assert pool.num_running_actors() == 0
        assert pool.num_active_actors() == 0
        assert pool.num_idle_actors() == 0
        assert pool.num_free_slots() == 0
        assert pool.get_pending_actor_refs() == [ready_ref]

    def test_pending_to_running(self, ray_start_regular_shared):
        if False:
            while True:
                i = 10
        pool = _ActorPool()
        actor = self._add_ready_worker(pool)
        picked_actor = pool.pick_actor()
        assert picked_actor == actor
        assert pool.num_total_actors() == 1
        assert pool.num_pending_actors() == 0
        assert pool.num_running_actors() == 1
        assert pool.num_active_actors() == 1
        assert pool.num_idle_actors() == 0
        assert pool.num_free_slots() == 3

    def test_repeated_picking(self, ray_start_regular_shared):
        if False:
            while True:
                i = 10
        pool = _ActorPool(max_tasks_in_flight=999)
        actor = self._add_ready_worker(pool)
        for _ in range(10):
            picked_actor = pool.pick_actor()
            assert picked_actor == actor

    def test_return_actor(self, ray_start_regular_shared):
        if False:
            print('Hello World!')
        pool = _ActorPool(max_tasks_in_flight=999)
        self._add_ready_worker(pool)
        for _ in range(10):
            picked_actor = pool.pick_actor()
        for _ in range(10):
            pool.return_actor(picked_actor)
        with pytest.raises(AssertionError):
            pool.return_actor(picked_actor)
        assert pool.num_total_actors() == 1
        assert pool.num_pending_actors() == 0
        assert pool.num_running_actors() == 1
        assert pool.num_active_actors() == 0
        assert pool.num_idle_actors() == 1
        assert pool.num_free_slots() == 999

    def test_pick_max_tasks_in_flight(self, ray_start_regular_shared):
        if False:
            return 10
        pool = _ActorPool(max_tasks_in_flight=2)
        actor = self._add_ready_worker(pool)
        assert pool.num_free_slots() == 2
        assert pool.pick_actor() == actor
        assert pool.num_free_slots() == 1
        assert pool.pick_actor() == actor
        assert pool.num_free_slots() == 0
        assert pool.pick_actor() is None

    def test_pick_ordering_lone_idle(self, ray_start_regular_shared):
        if False:
            return 10
        pool = _ActorPool()
        self._add_ready_worker(pool)
        pool.pick_actor()
        actor2 = self._add_ready_worker(pool)
        picked_actor = pool.pick_actor()
        assert picked_actor == actor2

    def test_pick_ordering_full_order(self, ray_start_regular_shared):
        if False:
            return 10
        pool = _ActorPool()
        actors = [self._add_ready_worker(pool) for _ in range(4)]
        picked_actors = [pool.pick_actor() for _ in range(4)]
        assert set(picked_actors) == set(actors)
        assert pool.num_total_actors() == 4
        assert pool.num_pending_actors() == 0
        assert pool.num_running_actors() == 4
        assert pool.num_active_actors() == 4
        assert pool.num_idle_actors() == 0

    def test_pick_all_max_tasks_in_flight(self, ray_start_regular_shared):
        if False:
            for i in range(10):
                print('nop')
        pool = _ActorPool(max_tasks_in_flight=2)
        actors = [self._add_ready_worker(pool) for _ in range(4)]
        picked_actors = [pool.pick_actor() for _ in range(8)]
        pick_counts = collections.Counter(picked_actors)
        assert len(pick_counts) == 4
        for (actor, count) in pick_counts.items():
            assert actor in actors
            assert count == 2
        assert pool.pick_actor() is None

    def test_pick_ordering_with_returns(self, ray_start_regular_shared):
        if False:
            while True:
                i = 10
        pool = _ActorPool()
        actor1 = self._add_ready_worker(pool)
        actor2 = self._add_ready_worker(pool)
        picked_actors = [pool.pick_actor() for _ in range(2)]
        assert set(picked_actors) == {actor1, actor2}
        pool.return_actor(actor2)
        assert pool.pick_actor() == actor2

    def test_kill_inactive_pending_actor(self, ray_start_regular_shared):
        if False:
            print('Hello World!')
        pool = _ActorPool()
        actor = PoolWorker.remote()
        ready_ref = actor.get_location.remote()
        pool.add_pending_actor(actor, ready_ref)
        killed = pool.kill_inactive_actor()
        assert killed
        assert pool.get_pending_actor_refs() == []
        time.sleep(1)
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(actor.get_location.remote())
        assert pool.num_total_actors() == 0
        assert pool.num_pending_actors() == 0
        assert pool.num_running_actors() == 0
        assert pool.num_active_actors() == 0
        assert pool.num_idle_actors() == 0
        assert pool.num_free_slots() == 0

    def test_kill_inactive_idle_actor(self, ray_start_regular_shared):
        if False:
            return 10
        pool = _ActorPool()
        actor = self._add_ready_worker(pool)
        killed = pool.kill_inactive_actor()
        assert killed
        assert pool.pick_actor() is None
        time.sleep(1)
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(actor.get_location.remote())
        assert pool.num_total_actors() == 0
        assert pool.num_pending_actors() == 0
        assert pool.num_running_actors() == 0
        assert pool.num_active_actors() == 0
        assert pool.num_idle_actors() == 0
        assert pool.num_free_slots() == 0

    def test_kill_inactive_active_actor_not_killed(self, ray_start_regular_shared):
        if False:
            for i in range(10):
                print('nop')
        pool = _ActorPool()
        actor = self._add_ready_worker(pool)
        assert pool.pick_actor() == actor
        killed = pool.kill_inactive_actor()
        assert not killed
        assert pool.pick_actor() == actor

    def test_kill_inactive_pending_over_idle(self, ray_start_regular_shared):
        if False:
            for i in range(10):
                print('nop')
        pool = _ActorPool()
        pending_actor = PoolWorker.remote()
        ready_ref = pending_actor.get_location.remote()
        pool.add_pending_actor(pending_actor, ready_ref)
        idle_actor = self._add_ready_worker(pool)
        killed = pool.kill_inactive_actor()
        assert killed
        assert pool.pick_actor() == idle_actor
        pool.return_actor(idle_actor)
        assert pool.get_pending_actor_refs() == []
        time.sleep(1)
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(pending_actor.get_location.remote())
        assert pool.num_total_actors() == 1
        assert pool.num_pending_actors() == 0
        assert pool.num_running_actors() == 1
        assert pool.num_active_actors() == 0
        assert pool.num_idle_actors() == 1
        assert pool.num_free_slots() == 4

    def test_kill_all_inactive_pending_actor_killed(self, ray_start_regular_shared):
        if False:
            for i in range(10):
                print('nop')
        pool = _ActorPool()
        actor = PoolWorker.remote()
        ready_ref = actor.get_location.remote()
        pool.add_pending_actor(actor, ready_ref)
        pool.kill_all_inactive_actors()
        assert pool.get_pending_actor_refs() == []
        assert not pool.pending_to_running(ready_ref)
        time.sleep(1)
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(actor.get_location.remote())
        assert pool.num_total_actors() == 0
        assert pool.num_pending_actors() == 0
        assert pool.num_running_actors() == 0
        assert pool.num_active_actors() == 0
        assert pool.num_idle_actors() == 0
        assert pool.num_free_slots() == 0

    def test_kill_all_inactive_idle_actor_killed(self, ray_start_regular_shared):
        if False:
            return 10
        pool = _ActorPool()
        actor = self._add_ready_worker(pool)
        pool.kill_all_inactive_actors()
        assert pool.pick_actor() is None
        time.sleep(1)
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(actor.get_location.remote())
        assert pool.num_total_actors() == 0
        assert pool.num_pending_actors() == 0
        assert pool.num_running_actors() == 0
        assert pool.num_active_actors() == 0
        assert pool.num_idle_actors() == 0
        assert pool.num_free_slots() == 0

    def test_kill_all_inactive_active_actor_not_killed(self, ray_start_regular_shared):
        if False:
            for i in range(10):
                print('nop')
        pool = _ActorPool()
        actor = self._add_ready_worker(pool)
        assert pool.pick_actor() == actor
        pool.kill_all_inactive_actors()
        assert pool.pick_actor() == actor

    def test_kill_all_inactive_future_idle_actors_killed(self, ray_start_regular_shared):
        if False:
            print('Hello World!')
        pool = _ActorPool()
        actor = self._add_ready_worker(pool)
        assert pool.pick_actor() == actor
        pool.kill_all_inactive_actors()
        assert pool.pick_actor() == actor
        for _ in range(2):
            pool.return_actor(actor)
        assert pool.pick_actor() is None
        time.sleep(1)
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(actor.get_location.remote())
        assert pool.num_total_actors() == 0
        assert pool.num_pending_actors() == 0
        assert pool.num_running_actors() == 0
        assert pool.num_active_actors() == 0
        assert pool.num_idle_actors() == 0
        assert pool.num_free_slots() == 0

    def test_kill_all_inactive_mixture(self, ray_start_regular_shared):
        if False:
            i = 10
            return i + 15
        pool = _ActorPool()
        actor1 = self._add_ready_worker(pool)
        assert pool.pick_actor() == actor1
        self._add_ready_worker(pool)
        actor3 = PoolWorker.remote()
        ready_ref = actor3.get_location.remote()
        pool.add_pending_actor(actor3, ready_ref)
        assert pool.num_total_actors() == 3
        assert pool.num_pending_actors() == 1
        assert pool.num_running_actors() == 2
        assert pool.num_active_actors() == 1
        assert pool.num_idle_actors() == 1
        assert pool.num_free_slots() == 7
        pool.kill_all_inactive_actors()
        assert pool.pick_actor() == actor1
        with pytest.raises(AssertionError):
            pool.add_pending_actor(actor3, ready_ref)
        pool.kill_all_inactive_actors()
        assert pool.pick_actor() == actor1
        for _ in range(3):
            pool.return_actor(actor1)
        assert pool.pick_actor() is None
        time.sleep(1)
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(actor1.get_location.remote())
        assert pool.num_total_actors() == 0
        assert pool.num_pending_actors() == 0
        assert pool.num_running_actors() == 0
        assert pool.num_active_actors() == 0
        assert pool.num_idle_actors() == 0
        assert pool.num_free_slots() == 0

    def test_all_actors_killed(self, ray_start_regular_shared):
        if False:
            return 10
        pool = _ActorPool()
        active_actor = self._add_ready_worker(pool)
        assert pool.pick_actor() == active_actor
        idle_actor = self._add_ready_worker(pool)
        pool.kill_all_actors()
        assert pool.pick_actor() is None
        time.sleep(1)
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(idle_actor.get_location.remote())
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(active_actor.get_location.remote())
        assert pool.num_total_actors() == 0
        assert pool.num_pending_actors() == 0
        assert pool.num_running_actors() == 0
        assert pool.num_active_actors() == 0
        assert pool.num_idle_actors() == 0
        assert pool.num_free_slots() == 0

    def test_locality_manager_actor_ranking(self):
        if False:
            for i in range(10):
                print('nop')
        pool = _ActorPool(max_tasks_in_flight=2)
        bundles = make_ref_bundles([[0] for _ in range(10)])
        fake_loc_map = {}
        for (i, b) in enumerate(bundles):
            fake_loc_map[b] = 'node1'
        pool._get_location = lambda b: fake_loc_map[b]
        actor1 = PoolWorker.remote(node_id='node1')
        ready_ref = actor1.get_location.remote()
        pool.add_pending_actor(actor1, ready_ref)
        ray.get(ready_ref)
        pool.pending_to_running(ready_ref)
        actor2 = PoolWorker.remote(node_id='node2')
        ready_ref = actor2.get_location.remote()
        pool.add_pending_actor(actor2, ready_ref)
        ray.get(ready_ref)
        pool.pending_to_running(ready_ref)
        res1 = pool.pick_actor(bundles[0])
        assert res1 == actor1
        res2 = pool.pick_actor(bundles[1])
        assert res2 == actor1
        res3 = pool.pick_actor(bundles[2])
        assert res3 == actor2
        res4 = pool.pick_actor(bundles[3])
        assert res4 == actor2
        res5 = pool.pick_actor(bundles[4])
        assert res5 is None

    def test_locality_manager_busyness_ranking(self):
        if False:
            print('Hello World!')
        pool = _ActorPool(max_tasks_in_flight=2)
        bundles = make_ref_bundles([[0] for _ in range(10)])
        fake_loc_map = {}
        for (i, b) in enumerate(bundles):
            fake_loc_map[b] = None
        pool._get_location = lambda b: fake_loc_map[b]
        actor1 = PoolWorker.remote(node_id='node1')
        ready_ref = actor1.get_location.remote()
        pool.add_pending_actor(actor1, ready_ref)
        ray.get(ready_ref)
        pool.pending_to_running(ready_ref)
        actor2 = PoolWorker.remote(node_id='node1')
        ready_ref = actor2.get_location.remote()
        pool.add_pending_actor(actor2, ready_ref)
        ray.get(ready_ref)
        pool.pending_to_running(ready_ref)
        pool._num_tasks_in_flight[actor2] = 1
        res1 = pool.pick_actor(bundles[0])
        assert res1 == actor1
        pool._num_tasks_in_flight[actor2] = 2
        res2 = pool.pick_actor(bundles[0])
        assert res2 == actor1
        res3 = pool.pick_actor(bundles[0])
        assert res3 is None

class TestAutoscalingConfig:

    def test_min_workers_validation(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValueError):
            AutoscalingConfig(min_workers=0, max_workers=2)

    def test_max_workers_validation(self):
        if False:
            return 10
        with pytest.raises(ValueError):
            AutoscalingConfig(min_workers=3, max_workers=2)

    def test_max_tasks_in_flight_validation(self):
        if False:
            return 10
        with pytest.raises(ValueError):
            AutoscalingConfig(min_workers=1, max_workers=2, max_tasks_in_flight=0)

    def test_full_specification(self):
        if False:
            while True:
                i = 10
        config = AutoscalingConfig(min_workers=2, max_workers=100, max_tasks_in_flight=3, ready_to_total_workers_ratio=0.8, idle_to_total_workers_ratio=0.25)
        assert config.min_workers == 2
        assert config.max_workers == 100
        assert config.max_tasks_in_flight == 3
        assert config.ready_to_total_workers_ratio == 0.8
        assert config.idle_to_total_workers_ratio == 0.25

    def test_from_compute(self):
        if False:
            i = 10
            return i + 15
        compute = ActorPoolStrategy(min_size=2, max_size=5, max_tasks_in_flight_per_actor=3)
        config = AutoscalingConfig.from_compute_strategy(compute)
        assert config.min_workers == 2
        assert config.max_workers == 5
        assert config.max_tasks_in_flight == 3
        assert config.ready_to_total_workers_ratio == 0.8
        assert config.idle_to_total_workers_ratio == 0.5

class TestAutoscalingPolicy:

    def test_min_workers(self):
        if False:
            return 10
        config = AutoscalingConfig(min_workers=1, max_workers=4)
        policy = AutoscalingPolicy(config)
        assert policy.min_workers == 1

    def test_max_workers(self):
        if False:
            print('Hello World!')
        config = AutoscalingConfig(min_workers=1, max_workers=4)
        policy = AutoscalingPolicy(config)
        assert policy.max_workers == 4

    def test_should_scale_up_over_min_workers(self):
        if False:
            i = 10
            return i + 15
        config = AutoscalingConfig(min_workers=1, max_workers=4)
        policy = AutoscalingPolicy(config)
        num_total_workers = 0
        num_running_workers = 0
        assert policy.should_scale_up(num_total_workers, num_running_workers)

    def test_should_scale_up_over_max_workers(self):
        if False:
            return 10
        config = AutoscalingConfig(min_workers=1, max_workers=4)
        policy = AutoscalingPolicy(config)
        num_total_workers = 4
        num_running_workers = 4
        assert not policy.should_scale_up(num_total_workers, num_running_workers)
        num_total_workers = 3
        num_running_workers = 3
        assert policy.should_scale_up(num_total_workers, num_running_workers)

    def test_should_scale_up_ready_to_total_ratio(self):
        if False:
            i = 10
            return i + 15
        config = AutoscalingConfig(min_workers=1, max_workers=4, ready_to_total_workers_ratio=0.5)
        policy = AutoscalingPolicy(config)
        num_total_workers = 2
        num_running_workers = 1
        assert not policy.should_scale_up(num_total_workers, num_running_workers)
        num_total_workers = 3
        num_running_workers = 2
        assert policy.should_scale_up(num_total_workers, num_running_workers)

    def test_should_scale_down_min_workers(self):
        if False:
            print('Hello World!')
        config = AutoscalingConfig(min_workers=2, max_workers=4)
        policy = AutoscalingPolicy(config)
        num_total_workers = 2
        num_idle_workers = 2
        assert not policy.should_scale_down(num_total_workers, num_idle_workers)
        num_total_workers = 3
        num_idle_workers = 3
        assert policy.should_scale_down(num_total_workers, num_idle_workers)

    def test_should_scale_down_idle_to_total_ratio(self):
        if False:
            i = 10
            return i + 15
        config = AutoscalingConfig(min_workers=1, max_workers=4, idle_to_total_workers_ratio=0.5)
        policy = AutoscalingPolicy(config)
        num_total_workers = 4
        num_idle_workers = 1
        assert not policy.should_scale_down(num_total_workers, num_idle_workers)
        num_total_workers = 4
        num_idle_workers = 3
        assert policy.should_scale_down(num_total_workers, num_idle_workers)

    def test_start_actor_timeout(ray_start_regular_shared):
        if False:
            return 10
        'Tests that ActorPoolMapOperator raises an exception on\n        timeout while waiting for actors.'
        from ray.data._internal.execution.operators import actor_pool_map_operator
        from ray.exceptions import GetTimeoutError
        original_timeout = actor_pool_map_operator.DEFAULT_WAIT_FOR_MIN_ACTORS_SEC
        actor_pool_map_operator.DEFAULT_WAIT_FOR_MIN_ACTORS_SEC = 1
        with pytest.raises(GetTimeoutError, match='Timed out while starting actors. This may mean that the cluster does not have enough resources for the requested actor pool.'):
            ray.data.range(10).map_batches(lambda x: x, batch_size=1, compute=ray.data.ActorPoolStrategy(size=5), num_gpus=100).take_all()
        actor_pool_map_operator.DEFAULT_WAIT_FOR_MIN_ACTORS_SEC = original_timeout
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))