import os
import sys
import pytest
import ray
from ray import serve
from ray._private.test_utils import wait_for_condition
from ray.serve._private.utils import get_all_live_placement_group_names
from ray.serve.context import _get_global_client
from ray.util.placement_group import PlacementGroup, get_current_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

def _get_pg_strategy(pg: PlacementGroup) -> str:
    if False:
        while True:
            i = 10
    return ray.util.placement_group_table(pg)['strategy']

@pytest.mark.skipif(sys.platform == 'win32', reason='Timing out on Windows.')
def test_basic(serve_instance):
    if False:
        return 10
    'Test the basic workflow: multiple replicas with their own PGs.'

    @serve.deployment(num_replicas=2, placement_group_bundles=[{'CPU': 1}, {'CPU': 0.1}])
    class D:

        def get_pg(self) -> PlacementGroup:
            if False:
                while True:
                    i = 10
            return get_current_placement_group()
    h = serve.run(D.bind(), name='pg_test')
    assert len(get_all_live_placement_group_names()) == 2
    unique_pgs = set(ray.get([h.get_pg.remote()._to_object_ref_sync() for _ in range(20)]))
    assert len(unique_pgs) == 2
    for pg in unique_pgs:
        assert _get_pg_strategy(pg) == 'PACK'
        assert pg.bundle_specs == [{'CPU': 1}, {'CPU': 0.1}]
    serve.delete('pg_test')
    assert len(get_all_live_placement_group_names()) == 0

@pytest.mark.skipif(sys.platform == 'win32', reason='Timing out on Windows.')
def test_upgrade_and_change_pg(serve_instance):
    if False:
        i = 10
        return i + 15
    'Test re-deploying a deployment with different PG bundles and strategy.'

    @serve.deployment(num_replicas=1, placement_group_bundles=[{'CPU': 1}, {'CPU': 0.1}], placement_group_strategy='STRICT_PACK')
    class D:

        def get_pg(self) -> PlacementGroup:
            if False:
                while True:
                    i = 10
            return get_current_placement_group()
    h = serve.run(D.bind(), name='pg_test')
    assert len(get_all_live_placement_group_names()) == 1
    original_pg = h.get_pg.remote().result()
    assert original_pg.bundle_specs == [{'CPU': 1}, {'CPU': 0.1}]
    assert _get_pg_strategy(original_pg) == 'STRICT_PACK'
    D = D.options(placement_group_bundles=[{'CPU': 2}, {'CPU': 0.2}], placement_group_strategy='SPREAD')
    h = serve.run(D.bind(), name='pg_test')
    assert len(get_all_live_placement_group_names()) == 1
    new_pg = h.get_pg.remote().result()
    assert new_pg.bundle_specs == [{'CPU': 2}, {'CPU': 0.2}]
    assert _get_pg_strategy(new_pg) == 'SPREAD'
    serve.delete('pg_test')
    assert len(get_all_live_placement_group_names()) == 0

@pytest.mark.skipif(sys.platform == 'win32', reason='Timing out on Windows.')
def test_pg_removed_on_replica_graceful_shutdown(serve_instance):
    if False:
        return 10
    'Verify that PGs are removed when a replica shuts down gracefully.'

    @serve.deployment(placement_group_bundles=[{'CPU': 1}])
    class D:

        def get_pg(self) -> PlacementGroup:
            if False:
                for i in range(10):
                    print('nop')
            return get_current_placement_group()
    h = serve.run(D.options(num_replicas=2).bind(), name='pg_test')
    assert len(get_all_live_placement_group_names()) == 2
    original_unique_pgs = set(ray.get([h.get_pg.remote()._to_object_ref_sync() for _ in range(20)]))
    assert len(original_unique_pgs) == 2
    h = serve.run(D.options(num_replicas=1).bind(), name='pg_test')
    assert len(get_all_live_placement_group_names()) == 1
    new_unique_pgs = set(ray.get([h.get_pg.remote()._to_object_ref_sync() for _ in range(20)]))
    assert len(new_unique_pgs) == 1
    assert not new_unique_pgs.issubset(original_unique_pgs)
    serve.delete('pg_test')
    assert len(get_all_live_placement_group_names()) == 0

@pytest.mark.skipif(sys.platform == 'win32', reason='Timing out on Windows.')
def test_pg_removed_on_replica_crash(serve_instance):
    if False:
        while True:
            i = 10
    'Verify that PGs are removed when a replica crashes unexpectedly.'

    @serve.deployment(placement_group_bundles=[{'CPU': 1}], health_check_period_s=0.1)
    class D:

        def die(self):
            if False:
                return 10
            os._exit(1)

        def get_pg(self) -> PlacementGroup:
            if False:
                i = 10
                return i + 15
            return get_current_placement_group()
    h = serve.run(D.bind(), name='pg_test')
    assert len(get_all_live_placement_group_names()) == 1
    pg = h.get_pg.remote().result()
    with pytest.raises(ray.exceptions.RayActorError):
        h.die.remote().result()

    def new_replica_scheduled():
        if False:
            i = 10
            return i + 15
        try:
            h.get_pg.remote().result()
        except ray.exceptions.RayActorError:
            return False
        return True
    wait_for_condition(new_replica_scheduled)
    new_pg = h.get_pg.remote().result()
    assert pg != new_pg
    assert len(get_all_live_placement_group_names()) == 1

@pytest.mark.skipif(sys.platform == 'win32', reason='Timing out on Windows.')
def test_pg_removed_after_controller_crash(serve_instance):
    if False:
        print('Hello World!')
    'Verify that PGs are removed normally after recovering from a controller crash.\n\n    If the placement group was not properly recovered in the replica recovery process,\n    it would be leaked here.\n    '

    @serve.deployment(placement_group_bundles=[{'CPU': 1}])
    class D:
        pass
    serve.run(D.bind(), name='pg_test')
    assert len(get_all_live_placement_group_names()) == 1
    ray.kill(_get_global_client()._controller, no_restart=False)
    serve.delete('pg_test')
    assert len(get_all_live_placement_group_names()) == 0

@pytest.mark.skipif(sys.platform == 'win32', reason='Timing out on Windows.')
def test_leaked_pg_removed_on_controller_recovery(serve_instance):
    if False:
        return 10
    'Verify that leaked PGs are removed on controller recovery.\n\n    A placement group can be "leaked" if the replica is killed while the controller is\n    down or the controller crashes between creating a placement group and its replica.\n\n    In these cases, the controller should detect the leak on recovery and delete the\n    leaked placement group(s).\n    '

    @serve.deployment(placement_group_bundles=[{'CPU': 1}], health_check_period_s=0.1)
    class D:

        def die(self):
            if False:
                i = 10
                return i + 15
            os._exit(1)

        def get_pg(self) -> PlacementGroup:
            if False:
                i = 10
                return i + 15
            return get_current_placement_group()
    h = serve.run(D.bind(), name='pg_test')
    prev_pg = h.get_pg.remote().result()
    assert len(get_all_live_placement_group_names()) == 1
    ray.kill(_get_global_client()._controller, no_restart=False)
    with pytest.raises(ray.exceptions.RayActorError):
        h.die.remote().result()

    def leaked_pg_cleaned_up():
        if False:
            for i in range(10):
                print('nop')
        try:
            new_pg = h.get_pg.remote().result()
        except ray.exceptions.RayActorError:
            return False
        return len(get_all_live_placement_group_names()) == 1 and new_pg != prev_pg
    wait_for_condition(leaked_pg_cleaned_up)
    serve.delete('pg_test')
    assert len(get_all_live_placement_group_names()) == 0

@pytest.mark.skipif(sys.platform == 'win32', reason='Timing out on Windows.')
def test_replica_actor_infeasible(serve_instance):
    if False:
        i = 10
        return i + 15
    "Test that we get a validation error if the replica doesn't fit in the bundle."

    class Infeasible:
        pass
    with pytest.raises(ValueError):
        serve.deployment(placement_group_bundles=[{'CPU': 0.1}])(Infeasible)
    with pytest.raises(ValueError):
        serve.deployment(Infeasible).options(placement_group_bundles=[{'CPU': 0.1}])

@pytest.mark.skipif(sys.platform == 'win32', reason='Timing out on Windows.')
def test_coschedule_actors_and_tasks(serve_instance):
    if False:
        return 10
    "Test that actor/tasks are placed in the replica's placement group by default."

    @ray.remote(num_cpus=1)
    class TestActor:

        def get_pg(self) -> PlacementGroup:
            if False:
                for i in range(10):
                    print('nop')
            return get_current_placement_group()

    @ray.remote
    def get_pg():
        if False:
            print('Hello World!')
        return get_current_placement_group()

    @serve.deployment(placement_group_bundles=[{'CPU': 1}, {'CPU': 1}])
    class Parent:

        def run_test(self):
            if False:
                print('Hello World!')
            a1 = TestActor.remote()
            assert ray.get(a1.get_pg.remote()) == get_current_placement_group()
            a2 = TestActor.remote()
            (ready, _) = ray.wait([a2.get_pg.remote()], timeout=0.1)
            assert len(ready) == 0
            ray.kill(a2)
            a3 = TestActor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=None)).remote()
            assert ray.get(a3.get_pg.remote()) is None
            assert ray.get(get_pg.options(num_cpus=0).remote()) == get_current_placement_group()
            with pytest.raises(ValueError):
                get_pg.options(num_cpus=2).remote()
            assert ray.get(get_pg.options(num_cpus=2, scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=None)).remote()) is None
    h = serve.run(Parent.bind())
    h.run_test.remote().result()
if __name__ == '__main__':
    sys.exit(pytest.main(['-v', '-s', __file__]))