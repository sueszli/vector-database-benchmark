import time
from collections import Counter
import pytest
import ray
from ray.air.execution.resources.placement_group import PlacementGroupResourceManager
from ray.air.execution.resources.request import ResourceRequest
REQUEST_2_CPU = ResourceRequest([{'CPU': 2}])
REQUEST_1_2_CPU = ResourceRequest([{'CPU': 1}, {'CPU': 2}])
REQUEST_0_2_CPU = ResourceRequest([{'CPU': 0}, {'CPU': 2}])

@pytest.fixture
def ray_start_4_cpus():
    if False:
        for i in range(10):
            print('nop')
    address_info = ray.init(num_cpus=4)
    yield address_info
    ray.shutdown()

def _count_pg_states():
    if False:
        print('Hello World!')
    counter = Counter()
    for (_, pg_info) in ray.util.placement_group_table().items():
        counter[pg_info['state']] += 1
    return counter

def test_request_cancel_resources(ray_start_4_cpus):
    if False:
        i = 10
        return i + 15
    'Test that canceling a resource request clears the PG futures.\n\n    - Create request\n    - Assert actual PG is created\n    - Cancel request\n    - Assert staging future is removed\n    - Assert actual PG is removed\n    '
    manager = PlacementGroupResourceManager(update_interval_s=0)
    assert not manager.has_resources_ready(REQUEST_2_CPU)
    manager.request_resources(REQUEST_2_CPU)
    pg_states = _count_pg_states()
    assert pg_states['PENDING'] + pg_states['CREATED'] == 1
    assert pg_states['REMOVED'] == 0
    assert manager.get_resource_futures()
    manager.cancel_resource_request(REQUEST_2_CPU)
    assert not manager.get_resource_futures()
    pg_states = _count_pg_states()
    assert pg_states['PENDING'] + pg_states['CREATED'] == 0
    assert pg_states['REMOVED'] == 1

def test_acquire_return_resources(ray_start_4_cpus):
    if False:
        print('Hello World!')
    'Tests that acquiring and returning resources works.\n\n    - At the start, no resources should be ready (no PG scheduled)\n    - Request resources for 2 CPUs\n    - (wait until they are ready)\n    - Assert that these 2 CPUs are available to be acquired\n    - Acquire\n    - Assert that there are no 2 CPU resources available anymore\n    - Free resources\n    - Assert that the 2 CPU resources are still not available (no new request)\n        - This is also tested in includes test_request_cancel_resources\n    '
    manager = PlacementGroupResourceManager(update_interval_s=0)
    assert not manager.has_resources_ready(REQUEST_2_CPU)
    manager.request_resources(REQUEST_2_CPU)
    ray.wait(manager.get_resource_futures(), num_returns=1)
    assert manager.has_resources_ready(REQUEST_2_CPU)
    pg_states = _count_pg_states()
    assert pg_states['CREATED'] == 1
    assert pg_states['REMOVED'] == 0
    acquired = manager.acquire_resources(REQUEST_2_CPU)
    assert not manager.has_resources_ready(REQUEST_2_CPU)
    manager.free_resources(acquired)
    assert not manager.has_resources_ready(REQUEST_2_CPU)
    pg_states = _count_pg_states()
    assert pg_states['CREATED'] == 0
    assert pg_states['REMOVED'] == 1

def test_request_pending(ray_start_4_cpus):
    if False:
        for i in range(10):
            print('nop')
    'Test that requesting too many resources leads to pending PGs.\n\n    - Cluster of 4 CPUs\n    - Request 3 PGs a 2 CPUs\n    - Acquire 2 PGs\n    - Assert no resources are available anymore\n    - Return both PGs\n    - Assert resources are available again\n    - Cancel request\n    - Assert no resources are available again\n    '
    manager = PlacementGroupResourceManager(update_interval_s=0)
    assert not manager.has_resources_ready(REQUEST_2_CPU)
    manager.request_resources(REQUEST_2_CPU)
    manager.request_resources(REQUEST_2_CPU)
    manager.request_resources(REQUEST_2_CPU)
    ray.wait(manager.get_resource_futures(), num_returns=2)
    assert manager.has_resources_ready(REQUEST_2_CPU)
    assert len(manager.get_resource_futures()) == 1
    pg_states = _count_pg_states()
    assert pg_states['CREATED'] == 2
    assert pg_states['PENDING'] == 1
    assert pg_states['REMOVED'] == 0
    acq1 = manager.acquire_resources(REQUEST_2_CPU)
    acq2 = manager.acquire_resources(REQUEST_2_CPU)
    assert not manager.has_resources_ready(REQUEST_2_CPU)
    manager.free_resources(acq1)
    manager.free_resources(acq2)
    ray.wait(manager.get_resource_futures(), num_returns=1)
    assert manager.has_resources_ready(REQUEST_2_CPU)
    pg_states = _count_pg_states()
    assert pg_states['CREATED'] == 1
    assert pg_states['PENDING'] == 0
    assert pg_states['REMOVED'] == 2
    manager.cancel_resource_request(REQUEST_2_CPU)
    assert not manager.has_resources_ready(REQUEST_2_CPU)
    pg_states = _count_pg_states()
    assert pg_states['CREATED'] == 0
    assert pg_states['PENDING'] == 0
    assert pg_states['REMOVED'] == 3

def test_acquire_unavailable(ray_start_4_cpus):
    if False:
        for i in range(10):
            print('nop')
    'Test that acquiring resources that are not available returns None.\n\n    - Try to acquire\n    - Assert this does not work\n    - Request resources\n    - Wait until ready\n    - Acquire\n    - Assert this did work\n    '
    manager = PlacementGroupResourceManager(update_interval_s=0)
    assert not manager.acquire_resources(REQUEST_2_CPU)
    manager.request_resources(REQUEST_2_CPU)
    ray.wait(manager.get_resource_futures(), num_returns=1)
    assert manager.acquire_resources(REQUEST_2_CPU)

def test_bind_two_bundles(ray_start_4_cpus):
    if False:
        for i in range(10):
            print('nop')
    'Test that binding two remote objects to a ready resource works.\n\n    - Request PG with 2 bundles (1 CPU and 2 CPUs)\n    - Bind two remote tasks to these bundles, execute\n    - Assert that resource allocation returns the correct resources: 1 CPU and 2 CPUs\n    '
    manager = PlacementGroupResourceManager(update_interval_s=0)
    manager.request_resources(REQUEST_1_2_CPU)
    ray.wait(manager.get_resource_futures(), num_returns=1)
    assert manager.has_resources_ready(REQUEST_1_2_CPU)

    @ray.remote
    def get_assigned_resources():
        if False:
            for i in range(10):
                print('nop')
        return ray.get_runtime_context().get_assigned_resources()
    acq = manager.acquire_resources(REQUEST_1_2_CPU)
    [av1] = acq.annotate_remote_entities([get_assigned_resources])
    res1 = ray.get(av1.remote())
    assert res1 == {'CPU': 1}
    [av1, av2] = acq.annotate_remote_entities([get_assigned_resources, get_assigned_resources])
    (res1, res2) = ray.get([av1.remote(), av2.remote()])
    assert res1 == {'CPU': 1}
    assert res2 == {'CPU': 2}

def test_bind_empty_head_bundle(ray_start_4_cpus):
    if False:
        for i in range(10):
            print('nop')
    'Test that binding two remote objects to a ready resource works with empty head.\n\n    - Request PG with 2 bundles (0 CPU and 2 CPUs)\n    - Bind two remote tasks to these bundles, execute\n    - Assert that resource allocation returns the correct resources: 0 CPU and 2 CPUs\n    '
    manager = PlacementGroupResourceManager(update_interval_s=0)
    assert REQUEST_0_2_CPU.head_bundle_is_empty
    manager.request_resources(REQUEST_0_2_CPU)
    ray.wait(manager.get_resource_futures(), num_returns=1)
    assert manager.has_resources_ready(REQUEST_0_2_CPU)

    @ray.remote
    def get_assigned_resources():
        if False:
            print('Hello World!')
        return ray.get_runtime_context().get_assigned_resources()
    acq = manager.acquire_resources(REQUEST_0_2_CPU)
    [av1] = acq.annotate_remote_entities([get_assigned_resources])
    res1 = ray.get(av1.remote())
    assert res1 == {}
    [av1, av2] = acq.annotate_remote_entities([get_assigned_resources, get_assigned_resources])
    (res1, res2) = ray.get([av1.remote(), av2.remote()])
    assert res1 == {}
    assert res2 == {'CPU': 2}

def test_capture_child_tasks(ray_start_4_cpus):
    if False:
        while True:
            i = 10
    'Test that child tasks are captured when creating placement groups.\n\n    - Request PG with 2 bundles (1 CPU and 2 CPUs)\n    - Bind a remote task that needs 2 CPUs to run\n    - Assert that it can be scheduled from within the first bundle\n\n    This is only the case if child tasks are captured in the placement groups, as\n    there is only 1 CPU available outside (on a 4 CPU cluster). The 2 CPUs\n    thus have to come from the placement group.\n    '
    manager = PlacementGroupResourceManager(update_interval_s=0)
    manager.request_resources(REQUEST_1_2_CPU)
    ray.wait(manager.get_resource_futures(), num_returns=1)
    assert manager.has_resources_ready(REQUEST_1_2_CPU)

    @ray.remote
    def needs_cpus():
        if False:
            return 10
        return 'Ok'

    @ray.remote
    def spawn_child_task(num_cpus: int):
        if False:
            i = 10
            return i + 15
        return ray.get(needs_cpus.options(num_cpus=num_cpus).remote())
    acq = manager.acquire_resources(REQUEST_1_2_CPU)
    [av1] = acq.annotate_remote_entities([spawn_child_task])
    res = ray.get(av1.remote(2), timeout=2.0)
    assert res

def test_clear_state(ray_start_4_cpus):
    if False:
        i = 10
        return i + 15
    'Test that clearing state will remove existing placement groups.\n\n    - Create resource request\n    - Wait until PG is scheduled\n    - Assert that Ray PG is created\n    - Call `mgr.clear()`\n    - Assert that resources are not ready anymore\n    - Assert that Ray PG is removed\n    '
    manager = PlacementGroupResourceManager(update_interval_s=0)
    manager.request_resources(REQUEST_1_2_CPU)
    ray.wait(manager.get_resource_futures(), num_returns=1)
    assert manager.has_resources_ready(REQUEST_1_2_CPU)
    pg_states = _count_pg_states()
    assert pg_states['CREATED'] == 1
    assert pg_states['PENDING'] == 0
    assert pg_states['REMOVED'] == 0
    manager.clear()
    assert not manager.has_resources_ready(REQUEST_1_2_CPU)
    pg_states = _count_pg_states()
    assert pg_states['CREATED'] == 0
    assert pg_states['PENDING'] == 0
    assert pg_states['REMOVED'] == 1

def test_internal_state(ray_start_4_cpus):
    if False:
        return 10
    'Test internal state mappings of the placement group manager.\n\n    This test makes assumptions and assertions around the internal state transition\n    of private properties of the placement group resource manager.\n\n    If you change internal handling logic of the manager, you may need to change this\n    test as well.\n    '
    manager = PlacementGroupResourceManager(update_interval_s=0)
    assert manager.update_interval_s == 0
    manager.has_resources_ready(REQUEST_2_CPU)
    assert not manager._request_to_ready_pgs[REQUEST_2_CPU]
    manager.request_resources(REQUEST_2_CPU)
    assert manager._request_to_staged_pgs[REQUEST_2_CPU]
    pg = list(manager._request_to_staged_pgs[REQUEST_2_CPU])[0]
    assert manager._pg_to_request[pg] == REQUEST_2_CPU
    assert manager._pg_to_staging_future[pg]
    fut = manager._pg_to_staging_future[pg]
    assert manager._staging_future_to_pg[fut] == pg
    while not manager.has_resources_ready(resource_request=REQUEST_2_CPU):
        time.sleep(0.05)
    assert manager._request_to_ready_pgs[REQUEST_2_CPU]
    assert not manager._request_to_staged_pgs[REQUEST_2_CPU]
    assert not manager._pg_to_staging_future
    assert not manager._staging_future_to_pg
    manager.cancel_resource_request(REQUEST_2_CPU)
    assert not manager._request_to_ready_pgs[REQUEST_2_CPU]
    assert not manager._pg_to_request
    manager.request_resources(REQUEST_2_CPU)
    manager.cancel_resource_request(REQUEST_2_CPU)
    assert not manager._pg_to_staging_future
    assert not manager._staging_future_to_pg
    assert not manager._request_to_staged_pgs[REQUEST_2_CPU]
    assert not manager._request_to_ready_pgs[REQUEST_2_CPU]
    assert not manager._pg_to_request
    manager.request_resources(REQUEST_2_CPU)
    pg = list(manager._request_to_staged_pgs[REQUEST_2_CPU])[0]
    while not manager.has_resources_ready(resource_request=REQUEST_2_CPU):
        time.sleep(0.05)
    acquired_resources = manager.acquire_resources(resource_request=REQUEST_2_CPU)
    assert not manager._pg_to_staging_future
    assert not manager._staging_future_to_pg
    assert not manager._request_to_staged_pgs[REQUEST_2_CPU]
    assert not manager._request_to_ready_pgs[REQUEST_2_CPU]
    assert manager._pg_to_request
    assert pg in manager._acquired_pgs
    manager.free_resources(acquired_resources)
    assert not manager._pg_to_request
    assert not manager._acquired_pgs
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))