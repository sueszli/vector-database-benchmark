from collections import Counter
import pytest
import ray
from ray.air import ResourceRequest
from ray.air.execution import FixedResourceManager, PlacementGroupResourceManager
from ray.air.execution._internal.actor_manager import RayActorManager
RESOURCE_MANAGERS = [FixedResourceManager, PlacementGroupResourceManager]

@pytest.fixture(scope='module')
def ray_start_4_cpus():
    if False:
        return 10
    address_info = ray.init(num_cpus=4)
    yield address_info
    ray.shutdown()

@ray.remote
class Actor:

    def foo(self, val, error: bool=False):
        if False:
            i = 10
            return i + 15
        if error:
            raise RuntimeError
        return val

@pytest.mark.parametrize('resource_manager_cls', RESOURCE_MANAGERS)
def test_resolve(ray_start_4_cpus, resource_manager_cls):
    if False:
        print('Hello World!')
    'Test that the `on_result` callback is invoked when a task completes.\n\n    - Instantiate global data object\n    - Schedule task that returns a value\n    - The callback writes the returned value to the global data object\n    '
    actor_manager = RayActorManager(resource_manager=resource_manager_cls())
    seen = {'data': 0}

    def result_callback(tracked_actor, result):
        if False:
            while True:
                i = 10
        seen['data'] = result
    tracked_actor = actor_manager.add_actor(cls=Actor, kwargs={}, resource_request=ResourceRequest([{'CPU': 4}]))
    actor_manager.schedule_actor_task(tracked_actor, 'foo', (4, False), on_result=result_callback)
    actor_manager.next()
    actor_manager.next()
    assert seen['data'] == 4

@pytest.mark.parametrize('resource_manager_cls', RESOURCE_MANAGERS)
@pytest.mark.parametrize('num_tasks', [1, 10, 100])
def test_resolve_many(ray_start_4_cpus, resource_manager_cls, num_tasks):
    if False:
        return 10
    'Schedule ``num_tasks`` tasks and wait until ``wait_for_events`` of them resolve.\n\n    Every resolved task will increase a counter by its return value (1).\n    '
    actor_manager = RayActorManager(resource_manager=resource_manager_cls())
    seen = {'data': 0}

    def result_callback(tracked_actor, result):
        if False:
            while True:
                i = 10
        seen['data'] += result
    tracked_actor = actor_manager.add_actor(cls=Actor, kwargs={}, resource_request=ResourceRequest([{'CPU': 4}]))
    actor_manager.next()
    for i in range(num_tasks):
        actor_manager.schedule_actor_task(tracked_actor, 'foo', (1, False), on_result=result_callback)
    for i in range(num_tasks):
        actor_manager.next()
        assert seen['data'] == i + 1

@pytest.mark.parametrize('resource_manager_cls', RESOURCE_MANAGERS)
def test_error_noop(ray_start_4_cpus, resource_manager_cls):
    if False:
        print('Hello World!')
    'When no `on_error` callback is specified, errors should be ignored.'
    actor_manager = RayActorManager(resource_manager=resource_manager_cls())
    tracked_actor = actor_manager.add_actor(cls=Actor, kwargs={}, resource_request=ResourceRequest([{'CPU': 4}]))
    actor_manager.schedule_actor_task(tracked_actor, 'foo', (1, True))
    actor_manager.next()
    actor_manager.next()

@pytest.mark.parametrize('resource_manager_cls', RESOURCE_MANAGERS)
def test_error_custom(ray_start_4_cpus, resource_manager_cls):
    if False:
        return 10
    'When an `on_error` callback is specified, it is invoked.'
    actor_manager = RayActorManager(resource_manager=resource_manager_cls())
    stats = Counter()

    def error_callback(tracked_actor, exception):
        if False:
            while True:
                i = 10
        stats['exception'] += 1
    tracked_actor = actor_manager.add_actor(cls=Actor, kwargs={}, resource_request=ResourceRequest([{'CPU': 4}]))
    actor_manager.schedule_actor_task(tracked_actor, 'foo', (1, True), on_error=error_callback)
    actor_manager.next()
    actor_manager.next()
    assert stats['exception'] == 1
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))