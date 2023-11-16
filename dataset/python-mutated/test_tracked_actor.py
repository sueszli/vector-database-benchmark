import threading
import time
from collections import Counter
import gc
from typing import Any, Optional, Type
import pytest
import ray
from ray.air import ResourceRequest
from ray.air.execution import FixedResourceManager, PlacementGroupResourceManager
from ray.air.execution._internal import Barrier
from ray.air.execution._internal.actor_manager import RayActorManager

def _raise(exception_type: Type[Exception]=RuntimeError, msg: Optional[str]=None):
    if False:
        print('Hello World!')

    def _raise_exception(*args, **kwargs):
        if False:
            while True:
                i = 10
        raise exception_type(msg)
    return _raise_exception

class Started(RuntimeError):
    pass

class Stopped(RuntimeError):
    pass

class Failed(RuntimeError):
    pass

class Result(RuntimeError):
    pass

@pytest.fixture(scope='module')
def ray_start_4_cpus():
    if False:
        for i in range(10):
            print('nop')
    address_info = ray.init(num_cpus=4)
    yield address_info
    ray.shutdown()

@pytest.fixture
def cleanup():
    if False:
        while True:
            i = 10
    gc.collect()
    yield

class Actor:

    def __init__(self, **kwargs):
        if False:
            return 10
        self.kwargs = kwargs

    def get_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        return self.kwargs

    def task(self, value: Any):
        if False:
            return 10
        return value

@ray.remote(num_cpus=4)
def fn():
    if False:
        i = 10
        return i + 15
    return True

@pytest.mark.parametrize('resource_manager_cls', [FixedResourceManager, PlacementGroupResourceManager])
@pytest.mark.parametrize('actor_cls', [Actor, ray.remote(Actor)])
@pytest.mark.parametrize('kill', [False, True])
def test_start_stop_actor(ray_start_4_cpus, resource_manager_cls, actor_cls, kill):
    if False:
        while True:
            i = 10
    'Test that starting and stopping actors work and invokes a callback.\n\n    - Start an actor\n    - Starting should trigger start callback\n    - Schedule actor task, which should resolve (meaning actor successfully started)\n    - Stop actor, which should resolve and trigger stop callback\n    - Schedule remote fn that takes up all cluster resources. This should resolve,\n      meaning that the actor was stopped successfully.\n    '
    actor_manager = RayActorManager(resource_manager=resource_manager_cls())
    tracked_actor = actor_manager.add_actor(cls=actor_cls, kwargs={'key': 'val'}, resource_request=ResourceRequest([{'CPU': 4}]), on_start=_raise(Started), on_stop=_raise(Stopped), on_error=_raise(Failed))
    with pytest.raises(Started):
        actor_manager.next()
    actor_manager.schedule_actor_task(tracked_actor, 'task', (1,), on_result=_raise(Result))
    with pytest.raises(Result):
        actor_manager.next()
    assert ray.available_resources().get('CPU', 0.0) == 0, ray.available_resources()
    actor_manager.remove_actor(tracked_actor, kill=kill)
    with pytest.raises(Stopped):
        actor_manager.next()
    assert ray.get(fn.remote(), timeout=5)

@pytest.mark.parametrize('resource_manager_cls', [FixedResourceManager, PlacementGroupResourceManager])
def test_start_many_actors(ray_start_4_cpus, resource_manager_cls):
    if False:
        while True:
            i = 10
    'Test that starting more actors than fit onto the cluster works.\n\n    - Request 10 actors\n    - 4 can be started. Assert they are started\n    - Stop 2\n    - Assert 2 are stopped and 2 new ones are started\n    '
    actor_manager = RayActorManager(resource_manager=resource_manager_cls())
    running_actors = []
    stats = Counter()

    def start_callback(tracked_actor):
        if False:
            print('Hello World!')
        running_actors.append(tracked_actor)
        stats['started'] += 1

    def stop_callback(tracked_actor):
        if False:
            return 10
        running_actors.remove(tracked_actor)
        stats['stopped'] += 1
    expected_actors = []
    for i in range(10):
        tracked_actor = actor_manager.add_actor(cls=Actor, kwargs={'key': 'val'}, resource_request=ResourceRequest([{'CPU': 1}]), on_start=start_callback, on_stop=stop_callback, on_error=_raise(Failed))
        expected_actors.append(tracked_actor)
    for i in range(4):
        actor_manager.next()
    assert stats['started'] == 4
    assert stats['stopped'] == 0
    assert len(running_actors) == 4
    assert set(running_actors) == set(expected_actors[:4])
    actor_manager.remove_actor(running_actors[0])
    actor_manager.remove_actor(running_actors[1])
    for i in range(4):
        actor_manager.next()
    assert stats['started'] == 6
    assert stats['stopped'] == 2
    assert len(running_actors) == 4

@pytest.mark.parametrize('resource_manager_cls', [FixedResourceManager, PlacementGroupResourceManager])
@pytest.mark.parametrize('where', ['init', 'fn'])
def test_actor_fail(ray_start_4_cpus, cleanup, resource_manager_cls, where):
    if False:
        return 10
    'Test that actor failures are handled properly.\n\n    - Start actor that either fails on init or in a task (RayActorError)\n    - Schedule task on actor\n    - Assert that the correct callbacks are called\n    '
    actor_manager = RayActorManager(resource_manager=resource_manager_cls())
    stats = Counter()

    @ray.remote
    class FailingActor:

        def __init__(self, where):
            if False:
                return 10
            self._where = where
            if self._where == 'init':
                raise RuntimeError('INIT')

        def fn(self):
            if False:
                while True:
                    i = 10
            if self._where == 'fn':
                raise SystemExit
            return True

    def fail_callback_actor(tracked_actor, exception):
        if False:
            i = 10
            return i + 15
        stats['failed_actor'] += 1

    def fail_callback_task(tracked_actor, exception):
        if False:
            print('Hello World!')
        stats['failed_task'] += 1
    tracked_actor = actor_manager.add_actor(cls=FailingActor, kwargs={'where': where}, resource_request=ResourceRequest([{'CPU': 1}]), on_error=fail_callback_actor)
    if where != 'init':
        actor_manager.next()
        assert stats['failed_actor'] == 0
        assert stats['failed_task'] == 0
        actor_manager.schedule_actor_task(tracked_actor, 'fn', on_error=fail_callback_task)
    actor_manager.next()
    assert stats['failed_actor'] == 1
    assert stats['failed_task'] == bool(where != 'init')

@pytest.mark.parametrize('resource_manager_cls', [FixedResourceManager, PlacementGroupResourceManager])
def test_stop_actor_before_start(ray_start_4_cpus, tmp_path, cleanup, resource_manager_cls):
    if False:
        for i in range(10):
            print('nop')
    'Test that actor failures are handled properly.\n\n    - Start actor that either fails on init or in a task (RayActorError)\n    - Schedule task on actor\n    - Assert that the correct callbacks are called\n    '
    actor_manager = RayActorManager(resource_manager=resource_manager_cls())
    hang_marker = tmp_path / 'hang.txt'

    @ray.remote
    class HangingActor:

        def __init__(self):
            if False:
                return 10
            while not hang_marker.exists():
                time.sleep(0.05)
    tracked_actor = actor_manager.add_actor(HangingActor, kwargs={}, resource_request=ResourceRequest([{'CPU': 1}]), on_start=_raise(RuntimeError, 'Should not have started'), on_stop=_raise(RuntimeError, 'Should not have stopped'))
    while not actor_manager.is_actor_started(tracked_actor):
        actor_manager.next(0.05)
    actor_manager.remove_actor(tracked_actor)
    hang_marker.write_text('')
    while actor_manager.is_actor_started(tracked_actor):
        actor_manager.next(0.05)
    assert actor_manager.num_live_actors == 0

@pytest.mark.parametrize('resource_manager_cls', [FixedResourceManager, PlacementGroupResourceManager])
@pytest.mark.parametrize('start_thread', [False, True])
def test_stop_actor_custom_future(ray_start_4_cpus, tmp_path, cleanup, resource_manager_cls, start_thread):
    if False:
        return 10
    "If we pass a custom stop future, the actor should still be shutdown by GC.\n\n    This should also be the case when we start a thread in the background, as we\n    do e.g. in Ray Tune's function runner.\n    "
    actor_manager = RayActorManager(resource_manager=resource_manager_cls())
    hang_marker = tmp_path / 'hang.txt'
    actor_name = f'stopping_actor_{resource_manager_cls.__name__}_{start_thread}'

    @ray.remote(name=actor_name)
    class HangingStopActor:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self._thread = None
            self._stop_event = threading.Event()
            if start_thread:

                def entrypoint():
                    if False:
                        return 10
                    while True:
                        print('Thread!')
                        time.sleep(1)
                        if self._stop_event.is_set():
                            sys.exit(0)
                self._thread = threading.Thread(target=entrypoint)
                self._thread.start()

        def stop(self):
            if False:
                for i in range(10):
                    print('nop')
            print('Waiting')
            while not hang_marker.exists():
                time.sleep(0.05)
            self._stop_event.set()
            print('stopped')
    start_barrier = Barrier(max_results=1)
    stop_barrier = Barrier(max_results=1)
    tracked_actor = actor_manager.add_actor(HangingStopActor, kwargs={}, resource_request=ResourceRequest([{'CPU': 1}]), on_start=start_barrier.arrive, on_stop=stop_barrier.arrive)
    while not start_barrier.completed:
        actor_manager.next(0.05)
    assert ray.get_actor(actor_name)
    stop_future = actor_manager.schedule_actor_task(tracked_actor, 'stop')
    actor_manager.remove_actor(tracked_actor, kill=False, stop_future=stop_future)
    assert not stop_barrier.completed
    hang_marker.write_text('!')
    while not stop_barrier.completed:
        actor_manager.next(0.05)
    with pytest.raises(ValueError):
        ray.get_actor(actor_name)
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))