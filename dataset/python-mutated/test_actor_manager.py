import functools
import os
from pathlib import Path
import pickle
import sys
import time
import unittest
import ray
from ray.util.state import list_actors
from ray.rllib.utils.actor_manager import FaultAwareApply, FaultTolerantActorManager

def load_random_numbers():
    if False:
        for i in range(10):
            print('nop')
    'Loads deterministic random numbers from data file.'
    rllib_dir = Path(__file__).parent.parent.parent
    pkl_file = os.path.join(rllib_dir, 'utils', 'tests', 'random_numbers.pkl')
    return pickle.load(open(pkl_file, 'rb'))
RANDOM_NUMS = load_random_numbers()

@ray.remote(max_restarts=-1)
class Actor(FaultAwareApply):

    def __init__(self, i, maybe_crash=True):
        if False:
            while True:
                i = 10
        self.random_numbers = RANDOM_NUMS[i]
        self.count = 0
        self.maybe_crash = maybe_crash
        self.config = {'recreate_failed_workers': True}

    def _maybe_crash(self):
        if False:
            i = 10
            return i + 15
        if not self.maybe_crash:
            return
        r = self.random_numbers[self.count]
        if r < 0.1:
            sys.exit(1)
        elif r < 0.2:
            raise AttributeError('sorry')

    def call(self):
        if False:
            while True:
                i = 10
        self.count += 1
        self._maybe_crash()
        return self.count

    def ping(self):
        if False:
            return 10
        self._maybe_crash()
        return 'pong'

def wait_for_restore():
    if False:
        while True:
            i = 10
    'Wait for Ray actor fault tolerence to restore all failed actors.'
    while True:
        states = [a['state'] == 'ALIVE' or a['state'] == 'DEAD' for a in list_actors(filters=[('class_name', '=', 'Actor')])]
        print('waiting ... ', states)
        if all(states):
            break
        time.sleep(0.5)

class TestActorManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            print('Hello World!')
        ray.init()

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            return 10
        ray.shutdown()

    def test_sync_call_healthy_only(self):
        if False:
            print('Hello World!')
        'Test synchronous remote calls to only healthy actors.'
        actors = [Actor.remote(i) for i in range(4)]
        manager = FaultTolerantActorManager(actors=actors)
        results = []
        for _ in range(10):
            results.extend(manager.foreach_actor(lambda w: w.call()).ignore_errors())
            wait_for_restore()
        self.assertEqual(len(results), 7)
        manager.clear()

    def test_sync_call_all_actors(self):
        if False:
            return 10
        'Test synchronous remote calls to all actors, regardless of their states.'
        actors = [Actor.remote(i) for i in range(4)]
        manager = FaultTolerantActorManager(actors=actors)
        results = []
        for _ in range(10):
            results.extend(manager.foreach_actor(lambda w: w.call(), healthy_only=False))
            wait_for_restore()
        self.assertEqual(len(results), 40)
        self.assertEqual(len([r for r in results if r.ok]), 15)
        manager.clear()

    def test_sync_call_return_obj_refs(self):
        if False:
            while True:
                i = 10
        'Test synchronous remote calls to all actors asking for raw ObjectRefs.'
        actors = [Actor.remote(i, maybe_crash=False) for i in range(4)]
        manager = FaultTolerantActorManager(actors=actors)
        results = list(manager.foreach_actor(lambda w: w.call(), healthy_only=False, return_obj_refs=True))
        self.assertEqual(len(results), 4)
        for r in results:
            self.assertTrue(r.ok)
            self.assertTrue(isinstance(r.get(), ray.ObjectRef))
        manager.clear()

    def test_sync_call_fire_and_forget(self):
        if False:
            i = 10
            return i + 15
        'Test synchronous remote calls with 0 timeout_seconds.'
        actors = [Actor.remote(i, maybe_crash=False) for i in range(4)]
        manager = FaultTolerantActorManager(actors=actors)
        results1 = []
        for _ in range(10):
            manager.probe_unhealthy_actors(mark_healthy=True)
            results1.extend(manager.foreach_actor(lambda w: w.call(), timeout_seconds=0))
            wait_for_restore()
        results2 = [r.get() for r in manager.foreach_actor(lambda w: w.call(), healthy_only=False).ignore_errors()]
        self.assertEqual(results2, [11, 11, 11, 11])
        manager.clear()

    def test_sync_call_same_actor_multiple_times(self):
        if False:
            print('Hello World!')
        'Test multiple synchronous remote calls to the same actor.'
        actors = [Actor.remote(i, maybe_crash=False) for i in range(4)]
        manager = FaultTolerantActorManager(actors=actors)
        results = manager.foreach_actor(lambda w: w.call(), remote_actor_ids=[0, 0])
        self.assertEqual([r.get() for r in results.ignore_errors()], [1, 2])
        manager.clear()

    def test_async_call_same_actor_multiple_times(self):
        if False:
            return 10
        'Test multiple asynchronous remote calls to the same actor.'
        actors = [Actor.remote(i, maybe_crash=False) for i in range(4)]
        manager = FaultTolerantActorManager(actors=actors)
        num_of_calls = manager.foreach_actor_async(lambda w: w.call(), remote_actor_ids=[0, 0])
        self.assertEqual(num_of_calls, 2)
        results = manager.fetch_ready_async_reqs(timeout_seconds=None)
        self.assertEqual([r.get() for r in results.ignore_errors()], [1, 2])
        manager.clear()

    def test_sync_call_not_ignore_error(self):
        if False:
            for i in range(10):
                print('nop')
        'Test synchronous remote calls that returns errors.'
        actors = [Actor.remote(i) for i in range(4)]
        manager = FaultTolerantActorManager(actors=actors)
        results = []
        for _ in range(10):
            manager.probe_unhealthy_actors(mark_healthy=True)
            results.extend(manager.foreach_actor(lambda w: w.call()))
            wait_for_restore()
        self.assertTrue(any([not r.ok for r in results]))
        manager.clear()

    def test_sync_call_not_bringing_back_actors(self):
        if False:
            return 10
        'Test successful remote calls will not bring back actors unless told to.'
        actors = [Actor.remote(i) for i in range(4)]
        manager = FaultTolerantActorManager(actors=actors)
        results = manager.foreach_actor(lambda w: w.call())
        self.assertTrue(any([not r.ok for r in results]))
        wait_for_restore()
        manager.probe_unhealthy_actors()
        self.assertEqual(manager.num_healthy_actors(), 2)
        manager.clear()

    def test_async_call(self):
        if False:
            print('Hello World!')
        'Test asynchronous remote calls work.'
        actors = [Actor.remote(i) for i in range(4)]
        manager = FaultTolerantActorManager(actors=actors)
        results = []
        for _ in range(10):
            manager.foreach_actor_async(lambda w: w.call())
            results.extend(manager.fetch_ready_async_reqs(timeout_seconds=None))
            wait_for_restore()
        self.assertEqual(len([r for r in results if r.ok]), 7)
        self.assertEqual(len([r for r in results if not r.ok]), 4)
        manager.clear()

    def test_async_calls_get_dropped_if_inflight_requests_over_limit(self):
        if False:
            print('Hello World!')
        'Test asynchronous remote calls get dropped if too many in-flight calls.'
        actors = [Actor.remote(i, maybe_crash=False) for i in range(4)]
        manager = FaultTolerantActorManager(actors=actors, max_remote_requests_in_flight_per_actor=2)
        num_of_calls = manager.foreach_actor_async(lambda w: w.call(), remote_actor_ids=[0, 0])
        self.assertEqual(num_of_calls, 2)
        num_of_calls = manager.foreach_actor_async(lambda w: w.call(), healthy_only=False, remote_actor_ids=[0])
        self.assertEqual(num_of_calls, 0)
        manager.clear()

    def test_healthy_only_works_for_list_of_functions(self):
        if False:
            while True:
                i = 10
        'Test healthy only mode works when a list of funcs are provided.'
        actors = [Actor.remote(i) for i in range(4)]
        manager = FaultTolerantActorManager(actors=actors)
        manager.set_actor_state(1, False)
        manager.set_actor_state(2, False)

        def f(id, _):
            if False:
                print('Hello World!')
            return id
        func = [functools.partial(f, i) for i in range(4)]
        manager.foreach_actor_async(func, healthy_only=True)
        results = manager.fetch_ready_async_reqs(timeout_seconds=None)
        self.assertEqual([r.get() for r in results], [0, 3])
        manager.clear()

    def test_len_of_func_not_match_len_of_actors(self):
        if False:
            i = 10
            return i + 15
        'Test healthy only mode works when a list of funcs are provided.'
        actors = [Actor.remote(i) for i in range(4)]
        manager = FaultTolerantActorManager(actors=actors)

        def f(id, _):
            if False:
                i = 10
                return i + 15
            return id
        func = [functools.partial(f, i) for i in range(3)]
        with self.assertRaisesRegexp(AssertionError, 'same number of callables') as _:
            (manager.foreach_actor_async(func, healthy_only=True),)
        manager.clear()

    def test_probe_unhealthy_actors(self):
        if False:
            i = 10
            return i + 15
        'Test probe brings back unhealthy actors.'
        actors = [Actor.remote(i, maybe_crash=False) for i in range(4)]
        manager = FaultTolerantActorManager(actors=actors)
        manager.set_actor_state(1, False)
        manager.set_actor_state(2, False)
        manager.probe_unhealthy_actors(mark_healthy=True)
        self.assertEqual(len(manager.healthy_actor_ids()), 4)

    def test_tags(self):
        if False:
            return 10
        'Test that tags work for async calls.'
        actors = [Actor.remote(i, maybe_crash=False) for i in range(4)]
        manager = FaultTolerantActorManager(actors=actors)
        manager.foreach_actor_async(lambda w: w.ping(), tag='pingpong')
        manager.foreach_actor_async(lambda w: w.call(), tag='call')
        time.sleep(1)
        results_ping_pong = manager.fetch_ready_async_reqs(tags='pingpong', timeout_seconds=5)
        results_call = manager.fetch_ready_async_reqs(tags='call', timeout_seconds=5)
        self.assertEquals(len(list(results_ping_pong)), 4)
        self.assertEquals(len(list(results_call)), 4)
        for result in results_ping_pong:
            data = result.get()
            self.assertEqual(data, 'pong')
            self.assertEqual(result.tag, 'pingpong')
        for result in results_call:
            data = result.get()
            self.assertEqual(data, 1)
            self.assertEqual(result.tag, 'call')
        manager.foreach_actor_async(lambda w: w.ping())
        manager.foreach_actor_async(lambda w: w.call())
        time.sleep(1)
        results = manager.fetch_ready_async_reqs(timeout_seconds=5)
        self.assertEquals(len(list(results)), 8)
        for result in results:
            data = result.get()
            self.assertEqual(result.tag, None)
            if isinstance(data, str):
                self.assertEqual(data, 'pong')
            elif isinstance(data, int):
                self.assertEqual(data, 2)
            else:
                raise ValueError('data is not str or int')
        manager.foreach_actor_async(lambda w: w.ping(), tag='pingpong')
        manager.foreach_actor_async(lambda w: w.call(), tag='call')
        time.sleep(1)
        results = manager.fetch_ready_async_reqs(timeout_seconds=5, tags=['pingpong', 'call'])
        self.assertEquals(len(list(results)), 8)
        for result in results:
            data = result.get()
            if isinstance(data, str):
                self.assertEqual(data, 'pong')
                self.assertEqual(result.tag, 'pingpong')
            elif isinstance(data, int):
                self.assertEqual(data, 3)
                self.assertEqual(result.tag, 'call')
            else:
                raise ValueError('data is not str or int')
        manager.foreach_actor_async(lambda w: w.ping(), tag='pingpong')
        manager.foreach_actor_async(lambda w: w.call(), tag='call')
        time.sleep(1)
        results = manager.fetch_ready_async_reqs(timeout_seconds=5, tags=['incorrect'])
        self.assertEquals(len(list(results)), 0)
        results = manager.fetch_ready_async_reqs(timeout_seconds=5)
        self.assertEquals(len(list(results)), 8)
        for result in results:
            data = result.get()
            if isinstance(data, str):
                self.assertEqual(data, 'pong')
                self.assertEqual(result.tag, 'pingpong')
            elif isinstance(data, int):
                self.assertEqual(data, 4)
                self.assertEqual(result.tag, 'call')
            else:
                raise ValueError('result is not str or int')
if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main(['-v', __file__]))