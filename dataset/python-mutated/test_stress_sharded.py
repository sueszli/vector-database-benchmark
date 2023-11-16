import numpy as np
import pytest
import ray

@pytest.fixture(params=[1])
def ray_start_sharded(request):
    if False:
        i = 10
        return i + 15
    ray.init(object_store_memory=int(0.5 * 10 ** 9), num_cpus=10, _redis_max_memory=10 ** 8)
    yield None
    ray.shutdown()

def test_submitting_many_tasks(ray_start_sharded):
    if False:
        print('Hello World!')

    @ray.remote
    def f(x):
        if False:
            while True:
                i = 10
        return 1

    def g(n):
        if False:
            i = 10
            return i + 15
        x = 1
        for i in range(n):
            x = f.remote(x)
        return x
    ray.get([g(100) for _ in range(100)])
    assert ray._private.services.remaining_processes_alive()

def test_submitting_many_actors_to_one(ray_start_sharded):
    if False:
        for i in range(10):
            print('nop')

    @ray.remote
    class Actor:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def ping(self):
            if False:
                i = 10
                return i + 15
            return

    @ray.remote
    class Worker:

        def __init__(self, actor):
            if False:
                for i in range(10):
                    print('nop')
            self.actor = actor

        def ping(self):
            if False:
                while True:
                    i = 10
            return ray.get(self.actor.ping.remote())
    a = Actor.remote()
    workers = [Worker.remote(a) for _ in range(10)]
    for _ in range(10):
        out = ray.get([w.ping.remote() for w in workers])
        assert out == [None for _ in workers]

def test_getting_and_putting(ray_start_sharded):
    if False:
        i = 10
        return i + 15
    for n in range(8):
        x = np.zeros(10 ** n)
        for _ in range(100):
            ray.put(x)
        x_id = ray.put(x)
        for _ in range(1000):
            ray.get(x_id)
    assert ray._private.services.remaining_processes_alive()

def test_getting_many_objects(ray_start_sharded):
    if False:
        print('Hello World!')

    @ray.remote
    def f():
        if False:
            while True:
                i = 10
        return 1
    n = 10 ** 4
    lst = ray.get([f.remote() for _ in range(n)])
    assert lst == n * [1]
    assert ray._private.services.remaining_processes_alive()
if __name__ == '__main__':
    import pytest
    import os
    import sys
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))