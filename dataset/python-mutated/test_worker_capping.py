import asyncio
import os
import pytest
import sys
import tempfile
import time
import ray
from ray._private.test_utils import Semaphore

def test_nested_tasks(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    ray.init(num_cpus=1)

    @ray.remote
    class Counter:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.count = 0

        def inc(self):
            if False:
                print('Hello World!')
            self.count += 1
            assert self.count < 20

        def dec(self):
            if False:
                while True:
                    i = 10
            self.count -= 1
    counter = Counter.remote()

    @ray.remote(num_cpus=1)
    def g():
        if False:
            i = 10
            return i + 15
        return None

    @ray.remote(num_cpus=1)
    def f():
        if False:
            for i in range(10):
                print('nop')
        ray.get(counter.inc.remote())
        res = ray.get(g.remote())
        ray.get(counter.dec.remote())
        return res
    (ready, _) = ray.wait([f.remote() for _ in range(1000)], timeout=60.0, num_returns=1000)
    assert len(ready) == 1000, len(ready)
    ray.get(ready)

def test_recursion(shutdown_only):
    if False:
        i = 10
        return i + 15
    ray.init(num_cpus=1)

    @ray.remote
    def summer(n):
        if False:
            print('Hello World!')
        if n == 0:
            return 0
        return n + ray.get(summer.remote(n - 1))
    assert ray.get(summer.remote(10)) == sum(range(11))

def test_out_of_order_scheduling(shutdown_only):
    if False:
        return 10
    "Ensure that when a task runs before its dependency, and they're of the same\n    scheduling class, the dependency is eventually able to run."
    ray.init(num_cpus=1)

    @ray.remote
    def foo(arg, path):
        if False:
            for i in range(10):
                print('nop')
        (ref,) = arg
        should_die = not os.path.exists(path)
        with open(path, 'w') as f:
            f.write('')
        if should_die:
            print('dying!!!')
            os._exit(-1)
        if ref:
            print('hogging the only available slot for a while')
            ray.get(ref)
            return 'done!'
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f'{tmpdir}/temp.txt'
        first = foo.remote((None,), path)
        second = foo.remote((first,), path)
        print(ray.get(second))

def test_limit_concurrency(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    ray.init(num_cpus=1)
    block_task = Semaphore.remote(0)
    block_driver = Semaphore.remote(0)
    ray.get([block_task.locked.remote(), block_driver.locked.remote()])

    @ray.remote(num_cpus=1)
    def foo():
        if False:
            i = 10
            return i + 15
        ray.get(block_driver.release.remote())
        ray.get(block_task.acquire.remote())
    refs = [foo.remote() for _ in range(20)]
    block_driver_refs = [block_driver.acquire.remote() for _ in range(20)]
    (ready, not_ready) = ray.wait(block_driver_refs, timeout=10, num_returns=20)
    assert len(not_ready) >= 1
    ray.get([block_task.release.remote() for _ in range(19)])
    (ready, not_ready) = ray.wait(block_driver_refs, timeout=10, num_returns=20)
    assert len(not_ready) == 0
    (ready, not_ready) = ray.wait(refs, num_returns=20, timeout=15)
    assert len(ready) == 19
    assert len(not_ready) == 1

def test_zero_cpu_scheduling(shutdown_only):
    if False:
        i = 10
        return i + 15
    ray.init(num_cpus=1)
    block_task = Semaphore.remote(0)
    block_driver = Semaphore.remote(0)

    @ray.remote(num_cpus=0)
    def foo():
        if False:
            while True:
                i = 10
        ray.get(block_driver.release.remote())
        ray.get(block_task.acquire.remote())
    foo.remote()
    foo.remote()
    ray.get(block_driver.acquire.remote())
    block_driver_ref = block_driver.acquire.remote()
    timeout_value = 5 if sys.platform == 'win32' else 1
    (_, not_ready) = ray.wait([block_driver_ref], timeout=timeout_value)
    assert len(not_ready) == 0

def test_exponential_wait(shutdown_only):
    if False:
        print('Hello World!')
    ray.init(num_cpus=2)
    num_tasks = 6

    @ray.remote(num_cpus=0)
    class Barrier:

        def __init__(self, limit):
            if False:
                print('Hello World!')
            self.i = 0
            self.limit = limit

        async def join(self):
            self.i += 1
            while self.i < self.limit:
                await asyncio.sleep(1)
    b = Barrier.remote(num_tasks)

    @ray.remote
    def f(i, start):
        if False:
            while True:
                i = 10
        delta = time.time() - start
        print('Launch', i, delta)
        ray.get(b.join.remote())
        return delta
    start = time.time()
    results = ray.get([f.remote(i, start) for i in range(num_tasks)])
    last_wait = results[-1] - results[-2]
    second_last = results[-2] - results[-3]
    assert second_last < last_wait < 4 * second_last
    assert 7 < last_wait

def test_spillback(ray_start_cluster):
    if False:
        print('Hello World!')
    'Ensure that we can spillback without waiting for the worker cap to be lifed'
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=1, resources={'head': 1}, _system_config={'worker_cap_initial_backoff_delay_ms': 36000000, 'worker_cap_max_backoff_delay_ms': 36000000})
    cluster.wait_for_nodes()
    ray.init(address=cluster.address)

    @ray.remote(num_cpus=0)
    class Counter:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.i = 0

        def inc(self):
            if False:
                return 10
            self.i = self.i + 1

        def get(self):
            if False:
                while True:
                    i = 10
            return self.i
    counter = Counter.remote()

    @ray.remote
    def get_node_id():
        if False:
            while True:
                i = 10
        return ray.get_runtime_context().get_node_id()

    @ray.remote
    def func(i, counter):
        if False:
            print('Hello World!')
        if i == 0:
            counter.inc.remote()
            while True:
                time.sleep(1)
        else:
            return ray.get_runtime_context().get_node_id()
    refs = [func.remote(i, counter) for i in range(2)]
    while ray.get(counter.get.remote()) != 1:
        time.sleep(0.1)
    time.sleep(1)
    cluster.add_node(num_cpus=1, resources={'worker': 1})
    worker_node_id = ray.get(get_node_id.options(num_cpus=0, resources={'worker': 1}).remote())
    assert ray.get(refs[1]) == worker_node_id
    ray.cancel(refs[0], force=True)

def test_idle_workers(shutdown_only):
    if False:
        return 10
    ray.init(num_cpus=2, _system_config={'idle_worker_killing_time_threshold_ms': 10})

    @ray.remote(num_cpus=0)
    class Actor:

        def get(self):
            if False:
                print('Hello World!')
            pass

    @ray.remote
    def getpid():
        if False:
            for i in range(10):
                print('nop')
        time.sleep(0.1)
        return os.getpid()
    for _ in range(3):
        pids = set(ray.get([getpid.remote() for _ in range(4)]))
        assert len(pids) <= 2, pids
        time.sleep(0.1)
    a1 = Actor.remote()
    a2 = Actor.remote()
    ray.get([a1.get.remote(), a2.get.remote()])
    for _ in range(3):
        pids = set(ray.get([getpid.remote() for _ in range(4)]))
        assert len(pids) <= 2, pids
        time.sleep(0.1)
    del a1
    del a2
    for _ in range(3):
        pids = set(ray.get([getpid.remote() for _ in range(4)]))
        assert len(pids) <= 2, pids
        time.sleep(0.1)
if __name__ == '__main__':
    os.environ['RAY_worker_cap_enabled'] = 'true'
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))