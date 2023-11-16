import sys
import os
import ray
import logging
from typing import Dict
from collections import Counter
import pytest
logger = logging.getLogger(__name__)

def my_threads() -> Dict[str, int]:
    if False:
        while True:
            i = 10
    '\n    Returns [(thread_id, thread_name)]\n    '
    pid = os.getpid()
    threads = Counter()
    proc_dir = f'/proc/{pid}/task'
    for tid_entry in os.listdir(proc_dir):
        comm_path = os.path.join(proc_dir, tid_entry, 'comm')
        if os.path.exists(comm_path):
            with open(comm_path, 'r') as comm_file:
                thread_name = comm_file.read().strip()
                threads[thread_name] += 1
    return threads
KNOWN_THREADS = {'grpc_global_tim', 'grpcpp_sync_ser', 'jemalloc_bg_thd'}

def assert_threads_are_bounded(prev_threads: Dict[str, int], now_threads: Dict[str, int]):
    if False:
        i = 10
        return i + 15
    '\n    Asserts that the threads did not grow unexpected.\n    Rule: For each (thread_name, count) in now_threads, it must either be <= the number\n    in prev_threads, or in KNOWN_THREADS.\n    '
    for (thread_name, count) in now_threads.items():
        if thread_name not in KNOWN_THREADS:
            target = prev_threads.get(thread_name, 0)
            assert count <= target, f'{thread_name} grows unexpectedly: expected <= {target}, got {count}. prev {prev_threads}, now: {now_threads}'

@ray.remote
def fibonacci(a, i):
    if False:
        return 10
    if i < 2:
        return 1
    f1 = fibonacci.remote(a, i - 1)
    f2 = fibonacci.remote(a, i - 2)
    return ray.get(a.add.remote(f1, f2))

@pytest.mark.skipif(sys.platform != 'linux', reason='procfs only works on linux.')
def test_threaded_actor_have_bounded_num_of_threads(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    ray.init()

    @ray.remote
    class A:

        def get_my_threads(self):
            if False:
                print('Hello World!')
            return my_threads()

        def add(self, i, j):
            if False:
                i = 10
                return i + 15
            return i + j
    a = A.options(max_concurrency=2).remote()
    prev_threads = ray.get(a.get_my_threads.remote())
    assert ray.get(fibonacci.remote(a, 1)) == 1
    now_threads = ray.get(a.get_my_threads.remote())
    assert_threads_are_bounded(prev_threads, now_threads)
    assert ray.get(fibonacci.remote(a, 10)) == 89
    now_threads = ray.get(a.get_my_threads.remote())
    assert_threads_are_bounded(prev_threads, now_threads)

@pytest.mark.skipif(sys.platform != 'linux', reason='procfs only works on linux.')
def test_async_actor_have_bounded_num_of_threads(shutdown_only):
    if False:
        return 10
    ray.init()

    @ray.remote
    class A:

        async def get_my_threads(self):
            return my_threads()

        async def add(self, i, j):
            return i + j
    a = A.options(max_concurrency=2).remote()
    prev_threads = ray.get(a.get_my_threads.remote())
    assert ray.get(fibonacci.remote(a, 1)) == 1
    now_threads = ray.get(a.get_my_threads.remote())
    assert_threads_are_bounded(prev_threads, now_threads)
    assert ray.get(fibonacci.remote(a, 10)) == 89
    now_threads = ray.get(a.get_my_threads.remote())
    assert_threads_are_bounded(prev_threads, now_threads)

@pytest.mark.skipif(sys.platform != 'linux', reason='procfs only works on linux.')
def test_async_actor_cg_have_bounded_num_of_threads(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    ray.init()

    @ray.remote(concurrency_groups={'io': 2, 'compute': 4})
    class A:

        async def get_my_threads(self):
            return my_threads()

        @ray.method(concurrency_group='io')
        async def io_add(self, i, j):
            return i + j

        @ray.method(concurrency_group='compute')
        async def compute_add(self, i, j):
            return i + j

        async def default_add(self, i, j):
            return i + j

    @ray.remote
    def fibonacci_cg(a, i):
        if False:
            return 10
        if i < 2:
            return 1
        f1 = fibonacci_cg.remote(a, i - 1)
        f2 = fibonacci_cg.remote(a, i - 2)
        assert ray.get(a.io_add.remote(1, 2)) == 3
        assert ray.get(a.compute_add.remote(4, 5)) == 9
        return ray.get(a.default_add.remote(f1, f2))
    a = A.options(max_concurrency=2).remote()
    prev_threads = ray.get(a.get_my_threads.remote())
    assert ray.get(fibonacci_cg.remote(a, 1)) == 1
    now_threads = ray.get(a.get_my_threads.remote())
    assert_threads_are_bounded(prev_threads, now_threads)
    assert ray.get(fibonacci_cg.remote(a, 10)) == 89
    now_threads = ray.get(a.get_my_threads.remote())
    assert_threads_are_bounded(prev_threads, now_threads)
if __name__ == '__main__':
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))