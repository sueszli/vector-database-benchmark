import asyncio
import pytest
import numpy as np
import sys
import time
import gc
import ray
from ray.experimental.state.api import list_actors
RECONSTRUCTION_CONFIG = {'health_check_failure_threshold': 10, 'health_check_period_ms': 100, 'health_check_timeout_ms': 100, 'health_check_initial_delay_ms': 0, 'max_direct_call_object_size': 100, 'task_retry_delay_ms': 100, 'object_timeout_milliseconds': 200, 'fetch_warn_timeout_milliseconds': 1000}

def assert_no_leak():
    if False:
        i = 10
        return i + 15
    gc.collect()
    core_worker = ray._private.worker.global_worker.core_worker
    ref_counts = core_worker.get_all_reference_counts()
    print(ref_counts)
    for rc in ref_counts.values():
        assert rc['local'] == 0
        assert rc['submitted'] == 0
    assert core_worker.get_memory_store_size() == 0

@pytest.mark.parametrize('delay', [True])
def test_reconstruction(monkeypatch, ray_start_cluster, delay):
    if False:
        print('Hello World!')
    with monkeypatch.context() as m:
        if delay:
            m.setenv('RAY_testing_asio_delay_us', 'CoreWorkerService.grpc_server.ReportGeneratorItemReturns=10000:1000000')
        cluster = ray_start_cluster
        cluster.add_node(num_cpus=0, _system_config=RECONSTRUCTION_CONFIG, enable_object_reconstruction=True)
        ray.init(address=cluster.address)
        node_to_kill = cluster.add_node(num_cpus=1, object_store_memory=10 ** 8)
        cluster.wait_for_nodes()

    @ray.remote(num_returns='streaming', max_retries=2)
    def dynamic_generator(num_returns):
        if False:
            return 10
        for i in range(num_returns):
            yield (np.ones(1000000, dtype=np.int8) * i)

    @ray.remote
    def fetch(x):
        if False:
            print('Hello World!')
        return x[0]
    gen = ray.get(dynamic_generator.remote(10))
    refs = []
    for i in range(5):
        refs.append(next(gen))
    cluster.remove_node(node_to_kill, allow_graceful=False)
    node_to_kill = cluster.add_node(num_cpus=1, object_store_memory=10 ** 8)
    for (i, ref) in enumerate(refs):
        print('first trial.')
        print('fetching ', i)
        assert ray.get(fetch.remote(ref)) == i
    cluster.remove_node(node_to_kill, allow_graceful=False)
    node_to_kill = cluster.add_node(num_cpus=1, object_store_memory=10 ** 8)
    for i in range(4):
        refs.append(next(gen))
    for (i, ref) in enumerate(refs):
        print('second trial')
        print('fetching ', i)
        assert ray.get(fetch.remote(ref)) == i
    cluster.remove_node(node_to_kill, allow_graceful=False)
    node_to_kill = cluster.add_node(num_cpus=1, object_store_memory=10 ** 8)
    for i in range(1):
        refs.append(next(gen))
    for (i, ref) in enumerate(refs):
        print('third trial')
        print('fetching ', i)
        with pytest.raises(ray.exceptions.RayTaskError) as e:
            ray.get(fetch.remote(ref))
        assert 'the maximum number of task retries has been exceeded' in str(e.value)

@pytest.mark.parametrize('failure_type', ['exception', 'crash'])
def test_reconstruction_retry_failed(ray_start_cluster, failure_type):
    if False:
        print('Hello World!')
    'Test the streaming generator retry fails in the second retry.'
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=0, _system_config=RECONSTRUCTION_CONFIG, enable_object_reconstruction=True)
    ray.init(address=cluster.address)

    @ray.remote(num_cpus=0)
    class SignalActor:

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.crash = False

        def set(self):
            if False:
                for i in range(10):
                    print('nop')
            self.crash = True

        def get(self):
            if False:
                i = 10
                return i + 15
            return self.crash
    signal = SignalActor.remote()
    ray.get(signal.get.remote())
    node_to_kill = cluster.add_node(num_cpus=1, object_store_memory=10 ** 8)
    cluster.wait_for_nodes()

    @ray.remote(num_returns='streaming')
    def dynamic_generator(num_returns, signal_actor):
        if False:
            for i in range(10):
                print('nop')
        for i in range(num_returns):
            if i == 3:
                should_crash = ray.get(signal_actor.get.remote())
                if should_crash:
                    if failure_type == 'exception':
                        raise Exception
                    else:
                        sys.exit(5)
            time.sleep(1)
            yield (np.ones(1000000, dtype=np.int8) * i)

    @ray.remote
    def fetch(x):
        if False:
            print('Hello World!')
        return x[0]
    gen = ray.get(dynamic_generator.remote(10, signal))
    refs = []
    for i in range(5):
        refs.append(next(gen))
    cluster.remove_node(node_to_kill, allow_graceful=False)
    node_to_kill = cluster.add_node(num_cpus=1, object_store_memory=10 ** 8)
    for (i, ref) in enumerate(refs):
        print('first trial.')
        print('fetching ', i)
        assert ray.get(fetch.remote(ref)) == i
    cluster.remove_node(node_to_kill, allow_graceful=False)
    node_to_kill = cluster.add_node(num_cpus=1, object_store_memory=10 ** 8)
    signal.set.remote()
    for ref in gen:
        refs.append(ref)
    for (i, ref) in enumerate(refs):
        print('second trial')
        print('fetching ', i)
        print(ref)
        if i < 3:
            assert ray.get(fetch.remote(ref)) == i
        else:
            with pytest.raises(ray.exceptions.RayTaskError) as e:
                assert ray.get(fetch.remote(ref)) == i
                assert 'The worker died' in str(e.value)

def test_generator_max_returns(monkeypatch, shutdown_only):
    if False:
        i = 10
        return i + 15
    '\n    Test when generator returns more than system limit values\n    (100 million by default), it fails a task.\n    '
    with monkeypatch.context() as m:
        m.setenv('RAY_max_num_generator_returns', '2')

        @ray.remote(num_returns='streaming')
        def generator_task():
            if False:
                return 10
            for _ in range(3):
                yield 1

        @ray.remote
        def driver():
            if False:
                print('Hello World!')
            gen = generator_task.remote()
            for ref in gen:
                assert ray.get(ref) == 1
        with pytest.raises(ray.exceptions.RayTaskError):
            ray.get(driver.remote())

def test_return_yield_mix(shutdown_only):
    if False:
        return 10
    '\n    Test the case where yield and return is mixed within a\n    generator task.\n    '

    @ray.remote
    def g():
        if False:
            i = 10
            return i + 15
        for i in range(3):
            yield i
            return
    generator = g.options(num_returns='streaming').remote()
    result = []
    for ref in generator:
        result.append(ray.get(ref))
    assert len(result) == 1
    assert result[0] == 0

def test_task_name_not_changed_for_iteration(shutdown_only):
    if False:
        print('Hello World!')
    'Handles https://github.com/ray-project/ray/issues/37147.\n    Verify the task_name is not changed for each iteration in\n    async actor generator task.\n    '

    @ray.remote
    class A:

        async def gen(self):
            task_name = asyncio.current_task().get_name()
            for i in range(5):
                assert task_name == asyncio.current_task().get_name(), f'{task_name} != {asyncio.current_task().get_name()}'
                yield i
            assert task_name == asyncio.current_task().get_name()
    a = A.remote()
    for obj_ref in a.gen.options(num_returns='streaming').remote():
        print(ray.get(obj_ref))

def test_async_actor_concurrent(shutdown_only):
    if False:
        print('Hello World!')
    'Verify the async actor generator tasks are concurrent.'

    @ray.remote
    class A:

        async def gen(self):
            for i in range(5):
                await asyncio.sleep(1)
                yield i
    a = A.remote()

    async def co():
        async for ref in a.gen.options(num_returns='streaming').remote():
            print(await ref)

    async def main():
        await asyncio.gather(co(), co(), co())
    s = time.time()
    asyncio.run(main())
    assert 4.5 < time.time() - s < 6.5

def test_no_memory_store_obj_leak(shutdown_only):
    if False:
        i = 10
        return i + 15
    "Fixes https://github.com/ray-project/ray/issues/38089\n\n    Verify there's no leak from in-memory object store when\n    using a streaming generator.\n    "
    ray.init()

    @ray.remote
    def f():
        if False:
            return 10
        for _ in range(10):
            yield 1
    for _ in range(10):
        for ref in f.options(num_returns='streaming').remote():
            del ref
        time.sleep(0.2)
    core_worker = ray._private.worker.global_worker.core_worker
    assert core_worker.get_memory_store_size() == 0
    assert_no_leak()
    for _ in range(10):
        for ref in f.options(num_returns='streaming').remote():
            break
        time.sleep(0.2)
    del ref
    core_worker = ray._private.worker.global_worker.core_worker
    assert core_worker.get_memory_store_size() == 0
    assert_no_leak()

def test_python_object_leak(shutdown_only):
    if False:
        return 10
    'Make sure the objects are not leaked\n    (due to circular references) when tasks run\n    for all the execution model in Ray actors.\n    '
    ray.init()

    @ray.remote
    class AsyncActor:

        def __init__(self):
            if False:
                return 10
            self.gc_garbage_len = 0

        def get_gc_garbage_len(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.gc_garbage_len

        async def gen(self, fail=False):
            gc.set_debug(gc.DEBUG_SAVEALL)
            gc.collect()
            self.gc_garbage_len = len(gc.garbage)
            print('Objects: ', self.gc_garbage_len)
            if fail:
                print('exception')
                raise Exception
            yield 1

        async def f(self, fail=False):
            gc.set_debug(gc.DEBUG_SAVEALL)
            gc.collect()
            self.gc_garbage_len = len(gc.garbage)
            print('Objects: ', self.gc_garbage_len)
            if fail:
                print('exception')
                raise Exception
            return 1

    @ray.remote
    class A:

        def __init__(self):
            if False:
                print('Hello World!')
            self.gc_garbage_len = 0

        def get_gc_garbage_len(self):
            if False:
                i = 10
                return i + 15
            return self.gc_garbage_len

        def f(self, fail=False):
            if False:
                print('Hello World!')
            gc.set_debug(gc.DEBUG_SAVEALL)
            gc.collect()
            self.gc_garbage_len = len(gc.garbage)
            print('Objects: ', self.gc_garbage_len)
            if fail:
                print('exception')
                raise Exception
            return 1

        def gen(self, fail=False):
            if False:
                print('Hello World!')
            gc.set_debug(gc.DEBUG_SAVEALL)
            gc.collect()
            self.gc_garbage_len = len(gc.garbage)
            print('Objects: ', self.gc_garbage_len)
            if fail:
                print('exception')
                raise Exception
            yield 1

    def verify_regular(actor, fail):
        if False:
            while True:
                i = 10
        for _ in range(100):
            try:
                ray.get(actor.f.remote(fail=fail))
            except Exception:
                pass
        assert ray.get(actor.get_gc_garbage_len.remote()) == 0

    def verify_generator(actor, fail):
        if False:
            while True:
                i = 10
        for _ in range(100):
            for ref in actor.gen.options(num_returns='streaming').remote(fail=fail):
                try:
                    ray.get(ref)
                except Exception:
                    pass
            assert ray.get(actor.get_gc_garbage_len.remote()) == 0
    print('Test regular actors')
    verify_regular(A.remote(), True)
    verify_regular(A.remote(), False)
    print('Test regular actors + generator')
    verify_generator(A.remote(), True)
    verify_generator(A.remote(), False)
    print('Test threaded actors')
    verify_regular(A.options(max_concurrency=10).remote(), True)
    verify_regular(A.options(max_concurrency=10).remote(), False)
    print('Test threaded actors + generator')
    verify_generator(A.options(max_concurrency=10).remote(), True)
    verify_generator(A.options(max_concurrency=10).remote(), False)
    print('Test async actors')
    verify_regular(AsyncActor.remote(), True)
    verify_regular(AsyncActor.remote(), False)
    print('Test async actors + generator')
    verify_generator(AsyncActor.remote(), True)
    verify_generator(AsyncActor.remote(), False)
    assert len(list_actors()) == 12
if __name__ == '__main__':
    import os
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))