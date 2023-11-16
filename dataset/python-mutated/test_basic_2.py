import array
import logging
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
from unittest.mock import MagicMock, patch
import pytest
from ray._private.ray_constants import KV_NAMESPACE_FUNCTION_TABLE
from ray._private.test_utils import client_test_enabled
from ray.cluster_utils import Cluster, cluster_not_supported
from ray.exceptions import GetTimeoutError, RayTaskError
from ray.tests.client_test_utils import create_remote_signal_actor
if client_test_enabled():
    from ray.util.client import ray
else:
    import ray
logger = logging.getLogger(__name__)

def test_variable_number_of_args(shutdown_only):
    if False:
        return 10
    ray.init(num_cpus=1)

    @ray.remote
    def varargs_fct1(*a):
        if False:
            for i in range(10):
                print('nop')
        return ' '.join(map(str, a))

    @ray.remote
    def varargs_fct2(a, *b):
        if False:
            print('Hello World!')
        return ' '.join(map(str, b))
    x = varargs_fct1.remote(0, 1, 2)
    assert ray.get(x) == '0 1 2'
    x = varargs_fct2.remote(0, 1, 2)
    assert ray.get(x) == '1 2'

    @ray.remote
    def f1(*args):
        if False:
            i = 10
            return i + 15
        return args

    @ray.remote
    def f2(x, y, *args):
        if False:
            return 10
        return (x, y, args)
    assert ray.get(f1.remote()) == ()
    assert ray.get(f1.remote(1)) == (1,)
    assert ray.get(f1.remote(1, 2, 3)) == (1, 2, 3)
    with pytest.raises(TypeError):
        f2.remote()
    with pytest.raises(TypeError):
        f2.remote(1)
    assert ray.get(f2.remote(1, 2)) == (1, 2, ())
    assert ray.get(f2.remote(1, 2, 3)) == (1, 2, (3,))
    assert ray.get(f2.remote(1, 2, 3, 4)) == (1, 2, (3, 4))

    def testNoArgs(self):
        if False:
            return 10

        @ray.remote
        def no_op():
            if False:
                for i in range(10):
                    print('nop')
            pass
        self.ray_start()
        ray.get(no_op.remote())

def test_defining_remote_functions(shutdown_only):
    if False:
        i = 10
        return i + 15
    ray.init(num_cpus=3)
    data = [(1, 2, 'a'), [0.0, 1.0, 1 << 62], 1 << 60, {'a': bytes(3)}]

    @ray.remote
    def g():
        if False:
            while True:
                i = 10
        return data
    ray.get(g.remote())

    @ray.remote
    def h():
        if False:
            while True:
                i = 10
        return array.array('d', [1.0, 2.0, 3.0])
    assert ray.get(h.remote()) == array.array('d', [1.0, 2.0, 3.0])

    @ray.remote
    def j():
        if False:
            while True:
                i = 10
        return time.time()
    ray.get(j.remote())

    @ray.remote
    def k(x):
        if False:
            for i in range(10):
                print('nop')
        return x + 1

    @ray.remote
    def k2(x):
        if False:
            while True:
                i = 10
        return ray.get(k.remote(x))

    @ray.remote
    def m(x):
        if False:
            return 10
        return ray.get(k2.remote(x))
    assert ray.get(k.remote(1)) == 2
    assert ray.get(k2.remote(1)) == 2
    assert ray.get(m.remote(1)) == 2

def test_redefining_remote_functions(shutdown_only):
    if False:
        print('Hello World!')
    ray.init(num_cpus=1)

    @ray.remote
    def f(x):
        if False:
            i = 10
            return i + 15
        return x + 1
    assert ray.get(f.remote(0)) == 1

    @ray.remote
    def f(x):
        if False:
            i = 10
            return i + 15
        return x + 10
    while True:
        val = ray.get(f.remote(0))
        assert val in [1, 10]
        if val == 10:
            break
        else:
            logger.info('Still using old definition of f, trying again.')

    @ray.remote
    def g():
        if False:
            return 10
        return nonexistent()
    with pytest.raises(RayTaskError, match='nonexistent'):
        ray.get(g.remote())

    def nonexistent():
        if False:
            print('Hello World!')
        return 1

    @ray.remote
    def g():
        if False:
            while True:
                i = 10
        return nonexistent()
    assert ray.get(g.remote()) == 1

    @ray.remote
    def h(i):
        if False:
            while True:
                i = 10

        @ray.remote
        def j():
            if False:
                print('Hello World!')
            return i
        return j.remote()
    for i in range(20):
        assert ray.get(ray.get(h.remote(i))) == i

def test_call_matrix(shutdown_only):
    if False:
        return 10
    ray.init(object_store_memory=1000 * 1024 * 1024)

    @ray.remote
    class Actor:

        def small_value(self):
            if False:
                return 10
            return 0

        def large_value(self):
            if False:
                while True:
                    i = 10
            return bytes(80 * 1024 * 1024)

        def echo(self, x):
            if False:
                while True:
                    i = 10
            if isinstance(x, list):
                x = ray.get(x[0])
            return x

    @ray.remote
    def small_value():
        if False:
            while True:
                i = 10
        return 0

    @ray.remote
    def large_value():
        if False:
            for i in range(10):
                print('nop')
        return bytes(80 * 1024 * 1024)

    @ray.remote
    def echo(x):
        if False:
            print('Hello World!')
        if isinstance(x, list):
            x = ray.get(x[0])
        return x

    def check(source_actor, dest_actor, is_large, out_of_band):
        if False:
            while True:
                i = 10
        print('CHECKING', 'actor' if source_actor else 'task', 'to', 'actor' if dest_actor else 'task', 'large_object' if is_large else 'small_object', 'out_of_band' if out_of_band else 'in_band')
        if source_actor:
            a = Actor.remote()
            if is_large:
                x_id = a.large_value.remote()
            else:
                x_id = a.small_value.remote()
        elif is_large:
            x_id = large_value.remote()
        else:
            x_id = small_value.remote()
        if out_of_band:
            x_id = [x_id]
        if dest_actor:
            b = Actor.remote()
            x = ray.get(b.echo.remote(x_id))
        else:
            x = ray.get(echo.remote(x_id))
        if is_large:
            assert isinstance(x, bytes)
        else:
            assert isinstance(x, int)
    for is_large in [False, True]:
        for source_actor in [False, True]:
            for dest_actor in [False, True]:
                for out_of_band in [False, True]:
                    check(source_actor, dest_actor, is_large, out_of_band)

def test_actor_call_order(shutdown_only):
    if False:
        i = 10
        return i + 15
    ray.init(num_cpus=4)

    @ray.remote
    def small_value():
        if False:
            i = 10
            return i + 15
        time.sleep(0.01 * random.randint(0, 10))
        return 0

    @ray.remote
    class Actor:

        def __init__(self):
            if False:
                return 10
            self.count = 0

        def inc(self, count, dependency):
            if False:
                i = 10
                return i + 15
            assert count == self.count
            self.count += 1
            return count
    a = Actor.remote()
    assert ray.get([a.inc.remote(i, small_value.remote()) for i in range(100)]) == list(range(100))

def test_actor_pass_by_ref_order_optimization(shutdown_only):
    if False:
        return 10
    ray.init(num_cpus=4)

    @ray.remote
    class Actor:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def f(self, x):
            if False:
                print('Hello World!')
            pass
    a = Actor.remote()

    @ray.remote
    def fast_value():
        if False:
            print('Hello World!')
        print('fast value')
        pass

    @ray.remote
    def slow_value():
        if False:
            for i in range(10):
                print('nop')
        print('start sleep')
        time.sleep(30)

    @ray.remote
    def runner(f):
        if False:
            i = 10
            return i + 15
        print('runner', a, f)
        return ray.get(a.f.remote(f.remote()))
    runner.remote(slow_value)
    time.sleep(1)
    x2 = runner.remote(fast_value)
    start = time.time()
    ray.get(x2)
    delta = time.time() - start
    assert delta < 10, 'did not skip slow value'

@pytest.mark.parametrize('ray_start_cluster', [{'num_cpus': 1, 'num_nodes': 1}, {'num_cpus': 1, 'num_nodes': 2}], indirect=True)
def test_call_chain(ray_start_cluster):
    if False:
        print('Hello World!')

    @ray.remote
    def g(x):
        if False:
            for i in range(10):
                print('nop')
        return x + 1
    x = 0
    for _ in range(100):
        x = g.remote(x)
    assert ray.get(x) == 100

@pytest.mark.xfail(cluster_not_supported, reason='cluster not supported')
@pytest.mark.skipif(client_test_enabled(), reason='init issue')
def test_system_config_when_connecting(ray_start_cluster):
    if False:
        for i in range(10):
            print('nop')
    config = {'object_timeout_milliseconds': 200}
    cluster = Cluster()
    cluster.add_node(_system_config=config, object_store_memory=100 * 1024 * 1024)
    cluster.wait_for_nodes()
    with pytest.raises(ValueError):
        ray.init(address=cluster.address, _system_config=config)
    ray.init(address=cluster.address)
    obj_ref = ray.put(bytes(40 * 1024 * 1024))
    for _ in range(5):
        put_ref = ray.put(bytes(40 * 1024 * 1024))
    del put_ref
    ray.get(obj_ref)

def test_get_multiple(ray_start_regular_shared):
    if False:
        print('Hello World!')
    object_refs = [ray.put(i) for i in range(10)]
    assert ray.get(object_refs) == list(range(10))
    indices = [random.choice(range(10)) for i in range(5)]
    indices += indices
    results = ray.get([object_refs[i] for i in indices])
    assert results == indices

def test_get_with_timeout(ray_start_regular_shared):
    if False:
        i = 10
        return i + 15
    SignalActor = create_remote_signal_actor(ray)
    signal = SignalActor.remote()
    start = time.time()
    ray.get(signal.wait.remote(should_wait=False), timeout=30)
    assert time.time() - start < 30
    result_id = signal.wait.remote()
    with pytest.raises(GetTimeoutError):
        ray.get(result_id, timeout=0.1)
    assert issubclass(GetTimeoutError, TimeoutError)
    with pytest.raises(TimeoutError):
        ray.get(result_id, timeout=0.1)
    with pytest.raises(GetTimeoutError):
        ray.get(result_id, timeout=0)
    ray.get(signal.send.remote())
    start = time.time()
    ray.get(result_id, timeout=30)
    assert time.time() - start < 30

def test_call_actors_indirect_through_tasks(ray_start_regular_shared):
    if False:
        print('Hello World!')

    @ray.remote
    class Counter:

        def __init__(self, value):
            if False:
                print('Hello World!')
            self.value = int(value)

        def increase(self, delta):
            if False:
                while True:
                    i = 10
            self.value += int(delta)
            return self.value

    @ray.remote
    def foo(object):
        if False:
            for i in range(10):
                print('nop')
        return ray.get(object.increase.remote(1))

    @ray.remote
    def bar(object):
        if False:
            while True:
                i = 10
        return ray.get(object.increase.remote(1))

    @ray.remote
    def zoo(object):
        if False:
            while True:
                i = 10
        return ray.get(object[0].increase.remote(1))
    c = Counter.remote(0)
    for _ in range(0, 100):
        ray.get(foo.remote(c))
        ray.get(bar.remote(c))
        ray.get(zoo.remote([c]))

def test_inline_arg_memory_corruption(ray_start_regular_shared):
    if False:
        for i in range(10):
            print('nop')

    @ray.remote
    def f():
        if False:
            for i in range(10):
                print('nop')
        return bytes(1000)

    @ray.remote
    class Actor:

        def __init__(self):
            if False:
                print('Hello World!')
            self.z = []

        def add(self, x):
            if False:
                for i in range(10):
                    print('nop')
            self.z.append(x)
            for prev in self.z:
                assert sum(prev) == 0, ('memory corruption detected', prev)
    a = Actor.remote()
    for i in range(100):
        ray.get(a.add.remote(f.remote()))

@pytest.mark.skipif(client_test_enabled(), reason='internal api')
def test_skip_plasma(ray_start_regular_shared):
    if False:
        while True:
            i = 10

    @ray.remote
    class Actor:

        def __init__(self):
            if False:
                return 10
            pass

        def f(self, x):
            if False:
                return 10
            return x * 2
    a = Actor.remote()
    obj_ref = a.f.remote(1)
    assert not ray._private.worker.global_worker.core_worker.object_exists(obj_ref)
    assert ray.get(obj_ref) == 2

@pytest.mark.skipif(client_test_enabled(), reason='internal api')
def test_actor_large_objects(ray_start_regular_shared):
    if False:
        return 10

    @ray.remote
    class Actor:

        def __init__(self):
            if False:
                return 10
            pass

        def f(self):
            if False:
                return 10
            time.sleep(1)
            return bytes(80000000)
    a = Actor.remote()
    obj_ref = a.f.remote()
    assert not ray._private.worker.global_worker.core_worker.object_exists(obj_ref)
    (done, _) = ray.wait([obj_ref])
    assert len(done) == 1
    assert ray._private.worker.global_worker.core_worker.object_exists(obj_ref)
    assert isinstance(ray.get(obj_ref), bytes)

def test_actor_pass_by_ref(ray_start_regular_shared):
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

        def f(self, x):
            if False:
                return 10
            return x * 2

    @ray.remote
    def f(x):
        if False:
            i = 10
            return i + 15
        return x

    @ray.remote
    def error():
        if False:
            for i in range(10):
                print('nop')
        sys.exit(0)
    a = Actor.remote()
    assert ray.get(a.f.remote(f.remote(1))) == 2
    fut = [a.f.remote(f.remote(i)) for i in range(100)]
    assert ray.get(fut) == [i * 2 for i in range(100)]
    with pytest.raises(Exception):
        ray.get(a.f.remote(error.remote()))

def test_actor_recursive(ray_start_regular_shared):
    if False:
        return 10

    @ray.remote
    class Actor:

        def __init__(self, delegate=None):
            if False:
                while True:
                    i = 10
            self.delegate = delegate

        def f(self, x):
            if False:
                for i in range(10):
                    print('nop')
            if self.delegate:
                return ray.get(self.delegate.f.remote(x))
            return x * 2
    a = Actor.remote()
    b = Actor.remote(a)
    c = Actor.remote(b)
    result = ray.get([c.f.remote(i) for i in range(100)])
    assert result == [x * 2 for x in range(100)]
    (result, _) = ray.wait([c.f.remote(i) for i in range(100)], num_returns=100)
    result = ray.get(result)
    assert result == [x * 2 for x in range(100)]

def test_actor_concurrent(ray_start_regular_shared):
    if False:
        return 10

    @ray.remote
    class Batcher:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.batch = []
            self.event = threading.Event()

        def add(self, x):
            if False:
                print('Hello World!')
            self.batch.append(x)
            if len(self.batch) >= 3:
                self.event.set()
            else:
                self.event.wait()
            return sorted(self.batch)
    a = Batcher.options(max_concurrency=3).remote()
    x1 = a.add.remote(1)
    x2 = a.add.remote(2)
    x3 = a.add.remote(3)
    r1 = ray.get(x1)
    r2 = ray.get(x2)
    r3 = ray.get(x3)
    assert r1 == [1, 2, 3]
    assert r1 == r2 == r3

def test_actor_max_concurrency(ray_start_regular_shared):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that an actor of max_concurrency=N should only run\n    N tasks at most concurrently.\n    '
    CONCURRENCY = 3

    @ray.remote
    class ConcurentActor:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.threads = set()

        def call(self):
            if False:
                i = 10
                return i + 15
            self.threads.add(threading.current_thread())

        def get_num_threads(self):
            if False:
                while True:
                    i = 10
            return len(self.threads)

    @ray.remote
    def call(actor):
        if False:
            print('Hello World!')
        for _ in range(CONCURRENCY * 100):
            ray.get(actor.call.remote())
        return
    actor = ConcurentActor.options(max_concurrency=CONCURRENCY).remote()
    ray.get([call.remote(actor) for _ in range(CONCURRENCY * 10)])
    assert ray.get(actor.get_num_threads.remote()) <= CONCURRENCY

def test_wait(ray_start_regular_shared):
    if False:
        i = 10
        return i + 15

    @ray.remote
    def f(delay):
        if False:
            for i in range(10):
                print('nop')
        time.sleep(delay)
        return
    object_refs = [f.remote(0), f.remote(0), f.remote(0), f.remote(0)]
    (ready_ids, remaining_ids) = ray.wait(object_refs)
    assert len(ready_ids) == 1
    assert len(remaining_ids) == 3
    (ready_ids, remaining_ids) = ray.wait(object_refs, num_returns=4)
    assert set(ready_ids) == set(object_refs)
    assert remaining_ids == []
    object_refs = [f.remote(0), f.remote(5)]
    (ready_ids, remaining_ids) = ray.wait(object_refs, timeout=0.5, num_returns=2)
    assert len(ready_ids) == 1
    assert len(remaining_ids) == 1
    x = ray.put(1)
    with pytest.raises(Exception):
        ray.wait([x, x])
    (ready_ids, remaining_ids) = ray.wait([])
    assert ready_ids == []
    assert remaining_ids == []
    obj_refs = [ray.put(i) for i in range(10)]
    (found, rest) = ray.wait(obj_refs, num_returns=2)
    assert len(found) == 2
    assert len(rest) == 8
    x = ray.put(1)
    with pytest.raises(TypeError):
        ray.wait(x)
    with pytest.raises(TypeError):
        ray.wait(1)
    with pytest.raises(TypeError):
        ray.wait([1])

def test_duplicate_args(ray_start_regular_shared):
    if False:
        i = 10
        return i + 15

    @ray.remote
    def f(arg1, arg2, arg1_duplicate, kwarg1=None, kwarg2=None, kwarg1_duplicate=None):
        if False:
            while True:
                i = 10
        assert arg1 == kwarg1
        assert arg1 != arg2
        assert arg1 == arg1_duplicate
        assert kwarg1 != kwarg2
        assert kwarg1 == kwarg1_duplicate
    arg1 = [1]
    arg2 = [2]
    ray.get(f.remote(arg1, arg2, arg1, kwarg1=arg1, kwarg2=arg2, kwarg1_duplicate=arg1))
    arg1 = ray.put([1])
    arg2 = ray.put([2])
    ray.get(f.remote(arg1, arg2, arg1, kwarg1=arg1, kwarg2=arg2, kwarg1_duplicate=arg1))

    @ray.remote
    class Actor:

        def f(self, arg1, arg2, arg1_duplicate, kwarg1=None, kwarg2=None, kwarg1_duplicate=None):
            if False:
                while True:
                    i = 10
            assert arg1 == kwarg1
            assert arg1 != arg2
            assert arg1 == arg1_duplicate
            assert kwarg1 != kwarg2
            assert kwarg1 == kwarg1_duplicate
    actor = Actor.remote()
    ray.get(actor.f.remote(arg1, arg2, arg1, kwarg1=arg1, kwarg2=arg2, kwarg1_duplicate=arg1))

@pytest.mark.skipif(client_test_enabled(), reason='internal api')
def test_get_correct_node_ip():
    if False:
        for i in range(10):
            print('nop')
    with patch('ray._private.worker') as worker_mock:
        node_mock = MagicMock()
        node_mock.node_ip_address = '10.0.0.111'
        worker_mock._global_node = node_mock
        found_ip = ray.util.get_node_ip_address()
        assert found_ip == '10.0.0.111'

def test_load_code_from_local(ray_start_regular_shared):
    if False:
        for i in range(10):
            print('nop')
    code_test = '\nimport os\nimport ray\n\nclass A:\n    @ray.remote\n    class B:\n        def get(self):\n            return "OK"\n\nif __name__ == "__main__":\n    current_path = os.path.dirname(__file__)\n    job_config = ray.job_config.JobConfig(code_search_path=[current_path])\n    ray.init({}, job_config=job_config)\n    b = A.B.remote()\n    print(ray.get(b.get.remote()))\n'
    with tempfile.TemporaryDirectory(suffix='a b') as tmpdir:
        test_driver = os.path.join(tmpdir, 'test_load_code_from_local.py')
        with open(test_driver, 'w') as f:
            f.write(code_test.format(repr(ray_start_regular_shared['address'])))
        output = subprocess.check_output([sys.executable, test_driver])
        assert b'OK' in output

@pytest.mark.skipif(client_test_enabled(), reason="JobConfig doesn't work in client mode")
def test_use_dynamic_function_and_class():
    if False:
        while True:
            i = 10
    ray.shutdown()
    current_path = os.path.dirname(__file__)
    job_config = ray.job_config.JobConfig(code_search_path=[current_path])
    ray.init(job_config=job_config)

    def foo1():
        if False:
            print('Hello World!')

        @ray.remote
        def foo2():
            if False:
                for i in range(10):
                    print('nop')
            return 'OK'
        return foo2

    @ray.remote
    class Foo:

        @ray.method(num_returns=1)
        def foo(self):
            if False:
                return 10
            return 'OK'
    f = foo1()
    assert ray.get(f.remote()) == 'OK'
    key_func = b'RemoteFunction:' + ray._private.worker.global_worker.current_job_id.hex().encode() + b':' + f._function_descriptor.function_id.binary()
    assert ray._private.worker.global_worker.gcs_client.internal_kv_exists(key_func, KV_NAMESPACE_FUNCTION_TABLE)
    foo_actor = Foo.remote()
    assert ray.get(foo_actor.foo.remote()) == 'OK'
    key_cls = b'ActorClass:' + ray._private.worker.global_worker.current_job_id.hex().encode() + b':' + foo_actor._ray_actor_creation_function_descriptor.function_id.binary()
    assert ray._private.worker.global_worker.gcs_client.internal_kv_exists(key_cls, namespace=KV_NAMESPACE_FUNCTION_TABLE)
if __name__ == '__main__':
    import pytest
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))