import atexit
import asyncio
import collections
import numpy as np
import os
import pytest
import signal
import sys
import time
import ray
from ray.actor import exit_actor
import ray.cluster_utils
from ray._private.test_utils import wait_for_condition, wait_for_pid_to_exit, generate_system_config_map, SignalActor
SIGKILL = signal.SIGKILL if sys.platform != 'win32' else signal.SIGTERM

@pytest.fixture
def ray_init_with_task_retry_delay():
    if False:
        i = 10
        return i + 15
    address = ray.init(_system_config={'task_retry_delay_ms': 100})
    yield address
    ray.shutdown()

@pytest.mark.parametrize('ray_start_regular', [{'object_store_memory': 150 * 1024 * 1024}], indirect=True)
@pytest.mark.skipif(sys.platform == 'win32', reason='Segfaults on CI')
def test_actor_spilled(ray_start_regular):
    if False:
        print('Hello World!')
    object_store_memory = 150 * 1024 * 1024

    @ray.remote
    class Actor:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            pass

        def create_object(self, size):
            if False:
                i = 10
                return i + 15
            return np.random.rand(size)
    a = Actor.remote()
    objects = []
    num_objects = 40
    for _ in range(num_objects):
        obj = a.create_object.remote(object_store_memory // num_objects)
        objects.append(obj)
        ray.get(obj)
    num_success = 0
    for obj in objects:
        val = ray.get(obj)
        assert isinstance(val, np.ndarray), val
        num_success += 1
    assert num_success == len(objects)

def test_actor_restart(ray_init_with_task_retry_delay):
    if False:
        print('Hello World!')
    'Test actor restart when actor process is killed.'

    @ray.remote(max_restarts=1)
    class RestartableActor:
        """An actor that will be restarted at most once."""

        def __init__(self):
            if False:
                print('Hello World!')
            self.value = 0

        def increase(self, exit=False):
            if False:
                while True:
                    i = 10
            if exit:
                os._exit(-1)
            self.value += 1
            return self.value

        def get_pid(self):
            if False:
                for i in range(10):
                    print('nop')
            return os.getpid()
    actor = RestartableActor.remote()
    results = [actor.increase.remote(exit=i == 100) for i in range(200)]
    i = 1
    while results:
        res = results[0]
        try:
            r = ray.get(res)
            if r != i:
                break
            results.pop(0)
            i += 1
        except ray.exceptions.RayActorError:
            break
    while results:
        try:
            ray.get(results[0])
        except ray.exceptions.RayActorError:
            results.pop(0)
        else:
            break
    if results:
        i = 1
        while results:
            r = ray.get(results.pop(0))
            assert r == i
            i += 1
        result = actor.increase.remote()
        assert ray.get(result) == r + 1
    else:

        def ping():
            if False:
                print('Hello World!')
            try:
                ray.get(actor.increase.remote())
                return True
            except ray.exceptions.RayActorError:
                return False
        wait_for_condition(ping)
    actor.increase.remote(exit=True)
    for _ in range(100):
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(actor.increase.remote())
    actor = RestartableActor.remote()
    actor.__ray_terminate__.remote()
    with pytest.raises(ray.exceptions.RayActorError):
        ray.get(actor.increase.remote())

def test_actor_restart_with_retry(ray_init_with_task_retry_delay):
    if False:
        while True:
            i = 10
    'Test actor restart when actor process is killed.'

    @ray.remote(max_restarts=1, max_task_retries=-1)
    class RestartableActor:
        """An actor that will be restarted at most once."""

        def __init__(self):
            if False:
                return 10
            self.value = 0

        def increase(self, delay=0):
            if False:
                return 10
            time.sleep(delay)
            self.value += 1
            return self.value

        def get_pid(self):
            if False:
                for i in range(10):
                    print('nop')
            return os.getpid()
    actor = RestartableActor.remote()
    pid = ray.get(actor.get_pid.remote())
    results = [actor.increase.remote() for _ in range(100)]
    os.kill(pid, SIGKILL)
    wait_for_pid_to_exit(pid)
    seq = list(range(1, 101))
    results = ray.get(results)
    failed_task_index = None
    for (i, res) in enumerate(results):
        if res != seq[0]:
            if failed_task_index is None:
                failed_task_index = i
            assert res + failed_task_index == seq[0]
        seq.pop(0)
    result = actor.increase.remote()
    assert ray.get(result) == results[-1] + 1
    results = [actor.increase.remote() for _ in range(100)]
    pid = ray.get(actor.get_pid.remote())
    os.kill(pid, SIGKILL)
    wait_for_pid_to_exit(pid)
    with pytest.raises(ray.exceptions.RayActorError):
        ray.get(actor.increase.remote())
    actor = RestartableActor.remote()
    actor.__ray_terminate__.remote()
    with pytest.raises(ray.exceptions.RayActorError):
        ray.get(actor.increase.remote())

def test_named_actor_max_task_retries(ray_init_with_task_retry_delay):
    if False:
        print('Hello World!')

    @ray.remote(num_cpus=0)
    class Counter:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.count = 0
            self.event = asyncio.Event()

        def increment(self):
            if False:
                print('Hello World!')
            self.count += 1
            self.event.set()

        async def wait_for_count(self, count):
            while True:
                if self.count >= count:
                    return
                await self.event.wait()
                self.event.clear()

    @ray.remote
    class ActorToKill:

        def __init__(self, counter):
            if False:
                return 10
            counter.increment.remote()

        def run(self, counter, signal):
            if False:
                print('Hello World!')
            counter.increment.remote()
            ray.get(signal.wait.remote())

    @ray.remote
    class CallingActor:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.actor = ray.get_actor('a')

        def call_other(self, counter, signal):
            if False:
                print('Hello World!')
            return ray.get(self.actor.run.remote(counter, signal))
    init_counter = Counter.remote()
    run_counter = Counter.remote()
    signal = SignalActor.remote()
    a = ActorToKill.options(name='a', max_restarts=-1, max_task_retries=-1).remote(init_counter)
    c = CallingActor.remote()
    ray.get(init_counter.wait_for_count.remote(1), timeout=30)
    ref = c.call_other.remote(run_counter, signal)
    ray.get(run_counter.wait_for_count.remote(1), timeout=30)
    ray.kill(a, no_restart=False)
    ray.get(init_counter.wait_for_count.remote(2), timeout=30)
    ray.get(run_counter.wait_for_count.remote(2), timeout=30)
    signal.send.remote()
    ray.get(ref, timeout=30)

def test_actor_restart_on_node_failure(ray_start_cluster):
    if False:
        return 10
    config = {'health_check_failure_threshold': 10, 'health_check_period_ms': 100, 'health_check_initial_delay_ms': 0, 'object_timeout_milliseconds': 1000, 'task_retry_delay_ms': 100}
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=0, _system_config=config)
    cluster.wait_for_nodes()
    ray.init(address=cluster.address)
    actor_node = cluster.add_node(num_cpus=1)
    cluster.wait_for_nodes()

    @ray.remote(num_cpus=1, max_restarts=1, max_task_retries=-1)
    class RestartableActor:
        """An actor that will be reconstructed at most once."""

        def __init__(self):
            if False:
                return 10
            self.value = 0

        def increase(self):
            if False:
                for i in range(10):
                    print('nop')
            self.value += 1
            return self.value

        def ready(self):
            if False:
                i = 10
                return i + 15
            return
    actor = RestartableActor.options(lifetime='detached').remote()
    ray.get(actor.ready.remote())
    results = [actor.increase.remote() for _ in range(100)]
    cluster.remove_node(actor_node)
    cluster.add_node(num_cpus=1)
    cluster.wait_for_nodes()
    seq = list(range(1, 101))
    results = ray.get(results)
    failed_task_index = None
    for (i, res) in enumerate(results):
        elm = seq.pop(0)
        if res != elm:
            if failed_task_index is None:
                failed_task_index = i
            assert res + failed_task_index == elm
    result = ray.get(actor.increase.remote())
    assert result == 1 or result == results[-1] + 1

def test_caller_actor_restart(ray_start_regular):
    if False:
        i = 10
        return i + 15
    'Test tasks from a restarted actor can be correctly processed\n    by the receiving actor.'

    @ray.remote(max_restarts=1, max_task_retries=-1)
    class RestartableActor:
        """An actor that will be restarted at most once."""

        def __init__(self, actor):
            if False:
                print('Hello World!')
            self.actor = actor

        def increase(self):
            if False:
                print('Hello World!')
            return ray.get(self.actor.increase.remote())

        def get_pid(self):
            if False:
                return 10
            return os.getpid()

    @ray.remote(max_restarts=1)
    class Actor:
        """An actor that will be restarted at most once."""

        def __init__(self):
            if False:
                print('Hello World!')
            self.value = 0

        def increase(self):
            if False:
                while True:
                    i = 10
            self.value += 1
            return self.value
    remote_actor = Actor.remote()
    actor = RestartableActor.remote(remote_actor)
    for _ in range(3):
        ray.get(actor.increase.remote())
    kill_actor(actor)
    assert ray.get(actor.increase.remote()) == 4

def test_caller_task_reconstruction(ray_start_regular):
    if False:
        for i in range(10):
            print('nop')
    'Test a retried task from a dead worker can be correctly processed\n    by the receiving actor.'

    @ray.remote(max_retries=5)
    def RetryableTask(actor):
        if False:
            return 10
        value = ray.get(actor.increase.remote())
        if value > 2:
            return value
        else:
            os._exit(0)

    @ray.remote(max_restarts=1)
    class Actor:
        """An actor that will be restarted at most once."""

        def __init__(self):
            if False:
                print('Hello World!')
            self.value = 0

        def increase(self):
            if False:
                while True:
                    i = 10
            self.value += 1
            return self.value
    remote_actor = Actor.remote()
    assert ray.get(RetryableTask.remote(remote_actor)) == 3

@pytest.mark.skipif(sys.platform == 'win32', reason='Very flaky on Windows.')
@pytest.mark.parametrize('ray_start_cluster_head', [generate_system_config_map(object_timeout_milliseconds=1000, health_check_initial_delay_ms=0, health_check_period_ms=1000, health_check_failure_threshold=10)], indirect=True)
def test_multiple_actor_restart(ray_start_cluster_head):
    if False:
        print('Hello World!')
    cluster = ray_start_cluster_head
    num_nodes = 5
    num_actors_at_a_time = 3
    num_function_calls_at_a_time = 10
    worker_nodes = [cluster.add_node(num_cpus=3) for _ in range(num_nodes)]

    @ray.remote(max_restarts=-1, max_task_retries=-1)
    class SlowCounter:

        def __init__(self):
            if False:
                return 10
            self.x = 0

        def inc(self, duration):
            if False:
                return 10
            time.sleep(duration)
            self.x += 1
            return self.x
    actors = [SlowCounter.remote() for _ in range(num_actors_at_a_time)]
    time.sleep(1)
    result_ids = collections.defaultdict(lambda : [])
    for node in worker_nodes:
        actors.extend([SlowCounter.remote() for _ in range(num_actors_at_a_time)])
        for j in range(len(actors)):
            actor = actors[j]
            for _ in range(num_function_calls_at_a_time):
                result_ids[actor].append(actor.inc.remote(j ** 2 * 1e-06))
        cluster.remove_node(node)
        for j in range(len(actors)):
            actor = actors[j]
            for _ in range(num_function_calls_at_a_time):
                result_ids[actor].append(actor.inc.remote(j ** 2 * 1e-06))
    for (_, result_id_list) in result_ids.items():
        results = ray.get(result_id_list)
        for (i, result) in enumerate(results):
            if i == 0:
                assert result == 1
            else:
                assert result == results[i - 1] + 1 or result == 1

def kill_actor(actor):
    if False:
        while True:
            i = 10
    'A helper function that kills an actor process.'
    pid = ray.get(actor.get_pid.remote())
    os.kill(pid, SIGKILL)
    wait_for_pid_to_exit(pid)

def test_decorated_method(ray_start_regular):
    if False:
        while True:
            i = 10

    def method_invocation_decorator(f):
        if False:
            for i in range(10):
                print('nop')

        def new_f_invocation(args, kwargs):
            if False:
                while True:
                    i = 10
            return (f([args[0], args[0]], {}), kwargs)
        return new_f_invocation

    def method_execution_decorator(f):
        if False:
            for i in range(10):
                print('nop')

        def new_f_execution(self, b, c):
            if False:
                while True:
                    i = 10
            return f(self, b + c)
        new_f_execution.__ray_invocation_decorator__ = method_invocation_decorator
        return new_f_execution

    @ray.remote
    class Actor:

        @method_execution_decorator
        def decorated_method(self, x):
            if False:
                i = 10
                return i + 15
            return x + 1
    a = Actor.remote()
    (object_ref, extra) = a.decorated_method.remote(3, kwarg=3)
    assert isinstance(object_ref, ray.ObjectRef)
    assert extra == {'kwarg': 3}
    assert ray.get(object_ref) == 7

@pytest.mark.parametrize('ray_start_cluster', [{'num_cpus': 1, 'num_nodes': 1}], indirect=True)
def test_actor_owner_worker_dies_before_dependency_ready(ray_start_cluster):
    if False:
        print('Hello World!')
    'Test actor owner worker dies before local dependencies are resolved.\n    This test verifies the scenario where owner worker\n    has failed before actor dependencies are resolved.\n    Reference: https://github.com/ray-project/ray/pull/8045\n    '

    @ray.remote
    class Actor:

        def __init__(self, dependency):
            if False:
                return 10
            print('actor: {}'.format(os.getpid()))
            self.dependency = dependency

        def f(self):
            if False:
                while True:
                    i = 10
            return self.dependency

    @ray.remote
    class Owner:

        def get_pid(self):
            if False:
                for i in range(10):
                    print('nop')
            return os.getpid()

        def create_actor(self, caller_handle):
            if False:
                for i in range(10):
                    print('nop')
            s = SignalActor.remote()
            actor_handle = Actor.remote(s.wait.remote())
            pid = os.getpid()
            signal_handle = SignalActor.remote()
            caller_handle.call.remote(pid, signal_handle, actor_handle)
            ray.get(signal_handle.wait.remote())
            os._exit(0)

    @ray.remote
    class Caller:

        def call(self, owner_pid, signal_handle, actor_handle):
            if False:
                i = 10
                return i + 15
            ray.get(signal_handle.send.remote())
            wait_for_pid_to_exit(owner_pid)
            oid = actor_handle.f.remote()
            ray.get(oid)

        def hang(self):
            if False:
                print('Hello World!')
            return True
    owner = Owner.remote()
    owner_pid = ray.get(owner.get_pid.remote())
    caller = Caller.remote()
    owner.create_actor.remote(caller)
    wait_for_pid_to_exit(owner_pid)
    wait_for_condition(lambda : ray.get(caller.hang.remote()))

@pytest.mark.parametrize('ray_start_cluster', [{'num_cpus': 3, 'num_nodes': 1}], indirect=True)
def test_actor_owner_node_dies_before_dependency_ready(ray_start_cluster):
    if False:
        i = 10
        return i + 15
    'Test actor owner node dies before local dependencies are resolved.\n    This test verifies the scenario where owner node\n    has failed before actor dependencies are resolved.\n    Reference: https://github.com/ray-project/ray/pull/8045\n    '

    @ray.remote
    class Actor:

        def __init__(self, dependency):
            if False:
                for i in range(10):
                    print('nop')
            print('actor: {}'.format(os.getpid()))
            self.dependency = dependency

        def f(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.dependency

    @ray.remote(resources={'node': 1})
    class Owner:

        def get_pid(self):
            if False:
                i = 10
                return i + 15
            return os.getpid()

        def create_actor(self, caller_handle):
            if False:
                i = 10
                return i + 15
            s = SignalActor.remote()
            actor_handle = Actor.remote(s.wait.remote())
            pid = os.getpid()
            signal_handle = SignalActor.remote()
            caller_handle.call.remote(pid, signal_handle, actor_handle)
            ray.get(signal_handle.wait.remote())

    @ray.remote(resources={'caller': 1})
    class Caller:

        def call(self, owner_pid, signal_handle, actor_handle):
            if False:
                return 10
            ray.get(signal_handle.send.remote())
            wait_for_pid_to_exit(owner_pid)
            oid = actor_handle.f.remote()
            ray.get(oid)

        def hang(self):
            if False:
                return 10
            return True
    cluster = ray_start_cluster
    node_to_be_broken = cluster.add_node(resources={'node': 1})
    cluster.add_node(resources={'caller': 1})
    owner = Owner.remote()
    owner_pid = ray.get(owner.get_pid.remote())
    caller = Caller.remote()
    ray.get(owner.create_actor.remote(caller))
    cluster.remove_node(node_to_be_broken)
    wait_for_pid_to_exit(owner_pid)
    wait_for_condition(lambda : ray.get(caller.hang.remote()))

def test_recreate_child_actor(ray_start_cluster):
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

        def ready(self):
            if False:
                while True:
                    i = 10
            return

    @ray.remote(max_restarts=-1, max_task_retries=-1)
    class Parent:

        def __init__(self):
            if False:
                return 10
            self.child = Actor.remote()

        def ready(self):
            if False:
                print('Hello World!')
            return ray.get(self.child.ready.remote())

        def pid(self):
            if False:
                while True:
                    i = 10
            return os.getpid()
    ray.init(address=ray_start_cluster.address)
    p = Parent.remote()
    pid = ray.get(p.pid.remote())
    os.kill(pid, 9)
    ray.get(p.ready.remote())

def test_actor_failure_per_type(ray_start_cluster):
    if False:
        print('Hello World!')
    cluster = ray_start_cluster
    cluster.add_node()
    ray.init(address='auto')

    @ray.remote
    class Actor:

        def check_alive(self):
            if False:
                i = 10
                return i + 15
            return os.getpid()

        def create_actor(self):
            if False:
                for i in range(10):
                    print('nop')
            self.a = Actor.remote()
            return self.a
    with pytest.raises(RuntimeError, match='Lost reference to actor') as exc_info:
        ray.get(Actor.remote().check_alive.remote())
    print(exc_info._excinfo[1])
    a = Actor.remote()
    ray.kill(a)
    with pytest.raises(ray.exceptions.RayActorError, match='it was killed by `ray.kill') as exc_info:
        ray.get(a.check_alive.remote())
    assert exc_info.value.actor_id == a._actor_id.hex()
    print(exc_info._excinfo[1])
    a = Actor.remote()
    pid = ray.get(a.check_alive.remote())
    os.kill(pid, 9)
    with pytest.raises(ray.exceptions.RayActorError, match='The actor is dead because its worker process has died') as exc_info:
        ray.get(a.check_alive.remote())
    assert exc_info.value.actor_id == a._actor_id.hex()
    print(exc_info._excinfo[1])
    owner = Actor.remote()
    a = ray.get(owner.create_actor.remote())
    ray.kill(owner)
    with pytest.raises(ray.exceptions.RayActorError, match='The actor is dead because its owner has died') as exc_info:
        ray.get(a.check_alive.remote())
    assert exc_info.value.actor_id == a._actor_id.hex()
    print(exc_info._excinfo[1])
    node_to_kill = cluster.add_node(resources={'worker': 1})
    a = Actor.options(resources={'worker': 1}).remote()
    ray.get(a.check_alive.remote())
    cluster.remove_node(node_to_kill)
    with pytest.raises(ray.exceptions.RayActorError, match='The actor is dead because its node has died.') as exc_info:
        ray.get(a.check_alive.remote())
    assert exc_info.value.actor_id == a._actor_id.hex()
    print(exc_info._excinfo[1])

def test_utf8_actor_exception(ray_start_regular):
    if False:
        i = 10
        return i + 15

    @ray.remote
    class FlakyActor:

        def __init__(self):
            if False:
                print('Hello World!')
            raise RuntimeError('你好呀，祝你有个好心情！')

        def ping(self):
            if False:
                print('Hello World!')
            return True
    actor = FlakyActor.remote()
    with pytest.raises(ray.exceptions.RayActorError):
        ray.get(actor.ping.remote())

def test_failure_during_dependency_resolution(ray_start_regular):
    if False:
        for i in range(10):
            print('nop')

    @ray.remote
    class Actor:

        def dep(self):
            if False:
                while True:
                    i = 10
            while True:
                time.sleep(1)

        def foo(self, x):
            if False:
                for i in range(10):
                    print('nop')
            return x

    @ray.remote
    def foo():
        if False:
            for i in range(10):
                print('nop')
        time.sleep(3)
        return 1
    a = Actor.remote()
    ray.get(a.foo.remote(1))
    ray.kill(a, no_restart=False)
    dep = a.dep.remote()
    ref = a.foo.remote(dep)
    with pytest.raises(ray.exceptions.RayActorError):
        ray.get(ref)

def test_exit_actor(shutdown_only, tmp_path):
    if False:
        while True:
            i = 10
    '\n    Verify TypeError is raised when exit_actor is not used\n    inside an actor.\n    '
    with pytest.raises(TypeError, match='exit_actor API is called on a non-actor worker'):
        exit_actor()

    @ray.remote
    def f():
        if False:
            return 10
        exit_actor()
    with pytest.raises(TypeError, match='exit_actor API is called on a non-actor worker'):
        ray.get(f.remote())
    '\n    Verify the basic case.\n    '

    @ray.remote
    class Actor:

        def exit(self):
            if False:
                i = 10
                return i + 15
            exit_actor()

    @ray.remote
    class AsyncActor:

        async def exit(self):
            exit_actor()
    a = Actor.remote()
    ray.get(a.__ray_ready__.remote())
    with pytest.raises(ray.exceptions.RayActorError) as exc_info:
        ray.get(a.exit.remote())
    assert 'exit_actor()' in str(exc_info.value)
    b = AsyncActor.remote()
    ray.get(b.__ray_ready__.remote())
    with pytest.raises(ray.exceptions.RayActorError) as exc_info:
        ray.get(b.exit.remote())
    assert 'exit_actor()' in str(exc_info.value)
    '\n    Verify atexit handler is called correctly.\n    '
    sync_temp_file = tmp_path / 'actor.log'
    async_temp_file = tmp_path / 'async_actor.log'
    sync_temp_file.touch()
    async_temp_file.touch()

    @ray.remote
    class Actor:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')

            def f():
                if False:
                    i = 10
                    return i + 15
                print('atexit handler')
                with open(sync_temp_file, 'w') as f:
                    f.write('Actor\n')
            atexit.register(f)

        def exit(self):
            if False:
                while True:
                    i = 10
            exit_actor()

    @ray.remote
    class AsyncActor:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')

            def f():
                if False:
                    print('Hello World!')
                print('atexit handler')
                with open(async_temp_file, 'w') as f:
                    f.write('Async Actor\n')
            atexit.register(f)

        async def exit(self):
            exit_actor()
    a = Actor.remote()
    ray.get(a.__ray_ready__.remote())
    b = AsyncActor.remote()
    ray.get(b.__ray_ready__.remote())
    with pytest.raises(ray.exceptions.RayActorError):
        ray.get(a.exit.remote())
    with pytest.raises(ray.exceptions.RayActorError):
        ray.get(b.exit.remote())

    def verify():
        if False:
            i = 10
            return i + 15
        with open(async_temp_file) as f:
            assert f.readlines() == ['Async Actor\n']
        with open(sync_temp_file) as f:
            assert f.readlines() == ['Actor\n']
        return True
    wait_for_condition(verify)

def test_exit_actor_queued(shutdown_only):
    if False:
        return 10
    "Verify after exit_actor is called the queued tasks won't execute."

    @ray.remote
    class RegressionSync:

        def f(self):
            if False:
                while True:
                    i = 10
            import time
            time.sleep(1)
            exit_actor()

        def ping(self):
            if False:
                while True:
                    i = 10
            pass

    @ray.remote
    class RegressionAsync:

        async def f(self):
            await asyncio.sleep(1)
            exit_actor()

        def ping(self):
            if False:
                return 10
            pass
    a = RegressionAsync.remote()
    a.f.remote()
    refs = [a.ping.remote() for _ in range(10000)]
    with pytest.raises(ray.exceptions.RayActorError) as exc_info:
        ray.get(refs)
    assert ' Worker unexpectedly exits' not in str(exc_info.value)
    a = RegressionSync.remote()
    a.f.remote()
    with pytest.raises(ray.exceptions.RayActorError) as exc_info:
        ray.get([a.ping.remote() for _ in range(10000)])
    assert ' Worker unexpectedly exits' not in str(exc_info.value)
if __name__ == '__main__':
    import pytest
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))