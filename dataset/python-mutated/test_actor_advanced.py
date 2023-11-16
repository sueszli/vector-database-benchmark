import os
import sys
import time
import numpy as np
import pytest
import ray
import ray._private.gcs_utils as gcs_utils
from ray.util.state import list_actors
import ray.cluster_utils
from ray._private.test_utils import SignalActor, convert_actor_state, kill_actor_and_wait_for_failure, make_global_state_accessor, run_string_as_driver, wait_for_condition, wait_for_pid_to_exit
from ray._private.ray_constants import gcs_actor_scheduling_enabled
from ray.experimental.internal_kv import _internal_kv_get, _internal_kv_put
try:
    import pytest_timeout
except ImportError:
    pytest_timeout = None

def test_remote_functions_not_scheduled_on_actors(ray_start_regular):
    if False:
        print('Hello World!')

    @ray.remote
    class Actor:

        def __init__(self):
            if False:
                print('Hello World!')
            pass

        def get_id(self):
            if False:
                for i in range(10):
                    print('nop')
            return ray.get_runtime_context().get_worker_id()
    a = Actor.remote()
    actor_id = ray.get(a.get_id.remote())

    @ray.remote
    def f():
        if False:
            print('Hello World!')
        return ray.get_runtime_context().get_worker_id()
    resulting_ids = ray.get([f.remote() for _ in range(100)])
    assert actor_id not in resulting_ids

def test_actors_on_nodes_with_no_cpus(ray_start_no_cpu):
    if False:
        print('Hello World!')

    @ray.remote
    class Foo:

        def method(self):
            if False:
                while True:
                    i = 10
            pass
    f = Foo.remote()
    (ready_ids, _) = ray.wait([f.method.remote()], timeout=0.1)
    assert ready_ids == []

@pytest.mark.skipif(gcs_actor_scheduling_enabled(), reason='This test relies on gcs server randomly choosing raylets ' + 'for actors without required resources, which is only supported by ' + 'raylet-based actor scheduler. The same test logic for gcs-based ' + 'actor scheduler can be found at `test_actor_distribution_balance`.')
def test_actor_load_balancing(ray_start_cluster):
    if False:
        i = 10
        return i + 15
    cluster = ray_start_cluster
    num_nodes = 3
    for i in range(num_nodes):
        cluster.add_node(num_cpus=1)
    ray.init(address=cluster.address)

    @ray.remote
    class Actor1:

        def __init__(self):
            if False:
                print('Hello World!')
            pass

        def get_location(self):
            if False:
                i = 10
                return i + 15
            return ray._private.worker.global_worker.node.unique_id
    num_actors = 30
    num_attempts = 20
    minimum_count = 5
    attempts = 0
    while attempts < num_attempts:
        actors = [Actor1.remote() for _ in range(num_actors)]
        locations = ray.get([actor.get_location.remote() for actor in actors])
        names = set(locations)
        counts = [locations.count(name) for name in names]
        print('Counts are {}.'.format(counts))
        if len(names) == num_nodes and all((count >= minimum_count for count in counts)):
            break
        attempts += 1
    assert attempts < num_attempts
    results = []
    for _ in range(1000):
        index = np.random.randint(num_actors)
        results.append(actors[index].get_location.remote())
    ray.get(results)

def test_actor_lifetime_load_balancing(ray_start_cluster):
    if False:
        while True:
            i = 10
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=0)
    num_nodes = 3
    for i in range(num_nodes):
        cluster.add_node(num_cpus=1)
    ray.init(address=cluster.address)

    @ray.remote(num_cpus=1)
    class Actor:

        def __init__(self):
            if False:
                return 10
            pass

        def ping(self):
            if False:
                return 10
            return
    actors = [Actor.remote() for _ in range(num_nodes)]
    ray.get([actor.ping.remote() for actor in actors])

@pytest.mark.parametrize('ray_start_regular', [{'resources': {'actor': 1}, 'num_cpus': 2}], indirect=True)
def test_deleted_actor_no_restart(ray_start_regular):
    if False:
        for i in range(10):
            print('nop')

    @ray.remote(resources={'actor': 1}, max_restarts=3)
    class Actor:

        def method(self):
            if False:
                return 10
            return 1

        def getpid(self):
            if False:
                while True:
                    i = 10
            return os.getpid()

    @ray.remote
    def f(actor, signal):
        if False:
            i = 10
            return i + 15
        ray.get(signal.wait.remote())
        return ray.get(actor.method.remote())
    signal = SignalActor.remote()
    a = Actor.remote()
    pid = ray.get(a.getpid.remote())
    x_id = f.remote(a, signal)
    del a
    ray.get(signal.send.remote())
    assert ray.get(x_id) == 1
    wait_for_pid_to_exit(pid)
    a = Actor.remote()
    pid = ray.get(a.getpid.remote())

def test_exception_raised_when_actor_node_dies(ray_start_cluster_head):
    if False:
        for i in range(10):
            print('nop')
    cluster = ray_start_cluster_head
    remote_node = cluster.add_node()

    @ray.remote(max_restarts=0, scheduling_strategy='SPREAD')
    class Counter:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.x = 0

        def node_id(self):
            if False:
                for i in range(10):
                    print('nop')
            return ray._private.worker.global_worker.node.unique_id

        def inc(self):
            if False:
                while True:
                    i = 10
            self.x += 1
            return self.x
    actor = Counter.remote()
    while ray.get(actor.node_id.remote()) != remote_node.unique_id:
        actor = Counter.remote()
    cluster.remove_node(remote_node)
    for _ in range(10):
        x_ids = [actor.inc.remote() for _ in range(5)]
        for x_id in x_ids:
            with pytest.raises(ray.exceptions.RayActorError):
                ray.get(x_id)

def test_actor_init_fails(ray_start_cluster_head):
    if False:
        while True:
            i = 10
    cluster = ray_start_cluster_head
    remote_node = cluster.add_node()

    @ray.remote(max_restarts=1, max_task_retries=-1)
    class Counter:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.x = 0

        def inc(self):
            if False:
                while True:
                    i = 10
            self.x += 1
            return self.x
    actors = [Counter.remote() for _ in range(15)]
    time.sleep(0.1)
    cluster.remove_node(remote_node)
    results = ray.get([actor.inc.remote() for actor in actors])
    assert results == [1 for actor in actors]

def test_reconstruction_suppression(ray_start_cluster_head):
    if False:
        for i in range(10):
            print('nop')
    cluster = ray_start_cluster_head
    num_nodes = 5
    worker_nodes = [cluster.add_node() for _ in range(num_nodes)]

    @ray.remote(max_restarts=1)
    class Counter:

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.x = 0

        def inc(self):
            if False:
                print('Hello World!')
            self.x += 1
            return self.x

    @ray.remote
    def inc(actor_handle):
        if False:
            for i in range(10):
                print('nop')
        return ray.get(actor_handle.inc.remote())
    actors = [Counter.remote() for _ in range(10)]
    ray.get([actor.inc.remote() for actor in actors])
    cluster.remove_node(worker_nodes[0])
    results = []
    for _ in range(10):
        results += [inc.remote(actor) for actor in actors]
    results = ray.get(results)

@pytest.fixture
def setup_queue_actor():
    if False:
        while True:
            i = 10
    ray.init(num_cpus=1, object_store_memory=int(150 * 1024 * 1024))

    @ray.remote
    class Queue:

        def __init__(self):
            if False:
                return 10
            self.queue = []

        def enqueue(self, key, item):
            if False:
                i = 10
                return i + 15
            self.queue.append((key, item))

        def read(self):
            if False:
                return 10
            return self.queue
    queue = Queue.remote()
    ray.get(queue.read.remote())
    yield queue
    ray.shutdown()

def test_fork(setup_queue_actor):
    if False:
        print('Hello World!')
    queue = setup_queue_actor

    @ray.remote
    def fork(queue, key, item):
        if False:
            i = 10
            return i + 15
        return ray.get(queue.enqueue.remote(key, item))
    num_iters = 100
    ray.get([fork.remote(queue, i, 0) for i in range(num_iters)])
    items = ray.get(queue.read.remote())
    for i in range(num_iters):
        filtered_items = [item[1] for item in items if item[0] == i]
        assert filtered_items == list(range(1))

def test_fork_consistency(setup_queue_actor):
    if False:
        i = 10
        return i + 15
    queue = setup_queue_actor

    @ray.remote
    def fork(queue, key, num_items):
        if False:
            print('Hello World!')
        x = None
        for item in range(num_items):
            x = queue.enqueue.remote(key, item)
        return ray.get(x)
    num_forks = 5
    num_items_per_fork = 100
    forks = [fork.remote(queue, i, num_items_per_fork) for i in range(num_forks)]
    for item in range(num_items_per_fork):
        local_fork = queue.enqueue.remote(num_forks, item)
    forks.append(local_fork)
    ray.get(forks)
    items = ray.get(queue.read.remote())
    for i in range(num_forks + 1):
        filtered_items = [item[1] for item in items if item[0] == i]
        assert filtered_items == list(range(num_items_per_fork))

def test_pickled_handle_consistency(setup_queue_actor):
    if False:
        for i in range(10):
            print('nop')
    queue = setup_queue_actor

    @ray.remote
    def fork(pickled_queue, key, num_items):
        if False:
            while True:
                i = 10
        queue = ray._private.worker.pickle.loads(pickled_queue)
        x = None
        for item in range(num_items):
            x = queue.enqueue.remote(key, item)
        return ray.get(x)
    num_forks = 10
    num_items_per_fork = 100
    new_queue = ray._private.worker.pickle.dumps(queue)
    forks = [fork.remote(new_queue, i, num_items_per_fork) for i in range(num_forks)]
    for item in range(num_items_per_fork):
        local_fork = queue.enqueue.remote(num_forks, item)
    forks.append(local_fork)
    ray.get(forks)
    items = ray.get(queue.read.remote())
    for i in range(num_forks + 1):
        filtered_items = [item[1] for item in items if item[0] == i]
        assert filtered_items == list(range(num_items_per_fork))

def test_nested_fork(setup_queue_actor):
    if False:
        return 10
    queue = setup_queue_actor

    @ray.remote
    def fork(queue, key, num_items):
        if False:
            print('Hello World!')
        x = None
        for item in range(num_items):
            x = queue.enqueue.remote(key, item)
        return ray.get(x)

    @ray.remote
    def nested_fork(queue, key, num_items):
        if False:
            return 10
        ray.get(fork.remote(queue, key + 1, num_items))
        x = None
        for item in range(num_items):
            x = queue.enqueue.remote(key, item)
        return ray.get(x)
    num_forks = 10
    num_items_per_fork = 100
    forks = [nested_fork.remote(queue, i, num_items_per_fork) for i in range(0, num_forks, 2)]
    ray.get(forks)
    items = ray.get(queue.read.remote())
    for i in range(num_forks):
        filtered_items = [item[1] for item in items if item[0] == i]
        assert filtered_items == list(range(num_items_per_fork))

@pytest.mark.skip('Garbage collection for distributed actor handles not implemented.')
def test_garbage_collection(setup_queue_actor):
    if False:
        i = 10
        return i + 15
    queue = setup_queue_actor

    @ray.remote
    def fork(queue):
        if False:
            print('Hello World!')
        for i in range(10):
            x = queue.enqueue.remote(0, i)
            time.sleep(0.1)
        return ray.get(x)
    x = fork.remote(queue)
    ray.get(queue.read.remote())
    del queue
    print(ray.get(x))

def test_calling_put_on_actor_handle(ray_start_regular):
    if False:
        for i in range(10):
            print('nop')

    @ray.remote
    class Counter:

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.x = 0

        def inc(self):
            if False:
                return 10
            self.x += 1
            return self.x

    @ray.remote
    def f():
        if False:
            while True:
                i = 10
        return Counter.remote()
    counter = Counter.remote()
    counter_id = ray.put(counter)
    new_counter = ray.get(counter_id)
    assert ray.get(new_counter.inc.remote()) == 1
    assert ray.get(counter.inc.remote()) == 2
    assert ray.get(new_counter.inc.remote()) == 3
    ray.get(f.remote())

def test_named_but_not_detached(ray_start_regular):
    if False:
        i = 10
        return i + 15
    address = ray_start_regular['address']
    driver_script = '\nimport ray\nray.init(address="{}")\n\n@ray.remote\nclass NotDetached:\n    def ping(self):\n        return "pong"\n\nactor = NotDetached.options(name="actor").remote()\nassert ray.get(actor.ping.remote()) == "pong"\nhandle = ray.get_actor("actor")\nassert ray.util.list_named_actors() == ["actor"]\nassert ray.get(handle.ping.remote()) == "pong"\n'.format(address)
    run_string_as_driver(driver_script)
    with pytest.raises(Exception):
        assert not ray.util.list_named_actors()
        detached_actor = ray.get_actor('actor')
        ray.get(detached_actor.ping.remote())

    def check_name_available(name):
        if False:
            i = 10
            return i + 15
        try:
            ray.get_actor(name)
            return False
        except ValueError:
            return True

    @ray.remote
    class A:
        pass
    a = A.options(name='my_actor_1').remote()
    ray.kill(a, no_restart=True)
    wait_for_condition(lambda : check_name_available('my_actor_1'))
    b = A.options(name='my_actor_2').remote()
    del b
    wait_for_condition(lambda : check_name_available('my_actor_2'))

def test_detached_actor(ray_start_regular):
    if False:
        i = 10
        return i + 15

    @ray.remote
    class DetachedActor:

        def ping(self):
            if False:
                for i in range(10):
                    print('nop')
            return 'pong'
    with pytest.raises(TypeError):
        DetachedActor._remote(lifetime='detached', name=1)
    with pytest.raises(ValueError, match='Actor name cannot be an empty string'):
        DetachedActor._remote(lifetime='detached', name='')
    with pytest.raises(ValueError):
        DetachedActor._remote(lifetime='detached', name='hi', namespace='')
    with pytest.raises(TypeError):
        DetachedActor._remote(lifetime='detached', name='hi', namespace=2)
    d = DetachedActor._remote(lifetime='detached', name='d_actor')
    assert ray.get(d.ping.remote()) == 'pong'
    with pytest.raises(ValueError, match='Please use a different name'):
        DetachedActor._remote(lifetime='detached', name='d_actor')
    address = ray_start_regular['address']
    get_actor_name = 'd_actor'
    create_actor_name = 'DetachedActor'
    driver_script = '\nimport ray\nray.init(address="{}", namespace="default_test_namespace")\n\nname = "{}"\nassert ray.util.list_named_actors() == [name]\nexisting_actor = ray.get_actor(name)\nassert ray.get(existing_actor.ping.remote()) == "pong"\n\n@ray.remote\ndef foo():\n    return "bar"\n\n@ray.remote\nclass NonDetachedActor:\n    def foo(self):\n        return "bar"\n\n@ray.remote\nclass DetachedActor:\n    def ping(self):\n        return "pong"\n\n    def foobar(self):\n        actor = NonDetachedActor.remote()\n        return ray.get([foo.remote(), actor.foo.remote()])\n\nactor = DetachedActor._remote(lifetime="detached", name="{}")\nray.get(actor.ping.remote())\n'.format(address, get_actor_name, create_actor_name)
    run_string_as_driver(driver_script)
    assert len(ray.util.list_named_actors()) == 2
    assert get_actor_name in ray.util.list_named_actors()
    assert create_actor_name in ray.util.list_named_actors()
    detached_actor = ray.get_actor(create_actor_name)
    assert ray.get(detached_actor.ping.remote()) == 'pong'
    assert ray.get(detached_actor.foobar.remote()) == ['bar', 'bar']

def test_detached_actor_cleanup(ray_start_regular):
    if False:
        print('Hello World!')

    @ray.remote
    class DetachedActor:

        def ping(self):
            if False:
                while True:
                    i = 10
            return 'pong'
    dup_actor_name = 'actor'

    def create_and_kill_actor(actor_name):
        if False:
            while True:
                i = 10
        detached_actor = DetachedActor.options(lifetime='detached', name=actor_name).remote()
        assert ray.get(detached_actor.ping.remote()) == 'pong'
        del detached_actor
        assert ray.util.list_named_actors() == [dup_actor_name]
        detached_actor = ray.get_actor(dup_actor_name)
        ray.kill(detached_actor)
        actor_status = ray._private.state.actors(actor_id=detached_actor._actor_id.hex())
        max_wait_time = 10
        wait_time = 0
        while actor_status['State'] != convert_actor_state(gcs_utils.ActorTableData.DEAD):
            actor_status = ray._private.state.actors(actor_id=detached_actor._actor_id.hex())
            time.sleep(1.0)
            wait_time += 1
            if wait_time >= max_wait_time:
                assert None, 'It took too much time to kill an actor: {}'.format(detached_actor._actor_id)
    create_and_kill_actor(dup_actor_name)
    create_and_kill_actor(dup_actor_name)
    address = ray_start_regular['address']
    driver_script = '\nimport ray\nimport ray._private.gcs_utils as gcs_utils\nimport time\nfrom ray._private.test_utils import convert_actor_state\nray.init(address="{}", namespace="default_test_namespace")\n\n@ray.remote\nclass DetachedActor:\n    def ping(self):\n        return "pong"\n\n# Make sure same name is creatable after killing it.\ndetached_actor = DetachedActor.options(lifetime="detached", name="{}").remote()\nassert ray.get(detached_actor.ping.remote()) == "pong"\nray.kill(detached_actor)\n# Wait until actor dies.\nactor_status = ray._private.state.actors(actor_id=detached_actor._actor_id.hex())\nmax_wait_time = 10\nwait_time = 0\nwhile actor_status["State"] != convert_actor_state(gcs_utils.ActorTableData.DEAD): # noqa\n    actor_status = ray._private.state.actors(actor_id=detached_actor._actor_id.hex())\n    time.sleep(1.0)\n    wait_time += 1\n    if wait_time >= max_wait_time:\n        assert None, (\n            "It took too much time to kill an actor")\n'.format(address, dup_actor_name)
    run_string_as_driver(driver_script)
    create_and_kill_actor(dup_actor_name)

@pytest.mark.parametrize('ray_start_regular', [{'local_mode': True}], indirect=True)
def test_detached_actor_local_mode(ray_start_regular):
    if False:
        for i in range(10):
            print('nop')
    RETURN_VALUE = 3

    @ray.remote
    class Y:

        def f(self):
            if False:
                while True:
                    i = 10
            return RETURN_VALUE
    Y.options(lifetime='detached', name='test').remote()
    assert ray.util.list_named_actors() == ['test']
    y = ray.get_actor('test')
    assert ray.get(y.f.remote()) == RETURN_VALUE
    ray.kill(y)
    assert not ray.util.list_named_actors()
    with pytest.raises(ValueError):
        ray.get_actor('test')

@pytest.mark.parametrize('ray_start_regular', [{'local_mode': True}], indirect=True)
def test_get_actor_local_mode(ray_start_regular):
    if False:
        while True:
            i = 10

    @ray.remote
    class A:

        def hi(self):
            if False:
                for i in range(10):
                    print('nop')
            return 'hi'
    a = A.options(name='hi').remote()
    b = ray.get_actor('hi')
    assert ray.get(b.hi.remote()) == 'hi'

@pytest.mark.parametrize('ray_start_cluster', [{'num_cpus': 3, 'num_nodes': 1, 'resources': {'first_node': 5}}], indirect=True)
def test_detached_actor_cleanup_due_to_failure(ray_start_cluster):
    if False:
        for i in range(10):
            print('nop')
    cluster = ray_start_cluster
    node = cluster.add_node(resources={'second_node': 1})
    cluster.wait_for_nodes()

    @ray.remote
    class DetachedActor:

        def ping(self):
            if False:
                print('Hello World!')
            return 'pong'

        def kill_itself(self):
            if False:
                for i in range(10):
                    print('nop')
            os._exit(0)
    worker_failure_actor_name = 'worker_failure_actor_name'
    node_failure_actor_name = 'node_failure_actor_name'

    def wait_until_actor_dead(handle):
        if False:
            return 10
        actor_status = ray._private.state.actors(actor_id=handle._actor_id.hex())
        max_wait_time = 10
        wait_time = 0
        while actor_status['State'] != convert_actor_state(gcs_utils.ActorTableData.DEAD):
            actor_status = ray._private.state.actors(actor_id=handle._actor_id.hex())
            time.sleep(1.0)
            wait_time += 1
            if wait_time >= max_wait_time:
                assert None, 'It took too much time to kill an actor: {}'.format(handle._actor_id)

    def create_detached_actor_blocking(actor_name, schedule_in_second_node=False):
        if False:
            return 10
        resources = {'second_node': 1} if schedule_in_second_node else {'first_node': 1}
        actor_handle = DetachedActor.options(lifetime='detached', name=actor_name, resources=resources).remote()
        assert ray.get(actor_handle.ping.remote()) == 'pong'
        return actor_handle
    deatched_actor = create_detached_actor_blocking(worker_failure_actor_name)
    deatched_actor.kill_itself.remote()
    wait_until_actor_dead(deatched_actor)
    deatched_actor = create_detached_actor_blocking(worker_failure_actor_name)
    assert ray.get(deatched_actor.ping.remote()) == 'pong'
    deatched_actor = create_detached_actor_blocking(node_failure_actor_name, schedule_in_second_node=True)
    cluster.remove_node(node)
    wait_until_actor_dead(deatched_actor)
    deatched_actor = create_detached_actor_blocking(node_failure_actor_name)
    assert ray.get(deatched_actor.ping.remote()) == 'pong'

def test_actor_creation_task_crash(ray_start_regular):
    if False:
        print('Hello World!')

    @ray.remote(max_restarts=0)
    class Actor:

        def __init__(self):
            if False:
                print('Hello World!')
            print('crash')
            os._exit(0)

        def f(self):
            if False:
                print('Hello World!')
            return 'ACTOR OK'
    a = Actor.remote()
    with pytest.raises(ray.exceptions.RayActorError) as excinfo:
        ray.get(a.f.remote())
    assert excinfo.value.actor_id == a._actor_id.hex()

    @ray.remote(max_restarts=3)
    class RestartableActor:

        def __init__(self):
            if False:
                print('Hello World!')
            count = self.get_count()
            count += 1
            if count < 3:
                self.set_count(count)
                print('crash: ' + str(count))
                os._exit(0)
            else:
                print('no crash')

        def f(self):
            if False:
                print('Hello World!')
            return 'ACTOR OK'

        def get_count(self):
            if False:
                i = 10
                return i + 15
            value = _internal_kv_get('count')
            if value is None:
                count = 0
            else:
                count = int(value)
            return count

        def set_count(self, count):
            if False:
                for i in range(10):
                    print('nop')
            _internal_kv_put('count', str(count), True)
    ra = RestartableActor.remote()
    ray.get(ra.f.remote())

@pytest.mark.parametrize('ray_start_regular', [{'num_cpus': 2, 'resources': {'a': 1}}], indirect=True)
def test_pending_actor_removed_by_owner(ray_start_regular):
    if False:
        print('Hello World!')

    @ray.remote(num_cpus=1, resources={'a': 1})
    class A:

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.actors = []

        def create_actors(self):
            if False:
                return 10
            self.actors = [B.remote() for _ in range(2)]

    @ray.remote(resources={'a': 1})
    class B:

        def ping(self):
            if False:
                while True:
                    i = 10
            return True

    @ray.remote(resources={'a': 1})
    def f():
        if False:
            return 10
        return True
    a = A.remote()
    ray.get(a.create_actors.remote())
    del a
    a = B.remote()
    assert ray.get(a.ping.remote())
    ray.kill(a)
    assert ray.get(f.remote())

def test_pickling_actor_handle(ray_start_regular_shared):
    if False:
        for i in range(10):
            print('nop')

    @ray.remote
    class Foo:

        def method(self):
            if False:
                i = 10
                return i + 15
            pass
    f = Foo.remote()
    new_f = ray._private.worker.pickle.loads(ray._private.worker.pickle.dumps(f))
    ray.get(new_f.method.remote())

def test_pickled_actor_handle_call_in_method_twice(ray_start_regular_shared):
    if False:
        i = 10
        return i + 15

    @ray.remote
    class Actor1:

        def f(self):
            if False:
                i = 10
                return i + 15
            return 1

    @ray.remote
    class Actor2:

        def __init__(self, constructor):
            if False:
                i = 10
                return i + 15
            self.actor = constructor()

        def step(self):
            if False:
                while True:
                    i = 10
            ray.get(self.actor.f.remote())
    a = Actor1.remote()
    b = Actor2.remote(lambda : a)
    ray.get(b.step.remote())
    ray.get(b.step.remote())

def test_kill(ray_start_regular_shared):
    if False:
        for i in range(10):
            print('nop')

    @ray.remote
    class Actor:

        def hang(self):
            if False:
                return 10
            while True:
                time.sleep(1)
    actor = Actor.remote()
    result = actor.hang.remote()
    (ready, _) = ray.wait([result], timeout=0.5)
    assert len(ready) == 0
    kill_actor_and_wait_for_failure(actor)
    with pytest.raises(ray.exceptions.RayActorError):
        ray.get(result)
    with pytest.raises(ValueError):
        ray.kill('not_an_actor_handle')

def test_get_actor_no_input(ray_start_regular_shared):
    if False:
        return 10
    for bad_name in [None, '', '    ']:
        with pytest.raises(ValueError):
            ray.get_actor(bad_name)

def test_actor_resource_demand(shutdown_only):
    if False:
        print('Hello World!')
    ray.shutdown()
    cluster = ray.init(num_cpus=3)
    global_state_accessor = make_global_state_accessor(cluster)

    @ray.remote(num_cpus=2)
    class Actor:

        def foo(self):
            if False:
                print('Hello World!')
            return 'ok'
    a = Actor.remote()
    ray.get(a.foo.remote())
    time.sleep(1)
    message = global_state_accessor.get_all_resource_usage()
    resource_usages = gcs_utils.ResourceUsageBatchData.FromString(message)
    assert len(resource_usages.resource_load_by_shape.resource_demands) == 0

    @ray.remote(num_cpus=80)
    class Actor2:
        pass
    actors = []
    actors.append(Actor2.remote())
    time.sleep(1)
    message = global_state_accessor.get_all_resource_usage()
    resource_usages = gcs_utils.ResourceUsageBatchData.FromString(message)
    assert len(resource_usages.resource_load_by_shape.resource_demands) == 1
    assert resource_usages.resource_load_by_shape.resource_demands[0].shape == {'CPU': 80.0}
    assert resource_usages.resource_load_by_shape.resource_demands[0].num_infeasible_requests_queued == 1
    actors.append(Actor2.remote())
    time.sleep(1)
    message = global_state_accessor.get_all_resource_usage()
    resource_usages = gcs_utils.ResourceUsageBatchData.FromString(message)
    assert len(resource_usages.resource_load_by_shape.resource_demands) == 1
    assert resource_usages.resource_load_by_shape.resource_demands[0].num_infeasible_requests_queued == 2
    global_state_accessor.disconnect()

def test_kill_pending_actor_with_no_restart_true():
    if False:
        print('Hello World!')
    cluster = ray.init()
    global_state_accessor = make_global_state_accessor(cluster)

    @ray.remote(resources={'WORKER': 1.0})
    class PendingActor:
        pass
    actor = PendingActor.remote()
    time.sleep(1)
    ray.kill(actor, no_restart=True)

    def condition1():
        if False:
            while True:
                i = 10
        message = global_state_accessor.get_all_resource_usage()
        resource_usages = gcs_utils.ResourceUsageBatchData.FromString(message)
        if len(resource_usages.resource_load_by_shape.resource_demands) == 0:
            return True
        return False
    wait_for_condition(condition1, timeout=10)
    global_state_accessor.disconnect()
    ray.shutdown()

def test_kill_pending_actor_with_no_restart_false():
    if False:
        i = 10
        return i + 15
    cluster = ray.init()
    global_state_accessor = make_global_state_accessor(cluster)

    @ray.remote(resources={'WORKER': 1.0}, max_restarts=1)
    class PendingActor:
        pass
    actor = PendingActor.remote()
    time.sleep(1)
    ray.kill(actor, no_restart=False)

    def condition1():
        if False:
            return 10
        message = global_state_accessor.get_all_resource_usage()
        resource_usages = gcs_utils.ResourceUsageBatchData.FromString(message)
        if len(resource_usages.resource_load_by_shape.resource_demands) == 0:
            return False
        return True
    wait_for_condition(condition1, timeout=10)
    ray.kill(actor, no_restart=False)

    def condition2():
        if False:
            i = 10
            return i + 15
        message = global_state_accessor.get_all_resource_usage()
        resource_usages = gcs_utils.ResourceUsageBatchData.FromString(message)
        if len(resource_usages.resource_load_by_shape.resource_demands) == 0:
            return True
        return False
    wait_for_condition(condition2, timeout=10)
    global_state_accessor.disconnect()
    ray.shutdown()

def test_actor_timestamps(ray_start_regular):
    if False:
        i = 10
        return i + 15

    @ray.remote
    class Foo:

        def get_id(self):
            if False:
                i = 10
                return i + 15
            return ray.get_runtime_context().get_actor_id()

        def kill_self(self):
            if False:
                for i in range(10):
                    print('nop')
            sys.exit(1)

    def graceful_exit():
        if False:
            i = 10
            return i + 15
        actor = Foo.remote()
        actor_id = ray.get(actor.get_id.remote())
        state_after_starting = ray._private.state.actors()[actor_id]
        time.sleep(1)
        del actor
        time.sleep(1)
        state_after_ending = ray._private.state.actors()[actor_id]
        assert state_after_starting['StartTime'] == state_after_ending['StartTime']
        start_time = state_after_ending['StartTime']
        end_time = state_after_ending['EndTime']
        assert end_time > start_time > 0, f'Start: {start_time}, End: {end_time}'

    def not_graceful_exit():
        if False:
            while True:
                i = 10
        actor = Foo.remote()
        actor_id = ray.get(actor.get_id.remote())
        state_after_starting = ray._private.state.actors()[actor_id]
        time.sleep(1)
        actor.kill_self.remote()
        time.sleep(1)
        state_after_ending = ray._private.state.actors()[actor_id]
        assert state_after_starting['StartTime'] == state_after_ending['StartTime']
        start_time = state_after_ending['StartTime']
        end_time = state_after_ending['EndTime']
        assert end_time > start_time > 0, f'Start: {start_time}, End: {end_time}'

    def restarted():
        if False:
            print('Hello World!')
        actor = Foo.options(max_restarts=1, max_task_retries=-1).remote()
        actor_id = ray.get(actor.get_id.remote())
        state_after_starting = ray._private.state.actors()[actor_id]
        time.sleep(1)
        actor.kill_self.remote()
        time.sleep(1)
        actor.kill_self.remote()
        time.sleep(1)
        state_after_ending = ray._private.state.actors()[actor_id]
        assert state_after_starting['StartTime'] == state_after_ending['StartTime']
        start_time = state_after_ending['StartTime']
        end_time = state_after_ending['EndTime']
        assert end_time > start_time > 0, f'Start: {start_time}, End: {end_time}'
    graceful_exit()
    not_graceful_exit()
    restarted()

def test_actor_namespace_access(ray_start_regular):
    if False:
        for i in range(10):
            print('nop')

    @ray.remote
    class A:

        def hi(self):
            if False:
                while True:
                    i = 10
            return 'hi'
    A.options(name='actor_in_current_namespace', lifetime='detached').remote()
    A.options(name='actor_name', namespace='namespace', lifetime='detached').remote()
    ray.get_actor('actor_in_current_namespace')
    ray.get_actor('actor_name', namespace='namespace')
    match_str = 'Failed to look up actor with name.*'
    with pytest.raises(ValueError, match=match_str):
        ray.get_actor('actor_name')

def test_get_actor_after_killed(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    ray.init(num_cpus=2)

    @ray.remote
    class A:

        def ready(self):
            if False:
                while True:
                    i = 10
            return True
    actor = A.options(name='actor', namespace='namespace', lifetime='detached').remote()
    ray.kill(actor)
    with pytest.raises(ValueError):
        ray.get_actor('actor', namespace='namespace')
    actor = A.options(name='actor_2', namespace='namespace', lifetime='detached', max_restarts=1, max_task_retries=-1).remote()
    ray.kill(actor, no_restart=False)
    assert ray.get(ray.get_actor('actor_2', namespace='namespace').ready.remote())

def test_get_actor_race_condition(shutdown_only):
    if False:
        i = 10
        return i + 15

    @ray.remote
    class Actor:

        def ping(self):
            if False:
                return 10
            return 'ok'

    @ray.remote
    def getter(name):
        if False:
            return 10
        try:
            try:
                actor = ray.get_actor(name)
            except Exception:
                print('Get failed, trying to create', name)
                actor = Actor.options(name=name, lifetime='detached').remote()
        except Exception:
            print('Someone else created it, trying to get')
            actor = ray.get_actor(name)
        result = ray.get(actor.ping.remote())
        return result

    def do_run(name, concurrency=4):
        if False:
            while True:
                i = 10
        name = 'actor_' + str(name)
        tasks = [getter.remote(name) for _ in range(concurrency)]
        result = ray.get(tasks)
        ray.kill(ray.get_actor(name))
        return result
    for i in range(50):
        CONCURRENCY = 8
        results = do_run(i, concurrency=CONCURRENCY)
        assert ['ok'] * CONCURRENCY == results

def test_get_actor_in_remote_workers(ray_start_cluster):
    if False:
        i = 10
        return i + 15
    'Make sure we can get and create actors without\n    race condition in a remote worker.\n\n    Check https://github.com/ray-project/ray/issues/20092. # noqa\n    '
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=0)
    cluster.add_node(num_cpus=1)
    ray.init(address=cluster.address, namespace='xxx')

    @ray.remote(num_cpus=0)
    class RemoteProc:

        def __init__(self):
            if False:
                return 10
            pass

        def procTask(self, a, b):
            if False:
                return 10
            print('[%s]-> %s' % (a, b))
            return (a, b)

    @ray.remote
    def submit_named_actors():
        if False:
            for i in range(10):
                print('nop')
        RemoteProc.options(name='test', lifetime='detached', max_concurrency=10, namespace='xxx').remote()
        proc = ray.get_actor('test', namespace='xxx')
        ray.get(proc.procTask.remote(1, 2))
        ray.kill(proc)
        RemoteProc.options(name='test', lifetime='detached', max_concurrency=10, namespace='xxx').remote()
        proc = ray.get_actor('test', namespace='xxx')
        return ray.get(proc.procTask.remote(1, 2))
    assert (1, 2) == ray.get(submit_named_actors.remote())

def test_resource_leak_when_cancel_actor_in_phase_of_creating(ray_start_cluster):
    if False:
        while True:
            i = 10
    'Make sure there is no resource leak when cancel an actor in phase of\n    creating.\n\n    Check https://github.com/ray-project/ray/issues/27743. # noqa\n    '
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=2)
    ray.init(address=cluster.address)
    cluster.wait_for_nodes()

    @ray.remote(num_cpus=1)
    class Actor:

        def __init__(self, signal_1, signal_2):
            if False:
                print('Hello World!')
            signal_1.send.remote()
            ray.get(signal_2.wait.remote())
            pass
    signal_1 = SignalActor.remote()
    signal_2 = SignalActor.remote()
    actor = Actor.remote(signal_1, signal_2)
    wait_for_condition(lambda : ray.available_resources()['CPU'] != 2)
    (ready_ids, _) = ray.wait([signal_1.wait.remote()], timeout=3.0)
    assert len(ready_ids) == 1
    ray.kill(actor)
    wait_for_condition(lambda : ray.available_resources()['CPU'] == 2)

def test_actor_gc(monkeypatch, shutdown_only):
    if False:
        while True:
            i = 10
    MAX_DEAD_ACTOR_CNT = 5
    with monkeypatch.context() as m:
        m.setenv('RAY_maximum_gcs_destroyed_actor_cached_count', MAX_DEAD_ACTOR_CNT)
        ray.init()

        @ray.remote
        class Actor:

            def ready(self):
                if False:
                    return 10
                pass
        actors = [Actor.remote() for _ in range(10)]
        ray.get([actor.ready.remote() for actor in actors])
        alive_actors = 0
        for a in list_actors():
            if a['state'] == 'ALIVE':
                alive_actors += 1
        assert alive_actors == 10
        del actors

        def verify_cached_dead_actor_cleaned():
            if False:
                return 10
            return len(list_actors()) == MAX_DEAD_ACTOR_CNT
        wait_for_condition(verify_cached_dead_actor_cleaned)
        actors = [Actor.options(lifetime='detached').remote() for _ in range(10)]
        ray.get([actor.ready.remote() for actor in actors])
        alive_actors = 0
        for a in list_actors():
            if a['state'] == 'ALIVE':
                alive_actors += 1
        assert alive_actors == 10
        for actor in actors:
            ray.kill(actor)
        wait_for_condition(verify_cached_dead_actor_cleaned)
        driver = '\nimport ray\nfrom ray.util.state import list_actors\nray.init("auto")\n\n@ray.remote\nclass A:\n    def ready(self):\n        pass\n\nactors = [A.remote() for _ in range(10)]\nray.get([actor.ready.remote() for actor in actors])\nalive_actors = 0\nfor a in list_actors():\n    if a["state"] == "ALIVE":\n        alive_actors += 1\nassert alive_actors == 10\n'
        run_string_as_driver(driver)
        wait_for_condition(verify_cached_dead_actor_cleaned)
        print(list_actors())
if __name__ == '__main__':
    import pytest
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))