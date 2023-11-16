import subprocess
import sys
import time
import grpc
import numpy as np
import psutil
import pytest
from grpc._channel import _InactiveRpcError
from ray.util.state import list_tasks
from ray._private.state_api_test_utils import verify_failed_task
import ray
import ray._private.ray_constants as ray_constants
import ray.experimental.internal_kv as internal_kv
from ray import NodeID
from ray._private.test_utils import SignalActor, get_error_message, init_error_pubsub, run_string_as_driver, wait_for_condition, kill_raylet
from ray.cluster_utils import Cluster, cluster_not_supported
from ray.core.generated import gcs_service_pb2, gcs_service_pb2_grpc, node_manager_pb2, node_manager_pb2_grpc
from ray.exceptions import LocalRayletDiedError

def search_raylet(cluster):
    if False:
        while True:
            i = 10
    'Return the number of running processes.'
    raylets = []
    for node in cluster.list_all_nodes():
        procs = node.all_processes
        raylet_proc_info = procs.get(ray_constants.PROCESS_TYPE_RAYLET)
        if raylet_proc_info:
            assert len(raylet_proc_info) == 1
            raylet = psutil.Process(raylet_proc_info[0].process.pid)
            if raylet.status() == 'running':
                raylets.append(psutil.Process(raylet_proc_info[0].process.pid))
    return raylets

def test_retry_system_level_error(ray_start_regular):
    if False:
        return 10

    @ray.remote
    class Counter:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.value = 0

        def increment(self):
            if False:
                print('Hello World!')
            self.value += 1
            return self.value

    @ray.remote(max_retries=1)
    def func(counter):
        if False:
            for i in range(10):
                print('nop')
        count = counter.increment.remote()
        if ray.get(count) == 1:
            import os
            os._exit(0)
        else:
            return 1
    counter1 = Counter.remote()
    r1 = func.remote(counter1)
    assert ray.get(r1) == 1
    counter2 = Counter.remote()
    r2 = func.options(max_retries=0).remote(counter2)
    with pytest.raises(ray.exceptions.WorkerCrashedError):
        ray.get(r2)

def test_retry_application_level_error(ray_start_regular):
    if False:
        i = 10
        return i + 15

    @ray.remote
    class Counter:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.value = 0

        def increment(self):
            if False:
                return 10
            self.value += 1
            return self.value

    @ray.remote(max_retries=1, retry_exceptions=True)
    def func(counter):
        if False:
            return 10
        count = counter.increment.remote()
        if ray.get(count) == 1:
            raise ValueError()
        else:
            return 2
    counter1 = Counter.remote()
    r1 = func.remote(counter1)
    assert ray.get(r1) == 2
    counter2 = Counter.remote()
    r2 = func.options(max_retries=0).remote(counter2)
    with pytest.raises(ValueError):
        ray.get(r2)
    counter3 = Counter.remote()
    r3 = func.options(retry_exceptions=False).remote(counter3)
    with pytest.raises(ValueError):
        ray.get(r3)

class CountError(Exception):
    pass

def test_retry_application_level_error_exception_filter(ray_start_regular):
    if False:
        for i in range(10):
            print('nop')

    @ray.remote
    class Counter:

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.value = 0

        def increment(self):
            if False:
                for i in range(10):
                    print('nop')
            self.value += 1
            return self.value

    @ray.remote(max_retries=1, retry_exceptions=[CountError])
    def func(counter):
        if False:
            while True:
                i = 10
        if counter is None:
            raise ValueError()
        count = counter.increment.remote()
        if ray.get(count) == 1:
            raise CountError()
        else:
            return 2
    r0 = func.remote(None)
    with pytest.raises(ValueError):
        ray.get(r0)
    counter1 = Counter.remote()
    r1 = func.remote(counter1)
    assert ray.get(r1) == 2
    counter2 = Counter.remote()
    r2 = func.options(max_retries=0).remote(counter2)
    with pytest.raises(CountError):
        ray.get(r2)
    counter3 = Counter.remote()
    r3 = func.options(retry_exceptions=False).remote(counter3)
    with pytest.raises(CountError):
        ray.get(r3)

@pytest.mark.xfail(cluster_not_supported, reason='cluster not supported')
def test_connect_with_disconnected_node(shutdown_only):
    if False:
        while True:
            i = 10
    config = {'health_check_failure_threshold': 50, 'health_check_period_ms': 10, 'health_check_initial_delay_ms': 0}
    cluster = Cluster()
    cluster.add_node(num_cpus=0, _system_config=config)
    ray.init(address=cluster.address)
    p = init_error_pubsub()
    errors = get_error_message(p, 1, timeout=5)
    assert len(errors) == 0
    dead_node = cluster.add_node(num_cpus=0)
    cluster.remove_node(dead_node, allow_graceful=False)
    errors = get_error_message(p, 1, ray_constants.REMOVED_NODE_ERROR)
    assert len(errors) == 1
    dead_node = cluster.add_node(num_cpus=0)
    cluster.remove_node(dead_node, allow_graceful=False)
    errors = get_error_message(p, 1, ray_constants.REMOVED_NODE_ERROR)
    assert len(errors) == 1
    removing_node = cluster.add_node(num_cpus=0)
    cluster.remove_node(removing_node, allow_graceful=True)
    errors = get_error_message(p, 1, ray_constants.REMOVED_NODE_ERROR, timeout=2)
    assert len(errors) == 0
    errors = get_error_message(p, 1, timeout=2)
    assert len(errors) == 0
    p.close()

def test_detached_actor_ref(call_ray_start):
    if False:
        print('Hello World!')
    address = call_ray_start
    driver_script = '\nimport ray\nimport time\n\n\n@ray.remote\ndef foo(x):\n    return ray.put(42)\n\n\n@ray.remote\nclass Actor:\n    def __init__(self):\n        self.ref = None\n\n    def invoke(self):\n        self.ref = foo.remote(0)\n        # Wait for the task to finish before exiting the driver.\n        ray.get(self.ref)\n\n    def get(self):\n        print("get", self.ref)\n        return self.ref\n\n\nif __name__ == "__main__":\n    ray.init(address="{}", namespace="default")\n    a = Actor.options(name="holder", lifetime="detached").remote()\n    # Wait for the task to finish before exiting the driver.\n    ray.get(a.invoke.remote())\n    print("success")\n'.format(address)
    out = run_string_as_driver(driver_script)
    assert 'success' in out
    import time
    time.sleep(5)
    ray.init(address=address, namespace='default')
    actor = ray.get_actor('holder')
    x = actor.get.remote()
    while isinstance(x, ray.ObjectRef):
        x = ray.get(x)
    assert x == 42

@pytest.mark.skipif(sys.platform == 'win32', reason='Failing on Windows.')
@pytest.mark.parametrize('debug_enabled', [False, True])
def test_object_lost_error(ray_start_cluster, debug_enabled):
    if False:
        while True:
            i = 10
    cluster = ray_start_cluster
    system_config = {'health_check_failure_threshold': 3, 'health_check_period_ms': 1000, 'health_check_initial_delay_ms': 0}
    if debug_enabled:
        system_config['record_ref_creation_sites'] = True
    cluster.add_node(num_cpus=0, _system_config=system_config)
    ray.init(address=cluster.address)
    worker_node = cluster.add_node(num_cpus=1)

    @ray.remote(num_cpus=1)
    class Actor:

        def __init__(self):
            if False:
                print('Hello World!')
            return

        def foo(self):
            if False:
                i = 10
                return i + 15
            return 'x' * 1000000

        def done(self):
            if False:
                print('Hello World!')
            return

    @ray.remote
    def borrower(ref):
        if False:
            for i in range(10):
                print('nop')
        ray.get(ref[0])

    @ray.remote
    def task_arg(ref):
        if False:
            return 10
        return
    a = Actor.remote()
    x = a.foo.remote()
    ray.get(a.done.remote())
    cluster.remove_node(worker_node, allow_graceful=False)
    cluster.add_node(num_cpus=1)
    y = borrower.remote([x])
    try:
        ray.get(x)
        assert False
    except ray.exceptions.ObjectLostError as e:
        error = str(e)
        print(error)
        assert ('actor call' in error) == debug_enabled
        assert ('test_object_lost_error' in error) == debug_enabled
    try:
        ray.get(y)
        assert False
    except ray.exceptions.RayTaskError as e:
        error = str(e)
        print(error)
        assert ('actor call' in error) == debug_enabled
        assert ('test_object_lost_error' in error) == debug_enabled
    try:
        ray.get(task_arg.remote(x))
    except ray.exceptions.RayTaskError as e:
        error = str(e)
        print(error)
        assert ('actor call' in error) == debug_enabled
        assert ('test_object_lost_error' in error) == debug_enabled

@pytest.mark.skipif(sys.platform == 'win32', reason='Failing on Windows.')
@pytest.mark.parametrize('ray_start_cluster_head', [{'num_cpus': 0, '_system_config': {'health_check_initial_delay_ms': 0, 'health_check_period_ms': 100, 'health_check_failure_threshold': 10}}], indirect=True)
def test_raylet_graceful_shutdown_through_rpc(ray_start_cluster_head, error_pubsub):
    if False:
        i = 10
        return i + 15
    '\n    Prepare the cluster.\n    '
    cluster = ray_start_cluster_head
    head_node_port = None
    for n in ray.nodes():
        head_node_port = int(n['NodeManagerPort'])
    worker = cluster.add_node(num_cpus=1)
    cluster.wait_for_nodes()
    worker_node_port = None
    for n in ray.nodes():
        port = int(n['NodeManagerPort'])
        if port != head_node_port and n['alive']:
            worker_node_port = port
    '\n    warm up the cluster\n    '

    @ray.remote
    def f():
        if False:
            print('Hello World!')
        pass
    ray.get(f.remote())

    def kill_raylet(ip, port, graceful=True):
        if False:
            i = 10
            return i + 15
        raylet_address = f'{ip}:{port}'
        channel = grpc.insecure_channel(raylet_address)
        stub = node_manager_pb2_grpc.NodeManagerServiceStub(channel)
        print(f'Sending a shutdown request to {ip}:{port}')
        try:
            stub.ShutdownRaylet(node_manager_pb2.ShutdownRayletRequest(graceful=graceful))
        except _InactiveRpcError:
            assert not graceful
    '\n    Kill the first worker ungracefully.\n    '
    ip = worker.node_ip_address
    kill_raylet(ip, worker_node_port, graceful=False)
    p = error_pubsub
    errors = get_error_message(p, 1, ray_constants.REMOVED_NODE_ERROR, timeout=10)
    assert 'has missed too many heartbeats from it' in errors[0]['error_message']
    try:
        wait_for_condition(lambda : len(search_raylet(cluster)) == 1)
    except Exception:
        print('More than one raylets are detected.')
        print(search_raylet(cluster))
    '\n    Kill the second worker gracefully.\n    '
    worker = cluster.add_node(num_cpus=0)
    worker_node_port = None
    for n in ray.nodes():
        port = int(n['NodeManagerPort'])
        if port != head_node_port and n['alive']:
            worker_node_port = port
    ip = worker.node_ip_address
    kill_raylet(ip, worker_node_port, graceful=True)
    p = error_pubsub
    errors = get_error_message(p, 1, ray_constants.REMOVED_NODE_ERROR, timeout=5)
    assert len(errors) == 0
    try:
        wait_for_condition(lambda : len(search_raylet(cluster)) == 1)
    except Exception:
        print('More than one raylets are detected.')
        print(search_raylet(cluster))
    '\n    Make sure head node is not dead.\n    '
    ray.get(f.options(num_cpus=0).remote())

@pytest.mark.skipif(sys.platform == 'win32', reason='Failing on Windows.')
@pytest.mark.parametrize('ray_start_cluster_head', [{'num_cpus': 0, '_system_config': {'health_check_failure_threshold': 10, 'health_check_period_ms': 100, 'health_check_initial_delay_ms': 0}}], indirect=True)
def test_gcs_drain(ray_start_cluster_head, error_pubsub):
    if False:
        i = 10
        return i + 15
    '\n    Prepare the cluster.\n    '
    cluster = ray_start_cluster_head
    head_node_id = ray.nodes()[0]['NodeID']
    NUM_NODES = 2
    for _ in range(NUM_NODES):
        cluster.add_node(num_cpus=1)
    worker_node_ids = []
    for n in ray.nodes():
        if n['NodeID'] != head_node_id:
            worker_node_ids.append(n['NodeID'])
    '\n    Warm up the cluster.\n    '

    @ray.remote(num_cpus=1)
    class A:

        def ready(self):
            if False:
                i = 10
                return i + 15
            pass
    actors = [A.remote() for _ in range(NUM_NODES)]
    ray.get([actor.ready.remote() for actor in actors])
    '\n    Test batch drain.\n    '
    gcs_server_addr = cluster.gcs_address
    options = ray_constants.GLOBAL_GRPC_OPTIONS
    channel = grpc.insecure_channel(gcs_server_addr, options)
    stub = gcs_service_pb2_grpc.NodeInfoGcsServiceStub(channel)
    r = gcs_service_pb2.DrainNodeRequest()
    for worker_id in worker_node_ids:
        data = r.drain_node_data.add()
        data.node_id = NodeID.from_hex(worker_id).binary()
    stub.DrainNode(r)
    p = error_pubsub
    errors = get_error_message(p, 1, ray_constants.REMOVED_NODE_ERROR, timeout=5)
    assert len(errors) == 0
    try:
        wait_for_condition(lambda : len(search_raylet(cluster)) == 1)
    except Exception:
        print('More than one raylets are detected.')
        print(search_raylet(cluster))
    '\n    Make sure the API is idempotent.\n    '
    for _ in range(10):
        stub.DrainNode(r)
    p = error_pubsub
    errors = get_error_message(p, 1, ray_constants.REMOVED_NODE_ERROR, timeout=5)
    assert len(errors) == 0
    '\n    Make sure the GCS states are updated properly.\n    '
    for n in ray.nodes():
        node_id = n['NodeID']
        is_alive = n['Alive']
        if node_id == head_node_id:
            assert is_alive
        if node_id in worker_node_ids:
            assert not is_alive
    '\n    Make sure head node is not dead and functional.\n    '
    a = A.options(num_cpus=0).remote()
    ray.get(a.ready.remote())

def test_worker_start_timeout(monkeypatch, ray_start_cluster):
    if False:
        i = 10
        return i + 15
    with monkeypatch.context() as m:
        m.setenv('RAY_testing_asio_delay_us', 'InternalKVGcsService.grpc_server.InternalKVGet=2000000:2000000')
        m.setenv('RAY_worker_register_timeout_seconds', '1')
        m.setenv('RAY_prestart_worker_first_driver', 'false')
        cluster = ray_start_cluster
        cluster.add_node(num_cpus=4, object_store_memory=1000000000.0)
        script = "\nimport ray\nray.init(address='auto')\n\n@ray.remote\ndef task():\n    return None\n\nray.get(task.remote(), timeout=3)\n"
        with pytest.raises(subprocess.CalledProcessError) as e:
            run_string_as_driver(script)
        assert "The process is still alive, probably it's hanging during start" in e.value.output.decode()
        assert 'Received a register request from an unknown worker shim process' not in e.value.output.decode()

def test_task_failure_when_driver_local_raylet_dies(ray_start_cluster):
    if False:
        print('Hello World!')
    cluster = ray_start_cluster
    head = cluster.add_node(num_cpus=4, resources={'foo': 1})
    cluster.wait_for_nodes()
    ray.init(address=cluster.address)

    @ray.remote(resources={'foo': 1})
    def func():
        if False:
            print('Hello World!')
        internal_kv._internal_kv_put('test_func', 'func')
        while True:
            time.sleep(1)
    func.remote()
    while not internal_kv._internal_kv_exists('test_func'):
        time.sleep(0.1)
    ret = func.options(name='task-local-raylet-dead').remote()

    def task_running():
        if False:
            i = 10
            return i + 15
        tasks = list_tasks(filters=[('name', '=', 'task-local-raylet-dead')])
        assert len(tasks) == 1
        assert tasks[0]['state'] == 'PENDING_NODE_ASSIGNMENT'
        return True
    wait_for_condition(task_running)
    head.kill_raylet()
    with pytest.raises(LocalRayletDiedError):
        ray.get(ret)
    wait_for_condition(verify_failed_task, name='task-local-raylet-dead', error_type='LOCAL_RAYLET_DIED', error_message='The worker failed to receive a response from the local raylet')

def test_locality_aware_scheduling_for_dead_nodes(shutdown_only):
    if False:
        print('Hello World!')
    'Test that locality-ware scheduling can handle dead nodes.'
    config = {'health_check_failure_threshold': 5, 'health_check_period_ms': 50, 'health_check_initial_delay_ms': 0}
    cluster = Cluster()
    cluster.add_node(num_cpus=4, resources={'node1': 1}, _system_config=config)
    cluster.wait_for_nodes()
    ray.init(address=cluster.address)
    node2 = cluster.add_node(num_cpus=4, resources={'node2': 1})
    node3 = cluster.add_node(num_cpus=4, resources={'node3': 1})
    node4 = cluster.add_node(num_cpus=4, resources={'node4': 1})
    cluster.wait_for_nodes()

    @ray.remote(resources={'node2': 0.1})
    def create_object():
        if False:
            i = 10
            return i + 15
        return np.zeros(10 * 1024 * 1024, dtype=np.uint8)
    obj1 = create_object.remote()
    obj2 = create_object.remote()

    @ray.remote
    class MyActor:

        def __init__(self, obj_refs):
            if False:
                for i in range(10):
                    print('nop')
            self.obj_refs = obj_refs
            self.obj = ray.get(obj_refs)

        def ready(self):
            if False:
                print('Hello World!')
            return True
    actors = [MyActor.options(resources={'node2': 0.1}).remote([obj1, obj2]), MyActor.options(resources={'node3': 0.1}).remote([obj1]), MyActor.options(resources={'node4': 0.1}).remote([obj2])]
    assert all((ray.get(actor.ready.remote()) is True for actor in actors))

    @ray.remote
    def func(obj1, obj2):
        if False:
            for i in range(10):
                print('nop')
        return ray._private.worker.global_worker.node.unique_id
    assert ray.get(func.remote(obj1, obj2)) == node2.unique_id
    node2.kill_raylet()
    time.sleep(1)
    target_node = ray.get(func.remote(obj1, obj2))
    assert target_node == node3.unique_id or target_node == node4.unique_id

def test_actor_task_fast_fail(ray_start_cluster):
    if False:
        while True:
            i = 10

    @ray.remote(max_restarts=1, max_task_retries=0)
    class SlowActor:

        def __init__(self, signal_actor):
            if False:
                print('Hello World!')
            if ray.get_runtime_context().was_current_actor_reconstructed:
                ray.get(signal_actor.wait.remote())

        def ping(self):
            if False:
                print('Hello World!')
            return 'pong'
    signal = SignalActor.remote()
    actor = SlowActor.remote(signal)
    ray.get(actor.ping.remote())
    ray.kill(actor, no_restart=False)
    time.sleep(1)
    with pytest.raises(ray.exceptions.RayActorError):
        ray.get(actor.ping.remote())
    signal.send.remote()
    time.sleep(1)
    ray.get(actor.ping.remote())

def test_task_crash_after_raylet_dead_throws_node_died_error():
    if False:
        return 10

    @ray.remote(max_retries=0)
    def sleeper():
        if False:
            while True:
                i = 10
        import os
        time.sleep(3)
        os.kill(os.getpid(), 9)
    with ray.init():
        ref = sleeper.remote()
        raylet = ray.nodes()[0]
        kill_raylet(raylet)
        with pytest.raises(ray.exceptions.NodeDiedError) as error:
            ray.get(ref)
        message = str(error)
        assert raylet['NodeManagerAddress'] in message

def test_accessing_actor_after_cluster_crashed(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    ray.init()

    @ray.remote
    class A:

        def f(self):
            if False:
                print('Hello World!')
            return
    a = A.remote()
    ray.get(a.f.remote())
    ray.shutdown()
    ray.init()
    with pytest.raises(Exception) as exc_info:
        ray.get(a.f.remote())
    assert "It might be dead or it's from a different cluster" in exc_info.value.args[0]
if __name__ == '__main__':
    import os
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))