import multiprocessing
import time
import warnings
from collections import defaultdict
import numpy as np
import pytest
import ray
from ray.cluster_utils import Cluster, cluster_not_supported
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
if multiprocessing.cpu_count() < 40 or ray._private.utils.get_system_memory() < 50 * 10 ** 9:
    warnings.warn('This test must be run on large machines.')

def create_cluster(num_nodes):
    if False:
        print('Hello World!')
    cluster = Cluster()
    for i in range(num_nodes):
        cluster.add_node(resources={str(i): 100}, object_store_memory=10 ** 9)
    ray.init(address=cluster.address)
    return cluster

@pytest.fixture()
def ray_start_cluster_with_resource():
    if False:
        print('Hello World!')
    num_nodes = 5
    cluster = create_cluster(num_nodes)
    yield (cluster, num_nodes)
    ray.shutdown()
    cluster.shutdown()

@pytest.mark.parametrize('ray_start_cluster_head', [{'num_cpus': 0, 'object_store_memory': 75 * 1024 * 1024}], indirect=True)
def test_object_transfer_during_oom(ray_start_cluster_head):
    if False:
        for i in range(10):
            print('nop')
    cluster = ray_start_cluster_head
    cluster.add_node(object_store_memory=75 * 1024 * 1024)

    @ray.remote
    def put():
        if False:
            for i in range(10):
                print('nop')
        return np.random.rand(5 * 1024 * 1024)
    _ = ray.put(np.random.rand(5 * 1024 * 1024))
    remote_ref = put.remote()
    ray.get(remote_ref)

@pytest.mark.skip(reason='TODO(ekl)')
def test_object_broadcast(ray_start_cluster_with_resource):
    if False:
        print('Hello World!')
    (cluster, num_nodes) = ray_start_cluster_with_resource

    @ray.remote
    def f(x):
        if False:
            return 10
        return
    x = np.zeros(1024 * 1024, dtype=np.uint8)

    @ray.remote
    def create_object():
        if False:
            i = 10
            return i + 15
        return np.zeros(1024 * 1024, dtype=np.uint8)
    object_refs = []
    for _ in range(3):
        x_id = ray.put(x)
        object_refs.append(x_id)
        ray.get([f._remote(args=[x_id], resources={str(i % num_nodes): 1}) for i in range(10 * num_nodes)])
    for _ in range(3):
        x_id = create_object.remote()
        object_refs.append(x_id)
        ray.get([f._remote(args=[x_id], resources={str(i % num_nodes): 1}) for i in range(10 * num_nodes)])
    time.sleep(1)
    transfer_events = ray._private.state.object_transfer_timeline()
    for x_id in object_refs:
        relevant_events = [event for event in transfer_events if event['cat'] == 'transfer_send' and event['args'][0] == x_id.hex() and (event['args'][2] == 1)]
        deduplicated_relevant_events = [event for event in relevant_events if event['cname'] != 'black']
        assert len(deduplicated_relevant_events) * 2 == len(relevant_events)
        relevant_events = deduplicated_relevant_events
        assert len(relevant_events) >= num_nodes - 1
        if len(relevant_events) > num_nodes - 1:
            warnings.warn('This object was transferred {} times, when only {} transfers were required.'.format(len(relevant_events), num_nodes - 1))
        assert len(relevant_events) <= (num_nodes - 1) * num_nodes / 2
        send_counts = defaultdict(int)
        for event in relevant_events:
            send_counts[event['pid'], event['tid']] += 1
        assert all((value == 1 for value in send_counts.values()))

def test_actor_broadcast(ray_start_cluster_with_resource):
    if False:
        while True:
            i = 10
    (cluster, num_nodes) = ray_start_cluster_with_resource

    @ray.remote
    class Actor:

        def ready(self):
            if False:
                return 10
            pass

        def set_weights(self, x):
            if False:
                print('Hello World!')
            pass
    actors = [Actor._remote(args=[], kwargs={}, num_cpus=0.01, resources={str(i % num_nodes): 1}) for i in range(30)]
    ray.get([a.ready.remote() for a in actors])
    object_refs = []
    for _ in range(5):
        x_id = ray.put(np.zeros(1024 * 1024, dtype=np.uint8))
        object_refs.append(x_id)
        ray.get([a.set_weights.remote(x_id) for a in actors])
    time.sleep(1)

def test_many_small_transfers(ray_start_cluster_with_resource):
    if False:
        while True:
            i = 10
    (cluster, num_nodes) = ray_start_cluster_with_resource

    @ray.remote
    def f(*args):
        if False:
            print('Hello World!')
        pass

    def do_transfers():
        if False:
            for i in range(10):
                print('nop')
        id_lists = []
        for i in range(num_nodes):
            id_lists.append([f._remote(args=[], kwargs={}, resources={str(i): 1}) for _ in range(1000)])
        ids = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                ids.append(f._remote(args=id_lists[j], kwargs={}, resources={str(i): 1}))
        ray.get(ids)
    do_transfers()
    do_transfers()
    do_transfers()
    do_transfers()

def test_pull_request_retry(ray_start_cluster):
    if False:
        return 10
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=0, num_gpus=1, object_store_memory=100 * 2 ** 20)
    cluster.add_node(num_cpus=1, num_gpus=0, object_store_memory=100 * 2 ** 20)
    cluster.wait_for_nodes()
    ray.init(address=cluster.address)

    @ray.remote
    def put():
        if False:
            return 10
        return np.zeros(64 * 2 ** 20, dtype=np.int8)

    @ray.remote(num_cpus=0, num_gpus=1)
    def driver():
        if False:
            return 10
        local_ref = ray.put(np.zeros(64 * 2 ** 20, dtype=np.int8))
        remote_ref = put.remote()
        (ready, _) = ray.wait([remote_ref], timeout=30)
        assert len(ready) == 1
        del local_ref
        (ready, _) = ray.wait([remote_ref], timeout=20)
        assert len(ready) > 0
    ray.get(driver.remote())

@pytest.mark.xfail(cluster_not_supported, reason='cluster not supported')
def test_pull_bundles_admission_control(ray_start_cluster):
    if False:
        for i in range(10):
            print('nop')
    cluster = ray_start_cluster
    object_size = int(6000000.0)
    num_objects = 10
    num_tasks = 10
    cluster.add_node(num_cpus=0, object_store_memory=2 * num_tasks * num_objects * object_size)
    cluster.wait_for_nodes()
    ray.init(address=cluster.address)
    cluster.add_node(num_cpus=1, object_store_memory=1.5 * num_objects * object_size)
    cluster.wait_for_nodes()

    @ray.remote
    def foo(*args):
        if False:
            while True:
                i = 10
        return
    args = []
    for _ in range(num_tasks):
        task_args = [ray.put(np.zeros(object_size, dtype=np.uint8)) for _ in range(num_objects)]
        args.append(task_args)
    tasks = [foo.remote(*task_args) for task_args in args]
    ray.get(tasks)

@pytest.mark.xfail(cluster_not_supported, reason='cluster not supported')
def test_pull_bundles_pinning(ray_start_cluster):
    if False:
        for i in range(10):
            print('nop')
    cluster = ray_start_cluster
    object_size = int(50000000.0)
    num_objects = 10
    cluster.add_node(num_cpus=0, object_store_memory=1000000000.0)
    cluster.wait_for_nodes()
    ray.init(address=cluster.address)
    cluster.add_node(num_cpus=1, object_store_memory=200000000.0)
    cluster.wait_for_nodes()

    @ray.remote(num_cpus=1)
    def foo(*args):
        if False:
            i = 10
            return i + 15
        return
    task_args = [ray.put(np.zeros(object_size, dtype=np.uint8)) for _ in range(num_objects)]
    ray.get(foo.remote(*task_args))

@pytest.mark.xfail(cluster_not_supported, reason='cluster not supported')
def test_pull_bundles_admission_control_dynamic(enable_mac_large_object_store, ray_start_cluster):
    if False:
        print('Hello World!')
    cluster = ray_start_cluster
    object_size = int(6000000.0)
    num_objects = 20
    num_tasks = 20
    cluster.add_node(num_cpus=0, object_store_memory=2 * num_tasks * num_objects * object_size)
    cluster.wait_for_nodes()
    ray.init(address=cluster.address)
    cluster.add_node(num_cpus=1, object_store_memory=2.5 * num_objects * object_size)
    cluster.wait_for_nodes()

    @ray.remote
    def foo(i, *args):
        if False:
            print('Hello World!')
        print('foo', i)
        return

    @ray.remote
    def allocate(i):
        if False:
            print('Hello World!')
        print('allocate', i)
        return np.zeros(object_size, dtype=np.uint8)
    args = []
    for _ in range(num_tasks):
        task_args = [ray.put(np.zeros(object_size, dtype=np.uint8)) for _ in range(num_objects)]
        args.append(task_args)
    allocated = [allocate.remote(i) for i in range(num_objects)]
    ray.get(allocated)
    tasks = [foo.remote(i, *task_args) for (i, task_args) in enumerate(args)]
    ray.get(tasks)
    del allocated

@pytest.mark.xfail(cluster_not_supported, reason='cluster not supported')
def test_max_pinned_args_memory(ray_start_cluster):
    if False:
        while True:
            i = 10
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=0, object_store_memory=200 * 1024 * 1024, _system_config={'max_task_args_memory_fraction': 0.7})
    ray.init(address=cluster.address)
    cluster.add_node(num_cpus=3, object_store_memory=100 * 1024 * 1024)

    @ray.remote
    def f(arg):
        if False:
            i = 10
            return i + 15
        time.sleep(1)
        return np.zeros(30 * 1024 * 1024, dtype=np.uint8)
    x = np.zeros(30 * 1024 * 1024, dtype=np.uint8)
    ray.get([f.remote(ray.put(x)) for _ in range(3)])

    @ray.remote
    def large_arg(arg):
        if False:
            i = 10
            return i + 15
        return
    ref = np.zeros(80 * 1024 * 1024, dtype=np.uint8)
    ray.get(large_arg.remote(ref))

@pytest.mark.xfail(cluster_not_supported, reason='cluster not supported')
def test_ray_get_task_args_deadlock(ray_start_cluster):
    if False:
        i = 10
        return i + 15
    cluster = ray_start_cluster
    object_size = int(6000000.0)
    num_objects = 10
    cluster.add_node(num_cpus=0, object_store_memory=4 * num_objects * object_size)
    cluster.wait_for_nodes()
    ray.init(address=cluster.address)
    cluster.add_node(num_cpus=1, object_store_memory=1.5 * num_objects * object_size)
    cluster.wait_for_nodes()

    @ray.remote
    def foo(*args):
        if False:
            while True:
                i = 10
        return

    @ray.remote
    def test_deadlock(get_args, task_args):
        if False:
            for i in range(10):
                print('nop')
        foo.remote(*task_args)
        ray.get(get_args)
    for i in range(5):
        start = time.time()
        get_args = [ray.put(np.zeros(object_size, dtype=np.uint8)) for _ in range(num_objects)]
        task_args = [ray.put(np.zeros(object_size, dtype=np.uint8)) for _ in range(num_objects)]
        ray.get(test_deadlock.remote(get_args, task_args))
        print(f'round {i} finished in {time.time() - start}')

def test_object_directory_basic(ray_start_cluster_with_resource):
    if False:
        for i in range(10):
            print('nop')
    (cluster, num_nodes) = ray_start_cluster_with_resource

    @ray.remote
    def task(x):
        if False:
            print('Hello World!')
        pass
    x_id = ray.put(np.zeros(1024 * 1024, dtype=np.uint8))
    ray.get(task.options(resources={str(3): 1}).remote(x_id), timeout=10)
    object_refs = []
    for _ in range(num_nodes):
        object_refs.append(ray.put(np.zeros(1024 * 1024, dtype=np.uint8)))
    ray.get([task.options(resources={str(i): 1}).remote(object_refs[i]) for i in range(num_nodes)])
    del object_refs

    @ray.remote
    class ObjectHolder:

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.x = ray.put(np.zeros(1024 * 1024, dtype=np.uint8))

        def get_obj(self):
            if False:
                return 10
            return self.x

        def ready(self):
            if False:
                for i in range(10):
                    print('nop')
            return True
    object_holders = [ObjectHolder.options(num_cpus=0.01, resources={str(i): 1}).remote() for i in range(num_nodes)]
    ray.get([o.ready.remote() for o in object_holders])
    object_refs = []
    for i in range(num_nodes):
        object_refs.append(object_holders[(i + 1) % num_nodes].get_obj.remote())
    ray.get([task.options(num_cpus=0.01, resources={str(i): 1}).remote(object_refs[i]) for i in range(num_nodes)])
    object_refs = []
    repeat = 10
    for _ in range(num_nodes):
        for _ in range(repeat):
            object_refs.append(ray.put(np.zeros(1024 * 1024, dtype=np.uint8)))
    tasks = []
    for i in range(num_nodes):
        for r in range(repeat):
            tasks.append(task.options(num_cpus=0.01, resources={str(i): 0.1}).remote(object_refs[i * r]))
    ray.get(tasks)
    object_refs = []
    for i in range(num_nodes):
        object_refs.append(object_holders[(i + 1) % num_nodes].get_obj.remote())
    tasks = []
    for i in range(num_nodes):
        for _ in range(10):
            tasks.append(task.options(num_cpus=0.01, resources={str(i): 0.1}).remote(object_refs[(i + 1) % num_nodes]))

def test_pull_bundle_deadlock(ray_start_cluster):
    if False:
        print('Hello World!')
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=0, _system_config={'max_direct_call_object_size': int(10000000.0)})
    ray.init(address=cluster.address)
    worker_node_1 = cluster.add_node(num_cpus=8, resources={'worker_node_1': 1})
    cluster.add_node(num_cpus=8, resources={'worker_node_2': 1}, object_store_memory=int(100000000.0 * 2 - 10))
    cluster.wait_for_nodes()

    @ray.remote(num_cpus=0)
    def get_node_id():
        if False:
            return 10
        return ray.get_runtime_context().get_node_id()
    worker_node_1_id = ray.get(get_node_id.options(resources={'worker_node_1': 0.1}).remote())
    worker_node_2_id = ray.get(get_node_id.options(resources={'worker_node_2': 0.1}).remote())
    object_a = ray.put(np.zeros(int(100000000.0), dtype=np.uint8))

    @ray.remote(scheduling_strategy=NodeAffinitySchedulingStrategy(worker_node_1_id, soft=True))
    def task_a_to_b(a):
        if False:
            while True:
                i = 10
        return np.zeros(int(100000000.0), dtype=np.uint8)
    object_b = task_a_to_b.remote(object_a)
    ray.wait([object_b], fetch_local=False)

    @ray.remote(scheduling_strategy=NodeAffinitySchedulingStrategy(worker_node_2_id, soft=False))
    def task_b_to_c(b):
        if False:
            while True:
                i = 10
        return 'c'
    object_c = task_b_to_c.remote(object_b)
    cluster.remove_node(worker_node_1, allow_graceful=False)
    assert ray.get(object_c) == 'c'

def test_object_directory_failure(ray_start_cluster):
    if False:
        i = 10
        return i + 15
    cluster = ray_start_cluster
    config = {'health_check_initial_delay_ms': 0, 'health_check_period_ms': 500, 'health_check_failure_threshold': 10, 'object_timeout_milliseconds': 200}
    cluster.add_node(_system_config=config)
    ray.init(address=cluster.address)
    num_nodes = 5
    for i in range(num_nodes):
        cluster.add_node(resources={str(i): 100})
    index_killing_node = num_nodes
    node_to_kill = cluster.add_node(resources={str(index_killing_node): 100}, object_store_memory=10 ** 9)

    @ray.remote
    class ObjectHolder:

        def __init__(self):
            if False:
                return 10
            self.x = ray.put(np.zeros(1024 * 1024, dtype=np.uint8))

        def get_obj(self):
            if False:
                print('Hello World!')
            return [self.x]

        def ready(self):
            if False:
                return 10
            return True
    oh = ObjectHolder.options(num_cpus=0.01, resources={str(index_killing_node): 1}).remote()
    obj = ray.get(oh.get_obj.remote())[0]

    @ray.remote
    def task(x):
        if False:
            i = 10
            return i + 15
        pass
    cluster.remove_node(node_to_kill, allow_graceful=False)
    tasks = []
    repeat = 3
    for i in range(num_nodes):
        for _ in range(repeat):
            tasks.append(task.options(resources={str(i): 1}).remote(obj))
    for t in tasks:
        with pytest.raises(ray.exceptions.RayTaskError):
            ray.get(t, timeout=10)

@pytest.mark.parametrize('ray_start_cluster_head', [{'num_cpus': 0, 'object_store_memory': 75 * 1024 * 1024, '_system_config': {'worker_lease_timeout_milliseconds': 0, 'object_manager_pull_timeout_ms': 20000, 'object_spilling_threshold': 1.0}}], indirect=True)
def test_maximize_concurrent_pull_race_condition(ray_start_cluster_head):
    if False:
        i = 10
        return i + 15
    cluster = ray_start_cluster_head
    cluster.add_node(num_cpus=8, object_store_memory=75 * 1024 * 1024)

    @ray.remote
    class RemoteObjectCreator:

        def put(self, i):
            if False:
                i = 10
                return i + 15
            return np.random.rand(i * 1024 * 1024)

        def idle(self):
            if False:
                return 10
            pass

    @ray.remote
    def f(x):
        if False:
            while True:
                i = 10
        print(f'timestamp={time.time()} pulled {len(x) * 8} bytes')
        time.sleep(1)
        return
    remote_obj_creator = RemoteObjectCreator.remote()
    remote_refs = [remote_obj_creator.put.remote(1) for _ in range(7)]
    print(remote_refs)
    ray.get(remote_obj_creator.idle.remote())
    local_refs = [ray.put(np.random.rand(1 * 1024 * 1024)) for _ in range(20)]
    remote_tasks = [f.remote(x) for x in local_refs]
    start = time.time()
    ray.get(remote_tasks)
    end = time.time()
    assert end - start < 20, 'Too much time spent in pulling objects, check the amount of time in retries'
if __name__ == '__main__':
    import sys
    import os
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))