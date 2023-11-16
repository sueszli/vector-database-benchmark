import os
import signal
import sys
import time
import pytest
import ray
import ray._private.ray_constants as ray_constants
from ray._private.test_utils import Semaphore, get_other_nodes
from ray.cluster_utils import Cluster, cluster_not_supported
SIGKILL = signal.SIGKILL if sys.platform != 'win32' else signal.SIGTERM

@pytest.mark.xfail(cluster_not_supported, reason='cluster not supported')
@pytest.fixture(params=[(1, 4), (4, 4)])
def ray_start_workers_separate_multinode(request):
    if False:
        for i in range(10):
            print('nop')
    num_nodes = request.param[0]
    num_initial_workers = request.param[1]
    cluster = Cluster()
    for _ in range(num_nodes):
        cluster.add_node(num_cpus=num_initial_workers, resources={'custom': num_initial_workers})
    ray.init(address=cluster.address)
    yield (num_nodes, num_initial_workers)
    ray.shutdown()
    cluster.shutdown()

def test_worker_failed(ray_start_workers_separate_multinode):
    if False:
        return 10
    (num_nodes, num_initial_workers) = ray_start_workers_separate_multinode
    block_worker = Semaphore.remote(0)
    block_driver = Semaphore.remote(0)
    ray.get([block_worker.locked.remote(), block_driver.locked.remote()])

    @ray.remote(num_cpus=1, resources={'custom': 1})
    def get_pids():
        if False:
            print('Hello World!')
        ray.get(block_driver.release.remote())
        ray.get(block_worker.acquire.remote())
        return os.getpid()
    total_num_workers = num_nodes * num_initial_workers
    pid_refs = [get_pids.remote() for _ in range(total_num_workers)]
    ray.get([block_driver.acquire.remote() for _ in range(total_num_workers)])
    ray.get([block_worker.release.remote() for _ in range(total_num_workers)])
    pids = set(ray.get(pid_refs))

    @ray.remote
    def f(x):
        if False:
            while True:
                i = 10
        time.sleep(0.5)
        return x
    object_refs = [f.remote(i) for i in range(num_initial_workers * num_nodes)]
    object_refs += [f.remote(object_ref) for object_ref in object_refs]
    time.sleep(0.1)
    for pid in pids:
        try:
            os.kill(pid, SIGKILL)
        except OSError:
            pass
        time.sleep(0.1)
    for object_ref in object_refs:
        try:
            ray.get(object_ref)
        except (ray.exceptions.RayTaskError, ray.exceptions.WorkerCrashedError):
            pass

def _test_component_failed(cluster, component_type):
    if False:
        print('Hello World!')
    'Kill a component on all worker nodes and check workload succeeds.'

    @ray.remote
    def f(x):
        if False:
            print('Hello World!')
        time.sleep(0.01)
        return x

    @ray.remote
    def g(*xs):
        if False:
            return 10
        time.sleep(0.01)
        return 1
    time.sleep(0.1)
    worker_nodes = get_other_nodes(cluster)
    assert len(worker_nodes) > 0
    for node in worker_nodes:
        process = node.all_processes[component_type][0].process
        x = 1
        for _ in range(1000):
            x = f.remote(x)
        xs = [g.remote(1)]
        for _ in range(100):
            xs.append(g.remote(*xs))
            xs.append(g.remote(1))
        process.terminate()
        time.sleep(1)
        process.kill()
        process.wait()
        assert not process.poll() is None
        ray.get(x)
        ray.get(xs)

def check_components_alive(cluster, component_type, check_component_alive):
    if False:
        for i in range(10):
            print('nop')
    'Check that a given component type is alive on all worker nodes.'
    worker_nodes = get_other_nodes(cluster)
    assert len(worker_nodes) > 0
    for node in worker_nodes:
        process = node.all_processes[component_type][0].process
        if check_component_alive:
            assert process.poll() is None
        else:
            print('waiting for ' + component_type + ' with PID ' + str(process.pid) + 'to terminate')
            process.wait()
            print('done waiting for ' + component_type + ' with PID ' + str(process.pid) + 'to terminate')
            assert not process.poll() is None

@pytest.mark.parametrize('ray_start_cluster', [{'num_cpus': 8, 'num_nodes': 4, '_system_config': {'health_check_initial_delay_ms': 0, 'health_check_failure_threshold': 10}}], indirect=True)
def test_raylet_failed(ray_start_cluster):
    if False:
        print('Hello World!')
    cluster = ray_start_cluster
    _test_component_failed(cluster, ray_constants.PROCESS_TYPE_RAYLET)
if __name__ == '__main__':
    import pytest
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))