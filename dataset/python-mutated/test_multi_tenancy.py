import os
import sys
import time
import pytest
import numpy as np
import ray
from ray.core.generated import common_pb2
from ray.core.generated import node_manager_pb2, node_manager_pb2_grpc
from ray._private.test_utils import wait_for_condition, run_string_as_driver, run_string_as_driver_nonblocking
from ray._private.utils import init_grpc_channel

def get_workers():
    if False:
        for i in range(10):
            print('nop')
    raylet = ray.nodes()[0]
    raylet_address = '{}:{}'.format(raylet['NodeManagerAddress'], raylet['NodeManagerPort'])
    channel = init_grpc_channel(raylet_address)
    stub = node_manager_pb2_grpc.NodeManagerServiceStub(channel)
    return [worker for worker in stub.GetNodeStats(node_manager_pb2.GetNodeStatsRequest()).core_workers_stats if worker.worker_type != common_pb2.DRIVER]

def test_initial_workers(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    ray.init(num_cpus=1, include_dashboard=True)
    wait_for_condition(lambda : len(get_workers()) == 1)

def test_multi_drivers(shutdown_only):
    if False:
        i = 10
        return i + 15
    info = ray.init(num_cpus=10)
    driver_code = '\nimport os\nimport sys\nimport ray\n\n\nray.init(address="{}")\n\n@ray.remote\nclass Actor:\n    def get_pid(self):\n        return os.getpid()\n\n@ray.remote\ndef get_pid():\n    return os.getpid()\n\npid_objs = []\n# Submit some normal tasks and get the PIDs of workers which execute the tasks.\npid_objs = pid_objs + [get_pid.remote() for _ in range(2)]\n# Create some actors and get the PIDs of actors.\nactors = [Actor.remote() for _ in range(2)]\npid_objs = pid_objs + [actor.get_pid.remote() for actor in actors]\n\npids = set([ray.get(obj) for obj in pid_objs])\n# Write pids to stdout\nprint("PID:" + str.join(",", [str(_) for _ in pids]))\n\nray.shutdown()\n    '.format(info['address'])
    driver_count = 3
    processes = [run_string_as_driver_nonblocking(driver_code) for _ in range(driver_count)]
    outputs = []
    for p in processes:
        out = p.stdout.read().decode('ascii')
        err = p.stderr.read().decode('ascii')
        p.wait()
        if p.returncode != 0:
            print('Driver with PID {} returned error code {}'.format(p.pid, p.returncode))
            print('STDOUT:\n{}'.format(out))
            print('STDERR:\n{}'.format(err))
        outputs.append((p, out))
    all_worker_pids = set()
    for (p, out) in outputs:
        assert p.returncode == 0
        for line in out.splitlines():
            if line.startswith('PID:'):
                worker_pids = [int(_) for _ in line.split(':')[1].split(',')]
                assert len(worker_pids) > 0
                for worker_pid in worker_pids:
                    assert worker_pid not in all_worker_pids, ('Worker process with PID {} is shared' + ' by multiple drivers.').format(worker_pid)
                    all_worker_pids.add(worker_pid)

def test_runtime_env(shutdown_only):
    if False:
        print('Hello World!')
    ray.init(job_config=ray.job_config.JobConfig(runtime_env={'env_vars': {'foo1': 'bar1', 'foo2': 'bar2'}}))

    @ray.remote
    def get_env(key):
        if False:
            print('Hello World!')
        return os.environ.get(key)
    assert ray.get(get_env.remote('foo1')) == 'bar1'
    assert ray.get(get_env.remote('foo2')) == 'bar2'

def test_worker_capping_kill_idle_workers(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    ray.init(num_cpus=0)
    assert len(get_workers()) == 0

    @ray.remote(num_cpus=0)
    class Actor:

        def ping(self):
            if False:
                while True:
                    i = 10
            pass
    actor = Actor.remote()
    ray.get(actor.ping.remote())
    assert len(get_workers()) == 1

    @ray.remote(num_cpus=0)
    def foo():
        if False:
            return 10
        time.sleep(10)
    obj1 = foo.remote()
    wait_for_condition(lambda : len(get_workers()) == 2)
    obj2 = foo.remote()
    wait_for_condition(lambda : len(get_workers()) == 3)
    ray.get([obj1, obj2])
    wait_for_condition(lambda : len(get_workers()) == 1)

def test_worker_capping_run_many_small_tasks(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    ray.init(num_cpus=2)

    @ray.remote(num_cpus=0.5)
    def foo():
        if False:
            i = 10
            return i + 15
        time.sleep(5)
    obj_refs = [foo.remote() for _ in range(4)]
    wait_for_condition(lambda : len(get_workers()) == 4)
    ray.get(obj_refs)
    wait_for_condition(lambda : len(get_workers()) == 2)
    time.sleep(1)
    assert len(get_workers()) == 2

def test_worker_capping_run_chained_tasks(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    ray.init(num_cpus=2)

    @ray.remote(num_cpus=0.5)
    def foo(x):
        if False:
            while True:
                i = 10
        if x > 1:
            return ray.get(foo.remote(x - 1)) + x
        else:
            time.sleep(5)
            return x
    obj = foo.remote(4)
    wait_for_condition(lambda : len(get_workers()) == 4)
    ray.get(obj)
    wait_for_condition(lambda : len(get_workers()) == 2)
    time.sleep(1)
    assert len(get_workers()) == 2

def test_worker_registration_failure_after_driver_exit(shutdown_only):
    if False:
        return 10
    info = ray.init(num_cpus=1)
    driver_code = '\nimport ray\nimport time\n\n\nray.init(address="{}")\n\n@ray.remote\ndef foo():\n    pass\n\n[foo.remote() for _ in range(100)]\n\nray.shutdown()\n    '.format(info['address'])

    def worker_registered():
        if False:
            return 10
        return len(get_workers()) == 1
    wait_for_condition(worker_registered)
    before = 1
    run_string_as_driver(driver_code)
    time.sleep(2)
    wait_for_condition(lambda : len(get_workers()) <= before)

def test_not_killing_workers_that_own_objects(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    ray.init(num_cpus=1, _system_config={'kill_idle_workers_interval_ms': 10, 'worker_lease_timeout_milliseconds': 0})
    expected_num_workers = 6

    @ray.remote
    def nested(i):
        if False:
            i = 10
            return i + 15
        if i >= expected_num_workers - 1:
            return [ray.put(np.ones(1 * 1024 * 1024, dtype=np.uint8))]
        else:
            return [ray.put(np.ones(1 * 1024 * 1024, dtype=np.uint8))] + ray.get(nested.remote(i + 1))
    ref = ray.get(nested.remote(0))
    num_workers = len(get_workers())
    time.sleep(1)
    ref2 = ray.get(nested.remote(0))
    cur_num_workers = len(get_workers())
    assert abs(num_workers - cur_num_workers) < 2, (num_workers, cur_num_workers)
    assert len(ref2) == expected_num_workers
    assert len(ref) == expected_num_workers

def test_kill_idle_workers_that_are_behind_owned_workers(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    N = 4
    ray.init(num_cpus=1, _system_config={'kill_idle_workers_interval_ms': 10, 'worker_lease_timeout_milliseconds': 0})

    @ray.remote
    def nested(i):
        if False:
            while True:
                i = 10
        if i >= N * 2 - 1:
            return [ray.put(np.ones(1 * 1024 * 1024, dtype=np.uint8))]
        elif i >= N:
            return [ray.put(np.ones(1 * 1024 * 1024, dtype=np.uint8))] + ray.get(nested.remote(i + 1))
        else:
            return [1] + ray.get(nested.remote(i + 1))
    ref = ray.get(nested.remote(0))
    assert len(ref) == N * 2
    num_workers = len(get_workers())
    assert num_workers == N * 2
    wait_for_condition(lambda : len(get_workers()) == N)
if __name__ == '__main__':
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))