import os
import sys
import signal
import threading
import json
from pathlib import Path
import ray
import numpy as np
import pytest
import psutil
import time
from ray._private.test_utils import SignalActor, wait_for_pid_to_exit, wait_for_condition, run_string_as_driver_nonblocking
SIGKILL = signal.SIGKILL if sys.platform != 'win32' else signal.SIGTERM

def test_worker_exit_after_parent_raylet_dies(ray_start_cluster):
    if False:
        for i in range(10):
            print('nop')
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=0)
    cluster.add_node(num_cpus=8, resources={'foo': 1})
    cluster.wait_for_nodes()
    ray.init(address=cluster.address)

    @ray.remote(resources={'foo': 1})
    class Actor:

        def get_worker_pid(self):
            if False:
                print('Hello World!')
            return os.getpid()

        def get_raylet_pid(self):
            if False:
                while True:
                    i = 10
            return int(os.environ['RAY_RAYLET_PID'])
    actor = Actor.remote()
    worker_pid = ray.get(actor.get_worker_pid.remote())
    raylet_pid = ray.get(actor.get_raylet_pid.remote())
    os.kill(raylet_pid, SIGKILL)
    os.waitpid(raylet_pid, 0)
    wait_for_pid_to_exit(raylet_pid)
    wait_for_pid_to_exit(worker_pid)

@pytest.mark.parametrize('ray_start_cluster_head', [{'num_cpus': 5, 'object_store_memory': 10 ** 8}], indirect=True)
def test_parallel_actor_fill_plasma_retry(ray_start_cluster_head):
    if False:
        for i in range(10):
            print('nop')

    @ray.remote
    class LargeMemoryActor:

        def some_expensive_task(self):
            if False:
                while True:
                    i = 10
            return np.zeros(10 ** 8 // 2, dtype=np.uint8)
    actors = [LargeMemoryActor.remote() for _ in range(5)]
    for _ in range(5):
        pending = [a.some_expensive_task.remote() for a in actors]
        while pending:
            ([done], pending) = ray.wait(pending, num_returns=1)

@pytest.mark.parametrize('ray_start_regular', [{'_system_config': {'task_retry_delay_ms': 500}}], indirect=True)
def test_async_actor_task_retries(ray_start_regular):
    if False:
        return 10
    signal = SignalActor.remote()

    @ray.remote
    class DyingActor:

        def __init__(self):
            if False:
                print('Hello World!')
            print('DyingActor init called')
            self.should_exit = False

        def set_should_exit(self):
            if False:
                for i in range(10):
                    print('nop')
            print('DyingActor.set_should_exit called')
            self.should_exit = True

        async def get(self, x, wait=False):
            print(f'DyingActor.get called with x={x}, wait={wait}')
            if self.should_exit:
                os._exit(0)
            if wait:
                await signal.wait.remote()
            return x
    dying = DyingActor.options(max_restarts=-1, max_task_retries=-1).remote()
    assert ray.get(dying.get.remote(1)) == 1
    ray.get(dying.set_should_exit.remote())
    assert ray.get(dying.get.remote(42)) == 42
    dying = DyingActor.options(max_restarts=-1, max_task_retries=-1).remote()
    ref_0 = dying.get.remote(0)
    assert ray.get(ref_0) == 0
    ref_1 = dying.get.remote(1, wait=True)
    for i in range(100):
        if ray.get(signal.cur_num_waiters.remote()) > 0:
            break
        time.sleep(0.1)
    assert ray.get(signal.cur_num_waiters.remote()) > 0
    ref_2 = dying.set_should_exit.remote()
    assert ray.get(ref_2) is None
    ref_3 = dying.get.remote(3)
    ray.get(signal.send.remote())
    assert ray.get(ref_1) == 1
    assert ray.get(ref_3) == 3

def test_actor_failure_async(ray_start_regular):
    if False:
        print('Hello World!')

    @ray.remote
    class A:

        def echo(self):
            if False:
                i = 10
                return i + 15
            pass

        def pid(self):
            if False:
                for i in range(10):
                    print('nop')
            return os.getpid()
    a = A.remote()
    rs = []

    def submit():
        if False:
            i = 10
            return i + 15
        for i in range(10000):
            r = a.echo.remote()
            r._on_completed(lambda x: 1)
            rs.append(r)
    t = threading.Thread(target=submit)
    pid = ray.get(a.pid.remote())
    t.start()
    from time import sleep
    sleep(0.1)
    os.kill(pid, SIGKILL)
    t.join()

@pytest.mark.parametrize('ray_start_regular', [{'_system_config': {'timeout_ms_task_wait_for_death_info': 100000000}}], indirect=True)
def test_actor_failure_async_2(ray_start_regular, tmp_path):
    if False:
        return 10
    p = tmp_path / 'a_pid'

    @ray.remote(max_restarts=1)
    class A:

        def __init__(self):
            if False:
                print('Hello World!')
            pid = os.getpid()
            if p.exists():
                p.write_text(str(pid))
                time.sleep(100000)
            else:
                p.write_text(str(pid))

        def pid(self):
            if False:
                i = 10
                return i + 15
            return os.getpid()
    a = A.remote()
    pid = ray.get(a.pid.remote())
    os.kill(int(pid), SIGKILL)

    def kill():
        if False:
            for i in range(10):
                print('nop')
        time.sleep(2)
        new_pid = int(p.read_text())
        while new_pid == pid:
            new_pid = int(p.read_text())
            time.sleep(1)
        os.kill(new_pid, SIGKILL)
    t = threading.Thread(target=kill)
    t.start()
    try:
        o = a.pid.remote()

        def new_task(_):
            if False:
                for i in range(10):
                    print('nop')
            print('new_task')
            a.pid.remote()
        o._on_completed(new_task)
        ray.get(o)
    except Exception:
        pass
    t.join()

@pytest.mark.parametrize('ray_start_regular', [{'_system_config': {'timeout_ms_task_wait_for_death_info': 100000000}}], indirect=True)
def test_actor_failure_async_3(ray_start_regular):
    if False:
        while True:
            i = 10

    @ray.remote(max_restarts=1)
    class A:

        def pid(self):
            if False:
                print('Hello World!')
            return os.getpid()
    a = A.remote()

    def new_task(_):
        if False:
            return 10
        print('new_task')
        a.pid.remote()
    t = a.pid.remote()
    t._on_completed(new_task)
    ray.kill(a)
    with pytest.raises(Exception):
        ray.get(t)

@pytest.mark.parametrize('ray_start_regular', [{'_system_config': {'timeout_ms_task_wait_for_death_info': 100000000}}], indirect=True)
def test_actor_failure_async_4(ray_start_regular, tmp_path):
    if False:
        print('Hello World!')
    from filelock import FileLock
    l_file = tmp_path / 'lock'
    l_lock = FileLock(l_file)
    l_lock.acquire()

    @ray.remote
    def f():
        if False:
            for i in range(10):
                print('nop')
        with FileLock(l_file):
            os.kill(os.getpid(), SIGKILL)

    @ray.remote(max_restarts=1)
    class A:

        def pid(self, x):
            if False:
                while True:
                    i = 10
            return os.getpid()
    a = A.remote()

    def new_task(_):
        if False:
            return 10
        print('new_task')
        a.pid.remote(None)
    t = a.pid.remote(f.remote())
    t._on_completed(new_task)
    ray.kill(a)
    l_lock.release()
    with pytest.raises(Exception):
        ray.get(t)

@pytest.mark.parametrize('ray_start_regular', [{'_system_config': {'timeout_ms_task_wait_for_death_info': 0, 'core_worker_internal_heartbeat_ms': 1000000}}], indirect=True)
def test_actor_failure_no_wait(ray_start_regular, tmp_path):
    if False:
        while True:
            i = 10
    p = tmp_path / 'a_pid'
    time.sleep(1)

    @ray.remote(max_restarts=1, max_task_retries=0)
    class A:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            pid = os.getpid()
            if p.exists():
                p.write_text(str(pid))
                time.sleep(100000)
            else:
                p.write_text(str(pid))

        def p(self):
            if False:
                i = 10
                return i + 15
            time.sleep(100000)

        def pid(self):
            if False:
                return 10
            return os.getpid()
    a = A.remote()
    pid = ray.get(a.pid.remote())
    t = a.p.remote()
    os.kill(int(pid), SIGKILL)
    with pytest.raises(ray.exceptions.RayActorError):
        ray.get(t)

@pytest.mark.skipif(sys.platform != 'linux', reason='Only works on linux.')
def test_no_worker_child_process_leaks(ray_start_cluster, tmp_path):
    if False:
        while True:
            i = 10
    '\n    Verify that processes created by Ray tasks and actors are\n    cleaned up after a Ctrl+C is sent to the driver. This is done by\n    creating an actor and task that each spawn a number of child\n    processes, sending a SIGINT to the driver process, and\n    verifying that all child processes are killed.\n\n    The driver script uses a temporary JSON file to communicate\n    the list of PIDs that are children of the Ray worker\n    processes.\n    '
    output_file_path = tmp_path / 'leaked_pids.json'
    ray_start_cluster.add_node()
    driver_script = f'\nimport ray\nimport json\nimport multiprocessing\nimport shutil\nimport time\nimport os\n\n@ray.remote\nclass Actor:\n    def create_leaked_child_process(self, num_to_leak):\n        print("Creating leaked process", os.getpid())\n\n        pids = []\n        for _ in range(num_to_leak):\n            proc = multiprocessing.Process(\n                target=time.sleep,\n                args=(1000,),\n                daemon=True,\n            )\n            proc.start()\n            pids.append(proc.pid)\n\n        return pids\n\n@ray.remote\ndef task():\n    print("Creating leaked process", os.getpid())\n    proc = multiprocessing.Process(\n        target=time.sleep,\n        args=(1000,),\n        daemon=True,\n    )\n    proc.start()\n\n    return proc.pid\n\nnum_to_leak_per_type = 10\n\nactor = Actor.remote()\nactor_leaked_pids = ray.get(actor.create_leaked_child_process.remote(\n    num_to_leak=num_to_leak_per_type,\n))\n\ntask_leaked_pids = ray.get([task.remote() for _ in range(num_to_leak_per_type)])\nleaked_pids = actor_leaked_pids + task_leaked_pids\n\nfinal_file = "{output_file_path}"\ntmp_file = final_file + ".tmp"\nwith open(tmp_file, "w") as f:\n    json.dump(leaked_pids, f)\nshutil.move(tmp_file, final_file)\n\nwhile True:\n    print(os.getpid())\n    time.sleep(1)\n    '
    driver_proc = run_string_as_driver_nonblocking(driver_script)
    wait_for_condition(condition_predictor=lambda : Path(output_file_path).exists(), timeout=30)
    with open(output_file_path, 'r') as f:
        pids = json.load(f)
    processes = [psutil.Process(pid) for pid in pids]
    assert all([proc.status() == psutil.STATUS_SLEEPING for proc in processes])
    driver_proc.send_signal(signal.SIGINT)
    wait_for_condition(condition_predictor=lambda : all([not proc.is_running() for proc in processes]), timeout=30)

@pytest.mark.skipif(sys.platform != 'linux', reason='Only works on linux.')
def test_worker_cleans_up_child_procs_on_raylet_death(ray_start_cluster, tmp_path):
    if False:
        print('Hello World!')
    '\n    CoreWorker kills its child processes if the raylet dies.\n    This test creates 20 leaked processes; 10 from a single actor task, and\n    10 from distinct non-actor tasks.\n\n    Once the raylet dies, the test verifies all leaked processes are cleaned up.\n    '
    output_file_path = tmp_path / 'leaked_pids.json'
    ray_start_cluster.add_node()
    driver_script = f'''\nimport ray\nimport json\nimport multiprocessing\nimport shutil\nimport time\nimport os\nimport setproctitle\n\ndef change_name_and_sleep(label: str, index: int) -> None:\n    proctitle = "child_proc_name_prefix_" + label + "_" + str(index)\n    setproctitle.setproctitle(proctitle)\n    time.sleep(1000)\n\ndef create_child_proc(label, index):\n    proc = multiprocessing.Process(\n        target=change_name_and_sleep,\n        args=(label, index,),\n        daemon=True,\n    )\n    proc.start()\n    return proc.pid\n\n@ray.remote\nclass LeakerActor:\n    def create_leaked_child_process(self, num_to_leak):\n        print("creating leaked process", os.getpid())\n\n        pids = []\n        for index in range(num_to_leak):\n            pid = create_child_proc("actor", index)\n            pids.append(pid)\n\n        return pids\n\n@ray.remote\ndef leaker_task(index):\n    print("Creating leaked process", os.getpid())\n    return create_child_proc("task", index)\n\nnum_to_leak_per_type = 10\nprint('starting actors')\nactor = LeakerActor.remote()\nactor_leaked_pids = ray.get(actor.create_leaked_child_process.remote(\n    num_to_leak=num_to_leak_per_type,\n))\n\ntask_leaked_pids = ray.get([\n    leaker_task.remote(index) for index in range(num_to_leak_per_type)\n])\nleaked_pids = actor_leaked_pids + task_leaked_pids\n\nfinal_file = "{output_file_path}"\ntmp_file = final_file + ".tmp"\nwith open(tmp_file, "w") as f:\n    json.dump(leaked_pids, f)\nshutil.move(tmp_file, final_file)\n\nwhile True:\n    print(os.getpid())\n    time.sleep(1)\n    '''
    print('Running string as driver')
    driver_proc = run_string_as_driver_nonblocking(driver_script)
    print('Waiting for child pids json')
    wait_for_condition(condition_predictor=lambda : Path(output_file_path).exists(), timeout=30)
    with open(output_file_path, 'r') as f:
        pids = json.load(f)
    processes = [psutil.Process(pid) for pid in pids]
    assert all([proc.status() == psutil.STATUS_SLEEPING for proc in processes])
    raylet_proc = [p for p in psutil.process_iter() if p.name() == 'raylet']
    assert len(raylet_proc) == 1
    raylet_proc = raylet_proc[0]
    raylet_proc.kill()
    raylet_proc.wait()
    print('Waiting for child procs to die')
    wait_for_condition(condition_predictor=lambda : all([not proc.is_running() for proc in processes]), timeout=30)
    driver_proc.kill()
if __name__ == '__main__':
    import pytest
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))