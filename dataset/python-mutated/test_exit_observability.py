import asyncio
import os
import signal
import sys
import pytest
from ray._private.state_api_test_utils import verify_failed_task
import ray
from ray._private.test_utils import run_string_as_driver, wait_for_condition
from ray.util.state import list_workers, list_nodes, list_tasks
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

def get_worker_by_pid(pid, detail=True):
    if False:
        print('Hello World!')
    for w in list_workers(detail=detail):
        if w['pid'] == pid:
            return w
    assert False

@pytest.mark.skipif(sys.platform == 'win32', reason='Failed on Windows')
def test_worker_exit_system_error(ray_start_cluster):
    if False:
        print('Hello World!')
    '\n    SYSTEM_ERROR\n    - (tested) Failure from the connection E.g., core worker dead.\n    - (tested) Unexpected exception or exit with exit_code !=0 on core worker.\n    - (tested for owner node death) Node died. Currently worker failure detection\n        upon node death is not detected by Ray. TODO(sang): Fix it.\n    - (Cannot test) Direct call failure.\n    '
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=4)
    ray.init(address=cluster.address)
    cluster.add_node(num_cpus=1, resources={'worker': 1})

    @ray.remote
    class Actor:

        def pid(self):
            if False:
                for i in range(10):
                    print('nop')
            import os
            return os.getpid()

        def exit(self, exit_code):
            if False:
                i = 10
                return i + 15
            sys.exit(exit_code)
    '\n    Failure from the connection\n    '
    a = Actor.remote()
    pid = ray.get(a.pid.remote())
    print(pid)
    os.kill(pid, signal.SIGKILL)

    def verify_connection_failure():
        if False:
            for i in range(10):
                print('nop')
        worker = get_worker_by_pid(pid)
        print(worker)
        type = worker['exit_type']
        detail = worker['exit_detail']
        return type == 'SYSTEM_ERROR' and 'OOM' in detail
    wait_for_condition(verify_connection_failure)
    '\n    Unexpected exception or exit with exit_code !=0 on core worker.\n    '
    a = Actor.remote()
    pid = ray.get(a.pid.remote())
    with pytest.raises(ray.exceptions.RayActorError):
        ray.get(a.exit.options(name='exit').remote(4))

    def verify_exit_failure():
        if False:
            while True:
                i = 10
        worker = get_worker_by_pid(pid)
        type = worker['exit_type']
        detail = worker['exit_detail']
        assert type == 'SYSTEM_ERROR' and 'exit code 4' in detail
        return verify_failed_task(name='exit', error_type='ACTOR_DIED', error_message='exit code 4')
    wait_for_condition(verify_exit_failure)

@pytest.mark.skipif(sys.platform == 'win32', reason='Failed on Windows')
def test_worker_exit_intended_user_exit(ray_start_cluster):
    if False:
        print('Hello World!')
    '\n    INTENDED_USER_EXIT\n    - (tested) Shutdown driver\n    - (tested) exit_actor\n    - (tested) exit(0)\n    - (tested) Actor kill request\n    - (tested) Task cancel request\n    '
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=4)
    ray.init(address=cluster.address)
    cluster.add_node(num_cpus=1, resources={'worker': 1})
    driver = '\nimport ray\nimport os\nray.init(address="{address}")\nprint(os.getpid())\nray.shutdown()\n'.format(address=cluster.address)
    a = run_string_as_driver(driver)
    driver_pid = int(a.strip().split('\n')[-1].strip())

    def verify_worker_exit_by_shutdown():
        if False:
            i = 10
            return i + 15
        worker = get_worker_by_pid(driver_pid)
        type = worker['exit_type']
        detail = worker['exit_detail']
        assert type == 'INTENDED_USER_EXIT' and 'ray.shutdown()' in detail
        return True
    wait_for_condition(verify_worker_exit_by_shutdown)

    @ray.remote
    class A:

        def pid(self):
            if False:
                return 10
            return os.getpid()

        def exit(self):
            if False:
                while True:
                    i = 10
            ray.actor.exit_actor()

        def exit_with_exit_code(self):
            if False:
                return 10
            sys.exit(0)

        def sleep_forever(self):
            if False:
                print('Hello World!')
            import time
            time.sleep(999999)
    a = A.remote()
    pid = ray.get(a.pid.remote())
    with pytest.raises(ray.exceptions.RayActorError, match='exit_actor'):
        ray.get(a.exit.options(name='exit').remote())

    def verify_worker_exit_actor():
        if False:
            return 10
        worker = get_worker_by_pid(pid)
        type = worker['exit_type']
        detail = worker['exit_detail']
        assert type == 'INTENDED_USER_EXIT' and 'exit_actor' in detail
        t = list_tasks(filters=[('name', '=', 'exit')])[0]
        assert t['state'] == 'FINISHED'
        return True
    wait_for_condition(verify_worker_exit_actor)
    a = A.remote()
    pid = ray.get(a.pid.remote())
    with pytest.raises(ray.exceptions.RayActorError, match='exit code 0'):
        ray.get(a.exit_with_exit_code.options(name='exit_with_exit_code').remote())

    def verify_exit_code_0():
        if False:
            print('Hello World!')
        worker = get_worker_by_pid(pid)
        type = worker['exit_type']
        detail = worker['exit_detail']
        assert type == 'INTENDED_USER_EXIT' and 'exit code 0' in detail
        t = list_tasks(filters=[('name', '=', 'exit_with_exit_code')])[0]
        assert t['state'] == 'FINISHED'
        return True
    wait_for_condition(verify_exit_code_0)
    a = A.remote()
    pid = ray.get(a.pid.remote())
    ray.kill(a)
    with pytest.raises(ray.exceptions.RayActorError, match='ray.kill'):
        ray.get(a.sleep_forever.options(name='sleep_forever').remote())

    def verify_exit_by_ray_kill():
        if False:
            while True:
                i = 10
        worker = get_worker_by_pid(pid)
        type = worker['exit_type']
        detail = worker['exit_detail']
        assert type == 'INTENDED_SYSTEM_EXIT' and 'ray.kill' in detail
        return verify_failed_task(name='sleep_forever', error_type='ACTOR_DIED', error_message='ray.kill')
    wait_for_condition(verify_exit_by_ray_kill)

    @ray.remote
    class PidDB:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.pid = None

        def record_pid(self, pid):
            if False:
                return 10
            self.pid = pid

        def get_pid(self):
            if False:
                return 10
            return self.pid
    p = PidDB.remote()

    @ray.remote
    def f():
        if False:
            i = 10
            return i + 15
        ray.get(p.record_pid.remote(os.getpid()))
        import time
        time.sleep(300)
    t = f.options(name='cancel-f').remote()
    wait_for_condition(lambda : ray.get(p.get_pid.remote()) is not None, timeout=300)
    ray.cancel(t, force=True)
    pid = ray.get(p.get_pid.remote())

    def verify_exit_by_ray_cancel():
        if False:
            while True:
                i = 10
        worker = get_worker_by_pid(pid)
        type = worker['exit_type']
        detail = worker['exit_detail']
        assert type == 'INTENDED_USER_EXIT' and 'ray.cancel' in detail
        return verify_failed_task(name='cancel-f', error_type='WORKER_DIED', error_message='Socket closed')
    wait_for_condition(verify_exit_by_ray_cancel)

@pytest.mark.skipif(sys.platform == 'win32', reason='Failed on Windows')
def test_worker_exit_intended_system_exit_and_user_error(ray_start_cluster):
    if False:
        print('Hello World!')
    '\n    INTENDED_SYSTEM_EXIT\n    - (not tested, hard to test) Unused resource removed\n    - (tested) Pg removed\n    - (tested) Idle\n    USER_ERROR\n    - (tested) Actor init failed\n    '
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=1)
    ray.init(address=cluster.address)

    @ray.remote
    def f():
        if False:
            return 10
        return ray.get(g.remote())

    @ray.remote
    def g():
        if False:
            print('Hello World!')
        return os.getpid()
    pid = ray.get(f.remote())

    def verify_exit_by_idle_timeout():
        if False:
            i = 10
            return i + 15
        worker = get_worker_by_pid(pid)
        type = worker['exit_type']
        detail = worker['exit_detail']
        return type == 'INTENDED_SYSTEM_EXIT' and 'it was idle' in detail
    wait_for_condition(verify_exit_by_idle_timeout)

    @ray.remote(num_cpus=1)
    class A:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.sleeping = False

        async def getpid(self):
            while not self.sleeping:
                await asyncio.sleep(0.1)
            return os.getpid()

        async def sleep(self):
            self.sleeping = True
            await asyncio.sleep(9999)
    pg = ray.util.placement_group(bundles=[{'CPU': 1}])
    a = A.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)).remote()
    a.sleep.options(name='sleep').remote()
    pid = ray.get(a.getpid.remote())
    ray.util.remove_placement_group(pg)

    def verify_exit_by_pg_removed():
        if False:
            print('Hello World!')
        worker = get_worker_by_pid(pid)
        type = worker['exit_type']
        detail = worker['exit_detail']
        assert verify_failed_task(name='sleep', error_type='ACTOR_DIED', error_message=['INTENDED_SYSTEM_EXIT', 'placement group was removed'])
        return type == 'INTENDED_SYSTEM_EXIT' and 'placement group was removed' in detail
    wait_for_condition(verify_exit_by_pg_removed)

    @ray.remote
    class PidDB:

        def __init__(self):
            if False:
                return 10
            self.pid = None

        def record_pid(self, pid):
            if False:
                for i in range(10):
                    print('nop')
            self.pid = pid

        def get_pid(self):
            if False:
                i = 10
                return i + 15
            return self.pid
    p = PidDB.remote()

    @ray.remote
    class FaultyActor:

        def __init__(self):
            if False:
                while True:
                    i = 10
            p.record_pid.remote(os.getpid())
            raise Exception('exception in the initialization method')

        def ready(self):
            if False:
                for i in range(10):
                    print('nop')
            pass
    a = FaultyActor.remote()
    wait_for_condition(lambda : ray.get(p.get_pid.remote()) is not None)
    pid = ray.get(p.get_pid.remote())

    def verify_exit_by_actor_init_failure():
        if False:
            return 10
        worker = get_worker_by_pid(pid)
        type = worker['exit_type']
        detail = worker['exit_detail']
        assert type == 'USER_ERROR' and 'exception in the initialization method' in detail
        return verify_failed_task(name='FaultyActor.__init__', error_type='TASK_EXECUTION_EXCEPTION', error_message='exception in the initialization method')
    wait_for_condition(verify_exit_by_actor_init_failure)

@pytest.mark.skipif(sys.platform == 'win32', reason="Failed on Windows because sigkill doesn't work on Windows")
def test_worker_start_end_time(shutdown_only):
    if False:
        i = 10
        return i + 15
    ray.init(num_cpus=1)

    @ray.remote
    class Worker:

        def ready(self):
            if False:
                return 10
            return os.getpid()
    worker = Worker.remote()
    pid = ray.get(worker.ready.remote())

    def verify():
        if False:
            while True:
                i = 10
        workers = list_workers(detail=True, filters=[('pid', '=', pid)])[0]
        print(workers)
        assert workers['start_time_ms'] > 0
        assert workers['end_time_ms'] == 0
        return True
    wait_for_condition(verify)
    ray.kill(worker)

    def verify():
        if False:
            return 10
        workers = list_workers(detail=True, filters=[('pid', '=', pid)])[0]
        assert workers['start_time_ms'] > 0
        assert workers['end_time_ms'] > 0
        return True
    wait_for_condition(verify)
    worker = Worker.remote()
    pid = ray.get(worker.ready.remote())
    os.kill(pid, signal.SIGKILL)

    def verify():
        if False:
            for i in range(10):
                print('nop')
        workers = list_workers(detail=True, filters=[('pid', '=', pid)])[0]
        assert workers['start_time_ms'] > 0
        assert workers['end_time_ms'] > 0
        return True
    wait_for_condition(verify)

def test_node_start_end_time(ray_start_cluster):
    if False:
        i = 10
        return i + 15
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=0)
    nodes = list_nodes(detail=True)
    head_node_id = nodes[0]['node_id']
    worker_node = cluster.add_node(num_cpus=0)
    nodes = list_nodes(detail=True)
    worker_node_data = list(filter(lambda x: x['node_id'] != head_node_id and x['state'] == 'ALIVE', nodes))[0]
    assert worker_node_data['start_time_ms'] > 0
    assert worker_node_data['end_time_ms'] == 0
    cluster.remove_node(worker_node, allow_graceful=True)
    nodes = list_nodes(detail=True)
    worker_node_data = list(filter(lambda x: x['node_id'] != head_node_id and x['state'] == 'DEAD', nodes))[0]
    assert worker_node_data['start_time_ms'] > 0
    assert worker_node_data['end_time_ms'] > 0
    worker_node = cluster.add_node(num_cpus=0)
    nodes = list_nodes(detail=True)
    worker_node_data = list(filter(lambda x: x['node_id'] != head_node_id and x['state'] == 'ALIVE', nodes))[0]
    assert worker_node_data['start_time_ms'] > 0
    assert worker_node_data['end_time_ms'] == 0
    cluster.remove_node(worker_node, allow_graceful=False)
    nodes = list_nodes(detail=True)
    worker_node_data = list(filter(lambda x: x['node_id'] != head_node_id and x['state'] == 'DEAD', nodes))[0]
    assert worker_node_data['start_time_ms'] > 0
    assert worker_node_data['end_time_ms'] > 0
if __name__ == '__main__':
    import pytest
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))