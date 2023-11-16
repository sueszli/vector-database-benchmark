import ray
from ray.util.state import list_workers
from ray._private.test_utils import get_load_metrics_report, run_string_as_driver, run_string_as_driver_nonblocking, wait_for_condition, get_resource_usage
import pytest
import os
from ray.util.state import list_objects
import subprocess
from ray._private.utils import get_num_cpus
import time
import sys

def test_infeasible_tasks(ray_start_cluster):
    if False:
        return 10
    cluster = ray_start_cluster

    @ray.remote
    def f():
        if False:
            i = 10
            return i + 15
        return
    cluster.add_node(resources={str(0): 100})
    ray.init(address=cluster.address)
    x_id = f._remote(args=[], kwargs={}, resources={str(1): 1})
    cluster.add_node(resources={str(1): 100})
    ray.get(x_id)
    driver_script = '\nimport ray\n\nray.init(address="{}")\n\n@ray.remote(resources={})\ndef f():\n{}pass  # This is a weird hack to insert some blank space.\n\nf.remote()\n'.format(cluster.address, '{str(2): 1}', '    ')
    run_string_as_driver(driver_script)
    cluster.add_node(resources={str(2): 100})
    ray.get([f._remote(args=[], kwargs={}, resources={str(i): 1}) for i in range(3)])

@pytest.mark.parametrize('call_ray_start', ['ray start --head'], indirect=True)
def test_kill_driver_clears_backlog(call_ray_start):
    if False:
        return 10
    driver = '\nimport ray\n\n@ray.remote\ndef f():\n    import time\n    time.sleep(300)\n\nrefs = [f.remote() for _ in range(10000)]\n\nray.get(refs)\n  '
    proc = run_string_as_driver_nonblocking(driver)
    ctx = ray.init(address=call_ray_start)

    def get_backlog_and_pending():
        if False:
            print('Hello World!')
        resources_batch = get_resource_usage(gcs_address=ctx.address_info['gcs_address'])
        backlog = resources_batch.resource_load_by_shape.resource_demands[0].backlog_size if resources_batch.resource_load_by_shape.resource_demands else 0
        pending = 0
        demands = get_load_metrics_report(webui_url=ctx.address_info['webui_url'])['resourceDemand']
        for demand in demands:
            (resource_dict, amount) = demand
            if 'CPU' in resource_dict:
                pending = amount
        return (pending, backlog)

    def check_backlog(expect_backlog) -> bool:
        if False:
            print('Hello World!')
        (pending, backlog) = get_backlog_and_pending()
        if expect_backlog:
            return pending > 0 and backlog > 0
        else:
            return pending == 0 and backlog == 0
    wait_for_condition(check_backlog, timeout=10, retry_interval_ms=1000, expect_backlog=True)
    os.kill(proc.pid, 9)
    wait_for_condition(check_backlog, timeout=10, retry_interval_ms=1000, expect_backlog=False)

def get_infeasible_queued(ray_ctx):
    if False:
        i = 10
        return i + 15
    resources_batch = get_resource_usage(gcs_address=ray_ctx.address_info['gcs_address'])
    infeasible_queued = resources_batch.resource_load_by_shape.resource_demands[0].num_infeasible_requests_queued if len(resources_batch.resource_load_by_shape.resource_demands) > 0 and hasattr(resources_batch.resource_load_by_shape.resource_demands[0], 'num_infeasible_requests_queued') else 0
    return infeasible_queued

def check_infeasible(expect_infeasible, ray_ctx) -> bool:
    if False:
        for i in range(10):
            print('nop')
    infeasible_queued = get_infeasible_queued(ray_ctx)
    if expect_infeasible:
        return infeasible_queued > 0
    else:
        return infeasible_queued == 0

@pytest.mark.parametrize('call_ray_start', ['ray start --head'], indirect=True)
def test_kill_driver_clears_infeasible(call_ray_start):
    if False:
        print('Hello World!')
    driver = '\nimport ray\n\n@ray.remote\ndef f():\n    pass\n\nray.get(f.options(num_cpus=99999999).remote())\n  '
    proc = run_string_as_driver_nonblocking(driver)
    ctx = ray.init(address=call_ray_start)
    wait_for_condition(check_infeasible, timeout=10, retry_interval_ms=1000, expect_infeasible=True, ray_ctx=ctx)
    os.kill(proc.pid, 9)
    wait_for_condition(check_infeasible, timeout=10, retry_interval_ms=1000, expect_infeasible=False, ray_ctx=ctx)

def test_kill_driver_keep_infeasible_detached_actor(ray_start_cluster):
    if False:
        while True:
            i = 10
    cluster = ray_start_cluster
    address = cluster.address
    cluster.add_node(num_cpus=1)
    driver_script = '\nimport ray\n\n@ray.remote\nclass A:\n    def fn(self):\n        pass\n\nray.init(address="{}", namespace="test_det")\n\nray.get(A.options(num_cpus=123, name="det", lifetime="detached").remote())\n'.format(cluster.address)
    proc = run_string_as_driver_nonblocking(driver_script)
    ctx = ray.init(address=address, namespace='test_det')
    wait_for_condition(check_infeasible, timeout=10, retry_interval_ms=1000, expect_infeasible=True, ray_ctx=ctx)
    os.kill(proc.pid, 9)
    cluster.add_node(num_cpus=200)
    det_actor = ray.get_actor('det')
    ray.get(det_actor.fn.remote())

@pytest.mark.parametrize('call_ray_start', ['ray start --head'], indirect=True)
def test_reference_global_import_does_not_leak_worker_upon_driver_exit(call_ray_start):
    if False:
        return 10
    driver = '\nimport ray\nimport numpy as np\nimport tensorflow\n\n@ray.remote(max_retries=0)\ndef leak_repro(obj):\n    tensorflow\n    return []\n\nrefs = []\nfor i in range(100_000):\n    refs.append(leak_repro.remote(i))\n\nray.get(refs)\n  '
    try:
        run_string_as_driver(driver)
    except subprocess.CalledProcessError:
        pass
    ray.init(address=call_ray_start)

    def no_object_leaks():
        if False:
            return 10
        objects = list_objects(_explain=True, timeout=3)
        return len(objects) == 0
    wait_for_condition(no_object_leaks, timeout=10, retry_interval_ms=1000)

@pytest.mark.skipif(sys.platform == 'win32', reason='subprocess command only works for unix')
@pytest.mark.parametrize('call_ray_start', ['ray start --head --system-config={"enable_worker_prestart":true}'], indirect=True)
def test_worker_prestart_on_node_manager_start(call_ray_start, shutdown_only):
    if False:
        for i in range(10):
            print('nop')

    def num_idle_workers(count):
        if False:
            print('Hello World!')
        result = subprocess.check_output('ps aux | grep ray::IDLE | grep -v grep', shell=True)
        return len(result.splitlines()) == count
    wait_for_condition(num_idle_workers, count=get_num_cpus())
    with ray.init():
        for _ in range(5):
            workers = list_workers(filters=[('worker_type', '=', 'WORKER')])
            assert len(workers) == get_num_cpus(), workers
            time.sleep(1)

@pytest.mark.parametrize('call_ray_start', ['ray start --head'], indirect=True)
def test_jobs_prestart_worker_once(call_ray_start, shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    with ray.init():
        workers = list_workers(filters=[('worker_type', '=', 'WORKER')])
        assert len(workers) == get_num_cpus(), workers
    with ray.init():
        for _ in range(5):
            workers = list_workers(filters=[('worker_type', '=', 'WORKER')])
            assert len(workers) == get_num_cpus(), workers
            time.sleep(1)
if __name__ == '__main__':
    import sys
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))