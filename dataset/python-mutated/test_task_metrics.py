from collections import defaultdict
import sys
import os
import copy
import pytest
import ray
from ray._private.metrics_agent import RAY_WORKER_TIMEOUT_S
from ray._private.test_utils import raw_metrics, run_string_as_driver, run_string_as_driver_nonblocking, wait_for_condition
METRIC_CONFIG = {'_system_config': {'metrics_report_interval_ms': 100}}
SLOW_METRIC_CONFIG = {'_system_config': {'metrics_report_interval_ms': 3000}}

def tasks_by_state(info) -> dict:
    if False:
        while True:
            i = 10
    return tasks_breakdown(info, lambda s: s.labels['State'])

def tasks_by_name_and_state(info) -> dict:
    if False:
        for i in range(10):
            print('nop')
    return tasks_breakdown(info, lambda s: (s.labels['Name'], s.labels['State']))

def tasks_by_all(info) -> dict:
    if False:
        i = 10
        return i + 15
    return tasks_breakdown(info, lambda s: (s.labels['Name'], s.labels['State'], s.labels['IsRetry']))

def tasks_breakdown(info, key_fn) -> dict:
    if False:
        i = 10
        return i + 15
    res = raw_metrics(info)
    if 'ray_tasks' in res:
        breakdown = defaultdict(int)
        for sample in res['ray_tasks']:
            key = key_fn(sample)
            breakdown[key] += sample.value
            if breakdown[key] == 0:
                del breakdown[key]
        print('Task label breakdown: {}'.format(breakdown))
        return breakdown
    else:
        return {}

def test_task_basic(shutdown_only):
    if False:
        return 10
    info = ray.init(num_cpus=2, **METRIC_CONFIG)
    driver = '\nimport ray\nimport time\n\nray.init("auto")\n\n@ray.remote\ndef f():\n    time.sleep(999)\na = [f.remote() for _ in range(10)]\nray.get(a)\n'
    proc = run_string_as_driver_nonblocking(driver)
    expected = {'RUNNING': 2.0, 'PENDING_NODE_ASSIGNMENT': 8.0}
    wait_for_condition(lambda : tasks_by_state(info) == expected, timeout=20, retry_interval_ms=500)
    assert tasks_by_name_and_state(info) == {('f', 'RUNNING'): 2.0, ('f', 'PENDING_NODE_ASSIGNMENT'): 8.0}
    proc.kill()

def test_task_job_ids(shutdown_only):
    if False:
        return 10
    info = ray.init(num_cpus=2, **METRIC_CONFIG)
    driver = '\nimport ray\nimport time\n\nray.init("auto")\n\n@ray.remote(num_cpus=0)\ndef f():\n    time.sleep(999)\na = [f.remote() for _ in range(1)]\nray.get(a)\n'
    procs = [run_string_as_driver_nonblocking(driver) for _ in range(3)]
    expected = {'RUNNING': 3.0}
    wait_for_condition(lambda : tasks_by_state(info) == expected, timeout=20, retry_interval_ms=500)
    metrics = raw_metrics(info)
    jobs_at_state = defaultdict(set)
    for sample in metrics['ray_tasks']:
        jobs_at_state[sample.labels['State']].add(sample.labels['JobId'])
    print('Jobs at state: {}'.format(jobs_at_state))
    assert len(jobs_at_state['RUNNING']) == 3, jobs_at_state
    for proc in procs:
        proc.kill()

def test_task_nested(shutdown_only):
    if False:
        print('Hello World!')
    info = ray.init(num_cpus=2, **METRIC_CONFIG)
    driver = '\nimport ray\nimport time\n\nray.init("auto")\n\n@ray.remote(num_cpus=0)\ndef wrapper():\n    @ray.remote\n    def f():\n        time.sleep(999)\n\n    ray.get([f.remote() for _ in range(10)])\n\nw = wrapper.remote()\nray.get(w)\n'
    proc = run_string_as_driver_nonblocking(driver)
    expected = {'RUNNING': 2.0, 'RUNNING_IN_RAY_GET': 1.0, 'PENDING_NODE_ASSIGNMENT': 8.0}
    wait_for_condition(lambda : tasks_by_state(info) == expected, timeout=20, retry_interval_ms=2000)
    assert tasks_by_name_and_state(info) == {('wrapper', 'RUNNING_IN_RAY_GET'): 1.0, ('f', 'RUNNING'): 2.0, ('f', 'PENDING_NODE_ASSIGNMENT'): 8.0}
    proc.kill()

def test_task_nested_wait(shutdown_only):
    if False:
        i = 10
        return i + 15
    info = ray.init(num_cpus=2, **METRIC_CONFIG)
    driver = '\nimport ray\nimport time\n\nray.init("auto")\n\n@ray.remote(num_cpus=0)\ndef wrapper():\n    @ray.remote\n    def f():\n        time.sleep(999)\n\n    ray.wait([f.remote() for _ in range(10)])\n\nw = wrapper.remote()\nray.get(w)\n'
    proc = run_string_as_driver_nonblocking(driver)
    expected = {'RUNNING': 2.0, 'RUNNING_IN_RAY_WAIT': 1.0, 'PENDING_NODE_ASSIGNMENT': 8.0}
    wait_for_condition(lambda : tasks_by_state(info) == expected, timeout=20, retry_interval_ms=2000)
    assert tasks_by_name_and_state(info) == {('wrapper', 'RUNNING_IN_RAY_WAIT'): 1.0, ('f', 'RUNNING'): 2.0, ('f', 'PENDING_NODE_ASSIGNMENT'): 8.0}
    proc.kill()

def test_task_wait_on_deps(shutdown_only):
    if False:
        return 10
    info = ray.init(num_cpus=2, **METRIC_CONFIG)
    driver = '\nimport ray\nimport time\n\nray.init("auto")\n\n@ray.remote\ndef f():\n    time.sleep(999)\n\n@ray.remote\ndef g(x):\n    time.sleep(999)\n\nx = f.remote()\na = [g.remote(x) for _ in range(5)]\nray.get(a)\n'
    proc = run_string_as_driver_nonblocking(driver)
    expected = {'RUNNING': 1.0, 'PENDING_ARGS_AVAIL': 5.0}
    wait_for_condition(lambda : tasks_by_state(info) == expected, timeout=20, retry_interval_ms=500)
    assert tasks_by_name_and_state(info) == {('f', 'RUNNING'): 1.0, ('g', 'PENDING_ARGS_AVAIL'): 5.0}
    proc.kill()

def test_actor_tasks_queued(shutdown_only):
    if False:
        print('Hello World!')
    info = ray.init(num_cpus=2, **METRIC_CONFIG)
    driver = '\nimport ray\nimport time\n\nray.init("auto")\n\n@ray.remote\nclass F:\n    def f(self):\n        time.sleep(999)\n\n    def g(self):\n        pass\n\na = F.remote()\n[a.g.remote() for _ in range(10)]\n[a.f.remote() for _ in range(1)]  # Further tasks should be blocked on this one.\nz = [a.g.remote() for _ in range(9)]\nray.get(z)\n'
    proc = run_string_as_driver_nonblocking(driver)
    expected = {'RUNNING': 1.0, 'SUBMITTED_TO_WORKER': 9.0, 'FINISHED': 11.0}
    wait_for_condition(lambda : tasks_by_state(info) == expected, timeout=20, retry_interval_ms=500)
    assert tasks_by_name_and_state(info) == {('F.__init__', 'FINISHED'): 1.0, ('F.g', 'FINISHED'): 10.0, ('F.f', 'RUNNING'): 1.0, ('F.g', 'SUBMITTED_TO_WORKER'): 9.0}
    proc.kill()

def test_task_finish(shutdown_only):
    if False:
        print('Hello World!')
    info = ray.init(num_cpus=2, **METRIC_CONFIG)
    driver = '\nimport ray\nimport time\n\nray.init("auto")\n\n@ray.remote\ndef f():\n    return "ok"\n\n@ray.remote\ndef g():\n    assert False\n\nf.remote()\ng.remote()\ntime.sleep(999)\n'
    proc = run_string_as_driver_nonblocking(driver)
    expected = {'FAILED': 1.0, 'FINISHED': 1.0}
    wait_for_condition(lambda : tasks_by_state(info) == expected, timeout=20, retry_interval_ms=500)
    assert tasks_by_name_and_state(info) == {('g', 'FAILED'): 1.0, ('f', 'FINISHED'): 1.0}
    proc.kill()

def test_task_retry(shutdown_only):
    if False:
        i = 10
        return i + 15
    info = ray.init(num_cpus=2, **METRIC_CONFIG)
    driver = '\nimport ray\nimport time\n\nray.init("auto")\n\n@ray.remote\ndef sleep():\n    time.sleep(999)\n\n@ray.remote\nclass Phaser:\n    def __init__(self):\n        self.i = 0\n\n    def inc(self):\n        self.i += 1\n        if self.i < 3:\n            raise ValueError("First two tries will fail")\n\nphaser = Phaser.remote()\n\n@ray.remote(retry_exceptions=True, max_retries=3)\ndef f():\n    ray.get(phaser.inc.remote())\n    ray.get(sleep.remote())\n\nf.remote()\ntime.sleep(999)\n'
    proc = run_string_as_driver_nonblocking(driver)
    expected = {('sleep', 'RUNNING', '0'): 1.0, ('f', 'FAILED', '0'): 1.0, ('f', 'FAILED', '1'): 1.0, ('f', 'RUNNING_IN_RAY_GET', '1'): 1.0, ('Phaser.__init__', 'FINISHED', '0'): 1.0, ('Phaser.inc', 'FINISHED', '0'): 1.0, ('Phaser.inc', 'FAILED', '0'): 2.0}
    wait_for_condition(lambda : tasks_by_all(info) == expected, timeout=20, retry_interval_ms=500)
    proc.kill()

@pytest.mark.skipif(sys.platform == 'win32', reason='Flaky on Windows. Timing out.')
def test_actor_task_retry(shutdown_only):
    if False:
        print('Hello World!')
    info = ray.init(num_cpus=2, **METRIC_CONFIG)
    driver = '\nimport ray\nimport os\nimport time\n\nray.init("auto")\n\n@ray.remote\nclass Phaser:\n    def __init__(self):\n        self.i = 0\n\n    def inc(self):\n        self.i += 1\n        if self.i < 3:\n            raise ValueError("First two tries will fail")\n\nphaser = Phaser.remote()\n\n@ray.remote(max_restarts=10, max_task_retries=10)\nclass F:\n    def f(self):\n        try:\n            ray.get(phaser.inc.remote())\n        except Exception:\n            print("RESTART")\n            os._exit(1)\n\nf = F.remote()\nray.get(f.f.remote())\ntime.sleep(999)\n'
    proc = run_string_as_driver_nonblocking(driver)
    expected = {('F.__init__', 'FINISHED', '0'): 1.0, ('F.f', 'FAILED', '0'): 1.0, ('F.f', 'FAILED', '1'): 1.0, ('F.f', 'FINISHED', '1'): 1.0, ('Phaser.__init__', 'FINISHED', '0'): 1.0, ('Phaser.inc', 'FINISHED', '0'): 1.0}
    wait_for_condition(lambda : tasks_by_all(info) == expected, timeout=20, retry_interval_ms=500)
    proc.kill()

@pytest.mark.skipif(sys.platform == 'win32', reason='Flaky on Windows.')
def test_task_failure(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    info = ray.init(num_cpus=2, **METRIC_CONFIG)
    driver = '\nimport ray\nimport time\nimport os\n\nray.init("auto")\n\n@ray.remote(max_retries=0)\ndef f():\n    print("RUNNING FAILING TASK")\n    os._exit(1)\n\n@ray.remote\ndef g():\n    assert False\n\nf.remote()\ng.remote()\ntime.sleep(999)\n'
    proc = run_string_as_driver_nonblocking(driver)
    expected = {'FAILED': 2.0}
    wait_for_condition(lambda : tasks_by_state(info) == expected, timeout=20, retry_interval_ms=500)
    proc.kill()

def test_concurrent_actor_tasks(shutdown_only):
    if False:
        while True:
            i = 10
    info = ray.init(num_cpus=2, **METRIC_CONFIG)
    driver = '\nimport ray\nimport asyncio\n\nray.init("auto")\n\n@ray.remote(max_concurrency=30)\nclass A:\n    async def f(self):\n        await asyncio.sleep(300)\n\na = A.remote()\nray.get([a.f.remote() for _ in range(40)])\n'
    proc = run_string_as_driver_nonblocking(driver)
    expected = {'RUNNING': 30.0, 'SUBMITTED_TO_WORKER': 10.0, 'FINISHED': 1.0}
    wait_for_condition(lambda : tasks_by_state(info) == expected, timeout=20, retry_interval_ms=500)
    proc.kill()

@pytest.mark.skipif(sys.platform == 'win32', reason='Flaky on Windows.')
def test_metrics_export_now(shutdown_only):
    if False:
        return 10
    info = ray.init(num_cpus=2, **SLOW_METRIC_CONFIG)
    driver = '\nimport ray\nimport time\n\nray.init("auto")\n\n@ray.remote\ndef f():\n    pass\na = [f.remote() for _ in range(10)]\nray.get(a)\n'
    for i in range(10):
        print('Run job', i)
        run_string_as_driver(driver)
        tasks_by_state(info)
    expected = {'FINISHED': 100.0}
    wait_for_condition(lambda : tasks_by_state(info) == expected, timeout=20, retry_interval_ms=500)

@pytest.mark.skipif(sys.platform == 'darwin', reason='Flaky on macos')
def test_pull_manager_stats(shutdown_only):
    if False:
        return 10
    info = ray.init(num_cpus=2, object_store_memory=100000000, **METRIC_CONFIG)
    driver = '\nimport ray\nimport time\nimport numpy as np\n\nray.init("auto")\n\n# Spill a lot of 10MiB objects. The object store is 100MiB, so pull manager will\n# only be able to pull ~9 total into memory at once, including running tasks.\nbuf = []\nfor _ in range(100):\n    buf.append(ray.put(np.ones(10 * 1024 * 1024, dtype=np.uint8)))\n\n@ray.remote\ndef f(x):\n    time.sleep(999)\n\nray.get([f.remote(x) for x in buf])'
    proc = run_string_as_driver_nonblocking(driver)

    def close_to_expected(stats):
        if False:
            print('Hello World!')
        assert len(stats) == 3, stats
        assert stats['RUNNING'] == 2, stats
        assert 7 <= stats['PENDING_NODE_ASSIGNMENT'] <= 17, stats
        assert 81 <= stats['PENDING_OBJ_STORE_MEM_AVAIL'] <= 91, stats
        assert sum(stats.values()) == 100, stats
        return True
    wait_for_condition(lambda : close_to_expected(tasks_by_state(info)), timeout=20, retry_interval_ms=500)
    proc.kill()

@pytest.mark.skipif(sys.platform == 'win32', reason='Flaky on Windows.')
def test_stale_view_cleanup_when_job_exits(monkeypatch, shutdown_only):
    if False:
        i = 10
        return i + 15
    with monkeypatch.context() as m:
        m.setenv(RAY_WORKER_TIMEOUT_S, 5)
        info = ray.init(num_cpus=2, **METRIC_CONFIG)
        print(info)
        driver = '\nimport ray\nimport time\nimport numpy as np\n\nray.init("auto")\n\n@ray.remote\ndef g():\n    time.sleep(999)\n\nray.get(g.remote())\n    '
        proc = run_string_as_driver_nonblocking(driver)
        expected = {'RUNNING': 1.0}
        wait_for_condition(lambda : tasks_by_state(info) == expected, timeout=20, retry_interval_ms=500)
        proc.kill()
        print('Killing a driver.')
        expected = {}
        wait_for_condition(lambda : tasks_by_state(info) == expected, timeout=20, retry_interval_ms=500)

@pytest.mark.skipif(sys.platform == 'win32', reason='Flaky on Windows. Timing out.')
def test_metrics_batch(shutdown_only):
    if False:
        return 10
    'Verify metrics_report_batch_size works correctly without data loss.'
    config_copy = copy.deepcopy(METRIC_CONFIG)
    config_copy['_system_config'].update({'metrics_report_batch_size': 1})
    info = ray.init(num_cpus=2, **config_copy)
    driver = '\nimport ray\nimport os\nimport time\n\nray.init("auto")\n\n@ray.remote\nclass Phaser:\n    def __init__(self):\n        self.i = 0\n\n    def inc(self):\n        self.i += 1\n        if self.i < 3:\n            raise ValueError("First two tries will fail")\n\nphaser = Phaser.remote()\n\n@ray.remote(max_restarts=10, max_task_retries=10)\nclass F:\n    def f(self):\n        try:\n            ray.get(phaser.inc.remote())\n        except Exception:\n            print("RESTART")\n            os._exit(1)\n\nf = F.remote()\nray.get(f.f.remote())\ntime.sleep(999)\n'
    proc = run_string_as_driver_nonblocking(driver)
    expected = {('F.__init__', 'FINISHED', '0'): 1.0, ('F.f', 'FAILED', '0'): 1.0, ('F.f', 'FAILED', '1'): 1.0, ('F.f', 'FINISHED', '1'): 1.0, ('Phaser.__init__', 'FINISHED', '0'): 1.0, ('Phaser.inc', 'FINISHED', '0'): 1.0}
    wait_for_condition(lambda : tasks_by_all(info) == expected, timeout=20, retry_interval_ms=500)
    proc.kill()
if __name__ == '__main__':
    import sys
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))