import gc
import logging
import os
import sys
import time
import subprocess
from unittest.mock import Mock, patch
import unittest
import pytest
import ray
import ray.cluster_utils
from ray._private.test_utils import run_string_as_driver, wait_for_pid_to_exit, client_test_enabled
from ray._private.resource_spec import HEAD_NODE_RESOURCE_NAME
logger = logging.getLogger(__name__)

def test_background_tasks_with_max_calls(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    ray.init(num_cpus=2, _system_config={'worker_cap_initial_backoff_delay_ms': 0})
    num_tasks = 3 if sys.platform == 'win32' else 10

    @ray.remote
    def g():
        if False:
            while True:
                i = 10
        time.sleep(0.1)
        return 0

    @ray.remote(max_calls=1, max_retries=0)
    def f():
        if False:
            print('Hello World!')
        return [g.remote()]
    nested = ray.get([f.remote() for _ in range(num_tasks)])
    ray.get([x[0] for x in nested])

    @ray.remote(max_calls=1, max_retries=0)
    def f():
        if False:
            while True:
                i = 10
        return (os.getpid(), g.remote())
    nested = ray.get([f.remote() for _ in range(num_tasks)])
    while nested:
        (pid, g_id) = nested.pop(0)
        assert ray.get(g_id) == 0
        del g_id
        gc.collect()
        wait_for_pid_to_exit(pid)

def test_actor_killing(shutdown_only):
    if False:
        return 10
    import ray
    ray.init(num_cpus=1)

    @ray.remote(num_cpus=1)
    class Actor:

        def foo(self):
            if False:
                while True:
                    i = 10
            return None
    worker_1 = Actor.remote()
    ray.kill(worker_1)
    worker_2 = Actor.remote()
    assert ray.get(worker_2.foo.remote()) is None
    ray.kill(worker_2)
    worker_1 = Actor.options(max_restarts=1, max_task_retries=-1).remote()
    ray.kill(worker_1, no_restart=False)
    assert ray.get(worker_1.foo.remote()) is None
    ray.kill(worker_1, no_restart=False)
    worker_2 = Actor.remote()
    assert ray.get(worker_2.foo.remote()) is None

def test_internal_kv(ray_start_regular):
    if False:
        print('Hello World!')
    import ray.experimental.internal_kv as kv
    assert kv._internal_kv_get('k1') is None
    assert kv._internal_kv_put('k1', 'v1') is False
    assert kv._internal_kv_put('k1', 'v1') is True
    assert kv._internal_kv_get('k1') == b'v1'
    assert kv._internal_kv_exists(b'k1') is True
    assert kv._internal_kv_exists(b'k2') is False
    assert kv._internal_kv_get('k1', namespace='n') is None
    assert kv._internal_kv_put('k1', 'v1', namespace='n') is False
    assert kv._internal_kv_put('k1', 'v1', namespace='n') is True
    assert kv._internal_kv_put('k1', 'v2', True, namespace='n') is True
    assert kv._internal_kv_get('k1', namespace='n') == b'v2'
    assert kv._internal_kv_del('k1') == 1
    assert kv._internal_kv_del('k1') == 0
    assert kv._internal_kv_get('k1') is None
    assert kv._internal_kv_put('k2', 'v2', namespace='n') is False
    assert kv._internal_kv_put('k3', 'v3', namespace='n') is False
    assert set(kv._internal_kv_list('k', namespace='n')) == {b'k1', b'k2', b'k3'}
    assert kv._internal_kv_del('k', del_by_prefix=True, namespace='n') == 3
    assert kv._internal_kv_del('x', del_by_prefix=True, namespace='n') == 0
    assert kv._internal_kv_get('k1', namespace='n') is None
    assert kv._internal_kv_get('k2', namespace='n') is None
    assert kv._internal_kv_get('k3', namespace='n') is None
    with pytest.raises(ray.exceptions.RaySystemError):
        kv._internal_kv_put('@namespace_', 'x', True)
    with pytest.raises(ray.exceptions.RaySystemError):
        kv._internal_kv_get('@namespace_', namespace='n')
    with pytest.raises(ray.exceptions.RaySystemError):
        kv._internal_kv_del('@namespace_def', namespace='n')
    with pytest.raises(ray.exceptions.RaySystemError):
        kv._internal_kv_list('@namespace_abc', namespace='n')

def test_exit_logging():
    if False:
        return 10
    log = run_string_as_driver('\nimport ray\n\n@ray.remote\nclass A:\n    def pid(self):\n        import os\n        return os.getpid()\n\n\na = A.remote()\nray.get(a.pid.remote())\n    ')
    assert 'Traceback' not in log

def test_worker_sys_path_contains_driver_script_directory(tmp_path, monkeypatch):
    if False:
        print('Hello World!')
    package_folder = tmp_path / 'package'
    package_folder.mkdir()
    init_file = tmp_path / 'package' / '__init__.py'
    init_file.write_text('')
    module1_file = tmp_path / 'package' / 'module1.py'
    module1_file.write_text(f"\nimport sys\nimport ray\nray.init()\n\n@ray.remote\ndef sys_path():\n    return sys.path\n\nassert r'{str(tmp_path / 'package')}' in ray.get(sys_path.remote())\n")
    subprocess.check_call(['python', str(module1_file)])
    module2_file = tmp_path / 'package' / 'module2.py'
    module2_file.write_text(f"\nimport sys\nimport ray\nray.init()\n\n@ray.remote\ndef sys_path():\n    return sys.path\n\nassert r'{str(tmp_path / 'package')}' not in ray.get(sys_path.remote())\n")
    monkeypatch.chdir(str(tmp_path))
    subprocess.check_call(['python', '-m', 'package.module2'])

@pytest.mark.skipif(sys.platform == 'win32', reason='Currently fails on Windows.')
def test_worker_kv_calls(monkeypatch, shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setenv('TEST_RAY_COLLECT_KV_FREQUENCY', '1')
    ray.init()

    @ray.remote
    def get_kv_metrics():
        if False:
            while True:
                i = 10
        from time import sleep
        sleep(2)
        return ray._private.utils._CALLED_FREQ
    freqs = ray.get(get_kv_metrics.remote())
    "\n    b'cluster' b'CLUSTER_METADATA'\n    b'tracing' b'tracing_startup_hook'\n    b'fun' b'IsolatedExports:01000000:\x00\x00\x00\x00\x00\x00\x00\x01'\n    "
    assert freqs['internal_kv_get'] == 3

@pytest.mark.skipif(sys.platform == 'win32', reason='Fails on Windows.')
@pytest.mark.parametrize('root_process_no_site', [0, 1])
@pytest.mark.parametrize('root_process_no_user_site', [0, 1])
def test_site_flag_inherited(shutdown_only, monkeypatch, root_process_no_site, root_process_no_user_site):
    if False:
        while True:
            i = 10
    monkeypatch.setenv('PYTHONPATH', ':'.join(sys.path))

    @ray.remote
    def get_flags():
        if False:
            for i in range(10):
                print('nop')
        return (sys.flags.no_site, sys.flags.no_user_site)
    with patch.multiple('ray._private.services', _no_site=Mock(return_value=root_process_no_site), _no_user_site=Mock(return_value=root_process_no_user_site)):
        ray.init()
        (worker_process_no_site, worker_process_no_user_site) = ray.get(get_flags.remote())
        assert worker_process_no_site == root_process_no_site
        assert worker_process_no_user_site == root_process_no_user_site

@pytest.mark.parametrize('preload', [True, False])
def test_preload_workers(ray_start_cluster, preload):
    if False:
        for i in range(10):
            print('nop')
    '\n    Verify preload_python_modules actually preloads modules in the Ray workers.\n    Also verify that it does not crash if a non-existent module is provided.\n    '
    cluster = ray_start_cluster
    expect_succeed_imports = ['html.parser', 'webbrowser']
    expect_fail_imports = ['fake_module_expect_ModuleNotFoundError']
    if preload:
        cluster.add_node(_system_config={'preload_python_modules': [*expect_succeed_imports, *expect_fail_imports]})
    else:
        cluster.add_node()

    @ray.remote(num_cpus=0)
    class Latch:
        """
        Used to ensure two separate worker processes.
        """

        def __init__(self, count):
            if False:
                while True:
                    i = 10
            self.count = count

        def decr(self):
            if False:
                print('Hello World!')
            self.count -= 1

        def is_ready(self):
            if False:
                return 10
            return self.count <= 0

    def wait_latch(latch):
        if False:
            i = 10
            return i + 15
        latch.decr.remote()
        while not ray.get(latch.is_ready.remote()):
            time.sleep(0.01)

    def assert_correct_imports():
        if False:
            for i in range(10):
                print('nop')
        import sys
        imported_modules = set(sys.modules.keys())
        if preload:
            for expected_import in expect_succeed_imports:
                assert expected_import in imported_modules, f'Expected {expected_import} to be in {imported_modules}'
            for unexpected_import in expect_fail_imports:
                assert unexpected_import not in imported_modules, f'Expected {unexpected_import} to not be in {imported_modules}'
        else:
            for unexpected_import in expect_succeed_imports:
                assert unexpected_import not in imported_modules, f'Expected {unexpected_import} to not be in {imported_modules}'

    @ray.remote(num_cpus=0)
    class Actor:

        def verify_imports(self, latch):
            if False:
                print('Hello World!')
            wait_latch(latch)
            assert_correct_imports()

    @ray.remote(num_cpus=0)
    def verify_imports(latch):
        if False:
            for i in range(10):
                print('nop')
        wait_latch(latch)
        assert_correct_imports()
    latch = Latch.remote(2)
    actor = Actor.remote()
    futures = [verify_imports.remote(latch), actor.verify_imports.remote(latch)]
    ray.get(futures)

@pytest.mark.skipif(client_test_enabled(), reason='only server mode')
def test_gcs_port_env(shutdown_only):
    if False:
        while True:
            i = 10
    try:
        with unittest.mock.patch.dict(os.environ):
            os.environ['RAY_GCS_SERVER_PORT'] = '12345'
            ray.init()
    except RuntimeError:
        pass

def test_head_node_resource(ray_start_cluster):
    if False:
        while True:
            i = 10
    'Test that the special head node resource is set.'
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=1)
    ray.init(address=cluster.address)
    assert ray.cluster_resources()[HEAD_NODE_RESOURCE_NAME] == 1
    cluster.add_node(num_cpus=1)
    assert ray.cluster_resources()[HEAD_NODE_RESOURCE_NAME] == 1

def test_head_node_resource_ray_init(shutdown_only):
    if False:
        while True:
            i = 10
    ray.init()
    assert ray.cluster_resources()[HEAD_NODE_RESOURCE_NAME] == 1

@pytest.mark.skipif(client_test_enabled(), reason='grpc deadlock with ray client')
def test_head_node_resource_ray_start(call_ray_start):
    if False:
        while True:
            i = 10
    ray.init(address=call_ray_start)
    assert ray.cluster_resources()[HEAD_NODE_RESOURCE_NAME] == 1
if __name__ == '__main__':
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))