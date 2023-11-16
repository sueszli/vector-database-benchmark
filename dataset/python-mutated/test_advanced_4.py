import mock
import subprocess
import sys
import pytest
import ray
from ray._raylet import check_health
from ray._private.test_utils import Semaphore, client_test_enabled, wait_for_condition, get_gcs_memory_used
from ray.experimental.internal_kv import _internal_kv_list

@pytest.fixture
def shutdown_only_with_initialization_check():
    if False:
        for i in range(10):
            print('nop')
    yield None
    ray.shutdown()
    assert not ray.is_initialized()

def test_initialized(shutdown_only_with_initialization_check):
    if False:
        for i in range(10):
            print('nop')
    assert not ray.is_initialized()
    ray.init(num_cpus=0)
    assert ray.is_initialized()

def test_initialized_local_mode(shutdown_only_with_initialization_check):
    if False:
        print('Hello World!')
    assert not ray.is_initialized()
    ray.init(num_cpus=0, local_mode=True)
    assert ray.is_initialized()

def test_ray_start_and_stop():
    if False:
        print('Hello World!')
    for i in range(10):
        subprocess.check_call(['ray', 'start', '--head'])
        subprocess.check_call(['ray', 'stop'])

def test_ray_memory(shutdown_only):
    if False:
        while True:
            i = 10
    ray.init(num_cpus=1)
    subprocess.check_call(['ray', 'memory'])

def test_jemalloc_env_var_propagate():
    if False:
        return 10
    'Test `propagate_jemalloc_env_var`'
    gcs_ptype = ray._private.ray_constants.PROCESS_TYPE_GCS_SERVER
    '\n    If the shared library path is not specified,\n    it should return an empty dict.\n    '
    expected = {}
    actual = ray._private.services.propagate_jemalloc_env_var(jemalloc_path='', jemalloc_conf='', jemalloc_comps=[], process_type=gcs_ptype)
    assert actual == expected
    actual = ray._private.services.propagate_jemalloc_env_var(jemalloc_path=None, jemalloc_conf='a,b,c', jemalloc_comps=[ray._private.ray_constants.PROCESS_TYPE_GCS_SERVER], process_type=gcs_ptype)
    assert actual == expected
    '\n    When the shared library is specified\n    '
    library_path = '/abc'
    expected = {'LD_PRELOAD': library_path, 'RAY_LD_PRELOAD': '1'}
    actual = ray._private.services.propagate_jemalloc_env_var(jemalloc_path=library_path, jemalloc_conf='', jemalloc_comps=[ray._private.ray_constants.PROCESS_TYPE_GCS_SERVER], process_type=gcs_ptype)
    assert actual == expected
    with pytest.raises(AssertionError):
        ray._private.services.propagate_jemalloc_env_var(jemalloc_path=library_path, jemalloc_conf='', jemalloc_comps='ray._private.ray_constants.PROCESS_TYPE_GCS_SERVER,', process_type=gcs_ptype)
    expected = {}
    actual = ray._private.services.propagate_jemalloc_env_var(jemalloc_path=library_path, jemalloc_conf='', jemalloc_comps=[ray._private.ray_constants.PROCESS_TYPE_RAYLET], process_type=gcs_ptype)
    '\n    When the malloc config is specified\n    '
    library_path = '/abc'
    malloc_conf = 'a,b,c'
    expected = {'LD_PRELOAD': library_path, 'MALLOC_CONF': malloc_conf, 'RAY_LD_PRELOAD': '1'}
    actual = ray._private.services.propagate_jemalloc_env_var(jemalloc_path=library_path, jemalloc_conf=malloc_conf, jemalloc_comps=[ray._private.ray_constants.PROCESS_TYPE_GCS_SERVER], process_type=gcs_ptype)
    assert actual == expected

def test_check_health(shutdown_only):
    if False:
        print('Hello World!')
    assert not check_health('127.0.0.1:8888')
    assert not check_health('ip:address:with:colon:name:8265')
    with pytest.raises(ValueError):
        check_health('bad_address_no_port')
    conn = ray.init()
    addr = conn.address_info['address']
    assert check_health(addr)

def test_check_health_version_check(shutdown_only):
    if False:
        return 10
    with mock.patch('ray.__version__', 'FOO-VERSION'):
        conn = ray.init()
        addr = conn.address_info['address']
        assert check_health(addr, skip_version_check=True)
        with pytest.raises(RuntimeError):
            check_health(addr)

def test_back_pressure(shutdown_only_with_initialization_check):
    if False:
        return 10
    ray.init()
    signal_actor = Semaphore.options(max_pending_calls=10).remote(value=0)
    try:
        for i in range(10):
            signal_actor.acquire.remote()
    except ray.exceptions.PendingCallsLimitExceeded:
        assert False
    with pytest.raises(ray.exceptions.PendingCallsLimitExceeded):
        signal_actor.acquire.remote()

    @ray.remote
    def release(signal_actor):
        if False:
            i = 10
            return i + 15
        ray.get(signal_actor.release.remote())
        return 1
    for i in range(10):
        ray.get(release.remote(signal_actor))
    try:
        signal_actor.acquire.remote()
    except ray.exceptions.PendingCallsLimitExceeded:
        assert False
    ray.shutdown()

def test_local_mode_deadlock(shutdown_only_with_initialization_check):
    if False:
        while True:
            i = 10
    ray.init(local_mode=True)

    @ray.remote
    class Foo:

        def __init__(self):
            if False:
                print('Hello World!')
            pass

        def ping_actor(self, actor):
            if False:
                return 10
            actor.ping.remote()
            return 3

    @ray.remote
    class Bar:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            pass

        def ping(self):
            if False:
                for i in range(10):
                    print('nop')
            return 1
    foo = Foo.remote()
    bar = Bar.remote()
    assert ray.get(foo.ping_actor.remote(bar)) == 3

def function_entry_num(job_id):
    if False:
        i = 10
        return i + 15
    from ray._private.ray_constants import KV_NAMESPACE_FUNCTION_TABLE
    return len(_internal_kv_list(b'IsolatedExports:' + job_id, namespace=KV_NAMESPACE_FUNCTION_TABLE)) + len(_internal_kv_list(b'RemoteFunction:' + job_id, namespace=KV_NAMESPACE_FUNCTION_TABLE)) + len(_internal_kv_list(b'ActorClass:' + job_id, namespace=KV_NAMESPACE_FUNCTION_TABLE)) + len(_internal_kv_list(b'FunctionsToRun:' + job_id, namespace=KV_NAMESPACE_FUNCTION_TABLE))

@pytest.mark.skipif(client_test_enabled(), reason="client api doesn't support namespace right now.")
def test_function_table_gc(call_ray_start):
    if False:
        for i in range(10):
            print('nop')
    'This test tries to verify that function table is cleaned up\n    after job exits.\n    '

    def f():
        if False:
            print('Hello World!')
        data = '0' * 1024 * 1024

        @ray.remote
        def r():
            if False:
                return 10
            nonlocal data

            @ray.remote
            class Actor:
                pass
        return r.remote()
    ray.init(address='auto', namespace='b')
    ray.get([f() for _ in range(500)])
    if sys.platform != 'win32':
        assert get_gcs_memory_used() > 500 * 1024 * 1024
    job_id = ray._private.worker.global_worker.current_job_id.hex().encode()
    assert function_entry_num(job_id) > 0
    ray.shutdown()
    ray.init(address='auto', namespace='a')
    wait_for_condition(lambda : function_entry_num(job_id) == 0, timeout=30)

@pytest.mark.skipif(client_test_enabled(), reason="client api doesn't support namespace right now.")
def test_function_table_gc_actor(call_ray_start):
    if False:
        for i in range(10):
            print('nop')
    "If there is a detached actor, the table won't be cleaned up."
    ray.init(address='auto', namespace='a')

    @ray.remote
    class Actor:

        def ready(self):
            if False:
                while True:
                    i = 10
            return
    a = Actor.options(lifetime='detached', name='a').remote()
    ray.get(a.ready.remote())
    job_id = ray._private.worker.global_worker.current_job_id.hex().encode()
    ray.shutdown()
    ray.init(address='auto', namespace='b')
    with pytest.raises(Exception):
        wait_for_condition(lambda : function_entry_num(job_id) == 0)
    a = ray.get_actor('a', namespace='a')
    ray.kill(a)
    wait_for_condition(lambda : function_entry_num(job_id) == 0)
    a = Actor.remote()
    ray.get(a.ready.remote())
    job_id = ray._private.worker.global_worker.current_job_id.hex().encode()
    ray.shutdown()
    ray.init(address='auto', namespace='c')
    wait_for_condition(lambda : function_entry_num(job_id) == 0)
if __name__ == '__main__':
    import os
    import pytest
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))