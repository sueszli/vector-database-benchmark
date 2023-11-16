import os
import sys
import grpc
import pytest
import requests
import ray
from ray import serve
from ray._private.test_utils import wait_for_condition
from ray.serve._private.constants import SERVE_DEFAULT_APP_NAME
from ray.serve._private.storage.kv_store import KVStoreError, RayInternalKVStore
from ray.serve.context import _get_global_client
from ray.tests.conftest import external_redis

@pytest.fixture(scope='function')
def serve_ha(external_redis, monkeypatch):
    if False:
        return 10
    monkeypatch.setenv('RAY_SERVE_KV_TIMEOUT_S', '1')
    address_info = ray.init(num_cpus=36, namespace='default_test_namespace', _metrics_export_port=9999, _system_config={'metrics_report_interval_ms': 1000, 'task_retry_delay_ms': 50})
    serve.start()
    yield (address_info, _get_global_client())
    ray.shutdown()

@pytest.mark.skipif(sys.platform == 'win32', reason="Failing on Windows, 'ForkedFunc' object has no attribute 'pid'")
def test_ray_internal_kv_timeout(serve_ha):
    if False:
        i = 10
        return i + 15
    kv1 = RayInternalKVStore()
    kv1.put('1', b'1')
    assert kv1.get('1') == b'1'
    ray.worker._global_node.kill_gcs_server()
    with pytest.raises(KVStoreError) as e:
        kv1.put('2', b'2')
    assert e.value.rpc_code in (grpc.StatusCode.UNAVAILABLE.value[0], grpc.StatusCode.DEADLINE_EXCEEDED.value[0])

@pytest.mark.skipif(sys.platform == 'win32', reason="Failing on Windows, 'ForkedFunc' object has no attribute 'pid'")
@pytest.mark.parametrize('use_handle', [False, True])
def test_controller_gcs_failure(serve_ha, use_handle):
    if False:
        return 10

    @serve.deployment
    def d(*args):
        if False:
            print('Hello World!')
        return f'{os.getpid()}'

    def call():
        if False:
            while True:
                i = 10
        if use_handle:
            handle = serve.get_app_handle(SERVE_DEFAULT_APP_NAME)
            ret = handle.remote().result()
        else:
            ret = requests.get('http://localhost:8000/d').text
        return ret
    serve.run(d.bind())
    pid = call()
    print('Kill GCS')
    ray.worker._global_node.kill_gcs_server()
    with pytest.raises(Exception):
        wait_for_condition(lambda : pid != call(), timeout=5, retry_interval_ms=1)
    print('Start GCS')
    ray.worker._global_node.start_gcs_server()
    with pytest.raises(Exception):
        wait_for_condition(lambda : call() != pid, timeout=4)
    serve.run(d.bind())
    for _ in range(10):
        assert pid != call()
    pid = call()
    print('Kill GCS')
    ray.worker._global_node.kill_gcs_server()
    with pytest.raises(KVStoreError):
        serve.run(d.options().bind())
    for _ in range(10):
        assert pid == call()
if __name__ == '__main__':
    sys.exit(pytest.main(['-v', '-s', '--forked', __file__]))