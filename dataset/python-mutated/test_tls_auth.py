import logging
import os
import sys
import subprocess
import pytest
from ray._private.test_utils import run_string_as_driver
logger = logging.getLogger(__name__)

def build_env():
    if False:
        i = 10
        return i + 15
    env = os.environ.copy()
    if sys.platform == 'win32' and 'SYSTEMROOT' not in env:
        env['SYSTEMROOT'] = 'C:\\Windows'
    return env

@pytest.mark.skipif(sys.platform == 'darwin', reason="Cryptography (TLS dependency) doesn't install in Mac build pipeline")
@pytest.mark.parametrize('use_tls', [True], indirect=True)
def test_init_with_tls(use_tls):
    if False:
        while True:
            i = 10
    run_string_as_driver('\nimport ray\ntry:\n    ray.init()\nfinally:\n    ray.shutdown()\n    ', env=build_env())

@pytest.mark.skipif(sys.platform == 'darwin', reason="Cryptography (TLS dependency) doesn't install in Mac build pipeline")
@pytest.mark.parametrize('use_tls', [True], indirect=True)
def test_put_get_with_tls(use_tls):
    if False:
        for i in range(10):
            print('nop')
    run_string_as_driver('\nimport ray\nray.init()\ntry:\n    for i in range(100):\n        value_before = i * 10**6\n        object_ref = ray.put(value_before)\n        value_after = ray.get(object_ref)\n        assert value_before == value_after\n\n    for i in range(100):\n        value_before = i * 10**6 * 1.0\n        object_ref = ray.put(value_before)\n        value_after = ray.get(object_ref)\n        assert value_before == value_after\n\n    for i in range(100):\n        value_before = "h" * i\n        object_ref = ray.put(value_before)\n        value_after = ray.get(object_ref)\n        assert value_before == value_after\n\n    for i in range(100):\n        value_before = [1] * i\n        object_ref = ray.put(value_before)\n        value_after = ray.get(object_ref)\n        assert value_before == value_after\nfinally:\n    ray.shutdown()\n    ', env=build_env())

@pytest.mark.skipif(sys.platform == 'darwin', reason="Cryptography (TLS dependency) doesn't install in Mac build pipeline")
@pytest.mark.parametrize('use_tls', [True], indirect=True, scope='module')
def test_submit_with_tls(use_tls):
    if False:
        i = 10
        return i + 15
    run_string_as_driver('\nimport ray\nray.init(num_cpus=2, num_gpus=1, resources={"Custom": 1})\n\n@ray.remote\ndef f(n):\n    return list(range(n))\n\nid1, id2, id3 = f._remote(args=[3], num_returns=3)\nassert ray.get([id1, id2, id3]) == [0, 1, 2]\n\n@ray.remote\nclass Actor:\n    def __init__(self, x, y=0):\n        self.x = x\n        self.y = y\n\n    def method(self, a, b=0):\n        return self.x, self.y, a, b\n\na = Actor._remote(\n    args=[0], kwargs={"y": 1}, num_gpus=1, resources={"Custom": 1})\n\nid1, id2, id3, id4 = a.method._remote(\n    args=["test"], kwargs={"b": 2}, num_returns=4)\nassert ray.get([id1, id2, id3, id4]) == [0, 1, "test", 2]\n    ', env=build_env())

@pytest.mark.skipif(sys.platform == 'darwin', reason="Cryptography (TLS dependency) doesn't install in Mac build pipeline")
@pytest.mark.parametrize('use_tls', [True], indirect=True)
def test_client_connect_to_tls_server(use_tls, call_ray_start):
    if False:
        while True:
            i = 10
    tls_env = build_env()
    without_tls_env = {k: v for (k, v) in tls_env.items() if 'TLS' not in k}
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_string_as_driver('\nfrom ray.util.client import ray as ray_client\nray_client.connect("localhost:10001")\n     ', env=without_tls_env)
    assert 'ConnectionError' in exc_info.value.output.decode('utf-8')
    run_string_as_driver('\nimport ray\nfrom ray.util.client import ray as ray_client\nray_client.connect("localhost:10001")\nassert ray.is_initialized()\n     ', env=tls_env)
if __name__ == '__main__':
    import pytest
    import sys
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))